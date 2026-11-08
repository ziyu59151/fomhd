#!/usr/bin/env python3
"""Bounded C1/C2 ion multirate research window with independent RK4 evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np


TOOL_PATH = Path(__file__).resolve()
SCRATCH_ROOT = TOOL_PATH.parents[1]
TASK_ROOT = SCRATCH_ROOT.parent
DECLARED_SOURCE_B = Path("/home/lxzeng/projects/fomhd")
DECLARED_BASELINE_COMMIT = "be5f1c07913409bfd377f22e9ea9d69ac614b158"
DECLARED_BASELINE_BRANCH = "codex/a1-multirate-ion"


def _select_source_project(tool_path: Path, override: str | None):
    if override:
        return Path(override).expanduser().resolve(), "FOMHD_SOURCE_ROOT"
    installed_root = Path(tool_path).resolve().parents[1]
    if (
        (installed_root / "pyproject.toml").is_file()
        and (installed_root / "src/fomhd").is_dir()
    ):
        return installed_root, "installed_source"
    return DECLARED_SOURCE_B, "declared_source_b_fallback"


SOURCE_PROJECT, SOURCE_PROJECT_ORIGIN = _select_source_project(
    TOOL_PATH, os.environ.get("FOMHD_SOURCE_ROOT")
)
SOURCE_ROOT = SOURCE_PROJECT / "src"
TEST_ROOT = SOURCE_PROJECT / "tests/warp"
TOOLS_ROOT = SOURCE_PROJECT / "tools"
RESEARCH_PATH = SCRATCH_ROOT / "src/fomhd/time_multirate.py"
TEST_PATH = SCRATCH_ROOT / "tests/warp/test_time_multirate.py"
PSEUDOCODE_ROOT = SCRATCH_ROOT / "docs/pseudocode"
for location in (
    SCRATCH_ROOT / "src",
    SOURCE_ROOT,
    TEST_ROOT,
    TOOLS_ROOT,
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from fomhd import core, time as time_module
from fomhd.stage import evaluate_stage, initialize
from fomhd.types import Failure, StageTag, cumulative_record


def _load_research():
    spec = importlib.util.spec_from_file_location(
        "fomhd.time_multirate_research", RESEARCH_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load bounded multirate research module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


research = _load_research()


def _synchronize(xp) -> None:
    if xp is not np:
        xp.cuda.get_current_stream().synchronize()


def _host(value, xp):
    return np.asarray(value) if xp is np else xp.asnumpy(value)


def _jsonable(value):
    module = type(value).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy

        if isinstance(value, cupy.ndarray):
            return _jsonable(cupy.asnumpy(value))
        if isinstance(value, cupy.generic):
            return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    if is_dataclass(value):
        return _jsonable(
            {
                item.name: getattr(value, item.name)
                for item in dataclass_fields(value)
            }
        )
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n"
    )


def _pack_state(context, state):
    xp = context.compute.xp
    nc = context.op.Nc
    q = 2 * context.p.K + 1
    p = xp.concatenate(
        (
            state.fluid[0].n.reshape(q, nc),
            state.fluid[0].mom.reshape(q, 3 * nc),
            state.fluid[0].perp.reshape(q, nc),
        ),
        axis=1,
    )
    z = xp.concatenate(
        (
            state.fluid[1].perp.reshape(q, nc),
            state.fluid[1].parallel.reshape(q, nc),
            state.B.reshape(q, context.op.Nb),
        ),
        axis=1,
    )
    return p, z


def _seed_periodic_sentinel(
    context,
    p,
    z,
    case,
    *,
    seed_mode: int,
    sentinel: float,
    base_amplitude: float,
):
    if sentinel <= 0.0:
        return p, z
    xp = context.compute.xp
    nc = context.op.Nc
    nb = context.op.Nb
    p = p.copy()
    z = z.copy()
    phase = (
        2.0
        * math.pi
        * xp.arange(nc, dtype=xp.float64)
        * seed_mode
        / float(nc)
    )
    density = xp.real(p[:, :nc])
    velocity = (
        base_amplitude
        * sentinel
        * case.v_alfven
        * xp.stack(
            (
                xp.cos(phase),
                0.5 * xp.sin(phase),
                xp.zeros_like(phase),
            ),
            axis=0,
        )[None]
    )
    p[:, nc : 4 * nc] += (
        density[:, None, :]
        * velocity
        * float(context.p.species[0].mass)
    ).reshape(p.shape[0], 3 * nc)
    b_blocks = z[:, 2 * nc :]
    per_component = nb // 3
    b_blocks[:, :per_component] += (
        base_amplitude * sentinel * case.B0 * xp.cos(phase)[None, :]
    )
    b_blocks[:, per_component : 2 * per_component] += (
        -base_amplitude
        * sentinel
        * case.B0
        * xp.sin(phase)[None, :]
    )
    return p, z


def _seed_closed_sentinel(
    context,
    p,
    z,
    *,
    seed_mode: int,
    sentinel: float,
    base_amplitude: float,
):
    if sentinel <= 0.0:
        return p, z
    xp = context.compute.xp
    nc = context.op.Nc
    density = core.inverse(
        p[:, :nc].reshape(-1), nc, context, True
    ).reshape(context.N, nc)
    phase = xp.arange(context.N, dtype=xp.float64)[:, None]
    phase = (
        2.0
        * math.pi
        * seed_mode
        * phase
        / float(context.N)
    )
    addition = xp.zeros(
        (context.N, 3, nc), dtype=xp.complex128
    )
    addition[:, 1, :] = (
        base_amplitude
        * sentinel
        * float(context.p.ref.velocity)
        * xp.cos(phase)
    )
    addition *= (
        density[:, None, :]
        * float(context.p.species[0].mass)
    )
    p = p.copy()
    p[:, nc : 4 * nc] += core.forward(
        addition.reshape(-1), 3 * nc, context
    ).reshape(p.shape[0], 3 * nc)
    return p, z


def build_case(config):
    from massless_verification_fixture import HallCase
    from warp_t2_field import (
        closed_inputs,
        periodic_inputs,
    )

    if config["case"] == "periodic":
        context, p, z, _ = periodic_inputs(
            config["points"],
            config["amplitude"],
            config["nonuniform"],
            config["device"],
            spatial_mode=config["seed_mode"],
        )
        # This new experiment assesses target accuracy and resolved stability
        # separately; retain the generic all-frequency estimate as a warning.
        context.p.dt_guard = "warn"
        case = HallCase(points=config["points"])
        p, z = _seed_periodic_sentinel(
            context,
            p,
            z,
            case,
            seed_mode=config["points"] // 2,
            sentinel=config["sentinel"],
            base_amplitude=config["amplitude"],
        )
        return context, p, z, case
    context, p, z, _ = closed_inputs(
        config["points"],
        config["amplitude"],
        config["nonuniform"],
        config["device"],
        spatial_mode=config["seed_mode"],
    )
    context.p.dt_guard = "warn"
    p, z = _seed_closed_sentinel(
        context,
        p,
        z,
        seed_mode=config["seed_mode"],
        sentinel=config["sentinel"],
        base_amplitude=config["amplitude"],
    )
    return context, p, z, None


def _scales(context):
    return research._physical_scales(context)


def _isolated_context(context):
    return research._full_context(context)


def _initial_state(context, p, z):
    return research._state_from_packed(context, p, z)


def _diagnostic_counters():
    return {
        "full_stage_attempts": 0,
        "full_stage_successes": 0,
        "full_stage_failures": 0,
        "by_purpose": {},
        "stage_energy_diagnostic_calls": 0,
        "stage_energy_rows": 0,
        "scope": (
            "harness diagnostics; separate from candidate counters and not a "
            "production ledger"
        ),
    }


def _diagnostic_stage_begin(counters, purpose):
    counters["full_stage_attempts"] += 1
    record = counters["by_purpose"].setdefault(
        purpose, {"attempts": 0, "successes": 0, "failures": 0}
    )
    record["attempts"] += 1


def _diagnostic_stage_end(counters, purpose, *, succeeded):
    counters["full_stage_successes" if succeeded else "full_stage_failures"] += 1
    counters["by_purpose"][purpose][
        "successes" if succeeded else "failures"
    ] += 1


def _stage_for(context, p, z, name, *, diagnostics=None):
    if diagnostics is not None:
        _diagnostic_stage_begin(diagnostics, name)
    try:
        local = _isolated_context(context)
        state = _initial_state(local, p, z)
        tag = StageTag(
            step=0,
            generation=1,
            trial=0,
            spectral_context=int(getattr(local, "spectral_id", 0)),
            time=float(local.p.start_time),
            stage=name,
        )
        result = evaluate_stage(state, tag, local, False)
    except Exception:
        if diagnostics is not None:
            _diagnostic_stage_end(diagnostics, name, succeeded=False)
        raise
    if diagnostics is not None:
        _diagnostic_stage_end(diagnostics, name, succeeded=True)
    return result


def _hard_stage_audit(context, p, z, name, *, diagnostics=None):
    """Audit the original budget policy without changing observation flags."""
    local = _isolated_context(context)
    local.p = copy.copy(context.p)
    local.p.budgets = {
        key: copy.copy(value)
        for key, value in context.p.budgets.items()
    }
    hard_flags = {
        key: bool(getattr(budget, "hard", True))
        for key, budget in local.p.budgets.items()
    }
    checks = []
    failure_code = None
    failure_message = None
    try:
        stage = _stage_for(
            local, p, z, name, diagnostics=diagnostics
        )
        checks = list(stage.checks)
    except Failure as failure:
        checks = list(failure.checks)
        failure_code = failure.code
        failure_message = str(failure)

    def failed(record):
        return not record.available or not record.passed

    hard_failed = [
        {
            "name": record.name,
            "population": record.population,
            "component": record.component,
            "norm": str(record.norm),
            "raw": float(record.raw),
            "threshold": float(record.threshold),
            "available": bool(record.available),
            "passed": bool(record.passed),
        }
        for record in checks
        if record.hard and failed(record)
    ]
    observed_failed = [
        {
            "name": record.name,
            "population": record.population,
            "component": record.component,
            "norm": str(record.norm),
            "raw": float(record.raw),
            "threshold": float(record.threshold),
            "available": bool(record.available),
            "passed": bool(record.passed),
            "reason": str(getattr(record, "reason", "")),
        }
        for record in checks
        if not record.hard and failed(record)
    ]
    return {
        "status": (
            "failed"
            if failure_code is not None
            else ("observations_failed" if observed_failed else "passed")
        ),
        "failure_code": failure_code,
        "failure_message": failure_message,
        "budget_hard_flags": hard_flags,
        "hard_checks": sum(bool(record.hard) for record in checks),
        "observation_checks": sum(not record.hard for record in checks),
        "failed_hard_checks": hard_failed,
        "failed_observation_checks": observed_failed,
    }


def _totals(stage):
    totals = stage.totals
    return {
        "number_ion": float(totals.number[0]),
        "number_electron": float(totals.number[1]),
        "ion_kinetic_J": float(totals.kinetic[0]),
        "ion_internal_J": float(totals.ion_internal),
        "electron_perp_J": float(totals.electron_perp),
        "electron_parallel_J": float(totals.electron_parallel),
        "magnetic_J": float(totals.magnetic),
        "total_J": float(totals.energy),
    }


def _run_reference(
    context,
    p0,
    z0,
    *,
    h,
    M,
    macros,
    coupling,
    rtol,
    max_iterations,
):
    local = _isolated_context(context)
    local.p = copy.copy(context.p)
    local.p.dt = (
        float(h)
        if coupling == "synchronous"
        else float(h) * int(M)
    )
    local.p.nsteps = (
        int(M) * int(macros)
        if coupling == "synchronous"
        else int(macros)
    )
    local.p.coupling = coupling
    local.p.micro_steps = 1 if coupling == "synchronous" else int(M)
    local.p.integrator = "rk4"
    if coupling == "waveform":
        local.p.waveform_rtol = float(rtol)
        local.p.waveform_max_iterations = int(max_iterations)
    state = _initial_state(local, p0, z0)
    cumulative = cumulative_record()
    captured = SimpleNamespace()
    committed = initialize(
        state, local, cumulative, False, captured=captured
    )
    endpoints = []
    frames = []
    for _ in range(local.p.nsteps):
        result = time_module.advance(committed, cumulative, local)
        endpoints.append(_pack_state(local, committed.state))
        frames.append(result.frame)
    p1, z1 = _pack_state(local, committed.state)
    return SimpleNamespace(
        context=local,
        committed=committed,
        cumulative=cumulative,
        p=p1,
        z=z1,
        endpoints=endpoints,
        frames=frames,
        initial_stage=SimpleNamespace(**vars(captured)),
        final_stage=frames[-1],
    )


def _reference_work(oracle):
    return {
        "production_advances": len(oracle.frames),
        "attempted_trials": int(oracle.cumulative.last_trial),
        "accepted_intervals": int(oracle.cumulative.accepted_intervals),
    }


def _run_reference_timed(
    context,
    p0,
    z0,
    *,
    h,
    M,
    macros,
    coupling,
    rtol,
    max_iterations,
    warmups,
    repetitions,
):
    def execute():
        return _run_reference(
            _isolated_context(context),
            p0,
            z0,
            h=h,
            M=M,
            macros=macros,
            coupling=coupling,
            rtol=rtol,
            max_iterations=max_iterations,
        )

    return _timed(
        execute,
        context.compute.xp,
        warmups,
        repetitions,
        observe=_reference_work,
    )


def _timed(
    function,
    xp,
    warmups: int,
    repetitions: int,
    *,
    observe=None,
):
    _synchronize(xp)
    executions = []

    def execute(kind):
        begin = time.perf_counter()
        try:
            value = function()
            _synchronize(xp)
        except Exception as error:
            _synchronize(xp)
            elapsed = time.perf_counter() - begin
            completed = _candidate_work(getattr(error, "completed_windows", ()))
            attempted = dict(getattr(error, "research_counters", {}))
            executions.append(
                {
                    "kind": kind,
                    "status": "failed",
                    "elapsed_s": elapsed,
                    "counters": {
                        key: int(completed.get(key, 0) + attempted.get(key, 0))
                        for key in set(completed).union(attempted)
                    },
                }
            )
            error.timing_executions = tuple(executions)
            error.timing_attempt_kind = kind
            raise
        elapsed = time.perf_counter() - begin
        execution = {
            "kind": kind,
            "status": "completed",
            "elapsed_s": elapsed,
            "counters": {} if observe is None else observe(value),
        }
        executions.append(execution)
        return value

    value = execute("cold")
    for _ in range(warmups):
        value = execute("warmup")
    samples = []
    for _ in range(repetitions):
        value = execute("sample")
        samples.append(executions[-1]["elapsed_s"])
    return value, {
        "cold_s": executions[0]["elapsed_s"],
        "samples_s": samples,
        "median_s": float(np.median(samples)),
        "minimum_s": float(np.min(samples)),
        "warmups": warmups,
        "repetitions": repetitions,
        "cold_executions": 1,
        "campaign_multiplier_per_algorithm": 1 + warmups + repetitions,
        "executions": executions,
        "scope": (
            "algorithm execution/synchronization; fixture and I/O excluded"
        ),
    }


def _simulation_timing(config, *, start_time=0.0):
    """Return the scheduled physical interval from control inputs."""
    microstep = float(config["h"])
    M = int(config["M"])
    macros = int(config["macros"])
    macro_H = microstep * M
    total = macro_H * macros
    start = float(start_time)
    return {
        "start_time_s": start,
        "end_time_s": start + total,
        "microstep_h_s": microstep,
        "macro_H_s": macro_H,
        "steps_per_macro": M,
        "macros": macros,
        "total_duration_s": total,
        "candidate_duration_s": total,
        "reference_duration_s": total,
    }


def _run_candidates(
    context,
    p0,
    z0,
    h,
    config,
):
    macro_H = _simulation_timing(config)["macro_H_s"]

    def execute():
        local = _isolated_context(context)
        p, z = p0.copy(), z0.copy()
        windows = []
        for window_index in range(config["macros"]):
            try:
                result = research.run(
                    config["method"],
                    local,
                    p,
                    z,
                    macro_H,
                    config["M"],
                    rtol=config["rtol"],
                    max_iterations=config["max_iterations"],
                    keep_stages=True,
                    endpoint_check=config["endpoint_check"],
                )
            except Exception as error:
                error.completed_windows = tuple(windows)
                error.failed_window_index = window_index
                error.failed_window_p = p.copy()
                error.failed_window_z = z.copy()
                raise
            windows.append(result)
            p, z = result.p.copy(), result.z.copy()
        return windows

    return _timed(
        execute,
        context.compute.xp,
        config["warmups"],
        config["repetitions"],
        observe=_candidate_work,
    )


def _candidate_work(windows):
    keys = research.Counters.__slots__
    return {
        key: int(
            sum(
                getattr(window, "counters", {}).get(key, 0)
                for window in windows
            )
        )
        for key in keys
    }


def _relative_error(actual, expected, scales):
    """Return raw and componentwise physical-scale normalized errors."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    scales = np.asarray(scales)
    if actual.shape != expected.shape or actual.shape != scales.shape:
        raise ValueError(
            "actual, expected, and scale shape must match componentwise"
        )
    difference = actual - expected
    raw = float(np.max(np.abs(difference)))
    normalizer = np.maximum(
        np.maximum(np.abs(scales), np.abs(expected)),
        1.0e-300,
    )
    normalized = float(
        np.max(np.abs(difference) / normalizer)
    )
    return raw, normalized


def _complex_mode(values, mode, phase):
    real = np.asarray(values, dtype=np.float64)
    return complex(
        (2.0 / real.size)
        * np.sum(real * np.exp(-1j * mode * phase))
    )


def _mode_evidence(context, p, z, mode, case):
    if case is None:
        raise ValueError(
            "phase/mode diagnostics require the one-dimensional periodic fixture"
        )
    nc = context.op.Nc
    xp = context.compute.xp
    density_all = core.inverse(
        p[:, :nc].reshape(-1), nc, context, True
    )
    momentum_all = core.inverse(
        p[:, nc : 4 * nc].reshape(-1),
        3 * nc,
        context,
        True,
    )
    density = xp.real(density_all)
    velocity_r = xp.real(
        momentum_all.reshape(context.N, 3, nc)[:, 0, :]
        / (
            density.reshape(context.N, nc)
            * float(context.p.species[0].mass)
        )
    ).reshape(-1)
    b_physical, _ = research._physical_b_from_retained(
        context, z[:, 2 * nc :]
    )
    magnetic_r = xp.real(
        b_physical.reshape(context.N, 3, nc)[:, 0, :]
    ).reshape(-1)
    velocity_r = _host(velocity_r, xp)
    magnetic_r = _host(magnetic_r, xp)
    phase = (
        2.0
        * math.pi
        * np.arange(velocity_r.size, dtype=np.float64)
        / float(velocity_r.size)
    )
    velocity = _complex_mode(velocity_r, mode, phase)
    magnetic = _complex_mode(magnetic_r, mode, phase)
    scale = (
        float(case.v_alfven)
        if case is not None
        else float(context.p.ref.velocity)
    )
    bscale = (
        float(case.B0)
        if case is not None
        else float(context.p.ref.magnetic)
    )
    return {
        "velocity_mode": {
            "real": velocity.real,
            "imag": velocity.imag,
            "amplitude": abs(velocity),
            "normalized_amplitude": abs(velocity) / max(scale, 1.0e-300),
            "phase_rad": float(np.angle(velocity)),
        },
        "magnetic_mode": {
            "real": magnetic.real,
            "imag": magnetic.imag,
            "amplitude": abs(magnetic),
            "normalized_amplitude": abs(magnetic) / max(bscale, 1.0e-300),
            "phase_rad": float(np.angle(magnetic)),
        },
    }


def _mode_history_summary(records):
    amplitudes = [record["velocity_mode"]["amplitude"] for record in records]
    phases = [record["velocity_mode"]["phase_rad"] for record in records]
    initial = amplitudes[0]
    denominator = max(initial, 1.0e-300)
    gains = [amplitude / denominator for amplitude in amplitudes]
    return {
        "amplitudes": amplitudes,
        "phases_rad": phases,
        "final_amplitude": amplitudes[-1],
        "maximum_amplitude": max(amplitudes),
        "final_gain": gains[-1],
        "maximum_gain": max(gains),
    }


def _mode_comparison(
    context,
    p0,
    z0,
    windows,
    oracle,
    *,
    target_mode,
    sentinel_mode,
    case,
):
    """Compare target and highest-mode sentinel on the periodic line only."""
    initial = {
        "target": _mode_evidence(context, p0, z0, target_mode, case),
        "sentinel": _mode_evidence(context, p0, z0, sentinel_mode, case),
    }

    def evidence(states, mode):
        return [
            _mode_evidence(context, p, z, mode, case)
            for p, z in ((p0, z0), *states)
        ]

    candidate_states = [(window.p, window.z) for window in windows]
    oracle_states = list(oracle.endpoints)
    target_candidate = evidence(candidate_states, target_mode)
    sentinel_candidate = evidence(candidate_states, sentinel_mode)
    target_oracle = evidence(oracle_states, target_mode)
    sentinel_oracle = evidence(oracle_states, sentinel_mode)
    candidate = target_candidate[-1]
    reference = target_oracle[-1]
    candidate_phase = candidate["velocity_mode"]["phase_rad"]
    oracle_phase = reference["velocity_mode"]["phase_rad"]
    independent = int(target_mode) != int(sentinel_mode)
    result = {
        "initial": initial["target"],
        "candidate": candidate,
        "oracle": reference,
        "amplitude_ratio": candidate["velocity_mode"]["amplitude"]
        / max(reference["velocity_mode"]["amplitude"], 1.0e-300),
        "phase_error_rad": float(
            np.angle(np.exp(1j * (candidate_phase - oracle_phase)))
        ),
        "target": {
            "mode": int(target_mode),
            "candidate_history": _mode_history_summary(target_candidate),
            "oracle_history": _mode_history_summary(target_oracle),
        },
        "sentinel": {
            "mode": int(sentinel_mode),
            "independent": independent,
            "candidate_history": _mode_history_summary(sentinel_candidate),
            "oracle_history": _mode_history_summary(sentinel_oracle),
            "reason": (
                ""
                if independent
                else "sentinel mode coincides with the target retained mode"
            ),
        },
    }
    return result


def _qualification(reference, *, endpoint_checks):
    endpoint_passed = all(
        item["passed"] is not False for item in endpoint_checks
    )
    return {
        "execution_completed": True,
        "scientific_qualified": False,
        "macro_order": "second-order design",
        "nonzero_hydro_order_evidence": False,
        "comparison_measurements_available": reference is not None,
        "endpoint_checks_passed": endpoint_passed,
        "reasons": [
            "execution completion is not scientific qualification",
            "second-order macro order remains a design property until "
            "nonzero-hydro refinement evidence is measured",
        ],
    }


def _stage_energy_rows(context, stages, h):
    """Observe instantaneous power and RK4 increments for selected stages."""
    xp = context.compute.xp
    nc = context.op.Nc
    m0 = xp.real(xp.asarray(context.op.M0)).astype(xp.float64)
    m0 = m0.reshape(nc)
    rows = []
    q = 2 * context.p.K + 1
    for stage in stages:
        p_view = stage.p.reshape(q, 5, nc)
        density = core.inverse(
            p_view[:, 0, :].reshape(-1), nc, context, True
        ).reshape(context.N, nc)
        momentum = core.inverse(
            p_view[:, 1:4, :].reshape(q, 3 * nc).reshape(-1),
            3 * nc,
            context,
            True,
        ).reshape(context.N, 3, nc)
        electric = core.inverse(
            stage.electric,
            context.op.Ne,
            context,
            True,
            context.p.ref.electric,
        ).reshape(context.N, context.op.Ne)
        interpolated = core.sample_apply(
            context.op.H,
            electric.reshape(-1),
            context.N,
            context.compute,
        ).reshape(context.N, 3, nc)
        velocity = momentum / (
            float(context.p.species[0].mass) * density[:, None, :]
        )
        rate = (
            float(context.p.species[0].charge)
            * density
            * xp.sum(velocity * interpolated, axis=1)
        )
        p_i = float(
            xp.real(
                xp.sum(
                    rate
                    * m0[None, :]
                    * (2.0 * math.pi / context.N)
                )
            ).item()
        )
        b = stage.z.reshape(
            -1, 2 * nc + context.op.Nb
        )[:, 2 * nc :]
        bdot = stage.z_rhs.reshape(
            -1, 2 * nc + context.op.Nb
        )[:, 2 * nc :]
        mb = xp.real(xp.asarray(context.op.MB)).astype(xp.float64)
        field_rate = float(
            xp.real(
                xp.sum(
                    mb[None, :]
                    * xp.real(xp.conj(b) * bdot)
                    / float(context.p.mu0)
                    * (2.0 * math.pi)
                )
            ).item()
        )
        zero = context.p.K
        electron_rate = float(
            xp.real(
                2.0
                * math.pi
                * xp.sum(
                    m0
                    * (
                        stage.z_rhs[zero, :nc]
                        + 0.5 * stage.z_rhs[zero, nc : 2 * nc]
                    )
                )
            ).item()
        )
        rows.append(
            {
                "outer": stage.outer,
                "micro_step": stage.micro_step,
                "rk_stage": stage.rk_stage,
                "P_i_instantaneous_W": p_i,
                "P_i_RK4_stage_increment_J": p_i
                * h
                * research._RK4_WEIGHTS[stage.rk_stage]
                / 6.0,
                "field_energy_rate_instantaneous_W": field_rate,
                "electron_energy_rate_instantaneous_W": electron_rate,
            }
        )
    return rows


def _final_outer_stages(windows):
    """Select only each completed window's converged final outer sweep."""
    selected = []
    for window in windows:
        if not window.iterations or not window.iterations[-1].converged:
            raise ValueError("stage diagnostics require a converged outer sweep")
        final_outer = window.iterations[-1].outer
        selected.extend(
            stage for stage in window.stages if stage.outer == final_outer
        )
    return selected


def _candidate_arrays(
    windows,
    context,
    *,
    start_time=0.0,
    macro_H,
    microstep,
    window_offset=0,
):
    rows = [
        (window_index, stage)
        for window_index, window in enumerate(
            windows, start=int(window_offset)
        )
        for stage in window.stages
    ]
    stages = [stage for _, stage in rows]
    if not stages:
        return {}
    offsets = (0.0, 0.5, 0.5, 1.0)
    times = [
        float(start_time)
        + window_index * float(macro_H)
        + (stage.micro_step + offsets[stage.rk_stage]) * float(microstep)
        for window_index, stage in rows
    ]
    available = np.array(
        [
            bool(
                getattr(
                    stage,
                    "p_rhs_available",
                    stage.p_rhs is not None,
                )
            )
            for stage in stages
        ],
        dtype=np.bool_,
    )
    arrays = {
        "candidate_stage_p": np.stack(
            [_host(stage.p, context.compute.xp) for stage in stages]
        ),
        "candidate_stage_z": np.stack(
            [_host(stage.z, context.compute.xp) for stage in stages]
        ),
        "candidate_stage_p_rhs_available": available,
        "candidate_stage_z_rhs": np.stack(
            [_host(stage.z_rhs, context.compute.xp) for stage in stages]
        ),
        "candidate_stage_electric": np.stack(
            [_host(stage.electric, context.compute.xp) for stage in stages]
        ),
        "candidate_stage_magnetic": np.stack(
            [_host(stage.magnetic, context.compute.xp) for stage in stages]
        ),
        "candidate_stage_outer": np.array(
            [stage.outer for stage in stages], dtype=np.int32
        ),
        "candidate_stage_window": np.array(
            [window_index for window_index, _ in rows], dtype=np.int32
        ),
        "candidate_stage_time_s": np.array(times, dtype=np.float64),
        "candidate_stage_micro": np.array(
            [stage.micro_step for stage in stages], dtype=np.int32
        ),
        "candidate_stage_rk": np.array(
            [stage.rk_stage for stage in stages], dtype=np.int32
        ),
    }
    iterations = [
        (window_index, record)
        for window_index, window in enumerate(
            windows, start=int(window_offset)
        )
        for record in getattr(window, "iterations", ())
    ]
    if iterations:
        arrays.update(
            candidate_iteration_outer=np.array(
                [record.outer for _, record in iterations],
                dtype=np.int32,
            ),
            candidate_iteration_window=np.array(
                [window_index for window_index, _ in iterations],
                dtype=np.int32,
            ),
            candidate_iteration_time_s=np.array(
                [
                    float(start_time)
                    + (window_index + 1) * float(macro_H)
                    for window_index, _ in iterations
                ],
                dtype=np.float64,
            ),
            candidate_iteration_residual_raw=np.array(
                [record.residual_raw for _, record in iterations],
                dtype=np.float64,
            ),
            candidate_iteration_residual_normalized=np.array(
                [record.residual_normalized for _, record in iterations],
                dtype=np.float64,
            ),
            candidate_iteration_actual_increment_max=np.array(
                [record.actual_increment_max for _, record in iterations],
                dtype=np.float64,
            ),
            candidate_iteration_converged=np.array(
                [record.converged for _, record in iterations],
                dtype=np.bool_,
            ),
        )
    if bool(np.all(available)):
        arrays["candidate_stage_p_rhs"] = np.stack(
            [
                _host(stage.p_rhs, context.compute.xp)
                for stage in stages
            ]
        )
    return arrays


def _iteration_payloads(windows, *, start_time, macro_H):
    return [
        {
            "window": window_index,
            "time_s": float(start_time)
            + (window_index + 1) * float(macro_H),
            "outer": record.outer,
            "residual_raw": record.residual_raw,
            "residual_normalized": record.residual_normalized,
            "scale": record.scale,
            "actual_increment_max": record.actual_increment_max,
            "converged": record.converged,
            "stage_count": record.stage_count,
        }
        for window_index, window in enumerate(windows)
        for record in getattr(window, "iterations", ())
    ]


def _failure_iteration_payloads(history, window_index, time_s=None):
    return [
        {
            "window": int(window_index),
            "time_s": None if time_s is None else float(time_s),
            "outer": int(record.outer),
            "residual_raw": float(record.residual_raw),
            "residual_normalized": float(record.residual_normalized),
            "scale": float(record.scale),
            "actual_increment_max": float(record.actual_increment_max),
            "converged": bool(record.converged),
            "stage_count": int(record.stage_count),
        }
        for record in history
    ]


def _failure_history_arrays(
    error,
    context,
    *,
    start_time,
    window_index,
    macro_H,
    microstep,
):
    arrays = {}
    history = list(getattr(error, "research_history", ()))
    if history:
        for name in ("guess_p", "endpoint_p", "target_p", "residual"):
            arrays[f"failure_iteration_{name}"] = np.stack(
                [
                    _host(getattr(record, name), context.compute.xp)
                    for record in history
                ]
            )
        arrays["failure_iteration_actual_increment_max"] = np.array(
            [record.actual_increment_max for record in history],
            dtype=np.float64,
        )
    stages = list(getattr(error, "research_stages", ()))
    if stages:
        fake_window = SimpleNamespace(stages=tuple(stages), iterations=())
        stage_arrays = _candidate_arrays(
            [fake_window],
            context,
            start_time=start_time,
            macro_H=macro_H,
            microstep=microstep,
            window_offset=window_index,
        )
        for key, value in stage_arrays.items():
            arrays[key.replace("candidate_", "failure_", 1)] = value
    return arrays


def _aggregate_execution_counters(executions):
    totals = {}
    for execution in executions:
        for key, value in execution.get("counters", {}).items():
            totals[str(key)] = totals.get(str(key), 0) + int(value)
    return totals


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(root: Path, *args: str):
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return None, str(error)
    if result.returncode:
        return None, result.stderr.strip() or f"git exited {result.returncode}"
    return result.stdout.strip(), None


def _git_provenance(root: Path):
    top_level, top_error = _git_value(root, "rev-parse", "--show-toplevel")
    head, head_error = _git_value(root, "rev-parse", "HEAD")
    branch, branch_error = _git_value(root, "branch", "--show-current")
    status, status_error = _git_value(
        root, "status", "--porcelain=v1", "--untracked-files=normal"
    )
    errors = [
        error
        for error in (top_error, head_error, branch_error, status_error)
        if error
    ]
    return {
        "top_level": top_level,
        "head": head,
        "branch": branch,
        "dirty": None if status is None else bool(status),
        "errors": errors,
    }


def _required_provenance_paths():
    return [
        TOOL_PATH,
        RESEARCH_PATH,
        TEST_PATH,
        SOURCE_PROJECT / "tools/warp_t2_field.py",
        SOURCE_PROJECT / "tests/warp/massless_verification_fixture.py",
        SOURCE_PROJECT / "tests/warp/test_massless_support.py",
        PSEUDOCODE_ROOT / "src/fomhd/time_multirate.md",
        PSEUDOCODE_ROOT / "tests/warp/test_time_multirate.py.md",
        PSEUDOCODE_ROOT / "tools/warp_multirate_ion.py.md",
        SOURCE_PROJECT / "docs/pseudocode/tools/warp_t2_field.py.md",
        SOURCE_PROJECT
        / "docs/pseudocode/tests/warp/massless_verification_fixture.py.md",
        SOURCE_PROJECT / "docs/pseudocode/tests/warp/test_massless_support.py.md",
        *sorted((SOURCE_ROOT / "fomhd").glob("*.py")),
    ]


def _optional_provenance_paths():
    return [
        *sorted(TASK_ROOT.glob("REQUEST*.md")),
        TASK_ROOT / "LOGS/worker.log",
        *sorted(SCRATCH_ROOT.glob("PATH_MAPPING_REPAIR*.md")),
        *sorted(SCRATCH_ROOT.glob("MANIFEST_REPAIR*.json")),
        *sorted((SCRATCH_ROOT / "evidence").glob("pre-repair*.sha256")),
    ]


def _provenance(config):
    required = list(
        dict.fromkeys(path.resolve() for path in _required_provenance_paths())
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "required provenance files missing: " + ", ".join(map(str, missing))
        )
    files = {str(path): _sha256(path) for path in required}
    for path in dict.fromkeys(_optional_provenance_paths()):
        if path.is_file():
            files[str(path.resolve())] = _sha256(path)
    source_git = _git_provenance(SOURCE_PROJECT)
    return {
        "source_commit": source_git["head"],
        "source_branch": source_git["branch"],
        "source_dirty": source_git["dirty"],
        "source_git": source_git,
        "declared_baseline": {
            "commit": DECLARED_BASELINE_COMMIT,
            "branch": DECLARED_BASELINE_BRANCH,
        },
        "task_root": str(TASK_ROOT),
        "task_root_scope": "optional worker evidence; not repository authority",
        "scratch_root": str(SCRATCH_ROOT),
        "source_root": str(SOURCE_PROJECT),
        "source_project_origin": SOURCE_PROJECT_ORIGIN,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": sys.version,
        "config": config,
        "files": files,
    }


def run(config, output: Path):
    if output.exists():
        raise ValueError("output must be a new directory")
    if config["device"] != "cpu" and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("CUDA runs require an owned Slurm allocation")
    output.mkdir(parents=True)
    _write_json(output / "input.json", config)
    provenance = _provenance(config)
    _write_json(output / "provenance.json", provenance)
    started = time.perf_counter()
    diagnostics = _diagnostic_counters()
    metadata = {
        "status": "failed",
        "scope": "bounded research window",
        "accepted_product": False,
        "full_coupled_acceptance": "not_claimed",
        "interval_work_budget": "not_implemented",
        "production_transaction": "not_used",
        "candidate": config["method"],
        "diagnostic_work": diagnostics,
    }
    arrays = {}
    windows = []
    xp = None
    try:
        context, p0, z0, case = build_case(config)
        xp = context.compute.xp
        if xp is not np:
            xp.get_default_memory_pool().set_limit(size=8 << 30)
        time_contract = _simulation_timing(
            config, start_time=float(context.p.start_time)
        )
        metadata.update(
            case=config["case"],
            points=config["points"],
            mode=config["seed_mode"],
            h=time_contract["microstep_h_s"],
            M=int(config["M"]),
            H=time_contract["macro_H_s"],
            macros=int(config["macros"]),
            K=int(context.p.K),
            Nphi=int(context.N),
            active_cells=int(context.op.Nc),
            magnetic_dofs=int(context.op.Nb),
            device=config["device"],
            time_contract=time_contract,
        )
        initial_stage = _stage_for(
            context,
            p0,
            z0,
            "multirate_initial",
            diagnostics=diagnostics,
        )
        metadata["initial_energy"] = _totals(initial_stage)

        windows, timing = _run_candidates(
            context,
            p0,
            z0,
            float(config["h"]),
            config,
        )
        candidate = windows[-1]
        candidate_p, candidate_z = candidate.p, candidate.z
        metadata["candidate_execution_completed"] = True
        # Keep the completed trial when a later reference or diagnostic fails.
        arrays.update(
            candidate_p=_host(candidate_p, xp),
            candidate_z=_host(candidate_z, xp),
            candidate_midpoint_p=_host(candidate.midpoint_p, xp),
            candidate_integrated_e=_host(candidate.integrated_electric, xp),
            candidate_integrated_b=_host(candidate.integrated_magnetic, xp),
            candidate_residual=_host(candidate.residual, xp),
            initial_p=_host(p0, xp),
            initial_z=_host(z0, xp),
        )
        metadata["timing"] = timing
        metadata["counters"] = _candidate_work(windows)
        metadata["window_counters"] = [
            dict(window.counters) for window in windows
        ]
        metadata["iteration_records"] = _iteration_payloads(
            windows,
            start_time=time_contract["start_time_s"],
            macro_H=time_contract["macro_H_s"],
        )
        metadata["returned_execution_counters"] = metadata["counters"]
        metadata["candidate_campaign_counters"] = (
            _aggregate_execution_counters(timing["executions"])
        )
        metadata["field_batch_work"] = {
            str(key): int(
                sum(
                    window.metadata.get("field_batch_work", {}).get(key, 0)
                    for window in windows
                )
            )
            for key in set().union(
                *(
                    window.metadata.get("field_batch_work", {})
                    for window in windows
                )
            )
        }
        metadata["field_batch_work_scope"] = (
            "FieldBatch normal/SVD/solve/rhs work from the returned candidate "
            "execution only; wall_normal_initial_solves counts only the "
            "adapter's one ion projection normal solve per sample when T has rows"
        )
        metadata["stage_history_bytes"] = int(
            sum(window.stage_history_bytes for window in windows)
        )
        metadata["endpoint_checks"] = [
            {
                "requested": bool(config["endpoint_check"]),
                "max_abs": (
                    None
                    if window.endpoint_check is None
                    else window.endpoint_check.max_abs
                ),
                "relative": (
                    None
                    if window.endpoint_check is None
                    else window.endpoint_check.relative
                ),
                "passed": (
                    None
                    if window.endpoint_check is None
                    else window.endpoint_check.passed
                ),
            }
            for window in windows
        ]

        final_stage = _stage_for(
            context,
            candidate_p,
            candidate_z,
            "multirate_final",
            diagnostics=diagnostics,
        )
        metadata["final_energy"] = _totals(final_stage)
        if config["case"] == "closed":
            metadata["original_budget_audit"] = {
                "initial": _hard_stage_audit(
                    context,
                    p0,
                    z0,
                    "multirate_initial_hard",
                    diagnostics=diagnostics,
                ),
                "final": _hard_stage_audit(
                    context,
                    candidate_p,
                    candidate_z,
                    "multirate_final_hard",
                    diagnostics=diagnostics,
                ),
                "scope": (
                    "exact original hard/observation flags; failures remain "
                    "diagnostic evidence and are not production acceptance"
                ),
            }

        oracle = None
        reference_timing = None
        if config["reference"] == "synchronous":
            oracle, reference_timing = _run_reference_timed(
                context,
                p0,
                z0,
                h=float(config["h"]),
                M=int(config["M"]),
                macros=int(config["macros"]),
                coupling="synchronous",
                rtol=config["rtol"],
                max_iterations=config["max_iterations"],
                warmups=config["warmups"],
                repetitions=config["repetitions"],
            )
            scales = _scales(context)
            p_raw, p_relative = _relative_error(
                _host(candidate_p, xp),
                _host(oracle.p, xp),
                _host(scales.p, xp),
            )
            z_raw, z_relative = _relative_error(
                _host(candidate_z, xp),
                _host(oracle.z, xp),
                _host(scales.z, xp),
            )
            metadata["reference"] = {
                "kind": "same-h synchronous complete production RK4",
                "steps": int(config["M"]) * int(config["macros"]),
                "elapsed_s": reference_timing["median_s"],
                "timing": reference_timing,
                "returned_execution_counters": _reference_work(oracle),
                "p_max_abs": p_raw,
                "p_normalized": p_relative,
                "z_max_abs": z_raw,
                "z_normalized": z_relative,
                "normalization": {
                    "P": "componentwise SI scales: density, momentum, pressure",
                    "Z": "componentwise SI scales: electron pressure, magnetic field",
                    "aggregate": "maximum componentwise normalized absolute error",
                },
            }
            metadata["oracle_final_energy"] = _totals(oracle.final_stage)
            arrays.update(
                oracle_p=np.asarray(_host(oracle.p, xp)),
                oracle_z=np.asarray(_host(oracle.z, xp)),
                oracle_initial_p=np.asarray(_host(p0, xp)),
                oracle_initial_z=np.asarray(_host(z0, xp)),
            )
        elif config["reference"] == "waveform":
            oracle, reference_timing = _run_reference_timed(
                context,
                p0,
                z0,
                h=float(config["h"]),
                M=int(config["M"]),
                macros=int(config["macros"]),
                coupling="waveform",
                rtol=config["rtol"],
                max_iterations=config["max_iterations"],
                warmups=config["warmups"],
                repetitions=config["repetitions"],
            )
            scales = _scales(context)
            p_raw, p_relative = _relative_error(
                _host(candidate_p, xp),
                _host(oracle.p, xp),
                _host(scales.p, xp),
            )
            z_raw, z_relative = _relative_error(
                _host(candidate_z, xp),
                _host(oracle.z, xp),
                _host(scales.z, xp),
            )
            metadata["reference"] = {
                "kind": "same-h complete production waveform",
                "macros": int(config["macros"]),
                "elapsed_s": reference_timing["median_s"],
                "timing": reference_timing,
                "returned_execution_counters": _reference_work(oracle),
                "p_max_abs": p_raw,
                "p_normalized": p_relative,
                "z_max_abs": z_raw,
                "z_normalized": z_relative,
                "normalization": {
                    "P": "componentwise SI scales: density, momentum, pressure",
                    "Z": "componentwise SI scales: electron pressure, magnetic field",
                    "aggregate": "maximum componentwise normalized absolute error",
                },
            }
            metadata["oracle_final_energy"] = _totals(oracle.final_stage)
            arrays.update(
                oracle_p=_host(oracle.p, xp),
                oracle_z=_host(oracle.z, xp),
                oracle_nodes_p=np.stack(
                    [_host(node[0], xp) for node in oracle.endpoints]
                ),
                oracle_nodes_z=np.stack(
                    [_host(node[1], xp) for node in oracle.endpoints]
                ),
            )
        else:
            metadata["reference"] = {"kind": "none"}

        if oracle is not None and config["case"] == "periodic":
            metadata["mode_comparison"] = _mode_comparison(
                context,
                p0,
                z0,
                windows,
                oracle,
                target_mode=config["seed_mode"],
                sentinel_mode=config["points"] // 2,
                case=case,
            )
            arrays.update(
                candidate_target_mode_amplitude=np.asarray(
                    metadata["mode_comparison"]["target"][
                        "candidate_history"
                    ]["amplitudes"]
                ),
                oracle_target_mode_amplitude=np.asarray(
                    metadata["mode_comparison"]["target"][
                        "oracle_history"
                    ]["amplitudes"]
                ),
                candidate_sentinel_mode_amplitude=np.asarray(
                    metadata["mode_comparison"]["sentinel"][
                        "candidate_history"
                    ]["amplitudes"]
                ),
                oracle_sentinel_mode_amplitude=np.asarray(
                    metadata["mode_comparison"]["sentinel"][
                        "oracle_history"
                    ]["amplitudes"]
                ),
            )
        elif config["case"] == "closed":
            metadata["mode_comparison"] = None
            metadata["mode_diagnostics"] = {
                "status": "not_applicable",
                "reason": (
                    "closed RZ flattening is not a toroidal mode projection; "
                    "component errors, constraints, and raw energy are retained"
                ),
            }
        else:
            metadata["mode_comparison"] = None
            metadata["mode_diagnostics"] = {
                "status": "unavailable",
                "reason": "no independent reference execution was requested",
            }

        if reference_timing is not None:
            same_protocol = (
                timing["warmups"] == reference_timing["warmups"]
                and timing["repetitions"] == reference_timing["repetitions"]
                and timing["cold_executions"]
                == reference_timing["cold_executions"]
            )
            metadata["campaign_work"] = {
                "protocol": {
                    "cold_executions_per_algorithm": 1,
                    "warmups": int(config["warmups"]),
                    "repetitions": int(config["repetitions"]),
                    "campaign_multiplier_per_algorithm": (
                        1
                        + int(config["warmups"])
                        + int(config["repetitions"])
                    ),
                },
                "candidate_executions": len(timing["executions"]),
                "reference_executions": len(
                    reference_timing["executions"]
                ),
                "total_algorithm_executions": (
                    len(timing["executions"])
                    + len(reference_timing["executions"])
                ),
                "candidate_counters_all_executions": metadata[
                    "candidate_campaign_counters"
                ],
                "reference_counters_all_executions": (
                    _aggregate_execution_counters(
                        reference_timing["executions"]
                    )
                ),
                "all_rejected_outer_work_charged": True,
                "fresh_isolated_context_per_execution": True,
                "scope": (
                    "all cold, warmup, and sampled candidate/reference "
                    "executions; returned-execution counters are separate"
                ),
            }
            metadata["timing_comparison"] = {
                "same_protocol": same_protocol,
                "candidate_to_reference_median_ratio": (
                    timing["median_s"] / reference_timing["median_s"]
                ),
                "speedup_claim": False,
                "reason": (
                    "matched wall-clock timing is descriptive; no same-accuracy "
                    "speedup was established"
                ),
            }

        stages = _final_outer_stages(windows)
        diagnostics["stage_energy_diagnostic_calls"] += 1
        diagnostics["stage_energy_rows"] += len(stages)
        metadata["stage_diagnostics"] = _stage_energy_rows(
            context, stages, float(config["h"])
        )
        metadata["stage_diagnostics_scope"] = {
            "selection": "each window's converged final outer RK4 waveform",
            "final_outer_indices": [
                window.iterations[-1].outer for window in windows
            ],
            "trial_stage_rows_excluded": (
                sum(len(window.stages) for window in windows) - len(stages)
            ),
            "interpretation": (
                "instantaneous diagnostic samples and per-stage RK4 "
                "increments; trial iterations are not a physical power ledger"
            ),
        }
        arrays.update(
            _candidate_arrays(
                windows,
                context,
                start_time=time_contract["start_time_s"],
                macro_H=time_contract["macro_H_s"],
                microstep=time_contract["microstep_h_s"],
            )
        )
        scales = _scales(context)
        arrays.update(
            candidate_p=_host(candidate_p, xp),
            candidate_z=_host(candidate_z, xp),
            candidate_midpoint_p=_host(candidate.midpoint_p, xp),
            candidate_integrated_e=_host(
                candidate.integrated_electric, xp
            ),
            candidate_integrated_b=_host(
                candidate.integrated_magnetic, xp
            ),
            candidate_residual=_host(candidate.residual, xp),
            initial_p=_host(p0, xp),
            initial_z=_host(z0, xp),
            p_scales=_host(scales.p, xp),
            z_scales=_host(scales.z, xp),
        )
        if oracle is not None and "oracle_nodes_p" not in arrays:
            arrays["oracle_nodes_p"] = np.stack(
                [_host(node[0], xp) for node in oracle.endpoints]
            )
            arrays["oracle_nodes_z"] = np.stack(
                [_host(node[1], xp) for node in oracle.endpoints]
            )

        metadata["limitations"] = [
            "research trial only; accepted state/ledger/cache were not changed",
            "no interval-work budget or same-accuracy production acceptance",
            "C1 is second-order macro coupling plus fourth-order micro RK4",
            "C2 uses frozen midpoint hydro and is not general T1 synchronous",
            "grid response is not an FK particle-phase claim",
            "raw energy diagnostics are not a production ledger",
            "CPU timing is not a CPU/GPU speedup qualification",
        ]
        metadata["qualification"] = _qualification(
            metadata.get("reference") if oracle is not None else None,
            endpoint_checks=metadata["endpoint_checks"],
        )
        metadata["execution_completed"] = True
        metadata["scientific_qualified"] = False
        metadata["status"] = "completed"
        metadata["output_array_bytes"] = int(
            sum(np.asarray(value).nbytes for value in arrays.values())
        )
        np.savez_compressed(output / "arrays.npz", **arrays)
    except Exception as error:
        completed_windows = list(
            getattr(error, "completed_windows", windows)
        )
        failed_window_index = getattr(
            error, "failed_window_index", None
        )
        failed_window = (
            None
            if failed_window_index is None
            else int(failed_window_index)
        )
        history = list(getattr(error, "research_history", ()))
        contract = metadata.get(
            "time_contract",
            _simulation_timing(config),
        )
        failed_iteration_time = (
            None
            if failed_window is None
            else contract["start_time_s"]
            + (failed_window + 1) * contract["macro_H_s"]
        )
        iterations = []
        if completed_windows:
            iterations.extend(
                _iteration_payloads(
                    completed_windows,
                    start_time=contract["start_time_s"],
                    macro_H=contract["macro_H_s"],
                )
            )
        if history:
            iterations.extend(
                _failure_iteration_payloads(
                    history,
                    failed_window if failed_window is not None else 0,
                    time_s=failed_iteration_time,
                )
            )
        current_failure_counters = getattr(
            error, "research_counters", None
        )
        if current_failure_counters is None:
            reported_counters = metadata.get("counters", {})
            completed_counters = {}
        else:
            completed_counters = _candidate_work(completed_windows)
            reported_counters = {
                key: int(
                    completed_counters.get(key, 0)
                    + current_failure_counters.get(key, 0)
                )
                for key in set(completed_counters).union(
                    current_failure_counters
                )
            }
        completed_field_work = {}
        for window in completed_windows:
            for key, value in window.metadata.get(
                "field_batch_work", {}
            ).items():
                completed_field_work[str(key)] = (
                    completed_field_work.get(str(key), 0) + int(value)
                )
        current_field_work = getattr(
            error, "research_field_batch_work", {}
        )
        reported_field_work = dict(completed_field_work)
        for key, value in current_field_work.items():
            reported_field_work[str(key)] = (
                reported_field_work.get(str(key), 0) + int(value)
            )
        if not reported_field_work:
            reported_field_work = metadata.get("field_batch_work", {})
        metadata.update(
            status="failed",
            execution_completed=False,
            scientific_qualified=False,
            failure_code=getattr(error, "code", type(error).__name__),
            failure_message=str(error),
            counters=reported_counters,
            completed_window_counters=completed_counters,
            iterations=iterations,
            completed_window_count=len(completed_windows),
            failed_window_index=failed_window,
            completed_current_window_iterations=len(history),
            completed_current_window_stages=len(
                getattr(error, "research_stages", ())
            ),
            field_batch_work=reported_field_work,
            timing_failure_executions=getattr(
                error, "timing_executions", ()
            ),
        )
        (output / "traceback.txt").write_text(traceback.format_exc())
        if xp is not None:
            if completed_windows:
                arrays.update(
                    _candidate_arrays(
                        completed_windows,
                        context,
                        start_time=contract["start_time_s"],
                        macro_H=contract["macro_H_s"],
                        microstep=contract["microstep_h_s"],
                    )
                )
            if getattr(error, "failed_window_p", None) is not None:
                arrays["failed_window_input_p"] = _host(
                    error.failed_window_p, xp
                )
            if getattr(error, "failed_window_z", None) is not None:
                arrays["failed_window_input_z"] = _host(
                    error.failed_window_z, xp
                )
            if getattr(error, "last_attempt_p", None) is not None:
                arrays["last_attempt_p"] = _host(error.last_attempt_p, xp)
            if getattr(error, "last_attempt_z", None) is not None:
                arrays["last_attempt_z"] = _host(error.last_attempt_z, xp)
            arrays.update(
                _failure_history_arrays(
                    error,
                    context,
                    start_time=contract["start_time_s"],
                    window_index=(
                        failed_window
                        if failed_window is not None
                        else 0
                    ),
                    macro_H=contract["macro_H_s"],
                    microstep=contract["microstep_h_s"],
                )
            )
        if arrays:
            np.savez_compressed(output / "failure-arrays.npz", **arrays)
    metadata["host_peak_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    metadata["total_elapsed_s"] = time.perf_counter() - started
    _write_json(output / "report.json", metadata)
    print(
        json.dumps(
            _jsonable({
                "status": metadata["status"],
                "candidate": config["method"],
                "reference": metadata.get("reference"),
                "counters": metadata.get("counters"),
                "failure_code": metadata.get("failure_code"),
            })
        ),
        flush=True,
    )
    return metadata


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("periodic", "closed"), default="periodic"
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda:0"), default="cpu"
    )
    parser.add_argument("--method", choices=("C1", "C2"), required=True)
    parser.add_argument(
        "--reference",
        choices=("synchronous", "waveform", "none"),
        default="synchronous",
    )
    parser.add_argument("--points", type=int, default=33)
    parser.add_argument("--amplitude", type=float, default=1.0e-4)
    parser.add_argument("--nonuniform", type=float, default=0.1)
    parser.add_argument("--h", type=float, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--macros", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--seed-mode", type=int, default=1)
    parser.add_argument("--sentinel", type=float, default=1.0e-4)
    parser.add_argument("--endpoint-check", action="store_true")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    parser = _parser()
    options = vars(parser.parse_args(argv))
    output = options.pop("output")
    options["method"] = options["method"].upper()
    if options["case"] == "closed":
        if options["points"] not in (8, 12):
            parser.error("closed case points must be 8 or 12")
        options["amplitude"] = 0.02
        options["nonuniform"] = 0.1
    elif (
        options["points"] < 9
        or options["points"] > 65
        or options["points"] % 2 == 0
    ):
        parser.error("periodic points must be odd and in [9,65]")
    if (
        not math.isfinite(options["h"])
        or options["h"] <= 0.0
        or options["M"] < 2
        or options["M"] > 256
        or (options["method"] == "C2" and options["M"] % 2)
        or not 1 <= options["macros"] <= 32
        or not 0.0 < options["rtol"] <= 1.0e-6
        or not 1 <= options["max_iterations"] <= 12
        or not 1 <= options["seed_mode"] <= options["points"] // 2
        or not 0.0 <= options["sentinel"] <= 0.1
        or not 0 <= options["warmups"] <= 2
        or not 1 <= options["repetitions"] <= 3
    ):
        parser.error("controls exceed the selected bounded experiment")
    result = run(options, output)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
