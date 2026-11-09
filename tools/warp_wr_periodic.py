#!/usr/bin/env python3
"""Segmented complete periodic A1-WR campaign runner."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fomhd import backend as backend_module
from fomhd import core, fluid, io, massless, massless_field, multirate_wr_checks
from fomhd import multirate_wr_physics, stage, time as time_module
from fomhd import operators, time_multirate, time_predictor, time_waveform, wall
from fomhd.types import Failure, cumulative_record


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import warp_multirate_wr as benchmark_helper  # noqa: E402

import wr_benchmark_fixture


HOST_LIMIT = 16 << 30
DEVICE_LIMIT = 8 << 30
OUTPUT_LIMIT = 6 << 30
RUN_SCHEMA = "fomhd-periodic-wr-campaign-v1"
COUPLINGS = ("synchronous", "predictor", "waveform", "multirate_wr")


def _complex_json(value):
    return {
        "real": _finite_json(float(value.real)),
        "imag": _finite_json(float(value.imag)),
    }


def _finite_json(value):
    if math.isfinite(value):
        return value
    if math.isnan(value):
        label = "NaN"
    elif value > 0.0:
        label = "+Inf"
    else:
        label = "-Inf"
    return {"value": None, "nonfinite": label}


def _jsonable(value):
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, complex):
        return _complex_json(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, float):
        return _finite_json(value)
    return io._json_value(value)


def _canonical(value):
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(value), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_npz(path, arrays):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.npz", dir=path.parent
    )
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _host_array(value, context):
    xp = context.compute.xp
    return np.asarray(value) if xp is np else xp.asnumpy(value)


def _normalize(config):
    value = dict(config)
    advance_fn = value.pop("advance_fn", time_module.advance)
    defaults = {
        "device": "cpu",
        "coupling": "multirate_wr",
        "M": 1,
        "s": 3,
        "h": 1.0e-7,
        "total_microsteps": 4096,
        "segment_macros": 1,
        "wall_seconds": 1600.0,
        "initial": "tangent",
        "points": 33,
        "nonuniform": 0.1,
        "amplitude": 0.02,
        "sentinel": 1.0e-4,
        "seed_mode": 1,
        "waveform_rtol": 1.0e-10,
        "waveform_max_iterations": 12,
        "wr_relaxation": 1.0,
        "output": None,
        "resume": None,
    }
    defaults.update(value)
    result = defaults
    if result["coupling"] not in COUPLINGS:
        raise ValueError(f"unsupported coupling: {result['coupling']}")
    for name in ("M", "s", "total_microsteps", "segment_macros", "points"):
        if type(result[name]) is not int or result[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if result["coupling"] == "multirate_wr" and result["s"] not in (3, 5):
        raise ValueError("multirate_wr requires s=3 or s=5")
    if result["points"] < 9 or result["points"] % 2 == 0:
        raise ValueError("points must be an odd integer >= 9")
    if result["total_microsteps"] % result["M"]:
        raise ValueError("total_microsteps must be divisible by M")
    for name in (
        "h",
        "wall_seconds",
        "nonuniform",
        "amplitude",
        "sentinel",
    ):
        if (
            type(result[name]) not in (int, float)
            or isinstance(result[name], bool)
            or not math.isfinite(float(result[name]))
            or float(result[name]) < 0.0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
    if float(result["h"]) <= 0.0:
        raise ValueError("h must be positive")
    if float(result["wall_seconds"]) > 1600.0:
        raise ValueError("wall_seconds must not exceed 1600")
    if result["initial"] not in ("tangent", "constant"):
        raise ValueError("initial must be tangent or constant")
    if result["device"] != "cpu":
        if not str(result["device"]).startswith("cuda:"):
            raise ValueError("device must be cpu or cuda:N")
        if not os.environ.get("SLURM_JOB_ID"):
            raise ValueError("CUDA campaign requires an owned SLURM allocation")
    if result["output"] is None:
        raise ValueError("output is required")
    result["output"] = Path(result["output"])
    result["resume"] = (
        None if result["resume"] is None else Path(result["resume"])
    )
    return result, advance_fn


def _scheme(config):
    return {
        "schema": RUN_SCHEMA,
        "device": config["device"],
        "coupling": config["coupling"],
        "M": int(config["M"]),
        "s": int(config["s"]),
        "h": float(config["h"]),
        "total_microsteps": int(config["total_microsteps"]),
        "initial": config["initial"],
        "waveform_rtol": float(config["waveform_rtol"]),
        "waveform_max_iterations": int(config["waveform_max_iterations"]),
        "wr_relaxation": float(config["wr_relaxation"]),
        "points": int(config["points"]),
        "nonuniform": float(config["nonuniform"]),
        "amplitude": float(config["amplitude"]),
        "sentinel": float(config["sentinel"]),
        "seed_mode": int(config["seed_mode"]),
    }


def _source_manifest():
    modules = {
        "fomhd.core": core,
        "fomhd.backend": backend_module,
        "fomhd.fluid": fluid,
        "fomhd.io": io,
        "fomhd.massless": massless,
        "fomhd.massless_field": massless_field,
        "fomhd.multirate_wr_checks": multirate_wr_checks,
        "fomhd.multirate_wr_physics": multirate_wr_physics,
        "fomhd.operators": operators,
        "fomhd.stage": stage,
        "fomhd.time": time_module,
        "fomhd.time_multirate": time_multirate,
        "fixture.wr_benchmark_fixture": wr_benchmark_fixture,
        "tools.warp_multirate_wr": benchmark_helper,
        "tools.warp_wr_periodic": Path(__file__),
    }
    result = {}
    for name, value in modules.items():
        if isinstance(value, Path):
            path = value
        else:
            path = inspect.getsourcefile(value) or value.__file__
        result[name] = _sha_file(path)
    return result


def _hardware(context=None):
    gpu_name = None
    if context is not None and context.compute.name == "cuda":
        cp = context.compute.xp
        ordinal = context.compute._wp_device.ordinal
        properties = cp.cuda.runtime.getDeviceProperties(ordinal)
        name = properties["name"]
        gpu_name = name.decode() if isinstance(name, bytes) else str(name)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": gpu_name,
    }


def _resource_sample(context):
    rss = 0
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        rss = pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        pass
    device_used = 0
    device_free = 0
    device_total = 0
    if context.compute.name == "cuda":
        cp = context.compute.xp
        device_used = int(cp.get_default_memory_pool().used_bytes())
        device_free, device_total = cp.cuda.runtime.memGetInfo()
    return {
        "host_rss_bytes": int(rss),
        "host_limit_bytes": HOST_LIMIT,
        "device_used_bytes": int(device_used),
        "device_free_bytes": int(device_free),
        "device_total_bytes": int(device_total),
        "device_limit_bytes": DEVICE_LIMIT,
    }


def _resource_exhausted(sample):
    return sample["host_rss_bytes"] > HOST_LIMIT or (
        sample["device_used_bytes"] > DEVICE_LIMIT
    )


def _output_bytes(directory):
    return sum(
        path.stat().st_size
        for path in Path(directory).rglob("*")
        if path.is_file()
    )


def _build(config):
    context, source_state, fixture = wr_benchmark_fixture.periodic_inputs(
        config["points"],
        config["amplitude"],
        config["nonuniform"],
        seed_mode=config["seed_mode"],
        sentinel=config["sentinel"],
        device=config["device"],
    )
    p = context.p
    p.coupling = config["coupling"]
    p.integrator = "rk4"
    p.dt_guard = "warn"
    p.waveform_rtol = config["waveform_rtol"]
    p.waveform_max_iterations = config["waveform_max_iterations"]
    p.memory_limit = HOST_LIMIT
    p.device_memory_limit = DEVICE_LIMIT
    total_macros = config["total_microsteps"] // config["M"]
    if p.coupling == "synchronous":
        p.dt = config["h"]
        p.micro_steps = 1
        p.nsteps = config["total_microsteps"]
        output_macros = config["total_microsteps"]
    else:
        p.dt = config["M"] * config["h"]
        p.micro_steps = config["M"]
        p.nsteps = total_macros
        output_macros = total_macros
    if p.coupling == "multirate_wr":
        p.ion_nodes = config["s"]
        p.wr_initial = config["initial"]
        p.wr_relaxation = config["wr_relaxation"]
    else:
        p.ion_nodes = p.wr_initial = p.wr_relaxation = None
    time_module.validate_time_controls(p)
    peak = core.preflight_memory(context)
    p_source, z_source = multirate_wr_physics.pack_state(context, source_state)
    initial_modes = benchmark_helper._window_mode_amplitudes(
        context,
        p_source,
        z_source,
        config["seed_mode"],
        config["points"] // 2,
    )
    return {
        "context": context,
        "source_state": source_state,
        "fixture": fixture,
        "output_macros": output_macros,
        "initial_P": p_source,
        "initial_Z": z_source,
        "initial_modes": initial_modes,
        "preflight_bytes": peak,
    }


def _check_summary(checks):
    rows = []
    hard_failed = 0
    soft_failed = 0
    unavailable = 0
    for check in checks:
        available = bool(getattr(check, "available", True))
        passed = bool(getattr(check, "passed", False))
        hard = bool(getattr(check, "hard", True))
        if not available:
            unavailable += 1
            continue
        if not passed:
            if hard:
                hard_failed += 1
            else:
                soft_failed += 1
            if hard or len(rows) < 24:
                rows.append(
                {
                    "name": str(getattr(check, "name", "")),
                    "hard": hard,
                    "available": available,
                    "passed": passed,
                    "raw": _jsonable(getattr(check, "raw", 0.0)),
                    "threshold": _jsonable(
                        getattr(check, "threshold", 0.0)
                    ),
                }
                )
    return {
        "count": len(checks),
        "hard_failed": hard_failed,
        "soft_failed": soft_failed,
        "unavailable": unavailable,
        "failures": rows,
    }


def _totals(result):
    totals = getattr(result, "totals", None)
    if totals is None:
        frame = getattr(result, "frame", None)
        totals = None if frame is None else getattr(frame, "totals", None)
    if totals is None:
        raise ValueError("accepted result has no energy totals")
    electron = float(totals.electron_perp + totals.electron_parallel)
    return {
        "total": float(totals.energy),
        "kinetic": [float(value) for value in totals.kinetic],
        "electron": electron,
        "magnetic": float(totals.magnetic),
    }


def _coupling_summary(result):
    metadata = getattr(result, "coupling_metadata", None)
    if metadata is None:
        metadata = getattr(result, "waveform_metadata", None)
    metadata = {} if metadata is None else metadata
    residual_history = metadata.get("residual_history")
    if residual_history is None:
        residual_history = []
    return {
        "sweeps": metadata.get("sweeps"),
        "ion_nodes": metadata.get("ion_nodes"),
        "residual_history": _jsonable(residual_history),
        "reported_work": _jsonable(metadata.get("work")),
        "endpoint": _jsonable(metadata.get("endpoint")),
    }


class _WorkSpy:
    def __init__(self, context):
        self.context = context
        self.counts = {
            "full_stage_calls": 0,
            "full_ion_calls": 0,
            "full_rhs_calls": 0,
            "field_evaluations": 0,
            "coarse_node_calls": 0,
            "wall_projections": 0,
            "warp_dispatch_attempts": 0,
            "warp_kernel_results": 0,
        }
        self.stack = contextlib.ExitStack()

    def _patch(self, owner, name, builder):
        if not hasattr(owner, name):
            return
        original = getattr(owner, name)

        def wrapper(*args, **kwargs):
            return builder(original, *args, **kwargs)

        self.stack.enter_context(_Patcher(owner, name, wrapper))

    def __enter__(self):
        def stage_call(original, *args, **kwargs):
            self.counts["full_stage_calls"] += 1
            return original(*args, **kwargs)

        for owner in (
            stage,
            time_module,
            time_predictor,
            time_waveform,
            time_multirate,
        ):
            self._patch(owner, "evaluate_stage", stage_call)

        def build(original, fluid, species, bn, bhat, active_context):
            if species == 0:
                self.counts["full_ion_calls"] += 1
            return original(fluid, species, bn, bhat, active_context)

        self._patch(massless, "build_non_electric", build)

        def finish(original, active_stage, species, active_context):
            if species == 0:
                self.counts["full_rhs_calls"] += 1
            return original(active_stage, species, active_context)

        self._patch(massless, "finish_fluid", finish)

        def field_call(original, *args, **kwargs):
            self.counts["field_evaluations"] += 1
            return original(*args, **kwargs)

        self._patch(massless_field, "evaluate", field_call)

        original_transport = multirate_wr_physics.MasslessWRPhysicsAdapter.transport

        def transport(adapter, *args, **kwargs):
            self.counts["coarse_node_calls"] += 1
            return original_transport(adapter, *args, **kwargs)

        self.stack.enter_context(
            _Patcher(
                multirate_wr_physics.MasslessWRPhysicsAdapter,
                "transport",
                transport,
            )
        )

        def project(original, *args, **kwargs):
            self.counts["wall_projections"] += 1
            return original(*args, **kwargs)

        for owner in (
            wall,
            massless_field,
            multirate_wr_physics,
            multirate_wr_checks,
        ):
            self._patch(owner, "project_wall", project)

        def magnetic(original, *args, **kwargs):
            self.counts["warp_dispatch_attempts"] += 1
            result = original(*args, **kwargs)
            self.counts["warp_kernel_results"] += int(result is not None)
            return result

        for owner in (massless, massless_field):
            self._patch(owner, "_warp_magnetic_force", magnetic)
        return self

    def __exit__(self, exc_type, exc, traceback_value):
        return self.stack.__exit__(exc_type, exc, traceback_value)


class _Patcher:
    def __init__(self, owner, name, replacement):
        self.owner = owner
        self.name = name
        self.replacement = replacement

    def __enter__(self):
        self.original = getattr(self.owner, self.name)
        setattr(self.owner, self.name, self.replacement)

    def __exit__(self, exc_type, exc, traceback_value):
        setattr(self.owner, self.name, self.original)
        return False


def _identity(config, built):
    parameters = _jsonable(built["context"].p)
    fixture = _jsonable(built["fixture"])
    return {
        "scheme": _scheme(config),
        "parameters": parameters,
        "fixture": fixture,
        "initial_fourier": _jsonable(built["initial_modes"]),
        "sources": _source_manifest(),
    }


def _encode_meta(value):
    return np.frombuffer(
        _canonical(value).encode("utf-8"),
        dtype=np.uint8,
    )


def _decode_meta(arrays, name):
    if name not in arrays:
        raise ValueError(f"checkpoint metadata missing: {name}")
    return json.loads(bytes(arrays[name]).decode("utf-8"))


def _save_checkpoint(path, context, state, cumulative, identity, progress):
    p, z = multirate_wr_physics.pack_state(context, state)
    arrays = {
        "accepted": np.asarray(True, dtype=np.bool_),
        "schema": np.asarray(RUN_SCHEMA),
        "step": np.asarray(state.step, dtype=np.uint64),
        "generation": np.asarray(state.generation, dtype=np.uint64),
        "time": np.asarray(state.time, dtype=np.float64),
        "completed_macros": np.asarray(
            progress["completed_macros"], dtype=np.uint64
        ),
        "progress_microsteps": np.asarray(
            progress["progress_microsteps"], dtype=np.uint64
        ),
        "accepted_P": _host_array(p, context),
        "accepted_Z": _host_array(z, context),
        "cumulative_json": _encode_meta(io._cumulative_json(cumulative)),
        "identity_json": _encode_meta(identity),
        "counters_json": _encode_meta(
            {
                "next_generation": int(context.next_generation),
                "next_trial": int(context.next_trial),
            "next_source": int(context.next_source),
            "last_source_id": int(context.next_source) - 1,
                "spectral_id": int(context.spectral_id),
            }
        ),
    }
    _atomic_npz(path, arrays)


def _load_checkpoint(path):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def _restore_state(context, arrays):
    return multirate_wr_physics.state_from_packed(
        context,
        context.compute.array(arrays["accepted_P"], dtype=context.compute.xp.complex128),
        context.compute.array(arrays["accepted_Z"], dtype=context.compute.xp.complex128),
        step=int(arrays["step"]),
        generation=int(arrays["generation"]),
        time=float(arrays["time"]),
    )


def _restore_checkpoint(directory, context, identity):
    directory = Path(directory)
    checkpoint = directory / "checkpoint.npz"
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    expected = run.get("checkpoint_sha256")
    if expected is None or _sha_file(checkpoint) != expected:
        raise ValueError("resume checkpoint hash mismatch")
    arrays = _load_checkpoint(checkpoint)
    if not bool(arrays["accepted"]):
        raise ValueError("resume checkpoint is not an accepted endpoint")
    saved_identity = _decode_meta(arrays, "identity_json")
    if saved_identity != _jsonable(identity):
        raise ValueError("resume scheme/input/source identity mismatch")
    cumulative = io._cumulative_from_json(
        _decode_meta(arrays, "cumulative_json")
    )
    state = _restore_state(context, arrays)
    if (
        state.step != cumulative.last_step
        or state.time != cumulative.last_time
        or state.time != core.time_at_step(context.p, state.step)
    ):
        raise ValueError("resume state and cumulative clocks disagree")
    counters = _decode_meta(arrays, "counters_json")
    if int(counters["last_source_id"]) != int(counters["next_source"]) - 1:
        raise ValueError("resume source counter identity is inconsistent")
    context.next_generation = int(counters["next_generation"])
    context.next_trial = int(counters["next_trial"])
    context.next_source = int(counters["next_source"])
    context.spectral_id = int(counters["spectral_id"])
    if context.next_source < 1 or context.next_generation <= state.generation:
        raise ValueError("resume context counters are inconsistent")
    progress = {
        "completed_macros": int(arrays["completed_macros"]),
        "progress_microsteps": int(arrays["progress_microsteps"]),
    }
    expected_progress = (
        state.step
        if context.p.coupling == "synchronous"
        else state.step * context.p.micro_steps
    )
    if (
        progress["completed_macros"] != state.step
        or progress["progress_microsteps"] != expected_progress
    ):
        raise ValueError("resume progress and accepted state disagree")
    return state, cumulative, progress, arrays


def _copy_prefix(previous, output, completed_macros):
    snapshot = Path(previous) / "snapshots.npz"
    if snapshot.exists():
        shutil.copyfile(snapshot, Path(output) / "snapshots.npz")
    source = Path(previous) / "history.jsonl"
    lines = source.read_text(encoding="utf-8").splitlines()
    if len(lines) < completed_macros:
        raise ValueError("resume history is shorter than its checkpoint")
    target = Path(output) / "history.jsonl"
    with target.open("w", encoding="utf-8") as stream:
        for line in lines[:completed_macros]:
            stream.write(line + "\n")


def _snapshot_arrays(
    context,
    source_p,
    source_z,
    first_p,
    first_z,
    final_p,
    final_z,
):
    return {
        "source_P": _host_array(source_p, context),
        "source_Z": _host_array(source_z, context),
        "first_P": _host_array(first_p, context),
        "first_Z": _host_array(first_z, context),
        "final_P": _host_array(final_p, context),
        "final_Z": _host_array(final_z, context),
    }


def _history_row(
    context,
    config,
    result,
    cumulative,
    spy,
    completed_macros,
    progress_microsteps,
    elapsed,
    memory,
    writer_seconds,
):
    p, z = multirate_wr_physics.pack_state(context, result.frame.canonical_state)
    endpoint_tag = getattr(result.frame, "tag", None)
    amplitudes = benchmark_helper._window_mode_amplitudes(
        context,
        p,
        z,
        config["seed_mode"],
        config["points"] // 2,
    )
    return {
        "schema": RUN_SCHEMA,
        "completed_macros": int(completed_macros),
        "progress_microsteps": int(progress_microsteps),
        "clock": {
            "step": int(result.frame.canonical_state.step),
            "generation": int(result.frame.canonical_state.generation),
            "time": float(result.frame.canonical_state.time),
            "trial": int(getattr(endpoint_tag, "trial", -1)),
        },
        "amplitudes": _jsonable(amplitudes),
        "energies": _totals(result),
        "checks": _check_summary(result.checks),
        "cumulative": _jsonable(io._cumulative_json(cumulative)),
        "coupling": _coupling_summary(result),
        "node_count": (
            config["s"] if config["coupling"] == "multirate_wr" else 1
        ),
        "work": {
            **spy.counts,
            "reported": _jsonable(
                (
                    getattr(result, "coupling_metadata", {}) or {}
                ).get("work")
            ),
        },
        "timing": {
            "advance_seconds": float(elapsed),
            "writer_seconds": float(writer_seconds),
            "synchronized_advance": True,
        },
        "memory": memory,
    }


def _append_history(path, row):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(_canonical(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _failure_payload(error, spy, elapsed, context):
    metadata = getattr(error, "coupling_metadata", None)
    return {
        "type": type(error).__name__,
        "code": getattr(error, "code", "CampaignFailure"),
        "message": str(error),
        "traceback": traceback.format_exc(),
        "failure_seconds": float(elapsed),
        "work": spy.counts,
        "checks": _check_summary(getattr(error, "checks", ())),
        "coupling_metadata": _jsonable(metadata),
        "adapter_work": _jsonable(getattr(error, "adapter_work", None)),
        "adapter_metadata": _jsonable(
            getattr(error, "adapter_metadata", None)
        ),
        "residual_history": _jsonable(
            getattr(error, "residual_history", None)
        ),
        "mixed_diagnostic_work": _jsonable(
            getattr(error, "mixed_diagnostic_work", None)
        ),
        "partial_array_names": sorted(
            str(name) for name in getattr(error, "partial_arrays", {})
        ),
    }


def _save_failure(directory, error, spy, elapsed, context):
    payload = _failure_payload(error, spy, elapsed, context)
    partial = {}
    for name, value in getattr(error, "partial_arrays", {}).items():
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            partial[str(name)] = _host_array(value, context)
    if partial:
        _atomic_npz(Path(directory) / "failure_partial.npz", partial)
    _atomic_json(Path(directory) / "failure.json", payload)
    return payload


def run_campaign(config):
    """Run one bounded segmented campaign and return its compact receipt."""
    cfg, advance_fn = _normalize(config)
    output = cfg["output"]
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    built = _build(cfg)
    context = built["context"]
    identity = _identity(cfg, built)
    total_macros = built["output_macros"]
    setup_start = time.monotonic()
    if cfg["resume"] is None:
        cumulative = cumulative_record()
        committed = stage.initialize(
            built["source_state"],
            context,
            cumulative,
            equilibrium=False,
        )
        progress = {"completed_macros": 0, "progress_microsteps": 0}
        previous_arrays = None
    else:
        state, cumulative, progress, previous_arrays = _restore_checkpoint(
            cfg["resume"], context, identity
        )
        committed = SimpleNamespace(
            state=state,
            cache=None,
            accepted=True,
            coupling_metadata=None,
        )
    setup_seconds = time.monotonic() - setup_start
    output.mkdir(parents=True)
    if cfg["resume"] is not None:
        _copy_prefix(
            cfg["resume"],
            output,
            progress["completed_macros"],
        )
    else:
        initial_p = built["initial_P"]
        initial_z = built["initial_Z"]
        _atomic_npz(
            output / "snapshots.npz",
            {
                "source_P": _host_array(initial_p, context),
                "source_Z": _host_array(initial_z, context),
            },
        )
    run_doc = {
        "schema": RUN_SCHEMA,
        "status": "running",
        "identity": identity,
        "identity_sha256": hashlib.sha256(
            _canonical(identity).encode()
        ).hexdigest(),
        "hardware": _hardware(context),
        "limits": {
            "host_bytes": HOST_LIMIT,
            "device_bytes": DEVICE_LIMIT,
            "output_bytes": OUTPUT_LIMIT,
            "wall_seconds": float(cfg["wall_seconds"]),
        },
        "total_macros": int(total_macros),
        "segment_macros": int(cfg["segment_macros"]),
        "preflight_bytes": int(built["preflight_bytes"]),
        "setup_seconds": float(setup_seconds),
        "advance_seconds": 0.0,
        "warmup": "none",
        "checkpoint_sha256": None,
    }
    if previous_arrays is not None:
        run_doc["checkpoint_sha256"] = _sha_file(
            cfg["resume"] / "checkpoint.npz"
        )
    _atomic_json(output / "run.json", run_doc)
    history_path = output / "history.jsonl"
    history_path.touch(exist_ok=True)
    deadline = time.monotonic() + float(cfg["wall_seconds"]) - setup_seconds
    completed_this = 0
    stop_reason = None
    writer_total = 0.0
    advance_total = 0.0
    memory = _resource_sample(context)
    try:
        while progress["completed_macros"] < total_macros:
            if completed_this >= cfg["segment_macros"]:
                stop_reason = "segment"
                break
            if time.monotonic() >= deadline:
                stop_reason = "wall"
                break
            context.compute.synchronize()
            advance_start = time.monotonic()
            spy = _WorkSpy(context)
            with spy:
                result = advance_fn(committed, cumulative, context)
            context.compute.synchronize()
            elapsed = time.monotonic() - advance_start
            advance_total += elapsed
            if not bool(getattr(result, "accepted", False)):
                raise Failure("CampaignAcceptance", "advance did not accept")
            state = committed.state
            if (
                state.step != cumulative.last_step
                or state.time != cumulative.last_time
                or state.time != core.time_at_step(context.p, state.step)
            ):
                raise Failure(
                    "CampaignClock",
                    "accepted state and cumulative clock disagree",
                )
            progress["completed_macros"] += 1
            progress["progress_microsteps"] += (
                1 if cfg["coupling"] == "synchronous" else cfg["M"]
            )
            memory = _resource_sample(context)
            writer_start = time.monotonic()
            p, z = multirate_wr_physics.pack_state(context, state)
            previous = _load_checkpoint(output / "snapshots.npz")
            source_p = previous["source_P"]
            source_z = previous["source_Z"]
            first_p = previous.get("first_P", p)
            first_z = previous.get("first_Z", z)
            pending_snapshots = output / ".pending-snapshots.npz"
            _atomic_npz(
                pending_snapshots,
                _snapshot_arrays(
                    context,
                    source_p,
                    source_z,
                    first_p,
                    first_z,
                    p,
                    z,
                ),
            )
            pending_checkpoint = output / ".pending-checkpoint.npz"
            _save_checkpoint(
                pending_checkpoint,
                context,
                state,
                cumulative,
                identity,
                progress,
            )
            checkpoint_write_seconds = time.monotonic() - writer_start
            row = _history_row(
                context,
                cfg,
                result,
                cumulative,
                spy,
                progress["completed_macros"],
                progress["progress_microsteps"],
                elapsed,
                memory,
                checkpoint_write_seconds,
            )
            _append_history(history_path, row)
            os.replace(pending_snapshots, output / "snapshots.npz")
            os.replace(pending_checkpoint, output / "checkpoint.npz")
            writer_total += time.monotonic() - writer_start
            run_doc["checkpoint_sha256"] = _sha_file(
                output / "checkpoint.npz"
            )
            run_doc["status"] = (
                "complete"
                if progress["completed_macros"] == total_macros
                else "partial"
            )
            run_doc["completed_macros"] = progress["completed_macros"]
            run_doc["progress_microsteps"] = progress["progress_microsteps"]
            run_doc["writer_seconds"] = writer_total
            run_doc["advance_seconds"] = advance_total
            _atomic_json(output / "run.json", run_doc)
            completed_this += 1
            if _resource_exhausted(memory):
                stop_reason = "resource"
                break
            if _output_bytes(output) > OUTPUT_LIMIT:
                stop_reason = "output-cap"
                break
    except Exception as error:
        elapsed = time.monotonic() - advance_start
        publication_error = None
        try:
            _save_failure(output, error, spy, elapsed, context)
        except Exception as writer_error:
            publication_error = (
                f"{type(writer_error).__name__}: {writer_error}"
            )
            minimal = {
                "type": type(error).__name__,
                "code": getattr(error, "code", "CampaignFailure"),
                "message": str(error),
                "failure_seconds": float(elapsed),
                "metadata_publication_error": publication_error,
            }
            try:
                _atomic_json(output / "failure.json", minimal)
            except Exception as minimal_error:
                publication_error += (
                    "; minimal failure publication also failed: "
                    f"{type(minimal_error).__name__}: {minimal_error}"
                )
        run_doc["status"] = "failed"
        run_doc["failure"] = str(getattr(error, "code", type(error).__name__))
        run_doc["completed_macros"] = progress["completed_macros"]
        run_doc["failure_seconds"] = elapsed
        if publication_error is not None:
            run_doc["metadata_publication_error"] = publication_error
        try:
            _atomic_json(output / "run.json", run_doc)
        except Exception as run_writer_error:
            if not hasattr(error, "campaign_run_publication_error"):
                error.campaign_run_publication_error = (
                    f"{type(run_writer_error).__name__}: {run_writer_error}"
                )
        raise
    complete = progress["completed_macros"] == total_macros
    prefix_arrays = _load_checkpoint(output / "checkpoint.npz")
    status = "complete" if complete else "partial"
    run_doc.update(
        {
            "status": status,
            "completed_macros": progress["completed_macros"],
            "progress_microsteps": progress["progress_microsteps"],
            "stop_reason": stop_reason,
            "writer_seconds": writer_total,
            "advance_seconds": advance_total,
            "memory": memory,
            "completed": complete,
        }
    )
    _atomic_json(output / "run.json", run_doc)
    history_rows = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
    ]
    return {
        "status": status,
        "output": output,
        "checkpoint": output / "checkpoint.npz",
        "history": history_path,
        "history_rows": history_rows,
        "completed_macros": progress["completed_macros"],
        "total_macros": total_macros,
        "progress_microsteps": progress["progress_microsteps"],
        "stop_reason": stop_reason,
        "identity_sha256": run_doc["identity_sha256"],
        "checkpoint_arrays": prefix_arrays,
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--coupling", choices=COUPLINGS, default="multirate_wr")
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--s", type=int, default=3)
    parser.add_argument("--h", type=float, required=True)
    parser.add_argument("--total-microsteps", type=int, default=4096)
    parser.add_argument("--segment-macros", type=int, required=True)
    parser.add_argument("--wall-seconds", type=float, default=1600.0)
    parser.add_argument(
        "--initial", choices=("tangent", "constant"), default="tangent"
    )
    parser.add_argument("--points", type=int, default=33)
    parser.add_argument("--nonuniform", type=float, default=0.1)
    parser.add_argument("--amplitude", type=float, default=0.02)
    parser.add_argument("--sentinel", type=float, default=1.0e-4)
    parser.add_argument("--seed-mode", type=int, default=1)
    parser.add_argument("--waveform-rtol", type=float, default=1.0e-10)
    parser.add_argument("--waveform-max-iterations", type=int, default=12)
    parser.add_argument("--wr-relaxation", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    config = vars(args)
    try:
        result = run_campaign(config)
    except FileExistsError as error:
        print(str(error), file=os.sys.stderr)
        return 3
    except (ValueError, OSError) as error:
        print(f"campaign input error: {error}", file=os.sys.stderr)
        return 2
    except Failure as error:
        print(f"campaign failure {error.code}: {error}", file=os.sys.stderr)
        return 4
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(result["output"]),
                "completed_macros": result["completed_macros"],
                "total_macros": result["total_macros"],
                "stop_reason": result["stop_reason"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
