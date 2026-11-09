#!/usr/bin/env python3
"""Matched-horizon cost harness for complete A0, T1 and A1-WR advances."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import resource
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


for thread_name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_name, "1")


def _source_root():
    candidate = Path(__file__).resolve().parents[1]
    if (candidate / "src" / "fomhd").is_dir():
        return candidate
    configured = os.environ.get("FOMHD_SOURCE_ROOT")
    if configured is None:
        raise RuntimeError(
            "scratch execution requires FOMHD_SOURCE_ROOT for the promoted source"
        )
    return Path(configured).resolve()


SOURCE_ROOT = _source_root()
for runtime_path in (
    SOURCE_ROOT / "src",
    SOURCE_ROOT / "tests" / "warp",
    SOURCE_ROOT,
):
    if str(runtime_path) not in sys.path:
        sys.path.insert(0, str(runtime_path))

import numpy as np  # noqa: E402

from fomhd import core, io, stage  # noqa: E402
from fomhd import time as time_module  # noqa: E402
from fomhd.massless import MODEL_NAME  # noqa: E402
from fomhd.types import (  # noqa: E402
    Failure,
    FluidState,
    State,
    cumulative_record,
)
from tools import warp_wr_periodic as periodic  # noqa: E402


SCHEMA = "fomhd-matched-wr-cost-v1"
H = 1.9794733885623e-10
AMPLITUDE = 0.02
NONUNIFORM = 0.1
SENTINEL = 1.0e-4
SEED_MODE = 1
RTOL = 1.0e-10
MAX_ITERATIONS = 12
RELAXATION = 1.0
WARMUP = 1
REPEATS = 3
MAX_WALL_SECONDS = 900.0
HOST_LIMIT = periodic.HOST_LIMIT
DEVICE_LIMIT = periodic.DEVICE_LIMIT
METHODS = ("synchronous", "waveform", "multirate_wr")


class CostLimit(Failure):
    def __init__(self, message):
        super().__init__("CostLimit", message)


def _validate_config(value):
    if not isinstance(value, dict):
        raise ValueError("configuration must be a JSON object")
    allowed = {
        "device",
        "points",
        "horizon_microsteps",
        "wr_M",
        "wr_ion_nodes",
        "methods",
        "wall_seconds",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError("unknown controls: " + ", ".join(unknown))
    result = {
        "device": "cpu",
        "points": None,
        "horizon_microsteps": None,
        "wr_M": None,
        "wr_ion_nodes": [3, 5],
        "methods": list(METHODS),
        "wall_seconds": MAX_WALL_SECONDS,
    }
    result.update(copy.deepcopy(value))
    if result["device"] not in ("cpu", "cuda:0"):
        raise ValueError("device must be cpu or cuda:0")
    if result["device"] == "cuda:0":
        if os.environ.get("FOMHD_TEST_GPU") != "1":
            raise ValueError("CUDA cost run requires FOMHD_TEST_GPU=1")
        if not os.environ.get("SLURM_JOB_ID"):
            raise ValueError("CUDA cost run requires an owned Slurm allocation")
    points = result["points"]
    if type(points) is not int or points < 9 or points % 2 != 1:
        raise ValueError("points must be an odd integer >= 9")
    horizon = result["horizon_microsteps"]
    if type(horizon) is not int or horizon not in (8, 32):
        raise ValueError("horizon_microsteps must be 8 or 32")
    wr_m = horizon if result["wr_M"] is None else result["wr_M"]
    if type(wr_m) is not int or wr_m <= 0 or horizon % wr_m:
        raise ValueError("wr_M must be a positive divisor of the horizon")
    result["wr_M"] = wr_m
    nodes = result["wr_ion_nodes"]
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("wr_ion_nodes must be a nonempty list")
    if any(type(node) is not int or node not in (3, 5) for node in nodes):
        raise ValueError("wr_ion_nodes entries must be 3 or 5")
    if len(set(nodes)) != len(nodes):
        raise ValueError("wr_ion_nodes entries must be unique")
    methods = result["methods"]
    if not isinstance(methods, list) or not methods:
        raise ValueError("methods must be a nonempty list")
    if len(set(methods)) != len(methods):
        raise ValueError("methods entries must be unique")
    if any(method not in METHODS for method in methods):
        raise ValueError("methods entries must be synchronous, waveform, or multirate_wr")
    wall = result["wall_seconds"]
    if (
        type(wall) not in (int, float)
        or isinstance(wall, bool)
        or not math.isfinite(float(wall))
        or float(wall) <= 0.0
        or float(wall) > MAX_WALL_SECONDS
    ):
        raise ValueError("wall_seconds must be finite in (0, 900]")
    result["wall_seconds"] = float(wall)
    result["horizon_seconds"] = float(horizon) * H
    result["warmup"] = WARMUP
    result["repeats"] = REPEATS
    return result


def _campaign_config(config, coupling, *, M, total_microsteps, ion_nodes=3):
    return {
        "device": config["device"],
        "coupling": coupling,
        "M": int(M),
        "s": int(ion_nodes),
        "h": H,
        "total_microsteps": int(total_microsteps),
        "segment_macros": 1,
        "wall_seconds": config["wall_seconds"],
        "initial": "tangent",
        "points": int(config["points"]),
        "nonuniform": NONUNIFORM,
        "amplitude": AMPLITUDE,
        "sentinel": SENTINEL,
        "seed_mode": SEED_MODE,
        "waveform_rtol": RTOL,
        "waveform_max_iterations": MAX_ITERATIONS,
        "wr_relaxation": RELAXATION,
        "output": Path(tempfile.gettempdir()) / "unused-cost-output",
    }


def _build_method(config, method, ion_nodes):
    horizon = config["horizon_microsteps"]
    if method == "synchronous":
        campaign = _campaign_config(
            config,
            "synchronous",
            M=1,
            total_microsteps=horizon,
        )
    elif method == "waveform":
        campaign = _campaign_config(
            config,
            "waveform",
            M=horizon,
            total_microsteps=horizon,
        )
    else:
        campaign = _campaign_config(
            config,
            "multirate_wr",
            M=config["wr_M"],
            total_microsteps=config["wr_M"],
            ion_nodes=ion_nodes,
        )
    built = periodic._build(campaign)
    context = built["context"]
    if method != "synchronous":
        context.p.dt = config["horizon_seconds"]
        time_module.validate_time_controls(context.p)
        core.preflight_memory(context)
    return built


def _state_snapshot(context, state):
    xp = context.compute.xp

    def copy(value):
        if xp is np:
            return np.asarray(value).copy()
        return xp.asnumpy(value).copy()

    return {
        "B": copy(state.B),
        "fluids": tuple(
            {
                name: copy(getattr(state.fluid[species], name))
                for name in ("n", "mom", "perp", "parallel")
            }
            for species in range(2)
        ),
        "step": int(state.step),
        "generation": int(state.generation),
        "time": float(state.time),
    }


def _state_from_snapshot(context, snapshot):
    xp = context.compute.xp

    def array(value):
        dtype = xp.complex128
        if xp is np:
            return np.asarray(value, dtype=dtype).copy()
        return xp.asarray(value, dtype=dtype).copy()

    fluids = tuple(
        FluidState(
            n=array(values["n"]),
            mom=array(values["mom"]),
            perp=array(values["perp"]),
            parallel=array(values["parallel"]),
        )
        for values in snapshot["fluids"]
    )
    return State(
        fluid=fluids,
        B=array(snapshot["B"]),
        step=int(snapshot["step"]),
        generation=int(snapshot["generation"]),
        time=float(snapshot["time"]),
    )


def _state_digest(snapshot):
    digest = hashlib.sha256()
    for name, value in (
        ("B", snapshot["B"]),
        *(
            (
                f"fluid{species}.{field}",
                snapshot["fluids"][species][field],
            )
            for species in range(2)
            for field in ("n", "mom", "perp", "parallel")
        ),
    ):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _cumulative_digest(value):
    return hashlib.sha256(
        periodic._canonical(value).encode("utf-8")
    ).hexdigest()


def _state_magnitudes(snapshot):
    values = {"B": snapshot["B"]}
    for species in range(2):
        for field in ("n", "mom", "perp", "parallel"):
            values[f"{species}_{field}"] = snapshot["fluids"][species][field]
    return {
        name: float(np.max(np.abs(value))) if np.asarray(value).size else 0.0
        for name, value in values.items()
    }


def _fixture_identity(built):
    fixture = built["fixture"]
    return {
        "target_mode": int(fixture["target_mode"]),
        "sentinel_mode": int(fixture["sentinel_mode"]),
        "sentinel_retained": bool(fixture["sentinel_retained"]),
        "physical_source_origin": str(fixture["physical_source_origin"]),
        "initial_fourier": periodic._jsonable(built["initial_modes"]),
    }


def _budget_identity(context):
    hard = sorted(
        name for name, budget in context.p.budgets.items() if budget.hard
    )
    observe = sorted(
        name for name, budget in context.p.budgets.items() if not budget.hard
    )
    if "quasineutrality.linf" not in hard:
        raise ValueError("fixture softened required quasineutrality.linf")
    if "theta_consistency.linf" not in observe:
        raise ValueError("fixture lacks expected optional theta consistency gate")
    registry = hashlib.sha256(
        json.dumps(
            {
                "hard": hard,
                "observe": observe,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "registry_sha256": registry,
        "hard_count": len(hard),
        "observe_count": len(observe),
        "quasineutrality_linf_hard": True,
        "theta_consistency_linf_observe": True,
        "observe_names": observe,
    }


def _counter_snapshot(context):
    return {
        "next_generation": int(context.next_generation),
        "next_trial": int(context.next_trial),
        "next_source": int(context.next_source),
        "spectral_id": int(context.spectral_id),
    }


def _restore_counters(context, counters):
    for name, value in counters.items():
        setattr(context, name, int(value))


def _memory_sample(context, peak):
    sample = periodic._resource_sample(context)
    rss = int(sample["host_rss_bytes"])
    peak["rss"] = max(int(peak["rss"]), rss)
    if context.compute.name == "cuda":
        pool = context.compute.xp.get_default_memory_pool()
        sample["device_reserved_bytes"] = int(pool.total_bytes())
        peak["device_used"] = max(
            int(peak["device_used"]),
            int(sample["device_used_bytes"]),
        )
        peak["device_reserved"] = max(
            int(peak["device_reserved"]),
            int(sample["device_reserved_bytes"]),
        )
    if (
        int(sample["host_rss_bytes"]) > HOST_LIMIT
        or int(sample["device_used_bytes"]) > DEVICE_LIMIT
    ):
        raise CostLimit("host or device resource limit exceeded")
    return sample


def _process_peak_rss():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _accepted_clock(context, committed, cumulative, method):
    state = committed.state
    if not bool(getattr(committed, "accepted", False)):
        raise Failure("CostAcceptance", "advance did not accept")
    io._check_accepted(committed)
    if (
        state.step != cumulative.last_step
        or state.time != cumulative.last_time
        or state.time != core.time_at_step(context.p, state.step)
    ):
        raise Failure("CostClock", "accepted state and cumulative clocks disagree")
    expected_step = context.p.nsteps
    if method == "multirate_wr" or method == "waveform":
        expected_step = 1
    if state.step != expected_step:
        raise Failure(
            "CostClock",
            f"accepted step {state.step} != expected {expected_step}",
        )
    if not math.isclose(
        state.time,
        float(context.p.start_time) + context.p.nsteps * context.p.dt,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise Failure("CostClock", "accepted time differs from declared horizon")
    return {
        "step": int(state.step),
        "time_s": float(state.time),
        "generation": int(state.generation),
        "cache_tag": periodic._jsonable(committed.cache.tag),
        "accepted_intervals": int(cumulative.accepted_intervals),
        "last_trial": int(cumulative.last_trial),
    }


def _run_horizon(
    config,
    method,
    context,
    state_snapshot,
    cumulative_snapshot,
    counters,
    *,
    advance_fn,
):
    input_state_sha256 = _state_digest(state_snapshot)
    input_cumulative_sha256 = _cumulative_digest(cumulative_snapshot)
    peak = {
        "rss": _process_peak_rss(),
        "device_used": 0,
        "device_reserved": 0,
    }
    samples = []
    reset_start = time.perf_counter()
    gc.collect()
    context.compute.synchronize()
    _restore_counters(context, counters)
    cumulative = io._cumulative_from_json(copy.deepcopy(cumulative_snapshot))
    state = _state_from_snapshot(context, state_snapshot)
    reset_seconds = time.perf_counter() - reset_start
    setup_start = time.perf_counter()
    committed = stage.initialize(state, context, cumulative, False, {})
    context.compute.synchronize()
    pre_run_counters = _counter_snapshot(context)
    samples.append(_memory_sample(context, peak))
    setup_seconds = time.perf_counter() - setup_start

    deadline = time.perf_counter() + config["wall_seconds"]
    spy = periodic._WorkSpy(context)
    failure = None
    advance_calls = 0
    final_result = None
    run_start = time.perf_counter()
    try:
        with spy:
            for _ in range(config["horizon_microsteps"] if method == "synchronous" else 1):
                if time.perf_counter() > deadline:
                    raise CostLimit("matched-horizon wall limit exceeded")
                final_result = advance_fn(committed, cumulative, context)
                advance_calls += 1
        context.compute.synchronize()
    except Exception as error:
        context.compute.synchronize()
        failure = error
    run_seconds = time.perf_counter() - run_start
    samples.append(_memory_sample(context, peak))
    row = {
        "method": method,
        "reset_seconds": float(reset_seconds),
        "setup_seconds": float(setup_seconds),
        "run_seconds": float(run_seconds),
        "total_seconds": float(reset_seconds + setup_seconds + run_seconds),
        "advance_calls": int(advance_calls),
        "input_state_sha256": input_state_sha256,
        "input_cumulative_sha256": input_cumulative_sha256,
        "pre_run_generation_counters": pre_run_counters,
        "post_run_generation_counters": _counter_snapshot(context),
        "spy": dict(spy.counts),
        "memory_samples": samples,
        "memory_peak": {
            "host_rss_bytes": int(peak["rss"]),
            "device_used_bytes": int(peak["device_used"]),
            "device_reserved_bytes": int(peak["device_reserved"]),
            "host_limit_bytes": HOST_LIMIT,
            "device_limit_bytes": DEVICE_LIMIT,
        },
        "generation_counters": copy.deepcopy(counters),
    }
    if failure is not None:
        row.update(
            status="failed",
            failure={
                "type": type(failure).__name__,
                "code": str(getattr(failure, "code", "CostFailure")),
                "message": str(failure),
                "traceback": traceback.format_exc(),
                "checks": periodic._check_summary(
                    getattr(failure, "checks", ())
                ),
                "work": periodic._jsonable(
                    getattr(failure, "adapter_work", None)
                ),
                "coupling_metadata": periodic._jsonable(
                    getattr(failure, "coupling_metadata", None)
                ),
            },
        )
        return row, spy.counts, failure

    metadata = getattr(final_result, "coupling_metadata", None) or {}
    transaction_work = periodic._jsonable(metadata.get("work"))
    bundle_work = periodic._jsonable(
        metadata.get("bundle_diagnostic_work")
    )
    result_work = periodic._jsonable(
        metadata.get("result_diagnostic_work")
    )
    row.update(
        status="accepted",
        clock=_accepted_clock(context, committed, cumulative, method),
        checks=periodic._check_summary(final_result.checks),
        transaction_work=transaction_work,
        bundle_diagnostic_work=bundle_work,
        result_diagnostic_work=result_work,
        counter_roles={
            "ion_transport_producer_calls": int(
                spy.counts["full_ion_calls"]
            ),
            "stage_evaluator_calls": int(
                spy.counts["full_stage_calls"]
            ),
            "full_rhs_finish_calls": int(spy.counts["full_rhs_calls"]),
            "coarse_A_producer_calls": int(spy.counts["coarse_node_calls"]),
            "fine_G_C_field_callback_calls": int(
                spy.counts["field_evaluations"]
            ),
            "field_callback_produces_both_G_and_C": True,
            "wall_projection_calls": int(spy.counts["wall_projections"]),
            "warp_dispatch_attempts": int(
                spy.counts["warp_dispatch_attempts"]
            ),
            "warp_kernel_results": int(spy.counts["warp_kernel_results"]),
            "full_ion_power_ledger_values": (
                None
                if transaction_work is None
                else int(transaction_work.get("full_ion_power_ledger_values", 0))
            ),
            "bundle_diagnostic_calls": bundle_work,
            "result_diagnostic_calls": result_work,
            "intersection_policy": (
                "raw counters are reported separately; overlapping producer "
                "counters are never summed into a synthetic work total"
            ),
        },
        residual_history_length=(
            None
            if metadata.get("residual_history") is None
            else len(metadata["residual_history"])
        ),
    )
    return row, spy.counts, None


def _method_identity(
    config,
    method,
    built,
    seed_snapshot,
    seed_cumulative,
    seed_counters,
):
    context = built["context"]
    plan = {
        "synchronous": (1, config["horizon_microsteps"], H),
        "waveform": (
            config["horizon_microsteps"],
            1,
            config["horizon_seconds"],
        ),
        "multirate_wr": (
            config["wr_M"],
            1,
            config["horizon_seconds"],
        ),
    }[method]
    return {
        "method": method,
        "coupling": context.p.coupling,
        "micro_steps": int(plan[0]),
        "declared_steps": int(plan[1]),
        "dt_s": float(plan[2]),
        "horizon_seconds": float(plan[1] * plan[2]),
        "ion_nodes": (
            int(context.p.ion_nodes)
            if method == "multirate_wr"
            else None
        ),
        "waveform_rtol": float(context.p.waveform_rtol),
        "waveform_max_iterations": int(context.p.waveform_max_iterations),
        "wr_initial": context.p.wr_initial,
        "wr_relaxation": (
            None
            if context.p.wr_relaxation is None
            else float(context.p.wr_relaxation)
        ),
        "initial_state_sha256": _state_digest(seed_snapshot),
        "initial_state_magnitudes": _state_magnitudes(seed_snapshot),
        "input_cumulative_sha256": _cumulative_digest(seed_cumulative),
        "fixture": _fixture_identity(built),
        "budgets": _budget_identity(context),
        "initial_clock": {
            "step": int(seed_snapshot["step"]),
            "time_s": float(seed_snapshot["time"]),
            "generation": int(seed_snapshot["generation"]),
        },
        "generation_counters": seed_counters,
        "preflight_bytes": int(built["preflight_bytes"]),
        "nonuniform_amplitude": float(NONUNIFORM),
        "electric_amplitude": float(AMPLITUDE),
        "high_mode_sentinel": float(SENTINEL),
    }


def _run_method(config, method, ion_nodes, advance_fn):
    method_start = time.perf_counter()
    built = _build_method(config, method, ion_nodes)
    context = built["context"]
    counters = _counter_snapshot(context)
    cumulative = cumulative_record()
    committed = stage.initialize(
        built["source_state"],
        context,
        cumulative,
        False,
        {},
    )
    context.compute.synchronize()
    state_snapshot = _state_snapshot(context, committed.state)
    cumulative_snapshot = copy.deepcopy(io._cumulative_json(cumulative))
    identity = _method_identity(
        config,
        method,
        built,
        state_snapshot,
        cumulative_snapshot,
        counters,
    )
    identity["hardware"] = periodic._hardware(context)
    context_setup_seconds = float(time.perf_counter() - method_start)
    warmup, warmup_counts, failure = _run_horizon(
        config,
        method,
        context,
        state_snapshot,
        cumulative_snapshot,
        counters,
        advance_fn=advance_fn,
    )
    rows = []
    if failure is None:
        for _ in range(REPEATS):
            row, counts, failure = _run_horizon(
                config,
                method,
                context,
                state_snapshot,
                cumulative_snapshot,
                counters,
                advance_fn=advance_fn,
            )
            rows.append(row)
            if failure is not None:
                break
    report = {
        "status": "accepted" if failure is None else "failed",
        "identity": identity,
        "context_setup_seconds": context_setup_seconds,
        "warmup": warmup,
        "warmup_spy": warmup_counts,
        "repeats": rows,
        "ledger_resets_inside_physical_run": 0,
    }
    if failure is None:
        seconds = [row["run_seconds"] for row in rows]
        report["run_seconds"] = {
            "samples": seconds,
            "median": float(np.median(seconds)),
            "minimum": float(min(seconds)),
            "maximum": float(max(seconds)),
        }
    return report, failure


def _source_identity(config):
    sources = periodic._source_manifest()
    sources["tools.warp_wr_cost"] = periodic._sha_file(Path(__file__))
    return {
        "source_root": str(SOURCE_ROOT),
        "sources": sources,
        "physics": {
            "model": MODEL_NAME,
            "H_s": config["horizon_seconds"],
            "base_h_s": H,
            "horizon_microsteps": config["horizon_microsteps"],
            "single_cpu_thread": True,
        },
    }


def run(config, *, advance_fn=None):
    """Run one CPU/GPU authored matched-horizon measurement."""

    cfg = _validate_config(config)
    advance_fn = time_module.advance if advance_fn is None else advance_fn
    plan = []
    for method in cfg["methods"]:
        if method == "multirate_wr":
            for node in cfg["wr_ion_nodes"]:
                plan.append((method, int(node)))
        else:
            plan.append((method, None))
    report = {
        "schema": SCHEMA,
        "status": "running",
        "config": cfg,
        "identity": _source_identity(cfg),
        "hardware": periodic._hardware(),
        "limits": {
            "host_bytes": HOST_LIMIT,
            "device_bytes": DEVICE_LIMIT,
            "wall_seconds_per_run": cfg["wall_seconds"],
            "warmup": WARMUP,
            "repeats": REPEATS,
            "warmup_semantics": (
                "one complete synchronized real horizon from the same cloned "
                "accepted input used by every measured repeat"
            ),
        },
        "methods": [],
        "no_scientific_acceptance": True,
        "work_counter_policy": (
            "raw and role-separated counters only; overlapping producer calls "
            "are not combined into a synthetic total"
        ),
        "speedup_claim": None,
    }
    failure = None
    initial_digests = set()
    for method, node in plan:
        row, error = _run_method(cfg, method, node or 3, advance_fn)
        report["methods"].append(row)
        initial_digests.add(row["identity"]["initial_state_sha256"])
        if error is not None:
            failure = error
            break
    report["physical_input_consistent"] = len(initial_digests) == 1
    report["all_initial_inputs_nontrivial"] = bool(
        report["methods"]
        and all(
            row["identity"]["electric_amplitude"] > 0.0
            and row["identity"]["nonuniform_amplitude"] > 0.0
            and row["identity"]["high_mode_sentinel"] > 0.0
            for row in report["methods"]
        )
    )
    if failure is not None:
        report["status"] = "failed"
        return report, failure
    if not report["physical_input_consistent"]:
        failure = Failure(
            "CostInput",
            "methods did not start from identical accepted physical inputs",
        )
        report["status"] = "failed"
        return report, failure
    report["status"] = "completed"
    return report, None


def _read_json(path):
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )


def run_paths(input_path, output_path):
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    report, failure = run(_read_json(input_path))
    periodic._atomic_json(output, report)
    return 0 if failure is None else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    try:
        return run_paths(arguments.input, arguments.output)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
