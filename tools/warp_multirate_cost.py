#!/usr/bin/env python3
"""Matched cost measurement for a lean synchronous multirate RK4 reference."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import resource
import shutil
import sys
import time
import traceback
from collections import namedtuple
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np


SOURCE_PROJECT = Path(
    os.environ.get("FOMHD_SOURCE_ROOT", str(Path(__file__).resolve().parents[1]))
).expanduser().resolve()
for path in (
    SOURCE_PROJECT / "src",
    SOURCE_PROJECT / "tests/warp",
    SOURCE_PROJECT / "tools",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import warp_multirate_ion as multirate


research = multirate.research
_OFFSETS = (0.0, 0.5, 0.5, 1.0)
LeanStage = namedtuple(
    "LeanStage",
    "window micro_step rk_stage time_s p z p_rhs z_rhs density_rate "
    "momentum_rate pressure_rate lorentz_rate electric magnetic",
)
LeanWindow = namedtuple("LeanWindow", "window p z stages counters")


def _copy(value):
    return value.copy()


def _sum_counters(windows):
    return {
        key: sum(int(window.counters.get(key, 0)) for window in windows)
        for key in research.Counters.__slots__
    }


def _lean_window(context, p0, z0, H, M, window, start_time):
    """Run four full adapter hydro evaluations per RK4 microstage."""
    if not math.isfinite(float(H)) or H <= 0 or type(M) is not int or M <= 0:
        raise ValueError("H must be positive/finite and M a positive integer")
    adapter, h, p, z, stages = (
        research.ReducedHydroAdapter(context),
        float(H) / M,
        _copy(p0),
        _copy(z0),
        [],
    )
    for micro in range(M):
        kp, kz = [], []
        for rk in range(4):
            if rk == 0:
                sp, sz = _copy(p), _copy(z)
            elif rk == 1:
                sp, sz = p + 0.5 * h * kp[0], z + 0.5 * h * kz[0]
            elif rk == 2:
                sp, sz = p + 0.5 * h * kp[1], z + 0.5 * h * kz[1]
            else:
                sp, sz = p + h * kp[2], z + h * kz[2]
            macro = adapter.macro(sp, sz)
            field = adapter.field(sp, sz)
            lorentz = adapter.lorentz(sp, field.electric, field.magnetic)
            p_rhs = macro.density_rate + macro.momentum_rate
            p_rhs = p_rhs + macro.pressure_rate + lorentz
            kp.append(_copy(p_rhs))
            kz.append(_copy(field.rhs))
            stages.append(
                LeanStage(
                    int(window), micro, rk,
                    float(start_time) + window * float(H)
                    + (micro + _OFFSETS[rk]) * h,
                    _copy(sp), _copy(sz), _copy(p_rhs), _copy(field.rhs),
                    _copy(macro.density_rate), _copy(macro.momentum_rate),
                    _copy(macro.pressure_rate), _copy(lorentz),
                    _copy(field.electric), _copy(field.magnetic),
                )
            )
        p = p + h * (kp[0] + 2 * kp[1] + 2 * kp[2] + kp[3]) / 6
        z = z + h * (kz[0] + 2 * kz[1] + 2 * kz[2] + kz[3]) / 6
    adapter.endpoint(p, z)
    return LeanWindow(window, p, z, tuple(stages), adapter.counters.snapshot())


def lean_run(context, p0, z0, *, H, M, macros=1, start_time=0.0):
    """Run the independent reference without accepted-state or ledger state."""
    if type(macros) is not int or macros <= 0:
        raise ValueError("macros must be a positive integer")
    p, z, windows = _copy(p0), _copy(z0), []
    for window in range(macros):
        result = _lean_window(context, p, z, H, M, window, start_time)
        windows.append(result)
        p, z = _copy(result.p), _copy(result.z)
    counters = MappingProxyType(_sum_counters(windows))
    if counters["full_hydro_eval"] != 4 * M * macros:
        raise RuntimeError("lean full-hydro count violates 4*M*macros")
    return SimpleNamespace(
        p=p, z=z, windows=tuple(windows),
        stages=tuple(stage for window in windows for stage in window.stages),
        counters=counters, actual_outer_sweeps=0,
        scope="independent lean synchronous RK4; no accepted transaction or ledger",
    )


def _run_lean_timed(context, p0, z0, h, config):
    def execute():
        local = multirate._isolated_context(context)
        return lean_run(
            local, p0, z0, H=float(h) * int(config["M"]),
            M=int(config["M"]), macros=int(config["macros"]),
            start_time=float(context.p.start_time),
        ).windows

    return multirate._timed(
        execute, context.compute.xp, int(config["warmups"]),
        int(config["repetitions"]), observe=_sum_counters,
    )


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _numpy(value):
    if type(value).__module__.split(".", 1)[0] == "cupy":
        import cupy

        return cupy.asnumpy(value)
    return np.asarray(value)


def _fingerprint(value):
    array = np.ascontiguousarray(_numpy(value))
    digest = hashlib.sha256()
    digest.update(f"{array.dtype}{array.shape}".encode())
    digest.update(array.tobytes())
    return {"sha256": digest.hexdigest(), "dtype": str(array.dtype),
            "shape": list(array.shape)}


def _compare(context, pa, za, pb, zb):
    scales = multirate._scales(context)
    p_raw, p_norm = multirate._relative_error(
        _numpy(pa), _numpy(pb), _numpy(scales.p)
    )
    z_raw, z_norm = multirate._relative_error(
        _numpy(za), _numpy(zb), _numpy(scales.z)
    )
    return {"p_max_abs": p_raw, "p_normalized": p_norm,
            "z_max_abs": z_raw, "z_normalized": z_norm}


def _phase(context, pa, za, pb, zb, mode, case):
    if case is None:
        return {"available": False, "reason": "closed RZ has no target phase"}
    left = multirate._mode_evidence(context, pa, za, mode, case)
    right = multirate._mode_evidence(context, pb, zb, mode, case)
    result = {"available": True, "mode": int(mode)}
    for name in ("velocity_mode", "magnetic_mode"):
        a, b = left[name], right[name]
        stem = name.removesuffix("_mode")
        result[f"{stem}_phase_error_rad"] = float(
            np.angle(np.exp(1j * (a["phase_rad"] - b["phase_rad"])))
        )
        result[f"{stem}_amplitude_relative"] = abs(
            a["amplitude"] / max(b["amplitude"], 1.0e-300) - 1
        )
    return result


def _diagnostic(context, p, z, name):
    multirate._synchronize(context.compute.xp)
    started = time.perf_counter()
    checks = list(multirate._stage_for(
        context, p, z, f"cost_diagnostic_{name}"
    ).checks)
    multirate._synchronize(context.compute.xp)
    return {
        "name": name, "elapsed_s": time.perf_counter() - started,
        "check_count": len(checks),
        "failed_hard": sum(
            bool(c.hard and (not c.available or not c.passed)) for c in checks
        ),
        "failed_observations": sum(
            bool(not c.hard and (not c.available or not c.passed)) for c in checks
        ),
        "scope": "outside matched timing; equal production-stage state checks",
    }


def _stack(stages, name):
    return np.stack([_numpy(getattr(stage, name)) for stage in stages])


def _lean_arrays(windows):
    stages = [stage for window in windows for stage in window.stages]
    names = (
        "p", "z", "p_rhs", "z_rhs", "density_rate", "momentum_rate",
        "pressure_rate", "lorentz_rate", "electric", "magnetic",
    )
    arrays = {f"lean_stage_{name}": _stack(stages, name) for name in names}
    arrays.update(
        lean_p=_numpy(windows[-1].p), lean_z=_numpy(windows[-1].z),
        lean_stage_window=np.array([s.window for s in stages], np.int32),
        lean_stage_micro=np.array([s.micro_step for s in stages], np.int32),
        lean_stage_rk=np.array([s.rk_stage for s in stages], np.int32),
        lean_stage_time_s=np.array([s.time_s for s in stages], np.float64),
    )
    return arrays


def _history_bytes(windows):
    fields = set(LeanStage._fields) - {"window", "micro_step", "rk_stage", "time_s"}
    return sum(
        int(getattr(stage, field).nbytes)
        for window in windows for stage in window.stages for field in fields
    )


def _historical(report):
    report = report or {}
    candidate = report.get("timing", {}).get("median_s")
    production = report.get("reference", {}).get("elapsed_s")
    return {
        "scope": "historical input-case report; not rerun by this utility",
        "candidate_median_s": candidate,
        "production_oracle_median_s": production,
        "candidate_to_production_ratio": (
            None if candidate is None or not production else candidate / production
        ),
        "speedup_claim": False,
    }


def _load_case(directory):
    raw = (directory / "input.json").read_bytes()
    config = json.loads(raw)
    required = (
        "case", "method", "points", "amplitude", "h", "M", "macros",
        "rtol", "max_iterations",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError("input.json missing controls: " + ", ".join(missing))
    report_path = directory / "report.json"
    return config, raw, (
        json.loads(report_path.read_text()) if report_path.is_file() else None
    )


def _memory(xp):
    if xp is np:
        return {"scope": "cpu", "used_bytes": None, "reserved_bytes": None}
    pool = xp.get_default_memory_pool()
    return {
        "scope": "cupy default memory pool",
        "used_bytes": int(pool.used_bytes()),
        "reserved_bytes": int(pool.total_bytes()),
        "limit_bytes": 8 << 30,
    }


def run(*, input_case_directory, device, output, warmups=1, repetitions=3):
    """Run one matched candidate/lean measurement into a new directory."""
    source_dir, output = (
        Path(input_case_directory).expanduser().resolve(),
        Path(output).expanduser().resolve(),
    )
    if device not in ("cpu", "cuda:0"):
        raise ValueError("device must be cpu or cuda:0")
    if device != "cpu" and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("CUDA measurement requires an owned Slurm allocation")
    if output.exists():
        raise ValueError("output must be a new directory")
    if type(warmups) is not int or not 0 <= warmups <= 2:
        raise ValueError("warmups must be an integer in [0, 2]")
    if type(repetitions) is not int or not 1 <= repetitions <= 3:
        raise ValueError("repetitions must be an integer in [1, 3]")

    source_config, source_bytes, source_report = _load_case(source_dir)
    config = copy.deepcopy(source_config)
    config.update(device=device, warmups=warmups, repetitions=repetitions)
    output.mkdir(parents=True)
    shutil.copyfile(source_dir / "input.json", output / "source_input.json")
    files = {"source_input.json": _sha256(output / "source_input.json")}
    files.update({
        str(path): _sha256(path)
        for name in ("input.json", "report.json", "arrays.npz", "provenance.json")
        if (path := source_dir / name).is_file()
    })
    started, xp = time.perf_counter(), None
    report = {"status": "failed", "files": files,
              "scope": "matched research-window cost; no acceleration claim"}
    try:
        context, p0, z0, case = multirate.build_case(config)
        xp = context.compute.xp
        if xp is not np:
            xp.get_default_memory_pool().set_limit(size=8 << 30)
        multirate._synchronize(xp)
        report["shared_fixture_setup_s"] = time.perf_counter() - started
        h, M, macros = (
            float(config["h"]), int(config["M"]), int(config["macros"])
        )
        actual = multirate._simulation_timing(
            config, start_time=float(context.p.start_time)
        )
        source_horizon = multirate._simulation_timing(source_config)[
            "total_duration_s"
        ]
        unchanged = all(
            config[key] == source_config[key]
            for key in source_config
            if key not in ("device", "warmups", "repetitions")
        )
        provenance = multirate._provenance(config)
        provenance.pop("declared_baseline", None)
        provenance["files"][str(Path(__file__).resolve())] = _sha256(Path(__file__))
        report.update(
            source_input={
                "path": str(source_dir / "input.json"),
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "config": source_config, "actual_config": config,
            },
            source_git=multirate._git_provenance(SOURCE_PROJECT),
            provenance=provenance,
            built_case_input={"p": _fingerprint(p0), "z": _fingerprint(z0)},
            protocol={
                "actual_h_s": h, "actual_H_s": h * M, "actual_M": M,
                "actual_macros": macros, "actual_horizon_s": h * M * macros,
                "source_horizon_s": source_horizon,
                "algorithm_controls_unchanged": unchanged,
                "horizon_unchanged": unchanged
                and source_horizon == h * M * macros,
                "device": device, "warmups": warmups,
                "repetitions": repetitions,
                "fresh_context_per_execution": True,
                "synchronized_timer_boundaries": True,
                "timed_scope": "fixture excluded; adapter, stages, coupling, "
                "and endpoint reconstruction included",
            },
            diagnostics={
                "pre": _diagnostic(context, p0, z0, "initial"), "post": {},
            },
        )
        candidate_windows, candidate_timing = multirate._run_candidates(
            context, p0, z0, h, config
        )
        lean_windows, lean_timing = _run_lean_timed(context, p0, z0, h, config)
        candidate, lean = candidate_windows[-1], lean_windows[-1]
        report["diagnostics"]["post"] = {
            "candidate": _diagnostic(
                context, candidate.p, candidate.z, "candidate_endpoint"
            ),
            "lean": _diagnostic(context, lean.p, lean.z, "lean_endpoint"),
        }
        candidate_counters = multirate._candidate_work(candidate_windows)
        lean_counters = _sum_counters(lean_windows)
        same_protocol = all(
            candidate_timing[key] == lean_timing[key]
            for key in ("warmups", "repetitions", "cold_executions")
        )
        expected_ids = {(micro, rk) for micro in range(M) for rk in range(4)}
        identities_match = all(
            {(s.micro_step, s.rk_stage) for s in window.stages} == expected_ids
            for window in lean_windows
        )
        expected_counts = {
            "full_hydro_eval": 4 * M * macros,
            "cheap_D": 4 * M * macros, "cheap_L": 4 * M * macros,
            "field_rhs": 4 * M * macros + macros,
            "endpoint_check": macros,
        }
        report.update(
            matched_cost={
                "candidate": candidate_timing, "lean_reference": lean_timing,
                "candidate_to_lean_median_ratio": candidate_timing["median_s"]
                / lean_timing["median_s"],
                "same_protocol": same_protocol, "speedup_claim": False,
                "scope": "same-GPU or same-CPU algorithm research-window cost",
            },
            existing_candidate_vs_production=_historical(source_report),
            counters={
                "candidate": candidate_counters, "lean": lean_counters,
                "candidate_actual_outer_sweeps": sum(
                    int(w.counters["actual_outer_calls"])
                    for w in candidate_windows
                ),
                "lean_actual_outer_sweeps": 0,
            },
            stage_invariants={
                "expected": expected_counts,
                "observed": {k: lean_counters[k] for k in expected_counts},
                "counts_match": all(
                    lean_counters[k] == value
                    for k, value in expected_counts.items()
                ),
                "all_four_rk_identities": identities_match,
                "stage_count": sum(len(w.stages) for w in lean_windows),
                "per_window_stage_counts": [len(w.stages) for w in lean_windows],
                "candidate_stage_split_terms": "not retained by the existing "
                "candidate; lean D/A_m/A_p/L/G terms are complete",
            },
        )
        pc, zc, pl, zl = map(_numpy, (candidate.p, candidate.z, lean.p, lean.z))
        checks = {"candidate_vs_lean": {
            **_compare(context, pc, zc, pl, zl),
            "phase": _phase(context, pc, zc, pl, zl, int(config["seed_mode"]), case),
        }}
        arrays = multirate._candidate_arrays(
            candidate_windows, context, start_time=actual["start_time_s"],
            macro_H=actual["macro_H_s"], microstep=actual["microstep_h_s"],
        )
        arrays.update(_lean_arrays(lean_windows))
        arrays.update(
            candidate_p=pc, candidate_z=zc,
            initial_p=_numpy(p0), initial_z=_numpy(z0),
            p_scales=_numpy(multirate._scales(context).p),
            z_scales=_numpy(multirate._scales(context).z),
        )
        checks["lean_vs_production_oracle"] = {
            "available": False, "reason": "input candidate output has no oracle arrays"
        }
        if (source_dir / "arrays.npz").is_file():
            with np.load(source_dir / "arrays.npz") as stored:
                if "oracle_p" in stored and "oracle_z" in stored:
                    op, oz = stored["oracle_p"], stored["oracle_z"]
                    checks["lean_vs_production_oracle"] = {
                        **_compare(context, pl, zl, op, oz),
                        "phase": _phase(
                            context, pl, zl, op, oz,
                            int(config["seed_mode"]), case,
                        ),
                    }
                    arrays.update(oracle_p=op, oracle_z=oz)
                if "candidate_p" in stored and "candidate_z" in stored:
                    checks["input_candidate_vs_fresh_candidate"] = _compare(
                        context, pc, zc, stored["candidate_p"], stored["candidate_z"]
                    )
        report.update(
            same_checks=checks,
            stage_history_bytes={
                "candidate": sum(
                    w.stage_history_bytes for w in candidate_windows
                ),
                "lean": _history_bytes(lean_windows),
                "written_scope": "one returned execution per algorithm",
            },
            limitations=[
                "research-window cost only; ratios are not acceleration claims",
                "no accepted transaction, ledger, restart, checkpoint, or FK solver",
                "candidate-vs-production timing is historical and separate",
                "parent owns the scientific gate and GPU deployment",
            ],
            array_bytes=sum(np.asarray(v).nbytes for v in arrays.values()),
        )
        np.savez_compressed(output / "arrays.npz", **arrays)
        report["status"] = "completed"
    except Exception as error:
        report.update(
            failure_code=getattr(error, "code", type(error).__name__),
            failure_message=str(error),
        )
        (output / "traceback.txt").write_text(traceback.format_exc())
        report["traceback_sha256"] = _sha256(output / "traceback.txt")
    finally:
        report["host_peak_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        report["device_memory"] = _memory(xp) if xp is not None else {
            "scope": "unavailable", "used_bytes": None, "reserved_bytes": None
        }
        report["total_elapsed_s"] = time.perf_counter() - started
        multirate._write_json(output / "report.json", report)
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-case-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    return parser


def main(argv=None):
    result = run(**vars(_parser().parse_args(argv)))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
