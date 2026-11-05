#!/usr/bin/env python3
"""Plot archived fixed-domain spatial, momentum, and eight-cycle mode evidence."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import textwrap

import numpy as np


SPATIAL_RUNS = ("spatial-pilot", "spatial-full", "spatial-n20",
                "spatial-disk-n24")
STATIC_RUNS = ("static-pilot", "static-full")
DYNAMIC_RUNS = ("dynamic-full", "dynamic-halfdt")
QUIET_RUNS = ("quiet-full", "quiet-refined")
MODE_RUNS = ("modes-n17", "modes-n33", "modes-n33-retry")
ALL_RUNS = SPATIAL_RUNS + STATIC_RUNS + DYNAMIC_RUNS + QUIET_RUNS + MODE_RUNS

EXPECTED_CASES = {
    "spatial-pilot": {
        f"{shape}_n{nodes}_{profile}"
        for nodes in (6, 8)
        for shape in ("rectangle", "disk")
        for profile in ("compact", "wall")
    },
    "spatial-full": {
        f"{shape}_n{nodes}_{profile}"
        for nodes in (12, 16)
        for shape in ("rectangle", "disk")
        for profile in ("compact", "wall")
    },
    "spatial-n20": {
        f"{shape}_n20_{profile}"
        for shape in ("rectangle", "disk")
        for profile in ("compact", "wall")
    },
    "spatial-disk-n24": {
        f"disk_n24_{profile}" for profile in ("compact", "wall")
    },
    "static-pilot": {
        f"{shape}-n{nodes}"
        for nodes in (6, 8)
        for shape in ("rectangle", "disk")
    },
    "static-full": {
        f"{shape}-n{nodes}"
        for nodes in (12, 16, 24)
        for shape in ("rectangle", "disk")
    },
    "dynamic-full": {
        f"rectangle-n{nodes}" for nodes in (3, 6, 9)
    },
    "dynamic-halfdt": {"rectangle-n6"},
    "quiet-full": {
        f"rectangle-n{nodes}" for nodes in (3, 6, 9)
    },
    "quiet-refined": {
        f"rectangle-n{nodes}" for nodes in (3, 6, 9)
    },
    "modes-n17": {
        "rk4-n640-a1e-07", "rk4-n1280-a1e-07"
    },
}

REPORT_KEYS = (
    "schema", "status", "backend_requested", "device", "execution_completed",
    "requested_case_count", "measured_case_count", "failed_case_count",
    "scientific_acceptance", "scope", "arguments", "software",
    "source_snapshot", "mode", "slurm_job_id", "hashes", "configuration",
    "points", "cycles", "integrators", "dt_guard", "failed_case", "failure",
    "passed",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _finite_number(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _close(left: float, right: float, label: str,
           rtol: float = 1e-11) -> None:
    if not math.isclose(float(left), float(right), rel_tol=rtol,
                        abs_tol=1e-30):
        raise ValueError(f"{label}: {left!r} != {right!r}")


def load_checked_npz(path: Path, expected_sha256: str | None = None) -> dict:
    """Load one NPZ from the audited bytes with pickle disabled."""
    data = path.read_bytes()
    actual = _sha256(data)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(
            f"NPZ SHA256 mismatch for {path}: {actual} != {expected_sha256}"
        )
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _all_finite(arrays: dict, label: str) -> None:
    for name, values in arrays.items():
        array = np.asarray(values)
        if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
            raise ValueError(f"{label}/{name} contains nonfinite values")


def _require_1d(arrays: dict, name: str, length: int | None = None) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"Spatial array is missing: {name}")
    values = np.asarray(arrays[name])
    if values.ndim != 1:
        raise ValueError(f"Spatial array {name} must be one-dimensional")
    if length is not None and len(values) != length:
        raise ValueError(
            f"Spatial array {name} length {len(values)} != {length}"
        )
    return values


def validate_spatial_case(case: dict, arrays: dict) -> dict:
    """Validate saved geometry and return only plotted spatial measurements."""
    _all_finite(arrays, str(case.get("label", "spatial")))
    geometry = case.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("Measured spatial case has no saved geometry")
    if case.get("status") != "measured":
        raise ValueError("Spatial success line requires measured status")
    if case.get("full_stage_hard_checks_pass") is not True:
        raise ValueError("Spatial success line requires hard checks to pass")

    cell_r = _require_1d(arrays, "cell_r_m")
    cell_z = _require_1d(arrays, "cell_z_m", len(cell_r))
    edge_r = _require_1d(arrays, "edge_r_m")
    _require_1d(arrays, "edge_z_m", len(edge_r))
    _require_1d(arrays, "actual_wall_E_slot", len(edge_r))
    cell_masks = {
        region: _require_1d(arrays, f"cell_region_{region}", len(cell_r))
        for region in ("all", "inner", "wall")
    }
    edge_masks = {
        region: _require_1d(arrays, f"edge_region_{region}", len(edge_r))
        for region in ("all", "inner", "wall")
    }
    if not np.all(cell_masks["all"]):
        raise ValueError("Saved active cell mask 'all' is not all true")
    if not np.all(edge_masks["all"]):
        raise ValueError("Saved active edge mask 'all' is not all true")
    if np.any(cell_masks["inner"] & cell_masks["wall"]):
        raise ValueError("Saved inner and wall cell regions overlap")
    if np.any(np.logical_not(cell_masks["inner"] | cell_masks["wall"])):
        raise ValueError("Saved inner/wall cell regions do not partition active cells")
    if np.any(edge_masks["inner"] & edge_masks["wall"]):
        raise ValueError("Saved inner and wall edge regions overlap")
    if np.any(np.logical_not(edge_masks["inner"] | edge_masks["wall"])):
        raise ValueError("Saved inner/wall edge regions do not partition active edges")

    nodes = int(case["nodes"])
    h_m = _finite_number(case["h_m"], "spatial h_m")
    if h_m <= 0:
        raise ValueError("Spatial h_m must be positive")
    if len(cell_r) != int(geometry["active_cells"]):
        raise ValueError("Saved cell count does not match geometry active_cells")
    if geometry.get("nodes") != nodes:
        raise ValueError("Geometry nodes do not match case nodes")
    if geometry.get("shape") != case.get("shape"):
        raise ValueError("Geometry shape does not match case shape")
    r_bounds = [float(value) for value in geometry["R_bounds_m"]]
    z_bounds = [float(value) for value in geometry["Z_bounds_m"]]
    if r_bounds != [1.0, 2.0] or z_bounds != [0.0, 1.0]:
        raise ValueError("Spatial evidence is not on the fixed [1,2]x[0,1] domain")
    _close(h_m, (r_bounds[1] - r_bounds[0]) / nodes, "R spacing")
    _close(h_m, (z_bounds[1] - z_bounds[0]) / nodes, "Z spacing")
    if np.any(cell_r < r_bounds[0] - 1e-14) or np.any(cell_r > r_bounds[1] + 1e-14):
        raise ValueError("Saved cell centers lie outside the fixed R bounds")
    if np.any(cell_z < z_bounds[0] - 1e-14) or np.any(cell_z > z_bounds[1] + 1e-14):
        raise ValueError("Saved cell centers lie outside the fixed Z bounds")
    if not math.isclose(float(geometry["inner_distance_m"]), 0.15,
                        rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Saved spatial wall band is not the fixed 0.15 m")

    target_area = _finite_number(
        geometry["analytic_target_area_m2"], "saved analytic target area"
    )
    saved_area = _finite_number(
        geometry["cell_mask_area_m2"], "saved cell mask area"
    )
    if target_area <= 0:
        raise ValueError("Saved analytic target area must be positive")
    computed_area = len(cell_r) * h_m * h_m
    _close(computed_area, saved_area, "active-cell area", rtol=1e-12)
    area_relative_error = (computed_area - target_area) / target_area

    ohm = case["ohm"]
    force = case["force"]
    for field in ("E", "E0"):
        unit = ohm[field]["all"].get("unit")
        if unit != "V/m":
            raise ValueError(f"{field} has unexpected unit {unit!r}")
    if force["projected_delta_force"]["all"].get("unit") != "Pa/m":
        raise ValueError("Projected pressure perturbation has unexpected unit")

    def finite_norm(group: dict, key: str, label: str) -> float:
        return _finite_number(group.get(key), label)

    e_all = ohm["E"]["all"]
    e0_all = ohm["E0"]["all"]
    correction_all = ohm["E_correction"]["all"]
    pressure_all = force["projected_delta_force"]["all"]
    row = {
        "run": case.get("_run"),
        "case": case.get("label"),
        "shape": case["shape"],
        "profile": case["profile"],
        "nodes": nodes,
        "h_m": h_m,
        "backend": case.get("_device", ""),
        "E_relative_weighted_l2": finite_norm(
            e_all, "relative_weighted_l2", "corrected E relative weighted L2"
        ),
        "E0_relative_weighted_l2": finite_norm(
            e0_all, "relative_weighted_l2", "E0 relative weighted L2"
        ),
        "E_weighted_rms_V_per_m": {
            region: finite_norm(
                ohm["E"][region], "weighted_rms",
                f"corrected E {region} weighted RMS"
            )
            for region in ("all", "inner", "wall")
        },
        "wall_correction_weighted_rms_V_per_m": finite_norm(
            correction_all, "weighted_rms", "wall correction weighted RMS"
        ),
        "projected_pressure_perturbation_relative_weighted_l2": finite_norm(
            pressure_all, "relative_weighted_l2",
            "projected pressure perturbation relative weighted L2"
        ),
        "units": {
            "E_relative_weighted_l2": "1",
            "E_weighted_rms_V_per_m": "V/m",
            "wall_correction_weighted_rms_V_per_m": "V/m",
            "projected_pressure_perturbation_relative_weighted_l2": "1",
        },
    }
    geometry_row = {
        "case": case.get("label"),
        "shape": case["shape"],
        "profile": case["profile"],
        "nodes": nodes,
        "h_m": h_m,
        "backend": case.get("_device", ""),
        "cells_r_m": [float(value) for value in cell_r],
        "cells_z_m": [float(value) for value in cell_z],
        "target_area_m2": target_area,
        "cell_mask_area_m2": saved_area,
        "area_relative_error": area_relative_error,
    }
    return {"spatial": row, "geometry": geometry_row}


def momentum_history(case: dict, arrays: dict) -> dict:
    """Validate dynamic counts and the signed cumulative momentum identity."""
    _all_finite(arrays, str(case.get("name", "dynamic")))
    steps = int(case["accepted_steps"])
    requested = int(case["requested_steps"])
    if not 0 <= steps <= requested or (
        steps != requested and case.get("status") != "failed"
    ):
        raise ValueError("Dynamic accepted/requested step count mismatch")
    time_s = np.asarray(arrays["time_s"], dtype=np.float64)
    momentum = np.asarray(arrays["momentum"])
    intervals = np.asarray(arrays["momentum_x_interval"], dtype=np.float64)
    if time_s.ndim != 1 or len(time_s) != steps + 1:
        raise ValueError("Dynamic time cadence does not match accepted steps")
    if momentum.ndim != 2 or momentum.shape != (steps + 1, 3):
        raise ValueError("Dynamic momentum array shape does not match steps")
    if intervals.shape != (steps,):
        raise ValueError("Dynamic momentum_x_interval shape does not match steps")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("Dynamic times must be strictly increasing")
    dt = _finite_number(case["dt_s"], "dynamic dt")
    _close(requested * dt, case["horizon_s"], "dynamic requested horizon")
    if not np.allclose(time_s, np.arange(steps + 1) * dt, rtol=1e-12, atol=0):
        raise ValueError("Dynamic cadence differs from reported dt")
    cumulative = np.cumsum(intervals)
    endpoint = float(cumulative[-1]) if steps else 0.0
    reported = _finite_number(
        case["signed_cumulative_momentum_x_defect"],
        "reported signed cumulative momentum_x defect",
    )
    _close(endpoint, reported, "dynamic cumulative momentum identity")
    return {
        "time_s": time_s[1:],
        "cumulative_momentum_x_kg_m_per_s": cumulative,
        "endpoint_momentum_x_kg_m_per_s": endpoint,
        "reported_endpoint_momentum_x_kg_m_per_s": reported,
    }


def validate_mode_case(case: dict, arrays: dict) -> dict:
    """Validate one hard-pass mode history before it can become a success line."""
    _all_finite(arrays, str(case.get("case_id", "mode")))
    if case.get("hard_pass") is not True or case.get("passed") is not True:
        raise ValueError("Mode success line requires passed and hard_pass")
    steps = int(case["steps"])
    accepted = int(case["accepted_steps"])
    if steps != accepted:
        raise ValueError("Mode accepted step count differs from requested steps")
    time_s = np.asarray(arrays["time_s"], dtype=np.float64)
    mode = np.asarray(arrays["mode"])
    if time_s.ndim != 1 or len(time_s) != steps + 1:
        raise ValueError("Mode time cadence must contain steps + 1 samples")
    if mode.shape != (steps + 1, 7):
        raise ValueError("Mode cadence must contain one 7-component row per time")
    if not np.all(np.diff(time_s) > 0):
        raise ValueError("Mode times must be strictly increasing")
    if not math.isclose(float(time_s[0]), 0.0, rel_tol=0.0, abs_tol=1e-30):
        raise ValueError("Mode history must begin at t=0")
    target = _finite_number(case["target_time_s"], "mode target horizon")
    accepted_time = _finite_number(
        case["accepted_time_s"], "mode accepted horizon"
    )
    dt_s = _finite_number(case["dt_s"], "mode dt")
    _close(time_s[-1], target, "mode history endpoint/target horizon")
    _close(accepted_time, target, "mode accepted/target horizon")
    _close(dt_s * steps, target, "mode dt*steps/target horizon")
    if not np.allclose(np.diff(time_s), dt_s, rtol=1e-12, atol=0.0):
        raise ValueError("Mode time array is not uniform with reported dt")
    fit = case.get("mode_fit")
    if not isinstance(fit, dict) or not fit.get("available"):
        raise ValueError("Mode fit is unavailable")
    if int(fit["sample_count"]) != len(time_s):
        raise ValueError("Mode fit sample count differs from cadence")
    if int(fit["interval_count"]) != steps:
        raise ValueError("Mode fit interval count differs from cadence")
    analytic = fit["analytic"]
    omega = _finite_number(
        analytic["angular_frequency_rad_s"], "mode analytic angular frequency"
    )
    _close(case["cycles"], 8.0, "eight cycles scope")
    _close(omega * target / (2 * math.pi), 8.0, "analytic cycles")
    density = mode[:, 3]
    if np.any(abs(density) == 0):
        raise ValueError("Mode history contains zero coefficients")
    phase = np.unwrap(np.angle(density))
    log_amplitude = np.log(abs(density))
    centered = time_s - np.mean(time_s)
    fit_measured = {}
    for values, key in ((phase, "angular_frequency_rad_s"),
                        (log_amplitude, "growth_rate_per_s")):
        slope = float(centered @ (values - np.mean(values)) / (centered @ centered))
        roundoff = 512 * np.finfo(float).eps * max(1., float(np.max(abs(values)))) / target
        if not math.isclose(slope, float(fit["measured"][key]),
                            rel_tol=1e-10, abs_tol=roundoff):
            raise ValueError(f"Mode history fit differs from report: {key}")
        fit_measured[key] = slope
    fit_measured["frequency_hz"] = fit_measured["angular_frequency_rad_s"] / (2 * math.pi)
    phase_residual = phase - (np.mean(phase) + fit_measured["angular_frequency_rad_s"] * centered)
    phase_rms = float(np.sqrt(np.mean(phase_residual ** 2)))
    if not math.isclose(phase_rms, float(fit["measured"]["phase_residual_rms_rad"]),
                        rel_tol=1e-10,
                        abs_tol=512 * np.finfo(float).eps * max(1., float(np.max(abs(phase))))):
        raise ValueError("Mode history phase fit residual differs from report")
    fit_measured["phase_residual_rms_rad"] = phase_rms
    _close(fit_measured["frequency_hz"], fit["measured"]["frequency_hz"],
           "mode history frequency fit")
    measured_cycles = abs(float(phase[-1] - phase[0])) / (2 * math.pi)
    _close(measured_cycles, fit["complete_measured_cycles"], "mode history cycles")
    if measured_cycles < 4 or steps / measured_cycles < 32:
        raise ValueError("Mode history has insufficient cycles or sampling")
    _close(steps / measured_cycles, fit["samples_per_cycle"], "mode history sampling")
    initial = mode[0, 3]
    continuum = np.real(initial * np.exp(1j * omega * time_s))
    normalized = mode[:, 3] / initial
    return {
        "time_s": time_s,
        "mode": mode,
        "continuum_density": continuum,
        "normalized_density": normalized,
        "refitted": fit_measured,
    }


def _scan_inputs(root: Path) -> dict:
    inputs = {}
    for run in ALL_RUNS:
        run_dir = root / run
        if not run_dir.is_dir():
            continue
        report = run_dir / "report.json"
        if report.is_file():
            inputs[_relative(root, report)] = _sha256(report.read_bytes())
        for path in sorted(run_dir.glob("*.npz")):
            inputs[_relative(root, path)] = _sha256(path.read_bytes())
    return inputs


def _verify_input_hashes(root: Path, inputs: dict) -> None:
    for relative, expected in inputs.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Evidence disappeared during collection: {relative}")
        if _sha256(path.read_bytes()) != expected:
            raise ValueError(f"Evidence changed during collection: {relative}")


def _read_report(root: Path, run: str) -> tuple[dict | None, str]:
    path = root / run / "report.json"
    if not path.is_file():
        return None, "report missing"
    try:
        report = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"report unavailable: {type(exc).__name__}: {exc}"
    if not isinstance(report, dict):
        return None, "report unavailable: top-level JSON is not an object"
    return report, ""


def _report_metadata(report: dict) -> dict:
    return {key: report[key] for key in REPORT_KEYS if key in report}


def _case_state(case: dict) -> str:
    status = str(case.get("status", "unknown"))
    if status in ("completed", "measured", "passed", "pass"):
        return "measured"
    if status in ("running", "pending", "incomplete"):
        return "incomplete"
    if status in ("skipped", "missing", "unavailable"):
        return status
    return "failed"


def _failure_summary(value) -> dict:
    if isinstance(value, dict):
        return {
            key: value[key]
            for key in ("code", "failure_code", "message", "error_type", "status")
            if key in value
        }
    if value:
        return {"message": str(value)}
    return {}


def _ledger_case(run: str, case: str, state: str, reason: str = "",
                 failure=None) -> dict:
    return {
        "run": run,
        "case": case,
        "state": state,
        "reason": reason,
        "failure": _failure_summary(failure),
    }


def _add_expected_missing(ledger_cases: list, run: str,
                          present: set[str]) -> None:
    for name in sorted(EXPECTED_CASES.get(run, set()) - present):
        ledger_cases.append(_ledger_case(
            run, name, "unavailable", "expected case missing from report"
        ))


def _run_state(report: dict | None, ledger_cases: list, run: str) -> str:
    if report is None:
        return "unavailable"
    status = str(report.get("status", "")).lower()
    if status in ("running", "pending", "incomplete"):
        return status
    if report.get("execution_completed") is False:
        return "incomplete"
    states = [row["state"] for row in ledger_cases if row["run"] == run]
    measured = any(state == "measured" for state in states)
    failed = any(state in ("failed", "unavailable", "skipped")
                 for state in states)
    if measured and failed:
        return "measured_with_failures"
    if measured:
        return "measured"
    if failed:
        return "failed"
    return status or "unknown"


def _spatial_failure_row(run: str, failure: dict, device: str) -> dict:
    arrays = failure.get("arrays", {})
    return {
        "run": run,
        "case": failure.get("case", "unknown"),
        "shape": str(failure.get("case", "")).split("_")[0],
        "nodes": int(failure.get("case", "n0").split("_n")[-1].split("_")[0])
        if "_n" in failure.get("case", "") else None,
        "state": "failed",
        "status": failure.get("status", "error"),
        "failure_code": failure.get("failure_code", ""),
        "message": failure.get("message", ""),
        "arrays_sha256": arrays.get("sha256"),
        "device": device,
    }


def _collect_spatial(root: Path, inputs: dict, reports: dict,
                     ledger_cases: list) -> dict:
    result = {"cases": [], "failures": [], "geometries": [],
              "suite_states": {}}
    for run in SPATIAL_RUNS:
        report, error = _read_report(root, run)
        if report is None:
            result["suite_states"][run] = "unavailable"
            reports[run] = {"state": "unavailable", "reason": error}
            _add_expected_missing(ledger_cases, run, set())
            continue
        reports[run] = {"state": "reported", **_report_metadata(report)}
        device = str(report.get("device", report.get("backend_requested", "")))
        present = set()
        reported_cases = report.get("cases", [])
        if not isinstance(reported_cases, list):
            raise ValueError(f"{run} spatial cases must be a list")
        for original in reported_cases:
            case = dict(original)
            case["_run"] = run
            case["_device"] = device
            label = str(case.get("label", "unknown"))
            present.add(label)
            state = _case_state(case)
            if state != "measured":
                ledger_cases.append(_ledger_case(
                    run, label, state, "case status is not measured",
                    case.get("failure"),
                ))
                result["failures"].append({
                    "run": run, "case": label, "state": state,
                    "shape": case.get("shape"), "nodes": case.get("nodes"),
                    "status": case.get("status"), "device": device,
                    "message": str(case.get("failure", "")),
                })
                continue
            if case.get("full_stage_hard_checks_pass") is not True:
                ledger_cases.append(_ledger_case(
                    run, label, "failed", "full-stage hard checks did not pass"
                ))
                result["failures"].append({
                    "run": run, "case": label, "state": "failed",
                    "shape": case.get("shape"), "nodes": case.get("nodes"),
                    "status": "hard_check_failed", "device": device,
                    "message": "full-stage hard checks did not pass",
                })
                continue
            arrays_name = case.get("arrays_file")
            if not isinstance(arrays_name, str) or Path(arrays_name).name != arrays_name:
                raise ValueError(f"{run}/{label} has an invalid sibling NPZ name")
            arrays_path = root / run / arrays_name
            if not arrays_path.is_file():
                ledger_cases.append(_ledger_case(
                    run, label, "unavailable", "arrays NPZ missing"
                ))
                result["failures"].append({
                    "run": run, "case": label, "state": "unavailable",
                    "shape": case.get("shape"), "nodes": case.get("nodes"),
                    "status": "missing_array", "device": device,
                    "message": "arrays NPZ missing",
                })
                continue
            arrays = load_checked_npz(arrays_path, case["arrays_sha256"])
            checked = validate_spatial_case(case, arrays)
            result["cases"].append(checked["spatial"])
            result["geometries"].append(checked["geometry"])
            ledger_cases.append(_ledger_case(
                run, label, "measured", "array hash, shape, and finite values audited"
            ))
        for failure in report.get("failures", []):
            row = _spatial_failure_row(run, failure, device)
            present.add(row["case"])
            result["failures"].append(row)
            ledger_cases.append(_ledger_case(
                run, row["case"], "failed", row["message"],
                {"failure_code": row["failure_code"], "status": row["status"]},
            ))
        _add_expected_missing(ledger_cases, run, present)
        result["suite_states"][run] = _run_state(report, ledger_cases, run)
        reports[run]["state"] = result["suite_states"][run]
    result["cases"].sort(key=lambda row: (row["shape"], row["profile"],
                                          row["nodes"]))
    return result


def _collect_static(root: Path, reports: dict,
                    ledger_cases: list) -> list:
    result = []
    for run in STATIC_RUNS:
        report, error = _read_report(root, run)
        if report is None:
            reports[run] = {"state": "unavailable", "reason": error}
            _add_expected_missing(ledger_cases, run, set())
            continue
        reports[run] = {"state": "reported", **_report_metadata(report)}
        device = str(report.get("device", ""))
        present = set()
        for case in report.get("cases", []):
            name = str(case.get("name", "unknown"))
            present.add(name)
            state = _case_state(case)
            if state != "measured":
                ledger_cases.append(_ledger_case(
                    run, name, state, "static case did not complete",
                    case.get("failure"),
                ))
                continue
            required = (
                "analytic_outward_x_N", "total_rate_N", "balance_defect_N",
                "h_m", "nodes", "shape",
            )
            if any(key not in case for key in required):
                ledger_cases.append(_ledger_case(
                    run, name, "unavailable", "static case fields missing"
                ))
                continue
            total_rate = np.asarray(case["total_rate_N"], dtype=float)
            balance = np.asarray(case["balance_defect_N"], dtype=float)
            if total_rate.shape != (3,) or balance.shape != (3,):
                raise ValueError(f"{run}/{name} static vectors must have length 3")
            if not np.all(np.isfinite(total_rate)) or not np.all(np.isfinite(balance)):
                raise ValueError(f"{run}/{name} static balance is nonfinite")
            result.append({
                "run": run,
                "case": name,
                "shape": case["shape"],
                "nodes": int(case["nodes"]),
                "h_m": _finite_number(case["h_m"], f"{run}/{name} h"),
                "device": device,
                "analytic_outward_x_N": _finite_number(
                    case["analytic_outward_x_N"],
                    f"{run}/{name} analytic outward traction",
                ),
                "total_rate_x_N": float(total_rate[0]),
                "balance_defect_linf_N": float(np.max(np.abs(balance))),
            })
            ledger_cases.append(_ledger_case(
                run, name, "measured", "completed static CUDA case"
            ))
        _add_expected_missing(ledger_cases, run, present)
        reports[run]["state"] = _run_state(report, ledger_cases, run)
    result.sort(key=lambda row: (row["shape"], -row["h_m"]))
    return result


def _dynamic_device(report: dict) -> str:
    return str(report.get("device", "cpu"))


def _collect_dynamic_run(root: Path, run: str, reports: dict,
                         ledger_cases: list, target: list) -> None:
    report, error = _read_report(root, run)
    if report is None:
        reports[run] = {"state": "unavailable", "reason": error}
        _add_expected_missing(ledger_cases, run, set())
        return
    reports[run] = {"state": "reported", **_report_metadata(report)}
    device = _dynamic_device(report)
    present = set()
    for case in report.get("cases", []):
        name = str(case.get("name", "unknown"))
        present.add(name)
        state = _case_state(case)
        if state != "measured":
            ledger_cases.append(_ledger_case(
                run, name, state,
                f"rejected after {case.get('accepted_steps', 0)}/{case.get('requested_steps', 0)} steps",
                case.get("failure"),
            ))
        if state not in ("measured", "failed") or "accepted_steps" not in case:
            continue
        path = root / run / f"{name}.npz"
        if not path.is_file():
            ledger_cases.append(_ledger_case(
                run, name, "unavailable", "dynamic trajectory NPZ missing"
            ))
            continue
        arrays = load_checked_npz(path)
        history = momentum_history(case, arrays)
        row = {
            "run": run,
            "kind": "quiet" if bool(case.get("quiet")) else "dynamic",
            "completed": state == "measured",
            "failure": case.get("failure"),
            "endpoint_semidiscrete_x_N": _finite_number(
                case["final_balance"]["balance_defect_N"][0], "semidiscrete defect"),
            "case": name,
            "shape": case.get("shape", "rectangle"),
            "nodes": int(case["nodes"]),
            "h_m": _finite_number(case["h_m"], f"{run}/{name} h"),
            "device": device,
            "requested_steps": int(case["requested_steps"]),
            "accepted_steps": int(case["accepted_steps"]),
            "dt_s": _finite_number(case["dt_s"], f"{run}/{name} dt"),
            "horizon_s": _finite_number(case["horizon_s"], f"{run}/{name} horizon"),
            "time_s": [float(value) for value in history["time_s"]],
            "cumulative_momentum_x_kg_m_per_s": [
                float(value)
                for value in history["cumulative_momentum_x_kg_m_per_s"]
            ],
            "endpoint_momentum_x_kg_m_per_s":
                history["endpoint_momentum_x_kg_m_per_s"],
            "field_drift_over_declared_scale": {
                key: _finite_number(value, f"{run}/{name} drift {key}")
                for key, value in case.get(
                    "field_drift_over_declared_scale", {}
                ).items()
            },
            "initial_total_rate_N": [
                float(value)
                for value in case.get("initial_balance", {}).get(
                    "total_rate_N", []
                )
            ],
        }
        target.append(row)
        if state == "measured":
            ledger_cases.append(_ledger_case(
                run, name, "measured",
                "dynamic cadence and cumulative momentum identity audited",
            ))
    _add_expected_missing(ledger_cases, run, present)
    reports[run]["state"] = _run_state(report, ledger_cases, run)


def _collect_dynamic(root: Path, reports: dict, ledger_cases: list) -> dict:
    rows = []
    for run in DYNAMIC_RUNS + QUIET_RUNS:
        _collect_dynamic_run(root, run, reports, ledger_cases, rows)
    return {"dynamic": [row for row in rows if row["kind"] == "dynamic"],
            "quiet": [row for row in rows if row["kind"] == "quiet"]}


def _collect_mode_run(root: Path, run: str, reports: dict,
                      ledger_cases: list, result: list) -> None:
    report, error = _read_report(root, run)
    if report is None:
        reports[run] = {"state": "unavailable", "reason": error}
        _add_expected_missing(ledger_cases, run, set())
        return
    reports[run] = {"state": "reported", **_report_metadata(report)}
    present = set()
    configuration = report.get("configuration", {})
    suite_points = int(report.get("points", configuration.get("points", 0)))
    suite_cycles = float(report.get("cycles", configuration.get("cycles", 0.0)))
    for case in report.get("cases", []):
        case_id = str(case.get("case_id", "unknown"))
        present.add(case_id)
        if (case.get("hard_pass") is not True
                or case.get("passed") is not True):
            ledger_cases.append(_ledger_case(
                run, case_id, "failed",
                "mode case did not pass hard checks", case.get("failure"),
            ))
            continue
        if int(case.get("points", -1)) != suite_points:
            raise ValueError(f"{run}/{case_id} points differ from report points")
        if not math.isclose(float(case.get("cycles", -1.0)), suite_cycles,
                            rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"{run}/{case_id} cycles differ from report cycles")
        arrays_name = case.get("arrays_file", f"{case_id}.npz")
        if not isinstance(arrays_name, str) or Path(arrays_name).name != arrays_name:
            raise ValueError(f"{run}/{case_id} has an invalid NPZ name")
        array_path = root / run / arrays_name
        if not array_path.is_file():
            ledger_cases.append(_ledger_case(
                run, case_id, "unavailable", "mode history NPZ missing"
            ))
            continue
        arrays = load_checked_npz(array_path, case.get("arrays_sha256"))
        checked = validate_mode_case(case, arrays)
        fit = case["mode_fit"]
        measured = {**fit["measured"], **checked["refitted"]}
        rk4 = fit["rk4_predicted"]
        analytic = fit["analytic"]
        steps = int(case["steps"])
        cycles = float(case["cycles"])
        time_cycles = checked["time_s"] / checked["time_s"][-1] * cycles
        density = checked["mode"][:, 3]
        normalized = checked["normalized_density"]
        row = {
            "run": run,
            "case": case_id,
            "points": suite_points,
            "cycles": cycles,
            "steps": steps,
            "accepted_steps": int(case["accepted_steps"]),
            "dt_s": _finite_number(case["dt_s"], f"{run}/{case_id} dt"),
            "target_time_s": _finite_number(
                case["target_time_s"], f"{run}/{case_id} target time"
            ),
            "relative_amplitude": _finite_number(
                case["relative_amplitude"], f"{run}/{case_id} amplitude"
            ),
            "device": str(case.get("device", "")),
            "integrator": str(case["integrator"]),
            "time_cycles": [float(value) for value in time_cycles],
            "actual_density_real": [
                float(value) for value in np.real(density)
            ],
            "continuum_density_real": [
                float(value) for value in checked["continuum_density"]
            ],
            "normalized_density_real": [
                float(value) for value in np.real(normalized)
            ],
            "normalized_density_imag": [
                float(value) for value in np.imag(normalized)
            ],
            "measured_frequency_Hz": _finite_number(
                measured["frequency_hz"], f"{run}/{case_id} measured frequency"
            ),
            "analytic_frequency_Hz": _finite_number(
                analytic["frequency_hz"], f"{run}/{case_id} analytic frequency"
            ),
            "rk4_predicted_frequency_Hz": _finite_number(
                rk4["frequency_hz"], f"{run}/{case_id} RK4 frequency"
            ),
            "measured_growth_rate_per_s": _finite_number(
                measured["growth_rate_per_s"],
                f"{run}/{case_id} measured damping",
            ),
            "rk4_predicted_growth_rate_per_s": _finite_number(
                rk4["growth_rate_per_s"],
                f"{run}/{case_id} RK4 damping",
            ),
            "continuum_growth_rate_per_s": _finite_number(
                analytic["growth_rate_per_s"],
                f"{run}/{case_id} continuum damping",
            ),
            "phase_residual_rms_rad": _finite_number(
                measured["phase_residual_rms_rad"],
                f"{run}/{case_id} phase residual",
            ),
            "complex_relative_rms_residual": _finite_number(
                measured["complex_relative_rms_residual"],
                f"{run}/{case_id} complex residual",
            ),
            "phase_coherence": _finite_number(
                measured["phase_coherence"], f"{run}/{case_id} coherence"
            ),
            "samples_per_cycle": _finite_number(
                fit["samples_per_cycle"], f"{run}/{case_id} samples per cycle"
            ),
            "complete_measured_cycles": _finite_number(
                fit["complete_measured_cycles"],
                f"{run}/{case_id} measured cycles",
            ),
        }
        row["frequency_relative_error"] = abs(
            row["measured_frequency_Hz"] - row["analytic_frequency_Hz"]
        ) / row["analytic_frequency_Hz"]
        result.append(row)
        ledger_cases.append(_ledger_case(
            run, case_id, "measured",
            "mode cadence, endpoint horizon, fit, and hard_pass audited",
        ))
    failed_case = report.get("failed_case")
    if isinstance(failed_case, str) and failed_case not in present:
        present.add(failed_case)
        ledger_cases.append(_ledger_case(
            run, failed_case, "failed", "report-level mode case failed",
            report.get("failure"),
        ))
    _add_expected_missing(ledger_cases, run, present)
    reports[run]["state"] = _run_state(report, ledger_cases, run)


def _mode_reconstruction(row: dict) -> dict:
    wavelength = np.linspace(0.0, 1.0, 161)
    time_indices = np.unique(np.linspace(
        0, len(row["time_cycles"]) - 1,
        min(161, len(row["time_cycles"])), dtype=int,
    ))
    complex_history = np.asarray(row["normalized_density_real"]) + 1j * np.asarray(
        row["normalized_density_imag"]
    )
    angle = 2 * np.pi * wavelength
    field = np.real(
        complex_history[time_indices, None] * np.exp(1j * angle)[None, :]
    )
    return {
        "points": row["points"],
        "case": row["case"],
        "wavelength_fraction": [float(value) for value in wavelength],
        "time_cycles": [
            float(value) for value in np.asarray(row["time_cycles"])[time_indices]
        ],
        "normalized_real_density": [
            [float(value) for value in line] for line in field
        ],
        "label": "measured-coefficient reconstruction over one periodic wavelength",
    }


def _mode_amplitude_controls(rows: list) -> list:
    controls = []
    for points in sorted({row["points"] for row in rows}):
        point_rows = [row for row in rows if row["points"] == points]
        for steps in sorted({row["steps"] for row in point_rows}, reverse=True):
            candidates = [row for row in point_rows if row["steps"] == steps]
            if len(candidates) < 2:
                continue
            ordered = sorted(candidates, key=lambda row: row["relative_amplitude"])
            half, full = ordered[0], ordered[-1]
            if not math.isclose(
                half["relative_amplitude"] / full["relative_amplitude"],
                0.5, rel_tol=1e-6, abs_tol=0.0,
            ):
                continue
            if (len(half["time_cycles"]) != len(full["time_cycles"])
                    or not np.allclose(
                        half["time_cycles"], full["time_cycles"],
                        rtol=1e-12, atol=0.0,
                    )):
                raise ValueError("Full/half mode histories have different cadence")
            half_norm = (
                np.asarray(half["normalized_density_real"])
                + 1j * np.asarray(half["normalized_density_imag"])
            )
            full_norm = (
                np.asarray(full["normalized_density_real"])
                + 1j * np.asarray(full["normalized_density_imag"])
            )
            difference = np.abs(full_norm - half_norm)
            controls.append({
                "points": points,
                "steps": steps,
                "full_case": full["case"],
                "half_case": half["case"],
                "time_cycles": full["time_cycles"],
                "normalized_history_difference": [
                    float(value) for value in difference
                ],
                "maximum_normalized_history_difference": float(
                    np.max(difference)
                ),
            })
    return controls


def _collect_modes(root: Path, reports: dict,
                   ledger_cases: list) -> dict:
    rows = []
    for run in MODE_RUNS:
        _collect_mode_run(root, run, reports, ledger_cases, rows)
    rows.sort(key=lambda row: (row["points"], -row["dt_s"]))
    reconstructions = [
        _mode_reconstruction(row)
        for row in rows
        if row["steps"] == max(
            candidate["steps"] for candidate in rows
            if candidate["points"] == row["points"]
        )
    ]
    devices = sorted({row["device"] for row in rows if row["device"]})
    suite_states = {
        run: reports.get(run, {}).get("state", "unavailable")
        for run in MODE_RUNS
    }
    return {
        "cases": rows,
        "reconstructions": reconstructions,
        "amplitude_controls": _mode_amplitude_controls(rows),
        "devices": devices,
        "suite_states": suite_states,
        "available": bool(rows),
    }


def collect(root: Path) -> dict:
    """Read, audit, and summarize all bounded archived evidence."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Evidence root does not exist: {root}")
    inputs = _scan_inputs(root)
    reports = {}
    ledger_cases = []
    spatial = _collect_spatial(root, inputs, reports, ledger_cases)
    static = _collect_static(root, reports, ledger_cases)
    dynamic = _collect_dynamic(root, reports, ledger_cases)
    modes = _collect_modes(root, reports, ledger_cases)
    _verify_input_hashes(root, inputs)

    run_states = {}
    for run in ALL_RUNS:
        rows = [row for row in ledger_cases if row["run"] == run]
        if run in reports:
            state = reports[run].get("state", "reported")
        else:
            state = "unavailable"
        reasons = {
            row["reason"] for row in rows
            if row["reason"] and row["state"] != "measured"
        }
        if not (root / run / "report.json").is_file():
            reasons.add("report missing")
        report_failure = reports.get(run, {}).get("failure")
        if isinstance(report_failure, dict):
            reasons.add(
                f"{report_failure.get('code', 'report failure')}: "
                f"{report_failure.get('message', '')}"
            )
        reasons = sorted(reasons)
        run_states[run] = {
            "state": state,
            "report_present": (root / run / "report.json").is_file(),
            "reason": "; ".join(reasons),
            "case_count": len(rows),
        }
    return {
        "root": str(root),
        "inputs": inputs,
        "reports": reports,
        "ledger": {
            "runs": run_states,
            "cases": ledger_cases,
        },
        "spatial": spatial,
        "static": static,
        "dynamic": dynamic["dynamic"],
        "quiet": dynamic["quiet"],
        "modes": modes,
        "units": {
            "spatial_E_relative_weighted_l2": "1",
            "spatial_E_weighted_rms": "V/m",
            "spatial_pressure_perturbation_relative_weighted_l2": "1",
            "static_force": "N",
            "dynamic_momentum": "kg*m/s",
            "mode_frequency": "Hz",
            "mode_growth_rate": "1/s",
            "mode_phase_residual": "rad",
        },
    }


def _save_figure(fig, output: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=180,
                    bbox_inches="tight")


def _shape_style(shape: str) -> tuple[str, str]:
    return ({"rectangle": "C0", "disk": "C1"}.get(shape, "C2"),
            {"rectangle": "o", "disk": "s"}.get(shape, "."))


def _render_geometry(data: dict, output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Patch, Rectangle

    geometries = data["spatial"]["geometries"]
    selected = {}
    for shape in ("rectangle", "disk"):
        shape_rows = [row for row in geometries if row["shape"] == shape]
        if not shape_rows:
            continue
        n8 = next((row for row in shape_rows if row["nodes"] == 8), None)
        largest = max(shape_rows, key=lambda row: row["nodes"])
        selected[shape] = [row for row in (n8, largest) if row is not None]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for row_index, shape in enumerate(("rectangle", "disk")):
        for col_index in range(2):
            axis = axes[row_index, col_index]
            shape_rows = selected.get(shape, [])
            if col_index >= len(shape_rows):
                axis.text(0.5, 0.5, "unavailable", ha="center", va="center",
                          transform=axis.transAxes)
                axis.set_axis_off()
                continue
            geometry = shape_rows[col_index]
            r_values = np.asarray(geometry["cells_r_m"])
            z_values = np.asarray(geometry["cells_z_m"])
            h_m = geometry["h_m"]
            patches = [
                Rectangle((r_value - h_m / 2, z_value - h_m / 2), h_m, h_m)
                for r_value, z_value in zip(r_values, z_values)
            ]
            axis.add_collection(PatchCollection(
                patches, facecolor="C0", edgecolor="white", linewidth=0.25,
                alpha=0.75, label="saved active cell centers",
            ))
            theta = np.linspace(0.0, 2 * np.pi, 129)
            if shape == "rectangle":
                axis.plot([1, 2, 2, 1, 1], [0, 0, 1, 1, 0], "k-",
                          linewidth=1.4, label="fixed rectangle boundary")
            else:
                polygon_r = 1.5 + 0.45 * np.cos(theta)
                polygon_z = 0.5 + 0.45 * np.sin(theta)
                axis.plot(polygon_r, polygon_z, "k-", linewidth=1.4,
                          label="fixed 128-gon")
                circle = np.linspace(0.0, 2 * np.pi, 721)
                axis.plot(1.5 + 0.45 * np.cos(circle),
                          0.5 + 0.45 * np.sin(circle), "k:", linewidth=1.2,
                          label="true circle r=0.45 m")
            axis.set(
                xlim=(0.98, 2.02), ylim=(-0.02, 1.02),
                xlabel="R (m)", ylabel="Z (m)", aspect="equal",
                title=f"{shape.title()} n={geometry['nodes']} | "
                      f"CUDA {geometry['backend']}",
            )
            axis.text(
                0.02, 0.98,
                f"h = {h_m:.5g} m\n"
                f"area relative error = {geometry['area_relative_error']:.3e}",
                transform=axis.transAxes, ha="left", va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.7"},
            )
            axis.legend(loc="lower right", fontsize=9, framealpha=0.9)
            axis.grid(alpha=0.2)
    failed_rectangle = [
        row for row in data["spatial"]["failures"]
        if row.get("shape") == "rectangle" and row.get("nodes") == 24
    ]
    failure_note = "rectangle n24: unavailable"
    if failed_rectangle:
        failure_note += f" ({failed_rectangle[0].get('failure_code')}: " \
                        f"{failed_rectangle[0].get('message', '')})"
    fig.suptitle(
        "Fixed-domain masks on saved production cells "
        f"({', '.join(sorted({row['backend'] for rows in selected.values() for row in rows}))})",
        fontsize=13,
    )
    fig.text(0.5, -0.02, textwrap.fill(failure_note, width=110),
             ha="center", va="top", fontsize=9, color="C3")
    _save_figure(fig, output, "fixed-geometry")
    plt.close(fig)


def _spatial_series(cases: list, shape: str, profile: str,
                    key: str) -> tuple[np.ndarray, np.ndarray]:
    rows = sorted(
        [row for row in cases
         if row["shape"] == shape and row["profile"] == profile],
        key=lambda row: row["h_m"], reverse=True,
    )
    return (np.asarray([row["h_m"] for row in rows]),
            np.asarray([row[key] for row in rows]))


def _render_spatial(data: dict, output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    cases = data["spatial"]["cases"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for row_index, profile in enumerate(("compact", "wall")):
        for shape in ("rectangle", "disk"):
            color, marker = _shape_style(shape)
            h_values, corrected = _spatial_series(
                cases, shape, profile, "E_relative_weighted_l2"
            )
            _, e0 = _spatial_series(
                cases, shape, profile, "E0_relative_weighted_l2"
            )
            axes[row_index, 0].loglog(
                h_values, corrected, marker=marker, color=color,
                linewidth=1.7, label=f"{shape}: corrected E",
            )
            axes[row_index, 0].loglog(
                h_values, e0, ":", marker=marker, color=color,
                linewidth=1.3, label=f"{shape}: E0",
            )

            for region, linestyle in (
                ("all", "-"), ("inner", "--"), ("wall", ":"),
            ):
                regional = [
                    row["E_weighted_rms_V_per_m"][region]
                    for row in sorted(
                        [item for item in cases
                         if item["shape"] == shape
                         and item["profile"] == profile],
                        key=lambda item: item["h_m"], reverse=True,
                    )
                ]
                axes[row_index, 1].semilogx(
                    h_values, regional, marker=marker, linestyle=linestyle,
                    color=color, label=f"{shape} {region}",
                )
            correction = _spatial_series(
                cases, shape, profile,
                "wall_correction_weighted_rms_V_per_m",
            )
            axes[row_index, 1].semilogx(
                correction[0], correction[1], marker=marker,
                linestyle="-.", color=color, alpha=0.65,
                label=f"{shape} wall correction",
            )

            pressure = _spatial_series(
                cases, shape, profile,
                "projected_pressure_perturbation_relative_weighted_l2",
            )
            axes[row_index, 2].loglog(
                pressure[0], pressure[1], marker=marker, color=color,
                linewidth=1.7, label=shape,
            )

        axis0, axis1, axis2 = axes[row_index]
        axis0.set(
            xlabel="h (m)", ylabel="Relative weighted L2 (dimensionless)",
            title=f"{profile.title()}: corrected E and E0",
        )
        axis1.set(
            xlabel="h (m)", ylabel="Weighted RMS (V/m)",
            title=f"{profile.title()}: regional E RMS and wall correction",
        )
        axis2.set(
            xlabel="h (m)", ylabel="Relative weighted L2 (dimensionless)",
            title=f"{profile.title()}: projected pressure perturbation",
        )
        for axis in (axis0, axis1, axis2):
            axis.invert_xaxis()
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.grid(True, which="both", alpha=0.2)
            axis.legend(fontsize=9)
    devices = sorted({row["backend"] for row in cases})
    fig.suptitle(
        f"Spatial continuum comparisons ({', '.join(devices)}): "
        "fixed 0.15 m wall band",
        fontsize=13,
    )
    fig.text(
        0.5, -0.02,
        "No order or threshold acceptance is inferred. Slopes are descriptive; "
        "compact profiles remain underresolved at coarse n.\n"
        "Raw-full boundary lift is not a pointwise force-error measurement.\n"
        + textwrap.fill("Omitted failed cases: " + "; ".join(
            f"{row['case']} ({row.get('failure_code', row.get('status', 'unavailable'))})"
            for row in data["spatial"]["failures"]), width=140),
        ha="center", va="top", fontsize=9,
    )
    _save_figure(fig, output, "spatial-continuum")
    plt.close(fig)


def _plot_momentum_history(axis, row: dict, label: str,
                           linestyle: str = "-") -> None:
    axis.plot(
        np.asarray(row["time_s"]) * 1e9,
        row["cumulative_momentum_x_kg_m_per_s"],
        linestyle=linestyle, marker=".", markersize=2.2, linewidth=1.3,
        label=label,
    )


def _render_momentum(data: dict, output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    static_rows = data["static"]
    dynamic_rows = data["dynamic"]
    quiet_rows = data["quiet"]

    for shape in ("rectangle", "disk"):
        rows = sorted([row for row in static_rows if row["shape"] == shape],
                      key=lambda row: row["h_m"], reverse=True)
        if not rows:
            continue
        color, marker = _shape_style(shape)
        h_values = [row["h_m"] for row in rows]
        axes[0, 0].loglog(
            h_values, [row["analytic_outward_x_N"] for row in rows],
            marker=marker, color=color, label=f"{shape} analytic traction",
        )
        axes[0, 0].loglog(
            h_values, [abs(row["total_rate_x_N"]) for row in rows],
            linestyle=":", marker=marker, color=color,
            label=f"{shape} |discrete rate_x|",
        )
        axes[0, 1].loglog(
            h_values, [row["balance_defect_linf_N"] for row in rows],
            marker=marker, color=color, label=shape,
        )
    axes[0, 0].set(
        xlabel="h (m)", ylabel="Force (N)",
        title="Static CUDA: analytic traction and discrete rate",
    )
    axes[0, 1].set(
        xlabel="h (m)", ylabel="max component residual (N)",
        title="Static CUDA: Cartesian balance residual",
    )

    for row in dynamic_rows:
        _plot_momentum_history(
            axes[0, 2], row,
            f"{row['case']} ({row['accepted_steps']} steps)",
        )
    axes[0, 2].set(
        xlabel="Accepted interval end time (ns)",
        ylabel="Signed cumulative defect (kg m/s)",
        title="Dynamic CPU: signed momentum_x interval sum",
    )
    for row in sorted(dynamic_rows, key=lambda item: item["nodes"]):
        axes[1, 0].loglog(
            row["h_m"], abs(row["endpoint_semidiscrete_x_N"]),
            marker="o", label=f"n={row['nodes']}, {row['accepted_steps']} steps",
        )
    axes[1, 0].set(
        xlabel="h (m)", ylabel="|endpoint semidiscrete force defect| (N)",
        title="Dynamic CPU: endpoint semidiscrete defect",
    )

    n6_rows = [
        row for row in dynamic_rows
        if row["nodes"] == 6 and row["shape"] == "rectangle"
    ]
    for row in sorted(n6_rows, key=lambda item: item["accepted_steps"]):
        _plot_momentum_history(
            axes[1, 1], row,
            f"{row['run']}: {row['accepted_steps']} steps",
            "--" if row["accepted_steps"] == 640 else "-",
        )
    axes[1, 1].set(
        xlabel="Accepted interval end time (ns)",
        ylabel="Signed cumulative defect (kg m/s)",
        title="CPU n6 time-step control: 320 versus 640",
    )

    for row in sorted(quiet_rows, key=lambda item: item["nodes"]):
        drift = row["field_drift_over_declared_scale"]
        if drift:
            axes[1, 2].loglog(
                row["h_m"], max(float(value) for value in drift.values()),
                marker="o" if row["completed"] else "x",
                label=("refined" if row["run"] == "quiet-refined" else "original")
                      + f" n={row['nodes']}, {row['accepted_steps']}/{row['requested_steps']}"
                      + (" complete" if row["completed"] else " rejected"),
            )
    axes[1, 2].set(
        xlabel="h (m)", ylabel="max drift / declared physical scale",
        title="CPU quiet field: original / refined projection",
    )

    for axis in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 2]):
        axis.invert_xaxis()
        axis.xaxis.set_minor_formatter(NullFormatter())
    for axis in axes.flat:
        axis.grid(True, which="both", alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=9)
    if not quiet_rows:
        axes[1, 2].text(
            0.5, 0.5, "quiet-full unavailable\n(no completed drift case)",
            transform=axes[1, 2].transAxes, ha="center", va="center",
            fontsize=10,
        )
    if not any(row["accepted_steps"] == 640 for row in n6_rows):
        axes[1, 1].text(
            0.98, 0.04, "dynamic-halfdt n6 unavailable",
            transform=axes[1, 1].transAxes, ha="right", va="bottom",
            fontsize=9, color="C3",
        )
    dynamic_n9_missing = any(
        row["run"] == "dynamic-full"
        and row["case"] == "rectangle-n9"
        and row["state"] == "unavailable"
        for row in data["ledger"]["cases"]
    )
    if dynamic_n9_missing:
        axes[1, 0].text(
            0.02, 0.04, "rectangle n9 unavailable",
            transform=axes[1, 0].transAxes, ha="left", va="bottom",
            fontsize=9, color="C3",
        )
    static_devices = sorted({row["device"] for row in static_rows})
    dynamic_devices = sorted({row["device"] for row in dynamic_rows + quiet_rows})
    fig.suptitle(
        "Momentum controls: static "
        f"{', '.join(static_devices) or 'unavailable'} and dynamic "
        f"{', '.join(dynamic_devices) or 'unavailable'}",
        fontsize=13,
    )
    static_n24 = sorted({
        row["case"] for row in data["ledger"]["cases"]
        if row["run"] == "static-full"
        and row["case"] in ("rectangle-n24", "disk-n24")
        and row["state"] == "failed"
    })
    n24_note = (
        f" Static n24 failures: {', '.join(static_n24)}."
        if static_n24 else ""
    )
    fig.text(
        0.5, -0.02,
        "momentum_x = p_R cos(phi) - p_phi sin(phi). Numerical wall reaction is "
        "not added to physical outward traction.\n"
        "Smaller defects need not be monotonic; no new acceptance threshold is applied."
        + n24_note + "\nQuiet rejections: " + "; ".join(
            f"{row['run']} n={row['nodes']} {row['accepted_steps']}/{row['requested_steps']} "
            f"({(row['failure'] or {}).get('code', 'unknown')})"
            for row in quiet_rows if not row["completed"]),
        ha="center", va="top", fontsize=9,
    )
    _save_figure(fig, output, "momentum-controls")
    plt.close(fig)


def _render_modes(data: dict, output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    rows = data["modes"]["cases"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    full_amplitude = {
        points: max(
            row["relative_amplitude"] for row in rows
            if row["points"] == points
        )
        for points in {row["points"] for row in rows}
    }
    plotted_rows = [
        row for row in rows
        if math.isclose(
            row["relative_amplitude"], full_amplitude[row["points"]],
            rel_tol=1e-12, abs_tol=0.0,
        )
    ]
    for points in sorted({row["points"] for row in plotted_rows}):
        color = f"C{points % 10}"
        point_rows = [row for row in plotted_rows if row["points"] == points]
        for row in point_rows:
            cycles = np.asarray(row["time_cycles"])
            actual = np.asarray(row["actual_density_real"])
            continuum = np.asarray(row["continuum_density_real"])
            line_label = f"n={points}, {row['steps']} steps"
            axes[0, 0].plot(cycles, actual, color=color, linewidth=1.5,
                            linestyle="-" if row["steps"] == max(
                                item["steps"] for item in point_rows
                            ) else "--", label=f"actual {line_label}")
            axes[0, 0].plot(cycles, continuum, color=color, linewidth=1.0,
                            linestyle=":", label=f"continuum {line_label}")
            axes[0, 2].loglog(
                row["dt_s"], row["frequency_relative_error"],
                marker="o", color=color,
                label=f"n={points}, {row['steps']} steps",
            )
            axes[1, 0].plot(
                row["dt_s"], -row["measured_growth_rate_per_s"],
                marker="o", color=color, linestyle="-",
                label=f"n={points}, {row['steps']} steps measured",
            )
            axes[1, 0].plot(
                row["dt_s"], -row["rk4_predicted_growth_rate_per_s"],
                marker="x", color=color, linestyle="--",
                label=f"n={points}, {row['steps']} steps RK4",
            )
            axes[1, 2].loglog(
                row["dt_s"], row["phase_residual_rms_rad"],
                marker="o", color=color,
                label=f"n={points}, {row['steps']} steps",
            )
    axes[0, 0].set(
        xlabel="Time (physical cycles)",
        ylabel="Re(a_e) (normalized coefficient)",
        title="Actual density histories versus continuum",
    )
    axes[0, 2].set(
        xlabel="dt (s)", ylabel="|f_measured - f_continuum| / f_continuum",
        title="Frequency relative error",
    )
    axes[1, 0].set(
        xlabel="dt (s)", ylabel="-gamma (1/s)",
        title="Damping rate: measured versus RK4 prediction",
    )
    axes[1, 0].axhline(0.0, color="0.3", linewidth=1.0, linestyle=":",
                       label="continuum gamma=0")
    axes[1, 0].set_ylim(bottom=0.0)
    axes[1, 2].set(
        xlabel="dt (s)", ylabel="Phase-fit residual RMS (rad)",
        title="Mode phase-fit residual",
    )

    reconstructions = data["modes"]["reconstructions"]
    if reconstructions:
        chosen = max(reconstructions, key=lambda row: row["points"])
        mesh = axes[0, 1].pcolormesh(
            chosen["wavelength_fraction"], chosen["time_cycles"],
            chosen["normalized_real_density"], shading="auto", cmap="RdBu_r",
        )
        fig.colorbar(mesh, ax=axes[0, 1], fraction=0.046, pad=0.03,
                     label="Re(a(t) exp(i k s))")
        axes[0, 1].set(
            xlabel="Position over one wavelength",
            ylabel="Time (physical cycles)",
            title=f"Measured-coefficient reconstruction, n={chosen['points']}",
        )
    else:
        axes[0, 1].text(
            0.5, 0.5, "reconstruction unavailable",
            transform=axes[0, 1].transAxes, ha="center", va="center",
        )

    controls = data["modes"]["amplitude_controls"]
    for control in controls:
        axes[1, 1].plot(
            control["time_cycles"],
            control["normalized_history_difference"],
            label=f"n={control['points']}, {control['steps']} steps",
        )
    axes[1, 1].set(
        xlabel="Time (physical cycles)",
        ylabel="|normalized full - normalized half|",
        title="Full/half-amplitude normalized histories",
    )
    if not controls:
        axes[1, 1].text(
            0.5, 0.5, "half-amplitude control unavailable",
            transform=axes[1, 1].transAxes, ha="center", va="center",
            fontsize=10,
        )
    for axis in (axes[0, 2], axes[1, 0], axes[1, 2]):
        axis.invert_xaxis()
        axis.xaxis.set_minor_formatter(NullFormatter())
    for axis in axes.flat:
        axis.grid(True, which="both", alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=9)
    devices = data["modes"]["devices"]
    unavailable = [
        f"{run} ({state})"
        for run, state in data["modes"]["suite_states"].items()
        if state not in ("measured",)
    ]
    fig.suptitle(
        "Eight-cycle modes ("
        f"{', '.join(devices) if devices else 'unavailable'} production stage; "
        "test-only periodic spectral adapter)",
        fontsize=13,
    )
    mode_failure = data["reports"].get("modes-n33", {}).get("failure", {})
    failure_text = ""
    if isinstance(mode_failure, dict) and mode_failure:
        failure_text = (
            f" modes-n33: {mode_failure.get('code', 'failure')} "
            f"{mode_failure.get('message', '')}."
        )
    note = (
        "n17/n33 are periodic adapter sample counts, not production LCFS grid "
        "refinement. True continuum gamma=0. "
        + (f"Unavailable/failed suites: {', '.join(unavailable)}." if unavailable else "")
        + failure_text
        + " No ICE, nonlinear, closed-wall, or performance claim is made."
    )
    fig.text(0.5, -0.02, textwrap.fill(note, width=150),
             ha="center", va="top", fontsize=9)

    _save_figure(fig, output, "eight-cycle-modes")
    plt.close(fig)


def render(data: dict, output: Path) -> str:
    """Render the four required PNG/PDF figure pairs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    _render_geometry(data, output)
    _render_spatial(data, output)
    _render_momentum(data, output)
    _render_modes(data, output)
    return matplotlib.__version__


def _write_scope(output: Path) -> None:
    (output / "SCOPE.md").write_text(
        "# Figure scope\n\n"
        "These figures are offline measurements from archived A0 fixed-domain "
        "evidence. Spatial norms use the saved production staggered layout; the "
        "0.15 m wall band is fixed and spatial slopes are descriptive. No new "
        "physical acceptance threshold or asymptotic order is selected. The raw "
        "full-force boundary lift is not a pointwise force-error measurement.\n\n"
        "Static evidence is CUDA and dynamic evidence is CPU. Dynamic momentum_x "
        "is the Cartesian p_R cos(phi) - p_phi sin(phi) defect accumulated from "
        "signed interval diagnostics. Numerical wall reaction is not added to "
        "physical outward traction. Quiet drift is normalized only by the scales "
        "declared in the reports.\n\n"
        "The eight-cycle cases use production stage, time transaction, and "
        "diagnostics with a test-only periodic spectral adapter. n17 and n33 are "
        "adapter sample counts, not production LCFS grid refinement. The "
        "reconstruction covers one periodic wavelength and is not a global "
        "Tokamak field. No ICE, nonlinear, closed-wall, or Warp-versus-C++ "
        "performance claim is made.\n\n"
        "Missing, running, failed, skipped, and unavailable cases remain in "
        "plot-data.json. The tool runs no solver and applies no hidden tolerance "
        "to physical errors.\n"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="archived run root (read only)")
    parser.add_argument("--output", type=Path, required=True,
                        help="new output directory (never overwritten)")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    data = collect(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="fomhd-mpl-") as cache_dir:
        os.environ["MPLCONFIGDIR"] = cache_dir
        matplotlib_version = render(data, args.output)
    (args.output / "plot-data.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    provenance = {
        "source_root": str(args.root.resolve()),
        "input_sha256": data["inputs"],
        "report_metadata": data["reports"],
        "command": [str(Path(sys.argv[0]))] + (
            list(argv) if argv is not None else sys.argv[1:]
        ),
        "script_sha256": _sha256(Path(__file__).read_bytes()),
        "python": sys.version,
        "numpy": np.__version__,
        "matplotlib": matplotlib_version,
        "solver_executed": False,
        "host_device_placement": "host reads; CPU-only rendering",
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n"
    )
    _write_scope(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
