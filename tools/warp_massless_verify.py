#!/usr/bin/env python3
"""CLI evidence tool for bounded massless Hall and spatial verification."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = SOURCE_ROOT.parents[1]
FIXTURE_DIR = SOURCE_ROOT / "tests" / "warp"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(FIXTURE_DIR) not in sys.path:
    sys.path.insert(0, str(FIXTURE_DIR))

from massless_verification_fixture import (  # noqa: E402
    HallCase,
    closed_spatial_evidence,
    cold_transverse_matrix,
    drift_ohm_tautology_evidence,
    hall_frequencies,
    production_transverse_matrix,
    reconstruction_evidence,
    run_hall_time_case,
    variable_density_velocity_evidence,
)


HARNESS_FILES = (
    "docs/pseudocode/tests/warp/massless_verification_fixture.py.md",
    "docs/pseudocode/tests/warp/test_massless_verification.py.md",
    "docs/pseudocode/tools/warp_massless_verify.py.md",
    "tests/warp/massless_verification_fixture.py",
    "tests/warp/test_massless_verification.py",
    "tools/warp_massless_verify.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    result = {}
    for path in sorted((SOURCE_ROOT / "fomhd").glob("*.py")):
        result[str(path.relative_to(SOURCE_ROOT))] = _sha256(path)
    return result


def _dependency_hashes() -> dict[str, Any]:
    dependencies = {}
    for name, version_module, hash_module in (
        ("numpy", "numpy", "numpy._core._multiarray_umath"),
        ("scipy", "scipy", "scipy.linalg._fblas"),
        ("pytest", "pytest", "pytest"),
        ("warp", "warp", "warp"),
        ("cupy", "cupy", "cupy"),
    ):
        entry: dict[str, Any] = {"available": False}
        try:
            spec = importlib.util.find_spec(hash_module)
            version_package = importlib.import_module(version_module)
            module = importlib.import_module(hash_module)
            entry.update(
                available=True,
                version=getattr(version_package, "__version__", None),
                module_file=getattr(module, "__file__", None),
                module_path=(
                    list(spec.submodule_search_locations or ())
                    if spec is not None
                    else []
                ),
            )
            module_file = getattr(module, "__file__", None)
            if module_file and Path(module_file).is_file():
                entry["sha256"] = _sha256(Path(module_file))
        except Exception as error:
            entry["reason"] = f"{type(error).__name__}: {error}"
        dependencies[name] = entry
    dependencies["python"] = {
        "available": True,
        "version": sys.version,
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }
    return dependencies


def _harness_manifest() -> dict[str, Any]:
    files = {}
    for relative in HARNESS_FILES:
        path = SOURCE_ROOT / relative
        files[relative] = {
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
    return {
        "scope": "harness files only; solver dependencies are hashed separately",
        "files": files,
    }


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, complex):
        return {"real": _json_value(value.real), "imag": _json_value(value.imag)}
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            import cupy as cp

            if isinstance(value, cp.ndarray):
                return _json_value(cp.asnumpy(value))
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return _json_value(vars(value))
    return str(value)


def _diagnostics(records: tuple[Any, ...] | list[Any]) -> list[dict[str, Any]]:
    return [_json_value(record) for record in records]


def _diagnostic_summary(
    records: tuple[Any, ...] | list[Any],
) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        item = {
            "name": getattr(record, "name", ""),
            "population": getattr(record, "population", ""),
            "component": getattr(record, "component", ""),
            "region": getattr(record, "region", ""),
            "unit": getattr(record, "unit", ""),
            "value_unit": getattr(record, "value_unit", ""),
            "norm": getattr(
                getattr(record, "norm", ""),
                "value",
                str(getattr(record, "norm", "")),
            ),
            "raw": _json_value(getattr(record, "raw", None)),
            "reference": _json_value(getattr(record, "reference", None)),
            "threshold": _json_value(getattr(record, "threshold", None)),
            "tag": _json_value(getattr(record, "tag", None)),
            "hard": bool(getattr(record, "hard", False)),
            "available": bool(getattr(record, "available", False)),
            "passed": bool(getattr(record, "passed", False)),
            "reason": (
                getattr(record, "reason", "")
                if not bool(getattr(record, "passed", False))
                else ""
            ),
        }
        key = json.dumps(
            {
                name: item[name]
                for name in (
                    "name",
                    "population",
                    "component",
                    "region",
                    "unit",
                    "value_unit",
                    "norm",
                    "hard",
                    "available",
                    "passed",
                )
            },
            sort_keys=True,
        )
        if key in unique:
            unique[key]["count"] += 1
        else:
            unique[key] = {**item, "count": 1}
    return list(unique.values())


def _diagnostic_report_summary(
    records: tuple[Any, ...] | list[Any],
) -> list[dict[str, Any]]:
    counts: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        item = {
            "name": getattr(record, "name", ""),
            "population": getattr(record, "population", ""),
            "component": getattr(record, "component", ""),
            "region": getattr(record, "region", ""),
            "hard": bool(getattr(record, "hard", False)),
            "available": bool(getattr(record, "available", False)),
            "passed": bool(getattr(record, "passed", False)),
        }
        key = tuple(item.values())
        if key in counts:
            counts[key]["count"] += 1
        else:
            counts[key] = {**item, "count": 1}
    return list(counts.values())


def _all_hard_pass(records: tuple[Any, ...] | list[Any]) -> bool:
    hard = [record for record in records if getattr(record, "hard", False)]
    return bool(hard) and all(
        bool(getattr(record, "available", False))
        and bool(getattr(record, "passed", False))
        for record in hard
    )


def _save_npz(
    path: Path,
    arrays: dict[str, Any],
    *,
    allow_runtime_capture: bool = False,
) -> None:
    cleaned = {}
    for name, value in arrays.items():
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            cleaned[name] = value
        elif allow_runtime_capture and isinstance(value, (list, tuple)):
            cleaned[name] = np.asarray(value)
        elif isinstance(value, np.generic) or np.isscalar(value):
            cleaned[name] = np.asarray(value)
        elif allow_runtime_capture:
            try:
                import cupy as cp

                if isinstance(value, cp.ndarray):
                    cleaned[name] = cp.asnumpy(value)
                    continue
            except Exception:
                pass
            raise TypeError(
                f"{name} is not a supported runtime array for explicit capture"
            )
        else:
            raise TypeError(
                f"{name} must be converted explicitly from its owning backend "
                "before NPZ output"
            )
    for name, value in cleaned.items():
        if value.dtype.hasobject:
            raise TypeError(f"{name} has object dtype; NPZ evidence must be pickle-free")
    if not cleaned:
        cleaned["empty"] = np.empty(0, dtype=np.float64)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **cleaned)


def _save_json(path: Path, payload: Any) -> None:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "xt", encoding="utf-8") as stream:
        json.dump(
            _json_value(payload),
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")


def _case_error(
    label: str,
    error: BaseException,
    output: Path | None = None,
) -> dict[str, Any]:
    partial_arrays = getattr(error, "partial_arrays", {}) or {}
    checks = getattr(error, "checks", ()) or ()
    stage = getattr(error, "stage", None)
    array_names = sorted(partial_arrays)
    array_file = None
    if output is not None and partial_arrays:
        array_path = output / f"{label}_failure.npz"
        _save_npz(
            array_path,
            partial_arrays,
            allow_runtime_capture=True,
        )
        array_file = array_path.name
    record_file = None
    if output is not None:
        record_path = output / f"{label}_failure.json"
        record_path.write_text(
            json.dumps(
                _json_value(
                    {
                        "case": label,
                        "code": getattr(error, "code", type(error).__name__),
                        "message": str(error),
                        "stage": stage,
                        "partial_array_names": array_names,
                        "checks": list(checks),
                    }
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        record_file = record_path.name
    return {
        "case": label,
        "status": "error",
        "failure_code": getattr(error, "code", type(error).__name__),
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
        "failure_stage": _json_value(stage),
        "failure_array_names": array_names,
        "failure_check_count": len(checks),
        "failure_arrays_file": array_file,
        "failure_record_file": record_file,
        "failure_arrays": (
            {
                name: f"{array_file}:{name}"
                for name in array_names
            }
            if array_file is not None
            else {}
        ),
    }


def _run_hall(
    case: HallCase,
    output: Path,
    device: str,
) -> dict[str, Any]:
    label = f"hall_kdi_{case.kdi:.6g}"
    try:
        production, matrices, evidence = production_transverse_matrix(
            case, device=device
        )
        oracle = cold_transverse_matrix(case)
        scale = max(float(np.linalg.norm(oracle, ord=2)), 1.0)
        matrix_errors = [
            float(np.linalg.norm(matrix - oracle, ord=2) / scale)
            for matrix in matrices
        ]
        production_values = np.linalg.eigvals(production)
        oracle_values = np.linalg.eigvals(oracle)
        frequency = hall_frequencies(case)
        hard_pass = _all_hard_pass(evidence["diagnostics"])
        passed = (
            max(matrix_errors) < 2.0e-10
            and hard_pass
            and float(np.max(np.abs(production_values.imag))) > 0.0
        )
        _save_npz(
            output / f"{label}.npz",
            {
                "production_matrix": production,
                "amplitude_matrices": np.stack(matrices),
                "independent_matrix": oracle,
                "production_eigenvalues": production_values,
                "independent_eigenvalues": oracle_values,
                "matrix_errors": np.asarray(matrix_errors),
            },
        )
        diagnostics_file = f"{label}_diagnostics.json"
        _save_json(
            output / diagnostics_file,
            {
                "case": label,
                "checks": [
                    _json_value(record)
                    for record in evidence["diagnostics"]
                ],
            },
        )
        return {
            "case": label,
            "kind": "controlled_local_parallel_hall",
            "status": "pass" if passed else "failed",
            "scope": "local periodic transverse theory; not global/ICE validation",
            "kdi": case.kdi,
            "k_per_m": case.k,
            "di_m": case.di,
            "amplitudes": [2.0e-6, 4.0e-6],
            "matrix_relative_errors": matrix_errors,
            "omega_minus": frequency.omega_minus,
            "omega_plus": frequency.omega_plus,
            "omega_plasma_independent": frequency.omega_plasma,
            "hard_diagnostics_pass": hard_pass,
            "hard_diagnostics": _diagnostic_report_summary(
                evidence["diagnostics"]
            ),
            "diagnostics_file": diagnostics_file,
            "failure_arrays": {},
            "raw_file": f"{label}.npz",
        }
    except BaseException as error:
        return _case_error(label, error, output)


def _run_time(
    case: HallCase,
    output: Path,
    integrator: str,
    steps_per_cycle: int,
    cycles: int,
    device: str,
) -> dict[str, Any]:
    label = f"time_{integrator}_{steps_per_cycle}spc_{cycles}cyc"
    try:
        result = run_hall_time_case(
            case,
            integrator=integrator,
            steps_per_cycle=steps_per_cycle,
            cycles=cycles,
            relative_amplitude=2.0e-6,
            device=device,
        )
        polynomial_ok = result.relative_error < 2.0e-11
        passed = (
            result.finite
            and result.accepted_steps == result.requested_steps
            and result.source_is_none
            and result.hard_checks_available
            and polynomial_ok
        )
        _save_npz(
            output / f"{label}.npz",
            {
                "actual_mode": result.actual_mode,
                "polynomial_mode": result.expected_mode,
                "relative_error": result.relative_error,
                "continuum_relative_error": result.continuum_relative_error,
                "amplification": result.amplification,
                "amplitude_ratio": result.amplitude_ratio,
                "phase_error_rad": result.phase_error_rad,
                "continuum_amplitude_ratio": result.continuum_amplitude_ratio,
                "continuum_phase_error_rad": result.continuum_phase_error_rad,
            },
        )
        diagnostics_file = f"{label}_diagnostics.json.gz"
        _save_json(
            output / diagnostics_file,
            {
                "case": label,
                "checks": [
                    _json_value(record)
                    for record in result.diagnostics
                ],
                "dt_reports": list(result.dt_reports),
            },
        )
        return {
            "case": label,
            "kind": "real_time_source_none",
            "status": "pass" if passed else "failed",
            "scope": "bounded linear eigenmode; nonlinear truncation not exercised",
            "integrator": integrator,
            "steps_per_cycle": steps_per_cycle,
            "cycles": cycles,
            "dt_s": result.dt,
            "final_time_s": result.final_time,
            "omega_rad_s": result.omega,
            "amplification": _json_value(result.amplification),
            "polynomial_relative_error": result.relative_error,
            "continuum_relative_error": result.continuum_relative_error,
            "amplitude_ratio_to_polynomial": result.amplitude_ratio,
            "phase_error_to_polynomial_rad": result.phase_error_rad,
            "amplitude_ratio_to_continuum": result.continuum_amplitude_ratio,
            "phase_error_to_continuum_rad": result.continuum_phase_error_rad,
            "initial_mode_error": result.initial_mode_error,
            "hard_diagnostics_pass": result.hard_checks_available,
            "failure_code": result.failure_code,
            "failure_message": result.failure_message,
            "diagnostics": _diagnostic_report_summary(result.diagnostics),
            "dt_reports": list(result.dt_reports),
            "diagnostics_file": diagnostics_file,
            "runtime_warning": result.warning,
            "failure_arrays": (
                {}
                if result.finite and polynomial_ok
                else {
                    "actual_mode": f"{label}.npz:actual_mode",
                    "relative_error": result.relative_error,
                }
            ),
            "raw_file": f"{label}.npz",
        }
    except BaseException as error:
        return _case_error(label, error, output)


def _run_spatial(
    nodes: int,
    compressible: bool,
    output: Path,
    device: str,
) -> dict[str, Any]:
    label = f"spatial_n{nodes}_{'compressible' if compressible else 'incompressible'}"
    try:
        evidence = closed_spatial_evidence(
            nodes=nodes,
            compressible=compressible,
            device=device,
        )
        delta_available = (
            evidence.retained_delta_relative_error["all"] < 2.0e-8
        )
        measured = (
            evidence.theta_relative_error["all"] < 2.0e-10
            and evidence.full_theta_relative_error["all"] < 2.0e-10
            and delta_available
            and np.isfinite(evidence.pressure_force_error["all"])
            and np.isfinite(evidence.ohm_error["all"])
        )
        arrays = {
            f"raw_{name}": value
            for name, value in evidence.arrays.items()
        }
        _save_npz(output / f"{label}.npz", arrays)
        return {
            "case": label,
            "kind": "closed_rectangle_operator_consistency",
            "status": "measured" if measured else "failed",
            "scope": (
                "prescribed smooth operator-level fields; separate from full "
                "nonlinear equilibrium evolution"
            ),
            "compressible": compressible,
            "nodes": nodes,
            "minimum_density": evidence.minimum_density,
            "maximum_density": evidence.maximum_density,
            "theta_relative_error": evidence.theta_relative_error,
            "full_padded_theta_relative_error": evidence.full_theta_relative_error,
            "retained_delta_theta_relative_error": evidence.retained_delta_relative_error,
            "wrong_djfe_over_ne_relative_error": evidence.wrong_djfe_relative_error,
            "pressure_force_relative_error": evidence.pressure_force_error,
            "pressure_norms": evidence.pressure_norms,
            "ohm_relative_error": evidence.ohm_error,
            "ohm_norms": evidence.ohm_norms,
            "delta_theta_at_roundoff": delta_available,
            "failure_arrays": {
                "retained_delta_theta_relative_error": (
                    evidence.retained_delta_relative_error
                ),
                "theta": f"{label}.npz:raw_theta",
                "theta_retained": f"{label}.npz:raw_theta_retained",
                "nonfinite_raw_fields": [
                    name
                    for name, values in evidence.arrays.items()
                    if not np.all(np.isfinite(values))
                ],
            },
            "raw_file": f"{label}.npz",
        }
    except BaseException as error:
        return _case_error(label, error, output)


def _run_density_layout(
    nodes: int,
    output: Path,
    device: str,
) -> dict[str, Any]:
    label = f"velocity_layout_n{nodes}"
    try:
        evidence = variable_density_velocity_evidence(
            nodes=nodes,
            device=device,
        )
        passed = evidence.relative_error < 2.0e-10
        _save_npz(
            output / f"{label}.npz",
            {
                f"raw_{name}": value
                for name, value in evidence.arrays.items()
            },
        )
        return {
            "case": label,
            "kind": "variable_density_qhfe_velocity",
            "status": "pass" if passed else "failed",
            "scope": "actual K=1 closed-grid QHFe/ne sample layout",
            "nodes": nodes,
            "density_min": evidence.density_min,
            "density_max": evidence.density_max,
            "relative_error": evidence.relative_error,
            "linf_error": evidence.linf_error,
            "failure_arrays": {
                "actual_velocity": f"{label}.npz:raw_actual_velocity",
                "expected_velocity": f"{label}.npz:raw_expected_velocity",
                "residual": f"{label}.npz:raw_residual",
            },
            "raw_file": f"{label}.npz",
        }
    except BaseException as error:
        return _case_error(label, error, output)


def _run_reconstruction(
    nodes: int,
    output: Path,
    device: str,
) -> dict[str, Any]:
    label = f"reconstruction_n{nodes}"
    try:
        evidence = reconstruction_evidence(
            nodes=nodes,
            device=device,
        )
        measured = (
            evidence.minimum_density > 0.0
            and evidence.maximum_abs_current > 0.0
            and evidence.current_relation_error < 1.0e-10
            and evidence.relative_wall_flux < 2.0e-10
        )
        _save_npz(
            output / f"{label}.npz",
            {
                f"raw_{name}": value
                for name, value in evidence.arrays.items()
            },
        )
        return {
            "case": label,
            "kind": "closed_qhfe_reconstruction",
            "status": "measured" if measured else "failed",
            "scope": (
                "smooth analytic edge flux plus nonzero Cdag-derived current "
                "through actual QHFe/ne; not a full stage equilibrium"
            ),
            "nodes": nodes,
            "h": evidence.h,
            "minimum_density": evidence.minimum_density,
            "maximum_density": evidence.maximum_density,
            "maximum_abs_current": evidence.maximum_abs_current,
            "current_relation_error": evidence.current_relation_error,
            "velocity_norms": evidence.velocity_norms,
            "theta_norms": evidence.theta_norms,
            "delta_theta_norms": evidence.delta_theta_norms,
            "maximum_wall_flux": evidence.maximum_wall_flux,
            "relative_wall_flux": evidence.relative_wall_flux,
            "failure_arrays": {
                name: f"{label}.npz:raw_{name}"
                for name in evidence.arrays
            },
            "raw_file": f"{label}.npz",
        }
    except BaseException as error:
        return _case_error(label, error, output)


def _run_drift_ohm(
    output: Path,
    device: str,
) -> dict[str, Any]:
    label = "diagnostic_drift_ohm_doubling"
    try:
        evidence = drift_ohm_tautology_evidence(device=device)
        passed = (
            evidence.gate_failed
            and evidence.baseline_linf < evidence.hard_threshold
            and evidence.doubled_linf > evidence.hard_threshold
            and evidence.electric_field_linf
            > 10.0 * evidence.hard_threshold
        )
        _save_npz(
            output / f"{label}.npz",
            {
                f"raw_{name}": value
                for name, value in evidence.arrays.items()
            },
        )
        return {
            "case": label,
            "kind": "diagnostic_negative_control",
            "status": "pass" if passed else "failed",
            "scope": (
                "nonzero transverse Bx perturbation; doubled E0 must fail "
                "the production hard drift-Ohm gate"
            ),
            "electric_field_linf": evidence.electric_field_linf,
            "baseline_linf": evidence.baseline_linf,
            "doubled_linf": evidence.doubled_linf,
            "hard_threshold": evidence.hard_threshold,
            "unit": evidence.unit,
            "gate_failed": evidence.gate_failed,
            "failure_arrays": {
                "residual": f"{label}.npz:raw_perturbed_residual",
                "field": f"{label}.npz:raw_E0",
            },
            "raw_file": f"{label}.npz",
        }
    except BaseException as error:
        return _case_error(label, error, output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="cpu", choices=("cpu", "cuda:0"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hall-kdi",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0],
    )
    parser.add_argument(
        "--integrators",
        nargs="+",
        choices=("midpoint", "rk4"),
        default=["midpoint", "rk4"],
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        nargs="+",
        default=[48, 96, 192],
    )
    parser.add_argument("--time-cycles", type=int, default=1)
    parser.add_argument(
        "--spatial-nodes",
        type=int,
        nargs="+",
        default=[4, 6, 8],
    )
    parser.add_argument("--hall-only", action="store_true")
    parser.add_argument("--skip-time", action="store_true")
    parser.add_argument("--skip-spatial", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.backend.startswith("cuda:") and not os.environ.get("SLURM_JOB_ID"):
        print(
            "refusing CUDA backend without SLURM_JOB_ID; no CPU fallback",
            file=sys.stderr,
        )
        return 2
    if args.output.exists():
        print(
            f"refusing existing output directory: {args.output}",
            file=sys.stderr,
        )
        return 2
    if args.time_cycles <= 0 or any(value < 32 for value in args.time_steps):
        print("time steps must be >=32 and cycles positive", file=sys.stderr)
        return 2
    if any(value < 4 for value in args.spatial_nodes):
        print("spatial grids require at least 4 cells", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "backend_requested": args.backend,
        "scope": {
            "included": [
                "controlled local periodic transverse Hall theory",
                "real midpoint and RK4 source=None advance",
                "closed-rectangle sign- and region-corrected operator norms",
                "smooth flux/current actual QHFe/ne reconstruction",
            ],
            "excluded": [
                "tokamak/global/ICE validation",
                "complete spectrum claim",
                "nonlinear benchmark acceptance",
                "GPU execution unless explicitly run in Slurm",
            ],
        },
        "source_hashes": _source_hashes(),
        "cases": [],
    }
    try:
        for kdi in args.hall_kdi:
            report["cases"].append(
                _run_hall(HallCase(kdi=kdi), args.output, args.backend)
            )
        if not args.hall_only:
            report["cases"].append(_run_drift_ohm(args.output, args.backend))
            if not args.skip_time:
                for integrator in args.integrators:
                    for steps in args.time_steps:
                        report["cases"].append(
                            _run_time(
                                HallCase(kdi=1.0),
                                args.output,
                                integrator,
                                steps,
                                args.time_cycles,
                                args.backend,
                            )
                        )
            if not args.skip_spatial:
                for nodes in args.spatial_nodes:
                    report["cases"].append(
                        _run_density_layout(nodes, args.output, args.backend)
                    )
                    report["cases"].append(
                        _run_reconstruction(nodes, args.output, args.backend)
                    )
                    report["cases"].append(
                        _run_spatial(nodes, False, args.output, args.backend)
                    )
                    report["cases"].append(
                        _run_spatial(nodes, True, args.output, args.backend)
                    )
        time_groups: dict[str, list[dict[str, Any]]] = {}
        for case in report["cases"]:
            if case.get("kind") == "real_time_source_none":
                time_groups.setdefault(case["integrator"], []).append(case)
        for integrator, group in time_groups.items():
            expected_order = 2.0 if integrator == "midpoint" else 4.0
            ordered = sorted(group, key=lambda item: item["steps_per_cycle"])
            previous: dict[str, Any] | None = None
            for case in ordered:
                case["expected_continuum_order"] = expected_order
                if previous is None:
                    case["observed_continuum_order"] = None
                    previous = case
                    continue
                error_now = float(case["continuum_relative_error"])
                error_previous = float(previous["continuum_relative_error"])
                if error_now > 0.0 and error_previous > 0.0:
                    observed = math.log(error_previous / error_now) / math.log(
                        case["steps_per_cycle"] / previous["steps_per_cycle"]
                    )
                else:
                    observed = math.nan
                case["observed_continuum_order"] = observed
                if not math.isfinite(observed) or observed < 0.6 * expected_order:
                    case["status"] = "failed"
                previous = case
        spatial_groups: dict[bool, list[dict[str, Any]]] = {}
        for case in report["cases"]:
            if case.get("kind") == "closed_rectangle_operator_consistency":
                spatial_groups.setdefault(
                    bool(case["compressible"]), []
                ).append(case)
        required_spatial: list[dict[str, Any]] = []
        for compressible, group in spatial_groups.items():
            ordered = sorted(group, key=lambda item: item["nodes"])
            previous: dict[str, Any] | None = None
            for case in ordered:
                if previous is None:
                    case["observed_orders"] = {}
                    case["order_scope"] = (
                        "fixed-domain weighted coarse-grid order; "
                        "no global-order claim"
                    )
                    case["wall_convergence"] = "not_yet_estimated"
                    previous = case
                    continue
                ratio = math.log(
                    case["nodes"] / previous["nodes"]
                )
                orders: dict[str, Any] = {}
                for region in ("all", "inner", "wall"):
                    for metric, key in (
                        ("pressure", "pressure_force_relative_error"),
                        ("ohm", "ohm_relative_error"),
                    ):
                        before = float(previous[key][region])
                        now = float(case[key][region])
                        if (
                            not math.isfinite(before)
                            or not math.isfinite(now)
                            or before <= 1.0e-12
                            or now <= 1.0e-12
                        ):
                            value: Any = "roundoff_or_unavailable"
                        else:
                            value = math.log(before / now) / ratio
                        orders[f"{metric}.{region}"] = value
                case["observed_orders"] = orders
                case["order_scope"] = (
                    "fixed-domain weighted coarse-grid order; "
                    "wall nonconvergence is reported, not hidden"
                )
                wall_orders = [
                    orders[f"{metric}.wall"]
                    for metric in ("pressure", "ohm")
                ]
                case["wall_convergence"] = (
                    "nonconvergent_or_mixed"
                    if any(
                        not isinstance(value, float) or value <= 0.0
                        for value in wall_orders
                    )
                    else "measured_decreasing"
                )
                requirements = {
                    f"{metric}.{region}": (
                        isinstance(orders[f"{metric}.{region}"], float)
                        and orders[f"{metric}.{region}"] > 0.0
                    )
                    for metric in ("pressure", "ohm")
                    for region in ("all", "inner")
                }
                case["required_order_checks"] = requirements
                if not all(requirements.values()):
                    case["status"] = "failed"
                required_spatial.extend(
                    {
                        "compressible": compressible,
                        "metric": key,
                        "passed": value,
                    }
                    for key, value in requirements.items()
                )
                previous = case

        reconstruction_groups: dict[int, list[dict[str, Any]]] = {}
        for case in report["cases"]:
            if case.get("kind") == "closed_qhfe_reconstruction":
                reconstruction_groups.setdefault(case["nodes"], []).append(case)
        reconstruction_cases = sorted(
            (
                case
                for group in reconstruction_groups.values()
                for case in group
            ),
            key=lambda item: item["nodes"],
        )
        previous_reconstruction: dict[str, Any] | None = None
        required_reconstruction: list[dict[str, Any]] = []
        for case in reconstruction_cases:
            if previous_reconstruction is None:
                case["observed_orders"] = {}
                case["order_scope"] = (
                    "fixed-domain reconstructed-velocity convergence; "
                    "wall values are reported separately"
                )
                case["wall_convergence"] = "not_yet_estimated"
                previous_reconstruction = case
                continue
            ratio = math.log(
                case["nodes"] / previous_reconstruction["nodes"]
            )
            orders: dict[str, Any] = {}
            for field in ("velocity", "theta", "delta_theta"):
                for region in ("all", "inner", "wall"):
                    before = float(
                        previous_reconstruction[f"{field}_norms"][region][
                            "weighted_relative_l2"
                        ]
                    )
                    now = float(
                        case[f"{field}_norms"][region][
                            "weighted_relative_l2"
                        ]
                    )
                    orders[f"{field}.{region}"] = (
                        math.log(before / now) / ratio
                        if before > 0.0 and now > 0.0
                        else "roundoff_or_unavailable"
                    )
            required = {
                f"{field}.{region}": (
                    isinstance(orders[f"{field}.{region}"], float)
                    and orders[f"{field}.{region}"] > 0.0
                )
                for field in ("velocity", "theta", "delta_theta")
                for region in ("all", "inner")
            }
            case["observed_orders"] = orders
            case["required_order_checks"] = required
            case["order_scope"] = (
                "fixed-domain reconstructed-velocity convergence; "
                "wall values are reported separately"
            )
            if not all(required.values()):
                case["status"] = "failed"
            wall_orders = [
                orders[f"{field}.wall"]
                for field in ("velocity", "theta", "delta_theta")
            ]
            case["wall_convergence"] = (
                "nonconvergent_or_mixed"
                if any(
                    not isinstance(value, float) or value <= 0.0
                    for value in wall_orders
                )
                else "measured_decreasing"
            )
            required_reconstruction.extend(
                {"metric": metric, "passed": passed}
                for metric, passed in required.items()
            )
            previous_reconstruction = case

        execution_complete = all(
            case["status"] in ("pass", "measured")
            for case in report["cases"]
        )
        scientific_limits = [
            "wall pressure/Ohm/reconstruction orders are descriptive",
            "tokamak/global/ICE validation not performed",
            "nonlinear long-horizon behavior not established",
        ]
        report["requirements"] = {
            "spatial_order_checks": required_spatial,
            "reconstruction_order_checks": required_reconstruction,
        }
        report["scientific_acceptance"] = (
            "not_determined_by_this_bounded_evidence"
        )
        report["scientific_limits"] = scientific_limits
        report["overall_status"] = (
            "completed_with_scope_limits"
            if execution_complete
            else "failed"
        )
        (args.output / "dependencies.json").write_text(
            json.dumps(_dependency_hashes(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.output / "harness_manifest.json").write_text(
            json.dumps(_harness_manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.output / "report.json").write_text(
            json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.output / "LOGS.txt").write_text(
            f"backend={args.backend}\noverall_status={report['overall_status']}\n",
            encoding="utf-8",
        )
    except BaseException as error:
        failure = _case_error("cli", error, args.output)
        report["cases"].append(failure)
        report["overall_status"] = "error"
        (args.output / "report.json").write_text(
            json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return (
        0
        if report["overall_status"] == "completed_with_scope_limits"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
