#!/usr/bin/env python3
"""Bounded A1-T0 temporal evidence using real solver paths and test fixtures."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FIXTURE_DIR = ROOT / "tests" / "warp"
for search_path in (str(SRC), str(FIXTURE_DIR), str(ROOT)):
    if search_path not in sys.path:
        sys.path.insert(0, search_path)

from fomhd import massless  # noqa: E402
from fomhd.backend import Backend  # noqa: E402
from fomhd.core import forward  # noqa: E402
from fomhd.geometry import build_rectangle  # noqa: E402
from fomhd.operators import build_operators  # noqa: E402
from fomhd.stage import initialize  # noqa: E402
from fomhd.time import advance  # noqa: E402
from fomhd.types import Context, Failure, cumulative_record  # noqa: E402
from massless_verification_fixture import (  # noqa: E402
    HallCase,
    build_periodic_context,
    hall_frequencies,
    initial_hall_eigenmode,
    uniform_state,
)
from test_massless_support import (  # noqa: E402
    equilibrium_state as closed_equilibrium_state,
    parameters as closed_parameters,
)


SCHEMA_VERSION = 1
VALID_DEVICES = ("cpu", "cuda:0")
VALID_MICRO_STEPS_PER_CYCLE = (64, 128)
DEFAULT_KDIS = (1.0,)
DEFAULT_MICRO_STEPS = (1, 2, 4, 8)
PERIODIC_SCOPE = (
    "Periodic temporal adapter for a single cold Hall eigenmode. It exercises "
    "the real production stage, diagnostics, canonicalization and time "
    "transaction, but is not closed-wall physical or experimental validation."
)
CLOSED_SCOPE = (
    "Small closed-rectangle equilibrium with a smooth nonzero perturbation, "
    "using the existing real production stage and hard physical diagnostics. "
    "This is bounded fixture evidence, not a full equilibrium or wall study."
)
PHASE_CONVENTION = (
    "signed residual phase is atan2(Im(projection), Re(projection)) in the "
    "principal interval [-pi, pi], equal to the residual modulo 2*pi; "
    "positive means the observed mode leads the reference under "
    "exp(+i*omega*T)"
)
PRINCIPAL_AMBIGUITY = (
    "phase is a principal-angle representative modulo 2*pi; results near "
    "+/-pi have the usual branch and direction ambiguity, and an unsigned "
    "amplitude cannot distinguish a vector from its negative"
)
COMPARISON_SEMANTICS = (
    "The continuum oracle is exp(lambda*T) acting on the retained complex "
    "vector. The synchronous run is a full-state RK4 reference at the same h "
    "or h/2; T0 is not asserted to equal the synchronous RK4 polynomial."
)
HARNESS_FILES = (
    "pyproject.toml",
    "uv.lock",
    "tools/warp_t0_verify.py",
    "docs/pseudocode/tools/warp_t0_verify.py.md",
    "tests/warp/test_warp_t0_verify.py",
    "docs/pseudocode/tests/warp/test_warp_t0_verify.py.md",
    "tests/warp/massless_verification_fixture.py",
    "docs/pseudocode/tests/warp/massless_verification_fixture.py.md",
    "tests/warp/test_massless_support.py",
    "docs/pseudocode/tests/warp/test_massless_support.py.md",
)
ENVIRONMENT_KEYS = ("SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES")
METADATA_INTEGER_KEYS = (
    "complete_stage_evaluations",
    "full_rhs_stage_evaluations",
    "endpoint_full_evaluations",
    "alias_evaluations",
    "total_model_evaluations",
    "retained_quadrature_entries",
    "assemble_canonicalization_calls",
)
FIELD_ORDER = (
    "n_i",
    "mom_i",
    "p_i",
    "p_perp_e",
    "p_parallel_e",
    "B",
)
FIELD_GROUPS = {
    "P": ("n_i", "mom_i", "p_i"),
    "Z": ("p_perp_e", "p_parallel_e", "B"),
    "full_PZ": FIELD_ORDER,
}
FIELD_UNITS = {
    "n_i": "1/m^3",
    "mom_i": "kg/(m^2*s)",
    "p_i": "Pa",
    "p_perp_e": "Pa",
    "p_parallel_e": "Pa",
    "B": "T",
}


def _finite_float(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _integer_sequence(name: str, values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty integer sequence")
    try:
        result = tuple(values)
    except TypeError as error:
        raise ValueError(
            f"{name} must be a nonempty integer sequence"
        ) from error
    if not result:
        raise ValueError(f"{name} must be nonempty")
    for value in result:
        _positive_integer(name, value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


def _float_sequence(name: str, values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty numeric sequence")
    try:
        items = tuple(values)
    except TypeError as error:
        raise ValueError(
            f"{name} must be a nonempty numeric sequence"
        ) from error
    if not items:
        raise ValueError(f"{name} must be nonempty")
    result = tuple(
        _finite_float(f"{name}[{index}]", value, positive=True)
        for index, value in enumerate(items)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


def normalize_controls(
    *,
    device: str = "cpu",
    kdi: Sequence[float] = DEFAULT_KDIS,
    points: int = 9,
    relative_amplitude: float = 2.0e-6,
    cycles: float = 1.0,
    micro_steps_per_cycle: int = 64,
    micro_steps: Sequence[int] = DEFAULT_MICRO_STEPS,
    repetitions: int = 1,
    fine_reference: bool = True,
    closed_only: bool = False,
    periodic_only: bool = False,
    closed_micro_steps: int = 4,
    closed_macro_dt: float = 2.0e-10,
    closed_horizon: float = 8.0e-10,
) -> dict[str, Any]:
    """Validate all controls before output or runtime state is created."""
    if device not in VALID_DEVICES:
        raise ValueError(f"device must be one of {VALID_DEVICES}")
    if closed_only and periodic_only:
        raise ValueError("closed_only and periodic_only are mutually exclusive")
    if isinstance(points, bool) or not isinstance(points, int):
        raise ValueError("points must be an odd integer >= 9")
    if points < 9 or points % 2 == 0:
        raise ValueError("points must be an odd integer >= 9")
    if micro_steps_per_cycle not in VALID_MICRO_STEPS_PER_CYCLE:
        raise ValueError(
            "micro_steps_per_cycle must be 64 or 128"
        )
    relative_amplitude = _finite_float(
        "relative_amplitude", relative_amplitude, positive=True
    )
    if relative_amplitude > 1.0e-3:
        raise ValueError("relative_amplitude must not exceed 1e-3")
    cycles = _finite_float("cycles", cycles, positive=True)
    total_micro = micro_steps_per_cycle * cycles
    total_micro_steps = int(round(total_micro))
    if not math.isclose(
        total_micro,
        total_micro_steps,
        rel_tol=0.0,
        abs_tol=1.0e-12 * max(1.0, total_micro),
    ):
        raise ValueError(
            "micro_steps_per_cycle*cycles must be an integer count"
        )
    micro_steps = _integer_sequence("micro_steps", micro_steps)
    if any(value > total_micro_steps for value in micro_steps):
        raise ValueError("each micro_steps value must fit inside the horizon")
    for value in micro_steps:
        if total_micro_steps % value:
            raise ValueError(
                "micro_steps must divide the total microstep count"
            )
    repetitions = _positive_integer("repetitions", repetitions)
    closed_micro_steps = _positive_integer(
        "closed_micro_steps", closed_micro_steps
    )
    closed_macro_dt = _finite_float(
        "closed_macro_dt", closed_macro_dt, positive=True
    )
    closed_horizon = _finite_float(
        "closed_horizon", closed_horizon, positive=True
    )
    closed_macros = closed_horizon / closed_macro_dt
    closed_macro_count = int(round(closed_macros))
    if closed_macro_count <= 0 or not math.isclose(
        closed_macros,
        closed_macro_count,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "closed_horizon/closed_macro_dt must be a positive integer"
        )
    kdi_values = _float_sequence("kdi", kdi)
    return {
        "device": device,
        "kdi": list(kdi_values),
        "points": points,
        "relative_amplitude": relative_amplitude,
        "cycles": cycles,
        "micro_steps_per_cycle": micro_steps_per_cycle,
        "micro_steps": list(micro_steps),
        "total_micro_steps": total_micro_steps,
        "repetitions": repetitions,
        "fine_reference": bool(fine_reference),
        "mode": "closed" if closed_only else "periodic",
        "closed_micro_steps": closed_micro_steps,
        "closed_macro_dt": closed_macro_dt,
        "closed_horizon": closed_horizon,
        "closed_macro_count": closed_macro_count,
    }


def _require_device(device: str, environment: Mapping[str, str] | None = None) -> str:
    environment = os.environ if environment is None else environment
    if device == "cpu":
        return device
    if device != "cuda:0":
        raise ValueError(f"unsupported device {device!r}")
    if not environment.get("SLURM_JOB_ID"):
        raise ValueError(
            "CUDA execution requires an existing Slurm allocation "
            "(SLURM_JOB_ID is unset)"
        )
    return device


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    if isinstance(value, complex):
        return {
            "real": _json_value(value.real),
            "imag": _json_value(value.imag),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "value") and hasattr(value, "name"):
        return _json_value(value.value)
    if hasattr(value, "__dict__"):
        return _json_value(vars(value))
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_json_value(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _save_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    cleaned: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            array = value
        elif isinstance(value, np.generic) or np.isscalar(value):
            array = np.asarray(value)
        elif hasattr(value, "get") and hasattr(value, "shape"):
            array = np.asarray(value.get())
        elif isinstance(value, (list, tuple)):
            array = np.asarray(value)
        else:
            continue
        if array.dtype.hasobject:
            raise TypeError(
                f"{name} has object dtype; evidence arrays must be pickle-free"
            )
        cleaned[name] = array
    if not cleaned:
        cleaned["empty"] = np.empty(0, dtype=np.float64)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **cleaned)


def check_summary(records: Iterable[Any]) -> dict[str, Any]:
    """Count all checks and group repeated failed records by identity."""
    summary = {
        "total": 0,
        "hard_total": 0,
        "hard_failed": 0,
        "soft_failed": 0,
        "unavailable": 0,
        "failed_records": [],
    }
    unique_failures: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        summary["total"] += 1
        hard = bool(getattr(record, "hard", False))
        available = bool(getattr(record, "available", False))
        passed = bool(getattr(record, "passed", False))
        if hard:
            summary["hard_total"] += 1
        if not available:
            summary["unavailable"] += 1
        if hard and (not available or not passed):
            summary["hard_failed"] += 1
        elif not hard and (not available or not passed):
            summary["soft_failed"] += 1
        if not available or not passed:
            norm = getattr(record, "norm", "")
            key = (
                getattr(record, "name", ""),
                getattr(record, "population", ""),
                getattr(record, "component", ""),
                getattr(record, "region", ""),
                getattr(record, "unit", ""),
                getattr(record, "value_unit", ""),
                getattr(norm, "value", str(norm)),
                hard,
                available,
                passed,
            )
            if key in unique_failures:
                unique_failures[key]["count"] += 1
            else:
                item = _json_value(record)
                item["count"] = 1
                unique_failures[key] = item
    summary["failed_records"] = list(unique_failures.values())
    return summary


def _merge_check_summaries(
    target: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    for name in (
        "total",
        "hard_total",
        "hard_failed",
        "soft_failed",
        "unavailable",
    ):
        target[name] += int(source[name])
    by_key = {
        (
            record.get("name", ""),
            record.get("population", ""),
            record.get("component", ""),
            record.get("region", ""),
            record.get("unit", ""),
            record.get("value_unit", ""),
            record.get("norm", ""),
            record.get("hard", False),
            record.get("available", False),
            record.get("passed", False),
        ): record
        for record in target["failed_records"]
    }
    for record in source["failed_records"]:
        key = (
            record.get("name", ""),
            record.get("population", ""),
            record.get("component", ""),
            record.get("region", ""),
            record.get("unit", ""),
            record.get("value_unit", ""),
            record.get("norm", ""),
            record.get("hard", False),
            record.get("available", False),
            record.get("passed", False),
        )
        if key in by_key:
            by_key[key]["count"] += int(record.get("count", 1))
        else:
            copied = dict(record)
            target["failed_records"].append(copied)
            by_key[key] = copied
    return target


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise Failure("Metadata", "coupling metadata is not a mapping")


def aggregate_coupling_metadata(
    records: Sequence[Any],
    *,
    coupling: str,
    micro_steps: int,
) -> dict[str, Any]:
    """Validate and sum the actual metadata emitted by every macro step."""
    if coupling != "predictor":
        raise Failure(
            "Metadata", "predictor metadata aggregation requires predictor"
        )
    expected = {
        "complete_stage_evaluations": 4 * micro_steps + 1,
        "full_rhs_stage_evaluations": 4 * micro_steps,
        "endpoint_full_evaluations": 1,
        "alias_evaluations": 2 * micro_steps,
        "total_model_evaluations": 6 * micro_steps + 1,
        "retained_quadrature_entries": 4 * micro_steps,
        "assemble_canonicalization_calls": 4 * micro_steps,
    }
    aggregate: dict[str, Any] = {
        name: 0 for name in METADATA_INTEGER_KEYS
    }
    aggregate["accepted_macro_steps"] = len(records)
    aggregate["expected_per_macro"] = dict(expected)
    aggregate["estimated_peak_bytes"] = 0
    aggregate["endpoint_tangent_correction"] = None
    aggregate["last_macro_dt"] = None
    aggregate["last_micro_dt"] = None
    for record in records:
        metadata = _mapping(record)
        if metadata.get("coupling") != "predictor":
            raise Failure("Metadata", "coupling metadata is not predictor")
        if metadata.get("micro_steps") != micro_steps:
            raise Failure("Metadata", "micro_steps metadata mismatch")
        for name, expected_value in expected.items():
            actual = metadata.get(name)
            if actual != expected_value:
                raise Failure(
                    "Metadata",
                    f"{name} mismatch: expected {expected_value}, got {actual}",
                )
            aggregate[name] += int(actual)
        peak = metadata.get("estimated_peak_bytes")
        if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
            raise Failure("Metadata", "estimated_peak_bytes is invalid")
        aggregate["estimated_peak_bytes"] = max(
            aggregate["estimated_peak_bytes"], peak
        )
        correction = metadata.get("endpoint_tangent_correction")
        if not isinstance(correction, Mapping):
            raise Failure(
                "Metadata", "endpoint_tangent_correction is unavailable"
            )
        aggregate["endpoint_tangent_correction"] = _json_value(correction)
        aggregate["last_macro_dt"] = metadata.get("macro_dt")
        aggregate["last_micro_dt"] = metadata.get("micro_dt")
    if not records:
        raise Failure("Metadata", "predictor run emitted no macro metadata")
    aggregate["memory_scope"] = (
        "estimated_peak_bytes is the solver preflight estimate for one macro; "
        "CuPy pool reserved bytes are not live peak memory"
    )
    return aggregate


def analytic_mode(
    initial: np.ndarray,
    eigenvalue: complex,
    horizon: float,
) -> np.ndarray:
    """Return the independent continuum continuation exp(lambda*T)."""
    values = np.asarray(initial, dtype=np.complex128)
    horizon = _finite_float("horizon", horizon)
    return values * np.exp(complex(eigenvalue) * horizon)


def modal_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """Return relative vector error, projected amplitude and signed phase."""
    observed = np.asarray(actual, dtype=np.complex128).reshape(-1)
    expected = np.asarray(reference, dtype=np.complex128).reshape(-1)
    if observed.shape != expected.shape:
        raise ValueError("modal vectors must have the same shape")
    expected_norm_squared = float(np.vdot(expected, expected).real)
    if not math.isfinite(expected_norm_squared) or expected_norm_squared <= 0.0:
        raise ValueError("reference modal vector must be finite and nonzero")
    projection = complex(np.vdot(expected, observed)) / expected_norm_squared
    reference_norm = math.sqrt(expected_norm_squared)
    return {
        "relative_vector_error": float(
            np.linalg.norm(observed - expected) / reference_norm
        ),
        "amplitude_ratio": float(abs(projection)),
        "signed_phase_rad": float(math.atan2(projection.imag, projection.real)),
        "principal_range_rad": [-math.pi, math.pi],
        "phase_convention": PHASE_CONVENTION,
        "principal_ambiguity": PRINCIPAL_AMBIGUITY,
    }


def normalized_field_errors(
    actual_fields: Mapping[str, np.ndarray],
    reference_fields: Mapping[str, np.ndarray],
    baseline_fields: Mapping[str, np.ndarray],
    *,
    physical_scales: Mapping[str, float],
    field_order: Sequence[str] = FIELD_ORDER,
    units: Mapping[str, str] = FIELD_UNITS,
    field_groups: Mapping[str, Sequence[str]] = FIELD_GROUPS,
    floor_semantics: str | None = None,
) -> dict[str, Any]:
    """Return per-SI-field errors and unit-invariant P/Z aggregates."""
    missing = [
        name
        for name in field_order
        if name not in actual_fields
        or name not in reference_fields
        or name not in baseline_fields
        or name not in physical_scales
        or name not in units
    ]
    if missing:
        raise ValueError(f"missing independent fields: {missing}")
    per_field: dict[str, dict[str, Any]] = {}
    for name in field_order:
        actual = np.asarray(actual_fields[name], dtype=np.complex128).reshape(-1)
        reference = np.asarray(
            reference_fields[name], dtype=np.complex128
        ).reshape(-1)
        baseline = np.asarray(
            baseline_fields[name], dtype=np.complex128
        ).reshape(-1)
        if not (actual.shape == reference.shape == baseline.shape):
            raise ValueError(f"{name} endpoint field shapes differ")
        floor = _finite_float(
            f"physical_scales[{name}]",
            physical_scales[name],
            positive=True,
        )
        absolute_error = float(np.linalg.norm(actual - reference))
        reference_perturbation = float(np.linalg.norm(reference - baseline))
        scale = max(reference_perturbation, floor)
        finite_values = (
            math.isfinite(absolute_error)
            and math.isfinite(reference_perturbation)
            and math.isfinite(scale)
        )
        relative_error = (
            absolute_error / scale
            if finite_values and scale > 0.0
            else math.inf
        )
        per_field[name] = {
            "unit": units[name],
            "absolute_error": absolute_error,
            "reference_perturbation_norm": reference_perturbation,
            "physical_amplitude_floor": floor,
            "scale": scale,
            "relative_error": float(relative_error),
            "limited_by_amplitude_floor": reference_perturbation <= floor,
            "scale_definition": (
                "max(||reference perturbation||_2, supplied physical "
                "reference or amplitude floor); background is removed"
            ),
        }

    aggregates: dict[str, dict[str, Any]] = {}
    for label, names in field_groups.items():
        unknown = [name for name in names if name not in per_field]
        if unknown:
            raise ValueError(f"{label} aggregate has unknown fields: {unknown}")
        values = np.asarray(
            [per_field[name]["relative_error"] for name in names],
            dtype=np.float64,
        )
        rms = (
            float(math.sqrt(float(np.mean(values**2))))
            if np.all(np.isfinite(values))
            else math.inf
        )
        aggregates[label] = {
            "dimensionless_rms_error": rms,
            "field_names": list(names),
            "field_count": len(names),
            "field_relative_errors": {
                name: per_field[name]["relative_error"] for name in names
            },
            "definition": (
                "dimensionless root mean square of per-field relative errors "
                "after each SI field is divided by its own max(reference "
                "perturbation, physical amplitude floor); no mixed-unit raw "
                "concatenation"
            ),
        }
    return {
        "per_field": per_field,
        "aggregates": aggregates,
        "semantics": (
            "Per-field absolute errors retain SI units. Zero cold-linear "
            f"{floor_semantics or 'components use the explicit n0*a, mi*n0*vA*a, pressure*a, or B0*a floor'}. "
            "Aggregates are dimensionless RMS values, never norms of "
            "concatenated raw units."
        ),
    }


def _host_array(context: Any, value: Any) -> np.ndarray:
    return np.asarray(context.compute.to_host(value), dtype=np.complex128).copy()


def _independent_fields(state: Any, context: Any) -> dict[str, np.ndarray]:
    ion = state.fluid[0]
    electron = state.fluid[1]
    return {
        "n_i": _host_array(context, ion.n).reshape(-1),
        "mom_i": _host_array(context, ion.mom).reshape(-1),
        "p_i": _host_array(context, ion.perp).reshape(-1),
        "p_perp_e": _host_array(context, electron.perp).reshape(-1),
        "p_parallel_e": _host_array(context, electron.parallel).reshape(-1),
        "B": _host_array(context, state.B).reshape(-1),
    }


def compatibility_fields(state: Any, context: Any) -> dict[str, np.ndarray]:
    """Return derived electron density and compatibility momentum."""
    electron = state.fluid[1]
    return {
        "n_e": _host_array(context, electron.n).reshape(-1),
        "mom_e": _host_array(context, electron.mom).reshape(-1),
    }


def _endpoint_arrays(state: Any, context: Any) -> dict[str, np.ndarray]:
    return {
        "ion_n": _host_array(context, state.fluid[0].n),
        "ion_momentum": _host_array(context, state.fluid[0].mom),
        "ion_perp": _host_array(context, state.fluid[0].perp),
        "electron_perp": _host_array(context, state.fluid[1].perp),
        "electron_parallel": _host_array(context, state.fluid[1].parallel),
        "magnetic_field": _host_array(context, state.B),
        "time_s": np.asarray(state.time, dtype=np.float64),
        "step": np.asarray(state.step, dtype=np.int64),
        "generation": np.asarray(state.generation, dtype=np.uint64),
    }


def _mode_coordinates(points: int) -> tuple[np.ndarray, np.ndarray]:
    coordinate = (
        2.0
        * math.pi
        * np.arange(points, dtype=np.float64)
        / float(points)
    )
    return np.cos(coordinate), np.sin(coordinate)


def mode_samples(
    vector: np.ndarray,
    *,
    amplitude: float,
    factor: complex,
    points: int,
) -> np.ndarray:
    """Return the real cosine/sine sample realization of a complex mode."""
    vector = np.asarray(vector, dtype=np.complex128).reshape(4)
    coordinate = (
        2.0
        * math.pi
        * np.arange(points, dtype=np.float64)
        / float(points)
    )
    coefficient = float(amplitude) * complex(factor) * vector
    return (
        np.real(coefficient)[:, None] * np.cos(coordinate)[None, :]
        - np.imag(coefficient)[:, None] * np.sin(coordinate)[None, :]
    )


def _project_mode(state: Any, case: HallCase, context: Any) -> np.ndarray:
    """Project four modal coefficients using device reductions only."""
    xp = context.compute.xp
    points = case.points
    cosine = xp.asarray(_mode_coordinates(points)[0], dtype=xp.float64)
    sine = xp.asarray(_mode_coordinates(points)[1], dtype=xp.float64)
    momentum = xp.asarray(state.fluid[0].mom).reshape(3, points)
    magnetic = xp.asarray(state.B).reshape(3, points)
    momentum_scale = (
        float(context.p.species[0].mass) * case.n0 * case.v_alfven
    )
    coefficients = []
    for values, scale in (
        (momentum[0], momentum_scale),
        (momentum[1], momentum_scale),
        (magnetic[0], case.B0),
        (magnetic[1], case.B0),
    ):
        real = xp.real(values)
        cosine_value = (
            2.0 * xp.sum(real * cosine, dtype=xp.float64) / float(points)
        )
        sine_value = (
            2.0 * xp.sum(real * sine, dtype=xp.float64) / float(points)
        )
        coefficients.append(
            (cosine_value - 1j * sine_value) / float(scale)
        )
    device_values = xp.stack(coefficients)
    return np.asarray(
        context.compute.to_host(device_values),
        dtype=np.complex128,
    )


def _periodic_mode_delta(
    context: Any,
    case: HallCase,
    vector: np.ndarray,
    relative_amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    nc = case.points
    shape = mode_samples(
        vector,
        amplitude=relative_amplitude,
        factor=1.0,
        points=nc,
    )
    zero_n = np.zeros(nc, dtype=np.complex128)
    zero_mom = np.zeros(3 * nc, dtype=np.complex128)
    zero_pressure = np.zeros(nc, dtype=np.complex128)
    zero_b = np.zeros(3 * nc, dtype=np.complex128)
    velocity_scale = (
        float(context.p.species[0].mass) * case.n0 * case.v_alfven
    )
    zero_mom[:nc] = velocity_scale * shape[0]
    zero_mom[nc : 2 * nc] = velocity_scale * shape[1]
    zero_b[:nc] = case.B0 * shape[2]
    zero_b[nc : 2 * nc] = case.B0 * shape[3]
    p_delta = np.concatenate((zero_n, zero_mom, zero_pressure))
    z_delta = np.concatenate(
        (zero_pressure, zero_pressure, zero_b)
    )
    return p_delta, z_delta


def _metadata_support_available() -> bool:
    try:
        importlib.import_module("fomhd.time_predictor")
    except Exception:
        return False
    return hasattr(advance, "__module__")


def _build_periodic_trajectory(
    case: HallCase,
    *,
    device: str,
    coupling: str,
    micro_steps: int,
    macro_dt: float,
    macro_count: int,
    relative_amplitude: float,
) -> SimpleNamespace:
    if coupling == "predictor" and not _metadata_support_available():
        raise Failure(
            "ToolUnavailable",
            "the installed fomhd package does not provide the A1-T0 "
            "predictor scheduler",
        )
    context = build_periodic_context(case, device=device)
    _verify_backend_identity(context, device)
    p = context.p
    p.integrator = "rk4"
    p.coupling = coupling
    p.micro_steps = micro_steps
    p.dt = macro_dt
    p.nsteps = macro_count
    p.dt_guard = "warn"
    raw, vector, eigenvalue = initial_hall_eigenmode(
        case,
        relative_amplitude=relative_amplitude,
        context=context,
        device=device,
    )
    captured: dict[str, Any] = {}
    cumulative = cumulative_record()
    committed = initialize(
        raw,
        context,
        cumulative,
        False,
        captured,
    )
    baseline = massless.canonicalize_state(
        uniform_state(context, case),
        context,
    )
    initial_mode = _project_mode(committed.state, case, context)
    initial_checks = list(captured.get("checks", ()))
    return SimpleNamespace(
        context=context,
        committed=committed,
        cumulative=cumulative,
        initial_mode=initial_mode,
        initial_checks=initial_checks,
        vector=np.asarray(vector, dtype=np.complex128).copy(),
        eigenvalue=complex(eigenvalue),
        baseline=baseline,
        coupling=coupling,
        micro_steps=micro_steps,
        macro_dt=macro_dt,
        macro_count=macro_count,
        requested_device=device,
    )


def _trajectory_result(
    trajectory: SimpleNamespace,
    *,
    project: Callable[[Any], np.ndarray] | None,
) -> dict[str, Any]:
    requested_device = getattr(trajectory, "requested_device", None)
    if requested_device is None:
        raise Failure(
            "BackendIdentity",
            "trajectory has no requested-device identity; no fallback",
        )
    _verify_backend_identity(trajectory.context, requested_device)
    checks = check_summary(trajectory.initial_checks)
    metadata_records: list[Mapping[str, Any]] = []
    times = [float(trajectory.committed.state.time)]
    mode_history: list[np.ndarray] = []
    if project is not None:
        mode_history.append(project(trajectory.committed.state))
    time_report = None
    warning_count = 0
    accepted = 0
    try:
        for _ in range(trajectory.macro_count):
            result = advance(
                trajectory.committed,
                trajectory.cumulative,
                trajectory.context,
                None,
            )
            accepted += 1
            _merge_check_summaries(checks, check_summary(result.checks))
            report = getattr(result, "time_step_report", None)
            if report is not None:
                time_report = _json_value(report)
                if not bool(report.get("accepted", False)):
                    warning_count += 1
            metadata = getattr(result, "coupling_metadata", None)
            if trajectory.coupling == "predictor":
                if metadata is None:
                    raise Failure(
                        "Metadata",
                        "predictor result has no coupling_metadata",
                    )
                metadata_records.append(_mapping(metadata))
            times.append(float(trajectory.committed.state.time))
            if project is not None:
                mode_history.append(project(trajectory.committed.state))
        if accepted != trajectory.macro_count:
            raise Failure("Time", "trajectory did not accept the full horizon")
    except Exception as error:
        setattr(error, "tool_trajectory", trajectory)
        setattr(error, "tool_progress", {
            "accepted_macro_steps": accepted,
            "check_summary": checks,
            "time_history_s": times,
            "mode_history": np.asarray(mode_history),
            "last_time_step_report": time_report,
        })
        if isinstance(error, Failure):
            partial = getattr(error, "partial_arrays", None)
            if partial is None:
                partial = {}
                error.partial_arrays = partial
            partial.setdefault("tool_mode_history", np.asarray(mode_history))
            partial.setdefault("tool_time_history_s", np.asarray(times))
        raise
    if trajectory.coupling == "predictor":
        coupling_metadata = aggregate_coupling_metadata(
            metadata_records,
            coupling="predictor",
            micro_steps=trajectory.micro_steps,
        )
    else:
        coupling_metadata = {
            "coupling": "synchronous",
            "micro_steps": 1,
            "accepted_macro_steps": accepted,
            "metadata_scope": "A0-compatible synchronous RK4",
        }
    return {
        "accepted_macro_steps": accepted,
        "checks": checks,
        "coupling_metadata": coupling_metadata,
        "time_history_s": np.asarray(times, dtype=np.float64),
        "mode_history": np.asarray(mode_history, dtype=np.complex128),
        "time_estimate_warning_count": warning_count,
        "last_time_estimate_report": time_report,
    }


def _memory_snapshot() -> dict[str, Any]:
    status: dict[str, str] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if ":" in line:
                name, value = line.split(":", 1)
                if name in ("VmRSS", "VmHWM", "VmPeak"):
                    status[name] = value.strip()
    except OSError:
        status = {}

    def kib_to_bytes(key: str) -> int | None:
        value = status.get(key)
        if value is None:
            return None
        try:
            return int(value.split()[0]) * 1024
        except (IndexError, ValueError):
            return None

    pool: dict[str, Any] = {
        "available": False,
        "reserved_is_not_live_peak": True,
    }
    if importlib.util.find_spec("cupy") is not None:
        try:
            cupy = importlib.import_module("cupy")
            memory_pool = cupy.get_default_memory_pool()
            pool = {
                "available": True,
                "reserved_bytes": int(memory_pool.total_bytes()),
                "live_bytes_at_end": int(memory_pool.used_bytes()),
                "reserved_is_not_live_peak": True,
            }
        except Exception as error:
            pool["reason"] = f"{type(error).__name__}: {error}"
    return {
        "process_rss_bytes": kib_to_bytes("VmRSS"),
        "process_rss_hwm_bytes": kib_to_bytes("VmHWM"),
        "process_vm_peak_bytes": kib_to_bytes("VmPeak"),
        "cupy_pool": pool,
        "scope": (
            "process-wide observations, not isolated per-case peaks; "
            "process RSS is not allocation peak; CuPy pool reserved bytes "
            "are not live peak bytes"
        ),
    }


def _execution_memory(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = _memory_snapshot()
    result["estimated_peak_bytes"] = metadata.get("estimated_peak_bytes")
    return result


def _result_status(checks: Mapping[str, Any], accepted: int, expected: int) -> str:
    if accepted != expected or int(checks["hard_failed"]) > 0:
        return "failed"
    if int(checks["soft_failed"]) > 0:
        return "measured_with_soft_failures"
    return "measured"


def _compose_periodic_result(
    case: HallCase,
    trajectory: SimpleNamespace,
    execution: Mapping[str, Any],
    *,
    case_id: str,
    relative_amplitude: float,
    macro_dt: float,
    macro_count: int,
) -> dict[str, Any]:
    backend_identity = _verify_backend_identity(
        trajectory.context,
        trajectory.requested_device,
    )
    endpoint = trajectory.committed.state
    actual_fields = _independent_fields(endpoint, trajectory.context)
    baseline_fields = _independent_fields(
        trajectory.baseline, trajectory.context
    )
    horizon = float(endpoint.time)
    factor = np.exp(trajectory.eigenvalue * horizon)
    analytic_p_delta, analytic_z_delta = _periodic_mode_delta(
        trajectory.context,
        case,
        trajectory.vector * factor,
        relative_amplitude,
    )
    n_count = baseline_fields["n_i"].size
    momentum_count = baseline_fields["mom_i"].size
    z_pressure_count = baseline_fields["p_perp_e"].size
    z_parallel_count = baseline_fields["p_parallel_e"].size
    analytic_fields = {
        "n_i": baseline_fields["n_i"] + analytic_p_delta[:n_count],
        "mom_i": baseline_fields["mom_i"]
        + analytic_p_delta[n_count : n_count + momentum_count],
        "p_i": baseline_fields["p_i"]
        + analytic_p_delta[n_count + momentum_count :],
        "p_perp_e": baseline_fields["p_perp_e"]
        + analytic_z_delta[:z_pressure_count],
        "p_parallel_e": baseline_fields["p_parallel_e"]
        + analytic_z_delta[
            z_pressure_count : z_pressure_count + z_parallel_count
        ],
        "B": baseline_fields["B"]
        + analytic_z_delta[z_pressure_count + z_parallel_count :],
    }
    expected_mode = relative_amplitude * trajectory.vector * factor
    actual_mode = execution["mode_history"][-1]
    initial_reference = relative_amplitude * trajectory.vector
    analytic_metrics = modal_metrics(actual_mode, expected_mode)
    field_scales = {
        "n_i": case.n0 * relative_amplitude,
        "mom_i": (
            float(trajectory.context.p.species[0].mass)
            * case.n0
            * case.v_alfven
            * relative_amplitude
        ),
        "p_i": case.pressure * relative_amplitude,
        "p_perp_e": case.pressure * relative_amplitude,
        "p_parallel_e": case.pressure * relative_amplitude,
        "B": case.B0 * relative_amplitude,
    }
    field_errors = normalized_field_errors(
        actual_fields,
        analytic_fields,
        baseline_fields,
        physical_scales=field_scales,
    )
    checks = execution["checks"]
    status = _result_status(
        checks,
        int(execution["accepted_macro_steps"]),
        macro_count,
    )
    expected_counts = execution["coupling_metadata"].get(
        "expected_per_macro"
    )
    if trajectory.coupling == "predictor":
        actual_counts = {
            name: execution["coupling_metadata"].get(name, 0)
            for name in METADATA_INTEGER_KEYS
        }
    else:
        actual_counts = {
            "available": False,
            "reason": (
                "A0-compatible synchronous results do not emit predictor "
                "coupling metadata"
            ),
        }
    if expected_counts is not None:
        expected_aggregate = {
            name: value * macro_count
            for name, value in expected_counts.items()
        }
        if actual_counts != expected_aggregate:
            raise Failure(
                "Metadata",
                "aggregate predicted stage counts do not match the horizon",
            )
    metadata = execution["coupling_metadata"]
    summary = {
        "case_id": case_id,
        "kind": (
            "periodic_t0_predictor"
            if trajectory.coupling == "predictor"
            else "periodic_synchronous_reference"
        ),
        "status": status,
        "scope": PERIODIC_SCOPE,
        "adapter": "periodic temporal adapter; not closed-wall validation",
        "device": str(trajectory.context.compute.name),
        "requested_device": backend_identity["requested"],
        "actual_device": backend_identity["actual"],
        "backend_identity": backend_identity,
        "kdi": case.kdi,
        "points": case.points,
        "relative_amplitude": relative_amplitude,
        "coupling": trajectory.coupling,
        "requested_micro_steps": trajectory.micro_steps,
        "accepted_macro_steps": int(execution["accepted_macro_steps"]),
        "requested_macro_steps": macro_count,
        "macro_dt_s": macro_dt,
        "micro_dt_s": macro_dt / trajectory.micro_steps,
        "horizon_s": horizon,
        "eigenvalue_rad_s": {
            "real": trajectory.eigenvalue.real,
            "imag": trajectory.eigenvalue.imag,
        },
        "omega_rad_s": trajectory.eigenvalue.imag,
        "initial_mode_relative_error": modal_metrics(
            execution["mode_history"][0], initial_reference
        )["relative_vector_error"],
        "endpoint_errors_vs_analytic": {
            "fields": field_errors["per_field"],
            "aggregates": field_errors["aggregates"],
            "semantics": field_errors["semantics"],
            "comparison_semantics": COMPARISON_SEMANTICS,
        },
        "field_physical_scales": field_scales,
        "endpoint_modal_metrics_vs_analytic": analytic_metrics,
        "actual_coupling_metadata": metadata,
        "actual_stage_counts": actual_counts,
        "time_estimate_warning_count": execution[
            "time_estimate_warning_count"
        ],
        "last_time_estimate_report": execution[
            "last_time_estimate_report"
        ],
        "check_summary": checks,
        "timing": {
            name: value
            for name, value in execution.items()
            if name.endswith("wall_seconds") or name == "timing_method"
        },
        "resources": _execution_memory(metadata),
        "comparison_semantics": COMPARISON_SEMANTICS,
    }
    return {
        "summary": summary,
        "arrays": {
            **{
                f"raw_{name}": value
                for name, value in actual_fields.items()
            },
            **{
                f"analytic_{name}": value
                for name, value in analytic_fields.items()
            },
            "mode_history": execution["mode_history"],
            "time_history_s": execution["time_history_s"],
            "initial_mode": execution["mode_history"][0],
            "analytic_endpoint_mode": expected_mode,
            **_endpoint_arrays(endpoint, trajectory.context),
        },
        "baseline": {"fields": baseline_fields},
        "fields": actual_fields,
        "analytic_fields": analytic_fields,
        "field_scales": field_scales,
        "eigenvalue": trajectory.eigenvalue,
        "endpoint_mode": actual_mode,
        "analytic_mode": expected_mode,
        "configuration": {
            "coupling": trajectory.coupling,
            "micro_steps": trajectory.micro_steps,
            "macro_dt_s": macro_dt,
            "macro_steps": macro_count,
            "horizon_s": horizon,
        },
    }


def _run_repetitions(
    *,
    build: Callable[[], SimpleNamespace],
    project: Callable[[Any], np.ndarray] | None,
    repetitions: int,
) -> dict[str, Any]:
    setup_started = time.perf_counter()
    warm = build()
    warm_setup = time.perf_counter() - setup_started
    warm.context.compute.synchronize()
    advance_started = time.perf_counter()
    _trajectory_result(warm, project=project)
    warm.context.compute.synchronize()
    warm_advance = time.perf_counter() - advance_started

    setup_times: list[float] = []
    advance_times: list[float] = []
    final_trajectory = None
    final_execution = None
    for _ in range(repetitions):
        started = time.perf_counter()
        trajectory = build()
        setup_times.append(time.perf_counter() - started)
        trajectory.context.compute.synchronize()
        started = time.perf_counter()
        execution = _trajectory_result(trajectory, project=project)
        trajectory.context.compute.synchronize()
        advance_times.append(time.perf_counter() - started)
        final_trajectory = trajectory
        final_execution = execution
    assert final_trajectory is not None and final_execution is not None
    final_execution.update(
        warmup_setup_wall_seconds=warm_setup,
        warmup_advance_wall_seconds=warm_advance,
        setup_wall_seconds=float(statistics.median(setup_times)),
        advance_wall_seconds=float(statistics.median(advance_times)),
        setup_wall_seconds_samples=list(setup_times),
        advance_wall_seconds_samples=list(advance_times),
        repetitions=repetitions,
        timing_method=(
            "separate fresh setup per repetition; warmup uses a distinct "
            "initialized trajectory; one synchronize before and after each "
            "timed trajectory region; includes advance, check aggregation, "
            "and modal history projection; median reported"
        ),
    )
    return {
        "trajectory": final_trajectory,
        "execution": final_execution,
    }


def run_periodic_case(
    case: HallCase,
    *,
    device: str,
    coupling: str,
    micro_steps: int,
    macro_dt: float,
    macro_count: int,
    relative_amplitude: float,
    repetitions: int,
    case_id: str,
) -> dict[str, Any]:
    """Run one fresh periodic trajectory from real solver initialization."""
    active_case = HallCase(
        kdi=case.kdi,
        points=case.points,
        identity=case.identity,
        n0=case.n0,
        B0=case.B0,
        pressure=case.pressure,
    )

    def build() -> SimpleNamespace:
        return _build_periodic_trajectory(
            active_case,
            device=device,
            coupling=coupling,
            micro_steps=micro_steps,
            macro_dt=macro_dt,
            macro_count=macro_count,
            relative_amplitude=relative_amplitude,
        )

    # A context-local closure avoids retaining state arrays across repetitions.
    projection_context: list[Any] = []

    def tracked_build() -> SimpleNamespace:
        trajectory = build()
        projection_context.clear()
        projection_context.append(trajectory.context)
        return trajectory

    def tracked_project(state: Any) -> np.ndarray:
        if not projection_context:
            raise Failure("Harness", "periodic projection context is unavailable")
        return _project_mode(state, active_case, projection_context[-1])

    try:
        repeated = _run_repetitions(
            build=tracked_build,
            project=tracked_project,
            repetitions=repetitions,
        )
        return _compose_periodic_result(
            active_case,
            repeated["trajectory"],
            repeated["execution"],
            case_id=case_id,
            relative_amplitude=relative_amplitude,
            macro_dt=macro_dt,
            macro_count=macro_count,
        )
    except Exception as error:
        if projection_context:
            setattr(error, "tool_context", projection_context[-1])
        setattr(error, "tool_case_configuration", {
            "case_id": case_id,
            "coupling": coupling,
            "micro_steps": micro_steps,
            "macro_dt_s": macro_dt,
            "macro_steps": macro_count,
            "relative_amplitude": relative_amplitude,
        })
        raise

def _compare_periodic_reference(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    field_errors = normalized_field_errors(
        candidate["fields"],
        reference["fields"],
        candidate["baseline"]["fields"],
        physical_scales=candidate["field_scales"],
    )
    return {
        "reference": label,
        "fields": field_errors["per_field"],
        "aggregates": field_errors["aggregates"],
        "modal_metrics": modal_metrics(
            candidate["endpoint_mode"],
            reference["endpoint_mode"],
        ),
        "field_semantics": field_errors["semantics"],
        "semantics": (
            "per-field endpoint comparison at an equal horizon; P/Z "
            "aggregates are dimensionless RMS values. The synchronous "
            "reference is RK4 at h or h/2, not a T0-equivalence or "
            "polynomial oracle"
        ),
    }


def _capture_failure_arrays(error: Exception) -> dict[str, np.ndarray]:
    partial = getattr(error, "partial_arrays", {}) or {}
    saved: dict[str, np.ndarray] = {}
    for name, value in partial.items():
        if value is None:
            continue
        try:
            if isinstance(value, np.ndarray):
                array = value
            elif isinstance(value, np.generic) or np.isscalar(value):
                array = np.asarray(value)
            elif hasattr(value, "get") and hasattr(value, "shape"):
                array = np.asarray(value.get())
            elif isinstance(value, (list, tuple)):
                array = np.asarray(value)
            else:
                continue
            if not array.dtype.hasobject:
                saved[str(name)] = array
        except Exception:
            continue
    return saved


def _write_failure(
    output: Path,
    case_id: str,
    error: Exception,
) -> dict[str, Any]:
    arrays = _capture_failure_arrays(error)
    array_path = None
    if arrays:
        filename = f"{case_id}-failure.npz"
        _save_npz(output / filename, arrays)
        array_path = filename
    checks = list(getattr(error, "checks", ()) or ())
    progress_value = getattr(error, "tool_progress", None)
    progress = None
    if isinstance(progress_value, Mapping):
        progress = {}
        for name, value in progress_value.items():
            if isinstance(value, np.ndarray):
                progress[name] = {
                    "saved_in_failure_npz": name in arrays,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            else:
                progress[name] = value
    configuration = getattr(error, "tool_case_configuration", None)
    payload = {
        "case_id": case_id,
        "status": "failed",
        "failure_code": getattr(error, "code", type(error).__name__),
        "error_type": type(error).__name__,
        "message": str(error),
        "failure_stage": _json_value(getattr(error, "stage", None)),
        "configuration": configuration,
        "progress": progress,
        "checks": checks,
        "partial_array_names": sorted(arrays),
        "partial_arrays_file": array_path,
        "traceback": traceback.format_exc(),
    }
    filename = f"{case_id}-failure.json"
    _write_json(output / filename, payload)
    return {
        "case_id": case_id,
        "status": "failed",
        "failure_code": payload["failure_code"],
        "message": str(error),
        "failure_record_file": filename,
        "failure_arrays_file": array_path,
        "partial_array_names": sorted(arrays),
        "check_summary": check_summary(checks),
    }


def _success_artifacts(
    output: Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    summary = dict(result["summary"])
    case_id = summary["case_id"]
    arrays_file = f"{case_id}.npz"
    _save_npz(output / arrays_file, result["arrays"])
    metadata_file = f"{case_id}-metadata.json"
    _write_json(
        output / metadata_file,
        {
            "case_id": case_id,
            "actual_coupling_metadata": summary["actual_coupling_metadata"],
            "timing": summary["timing"],
            "resources": summary["resources"],
            "configuration": result["configuration"],
        },
    )
    checks_file = f"{case_id}-failed-checks.json"
    _write_json(
        output / checks_file,
        {
            "case_id": case_id,
            "summary": summary["check_summary"],
            "failed_records": summary["check_summary"]["failed_records"],
        },
    )
    summary.update(
        endpoint_arrays_file=arrays_file,
        metadata_file=metadata_file,
        failed_checks_file=checks_file,
    )
    return summary


def _execute_case(
    output: Path,
    case_id: str,
    runner: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        result = runner()
    except Exception as error:
        return _write_failure(output, case_id, error), None
    return _success_artifacts(output, result), result


def _write_report(output: Path, report: Mapping[str, Any]) -> None:
    target = output / "report.json"
    if target.exists():
        target.unlink()
    _write_json(target, report)


def _annotate_periodic_comparisons(
    t0_result: Mapping[str, Any],
    sync_result: Mapping[str, Any] | None,
    fine_result: Mapping[str, Any] | None,
) -> None:
    summary = t0_result["summary"]
    summary["endpoint_errors_vs_synchronous_h"] = (
        _compare_periodic_reference(
            t0_result,
            sync_result,
            label="synchronous RK4 at h",
        )
        if sync_result is not None
        else {"available": False, "reason": "synchronous reference failed"}
    )
    summary["endpoint_errors_vs_fine_reference_h_over_2"] = (
        _compare_periodic_reference(
            t0_result,
            fine_result,
            label="synchronous RK4 at h/2",
        )
        if fine_result is not None
        else {
            "available": False,
            "reason": (
                "fine reference was disabled or failed; no h/2 endpoint "
                "comparison is available"
            ),
        }
    )
    if sync_result is not None:
        summary["synchronous_reference"] = (
            _compare_periodic_reference(
                sync_result,
                sync_result,
                label="self-consistency only",
            )
        )


def run_periodic_suite(
    output: Path,
    controls: Mapping[str, Any],
    *,
    report: dict[str, Any],
) -> None:
    repetitions = int(controls["repetitions"])
    for kdi in controls["kdi"]:
        case = HallCase(kdi=float(kdi), points=int(controls["points"]))
        omega = hall_frequencies(case).omega_plus
        period = 2.0 * math.pi / omega
        h = period / float(controls["micro_steps_per_cycle"])
        total_micro = int(controls["total_micro_steps"])
        stem = f"periodic_kdi{kdi:.6g}"
        sync_id = f"{stem}_sync_h"
        sync_report, sync_raw = _execute_case(
            output,
            sync_id,
            lambda: run_periodic_case(
                case,
                device=controls["device"],
                coupling="synchronous",
                micro_steps=1,
                macro_dt=h,
                macro_count=total_micro,
                relative_amplitude=controls["relative_amplitude"],
                repetitions=repetitions,
                case_id=sync_id,
            ),
        )
        report["cases"].append(sync_report)
        _write_report(output, report)

        fine_report = None
        fine_raw = None
        if controls["fine_reference"]:
            fine_id = f"{stem}_sync_h_over_2"
            fine_report, fine_raw = _execute_case(
                output,
                fine_id,
                lambda: run_periodic_case(
                    case,
                    device=controls["device"],
                    coupling="synchronous",
                    micro_steps=1,
                    macro_dt=h / 2.0,
                    macro_count=2 * total_micro,
                    relative_amplitude=controls["relative_amplitude"],
                    repetitions=repetitions,
                    case_id=fine_id,
                ),
            )
            report["cases"].append(fine_report)
            _write_report(output, report)

        if sync_raw is not None:
            sync_report["endpoint_errors_vs_analytic"] = sync_raw["summary"][
                "endpoint_errors_vs_analytic"
            ]
            if fine_raw is not None:
                sync_report["endpoint_errors_vs_fine_reference_h_over_2"] = (
                    _compare_periodic_reference(
                        sync_raw,
                        fine_raw,
                        label="synchronous RK4 at h/2",
                    )
                )

        for micro in controls["micro_steps"]:
            micro = int(micro)
            macro_count = total_micro // micro
            case_id = f"{stem}_t0_M{micro}"
            predictor_report, predictor_raw = _execute_case(
                output,
                case_id,
                lambda micro=micro, macro_count=macro_count, case_id=case_id: (
                    run_periodic_case(
                        case,
                        device=controls["device"],
                        coupling="predictor",
                        micro_steps=micro,
                        macro_dt=micro * h,
                        macro_count=macro_count,
                        relative_amplitude=controls["relative_amplitude"],
                        repetitions=repetitions,
                        case_id=case_id,
                    )
                ),
            )
            report["cases"].append(predictor_report)
            if predictor_raw is not None:
                _annotate_periodic_comparisons(
                    predictor_raw,
                    sync_raw,
                    fine_raw,
                )
                predictor_report.update(predictor_raw["summary"])
            _write_report(output, report)


def _verify_backend_identity(context: Any, requested_device: str) -> dict[str, Any]:
    actual_device = getattr(context.compute, "device", None)
    if actual_device != requested_device:
        raise Failure(
            "BackendIdentity",
            f"requested {requested_device}, constructed {actual_device}; "
            "no fallback",
        )
    actual_name = getattr(context.compute, "name", None)
    expected_name = "cpu" if requested_device == "cpu" else "cuda"
    if actual_name != expected_name:
        raise Failure(
            "BackendIdentity",
            f"requested {requested_device}, backend name {actual_name}; "
            "no fallback",
        )
    return {
        "requested": requested_device,
        "actual": actual_device,
        "backend_name": actual_name,
        "fallback_used": False,
    }


def build_closed_context(device: str) -> Context:
    """Build the small closed fixture on the explicitly selected backend."""
    p = closed_parameters(K=0, hard_diagnostics=True)
    nodes = 4
    hr = 0.6 / (nodes - 1)
    geometry = build_rectangle(nodes, nodes, 1.0, 0.0, hr, hr)
    backend = Backend(device)
    operators = build_operators(geometry, 0, backend=backend)
    context = Context(
        p=p,
        geometry=geometry,
        op=operators,
        compute=backend,
        N=p.Nphi,
        next_generation=2,
    )
    _verify_backend_identity(context, device)
    return context


def _closed_shape(context: Any) -> np.ndarray:
    r_min = float(np.min(context.geometry.cells.R))
    r_max = float(np.max(context.geometry.cells.R))
    z_min = float(np.min(context.geometry.cells.Z))
    z_max = float(np.max(context.geometry.cells.Z))
    r = (np.asarray(context.geometry.cells.R) - r_min) / (r_max - r_min)
    z = (np.asarray(context.geometry.cells.Z) - z_min) / (z_max - z_min)
    return np.sin(math.pi * r) ** 2 * np.sin(math.pi * z) ** 2


def perturb_closed_state(state: Any, context: Any) -> Any:
    xp = context.compute.xp
    nc = context.op.Nc
    shape = _closed_shape(context)
    physical_density = (
        float(context.p.ref.ni) * 2.0e-4 * shape.reshape(context.N, nc)
    )
    physical_pressure = (
        float(context.p.ref.pressure) * 2.0e-4
        * shape.reshape(context.N, nc)
    )
    density_mode = forward(
        xp.asarray(physical_density.reshape(-1)),
        nc,
        context,
    )
    pressure_mode = forward(
        xp.asarray(physical_pressure.reshape(-1)),
        nc,
        context,
    )
    state.fluid[0].n[:] = state.fluid[0].n + density_mode
    state.fluid[0].perp[:] = state.fluid[0].perp + pressure_mode
    state.fluid[1].perp[:] = state.fluid[1].perp + pressure_mode
    state.fluid[1].parallel[:] = (
        state.fluid[1].parallel + pressure_mode
    )
    return state


def _build_closed_trajectory(
    *,
    device: str,
    micro_steps: int,
    macro_dt: float,
    macro_count: int,
) -> SimpleNamespace:
    if not _metadata_support_available():
        raise Failure(
            "ToolUnavailable",
            "the installed fomhd package does not provide the A1-T0 "
            "predictor scheduler",
        )
    context = build_closed_context(device)
    p = context.p
    p.integrator = "rk4"
    p.coupling = "predictor"
    p.micro_steps = micro_steps
    p.dt = macro_dt
    p.nsteps = macro_count
    p.dt_guard = "warn"
    raw = perturb_closed_state(closed_equilibrium_state(context), context)
    captured: dict[str, Any] = {}
    cumulative = cumulative_record()
    committed = initialize(
        raw,
        context,
        cumulative,
        False,
        captured,
    )
    return SimpleNamespace(
        context=context,
        committed=committed,
        cumulative=cumulative,
        initial_mode=np.empty(0, dtype=np.complex128),
        initial_checks=list(captured.get("checks", ())),
        coupling="predictor",
        micro_steps=micro_steps,
        macro_dt=macro_dt,
        macro_count=macro_count,
        hard_diagnostics_requested=True,
        requested_device=device,
    )


def run_closed_case(
    *,
    device: str,
    micro_steps: int,
    macro_dt: float,
    macro_count: int,
    repetitions: int,
    case_id: str = "closed_smooth_t0",
) -> dict[str, Any]:
    def build() -> SimpleNamespace:
        return _build_closed_trajectory(
            device=device,
            micro_steps=micro_steps,
            macro_dt=macro_dt,
            macro_count=macro_count,
        )

    try:
        repeated = _run_repetitions(
            build=build,
            project=None,
            repetitions=repetitions,
        )
        trajectory = repeated["trajectory"]
        execution = repeated["execution"]
        backend_identity = _verify_backend_identity(
            trajectory.context,
            trajectory.requested_device,
        )
        endpoint = trajectory.committed.state
        canonical = massless.canonicalize_state(endpoint, trajectory.context)
        before_fields = _independent_fields(endpoint, trajectory.context)
        after_fields = _independent_fields(canonical, trajectory.context)
        before_compatibility = compatibility_fields(
            endpoint, trajectory.context
        )
        after_compatibility = compatibility_fields(
            canonical, trajectory.context
        )
        reference_scales = {
            "n_i": trajectory.context.p.ref.ni,
            "mom_i": (
                float(trajectory.context.p.species[0].mass)
                * trajectory.context.p.ref.ni
                * trajectory.context.p.ref.velocity
            ),
            "p_i": trajectory.context.p.ref.pressure,
            "p_perp_e": trajectory.context.p.ref.pressure,
            "p_parallel_e": trajectory.context.p.ref.pressure,
            "B": trajectory.context.p.ref.magnetic,
        }
        change = normalized_field_errors(
            after_fields,
            before_fields,
            before_fields,
            physical_scales=reference_scales,
        )
        compatibility_scales = {
            "n_e": trajectory.context.p.ref.ne,
            "mom_e": (
                float(trajectory.context.p.species[1].mass)
                * trajectory.context.p.ref.ne
                * trajectory.context.p.ref.velocity
            ),
        }
        compatibility_correction = normalized_field_errors(
            after_compatibility,
            before_compatibility,
            before_compatibility,
            physical_scales=compatibility_scales,
            field_order=("n_e", "mom_e"),
            units={
                "n_e": "1/m^3",
                "mom_e": "kg/(m^2*s)",
            },
            field_groups={"compatibility": ("n_e", "mom_e")},
            floor_semantics=(
                "derived electron compatibility fields use their SI "
                "reference density and momentum scales"
            ),
        )
        checks = execution["checks"]
        status = _result_status(
            checks,
            int(execution["accepted_macro_steps"]),
            macro_count,
        )
        metadata = execution["coupling_metadata"]
        summary = {
            "case_id": case_id,
            "kind": "closed_rectangle_smooth_t0",
            "status": status,
            "scope": CLOSED_SCOPE,
            "device": str(trajectory.context.compute.name),
            "requested_device": backend_identity["requested"],
            "actual_device": backend_identity["actual"],
            "backend_identity": backend_identity,
            "coupling": "predictor",
            "requested_micro_steps": micro_steps,
            "accepted_macro_steps": int(execution["accepted_macro_steps"]),
            "requested_macro_steps": macro_count,
            "macro_dt_s": macro_dt,
            "micro_dt_s": macro_dt / micro_steps,
            "horizon_s": float(endpoint.time),
            "hard_diagnostics_requested": True,
            "dt_guard": trajectory.context.p.dt_guard,
            "check_summary": checks,
            "actual_coupling_metadata": metadata,
            "actual_stage_counts": {
                name: metadata.get(name, 0) for name in METADATA_INTEGER_KEYS
            },
            "endpoint_recanonicalization": {
                "independent_fields": change["per_field"],
                "independent_aggregates": change["aggregates"],
                "derived_electron_correction": compatibility_correction[
                    "per_field"
                ],
                "derived_electron_correction_aggregates": (
                    compatibility_correction["aggregates"]
                ),
                "derived_electron_semantics": compatibility_correction[
                    "semantics"
                ],
                "performed_by": "fomhd.massless.canonicalize_state",
                "semantics": (
                    "second canonicalization call through the actual massless "
                    "solver. Independent P/Z are expected to be unchanged; "
                    "the derived electron density and compatibility momentum "
                    "correction is the reconstruction measurement. Existing "
                    "endpoint diagnostics own correctness, not this change "
                    "metric."
                ),
            },
            "resources": _execution_memory(metadata),
            "timing": {
                name: value
                for name, value in execution.items()
                if name.endswith("wall_seconds") or name == "timing_method"
            },
        }
        arrays = {
            **{
                f"before_{name}": value
                for name, value in before_fields.items()
            },
            **{
                f"after_{name}": value
                for name, value in after_fields.items()
            },
            **{
                f"before_{name}": value
                for name, value in before_compatibility.items()
            },
            **{
                f"after_{name}": value
                for name, value in after_compatibility.items()
            },
            "time_history_s": execution["time_history_s"],
            **_endpoint_arrays(canonical, trajectory.context),
        }
        return {
            "summary": summary,
            "arrays": arrays,
            "configuration": {
                "coupling": "predictor",
                "micro_steps": micro_steps,
                "macro_dt_s": macro_dt,
                "macro_steps": macro_count,
                "horizon_s": float(endpoint.time),
            },
        }
    except Exception as error:
        setattr(error, "tool_case_configuration", {
            "case_id": case_id,
            "coupling": "predictor",
            "micro_steps": micro_steps,
            "macro_dt_s": macro_dt,
            "macro_steps": macro_count,
        })
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_git(root: Path) -> dict[str, Any]:
    environment = {
        key: os.environ.get(key, "")
        for key in ("PATH", "HOME", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM")
    }
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout
        return {
            "commit": commit,
            "dirty": bool(status.strip()),
            "dirty_scope": "git status --porcelain; untracked files are dirty",
        }
    except Exception as error:
        return {
            "commit": "0" * 40,
            "dirty": True,
            "reason": f"{type(error).__name__}: {error}",
        }


def _file_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted((root / "src" / "fomhd").glob("*.py")):
        result[str(path.relative_to(root))] = _sha256(path)
    for relative in HARNESS_FILES:
        path = root / relative
        if path.is_file():
            result[relative] = _sha256(path)
    return result


def _library_versions() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    for distribution in (
        "scipy",
        "warp-lang",
        "cupy-cuda13x",
        "pytest",
    ):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "unavailable"
    return result


def build_provenance(
    root: Path,
    *,
    argv: Sequence[str],
    device: str,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return selected non-sensitive execution provenance."""
    environment = os.environ if environment is None else environment
    root = Path(root).resolve()
    selected_environment = {
        key: str(environment[key])
        for key in ENVIRONMENT_KEYS
        if key in environment
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "experimental verification using fixtures; not a production "
            "restart or scientific acceptance record"
        ),
        "argv": list(argv),
        "utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python_executable": sys.executable,
            "python_implementation": platform.python_implementation(),
        },
        "environment": selected_environment,
        "selected_device": device,
        "source_git": _source_git(root),
        "file_hashes": _file_hashes(root),
        "library_versions": _library_versions(),
        "precision": "float64/complex128",
        "fixture_scope": {
            "periodic": PERIODIC_SCOPE,
            "closed": CLOSED_SCOPE,
        },
    }


def run_scan(
    output: str | Path,
    *,
    device: str = "cpu",
    kdi: Sequence[float] = DEFAULT_KDIS,
    points: int = 9,
    relative_amplitude: float = 2.0e-6,
    cycles: float = 1.0,
    micro_steps_per_cycle: int = 64,
    micro_steps: Sequence[int] = DEFAULT_MICRO_STEPS,
    repetitions: int = 1,
    fine_reference: bool = True,
    closed_only: bool = False,
    periodic_only: bool = False,
    closed_micro_steps: int = 4,
    closed_macro_dt: float = 2.0e-10,
    closed_horizon: float = 8.0e-10,
    argv: Sequence[str] | None = None,
    _controls: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create a fresh evidence directory and run the requested bounded set."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite of existing output: {output}")
    controls = (
        normalize_controls(
            device=device,
            kdi=kdi,
            points=points,
            relative_amplitude=relative_amplitude,
            cycles=cycles,
            micro_steps_per_cycle=micro_steps_per_cycle,
            micro_steps=micro_steps,
            repetitions=repetitions,
            fine_reference=fine_reference,
            closed_only=closed_only,
            periodic_only=periodic_only,
            closed_micro_steps=closed_micro_steps,
            closed_macro_dt=closed_macro_dt,
            closed_horizon=closed_horizon,
        )
        if _controls is None
        else dict(_controls)
    )
    _require_device(controls["device"], environment)
    output.mkdir(parents=True, exist_ok=False)
    invocation = list(argv) if argv is not None else [str(Path(__file__))]
    provenance = build_provenance(
        ROOT,
        argv=invocation,
        device=controls["device"],
        environment=environment,
    )
    _write_json(output / "provenance.json", provenance)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "periodic": PERIODIC_SCOPE,
            "closed": CLOSED_SCOPE,
            "comparison_semantics": COMPARISON_SEMANTICS,
            "excluded": [
                "production restart",
                "closed-wall experimental validation",
                "global tokamak or ICE validation",
                "performance or speedup claim",
            ],
        },
        "controls": controls,
        "provenance_file": "provenance.json",
        "cases": [],
        "overall_status": "running",
        "scientific_acceptance": "not_determined_by_parent_owned_thresholds",
    }
    _write_report(output, report)
    try:
        if controls["mode"] == "periodic":
            run_periodic_suite(output, controls, report=report)
        else:
            case_report, _ = _execute_case(
                output,
                "closed_smooth_t0",
                lambda: run_closed_case(
                    device=controls["device"],
                    micro_steps=controls["closed_micro_steps"],
                    macro_dt=controls["closed_macro_dt"],
                    macro_count=controls["closed_macro_count"],
                    repetitions=controls["repetitions"],
                ),
            )
            report["cases"].append(case_report)
    except Exception as error:
        report["cases"].append(
            _write_failure(output, "scan", error)
        )
    measured = all(
        case.get("status") in (
            "measured",
            "measured_with_soft_failures",
        )
        for case in report["cases"]
    )
    report["overall_status"] = (
        "completed_with_scope_limits" if measured else "failed"
    )
    _write_report(output, report)
    with gzip.open(output / "log.jsonl.gz", "xt", encoding="utf-8") as stream:
        for case in report["cases"]:
            stream.write(
                json.dumps(_json_value(case), sort_keys=True) + "\n"
            )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=VALID_DEVICES, default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kdi", type=float, nargs="+", default=list(DEFAULT_KDIS))
    parser.add_argument("--points", type=int, default=9)
    parser.add_argument("--relative-amplitude", type=float, default=2.0e-6)
    parser.add_argument("--cycles", type=float, default=1.0)
    parser.add_argument(
        "--micro-steps-per-cycle",
        type=int,
        choices=VALID_MICRO_STEPS_PER_CYCLE,
        default=64,
    )
    parser.add_argument(
        "--micro-steps",
        type=int,
        nargs="+",
        default=list(DEFAULT_MICRO_STEPS),
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--fine-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the optional synchronous h/2 reference",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--periodic-only", action="store_true")
    modes.add_argument("--closed-only", action="store_true")
    parser.add_argument("--closed-micro-steps", type=int, default=4)
    parser.add_argument("--closed-macro-dt", type=float, default=2.0e-10)
    parser.add_argument("--closed-horizon", type=float, default=8.0e-10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(arguments)
    if args.output.exists():
        print(
            f"refusing overwrite of existing output: {args.output}",
            file=sys.stderr,
        )
        return 2
    try:
        controls = normalize_controls(
            device=args.device,
            kdi=args.kdi,
            points=args.points,
            relative_amplitude=args.relative_amplitude,
            cycles=args.cycles,
            micro_steps_per_cycle=args.micro_steps_per_cycle,
            micro_steps=args.micro_steps,
            repetitions=args.repetitions,
            fine_reference=args.fine_reference,
            closed_only=args.closed_only,
            periodic_only=args.periodic_only,
            closed_micro_steps=args.closed_micro_steps,
            closed_macro_dt=args.closed_macro_dt,
            closed_horizon=args.closed_horizon,
        )
    except ValueError as error:
        parser.error(str(error))
    try:
        _require_device(args.device)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    invocation = [str(Path(__file__).resolve()), *arguments]
    try:
        report = run_scan(
            args.output,
            _controls=controls,
            argv=invocation,
        )
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0 if report["overall_status"] == "completed_with_scope_limits" else 1


if __name__ == "__main__":
    raise SystemExit(main())
