#!/usr/bin/env python3
"""Controlled CPU T0 temporal benchmark for the periodic Hall fixture."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.linalg import expm


def _find_source_root() -> Path:
    override = os.environ.get("FOMHD_SOURCE_ROOT")
    candidates = (
        [Path(override)] if override else []
    ) + [
        Path(__file__).resolve().parents[1],
        Path.cwd(),
    ]
    for candidate in candidates:
        if (
            (candidate / "src" / "fomhd" / "time.py").is_file()
            and (
                candidate
                / "tests"
                / "warp"
                / "massless_verification_fixture.py"
            ).is_file()
        ):
            return candidate.resolve()
    raise RuntimeError(
        "cannot locate FOMHD source root; set FOMHD_SOURCE_ROOT"
    )


ROOT = _find_source_root()
for search_path in (
    ROOT / "src",
    ROOT / "tests" / "warp",
    ROOT,
):
    value = str(search_path)
    if value not in sys.path:
        sys.path.insert(0, value)

from fomhd import core, diagnostics, stability, time_predictor  # noqa: E402
from fomhd.stage import evaluate_stage, initialize  # noqa: E402
from fomhd.time import advance  # noqa: E402
from fomhd.types import Failure, cumulative_record  # noqa: E402
from massless_verification_fixture import (  # noqa: E402
    HallCase,
    build_periodic_context,
    cold_transverse_matrix,
    evaluate,
    uniform_state,
)


SCHEMA_VERSION = 1
EXPECTED_SOURCE_COMMIT = "91602ee825bb14943840449f473184a0edcd6b78"
VALID_DEVICES = ("cpu",)
VALID_MICRO_PER_PERIOD = (2048, 4096)
VALID_COUPLINGS = ("synchronous", "predictor")
VALID_SEEDS = ("target", "mixed")
TARGET_AMPLITUDE = 2.0e-6
HIGH_AMPLITUDE = 2.0e-8
TARGET_HARMONIC = 1
DEFAULT_MAIN_MICRO_STEPS = (4, 8, 16, 32, 64, 128, 256)
EPSILONS = (1.0e-5, 5.0e-6)
TRANSVERSE_INDEXES = (1, 2, 7, 8)
Z_INDEXES = (5, 6, 7, 8)
COORDINATE_NAMES = (
    "n_i",
    "mom_i_x",
    "mom_i_y",
    "mom_i_z",
    "p_i",
    "p_perp_e",
    "p_parallel_e",
    "B_x",
    "B_y",
)
MEMORY_CAP_BYTES = 4 << 30
ACCURACY_CRITERIA = {
    "max_target_modal_vector_error": 0.01,
    "max_target_amplitude_drift": 0.01,
    "max_unwrapped_phase_error_rad": 0.1,
}
GROWTH_FLAG_THRESHOLD = 0.01
SCOPE = (
    "Periodic temporal-isolation fixture using HallCase(kdi=1, points=9), "
    "K=0, Nphi=1 and the real production massless stage/time transaction. "
    "Spatial harmonic m is not toroidal K. This is not closed-wall, "
    "large-device, CAW-CFL, or nonlinear-stability validation."
)
REFERENCE_SEMANTICS = (
    "REF is the independent 4x4 cold transverse Hall continuum branch. "
    "The production 9x9 Jacobian is LIN evidence and retains acoustic/pressure "
    "modes; it is not a nonlinear trajectory or independent physical model."
)
SOURCE_FILES = (
    "pyproject.toml",
    "src/fomhd/time.py",
    "src/fomhd/time_predictor.py",
    "src/fomhd/stage.py",
    "src/fomhd/massless.py",
    "src/fomhd/stability.py",
    "src/fomhd/diagnostics.py",
    "src/fomhd/types.py",
    "tests/warp/massless_verification_fixture.py",
)


def _finite(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_device(device: str) -> None:
    if device not in VALID_DEVICES:
        raise ValueError("this controlled T0 benchmark is CPU only")


def _periodic_case(points: int, *, kdi: float = 1.0) -> HallCase:
    if isinstance(points, bool) or type(points) is not int:
        raise ValueError("points must be an odd integer >= 9")
    if points < 9 or points % 2 == 0:
        raise ValueError("points must be an odd integer >= 9")
    return HallCase(kdi=float(kdi), points=points)


def _normalized_eigenvector(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.complex128).reshape(-1).copy()
    if value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("eigenvector must be finite and nonempty")
    maximum = float(np.max(np.abs(value)))
    if maximum <= 0.0:
        raise ValueError("eigenvector is zero")
    value /= maximum
    pivot = int(np.argmax(np.abs(value)))
    value *= np.exp(-1j * np.angle(value[pivot]))
    return value


def independent_whistler(
    matrix: np.ndarray,
) -> tuple[complex, np.ndarray]:
    values, vectors = np.linalg.eig(np.asarray(matrix, dtype=np.complex128))
    positive = np.flatnonzero(values.imag > 0.0)
    if positive.size == 0:
        raise ValueError("matrix has no positive-frequency branch")
    index = int(positive[np.argmax(values[positive].imag)])
    return complex(values[index]), _normalized_eigenvector(vectors[:, index])


def schedule(
    *,
    coupling: str,
    micro_per_period: int,
    micro_steps: int,
    cycles: int,
    points: int,
) -> dict[str, Any]:
    """Resolve one case's H, h, and exact macro horizon."""
    if coupling not in VALID_COUPLINGS:
        raise ValueError(f"coupling must be one of {VALID_COUPLINGS}")
    if (
        isinstance(micro_per_period, bool)
        or type(micro_per_period) is not int
        or micro_per_period not in VALID_MICRO_PER_PERIOD
    ):
        raise ValueError(
            f"micro_per_period must be one of {VALID_MICRO_PER_PERIOD}"
        )
    micro_steps = _positive_integer("micro_steps", micro_steps)
    cycles = _positive_integer("cycles", cycles)
    case = _periodic_case(points)
    eigenvalue, _ = independent_whistler(cold_transverse_matrix(case))
    omega = float(eigenvalue.imag)
    if not math.isfinite(omega) or omega <= 0.0:
        raise ValueError("target Hall frequency is invalid")
    period = 2.0 * math.pi / omega
    horizon = cycles * period
    micro_dt = period / float(micro_per_period)
    total_micro_steps = cycles * micro_per_period
    if coupling == "synchronous":
        if micro_steps != 1:
            raise ValueError("synchronous coupling requires micro_steps=1")
        macro_dt = micro_dt
    else:
        if total_micro_steps % micro_steps:
            raise ValueError(
                "micro_steps must divide cycles*micro_per_period"
            )
        macro_dt = micro_steps * micro_dt
    macro_steps = int(round(horizon / macro_dt))
    if (
        macro_steps <= 0
        or not math.isclose(
            macro_steps * macro_dt,
            horizon,
            rel_tol=0.0,
            abs_tol=2.0e-15 * max(horizon, 1.0),
        )
    ):
        raise ValueError("macro horizon is not representable")
    return {
        "coupling": coupling,
        "micro_per_period": micro_per_period,
        "micro_steps": micro_steps,
        "cycles": cycles,
        "points": points,
        "omega_target_rad_s": omega,
        "target_period_s": period,
        "horizon_s": horizon,
        "micro_dt_s": micro_dt,
        "macro_dt_s": macro_dt,
        "total_micro_steps": total_micro_steps,
        "macro_steps": macro_steps,
    }


def _mode_angle(points: int, harmonic: int) -> np.ndarray:
    if harmonic < 0 or harmonic > (points - 1) // 2:
        raise ValueError("harmonic is not representable on the periodic grid")
    return (
        2.0
        * math.pi
        * harmonic
        * np.arange(points, dtype=np.float64)
        / float(points)
    )


def project_harmonic(values: Any, harmonic: int, points: int) -> np.ndarray:
    """Project real physical samples to the positive complex harmonic."""
    array = np.asarray(values, dtype=np.complex128)
    if array.size % points:
        raise ValueError("physical sample count is not divisible by points")
    samples = array.reshape(-1, points)
    scale = max(float(np.max(np.abs(samples))), 1.0)
    if float(np.max(np.abs(samples.imag))) > (
        4096.0 * np.finfo(np.float64).eps * scale
    ):
        raise ValueError("modal projection requires real physical samples")
    angle = _mode_angle(points, harmonic)
    factor = 1.0 if harmonic == 0 else 2.0
    result = (
        factor
        * (
            samples.real @ np.cos(angle)
            - 1j * samples.real @ np.sin(angle)
        )
        / float(points)
    )
    return result.reshape(-1)


def harmonic_samples(
    coefficients: Sequence[complex],
    harmonic: int,
    points: int,
) -> np.ndarray:
    """Return rows of real cosine/sine samples from complex coefficients."""
    value = np.asarray(coefficients, dtype=np.complex128).reshape(-1, 1)
    angle = _mode_angle(points, harmonic)
    return (
        value.real * np.cos(angle)[None, :]
        - value.imag * np.sin(angle)[None, :]
    )


def _add_harmonic(
    state: Any,
    coefficients: np.ndarray,
    harmonic: int,
    case: HallCase,
    context: Any,
) -> None:
    coefficient = np.asarray(coefficients, dtype=np.complex128).reshape(9)
    samples = harmonic_samples(coefficient, harmonic, case.points)
    momentum_scale = (
        float(context.p.species[0].mass) * case.n0 * case.v_alfven
    )
    state.fluid[0].n[: case.points] += case.n0 * samples[0]
    for component in range(3):
        start = component * case.points
        stop = (component + 1) * case.points
        state.fluid[0].mom[start:stop] += momentum_scale * samples[1 + component]
    state.fluid[0].perp[: case.points] += case.pressure * samples[4]
    state.fluid[1].perp[: case.points] += case.pressure * samples[5]
    state.fluid[1].parallel[: case.points] += case.pressure * samples[6]
    state.B[: case.points] += case.B0 * samples[7]
    state.B[case.points : 2 * case.points] += case.B0 * samples[8]


def _state_with_modal_coefficients(
    equilibrium: Any,
    coefficients: np.ndarray,
    harmonic: int,
    case: HallCase,
    context: Any,
) -> Any:
    state = copy.deepcopy(equilibrium)
    _add_harmonic(state, coefficients, harmonic, case, context)
    return state


def project_state(
    state: Any,
    context: Any,
    case: HallCase,
    harmonic: int,
) -> np.ndarray:
    """Return the normalized 9-coordinate modal state."""
    points = case.points
    host = context.compute.to_host
    momentum_scale = (
        float(context.p.species[0].mass) * case.n0 * case.v_alfven
    )
    momentum = project_harmonic(
        host(state.fluid[0].mom), harmonic, points
    )
    magnetic = project_harmonic(host(state.B), harmonic, points)[:2]
    return np.asarray(
        [
            project_harmonic(host(state.fluid[0].n), harmonic, points)[0]
            / case.n0,
            momentum[0] / momentum_scale,
            momentum[1] / momentum_scale,
            momentum[2] / momentum_scale,
            project_harmonic(host(state.fluid[0].perp), harmonic, points)[0]
            / case.pressure,
            project_harmonic(host(state.fluid[1].perp), harmonic, points)[0]
            / case.pressure,
            project_harmonic(
                host(state.fluid[1].parallel), harmonic, points
            )[0]
            / case.pressure,
            magnetic[0] / case.B0,
            magnetic[1] / case.B0,
        ],
        dtype=np.complex128,
    )


def _rhs_modal_vector(
    stage: Any,
    context: Any,
    case: HallCase,
    harmonic: int,
) -> np.ndarray:
    host = context.compute.to_host
    points = case.points
    momentum_scale = (
        float(context.p.species[0].mass) * case.n0 * case.v_alfven
    )
    momentum = project_harmonic(
        host(stage.rhs.fluid[0].mom), harmonic, points
    )
    magnetic = project_harmonic(host(stage.rhs.B), harmonic, points)[:2]
    return np.asarray(
        [
            project_harmonic(host(stage.rhs.fluid[0].n), harmonic, points)[0]
            / case.n0,
            momentum[0] / momentum_scale,
            momentum[1] / momentum_scale,
            momentum[2] / momentum_scale,
            project_harmonic(
                host(stage.rhs.fluid[0].perp), harmonic, points
            )[0]
            / case.pressure,
            project_harmonic(
                host(stage.rhs.fluid[1].perp), harmonic, points
            )[0]
            / case.pressure,
            project_harmonic(
                host(stage.rhs.fluid[1].parallel), harmonic, points
            )[0]
            / case.pressure,
            magnetic[0] / case.B0,
            magnetic[1] / case.B0,
        ],
        dtype=np.complex128,
    )


def production_jacobian(
    context: Any,
    case: HallCase,
    equilibrium: Any,
    harmonic: int,
    epsilon: float,
    *,
    columns: Sequence[int] = tuple(range(9)),
    rows: Sequence[int] = tuple(range(9)),
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate the actual production RHS by centered physical modes."""
    epsilon = _finite("epsilon", epsilon, positive=True)
    columns = tuple(int(value) for value in columns)
    rows = tuple(int(value) for value in rows)
    matrix = np.zeros((len(rows), len(columns)), dtype=np.complex128)
    for local_column, column in enumerate(columns):
        direction = np.zeros(9, dtype=np.complex128)
        direction[column] = 1.0
        plus_state = _state_with_modal_coefficients(
            equilibrium, epsilon * direction, harmonic, case, context
        )
        minus_state = _state_with_modal_coefficients(
            equilibrium, -epsilon * direction, harmonic, case, context
        )
        plus = _rhs_modal_vector(
            evaluate(plus_state, context), context, case, harmonic
        )
        minus = _rhs_modal_vector(
            evaluate(minus_state, context), context, case, harmonic
        )
        matrix[:, local_column] = (plus - minus)[list(rows)] / (2.0 * epsilon)
    return matrix, np.asarray(rows, dtype=np.int64)


def _derivative_evidence(
    context: Any,
    case: HallCase,
    harmonic: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    derivative = np.asarray(
        context.compute.to_host(context.op.DZ), dtype=np.complex128
    )
    eigenvalues, eigenvectors = np.linalg.eig(derivative)
    expected = harmonic * case.k
    candidates = np.flatnonzero(eigenvalues.imag > 0.0)
    if candidates.size == 0:
        raise ValueError("periodic derivative has no positive candidate")
    index = int(candidates[np.argmin(np.abs(eigenvalues[candidates].imag - expected))])
    effective = float(eigenvalues[index].imag)
    relative = abs(effective - expected) / expected
    if relative > 2.0e-12:
        raise ValueError(
            f"periodic derivative harmonic {harmonic} mismatch: "
            f"{effective} vs {expected}"
        )
    return (
        {
            "harmonic": harmonic,
            "expected_wavenumber_per_m": expected,
            "effective_wavenumber_per_m": effective,
            "relative_error": relative,
            "eigenvalue": complex(eigenvalues[index]),
            "eigenvector_normalization": "max(abs(vector))=1; pivot phase=0",
            "eigenvector": _normalized_eigenvector(eigenvectors[:, index]),
        },
        {
            "derivative_matrix": derivative,
            "derivative_eigenvalues": eigenvalues,
            "derivative_eigenvectors": eigenvectors,
        },
    )


def _independent_matrix_for_k(
    case: HallCase,
    actual_k: float,
) -> tuple[np.ndarray, HallCase]:
    reference_case = HallCase(
        kdi=actual_k * case.di,
        points=case.points,
        n0=case.n0,
        B0=case.B0,
        pressure=case.pressure,
    )
    return cold_transverse_matrix(reference_case), reference_case


def _nonlinear_residual(
    context: Any,
    case: HallCase,
    equilibrium: Any,
    harmonic: int,
    matrix: np.ndarray,
) -> dict[str, float]:
    coefficient = 3.0e-7 * np.asarray(
        [
            0.3,
            -0.7,
            0.2,
            0.1,
            0.4,
            -0.5,
            0.6,
            0.8,
            -0.9,
        ],
        dtype=np.complex128,
    )
    state = _state_with_modal_coefficients(
        equilibrium, coefficient, harmonic, case, context
    )
    actual = _rhs_modal_vector(
        evaluate(state, context), context, case, harmonic
    )
    linear = matrix @ coefficient
    scale = max(
        float(np.linalg.norm(actual)),
        float(np.linalg.norm(linear)),
        1.0e-300,
    )
    return {
        "coefficient_max": float(np.max(np.abs(coefficient))),
        "absolute_residual_norm": float(np.linalg.norm(actual - linear)),
        "relative_residual": float(np.linalg.norm(actual - linear) / scale),
        "semantics": (
            "production RHS at one deterministic nonlinear modal state minus "
            "the measured local Jacobian action; not a trajectory error"
        ),
    }


def rk4_scalar(argument: complex) -> complex:
    value = complex(argument)
    return (
        1.0
        + value
        + value**2 / 2.0
        + value**3 / 6.0
        + value**4 / 24.0
    )


def rk4_matrix_step(matrix: np.ndarray, dt: float) -> np.ndarray:
    """Return the matrix RK4 stability polynomial for one increment."""
    value = np.asarray(matrix, dtype=np.complex128)
    argument = float(dt) * value
    identity = np.eye(value.shape[0], dtype=np.complex128)
    return (
        identity
        + argument
        + argument @ argument / 2.0
        + argument @ argument @ argument / 6.0
        + argument @ argument @ argument @ argument / 24.0
    )


def critical_imaginary_y() -> float:
    """Measure the outer RK4 imaginary-axis stability boundary."""
    low = 0.0
    high = 4.0
    if abs(rk4_scalar(1j * high)) <= 1.0:
        raise RuntimeError("RK4 imaginary stability bracket failed")
    for _ in range(80):
        middle = 0.5 * (low + high)
        if abs(rk4_scalar(1j * middle)) <= 1.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _spectrum_stability(
    label: str,
    eigenvalues: Sequence[np.ndarray],
    *,
    micro_dt: float,
    period: float,
    total_micro_steps: int,
    critical_y: float,
) -> dict[str, Any]:
    all_values = np.concatenate(
        [np.asarray(value, dtype=np.complex128).reshape(-1) for value in eigenvalues]
    )
    maximum = float(np.max(np.abs(all_values.imag)))
    critical_dt = critical_y / max(maximum, 1.0e-300)
    mode_rows = []
    for value in all_values:
        y = micro_dt * float(value.imag)
        method = rk4_scalar(1j * y)
        accumulated = method**total_micro_steps
        mode_rows.append(
            {
                "eigenvalue_rad_s": value,
                "micro_omega_dt": y,
                "micro_absolute_amplification": abs(method),
                "horizon_amplification": abs(accumulated),
                "horizon_phase_error_rad": (
                    total_micro_steps
                    * abs(math.atan2(method.imag, method.real) - y)
                ),
            }
        )
    old = {}
    for divisor in (64, 128):
        dt = period / float(divisor)
        old[f"T/{divisor}"] = {
            "micro_dt_s": dt,
            "micro_omega_dt": dt * maximum,
            "safety_fraction": dt / critical_dt,
            "inside_outer_imaginary_interval": dt * maximum <= critical_y,
        }
    return {
        "label": label,
        "maximum_abs_imaginary_eigenvalue_rad_s": maximum,
        "critical_micro_dt_s": critical_dt,
        "chosen_micro_dt_s": micro_dt,
        "chosen_safety_fraction": micro_dt / critical_dt,
        "chosen_position_y": micro_dt * maximum,
        "old_positions": old,
        "modes": mode_rows,
        "semantics": (
            "RK4 polynomial facts for measured LIN spectra; not a nonlinear "
            "or whole-T0-map stability certificate"
        ),
    }


def _generic_stage_stability(
    base_stage: Any,
    context: Any,
    *,
    micro_dt: float,
    total_micro_steps: int,
) -> dict[str, Any]:
    estimate = stability.stage_frequency_estimate(base_stage, context)
    report = stability.assess_dt(
        "rk4",
        estimate["omega_estimate"],
        micro_dt,
        total_micro_steps,
        minimum_samples=context.p.minimum_samples_per_cycle,
        amplitude_budget=context.p.max_amplitude_error,
        phase_budget=context.p.max_phase_error_rad,
    )
    return {
        "omega_estimate_rad_s": estimate["omega_estimate"],
        "estimate": estimate,
        "assessment_at_chosen_micro_dt": report,
        "scope": (
            "existing production operator-resolution estimate; separate from "
            "the resolved full and frozen-ion spectra"
        ),
    }


def _split_blocks(matrix: np.ndarray) -> tuple[np.ndarray, ...]:
    value = np.asarray(matrix, dtype=np.complex128)
    p_indexes = list(range(5))
    z_indexes = list(range(5, 9))
    return (
        value[np.ix_(p_indexes, p_indexes)],
        value[np.ix_(p_indexes, z_indexes)],
        value[np.ix_(z_indexes, p_indexes)],
        value[np.ix_(z_indexes, z_indexes)],
    )


def _linear_t0_trajectory(
    p0: np.ndarray,
    z0: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    dt: float,
    micro_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    p_value = np.asarray(p0, dtype=np.complex128).copy()
    z_value = np.asarray(z0, dtype=np.complex128).copy()
    f0 = a @ p_value + b @ z_value
    accepted = p_value.copy()
    h = dt / float(micro_steps)
    clock = SimpleNamespace(dt=dt, micro_steps=micro_steps)
    for micro in range(micro_steps):
        derivatives = []
        for stage_index in range(4):
            offset = core.microstage_offset(clock, micro, stage_index)
            tangent = p_value + offset * f0
            if stage_index == 0:
                z_stage = z_value
            elif stage_index == 1:
                z_stage = z_value + (h / 2.0) * derivatives[0]
            elif stage_index == 2:
                z_stage = z_value + (h / 2.0) * derivatives[1]
            else:
                z_stage = z_value + h * derivatives[2]
            derivatives.append(c @ tangent + d @ z_stage)
            accepted = accepted + (h / 6.0) * (
                a @ tangent + b @ z_stage
            ) * (1.0 if stage_index in (0, 3) else 2.0)
        z_value = z_value + (h / 6.0) * (
            derivatives[0]
            + 2.0 * derivatives[1]
            + 2.0 * derivatives[2]
            + derivatives[3]
        )
    return accepted, z_value


def linear_t0_macro_map(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    dt: float,
    *,
    micro_steps: int,
) -> np.ndarray:
    """Return the exact fixed-tangent T0 linear map for P/Z blocks."""
    dt = _finite("dt", dt, positive=True)
    micro_steps = _positive_integer("micro_steps", micro_steps)
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    c = np.asarray(c, dtype=np.complex128)
    d = np.asarray(d, dtype=np.complex128)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("A must be square")
    p_size = a.shape[0]
    if b.ndim != 2:
        raise ValueError("B must be a matrix")
    z_size = b.shape[1]
    if (
        z_size <= 0
        or b.shape[0] != p_size
        or c.shape != (z_size, p_size)
        or d.shape != (z_size, z_size)
    ):
        raise ValueError("linear T0 block shapes are inconsistent")
    total = p_size + z_size
    result = np.zeros((total, total), dtype=np.complex128)
    for column in range(total):
        initial = np.zeros(total, dtype=np.complex128)
        initial[column] = 1.0
        p_end, z_end = _linear_t0_trajectory(
            initial[:p_size],
            initial[p_size:],
            a,
            b,
            c,
            d,
            dt,
            micro_steps,
        )
        result[:p_size, column] = p_end
        result[p_size:, column] = z_end
    return result


def build_audit(points: int = 9) -> dict[str, Any]:
    """Build actual-derivative, production-Jacobian, and stability evidence."""
    case = _periodic_case(points)
    context = build_periodic_context(case, device="cpu")
    context.p.memory_limit = MEMORY_CAP_BYTES
    context.p.device_memory_limit = MEMORY_CAP_BYTES
    equilibrium = uniform_state(context, case)
    baseline = evaluate(equilibrium, context)
    target_value, _ = independent_whistler(cold_transverse_matrix(case))
    period = 2.0 * math.pi / float(target_value.imag)
    micro_dt = period / 2048.0
    harmonic_max = (points - 1) // 2
    derivative_wavenumbers: dict[int, dict[str, Any]] = {}
    full_jacobians: dict[int, np.ndarray] = {}
    full_jacobians_second: dict[int, np.ndarray] = {}
    micro_jacobians: dict[int, np.ndarray] = {}
    micro_jacobians_second: dict[int, np.ndarray] = {}
    independent_matrices: dict[int, np.ndarray] = {}
    independent_values: dict[int, np.ndarray] = {}
    independent_vectors: dict[int, np.ndarray] = {}
    full_values: dict[int, np.ndarray] = {}
    full_vectors: dict[int, np.ndarray] = {}
    epsilon_difference: dict[int, float] = {}
    transverse_relative_error: dict[int, float] = {}
    nonlinear_residual: dict[int, dict[str, float]] = {}
    arrays: dict[str, np.ndarray] = {}

    host = context.compute.to_host
    baseline_rhs = _rhs_modal_vector(baseline, context, case, 1)
    baseline_arrays = {
        "rhs_ion_n": host(baseline.rhs.fluid[0].n),
        "rhs_ion_momentum": host(baseline.rhs.fluid[0].mom),
        "rhs_ion_perp": host(baseline.rhs.fluid[0].perp),
        "rhs_electron_perp": host(baseline.rhs.fluid[1].perp),
        "rhs_electron_parallel": host(baseline.rhs.fluid[1].parallel),
        "rhs_B": host(baseline.rhs.B),
    }
    arrays.update({f"baseline_{name}": value for name, value in baseline_arrays.items()})

    for harmonic in range(1, harmonic_max + 1):
        derivative, derivative_arrays = _derivative_evidence(
            context, case, harmonic
        )
        derivative_wavenumbers[harmonic] = derivative
        for name, value in derivative_arrays.items():
            arrays[f"{name}_m{harmonic}"] = value
        independent, _ = _independent_matrix_for_k(
            case, derivative["effective_wavenumber_per_m"]
        )
        independent_matrices[harmonic] = independent
        independent_values[harmonic], independent_vectors[harmonic] = np.linalg.eig(
            independent
        )
        first, _ = production_jacobian(
            context, case, equilibrium, harmonic, EPSILONS[0]
        )
        second, _ = production_jacobian(
            context, case, equilibrium, harmonic, EPSILONS[1]
        )
        micro, _ = production_jacobian(
            context,
            case,
            equilibrium,
            harmonic,
            EPSILONS[0],
            columns=Z_INDEXES,
            rows=Z_INDEXES,
        )
        micro_second, _ = production_jacobian(
            context,
            case,
            equilibrium,
            harmonic,
            EPSILONS[1],
            columns=Z_INDEXES,
            rows=Z_INDEXES,
        )
        full_jacobians[harmonic] = first
        full_jacobians_second[harmonic] = second
        micro_jacobians[harmonic] = micro
        micro_jacobians_second[harmonic] = micro_second
        full_values[harmonic], full_vectors[harmonic] = np.linalg.eig(first)
        epsilon_difference[harmonic] = float(np.max(np.abs(first - second)))
        transverse = first[np.ix_(TRANSVERSE_INDEXES, TRANSVERSE_INDEXES)]
        transverse_relative_error[harmonic] = float(
            np.linalg.norm(transverse - independent)
            / max(np.linalg.norm(independent), 1.0e-300)
        )
        nonlinear_residual[harmonic] = _nonlinear_residual(
            context, case, equilibrium, harmonic, first
        )
        arrays.update(
            {
                f"full_jacobian_eps1e5_m{harmonic}": first,
                f"full_jacobian_eps5e6_m{harmonic}": second,
                f"micro_jacobian_eps1e5_m{harmonic}": micro,
                f"micro_jacobian_eps5e6_m{harmonic}": micro_second,
                f"independent_matrix_m{harmonic}": independent,
                f"independent_eigenvalues_m{harmonic}": independent_values[harmonic],
                f"independent_eigenvectors_m{harmonic}": independent_vectors[harmonic],
                f"full_eigenvalues_m{harmonic}": full_values[harmonic],
                f"full_eigenvectors_m{harmonic}": full_vectors[harmonic],
            }
        )

    whole_map = {}
    for micro_steps in DEFAULT_MAIN_MICRO_STEPS:
        macro_dt = micro_steps * micro_dt
        macro_steps = 4096 // micro_steps
        harmonic_records = {}
        for harmonic in range(1, harmonic_max + 1):
            a, b, c, d = _split_blocks(full_jacobians[harmonic])
            macro_map = linear_t0_macro_map(
                a,
                b,
                c,
                d,
                macro_dt,
                micro_steps=micro_steps,
            )
            horizon_map = np.linalg.matrix_power(macro_map, macro_steps)
            cold_value, cold_vector = independent_whistler(
                independent_matrices[harmonic]
            )
            seed = np.zeros(9, dtype=np.complex128)
            seed[list(TRANSVERSE_INDEXES)] = cold_vector
            seed_norm = float(np.linalg.norm(seed))
            history = np.empty(macro_steps + 1, dtype=np.float64)
            value = seed.copy()
            for index in range(macro_steps + 1):
                history[index] = np.linalg.norm(value)
                value = macro_map @ value
            seed_gain_history = history / max(seed_norm, 1.0e-300)
            positive = int(
                np.count_nonzero(np.diff(seed_gain_history) > 1.0e-12)
            )
            per_macro_radius = float(
                np.max(np.abs(np.linalg.eigvals(macro_map)))
            )
            horizon_radius = float(
                np.max(np.abs(np.linalg.eigvals(horizon_map)))
            )
            alias = macro_alias_metadata(
                {harmonic: {"eigenvalue": cold_value}}, macro_dt
            )[str(harmonic)]
            harmonic_records[str(harmonic)] = {
                "cold_eigenvalue": cold_value,
                "seed_vector": seed,
                "seed_norm": seed_norm,
                "map": macro_map,
                "horizon_map": horizon_map,
                "per_macro_spectral_radius": per_macro_radius,
                "spectral_radius_power_over_horizon": (
                    per_macro_radius**macro_steps
                ),
                "horizon_spectral_radius": horizon_radius,
                "seed_gain": float(seed_gain_history[-1]),
                "maximum_seed_gain": float(np.max(seed_gain_history)),
                "seed_gain_history": seed_gain_history,
                "amplification_over_1_percent": bool(
                    seed_gain_history[-1] > 1.0 + GROWTH_FLAG_THRESHOLD
                ),
                "positive_growth_intervals": positive,
                "repeated_cumulative_growth": bool(
                    seed_gain_history[-1] > 1.0 + GROWTH_FLAG_THRESHOLD
                    and positive >= 2
                ),
                "transient_peak_only": bool(
                    np.max(seed_gain_history)
                    > 1.0 + GROWTH_FLAG_THRESHOLD
                    and seed_gain_history[-1]
                    <= 1.0 + GROWTH_FLAG_THRESHOLD
                ),
                "macro_alias": alias,
                "seed_norm_convention": "Euclidean norm of maxabs-normalized cold vector",
            }
            arrays[f"t0_macro_map_m{harmonic}_M{micro_steps}"] = macro_map
            arrays[
                f"t0_macro_map_power_m{harmonic}_M{micro_steps}"
            ] = horizon_map
            arrays[
                f"t0_seed_vector_m{harmonic}_M{micro_steps}"
            ] = seed
            arrays[
                f"t0_seed_gain_history_m{harmonic}_M{micro_steps}"
            ] = seed_gain_history
        whole_map[str(micro_steps)] = {
            "micro_steps": micro_steps,
            "micro_dt_s": micro_dt,
            "macro_dt_s": macro_dt,
            "macro_steps": macro_steps,
            "horizon_s": 2.0 * period,
            "harmonics": harmonic_records,
            "semantics": (
                "production 9x9 linear T0 map powers and actual independent "
                "cold-seed gains; nonnormal transients are retained"
            ),
        }

    zero_matrix = np.full((9, 9), np.nan + 0j, dtype=np.complex128)
    zero_valid = np.ones(9, dtype=bool)
    zero_failures = {}
    for column in range(9):
        try:
            partial_micro, _ = production_jacobian(
                context,
                case,
                equilibrium,
                0,
                EPSILONS[0],
                columns=(column,),
            )
        except Failure as failure:
            zero_valid[column] = False
            zero_failures[str(column)] = {
                "column": COORDINATE_NAMES[column],
                "failure_code": failure.code,
                "message": str(failure),
            }
            continue
        zero_matrix[:, column] = partial_micro[:, 0]
    zero_mode = {
        "eigenvalues_available": False,
        "reason": (
            "a complete numerical m=0 Jacobian was not measured; unavailable "
            "columns are masked rather than replaced by a fabricated zero"
        ),
        "failures": zero_failures,
        "analytic_stationarity": (
            "constant uniform perturbations are analytically stationary for "
            "the cold transverse Hall block; this is separate from the failed "
            "longitudinal-momentum finite-difference probe"
        ),
        "resolved_frequency_scope": (
            "m=1..mmax determine the resolved fastest-frequency audit"
        ),
    }
    critical_y = critical_imaginary_y()
    stability_report = {
        "critical_imaginary_y": critical_y,
        "critical_imaginary_prediction": 2.0 * math.sqrt(2.0),
        "full_spectrum": _spectrum_stability(
            "full production 9x9 spectra over m=1..mmax",
            [full_values[index] for index in range(1, harmonic_max + 1)],
            micro_dt=micro_dt,
            period=period,
            total_micro_steps=4096,
            critical_y=critical_y,
        ),
        "micro_spectrum": _spectrum_stability(
            "frozen-ion electron/field 4x4 spectra over m=1..mmax",
            [np.linalg.eigvals(micro_jacobians[index]) for index in range(1, harmonic_max + 1)],
            micro_dt=micro_dt,
            period=period,
            total_micro_steps=4096,
            critical_y=critical_y,
        ),
        "generic_stage_estimate": _generic_stage_stability(
            baseline,
            context,
            micro_dt=micro_dt,
            total_micro_steps=4096,
        ),
    }
    arrays["zero_mode_jacobian"] = zero_matrix
    arrays["zero_mode_valid_column_mask"] = zero_valid
    return {
        "schema_version": SCHEMA_VERSION,
        "case": case,
        "context": context,
        "equilibrium": equilibrium,
        "baseline": baseline,
        "baseline_rhs": baseline_rhs,
        "harmonic_max": harmonic_max,
        "derivative_wavenumbers": derivative_wavenumbers,
        "full_jacobians": full_jacobians,
        "full_jacobians_second": full_jacobians_second,
        "micro_jacobians": micro_jacobians,
        "micro_jacobians_second": micro_jacobians_second,
        "independent_matrices": independent_matrices,
        "independent_eigenvalues": independent_values,
        "independent_eigenvectors": independent_vectors,
        "full_eigenvalues": full_values,
        "full_eigenvectors": full_vectors,
        "jacobian_epsilon_difference": epsilon_difference,
        "transverse_relative_error": transverse_relative_error,
        "nonlinear_residual": nonlinear_residual,
        "whole_map": whole_map,
        "zero_mode_jacobian": zero_matrix,
        "zero_mode_valid_columns": zero_valid,
        "zero_mode": zero_mode,
        "stability": stability_report,
        "arrays": arrays,
        "reference_semantics": REFERENCE_SEMANTICS,
    }


def build_seed(
    *,
    points: int,
    seed: str,
    context: Any | None = None,
) -> dict[str, Any]:
    """Build target-only or two-harmonic real physical seed fields."""
    if seed not in VALID_SEEDS:
        raise ValueError(f"seed must be one of {VALID_SEEDS}")
    case = _periodic_case(points)
    if context is None:
        context = build_periodic_context(case, device="cpu")
        context.p.memory_limit = MEMORY_CAP_BYTES
        context.p.device_memory_limit = MEMORY_CAP_BYTES
    equilibrium = uniform_state(context, case)
    harmonics = (TARGET_HARMONIC,)
    if seed == "mixed":
        harmonics += ((points - 1) // 2,)
    state = copy.deepcopy(equilibrium)
    vectors: dict[int, np.ndarray] = {}
    transverse_vectors: dict[int, np.ndarray] = {}
    eigenvalues: dict[int, complex] = {}
    coefficients: dict[int, np.ndarray] = {}
    amplitudes: dict[int, float] = {}
    for harmonic in harmonics:
        independent, _ = _independent_matrix_for_k(case, harmonic * case.k)
        cold_value, cold_vector = independent_whistler(independent)
        embedded = np.zeros(9, dtype=np.complex128)
        embedded[list(TRANSVERSE_INDEXES)] = cold_vector
        amplitude = TARGET_AMPLITUDE if harmonic == 1 else HIGH_AMPLITUDE
        coefficient = amplitude * embedded
        _add_harmonic(state, coefficient, harmonic, case, context)
        vectors[harmonic] = embedded
        transverse_vectors[harmonic] = cold_vector
        eigenvalues[harmonic] = cold_value
        coefficients[harmonic] = coefficient
        amplitudes[harmonic] = amplitude
    projected = {
        harmonic: project_state(state, context, case, harmonic)
        for harmonic in harmonics
    }
    return {
        "seed": seed,
        "state": state,
        "context": context,
        "case": case,
        "harmonics": harmonics,
        "vectors": vectors,
        "transverse_vectors": transverse_vectors,
        "eigenvalues": eigenvalues,
        "coefficients": coefficients,
        "amplitudes": amplitudes,
        "projected_coefficients": projected,
        "normalization": (
            "independent cold4x4 whistler eigenvector max(abs(component))=1 "
            "with pivot phase zero, embedded in transverse9 coordinates; "
            "amplitude is the maximum normalized transverse component"
        ),
    }


def production_macro_map_comparison(
    *,
    points: int,
    harmonic: int,
    micro_steps: int,
) -> dict[str, Any]:
    """Compare linear T0 algebra with a real one-macro finite difference."""
    case = _periodic_case(points)
    context = build_periodic_context(case, device="cpu")
    context.p.memory_limit = MEMORY_CAP_BYTES
    context.p.device_memory_limit = MEMORY_CAP_BYTES
    equilibrium = uniform_state(context, case)
    matrix, _ = production_jacobian(
        context, case, equilibrium, harmonic, EPSILONS[0]
    )
    target_value, _ = independent_whistler(cold_transverse_matrix(case))
    period = 2.0 * math.pi / float(target_value.imag)
    micro_dt = period / 2048.0
    macro_dt = micro_steps * micro_dt
    a, b, c, d = _split_blocks(matrix)
    linear = linear_t0_macro_map(
        a, b, c, d, macro_dt, micro_steps=micro_steps
    )
    production = np.zeros((9, 9), dtype=np.complex128)
    delta = 1.0e-7
    context.p.coupling = "predictor"
    context.p.micro_steps = micro_steps
    context.p.integrator = "rk4"
    context.p.dt = macro_dt
    context.p.nsteps = 1
    context.p.dt_guard = "error"
    for column in range(9):
        endpoints = []
        for sign in (1.0, -1.0):
            coefficient = np.zeros(9, dtype=np.complex128)
            coefficient[column] = sign * delta
            raw = _state_with_modal_coefficients(
                equilibrium, coefficient, harmonic, case, context
            )
            committed = initialize(
                raw,
                context,
                cumulative_record(),
                equilibrium=False,
            )
            advance(committed, cumulative_record(), context, None)
            endpoints.append(
                project_state(committed.state, context, case, harmonic)
            )
        production[:, column] = (endpoints[0] - endpoints[1]) / (2.0 * delta)
    difference = production - linear
    return {
        "linear_map": linear,
        "production_map": production,
        "difference": difference,
        "relative_error": float(
            np.linalg.norm(difference)
            / max(np.linalg.norm(linear), 1.0e-300)
        ),
        "delta": delta,
        "macro_dt_s": macro_dt,
        "micro_steps": micro_steps,
        "semantics": (
            "central production finite difference of one actual T0 macro "
            "versus the exact fixed-tangent linear algebra"
        ),
    }


def unwrapped_phase(
    observed: Sequence[np.ndarray],
    reference: Sequence[np.ndarray],
) -> np.ndarray:
    """Return cumulative phase of observed/reference modal projections."""
    if len(observed) != len(reference):
        raise ValueError("observed and reference histories differ in length")
    ratios = []
    for actual, expected in zip(observed, reference):
        actual_value = np.asarray(actual, dtype=np.complex128)
        expected_value = np.asarray(expected, dtype=np.complex128)
        denominator = np.vdot(expected_value, expected_value)
        if abs(denominator) <= 0.0:
            raise ValueError("reference modal vector is zero")
        ratios.append(np.vdot(expected_value, actual_value) / denominator)
    return np.unwrap(np.angle(np.asarray(ratios, dtype=np.complex128)))


def qualify_accuracy(
    *,
    vector_error: float,
    amplitude_drift: float,
    phase_error_rad: float,
) -> dict[str, Any]:
    values = {
        "max_target_modal_vector_error": abs(float(vector_error)),
        "max_target_amplitude_drift": abs(float(amplitude_drift)),
        "max_unwrapped_phase_error_rad": abs(float(phase_error_rad)),
    }
    checks = {
        name: bool(math.isfinite(value) and value <= ACCURACY_CRITERIA[name])
        for name, value in values.items()
    }
    return {
        "measured": values,
        "criteria": dict(ACCURACY_CRITERIA),
        "checks": checks,
        "joint_qualified": all(checks.values()),
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return "NaN" if math.isnan(value) else (
            "Infinity" if value > 0.0 else "-Infinity"
        )
    if isinstance(value, complex):
        return {"real": _jsonable(value.real), "imag": _jsonable(value.imag)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value") and hasattr(value, "name"):
        return _jsonable(value.value)
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_jsonable(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, np.generic) or np.isscalar(value):
        array = np.asarray(value)
    elif hasattr(value, "get") and hasattr(value, "shape"):
        array = np.asarray(value.get())
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value)
    else:
        return None
    return None if array.dtype.hasobject else array


def _save_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    cleaned = {}
    for name, value in arrays.items():
        array = _array(value)
        if array is not None:
            cleaned[str(name)] = array
    if not cleaned:
        cleaned["empty"] = np.empty(0, dtype=np.float64)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **cleaned)


class _CheckAccumulator:
    def __init__(self) -> None:
        self.summary = {
            "total": 0,
            "hard_total": 0,
            "hard_failed": 0,
            "soft_failed": 0,
            "unavailable": 0,
            "failed_records": [],
        }
        self.grouped: dict[tuple[Any, ...], dict[str, Any]] = {}

    def update(self, record: Any) -> None:
        summary = self.summary
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
            key = (
                getattr(record, "name", ""),
                getattr(record, "population", ""),
                getattr(record, "component", ""),
                getattr(record, "region", ""),
                getattr(record, "unit", ""),
                hard,
                available,
            )
            if key not in self.grouped:
                self.grouped[key] = {"record": _jsonable(record), "count": 0}
            item = self.grouped[key]
            item["count"] += 1
            previous = item["record"]
            previous_raw = abs(float(previous.get("raw", 0.0)))
            current_raw = abs(float(getattr(record, "raw", 0.0)))
            if current_raw > previous_raw:
                item["record"] = _jsonable(record)

    def finish(self) -> dict[str, Any]:
        self.summary["failed_records"] = list(self.grouped.values())
        return self.summary


def _check_summary(records: Iterable[Any]) -> dict[str, Any]:
    accumulator = _CheckAccumulator()
    for record in records:
        accumulator.update(record)
    return accumulator.finish()


def _stream_checks(
    stream: Any,
    context_label: str,
    records: Iterable[Any],
    accumulator: _CheckAccumulator,
) -> int:
    count = 0
    for record in records:
        stream.write(
            json.dumps(
                {"context": context_label, "record": _jsonable(record)},
                sort_keys=True,
            )
            + "\n"
        )
        accumulator.update(record)
        count += 1
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_git() -> dict[str, Any]:
    environment = {
        key: os.environ.get(key, "")
        for key in ("PATH", "HOME", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM")
    }
    try:
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", str(ROOT), "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(ROOT), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            ).stdout.strip()
        )
    except Exception as error:
        return {
            "expected_commit": EXPECTED_SOURCE_COMMIT,
            "commit": "0" * 40,
            "branch": "",
            "dirty": True,
            "reason": f"{type(error).__name__}: {error}",
        }
    return {
        "expected_commit": EXPECTED_SOURCE_COMMIT,
        "commit": commit,
        "commit_matches_expected": commit == EXPECTED_SOURCE_COMMIT,
        "branch": branch,
        "dirty": dirty,
        "dirty_files_detected_by_hashes": True,
    }


def _provenance(argv: Sequence[str], device: str) -> dict[str, Any]:
    hashes = _runtime_source_hashes()
    tool_path = Path(__file__).resolve()
    proposed_root = tool_path.parents[1]
    proposed_additions = {}
    for relative in (
        "tools/warp_t0_controlled.py",
        "tests/warp/test_warp_t0_controlled.py",
        "docs/pseudocode/tools/warp_t0_controlled.py.md",
        "docs/pseudocode/tests/warp/test_warp_t0_controlled.py.md",
    ):
        path = proposed_root / relative
        if path.is_file():
            proposed_additions[relative] = _sha256(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "controlled T0 implementation audit and temporal cases",
        "argv": list(argv),
        "utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
        },
        "selected_device": device,
        "source_git": _source_git(),
        "source_hashes": hashes,
        "proposed_tool": {"path": str(tool_path), "sha256": _sha256(tool_path)},
        "proposed_addition_hashes": proposed_additions,
        "precision": "float64/complex128",
        "scope": SCOPE,
    }


def _require_fresh(output: Path) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite of existing output: {output}")
    return output


def _budget_snapshot(parameters: Any) -> dict[str, Any]:
    return {
        name: {
            "atol": budget.atol,
            "rtol": budget.rtol,
            "reference": budget.reference,
            "unit": budget.unit,
            "norm": budget.norm,
            "hard": budget.hard,
        }
        for name, budget in parameters.budgets.items()
    }


def _case_progress(
    accepted: int,
    expected: int,
    progress_every: int,
) -> None:
    if progress_every <= 0:
        return
    if (
        accepted == 1
        or accepted == expected
        or accepted % progress_every == 0
    ):
        print(
            f"accepted {accepted}/{expected} macro steps",
            file=sys.stderr,
        )


def _validate_predictor_metadata(record: Any, micro_steps: int) -> None:
    metadata = dict(record) if isinstance(record, Mapping) else vars(record)
    expected = {
        "complete_stage_evaluations": 4 * micro_steps + 1,
        "full_rhs_stage_evaluations": 4 * micro_steps,
        "endpoint_full_evaluations": 1,
        "alias_evaluations": 2 * micro_steps,
        "total_model_evaluations": 6 * micro_steps + 1,
        "retained_quadrature_entries": 4 * micro_steps,
        "assemble_canonicalization_calls": 4 * micro_steps,
    }
    if metadata.get("coupling") != "predictor":
        raise Failure("Metadata", "coupling metadata is not predictor")
    if metadata.get("micro_steps") != micro_steps:
        raise Failure("Metadata", "micro_steps metadata mismatch")
    for name, expected_value in expected.items():
        if metadata.get(name) != expected_value:
            raise Failure(
                "Metadata",
                f"{name} mismatch: expected {expected_value}, "
                f"got {metadata.get(name)}",
            )


def _mode_specs(seed: Mapping[str, Any], case: HallCase) -> dict[int, dict[str, Any]]:
    specs = {}
    for harmonic in seed["harmonics"]:
        transverse = seed["transverse_vectors"][harmonic]
        vector = seed["vectors"][harmonic]
        specs[harmonic] = {
            "amplitude": seed["amplitudes"][harmonic],
            "vector": vector,
            "transverse_vector": transverse,
            "eigenvalue": seed["eigenvalues"][harmonic],
        }
    return specs


def _prepare_case(
    *,
    coupling: str,
    micro_per_period: int,
    micro_steps: int,
    cycles: int,
    seed: str,
    points: int,
) -> SimpleNamespace:
    case_schedule = schedule(
        coupling=coupling,
        micro_per_period=micro_per_period,
        micro_steps=micro_steps,
        cycles=cycles,
        points=points,
    )
    case = _periodic_case(points)
    context = build_periodic_context(case, device="cpu")
    budgets_before = _budget_snapshot(context.p)
    context.p.memory_limit = MEMORY_CAP_BYTES
    context.p.device_memory_limit = MEMORY_CAP_BYTES
    context.p.coupling = coupling
    context.p.micro_steps = micro_steps
    context.p.integrator = "rk4"
    context.p.dt = case_schedule["macro_dt_s"]
    context.p.nsteps = case_schedule["macro_steps"]
    context.p.dt_guard = "error"
    built = build_seed(points=points, seed=seed, context=context)
    raw = built["state"]
    captured: dict[str, Any] = {}
    cumulative = cumulative_record()
    committed = initialize(
        raw,
        context,
        cumulative,
        equilibrium=False,
        captured=captured,
    )
    initial_checks = list(captured.get("checks", ()))
    budgets_after = _budget_snapshot(context.p)
    if budgets_after != budgets_before:
        raise RuntimeError("original solver budgets or hard flags changed")
    return SimpleNamespace(
        schedule=case_schedule,
        case=case,
        context=context,
        seed=built,
        raw=raw,
        committed=committed,
        cumulative=cumulative,
        initial_checks=initial_checks,
        initial_totals=captured.get("totals"),
        budgets=budgets_before,
        mode_specs=_mode_specs(built, case),
    )


def _state_arrays(state: Any, context: Any, prefix: str) -> dict[str, np.ndarray]:
    host = context.compute.to_host
    return {
        f"{prefix}_ion_n": np.asarray(host(state.fluid[0].n)).copy(),
        f"{prefix}_ion_momentum": np.asarray(host(state.fluid[0].mom)).copy(),
        f"{prefix}_ion_perp": np.asarray(host(state.fluid[0].perp)).copy(),
        f"{prefix}_electron_perp": np.asarray(host(state.fluid[1].perp)).copy(),
        f"{prefix}_electron_parallel": np.asarray(host(state.fluid[1].parallel)).copy(),
        f"{prefix}_B": np.asarray(host(state.B)).copy(),
    }


def _cumulative_arrays(
    cumulative: Any,
    committed: Any,
) -> dict[str, Any]:
    """Serialize the existing cumulative ledger without creating another."""
    arrays: dict[str, Any] = {}
    baseline = cumulative.baseline
    for name, value in vars(baseline).items():
        if name in ("totals", "momentum"):
            for child, child_value in vars(value).items():
                arrays[f"cumulative_baseline_{name}_{child}"] = child_value
        else:
            arrays[f"cumulative_baseline_{name}"] = value
    for species, integral in enumerate(cumulative.integrals):
        for name, value in vars(integral).items():
            arrays[f"cumulative_integral_{species}_{name}"] = value
    boundary = cumulative.boundary_integrals
    for name, value in vars(boundary).items():
        if name != "traces":
            arrays[f"cumulative_boundary_{name}"] = value
    arrays["cumulative_boundary_traces_json"] = np.asarray(
        json.dumps(_jsonable(boundary.traces), sort_keys=True)
    )
    arrays["cumulative_boundary_trace_count"] = np.asarray(
        len(boundary.traces), dtype=np.int64
    )
    for name in (
        "accepted_intervals",
        "last_trial",
        "last_step",
        "last_time",
    ):
        arrays[f"cumulative_{name}"] = getattr(cumulative, name)
    arrays["cumulative_transaction_accepted"] = np.asarray(
        bool(committed.accepted), dtype=np.bool_
    )
    arrays["cumulative_state_step"] = np.asarray(
        committed.state.step, dtype=np.int64
    )
    arrays["cumulative_state_generation"] = np.asarray(
        committed.state.generation, dtype=np.uint64
    )
    arrays["cumulative_state_time"] = np.asarray(
        committed.state.time, dtype=np.float64
    )
    return arrays


def _cumulative_metadata(cumulative: Any, committed: Any) -> dict[str, Any]:
    return {
        "available": True,
        "identity": {
            "accepted_intervals": int(cumulative.accepted_intervals),
            "last_trial": int(cumulative.last_trial),
            "last_step": int(cumulative.last_step),
            "last_time": float(cumulative.last_time),
            "state_step": int(committed.state.step),
            "state_generation": int(committed.state.generation),
            "state_time": float(committed.state.time),
        },
        "transaction_status": {
            "accepted": bool(committed.accepted),
            "accepted_intervals": int(cumulative.accepted_intervals),
            "last_step": int(cumulative.last_step),
        },
        "baseline_fields": sorted(vars(cumulative.baseline)),
        "integral_fields": {
            str(species): sorted(vars(integral))
            for species, integral in enumerate(cumulative.integrals)
        },
        "boundary_fields": sorted(vars(cumulative.boundary_integrals)),
        "semantics": (
            "the existing fomhd.types.cumulative_record serialized by value; "
            "not a second accounting ledger"
        ),
    }


def _totals_arrays(totals: Any) -> dict[str, float]:
    return {
        "number_ion": float(totals.number[0]),
        "number_electron": float(totals.number[1]),
        "kinetic_ion": float(totals.kinetic[0]),
        "kinetic_electron": float(totals.kinetic[1]),
        "ion_internal_J": float(totals.ion_internal),
        "electron_perp_J": float(totals.electron_perp),
        "electron_parallel_J": float(totals.electron_parallel),
        "magnetic_J": float(totals.magnetic),
        "total_energy_J": float(totals.energy),
    }


def _references_at(
    specs: Mapping[int, Mapping[str, Any]],
    times: np.ndarray,
) -> dict[int, np.ndarray]:
    output = {}
    for harmonic, spec in specs.items():
        value = spec["eigenvalue"]
        output[harmonic] = np.asarray(
            [
                spec["amplitude"]
                * spec["vector"]
                * np.exp(value * float(moment))
                for moment in times
            ],
            dtype=np.complex128,
        )
    return output


def _accuracy_history(
    actual: np.ndarray,
    expected: np.ndarray,
) -> dict[str, Any]:
    ratios = []
    vector_errors = []
    for observed, reference in zip(actual, expected):
        denominator = np.vdot(reference, reference)
        projection = np.vdot(reference, observed) / denominator
        ratios.append(projection)
        vector_errors.append(
            float(
                np.linalg.norm(observed - reference)
                / max(np.linalg.norm(reference), 1.0e-300)
            )
        )
    ratio = np.asarray(ratios, dtype=np.complex128)
    principal = np.angle(ratio)
    unwrapped = np.unwrap(principal)
    amplitude_drift = np.abs(np.abs(ratio) - 1.0)
    return {
        "relative_vector_error": np.asarray(vector_errors, dtype=np.float64),
        "amplitude_ratio": np.abs(ratio),
        "amplitude_drift": amplitude_drift,
        "phase_error_principal_rad": principal,
        "phase_error_unwrapped_rad": unwrapped,
    }


def _growth_summary(
    history: np.ndarray,
    times: np.ndarray,
    *,
    reference_history: np.ndarray | None,
    initial_ratio_defined: bool,
) -> dict[str, Any]:
    norms = np.linalg.norm(history, axis=1)
    initial = float(norms[0])
    if not initial_ratio_defined or initial <= 0.0:
        result = {
            "available": False,
            "reason": "initial high-harmonic norm is numerical zero",
            "initial_modal_norm": initial,
            "maximum_absolute_norm": float(np.max(norms)),
            "final_absolute_norm": float(norms[-1]),
            "maximum_amplitude_ratio": None,
            "final_amplitude_ratio": None,
            "amplitude_ratio_history": None,
            "normalized_norm_history": norms,
            "matched_synchronous": {
                "available": reference_history is not None,
                "semantics": "compare absolute norms; initial-relative gain is undefined",
            },
        }
        if reference_history is not None:
            reference = np.asarray(reference_history, dtype=np.complex128)
            if reference.shape != history.shape:
                raise ValueError("matched reference is not time-aligned")
            reference_norms = np.linalg.norm(reference, axis=1)
            differences = np.abs(norms - reference_norms)
            result["matched_synchronous"] = {
                "available": True,
                "maximum_absolute_norm_difference": float(
                    np.max(differences)
                ),
                "final_absolute_norm_difference": float(differences[-1]),
                "actual_final_absolute_norm": float(norms[-1]),
                "reference_final_absolute_norm": float(reference_norms[-1]),
                "amplitude_ratio_to_reference": None,
                "semantics": (
                    "target-only high mode is numerical zero; compare absolute "
                    "amplitudes only and do not form an initial-relative gain"
                ),
            }
        return result
    ratios = norms / initial
    maximum_index = int(np.argmax(ratios))
    increments = np.diff(ratios)
    repeated = int(np.count_nonzero(increments > 1.0e-12))
    result = {
        "available": True,
        "initial_modal_norm": initial,
        "maximum_absolute_norm": float(norms[maximum_index]),
        "final_absolute_norm": float(norms[-1]),
        "maximum_amplitude_ratio": float(ratios[maximum_index]),
        "maximum_time_s": float(times[maximum_index]),
        "final_amplitude_ratio": float(ratios[-1]),
        "amplification_over_1_percent": bool(
            float(ratios[maximum_index]) > 1.0 + GROWTH_FLAG_THRESHOLD
        ),
        "repeated_cumulative_growth": bool(
            ratios[-1] > 1.0 + GROWTH_FLAG_THRESHOLD and repeated >= 2
        ),
        "transient_growth_only": bool(
            ratios[maximum_index] > 1.0 + GROWTH_FLAG_THRESHOLD
            and ratios[-1] <= 1.0 + GROWTH_FLAG_THRESHOLD
        ),
        "positive_growth_intervals": repeated,
        "amplitude_ratio_history": ratios,
        "normalized_norm_history": ratios,
        "semantics": (
            "accepted-endpoint nonlinear modal norm relative to its initialized "
            "norm; flags growth but does not identify a physical instability"
        ),
    }
    if reference_history is not None:
        reference = np.asarray(reference_history, dtype=np.complex128)
        if reference.shape != history.shape:
            raise ValueError("matched reference is not time-aligned")
        errors = np.asarray(
            [
                np.linalg.norm(actual - expected)
                / max(np.linalg.norm(expected), 1.0e-300)
                for actual, expected in zip(history, reference)
            ],
            dtype=np.float64,
        )
        actual_endpoint = history[-1]
        reference_endpoint = reference[-1]
        ref_norm = float(np.linalg.norm(reference_endpoint))
        actual_norm = float(np.linalg.norm(actual_endpoint))
        reference_norms = np.linalg.norm(reference, axis=1)
        matched_ratios = norms / np.maximum(reference_norms, 1.0e-300)
        maximum_ratio = float(np.max(matched_ratios))
        final_ratio = float(matched_ratios[-1])
        maximum_growth = maximum_ratio > 1.0 + GROWTH_FLAG_THRESHOLD
        final_growth = final_ratio > 1.0 + GROWTH_FLAG_THRESHOLD
        result["matched_synchronous"] = {
            "available": True,
            "relative_modal_vector_error": float(
                np.linalg.norm(actual_endpoint - reference_endpoint)
                / max(ref_norm, 1.0e-300)
            ),
            "maximum_relative_modal_vector_error": float(np.max(errors)),
            "aligned_sample_count": int(reference.shape[0]),
            "amplitude_ratio_to_reference": actual_norm / max(ref_norm, 1.0e-300),
            "maximum_amplitude_ratio_to_reference": maximum_ratio,
            "final_amplitude_ratio_to_reference": final_ratio,
            "maximum_amplification_over_1_percent": maximum_growth,
            "final_amplification_over_1_percent": final_growth,
            "transient_amplification_over_1_percent": maximum_growth and not final_growth,
            "amplification_over_1_percent": maximum_growth,
        }
    else:
        result["matched_synchronous"] = {
            "available": False,
            "reason": "no --reference-case supplied",
        }
    return result


def align_reference_history(
    candidate_times: Sequence[float],
    reference_arrays: Mapping[str, np.ndarray],
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Select reference rows at every exact validated candidate time."""
    candidate = np.asarray(candidate_times, dtype=np.float64)
    if candidate.ndim != 1:
        raise ValueError("candidate times must be one-dimensional")
    if "accepted_time_s" not in reference_arrays or key not in reference_arrays:
        raise ValueError("reference history lacks times or requested data")
    reference_time = np.asarray(
        reference_arrays["accepted_time_s"], dtype=np.float64
    )
    values = np.asarray(reference_arrays[key], dtype=np.complex128)
    if reference_time.ndim != 1 or values.shape[0] != reference_time.size:
        raise ValueError("reference history has inconsistent lengths")
    indexes = []
    for moment in candidate:
        matches = np.flatnonzero(
            np.isclose(reference_time, moment, rtol=2.0e-13, atol=0.0)
        )
        if matches.size == 0:
            raise ValueError(f"missing candidate time {float(moment):.17g}")
        indexes.append(int(matches[0]))
    if len(set(indexes)) != len(indexes):
        raise ValueError("reference time matching is not one-to-one")
    return candidate.copy(), values[indexes].copy()


def macro_alias_metadata(
    specs: Mapping[int, Mapping[str, Any]],
    macro_dt: float,
) -> dict[str, dict[str, Any]]:
    """Report a harmonic's macro-step phase and folded Nyquist alias."""
    macro_dt = _finite("macro_dt", macro_dt, positive=True)
    result = {}
    for harmonic, spec in specs.items():
        omega = abs(float(spec["eigenvalue"].imag))
        phase = omega * macro_dt
        folded = abs(
            math.remainder(phase, 2.0 * math.pi)
        ) / macro_dt
        result[str(harmonic)] = {
            "omega_rad_s": omega,
            "macro_dt_s": macro_dt,
            "omega_macro_dt_rad": phase,
            "macro_samples_per_cycle": (
                2.0 * math.pi / phase if phase > 0.0 else math.inf
            ),
            "macro_nyquist_rad_s": math.pi / macro_dt,
            "macro_nyquist_alias_rad_s": folded,
            "semantics": (
                "macro endpoint phase is aliased when omega*H approaches pi; "
                "do not infer the physical frequency from unwrapped macro phase"
            ),
        }
    return result


def _runtime_source_hashes() -> dict[str, str]:
    paths = set(SOURCE_FILES)
    paths.update(str(path.relative_to(ROOT)) for path in (ROOT / "src/fomhd").rglob("*.py"))
    paths.add("uv.lock")
    hashes = {
        relative: _sha256(ROOT / relative)
        for relative in sorted(paths)
        if (ROOT / relative).is_file()
    }
    hashes["tools/warp_t0_controlled.py"] = _sha256(Path(__file__).resolve())
    return hashes


def _case_identity(case: HallCase, seed: str) -> dict[str, Any]:
    return {
        "physical": {
            "kdi": float(case.kdi),
            "points": int(case.points),
            "n0": float(case.n0),
            "B0": float(case.B0),
            "pressure": float(case.pressure),
            "K": 0,
            "Nphi": 1,
            "di": float(case.di),
            "k": float(case.k),
            "wavelength": float(case.wavelength),
        },
        "discrete": {
            "points": int(case.points),
            "K": 0,
            "Nphi": 1,
            "harmonics": [1, (case.points - 1) // 2],
            "target_amplitude": TARGET_AMPLITUDE,
            "high_amplitude": HIGH_AMPLITUDE if seed == "mixed" else 0.0,
        },
    }


def _decode_complex_json(value: Any) -> Any:
    if isinstance(value, Mapping) and set(value) == {"real", "imag"}:
        return complex(float(value["real"]), float(value["imag"]))
    if isinstance(value, list):
        return [_decode_complex_json(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _decode_complex_json(item)
            for key, item in value.items()
        }
    return value


def _seed_identity(seed: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "normalization": seed["normalization"],
        "vectors": {
            str(key): np.asarray(value, dtype=np.complex128)
            for key, value in seed["vectors"].items()
        },
        "transverse_vectors": {
            str(key): np.asarray(value, dtype=np.complex128)
            for key, value in seed["transverse_vectors"].items()
        },
        "eigenvalues": {
            str(key): complex(value) for key, value in seed["eigenvalues"].items()
        },
        "amplitudes": {
            str(key): float(value) for key, value in seed["amplitudes"].items()
        },
        "projected_coefficients": {
            str(key): np.asarray(value, dtype=np.complex128)
            for key, value in seed["projected_coefficients"].items()
        },
    }


def _validate_seed_identity(reference: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    reference = _decode_complex_json(reference)
    if reference.get("normalization") != expected["normalization"]:
        raise ValueError("matched reference seed normalization differs")
    for name in ("vectors", "transverse_vectors", "projected_coefficients"):
        if set(reference.get(name, {})) != set(expected[name]):
            raise ValueError(f"matched reference seed {name} keys differ")
        if any(
            not np.array_equal(
                np.asarray(reference[name][key], dtype=np.complex128),
                expected[name][key],
            )
            for key in expected[name]
        ):
            raise ValueError(f"matched reference seed {name} differ")
    if reference.get("amplitudes") != expected["amplitudes"]:
        raise ValueError("matched reference seed amplitudes differ")
    if reference.get("eigenvalues") != expected["eigenvalues"]:
        raise ValueError("matched reference seed eigenvalues differ")


def _load_matched_reference(
    path: Path | None,
    *,
    seed: str,
    horizon: float,
    points: int,
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    if path is None:
        return None, {"available": False, "reason": "not supplied"}
    directory = Path(path)
    case_path = directory / "case.json"
    history_path = directory / "history.npz"
    if not case_path.is_file() or not history_path.is_file():
        raise ValueError("reference-case must contain case.json and history.npz")
    metadata = json.loads(case_path.read_text(encoding="utf-8"))
    controls = metadata.get("controls", {})
    if controls.get("coupling") != "synchronous":
        raise ValueError("reference-case coupling must be synchronous")
    if controls.get("seed") != seed:
        raise ValueError("reference-case seed differs")
    if int(controls.get("points", -1)) != points:
        raise ValueError("reference-case points differ")
    if not math.isclose(
        float(metadata.get("schedule", {}).get("horizon_s", -1.0)),
        horizon,
        rel_tol=2.0e-12,
        abs_tol=0.0,
    ):
        raise ValueError("reference-case horizon differs")
    reference_source = metadata.get("source", {})
    if reference_source.get("source_hashes") != _runtime_source_hashes():
        raise ValueError(
            "matched reference numerical runtime source hashes differ"
        )
    case = _periodic_case(points)
    expected_identity = _case_identity(case, seed)
    if metadata.get("physical") != expected_identity["physical"]:
        raise ValueError("matched reference physical parameters differ")
    if metadata.get("discrete") != expected_identity["discrete"]:
        raise ValueError("matched reference discrete parameters differ")
    expected_seed = _seed_identity(build_seed(points=points, seed=seed))
    _validate_seed_identity(metadata.get("seed", {}), expected_seed)
    reference_commit = (
        reference_source
        .get("source_git", {})
        .get("commit")
    )
    current_commit = _source_git().get("commit")
    if reference_commit != current_commit:
        raise ValueError("reference-case source commit differs")
    with np.load(history_path, allow_pickle=False) as arrays:
        stored = {name: np.asarray(arrays[name]) for name in arrays.files}
    times = np.asarray(stored.get("accepted_time_s", ()), dtype=np.float64)
    if (
        times.ndim != 1
        or times.size < 2
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
        or not math.isclose(times[0], 0.0, rel_tol=0.0, abs_tol=1.0e-15)
        or not math.isclose(
            float(times[-1]),
            horizon,
            rel_tol=2.0e-12,
            abs_tol=0.0,
        )
    ):
        raise ValueError(
            "matched reference time coverage must span zero through the horizon"
        )
    return stored, {
        "available": True,
        "path": str(directory),
        "status": metadata.get("status"),
    }


def _artifact_size(path: Path) -> int:
    return int(path.stat().st_size) if path.is_file() else 0


def run_case(
    output: str | Path,
    *,
    device: str = "cpu",
    coupling: str = "predictor",
    micro_per_period: int = 2048,
    micro_steps: int = 128,
    cycles: int = 2,
    seed: str = "target",
    points: int = 9,
    progress_every: int | None = None,
    reference_case: str | Path | None = None,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run one fresh production case or retain its rejected history."""
    _validate_device(device)
    if seed not in VALID_SEEDS:
        raise ValueError(f"seed must be one of {VALID_SEEDS}")
    case_schedule = schedule(
        coupling=coupling,
        micro_per_period=micro_per_period,
        micro_steps=micro_steps,
        cycles=cycles,
        points=points,
    )
    if progress_every is None:
        progress_every = max(1, case_schedule["macro_steps"] // 10)
    output = _require_fresh(Path(output))
    reference_arrays, reference_metadata = (
        _load_matched_reference(
            Path(reference_case) if reference_case is not None else None,
            seed=seed,
            horizon=case_schedule["horizon_s"],
            points=points,
        )
    )
    output.mkdir(parents=True, exist_ok=False)
    invocation = list(argv) if argv is not None else [str(Path(__file__))]
    provenance = _provenance(invocation, device)
    prepared = None
    accepted = 0
    calls = 0
    failure = None
    failure_traceback = ""
    checks_path = output / "checks.jsonl.gz"
    checks_stream = gzip.open(checks_path, "xt", encoding="utf-8")
    check_accumulator = _CheckAccumulator()
    check_count = 0
    metadata_records: list[Mapping[str, Any]] = []
    times = []
    high_harmonic = (points - 1) // 2
    harmonics = (TARGET_HARMONIC, high_harmonic)
    mode9_history: dict[int, list[np.ndarray]] = {
        harmonic: [] for harmonic in harmonics
    }
    transverse_history: dict[int, list[np.ndarray]] = {
        harmonic: [] for harmonic in harmonics
    }
    full_state_history: dict[str, list[np.ndarray]] = {}
    totals_history = []
    started = time.perf_counter()
    try:
        prepared = _prepare_case(
            coupling=coupling,
            micro_per_period=micro_per_period,
            micro_steps=micro_steps,
            cycles=cycles,
            seed=seed,
            points=points,
        )
        context = prepared.context
        case = prepared.case
        check_count += _stream_checks(
            checks_stream,
            "initial",
            prepared.initial_checks,
            check_accumulator,
        )
        initial_state = prepared.committed.state
        times.append(float(initial_state.time))
        for harmonic in harmonics:
            mode9 = project_state(initial_state, context, case, harmonic)
            mode9_history[harmonic].append(mode9)
            transverse_history[harmonic].append(
                mode9[list(TRANSVERSE_INDEXES)]
            )
        initial_arrays = _state_arrays(initial_state, context, "initial")
        for name, value in initial_arrays.items():
            full_state_history.setdefault(name, []).append(value)
        if prepared.initial_totals is None:
            raise RuntimeError("initial stage totals are unavailable")
        totals_history.append(_totals_arrays(prepared.initial_totals))
        for macro in range(case_schedule["macro_steps"]):
            calls += 1
            result = advance(
                prepared.committed,
                prepared.cumulative,
                context,
                None,
            )
            accepted += 1
            check_count += _stream_checks(
                checks_stream,
                f"macro_{macro}",
                result.checks,
                check_accumulator,
            )
            metadata = getattr(result, "coupling_metadata", None)
            if metadata is not None:
                metadata_records.append(metadata)
            state = prepared.committed.state
            times.append(float(state.time))
            for harmonic in harmonics:
                mode9 = project_state(state, context, case, harmonic)
                mode9_history[harmonic].append(mode9)
                transverse_history[harmonic].append(
                    mode9[list(TRANSVERSE_INDEXES)]
                )
            for name, value in _state_arrays(
                state, context, "accepted"
            ).items():
                full_state_history.setdefault(name, []).append(value)
            totals_history.append(_totals_arrays(result.totals))
            if coupling == "predictor" and metadata is not None:
                _validate_predictor_metadata(metadata, micro_steps)
            _case_progress(
                accepted,
                case_schedule["macro_steps"],
                progress_every,
            )
    except Exception as error:
        failure_traceback = traceback.format_exc()
        if isinstance(error, Failure):
            failure = error
        else:
            failure = Failure("NumericalLibrary", str(error))
            failure.stage = getattr(error, "stage", None)
        check_count += _stream_checks(
            checks_stream,
            "failure",
            getattr(failure, "checks", ()),
            check_accumulator,
        )
    finally:
        checks_stream.close()
    elapsed = time.perf_counter() - started
    checks_summary = check_accumulator.finish()
    times_array = np.asarray(times, dtype=np.float64)
    mode9_arrays = {
        harmonic: np.asarray(values, dtype=np.complex128).reshape(-1, 9)
        for harmonic, values in mode9_history.items()
    }
    transverse_arrays = {
        harmonic: np.asarray(values, dtype=np.complex128).reshape(-1, 4)
        for harmonic, values in transverse_history.items()
    }
    target_array = transverse_arrays[TARGET_HARMONIC]
    high_array = transverse_arrays[high_harmonic]
    arrays: dict[str, Any] = {
        "accepted_time_s": times_array,
        "target_coeff_history": target_array,
        "high_coeff_history": high_array,
        "high_abs_amplitude_history": np.abs(high_array),
        "high_norm_history": np.linalg.norm(high_array, axis=1),
        **{
            f"mode_m{harmonic}_coeff_history": values
            for harmonic, values in mode9_arrays.items()
        },
    }
    arrays.update(
        {
            name: np.asarray(values)
            for name, values in full_state_history.items()
        }
    )
    if totals_history:
        for name in totals_history[0]:
            arrays[f"totals_{name}_history"] = np.asarray(
                [item[name] for item in totals_history], dtype=np.float64
            )
    cumulative_metadata = {"available": False}
    if prepared is not None:
        arrays.update(
            _cumulative_arrays(prepared.cumulative, prepared.committed)
        )
        cumulative_metadata = _cumulative_metadata(
            prepared.cumulative, prepared.committed
        )
    if prepared is not None:
        arrays.update(
            {
                **{
                    f"seed_{name}": value
                    for name, value in _state_arrays(
                        prepared.raw, prepared.context, "raw"
                    ).items()
                },
                "seed_state9": mode9_history[1][0],
                "seed_target_coeff": transverse_history[1][0],
                "seed_high_coeff": transverse_history[high_harmonic][0],
            }
        )
    partial_names: list[str] = []
    if failure is not None:
        for name, value in getattr(failure, "partial_arrays", {}).items():
            array = _array(value)
            if array is not None:
                arrays[name] = array
                partial_names.append(name)
    accuracy = None
    qualification = None
    per_harmonic_error_max = {}
    full_success = (
        failure is None and accepted == case_schedule["macro_steps"]
    )
    if prepared is not None and times:
        reference_specs = dict(prepared.mode_specs)
        if high_harmonic not in reference_specs:
            high_matrix, _ = _independent_matrix_for_k(
                case, high_harmonic * case.k
            )
            high_value, high_vector = independent_whistler(high_matrix)
            embedded_high = np.zeros(9, dtype=np.complex128)
            embedded_high[list(TRANSVERSE_INDEXES)] = high_vector
            reference_specs[high_harmonic] = {
                "amplitude": 0.0,
                "vector": embedded_high,
                "transverse_vector": high_vector,
                "eigenvalue": high_value,
            }
        references = _references_at(reference_specs, times_array)
        target_accuracy = _accuracy_history(
            target_array,
            references[TARGET_HARMONIC][:, list(TRANSVERSE_INDEXES)],
        )
        arrays["target_reference_history"] = references[TARGET_HARMONIC][
            :, list(TRANSVERSE_INDEXES)
        ]
        arrays["target_vector_error_history"] = target_accuracy[
            "relative_vector_error"
        ]
        arrays["target_amplitude_drift_history"] = target_accuracy[
            "amplitude_drift"
        ]
        arrays["target_phase_error_unwrapped_history"] = target_accuracy[
            "phase_error_unwrapped_rad"
        ]
        for harmonic in harmonics:
            expected_mode = references[harmonic]
            actual_mode = mode9_arrays[harmonic]
            differences = np.linalg.norm(
                actual_mode - expected_mode, axis=1
            )
            expected_norms = np.linalg.norm(expected_mode, axis=1)
            defined = expected_norms > 1.0e-25
            errors = np.full(differences.shape, np.nan, dtype=np.float64)
            errors[defined] = (
                differences[defined] / expected_norms[defined]
            )
            per_harmonic_error_max[str(harmonic)] = {
                "relative_max": (
                    float(np.max(errors[defined]))
                    if np.any(defined)
                    else None
                ),
                "absolute_max": float(np.max(differences)),
                "zero_reference_semantics": (
                    "relative error undefined; absolute error retained"
                    if not np.all(defined)
                    else "relative error defined at every accepted time"
                ),
            }
            arrays[f"mode_m{harmonic}_reference_history"] = expected_mode
            arrays[f"mode_m{harmonic}_vector_error_history"] = errors
        qualification = qualify_accuracy(
            vector_error=float(
                np.max(target_accuracy["relative_vector_error"])
            ),
            amplitude_drift=float(np.max(target_accuracy["amplitude_drift"])),
            phase_error_rad=float(
                np.max(
                    np.abs(target_accuracy["phase_error_unwrapped_rad"])
                )
            ),
        )
        if not full_success:
            qualification.update(
                joint_qualified=False,
                prefix_joint_qualified=qualification["joint_qualified"],
            )
        accuracy = {
            **qualification,
            "partial": not full_success,
            "full_window_status": (
                "UNDETERMINED"
                if not full_success
                else (
                    "QUALIFIED"
                    if qualification["joint_qualified"]
                    else "FAILED"
                )
            ),
            "per_harmonic_reference_max_relative_error": (
                per_harmonic_error_max
            ),
            "history_definition": (
                "maximum over all accepted macro endpoints including t=0; "
                "target is the normalized m=1 four-component transverse vector"
            ),
            "phase_definition": (
                "projection ratio to exact independent cold-Hall REF, then "
                "cumulative np.unwrap over accepted time history"
            ),
        }
    reference_history = None
    if reference_arrays is not None and high_array.size:
        _, reference_history = align_reference_history(
            times_array,
            reference_arrays,
            "high_coeff_history",
        )
    if high_array.size:
        high_growth = _growth_summary(
            high_array,
            times_array,
            reference_history=reference_history,
            initial_ratio_defined=(seed == "mixed"),
        )
        high_growth["partial"] = not full_success
        ratio_history = high_growth.get("amplitude_ratio_history")
        if ratio_history is not None:
            arrays["high_amplitude_ratio_history"] = np.asarray(ratio_history)
        arrays["high_norm_relative_initial_history"] = np.asarray(
            high_growth["normalized_norm_history"]
        )
    else:
        high_growth = {
            "available": False,
            "reason": "no accepted state was initialized",
            "partial": True,
        }
    history_path = output / "history.npz"
    _save_npz(history_path, arrays)
    failure_path = None
    failure_npz_path = None
    if failure is not None:
        failure_payload = {
            "case_id": f"t0_{coupling}_mpp{micro_per_period}_M{micro_steps}_{seed}",
            "status": "UNDETERMINED",
            "failure_code": failure.code,
            "failure_type": type(failure).__name__,
            "message": str(failure),
            "failure_stage": _jsonable(failure.stage),
            "accepted_macro_steps": accepted,
            "requested_macro_steps": case_schedule["macro_steps"],
            "last_accepted_time_s": times[-1] if times else None,
            "final_endpoint_available": False,
            "invented_rejected_endpoint": False,
            "partial_array_names": sorted(partial_names),
            "checks": _jsonable(getattr(failure, "checks", ())),
            "traceback": failure_traceback,
        }
        failure_path = output / "failure.json"
        _write_json(failure_path, failure_payload)
        if partial_names:
            failure_npz_path = output / "failure.npz"
            _save_npz(
                failure_npz_path,
                {
                    name: arrays[name]
                    for name in partial_names
                    if name in arrays
                },
            )
    accepted_metadata = None
    if coupling == "predictor" and metadata_records:
        integer_keys = (
            "complete_stage_evaluations",
            "full_rhs_stage_evaluations",
            "endpoint_full_evaluations",
            "alias_evaluations",
            "total_model_evaluations",
            "retained_quadrature_entries",
            "assemble_canonicalization_calls",
        )
        accepted_metadata = {
            "per_macro_available": True,
            "accepted_records": len(metadata_records),
            **{
                key: int(sum(int(record.get(key, 0)) for record in metadata_records))
                for key in integer_keys
            },
            "endpoint_tangent_correction": metadata_records[-1].get(
                "endpoint_tangent_correction"
            ),
        }
    else:
        accepted_metadata = {
            "per_macro_available": False,
            "reason": "A0 synchronous stage metadata is not emitted",
        }
    if failure is not None or accepted != case_schedule["macro_steps"]:
        status = "UNDETERMINED"
    elif checks_summary["soft_failed"]:
        status = "COMPLETED_WITH_SOFT_FAILURES"
    else:
        status = "COMPLETED"
    alias_specs = {}
    prepared_identity = {"physical": None, "discrete": None}
    if prepared is not None:
        prepared_identity = _case_identity(prepared.case, seed)
        alias_specs.update(prepared.mode_specs)
        if high_harmonic not in alias_specs:
            high_matrix, _ = _independent_matrix_for_k(
                case, high_harmonic * case.k
            )
            high_value, _ = independent_whistler(high_matrix)
            alias_specs[high_harmonic] = {"eigenvalue": high_value}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "case_id": f"t0_{coupling}_mpp{micro_per_period}_M{micro_steps}_{seed}",
        "scope": SCOPE,
        "controls": {
            "device": device,
            "coupling": coupling,
            "micro_per_period": micro_per_period,
            "micro_steps": micro_steps,
            "cycles": cycles,
            "seed": seed,
            "points": points,
            "target_amplitude": TARGET_AMPLITUDE,
            "high_amplitude": HIGH_AMPLITUDE if seed == "mixed" else 0.0,
            "target_harmonic": TARGET_HARMONIC,
            "high_harmonic": high_harmonic,
            "dt_guard": "error",
            "original_budgets_preserved": prepared is not None,
            "memory_cap_bytes": MEMORY_CAP_BYTES,
            "memory_cap_semantics": (
                "numeric resource cap only; no scientific threshold changed"
            ),
        },
        "schedule": case_schedule,
        "source": provenance,
        "physical": prepared_identity["physical"],
        "discrete": prepared_identity["discrete"],
        "cumulative": cumulative_metadata,
        "seed": (
            _seed_identity(prepared.seed) if prepared is not None else {}
        ),
        "budgets": prepared.budgets if prepared is not None else {},
        "accepted_macro_steps": accepted,
        "requested_macro_steps": case_schedule["macro_steps"],
        "last_accepted_time_s": times[-1] if times else None,
        "final_endpoint_available": status.startswith("COMPLETED"),
        "invented_rejected_endpoint": False,
        "accuracy": accuracy,
        "per_harmonic_reference_max_relative_error": (
            per_harmonic_error_max
        ),
        "macro_alias_by_harmonic": (
            macro_alias_metadata(
                alias_specs, case_schedule["macro_dt_s"]
            )
            if alias_specs
            else {}
        ),
        "high_frequency_growth": high_growth,
        "check_summary": checks_summary,
        "energy": {
            "totals_history_keys": (
                [
                    name
                    for name in totals_history[0]
                    if name.endswith("_J")
                ]
                if totals_history
                else []
            ),
            "raw_checks_file": checks_path.name,
            "original_thresholds": prepared.budgets if prepared is not None else {},
            "semantics": (
                "raw J-unit checks and totals retained; no tolerance changed"
            ),
        },
        "counts": {
            "advance_calls": calls,
            "accepted_macro_steps": accepted,
            "raw_check_records": check_count,
            "stage_metadata": accepted_metadata,
        },
        "timing": {"advance_wall_seconds": elapsed},
        "reference": {
            "comparison_semantics": (
                "analytic time REF uses the independent cold4x4 Hall "
                "eigenvector/eigenvalue embedded in transverse9 coordinates; "
                "production 9x9 spectra are LIN audit evidence only"
            ),
            "matched_case": reference_metadata,
        },
        "failure": (
            {
                "failure_code": failure.code,
                "failure_file": failure_path.name,
                "failure_arrays_file": (
                    failure_npz_path.name if failure_npz_path is not None else None
                ),
            }
            if failure is not None
            else None
        ),
        "artifacts": {
            "history_npz": history_path.name,
            "checks_jsonl_gzip": checks_path.name,
            "history_bytes": _artifact_size(history_path),
            "checks_bytes": _artifact_size(checks_path),
        },
        "limitations": [
            "periodic temporal-isolation fixture only",
            "m is a spatial harmonic, not toroidal K",
            "4x4 transverse REF does not cover pressure modes",
            "linear spectra and macro map do not prove nonlinear stability",
            "not a CAW CFL validation",
        ],
    }
    case_path = output / "case.json"
    _write_json(case_path, summary)
    summary["artifacts"]["case_json"] = case_path.name
    summary["artifacts"]["case_json_bytes"] = _artifact_size(case_path)
    return summary


def run_frozen_micro_control(
    output: str | Path,
    *,
    micro_per_period: int,
    steps: int | None = None,
    points: int = 9,
    device: str = "cpu",
    progress_every: int | None = None,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the prescribed-ion electron/field micro RK4 control."""
    _validate_device(device)
    if (
        isinstance(micro_per_period, bool)
        or type(micro_per_period) is not int
        or micro_per_period not in VALID_MICRO_PER_PERIOD
    ):
        raise ValueError(
            f"micro_per_period must be one of {VALID_MICRO_PER_PERIOD}"
        )
    case = _periodic_case(points)
    harmonic = (points - 1) // 2
    target_value, _ = independent_whistler(cold_transverse_matrix(case))
    period = 2.0 * math.pi / float(target_value.imag)
    micro_dt = period / float(micro_per_period)
    default_steps = micro_per_period // 2
    steps = default_steps if steps is None else _positive_integer("steps", steps)
    if steps > default_steps:
        raise ValueError("steps must not exceed the fixed T/2 micro horizon")
    if progress_every is None:
        progress_every = max(1, steps // 10)
    output = _require_fresh(Path(output))
    output.mkdir(parents=True, exist_ok=False)
    invocation = list(argv) if argv is not None else [str(Path(__file__))]
    provenance = _provenance(invocation, device)
    checks_path = output / "checks.jsonl.gz"
    checks_stream = gzip.open(checks_path, "xt", encoding="utf-8")
    accumulator = _CheckAccumulator()
    check_count = 0
    accepted = 0
    failure = None
    failure_traceback = ""
    times = [0.0]
    z_history = []
    rk4_reference = []
    exponential_reference = []
    p_errors = []
    seed_state9 = None
    micro_matrix = None
    micro_eigenvalue = None
    micro_eigenvector = None
    started = time.perf_counter()
    try:
        context = build_periodic_context(case, device="cpu")
        budgets_before = _budget_snapshot(context.p)
        context.p.memory_limit = MEMORY_CAP_BYTES
        context.p.device_memory_limit = MEMORY_CAP_BYTES
        context.p.integrator = "rk4"
        context.p.coupling = "predictor"
        context.p.micro_steps = 1
        context.p.dt = micro_dt
        context.p.nsteps = steps
        context.p.dt_guard = "error"
        equilibrium = uniform_state(context, case)
        micro_matrix, _ = production_jacobian(
            context,
            case,
            equilibrium,
            harmonic,
            EPSILONS[0],
            columns=Z_INDEXES,
            rows=Z_INDEXES,
        )
        values, vectors = np.linalg.eig(micro_matrix)
        positive = np.flatnonzero(values.imag > 0.0)
        if positive.size == 0:
            raise ValueError("frozen micro Jacobian has no positive mode")
        index = int(positive[np.argmax(values[positive].imag)])
        micro_eigenvalue = complex(values[index])
        micro_eigenvector = _normalized_eigenvector(vectors[:, index])
        seed_z = HIGH_AMPLITUDE * micro_eigenvector
        seed9 = np.zeros(9, dtype=np.complex128)
        seed9[list(Z_INDEXES)] = seed_z
        raw = _state_with_modal_coefficients(
            equilibrium, seed9, harmonic, case, context
        )
        captured = {}
        committed = initialize(
            raw,
            context,
            cumulative_record(),
            equilibrium=False,
            captured=captured,
        )
        initial_state = committed.state
        seed_state9 = project_state(initial_state, context, case, harmonic)
        p_initial = (
            initial_state.fluid[0].n,
            initial_state.fluid[0].mom,
            initial_state.fluid[0].perp,
        )
        z_seed = project_state(initial_state, context, case, harmonic)[
            list(Z_INDEXES)
        ]
        if not np.all(np.isfinite(z_seed)):
            raise ValueError("frozen seed is invalid")
        base_tag = time_predictor._tag(
            context,
            0,
            initial_state.generation,
            1,
            0.0,
            "frozen_micro_base",
            0,
        )
        base = evaluate_stage(initial_state, base_tag, context, False)
        time_predictor._check_guard(base, context)
        diagnostics.gate(base.checks)
        check_count += _stream_checks(
            checks_stream, "initial", base.checks, accumulator
        )
        check_count += _stream_checks(
            checks_stream,
            "initialization",
            captured.get("checks", ()),
            accumulator,
        )
        z_history.append(z_seed)
        rk4_reference.append(z_seed.copy())
        exponential_reference.append(z_seed.copy())
        p_errors.append(
            float(
                np.max(
                    np.abs(
                        project_state(
                            initial_state, context, case, harmonic
                        )[:5]
                        - seed_state9[:5]
                    )
                )
            )
        )
        state = initial_state
        for micro in range(steps):
            z_current = (
                state.fluid[1].perp,
                state.fluid[1].parallel,
                state.B,
            )
            derivatives = []
            for stage_index in range(4):
                if micro == 0 and stage_index == 0:
                    stage = base
                else:
                    z_stage = time_predictor._rk_stage_state(
                        context,
                        z_current,
                        derivatives,
                        stage_index,
                        micro_dt,
                    )
                    generation = int(context.next_generation)
                    context.next_generation = generation + 1
                    stage_time = core.microstage_time(
                        context.p, micro, 0, stage_index
                    )
                    tag = time_predictor._tag(
                        context,
                        micro,
                        generation,
                        1,
                        stage_time,
                        f"frozen_m{micro:04d}_r{stage_index + 1}",
                        micro,
                    )
                    stage_state = time_predictor._assemble_state(
                        context, p_initial, z_stage, tag
                    )
                    stage = evaluate_stage(
                        stage_state,
                        tag,
                        context,
                        stage_index in (1, 2),
                    )
                    diagnostics.gate(stage.checks)
                    check_count += _stream_checks(
                        checks_stream,
                        f"micro_{micro}_stage_{stage_index + 1}",
                        stage.checks,
                        accumulator,
                    )
                derivatives.append(
                    (
                        stage.rhs.fluid[1].perp,
                        stage.rhs.fluid[1].parallel,
                        stage.rhs.B,
                    )
                )
            z_next = time_predictor._rk_next(
                context, z_current, derivatives, micro_dt
            )
            generation = int(context.next_generation)
            context.next_generation = generation + 1
            end_time = core.microstage_time(context.p, micro, 0, 3)
            tag = time_predictor._tag(
                context,
                micro + 1,
                generation,
                1,
                end_time,
                f"frozen_m{micro:04d}_endpoint",
                micro,
            )
            state = time_predictor._assemble_state(
                context, p_initial, z_next, tag
            )
            endpoint = evaluate_stage(state, tag, context, False)
            diagnostics.gate(endpoint.checks)
            check_count += _stream_checks(
                checks_stream,
                f"micro_{micro}_endpoint",
                endpoint.checks,
                accumulator,
            )
            accepted += 1
            actual9 = project_state(state, context, case, harmonic)
            times.append(float((micro + 1) * micro_dt))
            z_history.append(actual9[list(Z_INDEXES)])
            rk4_reference.append(
                np.asarray(
                    np.linalg.matrix_power(
                        rk4_matrix_step(micro_matrix, micro_dt),
                        accepted,
                    )
                    @ z_seed
                )
            )
            exponential_reference.append(
                np.asarray(expm(micro_matrix * times[-1]) @ z_seed)
            )
            p_errors.append(
                float(
                    np.max(
                        np.abs(actual9[:5] - seed_state9[:5])
                    )
                )
            )
            if (
                progress_every > 0
                and (
                    accepted == steps
                    or accepted == 1
                    or accepted % progress_every == 0
                )
            ):
                print(
                    f"accepted {accepted}/{steps} frozen micro steps",
                    file=sys.stderr,
                )
        if _budget_snapshot(context.p) != budgets_before:
            raise RuntimeError("original solver budgets or hard flags changed")
    except Exception as error:
        failure_traceback = traceback.format_exc()
        failure = (
            error
            if isinstance(error, Failure)
            else Failure("NumericalLibrary", str(error))
        )
        check_count += _stream_checks(
            checks_stream,
            "failure",
            getattr(failure, "checks", ()),
            accumulator,
        )
    finally:
        checks_stream.close()
    elapsed = time.perf_counter() - started
    check_summary = accumulator.finish()
    full_horizon = accepted == default_steps
    if failure is not None:
        status = "UNDETERMINED"
    elif check_summary["soft_failed"]:
        status = "COMPLETED_WITH_SOFT_FAILURES"
    else:
        status = "COMPLETED"
    z_array = np.asarray(z_history, dtype=np.complex128).reshape(-1, 4)
    rk_array = np.asarray(
        rk4_reference, dtype=np.complex128
    ).reshape(-1, 4)
    exp_array = np.asarray(
        exponential_reference, dtype=np.complex128
    ).reshape(-1, 4)
    final_errors = {
        "rk4_relative": (
            float(
                np.linalg.norm(z_array[-1] - rk_array[-1])
                / max(np.linalg.norm(rk_array[-1]), 1.0e-300)
            )
            if z_array.size
            else None
        ),
        "exponential_relative": (
            float(
                np.linalg.norm(z_array[-1] - exp_array[-1])
                / max(np.linalg.norm(exp_array[-1]), 1.0e-300)
            )
            if z_array.size
            else None
        ),
    }
    arrays = {
        "accepted_time_s": np.asarray(times[: len(z_array)]),
        "z_coeff_history": z_array,
        "rk4_reference_history": rk_array,
        "exponential_reference_history": exp_array,
        "p_fixed_error_history": np.asarray(p_errors, dtype=np.float64),
        "seed_state9": (
            np.asarray(seed_state9)
            if seed_state9 is not None
            else np.empty(0, dtype=np.complex128)
        ),
        "micro_jacobian": (
            micro_matrix
            if micro_matrix is not None
            else np.empty((0, 0), dtype=np.complex128)
        ),
    }
    partial_names = []
    if failure is not None:
        for name, value in getattr(failure, "partial_arrays", {}).items():
            array = _array(value)
            if array is not None:
                arrays[name] = array
                partial_names.append(name)
    history_path = output / "history.npz"
    _save_npz(history_path, arrays)
    failure_path = None
    if failure is not None:
        failure_path = output / "failure.json"
        _write_json(
            failure_path,
            {
                "status": "UNDETERMINED",
                "failure_code": failure.code,
                "message": str(failure),
                "failure_stage": _jsonable(failure.stage),
                "accepted_micro_steps": accepted,
                "requested_micro_steps": steps,
                "partial_array_names": sorted(partial_names),
                "checks": _jsonable(getattr(failure, "checks", ())),
                "traceback": failure_traceback,
            },
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "frozen_ion_micro_control",
        "status": status,
        "scope": (
            "Prescribed-ion frozen electron/field micro RK4 experiment using "
            "the production massless stage. P never advances. This is not a "
            "coupled T0 trajectory, accepted energy ledger, CAW validation, "
            "or nonlinear stability proof."
        ),
        "controls": {
            "device": device,
            "points": points,
            "harmonic": harmonic,
            "micro_per_period": micro_per_period,
            "micro_dt_s": micro_dt,
            "requested_steps": steps,
            "fixed_horizon_steps": default_steps,
            "fixed_horizon_s": default_steps * micro_dt,
            "partial_smoke": not full_horizon,
            "amplitude": HIGH_AMPLITUDE,
            "dt_guard": "error",
            "memory_cap_bytes": MEMORY_CAP_BYTES,
        },
        "accepted_micro_steps": accepted,
        "final_endpoint_available": status.startswith("COMPLETED"),
        "full_fixed_horizon_completed": full_horizon,
        "full_fixed_horizon_status": (
            "COMPLETED" if full_horizon else "UNDETERMINED_PARTIAL_SMOKE"
        ),
        "invented_rejected_endpoint": False,
        "micro_eigenvalue_rad_s": micro_eigenvalue,
        "micro_eigenvector": micro_eigenvector,
        "normalization": (
            "production frozen4x4 fast eigenvector maxabs1/pivotphase0; "
            "seed coefficient is 2e-8 times this vector"
        ),
        "p_fixed_max_abs": (
            float(np.max(np.abs(p_errors))) if p_errors else None
        ),
        "final_errors": final_errors,
        "check_summary": check_summary,
        "counts": {
            "accepted_micro_steps": accepted,
            "raw_check_records": check_count,
            "time_advance_calls": 0,
        },
        "timing": {"wall_seconds": elapsed},
        "source": provenance,
        "failure": (
            {"failure_file": failure_path.name}
            if failure_path is not None
            else None
        ),
        "artifacts": {
            "history_npz": history_path.name,
            "checks_jsonl_gzip": checks_path.name,
            "history_bytes": _artifact_size(history_path),
            "checks_bytes": _artifact_size(checks_path),
        },
    }
    case_path = output / "case.json"
    _write_json(case_path, summary)
    summary["artifacts"]["case_json"] = case_path.name
    summary["artifacts"]["case_json_bytes"] = _artifact_size(case_path)
    return summary


def run_audit(
    output: str | Path,
    *,
    points: int = 9,
    device: str = "cpu",
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the resolved harmonic/Jacobian/stability audit."""
    _validate_device(device)
    output = _require_fresh(Path(output))
    audit = build_audit(points)
    output.mkdir(parents=True, exist_ok=False)
    arrays_path = output / "audit.npz"
    _save_npz(arrays_path, audit["arrays"])
    invocation = list(argv) if argv is not None else [str(Path(__file__))]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "AUDIT_COMPLETE",
        "scope": SCOPE,
        "controls": {
            "device": device,
            "points": points,
            "harmonic_max": audit["harmonic_max"],
            "epsilons": EPSILONS,
            "coordinate_order": COORDINATE_NAMES,
            "transverse_indexes": TRANSVERSE_INDEXES,
            "micro_indexes": Z_INDEXES,
            "main_micro_steps": DEFAULT_MAIN_MICRO_STEPS,
        },
        "derivative_wavenumbers": audit["derivative_wavenumbers"],
        "jacobian_epsilon_difference": audit["jacobian_epsilon_difference"],
        "transverse_relative_error": audit["transverse_relative_error"],
        "nonlinear_residual": audit["nonlinear_residual"],
        "full_eigenvalues": audit["full_eigenvalues"],
        "micro_eigenvalues": {
            harmonic: np.linalg.eigvals(matrix)
            for harmonic, matrix in audit["micro_jacobians"].items()
        },
        "independent_eigenvalues": audit["independent_eigenvalues"],
        "whole_map": audit["whole_map"],
        "zero_mode_measured_valid_max": float(
            np.max(
                np.abs(
                    audit["zero_mode_jacobian"][
                        :, audit["zero_mode_valid_columns"]
                    ]
                )
            )
        ),
        "zero_mode": audit["zero_mode"],
        "zero_mode_valid_column_mask": audit["zero_mode_valid_columns"],
        "stability": audit["stability"],
        "reference_semantics": audit["reference_semantics"],
        "provenance": _provenance(invocation, device),
        "artifacts": {
            "audit_npz": arrays_path.name,
            "audit_npz_bytes": _artifact_size(arrays_path),
        },
        "limitations": [
            "LIN evidence only; nonlinear trajectories are separate case runs",
            "full spectrum includes acoustic/pressure modes",
            "independent cold Hall matrix covers only the transverse 4x4 block",
            "critical imaginary-axis facts are not a nonlinear stability proof",
        ],
    }
    audit_path = output / "audit.json"
    _write_json(audit_path, report)
    report["artifacts"]["audit_json"] = audit_path.name
    report["artifacts"]["audit_json_bytes"] = _artifact_size(audit_path)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="resolved spectra and Jacobians")
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--device", default="cpu")
    audit.add_argument("--points", type=int, default=9)
    case = subparsers.add_parser("case", help="one production trajectory")
    case.add_argument("--output", type=Path, required=True)
    case.add_argument("--device", default="cpu")
    case.add_argument("--coupling", choices=VALID_COUPLINGS, required=True)
    case.add_argument(
        "--micro-per-period",
        type=int,
        choices=VALID_MICRO_PER_PERIOD,
        required=True,
    )
    case.add_argument("--micro-steps", type=int, required=True)
    case.add_argument("--cycles", type=int, default=2)
    case.add_argument("--seed", choices=VALID_SEEDS, required=True)
    case.add_argument("--points", type=int, default=9)
    case.add_argument("--progress-every", type=int, default=None)
    case.add_argument("--reference-case", type=Path)
    micro = subparsers.add_parser(
        "micro-control",
        help="prescribed-ion frozen electron/field RK4 control",
    )
    micro.add_argument("--output", type=Path, required=True)
    micro.add_argument("--device", default="cpu")
    micro.add_argument(
        "--micro-per-period",
        type=int,
        choices=VALID_MICRO_PER_PERIOD,
        required=True,
    )
    micro.add_argument("--steps", type=int)
    micro.add_argument("--points", type=int, default=9)
    micro.add_argument("--progress-every", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(arguments)
    invocation = [str(Path(__file__).resolve()), *arguments]
    try:
        if args.command == "audit":
            run_audit(
                args.output,
                points=args.points,
                device=args.device,
                argv=invocation,
            )
        elif args.command == "case":
            summary = run_case(
                args.output,
                device=args.device,
                coupling=args.coupling,
                micro_per_period=args.micro_per_period,
                micro_steps=args.micro_steps,
                cycles=args.cycles,
                seed=args.seed,
                points=args.points,
                progress_every=args.progress_every,
                reference_case=args.reference_case,
                argv=invocation,
            )
            if not summary["status"].startswith("COMPLETED"):
                return 1
        else:
            summary = run_frozen_micro_control(
                args.output,
                device=args.device,
                micro_per_period=args.micro_per_period,
                steps=args.steps,
                points=args.points,
                progress_every=args.progress_every,
                argv=invocation,
            )
            if not summary["status"].startswith("COMPLETED"):
                return 1
    except (ValueError, FileExistsError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
