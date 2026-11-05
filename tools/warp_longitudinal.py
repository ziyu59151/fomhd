#!/usr/bin/env python3
"""Unforced local finite-mass longitudinal time control, with a periodic adapter."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests/warp")]

from foundation_fixture import parameters  # noqa: E402
from massless_verification_fixture import _periodic_geometry, _periodic_operators  # noqa: E402
from mms_fixture import require_device, serialize_check_record, write_failure_arrays, write_failure_checks  # noqa: E402
from workflow_fixture import cumulative  # noqa: E402
from fomhd.backend import Backend  # noqa: E402
from fomhd.core import allocate_state  # noqa: E402
from fomhd.stage import initialize  # noqa: E402
from fomhd.time import advance  # noqa: E402
from fomhd.types import Context, Failure  # noqa: E402


SCOPE = (
    "Unforced small-amplitude parallel longitudinal mode of the finite-mass "
    "two-fluid/CGL equations, using a test-only periodic spectral adapter. "
    "Production stage, time transaction and hard diagnostics are unchanged. "
    "Not closed-wall spatial accuracy, a full stability region, Tokamak/ICE "
    "validation, a validation of physical CGL ordering, or a performance comparison."
)
POLYNOMIAL_TOLERANCE = 1e-5
MIN_POINTS = 9
MAX_POINTS = 65
MIN_CYCLES = 1.0
MAX_CYCLES = 16.0
MIN_STEPS = 4
MAX_STEPS = 16384
VALID_INTEGRATORS = ("midpoint", "rk4")
VALID_DT_GUARDS = ("error", "warn")
MINIMUM_FIT_CYCLES = 4.0
MINIMUM_FIT_SAMPLES_PER_CYCLE = 32.0

HARNESS_FILES = (
    "tools/warp_longitudinal.py",
    "tests/warp/test_longitudinal.py",
    "tests/warp/foundation_fixture.py",
    "tests/warp/workflow_fixture.py",
    "tests/warp/massless_verification_fixture.py",
    "tests/warp/mms_fixture.py",
    "docs/pseudocode/tools/warp_longitudinal.py.md",
    "docs/pseudocode/tests/warp/test_longitudinal.py.md",
)

FIT_UNITS = {
    "time_s": "s",
    "coefficient": "1",
    "angular_frequency_rad_s": "rad/s",
    "frequency_hz": "Hz",
    "growth_rate_per_s": "1/s",
    "phase_residual_rms_rad": "rad",
    "amplitude_log_residual_rms": "1",
    "phase_coherence": "1",
    "amplitude_coherence": "1",
    "complex_relative_rms_residual": "1",
    "complete_measured_cycles": "1",
    "samples_per_cycle": "1",
}


def polynomial(integrator, z):
    result = 1 + z + z*z/2
    if integrator == "rk4":
        result += z**3/6 + z**4/24
    return result


def _normalize_points(points: int) -> int:
    if (
        isinstance(points, bool)
        or not isinstance(points, int)
        or points < MIN_POINTS
        or points > MAX_POINTS
        or points % 2 == 0
    ):
        raise ValueError(
            f"points must be an odd integer in [{MIN_POINTS}, {MAX_POINTS}]"
        )
    return points


def _normalize_cycles(cycles: float) -> float:
    if isinstance(cycles, bool):
        raise ValueError(f"cycles must be finite in [{MIN_CYCLES:g}, {MAX_CYCLES:g}]")
    try:
        value = float(cycles)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"cycles must be finite in [{MIN_CYCLES:g}, {MAX_CYCLES:g}]"
        ) from error
    if not math.isfinite(value) or not MIN_CYCLES <= value <= MAX_CYCLES:
        raise ValueError(f"cycles must be finite in [{MIN_CYCLES:g}, {MAX_CYCLES:g}]")
    return value


def normalize_controls(
    *,
    points: int = MIN_POINTS,
    cycles: float = 2.0,
    steps: Sequence[int] = (4,),
    integrators: Sequence[str] = VALID_INTEGRATORS,
    dt_guard: str = "warn",
) -> dict[str, Any]:
    """Return validated scan controls before any runtime or output is created."""
    points = _normalize_points(points)
    cycles = _normalize_cycles(cycles)
    if isinstance(steps, (str, bytes)):
        raise ValueError("steps must be a nonempty sequence of unique integers")
    try:
        step_values = list(steps)
    except TypeError as error:
        raise ValueError(
            "steps must be a nonempty sequence of unique integers"
        ) from error
    if not step_values or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < MIN_STEPS
        or value > MAX_STEPS
        for value in step_values
    ):
        raise ValueError(
            f"steps must be unique integers in [{MIN_STEPS}, {MAX_STEPS}]"
        )
    if len(set(step_values)) != len(step_values):
        raise ValueError("steps must be unique")
    if isinstance(integrators, (str, bytes)):
        raise ValueError("integrators must be a nonempty sequence")
    try:
        integrator_values = list(integrators)
    except TypeError as error:
        raise ValueError("integrators must be a nonempty sequence") from error
    if (
        not integrator_values
        or any(
            not isinstance(value, str) or value not in VALID_INTEGRATORS
            for value in integrator_values
        )
        or len(set(integrator_values)) != len(integrator_values)
    ):
        raise ValueError(
            "integrators must be a nonempty unique subset of midpoint and rk4"
        )
    if dt_guard not in VALID_DT_GUARDS:
        raise ValueError("dt_guard must be error or warn")
    return {
        "points": points,
        "cycles": cycles,
        "steps": step_values,
        "integrators": integrator_values,
        "dt_guard": dt_guard,
    }


def _linear_fit(values: np.ndarray, times: np.ndarray) -> dict[str, Any]:
    centered = times - np.mean(times)
    total = float(np.dot(centered, centered))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("fit times do not span a positive interval")
    mean_value = float(np.mean(values))
    variations = values - mean_value
    slope = float(np.dot(centered, variations) / total)
    intercept_at_origin = mean_value - slope * float(np.mean(times))
    fitted = intercept_at_origin + slope * times
    residual = values - fitted
    residual_sum = float(np.dot(residual, residual))
    total_sum = float(np.dot(variations, variations))
    tolerance = (
        4096.0 * np.finfo(np.float64).eps * max(float(np.dot(values, values)), 1.0)
    )
    if total_sum <= tolerance:
        coherence = 1.0 if residual_sum <= tolerance else 0.0
    else:
        coherence = 1.0 - residual_sum / total_sum
    return {
        "slope": slope,
        "intercept_at_origin": intercept_at_origin,
        "residual_rms": math.sqrt(max(residual_sum / len(values), 0.0)),
        "coherence": min(1.0, max(0.0, float(coherence))),
        "fitted": fitted,
    }


def _fit_skeleton(sample_count: int) -> dict[str, Any]:
    return {
        "available": False,
        "reason": "",
        "coefficient": {"name": "normalized_electron_density", "unit": "1"},
        "sample_count": sample_count,
        "interval_count": max(sample_count - 1, 0),
        "complete_measured_cycles": None,
        "samples_per_cycle": None,
        "minimum_complete_cycles": MINIMUM_FIT_CYCLES,
        "minimum_samples_per_cycle": MINIMUM_FIT_SAMPLES_PER_CYCLE,
        "measured": {},
        "units": dict(FIT_UNITS),
    }


def fit_complex_mode(times, coefficients) -> dict[str, Any]:
    """Measure signed frequency and growth from observed samples only.

    ``coefficients`` may be the normalized electron-density history or the
    complete seven-component history produced by :func:`project`. No analytic
    frequency or production evolution is supplied.
    """
    try:
        raw_times = np.asarray(times)
        raw_modes = np.asarray(coefficients)
        sample_count = int(raw_times.size)
        time_values = np.asarray(times, dtype=np.float64).reshape(-1)
        mode_values = np.asarray(raw_modes, dtype=np.complex128)
        if mode_values.ndim == 2:
            if mode_values.shape[1] != 7:
                raise ValueError("complete mode history must have seven components")
            mode_values = mode_values[:, 3]
        else:
            mode_values = mode_values.reshape(-1)
    except (TypeError, ValueError) as error:
        result = _fit_skeleton(0)
        result["reason"] = f"fit histories must be numeric arrays: {error}"
        return result
    result = _fit_skeleton(sample_count)
    if mode_values.size != sample_count:
        result["reason"] = "time and coefficient histories differ in length"
        return result
    reasons: list[str] = []
    if sample_count < 2:
        reasons.append("fewer than two fit samples")
    if not np.all(np.isfinite(time_values)):
        reasons.append("time history must be finite and strictly increasing")
    elif sample_count >= 2 and not np.all(np.diff(time_values) > 0.0):
        reasons.append("time history must be finite and strictly increasing")
    elif sample_count >= 3:
        intervals = np.diff(time_values)
        if not np.allclose(intervals, intervals[0], rtol=1e-9, atol=0.):
            reasons.append("mode fitting requires uniform cadence without gaps")
    if not np.all(np.isfinite(mode_values)):
        reasons.append("complex coefficient samples must be finite")
    if np.any(np.abs(mode_values) == 0.0):
        reasons.append("complex coefficient samples must be nonzero")
    if reasons:
        result["reason"] = "; ".join(reasons)
        return result

    unwrapped_phase = np.unwrap(np.angle(mode_values))
    log_amplitude = np.log(np.abs(mode_values))
    measured_cycles = abs(float(unwrapped_phase[-1] - unwrapped_phase[0])) / (
        2.0 * math.pi
    )
    samples_per_cycle = (
        float(sample_count - 1) / measured_cycles if measured_cycles > 0.0 else None
    )
    phase_fit = _linear_fit(unwrapped_phase, time_values)
    amplitude_fit = _linear_fit(log_amplitude, time_values)
    reconstructed = np.exp(
        amplitude_fit["intercept_at_origin"]
        + amplitude_fit["slope"] * time_values
        + 1j
        * (phase_fit["intercept_at_origin"] + phase_fit["slope"] * time_values)
    )
    observed_norm = float(np.linalg.norm(mode_values))
    complex_residual = float(np.linalg.norm(mode_values - reconstructed))
    complex_relative = (
        complex_residual / observed_norm if observed_norm > 0.0 else math.inf
    )
    angular_frequency = float(phase_fit["slope"])
    growth_rate = float(amplitude_fit["slope"])
    result.update(
        complete_measured_cycles=measured_cycles,
        samples_per_cycle=samples_per_cycle,
        measured={
            "angular_frequency_rad_s": angular_frequency,
            "frequency_hz": angular_frequency / (2.0 * math.pi),
            "growth_rate_per_s": growth_rate,
            "phase_residual_rms_rad": float(phase_fit["residual_rms"]),
            "amplitude_log_residual_rms": float(amplitude_fit["residual_rms"]),
            "phase_coherence": float(phase_fit["coherence"]),
            "amplitude_coherence": float(amplitude_fit["coherence"]),
            "complex_relative_rms_residual": complex_relative,
        },
    )
    if measured_cycles < MINIMUM_FIT_CYCLES:
        reasons.append(
            f"fewer than {MINIMUM_FIT_CYCLES:g} measured cycles"
        )
    if (
        samples_per_cycle is None
        or samples_per_cycle
        < MINIMUM_FIT_SAMPLES_PER_CYCLE * (1.0 - 1.0e-12)
    ):
        reasons.append(
            f"fewer than {MINIMUM_FIT_SAMPLES_PER_CYCLE:g} samples per "
            "measured cycle"
        )
    result["available"] = not reasons
    result["reason"] = "; ".join(reasons)
    return result


def _integrator_prediction(integrator: str, omega: float, dt: float) -> dict[str, Any]:
    factor = polynomial(integrator, 1j * omega * dt)
    phase_per_step = math.atan2(factor.imag, factor.real)
    angular_frequency = phase_per_step / dt
    return {
        "integrator": integrator,
        "angular_frequency_rad_s": angular_frequency,
        "frequency_hz": angular_frequency / (2.0 * math.pi),
        "growth_rate_per_s": math.log(abs(factor)) / dt,
        "phase_per_step_rad": phase_per_step,
        "amplification_per_step": abs(factor),
        "units": {
            "angular_frequency_rad_s": "rad/s",
            "frequency_hz": "Hz",
            "growth_rate_per_s": "1/s",
            "phase_per_step_rad": "rad",
            "amplification_per_step": "1",
        },
    }


def _mode_fit_report(history, times, *, integrator, omega, dt) -> dict[str, Any]:
    observed = np.asarray(history, dtype=np.complex128)
    result = fit_complex_mode(times, observed[:, 3])
    result["measurement_source"] = (
        "complete accepted normalized electron-density history; no analytic "
        "frequency or production evolution is an input"
    )
    result["analytic"] = {
        "angular_frequency_rad_s": float(omega),
        "frequency_hz": float(omega) / (2.0 * math.pi),
        "growth_rate_per_s": 0.0,
        "phase_per_step_rad": float(omega) * dt,
        "units": {
            "angular_frequency_rad_s": "rad/s",
            "frequency_hz": "Hz",
            "growth_rate_per_s": "1/s",
            "phase_per_step_rad": "rad",
        },
    }
    result["rk4_predicted"] = _integrator_prediction("rk4", omega, dt)
    result["selected_integrator_predicted"] = _integrator_prediction(
        integrator, omega, dt
    )
    return result


def resolve_runtime_origin(
    package_file=None,
    *,
    loaded_modules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the executed package and its project metadata without fallback."""
    if package_file is None:
        package_file = importlib.import_module("fomhd").__file__
    source = Path(package_file).resolve()
    if source.name != "__init__.py" or source.parent.name != "fomhd":
        raise ValueError(f"imported fomhd package is malformed: {source}")
    package_root = source.parent
    if package_root.parent.name == "src":
        repository_root = package_root.parent.parent
        runtime_relative = Path("src/fomhd")
    else:
        repository_root = package_root.parent
        runtime_relative = Path("fomhd")
    metadata = {
        name: repository_root / name for name in ("pyproject.toml", "uv.lock")
    }
    missing = [str(path) for path, value in metadata.items() if not value.is_file()]
    if missing:
        raise ValueError(
            f"imported runtime has no matching project metadata: {missing}"
        )
    source_files = sorted(str(path.resolve()) for path in package_root.glob("*.py"))
    if not source_files:
        raise ValueError(f"imported fomhd package has no Python modules: {package_root}")
    if loaded_modules is None:
        loaded_modules = {
            name: module.__file__
            for name, module in sys.modules.items()
            if (name == "fomhd" or name.startswith("fomhd."))
            and getattr(module, "__file__", None) is not None
        }
    outside = []
    for name, origin in sorted(loaded_modules.items()):
        path = Path(origin).resolve()
        if not path.is_relative_to(package_root):
            outside.append(f"{name}={path}")
    if outside:
        raise ValueError(
            "loaded fomhd modules resolve outside the imported package: "
            + ", ".join(outside)
        )
    return {
        "root": str(repository_root.resolve()),
        "repository_root": str(repository_root.resolve()),
        "package_root": str(package_root),
        "runtime_relative": str(runtime_relative),
        "metadata": {name: str(path.resolve()) for name, path in metadata.items()},
        "source_files": source_files,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_source_snapshot(
    snapshot_root,
    *,
    package_file=None,
    loaded_modules: Mapping[str, Any] | None = None,
    harness_root=None,
    harness_files: Sequence[Path | str] | None = None,
    configuration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Archive only the resolved runtime and declared harness under SHA-256."""
    origin = resolve_runtime_origin(package_file, loaded_modules=loaded_modules)
    repository_root = Path(origin["repository_root"])
    package_root = Path(origin["package_root"])
    runtime_relative = Path(origin["runtime_relative"])
    destination_root = Path(snapshot_root)
    if harness_root is None:
        harness_root = ROOT
    harness_root = Path(harness_root).resolve()
    if harness_files is None:
        harness_files = [harness_root / relative for relative in HARNESS_FILES]
    entries: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}

    def archive(source_path: Path, relative: Path, role: str) -> None:
        source_path = source_path.resolve()
        relative = Path(relative)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        digest = _sha256(destination)
        source_hashes[relative.as_posix()] = digest
        entries.append(
            {
                "role": role,
                "path": relative.as_posix(),
                "source_path": str(source_path),
                "size_bytes": destination.stat().st_size,
                "sha256": digest,
            }
        )

    for source_name in origin["source_files"]:
        source_path = Path(source_name)
        archive(
            source_path,
            runtime_relative / source_path.relative_to(package_root),
            "runtime_module",
        )
    for name, source_name in origin["metadata"].items():
        source_path = Path(source_name)
        archive(
            source_path,
            Path(name),
            "dependency_lock" if name == "uv.lock" else "project_metadata",
        )
    for source_name in harness_files:
        source_path = Path(source_name)
        if not source_path.is_absolute():
            source_path = harness_root / source_path
        source_path = source_path.resolve()
        try:
            relative = source_path.relative_to(harness_root)
        except ValueError:
            relative = Path(source_path.name)
        archive(source_path, relative, "verification_harness")
    manifest = {
        "schema": "a0-longitudinal-source-snapshot-v1",
        "runtime_root": str(repository_root),
        "repository_root": str(repository_root),
        "package_root": str(package_root),
        "runtime_relative": runtime_relative.as_posix(),
        "harness_root": str(harness_root),
        "configuration": dict(configuration or {}),
        "file_count": len(entries),
        "source_hashes": source_hashes,
        "files": entries,
        "scope": (
            "actual imported fomhd package, matching root metadata, and declared "
            "longitudinal harness only"
        ),
    }
    manifest_path = destination_root / "SOURCE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def build_context(device="cpu", *, points=MIN_POINTS):
    """Return device-owned operators and an independent SI continuum mode."""
    points = _normalize_points(points)
    backend = Backend(require_device(device))
    p = parameters(0)
    p.electron_inertia = 1
    p.Nphi = 1
    p.inner_distance = 0.0
    k, density, pressure, magnetic = 1.0, 1e12, 1e-9, 1e-5
    geometry = _periodic_geometry(points, 2*math.pi/k)
    op = _periodic_operators(geometry, points, k, backend)
    xp = backend.xp
    gauge = xp.full((points, points), k*k/points, dtype=xp.complex128)
    op.Lphi = [op.G[0].T.conj() @ op.G[0] + gauge]
    op.selected_constraint_rows = backend.independent_rows(op.Bc_full, p.rank_rtol)
    op.Bc = backend.asarray(op.Bc_full[op.selected_constraint_rows, :])
    op.topology_flux_row[:] = 0
    op.topology_flux_row[0, 2*points:] = 1/points
    context = Context(p, geometry, op, backend, 1)

    mi, me = [s.mass for s in p.species]
    q = p.species[0].charge
    wi2 = density*q*q/(p.epsilon0*mi)
    we2 = density*q*q/(p.epsilon0*me)
    a = wi2 + k*k*(5/3)*pressure/(mi*density)
    b = we2 + k*k*3*pressure/(me*density)
    omega2 = (a+b+math.sqrt((a-b)**2+4*wi2*we2))/2
    omega = math.sqrt(omega2)
    speed = math.sqrt(we2)/k
    ion_density = wi2/(a-omega2)
    vector = np.array([ion_density, -omega*ion_density/(k*speed),
                       (5/3)*ion_density, 1., -omega/(k*speed), 1., 3.], complex)
    wave = backend.asarray(np.exp(2j*math.pi*np.arange(points)/points))
    defects = {
        "plane_wave": backend.scalar(xp.max(xp.abs(op.DZ @ wave-1j*k*wave)))/k,
        "constant_derivative": backend.scalar(xp.max(xp.abs(op.DZ.sum(axis=1))))/k,
        "CG": backend.scalar(xp.max(xp.abs(op.C[0] @ op.G[0])))/(k*k),
        "DC": backend.scalar(xp.max(xp.abs(op.DB[0] @ op.C[0])))/(k*k),
        "mean_gauge": backend.scalar(xp.max(xp.abs(op.Lphi[0].sum(axis=1)-k*k)))/(k*k),
        "derivative_skew": backend.scalar(xp.max(xp.abs(op.DZ+op.DZ.T.conj())))/k,
        "current_adjoint": backend.scalar(xp.max(xp.abs(op.DJ[0]+op.G[0].T.conj())))/k,
        "curl_adjoint": backend.scalar(xp.max(xp.abs(op.Cdag[0]-op.C[0].T.conj())))/k,
    }
    maximum = max(float(value) for value in defects.values())
    if maximum > 1e-12 or len(op.selected_constraint_rows) != points-1:
        raise Failure("Fixture", "periodic finite adapter identities failed")
    return context, dict(points=points, density=density, pressure=pressure, magnetic=magnetic,
                         k=k, speed=speed, omega=omega, vector=vector, wave=wave,
                         adapter_defects=defects, adapter_max_defect=maximum)


def project(state, context, mode, *, derivative=False):
    """Reduce seven mode coefficients on-device; transfer only seven scalars."""
    n = context.op.Nc
    mi, me = [s.mass for s in context.p.species]
    density, pressure, speed = mode["density"], mode["pressure"], mode["speed"]
    fields = [state.fluid[0].n/density,
              state.fluid[0].mom[2*n:]/(mi*density*speed),
              state.fluid[0].perp/pressure, state.fluid[1].n/density,
              state.fluid[1].mom[2*n:]/(me*density*speed),
              state.fluid[1].perp/pressure, state.fluid[1].parallel/pressure]
    xp = context.compute.xp
    coefficients = []
    for index, field in enumerate(fields):
        if not derivative and index in (0, 2, 3, 5, 6):
            field = field-1
        coefficients.append(context.compute.scalar(2*xp.mean(field*mode["wave"].conj())))
    return np.asarray(coefficients, dtype=np.complex128)


def run_case(
    integrator,
    steps,
    *,
    points=MIN_POINTS,
    cycles=2,
    amplitude=1e-7,
    device="cpu",
    dt_guard="warn",
):
    """Run one bounded local control and retain complete real diagnostics."""
    controls = normalize_controls(
        points=points,
        cycles=cycles,
        steps=[steps],
        integrators=[integrator],
        dt_guard=dt_guard,
    )
    steps = controls["steps"][0]
    cycles = controls["cycles"]
    if isinstance(amplitude, bool):
        raise ValueError("amplitude must be in (0, 1e-6]")
    try:
        amplitude = float(amplitude)
    except (TypeError, ValueError) as error:
        raise ValueError("amplitude must be in (0, 1e-6]") from error
    if not math.isfinite(amplitude) or not 0 < amplitude <= 1e-6:
        raise ValueError("amplitude must be in (0, 1e-6]")
    context, mode = build_context(device, points=controls["points"])
    p, backend = context.p, context.compute
    p.integrator, p.nsteps, p.dt_guard = integrator, steps, controls["dt_guard"]
    horizon = cycles*2*math.pi/mode["omega"]
    p.dt = horizon/steps
    factor = polynomial(integrator, 1j*mode["omega"]*p.dt)
    if amplitude*abs(factor)**steps > 1e-3:
        raise ValueError("predicted amplification leaves the bounded linear control")
    state = allocate_state(context)
    state.generation = 1
    n = context.op.Nc
    xp = backend.xp
    for species, offset in ((0, 0), (1, 3)):
        v = mode["vector"]
        wave = mode["wave"]
        state.fluid[species].n[:] = mode["density"]*(1+amplitude*xp.real(v[offset]*wave))
        state.fluid[species].mom[2*n:] = (
            p.species[species].mass*mode["density"]*mode["speed"]
            *amplitude*xp.real(v[offset+1]*wave))
        state.fluid[species].perp[:] = mode["pressure"]*(1+amplitude*xp.real(v[offset+2]*wave))
        if species:
            state.fluid[species].parallel[:] = mode["pressure"]*(1+amplitude*xp.real(v[6]*wave))
    state.B[2*n:] = mode["magnetic"]
    neutral_defect = float(abs(backend.scalar(xp.mean(state.fluid[0].n-state.fluid[1].n)))/mode["density"])
    if neutral_defect > 32*np.finfo(float).eps:
        raise Failure("Fixture", "periodic initial total charge is nonzero")
    ledger, captured = cumulative(), {}
    committed = initialize(state, context, ledger, False, captured)
    initial = project(committed.state, context, mode)
    prescribed = amplitude*mode["vector"]
    initial_error = np.linalg.norm(initial-prescribed)/np.linalg.norm(prescribed)
    rhs = project(captured["rhs"], context, mode, derivative=True)
    rhs_reference = 1j*mode["omega"]*prescribed
    rhs_error = np.linalg.norm(rhs-rhs_reference)/np.linalg.norm(rhs_reference)
    records = list(captured["checks"])
    initial_hard = sum(record.hard for record in records)
    history, times, reports = [initial], [committed.state.time], []
    failure = None
    started = time.monotonic()
    for _ in range(steps):
        try:
            result = advance(committed, ledger, context, None)
        except Failure as error:
            records.extend(error.checks)
            rejected_report = getattr(error, "time_step_report", None)
            if rejected_report is not None:
                reports.append(rejected_report)
            failure = error
            break
        records.extend(result.checks)
        reports.append(result.time_step_report)
        history.append(project(committed.state, context, mode))
        times.append(committed.state.time)
    backend.synchronize()
    actual = history[-1]
    accepted = committed.state.step
    expected = prescribed*factor**accepted
    continuum = prescribed*np.exp(1j*mode["omega"]*committed.state.time)
    polynomial_error = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    hard = [record for record in records if record.hard]
    hard_pass = bool(hard) and all(record.available and record.passed for record in hard)
    projection = np.vdot(prescribed, actual)/np.vdot(prescribed, prescribed).real
    history_array = np.asarray(history)
    times_array = np.asarray(times)
    mode_fit = _mode_fit_report(
        history_array,
        times_array,
        integrator=integrator,
        omega=mode["omega"],
        dt=p.dt,
    )
    passed = (failure is None and accepted == steps and hard_pass
              and initial_error < 1e-7 and rhs_error < POLYNOMIAL_TOLERANCE
              and polynomial_error < POLYNOMIAL_TOLERANCE)
    summary = dict(integrator=integrator, device=device, points=controls["points"],
                   cycles=cycles, dt_guard=controls["dt_guard"],
                   steps=steps, accepted_steps=accepted,
                   dt_s=p.dt, target_time_s=horizon, accepted_time_s=committed.state.time,
                   omega_rad_s=mode["omega"], frequency_Hz=mode["omega"]/(2*math.pi),
                   relative_amplitude=amplitude, adapter_max_defect=mode["adapter_max_defect"],
                   initial_neutrality_relative_defect=neutral_defect,
                   initial_mode_relative_error=float(initial_error), rhs_mode_relative_error=float(rhs_error),
                   polynomial_relative_error=polynomial_error, polynomial_tolerance=POLYNOMIAL_TOLERANCE,
                   continuum_relative_error=float(np.linalg.norm(actual-continuum)/np.linalg.norm(continuum)),
                   amplitude_ratio=float(abs(projection)), polynomial_amplitude_ratio=float(abs(factor)**accepted),
                   wrapped_continuum_phase_error_rad=float(np.angle(projection*np.exp(-1j*mode["omega"]*committed.state.time))),
                   initial_hard_records=initial_hard, hard_records=len(hard), hard_pass=hard_pass,
                   dt_warnings=sum(not report["accepted"] for report in reports),
                   mode_fit=mode_fit,
                   evolution_wall_seconds=time.monotonic()-started, passed=bool(passed),
                   failure=None if failure is None else dict(code=failure.code, message=str(failure)))
    return dict(summary=summary, history=history_array, times=times_array,
                diagnostics=records, time_step_reports=reports, failure=failure)


def run_scan(
    output,
    *,
    device="cpu",
    points=MIN_POINTS,
    steps=(4, 8, 32, 64, 128),
    cycles=2,
    integrators=VALID_INTEGRATORS,
    dt_guard="warn",
):
    """Write a fresh, fully attributed scan including half-amplitude controls."""
    controls = normalize_controls(
        points=points,
        cycles=cycles,
        steps=steps,
        integrators=integrators,
        dt_guard=dt_guard,
    )
    require_device(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source_snapshot = archive_source_snapshot(
        output/"source",
        harness_root=ROOT,
        harness_files=[ROOT / relative for relative in HARNESS_FILES],
        configuration=controls,
    )
    provenance = dict(
        source_hashes=source_snapshot["source_hashes"],
        source_manifest=source_snapshot["manifest_path"],
        configuration=controls,
        points=controls["points"],
        cycles=controls["cycles"],
        integrators=controls["integrators"],
        dt_guard=controls["dt_guard"],
        repository_root=source_snapshot["repository_root"],
        package_root=source_snapshot["package_root"],
        runtime_relative=source_snapshot["runtime_relative"],
        harness_root=source_snapshot["harness_root"],
        device=device,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        command=sys.argv,
    )
    (output/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    (output/"SCOPE.md").write_text(SCOPE+"\n")
    results = []
    for integrator in controls["integrators"]:
        for count, amplitude in (
            [(count, 1e-7) for count in controls["steps"]]
            + [(min(controls["steps"]), 5e-8)]
        ):
            name = f"{integrator}-n{count}-a{amplitude:.0e}"
            try:
                result = run_case(
                    integrator,
                    count,
                    points=controls["points"],
                    cycles=controls["cycles"],
                    amplitude=amplitude,
                    device=device,
                    dt_guard=controls["dt_guard"],
                )
            except (Failure, ValueError) as error:
                write_failure_arrays(output, error, stem=name+"-initial-failure")
                write_failure_checks(output, error, stem=name+"-initial-failure")
                failure = dict(code=getattr(error, "code", type(error).__name__), message=str(error))
                (output/(name+"-failure.json")).write_text(json.dumps(failure, indent=2)+"\n")
                (output/"report.json").write_text(json.dumps(dict(status="failed", scope=SCOPE,
                    configuration=controls, cases=results, failed_case=name, failure=failure,
                    passed=False), indent=2)+"\n")
                raise
            summary = result["summary"]
            summary["case_id"] = name
            results.append(summary)
            np.savez_compressed(output/(name+".npz"), time_s=result["times"], mode=result["history"])
            with gzip.open(output/(name+"-checks.json.gz"), "wt", encoding="utf-8") as stream:
                json.dump([serialize_check_record(record) for record in result["diagnostics"]], stream, allow_nan=False)
            (output/(name+"-dt.json")).write_text(json.dumps(result["time_step_reports"], indent=2, allow_nan=False)+"\n")
            if result["failure"] is not None:
                write_failure_arrays(output, result["failure"], stem=name+"-failure")
            report = dict(
                scope=SCOPE,
                status="running",
                configuration=controls,
                points=controls["points"],
                cycles=controls["cycles"],
                integrators=controls["integrators"],
                dt_guard=controls["dt_guard"],
                cases=results,
            )
            (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            print(json.dumps(summary), flush=True)
    sensitivity = []
    for integrator in controls["integrators"]:
        base, half = [
            row
            for row in results
            if row["integrator"] == integrator
            and row["steps"] == min(controls["steps"])
        ]
        error = abs(base["amplitude_ratio"]-half["amplitude_ratio"])/base["amplitude_ratio"]
        sensitivity.append(dict(integrator=integrator, relative_amplitude_change=error, passed=error < 1e-5))
    report.update(status="completed", configuration=controls,
                  points=controls["points"], cycles=controls["cycles"],
                  integrators=controls["integrators"], dt_guard=controls["dt_guard"],
                  amplitude_sensitivity=sensitivity,
                  passed=all(row["passed"] for row in results) and all(row["passed"] for row in sensitivity))
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--points", type=int, default=MIN_POINTS)
    parser.add_argument("--cycles", type=float, default=8.)
    parser.add_argument("--steps", default="1024,2048")
    parser.add_argument("--integrators", default="rk4")
    parser.add_argument("--dt-guard", choices=VALID_DT_GUARDS, default="error")
    args = parser.parse_args()
    try:
        steps = [int(value) for value in args.steps.split(",")]
        integrators = tuple(
            value.strip() for value in args.integrators.split(",") if value.strip()
        )
        controls = normalize_controls(
            points=args.points,
            cycles=args.cycles,
            steps=steps,
            integrators=integrators,
            dt_guard=args.dt_guard,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    report = run_scan(
        args.output,
        device=args.device,
        points=controls["points"],
        steps=controls["steps"],
        cycles=controls["cycles"],
        integrators=tuple(controls["integrators"]),
        dt_guard=controls["dt_guard"],
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
