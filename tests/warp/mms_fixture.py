"""Discrete coupled temporal MMS and smooth spatial verification fixtures."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fomhd import diagnostics, stability
from fomhd.backend import Backend
from fomhd.core import allocate_state, time_at_midpoint, time_at_step
from fomhd.geometry import build_rectangle
from fomhd.operators import OperatorOptions, build_operators, build_spatial_operators
from fomhd.stage import closed_reference, evaluate_stage, initialize
from fomhd.time import advance
from fomhd.types import Context, Failure, StageTag

from foundation_fixture import parameters


DURATION_S = 1.0e-8
SPEED_MPS = 0.1
MIDPOINT_LEVELS = (8, 16, 32)
RK4_LEVELS = (2, 4, 8)
SPATIAL_LEVELS = (8, 16)
SPATIAL_MODE = 2


def require_device(device="cpu", *, environ=None):
    """Normalize the selected backend and enforce the Slurm-only CUDA policy."""
    environment = os.environ if environ is None else environ
    selected = str(device).lower()
    if selected == "cpu":
        return "cpu"
    if selected == "cuda":
        selected = "cuda:0"
    if selected.startswith("cuda:"):
        index = selected[5:]
        if not index.isdigit():
            raise Failure("Backend", f"unsupported device {device!r}")
        if not environment.get("SLURM_JOB_ID"):
            raise Failure(
                "Backend",
                "CUDA execution requires an existing Slurm allocation "
                "(SLURM_JOB_ID is unset)",
            )
        return f"cuda:{int(index)}"
    raise Failure("Backend", f"unsupported device {device!r}")


def make_context(device="cpu"):
    """Construct the native 4x4, K=1, Nphi=9 coupled verification context."""
    device = require_device(device)
    p = parameters(K=1)
    p.Nphi = 9
    p.dt_guard = "warn"
    backend = Backend(device)
    geometry = build_rectangle(4, 4, 1.0, 0.0, 0.2, 0.2)
    operators = build_operators(geometry, p.K, backend=backend)
    return Context(
        p=p,
        geometry=geometry,
        op=operators,
        compute=backend,
        N=p.Nphi,
    )


def stage_tag(context, stage, *, step=0, generation=1, trial=0):
    """Return the scheduler-consistent tag expected by a trajectory call."""
    if stage in ("base", "initial", "restart"):
        tag_step = step
        tag_time = time_at_step(context.p, step)
    elif stage in ("midpoint", "midpoint_2"):
        tag_step = step
        tag_time = time_at_midpoint(context.p, step)
    elif stage in ("station", "endpoint"):
        tag_step = step + 1
        tag_time = time_at_step(context.p, step + 1)
    else:
        raise ValueError(f"unknown stage {stage!r}")
    return StageTag(
        step=tag_step,
        generation=generation,
        trial=trial,
        spectral_context=getattr(context, "spectral_id", 0),
        time=tag_time,
        stage=stage,
    )


def clone_state(state, context):
    """Copy a state without changing backend ownership or dtype."""
    out = allocate_state(context)
    for species in range(2):
        for name in ("n", "mom", "perp", "parallel"):
            target = getattr(out.fluid[species], name)
            source = getattr(state.fluid[species], name)
            if target.size:
                target[...] = source
    out.B[...] = state.B
    out.step = state.step
    out.generation = state.generation
    out.time = state.time
    return out


def state_difference(left, right, context):
    """Return an independently owned rate state ``left - right``."""
    out = allocate_state(context)
    xp = context.compute.xp
    for species in range(2):
        for name in ("n", "mom", "perp", "parallel"):
            target = getattr(out.fluid[species], name)
            if target.size:
                target[...] = (
                    getattr(left.fluid[species], name)
                    - getattr(right.fluid[species], name)
                )
    out.B[...] = left.B - right.B
    out.time = left.time
    out.step = left.step
    out.generation = left.generation
    return out


class NativeMMSTrajectory:
    """Analytic-in-time coupled trajectory matching ``coupled_verification``."""

    def __init__(self, context, *, duration=DURATION_S, speed=SPEED_MPS):
        self.context = context
        self.duration = float(duration)
        self.speed = float(speed)
        if not math.isfinite(self.duration) or self.duration <= 0.0:
            raise Failure("Fixture", "MMS duration must be positive and finite")
        if not math.isfinite(self.speed) or self.speed <= 0.0:
            raise Failure("Fixture", "MMS velocity must be positive and finite")
        ref = context.p.ref
        self.equilibrium = closed_reference(
            context,
            ref.ni,
            ref.pressure,
            ref.pressure
            * context.p.species[1].mass
            / context.p.species[0].mass,
            ref.magnetic,
        )
        xp = context.compute.xp
        cell_r = xp.asarray(context.geometry.cells.R)
        cell_z = xp.asarray(context.geometry.cells.Z)
        self._envelope = 0.6 + 0.2 * xp.cos(cell_r) * xp.cos(cell_z)

    def at(self, tag, derivative=False):
        """Evaluate the state or its exact analytic time derivative at ``tag``."""
        context = self.context
        if derivative:
            state = allocate_state(context)
        else:
            state = clone_state(self.equilibrium, context)
        state.step = tag.step
        state.generation = tag.generation
        state.time = tag.time
        scaled_time = tag.time / self.duration

        def wave(frequency):
            if derivative:
                return frequency * math.cos(frequency * scaled_time) / self.duration
            return math.sin(frequency * scaled_time)

        nc = int(context.op.Nc)
        ref = context.p.ref
        for species in range(2):
            mass = context.p.species[species].mass
            pressure = ref.pressure * mass / context.p.species[0].mass
            for cell in range(nc):
                envelope = self._envelope[cell]
                if not derivative:
                    state.fluid[species].perp[context.p.K * nc + cell] = pressure
                    if species:
                        state.fluid[species].parallel[
                            context.p.K * nc + cell
                        ] = pressure
                for mode in (-1, 1):
                    density_index = (context.p.K + mode) * nc + cell
                    momentum_index = (
                        (context.p.K + mode) * 3 * nc + nc + cell
                    )
                    state.fluid[species].n[density_index] = (
                        0.01
                        * (0.9999 if species else 1.0)
                        * ref.ni
                        * envelope
                        * wave(1.0)
                    )
                    state.fluid[species].mom[momentum_index] = (
                        0.5
                        * (1.3 if species else 1.0)
                        * mass
                        * ref.ni
                        * self.speed
                        * envelope
                        * wave(1.2)
                    )
                    state.fluid[species].perp[density_index] = (
                        0.015 * pressure * envelope * wave(0.7)
                    )
                    if species:
                        state.fluid[species].parallel[density_index] = (
                            0.011 * pressure * envelope * wave(0.7)
                        )
        magnetic_factor = (0.0 if derivative else 1.0) + 0.01 * wave(0.8)
        state.B[...] = self.equilibrium.B * magnetic_factor
        return state


class ManufacturedSource:
    """Test-only source ``U_dot(t) - R(U_reference(t))`` at exact references."""

    def __init__(self, trajectory, *, omit_ion_pressure=False):
        self.trajectory = trajectory
        self.context = trajectory.context
        self.omit_ion_pressure = bool(omit_ion_pressure)
        self.calls = []

    def __call__(self, actual, tag, context):
        reference = self.trajectory.at(tag)
        physical = evaluate_stage(reference, tag, context, False)
        exact_derivative = self.trajectory.at(tag, derivative=True)
        payload = state_difference(
            exact_derivative,
            physical.rhs,
            context,
        )
        if self.omit_ion_pressure:
            payload.fluid[0].perp.fill(0.0)
        self.calls.append(
            SimpleNamespace(
                tag=tag,
                actual=actual,
                reference=reference,
                physical=physical,
                payload=payload,
                source_id=physical.source_id,
            )
        )
        return payload

    def values(self, payload):
        """Return one maximum reduction for every required source component."""
        xp = self.context.compute.xp
        values = []
        for species in range(2):
            for name in ("n", "mom", "perp", "parallel"):
                array = getattr(payload.fluid[species], name)
                if array.size:
                    values.append(float(xp.max(xp.abs(array))))
        values.append(float(xp.max(xp.abs(payload.B))))
        return values


def _copy_cumulative():
    names = (
        "electric_node",
        "electric_edge",
        "pressure_kinetic",
        "pressure_internal",
        "magnetic",
        "wall",
        "convection",
        "perp_rate",
        "parallel_rate",
        "spectral_kinetic",
        "spectral_perp",
        "spectral_parallel",
    )
    return SimpleNamespace(
        baseline=None,
        integrals=[
            SimpleNamespace(**dict.fromkeys(names, 0.0)) for _ in range(2)
        ],
        boundary_integrals=SimpleNamespace(
            number=[0.0, 0.0],
            kinetic=[0.0, 0.0],
            internal_perp=[0.0, 0.0],
            internal_parallel=[0.0, 0.0],
            field_energy=0.0,
            total_energy=0.0,
        ),
        accepted_intervals=0,
        last_trial=0,
        last_step=0,
        last_time=0.0,
    )


def prepare_output_directory(path):
    """Create a new-or-empty run directory without overwriting prior evidence."""
    output = Path(path)
    if output.exists():
        if not output.is_dir():
            raise Failure("Output", "CLI output path exists and is not a directory")
        if any(output.iterdir()):
            raise Failure("Output", "CLI output directory must be new or empty")
    else:
        output.mkdir(parents=True)
    return output


def write_json(path, value):
    """Persist strict JSON with an exclusive file create."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    return destination


def _float_wire(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None:
        return None
    return str(value)


def _float_hex(value):
    if isinstance(value, np.generic):
        value = value.item()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0.0 else "-inf"
    return number.hex()


def serialize_check_record(check):
    """Convert one real production diagnostic record to bounded JSONL."""
    tag = check.tag
    record = {
        "name": str(check.name),
        "population": str(check.population),
        "region": str(check.region),
        "component": str(check.component),
        "producer": str(check.producer),
        "norm": str(check.norm),
        "unit": str(check.unit),
        "value_unit": str(check.value_unit),
        "stage": str(tag.stage),
        "step": int(tag.step),
        "generation": int(tag.generation),
        "trial": int(tag.trial),
        "spectral_context": int(tag.spectral_context),
        "time_s": float(tag.time),
        "time_hex": float(tag.time).hex(),
        "raw": _float_wire(check.raw),
        "raw_hex": _float_hex(check.raw),
        "reference": _float_wire(check.reference),
        "reference_hex": _float_hex(check.reference),
        "threshold": _float_wire(check.threshold),
        "threshold_hex": _float_hex(check.threshold),
        "available": bool(check.available),
        "hard": bool(check.hard),
        "passed": bool(check.passed),
        "reason": str(check.reason),
        "selected": int(getattr(check, "selected", 0)),
        "worst": int(getattr(check, "worst", 0)),
    }
    return record


def audit_check_records(records):
    """Detect hard failures, soft controls, and nonfinite raw diagnostics."""
    hard_failed = 0
    soft_failed = 0
    nonfinite = 0
    unavailable = 0
    hard_unavailable = 0
    for record in records:
        if not record["available"]:
            unavailable += 1
            if record["hard"]:
                hard_unavailable += 1
        if record.get("raw") is None or record.get("reference") is None:
            nonfinite += 1
        if not record["passed"]:
            if record["hard"]:
                hard_failed += 1
            else:
                soft_failed += 1
    failed = hard_failed + soft_failed
    return {
        "total": len(records),
        "failed": failed,
        "hard_failed": hard_failed,
        "soft_failed": soft_failed,
        "nonfinite": nonfinite,
        "unavailable": unavailable,
        "hard_unavailable": hard_unavailable,
        "hard_integrity_passed": (
            hard_failed == 0
            and hard_unavailable == 0
            and nonfinite == 0
        ),
        "all_records_passed": failed == 0 and unavailable == 0,
    }


def write_check_jsonl(path, checks):
    """Persist every tagged production check without reducing its raw values."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = [serialize_check_record(check) for check in checks]
    with destination.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return records, destination


def _host_array(value):
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    module = type(value).__module__
    if module.startswith("cupy") and hasattr(value, "get"):
        return np.asarray(value.get())
    return np.asarray(value)


def write_failure_arrays(output_dir, error, stem="failure"):
    """Preserve Failure.partial_arrays in NPZ, including NaN and infinity."""
    partial = getattr(error, "partial_arrays", {}) or {}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{stem}.npz"
    if not partial:
        return {
            "path": None,
            "sha256": None,
            "contains_nonfinite": False,
            "arrays": {},
            "scope": "Failure supplied no partial arrays",
        }
    arrays = {str(name): _host_array(value) for name, value in partial.items()}
    np.savez_compressed(path, **arrays)
    descriptors = {}
    contains_nonfinite = False
    for name, value in arrays.items():
        nonfinite = bool(
            np.issubdtype(value.dtype, np.inexact)
            and not np.all(np.isfinite(value))
        )
        contains_nonfinite = contains_nonfinite or nonfinite
        descriptors[name] = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "size": int(value.size),
            "contains_nonfinite": nonfinite,
        }
    return {
        "path": str(path.relative_to(output)),
        "sha256": sha256_file(path),
        "contains_nonfinite": contains_nonfinite,
        "arrays": descriptors,
        "scope": "raw Failure.partial_arrays; nonfinite values preserved",
    }


def write_failure_checks(output_dir, error, stem="failure"):
    """Preserve tagged checks attached directly to a Failure."""
    checks = list(getattr(error, "checks", []) or [])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not checks:
        return {
            "path": None,
            "sha256": None,
            "count": 0,
            "audit": audit_check_records([]),
        }
    records, path = write_check_jsonl(output / f"{stem}.jsonl", checks)
    return {
        "path": str(path.relative_to(output)),
        "sha256": sha256_file(path),
        "count": len(records),
        "audit": audit_check_records(records),
    }


def _normalized_error(context, actual, exact, scale):
    xp = context.compute.xp
    difference = actual - exact
    norm = xp.sqrt(xp.sum(xp.abs(difference) ** 2))
    denominator = float(scale) * math.sqrt(float(actual.size))
    value = float(norm) / denominator
    if not math.isfinite(value):
        raise Failure("MMS", "nonfinite normalized component error")
    return value


def normalized_errors(actual, exact, context):
    """Use the native component-wise scales and combined Euclidean norm."""
    ref = context.p.ref
    out = {}
    for species, suffix in ((0, "i"), (1, "e")):
        fluid_actual = actual.fluid[species]
        fluid_exact = exact.fluid[species]
        out[f"n_{suffix}"] = _normalized_error(
            context,
            fluid_actual.n,
            fluid_exact.n,
            ref.ni * 0.02,
        )
        out[f"mom_{suffix}"] = _normalized_error(
            context,
            fluid_actual.mom,
            fluid_exact.mom,
            context.p.species[species].mass * ref.ni * 0.1,
        )
        pressure_scale = (
            ref.pressure
            * context.p.species[species].mass
            / context.p.species[0].mass
            * 0.03
        )
        out[f"perp_{suffix}"] = _normalized_error(
            context,
            fluid_actual.perp,
            fluid_exact.perp,
            pressure_scale,
        )
        if species:
            out["parallel_e"] = _normalized_error(
                context,
                fluid_actual.parallel,
                fluid_exact.parallel,
                pressure_scale,
            )
    out["B"] = _normalized_error(
        context,
        actual.B,
        exact.B,
        ref.magnetic * 0.01,
    )
    out["combined"] = math.hypot(*(out.values()))
    return out


def _checks_are_clean(checks):
    for check in checks:
        finite = all(
            math.isfinite(float(value))
            for value in (check.raw, check.reference, check.threshold)
        )
        if check.available and not finite:
            return False
        if check.hard and (not check.available or not check.passed):
            return False
    return True


def _scalar(value):
    if isinstance(value, (int, float)):
        return float(value)
    return float(value.item())


def _rate_record(rate, context):
    totals = rate.totals
    xp = context.compute.xp
    return {
        "number_i_s^-1": _scalar(totals.number[0]),
        "number_e_s^-1": _scalar(totals.number[1]),
        "kinetic_i_W": _scalar(totals.kinetic[0]),
        "kinetic_e_W": _scalar(totals.kinetic[1]),
        "ion_internal_W": _scalar(totals.ion_internal),
        "electron_perp_W": _scalar(totals.electron_perp),
        "electron_parallel_W": _scalar(totals.electron_parallel),
        "magnetic_W": _scalar(totals.magnetic),
        "longitudinal_W": _scalar(totals.longitudinal),
        "energy_W": _scalar(totals.energy),
        "topology_Wb_s^-1": _scalar(totals.topology),
        "physical_flux_Wb_s^-1": _scalar(totals.physical_flux),
        "momentum_x_N": _scalar(rate.momentum.value[0]),
        "momentum_y_N": _scalar(rate.momentum.value[1]),
        "momentum_z_N": _scalar(rate.momentum.value[2]),
        "angular_momentum_Nm": _scalar(rate.momentum.angular),
        "longitudinal_rate_linf_Vpm": float(
            xp.max(xp.abs(rate.longitudinal_rate))
        ),
    }


def _integrate_source_intervals(interval, calls, context):
    by_stage = {}
    for call in calls:
        by_stage.setdefault(call.tag.stage, []).append(call)
    records = []
    totals = {}
    for quadrature in interval.quadrature:
        matches = by_stage.get(quadrature.stage, [])
        if len(matches) != 1:
            raise Failure(
                "ManufacturedSource",
                f"expected one {quadrature.stage} source, found {len(matches)}",
            )
        call = matches[0]
        actual_stage = evaluate_stage(call.actual, call.tag, context, False)
        rate = diagnostics.source_rates(
            call.payload,
            call.actual,
            actual_stage,
            context,
        )
        rate_json = _rate_record(rate, context)
        weight = float(quadrature.weight)
        for key, value in rate_json.items():
            totals[key] = totals.get(key, 0.0) + weight * value
        records.append(
            {
                "stage": call.tag.stage,
                "time_s": call.tag.time,
                "weight_s": weight,
                "source_id": int(call.source_id),
                "rates": rate_json,
                "response": {
                    "EL_Vpm": float(
                        context.compute.xp.max(
                            context.compute.xp.abs(actual_stage.cache.EL)
                        )
                    ),
                    "ET_Vpm": float(
                        context.compute.xp.max(
                            context.compute.xp.abs(actual_stage.cache.ET)
                        )
                    ),
                    "physical_B_rate_Tps": float(
                        context.compute.xp.max(
                            context.compute.xp.abs(actual_stage.rhs.B)
                        )
                    ),
                },
            }
        )
    return records, totals


def _configure(context, integrator, steps):
    if integrator not in ("midpoint", "rk4"):
        raise ValueError("integrator must be midpoint or rk4")
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    context.p.integrator = integrator
    context.p.nsteps = steps
    context.p.dt = DURATION_S / steps
    context.p.dt_guard = "warn"


def _json_plain(value):
    if isinstance(value, dict):
        return {str(key): _json_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_plain(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _time_report_record(report, policy):
    record = _json_plain(report)
    record["origin"] = "time.advance.base"
    record["dt_guard"] = policy
    return record


def _ledger_snapshot(cumulative, integrated_totals, intervals):
    ledger_fields = (
        "electric_node",
        "electric_edge",
        "pressure_kinetic",
        "pressure_internal",
        "magnetic",
        "wall",
        "convection",
        "perp_rate",
        "parallel_rate",
        "spectral_kinetic",
        "spectral_perp",
        "spectral_parallel",
    )
    boundary_fields = (
        "number",
        "kinetic",
        "internal_perp",
        "internal_parallel",
        "field_energy",
        "total_energy",
    )
    physical = []
    for species in range(2):
        physical.append(
            {
                name: float(getattr(cumulative.integrals[species], name))
                for name in ledger_fields
            }
        )
    boundary = {
        name: _json_plain(getattr(cumulative.boundary_integrals, name))
        for name in boundary_fields
    }
    source_stages = []
    for interval in intervals:
        for record in interval["rates"]:
            source_stages.append(
                {
                    "step": interval["step"],
                    "stage": record["stage"],
                    "time_s": record["time_s"],
                    "weight_s": interval["quadrature"][
                        [entry["stage"] for entry in interval["quadrature"]].index(
                            record["stage"]
                        )
                    ]["weight_s"],
                    "rates": record["values"],
                    "response": record["response"],
                }
            )
    return {
        "schema": "a0-mms-ledgers-v1",
        "accepted_intervals": int(cumulative.accepted_intervals),
        "last_step": int(cumulative.last_step),
        "last_time_s": float(cumulative.last_time),
        "physical_integrals": physical,
        "physical_boundary_integrals": boundary,
        "source_rate_times_seconds": integrated_totals,
        "source_stage_records": source_stages,
        "production_cumulative_contains_source": False,
    }


def run_level(
    context,
    *,
    integrator,
    steps,
    omit_ion_pressure=False,
    raw_root=None,
):
    """Run one complete production ``time.advance`` MMS verification level."""
    _configure(context, integrator, steps)
    trajectory = NativeMMSTrajectory(context)
    initial_tag = stage_tag(context, "initial")
    initial = trajectory.at(initial_tag)
    cumulative = _copy_cumulative()
    captured = {}
    committed = initialize(
        initial,
        context,
        cumulative,
        True,
        captured,
    )
    initial_stage = SimpleNamespace(**captured)
    estimate = stability.stage_frequency_estimate(initial_stage, context)
    guard = stability.assess_runtime(estimate, context, step=0)
    if guard["action"] not in ("continue", "warn"):
        raise Failure("TimeGuard", f"unexpected manufactured-source guard {guard}")

    source = ManufacturedSource(
        trajectory,
        omit_ion_pressure=omit_ion_pressure,
    )
    checks = list(captured.get("checks", []))
    time_step_reports = []
    intervals = []
    integrated_totals = {}
    maximum_response = {
        "EL_Vpm": 0.0,
        "ET_Vpm": 0.0,
        "physical_B_rate_Tps": 0.0,
    }
    for step in range(steps):
        first_call = len(source.calls)
        result = advance(committed, cumulative, context, source)
        guard_checks = [
            check for check in result.checks if check.name == "time_step_estimate"
        ]
        report = getattr(result, "time_step_report", None)
        if len(guard_checks) != 1 or report is None:
            raise Failure(
                "Provenance",
                "accepted step did not supply one production dt guard report",
            )
        report_record = _time_report_record(report, context.p.dt_guard)
        report_record["check_tag"] = {
            "step": int(guard_checks[0].tag.step),
            "generation": int(guard_checks[0].tag.generation),
            "trial": int(guard_checks[0].tag.trial),
            "time_s": float(guard_checks[0].tag.time),
            "stage": str(guard_checks[0].tag.stage),
        }
        time_step_reports.append(report_record)
        step_calls = source.calls[first_call:]
        source_records, source_totals = _integrate_source_intervals(
            result.interval,
            step_calls,
            context,
        )
        for key, value in source_totals.items():
            integrated_totals[key] = integrated_totals.get(key, 0.0) + value
        for record in source_records:
            for key, value in record["response"].items():
                maximum_response[key] = max(maximum_response[key], value)
        intervals.append(
            {
                "step": int(result.interval.right_step),
                "calls": [
                    {
                        "stage": call.tag.stage,
                        "time_s": call.tag.time,
                        "source_id": int(call.source_id),
                    }
                    for call in step_calls
                ],
                "quadrature": [
                    {
                        "stage": record["stage"],
                        "weight_s": record["weight_s"],
                    }
                    for record in source_records
                ],
                "rates": [
                    {
                        "stage": record["stage"],
                        "time_s": record["time_s"],
                        "values": record["rates"],
                        "response": record["response"],
                    }
                    for record in source_records
                ],
                "accepted": bool(result.interval.accepted),
            }
        )
        checks.extend(result.checks)
        if len(step_calls) != (2 if integrator == "midpoint" else 4):
            raise Failure(
                "ManufacturedSource",
                f"{integrator} source call count is {len(step_calls)}",
            )

    exact_tag = StageTag(
        step=committed.state.step,
        generation=committed.state.generation,
        trial=cumulative.last_trial,
        spectral_context=context.spectral_id,
        time=committed.state.time,
        stage="endpoint",
    )
    exact = trajectory.at(exact_tag)
    errors = normalized_errors(committed.state, exact, context)
    raw_records = [serialize_check_record(check) for check in checks]
    debug_gate = audit_check_records(raw_records)
    ledger_snapshot = _ledger_snapshot(
        cumulative,
        integrated_totals,
        intervals,
    )
    raw_evidence = {
        "check_count": len(raw_records),
        "checks_path": None,
        "checks_sha256": None,
        "ledger_path": None,
        "ledger_sha256": None,
        "debug_gate": debug_gate,
    }
    if raw_root is not None:
        raw_directory = Path(raw_root)
        raw_directory.mkdir(parents=True, exist_ok=True)
        records, checks_path = write_check_jsonl(
            raw_directory / "checks.jsonl",
            checks,
        )
        if len(records) != len(raw_records):
            raise Failure("Provenance", "raw check retention count changed")
        ledger_path = write_json(
            raw_directory / "ledgers.json",
            ledger_snapshot,
        )
        raw_evidence.update(
            {
                "checks_path": "checks.jsonl",
                "checks_sha256": sha256_file(checks_path),
                "ledger_path": "ledgers.json",
                "ledger_sha256": sha256_file(ledger_path),
            }
        )
    runtime_integrated = bool(
        len(time_step_reports) == steps
        and all(
            record["origin"] == "time.advance.base"
            and isinstance(record.get("accepted"), bool)
            for record in time_step_reports
        )
    )
    cold_pre_estimate = {
        "origin": "harness cold initial-stage scalar estimate",
        "runtime_integrated": False,
        "policy": context.p.dt_guard,
        "omega_estimate_rad_s": estimate["omega_estimate"],
        "omega_dt": guard["omega_dt"],
        "samples_per_cycle": guard["samples_per_cycle"],
        "accepted_by_scalar_estimate": guard["accepted"],
        "reasons": guard["reasons"],
        "action": guard["action"],
        "scope": estimate["scope"],
    }
    context.compute.synchronize()
    return {
        "integrator": integrator,
        "steps": steps,
        "dt_s": context.p.dt,
        "duration_s": DURATION_S,
        "levels_note": (
            "reference midpoint levels"
            if integrator == "midpoint"
            else "RK4 levels selected to remain above roundoff"
        ),
        "errors": errors,
        "accepted_intervals": int(cumulative.accepted_intervals),
        "check_count": len(checks),
        "all_hard_checks_passed": _checks_are_clean(checks),
        "time_step_reports": time_step_reports,
        "guard_runtime_integrated": runtime_integrated,
        "raw_evidence": raw_evidence,
        "maximum_response": maximum_response,
        "source_intervals": intervals,
        "source_integrals_rate_times_seconds": integrated_totals,
        "time_guard": {
            "policy": context.p.dt_guard,
            "runtime_integrated": runtime_integrated,
            "production_report_count": len(time_step_reports),
            "cold_pre_estimate": cold_pre_estimate,
        },
        "cumulative": {
            "accepted_intervals": int(cumulative.accepted_intervals),
            "last_step": int(cumulative.last_step),
            "last_time_s": float(cumulative.last_time),
            "source_terms_in_production_ledger": False,
        },
    }


def _orders(cases):
    names = tuple(cases[0]["errors"])
    result = {}
    for name in names:
        result[name] = [
            math.log2(cases[index]["errors"][name] / cases[index + 1]["errors"][name])
            for index in range(len(cases) - 1)
        ]
    return result


def _failure_record(error, *, failure_arrays=None, raw_checks=None):
    stage = getattr(error, "stage", None)
    return {
        "code": getattr(error, "code", type(error).__name__),
        "message": str(error),
        "stage": None
        if stage is None
        else {
            "step": stage.step,
            "generation": stage.generation,
            "trial": stage.trial,
            "time_s": stage.time,
            "name": stage.stage,
        },
        "check_count": len(getattr(error, "checks", [])),
        "partial_arrays": sorted(getattr(error, "partial_arrays", {})),
        "failure_arrays": failure_arrays
        if failure_arrays is not None
        else {
            "path": None,
            "sha256": None,
            "contains_nonfinite": False,
            "arrays": {},
        },
        "raw_checks": raw_checks
        if raw_checks is not None
        else {
            "path": None,
            "sha256": None,
            "count": 0,
        },
        "time_step_report": _json_plain(
            getattr(error, "time_step_report", None)
        ),
    }


def run_verification(
    integrator,
    *,
    device="cpu",
    levels=None,
    spatial=True,
    raw_root=None,
):
    """Run all requested levels and return deterministic JSON-compatible data."""
    if levels is None:
        levels = MIDPOINT_LEVELS if integrator == "midpoint" else RK4_LEVELS
    levels = tuple(int(level) for level in levels)
    cases = []
    for steps in levels:
        case_root = (
            Path(raw_root) / f"{integrator}-{steps}"
            if raw_root is not None
            else None
        )
        try:
            context = make_context(device)
            case = run_level(
                context,
                integrator=integrator,
                steps=steps,
                raw_root=case_root,
            )
            if case_root is not None:
                evidence = case["raw_evidence"]
                for key in ("checks_path", "ledger_path"):
                    relative = evidence[key]
                    evidence[key] = (
                        str(case_root / relative)
                        if relative is not None
                        else None
                    )
            cases.append(case)
        except Failure as error:
            failure_arrays = (
                write_failure_arrays(
                    case_root,
                    error,
                    f"failure-{integrator}-{steps}",
                )
                if case_root is not None
                else None
            )
            raw_checks = (
                write_failure_checks(
                    case_root,
                    error,
                    f"failure-checks-{integrator}-{steps}",
                )
                if case_root is not None
                else None
            )
            return {
                "integrator": integrator,
                "device": device,
                "levels": list(levels),
                "cases": cases,
                "passed": False,
                "failure": _failure_record(
                    error,
                    failure_arrays=failure_arrays,
                    raw_checks=raw_checks,
                ),
            }

    orders = _orders(cases)
    tolerance = (1.7, 2.3) if integrator == "midpoint" else (3.5, 4.5)
    accuracy = all(
        tolerance[0] < order < tolerance[1]
        for values in orders.values()
        for order in values
    )
    response = {
        key: max(case["maximum_response"][key] for case in cases)
        for key in ("EL_Vpm", "ET_Vpm", "physical_B_rate_Tps")
    }
    first = cases[0]
    midpoint_quadrature = [
        record["stage"]
        for record in first["source_intervals"][0]["quadrature"]
    ]
    nonzero_rates = {}
    for interval in first["source_intervals"]:
        for stage in interval["rates"]:
            for key, value in stage["values"].items():
                if abs(value) > 0.0:
                    nonzero_rates.setdefault(key, {})[stage["stage"]] = value
    nonzero_integrals = {
        key: value
        for key, value in first["source_integrals_rate_times_seconds"].items()
        if abs(value) > 0.0
    }
    passed = (
        accuracy
        and all(case["all_hard_checks_passed"] for case in cases)
        and all(case["guard_runtime_integrated"] for case in cases)
        and all(
            case["accepted_intervals"] == case["steps"]
            and all(value > 0.0 for value in response.values())
            for case in cases
        )
    )
    result = {
        "case": "coupled_discrete_temporal_MMS",
        "integrator": integrator,
        "device": device,
        "levels": list(levels),
        "method_order_acceptance": {
            "expected": 2 if integrator == "midpoint" else 4,
            "interval": list(tolerance),
        },
        "cases": cases,
        "orders": orders,
        "maximum_response": response,
        "parameters": {
            "K": 1,
            "Nphi": 9,
            "grid": [4, 4],
            "duration_s": DURATION_S,
            "speed_mps": SPEED_MPS,
            "density_reference_m^-3": 1.0e12,
            "pressure_reference_Pa": 1.0e-9,
            "charge_reference_C": 1.0e-7,
            "magnetic_reference_T": 1.0e-5,
            "reference_velocity_mps": 1.0,
        },
        "reference": (
            "independent analytic time trajectory; prescribed source "
            "U_dot - R(U_reference) evaluated only at the exact reference"
        ),
        "source_accounting": {
            "calls_per_step": 2 if integrator == "midpoint" else 4,
            "interval_quadrature_stages": midpoint_quadrature,
            "quadrature_weight_s": (
                first["dt_s"]
                if integrator == "midpoint"
                else None
            ),
            "per_stage_weights": (
                ["dt"]
                if integrator == "midpoint"
                else ["dt/6", "dt/3", "dt/3", "dt/6"]
            ),
            "nonzero_rates": nonzero_rates,
            "nonzero_integrals": nonzero_integrals,
            "production_cumulative_contains_source": False,
            "production_cumulative_check": first["cumulative"],
            "intervals": first["source_intervals"],
            "scope": (
                "test-only prescribed forcing; production cumulative ledger "
                "retains physical terms only"
            ),
        },
        "diagnostics": {
            "production_stage_advance_and_interval_gates": True,
            "check_count": sum(case["check_count"] for case in cases),
            "time_guard": {
                "runtime_integrated": all(
                    case["guard_runtime_integrated"] for case in cases
                ),
                "policy": first["time_guard"]["policy"],
                "cold_pre_estimate": first["time_guard"][
                    "cold_pre_estimate"
                ],
                "production_reports_per_level": {
                    str(case["steps"]): len(case["time_step_reports"])
                    for case in cases
                },
            },
            "raw_evidence": {
                str(case["steps"]): case["raw_evidence"] for case in cases
            },
        },
        "scope": (
            "Temporal verification of the coupled discrete equations under "
            "known manufactured sources; not physical or experimental validation"
        ),
        "passed": passed,
    }
    if spatial:
        result["spatial"] = spatial_verification(levels=SPATIAL_LEVELS)
        result["passed"] = bool(result["passed"] and result["spatial"]["passed"])
    return result


_CACHE = {}


def cached_case(integrator):
    if integrator not in _CACHE:
        _CACHE[integrator] = run_verification(
            integrator,
            device="cpu",
            spatial=False,
        )
    return _CACHE[integrator]


def aggregate_raw_checks(results, negative):
    """Count each normal and negative checks.jsonl sidecar exactly once."""
    by_level = {}
    normal_records = 0
    normal_sidecars = 0
    paths = []
    for integrator, result in results.items():
        for case in result.get("cases", []):
            evidence = case["raw_evidence"]
            count = int(evidence["check_count"])
            normal_records += count
            normal_sidecars += 1
            by_level[f"{integrator}-{case['steps']}"] = {
                "records": count,
                "path": evidence.get("checks_path"),
                "hard_integrity_passed": evidence["debug_gate"][
                    "hard_integrity_passed"
                ],
            }
            if evidence.get("checks_path") is not None:
                paths.append(str(evidence["checks_path"]))
    negative_evidence = negative["raw_evidence"]
    negative_records = int(negative_evidence["check_count"])
    by_level["negative"] = {
        "records": negative_records,
        "path": negative_evidence.get("checks_path"),
        "hard_integrity_passed": negative_evidence["debug_gate"][
            "hard_integrity_passed"
        ],
    }
    if negative_evidence.get("checks_path") is not None:
        paths.append(str(negative_evidence["checks_path"]))
    if len(paths) != len(set(paths)):
        raise Failure("Provenance", "raw sidecar path counted more than once")
    return {
        "normal_sidecar_count": normal_sidecars,
        "negative_sidecar_count": 1,
        "sidecar_count": normal_sidecars + 1,
        "normal_records": normal_records,
        "negative_records": negative_records,
        "total_records": normal_records + negative_records,
        "hard_integrity_passed": all(
            entry["hard_integrity_passed"] for entry in by_level.values()
        ),
        "by_level": by_level,
    }


def omitted_ion_pressure_control(device="cpu", *, raw_root=None):
    """Run the required negative source-omission control once."""
    context = make_context(device)
    result = run_level(
        context,
        integrator="midpoint",
        steps=8,
        omit_ion_pressure=True,
        raw_root=raw_root,
    )
    if raw_root is not None:
        evidence = result["raw_evidence"]
        raw_directory = Path(raw_root).resolve()
        for key in ("checks_path", "ledger_path"):
            relative = evidence[key]
            evidence[key] = (
                str(raw_directory / relative) if relative is not None else None
            )
    threshold = 0.1
    detected = bool(
        result["errors"]["perp_i"] > threshold
        and result["errors"]["combined"] > threshold
    )
    return {
        "case": "omitted_ion_pressure_source",
        "integrator": "midpoint",
        "steps": 8,
        "minimum_detected_error": threshold,
        "errors": result["errors"],
        "physical_gates_passed": result["all_hard_checks_passed"],
        "accepted_intervals": result["accepted_intervals"],
        "time_step_reports": result["time_step_reports"],
        "time_guard": result["time_guard"],
        "raw_evidence": result["raw_evidence"],
        "detected": detected,
        "passed": bool(detected and result["all_hard_checks_passed"]),
        "scope": (
            "The omitted manufactured source leaves the physical interval "
            "identity self-consistent but misses the analytic trajectory."
        ),
    }


class _ErrorStats:
    def __init__(self):
        self.inner_max = 0.0
        self.inner_scale = 0.0
        self.wall_max = 0.0
        self.total_max = 0.0
        self.total_scale = 0.0

    def account(self, radius, height, boundary, actual, exact):
        values = (radius, height, actual.real, actual.imag, exact.real, exact.imag)
        if not all(math.isfinite(value) for value in values):
            raise Failure("Spatial", "nonfinite analytic spatial verification value")
        error = abs(actual - exact)
        magnitude = abs(exact)
        self.total_max = max(self.total_max, error)
        self.total_scale = max(self.total_scale, magnitude)
        if boundary:
            self.wall_max = max(self.wall_max, error)
        if 1.25 <= radius <= 1.75 and 0.25 <= height <= 0.75:
            self.inner_max = max(self.inner_max, error)
            self.inner_scale = max(self.inner_scale, magnitude)

    def record(self):
        if self.inner_scale <= 0.0 or self.total_scale <= 0.0:
            raise Failure("Spatial", "empty analytic spatial scale")
        return {
            "inner_error": self.inner_max / self.inner_scale,
            "wall_error": self.wall_max / self.total_scale,
            "total_error": self.total_max / self.total_scale,
            "inner_max_error": self.inner_max,
            "wall_max_error": self.wall_max,
            "total_max_error": self.total_max,
            "inner_scale": self.inner_scale,
            "total_scale": self.total_scale,
        }


def _axis_value(coordinate, center):
    scaled = (coordinate - center) / 0.42
    if abs(scaled) >= 1.0:
        return 0.0
    return math.exp(-1.0 / (1.0 - scaled * scaled))


def _axis_derivative(coordinate, center):
    scaled = (coordinate - center) / 0.42
    if abs(scaled) >= 1.0:
        return 0.0
    one_minus = 1.0 - scaled * scaled
    return (
        _axis_value(coordinate, center)
        * (-2.0 * scaled / (0.42 * one_minus * one_minus))
    )


def _bump(radius, height):
    radial = _axis_value(radius, 1.5)
    axial = _axis_value(height, 0.5)
    return (
        radial * axial,
        _axis_derivative(radius, 1.5) * axial,
        radial * _axis_derivative(height, 0.5),
    )


def _spatial_level(n):
    geometry = build_rectangle(n + 1, n + 1, 1.0, 0.0, 1.0 / n, 1.0 / n)
    backend = Backend("cpu")
    operators = build_spatial_operators(
        geometry,
        SPATIAL_MODE,
        OperatorOptions(1.0e-10, 3 << 30),
        backend,
    )
    nr = len(geometry.r_faces.keys)
    nc = len(geometry.cells.keys)
    nz = len(geometry.z_faces.keys)
    nv = len(geometry.vertices.keys)
    complex_coefficient = 0.8 - 0.3j

    gradient_stats = _ErrorStats()
    scalar = np.zeros(nc, dtype=np.complex128)
    for index in range(nc):
        scalar[index] = complex_coefficient * _bump(
            geometry.cells.R[index], geometry.cells.Z[index]
        )[0]
    gradient = operators.G @ scalar
    for index in range(nr):
        radius = geometry.r_faces.R[index]
        height = geometry.r_faces.Z[index]
        gradient_stats.account(
            radius,
            height,
            geometry.r_faces.boundary[index] != 0,
            gradient[index],
            complex_coefficient
            * _bump(radius, height)[1],
        )
    for index in range(nc):
        radius = geometry.cells.R[index]
        height = geometry.cells.Z[index]
        gradient_stats.account(
            radius,
            height,
            geometry.cells.boundary[index] != 0,
            gradient[nr + index],
            1j * SPATIAL_MODE * complex_coefficient
            * _bump(radius, height)[0]
            / radius,
        )
    for index in range(nz):
        radius = geometry.z_faces.R[index]
        height = geometry.z_faces.Z[index]
        gradient_stats.account(
            radius,
            height,
            geometry.z_faces.boundary[index] != 0,
            gradient[nr + nc + index],
            complex_coefficient
            * _bump(radius, height)[2],
        )

    er_value, ephi_value, ez_value = 1.0, -0.5 + 0.25j, 0.75

    def electric(locations):
        out = np.zeros(len(locations.R), dtype=np.complex128)
        for index in range(len(locations.R)):
            bump = _bump(locations.R[index], locations.Z[index])[0]
            if locations is geometry.r_faces:
                out[index] = er_value * bump
            elif locations is geometry.cells:
                out[index] = ephi_value * bump
            else:
                out[index] = ez_value * bump
        return out

    electric_vector = np.concatenate(
        (
            electric(geometry.r_faces),
            electric(geometry.cells),
            electric(geometry.z_faces),
        )
    )

    def curl_stats(actual, flip_phi_derivative=False):
        stats = _ErrorStats()
        for index in range(nz):
            radius = geometry.z_faces.R[index]
            height = geometry.z_faces.Z[index]
            values = _bump(radius, height)
            exact = (
                1j * SPATIAL_MODE * ez_value * values[0] / radius
                - ephi_value * values[2]
            )
            stats.account(
                radius,
                height,
                geometry.z_faces.boundary[index] != 0,
                actual[index],
                exact,
            )
        for index in range(nv):
            radius = geometry.vertices.R[index]
            height = geometry.vertices.Z[index]
            values = _bump(radius, height)
            exact = er_value * values[2] - ez_value * values[1]
            stats.account(
                radius,
                height,
                geometry.vertices.boundary[index] != 0,
                actual[nz + index],
                -exact if flip_phi_derivative else exact,
            )
        for index in range(nr):
            radius = geometry.r_faces.R[index]
            height = geometry.r_faces.Z[index]
            values = _bump(radius, height)
            exact = (
                ephi_value * (values[0] / radius + values[1])
                - 1j * SPATIAL_MODE * er_value * values[0] / radius
            )
            stats.account(
                radius,
                height,
                geometry.r_faces.boundary[index] != 0,
                actual[nz + nv + index],
                exact,
            )
        return stats

    curl_actual = operators.C @ electric_vector
    curl_record = curl_stats(curl_actual).record()
    wrong_sign = curl_stats(curl_actual, flip_phi_derivative=True).record()

    br_value, bphi_value, bz_value = 0.75 + 0.1j, 0.25 - 0.5j, -0.5 + 0.2j

    def field(locations, value):
        out = np.zeros(len(locations.R), dtype=np.complex128)
        for index in range(len(locations.R)):
            out[index] = value * _bump(
                locations.R[index], locations.Z[index]
            )[0]
        return out

    field_vector = np.concatenate(
        (
            field(geometry.z_faces, br_value),
            field(geometry.vertices, bphi_value),
            field(geometry.r_faces, bz_value),
        )
    )
    divergence_actual = operators.DB @ field_vector
    divergence_stats = _ErrorStats()
    for index in range(nv):
        radius = geometry.vertices.R[index]
        height = geometry.vertices.Z[index]
        values = _bump(radius, height)
        exact = (
            br_value * (values[0] / radius + values[1])
            + bz_value * values[2]
            + 1j * SPATIAL_MODE * bphi_value * values[0] / radius
        )
        divergence_stats.account(
            radius,
            height,
            geometry.vertices.boundary[index] != 0,
            divergence_actual[index],
            exact,
        )

    jr_value, jphi_value, jz_value = 0.6 - 0.2j, -0.4 + 0.3j, 0.8 + 0.1j
    flux_vector = np.concatenate(
        (
            field(geometry.r_faces, jr_value),
            field(geometry.cells, jphi_value),
            field(geometry.z_faces, jz_value),
        )
    )
    adjoint_actual = operators.DJ @ flux_vector
    adjoint_stats = _ErrorStats()
    for index in range(nc):
        radius = geometry.cells.R[index]
        height = geometry.cells.Z[index]
        values = _bump(radius, height)
        exact = (
            jr_value * (values[0] / radius + values[1])
            + jz_value * values[2]
            + 1j * SPATIAL_MODE * jphi_value * values[0] / radius
        )
        adjoint_stats.account(
            radius,
            height,
            geometry.cells.boundary[index] != 0,
            adjoint_actual[index],
            exact,
        )

    cg = np.asarray(operators.C @ operators.G)
    dbc = np.asarray(operators.DB @ operators.C)
    cg_bound = (
        128.0
        * np.finfo(np.float64).eps
        * operators.C.shape[1]
        * np.max(np.abs(operators.C))
        * np.max(np.abs(operators.G))
    )
    dbc_bound = (
        128.0
        * np.finfo(np.float64).eps
        * operators.DB.shape[1]
        * np.max(np.abs(operators.DB))
        * np.max(np.abs(operators.C))
    )
    return {
        "N": n,
        "h_m": 1.0 / n,
        "gradient": gradient_stats.record(),
        "curl": curl_record,
        "divergence": divergence_stats.record(),
        "divergence_adjoint": adjoint_stats.record(),
        "compatibility": {
            "max_abs_CG": float(np.max(np.abs(cg))),
            "CG_roundoff_bound": float(cg_bound),
            "max_abs_DBC": float(np.max(np.abs(dbc))),
            "DBC_roundoff_bound": float(dbc_bound),
            "passed": bool(
                np.max(np.abs(cg)) <= cg_bound
                and np.max(np.abs(dbc)) <= dbc_bound
            ),
        },
        "wrong_sign_curl": wrong_sign,
    }


def spatial_verification(levels=SPATIAL_LEVELS):
    """Measure inner, wall, and total smooth spatial convergence independently."""
    try:
        records = [_spatial_level(int(level)) for level in levels]
    except Failure as error:
        return {
            "case": "smooth_rectangular_spatial_verification",
            "levels": list(levels),
            "passed": False,
            "failure": _failure_record(error),
        }
    operators = ("gradient", "curl", "divergence", "divergence_adjoint")
    orders = {scope: {} for scope in ("inner", "wall", "total")}

    def convergence_order(coarse, fine):
        if (
            not math.isfinite(coarse)
            or not math.isfinite(fine)
            or coarse <= 0.0
            or fine <= 0.0
        ):
            return None
        return math.log2(coarse / fine) / (len(records) - 1)

    for operator in operators:
        orders["inner"][operator] = convergence_order(
            records[0][operator]["inner_error"],
            records[-1][operator]["inner_error"],
        )
        orders["wall"][operator] = convergence_order(
            records[0][operator]["wall_error"],
            records[-1][operator]["wall_error"],
        )
        orders["total"][operator] = convergence_order(
            records[0][operator]["total_error"],
            records[-1][operator]["total_error"],
        )
    negative = records[0]["wrong_sign_curl"]["inner_error"] > 0.1
    compatibility = all(record["compatibility"]["passed"] for record in records)
    accuracy = all(
        value > 3.5 and math.isfinite(value)
        for value in orders["inner"].values()
    )
    return {
        "case": "smooth_rectangular_spatial_verification",
        "device": "cpu",
        "mode": SPATIAL_MODE,
        "region": {
            "R_m": [1.25, 1.75],
            "Z_m": [0.25, 0.75],
        },
        "levels": records,
        "orders": orders,
        "negative_control": {
            "inner_error": records[0]["wrong_sign_curl"]["inner_error"],
            "passed": bool(negative),
            "description": "deliberately reversed analytic curl vertex term",
        },
        "compatibility_passed": compatibility,
        "scope": (
            "CPU rectangular RZ Fourier analytic convergence; wall and total "
            "are reported separately and no global fourth-order claim is made"
        ),
        "passed": bool(accuracy and negative and compatibility),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_runtime_origin(explicit_root=None):
    """Derive runtime ownership from the imported package, not a trusted root."""
    package_root = Path(__import__("fomhd").__file__).resolve().parent
    if package_root.name != "fomhd":
        raise Failure("RuntimeIdentity", "imported fomhd package is malformed")
    if package_root.parent.name != "src":
        raise Failure(
            "RuntimeIdentity",
            f"imported fomhd package is not under src/fomhd: {package_root}",
        )
    runtime_root = package_root.parents[1]
    metadata = {
        name: runtime_root / name for name in ("pyproject.toml", "uv.lock")
    }
    missing = [str(path) for path in metadata.values() if not path.is_file()]
    if missing:
        raise Failure(
            "RuntimeIdentity",
            f"imported runtime has no matching repository metadata: {missing}",
        )
    modules = sorted(package_root.glob("*.py"))
    loaded = {}
    for name, module in sorted(sys.modules.items()):
        if name != "fomhd" and not name.startswith("fomhd."):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        path = Path(origin).resolve()
        if not path.is_relative_to(package_root):
            raise Failure(
                "RuntimeIdentity",
                f"loaded module {name} resolves outside imported package: {path}",
            )
        loaded[name] = str(path)
    if explicit_root is not None:
        requested = Path(explicit_root).resolve()
        if requested == package_root:
            requested = requested.parents[1]
        elif requested == package_root.parent:
            requested = requested.parent
        if requested != runtime_root:
            raise Failure(
                "RuntimeMismatch",
                f"RuntimeMismatch: explicit runtime root {requested} does not "
                f"match imported package root {runtime_root}",
            )
    return {
        "root": str(runtime_root),
        "package_root": str(package_root),
        "metadata": {name: str(path) for name, path in metadata.items()},
        "modules": [str(path) for path in modules],
        "loaded_modules": loaded,
    }


def _harness_origin(harness_root=None, cli_path=None):
    fixture = Path(__file__).resolve()
    actual_root = fixture.parents[2]
    if harness_root is not None and Path(harness_root).resolve() != actual_root:
        raise Failure(
            "HarnessMismatch",
            f"explicit harness root {Path(harness_root).resolve()} does not "
            f"match executed fixture root {actual_root}",
        )
    tool = (
        Path(cli_path).resolve()
        if cli_path is not None
        else actual_root / "tools" / "warp_mms.py"
    )
    files = {
        "tests/warp/mms_fixture.py": fixture,
        "tests/warp/test_mms.py": fixture.parent / "test_mms.py",
        "tests/warp/foundation_fixture.py": fixture.parent / "foundation_fixture.py",
        "tools/warp_mms.py": tool,
        "docs/pseudocode/tests/warp/mms_fixture.py.md": (
            actual_root / "docs/pseudocode/tests/warp/mms_fixture.py.md"
        ),
        "docs/pseudocode/tools/warp_mms.py.md": (
            actual_root / "docs/pseudocode/tools/warp_mms.py.md"
        ),
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise Failure("HarnessIdentity", f"missing executed harness file: {missing}")
    return actual_root, files


def build_identity(
    harness_root=None,
    *,
    runtime_root=None,
    cli_path=None,
):
    """Hash the actual imported runtime, repository metadata, and harness files."""
    runtime = resolve_runtime_origin(runtime_root)
    harness, harness_files = _harness_origin(harness_root, cli_path)
    runtime_root_path = Path(runtime["root"])
    runtime_hashes = {
        str(Path(path).relative_to(runtime_root_path)): sha256_file(Path(path))
        for path in runtime["modules"]
    }
    repository_hashes = {
        str(Path(path).relative_to(runtime_root_path)): sha256_file(Path(path))
        for path in runtime["metadata"].values()
    }
    harness_hashes = {
        relative: sha256_file(path) for relative, path in harness_files.items()
    }
    source_hashes = {}
    source_hashes.update(runtime_hashes)
    source_hashes.update(repository_hashes)
    source_hashes.update(
        {f"harness/{name}": value for name, value in harness_hashes.items()}
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "precision": "float64/complex128",
        "runtime_root": runtime["root"],
        "package_root": runtime["package_root"],
        "harness_root": str(harness),
        "imported_modules": runtime["loaded_modules"],
        "runtime_hashes": runtime_hashes,
        "repository_hashes": repository_hashes,
        "harness_hashes": harness_hashes,
        "source_hashes": source_hashes,
    }


def archive_source_snapshot(
    snapshot_root,
    *,
    runtime_root=None,
    harness_root=None,
    cli_path=None,
):
    """Archive the actual imported runtime and actual harness with SHA-256."""
    runtime = resolve_runtime_origin(runtime_root)
    harness, harness_files = _harness_origin(harness_root, cli_path)
    destination_root = Path(snapshot_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    entries = []

    def archive(source, relative, role):
        source = Path(source).resolve()
        relative = Path(relative)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        entries.append(
            {
                "role": role,
                "path": str(relative),
                "source_path": str(source),
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )

    runtime_root_path = Path(runtime["root"])
    for module in runtime["modules"]:
        path = Path(module)
        archive(
            path,
            Path("runtime") / path.relative_to(runtime_root_path),
            "runtime_module",
        )
    for name, meta_path in runtime["metadata"].items():
        archive(
            meta_path,
            Path("repository") / name,
            "dependency_lock" if name == "uv.lock" else "project_metadata",
        )
    for relative, source in harness_files.items():
        archive(source, Path("harness") / relative, "verification_harness")
    manifest = {
        "schema": "a0-source-snapshot-v2",
        "runtime_root": runtime["root"],
        "package_root": runtime["package_root"],
        "loaded_modules": runtime["loaded_modules"],
        "harness_root": str(harness),
        "file_count": len(entries),
        "runtime_module_count": len(runtime["modules"]),
        "files": entries,
        "scope": (
            "actual imported runtime checkout, matching pyproject/lock, and "
            "actual executed verification harness only"
        ),
    }
    manifest_path = destination_root / "SOURCE_MANIFEST.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest
