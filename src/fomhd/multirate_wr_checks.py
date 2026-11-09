"""Mixed coarse-ion/fine-field interval accounting for A1-WR-050.

This module consumes the accepted final old-node sweep. It returns host-only
integrated SI records and a narrow view of the actual fine field stages for
``massless_checks.check_step``. It does not run ion transport, commit a
cumulative ledger, or define a restart schema.
"""

from __future__ import annotations

import math
import numbers
import sys
from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import Any

import numpy as np

from . import core, massless_checks
from .core import forward, inverse
from .types import Failure


LEDGER_FIELDS = (
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
BOUNDARY_FIELDS = (
    "number",
    "kinetic",
    "internal_perp",
    "internal_parallel",
)
SourceKey = tuple[int, int, int, int, int, str, str, int, int, int]


def _finite_float(value: Any, name: str) -> float:
    if (
        not isinstance(value, numbers.Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise Failure("WRInterval", f"{name} must be a finite non-boolean real")
    return float(value)


@dataclass(frozen=True, slots=True)
class LedgerIntegral:
    """Twelve preintegrated SI work values for one species over one macro H."""

    electric_node: float
    electric_edge: float
    pressure_kinetic: float
    pressure_internal: float
    magnetic: float
    wall: float
    convection: float
    perp_rate: float
    parallel_rate: float
    spectral_kinetic: float
    spectral_perp: float
    spectral_parallel: float


@dataclass(frozen=True, slots=True)
class BoundaryIntegral:
    """Six frozen SI boundary integrals over one macro H."""

    number: tuple[float, float]
    kinetic: tuple[float, float]
    internal_perp: tuple[float, float]
    internal_parallel: tuple[float, float]
    field_energy: float
    total_energy: float


@dataclass(frozen=True, slots=True)
class MixedInterval:
    """Immutable host metadata and integrated values for one accepted window."""

    trial: int
    node_version: int
    left_step: int
    right_step: int
    left_time: float
    right_time: float
    ledger: tuple[LedgerIntegral, LedgerIntegral]
    boundary: BoundaryIntegral
    outward: tuple[float, float, float]
    outward_torque: float
    projection_reaction: tuple[float, float, float]
    projection_torque: float
    density_work_coarse: float
    density_work_fine: float
    kinetic_path: float
    coarse_sources: tuple[tuple[SourceKey, float], ...]
    fine_sources: tuple[tuple[SourceKey, float], ...]


@dataclass(slots=True)
class MixedDiagnosticWork:
    """Actual fine-work diagnostics performed outside the physical producer."""

    electron_ledger_calls: int = 0
    electron_pressure_only_projections: int = 0
    electron_ledger_inverse_transforms: int = 0
    wall_work_metric_calls: int = 0
    wall_work_inverse_transforms: int = 0
    wall_work_mode_envelope_sets: int = 0
    electron_work_gate_calls: int = 0
    electron_gate_transforms: int = 0
    work_gate_records: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "electron_ledger_calls": self.electron_ledger_calls,
            "electron_pressure_only_projections": (
                self.electron_pressure_only_projections
            ),
            "electron_ledger_inverse_transforms": (
                self.electron_ledger_inverse_transforms
            ),
            "wall_work_metric_calls": self.wall_work_metric_calls,
            "wall_work_inverse_transforms": self.wall_work_inverse_transforms,
            "wall_work_mode_envelope_sets": self.wall_work_mode_envelope_sets,
            "electron_work_gate_calls": self.electron_work_gate_calls,
            "electron_gate_transforms": self.electron_gate_transforms,
            "work_gate_records": self.work_gate_records,
        }


@dataclass(frozen=True, slots=True)
class FineFieldRhsView:
    """The actual field RHS plus the retained electron pressure rates."""

    B: Any
    fluid: tuple[None, SimpleNamespace]


@dataclass(frozen=True, slots=True)
class FineStageView:
    """Actual fine producer fields, with no full ion-stage claims."""

    kind: str
    complete_stage: bool
    diagnostic_scope: str
    state: Any
    fluid: tuple[Any, Any]
    cache: Any
    current: Any
    rhs: FineFieldRhsView
    tag: Any
    source_id: int
    checks: tuple[Any, ...]
    frame: Any


@dataclass(frozen=True, slots=True)
class MixedIntervalBundle:
    """One frozen interval and the fine quadrature that produced it."""

    interval: MixedInterval
    fine_views: tuple[FineStageView, ...]
    diagnostic_work: MixedDiagnosticWork


@dataclass(frozen=True, slots=True)
class _BoundarySummands:
    number: tuple[float, float]
    kinetic: tuple[float, float]
    internal_perp: tuple[float, float]
    internal_parallel: tuple[float, float]
    energy: tuple[float, float]
    field_energy: float


def _xp(context):
    return context.compute.xp


def _physical_real(value, name):
    return massless_checks._physical_real(value, name)


def _integrate_cells(values, components, context):
    return massless_checks._integrate_cells(values, components, context)


def _integrate_edges(values, context):
    return massless_checks._integrate_edges(values, context)


def _array_equal(left, right, xp):
    if not isinstance(left, xp.ndarray) or not isinstance(right, xp.ndarray):
        return False
    return bool(
        left.dtype == right.dtype
        and left.shape == right.shape
        and xp.array_equal(left, right)
    )


def _all_finite(value, xp):
    return bool(xp.all(xp.isfinite(value)))


def _source_key(source_id, tag) -> SourceKey:
    return (
        int(source_id),
        int(tag.step),
        int(tag.generation),
        int(tag.trial),
        int(tag.spectral_context),
        float(tag.time).hex(),
        str(tag.stage),
        int(getattr(tag, "micro_step", -1)),
        int(getattr(tag, "iteration", 0)),
        int(getattr(tag, "slab", 0)),
    )


def _validate_weight(weight, name):
    value = _finite_float(weight, name)
    if value <= 0.0:
        raise Failure("WRQuadrature", f"{name} must be finite and positive")
    return value


def _validate_source_records(window, left_state, context):
    s = int(window.s)
    M = int(window.M)
    version = int(window.final_input_version)
    if s not in (3, 5):
        raise Failure("WRInterval", "selected WR node count must be 3 or 5")
    if len(window.coarse_samples) != s:
        raise Failure("WREvidence", "coarse sample count does not match s")
    if len(window.fine_samples) != 4 * M:
        raise Failure("WREvidence", "fine sample count does not match 4*M")
    if version != int(window.accepted_node_version):
        raise Failure("WRVersion", "final input version is not accepted")
    if version != int(window.mapped_node_version) - 1:
        raise Failure("WRVersion", "mapped version does not follow old version")

    xp = _xp(context)
    coarse_keys = []
    fine_keys = []
    source_trials = set()
    for node, sample in enumerate(window.coarse_samples):
        record = sample.record
        if record is None:
            raise Failure("WREvidence", "coarse sample has no producer record")
        if sample.coarse_node != node:
            raise Failure("WREvidence", "coarse sample order is inconsistent")
        if not _array_equal(sample.P, window.P[node], xp):
            raise Failure("WREvidence", "coarse P does not match accepted nodes")
        if not _array_equal(sample.Z, window.Z_coarse[node], xp):
            raise Failure(
                "WREvidence",
                "coarse Z does not match accepted field nodes",
            )
        if sample.identity.iteration != version or record.identity != sample.identity:
            raise Failure("WRVersion", "coarse sample is not the accepted old version")
        if record.tag.iteration != version:
            raise Failure("WRVersion", "coarse producer version mismatch")
        if record.tag.spectral_context != int(context.spectral_id):
            raise Failure("WRIdentity", "coarse spectral context mismatch")
        if int(record.tag.slab) != 0:
            raise Failure("WRIdentity", "unexpected coarse slab source")
        if record.tag.step != int(left_state.step):
            raise Failure("WRIdentity", "coarse producer step mismatch")
        if record.tag.time != float(left_state.time):
            raise Failure("WRIdentity", "coarse producer clock mismatch")
        if record.full.source_id == 0:
            raise Failure("WRSource", "coarse producer source is absent")
        if record.split_relative_error > 2.0e-11:
            raise Failure("WRSplit", "coarse A+C producer identity is invalid")
        if not all(
            _all_finite(value, xp)
            for value in (record.A, record.C, record.F)
        ):
            raise Failure("WRSource", "coarse producer is non-finite")
        source_trials.add(int(record.tag.trial))
        if record.transport_projection is None:
            raise Failure("WREvidence", "coarse A projection was discarded")
        for name in ("value", "reaction", "multiplier", "multiplier_residual"):
            if not hasattr(record.transport_projection, name):
                raise Failure(
                    "WREvidence",
                    f"coarse A projection lacks {name}",
                )
        coarse_keys.append(
            _source_key(record.full.source_id, record.tag)
        )

    for sample in window.fine_samples:
        record = sample.record
        if record is None:
            raise Failure("WREvidence", "fine sample has no producer record")
        frame = record.frame
        if record.tag.iteration != version:
            raise Failure("WRVersion", "fine producer version mismatch")
        if record.tag.spectral_context != int(context.spectral_id):
            raise Failure("WRIdentity", "fine spectral context mismatch")
        if int(record.tag.slab) != 0:
            raise Failure("WRIdentity", "unexpected fine slab source")
        if sample.identity.iteration != version or record.identity != sample.identity:
            raise Failure("WRVersion", "fine sample is not the accepted old version")
        if sample.identity.role != "field_rk4":
            raise Failure("WRIdentity", "fine sample does not have field RK4 role")
        if frame.complete_stage:
            raise Failure("WREvidence", "fine field frame claims a complete stage")
        if frame.source_id == 0 or record.tag != frame.tag:
            raise Failure("WRSource", "fine source identity is incomplete")
        if not _array_equal(record.input_P, sample.P, _xp(context)):
            raise Failure("WREvidence", "fine input P does not match sample")
        if not _array_equal(record.input_Z, sample.Z, _xp(context)):
            raise Failure("WREvidence", "fine input Z does not match sample")
        if not _array_equal(record.G, sample.G, _xp(context)):
            raise Failure("WREvidence", "fine G does not match sample")
        if not _array_equal(record.C, sample.C, _xp(context)):
            raise Failure("WREvidence", "fine C does not match sample")
        if frame.ion_em_terms is None:
            raise Failure("WREvidence", "fine EM producer record is absent")
        if frame.ion_em_terms.tag != frame.tag:
            raise Failure("WRIdentity", "fine EM producer tag mismatch")
        if not _array_equal(
            record.feedback.n,
            -_xp(context).asarray(frame.ion.divergence_hat),
            _xp(context),
        ):
            raise Failure("WRSource", "fine density producer was altered")
        if not _array_equal(
            record.feedback.mom,
            forward(frame.ion_em_terms.projected, 3 * context.op.Nc, context),
            _xp(context),
        ):
            raise Failure("WRSource", "fine EM producer was altered")
        source_trials.add(int(record.tag.trial))
        fine_keys.append(_source_key(frame.source_id, record.tag))

    if len(source_trials) != 1:
        raise Failure("WRTrial", "coarse and fine sources do not share a trial")
    if len(set(coarse_keys)) != len(coarse_keys):
        raise Failure("WRSource", "coarse source identities are not unique")
    if len(set(fine_keys)) != len(fine_keys):
        raise Failure("WRSource", "fine source identities are not unique")
    if set(coarse_keys) & set(fine_keys):
        raise Failure("WRSource", "coarse and fine sources alias")
    return coarse_keys, fine_keys


def _validate_boundary(boundary):
    if not isinstance(boundary, BoundaryIntegral):
        raise Failure("WRInterval", "boundary must be a BoundaryIntegral")
    for name in BOUNDARY_FIELDS:
        pair = getattr(boundary, name)
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise Failure("WRInterval", f"boundary {name} must contain two values")
        for index, value in enumerate(pair):
            _finite_float(value, f"boundary {name}[{index}]")
    _finite_float(boundary.field_energy, "boundary field_energy")
    _finite_float(boundary.total_energy, "boundary total_energy")


def _validate_ledger(ledger, species):
    if not isinstance(ledger, LedgerIntegral):
        raise Failure("WRInterval", f"ledger[{species}] has the wrong type")
    if {item.name for item in fields(ledger)} != set(LEDGER_FIELDS):
        raise Failure("WRInterval", f"ledger[{species}] key set mismatch")
    for name in LEDGER_FIELDS:
        _finite_float(getattr(ledger, name), f"ledger[{species}].{name}")


def _validate_interval_shape(interval):
    if not isinstance(interval, MixedInterval):
        raise Failure("WRInterval", "mixed_interval has the wrong type")
    if len(interval.ledger) != 2:
        raise Failure("WRInterval", "mixed interval requires two ledgers")
    for species in range(2):
        _validate_ledger(interval.ledger[species], species)
    _validate_boundary(interval.boundary)
    if len(interval.outward) != 3 or len(interval.projection_reaction) != 3:
        raise Failure("WRInterval", "momentum vectors must have three components")
    for name, values in (
        ("outward", interval.outward),
        ("projection_reaction", interval.projection_reaction),
    ):
        for index, value in enumerate(values):
            _finite_float(value, f"{name}[{index}]")
    for name in (
        "outward_torque",
        "projection_torque",
        "density_work_coarse",
        "density_work_fine",
        "kinetic_path",
    ):
        _finite_float(getattr(interval, name), name)
    for name in ("coarse_sources", "fine_sources"):
        records = getattr(interval, name)
        if not records:
            raise Failure("WRQuadrature", f"{name} may not be empty")
        seen = set()
        for entry in records:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise Failure("WRQuadrature", f"{name} entry has wrong shape")
            key, weight = entry
            if not isinstance(key, tuple) or len(key) != 10:
                raise Failure("WRSource", f"{name} key has wrong shape")
            if key in seen:
                raise Failure("WRSource", f"{name} contains a duplicate source")
            seen.add(key)
            _validate_weight(weight, f"{name} weight")


def validate_mixed_identity(interval):
    """Validate the retained kinetic-path identity at scaled roundoff."""

    _validate_interval_shape(interval)
    ion = interval.ledger[0]
    represented = (
        ion.electric_node
        + ion.pressure_kinetic
        + ion.wall
        + ion.magnetic
        - ion.convection
        + ion.spectral_kinetic
    )
    residual = interval.kinetic_path - represented
    scale = max(
        abs(interval.kinetic_path),
        abs(ion.electric_node),
        abs(ion.pressure_kinetic),
        abs(ion.wall),
        abs(ion.magnetic),
        abs(ion.convection),
        abs(ion.spectral_kinetic),
        1.0e-300,
    )
    tolerance = 32768.0 * sys.float_info.epsilon * scale
    if not math.isfinite(residual) or abs(residual) > tolerance:
        raise Failure(
            "WRLedger",
            "kinetic_path does not match the assembled ion ledger "
            f"(residual={residual:.17g}, tolerance={tolerance:.17g})",
        )
    return interval


def _source_weights_match(interval, window):
    H = float(window.H)
    for name, records in (
        ("coarse", interval.coarse_sources),
        ("fine", interval.fine_sources),
    ):
        total = math.fsum(weight for _, weight in records)
        limit = 128.0 * sys.float_info.epsilon * max(abs(H), 1.0e-300)
        if abs(total - H) > limit:
            raise Failure(
                "WRQuadrature",
                f"{name} weights do not independently span H",
            )


def _clock_tolerance(parameters, scheduled):
    return (
        16.0
        * sys.float_info.epsilon
        * max(
            abs(float(scheduled)),
            abs(float(parameters.start_time)),
            abs(float(parameters.dt)),
            1.0e-300,
        )
    )


def _validate_scheduled_clock(actual, parameters, step, name):
    scheduled = core.time_at_step(parameters, int(step))
    value = float(actual)
    tolerance = _clock_tolerance(parameters, scheduled)
    if not math.isfinite(value) or abs(value - scheduled) > tolerance:
        raise Failure(
            "WRClock",
            f"{name} scheduler clock mismatch: "
            f"actual={value.hex()}, scheduled={scheduled.hex()}, "
            f"tolerance={tolerance.hex()}",
        )
    return scheduled


def validate_scheduler_identity(interval, context, left, right):
    """Validate macro endpoints against the scheduler's absolute clock."""

    _validate_interval_shape(interval)
    if interval.right_step != interval.left_step + 1:
        raise Failure("WRClock", "mixed interval must advance exactly one macro")
    if int(left.step) != interval.left_step or int(right.step) != interval.right_step:
        raise Failure("WRClock", "mixed interval step identities do not match")
    scheduled_left = _validate_scheduled_clock(
        interval.left_time,
        context.p,
        interval.left_step,
        "interval left",
    )
    scheduled_right = _validate_scheduled_clock(
        interval.right_time,
        context.p,
        interval.right_step,
        "interval right",
    )
    _validate_scheduled_clock(
        left.time,
        context.p,
        interval.left_step,
        "left state",
    )
    _validate_scheduled_clock(
        right.time,
        context.p,
        interval.right_step,
        "right endpoint",
    )
    return scheduled_left, scheduled_right


def _validate_endpoint_values(window, endpoint, context):
    xp = _xp(context)
    state = getattr(endpoint, "canonical_state", endpoint)
    modes = 2 * context.p.K + 1
    nc = context.op.Nc
    nb = context.op.Nb
    expected_p = xp.asarray(window.P_endpoint)
    expected_z = xp.asarray(window.Z_endpoint)
    actual_p = xp.concatenate(
        (
            xp.asarray(state.fluid[0].n).reshape(modes, nc),
            xp.asarray(state.fluid[0].mom).reshape(modes, 3 * nc),
            xp.asarray(state.fluid[0].perp).reshape(modes, nc),
        ),
        axis=1,
    )
    actual_z = xp.concatenate(
        (
            xp.asarray(state.fluid[1].perp).reshape(modes, nc),
            xp.asarray(state.fluid[1].parallel).reshape(modes, nc),
            xp.asarray(state.B).reshape(modes, nb),
        ),
        axis=1,
    )
    if actual_p.shape != expected_p.shape or actual_z.shape != expected_z.shape:
        raise Failure("WREndpoint", "endpoint packed shape mismatch")
    p_scale = xp.maximum(
        xp.maximum(xp.abs(expected_p), xp.abs(actual_p)),
        1.0e-300,
    )
    z_scale = xp.maximum(
        xp.maximum(xp.abs(expected_z), xp.abs(actual_z)),
        1.0e-300,
    )
    p_error = float(xp.max(xp.abs(actual_p - expected_p) / p_scale).item())
    z_error = float(xp.max(xp.abs(actual_z - expected_z) / z_scale).item())
    if p_error > 2.0e-12 or z_error > 2.0e-12:
        raise Failure(
            "WREndpoint",
            f"endpoint values do not reproduce accepted P/Z ({p_error}, {z_error})",
        )


def validate_mixed_interval(
    interval,
    window,
    left_state,
    endpoint,
    context,
):
    """Validate one frozen interval against its accepted final-sweep sources."""

    _validate_interval_shape(interval)
    validate_mixed_identity(interval)
    if float(window.H) != float(context.p.dt):
        raise Failure(
            "WRClock",
            "window H does not match the declared context H",
        )
    coarse_keys, fine_keys = _validate_source_records(window, left_state, context)
    version = int(window.final_input_version)
    expected_trial = int(window.coarse_samples[0].record.tag.trial)
    if interval.trial != expected_trial:
        raise Failure("WRTrial", "mixed interval source trial mismatch")
    if interval.node_version != version:
        raise Failure("WRVersion", "mixed interval accepted node version mismatch")
    end_tag = endpoint.tag
    validate_scheduler_identity(interval, context, left_state, end_tag)
    if endpoint.source_id == 0:
        raise Failure("WRSource", "endpoint source is absent")
    if int(end_tag.trial) != expected_trial:
        raise Failure("WRTrial", "endpoint trial identity mismatch")
    _validate_endpoint_values(window, endpoint, context)
    H = float(window.H)
    expected_coarse = tuple(
        (key, float(H * window.Q[-1, node]))
        for node, key in enumerate(coarse_keys)
    )
    expected_fine = tuple(
        (key, float(sample.weight))
        for key, sample in zip(fine_keys, window.fine_samples)
    )
    if interval.coarse_sources != expected_coarse:
        raise Failure("WRSource", "coarse source metadata was altered")
    if interval.fine_sources != expected_fine:
        raise Failure("WRSource", "fine source metadata was altered")
    _source_weights_match(interval, window)
    return interval


def _boundary_summands(state, stage, context):
    traces = stage.boundary._trace_data if hasattr(stage, "boundary") else massless_checks._wall_trace_data(state, stage, context)
    xp = _xp(context)
    nc = context.op.Nc
    ne = context.op.Ne
    number = [0.0, 0.0]
    kinetic = [0.0, 0.0]
    internal_perp = [0.0, 0.0]
    internal_parallel = [0.0, 0.0]
    energy = [0.0, 0.0]
    for sample in range(context.N):
        for data in traces:
            wall = data.wall
            edge = data.edge
            direction = data.direction
            area = data.area
            for species in range(2):
                fluid = stage.fluid[species]
                density = float(data.density[species][sample].item())
                normal_flux = wall.sign * float(
                    xp.real(xp.asarray(fluid.flux)).reshape(
                        context.N, ne
                    )[sample, edge]
                )
                perp = float(data.perp[species][sample].item())
                parallel = float(data.parallel[species][sample].item())
                velocity = data.velocity[species][sample]
                un = wall.sign * float(velocity[direction].real)
                kinetic_density = (
                    0.5
                    * context.p.species[species].mass
                    * density
                    * float(xp.sum(xp.abs(velocity) ** 2).item())
                    if species == 0
                    else 0.0
                )
                number[species] += area * normal_flux
                kinetic[species] += area * kinetic_density * un

                pressure_flux_available = (
                    hasattr(fluid, "pressure_flux")
                    and fluid.pressure_flux is not None
                )
                if pressure_flux_available:
                    pressure_flux = wall.sign * float(
                        xp.real(xp.asarray(fluid.pressure_flux)).reshape(
                            context.N, ne
                        )[sample, edge]
                    )
                    internal_perp[species] += area * pressure_flux * (
                        1.0
                        if species
                        else 1.0 / (context.p.species[0].gamma - 1.0)
                    )
                if (
                    species
                    and hasattr(fluid, "parallel_flux")
                    and fluid.parallel_flux is not None
                ):
                    parallel_flux = wall.sign * float(
                        xp.real(xp.asarray(fluid.parallel_flux)).reshape(
                            context.N, ne
                        )[sample, edge]
                    )
                    internal_parallel[species] += area * 0.5 * parallel_flux

                internal_density = (
                    perp + 0.5 * parallel
                    if species
                    else perp / (context.p.species[0].gamma - 1.0)
                )
                pressure_work = perp * un
                if species:
                    direction_b = float(
                        xp.real(data.B[direction][sample]).item()
                    )
                    parallel_b = float(
                        sum(
                            data.B[d][sample] * velocity[d]
                            for d in range(3)
                        ).real.item()
                    )
                    bmag = float(
                        xp.sqrt(
                            xp.sum(xp.stack(data.B)[:, sample] ** 2)
                        ).item()
                    )
                    pressure_work += (
                        (parallel - perp)
                        * parallel_b
                        * wall.sign
                        * direction_b
                        / bmag
                    )
                energy[species] += area * (
                    (kinetic_density + internal_density) * un + pressure_work
                )
    return _BoundarySummands(
        number=(number[0], number[1]),
        kinetic=(kinetic[0], kinetic[1]),
        internal_perp=(internal_perp[0], internal_perp[1]),
        internal_parallel=(internal_parallel[0], internal_parallel[1]),
        energy=(energy[0], energy[1]),
        field_energy=0.0,
    )


def _traction_components(data, sample, context):
    xp = _xp(context)
    direction = data.direction
    sigma = data.sigma
    b = [float(xp.real(component[sample]).item()) for component in data.B]
    b2 = sum(component * component for component in b)
    bmag = math.sqrt(b2)
    b_normal = b[direction]
    direction_b = b_normal / bmag
    velocity = data.velocity[0][sample]
    un = sigma * float(velocity[direction].real)
    density = float(data.density[0][sample].item())
    ion_perp = float(data.perp[0][sample].item())
    electron_perp = float(data.perp[1][sample].item())
    electron_parallel = float(data.parallel[1][sample].item())
    transport = []
    electromagnetic = []
    for component in range(3):
        unit = 1.0 if component == direction else 0.0
        magnetic = sigma * (
            0.5 * b2 * unit - b[component] * b_normal
        ) / context.p.mu0
        ion_ram = (
            context.p.species[0].mass
            * density
            * float(velocity[component].real)
            * un
        )
        ion_pressure = ion_perp * sigma * unit
        electron_pressure = (
            electron_perp * sigma * unit
            + (electron_parallel - electron_perp)
            * (b[component] / bmag)
            * direction_b
            * sigma
        )
        transport.append(ion_ram + ion_pressure)
        electromagnetic.append(magnetic + electron_pressure)
    return transport, electromagnetic


def _momentum_integrals(
    state,
    stage,
    reaction,
    context,
    *,
    traction_role,
):
    xp = _xp(context)
    nc = context.op.Nc
    if traction_role not in ("transport", "electromagnetic"):
        raise Failure(
            "WRTraction",
            "traction_role must be 'transport' or 'electromagnetic'",
        )
    m0 = xp.real(xp.asarray(context.op.M0)).astype(xp.float64)
    cell_radius = xp.asarray(context.geometry.cells.R).reshape(nc)
    traces = massless_checks._wall_trace_data(state, stage, context)
    samples = max(context.N, 3)
    outward = np.zeros(3)
    outward_torque = 0.0
    projection = np.zeros(3)
    projection_torque = 0.0
    reaction_samples = (
        xp.asarray(reaction).reshape(context.N, 3, nc)
        if reaction is not None
        else None
    )
    for q in range(samples):
        sample = 0 if context.N == 1 else q
        angle = 2.0 * math.pi * q / samples
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        weights = 2.0 * math.pi * m0 / samples
        if reaction_samples is not None:
            source = reaction_samples[sample]
            cartesian = (
                source[0] * cos_angle - source[1] * sin_angle,
                source[0] * sin_angle + source[1] * cos_angle,
                source[2],
            )
            for component in range(3):
                projection[component] += _physical_real(
                    xp.sum(weights * cartesian[component]),
                    "mixed projection reaction",
                )
            projection_torque += _physical_real(
                xp.sum(weights * cell_radius * source[1]),
                "mixed projection torque",
            )
        for data in traces:
            transport, electromagnetic = _traction_components(
                data, sample, context
            )
            components = (
                transport
                if traction_role == "transport"
                else electromagnetic
            )
            wall_weight = 2.0 * math.pi * data.wall.area_per_radian / samples
            surface_radius = (
                float(context.geometry.r_faces.R[data.wall.face])
                if data.wall.component == "r_face"
                else float(context.geometry.z_faces.R[data.wall.face])
            )
            cartesian = (
                components[0] * cos_angle - components[1] * sin_angle,
                components[0] * sin_angle + components[1] * cos_angle,
                components[2],
            )
            for component in range(3):
                outward[component] += wall_weight * cartesian[component]
            outward_torque += (
                wall_weight * surface_radius * components[1]
            )
    return (
        tuple(float(value) for value in outward),
        float(outward_torque),
        tuple(float(value) for value in projection),
        float(projection_torque),
    )


def _fine_view(sample):
    frame = sample.record.frame
    return FineStageView(
        kind=frame.kind,
        complete_stage=False,
        diagnostic_scope=frame.diagnostic_scope,
        state=frame.trial_state,
        fluid=(frame.ion, frame.electron),
        cache=frame.cache,
        current=frame.current,
        rhs=FineFieldRhsView(
            B=frame.field_rates.B,
            fluid=(
                None,
                SimpleNamespace(
                    perp=frame.field_rates.electron_perp,
                    parallel=frame.field_rates.electron_parallel,
                ),
            ),
        ),
        tag=frame.tag,
        source_id=frame.source_id,
        checks=tuple(frame.checks),
        frame=frame,
    )


def accumulate_mixed_interval(window, left_state, endpoint, context) -> MixedIntervalBundle:
    """Freeze the selected final old-version sweep into mixed interval evidence."""

    coarse_keys, fine_keys = _validate_source_records(window, left_state, context)
    xp = _xp(context)
    nc = context.op.Nc
    ne = context.op.Ne
    H = _finite_float(window.H, "window H")
    if H <= 0.0:
        raise Failure("WRInterval", "window H must be positive")
    coarse_weights = [H * float(window.Q[-1, node]) for node in range(window.s)]
    fine_weights = [
        _validate_weight(sample.weight, "fine sample weight")
        for sample in window.fine_samples
    ]
    for name, weights in (
        ("coarse", coarse_weights),
        ("fine", fine_weights),
    ):
        if any(not math.isfinite(weight) or weight <= 0.0 for weight in weights):
            raise Failure("WRQuadrature", f"{name} weights are invalid")
        total = math.fsum(weights)
        limit = 128.0 * sys.float_info.epsilon * max(H, 1.0e-300)
        if abs(total - H) > limit:
            raise Failure("WRQuadrature", f"{name} weights do not span H")

    density_coarse = 0.0
    pressure_kinetic = 0.0
    pressure_internal = 0.0
    legacy_convection = 0.0
    wall_transport = 0.0
    spectral_A = 0.0
    spectral_p = 0.0
    kinetic_A = 0.0
    coarse_traction = np.zeros(3)
    coarse_traction_torque = 0.0
    coarse_projection = np.zeros(3)
    coarse_projection_torque = 0.0
    coarse_boundary_values = []
    momentum_reference = (
        context.p.species[0].mass
        * context.p.ref.ni
        * context.p.ref.velocity
        / context.p.ref.time
    )
    for sample, weight in zip(window.coarse_samples, coarse_weights):
        record = sample.record
        full = record.full
        ion = record.ion_frame
        u = xp.asarray(ion.u).reshape(context.N, 3, nc)
        transport_force = (
            -xp.asarray(ion.convection) + xp.asarray(ion.pressure_force)
        )
        reaction = xp.asarray(record.transport_projection.reaction).reshape(
            context.N, 3, nc
        )
        a_mode = xp.asarray(record.A[:, nc : 4 * nc]).reshape(-1)
        a_samples = inverse(
            a_mode,
            3 * nc,
            context,
            True,
            momentum_reference,
        ).reshape(context.N, 3, nc)
        density_sampled = -xp.asarray(ion.divergence).reshape(context.N, nc)
        speed = 0.5 * context.p.species[0].mass * xp.sum(
            xp.abs(u) ** 2, axis=1
        )
        density_value = _physical_real(
            _integrate_cells(
                (speed * density_sampled).reshape(-1), 1, context
            ),
            "coarse density work",
        )
        wall_value = _physical_real(
            _integrate_cells(
                (xp.conj(u) * reaction).reshape(-1), 3, context
            ),
            "coarse A projection wall work",
        )
        spectral_a_value = _physical_real(
            _integrate_cells(
                (
                    xp.conj(u)
                    * (a_samples - xp.asarray(record.transport_projection.value).reshape(context.N, 3, nc))
                ).reshape(-1),
                3,
                context,
            ),
            "coarse A spectral kinetic transfer",
        )
        a_pressure_samples = inverse(
            xp.asarray(full.rhs.fluid[0].perp),
            nc,
            context,
            True,
            context.p.ref.pressure / context.p.ref.time,
        ).reshape(context.N, nc)
        spectral_p_value = _physical_real(
            _integrate_cells(
                (
                    a_pressure_samples
                    - xp.asarray(ion.pdot_perp).reshape(context.N, nc)
                ).reshape(-1)
                / (context.p.species[0].gamma - 1.0),
                1,
                context,
            ),
            "coarse ion pressure spectral transfer",
        )
        kinetic_a_value = _physical_real(
            _integrate_cells(
                (xp.conj(u) * a_samples).reshape(-1),
                3,
                context,
            ),
            "coarse A kinetic path",
        )
        density_coarse += weight * density_value
        pressure_kinetic += weight * float(full.ledger[0].pressure_kinetic)
        pressure_internal += weight * float(full.ledger[0].pressure_internal)
        legacy_convection += weight * float(full.ledger[0].convection)
        wall_transport += weight * wall_value
        spectral_A += weight * spectral_a_value
        spectral_p += weight * spectral_p_value
        kinetic_A += weight * kinetic_a_value
        outward, torque, projection, projection_torque = _momentum_integrals(
            full.canonical_state,
            full,
            record.transport_projection.reaction,
            context,
            traction_role="transport",
        )
        coarse_traction += weight * np.asarray(outward)
        coarse_traction_torque += weight * torque
        coarse_projection += weight * np.asarray(projection)
        coarse_projection_torque += weight * projection_torque
        coarse_boundary_values.append(
            (
                weight,
                _boundary_summands(full.canonical_state, full, context),
            )
        )

    fine_views = tuple(_fine_view(sample) for sample in window.fine_samples)
    fine_boundary_values = []
    fine_electron_ledgers = []
    diagnostic_work = MixedDiagnosticWork()
    density_fine = 0.0
    wall_fine = 0.0
    spectral_C = 0.0
    kinetic_C = 0.0
    electric_node = 0.0
    electric_edge = 0.0
    magnetic = 0.0
    fine_traction = np.zeros(3)
    fine_traction_torque = 0.0
    fine_projection = np.zeros(3)
    fine_projection_torque = 0.0
    density_reference = context.p.ref.ni / context.p.ref.time
    for sample, weight, view in zip(
        window.fine_samples, fine_weights, fine_views
    ):
        frame = view.frame
        terms = frame.ion_em_terms
        ion = frame.ion
        u = xp.asarray(ion.u).reshape(context.N, 3, nc)
        projected = xp.asarray(terms.projected).reshape(context.N, 3, nc)
        reaction = xp.asarray(terms.reaction).reshape(context.N, 3, nc)
        c_mode = forward(terms.projected, 3 * nc, context)
        c_samples = inverse(
            c_mode,
            3 * nc,
            context,
            True,
            momentum_reference,
        ).reshape(context.N, 3, nc)
        density_sampled = -xp.asarray(ion.divergence).reshape(context.N, nc)
        density_retained = inverse(
            -xp.asarray(ion.divergence_hat),
            nc,
            context,
            True,
            density_reference,
        ).reshape(context.N, nc)
        speed = 0.5 * context.p.species[0].mass * xp.sum(
            xp.abs(u) ** 2, axis=1
        )
        density_fine_value = _physical_real(
            _integrate_cells(
                (speed * density_sampled).reshape(-1), 1, context
            ),
            "fine density work",
        )
        electric_node_value = _physical_real(
            _integrate_cells(
                (xp.conj(u) * xp.asarray(terms.electric).reshape(context.N, 3, nc)).reshape(-1),
                3,
                context,
            ),
            "fine electric node work",
        )
        edge_electric = (
            context.p.species[0].charge
            * xp.conj(xp.asarray(ion.flux).reshape(context.N, ne))
            * inverse(
                frame.cache.E,
                ne,
                context,
                True,
                context.p.ref.electric,
            ).reshape(context.N, ne)
        )
        electric_edge_value = _physical_real(
            _integrate_edges(edge_electric.reshape(-1), context),
            "fine electric edge work",
        )
        magnetic_value = _physical_real(
            _integrate_cells(
                (
                    xp.conj(u)
                    * xp.asarray(terms.magnetic).reshape(context.N, 3, nc)
                ).reshape(-1),
                3,
                context,
            ),
            "fine magnetic work",
        )
        wall_value = _physical_real(
            _integrate_cells(
                (xp.conj(u) * reaction).reshape(-1),
                3,
                context,
            ),
            "fine C projection wall work",
        )
        spectral_c_value = _physical_real(
            _integrate_cells(
                (
                    xp.conj(u) * (c_samples - projected)
                ).reshape(-1),
                3,
                context,
            )
            - _integrate_cells(
                (speed * (density_retained - density_sampled)).reshape(-1),
                1,
                context,
            ),
            "fine C spectral kinetic transfer",
        )
        kinetic_c_value = _physical_real(
            _integrate_cells(
                (xp.conj(u) * c_samples).reshape(-1),
                3,
                context,
            )
            - _integrate_cells(
                (speed * density_retained).reshape(-1),
                1,
                context,
            ),
            "fine C kinetic path",
        )
        density_fine += weight * density_fine_value
        electric_node += weight * electric_node_value
        electric_edge += weight * electric_edge_value
        magnetic += weight * magnetic_value
        wall_fine += weight * wall_value
        spectral_C += weight * spectral_c_value
        kinetic_C += weight * kinetic_c_value
        fine_boundary_values.append(
            (weight, _boundary_summands(view.state, view, context))
        )
        diagnostic_work.electron_ledger_calls += 1
        diagnostic_work.electron_pressure_only_projections += 1
        diagnostic_work.electron_ledger_inverse_transforms += 4
        fine_electron_ledgers.append(
            (weight, massless_checks._electron_ledger(view, context))
        )
        outward, torque, projection, projection_torque = _momentum_integrals(
            view.state,
            view,
            terms.reaction,
            context,
            traction_role="electromagnetic",
        )
        fine_traction += weight * np.asarray(outward)
        fine_traction_torque += weight * torque
        fine_projection += weight * np.asarray(projection)
        fine_projection_torque += weight * projection_torque

    delta_density = density_coarse - density_fine
    ion_ledger = LedgerIntegral(
        electric_node=electric_node,
        electric_edge=electric_edge,
        pressure_kinetic=pressure_kinetic,
        pressure_internal=pressure_internal,
        magnetic=magnetic,
        wall=wall_transport + wall_fine,
        convection=legacy_convection - delta_density,
        perp_rate=pressure_internal,
        parallel_rate=0.0,
        spectral_kinetic=spectral_A + spectral_C,
        spectral_perp=spectral_p,
        spectral_parallel=0.0,
    )
    electron_values = {
        name: math.fsum(
            weight * float(getattr(ledger, name))
            for weight, ledger in fine_electron_ledgers
        )
        for name in LEDGER_FIELDS
    }
    electron_ledger = LedgerIntegral(**electron_values)

    boundary = BoundaryIntegral(
        number=(
            math.fsum(
                weight * terms.number[0]
                for weight, terms in fine_boundary_values
            ),
            math.fsum(
                weight * terms.number[1]
                for weight, terms in fine_boundary_values
            ),
        ),
        kinetic=(
            math.fsum(
                weight * terms.kinetic[0]
                for weight, terms in coarse_boundary_values
            ),
            0.0,
        ),
        internal_perp=(
            math.fsum(
                weight * terms.internal_perp[0]
                for weight, terms in coarse_boundary_values
            ),
            math.fsum(
                weight * terms.internal_perp[1]
                for weight, terms in fine_boundary_values
            ),
        ),
        internal_parallel=(
            0.0,
            math.fsum(
                weight * terms.internal_parallel[1]
                for weight, terms in fine_boundary_values
            ),
        ),
        field_energy=math.fsum(
            weight * terms.field_energy
            for weight, terms in fine_boundary_values
        ),
        total_energy=(
            math.fsum(
                weight * terms.energy[0]
                for weight, terms in coarse_boundary_values
            )
            + math.fsum(
                weight * (terms.energy[1] + terms.field_energy)
                for weight, terms in fine_boundary_values
            )
        ),
    )
    interval = MixedInterval(
        trial=int(window.coarse_samples[0].record.tag.trial),
        node_version=int(window.final_input_version),
        left_step=int(left_state.step),
        right_step=int(endpoint.tag.step),
        left_time=float(left_state.time),
        right_time=float(endpoint.tag.time),
        ledger=(ion_ledger, electron_ledger),
        boundary=boundary,
        outward=tuple(float(value) for value in coarse_traction + fine_traction),
        outward_torque=float(coarse_traction_torque + fine_traction_torque),
        projection_reaction=tuple(
            float(value) for value in coarse_projection + fine_projection
        ),
        projection_torque=float(
            coarse_projection_torque + fine_projection_torque
        ),
        density_work_coarse=float(density_coarse),
        density_work_fine=float(density_fine),
        kinetic_path=float(kinetic_A + kinetic_C),
        coarse_sources=tuple(
            (key, float(weight))
            for key, weight in zip(coarse_keys, coarse_weights)
        ),
        fine_sources=tuple(
            (key, float(weight))
            for key, weight in zip(fine_keys, fine_weights)
        ),
    )
    validate_mixed_interval(interval, window, left_state, endpoint, context)
    return MixedIntervalBundle(
        interval=interval,
        fine_views=fine_views,
        diagnostic_work=diagnostic_work,
    )
