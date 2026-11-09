"""Massless quasineutral physics adapter for the selected A1-WR window.

The adapter packs independent ion ``P=(n_i,m_i*n_i*u_i,p_i)`` and
electron/field ``Z=(p_e_perp,p_e_parallel,B)`` variables in retained
Fourier-mode rows. Fine stages evaluate only ``massless_field.evaluate`` and
``feedback_rate``. Coarse stages evaluate one complete physical stage and
decompose its ion RHS into transport ``A`` and grid response ``C``.
"""

from __future__ import annotations

import copy
import hashlib
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import massless, massless_field, stage as stage_module
from . import time_multirate_wr
from .core import forward
from .time_multirate_wr import WindowIdentity, WindowResult
from .time_multirate_wr import solve_window as _solve_window
from .types import Failure, FluidState, StageTag, State
from .wall import project_wall


UINT64_MAX = (1 << 64) - 1


def _xp(context):
    return context.compute.xp


def _fresh(context, name):
    current = int(getattr(context, name))
    if current >= UINT64_MAX:
        raise Failure("CounterOverflow", f"{name} identity exhausted")
    setattr(context, name, current + 1)
    return current


def _layout(context):
    return 2 * context.p.K + 1, context.op.Nc, context.op.Nb


def _require_packed(value, columns, name, context):
    xp = _xp(context)
    modes, _, _ = _layout(context)
    if not isinstance(value, xp.ndarray):
        raise Failure("Layout", f"{name} must be a {xp.__name__} array")
    if value.dtype != xp.complex128:
        raise Failure("Layout", f"{name} must have complex128 dtype")
    if value.shape != (modes, columns):
        raise Failure(
            "Layout",
            f"{name} shape {value.shape} does not match {(modes, columns)}",
        )
    if not bool(xp.all(xp.isfinite(value)).item()):
        raise Failure("Finite", f"{name} contains non-finite values")


def _validate_packed(p, z, context):
    modes, nc, nb = _layout(context)
    _require_packed(p, 5 * nc, "P", context)
    _require_packed(z, 2 * nc + nb, "Z", context)
    return modes, nc, nb


def physical_scales(context):
    """Return positive P/Z reference scales in the retained-mode layout."""

    modes, nc, nb = _layout(context)
    xp = _xp(context)
    ion = context.p.species[0]
    density = float(context.p.ref.ni)
    momentum = float(ion.mass) * density * float(context.p.ref.velocity)
    pressure = float(context.p.ref.pressure)
    magnetic = float(context.p.ref.magnetic)
    values = (density, momentum, pressure, magnetic)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise Failure("Scales", "physical reference scales must be positive")
    p_components = xp.concatenate(
        (
            xp.full((nc,), density, dtype=xp.float64),
            xp.full((3 * nc,), momentum, dtype=xp.float64),
            xp.full((nc,), pressure, dtype=xp.float64),
        )
    )
    z_components = xp.concatenate(
        (
            xp.full((2 * nc,), pressure, dtype=xp.float64),
            xp.full((nb,), magnetic, dtype=xp.float64),
        )
    )
    return (
        xp.ascontiguousarray(xp.tile(p_components, (modes, 1))),
        xp.ascontiguousarray(xp.tile(z_components, (modes, 1))),
    )


def state_from_packed(
    context,
    p,
    z,
    *,
    step: int,
    generation: int,
    time: float,
) -> State:
    """Build a raw State without canonicalization or ion transport."""

    modes, nc, nb = _validate_packed(p, z, context)
    xp = _xp(context)
    charge_ratio = float(context.p.species[0].charge) / (
        -float(context.p.species[1].charge)
    )
    ion_n = xp.ascontiguousarray(p[:, :nc]).reshape(-1).copy()
    ion_momentum = xp.ascontiguousarray(p[:, nc : 4 * nc]).reshape(-1).copy()
    ion_pressure = xp.ascontiguousarray(p[:, 4 * nc :]).reshape(-1).copy()
    electron_density = xp.asarray(charge_ratio * ion_n).copy()
    electron_momentum = xp.zeros(modes * 3 * nc, dtype=xp.complex128)
    electron_perp = xp.ascontiguousarray(z[:, :nc]).reshape(-1).copy()
    electron_parallel = xp.ascontiguousarray(z[:, nc : 2 * nc]).reshape(-1).copy()
    magnetic = xp.ascontiguousarray(z[:, 2 * nc :]).reshape(-1).copy()
    ion = FluidState(
        n=ion_n,
        mom=ion_momentum,
        perp=ion_pressure,
        parallel=xp.empty(0, dtype=xp.complex128),
    )
    electron = FluidState(
        n=electron_density,
        mom=electron_momentum,
        perp=electron_perp,
        parallel=electron_parallel,
    )
    return State(
        fluid=(ion, electron),
        B=magnetic,
        step=int(step),
        generation=int(generation),
        time=float(time),
    )


def pack_state(context, state):
    """Pack one raw state into the independent P/Z mode-row arrays."""

    modes, nc, nb = _layout(context)
    xp = _xp(context)
    ion = state.fluid[0]
    electron = state.fluid[1]
    expected = (
        (ion.n, modes * nc),
        (ion.mom, modes * 3 * nc),
        (ion.perp, modes * nc),
        (electron.perp, modes * nc),
        (electron.parallel, modes * nc),
        (state.B, modes * nb),
    )
    for value, size in expected:
        if value is None or tuple(value.shape) != (size,):
            raise Failure("Layout", "state cannot be packed into P/Z")
    p = xp.concatenate(
        (
            ion.n.reshape(modes, nc),
            ion.mom.reshape(modes, 3 * nc),
            ion.perp.reshape(modes, nc),
        ),
        axis=1,
    )
    z = xp.concatenate(
        (
            electron.perp.reshape(modes, nc),
            electron.parallel.reshape(modes, nc),
            state.B.reshape(modes, nb),
        ),
        axis=1,
    )
    return xp.ascontiguousarray(p), xp.ascontiguousarray(z)


def pack_stage_rhs(stage, species, context):
    """Pack one species' complete RHS in the P/Z component layout."""

    if species not in (0, 1):
        raise Failure("Species", "species index must be zero or one")
    modes, nc, nb = _layout(context)
    xp = _xp(context)
    fluid = stage.rhs.fluid[species]
    expected = (
        (fluid.n, modes * nc),
        (fluid.mom, modes * 3 * nc),
        (fluid.perp, modes * nc),
    )
    for value, size in expected:
        if value is None or tuple(value.shape) != (size,):
            raise Failure("Layout", f"species {species} RHS cannot be packed")
    if species == 0:
        return xp.ascontiguousarray(
            xp.concatenate(
                (
                    fluid.n.reshape(modes, nc),
                    fluid.mom.reshape(modes, 3 * nc),
                    fluid.perp.reshape(modes, nc),
                ),
                axis=1,
            )
        )
    if tuple(fluid.parallel.shape) != (modes * nc,):
        raise Failure("Layout", "electron pressure RHS cannot be packed")
    if tuple(stage.rhs.B.shape) != (modes * nb,):
        raise Failure("Layout", "magnetic RHS cannot be packed")
    return (
        xp.ascontiguousarray(
            xp.concatenate(
                (
                    fluid.n.reshape(modes, nc),
                    fluid.mom.reshape(modes, 3 * nc),
                    fluid.perp.reshape(modes, nc),
                ),
                axis=1,
            )
        ),
        xp.ascontiguousarray(
            xp.concatenate(
                (
                    fluid.perp.reshape(modes, nc),
                    fluid.parallel.reshape(modes, nc),
                    stage.rhs.B.reshape(modes, nb),
                ),
                axis=1,
            )
        ),
    )


@dataclass(slots=True)
class WorkCounts:
    full_stage_attempts: int = 0
    full_stage_calls: int = 0
    full_stage_failures: int = 0
    full_ion_power_ledger_values: int = 0
    full_electron_power_ledger_values: int = 0
    coarse_transport_calls: int = 0
    coarse_transport_wall_projections: int = 0
    coarse_grid_wall_projections: int = 0
    field_callback_calls: int = 0
    field_evaluations: int = 0
    alias_field_evaluations: int = 0
    feedback_calls: int = 0
    electromagnetic_rates: int = 0
    wall_projections: int = 0
    fine_full_stage_calls: int = 0
    split_validations: int = 0
    exchange_diagnostics: int = 0
    endpoint_reconstruction_calls: int = 0
    endpoint_full_stage_calls: int = 0
    fine_ion_power_ledger_calls: int = 0

    def snapshot(self):
        result = asdict(self)
        result["full_mixed_ledger_calls"] = 0
        return result


@dataclass(slots=True)
class FieldRecord:
    identity: WindowIdentity
    tag: StageTag
    input_P: Any
    input_Z: Any
    G: Any
    C: Any
    frame: massless_field.FieldFrame
    feedback: massless_field.IonRate
    alias_field_evaluations: int
    cache_object_id: int
    cache_key: tuple[Any, ...]
    cache_storage_bytes: int
    residual_max: dict[str, float]
    electric_exchange_W: complex | None
    electric_exchange_error: str | None = None


@dataclass(slots=True)
class CoarseRecord:
    identity: WindowIdentity
    tag: StageTag
    A: Any
    C: Any
    F: Any
    split_relative_error: float
    frame: Any
    full: Any
    ion_frame: Any
    transport_projection: Any


@dataclass(slots=True)
class EndpointRecord:
    tag: StageTag
    full: Any
    P: Any
    Z: Any
    totals: Any
    checks: tuple[Any, ...]
    production_accepted: bool = False
    trial_endpoint: bool = True
    interval_transaction_required: bool = True


@dataclass
class WindowPhysicsResult:
    window: WindowResult
    endpoint: EndpointRecord
    coarse_records: tuple[CoarseRecord, ...]
    fine_records: tuple[FieldRecord, ...]
    all_coarse_records: tuple[CoarseRecord, ...]
    all_fine_records: tuple[FieldRecord, ...]
    work: WorkCounts
    source: dict[str, Any]
    clock: dict[str, int]
    version_evidence: dict[str, Any]
    identity_evidence: dict[str, Any]
    caller_unchanged: bool
    ledger_unchanged: bool
    scope: str = (
        "massless A1-WR trial and endpoint reconstruction; "
        "not a production accepted transaction"
    )


def _hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _version_evidence():
    paths = {
        "src/fomhd/multirate_wr_physics.py": Path(__file__),
        "src/fomhd/massless_field.py": Path(massless_field.__file__),
        "src/fomhd/massless.py": Path(massless.__file__),
        "src/fomhd/time_multirate_wr.py": Path(time_multirate_wr.__file__),
    }
    return {
        "python": sys.version.split()[0],
        "source_hashes": {
            name: _hash_file(path) for name, path in paths.items()
        },
    }


def _residual_summary(frame, context):
    xp = _xp(context)
    summary = {}
    for name, value in frame.residuals.items():
        if value is None or getattr(value, "size", 0) == 0:
            summary[name] = 0.0
        else:
            summary[name] = float(xp.max(xp.abs(value)).item())
    return summary


def _electric_exchange(frame, context):
    """Return ion electric exchange from the counted padded producer record."""

    xp = _xp(context)
    nc = context.op.Nc
    terms = getattr(frame, "ion_em_terms", None)
    if terms is None:
        raise Failure(
            "FieldProducer",
            "feedback_rate did not retain its ion electric-force producer",
        )
    velocity = xp.asarray(frame.ion.u).reshape(context.N, 3, nc)
    weights = xp.real(xp.asarray(context.op.M0)).astype(xp.float64)
    value = xp.sum(
        xp.conj(velocity)
        * xp.asarray(terms.electric).reshape(context.N, 3, nc)
        * weights[None, None, :]
    )
    value = value * (2.0 * math.pi / context.N)
    return complex(value.item())


def _snapshot_equal(left, right, xp):
    if type(left) is not type(right):
        return False
    if isinstance(left, xp.ndarray):
        return bool(
            left.dtype == right.dtype
            and left.shape == right.shape
            and xp.array_equal(left, right)
        )
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _snapshot_equal(left[key], right[key], xp) for key in left
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _snapshot_equal(a, b, xp) for a, b in zip(left, right)
        )
    return left == right


class MasslessWRPhysicsAdapter:
    """Context-owned callback adapter for one accepted-left WR trial."""

    def __init__(
        self,
        context,
        base_state,
        *,
        slab: int = 0,
        trial_id: int | None = None,
    ):
        if int(getattr(context.p, "electron_inertia", 0)) != 0:
            raise Failure("Model", "WR adapter requires massless electrons")
        self.context = context
        self.base_state = base_state
        self.slab = int(slab)
        if trial_id is None:
            self.trial = _fresh(context, "next_trial")
        else:
            if isinstance(trial_id, bool) or int(trial_id) <= 0:
                raise Failure("WRTrial", "trial_id must be a positive integer")
            self.trial = int(trial_id)
        self.work = WorkCounts()
        self.fine_records: list[FieldRecord] = []
        self.coarse_records: list[CoarseRecord] = []
        self._source_start: int | None = None

    def _tag(self, identity, role):
        generation = _fresh(self.context, "next_generation")
        name = f"wr_{role}"
        if role == "field_rk4":
            name += f"_r{identity.rk_stage}"
        micro_step = -1 if identity.micro_step is None else int(identity.micro_step)
        return StageTag(
            step=int(self.base_state.step),
            generation=generation,
            trial=self.trial,
            spectral_context=int(self.context.spectral_id),
            time=float(self.base_state.time),
            stage=name,
            micro_step=micro_step,
            iteration=int(identity.iteration),
            slab=self.slab,
        )

    def _state(self, p, z, identity, role):
        tag = self._tag(identity, role)
        state = state_from_packed(
            self.context,
            p,
            z,
            step=tag.step,
            generation=tag.generation,
            time=tag.time,
        )
        return state, tag

    def _full_stage(self, state, tag):
        self.work.full_stage_attempts += 1
        try:
            result = stage_module.evaluate_stage(
                state,
                tag,
                self.context,
                False,
            )
        except Exception:
            self.work.full_stage_failures += 1
            raise
        self.work.full_stage_calls += 1
        self.work.full_ion_power_ledger_values += int(
            getattr(result, "ledger", (None, None))[0] is not None
        )
        self.work.full_electron_power_ledger_values += int(
            getattr(result, "ledger", (None, None))[1] is not None
        )
        return result

    def _attach_failure(self, failure, metadata):
        failure.adapter_work = self.work.snapshot()
        failure.adapter_metadata = dict(metadata)
        failure.adapter_fine_records = tuple(self.fine_records)
        failure.adapter_coarse_records = tuple(self.coarse_records)
        if self._source_start is not None:
            failure.adapter_source_delta = (
                int(self.context.next_source) - self._source_start
            )
        if not hasattr(failure, "partial_arrays"):
            failure.partial_arrays = {}
        for name, value in metadata.items():
            if isinstance(value, _xp(self.context).ndarray):
                failure.partial_arrays.setdefault(f"adapter_{name}", value)

    def field(self, theta, p, z, identity):
        """Evaluate one fine field/electron/grid-response callback."""

        state, tag = self._state(p, z, identity, "field_rk4")
        self.work.field_callback_calls += 1
        alias = identity.rk_stage in (1, 2)
        try:
            frame = massless_field.evaluate(
                state,
                tag,
                self.context,
                alias=alias,
            )
            self.work.field_evaluations += int(frame.work.field_evaluations)
            self.work.alias_field_evaluations += int(
                frame.work.alias_field_evaluations
            )
            self.work.feedback_calls += 1
            feedback = massless_field.feedback_rate(frame, self.context)
            self.work.electromagnetic_rates += int(
                frame.work.electromagnetic_rates
            )
            self.work.wall_projections += int(frame.work.wall_projections)
            xp = _xp(self.context)
            modes = 2 * self.context.p.K + 1
            nc = self.context.op.Nc
            nb = self.context.op.Nb
            G = xp.ascontiguousarray(
                xp.concatenate(
                    (
                        frame.field_rates.electron_perp.reshape(modes, nc),
                        frame.field_rates.electron_parallel.reshape(modes, nc),
                        frame.field_rates.B.reshape(modes, nb),
                    ),
                    axis=1,
                )
            )
            C = xp.ascontiguousarray(
                xp.concatenate(
                    (
                        feedback.n.reshape(modes, nc),
                        feedback.mom.reshape(modes, 3 * nc),
                        feedback.perp.reshape(modes, nc),
                    ),
                    axis=1,
                )
            )
            residual_max = _residual_summary(frame, self.context)
            try:
                exchange = _electric_exchange(frame, self.context)
                exchange_error = None
                self.work.exchange_diagnostics += 1
            except Exception as error:
                exchange = None
                exchange_error = f"{type(error).__name__}: {error}"
            wall_cache = frame.cache.wall_correction.cache
            record = FieldRecord(
                identity=identity,
                tag=tag,
                input_P=p.copy(),
                input_Z=z.copy(),
                G=G,
                C=C,
                frame=frame,
                feedback=feedback,
                alias_field_evaluations=int(
                    frame.work.alias_field_evaluations
                ),
                cache_object_id=id(wall_cache),
                cache_key=tuple(wall_cache.key),
                cache_storage_bytes=int(wall_cache.storage_bytes),
                residual_max=residual_max,
                electric_exchange_W=exchange,
                electric_exchange_error=exchange_error,
            )
            self.fine_records.append(record)
            return G, C, record
        except Failure as failure:
            self._attach_failure(
                failure,
                {"input_P": p.copy(), "input_Z": z.copy(), "tag": tag},
            )
            raise
        except Exception as error:
            failure = Failure("NumericalLibrary", str(error))
            self._attach_failure(
                failure,
                {"input_P": p.copy(), "input_Z": z.copy(), "tag": tag},
            )
            raise failure from error

    def transport(self, theta, p, z, identity):
        """Evaluate one complete coarse stage and return its ion transport A."""

        state, tag = self._state(p, z, identity, "transport_coarse")
        self.work.coarse_transport_calls += 1
        try:
            full = self._full_stage(state, tag)
            xp = _xp(self.context)
            nc = self.context.op.Nc
            F = pack_stage_rhs(full, 0, self.context)
            ion = full.fluid[0]
            transport_force = (
                -xp.asarray(ion.convection) + xp.asarray(ion.pressure_force)
            )
            transport_projection = project_wall(
                ion.wall, transport_force, self.context
            )
            self.work.coarse_transport_wall_projections += 1
            A = xp.concatenate(
                (
                    xp.zeros(
                        (2 * self.context.p.K + 1, nc),
                        dtype=xp.complex128,
                    ),
                    forward(
                        transport_projection.value,
                        3 * nc,
                        self.context,
                    ).reshape(2 * self.context.p.K + 1, 3 * nc),
                    full.rhs.fluid[0].perp.reshape(
                        2 * self.context.p.K + 1, nc
                    ),
                ),
                axis=1,
            )
            grid_force = (
                xp.asarray(ion.magnetic_force)
                + xp.asarray(full.terms[0].electric)
            )
            grid_projection = project_wall(ion.wall, grid_force, self.context)
            self.work.coarse_grid_wall_projections += 1
            C = xp.concatenate(
                (
                    full.rhs.fluid[0].n.reshape(2 * self.context.p.K + 1, nc),
                    forward(
                        grid_projection.value,
                        3 * nc,
                        self.context,
                    ).reshape(2 * self.context.p.K + 1, 3 * nc),
                    xp.zeros(
                        (2 * self.context.p.K + 1, nc),
                        dtype=xp.complex128,
                    ),
                ),
                axis=1,
            )
            A = xp.ascontiguousarray(A)
            C = xp.ascontiguousarray(C)
            defect = xp.max(xp.abs(A + C - F))
            scale = max(
                float(xp.max(xp.abs(F)).item()),
                float(xp.max(xp.abs(A)).item())
                + float(xp.max(xp.abs(C)).item()),
                1.0e-300,
            )
            split_relative = float(defect.item()) / scale
            if split_relative > 2.0e-11:
                raise Failure(
                    "WRSplit",
                    f"A+C disagrees with complete ion RHS: {split_relative}",
                )
            self.work.split_validations += 1
            record = CoarseRecord(
                identity=identity,
                tag=tag,
                A=A,
                C=C,
                F=F,
                split_relative_error=split_relative,
                frame=full,
                full=full,
                ion_frame=full.fluid[0],
                transport_projection=transport_projection,
            )
            self.coarse_records.append(record)
            return A, record
        except Failure as failure:
            self._attach_failure(
                failure,
                {"input_P": p.copy(), "input_Z": z.copy(), "tag": tag},
            )
            raise
        except Exception as error:
            failure = Failure("NumericalLibrary", str(error))
            self._attach_failure(
                failure,
                {"input_P": p.copy(), "input_Z": z.copy(), "tag": tag},
            )
            raise failure from error

    def solve_window(
        self,
        p0,
        z0,
        *,
        H,
        M,
        s,
        rtol=1.0e-10,
        max_iterations=12,
        relaxation=1.0,
        initial="tangent",
        ledger=None,
        endpoint_tag: StageTag | None = None,
        endpoint_tag_factory=None,
    ):
        """Solve one WR window and rebuild its real full endpoint once."""

        _validate_packed(p0, z0, self.context)
        xp = _xp(self.context)
        p_before = p0.copy()
        z_before = z0.copy()
        ledger_before = copy.deepcopy(ledger) if ledger is not None else None
        self._source_start = int(self.context.next_source)
        clock_start = time.perf_counter_ns()
        p_scales, _ = physical_scales(self.context)
        try:
            window = _solve_window(
                p0,
                z0,
                H,
                M,
                s,
                self.transport,
                self.field,
                xp=xp,
                p_scales=p_scales,
                rtol=rtol,
                max_iterations=max_iterations,
                relaxation=relaxation,
                initial=initial,
            )
            if endpoint_tag is not None and endpoint_tag_factory is not None:
                raise Failure(
                    "WREndpoint",
                    "provide endpoint_tag or endpoint_tag_factory, not both",
                )
            if endpoint_tag is None and endpoint_tag_factory is not None:
                endpoint_tag = endpoint_tag_factory()
            if endpoint_tag is None:
                endpoint_state, endpoint_tag = self._state(
                    window.P_endpoint,
                    window.Z_endpoint,
                    WindowIdentity(
                        iteration=window.sweeps,
                        micro_step=int(M) - 1,
                        rk_stage=None,
                        coarse_node=int(s) - 1,
                        role="endpoint_reconstruction",
                        theta=1.0,
                    ),
                    "endpoint_reconstruction",
                )
            else:
                if (
                    endpoint_tag.step != int(self.base_state.step) + 1
                    or not math.isfinite(float(endpoint_tag.time))
                    or float(endpoint_tag.time) <= float(self.base_state.time)
                    or endpoint_tag.trial != self.trial
                    or endpoint_tag.spectral_context
                    != int(self.context.spectral_id)
                ):
                    raise Failure(
                        "WREndpoint",
                        "parent endpoint tag does not match accepted macro identity",
                    )
                endpoint_state = state_from_packed(
                    self.context,
                    window.P_endpoint,
                    window.Z_endpoint,
                    step=endpoint_tag.step,
                    generation=endpoint_tag.generation,
                    time=endpoint_tag.time,
                )
            self.work.endpoint_reconstruction_calls += 1
            self.work.endpoint_full_stage_calls += 1
            endpoint = self._full_stage(endpoint_state, endpoint_tag)
            endpoint_p, endpoint_z = pack_state(
                self.context, endpoint.canonical_state
            )
            endpoint_scale = xp.maximum(
                xp.max(xp.abs(window.P_endpoint), axis=0),
                xp.max(xp.abs(endpoint_p), axis=0),
            )
            z_endpoint_scale = xp.maximum(
                xp.max(xp.abs(window.Z_endpoint), axis=0),
                xp.max(xp.abs(endpoint_z), axis=0),
            )
            if (
                float(
                    xp.max(
                        xp.abs(window.P_endpoint - endpoint_p)
                        / xp.maximum(endpoint_scale, 1.0e-300)
                    ).item()
                )
                > 2.0e-12
                or float(
                    xp.max(
                        xp.abs(window.Z_endpoint - endpoint_z)
                        / xp.maximum(z_endpoint_scale, 1.0e-300)
                    ).item()
                )
                > 2.0e-12
            ):
                raise Failure(
                    "WREndpoint",
                    "full endpoint frame does not reproduce accepted P/Z",
                )
            record = EndpointRecord(
                tag=endpoint_tag,
                full=endpoint,
                P=endpoint_p,
                Z=endpoint_z,
                totals=endpoint.totals,
                checks=tuple(endpoint.checks),
            )
            clock_end = time.perf_counter_ns()
            source_end = int(self.context.next_source)
            caller_unchanged = _snapshot_equal(p0, p_before, xp) and _snapshot_equal(
                z0, z_before, xp
            )
            ledger_unchanged = (
                ledger is None or _snapshot_equal(ledger, ledger_before, xp)
            )
            fields = self.fine_records
            accepted_times = {
                record.tag.time for record in fields
            }
            identity = {
                "accepted_left_clock_exact": accepted_times
                == {float(self.base_state.time)},
                "distinct_stage_generations": len(
                    {record.tag.generation for record in fields}
                )
                == len(fields),
                "middle_aliases_charged": sum(
                    record.alias_field_evaluations for record in fields
                ),
                "wall_cache_object_ids": sorted(
                    {record.cache_object_id for record in fields}
                ),
            }
            final_coarse = tuple(
                record
                for record in self.coarse_records
                if record.identity.iteration == window.accepted_node_version
            )
            final_fine = tuple(
                record
                for record in self.fine_records
                if record.identity.iteration == window.accepted_node_version
            )
            if len(final_coarse) != int(s) or len(final_fine) != 4 * int(M):
                raise Failure(
                    "WREvidence",
                    "final accepted record counts do not match s and 4*M",
                )
            return WindowPhysicsResult(
                window=window,
                endpoint=record,
                coarse_records=final_coarse,
                fine_records=final_fine,
                all_coarse_records=tuple(self.coarse_records),
                all_fine_records=tuple(self.fine_records),
                work=self.work,
                source={
                    "start": self._source_start,
                    "end": source_end,
                    "producer_delta": source_end - self._source_start,
                },
                clock={
                    "start_perf_counter_ns": clock_start,
                    "end_perf_counter_ns": clock_end,
                    "elapsed_ns": clock_end - clock_start,
                },
                version_evidence=_version_evidence(),
                identity_evidence=identity,
                caller_unchanged=caller_unchanged,
                ledger_unchanged=ledger_unchanged,
            )
        except Failure as failure:
            self._attach_failure(
                failure,
                {"input_P": p0.copy(), "input_Z": z0.copy()},
            )
            failure.caller_unchanged = _snapshot_equal(p0, p_before, xp) and (
                _snapshot_equal(z0, z_before, xp)
            )
            failure.ledger_unchanged = (
                ledger is None or _snapshot_equal(ledger, ledger_before, xp)
            )
            raise
