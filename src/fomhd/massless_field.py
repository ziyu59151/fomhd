"""Field-only evaluation for the massless quasineutral electron branch.

This module owns electromagnetic electron/field response, not an ion transport
step or a complete production stage. All arrays are FP64/complex128 arrays
owned by ``context.compute``; no CPU fallback is introduced for CUDA.
The returned trial state, Z rates, and derived records are fresh outputs. The
context keeps ownership of parameters, geometry, operators, and the fixed wall
cache; callers must keep the context alive while returned cache records live.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, NamedTuple

from . import diagnostics
from .core import (
    _hermitian_check,
    _history_sample_count,
    forward,
    inverse,
    mode_apply,
    sample_apply,
)
from .fluid import _sampled_divergence, _warp_magnetic_force
from .massless import (
    MODEL_NAME,
    _check_retained_massless,
    _copy_state,
    _derive_electron,
    _electron_fluid,
    _electroweak_parameters,
    _fresh,
    _initial_constraints,
    _massless_e0,
    _modes,
    _wall_correction,
    _xp,
)
from .types import Failure, StageTag, field_cache_record
from .wall import project_wall
from .wall_cache import get_wall_cache


FIELD_ONLY_KIND = "massless_field_only"


class IonRate(NamedTuple):
    """An ion grid-feedback rate in retained SI units.

    ``n`` has unit 1/(m^3*s), ``mom`` is momentum rate
    kg/(m^2*s^2), and ``perp`` is pressure rate Pa/s. Fields are backend-owned
    complex128 arrays in the standard ``[mode][entity]`` layout.
    """

    n: Any
    mom: Any
    perp: Any


@dataclass(frozen=True, slots=True)
class IonEMTerms:
    """Read-only padded ion EM force and wall-projection producer record.

    ``electric``, ``magnetic``, ``projected``, and ``reaction`` use the padded
    ``[sample][R,phi,Z][cell]`` nodal layout in kg/(m^2*s^2). ``multiplier`` and
    ``multiplier_residual`` use the actual wall-work layout for this context.
    Arrays are backend-owned and live with the frame/context.
    """

    electric: Any
    magnetic: Any
    projected: Any
    reaction: Any
    multiplier: Any
    multiplier_residual: Any
    tag: StageTag


@dataclass(slots=True)
class WorkCounts:
    """Actual operations charged to one field-only record."""

    field_evaluations: int = 0
    derive_electron: int = 0
    electron_cgl: int = 0
    electron_constraint_projections: int = 0
    electron_magnetic_forces: int = 0
    ion_magnetic_forces: int = 0
    warp_magnetic_kernel_evaluations: int = 0
    wall_cache_setups: int = 0
    wall_cache_evaluations: int = 0
    alias_field_evaluations: int = 0
    electromagnetic_rates: int = 0
    feedback_rates: int = 0
    wall_projections: int = 0
    ion_transport: int = 0
    finish_fluid: int = 0
    full_ion_rhs: int = 0


@dataclass(slots=True)
class FieldFrame:
    """A read-only field/electron record with an owned trial state."""

    kind: str
    complete_stage: bool
    diagnostic_scope: str
    tag: StageTag
    trial_state: Any
    field_rates: SimpleNamespace
    cache: Any
    ion: Any
    electron: Any
    BN: Any
    bhat: Any
    current: Any
    E0: Any
    compatibility_momentum: Any
    source_id: int
    model: str
    applicability: Any
    op: Any
    owner_context: Any
    owner_identity: tuple[Any, ...]
    checks: list[Any] = field(default_factory=list)
    residuals: dict[str, Any] = field(default_factory=dict)
    work: WorkCounts = field(default_factory=WorkCounts)
    ion_em_terms: IonEMTerms | None = None
    alias_frame: FieldFrame | None = None
    alias_work: WorkCounts | None = None


def _new_frame(tag, context):
    result = FieldFrame(
        kind=FIELD_ONLY_KIND,
        complete_stage=False,
        diagnostic_scope=(
            "massless electron CGL-B, Ohm, wall, and CT field response only; "
            "no ion transport or full-stage diagnostics"
        ),
        tag=tag,
        trial_state=None,
        field_rates=None,
        cache=None,
        ion=None,
        electron=None,
        BN=None,
        bhat=None,
        current=None,
        E0=None,
        compatibility_momentum=None,
        source_id=0,
        model=MODEL_NAME,
        applicability=SimpleNamespace(
            electron_inertia="not_applicable",
            electron_kinetic_energy="not_applicable",
            electron_kinetic_transport="not_applicable",
            electron_mechanical_stress="not_applicable",
            finite_charge_electric_stress="not_applicable",
            longitudinal_electric_energy="not_applicable",
            finite_charge_gauss="not_applicable",
            spectral_kinetic_transfer="not_applicable",
        ),
        op=None,
        owner_context=context,
        owner_identity=(
            id(context),
            id(context.p),
            id(context.op),
            id(context.geometry),
            id(context.compute),
            int(context.N),
            int(context.spectral_id),
        ),
        ion_em_terms=None,
    )
    return result


def _truth(out, name, value, tag):
    record = diagnostics._record(name=name, tag=tag)
    record.raw = 0.0 if value else 1.0
    record.reference = 1.0
    record.threshold = 0.0
    record.available = True
    record.passed = bool(value)
    out.append(record)


def _minimum(out, name, values, floor, unit, tag, population=""):
    record = diagnostics._record(
        name=name,
        tag=tag,
        unit=unit,
        population=population,
    )
    record.reference = float(floor)
    record.threshold = float(floor)
    record.available = int(values.size) > 0
    record.raw = (
        float(values.min().real) if record.available else math.inf
    )
    record.passed = record.available and record.raw > floor
    out.append(record)


def _field_residuals(frame, context):
    xp = _xp(context)
    nc = context.op.Nc
    ratio = _electroweak_parameters(context)[1]
    state = frame.trial_state
    ion = frame.ion
    electron = frame.electron
    density_rate = -xp.asarray(electron.divergence_hat)
    ion_density_rate = -xp.asarray(ion.divergence_hat)
    residuals = {
        "quasineutrality": (
            xp.asarray(state.fluid[1].n) - ratio * xp.asarray(state.fluid[0].n)
        ),
        "electron_density_compatibility": (
            density_rate - ratio * ion_density_rate
        ),
        "electron_momentum_compatibility": (
            xp.asarray(state.fluid[1].mom)
            - xp.asarray(frame.compatibility_momentum)
        ),
        "current_definition": (
            xp.asarray(frame.current)
            - mode_apply(context.op.Cdag, state.B, context.compute)
            / context.p.mu0
        ),
        "electron_flux_definition": (
            xp.asarray(electron.flux_hat)
            - (
                context.p.species[0].charge * xp.asarray(ion.flux_hat)
                - xp.asarray(frame.current)
            )
            / (-context.p.species[1].charge)
        ),
        "electron_flux_continuity": (
            density_rate
            + mode_apply(context.op.DJ, electron.flux_hat, context.compute)
        ),
        "djj": mode_apply(
            context.op.DJ,
            frame.current,
            context.compute,
        ),
        "ct_divergence": mode_apply(
            context.op.DB,
            frame.field_rates.B,
            context.compute,
        ),
        "ct_update": (
            xp.asarray(frame.field_rates.B)
            + mode_apply(context.op.C, frame.cache.E, context.compute)
        ),
        "wall_field_original_rows": xp.concatenate(
            [
                record.full_residual
                for record in frame.cache.wall_correction.modes
            ]
        ),
        "drift_ohm_residual": (
            xp.asarray(frame.cache.E)
            - xp.asarray(frame.E0)
            - xp.asarray(frame.cache.wall_reaction)
        ),
        "theta_consistency": (
            xp.asarray(electron.theta)
            - xp.asarray(electron.theta_consistency)
        ),
    }
    perp_expected = (
        -_sampled_divergence(electron.pressure_flux, context).reshape(
            context.N, nc
        )
        + xp.asarray(electron.perp).reshape(context.N, nc)
        * (
            xp.asarray(electron.strain).reshape(context.N, nc)
            - xp.asarray(electron.theta).reshape(context.N, nc)
        )
    )
    parallel_expected = (
        -_sampled_divergence(electron.parallel_flux, context).reshape(
            context.N, nc
        )
        - 2.0
        * xp.asarray(electron.parallel).reshape(context.N, nc)
        * xp.asarray(electron.strain).reshape(context.N, nc)
    )
    residuals["electron_cgl_perp"] = (
        xp.asarray(electron.pdot_perp).reshape(context.N, nc)
        - perp_expected
    ).reshape(-1)
    residuals["electron_cgl_parallel"] = (
        xp.asarray(electron.pdot_parallel).reshape(context.N, nc)
        - parallel_expected
    ).reshape(-1)
    return residuals


def _wall_flux_samples(flux, context):
    xp = _xp(context)
    nc = context.op.Nc
    nr = len(context.geometry.r_faces.keys)
    samples = xp.asarray(flux).reshape(context.N, context.op.Ne)
    result = xp.empty(
        (context.N, len(context.geometry.walls)),
        dtype=xp.complex128,
    )
    for index, wall in enumerate(context.geometry.walls):
        edge = (
            wall.face
            if wall.component == "r_face"
            else nr + nc + wall.face
        )
        result[:, index] = wall.sign * samples[:, edge]
    return result.reshape(-1)


def _check_frame(frame, context):
    xp = _xp(context)
    nc = context.op.Nc
    ne = context.op.Ne
    tag = frame.tag
    state = frame.trial_state
    out = frame.checks
    finite = all(
        bool(xp.all(xp.isfinite(value)).item())
        for value in (
            state.B,
            frame.field_rates.electron_perp,
            frame.field_rates.electron_parallel,
            frame.field_rates.B,
            frame.cache.E,
            frame.current,
            frame.electron.u,
            state.fluid[0].n,
            state.fluid[0].mom,
            state.fluid[0].perp,
            state.fluid[1].n,
            state.fluid[1].mom,
            state.fluid[1].perp,
            state.fluid[1].parallel,
        )
    )
    _truth(out, "finite", finite, tag)
    _truth(
        out,
        "model_massless_quasineutral",
        frame.model == MODEL_NAME
        and int(getattr(context.p, "electron_inertia", 0)) == 0
        and context.p.species[1].mass > 0.0,
        tag,
    )
    _truth(
        out,
        "independent_state_hermitian",
        True,
        tag,
    )
    _truth(
        out,
        "source_identity",
        (
            frame.source_id != 0
            and frame.cache.tag == tag
            and frame.ion.tag == tag
            and frame.ion.wall.tag == tag
            and frame.electron.tag == tag
            and frame.electron.wall.tag == tag
            and state.step == tag.step
            and state.generation == tag.generation
            and state.time == tag.time
        ),
        tag,
    )
    _minimum(
        out,
        "minimum_B",
        xp.sqrt(
            xp.sum(
                xp.abs(xp.asarray(frame.BN).reshape(context.N, 3, nc))
                ** 2,
                axis=1,
            )
        ).reshape(-1),
        context.p.B_min,
        "T",
        tag,
    )
    for species in range(2):
        fluid = frame.ion if species == 0 else frame.electron
        _minimum(
            out,
            "minimum_density",
            xp.real(xp.asarray(fluid.n)),
            context.p.species[species].n_min,
            "1/m^3",
            tag,
            context.p.species[species].id,
        )
    _hermitian_check(
        frame.field_rates.electron_perp,
        nc,
        context,
        "field electron perpendicular pressure rate",
    )
    _hermitian_check(
        frame.field_rates.electron_parallel,
        nc,
        context,
        "field electron parallel pressure rate",
    )
    _hermitian_check(
        frame.field_rates.B,
        context.op.Nb,
        context,
        "field magnetic rate",
    )
    _truth(out, "hermitian_field_rates", True, tag)

    m0 = xp.real(xp.asarray(context.op.M0)).astype(xp.float64)
    m1 = xp.real(xp.asarray(context.op.M1)).astype(xp.float64)
    mb = xp.real(xp.asarray(context.op.MB)).astype(xp.float64)
    wall_weights = xp.ones(
        len(context.geometry.walls),
        dtype=xp.float64,
    )
    residuals = frame.residuals
    diagnostics.measure(
        out,
        "quasineutrality",
        residuals["quasineutrality"],
        m0,
        "C/m^3",
        tag,
        context,
    )
    diagnostics.measure(
        out,
        "electron_density_compatibility",
        residuals["electron_density_compatibility"],
        m0,
        "1/(m^3*s)",
        tag,
        context,
    )
    diagnostics.measure(
        out,
        "electron_momentum_compatibility",
        residuals["electron_momentum_compatibility"],
        xp.tile(m0, 3),
        "kg/(m^2*s^2)",
        tag,
        context,
    )
    diagnostics.measure(
        out,
        "current_definition",
        residuals["current_definition"],
        m1,
        "A/m^2",
        tag,
        context,
    )
    diagnostics.measure(
        out,
        "electron_flux_definition",
        residuals["electron_flux_definition"],
        m1,
        "1/(m^2*s)",
        tag,
        context,
    )
    diagnostics.measure(
        out,
        "electron_flux_continuity",
        residuals["electron_flux_continuity"],
        m0,
        "1/(m^3*s)",
        tag,
        context,
    )
    diagnostics.measure(
        out,
        "djj",
        residuals["djj"],
        m0,
        "A/m^3",
        tag,
        context,
    )
    diagnostics.measure(
        out,
        "ct_divergence",
        residuals["ct_divergence"],
        context.geometry.wv,
        "T/(m*s)",
        tag,
        context,
    )
    diagnostics.measure(
        out,
        "ct_update",
        residuals["ct_update"],
        mb,
        "T/s",
        tag,
        context,
    )
    diagnostics.measure(
        out,
        "wall_current",
        _wall_flux_samples(
            inverse(frame.current, ne, context),
            context,
        ),
        wall_weights,
        "A/m^2",
        tag,
        context,
        samples=True,
    )
    for species in range(2):
        diagnostics.measure(
            out,
            f"wall_{'ion' if species == 0 else 'electron'}_flux",
            _wall_flux_samples(frame.electron.flux, context)
            if species
            else _wall_flux_samples(frame.ion.flux, context),
            wall_weights,
            "1/(m^2*s)",
            tag,
            context,
            samples=True,
        )
    diagnostics.measure(
        out,
        "wall_field_original_rows",
        residuals["wall_field_original_rows"],
        xp.ones(
            residuals["wall_field_original_rows"].size,
            dtype=xp.float64,
        ),
        "V/m^3",
        tag,
        context,
        samples=True,
    )
    diagnostics.measure(
        out,
        "drift_ohm_residual",
        residuals["drift_ohm_residual"],
        m1,
        "V/m",
        tag,
        context,
    )
    for name in ("electron_cgl_perp", "electron_cgl_parallel"):
        diagnostics.measure(
            out,
            name,
            residuals[name],
            m0,
            "Pa/s",
            tag,
            context,
            samples=True,
        )
    theta_start = len(out)
    diagnostics.measure(
        out,
        "theta_consistency",
        residuals["theta_consistency"],
        m0,
        "1/s",
        tag,
        context,
        samples=True,
    )
    for record in out[theta_start:]:
        record.hard = False
        record.reason = (
            "field-only convergence diagnostic: padded Fe/nf divergence "
            "versus derived velocity theta"
        )
    diagnostics.gate(out)


def _wall_runtime_counts(context):
    cache = getattr(context, "_wall_system_cache", None)
    evaluations = int(getattr(cache, "evaluations", 0))
    setups = int(getattr(context, "_wall_system_cache_builds", 0))
    return evaluations, setups


def _update_wall_counts(result, context, before):
    after = _wall_runtime_counts(context)
    result.work.wall_cache_evaluations = after[0] - before[0]
    result.work.wall_cache_setups = after[1] - before[1]


def _evaluate_once(state, tag, context):
    result = _new_frame(tag, context)
    wall_before = _wall_runtime_counts(context)
    try:
        get_wall_cache(context)
        if (
            tag.step != state.step
            or tag.generation != state.generation
            or tag.time != state.time
        ):
            raise Failure("StaleStage", "stage identity does not match state")

        result.work.field_evaluations = 1
        derived = _derive_electron(
            state,
            context,
            ion_transport=False,
        )
        result.work.derive_electron = 1
        canonical = _copy_state(
            state,
            context,
            derived.electron_density_hat,
            derived.compatibility_momentum,
        )
        _check_retained_massless(canonical, context)
        derived.state_B = canonical.B
        _initial_constraints(derived, context)
        result.source_id = _fresh(context, "next_source")
        result.op = context.op
        result.trial_state = canonical
        result.BN = derived.BN
        result.bhat = derived.bhat
        result.current = derived.current
        result.compatibility_momentum = derived.compatibility_momentum

        electron = _electron_fluid(derived, canonical, tag, context)
        result.work.electron_cgl = 1
        result.work.electron_constraint_projections = 2
        result.work.electron_magnetic_forces = 1
        if context.compute.name == "cuda":
            # _warp_magnetic_force dispatches exactly once for this electron
            # CGL magnetic-force call; CPU executes no Warp kernel.
            result.work.warp_magnetic_kernel_evaluations += 1
        derived.tag = tag
        derived.ion.tag = tag
        derived.ion.wall.tag = tag
        result.ion = derived.ion
        result.electron = electron
        e0 = _massless_e0(electron, derived, context)
        correction, wall_correction = _wall_correction(e0, context)
        _update_wall_counts(result, context, wall_before)
        electric = e0 + correction
        result.E0 = e0
        result.cache = field_cache_record(
            phi=None,
            EL=None,
            ET=electric,
            E=electric,
            multiplier=None,
            poisson_residual=None,
            tag=tag,
            poisson_solves=[],
            transverse_solve=None,
        )
        result.cache.wall_correction = wall_correction
        result.cache.wall_reaction = correction
        result.cache.E0 = e0
        result.cache.model = MODEL_NAME
        result.cache.applicability = result.applicability
        result.field_rates = SimpleNamespace(
            electron_perp=forward(
                electron.pdot_perp,
                context.op.Nc,
                context,
            ),
            electron_parallel=forward(
                electron.pdot_parallel,
                context.op.Nc,
                context,
            ),
            B=-mode_apply(context.op.C, electric, context.compute),
        )
        result.residuals = _field_residuals(result, context)
        _check_frame(result, context)
        return result
    except Failure as failure:
        _update_wall_counts(result, context, wall_before)
        _retain_field_failure(
            failure,
            tag,
            context,
            state,
            result,
            "field_trial",
        )
        raise
    except Exception as error:
        _update_wall_counts(result, context, wall_before)
        failure = Failure("NumericalLibrary", str(error))
        _retain_field_failure(
            failure,
            tag,
            context,
            state,
            result,
            "field_trial",
        )
        raise failure from error


def _alias_probe(canonical, tag, context, result):
    updated = copy.copy(context)
    updated._history_samples = _history_sample_count(context)
    updated.N = 2 * context.N + 1
    updated.spectral_id = context.spectral_id + 1
    alternate_tag = StageTag(
        step=tag.step,
        generation=tag.generation,
        trial=tag.trial,
        spectral_context=updated.spectral_id,
        time=tag.time,
        stage="alias",
        micro_step=tag.micro_step,
        iteration=tag.iteration,
        slab=tag.slab,
    )
    alternate = _evaluate_once(canonical, alternate_tag, updated)
    context.next_source = updated.next_source
    result.alias_frame = alternate
    result.alias_work = alternate.work
    result.work.field_evaluations += alternate.work.field_evaluations
    result.work.derive_electron += alternate.work.derive_electron
    result.work.electron_cgl += alternate.work.electron_cgl
    result.work.electron_constraint_projections += (
        alternate.work.electron_constraint_projections
    )
    result.work.electron_magnetic_forces += (
        alternate.work.electron_magnetic_forces
    )
    result.work.warp_magnetic_kernel_evaluations += (
        alternate.work.warp_magnetic_kernel_evaluations
    )
    result.work.wall_cache_evaluations += alternate.work.wall_cache_evaluations
    result.work.wall_cache_setups += alternate.work.wall_cache_setups
    result.work.alias_field_evaluations += alternate.work.field_evaluations

    diagnostics.measure(
        result.checks,
        "alias_perp",
        alternate.field_rates.electron_perp
        - result.field_rates.electron_perp,
        context.op.M0,
        "Pa/s",
        tag,
        context,
        context.p.species[1].id,
    )
    diagnostics.measure(
        result.checks,
        "alias_parallel",
        alternate.field_rates.electron_parallel
        - result.field_rates.electron_parallel,
        context.op.M0,
        "Pa/s",
        tag,
        context,
        context.p.species[1].id,
    )
    offsets = (
        0,
        len(context.geometry.z_faces.keys),
        len(context.geometry.z_faces.keys)
        + len(context.geometry.vertices.keys),
        context.op.Nb,
    )
    mb = _xp(context).real(
        _xp(context).asarray(context.op.MB)
    ).astype(_xp(context).float64)
    for component, name in enumerate(("R", "phi", "Z")):
        start, stop = offsets[component], offsets[component + 1]
        left = result.field_rates.B.reshape(
            _modes(context), context.op.Nb
        )[:, start:stop].reshape(-1)
        right = alternate.field_rates.B.reshape(
            _modes(context), context.op.Nb
        )[:, start:stop].reshape(-1)
        diagnostics.measure(
            result.checks,
            "alias_B",
            right - left,
            mb[start:stop],
            "T/s",
            tag,
            context,
            component=name,
        )
    diagnostics.gate(result.checks)


def evaluate(state, tag, context, alias=False):
    """Evaluate Z=(electron_perp,electron_parallel,B) without ion transport."""
    result = _evaluate_once(state, tag, context)
    if not alias:
        return result
    try:
        _alias_probe(result.trial_state, tag, context, result)
    except Failure as failure:
        failure.alias_failure = True
        if getattr(failure, "stage", None) is None:
            failure.stage = tag
        _retain_field_failure(
            failure,
            tag,
            context,
            state,
            result,
            "field_base",
        )
        raise
    except Exception as error:
        failure = Failure("NumericalLibrary", str(error))
        failure.alias_failure = True
        _retain_field_failure(
            failure,
            tag,
            context,
            state,
            result,
            "field_base",
        )
        raise failure from error
    return result


def _retain_field_failure(
    failure,
    tag,
    context,
    state,
    result,
    prefix,
):
    failure.field_only = True
    if not hasattr(failure, "field_work"):
        failure.field_work = result.work
    if getattr(failure, "stage", None) is None:
        failure.stage = tag
    if not hasattr(failure, "checks"):
        failure.checks = []
    if not hasattr(failure, "partial_arrays"):
        failure.partial_arrays = {}
    for check in getattr(result, "checks", ()):
        if check not in failure.checks:
            failure.checks.append(check)
    if result.trial_state is not None:
        trial = result.trial_state
        failure.partial_arrays.setdefault(
            f"{prefix}_trial_B",
            trial.B,
        )
        for species, ident in enumerate(("i", "e")):
            fluid = trial.fluid[species]
            failure.partial_arrays.setdefault(
                f"{prefix}_trial_n_{ident}",
                fluid.n,
            )
            failure.partial_arrays.setdefault(
                f"{prefix}_trial_momentum_{ident}",
                fluid.mom,
            )
            failure.partial_arrays.setdefault(
                f"{prefix}_trial_perp_{ident}",
                fluid.perp,
            )
    if state is not None:
        failure.partial_arrays.setdefault(f"{prefix}_input_B", state.B)
        for species, ident in enumerate(("i", "e")):
            fluid = state.fluid[species]
            failure.partial_arrays.setdefault(
                f"{prefix}_input_n_{ident}",
                fluid.n,
            )
            failure.partial_arrays.setdefault(
                f"{prefix}_input_momentum_{ident}",
                fluid.mom,
            )
            failure.partial_arrays.setdefault(
                f"{prefix}_input_perp_{ident}",
                fluid.perp,
            )
            if species == 1:
                failure.partial_arrays.setdefault(
                    f"{prefix}_input_parallel_e",
                    fluid.parallel,
                )
    if result.cache is not None and result.cache.E is not None:
        failure.partial_arrays.setdefault(f"{prefix}_E", result.cache.E)
    if result.field_rates is not None:
        failure.partial_arrays.setdefault(
            f"{prefix}_electron_perp",
            result.field_rates.electron_perp,
        )
        failure.partial_arrays.setdefault(
            f"{prefix}_electron_parallel",
            result.field_rates.electron_parallel,
        )
        failure.partial_arrays.setdefault(
            f"{prefix}_B",
            result.field_rates.B,
        )
    return None


def _require_field_frame(frame, context):
    if getattr(frame, "kind", None) != FIELD_ONLY_KIND:
        raise Failure("FieldFrame", "expected a field-only massless frame")
    owner = getattr(frame, "owner_context", None)
    context_identity = (
        id(context),
        id(context.p),
        id(context.op),
        id(context.geometry),
        id(context.compute),
        int(context.N),
        int(context.spectral_id),
    )
    if owner is not context or context_identity != frame.owner_identity:
        raise Failure(
            "FieldOwnership",
            "field frame is bound to a different producing context",
        )
    if frame.tag is None or frame.cache.tag != frame.tag:
        raise Failure("FieldFrame", "field frame tag identity is inconsistent")
    if frame.ion.tag != frame.tag or frame.ion.wall.tag != frame.tag:
        raise Failure("FieldFrame", "ion record does not belong to this field frame")
    if frame.electron.tag != frame.tag or frame.electron.wall.tag != frame.tag:
        raise Failure(
            "FieldFrame",
            "electron record does not belong to this field frame",
        )
    if (
        frame.trial_state.step != frame.tag.step
        or frame.trial_state.generation != frame.tag.generation
        or frame.trial_state.time != frame.tag.time
    ):
        raise Failure("StaleStage", "field frame state identity is stale")
    if frame.source_id == 0:
        raise Failure("FieldFrame", "field frame source identity is absent")
    if frame.ion.n.size != context.N * context.op.Nc:
        raise Failure("FieldFrame", "field frame context shape mismatch")


def _physical_finite(value, context, name):
    xp = _xp(context)
    if not bool(xp.all(xp.isfinite(value)).item()):
        raise Failure("Finite", f"{name} contains non-finite values")


def electromagnetic_rate(frame, context):
    """Return the wall-projected ion electromagnetic momentum response.

    The return tuple is ``(zero_n, momentum, zero_pressure)``. The momentum
    rate includes the pointwise ion Lorentz force and electric grid response,
    projected by the frame's actual fixed-density wall system. It is a grid EM
    response, not an ion-transport or full-ion evaluation.

    The same counted call stores the padded producer in ``frame.ion_em_terms``.
    Consumers must read that record rather than reconstructing padded forces or
    wall reactions from retained momentum, and must not mutate the record.
    """
    _require_field_frame(frame, context)
    xp = _xp(context)
    nc = context.op.Nc
    ion = frame.ion
    density = xp.asarray(ion.n).reshape(context.N, nc)
    velocity = xp.asarray(ion.u).reshape(context.N, 3, nc)
    magnetic = xp.asarray(frame.BN).reshape(context.N, 3, nc)
    _physical_finite(density, context, "EM ion density")
    _physical_finite(velocity, context, "EM ion velocity")
    _physical_finite(magnetic, context, "EM magnetic field")
    charge = float(context.p.species[0].charge)
    magnetic_force = _warp_magnetic_force(
        ion.u,
        frame.BN,
        ion.n,
        charge,
        context,
    )
    warp_used = magnetic_force is not None
    if magnetic_force is None:
        magnetic_force = charge * density[:, None, :] * xp.stack(
            (
                velocity[:, 1, :] * magnetic[:, 2, :]
                - velocity[:, 2, :] * magnetic[:, 1, :],
                velocity[:, 2, :] * magnetic[:, 0, :]
                - velocity[:, 0, :] * magnetic[:, 2, :],
                velocity[:, 0, :] * magnetic[:, 1, :]
                - velocity[:, 1, :] * magnetic[:, 0, :],
            ),
            axis=1,
        )
    else:
        # The Warp producer owns flat N*3*Nc storage. Reshape is a lifetime-
        # preserving view; no host transfer, copy, or borrowed raw pointer.
        magnetic_force = xp.asarray(magnetic_force).reshape(
            context.N,
            3,
            nc,
        )
    electric = inverse(
        frame.cache.E,
        context.op.Ne,
        context,
        True,
        context.p.ref.electric,
    )
    interpolated = sample_apply(
        context.op.H,
        electric,
        context.N,
        context.compute,
    ).reshape(context.N, 3, nc)
    electric_force = charge * density[:, None, :] * interpolated
    projection = project_wall(
        ion.wall,
        (magnetic_force + electric_force).reshape(-1),
        context,
    )
    frame.ion_em_terms = IonEMTerms(
        electric=electric_force.reshape(-1),
        magnetic=magnetic_force.reshape(-1),
        projected=projection.value,
        reaction=projection.reaction,
        multiplier=projection.multiplier,
        multiplier_residual=projection.multiplier_residual,
        tag=frame.tag,
    )
    momentum = forward(projection.value, 3 * nc, context)
    zero_density = xp.zeros(_modes(context) * nc, dtype=xp.complex128)
    zero_pressure = xp.zeros(_modes(context) * nc, dtype=xp.complex128)
    frame.work.electromagnetic_rates += 1
    frame.work.ion_magnetic_forces += 1
    frame.work.warp_magnetic_kernel_evaluations += int(warp_used)
    frame.work.wall_projections += 1
    return IonRate(zero_density, momentum, zero_pressure)


def feedback_rate(frame, context):
    """Return continuity plus electromagnetic ion-grid feedback.

    Density is the already-derived linear ion flux divergence. Transport,
    gradient, pressure, and full-ion RHS work are intentionally absent.
    """
    electromagnetic = electromagnetic_rate(frame, context)
    density = -_xp(context).asarray(frame.ion.divergence_hat)
    frame.work.feedback_rates += 1
    return IonRate(density, electromagnetic.mom, electromagnetic.perp)
