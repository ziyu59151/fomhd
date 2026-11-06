"""A0 stage assembly, initialization, and failure evidence."""

import copy
import math
from types import SimpleNamespace

from .core import (
    allocate_state,
    check_retained,
    forward,
    inverse,
    mode_apply,
    sample_apply,
)
from .fields import electrostatic, longitudinal, transverse
from .fluid import (
    build_fluid,
    build_non_electric,
    finish_fluid,
    power_ledger,
)
from .types import Failure, StageTag
from .wall import project_wall


UINT64_MAX = (1 << 64) - 1


def _xp(context):
    return context.compute.xp


def _modes(context):
    return 2 * context.p.K + 1


def _fresh(context, name):
    current = getattr(context, name)
    if current >= UINT64_MAX:
        raise Failure("CounterOverflow", "stage identity exhausted")
    setattr(context, name, current + 1)
    return current


def _empty_state_fluid():
    return SimpleNamespace(
        n=None, mom=None, perp=None, parallel=None
    )


def _empty_state():
    return SimpleNamespace(
        fluid=[_empty_state_fluid(), _empty_state_fluid()],
        B=None,
        step=0,
        generation=0,
        time=0.0,
    )


def _empty_cache():
    return SimpleNamespace(
        phi=None,
        EL=None,
        ET=None,
        E=None,
        multiplier=None,
        poisson_residual=None,
        tag=None,
        poisson_solves=[],
        transverse_solve=None,
    )


def _new_result(tag):
    return SimpleNamespace(
        tag=tag,
        rhs=_empty_state(),
        cache=_empty_cache(),
        BN=None,
        bhat=None,
        rho=None,
        current=None,
        fJ0=None,
        ELdot=None,
        phidot=None,
        ampere=None,
        AE=None,
        fluid=[None, None],
        terms=[None, None],
        ledger=[None, None],
        boundary=None,
        momentum=None,
        totals=None,
        checks=[],
        residuals={},
        source_id=0,
        op=None,
    )


def _copy_context(context, *, N=None, spectral_id=None):
    try:
        updated = copy.copy(context)
        if N is not None:
            updated.N = N
        if spectral_id is not None:
            updated.spectral_id = spectral_id
        return updated
    except Exception as error:
        raise Failure("Stage", f"cannot create alias context: {error}") from error


def _diagnostics():
    # Local import avoids diagnostics -> stage/physics initialization cycles.
    from . import diagnostics

    return diagnostics


def evaluate_stage(state, tag, context, alias=False):
    """Evaluate one complete physical stage without changing the input state."""
    if int(getattr(context.p, "electron_inertia", 0)) == 0:
        from .massless import evaluate_stage as evaluate_massless_stage

        return evaluate_massless_stage(state, tag, context, alias)
    result = _new_result(tag)
    try:
        if (
            tag.step != state.step
            or tag.generation != state.generation
            or tag.time != state.time
        ):
            raise Failure("StaleStage", "stage identity does not match state")
        check_retained(state, context)
        result.source_id = _fresh(context, "next_source")
        result.rhs = allocate_state(context)
        result.rhs.time = state.time
        result.rhs.step = state.step
        result.rhs.generation = state.generation
        result.op = context.op
        xp = _xp(context)
        nc = context.op.Nc
        ne = context.op.Ne
        modes = _modes(context)
        result.BN = inverse(
            sample_apply(context.op.QB, state.B, modes, context.compute),
            3 * nc,
            context,
            True,
        )
        bn = result.BN.reshape(context.N, 3, nc)
        magnitude = xp.sqrt(xp.sum(xp.abs(bn) ** 2, axis=1))
        if not bool(xp.all(xp.isfinite(magnitude) & (magnitude > context.p.B_min))):
            raise Failure("Magnetization", "nodal B below declared floor")
        result.bhat = (bn / magnitude[:, None, :]).reshape(-1)
        result.rho = xp.zeros(modes * nc, dtype=xp.complex128)
        result.current = xp.zeros(modes * ne, dtype=xp.complex128)
        result.fJ0 = xp.zeros(modes * ne, dtype=xp.complex128)
        for species in range(2):
            result.fluid[species] = build_fluid(
                state.fluid[species],
                species,
                result.BN,
                result.bhat,
                tag,
                context,
            )
            build_non_electric(
                result.fluid[species],
                species,
                result.BN,
                result.bhat,
                context,
            )
            projected = project_wall(
                result.fluid[species].wall,
                result.fluid[species].non_electric,
                context,
            )
            current_samples = sample_apply(
                context.op.I,
                projected.value
                * (
                    context.p.species[species].charge
                    / context.p.species[species].mass
                ),
                context.N,
                context.compute,
            )
            result.fJ0 += forward(current_samples, ne, context)
            result.rho += context.p.species[species].charge * state.fluid[species].n
            result.current += (
                context.p.species[species].charge * result.fluid[species].flux_hat
            )
        result.cache = electrostatic(result.rho, tag, context)
        transverse(result, context)
        for species in range(2):
            finish_fluid(result, species, context)
            result.ledger[species] = power_ledger(result, species, context)

        projected_current = longitudinal(result.current, context)
        result.ELdot = -projected_current / context.p.epsilon0
        result.residuals["longitudinal_projection"] = mode_apply(
            context.op.DJ,
            result.current - projected_current,
            context.compute,
        )
        ddiv = mode_apply(context.op.DJ, result.current, context.compute)
        result.phidot = xp.zeros(modes * nc, dtype=xp.complex128)
        rows = xp.full(
            (nc,),
            context.p.ref.current
            / (context.p.ref.length * context.p.epsilon0),
            dtype=xp.float64,
        )
        columns = xp.full(
            (nc,),
            context.p.ref.electric * context.p.ref.length / context.p.ref.time,
            dtype=xp.float64,
        )
        for stored, matrix in enumerate(context.op.Lphi):
            rhs = -ddiv[stored * nc : (stored + 1) * nc] / context.p.epsilon0
            linear = context.compute.linear(
                matrix,
                rhs,
                rows,
                columns,
                context.p.rank_rtol,
                64 * nc * xp.finfo(xp.float64).eps,
            )
            result.phidot[stored * nc : (stored + 1) * nc] = linear.x
        result.ampere = mode_apply(
            context.op.Cdag, state.B, context.compute
        ) - context.p.mu0 * (result.current - projected_current)

        diagnostics = _diagnostics()
        result.boundary = diagnostics.boundary_ledger(state, result, context)
        result.totals = diagnostics.state_totals(state, result, context)
        result.momentum = diagnostics.momentum(state, result, context)
        diagnostics.check_stage(state, result, context)
        if alias:
            _alias_probe(state, tag, context, result)
        return result
    except Failure as failure:
        retain_failure_evidence(failure, tag, context, state, result, "trial")
        raise
    except Exception as error:
        failure = Failure("NumericalLibrary", str(error))
        retain_failure_evidence(failure, tag, context, state, result, "trial")
        raise failure from error


def _alias_probe(state, tag, context, result):
    xp = _xp(context)
    alternate_context = _copy_context(
        context, N=2 * context.N + 1, spectral_id=context.spectral_id + 1
    )
    alternate_tag = StageTag(
        step=tag.step,
        generation=tag.generation,
        trial=tag.trial,
        spectral_context=alternate_context.spectral_id,
        time=tag.time,
        stage="alias",
        micro_step=tag.micro_step,
        iteration=tag.iteration,
        slab=tag.slab,
    )
    alternate = evaluate_stage(state, alternate_tag, alternate_context, False)
    context.next_source = alternate_context.next_source
    result.checks.extend(alternate.checks)
    diagnostics = _diagnostics()
    measure = getattr(diagnostics, "measure", None)
    gate = getattr(diagnostics, "gate", None)
    if measure is None or gate is None:
        raise Failure("DiagnosticShape", "alias diagnostics API is unavailable")
    names = []
    for species in range(2):
        ident = context.p.species[species].id
        base = result.rhs.fluid[species]
        other = alternate.rhs.fluid[species]
        measure(
            result.checks,
            "alias_n",
            other.n - base.n,
            context.op.M0,
            "1/(m^3*s)",
            tag,
            context,
            ident,
        )
        for component in range(3):
            base_component = base.mom.reshape(
                _modes(context), 3, context.op.Nc
            )[:, component, :].reshape(-1)
            other_component = other.mom.reshape(
                _modes(context), 3, context.op.Nc
            )[:, component, :].reshape(-1)
            measure(
                result.checks,
                "alias_mom",
                other_component - base_component,
                context.op.M0,
                "kg/(m^2*s^2)",
                tag,
                context,
                ident,
                False,
                ("R", "phi", "Z")[component],
            )
        measure(
            result.checks,
            "alias_perp",
            other.perp - base.perp,
            context.op.M0,
            "Pa/s",
            tag,
            context,
            ident,
        )
        if species == 1:
            measure(
                result.checks,
                "alias_parallel",
                other.parallel - base.parallel,
                context.op.M0,
                "Pa/s",
                tag,
                context,
                ident,
            )
        names.extend(["alias_n", "alias_mom"])
        names.extend(["alias_perp"])
        if species == 1:
            names.append("alias_parallel")
    offsets = [
        0,
        len(context.geometry.z_faces.keys),
        len(context.geometry.z_faces.keys) + len(context.geometry.vertices.keys),
        context.op.Nb,
    ]
    mb = xp.real(xp.asarray(context.op.MB)).astype(xp.float64)
    for component in range(3):
        start, stop = offsets[component], offsets[component + 1]
        base = result.rhs.B.reshape(_modes(context), context.op.Nb)[
            :, start:stop
        ].reshape(-1)
        other = alternate.rhs.B.reshape(_modes(context), context.op.Nb)[
            :, start:stop
        ].reshape(-1)
        measure(
            result.checks,
            "alias_B",
            other - base,
            mb[start:stop],
            "T/s",
            tag,
            context,
            "",
            False,
            ("R", "phi", "Z")[component],
        )
        names.append("alias_B")
    gate(result.checks, tuple(dict.fromkeys(names)))


def _retain_common_failure_evidence(
    failure, tag, context, state, result, prefix
):
    """Attach completed checks and stage arrays to a failure without promotion."""
    if getattr(failure, "stage", None) is None:
        failure.stage = tag
    if not hasattr(failure, "checks"):
        failure.checks = []
    if not hasattr(failure, "partial_arrays"):
        failure.partial_arrays = {}
    if result is not None:
        for check in getattr(result, "checks", []):
            if check not in failure.checks:
                failure.checks.append(check)
        for key, value in getattr(result, "residuals", {}).items():
            failure.partial_arrays.setdefault(f"{prefix}.{key}", value)
        if result.cache is not None and result.cache.E is not None:
            failure.partial_arrays.setdefault(f"{prefix}_E", result.cache.E)
    if state is not None:
        failure.partial_arrays.setdefault(f"{prefix}_B", state.B)
        for species, ident in enumerate(("i", "e")):
            fluid = state.fluid[species]
            failure.partial_arrays.setdefault(f"{prefix}_n_{ident}", fluid.n)
            failure.partial_arrays.setdefault(
                f"{prefix}_momentum_{ident}", fluid.mom
            )
            failure.partial_arrays.setdefault(f"{prefix}_p_{ident}", fluid.perp)
            if species == 1:
                failure.partial_arrays.setdefault(
                    f"{prefix}_p_parallel_e", fluid.parallel
                )
    def identity(check):
        return (
            check.name,
            check.population,
            check.component,
            getattr(check, "producer", ""),
            check.region,
            str(check.norm),
            check.tag.stage,
            check.tag.step,
            check.tag.generation,
            check.tag.trial,
            check.tag.spectral_context,
            float(check.tag.time).hex(),
            int(getattr(check.tag, "micro_step", -1)),
            int(getattr(check.tag, "iteration", 0)),
            int(getattr(check.tag, "slab", 0)),
        )

    try:
        required = _diagnostics().stage_requirements(context, tag)
    except Exception:
        required = []
    present = {identity(check) for check in failure.checks}
    for check in required:
        if identity(check) not in present:
            check.available = False
            check.passed = False
            check.reason = f"not run: {failure.code}"
            failure.checks.append(check)
            present.add(identity(check))


def retain_failure_evidence(failure, tag, context, state, result, prefix):
    """Retain model-specific evidence without changing the failure path."""
    if int(getattr(context.p, "electron_inertia", 0)) == 0:
        from .massless import retain_failure_evidence as retain_massless

        return retain_massless(failure, tag, context, state, result, prefix)
    return _retain_common_failure_evidence(
        failure, tag, context, state, result, prefix
    )


def initialize(state, context, cumulative, equilibrium, captured=None):
    """Validate and commit the supplied initial state."""
    if int(getattr(context.p, "electron_inertia", 0)) == 0:
        from .massless import initialize as initialize_massless

        return initialize_massless(
            state, context, cumulative, equilibrium, captured
        )
    if state.generation >= UINT64_MAX:
        raise Failure("CounterOverflow", "state generation exhausted")
    context.next_generation = max(context.next_generation, state.generation + 1)
    tag = StageTag(
        step=state.step,
        generation=state.generation,
        trial=_fresh(context, "next_trial"),
        spectral_context=context.spectral_id,
        time=state.time,
        stage="restart" if state.step else "initial",
    )
    stage = evaluate_stage(state, tag, context, True)
    try:
        diagnostics = _diagnostics()
        for evidence in context.op.setup_checks:
            check = SimpleNamespace(
                name=f"{evidence.name}.mode_{evidence.mode}",
                population="",
                region="all",
                unit=evidence.unit,
                value_unit=evidence.unit,
                reason="",
                component="",
                producer="diagnostics",
                tag=tag,
                raw=evidence.residual,
                reference=evidence.scale,
                threshold=evidence.roundoff_bound,
                norm="value",
                hard=True,
                available=True,
                passed=evidence.residual <= evidence.roundoff_bound,
                worst=0,
                selected=1,
            )
            stage.checks.append(check)
        diagnostics.measure(
            stage.checks,
            "initial_divB",
            mode_apply(context.op.DB, state.B, context.compute),
            context.geometry.wv,
            "T/m",
            tag,
            context,
        )
        diagnostics.measure(
            stage.checks,
            "initial_ampere",
            stage.ampere,
            context.op.M1,
            "T/m",
            tag,
            context,
        )
        if equilibrium:
            for species in range(2):
                ident = context.p.species[species].id
                diagnostics.measure(
                    stage.checks,
                    "equilibrium_n",
                    stage.rhs.fluid[species].n,
                    context.op.M0,
                    "1/(m^3*s)",
                    tag,
                    context,
                    ident,
                )
                momentum_weights = _xp(context).tile(
                    _xp(context)
                    .real(_xp(context).asarray(context.op.M0))
                    .astype(_xp(context).float64),
                    3,
                )
                diagnostics.measure(
                    stage.checks,
                    "equilibrium_mom",
                    stage.rhs.fluid[species].mom,
                    momentum_weights,
                    "kg/(m^2*s^2)",
                    tag,
                    context,
                    ident,
                )
                diagnostics.measure(
                    stage.checks,
                    "equilibrium_perp",
                    stage.rhs.fluid[species].perp,
                    context.op.M0,
                    "Pa/s",
                    tag,
                    context,
                    ident,
                )
                if species == 1:
                    diagnostics.measure(
                        stage.checks,
                        "equilibrium_parallel",
                        stage.rhs.fluid[species].parallel,
                        context.op.M0,
                        "Pa/s",
                        tag,
                        context,
                        ident,
                    )
            diagnostics.measure(
                stage.checks,
                "equilibrium_B",
                stage.rhs.B,
                context.op.MB,
                "T/s",
                tag,
                context,
            )
        diagnostics.gate(stage.checks, ("initial_divB", "initial_ampere"))
        cumulative.baseline = SimpleNamespace(
            divB=mode_apply(context.op.DB, state.B, context.compute),
            totals=stage.totals,
            momentum=stage.momentum,
            time=state.time,
            step=state.step,
        )
        cumulative.last_step = state.step
        cumulative.last_time = state.time
        if not hasattr(cumulative, "accepted_intervals"):
            cumulative.accepted_intervals = 0
        committed = SimpleNamespace(state=state, cache=stage.cache, accepted=True)
        if captured is not None:
            capture_update = getattr(captured, "update", None)
            if capture_update is not None:
                captured.update(vars(stage))
        return committed
    except Failure as failure:
        retain_failure_evidence(failure, tag, context, state, stage, "trial")
        raise


def closed_reference(
    context,
    density,
    ion_pressure,
    electron_pressure,
    B_reference,
    velocity=0,
):
    """Construct the native axisymmetric closed reference state."""
    values = (density, ion_pressure, electron_pressure, B_reference, velocity)
    if any(not math.isfinite(value) for value in values):
        raise Failure("Fixture", "invalid reference parameters")
    if density <= 0 or ion_pressure <= 0 or electron_pressure <= 0 or B_reference <= 0:
        raise Failure("Fixture", "invalid reference parameters")
    if context.p.species[1].charge == 0:
        raise Failure("Fixture", "electron charge must be nonzero")
    xp = _xp(context)
    nc = context.op.Nc
    state = allocate_state(context)
    for species in range(2):
        number = (
            density
            if species == 0
            else -context.p.species[0].charge * density
            / context.p.species[1].charge
        )
        zero = context.p.K * nc
        state.fluid[species].n[zero : zero + nc] = number
        state.fluid[species].perp[zero : zero + nc] = (
            ion_pressure if species == 0 else electron_pressure
        )
        if species == 1:
            state.fluid[species].parallel[zero : zero + nc] = electron_pressure
        if velocity and context.p.K:
            for mode in (-1, 1):
                base = (context.p.K + mode) * 3 * nc
                second = base + nc
                state.fluid[species].mom[second : second + nc] = (
                    context.p.species[species].mass * number * velocity / 2.0
                )
    topology = xp.asarray(context.op.topology_flux_row)
    if topology.ndim == 1:
        harmonic = xp.conj(topology) / (
            xp.real(xp.asarray(context.op.MB)).astype(xp.float64)
        )
    else:
        harmonic = xp.conj(topology[0]) / (
            xp.real(xp.asarray(context.op.MB)).astype(xp.float64)
        )
    observed = context.compute.mv(context.op.QB, harmonic)
    scale = float(xp.max(xp.abs(observed)))
    if not math.isfinite(scale) or scale <= 0:
        raise Failure("Fixture", "closed reference magnetic chain is empty")
    harmonic *= B_reference / scale
    zero = context.p.K * context.op.Nb
    state.B[zero : zero + context.op.Nb] = harmonic
    state.time = context.p.start_time
    state.step = 0
    state.generation = 1
    return state
