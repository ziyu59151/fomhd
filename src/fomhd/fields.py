"""Electrostatic and Darwin field solves for the A0 stage."""

from types import SimpleNamespace

from .core import forward, mode_apply, sample_apply
from .fluid import electric_response
from .types import Failure


def _xp(context):
    return context.compute.xp


def _modes(context):
    return 2 * context.p.K + 1


def _size(value, expected, name):
    if tuple(value.shape) != (expected,):
        raise Failure("Shape", f"{name} has an invalid extent")


def _finite(value, context):
    return bool(_xp(context).all(_xp(context).isfinite(value)))


def _hermitian_defect(values, entities, context):
    xp = _xp(context)
    stored = values.reshape(_modes(context), entities)
    defect = xp.max(xp.abs(xp.imag(stored[context.p.K])))
    for mode in range(1, context.p.K + 1):
        difference = stored[context.p.K - mode] - xp.conj(
            stored[context.p.K + mode]
        )
        defect = xp.maximum(defect, xp.max(xp.abs(difference)))
    return defect


def _checked_linear(matrix, rhs, row_scale, column_scale, context, name, tol):
    try:
        return context.compute.linear(
            matrix,
            rhs,
            row_scale,
            column_scale,
            context.p.rank_rtol,
            tol,
        )
    except Exception as error:
        raise Failure("LinearSolve", f"{name}: {error}") from error


def electrostatic(charge_hat, tag, context):
    """Solve the complete discrete Gauss/Poisson system in physical units."""
    xp = _xp(context)
    nc = context.op.Nc
    _size(charge_hat, _modes(context) * nc, "charge density")
    if not _finite(charge_hat, context):
        raise Failure("Finite", "charge density contains non-finite values")
    scale = max(1.0, float(xp.max(xp.abs(charge_hat))))
    if float(_hermitian_defect(charge_hat, nc, context)) > 2.0e-11 * scale:
        raise Failure("Hermitian", "charge density is not physical")

    phi = xp.zeros_like(charge_hat)
    electric = xp.zeros(
        _modes(context) * context.op.Ne, dtype=xp.complex128
    )
    poisson_residual = xp.zeros_like(charge_hat)
    solves = []
    rows = xp.full(
        (nc,), context.p.ref.charge / context.p.epsilon0, dtype=xp.float64
    )
    columns = xp.full(
        (nc,), context.p.ref.electric * context.p.ref.length, dtype=xp.float64
    )
    for stored, mode in enumerate(range(-context.p.K, context.p.K + 1)):
        start = stored * nc
        rhs = charge_hat[start : start + nc] / context.p.epsilon0
        linear = _checked_linear(
            context.op.Lphi[mode + context.p.K],
            rhs,
            rows,
            columns,
            context,
            "Poisson solve",
            64 * nc * xp.finfo(xp.float64).eps,
        )
        potential = xp.asarray(linear.x, dtype=xp.complex128)
        phi[start : start + nc] = potential
        poisson_residual[start : start + nc] = (
            context.op.Lphi[mode + context.p.K] @ potential - rhs
        )
        electric[
            stored * context.op.Ne : (stored + 1) * context.op.Ne
        ] = -(context.op.G[mode + context.p.K] @ potential)
        solves.append(linear)
    potential_scale = max(1.0, float(xp.max(xp.abs(phi))))
    electric_scale = max(1.0, float(xp.max(xp.abs(electric))))
    if (
        float(_hermitian_defect(phi, nc, context))
        > 2.0e-11 * potential_scale
        or float(_hermitian_defect(electric, context.op.Ne, context))
        > 2.0e-11 * electric_scale
    ):
        raise Failure("Hermitian", "Poisson physical field is not Hermitian")
    return SimpleNamespace(
        phi=phi,
        EL=electric,
        ET=xp.zeros(0, dtype=xp.complex128),
        E=xp.zeros(0, dtype=xp.complex128),
        multiplier=xp.zeros(0, dtype=xp.complex128),
        poisson_residual=poisson_residual,
        tag=tag,
        poisson_solves=solves,
        transverse_solve=None,
    )


def longitudinal(current_hat, context, tag=None):
    """Return the exact discrete gradient part of an edge current."""
    xp = _xp(context)
    nc = context.op.Nc
    ne = context.op.Ne
    _size(current_hat, _modes(context) * ne, "current")
    if not _finite(current_hat, context):
        raise Failure("Finite", "current contains non-finite values")
    result = xp.zeros_like(current_hat)
    rows = xp.full(
        (nc,), context.p.ref.current / context.p.ref.length, dtype=xp.float64
    )
    columns = xp.full(
        (nc,), context.p.ref.current * context.p.ref.length, dtype=xp.float64
    )
    for mode in range(-context.p.K, context.p.K + 1):
        stored = mode + context.p.K
        source = current_hat[stored * ne : (stored + 1) * ne]
        rhs = context.op.DJ[stored] @ source
        linear = _checked_linear(
            context.op.Lphi[stored],
            rhs,
            rows,
            columns,
            context,
            "longitudinal projection solve",
            64 * nc * xp.finfo(xp.float64).eps,
        )
        result[stored * ne : (stored + 1) * ne] = -(
            context.op.G[stored] @ xp.asarray(linear.x, dtype=xp.complex128)
        )
    return result


def _repeated(values, count, xp):
    return xp.tile(xp.real(xp.asarray(values)).astype(xp.float64), count)


def _materialize(action, size, context):
    xp = _xp(context)
    estimated = size * size * 16
    if context.p.memory_limit and estimated > context.p.memory_limit:
        raise Failure("Memory", "AE materialization exceeds memory_limit")
    try:
        matrix = xp.empty((size, size), dtype=xp.complex128)
        for column in range(size):
            basis = xp.zeros(size, dtype=xp.complex128)
            basis[column] = 1.0
            matrix[:, column] = action(basis)
        return matrix
    except Failure:
        raise
    except Exception as error:
        raise Failure("Memory", f"AE materialization failed: {error}") from error


def transverse(stage, context):
    """Solve the full constrained Darwin system and update field caches."""
    xp = _xp(context)
    modes = _modes(context)
    ne = context.op.Ne
    edge_size = modes * ne
    _size(stage.fJ0, edge_size, "no-electric current")
    _size(stage.cache.EL, edge_size, "electrostatic field")
    if not _finite(stage.fJ0, context) or not _finite(stage.cache.EL, context):
        raise Failure("Finite", "Darwin source field is non-finite")
    for species in range(2):
        fluid = stage.fluid[species]
        if fluid.n is None or not fluid.wall.samples:
            raise Failure("Stage", "fluid context is not prepared")

    ae_bytes = edge_size * edge_size * 16
    system_size = edge_size + int(context.op.Bc.shape[0])
    darwin_bytes = (
        4 * edge_size * edge_size + 2 * system_size * system_size
    ) * 16
    if context.p.memory_limit and max(ae_bytes, darwin_bytes) > context.p.memory_limit:
        raise Failure("Memory", "Darwin dense workspace exceeds memory_limit")

    def action(electric_hat):
        result = xp.zeros_like(electric_hat)
        for species in range(2):
            physical = context.p.species[species]
            force = electric_response(
                stage.fluid[species], species, electric_hat, context
            )
            samples = sample_apply(
                context.op.I, force, context.N, context.compute
            )
            result += forward(samples, ne, context) * (
                physical.charge / physical.mass
            )
        return result

    ae = _materialize(action, edge_size, context)
    m1_base = xp.real(xp.asarray(context.op.M1)).astype(xp.float64)
    m1 = _repeated(m1_base, modes, xp)
    curl = xp.zeros((edge_size, edge_size), dtype=xp.complex128)
    for stored in range(modes):
        c = xp.asarray(context.op.C[stored], dtype=xp.complex128)
        cdag = xp.asarray(context.op.Cdag[stored], dtype=xp.complex128)
        weighted_curl = m1_base[:, None] * (cdag @ c)
        curl[
            stored * ne : (stored + 1) * ne,
            stored * ne : (stored + 1) * ne,
        ] = weighted_curl / context.p.mu0
    weighted_response = m1[:, None] * ae
    adjoint = weighted_response.T.conj()
    weighted_scale = max(1.0, float(xp.max(xp.abs(weighted_response))))
    if float(xp.max(xp.abs(weighted_response - adjoint))) > (
        64 * edge_size * xp.finfo(xp.float64).eps * weighted_scale
    ):
        raise Failure("Adjoint", "M1*AE is not weighted self-adjoint")
    dynamic = curl + weighted_response

    bc = xp.asarray(context.op.Bc, dtype=xp.complex128)
    constraints = bc.shape[0]
    kkt = xp.zeros((system_size, system_size), dtype=xp.complex128)
    kkt[:edge_size, :edge_size] = dynamic
    kkt[:edge_size, edge_size:] = bc.T.conj()
    kkt[edge_size:, :edge_size] = bc
    trial_rhs = stage.fJ0 + action(stage.cache.EL)
    rhs = xp.zeros(system_size, dtype=xp.complex128)
    rhs[:edge_size] = -(m1 * trial_rhs)

    current_rate = context.p.ref.current / context.p.ref.time
    lambda_scale = current_rate * context.p.ref.length**4
    row_scale = xp.empty(system_size, dtype=xp.float64)
    column_scale = xp.empty(system_size, dtype=xp.float64)
    row_scale[:edge_size] = m1 * current_rate
    column_scale[:edge_size] = context.p.ref.electric
    row_scale[edge_size:] = context.p.ref.electric / context.p.ref.length
    column_scale[edge_size:] = lambda_scale
    try:
        linear = _checked_linear(
            kkt,
            rhs,
            row_scale,
            column_scale,
            context,
            "Darwin KKT solve",
            64 * system_size * xp.finfo(xp.float64).eps,
        )
    except Failure as error:
        try:
            scaled = kkt * column_scale[None, :] / row_scale[:, None]
            singular = xp.linalg.svd(scaled, compute_uv=False)
            detail = (
                f"{error}; AE={float(xp.max(xp.abs(ae)))} "
                f"dynamic={float(xp.max(xp.abs(dynamic)))} "
                f"constraints={constraints} smax={float(singular[0])} "
                f"smin={float(singular[-1])}"
            )
        except Exception:
            detail = str(error)
        raise Failure("DarwinRank", detail) from error

    solution = xp.asarray(linear.x, dtype=xp.complex128)
    et = solution[:edge_size]
    multiplier = solution[edge_size:]
    dynamic_residual = dynamic @ et + bc.T.conj() @ multiplier - rhs[:edge_size]
    full_divergence = xp.asarray(
        context.op.Bc_full, dtype=xp.complex128
    ) @ et
    strong = dynamic_residual / m1
    if (
        float(_hermitian_defect(et, ne, context))
        > 512
        * xp.finfo(xp.float64).eps
        * max(context.p.ref.electric, float(xp.max(xp.abs(et))))
        or float(_hermitian_defect(stage.cache.EL + et, ne, context))
        > 512
        * xp.finfo(xp.float64).eps
        * max(1.0, float(xp.max(xp.abs(stage.cache.EL + et))))
    ):
        raise Failure("Hermitian", "Darwin electric field is not physical")

    stage.AE = ae
    stage.cache.ET = et
    stage.cache.E = stage.cache.EL + et
    stage.cache.multiplier = multiplier
    stage.cache.transverse_solve = linear
    stage.cache.tag = stage.tag
    stage.residuals["transverse_kkt"] = xp.asarray(linear.residual)
    stage.residuals["transverse_dynamic"] = dynamic_residual
    stage.residuals["transverse_divergence_full"] = full_divergence
    stage.residuals["darwin_dynamic"] = strong
    stage.rhs.B = -mode_apply(context.op.C, et, context.compute)
    stage.residuals["transverse_ct"] = mode_apply(
        context.op.DB, stage.rhs.B, context.compute
    )
