"""Closed-grid wall momentum elimination for the A0 physics stage."""

from types import SimpleNamespace

from .types import Failure


def _xp(context):
    return context.compute.xp


def _size(value, expected, name):
    if tuple(value.shape) != (expected,):
        raise Failure("Shape", f"{name} has an invalid extent")


def _as_complex_xp(value, context):
    return context.compute.xp.asarray(value, dtype=context.compute.xp.complex128)


def _hermitian(matrix, context):
    xp = _xp(context)
    return xp.conj(xp.swapaxes(matrix, -1, -2))


def build_wall(n, species, tag, context):
    """Build the fixed-density normal momentum projection for every sample."""
    xp = _xp(context)
    nc = context.op.Nc
    _size(n, context.N * nc, "wall density")
    if context.N <= 0:
        raise Failure("Shape", "wall sample count is not positive")
    density = xp.real(_as_complex_xp(n, context)).reshape(context.N, nc)
    if not bool(xp.all(xp.isfinite(density))):
        raise Failure("Finite", "wall density contains non-finite values")
    if not bool(xp.all(density > species.n_min)):
        raise Failure("Domain", "wall density not positive")

    m0 = xp.real(xp.asarray(context.op.M0)).astype(xp.float64)
    entity_density = xp.tile(density, (1, 3))
    winv = species.mass * entity_density / xp.tile(m0, 3)
    if not bool(xp.all(xp.isfinite(winv) & (winv > 0))):
        raise Failure("Domain", "wall inverse inertial weight is not positive")

    wall_matrix = xp.asarray(context.op.T, dtype=xp.complex128)
    wall_adjoint = _hermitian(wall_matrix, context)
    n_ref = (
        context.p.ref.ni
        if species.id == context.p.species[0].id
        else context.p.ref.ne
    )
    row_scale = species.mass * n_ref * context.p.ref.velocity / context.p.ref.time
    col_scale = row_scale / (
        species.mass * n_ref / context.p.ref.length**3
    )
    samples = []
    for sample in range(context.N):
        weights = winv[sample]
        weighted_adjoint = wall_adjoint * weights[:, None]
        scaled = wall_matrix @ weighted_adjoint
        factor = None
        if wall_matrix.shape[0]:
            scaled_matrix = scaled * (col_scale / row_scale)
            singular = context.compute.singular_values(scaled_matrix)
            if singular.size == 0 or not bool(
                singular[0] > 0
                and singular[-1] > context.p.rank_rtol * singular[0]
            ):
                raise Failure("WallRank", "scaled wall matrix deficient")
            if context.compute.name == "cpu":
                factor = context.compute.factor(scaled_matrix)
        samples.append(
            SimpleNamespace(
                S=scaled,
                lu=factor,
                winv=weights,
                row_scale=row_scale,
                col_scale=col_scale,
            )
        )
    return SimpleNamespace(samples=samples, tag=tag)


def project_wall(wall, r, context):
    """Remove wall-normal momentum and return the physical wall reaction."""
    xp = _xp(context)
    nc = context.op.Nc
    entities = 3 * nc
    _size(r, context.N * entities, "wall response")
    if len(wall.samples) != context.N:
        raise Failure("Shape", "wall response sample count")
    sampled = _as_complex_xp(r, context).reshape(context.N, entities)
    value = sampled.copy()
    reaction = xp.zeros_like(sampled)
    wall_matrix = xp.asarray(context.op.T, dtype=xp.complex128)
    wall_adjoint = _hermitian(wall_matrix, context)
    rows = wall_matrix.shape[0]
    full_wall_matrix = xp.asarray(context.op.T_full, dtype=xp.complex128)
    multiplier = xp.zeros((context.N, rows), dtype=xp.complex128)
    multiplier_residual = xp.zeros_like(multiplier)

    def normal_solve(record, rhs):
        if record.lu is not None:
            return xp.asarray(
                context.compute.solve(record.lu, rhs / record.row_scale),
                dtype=xp.complex128,
            ) * record.col_scale
        linear = context.compute.linear(
            record.S,
            rhs,
            xp.full((rows,), record.row_scale, dtype=xp.float64),
            xp.full((rows,), record.col_scale, dtype=xp.float64),
            context.p.rank_rtol,
            1.0e-10,
        )
        return xp.asarray(linear.x, dtype=xp.complex128)

    epsilon = float(xp.finfo(xp.float64).eps)
    minimum_scale = float(xp.finfo(xp.float64).tiny)
    matrix_norm = (
        float(xp.max(xp.sum(xp.abs(wall_matrix), axis=1)).item())
        if rows
        else 0.0
    )

    def forward_bound(sample_value, trace_norm=matrix_norm):
        value_scale = context.compute.max_abs(sample_value)
        return (
            256.0
            * epsilon
            * trace_norm
            * max(value_scale, minimum_scale)
            + minimum_scale
        )

    for sample, record in enumerate(wall.samples):
        if rows == 0:
            continue
        rhs = wall_matrix @ sampled[sample]
        lam = normal_solve(record, rhs)
        sample_reaction = -(wall_adjoint @ lam) * record.winv
        value[sample] += sample_reaction
        forward = wall_matrix @ value[sample]
        converged = False
        for _ in range(5):
            forward_scale = context.compute.max_abs(forward)
            if forward_scale <= forward_bound(value[sample]):
                converged = True
                break
            correction = normal_solve(record, forward)
            correction_reaction = -(wall_adjoint @ correction) * record.winv
            corrected_value = value[sample] + correction_reaction
            corrected_forward = wall_matrix @ corrected_value
            corrected_scale = context.compute.max_abs(corrected_forward)
            if corrected_scale >= forward_scale:
                raise Failure(
                    "WallProjection",
                    "forward wall trace refinement stagnated",
                )
            value[sample] = corrected_value
            sample_reaction = sample_reaction + correction_reaction
            lam = lam + correction
            forward = corrected_forward
            if corrected_scale <= forward_bound(value[sample]):
                converged = True
                break
        if not converged:
            raise Failure(
                "WallProjection",
                "forward wall trace refinement did not converge in five corrections",
            )
        reaction[sample] = sample_reaction
        multiplier[sample] = lam
        multiplier_residual[sample] = record.S @ lam - rhs
    if not bool(
        xp.all(xp.isfinite(value))
        and xp.all(xp.isfinite(reaction))
        and xp.all(xp.isfinite(multiplier))
        and xp.all(xp.isfinite(multiplier_residual))
    ):
        raise Failure("Finite", "wall projection contains non-finite values")
    if full_wall_matrix.shape[0]:
        full_norm = float(
            xp.max(xp.sum(xp.abs(full_wall_matrix), axis=1)).item()
        )
        for sample_value in value:
            full_trace = full_wall_matrix @ sample_value
            if context.compute.max_abs(full_trace) > forward_bound(
                sample_value, full_norm
            ):
                raise Failure(
                    "WallProjection",
                    "full wall trace exceeds the projected-output roundoff bound",
                )
    return SimpleNamespace(
        value=value.reshape(-1),
        reaction=reaction.reshape(-1),
        multiplier=multiplier.reshape(-1),
        multiplier_residual=multiplier_residual.reshape(-1),
    )
