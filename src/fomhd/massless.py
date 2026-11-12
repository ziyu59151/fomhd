"""Massless, exactly quasineutral electron branch with constrained Ohm fields."""

from __future__ import annotations

import copy
import math
import sys
from types import SimpleNamespace

from .box_boundary import electron_pressure_traction, magnetic_divergence
from .core import (
    _hermitian_check,
    allocate_state,
    check_retained,
    forward,
    inverse,
    mode_apply,
    sample_apply,
)
from .fluid import (
    _component_flux,
    _pressure_tensor,
    _sampled_divergence,
    _transport_flux,
    _warp_magnetic_force,
    build_fluid,
    build_non_electric,
    finish_fluid,
    gradient_adjoint,
    velocity_gradient,
)
from .stage import _retain_common_failure_evidence
from .types import (
    Failure,
    FluidState,
    StageTag,
    State,
    field_cache_record,
    stage_result_record,
)
from .wall import _diagonal_constraint_normal, build_wall
from .wall_cache import apply_wall_cache, get_wall_cache


UINT64_MAX = (1 << 64) - 1
MODEL_NAME = "massless_quasineutral_cgl_b"


def _xp(context):
    return context.compute.xp


def _modes(context):
    return 2 * context.p.K + 1


def _fresh(context, name):
    current = int(getattr(context, name))
    if current >= UINT64_MAX:
        raise Failure("CounterOverflow", "stage identity exhausted")
    setattr(context, name, current + 1)
    return current


def _require_size(value, expected, name):
    if value is None or tuple(value.shape) != (int(expected),):
        raise Failure("Shape", f"{name} has an invalid extent")


def _finite(value, context, name):
    xp = _xp(context)
    if value is None or not bool(xp.all(xp.isfinite(value)).item()):
        raise Failure("Finite", f"{name} contains non-finite values")


def _max_abs(value, context):
    xp = _xp(context)
    if value is None or value.size == 0:
        return 0.0
    return float(xp.max(xp.abs(value)).item())


def _electroweak_parameters(context):
    ion = context.p.species[0]
    electron = context.p.species[1]
    e = -float(electron.charge)
    if float(ion.charge) <= 0.0:
        raise Failure("Model", "the primary ion charge must be positive")
    if not math.isfinite(e) or e <= 0.0:
        raise Failure("Model", "the electron charge must be negative")
    if not math.isfinite(float(electron.mass)) or float(electron.mass) <= 0.0:
        raise Failure("Model", "physical electron mass metadata must be positive")
    return e, float(ion.charge) / e


def _require_model(context):
    if int(getattr(context.p, "electron_inertia", 0)) != 0:
        raise Failure("Model", "massless branch requires electron_inertia=0")
    _electroweak_parameters(context)


def _signed_wall_extraction(context, mode_index):
    xp = _xp(context)
    nc = context.op.Nc
    nr = len(context.geometry.r_faces.keys)
    walls = context.geometry.walls
    result = xp.zeros((len(walls), context.op.Ne), dtype=xp.complex128)
    for row, wall in enumerate(walls):
        edge = (
            wall.face
            if wall.component == "r_face"
            else nr + nc + wall.face
        )
        result[row, edge] = float(wall.sign)
    return result


def _wall_extractions(context):
    if getattr(context.geometry, "lcfs_polygon", None) is not None:
        return get_wall_cache(context).extraction
    return [
        _signed_wall_extraction(context, stored)
        for stored in range(_modes(context))
    ]


def _project_diagonal_constraint(
    source,
    T,
    winv,
    diagonal,
    row_scale,
    column_scale,
    context,
):
    """Apply the checked pointwise projection for an isolated-row normal."""
    xp = _xp(context)
    scaled = diagonal * (column_scale / row_scale)
    largest = xp.max(scaled, axis=1)
    smallest = xp.min(scaled, axis=1)
    deficient = (
        ~xp.isfinite(largest)
        | ~xp.isfinite(smallest)
        | (largest <= 0.0)
        | (smallest <= context.p.rank_rtol * largest)
    )
    if bool(xp.any(deficient).item()):
        sample = int(xp.argmax(deficient.astype(xp.int8)).item())
        raise Failure(
            "WallRank",
            "selected wall reconstruction rows are deficient at "
            f"time row {sample}",
        )

    right = xp.matmul(source, T.T)
    multiplier = (
        (right / row_scale) / scaled
    ) * column_scale
    applied = diagonal * multiplier
    residual = applied - right

    def active_rows():
        scale = xp.maximum(
            xp.maximum(
                xp.max(xp.abs(right), axis=1),
                xp.max(xp.abs(applied), axis=1),
            ),
            1.0e-300,
        )
        return xp.any(
            xp.abs(residual) > 512.0 * sys.float_info.epsilon * scale[:, None],
            axis=1,
        )

    active = active_rows()
    for _ in range(5):
        if not bool(xp.any(active).item()):
            break
        correction = (
            (-residual / row_scale) / scaled
        ) * column_scale
        multiplier = xp.where(
            active[:, None],
            multiplier + correction,
            multiplier,
        )
        applied = diagonal * multiplier
        residual = applied - right
        active = active_rows()
    if bool(xp.any(active).item()):
        sample = int(xp.argmax(active.astype(xp.int8)).item())
        raise Failure(
            "WallProjection",
            "electron flux constraint refinement did not converge at "
            f"time row {sample}",
        )

    correction = winv * xp.matmul(multiplier, xp.conj(T))
    output = source - correction
    applied = diagonal * multiplier
    residual = applied - right
    final_scale = xp.maximum(
        xp.maximum(
            xp.max(xp.abs(right), axis=1),
            xp.max(xp.abs(applied), axis=1),
        ),
        1.0e-300,
    )
    rejected = xp.any(
        xp.abs(residual)
        > 4096.0 * sys.float_info.epsilon * final_scale[:, None],
        axis=1,
    )
    if bool(xp.any(rejected).item()):
        sample = int(xp.argmax(rejected.astype(xp.int8)).item())
        raise Failure(
            "WallProjection",
            "electron flux constraint residual is too large at "
            f"time row {sample}",
        )
    return output


def _constraint_project(nodal, density, context):
    """Apply Q pointwise using all selected wall rows."""
    xp = _xp(context)
    nc = context.op.Nc
    entities = 3 * nc
    _require_size(nodal, context.N * entities, "constraint projection input")
    _require_size(density, context.N * nc, "constraint projection density")
    _finite(nodal, context, "constraint projection input")
    _finite(density, context, "constraint projection density")
    density_samples = xp.real(density.reshape(context.N, nc))
    if not bool(
        xp.all(xp.isfinite(density_samples) & (density_samples > 0.0)).item()
    ):
        raise Failure("Domain", "electron density is not positive")
    m0 = xp.real(xp.asarray(context.op.M0)).astype(xp.float64)
    winv = xp.tile(density_samples / m0[None, :], (1, 3))
    T = xp.asarray(context.op.T, dtype=xp.complex128)
    T_h = xp.conj(T.T)
    source = xp.asarray(nodal, dtype=xp.complex128).reshape(
        context.N, entities
    )
    length = float(context.p.ref.length)
    row_scale = 1.0 / length**3
    column_scale = 1.0 / length**3
    if T.shape[0] == 0:
        return source.copy().reshape(-1)
    diagonal = _diagonal_constraint_normal(T, winv, xp)
    if diagonal is not None:
        output = _project_diagonal_constraint(
            source,
            T,
            winv,
            diagonal,
            row_scale,
            column_scale,
            context,
        )
    else:
        output = xp.empty_like(source)
        for sample in range(context.N):
            weighted = T * winv[sample][None, :]
            normal = weighted @ T_h
            scaled = normal * (column_scale / row_scale)
            singular = context.compute.singular_values(scaled)
            if (
                singular.size == 0
                or float(singular[0].item()) <= 0.0
                or float(singular[-1].item())
                <= context.p.rank_rtol * float(singular[0].item())
            ):
                raise Failure(
                    "WallRank",
                    "selected wall reconstruction rows are deficient",
                )
            right = T @ source[sample]
            factor = context.compute.factor(scaled)
            multiplier = (
                context.compute.solve(factor, right / row_scale)
                * column_scale
            )
            for _ in range(5):
                normal_residual = normal @ multiplier - right
                residual_scale = max(
                    _max_abs(right, context),
                    _max_abs(normal @ multiplier, context),
                    1.0e-300,
                )
                if _max_abs(normal_residual, context) <= (
                    512.0 * sys.float_info.epsilon * residual_scale
                ):
                    break
                correction = (
                    context.compute.solve(
                        factor, -normal_residual / row_scale
                    )
                    * column_scale
                )
                multiplier = multiplier + correction
            correction = winv[sample] * (T_h @ multiplier)
            output[sample] = source[sample] - correction
            residual = normal @ multiplier - right
            limit = 4096.0 * sys.float_info.epsilon * max(
                _max_abs(right, context),
                _max_abs(normal @ multiplier, context),
                1.0e-300,
            )
            if _max_abs(residual, context) > limit:
                raise Failure(
                    "WallProjection",
                    "electron flux constraint residual is too large",
                )
    full = sample_apply(
        context.op.T_full,
        output.reshape(-1),
        context.N,
        context.compute,
    )
    if _max_abs(full, context) > 4096.0 * sys.float_info.epsilon * max(
        _max_abs(output, context),
        _max_abs(source, context),
        1.0e-300,
    ):
        raise Failure("WallProjection", "full wall trace rejects constrained velocity")
    return output.reshape(-1)


def _derive_electron(state, context):
    _require_model(context)
    xp = _xp(context)
    nc = context.op.Nc
    ne = context.op.Ne
    modes = _modes(context)
    for species in range(2):
        _require_size(state.fluid[species].n, modes * nc, "density")
        _require_size(state.fluid[species].mom, modes * 3 * nc, "momentum")
        _require_size(state.fluid[species].perp, modes * nc, "pressure")
    _finite(state.fluid[0].n, context, "ion density")
    _finite(state.fluid[0].mom, context, "ion momentum")
    _finite(state.fluid[0].perp, context, "ion pressure")
    _finite(state.fluid[1].perp, context, "electron perpendicular pressure")
    _finite(state.fluid[1].parallel, context, "electron parallel pressure")
    _finite(state.B, context, "magnetic field")

    BN = inverse(
        sample_apply(context.op.QB, state.B, modes, context.compute),
        3 * nc,
        context,
        True,
    )
    bn = BN.reshape(context.N, 3, nc)
    magnitude = xp.sqrt(xp.sum(xp.abs(bn) ** 2, axis=1))
    if not bool(
        xp.all(xp.isfinite(magnitude) & (magnitude > context.p.B_min)).item()
    ):
        raise Failure("Magnetization", "nodal B is below the declared floor")
    bhat = (bn / magnitude[:, None, :]).reshape(-1)

    tag = StageTag(
        step=state.step,
        generation=state.generation,
        trial=0,
        spectral_context=getattr(context, "spectral_id", 0),
        time=state.time,
        stage="canonicalize",
    )
    ion = build_fluid(state.fluid[0], 0, BN, bhat, tag, context)
    build_non_electric(ion, 0, BN, bhat, context)

    electron_charge = float(context.p.species[1].charge)
    e = -electron_charge
    qi = float(context.p.species[0].charge)
    current = mode_apply(context.op.Cdag, state.B, context.compute) / context.p.mu0
    electron_flux_hat = (
        qi * ion.flux_hat - current
    ) / e
    electron_density_hat = (qi / e) * state.fluid[0].n
    electron_density = inverse(electron_density_hat, nc, context, True)
    electron_face_density = _component_flux(electron_density, context)
    sample_density = electron_density.reshape(context.N, nc)
    if not bool(
        xp.all(
            xp.isfinite(sample_density)
            & (xp.real(sample_density) > context.p.species[1].n_min)
        ).item()
    ):
        raise Failure("Domain", "derived electron density is not positive")
    if not bool(
        xp.all(
            xp.isfinite(electron_face_density).reshape(context.N, ne)
            & (
                xp.real(electron_face_density.reshape(context.N, ne))
                > context.p.species[1].face_min
            )
        ).item()
    ):
        raise Failure("Domain", "derived face electron density is not positive")
    electron_flux = inverse(electron_flux_hat, ne, context, True)
    h_flux = sample_apply(
        context.op.H,
        electron_flux,
        context.N,
        context.compute,
    )
    constrained_momentum = _constraint_project(
        h_flux, electron_density, context
    )
    electron_velocity = (
        constrained_momentum.reshape(context.N, 3, nc)
        / electron_density.reshape(context.N, 1, nc)
    ).reshape(-1)
    compatibility_momentum = forward(
        xp.real(
            xp.asarray(electron_density).reshape(context.N, 1, nc)
            * electron_velocity.reshape(context.N, 3, nc)
        ).reshape(-1),
        3 * nc,
        context,
    )
    compatibility_momentum = xp.asarray(
        compatibility_momentum, dtype=xp.complex128
    ) * float(context.p.species[1].mass)
    return SimpleNamespace(
        BN=BN,
        bhat=bhat,
        ion=ion,
        current=current,
        electron_density_hat=electron_density_hat,
        electron_density=electron_density,
        electron_face_density=electron_face_density,
        electron_flux_hat=electron_flux_hat,
        electron_flux=electron_flux,
        electron_velocity=electron_velocity,
        compatibility_momentum=compatibility_momentum,
        tag=tag,
    )


def _copy_state(state, context, electron_density, compatibility_momentum):
    xp = _xp(context)
    fluid = []
    for species in range(2):
        source = state.fluid[species]
        fluid.append(
            FluidState(
                n=(
                    xp.asarray(electron_density, dtype=xp.complex128).copy()
                    if species
                    else xp.asarray(source.n, dtype=xp.complex128).copy()
                ),
                mom=(
                    xp.asarray(
                        compatibility_momentum, dtype=xp.complex128
                    ).copy()
                    if species
                    else xp.asarray(source.mom, dtype=xp.complex128).copy()
                ),
                perp=xp.asarray(source.perp, dtype=xp.complex128).copy(),
                parallel=xp.asarray(source.parallel, dtype=xp.complex128).copy(),
            )
        )
    return State(
        fluid=(fluid[0], fluid[1]),
        B=xp.asarray(state.B, dtype=xp.complex128).copy(),
        step=state.step,
        generation=state.generation,
        time=state.time,
    )


def canonicalize_state(state, context):
    """Return a new state with only the derived electron compatibility fields changed."""
    derived = _derive_electron(state, context)
    return _copy_state(
        state,
        context,
        derived.electron_density_hat,
        derived.compatibility_momentum,
    )


def _check_retained_massless(state, context):
    """Validate independent state and derive-only electron compatibility fields."""
    xp = _xp(context)
    independent_electron = FluidState(
        n=state.fluid[1].n,
        mom=xp.zeros_like(state.fluid[1].mom),
        perp=state.fluid[1].perp,
        parallel=state.fluid[1].parallel,
    )
    probe = State(
        fluid=(state.fluid[0], independent_electron),
        B=state.B,
        step=state.step,
        generation=state.generation,
        time=state.time,
    )
    check_retained(probe, context)
    _hermitian_check(
        state.fluid[1].mom,
        3 * context.op.Nc,
        context,
        "electron compatibility momentum",
    )
    ratio = _electroweak_parameters(context)[1]
    density_defect = state.fluid[1].n - ratio * state.fluid[0].n
    scale = max(
        _max_abs(state.fluid[1].n, context),
        _max_abs(state.fluid[0].n, context),
        1.0e-300,
    )
    if _max_abs(density_defect, context) > (
        4096.0 * sys.float_info.epsilon * scale
    ):
        raise Failure(
            "CanonicalState",
            "electron density is not the exact quasineutral derivation",
        )


def _electron_fluid(derived, state, tag, context):
    xp = _xp(context)
    nc = context.op.Nc
    electron = context.p.species[1]
    density = derived.electron_density
    velocity = derived.electron_velocity
    pressure = inverse(state.fluid[1].perp, nc, context, True)
    parallel = inverse(state.fluid[1].parallel, nc, context, True)
    if not bool(
        xp.all(
            xp.isfinite(pressure).reshape(context.N, nc)
            & (xp.real(pressure).reshape(context.N, nc) > electron.p_min)
        ).item()
    ):
        raise Failure("Domain", "electron pressure is below its floor")
    if not bool(
        xp.all(
            xp.isfinite(parallel).reshape(context.N, nc)
            & (
                xp.real(parallel).reshape(context.N, nc)
                > electron.parallel_min
            )
        ).item()
    ):
        raise Failure("Domain", "electron parallel pressure is below its floor")
    fluid = SimpleNamespace(
        n=density,
        mom=xp.zeros(
            (context.N, 3, nc), dtype=xp.complex128
        ),
        u=velocity,
        perp=pressure,
        parallel=parallel,
        nf=derived.electron_face_density,
        flux=derived.electron_flux,
        flux_hat=derived.electron_flux_hat,
        divergence=_sampled_divergence(derived.electron_flux, context),
        divergence_hat=mode_apply(
            context.op.DJ, derived.electron_flux_hat, context.compute
        ),
        tensor=None,
        raw_moment=None,
        gradient=None,
        theta=None,
        strain=None,
        convection=None,
        pressure_force=None,
        magnetic_force=None,
        non_electric=None,
        pdot_perp=None,
        pdot_parallel=None,
        pressure_flux=None,
        parallel_flux=None,
        wall=build_wall(density, electron, tag, context),
        tag=tag,
    )
    gradient = velocity_gradient(fluid, velocity, context)
    gradient_samples = gradient.reshape(context.N, 3, 3, nc)
    direction = derived.bhat.reshape(context.N, 3, nc)
    theta = (
        gradient_samples[:, 0, 0, :]
        + gradient_samples[:, 1, 1, :]
        + gradient_samples[:, 2, 2, :]
    )
    strain = xp.einsum(
        "naj,nabj,nbj->nj", direction, gradient_samples, direction
    )
    tensor = _pressure_tensor(
        SimpleNamespace(perp=pressure, parallel=parallel), 1, derived.bhat, context
    )
    pressure_force = gradient_adjoint(fluid, tensor.reshape(-1), context)
    if getattr(context.op, "box_wall", None) is not None:
        pressure_force += electron_pressure_traction(state.B, fluid, context)
    magnetic_force = _warp_magnetic_force(
        velocity, derived.BN, density, electron.charge, context
    )
    if magnetic_force is None:
        bn = derived.BN.reshape(context.N, 3, nc)
        u = velocity.reshape(context.N, 3, nc)
        magnetic_force = electron.charge * xp.real(
            xp.asarray(density)
        ).reshape(context.N, 1, nc) * xp.stack(
            (
                u[:, 1, :] * bn[:, 2, :] - u[:, 2, :] * bn[:, 1, :],
                u[:, 2, :] * bn[:, 0, :] - u[:, 0, :] * bn[:, 2, :],
                u[:, 0, :] * bn[:, 1, :] - u[:, 1, :] * bn[:, 0, :],
            ),
            axis=1,
        )
    magnetic_force = magnetic_force.reshape(context.N, 3, nc)
    pressure_flux = _transport_flux(pressure, fluid, context)
    parallel_flux = _transport_flux(parallel, fluid, context)
    pressure_transport = _sampled_divergence(pressure_flux, context)
    parallel_transport = _sampled_divergence(parallel_flux, context)
    perp = pressure.reshape(context.N, nc)
    par = parallel.reshape(context.N, nc)
    fluid.gradient = gradient
    fluid.theta = theta.reshape(-1)
    fluid.strain = strain.reshape(-1)
    fluid.tensor = tensor.reshape(-1)
    fluid.raw_moment = tensor.reshape(-1)
    fluid.pressure_force = pressure_force
    fluid.magnetic_force = magnetic_force.reshape(-1)
    fluid.non_electric = (
        pressure_force + magnetic_force.reshape(-1)
    )
    fluid.convection = xp.zeros_like(pressure_force)
    fluid.pressure_flux = pressure_flux
    fluid.parallel_flux = parallel_flux
    fluid.pdot_perp = (
        -pressure_transport.reshape(context.N, nc)
        + perp * (strain - theta)
    ).reshape(-1)
    fluid.pdot_parallel = (
        -parallel_transport.reshape(context.N, nc) - 2.0 * par * strain
    ).reshape(-1)
    finite_volume_velocity = (
        derived.electron_flux.reshape(context.N, context.op.Ne)
        / derived.electron_face_density.reshape(
            context.N, context.op.Ne
        )
    ).reshape(-1)
    fluid.theta_consistency = _sampled_divergence(
        finite_volume_velocity, context
    )
    fluid.theta_consistency_hat = forward(
        fluid.theta_consistency, nc, context
    )
    fluid.theta_defect = theta.reshape(-1) - fluid.theta_consistency
    return fluid


def _massless_e0(fluid, derived, context):
    xp = _xp(context)
    nc = context.op.Nc
    e = -float(context.p.species[1].charge)
    density = derived.electron_density
    bn = derived.BN.reshape(context.N, 3, nc)
    velocity = derived.electron_velocity.reshape(context.N, 3, nc)
    cross = xp.stack(
        (
            velocity[:, 1, :] * bn[:, 2, :]
            - velocity[:, 2, :] * bn[:, 1, :],
            velocity[:, 2, :] * bn[:, 0, :]
            - velocity[:, 0, :] * bn[:, 2, :],
            velocity[:, 0, :] * bn[:, 1, :]
            - velocity[:, 1, :] * bn[:, 0, :],
        ),
        axis=1,
    )
    source = (
        fluid.pressure_force.reshape(context.N, 3, nc) / e
        - xp.real(xp.asarray(density)).reshape(context.N, 1, nc) * cross
    )
    projected = _constraint_project(source.reshape(-1), density, context)
    projected_velocity_units = (
        projected.reshape(context.N, 3, nc)
        / density.reshape(context.N, 1, nc)
    ).reshape(-1)
    e0 = forward(
        sample_apply(
            context.op.I,
            projected_velocity_units,
            context.N,
            context.compute,
        ),
        context.op.Ne,
        context,
    )
    return e0


def _wall_correction(e0, context):
    if getattr(context.geometry, "lcfs_polygon", None) is not None:
        return apply_wall_cache(get_wall_cache(context), e0)
    xp = _xp(context)
    ne = context.op.Ne
    m1 = xp.real(xp.asarray(context.op.M1)).astype(xp.float64)
    extraction = _wall_extractions(context)
    records = []
    correction = xp.zeros_like(e0)
    length = float(context.p.ref.length)
    electric = float(context.p.ref.electric)
    for stored in range(_modes(context)):
        source = e0[stored * ne : (stored + 1) * ne]
        S = extraction[stored]
        C = xp.asarray(context.op.C[stored], dtype=xp.complex128)
        Cdag = xp.asarray(context.op.Cdag[stored], dtype=xp.complex128)
        curl = Cdag @ C
        Aw = S @ curl @ (S.T.conj() / m1[:, None])
        rhs = -(S @ (curl @ source))
        hermitian_scale = max(_max_abs(Aw, context), 1.0e-300)
        if _max_abs(Aw - Aw.T.conj(), context) > (
            4096.0 * sys.float_info.epsilon * hermitian_scale
        ):
            raise Failure("WallField", "wall correction matrix is not Hermitian")
        rows = context.compute.independent_rows(Aw, context.p.rank_rtol)
        multiplier = xp.zeros(Aw.shape[0], dtype=xp.complex128)
        linear = None
        if rows:
            selected = xp.asarray(rows, dtype=xp.int64)
            principal = Aw[xp.ix_(selected, selected)]
            selected_rhs = rhs[selected]
            row_scale = xp.full(
                (len(rows),), electric / length**3, dtype=xp.float64
            )
            column_scale = xp.full(
                (len(rows),), electric * length**2, dtype=xp.float64
            )
            linear = context.compute.linear(
                principal,
                selected_rhs,
                row_scale,
                column_scale,
                context.p.rank_rtol,
                128.0 * max(len(rows), 1) * xp.finfo(xp.float64).eps,
            )
            values = xp.zeros_like(multiplier)
            values[selected] = linear.x
            multiplier = values
        full_residual = Aw @ multiplier - rhs
        scale = max(_max_abs(rhs, context), 1.0e-300)
        if _max_abs(full_residual, context) > 1.0e-8 * scale:
            raise Failure(
                "WallField",
                "constrained electric-field correction fails an original wall row",
            )
        correction[
            stored * ne : (stored + 1) * ne
        ] = (S.T.conj() @ multiplier) / m1
        omitted = sorted(set(range(rhs.size)) - set(rows))
        records.append(
            SimpleNamespace(
                mode=stored - context.p.K,
                rank=len(rows),
                selected_rows=list(rows),
                gauge_rows=omitted,
                gauge="omitted principal multipliers fixed to zero",
                rhs=rhs,
                multiplier=multiplier,
                full_residual=full_residual,
                linear=linear,
            )
        )
    return correction, SimpleNamespace(
        modes=records,
        rank=sum(record.rank for record in records),
        extraction=extraction,
    )


def _initial_constraints(derived, context):
    xp = _xp(context)
    modes = _modes(context)
    divergence = magnetic_divergence(derived.state_B, context)
    divergence_scale = 0.0
    for stored in range(modes):
        matrix = xp.asarray(context.op.DB[stored], dtype=xp.complex128)
        values = derived.state_B[
            stored * context.op.Nb : (stored + 1) * context.op.Nb
        ]
        divergence_scale = max(
            divergence_scale,
            _max_abs(matrix, context) * _max_abs(values, context),
        )
    divergence_bound = (
        4096.0
        * sys.float_info.epsilon
        * divergence_scale
        * max(context.op.Nb, 1)
    )
    if _max_abs(divergence, context) > divergence_bound:
        raise Failure(
            "InitialCompatibility", "initial magnetic field has nonzero divergence"
        )
    current_scale = 0.0
    for stored in range(modes):
        matrix = xp.asarray(context.op.Cdag[stored], dtype=xp.complex128)
        values = derived.state_B[
            stored * context.op.Nb : (stored + 1) * context.op.Nb
        ]
        current_scale = max(
            current_scale,
            _max_abs(matrix, context)
            * _max_abs(values, context)
            / context.p.mu0,
        )
    wall_current = []
    for stored, extraction in enumerate(_wall_extractions(context)):
        ne = context.op.Ne
        wall_current.append(
            extraction
            @ derived.current[stored * ne : (stored + 1) * ne]
        )
    full = _xp(context).concatenate(wall_current) if wall_current else []
    current_bound = (
        4096.0
        * sys.float_info.epsilon
        * current_scale
        * max(context.op.Nb, 1)
    )
    if _max_abs(full, context) > current_bound:
        raise Failure(
            "WallCurrent",
            "initial current violates the full closed-wall constraint: "
            f"residual={_max_abs(full, context)} "
            f"roundoff_bound={current_bound}",
        )


def _new_result(tag):
    result = stage_result_record(tag=tag)
    result.cache = field_cache_record(tag=tag)
    result.fluid = [None, None]
    result.terms = [None, None]
    result.ledger = [None, None]
    result.canonical_state = None
    result.E0 = None
    result.model = MODEL_NAME
    result.applicability = SimpleNamespace(
        electron_inertia="not_applicable",
        electron_kinetic_energy="not_applicable",
        electron_kinetic_transport="not_applicable",
        electron_mechanical_stress="not_applicable",
        finite_charge_electric_stress="not_applicable",
        longitudinal_electric_energy="not_applicable",
        finite_charge_gauss="not_applicable",
        spectral_kinetic_transfer="not_applicable",
    )
    return result


def evaluate_stage(state, tag, context, alias=False):
    """Evaluate one read-only massless stage and return its canonical state."""
    _require_model(context)
    result = _new_result(tag)
    try:
        if (
            tag.step != state.step
            or tag.generation != state.generation
            or tag.time != state.time
        ):
            raise Failure("StaleStage", "stage identity does not match state")
        canonical = canonicalize_state(state, context)
        _check_retained_massless(canonical, context)
        derived = _derive_electron(canonical, context)
        derived.state_B = canonical.B
        _initial_constraints(derived, context)
        result.source_id = _fresh(context, "next_source")
        result.canonical_state = canonical
        result.tag = tag
        result.op = context.op
        result.BN = derived.BN
        result.bhat = derived.bhat
        result.current = derived.current
        result.fJ0 = None
        result.rho = _xp(context).zeros(
            _modes(context) * context.op.Nc, dtype=_xp(context).complex128
        )
        result.AE = None
        result.ELdot = None
        result.phidot = None
        result.ampere = None

        electron = _electron_fluid(derived, canonical, tag, context)
        derived.ion.wall.tag = tag
        result.fluid = (derived.ion, electron)
        e0 = _massless_e0(electron, derived, context)
        correction, wall_correction = _wall_correction(e0, context)
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

        result.rhs = allocate_state(context)
        result.rhs.step = state.step
        result.rhs.generation = state.generation
        result.rhs.time = state.time
        finish_fluid(result, 0, context)
        qi_over_e = _electroweak_parameters(context)[1]
        result.rhs.fluid[1].n = qi_over_e * result.rhs.fluid[0].n
        result.rhs.fluid[1].mom = _xp(context).zeros_like(
            result.rhs.fluid[1].mom
        )
        result.rhs.fluid[1].perp = forward(
            electron.pdot_perp, context.op.Nc, context
        )
        result.rhs.fluid[1].parallel = forward(
            electron.pdot_parallel, context.op.Nc, context
        )
        result.rhs.B = -mode_apply(context.op.C, electric, context.compute)
        reaction = correction
        result.terms[1] = SimpleNamespace(
            electric=electron.pressure_force + electron.magnetic_force,
            force=reaction,
            reaction=reaction,
            multiplier=None,
            multiplier_residual=None,
            pressure=electron.pressure_force,
            magnetic=electron.magnetic_force,
        )

        from . import massless_checks

        massless_checks.populate_stage(canonical, result, context)
        massless_checks.check_stage(canonical, result, context)
        if alias:
            _alias_probe(canonical, tag, context, result)
        return result
    except Failure as failure:
        retain_failure_evidence(
            failure, tag, context, state, result, "trial"
        )
        raise
    except Exception as error:
        failure = Failure("NumericalLibrary", str(error))
        retain_failure_evidence(
            failure, tag, context, state, result, "trial"
        )
        raise failure from error


def _alias_probe(canonical, tag, context, result):
    updated = copy.copy(context)
    updated.N = 2 * context.N + 1
    updated.spectral_id = context.spectral_id + 1
    alternate_tag = StageTag(
        step=tag.step,
        generation=tag.generation,
        trial=tag.trial,
        spectral_context=updated.spectral_id,
        time=tag.time,
        stage="alias",
    )
    alternate = evaluate_stage(canonical, alternate_tag, updated, False)
    context.next_source = updated.next_source
    from . import diagnostics

    for species in range(2):
        diagnostics.measure(
            result.checks,
            "alias_n",
            alternate.rhs.fluid[species].n - result.rhs.fluid[species].n,
            context.op.M0,
            "1/(m^3*s)",
            tag,
            context,
            context.p.species[species].id,
        )
        diagnostics.measure(
            result.checks,
            "alias_mom",
            alternate.rhs.fluid[species].mom
            - result.rhs.fluid[species].mom,
            context.compute.xp.tile(
                context.compute.xp.real(
                    context.compute.xp.asarray(context.op.M0)
                ).astype(context.compute.xp.float64),
                3,
            ),
            "kg/(m^2*s^2)",
            tag,
            context,
            context.p.species[species].id,
        )
        diagnostics.measure(
            result.checks,
            "alias_perp",
            alternate.rhs.fluid[species].perp
            - result.rhs.fluid[species].perp,
            context.op.M0,
            "Pa/s",
            tag,
            context,
            context.p.species[species].id,
        )
    diagnostics.measure(
        result.checks,
        "alias_parallel",
        alternate.rhs.fluid[1].parallel - result.rhs.fluid[1].parallel,
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
    mb = _xp(context).real(_xp(context).asarray(context.op.MB)).astype(
        _xp(context).float64
    )
    for component, name in enumerate(("R", "phi", "Z")):
        start, stop = offsets[component], offsets[component + 1]
        left = result.rhs.B.reshape(_modes(context), context.op.Nb)[
            :, start:stop
        ].reshape(-1)
        right = alternate.rhs.B.reshape(_modes(context), context.op.Nb)[
            :, start:stop
        ].reshape(-1)
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


def combine(base, rhs, weight, tag, context):
    """Advance only independent massless variables, then canonicalize."""
    if not math.isfinite(weight):
        raise Failure("Time", "nonfinite stage weight")
    xp = _xp(context)
    fluid = []
    for species in range(2):
        left = base.fluid[species]
        right = rhs.fluid[species]
        fluid.append(
            FluidState(
                n=(
                    xp.asarray(left.n).copy()
                    if species
                    else context.compute.axpy(left.n, right.n, weight)
                ),
                mom=(
                    xp.asarray(left.mom).copy()
                    if species
                    else context.compute.axpy(left.mom, right.mom, weight)
                ),
                perp=context.compute.axpy(left.perp, right.perp, weight),
                parallel=context.compute.axpy(
                    left.parallel, right.parallel, weight
                ),
            )
        )
    candidate = State(
        fluid=(fluid[0], fluid[1]),
        B=context.compute.axpy(base.B, rhs.B, weight),
        step=tag.step,
        generation=tag.generation,
        time=tag.time,
    )
    try:
        return canonicalize_state(candidate, context)
    except Failure as failure:
        _retain_common_failure_evidence(
            failure, tag, context, candidate, None, "candidate"
        )
        raise


def initialize(state, context, cumulative, equilibrium, captured=None):
    """Validate and commit a canonical massless state."""
    _require_model(context)
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
    stage = evaluate_stage(state, tag, context, alias=False)
    try:
        from . import diagnostics

        diagnostics.measure(
            stage.checks,
            "initial_divB",
            magnetic_divergence(stage.canonical_state.B, context),
            context.geometry.wv,
            "T/m",
            tag,
            context,
        )
        wall_current = []
        for stored, extraction in enumerate(_wall_extractions(context)):
            wall_current.append(
                extraction
                @ stage.current[
                    stored * context.op.Ne : (stored + 1) * context.op.Ne
                ]
            )
        diagnostics.measure(
            stage.checks,
            "wall_initial_current",
            _xp(context).concatenate(wall_current).reshape(-1),
            _xp(context).ones(
                len(context.geometry.walls), dtype=_xp(context).float64
            ),
            "A/m^2",
            tag,
            context,
            samples=True,
        )
        if equilibrium:
            for species in range(2):
                diagnostics.measure(
                    stage.checks,
                    "equilibrium_n",
                    stage.rhs.fluid[species].n,
                    context.op.M0,
                    "1/(m^3*s)",
                    tag,
                    context,
                    context.p.species[species].id,
                )
                diagnostics.measure(
                    stage.checks,
                    "equilibrium_perp",
                    stage.rhs.fluid[species].perp,
                    context.op.M0,
                    "Pa/s",
                    tag,
                    context,
                    context.p.species[species].id,
                )
                diagnostics.measure(
                    stage.checks,
                    "equilibrium_mom",
                    stage.rhs.fluid[species].mom,
                    context.compute.xp.tile(
                        context.compute.xp.real(
                            context.compute.xp.asarray(context.op.M0)
                        ).astype(context.compute.xp.float64),
                        3,
                    ),
                    "kg/(m^2*s^2)",
                    tag,
                    context,
                    context.p.species[species].id,
                )
            diagnostics.measure(
                stage.checks,
                "equilibrium_parallel",
                stage.rhs.fluid[1].parallel,
                context.op.M0,
                "Pa/s",
                tag,
                context,
                context.p.species[1].id,
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
        diagnostics.gate(
            stage.checks, ("initial_divB", "wall_initial_current")
        )
        cumulative.baseline = SimpleNamespace(
            divB=magnetic_divergence(stage.canonical_state.B, context),
            totals=stage.totals,
            momentum=stage.momentum,
            time=stage.canonical_state.time,
            step=stage.canonical_state.step,
        )
        cumulative.last_step = stage.canonical_state.step
        cumulative.last_time = stage.canonical_state.time
        if not hasattr(cumulative, "accepted_intervals"):
            cumulative.accepted_intervals = 0
        from . import massless_checks

        cumulative.applicability = massless_checks.applicability()
        committed = SimpleNamespace(
            state=stage.canonical_state,
            cache=stage.cache,
            accepted=True,
        )
        if captured is not None:
            update = getattr(captured, "update", None)
            if update is not None:
                update(vars(stage))
        return committed
    except Failure as failure:
        retain_failure_evidence(failure, tag, context, state, stage, "trial")
        raise


def retain_failure_evidence(
    failure, tag, context, state, result, prefix
):
    """Retain branch evidence without recursive public-stage dispatch."""
    _retain_common_failure_evidence(
        failure, tag, context, state, result, prefix
    )
    from . import massless_checks

    present = {
        (
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
        )
        for check in failure.checks
    }
    for required in massless_checks.stage_requirements(context, tag):
        key = (
            required.name,
            required.population,
            required.component,
            getattr(required, "producer", ""),
            required.region,
            str(required.norm),
            required.tag.stage,
            required.tag.step,
            required.tag.generation,
            required.tag.trial,
            required.tag.spectral_context,
            float(required.tag.time).hex(),
        )
        if key in present:
            continue
        required.available = False
        required.passed = False
        required.hard = False
        required.reason = f"not run: {failure.code}"
        failure.checks.append(required)
        present.add(key)
    return None
