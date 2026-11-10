"""Finite-mass ion and CGL-B electron fluid physics."""

import math
import sys
from types import SimpleNamespace

from .core import forward, inverse, mode_apply, sample_apply
from .types import Failure
from .wall import build_wall, project_wall


_WARP_KERNELS = {}


def _xp(context):
    return context.compute.xp


def _modes(context):
    return 2 * context.p.K + 1


def _require_size(value, expected, name):
    if tuple(value.shape) != (expected,):
        raise Failure("Shape", f"{name} has an invalid extent")


def _require_species(species):
    if species not in (0, 1):
        raise Failure("Species", "species index must be 0 or 1")


def _require_finite(value, name, context):
    xp = _xp(context)
    if not bool(xp.all(xp.isfinite(value))):
        raise Failure("Finite", f"{name} contains non-finite values")


def _as_complex(value, context):
    return _xp(context).asarray(value, dtype=_xp(context).complex128)


def _geometry_kind(context):
    kind = getattr(context.geometry, "kind", None)
    if kind is None:
        kind = getattr(context.op, "geometry_kind", "staircase_v1")
    return kind


def _is_polygon(context):
    return _geometry_kind(context) == "polygon_dual_v1"


def _polygon_backend_real_array(context, value, expected_shape, name):
    xp = _xp(context)
    if not isinstance(value, xp.ndarray):
        raise Failure("Backend", f"{name} must be a backend array")
    if value.dtype != xp.float64 or tuple(value.shape) != tuple(expected_shape):
        raise Failure("Layout", f"{name} has an invalid dtype or shape")
    return value


def _polygon_face_normals(context, nf):
    xp = _xp(context)
    value = getattr(context.op, "face_normals_backend", None)
    if value is None:
        # Synthetic-test fallback only; production operators publish this array.
        value = xp.asarray(context.geometry.face_normals)
    return _polygon_backend_real_array(
        context, value, (nf, 2), "polygon face normals"
    )


def _polygon_cell_radius(context):
    xp = _xp(context)
    value = getattr(context.op, "cell_radius_backend", None)
    if value is None:
        # Synthetic-test fallback only; production operators publish this array.
        value = xp.asarray(context.geometry.cells.R)
    return _polygon_backend_real_array(
        context, value, (context.op.Nc,), "polygon cell radius"
    )


def _polygon_normal_data(context):
    xp = _xp(context)
    nc = context.op.Nc
    nf = context.op.Dface.shape[1]
    scalar_i = xp.asarray(context.op.scalar_I)
    face_normals = _polygon_face_normals(context, nf)
    if (
        context.op.Dface.ndim != 2
        or tuple(context.op.Dface.shape) != (nc, nf)
        or tuple(scalar_i.shape) != (context.op.Ne, nc)
    ):
        raise Failure("Layout", "polygon normal-face operators are inconsistent")
    return nf, scalar_i[:nf], face_normals


def _apply_mode_matrix(matrix, value, entities, context):
    modes = _modes(context)
    _require_size(value, modes * entities, "mode input")
    operator = _xp(context).asarray(matrix)
    sampled = value.reshape(modes, entities)
    return (sampled @ operator.T.conj()).reshape(-1)


def _component_flux(scalar, context):
    xp = _xp(context)
    nc = context.op.Nc
    _require_size(scalar, context.N * nc, "scalar interpolation")
    if _is_polygon(context):
        result = sample_apply(
            context.op.scalar_I, scalar, context.N, context.compute
        )
        _require_size(
            result,
            context.N * context.op.Ne,
            "polygon scalar interpolation",
        )
        return result
    nodes = xp.repeat(scalar.reshape(context.N, 1, nc), 3, axis=1)
    result = sample_apply(
        context.op.I, nodes.reshape(-1), context.N, context.compute
    )
    _require_size(result, context.N * context.op.Ne, "component interpolation")
    return result


def _sampled_divergence(edges, context):
    xp = _xp(context)
    nc = context.op.Nc
    if _is_polygon(context):
        nf = context.op.Dface.shape[1]
        if nf + nc != context.op.Ne:
            raise Failure("Layout", "polygon edge component offsets are inconsistent")
        samples = edges.reshape(context.N, context.op.Ne)
        normal = sample_apply(
            context.op.Dface,
            samples[:, :nf].reshape(-1),
            context.N,
            context.compute,
        ).reshape(context.N, nc)
        toroidal = context.compute.derivative(
            samples[:, nf:].reshape(-1), nc, context.N
        ).reshape(context.N, nc)
        radius = _polygon_cell_radius(context)
        return (normal + toroidal / radius[None, :]).reshape(-1)
    nr = context.op.DR.shape[1]
    nz = context.op.DZ.shape[1]
    if nr + nc + nz != context.op.Ne:
        raise Failure("Layout", "edge component offsets are inconsistent")
    samples = edges.reshape(context.N, context.op.Ne)
    radial = sample_apply(
        context.op.DR, samples[:, :nr].reshape(-1), context.N, context.compute
    ).reshape(context.N, nc)
    azimuthal = samples[:, nr : nr + nc].reshape(-1)
    azimuthal = context.compute.derivative(
        azimuthal, nc, context.N
    ).reshape(context.N, nc)
    axial = sample_apply(
        context.op.DZ,
        samples[:, nr + nc :].reshape(-1),
        context.N,
        context.compute,
    ).reshape(context.N, nc)
    radius = xp.asarray(context.geometry.cells.R, dtype=xp.float64)
    return (radial + axial + azimuthal / radius[None, :]).reshape(-1)


def _pressure_tensor(fluid, species, bhat, context):
    xp = _xp(context)
    nc = context.op.Nc
    pressure = xp.zeros(
        (context.N, 3, 3, nc), dtype=xp.complex128
    )
    perp = fluid.perp.reshape(context.N, 1, 1, nc)
    diagonal = xp.eye(3, dtype=xp.complex128).reshape(1, 3, 3, 1)
    pressure += perp * diagonal
    if species == 1:
        direction = bhat.reshape(context.N, 3, nc)
        difference = (fluid.parallel - fluid.perp).reshape(context.N, 1, 1, nc)
        pressure += (
            difference
            * direction[:, :, None, :]
            * xp.conj(direction[:, None, :, :])
        )
    return pressure


def _magnetic_force_kernel(wp):
    @wp.func
    def cmul(left: wp.vec2d, right: wp.vec2d):
        return wp.vec2d(
            left[0] * right[0] - left[1] * right[1],
            left[0] * right[1] + left[1] * right[0],
        )

    @wp.kernel
    def magnetic_force_kernel(
        u: wp.array(dtype=wp.vec2d),
        b: wp.array(dtype=wp.vec2d),
        n: wp.array(dtype=wp.vec2d),
        charge: wp.float64,
        nc: wp.int32,
        out: wp.array(dtype=wp.vec2d),
    ):
        index = wp.tid()
        sample = index // nc
        cell = index - sample * nc
        vector = sample * 3 * nc + cell
        density_charge = wp.vec2d(
            charge * n[index][0], charge * n[index][1]
        )
        ur = u[vector]
        uphi = u[vector + nc]
        uz = u[vector + 2 * nc]
        br = b[vector]
        bphi = b[vector + nc]
        bz = b[vector + 2 * nc]
        out[vector] = cmul(
            density_charge,
            cmul(uphi, bz) - cmul(uz, bphi),
        )
        out[vector + nc] = cmul(
            density_charge,
            cmul(uz, br) - cmul(ur, bz),
        )
        out[vector + 2 * nc] = cmul(
            density_charge,
            cmul(ur, bphi) - cmul(uphi, br),
        )

    return magnetic_force_kernel


def _warp_magnetic_force(velocity, magnetic, density, charge, context):
    """Evaluate the pointwise Lorentz force with the Warp FP64 kernel."""
    backend = context.compute
    if backend.name != "cuda":
        return None
    try:
        wp = backend._warp
        device = backend._wp_device
        key = (id(wp), int(device.ordinal))
        kernel = _WARP_KERNELS.get(key)
        if kernel is None:
            kernel = _magnetic_force_kernel(wp)
            _WARP_KERNELS[key] = kernel
        source_u, owner_u = backend._warp_array(velocity)
        source_b, owner_b = backend._warp_array(magnetic)
        source_n, owner_n = backend._warp_array(density)
        out = backend.array(backend.xp.empty_like(velocity))
        out_w, owner_out = backend._warp_array(out)
        # The storage owners must outlive every queued view on the stream.
        stream = backend._external_stream()
        wp.launch(
            kernel,
            dim=context.N * context.op.Nc,
            inputs=[
                source_u,
                source_b,
                source_n,
                float(charge),
                int(context.op.Nc),
            ],
            outputs=[out_w],
            device=device,
            stream=stream,
        )
        wp.synchronize_stream(stream)
        del owner_u, owner_b, owner_n, owner_out
        return out
    except Failure:
        raise
    except Exception as error:
        raise Failure("Warp", f"magnetic-force kernel failed: {error}") from error


def _validate_magnetic_samples(bn, bhat, context):
    xp = _xp(context)
    nc = context.op.Nc
    expected = context.N * 3 * nc
    _require_size(bn, expected, "magnetic vector")
    _require_size(bhat, expected, "magnetic direction")
    _require_finite(bn, "magnetic magnitude", context)
    _require_finite(bhat, "magnetic direction", context)
    vector = bn.reshape(context.N, 3, nc)
    direction = bhat.reshape(context.N, 3, nc)
    magnitude = xp.sqrt(xp.sum(xp.abs(vector) ** 2, axis=1))
    if not bool(xp.all(magnitude > context.p.B_min)):
        raise Failure("MagneticDomain", "magnetic magnitude is below B_min")
    direction_norm = xp.sum(xp.abs(direction) ** 2, axis=1)
    if not bool(xp.all(xp.abs(direction_norm - 1.0) <= 2.0e-10)):
        raise Failure("MagneticDirection", "magnetic direction is not normalized")


def build_fluid(state, species, bn, bhat, tag, context):
    """Recover primitive samples, shared flux, pressures, and wall factors."""
    _require_species(species)
    xp = _xp(context)
    nc = context.op.Nc
    modes = _modes(context)
    physical = context.p.species[species]
    _require_size(state.n, modes * nc, "density")
    _require_size(state.mom, modes * 3 * nc, "momentum")
    _require_size(state.perp, modes * nc, "perpendicular pressure")
    if species == 1:
        _require_size(state.parallel, modes * nc, "parallel pressure")
    _require_finite(state.n, "density", context)
    _require_finite(state.mom, "momentum", context)
    _require_finite(state.perp, "perpendicular pressure", context)
    if species == 1:
        _require_finite(state.parallel, "parallel pressure", context)
    _validate_magnetic_samples(bn, bhat, context)

    density = inverse(state.n, nc, context, True)
    momentum = inverse(state.mom, 3 * nc, context, True)
    perp = inverse(state.perp, nc, context, True)
    parallel = (
        inverse(state.parallel, nc, context, True)
        if species == 1
        else xp.zeros(0, dtype=xp.complex128)
    )
    density_samples = density.reshape(context.N, nc)
    perp_samples = perp.reshape(context.N, nc)
    if not bool(xp.all(xp.isfinite(xp.real(density_samples)))):
        raise Failure("Finite", "density is non-finite")
    if not bool(xp.all(xp.real(density_samples) > physical.n_min)):
        raise Failure("Domain", "density is not positive before reciprocal")
    if not bool(xp.all(xp.real(perp_samples) > physical.p_min)):
        raise Failure("Domain", "pressure is below its declared floor")
    if species == 1 and not bool(
        xp.all(xp.real(parallel.reshape(context.N, nc)) > physical.parallel_min)
    ):
        raise Failure("Domain", "parallel pressure is below its declared floor")

    velocity = momentum.reshape(context.N, 3, nc) / (
        physical.mass * density_samples[:, None, :]
    )
    flux_hat = _apply_mode_matrix(
        context.op.I, _as_complex(state.mom, context) / physical.mass, 3 * nc, context
    )
    flux = inverse(flux_hat, context.op.Ne, context, True)
    _require_finite(flux, "sample flux", context)
    face_density = _component_flux(density, context)
    if not bool(
        xp.all(xp.isfinite(face_density))
        and xp.all(xp.real(face_density) > physical.face_min)
    ):
        raise Failure("Domain", "interpolated density is not invertible")

    divergence_hat = mode_apply(
        context.op.DJ, flux_hat, context.compute
    )
    divergence = _sampled_divergence(flux, context)
    sampled_retained = forward(divergence, nc, context)
    scale = xp.maximum(
        1.0,
        xp.maximum(
            xp.max(xp.abs(divergence_hat)), xp.max(xp.abs(sampled_retained))
        ),
    )
    if float(xp.max(xp.abs(divergence_hat - sampled_retained))) > (
        2048.0 * xp.finfo(xp.float64).eps * float(scale)
    ):
        raise Failure(
            "FluidDivision",
            "retained and sampled density divergences disagree",
        )

    tensor = _pressure_tensor(
        SimpleNamespace(perp=perp, parallel=parallel),
        species,
        bhat,
        context,
    )
    inertial = (
        physical.mass
        * density_samples[:, None, None, :]
        * velocity[:, :, None, :]
        * velocity[:, None, :, :]
    )
    raw_moment = inertial + tensor
    wall = build_wall(density, physical, tag, context)
    return SimpleNamespace(
        n=density,
        mom=momentum,
        u=velocity.reshape(-1),
        perp=perp,
        parallel=parallel,
        nf=face_density,
        flux=flux,
        flux_hat=flux_hat,
        divergence=divergence,
        divergence_hat=divergence_hat,
        tensor=tensor.reshape(-1),
        raw_moment=raw_moment.reshape(-1),
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
        wall=wall,
        tag=tag,
    )


def velocity_gradient(fluid, x, context):
    """Return the compatible physical cylindrical gradient on all samples."""
    xp = _xp(context)
    nc = context.op.Nc
    _require_size(x, context.N * 3 * nc, "velocity-gradient input")
    _require_size(fluid.n, context.N * nc, "velocity-gradient density")
    _require_size(fluid.nf, context.N * context.op.Ne, "velocity-gradient denominator")
    _require_finite(x, "velocity-gradient input", context)
    values = x.reshape(context.N, 3, nc)
    if _is_polygon(context):
        nf, scalar_to_face, face_normals = _polygon_normal_data(context)
        density = fluid.n.reshape(context.N, 1, nc)
        face_density = fluid.nf.reshape(context.N, context.op.Ne)
        radius = _polygon_cell_radius(context)
        gradient = xp.zeros(
            (context.N, 3, 3, nc), dtype=xp.complex128
        )
        for component in range(3):
            weighted = density * values[:, component : component + 1, :]
            face_scalar = sample_apply(
                scalar_to_face,
                weighted.reshape(-1),
                context.N,
                context.compute,
            ).reshape(context.N, nf)
            normal_scalar = face_scalar / face_density[:, :nf]
            radial = sample_apply(
                context.op.Dface,
                (face_normals[:, 0][None, :] * normal_scalar).reshape(-1),
                context.N,
                context.compute,
            ).reshape(context.N, nc)
            axial = sample_apply(
                context.op.Dface,
                (face_normals[:, 1][None, :] * normal_scalar).reshape(-1),
                context.N,
                context.compute,
            ).reshape(context.N, nc)
            phi = context.compute.derivative(
                values[:, component, :].reshape(-1), nc, context.N
            ).reshape(context.N, nc)
            gradient[:, component, 0, :] = (
                radial - values[:, component, :] / radius[None, :]
            )
            gradient[:, component, 1, :] = phi / radius[None, :]
            gradient[:, component, 2, :] = axial
        gradient[:, 0, 1, :] -= values[:, 1, :] / radius[None, :]
        gradient[:, 1, 1, :] += values[:, 0, :] / radius[None, :]
        return gradient.reshape(-1)
    density = fluid.n.reshape(context.N, 1, nc)
    face_density = fluid.nf.reshape(context.N, context.op.Ne)
    radius = xp.asarray(context.geometry.cells.R, dtype=xp.float64)
    gradient = xp.zeros((context.N, 3, 3, nc), dtype=xp.complex128)
    nr = context.op.DR.shape[1]
    axial_offset = nr + nc
    for component in range(3):
        component_values = values[:, component, :].reshape(-1)
        derivative_phi = context.compute.derivative(
            component_values, nc, context.N
        ).reshape(context.N, nc)
        weighted = density * values[:, component : component + 1, :]
        weighted = xp.repeat(weighted, 3, axis=1)
        interpolated = sample_apply(
            context.op.I, weighted.reshape(-1), context.N, context.compute
        ).reshape(context.N, context.op.Ne)
        radial_edges = interpolated[:, :nr] / face_density[:, :nr]
        axial_edges = interpolated[:, axial_offset:] / face_density[:, axial_offset:]
        radial = sample_apply(
            context.op.DR, radial_edges.reshape(-1), context.N, context.compute
        ).reshape(context.N, nc)
        axial = sample_apply(
            context.op.DZ, axial_edges.reshape(-1), context.N, context.compute
        ).reshape(context.N, nc)
        gradient[:, component, 0, :] = radial - values[:, component, :] / radius[None, :]
        gradient[:, component, 1, :] = derivative_phi / radius[None, :]
        gradient[:, component, 2, :] = axial
    gradient[:, 0, 1, :] -= values[:, 1, :] / radius[None, :]
    gradient[:, 1, 1, :] += values[:, 0, :] / radius[None, :]
    return gradient.reshape(-1)


def gradient_adjoint(fluid, tensor, context):
    """Apply the metric-weighted adjoint of ``velocity_gradient``."""
    xp = _xp(context)
    nc = context.op.Nc
    _require_size(tensor, context.N * 9 * nc, "gradient-adjoint tensor")
    _require_size(fluid.n, context.N * nc, "gradient-adjoint density")
    _require_size(fluid.nf, context.N * context.op.Ne, "gradient-adjoint denominator")
    _require_finite(tensor, "gradient-adjoint tensor", context)
    values = tensor.reshape(context.N, 3, 3, nc)
    if _is_polygon(context):
        nf, scalar_to_face, face_normals = _polygon_normal_data(context)
        density = fluid.n.reshape(context.N, nc)
        face_density = fluid.nf.reshape(context.N, context.op.Ne)
        m0 = xp.real(xp.asarray(context.op.M0)).astype(xp.float64)
        radius = _polygon_cell_radius(context)
        result = xp.zeros((context.N, 3, nc), dtype=xp.complex128)
        for row in range(3):
            weighted = m0[None, None, :] * values[:, row, :, :]
            radial_rhs = sample_apply(
                context.op.Dface.T.conj(),
                weighted[:, 0, :].reshape(-1),
                context.N,
                context.compute,
            ).reshape(context.N, nf)
            axial_rhs = sample_apply(
                context.op.Dface.T.conj(),
                weighted[:, 2, :].reshape(-1),
                context.N,
                context.compute,
            ).reshape(context.N, nf)
            radial_scalar = (
                face_normals[:, 0][None, :]
                * radial_rhs
                / face_density[:, :nf]
            )
            axial_scalar = (
                face_normals[:, 1][None, :]
                * axial_rhs
                / face_density[:, :nf]
            )
            radial_nodes = sample_apply(
                scalar_to_face.T.conj(),
                radial_scalar.reshape(-1),
                context.N,
                context.compute,
            ).reshape(context.N, nc)
            axial_nodes = sample_apply(
                scalar_to_face.T.conj(),
                axial_scalar.reshape(-1),
                context.N,
                context.compute,
            ).reshape(context.N, nc)
            toroidal_rhs = (weighted[:, 1, :] / radius[None, :]).reshape(-1)
            toroidal = context.compute.derivative(
                toroidal_rhs, nc, context.N
            ).reshape(context.N, nc)
            result[:, row, :] = (
                density * (radial_nodes + axial_nodes)
                - weighted[:, 0, :] / radius[None, :]
                - toroidal
            ) / m0[None, :]
            if row == 0:
                result[:, row, :] += (
                    values[:, 1, 1, :] / radius[None, :]
                )
            elif row == 1:
                result[:, row, :] -= (
                    values[:, 0, 1, :] / radius[None, :]
                )
        return result.reshape(-1)
    density = fluid.n.reshape(context.N, nc)
    face_density = fluid.nf.reshape(context.N, context.op.Ne)
    m0 = xp.real(xp.asarray(context.op.M0)).astype(xp.float64)
    radius = xp.asarray(context.geometry.cells.R, dtype=xp.float64)
    nr = context.op.DR.shape[1]
    axial_offset = nr + nc
    result = xp.zeros((context.N, 3, nc), dtype=xp.complex128)
    for row in range(3):
        weighted = m0[None, None, :] * values[:, row, :, :]
        radial_rhs = sample_apply(
            context.op.DR.T.conj(),
            weighted[:, 0, :].reshape(-1),
            context.N,
            context.compute,
        ).reshape(context.N, nr)
        radial_edges = xp.zeros(
            (context.N, context.op.Ne), dtype=xp.complex128
        )
        radial_edges[:, :nr] = radial_rhs / face_density[:, :nr]
        radial_nodes = sample_apply(
            context.op.I.T.conj(),
            radial_edges.reshape(-1),
            context.N,
            context.compute,
        ).reshape(context.N, 3, nc)
        axial_rhs = sample_apply(
            context.op.DZ.T.conj(),
            weighted[:, 2, :].reshape(-1),
            context.N,
            context.compute,
        ).reshape(context.N, context.op.DZ.shape[1])
        axial_edges = xp.zeros(
            (context.N, context.op.Ne), dtype=xp.complex128
        )
        axial_edges[:, axial_offset:] = axial_rhs / face_density[:, axial_offset:]
        axial_nodes = sample_apply(
            context.op.I.T.conj(),
            axial_edges.reshape(-1),
            context.N,
            context.compute,
        ).reshape(context.N, 3, nc)
        azimuthal_nodes = density * (
            radial_nodes[:, 0, :] + axial_nodes[:, 2, :]
        )
        azimuthal_rhs = (weighted[:, 1, :] / radius[None, :]).reshape(-1)
        azimuthal_derivative = context.compute.derivative(
            azimuthal_rhs, nc, context.N
        ).reshape(context.N, nc)
        result[:, row, :] = (
            azimuthal_nodes
            - weighted[:, 0, :] / radius[None, :]
            - azimuthal_derivative
        ) / m0[None, :]
        if row == 0:
            result[:, row, :] += (
                m0[None, :] * values[:, 1, 1, :] / radius[None, :]
            ) / m0[None, :]
        elif row == 1:
            result[:, row, :] -= (
                m0[None, :] * values[:, 0, 1, :] / radius[None, :]
            ) / m0[None, :]
    return result.reshape(-1)


def _transport_flux(pressure, fluid, context):
    pressure_edges = _component_flux(pressure, context)
    density = fluid.nf.reshape(context.N, context.op.Ne)
    if not bool(_xp(context).all(density != 0)):
        raise Failure("FluidDomain", "interpolated density is zero")
    return (
        pressure_edges.reshape(context.N, context.op.Ne) / density
        * fluid.flux.reshape(context.N, context.op.Ne)
    ).reshape(-1)


def build_non_electric(fluid, species, bn, bhat, context):
    """Fill all state-dependent terms that do not contain the electric field."""
    _require_species(species)
    xp = _xp(context)
    nc = context.op.Nc
    _validate_magnetic_samples(bn, bhat, context)
    physical = context.p.species[species]
    gradient = velocity_gradient(fluid, fluid.u, context)
    q = gradient.reshape(context.N, 3, 3, nc)
    direction = bhat.reshape(context.N, 3, nc)
    velocity = fluid.u.reshape(context.N, 3, nc)
    momentum = fluid.mom.reshape(context.N, 3, nc)
    theta = q[:, 0, 0, :] + q[:, 1, 1, :] + q[:, 2, 2, :]
    strain = xp.einsum("naj,nabj,nbj->nj", direction, q, direction)

    advective = velocity[:, :, None, :] * momentum[:, None, :, :] / physical.mass
    advective_adjoint = gradient_adjoint(
        fluid, advective.reshape(-1), context
    ).reshape(context.N, 3, nc)
    divergence = fluid.divergence.reshape(context.N, nc)
    gradient_momentum = xp.einsum(
        "nabj,nbj->naj", q, momentum / physical.mass
    )
    convection = 0.5 * physical.mass * (
        -advective_adjoint + gradient_momentum + velocity * divergence[:, None, :]
    )
    pressure = _pressure_tensor(
        SimpleNamespace(perp=fluid.perp, parallel=fluid.parallel),
        species,
        bhat,
        context,
    )
    pressure_force = gradient_adjoint(
        fluid, pressure.reshape(-1), context
    ).reshape(context.N, 3, nc)
    magnetic = bn.reshape(context.N, 3, nc)
    magnetic_force = _warp_magnetic_force(
        velocity.reshape(-1),
        bn,
        fluid.n,
        physical.charge,
        context,
    )
    if magnetic_force is None:
        magnetic_force = physical.charge * fluid.n.reshape(
            context.N, 1, nc
        ) * xp.stack(
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
    magnetic_force = magnetic_force.reshape(context.N, 3, nc)
    fluid.gradient = gradient
    fluid.theta = theta.reshape(-1)
    fluid.strain = strain.reshape(-1)
    fluid.convection = convection.reshape(-1)
    fluid.pressure_force = pressure_force.reshape(-1)
    fluid.magnetic_force = magnetic_force.reshape(-1)
    fluid.non_electric = (
        -convection + pressure_force + magnetic_force
    ).reshape(-1)
    fluid.tensor = pressure.reshape(-1)

    pressure_flux = _transport_flux(fluid.perp, fluid, context)
    pressure_transport = _sampled_divergence(pressure_flux, context)
    perp = fluid.perp.reshape(context.N, nc)
    if species == 0:
        fluid.pdot_perp = (
            -pressure_transport.reshape(context.N, nc)
            - (physical.gamma - 1.0) * perp * theta
        ).reshape(-1)
        fluid.pdot_parallel = xp.zeros(0, dtype=xp.complex128)
        fluid.parallel_flux = xp.zeros(0, dtype=xp.complex128)
    else:
        parallel = fluid.parallel.reshape(context.N, nc)
        parallel_flux = _transport_flux(fluid.parallel, fluid, context)
        parallel_transport = _sampled_divergence(parallel_flux, context)
        fluid.pdot_perp = (
            -pressure_transport.reshape(context.N, nc)
            + perp * (strain - theta)
        ).reshape(-1)
        fluid.pdot_parallel = (
            -parallel_transport.reshape(context.N, nc) - 2.0 * parallel * strain
        ).reshape(-1)
        fluid.parallel_flux = parallel_flux
    fluid.pressure_flux = pressure_flux


def electric_response(fluid, species, electric_hat, context):
    """Return the wall-projected complex momentum response to an edge field."""
    _require_species(species)
    xp = _xp(context)
    nc = context.op.Nc
    _require_size(
        electric_hat, _modes(context) * context.op.Ne, "electric response field"
    )
    _require_finite(electric_hat, "electric response field", context)
    physical = context.p.species[species]
    electric = inverse(electric_hat, context.op.Ne, context, False)
    interpolated = sample_apply(
        context.op.H, electric, context.N, context.compute
    ).reshape(context.N, 3, nc)
    force = (
        physical.charge
        * fluid.n.reshape(context.N, 1, nc)
        * interpolated
    ).reshape(-1)
    return project_wall(fluid.wall, force, context).value


def finish_fluid(stage, species, context):
    """Project every momentum force and populate this species' RHS."""
    _require_species(species)
    xp = _xp(context)
    nc = context.op.Nc
    fluid = stage.fluid[species]
    physical = context.p.species[species]
    _require_size(
        stage.cache.E, _modes(context) * context.op.Ne, "field cache E"
    )
    electric = inverse(
        stage.cache.E,
        context.op.Ne,
        context,
        True,
        context.p.ref.electric,
    )
    interpolated = sample_apply(
        context.op.H, electric, context.N, context.compute
    ).reshape(context.N, 3, nc)
    electric_force = (
        physical.charge * fluid.n.reshape(context.N, 1, nc) * interpolated
    ).reshape(-1)
    projection = project_wall(
        fluid.wall, fluid.non_electric + electric_force, context
    )

    state = stage.rhs.fluid[species]
    state.n = -fluid.divergence_hat
    state.perp = forward(fluid.pdot_perp, nc, context)
    state.mom = forward(projection.value, 3 * nc, context)
    if species == 1:
        state.parallel = forward(fluid.pdot_parallel, nc, context)
    stage.terms[species] = SimpleNamespace(
        electric=electric_force,
        force=projection.value,
        reaction=projection.reaction,
        multiplier=projection.multiplier,
        multiplier_residual=projection.multiplier_residual,
    )
    wall_rows = sample_apply(
        context.op.T_full, projection.value, context.N, context.compute
    )
    stage.residuals[f"wall_fluid_{species}"] = wall_rows


def _integrate_cells(values, components, context):
    xp = _xp(context)
    nc = context.op.Nc
    _require_size(
        values, context.N * components * nc, "cell integral"
    )
    m0 = xp.real(xp.asarray(context.op.M0)).astype(xp.float64)
    sampled = values.reshape(context.N, components, nc)
    return (
        xp.sum(sampled * m0[None, None, :])
        * (2.0 * math.pi / context.N)
    )


def _integrate_edges(values, context):
    xp = _xp(context)
    _require_size(values, context.N * context.op.Ne, "edge integral")
    m1 = xp.real(xp.asarray(context.op.M1)).astype(xp.float64)
    return (
        xp.sum(values.reshape(context.N, context.op.Ne) * m1[None, :])
        * (2.0 * math.pi / context.N)
    )


def _physical_real(value, name):
    value = complex(value)
    scale = max(1.0, abs(value))
    if abs(value.imag) > 4096.0 * sys.float_info.epsilon * scale:
        raise Failure("Hermitian", f"{name} has an imaginary physical value")
    return value.real


def power_ledger(stage, species, context):
    """Record physical work rates and spectral transfer defects."""
    _require_species(species)
    xp = _xp(context)
    nc = context.op.Nc
    fluid = stage.fluid[species]
    terms = stage.terms[species]
    physical = context.p.species[species]
    electric = inverse(
        stage.cache.E,
        context.op.Ne,
        context,
        True,
        context.p.ref.electric,
    )
    u = fluid.u.reshape(context.N, 3, nc)
    electric_node = xp.conj(u) * terms.electric.reshape(context.N, 3, nc)
    pressure_kinetic = xp.conj(u) * fluid.pressure_force.reshape(context.N, 3, nc)
    magnetic_work = xp.conj(u) * fluid.magnetic_force.reshape(context.N, 3, nc)
    wall_work = xp.conj(u) * terms.reaction.reshape(context.N, 3, nc)
    convection = xp.conj(u) * fluid.convection.reshape(context.N, 3, nc)
    ndot_back = inverse(stage.rhs.fluid[species].n, nc, context, True)
    pidot_back = inverse(
        stage.rhs.fluid[species].mom, 3 * nc, context, True
    ).reshape(context.N, 3, nc)
    speed_squared = xp.sum(xp.abs(u) ** 2, axis=1)
    density_term = 0.5 * physical.mass * speed_squared
    kinetic_retained = xp.conj(u) * pidot_back
    kinetic_retained[:, 0, :] -= (
        density_term * ndot_back.reshape(context.N, nc)
    )
    kinetic_sample = xp.conj(u) * terms.force.reshape(context.N, 3, nc)
    kinetic_sample[:, 0, :] -= (
        density_term * (-fluid.divergence.reshape(context.N, nc))
    )
    convection[:, 0, :] -= density_term * fluid.divergence.reshape(
        context.N, nc
    )
    pressure = _pressure_tensor(
        SimpleNamespace(perp=fluid.perp, parallel=fluid.parallel),
        species,
        stage.bhat,
        context,
    )
    gradient = fluid.gradient.reshape(context.N, 3, 3, nc)
    contraction = xp.sum(pressure * gradient, axis=(1, 2))
    electric_edge = (
        physical.charge
        * xp.conj(fluid.flux.reshape(context.N, context.op.Ne))
        * electric.reshape(context.N, context.op.Ne)
    )
    perp = fluid.perp.reshape(context.N, nc)
    theta = fluid.theta.reshape(context.N, nc)
    strain = fluid.strain.reshape(context.N, nc)
    internal_perp = (
        -perp * theta
        if species == 0
        else perp * (strain - theta)
    )
    perp_back = inverse(
        stage.rhs.fluid[species].perp,
        nc,
        context,
        True,
        context.p.ref.pressure / context.p.ref.time,
    ).reshape(context.N, nc)
    perp_factor = 1.0 / (physical.gamma - 1.0) if species == 0 else 1.0
    result = SimpleNamespace(
        electric_node=_physical_real(
            _integrate_cells(electric_node.reshape(-1), 3, context),
            "electric node work",
        ),
        electric_edge=_physical_real(
            _integrate_edges(electric_edge.reshape(-1), context),
            "electric edge work",
        ),
        pressure_kinetic=_physical_real(
            _integrate_cells(pressure_kinetic.reshape(-1), 3, context),
            "pressure kinetic work",
        ),
        pressure_internal=_physical_real(
            -_integrate_cells(contraction.reshape(-1), 1, context),
            "pressure internal work",
        ),
        magnetic=_physical_real(
            _integrate_cells(magnetic_work.reshape(-1), 3, context),
            "magnetic work",
        ),
        wall=_physical_real(
            _integrate_cells(wall_work.reshape(-1), 3, context),
            "wall work",
        ),
        convection=_physical_real(
            _integrate_cells(convection.reshape(-1), 3, context),
            "convection pairing",
        ),
        perp_rate=_physical_real(
            _integrate_cells(internal_perp.reshape(-1), 1, context),
            "perpendicular internal energy source",
        ),
        parallel_rate=0.0,
        spectral_kinetic=_physical_real(
            _integrate_cells(
                (kinetic_retained - kinetic_sample).reshape(-1), 3, context
            ),
            "kinetic spectral transfer",
        ),
        spectral_perp=_physical_real(
            _integrate_cells(
                ((perp_back - fluid.pdot_perp.reshape(context.N, nc)) * perp_factor)
                .reshape(-1),
                1,
                context,
            ),
            "perpendicular internal spectral transfer",
        ),
        spectral_parallel=0.0,
    )
    if species == 1:
        parallel = fluid.parallel.reshape(context.N, nc)
        internal_parallel = -parallel * strain
        parallel_back = inverse(
            stage.rhs.fluid[species].parallel,
            nc,
            context,
            True,
            context.p.ref.pressure / context.p.ref.time,
        ).reshape(context.N, nc)
        result.parallel_rate = _physical_real(
            _integrate_cells(internal_parallel.reshape(-1), 1, context),
            "parallel internal energy source",
        )
        result.spectral_parallel = _physical_real(
            _integrate_cells(
                (
                    0.5
                    * (
                        parallel_back
                        - fluid.pdot_parallel.reshape(context.N, nc)
                    )
                ).reshape(-1),
                1,
                context,
            ),
            "parallel internal spectral transfer",
        )
    return result


def divergence(flux, context):
    """Sample-layout divergence retained for native physics callers."""
    _require_size(flux, context.N * context.op.Ne, "sampled flux")
    _require_finite(flux, "sampled flux", context)
    return _sampled_divergence(flux, context)
