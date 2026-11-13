"""Fixed normal magnetic flux for the grid-aligned relative complex."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .core import inverse, mode_apply, sample_apply
from .operators import apply_t3
from .types import Failure


def install_normal_flux(operators, geometry, flux):
    """Install outward Wb/rad, mode-major by wall segment and endpoint.

    Each of the two half-segments of a wall face closes the dual control
    volume at its corresponding endpoint. At a corner both sides contribute.
    The setup input is host complex128. The resulting divergence offset is
    owned by the operator backend and is constant across stages and restart.
    """
    if operators.wall_magnetic_divergence is not None:
        raise ValueError("rectangle normal magnetic flux is immutable")
    modes = 2 * operators.K + 1
    expected = (modes, len(geometry.walls), 2)
    if flux.dtype != np.dtype(np.complex128) or flux.shape != expected:
        raise ValueError(f"wall_magnetic_flux must be complex128 with shape {expected}")
    if not np.all(np.isfinite(flux)):
        raise ValueError("wall_magnetic_flux contains nonfinite values")
    envelope = max(float(np.max(np.abs(flux))), np.finfo(np.float64).tiny)
    if np.max(np.abs(flux - flux[::-1].conj())) > 32 * np.finfo(np.float64).eps * envelope:
        raise ValueError("wall_magnetic_flux must have Hermitian Fourier symmetry")
    cap = np.zeros((modes, len(geometry.vertices.keys)), dtype=np.complex128)
    for face, wall in enumerate(geometry.walls):
        cap[:, wall.vertices[0]] += flux[:, face, 0]
        cap[:, wall.vertices[1]] += flux[:, face, 1]
    cap /= geometry.wv[None, :]
    backend = operators.compute
    operators.wall_magnetic_divergence = backend.xp.stack(
        [apply_t3(operators, backend.asarray(row)) for row in cap]
    ).reshape(-1)
    xp = backend.xp
    nc, nr = operators.Nc, len(geometry.r_faces.keys)
    nz, nv = len(geometry.z_faces.keys), len(geometry.vertices.keys)
    sides = geometry.walls
    radial = np.array([wall.component == "r_face" for wall in sides])
    signs = np.array([wall.sign for wall in sides])
    faces = np.array([wall.face for wall in sides])
    areas = np.array([wall.area_per_radian for wall in sides])
    directions = xp.asarray(np.where(radial, 0, 2))
    edge = xp.asarray(np.where(radial, faces, nr + nc + faces))
    normal = xp.asarray(flux.sum(axis=2) * signs / areas, dtype=xp.complex128)
    rows = operators.I[edge].reshape(len(sides), 3, nc)
    operators.box_wall = SimpleNamespace(
        area=xp.asarray(areas),
        sign=xp.asarray(signs),
        direction=directions,
        tangent=2 - directions,
        normal_hat=normal.reshape(-1),
        magnetic_index=xp.asarray(np.where(radial, nz + nv + faces, faces)),
        toroidal_index=xp.asarray(np.array([wall.vertices for wall in sides]) + nz),
        interpolation=rows[xp.arange(len(sides)), directions],
    )


def wall_magnetic_trace(B, context):
    """Physical R, phi, Z at each wall segment in backend sample layout.

    The normal trace is the prescribed area mean. Tangential components use
    the selected wall-adjacent magnetic reconstruction.
    """
    wall = context.op.box_wall
    xp = context.compute.xp
    count = wall.area.size
    trace = xp.empty((context.N, 3, count), dtype=xp.float64)
    row = xp.arange(count)
    trace[:, wall.direction, row] = inverse(wall.normal_hat, count, context, True).real.reshape(context.N, count)
    reconstruction = context.op.wall_magnetic_reconstruction
    if reconstruction is None:
        values = inverse(B, context.op.Nb, context, True).real.reshape(context.N, -1)
        trace[:, wall.tangent, row] = values[:, wall.magnetic_index]
        trace[:, 1, :] = values[:, wall.toroidal_index].mean(axis=-1)
    else:
        from .ghost_toroidal import toroidal_trace

        op = context.op
        modes = 2 * op.K + 1
        samples = sample_apply(reconstruction, B, modes, context.compute)
        trace[:, wall.tangent, row] = inverse(samples, count, context, True).real.reshape(context.N, count)
        start = op.PZ_z.shape[0]
        vertices = B.reshape(modes, op.Nb)[:, start:start + op.Rbar_v.size]
        radial, axial = toroidal_trace(op.ghost_toroidal, vertices, context.compute)
        toroidal = xp.concatenate((radial.reshape(modes, -1), axial.reshape(modes, -1)), axis=1)
        samples = toroidal[:, op.wall_toroidal_trace_indices].reshape(-1)
        trace[:, 1, :] = inverse(samples, count, context, True).real.reshape(context.N, count)
    return trace


def electron_pressure_traction(B, fluid, context):
    """Return the negative weighted transpose of the actual pressure trace."""
    wall = context.op.box_wall
    xp = context.compute.xp
    b = wall_magnetic_trace(B, context)
    density = fluid.n.real.reshape(context.N, -1)
    rows = wall.interpolation.real
    wall_density = density @ rows.T
    if bool(xp.any(wall_density <= 0).item()):
        raise Failure("WallDensity", "wall pressure trace requires positive density")
    pp = fluid.perp.real.reshape(context.N, -1) @ rows.T
    pa = fluid.parallel.real.reshape(context.N, -1) @ rows.T
    b2 = xp.sum(b*b, axis=1)
    if bool(xp.any(b2 <= 0).item()):
        raise Failure("Magnetization", "wall pressure trace requires nonzero magnetic field")
    j = xp.arange(wall.area.size)
    bn = b[:, wall.direction, j] * wall.sign
    traction = b * ((pa - pp) * bn / b2)[:, None, :]
    traction[:, wall.direction, j] += pp * wall.sign
    load = ((traction * (wall.area / wall_density)[:, None, :]) @ rows)
    force = -load * (density / context.op.M0)[..., None, :]
    return xp.asarray(force, dtype=xp.complex128).reshape(-1)


def magnetic_divergence(B, context):
    """Evaluate total field divergence including prescribed wall flux in T/m."""
    internal = mode_apply(context.op.DB, B, context.compute)
    cap = getattr(context.op, "wall_magnetic_divergence", None)
    return internal if cap is None else internal + cap
