"""Actual-face wall geometry and diagnostics for ``polygon_dual_v1``."""

from __future__ import annotations

import math
import numbers
import sys
from types import SimpleNamespace

import numpy as np

from .core import inverse
from .types import Failure


POLYGON_KIND = "polygon_dual_v1"
PLANE_FACE = "plane_face"


def _xp(context):
    return context.compute.xp


def geometry_kind(context) -> str:
    """Return the explicit geometry layout, with the old layout as fallback."""
    return str(
        getattr(
            context.geometry,
            "kind",
            getattr(context.op, "geometry_kind", "staircase_v1"),
        )
    )


def is_polygon(context) -> bool:
    return geometry_kind(context) == POLYGON_KIND


def _face_count(context) -> int:
    normals = getattr(context.geometry, "face_normals", None)
    if normals is None or getattr(normals, "ndim", 0) != 2:
        raise Failure("WallGeometry", "polygon face normals are absent")
    count = int(normals.shape[0])
    if count <= 0 or int(context.op.Ne) != count + int(context.op.Nc):
        raise Failure(
            "WallGeometry",
            "polygon electric layout is not [normal faces, phi cells]",
        )
    return count


def _close(left: float, right: float, *, label: str) -> None:
    scale = max(abs(left), abs(right), 1.0)
    if abs(left - right) > 256.0 * sys.float_info.epsilon * scale:
        raise Failure(
            "WallGeometry",
            f"inconsistent polygon {label}: {left!r} != {right!r}",
        )


def wall_frame(context, wall) -> SimpleNamespace:
    """Validate and return one actual plane-face frame."""
    if not is_polygon(context):
        raise Failure("WallGeometry", "polygon frame requested for legacy layout")
    if getattr(wall, "component", None) != PLANE_FACE:
        raise Failure(
            "WallGeometry",
            "polygon walls must use component='plane_face'",
        )
    face = getattr(wall, "face", None)
    if isinstance(face, bool) or not isinstance(face, numbers.Integral):
        raise Failure("WallGeometry", "polygon wall face must be an integer")
    face = int(face)
    face_count = _face_count(context)
    if face < 0 or face >= face_count:
        raise Failure("WallGeometry", "polygon wall face is out of range")
    if getattr(wall, "sign", None) != 1:
        raise Failure("WallGeometry", "polygon wall sign must be 1")

    normal = np.asarray(context.geometry.face_normals[face], dtype=np.float64)
    tangent = np.asarray(context.geometry.face_tangents[face], dtype=np.float64)
    if normal.shape != (2,) or tangent.shape != (2,):
        raise Failure("WallGeometry", "polygon face frame is not two-dimensional")
    normal_R = float(normal[0])
    normal_Z = float(normal[1])
    tangent_R = float(tangent[0])
    tangent_Z = float(tangent[1])
    if not all(
        math.isfinite(value)
        for value in (normal_R, normal_Z, tangent_R, tangent_Z)
    ):
        raise Failure("WallGeometry", "polygon wall frame is nonfinite")
    _close(
        math.hypot(normal_R, normal_Z),
        1.0,
        label="normal norm",
    )
    _close(
        float(getattr(wall, "normal_R", normal_R)),
        normal_R,
        label="normal_R",
    )
    _close(
        float(getattr(wall, "normal_Z", normal_Z)),
        normal_Z,
        label="normal_Z",
    )
    _close(tangent_R, -normal_Z, label="tangent_R")
    _close(tangent_Z, normal_R, label="tangent_Z")
    _close(
        float(getattr(wall, "tangent_R", tangent_R)),
        tangent_R,
        label="wall tangent_R",
    )
    _close(
        float(getattr(wall, "tangent_Z", tangent_Z)),
        tangent_Z,
        label="wall tangent_Z",
    )

    radius = float(getattr(wall, "radius", math.nan))
    table_radius = float(context.geometry.faces.R[face])
    if not math.isfinite(radius) or radius <= 0.0:
        raise Failure("WallGeometry", "polygon wall radius must be positive")
    _close(radius, table_radius, label="face radius")
    ell = float(context.geometry.face_lengths[face])
    if not math.isfinite(ell) or ell <= 0.0:
        raise Failure("WallGeometry", "polygon face length must be positive")
    area_per_radian = float(getattr(wall, "area_per_radian", math.nan))
    if not math.isfinite(area_per_radian) or area_per_radian <= 0.0:
        raise Failure(
            "WallGeometry",
            "polygon wall area_per_radian must be positive",
        )
    _close(area_per_radian, radius * ell, label="area_per_radian")

    face_vertices = context.geometry.face_vertices
    if len(face_vertices.shape) != 2 or face_vertices.shape[1] != 2:
        raise Failure("WallGeometry", "face_vertices must have shape (Nf,2)")
    vertices = tuple(int(value) for value in wall.vertices)
    if vertices != tuple(int(value) for value in face_vertices[face]):
        raise Failure("WallGeometry", "wall endpoints disagree with face_vertices")
    near_cell = int(getattr(wall, "near_cell", -1))
    owners = context.geometry.face_cells[face]
    if int(owners[0]) != near_cell:
        raise Failure("WallGeometry", "wall near_cell is not the first owner")

    return SimpleNamespace(
        face=face,
        edge=face,
        vertices=vertices,
        near_cell=near_cell,
        normal_R=normal_R,
        normal_Z=normal_Z,
        tangent_R=tangent_R,
        tangent_Z=tangent_Z,
        normal3=(normal_R, 0.0, normal_Z),
        tangent3=(tangent_R, 0.0, tangent_Z),
        radius=radius,
        ell=ell,
        area_per_radian=area_per_radian,
        sign=1,
    )


def wall_edge(context, wall) -> int:
    """Return the native electric face coordinate for one wall."""
    if is_polygon(context):
        return wall_frame(context, wall).edge
    nc = int(context.op.Nc)
    if wall.component == "r_face":
        return int(wall.face)
    if wall.component == "z_face":
        return len(context.geometry.r_faces.keys) + nc + int(wall.face)
    raise Failure("WallGeometry", f"unsupported wall component {wall.component!r}")


def signed_wall_extraction(context, backend):
    """Build one signed extraction matrix using native global face rows."""
    xp = backend.xp
    result = xp.zeros(
        (len(context.geometry.walls), int(context.op.Ne)),
        dtype=xp.complex128,
    )
    for row, wall in enumerate(context.geometry.walls):
        result[row, wall_edge(context, wall)] = float(wall.sign)
    return backend.asarray(result)


def wall_flux_samples(flux, context):
    """Extract one signed native face flux for each wall and sample."""
    xp = _xp(context)
    samples = xp.asarray(flux).reshape(context.N, context.op.Ne)
    result = xp.empty(
        (context.N, len(context.geometry.walls)),
        dtype=xp.complex128,
    )
    for index, wall in enumerate(context.geometry.walls):
        result[:, index] = (
            wall.sign * samples[:, wall_edge(context, wall)]
        )
    return result.reshape(-1)


def magnetic_alias_groups(context):
    """Return retained-B diagnostic bounds and emitted/topology names."""
    if is_polygon(context):
        nf = _face_count(context)
        nv = len(context.geometry.vertices.keys)
        nb = int(context.op.Nb)
        if nb != nv + nf:
            raise Failure(
                "WallGeometry",
                "polygon magnetic layout is not [vertices, faces]",
            )
        return (
            (0, nv, nb),
            ("phi", "tangent"),
            ("phi", "Z"),
            (
                "polygon_dual_v1: B=[B_phi(vertices),"
                "B_tangent(faces)]; normal interval is empty"
            ),
        )
    nz = len(context.geometry.z_faces.keys)
    nv = len(context.geometry.vertices.keys)
    return (
        (0, nz, nz + nv, int(context.op.Nb)),
        ("R", "phi", "Z"),
        ("R", "phi", "Z"),
        "",
    )


WALL_OBSERVATION_CHANNELS = (
    "B_R",
    "B_phi",
    "B_Z",
    "E_n",
    "density_ion",
    "density_electron",
    "u_ion_R",
    "u_ion_phi",
    "u_ion_Z",
    "u_electron_R",
    "u_electron_phi",
    "u_electron_Z",
    "perp_ion",
    "perp_electron",
    "parallel_ion",
    "parallel_electron",
    "flux_ion",
    "flux_electron",
)
_WALL_OBSERVATION_INDEX = {
    name: index
    for index, name in enumerate(WALL_OBSERVATION_CHANNELS)
}


class WallTraceBatch(list):
    """Per-wall device views with one declared NumPy observation batch."""


def _trace_batch_identity(state, stage, context):
    tag = getattr(stage, "tag", None)
    return (
        id(state),
        id(stage),
        id(stage.cache.E),
        id(state.B),
        id(state.fluid[0].mom),
        id(state.fluid[1].mom),
        id(context.geometry),
        id(context.op),
        int(context.N),
        int(getattr(context, "spectral_id", 0)),
        int(getattr(state, "step", -1)),
        int(getattr(state, "generation", -1)),
        float(getattr(state, "time", math.nan)).hex(),
        (
            int(getattr(tag, "step", -1)),
            int(getattr(tag, "generation", -1)),
            int(getattr(tag, "trial", -1)),
            int(getattr(tag, "spectral_context", -1)),
            float(getattr(tag, "time", math.nan)).hex(),
            str(getattr(tag, "stage", "")),
        ),
    )


def trace_batch_matches(batch, state, stage, context) -> bool:
    return getattr(batch, "identity", None) == _trace_batch_identity(
        state,
        stage,
        context,
    )


def _host_channel(host, name):
    return host.packed[..., _WALL_OBSERVATION_INDEX[name]]


def _host_observation(packed, normals, tangents, radius, area, walls):
    packed = np.asarray(packed, dtype=np.float64)
    host = SimpleNamespace(
        packed=packed,
        channels=WALL_OBSERVATION_CHANNELS,
        placement="explicit read-only wall observation host export",
        normal3=normals,
        tangent3=tangents,
        radius=radius,
        area=area,
        wall_indices=np.arange(len(walls), dtype=np.int64),
        walls=tuple(walls),
    )
    host.B = np.stack(
        tuple(_host_channel(host, name) for name in ("B_R", "B_phi", "B_Z")),
        axis=-1,
    )
    host.electric_normal = _host_channel(host, "E_n")
    host.electric_wall = (
        host.electric_normal[..., None] * normals[None, :, :]
    )
    host.density = np.stack(
        tuple(
            _host_channel(host, name)
            for name in ("density_ion", "density_electron")
        ),
        axis=0,
    )
    host.velocity = np.stack(
        tuple(
            np.stack(
                tuple(
                    _host_channel(host, name)
                    for name in names
                ),
                axis=-1,
            )
            for names in (
                ("u_ion_R", "u_ion_phi", "u_ion_Z"),
                ("u_electron_R", "u_electron_phi", "u_electron_Z"),
            )
        ),
        axis=0,
    )
    host.perp = np.stack(
        tuple(
            _host_channel(host, name)
            for name in ("perp_ion", "perp_electron")
        ),
        axis=0,
    )
    host.parallel = np.stack(
        tuple(
            _host_channel(host, name)
            for name in ("parallel_ion", "parallel_electron")
        ),
        axis=0,
    )
    host.normal_flux = np.stack(
        tuple(
            _host_channel(host, name)
            for name in ("flux_ion", "flux_electron")
        ),
        axis=0,
    )
    return host


def host_traction_components(host, context):
    """Compute wall tractions once with NumPy after the declared export."""
    normal = host.normal3[None, :, :]
    b = host.B
    b2 = np.sum(b * b, axis=-1)
    bmag = np.sqrt(b2)
    if np.any(bmag <= 0.0):
        raise Failure("WallTrace", "wall tangential magnetic field is zero")
    b_dot_normal = np.sum(b * normal, axis=-1)
    bhat = b / bmag[..., None]
    un = np.sum(host.velocity[0] * normal, axis=-1)
    transport = (
        context.p.species[0].mass
        * host.density[0][..., None]
        * host.velocity[0]
        * un[..., None]
        + host.perp[0][..., None] * normal
    )
    maxwell = (
        0.5 * b2[..., None] * normal
        - b_dot_normal[..., None] * b
    ) / context.p.mu0
    electron_pressure = (
        host.perp[1][..., None] * normal
        + (host.parallel[1] - host.perp[1])[..., None]
        * bhat
        * (b_dot_normal / bmag)[..., None]
    )
    electromagnetic = maxwell + electron_pressure
    return transport, electromagnetic, transport + electromagnetic


def host_boundary_integrals(host, context):
    """Return NumPy boundary power/flux integrals from wall observations."""
    area = host.area[None, :]
    un_flux = host.normal_flux / host.density
    speed2 = np.sum(host.velocity * host.velocity, axis=-1)
    kinetic_density = np.zeros_like(host.density)
    kinetic_density[0] = (
        0.5 * context.p.species[0].mass * host.density[0] * speed2[0]
    )
    un_velocity = np.sum(
        host.velocity * host.normal3[None, :, :],
        axis=-1,
    )
    internal_density = np.stack(
        (
            host.perp[0] / (context.p.species[0].gamma - 1.0),
            host.perp[1] + 0.5 * host.parallel[1],
        ),
        axis=0,
    )
    pressure_work = host.perp * un_velocity
    bmag = np.sqrt(np.sum(host.B * host.B, axis=-1))
    if np.any(bmag <= 0.0):
        raise Failure("WallTrace", "wall tangential magnetic field is zero")
    bhat = host.B / bmag[..., None]
    pressure_work[1] += (
        (host.parallel[1] - host.perp[1])
        * np.sum(bhat * host.velocity[1], axis=-1)
        * np.sum(bhat * host.normal3[None, :, :], axis=-1)
    )
    number = np.sum(area * host.normal_flux, axis=(1, 2))
    kinetic = np.sum(
        area * kinetic_density * un_flux,
        axis=(1, 2),
    )
    internal_perp = np.stack(
        (
            np.sum(
                area * host.perp[0] * un_flux[0]
                / (context.p.species[0].gamma - 1.0),
                axis=(0, 1),
            ),
            np.sum(
                area * host.perp[1] * un_flux[1],
                axis=(0, 1),
            ),
        ),
        axis=0,
    )
    internal_parallel = np.array(
        (
            0.0,
            np.sum(
                area * 0.5 * host.parallel[1] * un_flux[1],
                axis=(0, 1),
            ),
        )
    )
    energy = np.sum(
        area[None, :, :]
        * (
            (kinetic_density + internal_density) * un_flux
            + pressure_work
        ),
        axis=(1, 2),
    )
    return SimpleNamespace(
        number=number,
        kinetic=kinetic,
        internal_perp=internal_perp,
        internal_parallel=internal_parallel,
        energy=energy,
        total_energy=float(np.sum(energy)),
        field_energy=0.0,
    )


def host_traction_integrals(host, role, samples):
    if role not in ("transport", "electromagnetic"):
        raise Failure(
            "WRTraction",
            "traction_role must be 'transport' or 'electromagnetic'",
        )
    components = (
        host.transport if role == "transport" else host.electromagnetic
    )
    outward = np.zeros(3)
    torque = 0.0
    for q in range(samples):
        sample = 0 if host.packed.shape[0] == 1 else q
        angle = 2.0 * math.pi * q / samples
        cosine = math.cos(angle)
        sine = math.sin(angle)
        cartesian = np.stack(
            (
                components[sample, :, 0] * cosine
                - components[sample, :, 1] * sine,
                components[sample, :, 0] * sine
                + components[sample, :, 1] * cosine,
                components[sample, :, 2],
            ),
            axis=1,
        )
        weights = (
            host.area * host.packed.shape[0]
            / samples
        )
        outward += np.sum(weights[:, None] * cartesian, axis=0)
        torque += float(
            np.sum(weights * host.radius * components[sample, :, 1])
        )
    return outward, torque


def _batched_wall_trace_data(state, stage, context):
    xp = _xp(context)
    nc = int(context.op.Nc)
    ne = int(context.op.Ne)
    walls = tuple(context.geometry.walls)
    if not walls:
        raise Failure("WallTrace", "polygon domain has no closed wall")
    frames = tuple(wall_frame(context, wall) for wall in walls)
    faces = np.array([frame.face for frame in frames], dtype=np.int64)
    vertices = np.array(
        [frame.vertices for frame in frames],
        dtype=np.int64,
    )
    normal3 = np.array([frame.normal3 for frame in frames], dtype=np.float64)
    tangent3 = np.array([frame.tangent3 for frame in frames], dtype=np.float64)
    radius = np.array([frame.radius for frame in frames], dtype=np.float64)
    area = np.array(
        [
            frame.area_per_radian * 2.0 * math.pi / context.N
            for frame in frames
        ],
        dtype=np.float64,
    )
    signs = xp.asarray(
        [wall.sign for wall in walls],
        dtype=xp.float64,
    )
    s_boundary = xp.asarray(context.op.S[faces], dtype=xp.float64)
    electric = inverse(
        stage.cache.E,
        ne,
        context,
        True,
        context.p.ref.electric,
    ).reshape(context.N, ne)
    magnetic = inverse(
        state.B,
        int(context.op.Nb),
        context,
        True,
    ).reshape(context.N, int(context.op.Nb))
    sampled_fields = []
    for species in range(2):
        fluid = stage.fluid[species]
        sampled_fields.append(
            SimpleNamespace(
                density=xp.real(xp.asarray(fluid.n)).reshape(
                    context.N,
                    nc,
                ),
                momentum=xp.real(
                    inverse(
                        state.fluid[species].mom,
                        3 * nc,
                        context,
                        True,
                    )
                ).reshape(context.N, 3, nc),
                perp=xp.real(xp.asarray(fluid.perp)).reshape(
                    context.N,
                    nc,
                ),
                parallel=(
                    xp.real(xp.asarray(fluid.parallel)).reshape(
                        context.N,
                        nc,
                    )
                    if species
                    else None
                ),
                flux=xp.real(xp.asarray(fluid.flux)).reshape(
                    context.N,
                    ne,
                ),
            )
        )
    density = [
        xp.einsum(
            "nc,wc->nw",
            sampled_fields[species].density,
            s_boundary,
        )
        for species in range(2)
    ]
    for species in range(2):
        floor = float(context.p.species[species].face_min)
        if not bool(xp.all(density[species] > floor).item()):
            raise Failure("WallDensity", "wall trace below density floor")
    momentum_trace = [
        xp.einsum(
            "nmc,wc->nmw",
            sampled_fields[species].momentum,
            s_boundary,
        )
        for species in range(2)
    ]
    masses = []
    for species in range(2):
        mass = float(context.p.species[species].mass)
        if not math.isfinite(mass) or mass <= 0.0:
            raise Failure("WallTrace", "wall species mass must be positive")
        masses.append(mass)
    velocity = [
        xp.swapaxes(momentum_trace[species], 1, 2)
        / (
            masses[species] * density[species][:, :, None]
        )
        for species in range(2)
    ]
    perpendicular = [
        xp.einsum(
            "nc,wc->nw",
            sampled_fields[species].perp,
            s_boundary,
        )
        for species in range(2)
    ]
    parallel = [
        perpendicular[0],
        xp.einsum(
            "nc,wc->nw",
            sampled_fields[1].parallel,
            s_boundary,
        ),
    ]
    normal_flux = [
        sampled_fields[species].flux[:, faces] * signs[None, :]
        for species in range(2)
    ]
    nv = len(context.geometry.vertices.keys)
    bphi = 0.5 * (
        xp.real(magnetic[:, vertices[:, 0]])
        + xp.real(magnetic[:, vertices[:, 1]])
    )
    bt = xp.real(magnetic[:, nv + faces])
    bwall = xp.empty(
        (context.N, len(walls), 3),
        dtype=xp.float64,
    )
    tangent_backend = xp.asarray(tangent3, dtype=xp.float64)
    normal_backend = xp.asarray(normal3, dtype=xp.float64)
    bwall[:, :, 0] = bt * tangent_backend[None, :, 0]
    bwall[:, :, 1] = bphi
    bwall[:, :, 2] = bt * tangent_backend[None, :, 2]
    electric_normal = xp.real(electric[:, faces])
    electric_wall = (
        electric_normal[:, :, None] * normal_backend[None, :, :]
    )
    packed_device = xp.stack(
        (
            bwall[:, :, 0],
            bwall[:, :, 1],
            bwall[:, :, 2],
            electric_normal,
            density[0],
            density[1],
            velocity[0][:, :, 0],
            velocity[0][:, :, 1],
            velocity[0][:, :, 2],
            velocity[1][:, :, 0],
            velocity[1][:, :, 1],
            velocity[1][:, :, 2],
            perpendicular[0],
            perpendicular[1],
            parallel[0],
            parallel[1],
            normal_flux[0],
            normal_flux[1],
        ),
        axis=-1,
    )
    packed_host = context.compute.to_host(packed_device)
    host = _host_observation(
        packed_host,
        normal3,
        tangent3,
        radius,
        area,
        walls,
    )
    host.transport, host.electromagnetic, host.traction = (
        host_traction_components(host, context)
    )
    host.boundary = host_boundary_integrals(host, context)
    result = WallTraceBatch()
    for wall_index, (wall, frame) in enumerate(zip(walls, frames)):
        view = SimpleNamespace(
            wall=wall,
            frame=frame,
            edge=frame.edge,
            direction=None,
            sigma=frame.sign,
            normal3=normal_backend[wall_index],
            tangent3=tangent_backend[wall_index],
            radius=frame.radius,
            area=float(area[wall_index]),
            B=tuple(
                bwall[:, wall_index, component]
                for component in range(3)
            ),
            Bwall=bwall[:, wall_index, :],
            electric_normal=electric_normal[:, wall_index],
            electric_wall=electric_wall[:, wall_index, :],
            density=tuple(
                density[species][:, wall_index]
                for species in range(2)
            ),
            velocity=tuple(
                velocity[species][:, wall_index, :]
                for species in range(2)
            ),
            perp=tuple(
                perpendicular[species][:, wall_index]
                for species in range(2)
            ),
            parallel=tuple(
                parallel[species][:, wall_index]
                for species in range(2)
            ),
            normal_flux=tuple(
                normal_flux[species][:, wall_index]
                for species in range(2)
            ),
            host=host,
            host_index=wall_index,
        )
        result.append(view)
    result.host = host
    result.identity = _trace_batch_identity(state, stage, context)
    return result


def wall_trace_data(state, stage, context):
    """Return actual-face traces for every polygon wall."""
    cached = getattr(getattr(stage, "boundary", None), "_trace_data", None)
    if trace_batch_matches(cached, state, stage, context):
        return cached
    return _batched_wall_trace_data(state, stage, context)


def maxwell_traction(b, normal, mu0, xp=None):
    """Return ``(B^2 I/2 - B B).n/mu0`` for one physical sample."""
    if xp is None:
        xp = np
    b = xp.asarray(b, dtype=xp.float64)
    normal = xp.asarray(normal, dtype=xp.float64)
    magnetic_pressure = 0.5 * xp.sum(b * b) / mu0
    return magnetic_pressure * normal - (
        xp.sum(b * normal) / mu0
    ) * b


def traction_components(data, sample, context):
    """Return ion transport and electromagnetic traction at one sample."""
    host = getattr(data, "host", None)
    if host is not None:
        wall_index = int(data.host_index)
        return (
            host.transport[sample, wall_index],
            host.electromagnetic[sample, wall_index],
        )
    xp = _xp(context)
    b = xp.stack(
        (
            data.B[0][sample],
            data.B[1][sample],
            data.B[2][sample],
        )
    )
    normal = xp.asarray(data.normal3, dtype=xp.float64)
    velocity = data.velocity[0][sample]
    density = data.density[0][sample]
    un = xp.real(xp.sum(velocity * normal))
    b2 = xp.sum(b * b)
    bmag = xp.sqrt(b2)
    if float(bmag.item()) <= 0.0:
        raise Failure("WallTrace", "wall tangential magnetic field is zero")
    bhat = b / bmag
    b_dot_normal = xp.real(xp.sum(b * normal))
    ion_transport = (
        float(context.p.species[0].mass) * density * velocity * un
        + data.perp[0][sample] * normal
    )
    electron_pressure = (
        data.perp[1][sample] * normal
        + (data.parallel[1][sample] - data.perp[1][sample])
        * bhat
        * (b_dot_normal / bmag)
    )
    electromagnetic = (
        maxwell_traction(
            b,
            normal,
            float(context.p.mu0),
            xp=xp,
        )
        + electron_pressure
    )
    return ion_transport, electromagnetic


def wall_traction(data, sample, context):
    host = getattr(data, "host", None)
    if host is not None:
        return host.traction[sample, int(data.host_index)]
    transport, electromagnetic = traction_components(
        data,
        sample,
        context,
    )
    return transport + electromagnetic


def wall_poynting_normal(
    electric_wall,
    magnetic_wall,
    normal,
    xp=None,
):
    """Return ``n.(Ewall x Bwall)`` for the ideal closed-wall trace."""
    if xp is None:
        xp = np
    electric_wall = xp.asarray(electric_wall, dtype=xp.float64)
    magnetic_wall = xp.asarray(magnetic_wall, dtype=xp.float64)
    normal = xp.asarray(normal, dtype=xp.float64)
    return xp.sum(xp.cross(electric_wall, magnetic_wall) * normal, axis=-1)
