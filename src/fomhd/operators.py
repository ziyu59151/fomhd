"""Compatible RZ/Fourier complex and fixed constraint operators."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from .backend import Backend
from .geometry import Geometry, Wall
from .types import Failure, operator_check_record


_I = np.complex128(1j)
_EPS = np.finfo(np.float64).eps


@dataclass
class OperatorOptions:
    rank_rtol: float = 1e-10
    memory_cap_bytes: int = 512 << 20


@dataclass
class Operators:
    K: int = 0
    Nc: int = 0
    Ne: int = 0
    Nb: int = 0
    M0: Any = None
    M1: Any = None
    MB: Any = None
    box_magnetic_norm: tuple[Any, Any, Any, float] | None = None
    Rbar_r: Any = None
    Rbar_v: Any = None
    b1r: Any = None
    b1z: Any = None
    b2r: Any = None
    b2z: Any = None
    PR_r: Any = None
    PZ_r: Any = None
    PR_z: Any = None
    PZ_z: Any = None
    PR_v: Any = None
    PZ_v: Any = None
    PR_r_factor: Any = None
    PZ_r_factor: Any = None
    PR_z_factor: Any = None
    PZ_z_factor: Any = None
    PR_v_factor: Any = None
    PZ_v_factor: Any = None
    G: list[Any] = field(default_factory=list)
    C: list[Any] = field(default_factory=list)
    DB: list[Any] = field(default_factory=list)
    DJ: list[Any] = field(default_factory=list)
    Cdag: list[Any] = field(default_factory=list)
    Lphi: list[Any] = field(default_factory=list)
    phi_factor: list[Any] = field(default_factory=list)
    I: Any = None
    H: Any = None
    QB: Any = None
    DR: Any = None
    DZ: Any = None
    T_full: Any = None
    T: Any = None
    selected_wall_rows: list[int] = field(default_factory=list)
    Bc_full: Any = None
    Bc: Any = None
    selected_constraint_rows: list[int] = field(default_factory=list)
    topology_flux_row: Any = None
    setup_checks: list[SimpleNamespace] = field(default_factory=list)
    poisson_spectra: list[Any] = field(default_factory=list)
    compute: Any = None
    wall_magnetic_divergence: Any = None
    box_wall: Any = None


@dataclass
class SpatialOperators:
    mode: int = 0
    Nc: int = 0
    Ne: int = 0
    Nb: int = 0
    G: Any = None
    C: Any = None
    DB: Any = None
    DJ: Any = None


def _options(
    value: OperatorOptions | dict[str, Any] | None,
) -> OperatorOptions:
    if value is None:
        return OperatorOptions()
    if isinstance(value, OperatorOptions):
        return value
    return OperatorOptions(**value)


def _checked_product(left: int, right: int, name: str) -> int:
    if left < 0 or right < 0:
        raise Failure("Memory", f"{name} size is negative")
    if left != 0 and right > sys.maxsize // left:
        raise Failure("Memory", f"{name} size product overflows")
    return left * right


def _checked_sum(left: int, right: int, name: str) -> int:
    if right > sys.maxsize - left:
        raise Failure("Memory", f"{name} size sum overflows")
    return left + right


def _require_memory(required: int, cap: int, name: str) -> None:
    if cap != 0 and required > cap:
        raise Failure(
            "Memory", f"{name} memory budget exceeded: required={required} limit={cap}"
        )


def _diagonal_inverse(values: Any, backend: Backend) -> Any:
    if bool(backend.xp.any(~backend.xp.isfinite(values)).item()) or bool(
        backend.xp.any(values <= 0.0).item()
    ):
        raise Failure("Memory", "diagonal inverse requires positive finite weights")
    return backend.asarray(backend.xp.diag(1.0 / values))


def _identity(size: int, backend: Backend) -> Any:
    return backend.asarray(backend.xp.eye(size, dtype=backend.xp.complex128))


def _diagonal(values: Any, backend: Backend) -> Any:
    if bool(backend.xp.any(~backend.xp.isfinite(values)).item()):
        raise Failure("Linear", "diagonal contains a non-finite value")
    return backend.asarray(backend.xp.diag(values))


def _scale_rows(matrix: Any, scale: Any, backend: Backend) -> Any:
    return backend.asarray(matrix * scale[:, None])


def _scale_columns(matrix: Any, scale: Any, backend: Backend) -> Any:
    return backend.asarray(matrix * scale[None, :])


def _adjoint(matrix: Any, backend: Backend) -> Any:
    return backend.asarray(matrix.T.conj())


def _set_block(
    destination: Any,
    row: int,
    column: int,
    block: Any,
    factor: complex = 1.0,
) -> None:
    rows, columns = block.shape
    if row + rows > destination.shape[0] or column + columns > destination.shape[1]:
        raise Failure("Shape", "block placement exceeds destination shape")
    destination[row : row + rows, column : column + columns] += factor * block


def _right_solve(block: Any, factor: Any, backend: Backend) -> Any:
    return _adjoint(backend.solve(factor, _adjoint(block, backend)), backend)


def _graph_filter(entities: Any, radial_axis: bool, backend: Backend) -> Any:
    count = len(entities.keys)
    result = np.eye(count, dtype=np.complex128)
    for first, key in enumerate(entities.keys):
        if entities.boundary[first]:
            continue
        following_key = (
            (key[0] + 1, key[1]) if radial_axis else (key[0], key[1] + 1)
        )
        second = entities.lookup.get(following_key)
        if second is None or entities.boundary[second]:
            continue
        result[first, first] += 1.0 / 24.0
        result[second, second] += 1.0 / 24.0
        result[first, second] -= 1.0 / 24.0
        result[second, first] -= 1.0 / 24.0
    return backend.asarray(result)


def _apply_graph_filter_host(
    entities: Any, values: np.ndarray, radial_axis: bool
) -> np.ndarray:
    if values.size != len(entities.keys):
        raise Failure("Shape", "graph-filter input size mismatch")
    result = values.copy()
    for first, key in enumerate(entities.keys):
        if entities.boundary[first]:
            continue
        for forward in (False, True):
            if radial_axis:
                if not forward and key[0] == 0:
                    continue
                neighbor_key = (key[0] + (1 if forward else -1), key[1])
            else:
                if not forward and key[1] == 0:
                    continue
                neighbor_key = (key[0], key[1] + (1 if forward else -1))
            second = entities.lookup.get(neighbor_key)
            if second is None or entities.boundary[second]:
                continue
            result[first] += (values[first] - values[second]) / 24.0
    return result


def _make_g2(
    geometry: Geometry,
    mode: int,
    g_r: Any,
    g_z: Any,
    backend: Backend,
) -> Any:
    nc = len(geometry.cells.keys)
    nr = len(geometry.r_faces.keys)
    nz = len(geometry.z_faces.keys)
    result = backend.asarray(
        backend.xp.zeros((nc + nr + nz, nc), dtype=backend.xp.complex128)
    )
    _set_block(result, 0, 0, g_r)
    radius = backend.asarray(geometry.cells.R)
    result[nr : nr + nc, :] = backend.xp.diag(
        _I * float(mode) / radius
    )
    _set_block(result, nr + nc, 0, g_z)
    return result


def _make_c2(
    geometry: Geometry,
    mode: int,
    g_r: Any,
    g_z: Any,
    b1r: Any,
    b1z: Any,
    backend: Backend,
) -> Any:
    nc = len(geometry.cells.keys)
    nr = len(geometry.r_faces.keys)
    nz = len(geometry.z_faces.keys)
    nv = len(geometry.vertices.keys)
    result = backend.asarray(
        backend.xp.zeros((nz + nv + nr, nc + nr + nz), dtype=backend.xp.complex128)
    )
    z_radius = backend.asarray(geometry.z_faces.R)
    cell_radius = backend.asarray(geometry.cells.R)
    result[0:nz, nr + nc :] += (
        _I * float(mode) * backend.xp.diag(1.0 / z_radius)
    )
    result[0:nz, nr : nr + nc] = -(
        _scale_columns(g_z, cell_radius, backend) / z_radius[:, None]
    )
    lr = backend.asarray(geometry.lr)
    av = backend.asarray(geometry.av)
    result[nz : nz + nv, 0:nr] = -(
        _adjoint(b1r, backend) * lr[None, :]
    ) / av[:, None]
    lz = backend.asarray(geometry.lz)
    result[nz : nz + nv, nr + nc :] = -(
        _adjoint(b1z, backend) * lz[None, :]
    ) / av[:, None]
    rbar_r = backend.asarray(geometry.Rbar_r)
    result[nz + nv :, nr : nr + nc] = (
        _scale_columns(g_r, cell_radius, backend) / rbar_r[:, None]
    )
    result[nz + nv :, 0:nr] += -_I * float(mode) * backend.xp.diag(1.0 / rbar_r)
    return result


def _make_db2(
    geometry: Geometry,
    mode: int,
    b1r: Any,
    b1z: Any,
    backend: Backend,
) -> Any:
    nr = len(geometry.r_faces.keys)
    nz = len(geometry.z_faces.keys)
    nv = len(geometry.vertices.keys)
    result = backend.asarray(
        backend.xp.zeros((nv, nz + nv + nr), dtype=backend.xp.complex128)
    )
    z_weight = backend.asarray(geometry.lz * geometry.z_faces.R)
    result[:, 0:nz] = _scale_columns(_adjoint(b1z, backend), z_weight, backend)
    result[:, 0:nz] /= backend.asarray(geometry.wv)[:, None]
    r_weight = backend.asarray(geometry.lr * geometry.Rbar_r)
    result[:, nz + nv :] = -_scale_columns(
        _adjoint(b1r, backend), r_weight, backend
    )
    result[:, nz + nv :] /= backend.asarray(geometry.wv)[:, None]
    result[:, nz : nz + nv] += (
        _I
        * float(mode)
        * (backend.asarray(geometry.av) / backend.asarray(geometry.wv))[:, None]
        * backend.xp.eye(nv, dtype=backend.xp.complex128)
    )
    return result


def _make_t1(operators: Operators, backend: Backend) -> Any:
    nc = operators.Nc
    nr = operators.PR_r.shape[0]
    nz = operators.PZ_z.shape[0]
    result = backend.asarray(
        backend.xp.zeros((nr + nc + nz, nr + nc + nz), dtype=backend.xp.complex128)
    )
    _set_block(result, 0, 0, operators.PR_r)
    _set_block(result, nr, nr, _identity(nc, backend))
    _set_block(result, nr + nc, nr + nc, operators.PZ_z)
    return result


def _make_t2(operators: Operators, rbar_r: Any, backend: Backend) -> Any:
    nc = operators.Nc
    nr = operators.PR_r.shape[0]
    nz = operators.PZ_z.shape[0]
    nv = operators.PR_v.shape[0]
    result = backend.asarray(
        backend.xp.zeros((nz + nv + nr, nz + nv + nr), dtype=backend.xp.complex128)
    )
    _set_block(result, 0, 0, operators.PZ_z)
    _set_block(result, nz, nz, backend.mm(operators.PR_v, operators.PZ_v))
    radial = backend.mm(
        _diagonal_inverse(rbar_r, backend),
        backend.mm(operators.PR_r, _diagonal(rbar_r, backend)),
    )
    _set_block(result, nz + nv, nz + nv, radial)
    return result


def _make_t3(geometry: Geometry, operators: Operators, backend: Backend) -> Any:
    rbar_v = backend.asarray(geometry.Rbar_v)
    return backend.mm(
        _diagonal_inverse(rbar_v, backend),
        backend.mm(
            operators.PR_v,
            backend.mm(operators.PZ_v, _diagonal(rbar_v, backend)),
        ),
    )


def _host_fixed_matrices(
    geometry: Geometry,
) -> dict[str, np.ndarray]:
    nc = len(geometry.cells.keys)
    nr = len(geometry.r_faces.keys)
    nz = len(geometry.z_faces.keys)
    nv = len(geometry.vertices.keys)
    b2r = np.zeros((nc, nr), dtype=np.complex128)
    b2z = np.zeros((nc, nz), dtype=np.complex128)
    b1r = np.zeros((nr, nv), dtype=np.complex128)
    b1z = np.zeros((nz, nv), dtype=np.complex128)
    for cell, (i, j) in enumerate(geometry.cells.keys):
        b2r[cell, geometry.r_faces.find(i + 1, j)] += 1.0
        b2r[cell, geometry.r_faces.find(i, j)] -= 1.0
        b2z[cell, geometry.z_faces.find(i, j + 1)] += 1.0
        b2z[cell, geometry.z_faces.find(i, j)] -= 1.0
    for face, (e, j) in enumerate(geometry.r_faces.keys):
        b1r[face, geometry.vertices.find(e, j + 1)] += 1.0
        b1r[face, geometry.vertices.find(e, j)] -= 1.0
    for face, (i, f) in enumerate(geometry.z_faces.keys):
        b1z[face, geometry.vertices.find(i, f)] += 1.0
        b1z[face, geometry.vertices.find(i + 1, f)] -= 1.0

    def local_filter(entities: Any, radial: bool) -> np.ndarray:
        result = np.eye(len(entities.keys), dtype=np.complex128)
        for first, key in enumerate(entities.keys):
            if entities.boundary[first]:
                continue
            following_key = (
                (key[0] + 1, key[1]) if radial else (key[0], key[1] + 1)
            )
            second = entities.lookup.get(following_key)
            if second is None or entities.boundary[second]:
                continue
            result[first, first] += 1.0 / 24.0
            result[second, second] += 1.0 / 24.0
            result[first, second] -= 1.0 / 24.0
            result[second, first] -= 1.0 / 24.0
        return result

    return {
        "b2r": b2r,
        "b2z": b2z,
        "b1r": b1r,
        "b1z": b1z,
        "PR_r": local_filter(geometry.r_faces, True),
        "PZ_r": local_filter(geometry.r_faces, False),
        "PR_z": local_filter(geometry.z_faces, True),
        "PZ_z": local_filter(geometry.z_faces, False),
        "PR_v": local_filter(geometry.vertices, True),
        "PZ_v": local_filter(geometry.vertices, False),
    }


def _make_fixed_spatial_state(
    geometry: Geometry, backend: Backend
) -> tuple[Operators, Any, Any, Any]:
    host = _host_fixed_matrices(geometry)
    operators = Operators(
        K=0,
        compute=backend,
        Nc=len(geometry.cells.keys),
        Ne=len(geometry.cells.keys) + len(geometry.r_faces.keys) + len(geometry.z_faces.keys),
        Nb=len(geometry.z_faces.keys) + len(geometry.vertices.keys) + len(geometry.r_faces.keys),
        M0=backend.real_array(geometry.M0),
        M1=backend.real_array(geometry.M1),
        MB=backend.real_array(geometry.MB),
        Rbar_r=backend.real_array(geometry.Rbar_r),
        Rbar_v=backend.real_array(geometry.Rbar_v),
        b1r=backend.asarray(host["b1r"]),
        b1z=backend.asarray(host["b1z"]),
        b2r=backend.asarray(host["b2r"]),
        b2z=backend.asarray(host["b2z"]),
        PR_r=backend.asarray(host["PR_r"]),
        PZ_r=backend.asarray(host["PZ_r"]),
        PR_z=backend.asarray(host["PR_z"]),
        PZ_z=backend.asarray(host["PZ_z"]),
        PR_v=backend.asarray(host["PR_v"]),
        PZ_v=backend.asarray(host["PZ_v"]),
    )
    for name in ("PR_r", "PZ_r", "PR_z", "PZ_z", "PR_v", "PZ_v"):
        setattr(operators, f"{name}_factor", backend.factor(getattr(operators, name)))
    t1 = _make_t1(operators, backend)
    t2 = _make_t2(operators, operators.Rbar_r, backend)
    t3 = _make_t3(geometry, operators, backend)
    return operators, t1, t2, t3


def _link_magnetic_norm(
    nodes: np.ndarray, offset: np.ndarray, weight: np.ndarray, chart: np.ndarray
) -> np.ndarray:
    """Fit constant and affine link averages in the energy metric."""
    shift = np.average(offset, weights=weight)
    gap = np.diff(nodes) + np.diff(offset)
    flux = np.cumsum(weight * (offset - shift))[:-1]
    if np.any(gap <= 0.0) or np.any(flux <= 0.0):
        raise Failure("Operators", "rectangular magnetic norm has invalid link moments")
    coupling = flux / gap
    raw = np.diag(weight)
    index = np.arange(coupling.size)
    raw[index, index] += coupling
    raw[index + 1, index + 1] += coupling
    raw[index, index + 1] = -coupling
    raw[index + 1, index] = -coupling
    left = np.linalg.solve(chart.T, raw)
    stored = np.linalg.solve(chart.T, left.T).T
    return 0.5 * (stored + stored.T)


def _box_magnetic_norm(
    geometry: Geometry, operators: Operators, backend: Backend
) -> tuple[Any, Any, Any, float]:
    """Build the two shared axial/radial line norms in the stored chart."""
    nr, nz = geometry.R.size, geometry.Z.size
    hr = float(geometry.R[1] - geometry.R[0])
    hz = float(geometry.Z[1] - geometry.Z[0])
    r_nodes = geometry.r_faces.R[:nr + 1]
    r_mean = geometry.Rbar_r[:nr + 1]
    lr = geometry.lr[:nr + 1]
    pr = backend.to_host(operators.PR_r[:nr + 1, :nr + 1]).real
    # The centroid of the sampled B_Z uses R dR rather than dR.
    radial = _link_magnetic_norm(
        r_nodes,
        r_mean - r_nodes + lr**2 / (12.0 * r_mean),
        r_mean * lr,
        pr * r_mean[None, :] / r_mean[:, None],
    )
    z_nodes = geometry.z_faces.Z[::nr]
    lz = geometry.lz[::nr]
    z_offset = np.zeros_like(lz)
    z_offset[0], z_offset[-1] = 0.5 * lz[0], -0.5 * lz[-1]
    pz = backend.to_host(operators.PZ_z[::nr, ::nr]).real
    axial = _link_magnetic_norm(z_nodes, z_offset, lz, pz)
    return (
        backend.asarray(radial),
        backend.asarray(axial),
        backend.real_array(hr * geometry.R),
        hz,
    )


def magnetic_weight(
    operators: Operators, values: Any, backend: Backend | None = None
) -> Any:
    """Apply the magnetic energy metric to stored, mode-major B values."""
    backend = backend or operators.compute
    xp = backend.xp
    b = xp.asarray(values).reshape(-1, operators.MB.size)
    norm = getattr(operators, "box_magnetic_norm", None)
    if norm is None:
        return (b * operators.MB).reshape(values.shape)
    radial, axial, radial_measure, hz = norm
    nr, nz = radial.shape[0] - 1, axial.shape[0] - 1
    split_r = nr * (nz + 1)
    split_z = split_r + (nr + 1) * (nz + 1)
    result = xp.empty_like(b)
    br = b[:, :split_r].reshape(-1, nz + 1, nr)
    bz = b[:, split_z:].reshape(-1, nz, nr + 1)
    result[:, :split_r] = ((axial @ br) * radial_measure).reshape(-1, split_r)
    result[:, split_r:split_z] = (
        b[:, split_r:split_z] * operators.MB[split_r:split_z]
    )
    result[:, split_z:] = (hz * (bz @ radial)).reshape(-1, (nr + 1) * nz)
    return result.reshape(values.shape)


def magnetic_pair(
    operators: Operators, left: Any, right: Any, backend: Backend
) -> Any:
    """Real magnetic inner product without the Fourier factor or permeability."""
    xp = backend.xp
    return xp.sum(xp.real(xp.conj(left) * magnetic_weight(operators, right, backend)))


def _select_columns(matrix: Any, first: int, count: int) -> Any:
    if first < 0 or count < 0 or first + count > matrix.shape[1]:
        raise Failure("Shape", "selected matrix columns are out of bounds")
    return matrix[:, first : first + count]


def _max_abs(matrix: Any, backend: Backend) -> float:
    return backend.max_abs(matrix)


def _assemble_spatial(
    geometry: Geometry,
    mode: int,
    fixed: Operators,
    t1: Any,
    t2: Any,
    t3: Any,
    verify_low_algebra: bool,
    options: OperatorOptions,
    backend: Backend,
) -> SpatialOperators:
    nc = len(geometry.cells.keys)
    nr = len(geometry.r_faces.keys)
    nz = len(geometry.z_faces.keys)
    nv = len(geometry.vertices.keys)
    ne = nc + nr + nz
    nb = nz + nv + nr
    lr = backend.asarray(geometry.lr)
    lz = backend.asarray(geometry.lz)
    g_r = backend.mm(_diagonal_inverse(lr, backend), _adjoint(fixed.b2r, backend))
    g_r = _scale_rows(g_r, -backend.asarray(backend.xp.ones(nr)), backend)
    g_z = backend.mm(_diagonal_inverse(lz, backend), _adjoint(fixed.b2z, backend))
    g_z = _scale_rows(g_z, -backend.asarray(backend.xp.ones(nz)), backend)
    g2 = _make_g2(geometry, mode, g_r, g_z, backend)
    c2 = _make_c2(geometry, mode, g_r, g_z, fixed.b1r, fixed.b1z, backend)
    db2 = _make_db2(geometry, mode, fixed.b1r, fixed.b1z, backend)
    if verify_low_algebra:
        low_cg = backend.mm(c2, g2)
        low_dbc = backend.mm(db2, c2)
        cg = _max_abs(low_cg, backend)
        dbc = _max_abs(low_dbc, backend)
        cg_bound = 128.0 * _EPS * ne * _max_abs(c2, backend) * _max_abs(g2, backend)
        dbc_bound = 128.0 * _EPS * nb * _max_abs(db2, backend) * _max_abs(c2, backend)
        if cg > cg_bound or dbc > dbc_bound:
            raise Failure(
                "Operators",
                f"low-order complex algebra failed: mode={mode} CG={cg} DBC={dbc}",
            )
    gradient = backend.mm(t1, g2)
    c2_transformed = backend.asarray(
        backend.xp.zeros((nb, ne), dtype=backend.xp.complex128)
    )
    _set_block(
        c2_transformed,
        0,
        0,
        _right_solve(_select_columns(c2, 0, nr), fixed.PR_r_factor, backend),
    )
    _set_block(c2_transformed, 0, nr, _select_columns(c2, nr, nc))
    _set_block(
        c2_transformed,
        0,
        nr + nc,
        _right_solve(_select_columns(c2, nr + nc, nz), fixed.PZ_z_factor, backend),
    )
    curl = backend.mm(t2, c2_transformed)
    db_transformed = apply_t2_inverse_block(fixed, db2, options, backend)
    divergence = backend.mm(t3, db_transformed)
    dj = backend.mm(
        _diagonal_inverse(fixed.M0, backend),
        backend.mm(_adjoint(gradient, backend), _diagonal(fixed.M1, backend)),
    )
    dj = backend.asarray(-dj)
    return SpatialOperators(
        mode=mode, Nc=nc, Ne=ne, Nb=nb, G=gradient, C=curl, DB=divergence, DJ=dj
    )


def _estimate_spatial_bytes(geometry: Geometry) -> int:
    nc = len(geometry.cells.keys)
    nr = len(geometry.r_faces.keys)
    nz = len(geometry.z_faces.keys)
    nv = len(geometry.vertices.keys)
    ne = nc + nr + nz
    nb = nz + nv + nr
    elements = 0

    def add(value: int) -> None:
        nonlocal elements
        elements = _checked_sum(elements, value, "spatial estimate")

    def pair(left: int, right: int, copies: int) -> None:
        add(_checked_product(_checked_product(left, right, "spatial estimate"), copies, "spatial estimate copies"))

    pair(ne, nc, 4)
    pair(nb, ne, 6)
    pair(nv, nb, 6)
    pair(nc, ne, 5)
    pair(nb, ne, 1)
    pair(nv, nb, 1)
    pair(ne, nc, 1)
    pair(nc, nr, 2)
    pair(nc, nz, 2)
    pair(nc, nr, 1)
    pair(nc, nz, 1)
    pair(nr, nv, 1)
    pair(nz, nv, 1)
    pair(nr, nr, 6)
    pair(nz, nz, 6)
    pair(nv, nv, 6)
    add(_checked_product(ne, ne, "spatial estimate"))
    pair(nb, nb, 2)
    return _checked_product(elements, 16, "spatial estimate bytes")


def _estimate_operator_bytes(geometry: Geometry, K: int) -> int:
    nc = len(geometry.cells.keys)
    nr = len(geometry.r_faces.keys)
    nz = len(geometry.z_faces.keys)
    nv = len(geometry.vertices.keys)
    ne = nc + nr + nz
    nb = nz + nv + nr
    modes = 2 * K + 1

    def square(value: int) -> int:
        return _checked_product(value, value, "operator estimate")

    elements = 0
    for value in (
        square(nr),
        square(nz),
        square(nv),
        square(nv),
        square(nc),
        square(nc),
        square(nr),
        square(nz),
        square(nc),
        square(nz),
        square(nc),
        square(nv),
        square(nv),
        _checked_product(ne, nc, "operator estimate"),
        _checked_product(nv, nb, "operator estimate"),
        _checked_product(ne, ne, "operator estimate"),
        _checked_product(ne, ne, "operator estimate"),
        _checked_product(ne, nb, "operator estimate"),
        _checked_product(nc, nr, "operator estimate"),
        _checked_product(nc, nz, "operator estimate"),
        _checked_product(len(geometry.walls), ne, "operator estimate"),
        _checked_product(
            modes, _checked_product(nc, ne, "operator estimate"), "operator estimate"
        ),
    ):
        elements = _checked_sum(elements, value, "operator estimate")
    per_mode_inner = _checked_sum(
        _checked_sum(
            _checked_product(ne, nc, "operator estimate"),
            _checked_product(nb, ne, "operator estimate"),
            "operator estimate",
        ),
        _checked_sum(
            _checked_product(nv, nb, "operator estimate"),
            square(nc),
            "operator estimate",
        ),
        "operator estimate",
    )
    per_mode = _checked_product(
        _checked_product(modes, per_mode_inner, "operator estimate"),
        2,
        "operator estimate",
    )
    constraint = _checked_product(
        modes,
        _checked_product(
            nc,
            _checked_product(
                modes,
                _checked_product(3, nc, "operator estimate"),
                "operator estimate",
            ),
            "operator estimate",
        ),
        "operator estimate",
    )
    elements = _checked_sum(elements, per_mode, "operator estimate")
    elements = _checked_sum(elements, constraint, "operator estimate")
    if geometry.lcfs_polygon is not None:
        elements = _checked_sum(
            elements,
            square(geometry.R.size + 1) + square(geometry.Z.size + 1) + geometry.R.size,
            "rectangular magnetic line norms",
        )
    byte_count = _checked_product(elements, 16, "operator estimate bytes")
    return _checked_sum(
        byte_count, byte_count // 2, "operator estimate with workspaces"
    )


def build_gradient_operator(
    geometry: Geometry,
    mode: int,
    options: OperatorOptions | dict[str, Any] | None = None,
    backend: Backend | None = None,
) -> Any:
    settings = _options(options)
    if not (0.0 < settings.rank_rtol < 1.0):
        raise Failure("Operators", "gradient operator rank tolerance is invalid")
    backend = backend or Backend("cpu")
    nc = len(geometry.cells.keys)
    nr = len(geometry.r_faces.keys)
    nz = len(geometry.z_faces.keys)
    ne = nc + nr + nz
    byte_count = _checked_product(
        _checked_product(ne, nc, "gradient operator"),
        16,
        "gradient operator bytes",
    )
    _require_memory(byte_count, settings.memory_cap_bytes, "gradient_operator")
    host = _host_fixed_matrices(geometry)
    g_r = np.diag(1.0 / geometry.lr) @ host["b2r"].T.conj()
    g_r = -g_r
    g_z = np.diag(1.0 / geometry.lz) @ host["b2z"].T.conj()
    g_z = -g_z
    gradient = np.zeros((ne, nc), dtype=np.complex128)
    gradient[0:nr] = g_r
    gradient[nr : nr + nc] = np.diag(_I * mode / geometry.cells.R)
    gradient[nr + nc :] = g_z
    for column in range(nc):
        gradient[0:nr, column] = _apply_graph_filter_host(
            geometry.r_faces, gradient[0:nr, column], True
        )
        gradient[nr + nc :, column] = _apply_graph_filter_host(
            geometry.z_faces, gradient[nr + nc :, column], False
        )
    return backend.asarray(gradient)


def build_spatial_operators(
    geometry: Geometry,
    mode: int,
    options: OperatorOptions | dict[str, Any] | None = None,
    backend: Backend | None = None,
) -> SpatialOperators:
    settings = _options(options)
    if not (0.0 < settings.rank_rtol < 1.0):
        raise Failure("Operators", "spatial operator rank tolerance is invalid")
    backend = backend or Backend("cpu")
    _require_memory(
        _estimate_spatial_bytes(geometry),
        settings.memory_cap_bytes,
        "spatial_operator_matrices",
    )
    fixed, t1, t2, t3 = _make_fixed_spatial_state(geometry, backend)
    return _assemble_spatial(
        geometry, mode, fixed, t1, t2, t3, False, settings, backend
    )


def apply_t2_inverse_block(
    operators: Operators,
    block: Any,
    options: OperatorOptions | dict[str, Any] | None = None,
    backend: Backend | None = None,
) -> Any:
    settings = _options(options)
    if not (0.0 < settings.rank_rtol < 1.0):
        raise Failure("Operators", "T2 inverse rank tolerance is invalid")
    backend = backend or operators.compute or Backend("cpu")
    nz = operators.PZ_z.shape[0]
    nv = operators.PR_v.shape[0]
    nr = operators.PR_r.shape[0]
    nb = nz + nv + nr
    if (
        operators.Nb != nb
        or operators.MB.size != nb
        or operators.Rbar_r.size != nr
        or block.ndim != 2
        or block.shape[1] != nb
    ):
        raise Failure("Shape", "T2 inverse operator shape mismatch")
    largest = max(nz, nv, nr)
    largest_block = _checked_product(largest, block.shape[0], "T2 inverse block workspace")
    elements = _checked_sum(
        _checked_sum(block.size, _checked_product(largest_block, 6, "T2 inverse workspaces"), "T2 inverse blocks"),
        _checked_product(largest, largest, "T2 inverse factor conversion"),
        "T2 inverse block elements",
    )
    _require_memory(
        _checked_product(elements, 16, "T2 inverse bytes"),
        settings.memory_cap_bytes,
        "T2_inverse",
    )
    transformed = backend.asarray(
        backend.xp.zeros(block.shape, dtype=backend.xp.complex128)
    )
    _set_block(
        transformed,
        0,
        0,
        _right_solve(_select_columns(block, 0, nz), operators.PZ_z_factor, backend),
    )
    vertex = _right_solve(
        _select_columns(block, nz, nv), operators.PZ_v_factor, backend
    )
    vertex = _right_solve(vertex, operators.PR_v_factor, backend)
    _set_block(transformed, 0, nz, vertex)
    if bool(backend.xp.any(operators.Rbar_r <= 0.0).item()):
        raise Failure("Memory", "R-face mean radius requires positive finite weights")
    radial = _scale_columns(
        _select_columns(block, nz + nv, nr),
        1.0 / operators.Rbar_r,
        backend,
    )
    radial = _right_solve(radial, operators.PR_r_factor, backend)
    radial = _scale_columns(radial, operators.Rbar_r, backend)
    _set_block(transformed, 0, nz + nv, radial)
    return transformed


def _select_rows(matrix: Any, rows: list[int], backend: Backend) -> Any:
    if any(row < 0 or row >= matrix.shape[0] for row in rows):
        raise Failure("Shape", "selected matrix row is out of bounds")
    return backend.asarray(matrix[rows, :])


def _block_diagonal(blocks: list[Any], backend: Backend) -> Any:
    rows = sum(block.shape[0] for block in blocks)
    columns = sum(block.shape[1] for block in blocks)
    result = backend.asarray(backend.xp.zeros((rows, columns), dtype=backend.xp.complex128))
    row_offset = 0
    column_offset = 0
    for block in blocks:
        _set_block(result, row_offset, column_offset, block)
        row_offset += block.shape[0]
        column_offset += block.shape[1]
    return result


def _host_interpolation(geometry: Geometry) -> tuple[np.ndarray, np.ndarray]:
    nc = len(geometry.cells.keys)
    nr = len(geometry.r_faces.keys)
    nz = len(geometry.z_faces.keys)
    i_r = np.zeros((nr, nc), dtype=np.complex128)
    for face, (e, j) in enumerate(geometry.r_faces.keys):
        if geometry.r_faces.boundary[face]:
            wall = next(
                wall
                for wall in geometry.walls
                if wall.component == "r_face" and wall.face == face
            )
            i_r[face, wall.near_cell] = 1.5
            i_r[face, wall.next_cell] = -0.5
            continue
        cells = [(e - 2, j), (e - 1, j), (e, j), (e + 1, j)]
        if all(geometry.cells.contains(i, j) for i, j in cells):
            weights = (-1.0 / 16.0, 9.0 / 16.0, 9.0 / 16.0, -1.0 / 16.0)
            for (i, jj), weight in zip(cells, weights):
                i_r[face, geometry.cells.find(i, jj)] = weight
        else:
            i_r[face, geometry.cells.find(e, j)] = 0.5
            i_r[face, geometry.cells.find(e - 1, j)] = 0.5
    i_z = np.zeros((nz, nc), dtype=np.complex128)
    for face, (i, f) in enumerate(geometry.z_faces.keys):
        if geometry.z_faces.boundary[face]:
            wall = next(
                wall
                for wall in geometry.walls
                if wall.component == "z_face" and wall.face == face
            )
            i_z[face, wall.near_cell] = 1.5
            i_z[face, wall.next_cell] = -0.5
            continue
        cells = [(i, f - 2), (i, f - 1), (i, f), (i, f + 1)]
        if all(geometry.cells.contains(ii, jj) for ii, jj in cells):
            weights = (-1.0 / 16.0, 9.0 / 16.0, 9.0 / 16.0, -1.0 / 16.0)
            for (ii, jj), weight in zip(cells, weights):
                i_z[face, geometry.cells.find(ii, jj)] = weight
        else:
            i_z[face, geometry.cells.find(i, f)] = 0.5
            i_z[face, geometry.cells.find(i, f - 1)] = 0.5
    return i_r, i_z


def _host_qb(geometry: Geometry) -> np.ndarray:
    nc = len(geometry.cells.keys)
    nr = len(geometry.r_faces.keys)
    nz = len(geometry.z_faces.keys)
    nv = len(geometry.vertices.keys)
    qb = np.zeros((3 * nc, nz + nv + nr), dtype=np.complex128)
    weights = (-1.0 / 16.0, 9.0 / 16.0, 9.0 / 16.0, -1.0 / 16.0)
    for cell, (i, j) in enumerate(geometry.cells.keys):
        def active(di: int, dj: int) -> bool:
            return geometry.is_active(i + di, j + dj)

        radial = all(active(0, dj) for dj in (-1, 0, 1))
        axial = all(active(di, 0) for di in (-1, 0, 1))
        patch = all(active(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1))
        if radial:
            for offset, weight in enumerate(weights):
                qb[cell, geometry.z_faces.find(i, j + offset - 1)] = weight
        else:
            qb[cell, geometry.z_faces.find(i, j)] = 0.5
            qb[cell, geometry.z_faces.find(i, j + 1)] = 0.5
        if axial:
            for offset, weight in enumerate(weights):
                qb[2 * nc + cell, nz + nv + geometry.r_faces.find(i + offset - 1, j)] = weight
        else:
            qb[2 * nc + cell, nz + nv + geometry.r_faces.find(i, j)] = 0.5
            qb[2 * nc + cell, nz + nv + geometry.r_faces.find(i + 1, j)] = 0.5
        if patch:
            for ii in range(4):
                for jj in range(4):
                    qb[nc + cell, nz + geometry.vertices.find(i + ii - 1, j + jj - 1)] = (
                        weights[ii] * weights[jj]
                    )
        else:
            for ii in range(2):
                for jj in range(2):
                    qb[nc + cell, nz + geometry.vertices.find(i + ii, j + jj)] = 0.25
    return qb


def build_operators(
    geometry: Geometry,
    K: int,
    options: OperatorOptions | dict[str, Any] | None = None,
    backend: Backend | None = None,
) -> Operators:
    settings = _options(options)
    if K < 0 or not (0.0 < settings.rank_rtol < 1.0):
        raise Failure("Operators", "operator mode or rank tolerance is invalid")
    backend = backend or Backend("cpu")
    _require_memory(
        _estimate_operator_bytes(geometry, K),
        settings.memory_cap_bytes,
        "operator_matrices",
    )
    operators, t1, t2, t3 = _make_fixed_spatial_state(geometry, backend)
    operators.K = K
    if geometry.lcfs_polygon is not None:
        operators.box_magnetic_norm = _box_magnetic_norm(geometry, operators, backend)
    nc = operators.Nc
    nr = len(geometry.r_faces.keys)
    nz = len(geometry.z_faces.keys)
    nb = operators.Nb
    for mode in range(-K, K + 1):
        spatial = _assemble_spatial(
            geometry, mode, operators, t1, t2, t3, True, settings, backend
        )
        curl_adjoint = _adjoint(spatial.C, backend)
        if operators.box_magnetic_norm is None:
            cdag = backend.mm(
                _diagonal_inverse(operators.M1, backend),
                backend.mm(curl_adjoint, _diagonal(operators.MB, backend)),
            )
        else:
            cdag = _scale_rows(
                magnetic_weight(operators, curl_adjoint),
                1.0 / operators.M1,
                backend,
            )
        lphi = backend.asarray(-backend.mm(spatial.DJ, spatial.G))
        spectrum = backend.singular_values(lphi)
        spectrum_host = backend.to_host(spectrum)
        if (
            spectrum_host.size == 0
            or spectrum_host[-1] <= settings.rank_rtol * spectrum_host[0]
        ):
            raise Failure("Operators", "Poisson matrix is numerically rank deficient")
        operators.G.append(spatial.G)
        operators.C.append(spatial.C)
        operators.DB.append(spatial.DB)
        operators.DJ.append(spatial.DJ)
        operators.Cdag.append(cdag)
        operators.Lphi.append(lphi)
        operators.poisson_spectra.append(spectrum)
        operators.phi_factor.append(backend.factor(lphi))

    i_r_host, i_z_host = _host_interpolation(geometry)
    i_r = backend.asarray(i_r_host)
    i_z = backend.asarray(i_z_host)
    operators.I = _block_diagonal([i_r, _identity(nc, backend), i_z], backend)
    node_weights = backend.real_array(np.tile(geometry.M0, 3))
    operators.H = backend.mm(
        _diagonal_inverse(node_weights, backend),
        backend.mm(_adjoint(operators.I, backend), _diagonal(operators.M1, backend)),
    )
    operators.DR = backend.asarray(_select_columns(operators.DJ[K], 0, nr).real)
    operators.DZ = backend.asarray(
        _select_columns(operators.DJ[K], nr + nc, nz).real
    )

    t_full_host = np.zeros((len(geometry.walls), 3 * nc), dtype=np.complex128)
    for wall_id, wall in enumerate(geometry.walls):
        if wall.component == "r_face":
            t_full_host[wall_id, :nc] = wall.sign * i_r_host[wall.face, :]
        else:
            t_full_host[wall_id, 2 * nc :] = wall.sign * i_z_host[wall.face, :]
    operators.T_full = backend.asarray(t_full_host)
    operators.selected_wall_rows = backend.independent_rows(
        operators.T_full, settings.rank_rtol
    )
    operators.T = _select_rows(
        operators.T_full, operators.selected_wall_rows, backend
    )
    operators.QB = backend.asarray(_host_qb(geometry))
    operators.Bc_full = _block_diagonal(operators.DJ, backend)
    operators.selected_constraint_rows = backend.independent_rows(
        operators.Bc_full, settings.rank_rtol
    )
    operators.Bc = _select_rows(
        operators.Bc_full, operators.selected_constraint_rows, backend
    )

    ell2 = backend.asarray(
        np.concatenate((np.zeros(nz), geometry.av, np.zeros(nr)))
    )
    adjoint_t2_factor = backend.factor(_adjoint(t2, backend))
    ell_column = backend.solve(adjoint_t2_factor, backend.xp.conj(ell2))
    operators.topology_flux_row = backend.asarray(backend.xp.conj(ell_column)[None, :])
    if any(
        len(group) != 2 * K + 1
        for group in (operators.G, operators.C, operators.DB, operators.Lphi)
    ):
        raise Failure("Operators", "operator mode count is inconsistent")
    operators.setup_checks = validate_operators(geometry, operators)
    return operators


def validate_operators(
    geometry: Geometry, operators: Operators
) -> list[SimpleNamespace]:
    backend = operators.compute or Backend("cpu")
    checks: list[SimpleNamespace] = []

    def check(
        name: str,
        mode: int,
        residual: Any,
        scale: float,
        terms: int,
        unit: str,
    ) -> None:
        value = backend.max_abs(residual)
        bound = 128.0 * _EPS * max(1, terms) * scale
        if not math.isfinite(scale) or scale < 0.0 or value > bound:
            raise Failure(
                "Operators",
                f"generated operator acceptance failed: {name} mode={mode}",
            )
        checks.append(
            operator_check_record(
                name=name,
                unit=unit,
                mode=mode,
                residual=value,
                scale=scale,
                roundoff_bound=bound,
            )
        )

    for index, mode in enumerate(range(-operators.K, operators.K + 1)):
        check(
            "CG",
            mode,
            backend.mm(operators.C[index], operators.G[index]),
            backend.max_abs(operators.C[index]) * backend.max_abs(operators.G[index]),
            operators.Ne,
            "1/m^2",
        )
        check(
            "DBC",
            mode,
            backend.mm(operators.DB[index], operators.C[index]),
            backend.max_abs(operators.DB[index]) * backend.max_abs(operators.C[index]),
            operators.Nb,
            "1/m^2",
        )
        residual = (
            operators.M0[:, None] * operators.DJ[index]
            + operators.G[index].T.conj() * operators.M1[None, :]
        )
        check(
            "gradient_divergence_adjoint",
            mode,
            residual,
            backend.max_abs(operators.G[index]) * backend.max_abs(operators.M1),
            2,
            "m^2",
        )

    divergence = operators.M0[None, :] @ operators.DJ[operators.K]
    for wall in geometry.walls:
        edge = (
            wall.face
            if wall.component == "r_face"
            else len(geometry.r_faces.keys) + operators.Nc + wall.face
        )
        divergence[0, edge] -= wall.sign * wall.area_per_radian
    surface = max(wall.area_per_radian for wall in geometry.walls)
    check("divergence_theorem", 0, divergence, surface, operators.Nc, "m^2")

    node_weights = backend.xp.tile(operators.M0, 3)
    transfer = (
        node_weights[:, None] * operators.H
        - operators.I.T.conj() * operators.M1[None, :]
    )
    check(
        "transfer_adjoint",
        0,
        transfer,
        backend.max_abs(operators.I) * backend.max_abs(operators.M1),
        2,
        "m^3",
    )
    check(
        "topology_curl",
        0,
        operators.topology_flux_row @ operators.C[operators.K],
        backend.max_abs(operators.topology_flux_row)
        * backend.max_abs(operators.C[operators.K]),
        operators.Nb,
        "m",
    )
    return checks


def _operator_backend(operators: Operators, value: Any) -> Backend:
    backend = operators.compute or Backend("cpu")
    if not isinstance(value, backend.xp.ndarray):
        raise Failure("Backend", "operator input is owned by a different backend")
    return backend


def apply_t1(operators: Operators, values: Any) -> Any:
    backend = _operator_backend(operators, values)
    nc = operators.Nc
    nr = operators.PR_r.shape[0]
    nz = operators.PZ_z.shape[0]
    if values.ndim != 1 or values.size != nr + nc + nz:
        raise Failure("Shape", "T1 input size mismatch")
    result = backend.xp.empty_like(values)
    result[0:nr] = operators.PR_r @ values[0:nr]
    result[nr : nr + nc] = values[nr : nr + nc]
    result[nr + nc :] = operators.PZ_z @ values[nr + nc :]
    return backend.asarray(result)


def apply_t1_inverse(operators: Operators, values: Any) -> Any:
    backend = _operator_backend(operators, values)
    nc = operators.Nc
    nr = operators.PR_r.shape[0]
    nz = operators.PZ_z.shape[0]
    if values.ndim != 1 or values.size != nr + nc + nz:
        raise Failure("Shape", "T1 inverse input size mismatch")
    result = backend.xp.empty_like(values)
    result[0:nr] = backend.solve(operators.PR_r_factor, values[0:nr])
    result[nr : nr + nc] = values[nr : nr + nc]
    result[nr + nc :] = backend.solve(operators.PZ_z_factor, values[nr + nc :])
    return backend.asarray(result)


def apply_t2(operators: Operators, values: Any) -> Any:
    backend = _operator_backend(operators, values)
    nr = operators.PR_r.shape[0]
    nz = operators.PZ_z.shape[0]
    nv = operators.PR_v.shape[0]
    if values.ndim != 1 or values.size != nz + nv + nr:
        raise Failure("Shape", "T2 input size mismatch")
    result = backend.xp.empty_like(values)
    result[0:nz] = operators.PZ_z @ values[0:nz]
    result[nz : nz + nv] = operators.PR_v @ (
        operators.PZ_v @ values[nz : nz + nv]
    )
    radial = values[nz + nv :] * operators.Rbar_r
    radial = operators.PR_r @ radial
    result[nz + nv :] = radial / operators.Rbar_r
    return backend.asarray(result)


def apply_t2_inverse(operators: Operators, values: Any) -> Any:
    backend = _operator_backend(operators, values)
    nr = operators.PR_r.shape[0]
    nz = operators.PZ_z.shape[0]
    nv = operators.PR_v.shape[0]
    if values.ndim != 1 or values.size != nz + nv + nr:
        raise Failure("Shape", "T2 inverse input size mismatch")
    result = backend.xp.empty_like(values)
    result[0:nz] = backend.solve(operators.PZ_z_factor, values[0:nz])
    vertex = backend.solve(operators.PR_v_factor, values[nz : nz + nv])
    result[nz : nz + nv] = backend.solve(operators.PZ_v_factor, vertex)
    radial = values[nz + nv :] * operators.Rbar_r
    radial = backend.solve(operators.PR_r_factor, radial)
    result[nz + nv :] = radial / operators.Rbar_r
    return backend.asarray(result)


def apply_t3(operators: Operators, values: Any) -> Any:
    backend = _operator_backend(operators, values)
    if values.ndim != 1 or values.size != operators.PR_v.shape[0]:
        raise Failure("Shape", "T3 input size mismatch")
    result = values * operators.Rbar_v
    result = operators.PZ_v @ result
    result = operators.PR_v @ result
    return backend.asarray(result / operators.Rbar_v)


def apply_t3_inverse(operators: Operators, values: Any) -> Any:
    backend = _operator_backend(operators, values)
    if values.ndim != 1 or values.size != operators.PR_v.shape[0]:
        raise Failure("Shape", "T3 inverse input size mismatch")
    result = backend.solve(operators.PR_v_factor, values * operators.Rbar_v)
    result = backend.solve(operators.PZ_v_factor, result)
    return backend.asarray(result / operators.Rbar_v)
