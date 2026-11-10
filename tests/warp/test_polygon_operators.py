from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fomhd.backend import Backend
from fomhd.fluid import gradient_adjoint, velocity_gradient
from fomhd.polygon_geometry import build_polygon_geometry
from fomhd.polygon_operators import build_polygon_operators
from fomhd.types import Failure


class _LimitedFlags:
    __slots__ = ("_c_contiguous", "_f_contiguous", "_owndata")

    def __init__(self) -> None:
        object.__setattr__(self, "_c_contiguous", True)
        object.__setattr__(self, "_f_contiguous", True)
        object.__setattr__(self, "_owndata", True)

    @property
    def c_contiguous(self):
        return self._c_contiguous

    @property
    def f_contiguous(self):
        return self._f_contiguous

    @property
    def owndata(self):
        return self._owndata


class _LimitedFlagArray(np.ndarray):
    @property
    def flags(self):
        return _LimitedFlags()


class _LimitedFlagBackend(Backend):
    def __init__(self, geometry) -> None:
        super().__init__("cpu")
        self.name = "cuda"
        self.geometry = geometry

    def real_array(self, value):
        result = super().real_array(value)
        if value is self.geometry.face_normals or value is self.geometry.cells.R:
            return result.view(_LimitedFlagArray)
        return result


def rectangle_geometry():
    R = np.linspace(0.55, 1.45, 10)
    Z = np.linspace(-0.45, 0.45, 10)
    polygon = [(0.5, -0.5), (1.5, -0.5), (1.5, 0.5), (0.5, 0.5)]
    return build_polygon_geometry(R, Z, polygon)


def regular_rectangle_geometry(nr: int, nz: int):
    hr = 0.13
    hz = 0.09
    r0 = 0.55
    z0 = -0.35
    R = r0 + hr * np.arange(nr, dtype=np.float64)
    Z = z0 + hz * np.arange(nz, dtype=np.float64)
    polygon = [
        (R[0] - 0.5 * hr, Z[0] - 0.5 * hz),
        (R[-1] + 0.5 * hr, Z[0] - 0.5 * hz),
        (R[-1] + 0.5 * hr, Z[-1] + 0.5 * hz),
        (R[0] - 0.5 * hr, Z[-1] + 0.5 * hz),
    ]
    return build_polygon_geometry(R, Z, polygon)


def patch_context(geometry, operators):
    return SimpleNamespace(
        N=1,
        p=SimpleNamespace(K=0),
        compute=operators.compute,
        op=operators,
        geometry=geometry,
    )


def patch_fluid(operators):
    density = np.ones(operators.Nc, dtype=np.complex128)
    return SimpleNamespace(
        n=density,
        nf=(operators.scalar_I @ density).astype(np.complex128),
        u=np.zeros(3 * operators.Nc, dtype=np.complex128),
    )


def interior_cells(geometry):
    result = []
    for cell, (i, j) in enumerate(geometry.cells.keys):
        begin = geometry.b2.indptr[cell]
        end = geometry.b2.indptr[cell + 1]
        faces = geometry.b2.indices[begin:end]
        if (
            2 <= i <= geometry.R.size - 3
            and 2 <= j <= geometry.Z.size - 3
            and faces.size
            and np.all(geometry.face_cells[faces, 1] >= 0)
        ):
            result.append(cell)
    return np.asarray(result, dtype=np.int64)


def weighted_norm(weight, values) -> float:
    return float(np.sqrt(np.sum(weight * np.abs(values) ** 2)))


def test_layout_shapes_and_explicit_finite_mass_unavailability() -> None:
    geometry = rectangle_geometry()
    operators = build_polygon_operators(geometry, 1)
    nf = len(geometry.faces.keys)
    nc = len(geometry.cells.keys)
    nv = len(geometry.vertices.keys)

    assert operators.kind == "polygon_dual_v1"
    assert operators.geometry_kind == "polygon_dual_v1"
    assert operators.matrix_layout_version == "polygon_dual_v1"
    assert (operators.Nf, operators.Nc, operators.Nv) == (nf, nc, nv)
    assert (operators.Ne, operators.Nb) == (nf + nc, nv + nf)
    assert len(operators.G) == len(operators.C) == len(operators.DB) == 3
    for index in range(3):
        assert operators.G[index].shape == (nf + nc, nc)
        assert operators.C[index].shape == (nv + nf, nf + nc)
        assert operators.DB[index].shape == (nv, nv + nf)
        assert operators.DJ[index].shape == (nc, nf + nc)
        assert operators.Cdag[index].shape == (nf + nc, nv + nf)

    assert operators.S.shape == (nf, nc)
    assert operators.scalar_I.shape == (nf + nc, nc)
    assert operators.I.shape == (nf + nc, 3 * nc)
    assert operators.H.shape == (3 * nc, nf + nc)
    assert operators.QB.shape == (3 * nc, nv + nf)
    assert operators.Dface.shape == (nc, nf)
    assert operators.DR.shape == (nc, nf)
    assert operators.DZ.shape == (nc, 0)
    assert operators.topology_flux_row.shape == (1, nv + nf)
    assert np.array_equal(operators.curvature_radius, geometry.cells.R)
    assert operators.face_normals_backend.shape == (nf, 2)
    assert operators.face_normals_backend.dtype == np.float64
    assert np.array_equal(operators.face_normals_backend, geometry.face_normals)
    assert operators.cell_radius_backend.shape == (nc,)
    assert operators.cell_radius_backend.dtype == np.float64
    assert np.array_equal(operators.cell_radius_backend, geometry.cells.R)
    with pytest.raises(ValueError):
        operators.face_normals_backend[0, 0] = 2.0
    with pytest.raises(ValueError):
        operators.cell_radius_backend[0] = 2.0
    assert operators.finite_mass_available is False
    assert operators.finite_mass_unavailable_reason
    assert operators.Lphi == operators.phi_factor == operators.poisson_spectra == []
    assert operators.Bc.shape[0] == 0
    assert operators.Bc_full.shape[0] == 0
    assert operators.selected_constraint_rows == []


def test_cuda_limited_flags_do_not_require_writeable_metric_flag() -> None:
    geometry = rectangle_geometry()
    backend = _LimitedFlagBackend(geometry)
    operators = build_polygon_operators(geometry, 0, backend=backend)

    for metric in (operators.face_normals_backend, operators.cell_radius_backend):
        assert isinstance(metric, _LimitedFlagArray)
        assert metric.flags.c_contiguous is True
        assert metric.flags.f_contiguous is True
        assert metric.flags.owndata is True
        assert not hasattr(metric.flags, "writeable")


@pytest.mark.parametrize("grid", [(8, 7), (11, 9)])
def test_regular_facets_use_symmetric_rows_and_report_actual_kind(grid) -> None:
    geometry = regular_rectangle_geometry(*grid)
    operators = build_polygon_operators(geometry, 0)
    owners = geometry.face_cells
    interior_faces = np.flatnonzero(owners[:, 1] >= 0)
    assert interior_faces.size > 0
    for face in interior_faces:
        first, second = owners[face]
        assert operators.S[face, first] == pytest.approx(0.5, abs=0.0)
        assert operators.S[face, second] == pytest.approx(0.5, abs=0.0)
        assert operators.S[face].sum() == pytest.approx(1.0, abs=0.0)
        report = operators.reconstruction.reports[face]
        assert report["kind"] == "regular_two_owner"
        assert report["support"] == 2
        assert report["rank"] == 2
    assert operators.reconstruction.regular_faces == interior_faces.size
    assert operators.reconstruction.polynomial_faces == len(geometry.walls)


@pytest.mark.parametrize("grid", [(8, 7), (11, 9)])
def test_regular_pressure_patches_through_actual_gradient_adjoint(grid) -> None:
    geometry = regular_rectangle_geometry(*grid)
    operators = build_polygon_operators(geometry, 0)
    context = patch_context(geometry, operators)
    fluid = patch_fluid(operators)
    selected = interior_cells(geometry)
    assert selected.size > 0

    for pressure, expected in (
        (geometry.cells.R, np.array([-1.0, 0.0, 0.0])),
        (geometry.cells.Z + 2.0, np.array([0.0, 0.0, -1.0])),
    ):
        tensor = np.zeros((1, 3, 3, operators.Nc), dtype=np.complex128)
        for component in range(3):
            tensor[:, component, component, :] = pressure
        force = gradient_adjoint(fluid, tensor.reshape(-1), context).reshape(
            3, operators.Nc
        )
        np.testing.assert_allclose(
            force[:, selected],
            np.tile(expected[:, None], (1, selected.size)),
            rtol=0.0,
            atol=2.0e-13,
        )


@pytest.mark.parametrize("grid", [(8, 7), (11, 9)])
def test_constant_normal_electric_patches_through_same_H(grid) -> None:
    geometry = regular_rectangle_geometry(*grid)
    operators = build_polygon_operators(geometry, 0)
    selected = interior_cells(geometry)
    for axis, expected in (
        (0, np.array([1.0, 0.0, 0.0])),
        (1, np.array([0.0, 0.0, 1.0])),
    ):
        edge = np.concatenate(
            (
                geometry.face_normals[:, axis],
                np.zeros(operators.Nc, dtype=np.float64),
            )
        ).astype(np.complex128)
        reconstructed = (operators.H @ edge).reshape(3, operators.Nc)
        np.testing.assert_allclose(
            reconstructed[:, selected],
            np.tile(expected[:, None], (1, selected.size)),
            rtol=0.0,
            atol=2.0e-13,
        )


@pytest.mark.parametrize("grid", [(8, 7), (11, 9)])
def test_regular_quadratic_radial_velocity_gradient_patch(grid) -> None:
    geometry = regular_rectangle_geometry(*grid)
    operators = build_polygon_operators(geometry, 0)
    context = patch_context(geometry, operators)
    fluid = patch_fluid(operators)
    selected = interior_cells(geometry)
    velocity = np.zeros((1, 3, operators.Nc), dtype=np.complex128)
    velocity[:, 0, :] = geometry.cells.Z**2

    gradient = velocity_gradient(
        fluid, velocity.reshape(-1), context
    ).reshape(1, 3, 3, operators.Nc)
    np.testing.assert_allclose(
        gradient[:, 0, 0, selected],
        0.0,
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        gradient[:, 0, 2, selected],
        (2.0 * geometry.cells.Z[selected])[None, :],
        rtol=0.0,
        atol=2.0e-13,
    )


def test_deliberately_clipped_face_fails_regular_predicate() -> None:
    from fomhd.polygon_operators import _is_regular_two_owner_face

    R = np.linspace(0.55, 1.45, 10)
    Z = np.linspace(-0.45, 0.45, 10)
    polygon = [(0.61, -0.5), (1.5, -0.5), (1.5, 0.5), (0.61, 0.5)]
    geometry = build_polygon_geometry(
        R,
        Z,
        polygon,
        options={"guard_factor": 0.3},
    )
    full_area = (R[1] - R[0]) * (Z[1] - Z[0])
    clipped = np.flatnonzero(geometry.area_c < 0.99 * full_area)
    assert clipped.size > 0
    candidates = [
        face
        for face, owners in enumerate(geometry.face_cells)
        if owners[1] >= 0
        and (
            owners[0] in clipped
            or owners[1] in clipped
        )
        and abs(geometry.face_normals[face, 0]) > 0.99
    ]
    assert candidates
    assert not any(
        _is_regular_two_owner_face(geometry, face) for face in candidates
    )


def test_degree2_scalar_reconstruction_keeps_polynomial_candidate() -> None:
    geometry = regular_rectangle_geometry(8, 7)
    operators = build_polygon_operators(
        geometry,
        0,
        options={"scalar_degree": 2},
    )
    assert operators.reconstruction.regular_faces == 0
    assert operators.reconstruction.polynomial_faces == operators.Nf
    assert operators.reconstruction.kinds == ("polynomial",)
    assert all(
        report["kind"] == "polynomial"
        for report in operators.reconstruction.reports
    )
    assert operators.reconstruction.max_support_cells > 2


@pytest.mark.parametrize("K", [0, 1])
def test_exact_low_order_chain_identities_on_nonuniform_ell_and_d(K: int) -> None:
    geometry = rectangle_geometry()
    operators = build_polygon_operators(geometry, K, backend=Backend("cpu"))
    ratios = geometry.face_lengths / geometry.face_distances
    assert np.ptp(ratios) > 0.5
    assert not np.allclose(
        geometry.face_lengths,
        geometry.face_distances,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert len(np.unique(np.round(geometry.face_distances, 14))) > 1

    for index in range(2 * K + 1):
        cg = operators.C[index] @ operators.G[index]
        dbc = operators.DB[index] @ operators.C[index]
        cg_scale = (
            np.max(np.abs(operators.C[index]))
            * np.max(np.abs(operators.G[index]))
        )
        dbc_scale = (
            np.max(np.abs(operators.DB[index]))
            * np.max(np.abs(operators.C[index]))
        )
        assert np.max(np.abs(cg), initial=0.0) <= 256 * np.finfo(float).eps * cg_scale
        assert np.max(np.abs(dbc), initial=0.0) <= 256 * np.finfo(float).eps * dbc_scale


def test_weighted_adjoints_and_incidence_aliases() -> None:
    geometry = rectangle_geometry()
    operators = build_polygon_operators(geometry, 0)
    nf = operators.Nf
    nc = operators.Nc
    nv = operators.Nv

    assert np.max(np.abs(operators.b2r @ operators.b1r)) == 0.0
    assert operators.b1z.shape == (0, nv)
    assert operators.b2z.shape == (nc, 0)
    for name in ("PR_r", "PZ_r", "PR_z", "PZ_z", "PR_v", "PZ_v"):
        matrix = getattr(operators, name)
        assert matrix.shape == (0, 0)
        assert getattr(operators, f"{name}_factor") is None

    physical = np.arange(3 * nc, dtype=np.float64) / 17.0 + 0.25j * np.arange(3 * nc)
    edges = (np.arange(nf + nc) % 11 - 5.0) / 7.0 + 0.1j
    lhs = np.sum(
        np.conj(physical)
        * np.tile(operators.M0, 3)
        * (operators.H @ edges)
    )
    rhs = np.sum(np.conj(operators.I @ physical) * operators.M1 * edges)
    assert abs(lhs - rhs) <= 2.0e-13 * max(1.0, abs(lhs), abs(rhs))

    boundary_faces = np.flatnonzero(geometry.face_cells[:, 1] < 0)
    assert operators.T_full.shape == (len(geometry.walls), 3 * nc)
    assert np.array_equal(operators.T_full, operators.I[boundary_faces, :])
    assert operators.T.shape[0] == len(operators.selected_wall_rows)
    assert np.array_equal(
        operators.T,
        operators.T_full[operators.selected_wall_rows],
    )


def test_I_maps_physical_vector_to_edge_cochain() -> None:
    geometry = rectangle_geometry()
    operators = build_polygon_operators(geometry, 0)
    nodes = np.arange(3 * operators.Nc, dtype=np.float64) / 9.0 + 0.3j
    radial = nodes[: operators.Nc]
    toroidal = nodes[operators.Nc : 2 * operators.Nc]
    axial = nodes[2 * operators.Nc :]
    mapped = operators.I @ nodes
    expected_normal = geometry.face_normals[:, 0] * (operators.S @ radial)
    expected_normal += geometry.face_normals[:, 1] * (operators.S @ axial)

    assert operators.I.shape == (operators.Ne, 3 * operators.Nc)
    assert np.max(np.abs(mapped[: operators.Nf] - expected_normal)) < 8.0e-13
    assert np.array_equal(mapped[operators.Nf :], toroidal)
    boundary_faces = np.flatnonzero(geometry.face_cells[:, 1] < 0)
    assert np.array_equal(operators.T_full, operators.I[boundary_faces, :])


def test_scalar_transfer_is_exact_for_linear_cell_data() -> None:
    geometry = rectangle_geometry()
    operators = build_polygon_operators(geometry, 0)
    cells = np.column_stack((geometry.cells.R, geometry.cells.Z))
    faces = np.column_stack((geometry.faces.R, geometry.faces.Z))
    values = 1.75 + 2.25 * cells[:, 0] - 0.6 * cells[:, 1]
    expected = 1.75 + 2.25 * faces[:, 0] - 0.6 * faces[:, 1]

    assert np.max(np.abs(operators.S @ np.ones(len(values)) - 1.0)) < 5.0e-13
    assert np.max(np.abs(operators.S @ values - expected)) < 5.0e-13
    assert np.max(np.abs(operators.scalar_I @ values - np.concatenate((expected, values)))) < 5.0e-13
    assert operators.reconstruction.degree == 1
    assert operators.reconstruction.max_support_cells <= 24
    assert operators.reconstruction.max_condition <= 1.0e8


def test_qb_recovers_linear_physical_vector_from_frames() -> None:
    geometry = rectangle_geometry()
    operators = build_polygon_operators(geometry, 0)
    faces = np.column_stack((geometry.faces.R, geometry.faces.Z))
    vertices = np.column_stack((geometry.vertices.R, geometry.vertices.Z))
    cells = np.column_stack((geometry.cells.R, geometry.cells.Z))

    def br(point):
        return 1.0 + 0.2 * point[..., 0] - 0.3 * point[..., 1]

    def bz(point):
        return -0.4 + 0.7 * point[..., 0] + 0.1 * point[..., 1]

    def bphi(point):
        return 2.4 - 0.7 * point[..., 0] + 0.3 * point[..., 1]

    bt = br(faces) * geometry.face_tangents[:, 0] + bz(
        faces
    ) * geometry.face_tangents[:, 1]
    states = np.concatenate((bphi(vertices), bt))
    reconstructed = operators.QB @ states
    expected = np.concatenate((br(cells), bphi(cells), bz(cells)))

    assert np.max(np.abs(reconstructed - expected)) < 8.0e-13
    assert operators.qb_diagnostics.max_support_faces <= 32
    assert operators.qb_diagnostics.max_condition <= 1.0e8


def test_topology_flux_row_is_closed_against_curl() -> None:
    geometry = rectangle_geometry()
    operators = build_polygon_operators(geometry, 0)
    nf = operators.Nf
    assert np.array_equal(
        operators.topology_flux_row[0],
        np.concatenate((geometry.av, np.zeros(nf))),
    )
    residual = operators.topology_flux_row @ operators.C[operators.K]
    scale = (
        np.max(np.abs(operators.topology_flux_row))
        * np.max(np.abs(operators.C[operators.K]))
    )
    assert np.max(np.abs(residual)) <= 256 * np.finfo(float).eps * scale


def test_toroidal_harmonic_is_current_free() -> None:
    geometry = rectangle_geometry()
    operators = build_polygon_operators(geometry, 0)
    state = np.concatenate((1.3 / geometry.Rbar_v, np.zeros(operators.Nf)))
    residual = operators.Cdag[operators.K] @ state
    scale = np.max(np.abs(operators.Cdag[operators.K])) * np.max(np.abs(state))
    assert np.max(np.abs(residual)) <= 256 * np.finfo(float).eps * scale
    assert np.max(np.abs(operators.DB[operators.K] @ state)) <= 256 * np.finfo(float).eps


def test_memory_rank_and_input_failures_are_explicit() -> None:
    geometry = rectangle_geometry()
    with pytest.raises(Failure, match="memory budget exceeded: requested=.*limit=1"):
        build_polygon_operators(geometry, 1, options={"memory_cap_bytes": 1})
    with pytest.raises(Failure, match="condition"):
        build_polygon_operators(geometry, 0, options={"max_condition": 1.0 + 1.0e-15})
    with pytest.raises(Failure, match="mode|K"):
        build_polygon_operators(geometry, -1)

    malformed = rectangle_geometry()
    malformed.kind = "staircase_v1"
    with pytest.raises(Failure, match="polygon_dual_v1"):
        build_polygon_operators(malformed, 0)
