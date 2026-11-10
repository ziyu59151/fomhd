"""Contract tests for the native polygon cochain state codec."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from fomhd import io
from fomhd.types import FluidState, State


K = 1
MODES = 2 * K + 1
NC = 2
NF = 3
NV = 4


def _table(keys, R, Z):
    return SimpleNamespace(
        keys=list(keys),
        R=np.asarray(R, dtype=np.float64),
        Z=np.asarray(Z, dtype=np.float64),
    )


def fake_polygon_geometry():
    cells = _table([(0, 0), (1, 0)], [2.8, 3.2], [0.1, -0.2])
    faces = _table([(0,), (1,), (2,)], [2.9, 3.1, 3.0], [0.0, 0.1, 0.2])
    vertices = _table(
        [(0,), (1,), (2,), (3,)],
        [2.7, 3.0, 3.3, 3.0],
        [-0.1, -0.3, 0.1, 0.3],
    )
    face_normals = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    face_tangents = np.array([[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    b1 = sparse.csr_matrix(
        np.array(
            [
                [1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0],
                [1.0, 0.0, -1.0, 0.0],
            ]
        )
    )
    b2 = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 1.0, 1.0],
            ]
        )
    )
    return SimpleNamespace(
        kind="polygon_dual_v1",
        R=np.array([2.7, 3.0, 3.3]),
        Z=np.array([-0.2, 0.0, 0.2]),
        cells=cells,
        faces=faces,
        r_faces=faces,
        z_faces=_table([], [], []),
        vertices=vertices,
        generator_R=cells.R.copy(),
        generator_Z=cells.Z.copy(),
        active_generator=np.ones((2,), dtype=bool),
        cell_polygons=[
            np.array([[2.5, -0.2], [3.0, -0.2], [3.0, 0.2], [2.5, 0.2]]),
            np.array([[3.0, -0.2], [3.5, -0.2], [3.5, 0.2], [3.0, 0.2]]),
        ],
        polygon_input=[
            SimpleNamespace(R=2.5, Z=-0.2),
            SimpleNamespace(R=3.5, Z=-0.2),
            SimpleNamespace(R=3.5, Z=0.2),
            SimpleNamespace(R=2.5, Z=0.2),
        ],
        polygon=[
            SimpleNamespace(R=2.5, Z=-0.2),
            SimpleNamespace(R=3.5, Z=-0.2),
            SimpleNamespace(R=3.5, Z=0.2),
            SimpleNamespace(R=2.5, Z=0.2),
        ],
        face_vertices=np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
        face_cells=np.array([[0, 1], [0, -1], [1, -1]], dtype=np.int64),
        face_normals=face_normals,
        face_tangents=face_tangents,
        face_lengths=np.array([0.4, 0.5, 0.4]),
        face_distances=np.array([0.25, 0.25, 0.25]),
        b1=b1,
        b2=b2,
        area_c=np.array([0.2, 0.2]),
        M0=np.array([2 * np.pi * 2.75 * 0.2, 2 * np.pi * 3.25 * 0.2]),
        M1=np.concatenate(
            (
                np.array([2 * np.pi * 2.9 * 0.4 * 0.25] * 3),
                np.array([2 * np.pi * 2.75 * 0.2, 2 * np.pi * 3.25 * 0.2]),
            )
        ),
        MB=np.concatenate(
            (
                np.array([0.2, 0.2, 0.2, 0.2]),
                np.array([2 * np.pi * 2.9 * 0.4 * 0.25] * 3),
            )
        ),
        av=np.array([0.08, 0.08, 0.08, 0.08]),
        wv=np.array([0.2, 0.2, 0.2, 0.2]),
        Rbar_v=np.array([2.7, 3.0, 3.3, 3.0]),
        Rbar_r=faces.R.copy(),
        walls=[],
    )


def _hermitian(values):
    values = np.asarray(values, dtype=np.complex128)
    result = values.copy()
    result[K] = result[K].real
    for mode in range(1, K + 1):
        result[K + mode] = np.conj(result[K - mode])
    return result


def fake_state():
    density_i = _hermitian([[2.0, 3.0], [0.2 + 0.1j, -0.1j], [0.0, 0.0]])
    density_e = 1.01 * density_i
    pressure_i = _hermitian([[600.0, 610.0], [2.0j, 0.1 - 1.0j], [0.0, 0.0]])
    pressure_e = _hermitian([[400.0, 405.0], [-1.0j, 0.2 + 0.5j], [0.0, 0.0]])
    mom_i = _hermitian(
        np.arange(MODES * 3 * NC, dtype=np.float64).reshape(MODES, 3, NC)
        * (1.0 + 0.1j)
    )
    mom_e = _hermitian(
        (0.5j * np.arange(MODES * 3 * NC).reshape(MODES, 3, NC))
    )
    b_phi = _hermitian(
        [[2.0, 2.1, 2.2, 2.3], [0.1j, 0.2j, -0.1j, -0.2j], [0.0] * 4]
    )
    b_t = _hermitian(
        [[0.1, -0.2, 0.3], [0.01 + 0.02j, 0.03j, -0.04j], [0.0] * 3]
    )
    return State(
        fluid=(
            FluidState(
                n=density_i.reshape(-1),
                mom=mom_i.reshape(-1),
                perp=pressure_i.reshape(-1),
                parallel=np.zeros(0, dtype=np.complex128),
            ),
            FluidState(
                n=density_e.reshape(-1),
                mom=mom_e.reshape(-1),
                perp=pressure_e.reshape(-1),
                parallel=pressure_e.reshape(-1),
            ),
        ),
        B=np.concatenate((b_phi, b_t), axis=1).reshape(-1),
        step=4,
        generation=7,
        time=1.25,
    )


def fake_polygon_operators(geometry):
    nc, nf, nv = NC, NF, NV
    ne = nf + nc
    nb = nv + nf
    return SimpleNamespace(
        geometry=geometry,
        geometry_kind="polygon_dual_v1",
        matrix_layout_version="polygon_dual_v1",
        reconstruction_version=1,
        matrix_fingerprint="matrix-only",
        G=[sparse.csr_matrix(np.eye(ne, nc, dtype=np.complex128))],
        C=[np.ones((nb, ne), dtype=np.complex128)],
        DB=[np.ones((nv, nb), dtype=np.complex128)],
        DJ=[np.ones((nc, ne), dtype=np.complex128)],
        Cdag=[np.ones((ne, nb), dtype=np.complex128)],
        I=np.ones((ne, 3 * nc), dtype=np.complex128),
        H=np.ones((3 * nc, ne), dtype=np.complex128),
        S=np.ones((nf, nc), dtype=np.float64),
        scalar_I=np.ones((ne, nc), dtype=np.float64),
        QB=np.ones((3 * nc, nb), dtype=np.complex128),
        T=np.ones((1, 3 * nc), dtype=np.complex128),
        T_full=np.ones((1, 3 * nc), dtype=np.complex128),
        M0=np.ones(nc, dtype=np.float64),
        M1=np.ones(nf + nc, dtype=np.float64),
        MB=np.ones(nv + nf, dtype=np.float64),
        topology_flux_row=np.ones((1, nv + nf), dtype=np.complex128),
        selected_wall_rows=[0],
        reconstruction=SimpleNamespace(
            degree=1,
            max_support_cells=8,
            min_support_cells=4,
            max_condition=10.0,
            min_condition=1.0,
        ),
        qb_diagnostics=SimpleNamespace(
            max_support_faces=8,
            max_support_vertices=4,
            max_condition=10.0,
        ),
    )


def test_native_state_round_trip_uses_face_tangential_cochains():
    geometry = fake_polygon_geometry()
    state = fake_state()

    arrays = io.encode_state(state, geometry)

    assert arrays["n_i"].shape == (MODES, NC)
    assert arrays["mom_i"].shape == (MODES, 3, NC)
    assert arrays["B_phi_native"].shape == (MODES, NV)
    assert arrays["B_t_native"].shape == (MODES, NF)
    assert "B_Z" not in arrays
    assert "B_R" not in arrays
    assert str(arrays["layout_schema"]) == "polygon_dual_v1:v1"
    assert arrays["geometry_fingerprint"].shape == ()
    np.testing.assert_array_equal(arrays["cell_entity_order"], np.arange(NC))
    np.testing.assert_array_equal(arrays["face_entity_order"], np.arange(NF))
    np.testing.assert_array_equal(arrays["vertex_entity_order"], np.arange(NV))

    restored = io.decode_state(arrays, geometry, K)

    np.testing.assert_array_equal(restored.fluid[0].n, state.fluid[0].n)
    np.testing.assert_array_equal(restored.fluid[1].mom, state.fluid[1].mom)
    np.testing.assert_array_equal(restored.B, state.B)
    assert restored.step == 0
    assert restored.generation == 1
    assert restored.time == 0.0


def test_native_vertex_cochain_round_trip_has_no_legacy_scatter():
    geometry = fake_polygon_geometry()
    vector = _hermitian(
        [[0.1, 0.2, 0.3, 0.4], [0.01j, -0.02j, 0.03j, -0.04j], [0.0] * 4]
    ).reshape(-1)

    encoded = io.encode_vertex(vector, geometry, K)
    decoded = io.decode_vertex(encoded, geometry, K)

    assert encoded.shape == (MODES, NV)
    np.testing.assert_array_equal(decoded, vector)


def test_changed_normal_changes_fingerprint_and_invalidates_decode():
    geometry = fake_polygon_geometry()
    state = fake_state()
    arrays = io.encode_state(state, geometry)
    before = str(arrays["geometry_fingerprint"])
    geometry.face_normals[0] = geometry.face_normals[0][::-1]

    assert io._polygon_geometry_fingerprint(geometry) != before
    with pytest.raises(ValueError, match="geometry fingerprint"):
        io.decode_state(arrays, geometry, K)


def test_changed_metric_changes_fingerprint():
    from fomhd import polygon_io

    geometry = fake_polygon_geometry()
    before = polygon_io.geometry_fingerprint(geometry)
    geometry.M0[0] *= 1.000001

    assert polygon_io.geometry_fingerprint(geometry) != before


@pytest.mark.parametrize("field", ["M1", "MB", "b1"])
def test_changed_hodge_or_incidence_identity_is_rejected(field):
    from fomhd import polygon_io

    geometry = fake_polygon_geometry()
    before = polygon_io.geometry_fingerprint(geometry)
    if field == "b1":
        geometry.b1 = geometry.b1.copy()
        geometry.b1.data[0] *= -1.0
    else:
        getattr(geometry, field)[0] *= 1.000001

    assert polygon_io.geometry_fingerprint(geometry) != before


def test_nonfinite_state_is_rejected():
    geometry = fake_polygon_geometry()
    state = fake_state()
    state.fluid[0].n = state.fluid[0].n.copy()
    state.fluid[0].n[NC] = np.nan

    with pytest.raises(ValueError, match="finite"):
        io.encode_state(state, geometry)


def test_wrong_state_dtype_is_rejected():
    geometry = fake_polygon_geometry()
    state = fake_state()
    state.fluid[0].n = state.fluid[0].n.real.copy()

    with pytest.raises(ValueError, match="complex128"):
        io.encode_state(state, geometry)


def test_wrong_native_shape_is_rejected_on_encode_and_decode():
    geometry = fake_polygon_geometry()
    state = fake_state()
    state.B = state.B[:-1].copy()
    with pytest.raises(ValueError, match="B state extent"):
        io.encode_state(state, geometry)

    arrays = io.encode_state(fake_state(), geometry)
    arrays["B_t_native"] = arrays["B_t_native"][:, :-1].copy()
    with pytest.raises(ValueError, match="shape"):
        io.decode_state(arrays, geometry, K)


def test_reversed_or_nonhermitian_mode_order_is_rejected():
    geometry = fake_polygon_geometry()
    arrays = io.encode_state(fake_state(), geometry)
    arrays["n_i"][K - 1] = arrays["n_i"][K - 1] + 0.25j

    with pytest.raises(ValueError, match="mode order|Hermitian"):
        io.decode_state(arrays, geometry, K)


def test_cross_layout_and_old_schema_are_rejected():
    geometry = fake_polygon_geometry()
    state = fake_state()
    geometry.kind = "staircase_v1"
    with pytest.raises(ValueError, match="polygon_dual_v1"):
        io.encode_state(state, geometry)

    geometry = fake_polygon_geometry()
    arrays = io.encode_state(state, geometry)
    arrays.pop("layout_schema")
    with pytest.raises(ValueError, match="layout_schema"):
        io.decode_state(arrays, geometry, K)


def test_operator_fingerprint_includes_native_geometry_identity():
    geometry = fake_polygon_geometry()
    operators = fake_polygon_operators(geometry)
    before = io._operator_fingerprint(operators, None)
    geometry.face_normals[1] *= -1.0

    after = io._operator_fingerprint(operators, None)

    assert before != after
    assert before != "matrix-only"


def test_operator_fingerprint_changes_with_actual_operator_values():
    geometry = fake_polygon_geometry()
    operators = fake_polygon_operators(geometry)
    before = io._operator_fingerprint(operators, None)
    operators.I[0, 0] += 1.0

    assert io._operator_fingerprint(operators, None) != before


def test_operator_fingerprint_includes_reconstruction_declaration():
    geometry = fake_polygon_geometry()
    operators = fake_polygon_operators(geometry)
    before = io._operator_fingerprint(operators, None)
    operators.reconstruction.degree = 2

    assert io._operator_fingerprint(operators, None) != before


def test_operator_fingerprint_uses_explicit_backend_export():
    geometry = fake_polygon_geometry()
    operators = fake_polygon_operators(geometry)

    class ExportOnly:
        def __init__(self, value):
            self.value = value

        def __array__(self, *args, **kwargs):
            raise AssertionError("implicit operator host conversion")

    operators.I = ExportOnly(operators.I)
    backend = SimpleNamespace(
        to_host=lambda value: value.value
        if isinstance(value, ExportOnly)
        else value
    )
    before = io._operator_fingerprint(operators, backend)
    operators.I.value[0, 0] += 1.0

    assert io._operator_fingerprint(operators, backend) != before
