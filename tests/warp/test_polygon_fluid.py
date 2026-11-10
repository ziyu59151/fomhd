"""Focused polygon-dual fluid actions on a synthetic non-axis mesh."""

import warnings
from types import SimpleNamespace

import numpy as np

from fomhd.backend import Backend
from fomhd.fluid import (
    _component_flux,
    _sampled_divergence,
    gradient_adjoint,
    velocity_gradient,
)
from fomhd.geometry import build_rectangle
from fomhd.operators import build_operators


NORMAL = 1.0 / np.sqrt(2.0)
CELL_R = np.array([4.0 / 3.0, 5.0 / 3.0], dtype=np.float64)
CELL_Z = np.array([1.0 / 3.0, 2.0 / 3.0], dtype=np.float64)
CELL_AREA = np.array([0.5, 0.5], dtype=np.float64)
CELL_M0 = CELL_AREA * CELL_R
SCALAR_TO_FACE = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.5, 0.5],
    ],
    dtype=np.float64,
)
NORMAL_FACES = 5
CELLS = 2
EDGES = NORMAL_FACES + CELLS
FACE_NORMALS = np.array(
    [
        [0.0, -1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [NORMAL, NORMAL],
    ],
    dtype=np.float64,
)
# Outward unit-normal finite-volume incidence. The shared slanted face has
# opposite owner signs for the two triangles.
DIV_FACE = np.array(
    [
        [-2.0, 0.0, 0.0, -2.0, 2.0 * NORMAL],
        [0.0, 2.0, 2.0, 0.0, -2.0 * NORMAL],
    ],
    dtype=np.float64,
)
SCALAR_I = np.vstack(
    (SCALAR_TO_FACE, np.eye(CELLS, dtype=np.float64))
)


def _context(N=5, *, geometry_kind=True, coordinate_backends=True):
    backend = Backend("cpu")
    op = SimpleNamespace(
        Nc=CELLS,
        Ne=EDGES,
        Nf=NORMAL_FACES,
        Dface=DIV_FACE.copy(),
        scalar_I=SCALAR_I.copy(),
        M0=CELL_M0.copy(),
    )
    if coordinate_backends:
        op.face_normals_backend = backend.real_array(FACE_NORMALS)
        op.cell_radius_backend = backend.real_array(CELL_R)
    geometry = SimpleNamespace(
        cells=SimpleNamespace(
            R=(
                np.full(CELLS, -999.0, dtype=np.float64)
                if coordinate_backends
                else CELL_R.copy()
            ),
            Z=CELL_Z.copy(),
        ),
        face_normals=(
            np.full((NORMAL_FACES, 2), -999.0, dtype=np.float64)
            if coordinate_backends
            else FACE_NORMALS.copy()
        ),
    )
    if geometry_kind:
        geometry.kind = "polygon_dual_v1"
        op.geometry_kind = "staircase_v1"
    else:
        op.geometry_kind = "polygon_dual_v1"
    return SimpleNamespace(
        N=N,
        p=SimpleNamespace(K=1),
        compute=backend,
        op=op,
        geometry=geometry,
    )


def _face_scalar(cell_scalar):
    samples = np.asarray(cell_scalar, dtype=np.complex128)
    if samples.ndim == 1:
        samples = samples[None, :]
        squeeze = True
    else:
        squeeze = False
    result = np.concatenate(
        (
            samples @ SCALAR_TO_FACE.T,
            samples,
        ),
        axis=1,
    )
    return result[0] if squeeze else result


def _base_fluid(context, density):
    density = np.asarray(density, dtype=np.complex128)
    return SimpleNamespace(
        n=density.reshape(-1),
        nf=_face_scalar(density).reshape(-1),
    )


def test_polygon_component_flux_uses_scalar_transfer_once():
    context = _context(N=3)
    scalar = np.array(
        [
            [1.0 + 0.5j, -2.0 + 0.25j],
            [0.3 - 1.2j, 4.0 - 0.75j],
            [-0.7 + 0.1j, 2.5 + 1.5j],
        ],
        dtype=np.complex128,
    ).reshape(-1)

    actual = _component_flux(scalar, context)
    expected = (
        scalar.reshape(context.N, CELLS) @ SCALAR_I.T
    ).reshape(-1)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_polygon_divergence_adds_normal_and_toroidal_terms():
    context = _context(N=5)
    index = np.arange(context.N, dtype=np.float64)[:, None]
    face_index = np.arange(NORMAL_FACES, dtype=np.float64)[None, :]
    normal_flux = (
        (1.0 - 0.2j) * index
        + (0.4 + 0.7j) * face_index
        - 0.3j
    )
    phi = 2.0 * np.pi * np.arange(context.N, dtype=np.float64) / context.N
    left_amplitude = 0.7 - 0.4j
    right_amplitude = -0.2 + 0.6j
    left_mode = left_amplitude * np.exp(1j * phi)
    right_mode = right_amplitude * np.exp(-2j * phi)
    toroidal_flux = np.stack(
        (left_mode + 0.25, right_mode - 0.15j), axis=1
    )
    edges = np.concatenate((normal_flux, toroidal_flux), axis=1).reshape(-1)
    derivative = np.stack(
        (
            1j * left_amplitude * np.exp(1j * phi),
            -2j * right_amplitude * np.exp(-2j * phi),
        ),
        axis=1,
    )
    expected = normal_flux @ DIV_FACE.T + (
        derivative / CELL_R[None, :]
    )

    actual = _sampled_divergence(edges, context).reshape(context.N, CELLS)

    np.testing.assert_allclose(actual, expected, rtol=2.0e-15, atol=2.0e-15)


def _expected_cellwise_gradient(values, density):
    density = np.asarray(density, dtype=np.complex128)
    if density.ndim > 1:
        density = density[0]
    face_density = SCALAR_TO_FACE @ density
    result = np.zeros((3, 3, CELLS), dtype=np.complex128)
    for component in range(3):
        weighted = density * values[component]
        normal_flux = (
            FACE_NORMALS[:, 0] * (SCALAR_TO_FACE @ weighted) / face_density
        )
        axial_flux = (
            FACE_NORMALS[:, 1] * (SCALAR_TO_FACE @ weighted) / face_density
        )
        result[component, 0, :] = DIV_FACE @ normal_flux - (
            values[component] / CELL_R
        )
        result[component, 2, :] = DIV_FACE @ axial_flux
    result[0, 1, :] = -values[1] / CELL_R
    result[1, 1, :] = values[0] / CELL_R
    return result


def test_polygon_cylindrical_constant_gradient_with_variable_density():
    context = _context(N=3)
    density = np.tile(
        np.array([2.0, 4.0], dtype=np.complex128), (context.N, 1)
    )
    fluid = _base_fluid(context, density)
    values = np.array([1.2 + 0.1j, 0.8 - 0.2j, -0.5 + 0.3j])
    velocity = np.broadcast_to(
        values[:, None], (3, context.N, CELLS)
    ).reshape(-1)

    actual = velocity_gradient(fluid, velocity, context).reshape(
        context.N, 3, 3, CELLS
    )
    expected = _expected_cellwise_gradient(values, density)

    np.testing.assert_allclose(
        actual,
        np.broadcast_to(expected, (context.N, 3, 3, CELLS)),
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_polygon_cylindrical_linear_gradient_with_variable_density():
    context = _context(N=1)
    density = np.array([2.0, 4.0], dtype=np.complex128)
    fluid = _base_fluid(context, density)
    values = np.array(
        [
            1.0 + 0.2 * CELL_R - 0.3 * CELL_Z,
            0.5 * CELL_R,
            -0.7 + 0.1 * CELL_R + 0.4 * CELL_Z,
        ],
        dtype=np.complex128,
    )
    velocity = values[:, None, :].reshape(-1)

    actual = velocity_gradient(fluid, velocity, context).reshape(
        3, 3, CELLS
    )
    expected = _expected_cellwise_gradient(values, density)

    np.testing.assert_allclose(actual, expected, rtol=2.0e-15, atol=2.0e-15)


def test_polygon_gradient_does_not_cast_real_contract_weights_to_float64():
    context = _context(N=1, coordinate_backends=False)
    context.op.scalar_I = np.asarray(SCALAR_I, dtype=np.complex128)
    density = np.array([2.0, 4.0], dtype=np.complex128)
    fluid = _base_fluid(context, density)
    values = np.array([1.2, -0.4, 0.7], dtype=np.complex128)
    velocity = np.broadcast_to(
        values[:, None, None], (3, context.N, CELLS)
    ).reshape(-1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = velocity_gradient(fluid, velocity, context).reshape(
            3, 3, CELLS
        )

    expected = _expected_cellwise_gradient(values, density)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-15, atol=2.0e-15)


def test_polygon_coordinate_actions_prefer_backend_mesh_arrays():
    context = _context(N=1)
    assert np.all(context.geometry.cells.R < 0.0)
    assert np.all(context.geometry.face_normals < 0.0)
    density = np.array([2.0, 4.0], dtype=np.complex128)
    fluid = _base_fluid(context, density)
    values = np.array([1.2, -0.4, 0.7], dtype=np.complex128)
    velocity = np.broadcast_to(
        values[:, None, None], (3, context.N, CELLS)
    ).reshape(-1)

    actual = velocity_gradient(fluid, velocity, context).reshape(3, 3, CELLS)
    expected = _expected_cellwise_gradient(values, density)

    np.testing.assert_allclose(actual, expected, rtol=2.0e-15, atol=2.0e-15)


def test_polygon_coordinate_fallback_is_limited_to_synthetic_mesh():
    context = _context(N=1, coordinate_backends=False)
    assert not hasattr(context.op, "face_normals_backend")
    assert not hasattr(context.op, "cell_radius_backend")
    density = np.array([2.0, 4.0], dtype=np.complex128)
    fluid = _base_fluid(context, density)
    values = np.array([1.2, -0.4, 0.7], dtype=np.complex128)
    velocity = np.broadcast_to(
        values[:, None, None], (3, context.N, CELLS)
    ).reshape(-1)

    actual = velocity_gradient(fluid, velocity, context).reshape(3, 3, CELLS)
    expected = _expected_cellwise_gradient(values, density)

    np.testing.assert_allclose(actual, expected, rtol=2.0e-15, atol=2.0e-15)


def test_polygon_gradient_adjoint_is_exact_for_complex_multimode_fields():
    context = _context(N=7)
    phi = 2.0 * np.pi * np.arange(context.N, dtype=np.float64) / context.N
    density = (
        2.0
        + 0.2 * np.cos(phi)[:, None]
        + 0.1 * np.sin(2.0 * phi)[:, None] * np.arange(CELLS)[None, :]
    ).astype(np.complex128)
    fluid = _base_fluid(context, density)
    velocity = np.zeros((context.N, 3, CELLS), dtype=np.complex128)
    tensor = np.zeros((context.N, 3, 3, CELLS), dtype=np.complex128)
    rng = np.random.default_rng(734)
    for mode in (-1, 0, 1):
        for component in range(3):
            for cell in range(CELLS):
                velocity[:, component, cell] += (
                    rng.standard_normal() + 1j * rng.standard_normal()
                ) * np.exp(1j * mode * phi)
        for row in range(3):
            for column in range(3):
                for cell in range(CELLS):
                    tensor[:, row, column, cell] += (
                        rng.standard_normal()
                        + 1j * rng.standard_normal()
                    ) * np.exp(1j * mode * phi)

    gradient = velocity_gradient(fluid, velocity.reshape(-1), context)
    adjoint = gradient_adjoint(fluid, tensor.reshape(-1), context)
    m0 = CELL_M0[None, :]
    left = np.sum(m0[:, None, None, :] * np.conj(gradient.reshape(
        context.N, 3, 3, CELLS
    )) * tensor)
    right = np.sum(m0[:, None, :] * np.conj(velocity) * adjoint.reshape(
        context.N, 3, CELLS
    ))

    np.testing.assert_allclose(right, left, rtol=2.0e-14, atol=2.0e-14)


def test_polygon_dispatch_can_fall_back_to_operator_geometry_kind():
    context = _context(N=2, geometry_kind=False)
    scalar = np.array(
        [[1.0 + 2.0j, -0.5 + 0.25j], [0.8 - 0.1j, 0.3 + 0.4j]]
    ).reshape(-1)
    actual = _component_flux(scalar, context)
    expected = (
        scalar.reshape(context.N, CELLS) @ SCALAR_I.T
    ).reshape(-1)
    np.testing.assert_array_equal(actual, expected)


def test_staircase_component_flux_keeps_repeated_vector_interpolation():
    context = _context(N=3)
    backend = context.compute
    geometry = build_rectangle(3, 3, 1.0, 0.0, 0.5, 0.5)
    operators = build_operators(geometry, 0, backend=backend)
    ncell = len(geometry.cells.keys)
    old = SimpleNamespace(
        N=context.N,
        compute=backend,
        op=operators,
        geometry=SimpleNamespace(kind="staircase_v1"),
    )
    scalar = np.array(
        [
            [1.0 + 0.2j, -0.5 + 0.7j, 0.2 - 0.4j, 0.9 + 0.1j],
            [0.3 - 0.1j, 0.8 + 0.4j, -0.7 + 0.2j, 0.1 - 0.6j],
            [-1.0 + 0.6j, 1.2 - 0.9j, 0.4 + 0.3j, -0.2 + 0.5j],
        ],
        dtype=np.complex128,
    ).reshape(-1)
    assert scalar.size == context.N * ncell
    nodes = np.repeat(scalar.reshape(context.N, 1, ncell), 3, axis=1)
    expected = backend.mm(
        operators.I,
        nodes.reshape(context.N, 3 * ncell).T,
    ).T.reshape(-1)

    actual = _component_flux(scalar, old)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
