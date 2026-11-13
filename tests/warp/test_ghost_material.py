"""Independent line tests for the rectangular material closure."""

import numpy as np
import pytest


@pytest.mark.parametrize("count", [2, 3, 4, 9])
def test_material_line_derivative_reproduces_affine_fields(count):
    from fomhd.ghost_material import material_line_derivative

    spacing = 0.13
    x = spacing * (np.arange(count) + 0.5)
    derivative = material_line_derivative(count, spacing)
    assert derivative.shape == (count, count)
    assert derivative.dtype == np.float64
    np.testing.assert_allclose(derivative @ np.ones(count), 0, atol=3e-15)
    np.testing.assert_allclose(derivative @ x, 1, rtol=0, atol=3e-15)


@pytest.mark.parametrize("count", [2, 3, 8])
def test_material_line_derivative_pairs_with_physical_wall_trace(count):
    from fomhd.ghost_material import material_line_derivative

    spacing = 0.2
    derivative = material_line_derivative(count, spacing)
    lower, upper = np.zeros((2, count))
    lower[:2], upper[-2:] = (1.5, -0.5), (-0.5, 1.5)
    rng = np.random.default_rng(903)
    p, u = rng.normal(size=(2, count)) + 1j * rng.normal(size=(2, count))
    surface_work = (upper @ p).conjugate() * (upper @ u)
    surface_work -= (lower @ p).conjugate() * (lower @ u)
    volume_work = spacing * (np.vdot(p, derivative @ u) + np.vdot(derivative @ p, u))
    np.testing.assert_allclose(volume_work, surface_work, rtol=0, atol=7e-15)

    pressure = spacing * (np.arange(count) + 0.5)
    traction = (upper @ pressure) * upper - (lower @ pressure) * lower
    force = derivative.T @ pressure - traction / spacing
    np.testing.assert_allclose(force, -1, rtol=0, atol=3e-15)


@pytest.mark.parametrize("count", [2, 3, 8])
def test_material_face_interpolation_consumes_affine_ghost_layers(count):
    from fomhd.ghost_material import material_face_interpolation

    centers = np.arange(count) + 0.5
    values = centers**2
    extended = np.r_[
        values[0] - 2 * (values[1] - values[0]),
        values[0] - (values[1] - values[0]),
        values,
        values[-1] + (values[-1] - values[-2]),
        values[-1] + 2 * (values[-1] - values[-2]),
    ]
    expected = np.convolve(extended, np.array([-1, 9, 9, -1]) / 16, mode="valid")
    interpolation = material_face_interpolation(count)
    assert interpolation.shape == (count + 1, count)
    assert interpolation.dtype == np.float64
    np.testing.assert_array_equal(interpolation @ values, expected)
    np.testing.assert_array_equal(interpolation @ np.ones(count), np.ones(count + 1))
    np.testing.assert_array_equal(interpolation @ centers, np.arange(count + 1))


def test_material_line_builders_reject_invalid_extents():
    from fomhd.ghost_material import material_face_interpolation, material_line_derivative

    with pytest.raises(ValueError, match="at least two cells"):
        material_face_interpolation(1)
    with pytest.raises(ValueError, match="at least two cells"):
        material_line_derivative(1, 1.0)


@pytest.mark.parametrize("spacing", [0.0, -0.5, np.inf, np.nan])
def test_material_line_derivative_rejects_invalid_spacing(spacing):
    from fomhd.ghost_material import material_line_derivative

    with pytest.raises(ValueError, match="finite positive spacing"):
        material_line_derivative(4, spacing)


@pytest.fixture
def material_box():
    from types import SimpleNamespace

    from fomhd.box_boundary import install_normal_flux
    from fomhd.geometry import build_aligned_box
    from fomhd.operators import build_operators
    from fomhd.types import Context

    radial = 1.1 + 0.16 * np.arange(6)
    axial = -0.3 + 0.15 * np.arange(5)
    g = build_aligned_box(radial, axial, [(1.4, -0.1), (1.6, -0.1), (1.5, 0.1)])
    op = build_operators(g, 1, {"ghost_wall": True, "include_finite_mass": False})
    caps = np.zeros((3, len(g.walls), 2), dtype=np.complex128)
    for index, wall in enumerate(g.walls):
        if wall.component == "z_face":
            caps[op.K, index] = wall.sign * wall.area_per_radian * 0.2
    install_normal_flux(op, g, caps)
    return Context(SimpleNamespace(K=op.K), g, op, op.compute, 5)


def test_box_material_gradient_has_zero_rigid_rotation_strain(material_box):
    from types import SimpleNamespace

    from fomhd.fluid import velocity_gradient

    c = material_box
    f = SimpleNamespace(n=np.ones(c.N * c.op.Nc), nf=np.ones(c.N * c.op.Ne))
    velocity = np.zeros((c.N, 3, c.op.Nc), dtype=np.complex128)
    velocity[:, 1] = 2.4 * c.geometry.cells.R
    gradient = velocity_gradient(f, velocity.ravel(), c).reshape(c.N, 3, 3, c.op.Nc)
    np.testing.assert_allclose(gradient[:, 1, 0], 2.4, rtol=0, atol=1e-13)
    np.testing.assert_allclose(gradient + gradient.swapaxes(1, 2), 0, atol=1e-13)


@pytest.mark.parametrize("variable_density,axial_slope", [(False, 0.8), (True, 0.0)])
def test_box_material_pressure_pairs_with_threaded_wall_traction(
    material_box, variable_density, axial_slope
):
    from types import SimpleNamespace

    from fomhd.box_boundary import electron_pressure_traction
    from fomhd.fluid import _component_flux, gradient_adjoint

    c = material_box
    pressure = np.tile(2.0 + axial_slope * c.geometry.cells.Z, c.N)
    density = np.ones(c.N * c.op.Nc)
    if variable_density:
        density = np.tile(1.7 + 0.3 * c.geometry.cells.R * c.geometry.cells.Z, c.N)
    fluid = SimpleNamespace(
        n=density, nf=_component_flux(density, c), perp=pressure, parallel=pressure
    )
    B = np.zeros((2 * c.op.K + 1, c.op.Nb), dtype=np.complex128)
    B[c.op.K] = np.r_[
        np.zeros(len(c.geometry.z_faces.keys)),
        1.7 / c.geometry.Rbar_v,
        np.full(len(c.geometry.r_faces.keys), 0.4),
    ]
    tensor = np.eye(3)[None, :, :, None] * pressure.reshape(c.N, 1, 1, c.op.Nc)
    force = gradient_adjoint(fluid, tensor.ravel(), c)
    force += electron_pressure_traction(B.ravel(), fluid, c)
    expected = np.zeros((c.N, 3, c.op.Nc))
    expected[:, 2] = -axial_slope
    np.testing.assert_allclose(force.reshape(expected.shape), expected, rtol=0, atol=1e-13)


def test_box_material_gradient_and_adjoint_pair_with_variable_density(material_box):
    from types import SimpleNamespace

    from fomhd.fluid import _component_flux, gradient_adjoint, velocity_gradient

    c = material_box
    rng = np.random.default_rng(96)
    density = 1.0 + 0.2 * rng.random(c.N * c.op.Nc)
    fluid = SimpleNamespace(n=density, nf=_component_flux(density, c))
    velocity_shape = (c.N, 3, c.op.Nc)
    tensor_shape = (c.N, 3, 3, c.op.Nc)
    u = rng.normal(size=velocity_shape) + 1j * rng.normal(size=velocity_shape)
    tensor = rng.normal(size=tensor_shape) + 1j * rng.normal(size=tensor_shape)
    gradient = velocity_gradient(fluid, u.ravel(), c).reshape(tensor.shape)
    adjoint = gradient_adjoint(fluid, tensor.ravel(), c).reshape(u.shape)
    np.testing.assert_allclose(
        np.vdot(gradient, c.op.M0 * tensor),
        np.vdot(u, c.op.M0 * adjoint),
        rtol=0,
        atol=2e-13,
    )
