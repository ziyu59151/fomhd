"""Analytic CPU tests for the bounded true-intercept triangle patch."""

from __future__ import annotations

import math

import numpy as np
import pytest
from tools.clt_boundary_patch_reference import (
    TriangleGeometry,
    affine_pressure,
    homogeneous_metric,
    lifted_metric,
    magnetic_divergence,
    normal_lumped_metric,
    p1_gradient,
    paired_power,
    pressure_gradient_force,
    pressure_gradient_identity,
    pressure_wall_reaction,
    reconstruct_field,
    sample_electric,
    sample_potential,
    sample_pressure,
)


def _constant(field_r: float, field_z: float):
    def evaluate(r, z):
        shape = np.broadcast_shapes(np.shape(r), np.shape(z))
        value = np.empty((*shape, 2), dtype=np.float64)
        value[..., 0] = field_r
        value[..., 1] = field_z
        return value

    return evaluate


def _oracle_volume_integral(geometry, integrand, order=12):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    s, t = np.meshgrid(nodes, nodes, indexing="ij")
    ws, wt = np.meshgrid(weights, weights, indexing="ij")
    x = geometry.a * s
    z = geometry.b * (1.0 - s) * t
    measure = geometry.a * geometry.b * (1.0 - s) * (geometry.r0 + x)
    return float(np.sum(measure * ws * wt * integrand(x, z)))


def _oracle_gram(geometry, count: int) -> np.ndarray:
    def matrix_entry(p, q):
        def evaluate(x, z):
            # Independent basis formulas, not the production evaluator.
            values = np.stack(
                (
                    np.stack((1.0 - z / geometry.b, x / geometry.b), axis=-1),
                    np.stack((z / geometry.a, 1.0 - x / geometry.a), axis=-1),
                    np.stack(
                        (
                            -z / (geometry.a * geometry.b),
                            x / (geometry.a * geometry.b),
                        ),
                        axis=-1,
                    ),
                ),
                axis=-2,
            )
            return np.sum(values[..., p, :] * values[..., q, :], axis=-1)

        return _oracle_volume_integral(geometry, evaluate)

    return np.array([[matrix_entry(p, q) for q in range(count)] for p in range(count)])


def test_constant_normal_electric_trace_has_zero_wall_moment():
    geometry = TriangleGeometry(3.0, 0.2, 0.7)
    length = math.hypot(geometry.a, geometry.b)
    normal = np.array([geometry.b, geometry.a]) / length
    trace = sample_electric(geometry, _constant(*normal), order=6)

    assert trace.e_r == pytest.approx(normal[0], abs=2e-15)
    assert trace.e_z == pytest.approx(normal[1], abs=2e-15)
    assert trace.wall_line_integral == pytest.approx(0.0, abs=2e-15)
    assert trace.circulation == pytest.approx(0.0, abs=2e-15)
    assert trace.curl_phi == pytest.approx(0.0, abs=2e-14)


def test_constant_tangent_electric_trace_retains_line_moment():
    geometry = TriangleGeometry(3.0, 0.2, 0.7)
    length = math.hypot(geometry.a, geometry.b)
    tangent = np.array([-geometry.a, geometry.b]) / length
    trace = sample_electric(geometry, _constant(*tangent), order=6)

    assert trace.wall_line_integral == pytest.approx(length, rel=2e-15)
    assert trace.wall_line_integral == pytest.approx(
        -geometry.a * trace.e_r + geometry.b * trace.e_z,
        rel=2e-15,
    )
    points_r = np.array([3.0, 3.05, 3.2, 3.0])
    points_z = np.array([0.0, 0.1, 0.0, 0.7])
    reconstructed = reconstruct_field(geometry, trace.coefficients, points_r, points_z)
    np.testing.assert_allclose(
        reconstructed, np.broadcast_to(tangent, reconstructed.shape), atol=2e-15
    )


def test_nonzero_curl_is_checked_by_boundary_circulation():
    geometry = TriangleGeometry(3.0, 0.2, 0.7)
    trace = sample_electric(
        geometry, lambda r, z: np.stack((z, np.zeros_like(z)), axis=-1)
    )

    assert trace.circulation == pytest.approx(-geometry.a * geometry.b / 2.0, rel=2e-15)
    assert trace.curl_phi == pytest.approx(1.0, rel=2e-15)


def test_homogeneous_wall_potential_has_exact_physical_gradient():
    geometry = TriangleGeometry(3.0, 0.2, 0.7)

    def potential(r, z):
        return geometry.signed_normal_distance(r, z)

    values = sample_potential(geometry, potential)
    expected = np.array([-values.o / geometry.a, -values.o / geometry.b])
    assert values.a == pytest.approx(0.0, abs=2e-15)
    assert values.b == pytest.approx(0.0, abs=2e-15)
    np.testing.assert_allclose(p1_gradient(geometry, values), expected, atol=2e-15)

    gradient = _constant(*expected)
    trace = sample_electric(geometry, gradient)
    np.testing.assert_allclose(trace.axis_coefficients, expected, atol=2e-15)
    assert trace.wall_line_integral == pytest.approx(0.0, abs=2e-15)


def test_curved_wall_potential_gradient_error_refines_without_exactness_claim():
    beta = 0.4

    def reference(geometry):
        def potential(r, z):
            distance = geometry.signed_normal_distance(r, z)
            tangent = geometry.wall_tangent_coordinate(r, z)
            return distance * (1.0 + beta * tangent)

        return potential

    def gradient(geometry):
        normal = np.asarray(geometry.wall_normal)
        tangent = np.asarray(geometry.wall_tangent)

        def field(r, z):
            distance = geometry.signed_normal_distance(r, z)
            coordinate = geometry.wall_tangent_coordinate(r, z)
            return normal * (1.0 + beta * coordinate)[..., None] + beta * distance[
                ..., None
            ] * np.broadcast_to(tangent, (*np.shape(distance), 2))

        return field

    errors = []
    for h in (0.4, 0.2):
        geometry = TriangleGeometry(3.0, 0.2 * h, 0.7 * h)
        trace = sample_electric(geometry, gradient(geometry), order=10)
        points_r = np.array([3.0 + 0.13 * h, 3.0 + 0.4 * h])
        points_z = np.array([0.11 * h, 0.27 * h])
        reconstructed = reconstruct_field(
            geometry, trace.coefficients, points_r, points_z
        )
        exact = gradient(geometry)(points_r, points_z)
        errors.append(float(np.max(np.abs(reconstructed - exact))))

    assert errors[0] > 1.0e-7
    assert errors[1] < 0.60 * errors[0]


def test_homogeneous_metric_matches_independent_quadrature_and_is_spd():
    geometry = TriangleGeometry(3.0, 0.2, 0.7)
    metric = homogeneous_metric(geometry)
    oracle = _oracle_gram(geometry, 2)

    np.testing.assert_allclose(metric.matrix, oracle, rtol=2e-14, atol=1e-15)
    assert metric.is_spd
    assert metric.condition_number < 100.0

    vector = np.array([0.3, -0.8])
    right = np.array([1.2, 0.5])
    np.testing.assert_allclose(metric.apply(vector), metric.matrix @ vector, atol=2e-15)
    np.testing.assert_allclose(metric.solve(metric.apply(vector)), vector, atol=2e-14)
    assert metric.pair(vector, right) == pytest.approx(
        vector @ metric.adjoint(right), rel=2e-15
    )


def test_lifted_metric_matches_physical_quadrature_and_spd_actions():
    geometry = TriangleGeometry(3.0, 0.2, 0.7)
    metric = lifted_metric(geometry, order=8)
    oracle = _oracle_gram(geometry, 3)

    np.testing.assert_allclose(metric.matrix, oracle, rtol=2e-14, atol=1e-15)
    np.testing.assert_allclose(metric.matrix, metric.matrix.T, atol=1e-15)
    assert metric.includes_wall
    assert metric.is_spd
    vector = np.array([0.2, -0.4, 0.9])
    np.testing.assert_allclose(metric.solve(metric.apply(vector)), vector, atol=3e-14)
    assert metric.pair(vector, vector) > 0.0


def test_normal_lumping_uses_selected_trace_vector():
    geometry = TriangleGeometry(3.0, 0.2, 0.7)
    metric = homogeneous_metric(geometry)
    trace = np.array([geometry.b, geometry.a])
    expected = np.diag((metric.matrix @ trace) / trace)

    np.testing.assert_allclose(normal_lumped_metric(geometry), expected, rtol=2e-15)


def test_paired_power_requires_the_tangent_current_wall_moment():
    geometry = TriangleGeometry(3.0, 0.2, 0.7)
    electric = _constant(0.0, 1.0)
    current = _constant(0.0, 2.0)
    expected = 2.0 * geometry.volume

    complete = paired_power(geometry, electric, current, order=8)
    omitted = paired_power(
        geometry, electric, current, order=8, include_wall_moment=False
    )
    current_trace = sample_electric(geometry, current, order=8)
    dropped_field = reconstruct_field(
        geometry,
        np.array([current_trace.e_r, current_trace.e_z, 0.0]),
        np.array([3.0 + geometry.a, 3.0]),
        np.array([0.0, geometry.b]),
    )

    assert complete.power == pytest.approx(expected, rel=3e-14)
    assert complete.field_power == pytest.approx(expected, rel=3e-14)
    assert complete.reconstruction_power_error < 3e-14 * expected
    assert abs(omitted.power - expected) > 0.25 * expected
    assert np.max(np.abs(dropped_field - np.array([0.0, 2.0]))) > 0.5


def test_magnetic_divergence_preserves_nonzero_constant_bz_wall_flux():
    geometry = TriangleGeometry(3.0, 0.2, 0.7)
    field = magnetic_divergence(geometry, _constant(0.0, 1.0), order=8)
    expected_wall = geometry.a * (geometry.r0 + geometry.a / 2.0)

    assert field.wall_flux == pytest.approx(expected_wall, rel=2e-15)
    assert field.wall_flux > 0.0
    assert field.axis_oa_flux == pytest.approx(-expected_wall, rel=2e-15)
    assert field.axis_ob_flux == pytest.approx(0.0, abs=2e-15)
    assert field.average_divergence == pytest.approx(0.0, abs=2e-15)


def test_constant_br_divergence_matches_cylindrical_volume_average():
    geometry = TriangleGeometry(3.0, 0.2, 0.7)
    value = 1.7
    field = magnetic_divergence(geometry, _constant(value, 0.0), order=8)
    expected = 3.0 * value / (3.0 * geometry.r0 + geometry.a)

    assert field.average_divergence == pytest.approx(expected, rel=3e-15)
    assert field.wall_flux > 0.0


def test_fluid_pressure_force_is_negative_gradient_and_identity_stays_positive():
    geometry = TriangleGeometry(3.0, 0.2, 0.7)
    p0, pr, pz = 2.3, -0.8, 1.4

    def pressure(r, z):
        return p0 + pr * (r - geometry.r0) + pz * z

    sampled = sample_pressure(geometry, pressure)
    affine = affine_pressure(geometry, sampled)
    force = pressure_gradient_force(geometry, affine, order=8)
    expected_force = -geometry.volume * np.array((pr, pz))
    np.testing.assert_allclose(force, expected_force, rtol=3e-14, atol=2e-14)

    velocity = _constant(0.35, -0.2)
    def divergence(r, z):
        return 0.35 / r

    balance = pressure_gradient_identity(
        geometry, affine, velocity, divergence, order=8
    )
    assert balance.gradient_pairing == pytest.approx(
        float(affine.gradient @ np.array((0.35, -0.2))) * geometry.volume,
        rel=3e-14,
    )
    assert balance.defect == pytest.approx(0.0, abs=3e-14)
    np.testing.assert_allclose(
        balance.wall_reaction,
        pressure_wall_reaction(geometry, affine, order=8),
        rtol=3e-14,
        atol=2e-14,
    )


def test_constant_pressure_has_zero_fluid_force():
    geometry = TriangleGeometry(3.0, 0.2, 0.7)
    constant = affine_pressure(
        geometry,
        sample_pressure(geometry, _constant_pressure(4.2)),
    )
    np.testing.assert_array_equal(
        pressure_gradient_force(geometry, constant), np.zeros(2)
    )


def test_reciprocal_r_tangent_current_refines_and_retains_wall_moment():
    """Use E0 in V/m and J0 in A/m so E and J have V/m and A/m^2."""
    r0 = 3.0
    aspect = 0.4
    current_scale_am = 1.7
    electric_scale_vm = 2.3
    levels = (0.5, 0.25, 0.125)
    relative_errors = []
    retained_power_errors = []
    omitted_power_errors = []

    for h in levels:
        geometry = TriangleGeometry(r0, aspect * h, h)
        normal = np.asarray(geometry.wall_normal)

        def current(r, z, geometry=geometry):
            return np.stack(
                (
                    current_scale_am * geometry.a / r,
                    -current_scale_am * geometry.b / r,
                ),
                axis=-1,
            )

        electric_trace = sample_electric(
            geometry, _constant(*(electric_scale_vm * normal)), order=10
        )
        current_trace = sample_electric(geometry, current, order=10)
        assert electric_trace.wall_line_integral == pytest.approx(0.0, abs=2e-15)
        assert abs(current_trace.wall_line_integral) > 0.0

        wall_field = current(r0, geometry.b)
        assert wall_field @ normal == pytest.approx(0.0, abs=3e-15)
        divergence = magnetic_divergence(geometry, current, order=10)
        assert divergence.average_divergence == pytest.approx(0.0, abs=2e-14)

        def reconstructed_power(coefficients, geometry=geometry, normal=normal):
            def integrand(
                x, z, geometry=geometry, coefficients=coefficients, normal=normal
            ):
                field = reconstruct_field(geometry, coefficients, r0 + np.asarray(x), z)
                electric = electric_scale_vm * normal
                return np.sum(field * electric, axis=-1)

            return _oracle_volume_integral(geometry, integrand)

        def relative_field_error(coefficients, geometry=geometry):
            def error_integrand(x, z, geometry=geometry, coefficients=coefficients):
                physical = current(r0 + np.asarray(x), z)
                field = reconstruct_field(geometry, coefficients, r0 + np.asarray(x), z)
                return np.sum((field - physical) ** 2, axis=-1)

            def physical_integrand(x, z, geometry=geometry):
                physical = current(r0 + np.asarray(x), z)
                return np.sum(physical**2, axis=-1)

            numerator = _oracle_volume_integral(geometry, error_integrand)
            denominator = _oracle_volume_integral(geometry, physical_integrand)
            return math.sqrt(numerator / denominator)

        retained = np.array(current_trace.coefficients)
        omitted = np.array((current_trace.e_r, current_trace.e_z, 0.0))
        relative_errors.append(relative_field_error(retained))
        retained_power_errors.append(abs(reconstructed_power(retained)))
        omitted_power_errors.append(abs(reconstructed_power(omitted)))

    observed_orders = np.log2(
        np.asarray(relative_errors[:-1]) / np.asarray(relative_errors[1:])
    )
    assert np.all(observed_orders > 0.75)
    assert retained_power_errors[-1] < omitted_power_errors[-1]
    assert omitted_power_errors[-1] > 10.0 * retained_power_errors[-1]


def _constant_pressure(value: float):
    def pressure(r, z):
        return np.full(np.broadcast_shapes(np.shape(r), np.shape(z)), value)

    return pressure


def test_invalid_geometry_and_order_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        TriangleGeometry(3.0, 0.0, 0.7)
    with pytest.raises(ValueError, match="order"):
        lifted_metric(TriangleGeometry(3.0, 0.2, 0.7), order=1)
