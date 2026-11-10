from __future__ import annotations

import copy
import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from fomhd.boundary_geometry import build_boundary_geometry
from fomhd.geometry import build_geometry


def _module():
    try:
        from fomhd import boundary_quadrature
    except ModuleNotFoundError as error:
        if error.name != "fomhd.boundary_quadrature":
            raise
        pytest.fail("fomhd.boundary_quadrature has not been implemented")
    return boundary_quadrature


def _rectangle_geometry():
    return build_geometry(
        1.0 + 0.1 * np.arange(5),
        -0.2 + 0.1 * np.arange(5),
        [(0.95, -0.25), (1.45, -0.25), (1.45, 0.25), (0.95, 0.25)],
    )


def _fraction_geometry(fraction: float):
    wall_r = 1.2 + fraction * 0.1
    return build_geometry(
        1.0 + 0.1 * np.arange(6),
        -0.2 + 0.1 * np.arange(5),
        [
            (0.96, -0.25),
            (wall_r - 0.02, -0.25),
            (wall_r + 0.02, 0.25),
            (0.96, 0.25),
        ],
    )


def _circle_geometry(
    nodes: int, *, shift_R: float = 0.0, shift_Z: float = 0.0
):
    R = np.linspace(0.9 + shift_R, 2.5 + shift_R, nodes)
    Z = np.linspace(-0.8 + shift_Z, 0.8 + shift_Z, nodes)
    angle = 2.0 * math.pi * np.arange(64) / 64.0
    polygon = np.column_stack(
        (1.7 + 0.68 * np.cos(angle), 0.68 * np.sin(angle))
    )
    return build_geometry(R, Z, polygon)


def _rectangle_moments():
    lower_R, upper_R = 0.95, 1.45
    lower_Z, upper_Z = -0.25, 0.25
    R0, Z0 = 1.2, 0.0
    L = 0.5
    area = (upper_R - lower_R) * (upper_Z - lower_Z)
    m0 = 0.5 * (upper_R**2 - lower_R**2) * (upper_Z - lower_Z)
    R2 = (upper_R**3 - lower_R**3) * (upper_Z - lower_Z) / 3.0
    RZ = (
        0.5 * (upper_R**2 - lower_R**2)
        * 0.5 * (upper_Z**2 - lower_Z**2)
    )
    return np.array(
        [m0, (R2 - R0 * m0) / L, (RZ - Z0 * m0) / L],
        dtype=np.float64,
    ), R0, Z0, L, area


def _triangle_moments(vertices):
    coordinates = np.asarray(vertices, dtype=np.float64)
    first = coordinates[1] - coordinates[0]
    second = coordinates[2] - coordinates[0]
    area = abs(first[0] * second[1] - first[1] * second[0]) / 2.0
    R0 = 0.5 * (np.min(coordinates[:, 0]) + np.max(coordinates[:, 0]))
    Z0 = 0.5 * (np.min(coordinates[:, 1]) + np.max(coordinates[:, 1]))
    L = max(np.ptp(coordinates[:, 0]), np.ptp(coordinates[:, 1]))
    radii = coordinates[:, 0]
    heights = coordinates[:, 1]
    pairs = ((0, 1), (1, 2), (0, 2))
    m0 = area * np.sum(radii) / 3.0
    R2 = area * (
        np.sum(radii**2)
        + sum(radii[i] * radii[j] for i, j in pairs)
    ) / 6.0
    RZ = area * np.sum(radii * heights) / 6.0
    RZ += area * sum(
        radii[i] * heights[j] + radii[j] * heights[i]
        for i, j in pairs
    ) / 12.0
    return np.array(
        [m0, (R2 - R0 * m0) / L, (RZ - Z0 * m0) / L],
        dtype=np.float64,
    ), R0, Z0, L, area


def _point_segment_distance(R, Z, start, end):
    delta_R = end[0] - start[0]
    delta_Z = end[1] - start[1]
    denominator = delta_R * delta_R + delta_Z * delta_Z
    parameter = (
        (R - start[0]) * delta_R + (Z - start[1]) * delta_Z
    ) / denominator
    parameter = min(1.0, max(0.0, parameter))
    return math.hypot(
        R - (start[0] + parameter * delta_R),
        Z - (start[1] + parameter * delta_Z),
    )


def _boundary_distance(R, Z, polygon):
    return min(
        _point_segment_distance(
            R, Z, polygon[index], polygon[(index + 1) % len(polygon)]
        )
        for index in range(len(polygon))
    )


def _snapshot(geometry, boundary_geometry):
    return (
        geometry.R.tobytes(),
        geometry.Z.tobytes(),
        geometry.active.tobytes(),
        geometry.M0.tobytes(),
        tuple(geometry.cells.keys),
        boundary_geometry.fingerprint,
        boundary_geometry.polygon_R_m.tobytes(),
        boundary_geometry.polygon_Z_m.tobytes(),
        tuple(boundary_geometry.cell_keys),
    )


def test_polygon_three_moments_match_rectangle_analytic_oracle() -> None:
    module = _module()
    polygon = np.array(
        [(0.95, -0.25), (1.45, -0.25), (1.45, 0.25), (0.95, 0.25)]
    )
    moments, R0, Z0, L = module._polygon_three_moments(polygon)
    expected, expected_R0, expected_Z0, expected_L, _ = _rectangle_moments()
    np.testing.assert_allclose(
        moments, expected, rtol=0.0, atol=2.0e-15
    )
    assert (R0, Z0, L) == pytest.approx(
        (expected_R0, expected_Z0, expected_L), abs=0.0
    )


def test_polygon_three_moments_match_triangle_analytic_oracle() -> None:
    module = _module()
    polygon = np.array([(0.7, -0.3), (1.9, -0.1), (0.9, 0.8)])
    moments, R0, Z0, L = module._polygon_three_moments(polygon)
    expected, expected_R0, expected_Z0, expected_L, _ = _triangle_moments(
        polygon
    )
    np.testing.assert_allclose(
        moments, expected, rtol=0.0, atol=4.0e-15
    )
    assert (R0, Z0, L) == pytest.approx(
        (expected_R0, expected_Z0, expected_L), abs=0.0
    )


def test_rectangle_keeps_core_weights_exact_and_changes_band_only() -> None:
    module = _module()
    geometry = _rectangle_geometry()
    boundary = build_boundary_geometry(geometry)
    result = module.build_boundary_volume_weights(geometry, boundary)
    old = np.asarray(geometry.M0)
    expected = _rectangle_moments()[0]
    assert result.old_moments_m3_per_rad[0] == pytest.approx(
        expected[0], rel=0.0, abs=2.0e-15
    )
    assert abs(result.old_moment_residual_m3_per_rad[1]) > 1.0e-8
    assert result.changed_cell_ids.size > 0
    np.testing.assert_array_equal(
        result.weights_m3_per_rad[~result.band_mask],
        old[~result.band_mask],
    )
    assert result.core_unchanged is True
    assert np.all(result.changed_cell_ids >= 0)
    assert np.all(result.band_mask[result.changed_cell_ids])
    assert np.all(~result.band_mask[result.core_cell_ids])


def test_rectangle_matching_all_three_old_moments_is_unchanged() -> None:
    module = _module()
    geometry = _rectangle_geometry()
    boundary = build_boundary_geometry(geometry)
    first = module.build_boundary_volume_weights(geometry, boundary)
    geometry.M0 = np.array(first.weights_m3_per_rad, copy=True)
    second = module.build_boundary_volume_weights(geometry, boundary)
    assert second.old_scaled_moment_error <= 1.0e-12
    assert second.changed_cell_ids.size == 0
    np.testing.assert_array_equal(second.weights_m3_per_rad, geometry.M0)
    assert second.core_unchanged is True


@pytest.mark.parametrize(
    "factory",
    [
        _rectangle_geometry,
        *[lambda value=value: _fraction_geometry(value)
          for value in (0.25, 0.49, 0.5, 0.51, 0.75)],
        lambda: _circle_geometry(8),
        lambda: _circle_geometry(12),
        lambda: _circle_geometry(16),
        lambda: _circle_geometry(12, shift_R=0.01),
        lambda: _circle_geometry(12, shift_Z=0.01),
    ],
    ids=[
        "rectangle",
        "fraction-0.25",
        "fraction-0.49",
        "fraction-0.50",
        "fraction-0.51",
        "fraction-0.75",
        "circle-n8",
        "circle-n12",
        "circle-n16",
        "circle-n12-shift-r",
        "circle-n12-shift-z",
    ],
)
def test_physical_cases_are_positive_and_match_three_moments(factory) -> None:
    module = _module()
    geometry = factory()
    boundary = build_boundary_geometry(geometry)
    result = module.build_boundary_volume_weights(geometry, boundary)
    assert np.any(result.band_mask)
    assert np.all(result.weights_m3_per_rad > 0.0)
    assert result.positive is True
    assert result.core_unchanged is True
    assert result.constraint_rank == 3
    assert np.isfinite(result.constraint_condition)
    assert result.scaled_moment_error <= 1.0e-12
    np.testing.assert_allclose(
        result.new_moments_m3_per_rad,
        result.polygon_moments_m3_per_rad,
        rtol=0.0,
        atol=1.0e-12 * max(np.max(np.abs(result.polygon_moments_m3_per_rad)), 1.0),
    )
    np.testing.assert_array_equal(
        result.weights_m3_per_rad[~result.band_mask],
        np.asarray(geometry.M0)[~result.band_mask],
    )


def test_band_mask_is_independent_euclidean_two_cell_neighborhood() -> None:
    module = _module()
    geometry = _circle_geometry(12)
    boundary = build_boundary_geometry(geometry)
    result = module.build_boundary_volume_weights(geometry, boundary)
    polygon = np.column_stack(
        (boundary.polygon_R_m, boundary.polygon_Z_m)
    )
    radius = 2.0 * max(
        float(geometry.R[1] - geometry.R[0]),
        float(geometry.Z[1] - geometry.Z[0]),
    )
    expected = np.array(
        [
            _boundary_distance(R, Z, polygon) <= radius
            for R, Z in zip(geometry.cells.R, geometry.cells.Z)
        ]
    )
    np.testing.assert_array_equal(result.band_mask, expected)
    assert result.band_radius_m == pytest.approx(radius)


def test_deterministic_fingerprint_changes_with_physical_quadrature() -> None:
    module = _module()
    geometry = _circle_geometry(12)
    boundary = build_boundary_geometry(geometry)
    first = module.build_boundary_volume_weights(geometry, boundary)
    second = module.build_boundary_volume_weights(geometry, boundary)
    assert first.fingerprint == second.fingerprint
    other_geometry = _circle_geometry(16)
    other_boundary = build_boundary_geometry(other_geometry)
    other = module.build_boundary_volume_weights(
        other_geometry, other_boundary
    )
    assert first.fingerprint != other.fingerprint
    assert first.geometry_fingerprint == module._geometry_fingerprint(geometry)
    assert first.boundary_geometry_fingerprint == boundary.fingerprint


def test_inputs_and_returned_arrays_are_immutable() -> None:
    module = _module()
    geometry = _fraction_geometry(0.49)
    boundary = build_boundary_geometry(geometry)
    before = _snapshot(geometry, boundary)
    result = module.build_boundary_volume_weights(geometry, boundary)
    assert _snapshot(geometry, boundary) == before
    with pytest.raises(FrozenInstanceError):
        result.kind = "changed"
    with pytest.raises(ValueError, match="read-only"):
        result.weights_m3_per_rad[0] = 0.0
    for value in (
        result.weights_m3_per_rad,
        result.old_weights_m3_per_rad,
        result.band_mask,
        result.changed_cell_ids,
        result.polygon_moments_m3_per_rad,
        result.moment_residual_m3_per_rad,
    ):
        assert value.flags.writeable is False


@pytest.mark.parametrize("band_cells", [0, -1, 1.5, True])
def test_band_cells_requires_positive_integer(band_cells) -> None:
    module = _module()
    geometry = _rectangle_geometry()
    boundary = build_boundary_geometry(geometry)
    with pytest.raises(ValueError, match="band_cells"):
        module.build_boundary_volume_weights(
            geometry, boundary, band_cells=band_cells
        )


def test_mismatched_boundary_geometry_is_rejected() -> None:
    module = _module()
    geometry = _rectangle_geometry()
    other = _fraction_geometry(0.5)
    boundary = build_boundary_geometry(other)
    with pytest.raises(ValueError, match="boundary geometry does not match"):
        module.build_boundary_volume_weights(geometry, boundary)


def test_same_grid_and_keys_with_different_polygon_is_rejected() -> None:
    module = _module()
    R = 1.0 + 0.1 * np.arange(5)
    Z = -0.2 + 0.1 * np.arange(5)
    outer = build_geometry(
        R,
        Z,
        [(0.95, -0.25), (1.45, -0.25), (1.45, 0.25), (0.95, 0.25)],
    )
    inset = build_geometry(
        R,
        Z,
        [(0.96, -0.24), (1.44, -0.24), (1.44, 0.24), (0.96, 0.24)],
    )
    assert outer.cells.keys == inset.cells.keys
    np.testing.assert_array_equal(outer.active, inset.active)
    outer_boundary = build_boundary_geometry(outer)
    inset_boundary = build_boundary_geometry(inset)
    with pytest.raises(ValueError, match="physical polygon does not match"):
        module.build_boundary_volume_weights(outer, inset_boundary)

    outer_result = module.build_boundary_volume_weights(outer, outer_boundary)
    inset_result = module.build_boundary_volume_weights(inset, inset_boundary)
    assert outer_result.positive and inset_result.positive
    assert outer_result.fingerprint != inset_result.fingerprint


def test_one_redundant_closing_polygon_vertex_is_accepted() -> None:
    module = _module()
    geometry = _rectangle_geometry()
    boundary = build_boundary_geometry(geometry)
    closed = replace(
        boundary,
        polygon_R_m=np.append(
            boundary.polygon_R_m, boundary.polygon_R_m[0]
        ),
        polygon_Z_m=np.append(
            boundary.polygon_Z_m, boundary.polygon_Z_m[0]
        ),
    )
    result = module.build_boundary_volume_weights(geometry, closed)
    assert result.positive is True
    assert result.scaled_moment_error <= 1.0e-12


def test_frozen_core_dominating_physical_moment_is_explicitly_infeasible() -> None:
    module = _module()
    geometry = copy.copy(_rectangle_geometry())
    geometry.M0 = np.array(geometry.M0, copy=True)
    center = geometry.cells.find(2, 2)
    geometry.M0[center] *= 1000.0
    boundary = build_boundary_geometry(geometry)
    with pytest.raises(module.BoundaryQuadratureInfeasible) as raised:
        module.build_boundary_volume_weights(geometry, boundary)
    error = raised.value
    assert error.code in {"nonpositive_weights", "moment_residual"}
    assert error.diagnostics["band_cells"] == 2
    assert error.diagnostics["geometry_fingerprint"]
    assert error.diagnostics["boundary_geometry_fingerprint"] == boundary.fingerprint
