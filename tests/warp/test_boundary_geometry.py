from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, fields, is_dataclass

import numpy as np
import pytest

from fomhd.geometry import Point2, build_geometry, build_rectangle


def _module():
    try:
        from fomhd import boundary_geometry
    except ModuleNotFoundError as error:
        if error.name != "fomhd.boundary_geometry":
            raise
        pytest.fail("fomhd.boundary_geometry has not been implemented")
    return boundary_geometry


def _oblique_geometry():
    geometry = build_geometry(
        np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5]),
        np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]),
        [(0.96, -0.34), (1.54, -0.28), (1.52, 0.32), (0.96, 0.34)],
    )
    return geometry, _module().build_boundary_geometry(geometry)


def _fraction_geometry(fraction):
    nodes_r = 1.0 + 0.1 * np.arange(6)
    nodes_z = -0.2 + 0.1 * np.arange(5)
    selected_r = 1.2
    wall_r = selected_r + fraction * 0.1
    polygon = [
        (0.96, -0.25),
        (wall_r - 0.02, -0.25),
        (wall_r + 0.02, 0.25),
        (0.96, 0.25),
    ]
    return build_geometry(nodes_r, nodes_z, polygon)


def _centered_rectangle():
    return build_geometry(
        1.0 + 0.1 * np.arange(5),
        -0.2 + 0.1 * np.arange(5),
        [(0.95, -0.25), (1.45, -0.25), (1.45, 0.25), (0.95, 0.25)],
    )


def _find_intercept(data, cell_key, direction, side):
    matches = [
        item
        for item in data.intercepts
        if item.cell_key == cell_key
        and item.direction == direction
        and item.side == side
    ]
    assert len(matches) == 1
    return matches[0]


def _geometry_snapshot(geometry):
    arrays = {}
    for name in (
        "R",
        "Z",
        "active",
        "M0",
        "M1",
        "MB",
        "lr",
        "lz",
        "av",
        "wv",
        "Rbar_r",
        "Rbar_v",
        "cell_wall_distance",
    ):
        value = getattr(geometry, name)
        arrays[name] = (value.tobytes(), value.flags.writeable)
    entities = {}
    for name in ("cells", "r_faces", "z_faces", "vertices"):
        table = getattr(geometry, name)
        entities[name] = (
            tuple(table.keys),
            table.R.tobytes(),
            table.Z.tobytes(),
            table.boundary.tobytes(),
        )
    return {
        "arrays": arrays,
        "entities": entities,
        "polygon_input": tuple((p.R, p.Z) for p in geometry.polygon_input),
        "polygon": tuple((p.R, p.Z) for p in geometry.polygon),
        "walls": tuple(repr(wall) for wall in geometry.walls),
        "wall_polyline": tuple((p.R, p.Z) for p in geometry.wall_polyline),
    }


def test_oblique_intercepts_use_actual_piecewise_linear_wall() -> None:
    geometry, data = _oblique_geometry()
    module = _module()
    assert isinstance(data, module.BoundaryGeometry)
    assert data.units["length"] == "m"
    assert data.units["area"] == "m^2"
    assert data.units["R_area_integral"] == "m^3/rad"
    assert len(data.intercepts) == 4 * len(geometry.cells.keys)
    assert [item.intercept_id for item in data.intercepts] == list(
        range(len(data.intercepts))
    )

    right = _find_intercept(data, (2, 3), "R", "+")
    expected_right_r = 1.54 + (0.0 + 0.28) * (1.52 - 1.54) / 0.6
    segment_r = 1.52 - 1.54
    segment_z = 0.6
    segment_length = np.hypot(segment_r, segment_z)
    np.testing.assert_allclose(
        [right.R_m, right.Z_m, right.distance_m],
        [expected_right_r, 0.0, expected_right_r - 1.2],
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        [right.normal_R, right.normal_Z],
        [segment_z / segment_length, -segment_r / segment_length],
        rtol=0.0,
        atol=1.0e-14,
    )
    assert right.source_segment_id == 1
    assert right.origin_R_m == 1.2
    assert right.origin_Z_m == 0.0

    left = _find_intercept(data, (2, 3), "R", "-")
    np.testing.assert_allclose(
        [left.R_m, left.Z_m, left.distance_m, left.normal_R, left.normal_Z],
        [0.96, 0.0, 0.24, -1.0, 0.0],
        rtol=0.0,
        atol=1.0e-14,
    )


def test_translated_near_horizontal_wall_uses_angular_not_length_tolerance() -> None:
    module = _module()
    polygon_template = [
        (0.95, -0.202),
        (16.95, -0.197),
        (16.95, 0.25),
        (0.95, 0.25),
    ]
    segment_R, segment_Z = 16.0, 0.005
    segment_length = np.hypot(segment_R, segment_Z)
    normals = np.array(
        [
            [segment_Z / segment_length, -segment_R / segment_length],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    for shift in (0.0, 1.0e9):
        polygon = np.array(
            [(R + shift, Z) for R, Z in polygon_template],
            dtype=np.float64,
        )
        tolerance_m = 1.0e-12 * max(1.0, float(np.max(np.abs(polygon))))
        if shift:
            assert tolerance_m == pytest.approx(1.0e-3)
        else:
            assert tolerance_m < 1.0e-9
        segment_id, distance, wall_R, wall_Z = module._ray_intercept(
            1.0,
            0.0,
            shift + 1.2,
            -0.2,
            polygon,
            normals,
            tolerance_m,
            (),
        )
        assert segment_id == 0
        assert wall_R - shift == pytest.approx(
            0.95 + 0.4 * segment_R,
            rel=0.0,
            abs=2.5e-7,
        )
        assert distance == pytest.approx(
            0.95 + 0.4 * segment_R - 1.2,
            rel=0.0,
            abs=2.5e-7,
        )
        assert wall_Z == pytest.approx(-0.2, rel=0.0, abs=2.0e-15)


@pytest.mark.parametrize(
    "fraction, expected_base",
    [(0.25, 5), (0.49, 5), (0.50, 1), (0.51, 1), (0.75, 1)],
)
def test_clt_fraction_threshold_equality_is_retained(
    fraction: float, expected_base: int
) -> None:
    geometry = _fraction_geometry(fraction)
    data = _module().build_boundary_geometry(geometry)
    i, j = 2, 2
    assert data.r_base_class[i, j] == expected_base
    assert data.active_cell_mask[i, j] == geometry.active[i + geometry.R.size * j]
    assert np.any(data.r_class == 4)
    assert np.any(data.z_class == 4)
    if expected_base == 5:
        assert data.r_class[i, j] == 5
    else:
        assert data.r_class[i, j] in (1, 2)


def test_clt_half_grid_guard_has_distinct_dropped_and_retained_sides() -> None:
    module = _module()
    reference = module.build_boundary_geometry(_fraction_geometry(0.5))
    half_width = 0.5 * float(reference.R_m[1] - reference.R_m[0])
    guard = float(reference.tolerance_m)
    below = _fraction_geometry(0.5 - 4.0 * guard / (2.0 * half_width))
    above = _fraction_geometry(0.5 + 4.0 * guard / (2.0 * half_width))

    below_data = module.build_boundary_geometry(below)
    below_hit = _find_intercept(below_data, (2, 2), "R", "+")
    assert below_hit.distance_m < half_width - guard
    assert below_data.r_base_class[2, 2] == 5

    above_data = module.build_boundary_geometry(above)
    above_hit = _find_intercept(above_data, (2, 2), "R", "+")
    assert above_hit.distance_m > half_width - guard
    assert above_data.r_base_class[2, 2] == 1


def test_two_direction_layers_after_dropped_points_are_irregular() -> None:
    geometry = build_geometry(
        1.0 + 0.1 * np.arange(7),
        -0.2 + 0.1 * np.arange(5),
        [(0.96, -0.25), (1.375, -0.25), (1.475, 0.25), (0.96, 0.25)],
    )
    data = _module().build_boundary_geometry(geometry)
    assert data.r_base_class[4, 2] == 5
    assert data.r_base_class[3, 2] == 1
    assert data.r_base_class[2, 2] == 1
    assert data.r_class[3, 2] == 2
    assert data.r_class[2, 2] == 2


def test_vertex_tangent_ray_is_rejected_as_ambiguous() -> None:
    geometry = build_geometry(
        1.05 + 0.1 * np.arange(5),
        -0.2 + 0.1 * np.arange(5),
        [
            (1.0, -0.25),
            (1.45, -0.25),
            (1.5, 0.0),
            (1.45, 0.25),
            (1.0, 0.25),
        ],
    )
    module = _module()
    with pytest.raises(module.BoundaryGeometryError, match="vertex|tangent|ambiguous"):
        module.build_boundary_geometry(geometry)


def test_grid_shift_and_refinement_keep_distance_and_change_fingerprint() -> None:
    module = _module()
    polygon = [
        (0.975, -0.225),
        (1.425, -0.225),
        (1.425, 0.225),
        (0.975, 0.225),
    ]
    coarse = build_geometry(
        0.9 + 0.1 * np.arange(7), -0.3 + 0.1 * np.arange(7), polygon
    )
    fine = build_geometry(
        0.85 + 0.05 * np.arange(13), -0.35 + 0.05 * np.arange(13), polygon
    )
    coarse_data = module.build_boundary_geometry(coarse)
    fine_data = module.build_boundary_geometry(fine)
    coarse_i = int(np.argmin(np.abs(coarse.R - 1.2)))
    coarse_j = int(np.argmin(np.abs(coarse.Z)))
    fine_i = int(np.argmin(np.abs(fine.R - 1.2)))
    fine_j = int(np.argmin(np.abs(fine.Z)))
    assert coarse_data.r_wall_distance_m[coarse_i, coarse_j] == pytest.approx(0.225)
    assert fine_data.r_wall_distance_m[fine_i, fine_j] == pytest.approx(0.225)
    assert coarse_data.fingerprint != fine_data.fingerprint


@pytest.mark.parametrize("axis", ["R", "Z"])
def test_nonuniform_source_grid_is_rejected(axis: str) -> None:
    module = _module()
    geometry = copy.copy(_centered_rectangle())
    if axis == "R":
        geometry.R = np.array([1.0, 1.1, 1.2, 1.3001, 1.4])
    else:
        geometry.Z = np.array([-0.2, -0.1, 0.0, 0.1001, 0.2])
    with pytest.raises(module.BoundaryGeometryError, match="not uniform"):
        module.build_boundary_geometry(geometry)


@pytest.mark.parametrize("axis", ["R", "Z"])
def test_near_uniform_source_grid_within_tolerance_is_accepted(axis: str) -> None:
    R = 1.0 + 0.1 * np.arange(5)
    Z = -0.2 + 0.1 * np.arange(5)
    if axis == "R":
        R[2] += 5.0e-14
    else:
        Z[2] += 5.0e-14
    geometry = build_geometry(
        R,
        Z,
        [(0.95, -0.25), (1.45, -0.25), (1.45, 0.25), (0.95, 0.25)],
    )
    data = _module().build_boundary_geometry(geometry)
    assert data.fingerprint
    assert len(data.intercepts) == 4 * len(geometry.cells.keys)


def test_input_geometry_is_bitwise_unchanged() -> None:
    geometry, _ = _oblique_geometry()
    before = _geometry_snapshot(geometry)
    _module().build_boundary_geometry(geometry)
    after = _geometry_snapshot(geometry)
    assert after == before


def test_quadrature_has_two_gauss_points_per_segment_and_exact_moments() -> None:
    geometry = _centered_rectangle()
    data = _module().build_boundary_geometry(geometry)
    quadrature = data.wall_quadrature
    assert np.array_equal(quadrature.gauss_index, np.tile([0, 1], 4))
    assert np.array_equal(quadrature.segment_id, np.repeat(np.arange(4), 2))
    assert quadrature.total_length_m == pytest.approx(2.0, abs=0.0)
    assert quadrature.total_R_length_m2_per_rad == pytest.approx(2.4, abs=1.0e-15)
    np.testing.assert_allclose(
        np.sum(
            quadrature.length_weight_m
            * (quadrature.R_m * quadrature.normal_R
               + quadrature.Z_m * quadrature.normal_Z)
        ),
        2.0 * data.polygon_area_m2,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        [
            np.sum(quadrature.length_weight_m * quadrature.normal_R),
            np.sum(quadrature.length_weight_m * quadrature.normal_Z),
        ],
        [0.0, 0.0],
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        [quadrature.R_m[:2], quadrature.normal_R[:2], quadrature.normal_Z[:2]],
        [
            [1.0556624327025934, 1.3443375672974063],
            [0.0, 0.0],
            [-1.0, -1.0],
        ],
        rtol=0.0,
        atol=1.0e-14,
    )


def test_existing_faces_link_centres_and_actual_wall() -> None:
    geometry = _centered_rectangle()
    data = _module().build_boundary_geometry(geometry)
    r_interior = data.axis_links[geometry.r_faces.find(2, 2)]
    assert r_interior.axis == "R"
    assert r_interior.physical_component == "E_R"
    assert r_interior.interior is True
    assert r_interior.length_m == pytest.approx(0.1, rel=0.0, abs=1.0e-15)
    assert r_interior.endpoint_R_m == (1.1, 1.2)
    assert r_interior.endpoint_Z_m == (0.0, 0.0)
    assert r_interior.wall_intercept_id == -1

    r_boundary_index = len(geometry.r_faces.keys) + geometry.z_faces.find(2, 0)
    z_boundary = data.axis_links[r_boundary_index]
    assert z_boundary.axis == "Z"
    assert z_boundary.physical_component == "E_Z"
    assert z_boundary.interior is False
    assert z_boundary.length_m == pytest.approx(0.05, rel=0.0, abs=1.0e-15)
    wall = data.intercepts[z_boundary.wall_intercept_id]
    assert wall.direction == "Z"
    assert wall.side == "-"
    assert (wall.R_m, wall.Z_m) == pytest.approx((1.2, -0.25), abs=1.0e-15)

    r_wall = data.axis_links[geometry.r_faces.find(0, 2)]
    wall = data.intercepts[r_wall.wall_intercept_id]
    assert r_wall.length_m == pytest.approx(0.05, rel=0.0, abs=1.0e-15)
    assert (wall.R_m, wall.Z_m) == pytest.approx((0.95, 0.0), abs=1.0e-15)


def test_regular_vertex_dual_loop_is_exact_rectangle_metric() -> None:
    geometry = _centered_rectangle()
    data = _module().build_boundary_geometry(geometry)
    vertex_id = geometry.vertices.find(2, 2)
    dual = data.vertex_duals[vertex_id]
    np.testing.assert_array_equal(
        np.column_stack((dual.loop_R_m, dual.loop_Z_m)),
        [[1.1, -0.1], [1.2, -0.1], [1.2, 0.0], [1.1, 0.0]],
    )
    assert dual.area_m2 == pytest.approx(0.01, rel=0.0, abs=1.0e-15)
    assert dual.R_area_integral_m3_per_rad == pytest.approx(
        0.0115, rel=0.0, abs=1.0e-15
    )
    assert dual.centroid_R_m == pytest.approx(1.15, rel=0.0, abs=1.0e-15)
    assert dual.centroid_Z_m == pytest.approx(-0.05, rel=0.0, abs=1.0e-15)
    assert data.dual_covered_area_m2 == pytest.approx(data.polygon_area_m2)
    assert data.uncovered_area_m2 == 0.0
    assert data.coverage_fraction == pytest.approx(1.0)


def test_uncovered_physical_area_is_reported_without_redistribution() -> None:
    geometry = build_geometry(
        1.0 + 0.08 * np.arange(8),
        -0.28 + 0.08 * np.arange(8),
        [
            (1.4727846628625463, 0.06726984997457276),
            (1.2614801020089947, 0.22566788590692183),
            (1.073967408927662, -0.1910053973063555),
            (1.4621468762990761, -0.07439638997797192),
        ],
    )
    data = _module().build_boundary_geometry(geometry)
    assert data.uncovered_area_m2 > 1.0e-10
    assert data.coverage_fraction < 1.0
    assert (
        data.dual_covered_area_m2 + data.uncovered_area_m2
        == pytest.approx(data.polygon_area_m2, rel=0.0, abs=1.0e-14)
    )


def test_clockwise_convex_input_still_has_outward_normals() -> None:
    geometry = build_geometry(
        1.0 + 0.1 * np.arange(5),
        -0.2 + 0.1 * np.arange(5),
        [(0.95, 0.25), (1.45, 0.25), (1.45, -0.25), (0.95, -0.25)],
    )
    data = _module().build_boundary_geometry(geometry)
    assert data.polygon_area_m2 == pytest.approx(0.25, rel=0.0, abs=1.0e-15)
    np.testing.assert_allclose(
        [data.wall_quadrature.normal_R[0], data.wall_quadrature.normal_Z[0]],
        [0.0, 1.0],
        rtol=0.0,
        atol=1.0e-15,
    )


def test_arrays_and_records_are_immutable_and_fingerprint_is_deterministic() -> None:
    geometry = _centered_rectangle()
    module = _module()
    first = module.build_boundary_geometry(geometry)
    second = module.build_boundary_geometry(geometry)
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64
    with pytest.raises(ValueError, match="read-only"):
        first.r_class[0, 0] = 1
    with pytest.raises(FrozenInstanceError):
        first.kind = "changed"
    with pytest.raises(TypeError):
        first.units["length"] = "cm"
    for value in fields(first):
        item = getattr(first, value.name)
        if isinstance(item, np.ndarray):
            assert item.flags.writeable is False
    for item in (
        first.wall_quadrature.segment_id,
        first.wall_quadrature.R_m,
        first.wall_quadrature.normal_Z,
        first.vertex_duals[0].loop_R_m,
        first.vertex_duals[0].loop_Z_m,
    ):
        assert item.flags.writeable is False
    for item in first.axis_links + first.intercepts + first.vertex_duals:
        assert is_dataclass(item)
    staircase = build_rectangle(4, 3, 0.5, -0.2, 0.1, 0.1)
    identity = module.build_boundary_geometry(staircase, kind="staircase")
    assert "staircase" in identity.kind
    np.testing.assert_array_equal(
        identity.wall_polyline_R_m,
        [point.R for point in staircase.wall_polyline],
    )
    np.testing.assert_array_equal(
        identity.wall_polyline_Z_m,
        [point.Z for point in staircase.wall_polyline],
    )
    assert identity.wall_polyline_R_m.flags.writeable is False


def test_nonconvex_collinear_and_unsupported_inputs_are_rejected() -> None:
    module = _module()
    geometry = _centered_rectangle()
    nonconvex = copy.copy(geometry)
    nonconvex.polygon = [
        Point2(0.95, -0.25),
        Point2(1.3, -0.25),
        Point2(1.3, 0.0),
        Point2(1.1, 0.1),
        Point2(1.3, 0.25),
        Point2(0.95, 0.25),
    ]
    with pytest.raises(module.BoundaryGeometryError, match="convex|degenerate"):
        module.build_boundary_geometry(nonconvex)

    collinear = copy.copy(geometry)
    collinear.polygon = [
        Point2(0.95, -0.25),
        Point2(1.2, -0.25),
        Point2(1.45, -0.25),
        Point2(1.45, 0.25),
        Point2(0.95, 0.25),
    ]
    with pytest.raises(
        module.BoundaryGeometryError, match="collinear|convex|degenerate"
    ):
        module.build_boundary_geometry(collinear)

    with pytest.raises(module.BoundaryGeometryError, match="unsupported"):
        module.build_boundary_geometry(geometry, kind="normal_face")
