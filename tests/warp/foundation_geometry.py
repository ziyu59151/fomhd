from __future__ import annotations

import numpy as np
import pytest

from fomhd.geometry import (
    Point2,
    build_geometry,
    build_rectangle,
    build_stepped_disk,
    point_segment_distance,
)


def rectangle():
    return build_rectangle(4, 3, 0.5, -0.2, 0.1, 0.1)


def test_rectangle_topology_order_and_weights() -> None:
    geometry = rectangle()
    assert len(geometry.cells.keys) == 6
    assert len(geometry.r_faces.keys) == 8
    assert len(geometry.z_faces.keys) == 9
    assert len(geometry.vertices.keys) == 12
    assert len(geometry.walls) == 10
    assert geometry.cell_id(0, 0) == 0
    assert geometry.cell_id(1, 0) == 1
    assert geometry.cells.keys[0] == (0, 0)
    assert geometry.active.shape == (12,)
    assert geometry.active[0] == 1
    assert np.all(geometry.M0 > 0.0)
    assert np.all(geometry.M1 > 0.0)
    assert np.all(geometry.MB > 0.0)
    assert geometry.diagnostics.hausdorff_lower_m == pytest.approx(
        np.hypot(0.05, 0.05)
    )
    assert geometry.diagnostics.hausdorff_upper_m >= geometry.diagnostics.hausdorff_lower_m


def test_stepped_disk_has_closed_both_direction_staircase_wall() -> None:
    geometry = build_stepped_disk(
        26, 30, 0.25, -0.375, 0.025, 0.025, 0.55, 0.0, 0.22
    )
    assert len(geometry.cells.keys) > 100
    assert len(geometry.walls) > 20
    assert {wall.component for wall in geometry.walls} == {"r_face", "z_face"}
    assert geometry.wall_polyline[0] == geometry.wall_polyline[-1]


def test_polygon_rejections() -> None:
    R = np.array([0.5, 0.6, 0.7])
    Z = np.array([-0.1, 0.0, 0.1])
    with pytest.raises(Exception, match="at least three"):
        build_geometry(R, Z, [(0.5, -0.1), (0.7, 0.1)])
    with pytest.raises(Exception, match="non-finite"):
        build_geometry(R, Z, [(0.5, -0.1), (np.nan, 0.1), (0.7, 0.1)])
    with pytest.raises(Exception, match="self-intersecting"):
        build_geometry(
            R,
            Z,
            [(0.5, -0.1), (0.7, 0.1), (0.7, -0.1), (0.5, 0.1)],
        )


def test_grid_rejections() -> None:
    polygon = [(0.5, -0.1), (0.7, -0.1), (0.7, 0.1), (0.5, 0.1)]
    with pytest.raises(Exception, match="non-positive|R<=0"):
        build_geometry(np.array([0.5, 0.4, 0.3]), np.array([-0.1, 0.0, 0.1]), polygon)
    with pytest.raises(Exception, match="not uniform"):
        build_geometry(
            np.array([0.5, 0.6, 0.75]),
            np.array([-0.1, 0.0, 0.1]),
            polygon,
        )


def test_geometry_memory_cap_is_rejected() -> None:
    with pytest.raises(Exception, match="geometry memory budget exceeded"):
        build_rectangle(
            4,
            3,
            0.5,
            -0.2,
            0.1,
            0.1,
            options={"memory_cap_bytes": 8},
        )


@pytest.mark.parametrize("nodes", [12, 14])
@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
def test_grid_envelope_accepts_roundoff_but_rejects_exterior_polygon(
    nodes: int, scale: float
) -> None:
    centers = (np.arange(nodes, dtype=np.float64) + 0.5) * (1.0 / nodes)
    R, Z = scale * (1.0 + centers), scale * centers
    polygon = [(scale, 0.0), (2.0 * scale, 0.0),
               (2.0 * scale, scale), (scale, scale)]
    geometry = build_geometry(R, Z, polygon)
    assert len(geometry.cells.keys) == nodes * nodes
    assert len(geometry.walls) == 4 * nodes
    exterior = [(r, z + 1.0e-8 * scale if z else z) for r, z in polygon]
    with pytest.raises(Exception, match="outside the supplied grid"):
        build_geometry(R, Z, exterior)


def test_point_segment_distance_clamps_projection() -> None:
    assert point_segment_distance(Point2(0, 1), Point2(0, 0), Point2(1, 0)) == 1.0
    assert point_segment_distance(Point2(2, 1), Point2(0, 0), Point2(1, 0)) == pytest.approx(
        np.sqrt(2.0)
    )
    assert point_segment_distance(Point2(3, 4), Point2(0, 0), Point2(0, 0)) == 5.0
