from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import sparse

from fomhd.polygon_geometry import build_polygon_geometry


def rectangle_mesh() -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    R = np.linspace(0.55, 1.45, 10)
    Z = np.linspace(-0.45, 0.45, 10)
    polygon = [(0.5, -0.5), (1.5, -0.5), (1.5, 0.5), (0.5, 0.5)]
    return R, Z, polygon


def regular_polygon(
    center_r: float,
    center_z: float,
    radius: float,
    sides: int,
) -> list[tuple[float, float]]:
    return [
        (
            center_r + radius * math.cos(2.0 * math.pi * index / sides),
            center_z + radius * math.sin(2.0 * math.pi * index / sides),
        )
        for index in range(sides)
    ]


def polygon_area(polygon: list[tuple[float, float]]) -> float:
    return 0.5 * sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    )


def test_rectangle_true_polygon_area_volume_and_flux() -> None:
    R, Z, polygon = rectangle_mesh()
    geometry = build_polygon_geometry(R, Z, polygon)

    assert geometry.kind == "polygon_dual_v1"
    assert geometry.R is not R and np.array_equal(geometry.R, R)
    assert np.array_equal(geometry.Z, Z)
    assert geometry.active.shape == (R.size * Z.size,)
    assert np.all(geometry.active == 1)
    assert len(geometry.cells.keys) == R.size * Z.size
    assert np.all(geometry.area_c > 0.0)
    assert np.all(geometry.M0 > 0.0)
    assert np.sum(geometry.area_c) == pytest.approx(1.0, abs=3.0e-14)
    assert np.sum(geometry.M0) == pytest.approx(1.0, abs=3.0e-14)
    assert geometry.diagnostics.coverage_area_relative < 3.0e-14
    assert geometry.diagnostics.coverage_M0_relative < 3.0e-14
    assert geometry.diagnostics.min_face_length > 0.0
    assert geometry.diagnostics.min_face_distance > 0.0

    closure = geometry.b2 @ geometry.b1
    assert sparse.isspmatrix_csr(geometry.b1)
    assert sparse.isspmatrix_csr(geometry.b2)
    assert closure.shape == (len(geometry.cells.keys), len(geometry.vertices.keys))
    assert closure.nnz == 0


def test_face_geometry_is_oriented_outward_from_first_owner() -> None:
    R, Z, polygon = rectangle_mesh()
    geometry = build_polygon_geometry(R, Z, polygon)

    vertices = np.column_stack((geometry.vertices.R, geometry.vertices.Z))
    cell_centers = np.column_stack((geometry.cells.R, geometry.cells.Z))
    assert geometry.face_vertices.shape == (len(geometry.faces.keys), 2)
    assert geometry.face_cells.shape == (len(geometry.faces.keys), 2)
    assert np.all(geometry.face_lengths > 0.0)
    assert np.all(geometry.face_distances > 0.0)
    assert np.allclose(
        np.linalg.norm(geometry.face_normals, axis=1),
        1.0,
        rtol=0.0,
        atol=3.0e-15,
    )
    assert np.allclose(
        geometry.face_normals,
        np.column_stack(
            (
                geometry.face_tangents[:, 1],
                -geometry.face_tangents[:, 0],
            )
        ),
        rtol=0.0,
        atol=3.0e-15,
    )
    for face, owners in enumerate(geometry.face_cells):
        start, end = geometry.face_vertices[face]
        tangent = vertices[end] - vertices[start]
        assert geometry.face_lengths[face] == pytest.approx(
            np.linalg.norm(tangent), rel=2.0e-15
        )
        assert geometry.face_tangents[face] @ tangent > 0.0
        first_center = cell_centers[owners[0]]
        midpoint = 0.5 * (vertices[start] + vertices[end])
        assert geometry.face_normals[face] @ (first_center - midpoint) < 0.0
        if owners[1] >= 0:
            assert geometry.face_normals[face] @ (
                cell_centers[owners[1]] - midpoint
            ) > 0.0
        else:
            assert geometry.r_faces is geometry.faces
            assert geometry.z_faces.keys == []


def test_actual_facets_satisfy_cell_gauss_identities() -> None:
    R, Z, polygon = rectangle_mesh()
    geometry = build_polygon_geometry(R, Z, polygon)
    weighted_radial = geometry.faces.R * geometry.face_lengths * geometry.face_normals[:, 0]
    weighted_axial = geometry.face_lengths * geometry.face_normals[:, 1]

    for cell in range(len(geometry.cells.keys)):
        face_ids = geometry.b2.indices[
            geometry.b2.indptr[cell] : geometry.b2.indptr[cell + 1]
        ]
        outward_sign = geometry.b2.data[
            geometry.b2.indptr[cell] : geometry.b2.indptr[cell + 1]
        ]
        assert np.sum(outward_sign * weighted_radial[face_ids]) == pytest.approx(
            geometry.area_c[cell], rel=2.0e-15
        )
        assert abs(np.sum(outward_sign * weighted_axial[face_ids])) <= 2.0e-15
        assert geometry.area_c[cell] / geometry.M0[cell] == pytest.approx(
            1.0 / geometry.cells.R[cell], rel=2.0e-15
        )


def test_irregular_polygon_uses_actual_selected_triangle_duals() -> None:
    R = np.linspace(0.5, 1.7, 10)
    Z = np.linspace(-0.3, 0.8, 10)
    polygon = [
        (0.4, -0.3),
        (1.5, -0.35),
        (1.8, 0.2),
        (1.2, 0.8),
        (0.4, 0.55),
    ]
    geometry = build_polygon_geometry(R, Z, polygon)

    canonical = {
        (float(r), float(z)): index
        for index, (r, z) in enumerate(zip(geometry.vertices.R, geometry.vertices.Z))
    }
    expected_av = np.zeros(len(geometry.vertices.keys), dtype=np.float64)
    expected_wv = np.zeros(len(geometry.vertices.keys), dtype=np.float64)
    for cell, polygon_cell in enumerate(geometry.cell_polygons):
        generator = geometry.cell_generators[cell]
        fan_area = 0.0
        fan_volume = 0.0
        for edge, first in enumerate(polygon_cell):
            second = polygon_cell[(edge + 1) % len(polygon_cell)]
            midpoint = 0.5 * (first + second)
            first_id = canonical[(float(first[0]), float(first[1]))]
            second_id = canonical[(float(second[0]), float(second[1]))]
            first_vector = first - generator
            second_vector = second - generator
            midpoint_vector = midpoint - generator
            first_area = 0.5 * abs(
                first_vector[0] * midpoint_vector[1]
                - first_vector[1] * midpoint_vector[0]
            )
            second_area = 0.5 * abs(
                second_vector[0] * midpoint_vector[1]
                - second_vector[1] * midpoint_vector[0]
            )
            first_volume = (
                first_area * (generator[0] + first[0] + midpoint[0]) / 3.0
            )
            second_volume = (
                second_area * (generator[0] + second[0] + midpoint[0]) / 3.0
            )
            expected_av[first_id] += first_area
            expected_av[second_id] += second_area
            expected_wv[first_id] += first_volume
            expected_wv[second_id] += second_volume
            fan_area += first_area + second_area
            fan_volume += first_volume + second_volume

        assert fan_area == pytest.approx(geometry.area_c[cell], rel=4.0e-14)
        assert fan_volume == pytest.approx(geometry.M0[cell], rel=4.0e-14)
    assert np.all(expected_av > 0.0)
    assert np.all(expected_wv > 0.0)
    np.testing.assert_allclose(
        geometry.av,
        expected_av,
        rtol=4.0e-14,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        geometry.wv,
        expected_wv,
        rtol=4.0e-14,
        atol=2.0e-15,
    )


def test_guard_removes_generators_without_deleting_polygon_volume() -> None:
    R, Z, polygon = rectangle_mesh()
    geometry = build_polygon_geometry(
        R,
        Z,
        polygon,
        options={"guard_factor": 0.6},
    )

    assert 0 < len(geometry.cells.keys) < R.size * Z.size
    assert geometry.diagnostics.removed_generators > 0
    assert np.count_nonzero(geometry.active) == len(geometry.cells.keys)
    assert np.sum(geometry.area_c) == pytest.approx(
        geometry.diagnostics.polygon_area, abs=4.0e-14
    )
    assert np.sum(geometry.M0) == pytest.approx(
        geometry.diagnostics.polygon_M0, abs=4.0e-14
    )


def test_clockwise_input_has_same_positive_covariant_geometry() -> None:
    R, Z, polygon = rectangle_mesh()
    counterclockwise = build_polygon_geometry(R, Z, polygon)
    clockwise = build_polygon_geometry(R, Z, list(reversed(polygon)))

    assert np.sum(clockwise.area_c) == pytest.approx(
        np.sum(counterclockwise.area_c), rel=2.0e-15
    )
    assert np.sum(clockwise.M0) == pytest.approx(
        np.sum(counterclockwise.M0), rel=2.0e-15
    )
    assert (clockwise.b2 @ clockwise.b1).nnz == 0
    assert np.all(clockwise.face_distances > 0.0)
    assert np.all(clockwise.face_lengths > 0.0)


@pytest.mark.parametrize("resolution", [6, 8, 10])
@pytest.mark.parametrize(
    "center",
    [(1.0, 0.0), (1.013, -0.007), (0.987, 0.011)],
    ids=["centered", "shifted-a", "shifted-b"],
)
def test_circle_chords_cover_full_polygon_at_three_resolutions(
    resolution: int,
    center: tuple[float, float],
) -> None:
    center_r, center_z = center
    radius = 0.8
    sides = 4 * resolution
    polygon = regular_polygon(center_r, center_z, radius, sides)
    spacing = 2.0 * radius / resolution
    coordinate_r = center_r + spacing * (np.arange(resolution) - 0.5 * (resolution - 1))
    coordinate_z = center_z + spacing * (np.arange(resolution) - 0.5 * (resolution - 1))
    geometry = build_polygon_geometry(coordinate_r, coordinate_z, polygon)

    expected_area = 0.5 * sides * radius**2 * math.sin(2.0 * math.pi / sides)
    expected_volume = center_r * expected_area
    assert len(geometry.cells.keys) > 0
    assert np.all(geometry.area_c > 0.0)
    assert np.all(geometry.M0 > 0.0)
    assert np.all(geometry.M1 > 0.0)
    assert np.all(geometry.MB > 0.0)
    assert np.all(geometry.av > 0.0)
    assert np.all(geometry.wv > 0.0)
    assert np.sum(geometry.area_c) == pytest.approx(expected_area, rel=2.0e-11)
    assert np.sum(geometry.M0) == pytest.approx(expected_volume, rel=2.0e-11)
    assert geometry.diagnostics.coverage_area_relative < 2.0e-11
    assert geometry.diagnostics.coverage_M0_relative < 2.0e-11
    assert geometry.diagnostics.min_face_length >= 0.0
    assert geometry.diagnostics.min_face_distance >= 0.0


@pytest.mark.parametrize(
    "polygon, message",
    [
        ([(0.5, -0.5), (1.5, 0.5)], "at least three"),
        (
            [(0.5, -0.5), (1.5, -0.5), (1.5, 0.5), (1.0, 0.1), (0.5, 0.5)],
            "convex",
        ),
        (
            [(0.5, -0.5), (1.5, 0.5), (1.5, -0.5), (0.5, 0.5)],
            "self-intersect",
        ),
    ],
)
def test_malformed_polygon_is_rejected(
    polygon: list[tuple[float, float]],
    message: str,
) -> None:
    R, Z, _ = rectangle_mesh()
    with pytest.raises(Exception, match=message):
        build_polygon_geometry(R, Z, polygon)


def test_bad_grid_and_bad_generator_condition_are_rejected() -> None:
    _, Z, polygon = rectangle_mesh()
    with pytest.raises(Exception, match="uniform|increasing"):
        build_polygon_geometry(np.array([0.55, 0.66, 0.82]), Z, polygon)
    with pytest.raises(Exception, match="generator|condition"):
        build_polygon_geometry(
            np.array([1.0, 1.0 + 2.0e-15]),
            np.array([0.0, 0.1]),
            [(0.9, -0.1), (1.1, -0.1), (1.1, 0.2), (0.9, 0.2)],
        )


def test_geometry_capacity_stop_reports_estimates() -> None:
    R, Z, polygon = rectangle_mesh()
    with pytest.raises(Exception, match="memory budget exceeded: requested=.*limit=8"):
        build_polygon_geometry(R, Z, polygon, options={"memory_cap_bytes": 8})
