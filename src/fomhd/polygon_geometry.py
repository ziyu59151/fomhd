"""Clipped-Voronoi geometry for the A050 ``polygon_dual_v1`` layout."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from scipy import sparse
from scipy.spatial import ConvexHull, Delaunay, QhullError, Voronoi, cKDTree

from .geometry import EntityTable, GeometryDiagnostics, Point2, point_segment_distance
from .types import Failure


_EPS = np.finfo(np.float64).eps


@dataclass
class PolygonGeometryOptions:
    memory_cap_bytes: int = 256 << 20
    grid_rtol: float = 1.0e-12
    guard_factor: float = 0.45
    vertex_rtol: float = 4096.0 * _EPS
    min_quality: float = 1.0e-14


@dataclass
class PolygonGeometryDiagnostics(GeometryDiagnostics):
    polygon_area: float = 0.0
    polygon_M0: float = 0.0
    coverage_area_error: float = 0.0
    coverage_M0_error: float = 0.0
    coverage_area_relative: float = 0.0
    coverage_M0_relative: float = 0.0
    min_cell_area: float = 0.0
    min_M0: float = 0.0
    min_face_length: float = 0.0
    min_face_distance: float = 0.0
    min_face_length_to_distance: float = 0.0
    min_vertex_area: float = 0.0
    min_vertex_volume: float = 0.0
    input_generators: int = 0
    retained_generators: int = 0
    removed_generators: int = 0
    min_generator_boundary_distance: float = 0.0
    guard_distance: float = 0.0
    estimated_bytes: int = 0


@dataclass
class PlaneWall:
    component: str
    face: int
    sign: int
    near_cell: int
    next_cell: int
    vertices: tuple[int, int]
    area_per_radian: float
    normal_R: float
    normal_Z: float
    tangent_R: float
    tangent_Z: float
    radius: float


@dataclass
class PolygonGeometry:
    kind: str
    R: np.ndarray
    Z: np.ndarray
    generator_R: np.ndarray
    generator_Z: np.ndarray
    polygon_input: list[Point2]
    polygon: list[Point2]
    active: np.ndarray
    cells: EntityTable
    faces: EntityTable
    r_faces: EntityTable
    z_faces: EntityTable
    vertices: EntityTable
    cell_polygons: list[np.ndarray]
    face_vertices: np.ndarray
    face_normals: np.ndarray
    face_tangents: np.ndarray
    face_lengths: np.ndarray
    face_distances: np.ndarray
    face_cells: np.ndarray
    b1: sparse.csr_matrix
    b2: sparse.csr_matrix
    area_c: np.ndarray
    M0: np.ndarray
    av: np.ndarray
    wv: np.ndarray
    Rbar_v: np.ndarray
    M1: np.ndarray
    MB: np.ndarray
    Rbar_r: np.ndarray
    lr: np.ndarray
    lz: np.ndarray
    cell_wall_distance: np.ndarray
    walls: list[PlaneWall]
    wall_polyline: list[Point2]
    diagnostics: PolygonGeometryDiagnostics
    cell_generators: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    def is_active(self, i: int, j: int) -> bool:
        if i < 0 or j < 0 or i >= self.R.size or j >= self.Z.size:
            return False
        return bool(self.active[i + self.R.size * j])

    def cell_id(self, i: int, j: int) -> int:
        return self.cells.find(i, j)

    def r_face_id(self, face: int, second: int = 0) -> int:
        return self.r_faces.find(face, second)

    def vertex_id(self, vertex: int, second: int = 0) -> int:
        return self.vertices.find(vertex, second)


def _checked_product(left: int, right: int, name: str) -> int:
    if left < 0 or right < 0:
        raise Failure("Memory", f"{name} size is negative")
    if left and right > sys.maxsize // left:
        raise Failure("Memory", f"{name} size product overflows")
    return left * right


def _estimate_bytes(nr: int, nz: int, polygon_vertices: int) -> int:
    grid = _checked_product(nr, nz, "polygon geometry")
    # The Voronoi graph, clipped polygons, canonical vertices, faces, and
    # transient arrays are all bounded by a generous per-generator allowance.
    generators = 1024
    boundary = 512 * polygon_vertices
    diagonal = 128 * min(nr, nz) * max(nr, nz)
    return grid * generators + boundary + diagonal


def _options(value: PolygonGeometryOptions | dict[str, Any] | None) -> PolygonGeometryOptions:
    if value is None:
        return PolygonGeometryOptions()
    if isinstance(value, PolygonGeometryOptions):
        return value
    try:
        return PolygonGeometryOptions(**value)
    except (TypeError, ValueError) as error:
        raise Failure("Geometry", f"invalid polygon geometry options: {error}") from error


def _validate_options(options: PolygonGeometryOptions) -> None:
    if not isinstance(options.memory_cap_bytes, int) or options.memory_cap_bytes < 0:
        raise Failure("Geometry", "memory cap must be a non-negative integer")
    if not 0.0 < options.grid_rtol < 1.0:
        raise Failure("Geometry", "invalid grid relative tolerance")
    if not 0.0 < options.guard_factor < 1.0:
        raise Failure("Geometry", "guard factor must lie in (0,1)")
    if not 0.0 < options.vertex_rtol < 1.0e-6:
        raise Failure("Geometry", "vertex tolerance is invalid")
    if not 0.0 < options.min_quality < 1.0:
        raise Failure("Geometry", "minimum geometry quality is invalid")


def _validate_grid(R: np.ndarray, Z: np.ndarray, rtol: float) -> tuple[float, float]:
    if R.ndim != 1 or Z.ndim != 1 or R.size < 2 or Z.size < 2:
        raise Failure("Geometry", "generator grids require one dimension and two nodes")
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z)):
        raise Failure("Geometry", "generator grid contains a non-finite value")
    hr = float(R[1] - R[0])
    hz = float(Z[1] - Z[0])
    if not (hr > 0.0 and hz > 0.0) or R[0] - 0.5 * hr <= 0.0:
        raise Failure("Geometry", "generator spacing is non-positive or reaches R<=0")
    for values, spacing, name in ((R, hr, "R"), (Z, hz, "Z")):
        for index in range(1, values.size):
            if not values[index] > values[index - 1]:
                raise Failure("Geometry", f"{name} generator grid is not increasing")
            expected = values[0] + index * spacing
            bound = rtol * abs(spacing) + 8.0 * _EPS * index * max(
                abs(values[0]), abs(values[index])
            )
            if abs(values[index] - expected) > bound:
                raise Failure("Geometry", f"{name} generator grid is not uniform")
    return hr, hz


def _point(value: Any) -> Point2:
    if isinstance(value, Point2):
        return Point2(float(value.R), float(value.Z))
    if isinstance(value, dict):
        return Point2(float(value["R"]), float(value["Z"]))
    return Point2(float(value[0]), float(value[1]))


def _signed_area(points: np.ndarray) -> np.longdouble:
    following = np.roll(points, -1, axis=0)
    products = (
        points[:, 0].astype(np.longdouble) * following[:, 1].astype(np.longdouble)
        - following[:, 0].astype(np.longdouble) * points[:, 1].astype(np.longdouble)
    )
    return np.longdouble(0.5) * np.sum(products, dtype=np.longdouble)


def _canonical_polygon(
    value: Iterable[Any],
) -> tuple[list[Point2], list[Point2], np.ndarray]:
    original = [_point(point) for point in value]
    if len(original) < 3:
        raise Failure("Geometry", "polygon requires at least three vertices")
    if any(not (math.isfinite(point.R) and math.isfinite(point.Z)) for point in original):
        raise Failure("Geometry", "polygon contains a non-finite coordinate")
    work = list(original)
    if work[0].R == work[-1].R and work[0].Z == work[-1].Z:
        work.pop()
    if len(work) < 3:
        raise Failure("Geometry", "polygon has fewer than three unique vertices")
    points = np.array([(point.R, point.Z) for point in work], dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(points))))
    tolerance = 64.0 * _EPS * scale
    for first in range(points.shape[0]):
        if np.linalg.norm(points[first] - points[(first + 1) % points.shape[0]]) <= tolerance:
            raise Failure("Geometry", "polygon has a zero-length segment")
        if np.any(np.linalg.norm(points[first + 1 :] - points[first], axis=1) <= tolerance):
            raise Failure("Geometry", "polygon has a duplicate vertex")
    area = _signed_area(points)
    if not np.isfinite(area) or abs(area) <= 128.0 * _EPS * scale * scale:
        raise Failure(
            "Geometry",
            "polygon is self-intersecting or non-convex; area cannot be certified",
        )
    try:
        hull = ConvexHull(points.copy())
    except QhullError as error:
        raise Failure("Geometry", "polygon is self-intersecting or non-convex") from error
    hull_vertices = np.asarray(hull.vertices, dtype=np.int64)
    if hull_vertices.size != points.shape[0]:
        raise Failure("Geometry", "polygon is self-intersecting or non-convex")
    hull_edges = {
        frozenset((int(hull_vertices[index]), int(hull_vertices[(index + 1) % hull_vertices.size])))
        for index in range(hull_vertices.size)
    }
    input_edges = {
        frozenset((index, (index + 1) % points.shape[0]))
        for index in range(points.shape[0])
    }
    if input_edges != hull_edges:
        raise Failure("Geometry", "polygon is self-intersecting or non-convex")
    canonical = list(work)
    if area < 0.0:
        canonical.reverse()
    canonical_points = np.array([(point.R, point.Z) for point in canonical], dtype=np.float64)
    return original, canonical, canonical_points


def _inside_convex(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    result = np.ones(points.shape[0], dtype=bool)
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % polygon.shape[0]]
        edge = second - first
        side = edge[0] * (points[:, 1] - first[1]) - edge[1] * (
            points[:, 0] - first[0]
        )
        result &= side > 0.0
    return result


def _boundary_distances(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    result = np.full(points.shape[0], np.inf, dtype=np.float64)
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % polygon.shape[0]]
        edge = second - first
        length = float(np.linalg.norm(edge))
        if not length > 0.0:
            raise Failure("Geometry", "polygon has a zero-length segment")
        side = np.abs(
            edge[0] * (points[:, 1] - first[1])
            - edge[1] * (points[:, 0] - first[0])
        ) / length
        result = np.minimum(result, side)
    return result


def _neighbors(points: np.ndarray) -> list[list[int]]:
    pairs: set[tuple[int, int]] = set()
    try:
        diagram = Voronoi(points.copy(), qhull_options="Qbb Qc Qz")
        for first, second in diagram.ridge_points:
            key = tuple(sorted((int(first), int(second))))
            if key[0] != key[1]:
                pairs.add(key)
    except QhullError:
        try:
            triangulation = Delaunay(points.copy(), qhull_options="Qbb Qc Qz")
            for simplex in triangulation.simplices:
                for first in range(3):
                    for second in range(first + 1, 3):
                        key = tuple(sorted((int(simplex[first]), int(simplex[second]))))
                        pairs.add(key)
        except QhullError as error:
            raise Failure(
                "GeometryAmbiguous",
                "retained generator condition is degenerate; Voronoi ownership is undefined",
            ) from error
    result = [[] for _ in range(points.shape[0])]
    for first, second in sorted(pairs):
        result[first].append(second)
        result[second].append(first)
    if any(not neighbors for neighbors in result):
        raise Failure(
            "GeometryAmbiguous",
            "a retained generator has no certified Voronoi neighbor",
        )
    return result


def _clip_halfplane(
    polygon: np.ndarray,
    midpoint: np.ndarray,
    direction: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    length = float(np.linalg.norm(direction))
    if not length > 0.0:
        raise Failure("GeometryAmbiguous", "Voronoi bisector is degenerate")
    unit = direction / length
    side = (polygon - midpoint) @ unit
    output: list[np.ndarray] = []
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % polygon.shape[0]]
        first_inside = side[index] <= tolerance
        second_inside = side[(index + 1) % polygon.shape[0]] <= tolerance
        if first_inside:
            output.append(first)
        if first_inside != second_inside:
            denominator = side[index] - side[(index + 1) % polygon.shape[0]]
            if denominator == 0.0:
                raise Failure("GeometryAmbiguous", "Voronoi clip intersection is ambiguous")
            parameter = side[index] / denominator
            output.append(first + parameter * (second - first))
    if not output:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(output, dtype=np.float64)


def _clean_polygon(points: np.ndarray, tolerance: float) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    work = np.asarray(points, dtype=np.float64)
    keep = np.linalg.norm(work - np.roll(work, -1, axis=0), axis=1) > tolerance
    work = work[keep]
    changed = True
    while changed and work.shape[0] >= 3:
        changed = False
        retained: list[np.ndarray] = []
        for index, point in enumerate(work):
            previous = work[(index - 1) % work.shape[0]]
            following = work[(index + 1) % work.shape[0]]
            before = point - previous
            after = following - point
            cross = abs(before[0] * after[1] - before[1] * after[0])
            length = np.linalg.norm(point - previous) + np.linalg.norm(following - point)
            if cross <= tolerance * max(length, tolerance):
                changed = True
                continue
            retained.append(point)
        work = np.asarray(retained, dtype=np.float64)
    return work


class _CanonicalVertices:
    def __init__(self, tolerance: float) -> None:
        self.tolerance = tolerance
        self.points: list[np.ndarray] = []
        self.bins: dict[tuple[int, int], list[int]] = {}

    def _key(self, point: np.ndarray) -> tuple[int, int]:
        return (
            int(np.rint(point[0] / self.tolerance)),
            int(np.rint(point[1] / self.tolerance)),
        )

    def add(self, point: np.ndarray) -> int:
        base = self._key(point)
        nearest = -1
        nearest_distance = math.inf
        for dr in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for index in self.bins.get((base[0] + dr, base[1] + dz), ()):
                    distance = float(np.linalg.norm(point - self.points[index]))
                    if distance < nearest_distance:
                        nearest = index
                        nearest_distance = distance
        if 0 <= nearest and nearest_distance <= self.tolerance:
            return nearest
        if 0 <= nearest and nearest_distance <= 4.0 * self.tolerance:
            raise Failure(
                "GeometryAmbiguous",
                "near-identical facet vertices are too close to certify",
            )
        index = len(self.points)
        self.points.append(point.copy())
        self.bins.setdefault(base, []).append(index)
        return index


def _metrics(points: np.ndarray) -> tuple[np.longdouble, np.longdouble, np.longdouble]:
    following = np.roll(points, -1, axis=0)
    x = points.astype(np.longdouble)
    y = following.astype(np.longdouble)
    cross = x[:, 0] * y[:, 1] - y[:, 0] * x[:, 1]
    area = np.sum(cross, dtype=np.longdouble) / np.longdouble(2.0)
    M0 = np.sum(
        (x[:, 0] + y[:, 0]) * cross, dtype=np.longdouble
    ) / np.longdouble(6.0)
    Mz = np.sum(
        (x[:, 1] + y[:, 1]) * cross, dtype=np.longdouble
    ) / np.longdouble(6.0)
    return area, M0, Mz


def _empty_table(name: str) -> EntityTable:
    return EntityTable(
        name=name,
        keys=[],
        dense_to_active=np.empty(0, dtype=np.int64),
        active_to_dense=np.empty(0, dtype=np.int64),
        R=np.empty(0, dtype=np.float64),
        Z=np.empty(0, dtype=np.float64),
        boundary=np.empty(0, dtype=np.uint8),
        lookup={},
    )


def _make_cells(
    keys: list[tuple[int, int]],
    nr: int,
    nz: int,
    centroids: np.ndarray,
) -> EntityTable:
    table = EntityTable(name="cells", keys=keys)
    table.dense_to_active = np.full(nr * nz, -1, dtype=np.int64)
    table.active_to_dense = np.empty(len(keys), dtype=np.int64)
    table.R = centroids[:, 0].copy()
    table.Z = centroids[:, 1].copy()
    table.boundary = np.zeros(len(keys), dtype=np.uint8)
    for cell, (i, j) in enumerate(keys):
        dense = i + nr * j
        table.dense_to_active[dense] = cell
        table.active_to_dense[cell] = dense
        table.lookup[(i, j)] = cell
    return table


def _make_vertices(points: np.ndarray, boundary: np.ndarray) -> EntityTable:
    table = EntityTable(name="vertices", keys=[(index, 0) for index in range(points.shape[0])])
    table.dense_to_active = np.arange(points.shape[0], dtype=np.int64)
    table.active_to_dense = np.arange(points.shape[0], dtype=np.int64)
    table.R = points[:, 0].copy()
    table.Z = points[:, 1].copy()
    table.boundary = boundary.astype(np.uint8)
    table.lookup = {(index, 0): index for index in range(points.shape[0])}
    return table


def _make_faces(midpoints: np.ndarray, boundary: np.ndarray) -> EntityTable:
    table = EntityTable(name="faces", keys=[(index, 0) for index in range(midpoints.shape[0])])
    table.dense_to_active = np.arange(midpoints.shape[0], dtype=np.int64)
    table.active_to_dense = np.arange(midpoints.shape[0], dtype=np.int64)
    table.R = midpoints[:, 0].copy()
    table.Z = midpoints[:, 1].copy()
    table.boundary = boundary.astype(np.uint8)
    table.lookup = {(index, 0): index for index in range(midpoints.shape[0])}
    return table


def build_polygon_geometry(
    R: Any,
    Z: Any,
    polygon: Iterable[Any],
    options: PolygonGeometryOptions | dict[str, Any] | None = None,
) -> PolygonGeometry:
    """Construct a guarded clipped-Voronoi true-polygon dual geometry."""
    settings = _options(options)
    _validate_options(settings)
    R_array = np.array(R, dtype=np.float64, copy=True)
    Z_array = np.array(Z, dtype=np.float64, copy=True)
    polygon_input, polygon_list, polygon_points = _canonical_polygon(polygon)
    estimate = _estimate_bytes(R_array.size, Z_array.size, polygon_points.shape[0])
    if settings.memory_cap_bytes and estimate > settings.memory_cap_bytes:
        raise Failure(
            "Memory",
            "polygon geometry memory budget exceeded: "
            f"requested={estimate} limit={settings.memory_cap_bytes}",
        )
    hr, hz = _validate_grid(R_array, Z_array, settings.grid_rtol)
    scale = max(
        1.0,
        float(np.max(np.abs(polygon_points))),
        float(np.max(np.abs(R_array))),
        float(np.max(np.abs(Z_array))),
    )
    vertex_tolerance = settings.vertex_rtol * scale
    grid_points = np.array(
        [(r, z) for z in Z_array for r in R_array],
        dtype=np.float64,
    )
    grid_keys = [
        (i, j)
        for j in range(Z_array.size)
        for i in range(R_array.size)
    ]
    inside = _inside_convex(grid_points, polygon_points)
    distances = np.full(grid_points.shape[0], np.inf, dtype=np.float64)
    if np.any(inside):
        distances[inside] = _boundary_distances(grid_points[inside], polygon_points)
    guard_distance = settings.guard_factor * min(hr, hz)
    retained_mask = inside & (distances >= guard_distance)
    retained_indices = np.flatnonzero(retained_mask)
    if retained_indices.size == 0:
        raise Failure(
            "Geometry",
            "generator guard removed every interior generator",
        )
    retained_points = grid_points[retained_indices]
    try:
        ConvexHull(retained_points.copy())
    except QhullError as error:
        raise Failure(
            "GeometryAmbiguous",
            "retained generator condition is degenerate or collinear",
        ) from error
    neighbors = _neighbors(retained_points)
    nc = retained_indices.size
    reference_area = abs(_signed_area(polygon_points))
    reference_M0 = _metrics(polygon_points)[1]
    clipped: list[np.ndarray] = []
    areas = np.empty(nc, dtype=np.float64)
    volumes = np.empty(nc, dtype=np.float64)
    zcenters = np.empty(nc, dtype=np.float64)
    for local, global_index in enumerate(retained_indices):
        current = polygon_points.copy()
        center = retained_points[local]
        for other in neighbors[local]:
            midpoint = 0.5 * (center + retained_points[other])
            current = _clip_halfplane(
                current,
                midpoint,
                retained_points[other] - center,
                vertex_tolerance,
            )
            if current.shape[0] < 3:
                raise Failure(
                    "GeometryAmbiguous",
                    f"Voronoi cell {int(global_index)} has fewer than three vertices",
                )
        current = _clean_polygon(current, vertex_tolerance)
        if current.shape[0] < 3:
            raise Failure(
                "Geometry",
                f"Voronoi cell {int(global_index)} collapsed after clipping",
            )
        area, M0, Mz = _metrics(current)
        area_scale = float(reference_area)
        if (
            not np.isfinite(area)
            or not np.isfinite(M0)
            or float(area) <= settings.min_quality * area_scale
            or float(M0) <= 0.0
        ):
            raise Failure(
                "Geometry",
                f"Voronoi cell {int(global_index)} has a non-positive or "
                "badly conditioned measure",
            )
        clipped.append(current)
        areas[local] = float(area)
        volumes[local] = float(M0)
        zcenters[local] = float(Mz / area)
    centroids = np.column_stack((volumes / areas, zcenters))

    canonical = _CanonicalVertices(vertex_tolerance)
    canonical_cells: list[np.ndarray] = []
    integer_cells: list[list[int]] = []
    for points in clipped:
        ids = [canonical.add(point) for point in points]
        canonical_cells.append(np.array(ids, dtype=np.int64))
        integer_cells.append(ids)
    vertex_points = np.asarray(canonical.points, dtype=np.float64)
    nv = vertex_points.shape[0]

    face_lookup: dict[tuple[int, int], tuple[int, int, int]] = {}
    face_records: list[tuple[int, int, int, int]] = []
    for cell, ids in enumerate(integer_cells):
        for offset, start in enumerate(ids):
            end = ids[(offset + 1) % len(ids)]
            key = tuple(sorted((int(start), int(end))))
            if key in face_lookup:
                first = face_lookup[key]
                if (start, end) == (first[0], first[1]):
                    raise Failure(
                        "GeometryAmbiguous",
                        "two cell edges share the same orientation",
                    )
                if first[2] < 0:
                    raise Failure("GeometryAmbiguous", "face has more than two owners")
                face_lookup[key] = (first[0], first[1], -2 - first[2])
                face_records[first[3]] = (first[0], first[1], first[2], cell)
            else:
                face_lookup[key] = (start, end, cell, len(face_records))
                face_records.append((start, end, cell, -1))

    nf = len(face_records)
    face_vertices = np.empty((nf, 2), dtype=np.int64)
    face_cells = np.empty((nf, 2), dtype=np.int64)
    face_lengths = np.empty(nf, dtype=np.float64)
    face_distances = np.empty(nf, dtype=np.float64)
    face_normals = np.empty((nf, 2), dtype=np.float64)
    face_tangents = np.empty((nf, 2), dtype=np.float64)
    face_midpoints = np.empty((nf, 2), dtype=np.float64)
    for face, (start, end, first_cell, second_cell) in enumerate(face_records):
        direction = vertex_points[end] - vertex_points[start]
        length = float(np.linalg.norm(direction))
        if not length > vertex_tolerance:
            raise Failure("Geometry", "canonical facet has a non-positive length")
        tangent = direction / length
        normal = np.array((tangent[1], -tangent[0]), dtype=np.float64)
        midpoint = 0.5 * (vertex_points[start] + vertex_points[end])
        if np.dot(normal, centroids[first_cell] - midpoint) >= 0.0:
            raise Failure("GeometryAmbiguous", "facet owner orientation is not outward")
        if second_cell >= 0:
            distance = abs(
                np.dot(
                    retained_points[second_cell] - retained_points[first_cell],
                    normal,
                )
            )
        else:
            distance = abs(
                np.dot(retained_points[first_cell] - midpoint, normal)
            )
        if not distance > settings.min_quality * min(hr, hz):
            raise Failure(
                "Geometry",
                "facet normal generator distance is non-positive or "
                "badly conditioned",
            )
        face_vertices[face] = (start, end)
        face_cells[face] = (first_cell, second_cell)
        face_lengths[face] = length
        face_distances[face] = distance
        face_normals[face] = normal
        face_tangents[face] = tangent
        face_midpoints[face] = midpoint

    rows = np.repeat(np.arange(nf, dtype=np.int64), 2)
    b1 = sparse.csr_matrix(
        (
            np.tile(np.array((-1.0, 1.0), dtype=np.float64), nf),
            (rows, face_vertices.reshape(-1)),
        ),
        shape=(nf, nv),
    )
    b2_rows: list[int] = []
    b2_cols: list[int] = []
    b2_data: list[float] = []
    for face, owners in enumerate(face_cells):
        b2_rows.append(int(owners[0]))
        b2_cols.append(face)
        b2_data.append(1.0)
        if owners[1] >= 0:
            b2_rows.append(int(owners[1]))
            b2_cols.append(face)
            b2_data.append(-1.0)
    b2 = sparse.csr_matrix(
        (np.asarray(b2_data, dtype=np.float64), (b2_rows, b2_cols)),
        shape=(nc, nf),
    )
    closure = b2 @ b1
    closure.eliminate_zeros()
    if closure.nnz != 0:
        raise Failure("GeometryAmbiguous", "facet incidence does not close")

    Rf = face_midpoints[:, 0]
    ell = face_lengths
    d = face_distances
    av = np.zeros(nv, dtype=np.float64)
    wv = np.zeros(nv, dtype=np.float64)
    for cell, ids in enumerate(integer_cells):
        generator = retained_points[cell]
        fan_area = np.longdouble(0.0)
        fan_volume = np.longdouble(0.0)
        for edge, first_id in enumerate(ids):
            second_id = ids[(edge + 1) % len(ids)]
            first = vertex_points[first_id]
            second = vertex_points[second_id]
            midpoint = 0.5 * (first + second)
            midpoint_vector = midpoint - generator
            for vertex_id, point in ((first_id, first), (second_id, second)):
                vector = point - generator
                triangle_area = np.longdouble(0.5) * abs(
                    np.longdouble(vector[0]) * np.longdouble(midpoint_vector[1])
                    - np.longdouble(vector[1]) * np.longdouble(midpoint_vector[0])
                )
                if not triangle_area > 0.0:
                    raise Failure(
                        "Geometry",
                        "selected barycentric triangle has a non-positive area",
                    )
                triangle_volume = (
                    triangle_area
                    * (
                        np.longdouble(generator[0])
                        + np.longdouble(point[0])
                        + np.longdouble(midpoint[0])
                    )
                    / np.longdouble(3.0)
                )
                av[vertex_id] += float(triangle_area)
                wv[vertex_id] += float(triangle_volume)
                fan_area += triangle_area
                fan_volume += triangle_volume
        area_tolerance = (
            np.longdouble(4096.0)
            * np.finfo(np.float64).eps
            * max(abs(np.longdouble(areas[cell])), reference_area)
        )
        volume_tolerance = (
            np.longdouble(4096.0)
            * np.finfo(np.float64).eps
            * max(abs(np.longdouble(volumes[cell])), abs(reference_M0))
        )
        if abs(fan_area - areas[cell]) > area_tolerance:
            raise Failure(
                "GeometryAmbiguous",
                "selected triangle fan does not recover the cell area",
            )
        if abs(fan_volume - volumes[cell]) > volume_tolerance:
            raise Failure(
                "GeometryAmbiguous",
                "selected triangle fan does not recover the cell R-volume",
            )
    if np.any(av <= 0.0) or np.any(wv <= 0.0):
        raise Failure("Geometry", "vertex dual area or volume is non-positive")
    Rbar_v = wv / av

    owner_tree = cKDTree(retained_points)
    keys = [grid_keys[int(index)] for index in retained_indices]
    cells = _make_cells(keys, R_array.size, Z_array.size, centroids)
    face_boundary = face_cells[:, 1] < 0
    vertices_boundary = np.zeros(nv, dtype=bool)
    vertices_boundary[face_vertices[face_boundary].reshape(-1)] = True
    faces = _make_faces(face_midpoints, face_boundary)
    vertices = _make_vertices(vertex_points, vertices_boundary)
    walls: list[PlaneWall] = []
    for face, owners in enumerate(face_cells):
        if owners[1] >= 0:
            continue
        owner = int(owners[0])
        _, nearest = owner_tree.query(retained_points[owner], k=2)
        next_cell = int(np.atleast_1d(nearest)[1])
        cells.boundary[owner] = 1
        cells.boundary[next_cell] = 1
        walls.append(
            PlaneWall(
                component="plane_face",
                face=face,
                sign=1,
                near_cell=owner,
                next_cell=next_cell,
                vertices=(int(face_vertices[face, 0]), int(face_vertices[face, 1])),
                area_per_radian=float(Rf[face] * ell[face]),
                normal_R=float(face_normals[face, 0]),
                normal_Z=float(face_normals[face, 1]),
                tangent_R=float(face_tangents[face, 0]),
                tangent_Z=float(face_tangents[face, 1]),
                radius=float(Rf[face]),
            )
        )
    if not walls:
        raise Failure("Geometry", "polygon mesh has no boundary wall")

    cell_wall_distance = np.array(
        [
            _boundary_distances(np.asarray([center]), polygon_points)[0]
            for center in centroids
        ],
        dtype=np.float64,
    )
    polygon_area = np.sum(areas, dtype=np.longdouble)
    polygon_M0 = np.sum(volumes, dtype=np.longdouble)
    area_error = abs(polygon_area - reference_area)
    M0_error = abs(polygon_M0 - reference_M0)
    area_relative = float(area_error / reference_area)
    M0_relative = float(M0_error / reference_M0)
    coverage_limit = max(4096.0 * _EPS, 1024.0 * _EPS * (nf + nv))
    if area_relative > coverage_limit or M0_relative > coverage_limit:
        raise Failure(
            "Geometry",
            "clipped Voronoi cells do not cover the complete polygon: "
            f"area_relative={area_relative:.3e} M0_relative={M0_relative:.3e}",
        )
    wall_polyline = polygon_list + [polygon_list[0]]
    diagnostics = PolygonGeometryDiagnostics(
        signed_polygon_area=float(reference_area),
        polygon_area=float(reference_area),
        active_cells=nc,
        wall_segments=len(walls),
        polygon_M0=float(reference_M0),
        coverage_area_error=float(area_error),
        coverage_M0_error=float(M0_error),
        coverage_area_relative=area_relative,
        coverage_M0_relative=M0_relative,
        min_cell_area=float(np.min(areas)),
        min_M0=float(np.min(volumes)),
        min_face_length=float(np.min(ell)),
        min_face_distance=float(np.min(d)),
        min_face_length_to_distance=float(np.min(ell / d)),
        min_vertex_area=float(np.min(av)),
        min_vertex_volume=float(np.min(wv)),
        input_generators=int(grid_points.shape[0]),
        retained_generators=int(nc),
        removed_generators=int(grid_points.shape[0] - nc),
        min_generator_boundary_distance=float(np.min(distances[retained_indices])),
        guard_distance=float(guard_distance),
        estimated_bytes=int(estimate),
    )
    active = np.zeros(R_array.size * Z_array.size, dtype=np.uint8)
    active[retained_indices] = 1
    return PolygonGeometry(
        kind="polygon_dual_v1",
        R=R_array,
        Z=Z_array,
        generator_R=retained_points[:, 0].copy(),
        generator_Z=retained_points[:, 1].copy(),
        polygon_input=polygon_input,
        polygon=polygon_list,
        active=active,
        cells=cells,
        faces=faces,
        r_faces=faces,
        z_faces=_empty_table("z_faces"),
        vertices=vertices,
        cell_polygons=[
            vertex_points[ids].copy()
            for ids in canonical_cells
        ],
        face_vertices=face_vertices,
        face_normals=face_normals,
        face_tangents=face_tangents,
        face_lengths=ell,
        face_distances=d,
        face_cells=face_cells,
        b1=b1,
        b2=b2,
        area_c=areas,
        M0=volumes,
        av=av,
        wv=wv,
        Rbar_v=Rbar_v,
        M1=np.concatenate((Rf * ell * d, volumes)),
        MB=np.concatenate((wv, Rf * ell * d)),
        Rbar_r=Rf.copy(),
        lr=d.copy(),
        lz=np.empty(0, dtype=np.float64),
        cell_wall_distance=cell_wall_distance,
        walls=walls,
        wall_polyline=wall_polyline,
        diagnostics=diagnostics,
        cell_generators=retained_points.copy(),
    )
