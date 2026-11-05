"""Closed staircase RZ geometry and topology matching the native mesh."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .types import Failure


EntityKey = tuple[int, int]


@dataclass
class Point2:
    R: float = 0.0
    Z: float = 0.0


@dataclass
class EntityTable:
    name: str = ""
    keys: list[EntityKey] = field(default_factory=list)
    dense_to_active: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    active_to_dense: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    R: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64))
    Z: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64))
    boundary: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint8))
    lookup: dict[EntityKey, int] = field(default_factory=dict)

    def contains(self, first: int, second: int) -> bool:
        return (int(first), int(second)) in self.lookup

    def find(self, first: int, second: int) -> int:
        try:
            return self.lookup[(int(first), int(second))]
        except KeyError as error:
            raise Failure("Geometry", f"{self.name} entity is absent") from error

    def dense_index(self, key: EntityKey) -> int:
        return int(self.active_to_dense[self.find(*key)])


@dataclass
class Wall:
    component: str = ""
    face: int = 0
    sign: int = 0
    near_cell: int = 0
    next_cell: int = 0
    vertices: tuple[int, int] = (0, 0)
    area_per_radian: float = 0.0


@dataclass
class GeometryOptions:
    memory_cap_bytes: int = 256 << 20
    grid_rtol: float = 1e-12


@dataclass
class GeometryDiagnostics:
    hausdorff_lower_m: float = 0.0
    hausdorff_upper_m: float = 0.0
    signed_polygon_area: float = 0.0
    active_cells: int = 0
    wall_segments: int = 0


@dataclass
class Geometry:
    R: np.ndarray
    Z: np.ndarray
    polygon_input: list[Point2]
    polygon: list[Point2]
    active: np.ndarray
    cells: EntityTable
    r_faces: EntityTable
    z_faces: EntityTable
    vertices: EntityTable
    M0: np.ndarray
    M1: np.ndarray
    MB: np.ndarray
    lr: np.ndarray
    lz: np.ndarray
    av: np.ndarray
    wv: np.ndarray
    Rbar_r: np.ndarray
    Rbar_v: np.ndarray
    cell_wall_distance: np.ndarray
    walls: list[Wall]
    wall_polyline: list[Point2]
    diagnostics: GeometryDiagnostics

    def is_active(self, i: int, j: int) -> bool:
        if i < 0 or j < 0 or i >= self.R.size or j >= self.Z.size:
            return False
        return bool(self.active[i + self.R.size * j])

    def cell_id(self, i: int, j: int) -> int:
        return self.cells.find(i, j)

    def r_face_id(self, e: int, j: int) -> int:
        return self.r_faces.find(e, j)

    def z_face_id(self, i: int, f: int) -> int:
        return self.z_faces.find(i, f)

    def vertex_id(self, e: int, f: int) -> int:
        return self.vertices.find(e, f)


def _checked_product(left: int, right: int, name: str) -> int:
    if left < 0 or right < 0:
        raise Failure("Geometry", f"{name} size is negative")
    if left != 0 and right > sys.maxsize // left:
        raise Failure("Geometry", f"{name} size product overflows")
    return left * right


def _checked_sum(left: int, right: int, name: str) -> int:
    if right > sys.maxsize - left:
        raise Failure("Geometry", f"{name} size sum overflows")
    return left + right


def _require_memory(required: int, cap: int) -> None:
    if cap != 0 and required > cap:
        raise Failure(
            "Memory",
            f"geometry memory budget exceeded: required={required} limit={cap}",
        )


def _point(value: Any) -> Point2:
    if isinstance(value, Point2):
        return value
    if isinstance(value, dict):
        return Point2(float(value["R"]), float(value["Z"]))
    return Point2(float(value[0]), float(value[1]))


def _finite_point(point: Point2) -> bool:
    return math.isfinite(point.R) and math.isfinite(point.Z)


def _same_point(left: Point2, right: Point2) -> bool:
    return left.R == right.R and left.Z == right.Z


def _orientation(a: Point2, b: Point2, c: Point2) -> np.longdouble:
    if (
        (a.R == b.R and c.R == a.R)
        or (a.Z == b.Z and c.Z == a.Z)
        or _same_point(c, a)
        or _same_point(c, b)
    ):
        return np.longdouble(0.0)
    ar, az = np.longdouble(a.R), np.longdouble(a.Z)
    br, bz = np.longdouble(b.R), np.longdouble(b.Z)
    cr, cz = np.longdouble(c.R), np.longdouble(c.Z)
    determinant = (br - ar) * (cz - az) - (bz - az) * (cr - ar)
    bound = (
        np.longdouble(16.0)
        * np.finfo(np.longdouble).eps
        * (
            (abs(br) + abs(ar)) * (abs(cz) + abs(az))
            + (abs(bz) + abs(az)) * (abs(cr) + abs(ar))
        )
    )
    if not np.isfinite(determinant) or abs(determinant) <= bound:
        raise Failure(
            "GeometryAmbiguous", "orientation sign cannot be certified"
        )
    return determinant


def _within_bounds(point: Point2, a: Point2, b: Point2) -> bool:
    return (
        min(a.R, b.R) <= point.R <= max(a.R, b.R)
        and min(a.Z, b.Z) <= point.Z <= max(a.Z, b.Z)
    )


def _segments_intersect(a: Point2, b: Point2, c: Point2, d: Point2) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if (
        ((o1 > 0 and o2 < 0) or (o1 < 0 and o2 > 0))
        and ((o3 > 0 and o4 < 0) or (o3 < 0 and o4 > 0))
    ):
        return True
    return (
        (o1 == 0 and _within_bounds(c, a, b))
        or (o2 == 0 and _within_bounds(d, a, b))
        or (o3 == 0 and _within_bounds(a, c, d))
        or (o4 == 0 and _within_bounds(b, c, d))
    )


def _signed_area(polygon: list[Point2]) -> np.longdouble:
    area = np.longdouble(0.0)
    scale = np.longdouble(0.0)
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        left = np.longdouble(first.R) * second.Z
        right = np.longdouble(second.R) * first.Z
        area += left - right
        scale += abs(left) + abs(right)
    bound = (
        np.longdouble(16 * len(polygon))
        * np.finfo(np.longdouble).eps
        * scale
    )
    if abs(area) <= bound:
        raise Failure("GeometryAmbiguous", "polygon area cannot be certified")
    return area / np.longdouble(2.0)


def _inside_polygon(R: float, Z: float, polygon: list[Point2]) -> bool:
    inside = False
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        if (first.Z <= Z < second.Z) or (second.Z <= Z < first.Z):
            side = _orientation(first, second, Point2(R, Z))
            if (second.Z > first.Z and side > 0) or (
                second.Z < first.Z and side < 0
            ):
                inside = not inside
    return inside


def _validate_polygon(
    input_polygon: Iterable[Any],
) -> tuple[list[Point2], list[Point2]]:
    original = [_point(point) for point in input_polygon]
    if len(original) < 3:
        raise Failure("Geometry", "polygon requires at least three vertices")
    if any(not _finite_point(point) for point in original):
        raise Failure("Geometry", "polygon contains a non-finite coordinate")
    polygon = list(original)
    if len(polygon) > 1 and _same_point(polygon[0], polygon[-1]):
        polygon.pop()
    if len(polygon) < 3:
        raise Failure(
            "Geometry", "polygon has fewer than three unique boundary vertices"
        )
    for first in range(len(polygon)):
        for second in range(first + 1, len(polygon)):
            if _same_point(polygon[first], polygon[second]):
                raise Failure("Geometry", "polygon has a duplicate vertex")
        if _same_point(polygon[first], polygon[(first + 1) % len(polygon)]):
            raise Failure("Geometry", "polygon has a zero-length segment")
    count = len(polygon)
    for first in range(count):
        a = polygon[first]
        b = polygon[(first + 1) % count]
        for second in range(first + 1, count):
            if second in (first, (first + 1) % count) or (second + 1) % count == first:
                continue
            if _segments_intersect(
                a, b, polygon[second], polygon[(second + 1) % count]
            ):
                raise Failure(
                    "Geometry", "polygon is self-intersecting or self-touching"
                )
    area = _signed_area(polygon)
    if area == 0 or not np.isfinite(area):
        raise Failure("Geometry", "polygon has zero or non-finite signed area")
    return original, polygon


def _validate_grid(R: np.ndarray, Z: np.ndarray, rtol: float) -> None:
    if not math.isfinite(rtol) or not 0.0 < rtol < 1.0:
        raise Failure("Geometry", "invalid grid relative tolerance")
    if R.size < 2 or Z.size < 2:
        raise Failure("Geometry", "grid requires at least two nodes per direction")
    if not np.all(np.isfinite(R)):
        raise Failure("Geometry", "R grid contains a non-finite value")
    if not np.all(np.isfinite(Z)):
        raise Failure("Geometry", "Z grid contains a non-finite value")
    hr = float(R[1] - R[0])
    hz = float(Z[1] - Z[0])
    if not (hr > 0.0 and hz > 0.0) or R[0] - 0.5 * hr <= 0.0:
        raise Failure("Geometry", "grid spacing is non-positive or reaches R<=0")
    eps = np.finfo(np.float64).eps
    for index in range(1, R.size):
        if not R[index] > R[index - 1]:
            raise Failure("Geometry", "R grid is not strictly increasing")
        expected = R[0] + float(index) * hr
        error = abs(R[index] - expected)
        bound = rtol * abs(hr) + 8.0 * eps * index * max(abs(R[0]), abs(R[index]))
        if error > bound:
            raise Failure("Geometry", "R grid is not uniform")
    for index in range(1, Z.size):
        if not Z[index] > Z[index - 1]:
            raise Failure("Geometry", "Z grid is not strictly increasing")
        expected = Z[0] + float(index) * hz
        error = abs(Z[index] - expected)
        bound = rtol * abs(hz) + 8.0 * eps * index * max(abs(Z[0]), abs(Z[index]))
        if error > bound:
            raise Failure("Geometry", "Z grid is not uniform")


def _ordered_keys(source: set[EntityKey]) -> list[EntityKey]:
    return sorted(source, key=lambda key: (key[1], key[0]))


def _make_table(
    name: str, source: set[EntityKey], width: int, height: int
) -> EntityTable:
    table = EntityTable(name=name, keys=_ordered_keys(source))
    dense_size = _checked_product(width, height, f"{name} dense table")
    table.dense_to_active = np.full(dense_size, -1, dtype=np.int64)
    table.active_to_dense = np.empty(len(table.keys), dtype=np.int64)
    table.R = np.empty(len(table.keys), dtype=np.float64)
    table.Z = np.empty(len(table.keys), dtype=np.float64)
    table.boundary = np.zeros(len(table.keys), dtype=np.uint8)
    for identifier, (first, second) in enumerate(table.keys):
        if first >= width or second >= height:
            raise Failure("Geometry", f"{name} key lies outside dense dimensions")
        dense = first + width * second
        if table.dense_to_active[dense] != -1:
            raise Failure("Geometry", f"{name} dense map is not one-to-one")
        table.dense_to_active[dense] = identifier
        table.active_to_dense[identifier] = dense
        table.lookup[(first, second)] = identifier
    return table


def _neighbor_count(
    geometry: Geometry, first: int, second: int, radial_face: bool
) -> int:
    count = 0
    if radial_face:
        if first != 0 and geometry.cells.contains(first - 1, second):
            count += 1
        if geometry.cells.contains(first, second):
            count += 1
    else:
        if second != 0 and geometry.cells.contains(first, second - 1):
            count += 1
        if geometry.cells.contains(first, second):
            count += 1
    return count


def _validate_wall_support(geometry: Geometry, wall: Wall) -> None:
    i, j = geometry.cells.keys[wall.near_cell]
    if wall.component == "r_face":
        next_i = i - 1 if wall.sign > 0 else i + 1
        if not geometry.cells.contains(next_i, j):
            raise Failure("Geometry", "radial wall lacks two-point inward support")
        wall.next_cell = geometry.cells.find(next_i, j)
    else:
        next_j = j - 1 if wall.sign > 0 else j + 1
        if not geometry.cells.contains(i, next_j):
            raise Failure("Geometry", "axial wall lacks two-point inward support")
        wall.next_cell = geometry.cells.find(i, next_j)


def _enumerate_topology(geometry: Geometry) -> None:
    cells: set[EntityKey] = set()
    r_faces: set[EntityKey] = set()
    z_faces: set[EntityKey] = set()
    vertices: set[EntityKey] = set()
    for j in range(geometry.Z.size):
        for i in range(geometry.R.size):
            if not geometry.is_active(i, j):
                continue
            cells.add((i, j))
            r_faces.update(((i, j), (i + 1, j)))
            z_faces.update(((i, j), (i, j + 1)))
            vertices.update(
                ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))
            )
    geometry.cells = _make_table("cells", cells, geometry.R.size, geometry.Z.size)
    geometry.r_faces = _make_table(
        "r_faces", r_faces, geometry.R.size + 1, geometry.Z.size
    )
    geometry.z_faces = _make_table(
        "z_faces", z_faces, geometry.R.size, geometry.Z.size + 1
    )
    geometry.vertices = _make_table(
        "vertices", vertices, geometry.R.size + 1, geometry.Z.size + 1
    )
    for identifier, (i, j) in enumerate(geometry.cells.keys):
        geometry.cells.R[identifier] = geometry.R[i]
        geometry.cells.Z[identifier] = geometry.Z[j]
    hr = float(geometry.R[1] - geometry.R[0])
    hz = float(geometry.Z[1] - geometry.Z[0])
    for identifier, (e, j) in enumerate(geometry.r_faces.keys):
        geometry.r_faces.R[identifier] = geometry.R[0] + (e - 0.5) * hr
        geometry.r_faces.Z[identifier] = geometry.Z[j]
    for identifier, (i, f) in enumerate(geometry.z_faces.keys):
        geometry.z_faces.R[identifier] = geometry.R[i]
        geometry.z_faces.Z[identifier] = geometry.Z[0] + (f - 0.5) * hz
    for identifier, (e, f) in enumerate(geometry.vertices.keys):
        geometry.vertices.R[identifier] = geometry.R[0] + (e - 0.5) * hr
        geometry.vertices.Z[identifier] = geometry.Z[0] + (f - 0.5) * hz

    for identifier, (e, j) in enumerate(geometry.r_faces.keys):
        count = _neighbor_count(geometry, e, j, True)
        if count not in (1, 2):
            raise Failure(
                "Geometry", "radial face does not have one or two cell neighbors"
            )
        geometry.r_faces.boundary[identifier] = 1 if count == 1 else 0
    for identifier, (i, f) in enumerate(geometry.z_faces.keys):
        count = _neighbor_count(geometry, i, f, False)
        if count not in (1, 2):
            raise Failure(
                "Geometry", "axial face does not have one or two cell neighbors"
            )
        geometry.z_faces.boundary[identifier] = 1 if count == 1 else 0
    for identifier, (e, f) in enumerate(geometry.vertices.keys):
        active = [
            e > 0 and f > 0 and geometry.cells.contains(e - 1, f - 1),
            geometry.cells.contains(e, f - 1),
            f > 0 and geometry.cells.contains(e - 1, f),
            geometry.cells.contains(e, f),
        ]
        count = sum(active)
        if count == 2 and not (
            (active[0] and active[1])
            or (active[0] and active[2])
            or (active[1] and active[3])
            or (active[2] and active[3])
        ):
            raise Failure("Geometry", "vertices meet only diagonally")
        geometry.vertices.boundary[identifier] = 1 if count < 4 else 0

    visited = np.zeros(len(geometry.cells.keys), dtype=bool)
    stack = [0]
    visited[0] = True
    reached = 0
    while stack:
        cell = stack.pop()
        reached += 1
        i, j = geometry.cells.keys[cell]
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if not geometry.cells.contains(ni, nj):
                continue
            neighbor = geometry.cells.find(ni, nj)
            if not visited[neighbor]:
                visited[neighbor] = True
                stack.append(neighbor)
    if reached != len(geometry.cells.keys):
        raise Failure("Geometry", "active cell domain is disconnected")

    walls: list[Wall] = []
    for identifier, (e, j) in enumerate(geometry.r_faces.keys):
        if geometry.r_faces.boundary[identifier] == 0:
            continue
        left = e > 0 and geometry.cells.contains(e - 1, j)
        right = geometry.cells.contains(e, j)
        if left == right:
            raise Failure(
                "Geometry", "radial wall face neighbor selection is ambiguous"
            )
        wall = Wall(
            component="r_face",
            face=identifier,
            sign=-1 if right else 1,
            near_cell=geometry.cells.find(e if right else e - 1, j),
            vertices=(
                geometry.vertices.find(e, j),
                geometry.vertices.find(e, j + 1),
            ),
        )
        _validate_wall_support(geometry, wall)
        geometry.cells.boundary[wall.near_cell] = 1
        geometry.cells.boundary[wall.next_cell] = 1
        walls.append(wall)
    for identifier, (i, f) in enumerate(geometry.z_faces.keys):
        if geometry.z_faces.boundary[identifier] == 0:
            continue
        lower = f > 0 and geometry.cells.contains(i, f - 1)
        upper = geometry.cells.contains(i, f)
        if lower == upper:
            raise Failure(
                "Geometry", "axial wall face neighbor selection is ambiguous"
            )
        wall = Wall(
            component="z_face",
            face=identifier,
            sign=-1 if upper else 1,
            near_cell=geometry.cells.find(i, f if upper else f - 1),
            vertices=(
                geometry.vertices.find(i, f),
                geometry.vertices.find(i + 1, f),
            ),
        )
        _validate_wall_support(geometry, wall)
        geometry.cells.boundary[wall.near_cell] = 1
        geometry.cells.boundary[wall.next_cell] = 1
        walls.append(wall)
    walls.sort(key=lambda wall: (wall.component, wall.face))
    if not walls:
        raise Failure("Geometry", "active domain has no boundary wall")
    geometry.walls = walls

    incidence: list[list[int]] = [[] for _ in geometry.vertices.keys]
    for identifier, wall in enumerate(walls):
        incidence[wall.vertices[0]].append(identifier)
        incidence[wall.vertices[1]].append(identifier)
    if any(segments and len(segments) != 2 for segments in incidence):
        raise Failure("Geometry", "wall graph is nonmanifold")
    wall_visited = np.zeros(len(walls), dtype=bool)
    wall_stack = [0]
    wall_visited[0] = True
    wall_reached = 0
    while wall_stack:
        wall = wall_stack.pop()
        wall_reached += 1
        for vertex in walls[wall].vertices:
            for following in incidence[vertex]:
                if not wall_visited[following]:
                    wall_visited[following] = True
                    wall_stack.append(following)
    if wall_reached != len(walls):
        raise Failure("Geometry", "wall graph is disconnected")
    euler = (
        len(geometry.vertices.keys)
        - len(geometry.r_faces.keys)
        - len(geometry.z_faces.keys)
        + len(geometry.cells.keys)
    )
    if euler != 1:
        raise Failure(
            "Geometry", "active domain does not have disk Euler characteristic"
        )

    current = walls[0].vertices[0]
    previous = -1
    geometry.wall_polyline = [
        Point2(geometry.vertices.R[current], geometry.vertices.Z[current])
    ]
    for _ in range(len(walls)):
        following = next(
            (candidate for candidate in incidence[current] if candidate != previous),
            -1,
        )
        if following < 0:
            raise Failure("Geometry", "wall cycle traversal failed")
        wall = walls[following]
        current = wall.vertices[1] if wall.vertices[0] == current else wall.vertices[0]
        geometry.wall_polyline.append(
            Point2(geometry.vertices.R[current], geometry.vertices.Z[current])
        )
        previous = following


def _compute_metrics(geometry: Geometry) -> None:
    hr = float(geometry.R[1] - geometry.R[0])
    hz = float(geometry.Z[1] - geometry.Z[0])
    nc = len(geometry.cells.keys)
    nr = len(geometry.r_faces.keys)
    nz = len(geometry.z_faces.keys)
    nv = len(geometry.vertices.keys)
    geometry.M0 = np.empty(nc, dtype=np.float64)
    geometry.lr = np.empty(nr, dtype=np.float64)
    geometry.lz = np.empty(nz, dtype=np.float64)
    geometry.Rbar_r = np.empty(nr, dtype=np.float64)
    geometry.av = np.zeros(nv, dtype=np.float64)
    geometry.wv = np.zeros(nv, dtype=np.float64)
    geometry.Rbar_v = np.empty(nv, dtype=np.float64)
    r_weight = np.empty(nr, dtype=np.float64)
    c_weight = np.empty(nc, dtype=np.float64)
    z_weight = np.empty(nz, dtype=np.float64)
    for cell in range(nc):
        geometry.M0[cell] = geometry.cells.R[cell] * hr * hz
        c_weight[cell] = geometry.M0[cell]
    for face, (e, j) in enumerate(geometry.r_faces.keys):
        count = _neighbor_count(geometry, e, j, True)
        geometry.lr[face] = hr if count == 2 else 0.5 * hr
        radius = geometry.r_faces.R[face]
        if count == 1:
            cell = (
                geometry.cells.find(e - 1, j)
                if e > 0 and geometry.cells.contains(e - 1, j)
                else geometry.cells.find(e, j)
            )
            geometry.Rbar_r[face] = 0.5 * (radius + geometry.cells.R[cell])
        else:
            geometry.Rbar_r[face] = radius
        r_weight[face] = radius * hz * geometry.lr[face]
    for face, (i, f) in enumerate(geometry.z_faces.keys):
        count = _neighbor_count(geometry, i, f, False)
        geometry.lz[face] = hz if count == 2 else 0.5 * hz
        z_weight[face] = geometry.z_faces.R[face] * hr * geometry.lz[face]
    for cell, (i, j) in enumerate(geometry.cells.keys):
        center = geometry.cells.R[cell]
        for de in range(2):
            for df in range(2):
                vertex = geometry.vertices.find(i + de, j + df)
                half = geometry.R[0] + (i + de - 0.5) * hr
                low, high = min(center, half), max(center, half)
                geometry.av[vertex] += hr * hz / 4.0
                geometry.wv[vertex] += (high * high - low * low) * hz / 4.0
    if np.any(geometry.av <= 0.0) or np.any(geometry.wv <= 0.0):
        raise Failure("Geometry", "vertex area or cylindrical weight is non-positive")
    geometry.Rbar_v = geometry.wv / geometry.av
    geometry.M1 = np.concatenate((r_weight, c_weight, z_weight))
    geometry.MB = np.concatenate(
        (
            z_weight,
            geometry.wv,
            geometry.Rbar_r * hz * geometry.lr,
        )
    )
    for wall in geometry.walls:
        if wall.component == "r_face":
            wall.area_per_radian = geometry.r_faces.R[wall.face] * hz
        else:
            wall.area_per_radian = geometry.z_faces.R[wall.face] * hr
    geometry.diagnostics.wall_segments = len(geometry.walls)


def point_segment_distance(point: Point2, a: Point2, b: Point2) -> float:
    dr, dz = b.R - a.R, b.Z - a.Z
    denominator = dr * dr + dz * dz
    if denominator == 0.0:
        return math.hypot(point.R - a.R, point.Z - a.Z)
    parameter = ((point.R - a.R) * dr + (point.Z - a.Z) * dz) / denominator
    parameter = min(1.0, max(0.0, parameter))
    return math.hypot(
        point.R - (a.R + parameter * dr),
        point.Z - (a.Z + parameter * dz),
    )


def _distance_to_segments(
    point: Point2, segments: list[tuple[Point2, Point2]]
) -> float:
    return min(
        (point_segment_distance(point, a, b) for a, b in segments),
        default=math.inf,
    )


def _compute_geometry_distance(geometry: Geometry) -> None:
    hr = float(geometry.R[1] - geometry.R[0])
    hz = float(geometry.Z[1] - geometry.Z[0])
    eta = min(hr, hz) / 8.0
    wall_segments = [
        (
            Point2(geometry.vertices.R[wall.vertices[0]], geometry.vertices.Z[wall.vertices[0]]),
            Point2(geometry.vertices.R[wall.vertices[1]], geometry.vertices.Z[wall.vertices[1]]),
        )
        for wall in geometry.walls
    ]
    polygon_segments = [
        (geometry.polygon[index], geometry.polygon[(index + 1) % len(geometry.polygon)])
        for index in range(len(geometry.polygon))
    ]
    lower = 0.0

    def accumulate(
        source: list[tuple[Point2, Point2]],
        target: list[tuple[Point2, Point2]],
    ) -> None:
        nonlocal lower
        for a, b in source:
            length = math.hypot(b.R - a.R, b.Z - a.Z)
            samples = max(1, int(math.ceil(length / eta)))
            for sample in range(samples + 1):
                parameter = sample / samples
                point = Point2(
                    a.R + parameter * (b.R - a.R),
                    a.Z + parameter * (b.Z - a.Z),
                )
                lower = max(lower, _distance_to_segments(point, target))

    accumulate(polygon_segments, wall_segments)
    accumulate(wall_segments, polygon_segments)
    geometry.diagnostics.hausdorff_lower_m = lower
    geometry.diagnostics.hausdorff_upper_m = lower + eta / 2.0
    geometry.diagnostics.active_cells = len(geometry.cells.keys)
    geometry.cell_wall_distance = np.array(
        [
            _distance_to_segments(Point2(R, Z), wall_segments)
            for R, Z in zip(geometry.cells.R, geometry.cells.Z)
        ],
        dtype=np.float64,
    )


def _estimate_geometry_bytes(
    nr: int, nz: int, polygon_vertices: int
) -> int:
    grid = _checked_product(nr, nz, "geometry estimate")
    dense_entities = _checked_sum(
        grid,
        _checked_product(_checked_sum(nr, 1, "geometry estimate"), nz, "geometry estimate"),
        "geometry estimate",
    )
    total_entities = _checked_sum(
        dense_entities,
        _checked_product(nr, _checked_sum(nz, 1, "geometry estimate"), "geometry estimate"),
        "geometry estimate",
    )
    return _checked_sum(
        _checked_sum(
            _checked_product(grid, 128, "geometry estimate"),
            _checked_product(total_entities, 96, "geometry estimate"),
            "geometry estimate",
        ),
        _checked_product(polygon_vertices, 128, "geometry estimate"),
        "geometry estimate",
    )


def _options(value: GeometryOptions | dict[str, Any] | None) -> GeometryOptions:
    if value is None:
        return GeometryOptions()
    if isinstance(value, GeometryOptions):
        return value
    return GeometryOptions(**value)


def build_geometry(
    R: Any,
    Z: Any,
    polygon: Iterable[Any],
    options: GeometryOptions | dict[str, Any] | None = None,
) -> Geometry:
    settings = _options(options)
    R_array = np.array(R, dtype=np.float64, copy=True)
    Z_array = np.array(Z, dtype=np.float64, copy=True)
    polygon_list = list(polygon)
    _require_memory(
        _estimate_geometry_bytes(R_array.size, Z_array.size, len(polygon_list)),
        settings.memory_cap_bytes,
    )
    _validate_grid(R_array, Z_array, settings.grid_rtol)
    polygon_input, canonical = _validate_polygon(polygon_list)
    hr = float(R_array[1] - R_array[0])
    hz = float(Z_array[1] - Z_array[0])
    lower_r = R_array[0] - 0.5 * hr
    upper_r = R_array[-1] + 0.5 * hr
    lower_z = Z_array[0] - 0.5 * hz
    upper_z = Z_array[-1] + 0.5 * hz
    rounding = 8.0 * np.finfo(np.float64).eps
    tolerance_r = rounding * max(abs(R_array[0]), abs(R_array[-1]), abs(hr))
    tolerance_z = rounding * max(abs(Z_array[0]), abs(Z_array[-1]), abs(hz))
    if any(
        point.R < lower_r - tolerance_r
        or point.R > upper_r + tolerance_r
        or point.Z < lower_z - tolerance_z
        or point.Z > upper_z + tolerance_z
        for point in canonical
    ):
        raise Failure(
            "Geometry", "polygon lies outside the supplied grid rectangles"
        )
    active = np.zeros(R_array.size * Z_array.size, dtype=np.uint8)
    active_count = 0
    for j, z in enumerate(Z_array):
        for i, r in enumerate(R_array):
            if _inside_polygon(float(r), float(z), canonical):
                active[i + R_array.size * j] = 1
                active_count += 1
    if active_count == 0:
        raise Failure("Geometry", "polygon masks no active cell centers")
    geometry = Geometry(
        R=R_array,
        Z=Z_array,
        polygon_input=polygon_input,
        polygon=canonical,
        active=active,
        cells=EntityTable(),
        r_faces=EntityTable(),
        z_faces=EntityTable(),
        vertices=EntityTable(),
        M0=np.empty(0),
        M1=np.empty(0),
        MB=np.empty(0),
        lr=np.empty(0),
        lz=np.empty(0),
        av=np.empty(0),
        wv=np.empty(0),
        Rbar_r=np.empty(0),
        Rbar_v=np.empty(0),
        cell_wall_distance=np.empty(0),
        walls=[],
        wall_polyline=[],
        diagnostics=GeometryDiagnostics(
            signed_polygon_area=float(_signed_area(canonical))
        ),
    )
    _enumerate_topology(geometry)
    _compute_metrics(geometry)
    _compute_geometry_distance(geometry)
    return geometry


def build_rectangle(
    nr: int,
    nz: int,
    r0: float,
    z0: float,
    h_r: float,
    h_z: float,
    options: GeometryOptions | dict[str, Any] | None = None,
) -> Geometry:
    if nr < 2 or nz < 2:
        raise Failure("Geometry", "rectangle requires at least 2 by 2 nodes")
    R = np.array([r0 + index * h_r for index in range(nr)], dtype=np.float64)
    Z = np.array([z0 + index * h_z for index in range(nz)], dtype=np.float64)
    polygon = [
        Point2(r0, z0),
        Point2(R[-1], z0),
        Point2(R[-1], Z[-1]),
        Point2(r0, Z[-1]),
    ]
    return build_geometry(R, Z, polygon, options)


def build_stepped_disk(
    nr: int,
    nz: int,
    r0: float,
    z0: float,
    h_r: float,
    h_z: float,
    center_r: float,
    center_z: float,
    radius: float,
    polygon_sides: int = 96,
    options: GeometryOptions | dict[str, Any] | None = None,
) -> Geometry:
    if polygon_sides < 8 or not radius > 0.0:
        raise Failure("Geometry", "stepped disk polygon input is invalid")
    if nr < 2 or nz < 2:
        raise Failure("Geometry", "stepped disk grid is too small")
    R = np.array([r0 + index * h_r for index in range(nr)], dtype=np.float64)
    Z = np.array([z0 + index * h_z for index in range(nz)], dtype=np.float64)
    polygon = [
        Point2(
            center_r + radius * math.cos(2.0 * math.pi * side / polygon_sides),
            center_z + radius * math.sin(2.0 * math.pi * side / polygon_sides),
        )
        for side in range(polygon_sides)
    ]
    return build_geometry(R, Z, polygon, options)
