"""Read-only structured-grid boundary geometry for the CLT intercept probe.

This module is geometry-only. It preserves every Geometry cell, face, vertex,
and physical component identity while exposing directional contour intercepts,
CLT point-class metadata, existing-axis links, vertex-dual metric probes, and
one shared polygon quadrature. It does not modify fields or boundary physics.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from .geometry import GeometryOptions, _validate_grid
from .types import Failure


class BoundaryGeometryError(ValueError):
    """Raised when boundary geometry is unsupported or ambiguous."""


_LENGTH_REL_TOL = 1.0e-12
_ANGULAR_TOL = 1.0e-12
_GAUSS_NODES = (
    (0.5 - 0.5 / math.sqrt(3.0), 0.5),
    (0.5 + 0.5 / math.sqrt(3.0), 0.5),
)
_UNITS = MappingProxyType(
    {
        "length": "m",
        "area": "m^2",
        "R_area_integral": "m^3/rad",
        "wall_length_weight": "m",
        "wall_R_length_weight": "m^2/rad",
        "outward_normal": "dimensionless",
    }
)


@dataclass(frozen=True, slots=True)
class WallIntercept:
    """One actual directional polygon intercept from an unchanged cell center."""

    intercept_id: int
    cell_id: int
    cell_key: tuple[int, int]
    direction: str
    side: str
    origin_R_m: float
    origin_Z_m: float
    R_m: float
    Z_m: float
    distance_m: float
    normal_R: float
    normal_Z: float
    source_segment_id: int
    boundary_on_grid: bool


@dataclass(frozen=True, slots=True)
class AxisLink:
    """A metric link over an existing R-face or Z-face."""

    link_id: int
    face_id: int
    face_kind: str
    face_key: tuple[int, int]
    axis: str
    physical_component: str
    interior: bool
    cell_ids: tuple[int, ...]
    endpoint_R_m: tuple[float, float]
    endpoint_Z_m: tuple[float, float]
    length_m: float
    wall_intercept_id: int
    source_segment_id: int


@dataclass(frozen=True, slots=True)
class VertexDual:
    """A clipped logical rectangle about one existing grid vertex."""

    vertex_id: int
    vertex_key: tuple[int, int]
    loop_R_m: np.ndarray
    loop_Z_m: np.ndarray
    area_m2: float
    R_area_integral_m3_per_rad: float
    centroid_R_m: float
    centroid_Z_m: float


@dataclass(frozen=True, slots=True)
class WallQuadrature:
    """One shared two-point Gauss table on every true-wall segment."""

    segment_id: np.ndarray
    gauss_index: np.ndarray
    R_m: np.ndarray
    Z_m: np.ndarray
    normal_R: np.ndarray
    normal_Z: np.ndarray
    length_weight_m: np.ndarray
    R_length_weight_m2_per_rad: np.ndarray
    segment_length_m: np.ndarray
    total_length_m: float
    total_R_length_m2_per_rad: float


@dataclass(frozen=True, slots=True)
class BoundaryGeometry:
    """Immutable intercept-boundary data with original structured identities."""

    kind: str
    units: MappingProxyType
    fingerprint: str
    tolerance_m: float
    R_m: np.ndarray
    Z_m: np.ndarray
    active_cell_mask: np.ndarray
    boundary_on_grid: np.ndarray
    r_base_class: np.ndarray
    r_class: np.ndarray
    z_base_class: np.ndarray
    z_class: np.ndarray
    r_wall_distance_m: np.ndarray
    z_wall_distance_m: np.ndarray
    cell_ids: np.ndarray
    cell_keys: tuple[tuple[int, int], ...]
    r_face_keys: tuple[tuple[int, int], ...]
    z_face_keys: tuple[tuple[int, int], ...]
    vertex_keys: tuple[tuple[int, int], ...]
    polygon_R_m: np.ndarray
    polygon_Z_m: np.ndarray
    polygon_area_m2: float
    dual_covered_area_m2: float
    uncovered_area_m2: float
    coverage_fraction: float
    intercepts: tuple[WallIntercept, ...]
    axis_links: tuple[AxisLink, ...]
    vertex_duals: tuple[VertexDual, ...]
    wall_quadrature: WallQuadrature


@dataclass(frozen=True, slots=True)
class StaircaseIdentity:
    """Read-only snapshot of the old staircase positions and identities."""

    kind: str
    units: MappingProxyType
    fingerprint: str
    active_cell_mask: np.ndarray
    cell_R_m: np.ndarray
    cell_Z_m: np.ndarray
    r_face_R_m: np.ndarray
    r_face_Z_m: np.ndarray
    z_face_R_m: np.ndarray
    z_face_Z_m: np.ndarray
    wall_face_ids: np.ndarray
    wall_components: tuple[str, ...]
    wall_vertex_ids: tuple[tuple[int, int], ...]
    wall_polyline_R_m: np.ndarray
    wall_polyline_Z_m: np.ndarray


def _readonly(value: Any, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def _validate_source_grid(R: np.ndarray, Z: np.ndarray) -> None:
    try:
        _validate_grid(R, Z, GeometryOptions().grid_rtol)
    except Failure as error:
        raise BoundaryGeometryError(f"unsupported structured grid: {error}") from error


def _geometry_arrays(geometry: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        R = np.asarray(geometry.R, dtype=np.float64)
        Z = np.asarray(geometry.Z, dtype=np.float64)
        active = np.asarray(geometry.active)
    except (AttributeError, TypeError, ValueError) as error:
        raise BoundaryGeometryError(
            "geometry is missing structured R/Z data"
        ) from error
    if R.ndim != 1 or Z.ndim != 1:
        raise BoundaryGeometryError("R and Z grids must be one-dimensional")
    if R.size < 2 or Z.size < 2:
        raise BoundaryGeometryError("R and Z grids require at least two nodes")
    if active.shape != (R.size * Z.size,):
        raise BoundaryGeometryError("active mask has an incompatible structured extent")
    _validate_source_grid(R, Z)
    return R, Z, active


def _geometry_polygon(geometry: Any) -> tuple[np.ndarray, list[Any]]:
    polygon = getattr(geometry, "polygon", None)
    if polygon is None:
        raise BoundaryGeometryError("geometry has no canonical polygon")
    source = list(polygon)
    if source and math.hypot(
        float(source[0].R) - float(source[-1].R),
        float(source[0].Z) - float(source[-1].Z),
    ) == 0.0:
        source.pop()
    if len(source) < 3:
        raise BoundaryGeometryError("polygon requires at least three vertices")
    points = np.array([(point.R, point.Z) for point in source], dtype=np.float64)
    if not np.all(np.isfinite(points)):
        raise BoundaryGeometryError("polygon contains a nonfinite coordinate")
    if np.any(points[:, 0] <= 0.0):
        raise BoundaryGeometryError("polygon must lie wholly in R>0")
    return points, source


def _tolerance(
    R: np.ndarray, Z: np.ndarray, polygon: np.ndarray
) -> float:
    values = (
        float(np.max(np.abs(R))),
        float(np.max(np.abs(Z))),
        float(np.ptp(R)),
        float(np.ptp(Z)),
        float(np.max(np.abs(polygon))),
        float(np.ptp(polygon[:, 0])),
        float(np.ptp(polygon[:, 1])),
    )
    return _LENGTH_REL_TOL * max(1.0, *values)


def _validate_convex(
    polygon: np.ndarray, tolerance_m: float
) -> tuple[int, np.ndarray]:
    count = len(polygon)
    if count < 3:
        raise BoundaryGeometryError("polygon requires at least three vertices")
    for first in range(count):
        for second in range(first + 1, count):
            if np.linalg.norm(polygon[first] - polygon[second]) <= tolerance_m:
                raise BoundaryGeometryError("polygon has duplicate vertices")
    count_values = np.arange(count)
    starts = polygon
    ends = polygon[(count_values + 1) % count]
    vectors = ends - starts
    lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    if np.any(lengths <= tolerance_m):
        raise BoundaryGeometryError("polygon has a zero-length or degenerate edge")
    previous = starts - polygon[(count_values - 1) % count]
    crosses = (
        previous[:, 0] * vectors[:, 1] - previous[:, 1] * vectors[:, 0]
    )
    bounds = tolerance_m * (np.hypot(previous[:, 0], previous[:, 1]) + lengths)
    if np.any(np.abs(crosses) <= bounds):
        raise BoundaryGeometryError("polygon has collinear or degenerate vertices")
    signs = np.sign(crosses)
    if np.any(signs != signs[0]):
        raise BoundaryGeometryError("polygon is nonconvex")
    area2 = np.sum(
        starts[:, 0] * ends[:, 1] - ends[:, 0] * starts[:, 1],
        dtype=np.longdouble,
    )
    area = float(area2 / np.longdouble(2.0))
    if not math.isfinite(area) or abs(area) <= tolerance_m * tolerance_m:
        raise BoundaryGeometryError("polygon has degenerate or ambiguous area")
    return (1 if area > 0.0 else -1), lengths


def _edge_data(
    polygon: np.ndarray, lengths: np.ndarray, orientation: int
) -> tuple[np.ndarray, np.ndarray]:
    ends = np.roll(polygon, -1, axis=0)
    directions = (ends - polygon) / lengths[:, None]
    normals = orientation * np.column_stack(
        (directions[:, 1], -directions[:, 0])
    )
    norm_lengths = np.hypot(normals[:, 0], normals[:, 1])
    normals /= norm_lengths[:, None]
    return directions, normals


def _polygon_area(polygon: np.ndarray) -> float:
    ends = np.roll(polygon, -1, axis=0)
    return abs(
        float(
            np.sum(
                polygon[:, 0] * ends[:, 1] - ends[:, 0] * polygon[:, 1],
                dtype=np.longdouble,
            )
            / np.longdouble(2.0)
        )
    )


def _quadrature(
    polygon: np.ndarray,
    lengths: np.ndarray,
    normals: np.ndarray,
) -> WallQuadrature:
    ends = np.roll(polygon, -1, axis=0)
    segment_id: list[int] = []
    gauss_index: list[int] = []
    points_R: list[float] = []
    points_Z: list[float] = []
    normal_R: list[float] = []
    normal_Z: list[float] = []
    weights: list[float] = []
    R_weights: list[float] = []
    for segment, (start, end) in enumerate(zip(polygon, ends)):
        for local, (node, unit_weight) in enumerate(_GAUSS_NODES):
            radius = start[0] + node * (end[0] - start[0])
            height = start[1] + node * (end[1] - start[1])
            weight = lengths[segment] * unit_weight
            segment_id.append(segment)
            gauss_index.append(local)
            points_R.append(radius)
            points_Z.append(height)
            normal_R.append(normals[segment, 0])
            normal_Z.append(normals[segment, 1])
            weights.append(weight)
            R_weights.append(radius * weight)
    length_array = np.repeat(lengths, len(_GAUSS_NODES))
    weight_array = np.asarray(weights, dtype=np.float64)
    R_weight_array = np.asarray(R_weights, dtype=np.float64)
    return WallQuadrature(
        segment_id=_readonly(segment_id, np.int64),
        gauss_index=_readonly(gauss_index, np.int64),
        R_m=_readonly(points_R, np.float64),
        Z_m=_readonly(points_Z, np.float64),
        normal_R=_readonly(normal_R, np.float64),
        normal_Z=_readonly(normal_Z, np.float64),
        length_weight_m=_readonly(weight_array),
        R_length_weight_m2_per_rad=_readonly(R_weight_array),
        segment_length_m=_readonly(length_array),
        total_length_m=float(np.sum(weight_array)),
        total_R_length_m2_per_rad=float(np.sum(R_weight_array)),
    )


def _segment_distance(
    point_R: float,
    point_Z: float,
    start_R: float,
    start_Z: float,
    end_R: float,
    end_Z: float,
) -> float:
    delta_R = end_R - start_R
    delta_Z = end_Z - start_Z
    denominator = delta_R * delta_R + delta_Z * delta_Z
    if denominator == 0.0:
        return math.hypot(point_R - start_R, point_Z - start_Z)
    parameter = (
        (point_R - start_R) * delta_R + (point_Z - start_Z) * delta_Z
    ) / denominator
    parameter = min(1.0, max(0.0, parameter))
    return math.hypot(
        point_R - (start_R + parameter * delta_R),
        point_Z - (start_Z + parameter * delta_Z),
    )


def _boundary_sources(
    point_R: float,
    point_Z: float,
    polygon: np.ndarray,
    tolerance_m: float,
) -> tuple[int, ...]:
    sources = []
    for segment, start in enumerate(polygon):
        end = polygon[(segment + 1) % len(polygon)]
        if _segment_distance(
            point_R, point_Z, start[0], start[1], end[0], end[1]
        ) <= tolerance_m:
            sources.append(segment)
    return tuple(sources)


def _ray_intercept(
    direction_R: float,
    direction_Z: float,
    origin_R: float,
    origin_Z: float,
    polygon: np.ndarray,
    normals: np.ndarray,
    tolerance_m: float,
    boundary_sources: tuple[int, ...],
) -> tuple[int, float, float, float]:
    boundary_segment = -1
    ignore_origin = False
    if boundary_sources:
        if len(boundary_sources) != 1:
            raise BoundaryGeometryError(
                "boundary-on-grid ambiguity at a contour vertex"
            )
        boundary_segment = boundary_sources[0]
        outward_projection = (
            direction_R * normals[boundary_segment, 0]
            + direction_Z * normals[boundary_segment, 1]
        )
        if abs(outward_projection) <= _ANGULAR_TOL:
            raise BoundaryGeometryError(
                "boundary-on-grid tangent ray is ambiguous"
            )
        if outward_projection > 0.0:
            return boundary_segment, 0.0, origin_R, origin_Z
        ignore_origin = True

    contacts: list[tuple[float, float, float, int]] = []
    for segment, start in enumerate(polygon):
        end = polygon[(segment + 1) % len(polygon)]
        segment_R = end[0] - start[0]
        segment_Z = end[1] - start[1]
        segment_length = math.hypot(segment_R, segment_Z)
        denominator = (
            direction_R * segment_Z - direction_Z * segment_R
        )
        delta_R = start[0] - origin_R
        delta_Z = start[1] - origin_Z
        if abs(denominator) <= _ANGULAR_TOL * segment_length:
            line_offset = abs(direction_R * delta_Z - direction_Z * delta_R)
            start_projection = direction_R * delta_R + direction_Z * delta_Z
            end_projection = (
                direction_R * (end[0] - origin_R)
                + direction_Z * (end[1] - origin_Z)
            )
            if (
                line_offset <= 2.0 * tolerance_m
                and max(start_projection, end_projection) >= -tolerance_m
            ):
                raise BoundaryGeometryError(
                    "polygon tangent or near-parallel overlap is ambiguous"
                )
            continue
        ray_parameter = (
            delta_R * segment_Z - delta_Z * segment_R
        ) / denominator
        segment_parameter = (
            delta_R * direction_Z - delta_Z * direction_R
        ) / denominator
        if ray_parameter < -tolerance_m:
            continue
        if (
            segment_parameter < -tolerance_m / segment_length
            or segment_parameter > 1.0 + tolerance_m / segment_length
        ):
            continue
        if (
            ignore_origin
            and segment == boundary_segment
            and abs(ray_parameter) <= tolerance_m
        ):
            continue
        ray_parameter = max(0.0, ray_parameter)
        point_R = origin_R + ray_parameter * direction_R
        point_Z = origin_Z + ray_parameter * direction_Z
        contacts.append((ray_parameter, point_R, point_Z, segment))

    if not contacts:
        raise BoundaryGeometryError(
            "cell ray has no contour intersection; geometry is unsupported"
        )
    contacts.sort(key=lambda item: item[0])
    nearest = contacts[0]
    cluster = tuple(
        item
        for item in contacts
        if math.hypot(item[1] - nearest[1], item[2] - nearest[2])
        <= 2.0 * tolerance_m
    )
    if len(cluster) != len(contacts):
        raise BoundaryGeometryError("cell ray has multiple contour intersections")
    source_segments = {item[3] for item in cluster}
    if len(source_segments) != 1:
        raise BoundaryGeometryError("polygon vertex contact is ambiguous")
    if any(
        math.hypot(nearest[1] - vertex[0], nearest[2] - vertex[1])
        <= tolerance_m
        for vertex in polygon
    ):
        raise BoundaryGeometryError("polygon vertex contact is ambiguous")
    segment = nearest[3]
    distance = math.hypot(nearest[1] - origin_R, nearest[2] - origin_Z)
    return segment, distance, nearest[1], nearest[2]


def _intercepts(
    geometry: Any,
    R: np.ndarray,
    Z: np.ndarray,
    polygon: np.ndarray,
    normals: np.ndarray,
    tolerance_m: float,
    boundary_on_grid: np.ndarray,
) -> tuple[
    tuple[WallIntercept, ...],
    dict[tuple[int, str, str], WallIntercept],
    np.ndarray,
    np.ndarray,
]:
    records: list[WallIntercept] = []
    lookup: dict[tuple[int, str, str], WallIntercept] = {}
    r_distance = np.full((R.size, Z.size), np.nan, dtype=np.float64)
    z_distance = np.full((R.size, Z.size), np.nan, dtype=np.float64)
    directions = (
        ("R", "-", -1.0, 0.0),
        ("R", "+", 1.0, 0.0),
        ("Z", "-", 0.0, -1.0),
        ("Z", "+", 0.0, 1.0),
    )
    for cell_id, (i, j) in enumerate(geometry.cells.keys):
        origin_R = float(geometry.cells.R[cell_id])
        origin_Z = float(geometry.cells.Z[cell_id])
        sources = _boundary_sources(
            origin_R, origin_Z, polygon, tolerance_m
        )
        if bool(sources) != bool(boundary_on_grid[i, j]):
            raise BoundaryGeometryError("boundary-on-grid metadata is inconsistent")
        distances: dict[str, list[float]] = {"R": [], "Z": []}
        for direction, side, direction_R, direction_Z in directions:
            segment, distance, wall_R, wall_Z = _ray_intercept(
                direction_R,
                direction_Z,
                origin_R,
                origin_Z,
                polygon,
                normals,
                tolerance_m,
                sources,
            )
            record = WallIntercept(
                intercept_id=len(records),
                cell_id=cell_id,
                cell_key=(int(i), int(j)),
                direction=direction,
                side=side,
                origin_R_m=origin_R,
                origin_Z_m=origin_Z,
                R_m=wall_R,
                Z_m=wall_Z,
                distance_m=distance,
                normal_R=float(normals[segment, 0]),
                normal_Z=float(normals[segment, 1]),
                source_segment_id=segment,
                boundary_on_grid=bool(sources),
            )
            records.append(record)
            lookup[(cell_id, direction, side)] = record
            distances[direction].append(distance)
        r_distance[i, j] = min(distances["R"])
        z_distance[i, j] = min(distances["Z"])
    return (
        tuple(records),
        lookup,
        r_distance,
        z_distance,
    )


def _point_classes(
    R: np.ndarray,
    Z: np.ndarray,
    active: np.ndarray,
    r_distance: np.ndarray,
    z_distance: np.ndarray,
    tolerance_m: float,
) -> tuple[np.ndarray, ...]:
    r_base = np.full((R.size, Z.size), 4, dtype=np.uint8)
    z_base = np.full((R.size, Z.size), 4, dtype=np.uint8)
    h_r = float(R[1] - R[0])
    h_z = float(Z[1] - Z[0])
    for j in range(Z.size):
        for i in range(R.size):
            if not bool(active[i + R.size * j]):
                continue
            # FOMHD keeps a tol_m equality band around CLT's literal half-cell
            # comparison; this is a roundoff guard, not a source-exact clone.
            r_base[i, j] = (
                5 if r_distance[i, j] < 0.5 * h_r - tolerance_m else 1
            )
            z_base[i, j] = (
                5 if z_distance[i, j] < 0.5 * h_z - tolerance_m else 1
            )

    def layers(base: np.ndarray, raw_axis: int) -> np.ndarray:
        result = base.copy()
        candidates: set[tuple[int, int]] = set()
        shape = base.shape
        outer = shape[1 - raw_axis]
        inner = shape[raw_axis]
        for outer_index in range(outer):
            for inner_index in range(1, inner - 1):
                index = [0, 0]
                index[raw_axis] = inner_index
                index[1 - raw_axis] = outer_index
                here = tuple(index)
                if base[here] != 1:
                    continue
                lower = list(index)
                upper = list(index)
                lower[raw_axis] -= 1
                upper[raw_axis] += 1
                lower_tuple = tuple(lower)
                upper_tuple = tuple(upper)
                if base[lower_tuple] > 2 and base[upper_tuple] == 1:
                    candidates.add(here)
                    candidates.add(upper_tuple)
                if base[upper_tuple] > 2 and base[lower_tuple] == 1:
                    candidates.add(here)
                    candidates.add(lower_tuple)
        for candidate in candidates:
            result[candidate] = 2
        return result

    return r_base, layers(r_base, 0), z_base, layers(z_base, 1)


def _cell_id(geometry: Any, i: int, j: int) -> int:
    try:
        return geometry.cells.find(i, j)
    except Exception as error:
        raise BoundaryGeometryError(
            f"existing cell support is missing at {(i, j)}"
        ) from error


def _axis_links(
    geometry: Any,
    intercepts: dict[tuple[int, str, str], WallIntercept],
) -> tuple[AxisLink, ...]:
    links: list[AxisLink] = []
    for face_id, (e, j) in enumerate(geometry.r_faces.keys):
        left = e > 0 and geometry.cells.contains(e - 1, j)
        right = geometry.cells.contains(e, j)
        if left and right:
            left_id = _cell_id(geometry, e - 1, j)
            right_id = _cell_id(geometry, e, j)
            start_R = float(geometry.cells.R[left_id])
            end_R = float(geometry.cells.R[right_id])
            start_Z = float(geometry.cells.Z[left_id])
            end_Z = float(geometry.cells.Z[right_id])
            length = abs(end_R - start_R)
            cell_ids = (left_id, right_id)
            wall_id = -1
            source_id = -1
            interior = True
        elif left != right:
            cell_id = _cell_id(geometry, e if right else e - 1, j)
            side = "-" if right else "+"
            wall = intercepts[(cell_id, "R", side)]
            start_R = float(geometry.cells.R[cell_id])
            start_Z = float(geometry.cells.Z[cell_id])
            end_R, end_Z = wall.R_m, wall.Z_m
            length = wall.distance_m
            cell_ids = (cell_id,)
            wall_id = wall.intercept_id
            source_id = wall.source_segment_id
            interior = False
        else:
            raise BoundaryGeometryError("r_face has ambiguous cell support")
        links.append(
            AxisLink(
                link_id=len(links),
                face_id=face_id,
                face_kind="r_face",
                face_key=(int(e), int(j)),
                axis="R",
                physical_component="E_R",
                interior=interior,
                cell_ids=cell_ids,
                endpoint_R_m=(start_R, end_R),
                endpoint_Z_m=(start_Z, end_Z),
                length_m=length,
                wall_intercept_id=wall_id,
                source_segment_id=source_id,
            )
        )

    for face_id, (i, f) in enumerate(geometry.z_faces.keys):
        lower = f > 0 and geometry.cells.contains(i, f - 1)
        upper = geometry.cells.contains(i, f)
        if lower and upper:
            lower_id = _cell_id(geometry, i, f - 1)
            upper_id = _cell_id(geometry, i, f)
            start_R = float(geometry.cells.R[lower_id])
            end_R = float(geometry.cells.R[upper_id])
            start_Z = float(geometry.cells.Z[lower_id])
            end_Z = float(geometry.cells.Z[upper_id])
            length = abs(end_Z - start_Z)
            cell_ids = (lower_id, upper_id)
            wall_id = -1
            source_id = -1
            interior = True
        elif lower != upper:
            cell_id = _cell_id(geometry, i, f if upper else f - 1)
            side = "-" if upper else "+"
            wall = intercepts[(cell_id, "Z", side)]
            start_R = float(geometry.cells.R[cell_id])
            start_Z = float(geometry.cells.Z[cell_id])
            end_R, end_Z = wall.R_m, wall.Z_m
            length = wall.distance_m
            cell_ids = (cell_id,)
            wall_id = wall.intercept_id
            source_id = wall.source_segment_id
            interior = False
        else:
            raise BoundaryGeometryError("z_face has ambiguous cell support")
        links.append(
            AxisLink(
                link_id=len(links),
                face_id=face_id,
                face_kind="z_face",
                face_key=(int(i), int(f)),
                axis="Z",
                physical_component="E_Z",
                interior=interior,
                cell_ids=cell_ids,
                endpoint_R_m=(start_R, end_R),
                endpoint_Z_m=(start_Z, end_Z),
                length_m=length,
                wall_intercept_id=wall_id,
                source_segment_id=source_id,
            )
        )
    return tuple(links)


def _clip_half_plane(
    points: list[tuple[float, float]],
    start: np.ndarray,
    end: np.ndarray,
    orientation: int,
    tolerance_m: float,
) -> list[tuple[float, float]]:
    if not points:
        return []
    edge_R = float(end[0] - start[0])
    edge_Z = float(end[1] - start[1])
    edge_length = math.hypot(edge_R, edge_Z)

    def side(point: tuple[float, float]) -> float:
        return orientation * (
            edge_R * (point[1] - start[1])
            - edge_Z * (point[0] - start[0])
        )

    output: list[tuple[float, float]] = []
    previous = points[-1]
    previous_side = side(previous)
    previous_inside = previous_side >= -tolerance_m * edge_length
    for current in points:
        current_side = side(current)
        current_inside = current_side >= -tolerance_m * edge_length
        if current_inside != previous_inside:
            denominator = previous_side - current_side
            if abs(denominator) > 0.0:
                fraction = previous_side / denominator
                output.append(
                    (
                        previous[0] + fraction * (current[0] - previous[0]),
                        previous[1] + fraction * (current[1] - previous[1]),
                    )
                )
        if current_inside:
            output.append(current)
        previous = current
        previous_side = current_side
        previous_inside = current_inside
    return output


def _clean_loop(
    points: list[tuple[float, float]], tolerance_m: float
) -> list[tuple[float, float]]:
    changed = True
    cleaned = list(points)
    while changed and len(cleaned) >= 3:
        changed = False
        compact: list[tuple[float, float]] = []
        for point in cleaned:
            if not compact or math.hypot(
                point[0] - compact[-1][0], point[1] - compact[-1][1]
            ) > tolerance_m:
                compact.append(point)
        if len(compact) > 1 and math.hypot(
            compact[0][0] - compact[-1][0],
            compact[0][1] - compact[-1][1],
        ) <= tolerance_m:
            compact.pop()
        if len(compact) < 3:
            return []
        again: list[tuple[float, float]] = []
        count = len(compact)
        for index, point in enumerate(compact):
            before = compact[(index - 1) % count]
            after = compact[(index + 1) % count]
            first_R = point[0] - before[0]
            first_Z = point[1] - before[1]
            second_R = after[0] - point[0]
            second_Z = after[1] - point[1]
            cross = first_R * second_Z - first_Z * second_R
            bound = tolerance_m * (
                math.hypot(first_R, first_Z)
                + math.hypot(second_R, second_Z)
            )
            if abs(cross) <= bound:
                changed = True
            else:
                again.append(point)
        cleaned = compact if not changed else again
    return cleaned if len(cleaned) >= 3 else []


def _clip_rectangle(
    polygon: np.ndarray,
    lower_R: float,
    upper_R: float,
    lower_Z: float,
    upper_Z: float,
    orientation: int,
    tolerance_m: float,
) -> list[tuple[float, float]]:
    points = [
        (lower_R, lower_Z),
        (upper_R, lower_Z),
        (upper_R, upper_Z),
        (lower_R, upper_Z),
    ]
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        points = _clip_half_plane(
            points, start, end, orientation, tolerance_m
        )
        if not points:
            return []
    return _clean_loop(points, tolerance_m)


def _dual_metrics(
    loop: list[tuple[float, float]]
) -> tuple[float, float, float, float]:
    coordinates = np.asarray(loop, dtype=np.longdouble)
    following = np.roll(coordinates, -1, axis=0)
    cross = (
        coordinates[:, 0] * following[:, 1]
        - following[:, 0] * coordinates[:, 1]
    )
    signed_area2 = np.sum(cross)
    signed_area = float(signed_area2 / np.longdouble(2.0))
    if signed_area <= 0.0 or not math.isfinite(signed_area):
        raise BoundaryGeometryError("clipped vertex dual has invalid orientation")
    R_integral = float(
        np.sum(
            (coordinates[:, 0] + following[:, 0])
            * cross
            / np.longdouble(6.0)
        )
    )
    Z_integral = float(
        np.sum(
            (coordinates[:, 1] + following[:, 1])
            * cross
            / np.longdouble(6.0)
        )
    )
    return (
        signed_area,
        R_integral,
        R_integral / signed_area,
        Z_integral / signed_area,
    )


def _vertex_duals(
    geometry: Any,
    R: np.ndarray,
    Z: np.ndarray,
    polygon: np.ndarray,
    orientation: int,
    tolerance_m: float,
) -> tuple[VertexDual, ...]:
    h_r = float(R[1] - R[0])
    h_z = float(Z[1] - Z[0])
    records: list[VertexDual] = []
    for vertex_id, (e, f) in enumerate(geometry.vertices.keys):
        lower_R = float(R[e - 1]) if e > 0 else float(R[0] - h_r)
        upper_R = float(R[e]) if e < R.size else float(R[-1] + h_r)
        lower_Z = float(Z[f - 1]) if f > 0 else float(Z[0] - h_z)
        upper_Z = float(Z[f]) if f < Z.size else float(Z[-1] + h_z)
        loop = _clip_rectangle(
            polygon,
            lower_R,
            upper_R,
            lower_Z,
            upper_Z,
            orientation,
            tolerance_m,
        )
        if not loop:
            raise BoundaryGeometryError(
                f"existing vertex {int(e), int(f)} has no physical dual area"
            )
        area, R_integral, centroid_R, centroid_Z = _dual_metrics(loop)
        if area <= tolerance_m * tolerance_m:
            raise BoundaryGeometryError(
                f"existing vertex {int(e), int(f)} has degenerate dual area"
            )
        loop_array = np.asarray(loop, dtype=np.float64)
        records.append(
            VertexDual(
                vertex_id=vertex_id,
                vertex_key=(int(e), int(f)),
                loop_R_m=_readonly(loop_array[:, 0]),
                loop_Z_m=_readonly(loop_array[:, 1]),
                area_m2=area,
                R_area_integral_m3_per_rad=R_integral,
                centroid_R_m=centroid_R,
                centroid_Z_m=centroid_Z,
            )
        )
    return tuple(records)


def _add_array(digest: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    if array.dtype.kind == "f":
        encoded = array.astype(np.dtype("<f8"), copy=False)
    elif array.dtype.kind in "iu":
        encoded = array.astype(np.dtype("<i8"), copy=False)
    elif array.dtype.kind == "b":
        encoded = array.astype(np.uint8, copy=False)
    else:
        encoded = array
    digest.update(b"A")
    digest.update(name.encode("utf-8"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(str(encoded.dtype.str).encode("ascii"))
    digest.update(encoded.tobytes(order="C"))


def _add_value(digest: Any, value: Any) -> None:
    if isinstance(value, np.ndarray):
        _add_array(digest, "nested", value)
    elif isinstance(value, MappingProxyType):
        digest.update(
            json.dumps(dict(value), sort_keys=True).encode("utf-8")
        )
    else:
        digest.update(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=True,
            ).encode("utf-8")
        )


def _record_digest(digest: Any, record: Any) -> None:
    digest.update(type(record).__name__.encode("ascii"))
    for item in fields(record):
        digest.update(item.name.encode("ascii"))
        _add_value(digest, getattr(record, item.name))


def _fingerprint(data: Any) -> str:
    digest = hashlib.sha256()
    digest.update(data.kind.encode("ascii"))
    digest.update(json.dumps(dict(data.units), sort_keys=True).encode("utf-8"))
    if isinstance(data, BoundaryGeometry):
        for name in (
            "R_m",
            "Z_m",
            "active_cell_mask",
            "boundary_on_grid",
            "r_base_class",
            "r_class",
            "z_base_class",
            "z_class",
            "r_wall_distance_m",
            "z_wall_distance_m",
            "cell_ids",
            "polygon_R_m",
            "polygon_Z_m",
        ):
            _add_array(digest, name, getattr(data, name))
        _add_value(digest, list(data.cell_keys))
        _add_value(digest, list(data.r_face_keys))
        _add_value(digest, list(data.z_face_keys))
        _add_value(digest, list(data.vertex_keys))
        _add_value(
            digest,
            [
                data.tolerance_m,
                data.polygon_area_m2,
                data.dual_covered_area_m2,
                data.uncovered_area_m2,
                data.coverage_fraction,
            ],
        )
        for record in data.intercepts + data.axis_links + data.vertex_duals:
            _record_digest(digest, record)
        _record_digest(digest, data.wall_quadrature)
    else:
        for name in (
            "active_cell_mask",
            "cell_R_m",
            "cell_Z_m",
            "r_face_R_m",
            "r_face_Z_m",
            "z_face_R_m",
            "z_face_Z_m",
            "wall_face_ids",
            "wall_polyline_R_m",
            "wall_polyline_Z_m",
        ):
            _add_array(digest, name, getattr(data, name))
        _add_value(digest, list(data.wall_components))
        _add_value(digest, [list(item) for item in data.wall_vertex_ids])
    return digest.hexdigest()


def _staircase_identity(geometry: Any) -> StaircaseIdentity:
    R, Z, active = _geometry_arrays(geometry)
    try:
        walls = tuple(geometry.walls)
        wall_polyline = tuple(geometry.wall_polyline)
        cells = geometry.cells
        r_faces = geometry.r_faces
        z_faces = geometry.z_faces
    except AttributeError as error:
        raise BoundaryGeometryError("geometry is missing staircase topology") from error
    wall_R = np.array([point.R for point in wall_polyline], dtype=np.float64)
    wall_Z = np.array([point.Z for point in wall_polyline], dtype=np.float64)
    data = StaircaseIdentity(
        kind="staircase_identity_v1",
        units=_UNITS,
        fingerprint="",
        active_cell_mask=_readonly(active.reshape((R.size, Z.size), order="F")),
        cell_R_m=_readonly(cells.R),
        cell_Z_m=_readonly(cells.Z),
        r_face_R_m=_readonly(r_faces.R),
        r_face_Z_m=_readonly(r_faces.Z),
        z_face_R_m=_readonly(z_faces.R),
        z_face_Z_m=_readonly(z_faces.Z),
        wall_face_ids=_readonly(np.arange(len(walls)), np.int64),
        wall_components=tuple(str(wall.component) for wall in walls),
        wall_vertex_ids=tuple(
            (int(wall.vertices[0]), int(wall.vertices[1])) for wall in walls
        ),
        wall_polyline_R_m=_readonly(wall_R),
        wall_polyline_Z_m=_readonly(wall_Z),
    )
    return replace(data, fingerprint=_fingerprint(data))


def build_boundary_geometry(
    geometry: Any, kind: str = "clt_intercept"
) -> BoundaryGeometry | StaircaseIdentity:
    """Build immutable intercept geometry over an unchanged structured Geometry.

    Parameters
    ----------
    geometry:
        Existing ``fomhd.geometry.Geometry``. It is read but never modified.
    kind:
        ``"clt_intercept"`` for the convex true-intercept probe or
        ``"staircase"`` for an identity snapshot of the existing staircase.

    Returns
    -------
    BoundaryGeometry
        For ``clt_intercept``, with lengths in metres, areas in square metres,
        cylindrical integrals in cubic metres per radian, original dense
        indices, and a deterministic SHA-256 fingerprint.
    StaircaseIdentity
        For ``staircase``, with unchanged existing positions and identities.
    """
    if not isinstance(kind, str):
        raise BoundaryGeometryError("kind must be a string")
    if kind == "staircase":
        return _staircase_identity(geometry)
    if kind != "clt_intercept":
        raise BoundaryGeometryError(f"unsupported boundary geometry kind: {kind}")

    R, Z, active = _geometry_arrays(geometry)
    polygon, _ = _geometry_polygon(geometry)
    tolerance_m = _tolerance(R, Z, polygon)
    orientation, lengths = _validate_convex(polygon, tolerance_m)
    _, normals = _edge_data(polygon, lengths, orientation)
    quadrature = _quadrature(polygon, lengths, normals)
    boundary_on_grid = np.zeros((R.size, Z.size), dtype=bool)
    for j, radius_height in enumerate(Z):
        for i, radius in enumerate(R):
            boundary_on_grid[i, j] = bool(
                _boundary_sources(
                    float(radius),
                    float(radius_height),
                    polygon,
                    tolerance_m,
                )
            )
    intercepts, intercept_lookup, r_distance, z_distance = (
        _intercepts(
            geometry,
            R,
            Z,
            polygon,
            normals,
            tolerance_m,
            boundary_on_grid,
        )
    )
    r_base, r_class, z_base, z_class = _point_classes(
        R,
        Z,
        active,
        r_distance,
        z_distance,
        tolerance_m,
    )
    links = _axis_links(geometry, intercept_lookup)
    duals = _vertex_duals(
        geometry,
        R,
        Z,
        polygon,
        orientation,
        tolerance_m,
    )
    polygon_area = _polygon_area(polygon)
    covered_area = math.fsum(item.area_m2 for item in duals)
    area_tolerance = tolerance_m * max(
        1.0,
        float(np.ptp(polygon[:, 0])),
        float(np.ptp(polygon[:, 1])),
    )
    uncovered = polygon_area - covered_area
    if uncovered < -area_tolerance:
        raise BoundaryGeometryError("clipped vertex duals overlap unexpectedly")
    uncovered_area = 0.0 if abs(uncovered) <= area_tolerance else max(0.0, uncovered)
    coverage_fraction = min(1.0, covered_area / polygon_area)
    data = BoundaryGeometry(
        kind="clt_intercept_v1",
        units=_UNITS,
        fingerprint="",
        tolerance_m=tolerance_m,
        R_m=_readonly(R),
        Z_m=_readonly(Z),
        active_cell_mask=_readonly(active.reshape((R.size, Z.size), order="F")),
        boundary_on_grid=_readonly(boundary_on_grid),
        r_base_class=_readonly(r_base),
        r_class=_readonly(r_class),
        z_base_class=_readonly(z_base),
        z_class=_readonly(z_class),
        r_wall_distance_m=_readonly(r_distance),
        z_wall_distance_m=_readonly(z_distance),
        cell_ids=_readonly(
            [geometry.cells.find(i, j) for i, j in geometry.cells.keys],
            np.int64,
        ),
        cell_keys=tuple((int(i), int(j)) for i, j in geometry.cells.keys),
        r_face_keys=tuple(
            (int(e), int(j)) for e, j in geometry.r_faces.keys
        ),
        z_face_keys=tuple(
            (int(i), int(f)) for i, f in geometry.z_faces.keys
        ),
        vertex_keys=tuple(
            (int(e), int(f)) for e, f in geometry.vertices.keys
        ),
        polygon_R_m=_readonly(polygon[:, 0]),
        polygon_Z_m=_readonly(polygon[:, 1]),
        polygon_area_m2=polygon_area,
        dual_covered_area_m2=covered_area,
        uncovered_area_m2=uncovered_area,
        coverage_fraction=coverage_fraction,
        intercepts=intercepts,
        axis_links=links,
        vertex_duals=duals,
        wall_quadrature=quadrature,
    )
    return replace(data, fingerprint=_fingerprint(data))
