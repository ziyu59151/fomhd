"""Three-moment positive scalar quadrature on unchanged structured sites."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from .boundary_geometry import BoundaryGeometry


_UNITS = MappingProxyType(
    {
        "weights": "m^3/rad",
        "moments": "m^3/rad",
        "length": "m",
    }
)
_RESIDUAL_LIMIT = 1.0e-12
_METHOD = "minimum_relative_change_three_moment_equality_v1"


class BoundaryQuadratureError(ValueError):
    """Raised when the fixed three-moment construction cannot be applied."""


class BoundaryQuadratureInfeasible(BoundaryQuadratureError):
    """Raised with the complete failed candidate and its diagnostics."""

    def __init__(self, code: str, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.code = code
        self.diagnostics = diagnostics


@dataclass(frozen=True, slots=True)
class BoundaryVolumeWeights:
    """Read-only proposed scalar weights and construction diagnostics."""

    kind: str
    units: MappingProxyType
    fingerprint: str
    weights_m3_per_rad: np.ndarray
    old_weights_m3_per_rad: np.ndarray
    band_mask: np.ndarray
    band_cell_ids: np.ndarray
    core_cell_ids: np.ndarray
    changed_cell_ids: np.ndarray
    cell_keys: tuple[tuple[int, int], ...]
    geometry_fingerprint: str
    boundary_geometry_fingerprint: str
    R0_m: float
    Z0_m: float
    L_m: float
    band_cells: int
    band_radius_m: float
    polygon_moments_m3_per_rad: np.ndarray
    core_moments_m3_per_rad: np.ndarray
    target_moments_m3_per_rad: np.ndarray
    old_moments_m3_per_rad: np.ndarray
    new_moments_m3_per_rad: np.ndarray
    old_moment_residual_m3_per_rad: np.ndarray
    moment_residual_m3_per_rad: np.ndarray
    moment_scale_m3_per_rad: float
    old_scaled_moment_error: float
    scaled_moment_error: float
    constraint_rank: int
    constraint_condition: float
    positive: bool
    core_unchanged: bool
    min_weight_m3_per_rad: float
    max_weight_m3_per_rad: float
    min_band_weight_m3_per_rad: float
    weight_ratio: float


def _readonly(values: Any, dtype: Any | None = None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _array_digest(digest: Any, name: str, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    if array.dtype.kind == "f":
        array = array.astype(np.dtype("<f8"), copy=False)
    elif array.dtype.kind in "iu":
        array = array.astype(np.dtype("<i8"), copy=False)
    elif array.dtype.kind == "b":
        array = array.astype(np.uint8, copy=False)
    digest.update(name.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))


def _geometry_fingerprint(geometry: Any) -> str:
    digest = hashlib.sha256()
    for name in ("R", "Z", "active", "M0"):
        _array_digest(digest, name, np.asarray(getattr(geometry, name)))
    _array_digest(
        digest,
        "polygon",
        np.array([(point.R, point.Z) for point in geometry.polygon]),
    )
    return digest.hexdigest()


def _polygon_three_moments(
    polygon: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    points = np.asarray(polygon, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("polygon requires at least three RZ vertices")
    if not np.all(np.isfinite(points)):
        raise ValueError("polygon coordinates must be finite")
    R0 = 0.5 * (float(np.min(points[:, 0])) + float(np.max(points[:, 0])))
    Z0 = 0.5 * (float(np.min(points[:, 1])) + float(np.max(points[:, 1])))
    L = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])))
    if not L > 0.0:
        raise ValueError("polygon span must be positive")
    result = np.zeros(3, dtype=np.longdouble)
    for index in range(1, len(points) - 1):
        triangle = np.asarray(
            [points[0], points[index], points[index + 1]],
            dtype=np.longdouble,
        )
        first = triangle[1] - triangle[0]
        second = triangle[2] - triangle[0]
        area = abs(first[0] * second[1] - first[1] * second[0]) / 2.0
        samples = np.asarray(
            [
                (triangle[0] + triangle[1]) / 2.0,
                (triangle[1] + triangle[2]) / 2.0,
                (triangle[2] + triangle[0]) / 2.0,
            ],
            dtype=np.longdouble,
        )
        radii = samples[:, 0]
        xnorm = (radii - R0) / L
        znorm = (samples[:, 1] - Z0) / L
        result += np.sum(
            (area / 3.0)
            * np.column_stack((radii, radii * xnorm, radii * znorm)),
            axis=0,
        )
    return np.asarray(result, dtype=np.float64), R0, Z0, L


def _segment_distance(
    R: float, Z: float, start: np.ndarray, end: np.ndarray
) -> float:
    delta_R = float(end[0] - start[0])
    delta_Z = float(end[1] - start[1])
    denominator = delta_R * delta_R + delta_Z * delta_Z
    if denominator == 0.0:
        return math.hypot(R - start[0], Z - start[1])
    parameter = (
        (R - start[0]) * delta_R + (Z - start[1]) * delta_Z
    ) / denominator
    parameter = min(1.0, max(0.0, parameter))
    return math.hypot(
        R - (start[0] + parameter * delta_R),
        Z - (start[1] + parameter * delta_Z),
    )


def _boundary_distance(R: float, Z: float, polygon: np.ndarray) -> float:
    return min(
        _segment_distance(
            R, Z, polygon[index], polygon[(index + 1) % len(polygon)]
        )
        for index in range(len(polygon))
    )


def _validate_band_cells(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("band_cells must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError("band_cells must be a positive integer")
    return value


def _canonical_polygon(points: Any) -> np.ndarray:
    polygon = np.asarray(points, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("physical polygon requires at least three RZ vertices")
    if np.array_equal(polygon[0], polygon[-1]):
        polygon = polygon[:-1]
    if len(polygon) < 3 or np.array_equal(polygon[0], polygon[-1]):
        raise ValueError("physical polygon has invalid closing vertices")
    return polygon


def _validate_inputs(geometry: Any, boundary: Any) -> np.ndarray:
    if not isinstance(boundary, BoundaryGeometry):
        raise ValueError("boundary_geometry must be a BoundaryGeometry")
    keys = tuple((int(i), int(j)) for i, j in geometry.cells.keys)
    if keys != boundary.cell_keys:
        raise ValueError("boundary geometry does not match active cell keys")
    if not np.array_equal(np.asarray(geometry.R), boundary.R_m):
        raise ValueError("boundary geometry does not match R grid")
    if not np.array_equal(np.asarray(geometry.Z), boundary.Z_m):
        raise ValueError("boundary geometry does not match Z grid")
    geometry_polygon = _canonical_polygon(
        [(point.R, point.Z) for point in geometry.polygon]
    )
    boundary_polygon = _canonical_polygon(
        np.column_stack(
            (boundary.polygon_R_m, boundary.polygon_Z_m)
        )
    )
    if not np.array_equal(geometry_polygon, boundary_polygon):
        raise ValueError("boundary geometry physical polygon does not match")
    if not np.array_equal(
        np.asarray(boundary.cell_ids),
        np.arange(len(keys), dtype=np.int64),
    ):
        raise ValueError("boundary geometry does not match active cell ids")
    expected_active = np.asarray(geometry.active).reshape(
        (geometry.R.size, geometry.Z.size), order="F"
    )
    if not np.array_equal(boundary.active_cell_mask, expected_active):
        raise ValueError("boundary geometry does not match active mask")
    if len(geometry.M0) != len(keys):
        raise ValueError("Geometry M0 length does not match active cells")
    if len(boundary.intercepts) != 4 * len(keys):
        raise ValueError("boundary geometry does not match scalar intercept sites")
    for cell_id, (i, j) in enumerate(keys):
        expected_R = float(geometry.R[i])
        expected_Z = float(geometry.Z[j])
        if (
            float(geometry.cells.R[cell_id]) != expected_R
            or float(geometry.cells.Z[cell_id]) != expected_Z
        ):
            raise ValueError("Geometry scalar coordinates do not match grid keys")
        for intercept in boundary.intercepts[4 * cell_id : 4 * cell_id + 4]:
            if (
                intercept.cell_id != cell_id
                or intercept.cell_key != (i, j)
                or intercept.origin_R_m != expected_R
                or intercept.origin_Z_m != expected_Z
            ):
                raise ValueError(
                    "boundary intercept origins do not match scalar sites"
                )
    return boundary_polygon


def _failure(
    code: str,
    message: str,
    **diagnostics: Any,
) -> BoundaryQuadratureInfeasible:
    normalized = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in diagnostics.items()
    }
    return BoundaryQuadratureInfeasible(code, message, normalized)


def _fit(
    geometry: Any,
    boundary: BoundaryGeometry,
    band_cells: int,
    R0: float,
    Z0: float,
    L: float,
    physical_moments: np.ndarray,
    band_mask: np.ndarray,
    weights0: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    xnorm = (np.asarray(geometry.cells.R) - R0) / L
    znorm = (np.asarray(geometry.cells.Z) - Z0) / L
    matrix = np.column_stack((np.ones(len(xnorm)), xnorm, znorm))
    band_ids = np.flatnonzero(band_mask)
    core_ids = np.flatnonzero(~band_mask)
    old_moments = matrix.T @ weights0
    old_residual = old_moments - physical_moments
    scale = max(
        float(np.max(np.abs(physical_moments))),
        float(np.max(np.abs(old_moments))),
        np.finfo(np.float64).tiny,
    )
    old_error = float(np.max(np.abs(old_residual))) / scale
    core_moments = matrix[core_ids].T @ weights0[core_ids]
    target = physical_moments - core_moments
    weighted_band = matrix[band_ids] * weights0[band_ids, None]
    gram = weighted_band.T @ weighted_band
    row_scale = np.max(np.abs(gram), axis=1)
    if np.any(row_scale <= 0.0) or not np.all(np.isfinite(row_scale)):
        raise _failure(
            "rank_loss",
            "normalized constraint matrix has a zero or nonfinite row",
            band_cells=band_cells,
            band_cell_ids=band_ids,
            geometry_fingerprint=_geometry_fingerprint(geometry),
            boundary_geometry_fingerprint=boundary.fingerprint,
        )
    normalized = gram / row_scale[:, None]
    rank = int(np.linalg.matrix_rank(normalized))
    condition = float(np.linalg.cond(normalized))
    diagnostics = {
        "core_moments": core_moments,
        "target_moments": target,
        "old_moments": old_moments,
        "old_residual": old_residual,
        "moment_scale": scale,
        "constraint_rank": rank,
        "constraint_condition": condition,
    }
    if rank != 3:
        raise _failure(
            "rank_loss",
            "normalized three-moment constraint matrix lost rank",
            band_cells=band_cells,
            band_cell_ids=band_ids,
            **diagnostics,
        )
    if old_error <= _RESIDUAL_LIMIT:
        return weights0.copy(), {
            **diagnostics,
            "band_ids": band_ids,
            "core_ids": core_ids,
            "old_error": old_error,
        }
    rhs = target - matrix[band_ids].T @ weights0[band_ids]
    y = np.linalg.solve(normalized, rhs / row_scale)
    delta = (
        weights0[band_ids, None] ** 2 * matrix[band_ids]
    ) @ y
    candidate = weights0.copy()
    candidate[band_ids] += delta
    return candidate, {
        **diagnostics,
        "band_ids": band_ids,
        "core_ids": core_ids,
        "old_error": old_error,
    }


def _fingerprint(data: BoundaryVolumeWeights) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(data.units), sort_keys=True).encode("utf-8"))
    digest.update(_METHOD.encode("ascii"))
    digest.update(data.geometry_fingerprint.encode("ascii"))
    digest.update(data.boundary_geometry_fingerprint.encode("ascii"))
    for item in fields(data):
        value = getattr(data, item.name)
        if isinstance(value, np.ndarray):
            _array_digest(digest, item.name, value)
        elif not isinstance(value, (MappingProxyType, tuple)):
            digest.update(
                json.dumps(value, sort_keys=True).encode("utf-8")
            )
    digest.update(json.dumps(data.cell_keys).encode("utf-8"))
    return digest.hexdigest()


def build_boundary_volume_weights(
    geometry: Any,
    boundary_geometry: BoundaryGeometry,
    band_cells: int = 2,
) -> BoundaryVolumeWeights:
    """Construct the fixed positive three-moment boundary quadrature proposal."""

    band_cells = _validate_band_cells(band_cells)
    polygon = _validate_inputs(geometry, boundary_geometry)
    moments, R0, Z0, L = _polygon_three_moments(polygon)
    weights0 = np.array(geometry.M0, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(weights0)) or not np.all(weights0 > 0.0):
        raise ValueError("Geometry M0 weights must be finite and positive")
    radius = band_cells * max(
        float(geometry.R[1] - geometry.R[0]),
        float(geometry.Z[1] - geometry.Z[0]),
    )
    band_mask = np.array(
        [
            _boundary_distance(float(R), float(Z), polygon) <= radius
            for R, Z in zip(geometry.cells.R, geometry.cells.Z)
        ],
        dtype=bool,
    )
    if not np.any(band_mask):
        raise _failure(
            "empty_band",
            "fixed boundary band contains no scalar sites",
            band_cells=band_cells,
            band_cell_ids=np.empty(0, dtype=np.int64),
            geometry_fingerprint=_geometry_fingerprint(geometry),
            boundary_geometry_fingerprint=boundary_geometry.fingerprint,
        )
    candidate, fit = _fit(
        geometry,
        boundary_geometry,
        band_cells,
        R0,
        Z0,
        L,
        moments,
        band_mask,
        weights0,
    )
    band_ids = fit["band_ids"]
    core_ids = fit["core_ids"]
    if not np.all(np.isfinite(candidate[band_ids])) or not np.all(
        candidate[band_ids] > 0.0
    ):
        raise _failure(
            "nonpositive_weights",
            "minimum-change candidate contains nonpositive band weights",
            band_cells=band_cells,
            band_cell_ids=band_ids,
            candidate_weights=candidate,
            min_band_weight=float(np.min(candidate[band_ids])),
            **{
                key: fit[key]
                for key in (
                    "core_moments",
                    "target_moments",
                    "old_moments",
                    "old_residual",
                    "moment_scale",
                    "constraint_rank",
                    "constraint_condition",
                )
            },
            geometry_fingerprint=_geometry_fingerprint(geometry),
            boundary_geometry_fingerprint=boundary_geometry.fingerprint,
        )
    xnorm = (np.asarray(geometry.cells.R) - R0) / L
    znorm = (np.asarray(geometry.cells.Z) - Z0) / L
    matrix = np.column_stack((np.ones(len(xnorm)), xnorm, znorm))
    new_moments = matrix.T @ candidate
    residual = new_moments - moments
    scale = float(fit["moment_scale"])
    new_error = float(np.max(np.abs(residual))) / scale
    if new_error > _RESIDUAL_LIMIT:
        raise _failure(
            "moment_residual",
            "candidate exceeds the scaled three-moment residual limit",
            band_cells=band_cells,
            band_cell_ids=band_ids,
            candidate_weights=candidate,
            new_moments=new_moments,
            moment_residual=residual,
            scaled_moment_error=new_error,
            constraint_rank=fit["constraint_rank"],
            constraint_condition=fit["constraint_condition"],
            geometry_fingerprint=_geometry_fingerprint(geometry),
            boundary_geometry_fingerprint=boundary_geometry.fingerprint,
        )
    changed = np.flatnonzero(candidate != weights0)
    data = BoundaryVolumeWeights(
        kind="boundary_volume_3moment_v1",
        units=_UNITS,
        fingerprint="",
        weights_m3_per_rad=_readonly(candidate),
        old_weights_m3_per_rad=_readonly(weights0),
        band_mask=_readonly(band_mask),
        band_cell_ids=_readonly(band_ids, np.int64),
        core_cell_ids=_readonly(core_ids, np.int64),
        changed_cell_ids=_readonly(changed, np.int64),
        cell_keys=tuple(boundary_geometry.cell_keys),
        geometry_fingerprint=_geometry_fingerprint(geometry),
        boundary_geometry_fingerprint=boundary_geometry.fingerprint,
        R0_m=R0,
        Z0_m=Z0,
        L_m=L,
        band_cells=band_cells,
        band_radius_m=radius,
        polygon_moments_m3_per_rad=_readonly(moments),
        core_moments_m3_per_rad=_readonly(fit["core_moments"]),
        target_moments_m3_per_rad=_readonly(fit["target_moments"]),
        old_moments_m3_per_rad=_readonly(fit["old_moments"]),
        new_moments_m3_per_rad=_readonly(new_moments),
        old_moment_residual_m3_per_rad=_readonly(fit["old_residual"]),
        moment_residual_m3_per_rad=_readonly(residual),
        moment_scale_m3_per_rad=scale,
        old_scaled_moment_error=float(fit["old_error"]),
        scaled_moment_error=new_error,
        constraint_rank=int(fit["constraint_rank"]),
        constraint_condition=float(fit["constraint_condition"]),
        positive=True,
        core_unchanged=bool(
            np.array_equal(candidate[core_ids], weights0[core_ids])
        ),
        min_weight_m3_per_rad=float(np.min(candidate)),
        max_weight_m3_per_rad=float(np.max(candidate)),
        min_band_weight_m3_per_rad=float(np.min(candidate[band_ids])),
        weight_ratio=float(np.max(candidate) / np.min(candidate)),
    )
    return replace(data, fingerprint=_fingerprint(data))
