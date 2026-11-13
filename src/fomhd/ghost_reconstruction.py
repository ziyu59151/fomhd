"""Fixed two-ghost line transfers for the rectangular magnetic field."""

import numpy as np

from .types import Failure


_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(4)


def _line_spacing(nodes):
    if nodes.ndim != 1 or nodes.size < 3 or not np.all(np.isfinite(nodes)):
        raise Failure("Operators", "invalid rectangular magnetic reconstruction coordinates")
    spacing = (nodes[-1] - nodes[0]) / (nodes.size - 1)
    if not np.isfinite(spacing) or spacing <= 0 or np.any(np.diff(nodes) <= 0):
        raise Failure("Operators", "unordered rectangular magnetic reconstruction coordinates")
    return spacing


def _interval_basis(lower, upper, radius, slope, degree):
    midpoint = (lower + upper) / 2
    point = midpoint[:, None] + (upper - lower)[:, None] / 2 * _GAUSS_NODES
    measure = (radius + slope * point) * _GAUSS_WEIGHTS
    return np.einsum(
        "iq,iqd->id", measure, point[:, :, None] ** np.arange(degree + 1)
    ) / measure.sum(axis=1)[:, None]


def _wall_link_rows(lower, upper, radius, slope):
    ghost_lo = np.array([-2.5, -1.5])
    ghost_hi = ghost_lo + 1
    # The off-domain metric supports polynomial evaluation only.
    ghost_slope = -slope if radius + slope * ghost_lo[0] <= 0 else slope
    count = lower.size
    physical = _interval_basis(lower, upper, radius, slope, count - 1)
    ghost = _interval_basis(ghost_lo, ghost_hi, radius, ghost_slope, count - 1)
    extension = np.linalg.solve(physical.T, ghost.T).T
    degree = count + 1
    supported = np.vstack(
        (
            _interval_basis(ghost_lo, ghost_hi, radius, ghost_slope, degree),
            _interval_basis(lower, upper, radius, slope, degree),
        )
    )
    point = np.array([0.5, 0.0])[:, None] ** np.arange(degree + 1)
    row = np.linalg.solve(supported.T, point.T).T
    return row[:, 2:] + row[:, :2] @ extension


def ghost_link_reconstruction(
    nodes: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    cylindrical: bool,
    chart: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return line matrices for physical cell and wall point values in teslas."""
    spacing = _line_spacing(nodes)
    count = nodes.size
    if (
        lower.shape != nodes.shape
        or upper.shape != nodes.shape
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(upper <= lower)
        or (cylindrical and np.any(lower <= 0))
        or (
            chart is not None
            and (chart.shape != (count, count) or not np.all(np.isfinite(chart)))
        )
    ):
        raise Failure("Operators", "invalid rectangular magnetic reconstruction support")

    left = (lower - nodes[0]) / spacing
    right = (upper - nodes[0]) / spacing
    extent = (nodes[-1] - nodes[0]) / spacing
    origin, slope = (nodes[0], spacing) if cylindrical else (1.0, 0.0)
    upper_origin = nodes[-1] if cylindrical else 1.0
    width = min(4, count)
    centers = np.zeros((count - 1, count))
    walls = np.zeros((2, count))
    try:
        centers[0, :width], walls[0, :width] = _wall_link_rows(left[:width], right[:width], origin, slope)
        centers[-1, -width:], walls[1, -width:] = _wall_link_rows(
            extent - right[-width:][::-1],
            extent - left[-width:][::-1],
            upper_origin,
            -slope,
        )[:, ::-1]
        for row in range(1, count - 2):
            start = row - 1
            support = slice(start, start + 4)
            center = (nodes[row] + nodes[row + 1]) / 2
            location = (center - nodes[0]) / spacing
            metric = center if cylindrical else 1.0
            basis = _interval_basis(
                left[support] - location, right[support] - location, metric, slope, 3
            )
            centers[row, support] = np.linalg.solve(basis.T, (1.0, 0.0, 0.0, 0.0))
        if chart is not None:
            centers = np.linalg.solve(chart.T, centers.T).T
            walls = np.linalg.solve(chart.T, walls.T).T
    except np.linalg.LinAlgError as error:
        raise Failure("Operators", "rectangular magnetic reconstruction is singular") from error
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(walls)):
        raise Failure("Operators", "rectangular magnetic reconstruction is not finite")
    return centers, walls
