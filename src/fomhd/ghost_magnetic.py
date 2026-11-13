"""Ghost-backed poloidal magnetic norms for aligned rectangular walls."""

import numpy as np

from .types import Failure


def _moments(lower, upper, metric_origin, metric_slope):
    """Return exact integral weights and normalized means through degree two."""
    width = upper - lower
    center = (lower + upper) / 2
    metric = metric_origin + metric_slope * center
    variance = width**2 / 12
    offset = metric_slope * variance / metric
    means = np.stack(
        (np.ones_like(center), center + offset, center**2 + variance + 2 * center * offset),
        axis=1,
    )
    return width * metric, means


def _ghost_curvature(lower, upper, metric_origin, metric_slope):
    """Pull a curvature stencil on two ghost links back to three true links."""
    _, physical = _moments(lower, upper, metric_origin, metric_slope)
    ghost_lower = np.array([-1.5, -2.5])
    ghost_upper = np.array([-0.5, -1.5])
    if metric_origin - 2.5 * metric_slope <= 0:
        metric_slope = -metric_slope
    _, ghosts = _moments(ghost_lower, ghost_upper, metric_origin, metric_slope)
    extrapolation = np.linalg.solve(physical.T, ghosts.T).T
    gap_wall = physical[0, 1] - ghosts[0, 1]
    gap_outer = ghosts[0, 1] - ghosts[1, 1]
    if gap_wall <= 0 or gap_outer <= 0:
        raise Failure("Operators", "rectangular magnetic ghost centroids are unordered")
    wall_link = np.array([1.0, 0.0, 0.0])
    curvature = (wall_link - extrapolation[0]) / gap_wall
    curvature -= (extrapolation[0] - extrapolation[1]) / gap_outer
    return curvature, physical[:, 2]


def ghost_link_norm(
    nodes: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    cylindrical: bool,
    chart: np.ndarray,
    *,
    inverse_radius_weight: bool = False,
) -> np.ndarray:
    """Build an FP64 stored-chart norm from physical links and two ghost layers."""
    count = nodes.size
    if (
        nodes.shape != (count,)
        or count < 3
        or lower.shape != nodes.shape
        or upper.shape != nodes.shape
        or chart.shape != (count, count)
        or not np.all(np.isfinite(nodes))
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or not np.all(np.isfinite(chart))
        or np.any(upper <= lower)
        or np.any(np.diff(nodes) <= 0)
        or (cylindrical and inverse_radius_weight)
        or ((cylindrical or inverse_radius_weight) and np.any(lower <= 0))
    ):
        raise Failure("Operators", "invalid rectangular magnetic link support")

    h = float((nodes[-1] - nodes[0]) / (count - 1))
    x = (nodes - nodes[0]) / h
    left = (lower - nodes[0]) / h
    right = (upper - nodes[0]) / h
    origin, slope = (nodes[0], h) if cylindrical else (1.0, 0.0)
    weight, physical = _moments(left, right, origin, slope)
    weight *= h
    if inverse_radius_weight:
        weight /= (lower + upper) / 2
    centers = physical[:, 1]
    offset = centers - x
    offset -= np.average(offset, weights=weight)
    gap = np.diff(centers)
    flux = np.cumsum(weight * offset)[:-1]
    if np.any(gap <= 0) or np.any(flux <= 0):
        raise Failure("Operators", "rectangular magnetic norm has invalid link moments")
    edge = np.arange(count - 1)
    coupling = flux / gap
    norm = np.diag(weight)
    norm[edge, edge] += coupling
    norm[edge + 1, edge + 1] += coupling
    norm[edge, edge + 1] -= coupling
    norm[edge + 1, edge] -= coupling
    inverse_gap = 1 / gap
    curvature = np.stack(
        (inverse_gap[:-1], -inverse_gap[:-1] - inverse_gap[1:], inverse_gap[1:]),
        axis=1,
    )

    try:
        lower_row, lower_quadratic = _ghost_curvature(left[:3], right[:3], origin, slope)
        upper_row, upper_quadratic = _ghost_curvature(
            x[-1] - right[-1:-4:-1],
            x[-1] - left[-1:-4:-1],
            nodes[-1] if cylindrical else 1.0,
            -slope,
        )
        upper_row, upper_quadratic = upper_row[::-1], upper_quadratic[::-1]
        lower_row *= (curvature[0] @ lower_quadratic) / (lower_row @ lower_quadratic)
        upper_row *= (curvature[-1] @ upper_quadratic) / (upper_row @ upper_quadratic)
        if count == 3:
            curvature[0] = (lower_row + upper_row) / 2
        else:
            curvature[0] = lower_row
            curvature[-1] = upper_row
        strength = np.full(count - 2, h / 64)
        if cylindrical:
            strength *= nodes[1:-1]
        if inverse_radius_weight:
            strength /= nodes[1:-1]
        strength[[0, -1]] *= 3 / 2 if count == 3 else 6 / 5
        stencil = np.arange(count - 2)
        for row in range(3):
            for column in range(3):
                norm[stencil + row, stencil + column] += (
                    strength * curvature[:, row] * curvature[:, column]
                )
        left_solve = np.linalg.solve(chart.T, norm)
        stored = np.linalg.solve(chart.T, left_solve.T).T
    except np.linalg.LinAlgError as error:
        raise Failure("Operators", "rectangular magnetic ghost closure is singular") from error
    if not np.all(np.isfinite(stored)):
        raise Failure("Operators", "rectangular magnetic ghost norm is not finite")
    return 0.5 * (stored + stored.T)
