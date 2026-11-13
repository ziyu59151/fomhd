"""Static material line operators on a rectangular cell-center grid."""

import math

import numpy as np


def material_face_interpolation(count: int) -> np.ndarray:
    """Return the face-by-center interpolation with two affine ghost layers."""
    if count < 2:
        raise ValueError("material line requires at least two cells")

    extension = np.zeros((count + 4, count), dtype=np.float64)
    centers = np.arange(count)
    extension[centers + 2, centers] = 1.0
    extension[:2, :2] = ((3.0, -2.0), (2.0, -1.0))
    extension[-2:, -2:] = extension[:2, :2][::-1, ::-1]

    return (
        -extension[:-3]
        + 9.0 * extension[1:-2]
        + 9.0 * extension[2:-1]
        - extension[3:]
    ) / 16.0


def material_line_derivative(count: int, spacing: float) -> np.ndarray:
    """Return a center-by-center point derivative with a midpoint SBP norm."""
    if count < 2:
        raise ValueError("material line requires at least two cells")
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("material derivative requires finite positive spacing")

    # Collect the skew endpoint and three-cell corrections into their bands.
    nearest = np.full(count - 1, 3.0 / 4.0, dtype=np.float64)
    nearest[0] += 1.0 / 8.0
    nearest[-1] += 1.0 / 8.0
    next_nearest = np.full(count - 2, -1.0 / 8.0, dtype=np.float64)
    upper = np.diag(nearest, 1) + np.diag(next_nearest, 2)
    derivative = upper - upper.T

    # The same affine traces arise at the two end faces of the extension.
    lower_trace = np.array([1.5, -0.5], dtype=np.float64)
    boundary = np.outer(lower_trace, lower_trace) / 2.0
    derivative[:2, :2] -= boundary
    derivative[-2:, -2:] += boundary[::-1, ::-1]
    return derivative / spacing
