"""Small oracle for the two 1-D irregular CLT stencil rules.

The formulas are copied from
``tokRZ_mpi_pll_acc_merged_v20.f90`` lines 22074-22664 and
``tokRZ_mpi_pll_acc_merged.f90`` lines 22001-22591. This module
does not implement CLT time stepping, geometry, or a PDE solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
Side = Literal["upper", "lower"]
Rule = Literal["four_point", "five_point"]

# Source IR-to-boundary search: abs(node-boundary) < 1.0001*(1+0.5)*h.
SOURCE_ASSOCIATION_RATIO = 1.0001
SOURCE_ASSOCIATION_LIMIT_H = 1.5 * SOURCE_ASSOCIATION_RATIO


class PointKind(IntEnum):
    """Active source type codes; type 3 is commented out in the sources."""

    REGULAR = 1
    IRREGULAR = 2
    OUTSIDE_DROPPED = 4
    INSIDE_DROPPED = 5


class DroppedPointError(ValueError):
    """Raised when a directionally dropped point must not use a stencil."""


@dataclass(frozen=True)
class IrregularStencil:
    """Node coordinates and weights measured in grid-spacing units."""

    rule: Rule
    side: Side
    boundary_offset_h: float
    nodes_h: FloatArray
    first_derivative_weights: FloatArray
    second_derivative_weights: FloatArray


def classify_directional_point(
    distance_to_boundary_h: float,
    *,
    inside: bool,
) -> PointKind:
    """Apply the source's exact directional half-grid comparison.

    ``inside=False`` represents a point not lying between the two boundary
    crossings in the requested direction. Exact boundary equality maps to
    type 5 because the active source leaves type 3 assignments commented out.
    """

    distance = float(distance_to_boundary_h)
    if not np.isfinite(distance):
        raise ValueError("boundary distance must be finite")
    if not inside:
        return PointKind.OUTSIDE_DROPPED
    if distance < 0.0:
        raise ValueError("inside-point boundary distance must be nonnegative")
    if distance < 0.5:
        return PointKind.INSIDE_DROPPED
    return PointKind.REGULAR


def irregular_rule_for_offset(boundary_offset_h: float) -> Rule:
    """Select the idealized branch for a retained geometric offset.

    The exact ``1.5h`` threshold belongs to the five-point branch. The
    source uses ``1.0001 * 1.5h`` only as its boundary-association search
    tolerance; it does not create another coefficient branch.
    """

    offset = float(boundary_offset_h)
    if not np.isfinite(offset) or offset < 0.0:
        raise ValueError("boundary offset must be finite and nonnegative")
    if offset < 0.5:
        raise DroppedPointError(
            "point is directionally dropped (d < 0.5h); "
            "the irregular stencil is not used"
        )
    if offset < 1.5:
        return "four_point"
    if offset < 2.5:
        return "five_point"
    raise ValueError("boundary offset is outside the retained source range")


def irregular_stencil(
    rule: Rule,
    boundary_offset_h: float,
    *,
    side: Side = "upper",
) -> IrregularStencil:
    """Return the source coefficient weights for one irregular branch.

    The boundary offset is the positive distance from the stencil center to
    the fractional boundary location. ``four_point`` uses the source
    ``d1fbm/d1fbp`` branch for ``[0.5h, 1.5h)`` and ``five_point`` uses
    ``d1fc/d2fc`` for ``[1.5h, 2.5h)``.
    """

    offset = float(boundary_offset_h)
    if not np.isfinite(offset) or offset < 0.0:
        raise ValueError("boundary offset must be finite and nonnegative")
    if side not in ("upper", "lower"):
        raise ValueError(f"unsupported side: {side!r}")
    if rule not in ("four_point", "five_point"):
        raise ValueError(f"unsupported irregular rule: {rule!r}")
    try:
        selected_rule = irregular_rule_for_offset(offset)
    except DroppedPointError:
        raise
    except ValueError as exc:
        raise ValueError(f"{rule} branch requires a retained 0.5h-2.5h offset") from exc
    if rule != selected_rule:
        if rule == "four_point":
            raise ValueError("four-point branch requires 0.5h <= offset < 1.5h")
        raise ValueError("five-point branch requires 1.5h <= offset < 2.5h")

    if rule == "four_point":
        upper = _upper_four_point(offset)
    else:
        upper = _upper_five_point(offset)

    if side == "upper":
        return IrregularStencil(
            rule=rule,
            side=side,
            boundary_offset_h=offset,
            nodes_h=upper[0],
            first_derivative_weights=upper[1],
            second_derivative_weights=upper[2],
        )

    upper_nodes, upper_first, upper_second = upper
    return IrregularStencil(
        rule=rule,
        side=side,
        boundary_offset_h=offset,
        nodes_h=-upper_nodes[::-1].copy(),
        first_derivative_weights=-upper_first[::-1].copy(),
        second_derivative_weights=upper_second[::-1].copy(),
    )


def _upper_four_point(offset: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    if not 0.5 <= offset < 1.5:
        raise ValueError("four-point branch requires 0.5h <= offset < 1.5h")

    p = -offset + offset**3 / 4.0
    q = offset**2 + offset**3 / 2.0
    denominator = p * 0.5 - q * 0.75
    if denominator == 0.0:
        raise ValueError("singular four-point source coefficient denominator")

    first_coefficient = 0.5 / denominator
    second_coefficient = -q / denominator
    third_coefficient = (
        1.0 + first_coefficient * offset - second_coefficient
    ) / 2.0

    nodes = np.array([-2.0, -1.0, 0.0, offset], dtype=np.float64)
    first_weights = np.array(
        [
            -third_coefficient,
            -second_coefficient,
            first_coefficient + second_coefficient + third_coefficient,
            -first_coefficient,
        ],
        dtype=np.float64,
    )

    # This is the same 4x4 moment system passed to source gauss_solve.
    moment_matrix = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            nodes,
            nodes**2 / 2.0,
            nodes**3 / 6.0,
        ],
        dtype=np.float64,
    )
    second_weights = np.linalg.solve(
        moment_matrix,
        np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64),
    )
    return nodes, first_weights, second_weights


def _upper_five_point(offset: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    if not 1.5 <= offset < 2.5:
        raise ValueError("five-point branch requires 1.5h <= offset < 2.5h")

    xp1 = 1.0
    xm1 = 1.0
    xp2 = offset
    xm2 = 2.0

    f1 = xp1 + xp1**4 / xm2**3
    f2 = xp1**2 - xp1**4 / xm2**2
    f3 = xp1**3 + xp1**4 / xm2
    g1 = xm1 - xm1**4 / xm2**3
    g2 = -xm1**2 + xm1**4 / xm2**2
    g3 = xm1**3 - xm1**4 / xm2
    h1 = xp2 + xp2**4 / xm2**3
    h2 = xp2**2 - xp2**4 / xm2**2
    h3 = xp2**3 + xp2**4 / xm2

    denominator = (
        (f1 * h3 - f3 * h1) * (g2 * h3 - g3 * h2)
        - (g1 * h3 - g3 * h1) * (f2 * h3 - f3 * h2)
    )
    if denominator == 0.0:
        raise ValueError("singular five-point first-derivative denominator")

    a1 = h3 * (g2 * h3 - g3 * h2) / denominator
    b1 = h3 * (h2 * f3 - h3 * f2) / denominator
    c1 = h3 * (f2 * g3 - f3 * g2) / denominator
    d1 = (a1 - b1 + xp2**2 * c1) / xm2**2

    p1 = xp1**2 + xp1 * xm2
    p2 = xp1**3 - xp1 * xm2**2
    p3 = xp1**4 + xp1 * xm2**3
    q1 = -xm1**2 + xm1 * xm2
    q2 = xm1**3 - xm1 * xm2**2
    q3 = -xm1**4 + xm1 * xm2**3
    r1 = xp2**2 + xp2 * xm2
    r2 = xp2**3 - xp2 * xm2**2
    r3 = xp2**4 + xp2 * xm2**3
    denominator2 = (
        (p1 * r3 - p3 * r1) * (q2 * r3 - q3 * r2)
        - (q1 * r3 - q3 * r1) * (p2 * r3 - p3 * r2)
    )
    if denominator2 == 0.0:
        raise ValueError("singular five-point second-derivative denominator")

    a2 = 2.0 * r3 * (q2 * r3 - q3 * r2) / denominator2
    b2 = 2.0 * r3 * (r2 * p3 - r3 * p2) / denominator2
    c2 = 2.0 * r3 * (p2 * q3 - p3 * q2) / denominator2
    d2 = -(xp1 * a2 + xm1 * b2 + xp2 * c2) / xm2

    nodes = np.array([-2.0, -1.0, 0.0, 1.0, offset], dtype=np.float64)
    first_weights = np.array(
        [-d1, -b1, -a1 + b1 - c1 + d1, a1, c1],
        dtype=np.float64,
    )
    second_weights = np.array(
        [-d2, -b2, -a2 + b2 - c2 + d2, a2, c2],
        dtype=np.float64,
    )
    return nodes, first_weights, second_weights
