from __future__ import annotations

import numpy as np
import pytest

from tools.clt_boundary_coefficients import (  # noqa: E402
    SOURCE_ASSOCIATION_LIMIT_H,
    SOURCE_ASSOCIATION_RATIO,
    DroppedPointError,
    PointKind,
    classify_directional_point,
    irregular_stencil,
    irregular_rule_for_offset,
)


FOUR_POINT_OFFSETS = (0.5, 0.51, 0.75, 1.25, 1.49)
FIVE_POINT_OFFSETS = (1.5, 1.75, 2.25, 2.49)


@pytest.mark.parametrize(
    ("distance_h", "expected"),
    [
        (0.25, PointKind.INSIDE_DROPPED),
        (0.49, PointKind.INSIDE_DROPPED),
        (0.5, PointKind.REGULAR),
        (0.51, PointKind.REGULAR),
        (0.75, PointKind.REGULAR),
    ],
)
def test_directional_half_grid_comparison_is_exact(
    distance_h: float,
    expected: PointKind,
) -> None:
    assert classify_directional_point(distance_h, inside=True) is expected


def test_boundary_equality_is_inside_dropped_and_outside_is_separate() -> None:
    assert classify_directional_point(0.0, inside=True) is PointKind.INSIDE_DROPPED
    assert classify_directional_point(0.5, inside=False) is PointKind.OUTSIDE_DROPPED


def test_near_tangent_short_distance_changes_kind_at_strict_threshold() -> None:
    below = np.nextafter(0.5, 0.0)
    above = np.nextafter(0.5, 1.0)
    assert classify_directional_point(below, inside=True) is PointKind.INSIDE_DROPPED
    assert classify_directional_point(0.5, inside=True) is PointKind.REGULAR
    assert classify_directional_point(above, inside=True) is PointKind.REGULAR


def test_source_association_tolerance_is_reported_separately() -> None:
    assert SOURCE_ASSOCIATION_RATIO == pytest.approx(1.0001)
    assert SOURCE_ASSOCIATION_LIMIT_H == pytest.approx(1.50015)


@pytest.mark.parametrize(
    ("offset_h", "expected_rule"),
    [
        (0.5, "four_point"),
        (1.25, "four_point"),
        (np.nextafter(1.5, 0.0), "four_point"),
        (1.5, "five_point"),
        (2.25, "five_point"),
    ],
)
def test_geometric_offset_selects_rule_with_far_side_at_exact_threshold(
    offset_h: float,
    expected_rule: str,
) -> None:
    assert irregular_rule_for_offset(offset_h) == expected_rule


def test_quarter_grid_drop_leaves_source_stencil_offsets() -> None:
    closest_offset_h = 0.25
    nearest_retained_h = 1.25
    next_retained_h = 2.25

    assert (
        classify_directional_point(closest_offset_h, inside=True)
        is PointKind.INSIDE_DROPPED
    )
    assert (
        classify_directional_point(nearest_retained_h, inside=True)
        is PointKind.REGULAR
    )
    for side in ("upper", "lower"):
        four_point = irregular_stencil(
            "four_point",
            nearest_retained_h,
            side=side,
        )
        five_point = irregular_stencil(
            "five_point",
            next_retained_h,
            side=side,
        )
        assert four_point.boundary_offset_h == nearest_retained_h
        assert five_point.boundary_offset_h == next_retained_h


@pytest.mark.parametrize("offset_h", FOUR_POINT_OFFSETS)
@pytest.mark.parametrize("side", ("upper", "lower"))
@pytest.mark.parametrize("power", range(4))
def test_four_node_first_derivative_moments_through_cubic(
    offset_h: float,
    side: str,
    power: int,
) -> None:
    stencil = irregular_stencil(
        "four_point",
        offset_h,
        side=side,
    )
    moment = np.dot(stencil.first_derivative_weights, stencil.nodes_h**power)
    expected = 1.0 if power == 1 else 0.0
    assert moment == pytest.approx(expected, abs=2e-11)


@pytest.mark.parametrize("offset_h", FOUR_POINT_OFFSETS)
@pytest.mark.parametrize("side", ("upper", "lower"))
def test_four_node_first_derivative_fourth_moment_sets_third_order(
    offset_h: float,
    side: str,
) -> None:
    stencil = irregular_stencil(
        "four_point",
        offset_h,
        side=side,
    )
    fourth_moment = np.dot(
        stencil.first_derivative_weights,
        stencil.nodes_h**4,
    )
    expected = 2.0 * offset_h * (1.0 if side == "upper" else -1.0)
    assert fourth_moment == pytest.approx(expected, abs=2e-11)


@pytest.mark.parametrize("offset_h", FIVE_POINT_OFFSETS)
@pytest.mark.parametrize("side", ("upper", "lower"))
@pytest.mark.parametrize("power", range(5))
def test_five_node_first_derivative_moments_through_quartic(
    offset_h: float,
    side: str,
    power: int,
) -> None:
    stencil = irregular_stencil(
        "five_point",
        offset_h,
        side=side,
    )
    moment = np.dot(stencil.first_derivative_weights, stencil.nodes_h**power)
    expected = 1.0 if power == 1 else 0.0
    assert moment == pytest.approx(expected, abs=4e-11)


@pytest.mark.parametrize("offset_h", FIVE_POINT_OFFSETS)
@pytest.mark.parametrize("side", ("upper", "lower"))
def test_five_node_first_derivative_fifth_moment_sets_fourth_order(
    offset_h: float,
    side: str,
) -> None:
    stencil = irregular_stencil(
        "five_point",
        offset_h,
        side=side,
    )
    fifth_moment = np.dot(
        stencil.first_derivative_weights,
        stencil.nodes_h**5,
    )
    expected = -2.0 * offset_h
    assert fifth_moment == pytest.approx(expected, abs=8e-11)


@pytest.mark.parametrize("offset_h", FOUR_POINT_OFFSETS)
@pytest.mark.parametrize("side", ("upper", "lower"))
@pytest.mark.parametrize("power", range(4))
def test_four_node_second_derivative_moments_through_cubic(
    offset_h: float,
    side: str,
    power: int,
) -> None:
    stencil = irregular_stencil(
        "four_point",
        offset_h,
        side=side,
    )
    moment = np.dot(stencil.second_derivative_weights, stencil.nodes_h**power)
    expected = 2.0 if power == 2 else 0.0
    assert moment == pytest.approx(expected, abs=8e-11)


@pytest.mark.parametrize("offset_h", FOUR_POINT_OFFSETS)
@pytest.mark.parametrize("side", ("upper", "lower"))
def test_four_node_second_derivative_fourth_moment_sets_second_order(
    offset_h: float,
    side: str,
) -> None:
    stencil = irregular_stencil(
        "four_point",
        offset_h,
        side=side,
    )
    fourth_moment = np.dot(
        stencil.second_derivative_weights,
        stencil.nodes_h**4,
    )
    assert fourth_moment == pytest.approx(2.0 * (3.0 * offset_h - 2.0))
    assert np.dot(
        irregular_stencil("four_point", 0.5).second_derivative_weights,
        irregular_stencil("four_point", 0.5).nodes_h**4,
    ) == pytest.approx(-1.0)


@pytest.mark.parametrize("offset_h", FIVE_POINT_OFFSETS)
@pytest.mark.parametrize("side", ("upper", "lower"))
@pytest.mark.parametrize("power", range(5))
def test_five_node_second_derivative_moments_through_quartic(
    offset_h: float,
    side: str,
    power: int,
) -> None:
    stencil = irregular_stencil(
        "five_point",
        offset_h,
        side=side,
    )
    moment = np.dot(stencil.second_derivative_weights, stencil.nodes_h**power)
    expected = 2.0 if power == 2 else 0.0
    assert moment == pytest.approx(expected, abs=8e-11)


@pytest.mark.parametrize("offset_h", FIVE_POINT_OFFSETS)
@pytest.mark.parametrize("side", ("upper", "lower"))
def test_five_node_second_derivative_fifth_moment_sets_third_order(
    offset_h: float,
    side: str,
) -> None:
    stencil = irregular_stencil(
        "five_point",
        offset_h,
        side=side,
    )
    fifth_moment = np.dot(
        stencil.second_derivative_weights,
        stencil.nodes_h**5,
    )
    upper_moment = 4.0 - 2.0 * offset_h
    expected = upper_moment if side == "upper" else -upper_moment
    assert fifth_moment == pytest.approx(expected, abs=8e-11)


@pytest.mark.parametrize(
    ("rule", "offsets"),
    [
        ("four_point", (1.5, 1.51)),
        ("five_point", (1.49, 2.5)),
    ],
)
def test_irregular_rule_domains_reject_other_geometric_branch(
    rule: str,
    offsets: tuple[float, ...],
) -> None:
    for offset_h in offsets:
        with pytest.raises(ValueError, match="branch requires"):
            irregular_stencil(rule, offset_h)


@pytest.mark.parametrize("distance_h", (0.0, 0.25, 0.49))
def test_directionally_dropped_point_has_no_used_irregular_stencil(
    distance_h: float,
) -> None:
    with pytest.raises(DroppedPointError, match="directionally dropped"):
        irregular_stencil("four_point", distance_h, side="upper")


@pytest.mark.parametrize("offset_h", (0.5, np.nextafter(0.5, 1.0)))
def test_four_node_rule_is_finite_at_short_retained_distance(
    offset_h: float,
) -> None:
    stencil = irregular_stencil("four_point", offset_h, side="upper")
    assert np.isfinite(stencil.first_derivative_weights).all()
    assert np.isfinite(stencil.second_derivative_weights).all()
    assert np.all(np.diff(stencil.nodes_h) > 0.0)
