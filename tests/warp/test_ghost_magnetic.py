"""Independent polynomial checks for the rectangular ghost magnetic norm."""

import numpy as np
import pytest

from fomhd.ghost_magnetic import ghost_link_norm


def _line(first, cells, cylindrical):
    h = 1.0 / cells
    nodes = first + h * np.arange(cells + 1)
    lower = np.maximum(nodes[0], nodes - h / 2)
    upper = np.minimum(nodes[-1], nodes + h / 2)
    chart = np.eye(cells + 1)
    for left in range(1, cells - 1):
        edge = np.zeros(cells + 1)
        edge[left:left + 2] = [1, -1]
        chart += np.outer(edge, edge) / 24
    if cylindrical:
        mean_radius = (lower + upper) / 2
        chart = chart * mean_radius[None, :] / mean_radius[:, None]

    # Independent Gauss integration of the physical interval means.
    gauss, quadrature = np.polynomial.legendre.leggauss(3)
    radius = (upper + lower)[:, None] / 2 + (upper - lower)[:, None] * gauss / 2
    measure = (upper - lower)[:, None] * quadrature / 2
    if cylindrical:
        measure = measure * radius
    weights = measure.sum(axis=1)
    fields = np.stack([np.ones_like(radius), radius, radius**2])
    means = (fields * measure).sum(axis=2) / weights
    return h, nodes, lower, upper, chart, weights, means


@pytest.mark.parametrize("cylindrical", [False, True])
def test_ghost_magnetic_norm_preserves_constant_and_affine_moments(cylindrical):
    h, x, lo, hi, chart, weights, means = _line(1.0e-8, 2, cylindrical)
    norm = ghost_link_norm(x, lo, hi, cylindrical, chart)
    np.testing.assert_allclose(norm, norm.T, rtol=0, atol=2e-15)
    assert np.linalg.eigvalsh(norm)[0] > 0
    raw = chart.T @ norm @ chart
    np.testing.assert_allclose(raw @ means[0], weights, rtol=0, atol=2e-15)
    np.testing.assert_allclose(
        np.diff(raw @ means[1] / weights) / h, 1.0, rtol=0, atol=3e-14
    )


@pytest.mark.parametrize("cells", [2, 8])
def test_ghost_magnetic_norm_recovers_axial_quadratic_curl(cells):
    h, x, lo, hi, chart, weights, means = _line(-0.4, cells, False)
    norm = ghost_link_norm(x, lo, hi, False, chart)
    dual = chart.T @ norm @ chart @ means[2]
    np.testing.assert_allclose(
        np.diff(dual / weights) / h, x[:-1] + x[1:], rtol=0, atol=3e-14
    )


def test_ghost_magnetic_norm_has_second_order_radial_curl_at_both_walls():
    errors = []
    for count in (8, 16, 32):
        h, x, lo, hi, chart, weights, means = _line(1.0, count, True)
        norm = ghost_link_norm(x, lo, hi, True, chart)
        dual = chart.T @ norm @ chart @ means[2]
        error = np.abs(np.diff(dual / weights) / h - (x[:-1] + x[1:]))
        errors.append(error[[0, 1, -2, -1]])
    np.testing.assert_array_less(errors[1], 0.3 * errors[0])
    np.testing.assert_array_less(errors[2], 0.3 * errors[1])
