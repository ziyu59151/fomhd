"""Independent checks for the factored ghost toroidal field actions."""

import numpy as np

from fomhd.backend import Backend
from fomhd.ghost_toroidal import (
    build_ghost_toroidal,
    toroidal_reconstruct,
    toroidal_trace,
    toroidal_weight,
)


def _intervals(nodes):
    edges = np.r_[nodes[0], (nodes[:-1] + nodes[1:]) / 2, nodes[-1]]
    return edges[:-1], edges[1:]


def _graph(size):
    difference = np.diff(np.eye(size), axis=0)
    return np.eye(size) + difference.T @ difference / 24


def _box(nr=6, nz=5):
    r, z = np.linspace(1.1, 2.1, nr + 1), np.linspace(-0.4, 0.6, nz + 1)
    rlo, rhi = _intervals(r)
    zlo, zhi = _intervals(z)
    area = (zhi - zlo)[:, None] * (rhi - rlo)
    rbar = (rlo + rhi) / 2
    pr, pz = _graph(nr - 1), _graph(nz - 1)
    backend = Backend("cpu")
    factors = build_ghost_toroidal(r, z, pr, pz, backend)
    return backend, factors, r, z, rbar, area, pr, pz


def test_toroidal_reconstruction_recovers_physical_area_mean_and_direct_harmonic():
    backend, factors, r, z, rbar, area, pr, pz = _box()
    rlo, rhi = _intervals(r)
    zlo, zhi = _intervals(z)
    r2 = (rhi**2 + rhi * rlo + rlo**2) / 3
    z2 = (zhi**2 + zhi * zlo + zlo**2) / 3
    perturbation = r2[None, :] * z2[:, None]
    flux_average = np.sum(area * perturbation / rbar) / np.sum(area / rbar)
    raw = (perturbation - flux_average) / rbar
    stored = raw.copy()
    stored[1:-1, 1:-1] = pz @ raw[1:-1, 1:-1] @ pr.T
    stored += 0.7 / rbar
    rc = (r[:-1] + r[1:]) / 2
    zc = (z[:-1] + z[1:]) / 2
    expected = (zc[:, None] ** 2 * rc**2 + 0.7 - flux_average) / rc
    values = np.stack((stored.ravel(), 1j * stored.ravel()))
    out = toroidal_reconstruct(factors, values, backend)
    np.testing.assert_allclose(out[0], expected.ravel(), rtol=0, atol=4e-14)
    np.testing.assert_allclose(out[1], 1j * expected.ravel(), rtol=0, atol=4e-14)

    radial_wall, axial_wall = toroidal_trace(factors, stored.ravel(), backend)
    np.testing.assert_allclose(
        radial_wall, (zc[:, None]**2 * r[[0, -1]]**2 + 0.7 - flux_average) / r[[0, -1]],
        rtol=0, atol=4e-14,
    )
    np.testing.assert_allclose(
        axial_wall, (z[[0, -1], None]**2 * rc**2 + 0.7 - flux_average) / rc,
        rtol=0, atol=4e-14,
    )


def test_toroidal_norm_is_positive_hermitian_and_keeps_harmonic():
    backend, factors, _, _, rbar, area, _, _ = _box()
    harmonic = np.broadcast_to(1 / rbar, area.shape).reshape(-1).astype(np.complex128)
    weighted = toroidal_weight(factors, harmonic, backend)
    np.testing.assert_allclose(weighted, area.ravel(), rtol=0, atol=2e-15)

    norm = toroidal_weight(factors, np.eye(area.size, dtype=np.complex128), backend)
    np.testing.assert_allclose(norm, norm.T.conj(), rtol=0, atol=2e-15)
    assert np.linalg.eigvalsh(norm)[0] > 0


def test_toroidal_norm_radial_quadratic_converges_at_both_sides():
    errors = []
    for count in (8, 16, 32):
        backend, factors, r, _, rbar, area, pr, _ = _box(nr=count, nz=3)
        rlo, rhi = _intervals(r)
        raw = np.broadcast_to((rlo**2 + rlo * rhi + rhi**2) / (3 * rbar), area.shape).copy()
        # Remove its harmonic coefficient so every test value uses the raw
        # finite-volume convention before the chart transform.
        coefficient = np.sum(area * raw) / np.sum(area / rbar)
        raw -= coefficient / rbar
        stored = raw.copy()
        stored[1:-1, 1:-1] = raw[1:-1, 1:-1] @ pr.T
        weight = toroidal_weight(factors, stored.ravel(), backend).reshape(area.shape)
        # The adjoint chart follows the physical moment action. Undo it to
        # test the raw radial derivative separately from the electric chart.
        raw_load = weight.copy()
        raw_load[1:-1, 1:-1] = weight[1:-1, 1:-1] @ pr
        raw_load /= area
        derivative = np.diff(raw_load, axis=1) / (r[1] - r[0])
        error = np.abs(derivative - (r[1:] + r[:-1]))
        errors.append(error[0, [0, 1, -2, -1]])
    np.testing.assert_array_less(errors[1], 0.32 * errors[0])
    np.testing.assert_array_less(errors[2], 0.32 * errors[1])
