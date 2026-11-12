"""Independent physical sampling of the fixed rectangular magnetic flux."""

import numpy as np
import pytest

from fomhd.geometry import build_aligned_box, build_geometry
from fomhd.operators import apply_t2, build_operators
from fomhd.types import Context


def _sampled_box(nr, nz, modes=0):
    r = 1.0 + (np.arange(nr) + 0.5) * (1.2 / nr)
    z = -0.5 + (np.arange(nz) + 0.5) / nz
    g = build_aligned_box(r, z, [(1.4, -0.15), (1.8, -0.15), (1.6, 0.2)])
    return g, build_operators(g, modes)


def _sample_axisymmetric_flux(g, op, psi, toroidal_flux=1.3):
    """Use the same integrated flux sampling as the FreeGS input mapping."""
    hr, hz = g.R[1] - g.R[0], g.Z[1] - g.Z[0]
    r, z = g.z_faces.R, g.z_faces.Z
    za = np.maximum(g.Z[0] - hz / 2, z - hz / 2)
    zb = np.minimum(g.Z[-1] + hz / 2, z + hz / 2)
    br = -(psi(r, zb) - psi(r, za)) / (r * g.lz)
    r, z = g.r_faces.R, g.r_faces.Z
    ra = np.maximum(g.R[0] - hr / 2, r - hr / 2)
    rb = np.minimum(g.R[-1] + hr / 2, r + hr / 2)
    bz = (psi(rb, z) - psi(ra, z)) / (g.Rbar_r * g.lr)
    harmonic = toroidal_flux / g.Rbar_v
    stored = apply_t2(op, np.r_[br, harmonic, bz].astype(np.complex128))
    stored[br.size : br.size + harmonic.size] = harmonic
    caps = np.zeros((2 * op.K + 1, len(g.walls), 2), dtype=np.complex128)
    for index, wall in enumerate(g.walls):
        r = g.vertices.R[list(wall.vertices)]
        z = g.vertices.Z[list(wall.vertices)]
        sampled = psi(
            np.array([r[0], r.mean(), r[1]]), np.array([z[0], z.mean(), z[1]])
        )
        sign = -wall.sign if wall.component == "r_face" else wall.sign
        caps[op.K, index] = sign * np.diff(sampled)
    return stored, caps


@pytest.mark.parametrize("direction", ["radial", "axial"])
def test_box_ampere_recovers_affine_link_means(direction):
    from fomhd.box_boundary import install_normal_flux, magnetic_divergence

    g, op = _sampled_box(7, 6)
    if direction == "radial":
        # B_Z = 0.4 + 0.7 R; B_R = 0.
        psi = lambda r, z: 0.2 * r**2 + (0.7 / 3) * r**3
        expected = np.full(op.Nc, -0.7)
    else:
        # B_R = R (0.4 + 0.7 Z); B_Z is independent of R.
        psi = lambda r, z: -r**2 * (0.4 * z + 0.35 * z**2)
        expected = 0.7 * g.cells.R
    b, caps = _sample_axisymmetric_flux(g, op, psi)
    install_normal_flux(op, g, caps)
    context = Context(None, g, op, op.compute, 1)
    np.testing.assert_allclose(magnetic_divergence(b, context), 0.0, atol=8e-13)
    nr, nz = len(g.r_faces.keys), len(g.z_faces.keys)
    exact = np.r_[np.zeros(nr), expected, np.zeros(nz)]
    np.testing.assert_allclose(op.Cdag[0] @ b, exact, rtol=0, atol=2e-12)


@pytest.mark.parametrize("case", ["mixed_vacuum", "solovev"])
def test_box_ampere_converges_at_edges_and_corners(case):
    from fomhd.box_boundary import install_normal_flux, magnetic_divergence

    if case == "mixed_vacuum":
        psi = lambda r, z: 0.1 * (r**4 - 4 * r**2 * z**2)
        exact_curl = lambda r: 0 * r
    else:
        psi = lambda r, z: -0.1 * ((r**2 - 1.6**2) ** 2 + 4 * r**2 * z**2)
        exact_curl = lambda r: 1.6 * r

    errors = []
    for nr, nz in [(7, 6), (14, 12)]:
        g, op = _sampled_box(nr, nz)
        b, caps = _sample_axisymmetric_flux(g, op, psi)
        install_normal_flux(op, g, caps)
        context = Context(None, g, op, op.compute, 1)
        np.testing.assert_allclose(magnetic_divergence(b, context), 0.0, atol=8e-13)
        nr_faces = len(g.r_faces.keys)
        curl = (op.Cdag[0] @ b)[nr_faces : nr_faces + op.Nc]
        error = np.abs(curl - exact_curl(g.cells.R))
        i, j = np.asarray(g.cells.keys).T
        radial = np.minimum(i, nr - 1 - i)
        axial = np.minimum(j, nz - 1 - j)
        layer = np.minimum(radial, axial)
        corners = (radial == 0) & (axial == 0)
        regions = [(layer == 0) & ~corners, layer == 1, corners, layer >= 2]
        errors.append(np.array([np.max(error[region]) for region in regions]))
    np.testing.assert_array_less(errors[1], errors[0] * [0.6, 0.6, 0.6, 0.3])


def test_box_ampere_and_faraday_pair_and_preserve_modal_constraints():
    from fomhd.box_boundary import install_normal_flux, magnetic_divergence
    from fomhd.operators import magnetic_weight

    g, op = _sampled_box(7, 6, modes=1)
    background, caps = _sample_axisymmetric_flux(g, op, lambda r, z: 0.35 * r**2)
    install_normal_flux(op, g, caps)
    context = Context(None, g, op, op.compute, 3)
    rng = np.random.default_rng(12)
    m1 = rng.normal(size=op.Ne) + 1j * rng.normal(size=op.Ne)
    e = np.stack((m1.conj(), rng.normal(size=op.Ne), m1))
    rate = np.stack([-curl @ field for curl, field in zip(op.C, e, strict=True)])
    b = 0.01 * rate
    b[op.K] += background
    np.testing.assert_allclose(magnetic_divergence(b.ravel(), context), 0.0, atol=8e-13)
    j = np.stack([curl @ field for curl, field in zip(op.Cdag, b, strict=True)])
    continuity = np.stack([dj @ current for dj, current in zip(op.DJ, j, strict=True)])
    np.testing.assert_allclose(continuity, 0.0, atol=8e-13)
    magnetic_power = np.vdot(rate, magnetic_weight(op, b.ravel()))
    electric_power = np.vdot(e, op.M1 * j)
    np.testing.assert_allclose(magnetic_power, -electric_power, atol=1e-12)
    assert np.vdot(b, magnetic_weight(op, b.ravel())).real > 0


def test_box_faraday_matches_a_sampled_flux_rate_with_fixed_caps():
    g, op = _sampled_box(7, 6)
    r0, r1 = g.R[0] - g.lr.min(), g.R[-1] + g.lr.min()
    z0, z1 = g.Z[0] - g.lz.min(), g.Z[-1] + g.lz.min()

    def flux_rate(r, z):
        return (r - r0) * (r1 - r) * (z - z0) * (z1 - z)

    # R E_phi = -partial_t psi. Tangential E vanishes on all four walls.
    expected, cap_rate = _sample_axisymmetric_flux(g, op, flux_rate, 0.0)
    e = np.zeros(op.Ne, dtype=np.complex128)
    start = len(g.r_faces.keys)
    e[start : start + op.Nc] = -flux_rate(g.cells.R, g.cells.Z) / g.cells.R
    np.testing.assert_allclose(-op.C[0] @ e, expected, atol=3e-14)
    np.testing.assert_allclose(cap_rate, 0.0, atol=3e-14)


def test_threaded_rectangle_divergence_and_faraday():
    from fomhd.box_boundary import install_normal_flux, magnetic_divergence

    # Axisymmetric constant axial field. The x-directed dual faces carry B_Z;
    # exactly one cap half-edge belongs to each end vertex of each cap segment.
    r = 1.1 + np.arange(5) * 0.2
    z = -0.3 + np.arange(4) * 0.2
    lo, hi = r[0] - 0.1, r[-1] + 0.1
    bottom, top = z[0] - 0.1, z[-1] + 0.1
    g = build_geometry(r, z, [(lo, bottom), (hi, bottom), (hi, top), (lo, top)])
    op = build_operators(g, 0)
    backend = op.compute
    b = np.zeros(op.Nb, dtype=np.complex128)
    b[-len(g.r_faces.keys):] = 0.7
    b = apply_t2(op, b)
    flux = np.zeros((1, len(g.walls), 2), dtype=np.complex128)
    for index, face in enumerate(g.walls):
        if face.component == "z_face":
            rs = g.vertices.R[list(face.vertices)]
            bounds = np.array([rs[0], rs.mean(), rs[1]])
            flux[0, index, :] = face.sign * 0.7 * np.diff(bounds**2) / 2
    assert np.max(abs(op.DB[0] @ b)) > 1.0
    install_normal_flux(op, g, flux)
    c = Context(None, g, op, backend, 1)
    np.testing.assert_allclose(magnetic_divergence(b, c), 0.0, atol=6e-14)
    e = np.random.default_rng(4).normal(size=op.Ne).astype(np.complex128)
    np.testing.assert_allclose(
        magnetic_divergence(b - 0.01 * (op.C[0] @ e), c), 0.0, atol=6e-14
    )
    with pytest.raises(ValueError, match="immutable"):
        install_normal_flux(op, g, flux)


def test_cap_data_require_modal_complex128():
    from fomhd.box_boundary import install_normal_flux

    r = np.arange(4) * 0.2 + 1.1
    z = np.arange(4) * 0.2 + 0.1
    g = build_geometry(r, z, [(1, 0), (1.8, 0), (1.8, 0.8), (1, 0.8)])
    op = build_operators(g, 1)
    flux = np.zeros((3, len(g.walls), 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="complex128"):
        install_normal_flux(op, g, flux.real)
    flux[0, 0, 0] = 1
    with pytest.raises(ValueError, match="Hermitian"):
        install_normal_flux(op, g, flux)


def test_fixed_box_electric_correction_reuses_existing_checked_system():
    from types import SimpleNamespace

    from fomhd.geometry import Point2
    from fomhd.massless import _wall_correction

    r = np.arange(4)*0.2 + 1.1
    z = np.arange(4)*0.2 + 0.1
    g = build_geometry(r, z, [(1,0),(1.8,0),(1.8,0.8),(1,0.8)])
    op = build_operators(g, 0)
    ref = SimpleNamespace(length=0.8, electric=1.0)
    p = SimpleNamespace(K=0, rank_rtol=1e-12, ref=ref, memory_limit=512<<20)
    c = Context(p, g, op, op.compute, 1)
    field = np.random.default_rng(3).normal(size=op.Ne).astype(np.complex128)
    expected, direct = _wall_correction(field, c)
    g.lcfs_polygon = [Point2(1.2, 0.2), Point2(1.5, 0.3), Point2(1.3, 0.6)]
    actual, first = _wall_correction(field, c)
    again, second = _wall_correction(0.9*field, c)
    assert first.cache is second.cache
    assert first.cache.setup_count == 1
    assert direct.rank == first.rank
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(again, 0.9*expected, rtol=1e-12, atol=1e-13)


def test_threaded_pressure_load_pairs_with_physical_wall_work():
    from types import SimpleNamespace

    from fomhd.box_boundary import electron_pressure_traction, install_normal_flux

    r = 1.1 + np.arange(4) * 0.2
    z = 0.1 + np.arange(4) * 0.2
    g = build_geometry(r, z, [(1, 0), (1.8, 0), (1.8, 0.8), (1, 0.8)])
    op = build_operators(g, 0)
    flux = np.zeros((1, len(g.walls), 2), dtype=np.complex128)
    for w, face in enumerate(g.walls):
        if face.component == "z_face":
            flux[0, w, :] = face.sign * face.area_per_radian / 2
    install_normal_flux(op, g, flux)
    context = Context(SimpleNamespace(K=0), g, op, op.compute, 1)
    nr, nz, nv = len(g.r_faces.keys), len(g.z_faces.keys), len(g.vertices.keys)
    B = np.r_[np.zeros(nz), 2*np.ones(nv), np.ones(nr)].astype(np.complex128)
    fluid = SimpleNamespace(
        n=np.full(op.Nc, 3.0, dtype=np.complex128),
        perp=np.full(op.Nc, 4.0, dtype=np.complex128),
        parallel=np.full(op.Nc, 9.0, dtype=np.complex128),
    )
    velocity = np.zeros((3, op.Nc))
    velocity[1] = g.cells.Z
    load = electron_pressure_traction(B, fluid, context).reshape(3, op.Nc)
    paired = -2*np.pi*np.sum(op.M0*np.sum(velocity*load, axis=0))
    # On either cap P_{phi n} = sigma*(9-4)*2/(2^2+1^2).
    exact = 2*np.pi*sum(
        face.area_per_radian * (1.6 if face.sign == 1 else 0.0)
        for face in g.walls if face.component == "z_face" and face.sign == 1
    )
    np.testing.assert_allclose(paired, exact, atol=2e-14)
