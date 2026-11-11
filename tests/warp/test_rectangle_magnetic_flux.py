"""Independent physical sampling of the fixed rectangular magnetic flux."""

import numpy as np
import pytest

from fomhd.geometry import build_geometry
from fomhd.operators import apply_t2, build_operators
from fomhd.types import Context


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
