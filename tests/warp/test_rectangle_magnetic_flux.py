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

def test_fine_wr_power_preserves_threaded_electron_wall_work():
    from types import SimpleNamespace

    from fomhd import massless, massless_checks, massless_field, multirate_wr_checks
    from fomhd.box_boundary import install_normal_flux
    from test_massless_support import context, stage_tag, uniform_state

    c = context(K=0, hard_diagnostics=False, nodes_r=6, nodes_z=6)
    g, op = c.geometry, c.op
    state = uniform_state(c)
    flux = np.zeros((1, len(g.walls), 2), dtype=np.complex128)
    axial = 0.5 * c.p.ref.magnetic
    for w, side in enumerate(g.walls):
        if side.component == "z_face":
            a, b = g.vertices.R[list(side.vertices)]
            radii = np.array([a, (a+b)/2, b])
            flux[0, w] = side.sign * axial * np.diff(radii**2)/2
    install_normal_flux(op, g, flux)
    root = np.sqrt(op.MB)
    # Harmonic extension of independently integrated axial cap data.
    correction = np.linalg.lstsq(
        op.DB[0] / root[None,:], -op.wall_magnetic_divergence, rcond=1e-12
    )[0] / root
    state.B += correction
    np.testing.assert_allclose(op.Cdag[0] @ correction, 0.0, atol=2e-17)

    ion = c.p.species[0]
    speed = 10.0 * (g.cells.Z - g.Z[0]) / c.p.ref.length
    state.fluid[0].mom.reshape(3, op.Nc)[1] = ion.mass * state.fluid[0].n * speed
    state.fluid[1].parallel[:] *= 2.0
    tag = stage_tag(state)
    full = massless.evaluate_stage(state, tag, c)
    frame = massless_field.evaluate(state, tag, c)
    view = multirate_wr_checks._fine_view(
        SimpleNamespace(record=SimpleNamespace(frame=frame))
    )
    ledger = massless_checks._electron_ledger(view, c)
    boundary = multirate_wr_checks._boundary_summands(view.state, view, c)
    expected = full.boundary.electron_pressure_work
    assert abs(expected) > 1e-12
    np.testing.assert_allclose(ledger.pressure_wall, expected, rtol=2e-12)
    np.testing.assert_allclose(boundary.energy[1], expected, rtol=2e-12)
    metrics = massless_checks._wall_work_metrics(view.state, view, c)
    residuals = massless_checks._fine_electron_work_residuals(ledger, metrics, c)
    np.testing.assert_allclose(residuals["electron_pressure_adjoint"], 0, atol=3e-10*abs(expected))
    np.testing.assert_allclose(residuals["electron_electric_pairing"], 0, atol=3e-10*abs(expected))
    # An implementation that omits the wall power fails on this physical input.
    assert abs(ledger.pressure_kinetic + ledger.pressure_internal_full) > 0.9*abs(expected)
