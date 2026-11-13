"""Independent samples for rectangular two-ghost magnetic reconstruction."""

from types import SimpleNamespace

import numpy as np
import pytest

from fomhd.box_boundary import install_normal_flux, magnetic_divergence, wall_magnetic_trace
from fomhd.core import reconstruct_magnetic
from fomhd.fluid import velocity_gradient
from fomhd.geometry import build_aligned_box
from fomhd.ghost_reconstruction import ghost_link_reconstruction
from fomhd.operators import apply_t2, build_operators, magnetic_weight
from fomhd.types import Context


@pytest.mark.parametrize("cylindrical", [False, True])
@pytest.mark.parametrize("cells", [2, 9])
def test_ghost_link_reconstruction_recovers_sampled_polynomials(cylindrical, cells):
    h = 1.0 / cells
    x = 0.01 + h * np.arange(cells + 1)
    lo = np.maximum(x[0], x - h / 2)
    hi = np.minimum(x[-1], x + h / 2)
    midpoint = (lo + hi) / 2
    graph = np.eye(cells + 1)
    for index in range(1, cells - 1):
        edge = np.zeros(cells + 1)
        edge[index : index + 2] = [1, -1]
        graph += np.outer(edge, edge) / 24
    if cylindrical:
        graph = graph * midpoint[None, :] / midpoint[:, None]

    gauss, quadrature = np.polynomial.legendre.leggauss(5)
    radius = midpoint[:, None] + (hi - lo)[:, None] * gauss / 2
    measure = (hi - lo)[:, None] * quadrature / 2
    if cylindrical:
        measure = measure * radius
    cell_rows, wall_rows = ghost_link_reconstruction(x, lo, hi, cylindrical, graph)
    raw_rows, _ = ghost_link_reconstruction(x, lo, hi, cylindrical, None)
    np.testing.assert_allclose(cell_rows @ graph, raw_rows, rtol=0, atol=3e-14)

    for degree in range(min(cells, 3) + 1):
        mean = (radius**degree * measure).sum(axis=1) / measure.sum(axis=1)
        stored = graph @ mean
        centers = (x[:-1] + x[1:]) / 2
        np.testing.assert_allclose(cell_rows @ stored, centers**degree, rtol=0, atol=3e-14)
        np.testing.assert_allclose(wall_rows @ stored, x[[0, -1]]**degree, rtol=0, atol=3e-14)


@pytest.mark.parametrize("cutoff", [0, 1])
def test_formal_ghost_magnetic_operators_keep_caps_and_power(cutoff):
    r = 1.0 + (np.arange(7) + 0.5) / 7
    z = (np.arange(6) + 0.5) / 6
    g = build_aligned_box(r, z, [(1.2, 0.2), (1.8, 0.2), (1.5, 0.6)])
    op = build_operators(g, cutoff, {"include_finite_mass": False, "ghost_wall": True})
    modes = 2 * cutoff + 1
    samples = 4 * cutoff + 1
    context = Context(SimpleNamespace(K=cutoff), g, op, op.compute, samples)

    nzf, nv = len(g.z_faces.keys), len(g.vertices.keys)
    raw = np.zeros(op.Nb, dtype=np.complex128)
    raw[nzf + nv :] = 2.0
    b = np.zeros((modes, op.Nb), dtype=np.complex128)
    b[cutoff] = apply_t2(op, raw)
    b[cutoff, nzf : nzf + nv] = 1.7 / g.Rbar_v

    caps = np.zeros((modes, len(g.walls), 2), dtype=np.complex128)
    wall_radius = []
    for index, face in enumerate(g.walls):
        if face.component == "z_face":
            radii = g.vertices.R[list(face.vertices)]
            interval = np.array([radii[0], radii.mean(), radii[1]])
            caps[cutoff, index] = face.sign * np.diff(interval**2)
            wall_radius.append(g.z_faces.R[face.face])
        else:
            wall_radius.append(g.r_faces.R[face.face])
    install_normal_flux(op, g, caps)
    center_field = np.array([np.zeros(op.Nc), 1.7 / g.cells.R, np.full(op.Nc, 2.0)])
    wall_field = np.array([np.zeros(len(g.walls)), 1.7 / np.array(wall_radius),
                           np.full(len(g.walls), 2.0)])
    centers = reconstruct_magnetic(op, b.ravel()).reshape(modes, -1)
    np.testing.assert_allclose(centers[cutoff], center_field.ravel(), rtol=0, atol=6e-14)
    trace = wall_magnetic_trace(b.ravel(), context)
    np.testing.assert_allclose(trace, np.broadcast_to(wall_field, trace.shape), rtol=0, atol=6e-14)

    rng = np.random.default_rng(40)
    electric = rng.normal(size=(modes, op.Ne)).astype(np.complex128)
    if cutoff:
        electric[2] += 1j * rng.normal(size=op.Ne)
        electric[0] = electric[2].conj()
    rate = np.array([-c @ e for c, e in zip(op.C, electric, strict=True)])
    updated = b + 0.01 * rate
    np.testing.assert_allclose(
        magnetic_divergence(updated.ravel(), context), 0, rtol=0, atol=8e-13
    )
    np.testing.assert_allclose(
        np.stack([db @ c for db, c in zip(op.DB, op.C, strict=True)]), 0,
        rtol=0, atol=8e-13,
    )
    magnetic_power = np.vdot(rate[-1], magnetic_weight(op, updated[-1]))
    electric_power = np.vdot(electric[-1], op.M1 * (op.Cdag[-1] @ updated[-1]))
    np.testing.assert_allclose(magnetic_power, -electric_power, rtol=0, atol=1e-12)

    rigid = np.zeros((samples, 3, op.Nc), dtype=np.complex128)
    rigid[:, 1] = 0.4 * g.cells.R
    density = SimpleNamespace(n=np.ones(samples * op.Nc), nf=np.ones(samples * op.Ne))
    gradient = velocity_gradient(density, rigid.ravel(), context).reshape(samples, 3, 3, -1)
    np.testing.assert_allclose(gradient + gradient.swapaxes(1, 2), 0, rtol=0, atol=6e-14)
    np.testing.assert_allclose(op.T_full @ rigid[0].ravel(), 0, rtol=0, atol=6e-14)
