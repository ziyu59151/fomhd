"""Exercise the actual closed-wall field correction on a retained m=±1 pair."""

from pathlib import Path
from types import SimpleNamespace
import argparse
import json

import numpy as np

from analytic import BOX, sample, weighted
from fomhd import operators
from fomhd.box_boundary import install_normal_flux, magnetic_divergence
from fomhd.geometry import build_aligned_box
from fomhd.massless import _wall_correction
from fomhd.types import Context


def audit(nr, nz):
    rlo, rhi, zlo, zhi = BOX
    r = rlo + (np.arange(nr) + .5) * ((rhi-rlo) / nr)
    z = zlo + (np.arange(nz) + .5) * ((zhi-zlo) / nz)
    g = build_aligned_box(r, z, [(1.4, -.15), (1.8, -.15), (1.6, .2)])
    op = operators.build_operators(g, 1)
    p = SimpleNamespace(K=1, memory_limit=2<<30, rank_rtol=1e-12,
                        ref=SimpleNamespace(length=1.0, electric=1.0))
    context = Context(p, g, op, op.compute, 3)
    b0, caps0 = sample(g, op, lambda r, z: .35*r*r)
    caps = np.zeros((3, len(g.walls), 2), dtype=np.complex128)
    caps[1] = caps0[0]
    install_normal_flux(op, g, caps)
    rng = np.random.default_rng(12)
    e1 = rng.normal(size=op.Ne) + 1j*rng.normal(size=op.Ne)
    e = np.stack((e1.conj(), rng.normal(size=op.Ne), e1))
    correction, record = _wall_correction(e.ravel(), context)
    e = e + correction.reshape(e.shape)
    rate = np.stack([-matrix @ field for matrix, field in zip(op.C, e, strict=True)])
    b = .01*rate
    b[1] += b0
    j = np.stack([matrix @ field for matrix, field in zip(op.Cdag, b, strict=True)])
    continuity = np.stack([matrix @ field for matrix, field in zip(op.DJ, j, strict=True)])
    wall_n = np.stack([matrix @ field for matrix, field in zip(record.cache.extraction, j, strict=True)])
    work_m = np.vdot(rate, weighted(op, b))
    work_e = np.vdot(e, op.M1 * j)
    eps = np.finfo(float).eps
    def max_abs(values):
        return float(np.max(abs(values)))
    return {
        "grid": [nr, nz],
        "cap_abs_max_Wb_per_rad": max_abs(caps),
        "DBC_matrix_max_m_inv2": [max_abs(db @ c) for db, c in zip(op.DB, op.C, strict=True)],
        "divB_plus_fixed_cap_after_update_max_T_per_m": max_abs(magnetic_divergence(b.ravel(), context)),
        "div_curl_B_max_T_per_m2": max_abs(continuity),
        "wall_normal_curl_B_max_T_per_m": max_abs(wall_n),
        "wall_normal_trace_finite": bool(np.all(np.isfinite(op.box_wall.normal_hat))),
        "power_pair_relative_error": float(abs(work_m+work_e)/max(abs(work_m),abs(work_e),eps)),
        "magnetic_metric_quadratic_form_positive": bool(np.vdot(b, weighted(op, b)).real > 0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tag")
    p.add_argument("target", type=Path)
    args = p.parse_args()
    result = {f"{r}x{z}": audit(r, z) for r, z in ((7, 6), (10, 8))}
    (args.target / f"modal-{args.tag}.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
