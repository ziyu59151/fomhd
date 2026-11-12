"""Analytic flux and compatible-curl measurements for the two A0 checkouts."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from fomhd import operators
from fomhd.box_boundary import install_normal_flux, magnetic_divergence
from fomhd.geometry import build_aligned_box
from fomhd.types import Context

MU0 = 1.25663706127e-6
BOX = (1.0, 2.2, -0.5, 0.5)
CASES = {
    "constant_axial": (lambda r, z: 0.35 * r**2, lambda r: np.zeros_like(r)),
    "affine_radial": (
        lambda r, z: 0.2 * r**2 + (0.7 / 3) * r**3,
        lambda r: np.full_like(r, -0.7),
    ),
    "affine_axial": (
        lambda r, z: -r**2 * (0.4 * z + 0.35 * z**2),
        lambda r: 0.7 * r,
    ),
    "mixed_vacuum": (
        lambda r, z: 0.1 * (r**4 - 4 * r**2 * z**2),
        lambda r: np.zeros_like(r),
    ),
    "solovev": (
        lambda r, z: -0.1 * ((r**2 - 1.6**2) ** 2 + 4 * r**2 * z**2),
        lambda r: 1.6 * r,
    ),
}


def sample(g, op, psi):
    """Physical dual-link flux differences and prescribed cap half-segments."""
    rlo, rhi, zlo, zhi = BOX
    hr, hz = g.R[1] - g.R[0], g.Z[1] - g.Z[0]
    r, z = g.z_faces.R, g.z_faces.Z
    za = np.maximum(zlo, z - hz / 2)
    zb = np.minimum(zhi, z + hz / 2)
    br = -(psi(r, zb) - psi(r, za)) / (r * g.lz)
    r, z = g.r_faces.R, g.r_faces.Z
    ra = np.maximum(rlo, r - hr / 2)
    rb = np.minimum(rhi, r + hr / 2)
    bz = (psi(rb, z) - psi(ra, z)) / (g.Rbar_r * g.lr)
    harmonic = 1.3 / g.Rbar_v
    b = operators.apply_t2(op, np.r_[br, harmonic, bz].astype(np.complex128))
    b[br.size : br.size + harmonic.size] = harmonic
    caps = np.zeros((1, len(g.walls), 2), dtype=np.complex128)
    for index, wall in enumerate(g.walls):
        r = g.vertices.R[list(wall.vertices)]
        z = g.vertices.Z[list(wall.vertices)]
        sign = -wall.sign if wall.component == "r_face" else wall.sign
        caps[0, index] = sign * np.diff(
            psi(np.array([r[0], r.mean(), r[1]]), np.array([z[0], z.mean(), z[1]]))
        )
    return b, caps


def measure(values):
    return {"rms": float(np.sqrt(np.mean(abs(values)**2))), "max": float(np.max(abs(values)))}


def weighted(op, b):
    if hasattr(operators, "magnetic_weight"):
        return operators.magnetic_weight(op, b)
    return b * op.MB


def stiffness(op):
    """Largest energy-weighted curl-pair eigenvalue, in m^-2."""
    scale = np.sqrt(op.M1)
    c, a = op.C[0].real.copy(), op.Cdag[0].real.copy()

    def mv(x):
        return scale * (a @ (c @ (x / scale)))

    operator = LinearOperator((op.Ne, op.Ne), dtype=np.float64, matvec=mv)
    rng = np.random.default_rng(11)
    value, vector = eigsh(operator, k=1, which="LA", tol=1e-12, v0=rng.normal(size=op.Ne))
    lam, v = value[0], vector[:, 0]
    return {
        "lambda_max_A_C_m_inv2": float(lam),
        "singular_curl_m_inv": float(np.sqrt(lam)),
        "ritz_relative_residual": float(np.linalg.norm(mv(v) - lam * v) / lam),
    }


def run_grid(nr, nz, target, tag):
    rlo, rhi, zlo, zhi = BOX
    R = rlo + (np.arange(nr) + 0.5) * ((rhi-rlo) / nr)
    Z = zlo + (np.arange(nz) + 0.5) * ((zhi-zlo) / nz)
    g = build_aligned_box(R, Z, [(1.4, -0.15), (1.8, -0.15), (1.6, 0.2)])
    started = perf_counter()
    op = operators.build_operators(g, 0, {"memory_cap_bytes": 4 << 30})
    setup_s = perf_counter() - started
    stiffness_record = stiffness(op)
    rng = np.random.default_rng(12)
    e = rng.normal(size=op.Ne).astype(np.complex128)
    rate = -(op.C[0] @ e)
    first, second = np.asarray(g.cells.keys).T
    lr, lz = np.minimum(first, nr-1-first), np.minimum(second, nz-1-second)
    depth = np.minimum(lr, lz)
    corners = (lr == 0) & (lz == 0)
    regions = {
        "first_excluding_corners": (depth == 0) & ~corners,
        "second": depth == 1,
        "corners": corners,
        "core": depth >= 2,
    }
    rfaces = len(g.r_faces.keys)
    wall_ids = [w.face if w.component == "r_face" else rfaces + op.Nc + w.face for w in g.walls]
    checks = {"curl_pair": stiffness_record, "build_s": setup_s, "cases": {}}
    data = dict(R=g.cells.R, Z=g.cells.Z, region_depth=depth, corners=corners)
    for name, (psi, exact_curl) in CASES.items():
        b, caps = sample(g, op, psi)
        op.wall_magnetic_divergence = None  # independent immutable initial flux per field
        install_normal_flux(op, g, caps)
        ctx = Context(None, g, op, op.compute, 1)
        j = (op.Cdag[0] @ b) / MU0
        jphi = j[rfaces : rfaces + op.Nc]
        expected = exact_curl(g.cells.R) / MU0
        error = jphi - expected
        data[f"{name}_jphi"] = jphi.real
        data[f"{name}_jphi_exact"] = expected
        data[f"{name}_jphi_error"] = error.real
        dp = (op.Cdag[0] @ rate) / MU0
        dt = 0.01
        b_trial = b + dt * rate
        work_m = np.vdot(rate, weighted(op, b_trial)) / MU0
        work_e = np.vdot(e, op.M1 * ((op.Cdag[0] @ b_trial) / MU0))
        power_scale = max(abs(work_m), abs(work_e), 1e-300)
        checks["cases"][name] = {
            "Jphi_error_A_per_m2": {key: measure(error[mask]) for key, mask in regions.items()},
            "Jphi_error_whole_A_per_m2": measure(error),
            "Jphi_exact_whole_A_per_m2": measure(expected),
            "divB_plus_caps_max_T_per_m": float(np.max(abs(magnetic_divergence(b, ctx)))),
            "cap_abs_max_Wb_per_rad": float(np.max(abs(caps))),
            "signed_cap_sum_Wb_per_rad": float(caps.real.sum()),
            "cap_div_after_algebraic_faraday_update_max_T_per_m": float(np.max(abs(magnetic_divergence(b_trial, ctx)))),
            "wall_Jn_max_A_per_m2": float(np.max(abs(j[wall_ids]))),
            "wall_Jdot_n_max_A_per_m2": float(np.max(abs(dp[wall_ids]))),
            "magnetic_plus_electric_power_relative": float(abs(work_m + work_e) / power_scale),
        }
    np.savez(target / f"{tag}-{nr}x{nz}.npz", **data)
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    path = Path(operators.__file__).resolve()
    out = {
        "source": str(path),
        "operators_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "physical_box_m": BOX,
        "mu0_H_per_m": MU0,
        "grids": {},
    }
    for nr, nz in ((7, 6), (14, 12), (28, 24)):
        key = f"{nr}x{nz}"
        out["grids"][key] = run_grid(nr, nz, args.target, args.tag)
        (args.target / f"{args.tag}.json").write_text(json.dumps(out, indent=2) + "\n")
        print(args.tag, key, "complete", flush=True)


if __name__ == "__main__":
    main()
