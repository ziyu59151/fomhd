"""Matrix of the frozen uniform axial cold Hall block, with solver wall algebra."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, eigvals, null_space

from fomhd import operators
from fomhd.geometry import build_aligned_box


from axial_hall_box import make_mode


def row_normalized(matrix):
    scale = np.linalg.norm(matrix, axis=1)
    active = scale > 0
    return matrix[active] / scale[active, None]


def build(n, wave):
    rlo, rhi = wave.r
    zlo, zhi = wave.z
    r = rlo + (np.arange(n) + 0.5) * ((rhi - rlo) / n)
    z = zlo + (np.arange(n) + 0.5) * ((zhi - zlo) / n)
    theta = np.arange(32) * 2*np.pi/32
    geom = build_aligned_box(r, z, list(zip(3 + .25*np.cos(theta), .25*np.sin(theta), strict=True)))
    op = operators.build_operators(geom, 0)
    c, a, d = op.C[0].real, op.Cdag[0].real, op.DB[0].real
    inject, pair, t = op.I.real, op.H.real, op.T.real
    n_u = 3 * op.Nc
    m0 = np.tile(op.M0, 3)
    constrained = t.T / m0[:, None]
    q = np.eye(n_u) - constrained @ np.linalg.solve(t @ constrained, t)
    trace = np.zeros((len(geom.walls), op.Ne))
    nr_faces = len(geom.r_faces.keys)
    for i, wall in enumerate(geom.walls):
        eidx = wall.face if wall.component == "r_face" else nr_faces + op.Nc + wall.face
        trace[i, eidx] = wall.sign
    p = trace.T / op.M1[:, None]
    load = trace @ a @ c
    wall_system = load @ p
    wall_idx = op.compute.independent_rows(wall_system.astype(np.complex128), 1e-12)
    pw = np.eye(op.Ne) - p[:, wall_idx] @ np.linalg.solve(wall_system[np.ix_(wall_idx, wall_idx)], load[wall_idx])
    wall_defect = np.linalg.norm(load @ pw) / np.linalg.norm(load)
    cross = np.zeros((n_u, n_u))
    cross[:op.Nc, op.Nc : 2*op.Nc] = np.eye(op.Nc)
    cross[op.Nc : 2*op.Nc, :op.Nc] = -np.eye(op.Nc)
    # u/v_A, B/B0, E/(v_A B0). Electron flow is the projected adjoint of
    # shared flux interpolation applied to Iu - d_i A b.
    ohm = -pw @ inject @ q @ cross @ q @ pair
    eu, eb = ohm @ inject, -wave.ion_inertial_length * ohm @ a
    gyromotion = wave.charge * wave.b0 / wave.ion_mass
    system = np.block([
        [gyromotion * q @ (cross + pair @ eu), gyromotion * q @ pair @ eb],
        [-wave.alfven_speed * c @ eu, -wave.alfven_speed * c @ eb],
    ])
    basis = block_diag(null_space(row_normalized(t), rcond=1e-12),
                       null_space(row_normalized(np.vstack([d, trace @ a])), rcond=1e-12))
    restricted = basis.T @ system @ basis
    invariant_defect = np.linalg.norm(system @ basis - basis @ restricted) / np.linalg.norm(system @ basis)
    spectrum = eigvals(restricted)
    max_abs_im = np.max(abs(spectrum.imag))
    max_abs_re = np.max(abs(spectrum.real))
    ideal_dt = np.sqrt(8) / max_abs_im if max_abs_re < 1e-10 * max_abs_im else None
    return spectrum, {
        "grid": [n, n],
        "full_dimension": system.shape[0],
        "constrained_dimension": restricted.shape[0],
        "wall_projection_relative_defect": float(wall_defect),
        "invariant_subspace_relative_defect": float(invariant_defect),
        "spectrum_max_abs_real_s_inv": float(max_abs_re),
        "spectrum_max_abs_imag_rad_per_s": float(max_abs_im),
        "relative_max_real": float(max_abs_re/max_abs_im),
        "ideal_imaginary_axis_rk4_dt_s": float(ideal_dt) if ideal_dt is not None else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    source = Path(operators.__file__).resolve()
    wave = make_mode(mu0=1.25663706127e-6, ion_mass=3.343583719e-27)
    result = {
        "source": str(source),
        "operators_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": "Frozen uniform axial cold Hall operator assembled from production interpolation, Ampere, Faraday, and wall constraints; excludes pressure and nonlinear evolution, treats reference current as zero, K=0.",
        "n_m_inv3": wave.density,
        "b0_T": wave.b0,
        "alfven_speed_m_per_s": wave.alfven_speed,
        "reference_low_mode_omega_rad_per_s": wave.omega,
        "grids": {},
    }
    for n in (6, 8, 10):
        spectrum, report = build(n, wave)
        key = f"{n}x{n}"
        result["grids"][key] = report
        np.save(args.target / f"cold-{args.tag}-{key}.npy", spectrum)
    (args.target / f"cold-{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
