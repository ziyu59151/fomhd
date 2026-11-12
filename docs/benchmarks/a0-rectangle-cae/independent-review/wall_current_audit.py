"""Separate weak-wall compatibility and current localization in the cold Ritz branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from box_reference import MU0, evaluate, load
from prepare import choose_branch


def quadrature_interval(lo, hi, degree):
    nodes, weights = leggauss(degree)
    scale = (hi - lo) / 2
    return (lo + hi) / 2 + scale * nodes, scale * weights


def radial_box_patch(result, branch, r_span, z_span, degree):
    r, wr = quadrature_interval(*r_span, degree)
    z, wz = quadrature_interval(*z_span, degree)
    rr, zz = np.meshgrid(r, z)
    weight = (rr * np.outer(wz, wr)).ravel()
    q = evaluate(result, branch, rr, zz)
    box = result["box"]
    guide = box.B0 * box.R0 / rr.ravel()
    force = np.zeros_like(q["J"])
    force[:, 0] = -guide * q["J"][:, 2]
    force[:, 2] = guide * q["J"][:, 0]
    residual = 1j * result["omega"][branch] * box.mass * box.density * q["u"] - force

    def integral(vector):
        return float(weight @ np.sum(abs(vector)**2, axis=1))

    return {
        "volume_per_rad": float(weight.sum()),
        "J_squared": integral(q["J"]),
        "J_toroidal_squared": integral(q["J"][:, 1:2]),
        "B_squared": integral(q["B"]),
        "force_squared": integral(force),
        "momentum_residual_squared": integral(residual),
    }


def wall_trace(result, branch, degree):
    box = result["box"]
    offset, weight = quadrature_interval(-box.a, box.a, degree)
    radial = box.R0 + offset
    sides = [
        ("R_lower", box.R0 - box.a, offset, 0, 2),
        ("R_upper", box.R0 + box.a, offset, 0, 2),
        ("Z_lower", radial, -box.a, 2, 0),
        ("Z_upper", radial, box.a, 2, 0),
    ]
    terms = []
    for name, r, z, normal, tangent in sides:
        values = evaluate(result, branch, r, z)
        magnitude = box.B0 * box.R0 / r
        surface_weight = r * weight
        terms.append({
            "side": name,
            "normal_current_peak": float(np.max(abs(values["J"][:, normal]))),
            "normal_velocity_peak": float(np.max(abs(values["u"][:, normal]))),
            "poloidal_tangential_current_peak": float(np.max(abs(values["J"][:, tangent]))),
            "normal_force_squared": float(surface_weight @ abs(magnitude * values["J"][:, tangent])**2),
            "surface_per_rad": float(surface_weight.sum()),
        })
    return terms


def measure(path, degree, collar):
    result = load(path)
    box = result["box"]
    i = choose_branch(result)
    edges = [-box.a, -box.a + collar, box.a - collar, box.a]
    r_bounds = list(zip(box.R0 + np.asarray(edges[:-1]), box.R0 + np.asarray(edges[1:])))
    z_bounds = list(zip(edges[:-1], edges[1:]))
    patches = {}
    for j, radial in enumerate(r_bounds):
        for k, axial in enumerate(z_bounds):
            label = "core" if j == k == 1 else "sides" if j == 1 or k == 1 else "corners"
            patches.setdefault(label, []).append(radial_box_patch(result, i, radial, axial, degree))
    sums = {
        label: {key: sum(patch[key] for patch in items) for key in items[0]}
        for label, items in patches.items()
    }
    total = {key: sum(row[key] for row in sums.values()) for key in sums["core"]}
    traces = wall_trace(result, i, 2 * degree)
    wall_force_rms = np.sqrt(sum(row["normal_force_squared"] for row in traces) /
                             sum(row["surface_per_rad"] for row in traces))
    volume_force_rms = np.sqrt(total["force_squared"] / total["volume_per_rad"])
    return {
        "order": box.order,
        "frequency_hz": float(result["omega"][i] / (2*np.pi)),
        "J_l2_mu0_over_B_l2_per_m": float(MU0 * np.sqrt(total["J_squared"] / total["B_squared"])),
        "J_toroidal_fraction": total["J_toroidal_squared"] / total["J_squared"],
        "J_squared_fraction": {name: row["J_squared"] / total["J_squared"] for name, row in sums.items()},
        "J_toroidal_fraction_by_region": {name: row["J_toroidal_squared"] / row["J_squared"] for name, row in sums.items()},
        "strong_momentum_residual_to_force": {
            name: float(np.sqrt(row["momentum_residual_squared"] / row["force_squared"]))
            for name, row in dict(sums, whole_box=total).items()
        },
        "wall_normal_force_rms_over_volume_force_rms": float(wall_force_rms / volume_force_rms),
        "wall_normal_to_poloidal_tangent_current_peak": (
            max(row["normal_current_peak"] for row in traces) /
            max(row["poloidal_tangential_current_peak"] for row in traces)
        ),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--quadrature-order", type=int, default=22)
    args = p.parse_args()
    collar = .05
    records = []
    for order in (12, 14, 16, 18):
        path = args.references / f"ritz-R15-n2e19-a0p6-p{order}.npz"
        row = measure(path, args.quadrature_order, collar)
        print(json.dumps(row), flush=True)
        records.append(row)
    args.output.write_text(json.dumps({
        "scope": "Independent weak cold Hall Ritz reference, no solver trajectory",
        "collar_m": collar,
        "quadrature_order_per_patch_direction": args.quadrature_order,
        "records": records,
    }, indent=2) + "\n")
