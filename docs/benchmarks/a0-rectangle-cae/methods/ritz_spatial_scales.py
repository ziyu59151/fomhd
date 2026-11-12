"""Measure spatial scales of the independently selected cold-Hall Ritz branch."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.polynomial import polyder, polyval2d
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq, minimize

import box_reference as reference


def parallel_field(result, index):
    box = result["box"]
    _, potential_basis = reference.terms(box)
    coefficients = np.einsum(
        "k,kcij->cij", result["a"][:, index],
        np.stack([coefficients for coefficients, _ in potential_basis]),
    )
    field = np.zeros_like(coefficients[0])
    field[:, :-1] = polyder(coefficients[0], axis=1) / box.a
    field[:-1, :] -= polyder(coefficients[2], axis=0) / box.a
    return field


def effective_wavelengths(box, field, minimum_wall_distance):
    nodes, weights = leggauss(2 * box.order + 12)
    fraction = 1 - minimum_wall_distance / box.a
    x, z = np.meshgrid(fraction * nodes, fraction * nodes, indexing="ij")
    volume = np.outer(weights, weights) * (box.R0 + box.a * x)
    field_norm = np.sum(volume * abs(polyval2d(x, z, field)) ** 2)
    return [
        float(2 * np.pi * box.a / np.sqrt(
            np.sum(volume * abs(polyval2d(x, z, polyder(field, axis=axis))) ** 2)
            / field_norm
        ))
        for axis in (0, 1)
    ]


def antinode_and_widths(box, field):
    nodes = np.linspace(-1, 1, 401)
    x, z = np.meshgrid(nodes, nodes, indexing="ij")
    amplitude = abs(polyval2d(x, z, field))
    i, j = np.unravel_index(amplitude.argmax(), amplitude.shape)
    peak = minimize(
        lambda point: -abs(polyval2d(*point, field)) / amplitude[i, j],
        [nodes[i], nodes[j]], bounds=[(-1, 1), (-1, 1)],
        method="L-BFGS-B", options={"ftol": 1e-14, "gtol": 1e-9},
    ).x
    peak_value = abs(polyval2d(*peak, field))
    intervals = []
    for axis in (0, 1):
        def half_height(node):
            point = peak.copy()
            point[axis] = node
            return abs(polyval2d(*point, field)) / peak_value - 0.5

        # Stop at the nearest half-height crossing on each side of the peak.
        # This keeps separate lobes out of the width.
        endpoints = []
        for wall in (-1, 1):
            last = peak[axis]
            for node in np.linspace(last, wall, 401)[1:]:
                if half_height(node) < 0:
                    endpoints.append(brentq(half_height, *sorted((node, last))))
                    break
                last = node
            else:
                endpoints.append(wall)
        intervals.append([float(box.a * x) for x in endpoints])
    return peak, intervals


def wall_half_amplitude_distances(box, field, peak):
    distances = {}
    for axis, direction in enumerate(("r", "z")):
        def amplitude(node):
            point = peak.copy()
            point[axis] = node
            return abs(polyval2d(*point, field))

        for wall, label in ((-1, "lower"), (1, "upper")):
            half_wall = 0.5 * amplitude(wall)
            nodes = np.linspace(wall, peak[axis], 301)
            crossing = next((i for i in range(1, nodes.size) if amplitude(nodes[i]) < half_wall), None)
            distance = None
            if crossing is not None:
                root = brentq(lambda node: amplitude(node) - half_wall,
                              *sorted(nodes[crossing - 1:crossing + 1]))
                distance = float(box.a * abs(root - wall))
            distances[f"{direction}_{label}"] = distance
    return distances


def measure(path):
    result = reference.load(path)
    box = result["box"]
    index = int(np.flatnonzero(
        (result["norm_omega"] > 0.4) & (result["compression"] > 0.6)
    )[0])
    field = parallel_field(result, index)
    x = np.array([-0.82, -0.33, 0.0, 0.28, 0.73])
    z = np.array([0.11, 0.77, -0.34, -0.91, 0.62])
    direct = reference.evaluate(result, index, box.R0 + box.a*x, box.a*z)["B"][:, 1]
    np.testing.assert_allclose(polyval2d(x, z, field), direct, rtol=3e-8, atol=1e-10)
    peak, intervals = antinode_and_widths(box, field)
    return {
        "archive": path.name, "order": box.order, "mode_index": index,
        "frequency_hz": float(result["omega"][index] / (2 * np.pi)),
        "half_width_m": box.a,
        "antinode_offset_r_z_m": (box.a * peak).tolist(),
        "effective_wavelength_r_z_m": effective_wavelengths(box, field, 0.0),
        "interior_effective_wavelength_r_z_m": effective_wavelengths(box, field, 0.20),
        "half_max_intervals_offset_r_z_m": intervals,
        "half_max_width_r_z_m": [hi - lo for lo, hi in intervals],
        "wall_to_half_amplitude_m": wall_half_amplitude_distances(box, field, peak),
    }, field, peak


def save_figure(measurement, field, peak, output):
    a = measurement["half_width_m"]
    phase_peak = polyval2d(*peak, field)
    n = np.linspace(-1, 1, 401)
    r, z = np.meshgrid(n, n)
    pattern = (polyval2d(r, z, field) / phase_peak).real
    fig, (ax, cuts) = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    image = ax.pcolormesh(a*r, a*z, pattern, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
    ax.axvline(a*peak[0], color="k", lw=0.8, ls="--")
    ax.axhline(a*peak[1], color="k", lw=0.8, ls=":")
    ax.set(xlabel="R − R₀ [m]", ylabel="Z [m]", aspect="equal", title="Ritz mode, phase at the antinode")
    fig.colorbar(image, ax=ax, label="Re(δB∥) / peak")
    for axis, direction, style in ((0, "R", "-"), (1, "Z", "--")):
        coordinates = [np.full_like(n, x) for x in peak]
        coordinates[axis] = n
        cuts.plot(a*n, abs(polyval2d(*coordinates, field) / phase_peak), style, label=direction)
    cuts.axhline(0.5, color="0.4", lw=1, ls=":")
    cuts.set(xlabel="R − R₀ or Z [m]", ylabel="|δB∥| / peak", title="Cuts through the antinode",
             xlim=(-a, a), ylim=(0, 1.05))
    cuts.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def frozen_common_support(paths, reference_field):
    # Midpoints keep the entire evaluation inside all cell-center grids.
    # Each mask edge is aligned with the quadrature cells.
    coordinates = -0.55 + 0.0025 * (np.arange(440) + 0.5)
    z, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    r = 15.0 + x
    wall_distance = 0.60 - np.maximum(abs(x), abs(z))
    exact = polyval2d(x / 0.60, z / 0.60, reference_field)
    masks = {
        "all_common_centers_support": wall_distance >= 0.05,
        "interior_ge_0p20m": wall_distance >= 0.20,
        "wall_band_0p05_to_0p20m": (wall_distance >= 0.05) & (wall_distance < 0.20),
    }
    records = []
    for path in paths:
        with np.load(path) as source:
            radial = np.unique(source["R"])
            axial = np.unique(source["Z"])
            shape = (axial.size, radial.size)
            np.testing.assert_array_equal(source["R"].reshape(shape)[0, :], radial)
            np.testing.assert_array_equal(source["Z"].reshape(shape)[:, 0], axial)
            mode = source["B_nodal"].reshape(3, *shape)[1]
        interpolated = RegularGridInterpolator((axial, radial), mode)(np.stack((z, r), axis=-1))
        overlaps = {}
        for label, mask in masks.items():
            a, b, weight = exact[mask], interpolated[mask], r[mask]
            product = np.vdot(a, weight*a).real * np.vdot(b, weight*b).real
            overlaps[label] = float(abs(np.vdot(a, weight*b)) / np.sqrt(product))
        records.append({"grid": [radial.size, axial.size], "archive": path.name, "overlap": overlaps})
    return {
        "scope": "Bilinear reconstruction of frozen A0 modes against the same order-16 analytic Ritz field.",
        "box_width_m": 1.20, "common_support_inset_m": 0.05, "midpoint_spacing_m": 0.0025,
        "results": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=Path("independent-review"))
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--include-order-18", action="store_true",
                        help="Measure the supplemental narrow-box reference")
    parser.add_argument("--frozen-narrow", type=Path, nargs=3,
                        help="Compare the narrow 12, 16 and 20 grid frozen branches on common support")
    args = parser.parse_args()
    records = []
    for label in ("a0p6", "a0p75"):
        orders = (12, 14, 16, 18) if args.include_order_18 and label == "a0p6" else (12, 14, 16)
        for order in orders:
            record, field, peak = measure(args.references / f"ritz-R15-n2e19-{label}-p{order}.npz")
            if order == 16:
                box_width = 2 * record["half_width_m"]
                record["grid_scales"] = [
                    {
                        "cells_per_direction": cells, "spacing_m": box_width / cells,
                        "cells_per_effective_wavelength_r_z": [
                            x * cells / box_width for x in record["effective_wavelength_r_z_m"]
                        ],
                        "cells_per_half_max_width_r_z": [
                            x * cells / box_width for x in record["half_max_width_r_z_m"]
                        ],
                    }
                    for cells in ((12, 16, 20, 24, 32) if label == "a0p6" else (20,))
                ]
                if label == "a0p6":
                    save_figure(record, field, peak, args.output / "ritz-spatial-scales.png")
                    if args.frozen_narrow:
                        comparison = frozen_common_support(args.frozen_narrow, field)
                        (args.output / "ritz-frozen-common-support.json").write_text(
                            json.dumps(comparison, indent=2) + "\n"
                        )
            records.append(record)
    (args.output / "ritz-spatial-scales.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
