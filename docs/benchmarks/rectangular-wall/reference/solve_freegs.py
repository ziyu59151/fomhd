"""Produce an independent lower-single-null equilibrium with FreeGS 0.8.2."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import freegs
from matplotlib.path import Path as PolygonPath
import numpy as np
from scipy.constants import mu_0
from scipy.integrate import romb
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq, root


XPOINTS = ((1.1, -0.6), (1.1, 0.8))
ISOFLUX = ((1.1, -0.6, 1.1, 0.6),)
F_T_M = 2.0
CROP_M = (0.6390625, 1.9109375, -0.875, 0.875)
P2_R_M = 2.25
PRESSURE_ION_PA = 1000.0
PRESSURE_ELECTRON_HALO_PA = 100.0
DENSITY_CORE_M3 = 2.0e19
DENSITY_HALO_M3 = 1.0e19


def edge_factor(psi_normalized):
    return np.maximum(1.0 - np.asarray(psi_normalized), 0.0)


def zero_profile(psi_normalized):
    return np.zeros_like(psi_normalized)


class PressureOnlyAtCurrent(freegs.jtor.ProfilesPprimeFfprime):
    """Normalize the pressure-only GS source to the fixed total current."""

    def __init__(self, current):
        self.current = current
        self.strength = 1.0
        super().__init__(
            lambda x: self.strength * edge_factor(x) ** 2,
            zero_profile,
            F_T_M,
            f_func=lambda x: F_T_M + zero_profile(x),
        )

    def Jtor(self, R, Z, psi, psi_bndry=None):
        self.strength = 1.0
        shape = super().Jtor(R, Z, psi, psi_bndry)
        axes, saddles = freegs.critical.find_critical(R, Z, psi)
        if axes and saddles and saddles[0][1] < axes[0][1]:
            # Select the core side of the lower saddle. Upstream core_mask can
            # include disconnected private-flux pixels in its 3×3 X block.
            shape = np.where(Z >= saddles[0][1], shape, 0.0)
        dr, dz = R[1, 0] - R[0, 0], Z[0, 1] - Z[0, 0]
        shape_integral = romb(romb(shape)) * dr * dz
        if shape_integral <= 0.0:
            raise RuntimeError("FreeGS pressure-only source lost its enclosed core")
        self.strength = self.current / shape_integral
        return self.strength * shape


def solve(count, current, rtol):
    machine = freegs.machine.TestTokamak()
    for name, coil in machine.coils:
        if name.startswith("P2"):
            coil.R = P2_R_M
    equilibrium = freegs.Equilibrium(
        tokamak=machine,
        Rmin=0.35, Rmax=P2_R_M - 0.05,
        Zmin=-1.0, Zmax=1.0,
        nx=count, ny=count,
        boundary=freegs.boundary.freeBoundaryHagenow,
    )
    constrain = freegs.control.constrain(xpoints=XPOINTS, isoflux=ISOFLUX)
    seed_pressure = 1000.0 * (current / 2.0e5) ** 2
    seed = freegs.jtor.ConstrainPaxisIp(equilibrium, seed_pressure, current, F_T_M)
    freegs.solve(equilibrium, seed, constrain, show=False)
    profiles = PressureOnlyAtCurrent(current)
    absolute, relative = freegs.solve(
        equilibrium, profiles, constrain,
        rtol=rtol, atol=1e-12, maxits=180, show=False, convergenceInfo=True,
    )
    return equilibrium, profiles.strength, absolute, relative


def current_from_flux(psi, radius, height):
    """Differentiate the exported total flux using independent fourth-order FD."""
    dr, dz = radius[1] - radius[0], height[1] - height[0]
    center = psi[2:-2, 2:-2]
    d_r = (
        psi[:-4, 2:-2] - 8 * psi[1:-3, 2:-2]
        + 8 * psi[3:-1, 2:-2] - psi[4:, 2:-2]
    ) / (12 * dr)
    d_rr = (
        -psi[:-4, 2:-2] + 16 * psi[1:-3, 2:-2] - 30 * center
        + 16 * psi[3:-1, 2:-2] - psi[4:, 2:-2]
    ) / (12 * dr**2)
    d_zz = (
        -psi[2:-2, :-4] + 16 * psi[2:-2, 1:-3] - 30 * center
        + 16 * psi[2:-2, 3:-1] - psi[2:-2, 4:]
    ) / (12 * dz**2)
    r = radius[2:-2, None]
    return -(d_rr - d_r / r + d_zz) / (mu_0 * r)


def core_separatrix(spline, axis, xpoint, bounds, count=360):
    """Trace the first separatrix crossing from the axis and include its saddle."""
    origin = axis[:2]
    direction_to_x = xpoint[:2] - origin
    saddle_angle = np.arctan2(direction_to_x[1], direction_to_x[0])
    points = [xpoint[:2]]
    for angle in saddle_angle + 2 * np.pi * np.arange(1, count) / count:
        direction = np.array([np.cos(angle), np.sin(angle)])
        near_bound = np.where(direction > 0, bounds[1], bounds[0])
        end = np.min((near_bound - origin) / direction)

        def flux(distance):
            return spline.ev(*(origin + distance * direction)) - xpoint[2]

        def derivative(distance):
            point = origin + distance * direction
            return (
                direction[0] * spline.ev(*point, dx=1)
                + direction[1] * spline.ev(*point, dy=1)
            )

        distance = np.linspace(0.0, end, 400)
        positions = origin[:, None] + distance * direction[:, None]
        values = spline.ev(*positions) - xpoint[2]
        crossings = np.flatnonzero(values <= 0.0)
        if crossings.size:
            upper = distance[crossings[0]]
        else:
            # The gap between the core and private region narrows at the X-point.
            # Include stationary points to recover roots between sampled radii.
            slopes = (
                direction[0] * spline.ev(*positions, dx=1)
                + direction[1] * spline.ev(*positions, dy=1)
            )
            minima = np.flatnonzero((slopes[:-1] < 0.0) & (slopes[1:] >= 0.0))
            upper = None
            for i in minima:
                minimum = brentq(derivative, distance[i], distance[i + 1])
                if flux(minimum) <= 0.0:
                    upper = minimum
                    break
            if upper is None:
                raise RuntimeError(f"No separatrix crossing at ray angle {angle:g}")
        crossing = brentq(flux, 0.0, upper, xtol=1e-13)
        points.append(origin + crossing * direction)
    return np.asarray(points)


def export(equilibrium, output, strength, elapsed, absolute, relative):
    psi = equilibrium.psi()
    opt, xpt = freegs.critical.find_critical(equilibrium.R, equilibrium.Z, psi)
    if not opt or not xpt:
        raise RuntimeError("Converged equilibrium lacks a magnetic axis or X-point")
    axis = np.asarray(opt[0])
    xpoint_native = np.asarray(xpt[0])
    if not np.isclose(equilibrium.psi_bndry, xpoint_native[2], atol=1e-10, rtol=1e-8):
        raise RuntimeError("FreeGS boundary flux does not select the primary X-point")
    radius = equilibrium.R[:, 0]
    height = equilibrium.Z[0, :]
    spline = RectBivariateSpline(radius, height, psi)
    xpoint_root = root(
        lambda point: [spline.ev(*point, dx=1), spline.ev(*point, dy=1)],
        xpoint_native[:2],
    )
    if not xpoint_root.success:
        raise RuntimeError("Could not refine the FreeGS X-point on its flux spline")
    xpoint = np.r_[xpoint_root.x, spline.ev(*xpoint_root.x)]
    span = axis[2] - xpoint[2]
    if span <= 0.0:
        raise RuntimeError("Positive toroidal current requires axis flux above boundary")
    native_mask = freegs.critical.core_mask(equilibrium.R, equilibrium.Z, psi, opt, xpt).astype(bool)
    native_separatrix = equilibrium.separatrix(npoints=360)
    native_error = float(np.max(np.abs(spline.ev(*native_separatrix.T) - xpoint[2])))
    separatrix = core_separatrix(
        spline, axis, xpoint,
        np.array([[radius[0], height[0]], [radius[-1], height[-1]]]),
    )
    separatrix_error = float(np.max(np.abs(spline.ev(*separatrix.T) - xpoint[2])))
    if separatrix_error / span > 1e-9:
        raise RuntimeError(f"Separatrix tracing mismatch {separatrix_error / span:g}")
    rlo, rhi, zlo, zhi = CROP_M
    r_keep = (radius >= rlo - 1e-12) & (radius <= rhi + 1e-12)
    z_keep = (height >= zlo - 1e-12) & (height <= zhi + 1e-12)
    crop = np.ix_(r_keep, z_keep)
    r, z = radius[r_keep], height[z_keep]
    for point in np.vstack((axis[:2], xpoint[:2], separatrix)):
        if not (r[0] < point[0] < r[-1] and z[0] < point[1] < z[-1]):
            raise RuntimeError(f"Crop does not enclose topology point {point}")
    # The retained crop ends before the relocated PF filaments and |Z|>=1.05 m coils.
    if not (r[-1] < P2_R_M and np.max(np.abs(z)) < 1.05):
        raise RuntimeError("Crop intersects the TestTokamak coils")
    rr, zz = equilibrium.R[crop], equilibrium.Z[crop]
    closed = PolygonPath(np.vstack((separatrix, separatrix[0])))
    points = np.column_stack((rr.ravel(), zz.ravel()))
    mask = closed.contains_points(points).reshape(rr.shape)
    if np.any(np.r_[mask[0], mask[-1], mask[:, 0], mask[:, -1]]):
        raise RuntimeError("Crop wall intersects the connected core mask")
    psi_crop = psi[crop]
    psin = (axis[2] - psi_crop) / span
    fraction = np.where(mask, edge_factor(psin) ** 3, 0.0)
    p_gs = strength * span * fraction / 3.0
    total_pressure = p_gs + PRESSURE_ION_PA + PRESSURE_ELECTRON_HALO_PA
    pprime = np.where(mask, strength * edge_factor(psin) ** 2, 0.0)
    density = DENSITY_HALO_M3 + (DENSITY_CORE_M3 - DENSITY_HALO_M3) * fraction
    jphi = rr * pprime
    j_native = equilibrium.Jtor[crop]
    if np.any(j_native[~mask] != 0.0):
        raise RuntimeError("The actual GS source extends outside the independently traced core")
    j_fd = current_from_flux(psi_crop, r, z)
    j_diff = j_fd - jphi[2:-2, 2:-2]
    j_scale = float(np.max(np.abs(jphi)))
    halo_error = np.abs(j_fd[~mask[2:-2, 2:-2]])
    b_r = equilibrium.Br(rr, zz)
    b_z = equilibrium.Bz(rr, zz)
    b_phi = equilibrium.Btor(rr, zz)
    output.mkdir(exist_ok=False)
    arrays = {
        "R_m": r, "Z_m": z, "psi_Wb_rad": psi_crop.T,
        "pressure_Pa": total_pressure.T, "Jphi_A_m2": jphi.T,
        "axis_RZ_m": axis[:2], "xpoint_RZ_m": xpoint[:2],
        "lcfs_RZ_m": separatrix,
        "psi_axis_Wb_rad": np.array(axis[2]),
        "psi_sep_Wb_rad": np.array(xpoint[2]),
        "F_T_m": np.array(F_T_M), "box_RZ_m": np.array([r[0], r[-1], z[0], z[-1]]),
        "psi_native_R_m": radius, "psi_native_Z_m": height,
        "psi_native_Wb_rad": psi.T, "native_core_mask": native_mask.T,
        "native_lcfs_RZ_m": native_separatrix, "native_xpoint_RZpsi": xpoint_native,
        "core_mask": mask.T, "p_gs_Pa": p_gs.T,
        "pprime_Pa_per_Wb_rad": pprime.T,
        "p_i_Pa": np.full_like(p_gs.T, PRESSURE_ION_PA),
        "p_e_Pa": p_gs.T + PRESSURE_ELECTRON_HALO_PA,
        "density_m3": density.T,
        "Jphi_native_A_m2": j_native.T,
        "Jphi_independent_FD_A_m2": j_fd.T,
        "BR_T": b_r.T, "BZ_T": b_z.T, "Bphi_T": b_phi.T,
        "all_opoints_RZpsi": np.asarray(opt), "all_xpoints_RZpsi": np.asarray(xpt),
    }
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise RuntimeError("FreeGS export contains nonfinite values")
    np.savez_compressed(output / "equilibrium.npz", **arrays)
    receipt = {
        "freegs_version": importlib.metadata.version("freegs"),
        "upstream_example": "https://github.com/freegs-plasma/freegs/blob/v0.8.2/01-freeboundary.py",
        "numpy_version": np.__version__,
        "scipy_version": importlib.metadata.version("scipy"),
        "freegs_N_R_Z": list(psi.shape),
        "crop_R_Z_m": [float(r[0]), float(r[-1]), float(z[0]), float(z[-1])],
        "crop_N_R_Z": [r.size, z.size],
        "axis_RZpsi": axis.tolist(), "xpoint_RZpsi": xpoint.tolist(),
        "psi_span_Wb_per_rad": float(span),
        "max_separatrix_relative_flux_error": separatrix_error / span,
        "max_native_separatrix_relative_flux_error": native_error / span,
        "Jphi_peak_A_m2": j_scale,
        "outside_traced_lcfs_source_peak_A_m2": float(np.max(np.abs(j_native[~mask]))),
        "native_mask_private_pixels": int(np.count_nonzero(native_mask[crop] & ~mask)),
        "Jphi_native_vs_final_profile_rel_linf": float(np.max(np.abs(j_native - jphi))) / j_scale,
        "Jphi_independent_FD_rel_linf": float(np.max(np.abs(j_diff))) / j_scale,
        "Jphi_independent_FD_rel_rms": float(np.sqrt(np.mean(j_diff**2))) / j_scale,
        "Jphi_halo_independent_FD_rel_linf": float(np.max(halo_error)) / j_scale,
        "F_T_m": F_T_M,
        "max_abs_RBphi_minus_F_T_m": float(np.max(np.abs(rr * b_phi - F_T_M))),
        "pprime_amplitude_Pa_per_Wb_rad": strength,
        "p_gs_axis_Pa": strength * span / 3.0,
        "n_axis_m3": DENSITY_CORE_M3, "n_halo_m3": DENSITY_HALO_M3,
        "p_i_constant_Pa": PRESSURE_ION_PA, "p_e_halo_Pa": PRESSURE_ELECTRON_HALO_PA,
        "Ip_A": float(equilibrium.plasmaCurrent()),
        "coil_currents_A": {name: float(current) for name, current in equilibrium.tokamak.getCurrents().items()},
        "solve_wall_seconds": elapsed,
        "P2_coil_radius_m": P2_R_M,
        "picard_iterations": len(absolute),
        "picard_final_abs_Wb_per_rad": float(absolute[-1]),
        "picard_final_relative_change": float(relative[-1]),
        "array_order": "Z first, R second; actual FreeGS point data transposed",
        "scope": "Poloidal flux includes external coils; crop contains no coil. The exterior has finite FOMHD density and pressure, and zero prescribed volume current. This is an equilibrium input, not FOMHD evolution evidence.",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, choices=(33, 65, 129, 257), default=65)
    parser.add_argument("--current", type=float, default=2.0e5)
    parser.add_argument("--rtol", type=float, default=1e-8)
    args = parser.parse_args()
    started = time.monotonic()
    equilibrium, strength, absolute, relative = solve(args.count, args.current, args.rtol)
    export(equilibrium, args.output, strength, time.monotonic() - started, absolute, relative)


if __name__ == "__main__":
    main()
