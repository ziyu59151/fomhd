"""FOMHD-side conversion of a pinned VEQPy equilibrium snapshot.

This module is NumPy-only. It never imports VEQPy and does not replace FOMHD
current, Ohm, RHS, or time-state construction.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


MU0_VEQ = 4.0e-7 * math.pi
MU0_FOMHD_REFERENCE = 1.25663706127e-6
ELEMENTARY_CHARGE_C = 1.602176634e-19
SOURCE_COMMIT = "2b64c9a1bffddc25c176bd52231bc3b481715c6c"
SOURCE_TREE = "ac8635c629d100db8998f9904793dbf697373eb7"
SOURCE_ARCHIVE_SHA256 = (
    "6b553ce32bb9261e4782af4542b7be15f14621f3256d303f9b175fdd582bfc1f"
)


def _readonly(values: Any) -> np.ndarray:
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class VeqSourceMetadata:
    source_commit: str
    source_tree: str
    archive_sha256: str
    mu0_VEQ_H_per_m: float
    electron_charge_C: float
    R0_m: float
    Z0_m: float
    a_m: float
    F_edge_Tm: float
    B0_T: float
    p_edge_target_Pa: float
    pprime_F_Pa_per_Wb_rad: float
    density_m3: float
    ion_pressure_fraction: float
    native_p0_Pa: float
    raw_p0_input_Pa: float
    Ip_A: float
    pressure_offset_Pa: float
    flux_orientation: str = "positive_interior_reversed_from_VEQ_Psi"
    representation: str = "constant_F_circular_pf_psin_ip"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "archive_sha256": self.archive_sha256,
            "mu0_VEQ_H_per_m": self.mu0_VEQ_H_per_m,
            "mu0_FOMHD_reference_H_per_m": MU0_FOMHD_REFERENCE,
            "mu0_relative_difference": abs(
                self.mu0_VEQ_H_per_m - MU0_FOMHD_REFERENCE
            )
            / self.mu0_VEQ_H_per_m,
            "electron_charge_C": self.electron_charge_C,
            "boundary": {
                "R0_m": self.R0_m,
                "Z0_m": self.Z0_m,
                "a_m": self.a_m,
                "F_edge_Tm": self.F_edge_Tm,
                "B0_T": self.B0_T,
            },
            "constant_F_family": {
                "F_Tm": self.F_edge_Tm,
                "Fprime_Tm": 0.0,
                "pprime_F_Pa_per_Wb_rad": self.pprime_F_Pa_per_Wb_rad,
                "p_edge_target_Pa": self.p_edge_target_Pa,
                "density_m3": self.density_m3,
                "ion_pressure_fraction": self.ion_pressure_fraction,
                "Ip_A": self.Ip_A,
            },
            "native_pressure": {
                "p0_native_Pa": self.native_p0_Pa,
                "raw_p0_input_Pa": self.raw_p0_input_Pa,
                "pressure_offset_Pa": self.pressure_offset_Pa,
                "offset_applies_to_gradient": False,
            },
            "raw_units": {
                "P_psin": "Pa",
                "FF_psin": "(T*m)^2",
                "p0": "Pa",
                "Ip": "A",
            },
            "physical_units": {
                "psi": "Wb/rad",
                "P_psi": "Pa/(Wb/rad)",
                "B": "T",
                "J": "A/m^2",
                "E": "V/m",
                "u": "m/s",
            },
            "current_semantics": {
                "J_profile": (
                    "continuous constant-F profile target; not reconstructed B"
                ),
                "J_ampere": "curl(B)/mu0 from reconstructed B",
                "native_jphi": "solver profile-derived jphi; normalization check only",
                "Ip_A": "continuous analytic target used by the Ip constraint",
                "I_p_profile": "discrete physical-area integral of J_profile",
                "I_p_ampere": (
                    "discrete physical-area integral of curl(B)/mu0; "
                    "no profile-current injection"
                ),
                "u_e_profile": "-J_profile/(e*n)",
                "u_e_ampere": "-J_ampere/(e*n)",
            },
            "wall_normal": {
                "primary": (
                    "smooth Fourier theta tangent collocated at each wall sample"
                ),
                "diagnostic": "separate polygon-segment midpoint chord sampling",
            },
            "axis_regularization": {
                "flux": "grad(psi_F)=0 exactly at the known stationary axis",
                "other_fields": "surface-mean physical-gradient radial limit",
                "duplicate_theta_samples_excluded": True,
            },
            "flux_conversion": {
                "VEQ_Psi": "Wb",
                "VEQ_psi": "Psi/(2*pi)",
                "psi_F": "psi_edge - Psi/(2*pi), psi_edge=Psi_edge/(2*pi)",
                "orientation": self.flux_orientation,
            },
            "representation": self.representation,
            "source_references": {
                "units_and_flux": "docs/veqpy/model.md:65-74,95-103",
                "jphi": "veqpy/model/equilibrium.py:2566-2576",
                "pressure": "veqpy/model/equilibrium.py:2324-2338,2566-2576",
                "p0_normalization": (
                    "veqpy/kernels/numba_kernel/numba_source.py:604-663; "
                    "veqpy/kernels/numba_kernel/runtime.py:303-312,420-422"
                ),
                "PF_psin_ip": (
                    "veqpy/kernels/numba_kernel/numba_source.py:1590-1682"
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class VeqEquilibriumSnapshot:
    metadata: VeqSourceMetadata
    r: np.ndarray
    theta: np.ndarray
    R_m: np.ndarray
    Z_m: np.ndarray
    radial_differentiator: np.ndarray
    psi_edge_Wb_per_rad: float
    psi_F_Wb_per_rad: np.ndarray
    F_Tm: np.ndarray
    Fprime_Tm: np.ndarray
    B_R_T: np.ndarray
    B_phi_T: np.ndarray
    B_Z_T: np.ndarray
    J_profile_R_A_m2: np.ndarray
    J_profile_phi_A_m2: np.ndarray
    J_profile_Z_A_m2: np.ndarray
    J_ampere_R_A_m2: np.ndarray
    J_ampere_phi_A_m2: np.ndarray
    J_ampere_Z_A_m2: np.ndarray
    I_p_profile_A: float
    I_p_ampere_A: float
    p_native_Pa: np.ndarray
    p_total_Pa: np.ndarray
    p_i_Pa: np.ndarray
    p_e_perp_Pa: np.ndarray
    p_e_parallel_Pa: np.ndarray
    n_m3: np.ndarray
    u_i_R_m_s: np.ndarray
    u_i_phi_m_s: np.ndarray
    u_i_Z_m_s: np.ndarray
    u_e_profile_R_m_s: np.ndarray
    u_e_profile_phi_m_s: np.ndarray
    u_e_profile_Z_m_s: np.ndarray
    u_e_ampere_R_m_s: np.ndarray
    u_e_ampere_phi_m_s: np.ndarray
    u_e_ampere_Z_m_s: np.ndarray
    E_profile_R_V_m: np.ndarray
    E_profile_phi_V_m: np.ndarray
    E_profile_Z_V_m: np.ndarray
    native_psi_Wb: np.ndarray
    native_p0_Pa: float
    native_jphi_A_m2: np.ndarray
    native_p_psi_Pa_per_Wb_rad: np.ndarray
    native_profile_jphi_residual_max_A_m2: float
    ampere_profile_phi_max_abs_residual_A_m2: float
    native_p_psi_offset_from_minus_pprime: float
    axis_gradient_samples_regularized: int
    axis_B_R_theta_spread_T: float
    axis_B_Z_theta_spread_T: float
    axis_B_poloidal_max_abs_T: float
    wall_normal_B_smooth_max_abs_T: float
    wall_normal_B_polygon_midpoint_max_abs_T: float


def make_source_metadata(
    *,
    source_commit: str,
    source_tree: str,
    archive_sha256: str,
    R0_m: float,
    Z0_m: float,
    a_m: float,
    F_edge_Tm: float,
    p_edge_target_Pa: float,
    pprime_F_Pa_per_Wb_rad: float,
    density_m3: float,
    ion_pressure_fraction: float,
    native_p0_Pa: float,
    raw_p0_input_Pa: float,
    Ip_A: float,
) -> VeqSourceMetadata:
    if not 0.0 < ion_pressure_fraction < 1.0:
        raise ValueError("ion_pressure_fraction must lie strictly between zero and one")
    if not density_m3 > 0.0 or not a_m > 0.0 or not R0_m > 0.0:
        raise ValueError("density, minor radius, and major radius must be positive")
    if not F_edge_Tm > 0.0 or not pprime_F_Pa_per_Wb_rad > 0.0:
        raise ValueError("constant-F family requires positive F and pprime")
    return VeqSourceMetadata(
        source_commit=str(source_commit),
        source_tree=str(source_tree),
        archive_sha256=str(archive_sha256),
        mu0_VEQ_H_per_m=MU0_VEQ,
        electron_charge_C=ELEMENTARY_CHARGE_C,
        R0_m=float(R0_m),
        Z0_m=float(Z0_m),
        a_m=float(a_m),
        F_edge_Tm=float(F_edge_Tm),
        B0_T=float(F_edge_Tm / R0_m),
        p_edge_target_Pa=float(p_edge_target_Pa),
        pprime_F_Pa_per_Wb_rad=float(pprime_F_Pa_per_Wb_rad),
        density_m3=float(density_m3),
        ion_pressure_fraction=float(ion_pressure_fraction),
        native_p0_Pa=float(native_p0_Pa),
        raw_p0_input_Pa=float(raw_p0_input_Pa),
        Ip_A=float(Ip_A),
        pressure_offset_Pa=float(p_edge_target_Pa - native_p0_Pa),
    )


def constant_f_profiles(
    metadata: VeqSourceMetadata, psi_F_Wb_per_rad: np.ndarray
) -> dict[str, np.ndarray]:
    """Evaluate the selected SI flux-function family on supplied psi_F."""

    psi = np.asarray(psi_F_Wb_per_rad, dtype=np.float64)
    if not np.all(np.isfinite(psi)):
        raise ValueError("psi_F must contain only finite values")
    F = np.full_like(psi, metadata.F_edge_Tm)
    Fprime = np.zeros_like(psi)
    pressure = (
        metadata.p_edge_target_Pa
        + metadata.pprime_F_Pa_per_Wb_rad * psi
    )
    return {
        "F_Tm": _readonly(F),
        "Fprime_Tm": _readonly(Fprime),
        "p_Pa": _readonly(pressure),
        "P_psi_Pa_per_Wb_rad": _readonly(
            np.full_like(psi, -metadata.pprime_F_Pa_per_Wb_rad)
        ),
    }


def _theta_derivative_1d(values: np.ndarray) -> np.ndarray:
    count = values.size
    frequencies = np.fft.rfftfreq(count, d=1.0 / count)
    spectrum = np.fft.rfft(values)
    spectrum *= 1j * frequencies
    return np.fft.irfft(spectrum, n=count)


def _theta_derivative(values: np.ndarray) -> np.ndarray:
    return np.stack(
        [_theta_derivative_1d(row) for row in values],
        axis=0,
    )


def _physical_gradient(
    values: np.ndarray,
    radial_differentiator: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    *,
    r: np.ndarray | None = None,
    axis_gradient: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    value_r = radial_differentiator @ values
    value_t = _theta_derivative(values)
    R_r = radial_differentiator @ R
    R_t = _theta_derivative(R)
    Z_r = radial_differentiator @ Z
    Z_t = _theta_derivative(Z)
    jacobian = R_r * Z_t - R_t * Z_r
    scale = max(float(np.max(np.abs(R))), float(np.max(np.abs(Z))), 1.0)
    bad = np.abs(jacobian) <= 64.0 * np.finfo(np.float64).eps * scale * scale
    bad_rows = np.any(bad, axis=1)
    excluded = int(np.sum(bad))
    if np.any(bad_rows[1:]):
        raise ValueError("evaluation grid has a singular non-axis radial row")
    if bad_rows[0] and not np.all(bad[0]):
        raise ValueError("evaluation grid has a partial singular axis row")
    if bad_rows[0]:
        axis_span = max(float(np.ptp(R[0])), float(np.ptp(Z[0])))
        if axis_span > 64.0 * np.finfo(np.float64).eps * scale:
            raise ValueError("first singular row is not the coalesced axis")
    safe = np.where(bad, 1.0, jacobian)
    derivative_R = (value_r * Z_t - value_t * Z_r) / safe
    derivative_Z = (R_r * value_t - R_t * value_r) / safe
    if excluded:
        if r is None or r.shape != (values.shape[0],):
            raise ValueError("axis regularization requires one radial node per row")
        if axis_gradient is None:
            axis_R, axis_Z = _axis_gradient_limit(
                derivative_R, derivative_Z, r, bad
            )
        else:
            axis_R, axis_Z = (float(axis_gradient[0]), float(axis_gradient[1]))
            if not np.isfinite(axis_R) or not np.isfinite(axis_Z):
                raise ValueError("axis gradient override must be finite")
        derivative_R[0] = axis_R
        derivative_Z[0] = axis_Z
    return derivative_R, derivative_Z, excluded


def _axis_gradient_limit(
    derivative_R: np.ndarray,
    derivative_Z: np.ndarray,
    r: np.ndarray,
    singular: np.ndarray,
) -> tuple[float, float]:
    """Extrapolate surface-mean Cartesian gradients from valid axis rings."""

    valid_rows = np.flatnonzero(~np.any(singular, axis=1))
    if valid_rows.size < 2:
        raise ValueError("axis gradient limit needs at least two valid radial rows")
    selected = valid_rows[np.argsort(r[valid_rows])[:3]]
    nodes = r[selected]
    if nodes[0] <= 0.0 or np.any(np.diff(nodes) <= 0.0):
        raise ValueError("axis gradient limit requires ordered positive rings")
    scaled = (nodes - nodes[0]) / (nodes[-1] - nodes[0])
    degree = min(2, selected.size - 1)
    design = np.vander(scaled, N=degree + 1, increasing=True)
    axis_values = []
    for gradient in (derivative_R, derivative_Z):
        surface_mean = np.mean(gradient[selected], axis=1)
        coefficients = np.linalg.solve(design, surface_mean)
        target = (0.0 - nodes[0]) / (nodes[-1] - nodes[0])
        axis_values.append(
            float(np.sum(coefficients * target ** np.arange(degree + 1)))
        )
    return axis_values[0], axis_values[1]


def _surface_mean(values: np.ndarray) -> float:
    return float(np.mean(values[-1]))


def _ampere_current(
    R: np.ndarray,
    Z: np.ndarray,
    B_R: np.ndarray,
    B_phi: np.ndarray,
    B_Z: np.ndarray,
    radial_differentiator: np.ndarray,
    r: np.ndarray,
    mu0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    B_R_R, B_R_Z, _ = _physical_gradient(
        B_R, radial_differentiator, R, Z, r=r
    )
    B_phi_R, B_phi_Z, _ = _physical_gradient(
        B_phi, radial_differentiator, R, Z, r=r
    )
    B_Z_R, B_Z_Z, _ = _physical_gradient(
        B_Z, radial_differentiator, R, Z, r=r
    )
    del B_R_R, B_Z_Z
    curl_R = -B_phi_Z
    curl_phi = B_R_Z - B_Z_R
    curl_Z = B_phi / R + B_phi_R
    return curl_R / mu0, curl_phi / mu0, curl_Z / mu0


def _integrated_profile(
    theta: np.ndarray,
    r: np.ndarray,
    values: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    radial_differentiator: np.ndarray,
) -> float:
    if theta.size < 4:
        raise ValueError("current integration requires at least four theta nodes")
    if not np.allclose(
        np.diff(theta), 2.0 * math.pi / theta.size, rtol=1.0e-12, atol=1.0e-14
    ):
        raise ValueError("current integration requires uniform theta nodes")
    R_r = radial_differentiator @ R
    Z_t = _theta_derivative(Z)
    R_t = _theta_derivative(R)
    Z_r = radial_differentiator @ Z
    jacobian = np.abs(R_r * Z_t - R_t * Z_r)
    theta_weight = 2.0 * math.pi / theta.size
    radial_values = np.sum(values * jacobian, axis=1) * theta_weight
    return float(np.trapezoid(radial_values, r))


def build_constant_f_snapshot(
    *,
    metadata: VeqSourceMetadata,
    r: np.ndarray,
    theta: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    radial_differentiator: np.ndarray,
    psi_Wb: np.ndarray,
    p_Pa: np.ndarray,
    p0_Pa: float,
    jphi_A_m2: np.ndarray,
    native_p_psi_Pa_per_Wb_rad: np.ndarray | None = None,
) -> VeqEquilibriumSnapshot:
    r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    psi = np.asarray(psi_Wb, dtype=np.float64)
    pressure_native = np.asarray(p_Pa, dtype=np.float64)
    native_jphi = np.asarray(jphi_A_m2, dtype=np.float64)
    matrix = np.asarray(radial_differentiator, dtype=np.float64)
    shape = (r.size, theta.size)
    if r.ndim != 1 or theta.ndim != 1 or R.shape != shape or Z.shape != shape:
        raise ValueError("native evaluation arrays have inconsistent (r,theta) shapes")
    if (
        psi.shape != shape
        or pressure_native.shape != shape
        or native_jphi.shape != shape
    ):
        raise ValueError("native VEQ fields must share the evaluation-grid shape")
    if matrix.shape != (r.size, r.size):
        raise ValueError("radial differentiator has an invalid shape")
    arrays = (R, Z, psi, pressure_native, native_jphi)
    if not np.all(np.isfinite(matrix)) or any(
        not np.all(np.isfinite(value)) for value in arrays
    ):
        raise ValueError("native VEQ snapshot contains nonfinite values")
    if abs(float(r[-1]) - 1.0) > 1e-12:
        raise ValueError("evaluation grid must include r=1")
    psi_edge = _surface_mean(psi) / (2.0 * math.pi)
    psi_F = psi_edge - psi / (2.0 * math.pi)
    native_p_edge = _surface_mean(pressure_native)
    metadata = replace(
        metadata, pressure_offset_Pa=metadata.p_edge_target_Pa - native_p_edge
    )
    p_total = pressure_native + metadata.pressure_offset_Pa
    psi_R, psi_Z, axis_samples = _physical_gradient(
        psi_F,
        matrix,
        R,
        Z,
        r=r,
        axis_gradient=(0.0, 0.0),
    )
    B_R = -psi_Z / R
    B_Z = psi_R / R
    B_phi = np.full(shape, metadata.F_edge_Tm) / R
    J_profile_phi = R * metadata.pprime_F_Pa_per_Wb_rad
    J_ampere_R, J_ampere_phi, J_ampere_Z = _ampere_current(
        R,
        Z,
        B_R,
        B_phi,
        B_Z,
        matrix,
        r,
        metadata.mu0_VEQ_H_per_m,
    )
    p_i = metadata.ion_pressure_fraction * p_total
    p_e = (1.0 - metadata.ion_pressure_fraction) * p_total
    charge_density = metadata.electron_charge_C * metadata.density_m3
    E_profile_R = (
        metadata.ion_pressure_fraction
        * metadata.pprime_F_Pa_per_Wb_rad
        * psi_R
        / charge_density
    )
    E_profile_Z = (
        metadata.ion_pressure_fraction
        * metadata.pprime_F_Pa_per_Wb_rad
        * psi_Z
        / charge_density
    )
    zero = np.zeros(shape, dtype=np.float64)
    u_e_profile_phi = -J_profile_phi / charge_density
    u_e_ampere_R = -J_ampere_R / charge_density
    u_e_ampere_phi = -J_ampere_phi / charge_density
    u_e_ampere_Z = -J_ampere_Z / charge_density
    smooth_wall = wall_normal_flux(
        theta, R[-1], Z[-1], B_R[-1], B_Z[-1]
    )
    midpoint_wall = wall_normal_flux_polygon_midpoint(
        R[-1], Z[-1], B_R[-1], B_Z[-1]
    )
    if native_p_psi_Pa_per_Wb_rad is None:
        native_p_psi = np.full(shape, np.nan)
        p_psi_offset = float("nan")
    else:
        native_p_psi = np.asarray(
            native_p_psi_Pa_per_Wb_rad, dtype=np.float64
        )
        if native_p_psi.ndim == 1 and native_p_psi.shape == (r.size,):
            native_p_psi = np.broadcast_to(native_p_psi[:, None], shape)
        if native_p_psi.shape != shape or not np.all(np.isfinite(native_p_psi)):
            raise ValueError("native P_psi must be finite on the evaluation grid")
        p_psi_offset = float(
            np.max(np.abs(native_p_psi + metadata.pprime_F_Pa_per_Wb_rad))
        )
    native_profile_current_residual = float(
        np.max(np.abs(native_jphi - J_profile_phi))
    )
    ampere_profile_residual = float(
        np.max(np.abs(J_ampere_phi - J_profile_phi))
    )
    I_p_profile = _integrated_profile(
        theta, r, J_profile_phi, R, Z, matrix
    )
    I_p_ampere = _integrated_profile(
        theta, r, J_ampere_phi, R, Z, matrix
    )
    one = np.ones(shape, dtype=np.float64)
    return VeqEquilibriumSnapshot(
        metadata=metadata,
        r=_readonly(r),
        theta=_readonly(theta),
        R_m=_readonly(R),
        Z_m=_readonly(Z),
        radial_differentiator=_readonly(matrix),
        psi_edge_Wb_per_rad=psi_edge,
        psi_F_Wb_per_rad=_readonly(psi_F),
        F_Tm=_readonly(one * metadata.F_edge_Tm),
        Fprime_Tm=_readonly(zero),
        B_R_T=_readonly(B_R),
        B_phi_T=_readonly(B_phi),
        B_Z_T=_readonly(B_Z),
        J_profile_R_A_m2=_readonly(zero),
        J_profile_phi_A_m2=_readonly(J_profile_phi),
        J_profile_Z_A_m2=_readonly(zero),
        J_ampere_R_A_m2=_readonly(J_ampere_R),
        J_ampere_phi_A_m2=_readonly(J_ampere_phi),
        J_ampere_Z_A_m2=_readonly(J_ampere_Z),
        I_p_profile_A=I_p_profile,
        I_p_ampere_A=I_p_ampere,
        p_native_Pa=_readonly(pressure_native),
        p_total_Pa=_readonly(p_total),
        p_i_Pa=_readonly(p_i),
        p_e_perp_Pa=_readonly(p_e),
        p_e_parallel_Pa=_readonly(p_e),
        n_m3=_readonly(one * metadata.density_m3),
        u_i_R_m_s=_readonly(zero),
        u_i_phi_m_s=_readonly(zero),
        u_i_Z_m_s=_readonly(zero),
        u_e_profile_R_m_s=_readonly(zero),
        u_e_profile_phi_m_s=_readonly(u_e_profile_phi),
        u_e_profile_Z_m_s=_readonly(zero),
        u_e_ampere_R_m_s=_readonly(u_e_ampere_R),
        u_e_ampere_phi_m_s=_readonly(u_e_ampere_phi),
        u_e_ampere_Z_m_s=_readonly(u_e_ampere_Z),
        E_profile_R_V_m=_readonly(E_profile_R),
        E_profile_phi_V_m=_readonly(zero),
        E_profile_Z_V_m=_readonly(E_profile_Z),
        native_psi_Wb=_readonly(psi),
        native_p0_Pa=float(p0_Pa),
        native_jphi_A_m2=_readonly(native_jphi),
        native_p_psi_Pa_per_Wb_rad=_readonly(native_p_psi),
        native_profile_jphi_residual_max_A_m2=(
            native_profile_current_residual
        ),
        ampere_profile_phi_max_abs_residual_A_m2=ampere_profile_residual,
        native_p_psi_offset_from_minus_pprime=p_psi_offset,
        axis_gradient_samples_regularized=axis_samples,
        axis_B_R_theta_spread_T=float(np.ptp(B_R[0])),
        axis_B_Z_theta_spread_T=float(np.ptp(B_Z[0])),
        axis_B_poloidal_max_abs_T=float(
            np.max(np.hypot(B_R[0], B_Z[0]))
        ),
        wall_normal_B_smooth_max_abs_T=smooth_wall["max_abs_T"],
        wall_normal_B_polygon_midpoint_max_abs_T=midpoint_wall["max_abs_T"],
    )


def _wall_inputs(
    theta: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    B_R: np.ndarray,
    B_Z: np.ndarray,
) -> tuple[np.ndarray, ...]:
    theta = np.asarray(theta, dtype=np.float64)
    values = tuple(np.asarray(value, dtype=np.float64) for value in (R, Z, B_R, B_Z))
    if theta.ndim != 1 or any(value.shape != theta.shape for value in values):
        raise ValueError("wall theta and R/Z/B arrays must share one 1D shape")
    finite = all(np.all(np.isfinite(value)) for value in (theta, *values))
    if theta.size < 4 or not finite:
        raise ValueError("wall inputs must be finite with at least four samples")
    return (theta, *values)


def _wall_orientation(R: np.ndarray, Z: np.ndarray) -> float:
    area2 = np.sum(R * np.roll(Z, -1) - np.roll(R, -1) * Z)
    if abs(float(area2)) <= np.finfo(np.float64).tiny:
        raise ValueError("wall polygon has ambiguous orientation")
    return 1.0 if area2 > 0.0 else -1.0


def wall_normal_flux(
    theta: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    B_R: np.ndarray,
    B_Z: np.ndarray,
) -> dict[str, float]:
    """Evaluate B.n with the smooth Fourier tangent collocated at each θ."""

    theta, R, Z, B_R, B_Z = _wall_inputs(theta, R, Z, B_R, B_Z)
    tangent_R = _theta_derivative_1d(R)
    tangent_Z = _theta_derivative_1d(Z)
    length = np.hypot(tangent_R, tangent_Z)
    if np.any(length <= 64.0 * np.finfo(np.float64).eps * max(np.max(np.abs(R)), 1.0)):
        raise ValueError("smooth wall tangent is degenerate")
    orientation = _wall_orientation(R, Z)
    normal_R = orientation * tangent_Z / length
    normal_Z = orientation * -tangent_R / length
    values = B_R * normal_R + B_Z * normal_Z
    return {
        "max_abs_T": float(np.max(np.abs(values))),
        "rms_T": float(np.sqrt(np.mean(values**2))),
        "orientation_sign": orientation,
    }


def wall_normal_flux_polygon_midpoint(
    R: np.ndarray, Z: np.ndarray, B_R: np.ndarray, B_Z: np.ndarray
) -> dict[str, float]:
    """Diagnostic only: chord normal and midpoint-averaged B per segment."""

    R = np.asarray(R, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    B_R = np.asarray(B_R, dtype=np.float64)
    B_Z = np.asarray(B_Z, dtype=np.float64)
    if R.ndim != 1 or any(value.shape != R.shape for value in (Z, B_R, B_Z)):
        raise ValueError("polygon wall arrays must share one 1D shape")
    dR = np.roll(R, -1) - R
    dZ = np.roll(Z, -1) - Z
    length = np.hypot(dR, dZ)
    orientation = _wall_orientation(R, Z)
    normal_R = orientation * dZ / length
    normal_Z = orientation * -dR / length
    midpoint_B_R = 0.5 * (B_R + np.roll(B_R, -1))
    midpoint_B_Z = 0.5 * (B_Z + np.roll(B_Z, -1))
    values = midpoint_B_R * normal_R + midpoint_B_Z * normal_Z
    return {
        "max_abs_T": float(np.max(np.abs(values))),
        "rms_T": float(np.sqrt(np.mean(values**2))),
    }


def strong_gs_metrics(snapshot: VeqEquilibriumSnapshot) -> dict[str, Any]:
    psi_R, psi_Z, _ = _physical_gradient(
        snapshot.psi_F_Wb_per_rad,
        snapshot.radial_differentiator,
        snapshot.R_m,
        snapshot.Z_m,
        r=snapshot.r,
        axis_gradient=(0.0, 0.0),
    )
    psi_RR, psi_RZ, _ = _physical_gradient(
        psi_R,
        snapshot.radial_differentiator,
        snapshot.R_m,
        snapshot.Z_m,
        r=snapshot.r,
    )
    psi_ZR, psi_ZZ, _ = _physical_gradient(
        psi_Z,
        snapshot.radial_differentiator,
        snapshot.R_m,
        snapshot.Z_m,
        r=snapshot.r,
    )
    strong = psi_RR - psi_R / snapshot.R_m + psi_ZZ
    rhs = (
        -snapshot.metadata.mu0_VEQ_H_per_m
        * snapshot.R_m**2
        * snapshot.metadata.pprime_F_Pa_per_Wb_rad
    )
    residual = strong - rhs
    rr = snapshot.r[:, None]
    masks = {
        "interior": (rr >= 0.2) & (rr <= 0.75),
        "near_wall": (rr >= 0.8) & (rr < 1.0),
    }
    scale = max(float(np.max(np.abs(rhs))), np.finfo(np.float64).tiny)
    valid_rows = snapshot.r > 0.0
    output: dict[str, Any] = {
        "mutual_derivative_max_abs": float(
            np.max(np.abs(psi_RZ[valid_rows] - psi_ZR[valid_rows]))
        ),
        "rhs_scale_T": scale,
        "axis": {
            "excluded_sample_count": int(snapshot.theta.size),
            "physical_point_count": 1,
            "flux_value_theta_spread": float(
                np.ptp(snapshot.psi_F_Wb_per_rad[0])
            ),
            "reconstructed_B_R_theta_spread_T": (
                snapshot.axis_B_R_theta_spread_T
            ),
            "reconstructed_B_Z_theta_spread_T": (
                snapshot.axis_B_Z_theta_spread_T
            ),
            "reconstructed_Bpol_max_abs_T": (
                snapshot.axis_B_poloidal_max_abs_T
            ),
            "gradient_regularization": (
                "known zero flux gradient; radial limit for other fields"
            ),
        },
        "valid_collocation": {
            "radial_row_count": int(np.count_nonzero(valid_rows)),
            "sample_count": int(np.count_nonzero(valid_rows)) * snapshot.theta.size,
        },
    }
    for name, mask in masks.items():
        values = residual[np.broadcast_to(mask, residual.shape)]
        if values.size == 0:
            raise ValueError(f"strong-GS {name} band contains no samples")
        output[name] = {
            "sample_count": int(values.size),
            "max_abs_T": float(np.max(np.abs(values))),
            "rms_T": float(np.sqrt(np.mean(values**2))),
            "normalized_max": float(np.max(np.abs(values)) / scale),
            "normalized_rms": float(np.sqrt(np.mean(values**2)) / scale),
        }
    output["wall_normal_B_smooth"] = wall_normal_flux(
        snapshot.theta,
        snapshot.R_m[-1],
        snapshot.Z_m[-1],
        snapshot.B_R_T[-1],
        snapshot.B_Z_T[-1],
    )
    output["wall_normal_B_polygon_midpoint"] = (
        wall_normal_flux_polygon_midpoint(
            snapshot.R_m[-1],
            snapshot.Z_m[-1],
            snapshot.B_R_T[-1],
            snapshot.B_Z_T[-1],
        )
    )
    return output


def write_equilibrium_product(
    prefix: str | Path, snapshot: VeqEquilibriumSnapshot
) -> tuple[Path, Path]:
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "r": snapshot.r,
        "theta": snapshot.theta,
        "R_m": snapshot.R_m,
        "Z_m": snapshot.Z_m,
        "radial_differentiator": snapshot.radial_differentiator,
        "psi_F_Wb_per_rad": snapshot.psi_F_Wb_per_rad,
        "F_Tm": snapshot.F_Tm,
        "B_R_T": snapshot.B_R_T,
        "B_phi_T": snapshot.B_phi_T,
        "B_Z_T": snapshot.B_Z_T,
        "J_profile_R_A_m2": snapshot.J_profile_R_A_m2,
        "J_profile_phi_A_m2": snapshot.J_profile_phi_A_m2,
        "J_profile_Z_A_m2": snapshot.J_profile_Z_A_m2,
        "J_ampere_R_A_m2": snapshot.J_ampere_R_A_m2,
        "J_ampere_phi_A_m2": snapshot.J_ampere_phi_A_m2,
        "J_ampere_Z_A_m2": snapshot.J_ampere_Z_A_m2,
        "p_native_Pa": snapshot.p_native_Pa,
        "p_total_Pa": snapshot.p_total_Pa,
        "p_i_Pa": snapshot.p_i_Pa,
        "p_e_perp_Pa": snapshot.p_e_perp_Pa,
        "p_e_parallel_Pa": snapshot.p_e_parallel_Pa,
        "n_m3": snapshot.n_m3,
        "u_i_R_m_s": snapshot.u_i_R_m_s,
        "u_i_phi_m_s": snapshot.u_i_phi_m_s,
        "u_i_Z_m_s": snapshot.u_i_Z_m_s,
        "u_e_profile_R_m_s": snapshot.u_e_profile_R_m_s,
        "u_e_profile_phi_m_s": snapshot.u_e_profile_phi_m_s,
        "u_e_profile_Z_m_s": snapshot.u_e_profile_Z_m_s,
        "u_e_ampere_R_m_s": snapshot.u_e_ampere_R_m_s,
        "u_e_ampere_phi_m_s": snapshot.u_e_ampere_phi_m_s,
        "u_e_ampere_Z_m_s": snapshot.u_e_ampere_Z_m_s,
        "E_profile_R_V_m": snapshot.E_profile_R_V_m,
        "E_profile_phi_V_m": snapshot.E_profile_phi_V_m,
        "E_profile_Z_V_m": snapshot.E_profile_Z_V_m,
        "native_psi_Wb": snapshot.native_psi_Wb,
        "native_jphi_A_m2": snapshot.native_jphi_A_m2,
        "native_p_psi_Pa_per_Wb_rad": snapshot.native_p_psi_Pa_per_Wb_rad,
    }
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")
    np.savez_compressed(npz_path, **arrays)
    metadata = snapshot.metadata.as_dict()
    metadata.update(
        {
            "psi_edge_Wb_per_rad": snapshot.psi_edge_Wb_per_rad,
            "native_p0_Pa": snapshot.native_p0_Pa,
            "native_profile_jphi_residual_max_A_m2": (
                snapshot.native_profile_jphi_residual_max_A_m2
            ),
            "ampere_profile_phi_max_abs_residual_A_m2": (
                snapshot.ampere_profile_phi_max_abs_residual_A_m2
            ),
            "I_p_profile_A": snapshot.I_p_profile_A,
            "I_p_ampere_A": snapshot.I_p_ampere_A,
            "native_p_psi_offset_from_minus_pprime": (
                snapshot.native_p_psi_offset_from_minus_pprime
            ),
            "axis_gradient_samples_regularized": (
                snapshot.axis_gradient_samples_regularized
            ),
            "axis_B_R_theta_spread_T": snapshot.axis_B_R_theta_spread_T,
            "axis_B_Z_theta_spread_T": snapshot.axis_B_Z_theta_spread_T,
            "axis_B_poloidal_max_abs_T": snapshot.axis_B_poloidal_max_abs_T,
            "wall_normal_B_smooth_max_abs_T": (
                snapshot.wall_normal_B_smooth_max_abs_T
            ),
            "wall_normal_B_polygon_midpoint_max_abs_T": (
                snapshot.wall_normal_B_polygon_midpoint_max_abs_T
            ),
            "strong_gs": strong_gs_metrics(snapshot),
        }
    )
    json_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return npz_path, json_path
