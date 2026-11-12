"""Independent linear cold Hall standing wave on a conducting RZ box.

This analysis module has no dependency on the FOMHD solver. SI units, a
positive singly charged deuteron, and zero physical pressure are used.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.constants import elementary_charge, mu_0, physical_constants
from scipy.optimize import brentq
from scipy.special import j0, j1, y0, y1


@dataclass(frozen=True)
class Mode:
    r: tuple[float, float]
    z: tuple[float, float]
    b0: float
    ion_mass: float
    charge: float
    mu0: float
    kappa: float
    k: tuple[float, float]
    ion_inertial_length: float
    density: float
    alfven_speed: float
    omega: float
    weights: tuple[float, float]
    toroidal_weights: tuple[complex, complex]
    toroidal_peak: float

    @property
    def period(self):
        return 2.0 * np.pi / self.omega


def _radial(r, r_lo, kappa):
    left = kappa * r_lo
    x = kappa * r
    return (
        r * (j1(x) * y1(left) - y1(x) * j1(left)),
        x * (j0(x) * y1(left) - y0(x) * j1(left)),
    )


@lru_cache(maxsize=8)
def make_mode(
    r=(2.4, 3.6),
    z=(-0.6, 0.6),
    b0=2.0,
    axial_modes=(1, 3),
    ion_mass=physical_constants["deuteron mass"][0],
    charge=elementary_charge,
    mu0=mu_0,
):
    """Return a wall-conforming cold mode with a density tuned to the box.

    Use two increasing positive longitudinal mode numbers of the same
    parity. The first radial Dirichlet root and the requested longitudinal
    pair determine the density. This density cannot be chosen independently
    of box dimensions and mode numbers while retaining the exact solution.
    """
    r_lo, r_hi = r
    z_lo, z_hi = z
    m1, m2 = axial_modes
    if not (0 < r_lo < r_hi and z_lo < z_hi and b0 > 0 and ion_mass > 0 and charge > 0 and mu0 > 0):
        raise ValueError("invalid box or physical parameters")
    if not (isinstance(m1, int) and isinstance(m2, int) and 0 < m1 < m2 and (m2 - m1) % 2 == 0):
        raise ValueError("longitudinal mode numbers must be increasing positive integers of the same parity")

    # For f=sqrt(R)y, the radial potential is 3/(4 R^2), which brackets
    # the first radial eigenvalue above the first constant-metric root.
    lower = np.pi / (r_hi - r_lo)
    upper = np.sqrt(lower**2 + 0.75 / r_lo**2)
    kappa = brentq(lambda k: _radial(r_hi, r_lo, k)[0], lower, upper, xtol=5e-15)
    k = np.pi * np.asarray(axial_modes) / (z_hi - z_lo)
    x1, x2 = k**2
    if x2 - x1 <= kappa**2:
        raise ValueError("the longitudinal separation must exceed kappa squared")
    total = x1 + x2 + kappa**2
    xi = kappa**2 + 2.0 * x1 * x2 / total
    di = np.sqrt((1.0 - 2.0 * xi / total) / xi)
    density = ion_mass / (mu0 * charge**2 * di**2)
    alfven = b0 / np.sqrt(mu0 * ion_mass * density)
    omega = alfven * np.sqrt(xi)
    dispersion = (xi - k**2 - kappa**2) * (xi - k**2) - xi * di**2 * k**2 * (k**2 + kappa**2)
    if np.max(np.abs(dispersion)) > 256 * np.finfo(float).eps * total**2:
        raise ArithmeticError("the selected wave numbers do not share the Hall frequency")

    # P is real and Q is imaginary. These weights cancel Q at both endcaps.
    q = (omega**2 - alfven**2 * (k**2 + kappa**2)) / (omega * alfven * di * k)
    a = np.array([q[1], -q[0]]) / np.max(np.abs(q))
    b = -1j * a * q
    r_sample = np.linspace(r_lo, r_hi, 4097)
    z_sample = np.linspace(0.0, z_hi - z_lo, 4097)
    radial_peak = np.max(np.abs(_radial(r_sample, r_lo, kappa)[0] / r_sample))
    axial_peak = np.max(np.abs(b[0] * np.cos(k[0] * z_sample) + b[1] * np.cos(k[1] * z_sample)))
    return Mode(
        r, z, b0, ion_mass, charge, mu0, kappa, tuple(map(float, k)), di,
        density, alfven, omega, tuple(map(float, a)), tuple(map(complex, b)),
        radial_peak * axial_peak,
    )


def wave(r, z, t=0.0, *, phase=0.0, amplitude=1e-4, mode=None):
    """Sample physical fields with phase convention Re[exp(i phase - i omega t)].

    R, Z, t, and phase broadcast. Vector components are (R, phi, Z) on the
    last axis. The continuous peak envelope of |delta B_phi| is amplitude*B0
    to the sampling accuracy of its normalization. The total stream function
    psi has zero gauge on R=0 for the background. Its R and Z endpoint
    differences give oriented physical poloidal flux per radian.

    The cold oracle has zero pressure perturbations. Constant positive floors
    may be added by a solver comparison, which then tests a finite-pressure
    approximation to the oracle.
    """
    mode = make_mode() if mode is None else mode
    r, z, t, phase = np.broadcast_arrays(r, z, t, phase)
    f, df = _radial(r, mode.r[0], mode.kappa)
    k = np.asarray(mode.k)
    arg = (z - mode.z[0])[..., None] * k
    sine = np.sin(arg)
    cosine = np.cos(arg)
    p = np.sum(sine * mode.weights, axis=-1)
    dp = np.sum(cosine * k * mode.weights, axis=-1)
    operator_p = -np.sum(sine * (k**2 + mode.kappa**2) * mode.weights, axis=-1)
    q = np.sum(cosine * mode.toroidal_weights, axis=-1)
    dq = -np.sum(sine * k * mode.toroidal_weights, axis=-1)
    zero = np.zeros_like(r)
    harmonic = (amplitude * mode.b0 / mode.toroidal_peak) * np.exp(1j * (phase - mode.omega * t))

    b_hat = np.stack((-f * dp, f * q, df * p), axis=-1) / r[..., None]
    j_hat = np.stack((-f * dq, -f * operator_p, df * q), axis=-1) / (mode.mu0 * r[..., None])
    force_to_velocity = 1j * mode.b0 / (mode.mu0 * mode.ion_mass * mode.density * mode.omega)
    u_hat = force_to_velocity * np.stack((-f * operator_p, f * dq, zero), axis=-1) / r[..., None]
    div_u_hat = -force_to_velocity * df * operator_p / r
    density_perturbation = np.real(-1j * mode.density / mode.omega * div_u_hat * harmonic)

    delta_b = np.real(b_hat * harmonic[..., None])
    current = np.real(j_hat * harmonic[..., None])
    ion = np.real(u_hat * harmonic[..., None])
    electron = ion - current / (mode.charge * mode.density)
    background_b = np.array([0.0, 0.0, mode.b0])
    electric = -np.cross(electron, background_b)
    delta_psi = np.real(f * p * harmonic)
    return {
        "B": background_b + delta_b,
        "delta_B": delta_b,
        "ion_velocity": ion,
        "electron_velocity": electron,
        "electric": electric,
        "current": current,
        "density": mode.density + density_perturbation,
        "delta_density": density_perturbation,
        "psi": 0.5 * mode.b0 * r**2 + delta_psi,
        "delta_psi": delta_psi,
        "b": np.real(f * q * harmonic),
        "delta_p_ion": zero,
        "delta_p_e_perp": zero,
        "delta_p_e_parallel": zero,
    }
