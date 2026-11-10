"""Linearized n=0 straight-cylinder Hall/CGL compressional reference.

The equations are linearized, not the finite-amplitude nonlinear system.
Default constants copy documented source literals without importing FOMHD;
callers may supply the actual context ``mi``, ``mu0``, and ``gamma``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.special import j0, j1, jn_zeros


J11 = float(jn_zeros(1, 1)[0])
SCOPE = (
    "linearized n=0 straight-cylinder Hall/CGL reference; not a "
    "finite-amplitude nonlinear system; not a finite-R0 toroidal "
    "eigenfrequency, and not an n=1 solution"
)
DEFAULTS = {
    "a": 0.5,
    "B0": 2.0,
    "n0": 2.0e19,
    "mi": 2.0 * 1.67262192595e-27,
    "pi0": 600.0,
    "pe0": 400.0,
    "gamma": 5.0 / 3.0,
    "mu0": 1.25663706127e-6,
    "e": 1.602176634e-19,
    "A": 1.0e-3,
}


def _positive(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _nonnegative(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _coordinates(R, Z) -> tuple[np.ndarray, np.ndarray]:
    try:
        radius, axial = np.broadcast_arrays(
            np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("R and Z must be numeric broadcast-compatible arrays") from error
    if radius.size == 0 or not np.isfinite(radius).all() or not np.isfinite(axial).all():
        raise ValueError("R and Z must be nonempty and finite")
    return radius, axial


@dataclass(frozen=True)
class CylinderCAW:
    """Linearized cylinder parameters; explicit caller SI values override defaults."""

    a: float = DEFAULTS["a"]
    B0: float = DEFAULTS["B0"]
    n0: float = DEFAULTS["n0"]
    mi: float = DEFAULTS["mi"]
    pi0: float = DEFAULTS["pi0"]
    pe0: float = DEFAULTS["pe0"]
    gamma: float = DEFAULTS["gamma"]
    mu0: float = DEFAULTS["mu0"]
    e: float = DEFAULTS["e"]
    A: float = DEFAULTS["A"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", _positive(self.a, "a"))
        object.__setattr__(self, "B0", _positive(self.B0, "B0"))
        object.__setattr__(self, "n0", _positive(self.n0, "n0"))
        object.__setattr__(self, "mi", _positive(self.mi, "mi"))
        object.__setattr__(self, "pi0", _nonnegative(self.pi0, "pi0"))
        object.__setattr__(self, "pe0", _nonnegative(self.pe0, "pe0"))
        object.__setattr__(self, "gamma", _positive(self.gamma, "gamma"))
        object.__setattr__(self, "mu0", _positive(self.mu0, "mu0"))
        object.__setattr__(self, "e", _positive(self.e, "e"))
        amplitude = float(self.A)
        if not math.isfinite(amplitude) or not 0.0 <= amplitude <= 1.0e-3:
            raise ValueError("amplitude A must be finite and in [0, 1e-3]")
        object.__setattr__(self, "A", amplitude)

    @property
    def j11(self) -> float:
        return J11

    @property
    def kappa(self) -> float:
        return self.j11 / self.a

    @property
    def vA(self) -> float:
        return self.B0 / math.sqrt(self.mu0 * self.mi * self.n0)

    @property
    def cfast(self) -> float:
        return math.sqrt(
            (
                self.B0**2 / self.mu0
                + self.gamma * self.pi0
                + 2.0 * self.pe0
            )
            / (self.mi * self.n0)
        )

    @property
    def omega(self) -> float:
        return self.kappa * self.cfast

    @property
    def period(self) -> float:
        return 2.0 * math.pi / self.omega

    def _local_coordinates(
        self, R, Z, R0: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        radius, axial = _coordinates(R, Z)
        center = float(R0)
        if not math.isfinite(center):
            raise ValueError("R0 must be finite")
        local_R = radius - center
        r = np.hypot(local_R, axial)
        tolerance = 64.0 * np.finfo(np.float64).eps * self.a
        if np.any(r > self.a + tolerance):
            raise ValueError("sample coordinates lie outside the circular cylinder")
        angle = np.arctan2(axial, local_R)
        return local_R, axial, r, angle

    def sample(self, R, Z, time: float, R0: float = 3.0) -> dict[str, np.ndarray]:
        """Return physical fields at local time ``time`` in seconds."""
        moment = float(time)
        if not math.isfinite(moment):
            raise ValueError("time must be finite and in seconds")
        local_R, axial, r, angle = self._local_coordinates(R, Z, R0)
        kr = self.kappa * r
        cosine = math.cos(self.omega * moment)
        sine = math.sin(self.omega * moment)
        j1r = j1(kr)
        j0r = j0(kr)
        velocity_amplitude = self.A * self.vA
        u_radial = velocity_amplitude * j1r * cosine
        divergence = self.kappa * velocity_amplitude * j0r * cosine
        compression_factor = (
            -self.n0 * self.kappa * velocity_amplitude / self.omega
            * j0r
            * sine
        )
        epsilon = 64.0 * np.finfo(np.float64).eps * self.a
        axis = r <= epsilon
        radial_unit_R = np.ones_like(r)
        radial_unit_Z = np.zeros_like(r)
        radial_unit_R[~axis] = local_R[~axis] / r[~axis]
        radial_unit_Z[~axis] = axial[~axis] / r[~axis]
        u_R = u_radial * radial_unit_R
        u_Z = u_radial * radial_unit_Z
        u_R_over_r = np.empty_like(r)
        u_R_over_r[axis] = velocity_amplitude * self.kappa / 2.0 * cosine
        u_R_over_r[~axis] = u_radial[~axis] / r[~axis]
        delta_pe_perp = 2.0 * self.pe0 * compression_factor / self.n0
        delta_pe_parallel = self.pe0 * compression_factor / self.n0
        delta_B = self.B0 * compression_factor / self.n0
        zero = np.zeros_like(r)
        return {
            "R": local_R + float(R0),
            "Z": axial,
            "r": r,
            "theta_geom": angle,
            "u_R": u_R,
            "u_phi": zero.copy(),
            "u_Z": u_Z,
            "u_radial": u_radial,
            "u_R_over_r": u_R_over_r,
            "u_guide_tor": zero.copy(),
            "theta": divergence,
            "delta_n": compression_factor,
            "delta_p_i": (
                self.gamma * self.pi0 * compression_factor / self.n0
            ),
            "delta_pe_perp": delta_pe_perp,
            "delta_pe_parallel": delta_pe_parallel,
            "delta_B_guide": delta_B,
            "B_R_perturbation": zero.copy(),
            "B_phi_perturbation": delta_B.copy(),
            "B_Z_perturbation": zero.copy(),
            "n_total": self.n0 + compression_factor,
            "p_i_total": self.pi0
            + self.gamma * self.pi0 * compression_factor / self.n0,
            "pe_perp_total": self.pe0 + delta_pe_perp,
            "pe_parallel_total": self.pe0 + delta_pe_parallel,
            "B_guide_total": self.B0 + delta_B,
            "psi": zero.copy(),
            "cos_phase": np.full_like(r, cosine),
            "sin_phase": np.full_like(r, sine),
        }

    def linear_pde_residuals(
        self, R, Z, time: float, R0: float = 3.0
    ) -> dict[str, np.ndarray]:
        """Return analytic-symbol linear continuity/CGL/guide-field residuals."""
        state = self.sample(R, Z, time, R0)
        cosine = state["cos_phase"]
        j0r = j0(self.kappa * state["r"])
        divergence = self.kappa * self.A * self.vA * j0r * cosine
        density_rate = (
            -self.n0
            * self.kappa
            * self.A
            * self.vA
            * j0r
            * cosine
        )
        ion_rate = (
            -self.gamma
            * self.pi0
            * self.kappa
            * self.A
            * self.vA
            * j0r
            * cosine
        )
        electron_perp_rate = (
            -2.0
            * self.pe0
            * self.kappa
            * self.A
            * self.vA
            * j0r
            * cosine
        )
        electron_parallel_rate = (
            -self.pe0
            * self.kappa
            * self.A
            * self.vA
            * j0r
            * cosine
        )
        guide_rate = (
            -self.B0
            * self.kappa
            * self.A
            * self.vA
            * j0r
            * cosine
        )
        return {
            "continuity": density_rate + self.n0 * divergence,
            "ion_pressure": ion_rate + self.gamma * self.pi0 * divergence,
            "electron_perp": electron_perp_rate + 2.0 * self.pe0 * divergence,
            "electron_parallel": electron_parallel_rate + self.pe0 * divergence,
            "guide_field": guide_rate + self.B0 * divergence,
            "bb_grad_u": np.zeros_like(divergence),
            "div_ue_minus_div_ui": np.zeros_like(divergence),
        }

    def boundary_checks(
        self, time, *, count: int = 128, R0: float = 3.0
    ) -> dict[str, float]:
        moments = np.asarray(time, dtype=np.float64).reshape(-1)
        if moments.size == 0 or not np.isfinite(moments).all():
            raise ValueError("time samples must be nonempty and finite")
        if isinstance(count, bool) or int(count) != count or int(count) < 8:
            raise ValueError("count must be an integer of at least 8")
        angle = np.linspace(0.0, 2.0 * np.pi, int(count), endpoint=False)
        wall_R = float(R0) + self.a * np.cos(angle)
        wall_Z = self.a * np.sin(angle)
        max_u = 0.0
        max_b_variation = 0.0
        max_p_variation = 0.0
        for moment in moments:
            state = self.sample(wall_R, wall_Z, float(moment), R0)
            max_u = max(max_u, float(np.max(np.abs(state["u_radial"]))))
            max_b_variation = max(
                max_b_variation,
                float(np.ptp(state["B_guide_total"])),
            )
            max_p_variation = max(
                max_p_variation,
                float(np.ptp(state["p_i_total"])),
                float(np.ptp(state["pe_perp_total"])),
                float(np.ptp(state["pe_parallel_total"])),
            )
        return {
            "j1_root": float(j1(self.kappa * self.a)),
            "max_abs_u_radial": max_u,
            "max_abs_delta_B_guide_variation": max_b_variation,
            "max_abs_delta_p_variation": max_p_variation,
            "boundary_flux": 0.0,
            "normal_B_perturbation": 0.0,
            "normal_current": 0.0,
            "tangential_electric_field": 0.0,
            "curl_electric_field": 0.0,
            "bb_grad_u": 0.0,
        }

    def project_velocity_amplitude(
        self, velocity, R, Z, *, R0: float = 3.0, M0=None
    ) -> dict[str, float]:
        """Project physical (R,phi,Z) velocity onto the J1 radial mode."""
        values = np.asarray(velocity, dtype=np.float64)
        local_R, axial, r, _ = self._local_coordinates(R, Z, R0)
        if values.shape != (r.size, 3):
            raise ValueError("velocity must have shape (N, 3)")
        if not np.isfinite(values).all():
            raise ValueError("velocity must be finite")
        weights = _weights(M0, r.size)
        epsilon = 64.0 * np.finfo(np.float64).eps * self.a
        radial_component = np.empty_like(r)
        axis = r <= epsilon
        radial_component[axis] = values[axis, 0]
        radial_component[~axis] = (
            local_R[~axis] * values[~axis, 0]
            + axial[~axis] * values[~axis, 2]
        ) / r[~axis]
        basis = j1(self.kappa * r)
        denominator = float(np.sum(weights * basis**2))
        if denominator <= 0.0:
            raise ValueError("velocity projection basis has zero weighted norm")
        coefficient = float(np.sum(weights * radial_component * basis) / denominator)
        return {
            "coefficient": coefficient,
            "fractional_amplitude": coefficient / self.vA,
            "vA": self.vA,
            "basis_weight": denominator,
        }

    def project_scalar(
        self, values, R, Z, *, R0: float = 3.0, M0=None
    ) -> dict[str, float]:
        """Project a scalar perturbation onto J0(kappa*r)."""
        samples = np.asarray(values, dtype=np.float64)
        _, _, r, _ = self._local_coordinates(R, Z, R0)
        if samples.shape != r.shape:
            raise ValueError("scalar values must match the R/Z sample shape")
        if not np.isfinite(samples).all():
            raise ValueError("scalar values must be finite")
        weights = _weights(M0, r.size)
        basis = j0(self.kappa * r)
        denominator = float(np.sum(weights * basis**2))
        if denominator <= 0.0:
            raise ValueError("scalar projection basis has zero weighted norm")
        coefficient = float(np.sum(weights * samples * basis) / denominator)
        return {
            "coefficient": coefficient,
            "basis_weight": denominator,
        }

    def large_aspect_comparison(
        self, aspects=(6.0, 15.0, 30.0, 60.0)
    ) -> list[dict[str, float | str | bool]]:
        """Report declared toroidal comparison scales, not a mode frequency."""
        rows = []
        cyclotron = self.e * self.B0 / self.mi
        rho_i = self.vA / cyclotron
        for value in aspects:
            aspect = float(value)
            if not math.isfinite(aspect) or aspect <= 1.0:
                raise ValueError("aspect ratios must be finite and greater than one")
            curvature = 1.0 / aspect
            hall_ratio = self.omega / cyclotron
            rows.append(
                {
                    "aspect_ratio": aspect,
                    "R0_m": aspect * self.a,
                    "straight_cylinder_omega_rad_s": self.omega,
                    "straight_cylinder_period_s": self.period,
                    "curvature_scale": curvature,
                    "guide_field_relative_variation": curvature,
                    "hall_frequency_ratio": hall_ratio,
                    "curvature_frequency_correction_scale_rad_s": (
                        self.omega * curvature
                    ),
                    "hall_frequency_correction_scale_rad_s": (
                        self.omega * hall_ratio
                    ),
                    "flr_parameter": self.kappa * rho_i,
                    "mi_kg": self.mi,
                    "mu0_H_m": self.mu0,
                    "gamma": self.gamma,
                    "frequency_predicted_for_torus": False,
                    "requires_actual_spatial_convergence": True,
                    "scope": (
                        "linearized declared large-aspect ordering only; not "
                        "a finite-amplitude nonlinear or toroidal "
                        "eigenfrequency and not a method-order claim"
                    ),
                }
            )
        return rows


def _weights(value, size: int) -> np.ndarray:
    if value is None:
        return np.ones(size, dtype=np.float64)
    weights = np.asarray(value, dtype=np.float64).reshape(-1)
    if weights.size != size:
        raise ValueError("M0 weights must match the sample count")
    if not np.isfinite(weights).all() or np.any(weights < 0.0) or np.sum(weights) <= 0.0:
        raise ValueError("M0 weights must be finite, nonnegative, and nonzero")
    return weights


def seed_velocity(
    R,
    Z,
    R0: float = 3.0,
    a: float = DEFAULTS["a"],
    amplitude: float = DEFAULTS["A"],
    *,
    B0: float = DEFAULTS["B0"],
    n0: float = DEFAULTS["n0"],
    mi: float = DEFAULTS["mi"],
    pi0: float = DEFAULTS["pi0"],
    pe0: float = DEFAULTS["pe0"],
    gamma: float = DEFAULTS["gamma"],
    mu0: float = DEFAULTS["mu0"],
    e: float = DEFAULTS["e"],
) -> np.ndarray:
    """Return the linearized physical (u_R,u_phi,u_Z) t=0 seed velocity."""
    model = CylinderCAW(
        a=a,
        B0=B0,
        n0=n0,
        mi=mi,
        pi0=pi0,
        pe0=pe0,
        gamma=gamma,
        mu0=mu0,
        e=e,
        A=amplitude,
    )
    state = model.sample(R, Z, 0.0, R0)
    return np.column_stack((state["u_R"], state["u_phi"], state["u_Z"]))


def fit_angular_frequency(
    times, values, *, omega_guess: float
) -> dict[str, float | int]:
    """Fit cos/sin/offset on the actual supplied samples only."""
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    if t.size != y.size or t.size < 5:
        raise ValueError("times and values need at least five matching samples")
    if not np.isfinite(t).all() or not np.isfinite(y).all() or np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be finite, strictly increasing, with finite values")
    guess = _positive(omega_guess, "omega_guess")
    matrix = np.column_stack(
        (np.cos(guess * t), np.sin(guess * t), np.ones_like(t))
    )
    initial_coefficients, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
    if rank != 3:
        raise ValueError("frequency fit initial sinusoid basis is rank deficient")
    initial = np.concatenate(([guess], initial_coefficients))
    amplitude_scale = max(float(np.max(np.abs(y))), 1.0e-300)

    def residual(parameters):
        omega, cosine_coefficient, sine_coefficient, offset = parameters
        return (
            cosine_coefficient * np.cos(omega * t)
            + sine_coefficient * np.sin(omega * t)
            + offset
            - y
        )

    def jacobian(parameters):
        omega, cosine_coefficient, sine_coefficient, _ = parameters
        phase = omega * t
        return np.column_stack(
            (
                -cosine_coefficient * t * np.sin(phase)
                + sine_coefficient * t * np.cos(phase),
                np.cos(phase),
                np.sin(phase),
                np.ones_like(t),
            )
        )

    result = least_squares(
        residual,
        initial,
        jac=jacobian,
        bounds=([0.5 * guess, -np.inf, -np.inf, -np.inf],
                [1.5 * guess, np.inf, np.inf, np.inf]),
        x_scale=[guess, amplitude_scale, amplitude_scale, amplitude_scale],
        xtol=1.0e-14,
        ftol=1.0e-14,
        gtol=1.0e-14,
        max_nfev=1000,
    )
    if not result.success:
        raise RuntimeError(f"frequency fit failed: {result.message}")
    omega, cosine_coefficient, sine_coefficient, offset = result.x
    residual_values = residual(result.x)
    return {
        "omega": float(omega),
        "cos_coefficient": float(cosine_coefficient),
        "sin_coefficient": float(sine_coefficient),
        "offset": float(offset),
        "sample_count": int(t.size),
        "rmse": float(np.sqrt(np.mean(residual_values**2))),
        "maximum_abs_residual": float(np.max(np.abs(residual_values))),
        "time_start_s": float(t[0]),
        "time_end_s": float(t[-1]),
    }
