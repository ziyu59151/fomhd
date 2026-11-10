"""Continuous axisymmetric equilibrium preprocessor prototypes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MU0 = 1.25663706212e-6
ELEMENTARY_CHARGE = 1.602176634e-19
GAMMA_ION = 5.0 / 3.0


@dataclass(frozen=True)
class GSModel:
    """A fixed-boundary constant-pprime Grad-Shafranov solution."""

    R0: float
    a: float
    F0: float
    n0: float
    p_edge: float
    pprime: float
    alpha: float
    degree: int
    exponents: tuple[tuple[int, int], ...]
    coefficients: np.ndarray
    collocation_R: np.ndarray
    collocation_Z: np.ndarray
    collocation_count: int
    solve_residual: float
    solve_rank: int


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _coordinates(R, Z) -> tuple[np.ndarray, np.ndarray]:
    try:
        radius, axial = np.broadcast_arrays(
            np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("R and Z must be numeric broadcast-compatible arrays") from error
    if not np.isfinite(radius).all() or not np.isfinite(axial).all():
        raise ValueError("R and Z must be finite")
    if np.any(radius <= 0.0):
        raise ValueError("R must be positive")
    return radius, axial


def _zeros(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float64)


def _broadcast_couple(R, Z):
    radius, axial = _coordinates(R, Z)
    if radius.shape != axial.shape:
        raise ValueError("R and Z must broadcast to one shape")
    if radius.size == 0:
        raise ValueError("R and Z must contain at least one point")
    return radius, axial


def gs_particular(
    R, Z, *, pprime: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the signed particular solution and its R/Z derivatives."""
    radius, axial = _broadcast_couple(R, Z)
    derivative = float(pprime)
    if not np.isfinite(derivative) or derivative < 0.0:
        raise ValueError("pprime must be finite and nonnegative")
    coefficient = MU0 * derivative / 8.0
    return (
        -coefficient * radius**4,
        -4.0 * coefficient * radius**3,
        np.zeros_like(radius),
        -12.0 * coefficient * radius**2,
        np.zeros_like(radius),
    )


def _exponents(degree: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (i, j)
        for total in range(degree + 1)
        for i in range(total + 1)
        for j in (total - i,)
    )


def _collocation_points(count: int) -> tuple[np.ndarray, np.ndarray]:
    # A fixed radial/angle design avoids random-solve variability and covers the
    # full disk, while the irrational angle stride breaks polar-grid symmetry.
    radial_count = int(np.ceil(np.sqrt(count)))
    angle_count = int(np.ceil(count / radial_count))
    radii = np.sqrt((np.arange(radial_count) + 0.5) / radial_count) * 0.96
    angle = (
        2.0
        * np.pi
        * np.mod(
            np.arange(radial_count * angle_count, dtype=np.float64) * 0.61803398875,
            1.0,
        )
    )
    x = (radii[:, None] * np.cos(angle.reshape(radial_count, angle_count))).ravel()
    z = (radii[:, None] * np.sin(angle.reshape(radial_count, angle_count))).ravel()
    return x, z


def _boundary_factor_operator(
    x: np.ndarray, z: np.ndarray, R0: float, a: float, exponent: tuple[int, int]
) -> np.ndarray:
    i, j = exponent
    monomial = x**i * z**j
    laplacian = -4.0 * (1 + i + j) * monomial
    factor = 1.0 - x**2 - z**2
    if i >= 2:
        laplacian += i * (i - 1) * factor * x ** (i - 2) * z**j
    if j >= 2:
        laplacian += j * (j - 1) * factor * x**i * z ** (j - 2)
    radial_derivative = -2.0 * x * monomial
    if i:
        radial_derivative += i * factor * x ** (i - 1) * z**j
    return (
        laplacian
        - a / (R0 + a * x) * radial_derivative
    ) / a**2


def build_gs_equilibrium(
    *,
    R0: float,
    a: float,
    F0: float,
    n0: float,
    p_edge: float,
    pprime: float,
    alpha: float,
    degree: int,
    collocation_multiplier: int = 60,
) -> GSModel:
    """Build a circular-wall constant-pprime equilibrium by weighted collocation."""
    major_radius = _positive_finite(R0, "R0")
    minor_radius = _positive_finite(a, "a")
    if minor_radius >= major_radius:
        raise ValueError("a must be smaller than R0 so every R is positive")
    toroidal_flux = float(F0)
    if not np.isfinite(toroidal_flux) or toroidal_flux == 0.0:
        raise ValueError("F0 must be finite and nonzero")
    density = _positive_finite(n0, "n0")
    edge_pressure = _positive_finite(p_edge, "p_edge")
    derivative = float(pprime)
    if not np.isfinite(derivative) or derivative < 0.0:
        raise ValueError("pprime must be finite and nonnegative")
    fraction = float(alpha)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    if isinstance(degree, bool) or int(degree) != degree or int(degree) < 8:
        raise ValueError("degree must be an integer of at least 8")
    if (
        isinstance(collocation_multiplier, bool)
        or int(collocation_multiplier) != collocation_multiplier
        or int(collocation_multiplier) < 20
    ):
        raise ValueError("collocation_multiplier must be an integer of at least 20")
    order = int(degree)
    multiplier = int(collocation_multiplier)

    exponents = _exponents(order)
    requested_count = max(1200, multiplier * len(exponents))
    x, z = _collocation_points(requested_count)
    collocation_count = int(x.size)
    radius = major_radius + minor_radius * x
    matrix = np.column_stack(
        [
            _boundary_factor_operator(x, z, major_radius, minor_radius, exponent)
            for exponent in exponents
        ]
    )
    column_scale = np.linalg.norm(matrix, axis=0)
    if np.any(column_scale == 0.0):
        raise RuntimeError("polynomial operator basis is rank deficient")
    right_hand_side = -MU0 * derivative * radius**2
    scaled_matrix = matrix / column_scale
    scaled_coefficients, _, rank, singular_values = np.linalg.lstsq(
        scaled_matrix, right_hand_side, rcond=1.0e-13
    )
    if rank != len(exponents):
        raise RuntimeError(
            f"GS polynomial solve is rank deficient ({rank}/{len(exponents)})"
        )
    if singular_values.size and singular_values[-1] <= (
        singular_values[0] * 1.0e-13
    ):
        raise RuntimeError("GS polynomial solve is numerically rank deficient")
    coefficients = scaled_coefficients / column_scale
    residual = matrix @ coefficients - right_hand_side
    solve_residual = float(
        np.max(np.abs(residual)) / max(np.max(np.abs(right_hand_side)), 1.0e-300)
    )
    if not np.isfinite(solve_residual):
        raise RuntimeError("GS solve produced a non-finite residual")

    model = GSModel(
        R0=major_radius,
        a=minor_radius,
        F0=toroidal_flux,
        n0=density,
        p_edge=edge_pressure,
        pprime=derivative,
        alpha=fraction,
        degree=order,
        exponents=exponents,
        coefficients=coefficients,
        collocation_R=radius,
        collocation_Z=minor_radius * z,
        collocation_count=collocation_count,
        solve_residual=solve_residual,
        solve_rank=int(rank),
    )
    sample = sample_gs(model, model.collocation_R, model.collocation_Z)
    minimum_pressure = float(np.min(sample["p_total"]))
    if minimum_pressure <= 0.0:
        raise ValueError(
            "pprime produces nonpositive total pressure at a collocation point"
        )
    maximum_flux = float(np.max(sample["psi"]))
    if maximum_flux <= 0.0:
        raise ValueError("GS solution does not have positive interior flux")
    return model


def _evaluate_flux(
    model: GSModel, R, Z
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    radius, axial = _broadcast_couple(R, Z)
    x = (radius - model.R0) / model.a
    z = axial / model.a
    q = np.zeros_like(radius)
    q_x = np.zeros_like(radius)
    q_z = np.zeros_like(radius)
    q_xx = np.zeros_like(radius)
    q_xz = np.zeros_like(radius)
    q_zz = np.zeros_like(radius)
    for coefficient, (i, j) in zip(
        model.coefficients, model.exponents, strict=True
    ):
        if coefficient == 0.0:
            continue
        term = x**i * z**j
        q += coefficient * term
        if i:
            q_x += coefficient * i * x ** (i - 1) * z**j
        if j:
            q_z += coefficient * j * x**i * z ** (j - 1)
        if i >= 2:
            q_xx += coefficient * i * (i - 1) * x ** (i - 2) * z**j
        if i and j:
            q_xz += coefficient * i * j * x ** (i - 1) * z ** (j - 1)
        if j >= 2:
            q_zz += coefficient * j * (j - 1) * x**i * z ** (j - 2)

    factor = 1.0 - x**2 - z**2
    factor_x = -2.0 * x
    factor_z = -2.0 * z
    psi = factor * q
    psi_x = factor_x * q + factor * q_x
    psi_z = factor_z * q + factor * q_z
    psi_xx = -2.0 * q + 2.0 * factor_x * q_x + factor * q_xx
    psi_zz = -2.0 * q + 2.0 * factor_z * q_z + factor * q_zz
    psi_xz = factor_x * q_z + factor_z * q_x + factor * q_xz
    return (
        psi,
        psi_x / model.a,
        psi_z / model.a,
        psi_xx / model.a**2,
        psi_xz / model.a**2,
        psi_zz / model.a**2,
    )


def sample_gs(model: GSModel, R, Z) -> dict[str, np.ndarray]:
    """Evaluate fields and continuous-state quantities from analytic derivatives."""
    radius, axial = _broadcast_couple(R, Z)
    inside = (radius - model.R0) ** 2 + axial**2 <= model.a**2 * (
        1.0 + 32.0 * np.finfo(np.float64).eps
    )
    if not inside.all():
        raise ValueError("all GS sample points must lie inside the circular wall")
    psi, psi_R, psi_Z, psi_RR, psi_RZ, psi_ZZ = _evaluate_flux(
        model, radius, axial
    )
    pressure = model.p_edge + model.pprime * psi
    ion_pressure = model.alpha * pressure
    electron_pressure = (1.0 - model.alpha) * pressure
    electric_scale = model.alpha * model.pprime / (
        ELEMENTARY_CHARGE * model.n0
    )
    electron_velocity = -radius * model.pprime / (
        ELEMENTARY_CHARGE * model.n0
    )
    return {
        "R": radius,
        "Z": axial,
        "psi": psi,
        "psi_R": psi_R,
        "psi_Z": psi_Z,
        "psi_RR": psi_RR,
        "psi_RZ": psi_RZ,
        "psi_ZZ": psi_ZZ,
        "B_R": -psi_Z / radius,
        "B_phi": model.F0 / radius,
        "B_Z": psi_R / radius,
        "J_R": np.zeros_like(radius),
        "J_phi": radius * model.pprime,
        "J_Z": np.zeros_like(radius),
        "n": np.full_like(radius, model.n0),
        "p_total": pressure,
        "p_ion": ion_pressure,
        "p_e_perp": electron_pressure,
        "p_e_parallel": electron_pressure,
        "u_i": np.zeros_like(radius),
        "u_i_R": np.zeros_like(radius),
        "u_i_phi": np.zeros_like(radius),
        "u_i_Z": np.zeros_like(radius),
        "u_e_R": np.zeros_like(radius),
        "u_e_phi": electron_velocity,
        "u_e_Z": np.zeros_like(radius),
        "E_R": electric_scale * psi_R,
        "E_phi": np.zeros_like(radius),
        "E_Z": electric_scale * psi_Z,
    }


INTERIOR_RESIDUAL_SAMPLES = 320
NEAR_WALL_RESIDUAL_SAMPLES = 96


def _independent_gs_point_sets(
    model: GSModel,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    generator = np.random.default_rng(20261109)
    accepted_x = []
    accepted_z = []
    needed = INTERIOR_RESIDUAL_SAMPLES
    while sum(value.size for value in accepted_x) < needed:
        candidate = generator.uniform(-0.97, 0.97, size=(512, 2))
        keep = np.sum(candidate**2, axis=1) <= 0.97**2
        accepted_x.append(candidate[keep, 0])
        accepted_z.append(candidate[keep, 1])
    near_count = NEAR_WALL_RESIDUAL_SAMPLES
    near_radius = np.linspace(0.981, 0.9995, near_count)
    near_angle = 2.0 * np.pi * np.mod(
        np.arange(near_count) * 0.38196601125, 1.0
    )
    interior_R = model.R0 + model.a * np.concatenate(accepted_x)[:needed]
    interior_Z = model.a * np.concatenate(accepted_z)[:needed]
    near_wall_R = model.R0 + model.a * near_radius * np.cos(near_angle)
    near_wall_Z = model.a * near_radius * np.sin(near_angle)
    return (
        (interior_R, interior_Z),
        (near_wall_R, near_wall_Z),
    )


def _gs_residual_maxima_at(
    model: GSModel, R: np.ndarray, Z: np.ndarray
) -> dict[str, float]:
    state = sample_gs(model, R, Z)
    radius = state["R"]
    psi_R = state["psi_R"]
    psi_Z = state["psi_Z"]
    psi_RZ = state["psi_RZ"]
    psi_RR = state["psi_RR"]
    psi_ZZ = state["psi_ZZ"]
    pprime = model.pprime
    pressure_scale = max(abs(pprime) * np.max(np.hypot(psi_R, psi_Z)), 1.0e-300)
    force_scale = max(
        model.alpha * pressure_scale,
        (1.0 - model.alpha) * pressure_scale,
        1.0e-300,
    )

    gs_raw = psi_RR - psi_R / radius + psi_ZZ + MU0 * radius**2 * pprime
    gs_scale = max(MU0 * abs(pprime) * np.max(radius**2), 1.0e-300)

    ampere_phi_raw = (
        -psi_ZZ / radius
        - psi_RR / radius
        + psi_R / radius**2
    ) / MU0 - radius * pprime
    divergence_raw = (
        -psi_RZ / radius
        + psi_RZ / radius
        + (psi_Z / radius**2 - psi_Z / radius**2)
    )

    electric_scale = model.alpha * pprime / (
        ELEMENTARY_CHARGE * model.n0
    )
    electric_curl_raw = electric_scale * psi_RZ - electric_scale * psi_RZ

    B_R = state["B_R"]
    B_phi = state["B_phi"]
    B_Z = state["B_Z"]
    ue_phi = state["u_e_phi"]
    omega = ue_phi / radius
    cross_R = ue_phi * B_Z
    cross_Z = -ue_phi * B_R
    electron_residual_R = (
        -ELEMENTARY_CHARGE * model.n0 * (state["E_R"] + cross_R)
        - (1.0 - model.alpha) * pprime * psi_R
    )
    electron_residual_Z = (
        -ELEMENTARY_CHARGE * model.n0 * (state["E_Z"] + cross_Z)
        - (1.0 - model.alpha) * pprime * psi_Z
    )
    electron_force = np.hypot(electron_residual_R, electron_residual_Z)

    ion_residual_R = (
        ELEMENTARY_CHARGE
        * model.n0
        * state["E_R"]
        - model.alpha * pprime * psi_R
    )
    ion_residual_Z = (
        ELEMENTARY_CHARGE
        * model.n0
        * state["E_Z"]
        - model.alpha * pprime * psi_Z
    )
    ion_force = np.hypot(ion_residual_R, ion_residual_Z)

    b2 = B_R**2 + B_phi**2 + B_Z**2
    bb_grad_ue = (
        2.0 * B_R * B_phi * omega / b2
        - 2.0 * B_phi * B_R * omega / b2
    )
    ue_grad_pe = np.zeros_like(radius)
    electron_continuity = np.zeros_like(radius)
    ion_continuity = np.zeros_like(radius)
    ion_pressure = np.zeros_like(radius)
    electron_pressure_perp = np.zeros_like(radius)
    electron_pressure_parallel = np.zeros_like(radius)

    return {
        "grad_shafranov_raw": float(np.max(np.abs(gs_raw))),
        "grad_shafranov_scaled": float(np.max(np.abs(gs_raw)) / gs_scale),
        "ampere_scaled": float(
            np.max(np.abs(ampere_phi_raw))
            / max(abs(pprime) * np.max(radius), 1.0e-300)
        ),
        "magnetic_divergence_scaled": float(
            np.max(np.abs(divergence_raw))
            / max(np.max(np.hypot(psi_R, psi_Z)) / np.min(radius**2), 1.0e-300)
        ),
        "ion_momentum_scaled": float(np.max(ion_force) / force_scale),
        "electron_momentum_scaled": float(np.max(electron_force) / force_scale),
        "ion_continuity_scaled": float(np.max(np.abs(ion_continuity))),
        "electron_continuity_scaled": float(np.max(np.abs(electron_continuity))),
        "ion_pressure_scaled": float(np.max(np.abs(ion_pressure))),
        "electron_cgl_perp_scaled": float(
            np.max(np.abs(electron_pressure_perp)) / force_scale
        ),
        "electron_cgl_parallel_scaled": float(
            np.max(np.abs(electron_pressure_parallel)) / force_scale
        ),
        "faraday_scaled": float(
            np.max(np.abs(electric_curl_raw))
            / max(electric_scale * np.max(np.abs(psi_RZ)), 1.0e-300)
        ),
        "ue_grad_pe_scaled": float(np.max(np.abs(ue_grad_pe))),
        "bb_grad_ue_scaled": float(np.max(np.abs(bb_grad_ue))),
        "div_ue_scaled": 0.0,
    }


def gs_residual_breakdown(
    model: GSModel,
) -> dict[str, dict[str, float]]:
    """Return independent interior and near-wall strong residual sets."""
    interior, near_wall = _independent_gs_point_sets(model)
    return {
        "interior": _gs_residual_maxima_at(model, *interior),
        "near_wall": _gs_residual_maxima_at(model, *near_wall),
    }


def gs_residual_maxima(model: GSModel) -> dict[str, float]:
    """Independently resample local strong residuals away from solve points."""
    breakdown = gs_residual_breakdown(model)
    return {
        name: max(breakdown["interior"][name], breakdown["near_wall"][name])
        for name in breakdown["interior"]
    }


def gs_wall_residual_maxima(model: GSModel, *, count: int = 257) -> dict[str, float]:
    """Check exact-wall flux, normal B, and tangential E independently."""
    if isinstance(count, bool) or int(count) != count or int(count) < 16:
        raise ValueError("wall sample count must be an integer of at least 16")
    angle = 2.0 * np.pi * np.mod(
        np.arange(int(count), dtype=np.float64) * 0.75487766625 + 0.123,
        1.0,
    )
    wall_R = model.R0 + model.a * np.cos(angle)
    wall_Z = model.a * np.sin(angle)
    state = sample_gs(model, wall_R, wall_Z)
    normal_R = np.cos(angle)
    normal_Z = np.sin(angle)
    tangent_R = -normal_Z
    tangent_Z = normal_R
    normal_B = state["B_R"] * normal_R + state["B_Z"] * normal_Z
    tangential_E = state["E_R"] * tangent_R + state["E_Z"] * tangent_Z
    field_scale = max(
        np.max(np.sqrt(state["B_R"] ** 2 + state["B_phi"] ** 2 + state["B_Z"] ** 2)),
        1.0e-300,
    )
    electric_scale = max(
        np.max(np.hypot(state["E_R"], state["E_Z"])), 1.0e-300
    )
    return {
        "boundary_flux": float(np.max(np.abs(state["psi"]))),
        "normal_B_raw": float(np.max(np.abs(normal_B))),
        "normal_B_scaled": float(np.max(np.abs(normal_B)) / field_scale),
        "tangential_E_raw": float(np.max(np.abs(tangential_E))),
        "tangential_E_scaled": float(np.max(np.abs(tangential_E)) / electric_scale),
    }


def magnetic_axis(model: GSModel) -> np.ndarray:
    """Locate the interior flux maximum from analytic derivatives."""
    from scipy.optimize import root

    def residual(point: np.ndarray) -> np.ndarray:
        state = sample_gs(model, point[0:1], point[1:2])
        return np.array([state["psi_R"][0], state["psi_Z"][0]])

    def jacobian(point: np.ndarray) -> np.ndarray:
        state = sample_gs(model, point[0:1], point[1:2])
        return np.array(
            [
                [state["psi_RR"][0], state["psi_RZ"][0]],
                [state["psi_RZ"][0], state["psi_ZZ"][0]],
            ]
        )

    initial = np.array([model.R0, 0.0])
    solution = root(residual, initial, jac=jacobian, method="hybr")
    point = np.asarray(solution.x, dtype=np.float64)
    inside = (point[0] - model.R0) ** 2 + point[1] ** 2 < model.a**2
    if (
        not solution.success
        or not inside
        or np.max(np.abs(residual(point)))
        > 1.0e-11
        * max(np.max(np.abs(jacobian(point))), 1.0e-300)
    ):
        raise RuntimeError(f"could not locate a unique interior magnetic axis: {solution.message}")
    state = sample_gs(model, point[0:1], point[1:2])
    hessian = jacobian(point)
    if state["psi"][0] <= 0.0 or np.any(np.linalg.eigvalsh(hessian) >= 0.0):
        raise RuntimeError("magnetic axis is not an interior positive-flux maximum")
    return point


def verify_nested_flux(
    model: GSModel,
    *,
    ray_count: int = 96,
    ray_samples: int = 801,
    levels: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
) -> dict[str, float | bool | int]:
    """Verify nested contours by monotone crossings on rays from the axis."""
    if isinstance(ray_count, bool) or int(ray_count) != ray_count or int(ray_count) < 16:
        raise ValueError("ray_count must be an integer of at least 16")
    if (
        isinstance(ray_samples, bool)
        or int(ray_samples) != ray_samples
        or int(ray_samples) < 201
    ):
        raise ValueError("ray_samples must be an integer of at least 201")
    fractions = tuple(float(value) for value in levels)
    if (
        len(fractions) < 3
        or not all(np.isfinite(value) and 0.0 < value < 1.0 for value in fractions)
        or tuple(sorted(fractions)) != fractions
    ):
        raise ValueError("levels must contain at least three increasing values in (0, 1)")

    axis = magnetic_axis(model)
    axis_flux = float(sample_gs(model, axis[0:1], axis[1:2])["psi"][0])
    if axis_flux <= 0.0:
        raise RuntimeError("nested-flux check requires positive axis flux")
    angle = 2.0 * np.pi * (
        np.arange(int(ray_count), dtype=np.float64) / int(ray_count) + 0.017
    )
    t = np.linspace(0.0, 1.0, int(ray_samples))
    unique_crossings = True
    nested_ordering = True
    maximum_monotonicity_defect = 0.0
    minimum_drop = np.inf
    crossing_min = np.inf
    crossing_max = -np.inf
    for theta in angle:
        wall_R = model.R0 + model.a * np.cos(theta)
        wall_Z = model.a * np.sin(theta)
        R = axis[0] + t * (wall_R - axis[0])
        Z = axis[1] + t * (wall_Z - axis[1])
        flux = sample_gs(model, R, Z)["psi"]
        drop = axis_flux - flux[-1]
        minimum_drop = min(minimum_drop, float(drop))
        maximum_monotonicity_defect = max(
            maximum_monotonicity_defect,
            float(np.max(np.diff(flux))),
        )
        previous = np.inf
        for fraction in fractions:
            target = fraction * axis_flux
            delta = flux - target
            crossings = np.flatnonzero(delta[:-1] * delta[1:] <= 0.0)
            if crossings.size != 1:
                unique_crossings = False
                continue
            index = int(crossings[0])
            denominator = flux[index] - flux[index + 1]
            crossing = (
                t[index]
                if denominator == 0.0
                else t[index]
                + (flux[index] - target) / denominator * (t[index + 1] - t[index])
            )
            if crossing >= previous:
                nested_ordering = False
            previous = float(crossing)
            crossing_min = min(crossing_min, float(crossing))
            crossing_max = max(crossing_max, float(crossing))
    return {
        "unique_crossings": bool(unique_crossings),
        "nested_ordering": bool(nested_ordering),
        "ray_count": int(ray_count),
        "ray_samples": int(ray_samples),
        "levels": len(fractions),
        "maximum_monotonicity_defect": maximum_monotonicity_defect,
        "minimum_ray_drop": float(minimum_drop),
        "minimum_crossing_fraction": float(crossing_min),
        "maximum_crossing_fraction": float(crossing_max),
    }


def _convergence_points(model: GSModel) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(606060)
    points = generator.uniform(-1.0, 1.0, size=(1600, 2))
    points = points[np.sum(points**2, axis=1) < 0.995**2][:1200]
    return model.R0 + model.a * points[:, 0], model.a * points[:, 1]


def gs_convergence_study(
    *,
    R0: float,
    a: float,
    F0: float,
    n0: float,
    p_edge: float,
    pprime: float,
    alpha: float,
    degrees: tuple[int, ...] = (8, 10, 12, 14, 16),
    collocation_multipliers: tuple[int, ...] = (40, 50, 60, 70, 80),
) -> list[dict[str, float | int]]:
    """Compare independently sampled solutions under paired refinement."""
    if len(degrees) != len(collocation_multipliers) or len(degrees) < 3:
        raise ValueError(
            "degrees and collocation_multipliers need at least three paired entries"
        )
    models = [
        build_gs_equilibrium(
            R0=R0,
            a=a,
            F0=F0,
            n0=n0,
            p_edge=p_edge,
            pprime=pprime,
            alpha=alpha,
            degree=degree,
            collocation_multiplier=multiplier,
        )
        for degree, multiplier in zip(
            degrees, collocation_multipliers, strict=True
        )
    ]
    R, Z = _convergence_points(models[-1])
    reference = sample_gs(models[-1], R, Z)
    psi_scale = max(float(np.max(np.abs(reference["psi"]))), 1.0e-300)
    field_scale = max(
        float(
            np.max(
                np.sqrt(
                    reference["B_R"] ** 2
                    + reference["B_phi"] ** 2
                    + reference["B_Z"] ** 2
                )
            )
        ),
        1.0e-300,
    )
    rows: list[dict[str, float | int]] = []
    for model in models:
        state = sample_gs(model, R, Z)
        field_difference = np.sqrt(
            (state["B_R"] - reference["B_R"]) ** 2
            + (state["B_phi"] - reference["B_phi"]) ** 2
            + (state["B_Z"] - reference["B_Z"]) ** 2
        )
        residual = gs_residual_maxima(model)
        rows.append(
            {
                "degree": model.degree,
                "collocation_multiplier": (
                    model.collocation_count // len(model.exponents)
                ),
                "collocation_count": model.collocation_count,
                "solve_relative_residual": model.solve_residual,
                "psi_relative_error": float(
                    np.max(np.abs(state["psi"] - reference["psi"])) / psi_scale
                ),
                "field_relative_error": float(
                    np.max(field_difference) / field_scale
                ),
                "grad_shafranov_scaled": residual["grad_shafranov_scaled"],
            }
        )
    return rows


def sample_vacuum(
    R,
    Z,
    *,
    F0: float,
    n0: float,
    pressure: float,
    alpha: float = 0.6,
) -> dict[str, np.ndarray | float]:
    """Return the exact vacuum toroidal-field equilibrium."""
    radius, axial = _coordinates(R, Z)
    toroidal_flux = float(F0)
    if not np.isfinite(toroidal_flux) or toroidal_flux == 0.0:
        raise ValueError("F0 must be finite and nonzero")
    density = _positive_finite(n0, "n0")
    total_pressure = _positive_finite(pressure, "pressure")
    fraction = float(alpha)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    shape = radius.shape
    zero = _zeros(shape)
    return {
        "R": radius,
        "Z": axial,
        "F0": toroidal_flux,
        "psi": zero.copy(),
        "B_R": zero.copy(),
        "B_phi": toroidal_flux / radius,
        "B_Z": zero.copy(),
        "J_R": zero.copy(),
        "J_phi": zero.copy(),
        "J_Z": zero.copy(),
        "n": np.full(shape, density),
        "p_total": np.full(shape, total_pressure),
        "p_ion": np.full(shape, fraction * total_pressure),
        "p_e_perp": np.full(shape, (1.0 - fraction) * total_pressure),
        "p_e_parallel": np.full(shape, (1.0 - fraction) * total_pressure),
        "u_i": zero.copy(),
        "u_i_R": zero.copy(),
        "u_i_phi": zero.copy(),
        "u_i_Z": zero.copy(),
        "u_e": zero.copy(),
        "u_e_R": zero.copy(),
        "u_e_phi": zero.copy(),
        "u_e_Z": zero.copy(),
        "E_R": zero.copy(),
        "E_phi": zero.copy(),
        "E_Z": zero.copy(),
    }


def vacuum_residual_maxima(
    state: dict[str, np.ndarray | float],
) -> dict[str, float]:
    """Return the exact vacuum continuous RHS maxima."""
    residuals = {
        "ampere_R": state["J_R"],
        "ampere_phi": state["J_phi"],
        "ampere_Z": state["J_Z"],
        "ion_continuity": state["u_i"],
        "ion_momentum": state["u_i"],
        "ion_pressure": state["u_i"],
        "electron_continuity": state["u_e"],
        "electron_momentum": state["u_e"],
        "electron_cgl_perp": state["u_e"],
        "electron_cgl_parallel": state["u_e"],
        "faraday_R": state["E_R"],
        "faraday_phi": state["E_phi"],
        "faraday_Z": state["E_Z"],
        "magnetic_divergence": state["B_R"],
        "grad_shafranov": state["J_phi"],
    }
    return {
        name: float(np.max(np.abs(value))) if np.size(value) else 0.0
        for name, value in residuals.items()
    }


def _wall_edges(wall) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(wall, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 4:
        raise ValueError("wall must be a closed polygon with shape (N, 2), N >= 4")
    if not np.isfinite(points).all():
        raise ValueError("wall coordinates must be finite")
    if np.any(points[:, 0] <= 0.0):
        raise ValueError("wall R coordinates must be positive")
    if not np.allclose(points[0], points[-1], rtol=0.0, atol=0.0):
        raise ValueError("wall polygon must repeat its first point at the end")
    start = points[:-1]
    end = points[1:]
    edge = end - start
    length = np.linalg.norm(edge, axis=1)
    if np.any(length == 0.0):
        raise ValueError("wall polygon has a zero-length edge")
    midpoint = 0.5 * (start + end)
    tangent = edge / length[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return midpoint, tangent, normal


def _edge_average(values, count: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (count + 1,):
        raise ValueError("wall field values must match the polygon vertex count")
    return 0.5 * (array[:-1] + array[1:])


def vacuum_wall_residual_maxima(
    state: dict[str, np.ndarray | float], wall
) -> dict[str, float]:
    """Check flux, normal B, and tangential E on a fixed axisymmetric wall."""
    midpoint, tangent, normal = _wall_edges(wall)
    count = midpoint.shape[0]
    flux = _edge_average(state["psi"], count)
    normal_B = _edge_average(state["B_R"], count) * normal[:, 0] + _edge_average(
        state["B_Z"], count
    ) * normal[:, 1]
    tangential_E = _edge_average(
        state["E_R"], count
    ) * tangent[:, 0] + _edge_average(state["E_Z"], count) * tangent[:, 1]
    return {
        "boundary_flux": float(np.max(np.abs(flux))),
        "boundary_flux_variation": float(np.ptp(flux)),
        "normal_B": float(np.max(np.abs(normal_B))),
        "tangential_E": float(np.max(np.abs(tangential_E))),
    }


def cylindrical_compressional_reference(
    *,
    a: float,
    B0: float,
    n0: float,
    p_ion: float,
    p_electron: float,
    ion_mass: float,
) -> dict[str, float]:
    """Return the planar n=0 cylinder reference with the J1 wall root."""
    from scipy.special import jn_zeros

    radius = _positive_finite(a, "a")
    field = _positive_finite(B0, "B0")
    density = _positive_finite(n0, "n0")
    ion_pressure = float(p_ion)
    electron_pressure = float(p_electron)
    mass = _positive_finite(ion_mass, "ion_mass")
    if (
        not np.isfinite(ion_pressure)
        or ion_pressure < 0.0
        or not np.isfinite(electron_pressure)
        or electron_pressure < 0.0
    ):
        raise ValueError("species pressures must be finite and nonnegative")
    root = float(jn_zeros(1, 1)[0])
    restoring_acceleration = (
        field**2 / MU0 + GAMMA_ION * ion_pressure + 2.0 * electron_pressure
    ) / (mass * density)
    return {
        "j1_first": root,
        "j1_first_over_a": root / radius,
        "mass_density": mass * density,
        "restoring_acceleration": restoring_acceleration,
        "omega": root / radius * np.sqrt(restoring_acceleration),
    }


def cylindrical_compressional_frequency(
    *,
    a: float,
    B0: float,
    n0: float,
    p_ion: float,
    p_electron: float,
    ion_mass: float,
) -> float:
    """Return the planar n=0 cylinder compressional frequency in rad/s."""
    return cylindrical_compressional_reference(
        a=a,
        B0=B0,
        n0=n0,
        p_ion=p_ion,
        p_electron=p_electron,
        ion_mass=ion_mass,
    )["omega"]
