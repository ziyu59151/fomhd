"""Independent continuous equation compatibility checks in cylindrical SI.

The auditor consumes a physical callable and evaluates derivatives outside the
source stage. Electron flow is reconstructed from the actual Ampere current.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

MU0 = 1.25663706127e-6
ELEMENTARY_CHARGE = 1.602176634e-19

_REQUIRED = (
    "psi",
    "B_R",
    "B_phi",
    "B_Z",
    "E_R",
    "E_phi",
    "E_Z",
    "n",
    "u_i_R",
    "u_i_phi",
    "u_i_Z",
    "p_i",
    "p_e_perp",
    "p_e_parallel",
    "pprime",
    "Fprime",
)
_STORED_CURRENT = ("J_R", "J_phi", "J_Z")
_ACTUAL_CURRENT = ("J_actual_R", "J_actual_phi", "J_actual_Z")
_ACTUAL_CURRENT_DERIVATIVES = (
    "DJ_actual_R_DR",
    "DJ_actual_R_DZ",
    "DJ_actual_phi_DR",
    "DJ_actual_phi_DZ",
    "DJ_actual_Z_DR",
    "DJ_actual_Z_DZ",
)


@dataclass(frozen=True)
class PlasmaConstants:
    """SI constants used by the continuous compatibility equations."""

    mu0: float = MU0
    electron_charge: float = ELEMENTARY_CHARGE
    ion_charge: float = ELEMENTARY_CHARGE
    gamma_ion: float = 5.0 / 3.0

    def __post_init__(self):
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (
                self.mu0,
                self.electron_charge,
                self.ion_charge,
                self.gamma_ion,
            )
        ):
            raise ValueError("plasma constants must be finite and positive")


DEFAULT_CONSTANTS = PlasmaConstants()


def _broadcast_r_z(r, z):
    try:
        radius, axial = np.broadcast_arrays(
            np.asarray(r, dtype=np.float64),
            np.asarray(z, dtype=np.float64),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("R and Z must be numeric and broadcast-compatible") from error
    if radius.size == 0:
        raise ValueError("audit points cannot be empty")
    if np.any(radius <= 0.0):
        raise ValueError("audit points require R > 0")
    if not np.isfinite(radius).all() or not np.isfinite(axial).all():
        raise ValueError("audit coordinates must be finite")
    return radius, axial


def _step(value):
    step = float(value)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("derivative step must be finite and positive")
    return step


def _sample(field, r, z, *, constants):
    try:
        raw = field(r, z)
    except Exception as error:
        raise ValueError("continuous field callback failed") from error
    if not isinstance(raw, dict):
        raise TypeError("continuous field callback must return a mapping")
    missing = [name for name in _REQUIRED if name not in raw]
    if missing:
        raise ValueError(f"continuous field callback is missing {missing[0]}")
    result = {}
    for name in _REQUIRED:
        try:
            value = np.asarray(raw[name], dtype=np.float64)
            value = np.broadcast_to(value, r.shape)
        except (TypeError, ValueError) as error:
            raise ValueError(f"field {name} has an incompatible shape") from error
        if not np.isfinite(value).all():
            raise ValueError(f"field {name} must be finite")
        result[name] = value
    for name in _STORED_CURRENT:
        if name in raw:
            value = np.asarray(raw[name], dtype=np.float64)
            value = np.broadcast_to(value, r.shape)
            if not np.isfinite(value).all():
                raise ValueError(f"field {name} must be finite")
            result[name] = value
    for name in (*_ACTUAL_CURRENT, *_ACTUAL_CURRENT_DERIVATIVES):
        if name in raw:
            value = np.asarray(raw[name], dtype=np.float64)
            value = np.broadcast_to(value, r.shape)
            if not np.isfinite(value).all():
                raise ValueError(f"field {name} must be finite")
            result[name] = value
    if np.any(result["n"] <= 0.0):
        raise ValueError("ion density must be positive")
    if (
        np.any(result["p_i"] <= 0.0)
        or np.any(result["p_e_perp"] <= 0.0)
        or np.any(result["p_e_parallel"] <= 0.0)
    ):
        raise ValueError("species pressures must be positive")
    magnitude = np.sqrt(result["B_R"] ** 2 + result["B_phi"] ** 2 + result["B_Z"] ** 2)
    if np.any(magnitude == 0.0):
        raise ValueError("magnetic field magnitude must be nonzero")
    result["B_magnitude"] = magnitude
    result["n_e"] = (
        abs(constants.ion_charge) * result["n"] / abs(constants.electron_charge)
    )
    return result


def _derivatives(field, r, z, step, constants):
    center = _sample(field, r, z, constants=constants)
    plus_r = _sample(field, r + step, z, constants=constants)
    minus_r = _sample(field, r - step, z, constants=constants)
    plus_z = _sample(field, r, z + step, constants=constants)
    minus_z = _sample(field, r, z - step, constants=constants)
    derivative_r = {
        name: (plus_r[name] - minus_r[name]) / (2.0 * step) for name in _REQUIRED
    }
    derivative_z = {
        name: (plus_z[name] - minus_z[name]) / (2.0 * step) for name in _REQUIRED
    }
    return center, (plus_r, minus_r, plus_z, minus_z), derivative_r, derivative_z


def _curl_axisymmetric(
    vector_r,
    vector_phi,
    vector_z,
    dphi_dr,
    dphi_dz,
    dr_dz,
    dz_dr,
    r,
):
    return np.stack(
        (
            -dphi_dz,
            dr_dz - dz_dr,
            vector_phi / r + dphi_dr,
        ),
        axis=-1,
    )


def _actual_current(field, r, z, step, constants):
    center = _sample(field, r, z, constants=constants)
    plus_r = _sample(field, r + step, z, constants=constants)
    minus_r = _sample(field, r - step, z, constants=constants)
    plus_z = _sample(field, r, z + step, constants=constants)
    minus_z = _sample(field, r, z - step, constants=constants)
    dbr_dz = (plus_z["B_R"] - minus_z["B_R"]) / (2.0 * step)
    dbz_dr = (plus_r["B_Z"] - minus_r["B_Z"]) / (2.0 * step)
    dbphi_dr = (plus_r["B_phi"] - minus_r["B_phi"]) / (2.0 * step)
    dbphi_dz = (plus_z["B_phi"] - minus_z["B_phi"]) / (2.0 * step)
    curl = _curl_axisymmetric(
        center["B_R"],
        center["B_phi"],
        center["B_Z"],
        dbphi_dr,
        dbphi_dz,
        dbr_dz,
        dbz_dr,
        r,
    )
    return curl / constants.mu0


def _electron_velocity(field, r, z, current, constants):
    state = _sample(field, r, z, constants=constants)
    ui = np.stack((state["u_i_R"], state["u_i_phi"], state["u_i_Z"]), axis=-1)
    numerator = constants.ion_charge * state["n"][..., None] * ui - current
    denominator = abs(constants.electron_charge) * state["n_e"][..., None]
    return numerator / denominator


def _velocity_gradient(
    r,
    u_r,
    u_phi,
    u_z,
    du_r_dr,
    du_r_dz,
    du_phi_dr,
    du_phi_dz,
    du_z_dr,
    du_z_dz,
):
    gradient = np.zeros((*np.shape(r), 3, 3), dtype=np.float64)
    gradient[..., 0, 0] = du_r_dr
    gradient[..., 0, 1] = -u_phi / r
    gradient[..., 0, 2] = du_r_dz
    gradient[..., 1, 0] = du_phi_dr
    gradient[..., 1, 1] = u_r / r
    gradient[..., 1, 2] = du_phi_dz
    gradient[..., 2, 0] = du_z_dr
    gradient[..., 2, 1] = 0.0
    gradient[..., 2, 2] = du_z_dz
    return gradient


def _cross(velocity, magnetic):
    return np.stack(
        (
            velocity[..., 1] * magnetic[..., 2] - velocity[..., 2] * magnetic[..., 1],
            velocity[..., 2] * magnetic[..., 0] - velocity[..., 0] * magnetic[..., 2],
            velocity[..., 0] * magnetic[..., 1] - velocity[..., 1] * magnetic[..., 0],
        ),
        axis=-1,
    )


def _divergence_axisymmetric(vector_r, vector_z, dvr_dr, dvz_dz, r):
    return dvr_dr + vector_r / r + dvz_dz


def audit_continuous(
    field,
    r,
    z,
    *,
    step,
    constants: PlasmaConstants = DEFAULT_CONSTANTS,
    derive_electron_from_current: bool = True,
    isotropic_electron_force: bool = True,
):
    """Evaluate independent continuous compatibility residuals at physical points."""
    radius, axial = _broadcast_r_z(r, z)
    step = _step(step)
    if not isinstance(derive_electron_from_current, bool):
        raise TypeError("derive_electron_from_current must be boolean")
    if not derive_electron_from_current:
        raise ValueError("this audit requires electron flow from actual Ampere current")
    if not isinstance(isotropic_electron_force, bool):
        raise TypeError("isotropic_electron_force must be boolean")
    center, shifts, dr, dz = _derivatives(field, radius, axial, step, constants)
    b = np.stack((center["B_R"], center["B_phi"], center["B_Z"]), axis=-1)
    e = np.stack((center["E_R"], center["E_phi"], center["E_Z"]), axis=-1)
    ui = np.stack((center["u_i_R"], center["u_i_phi"], center["u_i_Z"]), axis=-1)

    divergence_b = dr["B_R"] + center["B_R"] / radius + dz["B_Z"]
    curl_b = _curl_axisymmetric(
        center["B_R"],
        center["B_phi"],
        center["B_Z"],
        dr["B_phi"],
        dz["B_phi"],
        dz["B_R"],
        dr["B_Z"],
        radius,
    )
    supplied_current = all(
        name in center for name in (*_ACTUAL_CURRENT, *_ACTUAL_CURRENT_DERIVATIVES)
    )
    if supplied_current:
        current_actual = np.stack(
            (center["J_actual_R"], center["J_actual_phi"], center["J_actual_Z"]),
            axis=-1,
        )
        current_derivative_r = np.stack(
            (
                center["DJ_actual_R_DR"],
                center["DJ_actual_phi_DR"],
                center["DJ_actual_Z_DR"],
            ),
            axis=-1,
        )
        current_derivative_z = np.stack(
            (
                center["DJ_actual_R_DZ"],
                center["DJ_actual_phi_DZ"],
                center["DJ_actual_Z_DZ"],
            ),
            axis=-1,
        )
    else:
        current_actual = curl_b / constants.mu0
        current_derivative_r = None
        current_derivative_z = None
    ue = _electron_velocity(field, radius, axial, current_actual, constants)
    if supplied_current:
        denominator = constants.ion_charge * center["n"]
        ui_derivative_r = np.stack((dr["u_i_R"], dr["u_i_phi"], dr["u_i_Z"]), axis=-1)
        ui_derivative_z = np.stack((dz["u_i_R"], dz["u_i_phi"], dz["u_i_Z"]), axis=-1)
        due_dr = (
            ui_derivative_r
            - current_derivative_r / denominator[..., None]
            + constants.ion_charge
            * current_actual
            * dr["n"][..., None]
            / (denominator**2)[..., None]
        )
        due_dz = (
            ui_derivative_z
            - current_derivative_z / denominator[..., None]
            + constants.ion_charge
            * current_actual
            * dz["n"][..., None]
            / (denominator**2)[..., None]
        )
        ue_pr = ue_mr = ue_pz = ue_mz = None
    else:
        ue_pr = _electron_velocity(
            field,
            radius + step,
            axial,
            _actual_current(field, radius + step, axial, step, constants),
            constants,
        )
        ue_mr = _electron_velocity(
            field,
            radius - step,
            axial,
            _actual_current(field, radius - step, axial, step, constants),
            constants,
        )
        ue_pz = _electron_velocity(
            field,
            radius,
            axial + step,
            _actual_current(field, radius, axial + step, step, constants),
            constants,
        )
        ue_mz = _electron_velocity(
            field,
            radius,
            axial - step,
            _actual_current(field, radius, axial - step, step, constants),
            constants,
        )
        due_dr = (ue_pr - ue_mr) / (2.0 * step)
        due_dz = (ue_pz - ue_mz) / (2.0 * step)
    gradient_e = _velocity_gradient(
        radius,
        ue[..., 0],
        ue[..., 1],
        ue[..., 2],
        due_dr[..., 0],
        due_dz[..., 0],
        due_dr[..., 1],
        due_dz[..., 1],
        due_dr[..., 2],
        due_dz[..., 2],
    )
    bhat = b / center["B_magnitude"][..., None]
    theta = np.trace(gradient_e, axis1=-2, axis2=-1)
    strain = np.einsum(
        "...i,...ij,...j->...",
        bhat,
        gradient_e,
        bhat,
    )

    gradient_i = _velocity_gradient(
        radius,
        center["u_i_R"],
        center["u_i_phi"],
        center["u_i_Z"],
        dr["u_i_R"],
        dz["u_i_R"],
        dr["u_i_phi"],
        dz["u_i_phi"],
        dr["u_i_Z"],
        dz["u_i_Z"],
    )
    theta_i = np.trace(gradient_i, axis1=-2, axis2=-1)

    flux_i_r = center["n"] * center["u_i_R"]
    flux_i_z = center["n"] * center["u_i_Z"]
    ipr, imr, ipz, imz = shifts
    dflux_i_r_dr = (ipr["n"] * ipr["u_i_R"] - imr["n"] * imr["u_i_R"]) / (2.0 * step)
    dflux_i_z_dz = (ipz["n"] * ipz["u_i_Z"] - imz["n"] * imz["u_i_Z"]) / (2.0 * step)
    ion_continuity = _divergence_axisymmetric(
        flux_i_r, flux_i_z, dflux_i_r_dr, dflux_i_z_dz, radius
    )

    flux_e_r = center["n_e"] * ue[..., 0]
    flux_e_z = center["n_e"] * ue[..., 2]
    if supplied_current:
        dn_e_dr = abs(constants.ion_charge) / abs(constants.electron_charge) * dr["n"]
        dn_e_dz = abs(constants.ion_charge) / abs(constants.electron_charge) * dz["n"]
        dflux_e_r_dr = dn_e_dr * ue[..., 0] + center["n_e"] * due_dr[..., 0]
        dflux_e_z_dz = dn_e_dz * ue[..., 2] + center["n_e"] * due_dz[..., 2]
    else:
        epr = _sample(field, radius + step, axial, constants=constants)
        emr = _sample(field, radius - step, axial, constants=constants)
        epz = _sample(field, radius, axial + step, constants=constants)
        emz = _sample(field, radius, axial - step, constants=constants)
        dflux_e_r_dr = (epr["n_e"] * ue_pr[..., 0] - emr["n_e"] * ue_mr[..., 0]) / (
            2.0 * step
        )
        dflux_e_z_dz = (epz["n_e"] * ue_pz[..., 2] - emz["n_e"] * ue_mz[..., 2]) / (
            2.0 * step
        )
    electron_continuity = _divergence_axisymmetric(
        flux_e_r, flux_e_z, dflux_e_r_dr, dflux_e_z_dz, radius
    )

    grad_pi = np.stack((dr["p_i"], dz["p_i"]), axis=-1)
    grad_pe_perp = np.stack((dr["p_e_perp"], dz["p_e_perp"]), axis=-1)
    grad_pe_parallel = np.stack((dr["p_e_parallel"], dz["p_e_parallel"]), axis=-1)
    zero = np.zeros_like(center["pprime"])
    grad_pi_3 = np.stack((grad_pi[..., 0], zero, grad_pi[..., 1]), axis=-1)
    grad_pe_perp_3 = np.stack(
        (grad_pe_perp[..., 0], zero, grad_pe_perp[..., 1]), axis=-1
    )
    grad_pe_parallel_3 = np.stack(
        (grad_pe_parallel[..., 0], zero, grad_pe_parallel[..., 1]),
        axis=-1,
    )
    grad_pe_3 = 0.5 * (grad_pe_perp_3 + grad_pe_parallel_3)
    pressure_scale = np.maximum.reduce(
        (
            np.abs(center["p_e_perp"]),
            np.abs(center["p_e_parallel"]),
            np.full_like(center["p_e_perp"], np.finfo(np.float64).tiny),
        )
    )
    isotropic_pressure = (
        np.abs(center["p_e_perp"] - center["p_e_parallel"])
        <= 512.0 * np.finfo(np.float64).eps * pressure_scale
    )
    isotropic = bool(np.all(isotropic_pressure))
    if isotropic_electron_force and not isotropic:
        raise ValueError(
            "pressure force/Ohm audit requires isotropic electron pressure; "
            "anisotropic CGL tensor divergence is not implemented"
        )
    ion_force = (
        constants.ion_charge * center["n"][..., None] * (e + _cross(ui, b)) - grad_pi_3
    )
    electron_force = None
    ohmic_residual = None
    if isotropic_electron_force:
        electron_force = (
            -constants.electron_charge * center["n_e"][..., None] * (e + _cross(ue, b))
            - grad_pe_3
        )
    ion_pressure = (
        np.sum(ui * grad_pi_3, axis=-1) + constants.gamma_ion * center["p_i"] * theta_i
    )
    electron_cgl_perp = np.sum(ue * grad_pe_perp_3, axis=-1) - center["p_e_perp"] * (
        strain - 2.0 * theta
    )
    electron_cgl_parallel = np.sum(ue * grad_pe_parallel_3, axis=-1) + center[
        "p_e_parallel"
    ] * (theta + 2.0 * strain)
    if isotropic_electron_force:
        ohmic_residual = (
            e
            + _cross(ue, b)
            + grad_pe_3 / (constants.electron_charge * center["n_e"][..., None])
        )
    curl_e = _curl_axisymmetric(
        center["E_R"],
        center["E_phi"],
        center["E_Z"],
        dr["E_phi"],
        dz["E_phi"],
        dz["E_R"],
        dr["E_Z"],
        radius,
    )
    psi_second_r = (shifts[0]["psi"] - 2.0 * center["psi"] + shifts[1]["psi"]) / step**2
    psi_second_z = (shifts[2]["psi"] - 2.0 * center["psi"] + shifts[3]["psi"]) / step**2
    grad_shafranov = (
        psi_second_r
        - dr["psi"] / radius
        + psi_second_z
        + constants.mu0 * radius**2 * center["pprime"]
        + center["Fprime"] * radius * center["B_phi"]
    )

    if all(name in center for name in _STORED_CURRENT):
        stored_current = np.stack(
            (center["J_R"], center["J_phi"], center["J_Z"]), axis=-1
        )
        advertised_discrepancy = stored_current - current_actual
    else:
        stored_current = np.full_like(current_actual, np.nan)
        advertised_discrepancy = np.zeros_like(current_actual)

    ucross = _cross(ue, b)
    e_scale = np.maximum(
        np.linalg.norm(e, axis=-1),
        np.linalg.norm(ucross, axis=-1)
        + np.linalg.norm(grad_pe_3, axis=-1)
        / (constants.electron_charge * center["n_e"]),
    )
    force_scale = np.maximum(
        constants.ion_charge * center["n"] * np.linalg.norm(e + _cross(ui, b), axis=-1),
        np.linalg.norm(grad_pi_3, axis=-1),
    )
    current_scale = np.maximum(
        np.linalg.norm(current_actual, axis=-1),
        abs(center["pprime"]) * radius,
    )
    cgl_scale = np.maximum(
        center["p_e_perp"] * (np.abs(strain) + 2.0 * np.abs(theta)),
        abs(electron_cgl_perp),
    )
    faraday_scale = np.maximum(
        np.abs(center["E_phi"]) / radius,
        np.maximum(np.abs(dr["E_phi"]), np.abs(dz["E_phi"]))
        + np.maximum(np.abs(dz["E_R"]), np.abs(dr["E_Z"])),
    )
    return {
        **center,
        "R": radius,
        "Z": axial,
        "B": b,
        "E": e,
        "u_i": ui,
        "u_e": ue,
        "bhat": bhat,
        "J_actual": current_actual,
        "J_stored_comparison": stored_current,
        "advertised_current_discrepancy": advertised_discrepancy,
        "divergence_B": divergence_b,
        "ampere_residual": current_actual - curl_b / constants.mu0,
        "grad_shafranov": grad_shafranov,
        "ion_continuity": ion_continuity,
        "electron_continuity": electron_continuity,
        "ion_force": ion_force,
        "electron_force": electron_force,
        "isotropic_electron_pressure": isotropic,
        "isotropic_electron_force": isotropic_electron_force,
        "ion_pressure": ion_pressure,
        "velocity_gradient_electron": gradient_e,
        "theta": theta,
        "strain": strain,
        "electron_cgl_perp": electron_cgl_perp,
        "electron_cgl_parallel": electron_cgl_parallel,
        "ohmic_residual": ohmic_residual,
        "faraday": curl_e,
        "scales": {
            "electric": e_scale,
            "force": force_scale,
            "current": current_scale,
            "cgl": cgl_scale,
            "faraday": faraday_scale,
        },
        "derivative_step": step,
    }


def _error_norm(actual, expected):
    difference = np.asarray(actual) - np.asarray(expected)
    scale = np.sqrt(np.mean(np.asarray(expected) ** 2))
    error = float(np.sqrt(np.mean(difference**2)))
    if scale > 1.0e-300:
        return error / float(scale)
    return error


def refinement_table(
    field,
    r,
    z,
    *,
    steps,
    reference,
    keys,
    constants: PlasmaConstants = DEFAULT_CONSTANTS,
):
    """Compare selected audited quantities with independent references."""
    step_values = tuple(_step(value) for value in steps)
    if len(step_values) < 3:
        raise ValueError("refinement requires at least three derivative steps")
    key_values = tuple(keys)
    audited = [
        audit_continuous(
            field,
            r,
            z,
            step=step,
            constants=constants,
        )
        for step in step_values
    ]
    result = {}
    for key in key_values:
        if key not in reference:
            raise ValueError(f"reference is missing {key}")
        errors = [_error_norm(item[key], reference[key]) for item in audited]
        orders = [
            float(np.log2(coarse / fine)) if coarse > 0.0 and fine > 0.0 else None
            for coarse, fine in itertools.pairwise(errors)
        ]
        result[key] = {
            "steps": list(step_values),
            "errors": errors,
            "observed_order": orders,
        }
    return result


def wall_trace(
    field,
    *,
    r0,
    a,
    step,
    count=257,
    constants: PlasmaConstants = DEFAULT_CONSTANTS,
):
    """Evaluate initial boundary traces on a circle."""
    major = float(r0)
    minor = float(a)
    if (
        not math.isfinite(major)
        or not math.isfinite(minor)
        or major <= 0.0
        or minor <= 0.0
        or minor >= major
    ):
        raise ValueError("circle radii must be finite, positive and ordered")
    if isinstance(count, bool) or not isinstance(count, int) or count < 8:
        raise ValueError("wall count must be an integer >= 8")
    angle = 2.0 * np.pi * np.arange(count, dtype=np.float64) / count
    r = major + minor * np.cos(angle)
    z = minor * np.sin(angle)
    normal = np.column_stack((np.cos(angle), np.sin(angle)))
    tangent = np.column_stack((-np.sin(angle), np.cos(angle)))
    audit = audit_continuous(field, r, z, step=_step(step), constants=constants)
    poloidal_b = audit["B"][..., (0, 2)]
    poloidal_e = audit["E"][..., (0, 2)]
    poloidal_ue = audit["u_e"][..., (0, 2)]
    poloidal_ui = audit["u_i"][..., (0, 2)]
    poloidal_j = audit["J_actual"][..., (0, 2)]
    normal_b = np.sum(poloidal_b * normal, axis=-1)
    tangent_e = np.sum(poloidal_e * tangent, axis=-1)
    normal_current = np.sum(poloidal_j * normal, axis=-1)
    ion_flux = audit["n"] * np.sum(poloidal_ui * normal, axis=-1)
    electron_flux = audit["n_e"] * np.sum(poloidal_ue * normal, axis=-1)
    arc = 2.0 * np.pi * minor / count
    return {
        "R": r,
        "Z": z,
        "normal_R": normal[:, 0],
        "normal_Z": normal[:, 1],
        "psi": audit["psi"],
        "normal_B_flux": normal_b,
        "tangential_E": tangent_e,
        "normal_current": normal_current,
        "ion_normal_number_flux": ion_flux,
        "electron_normal_number_flux": electron_flux,
        "integrated_normal_B": float(np.sum(normal_b * r * arc)),
        "integrated_normal_current": float(np.sum(normal_current * r * arc)),
        "integrated_ion_number_flux": float(np.sum(ion_flux * r * arc)),
        "integrated_electron_number_flux": float(np.sum(electron_flux * r * arc)),
        "scales": {
            "magnetic": float(np.max(audit["B_magnitude"])),
            "electric": float(np.max(np.linalg.norm(audit["E"], axis=-1))),
            "number_flux": float(
                np.max(np.maximum(np.abs(ion_flux), np.abs(electron_flux)))
            ),
        },
    }
