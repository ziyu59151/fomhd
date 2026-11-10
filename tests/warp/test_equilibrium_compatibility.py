"""Independent continuous equation checks for the circle-equilibrium audit."""

from __future__ import annotations

import numpy as np
import pytest
from tools.equilibrium_compatibility import (
    ELEMENTARY_CHARGE,
    MU0,
    audit_continuous,
    refinement_table,
    wall_trace,
)


def _array(value, r):
    return np.broadcast_to(np.asarray(value, dtype=np.float64), np.shape(r))


def _base_field(r, z):
    return {
        "psi": _array(0.0, r),
        "B_R": _array(0.0, r),
        "B_phi": 6.0 / r,
        "B_Z": _array(0.0, r),
        "E_R": _array(0.0, r),
        "E_phi": _array(0.0, r),
        "E_Z": _array(0.0, r),
        "n": _array(2.0e19, r),
        "u_i_R": _array(0.0, r),
        "u_i_phi": _array(0.0, r),
        "u_i_Z": _array(0.0, r),
        "p_i": _array(600.0, r),
        "p_e_perp": _array(400.0, r),
        "p_e_parallel": _array(400.0, r),
        "pprime": _array(0.0, r),
        "Fprime": _array(0.0, r),
    }


def _supply_zero_actual_current(result, r):
    result["J_actual_R"] = _array(0.0, r)
    result["J_actual_phi"] = _array(0.0, r)
    result["J_actual_Z"] = _array(0.0, r)
    result["DJ_actual_R_DR"] = _array(0.0, r)
    result["DJ_actual_R_DZ"] = _array(0.0, r)
    result["DJ_actual_phi_DR"] = _array(0.0, r)
    result["DJ_actual_phi_DZ"] = _array(0.0, r)
    result["DJ_actual_Z_DR"] = _array(0.0, r)
    result["DJ_actual_Z_DZ"] = _array(0.0, r)
    return result


def _max_norm(values):
    return float(np.sqrt(np.mean(np.sum(np.asarray(values) ** 2, axis=-1))))


def test_exact_vacuum_has_zero_evaluated_continuous_residuals():
    r = np.array([2.8, 3.0, 3.2])
    z = np.array([-0.1, 0.0, 0.1])
    result = audit_continuous(_base_field, r, z, step=2.0e-5)

    for name in (
        "divergence_B",
        "ampere_residual",
        "grad_shafranov",
        "ion_continuity",
        "electron_continuity",
        "ion_force",
        "electron_force",
        "ion_pressure",
        "electron_cgl_perp",
        "electron_cgl_parallel",
        "ohmic_residual",
        "faraday",
    ):
        assert _max_norm(result[name]) < 1.0e-4, name


def test_actual_current_is_cylindrical_curl_not_advertised_value():
    beta = 0.4

    def field(r, z):
        result = _base_field(r, z)
        result["psi"] = beta * r**3 / 3.0
        result["B_Z"] = beta * r
        result["J_R"] = _array(0.0, r)
        result["J_phi"] = _array(12345.0, r)
        result["J_Z"] = _array(0.0, r)
        return result

    r = np.array([2.8, 3.0, 3.2])
    z = np.array([-0.1, 0.0, 0.1])
    result = audit_continuous(field, r, z, step=2.0e-5)
    expected = np.full_like(r, -beta / MU0)

    np.testing.assert_allclose(result["J_actual"][:, 1], expected, rtol=2e-8)
    np.testing.assert_allclose(
        result["advertised_current_discrepancy"][:, 1],
        12345.0 - expected,
        rtol=3e-11,
    )
    assert np.max(np.abs(result["J_actual"][:, 1] - 12345.0)) > 1.0e4


def test_supplied_actual_current_derivatives_remove_cylindrical_cancellation():
    def field(r, z):
        result = _base_field(r, z)
        result["B_phi"] = 2000.0 / r
        result["J_actual_R"] = _array(0.0, r)
        result["J_actual_phi"] = _array(0.0, r)
        result["J_actual_Z"] = _array(0.0, r)
        result["DJ_actual_R_DR"] = _array(0.0, r)
        result["DJ_actual_R_DZ"] = _array(0.0, r)
        result["DJ_actual_phi_DR"] = _array(0.0, r)
        result["DJ_actual_phi_DZ"] = _array(0.0, r)
        result["DJ_actual_Z_DR"] = _array(0.0, r)
        result["DJ_actual_Z_DZ"] = _array(0.0, r)
        return result

    r = np.array([2.8, 3.0, 3.2])
    z = np.array([-0.1, 0.0, 0.1])
    result = audit_continuous(field, r, z, step=2.0e-5)

    np.testing.assert_array_equal(result["J_actual"], 0.0)
    np.testing.assert_array_equal(result["u_e"], 0.0)
    np.testing.assert_array_equal(result["strain"], 0.0)
    np.testing.assert_array_equal(result["electron_cgl_perp"], 0.0)
    np.testing.assert_array_equal(result["electron_cgl_parallel"], 0.0)


def test_supplied_current_uses_q_multiplier_in_varying_density_derivative():
    current_r = 0.7
    density_slope = 1.0e18

    def field(r, z):
        result = _base_field(r, z)
        result["B_phi"] = (6.0 - current_r * z) / r
        result["B_Z"] = _array(1.0, r)
        result["n"] = 2.0e19 + density_slope * (r - 3.0)
        result["J_actual_R"] = _array(current_r, r)
        result["J_actual_phi"] = _array(0.0, r)
        result["J_actual_Z"] = _array(0.0, r)
        result["DJ_actual_R_DR"] = _array(0.0, r)
        result["DJ_actual_R_DZ"] = _array(0.0, r)
        result["DJ_actual_phi_DR"] = _array(0.0, r)
        result["DJ_actual_phi_DZ"] = _array(0.0, r)
        result["DJ_actual_Z_DR"] = _array(0.0, r)
        result["DJ_actual_Z_DZ"] = _array(0.0, r)
        return result

    r = np.array([2.8, 3.0, 3.2])
    z = np.array([-0.1, 0.0, 0.1])
    result = audit_continuous(field, r, z, step=1.0e-5)
    density = 2.0e19 + density_slope * (r - 3.0)
    expected_ue_r = -current_r / (ELEMENTARY_CHARGE * density)
    expected_due_r_dr = current_r * density_slope / (ELEMENTARY_CHARGE * density**2)
    expected_continuity = -current_r / (ELEMENTARY_CHARGE * r)

    np.testing.assert_allclose(result["u_e"][:, 0], expected_ue_r, rtol=2e-15)
    np.testing.assert_allclose(
        result["velocity_gradient_electron"][:, 0, 0],
        expected_due_r_dr,
        rtol=2e-15,
    )
    np.testing.assert_allclose(
        result["electron_continuity"], expected_continuity, rtol=2e-15
    )


def test_uniform_axial_velocity_has_zero_cylindrical_strain():
    axial_speed = 2.3

    def field(r, z):
        result = _supply_zero_actual_current(_base_field(r, z), r)
        result["B_phi"] = 6.0 / r
        result["B_Z"] = _array(1.0, r)
        result["u_i_Z"] = _array(axial_speed, r)
        return result

    r = np.array([2.8, 3.0, 3.2])
    z = np.array([-0.1, 0.0, 0.1])
    result = audit_continuous(field, r, z, step=1.0e-5)

    np.testing.assert_array_equal(result["velocity_gradient_electron"][..., 2, 1], 0.0)
    np.testing.assert_allclose(result["theta"], 0.0, atol=1.0e-15)
    np.testing.assert_allclose(result["strain"], 0.0, atol=1.0e-15)


def test_anisotropic_pressure_is_rejected_by_isotropic_force_auditor():
    def field(r, z):
        result = _supply_zero_actual_current(_base_field(r, z), r)
        result["p_e_perp"] = _array(700.0, r)
        result["p_e_parallel"] = _array(500.0, r)
        return result

    r = np.array([2.8, 3.0, 3.2])
    z = np.array([-0.1, 0.0, 0.1])
    with pytest.raises(ValueError, match="isotropic"):
        audit_continuous(field, r, z, step=1.0e-5)

    result = audit_continuous(
        field,
        r,
        z,
        step=1.0e-5,
        isotropic_electron_force=False,
    )
    assert result["electron_force"] is None
    assert result["ohmic_residual"] is None
    assert result["isotropic_electron_force"] is False
    assert np.all(np.isfinite(result["electron_cgl_parallel"]))


def test_varying_F_parallel_flow_control_has_nonzero_cgl_response():
    alpha = 0.35
    epsilon = 0.2

    def field(r, z):
        result = _base_field(r, z)
        result["psi"] = _array(0.0, r)
        result["B_R"] = epsilon * r * z
        result["B_phi"] = (6.0 + 0.8 * z) / r
        result["B_Z"] = -epsilon * z**2 + 2.0
        result["Fprime"] = _array(0.8, r)
        result["p_e_perp"] = _array(700.0, r)
        result["p_e_parallel"] = _array(500.0, r)
        result["pprime"] = _array(alpha, r)
        return result

    r = np.array([2.9, 3.0, 3.1])
    z = np.array([-0.12, -0.05, 0.1])
    result = audit_continuous(
        field,
        r,
        z,
        step=2.0e-5,
        isotropic_electron_force=False,
    )

    assert np.max(np.abs(result["theta"])) < 2.0e-6
    assert np.max(np.abs(result["strain"])) > 1.0e-3
    assert np.max(np.abs(result["electron_cgl_perp"])) > 1.0e-3
    assert np.max(np.abs(result["electron_cgl_parallel"])) > 1.0e-3
    assert not np.allclose(result["electron_cgl_perp"], 0.0)
    assert not np.allclose(result["electron_cgl_parallel"], 0.0)


def test_actual_current_error_observes_second_order_refinement():
    r0 = 3.0
    coefficient = 0.25

    def field(r, z):
        result = _base_field(r, z)
        x = r - r0
        result["psi"] = coefficient * x**5
        result["B_Z"] = 5.0 * coefficient * x**4 / r
        return result

    r = np.array([2.95, 3.0, 3.05])
    z = np.array([-0.1, 0.0, 0.1])
    x = r - r0
    derivative = 20.0 * coefficient * x**3 / r - 5.0 * coefficient * x**4 / r**2
    reference = {
        "J_actual": np.column_stack(
            (
                np.zeros_like(r),
                -derivative / MU0,
                np.zeros_like(r),
            )
        )
    }
    table = refinement_table(
        field,
        r,
        z,
        steps=(0.04, 0.02, 0.01),
        reference=reference,
        keys=("J_actual",),
    )

    errors = table["J_actual"]["errors"]
    orders = table["J_actual"]["observed_order"]
    assert errors[0] > errors[1] > errors[2]
    assert all(order > 1.8 for order in orders)


def test_circle_wall_trace_reports_actual_normal_and_tangent_residuals():
    def field(r, z):
        result = _base_field(r, z)
        result["B_R"] = _array(0.2, r)
        result["B_Z"] = _array(-0.1, r)
        result["E_R"] = _array(0.3, r)
        result["E_Z"] = _array(0.4, r)
        return result

    trace = wall_trace(field, r0=3.0, a=0.5, step=2.0e-5, count=17)

    assert trace["normal_B_flux"].shape == (17,)
    assert trace["tangential_E"].shape == (17,)
    assert trace["normal_current"].shape == (17,)
    assert trace["ion_normal_number_flux"].shape == (17,)
    assert trace["electron_normal_number_flux"].shape == (17,)
    assert np.max(np.abs(trace["normal_B_flux"])) > 1.0e-3
    assert np.max(np.abs(trace["tangential_E"])) > 1.0e-3


def test_nonfinite_callback_values_are_rejected():
    def field(r, z):
        result = _base_field(r, z)
        result["n"] = _array(np.nan, r)
        return result

    with pytest.raises(ValueError, match="finite"):
        audit_continuous(field, 3.0, 0.0, step=1.0e-5)
