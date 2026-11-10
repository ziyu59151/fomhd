from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import j0, j1, jn_zeros

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from tools.cylindrical_caw_control import (  # noqa: E402
    DEFAULTS,
    SCOPE,
    CylinderCAW,
    fit_angular_frequency,
    seed_velocity,
)


@pytest.fixture
def reference():
    return CylinderCAW(**DEFAULTS)


def test_sample_matches_exact_j1_j0_phases_and_perturbations(reference):
    radius = np.array([0.0, 0.5 * reference.a, 0.9 * reference.a])
    angle = np.array([0.0, 0.4, 1.1])
    R = 3.0 + radius * np.cos(angle)
    Z = radius * np.sin(angle)
    time = 0.25 * reference.period

    state = reference.sample(R, Z, time)
    cosine = np.cos(reference.omega * time)
    sine = np.sin(reference.omega * time)
    j1r = j1(reference.kappa * radius)
    j0r = j0(reference.kappa * radius)
    radial = reference.A * reference.vA * j1r * cosine
    factor = -reference.n0 * reference.kappa * reference.A * reference.vA / (
        reference.omega
    ) * j0r * sine

    np.testing.assert_allclose(state["u_R"], radial * np.cos(angle), atol=1.0e-15)
    np.testing.assert_allclose(state["u_Z"], radial * np.sin(angle), atol=1.0e-15)
    np.testing.assert_allclose(state["u_phi"], 0.0, atol=0.0)
    np.testing.assert_allclose(
        state["theta"],
        reference.kappa * reference.A * reference.vA * j0r * cosine,
        rtol=1.0e-14,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(state["delta_n"], factor, rtol=1.0e-14)
    np.testing.assert_allclose(
        state["delta_p_i"], reference.gamma * reference.pi0 * factor / reference.n0
    )
    np.testing.assert_allclose(
        state["delta_pe_perp"], 2.0 * reference.pe0 * factor / reference.n0
    )
    np.testing.assert_allclose(
        state["delta_pe_parallel"], reference.pe0 * factor / reference.n0
    )
    np.testing.assert_allclose(
        state["delta_B_guide"], reference.B0 * factor / reference.n0
    )
    np.testing.assert_allclose(state["psi"], 0.0, atol=0.0)
    axis_state = reference.sample(R[:1], Z[:1], 0.0)
    np.testing.assert_allclose(
        axis_state["u_R_over_r"][0],
        reference.A * reference.vA * reference.kappa / 2.0,
    )


def test_analytic_continuity_and_cgl_sources_close(reference):
    R = np.array([2.65, 3.0, 3.2])
    Z = np.array([-0.2, 0.0, 0.31])
    time = 0.37 * reference.period

    residuals = reference.linear_pde_residuals(R, Z, time)
    state = reference.sample(R, Z, time)
    divergence_scale = np.max(np.abs(state["theta"]))
    scales = {
        "continuity": reference.n0 * divergence_scale,
        "ion_pressure": reference.gamma * reference.pi0 * divergence_scale,
        "electron_perp": 2.0 * reference.pe0 * divergence_scale,
        "electron_parallel": reference.pe0 * divergence_scale,
        "guide_field": reference.B0 * divergence_scale,
        "bb_grad_u": 1.0,
        "div_ue_minus_div_ui": divergence_scale,
    }
    for name, value in residuals.items():
        assert np.max(np.abs(value)) / scales[name] < 2.0e-14


def test_wall_root_and_conducting_impermeable_boundary(reference):
    time = np.linspace(0.0, reference.period, 17)
    checks = reference.boundary_checks(time, count=64)

    assert abs(checks["j1_root"]) < 5.0e-15
    assert checks["max_abs_u_radial"] < 1.0e-10
    assert checks["max_abs_delta_B_guide_variation"] < 1.0e-14
    assert checks["max_abs_delta_p_variation"] < 1.0e-11
    assert checks["boundary_flux"] == 0.0
    assert checks["normal_B_perturbation"] == 0.0
    assert checks["normal_current"] == 0.0
    assert checks["tangential_electric_field"] == 0.0
    assert checks["curl_electric_field"] == 0.0
    assert checks["bb_grad_u"] == 0.0


def test_velocity_is_pure_fundamental_without_overtone(reference):
    time = reference.period * np.linspace(0.0, 1.0, 257, endpoint=False)
    series = np.array(
        [
            float(np.asarray(reference.sample(3.25, 0.0, float(moment))["u_radial"]))
            for moment in time
        ]
    )
    cosine = np.cos(reference.omega * time)
    sine = np.sin(reference.omega * time)

    fundamental_cos = 2.0 * np.mean(series * cosine)
    fundamental_sin = 2.0 * np.mean(series * sine)
    second_cos = 2.0 * np.mean(series * np.cos(2.0 * reference.omega * time))
    second_sin = 2.0 * np.mean(series * np.sin(2.0 * reference.omega * time))

    assert np.max(np.abs(series)) > 0.0
    np.testing.assert_allclose(fundamental_sin, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(second_cos, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(second_sin, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        fundamental_cos,
        reference.A * reference.vA * j1(reference.kappa * 0.25),
        rtol=2.0e-14,
    )


def test_seed_velocity_returns_physical_r_phi_z_order(reference):
    angle = np.array([0.0, np.pi / 2.0])
    radius = np.array([0.25, 0.4])
    R = 3.0 + radius * np.cos(angle)
    Z = radius * np.sin(angle)
    velocity = seed_velocity(
        R,
        Z,
        R0=3.0,
        a=reference.a,
        amplitude=reference.A,
        B0=reference.B0,
        n0=reference.n0,
        mi=reference.mi,
        pi0=reference.pi0,
        pe0=reference.pe0,
        gamma=reference.gamma,
        mu0=reference.mu0,
        e=reference.e,
    )

    assert velocity.shape == (2, 3)
    assert np.all(velocity[:, 1] == 0.0)
    assert velocity[0, 0] > 0.0
    np.testing.assert_allclose(velocity[1, 0], 0.0, atol=2.0e-16)
    assert velocity[1, 2] > 0.0
    expected = reference.A * reference.vA * j1(reference.kappa * radius)
    np.testing.assert_allclose(velocity[:, 0], expected * np.cos(angle), atol=1.0e-12)
    np.testing.assert_allclose(velocity[:, 2], expected * np.sin(angle), atol=1.0e-12)


def test_m0_weighted_j1_and_j0_projection_recovers_coefficients(reference):
    angle = np.linspace(0.0, 2.0 * np.pi, 257)[:-1]
    radius = np.linspace(0.03, 0.95 * reference.a, 17)
    rr, tt = np.meshgrid(radius, angle, indexing="xy")
    R = 3.0 + rr.ravel() * np.cos(tt.ravel())
    Z = rr.ravel() * np.sin(tt.ravel())
    M0 = 0.5 + rr.ravel()
    time = 0.2 * reference.period
    state = reference.sample(R, Z, time)

    velocity = np.column_stack((state["u_R"], state["u_phi"], state["u_Z"]))
    velocity_projection = reference.project_velocity_amplitude(
        velocity, R, Z, R0=3.0, M0=M0
    )
    density_projection = reference.project_scalar(
        state["delta_n"], R, Z, R0=3.0, M0=M0
    )

    np.testing.assert_allclose(
        velocity_projection["fractional_amplitude"],
        reference.A * np.cos(reference.omega * time),
        rtol=2.0e-13,
        atol=1.0e-15,
    )
    expected_density = (
        -reference.n0
        * reference.kappa
        * reference.A
        * reference.vA
        / reference.omega
        * np.sin(reference.omega * time)
    )
    np.testing.assert_allclose(
        density_projection["coefficient"],
        expected_density,
        rtol=2.0e-13,
        atol=1.0e-12,
    )


def test_frequency_fit_uses_only_actual_supplied_times(reference):
    times = reference.period * np.array(
        [0.0, 0.071, 0.163, 0.287, 0.402, 0.611, 0.744, 0.893]
    )
    values = 0.7 * np.cos(reference.omega * times + 0.31) + 1.2

    fit = fit_angular_frequency(
        times, values, omega_guess=reference.omega
    )

    assert fit["sample_count"] == times.size
    np.testing.assert_allclose(fit["omega"], reference.omega, rtol=2.0e-11)
    assert fit["rmse"] < 2.0e-12


def test_large_aspect_report_uses_declared_ordering_not_toroidal_truth(reference):
    rows = reference.large_aspect_comparison((6.0, 15.0, 30.0, 60.0))

    assert [row["aspect_ratio"] for row in rows] == [6.0, 15.0, 30.0, 60.0]
    for row in rows:
        np.testing.assert_allclose(row["curvature_scale"], 1.0 / row["aspect_ratio"])
        assert row["hall_frequency_ratio"] > 0.0
        assert row["flr_parameter"] > 0.0
        assert row["requires_actual_spatial_convergence"] is True
        assert "not" in row["scope"].lower()


def test_defaults_copy_documented_constants_and_scope_is_linearized():
    assert DEFAULTS["mu0"] == 1.25663706127e-6
    assert DEFAULTS["mi"] == 2.0 * 1.67262192595e-27
    assert "linearized" in SCOPE
    assert "not a finite-amplitude" in SCOPE


def test_caller_physical_constants_are_reported_in_comparison_provenance():
    custom = CylinderCAW(
        **{
            **DEFAULTS,
            "mi": 3.5e-27,
            "mu0": 1.3e-6,
            "gamma": 1.8,
        }
    )
    row = custom.large_aspect_comparison((15.0,))[0]

    assert row["mi_kg"] == custom.mi
    assert row["mu0_H_m"] == custom.mu0
    assert row["gamma"] == custom.gamma


def test_public_helper_exposes_parameter_and_comparison_metadata(reference):
    row = reference.large_aspect_comparison((15.0,))[0]

    assert reference.mi == DEFAULTS["mi"]
    assert reference.mu0 == DEFAULTS["mu0"]
    assert reference.gamma == DEFAULTS["gamma"]
    assert row["mi_kg"] == reference.mi
    assert row["mu0_H_m"] == reference.mu0
    assert row["gamma"] == reference.gamma
    assert row["requires_actual_spatial_convergence"] is True
    assert "linearized" in row["scope"]
    assert "not a finite-amplitude nonlinear" in SCOPE


def test_validation_rejects_bad_amplitude_time_and_exterior(reference):
    with pytest.raises(ValueError, match="amplitude"):
        CylinderCAW(**{**DEFAULTS, "A": 1.1e-3})
    with pytest.raises(ValueError, match="amplitude"):
        CylinderCAW(**{**DEFAULTS, "A": -1.0e-4})
    with pytest.raises(ValueError, match="amplitude"):
        CylinderCAW(**{**DEFAULTS, "A": np.nan})
    with pytest.raises(ValueError, match="time"):
        reference.sample(3.0, 0.0, np.nan)
    with pytest.raises(ValueError, match="outside"):
        reference.sample(3.0 + 1.01 * reference.a, 0.0, 0.0)


def test_reference_module_has_no_fomhd_imports(reference):
    source = (ROOT / "tools/cylindrical_caw_control.py").read_text()
    assert "import fomhd" not in source
    assert "from fomhd" not in source
    assert jn_zeros(1, 1)[0] == pytest.approx(reference.j11)


def test_declared_maximum_amplitude_keeps_totals_positive(reference):
    radius = np.linspace(0.0, reference.a, 65)
    angle = np.linspace(0.0, 2.0 * np.pi, 33)[:-1]
    rr, tt = np.meshgrid(radius, angle, indexing="xy")
    R = 3.0 + rr.ravel() * np.cos(tt.ravel())
    Z = rr.ravel() * np.sin(tt.ravel())
    for fraction in np.linspace(0.0, 1.0, 17):
        state = reference.sample(
            R, Z, fraction * reference.period
        )
        assert np.all(state["n_total"] > 0.0)
        assert np.all(state["p_i_total"] > 0.0)
        assert np.all(state["pe_perp_total"] > 0.0)
        assert np.all(state["pe_parallel_total"] > 0.0)
        assert np.all(state["B_guide_total"] > 0.0)
