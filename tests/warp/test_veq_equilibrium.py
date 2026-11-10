from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
ARCHIVE_SHA256 = (
    "6b553ce32bb9261e4782af4542b7be15f14621f3256d303f9b175fdd582bfc1f"
)
sys.path.insert(0, str(TOOLS))

from veq_equilibrium import (  # noqa: E402
    MU0_VEQ,
    _physical_gradient,
    build_constant_f_snapshot,
    constant_f_profiles,
    make_source_metadata,
    strong_gs_metrics,
    wall_normal_flux,
    wall_normal_flux_polygon_midpoint,
    write_equilibrium_product,
)


def _radial_differentiator(nodes):
    count = len(nodes)
    vandermonde = np.vander(nodes, N=count, increasing=True)
    derivative = np.zeros_like(vandermonde)
    for degree in range(1, count):
        derivative[:, degree] = degree * nodes ** (degree - 1)
    return np.linalg.solve(vandermonde.T, derivative.T).T


def _synthetic_native_snapshot():
    radial = np.linspace(0.05, 1.0, 17)
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    rr, tt = np.meshgrid(radial, theta, indexing="ij")
    R = 3.0 + 0.5 * rr * np.cos(tt)
    Z = 0.5 * rr * np.sin(tt)
    psi_edge = 0.01
    psi_f = psi_edge * (1.0 - rr**2)
    native_psi = 2.0 * np.pi * (psi_edge - psi_f)
    native_p = 800.0 + 20.0 * psi_f
    pprime = 1.0e4
    native_jphi = R * pprime
    return {
        "r": radial,
        "theta": theta,
        "R": R,
        "Z": Z,
        "radial_differentiator": _radial_differentiator(radial),
        "psi_Wb": native_psi,
        "p_Pa": native_p,
        "p0_Pa": 800.0,
        "jphi_A_m2": native_jphi,
    }


def _metadata():
    return make_source_metadata(
        source_commit="2b64c9a1bffddc25c176bd52231bc3b481715c6c",
        source_tree="ac8635c629d100db8998f9904793dbf697373eb7",
        archive_sha256=ARCHIVE_SHA256,
        R0_m=3.0,
        Z0_m=-0.2,
        a_m=0.5,
        F_edge_Tm=6.0,
        p_edge_target_Pa=1000.0,
        pprime_F_Pa_per_Wb_rad=1.0e4,
        density_m3=2.0e19,
        ion_pressure_fraction=0.6,
        native_p0_Pa=800.0,
        raw_p0_input_Pa=1000.0,
        Ip_A=23561.94490192345,
    )


def test_adapter_import_does_not_require_veqpy():
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(TOOLS)!r}); "
        "import veq_equilibrium; "
        "assert 'veqpy' not in sys.modules; "
        "print('adapter import clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "adapter import clean"


def test_constant_f_derivation_matches_circular_analytic_fields():
    native = _synthetic_native_snapshot()
    metadata = make_source_metadata(
        source_commit="2b64c9a1bffddc25c176bd52231bc3b481715c6c",
        source_tree="ac8635c629d100db8998f9904793dbf697373eb7",
        archive_sha256=ARCHIVE_SHA256,
        R0_m=3.0,
        Z0_m=0.0,
        a_m=0.5,
        F_edge_Tm=6.0,
        p_edge_target_Pa=1000.0,
        pprime_F_Pa_per_Wb_rad=1.0e4,
        density_m3=2.0e19,
        ion_pressure_fraction=0.6,
        native_p0_Pa=800.0,
        raw_p0_input_Pa=1000.0,
        Ip_A=23561.94490192345,
    )
    snapshot = build_constant_f_snapshot(metadata=metadata, **native)
    radial = snapshot.r
    theta = snapshot.theta
    rr, tt = np.meshgrid(radial, theta, indexing="ij")
    expected_grad_R = -0.08 * (snapshot.R_m - 3.0)
    expected_grad_Z = -0.08 * snapshot.Z_m
    np.testing.assert_allclose(
        snapshot.psi_F_Wb_per_rad, 0.01 * (1.0 - rr**2), rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        snapshot.B_R_T, -expected_grad_Z / snapshot.R_m, rtol=0, atol=2e-12
    )
    np.testing.assert_allclose(
        snapshot.B_Z_T, expected_grad_R / snapshot.R_m, rtol=0, atol=2e-12
    )
    np.testing.assert_allclose(snapshot.B_phi_T, 6.0 / snapshot.R_m)
    np.testing.assert_allclose(
        snapshot.J_profile_phi_A_m2, snapshot.R_m * 1.0e4
    )
    np.testing.assert_allclose(snapshot.J_profile_R_A_m2, 0.0)
    np.testing.assert_allclose(snapshot.J_profile_Z_A_m2, 0.0)
    assert np.all(np.isfinite(snapshot.J_ampere_phi_A_m2))
    assert abs(snapshot.I_p_ampere_A) > 0.0
    assert snapshot.ampere_profile_phi_max_abs_residual_A_m2 > 0.0
    np.testing.assert_allclose(snapshot.u_i_R_m_s, 0.0)
    np.testing.assert_allclose(snapshot.u_i_phi_m_s, 0.0)
    np.testing.assert_allclose(snapshot.u_i_Z_m_s, 0.0)
    np.testing.assert_allclose(
        snapshot.u_e_profile_phi_m_s,
        -snapshot.J_profile_phi_A_m2
        / (snapshot.metadata.electron_charge_C * 2e19),
    )
    np.testing.assert_allclose(
        snapshot.E_profile_R_V_m,
            0.6 * 1e4 * expected_grad_R
            / (snapshot.metadata.electron_charge_C * 2e19),
            rtol=2e-11,
            atol=2e-9,
    )
    np.testing.assert_allclose(
        snapshot.E_profile_Z_V_m,
            0.6 * 1e4 * expected_grad_Z
            / (snapshot.metadata.electron_charge_C * 2e19),
            rtol=2e-11,
            atol=2e-9,
    )


def test_circular_tangent_B_uses_collocated_smooth_wall_normal():
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    R = 3.0 + 0.5 * np.cos(theta)
    Z = 0.5 * np.sin(theta)
    tangent_B_R = -np.sin(theta)
    tangent_B_Z = np.cos(theta)
    smooth = wall_normal_flux(
        theta, R, Z, tangent_B_R, tangent_B_Z
    )
    chord = wall_normal_flux_polygon_midpoint(
        R, Z, tangent_B_R, tangent_B_Z
    )
    assert smooth["max_abs_T"] < 1.0e-14
    assert chord["max_abs_T"] < 1.0e-14
    dR = np.roll(R, -1) - R
    dZ = np.roll(Z, -1) - Z
    chord_normal_R = dZ / np.hypot(dR, dZ)
    chord_normal_Z = -dR / np.hypot(dR, dZ)
    start_vertex_artefact = tangent_B_R * chord_normal_R
    start_vertex_artefact += tangent_B_Z * chord_normal_Z
    assert np.max(np.abs(start_vertex_artefact)) > 1.0e-3

    outward_B_R = np.cos(theta)
    outward_B_Z = np.sin(theta)
    outward = wall_normal_flux(
        theta, R, Z, outward_B_R, outward_B_Z
    )
    assert outward["max_abs_T"] == pytest.approx(1.0, abs=1e-14)
    assert outward["rms_T"] == pytest.approx(1.0, abs=1e-14)


def test_shifted_axis_gradient_uses_unique_regular_limit():
    radial = np.linspace(0.0, 1.0, 17)
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    rr, tt = np.meshgrid(radial, theta, indexing="ij")
    R = 3.4 + 0.5 * rr * np.cos(tt)
    Z = -0.2 + 0.5 * rr * np.sin(tt)
    x = R - 3.4
    y = Z + 0.2
    values = (
        2.0
        + 0.3 * x
        - 0.2 * y
        + 0.07 * (x**2 + y**2)
        + 0.03 * (x**2 - y**2)
    )
    grad_R, grad_Z, axis_samples = _physical_gradient(
        values,
        _radial_differentiator(radial),
        R,
        Z,
        r=radial,
        axis_gradient=None,
    )
    assert axis_samples == theta.size
    np.testing.assert_allclose(grad_R[0], 0.3, rtol=0, atol=2e-12)
    np.testing.assert_allclose(grad_Z[0], -0.2, rtol=0, atol=2e-12)
    assert np.ptp(grad_R[0]) < 1e-14
    assert np.ptp(grad_Z[0]) < 1e-14

    first_ring_copy_R = grad_R[1]
    first_ring_copy_Z = grad_Z[1]
    assert np.ptp(first_ring_copy_R) > 1e-4
    assert np.ptp(first_ring_copy_Z) > 1e-4


def test_collapsed_off_axis_ring_is_rejected():
    radial = np.linspace(0.0, 1.0, 17)
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    rr, tt = np.meshgrid(radial, theta, indexing="ij")
    R = 3.4 + 0.5 * rr * np.cos(tt)
    Z = -0.2 + 0.5 * rr * np.sin(tt)
    R[5] = R[0]
    Z[5] = Z[0]
    with pytest.raises(ValueError, match="singular non-axis radial row"):
        _physical_gradient(
            R,
            _radial_differentiator(radial),
            R,
            Z,
            r=radial,
        )


def test_shifted_flux_axis_has_zero_poloidal_field_and_separate_diagnostics():
    radial = np.linspace(0.0, 1.0, 17)
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    rr, tt = np.meshgrid(radial, theta, indexing="ij")
    R = 3.4 + 0.5 * rr * np.cos(tt)
    Z = -0.2 + 0.5 * rr * np.sin(tt)
    psi_F = 0.01 * (1.0 - rr**2)
    native_psi = 2.0 * np.pi * (0.01 - psi_F)
    native_p = 800.0 + 20.0 * psi_F
    snapshot = build_constant_f_snapshot(
        metadata=_metadata(),
        r=radial,
        theta=theta,
        R=R,
        Z=Z,
        radial_differentiator=_radial_differentiator(radial),
        psi_Wb=native_psi,
        p_Pa=native_p,
        p0_Pa=800.0,
        jphi_A_m2=R * 1.0e4,
        native_p_psi_Pa_per_Wb_rad=np.full_like(R, -1.0e4),
    )
    np.testing.assert_array_equal(snapshot.B_R_T[0], 0.0)
    np.testing.assert_array_equal(snapshot.B_Z_T[0], 0.0)
    metrics = strong_gs_metrics(snapshot)
    assert metrics["axis"]["excluded_sample_count"] == theta.size
    assert metrics["axis"]["physical_point_count"] == 1
    assert metrics["axis"]["reconstructed_Bpol_max_abs_T"] == 0.0
    assert metrics["valid_collocation"]["radial_row_count"] == radial.size - 1


def test_native_pressure_and_declared_additive_offset_are_both_retained():
    native = _synthetic_native_snapshot()
    metadata = make_source_metadata(
        source_commit="2b64c9a1bffddc25c176bd52231bc3b481715c6c",
        source_tree="ac8635c629d100db8998f9904793dbf697373eb7",
        archive_sha256=ARCHIVE_SHA256,
        R0_m=3.0,
        Z0_m=0.0,
        a_m=0.5,
        F_edge_Tm=6.0,
        p_edge_target_Pa=1000.0,
        pprime_F_Pa_per_Wb_rad=1.0e4,
        density_m3=2.0e19,
        ion_pressure_fraction=0.6,
        native_p0_Pa=800.0,
        raw_p0_input_Pa=1000.0,
        Ip_A=23561.94490192345,
    )
    snapshot = build_constant_f_snapshot(metadata=metadata, **native)
    assert metadata.pressure_offset_Pa == pytest.approx(200.0)
    np.testing.assert_allclose(
        snapshot.p_total_Pa, native["p_Pa"] + metadata.pressure_offset_Pa
    )
    np.testing.assert_allclose(snapshot.p_native_Pa, native["p_Pa"])
    assert metadata.native_p0_Pa == 800.0
    assert metadata.raw_p0_input_Pa == 1000.0
    np.testing.assert_allclose(snapshot.p_total_Pa[-1, :], 1000.0)
    np.testing.assert_allclose(snapshot.p_i_Pa, 0.6 * snapshot.p_total_Pa)
    np.testing.assert_allclose(snapshot.p_e_perp_Pa, 0.4 * snapshot.p_total_Pa)
    np.testing.assert_allclose(snapshot.p_e_parallel_Pa, 0.4 * snapshot.p_total_Pa)


def test_constant_f_profile_function_is_explicit_and_si():
    metadata = make_source_metadata(
        source_commit="2b64c9a1bffddc25c176bd52231bc3b481715c6c",
        source_tree="ac8635c629d100db8998f9904793dbf697373eb7",
        archive_sha256=ARCHIVE_SHA256,
        R0_m=3.0,
        Z0_m=0.0,
        a_m=0.5,
        F_edge_Tm=6.0,
        p_edge_target_Pa=1000.0,
        pprime_F_Pa_per_Wb_rad=1.0e4,
        density_m3=2.0e19,
        ion_pressure_fraction=0.6,
        native_p0_Pa=7.0,
        raw_p0_input_Pa=1000.0,
        Ip_A=23561.94490192345,
    )
    psi = np.array([0.0, 0.002, 0.007])
    profiles = constant_f_profiles(metadata, psi)
    np.testing.assert_array_equal(profiles["F_Tm"], 6.0)
    np.testing.assert_array_equal(profiles["Fprime_Tm"], 0.0)
    np.testing.assert_allclose(profiles["p_Pa"], 1000.0 + 1e4 * psi)
    np.testing.assert_array_equal(profiles["P_psi_Pa_per_Wb_rad"], -1e4)


def test_metadata_distinguishes_raw_coordinate_derivatives_from_physical_derivative():
    metadata = make_source_metadata(
        source_commit="2b64c9a1bffddc25c176bd52231bc3b481715c6c",
        source_tree="ac8635c629d100db8998f9904793dbf697373eb7",
        archive_sha256=ARCHIVE_SHA256,
        R0_m=3.0,
        Z0_m=0.0,
        a_m=0.5,
        F_edge_Tm=6.0,
        p_edge_target_Pa=1000.0,
        pprime_F_Pa_per_Wb_rad=1.0e4,
        density_m3=2.0e19,
        ion_pressure_fraction=0.6,
        native_p0_Pa=7.848,
        raw_p0_input_Pa=1000.0,
        Ip_A=23561.94490192345,
    )
    record = metadata.as_dict()
    assert record["raw_units"]["P_psin"] == "Pa"
    assert record["raw_units"]["FF_psin"] == "(T*m)^2"
    assert record["physical_units"]["P_psi"] == "Pa/(Wb/rad)"
    assert record["mu0_VEQ_H_per_m"] == pytest.approx(4e-7 * math.pi)
    assert record["mu0_FOMHD_reference_H_per_m"] == 1.25663706127e-6
    assert 0.0 < record["mu0_relative_difference"] < 2.0e-10
    assert MU0_VEQ == pytest.approx(4e-7 * math.pi)
    assert record["flux_conversion"]["psi_F"] == (
        "psi_edge - Psi/(2*pi), psi_edge=Psi_edge/(2*pi)"
    )
    assert record["native_pressure"]["pressure_offset_Pa"] == pytest.approx(
        1000.0 - 7.848
    )


def test_arrival_snapshot_is_read_only_and_write_round_trips(tmp_path):
    native = _synthetic_native_snapshot()
    metadata = make_source_metadata(
        source_commit="2b64c9a1bffddc25c176bd52231bc3b481715c6c",
        source_tree="ac8635c629d100db8998f9904793dbf697373eb7",
        archive_sha256=ARCHIVE_SHA256,
        R0_m=3.0,
        Z0_m=0.0,
        a_m=0.5,
        F_edge_Tm=6.0,
        p_edge_target_Pa=1000.0,
        pprime_F_Pa_per_Wb_rad=1.0e4,
        density_m3=2.0e19,
        ion_pressure_fraction=0.6,
        native_p0_Pa=800.0,
        raw_p0_input_Pa=1000.0,
        Ip_A=23561.94490192345,
    )
    snapshot = build_constant_f_snapshot(metadata=metadata, **native)
    assert snapshot.B_R_T.flags.writeable is False
    assert snapshot.p_total_Pa.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        snapshot.B_R_T[0, 0] = 0.0
    npz_path, json_path = write_equilibrium_product(tmp_path / "case", snapshot)
    assert npz_path.is_file() and json_path.is_file()
    saved = json.loads(json_path.read_text())
    assert saved["raw_units"]["P_psin"] == "Pa"
    with np.load(npz_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["B_R_T"], snapshot.B_R_T)
        np.testing.assert_array_equal(archive["p_native_Pa"], snapshot.p_native_Pa)
        np.testing.assert_array_equal(
            archive["J_profile_phi_A_m2"], snapshot.J_profile_phi_A_m2
        )
        np.testing.assert_array_equal(
            archive["J_ampere_phi_A_m2"], snapshot.J_ampere_phi_A_m2
        )
        np.testing.assert_array_equal(
            archive["u_e_profile_phi_m_s"], snapshot.u_e_profile_phi_m_s
        )


def test_strong_gs_metrics_requires_nonempty_held_out_bands():
    native = _synthetic_native_snapshot()
    metadata = make_source_metadata(
        source_commit="2b64c9a1bffddc25c176bd52231bc3b481715c6c",
        source_tree="ac8635c629d100db8998f9904793dbf697373eb7",
        archive_sha256=ARCHIVE_SHA256,
        R0_m=3.0,
        Z0_m=0.0,
        a_m=0.5,
        F_edge_Tm=6.0,
        p_edge_target_Pa=1000.0,
        pprime_F_Pa_per_Wb_rad=1.0e4,
        density_m3=2.0e19,
        ion_pressure_fraction=0.6,
        native_p0_Pa=800.0,
        raw_p0_input_Pa=1000.0,
        Ip_A=23561.94490192345,
    )
    snapshot = build_constant_f_snapshot(metadata=metadata, **native)
    metrics = strong_gs_metrics(snapshot)
    assert metrics["interior"]["sample_count"] > 0
    assert metrics["near_wall"]["sample_count"] > 0
    assert np.isfinite(metrics["interior"]["max_abs_T"])
    assert np.isfinite(metrics["near_wall"]["rms_T"])
    assert np.isfinite(metrics["wall_normal_B_smooth"]["max_abs_T"])
    assert np.isfinite(
        metrics["wall_normal_B_polygon_midpoint"]["max_abs_T"]
    )


def test_prepare_cli_imports_without_veqpy():
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(TOOLS)!r}); "
        "import prepare_veq_equilibrium; "
        "assert 'veqpy' not in sys.modules; "
        "print('cli import clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "cli import clean"


def test_refinement_plans_respect_small_bounds():
    from prepare_veq_equilibrium import (
        CIRCLE_REFINEMENTS,
        FINE_EVALUATION_GRIDS,
    )

    assert len(CIRCLE_REFINEMENTS) == 3
    assert len(FINE_EVALUATION_GRIDS) == 3
    assert len(set(FINE_EVALUATION_GRIDS)) == 3
    for plan in CIRCLE_REFINEMENTS:
        active = (
            plan.h_count
            + plan.kappa_count
            + plan.s_count
            + plan.psin_count
        )
        assert 0 < active <= 100
        assert 4 <= plan.Nr <= 64
        assert 4 <= plan.Nt <= 128
    for Nr, Nt in FINE_EVALUATION_GRIDS:
        assert 4 <= Nr <= 64
        assert 4 <= Nt <= 128


def test_native_profile_export_retains_named_coefficients():
    from types import SimpleNamespace

    from prepare_veq_equilibrium import _native_shape_profile_arrays

    profile = SimpleNamespace(
        scale=0.5,
        power=0,
        envelope_power=1,
        amplitude_power=1.0,
        offset=0.0,
        coeff=np.array([1.0, 2.0]),
    )
    equilibrium = SimpleNamespace(shape_profiles={"h": profile})
    arrays, records = _native_shape_profile_arrays(equilibrium)
    np.testing.assert_array_equal(arrays["profile_coeff_h"], [1.0, 2.0])
    assert records["h"]["profile"] == "h"
    assert records["h"]["coeff"] == [1.0, 2.0]
