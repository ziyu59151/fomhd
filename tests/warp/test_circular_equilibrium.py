from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from tools.circular_equilibrium import (  # noqa: E402
    ELEMENTARY_CHARGE,
    MU0,
    build_gs_equilibrium,
    cylindrical_compressional_frequency,
    gs_convergence_study,
    gs_wall_residual_maxima,
    gs_particular,
    gs_residual_maxima,
    magnetic_axis,
    sample_gs,
    sample_vacuum,
    vacuum_residual_maxima,
    vacuum_wall_residual_maxima,
    verify_nested_flux,
)
from tools.prepare_circular_equilibrium import (  # noqa: E402
    SOURCE_ROOT,
    write_gs_fixture,
    write_vacuum_fixture,
)


def test_cli_source_root_derives_from_module_location():
    assert SOURCE_ROOT == ROOT


def test_vacuum_has_zero_current_flow_electric_field_and_rhs():
    R, Z = np.meshgrid([2.4, 3.0, 3.6], [-0.4, 0.0, 0.4], indexing="xy")

    state = sample_vacuum(R, Z, F0=6.0, n0=2.0e19, pressure=1000.0)
    residuals = vacuum_residual_maxima(state)

    np.testing.assert_allclose(state["B_R"], 0.0, atol=0.0)
    np.testing.assert_allclose(state["B_Z"], 0.0, atol=0.0)
    np.testing.assert_allclose(state["B_phi"], 6.0 / R, rtol=1.0e-15)
    np.testing.assert_allclose(state["p_ion"], 600.0, atol=0.0)
    np.testing.assert_allclose(state["p_e_perp"], 400.0, atol=0.0)
    np.testing.assert_allclose(state["p_e_parallel"], 400.0, atol=0.0)
    for name in ("J_R", "J_phi", "J_Z", "u_i", "u_e", "E_R", "E_phi", "E_Z"):
        np.testing.assert_allclose(state[name], 0.0, atol=0.0)
    assert residuals
    assert max(residuals.values()) == 0.0


def test_vacuum_satisfies_an_arbitrary_axisymmetric_wall():
    theta = np.linspace(0.0, 2.0 * np.pi, 97)
    wall = np.column_stack(
        (
            3.0 + 0.5 * np.cos(theta) * (1.0 + 0.08 * np.cos(3.0 * theta)),
            0.06 + 0.5 * np.sin(theta) * (1.0 + 0.05 * np.sin(4.0 * theta)),
        )
    )
    wall[-1] = wall[0]
    state = sample_vacuum(
        wall[:, 0], wall[:, 1], F0=6.0, n0=2.0e19, pressure=1000.0
    )

    residuals = vacuum_wall_residual_maxima(state, wall)

    assert residuals["boundary_flux"] == 0.0
    assert residuals["normal_B"] == 0.0
    assert residuals["tangential_E"] == 0.0


def test_gs_particular_has_the_required_signed_source():
    R = np.array([2.6, 3.0, 3.4])
    Z = np.zeros_like(R)
    pprime = 100.0

    psi, psi_R, psi_Z, psi_RR, psi_ZZ = gs_particular(R, Z, pprime=pprime)
    operator = psi_RR - psi_R / R + psi_ZZ

    np.testing.assert_allclose(psi, -MU0 * pprime * R**4 / 8.0)
    np.testing.assert_allclose(psi_R, -MU0 * pprime * R**3 / 2.0)
    np.testing.assert_allclose(psi_Z, 0.0)
    np.testing.assert_allclose(operator, -MU0 * pprime * R**2)


def test_gs_state_has_consistent_signs_and_zero_species_balance():
    model = build_gs_equilibrium(
        R0=3.0,
        a=0.5,
        F0=6.0,
        n0=2.0e19,
        p_edge=1000.0,
        pprime=100.0,
        alpha=0.6,
        degree=10,
    )
    R = np.array([2.65, 3.0, 3.35])
    Z = np.array([-0.2, 0.0, 0.2])
    state = sample_gs(model, R, Z)

    assert model.collocation_R.size == model.collocation_count
    np.testing.assert_allclose(state["B_R"], -state["psi_Z"] / R)
    np.testing.assert_allclose(state["B_Z"], state["psi_R"] / R)
    np.testing.assert_allclose(state["B_phi"], 6.0 / R)
    np.testing.assert_allclose(state["J_phi"], R * 100.0)
    np.testing.assert_allclose(
        state["u_e_phi"], -R * 100.0 / (ELEMENTARY_CHARGE * 2.0e19)
    )
    np.testing.assert_allclose(
        state["u_e_phi"] / R,
        -100.0 / (ELEMENTARY_CHARGE * 2.0e19),
    )
    np.testing.assert_allclose(
        state["E_R"],
        0.6 * 100.0 / (ELEMENTARY_CHARGE * 2.0e19) * state["psi_R"],
    )
    np.testing.assert_allclose(
        state["E_Z"],
        0.6 * 100.0 / (ELEMENTARY_CHARGE * 2.0e19) * state["psi_Z"],
    )
    np.testing.assert_allclose(state["p_ion"], 0.6 * state["p_total"])
    np.testing.assert_allclose(state["p_e_perp"], 0.4 * state["p_total"])

    residuals = gs_residual_maxima(model)
    assert residuals["ion_momentum_scaled"] < 1.0e-11
    assert residuals["electron_momentum_scaled"] < 1.0e-11
    assert residuals["faraday_scaled"] < 1.0e-11


def test_gs_circular_wall_and_shifted_axis_are_physical():
    model = build_gs_equilibrium(
        R0=3.0,
        a=0.5,
        F0=6.0,
        n0=2.0e19,
        p_edge=1000.0,
        pprime=1.0e4,
        alpha=0.6,
        degree=12,
    )

    wall = gs_wall_residual_maxima(model)
    axis = magnetic_axis(model)
    axis_state = sample_gs(model, axis[:1], axis[1:])

    assert wall["boundary_flux"] < 1.0e-15
    assert wall["normal_B_scaled"] < 1.0e-12
    assert wall["tangential_E_scaled"] < 1.0e-12
    assert 3.0 < axis[0] < 3.5
    assert abs(axis[1]) < 1.0e-10
    assert axis[0] - 3.0 > 0.02
    assert axis_state["psi"][0] > 0.0
    assert np.all(np.linalg.eigvalsh(
        np.array(
            [
                [axis_state["psi_RR"][0], axis_state["psi_RZ"][0]],
                [axis_state["psi_RZ"][0], axis_state["psi_ZZ"][0]],
            ]
        )
    ) < 0.0)


def test_gs_flux_surfaces_are_nested_and_pressure_current_are_bounded():
    model = build_gs_equilibrium(
        R0=3.0,
        a=0.5,
        F0=6.0,
        n0=2.0e19,
        p_edge=1000.0,
        pprime=1.0e4,
        alpha=0.6,
        degree=12,
    )
    topology = verify_nested_flux(model)
    axis = magnetic_axis(model)
    axis_flux = sample_gs(model, axis[:1], axis[1:])["psi"][0]
    interior = sample_gs(
        model,
        np.linspace(2.51, 3.49, 101),
        np.zeros(101),
    )

    assert topology["unique_crossings"] is True
    assert topology["nested_ordering"] is True
    assert topology["minimum_ray_drop"] > 0.99 * axis_flux
    assert np.min(interior["p_total"]) > 0.0
    assert np.max(interior["J_phi"]) < 4.0e4


def test_gs_spectral_basis_and_collocation_enrichment_converge():
    report = gs_convergence_study(
        R0=3.0,
        a=0.5,
        F0=6.0,
        n0=2.0e19,
        p_edge=1000.0,
        pprime=1.0e4,
        alpha=0.6,
        degrees=(8, 10, 12),
        collocation_multipliers=(40, 60, 80),
    )

    assert [row["collocation_count"] for row in report] == sorted(
        row["collocation_count"] for row in report
    )
    assert report[-1]["psi_relative_error"] < 1.0e-8
    assert report[-1]["grad_shafranov_scaled"] < 1.0e-9
    assert report[-1]["psi_relative_error"] <= (
        report[0]["psi_relative_error"] * 0.1
    )


def test_n0_cylinder_compressional_reference_uses_first_j1_wall_root():
    from scipy.special import jn_zeros

    frequency = cylindrical_compressional_frequency(
        a=0.5,
        B0=2.0,
        n0=2.0e19,
        p_ion=600.0,
        p_electron=400.0,
        ion_mass=3.343583719e-27,
    )
    expected = jn_zeros(1, 1)[0] / 0.5 * np.sqrt(
        (2.0**2 / MU0 + (5.0 / 3.0) * 600.0 + 2.0 * 400.0)
        / (3.343583719e-27 * 2.0e19)
    )

    assert frequency > 0.0
    assert np.isclose(frequency, expected, rtol=1.0e-14, atol=0.0)


def test_cli_fixture_records_si_layout_provenance_and_residuals():
    scratch = ROOT
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        result = write_gs_fixture(
            Path(directory),
            grid_size=65,
            degree=12,
            run_convergence=True,
        )
        with np.load(
            Path(directory) / "gs_equilibrium.npz", allow_pickle=False
        ) as fixture:
            assert fixture["psi"].shape == (65, 65)
            assert fixture["R_profile"].shape == (65,)
            assert fixture["Z_profile"].shape == (65,)
            assert "wall_mask" in fixture
            for name in (
                "u_i_R",
                "u_i_phi",
                "u_i_Z",
                "u_e_R",
                "u_e_phi",
                "u_e_Z",
            ):
                assert name in fixture
                assert fixture[name].shape == (65, 65)
            assert fixture["p_total"][fixture["wall_mask"]].min() > 0.0
        assert result["status"] == "PASS"
        assert result["layout"].startswith("psi[iz,ir]")
        assert result["units"]["psi"] == "Wb/rad"
        assert result["units"]["psi_R"] == "T m"
        assert result["units"]["psi_RR"] == "T"
        assert result["units"]["pressure"] == "Pa"
        assert result["provenance"]["source_tree"] == str(ROOT)
        assert result["checks"]["maximum_abs_Bpol_T"] < 0.02
        assert result["checks"]["wall"]["boundary_flux"] < 1.0e-15
        assert result["checks"]["residual_sample_counts"]["near_wall"] == 96
        assert (
            result["checks"]["near_wall_continuous_residuals"][
                "grad_shafranov_scaled"
            ]
            < 1.0e-9
        )


def test_vacuum_cli_fixture_checks_a_noncircular_closed_wall():
    scratch = ROOT
    angle = np.linspace(0.0, 2.0 * np.pi, 17)
    wall = np.column_stack(
        (
            3.0 + 0.55 * np.cos(angle),
            0.05 + 0.42 * np.sin(angle),
        )
    )
    wall[-1] = wall[0]

    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        result = write_vacuum_fixture(Path(directory), wall=wall)
        with np.load(
            Path(directory) / "vacuum_equilibrium.npz", allow_pickle=False
        ) as fixture:
            assert fixture["B_R"].shape == (17,)
            assert np.all(fixture["B_R"] == 0.0)
            assert np.all(fixture["E_R"] == 0.0)

    assert result["status"] == "PASS"
