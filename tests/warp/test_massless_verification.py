"""Independent verification of the massless quasineutral Hall branch."""

from __future__ import annotations

import os
from pathlib import Path
import importlib.util
import subprocess
import sys

import numpy as np
import pytest

import massless_verification_fixture as fixture
from massless_verification_fixture import (
    HallCase,
    _closed_context,
    _pressure_bump_derivatives,
    _runtime_finite,
    _region_norms,
    closed_spatial_evidence,
    cold_transverse_matrix,
    drift_ohm_tautology_evidence,
    hall_frequencies,
    initial_hall_eigenmode,
    production_transverse_matrix,
    reconstruction_evidence,
    run_hall_time_case,
    variable_density_velocity_evidence,
)


@pytest.mark.parametrize("kdi", [0.1, 1.0, 10.0])
def test_cold_transverse_hall_has_two_positive_branches(kdi: float) -> None:
    case = HallCase(kdi=kdi)
    frequencies = hall_frequencies(case)
    matrix = cold_transverse_matrix(case)

    assert frequencies.omega_minus > 0.0
    assert frequencies.omega_plus > frequencies.omega_minus
    eigenvalues = np.linalg.eigvals(matrix)
    positive_imaginary = sorted(
        float(value.imag)
        for value in eigenvalues
        if value.imag > 0.0
    )
    assert positive_imaginary == pytest.approx(
        [frequencies.omega_minus, frequencies.omega_plus],
        rel=2.0e-13,
        abs=0.0,
    )


def test_independent_oracle_is_cold_and_has_no_langmuir_branch() -> None:
    case = HallCase(kdi=1.0)
    matrix = cold_transverse_matrix(case)

    eigenvalues = np.linalg.eigvals(matrix)
    absolute = np.sort(np.abs(eigenvalues.imag))
    frequencies = hall_frequencies(case)
    assert np.all(np.abs(eigenvalues.real) <= 2.0e-12 * frequencies.omega_plus)
    assert np.all(absolute > 0.0)
    assert np.all(absolute < frequencies.omega_plasma)


def test_wrong_sign_hall_control_is_detected() -> None:
    case = HallCase(kdi=10.0)
    correct = cold_transverse_matrix(case)
    wrong = cold_transverse_matrix(case, hall_sign=-1)

    scale = max(float(np.linalg.norm(correct, ord=2)), 1.0)
    assert np.linalg.norm(wrong - correct, ord=2) > 1.0e-3 * scale


@pytest.mark.parametrize("kdi", [0.1, 1.0, 10.0])
def test_actual_stage_matches_independent_transverse_continuum(
    kdi: float,
) -> None:
    case = HallCase(kdi=kdi)
    production, matrices, evidence = production_transverse_matrix(case)
    oracle = cold_transverse_matrix(case)

    oracle_scale = max(float(np.linalg.norm(oracle, ord=2)), 1.0)
    amplitude_errors = [
        float(np.linalg.norm(matrix - oracle, ord=2) / oracle_scale)
        for matrix in matrices
    ]
    assert len(amplitude_errors) >= 2
    assert max(amplitude_errors) < 2.0e-10

    production_eigenvalues = np.linalg.eigvals(production)
    oracle_eigenvalues = np.linalg.eigvals(oracle)
    production_positive = np.sort(
        np.abs(production_eigenvalues.imag)
    )[2:]
    oracle_positive = np.sort(np.abs(oracle_eigenvalues.imag))[2:]
    np.testing.assert_allclose(
        production_positive,
        oracle_positive,
        rtol=2.0e-11,
        atol=0.0,
    )

    hard = [record for record in evidence["diagnostics"] if record.hard]
    assert hard
    assert all(record.available and record.passed for record in hard)


def test_high_k_hall_branch_rises() -> None:
    frequencies = [
        hall_frequencies(HallCase(kdi=kdi))
        for kdi in (0.1, 1.0, 10.0)
    ]
    assert frequencies[0].omega_plus < frequencies[1].omega_plus
    assert frequencies[1].omega_plus < frequencies[2].omega_plus
    assert frequencies[2].omega_plus > 10.0 * frequencies[0].omega_plus


@pytest.mark.parametrize("integrator", ["midpoint", "rk4"])
def test_actual_time_advance_matches_integrator_polynomial(
    integrator: str,
) -> None:
    case = HallCase(kdi=1.0)
    results = [
        run_hall_time_case(
            case,
            integrator=integrator,
            steps_per_cycle=steps,
            cycles=1,
            relative_amplitude=2.0e-6,
        )
        for steps in (48, 96, 192)
    ]

    for result in results:
        assert result.finite
        assert result.accepted_steps == result.requested_steps
        assert result.source_is_none
        assert result.hard_checks_available
        assert result.warning
        assert result.relative_error < 2.0e-11

    assert results[-1].continuum_relative_error < {
        "midpoint": 2.0e-3,
        "rk4": 1.0e-7,
    }[integrator]
    assert (
        results[1].continuum_relative_error
        < 0.35 * results[0].continuum_relative_error
    )
    assert (
        results[2].continuum_relative_error
        < 0.35 * results[1].continuum_relative_error
    )
    assert results[0].continuum_relative_error > 1.0e-12


def test_wrong_sign_hall_control_fails_time_oracle() -> None:
    case = HallCase(kdi=1.0)
    result = run_hall_time_case(
        case,
        integrator="midpoint",
        steps_per_cycle=96,
        cycles=1,
        relative_amplitude=2.0e-6,
        oracle_hall_sign=-1,
    )
    assert result.initial_mode_error < 2.0e-12
    assert result.failure_code is not None
    assert result.failure_message


def test_initial_eigenmode_is_a_real_finite_state() -> None:
    case = HallCase(kdi=1.0)
    state, vector, eigenvalue = initial_hall_eigenmode(
        case, relative_amplitude=2.0e-6
    )
    assert vector.shape == (4,)
    assert eigenvalue.imag > 0.0
    assert np.all(np.isfinite(state.B))
    assert np.all(np.isfinite(state.fluid[0].mom))


def test_incompressible_toroidal_density_control_rejects_djfe_over_ne() -> None:
    evidence = closed_spatial_evidence(
        nodes=5,
        compressible=False,
    )
    assert evidence.minimum_density > 0.0
    assert evidence.theta_relative_error["all"] < 2.0e-10
    assert evidence.full_theta_relative_error["all"] < 2.0e-10
    assert evidence.retained_delta_relative_error["all"] < 2.0e-10
    assert evidence.wrong_djfe_relative_error["all"] > 1.0e-2


def test_closed_rectangle_pressure_gradient_and_ohm_consistency() -> None:
    evidence = closed_spatial_evidence(
        nodes=6,
        compressible=True,
    )
    # The adjoint/interpolated pressure path is lower order and wall-sensitive;
    # this bounded coarse grid does not support a global fourth-order claim.
    assert np.isfinite(evidence.pressure_force_error["all"])
    assert np.isfinite(evidence.pressure_force_error["inner"])
    assert evidence.pressure_force_error["all"] < 0.2
    assert evidence.pressure_force_error["inner"] < 0.2
    assert evidence.ohm_error["all"] < 1.0e-2
    assert evidence.ohm_error["inner"] < 1.0e-2
    assert evidence.ohm_norms["inner"]["entities"] > 0
    assert evidence.ohm_norms["wall"]["entities"] > 0
    assert not np.isclose(
        evidence.ohm_norms["inner"]["weighted_relative_l2"],
        evidence.ohm_norms["wall"]["weighted_relative_l2"],
    )
    assert evidence.full_theta_relative_error["all"] < 2.0e-10


def test_delta_theta_uses_padded_quotient_derivative() -> None:
    evidence = closed_spatial_evidence(
        nodes=5,
        compressible=True,
    )
    assert evidence.theta_relative_error["all"] < 2.0e-10
    assert evidence.retained_delta_relative_error["all"] < 2.0e-8


@pytest.mark.parametrize("nodes", [4, 6])
def test_spatial_mms_density_uses_fixed_physical_coordinates(nodes: int) -> None:
    evidence = closed_spatial_evidence(nodes=nodes, compressible=True)
    phi = 2.0 * np.pi * np.arange(5) / 5
    z = np.repeat((np.arange(nodes) + 0.5) / nodes, nodes)
    expected = 1.0e16 * (
        1.0 + 0.18 * np.cos(phi)[:, None] + 0.07 * np.sin(phi)[:, None]
    ) * (1.0 + 0.10 * np.cos(2.0 * np.pi * z)[None, :])
    np.testing.assert_allclose(
        evidence.arrays["density"].reshape(5, -1), expected, rtol=2.0e-15
    )


def test_variable_density_velocity_uses_samplewise_broadcast() -> None:
    evidence = variable_density_velocity_evidence(nodes=6)
    assert evidence.density_min > 0.0
    assert evidence.density_max > evidence.density_min
    assert evidence.relative_error < 2.0e-10


def test_drift_ohm_residual_rejects_doubled_e0() -> None:
    evidence = drift_ohm_tautology_evidence()
    assert evidence.electric_field_linf > 10.0 * evidence.hard_threshold
    assert evidence.baseline_linf < evidence.hard_threshold
    assert evidence.doubled_linf > evidence.hard_threshold
    assert evidence.gate_failed
    assert evidence.unit == "V/m"


def test_actual_qhfe_reconstruction_converges_on_fixed_domain() -> None:
    levels = [
        reconstruction_evidence(nodes=nodes)
        for nodes in (4, 6, 8)
    ]
    for level in levels:
        assert level.minimum_density > 0.0
        assert level.delta_theta_norms["all"]["unit"] == "1/s"
        assert level.maximum_abs_current > 0.0
        assert level.current_relation_error < 1.0e-10
        assert level.relative_wall_flux < 2.0e-10
        for region in ("all", "inner"):
            assert (
                level.velocity_norms[region]["weighted_relative_l2"]
                > 1.0e-6
            )
            assert (
                level.theta_norms[region]["weighted_relative_l2"]
                > 1.0e-6
            )
    for field in ("velocity", "theta"):
        for region in ("all", "inner"):
            errors = [
                level.__dict__[f"{field}_norms"][region][
                    "weighted_relative_l2"
                ]
                for level in levels
            ]
            assert errors[1] < errors[0]
            assert errors[2] < errors[1]
    for region in ("all", "inner"):
        deltas = [
            level.delta_theta_norms[region]["weighted_relative_l2"]
            for level in levels
        ]
        assert deltas[1] < deltas[0]
        assert deltas[2] < deltas[1]
    assert levels[0].velocity_norms["all"]["weighted_relative_l2"] > 1.0e-3


def test_spatial_pressure_and_ohm_norms_converge_by_region() -> None:
    levels = [
        closed_spatial_evidence(nodes=nodes, compressible=True)
        for nodes in (4, 6, 8)
    ]
    for field in ("pressure_force_error", "ohm_error"):
        for region in ("all", "inner"):
            errors = [
                level.__dict__[field][region]
                for level in levels
            ]
            assert errors[1] < errors[0]
            assert errors[2] < errors[1]
    for level in levels:
        assert level.pressure_norms["all"]["unit"] == "Pa/m"
        assert level.ohm_norms["all"]["unit"] == "V/m"
        assert level.ohm_norms["inner"]["entities"] > 0
        assert level.ohm_norms["wall"]["entities"] > 0


def test_region_norm_marks_empty_region_unavailable() -> None:
    values = np.zeros((1, 1, 3), dtype=np.complex128)
    result = _region_norms(
        values,
        values,
        {"empty": np.zeros(3, dtype=bool)},
        np.ones(3),
        unit="1/s",
    )
    assert result["empty"]["available"] is False
    assert result["empty"]["entities"] == 0
    assert result["empty"]["weighted_relative_l2"] is None


def test_pressure_bump_vanishes_with_derivative_at_real_walls() -> None:
    for nodes in (4, 6, 8):
        context = _closed_context(nodes)
        r_mask = np.asarray(context.geometry.r_faces.boundary, dtype=bool)
        z_mask = np.asarray(context.geometry.z_faces.boundary, dtype=bool)
        r_wall = np.asarray(context.geometry.r_faces.R)[r_mask]
        z_wall = np.asarray(context.geometry.z_faces.Z)[z_mask]
        radial = _pressure_bump_derivatives(
            r_wall, np.full_like(r_wall, 0.37), np.array([0.0])
        )
        axial = _pressure_bump_derivatives(
            np.array([1.37]), z_wall, np.array([0.0])
        )
        np.testing.assert_allclose(radial[0], 0.0, atol=2.0e-14)
        np.testing.assert_allclose(radial[1], 0.0, atol=2.0e-13)
        np.testing.assert_allclose(axial[0], 0.0, atol=2.0e-14)
        np.testing.assert_allclose(axial[3], 0.0, atol=2.0e-13)
        assert min(context.geometry.cells.R) > 1.0
        assert max(context.geometry.cells.R) < 2.0
        assert min(context.geometry.cells.Z) > 0.0
        assert max(context.geometry.cells.Z) < 1.0


def test_cli_refuses_cuda_without_slurm_allocation() -> None:
    source_root = Path(__file__).resolve().parents[2]
    output = source_root / "evidence" / "cuda-refusal-not-created"
    assert not output.exists()
    environment = dict(os.environ)
    environment.pop("SLURM_JOB_ID", None)
    process = subprocess.run(
        [
            sys.executable,
            str(source_root / "tools" / "warp_massless_verify.py"),
            "--backend",
            "cuda:0",
            "--output",
            str(output),
            "--hall-only",
        ],
        cwd=source_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 2
    assert "SLURM_JOB_ID" in process.stderr
    assert "no CPU fallback" in process.stderr
    assert not output.exists()


def test_cli_refuses_existing_output_directory() -> None:
    source_root = Path(__file__).resolve().parents[2]
    process = subprocess.run(
        [
            sys.executable,
            str(source_root / "tools" / "warp_massless_verify.py"),
            "--backend",
            "cpu",
            "--output",
            str(source_root),
            "--hall-only",
        ],
        cwd=source_root,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 2
    assert "refusing existing output directory" in process.stderr


def test_runtime_finiteness_uses_backend_reduction() -> None:
    class _Reduced:
        def item(self) -> bool:
            return True

    class _Xp:
        def __init__(self) -> None:
            self.inputs: list[object] = []

        def isfinite(self, value: object) -> object:
            self.inputs.append(value)
            return value

        def all(self, value: object) -> _Reduced:
            assert value in self.inputs
            return _Reduced()

    xp = _Xp()
    context = type(
        "FakeContext",
        (),
        {"compute": type("FakeBackend", (), {"xp": xp})()},
    )()
    magnetic = object()
    momentum = object()
    state = type(
        "FakeState",
        (),
        {
            "B": magnetic,
            "fluid": (type("FakeFluid", (), {"mom": momentum})(),),
        },
    )()

    assert _runtime_finite(context, state)
    assert xp.inputs == [magnetic, momentum]


def test_npz_output_rejects_unconverted_runtime_array(tmp_path) -> None:
    source_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "warp_massless_verify_for_test",
        source_root / "tools" / "warp_massless_verify.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "unconverted-array-must-not-exist.npz"
    assert not path.exists()

    with pytest.raises(TypeError, match="converted explicitly"):
        module._save_npz(path, {"runtime_owned": object()})
    assert not path.exists()


@pytest.mark.parametrize("capture", [False, True])
def test_npz_output_rejects_object_dtype_before_writing(tmp_path, capture):
    source_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "warp_massless_verify_dtype_test",
        source_root / "tools" / "warp_massless_verify.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "object-must-not-exist.npz"
    with pytest.raises(TypeError, match="object dtype"):
        module._save_npz(
            output,
            {"bad": np.array([object()], dtype=object)},
            allow_runtime_capture=capture,
        )
    assert not output.exists()


def test_complete_diagnostic_serialization_retains_r5_bound_fields() -> None:
    source_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "warp_massless_verify_r5_test",
        source_root / "tools" / "warp_massless_verify.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = HallCase()
    context = fixture.build_periodic_context(case)
    result = fixture.evaluate(fixture.uniform_state(context, case), context)
    record = next(
        item
        for item in result.checks
        if item.name == "wall_field_reaction_work"
        and item.region == "all"
    )
    serialized = module._json_value(record)
    assert serialized["unit"] == "W"
    assert "compatibility_bound" in serialized
    assert "arithmetic_bound" in serialized
