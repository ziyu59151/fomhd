"""Behavioral tests for the controlled T0 temporal benchmark."""

from __future__ import annotations

import gzip
import importlib
import json
import math

import numpy as np
import pytest

from fomhd.types import Failure


def module():
    return importlib.import_module("tools.warp_t0_controlled")


@pytest.fixture(scope="module")
def audit():
    return module().build_audit(points=9)


def test_controls_reject_cuda_and_nondivisible_horizon_before_output(tmp_path):
    tool = module()
    cuda_output = tmp_path / "cuda"
    with pytest.raises(ValueError, match="CPU"):
        tool.run_case(
            cuda_output,
            device="cuda:0",
            coupling="predictor",
            micro_per_period=2048,
            micro_steps=16,
            cycles=2,
            seed="target",
            points=9,
        )
    assert not cuda_output.exists()

    invalid_output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="divi"):
        tool.run_case(
            invalid_output,
            device="cpu",
            coupling="predictor",
            micro_per_period=2048,
            micro_steps=17,
            cycles=2,
            seed="target",
            points=9,
        )
    assert not invalid_output.exists()


def test_existing_output_is_refused_without_mutation(tmp_path):
    marker = tmp_path / "preserve.txt"
    marker.write_text("original", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        module().run_audit(tmp_path, points=9)

    assert marker.read_text(encoding="utf-8") == "original"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["preserve.txt"]


def test_half_microstep_schedule_preserves_same_macro_horizon():
    tool = module()
    coarse = tool.schedule(
        coupling="predictor",
        micro_per_period=2048,
        micro_steps=256,
        cycles=2,
        points=9,
    )
    fine = tool.schedule(
        coupling="predictor",
        micro_per_period=4096,
        micro_steps=512,
        cycles=2,
        points=9,
    )

    assert coarse["macro_dt_s"] == pytest.approx(fine["macro_dt_s"])
    assert coarse["horizon_s"] == pytest.approx(fine["horizon_s"])
    assert coarse["macro_steps"] == fine["macro_steps"]
    assert fine["micro_dt_s"] == pytest.approx(coarse["micro_dt_s"] / 2)
    assert fine["micro_steps"] == 2 * coarse["micro_steps"]


def test_actual_periodic_derivative_recovers_spatial_wavenumbers(audit):
    case = audit["case"]
    for harmonic, expected in audit["derivative_wavenumbers"].items():
        assert expected["effective_wavenumber_per_m"] == pytest.approx(
            harmonic * case.k, rel=2.0e-14, abs=1.0e-14
        )
        assert expected["relative_error"] < 2.0e-14


def test_constant_harmonic_projection_uses_one_over_n():
    tool = module()
    constant = np.full(9, 3.5, dtype=np.complex128)

    assert tool.project_harmonic(constant, 0, 9)[0] == pytest.approx(3.5)
    assert tool.project_harmonic(constant, 1, 9)[0] == pytest.approx(
        0.0, abs=1.0e-14
    )


def test_full_jacobian_transverse_block_matches_independent_hall(audit):
    for harmonic in range(1, audit["harmonic_max"] + 1):
        full = audit["full_jacobians"][harmonic]
        transverse = full[np.ix_([1, 2, 7, 8], [1, 2, 7, 8])]
        independent = audit["independent_matrices"][harmonic]
        relative = np.linalg.norm(transverse - independent) / np.linalg.norm(
            independent
        )
        assert relative < 2.0e-9
        assert audit["jacobian_epsilon_difference"][harmonic] < 1.0e-7


def test_micro_jacobian_is_the_full_electron_field_block(audit):
    indexes = [5, 6, 7, 8]
    for harmonic in range(1, audit["harmonic_max"] + 1):
        full = audit["full_jacobians"][harmonic]
        micro = audit["micro_jacobians"][harmonic]
        np.testing.assert_allclose(
            micro,
            full[np.ix_(indexes, indexes)],
            rtol=0.0,
            atol=2.0e-8,
        )
        full_frequency = np.max(np.abs(full.imag))
        micro_frequency = np.max(np.abs(micro.imag))
        assert full_frequency >= micro_frequency


def test_full_spectrum_retains_pressure_modes_absent_from_transverse_reference(
    audit,
):
    for harmonic in range(1, audit["harmonic_max"] + 1):
        full_imag = np.sort(audit["full_eigenvalues"][harmonic].imag)
        independent_imag = np.sort(
            audit["independent_eigenvalues"][harmonic].imag
        )
        assert full_imag.size == 9
        assert independent_imag.size == 4
        assert np.count_nonzero(np.abs(full_imag) < 1.0e6) >= 1


def test_one_rk4_step_matches_its_definition():
    tool = module()
    argument = 0.1 + 0.2j

    observed = tool.rk4_scalar(argument)
    expected = (
        1.0
        + argument
        + argument**2 / 2.0
        + argument**3 / 6.0
        + argument**4 / 24.0
    )

    assert observed == pytest.approx(expected, rel=0.0, abs=0.0)


def test_linear_t0_map_matches_known_oscillator_coefficients():
    tool = module()
    h = 0.1
    matrix = tool.linear_t0_macro_map(
        np.zeros((1, 1), dtype=np.complex128),
        np.ones((1, 1), dtype=np.complex128),
        -np.ones((1, 1), dtype=np.complex128),
        np.zeros((1, 1), dtype=np.complex128),
        h,
        micro_steps=1,
    )

    np.testing.assert_allclose(
        matrix,
        [
            [1.0 - h * h / 2.0, h - h**3 / 6.0],
            [-h, 1.0 - h * h / 2.0],
        ],
        rtol=0.0,
        atol=2.0e-16,
    )


def test_production_one_macro_map_matches_exact_linear_t0_algebra():
    tool = module()
    comparison = tool.production_macro_map_comparison(
        points=9,
        harmonic=1,
        micro_steps=2,
    )

    assert comparison["relative_error"] < 5.0e-5
    assert comparison["linear_map"].shape == (9, 9)
    assert comparison["production_map"].shape == (9, 9)


def test_mixed_seed_projects_to_exact_independent_harmonic_amplitudes():
    tool = module()
    seed = tool.build_seed(points=9, seed="mixed")
    coefficients = seed["coefficients"]

    assert np.max(np.abs(coefficients[1])) == pytest.approx(2.0e-6)
    assert np.max(np.abs(coefficients[4])) == pytest.approx(2.0e-8)
    np.testing.assert_allclose(
        seed["vectors"][1],
        seed["coefficients"][1] / seed["amplitudes"][1],
        rtol=0.0,
        atol=2.0e-16,
    )
    np.testing.assert_allclose(
        seed["vectors"][4],
        seed["coefficients"][4] / seed["amplitudes"][4],
        rtol=0.0,
        atol=2.0e-16,
    )
    for harmonic in (1, 4):
        np.testing.assert_array_equal(
            seed["vectors"][harmonic][[0, 3, 4, 5, 6]], 0.0
        )


def test_analytic_reference_uses_cold_eigenvalue_not_production_value():
    tool = module()
    transverse = np.asarray([0.6, -0.6j, 1.0, -1.0j])
    full_vector = np.zeros(9, dtype=np.complex128)
    full_vector[list(tool.TRANSVERSE_INDEXES)] = transverse
    spec = {
        "amplitude": 2.0e-6,
        "vector": full_vector,
        "eigenvalue": 0.25j,
        "production_eigenvalue_for_negative_control": 0.90j,
    }

    reference = tool._references_at({1: spec}, np.asarray([0.0, 4.0]))[1]

    np.testing.assert_allclose(
        reference,
        [
            2.0e-6 * full_vector,
            2.0e-6 * full_vector * np.exp(1.0j),
        ],
        rtol=0.0,
        atol=2.0e-20,
    )


def test_mixed_initial_reference_error_is_per_harmonic_roundoff():
    tool = module()
    seed = tool.build_seed(points=9, seed="mixed")
    context = seed["context"]
    cumulative = __import__("fomhd.types", fromlist=["cumulative_record"])
    committed = tool.initialize(
        seed["state"],
        context,
        cumulative.cumulative_record(),
        equilibrium=False,
    )

    for harmonic in (1, 4):
        actual = tool.project_state(
            committed.state, context, seed["case"], harmonic
        )
        expected = seed["coefficients"][harmonic]
        assert np.max(np.abs(actual - expected)) < 1.0e-15


def test_phase_history_is_unwrapped_across_multiple_turns():
    tool = module()
    turns = np.asarray([0.0, 2.0, 4.0, 6.0])
    observed = np.exp(1j * (turns - 0.05))
    reference = np.exp(1j * turns)

    phase = tool.unwrapped_phase(observed, reference)

    np.testing.assert_allclose(
        phase, np.full(4, -0.05), atol=1.0e-15, rtol=0.0
    )


@pytest.mark.parametrize(
    "vector_error,amplitude_drift,phase_error,expected",
    [
        (0.0, 0.0, 0.0, True),
        (0.010000001, 0.0, 0.0, False),
        (0.0, -0.010000001, 0.0, False),
        (0.0, 0.0, 0.100000001, False),
    ],
)
def test_accuracy_qualification_uses_contract_criteria(
    vector_error, amplitude_drift, phase_error, expected
):
    result = module().qualify_accuracy(
        vector_error=vector_error,
        amplitude_drift=amplitude_drift,
        phase_error_rad=phase_error,
    )

    assert result["joint_qualified"] is expected
    assert result["criteria"]["max_target_modal_vector_error"] == 0.01
    assert result["criteria"]["max_target_amplitude_drift"] == 0.01
    assert result["criteria"]["max_unwrapped_phase_error_rad"] == 0.1


@pytest.mark.parametrize("seed", ["target", "mixed"])
def test_successful_case_writes_raw_checks_history_and_accuracy(
    tmp_path, monkeypatch, seed
):
    tool = module()
    original_schedule = tool.schedule

    def short_schedule(**kwargs):
        value = original_schedule(**kwargs)
        macro_dt = 1.0e-10
        macro_steps = 2
        value.update(
            macro_dt_s=macro_dt,
            micro_dt_s=macro_dt,
            horizon_s=macro_steps * macro_dt,
            total_micro_steps=macro_steps,
            macro_steps=macro_steps,
        )
        return value

    monkeypatch.setattr(tool, "schedule", short_schedule)
    output = tmp_path / f"success-{seed}"
    summary = tool.run_case(
        output,
        device="cpu",
        coupling="predictor",
        micro_per_period=2048,
        micro_steps=1,
        cycles=1,
        seed=seed,
        points=9,
        progress_every=0,
    )

    assert summary["status"].startswith("COMPLETED")
    assert summary["accepted_macro_steps"] == 2
    assert summary["final_endpoint_available"] is True
    assert summary["accuracy"]["joint_qualified"] is True
    assert summary["check_summary"]["hard_failed"] == 0
    if seed == "mixed":
        assert summary["high_frequency_growth"]["available"] is True
    else:
        assert summary["high_frequency_growth"]["available"] is False
        assert (
            summary["high_frequency_growth"]["amplitude_ratio_history"]
            is None
        )
    alias = summary["macro_alias_by_harmonic"]["1"]
    assert alias["omega_macro_dt_rad"] > 0.0
    assert alias["macro_nyquist_alias_rad_s"] >= 0.0
    assert (output / "case.json").is_file()
    checks_path = output / "checks.jsonl.gz"
    assert checks_path.is_file()
    with gzip.open(checks_path, "rt", encoding="utf-8") as stream:
        assert sum(1 for _ in stream) == summary["counts"]["raw_check_records"]
    assert not (output / "failure.json").exists()
    with np.load(output / "history.npz", allow_pickle=False) as history:
        assert history["accepted_time_s"].size == 3
        assert history["target_coeff_history"].shape == (3, 4)
        assert "totals_total_energy_J_history" in history.files
        for key in (
            "cumulative_baseline_divB",
            "cumulative_integral_0_electric_node",
            "cumulative_integral_1_electric_node",
            "cumulative_boundary_field_energy",
            "cumulative_last_step",
            "cumulative_transaction_accepted",
        ):
            assert key in history.files
        assert int(history["cumulative_last_step"]) == 2
        assert bool(history["cumulative_transaction_accepted"]) is True
        if seed == "mixed":
            assert history["high_coeff_history"].shape == (3, 4)
            assert history["high_amplitude_ratio_history"].shape == (3,)
    case_payload = json.loads(
        (output / "case.json").read_text(encoding="utf-8")
    )
    assert case_payload["cumulative"]["transaction_status"]["accepted"] is True
    assert case_payload["cumulative"]["identity"]["last_step"] == 2


def test_injected_rejection_is_undetermined_and_retains_accepted_prefix(
    tmp_path, monkeypatch
):
    tool = module()
    real_advance = tool.advance
    calls = 0

    def reject_second(committed, cumulative, context, source=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_advance(committed, cumulative, context, source)
        failure = Failure("InjectedRejection", "synthetic second-macro rejection")
        failure.partial_arrays = {
            "rejected_trial_array": np.asarray([3.0, 4.0])
        }
        raise failure

    monkeypatch.setattr(tool, "advance", reject_second)
    output = tmp_path / "rejected"
    summary = tool.run_case(
        output,
        device="cpu",
        coupling="predictor",
        micro_per_period=2048,
        micro_steps=1,
        cycles=1,
        seed="target",
        points=9,
        progress_every=0,
    )

    payload = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert summary["status"] == "UNDETERMINED"
    assert summary["accepted_macro_steps"] == 1
    assert summary["final_endpoint_available"] is False
    assert summary["invented_rejected_endpoint"] is False
    assert payload["failure_code"] == "InjectedRejection"
    assert payload["accepted_macro_steps"] == 1
    assert payload["final_endpoint_available"] is False
    assert "rejected_trial_array" in payload["partial_array_names"]
    assert summary["accuracy"]["full_window_status"] == "UNDETERMINED"
    assert summary["accuracy"]["partial"] is True
    with np.load(output / "history.npz", allow_pickle=False) as history:
        assert history["accepted_time_s"].size == 2
        assert history["target_reference_history"].shape == (2, 4)
        assert history["target_vector_error_history"].shape == (2,)
        assert history["high_coeff_history"].shape == (2, 4)
        assert history["high_abs_amplitude_history"].shape == (2, 4)
        assert int(history["cumulative_last_step"]) == 1
        assert bool(history["cumulative_transaction_accepted"]) is True
        assert history["cumulative_baseline_divB"].size > 0
        np.testing.assert_array_equal(
            history["rejected_trial_array"], [3.0, 4.0]
        )
    with np.load(output / "failure.npz", allow_pickle=False) as failure_arrays:
        np.testing.assert_array_equal(
            failure_arrays["rejected_trial_array"], [3.0, 4.0]
        )
    assert "endpoint_fields" not in payload


def test_initialization_rejection_writes_failure_and_empty_shaped_history(
    tmp_path, monkeypatch
):
    tool = module()

    def reject_initialization(**kwargs):
        failure = Failure("InitialRejected", "synthetic initialization rejection")
        failure.partial_arrays = {
            "initial_rejected_array": np.asarray([7.0])
        }
        raise failure

    monkeypatch.setattr(tool, "_prepare_case", reject_initialization)
    output = tmp_path / "initial-rejection"
    summary = tool.run_case(
        output,
        device="cpu",
        coupling="predictor",
        micro_per_period=2048,
        micro_steps=1,
        cycles=1,
        seed="target",
        points=9,
        progress_every=0,
    )

    assert summary["status"] == "UNDETERMINED"
    assert summary["accepted_macro_steps"] == 0
    assert summary["final_endpoint_available"] is False
    assert (output / "case.json").is_file()
    assert (output / "failure.json").is_file()
    assert (output / "checks.jsonl.gz").is_file()
    with np.load(output / "history.npz", allow_pickle=False) as history:
        assert history["target_coeff_history"].shape == (0, 4)
        assert history["high_coeff_history"].shape == (0, 4)
        assert history["high_abs_amplitude_history"].shape == (0, 4)
        np.testing.assert_array_equal(history["initial_rejected_array"], [7.0])


def test_matched_reference_times_are_aligned_exactly():
    tool = module()
    candidate = np.asarray([0.0, 1.0, 2.0])
    reference_times = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0])
    values = np.arange(20, dtype=np.complex128).reshape(5, 4)

    aligned_times, aligned_values = tool.align_reference_history(
        candidate,
        {"accepted_time_s": reference_times, "high_coeff_history": values},
        "high_coeff_history",
    )

    np.testing.assert_array_equal(aligned_times, candidate)
    np.testing.assert_array_equal(aligned_values, values[[0, 2, 4]])
    with pytest.raises(ValueError, match="missing candidate time"):
        tool.align_reference_history(
            candidate + 0.25,
            {"accepted_time_s": reference_times, "high_coeff_history": values},
            "high_coeff_history",
        )


def test_target_only_high_mode_uses_absolute_matched_control_not_ratio():
    tool = module()
    history = np.zeros((2, 4), dtype=np.complex128)
    reference = np.ones((2, 4), dtype=np.complex128) * 1.0e-12

    result = tool._growth_summary(
        history,
        np.asarray([0.0, 1.0]),
        reference_history=reference,
        initial_ratio_defined=False,
    )

    assert result["available"] is False
    assert result["amplitude_ratio_history"] is None
    assert result["matched_synchronous"]["amplitude_ratio_to_reference"] is None
    assert result["matched_synchronous"][
        "maximum_absolute_norm_difference"
    ] == pytest.approx(np.sqrt(4.0) * 1.0e-12)


def test_zero_mode_failure_is_masked_not_reported_as_eigenvalue(audit):
    zero = audit["zero_mode"]
    assert zero["eigenvalues_available"] is False
    assert bool(audit["zero_mode_valid_columns"][3]) is False
    assert zero["failures"]["3"]["failure_code"] == "FluidDivision"
    assert np.all(np.isnan(audit["zero_mode_jacobian"][:, 3]))


def test_whole_map_powers_and_seed_gains_are_retained(audit):
    whole = audit["whole_map"]
    for micro_steps in module().DEFAULT_MAIN_MICRO_STEPS:
        record = whole[str(micro_steps)]
        assert record["macro_steps"] == 4096 // micro_steps
        for harmonic in range(1, audit["harmonic_max"] + 1):
            mode = record["harmonics"][str(harmonic)]
            assert mode["map"].shape == (9, 9)
            assert mode["horizon_map"].shape == (9, 9)
            assert mode["seed_gain"] > 0.0
            assert mode["per_macro_spectral_radius"] > 0.0
            assert mode["horizon_spectral_radius"] > 0.0
            assert mode["spectral_radius_power_over_horizon"] > 0.0
            assert "power_bound_over_horizon" not in mode


def test_matched_transient_growth_keeps_maximum_and_final_flags():
    tool = module()
    history = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    reference = history.copy()
    reference[1] = [1.0, 0.0, 0.0, 0.0]

    result = tool._growth_summary(
        history,
        np.asarray([0.0, 0.5, 1.0]),
        reference_history=reference,
        initial_ratio_defined=True,
    )
    matched = result["matched_synchronous"]

    assert matched["maximum_amplitude_ratio_to_reference"] == pytest.approx(2.0)
    assert matched["final_amplitude_ratio_to_reference"] == pytest.approx(1.0)
    assert matched["maximum_amplification_over_1_percent"] is True
    assert matched["final_amplification_over_1_percent"] is False
    assert matched["transient_amplification_over_1_percent"] is True
    assert matched["amplification_over_1_percent"] is True


def _write_matched_reference_fixture(tmp_path, tool):
    case = tool._periodic_case(9)
    built = tool.build_seed(points=9, seed="target")
    horizon = 2.0 * math.pi / float(
        built["eigenvalues"][1].imag
    )
    metadata = {
        "status": "COMPLETED",
        "controls": {
            "coupling": "synchronous",
            "seed": "target",
            "points": 9,
            "target_amplitude": tool.TARGET_AMPLITUDE,
            "high_amplitude": 0.0,
            "target_harmonic": 1,
            "high_harmonic": 4,
        },
        "schedule": {"horizon_s": horizon},
        "source": tool._provenance(["reference"], "cpu"),
        "physical": {
            "kdi": case.kdi,
            "points": case.points,
            "n0": case.n0,
            "B0": case.B0,
            "pressure": case.pressure,
            "di": float(case.di),
            "k": float(case.k),
            "wavelength": float(case.wavelength),
            "K": 0,
            "Nphi": 1,
        },
        "discrete": {
            "points": 9,
            "K": 0,
            "Nphi": 1,
            "harmonics": [1, 4],
            "target_amplitude": tool.TARGET_AMPLITUDE,
            "high_amplitude": 0.0,
        },
        "seed": {
            "normalization": built["normalization"],
            "vectors": tool._jsonable(built["vectors"]),
            "transverse_vectors": tool._jsonable(
                built["transverse_vectors"]
            ),
            "eigenvalues": tool._jsonable(built["eigenvalues"]),
            "amplitudes": built["amplitudes"],
            "projected_coefficients": tool._jsonable(
                built["projected_coefficients"]
            ),
        },
    }
    times = np.asarray([0.0, horizon])
    high = np.zeros((2, 4), dtype=np.complex128)
    output = tmp_path / "matched-reference"
    output.mkdir()
    (output / "case.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    np.savez(
        output / "history.npz",
        accepted_time_s=times,
        high_coeff_history=high,
    )
    return output, metadata, horizon


@pytest.mark.parametrize(
    "mismatch,message",
    [
        ("source_hash", "numerical runtime source hashes"),
        ("seed_vector", "seed vectors"),
        ("physical", "physical parameters"),
        ("discrete", "discrete parameters"),
        ("coverage", "reference time coverage"),
    ],
)
def test_matched_reference_rejects_identity_or_coverage_mismatch(
    tmp_path, mismatch, message
):
    tool = module()
    output, metadata, horizon = _write_matched_reference_fixture(
        tmp_path, tool
    )
    tool._load_matched_reference(
        output,
        seed="target",
        horizon=horizon,
        points=9,
    )
    if mismatch == "source_hash":
        metadata["source"]["source_hashes"]["src/fomhd/time.py"] = "0" * 64
    elif mismatch == "seed_vector":
        metadata["seed"]["vectors"]["1"][1]["real"] += 1.0e-12
    elif mismatch == "physical":
        metadata["physical"]["n0"] *= 1.01
    elif mismatch == "discrete":
        metadata["discrete"]["Nphi"] = 2
    else:
        with np.load(output / "history.npz", allow_pickle=False) as arrays:
            np.savez(
                output / "history.npz",
                accepted_time_s=np.asarray([0.0, 0.5 * horizon]),
                high_coeff_history=np.asarray(
                    arrays["high_coeff_history"]
                ),
            )
    (output / "case.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        tool._load_matched_reference(
            output,
            seed="target",
            horizon=horizon,
            points=9,
        )


def test_frozen_ion_micro_control_keeps_p_fixed_and_matches_reference(
    tmp_path, monkeypatch
):
    tool = module()

    def forbidden_advance(*args, **kwargs):
        raise AssertionError("frozen micro control must not call time.advance")

    monkeypatch.setattr(tool, "advance", forbidden_advance)
    output = tmp_path / "frozen"
    summary = tool.run_frozen_micro_control(
        output,
        micro_per_period=2048,
        steps=2,
        points=9,
    )

    assert summary["kind"] == "frozen_ion_micro_control"
    assert summary["status"].startswith("COMPLETED")
    assert summary["accepted_micro_steps"] == 2
    assert summary["check_summary"]["hard_failed"] == 0
    assert summary["final_errors"]["exponential_relative"] < 2.0e-5
    assert summary["final_errors"]["rk4_relative"] < 1.0e-12
    assert summary["p_fixed_max_abs"] == 0.0
    with np.load(output / "history.npz", allow_pickle=False) as history:
        assert history["p_fixed_error_history"].max() == 0.0
        assert np.max(np.abs(history["seed_state9"][:5])) < 1.0e-15
        assert history["z_coeff_history"].shape == (3, 4)
