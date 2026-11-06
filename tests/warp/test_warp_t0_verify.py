"""Behavioral tests for the bounded A1-T0 evidence harness."""

from __future__ import annotations

import importlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from fomhd.types import Failure


def module():
    return importlib.import_module("tools.warp_t0_verify")


def test_existing_output_directory_is_refused_without_mutation(tmp_path):
    marker = tmp_path / "preserve.txt"
    marker.write_text("original", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        module().run_scan(tmp_path, device="cpu")

    assert marker.read_text(encoding="utf-8") == "original"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["preserve.txt"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"device": "cuda:1"},
        {"points": 8},
        {"points": 10},
        {"relative_amplitude": 0.0},
        {"cycles": 0.0},
        {"cycles": 1.1},
        {"micro_steps_per_cycle": 63},
        {"micro_steps": ()},
        {"micro_steps": (1, 1)},
        {"micro_steps": (0,)},
        {"repetitions": 0},
        {"closed_micro_steps": 0},
        {"closed_macro_dt": 0.0},
        {"closed_horizon": 3.0e-10, "closed_macro_dt": 2.0e-10},
    ],
)
def test_invalid_controls_do_not_create_output(tmp_path, overrides):
    output = tmp_path / "must-not-exist"
    arguments = {
        "device": "cpu",
        "points": 9,
        "relative_amplitude": 2.0e-6,
        "cycles": 1.0,
        "micro_steps_per_cycle": 64,
        "micro_steps": (1, 2, 4, 8),
        "repetitions": 1,
        "closed_micro_steps": 4,
        "closed_macro_dt": 2.0e-10,
        "closed_horizon": 8.0e-10,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError):
        module().run_scan(output, **arguments)

    assert not output.exists()


def test_modal_metrics_report_amplitude_and_signed_principal_phase():
    tool = module()
    reference = np.asarray([1.0 + 2.0j, -0.5 + 0.3j])
    actual = 1.25 * np.exp(0.37j) * reference

    metrics = tool.modal_metrics(actual, reference)

    assert metrics["amplitude_ratio"] == pytest.approx(1.25, rel=1e-14)
    assert metrics["signed_phase_rad"] == pytest.approx(0.37, abs=1e-14)
    assert metrics["principal_range_rad"] == [-math.pi, math.pi]
    assert "modulo 2*pi" in metrics["phase_convention"]
    assert "principal" in metrics["principal_ambiguity"]

    wrapped = tool.modal_metrics(
        np.exp(3.5j) * reference,
        reference,
    )
    assert wrapped["signed_phase_rad"] == pytest.approx(
        3.5 - 2.0 * math.pi,
        abs=1e-14,
    )


def test_analytic_mode_uses_positive_exponential_continuation():
    tool = module()
    initial = np.asarray([0.2 - 0.4j, 1.1 + 0.7j])
    eigenvalue = 0.15 + 0.8j
    horizon = 1.75

    endpoint = tool.analytic_mode(initial, eigenvalue, horizon)

    np.testing.assert_allclose(
        endpoint,
        initial * np.exp(eigenvalue * horizon),
        rtol=0.0,
        atol=0.0,
    )


def test_time_domain_mode_uses_real_cosine_sine_samples():
    tool = module()
    vector = np.asarray(
        [0.4 + 0.2j, -0.3 + 0.7j, 1.0 - 0.1j, 0.2 - 0.8j]
    )
    amplitude = 2.0e-6
    factor = np.exp(0.43j)
    points = 9

    samples = tool.mode_samples(
        vector,
        amplitude=amplitude,
        factor=factor,
        points=points,
    )

    assert np.max(np.abs(samples.imag)) == 0.0
    coordinate = 2.0 * math.pi * np.arange(points) / points
    cosine = 2.0 * np.dot(samples.real, np.cos(coordinate)) / points
    sine = 2.0 * np.dot(samples.real, np.sin(coordinate)) / points
    recovered = np.asarray(cosine - 1j * sine)
    np.testing.assert_allclose(
        recovered,
        amplitude * vector * factor,
        rtol=0.0,
        atol=8.0 * np.finfo(float).eps,
    )


def _field_error_inputs():
    baseline = {
        "n_i": np.asarray([1.0e8], dtype=np.complex128),
        "mom_i": np.asarray([3.0e5], dtype=np.complex128),
        "p_i": np.asarray([4.0e2], dtype=np.complex128),
        "p_perp_e": np.asarray([5.0e2], dtype=np.complex128),
        "p_parallel_e": np.asarray([6.0e2], dtype=np.complex128),
        "B": np.asarray([7.0e-1], dtype=np.complex128),
    }
    reference_delta = {
        "n_i": np.asarray([0.0], dtype=np.complex128),
        "mom_i": np.asarray([1.0], dtype=np.complex128),
        "p_i": np.asarray([0.0], dtype=np.complex128),
        "p_perp_e": np.asarray([0.0], dtype=np.complex128),
        "p_parallel_e": np.asarray([0.0], dtype=np.complex128),
        "B": np.asarray([2.0], dtype=np.complex128),
    }
    error = {
        "n_i": np.asarray([0.25], dtype=np.complex128),
        "mom_i": np.asarray([0.25], dtype=np.complex128),
        "p_i": np.asarray([-0.003], dtype=np.complex128),
        "p_perp_e": np.asarray([0.004], dtype=np.complex128),
        "p_parallel_e": np.asarray([-0.005], dtype=np.complex128),
        "B": np.asarray([0.01], dtype=np.complex128),
    }
    reference = {
        name: baseline[name] + reference_delta[name]
        for name in baseline
    }
    actual = {
        name: reference[name] + error[name]
        for name in baseline
    }
    scales = {
        "n_i": 1.0,
        "mom_i": 1.0,
        "p_i": 1.0,
        "p_perp_e": 1.0,
        "p_parallel_e": 1.0,
        "B": 1.0,
    }
    return actual, reference, baseline, scales


def test_normalized_field_errors_report_si_fields_and_amplitude_floors():
    tool = module()
    actual, reference, baseline, scales = _field_error_inputs()

    result = tool.normalized_field_errors(
        actual,
        reference,
        baseline,
        physical_scales=scales,
    )

    assert set(result["per_field"]) == {
        "n_i",
        "mom_i",
        "p_i",
        "p_perp_e",
        "p_parallel_e",
        "B",
    }
    assert result["per_field"]["n_i"]["unit"] == "1/m^3"
    assert result["per_field"]["mom_i"]["unit"] == "kg/(m^2*s)"
    assert result["per_field"]["p_i"]["unit"] == "Pa"
    assert result["per_field"]["B"]["unit"] == "T"
    assert result["per_field"]["n_i"]["relative_error"] == pytest.approx(0.25)
    assert result["per_field"]["mom_i"]["relative_error"] == pytest.approx(0.25)
    assert result["per_field"]["p_i"]["relative_error"] == pytest.approx(0.003)
    assert result["per_field"]["B"]["relative_error"] == pytest.approx(0.005)
    assert result["per_field"]["n_i"]["limited_by_amplitude_floor"] is True
    assert result["per_field"]["p_i"]["limited_by_amplitude_floor"] is True
    assert set(result["aggregates"]) == {"P", "Z", "full_PZ"}
    assert result["aggregates"]["full_PZ"]["field_count"] == 6
    assert "dimensionless" in result["aggregates"]["full_PZ"]["definition"]


def test_normalized_field_errors_are_unit_rescaling_invariant():
    tool = module()
    actual, reference, baseline, scales = _field_error_inputs()
    original = tool.normalized_field_errors(
        actual,
        reference,
        baseline,
        physical_scales=scales,
    )
    factor = 1.0e6
    scaled = tool.normalized_field_errors(
        {
            **actual,
            "mom_i": factor * actual["mom_i"],
        },
        {
            **reference,
            "mom_i": factor * reference["mom_i"],
        },
        {
            **baseline,
            "mom_i": factor * baseline["mom_i"],
        },
        physical_scales={
            **scales,
            "mom_i": factor * scales["mom_i"],
        },
    )

    for name in original["per_field"]:
        assert scaled["per_field"][name]["relative_error"] == pytest.approx(
            original["per_field"][name]["relative_error"],
            rel=1e-9,
            abs=1e-15,
        )
    for name in original["aggregates"]:
        assert (
            scaled["aggregates"][name]["dimensionless_rms_error"]
            == pytest.approx(
                original["aggregates"][name]["dimensionless_rms_error"],
                rel=1e-9,
                abs=1e-15,
            )
        )


def test_recanonicalization_detects_corrupt_derived_electron_fields():
    tool = module()
    from fomhd.massless import canonicalize_state
    from test_massless_support import context as closed_context, equilibrium_state

    context = closed_context(K=0, hard_diagnostics=True)
    state = equilibrium_state(context)
    canonical = canonicalize_state(state, context)
    density_error = 0.25 * context.p.ref.ne
    momentum_error = (
        0.25
        * float(context.p.species[1].mass)
        * context.p.ref.ne
        * context.p.ref.velocity
    )
    state.fluid[1].n[:] = state.fluid[1].n + density_error
    state.fluid[1].mom[:] = state.fluid[1].mom + momentum_error
    before = tool.compatibility_fields(state, context)
    after = tool.compatibility_fields(canonical, context)
    correction = tool.normalized_field_errors(
        after,
        before,
        before,
        physical_scales={
            "n_e": context.p.ref.ne,
            "mom_e": (
                float(context.p.species[1].mass)
                * context.p.ref.ne
                * context.p.ref.velocity
            ),
        },
        field_order=("n_e", "mom_e"),
        units={
            "n_e": "1/m^3",
            "mom_e": "kg/(m^2*s)",
        },
        field_groups={"compatibility": ("n_e", "mom_e")},
        floor_semantics=(
            "derived electron compatibility fields use their SI reference "
            "density and momentum scales"
        ),
    )

    assert correction["per_field"]["n_e"]["absolute_error"] > 0.0
    assert correction["per_field"]["mom_e"]["absolute_error"] > 0.0
    assert (
        correction["aggregates"]["compatibility"]["dimensionless_rms_error"]
        > 0.0
    )


def test_closed_smooth_perturbation_is_nonzero_and_preserves_hard_flags():
    tool = module()
    from test_massless_support import context as closed_context, equilibrium_state

    context = closed_context(K=0, hard_diagnostics=True)
    before = {
        name: budget.hard
        for name, budget in context.p.budgets.items()
    }
    baseline = equilibrium_state(context)
    perturbed = tool.perturb_closed_state(
        equilibrium_state(context),
        context,
    )
    baseline_fields = tool._independent_fields(baseline, context)
    perturbed_fields = tool._independent_fields(perturbed, context)

    assert any(
        np.linalg.norm(perturbed_fields[name] - baseline_fields[name]) > 0.0
        for name in ("n_i", "mom_i", "p_i")
    )
    assert any(
        np.linalg.norm(perturbed_fields[name] - baseline_fields[name]) > 0.0
        for name in ("p_perp_e", "p_parallel_e", "B")
    )
    assert {
        name: budget.hard
        for name, budget in context.p.budgets.items()
    } == before


def test_closed_context_honors_requested_backend_without_gpu(monkeypatch):
    tool = module()
    captured = {}

    class FakeBackend:
        def __init__(self, device):
            self.device = device
            self.name = "cpu" if device == "cpu" else "cuda"

    def fake_operators(geometry, K, *, backend):
        captured["geometry"] = geometry
        captured["K"] = K
        captured["backend"] = backend
        return SimpleNamespace()

    monkeypatch.setattr(tool, "Backend", FakeBackend, raising=False)
    monkeypatch.setattr(
        tool,
        "build_rectangle",
        lambda *args: SimpleNamespace(name="closed"),
        raising=False,
    )
    monkeypatch.setattr(tool, "build_operators", fake_operators, raising=False)

    context = tool.build_closed_context("cuda:0")

    assert context.compute.device == "cuda:0"
    assert context.compute.name == "cuda"
    assert captured["backend"] is context.compute


def test_closed_context_rejects_backend_fallback(monkeypatch):
    tool = module()

    class WrongBackend:
        def __init__(self, device):
            self.device = "cpu"
            self.name = "cpu"

    monkeypatch.setattr(tool, "Backend", WrongBackend, raising=False)
    monkeypatch.setattr(
        tool,
        "build_rectangle",
        lambda *args: SimpleNamespace(name="closed"),
        raising=False,
    )
    monkeypatch.setattr(
        tool,
        "build_operators",
        lambda geometry, K, *, backend: SimpleNamespace(),
        raising=False,
    )

    with pytest.raises(Failure, match="requested cuda:0.*constructed cpu"):
        tool.build_closed_context("cuda:0")


def test_successful_closed_result_writes_artifacts(monkeypatch, tmp_path):
    tool = module()
    reference = SimpleNamespace(
        ni=1.0,
        ne=1.0,
        velocity=1.0,
        pressure=1.0,
        magnetic=1.0,
    )
    context = SimpleNamespace(
        compute=SimpleNamespace(device="cpu", name="cpu"),
        p=SimpleNamespace(
            ref=reference,
            species=(
                SimpleNamespace(mass=2.0),
                SimpleNamespace(mass=1.0),
            ),
            dt_guard="warn",
        ),
    )
    trajectory = SimpleNamespace(
        context=context,
        committed=SimpleNamespace(state=SimpleNamespace(time=1.0e-10)),
        requested_device="cpu",
    )
    execution = {
        "accepted_macro_steps": 1,
        "checks": {
            "total": 1,
            "hard_total": 1,
            "hard_failed": 0,
            "soft_failed": 0,
            "unavailable": 0,
            "failed_records": [],
        },
        "coupling_metadata": {"estimated_peak_bytes": 1024},
        "time_history_s": np.asarray([0.0, 1.0e-10]),
    }
    before = {
        name: np.asarray([1.0], dtype=np.complex128)
        for name in ("n_i", "mom_i", "p_i", "p_perp_e", "p_parallel_e", "B")
    }
    after = {
        name: np.asarray([1.0], dtype=np.complex128)
        for name in before
    }
    before_compatibility = {
        "n_e": np.asarray([1.0], dtype=np.complex128),
        "mom_e": np.asarray([0.0], dtype=np.complex128),
    }
    after_compatibility = {
        "n_e": np.asarray([1.0], dtype=np.complex128),
        "mom_e": np.asarray([0.5], dtype=np.complex128),
    }
    monkeypatch.setattr(
        tool,
        "_run_repetitions",
        lambda **kwargs: {
            "trajectory": trajectory,
            "execution": execution,
        },
    )
    monkeypatch.setattr(
        tool.massless,
        "canonicalize_state",
        lambda state, context: "canonical",
    )
    monkeypatch.setattr(
        tool,
        "_independent_fields",
        lambda state, context: after if state == "canonical" else before,
    )
    monkeypatch.setattr(
        tool,
        "compatibility_fields",
        lambda state, context: (
            after_compatibility
            if state == "canonical"
            else before_compatibility
        ),
    )
    monkeypatch.setattr(
        tool,
        "_endpoint_arrays",
        lambda state, context: {
            "synthetic_time_s": np.asarray([1.0e-10]),
        },
    )
    monkeypatch.setattr(
        tool,
        "_execution_memory",
        lambda metadata: {
            "process_rss_bytes": 1,
            "cupy_pool": {"available": False},
            "estimated_peak_bytes": metadata["estimated_peak_bytes"],
        },
    )

    report, raw = tool._execute_case(
        tmp_path,
        "closed_synthetic",
        lambda: tool.run_closed_case(
            device="cpu",
            micro_steps=2,
            macro_dt=1.0e-10,
            macro_count=1,
            repetitions=1,
            case_id="closed_synthetic",
        ),
    )

    assert raw is not None
    assert report["status"] == "measured"
    assert (tmp_path / "closed_synthetic.npz").is_file()
    assert (tmp_path / "closed_synthetic-metadata.json").is_file()
    with np.load(
        tmp_path / "closed_synthetic.npz",
        allow_pickle=False,
    ) as arrays:
        np.testing.assert_array_equal(arrays["before_mom_e"], [0.0])
        np.testing.assert_array_equal(arrays["after_mom_e"], [0.5])


def _metadata(micro_steps, *, bad_complete=False):
    return {
        "coupling": "predictor",
        "micro_steps": micro_steps,
        "complete_stage_evaluations": 4 * micro_steps + 1 + int(bad_complete),
        "full_rhs_stage_evaluations": 4 * micro_steps,
        "endpoint_full_evaluations": 1,
        "alias_evaluations": 2 * micro_steps,
        "total_model_evaluations": 6 * micro_steps + 1,
        "retained_quadrature_entries": 4 * micro_steps,
        "assemble_canonicalization_calls": 4 * micro_steps,
        "estimated_peak_bytes": 8192,
        "endpoint_tangent_correction": {
            "n_i": {"relative_max_abs": 1.0e-8},
            "mom_i": {"relative_max_abs": 2.0e-8},
            "perp_i": {"relative_max_abs": 3.0e-8},
        },
    }


def test_predictor_metadata_aggregates_actual_stage_counts():
    tool = module()
    aggregate = tool.aggregate_coupling_metadata(
        [_metadata(4), _metadata(4)],
        coupling="predictor",
        micro_steps=4,
    )

    assert aggregate["accepted_macro_steps"] == 2
    assert aggregate["complete_stage_evaluations"] == 34
    assert aggregate["full_rhs_stage_evaluations"] == 32
    assert aggregate["alias_evaluations"] == 16
    assert aggregate["endpoint_full_evaluations"] == 2
    assert aggregate["total_model_evaluations"] == 50
    assert aggregate["expected_per_macro"][
        "complete_stage_evaluations"
    ] == 17
    assert aggregate["endpoint_tangent_correction"] == (
        _metadata(4)["endpoint_tangent_correction"]
    )

    with pytest.raises(Failure, match="complete_stage_evaluations"):
        tool.aggregate_coupling_metadata(
            [_metadata(4, bad_complete=True)],
            coupling="predictor",
            micro_steps=4,
        )


@pytest.mark.parametrize(
    "records,expected",
    [
        (
            [
                SimpleNamespace(hard=True, available=True, passed=True),
                SimpleNamespace(hard=False, available=True, passed=False),
                SimpleNamespace(hard=True, available=True, passed=False),
            ],
            {
                "total": 3,
                "hard_total": 2,
                "hard_failed": 1,
                "soft_failed": 1,
                "unavailable": 0,
            },
        ),
        (
            [SimpleNamespace(hard=True, available=False, passed=False)],
            {
                "total": 1,
                "hard_total": 1,
                "hard_failed": 1,
                "soft_failed": 0,
                "unavailable": 1,
            },
        ),
    ],
)
def test_check_summary_counts_failures_without_retaining_passing_records(
    records, expected
):
    summary = module().check_summary(records)

    for name, value in expected.items():
        assert summary[name] == value
    assert len(summary["failed_records"]) == expected["hard_failed"] + expected[
        "soft_failed"
    ]
    assert all(
        not record["passed"] for record in summary["failed_records"]
    )


def test_failed_case_is_retained_as_summary_checks_and_arrays(
    tmp_path, monkeypatch
):
    tool = module()

    def fail_case(*args, **kwargs):
        failure = Failure("InjectedFailure", "synthetic failed step")
        failure.checks = [
            SimpleNamespace(
                name="injected_check",
                population="ion",
                component="",
                region="all",
                unit="1",
                value_unit="1",
                norm="linf",
                raw=2.0,
                reference=1.0,
                threshold=1.0,
                hard=True,
                available=True,
                passed=False,
                reason="injected",
                tag=None,
            )
        ]
        failure.partial_arrays = {"injected_array": np.asarray([1.0, 2.0])}
        raise failure

    monkeypatch.setattr(tool, "run_periodic_case", fail_case)
    output = tmp_path / "failed-scan"
    report = tool.run_scan(
        output,
        device="cpu",
        micro_steps=(1,),
        fine_reference=False,
        repetitions=1,
    )

    assert report["overall_status"] == "failed"
    assert report["cases"]
    assert all(case["status"] == "failed" for case in report["cases"])
    failure_files = list(output.glob("*-failure.json"))
    array_files = list(output.glob("*-failure.npz"))
    assert failure_files
    assert array_files
    payload = json.loads(failure_files[0].read_text(encoding="utf-8"))
    assert payload["failure_code"] == "InjectedFailure"
    assert payload["checks"][0]["name"] == "injected_check"
    with np.load(array_files[0], allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["injected_array"], [1.0, 2.0])


def test_provenance_is_minimal_and_hashes_runtime_fixture_and_lock(monkeypatch):
    tool = module()
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-appear")
    environment = {
        "SLURM_JOB_ID": "12345",
        "CUDA_VISIBLE_DEVICES": "0",
    }

    provenance = tool.build_provenance(
        tool.ROOT,
        argv=["tools/warp_t0_verify.py", "--device", "cpu"],
        device="cpu",
        environment=environment,
    )

    assert provenance["argv"] == [
        "tools/warp_t0_verify.py",
        "--device",
        "cpu",
    ]
    assert provenance["environment"] == environment
    assert "UNRELATED_SECRET" not in json.dumps(provenance)
    assert provenance["utc"].endswith("Z")
    assert provenance["selected_device"] == "cpu"
    assert provenance["purpose"].startswith("experimental verification")
    assert len(provenance["source_git"]["commit"]) == 40
    assert isinstance(provenance["source_git"]["dirty"], bool)
    assert provenance["library_versions"]["python"]
    hashes = provenance["file_hashes"]
    for relative in (
        "pyproject.toml",
        "uv.lock",
        "src/fomhd/time.py",
        "tests/warp/massless_verification_fixture.py",
        "tests/warp/test_massless_support.py",
        "tools/warp_t0_verify.py",
    ):
        assert len(hashes[relative]) == 64


def test_cli_cuda_refusal_precedes_output_creation(tmp_path, monkeypatch):
    tool = module()
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    output = tmp_path / "cuda-refusal"

    status = tool.main(["--device", "cuda:0", "--output", str(output)])

    assert status == 2
    assert not output.exists()
