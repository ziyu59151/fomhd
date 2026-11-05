"""Catch incorrect finite-mode polarization, field adapters and time response."""

import importlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


def module():
    return importlib.import_module("tools.warp_longitudinal")


def test_longitudinal_mode_satisfies_independent_linearized_equations():
    tool = module()
    context, mode = tool.build_context("cpu")
    p = context.p
    mi, me = [s.mass for s in p.species]
    q = p.species[0].charge
    n, pressure, k, speed = mode["density"], mode["pressure"], mode["k"], mode["speed"]
    v = mode["vector"]
    electric = -1j*q*n*(v[0]-v[3])/(p.epsilon0*k)
    rhs = np.array([
        -1j*k*speed*v[1],
        q*electric/(mi*speed)-1j*k*pressure*v[2]/(mi*n*speed),
        -1j*k*(5/3)*speed*v[1],
        -1j*k*speed*v[4],
        -q*electric/(me*speed)-1j*k*pressure*v[6]/(me*n*speed),
        -1j*k*speed*v[4],
        -3j*k*speed*v[4],
    ])
    expected = 1j*mode["omega"]*v
    assert np.linalg.norm(rhs-expected)/np.linalg.norm(expected) < 1e-12
    cold = math.sqrt(n*q*q/p.epsilon0*(1/mi+1/me))
    assert cold < mode["omega"] < 1.00001*cold
    assert mode["adapter_max_defect"] < 1e-12


@pytest.mark.parametrize("points", [9, 17, 65])
def test_longitudinal_supported_point_counts_share_the_analytic_mode(points):
    tool = module()
    reference = tool.build_context("cpu", points=9)[1]
    context, mode = tool.build_context("cpu", points=points)
    assert context.op.Nc == points
    assert mode["omega"] == reference["omega"]
    np.testing.assert_array_equal(mode["vector"], reference["vector"])
    assert mode["adapter_max_defect"] < 1e-12


@pytest.mark.parametrize("points", [7, 8, 66, 9.0, True])
def test_longitudinal_invalid_point_counts_are_rejected(points):
    with pytest.raises(ValueError):
        module().build_context("cpu", points=points)


@pytest.mark.parametrize("integrator", ["midpoint", "rk4"])
def test_real_longitudinal_transaction_matches_coarse_amplification(integrator):
    result = module().run_case(integrator, 4)
    summary = result["summary"]
    assert summary["accepted_steps"] == 4
    assert summary["passed"]
    assert summary["initial_hard_records"] > 0
    assert summary["hard_records"] > summary["initial_hard_records"]
    assert summary["amplitude_ratio"] > 10
    assert summary["polynomial_relative_error"] < 1e-5
    assert summary["rhs_mode_relative_error"] < 1e-5
    assert result["history"].shape == (5, 7)
    assert summary["dt_warnings"] > 0
    assert summary["points"] == 9
    assert summary["cycles"] == 2
    assert summary["dt_guard"] == "warn"
    assert summary["mode_fit"]["available"] is False
    assert "fewer than 4" in summary["mode_fit"]["reason"]


def test_longitudinal_explicit_point_count_reaches_real_history():
    result = module().run_case("rk4", 4, points=17)
    assert result["history"].shape == (5, 7)
    assert result["summary"]["points"] == 17
    assert result["summary"]["adapter_max_defect"] < 1e-12


@pytest.mark.parametrize("integrator,ratio", [("midpoint", .3), ("rk4", .09)])
def test_longitudinal_time_refinement_reduces_continuum_error(integrator, ratio):
    coarse = module().run_case(integrator, 32)["summary"]
    fine = module().run_case(integrator, 64)["summary"]
    assert coarse["passed"] and fine["passed"]
    assert coarse["target_time_s"] == fine["target_time_s"]
    assert fine["continuum_relative_error"] < ratio*coarse["continuum_relative_error"]


@pytest.mark.parametrize("kwargs", [{"integrator":"implicit", "steps":4},
                                    {"integrator":"midpoint", "steps":0},
                                    {"integrator":"midpoint", "steps":4, "amplitude":0.},
                                    {"integrator":"rk4", "steps":4, "cycles":17},
                                    {"integrator":"rk4", "steps":4, "dt_guard":"ignore"}])
def test_longitudinal_invalid_requests_are_rejected(kwargs):
    with pytest.raises(ValueError):
        module().run_case(**kwargs)


def test_longitudinal_output_refuses_existing_directory(tmp_path):
    marker = tmp_path/"preserve.txt"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        module().run_scan(tmp_path, device="cpu", steps=[4])
    assert marker.read_text() == "original"


def test_longitudinal_empty_scan_does_not_create_output(tmp_path):
    output = tmp_path/"invalid-scan"
    with pytest.raises(ValueError):
        module().run_scan(output, steps=[])
    assert not output.exists()


def test_longitudinal_scan_preserves_numeric_history_and_initial_checks(tmp_path):
    import gzip

    output = tmp_path/"scan"
    report = module().run_scan(output, steps=[4])
    assert report["passed"] and len(report["cases"]) == 4
    assert report["configuration"] == {
        "points": 9,
        "cycles": 2.0,
        "steps": [4],
        "integrators": ["midpoint", "rk4"],
        "dt_guard": "warn",
    }
    assert all(row["passed"] for row in report["amplitude_sensitivity"])
    for case in report["cases"]:
        name = case["case_id"]
        with np.load(output/(name+".npz"), allow_pickle=False) as data:
            assert data["mode"].shape == (5, 7)
            assert data["time_s"][-1] == case["accepted_time_s"]
        with gzip.open(output/(name+"-checks.json.gz"), "rt") as stream:
            checks = json.load(stream)
        assert any(check["stage"] == "initial" for check in checks)
        assert all(check["available"] and check["passed"] for check in checks if check["hard"])
    provenance = json.loads((output/"provenance.json").read_text())
    assert provenance["configuration"] == report["configuration"]
    assert (output/"source/src/fomhd/time.py").is_file()
    assert provenance["points"] == 9
    assert provenance["cycles"] == 2.0
    assert provenance["integrators"] == ["midpoint", "rk4"]
    assert provenance["dt_guard"] == "warn"
    assert "src/fomhd/time.py" in provenance["source_hashes"]
    assert report["points"] == 9
    assert report["cycles"] == 2.0
    assert report["integrators"] == ["midpoint", "rk4"]
    assert report["dt_guard"] == "warn"
    manifest = json.loads((output/"source/SOURCE_MANIFEST.json").read_text())
    assert manifest["runtime_relative"] == "src/fomhd"
    assert manifest["configuration"] == report["configuration"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"points": 8},
        {"points": 67},
        {"cycles": 0},
        {"cycles": 17},
        {"steps": [4, 16385]},
        {"steps": []},
        {"steps": [4, 4]},
        {"integrators": []},
        {"integrators": ["implicit"]},
        {"integrators": ["rk4", "rk4"]},
        {"dt_guard": "ignore"},
    ],
)
def test_longitudinal_scan_limits_fail_before_output(tmp_path, kwargs):
    output = tmp_path/"must-not-exist"
    with pytest.raises(ValueError):
        module().run_scan(output, **kwargs)
    assert not output.exists()


def test_longitudinal_control_limit_normalization_accepts_declared_maxima():
    control = module().normalize_controls(
        points=65,
        cycles=16,
        steps=[16384],
        integrators=["rk4"],
        dt_guard="error",
    )
    assert control == {
        "points": 65,
        "cycles": 16.0,
        "steps": [16384],
        "integrators": ["rk4"],
        "dt_guard": "error",
    }


@pytest.mark.parametrize("integrator", ["midpoint", "rk4"])
def test_longitudinal_rejected_attempt_retains_structured_dt_warning(monkeypatch, integrator):
    tool = module()
    original = tool.build_context

    def inconsistent_poisson(device, *, points=9):
        context, mode = original(device, points=points)
        context.op.Lphi = [np.eye(context.op.Nc, dtype=complex)]
        return context, mode

    monkeypatch.setattr(tool, "build_context", inconsistent_poisson)
    result = tool.run_case(integrator, 4)
    assert result["failure"] is not None
    assert result["summary"]["accepted_steps"] == 0
    assert result["time_step_reports"] == [result["failure"].time_step_report]
    assert result["summary"]["dt_warnings"] == 1
    assert any(check.name == "time_step_estimate" and not check.passed
               for check in result["diagnostics"])


def test_longitudinal_error_guard_rejects_where_explicit_warn_continues():
    tool = module()
    warned = tool.run_case("rk4", 4, cycles=2, dt_guard="warn")
    rejected = tool.run_case("rk4", 4, cycles=2, dt_guard="error")
    assert warned["summary"]["passed"]
    assert warned["summary"]["accepted_steps"] == 4
    assert warned["summary"]["dt_warnings"] == 4
    assert rejected["summary"]["accepted_steps"] == 0
    assert rejected["summary"]["dt_guard"] == "error"
    assert not rejected["summary"]["passed"]
    assert rejected["failure"].code == "TimeStepGuard"
    assert rejected["time_step_reports"] == [rejected["failure"].time_step_report]
    assert rejected["history"].shape == (1, 7)
    assert rejected["summary"]["mode_fit"]["available"] is False


def test_cli_defaults_to_rk4_error_guard_and_eight_cycles(monkeypatch, tmp_path):
    tool = module()
    captured = {}

    def fake_run_scan(output, **kwargs):
        captured.update(output=output, **kwargs)
        return {"passed": True}

    monkeypatch.setattr(tool, "run_scan", fake_run_scan)
    output = tmp_path/"cli-output"
    monkeypatch.setattr(
        sys,
        "argv",
        ["warp_longitudinal.py", "--output", str(output)],
    )
    assert tool.main() == 0
    assert captured == {
        "output": output,
        "device": "cpu",
        "points": 9,
        "cycles": 8.0,
        "steps": [1024, 2048],
        "integrators": ("rk4",),
        "dt_guard": "error",
    }


def test_complex_mode_rejects_gapped_cadence():
    tool = module()
    times = np.r_[np.linspace(0., .7, 145), np.linspace(1.3, 2., 145)]
    result = tool.fit_complex_mode(times, np.exp(2j*np.pi*4*times))
    assert not result["available"]
    assert "cadence" in result["reason"]


def test_cli_serializes_explicit_selection(monkeypatch, tmp_path):
    tool = module()
    captured = {}

    def fake_run_scan(output, **kwargs):
        captured.update(output=output, **kwargs)
        return {"passed": True}

    monkeypatch.setattr(tool, "run_scan", fake_run_scan)
    output = tmp_path/"selected"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "warp_longitudinal.py",
            "--output", str(output),
            "--points", "17",
            "--cycles", "6",
            "--steps", "128,512",
            "--integrators", "midpoint,rk4",
            "--dt-guard", "warn",
        ],
    )
    assert tool.main() == 0
    assert captured["points"] == 17
    assert captured["cycles"] == 6.0
    assert captured["steps"] == [128, 512]
    assert captured["integrators"] == ("midpoint", "rk4")
    assert captured["dt_guard"] == "warn"


def _synthetic_mode(
    cycles=8.0,
    samples_per_cycle=32,
    *,
    frequency_sign=1,
    growth_rate=-0.2,
):
    intervals = int(round(cycles*samples_per_cycle))
    period = 1/3
    dt = period/samples_per_cycle
    times = np.arange(intervals+1, dtype=np.float64)*dt
    omega = frequency_sign*2*math.pi/period
    coefficients = 2.5*np.exp(growth_rate*times + 1j*(omega*times+0.37))
    return times, coefficients, omega, growth_rate


@pytest.mark.parametrize("frequency_sign,growth_rate", [(-1, -0.2), (1, 0.15)])
def test_synthetic_complex_mode_fit_recovers_signed_frequency_and_rate(
    frequency_sign,
    growth_rate,
):
    times, coefficients, omega, growth = _synthetic_mode(
        frequency_sign=frequency_sign,
        growth_rate=growth_rate,
    )
    fit = module().fit_complex_mode(times, coefficients)
    assert fit["available"]
    assert fit["reason"] == ""
    assert fit["coefficient"] == {"name": "normalized_electron_density", "unit": "1"}
    assert fit["complete_measured_cycles"] == pytest.approx(8)
    assert fit["samples_per_cycle"] == pytest.approx(32)
    measured = fit["measured"]
    assert measured["angular_frequency_rad_s"] == pytest.approx(omega, rel=1e-13)
    assert measured["frequency_hz"] == pytest.approx(omega/(2*math.pi), rel=1e-13)
    assert measured["growth_rate_per_s"] == pytest.approx(growth, abs=1e-13)
    assert measured["phase_residual_rms_rad"] < 1e-12
    assert measured["amplitude_log_residual_rms"] < 1e-12
    assert measured["phase_coherence"] > 1-1e-12
    assert measured["amplitude_coherence"] > 1-1e-12
    assert measured["complex_relative_rms_residual"] < 1e-12
    assert fit["units"]["angular_frequency_rad_s"] == "rad/s"
    assert fit["units"]["growth_rate_per_s"] == "1/s"


@pytest.mark.parametrize(
    "times,coefficients,reason",
    [
        (*_synthetic_mode(cycles=3.5)[:2], "fewer than 4"),
        (*_synthetic_mode(cycles=4, samples_per_cycle=31)[:2], "fewer than 32"),
    ],
)
def test_synthetic_complex_mode_fit_availability_thresholds(times, coefficients, reason):
    fit = module().fit_complex_mode(times, coefficients)
    assert fit["available"] is False
    assert reason in fit["reason"]


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda times, values: values.__setitem__(3, 0j), "zero"),
        (lambda times, values: values.__setitem__(3, complex(np.nan, 0)), "finite"),
        (lambda times, values: times.__setitem__(3, times[2]), "increasing"),
    ],
)
def test_synthetic_complex_mode_fit_rejects_invalid_samples(mutate, reason):
    times, coefficients, _, _ = _synthetic_mode()
    mutate(times, coefficients)
    fit = module().fit_complex_mode(times, coefficients)
    assert fit["available"] is False
    assert reason in fit["reason"]


def test_real_more_than_four_cycle_history_fit_is_independent_of_theory_fields():
    tool = module()
    result = tool.run_case(
        "rk4",
        160,
        cycles=4.25,
        dt_guard="error",
    )
    summary = result["summary"]
    fit = summary["mode_fit"]
    assert summary["passed"]
    assert fit["available"]
    assert fit["complete_measured_cycles"] >= 4
    assert fit["complete_measured_cycles"] == pytest.approx(4.25, rel=1e-5)
    assert fit["samples_per_cycle"] >= 32
    assert fit["measured"]["angular_frequency_rad_s"] > 0
    assert fit["analytic"]["angular_frequency_rad_s"] == summary["omega_rad_s"]
    assert fit["analytic"]["growth_rate_per_s"] == 0
    assert fit["rk4_predicted"]["integrator"] == "rk4"
    assert fit["selected_integrator_predicted"]["integrator"] == "rk4"
    assert (
        fit["measured"]["angular_frequency_rad_s"]
        != fit["rk4_predicted"]["angular_frequency_rad_s"]
    )
    assert summary["polynomial_tolerance"] == 1e-5
    assert summary["polynomial_relative_error"] < 1e-5
    measured_only = tool.fit_complex_mode(
        result["times"],
        result["history"],
    )
    assert measured_only["measured"] == fit["measured"]


def _fake_runtime_repo(root, layout, *, metadata=True, decoy=False):
    package = root/layout
    package.mkdir(parents=True)
    (package/"__init__.py").write_text("", encoding="utf-8")
    (package/"time.py").write_text("VALUE = 1\n", encoding="utf-8")
    if metadata:
        (root/"pyproject.toml").write_text(
            '[project]\nname = "fomhd"\nversion = "0"\n',
            encoding="utf-8",
        )
        (root/"uv.lock").write_text("version = 1\n", encoding="utf-8")
    if decoy:
        legacy = root/"fomhd"
        legacy.mkdir()
        (legacy/"__init__.py").write_text("LEGACY = True\n", encoding="utf-8")
    return package/"__init__.py"


def test_runtime_origin_resolves_legacy_and_src_package_layouts(tmp_path):
    tool = module()
    legacy = _fake_runtime_repo(tmp_path/"legacy-repo", "fomhd")
    legacy_origin = tool.resolve_runtime_origin(
        legacy,
        loaded_modules={},
    )
    assert Path(legacy_origin["repository_root"]) == tmp_path/"legacy-repo"
    assert Path(legacy_origin["runtime_relative"]) == Path("fomhd")

    src = _fake_runtime_repo(
        tmp_path/"src-repo",
        "src/fomhd",
        decoy=True,
    )
    src_origin = tool.resolve_runtime_origin(src, loaded_modules={})
    assert Path(src_origin["repository_root"]) == tmp_path/"src-repo"
    assert Path(src_origin["runtime_relative"]) == Path("src/fomhd")
    assert all(
        Path(path).is_relative_to(tmp_path/"src-repo/src/fomhd")
        for path in src_origin["source_files"]
    )


def test_runtime_origin_requires_matching_metadata_and_package_modules(tmp_path):
    tool = module()
    missing = _fake_runtime_repo(
        tmp_path/"missing-meta",
        "src/fomhd",
        metadata=False,
    )
    with pytest.raises(ValueError, match="metadata"):
        tool.resolve_runtime_origin(missing, loaded_modules={})

    package = _fake_runtime_repo(tmp_path/"outside", "src/fomhd")
    with pytest.raises(ValueError, match="outside"):
        tool.resolve_runtime_origin(
            package,
            loaded_modules={"fomhd.outside": tmp_path/"elsewhere.py"},
        )


def test_source_snapshot_archives_the_resolved_src_runtime_only(tmp_path):
    tool = module()
    runtime = _fake_runtime_repo(
        tmp_path/"runtime-repo",
        "src/fomhd",
        decoy=True,
    )
    harness = tmp_path/"harness"
    harness.mkdir()
    tool_file = harness/"warp_longitudinal.py"
    tool_file.write_text("# harness\n", encoding="utf-8")
    output = tmp_path/"snapshot"
    configuration = {
        "points": 17,
        "cycles": 6.0,
        "steps": [128],
        "integrators": ["rk4"],
        "dt_guard": "error",
    }
    manifest = tool.archive_source_snapshot(
        output,
        package_file=runtime,
        loaded_modules={},
        harness_root=harness,
        harness_files=[tool_file],
        configuration=configuration,
    )
    assert (output/"src/fomhd/__init__.py").is_file()
    assert (output/"src/fomhd/time.py").is_file()
    assert not (output/"fomhd").exists()
    assert (output/"warp_longitudinal.py").is_file()
    assert (output/"pyproject.toml").is_file()
    assert (output/"uv.lock").is_file()
    assert manifest["runtime_relative"] == "src/fomhd"
    assert manifest["configuration"] == configuration
    assert "src/fomhd/time.py" in manifest["source_hashes"]
