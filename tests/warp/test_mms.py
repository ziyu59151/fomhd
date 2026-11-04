"""Production-path coupled temporal MMS and independent spatial convergence."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import fomhd
from fomhd.types import Failure, check_record
import mms_fixture as mms


def _assert_close_state(context, left, right, label):
    xp = context.compute.xp
    for species in range(2):
        for name in ("n", "mom", "perp", "parallel"):
            actual = getattr(left.fluid[species], name)
            expected = getattr(right.fluid[species], name)
            error = float(xp.max(xp.abs(actual - expected))) if actual.size else 0.0
            assert error == 0.0, f"{label}: {name}_{species} changed by {error:.9g}"
    error = float(xp.max(xp.abs(left.B - right.B)))
    assert error == 0.0, f"{label}: B changed by {error:.9g}"


def test_source_uses_exact_reference_not_numerical_state():
    """A feedback source would change when the supplied numerical state changes."""
    context = mms.make_context("cpu")
    trajectory = mms.NativeMMSTrajectory(context)
    tag = mms.stage_tag(context, "midpoint", step=0)
    source = mms.ManufacturedSource(trajectory)
    first = source(SimpleNamespace(label="first"), tag, context)
    second = source(SimpleNamespace(label="second"), tag, context)
    _assert_close_state(context, first, second, "source state dependence")
    for value in source.values(first):
        assert value > 0.0, "manufactured source did not excite every coupled field"


def test_trajectory_analytic_derivative_is_nonzero_and_tagged():
    context = mms.make_context("cpu")
    trajectory = mms.NativeMMSTrajectory(context)
    tag = mms.stage_tag(context, "midpoint", step=0)
    exact = trajectory.at(tag)
    derivative = trajectory.at(tag, derivative=True)
    assert exact.step == derivative.step == tag.step
    assert exact.generation == derivative.generation == tag.generation
    assert exact.time == derivative.time == tag.time
    xp = context.compute.xp
    assert float(xp.max(xp.abs(derivative.B))) > 0.0
    assert float(xp.max(xp.abs(derivative.fluid[0].perp))) > 0.0
    assert float(xp.max(xp.abs(derivative.fluid[1].parallel))) > 0.0


def test_midpoint_discrete_mms_is_second_order_with_coupled_response():
    result = mms.cached_case("midpoint")
    assert result["passed"], result.get("failure", "midpoint MMS failed")
    assert result["levels"] == [8, 16, 32]
    for component in (
        "n_i",
        "mom_i",
        "perp_i",
        "n_e",
        "mom_e",
        "perp_e",
        "parallel_e",
        "B",
        "combined",
    ):
        for order in result["orders"][component]:
            assert 1.7 < order < 2.3, f"{component} midpoint order={order}"
    response = result["maximum_response"]
    assert response["EL_Vpm"] > 1e-20
    assert response["ET_Vpm"] > 1e-20
    assert response["physical_B_rate_Tps"] > 1e-20
    assert all(level["all_hard_checks_passed"] for level in result["cases"])
    assert all(level["accepted_intervals"] == level["steps"] for level in result["cases"])


def test_midpoint_source_is_accumulated_once_per_interval():
    result = mms.cached_case("midpoint")
    accounting = result["source_accounting"]
    assert accounting["calls_per_step"] == 2
    assert accounting["interval_quadrature_stages"] == ["midpoint"]
    assert accounting["quadrature_weight_s"] == result["cases"][0]["dt_s"]
    assert accounting["nonzero_rates"]
    assert accounting["nonzero_integrals"]
    assert not accounting["production_cumulative_contains_source"]
    for interval in accounting["intervals"]:
        assert len(interval["calls"]) == 2
        assert [call["stage"] for call in interval["calls"]] == [
            "base",
            "midpoint",
        ]


def test_rk4_discrete_mms_is_fourth_order_beyond_roundoff():
    result = mms.cached_case("rk4")
    assert result["passed"], result.get("failure", "RK4 MMS failed")
    assert result["levels"] == [2, 4, 8]
    assert all(
        error > 1e-11
        for case in result["cases"]
        for error in case["errors"].values()
    )
    for component in (
        "n_i",
        "mom_i",
        "perp_i",
        "n_e",
        "mom_e",
        "perp_e",
        "parallel_e",
        "B",
        "combined",
    ):
        for order in result["orders"][component]:
            assert 3.5 < order < 4.5, f"{component} RK4 order={order}"
    assert all(level["all_hard_checks_passed"] for level in result["cases"])


def test_omitted_ion_pressure_source_fails(tmp_path):
    raw_root = tmp_path / "raw-negative"
    negative = mms.omitted_ion_pressure_control("cpu", raw_root=raw_root)
    # The physical interval identity remains self-consistent, but the omitted
    # manufactured term must be visible as a first-order-in-duration state miss.
    assert negative["physical_gates_passed"]
    assert negative["detected"]
    assert negative["errors"]["perp_i"] > negative["minimum_detected_error"]
    assert negative["errors"]["combined"] > negative["minimum_detected_error"]
    reports = negative["time_step_reports"]
    assert len(reports) == 8
    assert all(report["origin"] == "time.advance.base" for report in reports)
    assert all(report["dt_guard"] == "warn" for report in reports)
    assert all(isinstance(report["accepted"], bool) for report in reports)
    assert all("estimate" in report and "reasons" in report for report in reports)
    checks_path = Path(negative["raw_evidence"]["checks_path"])
    ledger_path = Path(negative["raw_evidence"]["ledger_path"])
    assert checks_path.is_absolute() and checks_path.is_file()
    assert ledger_path.is_absolute() and ledger_path.is_file()


def test_raw_sidecar_aggregate_counts_negative_control_once():
    aggregate = mms.aggregate_raw_checks(
        {
            "midpoint": mms.cached_case("midpoint"),
            "rk4": mms.cached_case("rk4"),
        },
        {
            "raw_evidence": {
                "check_count": 7592,
                "debug_gate": {"hard_integrity_passed": True},
            }
        },
    )
    assert aggregate["normal_sidecar_count"] == 6
    assert aggregate["negative_sidecar_count"] == 1
    assert aggregate["sidecar_count"] == 7
    assert aggregate["normal_records"] == 73820
    assert aggregate["negative_records"] == 7592
    assert aggregate["total_records"] == 81412


def test_smooth_spatial_curl_divergence_and_adjoint_converge_interior():
    result = mms.spatial_verification(levels=(8, 16))
    assert result["passed"], result.get("failure", "spatial verification failed")
    assert result["negative_control"]["passed"]
    for operator in ("gradient", "curl", "divergence", "divergence_adjoint"):
        order = result["orders"]["inner"][operator]
        assert order > 3.5, f"{operator} interior order={order}"
        for level in result["levels"]:
            assert level[operator]["inner_error"] > 0.0
            assert np.isfinite(level[operator]["wall_error"])
            assert np.isfinite(level[operator]["total_error"])


def test_cuda_is_refused_without_slurm_allocation():
    with pytest.raises(Failure, match="Slurm|SLURM"):
        mms.require_device("cuda:0", environ={})
    assert mms.require_device("cpu", environ={}) == "cpu"


def test_cli_output_directory_must_be_new(tmp_path):
    output = tmp_path / "new-result"
    mms.prepare_output_directory(output)
    assert output.is_dir()
    marker = output / "existing.txt"
    marker.write_text("preserve me", encoding="utf-8")
    with pytest.raises(Failure, match="must be new or empty"):
        mms.prepare_output_directory(output)
    assert marker.read_text(encoding="utf-8") == "preserve me"


def test_harness_imports_current_main_solver():
    imported = Path(fomhd.__file__).resolve()
    assert imported.name == "__init__.py"
    assert imported.parent.name == "fomhd"
    assert "scratch/proposal" not in imported.parts


def test_production_guard_reports_are_retained_per_step():
    result = mms.cached_case("midpoint")
    guard = result["diagnostics"]["time_guard"]
    assert guard["runtime_integrated"] is True
    assert guard["cold_pre_estimate"]["runtime_integrated"] is False
    for case in result["cases"]:
        reports = case["time_step_reports"]
        assert len(reports) == case["steps"]
        assert all(report["origin"] == "time.advance.base" for report in reports)
        assert all(report["dt_guard"] == "warn" for report in reports)


def test_raw_checks_and_ledgers_are_preserved_with_hashes(tmp_path):
    context = mms.make_context("cpu")
    raw_root = tmp_path / "raw"
    result = mms.run_level(
        context,
        integrator="midpoint",
        steps=2,
        raw_root=raw_root,
    )
    evidence = result["raw_evidence"]
    assert evidence["check_count"] == result["check_count"]
    assert evidence["debug_gate"]["hard_integrity_passed"]
    assert evidence["debug_gate"]["soft_failed"] > 0
    assert evidence["debug_gate"]["unavailable"] > 0

    checks = raw_root / evidence["checks_path"]
    lines = checks.read_text(encoding="utf-8").splitlines()
    assert len(lines) == result["check_count"]
    records = [json.loads(line) for line in lines]
    required = {
        "name",
        "population",
        "region",
        "component",
        "producer",
        "norm",
        "unit",
        "stage",
        "time_s",
        "raw",
        "reference",
        "threshold",
        "available",
        "hard",
        "passed",
    }
    assert required <= set(records[0])
    assert all({"step", "generation", "trial"} <= set(record) for record in records)
    assert mms.sha256_file(checks) == evidence["checks_sha256"]

    ledger = raw_root / evidence["ledger_path"]
    ledger_record = json.loads(ledger.read_text(encoding="utf-8"))
    assert ledger_record["accepted_intervals"] == 2
    assert len(ledger_record["physical_integrals"]) == 2
    assert "source_rate_times_seconds" in ledger_record
    assert mms.sha256_file(ledger) == evidence["ledger_sha256"]


def test_raw_check_serialization_detects_failed_control():
    control = check_record(
        name="deliberate_soft_failure",
        raw=2.0,
        reference=1.0,
        threshold=1.0,
        available=True,
        hard=False,
        passed=False,
    )
    record = mms.serialize_check_record(control)
    audit = mms.audit_check_records([record])
    assert record["raw"] == 2.0
    assert record["threshold"] == 1.0
    assert record["passed"] is False
    assert audit["failed"] == 1
    assert audit["hard_integrity_passed"]
    assert not audit["all_records_passed"]


def test_failure_partial_arrays_preserve_nan_and_inf(tmp_path):
    error = Failure("SyntheticFailure", "raw arrays must survive")
    error.partial_arrays = {
        "bad": np.asarray([np.nan, np.inf, -np.inf], dtype=np.float64),
        "good": np.asarray([1.0 + 2.0j], dtype=np.complex128),
    }
    evidence = mms.write_failure_arrays(tmp_path, error, "synthetic_failure")
    path = tmp_path / evidence["path"]
    assert evidence["contains_nonfinite"] is True
    assert mms.sha256_file(path) == evidence["sha256"]
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"bad", "good"}
        assert np.isnan(archive["bad"][0])
        assert np.isposinf(archive["bad"][1])
        assert np.isneginf(archive["bad"][2])
        assert archive["good"][0] == 1.0 + 2.0j


def test_source_magnetic_divergence_negative_control():
    context = mms.make_context("cpu")
    trajectory = mms.NativeMMSTrajectory(context)
    tag = mms.stage_tag(context, "midpoint")
    source = mms.ManufacturedSource(trajectory)
    payload = source(SimpleNamespace(), tag, context)
    payload.B[context.p.K * context.op.MB.size] += 1.0e-5
    with pytest.raises(Failure, match="magnetic source must preserve divergence"):
        mms.diagnostics.check_source(payload, context)


def test_source_snapshot_and_identity_cover_runtime_and_harness(tmp_path):
    runtime_root = Path(fomhd.__file__).resolve().parents[1]
    harness_root = Path(mms.__file__).resolve().parents[2]
    snapshot = tmp_path / "source_snapshot"
    manifest = mms.archive_source_snapshot(
        snapshot,
        runtime_root=runtime_root,
        harness_root=harness_root,
    )
    destinations = {entry["path"]: entry for entry in manifest["files"]}
    runtime_modules = {
        f"runtime/fomhd/{path.name}"
        for path in (runtime_root / "fomhd").glob("*.py")
    }
    assert runtime_modules <= set(destinations)
    assert {
        "repository/pyproject.toml",
        "repository/uv.lock",
        "harness/tests/warp/mms_fixture.py",
        "harness/tests/warp/test_mms.py",
        "harness/tests/warp/foundation_fixture.py",
        "harness/tools/warp_mms.py",
    } <= set(destinations)
    for relative, entry in destinations.items():
        assert mms.sha256_file(snapshot / relative) == entry["sha256"]

    identity = mms.build_identity(
        harness_root,
        runtime_root=runtime_root,
    )
    assert identity["runtime_root"] == str(runtime_root.resolve())
    assert identity["harness_root"] == str(harness_root.resolve())
    assert identity["runtime_hashes"]["fomhd/time.py"] == mms.sha256_file(
        runtime_root / "fomhd/time.py"
    )
    assert identity["repository_hashes"]["pyproject.toml"] == mms.sha256_file(
        runtime_root / "pyproject.toml"
    )


def test_runtime_origin_uses_imported_package_and_rejects_mismatch(tmp_path):
    actual_root = Path(fomhd.__file__).resolve().parents[1]
    origin = mms.resolve_runtime_origin()
    assert Path(origin["root"]) == actual_root
    assert Path(origin["package_root"]) == actual_root / "fomhd"
    assert origin["loaded_modules"]
    assert all(
        Path(path).is_relative_to(actual_root / "fomhd")
        for path in origin["loaded_modules"].values()
    )
    with pytest.raises(Failure, match="RuntimeMismatch"):
        mms.resolve_runtime_origin(tmp_path / "different-runtime")
    with pytest.raises(Failure, match="RuntimeMismatch"):
        mms.build_identity(runtime_root=tmp_path / "different-runtime")
    rejected_snapshot = tmp_path / "rejected-snapshot"
    with pytest.raises(Failure, match="RuntimeMismatch"):
        mms.archive_source_snapshot(
            rejected_snapshot,
            runtime_root=tmp_path / "different-runtime",
        )
    assert not rejected_snapshot.exists()


def test_frozen_checkout_provenance_tracks_executed_package(tmp_path):
    source_root = Path(fomhd.__file__).resolve().parents[1]
    frozen_root = tmp_path / "frozen-runtime"
    (frozen_root / "fomhd").mkdir(parents=True)
    for module in (source_root / "fomhd").glob("*.py"):
        shutil.copyfile(module, frozen_root / "fomhd" / module.name)
    shutil.copyfile(source_root / "pyproject.toml", frozen_root / "pyproject.toml")
    shutil.copyfile(source_root / "uv.lock", frozen_root / "uv.lock")
    sentinel = "# frozen-checkout-provenance-sentinel\n"
    with (frozen_root / "fomhd" / "__init__.py").open("a", encoding="utf-8") as stream:
        stream.write(sentinel)

    harness_tests = Path(mms.__file__).resolve().parent
    output = tmp_path / "frozen-output"
    script = """
import json
import sys
from pathlib import Path
import fomhd
import mms_fixture
identity = mms_fixture.build_identity()
snapshot = mms_fixture.archive_source_snapshot(Path(sys.argv[1]))
print(json.dumps({
    "module": fomhd.__file__,
    "identity": identity,
    "snapshot": snapshot,
}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(frozen_root), str(harness_tests))
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(output)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    evidence = json.loads(completed.stdout)
    imported = Path(evidence["module"]).resolve()
    assert imported.is_relative_to(frozen_root / "fomhd")
    assert evidence["identity"]["runtime_root"] == str(frozen_root)
    assert evidence["snapshot"]["runtime_root"] == str(frozen_root)

    frozen_init = frozen_root / "fomhd" / "__init__.py"
    frozen_hash = mms.sha256_file(frozen_init)
    source_hash = mms.sha256_file(source_root / "fomhd" / "__init__.py")
    assert frozen_hash != source_hash
    assert (
        evidence["identity"]["runtime_hashes"]["fomhd/__init__.py"]
        == frozen_hash
    )
    snapshot_init = output / "runtime" / "fomhd" / "__init__.py"
    assert snapshot_init.read_text(encoding="utf-8").endswith(sentinel)
    assert mms.sha256_file(snapshot_init) == frozen_hash
    assert all(
        path.startswith(str(frozen_root / "fomhd"))
        for path in evidence["identity"]["imported_modules"].values()
    )
    runtime_entries = [
        entry
        for entry in evidence["snapshot"]["files"]
        if entry["role"] == "runtime_module"
    ]
    assert runtime_entries
    assert all(
        entry["source_path"].startswith(str(frozen_root / "fomhd"))
        for entry in runtime_entries
    )
    for entry in runtime_entries:
        relative = Path(entry["path"]).relative_to("runtime")
        identity_hash = evidence["identity"]["runtime_hashes"][str(relative)]
        assert entry["sha256"] == identity_hash


def test_cli_failure_record_preserves_partial_arrays(tmp_path, monkeypatch):
    tool_path = Path(mms.__file__).resolve().parents[2] / "tools" / "warp_mms.py"
    spec = importlib.util.spec_from_file_location("warp_mms_cli_test", tool_path)
    cli = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cli)

    def fail_with_arrays(*args, **kwargs):
        error = Failure("SyntheticCLI", "preserve raw CLI failure")
        error.partial_arrays = {
            "midpoint_B": np.asarray([np.nan, np.inf], dtype=np.float64)
        }
        error.checks = [
            check_record(
                name="failure_check",
                raw=2.0,
                reference=1.0,
                threshold=1.0,
                hard=True,
                passed=False,
            )
        ]
        raise error

    monkeypatch.setattr(cli, "run_verification", fail_with_arrays)
    output = tmp_path / "cli-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "warp_mms.py",
            "--device",
            "cpu",
            "--integrator",
            "midpoint",
            "--no-spatial",
            "--output",
            str(output),
        ],
    )
    assert cli.main() == 1
    record = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert record["status"] == "FAIL"
    assert record["failure"]["code"] == "SyntheticCLI"
    arrays = record["failure"]["failure_arrays"]
    assert arrays["contains_nonfinite"] is True
    with np.load(
        output / "failure" / arrays["path"],
        allow_pickle=False,
    ) as archive:
        assert np.isnan(archive["midpoint_B"][0])
        assert np.isposinf(archive["midpoint_B"][1])
    raw = record["failure"]["raw_checks"]
    failure_records = [
        json.loads(line)
        for line in (output / "failure" / raw["path"]).read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert failure_records[0]["name"] == "failure_check"
    assert failure_records[0]["hard"] is True
    assert failure_records[0]["passed"] is False
    assert mms.sha256_file(output / "failure" / raw["path"]) == raw["sha256"]
