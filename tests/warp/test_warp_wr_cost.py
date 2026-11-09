"""CPU contract and real-physics smoke tests for the matched-cost harness."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

import fomhd
from fomhd.types import Failure


SOURCE_ROOT = Path(fomhd.__file__).resolve().parents[2]
os.environ["FOMHD_SOURCE_ROOT"] = str(SOURCE_ROOT)
TOOL = Path(__file__).resolve().parents[2] / "tools" / "warp_wr_cost.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("warp_wr_cost", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def config(**changes):
    value = {
        "device": "cpu",
        "points": 33,
        "horizon_microsteps": 8,
        "wr_M": 8,
        "wr_ion_nodes": [3],
        "methods": ["synchronous", "waveform", "multirate_wr"],
        "wall_seconds": 120.0,
    }
    value.update(changes)
    return value


@pytest.fixture(scope="module")
def cpu_report():
    module = load_tool()
    report, failure = module.run(config())
    assert failure is None
    return module, report


def test_real_cpu_matched_horizon_contract(cpu_report):
    module, report = cpu_report
    assert report["schema"] == module.SCHEMA
    assert report["status"] == "completed"
    assert report["physical_input_consistent"] is True
    assert report["all_initial_inputs_nontrivial"] is True
    assert report["speedup_claim"] is None
    assert len(report["methods"]) == 3
    methods = {row["identity"]["method"]: row for row in report["methods"]}
    assert set(methods) == {"synchronous", "waveform", "multirate_wr"}

    horizon = report["config"]["horizon_seconds"]
    digests = {
        row["identity"]["initial_state_sha256"]
        for row in report["methods"]
    }
    assert len(digests) == 1
    for row in report["methods"]:
        assert row["identity"]["fixture"]["target_mode"] == 1
        assert row["identity"]["fixture"]["sentinel_mode"] == 16
        assert row["identity"]["fixture"]["sentinel_retained"] is True
        assert (
            row["identity"]["budgets"]["quasineutrality_linf_hard"]
            is True
        )
        assert (
            row["identity"]["budgets"]["theta_consistency_linf_observe"]
            is True
        )
        assert row["identity"]["horizon_seconds"] == pytest.approx(horizon)
        magnitudes = row["identity"]["initial_state_magnitudes"]
        assert magnitudes["0_n"] > 0.0
        assert magnitudes["0_mom"] > 0.0
        assert magnitudes["0_perp"] > 0.0
        assert magnitudes["1_perp"] > 0.0
        assert magnitudes["B"] > 0.0
        assert row["warmup"]["status"] == "accepted"
        assert row["warmup"]["run_seconds"] > 0.0
        assert len(row["repeats"]) == 3
        assert all(item["status"] == "accepted" for item in row["repeats"])
        assert row["ledger_resets_inside_physical_run"] == 0

    assert methods["synchronous"]["warmup"]["advance_calls"] == 8
    assert all(
        item["advance_calls"] == 8
        for item in methods["synchronous"]["repeats"]
    )
    for name in ("waveform", "multirate_wr"):
        assert methods[name]["warmup"]["advance_calls"] == 1
        assert all(
            item["advance_calls"] == 1
            for item in methods[name]["repeats"]
        )


def test_cpu_repeat_reset_setup_and_raw_counter_roles(cpu_report):
    _, report = cpu_report
    for row in report["methods"]:
        runs = [row["warmup"], *row["repeats"]]
        state_hashes = [item["input_state_sha256"] for item in runs]
        cumulative_hashes = [
            item["input_cumulative_sha256"] for item in runs
        ]
        assert all(
            value == row["identity"]["initial_state_sha256"]
            for value in state_hashes
        )
        assert all(
            value == row["identity"]["input_cumulative_sha256"]
            for value in cumulative_hashes
        )
        pre_run = [item["pre_run_generation_counters"] for item in runs]
        assert all(item == pre_run[0] for item in pre_run)
        for item in runs:
            assert item["reset_seconds"] >= 0.0
            assert item["setup_seconds"] >= 0.0
            assert item["run_seconds"] > 0.0
            assert item["total_seconds"] >= item["run_seconds"]
            assert item["memory_peak"]["host_rss_bytes"] > 0
            assert item["memory_peak"]["host_limit_bytes"] == 16 << 30
            assert item["memory_peak"]["device_limit_bytes"] == 8 << 30
            roles = item["counter_roles"]
            assert roles["field_callback_produces_both_G_and_C"] is True
            assert "intersection_policy" in roles
            assert "full_ion_complete_stage_calls" not in roles
            assert "full_ion_complete_rhs_values" not in roles
            assert roles["ion_transport_producer_calls"] == (
                item["spy"]["full_ion_calls"]
            )
            assert roles["stage_evaluator_calls"] == (
                item["spy"]["full_stage_calls"]
            )
            assert roles["full_rhs_finish_calls"] == (
                item["spy"]["full_rhs_calls"]
            )
            assert "coarse_A_producer_calls" in roles
        assert row["run_seconds"]["median"] > 0.0

    wr = next(
        row
        for row in report["methods"]
        if row["identity"]["method"] == "multirate_wr"
    )
    repeat = wr["repeats"][0]
    assert repeat["spy"]["coarse_node_calls"] > 0
    assert repeat["spy"]["field_evaluations"] > 0
    assert repeat["transaction_work"] is not None
    assert repeat["bundle_diagnostic_work"] is not None
    assert repeat["result_diagnostic_work"] is not None
    assert repeat["residual_history_length"] >= 1


def test_cuda_guard_and_matched_controls_validate(monkeypatch, tmp_path):
    module = load_tool()
    monkeypatch.delenv("FOMHD_TEST_GPU", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="FOMHD_TEST_GPU"):
        module.run(config(device="cuda:0"))
    monkeypatch.setenv("FOMHD_TEST_GPU", "1")
    with pytest.raises(ValueError, match="Slurm"):
        module.run(config(device="cuda:0"))
    with pytest.raises(ValueError, match="horizon"):
        module.run(config(horizon_microsteps=16))
    with pytest.raises(ValueError, match="divisor"):
        module.run(config(horizon_microsteps=8, wr_M=3))
    with pytest.raises(ValueError, match="wall_seconds"):
        module.run(config(wall_seconds=901))

    output = tmp_path / "report.json"
    output.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing existing"):
        module.run_paths(tmp_path / "not-read.json", output)
    assert output.read_text(encoding="utf-8") == "preserve"


def test_physical_failure_keeps_raw_evidence_and_no_speedup():
    module = load_tool()

    def fail_advance(*args, **kwargs):
        raise Failure("InjectedGate", "raw matched-cost gate failure")

    report, failure = module.run(
        config(methods=["synchronous"]),
        advance_fn=fail_advance,
    )
    assert failure is not None
    assert report["status"] == "failed"
    assert report["speedup_claim"] is None
    row = report["methods"][0]
    assert row["status"] == "failed"
    assert row["warmup"]["status"] == "failed"
    assert row["warmup"]["failure"]["code"] == "InjectedGate"
    assert "run_seconds" not in row
