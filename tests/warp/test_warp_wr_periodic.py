"""CPU regressions for the segmented periodic WR campaign runner."""

from __future__ import annotations

import json

import numpy as np
import pytest

import fomhd.io as io
from fomhd.types import Failure
from fomhd import time as time_module


def _campaign(output, *, M=1, h=1.0e-10, segment_macros=2, **changes):
    config = {
        "device": "cpu",
        "coupling": "multirate_wr",
        "M": M,
        "s": 3,
        "h": h,
        "total_microsteps": 2,
        "segment_macros": segment_macros,
        "wall_seconds": 60.0,
        "initial": "tangent",
        "points": 9,
        "nonuniform": 0.1,
        "amplitude": 0.02,
        "sentinel": 1.0e-4,
        "seed_mode": 1,
        "waveform_rtol": 1.0e-8,
        "waveform_max_iterations": 8,
        "wr_relaxation": 1.0,
        "output": str(output),
        "resume": None,
    }
    config.update(changes)
    return config


def _checkpoint(path):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def _cumulative(arrays):
    return json.loads(bytes(arrays["cumulative_json"]).decode("utf-8"))


def test_two_macro_restart_matches_uninterrupted(tmp_path):
    from tools.warp_wr_periodic import run_campaign

    uninterrupted = run_campaign(
        _campaign(tmp_path / "uninterrupted"),
    )
    prefix = run_campaign(
        _campaign(
            tmp_path / "prefix",
            segment_macros=1,
        ),
    )
    resumed = run_campaign(
        _campaign(
            tmp_path / "resumed",
            segment_macros=1,
            resume=str(prefix["output"]),
        ),
    )

    assert uninterrupted["status"] == "complete"
    assert prefix["status"] == "partial"
    assert resumed["status"] == "complete"
    full_arrays = _checkpoint(uninterrupted["checkpoint"])
    resumed_arrays = _checkpoint(resumed["checkpoint"])
    np.testing.assert_array_equal(
        resumed_arrays["accepted_P"],
        full_arrays["accepted_P"],
    )
    np.testing.assert_array_equal(
        resumed_arrays["accepted_Z"],
        full_arrays["accepted_Z"],
    )
    assert _cumulative(resumed_arrays) == _cumulative(full_arrays)
    history = [
        json.loads(line)
        for line in resumed["history"].read_text(encoding="utf-8").splitlines()
    ]
    assert [item["completed_macros"] for item in history] == [1, 2]
    assert all(item["work"]["full_ion_calls"] > 0 for item in history)
    assert all(item["work"]["field_evaluations"] > 0 for item in history)
    with np.load(
        resumed["output"] / "snapshots.npz",
        allow_pickle=False,
    ) as snapshots:
        assert {
            "source_P",
            "source_Z",
            "first_P",
            "first_Z",
            "final_P",
            "final_Z",
        } == set(snapshots.files)


def test_output_scheme_and_failed_trial_controls(tmp_path, monkeypatch):
    from tools.warp_wr_periodic import run_campaign

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        run_campaign(_campaign(existing))

    prefix = run_campaign(
        _campaign(
            tmp_path / "prefix",
            segment_macros=1,
        ),
    )
    with pytest.raises(ValueError, match="scheme|identity|resume"):
        run_campaign(
            _campaign(
                tmp_path / "foreign",
                segment_macros=1,
                coupling="waveform",
                resume=str(prefix["output"]),
            )
        )

    original = time_module.advance
    calls = 0

    def fail_second(committed, cumulative, context):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise Failure("InjectedTrial", "bounded second-macro rejection")
        return original(committed, cumulative, context)

    with pytest.raises(Failure, match="second-macro"):
        run_campaign(
            _campaign(
                tmp_path / "failure",
                advance_fn=fail_second,
            )
        )
    failed = _checkpoint(tmp_path / "failure" / "checkpoint.npz")
    history = (tmp_path / "failure" / "history.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(history) == 1
    assert int(failed["completed_macros"]) == 1
    assert (tmp_path / "failure" / "failure.json").is_file()


def test_fixed_H_halving_h_does_not_multiply_full_ion_calls(tmp_path):
    from tools.warp_wr_periodic import run_campaign

    coarse = run_campaign(
        _campaign(
            tmp_path / "coarse",
            M=1,
            h=4.0e-10,
            total_microsteps=1,
            segment_macros=1,
        )
    )
    fine = run_campaign(
        _campaign(
            tmp_path / "fine",
            M=2,
            h=2.0e-10,
            total_microsteps=2,
            segment_macros=1,
        )
    )
    coarse_work = coarse["history_rows"][-1]["work"]
    fine_work = fine["history_rows"][-1]["work"]
    assert fine_work["full_ion_calls"] <= coarse_work["full_ion_calls"]
    assert fine_work["coarse_node_calls"] <= coarse_work["coarse_node_calls"] * 2
    assert fine_work["reported"]["alias_field_evaluations"] == (
        2 * coarse_work["reported"]["alias_field_evaluations"]
    )


@pytest.mark.parametrize(
    "coupling",
    ("synchronous", "predictor", "waveform", "multirate_wr"),
)
def test_supported_couplings_use_complete_advance(tmp_path, coupling):
    from tools.warp_wr_periodic import run_campaign

    result = run_campaign(
        _campaign(
            tmp_path / coupling,
            coupling=coupling,
            total_microsteps=1,
            segment_macros=1,
        )
    )
    assert result["status"] == "complete"
    row = result["history_rows"][-1]
    assert row["clock"]["step"] == 1
    assert row["timing"]["synchronized_advance"] is True
    assert row["checks"]["hard_failed"] == 0


def test_cli_limits_and_gpu_guard_are_rejected_before_output(tmp_path, monkeypatch):
    from tools.warp_wr_periodic import run_campaign

    with pytest.raises(ValueError, match="divisible"):
        run_campaign(
            _campaign(
                tmp_path / "indivisible",
                total_microsteps=3,
                M=2,
            )
        )
    with pytest.raises(ValueError, match="1600"):
        run_campaign(
            _campaign(
                tmp_path / "wall",
                wall_seconds=1601.0,
            )
        )
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="SLURM"):
        run_campaign(
            _campaign(
                tmp_path / "gpu",
                device="cuda:0",
            )
        )


def test_actual_33_m32_rejection_is_not_masked_by_metadata_publication(tmp_path):
    from tools.warp_wr_periodic import run_campaign

    output = tmp_path / "actual-probe"
    with pytest.raises(Failure) as caught:
        run_campaign(
            _campaign(
                output,
                M=32,
                h=1.9794733885623e-10,
                total_microsteps=4096,
                segment_macros=1,
                wall_seconds=1600.0,
                points=33,
                waveform_rtol=1.0e-10,
                waveform_max_iterations=12,
            )
        )
    assert caught.value.code == "HardGate"
    assert "kinetic_energy" in str(caught.value)

    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    failure = json.loads(
        (output / "failure.json").read_text(encoding="utf-8")
    )
    assert run["status"] == "failed"
    assert run["failure"] == "HardGate"
    assert failure["code"] == "HardGate"
    assert "kinetic_energy" in failure["message"]
    assert not (output / "checkpoint.npz").exists()
    assert (output / "history.jsonl").read_text(encoding="utf-8") == ""
    # Safe metadata uses explicit nonfinite sentinels and cannot contain bare
    # JSON NaN/Infinity.
    text = json.dumps(failure, allow_nan=False)
    assert "NaN" not in text or "nonfinite" in text
