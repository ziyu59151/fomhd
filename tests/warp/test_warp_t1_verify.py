"""Focused tests for the bounded A1-T1 evidence driver."""

from __future__ import annotations

import gzip
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from fomhd.types import Failure, StageTag


def module():
    return importlib.import_module("tools.warp_t1_verify")


def _augmented_t0_map(a, beta, H):
    generator = np.asarray(
        [
            [0.0, 1j * a, 0.0, 0.0],
            [0.0, 1j * beta, 1j * a, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    initial = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1j * a],
        ],
        dtype=np.complex128,
    )
    return (expm(generator * H) @ initial)[:2]


def _fine_rk4_endpoint(a, beta, H, M, seed):
    value = np.asarray(seed, dtype=np.complex128).copy()
    h = H / M

    def rhs(state):
        return np.asarray(
            [1j * a * state[1], 1j * a * state[0] + 1j * beta * state[1]],
            dtype=np.complex128,
        )

    for _ in range(M):
        k1 = rhs(value)
        k2 = rhs(value + 0.5 * h * k1)
        k3 = rhs(value + 0.5 * h * k2)
        k4 = rhs(value + h * k3)
        value += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return value


def test_exact_t0_map_matches_augmented_exponential():
    tool = module()
    a = 2.3
    beta = 0.7
    H = 0.17

    observed = tool.exact_t0_map(a, beta, H)
    expected = _augmented_t0_map(a, beta, H)

    np.testing.assert_allclose(
        observed,
        expected,
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_linear_waveform_converges_to_independent_fine_rk4():
    tool = module()
    a = 1.3
    beta = 0.4
    H = 0.2
    M = 4
    seed = np.asarray([0.3 - 0.2j, 0.8 + 0.1j])

    result = tool.linear_waveform_sweeps(
        a,
        beta,
        H,
        M,
        seed,
        max_iterations=40,
        rtol=1.0e-13,
    )
    expected = _fine_rk4_endpoint(a, beta, H, M, seed)

    assert result["converged"] is True
    assert len(result["iterations"]) > 1
    assert np.asarray(result["path"]).shape == (M, 4, 2)
    np.testing.assert_allclose(
        result["endpoint"],
        expected,
        rtol=0.0,
        atol=2.0e-14,
    )
    midpoint_difference = np.asarray(result["path"])[:, 1] - np.asarray(
        result["path"]
    )[:, 2]
    assert np.max(np.abs(midpoint_difference)) > 0.0


def test_linear_waveform_stopping_is_repeatable_and_ion_only():
    tool = module()
    arguments = (1.3, 0.4, 0.2, 4, np.asarray([0.3 - 0.2j, 0.8 + 0.1j]))
    results = [
        tool.linear_waveform_sweeps(
            *arguments,
            12,
            rtol=1.0e-10,
        )
        for _ in range(8)
    ]

    first = results[0]
    assert first["stopping_metric"] == (
        "ion_stage_u_path_and_ion_endpoint"
    )
    assert np.asarray(first["path"]).shape == (4, 4, 2)
    assert np.asarray(first["z_stage_path"]).shape == (4, 4)
    assert np.all(np.isfinite(np.asarray(first["z_stage_path"])))
    for record in first["iterations"]:
        ion_criterion = max(
            record["path"]["ion"]["relative_max_abs"],
            record["endpoint"]["ion"]["relative_max_abs"],
        )
        assert record["converged"] == (ion_criterion <= first["rtol"])
        assert np.isfinite(record["z_observation"]["relative_max_abs"])

    for result in results[1:]:
        assert result["converged"] == first["converged"]
        assert result["sweeps"] == first["sweeps"]
        np.testing.assert_array_equal(result["path"], first["path"])
        np.testing.assert_array_equal(
            result["z_stage_path"], first["z_stage_path"]
        )
        np.testing.assert_array_equal(result["endpoint"], first["endpoint"])
        assert [
            item["converged"] for item in result["iterations"]
        ] == [item["converged"] for item in first["iterations"]]


def test_zero_seeded_fast_amplitude_stays_zero():
    tool = module()
    seed = np.zeros(2, dtype=np.complex128)

    result = tool.linear_waveform_sweeps(
        0.8,
        -0.2,
        0.15,
        3,
        seed,
        max_iterations=12,
        rtol=1.0e-12,
    )

    assert result["converged"] is True
    np.testing.assert_allclose(result["endpoint"], seed, atol=1.0e-15)
    assert np.all(np.isfinite(np.asarray(result["path"])))
    assert np.all(np.isfinite(np.asarray(result["z_stage_path"])))


def test_vector_history_comparison_uses_euclidean_vector_error():
    tool = module()
    actual = np.asarray([[3.0 + 0.0j, 4.0 + 0.0j]])
    reference = np.asarray([[0.0 + 0.0j, 1.0 + 0.0j]])

    result = tool._vector_history_comparison(
        actual,
        reference,
        relative_defined=True,
    )

    assert result["maximum_relative_vector_error"] == pytest.approx(
        np.sqrt(18.0)
    )
    assert result["final_relative_vector_error"] == pytest.approx(
        np.sqrt(18.0)
    )
    assert result["maximum_absolute_vector_error"] == pytest.approx(
        np.sqrt(18.0)
    )


def _baseline_fixture(
    root,
    tool,
    *,
    M=2,
    micro_per_period=2048,
    seed="target",
    accepted_steps=2,
):
    schedule = _short_schedule()
    schedule.update(micro_per_period=micro_per_period, micro_steps=M)
    assert 0 <= accepted_steps <= schedule["macro_steps"]
    times = np.arange(accepted_steps + 1, dtype=np.float64)
    times *= schedule["macro_dt_s"]
    target = np.zeros((times.size, 4), dtype=np.complex128)
    target[:, 0] = np.linspace(1.0, 1.0 + 0.1 * accepted_steps, times.size)
    high = np.zeros_like(target)
    if seed == "mixed":
        high[:, 0] = np.linspace(
            0.01,
            0.01 + 0.001 * accepted_steps,
            times.size,
        )
    case = {
        "status": (
            "COMPLETED_WITH_SOFT_FAILURES"
            if accepted_steps == schedule["macro_steps"]
            else "UNDETERMINED"
        ),
        "controls": {
            "device": "cpu",
            "seed": seed,
            "micro_per_period": micro_per_period,
            "micro_steps": M,
            "cycles": 2,
            "points": 9,
        },
        "schedule": schedule,
        "physical": {"kdi": 1.0, "K": 0, "Nphi": 1},
        "source": {
            "source_git": {"commit": "a" * 40, "dirty": False},
            "source_hashes": {"src/fomhd/time.py": "b" * 64},
        },
        "budgets": {},
        "seed": {},
        "counts": {
            "accepted_macro_steps": accepted_steps,
            "requested_macro_steps": schedule["macro_steps"],
        },
    }
    history = {
        "accepted_time_s": times,
        "target_coeff_history": target,
        "high_coeff_history": high,
        "initial_ion_n": np.asarray([1.0]),
        "totals_total_energy_J_history": np.linspace(1.0, 1.1, times.size),
    }
    if accepted_steps:
        history["accepted_ion_n"] = np.arange(
            accepted_steps, dtype=np.float64
        ).reshape(accepted_steps, 1)
    case_path = root / "case.json"
    history_path = root / "history.npz"
    root.mkdir()
    case_path.write_text(json.dumps(case), encoding="utf-8")
    np.savez(history_path, **history)
    return root


def _sync_fixture(root, tool, seed="target"):
    root.mkdir()
    source_hashes = {"src/fomhd/time.py": "b" * 64}
    case_identity = tool.t0._case_identity(
        tool.t0._periodic_case(9),
        seed,
    )
    case = {
        "status": "COMPLETED_WITH_SOFT_FAILURES",
        "controls": {
            "coupling": "synchronous",
            "seed": seed,
            "micro_per_period": 2048,
            "points": 9,
        },
        "schedule": _short_schedule(),
        "physical": case_identity["physical"],
        "discrete": case_identity["discrete"],
        "seed": tool.t0._jsonable(
            tool.t0._seed_identity(
                tool.t0.build_seed(points=9, seed=seed)
            )
        ),
        "source": {
            "source_git": {"commit": "e" * 40, "dirty": False},
            "source_hashes": source_hashes,
        },
    }
    arrays = {
        "accepted_time_s": np.asarray([0.0, 1.0, 2.0]),
        "target_coeff_history": np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0]]
        ),
        "high_coeff_history": np.zeros((3, 4), dtype=np.complex128),
    }
    (root / "case.json").write_text(json.dumps(case), encoding="utf-8")
    np.savez(root / "history.npz", **arrays)
    return root, source_hashes


def _short_schedule():
    return {
        "coupling": "predictor",
        "micro_per_period": 2048,
        "micro_steps": 2,
        "cycles": 2,
        "points": 9,
        "omega_target_rad_s": 1.0,
        "target_period_s": 6.283185307179586,
        "horizon_s": 2.0,
        "micro_dt_s": 0.5,
        "macro_dt_s": 1.0,
        "total_micro_steps": 4,
        "macro_steps": 2,
    }


class _FakeTotals:
    pass


class _FakeDeviceArray:
    def __init__(self, value, dtype, device):
        self.host = np.asarray(value, dtype=dtype).copy()
        self.dtype = np.dtype(dtype)
        self.device = device


class _TrackingBackend:
    def __init__(self, device):
        self.device = device
        self.name = "cpu" if device == "cpu" else "cuda"
        self.uploads = []

    def array(self, value, dtype=None):
        if not isinstance(value, np.ndarray):
            raise TypeError("explicit upload requires a NumPy host array")
        dtype = np.dtype(dtype)
        self.uploads.append((value.shape, dtype, value.copy()))
        return _FakeDeviceArray(value, dtype, self.device)


def _fake_seed_state():
    def fluid(scale):
        return SimpleNamespace(
            n=np.full(3, scale + 1.0, dtype=np.complex128),
            mom=np.full((3, 3), scale + 2.0, dtype=np.complex128),
            perp=np.full(3, scale + 3.0, dtype=np.complex128),
            parallel=np.full(3, scale + 4.0, dtype=np.complex128),
        )

    return SimpleNamespace(
        fluid=(fluid(0.0), fluid(10.0)),
        B=np.full(6, 7.0, dtype=np.complex128),
        step=4,
        generation=9,
        time=1.25e-7,
    )


def _fake_built_seed(state):
    return {
        "state": state,
        "context": "cpu-construction-context",
        "normalization": "unchanged-seed-normalization",
        "vectors": {1: np.asarray([1.0 + 2.0j])},
        "coefficients": {1: np.asarray([3.0 + 4.0j])},
        "amplitudes": {1: 5.0},
        "eigenvalues": {1: 6.0j},
        "projected_coefficients": {1: np.asarray([7.0 + 8.0j])},
    }


def _fake_prepared(tool, *, fail_at=None):
    case = tool.t0._periodic_case(9)
    p = SimpleNamespace(
        coupling="waveform",
        micro_steps=2,
        waveform_rtol=1.0e-10,
        waveform_max_iterations=12,
        integrator="rk4",
        dt=1.0,
        nsteps=2,
        dt_guard="error",
        memory_limit=tool.MEMORY_CAP_BYTES,
        device_memory_limit=tool.MEMORY_CAP_BYTES,
        budgets={},
    )
    compute = SimpleNamespace(
        device="cpu",
        name="cpu",
        synchronize=lambda: None,
    )
    state = SimpleNamespace(
        step=0,
        generation=0,
        time=0.0,
        payload=np.asarray([0.0]),
    )
    committed = SimpleNamespace(state=state, accepted=True)
    calls = {"count": 0, "advanced_before_input": False}

    def advance(committed, cumulative, context, source):
        calls["count"] += 1
        if calls["count"] == fail_at:
            error = Failure(
                "WaveformConvergence",
                "synthetic sweep limit reached",
            )
            error.stage = StageTag(step=1, stage="waveform")
            error.checks = []
            error.partial_arrays = {
                "current_trial": np.asarray([3.0, 4.0])
            }
            error.coupling_metadata = {
                "coupling": "waveform",
                "iterations": [
                    {
                        "iteration": 0,
                        "path": {"u": {"relative_max_abs": 0.2}},
                        "endpoint": {"u": {"relative_max_abs": 0.1}},
                        "unknown_nested": {"preserve": [1, 2, 3]},
                    }
                ],
            }
            raise error
        old_time = committed.state.time
        committed.state = SimpleNamespace(
            step=calls["count"],
            generation=calls["count"] + 1,
            time=old_time + 1.0,
            payload=np.asarray([float(calls["count"])]),
        )
        return SimpleNamespace(
            checks=[],
            totals=_FakeTotals(),
            coupling_metadata={
                "coupling": "waveform",
                "iterations": [
                    {
                        "iteration": 0,
                        "path": {"u": {"relative_max_abs": 0.1}},
                        "endpoint": {"u": {"relative_max_abs": 0.05}},
                    },
                    {
                        "iteration": 1,
                        "path": {"u": {"relative_max_abs": 1.0e-12}},
                        "endpoint": {
                            "u": {"relative_max_abs": 5.0e-13}
                        },
                        "energy_checks": [{"raw": 0.0}],
                    },
                ],
                "total_model_evaluations": 18,
                "peak_memory_bytes": 4096,
            },
        )

    return SimpleNamespace(
        schedule=_short_schedule(),
        case=case,
        context=SimpleNamespace(p=p, compute=compute),
        seed={"seed": "target"},
        raw=SimpleNamespace(),
        committed=committed,
        cumulative=object(),
        initial_checks=[],
        initial_totals=_FakeTotals(),
        raw_state_arrays={},
        seed_upload={
            "explicit_upload": False,
            "uploaded_field_count": 0,
            "runtime_cpu_fallback": False,
        },
        budgets={},
        mode_specs={
            1: {
                "amplitude": 2.0e-6,
                "vector": np.eye(9, dtype=np.complex128)[1],
                "transverse_vector": np.asarray([1, 0, 0, 0]),
                "eigenvalue": 1j,
            },
            4: {
                "amplitude": 0.0,
                "vector": np.eye(9, dtype=np.complex128)[1],
                "transverse_vector": np.asarray([1, 0, 0, 0]),
                "eigenvalue": 4j,
            },
        },
        calls=calls,
        advance=advance,
    )


def _install_fake_pipeline(monkeypatch, tool, *, fail_at=None):
    schedule = _short_schedule()
    prepared = _fake_prepared(tool, fail_at=fail_at)

    def sync_reference(baseline_path, controls):
        if controls["seed"] == "mixed":
            prepared.mode_specs[4]["amplitude"] = 2.0e-8
        high = (
            np.zeros((3, 4), dtype=np.complex128)
            if controls["seed"] == "target"
            else np.asarray(
                [[0.01, 0, 0, 0], [0.011, 0, 0, 0], [0.012, 0, 0, 0]],
                dtype=np.complex128,
            )
        )
        return {
            "available": True,
            "path": str(Path(baseline_path).parent / "sync-h2048-target"),
            "case_json_path": "sync/case.json",
            "history_npz_path": "sync/history.npz",
            "case_json_sha256": "d" * 64,
            "history_npz_sha256": "f" * 64,
            "combined_sha256": "0" * 64,
            "identity": {
                "coupling": "synchronous",
                "seed": controls["seed"],
                "points": 9,
                "horizon_s": schedule["horizon_s"],
                "source": {},
            },
            "compatibility": {
                "commit_mismatch_accepted": False,
                "numerical_source_hashes_equal": True,
            },
            "case": {},
            "metadata": {"available": True},
            "arrays": {
                "accepted_time_s": np.asarray([0.0, 1.0, 2.0]),
                "target_coeff_history": np.asarray(
                    [
                        [0.1, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.1, 0.1],
                    ],
                    dtype=np.complex128,
                ),
                "high_coeff_history": high,
            },
        }

    monkeypatch.setattr(tool, "schedule", lambda **kwargs: dict(schedule))
    monkeypatch.setattr(
        tool,
        "_load_synchronous_reference",
        sync_reference,
    )
    monkeypatch.setattr(
        tool,
        "_prepare_runtime",
        lambda controls, baseline: prepared,
    )
    monkeypatch.setattr(tool, "_validate_prepared_runtime", lambda *args: None)
    monkeypatch.setattr(
        tool,
        "_capture_endpoint",
        lambda state, context, case, harmonics: {
            "mode9": {
                harmonic: np.full(9, 0.1 * harmonic, dtype=np.complex128)
                for harmonic in harmonics
            },
            "transverse": {
                harmonic: np.full(
                    len(tool.TRANSVERSE_INDEXES),
                    0.1 * harmonic,
                    dtype=np.complex128,
                )
                for harmonic in harmonics
            },
            "state_arrays": {
                "ion_n": np.asarray([state.time + 1.0]),
                "B": np.asarray([state.time + 2.0]),
            },
        },
    )
    monkeypatch.setattr(
        tool,
        "_totals_to_dict",
        lambda totals: {"total_energy_J": 1.0, "number_ion": 1.0},
    )
    monkeypatch.setattr(
        tool,
        "_snapshot_cumulative",
        lambda cumulative, committed: (
            {
                "cumulative_accepted_intervals": np.asarray(
                    committed.state.step
                ),
                "cumulative_state_time": np.asarray(committed.state.time),
            },
            {
                "available": True,
                "identity": {
                    "accepted_intervals": committed.state.step,
                    "state_time": committed.state.time,
                },
            },
        ),
    )
    monkeypatch.setattr(tool, "advance", prepared.advance)
    monkeypatch.setattr(
        tool,
        "_build_provenance",
        lambda argv, device: {
            "source_git": {"commit": "a" * 40, "dirty": False},
            "file_hashes": {"tools/warp_t1_verify.py": "c" * 64},
            "selected_device": device,
        },
    )
    return prepared


def test_invalid_controls_and_existing_output_have_no_mutation(tmp_path):
    tool = module()
    invalid = tmp_path / "invalid"
    with pytest.raises(ValueError, match="micro_per_period"):
        tool.run_case(
            invalid,
            M=2,
            micro_per_period=1024,
            t0_baseline=tmp_path / "missing",
        )
    assert not invalid.exists()

    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "preserve.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        tool.run_case(
            existing,
            M=2,
            t0_baseline=tmp_path / "missing",
        )
    assert marker.read_text(encoding="utf-8") == "keep"
    assert sorted(path.name for path in existing.iterdir()) == ["preserve.txt"]


def test_cli_defaults_and_waveform_validation():
    tool = module()
    args = tool._parser().parse_args(["--output", "case"])
    assert args.device == "cpu"
    assert args.M == 4
    assert args.micro_per_period == 2048
    assert args.cycles == 2
    assert args.rtol == 1.0e-10
    assert args.max_iterations == 12

    for name, overrides in (
        ("waveform_rtol", {"rtol": 0.0}),
        ("waveform_max_iterations", {"max_iterations": 0}),
        ("waveform_max_iterations", {"max_iterations": True}),
    ):
        arguments = {
            "device": "cpu",
            "M": 2,
            "micro_per_period": 2048,
            "cycles": 2,
            "seed": "target",
            "rtol": 1.0e-10,
            "max_iterations": 12,
        }
        arguments.update(overrides)
        with pytest.raises(ValueError, match=name):
            tool.normalize_controls(**arguments)


def test_input_and_provenance_exist_before_first_advance(tmp_path, monkeypatch):
    tool = module()
    baseline = _baseline_fixture(tmp_path / "baseline", tool)
    prepared = _install_fake_pipeline(monkeypatch, tool)
    output = tmp_path / "output"
    original_advance = prepared.advance

    def guarded_advance(*args):
        assert (output / "input.json").is_file()
        assert (output / "provenance.json").is_file()
        return original_advance(*args)

    monkeypatch.setattr(tool, "advance", guarded_advance)
    tool.run_case(
        output,
        M=2,
        micro_per_period=2048,
        t0_baseline=baseline,
        argv=["warp_t1_verify.py"],
    )

    payload = json.loads((output / "input.json").read_text(encoding="utf-8"))
    assert payload["t0_predictor_baseline"]["path"] == str(
        baseline.resolve()
    )
    assert len(payload["t0_predictor_baseline"]["case_json_sha256"]) == 64
    assert len(payload["t0_predictor_baseline"]["history_npz_sha256"]) == 64
    assert len(payload["synchronous_reference"]["case_json_sha256"]) == 64
    assert len(payload["synchronous_reference"]["history_npz_sha256"]) == 64
    assert payload["source"]["source_git"]["commit"] == "a" * 40
    assert payload["controls"]["coupling"] == "waveform"
    provenance = json.loads(
        (output / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["synchronous_reference"]["case_json_sha256"] == (
        "d" * 64
    )
    assert provenance["synchronous_reference"]["history_npz_sha256"] == (
        "f" * 64
    )


def test_success_writes_history_iterations_and_case(tmp_path, monkeypatch):
    tool = module()
    baseline = _baseline_fixture(tmp_path / "baseline", tool)
    _install_fake_pipeline(monkeypatch, tool)
    output = tmp_path / "success"

    summary = tool.run_case(
        output,
        M=2,
        micro_per_period=2048,
        t0_baseline=baseline,
    )

    assert summary["status"] == "COMPLETED"
    assert summary["accepted_macro_steps"] == 2
    assert summary["expected_macro_steps"] == 2
    assert summary["iterations"]["total"] == 4
    assert summary["counts"]["model_calls"]["total_model_evaluations"] == 36
    assert summary["memory"]["reported_estimates"]["peak_memory_bytes"] == 4096
    assert (output / "case.json").is_file()
    assert (output / "history.npz").is_file()
    assert (output / "accepted-history-raw.npz").is_file()
    assert not (output / "failure.json").exists()
    with np.load(
        output / "accepted-history-raw.npz",
        allow_pickle=False,
    ) as raw_history:
        assert raw_history["accepted_time_s"].size == 3
        assert raw_history["target_coeff_history"].shape == (3, 4)
        np.testing.assert_array_equal(
            raw_history["totals_total_energy_J_history"],
            np.ones(3),
        )
    with np.load(output / "history.npz", allow_pickle=False) as history:
        assert history["accepted_time_s"].size == 3
        assert history["target_coeff_history"].shape == (3, 4)
        assert "cumulative_accepted_intervals" in history.files
        assert "t0_target_coeff_history" in history.files
        assert history["sync_target_coeff_history"].shape == (3, 4)
        assert history["sync_high_coeff_history"].shape == (3, 4)
    with gzip.open(
        output / "checks.jsonl.gz", "rt", encoding="utf-8"
    ) as stream:
        assert sum(1 for _ in stream) == 0
    with (output / "iterations.jsonl").open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    assert len(records) == 2
    assert records[0]["metadata"]["iterations"][1]["iteration"] == 1
    assert summary["synchronous_reference"]["identity"]["coupling"] == (
        "synchronous"
    )
    assert summary["synchronous_reference"]["matched_time_count"] == 3
    assert "t0_predictor_baseline" in summary


def test_postprocess_failure_retains_raw_accepted_history(
    tmp_path, monkeypatch
):
    tool = module()
    baseline = _baseline_fixture(tmp_path / "baseline", tool)
    _install_fake_pipeline(monkeypatch, tool)
    output = tmp_path / "raw-before-postprocess"

    def reject_postprocess(*args, **kwargs):
        raise RuntimeError("injected postprocess failure")

    monkeypatch.setattr(
        tool,
        "_synchronous_comparisons",
        reject_postprocess,
    )
    with pytest.raises(RuntimeError, match="injected postprocess"):
        tool.run_case(
            output,
            M=2,
            micro_per_period=2048,
            t0_baseline=baseline,
        )

    raw_path = output / "accepted-history-raw.npz"
    assert raw_path.is_file()
    with np.load(raw_path, allow_pickle=False) as history:
        assert history["accepted_time_s"].size == 3
        assert history["target_coeff_history"].shape == (3, 4)
        assert history["accepted_ion_n"].shape == (2, 1)
        assert "cumulative_accepted_intervals" in history.files


def test_target_only_high_has_no_ratios_against_noise(tmp_path, monkeypatch):
    tool = module()
    baseline = _baseline_fixture(tmp_path / "baseline", tool)
    _install_fake_pipeline(monkeypatch, tool)
    output = tmp_path / "target-only"

    summary = tool.run_case(
        output,
        M=2,
        micro_per_period=2048,
        t0_baseline=baseline,
    )

    growth = summary["high_frequency_growth"]
    assert growth["matched_synchronous"]["reference_source"] == (
        "synchronous_fine_h_rk4"
    )
    assert growth["matched_synchronous"]["relative_modal_vector_error"] is None
    assert growth["matched_t0_predictor"]["relative_modal_vector_error"] is None
    assert np.isfinite(
        growth["matched_synchronous"]["actual_final_absolute_norm"]
    )
    assert np.isfinite(
        growth["matched_synchronous"]["reference_final_absolute_norm"]
    )
    assert np.isfinite(growth["matched_t0_predictor"]["final_actual_norm"])
    assert np.isfinite(growth["matched_t0_predictor"]["final_reference_norm"])

    def assert_no_noise_ratios(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if "ratio" in key.lower():
                    assert child is None, (key, child)
                assert_no_noise_ratios(child)
        elif isinstance(value, list):
            for child in value:
                assert_no_noise_ratios(child)

    assert_no_noise_ratios(growth["matched_synchronous"])
    assert_no_noise_ratios(growth["matched_t0_predictor"])


def test_mixed_high_growth_uses_synchronous_not_predictor(
    tmp_path, monkeypatch
):
    tool = module()
    baseline = _baseline_fixture(
        tmp_path / "baseline",
        tool,
        seed="mixed",
    )
    _install_fake_pipeline(monkeypatch, tool)
    output = tmp_path / "mixed"

    summary = tool.run_case(
        output,
        M=2,
        micro_per_period=2048,
        seed="mixed",
        t0_baseline=baseline,
    )

    growth = summary["high_frequency_growth"]
    assert growth["matched_synchronous"]["reference_source"] == (
        "synchronous_fine_h_rk4"
    )
    assert growth["matched_synchronous"]["maximum_amplitude_ratio_to_reference"] > 1
    assert "matched_t0_predictor" in growth
    assert growth["matched_t0_predictor"]["reference_source"] == (
        "t0_predictor_prefix"
    )


@pytest.mark.parametrize("accepted_steps", [0, 1])
def test_rejected_t0_prefix_allows_longer_t1_run(
    tmp_path, monkeypatch, accepted_steps
):
    tool = module()
    baseline = _baseline_fixture(
        tmp_path / "baseline",
        tool,
        accepted_steps=accepted_steps,
    )
    _install_fake_pipeline(monkeypatch, tool)
    output = tmp_path / f"prefix-{accepted_steps}"

    summary = tool.run_case(
        output,
        M=2,
        micro_per_period=2048,
        t0_baseline=baseline,
    )

    assert summary["status"] == "COMPLETED"
    assert summary["accepted_macro_steps"] == 2
    coverage = summary["t0_predictor_baseline"]["coverage"]
    assert coverage["matched_macro_endpoints"] == accepted_steps
    assert coverage["expected_macro_steps"] == 2
    assert coverage["future_macro_endpoints_unavailable"] == 2 - accepted_steps
    assert coverage["future_endpoints_compared"] is False
    assert (
        summary["high_frequency_growth"]["matched_t0_predictor"]["coverage"][
            "future_samples_unavailable"
        ]
        == 2 - accepted_steps
    )
    with np.load(output / "history.npz", allow_pickle=False) as history:
        assert history["accepted_time_s"].size == 3
        assert history["t0_target_coeff_history"].shape == (
            accepted_steps + 1,
            4,
        )
        if accepted_steps:
            assert history["t0_accepted_ion_n"].shape == (accepted_steps, 1)
        else:
            assert "t0_accepted_ion_n" not in history.files


def test_failure_retains_traceback_partial_and_cumulative_evidence(
    tmp_path, monkeypatch
):
    tool = module()
    baseline = _baseline_fixture(tmp_path / "baseline", tool)
    _install_fake_pipeline(monkeypatch, tool, fail_at=2)
    output = tmp_path / "failure"

    summary = tool.run_case(
        output,
        M=2,
        micro_per_period=2048,
        t0_baseline=baseline,
    )

    assert summary["status"] == "UNDETERMINED"
    assert summary["accepted_macro_steps"] == 1
    assert summary["failure"]["failure_code"] == "WaveformConvergence"
    payload = json.loads(
        (output / "failure.json").read_text(encoding="utf-8")
    )
    assert "synthetic sweep limit reached" in payload["traceback"]
    assert payload["failure_stage"]["stage"] == "waveform"
    assert payload["scheduled_rejected_time_s"] == pytest.approx(2.0)
    assert payload["accepted_macro_steps"] == 1
    assert payload["invented_rejected_endpoint"] is False
    assert (
        payload["coupling_metadata"]["iterations"][0]["unknown_nested"][
            "preserve"
        ]
        == [1, 2, 3]
    )
    with np.load(output / "failure.npz", allow_pickle=False) as failure:
        assert "partial_current_trial" in failure.files
        assert "accepted_time_s" in failure.files
        assert "cumulative_accepted_intervals" in failure.files
    with (output / "iterations.jsonl").open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    assert records[-1]["status"] == "failed"
    assert records[-1]["metadata"]["iterations"][0]["iteration"] == 0


def test_baseline_clock_mismatch_is_detected(tmp_path):
    tool = module()
    baseline = _baseline_fixture(tmp_path / "baseline", tool)
    with np.load(baseline / "history.npz", allow_pickle=False) as history:
        arrays = {name: history[name] for name in history.files}
    arrays["accepted_time_s"][-1] += 1.0e-3
    np.savez(baseline / "history.npz", **arrays)
    controls = {
        "device": "cpu",
        "M": 2,
        "micro_per_period": 2048,
        "cycles": 2,
        "seed": "target",
        "points": 9,
        "schedule": _short_schedule(),
    }

    with pytest.raises(ValueError, match="clock"):
        tool._load_t0_baseline(baseline, controls)


def test_archived_t0_initial_singleton_axis_matches_unbatched_capture():
    tool = module()
    controls = tool.normalize_controls(
        device="cpu",
        M=4,
        micro_per_period=2048,
        cycles=2,
        seed="target",
        rtol=1.0e-10,
        max_iterations=12,
    )
    baseline = tool._load_t0_baseline(
        Path(
            "/home/lxzeng/runs/fomhd/"
            "a1-t0-controlled__20261106T163530Z/"
            "matrix-final/cases/t0-h2048-M4-target"
        ),
        controls,
    )
    arrays = baseline["arrays"]
    assert arrays["initial_ion_n"].shape == (1, 9)
    assert arrays["initial_ion_momentum"].shape == (1, 27)
    initial = {
        name.removeprefix("initial_"): np.asarray(value[0])
        for name, value in arrays.items()
        if name.startswith("initial_")
    }
    accepted = {
        name.removeprefix("accepted_"): np.asarray(value)
        for name, value in arrays.items()
        if name.startswith("accepted_") and name != "accepted_time_s"
    }
    totals = {
        name.removeprefix("totals_").removesuffix("_history"): np.asarray(value)
        for name, value in arrays.items()
        if name.startswith("totals_") and name.endswith("_history")
    }

    report, saved = tool._baseline_comparisons(
        times=arrays["accepted_time_s"],
        target=arrays["target_coeff_history"],
        high=arrays["high_coeff_history"],
        initial_state_arrays=initial,
        state_arrays=accepted,
        totals_by_name=totals,
        baseline=baseline,
    )

    assert set(report["initial_state"]) == set(initial)
    assert all(
        item["maximum_absolute_error"] == 0.0
        for item in report["initial_state"].values()
    )
    assert saved["t0_initial_ion_n"].shape == (1, 9)
    assert saved["t0_initial_ion_momentum"].shape == (1, 27)


def test_initial_singleton_axis_removal_is_strict():
    tool = module()
    actual = np.zeros(9)
    expected_one_singleton = np.zeros((1, 9))
    expected_different = np.zeros((1, 2, 9))

    canonical_actual, canonical_expected = (
        tool._canonicalize_initial_array(actual, expected_one_singleton)
    )
    assert canonical_actual.shape == (9,)
    assert canonical_expected.shape == (9,)
    with pytest.raises(ValueError, match="initial"):
        tool._canonicalize_initial_array(actual, expected_different)


def test_direct_cli_bootstraps_without_pythonpath(tmp_path):
    tool = module()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("FOMHD_SOURCE_ROOT", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [sys.executable, str(Path(tool.__file__).resolve()), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--M" in completed.stdout
    assert "--micro-per-period" in completed.stdout


def test_source_hash_helper_includes_production_waveform(tmp_path):
    tool = module()
    source = tmp_path / "source"
    waveform = source / "src" / "fomhd" / "time_waveform.py"
    waveform.parent.mkdir(parents=True)
    waveform.write_text("# production waveform\n", encoding="utf-8")

    hashes = tool._add_t1_source_hashes({}, source)

    assert "src/fomhd/time_waveform.py" in hashes
    assert len(hashes["src/fomhd/time_waveform.py"]) == 64


def test_synchronous_reference_uses_t0_loader_and_records_hashes(
    tmp_path, monkeypatch
):
    tool = module()
    sync, _ = _sync_fixture(tmp_path / "sync-h2048-target", tool)
    controls = {
        "seed": "target",
        "points": 9,
        "schedule": _short_schedule(),
    }
    calls = {}

    def load(path, *, seed, horizon, points):
        calls.update(
            path=Path(path),
            seed=seed,
            horizon=horizon,
            points=points,
        )
        return (
            {
                "accepted_time_s": np.asarray([0.0, 1.0, 2.0]),
                "target_coeff_history": np.ones((3, 4)),
                "high_coeff_history": np.zeros((3, 4)),
            },
            {"available": True, "status": "COMPLETED_WITH_SOFT_FAILURES"},
        )

    monkeypatch.setattr(tool.t0, "_load_matched_reference", load)
    reference = tool._load_synchronous_reference(sync, controls)

    assert calls["path"] == sync
    assert calls["seed"] == "target"
    assert calls["horizon"] == _short_schedule()["horizon_s"]
    assert calls["points"] == 9
    assert reference["path"] == str(sync.resolve())
    assert len(reference["case_json_sha256"]) == 64
    assert len(reference["history_npz_sha256"]) == 64
    assert reference["identity"]["coupling"] == "synchronous"


def test_sync_reference_accepts_documented_commit_only_mismatch(
    tmp_path, monkeypatch
):
    tool = module()
    sync, source_hashes = _sync_fixture(
        tmp_path / "sync-h2048-target",
        tool,
    )
    controls = {
        "seed": "target",
        "points": 9,
        "schedule": _short_schedule(),
    }

    def reject_commit(*args, **kwargs):
        raise ValueError("reference-case source commit differs")

    monkeypatch.setattr(tool.t0, "_load_matched_reference", reject_commit)
    monkeypatch.setattr(
        tool.t0,
        "_runtime_source_hashes",
        lambda: dict(source_hashes),
    )
    reference = tool._load_synchronous_reference(sync, controls)

    assert reference["compatibility"]["commit_mismatch_accepted"] is True
    assert reference["compatibility"]["numerical_source_hashes_equal"] is True


def test_sync_reference_records_integrated_source_hash_differences(
    tmp_path, monkeypatch
):
    tool = module()
    sync, source_hashes = _sync_fixture(
        tmp_path / "sync-h2048-target",
        tool,
    )
    controls = {
        "seed": "target",
        "points": 9,
        "schedule": _short_schedule(),
    }

    def reject_hashes(*args, **kwargs):
        raise ValueError(
            "matched reference numerical runtime source hashes differ"
        )

    current_hashes = {
        **source_hashes,
        "src/fomhd/time_waveform.py": "9" * 64,
    }
    monkeypatch.setattr(tool.t0, "_load_matched_reference", reject_hashes)
    monkeypatch.setattr(
        tool.t0,
        "_runtime_source_hashes",
        lambda: dict(current_hashes),
    )
    monkeypatch.setattr(
        tool.t0,
        "_source_git",
        lambda: {"commit": "f" * 40, "dirty": True},
    )

    reference = tool._load_synchronous_reference(sync, controls)

    compatibility = reference["compatibility"]
    assert compatibility["source_identity_mismatch_accepted"] is True
    assert compatibility["numerical_source_hashes_equal"] is False
    assert compatibility["different_source_files"] == [
        "src/fomhd/time_waveform.py"
    ]


def _owning_state_arrays(state):
    return [
        *(getattr(fluid, name) for fluid in state.fluid for name in (
            "n",
            "mom",
            "perp",
            "parallel",
        )),
        state.B,
    ]


def test_seed_upload_moves_all_owning_fields_to_fake_cuda():
    tool = module()
    host_state = _fake_seed_state()
    built = _fake_built_seed(host_state)
    backend = _TrackingBackend("cuda:0")
    context = SimpleNamespace(compute=backend)

    uploaded, report = tool._seed_state_for_backend(built, context)

    assert len(backend.uploads) == 9
    assert report["uploaded_field_count"] == 9
    arrays = _owning_state_arrays(uploaded)
    assert len(arrays) == 9
    assert all(isinstance(value, _FakeDeviceArray) for value in arrays)
    assert all(value.device == "cuda:0" for value in arrays)
    assert all(value.dtype == np.dtype(np.complex128) for value in arrays)
    assert uploaded.step == host_state.step
    assert uploaded.generation == host_state.generation
    assert uploaded.time == host_state.time
    assert built["normalization"] == "unchanged-seed-normalization"
    np.testing.assert_array_equal(built["coefficients"][1], [3.0 + 4.0j])


def test_cpu_seed_route_preserves_original_state_without_upload():
    tool = module()
    host_state = _fake_seed_state()
    built = _fake_built_seed(host_state)
    backend = _TrackingBackend("cpu")
    context = SimpleNamespace(compute=backend)

    selected, report = tool._seed_state_for_backend(built, context)

    assert selected is host_state
    assert backend.uploads == []
    assert report["explicit_upload"] is False
    for actual, expected in zip(
        _owning_state_arrays(selected),
        _owning_state_arrays(host_state),
    ):
        np.testing.assert_array_equal(actual, expected)


def test_seed_upload_failure_does_not_fall_back_to_cpu():
    tool = module()
    host_state = _fake_seed_state()
    built = _fake_built_seed(host_state)

    class RejectUpload:
        device = "cuda:0"

        @staticmethod
        def array(value, dtype=None):
            raise TypeError("selected backend rejected host upload")

    context = SimpleNamespace(compute=RejectUpload())
    with pytest.raises(Failure, match="upload failed") as caught:
        tool._seed_state_for_backend(built, context)

    assert caught.value.code == "SeedUpload"
    assert all(
        isinstance(value, np.ndarray)
        for value in _owning_state_arrays(built["state"])
    )


def test_prepare_runtime_builds_seed_without_cuda_context(monkeypatch):
    tool = module()
    backend = _TrackingBackend("cuda:0")
    p = SimpleNamespace(budgets={})
    context = SimpleNamespace(p=p, compute=backend)
    host_state = _fake_seed_state()
    built = _fake_built_seed(host_state)
    build_kwargs = {}

    def build_seed(**kwargs):
        build_kwargs.update(kwargs)
        return built

    def initialize(state, selected_context, cumulative, **kwargs):
        assert selected_context is context
        assert all(
            isinstance(value, _FakeDeviceArray)
            for value in _owning_state_arrays(state)
        )
        kwargs["captured"]["checks"] = []
        return SimpleNamespace(state=state, accepted=True)

    monkeypatch.setattr(
        tool.t0,
        "build_periodic_context",
        lambda case, device: context,
    )
    monkeypatch.setattr(tool.t0, "build_seed", build_seed)
    monkeypatch.setattr(tool.t0, "_mode_specs", lambda seed, case: {})
    monkeypatch.setattr(tool, "initialize", initialize)
    controls = {
        "device": "cuda:0",
        "points": 9,
        "M": 4,
        "seed": "target",
        "memory_limit": tool.MEMORY_CAP_BYTES,
        "device_memory_limit": tool.MEMORY_CAP_BYTES,
        "waveform_rtol": 1.0e-10,
        "waveform_max_iterations": 12,
        "schedule": {
            "macro_dt_s": 1.0e-10,
            "macro_steps": 1,
        },
    }

    prepared = tool._prepare_runtime(controls, {})

    assert "context" not in build_kwargs
    assert build_kwargs == {"points": 9, "seed": "target"}
    assert built["context"] is context
    assert prepared.seed_upload["uploaded_field_count"] == 9
    assert len(backend.uploads) == 9
