"""End-to-end behavior tests for the bounded WR benchmark runner."""

from __future__ import annotations

import base64
import gc
import importlib.util
import io
import json
import os
import subprocess
import sys
import weakref
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


TOOL = Path(__file__).resolve().parents[2] / "tools" / "warp_multirate_wr.py"
ENV = {
    **os.environ,
    "PYTHONDONTWRITEBYTECODE": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


@lru_cache(maxsize=1)
def load_tool():
    if not TOOL.is_file():
        pytest.fail(f"runner is absent: {TOOL}")
    spec = importlib.util.spec_from_file_location("warp_multirate_wr", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def config(**changes):
    value = {
        "case": "periodic",
        "points": 9,
        "K": 0,
        "amplitude": 1.0e-4,
        "nonuniform": 0.1,
        "seed_mode": 1,
        "sentinel": 1.0e-4,
        "h": 1.0e-12,
        "M": 1,
        "s": 3,
        "macros": 1,
        "device": "cpu",
        "coupling": "wr",
        "rtol": 1.0e-8,
        "max_iterations": 6,
        "wall_seconds": 60.0,
    }
    value.update(changes)
    return value


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_runner_rejects_unknown_and_missing_controls_before_output():
    module = load_tool()
    missing = config()
    missing.pop("seed_mode")
    with pytest.raises(ValueError, match="missing"):
        module.validate_config(missing)
    with pytest.raises(ValueError, match="unknown"):
        module.validate_config({**config(), "surprise": 1})
    defaulted = config()
    defaulted.pop("h")
    resolved = module.validate_config(defaulted)
    assert resolved["h"] == module.DEFAULT_H
    assert resolved["H"] == module.DEFAULT_H * defaulted["M"]
    with pytest.raises(ValueError, match="nine|9"):
        module.validate_config(config(points=7))


def test_runtime_source_paths_relocate_with_the_isolated_source_tree():
    module = load_tool()
    paths = module._source_paths()
    root = module.SOURCE_ROOT.resolve()
    assert "docs/pseudocode/src/fomhd/multirate_wr_physics.py.md" in paths
    assert all(path.resolve().is_relative_to(root) for path in paths.values())


def test_existing_output_is_refused_without_mutation(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "result.json"
    write_json(input_path, config())
    output_path.write_bytes(b'{"preserve":true}\n')
    before = output_path.read_bytes()

    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=TOOL.parents[1],
        env=ENV,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert output_path.read_bytes() == before


def test_periodic_cli_writes_lossless_arrays_and_independent_references(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "result.json"
    value = config()
    write_json(input_path, value)
    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=TOOL.parents[1],
        env=ENV,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(output_path.read_text())

    assert report["status"] == "completed"
    assert report["config"] == value
    assert report["production_accepted"] is False
    assert report["source_hashes"]
    assert {row["kind"] for row in report["completed_windows"]} == {
        "candidate",
        "same_h_reference",
        "half_h_reference",
    }
    assert set(report["references"]) == {"same_h", "half_h"}
    assert report["references"]["same_h"]["full_stage_calls"] >= 5
    assert report["references"]["half_h"]["full_stage_calls"] > (
        report["references"]["same_h"]["full_stage_calls"]
    )
    assert report["references"]["same_h"]["alias_evaluations"] == 2
    assert report["references"]["half_h"]["alias_evaluations"] == 4
    assert report["comparisons"]["accepted_vs_same_h"]["p_normalized"] >= 0.0
    assert report["comparisons"]["accepted_vs_half_h"]["z_normalized"] >= 0.0
    assert report["stage_identities"]
    assert report["timing"]["total_elapsed_s"] >= 0.0
    assert report["memory_samples"]

    encoded = base64.b64decode(report["arrays_npz_base64"], validate=True)
    with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
        for name in (
            "source_P",
            "source_Z",
            "accepted_P",
            "accepted_Z",
            "reference_same_h_P",
            "reference_same_h_Z",
            "reference_half_h_P",
            "reference_half_h_Z",
        ):
            assert name in archive
        assert archive["source_P"].dtype == np.complex128
        assert archive["source_Z"].dtype == np.complex128
    assert report["window_evidence"][0]["sweeps"] >= 1
    assert report["window_evidence"][0]["s"] == 3
    assert any(row.get("kind") == "coarse" for row in report["stage_identities"])
    assert any(row.get("kind") == "fine" for row in report["stage_identities"])
    candidate_coarse = [
        row
        for row in report["stage_identities"]
        if row.get("kind") == "coarse"
    ]
    candidate_fine = [
        row
        for row in report["stage_identities"]
        if row.get("kind") == "fine"
    ]
    assert len(candidate_coarse) == 3
    assert len(candidate_fine) == 4
    middles = [
        row
        for row in candidate_fine
        if row["window_identity"]["rk_stage"] in (1, 2)
    ]
    assert len(middles) == 2
    assert len({row["tag"]["generation"] for row in middles}) == 2
    shapes = report["window_evidence"][0]["stage_output_shapes"]
    assert shapes["coarse"][0]["A"]["shape"]
    assert shapes["coarse"][0]["transport_projection_value"]["shape"]
    assert shapes["fine"][0]["G"]["shape"]
    assert report["window_evidence"][0]["endpoint"]["trial_endpoint"] is True
    assert (
        report["window_evidence"][0]["endpoint"][
            "interval_transaction_required"
        ]
        is True
    )


def test_east_returns_explicit_reuse_interface_without_running(tmp_path):
    module = load_tool()
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "east.json"
    write_json(
        input_path,
        config(
            case="east",
            points=4,
            K=1,
            amplitude=0.02,
            seed_mode=1,
            sentinel=1.0e-4,
        ),
    )
    assert module.run_paths(input_path, output_path) == 2
    report = json.loads(output_path.read_text())
    assert report["status"] == "unsupported_east"
    assert report["east_interface"]["pinned_input"].endswith("g048059.03650")
    assert report["east_interface"]["K"] == 1
    assert report["east_interface"]["Nphi"] == 5
    assert "parent owns" in report["east_interface"]["scope"]


def test_cuda_is_rejected_outside_owned_slurm(monkeypatch, tmp_path):
    module = load_tool()
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="Slurm"):
        module.validate_config(config(device="cuda:0"))


def test_memory_resource_gate_uses_actual_usage_not_reserved_pool():
    module = load_tool()
    module._check_resource_sample(
        {
            "host_rss_bytes": 15 << 30,
            "device_used_bytes": 7 << 30,
            "device_reserved_bytes": 32 << 30,
        }
    )
    with pytest.raises(module.Failure, match="host|device|memory|resource"):
        module._check_resource_sample(
            {
                "host_rss_bytes": (16 << 30) + 1,
                "device_used_bytes": 1,
            }
        )
    with pytest.raises(module.Failure, match="host|device|memory|resource"):
        module._check_resource_sample(
            {
                "host_rss_bytes": 1,
                "device_used_bytes": (8 << 30) + 1,
                "device_reserved_bytes": 9 << 30,
            }
        )


def test_resource_gate_retains_violating_memory_sample(monkeypatch):
    module = load_tool()
    monkeypatch.setattr(
        module,
        "_host_memory",
        lambda: {
            "host_rss_bytes": (16 << 30) + 1,
            "host_peak_rss_bytes": (16 << 30) + 1,
        },
    )
    samples = []
    context = SimpleNamespace(compute=SimpleNamespace(xp=np))
    with pytest.raises(module.Failure, match="host|memory|resource") as caught:
        module._memory_sample(context, "over-limit", samples)
    assert samples[0]["host_rss_bytes"] == (16 << 30) + 1
    assert caught.value.memory_sample == samples[0]
    assert caught.value.code == "ResourceLimit"


def test_candidate_keeps_only_compact_completed_macro_records(
    monkeypatch, tmp_path
):
    module = load_tool()
    context = SimpleNamespace(
        p=SimpleNamespace(start_time=0.0),
        op=SimpleNamespace(Nc=1),
    )
    state = SimpleNamespace(time=0.0)
    references = []

    class FakeResult:
        def __init__(self, endpoint, window):
            self.endpoint = endpoint
            self.work = SimpleNamespace(
                snapshot=lambda: {"full_stage_calls": 1}
            )
            self.window = window

    class FakeAdapter:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def solve_window(self, *args, **kwargs):
            del args, kwargs
            endpoint = SimpleNamespace(
                P=np.zeros((1, 5), dtype=np.complex128),
                Z=np.zeros((1, 4), dtype=np.complex128),
                full=SimpleNamespace(canonical_state=SimpleNamespace(time=0.0)),
                totals=SimpleNamespace(),
            )
            result = FakeResult(
                endpoint,
                SimpleNamespace(
                    s=3,
                    M=1,
                    sweeps=1,
                    accepted_node_version=0,
                    residual_max_scaled=0.0,
                    raw_norm=0.0,
                    scaled_norm=0.0,
                        residual_history=(),
                        Q=np.eye(3),
                        P=np.zeros((3, 1, 5), dtype=np.complex128),
                        Z_coarse=np.zeros((3, 1, 4), dtype=np.complex128),
                        fine_P=np.zeros((1, 4, 1, 5), dtype=np.complex128),
                        fine_Z=np.zeros((1, 4, 1, 4), dtype=np.complex128),
                        fine_G=np.zeros((1, 4, 1, 4), dtype=np.complex128),
                        fine_C=np.zeros((1, 4, 1, 5), dtype=np.complex128),
                    ),
                )
            references.append(weakref.ref(result))
            return result

    monkeypatch.setattr(
        module,
        "_build_periodic",
        lambda config: (context, state, {}),
    )
    monkeypatch.setattr(
        module.wr,
        "pack_state",
        lambda owner, value: (
            np.zeros((1, 5), dtype=np.complex128),
            np.zeros((1, 4), dtype=np.complex128),
        ),
    )
    monkeypatch.setattr(
        module.wr,
        "MasslessWRPhysicsAdapter",
        FakeAdapter,
    )
    monkeypatch.setattr(module, "_add_arrays", lambda *args: None)
    monkeypatch.setattr(module, "_memory_sample", lambda *args: {})
    monkeypatch.setattr(module, "_window_evidence", lambda *args: {})
    monkeypatch.setattr(module, "_stage_identities", lambda *args: [])
    monkeypatch.setattr(module, "_window_mode_amplitudes", lambda *args: {})

    result = module._run_candidate(
        {
            "macros": 3,
            "H": 1.0e-12,
            "M": 1,
            "s": 3,
            "rtol": 1.0e-8,
            "max_iterations": 2,
            "seed_mode": 1,
            "points": 9,
        },
        float("inf"),
        {},
        [],
    )
    assert len(result["windows"]) == 3
    assert all("result" not in window for window in result["windows"])
    assert "accepted_P" not in result["windows"][-1]
    gc.collect()
    assert references[0]() is None
    assert references[1]() is None
    assert references[2]() is result["last_result"]


def test_runtime_failure_writes_preserved_failure_evidence(tmp_path):
    module = load_tool()
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "failed.json"
    value = config(
        amplitude=0.02,
        h=1.0e-12,
        rtol=1.0e-30,
        max_iterations=1,
    )
    write_json(input_path, value)

    assert module.run_paths(input_path, output_path) == 1
    report = json.loads(output_path.read_text())
    assert report["status"] == "failed"
    assert report["failure"]["message"]
    assert report["failure"]["traceback"]
    assert report["completed_windows"] == []
    decoded = base64.b64decode(report["arrays_npz_base64"], validate=True)
    with np.load(io.BytesIO(decoded), allow_pickle=False) as archive:
        assert "source_P" in archive
        assert "source_Z" in archive
        assert any(name.startswith("failure_") for name in archive.files)


def test_synchronous_coupling_never_instantiates_the_wr_engine(
    monkeypatch, tmp_path
):
    module = load_tool()
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "synchronous.json"
    write_json(input_path, config(coupling="synchronous"))

    def forbidden(*args, **kwargs):
        raise AssertionError("synchronous reference instantiated the WR adapter")

    monkeypatch.setattr(module.wr, "MasslessWRPhysicsAdapter", forbidden)
    assert module.run_paths(input_path, output_path) == 0
    report = json.loads(output_path.read_text())
    assert report["status"] == "completed"
    assert report["window_evidence"] == []
    assert set(report["references"]) == {"same_h", "half_h"}


def test_failure_retains_current_completed_candidate_windows(
    monkeypatch, tmp_path
):
    module = load_tool()
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "partial.json"
    write_json(input_path, config(macros=2))
    original = module.wr.MasslessWRPhysicsAdapter
    calls = {"count": 0}

    class SecondWindowFails:
        def __init__(self, *args, **kwargs):
            calls["count"] += 1
            self._inner = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def solve_window(self, *args, **kwargs):
            if calls["count"] == 2:
                raise module.Failure(
                    "InjectedSecondWindow",
                    "injected failure after one completed macro",
                )
            return self._inner.solve_window(*args, **kwargs)

    monkeypatch.setattr(module.wr, "MasslessWRPhysicsAdapter", SecondWindowFails)
    assert module.run_paths(input_path, output_path) == 1
    report = json.loads(output_path.read_text())
    assert report["status"] == "failed"
    assert report["failure"]["code"] == "InjectedSecondWindow"
    assert report["partial_candidate"]["completed_macros"] == [0]
    assert report["partial_candidate"]["windows"][0]["macro"] == 0
    assert report["partial_candidate"]["mode_history"][0]["macro"] == 0
    assert "accepted_macro_0000_P" in report["array_names"]
