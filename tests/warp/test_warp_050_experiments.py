from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRATCH = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(
    os.environ.get(
        "FOMHD_SOURCE_ROOT",
        "/home/lxzeng/.codex/worktrees/fomhd-050-experiments/fomhd",
    )
).resolve()
TOOL = SCRATCH / "tools" / "warp_050_experiments.py"


def _tool():
    return importlib.import_module("tools.warp_050_experiments")


def test_device_memory_sample_survives_supported_pinned_pool_api(monkeypatch):
    """Pinned pools expose block counts, while device pools expose bytes."""
    fake = SimpleNamespace(
        get_default_memory_pool=lambda: SimpleNamespace(
            used_bytes=lambda: 12345, total_bytes=lambda: 23456
        ),
        get_default_pinned_memory_pool=lambda: SimpleNamespace(
            n_free_blocks=lambda: 3
        ),
        cuda=SimpleNamespace(runtime=SimpleNamespace(
            memGetInfo=lambda: (100000, 200000)
        )),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake)
    sample = _tool()._memory_sample(
        SimpleNamespace(compute=SimpleNamespace(name="cuda")), 0.25
    )
    assert sample["device_pool_used_bytes"] == 12345
    assert sample["device_pool_total_bytes"] == 23456
    assert sample["pinned_pool_free_blocks"] == 3
    assert sample["device_wide_used_bytes"] == 100000
    assert "device_allocator_error" not in sample


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    parts = [str(SOURCE_ROOT / "src"), str(SOURCE_ROOT), str(SCRATCH)]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["FOMHD_SOURCE_ROOT"] = str(SOURCE_ROOT)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    return env


@pytest.fixture(scope="module")
def small_cae_case():
    from tools.warp_tokamak_cae import background, make_context

    context, meta, equilibrium = make_context(
        "circle", 4, 0, "cpu", memory_limit_bytes=16 << 30
    )
    initial, arrays, mapping = background(context, meta, equilibrium)
    return context, meta, initial, arrays, mapping


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_seed_archive(
    directory: Path,
    context,
    initial,
    *,
    identity=None,
    frequency_hz: float = 1.0e6,
) -> dict:
    from tools.warp_tokamak_cae import host, physical_identity

    directory.mkdir()
    nc = context.op.Nc
    seed = {
        "seed_n": np.zeros(nc, dtype=np.complex128),
        "seed_mom": np.zeros(3 * nc, dtype=np.complex128),
        "seed_pi": np.zeros(nc, dtype=np.complex128),
        "seed_perp": np.zeros(nc, dtype=np.complex128),
        "seed_parallel": np.zeros(nc, dtype=np.complex128),
        "seed_B": np.zeros(context.op.Nb, dtype=np.complex128),
    }
    seed["seed_n"][0] = context.p.ref.ni
    seed["seed_mom"][1] = (
        context.p.species[0].mass
        * context.p.ref.ni
        * context.p.ref.velocity
    )
    seed["seed_pi"][0] = context.p.ref.pressure
    seed["seed_perp"][0] = context.p.ref.pressure
    seed["seed_parallel"][0] = context.p.ref.pressure
    seed["seed_B"][0] = context.p.ref.magnetic
    backgrounds = {
        "background_B": host(initial.B, context),
        "background_n": host(initial.fluid[0].n, context),
        "background_mom": host(initial.fluid[0].mom, context),
        "background_pi": host(initial.fluid[0].perp, context),
        "background_perp": host(initial.fluid[1].perp, context),
        "background_parallel": host(initial.fluid[1].parallel, context),
    }
    np.savez(directory / "spectrum.npz", **seed, **backgrounds)
    source_hashes = {
        path.name: _sha(path)
        for path in (SOURCE_ROOT / "src" / "fomhd").glob("*.py")
        if path.name != "energy.py"
    }
    report = {
        "kind": "circle",
        "nodes": 4,
        "K": 0,
        "mode": 0,
        "source_hashes": source_hashes,
        "physical_identity": (
            physical_identity(context) if identity is None else identity
        ),
        "selected": {
            "index": 7,
            "frequency_hz": frequency_hz,
            "growth_rate_per_s": -1000.0,
        },
    }
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def test_parser_accepts_only_bounded_t1_cases():
    parser = _tool()._parser()
    for nodes in (12, 16, 24, 32):
        args = parser.parse_args(
            [
                "--kind",
                "circle",
                "--nodes",
                str(nodes),
                "--K",
                "1",
                "--mode",
                "1",
                "--device",
                "cpu",
                "--coupling",
                "waveform",
                "--micro-steps",
                "4",
                "--horizon-s",
                "1e-8",
                "--h",
                "auto",
                "--probe-macros",
                "1",
                "--snapshot-every",
                "2",
                "--output",
                "/tmp/new-run",
            ]
        )
        assert args.nodes == nodes
        assert args.device == "cpu"
        assert args.probe_macros == 1
        assert args.snapshot_every == 2
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--kind",
                "circle",
                "--nodes",
                "4",
                "--horizon-s",
                "1e-8",
                "--output",
                "/tmp/new-run",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--kind",
                "circle",
                "--nodes",
                "12",
                "--horizon-s",
                "1e-8",
                "--wall-seconds",
                "1601",
                "--output",
                "/tmp/new-run",
            ]
        )


def test_time_plan_uses_same_physical_horizon_and_micro_h():
    plan = _tool()
    horizon = 1.2e-6
    h = 1.0e-7
    synchronous = plan.plan_time(horizon, h, 1, "synchronous")
    waveform = plan.plan_time(horizon, h, 4, "waveform")
    assert synchronous["dt_s"] == pytest.approx(h, rel=0, abs=0)
    assert waveform["dt_s"] == pytest.approx(4 * h, rel=0, abs=0)
    assert synchronous["nsteps"] == 12
    assert waveform["nsteps"] == 3
    assert synchronous["horizon_s"] == pytest.approx(horizon, rel=0, abs=0)
    assert waveform["horizon_s"] == pytest.approx(horizon, rel=0, abs=0)
    assert synchronous["micro_steps_total"] == 12
    assert waveform["micro_steps_total"] == 12


def test_snapshot_cadence_is_bounded_and_always_contains_final_step():
    bounded = _tool().snapshot_steps(101, 16)
    assert bounded[-1] == 101
    assert len(bounded) <= math.ceil(101 / 16) + 2
    assert all(right > left for left, right in zip(bounded, bounded[1:]))


def test_archive_seed_read_does_not_require_eigenpairs(
    tmp_path, small_cae_case
):
    context, _, initial, _, _ = small_cae_case
    archive = tmp_path / "spectrum"
    report = _write_seed_archive(archive, context, initial)
    with np.load(archive / "spectrum.npz", allow_pickle=False) as arrays:
        assert "jacobian" not in arrays.files
        assert "eigenvalues" not in arrays.files
        assert "eigenvectors" not in arrays.files
    loaded = _tool().read_archive_seed(
        archive,
        context,
        initial,
        kind="circle",
        nodes=4,
        K=0,
        mode=0,
        target_frequency_hz=1.0e6,
    )
    assert loaded["selected"]["index"] == 7
    assert loaded["selected"]["frequency_hz"] == pytest.approx(1.0e6)
    assert set(loaded["fields"]) == {
        "n",
        "mom",
        "pi",
        "perp",
        "parallel",
        "B",
    }
    assert loaded["eigenpair_arrays_read"] == []
    assert loaded["source_hash_match"] is True
    assert loaded["report_source_hashes"] == report["source_hashes"]


def test_archive_physical_identity_mismatch_is_rejected(
    tmp_path, small_cae_case
):
    context, _, initial, _, _ = small_cae_case
    malformed = _tool().physical_identity(context)
    malformed["references"]["ni"] *= 1.000001
    archive = tmp_path / "bad-spectrum"
    _write_seed_archive(archive, context, initial, identity=malformed)
    with pytest.raises(ValueError, match="physical identity mismatch"):
        _tool().read_archive_seed(
            archive,
            context,
            initial,
            kind="circle",
            nodes=4,
            K=0,
            mode=0,
            target_frequency_hz=1.0e6,
        )


def test_runtime_source_resolution_precedes_editable_installation():
    code = (
        "import json;"
        "from tools.warp_050_experiments import source_identity;"
        "print(json.dumps(source_identity()))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=SCRATCH,
        env=_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    identity = json.loads(completed.stdout)
    assert Path(identity["source_root"]) == SOURCE_ROOT
    assert Path(identity["fomhd_package"]) == (
        SOURCE_ROOT / "src" / "fomhd" / "__init__.py"
    )


def test_tiny_cpu_waveform_macro_probe_is_budgeted(tmp_path):
    from tools.warp_tokamak_cae import background, make_context

    def small_context(kind, nodes, K, device, equilibrium, memory_limit_bytes):
        return make_context(
            kind,
            4,
            K,
            device,
            equilibrium,
            memory_limit_bytes=memory_limit_bytes,
        )

    h = 2.75e-10
    micro_steps = 2
    horizon = 2 * micro_steps * h
    output = tmp_path / "probe"
    args = SimpleNamespace(
        kind="circle",
        nodes=4,
        K=0,
        mode=0,
        device="cpu",
        equilibrium=None,
        coupling="waveform",
        micro_steps=micro_steps,
        horizon_s=horizon,
        h=h,
        output=output,
        wall_seconds=120.0,
        probe_macros=1,
        snapshot_every=16,
        target_frequency_hz=None,
        spectrum_directory=None,
        amplitude=1.0e-4,
    )
    status = _tool().run(args, context_factory=small_context)
    assert status == 2
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "budget_stopped"
    assert report["stop_reason"] == "macro_probe"
    assert report["requested_horizon_s"] == pytest.approx(horizon)
    assert report["planned_macros"] == 2
    assert report["reached_macros"] == 1
    assert report["control"]["accepted_steps"] == 1
    assert report["perturbed"]["accepted_steps"] == 1
    assert report["control"]["time_s"] == pytest.approx(h * micro_steps)
    assert report["perturbed"]["time_s"] == pytest.approx(h * micro_steps)
    assert report["control"]["ledger_accepted_intervals"] == 1
    assert report["perturbed"]["ledger_accepted_intervals"] == 1
    assert report["coupling"]["actual_model_evaluations"] > 0
    assert {
        check["gate"]
        for ensemble in ("control", "perturbed")
        for check in report[ensemble]["equilibrium_residual_checks"]
    } == {"observe"}
    assert (output / "checkpoints" / "control.npz").is_file()
    assert (output / "checkpoints" / "perturbed.npz").is_file()
    assert (output / "snapshots" / "control-step-00000001.npz").is_file()
    assert (output / "snapshots" / "perturbed-step-00000001.npz").is_file()


def test_source_snapshot_keeps_standard_pseudocode_mirror(monkeypatch, tmp_path):
    tool = _tool()
    mirror = tmp_path / 'docs/pseudocode/tools/warp_050_experiments.py.md'
    mirror.parent.mkdir(parents=True)
    mirror.write_text('Reviewed experiment schedule.\n')
    monkeypatch.setattr(tool, 'SCRATCH_ROOT', tmp_path)
    output = tmp_path / 'run'
    output.mkdir()
    snapshot = tool._source_snapshot(output)
    target = output / 'provenance/docs/pseudocode/tools/warp_050_experiments.py.md'
    assert target.read_bytes() == mirror.read_bytes()
    assert snapshot['files'][str(target.relative_to(output))] == hashlib.sha256(mirror.read_bytes()).hexdigest()
