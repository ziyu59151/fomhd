"""Behavioral checks for the offline Tokamak CAE response gallery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.plots.plot_tokamak_cae import (
    main,
    mode_history,
    physical_at_phi,
    select_default_cases,
)


REAL_ROOT = Path(
    "/home/lxzeng/runs/fomhd/a0-tokamak-cae__20261105T154538Z"
)


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def _fixture(root: Path, *, status: str = "budget_stopped") -> Path:
    case = root / "circle4-n0-test-r2"
    case.mkdir(parents=True)
    times = np.array([0.0, 1.0e-9, 2.0e-9])
    control_energy = [
        {
            "time_s": float(t),
            "step": index,
            "energy": {
                "ion_J": 10.0 + index,
                "electron_J": 20.0 + index,
                "field_J": 1000.0 + index,
                "total_J": 1030.0 + 3 * index,
                "balance_J": 0.0,
                "relative_total_change": index * 1.0e-4,
                "outward_J": 0.0,
                "source_J": 0.0,
            },
        }
        for index, t in enumerate(times)
    ]
    perturbed_energy = [
        {
            "time_s": float(t),
            "step": index,
            "energy": {
                "ion_J": 11.0 + 2 * index,
                "electron_J": 21.0,
                "field_J": 1001.0 - index,
                "total_J": 1033.0 + index,
                "balance_J": float((index - 1) * 1.0e-8),
                "relative_total_change": (index - 1) * 1.0e-9,
                "outward_J": index * 1.0e-12,
                "source_J": 0.0,
            },
        }
        for index, t in enumerate(times)
    ]
    _json(case / "control-history.json", control_energy)
    _json(case / "perturbed-history.json", perturbed_energy)

    density = np.full((3, 5, 3), 100.0, dtype=np.complex128)
    density[:, 1:, :] = 999.0
    control_density = np.full_like(density, 100.0)
    velocity = np.zeros((3, 5, 3, 3), dtype=np.complex128)
    velocity[:, :, 0, :] = 1.0 + np.arange(3)
    velocity[:, :, 2, :] = 0.5
    bn = np.zeros((3, 5, 3, 3), dtype=np.complex128)
    bn[:, :, 1, :] = 2.0
    np.savez(
        case / "control-signals.npz",
        time_s=times,
        density=control_density.reshape(3, 5 * 3),
        velocity=np.zeros_like(velocity).reshape(3, 5 * 3 * 3),
        BN=np.zeros_like(bn).reshape(3, 5 * 3 * 3),
        B=np.zeros_like(bn),
    )
    np.savez(
        case / "perturbed-signals.npz",
        time_s=times,
        density=density.reshape(3, 5 * 3),
        velocity=velocity.reshape(3, 5 * 3 * 3),
        BN=bn.reshape(3, 5 * 3 * 3),
        B=bn,
    )
    delta_n = np.zeros((3, 5, 3), dtype=np.float64)
    delta_n[0, :, 0] = 2.0
    delta_n[1, :, 0] = 3.0
    delta_n[2, :, 2] = 5.0
    np.savez(
        case / "matched-response.npz",
        time_s=times,
        delta_B=np.ones((3, 3, 3), dtype=np.complex128),
        delta_n=delta_n,
        delta_velocity=np.ones((3, 5, 3, 3), dtype=np.complex128),
        parallel_B=np.ones((3, 3)),
        probe_cells=np.array([0, 2]),
        perturbation_magnetic_J=np.array([1.0, 2.0, 4.0]),
        perturbation_ion_kinetic_J=np.array([0.5, 0.7, 1.0]),
    )
    np.savez(case / "perturbation_mapping.npz",
             raw_B=np.ones(7, dtype=np.complex128),
             mapped_B=np.ones(7, dtype=np.complex128),
             raw_velocity=np.ones((3, 3), dtype=np.complex128),
             mapped_velocity=np.ones((3, 3), dtype=np.complex128))
    np.savez(case / "input_mapping.npz",
             R=np.array([1.5, 1.7, 1.9]),
             Z=np.array([0.0, 0.1, -0.1]),
             polygon=np.array([[1.0, -1.0], [2.0, -1.0],
                               [2.0, 1.0], [1.0, 1.0]]),
             background_raw_B=np.ones(7, dtype=np.complex128),
             background_mapped_B=np.ones(7, dtype=np.complex128))
    (case / "case.toml").write_text(
        'kind = "circle"\n'
        "nodes = 4\n"
        "nphi = 0\n"
        "K = 0\n"
        "Nphi = 5\n"
    )
    _json(case / "report.json", {
        "status": status,
        "device": "cpu",
        "failure": {"code": "BudgetStop", "message": "synthetic short prefix"},
        "case": {
            "kind": "circle",
            "nodes": 4,
            "K": 0,
            "Nphi": 5,
            "active_cells": 3,
            "profile_assumption": "uniform synthetic plasma",
        },
        "source_hashes": {"tools/warp_tokamak_cae.py": "a" * 64},
        "mode_fits": [{
            "cell": 2,
            "fit": {
                "available": False,
                "reason": "fewer than nine samples",
                "frequency_hz": 987654321.0,
            },
        }],
    })
    return case


def test_default_selection_uses_latest_semantic_revision(tmp_path):
    for name in (
        "circle12-n0-cpu-pilot-r1",
        "circle12-n0-cpu-pilot-r2",
        "circle12-n1-cpu-pilot-r3",
        "east12-n0-cpu-smoke-r1",
        "east12-n0-cpu-smoke-r2",
        "circle-cpu-smoke-r5",
        "spectrum-circle12-k1-n1-r1",
    ):
        (tmp_path / name).mkdir()
        _json(tmp_path / name / "report.json", {"status": "failed"})

    assert select_default_cases(tmp_path) == [
        "circle12-n0-cpu-pilot-r2",
        "circle12-n1-cpu-pilot-r3",
        "east12-n0-cpu-smoke-r2",
    ]


def test_mode_history_retains_the_nonzero_zero_fourier_mean():
    samples = np.zeros((2, 5, 2), dtype=np.complex128)
    samples[:, 0, :] = 2.0
    samples[:, 1:, :] = 100.0
    actual = mode_history(samples, 0)
    np.testing.assert_array_equal(actual, np.full((2, 2), 80.4))


def test_mode_history_one_retains_complex_coefficient():
    samples = np.exp(1j * np.arange(5)[None, :, None] * 2 * np.pi / 5)
    samples = np.broadcast_to(samples, (3, 5, 2)).copy()
    actual = mode_history(samples, 1)
    np.testing.assert_allclose(actual, np.ones((3, 2), dtype=complex))


def test_n0_once_and_n1_half_coefficient_give_same_phi0_physics():
    amplitude = 5.0
    n0 = physical_at_phi(amplitude, 0, phi=0.0)
    n1 = physical_at_phi(amplitude / 2, 1, phi=0.0)
    np.testing.assert_allclose(n0, amplitude)
    np.testing.assert_allclose(n1, amplitude)
    np.testing.assert_allclose(n0, n1)


def test_synthetic_cli_preserves_stop_and_hashes_every_input(tmp_path):
    run_root = tmp_path / "runs"
    case = _fixture(run_root)
    output = tmp_path / "figures"

    assert main(["--run-root", str(run_root), "--output", str(output),
                 "--cases", case.name]) == 0

    receipt = json.loads((output / "receipts/report.json").read_text())
    assert receipt["gallery_id"] == "cae-20261106"
    assert receipt["scientific_acceptance"] is False
    assert receipt["cases"][0]["solver_status"] == "budget_stopped"
    assert receipt["cases"][0]["compression"]["time_s"] == pytest.approx(2e-9)
    assert receipt["cases"][0]["compression"]["probe_cell"] == 2
    assert receipt["cases"][0]["compression"]["scale_m-3"] == 100.0
    assert "scale_kg_m-3" not in receipt["cases"][0]["compression"]
    assert receipt["cases"][0]["compression"]["delta_n_phi0_m-3"] == 5.0
    assert receipt["cases"][0]["frequency"]["available"] is False
    assert {
        item["role"] for item in receipt["inputs"]
        if item["path"].endswith("report.json")
    } == {"report"}
    read_names = {
        item["path"].rsplit("/", 1)[-1] for item in receipt["inputs"]
    }
    assert read_names == {
        "report.json", "case.toml", "control-history.json",
        "perturbed-history.json", "control-signals.npz",
        "perturbed-signals.npz", "matched-response.npz",
        "perturbation_mapping.npz", "input_mapping.npz",
    }
    for item in receipt["inputs"]:
        path = case / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        assert item["size_bytes"] == path.stat().st_size
    for stem in ("overview", "structure"):
        assert (output / stem / f"{case.name}.png").is_file()
        assert (output / stem / f"{case.name}.svg").is_file()
    fragment = (output / "fragment.html").read_text()
    assert 'id="cae-20261106"' in fragment
    assert "budget_stopped" in fragment
    assert "fewer than nine samples" in fragment
    assert "987654321" not in fragment
    assert "Ion total energy (kinetic+internal)" in fragment
    assert "kg/m^3" not in fragment
    assert "m^-3" in fragment
    assert "A/2" in fragment
    assert "2 Re(c exp(i*n*phi))" in (
        receipt["cases"][0]["field_meanings"]["physical_structure"]
    )
    assert "zero-Fourier" in fragment


def test_missing_explicit_case_gets_visible_placeholders(tmp_path):
    run_root = tmp_path / "runs"
    run_root.mkdir()
    output = tmp_path / "figures"

    assert main(["--run-root", str(run_root), "--output", str(output),
                 "--cases", "missing-case-r1"]) == 0

    receipt = json.loads((output / "receipts/report.json").read_text())
    assert receipt["cases"][0]["solver_status"] == "unavailable"
    assert receipt["cases"][0]["errors"]
    assert (output / "overview/missing-case-r1.png").is_file()
    assert (output / "structure/missing-case-r1.svg").is_file()


def test_existing_output_is_refused_before_input_read(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("unchanged")

    with pytest.raises(FileExistsError):
        main(["--run-root", str(tmp_path / "absent"),
              "--output", str(tmp_path)])

    assert marker.read_text() == "unchanged"


def test_real_short_case_smoke(tmp_path):
    if not (REAL_ROOT / "east12-n0-cpu-smoke-r2/report.json").is_file():
        pytest.skip("real short CPU case is unavailable")
    output = tmp_path / "actual"
    assert main([
        "--run-root", str(REAL_ROOT), "--output", str(output),
        "--cases", "east12-n0-cpu-smoke-r2",
    ]) == 0
    receipt = json.loads((output / "receipts/report.json").read_text())
    case = receipt["cases"][0]
    assert case["solver_status"] == "completed"
    assert case["backend"] == "cpu"
    assert case["mode"] == 0
    assert case["equilibrium_source_sha256"].startswith("df8f073b")
    assert case["compression"]["available"] is True
    assert case["compression"]["time_s"] == pytest.approx(8.318055109186038e-11)
    assert case["compression"]["probe_cell"] == 60
    assert case["compression"]["scale_m-3"] == pytest.approx(2e19)
    assert case["frequency"]["available"] is False
    assert receipt["input_count"] == 9
