"""Behavioral checks for the offline spatial and mode evidence plot tool."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.plots.plot_spatial_modes import (
    collect,
    load_checked_npz,
    main,
    momentum_history,
    validate_mode_case,
)


REAL_ROOT = Path("/home/lxzeng/runs/fomhd/a0-spatial-modes__20261105T055727Z")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def test_spatial_npz_hash_mismatch_is_rejected(tmp_path):
    path = tmp_path / "spatial.npz"
    np.savez(path, finite=np.array([1.0, 2.0]))
    original = path.read_bytes()
    expected = hashlib.sha256(original).hexdigest()
    path.write_bytes(original + b"changed")

    with pytest.raises(ValueError, match="SHA256"):
        load_checked_npz(path, expected)


def test_collect_retains_failed_case_and_missing_run(tmp_path):
    _write_json(tmp_path / "static-full" / "report.json", {
        "scope": "test",
        "device": "cuda",
        "mode": "static",
        "slurm_job_id": "1",
        "hashes": {},
        "cases": [{
            "name": "rectangle-n24",
            "nodes": 24,
            "shape": "rectangle",
            "status": "failed",
            "failure": {"code": "Memory", "message": "intentional test failure"},
        }],
    })
    _write_json(tmp_path / "modes-n33" / "report.json", {
        "status": "failed",
        "scope": "test",
        "configuration": {"points": 33, "cycles": 8.0},
        "cases": [],
        "failed_case": "rk4-n1280-a1e-07",
        "failure": {"code": "Backend", "message": "intentional backend failure"},
        "passed": False,
    })

    data = collect(tmp_path)

    failed = [
        row for row in data["ledger"]["cases"]
        if row["run"] == "static-full" and row["case"] == "rectangle-n24"
    ]
    assert len(failed) == 1
    assert failed[0]["state"] == "failed"
    assert failed[0]["failure"]["code"] == "Memory"
    quiet = data["ledger"]["runs"]["quiet-full"]
    assert quiet["state"] == "unavailable"
    assert "report missing" in quiet["reason"]
    mode_failure = [
        row for row in data["ledger"]["cases"]
        if row["run"] == "modes-n33"
        and row["case"] == "rk4-n1280-a1e-07"
    ]
    assert len(mode_failure) == 1
    assert mode_failure[0]["state"] == "failed"
    assert mode_failure[0]["failure"]["code"] == "Backend"
    assert len(data["inputs"]) == 2


def test_mode_cadence_must_match_steps():
    case = {
        "case_id": "bad-cadence",
        "steps": 2,
        "accepted_steps": 2,
        "dt_s": 0.5,
        "target_time_s": 1.0,
        "accepted_time_s": 1.0,
        "cycles": 1.0,
        "points": 17,
        "hard_pass": True,
        "passed": True,
        "mode_fit": {"sample_count": 3, "interval_count": 2},
    }
    arrays = {
        "time_s": np.array([0.0, 0.5, 1.0, 1.5]),
        "mode": np.zeros((4, 7), dtype=np.complex128),
    }

    with pytest.raises(ValueError, match="cadence"):
        validate_mode_case(case, arrays)


def test_dynamic_cumulative_identity_uses_signed_interval_sum():
    case = {
        "name": "rectangle-n6",
        "status": "completed",
        "requested_steps": 3,
        "accepted_steps": 3,
        "dt_s": 1.0,
        "horizon_s": 3.0,
        "signed_cumulative_momentum_x_defect": 1.25,
    }
    arrays = {
        "time_s": np.array([0.0, 1.0, 2.0, 3.0]),
        "momentum": np.zeros((4, 3)),
        "momentum_x_interval": np.array([1.0, -0.25, 0.5]),
    }

    history = momentum_history(case, arrays)

    np.testing.assert_array_equal(history["time_s"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        history["cumulative_momentum_x_kg_m_per_s"], [1.0, 0.75, 1.25]
    )
    assert history["endpoint_momentum_x_kg_m_per_s"] == 1.25

    case["signed_cumulative_momentum_x_defect"] = 1.0
    with pytest.raises(ValueError, match="cumulative"):
        momentum_history(case, arrays)


def test_dynamic_wrong_physical_cadence_is_rejected():
    case = dict(name="timing", status="completed", requested_steps=3,
                accepted_steps=3, dt_s=1.0, horizon_s=3.0,
                signed_cumulative_momentum_x_defect=0.0)
    arrays = dict(time_s=np.arange(4) * 10.0, momentum=np.zeros((4, 3)),
                  momentum_x_interval=np.zeros(3))
    with pytest.raises(ValueError, match="cadence|horizon"):
        momentum_history(case, arrays)


def test_rejected_partial_momentum_is_retained():
    case = dict(name="partial", status="failed", requested_steps=3,
                accepted_steps=2, dt_s=1.0, horizon_s=3.0,
                signed_cumulative_momentum_x_defect=0.3)
    arrays = dict(time_s=np.arange(3), momentum=np.zeros((3, 3)),
                  momentum_x_interval=np.array([0.1, 0.2]))
    history = momentum_history(case, arrays)
    assert history["time_s"][-1] == 2.0


def _mode_evidence():
    steps, omega, target = 640, 16 * np.pi, 1.0
    times = np.linspace(0, target, steps + 1)
    mode = np.ones((steps + 1, 7), dtype=complex) * 1e-7
    mode[:, 3] *= np.exp(1j * omega * times)
    case = dict(case_id="eight-cycles", steps=steps, accepted_steps=steps,
                dt_s=target / steps, target_time_s=target,
                accepted_time_s=target, cycles=8.0, hard_pass=True, passed=True,
                mode_fit=dict(available=True, sample_count=steps + 1,
                    interval_count=steps, complete_measured_cycles=8.0,
                    samples_per_cycle=80.0,
                    measured=dict(angular_frequency_rad_s=omega,
                        frequency_hz=8.0, growth_rate_per_s=0.0,
                        phase_residual_rms_rad=0.0),
                    analytic=dict(angular_frequency_rad_s=omega,
                                  frequency_hz=8.0)))
    return case, dict(time_s=times, mode=mode)


@pytest.mark.parametrize("corruption", ["constant_history", "cycles", "growth", "phase_ripple"])
def test_mode_measurement_must_match_loaded_history(corruption):
    case, arrays = _mode_evidence()
    validate_mode_case(case, arrays)
    if corruption == "constant_history":
        arrays["mode"][:, 3] = 1e-7
    elif corruption == "cycles":
        case["cycles"] = 4.0
    elif corruption == "growth":
        case["mode_fit"]["measured"]["growth_rate_per_s"] = 0.5
    else:
        phase = 0.05 * (np.cos(2 * np.pi * arrays["time_s"]) - 1)
        arrays["mode"][:, 3] *= np.exp(1j * phase)
    with pytest.raises(ValueError, match="history|cycles|fit"):
        validate_mode_case(case, arrays)


def test_missing_spatial_array_still_renders_failure(tmp_path):
    from tools.plots.plot_spatial_modes import _render_geometry
    import matplotlib
    matplotlib.use("Agg")
    _write_json(tmp_path / "spatial-pilot" / "report.json", {
        "device": "cuda", "cases": [{"label": "rectangle_n8_wall",
            "status": "measured", "full_stage_hard_checks_pass": True,
            "arrays_file": "missing.npz", "shape": "rectangle", "nodes": 8}],
    })
    data = collect(tmp_path)
    output = tmp_path / "figures"
    output.mkdir()
    _render_geometry(data, output)
    assert (output / "fixed-geometry.png").is_file()
    assert any(row["state"] == "unavailable" for row in data["ledger"]["cases"])


def test_existing_output_refused_before_collection(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("unchanged")

    with pytest.raises(FileExistsError):
        main(["--root", str(tmp_path / "missing"), "--output", str(tmp_path)])

    assert marker.read_text() == "unchanged"


def test_real_archive_collection_smoke():
    if not (REAL_ROOT / "spatial-pilot" / "report.json").exists():
        pytest.skip("parent evidence root is not present")

    data = collect(REAL_ROOT)

    assert data["spatial"]["cases"]
    assert data["modes"]["available"]
    assert data["ledger"]["runs"]["spatial-pilot"]["state"] == "measured"
    assert all(len(value) == 64 for value in data["inputs"].values())
    case_names = {
        (row["run"], row["case"]) for row in data["ledger"]["cases"]
    }
    assert ("spatial-pilot", "rectangle_n8_compact") in case_names
    assert ("modes-n17", "rk4-n640-a1e-07") in case_names
