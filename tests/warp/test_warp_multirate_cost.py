"""Focused tests for the matched lean synchronous cost reference."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest


SCRATCH_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PROJECT = Path(
    os.environ.get("FOMHD_SOURCE_ROOT", str(SCRATCH_ROOT))
)
TOOL_PATH = SCRATCH_ROOT / "tools/warp_multirate_cost.py"
for path in (
    SOURCE_PROJECT / "src",
    SOURCE_PROJECT / "tests/warp",
    SOURCE_PROJECT / "tools",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("FOMHD_SOURCE_ROOT", str(SOURCE_PROJECT))


def _cost_tool():
    spec = importlib.util.spec_from_file_location(
        "warp_multirate_cost_under_test", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("case", "points", "h", "M"),
    (("periodic", 9, 5.0e-12, 2), ("closed", 8, 1.0e-12, 1)),
)
def test_lean_window_matches_complete_production_rk4(case, points, h, M):
    cost = _cost_tool()
    ion = cost.multirate
    config = {
        "case": case,
        "device": "cpu",
        "method": "C1",
        "reference": "synchronous",
        "points": points,
        "amplitude": 1.0e-4 if case == "periodic" else 0.02,
        "nonuniform": 0.1,
        "h": h,
        "M": M,
        "macros": 1,
        "rtol": 1.0e-9,
        "max_iterations": 4,
        "seed_mode": 1,
        "sentinel": 1.0e-4,
        "endpoint_check": True,
        "warmups": 0,
        "repetitions": 1,
    }
    context, p0, z0, _ = ion.build_case(config)
    production = ion._run_reference(
        context,
        p0,
        z0,
        h=h,
        M=M,
        macros=1,
        coupling="synchronous",
        rtol=config["rtol"],
        max_iterations=config["max_iterations"],
    )
    lean = cost.lean_run(context, p0, z0, H=h * M, M=M, macros=1)

    scales = cost.research._physical_scales(context)
    p_raw, p_error = ion._relative_error(lean.p, production.p, scales.p)
    z_raw, z_error = ion._relative_error(lean.z, production.z, scales.z)
    assert p_error < 2.0e-10, (p_raw, p_error)
    assert z_error < 2.0e-10, (z_raw, z_error)

    assert len(lean.stages) == 4 * M
    assert lean.counters["full_hydro_eval"] == 4 * M
    assert lean.counters["cheap_D"] == 4 * M
    assert lean.counters["cheap_L"] == 4 * M
    assert lean.counters["field_rhs"] == 4 * M + 1
    assert lean.counters["endpoint_check"] == 1
    assert {
        (stage.micro_step, stage.rk_stage) for stage in lean.stages
    } == {
        (micro, rk)
        for micro in range(M)
        for rk in range(4)
    }
    assert all(
        value is not None
        for stage in lean.stages
        for value in (
            stage.density_rate,
            stage.momentum_rate,
            stage.pressure_rate,
            stage.lorentz_rate,
            stage.z_rhs,
            stage.electric,
            stage.magnetic,
        )
    )


def test_cost_run_keeps_protocol_and_evidence_separate(tmp_path):
    cost = _cost_tool()
    ion = cost.multirate
    config = {
        "case": "periodic",
        "device": "cpu",
        "method": "C1",
        "reference": "synchronous",
        "points": 9,
        "amplitude": 1.0e-4,
        "nonuniform": 0.1,
        "h": 5.0e-12,
        "M": 2,
        "macros": 2,
        "rtol": 1.0e-9,
        "max_iterations": 4,
        "seed_mode": 1,
        "sentinel": 1.0e-4,
        "endpoint_check": True,
        "warmups": 0,
        "repetitions": 1,
    }
    case_dir = tmp_path / "candidate"
    ion.run(config, case_dir)
    source_bytes = (case_dir / "input.json").read_bytes()
    output = tmp_path / "comparison"

    result = cost.run(
        input_case_directory=case_dir,
        device="cpu",
        output=output,
        warmups=0,
        repetitions=1,
    )

    assert (case_dir / "input.json").read_bytes() == source_bytes
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "completed"
    assert report["source_input"]["sha256"] == cost._sha256(case_dir / "input.json")
    assert report["protocol"]["horizon_unchanged"] is True
    assert report["protocol"]["actual_h_s"] == config["h"]
    assert report["protocol"]["actual_M"] == config["M"]
    assert report["protocol"]["actual_macros"] == config["macros"]
    assert report["matched_cost"]["speedup_claim"] is False
    assert report["existing_candidate_vs_production"]["scope"].startswith(
        "historical input-case"
    )
    assert report["same_checks"]["candidate_vs_lean"]["p_normalized"] < 1.0e-8
    assert report["counters"]["lean"]["full_hydro_eval"] == (
        4 * config["M"] * config["macros"]
    )
    assert report["stage_invariants"]["all_four_rk_identities"] is True
    assert report["stage_invariants"]["counts_match"] is True
    assert report["stage_invariants"]["stage_count"] == 4 * config["M"] * config["macros"]

    with np.load(output / "arrays.npz") as arrays:
        assert arrays["lean_stage_p"].shape[0] == 4 * config["M"] * config["macros"]
        assert arrays["lean_stage_z"].shape == arrays["lean_stage_p"].shape
        assert arrays["lean_stage_density_rate"].shape == arrays["lean_stage_p"].shape
        assert arrays["lean_stage_rk"].tolist() == [0, 1, 2, 3] * config["M"] * config["macros"]
        assert arrays["candidate_p"].shape == arrays["lean_p"].shape
        assert arrays["candidate_z"].shape == arrays["lean_z"].shape
        assert arrays["lean_stage_p"].nbytes > 0
        assert "source_input.json" in report["files"]


def test_cuda_run_requires_owned_slurm_allocation(tmp_path, monkeypatch):
    cost = _cost_tool()
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="Slurm allocation"):
        cost.run(
            input_case_directory=tmp_path / "missing",
            device="cuda:0",
            output=tmp_path / "output",
            warmups=1,
            repetitions=3,
        )
    assert not (tmp_path / "output").exists()
