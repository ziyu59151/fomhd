"""Catch incorrect finite-mode polarization, field adapters and time response."""

import importlib
import math

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


@pytest.mark.parametrize("integrator,ratio", [("midpoint", .3), ("rk4", .09)])
def test_longitudinal_time_refinement_reduces_continuum_error(integrator, ratio):
    coarse = module().run_case(integrator, 32)["summary"]
    fine = module().run_case(integrator, 64)["summary"]
    assert coarse["passed"] and fine["passed"]
    assert coarse["target_time_s"] == fine["target_time_s"]
    assert fine["continuum_relative_error"] < ratio*coarse["continuum_relative_error"]


@pytest.mark.parametrize("kwargs", [{"integrator":"implicit", "steps":4},
                                    {"integrator":"midpoint", "steps":0},
                                    {"integrator":"midpoint", "steps":4, "amplitude":0.}])
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
    import json

    output = tmp_path/"scan"
    report = module().run_scan(output, steps=[4])
    assert report["passed"] and len(report["cases"]) == 4
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


@pytest.mark.parametrize("integrator", ["midpoint", "rk4"])
def test_longitudinal_rejected_attempt_retains_structured_dt_warning(monkeypatch, integrator):
    tool = module()
    original = tool.build_context

    def inconsistent_poisson(device):
        context, mode = original(device)
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
