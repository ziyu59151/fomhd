"""Catch wrong normalization and silently accepted incomplete plot evidence."""
import importlib
import json
import math

import numpy as np
import pytest


def fixture_report(root):
    cases = []
    for method in ("midpoint", "rk4"):
        name = method + "-n4"
        history = np.zeros((5, 7), dtype=complex)
        history[:, 3] = 1e-7*np.array([1, 1j, -1, -1j, 1])
        np.savez(root/(name+".npz"), time_s=np.arange(5)*math.pi/2, mode=history)
        cases.append(dict(case_id=name, integrator=method, device="cpu", steps=4,
                          accepted_steps=4, dt_s=math.pi/2, omega_rad_s=1.,
                          target_time_s=2*math.pi, accepted_time_s=2*math.pi,
                          relative_amplitude=1e-7, passed=True, hard_pass=True,
                          amplitude_ratio=1., polynomial_amplitude_ratio=1.,
                          continuum_relative_error=1e-8))
        half = dict(cases[-1], case_id=name+"-half", relative_amplitude=5e-8)
        np.savez(root/(half["case_id"]+".npz"), time_s=np.arange(5)*math.pi/2,
                 mode=history*.5)
        cases.append(half)
    report = dict(status="completed", passed=True, cases=cases,
                  amplitude_sensitivity=[dict(integrator=method, passed=True,
                                              relative_amplitude_change=0.)
                                         for method in ("midpoint", "rk4")])
    path = root/"report.json"
    path.write_text(json.dumps(report))
    return path, report


def test_plot_normalizes_electron_mode_and_preserves_all_evidence(tmp_path):
    path, report = fixture_report(tmp_path)
    data = importlib.import_module("tools.plots.plot_warp_longitudinal").collect(path)
    assert data["cases"] == report["cases"]
    assert data["amplitude_sensitivity"] == report["amplitude_sensitivity"]
    assert len(data["input_sha256"]) == 5
    np.testing.assert_allclose(data["histories"]["midpoint"]["cycles"], [0, .25, .5, .75, 1])
    np.testing.assert_allclose(data["histories"]["rk4"]["electron_real"], [1, 0, -1, 0, 1])


@pytest.mark.parametrize("damage", ["failed", "horizon", "history"])
def test_plot_rejects_failed_or_mismatched_evidence(tmp_path, damage):
    path, report = fixture_report(tmp_path)
    if damage == "failed":
        report["cases"][0]["passed"] = False
    elif damage == "horizon":
        report["cases"][0]["target_time_s"] *= 2
    else:
        np.savez(tmp_path/"midpoint-n4.npz", time_s=np.arange(4), mode=np.zeros((4, 7)))
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        importlib.import_module("tools.plots.plot_warp_longitudinal").collect(path)


def test_longitudinal_plot_refuses_output_before_reading_input(tmp_path):
    marker = tmp_path/"keep.txt"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        importlib.import_module("tools.plots.plot_warp_longitudinal").main(
            ["--input", "missing.json", "--output", str(tmp_path)])
    assert marker.read_text() == "original"


@pytest.mark.parametrize("damage", ["empty", "missing", "duplicate", "orphan", "value"])
def test_plot_requires_complete_matching_amplitude_controls(tmp_path, damage):
    path, report = fixture_report(tmp_path)
    if damage == "empty":
        report["amplitude_sensitivity"] = []
    elif damage == "missing":
        report["amplitude_sensitivity"].pop()
    elif damage == "duplicate":
        report["amplitude_sensitivity"][1] = report["amplitude_sensitivity"][0]
    elif damage == "orphan":
        report["cases"] = [row for row in report["cases"] if row["relative_amplitude"] == 1e-7]
    else:
        report["amplitude_sensitivity"][0]["relative_amplitude_change"] = .5
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        importlib.import_module("tools.plots.plot_warp_longitudinal").collect(path)
