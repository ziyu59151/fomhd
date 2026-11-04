"""Behavioral checks for offline numerical-evidence transformations."""
import math
import json

import numpy as np
import pytest

from tools.plots.plot_warp_verification import (
    collect, main, positive_frequencies_hz, refinement, require_positive,
)


def test_eigenfrequencies_use_positive_branch_and_si_hertz():
    values = 2j * math.pi * np.array([-7.0, 3.0, -3.0, 7.0])
    np.testing.assert_allclose(positive_frequencies_hz(values), [3.0, 7.0])


@pytest.mark.parametrize("values", [
    [1j, -1j, complex(float("nan"), 0), -3j],
    [1j, -1j, 0j, 0j],
])
def test_eigenfrequencies_reject_incomplete_or_nonfinite_spectrum(values):
    with pytest.raises(ValueError):
        positive_frequencies_hz(np.array(values))


def test_refinement_uses_dt_not_point_index():
    rows = [{"dt_s": 0.25, "error": 0.0625, "final_time_s": 1.0},
            {"dt_s": 1.0, "error": 1.0, "final_time_s": 1.0},
            {"dt_s": 0.5, "error": 0.25, "final_time_s": 1.0}]
    result = refinement(rows)
    assert result["dt_s"] == [1.0, 0.5, 0.25]
    assert result["orders"] == pytest.approx([2.0, 2.0])


@pytest.mark.parametrize("duplicate,horizon", [(True, 1.0), (False, 2.0)])
def test_refinement_rejects_duplicate_steps_or_changed_horizon(duplicate, horizon):
    rows = [{"dt_s": 1.0, "error": 1.0, "final_time_s": 1.0},
            {"dt_s": 1.0 if duplicate else 0.5, "error": 0.25,
             "final_time_s": horizon}]
    with pytest.raises(ValueError):
        refinement(rows)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -1.0, 0.0])
def test_log_values_are_not_silently_clipped(value):
    with pytest.raises(ValueError):
        require_positive(value)


def test_existing_output_refused_before_input_read(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("unchanged")
    with pytest.raises(FileExistsError):
        main(["--massless", "missing.json", "--mms", "missing.json",
              "--output", str(tmp_path)])
    assert marker.read_text() == "unchanged"


def test_collected_data_preserves_undrawn_mms_negative_control(tmp_path):
    np.savez(tmp_path / "hall.npz", production_eigenvalues=np.array([1j, -1j, 2j, -2j]))
    cases = [{"case": "hall", "kind": "controlled_local_parallel_hall", "status": "pass",
              "raw_file": "hall.npz", "kdi": 1.0, "omega_minus": 1.0, "omega_plus": 2.0}]
    mms = {"device": "cuda:0", "passed": True, "status": "PASS", "results": {},
           "omitted_ion_pressure_source_control": {"case": "omitted_ion_pressure_source",
               "detected": True, "passed": True, "physical_gates_passed": True}}
    for method in ("midpoint", "rk4"):
        mms["results"][method] = {"device": "cuda:0", "passed": True, "cases": []}
        for dt in (1.0, 0.5):
            cases.append({"case": f"{method}{dt}", "kind": "real_time_source_none", "status": "pass",
                          "integrator": method, "dt_s": dt, "continuum_relative_error": dt**2,
                          "final_time_s": 1.0})
            mms["results"][method]["cases"].append({"dt_s": dt, "duration_s": 1.0,
                                                    "errors": {"combined": dt**2}})
    for nodes in (4, 8):
        def norms(unit):
            return {region: {"available": True, "unit": unit, "entities": 4,
                             "weighted_relative_l2": 0.1}
                    for region in ("all", "inner", "wall")}
        cases.append({"case": f"recon{nodes}", "kind": "closed_qhfe_reconstruction",
                      "status": "measured", "nodes": nodes,
                      "velocity_norms": norms("m/s"), "theta_norms": norms("1/s")})
        cases.append({"case": f"pressure{nodes}", "kind": "closed_rectangle_operator_consistency",
                      "status": "measured", "nodes": nodes, "compressible": True,
                      "pressure_norms": norms("Pa/m")})
    massless = {"backend_requested": "cpu", "scientific_acceptance": "not_determined",
                "cases": cases}
    massless_path, mms_path = tmp_path / "massless.json", tmp_path / "mms.json"
    massless_path.write_text(json.dumps(massless))
    mms_path.write_text(json.dumps(mms))
    data = collect(massless_path, mms_path)
    assert data["mms_status"]["negative_control"] == mms["omitted_ion_pressure_source_control"]
    assert data["mms_status"]["groups"] == {"midpoint": True, "rk4": True}
    assert len(data["case_statuses"]) == len(cases)
    assert len(data["input_sha256"]) == 3
    mms["omitted_ion_pressure_source_control"]["detected"] = False
    mms_path.write_text(json.dumps(mms))
    with pytest.raises(ValueError, match="negative control"):
        collect(massless_path, mms_path)
