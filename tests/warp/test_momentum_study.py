"""Independent sign, metric, and fixed-domain controls for momentum studies."""

import importlib.util
from pathlib import Path

import numpy as np


path = Path(__file__).resolve().parents[2] / "tools/warp_momentum_study.py"
spec = importlib.util.spec_from_file_location("momentum_study", path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_fixed_old_domain_and_flux():
    for n in (3, 5):
        context = study.build_context(n, "rectangle", original=True)
        assert np.isclose(sum(context.geometry.M0) * 2 * np.pi, 2*np.pi*1.2*.36)
        state = study.reference(context)
        actual = (context.op.topology_flux_row @ state.B[context.op.Nb:2*context.op.Nb]).item()
        assert np.isclose(actual.real, study.FIXED_FLUX, rtol=1e-12, atol=0)


def test_complete_stage_has_nonzero_balanced_pressure_traction():
    context = study.build_context(4, "rectangle")
    report = study.static_control(context, "rectangle")
    scale = report["analytic_outward_x_N"]
    assert scale > 0
    assert abs(report["outward_N"][0] / scale - 1) < 1e-10
    assert abs(report["total_rate_N"][0] / scale + 1) < 1e-10
    assert max(abs(x) for x in report["balance_defect_N"]) < 1e-10 * scale


def test_quiet_rk4_transaction_preserves_reference():
    context = study.build_context(3, "rectangle", original=True)
    result, arrays = study.evolve(context, steps=2, horizon=2e-10, quiet=True)
    assert result["status"] == "completed"
    assert result["accepted_steps"] == 2
    assert result["maximum_relative_state_drift"] < 1e-10
    assert len(arrays["time_s"]) == 3
    assert "initial_rhs__mom_e" in arrays


def test_rejected_step_retains_partial_arrays_and_guard_report():
    context = study.build_context(3, "rectangle", original=True)
    result, arrays = study.evolve(context, steps=1)
    assert result["status"] == "failed"
    assert result["failure"]["code"] == "TimeStepGuard"
    assert result["time_step_reports"][-1]["accepted"] is False
    assert any(name.startswith("failure__") for name in arrays)
    assert result["failure"]["stage"]
