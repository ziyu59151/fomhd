"""Formal old-staircase initialization integration for the frozen circle."""

from __future__ import annotations

import tomllib

import pytest
from tools.circular_equilibrium import build_gs_equilibrium
from tools.prepare_staircase_equilibrium import build_and_run


def test_grid8_case_reaches_formal_initialize_and_writes_full_rhs(tmp_path):
    model = build_gs_equilibrium(
        R0=3.0,
        a=0.5,
        F0=6.0,
        n0=2.0e19,
        p_edge=1000.0,
        pprime=1.0e4,
        alpha=0.6,
        degree=8,
    )

    result = build_and_run(tmp_path, model, 8)

    assert result["status"] == "completed"
    assert result["load_and_initialize"] == "passed"
    assert result["checks"]["hard_failures"] == 0
    assert result["canonical_changes"]["B_R"]["max_abs_change"] == 0.0
    assert result["canonical_changes"]["mom_e"]["max_abs_change"] > 0.0
    assert result["initial_rhs_maxima"]["p_perp_e"] > 0.0
    assert (tmp_path / "g8" / "initial.npz").is_file()
    assert (tmp_path / "g8" / "initial_rhs.npz").is_file()


def test_shifted_radius_toml_roundtrip_uses_actual_model_grid(tmp_path):
    model = build_gs_equilibrium(
        R0=4.2,
        a=0.35,
        F0=6.0,
        n0=2.0e19,
        p_edge=1000.0,
        pprime=1.0e4,
        alpha=0.6,
        degree=8,
    )

    result = build_and_run(tmp_path, model, 8)
    document = tomllib.loads((tmp_path / "g8" / "case.toml").read_text())
    spacing = 2.0 * model.a / 8

    assert result["status"] == "completed"
    assert document["grid"]["R0"] == pytest.approx(model.R0 - model.a + spacing / 2.0)
    assert document["grid"]["Z0"] == pytest.approx(-model.a + spacing / 2.0)
