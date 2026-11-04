"""Regression tests for round-4 wall-work backward-error bounds."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import fomhd.massless as massless
from fomhd.core import inverse
from fomhd.massless_checks import check_stage, _stage_residuals

from test_massless_support import context, stage_tag, uniform_state


def _record(stage, name):
    return next(
        item
        for item in stage.checks
        if item.name == name
        and item.region == "all"
        and str(item.norm) == "value"
    )


def _operator_work_scale(state, stage, context, species):
    if species == "current":
        flux_hat = np.asarray(stage.current)
    else:
        flux_hat = np.asarray(stage.fluid[species].flux_hat)
    reaction = np.asarray(stage.cache.wall_reaction).reshape(
        2 * context.p.K + 1, context.op.Ne
    )
    m1 = np.real(np.asarray(context.op.M1))
    return float(
        2.0
        * np.pi
        * np.sum(
            np.abs(reaction)
            * np.abs(flux_hat.reshape(reaction.shape))
            * m1[None, :]
        )
    )


@pytest.mark.parametrize("field_multiplier", (1.0, 100.0))
def test_upstream_arithmetic_bound_survives_field_rescaling(field_multiplier):
    c = context(hard_diagnostics=False, nodes_r=6, nodes_z=6)
    state = uniform_state(c)
    state.B *= field_multiplier
    stage = massless.evaluate_stage(state, stage_tag(state), c)
    for name in ("electron", "current"):
        item = _record(stage, f"wall_field_reaction_work_{name}")
        assert item.arithmetic_bound >= item.upstream_bound
        assert item.threshold == max(
            item.configured_threshold, item.arithmetic_bound
        )
        assert abs(item.raw) <= item.arithmetic_bound
        assert item.passed
        assert item.sample_work is not None
        assert item.transform_difference == pytest.approx(
            item.sample_work - item.modal_work
        )


def test_volume_rescaling_multiplies_work_and_scale_once():
    c = context(hard_diagnostics=False, nodes_r=6, nodes_z=6)
    state = uniform_state(c)
    stage = massless.evaluate_stage(state, stage_tag(state), c)
    base = _stage_residuals(state, stage, c)
    factor = 27.0
    scaled_context = copy.copy(c)
    scaled_context.op = copy.copy(c.op)
    scaled_context.op.M1 = c.op.M1 * factor
    scaled_stage = copy.copy(stage)
    scaled_stage.cache = copy.copy(stage.cache)
    scaled_stage.cache.wall_correction = copy.copy(
        stage.cache.wall_correction
    )
    scaled_stage.cache.wall_correction.modes = [
        copy.copy(mode) for mode in stage.cache.wall_correction.modes
    ]
    for mode in scaled_stage.cache.wall_correction.modes:
        mode.multiplier = mode.multiplier * factor
    scaled = _stage_residuals(state, scaled_stage, scaled_context)
    for name in ("electron", "current"):
        for key in (
            f"wall_field_reaction_work_{name}",
            f"wall_field_reaction_scale_{name}",
        ):
            assert scaled[key][0].real == pytest.approx(
                factor * base[key][0].real, rel=2.0e-12, abs=0.0
            )


@pytest.mark.parametrize("field_multiplier", (1.0, 100.0))
def test_injected_wall_work_fails_derived_gate(field_multiplier):
    c = context(hard_diagnostics=False, nodes_r=6, nodes_z=6)
    state = uniform_state(c)
    state.B *= field_multiplier
    stage = massless.evaluate_stage(state, stage_tag(state), c)
    stage.cache.wall_reaction = stage.cache.wall_reaction + 0.1
    stage.residuals.update(_stage_residuals(state, stage, c))
    stage.checks = []
    check_stage(state, stage, c)
    for name in ("electron", "current"):
        item = _record(stage, f"wall_field_reaction_work_{name}")
        assert not item.passed
        assert abs(item.raw) > item.threshold


def test_nonzero_k_wall_work_metrics_and_transform_evidence():
    c = context(K=1, hard_diagnostics=False, nodes_r=6, nodes_z=6)
    state = uniform_state(c)
    stage = massless.evaluate_stage(state, stage_tag(state), c)
    item = _record(stage, "wall_field_reaction_work_electron")
    assert np.isfinite(item.raw)
    assert np.isfinite(item.arithmetic_bound)
    assert np.isfinite(item.transform_difference)
    assert item.transform_difference == pytest.approx(
        item.sample_work - item.modal_work
    )
