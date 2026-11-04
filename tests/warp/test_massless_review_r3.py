"""Regression tests for round-3 wall-work dimensional scaling."""

from __future__ import annotations

import numpy as np
import pytest

import fomhd.massless as massless
from fomhd.core import inverse
from fomhd.massless_checks import _stage_residuals

from test_massless_support import context, stage_tag, uniform_state


def _record(stage, name):
    return next(
        item
        for item in stage.checks
        if item.name == name
        and item.population == ""
        and item.region == "all"
        and str(item.norm) == "value"
    )


def _expected_wall_work_scale(stage, context, species):
    reaction = inverse(
        stage.cache.wall_reaction,
        context.op.Ne,
        context,
        True,
        context.p.ref.electric,
    ).reshape(context.N, context.op.Ne)
    flux = np.asarray(stage.fluid[species].flux).reshape(
        context.N, context.op.Ne
    )
    charge = context.p.species[species].charge
    edge_weights = np.real(np.asarray(context.op.M1))
    return float(
        np.sum(
            np.abs(charge * flux * reaction) * edge_weights[None, :]
        )
        * (2.0 * np.pi / context.N)
    )


def test_wall_work_reference_uses_actual_edge_pairing_in_watts():
    c = context(hard_diagnostics=False, nodes_r=6, nodes_z=6)
    state = uniform_state(c)
    stage = massless.evaluate_stage(state, stage_tag(state), c)
    expected = _expected_wall_work_scale(stage, c, 1)
    record = _record(stage, "wall_field_reaction_work_electron")
    assert record.unit == "W"
    assert abs(record.reference - expected) <= (
        2.0e-11 * max(abs(expected), 1.0e-300)
    )
    configured = c.p.budgets["wall_field_reaction_work_electron.value"]
    expected_threshold = max(
        float(
            stage.residuals[
                "wall_field_reaction_arithmetic_bound_electron"
            ][0].real
        ),
        configured.atol + configured.rtol * configured.reference,
    )
    assert record.threshold == pytest.approx(expected_threshold)


def test_wall_work_scale_zero_rescaling_and_nonzero_injection_controls():
    c = context(hard_diagnostics=False, nodes_r=6, nodes_z=6)
    state = uniform_state(c)
    stage = massless.evaluate_stage(state, stage_tag(state), c)
    original = stage.cache.wall_reaction.copy()
    base = _stage_residuals(state, stage, c)
    base_work = float(base["wall_field_reaction_work_electron"][0].real)
    base_scale = float(base["wall_field_reaction_scale_electron"][0].real)

    stage.cache.wall_reaction = original * 0.0
    zero = _stage_residuals(state, stage, c)
    assert zero["wall_field_reaction_work_electron"][0].real == 0.0
    assert zero["wall_field_reaction_scale_electron"][0].real == 0.0

    stage.cache.wall_reaction = original * 3.0
    scaled = _stage_residuals(state, stage, c)
    assert scaled["wall_field_reaction_work_electron"][0].real == pytest.approx(
        3.0 * base_work
    )
    assert scaled["wall_field_reaction_scale_electron"][0].real == pytest.approx(
        3.0 * base_scale
    )

    stage.cache.wall_reaction = original + 0.1 * np.ones_like(original)
    injected = _stage_residuals(state, stage, c)
    injected_work = float(
        injected["wall_field_reaction_work_electron"][0].real
    )
    injected_scale = float(
        injected["wall_field_reaction_arithmetic_bound_electron"][0].real
    )
    assert abs(injected_work) > injected_scale
    stage.cache.wall_reaction = original
