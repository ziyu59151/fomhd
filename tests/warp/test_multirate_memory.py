"""Conservative retained-memory accounting tests for A1-WR."""

from __future__ import annotations

import pytest

from fomhd import core
from fomhd.types import Failure
from test_time_predictor import _fake_memory_context


def _wr_context(
    *,
    N=5,
    history_samples=1,
    micro_steps=2,
    max_iterations=2,
    ion_nodes=3,
    walls=(),
):
    context = _fake_memory_context(K=1, walls=list(walls))
    context.N = N
    context._history_samples = history_samples
    context.p.coupling = "multirate_wr"
    context.p.micro_steps = micro_steps
    context.p.waveform_max_iterations = max_iterations
    context.p.ion_nodes = ion_nodes
    context.p.wr_initial = "tangent"
    context.p.wr_relaxation = 1.0
    return context


def test_wr_memory_grows_with_microsteps_outer_sweeps_and_nodes():
    one_sweep = core.predictor_history_entries(
        _wr_context(max_iterations=1)
    )
    two_sweeps = core.predictor_history_entries(
        _wr_context(max_iterations=2)
    )
    four_sweeps = core.predictor_history_entries(
        _wr_context(max_iterations=4)
    )
    more_microsteps = core.predictor_history_entries(
        _wr_context(micro_steps=4, max_iterations=2)
    )
    five_nodes = core.predictor_history_entries(
        _wr_context(ion_nodes=5, max_iterations=2)
    )
    assert 0 < one_sweep < two_sweeps < four_sweeps
    assert core.predictor_history_entries(
        _wr_context(micro_steps=2, max_iterations=4)
    ) > core.predictor_history_entries(
        _wr_context(micro_steps=4, max_iterations=2)
    )
    assert more_microsteps > two_sweeps
    assert five_nodes > two_sweeps


def test_wr_wall_history_counts_every_retained_sweep():
    walls = (object(), object())
    one = core.predictor_wall_history_entries(
        _wr_context(max_iterations=1, walls=walls)
    )
    four = core.predictor_wall_history_entries(
        _wr_context(max_iterations=4, walls=walls)
    )
    more_micro = core.predictor_wall_history_entries(
        _wr_context(micro_steps=4, max_iterations=1, walls=walls)
    )
    five_nodes = core.predictor_wall_history_entries(
        _wr_context(ion_nodes=5, max_iterations=1, walls=walls)
    )
    assert 0 < one < four
    assert more_micro > one
    assert five_nodes > one


def test_wr_alias_memory_owns_its_padded_sample_identity():
    five = core.predictor_history_entries(
        _wr_context(N=5, history_samples=1)
    )
    nine = core.predictor_history_entries(
        _wr_context(N=9, history_samples=1)
    )
    assert nine > five


@pytest.mark.parametrize(
    "name,value",
    (
        ("micro_steps", True),
        ("micro_steps", 0),
        ("waveform_max_iterations", False),
        ("waveform_max_iterations", 0),
        ("ion_nodes", True),
        ("ion_nodes", 4),
    ),
)
def test_wr_memory_rejects_invalid_controls(name, value):
    context = _wr_context()
    setattr(context.p, name, value)
    with pytest.raises(
        Failure, match="Memory|Config|positive|3 or 5"
    ):
        core.predictor_history_entries(context)


def test_a0_predictor_and_waveform_estimates_remain_unchanged():
    context = _fake_memory_context(K=1, walls=[])
    context.p.micro_steps = 2
    assert core.predictor_history_entries(context) == 0
    context.p.coupling = "predictor"
    assert core.predictor_history_entries(context) == 28328
    context.p.coupling = "waveform"
    context.p.waveform_max_iterations = 1
    assert core.predictor_history_entries(context) == 28868
    context.p.waveform_max_iterations = 64
    assert core.predictor_history_entries(context) == 28868


def test_wr_capacity_preflight_reports_conservative_rejection():
    context = _wr_context(max_iterations=4)
    estimate = core.estimated_peak_bytes(context)
    context.p.memory_limit = max(1, estimate - 1)
    with pytest.raises(Failure, match="Memory|memory"):
        core.preflight_memory(context)
