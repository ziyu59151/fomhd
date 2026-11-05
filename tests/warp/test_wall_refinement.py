"""Regression tests for bounded finite-mass wall projection refinement."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fomhd import core
from fomhd.types import Failure, StageTag
from fomhd.wall import build_wall, project_wall
from massless_verification_fixture import HallCase, build_periodic_context


ROOT = Path(__file__).resolve().parents[2]
STUDY_PATH = ROOT / "tools/warp_momentum_study.py"
SPEC = importlib.util.spec_from_file_location("wall_refinement_study", STUDY_PATH)
assert SPEC is not None and SPEC.loader is not None
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)

_EPS = np.finfo(np.float64).eps
_TINY = np.finfo(np.float64).tiny


def _maximum(value) -> float:
    xp = (
        value.__array_namespace__()
        if hasattr(value, "__array_namespace__")
        else np
    )
    return float(xp.max(xp.abs(value)).item()) if value.size else 0.0


def _trace_bound(context, matrix, values):
    xp = context.compute.xp
    matrix_norm = float(xp.max(xp.sum(xp.abs(matrix), axis=1)).item())
    value_scale = _maximum(values)
    return (
        256.0
        * _EPS
        * matrix_norm
        * max(value_scale, _TINY)
        + _TINY
    )


def _assert_projection_bookkeeping(context, wall, original, projected):
    xp = context.compute.xp
    value = xp.asarray(projected.value).reshape(
        context.N, 3 * context.op.Nc
    )
    reaction = xp.asarray(projected.reaction).reshape(
        context.N, 3 * context.op.Nc
    )
    multiplier = xp.asarray(projected.multiplier).reshape(
        context.N, context.op.T.shape[0]
    )
    residual = xp.asarray(projected.multiplier_residual).reshape(
        context.N, context.op.T.shape[0]
    )
    sampled = xp.asarray(original, dtype=xp.complex128).reshape(
        context.N, 3 * context.op.Nc
    )
    wall_matrix = xp.asarray(context.op.T, dtype=xp.complex128)
    wall_adjoint = xp.conj(wall_matrix.T)

    full_trace = core.sample_apply(
        context.op.T_full, projected.value, context.N, context.compute
    )
    assert _maximum(full_trace) <= _trace_bound(
        context, xp.asarray(context.op.T_full), value
    )

    consistency_scale = max(
        _maximum(sampled), _maximum(value), _maximum(reaction)
    )
    assert _maximum(value - sampled - reaction) <= (
        512.0 * _EPS * consistency_scale
    )

    for sample, record in enumerate(wall.samples):
        rhs = wall_matrix @ sampled[sample]
        expected_reaction_weighted = -(wall_adjoint @ multiplier[sample])
        weighted_scale = max(
            _maximum(expected_reaction_weighted),
            _maximum(reaction[sample] / record.winv),
        )
        assert _maximum(
            reaction[sample] / record.winv - expected_reaction_weighted
        ) <= 512.0 * _EPS * weighted_scale

        expected_residual = record.S @ multiplier[sample] - rhs
        residual_scale = max(
            _maximum(expected_residual), _maximum(residual[sample])
        )
        assert _maximum(residual[sample] - expected_residual) <= (
            512.0 * _EPS * residual_scale
        )

    metric_inverse = xp.concatenate([sample.winv for sample in wall.samples])
    work = xp.sum(
        xp.conj(value.reshape(-1)) * reaction.reshape(-1) / metric_inverse
    )
    work_bound = max(
        2.0e-9,
        1.0e-12
        * float(xp.sum(xp.abs(sampled) ** 2).item())
        / float(xp.min(metric_inverse).item()),
    )
    assert abs(complex(work)) <= work_bound


@pytest.mark.parametrize("nodes", (6, 9))
def test_constant_pressure_projection_satisfies_quiet_wall_state_horizon(
    nodes: int,
) -> None:
    context = STUDY.build_context(nodes, "rectangle", original=True)
    reference = STUDY.reference(context)
    stage = STUDY.evaluate(reference, context)
    raw = stage.fluid[1].non_electric + stage.terms[1].electric
    projection = project_wall(stage.fluid[1].wall, raw, context)
    _assert_projection_bookkeeping(
        context, stage.fluid[1].wall, raw, projection
    )

    retained_rate = core.forward(
        projection.value, 3 * context.op.Nc, context
    )
    leaked = copy.copy(reference)
    leaked.fluid = list(reference.fluid)
    leaked.fluid[1] = copy.copy(reference.fluid[1])
    leaked.fluid[1].mom = retained_rate * STUDY.HORIZON

    core.check_retained(leaked, context)


def test_nonzero_projection_refines_forward_trace_and_preserves_bookkeeping():
    context = STUDY.build_context(4, "rectangle")
    xp = context.compute.xp
    state = STUDY.reference(context)
    tag = StageTag(0, 1, 1, 0, 0.0, "refinement")
    density = core.inverse(
        state.fluid[0].n, context.op.Nc, context, physical=True
    )
    wall = build_wall(density, context.p.species[0], tag, context)

    rng = np.random.default_rng(20261105)
    original = (
        rng.standard_normal(context.N * 3 * context.op.Nc)
        + 1j * rng.standard_normal(context.N * 3 * context.op.Nc)
    )
    original_copy = original.copy()
    projected = project_wall(wall, original, context)
    np.testing.assert_array_equal(original, original_copy)

    assert _maximum(xp.asarray(projected.reaction)) > 0.0
    assert _maximum(xp.asarray(projected.value)) > 0.0
    _assert_projection_bookkeeping(context, wall, original, projected)

    again = project_wall(wall, projected.value, context)
    idempotence_scale = max(
        _maximum(xp.asarray(projected.value)),
        _maximum(xp.asarray(again.value)),
    )
    assert _maximum(xp.asarray(again.value) - projected.value) <= (
        512.0 * _EPS * idempotence_scale
    )


def test_empty_wall_returns_input_unchanged():
    case = HallCase()
    context = build_periodic_context(case)
    xp = context.compute.xp
    density = xp.full(
        context.N * context.op.Nc,
        case.n0,
        dtype=xp.complex128,
    )
    tag = StageTag(0, 1, 1, 0, 0.0, "empty-wall")
    wall = build_wall(density, context.p.species[0], tag, context)

    rng = np.random.default_rng(17)
    original = xp.asarray(
        rng.standard_normal(context.N * 3 * context.op.Nc)
        + 1j * rng.standard_normal(context.N * 3 * context.op.Nc),
        dtype=xp.complex128,
    )
    original_copy = original.copy()
    projected = project_wall(wall, original, context)

    np.testing.assert_array_equal(original, original_copy)
    np.testing.assert_array_equal(projected.value, original)
    assert projected.reaction.shape == original.shape
    assert projected.multiplier.shape == (0,)
    assert projected.multiplier_residual.shape == (0,)
    assert not np.any(projected.reaction)


def test_projection_rejects_inconsistent_unselected_wall_row():
    context = STUDY.build_context(4, "rectangle")
    state = STUDY.reference(context)
    tag = StageTag(0, 1, 1, 0, 0.0, "full-row-negative")
    density = core.inverse(state.fluid[0].n, context.op.Nc, context, physical=True)
    wall = build_wall(density, context.p.species[0], tag, context)
    extra_row = np.zeros((1, 3 * context.op.Nc), dtype=complex)
    extra_row[0, context.op.Nc] = 1.0
    context.op.T_full = np.vstack((context.op.T_full, extra_row))
    force = np.tile(extra_row[0], context.N)
    with pytest.raises(Failure, match="full wall trace"):
        project_wall(wall, force, context)
