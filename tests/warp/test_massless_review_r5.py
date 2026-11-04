"""Final R5 contracts for checked wall-work allowances and active modes."""

from __future__ import annotations

import math

import numpy as np
import pytest

import fomhd.massless as massless
from fomhd.core import inverse
from fomhd.massless_checks import (
    _bound_real,
    _gamma_operation,
    _stage_residuals,
    _wall_work_metrics,
    check_stage,
)
from fomhd.types import Failure

from test_massless_support import context, stage_tag, uniform_state


def _record(stage, name):
    return next(
        item
        for item in stage.checks
        if item.name == name
        and item.region == "all"
        and str(item.norm) == "value"
    )


def _amplified_state():
    c = context(hard_diagnostics=False, nodes_r=6, nodes_z=6)
    state = uniform_state(c)
    state.B *= 100.0
    return c, state


def _current_perturbation(c, state, fraction):
    base = massless.evaluate_stage(state, stage_tag(state), c)
    curl = np.asarray(c.op.C[0])
    adjoint = np.asarray(c.op.Cdag[0])
    extraction = np.asarray(base.cache.wall_correction.extraction[0])
    allowance = (
        4096.0
        * np.finfo(np.float64).eps
        * c.op.Nb
        * np.max(np.abs(adjoint))
        * np.max(np.abs(state.B))
        / c.p.mu0
    )
    weighted_reaction = np.real(c.op.M1) * base.cache.wall_reaction
    perturbation = curl @ ((adjoint @ curl).conj().T @ weighted_reaction)
    boundary_current = extraction @ (adjoint @ perturbation / c.p.mu0)
    perturbation *= fraction * allowance / np.max(np.abs(boundary_current))
    state.B += perturbation
    return allowance, extraction


def _momentum_perturbation(c, state, fraction):
    state.fluid[0].mom.reshape(3, c.op.Nc)[1] = (
        c.p.species[0].mass * c.p.ref.ni * 1.0e5
    )
    base = massless.evaluate_stage(state, stage_tag(state), c)
    allowance = 2048.0 * np.finfo(np.float64).eps * max(
        np.max(np.abs(state.fluid[0].mom)),
        c.p.species[0].mass * c.p.ref.ni * c.p.ref.velocity,
    )
    perturbation = np.asarray(c.op.I).conj().T @ (
        np.real(c.op.M1) * base.cache.wall_reaction
    )
    perturbation *= fraction * allowance / np.max(
        np.abs(c.op.T_full @ perturbation)
    )
    state.fluid[0].mom += perturbation
    return allowance


def test_aggregate_bound_retains_components_and_addition_roundoff():
    c, state = _amplified_state()
    stage = massless.evaluate_stage(state, stage_tag(state), c)
    ion = _record(stage, "wall_field_reaction_work_ion")
    electron = _record(stage, "wall_field_reaction_work_electron")
    current = _record(stage, "wall_field_reaction_work_current")
    aggregate = _record(stage, "wall_field_reaction_work")
    assert aggregate.arithmetic_bound >= (
        ion.arithmetic_bound + electron.arithmetic_bound
    )
    assert aggregate.compatibility_bound >= (
        ion.compatibility_bound + electron.compatibility_bound
    )
    assert all(item.passed for item in (ion, electron, current, aggregate))


def test_half_allowance_inputs_pass_and_twice_allowance_inputs_fail():
    for producer in ("current", "momentum"):
        c, state = _amplified_state()
        if producer == "current":
            allowance, extraction = _current_perturbation(c, state, 0.5)
            stage = massless.evaluate_stage(state, stage_tag(state), c)
            actual = float(np.max(np.abs(extraction @ stage.current)))
            assert actual < allowance
            assert _record(
                stage, "wall_field_reaction_work_electron"
            ).passed
            assert _record(stage, "wall_field_reaction_work_current").passed
        else:
            allowance = _momentum_perturbation(c, state, 0.5)
            stage = massless.evaluate_stage(state, stage_tag(state), c)
            actual = float(
                np.max(np.abs(c.op.T_full @ state.fluid[0].mom))
            )
            assert actual < allowance
            assert _record(stage, "wall_field_reaction_work_ion").passed

        c_bad, state_bad = _amplified_state()
        if producer == "current":
            _current_perturbation(c_bad, state_bad, 2.0)
            message = "initial current violates"
        else:
            _momentum_perturbation(c_bad, state_bad, 2.0)
            message = "input momentum violates"
        with pytest.raises(Failure, match=message):
            massless.evaluate_stage(
                state_bad, stage_tag(state_bad), c_bad
            )


@pytest.mark.parametrize("mode", ("reaction", "producer_inputs"))
def test_zero_metric_inputs_have_zero_work_and_valid_bound(mode):
    c, state = _amplified_state()
    stage = massless.evaluate_stage(state, stage_tag(state), c)
    if mode == "reaction":
        stage.cache.wall_reaction *= 0.0
    else:
        state.B *= 0.0
        stage.current *= 0.0
        for fluid in stage.fluid:
            fluid.flux_hat *= 0.0
            fluid.flux *= 0.0
    metrics = _wall_work_metrics(state, stage, c)
    for item in metrics.values():
        assert item.work == 0.0
        assert item.sample_work == 0.0
        assert item.transform_difference == 0.0
        assert item.absolute_scale == 0.0
        assert math.isfinite(item.upstream_bound)
        assert math.isfinite(item.compatibility_bound)
        assert item.upstream_bound >= 0.0
        assert item.compatibility_bound >= 0.0
        assert math.isfinite(item.arithmetic_bound)
        assert item.arithmetic_bound >= 0.0


def _active_mode_state():
    c = context(K=1, hard_diagnostics=False, nodes_r=6, nodes_z=6)
    state = uniform_state(c)
    nb, nc, ne = c.op.Nb, c.op.Nc, c.op.Ne
    extraction = np.zeros((len(c.geometry.walls), ne))
    nr = len(c.geometry.r_faces.keys)
    for row, wall in enumerate(c.geometry.walls):
        edge = (
            wall.face
            if wall.component == "r_face"
            else nr + nc + wall.face
        )
        extraction[row, edge] = wall.sign
    constraints = np.vstack(
        (extraction @ c.op.Cdag[2], c.op.DB[2])
    )
    _, singular, vh = np.linalg.svd(constraints, full_matrices=True)
    rank = np.count_nonzero(singular > c.p.rank_rtol * singular[0])
    nullspace = vh[rank:].conj().T
    rng = np.random.default_rng(184)
    wave = nullspace @ (
        rng.normal(size=nullspace.shape[1])
        + 1j * rng.normal(size=nullspace.shape[1])
    )
    wave *= 5.0e-7 / np.max(np.abs(wave))
    state.B.reshape(3, nb)[2] = wave
    state.B.reshape(3, nb)[0] = wave.conj()
    state.fluid[0].n.reshape(3, nc)[2] = (0.04 + 0.03j) * c.p.ref.ni
    state.fluid[0].n.reshape(3, nc)[0] = (0.04 - 0.03j) * c.p.ref.ni
    state.fluid[0].mom.reshape(3, 3, nc)[:, 1, :] = (
        100.0 * c.p.species[0].mass * state.fluid[0].n.reshape(3, nc)
    )
    return c, state


def test_active_harmonics_match_direct_parseval_and_transform_isolation():
    c, state = _active_mode_state()
    stage = massless.evaluate_stage(state, stage_tag(state), c)
    reaction = stage.cache.wall_reaction.reshape(3, c.op.Ne)
    assert np.linalg.norm(reaction[[0, 2]]) > 0.0
    assert np.linalg.norm(
        stage.current.reshape(3, c.op.Ne)[[0, 2]]
    ) > 1.0e-3
    phase = np.exp(
        2j
        * np.pi
        * np.arange(c.N)[:, None]
        / c.N
        * np.arange(-1, 2)[None, :]
    )
    for name, flux in (
        "ion",
        c.p.species[0].charge * stage.fluid[0].flux_hat,
    ), (
        "electron",
        c.p.species[1].charge * stage.fluid[1].flux_hat,
    ), ("current", stage.current):
        direct = (
            (phase @ flux.reshape(3, c.op.Ne)).conj()
            * (phase @ reaction)
            * c.op.M1[None, :]
        )
        expected = float(2.0 * np.pi / c.N * np.sum(direct).real)
        scale = float(2.0 * np.pi / c.N * np.sum(np.abs(direct)).real)
        item = _record(stage, f"wall_field_reaction_work_{name}")
        assert abs(item.raw - expected) <= (
            4096.0 * np.finfo(np.float64).eps * scale
        )
        assert item.transform_difference == pytest.approx(
            item.sample_work - item.modal_work
        )
    before = _wall_work_metrics(state, stage, c)["electron"]
    original = stage.fluid[1].flux.copy()
    sampled_reaction = inverse(stage.cache.wall_reaction, c.op.Ne, c)
    stage.fluid[1].flux += (
        1.0e16 * sampled_reaction / np.max(np.abs(sampled_reaction))
    )
    after = _wall_work_metrics(state, stage, c)["electron"]
    assert after.work == before.work
    assert after.arithmetic_bound == before.arithmetic_bound
    assert abs(
        after.transform_difference - before.transform_difference
    ) > before.arithmetic_bound
    stage.fluid[1].flux = original


def test_gamma_domain_and_finite_nonnegative_bounds():
    eps = np.finfo(np.float64).eps
    assert _gamma_operation(0.0, eps) == 0.0
    with pytest.raises(Failure, match="operation count"):
        _gamma_operation(-1.0, eps)
    with pytest.raises(Failure, match="operation count"):
        _gamma_operation(math.inf, eps)
    with pytest.raises(Failure, match="operation count"):
        _gamma_operation(1.0 / eps, eps)
    c, state = _active_mode_state()
    stage = massless.evaluate_stage(state, stage_tag(state), c)
    for item in _wall_work_metrics(state, stage, c).values():
        for name in (
            "absolute_scale",
            "upstream_bound",
            "compatibility_bound",
            "arithmetic_bound",
        ):
            value = getattr(item, name)
            assert math.isfinite(value)
            assert value >= 0.0


@pytest.mark.parametrize(
    "value", [math.inf, math.nan, -1.0, complex(0, math.inf), complex(0, math.nan)]
)
def test_reduced_bound_guard_rejects_nonfinite_or_negative_scalars(value):
    with pytest.raises(Failure, match="finite nonnegative error bound"):
        _bound_real(value, "synthetic reduced bound")


@pytest.fixture(scope="module")
def bound_record_fixture():
    c, state = _amplified_state()
    return c, state, massless.evaluate_stage(state, stage_tag(state), c)


@pytest.mark.parametrize("suffix", ["_ion", "_electron", "_current", ""])
@pytest.mark.parametrize(
    "field", ["scale", "upstream_bound", "compatibility_bound", "arithmetic_bound"]
)
@pytest.mark.parametrize(
    "value", [math.inf, math.nan, -1.0, complex(0, math.inf), complex(0, math.nan)]
)
def test_invalid_record_bound_is_rejected(
    bound_record_fixture, monkeypatch, suffix, field, value
):
    c, state, stage = bound_record_fixture
    key = f"wall_field_reaction_{field}{suffix}"
    monkeypatch.setitem(stage.residuals, key, np.array([value], dtype=np.complex128))
    stage.checks = []
    with pytest.raises(Failure, match="finite nonnegative error bound"):
        check_stage(state, stage, c)


def test_finite_inputs_that_overflow_a_paired_bound_are_rejected():
    c, state = _amplified_state()
    stage = massless.evaluate_stage(state, stage_tag(state), c)
    state.B *= 1.0e280
    stage.current *= 1.0e280
    for fluid in stage.fluid:
        fluid.flux_hat *= 1.0e280
        fluid.flux *= 1.0e280
    stage.cache.wall_reaction *= 1.0e30
    assert np.all(np.isfinite(state.B))
    assert np.all(np.isfinite(stage.current))
    assert np.all(np.isfinite(stage.cache.wall_reaction))
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(Failure, match="finite nonnegative error bound"):
            _wall_work_metrics(state, stage, c)


def test_nonzero_harmonic_reaction_mutation_is_rejected():
    c, state = _active_mode_state()
    stage = massless.evaluate_stage(state, stage_tag(state), c)
    injection = stage.fluid[1].flux_hat.reshape(3, c.op.Ne).copy()
    injection[1] = 0.0
    injection *= 0.1 / np.max(np.abs(injection))
    stage.cache.wall_reaction += injection.reshape(-1)
    stage.residuals.update(_stage_residuals(state, stage, c))
    stage.checks = []
    check_stage(state, stage, c)
    item = _record(stage, "wall_field_reaction_work_electron")
    assert not item.passed
    assert abs(item.raw) > item.threshold
