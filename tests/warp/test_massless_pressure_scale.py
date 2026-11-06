"""Strict regression for massless pressure-work floating-point scaling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fomhd import massless_checks
from fomhd.stage import initialize
from fomhd.time import advance
from fomhd.types import Failure, cumulative_record

from massless_verification_fixture import (
    HallCase,
    _branch_eigenpair,
    build_periodic_context,
    evaluate as evaluate_hall,
    initial_hall_eigenmode,
    uniform_state,
)


def _hall_context_and_initial_state():
    case = HallCase(kdi=1.0)
    _, _, eigenvalue = _branch_eigenpair(case)
    context = build_periodic_context(case)
    state, _, _ = initial_hall_eigenmode(
        case,
        relative_amplitude=2.0e-6,
        context=context,
    )
    context.p.coupling = "predictor"
    context.p.micro_steps = 1
    context.p.integrator = "rk4"
    context.p.nsteps = 1
    context.p.dt = 2.0 * math.pi / float(eigenvalue.imag) / 64.0
    context.p.dt_guard = "warn"
    return context, state


def _budget_snapshot(parameters):
    return {
        name: (
            budget.atol,
            budget.rtol,
            budget.reference,
            budget.unit,
            budget.norm,
            budget.hard,
        )
        for name, budget in parameters.budgets.items()
    }


def test_hall_first_predictor_k3_uses_pre_cancellation_pressure_operands():
    context, state = _hall_context_and_initial_state()
    before_budgets = _budget_snapshot(context.p)
    cumulative = cumulative_record()
    committed = initialize(state, context, cumulative, equilibrium=False)

    result = advance(committed, cumulative, context)

    assert _budget_snapshot(context.p) == before_budgets
    assert result.accepted
    records = [
        record
        for record in result.checks
        if record.name == "electron_pressure_adjoint"
        and record.population == context.p.species[1].id
        and record.tag.stage == "predictor_m000_r3"
        and record.tag.spectral_context == 0
    ]
    assert len(records) == 1
    record = records[0]
    assert record.hard is True
    assert record.available is True
    assert record.passed is True
    assert math.isfinite(record.raw)
    assert math.isfinite(record.reference)
    assert abs(record.raw) < 1.0e-20
    assert record.reference > 1.0e-8
    assert record.threshold >= (
        4096.0 * np.finfo(np.float64).eps * record.reference
    )
    assert abs(record.raw) <= record.threshold

    pairing = [
        item
        for item in result.checks
        if item.name == "pressure_pairing"
        and item.tag.stage == "predictor_m000_r3"
        and item.tag.spectral_context == 0
    ]
    assert len(pairing) == 1
    assert pairing[0].hard is True
    assert pairing[0].passed is True
    assert pairing[0].configured_threshold > 0.0

    unrelated = {
        record.name: record
        for record in result.checks
        if record.tag.stage == "predictor_m000_r3"
        and record.tag.spectral_context == 0
        and record.name
        in (
            "electron_pressure_only_pairing",
            "electron_pressure_truncation",
        )
    }
    assert set(unrelated) == {
        "electron_pressure_only_pairing",
        "electron_pressure_truncation",
    }
    original_scale = unrelated["electron_pressure_truncation"].reference
    assert original_scale > 0.0
    assert original_scale < 1.0e-3 * record.reference
    assert unrelated["electron_pressure_only_pairing"].reference == (
        pytest.approx(original_scale)
    )
    only_budget = context.p.budgets[
        "electron_pressure_only_pairing.electron.value"
    ]
    expected_configured = only_budget.atol + (
        only_budget.rtol * only_budget.reference
    )
    assert unrelated["electron_pressure_only_pairing"].configured_threshold == (
        pytest.approx(expected_configured)
    )


def test_corrupted_pressure_work_still_fails_strict_hard_check():
    case = HallCase(kdi=1.0)
    context = build_periodic_context(case)
    state = uniform_state(context, case)
    result = evaluate_hall(state, context)
    result.checks = []
    result.residuals["electron_pressure_adjoint"] = context.compute.array(
        np.asarray([1.0e-3], dtype=np.complex128)
    )

    with pytest.raises(Failure, match="electron_pressure_adjoint") as caught:
        massless_checks.check_stage(state, result, context)

    records = [
        record
        for record in caught.value.checks
        if record.name == "electron_pressure_adjoint"
        and record.population == context.p.species[1].id
    ]
    assert records
    record = records[0]
    assert record.hard is True
    assert record.available is True
    assert record.passed is False
    assert abs(record.raw) > record.threshold
