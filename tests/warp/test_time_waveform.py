"""A1 T1 ion-stage-path waveform scheduler and transaction tests."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest
from test_time_predictor import (
    _cumulative,
    _parts,
    _set_identity,
    _toy_fluid,
    _toy_stage,
    _ToyCompute,
)

from fomhd import core, diagnostics, io
from fomhd import time as time_module
from fomhd.types import Failure, StageTag, check_record

_A = np.asarray(
    [
        [0.0, 0.35, 0.08, 0.60, 0.00, 0.00],
        [-0.35, 0.0, 0.22, 0.00, 0.55, 0.00],
        [0.00, -0.22, 0.0, 0.00, 0.00, 0.45],
        [0.80, 0.00, 0.00, 0.0, 0.25, 0.00],
        [0.00, 0.75, 0.00, -0.25, 0.0, 0.20],
        [0.00, 0.00, 0.65, 0.00, -0.20, 0.0],
    ],
    dtype=np.float64,
)
_C = np.asarray([1.0, -0.5, 0.25, 0.10, -0.20, 0.30])


def _linear_rhs(clock, ion, electron):
    del clock
    values = np.asarray(ion + electron, dtype=np.float64)
    derivative = _A @ values + _C
    return tuple(derivative[:3]), tuple(derivative[3:])


def _fine_rk4(initial, h, micro_steps):
    values = np.asarray(initial, dtype=np.float64).copy()
    for _ in range(micro_steps):
        k1 = _A @ values + _C
        k2 = _A @ (values + 0.5 * h * k1) + _C
        k3 = _A @ (values + 0.5 * h * k2) + _C
        k4 = _A @ (values + h * k3) + _C
        values += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return values


class _CountingCompute(_ToyCompute):
    def __init__(self):
        super().__init__()
        self.max_abs_calls = 0

    def max_abs(self, value):
        self.max_abs_calls += 1
        return super().max_abs(value)


class ToyWaveform:
    def __init__(
        self,
        monkeypatch,
        *,
        H=0.2,
        M=4,
        rtol=1.0e-12,
        max_iterations=12,
        fail_check_call=None,
        fail_stage=None,
        fail_assembly_stage=None,
    ):
        self.H = H
        self.M = M
        self.fail_check_call = fail_check_call
        self.fail_stage = fail_stage
        self.fail_assembly_stage = fail_assembly_stage
        self.evaluated = []
        self.stages = []
        self.assembled = []
        self.check_contexts = []
        self.retained = []
        self.source_id = 0
        self.alias_calls = 0
        self.evaluate_attempts = 0
        self.assembly_attempts = 0
        self.p = SimpleNamespace(
            start_time=0.0,
            dt=H,
            nsteps=1,
            electron_inertia=0,
            integrator="rk4",
            coupling="waveform",
            micro_steps=M,
            waveform_rtol=rtol,
            waveform_max_iterations=max_iterations,
            dt_guard="warn",
            minimum_samples_per_cycle=32.0,
            max_amplitude_error=0.01,
            max_phase_error_rad=0.1,
            species=(
                SimpleNamespace(id="i", charge=1.0, mass=2.0, n_min=0.0),
                SimpleNamespace(id="e", charge=-1.0, mass=1.0, n_min=0.0),
            ),
            ref=SimpleNamespace(
                ni=1.0,
                ne=1.0,
                velocity=1.0,
                pressure=1.0,
                magnetic=1.0,
            ),
            budgets={
                "total_energy_interval.value": SimpleNamespace(hard=True),
                "field_energy_B_interval.value": SimpleNamespace(hard=True),
                "kinetic_energy.i.value": SimpleNamespace(hard=True),
                "species_continuity.value": SimpleNamespace(hard=True),
            },
        )
        self.compute = _CountingCompute()
        self.context = SimpleNamespace(
            p=self.p,
            compute=self.compute,
            next_generation=2,
            next_trial=1,
            next_source=1,
            spectral_id=0,
            last_dt_report=None,
        )
        initial = _set_identity(
            _toy_fluid((1.0, 2.0, 3.0), (0.2, 0.3, 0.4)),
            StageTag(step=0, generation=1, time=0.0),
        )
        initial.accepted = True
        self.committed = SimpleNamespace(
            state=initial,
            cache=SimpleNamespace(tag="initial"),
            accepted=True,
        )
        self.cumulative = _cumulative()
        from fomhd import time_predictor, time_waveform

        self.module = time_waveform

        def assemble(context, ion, electron, tag):
            self.assembly_attempts += 1
            if self.fail_assembly_stage == tag.stage:
                failure = Failure(
                    "Domain", "injected waveform assembly failure"
                )
                failure.stage = tag
                raise failure
            ion = tuple(
                float(np.real(np.asarray(value).reshape(-1)[0]))
                for value in ion
            )
            electron = tuple(
                float(np.real(np.asarray(value).reshape(-1)[0]))
                for value in electron
            )
            self.assembled.append((ion, electron, tag))
            state = _toy_fluid(ion, electron)
            return _set_identity(state, tag)

        def evaluate(state, tag, context, alias=False):
            self.evaluate_attempts += 1
            if self.fail_stage == tag.stage:
                failure = Failure("Domain", "injected waveform stage failure")
                failure.stage = tag
                raise failure
            self.source_id += 1
            source_id = self.source_id
            self.context.next_source = self.source_id + 1
            self.evaluated.append((state, tag, alias))
            ion, electron = _parts(state)
            f_ion, g_electron = _linear_rhs(tag.time, ion, electron)
            derivative = _toy_fluid(f_ion, g_electron)
            stage = _toy_stage(derivative, tag, source_id)
            self.stages.append(stage)
            if alias:
                self.alias_calls += 1
                self.source_id += 1
                self.context.next_source = self.source_id + 1
            return stage

        def check_step(
            left,
            base,
            midpoint_state,
            midpoint,
            right,
            end,
            cumulative,
            context,
            prescribed_source=None,
            *,
            quadrature=None,
        ):
            del left, base, midpoint_state, right, cumulative
            del end, prescribed_source
            self.check_contexts.append(context)
            call = len(self.check_contexts)
            if self.fail_check_call == call:
                failure = Failure("HardGate", "injected final interval failure")
                failure.stage = midpoint.tag
                failure.checks = [
                    check_record(
                        name="total_energy_interval",
                        unit="J",
                        tag=midpoint.tag,
                        raw=2.0,
                        reference=3.0,
                        threshold=0.0,
                        passed=False,
                        hard=False,
                    )
                ]
                raise failure
            records = []
            for name in ("total_energy_interval", "field_energy_B_interval"):
                key = f"{name}.value"
                records.append(
                    check_record(
                        name=name,
                        unit="J",
                        tag=midpoint.tag,
                        raw=1.0,
                        reference=1.0,
                        threshold=0.0,
                        passed=False,
                        hard=context.p.budgets[key].hard,
                    )
                )
            records.append(
                check_record(
                    name="kinetic_energy",
                    population="i",
                    unit="J",
                    tag=midpoint.tag,
                    raw=1.0,
                    reference=1.0,
                    threshold=0.0,
                    passed=False,
                    hard=context.p.budgets["kinetic_energy.i.value"].hard,
                )
            )
            return SimpleNamespace(
                checks=records,
                totals=self.stages[-1].totals,
                momentum=self.stages[-1].momentum,
                interval=SimpleNamespace(
                    accepted=False,
                    weight=self.H,
                    number_flux=[np.zeros(1), np.zeros(1)],
                    current=[np.zeros(1), np.zeros(1)],
                    quadrature=[
                        SimpleNamespace(
                            source_id=stage.source_id,
                            weight=weight,
                            stage=stage.tag.stage,
                        )
                        for _, stage, weight in quadrature
                    ],
                ),
            )

        monkeypatch.setattr(self.module, "evaluate_stage", evaluate)
        monkeypatch.setattr(self.module, "_assemble_state", assemble)
        def check_guard(stage, context):
            del context
            stage.checks.append(
                check_record(
                    name="time_step_estimate",
                    unit="1",
                    value_unit="1",
                    tag=stage.tag,
                    raw=0.5,
                    reference=1.0,
                    threshold=1.0,
                    passed=True,
                    hard=True,
                )
            )

        monkeypatch.setattr(self.module, "_check_guard", check_guard)
        monkeypatch.setattr(
            time_predictor,
            "retain_failure_evidence",
            lambda failure, tag, context, state, result, prefix:
            self.retained.append((prefix, tag, state, result)),
        )
        monkeypatch.setattr(core, "preflight_memory", lambda context: 4096)
        monkeypatch.setattr(diagnostics, "check_step", check_step)
        monkeypatch.setattr(diagnostics, "gate", lambda *args, **kwargs: None)

    def run(self):
        return time_module.advance(
            self.committed, self.cumulative, self.context
        )

    def endpoint(self):
        ion, electron = _parts(self.committed.state)
        return np.asarray(ion + electron, dtype=np.float64)


def _iteration_metric(result):
    return result.coupling_metadata["iterations"]


def _midpoint_states(toy, iteration):
    values = {}
    for state, tag, _ in toy.evaluated:
        if tag.iteration != iteration:
            continue
        if tag.stage.endswith("_r2"):
            values["p2"] = state
        elif tag.stage.endswith("_r3"):
            values["p3"] = state
    return values


def test_linear_fixed_point_matches_independent_fine_rk4(monkeypatch):
    toy = ToyWaveform(monkeypatch, H=0.2, M=4, rtol=1.0e-13)
    result = toy.run()
    expected = _fine_rk4(
        (1.0, 2.0, 3.0, 0.2, 0.3, 0.4),
        toy.H / toy.M,
        toy.M,
    )

    assert result.coupling_metadata["converged"] is True
    np.testing.assert_allclose(toy.endpoint(), expected, rtol=0.0, atol=2.0e-12)
    assert toy.committed.state.step == 1
    assert toy.cumulative.accepted_intervals == 1
    assert toy.compute.sync_calls == 1
    assert toy.compute.axpy_calls_after_sync == 0
    assert sum(
        record.name == "time_step_estimate"
        for record in result.checks
    ) == 1


def test_path_and_endpoint_residuals_decrease_on_moderate_case(monkeypatch):
    toy = ToyWaveform(monkeypatch, H=0.3, M=5, rtol=1.0e-12)
    result = toy.run()
    iterations = _iteration_metric(result)

    assert len(iterations) >= 2
    path = [item["path_max_normalized"] for item in iterations]
    endpoint = [item["endpoint_max_normalized"] for item in iterations]
    assert all(left > right for left, right in pairwise(path))
    assert all(left > right for left, right in pairwise(endpoint))
    assert path[-1] <= toy.p.waveform_rtol
    assert endpoint[-1] <= toy.p.waveform_rtol
    assert all(
        set(item["path"]) == {"n_i", "mom_i", "perp_i"}
        for item in iterations
    )
    assert all(
        set(item["endpoint"]) == {"n_i", "mom_i", "perp_i"}
        for item in iterations
    )
    assert toy.compute.max_abs_calls > 0


def test_distinct_midpoints_and_iteration_stage_identity(monkeypatch):
    toy = ToyWaveform(monkeypatch, H=0.25, M=3, rtol=1.0e-12)
    result = toy.run()
    last = result.coupling_metadata["sweeps"] - 1
    midpoint = _midpoint_states(toy, last)

    assert midpoint["p2"].fluid[1].perp[0] != midpoint["p3"].fluid[1].perp[0]
    assert midpoint["p2"].B[0] != midpoint["p3"].B[0]
    tags = [tag for _, tag, _ in toy.evaluated]
    assert [tag.iteration for tag in tags] == sorted(tag.iteration for tag in tags)
    assert len({tag.generation for tag in tags}) == len(tags)
    assert {tag.micro_step for tag in tags if "_m" in tag.stage} <= set(
        range(toy.M)
    )
    for _, tag, alias in toy.evaluated:
        assert alias is ("_r2" in tag.stage or "_r3" in tag.stage)
    alias_calls = result.coupling_metadata["alias_evaluations"]
    assert alias_calls == toy.alias_calls
    assert alias_calls == 2 * toy.M * result.coupling_metadata["sweeps"]
    assert result.coupling_metadata["complete_stage_evaluations"] == len(
        toy.evaluated
    )
    assert result.coupling_metadata["total_model_evaluations"] == (
        len(toy.evaluated) + alias_calls
    )
    assert toy.context.next_source - 1 == len(toy.evaluated) + alias_calls


def test_initial_tangent_from_true_base_and_left_state_unchanged(monkeypatch):
    toy = ToyWaveform(monkeypatch, H=0.2, M=4, rtol=1.0e-13)
    left = toy.committed.state
    snapshot = _parts(left)
    initial_f0 = _linear_rhs(0.0, *snapshot)[0]

    result = toy.run()

    first_path = [
        state
        for state, tag, _ in toy.evaluated
        if (
            tag.iteration == 0
            and tag.stage.startswith("waveform_")
            and "_m" in tag.stage
        )
    ]
    for state, tag in zip(first_path, [
        tag
        for _, tag, _ in toy.evaluated
        if (
            tag.iteration == 0
            and tag.stage.startswith("waveform_")
            and "_m" in tag.stage
        )
    ]):
        micro = tag.micro_step
        rk_stage = int(tag.stage.rsplit("_r", 1)[1]) - 1
        offset = core.microstage_offset(toy.p, micro, rk_stage)
        expected = np.asarray(snapshot[0]) + offset * np.asarray(initial_f0)
        actual = np.asarray(_parts(state)[0])
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)

    np.testing.assert_array_equal(np.real(left.fluid[0].n), 1.0)
    np.testing.assert_array_equal(np.real(left.fluid[0].mom), 2.0)
    np.testing.assert_array_equal(np.real(left.fluid[0].perp), 3.0)
    np.testing.assert_array_equal(np.real(left.fluid[1].perp), 0.2)
    np.testing.assert_array_equal(np.real(left.fluid[1].parallel), 0.3)
    np.testing.assert_array_equal(np.real(left.B), 0.4)
    assert result.accepted is True


def test_intermediate_energy_checks_are_observations_and_final_is_hard(
    monkeypatch,
):
    toy = ToyWaveform(monkeypatch, H=0.2, M=2, rtol=1.0e-12)
    result = toy.run()

    assert len(toy.check_contexts) == result.coupling_metadata["sweeps"] + 1
    for context in toy.check_contexts[:-1]:
        assert context.p.budgets["total_energy_interval.value"].hard is False
        assert context.p.budgets["field_energy_B_interval.value"].hard is False
        assert context.p.budgets["kinetic_energy.i.value"].hard is False
        assert context.p.budgets["species_continuity.value"].hard is True
    final_context = toy.check_contexts[-1]
    assert final_context is toy.context
    assert final_context.p.budgets["total_energy_interval.value"].hard is True
    assert toy.p.budgets["total_energy_interval.value"].hard is True
    for iteration in result.coupling_metadata["iterations"]:
        assert set(iteration["energy_budgets"]) == {
            "field_energy_B_interval",
            "kinetic_energy.i",
            "total_energy_interval",
        }
        assert all(
            isinstance(value["raw"], float)
            for value in iteration["energy_budgets"].values()
        )
    json.dumps(result.coupling_metadata, allow_nan=False)


def test_nonconvergence_rolls_back_without_trial_ledger_leakage(monkeypatch):
    toy = ToyWaveform(
        monkeypatch,
        H=0.3,
        M=4,
        rtol=1.0e-15,
        max_iterations=1,
    )
    old_state = toy.committed.state
    old_cache = toy.committed.cache

    with pytest.raises(Failure) as caught:
        toy.run()

    failure = caught.value
    assert failure.code == "WaveformConvergence"
    assert failure.coupling_metadata["converged"] is False
    assert failure.coupling_metadata["sweeps"] == 1
    assert len(failure.coupling_metadata["iterations"]) == 1
    assert toy.committed.state is old_state
    assert toy.committed.cache is old_cache
    assert toy.cumulative.accepted_intervals == 0
    assert toy.cumulative.last_trial == 0
    assert toy.cumulative.last_step == 0
    assert toy.compute.sync_calls == 0


def test_final_check_failure_rolls_back_after_converged_sweeps(monkeypatch):
    toy = ToyWaveform(
        monkeypatch,
        H=0.2,
        M=3,
        rtol=1.0,
        max_iterations=8,
        fail_check_call=2,
    )
    old_state = toy.committed.state

    with pytest.raises(Failure, match="final interval"):
        toy.run()

    assert toy.committed.state is old_state
    assert toy.cumulative.accepted_intervals == 0
    assert toy.cumulative.last_trial == 0
    assert toy.cumulative.last_step == 0
    assert toy.compute.sync_calls == 0


def test_trial_endpoint_failure_retains_reconstructed_ion_residual(monkeypatch):
    toy = ToyWaveform(
        monkeypatch,
        H=0.2,
        M=2,
        rtol=1.0,
        max_iterations=4,
        fail_stage="waveform_i000_endpoint",
    )

    with pytest.raises(Failure, match="injected waveform") as caught:
        toy.run()

    failure = caught.value
    metadata = failure.coupling_metadata
    assert metadata["converged"] is True
    assert metadata["sweeps"] == 1
    assert metadata["attempted_iteration"] == 0
    assert metadata["attempted_stage"]["stage"] == "waveform_i000_endpoint"
    assert metadata["complete_stage_evaluations"] == toy.evaluate_attempts
    record = metadata["iterations"][0]
    assert set(record["path"]) == {"n_i", "mom_i", "perp_i"}
    assert set(record["endpoint"]) == {"n_i", "mom_i", "perp_i"}
    assert record["energy_available"] is False
    assert record["energy_budgets"] == {}
    assert toy.committed.state.step == 0
    assert toy.cumulative.accepted_intervals == 0


def test_sweep_check_failure_retains_current_residual_and_energy(
    monkeypatch,
):
    toy = ToyWaveform(
        monkeypatch,
        H=0.2,
        M=2,
        rtol=1.0,
        max_iterations=4,
        fail_check_call=1,
    )

    with pytest.raises(Failure, match="injected final interval") as caught:
        toy.run()

    failure = caught.value
    metadata = failure.coupling_metadata
    assert metadata["converged"] is True
    assert metadata["attempted_iteration"] == 0
    assert metadata["complete_stage_evaluations"] == toy.evaluate_attempts
    record = metadata["iterations"][0]
    assert set(record["path"]) == {"n_i", "mom_i", "perp_i"}
    assert record["energy_available"] is True
    assert record["energy_budgets"]["total_energy_interval"]["raw"] == 2.0
    assert toy.committed.state.step == 0
    assert toy.cumulative.accepted_intervals == 0


def test_final_endpoint_failure_rebuilds_current_metadata(monkeypatch):
    toy = ToyWaveform(
        monkeypatch,
        H=0.2,
        M=2,
        rtol=1.0,
        max_iterations=4,
        fail_stage="endpoint",
    )

    with pytest.raises(Failure, match="injected waveform") as caught:
        toy.run()

    failure = caught.value
    metadata = failure.coupling_metadata
    assert metadata["converged"] is True
    assert metadata["sweeps"] == 1
    assert metadata["attempted_iteration"] == 0
    assert metadata["attempted_stage"]["stage"] == "endpoint"
    assert metadata["complete_stage_evaluations"] == toy.evaluate_attempts
    record = metadata["iterations"][0]
    assert set(record["path"]) == {"n_i", "mom_i", "perp_i"}
    assert record["energy_available"] is True
    assert toy.committed.state.step == 0
    assert toy.cumulative.accepted_intervals == 0


def test_sync_failure_retains_final_hard_checks_and_counts(monkeypatch):
    toy = ToyWaveform(
        monkeypatch,
        H=0.2,
        M=2,
        rtol=1.0e-6,
        max_iterations=4,
    )

    def fail_sync():
        raise RuntimeError("injected waveform synchronize failure")

    monkeypatch.setattr(toy.compute, "synchronize", fail_sync)
    with pytest.raises(
        Failure, match="injected waveform synchronize"
    ) as caught:
        toy.run()

    failure = caught.value
    metadata = failure.coupling_metadata
    assert metadata["converged"] is True
    assert metadata["complete_stage_evaluations"] == toy.evaluate_attempts
    assert metadata["total_model_evaluations"] == (
        toy.evaluate_attempts + toy.alias_calls
    )
    assert metadata["sweeps"] > 1
    assert metadata["attempted_stage"]["stage"] == "endpoint"
    final_energy = [
        record
        for record in failure.checks
        if record.name == "total_energy_interval"
    ]
    assert final_energy
    assert all(record.hard is True for record in final_energy)
    assert sum(
        record.name == "time_step_estimate"
        for record in failure.checks
    ) == 1
    assert toy.committed.state.step == 0
    assert toy.cumulative.accepted_intervals == 0


def test_failure_retains_only_current_sweep_stage_arrays(monkeypatch):
    toy = ToyWaveform(
        monkeypatch,
        H=0.2,
        M=3,
        rtol=1.0e-12,
        max_iterations=8,
        fail_stage="waveform_i001_m001_r3",
    )

    with pytest.raises(Failure, match="injected waveform") as caught:
        toy.run()

    assert toy.retained
    assert {tag.iteration for _, tag, _, _ in toy.retained} == {1}
    completed = [entry for entry in toy.retained if entry[2] is not None]
    assert len(completed) >= 3
    metadata = caught.value.coupling_metadata
    assert metadata["attempted_iteration"] == 1
    assert metadata["attempted_stage"]["stage"] == "waveform_i001_m001_r3"
    assert metadata["quadrature_stage_evaluations"] == 6
    assert metadata["scheduled_quadrature_stage_evaluations"] == 12
    assert toy.cumulative.accepted_intervals == 0


def test_assembly_failure_counts_only_actual_evaluate_calls(monkeypatch):
    toy = ToyWaveform(
        monkeypatch,
        H=0.2,
        M=2,
        rtol=1.0e-12,
        max_iterations=4,
        fail_assembly_stage="waveform_i000_m000_r2",
    )

    with pytest.raises(Failure, match="injected waveform assembly") as caught:
        toy.run()

    failure = caught.value
    metadata = failure.coupling_metadata
    assert metadata["complete_stage_evaluations"] == toy.evaluate_attempts == 1
    assert metadata["full_rhs_stage_evaluations"] == 1
    assert metadata["endpoint_full_evaluations"] == 0
    assert metadata["assemble_canonicalization_calls"] == (
        toy.assembly_attempts
    )
    assert metadata["assemble_canonicalization_calls"] == 1
    assert metadata["total_model_evaluations"] == 1
    assert metadata["quadrature_stage_evaluations"] == 1
    assert metadata["scheduled_quadrature_stage_evaluations"] == 4 * toy.M
    assert metadata["attempted_iteration"] == 0
    assert metadata["attempted_stage"]["stage"] == "waveform_i000_m000_r2"
    assert toy.committed.state.step == 0
    assert toy.cumulative.accepted_intervals == 0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("waveform_rtol", True),
        ("waveform_rtol", 0.0),
        ("waveform_rtol", -1.0),
        ("waveform_rtol", math.inf),
        ("waveform_rtol", math.nan),
        ("waveform_rtol", "1e-10"),
        ("waveform_max_iterations", True),
        ("waveform_max_iterations", 1.5),
        ("waveform_max_iterations", 0),
        ("waveform_max_iterations", -1),
    ],
)
def test_direct_invalid_waveform_controls_precede_mutation(
    monkeypatch, name, value
):
    toy = ToyWaveform(monkeypatch, H=0.2, M=2)
    setattr(toy.p, name, value)
    old_state = toy.committed.state
    old_trial = toy.context.next_trial
    old_generation = toy.context.next_generation

    with pytest.raises(Failure, match="waveform"):
        toy.run()

    assert toy.committed.state is old_state
    assert toy.context.next_trial == old_trial
    assert toy.context.next_generation == old_generation
    assert toy.cumulative.accepted_intervals == 0


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"electron_inertia": 1}, "waveform.*massless"),
        ({"integrator": "midpoint"}, "waveform.*rk4"),
    ],
)
def test_waveform_requires_massless_rk4_before_mutation(
    monkeypatch, updates, match
):
    toy = ToyWaveform(monkeypatch, H=0.2, M=2)
    for name, value in updates.items():
        setattr(toy.p, name, value)
    old_state = toy.committed.state

    with pytest.raises(Failure, match=match):
        toy.run()

    assert toy.committed.state is old_state
    assert toy.compute.sync_calls == 0


def _waveform_case(
    root,
    *,
    rtol=1.0e-4,
    max_iterations=12,
    include_controls=True,
    coupling="waveform",
    micro_steps=2,
):
    from test_model_routing import _context, _state, _write_case

    context = _context(inertia=0, integrator="rk4", nsteps=1, hard=True)
    state = _state(context)
    case = _write_case(
        root,
        context.p,
        context.geometry,
        state,
        "rk4",
        1,
        coupling=coupling,
        micro_steps=micro_steps,
    )
    text = case.read_text(encoding="utf-8")
    if include_controls:
        marker = next(
            line for line in text.splitlines() if line.startswith("dt_guard=")
        )
        text = text.replace(
            marker,
            f"waveform_rtol={rtol!r}\n"
            f"waveform_max_iterations={max_iterations}\n"
            f"{marker}",
            1,
        )
        case.write_text(text, encoding="utf-8")
        source_path = case.parent / "source.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source["input_hashes"]["case"] = hashlib.sha256(
            case.read_bytes()
        ).hexdigest()
        source_path.write_text(json.dumps(source), encoding="utf-8")
    return case, context, state


def test_config_defaults_are_effective_and_semantically_equal(tmp_path):
    from fomhd.types import Parameters

    assert Parameters().waveform_rtol == 1.0e-10
    assert Parameters().waveform_max_iterations == 12

    omitted, _, _ = _waveform_case(
        tmp_path / "omitted", include_controls=False
    )
    explicit, _, _ = _waveform_case(
        tmp_path / "explicit", rtol=1.0e-10, max_iterations=12
    )
    left = io.load_case(omitted)
    right = io.load_case(explicit)
    assert left.parameters.coupling == "waveform"
    assert left.parameters.waveform_rtol == 1.0e-10
    assert left.parameters.waveform_max_iterations == 12
    assert right.parameters.waveform_rtol == 1.0e-10
    assert right.parameters.waveform_max_iterations == 12
    assert left.semantic_hash == right.semantic_hash


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("waveform_rtol", "true"),
        ("waveform_rtol", "0.0"),
        ("waveform_rtol", "-1.0"),
        ("waveform_rtol", "nan"),
        ("waveform_rtol", "inf"),
        ("waveform_rtol", '"1e-10"'),
        ("waveform_max_iterations", "true"),
        ("waveform_max_iterations", "1.5"),
        ("waveform_max_iterations", "0"),
        ("waveform_max_iterations", "-1"),
    ],
)
def test_config_rejects_invalid_waveform_controls(
    tmp_path, name, value
):
    case, _, _ = _waveform_case(tmp_path / name)
    text = case.read_text(encoding="utf-8")
    old = next(
        line for line in text.splitlines() if line.startswith(f"{name}=")
    )
    text = text.replace(old, f"{name}={value}", 1)
    case.write_text(text, encoding="utf-8")
    source_path = case.parent / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["input_hashes"]["case"] = hashlib.sha256(
        case.read_bytes()
    ).hexdigest()
    source_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="waveform"):
        io.load_case(case)


def test_waveform_controls_are_not_silently_accepted_for_legacy_coupling(
    tmp_path,
):
    case, _, _ = _waveform_case(
        tmp_path / "legacy",
        coupling="predictor",
        micro_steps=2,
        include_controls=False,
    )
    text = case.read_text(encoding="utf-8").replace(
        'dt_guard="warn"',
        'waveform_rtol=1.0e-10\n'
        "waveform_max_iterations=12\n"
        'dt_guard="warn"',
        1,
    )
    case.write_text(text, encoding="utf-8")
    source_path = case.parent / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["input_hashes"]["case"] = hashlib.sha256(
        case.read_bytes()
    ).hexdigest()
    source_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="waveform.*waveform coupling"):
        io.load_case(case)


def test_waveform_memory_counts_two_paths_and_one_retained_sweep():
    from test_time_predictor import _fake_memory_context

    context = _fake_memory_context(K=1, walls=[object(), object()])
    context.p.coupling = "synchronous"
    context.p.micro_steps = 1
    context.p.waveform_max_iterations = 12
    synchronous = core.estimated_peak_bytes(context)

    context.p.coupling = "predictor"
    context.p.micro_steps = 4
    predictor = core.estimated_peak_bytes(context)
    predictor_entries = core.predictor_history_entries(context)

    context.p.coupling = "waveform"
    context.p.waveform_rtol = 1.0e-10
    one = core.estimated_peak_bytes(context)
    waveform_entries = core.predictor_history_entries(context)
    context.p.waveform_max_iterations = 1
    one_iteration = core.estimated_peak_bytes(context)
    context.p.waveform_max_iterations = 64
    many_iterations = core.estimated_peak_bytes(context)

    assert synchronous < predictor < one
    assert one == one_iteration == many_iterations
    ion_stage_entries = (2 * context.p.K + 1) * 5 * context.op.Nc
    assert waveform_entries - predictor_entries == (
        2 * 4 * context.p.micro_steps * ion_stage_entries
        + 2 * ion_stage_entries
    )
    assert core.predictor_history_entries(context) > (
        core.predictor_wall_history_entries(context)
    )


def test_stability_guard_accepts_waveform_with_micro_horizon():
    from fomhd import stability

    p = SimpleNamespace(
        dt=0.4,
        nsteps=4,
        micro_steps=4,
        electron_inertia=0,
        integrator="rk4",
        coupling="waveform",
        dt_guard="warn",
        minimum_samples_per_cycle=32.0,
        max_amplitude_error=0.01,
        max_phase_error_rad=0.1,
    )
    context = SimpleNamespace(p=p)
    report = stability.assess_predictor_runtime(
        {"omega_estimate": 20.0}, context, step=1
    )
    assert report["coupling"] == "waveform"
    assert report["micro_horizon_steps"] == 12
    assert report["horizon_s"] == pytest.approx(1.2)
    assert "micro-RK4 only" in report["scope"]


def test_waveform_restart_records_and_validates_controls(tmp_path):
    from fomhd.__main__ import _run_command

    case, _, _ = _waveform_case(
        tmp_path / "waveform",
        rtol=1.0e-3,
        max_iterations=20,
    )
    output = tmp_path / "waveform-run"
    assert _run_command(["run", str(case), "--output", str(output)]) == 0
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert manifest["coupling"] == "waveform"
    assert manifest["waveform_rtol"] == 1.0e-3
    assert manifest["waveform_max_iterations"] == 20
    evidence = manifest["time_coupling_evidence"]
    assert evidence["coupling"] == "waveform"
    assert evidence["converged"] is True
    assert evidence["sweeps"] == len(evidence["iterations"])
    assert len(evidence["quadrature_weights"]) == 4 * evidence["micro_steps"]
    assert evidence["alias_evaluations"] == (
        2 * evidence["micro_steps"] * evidence["sweeps"]
    )
    assert evidence["total_model_evaluations"] == (
        evidence["complete_stage_evaluations"]
        + evidence["alias_evaluations"]
    )

    latest = json.loads(
        (output / "checkpoints/latest.json").read_text(encoding="utf-8")
    )
    checkpoint = output / "checkpoints" / latest["file"]
    arrays = io.read_npz(checkpoint)
    assert "waveform_rtol" in arrays
    assert "waveform_max_iterations" in arrays
    loaded = io.load_case(case)
    saved = io.read_restart(checkpoint, loaded)
    assert saved.state.step == 1

    changed, _, _ = _waveform_case(
        tmp_path / "changed",
        rtol=1.0e-4,
        max_iterations=20,
    )
    changed_case = io.load_case(changed)
    with pytest.raises(ValueError, match="waveform|semantic"):
        io.read_restart(checkpoint, changed_case)
