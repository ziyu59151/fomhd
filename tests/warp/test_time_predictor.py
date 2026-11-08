"""A1 T0 predictor scheduler, configuration and bounded massless checks."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from fomhd import core, diagnostics, io, stability
from fomhd import time as time_module
from fomhd.types import Failure, StageTag


def _toy_fluid(ion, electron):
    n, mom, perp = ion
    p_perp, p_parallel, magnetic = electron
    return SimpleNamespace(
        fluid=(
            SimpleNamespace(
                n=np.asarray([n], dtype=np.complex128),
                mom=np.asarray([mom], dtype=np.complex128),
                perp=np.asarray([perp], dtype=np.complex128),
                parallel=np.zeros(0, dtype=np.complex128),
            ),
            SimpleNamespace(
                n=np.zeros(1, dtype=np.complex128),
                mom=np.zeros(1, dtype=np.complex128),
                perp=np.asarray([p_perp], dtype=np.complex128),
                parallel=np.asarray([p_parallel], dtype=np.complex128),
            ),
        ),
        B=np.asarray([magnetic], dtype=np.complex128),
        step=0,
        generation=0,
        time=0.0,
    )


def _set_identity(state, tag):
    state.step = tag.step
    state.generation = tag.generation
    state.time = tag.time
    return state


def _parts(state):
    return (
        (
            float(np.real(state.fluid[0].n[0])),
            float(np.real(state.fluid[0].mom[0])),
            float(np.real(state.fluid[0].perp[0])),
        ),
        (
            float(np.real(state.fluid[1].perp[0])),
            float(np.real(state.fluid[1].parallel[0])),
            float(np.real(state.B[0])),
        ),
    )


def _toy_stage(rhs, tag, source_id):
    rhs = _set_identity(rhs, tag)
    ledger_names = (
        "electric_node",
        "electric_edge",
        "pressure_kinetic",
        "pressure_internal",
        "magnetic",
        "wall",
        "convection",
        "perp_rate",
        "parallel_rate",
        "spectral_kinetic",
        "spectral_perp",
        "spectral_parallel",
    )
    ledger = SimpleNamespace(**{name: 1.0 for name in ledger_names})
    return SimpleNamespace(
        tag=tag,
        rhs=rhs,
        cache=SimpleNamespace(tag=tag, E=np.zeros(1)),
        checks=[],
        ledger=[ledger, ledger],
        boundary=SimpleNamespace(
            number=[0.0, 0.0],
            kinetic=[0.0, 0.0],
            internal_perp=[0.0, 0.0],
            internal_parallel=[0.0, 0.0],
            field_energy=0.0,
            total_energy=0.0,
        ),
        momentum=SimpleNamespace(
            value=[0.0, 0.0, 0.0],
            outward=[0.0, 0.0, 0.0],
            wall_reaction=[0.0, 0.0, 0.0],
            angular=0.0,
            outward_torque=0.0,
            wall_torque=0.0,
            angular_samples=1,
        ),
        fluid=[
            SimpleNamespace(
                flux_hat=np.zeros(1),
                divergence_hat=np.zeros(1),
            )
            for _ in range(2)
        ],
        totals=SimpleNamespace(
            number=np.zeros(2),
            kinetic=np.zeros(2),
            ion_internal=0.0,
            electron_perp=0.0,
            electron_parallel=0.0,
            magnetic=0.0,
            longitudinal=0.0,
            topology=0.0,
            physical_flux=0.0,
            energy=0.0,
        ),
        source_id=source_id,
    )


def _cumulative():
    ledger_names = (
        "electric_node",
        "electric_edge",
        "pressure_kinetic",
        "pressure_internal",
        "magnetic",
        "wall",
        "convection",
        "perp_rate",
        "parallel_rate",
        "spectral_kinetic",
        "spectral_perp",
        "spectral_parallel",
    )
    return SimpleNamespace(
        baseline=SimpleNamespace(divB=np.zeros(1), step=0, time=0.0),
        integrals=[
            SimpleNamespace(**{name: 0.0 for name in ledger_names})
            for _ in range(2)
        ],
        boundary_integrals=SimpleNamespace(
            number=[0.0, 0.0],
            kinetic=[0.0, 0.0],
            internal_perp=[0.0, 0.0],
            internal_parallel=[0.0, 0.0],
            field_energy=0.0,
            total_energy=0.0,
        ),
        accepted_intervals=0,
        last_trial=0,
        last_step=0,
        last_time=0.0,
    )


class _ToyCompute:
    name = "cpu"
    xp = np

    def __init__(self):
        self.sync_calls = 0
        self.synchronized = False
        self.axpy_calls_after_sync = 0

    def axpy(self, base, rhs, weight):
        if self.synchronized:
            self.axpy_calls_after_sync += 1
        return np.asarray(base) + weight * np.asarray(rhs)

    def array(self, value, dtype=None):
        return np.asarray(value, dtype=dtype)

    def max_abs(self, value):
        return float(np.max(np.abs(value)))

    def synchronize(self):
        self.sync_calls += 1
        self.synchronized = True


class ToyPredictor:
    def __init__(self, monkeypatch, *, H=0.2, M=2, rhs):
        self.monkeypatch = monkeypatch
        self.H = H
        self.M = M
        self.rhs = rhs
        self.evaluated = []
        self.stages = []
        self.assembled = []
        self.check_calls = []
        self.retained = []
        self.source_id = 0
        self.p = SimpleNamespace(
            start_time=0.0,
            dt=H,
            nsteps=1,
            electron_inertia=0,
            integrator="rk4",
            coupling="predictor",
            micro_steps=M,
            dt_guard="warn",
            minimum_samples_per_cycle=32.0,
            max_amplitude_error=0.01,
            max_phase_error_rad=0.1,
            species=(
                SimpleNamespace(id="i", charge=1.0, mass=2.0, n_min=0.0),
                SimpleNamespace(id="e", charge=-1.0, mass=1.0, n_min=0.0),
            ),
            ref=SimpleNamespace(ni=1.0, ne=1.0, velocity=1.0, pressure=1.0),
        )
        self.compute = _ToyCompute()
        self.context = SimpleNamespace(
            p=self.p,
            compute=self.compute,
            next_generation=2,
            next_trial=1,
            next_source=1,
            spectral_id=0,
            last_dt_report=None,
        )
        initial = _set_identity(_toy_fluid((1.0, 2.0, 3.0), (0.2, 0.3, 0.4)),
                                StageTag(step=0, generation=1, time=0.0))
        initial.accepted = True
        self.committed = SimpleNamespace(
            state=initial,
            cache=SimpleNamespace(),
            accepted=True,
        )
        self.cumulative = _cumulative()
        from fomhd import time_predictor

        self.module = time_predictor

        def assemble(context, ion, electron, tag):
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
            self.source_id += 1
            self.evaluated.append((state, tag, alias))
            ion, electron = _parts(state)
            f_ion, g_electron = self.rhs(tag.time, ion, electron)
            derivative = _toy_fluid(f_ion, g_electron)
            stage = _toy_stage(derivative, tag, self.source_id)
            self.stages.append(stage)
            return stage

        def check_step(left, base, midpoint_state, midpoint, right, end,
                       cumulative, context, prescribed_source=None, *,
                       quadrature=None):
            self.check_calls.append(quadrature)
            return SimpleNamespace(
                checks=[],
                totals=end.totals,
                momentum=end.momentum,
                interval=SimpleNamespace(
                    accepted=False,
                    weight=H,
                    number_flux=[np.zeros(1), np.zeros(1)],
                    current=[np.zeros(1), np.zeros(1)],
                ),
            )

        monkeypatch.setattr(
            self.module, "evaluate_stage", evaluate, raising=False
        )
        monkeypatch.setattr(
            self.module, "_assemble_state", assemble, raising=False
        )
        monkeypatch.setattr(
            self.module, "_check_guard", lambda stage, context: None,
            raising=False,
        )
        monkeypatch.setattr(
            self.module,
            "retain_failure_evidence",
            lambda failure, tag, context, state, result, prefix:
            self.retained.append((prefix, tag, state)),
            raising=False,
        )
        monkeypatch.setattr(
            core, "preflight_memory", lambda context: None, raising=False
        )
        monkeypatch.setattr(diagnostics, "check_step", check_step)
        monkeypatch.setattr(diagnostics, "gate", lambda *args, **kwargs: None)

    def run(self):
        return time_module.advance(
            self.committed, self.cumulative, self.context
        )


def _constant_rhs(time, ion, electron):
    del time
    return (1.0, 2.0, 3.0), (0.0, 0.0, 0.0)


def _feedback_rhs(time, ion, electron):
    del time
    n, mom, perp = ion
    pe, pp, b = electron
    return (
        (1.0 + 0.5 * pe, 1.0 + mom - 0.25 * pp, 1.0 + perp + 0.2 * b),
        (n - 0.2 * pe, mom + 0.3 * pp, b + 0.1 * n),
    )


def _synchronous_feedback(h, ion, electron):
    def derivative(values):
        return _feedback_rhs(0.0, values[:3], values[3:])

    values = np.asarray(ion + electron, dtype=np.float64)
    k1 = np.asarray(derivative(values)[0] + derivative(values)[1])
    k2 = np.asarray(
        derivative(values + 0.5 * h * k1)[0]
        + derivative(values + 0.5 * h * k1)[1]
    )
    k3 = np.asarray(
        derivative(values + 0.5 * h * k2)[0]
        + derivative(values + 0.5 * h * k2)[1]
    )
    k4 = np.asarray(
        derivative(values + h * k3)[0]
        + derivative(values + h * k3)[1]
    )
    return values + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def test_stage_schedule_weights_identity_and_single_commit(monkeypatch):
    toy = ToyPredictor(monkeypatch, H=0.2, M=2, rhs=_constant_rhs)
    result = toy.run()
    quadrature = toy.check_calls[0]

    assert len(toy.evaluated) == 4 * toy.M + 1
    assert len(quadrature) == 4 * toy.M
    assert len(toy.check_calls) == 1
    h = toy.H / toy.M
    np.testing.assert_allclose(
        [item[2] for item in quadrature],
        [h / 6, h / 3, h / 3, h / 6] * toy.M,
        rtol=0.0,
        atol=2.0e-17,
    )
    tags = [item[1].tag for item in quadrature]
    assert [tag.micro_step for tag in tags] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert len({item[1].generation for item in toy.evaluated}) == 4 * toy.M + 1
    assert len({item[1].source_id for item in quadrature}) == 4 * toy.M
    assert toy.evaluated[0][1].stage == "base"
    assert toy.evaluated[1][1].stage == "predictor_m000_r2"
    assert toy.evaluated[4][1].stage == "predictor_m001_r1"
    assert toy.committed.state.step == 1
    assert toy.cumulative.accepted_intervals == 1
    assert result.interval.accepted is True
    assert result.coupling_metadata["micro_steps"] == toy.M
    assert result.coupling_metadata["complete_stage_evaluations"] == 4 * toy.M + 1
    assert result.coupling_metadata["alias_evaluations"] == 2 * toy.M
    assert result.coupling_metadata["total_model_evaluations"] == 6 * toy.M + 1
    assert toy.compute.sync_calls == 1
    assert toy.compute.axpy_calls_after_sync == 0
    assert toy.cumulative.integrals[0].electric_edge == pytest.approx(toy.H)
    assert toy.cumulative.integrals[1].electric_edge == pytest.approx(toy.H)
    correction = result.coupling_metadata["endpoint_tangent_correction"]
    assert set(correction) == {"n_i", "mom_i", "perp_i"}
    assert all(
        math.isfinite(item["relative_max_abs"])
        for item in correction.values()
    )


def test_evidence_slot_lookup_is_linear_in_stage_count(monkeypatch):
    original_equal = StageTag.__eq__
    comparisons = 0

    def counted_equal(left, right):
        nonlocal comparisons
        comparisons += 1
        return original_equal(left, right)

    monkeypatch.setattr(StageTag, "__eq__", counted_equal)
    toy = ToyPredictor(monkeypatch, H=0.5, M=6, rhs=_constant_rhs)
    toy.run()
    assert comparisons <= 4 * toy.M + 1


def test_fixed_tangent_path_and_distinct_same_time_k2_k3(monkeypatch):
    toy = ToyPredictor(monkeypatch, H=0.3, M=2, rhs=_feedback_rhs)
    toy.run()
    same_time = [
        state
        for state, tag, _ in toy.evaluated
        if math.isclose(tag.time, 0.075, rel_tol=0.0, abs_tol=1.0e-16)
    ]
    assert len(same_time) == 2
    assert same_time[0].fluid[1].perp[0] != same_time[1].fluid[1].perp[0]
    assert same_time[0].B[0] != same_time[1].B[0]

    f0 = np.asarray(
        _feedback_rhs(0.0, (1.0, 2.0, 3.0), (0.2, 0.3, 0.4))[0]
    )
    p0 = np.asarray([1.0, 2.0, 3.0])
    for state, tag, _ in toy.evaluated:
        if tag.stage == "base" or tag.micro_step < 0:
            continue
        expected = p0 + tag.time * f0
        actual = np.asarray([
            np.real(state.fluid[0].n[0]),
            np.real(state.fluid[0].mom[0]),
            np.real(state.fluid[0].perp[0]),
        ])
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)


def test_non_power_two_clock_and_nonzero_start_share_tangent_fraction(
    monkeypatch,
):
    toy = ToyPredictor(monkeypatch, H=0.7, M=3, rhs=_feedback_rhs)
    toy.p.start_time = 0.125
    toy.committed.state.time = 0.125
    toy.cumulative.last_time = 0.125
    toy.run()
    f0 = np.asarray(
        _feedback_rhs(
            0.125,
            (1.0, 2.0, 3.0),
            (0.2, 0.3, 0.4),
        )[0]
    )
    p0 = np.asarray([1.0, 2.0, 3.0])
    for state, tag, _ in toy.evaluated:
        if tag.micro_step < 0 or tag.stage == "base":
            continue
        offset = core.microstage_offset(
            toy.p,
            tag.micro_step,
            (
                0
                if tag.stage.endswith("r1")
                else int(tag.stage.rsplit("r", 1)[1]) - 1
            ),
        )
        expected = p0 + offset * f0
        actual = np.asarray([
            np.real(state.fluid[0].n[0]),
            np.real(state.fluid[0].mom[0]),
            np.real(state.fluid[0].perp[0]),
        ])
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3.0e-15)

    same_time = {}
    for _, tag, _ in toy.evaluated:
        same_time.setdefault(tag.time, []).append(tag)
    for tags in same_time.values():
        k2 = [tag for tag in tags if tag.stage.endswith("r2")]
        k3 = [tag for tag in tags if tag.stage.endswith("r3")]
        if k2 and k3:
            assert k2[0].generation != k3[0].generation
            assert k2[0].spectral_context == k3[0].spectral_context


def test_predictor_m1_is_not_synchronous_full_state_rk4(monkeypatch):
    toy = ToyPredictor(monkeypatch, H=0.1, M=1, rhs=_feedback_rhs)
    result = toy.run()
    state = toy.committed.state
    actual = np.asarray([
        np.real(state.fluid[0].n[0]),
        np.real(state.fluid[0].mom[0]),
        np.real(state.fluid[0].perp[0]),
        np.real(state.fluid[1].perp[0]),
        np.real(state.fluid[1].parallel[0]),
        np.real(state.B[0]),
    ])
    expected = _synchronous_feedback(
        0.1,
        (1.0, 2.0, 3.0),
        (0.2, 0.3, 0.4),
    )
    assert np.max(np.abs(actual - expected)) > 1.0e-6
    assert result.coupling_metadata["full_rhs_stage_evaluations"] == 4


def test_z_rk4_and_ion_quadrature_use_the_same_stage_weights(monkeypatch):
    toy = ToyPredictor(monkeypatch, H=0.2, M=1, rhs=_feedback_rhs)
    toy.run()
    h = 0.2
    z0 = np.asarray([0.2, 0.3, 0.4])
    derivatives = [
        np.asarray([
            np.real(stage.rhs.fluid[1].perp[0]),
            np.real(stage.rhs.fluid[1].parallel[0]),
            np.real(stage.rhs.B[0]),
        ])
        for stage in toy.stages
    ]
    assembled_z = [
        np.asarray([
            np.real(item[1][0]),
            np.real(item[1][1]),
            np.real(item[1][2]),
        ])
        for item in toy.assembled
    ]
    np.testing.assert_allclose(
        assembled_z[0], z0 + 0.5 * h * derivatives[0]
    )
    np.testing.assert_allclose(
        assembled_z[1], z0 + 0.5 * h * derivatives[1]
    )
    np.testing.assert_allclose(
        assembled_z[2], z0 + h * derivatives[2]
    )
    z_end = np.asarray([
        np.real(toy.committed.state.fluid[1].perp[0]),
        np.real(toy.committed.state.fluid[1].parallel[0]),
        np.real(toy.committed.state.B[0]),
    ])
    np.testing.assert_allclose(
        z_end,
        z0 + h * sum(
            weight * derivative
            for weight, derivative in zip(
                (1 / 6, 2 / 6, 2 / 6, 1 / 6),
                derivatives,
            )
        ),
    )
    ion0 = np.asarray([1.0, 2.0, 3.0])
    ion_end = np.asarray([
        np.real(toy.committed.state.fluid[0].n[0]),
        np.real(toy.committed.state.fluid[0].mom[0]),
        np.real(toy.committed.state.fluid[0].perp[0]),
    ])
    ion_derivatives = [
        np.asarray([
            np.real(stage.rhs.fluid[0].n[0]),
            np.real(stage.rhs.fluid[0].mom[0]),
            np.real(stage.rhs.fluid[0].perp[0]),
        ])
        for stage in toy.stages
    ]
    np.testing.assert_allclose(
        ion_end,
        ion0 + h * sum(
            weight * derivative
            for weight, derivative in zip(
                (1 / 6, 2 / 6, 2 / 6, 1 / 6),
                ion_derivatives,
            )
        ),
    )


@pytest.mark.parametrize("micro_steps", [8, 16, 32])
def test_oscillatory_ion_impulse_cancels_without_endpoint_shortcut(
    monkeypatch, micro_steps
):
    omega = 2 * math.pi
    H = 1.0

    def oscillatory(clock, ion, electron):
        del ion, electron
        value = math.cos(omega * clock)
        return (value, 2.0 * value, 3.0 * value), (0.0, 0.0, 0.0)

    toy = ToyPredictor(monkeypatch, H=H, M=micro_steps, rhs=oscillatory)
    toy.run()
    endpoint = np.asarray([
        np.real(toy.committed.state.fluid[0].n[0]),
        np.real(toy.committed.state.fluid[0].mom[0]),
        np.real(toy.committed.state.fluid[0].perp[0]),
    ])
    correction = endpoint - np.asarray([1.0, 2.0, 3.0])
    endpoint_shortcut = H * np.asarray([1.0, 2.0, 3.0])
    floating_scale = (
        1024.0
        * np.finfo(np.float64).eps
        * micro_steps
        * np.max(endpoint_shortcut)
    )
    assert np.max(np.abs(correction)) <= floating_scale


def test_noninteger_cycle_impulse_has_micro_rk4_fourth_order(monkeypatch):
    H = 1.0
    omega = 2.0 * math.pi * 0.73
    exact_integral = (1.0 - math.cos(omega * H)) / omega

    def forcing(clock, ion, electron):
        del ion, electron
        value = math.sin(omega * clock)
        return (value, 2.0 * value, 3.0 * value), (0.0, 0.0, 0.0)

    errors = []
    for micro_steps in (4, 8, 16, 32):
        toy = ToyPredictor(
            monkeypatch,
            H=H,
            M=micro_steps,
            rhs=forcing,
        )
        toy.run()
        endpoint = np.asarray([
            np.real(toy.committed.state.fluid[0].n[0]),
            np.real(toy.committed.state.fluid[0].mom[0]),
            np.real(toy.committed.state.fluid[0].perp[0]),
        ])
        exact = np.asarray([1.0, 2.0, 3.0]) + np.asarray(
            [1.0, 2.0, 3.0]
        ) * exact_integral
        errors.append(float(np.max(np.abs(endpoint - exact))))

    orders = [
        math.log(errors[index] / errors[index + 1], 2.0)
        for index in range(1, len(errors) - 1)
    ]
    assert errors[-1] < errors[0]
    assert all(3.5 < order < 4.5 for order in orders), (errors, orders)


def test_configuration_defaults_are_semantically_equal(tmp_path):
    from test_model_routing import _context, _state, _write_case

    context = _context(inertia=0, integrator="rk4", nsteps=0, hard=True)
    state = _state(context)
    omitted = _write_case(
        tmp_path / "omitted",
        context.p,
        context.geometry,
        state,
        "rk4",
        0,
    )
    explicit = _write_case(
        tmp_path / "explicit",
        context.p,
        context.geometry,
        state,
        "rk4",
        0,
        coupling="synchronous",
        micro_steps=1,
    )
    left = io.load_case(omitted)
    right = io.load_case(explicit)
    assert left.parameters.coupling == "synchronous"
    assert left.parameters.micro_steps == 1
    assert right.parameters.coupling == "synchronous"
    assert right.parameters.micro_steps == 1
    assert left.semantic_hash == right.semantic_hash


@pytest.mark.parametrize(
    ("inertia", "integrator", "coupling", "micro_steps", "match"),
    [
        (0, "rk4", "synchronous", 2, "synchronous.*micro_steps"),
        (1, "rk4", "predictor", 2, "predictor.*massless"),
        (0, "midpoint", "predictor", 2, "predictor.*rk4"),
        (0, "rk4", "unknown", 1, "unknown.*coupling"),
        (0, "rk4", "predictor", True, "micro_steps"),
        (0, "rk4", "predictor", 1.5, "micro_steps"),
        (0, "rk4", "predictor", 0, "micro_steps"),
        (0, "rk4", "predictor", -1, "micro_steps"),
        (0, "rk4", "predictor", "2", "micro_steps"),
    ],
)
def test_invalid_coupling_configuration_rejected(
    tmp_path, inertia, integrator, coupling, micro_steps, match
):
    from test_model_routing import _context, _state, _write_case

    context = _context(inertia=inertia, integrator=integrator)
    case = _write_case(
        tmp_path / "invalid",
        context.p,
        context.geometry,
        _state(context),
        integrator,
        1,
        coupling=coupling,
        micro_steps=micro_steps,
    )
    with pytest.raises(ValueError, match=match):
        io.load_case(case)


def test_direct_invalid_and_source_rejection_precede_mutation(monkeypatch):
    toy = ToyPredictor(monkeypatch, H=0.1, M=2, rhs=_constant_rhs)
    toy.p.coupling = "synchronous"
    old_trial = toy.context.next_trial
    old_state = toy.committed.state
    with pytest.raises(Failure, match="synchronous.*micro_steps"):
        time_module.advance(toy.committed, toy.cumulative, toy.context)
    assert toy.context.next_trial == old_trial
    assert toy.committed.state is old_state

    toy.p.coupling = "predictor"
    callback_calls = []

    def source(*args):
        callback_calls.append(args)
        raise AssertionError("source callback was invoked")

    old_trial = toy.context.next_trial
    with pytest.raises(Failure, match="source"):
        time_module.advance(
            toy.committed, toy.cumulative, toy.context, source=source
        )
    assert callback_calls == []
    assert toy.context.next_trial == old_trial
    assert toy.committed.state is old_state


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"electron_inertia": 1}, "predictor.*massless"),
        ({"integrator": "midpoint"}, "predictor.*rk4"),
        ({"micro_steps": True}, "micro_steps"),
        ({"micro_steps": 1.5}, "micro_steps"),
        ({"micro_steps": 0}, "micro_steps"),
        ({"micro_steps": -1}, "micro_steps"),
    ],
)
def test_direct_invalid_predictor_controls_precede_mutation(
    monkeypatch, updates, match
):
    toy = ToyPredictor(monkeypatch, H=0.1, M=2, rhs=_constant_rhs)
    for name, value in updates.items():
        setattr(toy.p, name, value)
    old_state = toy.committed.state
    old_trial = toy.context.next_trial
    old_generation = toy.context.next_generation
    with pytest.raises(Failure, match=match):
        time_module.advance(toy.committed, toy.cumulative, toy.context)
    assert toy.context.next_trial == old_trial
    assert toy.context.next_generation == old_generation
    assert toy.committed.state is old_state


@pytest.mark.parametrize(
    ("failure_stage", "synchronize_failure"),
    [
        ("predictor_m001_r3", False),
        ("endpoint", False),
        ("", True),
    ],
)
def test_late_endpoint_and_synchronize_failure_rollback(
    monkeypatch, failure_stage, synchronize_failure
):
    toy = ToyPredictor(monkeypatch, H=0.2, M=2, rhs=_constant_rhs)
    old_state = toy.committed.state
    old_last_trial = toy.cumulative.last_trial
    old_last_step = toy.cumulative.last_step

    if failure_stage:
        original = toy.module.evaluate_stage

        def failing(state, tag, context, alias=False):
            if tag.stage == failure_stage:
                failure = Failure("Domain", "injected stage failure")
                failure.stage = tag
                raise failure
            return original(state, tag, context, alias)

        monkeypatch.setattr(toy.module, "evaluate_stage", failing)
    if synchronize_failure:
        def fail_sync():
            raise RuntimeError("injected synchronize failure")

        monkeypatch.setattr(toy.compute, "synchronize", fail_sync)

    with pytest.raises(Failure):
        toy.run()
    assert toy.committed.state is old_state
    assert toy.cumulative.last_trial == old_last_trial
    assert toy.cumulative.last_step == old_last_step
    assert toy.cumulative.accepted_intervals == 0
    assert toy.retained
    assert len(toy.retained) >= 4 * toy.M + 1


def test_interval_check_failure_preserves_completed_and_not_run_evidence(
    monkeypatch,
):
    toy = ToyPredictor(monkeypatch, H=0.2, M=2, rhs=_constant_rhs)
    old_state = toy.committed.state

    def reject_interval(*args, **kwargs):
        failure = Failure("Time", "injected interval failure")
        failure.stage = toy.evaluated[-1][1]
        raise failure

    monkeypatch.setattr(diagnostics, "check_step", reject_interval)
    with pytest.raises(Failure, match="interval"):
        toy.run()
    assert toy.committed.state is old_state
    assert toy.cumulative.accepted_intervals == 0
    completed = [item for item in toy.retained if item[2] is not None]
    assert len(completed) >= 4 * toy.M + 1


@pytest.mark.parametrize(
    (
        "failure_kind",
        "expected_code",
        "expected_stage",
        "expected_step",
        "expected_micro",
    ),
    [
        (
            "intermediate",
            "Magnetization",
            "predictor_m000_r2",
            0,
            0,
        ),
        ("endpoint", "Magnetization", "endpoint", 1, -1),
        ("endpoint-library", "NumericalLibrary", "endpoint", 1, -1),
    ],
)
def test_assembly_canonicalization_failure_preserves_candidate_and_tag(
    monkeypatch,
    failure_kind,
    expected_code,
    expected_stage,
    expected_step,
    expected_micro,
):
    from fomhd import time_predictor
    from fomhd.stage import initialize
    from test_massless_support import context as massless_context, uniform_state

    context = massless_context(K=0, hard_diagnostics=False)
    context.p.coupling = "predictor"
    context.p.micro_steps = 2
    context.p.integrator = "rk4"
    context.p.dt_guard = "warn"
    context.p.nsteps = 1
    cumulative = _cumulative()
    committed = initialize(
        uniform_state(context),
        context,
        cumulative,
        False,
    )
    old_state = committed.state
    old_cache = committed.cache
    original_canonicalize = time_predictor.canonicalize_state
    captured = []
    intermediate_time = core.microstage_time(context.p, 0, 0, 1)
    endpoint_time = core.time_at_step(context.p, 1)

    def fail_at_target(raw, active_context):
        target = (
            raw.step == 0
            and math.isclose(
                raw.time,
                intermediate_time,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            if failure_kind == "intermediate"
            else raw.step == 1
            and math.isclose(
                raw.time,
                endpoint_time,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        )
        if target:
            captured.append(raw)
        if target and failure_kind == "endpoint-library":
            raise RuntimeError("injected canonicalization library failure")
        if target:
            raw.B.fill(0.0)
        return original_canonicalize(raw, active_context)

    monkeypatch.setattr(
        time_predictor,
        "canonicalize_state",
        fail_at_target,
    )
    with pytest.raises(Failure) as caught:
        time_module.advance(committed, cumulative, context)

    assert captured
    raw = captured[0]
    failure = caught.value
    assert failure.code == expected_code
    assert failure.stage is not None
    assert failure.stage.stage == expected_stage
    assert failure.stage.step == expected_step
    assert failure.stage.micro_step == expected_micro
    assert failure.stage.generation == raw.generation
    assert failure.stage.time == raw.time
    assert tuple(raw.fluid[0].n.shape) == tuple(old_state.fluid[0].n.shape)
    expected_arrays = {
        "candidate_B": raw.B,
        "candidate_n_i": raw.fluid[0].n,
        "candidate_momentum_i": raw.fluid[0].mom,
        "candidate_p_i": raw.fluid[0].perp,
        "candidate_n_e": raw.fluid[1].n,
        "candidate_momentum_e": raw.fluid[1].mom,
        "candidate_p_e": raw.fluid[1].perp,
        "candidate_p_parallel_e": raw.fluid[1].parallel,
    }
    for key, expected in expected_arrays.items():
        assert key in failure.partial_arrays
        np.testing.assert_array_equal(failure.partial_arrays[key], expected)
    if failure_kind == "endpoint-library":
        assert np.max(np.abs(raw.B)) > 0.0
    else:
        np.testing.assert_array_equal(raw.B, 0.0)

    assert committed.state is old_state
    assert committed.cache is old_cache
    assert cumulative.accepted_intervals == 0
    assert cumulative.last_step == 0
    assert cumulative.last_time == context.p.start_time


def test_microstage_clocks_use_integer_fraction_and_endpoint_clock():
    p = SimpleNamespace(
        start_time=0.1,
        dt=0.8,
        micro_steps=4,
    )
    t_k1 = core.microstage_time(p, 0, 0, 0)
    t_k2 = core.microstage_time(p, 0, 0, 1)
    t_k3 = core.microstage_time(p, 0, 0, 2)
    t_k4 = core.microstage_time(p, 0, 0, 3)
    t_next = core.microstage_time(p, 0, 1, 0)
    assert t_k1 == core.time_at_step(p, 0)
    assert t_k2 == t_k3
    assert t_k4 == t_next
    assert t_k1 < t_k2 < t_k4 < core.time_at_step(p, 1)

    unrepresentable = SimpleNamespace(
        start_time=1.0e16,
        dt=1.0,
        micro_steps=4,
    )
    with pytest.raises(Failure, match="represent"):
        core.microstage_time(unrepresentable, 0, 0, 1)


def test_predictor_guard_uses_micro_horizon_and_labels_scope():
    p = SimpleNamespace(
        dt=0.4,
        nsteps=4,
        micro_steps=4,
        electron_inertia=0,
        integrator="rk4",
        coupling="predictor",
        dt_guard="warn",
        minimum_samples_per_cycle=32.0,
        max_amplitude_error=0.01,
        max_phase_error_rad=0.1,
    )
    context = SimpleNamespace(p=p)
    report = stability.assess_predictor_runtime(
        {"omega_estimate": 20.0}, context, step=1
    )
    assert report["horizon_s"] == pytest.approx(1.2)
    assert report["micro_horizon_steps"] == 12
    assert report["omega_dt"] == pytest.approx(20.0 * 0.1)
    assert report["micro_steps"] == 4
    assert "micro-RK4 only" in report["scope"]
    assert "macro-coupling" in report["scope"]


def test_predictor_guard_error_leaves_massless_commit_unchanged():
    from fomhd.stage import initialize
    from test_massless_support import context as massless_context, uniform_state

    context = massless_context(K=0, hard_diagnostics=False)
    context.p.coupling = "predictor"
    context.p.micro_steps = 2
    context.p.integrator = "rk4"
    context.p.dt_guard = "error"
    context.p.minimum_samples_per_cycle = 1.0e12
    context.p.nsteps = 1
    cumulative = _cumulative()
    committed = initialize(
        uniform_state(context),
        context,
        cumulative,
        False,
    )
    state = committed.state
    with pytest.raises(Failure) as caught:
        time_module.advance(committed, cumulative, context)
    assert caught.value.code == "TimeStepGuard"
    assert committed.state is state
    assert cumulative.accepted_intervals == 0
    report = caught.value.time_step_report
    assert report["coupling"] == "predictor"
    assert report["micro_steps"] == 2
    assert report["micro_horizon_steps"] == 2
    assert report["horizon_s"] == pytest.approx(context.p.dt)
    assert "micro-RK4 only" in report["scope"]


def test_unrepresentable_micro_clock_precedes_counter_mutation(monkeypatch):
    toy = ToyPredictor(monkeypatch, H=1.0, M=2, rhs=_constant_rhs)
    toy.p.start_time = 1.0e16
    toy.p.dt = 1.0
    toy.committed.state.time = 1.0e16
    toy.cumulative.last_time = 1.0e16
    old_trial = toy.context.next_trial
    old_generation = toy.context.next_generation
    with pytest.raises(Failure, match="representable"):
        time_module.advance(toy.committed, toy.cumulative, toy.context)
    assert toy.context.next_trial == old_trial
    assert toy.context.next_generation == old_generation
    assert toy.cumulative.accepted_intervals == 0


def test_memory_preflight_grows_with_retained_micro_history():
    from test_massless_support import context as massless_context

    context = massless_context(K=0, hard_diagnostics=False)
    context.p.coupling = "synchronous"
    context.p.micro_steps = 1
    synchronous = core.estimated_peak_bytes(context)
    context.p.coupling = "predictor"
    context.p.micro_steps = 2
    two = core.estimated_peak_bytes(context)
    context.p.micro_steps = 4
    four = core.estimated_peak_bytes(context)
    assert synchronous < two < four

    context.p.micro_steps = 1 << 63
    with pytest.raises(Failure, match="memory"):
        core.estimated_peak_bytes(context)


def _fake_memory_context(*, K, walls, inertia=0, memory_limit=1 << 40):
    empty_matrix = np.zeros((0, 0))
    vector = np.zeros(0)
    return SimpleNamespace(
        p=SimpleNamespace(
            K=K,
            electron_inertia=inertia,
            coupling="synchronous",
            micro_steps=1,
            memory_limit=memory_limit,
            device_memory_limit=1 << 40,
        ),
        compute=SimpleNamespace(name="cpu"),
        N=1,
        geometry=SimpleNamespace(walls=walls),
        op=SimpleNamespace(
            Ne=3,
            Nc=2,
            Nb=4,
            Bc=empty_matrix,
            Bc_full=vector,
            G=[empty_matrix] * (2 * K + 1),
            C=[empty_matrix] * (2 * K + 1),
            DB=[empty_matrix] * (2 * K + 1),
            DJ=[empty_matrix] * (2 * K + 1),
            Cdag=[empty_matrix] * (2 * K + 1),
            Lphi=[],
            I=vector,
            H=vector,
            QB=vector,
            T=vector,
            T_full=vector,
            PR_r=vector,
            PZ_r=vector,
            PR_z=vector,
            PZ_z=vector,
            PR_v=vector,
            PZ_v=vector,
            MB=np.ones(4),
        ),
    )


def test_fixed_wall_cache_bound_grows_with_modes_and_walls():
    K0W2 = _fake_memory_context(K=0, walls=[object(), object()])
    K1W2 = _fake_memory_context(K=1, walls=[object(), object()])
    K0W4 = _fake_memory_context(K=0, walls=[object()] * 4)
    no_walls = _fake_memory_context(K=0, walls=[])

    bound_0w2 = core.fixed_wall_cache_bytes(K0W2)
    bound_1w2 = core.fixed_wall_cache_bytes(K1W2)
    bound_0w4 = core.fixed_wall_cache_bytes(K0W4)
    assert core.fixed_wall_cache_bytes(no_walls) == 0
    assert 0 < bound_0w2 < bound_1w2
    assert 0 < bound_0w2 < bound_0w4
    assert bound_1w2 == 3 * bound_0w2

    base = core.estimated_peak_bytes(no_walls)
    with_walls = core.estimated_peak_bytes(K0W2)
    setup_temporaries = 16 * (4 * 2 + 3 * 2 + 5 * 2 * 2)
    assert with_walls == base + bound_0w2 + setup_temporaries

    finite = _fake_memory_context(
        K=1, walls=[object()] * 4, inertia=1
    )
    assert core.fixed_wall_cache_bytes(finite) == 0

    K0W2.p.memory_limit = with_walls - 1
    with pytest.raises(Failure, match="memory"):
        core.preflight_memory(K0W2)


def test_retained_predictor_wall_samples_scale_with_stages_and_walls():
    no_walls = _fake_memory_context(K=0, walls=[])
    one_wall = _fake_memory_context(K=0, walls=[object()])
    two_walls = _fake_memory_context(K=0, walls=[object(), object()])
    four_walls = _fake_memory_context(K=0, walls=[object()] * 4)
    for context in (no_walls, one_wall, two_walls, four_walls):
        context.p.coupling = "predictor"
        context.p.micro_steps = 1

    entries = [
        core.predictor_wall_history_entries(context)
        for context in (no_walls, one_wall, two_walls, four_walls)
    ]
    assert entries[0] == 0
    assert 0 < entries[1] < entries[2] < entries[3]

    two_walls_two_micro = _fake_memory_context(
        K=0, walls=[object(), object()]
    )
    two_walls_two_micro.p.coupling = "predictor"
    two_walls_two_micro.p.micro_steps = 2
    assert core.predictor_wall_history_entries(
        two_walls_two_micro
    ) == 2 * entries[2]

    no_wall_peak = core.estimated_peak_bytes(no_walls)
    two_wall_peak = core.estimated_peak_bytes(two_walls)
    assert two_wall_peak - no_wall_peak == (
        entries[2] * 16 + core.fixed_wall_cache_bytes(two_walls)
        + 16 * (4 * 2 + 3 * 2 + 5 * 2 * 2)
    )


def test_massless_alias_probe_preserves_micro_identity(monkeypatch):
    from fomhd import massless

    nc = 1
    nb = 2
    state = SimpleNamespace()
    tag = StageTag(
        step=2,
        generation=3,
        trial=4,
        spectral_context=0,
        time=0.5,
        stage="predictor_m001_r2",
        micro_step=1,
        iteration=0,
        slab=0,
    )
    rhs = SimpleNamespace(
        fluid=(
            SimpleNamespace(
                n=np.zeros(1),
                mom=np.zeros(3),
                perp=np.zeros(1),
            ),
            SimpleNamespace(
                n=np.zeros(1),
                mom=np.zeros(3),
                perp=np.zeros(1),
                parallel=np.zeros(1),
            ),
        ),
        B=np.zeros(nb),
    )
    result = SimpleNamespace(checks=[], rhs=rhs)
    alternate = SimpleNamespace(checks=[], rhs=rhs)
    context = SimpleNamespace(
        N=2,
        spectral_id=0,
        next_source=1,
        compute=SimpleNamespace(xp=np),
        p=SimpleNamespace(
            K=0,
            species=(
                SimpleNamespace(id="i"),
                SimpleNamespace(id="e"),
            )
        ),
        op=SimpleNamespace(
            Nb=nb,
            MB=np.ones(nb),
            M0=np.ones(nc),
        ),
        geometry=SimpleNamespace(
            z_faces=SimpleNamespace(keys=[0]),
            vertices=SimpleNamespace(keys=[0]),
        ),
    )
    captured = []

    def fake_evaluate(_state, alternate_tag, alternate_context, alias=False):
        captured.append((alternate_tag, alternate_context))
        return alternate

    monkeypatch.setattr(massless, "evaluate_stage", fake_evaluate)
    monkeypatch.setattr(diagnostics, "measure", lambda *args, **kwargs: None)
    monkeypatch.setattr(diagnostics, "gate", lambda *args, **kwargs: None)
    massless._alias_probe(state, tag, context, result)
    alias_tag, alias_context = captured[0]
    assert alias_context.spectral_id == 1
    assert alias_tag.micro_step == tag.micro_step
    assert alias_tag.iteration == tag.iteration
    assert alias_tag.slab == tag.slab


def test_real_massless_equilibrium_and_perturbed_interval():
    from fomhd.massless import canonicalize_state
    from fomhd.stage import initialize
    from test_massless_support import (
        context as massless_context,
        equilibrium_state,
        uniform_state,
    )

    context = massless_context(K=0, hard_diagnostics=False)
    context.p.coupling = "predictor"
    context.p.micro_steps = 2
    context.p.integrator = "rk4"
    context.p.dt_guard = "warn"
    context.p.nsteps = 1

    equilibrium_cumulative = _cumulative()
    equilibrium = initialize(
        equilibrium_state(context),
        context,
        equilibrium_cumulative,
        False,
    )
    result = time_module.advance(
        equilibrium, equilibrium_cumulative, context
    )
    assert result.accepted
    assert len(result.interval.quadrature) == 8
    assert equilibrium.state.time == core.time_at_step(context.p, 1)

    perturbed_context = massless_context(K=0, hard_diagnostics=False)
    perturbed_context.p.coupling = "predictor"
    perturbed_context.p.micro_steps = 2
    perturbed_context.p.integrator = "rk4"
    perturbed_context.p.dt_guard = "warn"
    perturbed_context.p.nsteps = 1
    cumulative = _cumulative()
    committed = initialize(
        uniform_state(perturbed_context),
        perturbed_context,
        cumulative,
        False,
    )
    initial_source = perturbed_context.next_source
    result = time_module.advance(committed, cumulative, perturbed_context)
    assert result.coupling_metadata["alias_evaluations"] == 4
    assert result.coupling_metadata["total_model_evaluations"] == 13
    assert result.coupling_metadata["complete_stage_evaluations"] == 9
    assert perturbed_context.next_source == initial_source + 13
    canonical = canonicalize_state(committed.state, perturbed_context)
    np.testing.assert_allclose(
        canonical.fluid[1].n,
        perturbed_context.p.species[0].charge
        / (-perturbed_context.p.species[1].charge)
        * canonical.fluid[0].n,
        rtol=2.0e-12,
        atol=1.0e-20,
    )
    names = {record.name for record in result.checks}
    assert "species_continuity" in names
    assert "ct_update" in names
    tags = [entry.stage for entry in result.interval.quadrature]
    assert len(tags) == 8
    assert len(set(tags)) == 8


def test_predictor_restart_identity_accepts_same_and_rejects_changed(
    tmp_path,
):
    from fomhd.__main__ import _run_command
    from test_model_routing import _context, _state, _write_case

    context = _context(inertia=0, integrator="rk4", nsteps=1, hard=True)
    state = _state(context)
    case = _write_case(
        tmp_path / "predictor",
        context.p,
        context.geometry,
        state,
        "rk4",
        1,
        coupling="predictor",
        micro_steps=2,
    )
    output = tmp_path / "predictor-run"
    assert _run_command(["run", str(case), "--output", str(output)]) == 0
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert manifest["coupling"] == "predictor"
    assert manifest["micro_steps"] == 2
    evidence = manifest["time_coupling_evidence"]
    assert evidence["micro_steps"] == 2
    assert evidence["full_rhs_stage_evaluations"] == 8
    assert evidence["endpoint_full_evaluations"] == 1
    assert evidence["complete_stage_evaluations"] == 9
    assert evidence["retained_quadrature_entries"] == 8
    assert set(evidence["endpoint_tangent_correction"]) == {
        "n_i",
        "mom_i",
        "perp_i",
    }
    reports = [
        json.loads(line)
        for line in (output / "time_estimates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert reports[-1]["coupling"] == "predictor"
    assert reports[-1]["micro_steps"] == 2
    assert "micro-RK4 only" in reports[-1]["scope"]
    latest = json.loads(
        (output / "checkpoints/latest.json").read_text(encoding="utf-8")
    )
    checkpoint = output / "checkpoints" / latest["file"]
    loaded = io.load_case(case)
    saved = io.read_restart(checkpoint, loaded)
    assert saved.state.step == 1
    assert saved.parent["checkpoint"] == str(checkpoint.absolute())

    changed = _write_case(
        tmp_path / "changed",
        context.p,
        context.geometry,
        state,
        "rk4",
        1,
        coupling="synchronous",
        micro_steps=1,
    )
    changed_case = io.load_case(changed)
    with pytest.raises(ValueError, match="coupling|micro_steps|semantic"):
        io.read_restart(checkpoint, changed_case)


def test_failure_history_merge_preserves_records_without_quadratic_scan(monkeypatch):
    from types import SimpleNamespace
    from fomhd import stage
    from fomhd.types import Failure, StageTag, check_record

    comparisons = 0

    class CountedRecord(SimpleNamespace):
        def __eq__(self, other):
            nonlocal comparisons
            comparisons += 1
            return super().__eq__(other)

    original = [
        CountedRecord(**vars(check_record(name="probe", tag=StageTag(generation=i))))
        for i in range(400)
    ]
    failure = Failure("HardGate", "retained diagnostic history")
    failure.checks = list(original)
    monkeypatch.setattr(stage._diagnostics(), "stage_requirements", lambda *_: [])
    context = SimpleNamespace()
    for record in original:
        copied = CountedRecord(**vars(record))
        result = SimpleNamespace(checks=[copied], residuals={}, cache=None)
        stage._retain_common_failure_evidence(
            failure, record.tag, context, None, result, "probe"
        )
    assert comparisons <= 2 * len(original)
    assert [id(record) for record in failure.checks] == [id(record) for record in original]

    changed = CountedRecord(**{**vars(original[0]), "raw": 2.0})
    external = check_record(name="external", tag=StageTag(generation=1000))
    failure.checks.append(external)
    result = SimpleNamespace(checks=[changed], residuals={}, cache=None)
    stage._retain_common_failure_evidence(
        failure, changed.tag, context, None, result, "changed"
    )
    assert failure.checks[-2] is external
    assert failure.checks[-1] is changed
