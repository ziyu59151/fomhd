"""Behavioral tests for the mixed coarse-ion/fine-field interval ledger."""

from __future__ import annotations

import copy
import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest

from fomhd import core, massless, massless_checks, stage
from fomhd import time as time_module
from fomhd.core import inverse
from fomhd.multirate_wr_physics import (
    MasslessWRPhysicsAdapter,
    pack_state,
    state_from_packed,
)
from fomhd.multirate_wr_checks import (
    LEDGER_FIELDS,
    BoundaryIntegral,
    LedgerIntegral,
    MixedDiagnosticWork,
    MixedInterval,
    SourceKey,
    _momentum_integrals,
    _traction_components,
    accumulate_mixed_interval,
    validate_mixed_identity,
    validate_mixed_interval,
    validate_scheduler_identity,
)
from fomhd.types import Failure, StageTag
from wr_benchmark_fixture import closed_wall_inputs, periodic_inputs


FINE_WORK_GATE_NAMES = (
    "electron_pressure_adjoint",
    "electron_magnetic_work",
    "pressure_pairing",
    "electron_electric_pairing",
    "electric_pairing",
    "electron_pressure_only_pairing",
    "electron_pressure_truncation",
    "wall_field_reaction_work_ion",
    "wall_field_reaction_work_electron",
    "wall_field_reaction_work_current",
    "wall_field_reaction_work",
)


def _full_left(context, p, z, *, stage_name):
    generation = int(context.next_generation)
    context.next_generation += 1
    tag = StageTag(
        step=0,
        generation=generation,
        trial=int(context.next_trial),
        spectral_context=context.spectral_id,
        time=float(context.p.start_time),
        stage=stage_name,
        slab=0,
    )
    state = state_from_packed(
        context,
        p,
        z,
        step=tag.step,
        generation=tag.generation,
        time=tag.time,
    )
    return state, stage.evaluate_stage(state, tag, context, False)


def _empty_cumulative():
    return SimpleNamespace(
        baseline=None,
        integrals=[
            SimpleNamespace(**dict.fromkeys(LEDGER_FIELDS, 0.0))
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


def _solve(context, state, *, H, M, s=5, initial="tangent"):
    p, z = pack_state(context, state)
    adapter = MasslessWRPhysicsAdapter(context, state)
    result = adapter.solve_window(
        p,
        z,
        H=H,
        M=M,
        s=s,
        rtol=2.0e-9,
        max_iterations=20,
        relaxation=0.7,
        initial=initial,
    )
    # The concurrently developed adapter currently evaluates the accepted
    # endpoint arrays under its accepted-left clock. Build the physically
    # clocked full endpoint required by the interval transaction.
    generation = int(context.next_generation)
    context.next_generation += 1
    endpoint_tag = StageTag(
        step=int(state.step) + 1,
        generation=generation,
        trial=result.window.coarse_samples[0].record.tag.trial,
        spectral_context=context.spectral_id,
        time=float(state.time) + float(H),
        stage="mixed_test_endpoint",
        slab=0,
    )
    endpoint_state = state_from_packed(
        context,
        result.window.P_endpoint,
        result.window.Z_endpoint,
        step=endpoint_tag.step,
        generation=endpoint_tag.generation,
        time=endpoint_tag.time,
    )
    endpoint = stage.evaluate_stage(
        endpoint_state,
        endpoint_tag,
        context,
        False,
    )
    result.endpoint.tag = endpoint_tag
    result.endpoint.full = endpoint
    result.endpoint.totals = endpoint.totals
    result.endpoint.checks = tuple(endpoint.checks)
    return p, z, result


@pytest.fixture(scope="module")
def periodic_case():
    context, state, _ = periodic_inputs(
        9,
        2.0e-4,
        0.12,
        seed_mode=1,
        sentinel=1.0e-4,
    )
    p, z, physics = _solve(
        context,
        state,
        H=float(context.p.dt),
        M=2,
        s=5,
    )
    bundle = accumulate_mixed_interval(
        physics.window,
        state,
        physics.endpoint.full,
        context,
    )
    return context, state, p, z, physics, bundle


@pytest.fixture(scope="module")
def closed_case():
    context, state, _ = closed_wall_inputs(nodes_r=4, nodes_z=4)
    p, z, physics = _solve(
        context,
        state,
        H=float(context.p.dt),
        M=1,
        s=3,
    )
    bundle = accumulate_mixed_interval(
        physics.window,
        state,
        physics.endpoint.full,
        context,
    )
    return context, state, p, z, physics, bundle


def _check_real_case(case):
    context, state, p, z, physics, bundle = case
    interval = bundle.interval
    validate_mixed_interval(interval, physics.window, state, physics.endpoint.full, context)
    validate_mixed_identity(interval)

    left_state, left = _full_left(
        context,
        p,
        z,
        stage_name="mixed_check_left",
    )
    end = physics.endpoint.full
    quadrature = [
        (view.state, view, weight)
        for view, weight in zip(
            bundle.fine_views,
            physics.window.fine_weights.reshape(-1),
        )
    ]
    cumulative = _empty_cumulative()
    checked = massless_checks.check_step(
        left_state,
        left,
        None,
        None,
        end.canonical_state,
        end,
        cumulative,
        context,
        quadrature=quadrature,
        mixed_interval=interval,
    )
    assert checked.interval.mixed_interval is interval
    assert checked.interval.mixed_attribution["density_quadrature_defect"] == (
        interval.density_work_coarse - interval.density_work_fine
    )
    assert isinstance(bundle.diagnostic_work, MixedDiagnosticWork)
    assert isinstance(checked.diagnostic_work, MixedDiagnosticWork)
    assert bundle.diagnostic_work.electron_ledger_calls == len(bundle.fine_views)
    assert (
        checked.diagnostic_work.electron_work_gate_calls
        == len(bundle.fine_views)
    )
    assert (
        checked.diagnostic_work.work_gate_records
        == len(FINE_WORK_GATE_NAMES) * len(bundle.fine_views)
    )
    assert all(
        not hasattr(view, "ledger")
        and not hasattr(view, "momentum")
        and not hasattr(view, "boundary")
        and view.rhs.fluid[0] is None
        and view.complete_stage is False
        for view in bundle.fine_views
    )
    return checked


def test_interval_has_exact_frozen_fields_and_source_weights(periodic_case):
    _, _, _, _, physics, bundle = periodic_case
    interval = bundle.interval
    assert dataclasses.is_dataclass(interval)
    assert interval.__dataclass_params__.frozen
    assert {field.name for field in dataclasses.fields(LedgerIntegral)} == set(
        LEDGER_FIELDS
    )
    assert {field.name for field in dataclasses.fields(BoundaryIntegral)} == {
        "number",
        "kinetic",
        "internal_perp",
        "internal_parallel",
        "field_energy",
        "total_energy",
    }
    assert len(interval.ledger) == 2
    assert len(interval.coarse_sources) == physics.window.s
    assert len(interval.fine_sources) == 4 * physics.window.M
    assert math.isclose(
        sum(weight for _, weight in interval.coarse_sources),
        physics.window.H,
        rel_tol=8.0e-15,
        abs_tol=8.0e-15 * physics.window.H,
    )
    assert math.isclose(
        sum(weight for _, weight in interval.fine_sources),
        physics.window.H,
        rel_tol=8.0e-15,
        abs_tol=8.0e-15 * physics.window.H,
    )
    assert all(
        isinstance(key, tuple) and len(key) == 10
        for key, _ in interval.coarse_sources + interval.fine_sources
    )
    assert interval.node_version == physics.window.final_input_version
    assert interval.trial == physics.window.coarse_samples[0].record.tag.trial


def test_scheduler_identity_uses_direct_clocks_at_nonzero_late_step(
    periodic_case,
):
    context, _, _, _, _, bundle = periodic_case
    late_context = copy.copy(context)
    late_context.p = SimpleNamespace(start_time=1.0, dt=2.0e-12)
    left_step = 23
    right_step = 24
    left_time = core.time_at_step(late_context.p, left_step)
    right_time = core.time_at_step(late_context.p, right_step)
    interval = dataclasses.replace(
        bundle.interval,
        left_step=left_step,
        right_step=right_step,
        left_time=left_time,
        right_time=right_time,
        coarse_sources=tuple(
            (key, 2.0 * weight)
            for key, weight in bundle.interval.coarse_sources
        ),
        fine_sources=tuple(
            (key, 2.0 * weight)
            for key, weight in bundle.interval.fine_sources
        ),
    )
    left = SimpleNamespace(step=left_step, time=left_time)
    right = SimpleNamespace(step=right_step, time=right_time)
    validated = validate_scheduler_identity(
        interval,
        late_context,
        left,
        right,
    )
    assert validated == (left_time, right_time)
    assert abs((right_time - left_time) - late_context.p.dt) > (
        2.0e-15 * late_context.p.dt
    )
    assert math.isclose(
        math.fsum(weight for _, weight in interval.coarse_sources),
        late_context.p.dt,
        rel_tol=8.0e-15,
        abs_tol=8.0e-15 * late_context.p.dt,
    )
    assert math.isclose(
        math.fsum(weight for _, weight in interval.fine_sources),
        late_context.p.dt,
        rel_tol=8.0e-15,
        abs_tol=8.0e-15 * late_context.p.dt,
    )
    roundoff = (
        16.0
        * np.finfo(np.float64).eps
        * max(abs(right_time), abs(late_context.p.start_time))
    )
    validate_scheduler_identity(
        interval,
        late_context,
        left,
        SimpleNamespace(step=right_step, time=right_time + roundoff),
    )
    with pytest.raises(Failure, match="clock|scheduler|identity"):
        validate_scheduler_identity(
            interval,
            late_context,
            left,
            SimpleNamespace(step=right_step, time=right_time + 1.0e-10),
        )


def test_window_h_must_match_declared_context_h(periodic_case):
    context, state, _, _, physics, bundle = periodic_case
    mismatched = copy.copy(context)
    mismatched.p = SimpleNamespace(start_time=1.0, dt=2.0e-12)
    with pytest.raises(Failure, match="declared H|window H"):
        validate_mixed_interval(
            bundle.interval,
            physics.window,
            state,
            physics.endpoint.full,
            mismatched,
        )


def test_role_specific_traction_uses_independent_nonzero_components(
    closed_case, monkeypatch
):
    context, *_ = closed_case
    xp = context.compute.xp
    n = context.N

    def constant(value):
        return xp.full((n,), value, dtype=xp.float64)

    wall = SimpleNamespace(
        component="r_face",
        face=0,
        area_per_radian=0.7,
    )
    trace = SimpleNamespace(
        wall=wall,
        direction=0,
        sigma=1.0,
        area=2.0,
        B=(constant(0.25), constant(0.4), constant(0.3)),
        density=(constant(1.0), constant(0.8)),
        velocity=(
            xp.stack(
                (
                    constant(0.2),
                    constant(-0.3),
                    constant(0.4),
                ),
                axis=1,
            ),
            xp.stack(
                (
                    constant(0.1),
                    constant(0.2),
                    constant(-0.1),
                ),
                axis=1,
            ),
        ),
        perp=(constant(2.0), constant(3.0)),
        parallel=(constant(2.0), constant(5.0)),
    )
    monkeypatch.setattr(
        massless_checks,
        "_wall_trace_data",
        lambda state, stage, active_context: [trace],
    )

    b = (0.25, 0.4, 0.3)
    b2 = sum(value * value for value in b)
    bmag = math.sqrt(b2)
    mass = context.p.species[0].mass
    transport = [
        mass * 0.2 * 0.2 + 2.0,
        mass * (-0.3) * 0.2,
        mass * 0.4 * 0.2,
    ]
    electromagnetic = []
    for component in range(3):
        unit = 1.0 if component == 0 else 0.0
        magnetic = (
            0.5 * b2 * unit - b[component] * b[0]
        ) / context.p.mu0
        electron_pressure = (
            3.0 * unit
            + (5.0 - 3.0)
            * (b[component] / bmag)
            * (b[0] / bmag)
        )
        electromagnetic.append(magnetic + electron_pressure)

    def hand_integral(components):
        outward = np.zeros(3)
        torque = 0.0
        samples = max(n, 3)
        radius = float(context.geometry.r_faces.R[wall.face])
        for q in range(samples):
            sample = 0 if n == 1 else q
            del sample
            angle = 2.0 * math.pi * q / samples
            cosine = math.cos(angle)
            sine = math.sin(angle)
            wall_weight = (
                2.0 * math.pi * wall.area_per_radian / samples
            )
            cartesian = (
                components[0] * cosine - components[1] * sine,
                components[0] * sine + components[1] * cosine,
                components[2],
            )
            for component in range(3):
                outward[component] += wall_weight * cartesian[component]
            torque += wall_weight * radius * components[1]
        return tuple(outward), torque

    expected_transport = hand_integral(transport)
    expected_electromagnetic = hand_integral(electromagnetic)
    dummy = SimpleNamespace()
    reaction = xp.zeros(
        (n, 3, context.op.Nc),
        dtype=xp.complex128,
    )
    reaction[:, 1, :] = 0.25 + 0.0j
    reaction = reaction.reshape(-1)
    actual_transport = _momentum_integrals(
        dummy,
        dummy,
        reaction,
        context,
        traction_role="transport",
    )
    actual_electromagnetic = _momentum_integrals(
        dummy,
        dummy,
        reaction,
        context,
        traction_role="electromagnetic",
    )
    np.testing.assert_allclose(
        actual_transport[0], expected_transport[0], rtol=2.0e-14, atol=1.0e-30
    )
    np.testing.assert_allclose(
        actual_electromagnetic[0],
        expected_electromagnetic[0],
        rtol=2.0e-14,
        atol=1.0e-30,
    )
    assert actual_transport[1] == pytest.approx(
        expected_transport[1], rel=2.0e-14, abs=1.0e-30
    )
    assert actual_electromagnetic[1] == pytest.approx(
        expected_electromagnetic[1], rel=2.0e-14, abs=1.0e-30
    )
    assert any(value != 0.0 for value in transport)
    assert any(value != 0.0 for value in electromagnetic)
    assert actual_transport != actual_electromagnetic
    assert actual_transport[2:] == actual_electromagnetic[2:]


def _independent_selected_traction(state, stage, context, role):
    xp = context.compute.xp
    nc = context.op.Nc
    m0 = xp.real(xp.asarray(context.op.M0)).astype(xp.float64)
    traces = massless_checks._wall_trace_data(state, stage, context)
    samples = max(context.N, 3)
    outward = np.zeros(3)
    torque = 0.0
    for q in range(samples):
        sample = 0 if context.N == 1 else q
        angle = 2.0 * math.pi * q / samples
        cosine = math.cos(angle)
        sine = math.sin(angle)
        weights = 2.0 * math.pi * m0 / samples
        del weights
        for data in traces:
            transport, electromagnetic = _traction_components(
                data, sample, context
            )
            selected = (
                transport if role == "transport" else electromagnetic
            )
            wall_weight = (
                2.0 * math.pi * data.wall.area_per_radian / samples
            )
            radius = (
                float(context.geometry.r_faces.R[data.wall.face])
                if data.wall.component == "r_face"
                else float(context.geometry.z_faces.R[data.wall.face])
            )
            cartesian = (
                selected[0] * cosine - selected[1] * sine,
                selected[0] * sine + selected[1] * cosine,
                selected[2],
            )
            for component in range(3):
                outward[component] += wall_weight * cartesian[component]
            torque += wall_weight * radius * selected[1]
    return outward, torque


def test_real_nonuniform_k1_geometry_selects_role_traction():
    from test_massless_field import _closed_nonuniform_fixture

    context, state = _closed_nonuniform_fixture()
    from test_massless_support import compatible_magnetic_field

    state.B = state.B + compatible_magnetic_field(
        context,
        seed=29,
        scale=0.1 * float(context.p.ref.magnetic),
    )
    tag = StageTag(
        step=state.step,
        generation=state.generation,
        trial=int(context.next_trial),
        spectral_context=context.spectral_id,
        time=state.time,
        stage="traction_ownership_oracle",
        slab=0,
    )
    full = stage.evaluate_stage(state, tag, context, False)
    transport, transport_torque = _independent_selected_traction(
        full.canonical_state,
        full,
        context,
        "transport",
    )
    electromagnetic, electromagnetic_torque = (
        _independent_selected_traction(
            full.canonical_state,
            full,
            context,
            "electromagnetic",
        )
    )
    expected_double = transport + electromagnetic
    actual_transport = _momentum_integrals(
        full.canonical_state,
        full,
        None,
        context,
        traction_role="transport",
    )
    actual_electromagnetic = _momentum_integrals(
        full.canonical_state,
        full,
        None,
        context,
        traction_role="electromagnetic",
    )
    np.testing.assert_allclose(
        actual_transport[0],
        transport,
        rtol=3.0e-12,
        atol=3.0e-12 * max(float(np.max(np.abs(transport))), 1.0),
    )
    np.testing.assert_allclose(
        actual_electromagnetic[0],
        electromagnetic,
        rtol=3.0e-12,
        atol=3.0e-12 * max(
            float(np.max(np.abs(electromagnetic))),
            1.0,
        ),
    )
    assert actual_transport[1] == pytest.approx(
        transport_torque,
        rel=3.0e-12,
        abs=3.0e-12 * max(abs(transport_torque), 1.0),
    )
    assert actual_electromagnetic[1] == pytest.approx(
        electromagnetic_torque,
        rel=3.0e-12,
        abs=3.0e-12 * max(abs(electromagnetic_torque), 1.0),
    )
    assert np.max(np.abs(transport)) > 0.0
    assert np.max(np.abs(electromagnetic)) > 0.0
    component_scale = max(
        float(np.max(np.abs(transport))),
        float(np.max(np.abs(electromagnetic))),
        1.0,
    )
    assert np.max(np.abs(expected_double - transport)) > (
        4096.0
        * np.finfo(np.float64).eps
        * component_scale
    )


def test_s5_density_is_integrated_by_actual_fine_continuity(periodic_case):
    context, state, p, z, physics, bundle = periodic_case
    window = physics.window
    views = bundle.fine_views
    nc = context.op.Nc
    fine_density = sum(
        -weight * np.asarray(view.frame.ion.divergence_hat)
        for view, weight in zip(views, window.fine_weights.reshape(-1))
    ).reshape(window.P.shape[1], nc)
    selected_change = np.asarray(window.P[-1, :, :nc]) - np.asarray(
        p[:, :nc]
    )
    np.testing.assert_allclose(
        fine_density,
        selected_change,
        rtol=5.0e-8,
        atol=5.0e-8 * max(float(np.max(np.abs(selected_change))), 1.0e-300),
    )
    assert np.max(np.abs(fine_density)) > 0.0
    assert abs(bundle.interval.density_work_fine) > 0.0
    assert math.isfinite(bundle.interval.kinetic_path)


def test_ledger_identity_detects_missing_or_double_counted_term(periodic_case):
    *_, bundle = periodic_case
    interval = bundle.interval
    validate_mixed_identity(interval)
    for field, value in (
        ("electric_node", 0.0),
        ("pressure_kinetic", 2.0 * interval.ledger[0].pressure_kinetic),
    ):
        damaged_ion = dataclasses.replace(interval.ledger[0], **{field: value})
        damaged = dataclasses.replace(
            interval,
            ledger=(damaged_ion, interval.ledger[1]),
        )
        with pytest.raises(Failure, match="kinetic|density|ledger"):
            validate_mixed_identity(damaged)


def test_pressure_internal_preserves_ion_and_electron_semantics(periodic_case):
    context, _, _, _, physics, bundle = periodic_case
    nc = context.op.Nc
    ion_expected = 0.0
    for sample, weight in zip(
        physics.window.coarse_samples,
        (
            physics.window.H * physics.window.Q[-1, node]
            for node in range(physics.window.s)
        ),
    ):
        ion = sample.record.ion_frame
        contribution = -massless_checks._integrate_cells(
            (
                np.asarray(ion.perp).reshape(context.N, nc)
                * np.asarray(ion.theta).reshape(context.N, nc)
            ).reshape(-1),
            1,
            context,
        )
        ion_expected += weight * float(contribution.real)

    electron_expected = 0.0
    pressure_reference = context.p.ref.pressure / context.p.ref.time
    for sample, view, weight in zip(
        physics.window.fine_samples,
        bundle.fine_views,
        physics.window.fine_weights.reshape(-1),
    ):
        perp = inverse(
            sample.record.frame.field_rates.electron_perp,
            nc,
            context,
            True,
            pressure_reference,
        ).reshape(context.N, nc)
        parallel = inverse(
            sample.record.frame.field_rates.electron_parallel,
            nc,
            context,
            True,
            pressure_reference,
        ).reshape(context.N, nc)
        electron_expected += weight * float(
            massless_checks._integrate_cells(
                (perp + 0.5 * parallel).reshape(-1),
                1,
                context,
            ).real
        )

    assert bundle.interval.ledger[0].pressure_internal == pytest.approx(
        ion_expected, rel=2.0e-12, abs=2.0e-15
    )
    assert bundle.interval.ledger[1].pressure_internal == pytest.approx(
        electron_expected, rel=2.0e-12, abs=2.0e-15
    )


@pytest.mark.parametrize("bad_weight", [float("nan"), float("inf"), -1.0e-12])
def test_accumulator_rejects_invalid_fine_weights(periodic_case, bad_weight):
    context, state, _, _, physics, _ = periodic_case
    sample = dataclasses.replace(
        physics.window.fine_samples[0],
        weight=bad_weight,
    )
    damaged = dataclasses.replace(
        physics.window,
        fine_samples=(sample,) + physics.window.fine_samples[1:],
    )
    with pytest.raises(Failure, match="weight|quadrature"):
        accumulate_mixed_interval(damaged, state, physics.endpoint.full, context)


def test_accumulator_rejects_wrong_node_version(periodic_case):
    context, state, _, _, physics, _ = periodic_case
    damaged = dataclasses.replace(
        physics.window,
        final_input_version=physics.window.final_input_version + 1,
        mapped_node_version=physics.window.mapped_node_version + 1,
    )
    with pytest.raises(Failure, match="version|source|identity"):
        accumulate_mixed_interval(damaged, state, physics.endpoint.full, context)


def test_validator_rejects_wrong_source_and_trial(periodic_case):
    context, state, _, _, physics, bundle = periodic_case
    interval = bundle.interval
    source, weight = interval.fine_sources[0]
    damaged_intervals = []
    for source_index in (0, 2, 3):
        wrong_source = list(source)
        wrong_source[source_index] += 1
        bad_sources = (
            (tuple(wrong_source), weight),
        ) + interval.fine_sources[1:]
        damaged_intervals.append(
            dataclasses.replace(interval, fine_sources=bad_sources)
        )
    damaged_intervals.append(
        dataclasses.replace(interval, trial=interval.trial + 1)
    )
    for damaged in damaged_intervals:
        with pytest.raises(Failure, match="source|trial|identity|version"):
            validate_mixed_interval(
                damaged,
                physics.window,
                state,
                physics.endpoint.full,
                context,
            )


def test_pending_mixed_consumes_exact_integrals_once(periodic_case):
    *_, bundle = periodic_case
    interval = bundle.interval
    cumulative = SimpleNamespace(
        baseline=None,
        integrals=[
            SimpleNamespace(**{name: 10.0 + component for name in LEDGER_FIELDS})
            for component in range(2)
        ],
        boundary_integrals=BoundaryIntegral(
            number=(20.0, 21.0),
            kinetic=(22.0, 23.0),
            internal_perp=(24.0, 25.0),
            internal_parallel=(26.0, 27.0),
            field_energy=28.0,
            total_energy=29.0,
        ),
        accepted_intervals=4,
        last_trial=9,
        last_step=7,
        last_time=0.5,
    )
    endpoint = SimpleNamespace(step=8, time=0.75)
    pending = time_module._pending_cumulative(
        cumulative,
        (),
        endpoint,
        trial=interval.trial,
        mixed_interval=interval,
    )
    for component in range(2):
        for name in LEDGER_FIELDS:
            assert getattr(pending.integrals[component], name) == pytest.approx(
                getattr(cumulative.integrals[component], name)
                + getattr(interval.ledger[component], name),
                rel=2.0e-15,
                abs=2.0e-15,
            )
        for name in (
            "number",
            "kinetic",
            "internal_perp",
            "internal_parallel",
        ):
            assert getattr(pending.boundary_integrals, name)[component] == (
                pytest.approx(
                    getattr(cumulative.boundary_integrals, name)[component]
                    + getattr(interval.boundary, name)[component]
                )
            )
    assert pending.boundary_integrals.field_energy == pytest.approx(
        cumulative.boundary_integrals.field_energy
        + interval.boundary.field_energy
    )
    assert pending.boundary_integrals.total_energy == pytest.approx(
        cumulative.boundary_integrals.total_energy
        + interval.boundary.total_energy
    )
    assert pending.accepted_intervals == 5
    assert (pending.last_trial, pending.last_step, pending.last_time) == (
        interval.trial,
        8,
        0.75,
    )
    assert cumulative.accepted_intervals == 4
    with pytest.raises(Failure, match="trial"):
        time_module._pending_cumulative(
            cumulative,
            (),
            endpoint,
            trial=interval.trial + 1,
            mixed_interval=interval,
        )


def test_pending_none_path_keeps_original_h_weighted_quadrature():
    ledger = SimpleNamespace(**dict.fromkeys(LEDGER_FIELDS, 0.0))
    ledger.electric_edge = 2.0
    boundary = SimpleNamespace(
        number=[3.0, 4.0],
        kinetic=[5.0, 6.0],
        internal_perp=[7.0, 8.0],
        internal_parallel=[9.0, 10.0],
        field_energy=11.0,
        total_energy=12.0,
    )
    stage_record = SimpleNamespace(ledger=[ledger, ledger], boundary=boundary)
    cumulative = SimpleNamespace(
        baseline=None,
        integrals=[
            SimpleNamespace(**dict.fromkeys(LEDGER_FIELDS, 1.0))
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
    pending = time_module._pending_cumulative(
        cumulative,
        [(None, stage_record, 0.25)],
        SimpleNamespace(step=1, time=0.2),
        trial=3,
    )
    assert pending.integrals[0].electric_edge == pytest.approx(1.5)
    assert pending.boundary_integrals.number == pytest.approx([0.75, 1.0])
    assert pending.boundary_integrals.total_energy == pytest.approx(3.0)


def test_real_periodic_mixed_check_step(periodic_case):
    _check_real_case(periodic_case)


def test_real_closed_wall_k1_mixed_check_step(closed_case):
    _check_real_case(closed_case)


def test_closed_wall_fine_work_gates_match_full_stage_without_transport(
    closed_case, monkeypatch
):
    context, accepted, p, z, physics, bundle = closed_case
    sample = physics.window.fine_samples[-1]
    view = bundle.fine_views[-1]
    generation = int(context.next_generation)
    context.next_generation += 1
    full_tag = StageTag(
        step=int(accepted.step),
        generation=generation,
        trial=bundle.interval.trial,
        spectral_context=context.spectral_id,
        time=float(accepted.time),
        stage="work_gate_full_oracle",
        micro_step=view.tag.micro_step,
        iteration=view.tag.iteration,
        slab=0,
    )
    full_state = state_from_packed(
        context,
        sample.P,
        sample.Z,
        step=full_tag.step,
        generation=full_tag.generation,
        time=full_tag.time,
    )
    full = stage.evaluate_stage(full_state, full_tag, context, False)
    full_records = {
        record.name: record
        for record in full.checks
        if record.name in FINE_WORK_GATE_NAMES
    }
    assert set(full_records) == set(FINE_WORK_GATE_NAMES)

    left_state, left = _full_left(
        context,
        p,
        z,
        stage_name="work_gate_left_oracle",
    )
    quadrature = [
        (stage_view.state, stage_view, weight)
        for stage_view, weight in zip(
            bundle.fine_views,
            physics.window.fine_weights.reshape(-1),
        )
    ]

    def forbidden(*args, **kwargs):
        raise AssertionError("fine work gates evaluated ion transport")

    monkeypatch.setattr(massless, "build_non_electric", forbidden)
    checked = massless_checks.check_step(
        left_state,
        left,
        None,
        None,
        physics.endpoint.full.canonical_state,
        physics.endpoint.full,
        _empty_cumulative(),
        context,
        quadrature=quadrature,
        mixed_interval=bundle.interval,
    )
    fine_records = {
        record.name: record
        for record in checked.checks
        if record.name in FINE_WORK_GATE_NAMES and record.tag == view.tag
    }
    assert set(fine_records) == set(FINE_WORK_GATE_NAMES)
    for name in FINE_WORK_GATE_NAMES:
        full_record = full_records[name]
        fine_record = fine_records[name]
        for attribute in (
            "raw",
            "reference",
            "threshold",
            "hard",
            "available",
            "passed",
        ):
            assert getattr(fine_record, attribute) == getattr(
                full_record, attribute
            ), f"{name}.{attribute}"
        for attribute in (
            "absolute_final_terms",
            "configured_threshold",
            "modal_work",
            "sample_work",
            "transform_difference",
            "upstream_bound",
            "compatibility_bound",
            "arithmetic_bound",
        ):
            if hasattr(full_record, attribute):
                assert getattr(fine_record, attribute) == getattr(
                    full_record, attribute
                ), f"{name}.{attribute}"
    for name in (
        "wall_field_reaction_work_ion",
        "wall_field_reaction_work_electron",
        "wall_field_reaction_work_current",
        "wall_field_reaction_work",
    ):
        assert fine_records[name].arithmetic_bound >= 0.0

    modes = 2 * context.p.K + 1
    cases = len(bundle.fine_views)
    assert checked.diagnostic_work.electron_ledger_calls == cases
    assert (
        checked.diagnostic_work.electron_pressure_only_projections == cases
    )
    assert checked.diagnostic_work.electron_ledger_inverse_transforms == 4 * cases
    assert checked.diagnostic_work.wall_work_metric_calls == cases
    assert checked.diagnostic_work.wall_work_inverse_transforms == 2 * cases
    assert checked.diagnostic_work.wall_work_mode_envelope_sets == modes * cases
    assert checked.diagnostic_work.electron_work_gate_calls == cases
    assert checked.diagnostic_work.electron_gate_transforms == cases
    assert (
        checked.diagnostic_work.work_gate_records
        == len(FINE_WORK_GATE_NAMES) * cases
    )


def test_rejected_mixed_interval_retains_work_gate_counts(
    closed_case, monkeypatch
):
    context, accepted, p, z, physics, bundle = closed_case
    left_state, left = _full_left(
        context,
        p,
        z,
        stage_name="rejected_work_gate_left",
    )
    quadrature = [
        (view.state, view, weight)
        for view, weight in zip(
            bundle.fine_views,
            physics.window.fine_weights.reshape(-1),
        )
    ]
    original_bound = massless_checks._bound_real

    def reject_after_partial_work(value, name):
        if name == "ion absolute scale":
            raise Failure("InjectedWorkGate", "reject completed work gate")
        return original_bound(value, name)

    def forbidden(*args, **kwargs):
        raise AssertionError("rejected fine work path evaluated ion transport")

    monkeypatch.setattr(massless, "build_non_electric", forbidden)
    monkeypatch.setattr(massless_checks, "_bound_real", reject_after_partial_work)
    with pytest.raises(Failure, match="reject completed work gate") as caught:
        massless_checks.check_step(
            left_state,
            left,
            None,
            None,
            physics.endpoint.full.canonical_state,
            physics.endpoint.full,
            _empty_cumulative(),
            context,
            quadrature=quadrature,
            mixed_interval=bundle.interval,
        )
    work = caught.value.mixed_diagnostic_work
    assert work.electron_ledger_calls == 1
    assert work.electron_pressure_only_projections == 1
    assert work.wall_work_metric_calls == 1
    assert work.electron_work_gate_calls == 1
    assert work.work_gate_records == 7


def test_accumulation_and_check_do_not_run_fine_ion_transport(
    periodic_case, monkeypatch
):
    context, state, _, _, physics, _ = periodic_case

    def forbidden(*args, **kwargs):
        raise AssertionError("fine mixed path evaluated ion transport")

    monkeypatch.setattr(massless, "build_non_electric", forbidden)
    bundle = accumulate_mixed_interval(
        physics.window,
        state,
        physics.endpoint.full,
        context,
    )
    assert all(
        view.frame.kind == "massless_field_only"
        for view in bundle.fine_views
    )
