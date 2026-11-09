"""Integration tests for the complete massless A1-WR macro transaction."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fomhd import core, io, massless_checks, massless_field, stage
from fomhd import time as time_module
from fomhd.__main__ import _context as command_context
from fomhd.multirate_wr_checks import LEDGER_FIELDS
from fomhd.types import Failure, check_record
from wr_benchmark_fixture import closed_wall_inputs, periodic_inputs


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "fomhd"
    / "time_multirate.py"
)


@pytest.fixture(scope="module")
def multirate_module():
    if not MODULE_PATH.is_file():
        return None
    importlib.invalidate_caches()
    package = importlib.import_module("fomhd")
    name = "fomhd.time_multirate"
    missing = object()
    previous_module = sys.modules.get(name, missing)
    previous_attribute = getattr(package, "time_multirate", missing)
    module = importlib.import_module("fomhd.time_multirate")
    assert Path(module.__file__).resolve() == MODULE_PATH.resolve()
    sys.modules[name] = module
    package.time_multirate = module
    assert sys.modules[name] is module
    assert package.time_multirate is module
    try:
        yield module
    finally:
        restore = (
            previous_module
            if previous_module is not missing
            else previous_attribute
        )
        if restore is missing:
            sys.modules.pop(name, None)
            if hasattr(package, "time_multirate"):
                del package.time_multirate
        else:
            sys.modules[name] = restore
            package.time_multirate = restore


def _empty_cumulative(start_time=0.0):
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
        last_time=start_time,
    )


def _configure(parameters, *, nsteps=1, micro_steps=2, ion_nodes=3):
    parameters.coupling = "multirate_wr"
    parameters.integrator = "rk4"
    parameters.micro_steps = micro_steps
    parameters.ion_nodes = ion_nodes
    parameters.waveform_rtol = 2.0e-8
    parameters.waveform_max_iterations = 20
    parameters.wr_initial = "tangent"
    parameters.wr_relaxation = 0.8
    parameters.dt = 2.0e-12
    parameters.nsteps = nsteps
    parameters.dt_guard = "warn"
    parameters.minimum_samples_per_cycle = 1.0
    parameters.memory_limit = 1 << 30
    parameters.device_memory_limit = 1 << 30


def _periodic_runtime(*, nsteps=1, micro_steps=2, ion_nodes=3):
    context, state, _ = closed_wall_inputs(nodes_r=4, nodes_z=4)
    _configure(
        context.p,
        nsteps=nsteps,
        micro_steps=micro_steps,
        ion_nodes=ion_nodes,
    )
    cumulative = _empty_cumulative(context.p.start_time)
    committed = stage.initialize(
        state,
        context,
        cumulative,
        False,
    )
    return context, committed, cumulative


def _closed_runtime(*, nsteps=1):
    context, state, _ = closed_wall_inputs(nodes_r=4, nodes_z=4)
    _configure(context.p, nsteps=nsteps, micro_steps=1, ion_nodes=3)
    cumulative = _empty_cumulative(context.p.start_time)
    committed = stage.initialize(
        state,
        context,
        cumulative,
        False,
    )
    return context, committed, cumulative


def _write_multirate_case(
    root,
    context,
    state,
    *,
    nsteps,
    ion_nodes=3,
    relaxation=0.8,
):
    from test_050_transactions import _reset_case_hash
    from test_model_routing import _write_case

    case = _write_case(
        root,
        context.p,
        context.geometry,
        state,
        "rk4",
        nsteps,
        coupling="multirate_wr",
        micro_steps=context.p.micro_steps,
        dt_guard="warn",
        minimum_samples_per_cycle=1.0,
    )
    text = case.read_text(encoding="utf-8")
    lines = text.splitlines()
    position = lines.index("[boundary]")
    lines[position:position] = [
        f"waveform_rtol={context.p.waveform_rtol!r}",
        f"waveform_max_iterations={context.p.waveform_max_iterations}",
        f"ion_nodes={ion_nodes}",
        f'wr_initial="{context.p.wr_initial}"',
        f"wr_relaxation={relaxation!r}",
    ]
    case.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _reset_case_hash(case)
    return case


def _host_state(context, state):
    xp = context.compute.xp
    return {
        "B": np.asarray(xp.asnumpy(state.B) if xp is not np else state.B).copy(),
        **{
            f"{species}_{name}": np.asarray(
                xp.asnumpy(getattr(state.fluid[species], name))
                if xp is not np
                else getattr(state.fluid[species], name)
            ).copy()
            for species in range(2)
            for name in ("n", "mom", "perp", "parallel")
        },
    }


def _assert_rollback(context, committed, cumulative, state, cumulative_json):
    assert committed.state is state
    assert committed.accepted is True
    io._check_accepted(committed)
    assert io._cumulative_json(cumulative) == cumulative_json


def test_multirate_controls_validate_and_legacy_rejections_are_explicit(
    multirate_module,
):
    context, committed, cumulative = _periodic_runtime()
    del multirate_module, committed, cumulative
    assert time_module.validate_time_controls(context.p) == (
        "multirate_wr",
        2,
    )
    changes = (
        ("electron_inertia", 1),
        ("integrator", "midpoint"),
        ("ion_nodes", 4),
        ("wr_initial", "linear"),
        ("wr_relaxation", 0.0),
        ("wr_relaxation", 1.01),
        ("waveform_rtol", 0.0),
        ("waveform_max_iterations", 0),
    )
    for name, value in changes:
        damaged = copy.copy(context.p)
        setattr(damaged, name, value)
        with pytest.raises(Failure, match="requires|must|coupling"):
            time_module.validate_time_controls(damaged)
    legacy = copy.copy(context.p)
    legacy.coupling = "synchronous"
    with pytest.raises(Failure, match="require|must|coupling"):
        time_module.validate_time_controls(legacy)


def test_manufactured_source_is_rejected_before_counter_mutation(
    multirate_module,
):
    context, committed, cumulative = _periodic_runtime()
    state = committed.state
    ledger = copy.deepcopy(io._cumulative_json(cumulative))
    counters = (
        context.next_trial,
        context.next_generation,
        context.next_source,
    )
    with pytest.raises(Failure, match="source|Manufactured"):
        time_module.advance(
            committed,
            cumulative,
            context,
            source=lambda state, tag, context: None,
        )
    assert (
        context.next_trial,
        context.next_generation,
        context.next_source,
    ) == counters
    _assert_rollback(
        context,
        committed,
        cumulative,
        state,
        ledger,
    )


def test_first_base_failure_is_preserved_without_metadata_masking(
    multirate_module, monkeypatch
):
    context, committed, cumulative = _periodic_runtime()
    old_state = committed.state
    old_cache = committed.cache
    old_ledger = copy.deepcopy(io._cumulative_json(cumulative))
    original = multirate_module.evaluate_stage

    def fail_base(value, tag, owner, alias=False):
        if tag.stage == "multirate_base":
            failure = Failure(
                "OriginalBaseFailure",
                "accepted-left base intentionally failed",
            )
            failure.stage = tag
            failure.checks.append(
                check_record(
                    name="injected_base_guard",
                    tag=tag,
                    raw=2.0,
                    reference=1.0,
                    threshold=1.0,
                    passed=False,
                )
            )
            failure.partial_arrays["injected_base_input"] = value.fluid[0].n
            raise failure
        return original(value, tag, owner, alias)

    monkeypatch.setattr(
        multirate_module,
        "evaluate_stage",
        fail_base,
    )
    with pytest.raises(Failure) as caught:
        time_module.advance(committed, cumulative, context)

    failure = caught.value
    assert failure.code == "OriginalBaseFailure"
    assert str(failure) == "accepted-left base intentionally failed"
    assert failure.stage.stage == "multirate_base"
    assert any(
        check.name == "injected_base_guard"
        for check in failure.checks
    )
    assert "injected_base_input" in failure.partial_arrays
    assert failure.coupling_metadata["rollback"] is True
    assert failure.coupling_metadata["base_source_id"] is None
    assert failure.coupling_metadata["residual_history"] is None
    assert failure.coupling_metadata["final_normalized_residual_max"] is None
    attempts = failure.coupling_metadata["work"]
    assert attempts["base_full_stage_attempts"] == 1
    assert attempts["base_full_stage_calls"] == 0
    assert attempts["base_full_stage_failures"] == 1
    assert attempts["total_full_stage_attempts"] == 1
    assert attempts["total_full_stage_calls"] == 0
    assert attempts["total_full_stage_failures"] == 1
    assert attempts["full_ion_power_ledger_values"] == 0
    _assert_rollback(
        context,
        committed,
        cumulative,
        old_state,
        old_ledger,
    )
    assert committed.cache is old_cache


def test_one_macro_commits_once_with_mixed_interval_and_matching_endpoint(
    multirate_module,
):
    context, committed, cumulative = _periodic_runtime()
    left = committed.state
    result = time_module.advance(committed, cumulative, context)

    assert result.accepted is True
    assert result.interval.accepted is True
    assert committed.state.step == 1
    assert committed.state.time == context.p.dt
    assert committed.state.generation == result.frame.tag.generation
    macro_generations = list(
        result.coupling_metadata["coarse_generations"]
        + result.coupling_metadata["fine_generations"]
    )
    assert result.frame.tag.generation == max(macro_generations + [
        result.frame.tag.generation
    ])
    assert context.next_generation == result.frame.tag.generation + 1
    assert committed.state.step == result.frame.tag.step
    assert committed.state.time == result.frame.tag.time
    assert committed.cache is result.frame.cache
    assert committed.cache.tag == result.frame.tag
    assert cumulative.accepted_intervals == 1
    assert cumulative.last_step == 1
    assert cumulative.last_time == context.p.dt
    assert cumulative.last_trial == result.interval.trial
    assert result.interval.mixed_interval is result.coupling_metadata["interval"]
    assert result.coupling_metadata["coupling"] == "multirate_wr"
    assert result.coupling_metadata["micro_steps"] == 2
    assert result.coupling_metadata["ion_nodes"] == 3
    assert result.coupling_metadata["wr_initial"] == "tangent"
    assert result.coupling_metadata["wr_relaxation"] == 0.8
    assert result.coupling_metadata["work"]["fine_full_stage_calls"] == 0
    assert (
        result.coupling_metadata["waveform_residual_scaled"]
        <= 1.0
    )
    base_kinetic = result.coupling_metadata["base_ion_kinetic_J"]
    endpoint_kinetic = result.coupling_metadata["endpoint_ion_kinetic_J"]
    kinetic_path = result.coupling_metadata["interval"].kinetic_path
    assert result.coupling_metadata["epsilon_K_J"] == pytest.approx(
        endpoint_kinetic - base_kinetic - kinetic_path,
        rel=2.0e-15,
        abs=2.0e-300,
    )
    sweeps = result.coupling_metadata["sweeps"]
    fine_samples = 4 * result.coupling_metadata["micro_steps"]
    bundle_work = result.coupling_metadata["bundle_diagnostic_work"]
    result_work = result.coupling_metadata["result_diagnostic_work"]
    assert bundle_work["electron_ledger_calls"] == fine_samples
    assert bundle_work["electron_pressure_only_projections"] == fine_samples
    assert bundle_work["electron_ledger_inverse_transforms"] == 4 * fine_samples
    assert result_work["electron_ledger_calls"] == fine_samples
    assert result_work["wall_work_metric_calls"] == fine_samples
    assert result_work["electron_work_gate_calls"] == fine_samples
    work = result.coupling_metadata["work"]
    assert work["fine_ion_power_ledger_calls"] == 0
    assert work["total_full_stage_attempts"] == (
        work["adapter_full_stage_attempts"]
        + work["base_full_stage_attempts"]
    )
    assert work["total_full_stage_calls"] == (
        work["adapter_full_stage_calls"]
        + work["base_full_stage_calls"]
    )
    assert work["total_full_stage_attempts"] == work["total_full_stage_calls"]
    assert work["full_ion_power_ledger_values"] == (
        work["total_full_stage_calls"]
    )
    assert work["full_electron_power_ledger_values"] == (
        work["total_full_stage_calls"]
    )
    assert sweeps >= 1
    residual_history = result.coupling_metadata["residual_history"]
    assert isinstance(residual_history, list)
    assert len(residual_history) == sweeps
    assert all(isinstance(record, dict) for record in residual_history)
    assert residual_history[-1]["normalized_max"] == pytest.approx(
        result.coupling_metadata["final_normalized_residual_max"]
    )
    assert residual_history[-1]["raw_linf"] == pytest.approx(
        result.coupling_metadata["final_raw_norm"]
    )
    endpoint_summary = result.coupling_metadata[
        "endpoint_residual_summary"
    ]
    assert endpoint_summary["endpoint_raw_linf"] == pytest.approx(
        residual_history[-1]["endpoint_raw_linf"]
    )
    assert endpoint_summary["endpoint_normalized_linf"] == pytest.approx(
        residual_history[-1]["endpoint_normalized_linf"]
    )
    assert result.coupling_metadata["endpoint"]["production_accepted"] is True
    assert result.coupling_metadata["endpoint"]["trial_endpoint"] is False
    assert (
        result.coupling_metadata["endpoint"]["interval_transaction_required"]
        is False
    )
    mixed = result.interval.mixed_interval
    for species in range(2):
        for name in LEDGER_FIELDS:
            assert getattr(cumulative.integrals[species], name) == (
                pytest.approx(getattr(mixed.ledger[species], name), rel=2e-15)
            )
    boundary = cumulative.boundary_integrals
    for name in (
        "number",
        "kinetic",
        "internal_perp",
        "internal_parallel",
    ):
        assert getattr(boundary, name) == pytest.approx(
            getattr(mixed.boundary, name)
        )
    assert boundary.field_energy == pytest.approx(mixed.boundary.field_energy)
    assert boundary.total_energy == pytest.approx(mixed.boundary.total_energy)
    assert left.step == 0


def test_two_consecutive_macros_preserve_endpoint_clocks_and_ledger(
    multirate_module,
):
    context, committed, cumulative = _periodic_runtime(nsteps=2)
    first = time_module.advance(committed, cumulative, context)
    first_trial = first.interval.trial
    second = time_module.advance(committed, cumulative, context)

    assert first.accepted and second.accepted
    assert first_trial != second.interval.trial
    assert committed.state.step == 2
    assert committed.state.time == 2.0 * context.p.dt
    assert cumulative.accepted_intervals == 2
    assert cumulative.last_trial == second.interval.trial
    assert (
        second.coupling_metadata["left_step"],
        second.coupling_metadata["right_step"],
    ) == (1, 2)


def test_nonzero_start_32_macros_use_declared_h_without_clock_cancellation(
    multirate_module,
):
    context, state, _ = periodic_inputs(
        9,
        0.0,
        0.0,
        seed_mode=1,
        sentinel=0.0,
    )
    _configure(context.p, nsteps=32, micro_steps=1, ion_nodes=3)
    context.p.start_time = 1.0
    context.p.dt = 2.0e-12
    state.time = context.p.start_time
    cumulative = _empty_cumulative(context.p.start_time)
    committed = stage.initialize(
        state,
        context,
        cumulative,
        False,
    )
    for step in range(32):
        result = time_module.advance(
            committed, cumulative, context
        )
        assert result.accepted
        assert result.coupling_metadata["interval"].right_time == (
            core.time_at_step(context.p, step + 1)
        )
        assert result.coupling_metadata["interval"].left_time == (
            core.time_at_step(context.p, step)
        )
        assert committed.state.step == step + 1
        assert committed.state.time == core.time_at_step(
            context.p, step + 1
        )
    assert cumulative.accepted_intervals == 32
    assert cumulative.last_step == 32
    assert cumulative.last_time == core.time_at_step(context.p, 32)


def test_full_ion_call_count_is_independent_of_microsteps(multirate_module):
    observations = []
    for micro_steps in (1, 2):
        context, committed, cumulative = _periodic_runtime(
            nsteps=1,
            micro_steps=micro_steps,
            ion_nodes=3,
        )
        result = time_module.advance(committed, cumulative, context)
        work = result.coupling_metadata["work"]
        sweeps = result.coupling_metadata["sweeps"]
        assert work["full_stage_calls"] == 2 + 3 * sweeps
        assert work["coarse_transport_calls"] == 1 + 3 * sweeps
        assert work["endpoint_full_stage_calls"] == 1
        assert work["field_callback_calls"] == 1 + 4 * micro_steps * sweeps
        observations.append(
            {
                "adapter_full": work["full_stage_calls"],
                "base_full": work["base_full_stage_calls"],
                "coarse": work["coarse_transport_calls"],
                "endpoint": work["endpoint_full_stage_calls"],
                "field_callbacks": work["field_callback_calls"],
                "sweeps": sweeps,
            }
        )
    assert all(row["base_full"] == 1 for row in observations)
    assert all(row["endpoint"] == 1 for row in observations)
    assert all(
        row["adapter_full"] == 2 + 3 * row["sweeps"]
        for row in observations
    )


def test_full_power_ledger_spy_matches_charged_metadata(
    multirate_module, monkeypatch
):
    context, committed, cumulative = _periodic_runtime(
        nsteps=1,
        micro_steps=1,
        ion_nodes=3,
    )
    calls = {"count": 0}
    original = massless_checks.power_ledger

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(massless_checks, "power_ledger", counted)
    result = time_module.advance(committed, cumulative, context)
    work = result.coupling_metadata["work"]
    assert calls["count"] == work["total_full_stage_calls"]
    assert calls["count"] == work["full_ion_power_ledger_values"]
    assert work["fine_ion_power_ledger_calls"] == 0


def test_real_closed_wall_k1_macro_commits(multirate_module):
    context, committed, cumulative = _closed_runtime()
    result = time_module.advance(committed, cumulative, context)
    assert result.accepted is True
    assert committed.state.step == 1
    assert committed.cache.tag == result.frame.tag
    assert result.interval.mixed_interval is not None
    assert cumulative.accepted_intervals == 1
    assert result.coupling_metadata["work"]["fine_full_stage_calls"] == 0


@pytest.mark.parametrize("failure_point", ["fine", "coarse", "endpoint"])
def test_injected_transaction_failure_rolls_back_every_physical_commit(
    multirate_module, monkeypatch, failure_point
):
    context, committed, cumulative = _periodic_runtime()
    old_state = committed.state
    old_cache = committed.cache
    old_ledger = copy.deepcopy(io._cumulative_json(cumulative))
    old_host = _host_state(context, old_state)
    original = stage.evaluate_stage
    calls = {"count": 0}

    def injected(value, tag, owner, alias=False):
        calls["count"] += 1
        if (
            failure_point == "fine"
            and "field_rk4" in tag.stage
            and calls["count"] >= 2
        ) or (
            failure_point == "coarse"
            and tag.stage == "wr_transport_coarse"
        ) or (
            failure_point == "endpoint"
            and tag.stage == "endpoint"
        ):
            failure = Failure(
                "InjectedTransaction",
                f"injected {failure_point} transaction failure",
            )
            failure.stage = tag
            raise failure
        return original(value, tag, owner, alias)

    if failure_point == "fine":
        original_field = massless_field.evaluate

        def injected_field(value, tag, owner, alias=False):
            calls["count"] += 1
            if "field_rk4_r1" in tag.stage and calls["count"] >= 2:
                failure = Failure(
                    "InjectedTransaction",
                    "injected fine transaction failure",
                )
                failure.stage = tag
                raise failure
            return original_field(value, tag, owner, alias)

        monkeypatch.setattr(
            massless_field, "evaluate", injected_field
        )
    else:
        monkeypatch.setattr(stage, "evaluate_stage", injected)

    with pytest.raises(Failure, match="injected") as caught:
        time_module.advance(committed, cumulative, context)

    _assert_rollback(
        context,
        committed,
        cumulative,
        old_state,
        old_ledger,
    )
    assert committed.cache is old_cache
    current_host = _host_state(context, committed.state)
    for name, value in old_host.items():
        np.testing.assert_array_equal(current_host[name], value)
    failure = caught.value
    assert failure.coupling_metadata["rollback"] is True
    assert failure.coupling_metadata["work"]["base_full_stage_calls"] == 1
    assert failure.multirate_evidence


def test_nonconvergence_rolls_back_without_acceptance(multirate_module):
    context, committed, cumulative = _periodic_runtime()
    context.p.waveform_rtol = 1.0e-30
    context.p.waveform_max_iterations = 1
    old_state = committed.state
    old_ledger = copy.deepcopy(io._cumulative_json(cumulative))
    with pytest.raises(Failure, match="converg|Multirate") as caught:
        time_module.advance(committed, cumulative, context)
    _assert_rollback(
        context,
        committed,
        cumulative,
        old_state,
        old_ledger,
    )
    metadata = caught.value.coupling_metadata
    assert isinstance(metadata["residual_history"], list)
    assert len(metadata["residual_history"]) == 1
    assert all(
        isinstance(record, dict) for record in metadata["residual_history"]
    )
    assert metadata["final_normalized_residual_max"] > 1.0
    assert metadata["endpoint_residual_summary"] is not None


def test_duplicate_trial_fails_without_second_physical_commit(
    multirate_module,
):
    context, committed, cumulative = _periodic_runtime(nsteps=2)
    accepted = time_module.advance(committed, cumulative, context)
    accepted_state = committed.state
    accepted_ledger = copy.deepcopy(io._cumulative_json(cumulative))
    cumulative.last_trial = int(context.next_trial)
    duplicate_ledger = copy.deepcopy(io._cumulative_json(cumulative))
    with pytest.raises(Failure, match="accumulated") as caught:
        time_module.advance(committed, cumulative, context)
    assert caught.value.code == "DuplicateInterval"
    _assert_rollback(
        context,
        committed,
        cumulative,
        accepted_state,
        duplicate_ledger,
    )
    assert accepted.interval.accepted is True


def test_endpoint_cache_generation_mismatch_rolls_back(
    multirate_module, monkeypatch
):
    context, committed, cumulative = _periodic_runtime()
    state = committed.state
    ledger = copy.deepcopy(io._cumulative_json(cumulative))
    original = multirate_module.MasslessWRPhysicsAdapter.solve_window

    def corrupt_endpoint(adapter, *args, **kwargs):
        result = original(adapter, *args, **kwargs)
        result.endpoint.full.canonical_state.generation += 1
        return result

    monkeypatch.setattr(
        multirate_module.MasslessWRPhysicsAdapter,
        "solve_window",
        corrupt_endpoint,
    )
    with pytest.raises(Failure, match="endpoint|generation|cache"):
        time_module.advance(committed, cumulative, context)
    _assert_rollback(
        context,
        committed,
        cumulative,
        state,
        ledger,
    )


def test_metadata_failure_before_commit_does_not_commit_physical_state(
    multirate_module, monkeypatch
):
    context, committed, cumulative = _periodic_runtime()
    state = committed.state
    cache = committed.cache
    ledger = copy.deepcopy(io._cumulative_json(cumulative))
    original = multirate_module._metadata

    def fail_success_metadata(*args, **kwargs):
        if kwargs["rollback"] is False:
            failure = Failure(
                "InjectedMetadata",
                "intentional accepted-metadata failure",
            )
            failure.stage = kwargs["left"]
            raise failure
        return original(*args, **kwargs)

    monkeypatch.setattr(
        multirate_module,
        "_metadata",
        fail_success_metadata,
    )
    with pytest.raises(Failure) as caught:
        time_module.advance(committed, cumulative, context)
    assert caught.value.code == "InjectedMetadata"
    assert caught.value.coupling_metadata["rollback"] is True
    residual_history = caught.value.coupling_metadata["residual_history"]
    assert isinstance(residual_history, list)
    assert residual_history
    assert all(isinstance(record, dict) for record in residual_history)
    _assert_rollback(
        context,
        committed,
        cumulative,
        state,
        ledger,
    )
    assert committed.cache is cache
    assert committed.state.step == 0
    assert cumulative.accepted_intervals == 0


def test_multirate_config_semantics_include_exact_scheme_identity(tmp_path):
    from test_model_routing import _context, _state

    context = _context(inertia=0, nsteps=1, hard=True)
    state = _state(context)
    _configure(context.p, nsteps=1, micro_steps=2, ion_nodes=3)
    one = _write_multirate_case(
        tmp_path / "one", context, state, nsteps=1
    )
    two = _write_multirate_case(
        tmp_path / "two", context, state, nsteps=2
    )
    changed = _write_multirate_case(
        tmp_path / "changed",
        context,
        state,
        nsteps=1,
        ion_nodes=5,
        relaxation=0.7,
    )
    first = io.load_case(one)
    second = io.load_case(two)
    other = io.load_case(changed)
    assert first.parameters.coupling == "multirate_wr"
    assert first.parameters.micro_steps == 2
    assert first.parameters.ion_nodes == 3
    assert first.parameters.wr_initial == "tangent"
    assert first.parameters.wr_relaxation == 0.8
    assert first.semantic_hash == second.semantic_hash
    assert first.semantic_hash != other.semantic_hash
    assert other.parameters.ion_nodes == 5
    assert other.parameters.wr_relaxation == 0.7


def test_multirate_config_rejects_missing_or_legacy_only_controls(tmp_path):
    from test_050_transactions import _reset_case_hash
    from test_model_routing import _context, _state, _write_case

    context = _context(inertia=0, nsteps=1, hard=True)
    state = _state(context)
    _configure(context.p, nsteps=1, micro_steps=2, ion_nodes=3)
    missing = _write_case(
        tmp_path / "missing",
        context.p,
        context.geometry,
        state,
        "rk4",
        1,
        coupling="multirate_wr",
        micro_steps=2,
    )
    with pytest.raises(ValueError, match="ion_nodes|control|missing"):
        io.load_case(missing)
    legacy = _write_case(
        tmp_path / "legacy",
        context.p,
        context.geometry,
        state,
        "rk4",
        1,
        coupling="synchronous",
    )
    text = legacy.read_text(encoding="utf-8")
    text = text.replace(
        'coupling="synchronous"',
        'coupling="synchronous"\nion_nodes=3\n'
        'wr_initial="tangent"\nwr_relaxation=1.0',
    )
    legacy.write_text(text, encoding="utf-8")
    _reset_case_hash(legacy)
    with pytest.raises(ValueError, match="multirate_wr|legacy|control"):
        io.load_case(legacy)


def test_checkpoint_restart_matches_uninterrupted_state_and_full_ledger(
    multirate_module, tmp_path
):
    from test_model_routing import _context, _state

    context = _context(inertia=0, nsteps=1, hard=True)
    state = _state(context)
    _configure(context.p, nsteps=2, micro_steps=2, ion_nodes=3)
    context.p.start_time = 1.0
    state.time = context.p.start_time
    first_case = _write_multirate_case(
        tmp_path / "case1", context, state, nsteps=1
    )
    second_case = _write_multirate_case(
        tmp_path / "case2", context, state, nsteps=2
    )
    first_loaded = io.load_case(first_case)
    second_loaded = io.load_case(second_case)
    assert first_loaded.semantic_hash == second_loaded.semantic_hash

    first_context = command_context(first_loaded)
    first_ledger = _empty_cumulative(first_context.p.start_time)
    first_committed = stage.initialize(
        io.load_initial(first_loaded),
        first_context,
        first_ledger,
        False,
    )
    time_module.advance(first_committed, first_ledger, first_context)
    run = io.Run(first_loaded, require_project_snapshot=False)
    try:
        run.checkpoint(first_committed, first_ledger, first_context)
    finally:
        run.close()
    latest = json.loads(
        (run.directory / "checkpoints/latest.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoint = run.directory / "checkpoints" / latest["file"]
    saved = io.read_restart(checkpoint, second_loaded)
    assert saved.state.step == 1
    assert saved.cumulative.accepted_intervals == 1

    resumed_context = command_context(second_loaded)
    resumed_context.next_trial = saved.trial + 1
    resumed_context.next_source = saved.source + 1
    validation = _empty_cumulative(second_loaded.parameters.start_time)
    resumed_committed = stage.initialize(
        saved.state,
        resumed_context,
        validation,
        False,
    )
    resumed_ledger = saved.cumulative
    time_module.advance(
        resumed_committed, resumed_ledger, resumed_context
    )

    direct_context = command_context(second_loaded)
    direct_ledger = _empty_cumulative(direct_context.p.start_time)
    direct_committed = stage.initialize(
        io.load_initial(second_loaded),
        direct_context,
        direct_ledger,
        False,
    )
    time_module.advance(direct_committed, direct_ledger, direct_context)
    time_module.advance(direct_committed, direct_ledger, direct_context)

    resumed_host = _host_state(resumed_context, resumed_committed.state)
    direct_host = _host_state(direct_context, direct_committed.state)
    for name in direct_host:
        if direct_host[name].size == 0:
            assert resumed_host[name].size == 0
            continue
        np.testing.assert_allclose(
            resumed_host[name],
            direct_host[name],
            rtol=2.0e-12,
            atol=2.0e-12 * max(float(np.max(np.abs(direct_host[name]))), 1.0e-300),
        )
    assert resumed_ledger.accepted_intervals == 2
    assert direct_ledger.accepted_intervals == 2
    for species in range(2):
        for name in LEDGER_FIELDS:
            assert getattr(
                resumed_ledger.integrals[species], name
            ) == pytest.approx(
                getattr(direct_ledger.integrals[species], name),
                rel=3.0e-12,
                abs=3.0e-12,
            )
    for name in (
        "number",
        "kinetic",
        "internal_perp",
        "internal_parallel",
    ):
        assert getattr(
            resumed_ledger.boundary_integrals, name
        ) == pytest.approx(
            getattr(direct_ledger.boundary_integrals, name),
            rel=3.0e-12,
            abs=3.0e-12,
        )
    assert (
        resumed_ledger.boundary_integrals.field_energy
        == pytest.approx(
            direct_ledger.boundary_integrals.field_energy,
            rel=3.0e-12,
            abs=3.0e-12,
        )
    )
    assert (
        resumed_ledger.boundary_integrals.total_energy
        == pytest.approx(
            direct_ledger.boundary_integrals.total_energy,
            rel=3.0e-12,
            abs=3.0e-12,
        )
    )

    mismatched = _write_multirate_case(
        tmp_path / "case-mismatch",
        context,
        state,
        nsteps=1,
        ion_nodes=5,
    )
    with pytest.raises(ValueError, match="semantic|control|ion_nodes"):
        io.read_restart(checkpoint, io.load_case(mismatched))


def test_cli_run_and_restart_route_through_multirate_transaction(
    multirate_module, tmp_path
):
    from fomhd.__main__ import _run_command
    from test_model_routing import _context, _state

    context = _context(inertia=0, nsteps=1, hard=True)
    state = _state(context)
    _configure(context.p, nsteps=1, micro_steps=2, ion_nodes=3)
    case = _write_multirate_case(
        tmp_path / "cli-case", context, state, nsteps=1
    )
    output = tmp_path / "cli-run"
    assert _run_command(["run", str(case), "--output", str(output)]) == 0
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["coupling"] == "multirate_wr"
    assert manifest["ion_nodes"] == 3
    latest = json.loads(
        (output / "checkpoints/latest.json").read_text(encoding="utf-8")
    )
    checkpoint = output / "checkpoints" / latest["file"]
    loaded = io.load_case(case)
    saved = io.read_restart(checkpoint, loaded)
    assert saved.state.step == 1
    assert saved.cumulative.accepted_intervals == 1
