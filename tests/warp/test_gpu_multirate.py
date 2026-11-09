"""Owned-GPU acceptance gates for the complete A1-WR transaction."""

from __future__ import annotations

import copy
import json
import os

import numpy as np
import pytest

from fomhd import io
from fomhd import massless_checks, massless_field, stage, time_multirate
from fomhd.__main__ import _run_command
from fomhd.geometry import build_rectangle
from fomhd.multirate_wr_checks import LEDGER_FIELDS
from fomhd.types import Failure
from test_050_transactions import (
    _assert_raw_cumulative_matches,
    _assert_state_matches,
)
from test_gpu_massless import (
    _assert_owned,
    _context_for,
    _forbid_to_host,
    _host_state,
    _latest_checkpoint,
    _parameters_for,
    _state_from_host,
    _switch_case_to_cuda,
)
from test_model_routing import (
    _cumulative,
    _state as routing_state,
    _write_case,
)


WR_RTOL = 1.0e-10
WR_MAX_ITERATIONS = 12
WR_MICRO_STEPS = 2
PHYSICAL_RTOL = 2.0e-9

GPU_ONLY = pytest.mark.skipif(
    os.environ.get("FOMHD_TEST_GPU") != "1",
    reason="explicit owned-Slurm GPU test not requested",
)


def _owned_gpu_device():
    assert os.environ.get("FOMHD_TEST_GPU") == "1", (
        "CUDA tests require FOMHD_TEST_GPU=1"
    )
    assert os.environ.get("SLURM_JOB_ID"), (
        "CUDA tests require an owned Slurm allocation"
    )
    return "cuda:0"


@pytest.fixture
def gpu_device():
    return _owned_gpu_device()


def test_gpu_gate_requires_request_and_allocation(monkeypatch):
    monkeypatch.delenv("FOMHD_TEST_GPU", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(AssertionError, match="FOMHD_TEST_GPU"):
        _owned_gpu_device()
    monkeypatch.setenv("FOMHD_TEST_GPU", "1")
    with pytest.raises(AssertionError, match="Slurm"):
        _owned_gpu_device()
    monkeypatch.setenv("SLURM_JOB_ID", "guard-only")
    assert _owned_gpu_device() == "cuda:0"


def _configure_multirate(parameters, *, nsteps, ion_nodes):
    parameters.coupling = "multirate_wr"
    parameters.integrator = "rk4"
    parameters.micro_steps = WR_MICRO_STEPS
    parameters.ion_nodes = ion_nodes
    parameters.waveform_rtol = WR_RTOL
    parameters.waveform_max_iterations = WR_MAX_ITERATIONS
    parameters.wr_initial = "tangent"
    parameters.wr_relaxation = 0.8
    parameters.nsteps = nsteps
    parameters.dt_guard = "warn"
    parameters.start_time = 1.0


def _cpu_input(*, nsteps, ion_nodes, hard=False):
    parameters = _parameters_for(
        "rk4",
        nsteps=nsteps,
        hard=hard,
    )
    _configure_multirate(
        parameters,
        nsteps=nsteps,
        ion_nodes=ion_nodes,
    )
    geometry = build_rectangle(6, 6, 0.5, -0.5, 0.2, 0.2)
    context = _context_for(parameters, geometry, "cpu")
    return context, routing_state(context)


def _context_for_device(parameters, geometry, device):
    return _context_for(parameters, geometry, device)


def _cpu_gpu_pair(*, nsteps, ion_nodes, gpu_device):
    cpu, cpu_state = _cpu_input(
        nsteps=nsteps,
        ion_nodes=ion_nodes,
    )
    source = _host_state(cpu, cpu_state)
    gpu_parameters = copy.deepcopy(cpu.p)
    gpu = _context_for_device(gpu_parameters, cpu.geometry, gpu_device)
    gpu_state = _state_from_host(gpu, source)
    gpu.compute.synchronize()
    actual = _host_state(gpu, gpu_state)
    for name, expected in source.items():
        np.testing.assert_array_equal(actual[name], expected)
    assert cpu.compute.name == "cpu"
    assert gpu.compute.name == "cuda"
    assert gpu.p is not cpu.p
    assert gpu.p.dt == cpu.p.dt
    return cpu, cpu_state, gpu, gpu_state


def test_cpu_fixture_argument_contract():
    cpu, _ = _cpu_input(nsteps=1, ion_nodes=3)
    duplicate = _context_for_device(cpu.p, cpu.geometry, "cpu")
    assert duplicate.p is cpu.p
    assert duplicate.geometry is cpu.geometry
    assert duplicate.compute.name == "cpu"


def _start(context, state):
    cumulative = _cumulative()
    cumulative.last_time = float(context.p.start_time)
    committed = stage.initialize(state, context, cumulative, False, {})
    return committed, cumulative


def _advance_started(committed, cumulative, context, macros):
    results = []
    for _ in range(macros):
        result = time_multirate.advance(
            committed,
            cumulative,
            context,
        )
        results.append(result)
        assert result.accepted
        assert result.interval.accepted
        io._check_accepted(committed)
        assert committed.state.step == result.frame.tag.step
        assert committed.state.time == result.frame.tag.time
        assert (
            committed.state.generation
            == result.frame.tag.generation
        )
        assert committed.cache is result.frame.cache
        assert committed.cache.tag == result.frame.tag
        assert cumulative.last_trial == result.interval.trial
        assert cumulative.last_step == committed.state.step
        assert cumulative.last_time == committed.state.time
    return results


def _run(context, state, macros):
    committed, cumulative = _start(context, state)
    results = _advance_started(
        committed,
        cumulative,
        context,
        macros,
    )
    return committed, cumulative, results


def _counter_delta(before, after):
    return {
        name: int(after[name]) - int(before[name])
        for name in before
    }


def _run_counted(context, state, macros, counts):
    before_setup = dict(counts)
    committed, cumulative = _start(context, state)
    setup_counts = _counter_delta(before_setup, counts)
    before_macro = dict(counts)
    results = _advance_started(
        committed,
        cumulative,
        context,
        macros,
    )
    macro_delta = _counter_delta(before_macro, counts)
    return committed, cumulative, results, setup_counts, macro_delta


def _gpu_window_arrays(context, physics):
    window = physics.window
    values = [
        window.P,
        window.Z_coarse,
        window.fine_P,
        window.fine_Z,
        window.fine_G,
        window.fine_C,
        window.coarse_A,
        window.C_integrals,
        window.fine_weights,
    ]
    _assert_owned(context, values)
    return values


def _install_window_capture(monkeypatch):
    captured = {"cpu": [], "cuda": []}
    original = time_multirate.MasslessWRPhysicsAdapter.solve_window

    def capture(adapter, *args, **kwargs):
        result = original(adapter, *args, **kwargs)
        name = adapter.context.compute.name
        captured[name].append(result)
        if name == "cuda":
            _gpu_window_arrays(adapter.context, result)
        return result

    monkeypatch.setattr(
        time_multirate.MasslessWRPhysicsAdapter,
        "solve_window",
        capture,
    )
    return captured


def _install_work_spies(monkeypatch):
    counts = {
        "power_ledger": 0,
        "electron_ledger": 0,
        "fine_work_checks": 0,
        "wall_work_metrics": 0,
    }
    originals = {
        "power_ledger": massless_checks.power_ledger,
        "electron_ledger": massless_checks._electron_ledger,
        "fine_work_checks": massless_checks._fine_electron_work_checks,
        "wall_work_metrics": massless_checks._wall_work_metrics,
    }

    def wrap(name):
        def counted(*args, **kwargs):
            counts[name] += 1
            return originals[name](*args, **kwargs)

        return counted

    monkeypatch.setattr(
        massless_checks,
        "power_ledger",
        wrap("power_ledger"),
    )
    monkeypatch.setattr(
        massless_checks,
        "_electron_ledger",
        wrap("electron_ledger"),
    )
    monkeypatch.setattr(
        massless_checks,
        "_fine_electron_work_checks",
        wrap("fine_work_checks"),
    )
    monkeypatch.setattr(
        massless_checks,
        "_wall_work_metrics",
        wrap("wall_work_metrics"),
    )
    return counts


def _host(context, value):
    return np.asarray(context.compute.to_host(value)).copy()


def _value_metric(actual, expected, reference):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    if actual.size == 0:
        return 0.0
    scale = max(
        abs(float(reference)),
        float(np.max(np.abs(actual))),
        float(np.max(np.abs(expected))),
        1.0e-300,
    )
    return float(np.max(np.abs(actual - expected))) / scale


def _assert_owned_gpu_close(
    actual,
    expected,
    reference,
    label,
    *,
    rtol=PHYSICAL_RTOL,
):
    metric = _value_metric(actual, expected, reference)
    assert metric <= rtol, f"{label}: normalized error {metric:.6e}"


def _packed_components(context, kind, *, multiplier=1.0):
    nc = context.op.Nc
    nb = context.op.Nb
    ref = context.p.ref
    momentum = (
        context.p.species[0].mass * ref.ni * ref.velocity
    )
    if kind == "P":
        values = (
            ("ion_density", slice(0, nc), ref.ni),
            ("ion_momentum", slice(nc, 4 * nc), momentum),
            ("ion_pressure", slice(4 * nc, 5 * nc), ref.pressure),
        )
    elif kind == "Z":
        values = (
            ("electron_perp", slice(0, nc), ref.pressure),
            ("electron_parallel", slice(nc, 2 * nc), ref.pressure),
            ("magnetic", slice(2 * nc, 2 * nc + nb), ref.magnetic),
        )
    else:
        raise ValueError(kind)
    return tuple(
        (name, index, reference * multiplier)
        for name, index, reference in values
    )


def _assert_packed_parity(
    label,
    gpu_context,
    gpu_value,
    cpu_context,
    cpu_value,
    kind,
    *,
    multiplier=1.0,
):
    actual = _host(gpu_context, gpu_value)
    expected = _host(cpu_context, cpu_value)
    for name, index, reference in _packed_components(
        gpu_context,
        kind,
        multiplier=multiplier,
    ):
        _assert_owned_gpu_close(
            actual[..., index],
            expected[..., index],
            reference,
            f"{label}.{name}",
        )


def _compare_window(gpu_context, gpu_window, cpu_context, cpu_window):
    inv_time = 1.0 / cpu_context.p.ref.time
    for label, gpu_value, cpu_value, kind, multiplier in (
        ("P_nodes", gpu_window.P_nodes, cpu_window.P_nodes, "P", 1.0),
        (
            "Z_coarse",
            gpu_window.Z_coarse,
            cpu_window.Z_coarse,
            "Z",
            1.0,
        ),
        ("fine_P", gpu_window.fine_P, cpu_window.fine_P, "P", 1.0),
        ("fine_Z", gpu_window.fine_Z, cpu_window.fine_Z, "Z", 1.0),
        (
            "fine_G",
            gpu_window.fine_G,
            cpu_window.fine_G,
            "Z",
            inv_time,
        ),
        (
            "fine_C",
            gpu_window.fine_C,
            cpu_window.fine_C,
            "P",
            inv_time,
        ),
        (
            "coarse_A",
            gpu_window.coarse_A,
            cpu_window.coarse_A,
            "P",
            inv_time,
        ),
        (
            "C_integrals",
            gpu_window.C_integrals,
            cpu_window.C_integrals,
            "P",
            cpu_window.H,
        ),
    ):
        _assert_packed_parity(
            label,
            gpu_context,
            gpu_value,
            cpu_context,
            cpu_value,
            kind,
            multiplier=multiplier,
        )
    _assert_owned_gpu_close(
        _host(gpu_context, gpu_window.fine_weights),
        _host(cpu_context, cpu_window.fine_weights),
        1.0,
        "fine_weights",
    )
    assert gpu_window.fine_identities == cpu_window.fine_identities
    assert gpu_window.H == cpu_window.H
    assert gpu_window.M == cpu_window.M == WR_MICRO_STEPS
    assert gpu_window.s == cpu_window.s

    fine_z = _host(gpu_context, gpu_window.fine_Z)
    fine_g = _host(gpu_context, gpu_window.fine_G)
    half_h = gpu_window.h / 2.0
    for micro_step in range(gpu_window.M):
        for rk_stage, rate_stage in ((1, 0), (2, 1)):
            asserted = (
                fine_z[micro_step, 0]
                + half_h * fine_g[micro_step, rate_stage]
            )
            for name, index, reference in _packed_components(
                gpu_context,
                "Z",
            ):
                _assert_owned_gpu_close(
                    fine_z[micro_step, rk_stage, ..., index],
                    asserted[..., index],
                    reference,
                    f"midpoint[{micro_step},{rk_stage}].{name}",
                )
    for micro_step in range(gpu_window.M):
        identities = [
            identity
            for identity in gpu_window.fine_identities
            if identity.micro_step == micro_step
            and identity.rk_stage in (1, 2)
        ]
        assert len(identities) == 2
        assert identities[0] != identities[1]


def _state_scales(parameters):
    ref = parameters.ref
    return {
        "B": ref.magnetic,
        "0_n": ref.ni,
        "0_mom": (
            parameters.species[0].mass
            * ref.ni
            * ref.velocity
        ),
        "0_perp": ref.pressure,
        "0_parallel": ref.pressure,
        "1_n": ref.ne,
        "1_mom": (
            parameters.species[1].mass
            * ref.ne
            * ref.velocity
        ),
        "1_perp": ref.pressure,
        "1_parallel": ref.pressure,
    }


def _compare_accepted_state(cpu_context, cpu_state, gpu_context, gpu_state):
    expected = _host_state(cpu_context, cpu_state)
    actual = _host_state(gpu_context, gpu_state)
    for name, reference in _state_scales(cpu_context.p).items():
        _assert_owned_gpu_close(
            actual[name],
            expected[name],
            reference,
            f"accepted.{name}",
        )


def _compare_cumulative(cpu_context, cpu_ledger, gpu_context, gpu_ledger):
    assert gpu_ledger.accepted_intervals == cpu_ledger.accepted_intervals
    assert gpu_ledger.last_step == cpu_ledger.last_step
    assert gpu_ledger.last_time == cpu_ledger.last_time
    assert gpu_ledger.last_trial == cpu_ledger.last_trial
    ref = cpu_context.p.ref
    energy = ref.pressure * ref.length**3
    number = ref.ni * ref.length**3
    magnetic_energy = (
        ref.magnetic**2 * ref.length**3 / cpu_context.p.mu0
    )
    total_energy = energy + magnetic_energy
    for species in range(2):
        for name in LEDGER_FIELDS:
            _assert_owned_gpu_close(
                getattr(gpu_ledger.integrals[species], name),
                getattr(cpu_ledger.integrals[species], name),
                energy,
                f"ledger[{species}].{name}",
            )
        _assert_owned_gpu_close(
            gpu_ledger.boundary_integrals.number[species],
            cpu_ledger.boundary_integrals.number[species],
            number,
            f"boundary.number[{species}]",
        )
        for name in ("kinetic", "internal_perp", "internal_parallel"):
            _assert_owned_gpu_close(
                getattr(gpu_ledger.boundary_integrals, name)[species],
                getattr(cpu_ledger.boundary_integrals, name)[species],
                energy,
                f"boundary.{name}[{species}]",
            )
    for name, reference in (
        ("field_energy", magnetic_energy),
        ("total_energy", total_energy),
    ):
        _assert_owned_gpu_close(
            getattr(gpu_ledger.boundary_integrals, name),
            getattr(cpu_ledger.boundary_integrals, name),
            reference,
            f"boundary.{name}",
        )


def _assert_actual_work_spies(counts, results, setup_counts):
    work = [result.coupling_metadata["work"] for result in results]
    bundle = [
        result.coupling_metadata["bundle_diagnostic_work"]
        for result in results
    ]
    result_work = [
        result.coupling_metadata["result_diagnostic_work"]
        for result in results
    ]
    full_calls = sum(row["total_full_stage_calls"] for row in work)
    assert setup_counts
    assert all(
        row["power_ledger"] == 1
        and row["electron_ledger"] == 1
        and row["wall_work_metrics"] == 1
        and row["fine_work_checks"] == 0
        for row in setup_counts
    )
    assert counts["power_ledger"] == full_calls
    assert sum(
        row["full_ion_power_ledger_values"] for row in work
    ) == full_calls
    assert sum(
        row["full_electron_power_ledger_values"] for row in work
    ) == full_calls
    electron_calls = sum(
        row["electron_ledger_calls"] for row in bundle
    ) + sum(row["electron_ledger_calls"] for row in result_work)
    assert counts["electron_ledger"] == full_calls + electron_calls
    work_gate_calls = sum(
        row["electron_work_gate_calls"] for row in result_work
    )
    assert counts["fine_work_checks"] == work_gate_calls
    assert counts["wall_work_metrics"] == full_calls + sum(
        row["wall_work_metric_calls"] for row in result_work
    )
    assert full_calls > 0
    assert electron_calls > 0
    assert work_gate_calls > 0
    total_observed = {
        name: int(counts[name])
        + sum(int(row[name]) for row in setup_counts)
        for name in counts
    }
    assert total_observed["power_ledger"] == (
        counts["power_ledger"] + len(setup_counts)
    )


@GPU_ONLY
@pytest.mark.parametrize(("macros", "ion_nodes"), ((1, 3), (2, 5)))
def test_complete_wr_cpu_cuda_parity(
    macros, ion_nodes, gpu_device, monkeypatch
):
    cpu, cpu_state, gpu, gpu_state = _cpu_gpu_pair(
        nsteps=macros,
        ion_nodes=ion_nodes,
        gpu_device=gpu_device,
    )
    captured = _install_window_capture(monkeypatch)
    counts = _install_work_spies(monkeypatch)

    (
        cpu_committed,
        cpu_ledger,
        cpu_results,
        cpu_setup,
        cpu_macro,
    ) = _run_counted(
        cpu,
        cpu_state,
        macros,
        counts,
    )
    cpu_dt = float(cpu.p.dt)
    with _forbid_to_host(gpu.compute):
        (
            gpu_committed,
            gpu_ledger,
            gpu_results,
            gpu_setup,
            gpu_macro,
        ) = _run_counted(
            gpu,
            gpu_state,
            macros,
            counts,
        )
    gpu.compute.synchronize()
    assert cpu.p.dt == gpu.p.dt == cpu_dt

    assert len(captured["cpu"]) == len(captured["cuda"]) == macros
    for cpu_result, gpu_result, cpu_window, gpu_window in zip(
        cpu_results,
        gpu_results,
        captured["cpu"],
        captured["cuda"],
    ):
        assert gpu_result.coupling_metadata["coupling"] == (
            "multirate_wr"
        )
        assert gpu_result.coupling_metadata["micro_steps"] == (
            WR_MICRO_STEPS
        )
        assert gpu_result.coupling_metadata["ion_nodes"] == ion_nodes
        assert gpu_result.coupling_metadata["waveform_rtol"] == WR_RTOL
        assert gpu_result.coupling_metadata[
            "waveform_max_iterations"
        ] == WR_MAX_ITERATIONS
        assert gpu_result.frame.tag == cpu_result.frame.tag
        assert gpu_result.frame.cache.tag == cpu_result.frame.cache.tag
        _compare_window(gpu, gpu_window.window, cpu, cpu_window.window)

    _compare_accepted_state(
        cpu,
        cpu_committed.state,
        gpu,
        gpu_committed.state,
    )
    _compare_cumulative(cpu, cpu_ledger, gpu, gpu_ledger)
    assert cpu_committed.state.step == gpu_committed.state.step == macros
    assert cpu_ledger.accepted_intervals == (
        gpu_ledger.accepted_intervals
    ) == macros
    assert cpu_committed.cache.tag == gpu_committed.cache.tag
    macro_delta = {
        name: cpu_macro[name] + gpu_macro[name]
        for name in cpu_macro
    }
    _assert_actual_work_spies(
        macro_delta,
        cpu_results + gpu_results,
        [cpu_setup, gpu_setup],
    )


def test_cpu_actual_producer_setup_and_macro_delta(monkeypatch):
    cpu, cpu_state = _cpu_input(nsteps=1, ion_nodes=3)
    counts = _install_work_spies(monkeypatch)
    (
        _,
        _,
        results,
        setup_counts,
        macro_delta,
    ) = _run_counted(
        cpu,
        cpu_state,
        1,
        counts,
    )
    assert setup_counts["power_ledger"] == 1
    assert setup_counts["electron_ledger"] == 1
    assert setup_counts["wall_work_metrics"] == 1
    _assert_actual_work_spies(
        macro_delta,
        results,
        [setup_counts],
    )
    assert counts["power_ledger"] == (
        setup_counts["power_ledger"] + macro_delta["power_ledger"]
    )


def test_cpu_midpoint_stage_recurrence(monkeypatch):
    assert PHYSICAL_RTOL == 2.0e-9
    cpu, cpu_state = _cpu_input(nsteps=1, ion_nodes=3)
    captured = _install_window_capture(monkeypatch)
    _run(cpu, cpu_state, 1)
    window = captured["cpu"][0].window
    fine_z = _host(cpu, window.fine_Z)
    fine_g = _host(cpu, window.fine_G)
    half_h = window.h / 2.0
    for micro_step in range(window.M):
        for rk_stage, rate_stage in ((1, 0), (2, 1)):
            expected = (
                fine_z[micro_step, 0]
                + half_h * fine_g[micro_step, rate_stage]
            )
            for name, index, reference in _packed_components(cpu, "Z"):
                _assert_owned_gpu_close(
                    fine_z[micro_step, rk_stage, ..., index],
                    expected[..., index],
                    reference,
                    f"cpu-midpoint[{micro_step},{rk_stage}].{name}",
                )


def test_cpu_multirate_case_writer_uses_valid_massless_gates(tmp_path):
    from test_time_multirate import _write_multirate_case

    context, state = _cpu_input(
        nsteps=2,
        ion_nodes=3,
        hard=True,
    )
    case = _write_multirate_case(
        tmp_path / "cpu-case",
        context,
        state,
        nsteps=1,
        ion_nodes=3,
    )
    loaded = io.load_case(case)
    assert loaded.parameters.coupling == "multirate_wr"
    assert loaded.parameters.ion_nodes == 3
    assert loaded.parameters.micro_steps == WR_MICRO_STEPS
    assert loaded.parameters.waveform_rtol == WR_RTOL
    assert loaded.parameters.waveform_max_iterations == WR_MAX_ITERATIONS
    assert loaded.parameters.budgets["quasineutrality.linf"].hard is True
    assert loaded.parameters.budgets["theta_consistency.linf"].hard is False


def _prepare_gpu_rollback(gpu_device, monkeypatch):
    _, _, context, state = _cpu_gpu_pair(
        nsteps=2,
        ion_nodes=3,
        gpu_device=gpu_device,
    )
    committed, ledger = _start(context, state)
    accepted = time_multirate.advance(committed, ledger, context)
    assert accepted.accepted
    old_state = committed.state
    old_cache = committed.cache
    old_ledger = copy.deepcopy(io._cumulative_json(ledger))
    old_host = _host_state(context, committed.state)
    return context, committed, ledger, old_state, old_cache, old_ledger, old_host


def _assert_gpu_rollback(
    context,
    committed,
    ledger,
    old_state,
    old_cache,
    old_ledger,
    old_host,
):
    context.compute.synchronize()
    assert committed.state is old_state
    assert committed.cache is old_cache
    assert committed.accepted is True
    io._check_accepted(committed)
    current = _host_state(context, committed.state)
    for name, expected in old_host.items():
        np.testing.assert_array_equal(current[name], expected)
    assert io._cumulative_json(ledger) == old_ledger
    assert context.compute.name == "cuda"


@GPU_ONLY
@pytest.mark.parametrize("failure_point", ("fine", "coarse", "endpoint"))
def test_cuda_complete_transaction_failure_rolls_back(
    failure_point, gpu_device, monkeypatch
):
    (
        context,
        committed,
        ledger,
        old_state,
        old_cache,
        old_ledger,
        old_host,
    ) = _prepare_gpu_rollback(gpu_device, monkeypatch)

    if failure_point == "fine":
        original = massless_field.evaluate

        def inject(value, tag, owner, alias=False):
            if "field_rk4" in tag.stage:
                failure = Failure(
                    "InjectedGpu", "injected CUDA fine failure"
                )
                failure.stage = tag
                raise failure
            return original(value, tag, owner, alias)

        monkeypatch.setattr(massless_field, "evaluate", inject)
    else:
        original = stage.evaluate_stage

        def inject(value, tag, owner, alias=False):
            expected = (
                "wr_transport_coarse"
                if failure_point == "coarse"
                else "endpoint"
            )
            if tag.stage == expected:
                failure = Failure(
                    "InjectedGpu",
                    f"injected CUDA {failure_point} failure",
                )
                failure.stage = tag
                raise failure
            return original(value, tag, owner, alias)

        monkeypatch.setattr(stage, "evaluate_stage", inject)

    with _forbid_to_host(context.compute):
        with pytest.raises(Failure, match="injected CUDA") as caught:
            time_multirate.advance(committed, ledger, context)
    failure = caught.value
    assert failure.code == "InjectedGpu"
    assert failure.stage is not None
    assert failure.multirate_evidence
    assert failure.coupling_metadata["rollback"] is True
    _assert_gpu_rollback(
        context,
        committed,
        ledger,
        old_state,
        old_cache,
        old_ledger,
        old_host,
    )


@GPU_ONLY
def test_cuda_nonconvergence_rolls_back_without_acceptance(
    gpu_device, monkeypatch
):
    (
        context,
        committed,
        ledger,
        old_state,
        old_cache,
        old_ledger,
        old_host,
    ) = _prepare_gpu_rollback(gpu_device, monkeypatch)
    context.p.waveform_rtol = 1.0e-30
    context.p.waveform_max_iterations = 1
    with _forbid_to_host(context.compute):
        with pytest.raises(Failure, match="converg|Multirate") as caught:
            time_multirate.advance(committed, ledger, context)
    failure = caught.value
    assert failure.code == "MultirateConvergence"
    assert failure.coupling_metadata["rollback"] is True
    history = failure.coupling_metadata["residual_history"]
    assert isinstance(history, list)
    assert len(history) == 1
    assert all(isinstance(record, dict) for record in history)
    _assert_gpu_rollback(
        context,
        committed,
        ledger,
        old_state,
        old_cache,
        old_ledger,
        old_host,
    )


def _write_gpu_case(root, context, state, *, nsteps, ion_nodes=3):
    from test_time_multirate import _write_multirate_case

    case = _write_multirate_case(
        root,
        context,
        state,
        nsteps=nsteps,
        ion_nodes=ion_nodes,
        relaxation=0.8,
    )
    _switch_case_to_cuda(case)
    return case


def _run_cli(case, output, checkpoint=None):
    command = ["run", str(case), "--output", str(output)]
    if checkpoint is not None:
        command.extend(["--restart", str(checkpoint)])
    assert _run_command(command) == 0
    manifest = json.loads(
        (output / "run.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert manifest["backend"] == "cuda"
    assert manifest["coupling"] == "multirate_wr"
    return _latest_checkpoint(output), manifest


@GPU_ONLY
def test_cuda_cli_checkpoint_restart_matches_uninterrupted(
    tmp_path, gpu_device
):
    context, state = _cpu_input(
        nsteps=2,
        ion_nodes=3,
        hard=True,
    )
    first_case = _write_gpu_case(
        tmp_path / "case-first",
        context,
        state,
        nsteps=1,
    )
    second_case = _write_gpu_case(
        tmp_path / "case-second",
        context,
        state,
        nsteps=2,
    )
    first_loaded = io.load_case(first_case)
    second_loaded = io.load_case(second_case)
    assert first_loaded.semantic_hash == second_loaded.semantic_hash

    first_checkpoint, first_manifest = _run_cli(
        first_case,
        tmp_path / "first",
    )
    assert first_manifest["last_accepted_step"] == 1
    saved = io.read_restart(first_checkpoint, second_loaded)
    assert saved.state.step == 1
    assert saved.cumulative.accepted_intervals == 1

    restart_checkpoint, restart_manifest = _run_cli(
        second_case,
        tmp_path / "restarted",
        first_checkpoint,
    )
    direct_checkpoint, direct_manifest = _run_cli(
        second_case,
        tmp_path / "direct",
    )
    assert restart_manifest["last_accepted_step"] == 2
    assert direct_manifest["last_accepted_step"] == 2
    restarted = io.read_restart(restart_checkpoint, second_loaded)
    direct = io.read_restart(direct_checkpoint, second_loaded)
    restarted_context = _context_for_device(
        second_loaded.parameters,
        second_loaded.geometry,
        gpu_device,
    )
    direct_context = _context_for_device(
        second_loaded.parameters,
        second_loaded.geometry,
        gpu_device,
    )
    assert restarted.cumulative.accepted_intervals == 2
    assert direct.cumulative.accepted_intervals == 2
    assert restarted.state.step == direct.state.step == 2
    _assert_state_matches(
        restarted_context,
        restarted.state,
        direct_context,
        direct.state,
        second_loaded.parameters,
    )
    _assert_raw_cumulative_matches(
        restarted.cumulative,
        direct.cumulative,
        second_loaded.parameters,
        restarted_context.compute,
        direct_context.compute,
    )

    mismatch_case = _write_gpu_case(
        tmp_path / "case-mismatch",
        context,
        state,
        nsteps=2,
        ion_nodes=5,
    )
    with pytest.raises(ValueError, match="semantic|control|ion_nodes"):
        io.read_restart(
            first_checkpoint,
            io.load_case(mismatch_case),
        )
