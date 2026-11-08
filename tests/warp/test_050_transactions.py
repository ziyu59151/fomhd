"""Bounded real-T1 restart and rollback acceptance for CPU and owned GPU."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fomhd import io
from fomhd import time as time_module
from fomhd import time_waveform
from fomhd.__main__ import _context as command_context
from fomhd.__main__ import _run_command
from fomhd.stage import initialize
from fomhd.types import Failure
from test_gpu_massless import (
    _host_state,
    _latest_checkpoint,
    _switch_case_to_cuda,
)
from test_model_routing import _cumulative, _write_case
from test_time_waveform import _waveform_case
from workflow_fixture import assert_dimensional


T1_RTOL = 1.0e-10
T1_MAX_ITERATIONS = 12
T1_MICRO_STEPS = 2
PHYSICAL_RTOL = 2.0e-9


@pytest.fixture(params=("cpu", "cuda:0"))
def backend(request):
    if request.param == "cuda:0" and (
        os.environ.get("FOMHD_TEST_GPU") != "1"
        or not os.environ.get("SLURM_JOB_ID")
    ):
        pytest.skip(
            "CUDA regression requires FOMHD_TEST_GPU=1 and SLURM_JOB_ID"
        )
    return request.param


def _reset_case_hash(case):
    source_path = case.parent / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["input_hashes"]["case"] = hashlib.sha256(
        case.read_bytes()
    ).hexdigest()
    source_path.write_text(json.dumps(source), encoding="utf-8")


def _write_t1_case(root, context, state, nsteps, backend):
    case = _write_case(
        root,
        context.p,
        context.geometry,
        state,
        "rk4",
        nsteps,
        coupling="waveform",
        micro_steps=T1_MICRO_STEPS,
    )
    text = case.read_text(encoding="utf-8")
    lines = [
        line
        for line in text.splitlines()
        if not line.startswith(
            ("waveform_rtol=", "waveform_max_iterations=")
        )
    ]
    index = next(
        i for i, line in enumerate(lines) if line.startswith("dt_guard=")
    )
    lines[index:index] = [
        f"waveform_rtol={T1_RTOL!r}",
        f"waveform_max_iterations={T1_MAX_ITERATIONS}",
    ]
    case.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _reset_case_hash(case)
    if backend == "cuda:0":
        _switch_case_to_cuda(case)
    return case


def _assert_nonzero_initial(case):
    loaded = io.load_case(case)
    context = command_context(loaded)
    state = io.load_initial(loaded)
    host = _host_state(context, state)
    reference = loaded.parameters.ref
    assert np.max(np.abs(host["B"])) > 1.0e-6 * reference.magnetic
    assert np.max(np.abs(host["0_n"])) > 1.0e-6 * reference.ni
    assert np.max(np.abs(host["1_perp"])) > (
        1.0e-6 * reference.pressure
    )


def _state_references(parameters):
    ref = parameters.ref
    ion, electron = parameters.species
    return {
        "B": ref.magnetic,
        "0_n": ref.ni,
        "0_mom": ion.mass * ref.ni * ref.velocity,
        "0_perp": ref.pressure,
        "1_n": ref.ne,
        "1_mom": electron.mass * ref.ne * ref.velocity,
        "1_perp": ref.pressure,
        "1_parallel": ref.pressure,
    }


def _assert_state_matches(
    actual_context,
    actual_state,
    expected_context,
    expected_state,
    parameters,
):
    actual = _host_state(actual_context, actual_state)
    expected = _host_state(expected_context, expected_state)
    for name, reference in _state_references(parameters).items():
        assert_dimensional(
            actual[name],
            expected[name],
            reference,
            f"accepted_{name}",
            rtol=PHYSICAL_RTOL,
        )
    assert actual_state.step == expected_state.step
    assert actual_state.time == expected_state.time


def _assert_physical_value(actual, expected, reference, label):
    assert_dimensional(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        reference,
        label,
        rtol=PHYSICAL_RTOL,
    )


def _assert_raw_cumulative_matches(
    actual,
    expected,
    parameters,
    actual_compute,
    expected_compute,
):
    ref = parameters.ref
    cell_number = ref.ni * ref.length**3
    energy = ref.pressure * ref.length**3
    magnetic_energy = ref.magnetic**2 * ref.length**3 / parameters.mu0
    total_energy = energy + magnetic_energy
    momentum = (
        parameters.species[0].mass
        * ref.ni
        * ref.velocity
        * ref.length**3
    )
    angular_momentum = momentum * ref.length

    assert actual.accepted_intervals == expected.accepted_intervals == 2
    assert actual.last_trial > 0
    assert expected.last_trial > 0
    assert actual.last_step == expected.last_step == 2
    assert actual.last_time == expected.last_time

    for species in range(2):
        for name in vars(actual.integrals[species]):
            _assert_physical_value(
                getattr(actual.integrals[species], name),
                getattr(expected.integrals[species], name),
                energy,
                f"integrals[{species}].{name}",
            )
        _assert_physical_value(
            actual.boundary_integrals.number[species],
            expected.boundary_integrals.number[species],
            cell_number,
            f"boundary.number[{species}]",
        )
        for name in ("kinetic", "internal_perp", "internal_parallel"):
            _assert_physical_value(
                getattr(actual.boundary_integrals, name)[species],
                getattr(expected.boundary_integrals, name)[species],
                energy,
                f"boundary.{name}[{species}]",
            )
    for name in ("field_energy", "total_energy"):
        _assert_physical_value(
            getattr(actual.boundary_integrals, name),
            getattr(expected.boundary_integrals, name),
            total_energy,
            f"boundary.{name}",
        )

    actual_baseline = actual.baseline
    expected_baseline = expected.baseline
    assert actual_baseline.step == expected_baseline.step
    assert actual_baseline.time == expected_baseline.time
    assert_dimensional(
        actual_compute.to_host(actual_baseline.divB),
        expected_compute.to_host(expected_baseline.divB),
        ref.magnetic / ref.length,
        "baseline.divB",
        rtol=PHYSICAL_RTOL,
    )
    totals_reference = {
        "number": cell_number,
        "kinetic": energy,
        "ion_internal": energy,
        "electron_perp": energy,
        "electron_parallel": energy,
        "magnetic": magnetic_energy,
        "longitudinal": energy,
        "energy": total_energy,
        "topology": ref.magnetic * ref.length**2,
        "physical_flux": ref.magnetic * ref.length**2,
    }
    for name, reference in totals_reference.items():
        _assert_physical_value(
            getattr(actual_baseline.totals, name),
            getattr(expected_baseline.totals, name),
            reference,
            f"baseline.totals.{name}",
        )
    for name in ("value", "outward", "wall_reaction"):
        _assert_physical_value(
            getattr(actual_baseline.momentum, name),
            getattr(expected_baseline.momentum, name),
            momentum,
            f"baseline.momentum.{name}",
        )
    for name in ("angular", "outward_torque", "wall_torque"):
        _assert_physical_value(
            getattr(actual_baseline.momentum, name),
            getattr(expected_baseline.momentum, name),
            angular_momentum,
            f"baseline.momentum.{name}",
        )
    assert (
        actual_baseline.momentum.angular_samples
        == expected_baseline.momentum.angular_samples
    )


def test_real_t1_direct_two_macro_matches_production_restart(
    tmp_path, backend
):
    root = tmp_path / backend.replace(":", "-")
    _, context, state = _waveform_case(
        root,
        rtol=T1_RTOL,
        max_iterations=T1_MAX_ITERATIONS,
        micro_steps=T1_MICRO_STEPS,
    )
    first_case = _write_t1_case(root, context, state, 1, backend)
    _assert_nonzero_initial(first_case)
    first = root / "production-first"
    assert _run_command(
        ["run", str(first_case), "--output", str(first)]
    ) == 0
    checkpoint1 = _latest_checkpoint(first)
    first_manifest = json.loads(
        (first / "run.json").read_text(encoding="utf-8")
    )
    assert first_manifest["status"] == "completed"
    assert first_manifest["last_accepted_step"] == 1

    second_case = _write_t1_case(root, context, state, 2, backend)
    first_loaded = io.load_case(first_case)
    second_loaded = io.load_case(second_case)
    assert first_loaded.semantic_hash == second_loaded.semantic_hash
    first_restart = io.read_restart(checkpoint1, first_loaded)
    assert first_restart.state.step == 1
    assert first_restart.cumulative.accepted_intervals == 1
    second = root / "production-restart"
    assert _run_command(
        [
            "run",
            str(second_case),
            "--output",
            str(second),
            "--restart",
            str(checkpoint1),
        ]
    ) == 0
    checkpoint2 = _latest_checkpoint(second)
    restarted = io.read_restart(checkpoint2, second_loaded)
    second_manifest = json.loads(
        (second / "run.json").read_text(encoding="utf-8")
    )
    assert second_manifest["status"] == "completed"
    assert second_manifest["last_accepted_step"] == 2
    assert restarted.state.step == 2
    assert restarted.cumulative.accepted_intervals == 2
    assert restarted.cumulative.last_step == 2

    direct_case = io.load_case(second_case)
    direct_context = command_context(direct_case)
    direct_state = io.load_initial(direct_case)
    direct_ledger = _cumulative()
    committed = initialize(
        direct_state, direct_context, direct_ledger, False, {}
    )
    for _ in range(2):
        result = time_module.advance(
            committed, direct_ledger, direct_context
        )
        assert result.accepted
        assert result.interval.accepted
        metadata = result.coupling_metadata
        assert metadata["coupling"] == "waveform"
        assert metadata["converged"] is True
        assert metadata["micro_steps"] == T1_MICRO_STEPS
        assert metadata["waveform_rtol"] == T1_RTOL
        assert metadata["waveform_max_iterations"] == T1_MAX_ITERATIONS
    assert committed.state.step == 2
    assert direct_ledger.accepted_intervals == 2

    restarted_context = command_context(second_loaded)
    _assert_state_matches(
        restarted_context,
        restarted.state,
        direct_context,
        committed.state,
        direct_case.parameters,
    )
    _assert_raw_cumulative_matches(
        restarted.cumulative,
        direct_ledger,
        direct_case.parameters,
        restarted_context.compute,
        direct_context.compute,
    )


def test_real_t1_inner_stage_failure_rolls_back(
    tmp_path, backend, monkeypatch
):
    root = tmp_path / f"rollback-{backend.replace(':', '-')}"
    _, context, state = _waveform_case(
        root,
        rtol=T1_RTOL,
        max_iterations=T1_MAX_ITERATIONS,
        micro_steps=T1_MICRO_STEPS,
    )
    case = _write_t1_case(root, context, state, 2, backend)
    loaded = io.load_case(case)
    runtime = command_context(loaded)
    ledger = _cumulative()
    committed = initialize(
        io.load_initial(loaded), runtime, ledger, False, {}
    )
    accepted = time_module.advance(committed, ledger, runtime)
    assert accepted.accepted
    assert committed.state.step == 1
    assert ledger.accepted_intervals == 1

    io._check_accepted(committed)
    old_state = committed.state
    old_cache = committed.cache
    old_ledger = copy.deepcopy(io._cumulative_json(ledger))
    old_baseline = runtime.compute.to_host(
        ledger.baseline.divB
    ).copy()
    old_host = _host_state(runtime, committed.state)
    old_step = committed.state.step
    old_time = committed.state.time
    old_generation = committed.state.generation
    old_trial = ledger.last_trial

    original = time_waveform.evaluate_stage
    calls = {"count": 0}

    def fail_after_two_real_stages(state, tag, context, alias):
        calls["count"] += 1
        if calls["count"] == 3:
            failure = Failure(
                "Injected", "Injected T1 inner-stage failure"
            )
            failure.stage = tag
            raise failure
        return original(state, tag, context, alias)

    monkeypatch.setattr(
        time_waveform, "evaluate_stage", fail_after_two_real_stages
    )
    with pytest.raises(
        Failure, match="Injected T1 inner-stage failure"
    ) as caught:
        time_module.advance(committed, ledger, runtime)
    failure = caught.value

    assert calls["count"] == 3
    assert committed.state is old_state
    assert committed.cache is old_cache
    assert committed.state.step == old_step
    assert committed.state.time == old_time
    assert committed.state.generation == old_generation
    current_host = _host_state(runtime, committed.state)
    for name, value in old_host.items():
        np.testing.assert_array_equal(current_host[name], value)
    assert io._cumulative_json(ledger) == old_ledger
    np.testing.assert_array_equal(
        runtime.compute.to_host(ledger.baseline.divB), old_baseline
    )
    assert ledger.accepted_intervals == 1
    assert ledger.last_trial == old_trial
    assert ledger.last_step == old_step
    assert ledger.last_time == old_time
    io._check_accepted(committed)

    metadata = failure.coupling_metadata
    attempted = metadata["attempted_stage"]
    assert failure.stage.stage == "waveform_i000_m000_r3"
    assert failure.stage.step == 1
    assert failure.stage.iteration == 0
    assert metadata["attempted_iteration"] == 0
    assert attempted["stage"] == failure.stage.stage
    assert attempted["step"] == failure.stage.step
    assert attempted["trial"] == failure.stage.trial
    assert attempted["generation"] == failure.stage.generation
    assert failure.stage.generation != committed.state.generation
    assert failure.partial_arrays
    retained_prefix = "waveform_i000_m000_r2_"
    retained_names = [
        name for name in failure.partial_arrays
        if name.startswith(retained_prefix)
    ]
    assert retained_names
    assert all(
        isinstance(failure.partial_arrays[name], runtime.compute.xp.ndarray)
        for name in retained_names
    )

    bad_tag = SimpleNamespace(
        step=committed.state.step,
        generation=failure.stage.generation,
        time=committed.state.time,
        stage="endpoint",
    )
    stale = SimpleNamespace(
        accepted=True,
        state=committed.state,
        cache=SimpleNamespace(tag=bad_tag),
    )
    with pytest.raises(ValueError, match="matching accepted cache"):
        io._check_accepted(stale)
