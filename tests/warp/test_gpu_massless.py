"""Massless GPU workflow acceptance for an owned Slurm allocation."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from fomhd import io
from fomhd.__main__ import _run_command
from fomhd.backend import Backend
from fomhd.core import allocate_state
from fomhd.geometry import build_rectangle
from fomhd.operators import build_operators
from fomhd.stage import closed_reference, initialize
from fomhd.time import advance
from fomhd.types import Context, Failure, StageTag
from test_massless_support import uniform_state
from test_model_routing import (
    _context as routing_context,
    _cumulative,
    _latest_checkpoint,
    _massless_budgets,
    _parameters,
    _write_case,
)
from workflow_fixture import assert_dimensional


GPU_ONLY = pytest.mark.skipif(
    os.environ.get("FOMHD_TEST_GPU") != "1",
    reason="explicit Slurm GPU test not requested",
)
EXPECTED_STAGE_EVALUATIONS = {"midpoint": 4, "rk4": 7}
EXPECTED_NEGATIVE_CANDIDATE_STAGES = 2
EXPECTED_CLI_STAGES = 10


@pytest.fixture
def gpu_device():
    assert os.environ.get("SLURM_JOB_ID"), (
        "GPU tests require an owned Slurm allocation"
    )
    return "cuda:0"


def _parameters_for(integrator, *, nsteps=1, hard=False):
    p = _parameters(
        inertia=0,
        integrator=integrator,
        nsteps=nsteps,
        hard=hard,
    )
    p.dt = 2.0e-13
    p.dt_guard = "warn"
    return p


def _context_for(parameters, geometry, device):
    backend = Backend(device)
    operators = build_operators(geometry, 0, backend=backend)
    return Context(
        p=parameters,
        geometry=geometry,
        op=operators,
        compute=backend,
        N=parameters.Nphi,
        next_generation=2,
    )


def _host_state(context, state):
    result = {"B": context.compute.to_host(state.B).copy()}
    for species in range(2):
        for name in ("n", "mom", "perp", "parallel"):
            result[f"{species}_{name}"] = context.compute.to_host(
                getattr(state.fluid[species], name)
            ).copy()
    return result


def _state_from_host(context, values):
    state = allocate_state(context)
    state.B = context.compute.array(values["B"], np.complex128)
    for species in range(2):
        for name in ("n", "mom", "perp", "parallel"):
            setattr(
                state.fluid[species],
                name,
                context.compute.array(
                    values[f"{species}_{name}"], np.complex128
                ),
            )
    state.step = 0
    state.generation = 1
    state.time = context.p.start_time
    return state


def _fixture(integrator, gpu_device):
    parameters = _parameters_for(integrator)
    geometry = build_rectangle(6, 6, 0.5, -0.5, 0.2, 0.2)
    cpu = _context_for(parameters, geometry, "cpu")
    cuda = _context_for(parameters, geometry, gpu_device)
    source = uniform_state(cpu)
    host = _host_state(cpu, source)
    cpu_state = _state_from_host(cpu, host)
    cuda_state = _state_from_host(cuda, host)
    np.testing.assert_array_equal(
        cuda.compute.to_host(cuda_state.B), host["B"]
    )
    for species in range(2):
        for name in ("n", "mom", "perp", "parallel"):
            np.testing.assert_array_equal(
                cuda.compute.to_host(
                    getattr(cuda_state.fluid[species], name)
                ),
                host[f"{species}_{name}"],
            )
    return cpu, cpu_state, cuda, cuda_state


@contextlib.contextmanager
def _forbid_to_host(*backends):
    saved = []

    def reject(*args, **kwargs):
        raise AssertionError("whole-array host transfer during GPU stage")

    try:
        for backend in backends:
            present = "to_host" in vars(backend)
            original = vars(backend).get("to_host")
            saved.append((backend, present, original))
            backend.to_host = reject
        yield
    finally:
        for backend, present, original in saved:
            if present:
                backend.to_host = original
            else:
                del backend.to_host


@contextlib.contextmanager
def _count_massless_stages():
    module = importlib.import_module("fomhd.massless")
    original = module.evaluate_stage
    count = {"calls": 0}

    def counted(*args, **kwargs):
        count["calls"] += 1
        return original(*args, **kwargs)

    module.evaluate_stage = counted
    try:
        yield count
    finally:
        module.evaluate_stage = original


def _run_one_step(context, state, integrator):
    cumulative = _cumulative()
    captured = {}
    with _forbid_to_host(context.compute):
        committed = initialize(
            state, context, cumulative, False, captured
        )
        with _count_massless_stages() as count:
            result = advance(committed, cumulative, context)
    assert count["calls"] == EXPECTED_STAGE_EVALUATIONS[integrator]
    assert committed.state.step == 1
    assert cumulative.accepted_intervals == 1
    assert result.accepted and result.interval.accepted
    return committed, cumulative, captured, result


def _assert_owned(context, arrays):
    xp = context.compute.xp
    for value in arrays:
        assert isinstance(value, xp.ndarray)
        if context.compute.name == "cuda":
            assert int(value.device.id) == 0


def _compare_initial(cpu, cpu_captured, gpu, gpu_captured):
    ref = cpu.p.ref
    for name, reference, unit in (
        ("E", ref.electric, "E"),
        ("E0", ref.electric, "E0"),
        ("current", ref.current, "current"),
    ):
        left = (
            cpu_captured["cache"].E
            if name == "E"
            else cpu_captured["cache"].E0
            if name == "E0"
            else cpu_captured["current"]
        )
        right = (
            gpu_captured["cache"].E
            if name == "E"
            else gpu_captured["cache"].E0
            if name == "E0"
            else gpu_captured["current"]
        )
        assert_dimensional(
            gpu.compute.to_host(right),
            cpu.compute.to_host(left),
            reference,
            unit,
        )
    flux_reference = ref.ni * ref.velocity
    for name in ("flux", "flux_hat"):
        assert_dimensional(
            gpu.compute.to_host(
                getattr(gpu_captured["fluid"][1], name)
            ),
            cpu.compute.to_host(
                getattr(cpu_captured["fluid"][1], name)
            ),
            flux_reference,
            f"electron_{name}",
        )
    assert_dimensional(
        gpu.compute.to_host(gpu_captured["fluid"][1].u),
        cpu.compute.to_host(cpu_captured["fluid"][1].u),
        ref.velocity,
        "electron_padded_ue",
    )
    rhs_reference = {
        (0, "n"): ref.ni / ref.time,
        (0, "mom"): (
            cpu.p.species[0].mass
            * ref.ni
            * ref.velocity
            / ref.time
        ),
        (0, "perp"): ref.pressure / ref.time,
        (1, "n"): ref.ne / ref.time,
        (1, "mom"): (
            cpu.p.species[1].mass
            * ref.ne
            * ref.velocity
            / ref.time
        ),
        (1, "perp"): ref.pressure / ref.time,
        (1, "parallel"): ref.pressure / ref.time,
    }
    for species in range(2):
        for name in ("n", "mom", "perp", "parallel"):
            reference = rhs_reference.get((species, name))
            if reference is None:
                continue
            assert_dimensional(
                gpu.compute.to_host(
                    getattr(gpu_captured["rhs"].fluid[species], name)
                ),
                cpu.compute.to_host(
                    getattr(cpu_captured["rhs"].fluid[species], name)
                ),
                reference,
                f"rhs_{species}_{name}",
            )
    assert_dimensional(
        gpu.compute.to_host(gpu_captured["rhs"].B),
        cpu.compute.to_host(cpu_captured["rhs"].B),
        ref.magnetic / ref.time,
        "rhs_B",
    )


def test_massless_host_fixture_preflight():
    parameters = _parameters_for("midpoint")
    geometry = build_rectangle(6, 6, 0.5, -0.5, 0.2, 0.2)
    context = _context_for(parameters, geometry, "cpu")
    state = uniform_state(context)
    tag = StageTag(0, 1, 1, 0, 0.0, "fixture")
    stage = importlib.import_module("fomhd.massless").evaluate_stage(
        state, tag, context
    )
    current = context.compute.to_host(stage.current)
    current_term = (
        np.max(np.abs(np.asarray(context.op.Cdag[0])))
        * np.max(np.abs(context.compute.to_host(state.B)))
        / context.p.mu0
    )
    roundoff = (
        4096.0
        * np.finfo(np.float64).eps
        * current_term
        * context.op.Nb
    )
    assert np.max(np.abs(current)) > 1.0e4 * roundoff
    assert stage.model == "massless_quasineutral_cgl_b"


@GPU_ONLY
@pytest.mark.parametrize("integrator", ["midpoint", "rk4"])
def test_massless_cpu_cuda_stage_and_time_parity(
    integrator, gpu_device
):
    cpu, cpu_state, gpu, gpu_state = _fixture(
        integrator, gpu_device
    )
    cpu_committed, cpu_ledger, cpu_captured, cpu_result = (
        _run_one_step(cpu, cpu_state, integrator)
    )
    gpu_committed, gpu_ledger, gpu_captured, gpu_result = (
        _run_one_step(gpu, gpu_state, integrator)
    )

    _compare_initial(cpu, cpu_captured, gpu, gpu_captured)
    assert gpu_committed.state.step == cpu_committed.state.step == 1
    assert (
        gpu_ledger.accepted_intervals
        == cpu_ledger.accepted_intervals
        == 1
    )
    for species in range(2):
        for name in ("n", "mom", "perp", "parallel"):
            reference = (
                cpu.p.ref.ni
                if name == "n"
                else cpu.p.species[species].mass
                * cpu.p.ref.ni
                * cpu.p.ref.velocity
                if name == "mom"
                else cpu.p.ref.pressure
            )
            assert_dimensional(
                gpu.compute.to_host(
                    getattr(gpu_committed.state.fluid[species], name)
                ),
                cpu.compute.to_host(
                    getattr(cpu_committed.state.fluid[species], name)
                ),
                reference,
                f"endpoint_{species}_{name}",
            )
    assert_dimensional(
        gpu.compute.to_host(gpu_committed.state.B),
        cpu.compute.to_host(cpu_committed.state.B),
        cpu.p.ref.magnetic,
        "endpoint_B",
    )
    assert gpu_result.totals.energy == pytest.approx(
        cpu_result.totals.energy,
        rel=2.0e-9,
    )
    _assert_owned(
        gpu,
        [
            gpu_state.B,
            *(
                getattr(fluid, name)
                for fluid in gpu_committed.state.fluid
                for name in ("n", "mom", "perp", "parallel")
            ),
            gpu_captured["cache"].E,
            gpu_captured["cache"].E0,
            gpu_captured["current"],
            gpu_captured["fluid"][1].flux,
            gpu_captured["fluid"][1].u,
        ],
    )


@GPU_ONLY
def test_massless_gpu_rejected_candidate_retains_evidence(
    gpu_device,
):
    _, _, context, state = _fixture("midpoint", gpu_device)
    rng = np.random.default_rng(901)
    momentum = rng.normal(
        size=state.fluid[0].mom.size
    ).astype(np.complex128)
    constraint = np.asarray(
        context.compute.to_host(context.op.T_full)
    )
    momentum -= np.linalg.pinv(constraint) @ (constraint @ momentum)
    momentum *= context.p.species[0].mass * context.p.ref.ni
    state.fluid[0].mom = context.compute.array(
        momentum, np.complex128
    )

    ledger = _cumulative()
    captured = {}
    with _forbid_to_host(context.compute):
        committed = initialize(
            state, context, ledger, False, captured
        )
    accepted = _host_state(context, committed.state)
    rate = context.compute.to_host(
        captured["rhs"].fluid[0].n
    ).real
    density = context.compute.to_host(
        committed.state.fluid[0].n
    ).real
    context.p.dt = 4.0 * float(
        np.min(density[rate < 0] / -rate[rate < 0])
    )
    old_ledger = {
        "accepted_intervals": ledger.accepted_intervals,
        "last_trial": ledger.last_trial,
        "last_step": ledger.last_step,
        "last_time": ledger.last_time,
    }
    with _forbid_to_host(context.compute):
        with _count_massless_stages() as count:
            with pytest.raises(Failure) as error:
                advance(committed, ledger, context)
    failure = error.value
    assert count["calls"] == EXPECTED_NEGATIVE_CANDIDATE_STAGES - 1
    assert failure.code == "Domain"
    assert failure.stage.stage == "midpoint"
    assert failure.stage.time == pytest.approx(context.p.dt / 2.0)
    for name in (
        "B",
        "n_i",
        "momentum_i",
        "p_i",
        "n_e",
        "momentum_e",
        "p_e",
        "p_parallel_e",
    ):
        assert f"candidate_{name}" in failure.partial_arrays
    _assert_owned(
        context,
        [
            failure.partial_arrays[f"candidate_{name}"]
            for name in (
                "B",
                "n_i",
                "momentum_i",
                "p_i",
                "n_e",
                "momentum_e",
                "p_e",
                "p_parallel_e",
            )
        ],
    )
    candidate_density = context.compute.to_host(
        failure.partial_arrays["candidate_n_i"]
    )
    assert np.any(candidate_density.real <= 0.0)
    accepted_after = _host_state(context, committed.state)
    for name in accepted:
        np.testing.assert_array_equal(accepted_after[name], accepted[name])
    assert ledger.accepted_intervals == old_ledger["accepted_intervals"]
    assert ledger.last_trial == old_ledger["last_trial"]
    assert ledger.last_step == old_ledger["last_step"]
    assert ledger.last_time == old_ledger["last_time"]


def _case_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _switch_case_to_cuda(case_path):
    case_path = Path(case_path)
    text = case_path.read_text(encoding="utf-8")
    needle = 'backend="cpu"'
    if text.count(needle) != 1:
        raise AssertionError("case must contain one CPU backend selector")
    case_path.write_text(
        text.replace(needle, 'backend="cuda"'),
        encoding="utf-8",
    )
    source_path = case_path.parent / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["input_hashes"]["case"] = _case_digest(case_path)
    source_path.write_text(json.dumps(source), encoding="utf-8")


@GPU_ONLY
def test_massless_gpu_cli_run_restart_and_model_mismatch(
    tmp_path, gpu_device, monkeypatch
):
    parameters = _parameters_for("midpoint", hard=True)
    parameters.budgets = _massless_budgets(parameters)
    geometry = build_rectangle(6, 6, 0.5, -0.5, 0.2, 0.2)
    cpu_context = _context_for(parameters, geometry, "cpu")
    state = uniform_state(cpu_context)
    root = tmp_path / "massless-gpu"
    case_path = _write_case(
        root,
        parameters,
        geometry,
        state,
        "midpoint",
        1,
    )
    _switch_case_to_cuda(case_path)
    first = root / "first"
    with _count_massless_stages() as first_count:
        assert _run_command(
            ["run", str(case_path), "--output", str(first)]
        ) == 0
    checkpoint1 = _latest_checkpoint(first)
    manifest1 = json.loads(
        (first / "run.json").read_text(encoding="utf-8")
    )
    assert manifest1["backend"] == "cuda"
    assert manifest1["model_variant"] == "massless_quasineutral_cgl_b"
    assert manifest1["last_accepted_step"] == 1

    second_case = _write_case(
        root,
        parameters,
        geometry,
        state,
        "midpoint",
        2,
    )
    _switch_case_to_cuda(second_case)
    second = root / "second"
    with _count_massless_stages() as second_count:
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
    assert (
        first_count["calls"] + second_count["calls"]
        == EXPECTED_CLI_STAGES
    )
    checkpoint2 = _latest_checkpoint(second)
    loaded = io.load_case(
        second_case, {"output": str(tmp_path / "unused")}
    )
    saved = io.read_restart(checkpoint2, loaded)
    assert saved.state.step == 2
    assert saved.cumulative.baseline.step == 0
    assert saved.cumulative.accepted_intervals == 2
    assert saved.cumulative.last_step == 2
    assert isinstance(saved.state.B, loaded.compute.xp.ndarray)
    assert int(saved.state.B.device.id) == 0
    electron_density = loaded.compute.to_host(saved.state.fluid[1].n)
    ion_density = loaded.compute.to_host(saved.state.fluid[0].n)
    np.testing.assert_allclose(
        electron_density,
        loaded.parameters.species[0].charge
        / (-loaded.parameters.species[1].charge)
        * ion_density,
        rtol=2.0e-12,
        atol=0.0,
    )

    finite = routing_context(inertia=1, nsteps=1, hard=True)
    finite_state = closed_reference(
        finite,
        finite.p.ref.ni,
        finite.p.ref.pressure,
        finite.p.ref.pressure,
        finite.p.ref.magnetic,
    )
    finite_case_path = _write_case(
        root / "finite",
        finite.p,
        finite.geometry,
        finite_state,
        "midpoint",
        1,
    )
    _switch_case_to_cuda(finite_case_path)
    finite_case = io.load_case(
        finite_case_path,
        {"output": str(tmp_path / "finite-unused")},
    )

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("restart decoded before model mismatch")

    monkeypatch.setattr(io, "decode_state", forbidden_decode)
    with pytest.raises(ValueError, match="model/applicability mismatch"):
        io.read_restart(checkpoint2, finite_case)
