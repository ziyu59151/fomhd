import importlib
import json
import math
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[2] / "src" / "fomhd"


class Failure(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.checks = []
        self.stage = None
        self.partial_arrays = {}


class Tables:
    def __init__(self, keys):
        self.keys = list(keys)
        self.R = np.asarray([1.0 + i for i, _ in self.keys])
        self.Z = np.asarray([0.5 + j for _, j in self.keys])
        self.boundary = np.zeros(len(self.keys), dtype=bool)
        self.active_to_dense = {key: i for i, key in enumerate(self.keys)}
        self.dense_to_active = np.asarray(list(range(len(self.keys))))

    def contains(self, i, j):
        return (i, j) in self.active_to_dense

    def find(self, i, j):
        return self.active_to_dense[(i, j)]


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "fomhd" or name.startswith("fomhd.")
    }
    for name in saved_modules:
        sys.modules.pop(name, None)
    package = types.ModuleType("fomhd")
    package.__path__ = [str(SOURCE)]
    sys.modules["fomhd"] = package

    type_module = types.ModuleType("fomhd.types")
    type_module.Failure = Failure
    type_module.SimpleNamespace = SimpleNamespace

    class StageTag(SimpleNamespace):
        def __init__(self, step, generation, trial, spectral_context, time,
                     stage, micro_step=-1, iteration=0, slab=0):
            super().__init__(
                step=step, generation=generation, trial=trial,
                spectral_context=spectral_context, time=time, stage=stage,
                micro_step=micro_step, iteration=iteration, slab=slab)

    type_module.StageTag = StageTag

    class FluidState(SimpleNamespace):
        def __init__(self, n=None, mom=None, perp=None, parallel=None):
            super().__init__(n=n, mom=mom, perp=perp, parallel=parallel)

    class State(SimpleNamespace):
        def __init__(self, fluid=None, B=None, step=0, generation=0, time=0.0):
            super().__init__(fluid=fluid, B=B, step=step,
                             generation=generation, time=time)

    class Species(SimpleNamespace):
        def __init__(self, id="", mass=0.0, charge=0.0, gamma=0.0,
                     n_min=0.0, face_min=0.0, p_min=0.0, parallel_min=0.0):
            super().__init__(id=id, mass=mass, charge=charge, gamma=gamma,
                             n_min=n_min, face_min=face_min, p_min=p_min,
                             parallel_min=parallel_min)

    class References(SimpleNamespace):
        def __init__(self, **values):
            super().__init__(**values)

    class Budget(SimpleNamespace):
        def __init__(self, **values):
            super().__init__(**values)

    class Parameters(SimpleNamespace):
        def __init__(self, **values):
            super().__init__(**values)

    type_module.FluidState = FluidState
    type_module.State = State
    type_module.Species = Species
    type_module.References = References
    type_module.Budget = Budget
    type_module.Parameters = Parameters
    sys.modules["fomhd.types"] = type_module

    core_module = types.ModuleType("fomhd.core")

    def combine(base, rhs, weight, tag, context):
        out = SimpleNamespace(
            fluid=[
                SimpleNamespace(
                    n=base.fluid[a].n + weight * rhs.fluid[a].n,
                    mom=base.fluid[a].mom + weight * rhs.fluid[a].mom,
                    perp=base.fluid[a].perp + weight * rhs.fluid[a].perp,
                    parallel=(
                        base.fluid[a].parallel + weight * rhs.fluid[a].parallel
                        if a
                        else None
                    ),
                )
                for a in range(2)
            ],
            B=base.B + weight * rhs.B,
            step=tag.step,
            generation=tag.generation,
            time=tag.time,
        )
        return out

    core_module.combine = combine
    core_module.time_at_step = lambda p, n: p.start_time + n * p.dt
    core_module.time_at_midpoint = lambda p, n: p.start_time + (n + 0.5) * p.dt
    sys.modules["fomhd.core"] = core_module

    stage_module = types.ModuleType("fomhd.stage")
    stage_module.evaluate_stage = None
    stage_module.retain_failure_evidence = lambda *args, **kwargs: None
    sys.modules["fomhd.stage"] = stage_module

    loaded = {
        "time": importlib.import_module("fomhd.time"),
        "io": importlib.import_module("fomhd.io"),
        "diagnostics": importlib.import_module("fomhd.diagnostics"),
    }
    # These tests use a deliberately minimal ODE double; the physical guard
    # is exercised through the real stage in test_stability.py.
    loaded["time"]._check_dt = lambda stage, context: None
    loaded["stage_module"] = stage_module
    loaded["tmp"] = tmp_path_factory.mktemp("runtime")
    try:
        yield loaded
    finally:
        for name in [
            name
            for name in list(sys.modules)
            if name == "fomhd" or name.startswith("fomhd.")
        ]:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


def _tag(step, generation, trial, time, stage):
    return SimpleNamespace(
        step=step,
        generation=generation,
        trial=trial,
        spectral_context=0,
        time=time,
        stage=stage,
    )


def _state(x):
    fluid = []
    for a in range(2):
        fluid.append(
            SimpleNamespace(
                n=np.asarray([x + a], dtype=np.complex128),
                mom=np.asarray([[x + a]], dtype=np.complex128),
                perp=np.asarray([2.0 + a], dtype=np.complex128),
                parallel=np.asarray([3.0], dtype=np.complex128) if a else None,
            )
        )
    return SimpleNamespace(fluid=fluid, B=np.asarray([4.0], dtype=np.complex128),
                           step=0, generation=0, time=0.0)


def _stage(x, tag):
    return SimpleNamespace(
        tag=tag,
        rhs=_state(x),
        cache=SimpleNamespace(tag=tag, EL=np.asarray([0.0]),
                              E=np.asarray([0.0])),
        checks=[],
        ledger=[SimpleNamespace(electric_edge=1.0), SimpleNamespace(electric_edge=2.0)],
        boundary=SimpleNamespace(
            number=[0.0, 0.0], kinetic=[0.0, 0.0],
            internal_perp=[0.0, 0.0], internal_parallel=[0.0, 0.0],
            field_energy=0.0, total_energy=0.0,
        ),
        momentum=SimpleNamespace(value=[0.0] * 3, outward=[0.0] * 3,
                                 angular=0.0, outward_torque=0.0),
        fluid=[SimpleNamespace(flux_hat=np.asarray([0.0]),
                               divergence_hat=np.asarray([0.0]))
               for _ in range(2)],
        totals=SimpleNamespace(
            number=np.asarray([1.0, 2.0]), kinetic=np.asarray([3.0, 4.0]),
            ion_internal=5.0, electron_perp=6.0, electron_parallel=7.0,
            magnetic=8.0, longitudinal=9.0, energy=42.0).__class__(
                number=np.asarray([1.0, 2.0]), kinetic=np.asarray([3.0, 4.0]),
                ion_internal=5.0, electron_perp=6.0, electron_parallel=7.0,
                magnetic=8.0, longitudinal=9.0, energy=42.0, topology=0.0,
                physical_flux=0.0),
    )


def test_midpoint_uses_half_then_full_step_and_commits_once(runtime, monkeypatch):
    time_module = runtime["time"]
    calls = []

    def evaluate(state, tag, context, alias=False):
        calls.append((tag.stage, tag.time, alias))
        return _stage(1.0, tag)

    monkeypatch.setattr(time_module, "evaluate_stage", evaluate)
    diagnostics = sys.modules["fomhd.diagnostics"]

    def check_step(left, base, midpoint, mid, right, end, cumulative,
                   context, *args, **kwargs):
        return SimpleNamespace(
            checks=[],
            totals=end.totals,
            momentum=end.momentum,
            interval=SimpleNamespace(accepted=False, weight=context.p.dt),
        )

    monkeypatch.setattr(diagnostics, "check_step", check_step)
    context = SimpleNamespace(
        p=SimpleNamespace(dt=0.2, nsteps=1, start_time=0.0,
                          integrator="midpoint", electron_inertia=1),
        compute=SimpleNamespace(synchronize=lambda: None),
        next_generation=1, next_trial=1,
    )
    committed = SimpleNamespace(
        state=_state(0.0),
        cache=SimpleNamespace(),
        accepted=True,
    )
    committed.state.step = 0
    committed.state.generation = 0
    committed.state.time = 0.0
    cumulative = SimpleNamespace(
        last_step=0, last_time=0.0, accepted_intervals=0, last_trial=0,
        baseline=SimpleNamespace(divB=np.asarray([0.0]), step=0, time=0.0),
        integrals=[SimpleNamespace(electric_edge=0.0, electric_node=0.0,
                                   pressure_kinetic=0.0, pressure_internal=0.0,
                                   magnetic=0.0, wall=0.0, convection=0.0,
                                   perp_rate=0.0, parallel_rate=0.0,
                                   spectral_kinetic=0.0, spectral_perp=0.0,
                                   spectral_parallel=0.0) for _ in range(2)],
        boundary_integrals=SimpleNamespace(
            number=[0.0, 0.0], kinetic=[0.0, 0.0],
            internal_perp=[0.0, 0.0], internal_parallel=[0.0, 0.0],
            field_energy=0.0, total_energy=0.0),
    )
    old = committed.state
    result = time_module.advance(committed, cumulative, context)
    assert calls == [("base", 0.0, False), ("midpoint", 0.1, True),
                     ("endpoint", 0.2, False)]
    assert committed.state is not old
    assert committed.state.step == 1
    assert cumulative.accepted_intervals == 1
    assert cumulative.integrals[0].electric_edge == pytest.approx(0.2)
    assert cumulative.integrals[1].electric_edge == pytest.approx(0.4)
    assert result.interval.accepted is True
    assert result.interval.weight == pytest.approx(0.2)


def test_rk4_uses_four_explicit_stages_and_weighted_quadrature(runtime,
                                                               monkeypatch):
    time_module = runtime["time"]
    seen = []
    captured = {}

    def evaluate(state, tag, context, alias=False):
        seen.append((tag.stage, tag.time, state.fluid[0].n[0].real))
        return _stage(1.0, tag)

    monkeypatch.setattr(time_module, "evaluate_stage", evaluate)
    diagnostics = sys.modules["fomhd.diagnostics"]

    def check_step(left, base, midpoint, mid, right, end, cumulative, context,
                   *args, quadrature=None, prescribed_source=None):
        captured["tags"] = [x[1].tag.stage for x in quadrature]
        captured["weights"] = [x[2] for x in quadrature]
        return SimpleNamespace(checks=[], totals=end.totals,
                               momentum=end.momentum,
                               interval=SimpleNamespace(
                                   accepted=False, weight=context.p.dt))

    monkeypatch.setattr(diagnostics, "check_step", check_step)
    context = SimpleNamespace(
        p=SimpleNamespace(dt=0.3, nsteps=1, start_time=0.0,
                          integrator="rk4", electron_inertia=1),
        compute=SimpleNamespace(synchronize=lambda: None),
        next_generation=1, next_trial=1,
    )
    committed = SimpleNamespace(state=_state(0.0), cache=SimpleNamespace(),
                                accepted=True)
    cumulative = SimpleNamespace(
        last_step=0, last_time=0.0, accepted_intervals=0, last_trial=0,
        baseline=SimpleNamespace(divB=np.asarray([0.0]), step=0, time=0.0),
        integrals=[SimpleNamespace(electric_edge=0.0, electric_node=0.0,
                                   pressure_kinetic=0.0, pressure_internal=0.0,
                                   magnetic=0.0, wall=0.0, convection=0.0,
                                   perp_rate=0.0, parallel_rate=0.0,
                                   spectral_kinetic=0.0, spectral_perp=0.0,
                                   spectral_parallel=0.0) for _ in range(2)],
        boundary_integrals=SimpleNamespace(
            number=[0.0, 0.0], kinetic=[0.0, 0.0],
            internal_perp=[0.0, 0.0], internal_parallel=[0.0, 0.0],
            field_energy=0.0, total_energy=0.0),
    )
    result = time_module.advance(committed, cumulative, context)
    assert [x[0] for x in seen] == ["base", "midpoint", "midpoint_2", "station",
                                    "endpoint"]
    assert captured["tags"] == ["base", "midpoint", "midpoint_2", "station"]
    np.testing.assert_allclose(captured["weights"],
                               [0.05, 0.1, 0.1, 0.05])
    assert result.interval.weight == pytest.approx(0.3)
    assert committed.state.step == 1
    assert cumulative.accepted_intervals == 1
    assert cumulative.integrals[0].electric_edge == pytest.approx(0.3)
    assert cumulative.integrals[1].electric_edge == pytest.approx(0.6)


def test_rk4_k3_failure_retains_completed_k2_and_k3_tag(runtime, monkeypatch):
    time_module = runtime["time"]
    retained = []

    def evaluate(state, tag, context, alias=False):
        if tag.stage == "midpoint_2":
            failure = Failure("Domain", "k3 rejected")
            failure.stage = tag
            raise failure
        return _stage(1.0, tag)

    monkeypatch.setattr(time_module, "evaluate_stage", evaluate)
    monkeypatch.setattr(
        time_module,
        "retain_failure_evidence",
        lambda failure, tag, context, state, result, prefix:
        retained.append((prefix, tag.stage, tag.time)),
    )
    monkeypatch.setattr(
        sys.modules["fomhd.diagnostics"],
        "gate",
        lambda records, required=None: None,
    )
    context = SimpleNamespace(
        p=SimpleNamespace(dt=0.3, nsteps=1, start_time=0.0,
                          integrator="rk4", electron_inertia=1),
        compute=SimpleNamespace(synchronize=lambda: None),
        next_generation=1, next_trial=1,
    )
    committed = SimpleNamespace(state=_state(0.0), cache=SimpleNamespace(),
                                accepted=True)
    cumulative = SimpleNamespace(
        last_step=0, last_time=0.0, accepted_intervals=0, last_trial=0,
        baseline=SimpleNamespace(divB=np.asarray([0.0]), step=0, time=0.0),
        integrals=[],
        boundary_integrals=SimpleNamespace(),
    )
    with pytest.raises(Failure) as caught:
        time_module.advance(committed, cumulative, context)
    assert caught.value.stage.stage == "midpoint_2"
    assert ("base", "base", 0.0) in retained
    assert any(prefix == "midpoint" and stage == "midpoint"
               for prefix, stage, _ in retained)
    assert any(prefix == "midpoint_2" and stage == "midpoint_2"
               for prefix, stage, _ in retained)
    assert any(prefix == "station" and stage == "station"
               for prefix, stage, _ in retained)
    assert any(prefix == "endpoint" and stage == "endpoint"
               for prefix, stage, _ in retained)


def test_rejected_source_payload_is_retained_before_validation(runtime,
                                                              monkeypatch):
    time_module = runtime["time"]
    retained = []

    def evaluate(state, tag, context, alias=False):
        return _stage(1.0, tag)

    def reject_source(source, context):
        raise Failure("ManufacturedSource", "rejected source")

    def source(state, tag):
        return _state(9.0)

    monkeypatch.setattr(time_module, "evaluate_stage", evaluate)
    monkeypatch.setattr(
        time_module,
        "retain_failure_evidence",
        lambda failure, tag, context, state, result, prefix:
        retained.append((prefix, tag.stage, state)),
    )
    monkeypatch.setattr(
        sys.modules["fomhd.diagnostics"], "check_source", reject_source
    )
    context = SimpleNamespace(
        p=SimpleNamespace(dt=0.1, nsteps=1, start_time=0.0,
                          integrator="midpoint", electron_inertia=1),
        compute=SimpleNamespace(synchronize=lambda: None),
        next_generation=1, next_trial=1,
    )
    committed = SimpleNamespace(state=_state(0.0), cache=SimpleNamespace(),
                                accepted=True)
    cumulative = SimpleNamespace(last_step=0, last_time=0.0,
                                 accepted_intervals=0, last_trial=0)
    with pytest.raises(Failure) as caught:
        time_module.advance(committed, cumulative, context, source)
    assert caught.value.stage.stage == "base"
    source_records = [item for item in retained if item[0].startswith("source_")]
    assert len(source_records) == 1
    assert source_records[0][0] == "source_0"
    assert source_records[0][1] == "base"
    assert source_records[0][2] is not None


def test_rejected_advance_leaves_committed_state_and_ledger_unchanged(
    runtime, monkeypatch
):
    time_module = runtime["time"]

    def evaluate(state, tag, context, alias=False):
        if tag.stage == "midpoint":
            failure = Failure("Domain", "midpoint rejected")
            failure.stage = tag
            raise failure
        return _stage(1.0, tag)

    monkeypatch.setattr(time_module, "evaluate_stage", evaluate)
    monkeypatch.setattr(
        sys.modules["fomhd.diagnostics"],
        "gate",
        lambda records, required=None: None,
    )
    context = SimpleNamespace(
        p=SimpleNamespace(dt=0.1, nsteps=1, start_time=0.0,
                          integrator="midpoint", electron_inertia=1),
        compute=SimpleNamespace(synchronize=lambda: None),
        next_generation=1, next_trial=1,
    )
    state = _state(0.0)
    committed = SimpleNamespace(state=state, cache=SimpleNamespace(),
                                accepted=True)
    cumulative = SimpleNamespace(last_step=0, last_time=0.0,
                                 accepted_intervals=0, last_trial=0)
    old_state = committed.state
    old_values = vars(cumulative).copy()
    with pytest.raises(Failure) as caught:
        time_module.advance(committed, cumulative, context)
    assert caught.value.stage.stage == "midpoint"
    assert committed.state is old_state
    assert vars(cumulative) == old_values


def test_time_routes_massless_branch_into_transaction_validation(runtime):
    time_module = runtime["time"]
    context = SimpleNamespace(
        p=SimpleNamespace(electron_inertia=0, integrator="midpoint"),
        compute=SimpleNamespace(),
    )
    with pytest.raises(Failure) as caught:
        time_module.advance(None, None, context)
    assert caught.value.code == "Time"


def test_load_case_rejects_unknown_key_before_creating_output(runtime, tmp_path):
    io = runtime["io"]
    case = tmp_path / "case.toml"
    case.write_text(
        'schema_version=1\nmodel="A0"\nunits="SI"\nlayout_version=1\n'
        'case_name="x"\nunexpected=1\n',
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="unknown key case.unexpected"):
        io.load_case(case)
    assert not (tmp_path / "run").exists()


def test_safe_npz_rejects_object_dtype(runtime, tmp_path):
    io = runtime["io"]
    path = tmp_path / "bad.npz"
    np.savez(path, payload=np.asarray([object()], dtype=object))
    limits = io.Limits()
    with pytest.raises(Exception, match="object|pickle|unsupported"):
        io.read_npz(path, limits, {"payload"})


def test_atomic_checkpoint_pointer_is_published_after_file(runtime, tmp_path):
    io = runtime["io"]
    target = tmp_path / "checkpoint.npz"
    payload = {"value": np.asarray([3.0], dtype=np.float64)}
    def write(path):
        with path.open("r+b") as handle:
            np.savez(handle, **payload)

    digest = io.atomic_publish(target, write)
    assert io.sha256_file(target) == digest
    with np.load(target, allow_pickle=False) as loaded:
        np.testing.assert_array_equal(loaded["value"], [3.0])


def test_atomic_publish_fault_preserves_target_and_unpublished_temp(runtime,
                                                                   tmp_path):
    io = runtime["io"]
    target = tmp_path / "latest.json"
    target.write_bytes(b"old")

    def fault(phase):
        if phase == "written":
            raise RuntimeError("injected")

    with pytest.raises(io.PublicationError) as caught:
        io.atomic_publish(
            target,
            lambda path: path.write_bytes(b"new"),
            replace=True,
            fault=fault,
        )
    assert target.read_bytes() == b"old"
    assert not caught.value.published
    assert caught.value.temporary.exists()


def test_failure_record_uses_typed_nonfinite_values(runtime):
    io = runtime["io"]
    assert io.typed_value(float("nan")) == {"value": "NaN", "type": "float64"}
    assert io.typed_value(float("inf")) == {"value": "+Inf", "type": "float64"}


def test_gate_rejects_duplicate_and_unavailable_hard_records(runtime):
    diagnostics = importlib.reload(importlib.import_module("fomhd.diagnostics"))
    tag = _tag(0, 0, 1, 0.0, "base")
    base = SimpleNamespace(
        name="finite", population="", component="", producer="diagnostics",
        region="all", norm="value", tag=tag, hard=True, available=True,
        passed=True, raw=0.0, threshold=0.0,
    )
    with pytest.raises(Failure):
        diagnostics.gate([base, base])
    base.available = False
    with pytest.raises(Failure):
        diagnostics.gate([base])


def test_gate_records_missing_diagnostic_raises_failure(runtime):
    diagnostics = importlib.reload(importlib.import_module("fomhd.diagnostics"))
    required = diagnostics._record(
        name="missing",
        tag=_tag(0, 0, 1, 0.0, "base"),
    )
    with pytest.raises(Failure) as caught:
        diagnostics.gate_records([], [required])
    assert caught.value.code == "MissingDiagnostic"


def test_parameter_defaults_accept_integrator_contract():
    source = (SOURCE / "io.py").read_text(encoding="utf-8")
    assert "integrator" in source and "dt_guard" in source
    assert "allow_pickle=False" in source


def test_config_defaults_use_massless_rk4_and_preserve_explicit_finite_midpoint(
    runtime,
):
    io = runtime["io"]
    species = {
        "ion_id": "ion", "electron_id": "electron", "qi": 1.0, "mi": 1.0,
        "qe": -1.0, "me": 1.0, "gamma_i": 5.0 / 3.0,
    }
    solver = {
        "rank_rtol": 1.0e-10,
        "memory_limit_bytes": 1 << 20,
        "device_memory_limit_bytes": 1 << 20,
        "constants": {
            "epsilon0_Fpm": 1.0, "mu0_Hpm": 1.0, "version": "test",
        },
        "references": {
            "L_m": 1.0, "t_s": 1.0, "n_i_m3": 1.0, "n_e_m3": 1.0,
            "u_mps": 1.0, "E_Vpm": 1.0, "B_T": 1.0, "J_Apm2": 1.0,
            "rho_Cpm3": 1.0, "p_Pa": 1.0,
        },
    }
    grid = {"K": 0, "Nphi": 3}
    time = {"dt": 0.1, "start_time": 0.0, "nsteps": 1}
    budgets = {
        "domain": {
            "n_min_i": 0.0, "n_min_e": 0.0,
            "n_face_min_i": 0.0, "n_face_min_e": 0.0,
            "p_min_i": 0.0, "p_min_perp_e": 0.0,
            "p_min_parallel_e": 0.0, "B_min": 0.0,
        },
        "checks": {},
    }
    diagnostics = {"inner_distance_m": 0.1}
    parameters = io._build_parameters(
        species, solver, grid, time, budgets, diagnostics
    )
    assert parameters.electron_inertia == 0
    assert parameters.species[1].mass == 1.0
    assert parameters.integrator == "rk4"
    assert parameters.dt_guard == "error"
    assert parameters.minimum_samples_per_cycle == 32.0
    assert parameters.max_amplitude_error == 0.01
    assert parameters.max_phase_error_rad == 0.1
    species["electron_inertia"] = 1
    time.update(
        integrator="midpoint",
        minimum_samples_per_cycle=48.0,
        max_amplitude_error=0.005,
        max_phase_error_rad=0.05,
    )
    configured = io._build_parameters(
        species, solver, grid, time, budgets, diagnostics
    )
    assert configured.integrator == "midpoint"
    assert configured.electron_inertia == 1
    assert configured.minimum_samples_per_cycle == 48.0
    assert configured.max_amplitude_error == 0.005
    assert configured.max_phase_error_rad == 0.05


def test_measure_uses_device_weighted_l2_reference(runtime):
    diagnostics = importlib.reload(importlib.import_module("fomhd.diagnostics"))
    tag = _tag(0, 0, 1, 0.0, "base")
    weights = np.asarray([2.0, 4.0])
    budgets = {}
    for norm in ("linf", "weighted_L2"):
        budgets[f"probe.{norm}"] = SimpleNamespace(
            atol=0.0, rtol=1.0e-12, reference=1.5, unit="V/m",
            norm=norm, hard=False,
        )
    cells = Tables([(0, 0), (1, 0)])
    geometry = SimpleNamespace(
        walls=[],
        cells=cells,
        z_faces=Tables([(0, 0), (1, 0)]),
        r_faces=Tables([(0, 0), (1, 0)]),
        vertices=Tables([(0, 0), (1, 0)]),
        wv=np.asarray([1.0, 1.0]),
        cell_wall_distance=np.asarray([0.2, 0.3]),
    )
    backend = SimpleNamespace(
        xp=np, scalar=lambda value: value, to_host=np.asarray,
    )
    context = SimpleNamespace(
        p=SimpleNamespace(budgets=budgets, inner_distance=0.1),
        geometry=geometry,
        op=SimpleNamespace(
            M0=weights,
            M1=np.asarray([1.0, 1.0, 1.0, 1.0]),
            MB=np.asarray([1.0, 1.0]),
            Nc=2,
            T=SimpleNamespace(shape=(0, 3 * 2)),
            T_full=SimpleNamespace(shape=(0, 3 * 2)),
        ),
        compute=backend,
        N=3,
    )
    records = []
    diagnostics.measure(
        records,
        "probe",
        np.asarray([1.0 + 0.0j, 1.0 + 0.0j]),
        weights,
        "V/m",
        tag,
        context,
    )
    weighted = next(x for x in records
                    if x.region == "all" and x.norm == "weighted_L2")
    expected = math.sqrt(2 * math.pi * (weights * np.ones(2)).sum())
    assert weighted.raw == pytest.approx(expected, rel=1e-14)
    assert weighted.reference == pytest.approx(1.5 * expected, rel=1e-14)
    assert weighted.selected == 2


def test_diagnostic_inverse_defaults_to_physical_real_checks(runtime,
                                                             monkeypatch):
    diagnostics = importlib.reload(importlib.import_module("fomhd.diagnostics"))
    calls = []

    def inverse(values, entities, context, physical, reference):
        calls.append((entities, physical, reference))
        return values

    monkeypatch.setattr(
        sys.modules["fomhd.core"], "inverse", inverse, raising=False
    )
    context = SimpleNamespace()
    diagnostics._inverse(context, np.asarray([1.0j]), 1)
    diagnostics._inverse(context, np.asarray([1.0j]), 1, False)
    assert calls == [(1, True, 0.0), (1, False, 0.0)]


def test_field_exchange_converts_integrated_energy_to_watts(runtime):
    diagnostics = importlib.reload(importlib.import_module("fomhd.diagnostics"))
    dt = 1.0e-12
    balanced = 0.0 * dt
    injected_integrated = 2.0 * dt
    assert diagnostics._integrated_power_to_rate(1.0 * dt, dt) == 1.0
    assert diagnostics._integrated_power_to_rate(injected_integrated, dt) == 2.0
    budget = SimpleNamespace(
        atol=1.0e-12, rtol=1.0e-9, reference=1.0,
        unit="W", norm="value", hard=True,
    )
    context = SimpleNamespace(
        p=SimpleNamespace(budgets={"field_exchange.value": budget}),
        compute=SimpleNamespace(xp=np, scalar=lambda value: value,
                                to_host=np.asarray),
    )
    tag = _tag(0, 1, 1, 0.5e-12, "interval")
    passed = []
    diagnostics.scalar_check(
        passed, "field_exchange",
        diagnostics._integrated_power_to_rate(balanced, dt),
        "W", tag, context,
    )
    assert passed[0].passed
    failed = []
    diagnostics.scalar_check(
        failed, "field_exchange",
        diagnostics._integrated_power_to_rate(injected_integrated, dt),
        "W", tag, context,
    )
    assert not failed[0].passed
    with pytest.raises(Failure):
        diagnostics.gate(failed)


def test_wall_trace_uses_normal_weights_and_unsigned_electric_trace(runtime,
                                                                   monkeypatch):
    diagnostics = importlib.reload(importlib.import_module("fomhd.diagnostics"))
    xp = np
    cell_keys = [(0, 0), (1, 0)]
    cells = Tables(cell_keys)
    z_faces = Tables([(0, 0)])
    vertices = Tables([(0, 0), (1, 0)])
    r_faces = Tables([(0, 0)])
    geometry = SimpleNamespace(
        cells=cells, z_faces=z_faces, vertices=vertices, r_faces=r_faces,
        walls=[SimpleNamespace(
            component="z_face", face=0, sign=-1, near_cell=0, next_cell=1,
            vertices=(0, 1), area_per_radian=1.0,
        )],
    )
    identity = np.zeros((4, 6), dtype=np.complex128)
    identity[0, 0] = 1
    identity[1, 1] = 1
    identity[3, 4] = 0.5
    identity[3, 5] = 0.5
    ion = SimpleNamespace(
        n=np.asarray([1.0, 2.0], dtype=np.complex128),
        mom=np.asarray([3, 4, 5, 6, 7, 8], dtype=np.complex128),
        perp=np.asarray([9.0, 11.0], dtype=np.complex128),
        parallel=np.zeros(0, dtype=np.complex128),
        flux=np.zeros(4, dtype=np.complex128),
        pressure_flux=np.zeros(4, dtype=np.complex128),
        parallel_flux=np.zeros(0, dtype=np.complex128),
    )
    electron = SimpleNamespace(
        n=np.asarray([1.0, 2.0], dtype=np.complex128),
        mom=np.asarray([3, 4, 5, 6, 7, 8], dtype=np.complex128),
        perp=np.asarray([9.0, 11.0], dtype=np.complex128),
        parallel=np.asarray([13.0, 15.0], dtype=np.complex128),
        flux=np.zeros(4, dtype=np.complex128),
        pressure_flux=np.zeros(4, dtype=np.complex128),
        parallel_flux=np.zeros(4, dtype=np.complex128),
    )
    result = SimpleNamespace(
        fluid=[ion, electron],
        cache=SimpleNamespace(EL=np.asarray([7.0]), ET=np.asarray([11.0])),
    )
    state = SimpleNamespace(B=np.asarray([2.0, 0.0, 0.0, 0.0]))
    context = SimpleNamespace(
        N=1,
        p=SimpleNamespace(
            B_min=0.0, mu0=1.0, epsilon0=1.0,
            electron_inertia=1,
            ref=SimpleNamespace(electric=1.0),
            species=(
                SimpleNamespace(id="ion", mass=1.0, gamma=2.0, face_min=0.0),
                SimpleNamespace(id="electron", mass=1.0, gamma=0.0,
                                face_min=0.0),
            ),
        ),
        geometry=geometry,
            op=SimpleNamespace(
                Nc=2, I=identity, M1=np.ones(4), MB=np.ones(4)
            ),
        compute=SimpleNamespace(
            xp=xp,
            scalar=lambda value: complex(value).real,
            to_host=np.asarray,
        ),
    )
    inverse_values = iter(
        (
            state.B,
            np.asarray([0.0, 0.0, 0.0, 7.0]),
            np.asarray([0.0, 0.0, 0.0, 11.0]),
        )
    )
    monkeypatch.setattr(
        diagnostics,
        "_inverse",
        lambda *args, **kwargs: next(inverse_values),
    )
    boundary = diagnostics.boundary_ledger(state, result, context)
    trace = boundary.traces[0]
    density = 1.5
    np.testing.assert_allclose(
        [float(value) for value in trace.velocity[0]],
        [3.5 / density, 5.5 / density, 7.5 / density],
        rtol=0.0,
        atol=0.0,
    )
    assert float(trace.EL[2]) == 7.0
    assert float(trace.ET[2]) == 11.0


@pytest.mark.parametrize("divergence,rejected", [(5.0e-22, False), (2.0e-21, True)])
def test_check_source_divergence_uses_exact_subunit_physical_scale(
    runtime, monkeypatch, divergence, rejected
):
    diagnostics = importlib.reload(importlib.import_module("fomhd.diagnostics"))
    zeros = lambda: np.zeros(1, dtype=np.complex128)
    source = SimpleNamespace(
        fluid=[
            SimpleNamespace(n=zeros(), mom=np.zeros(3, dtype=np.complex128), perp=zeros(),
                           parallel=zeros()),
            SimpleNamespace(n=zeros(), mom=np.zeros(3, dtype=np.complex128), perp=zeros(),
                           parallel=zeros()),
        ],
        B=zeros(),
    )
    context = SimpleNamespace(
        K=0,
        N=1,
        p=SimpleNamespace(
            K=0,
            electron_inertia=1,
            ref=SimpleNamespace(
                ni=1.0, ne=1.0, pressure=1.0, velocity=1.0,
                magnetic=1.0e-12, length=1.0, time=1.0,
            ),
            species=(
                SimpleNamespace(mass=1.0), SimpleNamespace(mass=1.0)
            ),
        ),
        op=SimpleNamespace(Nc=1, MB=np.ones(1), T_full=np.zeros((0, 3)),
                           DB=np.ones((1, 1))),
        compute=SimpleNamespace(xp=np, scalar=lambda value: np.asarray(value).item(),
                                to_host=np.asarray),
    )
    monkeypatch.setattr(
        diagnostics, "_mode_apply", lambda *args: np.asarray([divergence])
    )
    if rejected:
        with pytest.raises(Failure, match="magnetic source must preserve divergence") as caught:
            diagnostics.check_source(source, context)
        assert caught.value.code == "ManufacturedSource"
    else:
        assert diagnostics.check_source(source, context)


def test_state_encoder_round_trip_preserves_compact_layout(runtime):
    io = runtime["io"]
    type_module = sys.modules["fomhd.types"]
    FluidState = type_module.FluidState
    State = type_module.State
    cells = Tables([(0, 0), (1, 0)])
    z_faces = Tables([(0, 0), (1, 0)])
    vertices = Tables([(0, 0), (1, 0)])
    r_faces = Tables([(0, 0), (1, 0)])
    geometry = SimpleNamespace(
        cells=cells, z_faces=z_faces, vertices=vertices, r_faces=r_faces,
        R=np.asarray([1.0, 2.0, 3.0]), Z=np.asarray([0.0, 1.0]),
        MB=np.asarray([1.0] * 6),
    )
    n = np.asarray([1 + 0j, 2 + 0j])
    mom = np.asarray([0.1 + 0j, 0.2 + 0j, 0.3 + 0j,
                      0.4 + 0j, 0.5 + 0j, 0.6 + 0j])
    state = State(
        fluid=(
            FluidState(n=n, mom=mom, perp=n.copy(),
                       parallel=np.zeros(0, dtype=np.complex128)),
            FluidState(n=2 * n, mom=2 * mom, perp=3 * n, parallel=4 * n),
        ),
        B=np.arange(6, dtype=np.complex128),
        step=7, generation=9, time=0.7,
    )
    arrays = io.encode_state(state, geometry)
    decoded = io.decode_state(arrays, geometry, 0)
    np.testing.assert_array_equal(decoded.fluid[0].n, n)
    np.testing.assert_array_equal(decoded.fluid[0].mom, mom)
    assert decoded.fluid[0].parallel.shape == (0,)
    np.testing.assert_array_equal(decoded.fluid[1].parallel, 4 * n)
    np.testing.assert_array_equal(decoded.B, state.B)


def test_codec_state_upload_is_explicit_and_contiguous(runtime):
    io = runtime["io"]
    type_module = sys.modules["fomhd.types"]
    FluidState = type_module.FluidState
    State = type_module.State
    calls = []

    class Backend:
        def array(self, value, dtype=None):
            calls.append((tuple(value.shape), np.dtype(dtype)))
            return np.ascontiguousarray(value, dtype=dtype)

    host_state = State(
        fluid=(
            FluidState(
                n=np.ones(2, dtype=np.complex128),
                mom=np.ones(6, dtype=np.complex128),
                perp=np.ones(2, dtype=np.complex128),
                parallel=np.zeros(0, dtype=np.complex128),
            ),
            FluidState(
                n=np.ones(2, dtype=np.complex128),
                mom=np.ones(6, dtype=np.complex128),
                perp=np.ones(2, dtype=np.complex128),
                parallel=np.ones(2, dtype=np.complex128),
            ),
        ),
        B=np.ones(4, dtype=np.complex128),
    )
    uploaded = io._upload_state(host_state, Backend())
    assert len(calls) == 9
    assert all(dtype == np.dtype(np.complex128) for _, dtype in calls)
    assert uploaded.fluid[0].parallel.shape == (0,)
    assert all(
        value.flags.c_contiguous
        for fluid in uploaded.fluid
        for value in (fluid.n, fluid.mom, fluid.perp, fluid.parallel)
    )
    assert uploaded.B.flags.c_contiguous


def test_checkpoint_and_restart_preserve_ledger_and_reject_tamper(runtime,
                                                                 tmp_path):
    io = runtime["io"]
    type_module = sys.modules["fomhd.types"]
    FluidState = type_module.FluidState
    State = type_module.State
    cells = Tables([(0, 0), (1, 0)])
    z_faces = Tables([(0, 0), (1, 0)])
    vertices = Tables([(0, 0), (1, 0)])
    r_faces = Tables([(0, 0), (1, 0)])
    geometry = SimpleNamespace(
        cells=cells, z_faces=z_faces, vertices=vertices, r_faces=r_faces,
        R=np.asarray([1.0, 2.0, 3.0]), Z=np.asarray([0.0, 1.0]),
        MB=np.asarray([1.0] * 6),
    )
    n = np.asarray([1 + 0j, 2 + 0j])
    mom = np.zeros(6, dtype=np.complex128)
    state = State(
        fluid=(
            FluidState(n=n, mom=mom, perp=n.copy(),
                       parallel=np.zeros(0, dtype=np.complex128)),
            FluidState(n=n, mom=mom, perp=n.copy(), parallel=n.copy()),
        ),
        B=np.arange(6, dtype=np.complex128),
        step=1, generation=3, time=1e-9,
    )
    cache = SimpleNamespace(tag=_tag(1, 3, 1, 1e-9, "endpoint"))
    committed = SimpleNamespace(state=state, cache=cache, accepted=True)
    ledger = SimpleNamespace(**{
        name: 0.0 for name in (
            "electric_node", "electric_edge", "pressure_kinetic",
            "pressure_internal", "magnetic", "wall", "convection",
            "perp_rate", "parallel_rate", "spectral_kinetic",
            "spectral_perp", "spectral_parallel")
    })
    boundary = SimpleNamespace(
        number=[0.0, 0.0], kinetic=[0.0, 0.0],
        internal_perp=[0.0, 0.0], internal_parallel=[0.0, 0.0],
        field_energy=0.0, total_energy=0.0, traces=[],
    )
    baseline = SimpleNamespace(
        divB=np.zeros(2, dtype=np.complex128),
        totals=SimpleNamespace(
            number=[1.0, 2.0], kinetic=[3.0, 4.0],
            ion_internal=5.0, electron_perp=6.0, electron_parallel=7.0,
            magnetic=8.0, longitudinal=9.0, energy=42.0,
            topology=0.0, physical_flux=0.0,
        ),
        momentum=SimpleNamespace(
            value=[0.0, 0.0, 0.0], outward=[0.0, 0.0, 0.0],
            wall_reaction=[0.0, 0.0, 0.0], angular=0.0,
            outward_torque=0.0, wall_torque=0.0, angular_samples=3,
        ),
        step=0, time=0.0,
    )
    cumulative = SimpleNamespace(
        baseline=baseline, integrals=[ledger, ledger],
        boundary_integrals=boundary, accepted_intervals=1,
        last_trial=1, last_step=1, last_time=1e-9,
    )
    case_path = tmp_path / "case.toml"
    geometry_path = tmp_path / "geometry.npz"
    initial_path = tmp_path / "initial.npz"
    source_path = tmp_path / "source.json"
    case_path.write_text("case", encoding="utf-8")
    np.savez(geometry_path, x=np.asarray([1.0]))
    np.savez(initial_path, x=np.asarray([1.0]))
    source_path.write_text("{}", encoding="utf-8")
    input_hashes = {
        "case": io.sha256_file(case_path),
        "geometry": io.sha256_file(geometry_path),
        "initial": io.sha256_file(initial_path),
        "source": io.sha256_file(source_path),
    }
    parameters = SimpleNamespace(
        K=0, dt=1e-9, start_time=0.0, nsteps=2,
        integrator="midpoint", electron_inertia=1,
        species=(SimpleNamespace(id="ion"), SimpleNamespace(id="electron")),
    )
    case = SimpleNamespace(
        output=tmp_path / "run",
        case_path=case_path,
        geometry_path=geometry_path,
        initial_path=initial_path,
        source_path=source_path,
        name="checkpoint_test",
        backend="cpu",
        parameters=parameters,
        geometry=geometry,
        limits=io.Limits(),
        input_hashes=input_hashes,
        source={"provider_name": "test"},
        resolved={"case": "test"},
        config_hash="config",
        semantic_hash="semantic",
        geometry_hash="geometry",
        operator_hash="operators",
        compute=SimpleNamespace(
            synchronize=lambda: None,
            xp=np,
            array=lambda value, dtype=None: np.ascontiguousarray(
                value, dtype=dtype
            ),
        ),
    )
    context = SimpleNamespace(
        p=parameters, compute=case.compute,
        next_source=2, next_trial=2,
    )
    run = io.Run(case, require_project_snapshot=False)
    try:
        assert run.manifest["time_guard"]["minimum_samples_per_cycle"] == 32.0
        assert run.manifest["time_guard"]["max_amplitude_error"] == 0.01
        assert run.manifest["time_guard"]["max_phase_error_rad"] == 0.1
        archived_time = run.path() / "provenance/source/src/fomhd/time.py"
        assert archived_time.is_file()
        assert (
            run.path() / "provenance/source/src/fomhd/__main__.py"
        ).is_file()
        assert io.sha256_file(archived_time) == io.sha256_file(
            Path(__file__).resolve().parents[2] / "src/fomhd/time.py"
        )
        run.checkpoint(committed, cumulative, context)
        checkpoint = next(
            (run.path() / "checkpoints").glob("restart_*.npz")
        )
        restarted = io.read_restart(checkpoint, case)
        assert restarted.state.step == 1
        assert restarted.cumulative.accepted_intervals == 1
        assert restarted.cumulative.last_trial == 1
        failure = Failure("Domain", "nonfinite evidence")
        failure.partial_arrays = {
            "bad": np.asarray([np.nan, np.inf], dtype=np.complex128),
        }
        run.failure(failure, 1, "numerical_trial")
        partial = next((run.path() / "failure").glob("*.npz"))
        with np.load(partial, allow_pickle=False) as evidence:
            assert not np.isfinite(evidence["bad"]).all()
        with pytest.raises(ValueError, match="nonfinite"):
            io.read_npz(partial, io.Limits())
        source_path.write_text('{"changed":true}', encoding="utf-8")
        case.output = tmp_path / "changed-input-run"
        with pytest.raises(ValueError, match="captured input hash mismatch"):
            io.Run(case, require_project_snapshot=False)
        inspection = io.inspect(run.path())
        assert inspection["inspection"]["last_history_step"] is None
        assert inspection["inspection"]["interrupted_output"] is False
        with checkpoint.open("r+b") as handle:
            handle.seek(0)
            handle.write(b"XX")
        with pytest.raises(Exception, match="hash|malformed|NPZ|magic"):
            io.read_restart(checkpoint, case)
    finally:
        run.close()


def test_cumulative_json_requires_complete_zero_initial_ledger(runtime):
    io = runtime["io"]
    ledger_names = (
        "electric_node", "electric_edge", "pressure_kinetic",
        "pressure_internal", "magnetic", "wall", "convection",
        "perp_rate", "parallel_rate", "spectral_kinetic",
        "spectral_perp", "spectral_parallel",
    )
    ledger = SimpleNamespace(**{name: 0.0 for name in ledger_names})
    boundary = SimpleNamespace(
        number=[0.0, 0.0], kinetic=[0.0, 0.0],
        internal_perp=[0.0, 0.0], internal_parallel=[0.0, 0.0],
        field_energy=0.0, total_energy=0.0,
    )
    totals = SimpleNamespace(
        number=[0.0, 0.0], kinetic=[0.0, 0.0],
        ion_internal=0.0, electron_perp=0.0, electron_parallel=0.0,
        magnetic=0.0, longitudinal=0.0, energy=0.0,
        topology=0.0, physical_flux=0.0,
    )
    momentum = SimpleNamespace(
        value=[0.0, 0.0, 0.0], outward=[0.0, 0.0, 0.0],
        wall_reaction=[0.0, 0.0, 0.0], angular=0.0,
        outward_torque=0.0, wall_torque=0.0, angular_samples=3,
    )
    cumulative = SimpleNamespace(
        baseline=SimpleNamespace(
            totals=totals, momentum=momentum, step=0, time=0.0,
        ),
        integrals=[ledger, ledger], boundary_integrals=boundary,
        accepted_intervals=0, last_trial=0, last_step=0, last_time=0.0,
    )
    valid = io._cumulative_json(cumulative)
    parsed = io._cumulative_from_json(valid)
    assert parsed.accepted_intervals == 0
    assert len(parsed.integrals) == 2

    empty = json.loads(json.dumps(valid))
    empty["integrals"] = []
    with pytest.raises(ValueError, match="integrals"):
        io._cumulative_from_json(empty)

    nonzero_boundary = json.loads(json.dumps(valid))
    nonzero_boundary["boundary_integrals"]["field_energy"] = 123.0
    with pytest.raises(ValueError, match="zero"):
        io._cumulative_from_json(nonzero_boundary)

    missing = json.loads(json.dumps(valid))
    missing["boundary_integrals"].pop("total_energy")
    with pytest.raises(ValueError, match="boundary"):
        io._cumulative_from_json(missing)
