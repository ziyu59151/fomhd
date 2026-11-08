"""Behavioral tests for the bounded time-batched massless field producer."""

from __future__ import annotations

import copy
import importlib.util
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "warp"
MODULE_PATH = PROJECT_ROOT / "src/fomhd/field_batch.py"
for path in (SOURCE_ROOT, FIXTURE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fomhd.core import allocate_state, forward, inverse
from fomhd.massless import evaluate_stage
from fomhd.types import Failure
from fomhd.wall_cache import get_wall_cache
from test_massless_support import (
    context as closed_context,
    stage_tag,
    uniform_state,
)


def _load_module():
    assert MODULE_PATH.exists(), (
        "selected FieldBatch API is absent; expected scratch/field_batch.py"
    )
    spec = importlib.util.spec_from_file_location("field_batch_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_periodic_empty_wall_is_not_a_missing_reduction_domain():
    from massless_verification_fixture import HallCase, build_periodic_context, uniform_state as periodic_state
    case = HallCase(points=9)
    c = build_periodic_context(case)
    state = periodic_state(c,case)
    nc = c.op.Nc
    p = np.concatenate((state.fluid[0].n,state.fluid[0].mom,state.fluid[0].perp))[None,None]
    z = np.concatenate((state.fluid[1].perp,state.fluid[1].parallel,state.B))[None,None]
    result = _load_module().FieldBatch(c).rhs(p,z)
    assert result.rhs.shape == z.shape
    np.testing.assert_allclose(result.rhs,0.,rtol=0,atol=1e-6)


@pytest.mark.parametrize("name,value",[("electron_inertia",1),("rank_rtol",1e-10)])
def test_semantic_parameter_changes_reject_cached_producer(batch_fixture,name,value):
    producer=_load_module().FieldBatch(batch_fixture.context)
    setattr(batch_fixture.context.p,name,value)
    with pytest.raises(Failure):
        producer.rhs(batch_fixture.p,batch_fixture.z)


def test_sparse_wall_incidence_gram_equals_dense_definition(batch_fixture):
    producer=_load_module().FieldBatch(batch_fixture.context)
    assert hasattr(producer,"weighted_normal"),"bounded wall Gram assembly is missing"
    c=batch_fixture.context
    rng=np.random.default_rng(8)
    weights=.8+rng.random((2,c.N,3*c.op.Nc))
    actual=producer.weighted_normal(weights)
    expected=np.empty_like(actual)
    for b in range(2):
        for n in range(c.N):
            expected[b,n]=(c.op.T*weights[b,n])@c.op.T.conj().T
    np.testing.assert_allclose(actual,expected,rtol=1e-14,atol=1e-14)


def test_field_batch_memory_budget_precedes_numeric_allocation(batch_fixture):
    module=_load_module()
    assert hasattr(module.FieldBatch,"workspace_estimate"),"batch memory preflight is missing"
    producer=module.FieldBatch(batch_fixture.context,workspace_limit_bytes=1)
    with pytest.raises(Failure) as caught:
        producer.rhs(batch_fixture.p,batch_fixture.z)
    assert caught.value.code=="MemoryBudget"


def _periodic_real(context, values):
    return forward(np.asarray(values, dtype=np.float64).reshape(-1), 1, context)


def _scalar_mode(context, amplitude, phase):
    nc = context.op.Nc
    coordinate = 2.0 * math.pi * np.arange(context.N) / float(context.N)
    scalar = np.tile(
        amplitude * np.cos(coordinate + phase)[:, None],
        (1, nc),
    )
    return forward(scalar.reshape(-1), nc, context)


def _state_and_arrays(context, index):
    state = uniform_state(context)
    xp = context.compute.xp
    nc = context.op.Nc
    modes = 2 * context.p.K + 1
    scale = 1.0 + 0.04 * index
    state.B = xp.asarray(state.B * scale, dtype=xp.complex128)

    state.fluid[0].n += _scalar_mode(
        context, 0.015 * context.p.ref.ni, 0.17 + 0.2 * index
    )
    state.fluid[0].perp += _scalar_mode(
        context, 0.01 * context.p.ref.pressure, -0.23 + 0.1 * index
    )
    state.fluid[1].perp += _scalar_mode(
        context, 0.04 * context.p.ref.pressure, 0.31 + 0.15 * index
    )
    state.fluid[1].parallel += _scalar_mode(
        context, 0.03 * context.p.ref.pressure, -0.41 + 0.08 * index
    )

    density = inverse(state.fluid[0].n, nc, context, True).reshape(
        context.N, nc
    )
    coordinate = 2.0 * math.pi * np.arange(context.N) / float(context.N)
    velocity = np.zeros((context.N, 3, nc), dtype=np.complex128)
    velocity[:, 1, :] = (
        0.012 * (1.0 + 0.2 * index) * np.cos(coordinate + 0.1 * index)[:, None]
    )
    raw_momentum = (
        context.p.species[0].mass
        * density[:, None, :]
        * velocity
    ).reshape(-1)
    from fomhd.massless import _constraint_project

    momentum_nodes = _constraint_project(
        raw_momentum,
        density.reshape(-1),
        context,
    )
    state.fluid[0].mom = forward(momentum_nodes, 3 * nc, context)
    state.step = 0
    state.generation = 1
    state.time = context.p.start_time

    p_item = np.concatenate(
        (
            np.asarray(state.fluid[0].n).reshape(modes, nc),
            np.asarray(state.fluid[0].mom).reshape(modes, 3, nc)[:, 0, :],
            np.asarray(state.fluid[0].mom).reshape(modes, 3, nc)[:, 1, :],
            np.asarray(state.fluid[0].mom).reshape(modes, 3, nc)[:, 2, :],
            np.asarray(state.fluid[0].perp).reshape(modes, nc),
        ),
        axis=1,
    )
    z_item = np.concatenate(
        (
            np.asarray(state.fluid[1].perp).reshape(modes, nc),
            np.asarray(state.fluid[1].parallel).reshape(modes, nc),
            np.asarray(state.B).reshape(modes, context.op.Nb),
        ),
        axis=1,
    )
    return state, p_item, z_item


def _state_from_arrays(context, p_item, z_item):
    state = allocate_state(context)
    nc = context.op.Nc
    modes = 2 * context.p.K + 1
    p_view = np.asarray(p_item).reshape(modes, 5, nc)
    z_view = np.asarray(z_item).reshape(modes, 2 * nc + context.op.Nb)
    state.fluid[0].n = context.compute.array(p_view[:, 0, :].reshape(-1))
    state.fluid[0].mom = context.compute.array(
        p_view[:, 1:4, :].reshape(modes, 3 * nc)
    )
    state.fluid[0].perp = context.compute.array(p_view[:, 4, :].reshape(-1))
    state.fluid[1].perp = context.compute.array(
        z_view[:, :nc].reshape(-1)
    )
    state.fluid[1].parallel = context.compute.array(
        z_view[:, nc : 2 * nc].reshape(-1)
    )
    state.B = context.compute.array(
        z_view[:, 2 * nc :].reshape(-1)
    )
    state.step = 0
    state.generation = 1
    state.time = context.p.start_time
    return state


def _pack_batch(context, count=3):
    states = []
    p_values = []
    z_values = []
    for index in range(count):
        state, p_item, z_item = _state_and_arrays(context, index)
        states.append(state)
        p_values.append(p_item)
        z_values.append(z_item)
    return states, np.stack(p_values), np.stack(z_values)


@pytest.fixture
def batch_fixture():
    context = closed_context(K=1, hard_diagnostics=False)
    get_wall_cache(context)
    states, p_values, z_values = _pack_batch(context)
    return SimpleNamespace(
        context=context,
        states=states,
        p=p_values,
        z=z_values,
    )


def _assert_per_item_close(actual, expected, *, rtol=2.0e-10):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.ndim >= 2
    for index in range(actual.shape[0]):
        reference = float(np.linalg.norm(expected[index]))
        difference = float(np.linalg.norm(actual[index] - expected[index]))
        limit = rtol * max(reference, 1.0e-300)
        assert difference <= limit, (
            f"batch item {index}: difference={difference:.6e}, "
            f"reference={reference:.6e}, limit={limit:.6e}"
        )


def test_field_batch_module_is_available():
    """Catch removal or accidental relocation of the selected API module."""
    module = _load_module()
    assert hasattr(module, "FieldBatch")


def test_batch_field_rhs_matches_scalar_massless_stage(batch_fixture):
    """Catch any field, Hall-current, Ohm, wall, pressure, or B equation drift."""
    module = _load_module()
    context = batch_fixture.context
    producer = module.FieldBatch(context)
    actual = producer.rhs(batch_fixture.p, batch_fixture.z)

    expected_rhs = []
    expected_electric = []
    expected_current = []
    expected_velocity = []
    expected_density = []
    for state in batch_fixture.states:
        stage = evaluate_stage(state, stage_tag(state), context)
        expected_rhs.append(
            np.concatenate(
                (
                    np.asarray(stage.rhs.fluid[1].perp),
                    np.asarray(stage.rhs.fluid[1].parallel),
                    np.asarray(stage.rhs.B),
                )
            )
        )
        expected_electric.append(np.asarray(stage.cache.E))
        expected_current.append(np.asarray(stage.current))
        expected_velocity.append(np.asarray(stage.fluid[1].u))
        expected_density.append(np.asarray(stage.fluid[1].n))

    assert actual.rhs.shape == batch_fixture.z.shape
    nc, modes = context.op.Nc, 2*context.p.K+1
    packed_rhs = []
    for value in expected_rhs:
        packed_rhs.append(np.concatenate((value[:modes*nc].reshape(modes,nc), value[modes*nc:2*modes*nc].reshape(modes,nc), value[2*modes*nc:].reshape(modes,context.op.Nb)),axis=1))
    expected = np.stack(packed_rhs)
    _assert_per_item_close(actual.rhs[:, :, :nc], expected[:, :, :nc])
    _assert_per_item_close(actual.rhs[:, :, nc:2*nc], expected[:, :, nc:2*nc])
    _assert_per_item_close(actual.rhs[:, :, 2*nc:], expected[:, :, 2*nc:])
    _assert_per_item_close(actual.electric, np.stack(expected_electric).reshape(actual.electric.shape))
    _assert_per_item_close(actual.current, np.stack(expected_current))
    _assert_per_item_close(actual.velocity, np.stack(expected_velocity))
    _assert_per_item_close(actual.density, np.stack(expected_density))
    assert np.all(np.isfinite(np.asarray(actual.rhs)))


def test_batch_items_match_individual_calls_and_order_permutation(batch_fixture):
    """Catch a reduction over batch indices or an index-dependent operator."""
    module = _load_module()
    context = batch_fixture.context
    producer = module.FieldBatch(context)
    batched = producer.rhs(batch_fixture.p, batch_fixture.z)
    for index in range(batch_fixture.p.shape[0]):
        individual = producer.rhs(
            batch_fixture.p[index : index + 1],
            batch_fixture.z[index : index + 1],
        )
        for name in ("rhs", "electric", "current", "velocity", "density"):
            _assert_per_item_close(
                getattr(individual, name),
                getattr(batched, name)[index : index + 1],
                rtol=2.0e-11,
            )

    order = np.array([2, 0, 1])
    permuted = producer.rhs(batch_fixture.p[order], batch_fixture.z[order])
    for target, source in enumerate(order):
        for name in ("rhs", "electric", "current", "velocity", "density"):
            _assert_per_item_close(
                getattr(permuted, name)[target : target + 1],
                getattr(batched, name)[source : source + 1],
                rtol=2.0e-11,
            )


def test_inputs_are_read_only_and_outputs_do_not_alias(batch_fixture):
    """Catch accidental in-place normalization or returned input views."""
    module = _load_module()
    p_before = batch_fixture.p.copy()
    z_before = batch_fixture.z.copy()
    result = module.FieldBatch(batch_fixture.context).rhs(
        batch_fixture.p, batch_fixture.z
    )
    np.testing.assert_array_equal(batch_fixture.p, p_before)
    np.testing.assert_array_equal(batch_fixture.z, z_before)
    for value in (
        result.rhs,
        result.electric,
        result.current,
        result.velocity,
        result.density,
    ):
        assert not np.shares_memory(np.asarray(value), batch_fixture.p)
        assert not np.shares_memory(np.asarray(value), batch_fixture.z)


def test_repeated_calls_leave_context_and_ledger_unchanged(batch_fixture):
    """Catch hidden caching, source-id, wall-counter, or ledger mutation."""
    module = _load_module()
    context = batch_fixture.context
    cache = get_wall_cache(context)
    context.cumulative = SimpleNamespace(
        accepted_intervals=4,
        last_step=3,
        last_time=0.5,
        payload={"sentinel": (1, 2, 3)},
    )
    before = {
        "generation": context.next_generation,
        "trial": context.next_trial,
        "source": context.next_source,
        "cache_builds": getattr(context, "_wall_system_cache_builds", 0),
        "cache_solves": cache.solves,
        "cache_evaluations": cache.evaluations,
        "cumulative": copy.deepcopy(context.cumulative),
    }
    producer = module.FieldBatch(context)
    producer.rhs(batch_fixture.p, batch_fixture.z)
    producer.rhs(batch_fixture.p, batch_fixture.z)
    assert context.next_generation == before["generation"]
    assert context.next_trial == before["trial"]
    assert context.next_source == before["source"]
    assert getattr(context, "_wall_system_cache_builds", 0) == before[
        "cache_builds"
    ]
    assert cache.solves == before["cache_solves"]
    assert cache.evaluations == before["cache_evaluations"]
    assert context.cumulative == before["cumulative"]


def test_invalid_imaginary_samples_are_rejected_per_batch_item(batch_fixture):
    """Catch a global norm hiding a non-real item behind a much larger item."""
    module = _load_module()
    context = batch_fixture.context
    p_values = batch_fixture.p.copy()
    z_values = batch_fixture.z.copy()
    p_values[0, context.p.K, 0] += 1.0j
    p_values[1] *= 1.0e12
    z_values[1] *= 1.0e12
    with pytest.raises(Failure, match="real|imaginary|Imaginary"):
        module.FieldBatch(context).rhs(p_values, z_values)


def test_nonpositive_electron_pressure_is_rejected(batch_fixture):
    """Catch accidental continuation with a nonphysical CGL pressure."""
    module = _load_module()
    z_values = batch_fixture.z.copy()
    z_values[1, batch_fixture.context.p.K, 0] = -abs(
        z_values[1, batch_fixture.context.p.K, 0]
    )
    with pytest.raises(Failure, match="pressure|Domain"):
        module.FieldBatch(batch_fixture.context).rhs(batch_fixture.p, z_values)


@pytest.mark.parametrize(
    ("p_mutator", "z_mutator", "message"),
    (
    (lambda p, z, c: p.real.astype(np.float64), lambda p, z, c: z, "dtype"),
        (
            lambda p, z, c: p[:, :-1],
            lambda p, z, c: z,
            "shape|layout",
        ),
        (
            lambda p, z, c: p,
            lambda p, z, c: np.zeros_like(z),
            "magnetic|Magnetization|B",
        ),
    ),
)
def test_invalid_dtype_layout_and_zero_field_are_rejected(
    batch_fixture, p_mutator, z_mutator, message
):
    """Catch silent coercion, layout misinterpretation, or zero-B dividing."""
    module = _load_module()
    p_values = p_mutator(batch_fixture.p, batch_fixture.z, batch_fixture.context)
    z_values = z_mutator(batch_fixture.p, batch_fixture.z, batch_fixture.context)
    with pytest.raises(Failure, match=message):
        module.FieldBatch(batch_fixture.context).rhs(p_values, z_values)


def test_operator_version_identity_is_enforced_before_evaluation(batch_fixture):
    """Catch reuse of factors after the fixed operator identity changes."""
    module = _load_module()
    context = batch_fixture.context
    context.operator_hash = "snapshot-a"
    producer = module.FieldBatch(context)
    context.operator_hash = "snapshot-b"
    with pytest.raises(Failure, match="identity|version|snapshot"):
        producer.rhs(batch_fixture.p, batch_fixture.z)


def test_fourier_azimuthal_derivative_manufactured_case(batch_fixture):
    """Catch a wrong signed-frequency convention in the cylindrical gradient."""
    module = _load_module()
    context = batch_fixture.context
    nc = context.op.Nc
    coordinate = 2.0 * math.pi * np.arange(context.N) / float(context.N)
    values = (
        0.003 * np.cos(coordinate)[:, None]
        + 0.004 * np.sin(coordinate)[:, None]
    )
    values = np.broadcast_to(values, (context.N, nc)).copy()
    batch = np.stack((values, 2.0 * values))
    full = np.fft.fft(batch, axis=1)
    signed = np.arange(context.N, dtype=np.float64)
    signed = np.where(signed <= (context.N - 1) // 2, signed, signed - context.N)
    expected = np.fft.ifft(
        full * (1j * signed)[None, :, None] / float(context.N),
        axis=1,
    ) * float(context.N)
    actual = np.asarray(module._derivative(batch, context))
    _assert_per_item_close(actual, expected, rtol=2.0e-12)


@pytest.mark.skipif(os.environ.get("FOMHD_TEST_GPU") != "1", reason="explicit owned Slurm GPU run required")
@pytest.mark.parametrize("unit,points", [("E2",9),("E3",4),("E3",6)])
def test_cuda_full_batch_matches_cpu_field_producer(unit,points):
    assert os.environ.get("SLURM_JOB_ID")
    from tools.warp_t2_field import periodic_inputs,closed_inputs
    from fomhd.operators import build_operators
    from fomhd.backend import Backend
    build=periodic_inputs if unit=="E2" else closed_inputs
    c,p,z,_=build(points,.02,.1,"cpu")
    p=np.repeat(p[None],3,axis=0)
    z=np.repeat(z[None],3,axis=0)
    cpu=_load_module().FieldBatch(c).rhs(p,z)
    gpu=Backend("cuda:0")
    c.compute=gpu
    c.op=build_operators(c.geometry,c.p.K,backend=gpu) if unit=="E3" else c.op
    if unit=="E2":
        from massless_verification_fixture import HallCase,build_periodic_context
        c=build_periodic_context(HallCase(points=9),device="cuda:0")
    actual=_load_module().FieldBatch(c).rhs(gpu.array(p),gpu.array(z))
    nc=c.op.Nc
    for low,high in ((0,nc),(nc,2*nc),(2*nc,None)):
        np.testing.assert_allclose(gpu.to_host(actual.rhs[:,:,low:high]),cpu.rhs[:,:,low:high],rtol=2e-9,atol=1e-10)
    np.testing.assert_allclose(gpu.to_host(actual.electric),cpu.electric,rtol=2e-9,atol=1e-10)
    assert actual.rhs.dtype==gpu.xp.complex128


def test_exact_diagonal_normal_specialization_preserves_physics_and_rank_checks():
    from tools.warp_t2_field import closed_inputs
    c,p,z,_=closed_inputs(6,.02,.1,'cpu')
    module=_load_module()
    p=np.broadcast_to(p,(3,)+p.shape).copy()
    z=np.broadcast_to(z,(3,)+z.shape).copy()
    try:
        fast=module.FieldBatch(c,structured_normal=True)
        dense=module.FieldBatch(c,structured_normal=False)
    except TypeError:
        pytest.fail('selected exact diagonal normal representation is missing')
    assert fast.diagonal_normal
    a,b=fast.rhs(p,z),dense.rhs(p,z)
    for key in ['rhs','electric','velocity','current']:
        np.testing.assert_allclose(getattr(a,key),getattr(b,key),rtol=1e-9,atol=1e-10)
    assert fast.work['normal_svd_systems']==0
    assert fast.work['normal_diagonal_systems']>0
    assert dense.work['normal_svd_systems']>0
    assert fast.workspace_estimate(32)['total']<dense.workspace_estimate(32)['total']
    density=np.full((3,c.N,c.op.Nc),c.p.ref.ne,dtype=np.complex128)
    density[:,:,0]*=1e20
    with pytest.raises(Failure,match='deficient'):
        fast._constraint_project(np.zeros((3,c.N,3*c.op.Nc),dtype=np.complex128),density)
