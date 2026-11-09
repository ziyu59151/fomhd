"""Behavioral tests for the massless A1-WR physics adapter."""

from __future__ import annotations

import copy
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from fomhd import massless, massless_field, stage
from fomhd.time_multirate_wr import WindowIdentity
from fomhd.types import Failure, StageTag
from wr_benchmark_fixture import closed_wall_inputs, periodic_inputs


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "fomhd"
    / "multirate_wr_physics.py"
)


@lru_cache(maxsize=1)
def _api():
    if not MODULE_PATH.is_file():
        pytest.fail(f"feature module is absent: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location(
        "fomhd.multirate_wr_physics", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pack_state(*args, **kwargs):
    return _api().pack_state(*args, **kwargs)


def pack_stage_rhs(*args, **kwargs):
    return _api().pack_stage_rhs(*args, **kwargs)


def physical_scales(*args, **kwargs):
    return _api().physical_scales(*args, **kwargs)


def state_from_packed(*args, **kwargs):
    return _api().state_from_packed(*args, **kwargs)


def new_adapter(*args, **kwargs):
    return _api().MasslessWRPhysicsAdapter(*args, **kwargs)


def _field_identity(micro_step=0, rk_stage=0, iteration=0):
    fraction = (0.0, 0.5, 0.5, 1.0)[rk_stage]
    return WindowIdentity(
        iteration=iteration,
        micro_step=micro_step,
        rk_stage=rk_stage,
        coarse_node=None,
        role="field_rk4",
        theta=(micro_step + fraction) / 2.0,
    )


def _coarse_identity(node=0):
    return WindowIdentity(
        iteration=0,
        micro_step=0,
        rk_stage=None,
        coarse_node=node,
        role="transport_coarse",
        theta=(0.0, 0.5, 1.0)[node],
    )


def _array_snapshot(state):
    return {
        "B": np.asarray(state.B).copy(),
        "ion_n": np.asarray(state.fluid[0].n).copy(),
        "ion_mom": np.asarray(state.fluid[0].mom).copy(),
        "ion_p": np.asarray(state.fluid[0].perp).copy(),
        "electron_n": np.asarray(state.fluid[1].n).copy(),
        "electron_perp": np.asarray(state.fluid[1].perp).copy(),
        "electron_parallel": np.asarray(state.fluid[1].parallel).copy(),
    }


def _full_rhs(context, p, z, *, time, generation, stage_name):
    tag = StageTag(
        step=0,
        generation=generation,
        trial=9000 + generation,
        spectral_context=context.spectral_id,
        time=time,
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
    full = stage.evaluate_stage(state, tag, context, False)
    _, z_rhs = pack_stage_rhs(full, species=1, context=context)
    return pack_stage_rhs(full, species=0, context=context), z_rhs


def _independent_full_rk4(context, p0, z0, h, micro_steps):
    p = np.asarray(p0).copy()
    z = np.asarray(z0).copy()
    generation = 50000
    for micro in range(micro_steps):
        kp, kz = [], []
        for rk, fraction in enumerate((0.0, 0.5, 0.5, 1.0)):
            if rk == 0:
                sp, sz = p, z
            elif rk == 1:
                sp, sz = p + 0.5 * h * kp[0], z + 0.5 * h * kz[0]
            elif rk == 2:
                sp, sz = p + 0.5 * h * kp[1], z + 0.5 * h * kz[1]
            else:
                sp, sz = p + h * kp[2], z + h * kz[2]
            rp, rz = _full_rhs(
                context,
                sp,
                sz,
                time=(micro + fraction) * h,
                generation=generation,
                stage_name="reference_rk4",
            )
            generation += 1
            kp.append(rp)
            kz.append(rz)
        p = p + h * (kp[0] + 2 * kp[1] + 2 * kp[2] + kp[3]) / 6.0
        z = z + h * (kz[0] + 2 * kz[1] + 2 * kz[2] + kz[3]) / 6.0
    return p, z


def _relative(left, right, scale):
    return float(
        np.max(np.abs(np.asarray(left) - np.asarray(right)) / np.asarray(scale))
    )


def test_packed_state_round_trip_preserves_independent_fields():
    context, state, _ = periodic_inputs(
        9, 1.0e-4, 0.1, seed_mode=1, sentinel=1.0e-4
    )
    p, z = pack_state(context, state)
    rebuilt = state_from_packed(
        context,
        p,
        z,
        step=state.step,
        generation=state.generation,
        time=state.time,
    )

    np.testing.assert_array_equal(rebuilt.fluid[0].n, state.fluid[0].n)
    np.testing.assert_array_equal(rebuilt.fluid[0].mom, state.fluid[0].mom)
    np.testing.assert_array_equal(rebuilt.fluid[0].perp, state.fluid[0].perp)
    np.testing.assert_array_equal(rebuilt.fluid[1].perp, state.fluid[1].perp)
    np.testing.assert_array_equal(
        rebuilt.fluid[1].parallel, state.fluid[1].parallel
    )
    np.testing.assert_array_equal(rebuilt.B, state.B)
    np.testing.assert_allclose(
        rebuilt.fluid[1].n,
        context.p.species[0].charge
        / (-context.p.species[1].charge)
        * rebuilt.fluid[0].n,
        rtol=0.0,
        atol=0.0,
    )
    assert np.all(rebuilt.fluid[1].mom == 0.0)
    assert p.dtype == z.dtype == np.complex128
    p_scale, z_scale = physical_scales(context)
    assert p_scale.shape == p.shape
    assert z_scale.shape == z.shape
    assert np.all(p_scale > 0.0)
    assert np.all(z_scale > 0.0)


def test_prior_33_point_controls_include_actual_high_mode16_seed():
    context, state, metadata = periodic_inputs(
        33,
        0.02,
        0.1,
        seed_mode=1,
        sentinel=1.0e-4,
    )
    p, z = pack_state(context, state)
    nc = context.op.Nc
    b_r = np.asarray(z[0, 2 * nc : 2 * nc + nc])
    mode16 = np.fft.fft(b_r)[16] / nc

    assert metadata["target_mode"] == 1
    assert metadata["sentinel_mode"] == 16
    assert metadata["sentinel_retained"] is True
    assert float(np.real(mode16)) == pytest.approx(
        0.5 * 0.02 * 1.0e-4 * context.p.ref.magnetic,
        rel=1.0e-12,
        abs=2.0e-20,
    )
    assert float(np.imag(mode16)) == pytest.approx(
        0.0, abs=1.0e-18
    )
    assert float(np.real(np.min(np.asarray(state.fluid[0].n)))) > 0.0
    assert p.shape == (1, 5 * nc)
    assert z.shape == (1, 2 * nc + context.op.Nb)


def test_middle_field_callback_has_no_hidden_ion_transport(monkeypatch):
    context, state, _ = periodic_inputs(
        9, 1.0e-4, 0.1, seed_mode=1, sentinel=1.0e-4
    )
    p, z = pack_state(context, state)
    calls = {"transport": 0}

    def transport_spy(*args, **kwargs):
        calls["transport"] += 1
        raise AssertionError("fine field callback evaluated ion transport")

    adapter_module = _api()

    def forbidden_reconstruction(*args, **kwargs):
        raise AssertionError("exchange diagnostic reconstructed a field producer")

    monkeypatch.setattr(massless, "build_non_electric", transport_spy)
    monkeypatch.setattr(
        adapter_module, "project_wall", forbidden_reconstruction
    )
    adapter = new_adapter(context, state)
    G, C, record = adapter.field(0.25, p, z, _field_identity(rk_stage=1))

    assert calls["transport"] == 0
    assert G.shape == z.shape
    assert C.shape == p.shape
    assert record.frame.kind == massless_field.FIELD_ONLY_KIND
    assert record.frame.complete_stage is False
    assert record.frame.alias_frame is not None
    assert record.frame.work.field_evaluations == 2
    assert record.frame.work.alias_field_evaluations == 1
    assert record.frame.ion_em_terms is not None
    assert record.electric_exchange_error is None
    assert record.electric_exchange_W is not None
    xp = context.compute.xp
    nc = context.op.Nc
    velocity = xp.asarray(record.frame.ion.u).reshape(context.N, 3, nc)
    weights = xp.real(xp.asarray(context.op.M0)).astype(xp.float64)
    expected_exchange = xp.sum(
        xp.conj(velocity)
        * record.frame.ion_em_terms.electric.reshape(context.N, 3, nc)
        * weights[None, None, :]
    ) * (2.0 * np.pi / context.N)
    assert record.electric_exchange_W == pytest.approx(
        complex(expected_exchange.item()), rel=2.0e-15, abs=2.0e-15
    )
    assert record.alias_field_evaluations == 1
    assert adapter.work.field_callback_calls == 1
    assert adapter.work.field_evaluations == 2
    assert adapter.work.fine_full_stage_calls == 0
    assert adapter.work.wall_projections == adapter.work.feedback_calls


def test_coarse_split_matches_one_complete_stage_rhs():
    context, state, _ = periodic_inputs(
        9, 1.0e-4, 0.1, seed_mode=1, sentinel=1.0e-4
    )
    p, z = pack_state(context, state)
    adapter = new_adapter(context, state)
    A, record = adapter.transport(0.5, p, z, _coarse_identity())
    nc = context.op.Nc

    assert adapter.work.full_stage_calls == 1
    assert adapter.work.coarse_transport_calls == 1
    assert adapter.work.coarse_transport_wall_projections == 1
    assert adapter.work.coarse_grid_wall_projections == 1
    assert np.all(A[:, :nc] == 0.0)
    assert np.all(record.C[:, 4 * nc : 5 * nc] == 0.0)
    np.testing.assert_allclose(
        A + record.C,
        record.F,
        rtol=2.0e-11,
        atol=2.0e-11 * max(1.0, float(np.max(np.abs(record.F)))),
    )
    assert record.split_relative_error <= 2.0e-11
    assert record.full.source_id == record.frame.source_id
    assert record.transport_projection is not None
    assert record.transport_projection.value.shape == (context.N * 3 * nc,)
    assert record.transport_projection.reaction is not None
    assert record.transport_projection.multiplier is not None
    assert record.ion_frame is record.frame.fluid[0]


def test_cpu_window_matches_independent_full_rhs_rk4_and_keeps_s_nodes():
    context, state, _ = periodic_inputs(
        9, 1.0e-4, 0.1, seed_mode=1, sentinel=1.0e-4
    )
    p0, z0 = pack_state(context, state)
    p0_before, z0_before = p0.copy(), z0.copy()
    state_before = _array_snapshot(state)
    ledger = {"accepted_macros": 7, "sentinel": np.array([3.0])}
    ledger_before = copy.deepcopy(ledger)
    h = 1.0e-12
    adapter = new_adapter(context, state)
    result = adapter.solve_window(
        p0,
        z0,
        H=h,
        M=1,
        s=3,
        rtol=1.0e-8,
        max_iterations=10,
        ledger=ledger,
    )
    reference_p, reference_z = _independent_full_rk4(
        context, p0, z0, h, micro_steps=1
    )
    p_scale, z_scale = physical_scales(context)

    assert result.window.P.shape[0] == 3
    assert result.window.P_endpoint.shape == p0.shape
    np.testing.assert_array_equal(result.window.P_endpoint, result.window.P[-1])
    assert result.endpoint.production_accepted is False
    assert result.endpoint.full.checks
    assert result.work.full_stage_calls >= 5
    assert result.work.endpoint_reconstruction_calls == 1
    assert len(result.coarse_records) == result.window.s
    assert len(result.fine_records) == 4 * result.window.M
    assert len(result.all_coarse_records) > len(result.coarse_records)
    assert len(result.all_fine_records) > len(result.fine_records)
    assert all(
        record.identity.iteration == result.window.accepted_node_version
        for record in result.coarse_records + result.fine_records
    )
    middle = [
        record
        for record in result.fine_records
        if record.identity.rk_stage in (1, 2)
    ]
    assert len(middle) == 2 * result.window.M
    assert len({record.tag.generation for record in middle}) == len(middle)
    assert result.source["producer_delta"] == (
        result.work.full_stage_calls + result.work.field_evaluations
    )
    assert result.endpoint.trial_endpoint is True
    assert result.endpoint.interval_transaction_required is True
    assert _relative(result.window.P_endpoint, reference_p, p_scale) < 2.0e-5
    assert _relative(result.window.Z_endpoint, reference_z, z_scale) < 2.0e-5
    np.testing.assert_array_equal(p0, p0_before)
    np.testing.assert_array_equal(z0, z0_before)
    np.testing.assert_array_equal(
        np.asarray(state.fluid[0].n), state_before["ion_n"]
    )
    assert ledger == ledger_before
    assert result.caller_unchanged is True
    assert result.endpoint.totals is not None


def test_coarse_callback_count_is_unchanged_when_m_doubles():
    observations = []
    for micro_steps in (1, 2):
        context, state, _ = periodic_inputs(
            9, 0.0, 0.0, seed_mode=1, sentinel=0.0
        )
        p0, z0 = pack_state(context, state)
        adapter = new_adapter(context, state)
        result = adapter.solve_window(
            p0,
            z0,
            H=2.0e-12,
            M=micro_steps,
            s=5,
            rtol=1.0e-9,
            max_iterations=4,
        )
        observations.append(
            (
                result.window.sweeps,
                result.work.coarse_transport_calls,
                result.work.field_callback_calls,
                result.window.P.shape[0],
            )
        )

    assert observations[0][0] == observations[1][0]
    assert observations[0][1] == observations[1][1]
    assert observations[0][3] == observations[1][3] == 5
    assert observations[1][2] == 1 + 2 * (observations[0][2] - 1)


def test_identities_cache_and_middle_stage_states_are_distinct():
    context, state, _ = periodic_inputs(
        9, 1.0e-4, 0.1, seed_mode=1, sentinel=1.0e-4
    )
    p0, z0 = pack_state(context, state)
    adapter = new_adapter(context, state)
    result = adapter.solve_window(
        p0,
        z0,
        H=2.0e-12,
        M=2,
        s=3,
        rtol=1.0e-8,
        max_iterations=6,
    )

    fields = adapter.fine_records
    assert fields
    assert all(record.tag.time == state.time for record in fields)
    generations = [record.tag.generation for record in fields]
    assert len(generations) == len(set(generations))
    cache_ids = {
        id(record.frame.cache.wall_correction.cache) for record in fields
    }
    assert len(cache_ids) == 1
    by_micro = {}
    for record in fields:
        by_micro.setdefault((record.tag.iteration, record.tag.micro_step), {})[
            record.tag.micro_step
        ] = record
    for (_, micro), records in by_micro.items():
        actual = [
            record
            for record in fields
            if record.tag.micro_step == micro
            and record.identity.rk_stage in (1, 2)
        ]
        if len(actual) >= 4:
            pairs = {}
            for record in actual:
                pairs.setdefault(record.tag.iteration, {})[
                    record.identity.rk_stage
                ] = record
            for pair in pairs.values():
                if 1 in pair and 2 in pair:
                    assert pair[1].tag.generation != pair[2].tag.generation
                    assert pair[1].tag.time == pair[2].tag.time
                    assert not np.array_equal(
                        pair[1].input_Z, pair[2].input_Z
                    )
    assert result.identity_evidence["accepted_left_clock_exact"] is True


def test_failure_preserves_evidence_and_leaves_inputs_and_ledger_untouched(
    monkeypatch,
):
    context, state, _ = periodic_inputs(
        9, 1.0e-4, 0.1, seed_mode=1, sentinel=1.0e-4
    )
    p0, z0 = pack_state(context, state)
    p_before, z_before = p0.copy(), z0.copy()
    state_before = _array_snapshot(state)
    ledger = {"accepted_macros": 11}
    adapter = new_adapter(context, state)
    original = massless_field.evaluate

    def injected(value, tag, owner, alias=False):
        if tag.stage.endswith("field_rk4_r1"):
            failure = Failure("InjectedField", "injected midpoint failure")
            failure.stage = tag
            failure.partial_arrays["injected"] = np.asarray([1.0])
            raise failure
        return original(value, tag, owner, alias)

    monkeypatch.setattr(massless_field, "evaluate", injected)
    with pytest.raises(Failure) as caught:
        adapter.solve_window(
            p0,
            z0,
            H=2.0e-12,
            M=2,
            s=3,
            rtol=1.0e-8,
            max_iterations=2,
            ledger=ledger,
        )

    failure = caught.value
    assert failure.code == "InjectedField"
    assert "injected" in failure.partial_arrays
    assert failure.multirate_partial_arrays
    assert failure.adapter_work["field_callback_calls"] >= 1
    assert ledger == {"accepted_macros": 11}
    np.testing.assert_array_equal(p0, p_before)
    np.testing.assert_array_equal(z0, z_before)
    assert np.array_equal(np.asarray(state.fluid[0].mom), state_before["ion_mom"])


def test_real_closed_wall_field_and_split_smoke():
    context, state, _ = closed_wall_inputs()
    p, z = pack_state(context, state)
    adapter = new_adapter(context, state)
    G, C, field_record = adapter.field(
        0.0, p, z, _field_identity(rk_stage=0)
    )
    assert G.shape == z.shape
    assert C.shape == p.shape
    assert field_record.frame.complete_stage is False

    A, coarse_record = adapter.transport(0.0, p, z, _coarse_identity())
    nc = context.op.Nc
    assert np.all(A[:, :nc] == 0.0)
    assert np.all(coarse_record.C[:, 4 * nc : 5 * nc] == 0.0)
    np.testing.assert_allclose(
        A + coarse_record.C,
        coarse_record.F,
        rtol=2.0e-10,
        atol=2.0e-10 * max(1.0, float(np.max(np.abs(coarse_record.F)))),
    )
    assert adapter.work.full_stage_calls == 1
    assert adapter.work.fine_full_stage_calls == 0
