"""Behavioral contracts for the field-only massless stage."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from fomhd import massless
from fomhd.core import forward, inverse, sample_apply
from fomhd.types import Failure
from fomhd.wall import project_wall

from massless_verification_fixture import (
    HallCase,
    build_periodic_context,
    initial_hall_eigenmode,
    stage_tag as periodic_stage_tag,
)
from test_massless_support import (
    compatible_magnetic_field,
    context as closed_context,
    stage_tag as closed_stage_tag,
    uniform_state,
    wall_incompatible_current_field,
)


def _periodic_nonuniform_fixture():
    case = HallCase(kdi=0.8, points=9, n0=1.0e16, B0=0.1, pressure=0.4)
    context = build_periodic_context(case)
    state, _, _ = initial_hall_eigenmode(
        case,
        relative_amplitude=2.0e-6,
        context=context,
    )
    nc = context.op.Nc
    axial = np.linspace(-0.08, 0.08, nc)
    state.fluid[0].n *= context.compute.array(1.0 + axial)
    state.fluid[0].perp *= context.compute.array(1.0 + 0.06 * axial)
    state.fluid[1].perp *= context.compute.array(1.0 - 0.04 * axial)
    state.fluid[1].parallel *= context.compute.array(1.0 + 0.03 * axial)
    return context, state


def _closed_nonuniform_fixture():
    context = closed_context(
        K=1,
        hard_diagnostics=False,
        nodes_r=4,
        nodes_z=4,
    )
    state = uniform_state(context)
    state.B = context.compute.asarray(
        state.B + 0.37 * compatible_magnetic_field(context, seed=19)
    )
    nc = context.op.Nc
    retained = context.p.K * nc
    axial = np.linspace(-0.12, 0.12, nc)
    state.fluid[0].n[retained : retained + nc] *= context.compute.array(
        1.0 + axial
    )
    state.fluid[0].perp[retained : retained + nc] *= context.compute.array(
        1.0 + 0.07 * axial
    )
    state.fluid[1].perp[retained : retained + nc] *= context.compute.array(
        1.0 - 0.05 * axial
    )
    state.fluid[1].parallel[retained : retained + nc] *= context.compute.array(
        1.0 + 0.04 * axial
    )

    preliminary = massless._derive_electron(
        state,
        context,
        ion_transport=False,
    )
    rng = np.random.default_rng(503)
    force = context.compute.array(
        rng.standard_normal(context.N * 3 * nc)
    )
    projected = project_wall(preliminary.ion.wall, force, context).value
    state.fluid[0].mom = forward(projected, 3 * nc, context)
    return context, state


def _state_snapshot(state, context):
    values = {}
    for species in range(2):
        fluid = state.fluid[species]
        for name in ("n", "mom", "perp", "parallel"):
            values[f"{species}.{name}"] = context.compute.to_host(
                getattr(fluid, name)
            ).copy()
    values["B"] = context.compute.to_host(state.B).copy()
    values["identity"] = (
        state.step,
        state.generation,
        state.time,
    )
    return values


def _assert_unchanged(before, state, context):
    after = _state_snapshot(state, context)
    for key, expected in before.items():
        if key == "identity":
            assert after[key] == expected
        else:
            np.testing.assert_array_equal(after[key], expected, err_msg=key)


def _max_relative_error(actual, expected):
    actual = np.asarray(actual, dtype=np.complex128)
    expected = np.asarray(expected, dtype=np.complex128)
    scale = max(float(np.max(np.abs(expected))), 1.0e-300)
    return float(np.max(np.abs(actual - expected)) / scale)


def _assert_field_equivalence(field, full, context):
    xp = context.compute
    cases = {
        "electron_perp": (field.field_rates.electron_perp, full.rhs.fluid[1].perp),
        "electron_parallel": (
            field.field_rates.electron_parallel,
            full.rhs.fluid[1].parallel,
        ),
        "B": (field.field_rates.B, full.rhs.B),
        "E": (field.cache.E, full.cache.E),
        "J": (field.current, full.current),
        "electron_velocity": (field.electron.u, full.fluid[1].u),
        "electron_density": (field.trial_state.fluid[1].n, full.canonical_state.fluid[1].n),
        "electron_momentum": (
            field.trial_state.fluid[1].mom,
            full.canonical_state.fluid[1].mom,
        ),
    }
    errors = {}
    for name, (actual, expected) in cases.items():
        errors[name] = _max_relative_error(
            xp.to_host(actual),
            xp.to_host(expected),
        )
        assert errors[name] <= 2.0e-12, name
    return errors


def test_derive_electron_skips_only_ion_transport_when_requested(monkeypatch):
    context = closed_context(K=1, hard_diagnostics=False)
    state = uniform_state(context)
    calls = []
    original = massless.build_non_electric

    def record(fluid, species, bn, bhat, active_context):
        calls.append(species)
        return original(fluid, species, bn, bhat, active_context)

    monkeypatch.setattr(massless, "build_non_electric", record)
    default = massless._derive_electron(state, context)
    assert calls == [0]

    calls.clear()
    skipped = massless._derive_electron(
        state,
        context,
        ion_transport=False,
    )
    assert calls == []
    for name in (
        "BN",
        "bhat",
        "current",
        "electron_density_hat",
        "electron_density",
        "electron_flux_hat",
        "electron_velocity",
        "compatibility_momentum",
    ):
        np.testing.assert_array_equal(
            getattr(skipped, name),
            getattr(default, name),
            err_msg=name,
        )


def test_derive_electron_rejects_non_boolean_transport_switch():
    context = closed_context(K=0, hard_diagnostics=False)
    state = uniform_state(context)
    with pytest.raises(Failure) as caught:
        massless._derive_electron(
            state,
            context,
            ion_transport=0,
        )
    assert caught.value.code == "Model"
    assert "ion_transport" in str(caught.value)


@pytest.mark.parametrize("fixture", ("periodic_nonuniform", "closed_k1"))
def test_field_rates_match_independent_full_stage(fixture):
    if fixture == "periodic_nonuniform":
        context, state = _periodic_nonuniform_fixture()
        tag = periodic_stage_tag(state, "field_only_oracle")
    else:
        context, state = _closed_nonuniform_fixture()
        tag = closed_stage_tag(state)
    before = _state_snapshot(state, context)

    full = massless.evaluate_stage(state, tag, context, alias=False)
    from fomhd.massless_field import evaluate

    field = evaluate(state, tag, context, alias=False)

    _assert_field_equivalence(field, full, context)
    _assert_unchanged(before, state, context)
    assert field.kind == "massless_field_only"
    assert field.complete_stage is False
    assert not hasattr(field, "rhs")
    assert not hasattr(field, "ledger")
    assert field.work.field_evaluations == 1
    assert field.work.derive_electron == 1
    assert field.work.ion_transport == 0
    assert field.work.finish_fluid == 0
    assert field.work.full_ion_rhs == 0
    assert field.work.wall_cache_evaluations == 1
    assert field.cache.tag == tag
    assert field.tag == tag
    check_names = {check.name for check in field.checks}
    assert {"ct_divergence", "ct_update", "drift_ohm_residual"} <= check_names
    if fixture == "closed_k1":
        assert {
            "wall_current",
            "wall_field_original_rows",
        } <= check_names


def test_em_and_feedback_rates_match_full_force_split_without_transport(
    monkeypatch,
):
    context, state = _closed_nonuniform_fixture()
    tag = closed_stage_tag(state)
    full = massless.evaluate_stage(state, tag, context, alias=False)
    from fomhd import massless_field

    frame = massless_field.evaluate(state, tag, context, alias=False)
    before = _state_snapshot(state, context)
    calls = []

    def reject_ion_transport(fluid, species, bn, bhat, active_context):
        calls.append(species)
        raise AssertionError("new field/EM path called build_non_electric")

    monkeypatch.setattr(
        massless,
        "build_non_electric",
        reject_ion_transport,
    )
    electromagnetic = massless_field.electromagnetic_rate(frame, context)
    feedback = massless_field.feedback_rate(frame, context)
    _assert_unchanged(before, state, context)
    assert calls == []

    xp = context.compute.xp
    nc = context.op.Nc
    qi = context.p.species[0].charge
    ion = frame.ion
    density = xp.asarray(ion.n).reshape(context.N, nc)
    velocity = xp.asarray(ion.u).reshape(context.N, 3, nc)
    magnetic = xp.asarray(frame.BN).reshape(context.N, 3, nc)
    magnetic_force = qi * density[:, None, :] * xp.stack(
        (
            velocity[:, 1, :] * magnetic[:, 2, :]
            - velocity[:, 2, :] * magnetic[:, 1, :],
            velocity[:, 2, :] * magnetic[:, 0, :]
            - velocity[:, 0, :] * magnetic[:, 2, :],
            velocity[:, 0, :] * magnetic[:, 1, :]
            - velocity[:, 1, :] * magnetic[:, 0, :],
        ),
        axis=1,
    )
    electric = inverse(
        frame.cache.E,
        context.op.Ne,
        context,
        True,
        context.p.ref.electric,
    )
    interpolated = sample_apply(
        context.op.H,
        electric,
        context.N,
        context.compute,
    ).reshape(context.N, 3, nc)
    electric_force = qi * density[:, None, :] * interpolated
    expected_projection = project_wall(
        ion.wall,
        (magnetic_force + electric_force).reshape(-1),
        context,
    )
    expected_em = forward(
        expected_projection.value,
        3 * nc,
        context,
    )
    np.testing.assert_allclose(
        electromagnetic.mom,
        expected_em,
        rtol=2.0e-12,
        atol=0.0,
    )
    full_em = forward(
        project_wall(
            full.fluid[0].wall,
            (
                xp.asarray(full.fluid[0].magnetic_force)
                + xp.asarray(full.terms[0].electric)
            ),
            context,
        ).value,
        3 * nc,
        context,
    )
    np.testing.assert_allclose(
        electromagnetic.mom,
        full_em,
        rtol=2.0e-12,
        atol=0.0,
    )
    transport_only = forward(
        project_wall(
            full.fluid[0].wall,
            xp.asarray(full.fluid[0].non_electric)
            - xp.asarray(full.fluid[0].magnetic_force),
            context,
        ).value,
        3 * nc,
        context,
    )
    split_defect = float(
        xp.max(
            xp.abs(
                full.rhs.fluid[0].mom
                - electromagnetic.mom
                - transport_only
            )
        ).item()
    )
    split_scale = max(
        float(xp.max(xp.abs(full.rhs.fluid[0].mom)).item()),
        float(xp.max(xp.abs(transport_only)).item()),
        float(xp.max(xp.abs(electromagnetic.mom)).item()),
        1.0e-300,
    )
    split_bound = (
        8192.0
        * np.finfo(np.float64).eps
        * split_scale
        * max(len(context.geometry.walls), 1)
    )
    assert split_defect <= split_bound
    np.testing.assert_array_equal(
        np.asarray(electromagnetic.n, dtype=np.complex128),
        np.zeros(2 * context.p.K * nc + nc, dtype=np.complex128),
    )
    np.testing.assert_array_equal(
        np.asarray(electromagnetic.perp, dtype=np.complex128),
        np.zeros(2 * context.p.K * nc + nc, dtype=np.complex128),
    )
    np.testing.assert_array_equal(
        feedback.n,
        -xp.asarray(frame.ion.divergence_hat),
    )
    np.testing.assert_array_equal(feedback.n, full.rhs.fluid[0].n)
    np.testing.assert_array_equal(feedback.mom, electromagnetic.mom)
    np.testing.assert_array_equal(feedback.perp, electromagnetic.perp)
    assert frame.work.electromagnetic_rates == 2
    assert frame.work.feedback_rates == 1
    assert frame.work.ion_magnetic_forces == 2
    assert frame.work.wall_projections == 2
    assert frame.work.ion_transport == 0


def test_em_rate_retains_single_padded_producer_record(monkeypatch):
    context, state = _closed_nonuniform_fixture()
    tag = closed_stage_tag(state)
    from fomhd import massless_field

    frame = massless_field.evaluate(state, tag, context, alias=False)
    assert getattr(frame, "ion_em_terms", None) is None
    before_state = _state_snapshot(state, context)
    before_work = copy.copy(frame.work)

    xp = context.compute.xp
    nc = context.op.Nc
    qi = context.p.species[0].charge
    ion = frame.ion
    density = xp.asarray(ion.n).reshape(context.N, nc)
    velocity = xp.asarray(ion.u).reshape(context.N, 3, nc)
    magnetic = xp.asarray(frame.BN).reshape(context.N, 3, nc)
    expected_magnetic = qi * density[:, None, :] * xp.stack(
        (
            velocity[:, 1, :] * magnetic[:, 2, :]
            - velocity[:, 2, :] * magnetic[:, 1, :],
            velocity[:, 2, :] * magnetic[:, 0, :]
            - velocity[:, 0, :] * magnetic[:, 2, :],
            velocity[:, 0, :] * magnetic[:, 1, :]
            - velocity[:, 1, :] * magnetic[:, 0, :],
        ),
        axis=1,
    )
    electric = inverse(
        frame.cache.E,
        context.op.Ne,
        context,
        True,
        context.p.ref.electric,
    )
    interpolated = sample_apply(
        context.op.H,
        electric,
        context.N,
        context.compute,
    ).reshape(context.N, 3, nc)
    expected_electric = qi * density[:, None, :] * interpolated

    projection_calls = 0
    original_project_wall = massless_field.project_wall

    def counted_projection(wall, values, active_context):
        nonlocal projection_calls
        projection_calls += 1
        return original_project_wall(wall, values, active_context)

    monkeypatch.setattr(
        massless_field,
        "project_wall",
        counted_projection,
    )
    electromagnetic = massless_field.electromagnetic_rate(frame, context)
    assert projection_calls == 1
    assert isinstance(electromagnetic, massless_field.IonRate)
    assert len(electromagnetic) == 3

    assert frame.work.electromagnetic_rates == (
        before_work.electromagnetic_rates + 1
    )
    assert frame.work.ion_magnetic_forces == (
        before_work.ion_magnetic_forces + 1
    )
    assert frame.work.wall_projections == before_work.wall_projections + 1
    assert frame.work.warp_magnetic_kernel_evaluations == (
        before_work.warp_magnetic_kernel_evaluations
        + int(context.compute.name == "cuda")
    )
    for name in (
        "field_evaluations",
        "derive_electron",
        "electron_cgl",
        "electron_constraint_projections",
        "electron_magnetic_forces",
        "wall_cache_setups",
        "wall_cache_evaluations",
        "alias_field_evaluations",
        "feedback_rates",
        "ion_transport",
        "finish_fluid",
        "full_ion_rhs",
    ):
        assert getattr(frame.work, name) == getattr(before_work, name)

    terms = frame.ion_em_terms
    assert terms is not None
    assert terms.tag is tag
    with pytest.raises(AttributeError):
        terms.tag = closed_stage_tag(state)
    expected_projection = counted_projection(
        ion.wall,
        (expected_magnetic + expected_electric).reshape(-1),
        context,
    )
    projection_calls_after_independent = projection_calls
    for actual, expected_array in (
        (terms.electric, expected_electric.reshape(-1)),
        (terms.magnetic, expected_magnetic.reshape(-1)),
        (terms.projected, expected_projection.value),
        (terms.reaction, expected_projection.reaction),
    ):
        assert tuple(actual.shape) == (context.N * 3 * nc,)
        np.testing.assert_array_equal(actual, expected_array)
    for actual, expected_array in (
        (terms.multiplier, expected_projection.multiplier),
        (
            terms.multiplier_residual,
            expected_projection.multiplier_residual,
        ),
    ):
        assert tuple(actual.shape) == (
            context.N * len(context.geometry.walls),
        )
        np.testing.assert_array_equal(actual, expected_array)
    np.testing.assert_array_equal(
        electromagnetic.mom,
        forward(terms.projected, 3 * nc, context),
    )
    _assert_unchanged(before_state, state, context)

    retained = {
        name: xp.asarray(getattr(terms, name)).copy()
        for name in (
            "electric",
            "magnetic",
            "projected",
            "reaction",
            "multiplier",
            "multiplier_residual",
        )
    }
    projection_calls = 0
    assert frame.ion_em_terms is terms
    for name, expected_array in retained.items():
        np.testing.assert_array_equal(getattr(terms, name), expected_array)
    assert projection_calls == 0
    assert projection_calls_after_independent == 2
    assert frame.work.ion_transport == 0
    assert frame.work.finish_fluid == 0
    assert frame.work.full_ion_rhs == 0


def test_em_rate_normalizes_flat_warp_magnetic_producer(monkeypatch):
    context, state = _closed_nonuniform_fixture()
    tag = closed_stage_tag(state)
    from fomhd import massless_field

    frame = massless_field.evaluate(state, tag, context, alias=False)
    xp = context.compute.xp
    nc = context.op.Nc
    qi = context.p.species[0].charge
    ion = frame.ion
    density = xp.asarray(ion.n).reshape(context.N, nc)
    velocity = xp.asarray(ion.u).reshape(context.N, 3, nc)
    magnetic = xp.asarray(frame.BN).reshape(context.N, 3, nc)
    expected_magnetic = qi * density[:, None, :] * xp.stack(
        (
            velocity[:, 1, :] * magnetic[:, 2, :]
            - velocity[:, 2, :] * magnetic[:, 1, :],
            velocity[:, 2, :] * magnetic[:, 0, :]
            - velocity[:, 0, :] * magnetic[:, 2, :],
            velocity[:, 0, :] * magnetic[:, 1, :]
            - velocity[:, 1, :] * magnetic[:, 0, :],
        ),
        axis=1,
    )
    flat_warp_output = context.compute.array(expected_magnetic.reshape(-1))
    warp_calls = 0

    def flat_cuda_producer(velocity_values, magnetic_values, density_values,
                           charge, active_context):
        nonlocal warp_calls
        warp_calls += 1
        assert tuple(flat_warp_output.shape) == (
            active_context.N * 3 * active_context.op.Nc,
        )
        return flat_warp_output

    monkeypatch.setattr(
        massless_field,
        "_warp_magnetic_force",
        flat_cuda_producer,
    )
    before_work = copy.copy(frame.work)
    electromagnetic = massless_field.electromagnetic_rate(frame, context)
    assert warp_calls == 1

    electric = inverse(
        frame.cache.E,
        context.op.Ne,
        context,
        True,
        context.p.ref.electric,
    )
    interpolated = sample_apply(
        context.op.H,
        electric,
        context.N,
        context.compute,
    ).reshape(context.N, 3, nc)
    expected_electric = qi * density[:, None, :] * interpolated
    expected_projection = project_wall(
        ion.wall,
        (expected_magnetic + expected_electric).reshape(-1),
        context,
    )
    expected_rate = forward(
        expected_projection.value,
        3 * nc,
        context,
    )
    np.testing.assert_array_equal(electromagnetic.mom, expected_rate)

    terms = frame.ion_em_terms
    assert terms is not None
    assert np.shares_memory(
        np.asarray(flat_warp_output),
        np.asarray(terms.magnetic),
    )
    for name, expected_array in (
        ("electric", expected_electric.reshape(-1)),
        ("magnetic", expected_magnetic.reshape(-1)),
        ("projected", expected_projection.value),
        ("reaction", expected_projection.reaction),
    ):
        actual = getattr(terms, name)
        assert tuple(actual.shape) == (context.N * 3 * nc,)
        np.testing.assert_array_equal(actual, expected_array)
    np.testing.assert_array_equal(
        terms.multiplier,
        expected_projection.multiplier,
    )
    np.testing.assert_array_equal(
        terms.multiplier_residual,
        expected_projection.multiplier_residual,
    )
    assert frame.work.electromagnetic_rates == (
        before_work.electromagnetic_rates + 1
    )
    assert frame.work.ion_magnetic_forces == (
        before_work.ion_magnetic_forces + 1
    )
    assert frame.work.warp_magnetic_kernel_evaluations == (
        before_work.warp_magnetic_kernel_evaluations + 1
    )
    assert frame.work.wall_projections == before_work.wall_projections + 1
    assert frame.work.ion_transport == 0
    assert frame.work.finish_fluid == 0
    assert frame.work.full_ion_rhs == 0


def test_electron_warp_kernel_count_tracks_cuda_base_and_alias(monkeypatch):
    context, state = _closed_nonuniform_fixture()
    tag = closed_stage_tag(state)
    from fomhd import massless_field

    cpu_frame = massless_field.evaluate(state, tag, context, alias=False)
    assert cpu_frame.work.electron_magnetic_forces == 1
    assert cpu_frame.work.warp_magnetic_kernel_evaluations == 0

    real_backend = context.compute

    class CudaNamedBackend:
        name = "cuda"

        def __getattr__(self, item):
            return getattr(real_backend, item)

    cuda_context = copy.copy(context)
    cuda_context.compute = CudaNamedBackend()
    dispatch_samples = []

    def electron_warp_dispatch(
        velocity,
        magnetic,
        density,
        charge,
        active_context,
    ):
        dispatch_samples.append(active_context.N)
        return active_context.compute.array(
            np.zeros(velocity.shape, dtype=np.complex128)
        )

    monkeypatch.setattr(
        massless,
        "_warp_magnetic_force",
        electron_warp_dispatch,
    )
    frame = massless_field.evaluate(
        state,
        tag,
        cuda_context,
        alias=True,
    )

    assert dispatch_samples == [context.N, 2 * context.N + 1]
    assert frame.work.electron_magnetic_forces == 2
    assert frame.work.warp_magnetic_kernel_evaluations == 2
    assert frame.work.ion_magnetic_forces == 0
    assert frame.alias_frame is not None
    assert frame.alias_frame.work.electron_magnetic_forces == 1
    assert frame.alias_frame.work.warp_magnetic_kernel_evaluations == 1
    assert frame.work.ion_transport == 0
    assert frame.work.finish_fluid == 0
    assert frame.work.full_ion_rhs == 0


def test_em_rejects_same_shape_foreign_context_without_charging(monkeypatch):
    context, state = _closed_nonuniform_fixture()
    tag = closed_stage_tag(state)
    from fomhd import massless_field

    frame = massless_field.evaluate(state, tag, context, alias=False)
    assert frame.owner_context is context
    assert frame.ion_em_terms is None
    before_work = copy.copy(frame.work)
    before_state = _state_snapshot(state, context)

    foreign = closed_context(
        K=1,
        hard_diagnostics=False,
        nodes_r=4,
        nodes_z=4,
    )
    foreign.p.species[0].charge *= 2.0
    assert foreign.N == context.N
    assert foreign.op.Nc == context.op.Nc
    assert foreign.op.Ne == context.op.Ne
    assert foreign.p.species[0].charge == (
        2.0 * context.p.species[0].charge
    )

    calls = []
    original_warp = massless_field._warp_magnetic_force

    def forbidden_warp(*args):
        active_context = args[-1]
        calls.append(active_context)
        raise AssertionError(
            f"foreign ownership reached Warp dispatch for N={active_context.N}"
        )

    monkeypatch.setattr(
        massless_field,
        "_warp_magnetic_force",
        forbidden_warp,
    )
    for operation in (
        massless_field.electromagnetic_rate,
        massless_field.feedback_rate,
    ):
        with pytest.raises(Failure) as caught:
            operation(frame, foreign)
        assert caught.value.code == "FieldOwnership"
    assert calls == []
    assert frame.ion_em_terms is None
    for name in (
        "electromagnetic_rates",
        "feedback_rates",
        "ion_magnetic_forces",
        "warp_magnetic_kernel_evaluations",
        "wall_projections",
        "ion_transport",
        "finish_fluid",
        "full_ion_rhs",
    ):
        assert getattr(frame.work, name) == getattr(before_work, name)
    _assert_unchanged(before_state, state, context)

    monkeypatch.setattr(
        massless_field,
        "_warp_magnetic_force",
        original_warp,
    )
    electromagnetic = massless_field.electromagnetic_rate(
        frame,
        frame.owner_context,
    )
    assert frame.ion_em_terms is not None
    assert frame.work.electromagnetic_rates == (
        before_work.electromagnetic_rates + 1
    )
    assert frame.work.ion_magnetic_forces == (
        before_work.ion_magnetic_forces + 1
    )
    assert frame.work.wall_projections == before_work.wall_projections + 1
    assert np.isfinite(np.asarray(electromagnetic.mom)).all()


def test_alias_frame_exposes_its_padded_owner_and_rejects_base_context():
    context, state = _closed_nonuniform_fixture()
    tag = closed_stage_tag(state)
    from fomhd import massless_field

    frame = massless_field.evaluate(state, tag, context, alias=True)
    alias = frame.alias_frame
    assert alias is not None
    assert alias.owner_context is not context
    assert alias.owner_context.N == 2 * context.N + 1
    assert alias.owner_context.spectral_id == alias.tag.spectral_context
    before_work = copy.copy(alias.work)
    with pytest.raises(Failure) as caught:
        massless_field.electromagnetic_rate(alias, context)
    assert caught.value.code == "FieldOwnership"
    assert alias.ion_em_terms is None
    assert alias.work.electromagnetic_rates == before_work.electromagnetic_rates

    massless_field.electromagnetic_rate(alias, alias.owner_context)
    assert alias.ion_em_terms is not None
    assert alias.ion_em_terms.tag is alias.tag
    assert alias.work.electromagnetic_rates == (
        before_work.electromagnetic_rates + 1
    )
    assert alias.work.wall_projections == before_work.wall_projections + 1


def test_field_alias_is_padded_field_only_and_charged(monkeypatch):
    context, state = _closed_nonuniform_fixture()
    tag = closed_stage_tag(state)
    before = _state_snapshot(state, context)
    calls = []
    updated = copy.copy(context)
    updated.N = 2 * context.N + 1
    updated.spectral_id = context.spectral_id + 1
    updated._history_samples = context.N
    alternate_tag = type(tag)(
        step=tag.step,
        generation=tag.generation,
        trial=tag.trial,
        spectral_context=updated.spectral_id,
        time=tag.time,
        stage="alias",
        micro_step=tag.micro_step,
        iteration=tag.iteration,
        slab=tag.slab,
    )
    full_alternate = massless.evaluate_stage(
        state,
        alternate_tag,
        updated,
        alias=False,
    )

    def reject_ion_transport(fluid, species, bn, bhat, active_context):
        calls.append(species)
        raise AssertionError("field alias called ion transport")

    monkeypatch.setattr(
        massless,
        "build_non_electric",
        reject_ion_transport,
    )
    from fomhd import massless_checks, massless_field

    def reject_full_stage_path(*args, **kwargs):
        raise AssertionError("field alias used the legacy full-stage path")

    monkeypatch.setattr(
        massless_checks,
        "check_stage",
        reject_full_stage_path,
    )
    monkeypatch.setattr(
        massless,
        "_alias_probe",
        reject_full_stage_path,
    )

    result = massless_field.evaluate(state, tag, context, alias=True)
    _assert_unchanged(before, state, context)
    assert calls == []
    assert result.alias_frame is not None
    assert result.alias_frame.tag.spectral_context == tag.spectral_context + 1
    assert result.alias_frame.tag.stage == "alias"
    xp = context.compute
    for actual, expected in (
        (
            result.alias_frame.field_rates.electron_perp,
            full_alternate.rhs.fluid[1].perp,
        ),
        (
            result.alias_frame.field_rates.electron_parallel,
            full_alternate.rhs.fluid[1].parallel,
        ),
        (result.alias_frame.field_rates.B, full_alternate.rhs.B),
    ):
        np.testing.assert_array_equal(
            xp.to_host(actual),
            xp.to_host(expected),
        )
    assert result.work.field_evaluations == 2
    assert result.work.alias_field_evaluations == 1
    assert result.work.derive_electron == 2
    assert result.work.electron_cgl == 2
    assert result.work.ion_transport == 0
    assert result.work.finish_fluid == 0
    assert result.work.warp_magnetic_kernel_evaluations == 0
    assert result.alias_frame.work.warp_magnetic_kernel_evaluations == 0
    assert result.work.wall_cache_evaluations == 2
    names = {check.name for check in result.checks}
    assert {"alias_perp", "alias_parallel", "alias_B"} <= names
    assert "alias_n" not in names
    assert "alias_mom" not in names

    massless_field.electromagnetic_rate(result, context)
    massless_field.feedback_rate(result, context)
    assert calls == []


def test_distinct_tags_have_distinct_owned_caches_and_rates():
    context, state = _closed_nonuniform_fixture()
    first_tag = closed_stage_tag(state)
    second_tag = copy.copy(first_tag)
    second_tag.stage = "rk4_second_middle"
    second_tag.micro_step = 3
    second_tag.iteration = 2
    from fomhd.massless_field import evaluate

    first = evaluate(state, first_tag, context)
    second = evaluate(state, second_tag, context)
    assert first.tag == first_tag
    assert second.tag == second_tag
    assert first.cache.tag == first_tag
    assert second.cache.tag == second_tag
    assert first.cache is not second.cache
    assert first.field_rates.B is not second.field_rates.B
    assert first.trial_state is not second.trial_state
    np.testing.assert_array_equal(first.cache.E, second.cache.E)
    np.testing.assert_array_equal(first.field_rates.B, second.field_rates.B)


def test_original_budget_closed_wall_K1_rejects_both_paths():
    context = closed_context(
        K=1,
        hard_diagnostics=True,
        nodes_r=4,
        nodes_z=4,
    )
    state = uniform_state(context)
    tag = closed_stage_tag(state)
    from fomhd.massless_field import evaluate

    def failed_wall_flux(failure):
        records = [
            check
            for check in failure.checks
            if check.name == "wall_electron_flux"
            and check.hard
            and check.available
            and not check.passed
            and float(check.raw) > float(check.threshold)
        ]
        assert records
        record = max(records, key=lambda item: float(item.raw))
        assert float(record.threshold) == 20000.0
        assert float(record.raw) > float(record.threshold)
        return float(record.raw)

    with pytest.raises(Failure) as legacy:
        massless.evaluate_stage(state, tag, context, alias=False)
    assert legacy.value.code == "HardGate"
    legacy_raw = failed_wall_flux(legacy.value)

    with pytest.raises(Failure) as field:
        evaluate(state, tag, context, alias=False)
    assert field.value.code == "HardGate"
    field_raw = failed_wall_flux(field.value)
    assert field_raw == pytest.approx(
        legacy_raw,
        rel=0.0,
        abs=1.0e-10,
    )


def test_incompatible_initial_current_is_rejected_with_field_evidence():
    context = closed_context(K=0, hard_diagnostics=False)
    state = uniform_state(context)
    state.B = wall_incompatible_current_field(context)
    tag = closed_stage_tag(state)
    from fomhd.massless_field import evaluate

    with pytest.raises(Failure) as caught:
        evaluate(state, tag, context)
    failure = caught.value
    assert failure.code in {"WallCurrent", "InitialCompatibility"}
    assert failure.stage == tag
    assert failure.field_only is True
    assert failure.partial_arrays


def test_failed_field_constraint_retains_tag_and_partial_arrays(monkeypatch):
    context, state = _closed_nonuniform_fixture()
    tag = closed_stage_tag(state)
    from fomhd import massless_field

    def fail(_derived, _context):
        raise Failure("InjectedConstraint", "bounded injected field failure")

    monkeypatch.setattr(massless_field, "_initial_constraints", fail)
    with pytest.raises(Failure) as caught:
        massless_field.evaluate(state, tag, context)
    failure = caught.value
    assert failure.code == "InjectedConstraint"
    assert failure.stage == tag
    assert failure.field_only is True
    assert failure.partial_arrays
    assert any("_input_" in key for key in failure.partial_arrays)
    for check in failure.checks:
        assert check.tag == tag
