"""Behavioral tests for the massless quasineutral electron branch."""

from __future__ import annotations

import numpy as np
import pytest

from fomhd.core import inverse, mode_apply, sample_apply
from fomhd.fluid import gradient_adjoint, velocity_gradient
from fomhd.types import Failure

from test_massless_support import (
    context,
    equilibrium_state,
    perturb_only_compatibility_momentum,
    stage_tag,
    uniform_state,
    wall_incompatible_current_field,
    weighted_dot,
)


def _copy_state_arrays(state):
    return {
        "i.n": np.asarray(state.fluid[0].n).copy(),
        "i.mom": np.asarray(state.fluid[0].mom).copy(),
        "i.perp": np.asarray(state.fluid[0].perp).copy(),
        "e.n": np.asarray(state.fluid[1].n).copy(),
        "e.mom": np.asarray(state.fluid[1].mom).copy(),
        "e.perp": np.asarray(state.fluid[1].perp).copy(),
        "e.parallel": np.asarray(state.fluid[1].parallel).copy(),
        "B": np.asarray(state.B).copy(),
    }


def _assert_same_state_arrays(state, expected):
    actual = _copy_state_arrays(state)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


def test_canonicalize_is_immutable_and_compatibility_momentum_is_not_consumed():
    from fomhd.massless import canonicalize_state, evaluate_stage

    c = context(hard_diagnostics=False)
    state = uniform_state(c)
    state.fluid[1].n.fill(3.0 * c.p.ref.ne)
    state.fluid[1].mom.fill(7.0)
    before = _copy_state_arrays(state)

    canonical = canonicalize_state(state, c)
    _assert_same_state_arrays(state, before)
    assert canonical is not state
    assert canonical.fluid[1].n is not state.fluid[1].n
    assert canonical.fluid[1].mom is not state.fluid[1].mom
    np.testing.assert_allclose(
        np.asarray(canonical.fluid[1].n),
        c.p.species[0].charge
        * np.asarray(canonical.fluid[0].n)
        / (-c.p.species[1].charge),
        rtol=2.0e-13,
        atol=0.0,
    )

    canonical_before = _copy_state_arrays(canonical)
    first = evaluate_stage(canonical, stage_tag(canonical), c)
    _assert_same_state_arrays(canonical, canonical_before)
    perturbed = perturb_only_compatibility_momentum(canonical, c)
    second = evaluate_stage(perturbed, stage_tag(perturbed), c)
    for first_fluid, second_fluid in zip(first.rhs.fluid, second.rhs.fluid):
        for name in ("n", "mom", "perp", "parallel"):
            np.testing.assert_allclose(
                np.asarray(getattr(second_fluid, name)),
                np.asarray(getattr(first_fluid, name)),
                rtol=2.0e-12,
                atol=0.0,
                err_msg=f"electron momentum was consumed: {name}",
            )
    np.testing.assert_allclose(second.rhs.B, first.rhs.B, rtol=2.0e-12, atol=0.0)


def test_nonzero_wall_compatible_current_and_electron_flux_definitions():
    from fomhd.massless import evaluate_stage

    c = context(hard_diagnostics=False, nodes_r=6, nodes_z=6)
    state = uniform_state(c)
    result = evaluate_stage(state, stage_tag(state), c)
    xp = c.compute.xp
    current_reference = (
        c.p.species[0].charge * result.fluid[0].flux_hat
        + c.p.species[1].charge * result.fluid[1].flux_hat
    )
    expected_j = mode_apply(c.op.Cdag, state.B, c.compute) / c.p.mu0
    np.testing.assert_allclose(result.current, expected_j, rtol=2.0e-12, atol=0.0)
    current_norm = float(xp.max(xp.abs(result.current)))
    current_term = 0.0
    for stored in range(2 * c.p.K + 1):
        matrix = np.asarray(c.op.Cdag[stored])
        values = np.asarray(state.B)[
            stored * c.op.Nb : (stored + 1) * c.op.Nb
        ]
        current_term = max(
            current_term,
            float(np.sum(np.abs(matrix) * np.abs(values)[None, :]))
            / c.p.mu0,
        )
    roundoff_scale = 8192.0 * np.finfo(np.float64).eps * current_term
    assert current_norm > 1.0e4 * roundoff_scale
    assert current_norm > 1.0e-6 * current_term
    curl_e = mode_apply(c.op.C, result.cache.E, c.compute)
    curl_norm = float(xp.max(xp.abs(curl_e)))
    curl_term = 0.0
    for stored in range(2 * c.p.K + 1):
        matrix = np.asarray(c.op.C[stored])
        values = np.asarray(result.cache.E)[
            stored * c.op.Ne : (stored + 1) * c.op.Ne
        ]
        curl_term = max(
            curl_term,
            float(np.sum(np.abs(matrix) * np.abs(values)[None, :])),
        )
    curl_roundoff = 8192.0 * np.finfo(np.float64).eps * curl_term
    assert curl_norm > 1.0e4 * curl_roundoff
    assert curl_norm > 1.0e-6 * curl_term
    np.testing.assert_allclose(
        current_reference,
        result.current,
        rtol=2.0e-12,
        atol=0.0,
    )
    nr = len(c.geometry.r_faces.keys)
    extraction = np.zeros((len(c.geometry.walls), c.op.Ne), dtype=np.complex128)
    for row, wall in enumerate(c.geometry.walls):
        edge = (
            wall.face
            if wall.component == "r_face"
            else nr + c.op.Nc + wall.face
        )
        extraction[row, edge] = wall.sign
    for flux in (
        result.fluid[0].flux_hat,
        result.fluid[1].flux_hat,
        result.current,
    ):
        stored = flux.reshape(2 * c.p.K + 1, c.op.Ne)
        wall = stored @ extraction.T
        assert float(xp.max(xp.abs(wall))) <= 1.0e-9 * max(
            float(xp.max(xp.abs(flux))), 1.0e-300
        )
    divergence = mode_apply(c.op.DJ, result.current, c.compute)
    assert float(xp.max(xp.abs(divergence))) <= 1.0e-9 * max(
        float(xp.max(xp.abs(result.current))), 1.0e-300
    )


def test_uniform_closed_current_free_equilibrium_has_zero_force():
    from fomhd.massless import evaluate_stage

    c = context(hard_diagnostics=False)
    state = equilibrium_state(c)
    result = evaluate_stage(state, stage_tag(state), c)
    xp = c.compute.xp
    term_scales = {
        "ion_momentum": (
            c.p.species[0].mass
            * c.p.ref.ni
            * c.p.ref.velocity
            / c.p.ref.time
        ),
        "electron_pressure": c.p.ref.pressure / c.p.ref.time,
        "field": c.p.ref.magnetic / c.p.ref.time,
    }
    residuals = {
        "ion_momentum": float(xp.max(xp.abs(result.rhs.fluid[0].mom))),
        "electron_pressure": max(
            float(xp.max(xp.abs(result.rhs.fluid[1].perp))),
            float(xp.max(xp.abs(result.rhs.fluid[1].parallel))),
        ),
        "field": float(xp.max(xp.abs(result.rhs.B))),
    }
    for name, residual in residuals.items():
        assert residual <= 1.0e-9 * term_scales[name], (name, residual)


def test_variable_density_pressure_adjoint_and_power_pairing():
    from fomhd.massless import evaluate_stage

    c = context(hard_diagnostics=False)
    nc = c.op.Nc
    state = uniform_state(c)
    zero = c.p.K * nc
    r = np.linspace(0.8, 1.2, nc)
    state.fluid[0].n[zero : zero + nc] *= r
    state.fluid[0].perp[zero : zero + nc] *= r
    state.fluid[1].perp[zero : zero + nc] *= r
    state.fluid[1].parallel[zero : zero + nc] *= 0.7 * r
    result = evaluate_stage(state, stage_tag(state), c)
    electron = result.fluid[1]
    xp = c.compute.xp

    rng = np.random.default_rng(91)
    tensor = c.compute.array(
        rng.standard_normal(electron.tensor.size)
        + 1j * rng.standard_normal(electron.tensor.size)
    )
    gradient = velocity_gradient(electron, electron.u, c)
    adjoint = gradient_adjoint(electron, tensor, c)
    m0 = xp.real(xp.asarray(c.op.M0)).astype(xp.float64)
    tensor_weights = xp.tile(m0, 9)
    velocity_weights = xp.tile(m0, 3)
    left = weighted_dot(c, gradient, tensor, tensor_weights)
    right = weighted_dot(c, electron.u, adjoint, velocity_weights)
    assert abs(left - right) <= 2.0e-9 * max(abs(left), abs(right), 1.0e-300)

    pressure_power = result.ledger[1].pressure_kinetic
    internal_power = result.ledger[1].pressure_internal
    electric_power = result.ledger[1].pressure_electric_edge
    scale = max(abs(pressure_power), abs(internal_power), 1.0e-300)
    assert abs(pressure_power + internal_power) <= 2.0e-8 * scale
    electric_scale = max(abs(pressure_power), abs(electric_power), 1.0e-300)
    assert abs(pressure_power + electric_power) <= 2.0e-8 * electric_scale


def test_wall_correction_checks_all_original_rows_and_zero_species_work():
    from fomhd.massless import evaluate_stage

    c = context(hard_diagnostics=False)
    state = uniform_state(c)
    result = evaluate_stage(state, stage_tag(state), c)
    correction = result.cache.wall_correction
    assert correction.rank == sum(record.rank for record in correction.modes)
    for record in correction.modes:
        residual = np.asarray(record.full_residual)
        rhs = np.asarray(record.rhs)
        assert residual.size == rhs.size
        np.testing.assert_allclose(
            residual,
            np.zeros_like(residual),
            rtol=0.0,
            atol=2.0e-9 * max(float(np.max(np.abs(rhs))), 1.0e-300),
        )
        assert len(record.selected_rows) == record.rank
        omitted = sorted(set(range(rhs.size)) - set(record.selected_rows))
        np.testing.assert_array_equal(record.gauge_rows, omitted)

    reaction = result.cache.E - result.E0
    m1 = xp_tile(c, result.cache.E.size)
    mode = correction.modes[0]
    for flux, charge in (
        (result.fluid[0].flux, c.p.species[0].charge),
        (result.fluid[1].flux, c.p.species[1].charge),
        (inverse(result.current, c.op.Ne, c), 1.0),
    ):
        work = charge * weighted_dot(c, flux, reaction, m1)
        wall_flux = np.zeros(len(c.geometry.walls), dtype=np.complex128)
        for wall_index, wall in enumerate(c.geometry.walls):
            edge = (
                wall.face
                if wall.component == "r_face"
                else len(c.geometry.r_faces.keys) + c.op.Nc + wall.face
            )
            wall_flux[wall_index] = wall.sign * np.asarray(flux)[edge]
        algebraic_work = charge * np.sum(
            wall_flux * np.asarray(mode.multiplier)
        )
        work_scale = max(
            float(np.max(np.abs(wall_flux)))
            * float(np.max(np.abs(mode.multiplier))),
            1.0e-300,
        )
        assert abs(work - algebraic_work) <= 2.0e-10 * work_scale
        full_flux_scale = max(
            float(np.max(np.abs(np.asarray(flux)))), 1.0e-300
        )
        roundoff_bound = (
            8192.0
            * np.finfo(np.float64).eps
            * full_flux_scale
            * float(np.max(np.abs(mode.multiplier)))
            * len(c.geometry.walls)
        )
        assert abs(work) <= roundoff_bound


def test_pressure_tensor_moments_and_cgl_source_identities():
    from fomhd.massless import evaluate_stage

    c = context(hard_diagnostics=False)
    state = uniform_state(c)
    result = evaluate_stage(state, stage_tag(state), c)
    electron = result.fluid[1]
    xp = c.compute.xp
    nc = c.op.Nc
    tensor = electron.tensor.reshape(c.N, 3, 3, nc)
    direction = np.asarray(result.bhat).reshape(c.N, 3, nc)
    perp = electron.perp.reshape(c.N, nc)
    parallel = electron.parallel.reshape(c.N, nc)
    expected = perp[:, None, None, :] * xp.eye(
        3, dtype=xp.complex128
    ).reshape(1, 3, 3, 1)
    expected += (
        (parallel - perp)[:, None, None, :]
        * direction[:, :, None, :]
        * xp.conj(direction[:, None, :, :])
    )
    np.testing.assert_allclose(tensor, expected, rtol=2.0e-12, atol=0.0)

    gradient = electron.gradient.reshape(c.N, 3, 3, nc)
    contraction = xp.sum(tensor * gradient, axis=(1, 2))
    source = (
        perp * (electron.strain.reshape(c.N, nc) - electron.theta.reshape(c.N, nc))
        - parallel * electron.strain.reshape(c.N, nc)
    )
    np.testing.assert_allclose(
        source + contraction,
        xp.zeros_like(source),
        rtol=0.0,
        atol=2.0e-8 * max(float(xp.max(xp.abs(source))), 1.0e-300),
    )


def test_theta_consistency_uses_density_interpolation_not_global_ne():
    from fomhd.massless import evaluate_stage

    c = context(hard_diagnostics=False)
    state = uniform_state(c)
    result = evaluate_stage(state, stage_tag(state), c)
    electron = result.fluid[1]
    xp = c.compute.xp
    samples = electron.flux.reshape(c.N, c.op.Ne) / electron.nf.reshape(
        c.N, c.op.Ne
    )
    from fomhd.core import forward

    expected = mode_apply(c.op.DJ, forward(samples.reshape(-1), c.op.Ne, c), c.compute)
    electron.theta_consistency = expected
    defect = electron.theta - inverse(expected, c.op.Nc, c)
    np.testing.assert_allclose(
        defect,
        result.residuals["theta_consistency"],
        rtol=2.0e-12,
        atol=0.0,
    )
    assert np.isfinite(np.asarray(expected)).all()


def test_totals_exclude_electron_kinetic_and_longitudinal_energy():
    from fomhd.massless import evaluate_stage

    c = context(hard_diagnostics=False)
    state = uniform_state(c)
    result = evaluate_stage(state, stage_tag(state), c)
    assert result.totals.kinetic[1] == 0.0
    assert result.totals.longitudinal == 0.0
    assert result.totals.energy == (
        result.totals.kinetic[0]
        + result.totals.ion_internal
        + result.totals.electron_perp
        + result.totals.electron_parallel
        + result.totals.magnetic
    )
    assert result.totals.applicability.electron_kinetic_energy == "not_applicable"
    assert (
        result.totals.applicability.longitudinal_electric_energy
        == "not_applicable"
    )


def test_configured_hard_stage_gates_accept_physical_fixture():
    from fomhd.massless import evaluate_stage

    c = context(hard_diagnostics=True)
    for name, budget in c.p.budgets.items():
        if name.startswith("ct_divergence.") or name.startswith(
            "wall_electron_flux."
        ):
            budget.rtol = 1.0e-5
    state = uniform_state(c)
    result = evaluate_stage(state, stage_tag(state), c)
    hard = [record for record in result.checks if record.hard]
    assert hard
    assert all(record.available and record.passed for record in hard)
    theta = [
        record
        for record in result.checks
        if record.name == "theta_consistency"
    ]
    assert theta
    assert all(not record.hard for record in theta)
    assert any(
        not record.available
        and record.name == "electron_kinetic_energy"
        for record in result.checks
    )


def test_multimode_storage_alias_and_full_wall_constraints():
    from fomhd.massless import evaluate_stage

    c = context(K=1, hard_diagnostics=False)
    state = uniform_state(c)
    result = evaluate_stage(state, stage_tag(state), c, alias=True)
    xp = c.compute.xp
    assert result.cache.E.size == 3 * c.op.Ne
    assert result.rhs.B.size == 3 * c.op.Nb
    assert len(result.cache.wall_correction.modes) == 3
    nr = len(c.geometry.r_faces.keys)
    S = np.zeros((len(c.geometry.walls), c.op.Ne), dtype=np.complex128)
    for row, wall in enumerate(c.geometry.walls):
        edge = (
            wall.face
            if wall.component == "r_face"
            else nr + c.op.Nc + wall.face
        )
        S[row, edge] = wall.sign
    for name, flux in (
        ("ion", result.fluid[0].flux_hat),
        ("electron", result.fluid[1].flux_hat),
        ("current", result.current),
    ):
        stored = flux.reshape(3, c.op.Ne)
        wall_flux = stored @ S.T
        scale = float(xp.max(xp.abs(stored)))
        assert float(xp.max(xp.abs(wall_flux))) <= 1.0e-9 * max(
            scale, 1.0e-300
        ), name
    assert any(check.name == "alias_B" for check in result.checks)
    assert any(check.name == "alias_mom" for check in result.checks)


def test_zero_physical_electron_mass_metadata_is_rejected():
    from fomhd.massless import canonicalize_state

    c = context(hard_diagnostics=False)
    state = uniform_state(c)
    c.p.species[1].mass = 0.0
    with pytest.raises(Failure) as error:
        canonicalize_state(state, c)
    assert error.value.code == "Model"


def test_initialize_rejects_incompatible_current_and_commits_canonical_state():
    from types import SimpleNamespace

    from fomhd.massless import initialize
    from fomhd.types import cumulative_record

    c = context(hard_diagnostics=False)
    bad = uniform_state(c)
    bad.B = wall_incompatible_current_field(c)
    cumulative = cumulative_record()
    with pytest.raises(Failure) as error:
        initialize(bad, c, cumulative, equilibrium=False)
    assert error.value.code in {"WallCurrent", "InitialCompatibility"}
    assert "roundoff_bound" in str(error.value)
    assert cumulative.baseline.divB is None

    good = uniform_state(c, compatible=True)
    good.fluid[1].n.fill(2.0 * c.p.ref.ne)
    committed = initialize(good, c, cumulative, equilibrium=False)
    assert committed.accepted
    np.testing.assert_allclose(
        committed.state.fluid[1].n,
        c.compute.array(good.fluid[0].n)
        * c.p.species[0].charge
        / (-c.p.species[1].charge),
        rtol=2.0e-12,
        atol=0.0,
    )
    assert cumulative.last_step == good.step
    assert cumulative.last_time == good.time


@pytest.mark.parametrize("integrator", ("midpoint", "rk4"))
def test_explicit_advance_routes_and_preserves_massless_constraints(integrator):
    from types import SimpleNamespace

    from fomhd.massless import canonicalize_state, initialize
    from fomhd.time import advance

    c = context(hard_diagnostics=False)
    c.p.nsteps = 1
    c.p.integrator = integrator
    c.p.dt_guard = "warn"
    state = uniform_state(c)
    from fomhd.types import cumulative_record

    cumulative = cumulative_record()
    committed = initialize(state, c, cumulative, equilibrium=False)
    before = np.asarray(committed.state.fluid[1].n).copy()
    result = advance(committed, cumulative, c)

    assert result.accepted
    assert committed.accepted
    assert committed.state.step == 1
    assert committed.state.generation != state.generation
    np.testing.assert_allclose(
        committed.state.fluid[1].n,
        canonicalize_state(committed.state, c).fluid[1].n,
        rtol=2.0e-12,
        atol=0.0,
    )
    assert committed.state.fluid[1].n.shape == before.shape
    assert result.interval.accepted
    assert cumulative.accepted_intervals == 1
    assert cumulative.last_step == 1
    assert result.interval.applicability.electron_kinetic_energy == "not_applicable"
    momentum_records = [
        check
        for check in result.checks
        if check.name.startswith("momentum_balance_interval_")
        and check.region == "all"
        and str(check.norm) == "value"
    ]
    assert len(momentum_records) == 3
    assert all(check.unit == "kg*m/s" for check in momentum_records)


def test_manufactured_independent_electron_source_is_rejected():
    from types import SimpleNamespace

    from fomhd.massless import initialize
    from fomhd.types import cumulative_record
    from fomhd.time import advance

    c = context(hard_diagnostics=False)
    c.p.nsteps = 1
    c.p.dt_guard = "warn"
    state = uniform_state(c)
    cumulative = cumulative_record()
    committed = initialize(state, c, cumulative, equilibrium=False)

    def electron_only_source(state, tag):
        source = type(state)(
            fluid=(
                type(state.fluid[0])(
                    n=np.zeros_like(state.fluid[0].n),
                    mom=np.zeros_like(state.fluid[0].mom),
                    perp=np.zeros_like(state.fluid[0].perp),
                    parallel=np.zeros_like(state.fluid[0].parallel),
                ),
                type(state.fluid[1])(
                    n=np.ones_like(state.fluid[1].n),
                    mom=np.ones_like(state.fluid[1].mom),
                    perp=np.zeros_like(state.fluid[1].perp),
                    parallel=np.zeros_like(state.fluid[1].parallel),
                ),
            ),
            B=np.zeros_like(state.B),
            step=state.step,
            generation=state.generation,
            time=state.time,
        )
        return source

    with pytest.raises(Failure) as error:
        advance(committed, cumulative, c, electron_only_source)
    assert error.value.code == "ManufacturedSource"
    assert committed.state.step == 0
    assert cumulative.accepted_intervals == 0


def test_failure_evidence_preserves_priors_and_adds_massless_requirements():
    from types import SimpleNamespace

    from fomhd.massless import retain_failure_evidence
    from fomhd.types import Failure

    c = context(hard_diagnostics=False)
    state = uniform_state(c)
    tag = stage_tag(state)
    prior = SimpleNamespace(name="prior", population="", component="",
                            producer="test", region="all", norm="value",
                            tag=tag, available=True, passed=False, hard=False)
    failure = Failure("Injected", "injected failure")
    failure.checks.append(prior)
    retain_failure_evidence(
        failure, tag, c, state, None, "failure_test"
    )
    assert prior in failure.checks
    assert failure.stage == tag
    assert any(
        check.name == "quasineutrality"
        and not check.available
        and not check.passed
        for check in failure.checks
    )
    assert "failure_test_B" in failure.partial_arrays


def xp_tile(context, entities: int):
    xp = context.compute.xp
    m1 = xp.real(xp.asarray(context.op.M1)).astype(xp.float64)
    return xp.tile(m1, entities // c_entity_count(context))


def c_entity_count(context) -> int:
    return int(context.op.M1.size)
