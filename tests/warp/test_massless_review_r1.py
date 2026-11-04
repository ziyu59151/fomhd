"""Regression tests for formal review round 1."""

from __future__ import annotations

import numpy as np

import fomhd.massless as massless
import fomhd.stage as finite_stage
from fomhd.core import forward, inverse, sample_apply
from fomhd.fluid import _sampled_divergence
from fomhd.massless_checks import _stage_residuals, _wall_flux_samples

from test_massless_support import context, stage_tag, uniform_state


def _nonaxisymmetric_density_state():
    c = context(K=1, hard_diagnostics=False, nodes_r=6, nodes_z=6)
    state = uniform_state(c)
    nc = c.op.Nc
    state.fluid[0].n[:nc] = 0.1 * c.p.ref.ni
    state.fluid[0].n[-nc:] = 0.1 * c.p.ref.ni
    density = state.fluid[0].n.reshape(3, nc)
    state.fluid[0].mom.reshape(3, 3, nc)[:, 1, :] = (
        c.p.species[0].mass * density * 100.0
    )
    return c, state


def _cartesian_ion_momentum(stage, context):
    phi = 2.0 * np.pi * np.arange(context.N) / context.N
    momentum = np.asarray(stage.fluid[0].mom).reshape(
        context.N, 3, context.op.Nc
    )
    weights = np.real(context.op.M0)[None, :] * 2.0 * np.pi / context.N
    components = (
        momentum[:, 0, :] * np.cos(phi[:, None])
        - momentum[:, 1, :] * np.sin(phi[:, None]),
        momentum[:, 0, :] * np.sin(phi[:, None])
        + momentum[:, 1, :] * np.cos(phi[:, None]),
        momentum[:, 2, :],
    )
    return np.array(
        [float(np.real(np.sum(value * weights))) for value in components]
    )


def _record(stage, name, population=""):
    return next(
        record
        for record in stage.checks
        if record.name == name
        and record.population == population
        and record.region == "all"
        and str(record.norm) == "value"
    )


def test_nonaxisymmetric_density_velocity_and_theta_use_shaped_samples():
    c, state = _nonaxisymmetric_density_state()
    derived = massless._derive_electron(state, c)
    nc = c.op.Nc
    h_flux = sample_apply(
        c.op.H, derived.electron_flux, c.N, c.compute
    )
    projected = massless._constraint_project(
        h_flux, derived.electron_density, c
    ).reshape(c.N, 3, nc)
    expected_velocity = (
        projected / derived.electron_density.reshape(c.N, 1, nc)
    ).reshape(-1)
    velocity_error = np.linalg.norm(
        derived.electron_velocity - expected_velocity
    ) / np.linalg.norm(expected_velocity)
    assert velocity_error <= 2.0e-12

    stage = massless.evaluate_stage(state, stage_tag(state), c)
    electron = stage.fluid[1]
    expected_theta = _sampled_divergence(
        electron.flux.reshape(c.N, c.op.Ne)
        / electron.nf.reshape(c.N, c.op.Ne),
        c,
    )
    theta_error = np.linalg.norm(
        electron.theta_consistency - expected_theta
    ) / np.linalg.norm(expected_theta)
    assert theta_error <= 2.0e-12
    assert np.max(np.abs(electron.theta_defect)) > 0.0
    magnetic = stage.ledger[1].magnetic
    magnetic_scale = stage.ledger[1].magnetic_scale
    assert abs(magnetic) <= 2.0e-10 * max(magnetic_scale, 1.0e-300)
    assert _record(stage, "electron_magnetic_work", "electron").passed


def test_cartesian_momentum_and_actual_stage_momentum_defect():
    c, state = _nonaxisymmetric_density_state()
    stage = massless.evaluate_stage(state, stage_tag(state), c)
    expected = _cartesian_ion_momentum(stage, c)
    actual = np.asarray(stage.momentum.value)
    scale = max(float(np.max(np.abs(expected))), 1.0e-300)
    assert np.max(np.abs(actual - expected)) <= 2.0e-10 * scale
    assert np.max(np.abs(stage.momentum.outward)) > 0.0

    pdot_samples = inverse(
        stage.rhs.fluid[0].mom, 3 * c.op.Nc, c
    ).reshape(c.N, 3, c.op.Nc)
    phi = 2.0 * np.pi * np.arange(c.N) / c.N
    weights = np.real(c.op.M0)[None, :] * 2.0 * np.pi / c.N
    pdot = np.array(
        [
            np.real(
                np.sum(
                    (
                        pdot_samples[:, 0, :] * np.cos(phi[:, None])
                        - pdot_samples[:, 1, :] * np.sin(phi[:, None])
                    )
                    * weights
                )
            ),
            np.real(
                np.sum(
                    (
                        pdot_samples[:, 0, :] * np.sin(phi[:, None])
                        + pdot_samples[:, 1, :] * np.cos(phi[:, None])
                    )
                    * weights
                )
            ),
            np.real(np.sum(pdot_samples[:, 2, :] * weights)),
        ]
    )
    resolved = pdot + np.asarray(stage.momentum.outward)
    defect = np.asarray(stage.momentum.balance_defect)
    assert np.max(np.abs(defect - resolved)) <= 2.0e-10 * max(
        float(np.max(np.abs(resolved))), 1.0e-300
    )

    expected_outward = np.zeros(3)
    samples = max(c.N, 3)
    wall_count = len(c.geometry.walls)
    for q in range(samples):
        sample = 0 if c.N == 1 else q
        angle = 2.0 * np.pi * q / samples
        for wall_index, wall in enumerate(c.geometry.walls):
            trace = stage.boundary.traces[q * wall_count + wall_index]
            direction = 0 if wall.component == "r_face" else 2
            normal = wall.sign
            b = np.asarray(trace.B)
            b2 = float(np.dot(b, b))
            bhat = b / np.sqrt(b2)
            velocity = np.asarray(trace.velocity[0])
            un = normal * velocity[direction]
            stress = np.zeros((3, 3))
            for component in range(3):
                unit = 1.0 if component == direction else 0.0
                stress[component, component] += (
                    trace.perp[0]
                    + trace.perp[1]
                    + 0.5 * b2 / c.p.mu0
                )
                for other in range(3):
                    stress[component, other] -= (
                        b[component] * b[other] / c.p.mu0
                        + (
                            trace.parallel[1] - trace.perp[1]
                        )
                        * bhat[component]
                        * bhat[other]
                    )
                    stress[component, other] += (
                        c.p.species[0].mass
                        * trace.density[0]
                        * velocity[component]
                        * velocity[other]
                    )
            normal_vector = np.zeros(3)
            normal_vector[direction] = normal
            traction = stress @ normal_vector
            cartesian = np.array(
                [
                    traction[0] * np.cos(angle)
                    - traction[1] * np.sin(angle),
                    traction[0] * np.sin(angle)
                    + traction[1] * np.cos(angle),
                    traction[2],
                ]
            )
            expected_outward += (
                2.0 * np.pi * wall.area_per_radian / samples * cartesian
            )
    assert np.max(np.abs(np.asarray(stage.momentum.outward) - expected_outward)) <= (
        2.0e-10 * max(float(np.max(np.abs(expected_outward))), 1.0e-300)
    )


def test_boundary_uses_actual_wall_traces_and_omits_electron_kinetic():
    c, state = _nonaxisymmetric_density_state()
    nc = c.op.Nc
    raw = (
        np.random.default_rng(63).normal(size=c.N * 3 * nc)
        * c.p.species[0].mass
        * c.p.ref.ni
        * 1.0e5
    )
    density = inverse(state.fluid[0].n, nc, c)
    state.fluid[0].mom[:] = forward(
        massless._constraint_project(raw, density, c),
        3 * nc,
        c,
    )
    state.fluid[1].perp.reshape(3, nc)[1] *= np.linspace(0.8, 1.2, nc)
    state.fluid[1].parallel.reshape(3, nc)[1] *= np.linspace(0.6, 1.2, nc)
    stage = massless.evaluate_stage(state, stage_tag(state), c)
    assert stage.boundary.kinetic[1] == 0.0
    assert (
        stage.boundary.applicability.electron_kinetic_transport
        == "not_applicable"
    )

    relative_wall_flux = float(
        np.max(np.abs(_wall_flux_samples(stage.fluid[0].flux, c)))
        / np.max(np.abs(stage.fluid[0].flux))
    )
    max_normal_velocity = max(
        float(np.max(np.abs(data.velocity[0][:, data.direction])))
        for data in stage.boundary._trace_data
    )
    max_speed = max(
        float(np.max(np.linalg.norm(data.velocity[0], axis=1)))
        for data in stage.boundary._trace_data
    )
    max_density = max(
        float(np.max(data.density[0]))
        for data in stage.boundary._trace_data
    )
    surface_area = sum(
        data.area for data in stage.boundary._trace_data
    )
    kinetic_bound = (
        2.0
        * 0.5
        * c.p.species[0].mass
        * max_density
        * max_speed**2
        * max_normal_velocity
        * surface_area
    )
    assert abs(stage.boundary.kinetic[0]) <= kinetic_bound, relative_wall_flux
    assert any(np.max(np.abs(trace.B)) > 0.0 for trace in stage.boundary.traces)

    wall = c.geometry.walls[0]
    nr = len(c.geometry.r_faces.keys)
    edge = (
        wall.face
        if wall.component == "r_face"
        else nr + nc + wall.face
    )
    direction = 0 if wall.component == "r_face" else 2
    row = np.asarray(c.op.I[edge]).reshape(3, nc)[direction]
    expected_density = float(
        np.real(np.asarray(stage.fluid[0].n).reshape(c.N, nc)[0] @ row)
    )
    assert stage.boundary.traces[0].density[0] == expected_density
    assert (
        stage.boundary.traces[len(c.geometry.walls)].density[0]
        != stage.boundary.traces[0].density[0]
    )
    expected_ion_internal = 0.0
    expected_parallel_internal = 0.0
    for data in stage.boundary._trace_data:
        ion_pressure_flux = data.wall.sign * np.real(
            np.asarray(stage.fluid[0].pressure_flux).reshape(c.N, c.op.Ne)[
                :, data.edge
            ]
        )
        electron_parallel_flux = data.wall.sign * np.real(
            np.asarray(stage.fluid[1].parallel_flux).reshape(c.N, c.op.Ne)[
                :, data.edge
            ]
        )
        expected_ion_internal += (
            np.sum(data.area * ion_pressure_flux)
            / (c.p.species[0].gamma - 1.0)
        )
        expected_parallel_internal += np.sum(
            0.5 * data.area * electron_parallel_flux
        )
    assert abs(stage.boundary.internal_perp[0] - expected_ion_internal) <= (
        2.0e-12 * max(abs(expected_ion_internal), 1.0e-300)
    )
    assert abs(
        stage.boundary.internal_parallel[1] - expected_parallel_internal
    ) <= 2.0e-12 * max(abs(expected_parallel_internal), 1.0e-300)


def test_wall_trace_uses_normal_row_for_all_velocity_components():
    c, state = _nonaxisymmetric_density_state()
    stage = massless.evaluate_stage(state, stage_tag(state), c)
    nc = c.op.Nc
    wall = c.geometry.walls[0]
    nr = len(c.geometry.r_faces.keys)
    edge = (
        wall.face
        if wall.component == "r_face"
        else nr + nc + wall.face
    )
    direction = 0 if wall.component == "r_face" else 2
    normal_row = np.real(
        np.asarray(c.op.I[edge]).reshape(3, nc)[direction]
    )
    density_trace = (
        np.real(np.asarray(stage.fluid[0].n).reshape(c.N, nc))
        @ normal_row
    )
    momentum_phi = np.asarray(stage.fluid[0].mom).reshape(
        c.N, 3, nc
    )[:, 1, :]
    expected_phi = (
        np.real(momentum_phi @ normal_row)
        / (c.p.species[0].mass * density_trace)
    )
    trace_phi = np.array(
        [
            stage.boundary.traces[q * len(c.geometry.walls)].velocity[0][1]
            for q in range(c.N)
        ]
    )
    assert np.max(np.abs(trace_phi)) > 1.0
    assert np.max(np.abs(trace_phi - expected_phi)) <= (
        2.0e-12 * max(float(np.max(np.abs(expected_phi))), 1.0e-300)
    )


def test_ohm_uses_stored_reaction_and_detects_actual_field_mutation():
    c = context(hard_diagnostics=False)
    state = uniform_state(c)
    stage = massless.evaluate_stage(state, stage_tag(state), c)
    assert hasattr(stage.cache, "wall_reaction")
    for name in (
        "wall_field_reaction_work_ion",
        "wall_field_reaction_work_electron",
        "wall_field_reaction_work_current",
    ):
        assert _record(stage, name).available
    mutated = stage.cache.E + 0.123
    original = stage.cache.E
    stage.cache.E = mutated
    residual = _stage_residuals(state, stage, c)
    stage.cache.E = original
    assert np.max(np.abs(residual["drift_ohm_residual"])) > 0.12
    mutated_reaction = stage.cache.wall_reaction + 0.01
    original_reaction = stage.cache.wall_reaction
    stage.cache.wall_reaction = mutated_reaction
    residual = _stage_residuals(state, stage, c)
    stage.cache.wall_reaction = original_reaction
    assert np.max(np.abs(residual["drift_ohm_residual"])) > 0.009


def test_full_electric_power_not_pressure_only_field():
    c = context(hard_diagnostics=False, nodes_r=6, nodes_z=6)
    state = uniform_state(c)
    nc = c.op.Nc
    raw = (
        np.random.default_rng(63).normal(size=3 * nc)
        * c.p.species[0].mass
        * c.p.ref.ni
        * 1.0e5
    )
    state.fluid[0].mom[:] = massless._constraint_project(
        raw, state.fluid[0].n, c
    )
    state.fluid[1].perp[:] *= np.linspace(0.8, 1.2, nc)
    state.fluid[1].parallel[:] *= np.linspace(0.6, 1.2, nc)
    normal = massless.evaluate_stage(state, stage_tag(state), c)
    assert _record(
        normal, "electron_electric_pairing", "electron"
    ).passed
    original = massless._massless_e0
    try:
        massless._massless_e0 = lambda fluid, derived, ctx: (
            2.0 * original(fluid, derived, ctx)
        )
        stage = massless.evaluate_stage(state, stage_tag(state), c)
    finally:
        massless._massless_e0 = original

    record = _record(stage, "electron_electric_pairing", "electron")
    assert not record.passed
    assert abs(record.raw) > 1.0e-6
    ledger = stage.ledger[1]
    assert ledger.pressure_internal - ledger.electric_edge == (
        stage.residuals["electron_electric_pairing"][0].real
    )


def test_no_electron_kinetic_or_spectral_kinetic_channel():
    c = context(hard_diagnostics=False, nodes_r=6, nodes_z=6)
    state = uniform_state(c)
    nc = c.op.Nc
    raw = (
        np.random.default_rng(63).normal(size=3 * nc)
        * c.p.species[0].mass
        * c.p.ref.ni
        * 1.0e5
    )
    state.fluid[0].mom[:] = massless._constraint_project(
        raw, state.fluid[0].n, c
    )
    stage = massless.evaluate_stage(state, stage_tag(state), c)
    ledger = stage.ledger[1]
    assert ledger.spectral_kinetic == 0.0
    assert ledger.spectral_perp == 0.0
    assert ledger.spectral_parallel == 0.0
    assert (
        ledger.applicability.electron_kinetic_energy
        == "not_applicable"
    )
    assert (
        ledger.applicability.spectral_kinetic_transfer
        == "not_applicable"
    )
    assert hasattr(ledger, "node_edge_reconstruction_difference")
    expected_pressure_internal = float(
        np.real(
            np.sum(
                np.real(c.op.M0)[None, :]
                * (
                    inverse(stage.rhs.fluid[1].perp, nc, c).reshape(
                        c.N, nc
                    )
                    + 0.5
                    * inverse(stage.rhs.fluid[1].parallel, nc, c).reshape(
                        c.N, nc
                    )
                )
                * (2.0 * np.pi / c.N)
            )
        )
    )
    pressure_scale = float(
        np.sum(
            np.real(c.op.M0)[None, :]
            * (
                np.abs(np.real(inverse(stage.rhs.fluid[1].perp, nc, c).reshape(c.N, nc)))
                + 0.5
                * np.abs(
                    np.real(
                        inverse(stage.rhs.fluid[1].parallel, nc, c).reshape(
                            c.N, nc
                        )
                    )
                )
            )
            * (2.0 * np.pi / c.N)
        )
    )
    assert abs(ledger.pressure_internal - expected_pressure_internal) <= (
        8192.0 * np.finfo(np.float64).eps * pressure_scale
    )
    assert ledger.pressure_truncation == (
        ledger.pressure_internal - ledger.pressure_internal_full
    )
    assert ledger.node_edge_reconstruction_difference == (
        ledger.node_electric_work - ledger.electric_edge
    )


def test_failure_evidence_does_not_redispatch_through_stage(monkeypatch):
    from fomhd.types import Failure

    c = context(hard_diagnostics=False)
    state = uniform_state(c)
    tag = stage_tag(state)
    failure = Failure("Injected", "round-1 recursion mutation")

    def recurse(*args, **kwargs):
        raise AssertionError("massless failure evidence redispatched")

    monkeypatch.setattr(finite_stage, "retain_failure_evidence", recurse)
    massless.retain_failure_evidence(
        failure, tag, c, state, None, "r1"
    )
    assert failure.stage == tag
    assert "r1_B" in failure.partial_arrays
    assert any(check.name == "quasineutrality" for check in failure.checks)


def test_scratch_public_stage_dispatch_reaches_massless_branch():
    from fomhd.stage import evaluate_stage

    c = context(hard_diagnostics=False)
    state = uniform_state(c)
    result = evaluate_stage(state, stage_tag(state), c)
    assert result.model == massless.MODEL_NAME
    assert not hasattr(result, "AE") or result.AE is None
