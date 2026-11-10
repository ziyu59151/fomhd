"""Oracle comparisons for batched polygon boundary diagnostics."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from fomhd import core, massless_checks, multirate_wr_checks, polygon_wall
from fomhd.types import Failure

from test_polygon_projection import _polygon_case


class _AsarraySpy:
    def __init__(self, xp):
        self.xp = xp
        self.shapes = []

    def asarray(self, value, dtype=None):
        if hasattr(value, "shape"):
            shape = tuple(value.shape)
        elif hasattr(value, "__len__"):
            shape = (len(value),)
        else:
            shape = ()
        self.shapes.append(shape)
        return self.xp.asarray(value, dtype=dtype)

    def __getattr__(self, name):
        return getattr(self.xp, name)


def _fixture(nphi, seed=3100):
    base, _retained, _full = _polygon_case(8)
    context = copy.copy(base)
    context.N = nphi
    rng = np.random.default_rng(seed + nphi)
    nc = context.op.Nc
    nf = len(context.geometry.faces.keys)
    nv = len(context.geometry.vertices.keys)
    ne = context.op.Ne
    nb = context.op.Nb
    phase = 2.0 * np.pi * np.arange(nphi) / nphi
    ion_density = np.broadcast_to(
        1.0e12 * (1.0 + 0.1 * rng.random(nc))[None, :],
        (nphi, nc),
    ).copy()
    electron_density = 0.93 * ion_density
    ion_velocity = np.stack(
        (
            np.broadcast_to(
                (0.4 + 0.1 * rng.standard_normal(nc))[None, :],
                (nphi, nc),
            ),
            np.broadcast_to(
                (-0.3 + 0.1 * rng.standard_normal(nc))[None, :],
                (nphi, nc),
            ),
            np.broadcast_to(
                (0.2 + 0.1 * rng.standard_normal(nc))[None, :],
                (nphi, nc),
            ),
        ),
        axis=1,
    )
    electron_velocity = np.stack(
        (
            np.broadcast_to(
                (-0.25 + 0.05 * rng.standard_normal(nc))[None, :],
                (nphi, nc),
            ),
            np.broadcast_to(
                (0.35 + 0.05 * rng.standard_normal(nc))[None, :],
                (nphi, nc),
            ),
            np.broadcast_to(
                (0.15 + 0.05 * rng.standard_normal(nc))[None, :],
                (nphi, nc),
            ),
        ),
        axis=1,
    )
    ion_perp = 0.8 + 0.15 * np.cos(phase)[:, None] * np.ones((1, nc))
    electron_perp = 1.1 + 0.08 * np.sin(phase)[:, None] * np.ones((1, nc))
    electron_parallel = (
        0.45 + 0.09 * np.cos(phase)[:, None] * np.ones((1, nc))
    )
    densities = (ion_density, electron_density)
    velocities = (ion_velocity, electron_velocity)
    perpendicular = (ion_perp, electron_perp)
    parallel = (ion_perp, electron_parallel)
    masses = np.array(
        (
            context.p.species[0].mass,
            context.p.species[1].mass,
        )
    )
    momentum = tuple(
        masses[index] * densities[index][:, None, :] * velocities[index]
        for index in range(2)
    )
    face_scale = np.asarray(context.op.S)
    face_density = tuple(density @ face_scale.T for density in densities)
    face_velocity = []
    for species in range(2):
        momentum_trace = np.einsum(
            "nwc,fc->nfw",
            momentum[species],
            face_scale,
        )
        face_velocity.append(
            momentum_trace
            / (masses[species] * face_density[species][:, :, None])
        )
    face_normals = context.geometry.face_normals[
        [item.face for item in context.geometry.walls]
    ]
    normal3 = np.column_stack(
        (
            face_normals[:, 0],
            np.zeros(face_normals.shape[0]),
            face_normals[:, 1],
        )
    )
    flux = np.zeros((nphi, ne), dtype=np.float64)
    wall_faces = np.array(
        [item.face for item in context.geometry.walls],
        dtype=np.int64,
    )
    for wall, face in enumerate(wall_faces):
        flux[:, face] = (
            face_density[0][:, face]
            * np.sum(
                face_velocity[0][:, face, :] * normal3[wall],
                axis=1,
            )
        )
    magnetic = np.concatenate(
        (
            np.broadcast_to(
                (
                    2.6
                    + 0.05
                    * np.linspace(
                        -1.0,
                        1.0,
                        nv,
                        dtype=np.float64,
                    )
                )[None, :],
                (nphi, nv),
            ),
            np.full((nphi, nf), -0.7, dtype=np.float64),
        ),
        axis=1,
    )
    electric = np.concatenate(
        (
            np.full((nphi, nf), 4.0, dtype=np.float64),
            np.full((nphi, nc), 8.0, dtype=np.float64),
        ),
        axis=1,
    )
    reaction = rng.standard_normal((nphi, 3, nc)) * 0.2
    ion_rate = np.broadcast_to(
        (rng.standard_normal((1, 3, nc)) * 0.3),
        (nphi, 3, nc),
    ).copy()

    state_fluid = tuple(
        type("Fluid", (), {})()
        for _ in range(2)
    )
    for species in range(2):
        state_fluid[species].n = core.forward(
            densities[species].reshape(-1),
            nc,
            context,
        )
        state_fluid[species].mom = core.forward(
            momentum[species].reshape(-1),
            3 * nc,
            context,
        )
        state_fluid[species].perp = core.forward(
            perpendicular[species].reshape(-1),
            nc,
            context,
        )
        state_fluid[species].parallel = core.forward(
            parallel[species].reshape(-1),
            nc,
            context,
        )
    state = type("State", (), {})()
    state.fluid = state_fluid
    state.B = core.forward(magnetic.reshape(-1), nb, context)
    state.step = 2
    state.generation = 3
    state.time = 0.25

    stage_fluid = tuple(
        type("Fluid", (), {})()
        for _ in range(2)
    )
    for species in range(2):
        stage_fluid[species].n = densities[species].reshape(-1)
        stage_fluid[species].mom = momentum[species].reshape(-1)
        stage_fluid[species].perp = perpendicular[species].reshape(-1)
        stage_fluid[species].parallel = parallel[species].reshape(-1)
        stage_fluid[species].flux = flux.reshape(-1)
    stage = type("Stage", (), {})()
    stage.fluid = stage_fluid
    stage.cache = type("Cache", (), {})()
    stage.cache.E = core.forward(
        electric.reshape(-1),
        ne,
        context,
    )
    stage.tag = type("Tag", (), {})()
    for name, value in (
        ("step", state.step),
        ("generation", state.generation),
        ("trial", 7),
        ("time", state.time),
        ("stage", "batch_oracle"),
    ):
        setattr(stage.tag, name, value)
    fixture = type("Fixture", (), {})()
    fixture.context = context
    fixture.state = state
    fixture.stage = stage
    fixture.density = densities
    fixture.velocity = velocities
    fixture.perp = perpendicular
    fixture.parallel = parallel
    fixture.magnetic = magnetic
    fixture.electric = electric
    fixture.flux = flux
    fixture.reaction = reaction
    fixture.ion_rate = ion_rate
    return fixture


def _oracle_traces(fixture):
    context = fixture.context
    walls = context.geometry.walls
    nc = context.op.Nc
    nv = len(context.geometry.vertices.keys)
    traces = []
    for wall in walls:
        normal = np.array(
            (wall.normal_R, 0.0, wall.normal_Z),
            dtype=np.float64,
        )
        tangent = np.array(
            (wall.tangent_R, 0.0, wall.tangent_Z),
            dtype=np.float64,
        )
        s_row = np.asarray(context.op.S[wall.face])
        density = {
            species: fixture.density[species] @ s_row
            for species in range(2)
        }
        momentum_trace = {
            species: np.stack(
                tuple(
                    (
                        context.p.species[species].mass
                        * fixture.density[species][:, None, :]
                        * fixture.velocity[species]
                    )[:, component, :]
                    @ s_row
                    for component in range(3)
                ),
                axis=1,
            )
            for species in range(2)
        }
        velocity = {
            species: momentum_trace[species]
            / (
                context.p.species[species].mass
                * density[species][:, None]
            )
            for species in range(2)
        }
        perp = {
            species: fixture.perp[species] @ s_row
            for species in range(2)
        }
        parallel = {
            species: fixture.parallel[species] @ s_row
            for species in range(2)
        }
        bphi = 0.5 * (
            fixture.magnetic[:, wall.vertices[0]]
            + fixture.magnetic[:, wall.vertices[1]]
        )
        bt = fixture.magnetic[:, nv + wall.face]
        bwall = np.stack(
            (
                bt * tangent[0],
                bphi,
                bt * tangent[2],
            ),
            axis=1,
        )
        electric_normal = fixture.electric[:, wall.face]
        traces.append(
            type("Trace", (), {})()
        )
        trace = traces[-1]
        trace.wall = wall
        trace.normal3 = tuple(normal)
        trace.tangent3 = tuple(tangent)
        trace.radius = wall.radius
        trace.area = (
            wall.area_per_radian * 2.0 * math.pi / context.N
        )
        trace.B = tuple(bwall[:, component] for component in range(3))
        trace.electric_normal = electric_normal
        trace.electric_wall = (
            electric_normal[:, None] * normal[None, :]
        )
        trace.density = tuple(density[species] for species in range(2))
        trace.velocity = tuple(velocity[species] for species in range(2))
        trace.perp = tuple(perp[species] for species in range(2))
        trace.parallel = tuple(parallel[species] for species in range(2))
        trace.normal_flux = (
            wall.sign * fixture.flux[:, wall.face],
            wall.sign * fixture.flux[:, wall.face],
        )
    return traces


def _oracle_boundary(fixture, traces):
    context = fixture.context
    number = np.zeros(2)
    kinetic = np.zeros(2)
    internal_perp = np.zeros(2)
    internal_parallel = np.zeros(2)
    energy = np.zeros(2)
    for sample in range(context.N):
        for trace in traces:
            normal = np.asarray(trace.normal3)
            b = np.stack(trace.B, axis=1)
            for species in range(2):
                density = trace.density[species][sample]
                flux = trace.normal_flux[species][sample]
                un_flux = flux / density
                velocity = trace.velocity[species][sample]
                kinetic_density = (
                    0.5
                    * context.p.species[species].mass
                    * density
                    * np.dot(velocity, velocity)
                    if species == 0
                    else 0.0
                )
                perp = trace.perp[species][sample]
                parallel = trace.parallel[species][sample]
                number[species] += trace.area * flux
                kinetic[species] += (
                    trace.area * kinetic_density * un_flux
                )
                internal_perp[species] += (
                    trace.area
                    * perp
                    * un_flux
                    * (
                        1.0
                        if species
                        else 1.0 / (context.p.species[0].gamma - 1.0)
                    )
                )
                if species:
                    internal_parallel[species] += (
                        trace.area * 0.5 * parallel * un_flux
                    )
                internal_density = (
                    perp + 0.5 * parallel
                    if species
                    else perp / (context.p.species[0].gamma - 1.0)
                )
                pressure_work = perp * np.dot(velocity, normal)
                if species:
                    bmag = np.linalg.norm(b[sample])
                    bhat = b[sample] / bmag
                    pressure_work += (
                        (parallel - perp)
                        * np.dot(bhat, velocity)
                        * np.dot(bhat, normal)
                    )
                energy[species] += trace.area * (
                    (kinetic_density + internal_density) * un_flux
                    + pressure_work
                )
    return SimpleNamespace(
        number=number,
        kinetic=kinetic,
        internal_perp=internal_perp,
        internal_parallel=internal_parallel,
        energy=energy,
    )


def _oracle_traction(trace, sample, context):
    normal = np.asarray(trace.normal3)
    b = np.stack(trace.B, axis=1)[sample]
    velocity = trace.velocity[0][sample]
    density = trace.density[0][sample]
    bmag = np.linalg.norm(b)
    bhat = b / bmag
    un = np.dot(velocity, normal)
    transport = (
        context.p.species[0].mass
        * density
        * velocity
        * un
        + trace.perp[0][sample] * normal
    )
    maxwell = (
        0.5 * np.dot(b, b) * normal
        - np.dot(b, normal) * b
    ) / context.p.mu0
    electron = (
        trace.perp[1][sample] * normal
        + (trace.parallel[1][sample] - trace.perp[1][sample])
        * bhat
        * np.dot(bhat, normal)
    )
    return transport, maxwell + electron


def _oracle_wall_integrals(fixture, traces, role):
    context = fixture.context
    samples = max(context.N, 3)
    outward = np.zeros(3)
    torque = 0.0
    torque_scale = 0.0
    for q in range(samples):
        sample = 0 if context.N == 1 else q
        angle = 2.0 * math.pi * q / samples
        cosine = math.cos(angle)
        sine = math.sin(angle)
        for trace in traces:
            transport, electromagnetic = _oracle_traction(
                trace,
                sample,
                context,
            )
            components = (
                transport if role == "transport" else electromagnetic
            )
            wall_weight = (
                2.0
                * math.pi
                * trace.wall.area_per_radian
                / samples
            )
            cartesian = np.array(
                (
                    components[0] * cosine - components[1] * sine,
                    components[0] * sine + components[1] * cosine,
                    components[2],
                )
            )
            outward += wall_weight * cartesian
            torque += wall_weight * trace.radius * components[1]
            torque_scale += (
                wall_weight
                * trace.radius
                * np.linalg.norm(components)
            )
    return outward, torque, torque_scale


def _oracle_volume_integrals(fixture):
    context = fixture.context
    nc = context.op.Nc
    samples = max(context.N, 3)
    m0 = np.asarray(context.op.M0)
    cell_radius = np.asarray(context.geometry.cells.R)
    ion_momentum = fixture.density[0][:, None, :] * (
        context.p.species[0].mass * fixture.velocity[0]
    )
    value = np.zeros(3)
    pdot = np.zeros(3)
    reaction_xyz = np.zeros(3)
    angular = 0.0
    angular_rate = 0.0
    wall_torque = 0.0
    for q in range(samples):
        sample = 0 if context.N == 1 else q
        angle = 2.0 * math.pi * q / samples
        cosine = math.cos(angle)
        sine = math.sin(angle)
        weights = 2.0 * math.pi * m0 / samples
        for target, source in (
            (value, ion_momentum[sample]),
            (pdot, fixture.ion_rate[sample]),
            (reaction_xyz, fixture.reaction[sample]),
        ):
            cartesian = (
                source[0] * cosine - source[1] * sine,
                source[0] * sine + source[1] * cosine,
                source[2],
            )
            for component in range(3):
                target[component] += np.sum(
                    weights * cartesian[component]
                )
        angular += np.sum(
            weights * cell_radius * ion_momentum[sample, 1, :]
        )
        angular_rate += np.sum(
            weights * cell_radius * fixture.ion_rate[sample, 1, :]
        )
        wall_torque += np.sum(
            weights * cell_radius * fixture.reaction[sample, 1, :]
        )
    return SimpleNamespace(
        value=value,
        pdot=pdot,
        reaction=reaction_xyz,
        angular=angular,
        angular_rate=angular_rate,
        wall_torque=wall_torque,
    )


@pytest.mark.parametrize("nphi", [1, 5])
def test_batched_traces_match_all_scalar_oracle_channels(nphi):
    fixture = _fixture(nphi)
    expected = _oracle_traces(fixture)
    actual = polygon_wall.wall_trace_data(
        fixture.state,
        fixture.stage,
        fixture.context,
    )
    assert len(actual) == len(expected)
    assert actual.host is not None
    assert actual.host.packed.shape == (
        nphi,
        len(expected),
        18,
    )
    for wall, (got, want) in enumerate(zip(actual, expected)):
        np.testing.assert_allclose(
            np.stack(got.B, axis=1),
            np.stack(want.B, axis=1),
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        for name in (
            "electric_normal",
            "density",
            "velocity",
            "perp",
            "parallel",
            "normal_flux",
        ):
            got_values = getattr(got, name)
            want_values = getattr(want, name)
            if name == "electric_normal":
                got_values, want_values = (got_values,), (want_values,)
            for species_or_zero in range(len(want_values)):
                np.testing.assert_allclose(
                    got_values[species_or_zero],
                    want_values[species_or_zero],
                    rtol=2.0e-13,
                    atol=2.0e-13,
                )
        np.testing.assert_allclose(
            got.electric_wall,
            want.electric_wall,
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        assert got.normal3 == pytest.approx(want.normal3)
        assert got.tangent3 == pytest.approx(want.tangent3)
        assert got.radius == pytest.approx(want.radius)
        assert got.area == pytest.approx(want.area)
        assert got.wall is want.wall
        assert actual.host.wall_indices[wall] == wall


def test_view_building_reuses_one_backend_normal_array():
    fixture = _fixture(5, seed=3450)
    context = copy.copy(fixture.context)
    context.compute = copy.copy(context.compute)
    spy = _AsarraySpy(context.compute.xp)
    context.compute.xp = spy
    batch = polygon_wall.wall_trace_data(
        fixture.state,
        fixture.stage,
        context,
    )
    wall_count = len(context.geometry.walls)
    assert spy.shapes.count((wall_count, 3)) >= 2
    assert (3,) not in spy.shapes
    expected_normal = np.asarray(
        [wall_frame.normal3 for wall_frame in (
            polygon_wall.wall_frame(context, wall)
            for wall in context.geometry.walls
        )]
    )
    np.testing.assert_allclose(batch.host.normal3, expected_normal)
    for wall_index, data in enumerate(batch):
        expected_electric = (
            batch.host.electric_normal[:, wall_index, None]
            * expected_normal[wall_index][None, :]
        )
        np.testing.assert_allclose(
            np.asarray(data.electric_wall),
            expected_electric,
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize("nphi", [1, 5])
def test_batched_boundary_momentum_and_wr_integrals_match_scalar_oracle(nphi):
    fixture = _fixture(nphi, seed=3700)
    expected_traces = _oracle_traces(fixture)
    batch = polygon_wall.wall_trace_data(
        fixture.state,
        fixture.stage,
        fixture.context,
    )
    fixture.stage.boundary = type(
        "BoundarySource",
        (),
        {"_trace_data": batch},
    )()

    expected_boundary = _oracle_boundary(fixture, expected_traces)
    boundary = massless_checks._boundary(
        fixture.state,
        fixture.stage,
        fixture.context,
    )
    np.testing.assert_allclose(
        boundary.number,
        expected_boundary.number,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        boundary.kinetic,
        expected_boundary.kinetic,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        boundary.internal_perp,
        expected_boundary.internal_perp,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        boundary.internal_parallel,
        expected_boundary.internal_parallel,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert boundary.total_energy == pytest.approx(
        float(np.sum(expected_boundary.energy)),
        rel=2.0e-12,
        abs=2.0e-12,
    )

    wr = multirate_wr_checks._boundary_summands(
        fixture.state,
        fixture.stage,
        fixture.context,
    )
    np.testing.assert_allclose(wr.number, boundary.number, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(wr.kinetic, boundary.kinetic, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        wr.internal_perp,
        boundary.internal_perp,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        wr.internal_parallel,
        boundary.internal_parallel,
        rtol=0.0,
        atol=0.0,
    )

    expected_volume = _oracle_volume_integrals(fixture)
    fixture.stage.fluid[0].mom = (
        fixture.density[0][:, None, :]
        * fixture.context.p.species[0].mass
        * fixture.velocity[0]
    ).reshape(-1)
    fixture.stage.fluid[1].mom = np.zeros_like(fixture.stage.fluid[0].mom)
    result = type("Result", (), {})()
    result.fluid = fixture.stage.fluid
    result.terms = [
        type(
            "Terms",
            (),
            {"reaction": fixture.reaction},
        )(),
        type("Terms", (), {})(),
    ]
    result.rhs = type("Rhs", (), {})()
    result.rhs.fluid = [
        type(
            "Rate",
            (),
            {
                "mom": core.forward(
                    fixture.ion_rate.reshape(-1),
                    3 * fixture.context.op.Nc,
                    fixture.context,
                )
            },
        )(),
        type("Rate", (), {"mom": np.zeros(fixture.context.N * 3 * fixture.context.op.Nc, dtype=np.complex128)})(),
    ]
    result.boundary = fixture.stage.boundary
    momentum = massless_checks._polygon_momentum(
        fixture.state,
        result,
        fixture.context,
    )
    np.testing.assert_allclose(
        momentum.value,
        expected_volume.value,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        momentum.pdot,
        expected_volume.pdot,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        momentum.wall_reaction,
        expected_volume.reaction,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert momentum.angular == pytest.approx(
        expected_volume.angular,
        rel=2.0e-12,
        abs=2.0e-12,
    )
    assert momentum.angular_rate == pytest.approx(
        expected_volume.angular_rate,
        rel=2.0e-12,
        abs=2.0e-12,
    )
    assert momentum.wall_torque == pytest.approx(
        expected_volume.wall_torque,
        rel=2.0e-12,
        abs=2.0e-12,
    )
    for role in ("transport", "electromagnetic"):
        outward, torque, torque_scale = _oracle_wall_integrals(
            fixture,
            expected_traces,
            role,
        )
        wr_integrals = multirate_wr_checks._momentum_integrals(
            fixture.state,
            fixture.stage,
            fixture.reaction.reshape(-1),
            fixture.context,
            traction_role=role,
        )
        np.testing.assert_allclose(
            wr_integrals[0],
            outward,
            rtol=2.0e-12,
            atol=2.0e-12 * max(float(np.max(np.abs(outward))), 1.0),
        )
        assert wr_integrals[1] == pytest.approx(
            torque,
            rel=2.0e-12,
            abs=2.0e-12 * max(torque_scale, 1.0),
        )
    total_expected = (
        _oracle_wall_integrals(
            fixture,
            expected_traces,
            "transport",
        )[0]
        + _oracle_wall_integrals(
            fixture,
            expected_traces,
            "electromagnetic",
        )[0]
    )
    np.testing.assert_allclose(
        momentum.outward,
        total_expected,
        rtol=2.0e-12,
        atol=2.0e-12 * max(float(np.max(np.abs(total_expected))), 1.0),
    )
    expected_outward_torque = sum(
        _oracle_wall_integrals(
            fixture,
            expected_traces,
            role,
        )[1]
        for role in ("transport", "electromagnetic")
    )
    expected_outward_torque_scale = sum(
        _oracle_wall_integrals(
            fixture,
            expected_traces,
            role,
        )[2]
        for role in ("transport", "electromagnetic")
    )
    assert momentum.outward_torque == pytest.approx(
        expected_outward_torque,
        rel=2.0e-12,
        abs=2.0e-12 * max(expected_outward_torque_scale, 1.0),
    )
    assert momentum.angular_balance_defect == pytest.approx(
        expected_volume.angular_rate + expected_outward_torque,
        rel=2.0e-12,
        abs=2.0e-12
        * max(
            abs(expected_volume.angular_rate),
            expected_outward_torque_scale,
            1.0,
        ),
    )
    np.testing.assert_allclose(
        momentum.balance_defect,
        expected_volume.pdot + total_expected,
        rtol=2.0e-12,
        atol=2.0e-12
        * max(
            float(np.max(np.abs(expected_volume.pdot))),
            float(np.max(np.abs(total_expected))),
            1.0,
        ),
    )


def test_wall_trace_batch_exports_one_small_packed_array(monkeypatch):
    fixture = _fixture(5, seed=4200)
    context = fixture.context
    calls = []
    original = context.compute.to_host

    def counted(value):
        calls.append(
            (
                tuple(value.shape),
                value.dtype,
                int(value.nbytes),
            )
        )
        return original(value)

    monkeypatch.setattr(context.compute, "to_host", counted)
    batch = polygon_wall.wall_trace_data(
        fixture.state,
        fixture.stage,
        context,
    )
    assert len(calls) == 1
    shape, dtype, nbytes = calls[0]
    assert shape == (5, len(context.geometry.walls), 18)
    assert dtype == np.dtype(np.float64)
    assert nbytes == 8 * 5 * len(context.geometry.walls) * 18
    assert batch.host.packed.shape == shape
    assert batch.host.placement == (
        "explicit read-only wall observation host export"
    )
    assert 18 < 3 * context.op.Nc
    fixture.stage.boundary = type(
        "BoundarySource",
        (),
        {"_trace_data": batch},
    )()
    massless_checks._boundary(
        fixture.state,
        fixture.stage,
        context,
    )
    multirate_wr_checks._boundary_summands(
        fixture.state,
        fixture.stage,
        context,
    )
    assert len(calls) == 1
    fixture.state.B = fixture.state.B.copy()
    polygon_wall.wall_trace_data(
        fixture.state,
        fixture.stage,
        context,
    )
    assert len(calls) == 2


def test_positive_density_batch_rejects_before_division_and_export(monkeypatch):
    fixture = _fixture(5, seed=4300)
    context = fixture.context
    fixture.density[0][0, 0] = 0.0
    fixture.stage.fluid[0].n = fixture.density[0].reshape(-1)
    state_n = core.forward(
        fixture.density[0].reshape(-1),
        context.op.Nc,
        context,
    )
    fixture.state.fluid[0].n = state_n
    calls = 0

    def counted(value):
        nonlocal calls
        calls += 1
        return np.asarray(value)

    monkeypatch.setattr(context.compute, "to_host", counted)
    with pytest.raises(Failure, match="density"):
        polygon_wall.wall_trace_data(
            fixture.state,
            fixture.stage,
            context,
        )
    assert calls == 0


@pytest.mark.parametrize("species_index", [0, 1])
@pytest.mark.parametrize(
    "invalid_mass",
    [0.0, -1.0, math.nan, math.inf],
    ids=["zero", "negative", "nan", "infinity"],
)
def test_species_mass_validation_precedes_velocity_division_and_export(
    monkeypatch,
    species_index,
    invalid_mass,
):
    fixture = _fixture(5, seed=4400 + species_index)
    context = copy.copy(fixture.context)
    context.p = copy.copy(context.p)
    species = list(context.p.species)
    species[species_index] = copy.copy(species[species_index])
    species[species_index].mass = invalid_mass
    context.p.species = tuple(species)
    calls = 0

    def counted(value):
        nonlocal calls
        calls += 1
        return np.asarray(value)

    monkeypatch.setattr(context.compute, "to_host", counted)
    with pytest.raises(Failure, match="wall species mass must be positive") as error:
        polygon_wall.wall_trace_data(
            fixture.state,
            fixture.stage,
            context,
        )
    assert error.value.code == "WallTrace"
    assert calls == 0
