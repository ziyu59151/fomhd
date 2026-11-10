"""Independent CPU checks for actual-normal polygon wall handling."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from fomhd import (
    core,
    massless,
    massless_checks,
    massless_field,
    multirate_wr_checks,
    polygon_wall,
    polygon_projection,
    wall_cache,
)
from fomhd.backend import Backend
from fomhd.types import Failure
from fomhd.wall import project_wall


POLYGON = "polygon_dual_v1"
STAIRCASE = "staircase_v1"


def _table(radius, keys, z=None):
    radius = np.asarray(radius, dtype=np.float64)
    return SimpleNamespace(
        keys=tuple(keys),
        R=radius,
        Z=(
            np.asarray(z, dtype=np.float64)
            if z is not None
            else np.zeros_like(radius)
        ),
        boundary=np.zeros(radius.size, dtype=np.uint8),
    )


def _context(kind=POLYGON):
    backend = Backend("cpu")
    n = 9
    nc = 2
    nf = 1
    ne = nc + nf
    nv = 2
    nb = nv + nf
    normal = np.array((0.6, 0.8), dtype=np.float64)
    tangent = np.array((-normal[1], normal[0]), dtype=np.float64)
    face_scale = np.array(((0.25, 0.75),), dtype=np.float64)
    radius = 2.4
    ell = 1.25
    vertices = np.array(
        (
            (
                2.2 - 0.625 * tangent[0],
                0.55 - 0.625 * tangent[1],
            ),
            (
                2.2 + 0.625 * tangent[0],
                0.55 + 0.625 * tangent[1],
            ),
        )
    )
    wall = SimpleNamespace(
        component="plane_face",
        face=0,
        sign=1,
        normal_R=float(normal[0]),
        normal_Z=float(normal[1]),
        tangent_R=float(tangent[0]),
        tangent_Z=float(tangent[1]),
        radius=radius,
        area_per_radian=radius * ell,
        vertices=(0, 1),
        near_cell=0,
        next_cell=1,
    )
    faces = _table(
        (radius,),
        (("face-center", 0),),
        (0.55,),
    )
    r_faces = faces if kind == POLYGON else _table(
        (2.3, 1.8),
        (("r", 0), ("r", 1)),
        (10.0, 20.0),
    )
    z_faces = _table((), (), ())
    if kind != POLYGON:
        r_faces.R[:] = (2.3, 1.8)
        r_faces.Z[:] = (10.0, 20.0)
    geometry = SimpleNamespace(
        kind=kind,
        faces=faces,
        r_faces=r_faces,
        z_faces=z_faces,
        vertices=_table(
            (vertices[0, 0], vertices[1, 0]),
            (("v", 0), ("v", 1)),
            (vertices[0, 1], vertices[1, 1]),
        ),
        face_normals=normal[None, :].copy(),
        face_tangents=tangent[None, :].copy(),
        face_vertices=np.array(((0, 1),), dtype=np.int64),
        face_lengths=np.array((ell,), dtype=np.float64),
        face_distances=np.array((0.37,), dtype=np.float64),
        face_cells=np.array(((0, -1),), dtype=np.int64),
        walls=[wall],
        cells=SimpleNamespace(
            keys=(("cell", 0), ("cell", 1)),
            R=np.array((2.1, 2.7), dtype=np.float64),
            Z=np.array((0.5, 0.6), dtype=np.float64),
            boundary=np.ones(nc, dtype=np.uint8),
        ),
    )
    interpolation = np.array(((0.25, 0.75),), dtype=np.float64)
    vector_interpolation = np.zeros((3 * nc, ne), dtype=np.complex128)
    vector_interpolation[0:nc, 0] = normal[0] * interpolation[0]
    vector_interpolation[nc : 2 * nc, nc] = 1.0
    vector_interpolation[2 * nc : 3 * nc, 0] = normal[1] * interpolation[0]
    wall_normal_rows = np.concatenate(
        (
            normal[0] * interpolation[0],
            np.zeros(nc, dtype=np.float64),
            normal[1] * interpolation[0],
        )
    )[None, :].astype(np.complex128)
    op = SimpleNamespace(
        Nc=nc,
        Ne=ne,
        Nb=nb,
        S=interpolation,
        I=vector_interpolation,
        T_full=wall_normal_rows,
        T=wall_normal_rows.copy(),
        M0=np.array((0.8, 1.2), dtype=np.float64),
        M1=np.array((2.0, 3.0, 1.7), dtype=np.float64),
        MB=np.array((1.1, 1.3, 0.9), dtype=np.float64),
    )
    species = (
        SimpleNamespace(
            id="ion",
            mass=2.0,
            charge=3.0,
            gamma=5.0 / 3.0,
            n_min=0.0,
            face_min=0.0,
            p_min=0.0,
            parallel_min=0.0,
        ),
        SimpleNamespace(
            id="electron",
            mass=0.5,
            charge=-1.0,
            gamma=0.0,
            n_min=0.0,
            face_min=0.0,
            p_min=0.0,
            parallel_min=0.0,
        ),
    )
    context = SimpleNamespace(
        N=n,
        compute=backend,
        geometry=geometry,
        op=op,
        p=SimpleNamespace(
            K=2,
            species=species,
            mu0=1.5,
            rank_rtol=1.0e-12,
            ref=SimpleNamespace(electric=1.0, length=1.0),
        ),
        spectral_id=0,
        next_source=1,
        next_trial=1,
        next_generation=1,
    )
    return context, wall, normal, tangent


def _trace_fixture(kind=POLYGON):
    context, wall, normal, tangent = _context(kind)
    backend = context.compute
    n = context.N
    nc = context.op.Nc
    nf = len(context.geometry.faces.keys)
    ne = context.op.Ne
    nb = context.op.Nb
    phase = 2.0 * np.pi * np.arange(n) / n

    ion_density = (2.0 + 0.2 * np.cos(phase))[:, None] * np.array(
        ((1.1, 0.9),)
    )
    electron_density = 0.75 * ion_density
    ion_velocity = np.stack(
        (
            np.stack(
                (
                    0.9 + 0.1 * np.sin(phase),
                    0.3 + 0.05 * np.cos(phase),
                ),
                axis=1,
            ),
            np.broadcast_to(
                (-0.65 + 0.08 * np.cos(phase))[:, None],
                (n, nc),
            ),
            np.stack(
                (
                    0.2 + 0.1 * np.cos(phase),
                    -0.15 + 0.05 * np.sin(phase),
                ),
                axis=1,
            ),
        ),
        axis=1,
    )
    electron_velocity = np.stack(
        (
            np.broadcast_to(
                (0.35 + 0.03 * np.sin(phase))[:, None],
                (n, nc),
            ),
            np.broadcast_to(
                (0.45 + 0.04 * np.cos(phase))[:, None],
                (n, nc),
            ),
            np.broadcast_to(
                (-0.2 + 0.02 * np.sin(phase))[:, None],
                (n, nc),
            ),
        ),
        axis=1,
    )
    densities = (ion_density, electron_density)
    velocities = (ion_velocity, electron_velocity)
    pressures_perp = (
        0.7 + 0.05 * np.cos(phase)[:, None] * np.ones((1, nc)),
        1.1 + 0.08 * np.sin(phase)[:, None] * np.ones((1, nc)),
    )
    pressures_parallel = (
        pressures_perp[0],
        0.4 + 0.06 * np.cos(phase)[:, None] * np.ones((1, nc)),
    )
    mass = np.array((context.p.species[0].mass, context.p.species[1].mass))
    momentum = tuple(
        mass[index] * densities[index][:, None, :] * velocities[index]
        for index in range(2)
    )
    face_density = tuple(density @ context.op.S[0] for density in densities)
    traced_momentum = tuple(
        np.stack(
            tuple(
                momentum[index][:, component, :] @ context.op.S[0]
                for component in range(3)
            ),
            axis=1,
        )
        for index in range(2)
    )
    face_velocity = tuple(
        traced_momentum[index]
        / (mass[index] * face_density[index][:, None])
        for index in range(2)
    )
    normal3 = np.array((normal[0], 0.0, normal[1]))
    ion_flux = np.zeros((n, ne), dtype=np.float64)
    ion_flux[:, 0] = face_density[0] * (
        face_velocity[0] @ normal3
    )
    electron_flux = np.zeros((n, ne), dtype=np.float64)
    electron_flux[:, 0] = 0.4 * face_density[1] * (
        face_velocity[1] @ normal3
    )
    fluxes = (ion_flux, electron_flux)
    bphi0 = 3.0 + 0.2 * np.cos(phase)
    bphi1 = 3.4 + 0.1 * np.sin(phase)
    bt = -0.8 + 0.07 * np.cos(phase)
    magnetic = np.stack((bphi0, bphi1, bt), axis=1)
    electric_normal = 5.0 + 0.3 * np.sin(phase)
    electric_phi = 11.0 + np.cos(phase)
    electric = np.stack(
        (
            electric_normal,
            np.full(n, 2.0),
            electric_phi,
        ),
        axis=1,
    )

    fluid = []
    state_fluid = []
    for index in range(2):
        fluid.append(
            SimpleNamespace(
                n=densities[index].reshape(-1),
                mom=np.zeros(n * 3 * nc, dtype=np.complex128),
                u=velocities[index].reshape(-1),
                perp=pressures_perp[index].reshape(-1),
                parallel=pressures_parallel[index].reshape(-1),
                flux=fluxes[index].reshape(-1),
                pressure_flux=np.zeros(n * ne, dtype=np.complex128),
                parallel_flux=np.zeros(n * ne, dtype=np.complex128),
            )
        )
        state_fluid.append(
            SimpleNamespace(
                n=core.forward(
                    densities[index].reshape(-1),
                    nc,
                    context,
                ),
                mom=core.forward(
                    momentum[index].reshape(-1),
                    3 * nc,
                    context,
                ),
                perp=core.forward(
                    pressures_perp[index].reshape(-1),
                    nc,
                    context,
                ),
                parallel=core.forward(
                    pressures_parallel[index].reshape(-1),
                    nc,
                    context,
                ),
            )
        )
    state = SimpleNamespace(
        fluid=tuple(state_fluid),
        B=core.forward(magnetic.reshape(-1), nb, context),
    )
    stage = SimpleNamespace(
        fluid=tuple(fluid),
        cache=SimpleNamespace(
            E=core.forward(electric.reshape(-1), ne, context)
        ),
    )
    expected = SimpleNamespace(
        density=face_density,
        velocity=face_velocity,
        flux=fluxes,
        magnetic=magnetic,
        electric=electric,
        pressure_perp=tuple(
            pressure @ context.op.S[0] for pressure in pressures_perp
        ),
        pressure_parallel=tuple(
            pressure @ context.op.S[0] for pressure in pressures_parallel
        ),
        normal=normal,
        tangent=tangent,
    )
    return context, wall, state, stage, expected


def _independent_traction(data, sample, context):
    normal = np.asarray(data.normal3, dtype=np.float64)
    b = np.array([value[sample] for value in data.B], dtype=np.float64)
    density = float(data.density[0][sample])
    velocity = np.asarray(data.velocity[0][sample], dtype=np.float64)
    ion_perp = float(data.perp[0][sample])
    electron_perp = float(data.perp[1][sample])
    electron_parallel = float(data.parallel[1][sample])
    bmag = np.linalg.norm(b)
    bhat = b / bmag
    stress = (
        ion_perp * np.eye(3)
        + electron_perp * np.eye(3)
        + (electron_parallel - electron_perp) * np.outer(bhat, bhat)
        + (0.5 * np.dot(b, b) / context.p.mu0) * np.eye(3)
        - np.outer(b, b) / context.p.mu0
        + context.p.species[0].mass * density * np.outer(velocity, velocity)
    )
    return stress @ normal


class _XpSpy:
    def __init__(self):
        self.asarray_calls = 0

    def asarray(self, value, dtype=None):
        self.asarray_calls += 1
        return np.asarray(value, dtype=dtype)

    def __getattr__(self, name):
        return getattr(np, name)


def test_polygon_extraction_uses_the_global_face_index():
    context, wall, _normal, _tangent = _context()
    extraction = wall_cache._signed_wall_extraction(
        context,
        context.compute,
    )
    expected = np.zeros((1, context.op.Ne), dtype=np.complex128)
    expected[0, wall.face] = 1.0
    np.testing.assert_array_equal(extraction, expected)

    mode_extractions = massless._wall_extractions(context)
    assert len(mode_extractions) == 2 * context.p.K + 1
    for mode_extraction in mode_extractions:
        np.testing.assert_array_equal(mode_extraction, expected)

    groups, names, measure_names, layout = (
        polygon_wall.magnetic_alias_groups(context)
    )
    assert groups == (
        0,
        len(context.geometry.vertices.keys),
        context.op.Nb,
    )
    assert names == ("phi", "tangent")
    assert measure_names == ("phi", "Z")
    assert "B_phi(vertices)" in layout
    assert "B_tangent(faces)" in layout
    assert "normal interval is empty" in layout

    flux = np.arange(context.N * context.op.Ne, dtype=np.float64).reshape(
        context.N, context.op.Ne
    )
    field_flux = massless_field._wall_flux_samples(flux, context)
    check_flux = massless_checks._wall_flux_samples(flux, context)
    np.testing.assert_array_equal(
        field_flux.reshape(context.N, 1)[:, 0],
        flux[:, wall.face],
    )
    np.testing.assert_array_equal(
        check_flux.reshape(context.N, 1)[:, 0],
        flux[:, wall.face],
    )

    legacy, legacy_wall, _legacy_n, _legacy_t = _context(STAIRCASE)
    legacy_wall.component = "r_face"
    legacy_wall.face = 1
    legacy_wall.sign = -1
    legacy_extraction = wall_cache._signed_wall_extraction(
        legacy,
        legacy.compute,
    )
    np.testing.assert_array_equal(
        legacy_extraction,
        np.array(
            ((0.0, -1.0, 0.0),),
            dtype=np.complex128,
        ),
    )
    legacy_groups, legacy_names, legacy_measure_names, legacy_layout = (
        polygon_wall.magnetic_alias_groups(legacy)
    )
    assert legacy_names == ("R", "phi", "Z")
    assert legacy_measure_names == legacy_names
    assert legacy_groups[-1] == legacy.op.Nb
    assert legacy_layout == ""


def test_diagonal_trace_reconstructs_physical_fields_and_zero_poynting():
    context, _wall, state, stage, expected = _trace_fixture()
    data = massless_checks._wall_trace_data(state, stage, context)[0]

    assert data.direction is None
    assert data.wall.component == "plane_face"
    assert data.normal3 == pytest.approx(
        (expected.normal[0], 0.0, expected.normal[1])
    )
    assert data.tangent3 == pytest.approx(
        (expected.tangent[0], 0.0, expected.tangent[1])
    )
    assert data.radius == pytest.approx(data.wall.radius)
    for species in range(2):
        np.testing.assert_allclose(
            data.density[species],
            expected.density[species],
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            data.velocity[species],
            expected.velocity[species],
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            data.perp[species],
            expected.pressure_perp[species],
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            data.parallel[species],
            expected.pressure_parallel[species],
            rtol=2.0e-14,
            atol=2.0e-14,
        )

    expected_b = np.stack(
        (
            expected.magnetic[:, 2] * expected.tangent[0],
            0.5 * (expected.magnetic[:, 0] + expected.magnetic[:, 1]),
            expected.magnetic[:, 2] * expected.tangent[1],
        ),
        axis=1,
    )
    actual_b = np.stack(data.B, axis=1)
    normal3 = np.array((expected.normal[0], 0.0, expected.normal[1]))
    np.testing.assert_allclose(actual_b, expected_b, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(
        data.normal_flux[0],
        expected.flux[0][:, 0],
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        data.normal_flux[1],
        expected.flux[1][:, 0],
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    expected_e = (
        expected.electric[:, 0, None] * np.asarray(data.normal3)[None, :]
    )
    np.testing.assert_allclose(
        data.electric_wall,
        expected_e,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(actual_b @ normal3, 0.0, atol=2.0e-14)
    assert np.count_nonzero(expected_b[:, 1]) == context.N
    assert np.count_nonzero(expected_b[:, 0]) == context.N
    assert np.count_nonzero(expected_b[:, 2]) == context.N

    poynting = polygon_wall.wall_poynting_normal(
        data.electric_wall,
        actual_b,
        data.normal3,
    )
    np.testing.assert_allclose(poynting, 0.0, atol=2.0e-14)


def test_polygon_stress_helpers_use_explicit_backend_without_inference():
    xp = _XpSpy()
    magnetic = np.array((0.25, 0.4, 0.3), dtype=np.float64)
    normal = np.array((0.6, 0.0, 0.8), dtype=np.float64)
    electric = np.array((1.5, 0.0, 2.0), dtype=np.float64)
    magnetic_traction = polygon_wall.maxwell_traction(
        magnetic,
        normal,
        1.25,
        xp=xp,
    )
    poynting = polygon_wall.wall_poynting_normal(
        electric,
        magnetic,
        normal,
        xp=xp,
    )
    assert xp.asarray_calls == 5
    assert np.all(np.isfinite(magnetic_traction))
    assert abs(float(poynting)) <= 2.0e-15

    context, _wall, state, stage, _expected = _trace_fixture()
    data = massless_checks._wall_trace_data(state, stage, context)[0]
    original = context.compute.xp
    context.compute.xp = xp
    try:
        before = xp.asarray_calls
        transport, electromagnetic = polygon_wall.traction_components(
            data,
            0,
            context,
        )
    finally:
        context.compute.xp = original
    # Batched observations explicitly live on the host after one export.
    assert xp.asarray_calls == before
    assert np.all(np.isfinite(transport))
    assert np.all(np.isfinite(electromagnetic))

    # The unbatched helper still honors the caller's explicit backend.
    scalar_data = SimpleNamespace(**vars(data))
    del scalar_data.host
    context.compute.xp = xp
    try:
        scalar_transport, scalar_electromagnetic = (
            polygon_wall.traction_components(scalar_data, 0, context)
        )
    finally:
        context.compute.xp = original
    assert xp.asarray_calls > before
    np.testing.assert_allclose(scalar_transport, transport, rtol=2.0e-14)
    np.testing.assert_allclose(
        scalar_electromagnetic, electromagnetic, rtol=2.0e-14
    )


def test_polygon_trace_inverse_transforms_each_field_once(monkeypatch):
    context, _wall, state, stage, _expected = _trace_fixture()
    calls = []
    original = polygon_wall.inverse

    def counted(values, entities, active_context, physical=True, reference=0.0):
        calls.append((values.shape, entities, physical, reference))
        return original(
            values,
            entities,
            active_context,
            physical,
            reference,
        )

    monkeypatch.setattr(polygon_wall, "inverse", counted)
    traces = polygon_wall.wall_trace_data(state, stage, context)
    assert len(traces) == 1
    assert [entry[1] for entry in calls] == [
        context.op.Ne,
        context.op.Nb,
        3 * context.op.Nc,
        3 * context.op.Nc,
    ]
    assert all(entry[2] is True for entry in calls)
    assert sum(entry[3] > 0.0 for entry in calls) == 1


def test_total_and_maxwell_traction_and_torque_use_dot_products():
    context, _wall, state, stage, expected = _trace_fixture()
    data = massless_checks._wall_trace_data(state, stage, context)[0]
    traces = massless_checks._wall_trace_data(state, stage, context)

    integrated = np.zeros(3)
    integrated_torque = 0.0
    samples = max(context.N, 3)
    for q in range(samples):
        sample = 0 if context.N == 1 else q
        angle = 2.0 * math.pi * q / samples
        expected_traction = _independent_traction(data, sample, context)
        actual_traction = np.asarray(
            polygon_wall.wall_traction(data, sample, context)
        )
        np.testing.assert_allclose(
            actual_traction,
            expected_traction,
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        transport, electromagnetic = (
            multirate_wr_checks._traction_components(
                data,
                sample,
                context,
            )
        )
        np.testing.assert_allclose(
            np.asarray(transport) + np.asarray(electromagnetic),
            expected_traction,
            rtol=2.0e-14,
            atol=2.0e-14,
        )

        magnetic = np.array([value[sample] for value in data.B])
        normal3 = np.array(
            (
                expected.normal[0],
                0.0,
                expected.normal[1],
            )
        )
        independent_maxwell = (
            0.5 * np.dot(magnetic, magnetic) * normal3
            - np.dot(magnetic, normal3) * magnetic
        ) / context.p.mu0
        np.testing.assert_allclose(
            polygon_wall.maxwell_traction(
                magnetic,
                normal3,
                context.p.mu0,
            ),
            independent_maxwell,
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        torque = data.radius * expected_traction[1]
        assert torque == pytest.approx(
            data.radius * float(np.asarray(actual_traction)[1]),
            rel=2.0e-14,
            abs=2.0e-14,
        )
        if torque != 0.0:
            assert np.sign(torque) == np.sign(expected_traction[1])

        cosine = math.cos(angle)
        sine = math.sin(angle)
        wall_weight = (
            2.0 * math.pi * data.wall.area_per_radian / samples
        )
        cartesian = np.array(
            (
                expected_traction[0] * cosine
                - expected_traction[1] * sine,
                expected_traction[0] * sine
                + expected_traction[1] * cosine,
                expected_traction[2],
            )
        )
        integrated += wall_weight * cartesian
        integrated_torque += wall_weight * data.radius * expected_traction[1]

    actual = multirate_wr_checks._momentum_integrals(
        state,
        stage,
        None,
        context,
        traction_role="transport",
    )
    transport_actual = np.asarray(actual[0])
    expected_transport = np.zeros(3)
    expected_transport_torque = 0.0
    for q in range(samples):
        sample = 0 if context.N == 1 else q
        angle = 2.0 * math.pi * q / samples
        normal3 = np.array(
            (
                expected.normal[0],
                0.0,
                expected.normal[1],
            )
        )
        density = data.density[0][sample]
        velocity = np.real(data.velocity[0][sample])
        transport = (
            context.p.species[0].mass
            * density
            * velocity
            * np.dot(velocity, normal3)
            + data.perp[0][sample] * normal3
        )
        wall_weight = (
            2.0 * math.pi * data.wall.area_per_radian / samples
        )
        expected_transport += wall_weight * np.array(
            (
                transport[0] * math.cos(angle)
                - transport[1] * math.sin(angle),
                transport[0] * math.sin(angle)
                + transport[1] * math.cos(angle),
                transport[2],
            )
        )
        expected_transport_torque += (
            wall_weight * traces[0].radius * transport[1]
        )
    np.testing.assert_allclose(transport_actual, expected_transport, rtol=2.0e-14, atol=2.0e-14)
    assert actual[1] == pytest.approx(
        expected_transport_torque,
        rel=2.0e-14,
        abs=2.0e-14,
    )
    assert np.linalg.norm(integrated) > 0.0
    assert abs(integrated_torque) > 0.0


def test_wall_projection_removes_normal_momentum_not_tangential_momentum():
    context, wall, normal, _tangent = _context()
    xp = context.compute.xp
    density = np.array((1.5, 2.5), dtype=np.float64)
    weight = (
        context.p.species[0].mass
        * density[None, :]
        / context.op.M0[None, :]
    )
    weighted = np.tile(weight, (1, 3))
    scaled = (
        context.op.T
        @ (weighted[0][:, None] * context.op.T.conj().T)
    )
    record = SimpleNamespace(
        S=xp.asarray(scaled),
        lu=None,
        basis=polygon_projection.prepare(
            context.op.T,
            weighted[0],
            context,
            label="test wall",
        ),
        winv=xp.asarray(weighted[0]),
        row_scale=1.0,
        col_scale=1.0,
    )
    wall_object = SimpleNamespace(samples=[record] * context.N)
    velocity = np.array((1.2, -0.7, 0.4), dtype=np.float64)
    momentum = np.broadcast_to(
        (
            context.p.species[0].mass
            * density[None, :]
            * velocity[:, None]
        )[None, :, :],
        (context.N, 3, context.op.Nc),
    ).copy()
    projected = project_wall(
        wall_object,
        momentum.reshape(-1),
        context,
    )
    value = np.asarray(projected.value).reshape(
        context.N,
        3,
        context.op.Nc,
    )
    reaction = np.asarray(projected.reaction).reshape(value.shape)
    for sample in range(context.N):
        trace = context.op.T @ value[sample].reshape(-1)
        assert abs(float(np.real(trace[0]))) <= 1.0e-14
        normal3 = np.array((normal[0], 0.0, normal[1]))
        normal_momentum = np.einsum(
            "cn,c->n",
            value[sample],
            normal3,
        )
        tangential = value[sample] - np.outer(
            normal3,
            normal_momentum,
        )
        assert np.linalg.norm(tangential) > 0.1

    weighted_reaction_work = np.sum(
        value
        * reaction
        / weighted.reshape(1, 3, context.op.Nc)
    )
    assert abs(float(np.real(weighted_reaction_work))) <= 2.0e-14
    assert np.linalg.norm(reaction) > 0.0


def test_boundary_uses_shared_polygon_face_flux_and_frame():
    context, _wall, state, stage, expected = _trace_fixture()
    boundary = massless_checks._boundary(state, stage, context)
    wr = multirate_wr_checks._boundary_summands(
        state,
        stage,
        context,
    )
    samples = max(context.N, 3)
    sample_weight = (
        2.0
        * math.pi
        * context.geometry.walls[0].area_per_radian
        / samples
    )
    for species in range(2):
        expected_number = (
            expected.flux[species][:, 0].sum() * sample_weight
        )
        assert boundary.number[species] == pytest.approx(
            expected_number,
            rel=2.0e-14,
            abs=2.0e-14,
        )
    assert boundary.traces
    trace = boundary.traces[0]
    assert trace.direction is None
    assert trace.normal3 == pytest.approx(
        (expected.normal[0], 0.0, expected.normal[1])
    )
    assert trace.tangent3 == pytest.approx(
        (expected.tangent[0], 0.0, expected.tangent[1])
    )
    assert trace.radius == pytest.approx(context.geometry.walls[0].radius)
    assert trace.normal_flux[0] == pytest.approx(expected.flux[0][0, 0])
    assert trace.normal_flux[1] == pytest.approx(expected.flux[1][0, 0])
    np.testing.assert_allclose(
        wr.number,
        boundary.number,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        wr.kinetic,
        boundary.kinetic,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        wr.internal_perp,
        boundary.internal_perp,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        wr.internal_parallel,
        boundary.internal_parallel,
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_polygon_path_rejects_axis_labeled_walls():
    context, wall, _normal, _tangent = _context()
    wall.component = "r_face"
    with pytest.raises(Failure, match="plane_face"):
        polygon_wall.wall_frame(context, wall)


def test_polygon_path_rejects_zero_radius():
    context, wall, _normal, _tangent = _context()
    wall.radius = 0.0
    wall.area_per_radian = 0.0
    context.geometry.faces.R[0] = 0.0
    with pytest.raises(Failure, match="radius"):
        polygon_wall.wall_frame(context, wall)
