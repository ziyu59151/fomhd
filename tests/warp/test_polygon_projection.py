"""Independent weighted projection tests on actual polygon operators."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from fomhd import massless, polygon_projection, wall
from fomhd.backend import Backend
from fomhd.geometry import Point2
from fomhd.polygon_geometry import (
    PolygonGeometryOptions,
    build_polygon_geometry,
)
from fomhd.polygon_operators import (
    PolygonOperatorOptions,
    build_polygon_operators,
)
from fomhd.types import Context, Failure, StageTag
from test_massless_support import parameters


_CASES = {}


def _polygon_case(nodes):
    cached = _CASES.get(nodes)
    if cached is not None:
        return cached
    segments = 4 * nodes
    polygon = [
        Point2(
            1.6 + 0.85 * math.cos(2.0 * math.pi * index / segments),
            0.5 + 0.65 * math.sin(2.0 * math.pi * index / segments),
        )
        for index in range(segments)
    ]
    radius = np.linspace(0.95, 2.25, nodes, dtype=np.float64)
    height = np.linspace(-0.25, 1.25, nodes, dtype=np.float64)
    geometry = build_polygon_geometry(
        radius,
        height,
        polygon,
        PolygonGeometryOptions(memory_cap_bytes=256 << 20),
    )
    backend = Backend("cpu")
    operators = build_polygon_operators(
        geometry,
        0,
        PolygonOperatorOptions(
            memory_cap_bytes=512 << 20,
            rank_rtol=1.0e-10,
        ),
        backend,
    )
    params = parameters(K=0, hard_diagnostics=False)
    params.rank_rtol = 1.0e-10
    context = Context(
        p=params,
        geometry=geometry,
        op=operators,
        compute=backend,
        N=5,
    )
    case = (context, np.asarray(operators.T), np.asarray(operators.T_full))
    _CASES[nodes] = case
    return case


def _density(context, seed):
    rng = np.random.default_rng(seed)
    nc = context.op.Nc
    cell_shape = 0.75 + 0.5 * rng.random((context.N, nc))
    return 1.0e12 * cell_shape


def _weights(density, context):
    m0 = np.real(np.asarray(context.op.M0))
    return np.tile(density / m0[None, :], (1, 3))


def _gate_bound(source, value):
    scale = max(
        float(np.max(np.abs(source))),
        float(np.max(np.abs(value))),
        1.0e-300,
    )
    return 4096.0 * np.finfo(np.float64).eps * scale


@pytest.mark.parametrize("nodes", [8, 12])
def test_actual_nonaxis_projection_matches_weighted_gram_oracle(nodes):
    context, retained, full = _polygon_case(nodes)
    assert full.shape[0] > retained.shape[0]
    assert retained.shape == (len(context.op.selected_wall_rows), 3 * context.op.Nc)
    wall_faces = np.array(
        [item.face for item in context.geometry.walls],
        dtype=np.int64,
    )
    normals = context.geometry.face_normals[wall_faces]
    assert np.any(
        (np.abs(normals[:, 0]) > 0.1)
        & (np.abs(normals[:, 1]) > 0.1)
    )

    density = _density(context, 700 + nodes)
    winv = _weights(density, context)
    rng = np.random.default_rng(800 + nodes)
    source = (
        rng.standard_normal(context.N * 3 * context.op.Nc)
        + 1j
        * rng.standard_normal(context.N * 3 * context.op.Nc)
    )
    expected_values = np.empty_like(source)
    for sample in range(context.N):
        start = sample * 3 * context.op.Nc
        stop = start + 3 * context.op.Nc
        sample_source = source[start:stop]
        weights = winv[sample]
        normal = (
            retained
            @ np.diag(weights)
            @ retained.conj().T
        )
        multiplier = np.linalg.solve(
            normal,
            retained @ sample_source,
        )
        correction = (
            weights
            * (retained.conj().T @ multiplier)
        )
        expected = sample_source - correction
        expected_reaction = expected - sample_source
        expected_values[start:stop] = expected

        projected = polygon_projection.project(
            sample_source,
            retained,
            full,
            weights,
            context,
        )
        assert projected.rank == retained.shape[0]
        assert projected.refinements <= 5
        np.testing.assert_allclose(
            projected.value,
            expected,
            rtol=2.0e-9,
            atol=2.0e-9 * max(float(np.max(np.abs(sample_source))), 1.0),
        )
        np.testing.assert_allclose(
            projected.reaction,
            expected_reaction,
            rtol=2.0e-9,
            atol=2.0e-9 * max(float(np.max(np.abs(sample_source))), 1.0),
        )
        np.testing.assert_allclose(
            projected.reaction,
            -weights * (retained.conj().T @ projected.multiplier),
            rtol=2.0e-10,
            atol=2.0e-10 * max(float(np.max(np.abs(projected.reaction))), 1.0),
        )
        assert float(np.max(np.abs(projected.full_trace))) <= _gate_bound(
            sample_source,
            projected.value,
        )
        weighted_work = np.sum(
            projected.value
            * projected.reaction
            / weights
        )
        assert abs(weighted_work) <= 2.0e-10 * max(
            float(np.sum(np.abs(projected.value * projected.reaction / weights))),
            1.0,
        )
        expected_residual = (
            retained
            @ (
                weights
                * (retained.conj().T @ projected.multiplier)
            )
            - retained @ sample_source
        )
        np.testing.assert_allclose(
            projected.multiplier_residual,
            expected_residual,
            rtol=2.0e-11,
            atol=2.0e-11 * max(
                float(np.max(np.abs(expected_residual))),
                1.0,
            ),
        )
    actual = massless._constraint_project(
        source,
        density.reshape(-1),
        context,
    )
    np.testing.assert_allclose(
        actual,
        expected_values,
        rtol=2.0e-9,
        atol=2.0e-9 * max(float(np.max(np.abs(source))), 1.0),
    )


def test_weighted_projection_preserves_independent_nullspace_component():
    context, retained, full = _polygon_case(12)
    density = _density(context, 901)
    weights = _weights(density, context)[0]
    _, _, vh = np.linalg.svd(retained, full_matrices=True)
    rank = retained.shape[0]
    nullspace = vh[rank:].conj().T
    assert nullspace.shape[1] > 0
    rng = np.random.default_rng(902)
    null_component = nullspace @ (
        rng.standard_normal(nullspace.shape[1])
        + 1j * rng.standard_normal(nullspace.shape[1])
    )
    row_coefficients = (
        rng.standard_normal(rank)
        + 1j * rng.standard_normal(rank)
    )
    row_component = weights * (
        retained.conj().T @ row_coefficients
    )
    source = row_component + null_component
    projected = polygon_projection.project(
        source,
        retained,
        full,
        weights,
        context,
    )
    np.testing.assert_allclose(
        projected.value,
        null_component,
        rtol=2.0e-10,
        atol=2.0e-10 * max(float(np.max(np.abs(source))), 1.0),
    )
    weighted_work = np.sum(
        projected.value * projected.reaction / weights
    )
    assert abs(weighted_work) <= 2.0e-10 * max(
        float(np.sum(np.abs(projected.value * projected.reaction / weights))),
        1.0,
    )


def test_ill_conditioned_nonaxis_rows_keep_the_same_weighted_nullspace():
    context, retained, full = _polygon_case(8)
    row_scale = np.logspace(0.0, 6.0, retained.shape[0])
    ill_conditioned = row_scale[:, None] * retained
    assert np.linalg.cond(ill_conditioned) > 1.0e6
    density = _density(context, 951)
    weights = _weights(density, context)[0]
    _, _, vh = np.linalg.svd(ill_conditioned, full_matrices=True)
    rank = ill_conditioned.shape[0]
    nullspace = vh[rank:].conj().T
    rng = np.random.default_rng(952)
    null_component = nullspace @ (
        rng.standard_normal(nullspace.shape[1])
        + 1j * rng.standard_normal(nullspace.shape[1])
    )
    row_component = weights * (
        ill_conditioned.conj().T
        @ (
            rng.standard_normal(rank)
            + 1j * rng.standard_normal(rank)
        )
    )
    source = row_component + null_component
    projected = polygon_projection.project(
        source,
        ill_conditioned,
        full,
        weights,
        context,
    )
    np.testing.assert_allclose(
        projected.value,
        null_component,
        rtol=3.0e-9,
        atol=3.0e-9 * max(float(np.max(np.abs(source))), 1.0),
    )


def test_outward_normal_source_has_inward_axis_and_nonaxis_reaction():
    context, retained, full = _polygon_case(8)
    density = _density(context, 975)
    weights = _weights(density, context)[0]

    axis_row = np.array(((1.0, 0.0, 0.0),), dtype=np.complex128)
    axis_source = np.array((1.0, 0.0, 0.0), dtype=np.complex128)
    axis = polygon_projection.project(
        axis_source,
        axis_row,
        axis_row,
        weights[:3],
        context,
        label="axis oracle",
    )
    axis_lambda = np.linalg.solve(
        axis_row @ np.diag(weights[:3]) @ axis_row.conj().T,
        axis_row @ axis_source,
    )
    expected_axis_reaction = -(
        weights[:3] * (axis_row.conj().T @ axis_lambda)
    )
    np.testing.assert_allclose(
        axis.reaction,
        expected_axis_reaction,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert np.real(axis.reaction[0]) < 0.0

    normal = np.array((0.6, 0.0, 0.8), dtype=np.float64)
    nonaxis_row = normal.astype(np.complex128)[None, :]
    nonaxis_source = normal.astype(np.complex128)
    nonaxis = polygon_projection.project(
        nonaxis_source,
        nonaxis_row,
        nonaxis_row,
        weights[:3],
        context,
        label="nonaxis oracle",
    )
    nonaxis_lambda = np.linalg.solve(
        nonaxis_row
        @ np.diag(weights[:3])
        @ nonaxis_row.conj().T,
        nonaxis_row @ nonaxis_source,
    )
    expected_nonaxis_reaction = -(
        weights[:3] * (nonaxis_row.conj().T @ nonaxis_lambda)
    )
    np.testing.assert_allclose(
        nonaxis.reaction,
        expected_nonaxis_reaction,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert np.real(np.dot(nonaxis.reaction, normal)) < 0.0


def test_massless_and_ion_wall_projections_share_polygon_geometry():
    context, _retained, _full = _polygon_case(8)
    one_sample = copy.copy(context)
    one_sample.N = 1
    density = _density(one_sample, 1001)[0]
    rng = np.random.default_rng(1002)
    source = (
        rng.standard_normal(3 * one_sample.op.Nc)
        + 1j * rng.standard_normal(3 * one_sample.op.Nc)
    )
    tag = StageTag(stage="polygon_projection_test")
    ion_wall = wall.build_wall(
        density.astype(np.complex128),
        one_sample.p.species[0],
        tag,
        one_sample,
    )
    assert ion_wall.samples[0].basis is not None
    assert ion_wall.samples[0].S.shape == (
        one_sample.op.T.shape[0],
        one_sample.op.T.shape[0],
    )
    ion_projection = wall.project_wall(
        ion_wall,
        source,
        one_sample,
    )
    electron_value = massless._constraint_project(
        source,
        density,
        one_sample,
    )
    np.testing.assert_allclose(
        electron_value,
        ion_projection.value,
        rtol=2.0e-11,
        atol=2.0e-11 * max(float(np.max(np.abs(source))), 1.0),
    )
    weights = _weights(density[None, :], one_sample)[0]
    np.testing.assert_allclose(
        ion_projection.reaction,
        electron_value - source,
        rtol=2.0e-11,
        atol=2.0e-11 * max(float(np.max(np.abs(source))), 1.0),
    )
    weighted_work = np.sum(
        ion_projection.value * ion_projection.reaction / weights
    )
    assert abs(weighted_work) <= 2.0e-11 * max(
        float(
            np.sum(
                np.abs(ion_projection.value)
                * np.abs(ion_projection.reaction)
                / weights
            )
        ),
        1.0,
    )
    assert ion_projection.rank == one_sample.op.T.shape[0]
    assert ion_projection.refinements <= 5


def test_rank_deficient_retained_rows_fail_without_dropping_constraint():
    context, retained, full = _polygon_case(8)
    density = _density(context, 1101)
    weights = _weights(density, context)[0]
    source = np.ones(3 * context.op.Nc, dtype=np.complex128)
    deficient = np.vstack((retained, retained[0]))
    with pytest.raises(Failure, match="rank"):
        polygon_projection.project(
            source,
            deficient,
            full,
            weights,
            context,
        )


def test_inconsistent_full_row_is_retained_as_explicit_failure():
    context, retained, full = _polygon_case(8)
    density = _density(context, 1201)
    weights = _weights(density, context)[0]
    source = np.ones(3 * context.op.Nc, dtype=np.complex128)
    extra = np.zeros((1, retained.shape[1]), dtype=np.complex128)
    extra[0, 0] = 1.0
    impossible = np.vstack((full, extra))
    with pytest.raises(Failure, match="full|forward|wall"):
        polygon_projection.project(
            source,
            retained,
            impossible,
            weights,
            context,
        )
