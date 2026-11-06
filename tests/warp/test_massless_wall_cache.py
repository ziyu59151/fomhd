"""Contracts for fixed massless wall-system preparation and reuse."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from test_massless_support import context, stage_tag, uniform_state

from fomhd import massless, wall_cache
from fomhd.backend import Backend
from fomhd.types import Failure


def _nonzero_sources(c, base):
    rng = np.random.default_rng(20261106)
    noise = c.compute.array(
        rng.standard_normal(base.size) + 1j * rng.standard_normal(base.size)
    )
    return (
        base,
        c.compute.asarray(0.37 * base + 1.0e-3 * noise),
        c.compute.asarray(-0.83 * base - 2.0e-3 * noise),
    )


def _dense_wall_equation(c, stored):
    m1 = np.real(np.asarray(c.op.M1)).astype(np.float64)
    extraction = massless._signed_wall_extraction(c, stored)
    curl = np.asarray(c.op.Cdag[stored]) @ np.asarray(c.op.C[stored])
    matrix = extraction @ curl @ (extraction.T.conj() / m1[:, None])
    return extraction, matrix


def _selected_solution(c, stored, matrix, rhs):
    rows = c.compute.independent_rows(matrix, c.p.rank_rtol)
    multiplier = np.zeros(matrix.shape[0], dtype=np.complex128)
    if rows:
        selected = np.asarray(rows, dtype=np.int64)
        principal = matrix[np.ix_(selected, selected)]
        multiplier[selected] = np.linalg.solve(principal, rhs[selected])
    return rows, multiplier


def test_prepared_backend_linear_matches_existing_linear_for_several_rhs():
    backend = Backend("cpu")
    matrix = backend.array(
        [
            [5.0 + 0.0j, 1.0 + 0.5j, 0.25 + 0.0j],
            [1.0 - 0.5j, 4.0 + 0.0j, -0.75 + 0.125j],
            [0.25 + 0.0j, -0.75 - 0.125j, 3.0 + 0.0j],
        ]
    )
    row_scale = backend.real_array([0.5, 1.5, 2.0])
    column_scale = backend.real_array([3.0, 0.75, 1.25])
    prepared = backend.prepare_linear(
        matrix,
        row_scale,
        column_scale,
        1.0e-12,
        1.0e-12,
    )
    for rhs in (
        backend.array([1.0 + 0.0j, -2.0 + 1.0j, 0.5 - 3.0j]),
        backend.array([0.25 - 0.5j, 4.0 + 2.0j, -1.0 + 0.0j]),
        backend.array([-3.0 + 0.0j, 0.0 + 1.0j, 2.0 + 2.0j]),
    ):
        expected = backend.linear(
            matrix,
            rhs,
            row_scale,
            column_scale,
            1.0e-12,
            1.0e-12,
        )
        actual = backend.linear_prepared(prepared, rhs)
        np.testing.assert_allclose(actual.x, expected.x, rtol=1.0e-12, atol=1.0e-14)
        np.testing.assert_allclose(
            actual.residual,
            expected.residual,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        np.testing.assert_allclose(
            actual.singular_values,
            expected.singular_values,
            rtol=0.0,
            atol=0.0,
        )
        assert actual.scaled_residual == pytest.approx(
            expected.scaled_residual, rel=1.0e-14, abs=1.0e-16
        )
        assert actual.backward_error == pytest.approx(
            expected.backward_error, rel=1.0e-14, abs=1.0e-16
        )
        assert actual.refinement_steps == expected.refinement_steps
        np.testing.assert_array_equal(actual.row_scale, expected.row_scale)
        np.testing.assert_array_equal(actual.column_scale, expected.column_scale)


def test_wall_aw_uses_only_thin_products_and_matches_dense_equation(monkeypatch):
    c = context(K=1, hard_diagnostics=False, nodes_r=6, nodes_z=6)
    modes = 2 * c.p.K + 1
    source = c.compute.array(
        np.random.default_rng(11).standard_normal(modes * c.op.Ne)
    )
    products = []
    original_mm = Backend.mm

    def recording_mm(self, left, right):
        products.append((tuple(left.shape), tuple(right.shape)))
        return original_mm(self, left, right)

    monkeypatch.setattr(Backend, "mm", recording_mm)
    correction, records = massless._wall_correction(source, c)
    assert np.max(np.abs(correction)) > 0.0
    assert products
    assert all(left != (c.op.Ne, c.op.Ne) for left, _ in products)
    assert any(
        left == (c.op.Nb, c.op.Ne)
        and right == (c.op.Ne, len(c.geometry.walls))
        for left, right in products
    )
    assert any(
        left == (c.op.Ne, c.op.Nb)
        and right == (c.op.Nb, len(c.geometry.walls))
        for left, right in products
    )
    assert any(
        left == (len(c.geometry.walls), c.op.Ne)
        and right == (c.op.Ne, len(c.geometry.walls))
        for left, right in products
    )

    cache = records.cache
    assert cache.setup_count == 1
    assert cache.solves == len(records.modes)
    assert cache.storage_bytes > 0
    wall_count = len(c.geometry.walls)
    expected_bytes = sum(
        16
        * (
            2 * wall_count * c.op.Ne
            + wall_count**2
            + 4 * record.rank**2
        )
        + 28 * record.rank
        + 8 * wall_count
        for record in records.modes
    )
    assert cache.storage_bytes == expected_bytes
    for stored, (record, cached) in enumerate(
        zip(records.modes, cache.modes, strict=True)
    ):
        extraction, dense = _dense_wall_equation(c, stored)
        np.testing.assert_array_equal(cached.extraction, extraction)
        np.testing.assert_allclose(
            cached.matrix,
            dense,
            rtol=2.0e-14,
            atol=2.0e-14 * max(float(np.max(np.abs(dense))), 1.0),
        )
        assert record.mode == stored - c.p.K
        assert record.selected_rows == list(cached.selected_rows)
        assert record.gauge_rows == list(cached.gauge_rows)
        assert record.rank == len(record.selected_rows)
        rhs_scale = max(
            float(np.max(np.abs(record.rhs))),
            1.0e-300,
        )
        assert float(np.max(np.abs(record.full_residual))) <= 1.0e-8 * rhs_scale


def test_wall_cache_reuses_setup_for_multiple_nonzero_rhs(monkeypatch):
    c = context(K=1, hard_diagnostics=False, nodes_r=6, nodes_z=6)
    modes = 2 * c.p.K + 1
    base = c.compute.array(
        np.random.default_rng(17).standard_normal(modes * c.op.Ne)
        + 1j * np.random.default_rng(18).standard_normal(modes * c.op.Ne)
    )
    _first, first_records = massless._wall_correction(base, c)
    cache = first_records.cache

    setup_counts = {"independent_rows": 0, "singular_values": 0, "factor": 0}
    for name in setup_counts:
        original = getattr(Backend, name)

        def counted(self, *args, __name=name, __original=original, **kwargs):
            setup_counts[__name] += 1
            return __original(self, *args, **kwargs)

        monkeypatch.setattr(Backend, name, counted)

    for source in _nonzero_sources(c, base)[1:]:
        for name in setup_counts:
            setup_counts[name] = 0
        correction, records = massless._wall_correction(source, c)
        assert setup_counts == {
            "independent_rows": 0,
            "singular_values": 0,
            "factor": 0,
        }
        assert np.max(np.abs(correction)) > 0.0
        assert records.cache is cache
        for stored, record in enumerate(records.modes):
            extraction, matrix = _dense_wall_equation(c, stored)
            expected_rows, expected_multiplier = _selected_solution(
                c, stored, matrix, record.rhs
            )
            assert record.selected_rows == expected_rows
            np.testing.assert_allclose(
                record.multiplier,
                expected_multiplier,
                rtol=2.0e-10,
                atol=2.0e-10
                * max(float(np.max(np.abs(record.multiplier))), 1.0),
            )
            expected_correction = (
                extraction.T.conj() @ expected_multiplier
            ) / np.real(np.asarray(c.op.M1))
            stored_slice = slice(stored * c.op.Ne, (stored + 1) * c.op.Ne)
            np.testing.assert_allclose(
                np.asarray(correction[stored_slice]),
                expected_correction,
                rtol=2.0e-10,
                atol=2.0e-10
                * max(float(np.max(np.abs(expected_correction))), 1.0),
            )
    assert cache.setup_count == 1
    assert cache.solves == 3 * len(cache.modes)


@pytest.mark.parametrize(
    "change",
    ("operator", "geometry", "reference", "rank", "backend", "identity_hash"),
)
def test_wall_cache_invalidates_on_identity_changes(change, monkeypatch):
    c = context(K=0, hard_diagnostics=False, nodes_r=4, nodes_z=4)
    source = c.compute.array(
        np.random.default_rng(29).standard_normal(c.op.Ne)
    )
    _, first = massless._wall_correction(source, c)
    first_cache = first.cache
    first_key = first_cache.key

    if change == "operator":
        c.op = copy.copy(c.op)
    elif change == "geometry":
        c.geometry = copy.copy(c.geometry)
    elif change == "reference":
        c.p.ref.length *= 1.01
    elif change == "rank":
        c.p.rank_rtol = 2.0e-12
    elif change == "backend":
        c.compute = Backend("cpu")
        c.op.compute = c.compute
    else:
        c.operator_hash = "replaced-operator-identity"

    _, second = massless._wall_correction(source, c)
    assert second.cache is not first_cache
    assert second.cache.key != first_key
    assert second.cache.setup_count == 1
    assert getattr(c, "_wall_system_cache_builds", 0) == 2


def test_massless_stage_reuses_wall_setup_and_repeats_identical_outputs():
    c = context(K=1, hard_diagnostics=False, nodes_r=6, nodes_z=6)
    state = uniform_state(c)
    tag = stage_tag(state)
    first = massless.evaluate_stage(state, tag, c)
    first_cache = first.cache.wall_correction.cache
    second = massless.evaluate_stage(state, tag, c)
    assert second.cache.wall_correction.cache is first_cache
    assert first_cache.setup_count == 1
    np.testing.assert_array_equal(second.E0, first.E0)
    np.testing.assert_array_equal(second.cache.E, first.cache.E)
    np.testing.assert_array_equal(second.cache.wall_reaction, first.cache.wall_reaction)
    np.testing.assert_array_equal(second.rhs.B, first.rhs.B)


def test_actual_alias_stage_shares_fixed_wall_setup(monkeypatch):
    c = context(K=1, hard_diagnostics=False, nodes_r=4, nodes_z=4)
    state = uniform_state(c)
    tag = stage_tag(state)
    first = massless.evaluate_stage(state, tag, c, alias=True)
    cache = first.cache.wall_correction.cache
    builds = 0
    original = wall_cache._build_wall_cache

    def counted(context):
        nonlocal builds
        builds += 1
        return original(context)

    monkeypatch.setattr(wall_cache, "_build_wall_cache", counted)
    second = massless.evaluate_stage(state, tag, c, alias=True)
    assert builds == 0
    assert second.cache.wall_correction.cache is cache
    np.testing.assert_array_equal(second.cache.E, first.cache.E)
    np.testing.assert_array_equal(second.rhs.B, first.rhs.B)
    assert any(check.name == "alias_B" for check in second.checks)


def test_wall_cache_setup_library_failure_retains_model_evidence(monkeypatch):
    c = context(K=0, hard_diagnostics=False, nodes_r=4, nodes_z=4)
    state = uniform_state(c)
    tag = stage_tag(state)

    def fail(*args, **kwargs):
        raise RuntimeError("injected wall factorization failure")

    monkeypatch.setattr(Backend, "prepare_linear", fail)
    with pytest.raises(Failure) as error:
        massless.evaluate_stage(state, tag, c)
    assert error.value.code == "NumericalLibrary"
    assert error.value.stage is not None
    assert error.value.stage.stage == "test"
    assert error.value.partial_arrays
    assert "injected wall factorization failure" in str(error.value)
    assert error.value.checks


def test_tiny_memory_cap_fails_before_wall_allocations(monkeypatch):
    c = context(K=1, hard_diagnostics=False, nodes_r=6, nodes_z=6)
    c.p.memory_limit = 1
    source = c.compute.array(
        np.random.default_rng(41).standard_normal(
            (2 * c.p.K + 1) * c.op.Ne
        )
    )
    calls = {"mm": 0, "factor": 0}
    for name in calls:
        original = getattr(Backend, name)

        def counted(self, *args, __name=name, __original=original, **kwargs):
            calls[__name] += 1
            return __original(self, *args, **kwargs)

        monkeypatch.setattr(Backend, name, counted)

    with pytest.raises(Failure, match="memory budget exceeded"):
        massless._wall_correction(source, c)
    assert calls == {"mm": 0, "factor": 0}


def test_wall_rhs_and_conditioning_failures_are_explicit(monkeypatch):
    c = context(K=0, hard_diagnostics=False, nodes_r=4, nodes_z=4)
    invalid = c.compute.array(np.ones(c.op.Ne, dtype=np.complex128))
    invalid[1] = np.nan
    with pytest.raises(Failure, match="wall correction right-hand side"):
        massless._wall_correction(invalid, c)

    other = context(K=0, hard_diagnostics=False, nodes_r=4, nodes_z=4)
    monkeypatch.setattr(
        Backend,
        "independent_rows",
        lambda self, matrix, rank_rtol: list(range(matrix.shape[0])),
    )
    source = other.compute.array(
        np.random.default_rng(37).standard_normal(other.op.Ne)
    )
    with pytest.raises(Failure, match="singular or ill-conditioned"):
        massless._wall_correction(source, other)
