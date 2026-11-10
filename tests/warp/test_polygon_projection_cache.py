from __future__ import annotations

import copy
import gc
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from fomhd import core, polygon_projection
from fomhd.backend import Backend
from fomhd.polygon_geometry import build_polygon_geometry
from fomhd.polygon_operators import build_polygon_operators


def _retained_matrix(rows: int = 3, columns: int = 6) -> np.ndarray:
    matrix = np.zeros((rows, columns), dtype=np.complex128)
    for row in range(rows):
        matrix[row, row] = 1.0 + 0.1j * row
    if rows > 1:
        matrix[0, rows] = 0.25 - 0.1j
        matrix[1, rows + 1] = -0.2 + 0.3j
    return matrix


def _full_matrix(retained: np.ndarray) -> np.ndarray:
    extra = retained[:2] * (0.5 - 0.25j)
    return np.vstack((retained, extra))


def _context(retained: np.ndarray):
    backend = Backend("cpu")
    operators = SimpleNamespace(
        matrix_layout_version="polygon_dual_v1",
        geometry_kind="polygon_dual_v1",
        selected_wall_rows=list(range(retained.shape[0])),
        T=retained.copy(),
    )
    geometry = SimpleNamespace(kind="polygon_dual_v1")
    parameters = SimpleNamespace(rank_rtol=1.0e-10, K=0)
    return SimpleNamespace(
        N=2,
        p=parameters,
        op=operators,
        geometry=geometry,
        compute=backend,
    )


def _winv(columns: int = 6, offset: int = 0) -> np.ndarray:
    return np.linspace(0.5, 1.5, columns) + 0.1 * offset


def _source(columns: int = 6, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal(columns)
        + 1j * rng.standard_normal(columns)
    )


@pytest.fixture
def svd_count(monkeypatch):
    counter = SimpleNamespace(calls=0)
    original = np.linalg.svd

    def counted(*args, **kwargs):
        counter.calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", counted)
    return counter


def test_repeated_exact_basis_hits_and_preserves_all_projection_outputs(svd_count):
    retained = _retained_matrix()
    full = _full_matrix(retained)
    weights = _winv()
    source = _source()
    context = _context(retained)

    first_basis = polygon_projection.prepare(retained, weights, context)
    first = polygon_projection.apply(
        first_basis, source, retained, full, context
    )
    second_basis = polygon_projection.prepare(retained, weights, context)
    second = polygon_projection.apply(
        second_basis, source, retained, full, context
    )

    assert first_basis is second_basis
    assert svd_count.calls == 1
    for name in (
        "value",
        "reaction",
        "multiplier",
        "multiplier_residual",
        "retained_trace",
        "full_trace",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    cache = context._polygon_projection_cache
    assert (cache.hits, cache.misses, cache.builds, cache.evictions) == (
        1,
        1,
        1,
        0,
    )


def test_one_bit_density_change_misses_and_rebuilds(svd_count):
    retained = _retained_matrix()
    context = _context(retained)
    weights = _winv()
    polygon_projection.prepare(retained, weights, context)
    changed = weights.copy()
    changed[0] = np.nextafter(changed[0], np.inf)
    changed_basis = polygon_projection.prepare(retained, changed, context)

    assert svd_count.calls == 2
    assert changed_basis is not next(
        iter(context._polygon_projection_cache.entries.values())
    ).basis
    assert context._polygon_projection_cache.misses == 2


def test_mutating_caller_matrix_or_weight_array_causes_miss(svd_count):
    retained = _retained_matrix()
    full = _full_matrix(retained)
    context = _context(retained)
    weights = _winv()
    polygon_projection.prepare(retained, weights, context)

    retained[0, 0] = np.nextafter(retained[0, 0].real, np.inf)
    polygon_projection.prepare(retained, weights, context)
    assert svd_count.calls == 2

    weights[0] = np.nextafter(weights[0], np.inf)
    polygon_projection.prepare(retained, weights, context)
    assert svd_count.calls == 3
    assert np.all(np.isfinite(full))


@pytest.mark.parametrize(
    "change",
    ["rank", "backend", "samples", "operator", "selected_rows", "layout"],
)
def test_context_or_operator_metadata_change_misses(svd_count, change):
    retained = _retained_matrix()
    weights = _winv()
    context = _context(retained)
    polygon_projection.prepare(retained, weights, context)
    if change == "rank":
        context.p.rank_rtol = 1.0e-9
    elif change == "backend":
        context.compute = Backend("cpu")
    elif change == "samples":
        context.N += 1
    elif change == "operator":
        context.op = SimpleNamespace(
            matrix_layout_version="polygon_dual_v1",
            geometry_kind="polygon_dual_v1",
            selected_wall_rows=list(range(retained.shape[0])),
            T=retained.copy(),
        )
    elif change == "selected_rows":
        context.op.selected_wall_rows = list(reversed(range(retained.shape[0])))
    else:
        context.op.matrix_layout_version = "other_layout"
    polygon_projection.prepare(retained, weights, context)

    assert svd_count.calls == 2
    assert context._polygon_projection_cache.misses == 2


def test_cache_is_two_entry_lru_with_exact_byte_accounting(svd_count):
    retained = _retained_matrix()
    context = _context(retained)
    for offset in range(3):
        polygon_projection.prepare(
            retained,
            _winv(offset=offset),
            context,
        )

    cache = context._polygon_projection_cache
    assert svd_count.calls == 3
    assert len(cache.entries) == 2
    assert cache.capacity == 2
    assert cache.evictions == 1
    assert cache.actual_bytes == sum(
        entry.actual_bytes for entry in cache.entries.values()
    )
    for entry in cache.entries.values():
        expected = (
            entry.matrix_key.nbytes
            + entry.winv_key.nbytes
            + entry.basis.left.nbytes
            + entry.basis.singular.nbytes
            + entry.basis.right_h.nbytes
            + entry.basis.sqrt_winv.nbytes
        )
        assert entry.actual_bytes == expected


def test_evicted_equality_keys_are_collectable_while_basis_alive():
    retained = _retained_matrix()
    context = _context(retained)
    first_basis = polygon_projection.prepare(
        retained,
        _winv(offset=0),
        context,
    )
    first_entry = next(
        iter(context._polygon_projection_cache.entries.values())
    )
    matrix_key = weakref.ref(first_entry.matrix_key)
    winv_key = weakref.ref(first_entry.winv_key)
    assert not hasattr(first_basis, "matrix_key")
    assert not hasattr(first_basis, "winv_key")
    del first_entry

    second_basis = polygon_projection.prepare(
        retained,
        _winv(offset=1),
        context,
    )
    third_basis = polygon_projection.prepare(
        retained,
        _winv(offset=2),
        context,
    )
    assert len(context._polygon_projection_cache.entries) == 2
    assert context._polygon_projection_cache.evictions == 1
    gc.collect()
    assert matrix_key() is None
    assert winv_key() is None
    assert first_basis is not second_basis
    assert second_basis is not third_basis


def test_rank_failure_does_not_insert_basis(svd_count):
    context = _context(_retained_matrix())
    deficient = np.zeros((2, 6), dtype=np.complex128)
    deficient[:, 0] = 1.0
    with pytest.raises(Exception, match="rank"):
        polygon_projection.prepare(deficient, _winv(), context)

    cache = context._polygon_projection_cache
    assert svd_count.calls == 1
    assert cache.builds == 0
    assert cache.actual_bytes == 0
    assert len(cache.entries) == 0


def test_apply_does_not_mutate_basis_or_allow_cpu_alias_writes():
    retained = _retained_matrix()
    full = _full_matrix(retained)
    context = _context(retained)
    basis = polygon_projection.prepare(retained, _winv(), context)
    entry = next(iter(context._polygon_projection_cache.entries.values()))
    assert not hasattr(basis, "matrix_key")
    assert not hasattr(basis, "winv_key")
    saved = {
        f"basis.{name}": np.array(getattr(basis, name), copy=True)
        for name in ("left", "singular", "right_h", "sqrt_winv")
    }
    saved["entry.matrix_key"] = np.array(entry.matrix_key, copy=True)
    saved["entry.winv_key"] = np.array(entry.winv_key, copy=True)
    polygon_projection.apply(basis, _source(), retained, full, context)
    for label, expected in saved.items():
        owner, name = label.split(".")
        actual = getattr(basis if owner == "basis" else entry, name)
        np.testing.assert_array_equal(actual, expected)
        assert actual.flags.writeable is False
        with pytest.raises(ValueError):
            actual.flat[0] = 0.0


def test_fresh_context_is_uncached_oracle(svd_count):
    retained = _retained_matrix()
    full = _full_matrix(retained)
    weights = _winv()
    source = _source()
    first_context = _context(retained)
    oracle_context = _context(retained)
    cached = polygon_projection.project(
        source, retained, full, weights, first_context
    )
    oracle = polygon_projection.project(
        source, retained, full, weights, oracle_context
    )

    assert svd_count.calls == 2
    assert first_context._polygon_projection_cache is not (
        oracle_context._polygon_projection_cache
    )
    for name in ("value", "reaction", "multiplier", "multiplier_residual"):
        np.testing.assert_allclose(
            getattr(cached, name),
            getattr(oracle, name),
            rtol=0.0,
            atol=0.0,
        )


def test_shallow_context_copy_does_not_share_cache(svd_count):
    retained = _retained_matrix()
    weights = _winv()
    context = _context(retained)
    polygon_projection.prepare(retained, weights, context)
    copied = copy.copy(context)
    polygon_projection.prepare(retained, weights, copied)

    assert svd_count.calls == 2
    assert copied._polygon_projection_cache is not (
        context._polygon_projection_cache
    )


def test_core_peak_includes_two_entry_polygon_projection_bound(monkeypatch):
    radius = np.linspace(0.55, 1.45, 10)
    height = np.linspace(-0.45, 0.45, 10)
    geometry = build_polygon_geometry(
        radius,
        height,
        [(0.5, -0.5), (1.5, -0.5), (1.5, 0.5), (0.5, 0.5)],
    )
    operators = build_polygon_operators(geometry, 0)
    params = SimpleNamespace(
        K=0,
        electron_inertia=0,
        coupling="synchronous",
        micro_steps=1,
        waveform_max_iterations=1,
        ion_nodes=3,
        memory_limit=0,
        device_memory_limit=0,
    )
    context = SimpleNamespace(
        p=params,
        geometry=geometry,
        op=operators,
        compute=operators.compute,
        N=2,
    )
    rows, columns = operators.T.shape
    expected = 2 * (
        16 * rows * columns
        + 8 * columns
        + 16 * rows * rows
        + 8 * rows
        + 16 * rows * columns
        + 8 * columns
    )
    assert core.polygon_projection_cache_bytes(context) == expected

    original = core.polygon_projection_cache_bytes
    monkeypatch.setattr(core, "polygon_projection_cache_bytes", lambda _: 0)
    baseline = core.estimated_peak_bytes(context)
    monkeypatch.setattr(
        core,
        "polygon_projection_cache_bytes",
        lambda _: expected,
    )
    assert core.estimated_peak_bytes(context) - baseline == expected
    monkeypatch.setattr(core, "polygon_projection_cache_bytes", original)

    geometry.kind = "staircase_v1"
    operators.geometry_kind = "staircase_v1"
    assert core.polygon_projection_cache_bytes(context) == 0
