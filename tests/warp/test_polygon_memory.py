"""Independent polygon persistent-storage and preflight tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fomhd import core
from fomhd.backend import Backend
from fomhd.geometry import build_rectangle
from fomhd.operators import build_operators
from fomhd.polygon_geometry import build_polygon_geometry
from fomhd.polygon_operators import build_polygon_operators
from fomhd.types import Failure


_POLYGON_ARRAYS = (
    "S",
    "scalar_I",
    "Dface",
    "face_normals_backend",
    "cell_radius_backend",
    "curvature_radius",
)


def _polygon_context(K=0):
    radius = np.linspace(0.55, 1.45, 10)
    axial = np.linspace(-0.45, 0.45, 10)
    polygon = [
        (0.5, -0.5),
        (1.5, -0.5),
        (1.5, 0.5),
        (0.5, 0.5),
    ]
    geometry = build_polygon_geometry(radius, axial, polygon)
    operators = build_polygon_operators(
        geometry, K, backend=Backend("cpu")
    )
    parameters = SimpleNamespace(
        K=K,
        electron_inertia=0,
        coupling="synchronous",
        micro_steps=1,
        waveform_max_iterations=1,
        ion_nodes=3,
        memory_limit=0,
        device_memory_limit=0,
    )
    return SimpleNamespace(
        p=parameters,
        geometry=geometry,
        op=operators,
        compute=Backend("cpu"),
        N=8 * K + 5,
    )


def _csr_bytes(matrix):
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes


def _without_array(context, name):
    operators = SimpleNamespace(**vars(context.op))
    delattr(operators, name)
    return SimpleNamespace(
        p=context.p,
        geometry=context.geometry,
        op=operators,
        compute=context.compute,
        N=context.N,
    )


def test_polygon_inventory_covers_every_named_buffer_and_csr():
    context = _polygon_context()
    storage = core.massless_matrix_storage(context)

    expected_named = 0
    for name in _POLYGON_ARRAYS:
        actual = storage[f"polygon_{name}_bytes"]
        expected = getattr(context.op, name).nbytes
        assert actual == expected
        expected_named += expected
    expected_csr = _csr_bytes(context.geometry.b1) + _csr_bytes(
        context.geometry.b2
    )

    assert storage["polygon_operator_array_bytes"] == expected_named
    assert storage["polygon_csr_incidence_bytes"] == expected_csr
    assert storage["polygon_csr_nnz_entries"] == (
        context.geometry.b1.nnz + context.geometry.b2.nnz
    )
    assert storage["polygon_persistent"] == expected_named + expected_csr
    assert storage["persistent"] == (
        storage["mode_matrices"]
        + storage["fixed_matrices"]
        + storage["filter_matrices"]
        + storage["metric_vectors"]
        + storage["filter_factors"]
        + storage["finite_storage_oracle"]
        + storage["polygon_persistent"]
    )


def test_each_polygon_array_addition_is_covered_exactly():
    context = _polygon_context()
    full = core.massless_matrix_storage(context)

    for name in _POLYGON_ARRAYS:
        reduced = core.massless_matrix_storage(
            _without_array(context, name)
        )
        assert full["polygon_persistent"] - reduced[
            "polygon_persistent"
        ] == getattr(context.op, name).nbytes


def test_polygon_alias_duplicate_is_counted_and_disclosed():
    context = _polygon_context()
    assert np.shares_memory(context.op.Dface, context.op.DR)
    assert not np.shares_memory(context.op.S, context.op.scalar_I)
    assert np.shares_memory(
        context.op.curvature_radius,
        context.op.cell_radius_backend,
    )

    storage = core.massless_matrix_storage(context)

    assert storage["polygon_alias_duplicate_bytes"] == (
        context.op.Dface.nbytes
        + context.op.cell_radius_backend.nbytes
    )


def test_polygon_nnz_csr_is_counted_when_present():
    context = _polygon_context()
    baseline = core.massless_matrix_storage(context)
    nnz = np.asarray(
        context.geometry.b1.nnz + context.geometry.b2.nnz,
        dtype=np.int64,
    )
    context.op.nnzCSR = nnz

    storage = core.massless_matrix_storage(context)

    assert storage["polygon_nnzCSR_entries"] == int(nnz)
    assert storage["polygon_nnzCSR_bytes"] == nnz.nbytes
    assert storage["polygon_persistent"] == (
        baseline["polygon_persistent"] + nnz.nbytes
    )


def test_polygon_capacity_rejects_before_state_allocation():
    context = _polygon_context()
    peak = core.estimated_peak_bytes(context)
    context.p.memory_limit = peak - 1
    allocations = []

    class NoAllocationBackend:
        name = "cpu"
        xp = np

        def array(self, value, dtype=None):
            allocations.append((value, dtype))
            raise AssertionError("state allocation reached after cap rejection")

    context.compute = NoAllocationBackend()
    with pytest.raises(Failure, match="memory budget|Memory"):
        core.allocate_state(context)

    assert allocations == []
    context.p.memory_limit = peak
    assert core.preflight_memory(context) == peak


def test_staircase_inventory_is_bitwise_unchanged():
    backend = Backend("cpu")
    geometry = build_rectangle(3, 3, 1.0, 0.0, 0.5, 0.5)
    operators = build_operators(geometry, 0, backend=backend)
    context = SimpleNamespace(
        p=SimpleNamespace(K=0, electron_inertia=0),
        geometry=geometry,
        op=operators,
        compute=backend,
        N=5,
    )

    assert core.massless_matrix_storage(context) == {
        "mode_matrices": 15824,
        "fixed_matrices": 16848,
        "filter_matrices": 4896,
        "metric_vectors": 448,
        "filter_factors": 9960,
        "finite_storage_oracle": 2864,
        "persistent": 50840,
        "diagnostic_assembly_bytes": 9600,
        "diagnostic_retained_bytes": 4992,
        "adjoint_temporary_bytes": 3840,
        "temporary_matrices": 9856,
    }
