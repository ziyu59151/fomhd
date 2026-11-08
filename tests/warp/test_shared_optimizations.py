"""Shared massless storage and constraint-projection optimizations."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(
    os.environ.get("FOMHD_SHARED_SOURCE_ROOT", PROJECT_ROOT)
)
for path in (
    PROJECT_ROOT / "src",
    SOURCE_ROOT / "src",
    SOURCE_ROOT,
    SOURCE_ROOT / "tests" / "warp",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fomhd


_PACKAGE_OVERRIDE = PROJECT_ROOT / "src" / "fomhd"
if _PACKAGE_OVERRIDE.is_dir():
    override = str(_PACKAGE_OVERRIDE)
    if override not in fomhd.__path__:
        fomhd.__path__.insert(0, override)

from fomhd import io, massless
from fomhd.backend import Backend
from fomhd.core import estimated_peak_bytes
from fomhd.geometry import build_geometry, build_rectangle
from fomhd.operators import OperatorOptions, build_operators, validate_operators
from fomhd.types import Context, Failure
from massless_verification_fixture import _independent_constraint
from test_massless_support import (
    context as massless_context,
    parameters as massless_parameters,
    stage_tag,
    uniform_state,
)


def _ell():
    R = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    Z = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
    polygon = [
        (0.5, -0.3),
        (1.1, -0.3),
        (1.1, 0.0),
        (0.8, 0.0),
        (0.8, 0.4),
        (0.5, 0.4),
    ]
    return build_geometry(R, Z, polygon)


def _storage_bytes(value):
    if isinstance(value, list):
        return sum(_storage_bytes(item) for item in value)
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    if hasattr(value, "__dict__") and value.__class__.__name__ != "Backend":
        return sum(_storage_bytes(item) for item in vars(value).values())
    return 0


def _operator_payload_bytes(operators):
    return sum(
        _storage_bytes(value)
        for name, value in vars(operators).items()
        if name not in {"compute", "setup_checks"}
    )


def _massless_options(**changes):
    values = {
        "rank_rtol": 1e-10,
        "memory_cap_bytes": 512 << 20,
        "include_finite_mass": False,
    }
    values.update(changes)
    return OperatorOptions(**values)


@pytest.mark.parametrize("factory", [
    lambda: build_rectangle(4, 3, 0.5, -0.2, 0.1, 0.1),
    _ell,
], ids=["rectangle-real-wall", "stepped-real-wall"])
def test_selected_storage_preserves_shared_operators_and_skips_finite_work(
    factory,
):
    geometry = factory()
    full = build_operators(geometry, 1)
    reduced = build_operators(
        geometry,
        1,
        _massless_options(),
        full.compute,
    )

    assert full.finite_mass_available
    assert not reduced.finite_mass_available
    assert reduced.finite_mass_unavailable_reason
    assert reduced.omitted_finite_mass == (
        "Lphi",
        "phi_factor",
        "poisson_spectra",
        "Bc_full",
        "Bc",
        "selected_constraint_rows",
    )
    assert reduced.Lphi == []
    assert reduced.phi_factor == []
    assert reduced.poisson_spectra == []
    assert reduced.Bc_full.shape[0] == 0
    assert reduced.Bc.shape[0] == 0
    assert reduced.selected_constraint_rows == []

    for name in ("G", "C", "DB", "DJ", "Cdag"):
        for expected, actual in zip(getattr(full, name), getattr(reduced, name)):
            np.testing.assert_array_equal(actual, expected)
    for name in (
        "I",
        "H",
        "QB",
        "T_full",
        "T",
        "DR",
        "DZ",
        "topology_flux_row",
    ):
        np.testing.assert_array_equal(
            getattr(reduced, name), getattr(full, name)
        )
    assert reduced.selected_wall_rows == full.selected_wall_rows
    np.testing.assert_array_equal(reduced.M0, full.M0)
    np.testing.assert_array_equal(reduced.M1, full.M1)
    np.testing.assert_array_equal(reduced.MB, full.MB)

    expected_checks = [
        (
            check.name,
            check.mode,
            check.residual,
            check.scale,
            check.roundoff_bound,
        )
        for check in full.setup_checks
    ]
    actual_checks = [
        (
            check.name,
            check.mode,
            check.residual,
            check.scale,
            check.roundoff_bound,
        )
        for check in validate_operators(geometry, reduced)
    ]
    assert actual_checks == expected_checks

    full_bytes = _operator_payload_bytes(full)
    reduced_bytes = _operator_payload_bytes(reduced)
    assert full.selected_constraint_rows == list(
        range(full.Bc_full.shape[0])
    )
    assert reduced.finite_mass_omitted_bytes == full_bytes - reduced_bytes
    assert reduced.finite_mass_omitted_bytes > 0


def test_selected_storage_preflight_uses_only_retained_operators():
    from fomhd import operators as operators_module

    geometry = build_rectangle(4, 3, 0.5, -0.2, 0.1, 0.1)
    full_estimate = operators_module._estimate_operator_bytes(
        geometry, 1, include_finite_mass=True
    )
    reduced_estimate = operators_module._estimate_operator_bytes(
        geometry, 1, include_finite_mass=False
    )
    assert 0 < reduced_estimate < full_estimate

    reduced = build_operators(
        geometry,
        1,
        _massless_options(memory_cap_bytes=reduced_estimate),
    )
    assert not reduced.finite_mass_available
    with pytest.raises(Failure, match="operator_matrices memory budget"):
        build_operators(
            geometry,
            1,
            OperatorOptions(
                rank_rtol=1e-10,
                memory_cap_bytes=reduced_estimate,
                include_finite_mass=True,
            ),
        )


class _CountingBackend(Backend):
    def __init__(self):
        super().__init__("cpu")
        self.factor_calls = 0
        self.singular_value_calls = 0

    def factor(self, matrix):
        self.factor_calls += 1
        return super().factor(matrix)

    def singular_values(self, matrix):
        self.singular_value_calls += 1
        return super().singular_values(matrix)


def test_massless_storage_skips_finite_phi_factors_and_spectra():
    geometry = build_rectangle(4, 3, 0.5, -0.2, 0.1, 0.1)
    backend = _CountingBackend()
    build_operators(geometry, 1, backend=backend)
    full_factor_calls = backend.factor_calls
    full_singular_value_calls = backend.singular_value_calls
    before_factor_calls = backend.factor_calls
    before_singular_value_calls = backend.singular_value_calls
    build_operators(
        geometry,
        1,
        _massless_options(),
        backend,
    )
    reduced_factor_calls = backend.factor_calls - before_factor_calls
    reduced_singular_value_calls = (
        backend.singular_value_calls - before_singular_value_calls
    )
    assert full_factor_calls - reduced_factor_calls == 2 * 1 + 1
    assert (
        full_singular_value_calls - reduced_singular_value_calls
        >= 2 * 1 + 1
    )


def test_runtime_plan_preserves_finite_default_and_counts_massless_components():
    from fomhd import core

    finite = massless_context(K=1, hard_diagnostics=False)
    finite.p.electron_inertia = 1
    assert finite.op.finite_mass_available
    finite_plan = estimated_peak_bytes(finite)
    full_operators = finite.op

    # Same complete operator set, selected massless stage: only the finite
    # dense field system and finite-only resident inventory disappear.
    finite.p.electron_inertia = 0
    massless_full_plan = estimated_peak_bytes(finite)
    full_storage = core.massless_matrix_storage(finite)
    assert full_storage["finite_storage_oracle"] > 0
    assert massless_full_plan == 2_950_140

    reduced_operators = build_operators(
        finite.geometry,
        finite.p.K,
        _massless_options(),
        finite.compute,
    )
    massless = Context(
        p=finite.p,
        geometry=finite.geometry,
        op=reduced_operators,
        compute=finite.compute,
        N=finite.N,
        next_generation=2,
    )
    massless_plan = estimated_peak_bytes(massless)
    reduced_storage = core.massless_matrix_storage(massless)
    assert reduced_storage["finite_storage_oracle"] == 0
    assert (
        massless_full_plan - massless_plan
        == full_storage["finite_storage_oracle"]
    )

    # Hand-derived K1 rectangle fixture: M=3, Nc=9, Ne=33, Nb=40, Bc rows=27.
    assert finite_plan == 12_668_352
    assert massless_plan == 2_852_616

    massless.p.memory_limit = massless_plan - 1
    with pytest.raises(Failure, match="complete A0 peak storage plan"):
        core.preflight_memory(massless)
    massless.p.memory_limit = massless_plan
    assert core.preflight_memory(massless) == massless_plan


def _compare_stage_rhs(left, right):
    for species in range(2):
        for name in ("n", "mom", "perp", "parallel"):
            np.testing.assert_allclose(
                getattr(right.rhs.fluid[species], name),
                getattr(left.rhs.fluid[species], name),
                rtol=2e-13,
                atol=2e-15,
            )
    np.testing.assert_allclose(right.rhs.B, left.rhs.B, rtol=2e-13, atol=2e-15)
    for name in ("E", "ET"):
        np.testing.assert_allclose(
            getattr(right.cache, name),
            getattr(left.cache, name),
            rtol=2e-13,
            atol=2e-15,
        )
    left_records = left.cache.wall_correction.modes
    right_records = right.cache.wall_correction.modes
    assert len(right_records) == len(left_records)
    for actual, expected in zip(right_records, left_records):
        assert (
            actual.mode,
            actual.rank,
            actual.selected_rows,
            actual.gauge_rows,
        ) == (
            expected.mode,
            expected.rank,
            expected.selected_rows,
            expected.gauge_rows,
        )
        np.testing.assert_allclose(
            actual.full_residual,
            expected.full_residual,
            rtol=2e-13,
            atol=2e-15,
        )


def test_massless_and_full_operator_builds_give_same_stage_rhs_and_constraints():
    full_context = massless_context(K=1, hard_diagnostics=False)
    state = uniform_state(full_context)
    tag = stage_tag(state)
    reduced_operators = build_operators(
        full_context.geometry,
        full_context.p.K,
        _massless_options(),
        full_context.compute,
    )
    reduced_context = Context(
        p=full_context.p,
        geometry=full_context.geometry,
        op=reduced_operators,
        compute=full_context.compute,
        N=full_context.N,
        next_generation=2,
    )
    assert estimated_peak_bytes(reduced_context) < estimated_peak_bytes(
        full_context
    )

    full = massless.evaluate_stage(state, tag, full_context)
    reduced = massless.evaluate_stage(state, tag, reduced_context)
    _compare_stage_rhs(full, reduced)


def _projection_context():
    p = massless_parameters(K=1, hard_diagnostics=False)
    geometry = _ell()
    backend = Backend("cpu")
    operators = build_operators(geometry, p.K, backend=backend)
    return Context(
        p=p,
        geometry=geometry,
        op=operators,
        compute=backend,
        N=p.Nphi,
        next_generation=2,
    )


def _projection_values(context):
    rng = np.random.default_rng(19)
    nc = context.op.Nc
    entity_shape = (context.N, 3 * nc)
    source = rng.standard_normal(entity_shape) + 1j * rng.standard_normal(entity_shape)
    density = context.p.ref.ne * (
        0.8 + 0.4 * rng.random((context.N, nc))
    )
    return (
        np.asarray(source, dtype=np.complex128).reshape(-1),
        np.asarray(density, dtype=np.complex128).reshape(-1),
    )


def _count_singular_values(context):
    calls = {"count": 0}
    original = context.compute.singular_values

    def counted(matrix):
        calls["count"] += 1
        return original(matrix)

    context.compute.singular_values = counted
    return calls


def test_diagonal_projection_matches_dense_definition_without_svd_or_cached_t():
    context = _projection_context()
    source, density = _projection_values(context)
    full_T = np.asarray(context.op.T).copy()
    rows = [0, 1, 2, 3]
    context.op.T = context.compute.asarray(full_T[rows])
    context.op.T_full = context.compute.asarray(full_T[rows])
    calls = _count_singular_values(context)

    actual = massless._constraint_project(source, density, context)
    expected = _independent_constraint(source, density, context)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-13)
    assert actual.dtype == np.complex128
    assert calls["count"] == 0

    changed_rows = [4, 5, 7, 9]
    context.op.T = context.compute.asarray(full_T[changed_rows])
    context.op.T_full = context.compute.asarray(full_T[changed_rows])
    changed = massless._constraint_project(source, density, context)
    changed_expected = _independent_constraint(source, density, context)
    np.testing.assert_allclose(
        changed, changed_expected, rtol=2e-11, atol=2e-13
    )
    assert not np.allclose(changed, actual)
    assert calls["count"] == 0


def test_coupled_wall_rows_use_checked_dense_fallback():
    context = _projection_context()
    source, density = _projection_values(context)
    calls = _count_singular_values(context)
    actual = massless._constraint_project(source, density, context)
    expected = _independent_constraint(source, density, context)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-13)
    assert calls["count"] == context.N


def test_diagonal_projection_preserves_rank_defect_failure():
    context = _projection_context()
    source, density = _projection_values(context)
    full_T = np.asarray(context.op.T).copy()
    rows = [0, 1, 2, 3]
    context.op.T = context.compute.asarray(full_T[rows])
    context.op.T_full = context.compute.asarray(full_T[rows])
    density = density.reshape(context.N, context.op.Nc)
    density[:, 0] = 1e-20 * context.p.ref.ne
    density[:, 1] = 1e-20 * context.p.ref.ne

    with pytest.raises(Failure, match="deficient"):
        massless._constraint_project(
            source, density.reshape(-1), context
        )


def test_load_case_selects_storage_from_parsed_model_and_keeps_fingerprint(
    tmp_path,
):
    from test_model_routing import _context as routing_context
    from test_model_routing import _state as routing_state
    from test_model_routing import _write_case
    from fomhd.stage import closed_reference

    mass_context = routing_context(inertia=0, nsteps=0, hard=True)
    mass_state = routing_state(mass_context)
    mass_case_path = _write_case(
        tmp_path / "massless",
        mass_context.p,
        mass_context.geometry,
        mass_state,
        "rk4",
        0,
    )
    mass_case = io.load_case(mass_case_path)
    assert mass_case.operators.finite_mass_available is False
    first_fingerprint = mass_case.operator_hash
    mass_case.operators.Lphi.append(None)
    mass_case.operators.Bc = None
    mass_case.operators.Bc_full = None
    assert io._operator_fingerprint(mass_case.operators, mass_case.compute) == (
        first_fingerprint
    )

    finite_context = routing_context(inertia=1, nsteps=0, hard=True)
    finite_state = closed_reference(
        finite_context,
        finite_context.p.ref.ni,
        finite_context.p.ref.pressure,
        finite_context.p.ref.pressure,
        finite_context.p.ref.magnetic,
    )
    finite_case_path = _write_case(
        tmp_path / "finite",
        finite_context.p,
        finite_context.geometry,
        finite_state,
        "midpoint",
        0,
    )
    finite_case = io.load_case(finite_case_path)
    assert finite_case.operators.finite_mass_available is True
    assert len(finite_case.operators.Lphi) == 1
    assert finite_case.operators.Bc.shape[0] > 0
    finite_fingerprint = finite_case.operator_hash
    finite_case.operators.Lphi[0][0, 0] += 1.0
    assert io._operator_fingerprint(
        finite_case.operators, finite_case.compute
    ) != finite_fingerprint
    assert mass_case.semantic_hash != finite_case.semantic_hash


def test_preflight_rejects_reduced_storage_for_finite_model_before_mutation():
    from fomhd import core
    from fomhd.stage import evaluate_stage

    context = massless_context(K=1, hard_diagnostics=False)
    context.op = build_operators(
        context.geometry,
        context.p.K,
        _massless_options(),
        context.compute,
    )
    state = uniform_state(context)
    context.p.electron_inertia = 1
    counters = (
        context.next_generation,
        context.next_trial,
        context.next_source,
    )

    with pytest.raises(Failure, match="finite-mass electron model") as error:
        core.preflight_memory(context)
    assert error.value.code == "Model"
    assert (
        context.next_generation,
        context.next_trial,
        context.next_source,
    ) == counters

    with pytest.raises(Failure, match="finite-mass electron model") as error:
        evaluate_stage(state, stage_tag(state), context)
    assert error.value.code == "Model"
    assert (
        context.next_generation,
        context.next_trial,
        context.next_source,
    ) == counters

    finite = massless_context(K=1, hard_diagnostics=False)
    finite.p.electron_inertia = 1
    assert finite.op.finite_mass_available
    assert core.preflight_memory(finite) == 12_668_352


class _ShapeOnlyArray:
    def __init__(self, shape, itemsize=16):
        self.shape = tuple(shape)
        self.itemsize = itemsize
        self.size = int(np.prod(self.shape, dtype=np.int64))
        self.nbytes = self.size * self.itemsize


def _east32_shape_operators():
    modes = 3
    nc, nr, nz, nv, ne, nb, walls = 796, 828, 828, 861, 2452, 2517, 128
    complex_matrix = lambda rows, columns: _ShapeOnlyArray((rows, columns))
    real_vector = lambda size: _ShapeOnlyArray((size,), itemsize=8)
    integer_vector = lambda size: _ShapeOnlyArray((size,), itemsize=4)
    per_mode = {
        "G": complex_matrix(ne, nc),
        "C": complex_matrix(nb, ne),
        "DB": complex_matrix(nv, nb),
        "DJ": complex_matrix(nc, ne),
        "Cdag": complex_matrix(ne, nb),
    }
    operators = SimpleNamespace(
        K=1,
        Nc=nc,
        Ne=ne,
        Nb=nb,
        **{
            name: [value] * modes
            for name, value in per_mode.items()
        },
        I=complex_matrix(ne, 3 * nc),
        H=complex_matrix(3 * nc, ne),
        QB=complex_matrix(3 * nc, nb),
        T_full=complex_matrix(walls, 3 * nc),
        T=complex_matrix(walls, 3 * nc),
        DR=complex_matrix(nc, nr),
        DZ=complex_matrix(nc, nz),
        topology_flux_row=complex_matrix(1, nb),
        PR_r=complex_matrix(nr, nr),
        PZ_r=complex_matrix(nr, nr),
        PR_z=complex_matrix(nz, nz),
        PZ_z=complex_matrix(nz, nz),
        PR_v=complex_matrix(nv, nv),
        PZ_v=complex_matrix(nv, nv),
        b1r=complex_matrix(nr, nv),
        b1z=complex_matrix(nz, nv),
        b2r=complex_matrix(nc, nr),
        b2z=complex_matrix(nc, nz),
        M0=real_vector(nc),
        M1=real_vector(ne),
        MB=real_vector(nb),
        Rbar_r=real_vector(nr),
        Rbar_v=real_vector(nv),
        Lphi=[],
        phi_factor=[],
        poisson_spectra=[],
        Bc_full=complex_matrix(0, modes * ne),
        Bc=complex_matrix(0, modes * ne),
        finite_mass_available=False,
    )
    for name in ("PR_r", "PZ_r", "PR_z", "PZ_z", "PR_v", "PZ_v"):
        matrix = getattr(operators, name)
        setattr(
            operators,
            f"{name}_factor",
            SimpleNamespace(
                original=matrix,
                factors=complex_matrix(*matrix.shape),
                pivots=integer_vector(matrix.shape[0]),
            ),
        )
    return operators


def test_massless_plan_includes_production_diagnostic_matrix_lifetimes():
    from fomhd import core

    c = massless_context(K=1, hard_diagnostics=False)
    c.op = _east32_shape_operators()
    c.geometry = SimpleNamespace(walls=[None] * 128)
    storage = core.massless_matrix_storage(c)
    # 8-byte abs(I), 16-byte complex128 Cdag stack, 8-byte abs(Cdag).
    assert storage["diagnostic_assembly_bytes"] == 491_204_256
    assert storage["temporary_matrices"] >= 491_204_256


def test_massless_plan_uses_owned_inventory_and_bounded_live_temporaries():
    from fomhd import core
    from fomhd.wall_cache import get_wall_cache

    context = massless_context(K=1, hard_diagnostics=False)
    operators = build_operators(
        context.geometry,
        context.p.K,
        _massless_options(),
        context.compute,
    )
    context.op = operators
    storage = core.massless_matrix_storage(context)
    fixed = (
        operators.I,
        operators.H,
        operators.QB,
        operators.T_full,
        operators.T,
        operators.DR,
        operators.DZ,
        operators.topology_flux_row,
        operators.b1r,
        operators.b1z,
        operators.b2r,
        operators.b2z,
    )
    filters = (
        operators.PR_r,
        operators.PZ_r,
        operators.PR_z,
        operators.PZ_z,
        operators.PR_v,
        operators.PZ_v,
    )
    metrics = (
        operators.M0,
        operators.M1,
        operators.MB,
        operators.Rbar_r,
        operators.Rbar_v,
    )
    factors = tuple(
        getattr(operators, f"{name}_factor") for name in (
            "PR_r",
            "PZ_r",
            "PR_z",
            "PZ_z",
            "PR_v",
            "PZ_v",
        )
    )
    assert storage["mode_matrices"] == sum(
        matrix.nbytes
        for name in ("G", "C", "DB", "DJ", "Cdag")
        for matrix in getattr(operators, name)
    )
    assert storage["fixed_matrices"] == sum(matrix.nbytes for matrix in fixed)
    assert storage["filter_matrices"] == sum(matrix.nbytes for matrix in filters)
    assert storage["metric_vectors"] == sum(matrix.nbytes for matrix in metrics)
    assert storage["filter_factors"] == sum(
        value.nbytes
        for factor in factors
        for value in (factor.original, factor.factors, factor.pivots)
    )
    assert storage["finite_storage_oracle"] == 0
    assert storage["persistent"] == sum(
        storage[name]
        for name in (
            "mode_matrices",
            "fixed_matrices",
            "filter_matrices",
            "metric_vectors",
            "filter_factors",
            "finite_storage_oracle",
        )
    )
    nr = len(context.geometry.r_faces.keys)
    nz = len(context.geometry.z_faces.keys)
    nv = len(context.geometry.vertices.keys)
    walls = len(context.geometry.walls)
    wall_entries = (
        operators.Nb * walls
        + operators.Ne * walls
        + 5 * walls * walls
    )
    projection_entries = (
        2 * operators.T.shape[0] * 3 * operators.Nc
        + operators.T.shape[0] ** 2
    )
    assert storage["temporary_matrices"] == (
        max(16 * max(wall_entries, projection_entries),
            8 * operators.I.size + 24 * sum(x.size for x in operators.Cdag),
            8 * (operators.I.size + sum(x.size for x in operators.Cdag) + operators.T_full.size),
            16 * (operators.I.size + operators.DR.size + operators.DZ.size))
    )
    wall_cache = get_wall_cache(context)
    for stored, cached in enumerate(wall_cache.modes):
        assert cached.C is operators.C[stored]
        assert cached.Cdag is operators.Cdag[stored]
    assert core.massless_matrix_storage(context) == storage

    scale = massless_context(K=1, hard_diagnostics=False)
    scale.p.electron_inertia = 0
    scale.p.Nphi = 5
    scale.op = _east32_shape_operators()
    scale.geometry = SimpleNamespace(walls=[None] * 128)
    scaled = core.massless_matrix_storage(scale)
    assert scaled["persistent"] == 1_445_112_760
    assert scaled["temporary_matrices"] == 491_204_256
    assert core.estimated_peak_bytes(scale) == 2_201_808_984
    assert core.estimated_peak_bytes(scale) < 8 << 30
    scale.p.coupling = "predictor"
    scale.p.micro_steps = 4
    assert core.estimated_peak_bytes(scale) == 5_075_239_512
    assert core.estimated_peak_bytes(scale) < 8 << 30

    oversized = _east32_shape_operators()
    oversized.G = [_ShapeOnlyArray((sys.maxsize, 2))] * 3
    scale.op = oversized
    with pytest.raises(Failure, match="overflows"):
        core.massless_matrix_storage(scale)


def test_massless_plan_rejects_incomplete_resident_mode_inventory():
    from fomhd import core

    context = massless_context(K=1, hard_diagnostics=False)
    context.op = build_operators(
        context.geometry,
        context.p.K,
        _massless_options(),
        context.compute,
    )
    context.op.G.pop()

    with pytest.raises(Failure, match="resident operator mode count"):
        core.massless_matrix_storage(context)


def test_restart_rejects_actual_full_storage_massless_fingerprint(tmp_path):
    from test_model_routing import (
        _context as routing_context,
        _cumulative,
        _latest_checkpoint,
        _state,
        _write_case,
    )
    from fomhd.__main__ import _context as command_context
    from fomhd.stage import initialize

    context = routing_context(inertia=0, nsteps=0, hard=True)
    state = _state(context)
    case_path = _write_case(
        tmp_path / "massless-storage-boundary",
        context.p,
        context.geometry,
        state,
        "rk4",
        0,
    )
    current = io.load_case(case_path)
    assert current.operators.finite_mass_available is False

    full_operators = build_operators(
        current.geometry,
        current.parameters.K,
        backend=current.compute,
    )
    full_fingerprint = io._operator_fingerprint(
        full_operators, current.compute
    )
    assert full_fingerprint != current.operator_hash

    legacy = copy.copy(current)
    legacy.output = tmp_path / "legacy-full-storage-run"
    legacy.operators = full_operators
    legacy.operator_hash = full_fingerprint
    legacy_context = command_context(legacy)
    cumulative = _cumulative()
    committed = initialize(
        state,
        legacy_context,
        cumulative,
        False,
    )
    legacy_run = io.Run(legacy)
    try:
        legacy_run.checkpoint(committed, cumulative, legacy_context)
        legacy_checkpoint = _latest_checkpoint(legacy_run.path())
    finally:
        legacy_run.close()

    current.output = tmp_path / "current-reduced-run"
    current_context = command_context(current)
    current_cumulative = _cumulative()
    current_committed = initialize(
        state,
        current_context,
        current_cumulative,
        False,
    )
    current_run = io.Run(current)
    try:
        current_run.checkpoint(
            current_committed,
            current_cumulative,
            current_context,
        )
        current_checkpoint = _latest_checkpoint(current_run.path())
    finally:
        current_run.close()
    assert io.read_restart(current_checkpoint, current).state.step == 0

    with pytest.raises(
        ValueError,
        match="operator storage.*pre-0.5 full-storage.*incompatible",
    ):
        io.read_restart(legacy_checkpoint, current)
