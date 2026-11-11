"""Fixed closed-wall systems for the massless electron branch."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from .types import Failure


@dataclass(slots=True)
class CachedWallMode:
    """Immutable fixed factors for one retained Fourier mode."""

    stored: int
    mode: int
    C: Any
    Cdag: Any
    extraction: Any
    weighted_adjoint: Any
    matrix: Any
    selected_rows: tuple[int, ...]
    gauge_rows: tuple[int, ...]
    prepared: SimpleNamespace | None


@dataclass(slots=True)
class WallSystemCache:
    """Context-attached fixed systems, shareable by same-scope context copies."""

    key: tuple[Any, ...]
    backend: Any
    modes: tuple[CachedWallMode, ...]
    setup_count: int
    solves: int
    evaluations: int
    storage_bytes: int
    storage_breakdown: dict[str, int]
    storage_formula: str

    @property
    def extraction(self) -> list[Any]:
        return [mode.extraction for mode in self.modes]


def _maximum(value: Any, backend: Any) -> float:
    return (
        float(backend.xp.max(backend.xp.abs(value)).item())
        if value.size
        else 0.0
    )


def _signed_wall_extraction(context, backend):
    xp = backend.xp
    nc = context.op.Nc
    nr = len(context.geometry.r_faces.keys)
    result = xp.zeros(
        (len(context.geometry.walls), context.op.Ne),
        dtype=xp.complex128,
    )
    for row, wall in enumerate(context.geometry.walls):
        edge = (
            wall.face
            if wall.component == "r_face"
            else nr + nc + wall.face
        )
        result[row, edge] = float(wall.sign)
    return backend.asarray(result)


def _cache_key(context) -> tuple[Any, ...]:
    operation = context.op
    backend = context.compute
    reference = context.p.ref
    identity_hash = (
        getattr(context, "operator_hash", "")
        or getattr(operation, "matrix_fingerprint", "")
    )
    return (
        id(operation),
        id(context.geometry),
        id(backend),
        str(getattr(backend, "name", "")),
        str(getattr(backend, "device", "")),
        id(context.p),
        id(reference),
        float(reference.length).hex(),
        float(reference.electric).hex(),
        float(context.p.rank_rtol).hex(),
        int(context.op.K),
        int(context.op.Ne),
        int(context.op.Nb),
        int(context.op.Nc),
        str(identity_hash),
        str(getattr(context, "rule_hash", "")),
        str(getattr(context, "geometry_hash", "")),
    )


def _add_bytes(breakdown: dict[str, int], name: str, value: Any) -> None:
    if value is None:
        return
    size = int(getattr(value, "nbytes", 0))
    breakdown[name] = breakdown.get(name, 0) + size


def _mode_storage(mode: CachedWallMode) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    _add_bytes(breakdown, "extraction", mode.extraction)
    _add_bytes(breakdown, "weighted_adjoint", mode.weighted_adjoint)
    _add_bytes(breakdown, "Aw", mode.matrix)
    prepared = mode.prepared
    if prepared is None:
        return breakdown
    _add_bytes(breakdown, "principal", prepared.original)
    _add_bytes(breakdown, "scaled_principal", prepared.scaled_matrix)
    _add_bytes(breakdown, "row_scale", prepared.row_scale)
    _add_bytes(breakdown, "column_scale", prepared.column_scale)
    _add_bytes(breakdown, "singular_values", prepared.singular_values)
    factor = prepared.factor
    _add_bytes(breakdown, "factor_original", factor.original)
    _add_bytes(breakdown, "LU_factors", factor.factors)
    _add_bytes(breakdown, "LU_pivots", factor.pivots)
    return breakdown


def _build_wall_cache(context) -> WallSystemCache:
    from .core import preflight_memory

    preflight_memory(context)
    backend = context.compute
    xp = backend.xp
    m1 = backend.asarray(
        xp.real(xp.asarray(context.op.M1)).astype(xp.float64),
        xp.float64,
    )
    length = float(context.p.ref.length)
    electric = float(context.p.ref.electric)
    if not math.isfinite(length) or length <= 0.0:
        raise Failure("Reference", "wall cache length reference must be positive")
    if not math.isfinite(electric) or electric <= 0.0:
        raise Failure("Reference", "wall cache electric reference must be positive")

    cached_modes = []
    index_bytes = 0
    storage_breakdown: dict[str, int] = {}
    for stored in range(2 * context.p.K + 1):
        extraction = _signed_wall_extraction(context, backend)
        weighted_adjoint = backend.asarray(
            xp.conj(extraction.T) / m1[:, None]
        )
        curl_boundary = backend.mm(
            backend.asarray(context.op.C[stored], xp.complex128),
            weighted_adjoint,
        )
        thin = backend.mm(
            backend.asarray(context.op.Cdag[stored], xp.complex128),
            curl_boundary,
        )
        matrix = backend.mm(extraction, thin)
        hermitian_scale = max(_maximum(matrix, backend), 1.0e-300)
        if _maximum(matrix - xp.conj(matrix.T), backend) > (
            4096.0 * sys.float_info.epsilon * hermitian_scale
        ):
            raise Failure("WallField", "wall correction matrix is not Hermitian")
        rows = tuple(
            int(row)
            for row in backend.independent_rows(
                matrix, context.p.rank_rtol
            )
        )
        selected_set = set(rows)
        gauge_rows = tuple(
            row for row in range(matrix.shape[0]) if row not in selected_set
        )
        prepared = None
        if rows:
            selected = backend.asarray(rows, xp.int64)
            principal = matrix[xp.ix_(selected, selected)]
            row_scale = xp.full(
                (len(rows),), electric / length**3, dtype=xp.float64
            )
            column_scale = xp.full(
                (len(rows),), electric * length**2, dtype=xp.float64
            )
            prepared = backend.prepare_linear(
                principal,
                row_scale,
                column_scale,
                context.p.rank_rtol,
                128.0 * max(len(rows), 1) * xp.finfo(xp.float64).eps,
            )
        mode = CachedWallMode(
            stored=stored,
            mode=stored - context.p.K,
            C=backend.asarray(context.op.C[stored], xp.complex128),
            Cdag=backend.asarray(context.op.Cdag[stored], xp.complex128),
            extraction=extraction,
            weighted_adjoint=weighted_adjoint,
            matrix=matrix,
            selected_rows=rows,
            gauge_rows=gauge_rows,
            prepared=prepared,
        )
        cached_modes.append(mode)
        for name, byte_count in _mode_storage(mode).items():
            storage_breakdown[name] = (
                storage_breakdown.get(name, 0) + byte_count
            )
        index_bytes += 8 * (len(rows) + len(gauge_rows))
    storage_breakdown["selected_gauge_indices"] = index_bytes
    return WallSystemCache(
        key=_cache_key(context),
        backend=backend,
        modes=tuple(cached_modes),
        setup_count=1,
        solves=0,
        evaluations=0,
        storage_bytes=sum(storage_breakdown.values()),
        storage_breakdown=storage_breakdown,
        storage_formula=(
            "sum per-mode S+weighted_adjoint+Aw"
            "+(principal+scaled_principal+factor_original+LU_factors)"
            "+row_scale+column_scale+singular_values+LU_pivots"
            "+8*(selected+gauge); borrowed C/Cdag and transient RHS vectors"
            " excluded"
        ),
    )


def get_wall_cache(context) -> WallSystemCache:
    """Return the identity-matched cache, building it once when necessary."""
    key = _cache_key(context)
    cache = getattr(context, "_wall_system_cache", None)
    if cache is not None and cache.key == key:
        return cache
    try:
        cache = _build_wall_cache(context)
    except Failure:
        raise
    except Exception as error:
        raise Failure(
            "NumericalLibrary",
            f"wall-system cache setup failed: {error}",
        ) from error
    context._wall_system_cache = cache
    context._wall_system_cache_builds = (
        int(getattr(context, "_wall_system_cache_builds", 0)) + 1
    )
    return cache


def apply_wall_cache(cache: WallSystemCache, e0):
    """Apply every cached fixed system to one retained electric field."""
    backend = cache.backend
    xp = backend.xp
    ne = int(cache.modes[0].extraction.shape[1]) if cache.modes else 0
    expected = len(cache.modes) * ne
    if not isinstance(e0, xp.ndarray):
        raise Failure("Backend", "wall correction right-hand side is not backend-owned")
    if e0.ndim != 1 or tuple(e0.shape) != (expected,):
        raise Failure("Shape", "wall correction right-hand side has invalid extent")
    if e0.dtype != xp.complex128:
        raise Failure("Backend", "wall correction right-hand side must be complex128")
    e0 = backend.asarray(e0, xp.complex128)
    if not bool(xp.all(xp.isfinite(e0)).item()):
        raise Failure(
            "Finite", "wall correction right-hand side contains non-finite values"
        )

    records = []
    correction = xp.zeros(expected, dtype=xp.complex128)
    for cached in cache.modes:
        source = e0[
            cached.stored * ne : (cached.stored + 1) * ne
        ]
        curled = backend.mv(cached.C, source)
        curl_adjoint = backend.mv(cached.Cdag, curled)
        rhs = backend.asarray(-backend.mv(cached.extraction, curl_adjoint))
        multiplier = xp.zeros(rhs.shape[0], dtype=xp.complex128)
        linear = None
        if cached.prepared is not None:
            selected = backend.asarray(cached.selected_rows, xp.int64)
            linear = backend.linear_prepared(
                cached.prepared,
                rhs[selected],
            )
            multiplier[selected] = linear.x
        full_residual = backend.asarray(
            backend.mv(cached.matrix, multiplier) - rhs
        )
        rhs_scale = max(_maximum(rhs, backend), 1.0e-300)
        if _maximum(full_residual, backend) > 1.0e-8 * rhs_scale:
            raise Failure(
                "WallField",
                "constrained electric-field correction fails an original wall row",
            )
        correction[
            cached.stored * ne : (cached.stored + 1) * ne
        ] = backend.mv(cached.weighted_adjoint, multiplier)
        records.append(
            SimpleNamespace(
                mode=cached.mode,
                rank=len(cached.selected_rows),
                selected_rows=list(cached.selected_rows),
                gauge_rows=list(cached.gauge_rows),
                gauge="omitted principal multipliers fixed to zero",
                rhs=rhs,
                multiplier=multiplier,
                full_residual=full_residual,
                linear=linear,
            )
        )
    cache.evaluations += 1
    cache.solves += len(cache.modes)
    return backend.asarray(correction), SimpleNamespace(
        modes=records,
        rank=sum(record.rank for record in records),
        extraction=cache.extraction,
        cache=cache,
    )
