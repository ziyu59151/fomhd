"""Weighted thin-SVD wall projection for ``polygon_dual_v1``."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from types import SimpleNamespace

from .types import Failure


_CACHE_ATTR = "_polygon_projection_cache"
_CACHE_CAPACITY = 2


@dataclass
class _CacheEntry:
    backend: object
    device: object
    stream: object
    operator: object
    layout: tuple[object, ...]
    samples: int
    rank_rtol: float
    selected_rows: tuple[int, ...]
    shape: tuple[int, int]
    matrix_key: object
    winv_key: object
    actual_bytes: int
    basis: SimpleNamespace


class _ProjectionCache:
    def __init__(self, owner_context) -> None:
        self.owner = owner_context
        self.capacity = _CACHE_CAPACITY
        self.entries: OrderedDict[int, _CacheEntry] = OrderedDict()
        self.next_token = 0
        self.hits = 0
        self.misses = 0
        self.builds = 0
        self.evictions = 0
        self.invalidations = 0

    @property
    def actual_bytes(self) -> int:
        return sum(entry.actual_bytes for entry in self.entries.values())


def _xp(context):
    return context.compute.xp


def _cache(context) -> _ProjectionCache:
    cache = getattr(context, _CACHE_ATTR, None)
    if not isinstance(cache, _ProjectionCache) or cache.owner is not context:
        cache = _ProjectionCache(context)
        setattr(context, _CACHE_ATTR, cache)
    return cache


def cache_status(context):
    """Return read-only cache counters and exact owned-array bytes."""
    cache = _cache(context)
    return SimpleNamespace(
        capacity=cache.capacity,
        size=len(cache.entries),
        hits=cache.hits,
        misses=cache.misses,
        builds=cache.builds,
        evictions=cache.evictions,
        invalidations=cache.invalidations,
        actual_bytes=cache.actual_bytes,
    )


def _stream_identity(context):
    backend = context.compute
    if backend.name == "cpu":
        return None
    return int(backend.xp.cuda.get_current_stream().ptr)


def _layout_identity(context) -> tuple[object, ...]:
    return (
        getattr(context.op, "matrix_layout_version", None),
        getattr(context.op, "geometry_kind", None),
        getattr(getattr(context, "geometry", None), "kind", None),
        getattr(context.op, "Nc", None),
        getattr(context.op, "Ne", None),
        getattr(context.op, "Nf", None),
    )


def _selected_rows(context) -> tuple[int, ...]:
    return tuple(int(row) for row in getattr(context.op, "selected_wall_rows", ()))


def _metadata_matches(entry: _CacheEntry, context, matrix, rank_rtol, selected_rows):
    return (
        entry.backend is context.compute
        and entry.device == getattr(context.compute, "device", None)
        and entry.stream == _stream_identity(context)
        and entry.operator is context.op
        and entry.layout == _layout_identity(context)
        and entry.samples == context.N
        and entry.rank_rtol == rank_rtol
        and entry.selected_rows == selected_rows
        and entry.shape == matrix.shape
    )


def _payload_matches(entry: _CacheEntry, matrix, winv, xp) -> bool:
    return bool(xp.array_equal(entry.matrix_key, matrix)) and bool(
        xp.array_equal(entry.winv_key, winv)
    )


def _protect_cpu_arrays(backend, basis) -> None:
    if backend.name != "cpu":
        return
    for name in (
        "left",
        "singular",
        "right_h",
        "sqrt_winv",
    ):
        getattr(basis, name).flags.writeable = False


def _protect_cpu_key(backend, value) -> None:
    if backend.name == "cpu":
        value.flags.writeable = False


def _basis_context_matches(basis, context) -> bool:
    return (
        basis.backend is context.compute
        and basis.device == getattr(context.compute, "device", None)
        and basis.stream == _stream_identity(context)
        and basis.operator is context.op
        and basis.layout == _layout_identity(context)
        and basis.samples == context.N
        and basis.rank_rtol == float(context.p.rank_rtol)
        and basis.selected_rows == _selected_rows(context)
    )


def _maximum(value, xp) -> float:
    return (
        float(xp.max(xp.abs(value)).item())
        if value.size
        else 0.0
    )


def _prepare_uncached(matrix, winv, context, *, label="wall"):
    """Prepare the full retained-row weighted thin SVD basis."""
    xp = _xp(context)
    matrix = xp.asarray(matrix, dtype=xp.complex128)
    winv = xp.asarray(winv, dtype=xp.float64)
    if matrix.ndim != 2 or winv.ndim != 1:
        raise Failure("Shape", f"{label} projection dimensions are invalid")
    rows, columns = matrix.shape
    if winv.size != columns:
        raise Failure("Shape", f"{label} projection weight extent mismatch")
    if not bool(xp.all(xp.isfinite(matrix)).item()):
        raise Failure("Finite", f"{label} projection matrix is nonfinite")
    if not bool(
        xp.all(xp.isfinite(winv) & (winv > 0.0)).item()
    ):
        raise Failure("Domain", f"{label} projection weights are not positive")
    sqrt_winv = xp.sqrt(winv)
    if rows:
        weighted = matrix * sqrt_winv[None, :]
        if not bool(xp.all(xp.isfinite(weighted)).item()):
            raise Failure("Finite", f"{label} weighted matrix is nonfinite")
        left, singular, right_h = xp.linalg.svd(
            weighted,
            full_matrices=False,
        )
    else:
        left = xp.zeros((0, 0), dtype=xp.complex128)
        singular = xp.zeros((0,), dtype=xp.float64)
        right_h = xp.zeros((0, columns), dtype=xp.complex128)
    if singular.size != rows:
        raise Failure(
            "WallRank",
            f"{label} retained constraints do not have full row rank",
        )
    if rows:
        largest = float(singular[0].item())
        smallest = float(singular[-1].item())
        threshold = float(context.p.rank_rtol) * largest
        if (
            not math.isfinite(largest)
            or not math.isfinite(smallest)
            or largest <= 0.0
            or smallest <= threshold
        ):
            raise Failure(
                "WallRank",
                f"{label} retained constraints are rank deficient: "
                f"sigma={smallest:.17g}, threshold={threshold:.17g}",
            )
        condition = largest / smallest
    else:
        condition = 1.0
    return SimpleNamespace(
        left=left,
        singular=singular,
        right_h=right_h,
        sqrt_winv=sqrt_winv,
        rows=rows,
        columns=columns,
        rank=rows,
        condition=condition,
    )


def prepare(matrix, winv, context, *, label="wall"):
    """Return the exact context-local prepared basis, building on a miss."""
    xp = _xp(context)
    matrix = xp.asarray(matrix, dtype=xp.complex128)
    winv = xp.asarray(winv, dtype=xp.float64)
    if matrix.ndim != 2 or winv.ndim != 1:
        raise Failure("Shape", f"{label} projection dimensions are invalid")
    rows, columns = matrix.shape
    if winv.size != columns:
        raise Failure("Shape", f"{label} projection weight extent mismatch")
    if not bool(xp.all(xp.isfinite(matrix)).item()):
        raise Failure("Finite", f"{label} projection matrix is nonfinite")
    if not bool(
        xp.all(xp.isfinite(winv) & (winv > 0.0)).item()
    ):
        raise Failure("Domain", f"{label} projection weights are not positive")

    cache = _cache(context)
    rank_rtol = float(context.p.rank_rtol)
    selected_rows = _selected_rows(context)
    for token, entry in list(cache.entries.items()):
        if not _metadata_matches(
            entry,
            context,
            matrix,
            rank_rtol,
            selected_rows,
        ):
            del cache.entries[token]
            cache.invalidations += 1
            continue
        if _payload_matches(entry, matrix, winv, xp):
            cache.entries.move_to_end(token)
            cache.hits += 1
            return entry.basis

    cache.misses += 1
    basis = _prepare_uncached(matrix, winv, context, label=label)
    backend = context.compute
    matrix_key = backend.asarray(
        xp.array(matrix, dtype=xp.complex128, copy=True)
    )
    winv_key = backend.real_array(
        xp.array(winv, dtype=xp.float64, copy=True)
    )
    basis.working_bytes = sum(
        value.nbytes
        for value in (
            basis.left,
            basis.singular,
            basis.right_h,
            basis.sqrt_winv,
        )
    )
    basis.backend = backend
    basis.device = getattr(backend, "device", None)
    basis.stream = _stream_identity(context)
    basis.operator = context.op
    basis.layout = _layout_identity(context)
    basis.samples = context.N
    basis.rank_rtol = rank_rtol
    basis.selected_rows = selected_rows
    basis.shape = matrix.shape
    _protect_cpu_arrays(backend, basis)
    _protect_cpu_key(backend, matrix_key)
    _protect_cpu_key(backend, winv_key)

    while len(cache.entries) >= cache.capacity:
        cache.entries.popitem(last=False)
        cache.evictions += 1
    cache.next_token += 1
    cache.entries[cache.next_token] = _CacheEntry(
        backend=basis.backend,
        device=basis.device,
        stream=basis.stream,
        operator=basis.operator,
        layout=basis.layout,
        samples=basis.samples,
        rank_rtol=basis.rank_rtol,
        selected_rows=basis.selected_rows,
        shape=basis.shape,
        matrix_key=matrix_key,
        winv_key=winv_key,
        actual_bytes=(
            matrix_key.nbytes
            + winv_key.nbytes
            + basis.working_bytes
        ),
        basis=basis,
    )
    cache.builds += 1
    return basis


def _row_sum_norm(matrix, xp) -> float:
    if matrix.size == 0:
        return 0.0
    return float(xp.max(xp.sum(xp.abs(matrix), axis=1)).item())


def _forward_limits(matrix, value, source, xp):
    epsilon = float(xp.finfo(xp.float64).eps)
    minimum = float(xp.finfo(xp.float64).tiny)
    value_scale = max(_maximum(value, xp), minimum)
    source_scale = max(
        _maximum(source, xp),
        _maximum(value, xp),
        minimum,
    )
    generic = 4096.0 * epsilon * source_scale
    forward = (
        256.0 * epsilon * _row_sum_norm(matrix, xp) * value_scale
        + minimum
    )
    return min(generic, forward)


def _is_forward_admissible(matrix, value, source, xp) -> bool:
    if matrix.size == 0:
        return True
    trace = matrix @ value
    if not bool(xp.all(xp.isfinite(trace)).item()):
        return False
    return _maximum(trace, xp) <= _forward_limits(
        matrix,
        value,
        source,
        xp,
    )


def apply(basis, source, retained, full, context):
    """Apply one prepared weighted projection and refine existing row gates."""
    xp = _xp(context)
    if not _basis_context_matches(basis, context):
        raise Failure(
            "Cache",
            "projection basis belongs to a different backend, stream, "
            "operator, layout, sample count, row selection, or tolerance",
        )
    source = xp.asarray(source, dtype=xp.complex128)
    retained = xp.asarray(retained, dtype=xp.complex128)
    full = xp.asarray(full, dtype=xp.complex128)
    if source.ndim != 1 or source.size != basis.columns:
        raise Failure("Shape", "wall projection source extent mismatch")
    if retained.shape != (basis.rows, basis.columns):
        raise Failure("Shape", "retained wall projection matrix changed")
    if full.ndim != 2 or full.shape[1] != basis.columns:
        raise Failure("Shape", "full wall projection matrix shape mismatch")
    if not bool(xp.all(xp.isfinite(source)).item()):
        raise Failure("Finite", "wall projection source is nonfinite")

    transformed = source / basis.sqrt_winv
    coefficients = basis.right_h @ transformed
    if basis.rank:
        correction = basis.sqrt_winv * (
            basis.right_h.conj().T @ coefficients
        )
        multiplier = basis.left @ (
            coefficients / basis.singular
        )
    else:
        correction = xp.zeros_like(source)
        multiplier = xp.zeros((0,), dtype=xp.complex128)
    value = source - correction
    refinements = 0
    for _ in range(5):
        if _is_forward_admissible(
            retained,
            value,
            source,
            xp,
        ) and _is_forward_admissible(
            full,
            value,
            source,
            xp,
        ):
            break
        if basis.rank == 0:
            break
        transformed = value / basis.sqrt_winv
        coefficients = basis.right_h @ transformed
        rowspace = basis.sqrt_winv * (
            basis.right_h.conj().T @ coefficients
        )
        value = value - rowspace
        multiplier = multiplier + basis.left @ (
            coefficients / basis.singular
        )
        refinements += 1
    retained_trace = retained @ value
    full_trace = full @ value
    if not _is_forward_admissible(
        retained,
        value,
        source,
        xp,
    ) or not _is_forward_admissible(
        full,
        value,
        source,
        xp,
    ):
        raise Failure(
            "WallProjection",
            "weighted projection did not satisfy the existing "
            "forward/full-row gate in five corrections: "
            f"retained={_maximum(retained_trace, xp):.17g}, "
            f"full={_maximum(full_trace, xp):.17g}",
        )
    if not bool(
        xp.all(xp.isfinite(value))
        and xp.all(xp.isfinite(multiplier))
    ):
        raise Failure("Finite", "wall projection output is nonfinite")
    retained_adjoint = retained.conj().T
    winv = basis.sqrt_winv * basis.sqrt_winv
    multiplier_residual = (
        retained @ (winv * (retained_adjoint @ multiplier))
        - retained @ source
    )
    if not bool(xp.all(xp.isfinite(multiplier_residual)).item()):
        raise Failure("Finite", "wall projection multiplier residual is nonfinite")
    return SimpleNamespace(
        value=value,
        reaction=value - source,
        multiplier=multiplier,
        multiplier_residual=multiplier_residual,
        retained_trace=retained_trace,
        full_trace=full_trace,
        rank=basis.rank,
        refinements=refinements,
        condition=basis.condition,
    )


def project(source, retained, full, winv, context, *, label="wall"):
    """Prepare and apply one weighted projection in a single call."""
    return apply(
        prepare(retained, winv, context, label=label),
        source,
        retained,
        full,
        context,
    )
