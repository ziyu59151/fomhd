# Polygon weighted thin-SVD wall projection

Status: selected polygon-only numerical repair for `polygon_dual_v1`.
The staircase Gram/LU path and all time equations and diagnostic thresholds
remain unchanged.

## Problem

The existing wall projection solves

```text
S lambda = T source
S = T diag(winv) T^H
value = source - winv T^H lambda
```

For a non-axis actual-normal polygon wall, the Gram matrix squares the
condition number of `T`. Runtime N=8/12 full-vacuum electron projection
therefore can fail the existing forward/full-trace gate even when the physical
constraint rows are full rank.

## Weighted thin SVD

All arrays remain backend-owned FP64/complex128 arrays. Let

```text
sqrt_winv = sqrt(winv)
A = T diag(sqrt_winv)
A = U diag(sigma) Vh                      thin SVD
x = source / sqrt_winv
coefficients = Vh x
correction = sqrt_winv (Vh^H coefficients)
value = source - correction
```

`value` is the orthogonal projection of `x` onto the nullspace of `A`,
transformed back to source coordinates. It is a genuine weighted constraint
projection, not an RHS subtraction.

The retained multiplier is

```text
lambda = U (coefficients / sigma)
reaction = value - source
reaction = -winv T^H lambda
```

The negative reaction sign preserves the existing `wall.project_wall`
contract. An outward-normal source therefore produces an inward reaction.

The reported multiplier residual is the actual normal-equation residual

```text
multiplier_residual =
    T (winv * (T^H lambda)) - T source
```

It is not the forward constraint trace and is not multiplied by an extra
minus sign.

Every retained row must remain a physical constraint. The rank gate is

```text
sigma_i > rank_rtol * sigma_0
```

for every singular value of the retained-row basis, and `len(sigma)` must be
`T.shape[0]`. A deficient basis raises `WallRank`; no row is dropped and no
threshold is relaxed.

## Forward refinement

After the thin projection, recompute the retained-row and full-row traces:

```text
residual = T value
full_residual = T_full value
```

If either exceeds the existing forward gate, apply up to five corrections
using the same thin basis:

```text
delta = sqrt_winv Vh^H Vh (value / sqrt_winv)
value = value - delta
```

The correction updates `reaction=source-value` and
`lambda=lambda+U((Vh(delta/sqrt_winv))/sigma)`. It does not replace the
projection with a different solver and does not alter the existing gate
scale.

## Shared APIs

`polygon_projection.prepare(T, winv, context)` stores the thin SVD basis,
scale, rank, and condition. `polygon_projection.apply(...)` returns
`value`, signed `reaction`, `multiplier`, normal-equation
`multiplier_residual`, forward trace, full trace, rank, and refinement count.

`wall.build_wall` for polygon stores the basis in each sample and keeps the
Gram `S` record for compatibility. `wall.project_wall` uses the stored basis.
The legacy build/project path is unchanged.

`massless._constraint_project` dispatches polygon to the same helper and
continues returning its existing sampled output array. The existing
full-trace gate and all work/diagnostic contracts remain in force.

## Context-local exact factor reuse

Each call context owns at most two prepared bases in
`context._polygon_projection_cache`; there is no module-global or
cross-context cache. The cache is an LRU list with `hits`, `misses`, `builds`,
`evictions`, and `actual_bytes` counters. Preparing a miss always runs the
existing full SVD and every rank/refinement gate. A failure never inserts a
partial basis.

The cache stores its owner context identity. A shallow `copy.copy(context)`
therefore receives a fresh cache instead of inheriting the source context's
LRU.

An entry matches only when all of the following hold:

```text
compute backend and device identity
current CUDA stream identity (CPU uses a fixed token)
operator object, declared matrix/geometry layout
context.N and context.p.rank_rtol
selected wall-row identity and matrix shape
full backend array_equal(T_cached_copy, T_current)
full backend array_equal(winv_cached_copy, winv_current)
```

The cached `T` and positive `winv` are owned copies, so mutating the caller's
input or changing one FP64 bit causes a miss. Exact `T`/`winv` equality is
required; rounded hashes and pointer-only identity are forbidden. The label is
metadata only. A metadata mismatch may clear unusable entries; a payload
mismatch misses and prepares again. The oldest of two valid entries is evicted
only after a successful new preparation.

Equality-key snapshots belong to the cache entry, not to the returned working
basis. `wall.build_wall` and other callers may keep an SVD working basis for
every sample; evicting its LRU entry must then free that entry's copied `T` and
`winv` keys even while the basis remains alive.

Persistent entry bytes are exactly the cache-entry and working-array bytes:

```text
entry_bytes =
    T_copy + winv_copy             # owned only by _CacheEntry
  + U + singular_values + Vh + sqrt(winv)  # owned working basis
```

Cached CPU arrays are write-protected. CUDA arrays remain backend-owned under
the no-mutation/no-host-fallback contract. `apply` never mutates a basis. The
returned basis contains only SVD working arrays plus backend/operator/stream
ownership metadata; it has no equality-key copy.

## Tests

- actual non-axis polygon T at N=8 and N=12;
- variable cell density and complex source;
- independent Gram normal-equation oracle;
- independent weighted-nullspace preservation;
- redundant `T_full` rows with selected independent retained rows;
- reaction/multiplier identity and zero reciprocal-weighted reaction work;
- outward-normal source with inward reaction on axis and non-axis rows;
- identical polygon output for massless electron and ion fixed-density wall
  projection on the same T/density;
- explicit rank-deficient failure;
- unchanged staircase tests.

Additional cache tests count actual CPU SVD calls and cover exact repeated
hits, one-bit density and matrix mutations, operator/rank/backend/N changes,
two-entry eviction, failed-preparation non-insertion, no alias mutation, and
an uncached fresh-context oracle. Density is never frozen: every distinct
positive `winv` either hits an exact old entry or builds an exact new basis.
The lifetime regression keeps more than two returned bases alive, evicts the
oldest entry, and verifies both copied equality keys become collectable while
the evicted working basis remains referenced.

## Explicit backend

`polygon_wall.maxwell_traction` and `wall_poynting_normal` take an explicit
`xp` argument. Context-driven traction calls pass `context.compute.xp`.
Independent CPU tests may omit it and receive NumPy. Backend inference from
`__array_namespace__` is forbidden because installed CuPy 14.2.0 arrays do
not provide that hook; no device array is silently converted to host NumPy.
