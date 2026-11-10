# Fixed massless wall-system cache implementation pseudocode

Source: `src/fomhd/wall_cache.py`.

## A1-PY-WALL-CACHE-01: Thin fixed construction

```text
preflight_memory(context)     # once, before S or any thin/factor allocation
For each stored Fourier mode:
  S = dense signed extraction of all original non-periodic wall rows
  weighted_adjoint = S^H / M1                 # Ne x walls
  CT = C @ weighted_adjoint                   # Nb x walls
  Qw = Cdag @ CT                              # Ne x walls
  Aw = S @ Qw                                 # walls x walls
  require Hermitian Aw
  rows = IndependentRows(Aw,rank_rtol)
  gauge_rows = sorted(all - rows)
  if rows:
    principal = Aw[rows,rows]
    row_scale = electric/length^3
    column_scale = electric*length^2
    prepared = PrepareLinear(principal,row_scale,column_scale)
  else:
    principal = empty
    prepared = none
```

`Cdag@C` is never formed. Fixed arrays remain in the backend's NumPy or CuPy
namespace; no host copy occurs during preparation or application.

The preflight is called exactly once at `_build_wall_cache` entry. The parent
memory integration adds the conservative fixed-wall storage term to
`core.preflight_memory`; a cap failure is raised before wall extraction,
matrix products, rank SVDs, or LU work.

## A1-PY-WALL-CACHE-02: Per-right-hand-side solve

```text
Validate E0 is finite with extent modes*Ne.
For each cached mode:
  source = E0[stored mode block]
  rhs = -S @ (Cdag @ (C @ source))
  if prepared:
    linear = LinearPrepared(prepared,rhs[rows])
    lambda[rows] = linear.x
  else:
    lambda = 0
  lambda[gauge_rows] = 0
  residual = Aw@lambda-rhs                # every original wall row
  require maxabs(residual) <= 1e-8*max(maxabs(rhs),1e-300)
  correction = weighted_adjoint @ lambda
```

Each result mode retains its current `rhs`, multiplier, full residual, rank,
selected/gauge rows, and full `linear_record`. The cache is never treated as a
right-hand-side or solution cache.

## A1-PY-WALL-CACHE-03: Identity, ownership, and storage

The cache is attached only to `context._wall_system_cache`. A shallow copied
context (the alias probe) naturally shares that fixed cache when it retains the
same physical/operator scope. Its key contains the operator, geometry,
backend, parameter-set, and reference object identities; backend name/device;
reference length/electric; `rank_rtol`; operator `K/Ne/Nb/Nc`; and precomputed
operator/rule/geometry hashes when available. It does not contain the context
instance, physical sample count `N`, or `spectral_id`, because `S`, `Aw`,
selection, scaling, SVD, and LU depend only on the retained Fourier-mode
operators. It does not hash operator matrices during a stage.

Operators and geometry are immutable post-construction contracts. Replacing or
mutating them requires a new identity (or a new precomputed identity hash).
Changed operators, geometry, backend, references, rank, parameter-set
identity, or stale/replaced hashes invalidate the cache. No fixed factors are
shared across those boundaries.

Exact persistent cache storage is the sum of `.nbytes` for:

```text
per mode:
  S: walls*Ne complex128
  weighted_adjoint: Ne*walls complex128
  Aw: walls*walls complex128
  row_scale and column_scale: rank float64 each
  cached spectrum: rank float64
  principal and LU original/scaled matrices: rank*rank complex128 each
  LU factors: rank*rank complex128
  pivots: rank int32
plus:
  selected and gauge index lists: 8 bytes per integer
```

For `M=2K+1`, wall-row count `W`, edge count `Ne`, and per-mode selected
rank `r_i`, the numeric-buffer accounting total (excluding Python object overhead) is

```text
B_i = 16*(2*W*Ne + W^2 + 4*r_i^2) + 28*r_i + 8*W
B   = sum_i B_i
```

The `4*r_i^2` complex blocks are principal, scaled principal, the existing
LU record's original matrix, and LU factors. The `28*r_i` bytes are two
float64 scaling vectors, one float64 spectrum, and one int32 pivot vector. The final
`8*W` covers the selected/gauge partition. Borrowed `C/Cdag` references are
not cache-owned bytes.

The active `linear_prepared` record owns references to the same cached
principal/spectrum and does not duplicate their bytes. Per-stage vectors and
thin right-hand-side temporaries are transient and excluded. The formula is
reported as a per-component breakdown so the parent can reserve it directly.

## A1-PY-WALL-CACHE-04: Failure behavior

Construction wraps unexpected library failures as `NumericalLibrary` at the
massless stage boundary while preserving the original traceback by exception
chaining. Hermitian, rank, conditioning, nonfinite right-hand-side, prepared
solve, and full-row residual failures remain explicit `Failure` objects.
There is no CPU fallback and no whole-array host transfer.


## A050 polygon wall dispatch

For polygon_dual_v1 the shared wall cache obtains signed normal extraction from [polygon_wall](polygon_wall.py.md), using actual face IDs and normals. Cache identity, invalidation, ownership and the existing field solve are unchanged. No consumer reinterprets a polygon normal facet as an R or Z face.
