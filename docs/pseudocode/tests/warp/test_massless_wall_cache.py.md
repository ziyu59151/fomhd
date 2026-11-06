# Test massless wall cache

Source: `tests/warp/test_massless_wall_cache.py`.

## A1-PY-TEST-WALL-CACHE-01: Prepared solve equivalence

Prepare one scaled dense system, then solve several independent nonzero
complex right-hand sides. Compare every `linear_record` field with the
unchanged compatibility `Backend.linear` path.

## A1-PY-TEST-WALL-CACHE-02: Equations, rows, and thin storage

For K=1, record backend matrix-matrix shapes while constructing `Aw`. Require
only `Nb x walls`, `Ne x walls`, and `walls x walls` thin products, never
`Ne x Ne`. Compare `Aw` with the former dense equation, require deterministic
principal rows/gauge and every original-row residual, and require positive
exact storage accounting.

## A1-PY-TEST-WALL-CACHE-03: Reuse and invalidation

Solve several nonzero right-hand sides in one context. After initial setup,
require zero independent-row, singular-value, and LU calls and unchanged
records. Repeated complete stages and actual `alias=True` stages must reuse the
same fixed cache and produce identical fields; the alias probe's copied
context may share fixed factors and counters. Operator, geometry, reference,
rank, backend, and replaced identity-hash changes must each rebuild exactly
once. A tiny memory cap must fail before backend matrix-matrix or LU calls.
Fresh cache setup failure must retain model failure evidence; nonfinite
right-hand sides and ill-conditioned systems must fail explicitly.

Covered top-level functions and fixture classes:

- `test_prepared_backend_linear_matches_existing_linear_for_several_rhs`
- `test_wall_aw_uses_only_thin_products_and_matches_dense_equation`
- `test_wall_cache_reuses_setup_for_multiple_nonzero_rhs`
- `test_wall_cache_invalidates_on_identity_changes`
- `test_massless_stage_reuses_wall_setup_and_repeats_identical_outputs`
- `test_actual_alias_stage_shares_fixed_wall_setup`
- `test_wall_cache_setup_library_failure_retains_model_evidence`
- `test_tiny_memory_cap_fails_before_wall_allocations`
- `test_wall_rhs_and_conditioning_failures_are_explicit`

CPU evidence only; no CUDA claim is made by this test mirror.
