# Shared massless storage and wall-projection tests

Source: `tests/warp/test_shared_optimizations.py`.

## Bounded operator storage

```text
for real small wall geometry rectangle and stepped ell at K=1:
  build default-finite and explicitly massless operators
  require omitted Lphi/factor/spectrum and global Bc storage
  require G,C,DB,DJ,Cdag,I,H,QB,T,T_full,DR,DZ,topology identical
  require shared validation checks and zero-row unavailable placeholders
  require the selected-storage preflight accepts a cap below full-finite need
  require finite-only phi factor and SVD calls disappear
  account exact retained bytes for omitted facilities

run the same canonical massless state through both operator builds:
  require identical stage RHS, electric/wall fields, and constraint traces

compare runtime plans with the complete default-finite operators and selected
massless operators:
  finite default plan remains at its checked component count
  massless uses exact owned operator/filter-factor inventories
  borrowed wall-cache C/Cdag do not add storage
  live dense temporaries use checked actual-shape counts
  sampled/state/history, retained T1 history and fixed-wall terms remain
  preflight still rejects one byte below the reduced plan and accepts the plan

construct the archived EAST32 K1 shape-only context:
  Nc=796, Nr=Nz=828, Nv=861, Ne=2452, Nb=2517, Nphi=5, W=128
  require the exact synchronous A0 persistent/temporary/plan byte breakdown
  require synchronous A0 and predictor M4/T1 plans below 8 GiB
  force an oversized shape product and require Failure("Memory")
  remove one resident mode matrix and require Failure("Shape") instead of a
    silently zeroed temporary allowance
```

## Constraint projection

```text
construct a selected T whose nonzero columns have at most one row:
  require batched diagonal projection matches independent lstsq definition
  require no SVD call, complex128 output, and all-wall residual check
  replace T and require the next call uses the new value, not a cached table

use the actual coupled wall T:
  require the checked SVD/LU fallback and dense-definition agreement

make a selected row rank-deficient through positive current-stage density:
  require the existing WallRank failure
```

## IO model seam

```text
write and load the same small massless and finite cases:
  massless selects omitted storage; finite retains complete storage
  repeated massless fingerprints agree and ignore omitted placeholders
  effective electron inertia remains in semantic restart identity

form a finite-model context with massless reduced operators:
  require preflight and stage dispatch fail with Failure("Model")
  require source/trial/generation counters are unchanged
  require the default complete finite storage still passes unchanged

form the actual full-storage massless fingerprint with massless parameters:
  write a full-storage checkpoint through the existing Run contract
  write and restore a same-storage reduced checkpoint
  require the full-storage checkpoint to fail the explicit pre-0.5 boundary
```

These checks cover CPU storage and algebra only. They do not claim CUDA
execution, production-scale memory, or spatially parallel time integration.
