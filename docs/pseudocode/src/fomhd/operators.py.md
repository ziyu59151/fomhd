# `src/fomhd/operators.py` implementation pseudocode

## A0-PY-OP-01 Fixed complex

1. Build cell-to-edge incidence `b2r`, `b2z`, edge-to-vertex incidence
   `b1r`, `b1z`, and graph filters `PR/PZ` exactly as native.
2. A graph filter starts from identity. For each nonboundary entity with a
   nonboundary positive neighbor, add `1/24` to each diagonal and subtract
   `1/24` from the two off-diagonal entries.
3. Keep all matrices complex128 on `backend.xp`. Geometry metrics are copied
   to backend float64 arrays.
4. Reject matrix assembly when the exact native dense-memory estimate exceeds
   `OperatorOptions.memory_cap_bytes`.

## A0-PY-OP-02 Compatible low-order operators

1. `Gr = -diag(1/lr) b2r^H`, `Gz = -diag(1/lz) b2z^H`.
2. Assemble `G2` in edge order (R-face, cell, Z-face):
   R block `Gr`, phi block `i*m/Rc`, Z block `Gz`.
3. Assemble `C2` in face-node order (Z-face, vertex, R-face) using the native
   mode terms, then apply `T1`, `T2`, and the graph inverse factors in the
   native noncommuting order.
4. Assemble `DB2` in vertex rows and apply `T3` after the bounded inverse T2
   block action. Never replace transforms with commuting approximations.
5. `DJ=-diag(1/M0) G^H diag(M1)`.
6. Compute each Poisson matrix `Lphi=-DJ G`, require full numerical rank, keep
   its complete singular spectrum, and factor the original matrix.

## A0-PY-OP-03 Couplings and complete constraint rows

1. Build R/Z interpolation matrices `I_r`, `I_z`: wall rows use
   `1.5 near -0.5 next`; four-cell interior rows use
   `[-1/16,9/16,9/16,-1/16]`; otherwise use two-cell averages.
2. `I=blockdiag(I_r,I,I_z)` and
   `H=diag(1/node_weights) I^H diag(M1)`.
3. Build `QB` with fourth-order tensor-product averages when all required
   support exists, otherwise the native two-point/patch fallback.
4. Keep every mode in `DJ`; `Bc_full=blockdiag(DJ_m)` and select deterministic
   independent rows into `Bc`. Keep every wall row in `T_full` and select
   deterministic independent rows into `T`.
5. Build the topology flux row by solving with an adjoint T2 factor; do not
   form an inverse.

## A0-PY-OP-04 Acceptance checks

1. For every mode check `CG=0`, `DBC=0`, and
   `M0*DJ + G^H*M1=0` against native roundoff bounds.
2. Check the closed-wall divergence theorem, I/H weighted adjoint, and
   topology curl row.
3. Raise on any bound failure; otherwise return all named
   `SimpleNamespace` checks.
4. Build all temporary vectors through the owning `backend.xp`; never apply a
   NumPy operation directly to a CuPy device array during validation.

## A0-PY-OP-05 Transform application

1. `apply_t1/inverse` operate in edge order.
2. `apply_t2` computes `PZ_z`, then `PR_v(PZ_v)`, then radial
   `diag(1/Rbar_r) PR_r diag(Rbar_r)`.
3. `apply_t2_inverse` solves in the exact reverse factor order:
   `PZ_z^-1`, then `PZ_v^-1`, then `PR_v^-1`, and finally the scaled
   `PR_r^-1`.
4. `apply_t3` and inverse use `diag(1/Rbar_v) PR_v PZ_v diag(Rbar_v)` and
   reverse solve order. Reject every shape mismatch.
5. `apply_t2_inverse_block` defaults to `operators.compute` when no backend is
   supplied, so its right solves can never switch a device operator to CPU.

## A0-PY-OP-06 Selected finite-mass storage

`OperatorOptions.include_finite_mass` defaults to `True`. It is storage
selection, not a model switch: every caller keeps the finite system unless it
explicitly requests massless construction.

```text
build_operators(geometry,K,options,backend):
  require options.include_finite_mass is boolean
  finite_plan=estimate(lphi, isolated phi LU, Bc_full, maximum selected Bc)
  preflight=full_dense_plan-finite_plan when massless
           full_dense_plan otherwise
  build G,C,DB,DJ,Cdag,I,H,QB,T_full,T,DR,DZ,topology and all shared factors
  if finite:
    build Lphi, spectra, phi factors, Bc_full, selected Bc
    mark finite facilities available
  else:
    keep Lphi/phi factors/spectra empty and Bc rows empty
    omit constraint-row selection; do not build and discard a Poisson matrix
    mark finite facilities unavailable with an explicit reason and omitted names
  validate every shared algebra, adjoint, topology and wall check
```

The massless placeholders have zero rows and are not ordinary finite
constraints. Finite field solves require `finite_mass_available`; the selected
massless stage and wall cache do not read `Lphi`, `phi_factor`, `Bc_full`, or
`Bc`. The global dense `Bc_full` remains required for finite Darwin solves and
full-row divergence checks.

The consumer audit found that `fields.electrostatic`, `fields.transverse`, and
the finite `stage.evaluate_stage` path read the Poisson/Darwin storage. The
massless stage dispatches before those paths. `core.estimated_peak_bytes` reads
`Bc.shape` and `Bc_full.size` only for its generic plan, so zero-row
placeholders safely reduce that plan. The IO fingerprint conditionally skips
the unavailable records and hashes an explicit unavailable marker. No shared
massless consumer was found to require any omitted finite quantity.

The memory estimate reports the omitted Lphi, retained phi factor copies and
pivots, full Bc, and maximum selected-Bc storage separately. This is storage
selection only. High-order curl retains graph-filter inverses and is not
treated as a sparse operator.
