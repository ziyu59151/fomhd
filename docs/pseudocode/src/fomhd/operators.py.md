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
6. When `include_finite_mass` is true, compute each Poisson matrix
   `Lphi=-DJ G`, require full numerical rank, keep its complete singular
   spectrum, and factor the original matrix. This remains the direct
   builder default for finite-mass callers and reference tests.

## A0-PY-OP-03 Couplings and complete constraint rows

1. Build R/Z interpolation matrices `I_r`, `I_z`: wall rows use
   `1.5 near -0.5 next`; four-cell interior rows use
   `[-1/16,9/16,9/16,-1/16]`; otherwise use two-cell averages.
2. `I=blockdiag(I_r,I,I_z)` and
   `H=diag(1/node_weights) I^H diag(M1)`.
3. Build `QB` with fourth-order tensor-product averages when all required
   support exists, otherwise the native two-point/patch fallback. Emit the
   same coefficients directly into a CSR matrix for massless aligned boxes,
   and keep the dense matrix in the complete reference builder and legacy
   geometry path.
4. Keep every mode in `DJ` and every wall row in `T_full`. Select
   deterministic independent rows into `T`. Finite storage also builds
   `Bc_full=blockdiag(DJ_m)` and selects independent rows into `Bc`.
   Massless-only storage omits the finite matrices and their workspaces,
   stores zero-row constraint placeholders, and records unavailable finite
   storage. The shared matrix actions and all wall and topology checks run
   in either layout.
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

## Aligned-box Ampère closure

For the aligned box, compute two magnetic norm blocks on the existing
half/full-link samples. In each direction integrate the link weight and
centroid, with R weighting for radial links. Subtract the weighted mean of the
centroid offsets, cumulatively sum the resulting weighted offsets, and divide
by each adjacent centroid gap. Use the positive coefficients in a symmetric
tridiagonal update to the raw diagonal link norm. Transform each norm to the
existing stored chart by congruence with the inverse one-dimensional graph
factor. Keep the toroidal magnetic block diagonal.

Apply the two small norm matrices along lines of the magnetic component
arrays, with transverse volume factors. Use this action on the rows of
(C^*) and divide by (M_1) to assemble Ampère. The same magnetic pairing
integrates magnetic energy and induction power. The cell/face transfer,
Faraday and divergence matrices, and installed wall caps retain their
existing representations.
