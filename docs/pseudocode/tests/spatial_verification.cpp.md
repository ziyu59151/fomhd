# Spatial verification implementation pseudocode

Stable step ID: `A0-VERIFY-SPATIAL`.

Source under test: `include/fomhd/operators.hpp` and
`src/numerics/operators.cpp`.

Test executable: `tests/spatial_verification.cpp`.

## Interface

`build_spatial_operators(geometry, mode, options)` builds one Fourier mode of
the production RZ/Yee differential operators without materializing dense KKT,
constraint, SVD, Poisson-factorization, or transfer systems. It returns:

```text
SpatialOperators:
  mode: signed toroidal Fourier mode
  Nc, Ne, Nb: entity counts
  G:  (R-face, cell, Z-face) x cells
  C:  (Z-face, vertex, R-face) x (R-face, cell, Z-face)
  DB: vertices x (Z-face, vertex, R-face)
  DJ: cells x (R-face, cell, Z-face)
```

All matrices are host FP64 complex coefficients and use SI derivative units.
Rows and columns use the same physical component order as `Operators`. The
builder uses the prescribed GC7 products exactly:

```text
G  = T1 G2
C  = T2 C2 T1^-1
DB = T3 DB2 T2^-1
DJ = -M0^-1 G* M1
```

`T1^-1` and `T2^-1` are applied through the existing block LU factors with
multiple right-hand sides. `build_operators` consumes the same per-mode
builder, while `build_gradient_operator` keeps its lightweight
gradient-only path.

## Manufactured fields

Use the rectangle `R in [1,2]` and `Z in [0,1]`, with
`h_R=h_Z=1/N`. For `s=(x-center)/w`, define

```text
A(s) = exp(-1/(1-s^2)),  |s| < 1
A(s) = 0,                |s| >= 1
A'(s) = A(s) * (-2*s / (w*(1-s^2)^2))
```

Use `center_R=1.5`, `center_Z=0.5`, `w=0.42`, and

```text
BR(R) = A((R-center_R)/w)
BZ(Z) = A((Z-center_Z)/w)
B(R,Z) = BR(R) BZ(Z)
dR B = BR'(R) BZ(Z)
dZ B = BR(R) BZ'(Z)
```

Every scalar or vector component is a nonzero constant times
`B(R,Z) exp(i*k*phi)` with `k=2`. Since all operators return Fourier
coefficients, evaluate the analytic reference at physical `phi=0`.

The support is contained in `[1.08,1.92] x [0.08,0.92]`, away from the fixed
walls. Compare all three refinements on the same physical interior box
`R in [1.25,1.75]`, `Z in [0.25,0.75]`. Report entity-boundary errors
separately as wall error and all entity errors as global error; make no
global fourth-order or wall-order claim.

## Exact cylindrical derivatives

For a scalar coefficient `f`, at the actual entity coordinates:

```text
G_R     = dR f
G_phi   = i*k*f/R
G_Z     = dZ f
```

For an electric vector `E=(E_R,E_phi,E_Z)`:

```text
curl_R   = i*k*E_Z/R - dZ(E_phi)
curl_phi = dZ(E_R) - dR(E_Z)
curl_Z   = dR(E_phi) + E_phi/R - i*k*E_R/R
```

For a magnetic vector `B=(B_R,B_phi,B_Z)`:

```text
divB = dR(B_R) + B_R/R + dZ(B_Z) + i*k*B_phi/R
```

For an edge flux `J=(J_R,J_phi,J_Z)`, `DJ*J` uses the same divergence formula,
including the cylindrical radial term.

Evaluate `G`, `C`, `DB`, and `DJ` only at the positions of their actual output
entities. Do not substitute a Cartesian derivative, a same-location stencil,
or a construction identity such as `C*G=0` or `DB*C=0` for the analytic
comparison.

## Batched right solves and irregular-mask equivalence

Do not materialize `T1^-1` or `T2^-1`. For symmetric real `P`, apply a right
inverse action to a block without forming the inverse:

```text
right_solve(A,P) = adjoint(solve(P_factor,adjoint(A)))
                 = A*P^-1
```

For `C2*T1^-1`, right-solve the ER columns with `PR_r` and the EZ columns with
`PZ_z`; leave the cell-centered Ephi columns unchanged.

For `DB2*T2^-1`, use the column blocks:

```text
BR: right_solve(A, PZ_z)
Bphi: right_solve(right_solve(A, PZ_v), PR_v)
BZ: scale columns /Rbar; right_solve with PR_r; scale columns *Rbar
```

Construct a stepped disk whose masks make `PR_v*PZ_v` differ from
`PZ_v*PR_v`. Apply the production `apply_t2_inverse_block` to an identity
block and compare every column with the original per-column action. Run the
block comparison and full `C*G`/`DB*C` checks separately for modes `-2`, `0`,
and `+2`:

```text
block_action = apply_t2_inverse_block(operators, identity(Nb))
for each unit basis column e_j:
  per_column[:,j] = apply_t2_inverse(operators, e_j)

require max_abs(block_action-per_column)
  <= 512*epsilon*Nb*max_abs(block_action)
```

The same stepped geometry must pass full, independently multiplied:

```text
max_abs(C*G)  <= 128*epsilon*Ne*max_abs(C)*max_abs(G)
max_abs(DB*C) <= 128*epsilon*Nb*max_abs(DB)*max_abs(C)
```

These checks verify both factor order and full complex compatibility on a
noncommuting mask. They do not replace the rectangular analytic MMS.

## Error and acceptance pseudocode

```text
for each mode k in (-2,0,+2):
  block_action = apply_t2_inverse_block(operators, identity(Nb))
  compare block_action with per-column apply_t2_inverse
  spatial = build_spatial_operators(geometry, k, 512 MiB cap)
  require full CG and DBC roundoff checks

for N in (8,16,32):
  construct rectangle with N+1 nodes in each direction
  spatial = build_spatial_operators(geometry, +2, 3 GiB cap)
  for each operator in (G,C,DB,DJ):
    combine every component and every actual output-stat point
    reject nonfinite R, Z, actual, analytic reference, error, or magnitude
      before accumulating any maximum
    E_interior = max |actual-exact| over the fixed interior
    A_interior = max |exact| over the fixed interior
    E_wall = max |actual-exact| over entity-boundary points
    E_global = max |actual-exact| over all output points
    A_global = max |exact| over all output points
    interior_normalized = E_interior / A_interior
    wall_normalized = E_wall / A_global
    global_normalized = E_global / A_global
  retain each operator's combined interior_normalized error

for adjacent refinement pairs:
  order = log2(coarse_error/fine_error)
  require the fine error is finite and greater than 1e-12
  require interior order >= 3.5 for G, C, DB, and DJ

at N=8:
  build mode -2 and require its signed analytic curl residual remains finite
  and below 0.1
  deliberately reverse the analytic curl_phi derivative reference
  require that negative-control error exceeds 0.1

always:
  require account() rejects an actual NaN and an analytic reference Inf

at N=32:
  require a 2 GiB operator-matrix budget is rejected
```

The sign/derivative control must fail. A test that accepts the wrong
`curl_phi` formula or silently ignores a nonfinite accumulation input is
invalid.

## Public right-solve workspace control

A tall RHS block must be rejected before allocating solve temporaries when its
budget covers only two dense blocks. With `L=max(nz,nv,nr)`, conservatively
budget `(block.size + 6*L*block.rows + L*L)*sizeof(Complex)` using checked size
arithmetic. This includes retained vertex data, selected and adjoint blocks,
LAPACK RHS/result copies, and factor conversion. Entity ordering belongs to
columns. The production aggregate estimate separately covers all simultaneously
live spatial matrices; neither bound changes a discretization or tolerance.
