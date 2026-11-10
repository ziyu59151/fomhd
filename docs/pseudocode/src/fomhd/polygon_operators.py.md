# `polygon_operators.py` implementation pseudocode

## A050-POLY-OP-01 layout and ownership

`build_polygon_operators(geometry,K,options=None,backend=None)` returns the
existing `Operators` record with

```text
kind = geometry_kind = "polygon_dual_v1"
matrix_layout_version = "polygon_dual_v1"
Nf = len(geometry.faces.keys)
Nc = len(geometry.cells.keys)
Nv = len(geometry.vertices.keys)
Ne = Nf + Nc
Nb = Nv + Nf
E = [En(Nf), Ephi(Nc)]
B = [Bphi(Nv), Bt(Nf)]
```

The modes are stored in the existing order `-K,...,K`. Geometry remains
immutable host FP64. Operator matrices belong to the explicitly supplied
`Backend`; no device mismatch or CPU fallback is permitted. This first unit is
a dense bounded oracle and does not expose a full RHS or finite-mass solve.

## A050-POLY-OP-02 dense memory gate

Before allocating any dense matrix, compute a conservative upper bound from
`Nf,Nc,Nv,K`, the sparse and dense compatibility incidences, `T/T_full`,
local LSQ supports, and at least one materialized backend/Host copy. Compare
the bound to
`options.memory_cap_bytes`. Raise `Failure("Memory", ...)` with requested
bytes, actual shapes, and the cap when it is exceeded.

The budget never rounds a shape down and never attempts a reduced allocation
after a capacity stop.

## A050-POLY-OP-03 low-order blocks

For mode `k`, with `D=diag(face_distances)`, `Av=diag(vertex_areas)`,
`Wv=diag(vertex_volumes)`, `Rf=face_midpoint_R`, and `Rc=M0/area_c`:

```text
g       = -D^-1 B2^T                                (Nf,Nc)
G       = [g; i*k/Rc]                              (Ne,Nc)
C       = [ -Av^-1 B1^T D                 0
            -i*k Rf^-1             Rf^-1 g Rc ]   (Nb,Ne)
DB      = [ i*k Av/Wv        -Wv^-1 B1^T D Rf ]    (Nv,Nb)
DJ      = -M0^-1 G^H M1                            (Nc,Ne)
Cdag    = M1^-1 C^H MB                             (Ne,Nb)
```

The same `d` appears in `G`, `C_phi,n`, and `DB_t`; `ell` is not substituted
for `d`. `M1=[Rf*ell*d,M0]` and `MB=[Wv,Rf*ell*d]`. Each mode is retained in
`G,C,DB,DJ,Cdag`. The identities

```text
C G = 0
DB C = 0
M0 DJ + G^H M1 = 0
```

are checked with backend reductions before return.

The legacy `b1r,b1z,b2r,b2z` aliases are mathematically genuine:
`B1` and the empty Z incidence, and `B2` and the empty Z incidence. All six
legacy `PR/PZ` filter fields are explicit `(0,0)` backend arrays with `None`
factors. The polygon branch uses `I/T/H` directly; `apply_t2_inverse` is not a
valid graph action. No high-order filter or Poisson matrix is fabricated.

## A050-POLY-OP-04 scalar transfer and physical vector embedding

Build `S(Nf,Nc)` by local scaled polynomial least squares. For each face
midpoint, grow a nearest-cell support from 4 toward at most 24 cells until the
polynomial basis has full rank. Scale coordinates by a positive local radius
and scale columns by their norms. A linear basis is the minimum accepted
model; quadratic is enabled only by options and follows the same rank and
condition checks.

The value at the query is the constant coefficient in the original scaled
coordinates. If `A_scaled=A/column_scale` and `P=pinv(A_scaled)`, the
interpolation weights are `P[0,:]/column_scale[0]`, not `P[0,:]`.

Reject a support with rank deficiency or condition number above
`max_condition=1e8`; do not extrapolate or fill an unavailable value. Record
support sizes, rank, condition numbers, and the chosen polynomial degree.
Also reject more than 32 incident faces on one cell, since this bounded unit
does not grow the local support without limit.

Define:

```text
scalar_I = [S; Icell]                 (Ne,Nc)
I = [nR*S, 0,    nZ*S
     0,    Icell, 0]                  (Ne,3Nc)
H = diag(tile(M0,3))^-1 I^H diag(M1)  (3Nc,Ne)
```

`T_full=I[boundary_faces,:]` has shape `(W,3*Nc)`, one actual polygon face per
row. `T` is the deterministic independent-row subset selected and materialized
by the backend. No boundary row is inferred from an axis-aligned staircase.

## A050-POLY-OP-05 local scalar and QB reconstruction

`QB(3Nc,Nb)` reconstructs physical `(B_R,B_phi,B_Z)` at cell centroids:

1. Fit `B_phi` from vertex samples with a bounded scaled linear LSQ.
2. At each face, `Bt=t_R*B_R+t_Z*B_Z` is one measurement, not two independent
   scalar samples.
3. Fit the six linear vector coefficients jointly with rows
   `[t_R,t_R*x,t_R*z,t_Z,t_Z*x,t_Z*z]`.
4. Undo the constant coefficient scales: use `P[0]/column_scale[0]` and
   `P[3]/column_scale[3]` as direct `B_R` and `B_Z` weights; apply no second
   tangent factor.
5. Place those weights directly in the `B_t` columns of `QB`. Fit `B_phi`
   separately from vertex samples.

Scalar supports use at most 24 cells; joint QB fits use at most 32 faces.
Rank and scaled condition are checked with the same `1e8` limit. A rank
failure is an explicit geometry or reconstruction failure, never a silently
filled zero.

## A050-POLY-OP-06 compatibility and diagnostics

Set `Dface=DJ[K][:,:Nf]`. `DR` documents the same normal-face compatibility
count and `DZ` is an empty `(Nc,0)` table. New fluid consumers must use
`Dface`; these aliases do not mean a staircase R/Z split.

Set `curvature_radius=geometry.cells.R` and
`topology_flux_row=[Av,0_Bt]` of length `Nb`. Then the topology row contracted
with `C[K]` is zero up to roundoff. Set
`face_normals_backend=backend.real_array(geometry.face_normals)` and
`cell_radius_backend=backend.real_array(geometry.cells.R)`, so the new RHS
consumes backend-owned metrics without repeating a host geometry upload. On
CPU, also set `flags.writeable=False`. CUDA CuPy 14.2 exposes only
`c_contiguous`, `f_contiguous`, and `owndata`; do not access or emulate a
`writeable` flag there. The CUDA arrays are immutable by ownership contract:
the operator owns them, consumers must not mutate them, and there is no host
fallback.

```text
finite_mass_available = False
Lphi, phi_factor, poisson_spectra = []
Bc, Bc_full, selected_constraint_rows = empty
```

The unavailability reason and omitted storage names are explicit. Fill
`setup_checks` with `CG`, `DB_C`, weighted gradient/divergence adjoint,
transfer/H pairing, and topology residuals and scales.

## A050-POLY-OP-07 acceptance boundary

CPU acceptance covers:

- varying face length and distance on a nonuniform true-polygon mesh;
- exact incidence closure and sign covariance;
- `CG`, `DBC`, weighted adjoints, and positive measures;
- linear scalar reconstruction and polynomial QB reconstruction;
- independent-row reuse for `T`;
- capacity and rank failures.

This unit does not claim a full physical RHS, finite electron mass, a
production sparse inverse, or GPU execution. The parent owns those later
changes and must run them on the same interface and metrics.

## A050-POLY-OP-08 regular two-owner scalar repair

For `scalar_degree=1` only, detect an actual regular interior facet without
using its index as evidence. For each owner cell require:

- the selected cell polygon has four axis-aligned edges;
- its R and Z spans equal the supplied generator-grid spacings `hR,hZ`;
- its polygon area equals `hR*hZ`;
- its area centroid equals its retained generator coordinate.

For an interior face require both owners to pass, the first and second centroids
to differ only along one axis, the actual face midpoint to equal their centroid
midpoint, the face normal to align with the centroid separation, and the face
length to equal the full transverse grid spacing (`hZ` for a radial separation,
`hR` for an axial separation). All comparisons use
`4096*eps*max(scale)`-style relative FP64 bounds. A boundary face or any
clipped/irregular face fails this predicate.

For a regular facet only, replace its scalar polynomial row by the symmetric
two-owner average:

```text
S[f,first]  = 0.5
S[f,second] = 0.5
```

Keep every other `S` row from local rank/condition-checked polynomial LSQ.
Degree 2 is unchanged. Record each regular row as `kind=regular_two_owner`,
`support=2`, actual rank `2`, condition `1`, and the two owner indices. Record
other rows as `kind=polynomial` with their actual support/rank/condition.

`I`, `H`, `T`, and the fluid velocity-gradient/gradient-adjoint actions consume
the resulting `S` unchanged. This repair does not alter `G`, `C`, `DB`, `DJ`,
geometry centroids, grid coordinates, or the degree-2 candidate.

Regular interior patch acceptance uses the actual fluid consumer:

- isotropic pressure `p=R` gives momentum force `[-1,0,0]`;
- isotropic pressure `p=Z+2` gives momentum force `[0,0,-1]`;
- reconstructed constant `E_R` or `E_Z` through `H` remains constant;
- `u_R=Z^2` has `d_R=0`, `d_Z=2*Z`, including the centered cylindrical
  connection terms.

These are bounded consistency patches at roundoff level, not an irregular
boundary error or full-physics accuracy claim.
