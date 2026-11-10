# `src/fomhd/boundary_quadrature.py` implementation pseudocode

## A050-PY-VOLQ-00 Scope and public contract

`build_boundary_volume_weights(geometry, boundary_geometry, band_cells=2)`
constructs one proposed positive scalar quadrature on the unchanged structured
scalar sites. It is boundary-only geometry work. It does not modify fields,
geometry, boundary values, stage state, time stepping, operators, or physical
conditions.

The input `boundary_geometry` must be the current immutable
`fomhd.boundary_geometry.BoundaryGeometry` for the same `Geometry`: active
cell keys, lengths, physical polygon, and fingerprints must agree. Return a frozen
`BoundaryVolumeWeights` with SI `m^3/rad` weights, explicit units, original
cell keys/ids, old and proposed moments, residuals, band mask, changed ids,
normalized constraint condition/rank, positivity, unchanged-core evidence,
and a deterministic fingerprint. Every returned array is read-only.

The construction is a three-moment quadrature experiment, not a proof of a
completed summation-by-parts, traction, force, or physical boundary closure.
The solver remains on its existing geometry and weights until separate parent
coordination.

## A050-PY-VOLQ-00A Canonical geometry pairing

Before constructing moments or a band, validate the supplied geometry pair:

1. Convert the source `Geometry.polygon` and
   `boundary_geometry.polygon_R_m/polygon_Z_m` to ordered `(R,Z)` tuples.
2. Remove exactly one redundant closing vertex when the first and last pair
   are bitwise equal. This is allowed because the source polygon validator
   accepts one repeated closing vertex. Do not sort, rotate, reverse, snap, or
   approximately compare vertices.
3. Require exact canonical tuple equality. A boundary table from a physically
   different polygon must fail even when its grid and active keys coincide.
4. Require identical R/Z grids, ordered active cell keys, cell ids
   `0..Nc-1`, and the active mask in original `i + nr*j` order.
5. For every active `(i,j)` key, require the retained `Geometry.cells.R/Z`
   coordinates to equal `R[i]/Z[j]` exactly. Require every intercept origin and
   cell id to agree with that same retained scalar site.

These checks reject a mismatched physical polygon without rebuilding or
substituting a geometry backend.

## A050-PY-VOLQ-01 Stable physical polygon moments

Let polygon coordinates be immutable FP64. Choose

```text
R0 = 1/2 * (min R + max R)
Z0 = 1/2 * (min Z + max Z)
L  = max(max R-min R, max Z-min Z)
x  = (R-R0)/L
z  = (Z-Z0)/L
```

Use the first polygon vertex as a fan origin. For every triangle in the fan,
evaluate the 3-point degree-2 rule at the three edge midpoints with equal
weights `triangle_area/3`. Compute in long-double accumulation:

```text
m0 = integral R dA
mx = integral R*x dA
mz = integral R*z dA
```

This is exactly degree 2 for these integrands. Do not infer moments from an
area centroid or from the existing vertex-dual metric. Verify the helper
against independent rectangle and triangle analytic formulas.

## A050-PY-VOLQ-02 Fixed boundary band

Require `band_cells` to be a positive integer. Its physical radius is

```text
band_radius = band_cells * max(hR, hZ)
```

For every active scalar site at the unchanged cell centre, compute the minimum
Euclidean distance to the actual polygon edges. Mark it in the boundary band
when `distance <= band_radius`. Preserve original active order and ids.

Do not retry with a larger radius. The caller may explicitly request another
positive band width; a default failure does not silently increase it.

## A050-PY-VOLQ-03 Minimum relative-change equality fit

Build the normalized site matrix

```text
A = [1, x, z]                         shape Nc x 3
w0 = exact copy of old Geometry M0    shape Nc
```

Core sites outside the band remain frozen with exactly their old weights.
Compute

```text
core_moments = A_core.T w0_core
target       = physical_moments - core_moments
band_rhs     = target - A_band.T w0_band
D            = diag(w0_band^2)
G            = A_band D A_band.T      shape 3 x 3
```

Normalize each row of `G` and the corresponding `band_rhs` by the row's
largest absolute value. Report the SVD rank and condition number of this
normalized 3x3 matrix. Rank loss is an explicit infeasibility.

Solve

```text
y     = G_normalized^-1 band_rhs_normalized
delta = D A_band.T y
w_band = w0_band + delta
```

This is the declared minimum `sum(delta_i^2/w0_i^2)` equality fit. Do not add
a global residual offset, rescale core/interior weights, clip weights, or
introduce another band automatically.

If all old positive weights already satisfy **all three** moments within the
scaled tolerance, return them bitwise unchanged with empty `changed_cell_ids`.
A full rectangular midpoint `M0` integrates `integral R dA` exactly, but it
does not generally integrate `integral R*x dA` exactly. Therefore a rectangle
may legitimately receive boundary-band changes from the frozen-core midpoint
defect. Unchanged rectangle weights are not a gate unless all three old
moments already match.

## A050-PY-VOLQ-04 Acceptance and failures

Require every old weight finite and strictly positive. Require every proposed
band weight finite and strictly positive. For one global scale

```text
scale = max(maxabs(physical_moments), maxabs(old_moments), tiny)
error = maxabs(new_moments - physical_moments) / scale
```

Require `error <= 1e-12`. If the normalized matrix loses rank, a candidate is
nonpositive/nonfinite, or the scaled moment residual exceeds the limit, raise
`BoundaryQuadratureInfeasible` with a stable code, input fingerprints, band
cell ids, weights/condition/rank, moments, residuals, and scale. Never return
a partially repaired result.

Populate changed ids only where the proposed weight differs from the old
weight. Confirm that every changed id is in the band and that every core
weight and original input is bitwise unchanged.

## A050-PY-VOLQ-05 Fingerprint

Hash the kind, units, normalized-fit definition, band width/radius, original
geometry fingerprint, boundary geometry fingerprint, old weights, polygon
moments, band mask/ids, condition/rank, proposed weights, and residuals using
explicit shape/dtype/order markers and little-endian FP64 bytes. Exclude timing
from the semantic fingerprint.
