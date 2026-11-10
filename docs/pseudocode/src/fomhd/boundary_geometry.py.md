# `src/fomhd/boundary_geometry.py` implementation pseudocode

## A050-PY-BGEO-00 Scope and public contract

`build_boundary_geometry(geometry, kind="clt_intercept")` is a geometry-only
probe for the existing structured `Geometry`. It does not create a mesh, move
cell centers, alter `geometry.active`, replace `r_faces`/`z_faces`, or supply
field values. All physical lengths are SI metres (`m`), areas are `m^2`, and
cylindrical volume integrals are `m^3/rad`. Every array returned to the caller
has `writeable=False`; records are frozen tuples. The returned fingerprint is
SHA-256 over the semantic NumPy arrays and ordered records, so equal inputs
produce equal fingerprints without relying on Python object identity.

Accept only:

1. `kind="clt_intercept"` for a strictly convex, nondegenerate polygon wholly
   in `R>0`;
2. `kind="staircase"` as a read-only identity snapshot of the existing
   staircase polyline and wall geometry.

Raise `BoundaryGeometryError` for unsupported kinds, nonconvex polygons,
collinear/duplicate vertices, nonpositive radius, nonfinite coordinates,
ambiguous ray contacts, or inconsistent Geometry topology. Never fill a wall
value from a point outside the polygon.

Let the finite geometry tolerance be

```text
tol_m = 1e-12 * max(1, max(abs(R)), max(abs(Z)), polygon span)
tol_angle = 1e-12  # dimensionless
```

Use `tol_m` only for coordinate-scaled quantities with length units: duplicate
vertices, point/line distances, endpoint contact, boundary membership, ray
origin offsets, and area cancellation. A ray direction is unit length, so

```text
cross(unit_ray, segment) / |segment|
```

is a dimensionless sine. Compare it with `tol_angle`, never with `tol_m`.
Unit mismatches are implementation errors even when they happen to pass at
unit-scale coordinates. Any contact within `tol_m` of a real polygon vertex is
rejected as ambiguous; never select or average one incident normal. Duplicate
contacts may reduce to one hit only when they come from one source segment.

Validate the supplied R/Z arrays with the existing source Geometry grid
validator and its default `GeometryOptions.grid_rtol=1e-12`, not merely with a
strictly-increasing check. This is the same support on which the source builder
operates and is required because directional classes and logical dual
rectangles use only `hR=R[1]-R[0]` and `hZ=Z[1]-Z[0]`. Reject a nonuniform grid
through `BoundaryGeometryError`; do not duplicate the full Geometry builder.

## A050-PY-BGEO-01 Convex polygon and ordered physical edges

1. Copy the canonical Geometry polygon coordinates to host FP64 arrays without
   changing the source objects.
2. Remove one repeated closing vertex. Reject fewer than three vertices,
   duplicate vertices, zero-length edges, and every finite coordinate outside
   `R>0`.
3. For every triple of consecutive vertices, require a nonzero orientation sign
   with magnitude above the coordinate-scaled tolerance. Require all signs to
   agree; otherwise reject as nonconvex or degenerate.
4. Preserve polygon input order and edge index. For a segment `a -> b`, set

   ```text
   ell = hypot(bR-aR, bZ-aZ)
   t = ((bR-aR)/ell, (bZ-aZ)/ell)
   ```

   For CCW input use outward `n=(tZ,-tR)`; for CW input use
   `n=(-tZ,tR)`.
5. Build the single true-wall quadrature table for all polygon edges with
   two-point Gauss-Legendre nodes
   `u=1/2*(1-s/sqrt(3))` and equal one-dimensional weights transformed by
   `ell/2`. Store edge index, Gauss index, point, outward normal, length weight
   `ell*w`, and cylindrical weight `R*ell*w`. Use this table for every later
   wall consumer.

## A050-PY-BGEO-02 Directional intercepts

For each existing active cell in the Geometry cell-table order, cast four
closed-line rays from its unchanged centre: `R-`, `R+`, `Z-`, `Z+`.

For each ray:

1. Solve its intersection with every polygon edge. Skip nonparallel misses.
   Treat a collinear/near-collinear overlap as tangent ambiguity.
2. Retain only ray parameters `t>=0` within the finite tolerance. Deduplicate
   contacts within `tol_m`.
3. Require exactly one contact point from one source segment. If one physical
   point is sourced by multiple edges, reject the contour-vertex ambiguity.
4. Reject any contact within `tol_m` of a polygon vertex. This includes
   near-tangent endpoint contacts; do not silently choose a normal.
5. Store the original cell id/key, direction (`R` or `Z`), side (`-` or `+`),
   origin, exact contact, nonnegative directional distance, actual piecewise-
   linear outward normal, polygon edge id, and deterministic intercept id.

If an active centre lies on the polygon within `tol_m`, mark it explicitly as
`boundary_on_grid`. For the half-plane direction with `dot(direction,n)>0`,
use the zero-distance contact at the centre. The opposite ray uses its positive
intersection. A tangent direction or vertex contact is not guessed; it is
rejected as ambiguous. A zero distance is therefore never interpreted as an
exterior-filled value.

Store dense `boundary_on_grid` metadata using the original flat order
`i + nr*j`.

## A050-PY-BGEO-03 Directional CLT point classes

Create uint8 metadata arrays with the original dense `(nr,nz)` ordering:

```text
4 outside
5 inside but dropped in that direction
1 retained regular candidate
2 retained irregular within two layers of dropped/outside
```

For each direction independently:

1. Set every grid point outside or on the exterior half-open side of the
   polygon to `4`. Existing active cells remain active regardless of class.
2. For an active centre, take the minimum of its two directional intercept
   distances in that coordinate axis.
3. Source CLT uses the literal directional test `distance < 0.5*h_axis`.
   FOMHD adapts that comparison to

   ```text
   distance < 0.5*h_axis - tol_m
   ```

   so arithmetic within `tol_m` of equality is retained as base class `1`.
   This guarded comparison is a project numerical adaptation, not an exact
   clone of CLT's source expression. Equality and values above the guard are
   retained; values distinctly below the guard are class `5`.
4. Scan in increasing coordinate order and mark both a retained candidate and
   its retained neighbour as irregular `2` when one side is dropped/outside.
   This reproduces CLT's two directional layers without deleting a cell.
5. Keep both the evanescent base class and the final class. Classification is
   separate metadata: it does not change physical volume, links, dual geometry,
   storage layout, or boundary conditions.

`boundary_on_grid` plus the final base/final classes expose boundary-on-grid
ambiguity rather than silently converting it to an exterior point.

## A050-PY-BGEO-04 Existing face links

For every existing `r_face` and `z_face`, create one axis link in the Geometry
face-table order.

1. An interior face has two active cell neighbours. Join the lower-coordinate
   R centre to the higher-coordinate R centre for an R face, or the lower-
   coordinate Z centre to the higher-coordinate Z centre for a Z face.
2. A boundary face has one active centre. Join that last-inside centre to the
   intercept in the missing outward side. Store the true intercept id and
   source edge id.
3. Store face id/key, `axis` (`R` or `Z`), physical component (`E_R` or `E_Z`),
   interior flag, ordered endpoints, length, one or two original cell ids, and
   the boundary intercept id (`-1` for interior links).

The links are metadata over the existing Cartesian face variables. They do
not introduce a normal-face variable and do not change component layout.

## A050-PY-BGEO-05 Existing-vertex metric probe

For every existing vertex, form the logical rectangle between adjacent R and Z
cell-centre lines. At a grid edge, use the adjacent virtual cell-centre line
one spacing beyond the grid so the half-envelope remains representable. Clip
this CCW rectangle against every half-plane of the convex polygon using
Sutherland-Hodgman clipping, removing duplicate and collinear output points.

For each nonempty loop, compute:

```text
area2 = sum cross(p_i,p_(i+1))
area = area2/2
R_integral = sum (R_i+R_j)*cross_i,j / 6
Z_integral = sum (Z_i+Z_j)*cross_i,j / 6
centroid = (R_integral/area, Z_integral/area)
```

Store original vertex id/key and a read-only CCW loop. An untouched regular
interior loop is exactly
`[R_lo,Z_lo],[R_hi,Z_lo],[R_hi,Z_hi],[R_lo,Z_hi]`, with area `hR*hZ`.

The logical dual rectangles have disjoint interiors and tile the plane before
clipping, so their clipped areas add. Expose

```text
uncovered_area = abs(polygon_area) - sum(dual_area)
coverage_fraction = sum(dual_area)/abs(polygon_area)
```

with cancellation checked against `tol_m`. Report positive uncovered area;
never enlarge, relocate, merge, or delete a dual neighbourhood to hide it.

## A050-PY-BGEO-06 Immutability and fingerprint

Return a frozen `BoundaryGeometry` containing:

- explicit SI unit names;
- original grids, active mask, cell/face/vertex keys and ids;
- directional intercept records and dense boundary/class metadata;
- existing-face axis links and existing-vertex dual records;
- the common true-wall quadrature table;
- polygon area, covered/uncovered area and fingerprint.

Set every NumPy array to read-only before constructing the result. Set source
Geometry arrays read-only only if they were already read-only; never mutate or
reassign the input. Hash kind, units, grids, active mask, polygon coordinates,
derived arrays, and ordered records with explicit shape/dtype/order markers and
FP64 little-endian byte encodings.

For `kind="staircase"`, copy the existing `geometry.wall_polyline` positions,
wall identities, and face positions into the same immutable representation and
hash them without changing or snapping any coordinate.
