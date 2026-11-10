# `polygon_geometry.py` implementation pseudocode

## A050-POLY-GEO-01 identity and host ownership

`build_polygon_geometry(R, Z, polygon, options=None)` returns an immutable
`PolygonGeometry(kind="polygon_dual_v1")`. NumPy FP64 owns all geometry and
reconstruction data. `R` and `Z` are copied Cartesian generator grids; they
are not interpreted as stair-step face coordinates. The supplied convex
polygon is the physical domain for one radian of toroidal extent.

The option record contains:

```text
memory_cap_bytes = 256 MiB
grid_rtol = 1e-12
guard_factor = 0.45
vertex_rtol = 4096*eps
min_quality = 1e-14
```

The geometry estimate is checked before the active mask, Voronoi diagram, cell
polygons, or topology are allocated. A cap stop reports the estimated bytes and
the limit.

## A050-POLY-GEO-02 input certification

1. Require finite, strictly increasing, uniform one-dimensional `R` and `Z`
   grids with at least two nodes, positive spacing, and `R[0]-hR/2 > 0`.
2. Require at least three finite unique polygon vertices. Remove one repeated
   closing vertex.
3. Reject duplicate vertices, zero-length or self-intersecting/touching edges,
   ambiguous orientation, zero area, and clockwise/counterclockwise changes.
4. Canonicalize a valid input polygon to counterclockwise order. Keep the
   supplied point sequence in `polygon_input`; use the canonical sequence in
   `polygon`.
5. Do not move polygon coordinates. The Cartesian node grids are generator
   coordinates, not a required staircase bounding rectangle; coverage is
   enforced against the full polygon after clipping.

## A050-POLY-GEO-03 guarded generators

For every grid point strictly inside the convex polygon, compute its exact
Euclidean distance to the polygon boundary. Retain it when

```text
distance_to_boundary >= guard_factor*min(hR,hZ)
```

The default factor is 0.45. A removed generator contributes no cell. More
importantly, no area is removed from the physical polygon: after filtering,
the Voronoi bisectors are recomputed from the retained generators and every
cell is clipped by the full polygon. The original grid index is retained as
the cell key, and `active[i + nr*j]` records whether that key survived.

Reject if no generator survives, fewer than three noncollinear generators
remain, or the Voronoi/Delaunay construction is numerically ambiguous.

## A050-POLY-GEO-04 clipped Voronoi cells

Use `scipy.spatial.Voronoi` (with a Delaunay-neighbor fallback only to recover
the same ridge graph) on the retained generator coordinates. Its ridge graph
is linear in the number of generators; cell clipping never scans all generator
pairs.

For cell `i`, start with the complete CCW polygon. For each ridge connecting
`i` to `j`, clip against

```text
(x - (g_i+g_j)/2) dot (g_j-g_i) <= 0
```

using Sutherland-Hodgman clipping. Duplicate consecutive points and
certified-near-collinear points are removed. A resulting cell must be a
positive-area simple CCW polygon. Intersection vertices from different cells
are identified by a deterministic spatial hash at the certified tolerance
`vertex_rtol*coordinate_scale`; endpoints that are closer than this tolerance
are the same geometric vertex. Ambiguous clusters are rejected rather than
merged by an uncertified guess.

The resulting polygons partition the complete input polygon. For each polygon,
compute with extended-precision accumulation:

```text
area_c = 1/2 sum cross(x_a,x_b)
M0     = 1/6 sum (R_a+R_b) cross(x_a,x_b)
Zc     = 1/6 sum (Z_a+Z_b) cross(x_a,x_b) / area_c
Rc     = M0/area_c
```

Do not floor a positive measure. A non-positive or non-finite cell area or
R-weighted volume is a geometry failure. `cells.R` and `cells.Z` contain the
area centroid; `generator_R` and `generator_Z` remain separate metadata.

## A050-POLY-GEO-05 oriented facets and incidence

Each CCW cell edge is oriented so its outward unit normal is

```text
n = (dZ, -dR)/ell
t = (-n_Z, n_R) = (dR,dZ)/ell
```

For a first owner, collect the directed canonical endpoints. A matching edge
from the second owner must have the opposite direction. At most two owners are
accepted. Faces follow deterministic cell and edge enumeration order.
`face_vertices` is the
first-owner direction, `face_normals=(n_R,n_Z)` is outward from the first
owner, and `face_tangents=(-n_Z,n_R)`. Interior face normals point from the
first owner toward the second; the second owner entry is `-1` on a boundary.

For an interior face:

```text
ell = |x_end-x_start|
d   = |(generator_second-generator_first) dot n|
```

For a boundary face, `d` is the positive normal distance from its owner
generator to the facet midpoint plane. `ell` and `d` are distinct stored
quantities and are never substituted for one another.

Construct:

```text
b1 = scipy.sparse.csr_matrix(Nf,Nv), row = end-start
b2 = scipy.sparse.csr_matrix(Nc,Nf), row = +first,-second
```

Require `b2 @ b1` to have exactly zero stored entries. The face and vertex
tables are host `EntityTable` records; `r_faces` is the same face object only
as a count/storage alias, and `z_faces` is an empty `EntityTable`.

## A050-POLY-GEO-06 measures and walls

For each cell, use the retained generator `c` as the fan center. For every
directed edge from `v_i` to `v_j`, let `m=(v_i+v_j)/2` and assign the two
actual triangles `(c,m,v_i)` and `(c,m,v_j)` to `v_i` and `v_j`:

```text
a = 1/2 |cross(v-c, m-c)|
av_v += a
wv_v += a*(c_R+v_R+m_R)/3
Rbar_v = wv/av
```

Every selected triangle must have positive area. For each cell, the two
triangles from every edge must recover the independently computed cell area
and R-volume within the roundoff gate; their sum over all cells therefore
recovers the full polygon measures. No equal area-per-vertex share is used.
Use

```text
Rbar_r = face midpoint R
M1 = [Rbar_r*ell*d, M0]
MB = [wv, Rbar_r*ell*d]
lr = d
lz = empty
```

`faces.R/Z` are exact facet midpoints. `vertices.R/Z` are the canonical
endpoints. `wall_polyline` follows the actual polygon boundary. Every boundary
face produces one `PlaneWall`:

```text
component = "plane_face"
face = global face index
sign = 1
near_cell = first owner
next_cell = nearest other cell
vertices = canonical endpoint pair
area_per_radian = midpoint R * ell
normal_R/Z, tangent_R/Z, radius
```

`cell_wall_distance` is the true point-to-closed-polygon distance at each
cell centroid.

## A050-POLY-GEO-07 coverage and quality diagnostics

Report at least:

```text
polygon_area, polygon_M0
coverage_area_error = |sum(area_c)-polygon_area|
coverage_M0_error   = |sum(M0)-polygon_M0|
coverage_area_relative, coverage_M0_relative
min_cell_area, min_M0, min_face_length, min_face_distance
min_vertex_area, min_vertex_volume
min_face_length_to_distance
retained_generators, removed_generators, input_generators
```

Coverage is checked against tight roundoff-scale bounds, not hidden by filling
or renormalizing area. `active_cells`, `wall_segments`, and the existing signed
polygon area remain available through the geometry diagnostics.

## A050-POLY-GEO-08 compatibility

`is_active(i,j)` reports whether the original generator key survived the
guard. `cell_id(i,j)` returns the stable cell index for a retained key. Legacy
`r_face_id` remains meaningful through the `r_faces` alias. No API in this
module writes `E_R`, `B_Z`, or a staircase `lz`; those interpretations are
forbidden for `polygon_dual_v1`. Finite-mass storage is also explicitly
unavailable.
