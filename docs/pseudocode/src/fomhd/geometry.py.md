# `src/fomhd/geometry.py` implementation pseudocode

## A0-PY-GEO-01 Input and polygon validation

1. Estimate setup bytes before allocation and reject a cap exceedance:
   `128*nr*nz + 96*(nr*nz+nz*(nr+1)+nr*(nz+1)) +
   128*polygon_vertices`.
2. Require at least two monotone, finite, uniform R and Z grid nodes with
   positive spacing and `R_0-hR/2>0`.
3. Require a finite polygon with at least three unique vertices after removing
   one repeated closing vertex. Reject duplicates, zero-length edges,
   self-intersection/touching, ambiguous orientation, and uncertifiable area.
4. Require every polygon coordinate inside the grid rectangle including half
   cells. For each axis allow only coordinate rounding of
   `8*eps*max(abs(first_node),abs(last_node),abs(spacing))` in the envelope
   comparison. Do not move coordinates or change the active-cell predicate.
   Reject polygons beyond this arithmetic envelope.

## A0-PY-GEO-02 Active mask and ordered entities

1. Include a cell when its center is strictly inside the polygon using the
   native half-open ray-crossing rule.
2. Enumerate unique cells, R-faces, Z-faces, and vertices from active cells.
3. Order every entity by `(second, first)`, with dense index
   `first + width*second`.
4. Store the active mask as the native flat dense vector with index
   `i + nr*j`; do not reorder it by active-cell ID.
5. Compute physical coordinates by the native half-cell staggering:
   R-faces at `(i-1/2,j)`, Z-faces at `(i,j-1/2)`, and vertices at
   `(i-1/2,j-1/2)` in offset coordinates.
6. Mark faces with one active neighbor as boundary. Mark vertices with fewer
   than four active incident cells as boundary; reject diagonal-only contact.
7. Reject disconnected cells, nonmanifold or disconnected walls, domains
   without a wall, and domains whose Euler characteristic is not one.

## A0-PY-GEO-03 Walls and orientation

1. For each boundary R-face with active right cell, set `sign=-1`, near/right
   cell and next/inward left cell. Otherwise set `sign=+1`, near/left cell and
   next/inward right cell.
2. Use the corresponding upper/lower rule for boundary Z-faces.
3. Require a second inward cell for every wall-adjacent entity.
4. Sort walls by component (`r_face`, then `z_face`) and face index.
5. Traverse the connected wall cycle from the first sorted wall's first vertex
   to produce `wall_polyline`.

## A0-PY-GEO-04 Measures

1. Cell measure: `M0=Rc*hR*hZ`.
2. R-face length factor: `lr=hR` for two neighbors, else `hR/2`.
3. Z-face length factor: `lz=hZ` for two neighbors, else `hZ/2`.
4. Boundary R-face mean radius is half the sum of face and near-cell radius;
   an interior R-face uses its own radius.
5. Vertex area accumulates a quarter cell area per incident cell. Vertex
   cylindrical weight accumulates
   `(R_hi^2-R_lo^2)*hZ/4`. Require both positive and set
   `Rbar_v=wv/av`.
6. Preserve concatenation order:
   `M1=[R-face, cell, Z-face]` and `MB=[Z-face, vertex, R-face]`.
7. A wall area per radian is `R_face*hZ` or `R_face*hR`.

## A0-PY-GEO-05 Staircase distance

1. Let `eta=min(hR,hZ)/8`, sample polygon and wall segments at spacing bounded
   by `eta`, and take maximum point-to-opposite-segment distance in both
   directions. Report `[lower,lower+eta/2]`.
2. Store each cell-center distance to the closed staircase wall.
3. `point_segment_distance` clamps the projection parameter to `[0,1]`;
   a zero-length segment reduces to endpoint distance.
