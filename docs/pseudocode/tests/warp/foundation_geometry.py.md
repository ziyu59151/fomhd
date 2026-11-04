# Foundation geometry

Source: `tests/warp/foundation_geometry.py`.

Grid-envelope regression: build the same full rectangle at 12 and 14 cells,
with three physical length scales. Accept rounding-level endpoint differences,
preserve every active cell and wall, and reject a true exterior polygon.

## A0-PY-TEST-FOUNDATION-GEOMETRY-01 - Procedure and assertions

```text
Build a small rectangular mesh; check cell/face/vertex ordering, closed-wall
count and signs, positive cylindrical weights and Hausdorff bounds.
Build a stepped disk and require both radial and axial wall directions and a
closed wall polyline.
Reject malformed/self-intersecting polygons, invalid/nonuniform grids and
memory-cap violations before materializing oversized geometry.
Compare point-segment distance against interior, endpoint and degenerate
segment analytic distances.
```

Covered top-level functions and fixture classes:

- `rectangle`
- `test_rectangle_topology_order_and_weights`
- `test_stepped_disk_has_closed_both_direction_staircase_wall`
- `test_polygon_rejections`
- `test_grid_rejections`
- `test_geometry_memory_cap_is_rejected`
- `test_grid_envelope_accepts_roundoff_but_rejects_exterior_polygon`
- `test_point_segment_distance_clamps_projection`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.
