# tools/clt_boundary_coefficients.py

Implementation-level pseudocode for the source-derived coefficient oracle.
The source map is [CLT_BOUNDARY_SOURCE_MAP.md](../../CLT_BOUNDARY_SOURCE_MAP.md).

## Public contract

Inputs are dimensionless ratios to grid spacing and a side (`upper` or `lower`).
The oracle is only a coefficient/moment oracle; it does not solve a PDE, build a
geometry, or model CLT transport.

```text
PointKind:
    REGULAR = 1
    IRREGULAR = 2
    OUTSIDE_DROPPED = 4
    INSIDE_DROPPED = 5

classify_directional_point(distance_h, inside):
    if not inside: return OUTSIDE_DROPPED
    if distance_h <= 0: return INSIDE_DROPPED
    if distance_h < 0.5: return INSIDE_DROPPED
    return REGULAR

SOURCE_ASSOCIATION_RATIO = 1.0001
SOURCE_ASSOCIATION_LIMIT_H = 1.0001 * 1.5

irregular_rule_for_offset(boundary_offset_h):
    if offset < 0.5: raise DroppedPointError
    if offset < 1.5: return "four_point"
    if offset < 2.5: return "five_point"
    raise outside retained source association range

IrregularStencil:
    rule, side, boundary_offset_h, nodes_h,
    first_derivative_weights, second_derivative_weights

irregular_stencil(rule, boundary_offset_h, side):
    validate finite positive offset and distinct source nodes
    if boundary_offset_h < 0.5:
        raise DroppedPointError ("directional stencil is unused")
    reject a requested rule that does not match the geometric offset
    evaluate the exact source formula for the selected branch:
        "four_point", 0.5h <= offset < 1.5h:
            source +/-1 d1fbp/d1fbm and moment-solved d2fbp/d2fbm
        "five_point", 1.5h <= offset < 2.5h:
            source +/-2 d1fc/d2fc coefficient formulas
    reflect nodes and weights when side is lower
    return weights indexed by nodes_h
```

The source writes differences such as `A*(fp1-f0)`. The oracle expands each rule into
the equivalent node-weight vector, then verifies the polynomial moments directly.
At idealized uniform geometry, the exact `1.5h` threshold belongs to the five-point
branch. The source search uses `deltaxz=1.0001*(1+0.5)*h=1.50015h`; this is reported
as an association tolerance, not another geometric classification band.

## Tests

1. Check directional drop classification at distances `0.25`, `0.49`, `0.5`, `0.51`,
   and `0.75`; equality at `0.5h` is retained, values below it are dropped.
2. Check exact/near-tangent behavior using zero distance and values immediately below
   and above `0.5h`.
3. Check the four-node first derivative preserves degrees 0-3 at the origin and has
   the source degree-4 moment `2*b*h^4`, giving third-order first derivative.
4. Check the four-node second derivative preserves degrees 0-3 and has degree-4
   moment `2*(3*b-2)*h^4`, giving generally second order.
5. Check the five-node first derivative preserves degrees 0-4 and has degree-5
   moment `-2*b*h^5`, giving fourth order.
6. Check the five-node second derivative preserves degrees 0-4 and has degree-5
   moment `(4-2*b)*h^5`, giving generally third order.
7. Check geometric offsets `0.5`, `1.25`, `2.25` and both sides of `0.5h` and `1.5h`.
8. Check lower-side reflection for first and second derivatives.
9. Check that a directionally dropped point raises and therefore no irregular stencil
   is silently used.

The initial task retained its red test log in `LOGS/`. Repair 1's three stages are
retained under `scratch/repair1_logs/`: missing API, the first corrected formula run,
and the final green run.
