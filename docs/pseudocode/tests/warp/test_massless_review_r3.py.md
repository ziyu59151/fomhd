# Test massless review r3

Source: `tests/warp/test_massless_review_r3.py`.

## A0-PY-TEST-TEST-MASSLESS-REVIEW-R3-01 - Procedure and assertions

```text
Compute wall-reaction work and cancellation scale independently from actual
edge fluxes, electric reaction and cylindrical M1 weights, in watts.
Compare stored dimensional reference and max(configured,arithmetic) threshold.
Set reaction to zero and require zero work/scale; multiply reaction and require
linear scaling exactly once, without a second mass-matrix factor.
Inject nonzero reaction work and require it to exceed the arithmetic bound.
R4 separately tests rejection by the complete configured hard threshold.
Transform comparisons are observations.
```

Covered top-level functions and fixture classes:

- `_record`
- `_expected_wall_work_scale`
- `test_wall_work_reference_uses_actual_edge_pairing_in_watts`
- `test_wall_work_scale_zero_rescaling_and_nonzero_injection_controls`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.
