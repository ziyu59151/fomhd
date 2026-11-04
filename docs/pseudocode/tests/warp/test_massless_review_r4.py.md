# Test massless review r4

Source: `tests/warp/test_massless_review_r4.py`.

## A0-PY-TEST-TEST-MASSLESS-REVIEW-R4-01 - Procedure and assertions

```text
Scale background field by one and one hundred; require upstream/total bounds
and sample/modal transform evidence retained, with the same physical gate.
Scale edge volume weights and consistent reaction multipliers together;
require dimensional work and its scale multiply exactly once.
Inject a nonphysical reaction and require wall-work records fail the derived
gate at both field magnitudes.
Evaluate K=1 storage and require finite nonzero-mode-compatible work metrics;
this storage check alone does not establish a nonzero-harmonic perturbation.
```

Covered top-level functions and fixture classes:

- `_record`
- `_operator_work_scale`
- `test_upstream_arithmetic_bound_survives_field_rescaling`
- `test_volume_rescaling_multiplies_work_and_scale_once`
- `test_injected_wall_work_fails_derived_gate`
- `test_nonzero_k_wall_work_metrics_and_transform_evidence`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.
