# Test massless review r2

Source: `tests/warp/test_massless_review_r2.py`.

## A0-PY-TEST-TEST-MASSLESS-REVIEW-R2-01 - Procedure and assertions

```text
Independently compute Cartesian wall traction from ion ram pressure,
isotropic ion pressure, anisotropic electron pressure using bhat, and magnetic
Maxwell stress using physical B. Apply the signed normal to every term.
For an oblique field with nonunit magnitude compare the implementation to
this independent traction; no electron ram/electric Maxwell stress is added.
For current-free equilibrium require opposite-wall tractions cancel within
summation roundoff and retain momentum-rate units kg*m/s^2.
```

Covered top-level functions and fixture classes:

- `_independent_traction`
- `test_oblique_anisotropic_pressure_uses_physical_b_and_bhat_stress`
- `test_current_free_equilibrium_opposite_wall_tractions_cancel`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.
