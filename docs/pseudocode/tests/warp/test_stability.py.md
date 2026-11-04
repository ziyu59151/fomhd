# Test stability

Source: `tests/warp/test_stability.py`.

## A0-PY-TEST-TEST-STABILITY-01 - Procedure and assertions

```text
Compare midpoint and RK4 amplification polynomials to analytic oscillator
values. Require midpoint growth for any nonzero imaginary argument and RK4's
2*sqrt(2) boundary. Compare long-horizon amplitude/phase errors to direct powers.
Compute SI frequency scales and verify Hz/rad-s conversion, density square-root
scaling, finite-inertia whistler saturation and massless k-squared scaling.
Reject invalid inputs/budgets. Estimate actual stage bounds for each electron
model: plasma/cyclotron/pressure for finite inertia, Hall/ion/Alfven/bulk terms
for massless; finer resolution must increase the Hall estimate.
Check error policy rejects before commit; warn records failure and allows only
otherwise valid transactions. Verify accepted step and explicit guard records.
```

Covered top-level functions and fixture classes:

- `module`
- `test_midpoint_growth_has_no_nonzero_imaginary_stable_interval`
- `test_rk4_boundary_and_finite_horizon_phase`
- `test_frequency_units_and_massless_hall_limit`
- `test_invalid_guard_configuration_is_rejected`
- `test_stage_frequency_estimates_follow_the_selected_model`
- `test_frequency_guard_rejects_and_warn_override_is_explicit`
- `test_real_time_transaction_guard_and_warn_override`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.
