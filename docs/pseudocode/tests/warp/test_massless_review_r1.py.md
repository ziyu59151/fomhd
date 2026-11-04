# Test massless review r1

Source: `tests/warp/test_massless_review_r1.py`.

## A0-PY-TEST-TEST-MASSLESS-REVIEW-R1-01 - Procedure and assertions

```text
Construct nonaxisymmetric density and compare shaped-sample electron velocity
and theta; do not divide flattened vectors by an incompatible scalar layout.
Integrate ion momentum in Cartesian coordinates and compare actual wall
traction and stage momentum defect; rotate the toroidal basis explicitly.
Build wall traces from the same normal interpolation row for every velocity
component, with ion internal and electron parallel-pressure transport factors.
Exclude electron kinetic transport/energy and unsupported spectral kinetic work.
Mutate E/E0 and the stored wall reaction independently; require Ohm/electric
power diagnostics to detect actual inconsistencies, not reconstruct a tautology.
Retain pressure truncation and node-edge work differences separately.
Ensure failure evidence uses the common helper without recursive redispatch;
public stage routing returns the massless result without a finite Darwin AE.
```

Covered top-level functions and fixture classes:

- `_nonaxisymmetric_density_state`
- `_cartesian_ion_momentum`
- `_record`
- `test_nonaxisymmetric_density_velocity_and_theta_use_shaped_samples`
- `test_cartesian_momentum_and_actual_stage_momentum_defect`
- `test_boundary_uses_actual_wall_traces_and_omits_electron_kinetic`
- `test_wall_trace_uses_normal_row_for_all_velocity_components`
- `test_ohm_uses_stored_reaction_and_detects_actual_field_mutation`
- `test_full_electric_power_not_pressure_only_field`
- `test_no_electron_kinetic_or_spectral_kinetic_channel`
- `test_failure_evidence_does_not_redispatch_through_stage`
- `test_scratch_public_stage_dispatch_reaches_massless_branch`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.
