# Test physics

Source: `tests/warp/test_physics.py`.

## A0-PY-TEST-TEST-PHYSICS-01 - Procedure and assertions

```text
Load independent native ordinary/stress exports from ORACLE_ROOT; build the
same real CPU stage. Dimensional comparisons reject zeroed small signals,
relative errors hidden by a unit floor, and nonfinite observations.
Check weighted wall projection constraints, idempotence and no work with
uniform/nonuniform density; reject nonpositive density before division.
Check cylindrical velocity-gradient adjoints, CGL pressure moments/sources
and invalid magnetic direction handling.
Solve electrostatic/longitudinal fields and recheck all full constraint rows.
Compare the materialized complex-linear AE against direct electric response
and its weighted adjoint. For a nontrivial forced stress state compare the
transverse solve to an independently assembled direct KKT system.
Verify kinetic density work is subtracted once, stale stages rejected,
nontrivial RHS and initialization constraints retained.
Compare stage/field/RHS/ledger arrays to independent native outputs using
physical force/energy scales. Compile the actual magnetic Warp kernel on CPU
and verify its formula, packed indexing and shared-array ownership.
```

Covered top-level functions and fixture classes:

- `parameters`
- `_weighted_dot`
- `_oracle_fixture`
- `_assert_native_close`
- `test_native_comparison_rejects_zeroed_small_nonzero_arrays`
- `test_native_comparison_rejects_relative_errors_hidden_by_unit_floor`
- `test_native_comparison_rejects_nonfinite_values`
- `_cell_integral`
- `_fJ0_term_reference`
- `_zero_reference`
- `_assert_oracle_array`
- `_physical_samples`
- `_forced_field_state`
- `test_wall_projection_is_constrained_idempotent_and_conserves_no_work`
- `test_wall_nonuniform_density_preserves_component_cell_order_and_full_rows`
- `test_wall_rejects_nonpositive_density_before_division`
- `test_velocity_gradient_weighted_adjoint_and_trace_identity`
- `test_pressure_tensor_cgl_sources_and_invalid_magnetic_input`
- `test_electrostatic_and_longitudinal_cover_all_rows`
- `test_materialized_response_is_complex_linear_and_weighted_self_adjoint`
- `test_forced_transverse_solve_matches_independent_direct_kkt`
- `test_power_ledger_subtracts_kinetic_density_term_once`
- `test_nontrivial_stage_and_initialize_reject_invalid_closed_reference`
- `test_physics_matches_independent_native_oracle`
- `test_physics_matches_native_stress_oracle`
- `test_warp_magnetic_force_kernel_compiles_and_matches_formula`
- `test_warp_magnetic_force_unpacks_and_retains_array_storage`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.
