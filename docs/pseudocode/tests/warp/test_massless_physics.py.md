# Test massless physics

Source: `tests/warp/test_massless_physics.py`.

## A0-PY-TEST-TEST-MASSLESS-PHYSICS-01 - Procedure and assertions

```text
Copy and canonicalize state; require immutable inputs and n_e derived from n_i.
Perturb stored compatibility electron momentum and require unchanged physical
massless fluxes, fields and RHS.
Use nonzero compatible current to verify J=Cdag B/mu0, shared electron number
flux, divergence and every full-wall row. Check current-free force balance.
With variable density, compare pressure adjoints, electric/internal power,
CGL tensor moments, density-interpolated theta and reaction work.
Verify massless totals omit electron kinetic and longitudinal electric energy;
inapplicable diagnostics remain unavailable, not passed finite-model gates.
Exercise active Fourier storage, alias audits, closed-wall corrections and
both explicit integrators through public routing with exactly one commit.
Reject zero physical mass metadata, incompatible initial current, invalid
number/pressure/magnetic input and unsupported manufactured-source hooks.
On failure retain raw candidate/check evidence without changing accepted state.
```

Covered top-level functions and fixture classes:

- `_copy_state_arrays`
- `_assert_same_state_arrays`
- `test_canonicalize_is_immutable_and_compatibility_momentum_is_not_consumed`
- `test_nonzero_wall_compatible_current_and_electron_flux_definitions`
- `test_uniform_closed_current_free_equilibrium_has_zero_force`
- `test_variable_density_pressure_adjoint_and_power_pairing`
- `test_wall_correction_checks_all_original_rows_and_zero_species_work`
- `test_pressure_tensor_moments_and_cgl_source_identities`
- `test_theta_consistency_uses_density_interpolation_not_global_ne`
- `test_totals_exclude_electron_kinetic_and_longitudinal_energy`
- `test_configured_hard_stage_gates_accept_physical_fixture`
- `test_multimode_storage_alias_and_full_wall_constraints`
- `test_zero_physical_electron_mass_metadata_is_rejected`
- `test_initialize_rejects_incompatible_current_and_commits_canonical_state`
- `test_explicit_advance_routes_and_preserves_massless_constraints`
- `test_manufactured_independent_electron_source_is_rejected`
- `test_failure_evidence_preserves_priors_and_adds_massless_requirements`
- `xp_tile`
- `c_entity_count`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.
