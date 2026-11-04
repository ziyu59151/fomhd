# Test runtime

Source: `tests/warp/test_runtime.py`.

## A0-PY-TEST-TEST-RUNTIME-01 - Procedure and assertions

```text
Use small tagged transaction doubles to inspect stage times and ownership:
midpoint uses half/full trials; RK4 uses four explicit RHS stages and its
weighted interval quadrature. Only accepted endpoint publication mutates the
committed state and cumulative ledger.
Fail K3 or a source payload, retain completed stage/raw source evidence and
preserve the old commit. Massless routes through the same transaction checks.
Reject unknown TOML, unsafe NPZ/object arrays, duplicate/missing/unavailable
hard diagnostics and inconsistent configuration. Verify default physical mass,
integrator and dt budgets.
Use weighted device reductions, physical-real inverse defaults, correct
energy/time power units, signed wall trace weights and dimensional source
bounds; test failure of deliberate work/source errors.
Round-trip state codecs and explicit contiguous backend uploads.
Publish checkpoint before pointer with matching hash; inject atomic-publish
failure and preserve old target plus unpublished temporary evidence.
Restart a complete ledger and reject tampering/incomplete cumulative data;
check source archive, model/time provenance and typed nonfinite failures.
```

Covered top-level functions and fixture classes:

- `Failure`
- `Tables`
- `runtime`
- `_tag`
- `_state`
- `_stage`
- `test_midpoint_uses_half_then_full_step_and_commits_once`
- `test_rk4_uses_four_explicit_stages_and_weighted_quadrature`
- `test_rk4_k3_failure_retains_completed_k2_and_k3_tag`
- `test_rejected_source_payload_is_retained_before_validation`
- `test_rejected_advance_leaves_committed_state_and_ledger_unchanged`
- `test_time_routes_massless_branch_into_transaction_validation`
- `test_load_case_rejects_unknown_key_before_creating_output`
- `test_safe_npz_rejects_object_dtype`
- `test_atomic_checkpoint_pointer_is_published_after_file`
- `test_atomic_publish_fault_preserves_target_and_unpublished_temp`
- `test_failure_record_uses_typed_nonfinite_values`
- `test_gate_rejects_duplicate_and_unavailable_hard_records`
- `test_gate_records_missing_diagnostic_raises_failure`
- `test_parameter_defaults_accept_integrator_contract`
- `test_config_defaults_preserve_mass_metadata_and_midpoint`
- `test_measure_uses_device_weighted_l2_reference`
- `test_diagnostic_inverse_defaults_to_physical_real_checks`
- `test_field_exchange_converts_integrated_energy_to_watts`
- `test_wall_trace_uses_normal_weights_and_unsigned_electric_trace`
- `test_check_source_divergence_uses_exact_subunit_physical_scale`
- `test_state_encoder_round_trip_preserves_compact_layout`
- `test_codec_state_upload_is_explicit_and_contiguous`
- `test_checkpoint_and_restart_preserve_ledger_and_reject_tamper`
- `test_cumulative_json_requires_complete_zero_initial_ledger`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.
