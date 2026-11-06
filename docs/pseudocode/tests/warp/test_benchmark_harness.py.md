# Test benchmark harness

Source: `tests/warp/test_benchmark_harness.py`.

## A0-PY-TEST-TEST-BENCHMARK-HARNESS-01 - Procedure and assertions

```text
Import the actual CLI by path; require import alone creates no run files.
Require the pilot CLI default to be RK4 and exercise Python case generation
without a native `fomhd_make_case` executable.
Require provenance to name the workload as historical finite-electron with
explicit `electron_inertia=1`.
Construct equal-final-time plans and a positive Hermitian density perturbation.
Archive executed tool bytes and verify hashes, without attributing a copied
snapshot to an unrelated parent Git worktree; a real checkout retains status.
Write synthetic failure NPZ and preserve NaN/Inf and last accepted identity.
Classify nested BudgetExhausted through the exception chain; run a real short
CPU timeout and require exit 124, top/nested budget_exhausted statuses,
failed raw arrays and rejection report. Preserve wrapper context in messages.
Exclude positivity and zero-threshold records from normalized residual ratios.
Refuse output overwrite and use warn guard only for intentional scans.
Check selected CUDA ordinal/stream provenance with a scoped device double.
Fit rotating complex and standing real signals with known frequencies; mark
short, monotonic and aperiodic records unavailable without invented signs.
Render an all-failed plot without fabricated y values.
Run the actual small CPU pilot and plot it; verify accepted time, state/check
artifacts, explicit device/no fallback, timings and unavailable short frequency.
```

Covered top-level functions and fixture classes:

- `_load`
- `harness`
- `_environment`
- `test_tool_source_paths_resolve_executed_files`
- `test_import_has_no_run_side_effect`
- `test_pilot_cli_default_integrator_is_rk4`
- `test_benchmark_provenance_declares_historical_finite_scope`
- `test_scan_plan_preserves_equal_final_time`
- `test_density_seed_is_hermitian_and_positive`
- `test_failure_arrays_preserve_nan_and_last_accepted_state`
- `test_budget_classifier_traverses_actual_solver_failure`
- `test_wrapped_budget_updates_nested_run_status`
- `test_source_snapshot_does_not_inherit_parent_git_identity`
- `test_actual_checkout_retains_git_status`
- `test_diagnostic_summary_excludes_positivity_and_zero_threshold`
- `test_output_creation_refuses_overwrite`
- `test_rendered_case_uses_dt_guard_only_for_scan`
- `test_frequency_fit_marks_short_records_unavailable`
- `test_device_provenance_uses_nonzero_backend_ordinal`
- `test_frequency_fit_measures_actual_samples_shape`
- `test_frequency_measures_real_standing_wave_without_sign`
- `test_frequency_measures_complex_rotating_mode_with_known_sign`
- `test_frequency_rejects_monotonic_growth`
- `test_frequency_rejects_aperiodic_failure_tail`
- `test_all_failed_plot_does_not_require_successful_y_values`
- `test_real_cpu_smoke_and_plot`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.
