# Test mms

Source: `tests/warp/test_mms.py`.

## A0-PY-TEST-TEST-MMS-01 - Procedure and assertions

```text
Build the exact tagged coupled trajectory and analytic derivative; change a
supplied numerical trial and require the MMS source remain unchanged.
Advance real midpoint at 8/16/32 steps and RK4 at 2/4/8; compare each state
component and combined normalized error to the exact trajectory. Require
orders in [1.7,2.3] and [3.5,4.5], nontrivial EL/ET/B response and hard gates.
Verify source calls and quadrature accumulate each interval once; retain
physical production ledger separately from the manufactured source ledger.
Omit ion pressure forcing and require measurable continuum error even when
ordinary physical gates pass. Reject a divergent manufactured magnetic source.
Refine actual smooth spatial curl/divergence/adjoint operators and require
interior order above 3.5; preserve wall/total errors separately and run a
negative spatial sign control.
Refuse CUDA outside Slurm and refuse existing output directories.
Keep production dt reports, complete checks, ledgers and failure NaN/Inf arrays.
Hash `src/fomhd` runtime/harness/dependency sources and archives, test execution
from a frozen checkout and from outside the repository, reject mixed-package
origins, and aggregate negative controls once. Inject a CLI failure and verify
exact phase/check/partial-array evidence.
```

Covered top-level functions and fixture classes:

- `_assert_close_state`
- `test_source_uses_exact_reference_not_numerical_state`
- `test_trajectory_analytic_derivative_is_nonzero_and_tagged`
- `test_midpoint_discrete_mms_is_second_order_with_coupled_response`
- `test_midpoint_source_is_accumulated_once_per_interval`
- `test_rk4_discrete_mms_is_fourth_order_beyond_roundoff`
- `test_omitted_ion_pressure_source_fails`
- `test_raw_sidecar_aggregate_counts_negative_control_once`
- `test_smooth_spatial_curl_divergence_and_adjoint_converge_interior`
- `test_cuda_is_refused_without_slurm_allocation`
- `test_cli_output_directory_must_be_new`
- `test_harness_imports_current_main_solver`
- `test_production_guard_reports_are_retained_per_step`
- `test_raw_checks_and_ledgers_are_preserved_with_hashes`
- `test_raw_check_serialization_detects_failed_control`
- `test_failure_partial_arrays_preserve_nan_and_inf`
- `test_source_magnetic_divergence_negative_control`
- `test_source_snapshot_and_identity_cover_runtime_and_harness`
- `test_runtime_origin_uses_imported_package_and_rejects_mismatch`
- `test_exact_checkout_executes_and_provenance_works_outside_repository`
- `test_frozen_checkout_provenance_tracks_executed_package`
- `test_cli_failure_record_preserves_partial_arrays`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.
