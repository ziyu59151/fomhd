# Test model routing

Source: `tests/warp/test_model_routing.py`.

## A0-PY-TEST-TEST-MODEL-ROUTING-01 - Procedure and assertions

```text
Build finite or massless parameters with configured species IDs and actual
closed-wall-compatible nonzero current; create safe TOML/NPZ inputs.
Check canonical massless production ignores stored compatibility momentum.
Run actual CPU midpoint/RK4 transactions and commit state/ledger exactly once.
Run the real CLI with omitted and explicit midpoint configuration; require
omitted configuration to select RK4 and explicit midpoint to remain midpoint.
Reject invalid candidates and unsupported source hooks without modifying any
accepted array, time, step, trial or cumulative ledger; retain raw candidates.
Test error/warn dt policy and persistence of the full estimate before/after
acceptance. Warn never bypasses hard physical gates.
Use narrow route doubles only to prove returned wall/diagnostic objects remain
unchanged and failure-evidence helpers do not recurse through public dispatch.
Run/check/restart the real CLI, verify species budget registry and model
metadata, compare restarted results and reject cross-model checkpoints before
state decode. Serialize unavailable phi with availability/reason metadata,
without a null/object array or a spurious finite-charge PASS.
```

Covered top-level functions and fixture classes:

- `_massless_budgets`
- `_parameters`
- `_context`
- `_wall_extraction`
- `_compatible_magnetic_field`
- `_state`
- `_cumulative`
- `_copy_state`
- `test_nonzero_compatible_current_fixture_and_untrusted_compatibility_momentum`
- `test_actual_cpu_midpoint_and_rk4_commit_once`
- `test_cli_integrator_default_and_explicit_midpoint`
- `test_guard_error_preserves_commit_and_warn_publishes_report`
- `test_massless_source_hook_is_rejected_without_commit`
- `_toml_value`
- `_write_case`
- `_latest_checkpoint`
- `_rewrite_npz`
- `test_massless_cli_uses_configured_species_ids_in_budget_registry`
- `test_rejected_massless_combine_retains_raw_candidate_without_commit`
- `test_failure_evidence_uses_common_helper_without_public_dispatch_recursion`
- `test_massless_diagnostics_preserve_returned_ledgers`
- `test_massless_stage_result_and_wall_correction_are_returned_unchanged`
- `test_finite_guard_error_persists_full_report_without_commit`
- `test_cli_massless_init_run_restarts_guard_report_and_predecode_model_rejection`
- `test_snapshot_serializes_unavailable_phi_without_null_array`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.

SI reference contract: current and charge references are already densities;
use Jref for wall/current definitions, Jref/L for current divergence, rhoref
for quasineutrality, Eref/L**2 for SCdagCE, and P*u*L**2 for advective
power. Interval charge is rhoref*L**3 and interval momentum is mi*ni*u*L**3.
Relative tolerances and hard/soft flags are independent of these units.
