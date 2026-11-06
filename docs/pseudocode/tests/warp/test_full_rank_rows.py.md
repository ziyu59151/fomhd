# Full-row-rank independent row tests

Source: `tests/warp/test_full_rank_rows.py`.

## A0-PY-TEST-FULL-RANK-ROWS-01 - Procedure and assertions

```text
Preserve a small old-prefix oracle that independently repeats the sequential
normalize-and-SVD selection rule.

For real and complex full-row-rank matrices, compare the selected original row
indices exactly with the oracle and hand-checked lists. With a CPU Backend
subclass that counts singular_values calls, require exactly one SVD on the
proved-full-rank path.

For rank-deficient, zero-row, near-threshold-above/below, and tall m>n
matrices, compare exact indices with the oracle. Require the rank-deficient
fallback to use more than the one probe SVD while preserving the old selection.
Require all-zero input to return no indices without calling SVD.

Reject malformed arrays, nonfinite values, and invalid rank tolerances with the
existing Failure codes. Inject a singular-value failure and require it to
propagate through the full-rank gate rather than being converted into a
fallback decision.
```

Covered top-level functions and fixture classes:

- `_old_prefix_rows`
- `_CountingBackend`
- `_FailingSvdBackend`
- `test_full_rank_real_rows_use_one_svd`
- `test_complex_full_rank_rows_match_old_prefix_oracle`
- `test_rank_deficient_rows_fall_back_and_match_old_prefix_oracle`
- `test_zero_rows_are_skipped_in_original_order`
- `test_all_zero_rows_do_not_run_svd`
- `test_near_threshold_rows_match_old_prefix_oracle`
- `test_tall_matrix_matches_old_prefix_oracle`
- `test_invalid_inputs_preserve_failure_codes`
- `test_singular_value_failure_propagates`

This mirror checks CPU backend selection behavior and mathematical equivalence.
It does not claim CUDA execution or full rank-gate scan performance; those
require independent CUDA and larger-grid evidence.
