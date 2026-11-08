# Bounded T1 restart and rollback regressions

Source: `scratch/tests/warp/test_050_transactions.py`

Status: final scratch-only acceptance contract for real T1 transactions. The
selected FOMHD worktree is read-only. This contract does not change the solver,
physics budgets, or time-integration algorithm.

## Fixture and backend contract

1. Build the existing small real closed rectangle, massless A0 state through
   `test_time_waveform._waveform_case` and
   `test_model_routing._context/_state/_write_case`.
2. Require massless RK4, waveform coupling, a positive micro-step count, and
   the real T1 controls `waveform_rtol=1e-10` and
   `waveform_max_iterations=12`. Keep the existing hard diagnostics and
   budgets unchanged.
3. Keep the routing fixture's nonzero wall-compatible magnetic perturbation.
   Assert that the selected initial state is not the all-zero state.
4. Parameterize `cpu` and `cuda:0`. CPU always executes. CUDA executes only
   when both `FOMHD_TEST_GPU=1` and `SLURM_JOB_ID` are explicitly present;
   otherwise that parameter case is skipped before constructing a backend.
   Use `_switch_case_to_cuda` and its case-hash update for CUDA. No GPU work is
   launched by the bounded CPU verification run.

## Direct and checkpoint-restart acceptance

1. Create the same nsteps=1 waveform case used by production. For CUDA, convert
   the case and source hash before loading it.
2. Run production CLI nsteps=1 to a fresh output directory, locate its durable
   checkpoint through `_latest_checkpoint`, and require one accepted interval.
3. Rewrite the same fixture root as nsteps=2. Reinsert the T1 controls and
   reset the source-manifest case hash. Reapply the CUDA backend rewrite when
   selected. The semantic hash excludes the allowed horizon extension.
4. Run production CLI nsteps=2 with `--restart checkpoint1` to a second fresh
   output. Read its durable checkpoint only through `io.read_restart`.
   Require `accepted_intervals == 2`, `last_step == 2`, and a time of exactly
   two physical macro steps.
5. Separately load and initialize the same nsteps=2 case with the current
   selected source. Call `fomhd.time.advance` twice. Require each result and
   interval to be accepted and require converged waveform metadata.
6. Compare the direct accepted state to the restart-read accepted state only.
   Compare every magnetic and fluid field with `assert_dimensional` at an
   appropriate physical reference and tolerance. Compare step and time
   exactly.
7. Compare every raw cumulative component: both species ledgers, every
   boundary integral, the one baseline divergence/totals/momentum record,
   accepted count, last step, and last time. Require both last-trial counters
   to be nonzero, but do not compare their values because production restart
   initialization consumes a new trial identity. Use physical references for
   state and cumulative tolerances. Do not use trial arrays as accepted
   evidence.

## Forced inner-stage rollback

1. Initialize a fresh nsteps=2 case and accept exactly one macro through the
   real production scheduler. Snapshot the committed state, cache tag,
   generation, cumulative baseline divergence, and full JSON-safe physical
   ledger after that commit.
2. Wrap `fomhd.time_waveform.evaluate_stage`. Let at least two real stages on
   the second macro complete, then raise
   `Failure("Injected", "Injected T1 inner-stage failure")` with the exact
   scheduled `StageTag` attached.
3. Require the failure to retain model arrays, its stage tag, and waveform
   iteration metadata from the attempted second macro.
4. Assert the committed state object, generation, step, time, physical arrays,
   accepted cache tag, baseline divergence, and every physical-ledger
   component are unchanged. Require one accepted interval and the original
   last-trial id.
5. Require the accepted committed cache to still pass the existing
   `io._check_accepted` generation contract. A proxy whose cache tag uses the
   failed trial generation and a different endpoint identity must be rejected
   by that same contract. This proves the failed cache was not committed.
