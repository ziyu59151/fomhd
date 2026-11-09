# Matched-horizon WR cost tests

Source: `tests/warp/test_warp_wr_cost.py`.

```text
real CPU fixture:
  use the promoted source and tools.warp_wr_periodic
  33 points, amplitude .02, nonuniform .1, mode1/mode16 sentinel
  horizon=8h, M=8, s=3
  run synchronous, waveform and multirate_wr through real advance

contract:
  require identical nontrivial initial accepted state digest across methods
  require identical immutable cumulative digest
  require hard quasineutrality and optional observe theta-consistency
  require equal physical horizon
  synchronous has 8 advance calls; waveform/WR have one per horizon
  warmup and all three repeats are accepted real horizons
  warmup and every repeat use the same input state/cumulative digests
  pre-run generation counters match across every repeat
  setup/reset/run times and resource peaks are present
  raw role-separated counters are present and never summed
  verify ion_transport_producer_calls == spy.full_ion_calls
  verify stage_evaluator_calls == spy.full_stage_calls
  verify full_rhs_finish_calls == spy.full_rhs_calls
  reject obsolete full_ion_complete_* labels
  no speedup or scientific-acceptance claim exists

CUDA guards:
  reject CUDA without FOMHD_TEST_GPU=1
  reject CUDA without SLURM_JOB_ID before context construction
  reject unsupported horizon, non-divisor wr_M and wall>900
  refuse an existing output path without changing it

failure:
  injected raw advance Failure is retained with no timing summary,
  no speedup claim, and one attempted real warmup
```
