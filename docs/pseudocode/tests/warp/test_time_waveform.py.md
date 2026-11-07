# A1 T1 waveform scheduler tests

Source: `tests/warp/test_time_waveform.py`.

## A1-PY-TEST-WAVEFORM-01 - Independent linear fixed point

```text
coupled six-field linear F(P,Z),G(P,Z):
  initialize every stage from P0+tau*F0 and Z0+tau*G0
  preserve distinct same-clock r2/r3 stages and unique iteration identities
  rebuild each microstep P stage from its q_j and F1..F4
  advance Z causally with all four G stages
  compare only n_i,mom_i,perp_i at every stage and endpoint
  do not retain or include Z stage paths in the fixed-point residual
  require decreasing normalized path/endpoint residuals
  require the converged endpoint to match an independently stepped fine RK4
  require the accepted left state arrays to remain unchanged
```

This is an algorithmic ODE check only; it is not a production physics
validation or a GPU result.

## A1-PY-TEST-WAVEFORM-02 - Transaction, metadata and budgets

```text
every sweep calls check_step for interval evidence
  energy interval budgets are softened only in the trial copy
  positivity, constraints, non-energy checks, and library failures remain hard
  final converged interval uses the unchanged configured hard budgets
  result metadata reports sweeps, convergence, actual model calls, complete
    quadrature weights, energy-budget scalars, controls, and peak memory
  evaluated r2/r3 stages use alias=True; r1/r4 use alias=False
  metadata alias count equals 2*M*sweeps and actual model-call delta includes it

nonconvergence raises WaveformConvergence and preserves iteration metrics
residual metrics are appended before endpoint/interval checks can fail
endpoint, interval, final-endpoint, pending-ledger, and sync failures preserve
  current-sweep residuals, current counters and attempted stage identity
final hard interval checks replace softened trial checks before pending/sync
initial base time_step_estimate remains in final failure checks exactly once
failure quadrature count is current entries; scheduled 4*M is reported separately
assembly failure does not count an evaluate_stage call that was never made
all failures leave accepted state/cache and cumulative ledgers unchanged
the final commit synchronizes once and performs no post-sync state arithmetic
```

## A1-PY-TEST-WAVEFORM-03 - Configuration, memory and restart

```text
direct and loader validation reject bool/nonfinite/nonpositive tolerances
  and bool/fractional/nonpositive sweep limits before trial mutation
waveform requires massless RK4; waveform-only keys are rejected by legacy schemes
semantic identity injects effective waveform defaults and differs when changed
memory includes old/new complete ion paths and endpoints plus one retained
  stage sweep; the estimate is independent of sweep limit
checkpoints record and compare waveform tolerance and sweep limit
legacy synchronous/predictor checkpoint key sets remain accepted
```

Passing these tests supports the bounded Python/CPU scheduling contract. It
does not establish GPU parity, macro-coupling stability, production accuracy,
or performance.
