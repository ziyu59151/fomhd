# `tests/warp/test_warp_t1_verify.py` pseudocode

Source: `tests/warp/test_warp_t1_verify.py`.

## A1-PY-TEST-T1-VERIFY-01 - Independent linear model

```text
Construct the 4x4 augmented constant-coefficient system for exact_t0_map:
  the first two rows are the T0 frozen-ion-path fast equations
  the last two rows advance the ion tangent and its constant derivative
  columns map (u0,b0) through the augmented initial vector
  compare exp(AH) projection with exact_t0_map at nonzero beta and moderate H

Run linear_waveform_sweeps on a moderate coupled system.
Independently advance the same 2x2 coupled RHS with ordinary synchronous
fine-h RK4 for M microsteps.
Require convergence, exact agreement with that discrete fixed point, a full
(M,4,2) stage path, and unequal values for the two midpoint stages.

Run the same linear case repeatedly. Require identical converged state, sweep
count, ion path, endpoint and initialized Z observations. For every sweep,
require convergence to equal max(ion-stage-path, ion-endpoint) <= rtol; no Z
observation may enter that decision.

Set b0=0 and require finite unchanged zero fast amplitude.
```

## A1-PY-TEST-T1-VERIFY-02 - Controls and baseline identity

```text
Use synthetic T0 case.json/history.npz fixtures with exact H-spaced clocks.
Reject an unsupported micro-per-period before creating output.
Reject an existing output directory with a sentinel file and no mutation.
Reject a saved T0 clock that differs from the exact candidate macro clock.
Require baseline case/history names and record both SHA-256 hashes in input.
Check CLI defaults M=4, h=T/2048, two cycles, rtol=1e-10 and 12 sweeps.
Reject zero tolerance and zero/boolean sweep limits.
Accept a one-clock initialization-only prefix and a multi-step rejected prefix.
Run a longer T1 trajectory over each and verify only common endpoints compare,
while future T0 references are explicitly unavailable and never fabricated.

Require synchronous case/history hashes and identity in input/provenance.
Test the exact t0._load_matched_reference path and the explicit preserved-source
compatibility path, including recorded changed source files.

For the initialization boundary, call _prepare_runtime with a fake CUDA context
and require t0.build_seed to receive no context. Use a fake backend that counts
context.compute.array calls and rejects unuploaded host arrays. Require exactly
the nine owning State fields to become backend arrays while step, generation,
time, seed vectors, coefficients and normalization remain unchanged.

Run the same seed-boundary helper on CPU and require the original NumPy State
object and all values to remain unchanged with zero upload calls.

Load a real archived T0 history and verify its initial fields retain singleton
time axes `(1,9)` and `(1,27)`. Compare unbatched T1 copies with the complete
baseline and require all initial comparisons to pass. Add a strict negative
control where removing a leading singleton would still leave a different shape.
```

## A1-PY-TEST-T1-VERIFY-03 - Production evidence with synthetic records

```text
Do not wait for the separately implemented waveform core.
Replace only the runtime preparation/advance/capture seams with synthetic
production-shaped objects and records.

Before the first advance, assert that input.json and provenance.json exist.
Use coupling_metadata["iterations"] with multiple synthetic sweeps, including
unknown nested fields. Verify that iterations.jsonl preserves the whole
metadata rather than decoding a fixed residual schema.

On success verify:
  case status, accepted/expected counts
  iteration total, model-call sum and memory estimate
  history.npz initial/every-accepted modal/state/totals and T0-aligned arrays
  checks.jsonl.gz exists
  no failure artifacts
  accepted-history-raw.npz matches the accepted modal/state/totals data

Force _synchronous_comparisons to raise after advancement and before final
export. Require accepted-history-raw.npz to remain readable with all 1025
accepted clocks in production or the complete synthetic accepted prefix in the
focused fixture.
  target/high synchronous arrays and matched clocks are retained
  mixed growth uses the synchronous high history under matched_synchronous
  T0-predictor high comparison has its own matched_t0_predictor key
  target-only high ratios and relative errors are null/undefined at every
    nesting level while absolute norms remain present

Test vector-history error with vectors that produce different maximum-component
and Euclidean values, and require the Euclidean definition.

On a synthetic WaveformConvergence failure verify:
  case remains undetermined
  traceback, stage, checks and scheduled rejected time are retained
  no rejected endpoint is invented
  failure.npz includes partial arrays, accepted prefix and cumulative snapshot
  the failed macro has its own iterations.jsonl line and metadata

Add a production time_waveform source hash to a temporary source tree and require
its SHA-256 entry. Invoke the staged script directly with PYTHONPATH removed
from a temporary working directory and require `--help` to succeed.
```

## A1-PY-TEST-T1-VERIFY-04 - Scope

```text
These tests prove the harness/control/serialization contracts and the
independent 2x2 algebra. They do not prove production T1 numerical behavior,
CPU/CUDA parity, residual convergence in the nonlinear fixture, or any
scientific threshold. A production smoke may run only after the parent
integrates fomhd.time waveform dispatch.
```
