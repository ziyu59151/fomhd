# `warp_t1_verify.py` pseudocode

Source proposal: `tools/warp_t1_verify.py`.

## A1-PY-T1-VERIFY-01 - Scope and controls

```text
Use the real production periodic Hall fixture and the real selected backend.
This is bounded A1-T1 evidence, not a restart format, scientific acceptance,
closed-wall validation, or performance claim.

Reuse tools.warp_t0_controlled for:
  HallCase(points=9) and build_periodic_context(case, device)
  schedule(coupling="predictor") to obtain the exact H/h/horizon
  build_seed and its original target/mixed amplitudes
  project_state, mode specifications and independent cold REF helpers
  state, totals, cumulative-ledger and JSON/NPZ serialization helpers
  check streaming, accuracy, growth and baseline alignment helpers
Do not copy the T0 runner and do not modify T0 semantics.

CLI:
  --output fresh-directory
  --device {cpu,cuda:0}
  --M positive integer
  --micro-per-period {2048,4096}
  --cycles default 2
  --seed {target,mixed}
  --rtol default 1e-10, stored as waveform_rtol
  --max-iterations default 12, stored as waveform_max_iterations
  --points default 9
  --t0-baseline optional direct T0 case directory

If --t0-baseline is omitted, derive:
  /home/lxzeng/runs/fomhd/a1-t0-controlled__20261106T163530Z/
  matrix-final/cases/t0-h<micro_per_period>-M<M>-<seed>
or use FOMHD_T0_BASELINE_ROOT when set.

Validate every control, the baseline identity, clocks and hashes before
creating output. Refuse an existing output without mutation.
```

## A1-PY-T1-VERIFY-02 - Independent 2x2 analysis helpers

```text
Use the selected linear model from A1_T0_LINEAR_GAIN:
  u' = i*a*b
  b' = i*a*u + i*beta*b
where u is the recorded ion amplitude and b is the fast field amplitude.

exact_t0_map(a,beta,H):
  z = i*beta*H
  f_j = H^(j+1)*phi_(j+1)(z), j=0,1,2
  return
    [1-a^2*f1,             i*a*(f0-a^2*f2)]
    [i*a*f0,               exp(z)-a^2*f1    ]
Implement phi with a short Taylor series for small z and the exponential
difference otherwise, avoiding direct cancellation near z=0.
The tests cross-check this against the augmented constant-coefficient
exponential:
  (u_tilde)' = i*a*b_tilde
  (b_tilde)' = i*a*u_p + i*beta*b_tilde
  u_p' = d
  d' = 0
  initial (u0,b0,u0,i*a*b0)

linear_waveform_sweeps(a,beta,H,M,seed,max_iterations,*,rtol=1e-10):
  Build the iteration-zero tangent on every one of 4M ion stages.
  For each sweep:
    reset Z to accepted Z0 and q to accepted P0
    for each microstep:
      evaluate four complete (F_u,F_b) RHS stages using the old ion path
      keep the two midpoint stages distinct, even at equal clock time
      update causal Z RK4 stages
      reconstruct Pnew=(q,q+hF1/2,q+hF2/2,q+hF3)
      q=q+h*(F1+2F2+2F3+F4)/6
    initialize and record causal Z stage observations as diagnostics only
    measure the ion u stage path and ion u endpoint relative updates
    stop only when both selected ion metrics are <= rtol
  Return path, endpoint, ion stopping metric, initialized Z observations and
  sweep records. Z observations never enter convergence.

The first T0 sweep is not the exact T0 map because production Z is discretely
RK4. The converged linear waveform fixed point is the independent synchronous
fine-h RK4 path. No residual or exact-continuum claim is mixed with these.
```

## A1-PY-T1-VERIFY-03 - Baseline and input identity

```text
Load a preserved T0 case only if:
  case.json and history.npz exist
  points=9, K=0, Nphi=1, kdi=1
  seed, micro-per-period, M and cycles match
  T0 H, h, horizon and macro count match
  saved clocks equal expected_times[:len(saved_times)]
  the saved accepted count equals max(0,len(saved_times)-1)
  saved target/high modal histories and accepted state histories exist
Record:
  baseline absolute path
  case.json SHA-256 and history.npz SHA-256
  combined baseline hash
  baseline Git/source identity fields retained by T0
  original-prefix/complete-window flags and clock-alignment tolerances

A valid rejected prefix is usable. This includes an initialization-only
prefix with one clock sample and no accepted-state arrays. T1 may advance past
the T0 rejection. Compare only exactly matched common endpoints; label every
future T1 endpoint "T0 unavailable". Never fabricate a T0 continuation and
never reject the T1 run merely because its horizon is longer.

Load the actual synchronous fine-h reference from the sibling:
  sync-h<micro_per_period>-<seed>
Use t0._load_matched_reference first. The preserved sync file predates T1
source integration, so if and only if that helper rejects source hashes/commit,
independently revalidate coupling, seed, horizon, physical/discrete identity,
seed normalization and zero-through-horizon clocks before accepting it.
Record every differing source file and both source commits explicitly.

Use t0.align_reference_history to compare T1 target and high four-vectors with
the synchronous reference on exactly the same accepted clocks. Compute every
relative history row as ||actual-reference||_2/||reference||_2, never as a
maximum-component ratio.

For target-only runs, the high harmonic is numerical noise. Do not compute
amplitude or relative ratios against T0-predictor or synchronous noise at any
nesting level. Retain absolute vector/norm differences and explicitly mark all
high ratios undefined.

Archived T0 initial State fields retain one leading singleton time axis:
  initial_ion_n             (1,9)
  initial_ion_momentum      (1,27)
  initial_B                 (1,27)
while T1's diagnostic capture is unbatched as (9,) or (27,).
For INITIAL fields only, remove exactly one leading axis iff it has length one
and the remaining shape exactly equals the T1 field. Do not squeeze arbitrary
axes or permit any other layout mismatch.

Before calling time.advance, create a fresh output directory and write
input.json containing:
  validated controls and schedule
  exact T0 predictor path/hash and original budget snapshot
  synchronous case/history path/hashes, identity and compatibility record
  source Git commit/branch/dirty state
  per-file source/fixture/T0/T1 hashes
  src/fomhd/time_waveform.py whenever that production file exists
  expected memory caps, backend and precision
Also write provenance.json with the same source identity and platform data.
No advancement occurs before both files exist.

Direct execution as `python tools/warp_t1_verify.py` adds the candidate root,
selected source root and source import paths before importing
tools.warp_t0_controlled. `FOMHD_SOURCE_ROOT` is the explicit override.
```

## A1-PY-T1-VERIFY-04 - Production initialization and advancement

```text
Build the original HallCase(points=9), K=0, Nphi=1 fixture on the selected
backend. Set exactly the original 4 GiB numeric memory caps.

Set:
  p.coupling = "waveform"
  p.micro_steps = M
  p.waveform_rtol = CLI rtol
  p.waveform_max_iterations = CLI max iterations
  p.integrator = "rk4"
  p.dt = H
  p.nsteps = macro count
  p.dt_guard = "error"

Build the original target/mixed seed through build_seed. Compare its complete
seed identity and amplitudes with the T0 baseline; do not normalize or scale
it. Initialize the real production state and cumulative ledger. Verify:
  actual backend equals the requested backend
  budgets/hard flags are byte-for-byte equal after setup
  schedule/controls and seed identity equal the baseline

Seed construction and selected-backend ownership are separate:
  built = t0.build_seed(points,seed)       # CPU/NumPy construction/default
  built["context"] = selected_context
  if selected device is CPU:
    use built["state"] unchanged
  else:
    copy the host State identity and upload exactly these nine owning arrays
    through selected_context.compute.array(..., dtype=complex128):
      fluid[0].n, fluid[0].mom, fluid[0].perp, fluid[0].parallel
      fluid[1].n, fluid[1].mom, fluid[1].perp, fluid[1].parallel
      B
    retain step/generation/time and every seed coefficient, amplitude,
    eigenvalue and normalization unchanged

Never pass a CUDA/CuPy context into t0.build_seed or t0._add_harmonic.
Record the CPU construction and explicit one-time upload boundary in the case
summary. This is declared initial-condition setup, not a runtime CPU fallback.
Upload failure is a SeedUpload failure and never silently switches backend.

Advance one macro at a time with source=None. Keep the initial endpoint and
every accepted endpoint. After each accepted macro retain modal vectors, raw
state arrays and totals. The only state host transfers are these explicitly
requested accepted diagnostics.

Never catch and continue after a failure. Do not reduce the macro step, loosen
a tolerance, normalize a seed, or fall back to another backend/coupling.
```

## A1-PY-T1-VERIFY-05 - Streaming evidence and failure retention

```text
Create checks.jsonl.gz and iterations.jsonl before advancement.

Stream initialization checks immediately. For each accepted macro stream all
returned checks and append one full scalar JSON record to iterations.jsonl.
If advance raises, append one failed-macro record carrying the failure's whole
coupling_metadata, including all iteration/sweep residual records, then stop.
Do not depend on a particular residual key layout.

At the end write:
  accepted-history-raw.npz
    immediately after raw initial/accepted modal, State and totals histories,
    cumulative ledger, seed arrays and failure partial arrays are assembled
    before independent REF, T0-predictor or synchronous comparison/postprocess
    so a later exporter bug cannot discard the completed accepted history
  history.npz
    initial and every accepted modal/state history
    every accepted totals history
    independent cold REF/error histories
    isolated synchronous fine-h target/high comparisons on matching clocks
    separately named T0-predictor prefix modal/state/totals comparisons
    explicit matched/future-unavailable coverage metadata
    cumulative accepted ledger and state identity
    original seed arrays and scalar seed identity
  case.json
    accepted/expected steps, horizon and wall time
    maximum/final high seeded norm ratio
    target maximum relative vector error, phase and amplitude drift
    synchronous target/high Euclidean vector errors
    T0-predictor target/high prefix errors under a distinct key
    undefined target-only high ratios; absolute noise norms retained
    unchanged hard-check summary and original budgets
    iteration count/range, actual model-call metadata and memory estimates
    backend, T0 comparison, provenance and explicit limitations

On any failure also write:
  failure.json
    full traceback
    failure code/stage and every returned check
    scheduled rejected time
    accepted prefix and no invented rejected endpoint
    complete failure coupling metadata
    partial-array names
  failure.npz when arrays exist
    failure.partial_arrays
    accepted endpoint/modal/totals prefix
    cumulative ledger snapshot

A nonconvergence is an undetermined failed case, not a completed result.
Never claim a production numeric result unless an actual CLI execution
produced the artifacts.
```

## A1-PY-T1-VERIFY-06 - Validation

```text
Focused tests use synthetic production records and do not require the
separately implemented T1 core:
  invalid controls and existing output do not mutate/create output
  exact T0 map matches the augmented exponential
  converged linear waveform matches independent fine RK4
  two midpoint stage states are retained distinctly
  zero seeded b remains finite and converges
  repeated identical linear sweeps are byte-identical and ion-criterion-only
  input/provenance exist before the first advance
  fake CUDA seed construction never receives the CUDA context
  all nine uploaded state fields are selected-backend complex128 arrays
  CPU seed state and coefficients remain identical with zero uploads
  actual archived T0 initial (1,9)/(1,27) fields compare to unbatched T1
  different initial layouts still fail strictly
  injected postprocess failure leaves accepted-history-raw.npz intact
  unknown nested residual metadata is preserved in iterations.jsonl
  success writes checks/history/case without failure artifacts
  failure retains traceback, stage/checks, partial arrays, accepted/cumulative
    snapshots, failure metadata and scheduled rejected time
  baseline mismatches are rejected before output
  synchronous case/history hashes and identity are in input/provenance
  synchronous growth and T0-predictor growth remain separately named
  Euclidean vector error and target-only no-noise-ratio regressions pass

Acceptance commands:
  new focused pytest file
  tools import check
  CLI --help

If fomhd.time does not yet route coupling="waveform", run tests/help only and
record the production smoke as explicitly blocked by that missing integration.
```
