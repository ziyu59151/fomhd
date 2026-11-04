# `tools/warp_benchmark.py` implementation pseudocode

## A0-PY-BENCHMARK-01 - Contract

```text
CLI pilot:
  require --output, a grid list, K=1, one device, an explicit wall-clock budget
  create the output root exclusively; reject every pre-existing path

CLI scan:
  require --output, one grid, K=1, one device, explicit nsteps, integrators,
  cycles, and an explicit wall-clock budget

Invariant:
  requested backend == TOML backend == Backend.name
  never retry a failed CUDA operation on CPU
```

## A0-PY-BENCHMARK-02 - Case bootstrap

```text
locate the repository-owned fomhd_make_case native generator
run it once in output/work/seed with the documented fixture settings
verify n=1e12, P=1e-9, B=1e-5 from the generated TOML

resolve the actually executed benchmark and plotting CLI files from __file__
archive their exact bytes under provenance/tools
record their SHA-256 hashes together with solver source hashes
re-read archived bytes and require byte-for-byte and hash equality
query git --show-toplevel at the source directory
if the resolved Git root equals the selected source root:
  record commit and status for that checkout
else:
  leave commit/status unavailable; source hashes identify the snapshot
  do not attribute a parent repository's status to the copied source

for each required grid:
  build the rectangular geometry with public build_rectangle
  write R_nodes, Z_nodes, lcfs_R, lcfs_Z
  render a provisional case with an existing seed initial file
  load_case
  construct Context
  state = stage.closed_reference(context, n, P, P, B, velocity=0)
  for every active cell:
    envelope = product of sine-square radial and axial envelopes
    ensure envelope is finite, positive, and bounded by one
  set electron density +1 coefficient to 0.5*1e-6*n*envelope
  set electron density -1 coefficient to its conjugate
  leave ion density, pressures, momentum, and harmonic B unchanged
  encode_state
  publish initial.npz and source_manifest.json
  re-load the final case to prove the input hashes and solver schema match
  load_initial and initialize in the real runtime
  require positivity, finite values, initial Ampere/divB checks, and all
    physical hard gates to pass
```

The native generator is a production bootstrap dependency, not a test
oracle. It prevents a second hand-maintained diagnostic-budget registry.
Tests never substitute the solver's `initialize` or `advance`.

## A0-PY-BENCHMARK-03 - One solver case

```text
start setup timer
case = io.load_case(case_path)
state = io.load_initial(case)
context = Context(case.parameters, case.geometry, case.operators,
                  case.compute, case.parameters.Nphi)
cumulative = zero cumulative ledger
committed = stage.initialize(state, context, cumulative,
                             equilibrium=False, captured)
case.compute.synchronize()
record setup, first-step, and steady timers separately

for each requested step:
  enforce the wall-clock deadline
  start = perf_counter
  result = time.advance(committed, cumulative, context)
  context.compute.synchronize()
  elapsed = perf_counter - start
  collect actual accepted:
    complex electron-density coefficient at selected cell and mode
    E RMS norm and E energy
    ion/electron density and velocity RMS norms
    total energy and particle numbers
    max normalized diagnostic residual and hard-failure count
  keep compact scalar history for every step
  retain full state at initial, final, and a bounded number of snapshots

on Failure:
  record code, message, stage identity, checks, and accepted step
  traverse __cause__/__context__ to find a wrapped BudgetExhausted
  classify the whole run as budget_exhausted and return command exit 124
  retain the production wrapper error and full exception chain
  call Run.failure to preserve native failure evidence, then update the
    nested run status to budget_exhausted
  copy every retained partial array to failed_raw.npz even if nonfinite
  write a last-accepted checkpoint through public state encoding
  write rejection_report.json with the actual last time_step_report
  continue with the next independent case
```

The selected observation cell is the active cell with maximum smooth
envelope. Residual normalization uses only available `linf`/`weighted_L2`
checks with positive finite thresholds. Threshold-zero or positive-floor
minimum checks are reported separately as actual positivity minima. The run
peak is taken over initial and every accepted base/midpoint/station/endpoint/
interval check, never only the final endpoint.

For every accepted `time.advance`, copy the actual
`result.time_step_report` to `time_step_reports.json` and to the public
`Run.time_estimate` record when that API exists. Keep the cold scalar
preassessment in `benchmark.json` and `result.json` as separate metadata.

The frequency analysis uses only actual accepted samples. It requires finite
matched times, uniform cadence, at least 16 samples, at least two measured
cycles, at least eight samples per measured cycle, concentrated spectral
power, repeated detrended oscillations, and half-record peak agreement.
After the spectral estimate it fits both Fourier components:

```text
y(t) = a_positive * exp(+i*omega*t)
     + a_negative * exp(-i*omega*t)
```

after linear complex detrending. A dominant component reports its propagation
sign. A real standing wave fits both components and is reported with
`propagation_sign=undetermined` and no signed frequency. Monotonic growth and
aperiodic failure tails are `UNAVAILABLE`, never inferred as a normal-mode
frequency. Raw samples and FFT spectra remain in `observations.npz`. The
analytic plasma frequency is a cold approximate reference.

## A0-PY-BENCHMARK-04 - Pilot result

```text
for each grid:
  run a fresh setup, initial transaction, first actual step, then steady steps
  report:
    setup seconds
    initialization seconds
    first-step seconds
    steady median and half interdecile spread
    core.estimated_peak_bytes
    actual free/total device memory when the selected backend is CUDA

CUDA provenance:
  parse the selected ordinal from the production Backend.device identity
  never read device_ordinal from Parameters and never assume ordinal zero
  enter CuPy Device(ordinal) for the whole provenance probe
  query CUDA properties, memory, and current stream inside that context
  leave the context so the caller's prior ambient device is restored
  record requested backend, actual backend, ordinal, free/total memory
```

## A0-PY-BENCHMARK-05 - Scan plan

```text
omega_plasma = hypot(omega_pe, omega_pi) from stability.frequency_scales
T = cycles * 2*pi/omega_plasma
for every integrator and nsteps:
  dt = T/nsteps
  assert |dt*nsteps - T| <= 32*ulp(T)
  set time.dt_guard = warn
  record the approximated method/horizon assessment
  use the exact same geometry and initial.npz hash for every case
```

The scan does not treat `warn` as a bypass. Every physical, positivity,
linear-algebra, and numerical failure remains fatal for that case.

## A0-PY-BENCHMARK-06 - Machine-readable products

```text
benchmark.json       command, model, method, time, grid, dependencies, source
                     hashes, device properties/current memory, input hashes
summary.json         target/logical results and provenance summary
cases.csv            compact case status and timing/residual table
observations.npz     every actual signal observation, with NaN preserved
input_metadata.json  controlled numerical initial-condition definitions
cases/<id>/case.toml
cases/<id>/source_manifest.json
cases/<id>/result.json
cases/<id>/history.csv
cases/<id>/events.csv
cases/<id>/initial_checks.json
cases/<id>/final_checks.json
cases/<id>/states.npz
cases/<id>/failure.json and failed_raw.npz when applicable
cases/<id>/diagnostic_events.json
cases/<id>/time_step_reports.json
cases/<id>/rejection_report.json when applicable
cases/<id>/run/       native solver Run provenance, final checkpoint and
                     native failure evidence
```

All JSON serialization uses typed NaN/Infinity records. JSON never emits a
bare nonfinite token.
