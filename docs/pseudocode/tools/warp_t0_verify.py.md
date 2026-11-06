# `warp_t0_verify.py` pseudocode

Source: `tools/warp_t0_verify.py`.

## A1-PY-T0-VERIFY-01 - Scope and controls

```text
The tool is experimental fixture evidence, not a production restart.
Use the real production stage, diagnostics, canonicalization and time
transaction. Do not import or copy the sibling T0 implementation.

Parse device=cpu|cuda:0, a new output path, k*di choices, odd points>=9,
relative amplitude, cycles, micro steps per cycle in {64,128}, and M list.
Validate every control before creating output:
  cycles*micro_steps_per_cycle must be an integer microstep count
  every M must be positive, unique, fit the horizon, and divide the count
  closed H/M/horizon must be positive with integer macro count
CLI time controls are:
  kdi default [1]
  points default 9
  relative amplitude default 2e-6
  cycles default 1
  steps per cycle default 64
  M default [1,2,4,8]
  repetitions default 1
  optional h/2 synchronous reference default enabled
Default execution is periodic-only; --closed-only selects the closed case.
```

## A1-PY-T0-VERIFY-02 - Equal-h periodic schedules

```text
For HallCase(kdi,points):
  omega = positive cold Hall eigenfrequency used by the fixture
  period = 2*pi/omega
  h = period/micro_steps_per_cycle
  T = cycles*period

Run A0-compatible synchronous RK4 with H=h and T/h macro steps.
Optionally run synchronous RK4 with H=h/2 and 2*T/h macro steps.
For each M:
  H=M*h
  macro_count=(T/h)/M, required integer
  Parameters.coupling="predictor"
  Parameters.micro_steps=M
  Parameters.integrator="rk4"
  Parameters.dt=H
  Parameters.nsteps=macro_count

Build each trajectory from initial_hall_eigenmode and commit it through the
actual initialize path. Use source=None. The periodic adapter is labeled as
temporal fixture evidence, not closed-wall physical validation.
```

## A1-PY-T0-VERIFY-03 - Endpoint comparisons

```text
Retain the full final independent vectors
  P=(ion n,momentum,perp)
  Z=(electron perp,parallel,B)
and one four-component modal coefficient per accepted endpoint.

For each run compare the full P+Z endpoint to:
  baseline + real cosine/sine realization of
  relative_amplitude*vector*exp(lambda*T)

Do not multiply real physical samples directly by the complex exponential.
Rebuild both real quadratures from the phase-shifted complex coefficient.

Report these independent fields separately:
  n_i, mom_i, p_i, p_perp_e, p_parallel_e, B
Each field retains:
  SI absolute error
  reference perturbation L2 norm
  physical cold-Hall amplitude floor
  per-field relative error

Use cold-Hall floors:
  n0*a
  mi*n0*vA*a
  pressure*a (p_i,p_perp_e,p_parallel_e)
  B0*a
Use max(reference perturbation norm,floor) as each field's scale. Mark zero
linear components limited by the explicit amplitude floor.

Never form a norm from concatenated raw SI fields. For P, Z and full P+Z,
first divide each field error by its own scale and then report a dimensionless
RMS over the field-relative errors. Keep the projected amplitude ratio and
signed phase in the principal interval [-pi,pi], with modulo-2*pi ambiguity
labels, on the normalized Hall modal vector.

Compare T0 independently to the full synchronous endpoint at h and, when
enabled, h/2 using the same per-field contract. Do not present synchronous
RK4 as a polynomial-equivalent T0
oracle and do not choose accuracy pass thresholds in this tool.
```

## A1-PY-T0-VERIFY-04 - Stage counts, timing and resources

```text
For every predictor macro require actual coupling_metadata fields:
  complete stage evaluations=4*M+1
  full-RHS stage evaluations=4*M
  endpoint evaluations=1
  alias evaluations=2*M
  total model evaluations=6*M+1
  retained quadrature entries=4*M
Hide no mismatch: fail the case if emitted counts differ.
Aggregate counts across accepted macro steps and retain the last metadata
including endpoint_tangent_correction.

Build a separate initialized warmup trajectory and discard it.
For each repetition:
  construct a fresh trajectory before the timing region
  synchronize
  time the accepted advance loop, check aggregation and modal history projection
  synchronize
  record setup and advance times separately
Report the median and samples; do not infer a speedup.

Observe process-wide RSS/HWM, CuPy pool reserved and live-at-end bytes, solver
estimated_peak_bytes, and endpoint tangent correction. These are not isolated
per-case live peaks. Label pool reserved bytes as different from live peak memory.
```

## A1-PY-T0-VERIFY-05 - Bounded closed-wall case

```text
Build the existing small closed-rectangle massless context with
hard_diagnostics=True on the explicitly requested backend and use the existing
equilibrium fixture. Construct geometry and operators with Backend(device);
do not call a CPU-hardcoded fixture helper.
Add only a nonzero smooth boundary-shaped density and pressure perturbation.
Do not change hard flags or budgets.
Set fixed H, M and horizon, run the actual T0 transaction, and recanonicalize
the endpoint through fomhd.massless.canonicalize_state.

Report accepted steps, actual stage metadata, check counts, failed records,
independent P/Z change, and the derived electron fields that canonicalization
actually reconstructs: n_e and compatibility momentum density. Retain before
and after arrays for both; report their SI absolute/scale/relative corrections
and a unit-invariant aggregate. Independent P/Z equality alone is not
reconstruction evidence. Existing endpoint diagnostics own correctness.

Report requested/actual backend identity and failure evidence. Verify
context.compute.device before measurement. A mismatch is an error; there is
no CPU or other backend fallback. An unchanged hard diagnostic failure remains
a failed case; it is not clipped or relaxed.

Use the same `resources` key in closed and periodic successful summaries so
the common artifact writer can serialize both. A successful closed trajectory
must produce its NPZ, metadata JSON and failed-check JSON before report
completion.
```

## A1-PY-T0-VERIFY-06 - Evidence and provenance

```text
Refuse an existing output before writing.
Write provenance with exact invocation argv, UTC, platform/CPU, selected
device, only SLURM_JOB_ID and CUDA_VISIBLE_DEVICES, source Git SHA/dirty flag,
all runtime module hashes, declared harness/fixture/test/mirror hashes,
pyproject/uv.lock and selected library versions.
State that the tool is experimental fixture verification and not a restart.

Per successful case write:
  full endpoint field arrays and P/Z vectors as pickle-free NPZ
  time history and four-mode amplitude history
  compact actual metadata and timing/resources JSON
  check counts and grouped failed records only
Per failed case write:
  case failure JSON with full thrown check evidence
  all available partial arrays as pickle-free NPZ
Keep failed cases in report.json. Exit nonzero for a failed case or an
unexpected harness/solver error; never synthesize a source or run manifest.
```
