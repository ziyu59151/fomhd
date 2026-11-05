# Finite-mass longitudinal time control

## A0-PY-LONG-01: Independent mode and periodic adapter

Use SI constants and positive masses from the finite foundation fixture.
For k parallel to B, compute the analytic high warm two-fluid root from
the 2-by-2 density equation. Set lambda to exactly i*omega; derive the seven
relative density/velocity/pressure components from continuity and CGL.
Never obtain this reference from the production-stage Jacobian.

Accept an explicit point count. It must be an integer odd count in [9, 65];
the default is 9 for compatibility. Construct the test-only periodic spectral
geometry and operators at that count. Replace the massless adapter's dummy
Poisson matrix with G^*G plus its mean gauge; select independent
current-divergence rows while retaining the full rows. Check derivative, CG,
DC, adjoints and mean-gauge identities before initialization. The continuous
frequency and eigenvector do not depend on a first-estimate frequency obtained
from the production stage. Reject nonneutral periodic input without subtracting
its mean charge. The production boundary and runtime modules remain unchanged.

## A0-PY-LONG-02: Equal-horizon transaction and selection

Build a positive uniform state plus a real micro-amplitude eigenmode.
Use the finite-mass initialize/advance transaction without a manufactured
source. Preserve initialization and interval diagnostics. The transaction owns
all acceptance and rejection decisions.

Validate points, cycles, steps, amplitude, integrator and dt_guard before
constructing a runtime or output directory. Cycles are finite in [1, 16];
steps are integers in [4, 16384]. Programmatic run_case defaults retain the
legacy `warn` guard for explicit instability regressions. New CLI controls
default to nine points, 4.00 cycles, RK4 only and `dt_guard=error`; `warn`
remains an explicit choice. The
guard is evaluated at every production step over the remaining horizon.
An `error` result rejects the trial and is never relabeled as a passing warning.

For each accepted endpoint, reduce seven complex mode coefficients on the
selected device and copy only those scalars to host. Retain the complete
accepted history. Compare the endpoint with both the independent continuum
phase and the selected integrator polynomial at the actual accepted time.
Keep rejected endpoints distinct from requested completion. Record points,
cycles, selected integrators and dt_guard explicitly in provenance and report
data, in addition to the normalized configuration object.
Record amplitude and phase, initial polarization, adapter checks, hard-record
availability, dt warnings, and a half-amplitude control. When an attempt is
rejected, retain its structured timestep report if supplied by the production
failure. Warning counts include these rejected attempts; do not invent a
report for a failure that did not supply one.

## A0-PY-LONG-03: Full-history complex mode measurement

Measure the normalized electron-density Fourier coefficient from the complete
accepted history. This coefficient is dimensionless and is produced only by
the device modal projection; the measurement function receives time and the
observed coefficient, not the analytic frequency or eigenvector.

Sort is forbidden: require strictly increasing finite times and finite,
nonzero complex coefficients. Unwrap the observed phase in time and fit

```text
phase(t)    = phase_0 + omega_measured t
log|a(t)|  = log_a0 + gamma_measured t
```

by ordinary least squares, using the time origin for numerical conditioning.
The signed angular frequency is the phase slope in rad/s and the signed growth
or damping rate is the log-amplitude slope in 1/s. A result is usable only
when the complete measured phase span is at least four cycles and

```text
(sample_count - 1) / measured_phase_cycles >= 32.
```

Otherwise return `available=false` with a bounded English reason and the
declared thresholds. Never substitute the known analytic or polynomial
frequency for a missing measurement.

For each linear fit report residual RMS in its fitted units and coherence
`max(0, min(1, 1 - SSE/SST))`; a zero SST with zero residual has coherence one.
Also report the relative RMS residual of the complex reconstruction from the
two fits. Report measured angular frequency (rad/s), ordinary frequency (Hz),
growth/damping rate (1/s), complete cycles and samples per cycle. Separately
report the analytic angular frequency/frequency with zero growth and the RK4
prediction `phase(R(i omega dt))/dt`, `log|R(i omega dt)|/dt`. Record the
selected integrator's prediction too. All comparison quantities retain
explicit units and are not inputs to the observed regression.

## A0-PY-LONG-04: Evidence, source identity and scope

Resolve runtime ownership from the actual imported `fomhd` package directory.
For a package at `fomhd/__init__.py`, use its parent root; for
`src/fomhd/__init__.py`, use the root above `src`. Require matching
`pyproject.toml` and `uv.lock` at that root. Archive the imported package and
declared harness files under their actual relative paths. Never fall back to a
legacy `fomhd/` tree, a working-directory guess, or native source.

Require an explicit CPU or owned-Slurm CUDA backend and a new output directory.
Archive executed Python sources, dependency lock, device and job identity.
Write numeric NPZ histories over the whole accepted interval, complete gzip
JSON diagnostics, summary JSON, and an English scope note. Record the exact
failure if any case is rejected.

The default programmatic scan preserves the legacy two-cycle, midpoint/RK4,
4/8/32/64/128-step and `warn` behavior. CLI selection defaults to the bounded
RK4 control. The small-amplitude endpoint comparison against the selected
integrator polynomial retains the declared relative tolerance of 1e-5 and
does not silently relax it for nonlinear or long-horizon drift. Continuum
error is measured, never required to vanish in coarse cases.

This is a periodic local time control, not production-wall spatial accuracy,
a complete stability region, or a speed benchmark. The chosen parameter scales
do not validate physical CGL ordering.

The stable CLI defaults are RK4, error guard, eight cycles, and 1024/2048 steps. Four-cycle coverage alone is insufficient when numerical phase lags. Explicit coarse controls remain available. Initial guard/prediction failures retain a failed report; actual loaded src/fomhd files are snapshotted.
