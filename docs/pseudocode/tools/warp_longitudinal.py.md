# Finite-mass longitudinal time control

## A0-PY-LONG-01: Independent mode and periodic adapter

Use SI constants and positive masses from the finite foundation fixture.
For k parallel to B, compute the analytic high warm two-fluid root from
the 2-by-2 density equation. Set lambda to exactly i*omega; derive the seven
relative density/velocity/pressure components from continuity and CGL.
Never obtain this reference from the production-stage Jacobian.

Reuse the test-only nine-point spectral geometry and operators. Replace
the massless adapter's dummy Poisson matrix with G^*G plus its mean gauge;
select independent current-divergence rows, retaining the full rows.
Check derivative, CG, DC, adjoints and mean-gauge identities before initialization.
Reject nonneutral periodic input without subtracting its mean charge.
The production boundary and runtime modules remain unchanged.

## A0-PY-LONG-02: Equal-horizon transaction

Build a positive uniform state plus a real micro-amplitude eigenmode.
Use the finite-mass initialize/advance transaction without a manufactured
source. Preserve initialization and interval diagnostics. Only dt_guard is
warn for deliberate time-error controls; retain every physical hard gate.

For each accepted endpoint, reduce seven complex mode coefficients on the
selected device and copy only those scalars to host. Compare the endpoint
with both the independent continuum phase and the integrator polynomial
at the actual accepted time. Keep rejected endpoints distinct from requested
completion. Record amplitude and phase, initial polarization, adapter checks,
hard-record availability, dt warnings, and a half-amplitude control.
When an attempt is rejected, retain its structured timestep report if supplied
by the production failure. Warning counts include these rejected attempts;
do not invent a report for a failure that did not supply one.

## A0-PY-LONG-03: Evidence and scope

Require an explicit CPU or owned-Slurm CUDA backend and a new output directory.
Archive executed Python sources, dependency lock, device and job identity.
Write numeric NPZ histories, complete gzip JSON diagnostics, summary JSON,
and an English scope note. Record the exact failure if any case is rejected.
The default two-cycle scan uses 4, 8, 32, 64 and 128 steps for midpoint/RK4.
The small-amplitude polynomial comparison uses a declared relative tolerance
of 1e-5; continuum error is measured, never required to vanish in coarse cases.
This is a periodic local time control, not production-wall spatial accuracy,
a complete stability region, or a speed benchmark.
The chosen parameter scales do not validate physical CGL ordering.
