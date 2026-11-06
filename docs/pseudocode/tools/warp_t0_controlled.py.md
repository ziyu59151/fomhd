# `warp_t0_controlled.py` pseudocode

Source proposal: `tools/warp_t0_controlled.py`.

## A1-PY-T0-CONTROLLED-01 - Scope, units, and ownership

```text
Use the existing periodic HallCase(kdi=1, points=9), K=0, Nphi=1,
SI defaults, production stage/evaluate/initialize/advance/diagnostics,
and the existing period/frequency fixture formulas.

Modes:
  m is the spatial Fourier harmonic on the periodic z samples.
  m is not the toroidal harmonic K.
  primary target m=1; mixed control adds m=mmax=(points-1)//2.
  This is fixture temporal evidence, not closed-wall or large-device evidence.

All state arrays are owned by the selected CPU backend in complex128.
tools output plain NumPy float64/complex128 arrays after explicit to_host.
No CUDA setup, fallback, or device transfer is permitted.

Normalized 9-coordinate modal order:
  0 n_i/n0
  1..3 mom_i/(mi*n0*vA)
  4 p_i/p0
  5 p_perp_e/p0
  6 p_parallel_e/p0
  7 Bx/B0
  8 By/B0

For m>0, Bz is zero so axial divB is zero. At m=0, check the zero action
separately. Pressure modes remain in the full spectrum.
```

## A1-PY-T0-CONTROLLED-02 - Actual harmonic and Jacobian audit

```text
Build one actual periodic context and uniform equilibrium.

For m=1..mmax:
  S=DZ from the actual periodic operator.
  compute eigenvalues of S.
  select positive spectral eigenvalue k_m; require |k_m-m*k1| is roundoff
  scale the eigenvalue normalization back to rad/s and save k_m.
  Build the independent 4x4 cold Hall matrix using k_m.
  Save all independent eigenvalues/eigenvectors.
  State explicitly that its transverse block does not cover pressure modes.

For epsilon in (1e-5, 5e-6), m=1..mmax:
  evaluate baseline RHS.
  for each of the 9 coordinates:
    add +epsilon and -epsilon times cos(m*theta) in physical samples.
    evaluate the actual massless production stage.
    project each RHS component on exp(i*m*theta):
      1/N sum(real(rhs)) at m=0
      2/N sum(real(rhs)*cos(m*theta)
              - i*real(rhs)*sin(m*theta)).
    normalize the output by the same coordinate scale.
    centered-differentiate to a complex 9x9 Jacobian.
  compute the actual full Jacobian eigenspectrum.
  compare the [mom-x,mom-y,Bx,By] block with the independent matrix.
  save baseline RHS, both matrices, their difference, and nonlinear residual.

Build the micro 4x4 Jacobian in order (pe_perp,pe_parallel,Bx,By)
with all ion coords fixed. Require it to equal the full Jacobian's Z/Z block
within the measured two-epsilon residual.
```

## A1-PY-T0-CONTROLLED-03 - Independent references and seeds

```text
For each selected m:
  choose the positive cold-Hall whistler branch with maximum positive imag.
  normalize the independent cold4x4 vector as max(abs(v))=1 and rotate its
    largest component to zero phase.
  embed only that cold vector in the transverse9 coordinates
    [mom_x,mom_y,Bx,By]; all other coordinates are exactly zero.
  use the cold eigenvalue for every analytic REF trajectory.
  save exact seed vectors, normalization, and branch frequencies.

Production 9x9 eigenvectors/eigenvalues remain LIN audit evidence only.
They are never the seed, analytic time reference, or a self-oracle.
Construct raw state fields from the embedded cold vector using real cosine
samples.
Target m=1 amplitude is 2e-6.
Mixed seed uses the independent cold m=mmax vector at amplitude 2e-8.
Save the exact real seed arrays before initialization.

The continuous comparison is REF:
  normalized 4-component initial vector * amplitude * exp(lambda_cold*t).
Compare m=1 and mmax references separately. Never add different spatial
harmonics into one modal vector. A mixed-grid joint check, if needed, first
reconstructs physical fields at each grid point.
```

## A1-PY-T0-CONTROLLED-04 - Schedules and cases

```text
target Hall frequency = independent positive m=1 cold whistler imag(lambda).
period T = 2*pi/omega_target.
base microstep h = T/micro_per_period.
horizon = cycles*T.
Default main M set is 4,8,16,32,64,128,256; each remains a separate case.

Predictor:
  H=M*h; require cycles*micro_per_period % M == 0
  p.coupling=predictor, p.micro_steps=M, p.integrator=rk4
Synchronous:
  require M=1; H=h
both:
  p.dt=H, p.nsteps=horizon/H
  p.dt_guard=error; preserve every original budget and hard flag
  set only p.memory_limit=p.device_memory_limit=4 GiB

Run only through fomhd.time.advance(source=None). Never hand-code a case.
At each accepted macro save time, independent state, m=1 and mmax
4-component coefficients, checks, totals, and the committed state.
Progress is printed only at a bounded interval.

A rejected step:
  retain the thrown Failure checks, partial arrays, accepted prefix,
  last accepted state/time, and no final/rejected endpoint.
  Empty initialization histories retain shapes (0,9) and (0,4).
  Initialization failures still write case.json and failure artifacts.
  Save the existing cumulative baseline, integrals, boundary integrals and
  accepted state identity on both success and rejection; never create a
  second ledger or commit a rejected trial.
  status=UNDETERMINED, never pass or inferred endpoint.

Stream every returned check immediately to one gzip JSONL file. Update
failure groups and counts incrementally; do not retain all checks in a Python
list. Retain full failure checks and arrays separately.

For an accepted prefix, still save target REF/error history and high-harmonic
absolute/gain history, explicitly marked partial. Full-window qualification
and stability remain UNDETERMINED after any rejection.

Always retain absolute m=mmax amplitude/norm history, including target-only
runs. A ratio to an initial numerical zero is stored as null/undefined.

Serialize the existing cumulative ledger on success and rejection:
baseline fields, both species' integral arrays, boundary integrals, accepted
identity/counters, and transaction status. Initialization rejection before a
prepared trajectory still writes case/failure artifacts with (0,4) modal
history shapes.
```

## A1-PY-T0-CONTROLLED-05 - Accuracy, growth, and artifacts

```text
At every accepted endpoint compare m=1 4-vector with the exact continuum REF:
  relative vector error <= 1%
  amplitude drift <= 1%
  unwrapped signed phase error <= 0.1 rad
Report the three tests separately and jointly.
Use cumulative phase unwrapping across accepted history; never wrap the
full-horizon phase before judging it.

For m=mmax save modal norm/amplitude relative to initial and, when supplied,
the same-horizon matched synchronous endpoint. Flag >1% amplification.
Separate transient maxima from repeated cumulative growth using accepted
norm history; do not label a local linear prediction as nonlinear growth.
Before advancing, require a complete synchronous reference with the same
physical/discrete parameters, exact seed normalization/vectors/amplitudes,
numerical runtime file hashes (all package sources, fixture, tool and lockfile),
and time coverage from zero through the requested horizon. A commit alone
does not identify dirty numerical source.
Align candidate macro times to the reference's saved micro times. Never
require equal array lengths. Retain maximum/final matched norm ratios and
separate transient and final growth flags; the maximum controls the declared
growth criterion.
Report maximum and final matched ratios separately. A maximum above the
threshold remains a transient flag even when the final ratio returns below it.

Before advancing, validate the matched reference's numerical runtime source
hashes, exact seed vectors/normalization/amplitudes, physical/discrete
parameters, and initial-to-final horizon coverage. Do not compare
documentation or report-file hashes.

Save raw hard/soft checks, original thresholds, J-unit energy checks and
totals, both epsilon Jacobians, full/micro spectra, source hashes, controls,
timing, advance/model-call/storage counts, and accepted history.

For each harmonic and H, save omega*H, macro samples/cycle, and the folded
macro-Nyquist alias. Never recover a physical frequency from a naive unwrapped
macro-endpoint phase.

Fresh output only:
  audit -> audit.json + audit.npz
  case  -> case.json + history.npz
  failure -> failure.json (+ failure.npz when arrays exist)
Fail invalid controls and accidental CUDA before creating output.
```

## A1-PY-T0-CONTROLLED-06 - Linear T0 macro-map

```text
Given P'=A P+B Z and Z'=C P+D Z:
  f0=A P0+B Z0
  P_hat(tau)=P0+tau*f0
  for each micro j:
    form RK4 Z stages at c=(0,1/2,1/2,1)
    evaluate Zdot=C*P_hat(stage)+D*Z_stage
    accumulate P with h*(1,2,2,1)/6 over all 4M stages
    advance Z by RK4
  return the 9x9 linear map.

This mirrors the fixed-tangent predictor algebra exactly. It is LIN evidence,
not a nonlinear trajectory. Check it against a production one-macro central
finite difference for small amplitude; keep both results.

For M=4,8,16,32,64,128,256 at h*=T/2048 and fixed 2T:
  save G for one macro
  save G^n for the full 2T horizon
  report per-macro and horizon spectral radii
  label rho(G)^n as spectral_radius_power_over_horizon, not an amplification bound
  propagate the exact saved independent cold seed through G^n
  save the seed gain history, maximum/final gain, and transient/repeated facts
  rho(G)^n is a spectral-radius power, never an amplification upper bound
Eigenvalues alone are insufficient because G can be nonnormal.
```

## A1-PY-T0-CONTROLLED-07 - RK4 stability facts

```text
Use fomhd.stability.amplification('rk4',i*y).
Measure the critical imaginary-axis point by bisection.
For the full and micro spectra report:
  max imaginary eigenvalue
  critical h
  chosen safety fraction h/hcritical
  old T/64 and T/128 positions
  per-mode RK4 horizon amplification and phase over cycles
Report the existing production generic stage estimate separately.
These are linear polynomial facts, not a nonlinear stability certificate.
```

## A1-PY-T0-CONTROLLED-08 - Frozen-ion micro control

```text
This separate prescribed-ion diagnostic never calls fomhd.time.advance.

For m=mmax=(points-1)//2:
  build the production 4x4 frozen Z Jacobian
    (pe_perp,pe_parallel,Bx,By) with all ion coordinates fixed.
  select its resolved positive fast eigenmode and normalize maxabs1.
  seed only Z at amplitude 2e-8; P is the uniform ion state and is invariant.
  horizon=T/2.
  micro_per_period=2048 ->1024 steps at h*=T/2048.
  micro_per_period=4096 ->2048 steps at h*/2.

The manual diagnostic declares one microstep of retained workspace:
  p.micro_steps=1, p.dt=h, p.nsteps=number of microsteps.
  Obtain absolute stage clocks with core.microstage_time(p,micro,0,rkstage).
  Do not declare the entire diagnostic as a retained predictor window.
For each microstep use the production massless stage at reconstructed states
and the existing predictor RK4 helpers. Consume only Z RHS; never accumulate P.
Keep hard stage diagnostics and dt_guard=error active.
Compare accepted micro Z vectors with:
  the frozen-block matrix exponential,
  the frozen-block RK4 polynomial over the same h/count.

Save clearly that this is a frozen-ion micro experiment, not a coupled T0
trajectory, energy ledger, or full-system stability claim.
```
