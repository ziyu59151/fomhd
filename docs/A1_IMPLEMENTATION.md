# A1 implementation contracts

Status: active, bounded T0 and T1 implementations. The design repository owns
the selected mathematics in
`mainline_design/numerics/algorithm_design/A1/A1_ALGORITHM_DESIGN.md`.
Its `TIME_DECOUPLING_DESIGN.md` is the author's format-only converted reference;
historical defaults in that body do not replace massless quasineutral CGL-B
electrons, RK4, FP64/complex128, or the selected time contracts.

A1 inherits A0 source and mirrored pseudocode. There is one solver package;
`release/a0` at `9a38bf41e7bc1fc50186c81967312a7ec5f028e8` owns the separate
maintenance baseline. A0's native implementation remains on its own
`reference/a0-native` branch. No new dependencies are required for T0.

## Selected time controls

```toml
[time]
coupling = "predictor" # Experimental T0; omitted defaults to "synchronous".
micro_steps = 4        # Positive integer M; synchronous requires 1.
integrator = "rk4"     # Predictor requires RK4 and electron_inertia=0.
```

Existing `dt` is the macro interval H and `nsteps` counts accepted macro
intervals. Micro h=H/M. Other required time keys keep their existing meaning.
The spectral guard checks micro RK4 over the remaining physical horizon; it
does not certify the macro predictor's coupling error or stability.

Independent ion density, momentum density and scalar pressure follow a fixed
start-RHS tangent path during the window. Electron perpendicular/parallel
pressures and B advance by RK4 microsteps on that path. All four stages of
every microstep contribute the complete ion RHS, not just electromagnetic
force, to the accepted ion endpoint. The initial tangent is not an additional
impulse. Endpoint electron compatibility and field cache are reconstructed
after the ion correction.

Macro/micro/iteration/slab identities and the two distinct RK4 midpoint stages
remain explicit. Shared fluxes, currents, boundary power and cumulative
ledgers consume the same physical-second quadrature weights. A window commits
once after checks and synchronization; failures leave accepted state and
cumulative accounts unchanged. Only full accepted endpoints can restart.
Restart records distinguish the effective coupling and microstep count.

## Cost and implementation boundary

Total work, peak storage, dependency span and wall time are distinct measures.
T0 retains quadrature stage history for existing interval diagnostics, so
storage grows linearly with M. The conservative preflight includes this
history. T0 still evaluates the full ion RHS at every microstage, so it does
not establish an asymptotic work reduction or an overall fourth-order method.

The fixed electric-wall correction reuses geometry/mode/boundary factors.
It applies C followed by Cdag rather than materializing a dense Cdag@C per
stage. Density-dependent projections still rebuild using the actual stage
density. Local operators and shared setup remain largely dense; full sparse
conversion, streaming history, spatial decomposition and multi-GPU execution
are separate implementation units.

T1 implements complete ion stage-path waveform iteration. T2 initial-history
methods, T3 time slabs and T4 JVP correction still require separate reports and
the user's decision at each layer.
Installed NCCL/CuPy support is not evidence that multi-GPU solver execution
has been verified.

At a fixed total of N microsteps, actual complete model evaluations are
6N+N/M for T0 and 7N for synchronous RK4, including midpoint alias probes.
Both remain O(N); assembly and diagnostics are additional work. Failure
evidence uses cached diagnostic/tag buckets while preserving full record
equality and order, avoiding quadratic scans of complete microstage history. Success
history storage includes each stage's density-dependent wall matrices and
factors, not only independent state vectors.

The pressure-adjoint roundoff scale includes the absolute pressure-rate
operands before CGL cancellation, velocity-pressure-force products, and
pressure-gradient products. Its residual, configured physical budgets, hard
status and 4096*epsilon factor are unchanged. This scale applies only to the
adjoint check and its pressure-pairing duplicate. Independent force/E
comparisons and a corrupted-work hard-gate control remain separate checks.

The experimental CLI `tools/warp_t0_verify.py` compares identical microstep
sizes and physical horizons. It retains per-field SI errors, dimensionless
P/Z errors, phase, actual stage counts and failed arrays. Its timed trajectory
includes check aggregation and modal history projection. RSS/HWM and CuPy pool
statistics are process-wide observations, not isolated per-case live peaks.

Measured scope and retained failures are recorded in
[A1_T0_EVIDENCE.md](A1_T0_EVIDENCE.md). The user authorized T1 on 7 November 2026.

## Controlled T0 temporal baseline

`tools/warp_t0_controlled.py` implements the separate audit, trajectory and
prescribed-ion micro controls specified in design decision D064 and
`A1/A1_T0_BENCHMARK.md`. Its independent cold Hall reference, production
modal Jacobian and nonlinear accepted trajectories have separate identities.
The parallel periodic fixture isolates time integration; it does not establish
general fast-magnetosonic/CAW CFL, closed-wall or production-scale accuracy.

The fixed comparison uses nine periodic samples, spatial target harmonic 1,
high harmonic 4, toroidal K=0, and a two-target-period observation window.
The common microstep is T/2048, subject to the actual full/frozen spectra and
the unchanged strict frequency guard. Main M=4,8,16,32,64,128,256 cases use
both target-only and mixed seeds. Synchronous T/2048 and T/4096 references
and fixed-H T/16 and T/8 refinements preserve the same state and horizon.
A separate T/2 frozen-ion electron/field control consumes production stage
RHS values without creating a coupled energy ledger.

Target vector error, amplitude drift and unwrapped phase are reported
separately against 1%, 1% and 0.1 rad criteria. Seeded high-mode gains include
maximum and final values against the initial state and synchronous reference.
Target-only high modes use absolute values because their initial norm is
numerical zero. A strict rejection retains the accepted prefix, cumulative
ledger, rejected candidate and checks; requested-window stability remains
undetermined. The complete linear macro map and seed transients provide
local linear evidence, not nonlinear acceptance.

This runner is a CPU scientific baseline. All original numerical checks and
budgets remain active; only the retained-history resource cap is set to 4 GiB.
Elapsed times from concurrent scientific cases are not a speedup benchmark.
T1 must reuse the recorded inputs, horizon and criteria to distinguish better
accuracy at the same H from expansion of the usable H range.


## T1 complete stage-path waveform iteration

The selected discrete equations and acceptance matrix are in the design
repository's `A1/A1_T1_WAVEFORM.md`; the independent T0 gain derivation is
`A1/A1_T0_LINEAR_GAIN.md`. Executable pseudocode precedes implementation in
`docs/pseudocode/src/fomhd/time_waveform.py.md`.

```toml
[time]
coupling = "waveform"
micro_steps = 32
integrator = "rk4"
waveform_rtol = 1e-10
waveform_max_iterations = 12
```

Each sweep resets electron/field state to the accepted left endpoint, scans
causal RK4 microsteps along the current ion guess, and reconstructs every ion
RK4 stage and prefix endpoint from the full stage RHS. The two midpoint stages
are distinct. The converged algebraic equations equal synchronous fine RK4
under the same canonicalization; finite-tolerance results still require an
actual reference comparison. The explicit stage dependency makes this selected
path map causal and nilpotent in exact arithmetic. Large windows can still
produce transient growth, invalid trial states, or exhaust the sweep limit.

Path and macro-endpoint updates for each of the three ion fields are normalized
separately by physical references and actual field magnitudes. Intermediate
energy residuals are observations with original thresholds; final converged
intervals use unchanged hard gates. Failure leaves accepted state/cache and
cumulative accounts unchanged and retains current residuals, calls and checks.
Restart records/validates waveform controls; legacy scheme identities remain
compatible. Two ion stage paths and one full diagnostic sweep are retained;
internal prefix endpoints share r1 storage.

For S completed sweeps of M microsteps, complete stage evaluations are
S*(4M+1)+1, alias probes are 2MS, and total model calls are S*(6M+1)+1.
Failure counts use actual attempted calls and completed quadrature entries.
The single-device backend is inherited from Warp/CuPy; waveform sweeps and
microsteps remain serial in time. Cross-slab time parallelism belongs to T3.
The controlled runner is `tools/warp_t1_verify.py`; CPU walltime in this small
fixture is a diagnostic, not a speedup qualification. T0 raw evidence and
`release/a1-t0` at fc9a19e remain the comparison baseline.
