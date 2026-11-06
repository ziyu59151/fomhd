# A1 implementation contracts

Status: active, bounded implementation through T0. The design repository owns
the selected mathematics in
`mainline_design/numerics/algorithm_design/A1/A1_ALGORITHM_DESIGN.md`.
Its `TIME_DECOUPLING_DESIGN.md` is the author's format-only converted reference;
historical defaults in that body do not replace massless quasineutral CGL-B
electrons, RK4, FP64/complex128, or the selected T0 contract.

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

T1 waveform iteration, T2 initial-history methods, T3 time slabs and T4 JVP
correction require separate reports and the user's decision at each layer.
Installed NCCL/CuPy support is not evidence that multi-GPU solver execution
has been verified.

At a fixed total of N microsteps, actual complete model evaluations are
6N+N/M for T0 and 7N for synchronous RK4, including midpoint alias probes.
Both remain O(N); assembly and diagnostics are additional work. Failure
evidence still inherits potentially quadratic Python record merging. Success
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
[A1_T0_EVIDENCE.md](A1_T0_EVIDENCE.md). T1 requires the user's next decision.
