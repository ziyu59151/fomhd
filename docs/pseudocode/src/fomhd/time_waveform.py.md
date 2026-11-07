# A1 T1 complete ion stage-path fixed-point iteration

Status: selected implementation contract, 7 November 2026. Algorithm authority:
`A1/TIME_DECOUPLING_DESIGN.md`, sections 5–6, and `A1_T1_WAVEFORM.md`.

## A1-PY-WAVEFORM-01 — Discrete target and reconstruction

Let P contain independent ion density [m^-3], momentum density [kg m^-2 s^-1],
and isotropic pressure [Pa]. Z contains electron perpendicular/parallel pressure
[Pa] and magnetic flux [Wb]. Existing modal/node layouts and complex128 storage
are unchanged. All arrays belong to the transaction on the selected backend;
only residual scalars cross to host. Canonicalization reconstructs massless
compatibility fields at each evaluated stage, never a cached previous stage.

The ion guess stores four distinct states per microstep; the two midpoint
stages remain distinct even though their clocks coincide. Initialize every
stage from P0+tau*F0. Each sweep resets Z to the accepted left endpoint.

```text
for sweep k = 0 .. waveform_max_iterations-1:
  Z = accepted Z0
  q = accepted P0
  for micro j=0..M-1:
    evaluate four complete RHS (F_jr,G_jr) using old ion P_jr^k
      and causal Z RK4 stages, with unique trial/generation/iteration identity
    Z_next = Z + h*(G1+2G2+2G3+G4)/6
    Pnew_j1 = q
    Pnew_j2 = q + h*F1/2
    Pnew_j3 = q + h*F2/2
    Pnew_j4 = q + h*F3
    q = q + h*(F1+2F2+2F3+F4)/6
    retain final-sweep quadrature (state,stage,h*b_r), b=(1,2,2,1)/6
  endpoint_new = q
  measure all-stage ion-path update AND endpoint update against the old P guess
  report n_i,mom_i,perp_i raw max-absolute difference, physical scale and ratio
  physical scales = max(field physical reference, max_abs(old), max_abs(new))
  convergence requires all three ion stage paths and endpoint updates
    <= rtol*scale
  append this iteration and residual before endpoint assembly/evaluation
    or any interval check can fail
  set coupling_converged from the measured three-field residual
  then assemble/evaluate endpoint and run the trial interval check
  add interval-energy scalar records only if the trial check produced them
  record raw interval energy checks read-only for this sweep
  if converged: proceed to final diagnostics/acceptance
  else replace old path and old endpoint; discard old sweep history
```

Default `waveform_rtol=1e-10`, `waveform_max_iterations=12` (sweeps includes
the initial tangent scan). Strictly validate finite positive tolerance and
positive non-boolean integer sweep limit, massless RK4 only. A one-sweep
run is not automatically accepted: it must meet the same residual criterion.

At the fixed point, the three-field P-stage reconstruction and the strictly
causal Z RK4 map are the full fine-RK4 stage equations under the same
canonicalization, constraints and source model. Z is a deterministic causal
function of the current P path and resets to accepted Z0 each sweep; it is not
an iteration residual. The resulting strictly causal P-stage map is nilpotent
and targets fine RK4. This is an algebraic fixed-point claim, not a generic
spectral-divergence or convergence-rate claim. Preserve residual/endpoint
error and compare the actual synchronous fine reference in tests and
benchmarks.

## A1-PY-WAVEFORM-02 — Transaction, failures and evidence

Microclocks use existing integer-index scheduler helpers. Every stage tag
includes macro, micro, iteration, slab and generation. The first stage can
reuse the base RHS if its identity is re-evaluated or explicitly represented;
all model-call counts must describe actual calls. No repeated trial work enters
the cumulative ledger. Intermediate sweep energy defects are measured, not
accepted intervals; ordinary required stage positivity/constraint gates remain.

Preserve the T0 alias schedule: evaluated r2/r3 stages call
`evaluate_stage(...,alias=True)`; r1/r4 and the final endpoint use
`alias=False`. The base reuse at sweep 0, micro 0, r1 is also unaliased. Metadata
reports directed complete-stage calls, alias calls, and the actual source/model
call-count delta separately. Assembly attempts increment before
`_assemble_state`; complete/stage/endpoint evaluation counters increment only
after assembly succeeds and immediately before `evaluate_stage`.

For a trial sweep, call `diagnostics.check_step` with a shallow context copy.
Copy each budget record and set `hard=False` only for the interval-energy keys:
`kinetic_energy`, `ion_internal`, `electron_perp`, `electron_parallel`,
`field_energy_B_interval`, `field_exchange_interval`, and
`total_energy_interval`. Never mutate configured budget objects. The final
converged interval uses the original context and therefore unchanged hard gates.

After convergence, rebuild the final endpoint and cache using the corrected
ion endpoint and last swept Z. Run unchanged final interval diagnostics using
the last sweep's complete stage quadrature. Include the initial base
`time_step_estimate` check exactly once even when convergence occurs after
sweep 0. Assign the final hard-check result as current failure evidence before
pending-ledger construction or synchronization. Gate all necessary checks, form
one pending ledger, synchronize the backend and commit exactly once. Never
append another tangent or correction impulse. Restart contains accepted
endpoints only, selected coupling, microstep count, tolerance and sweep limit;
no partially iterated window can be restored.

Nonconvergence is `WaveformConvergence` with full per-sweep metrics and current
failure evidence; leave accepted state/cache and cumulative ledgers unchanged.
No silent synchronous fallback, step reduction or threshold relaxation. On any
failure rebuild metadata from the current counters, current quadrature and
scalar history; never retain an earlier metadata snapshot. Preserve
`converged` from the latest measured three-field residual and report the current
attempted iteration and stage identity. `quadrature_stage_evaluations` is
`len(current quadrature)`; report scheduled `4*M` separately. Memory
preflight includes two ion stage paths plus endpoints and one sweep's retained
state/RHS/cache history; histories do not multiply by iteration count.

## A1-PY-WAVEFORM-03 — Required verification

Use an independent coupled linear RHS to establish stage reconstruction and
the fine-RK4 fixed point, distinct midpoint values, decreasing path residual,
endpoint correction, and unchanged left endpoint. Inject nonconvergence and
final check/library failures to establish rollback and zero ledger leakage.
Check metadata actual calls, iteration identity, no whole-state host transfer,
the r2/r3 alias schedule and call counts, configuration and restart mismatch
handling, memory accounting and bounded production CPU/CUDA parity. Existing
T0 tests and raw baselines remain intact.

Internal ion endpoint q_j storage aliases the corresponding reconstructed r1
stage; it is not an additional M+1-array history. Only the old/new final macro
endpoints are extra arrays in the memory plan.
