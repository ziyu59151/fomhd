# A1 implementation contracts

D072 selects a few-node multirate waveform method on the 0.5.0 branch.
T2-E, C1/C2 and unimplemented T tiers are retained research, not prerequisites.
The new target is documented in design authority `A1/A1_WR_050.md`; few ion
nodes remain after convergence. The inherited A0/T0/T1 contracts below retain
their original meaning. A new runtime capability needs its own complete
interval-work, rollback, restart and CPU/GPU acceptance evidence.

Status: active few-node WR implementation on the experiment branch, with separate
correctness, physical-range and cost verification. T0/T1 and bounded fixed-ion
T2-E1/E2/E3 remain implemented references. The design repository owns
the selected mathematics in
`mainline_design/numerics/algorithm_design/A1/A1_ALGORITHM_DESIGN.md`.
Its `TIME_DECOUPLING_DESIGN.md` is an evolving design reference, revised under
D066; the original format-only conversion remains archived. Current defaults
are massless quasineutral CGL-B electrons, RK4 and FP64/complex128.
The [roadmap](ROADMAP.md) distinguishes A3 hybrid 1.0 from A4 all-ion kinetics.
T2-E is an experimental field-window API and benchmark; it does not add a
complete coupled runtime scheme. See [T2-E verification](A1_T2_E_VERIFICATION.md).
D067 further distinguishes T1 stage values from continuous path representations,
uses explicit endpoint projections for path maps/residuals, and scopes complete
4M ion RHS quadrature to the existing fluid T0/T1 references.

A1 inherits A0 source and mirrored pseudocode. There is one solver package;
`release/a0` at `9a38bf41e7bc1fc50186c81967312a7ec5f028e8` owns the separate
maintenance baseline. A0's native implementation remains on its own
`reference/a0-native` branch. No new dependencies are required for T0.

## Active few-node time controls

```toml
[time]
coupling = "multirate_wr"
micro_steps = 8
ion_nodes = 3
wr_initial = "tangent"
wr_relaxation = 1.0
waveform_rtol = 1e-10
waveform_max_iterations = 12
integrator = "rk4"
```

`dt` is H, `h=H/M`, and `nsteps` counts accepted macros. Ion density,
momentum and pressure are solved directly at three or five Lobatto nodes.
The real field RK4 stages consume the polynomial from one immutable ion-node
version. Coarse transport and fine continuity/electromagnetic response use
their declared integration weights in the same complete transaction. This
changed discrete target is specified in the design authority
`A1/A1_WR_050.md`; it does not recover all T1 ion microstages.

See [the generic window](pseudocode/src/fomhd/time_multirate_wr.py.md),
[the physical split](pseudocode/src/fomhd/multirate_wr_physics.py.md),
[the mixed ledger](pseudocode/src/fomhd/multirate_wr_checks.py.md) and
[the complete transaction](pseudocode/src/fomhd/time_multirate.py.md).
The stage count is a work metric, separate from actual GPU acceleration and
future particle operations. The present field scan is causal.

## Retained T0 time controls

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

T1 implements complete ion stage-path waveform iteration. Its exact fixed point
is fine synchronous RK4; it is a serial coupling reference and still computes
all microstage ion RHS values on every sweep. Synchronous fluid work is already
acceptable. A1 uses that controllable platform to design transferable methods
for future expensive FK work, not to infer success from fluid convergence or
fewer sweeps alone.

The retained D068/D070 research ordering was T2-E electron/field windows, T2-I
known-ion-increment prefixes and stage reconstruction, T2-C integration, then T3
scaling/deployment. D072 removes that ordering as a prerequisite. The retained
field units include a uniform-periodic affine RK4 prefix,
true nonlinear periodic residual correction, and bounded nonuniform constrained
RZ experiments. They have genuine CPU tree schedules, independent sequential
oracles, GPU time batches, and scoped same-GPU comparisons. Particle prototypes and
two-way FK closure are not prerequisites for the field algorithm. History guesses,
slabs and Block Jacobi remain tools/controls; JVP and propagation preconditioners
may be designed in T2. The original T0/T1 implementation remains unchanged.
See [the planned executable schedule](pseudocode/T2_WINDOW_PLAN.md) and the design
`A1/A1_VALIDATION_AND_SCALING.md` for reference coverage, budgets and stopping.

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
microsteps remain serial in this retained T1 reference. The separate D070
fixed-ion T2-E API implements time-batched residuals and prefix correction;
complete coupling and T3 scaling/deployment remain pending.
The controlled runner is `tools/warp_t1_verify.py`; CPU walltime in this small
fixture is a diagnostic, not a speedup qualification. T0 raw evidence and
`release/a1-t0` at fc9a19e remain the comparison baseline.

Controlled T1 results, known tolerance limits, GPU scope and reproducible
artifacts are recorded in [A1_T1_EVIDENCE.md](A1_T1_EVIDENCE.md).

## Transfer constraints for future particles

A3 keeps bulk-fluid ions and adds FK fast ions on the macro side, with
electrons/fields on the micro side. A4 replaces bulk ions with FK too. Macro
commit frequency and orbit subdivisions are independent: gyrophase, resonance
and target accuracy decide the latter.

Future implementation pseudocode must identify the actual particle traversals,
pushes, gathers, snapshot/path deposition and feedback construction per sweep
and window, together with prediction, reconstruction, memory, communication,
checks and replay costs. Model-stage calls are not particle-push counts.
JVP residual evaluations may repeat expensive orbits. Compare with a reasonable
same-accuracy particle-subcycled reference. Four full sweeps reduced to two
still need a net-work or resource-qualified elapsed-time benefit against that
reference. The design authority is `A1/A1_FK_COST_AND_TRANSFER.md`.

Particle states, weights, boundary events and sources must retain population,
stage, iteration and trial identity, with one source producer, rollback and one
final commit. Moment and path currents remain distinct. Complete A3 feedback
and the all-ion closure are future designs, not new APIs in this source tree.

Actual orbit and E/B gather counts are summed per particle, traversal and
substep; N times an average is valid only for the measured arithmetic average.
Snapshot visits, path segments and stencil writes are separate counts. A shared
source is unique for one stage/feedback-node and generation/trial key, not once
for an entire macro window. Distinct midpoint identities remain distinct.
Future FK representation and closure dependencies are recorded separately in
`A1/A1_FK_COST_AND_TRANSFER.md`; D067 proposals remain archived. The current T2
plan concerns the fluid field window and known-RHS prefix, not a particle gate.

## Implemented fixed-ion field solver and pending integration (D070)

Each real field-window residual reads one old trajectory batch, preserves all
four RK stages and reconstructs the true massless Ohm/CGL constraints. Frozen
fast propagation is an approximate inverse, not frozen physics or a new implicit
time integrator. Matrix-free JVP alone does not provide a parallel inverse.
Production field transport and constraints remain spatially coupled.

An approximate field solve requires the original field residual and complete
ion-stage/endpoint map defect; T1's small ion update alone is insufficient.
Compatible magnetic trial data must preserve CT/wall/flux constraints; electron
canonicalization does not repair arbitrary B. Extend inner-iteration/cache keys
when connecting this API to the outer transaction. The field-only module uses
explicit window/node/stage arrays, trial history and frozen producer identity;
it does not mutate StageTag or shared Context counters.

Count outer S, each field r_Z, future orbit r_p and actual per-particle q
separately. Charge setup/refresh, derivatives, real RHS/constraint applications,
communication, checks, trial rejection and replay. Retain one final stage history
and one commit; no inner history leaks into accepted ledgers. New restart
semantics need compatibility tests and remain unimplemented.

Validate CPU A1 schedules first, then matching GPU over full waveforms, stages,
forces, constraints, cache versions and rollback, then same-GPU A0/T1/new-method
end-to-end performance at equal error and horizon. CPU A0 is not analytic truth;
keep independent manufactured/analytic/sequential oracles. Fixed H with smaller
h and fixed h with longer H are distinct tests. Main-memory and device-memory
peaks, synchronized times, cold setup/compilation/warmup and rejected work remain
separate measurements. No logarithmic-span or CPU/GPU ratio is an algorithmic
speedup qualification.

## Rectangular outer wall on the common A0/A1 spatial path

The D075 rectangle uses the regular RZ field complex and preserves prescribed
normal magnetic flux. It is available through the formal case loader for
synchronous RK4, predictor, complete waveform and few-node WR.
The LCFS is a non-masking interior marker. The input and restart identity
include the physical geometry and signed initial wall data.
See the [source reading guide](pseudocode/src/fomhd/box_boundary.py.md) for
full/fine pressure-force ownership and wall energy quadrature.
[The matched comparison](/home/lxzeng/runs/fomhd/a0-hym-rectangle__20261111T060300Z.X6egNI/REPORT.md)
records the implemented scope and completed A0/A1 verification.
Historical staircase and research-branch outcomes keep their input geometry.
