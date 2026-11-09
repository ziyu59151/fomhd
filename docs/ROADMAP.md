# FOMHD roadmap and release scope

Status: D072 selects few ion nodes or trajectory coefficients, field
microsteps, and multirate waveform relaxation on the 0.5.0 branch. Historical
T0/T1, T2-E and C1/C2 evidence and failures remain unchanged. T2-E, C1/C2 and
unimplemented T tiers are research branches; they are not development or
integration prerequisites. A3/1.0 and A4 retain the scoped roadmap below.

The sibling design repository owns the full definitions in
`mainline_design/numerics/algorithm_design/A0_A3_ALGORITHM_SELECTION.md`
(its retained filename now covers A0 through A4),
`A1/A1_ALGORITHM_DESIGN.md`, and `A1/A1_FK_COST_AND_TRANSFER.md`.
They are review context, never a build or runtime dependency.

## Stages and 1.0

| Stage | Model and purpose | Status |
|---|---|---|
| A0 | Fluid reference: bulk-ion fluid and fluid electrons | Bounded reference evidence retained on `release/a0` |
| A1 | Use the controllable two-fluid system to study time decoupling, parallel iteration, storage and computational scaling for future kinetic hybrids | T0/T1 retained as references; D072 few-node multirate waveform method is active; T2-E and C1/C2 remain research controls |
| A2 | Passive particle verification; read-only fields, no deposition or feedback | Planned; remains optional as an independent stage |
| A3 | Retain bulk-ion fluid and fluid electrons; add self-consistent fully kinetic (FK) fast-ion populations | Planned; stable, reproducible, scoped acceptance is the 1.0 milestone |
| A4 | Replace bulk ions with FK representations too, so all ion species are kinetic; electrons remain fluid | Long-term work after 1.0, not part of the current implementation |

Massless quasineutral CGL-B electrons remain the default through 1.0.
A4 needs its own closure and applicability review; this roadmap does not
restore finite electron inertia or select additional physics.
Do not count one population as both a fluid and a full kinetic response.

Before an actual 1.0 release, the A3 acceptance package must provide:

- An explicit model, population ownership, particle representation, time scheme,
  boundary, precision and supported hardware scope.
- An actual two-way feedback case, zero-FK and disabled-feedback controls,
  consistent species charge/current/power, and multi-population controls for
  the support being claimed.
- A declared observation window and response/phase/growth error budgets,
  appropriate time, orbit, feedback, grid and marker controls, and retained
  hard failures and known residuals. Physical unstable growth is allowed.
- Exact source, lock, inputs, seeds, hardware and commands; repeatability within
  declared numerical/statistical budgets; accepted-endpoint restart agreement.
- Reviewed design/pseudocode/source correspondence, usable figures, known
  limitations, and the user's acceptance of the concrete release scope before
  creating a release branch or immutable tag.

Specific A3 cases and numerical budgets remain to be defined. All-ion kinetic
physics, every long-term target, every T tier, and a speedup result are not
universal prerequisites for 1.0. An ICE claim needs appropriate gyrophase,
resonance and feedback evidence; fluid bulk cannot validate its missing kinetic
damping. No 1.0 release is declared here.

A4 first defines the all-ion charge/current/moment closure, then replaces one
bulk population in controlled tests, checks its coupling with fast ions, and
extends to all ion populations. The minority coupling of A3 must not be copied
without deriving the replacement model. Fluid and kinetic results are compared
only in identified common limits.

## What the A1 time experiments establish

Synchronous fluid work is currently acceptable. Two-fluid A1 is a validation
platform while a stable FK-fluid feedback implementation is not yet available;
transferability to expensive future particle calculations is a selection
criterion from the outset.

T0 exposed the mismatch between a predicted ion path and the integrated ion
path. T1 supplies a serial, self-consistent complete stage-path reference whose
exact fixed point is synchronous fine-step RK4. Each sweep still evaluates the
complete ion RHS at every microstep stage. Committing only at a macro endpoint
does not establish fewer ion calculations.

For A3, both bulk-fluid ions and FK fast ions belong to the macro side, and
electrons/fields to the micro side. Macro scheduling does not prescribe a
single large orbit step: orbit subdivisions must resolve the target gyrophase,
resonance, spatial crossing and accuracy requirements. A4 must inherit the
corresponding all-ion capability.

Future proposals must count particle pushes, field gathers, snapshot/path
deposition, feedback construction, full orbit traversals, history reconstruction,
communication, checks, rejected trials and JVP replay per sweep and macro window.
Microstep count, orbit subdivisions, population counts and marker counts are
separate parameters. Compare with a reasonable synchronous or particle-subcycled
reference that reuses field and moment data, not an artificially costly replay.

## Current selected route and retained research

D072 supersedes the development order in this section's historical T mapping.
The active method directly solves a small number of ion nodes and exchanges
their polynomial waveform with real RK4 field microsteps. Expensive ion
transport remains coarse after convergence. Its mixed-history discrete target
has independent accuracy, stability, work and transaction acceptance. The
design authority is `A1/A1_WR_050.md`. New accepted capabilities may merge to
main only after complete advance, cumulative work, rollback, restart,
CPU/GPU integration and independent review pass the declared scope.
Otherwise the method stays on the experimental branch with explicit blockers.

The following T mapping is retained research context, not a required sequence.

T0 retains the single tangent prediction and full history quadrature. T1 retains
the serial complete-stage waveform reference, original data, tolerance limits,
M128 failure and bounded GPU scope. Their historical reports' next-step sections
retain their historical meaning; they do not set the current implementation order.

- T2-E implements bounded fixed-ion-path electron/field RK4 windows: actual
  CPU/GPU affine trees, nonlinear periodic true-residual correction and real
  nonuniform RZ/wall controls. Same-GPU benefits, long-window convergence and
  constraint-representation failures are recorded separately. Structured
  large-space propagation and storage still need work; no dense field prefix
  is required or constructed.
- D071 scales complete A0/T1 now and screens at most two low-ion-work
  multirate discretizations. Expensive ion work, cheap grid feedback and field
  work are counted separately. Fixed-H h refinement and fixed-h H scans answer
  different questions; useful prototypes proceed CPU then GPU.
- Known-RHS ion prefixes only shorten accumulation dependencies. Integration
  and space/time deployment tools remain available when a selected algorithm
  needs them; they do not impose the development order for this campaign.
- Spatial and temporal multi-GPU are separate later measurements after
  single-GPU capacity and bottlenecks are established.

Former T2 history prediction remains an initialization tool. Former T2 particle
operations/feedback proposals remain future FK work, not field-solver prerequisites.
Former T3 slabs are grouping/deployment tools, not a prescribed algorithm. Former
T4 JVP/preconditioning may enter T2 when the field method needs them; no separate
mandatory T4 remains. Joint full-system Newton is conditional, not added scope.

Reducing expensive work and accepting more work to shorten serial dependence are
both valid research directions. No candidate must first reduce total particle
operations. Small iteration counts and ideal logarithmic span are not measured
end-to-end gains. Fixed complete E/B history may give low-dimensional independent
particle trajectories; fixing B and electron pressure while reconstructing E from
new ion feedback does not guarantee that independence. Future orbit Newton-prefix
iterations must correct the real nonlinear orbit and count matrix/vector work.

For each bounded unit, implement a real CPU A1 window/tree schedule, compare with
an independent oracle, validate the matching GPU algorithm, then compare old and
new methods on the same GPU at matched error and physical horizon. Existing CPU
T0/T1 are reusable; CPU A0 is a reference implementation, not analytic truth.
CPU/GPU timings express hardware/backend benefit and are reported separately.
Large GPU cases need not be rerun completely on CPU at the same size.

The full candidate equations and first-unit caps are in design authority
`A1/A1_T2_CANDIDATES.md`; references, actual dense bottlenecks, grid/mode and
H/h/M/horizon growth, memory/jobs, acceptance/stopping and report templates are in
`A1/A1_VALIDATION_AND_SCALING.md`. The planned executable schedule is
[pseudocode/T2_WINDOW_PLAN.md](pseudocode/T2_WINDOW_PLAN.md). Implementation, correctness and performance claims require their versioned
unit reports. A local field or kernel gain is not complete-system acceptance.
