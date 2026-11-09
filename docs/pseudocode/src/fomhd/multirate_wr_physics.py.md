# Massless A1-WR physics adapter pseudocode

Status: bounded implementation contract for parent review. This is not a
production accepted-state or cumulative-ledger transaction.

Sources:

- `mainline_design/numerics/algorithm_design/A1/A1_WR_050.md`
- sibling `src/fomhd/time_multirate_wr.py`
- sibling `src/fomhd/massless_field.py`
- sibling `src/fomhd/massless.py`

## Layout and ownership

```text
context owns Parameters, Geometry, Operators, Backend, stream/device, and the
fixed wall cache. The adapter owns its counters and returned trial records.

P: complex128 array [q, 5*Nc]
   columns 0:Nc                  ion density [1/m^3]
   columns Nc:4Nc                ion momentum density [kg/(m^2*s)]
   columns 4Nc:5Nc               isotropic ion pressure [Pa]
Z: complex128 array [q, 2*Nc+Nb]
   columns 0:Nc                  electron perpendicular pressure [Pa]
   columns Nc:2Nc                electron parallel pressure [Pa]
   columns 2Nc:2Nc+Nb            magnetic field in the existing entity layout

q = 2*K+1, modes are stored -K..K, vectors are R,phi,Z, and
Nc/Nb are context operator counts. Arrays stay on context.compute.xp.
```

`state_from_packed` allocates fresh component arrays, copies the five
independent P blocks, derives electron density exactly from quasineutrality,
sets only compatibility electron momentum storage to zero, copies the Z
blocks, and sets the requested identity. It never calls
`canonicalize_state`, `_assemble_state`, or a full ion RHS.

`pack_state` and `pack_stage_rhs` perform the inverse layout for read-only
evidence and split validation.

## Callback identities

```text
one adapter window:
  base = accepted-left State
  trial = fresh context.next_trial
  slab = logical macro index
  all StageTag.time values equal the exact float base.time

each physical callback:
  generation = fresh context.next_generation
  StageTag.step = base.step
  StageTag.trial = trial
  StageTag.spectral_context = context.spectral_id
  StageTag.slab = slab
  StageTag.stage = "wr_<role>"
  iteration, micro_step, rk_stage copied from WindowIdentity

the two same-theta RK middle stages differ by generation, rk_stage, and Z
state. No native source identity is fabricated or inferred.
```

The raw state generation and time equal the new StageTag values. Producer
deltas are measured around actual calls:

```text
context.next_generation, next_trial, next_source
massless_field.WorkCounts
wall-cache evaluation/setup counters
adapter callback and endpoint counters
time.perf_counter_ns
```

## Fine field callback

```text
field(theta,P,Z,identity):
  frame = massless_field.evaluate(state_from_packed(base,P,Z), tag,
                                  context,
                                  alias=(rk_stage in {1,2}))
  feedback = massless_field.feedback_rate(frame,context)
  G = concat(frame.field_rates.electron_perp,
             electron_parallel,
             B)
  C = concat(feedback.n,
             feedback.mom,
             feedback.perp)
  feedback_rate stores frame.ion_em_terms from the same counted producer call
  retain the frame, feedback, tag, checked residual maxima, wall-cache identity,
  and scalar fine electric exchange <u, electric> from ion_em_terms.electric
  (unprojected node-force convention; wall work remains separate)
  return (G,C,FieldRecord)
```

No `stage.evaluate_stage`, `finish_fluid`, `build_non_electric`, or
`power_ledger` call occurs in a fine field callback. Middle-stage alias calls
are charged from actual `WorkCounts`; guessed `s` or `4M` multipliers are not
used. `fine_ion_power_ledger_calls` is explicitly zero; it is not presented as
a total transaction power-ledger count.

## Coarse transport callback

```text
transport(theta,P,Z,identity):
  full = _full_stage(state_from_packed(base,P,Z),tag)
  F = pack_stage_rhs(full,species=0)

  A.n = 0
  A.mom = FFT(ProjectWall(ion.wall,
                            -ion.convection + ion.pressure_force))
  A.pi = F.ion_pressure_rate

  C_from_frame.n = F.ion_continuity
  C_from_frame.mom = FFT(ProjectWall(ion.wall,
                                     ion.magnetic_force +
                                     full.terms[0].electric))
  C_from_frame.pi = 0

  require A+C-F within a scale-aware FP64 identity tolerance
  evaluate feedback-equivalent C using massless_field-style actual terms,
  without a second full RHS
  return (A,CoarseRecord(full,ion,A,C,F,split,
                         transport_projection,identity,...))
```

The coarse callback returns `A`; the fine callback returns its own `C`.
`CoarseRecord.C` and `F` are validation evidence only. One complete physical
stage is charged per transport callback. `_full_stage` increments an attempt
before the physical call, increments success/failure at the actual boundary,
and counts ion/electron ledger values only when the full producer returns them.
Failed attempts are never relabeled as successful ledger calls.

## Window solve and endpoint

```text
solve_window(adapter,P0,Z0,H,M,s,...,endpoint_tag_factory):
  snapshot caller state/P/Z and cumulative-ledger sentinel
  time_multirate_wr.solve_window(P0,Z0,H,M,s,
                                 adapter.transport,adapter.field,...)
  if success:
    full_endpoint = _full_stage(
        state_from_packed(accepted_left,window.P[-1],window.Z_endpoint),
        endpoint_tag_factory())
    require endpoint packed independent P/Z equals accepted endpoint
  accepted records = callback records whose iteration equals
                     window.accepted_node_version
  require exactly s accepted coarse and 4*M accepted fine records
  retain all callback records separately with explicit counts
  return WindowPhysicsResult(window,full_endpoint,
                             accepted_records,all_records,
                             counters,
                             source_deltas,clock,version_evidence)

endpoint full stage is charged once and marked
trial_endpoint=true, interval_transaction_required=true, and
production_accepted=false. Failure preserves the window partial arrays,
adapter counters, latest actual source delta, and unchanged caller inputs.
```

The result contains raw endpoint totals, endpoint checks, fine field checks and
residual maxima, scalar electric exchange samples, and no full mixed-ledger
claim.

## Benchmark runner

```text
warp_multirate_wr --input INPUT.json --output OUTPUT.json

strict input:
  case, points, K, amplitude, nonuniform, seed_mode, sentinel, h, M, s,
  macros, device, coupling, rtol, max_iterations, wall_seconds
  h may be omitted and defaults to 1.9794733885623e-10 s

periodic:
  use the physical HallCase/backend/operator builders and only the density,
  velocity, pressure, and B seed formulas from the existing T2 periodic
  fixture builders; add the independent high-mode sentinel seed;
  run one adapter WR window per macro and carry only accepted old endpoints

reference:
  independent straight full-RHS RK4 at the same h and at h/2 (2M substeps per
  macro), using evaluate_stage and direct RK4 combinations, never solve_window

east:
  validate and return an explicit unsupported interface result until the
  parent supplies the pinned g048059.03650 equilibrium and actual K/Nphi/h
  spectrum through the documented existing `_make_context` seam. This is an
  explicit temporary parent-owned extension and not representative coverage.

output:
  create only a new file, refuse any existing target; write JSON atomically;
  retain exact config, source hashes, stage identities, diagnostics,
  counters, wall/CPU memory samples, scalar timings, accepted mode amplitudes,
  bounded lossless base64 NPZ state/stage snapshots, failures, and completed
  macro windows. CUDA runs require an owned Slurm allocation; no fallback.
  Every runtime source/evidence path resolves relative to the executable
  source root; no worker-scratch path is a runtime dependency.
```
