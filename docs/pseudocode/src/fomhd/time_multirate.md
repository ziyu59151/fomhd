# Bounded reduced-hydro ion multirate candidates C1 and C2

## Status and scope

This is a research-only implementation specification. It defines two bounded
one-macro-window algorithms for the existing massless-electron/CGL-B branch.
It does not select a production integrator, create an accepted transaction,
implement an interval-work budget, apply a production accept/reject decision,
or establish scientific acceptance. Trial iterations are numerical research
records, not physical energy or power ledgers.

Authority remains with `/home/lxzeng/fomhd-design` and the selected physical
equations in `fomhd.massless`, `fomhd.fluid`, and `fomhd.field_batch`. The
pseudocode describes orchestration and numerical approximations only.

State layout:

```text
P[q, 5*Nc] = (n_i, mom_i_R, mom_i_phi, mom_i_Z, p_i)
Z[q, 2*Nc + Nb] = (p_perp_e, p_parallel_e, B)
```

`q = 2*K+1`; mode order is `-K..K`. All owners are FP64/complex128 on the
context-selected NumPy or CuPy backend.

The split uses the current scalar-pressure ion branch:

```text
D(P)   = -DJ I mom_i/m_i
A_m(P) = FFT Pi_n[-convection(P) + pressure_force(P)]
A_p(P) = FFT pdot_i(P)
L(P,E,B) = FFT Pi_n[
    q_i n_i H E
    + (q_i/m_i) mom_i_samples cross QB*B
]
G(P,Z) = real massless-electron/field RHS from FieldBatch
```

`Pi_n` is the current density-dependent wall momentum projection. `H` is the
edge-to-node interpolation operator, not a time interval. `A_m` and `A_p` are
B-independent for this scalar-pressure branch, but `build_non_electric`
requires valid magnetic metadata. The adapter therefore supplies the current
valid `B`/`bhat`, extracts only `-convection + pressure_force`, and never treats
the whole `build_non_electric` output as macro acceleration.

Here the temporal macro interval is the distinct symbol `H_t`, and the actual
microstep is `h=H_t/M`. The harness passes `H_t=M*h` to each research window;
it never passes the requested microstep as the macro interval.

## Common API

```text
run(method, context, P0, Z0, H_t, M, *, rtol, max_iterations,
    keep_stages, endpoint_check):
  require method in {C1,C2}
  require H_t>0, M>=1, max_iterations>=1
  require C2 M even and M>=2
  require P0/Z0 backend arrays with exact layouts and copied ownership
  create fresh counters and immutable iteration records
  do not modify P0, Z0, context counters, accepted state, ledgers, or cache

  if method==C1:
    return centered_impulse(...)
  return frozen_midpoint(...)
```

On every failure:

```text
attach counters, completed iteration history, last attempted P/Z,
  residual, and every completed current-sweep stage to a research Failure
record the attempted P/Z stage inputs before dispatching each callback
preserve the caller's last returned trial unchanged
do not commit or repair arrays
```

The returned trial owns copies. Iteration records and stage records are tuples
of frozen records. Inputs may be borrowed only during one call.

## Adapter and counters

```text
adapter.macro(P,Z):
  count full_hydro_eval += 1 before any reconstruction work
  D   = cheap_density_rate(P)
  ion = build_fluid(P ion block, species 0, current valid B/bhat)
  build_non_electric(ion, ...)
  Fm  = Pi_n[-ion.convection + ion.pressure_force]
  A_m = FFT Fm
  A_p = FFT ion.pdot_perp
  count cheap_D += 1
  count wall_projection += 1 and wall_normal_initial_solves += N
    immediately before project_wall, but use += 0 when T has zero rows

adapter.field(P,Z):
  count field_rhs += 1 before dispatch
  result = existing FieldBatch.rhs(P[None], Z[None])
  return G=result.rhs[0], E=result.electric[0], B=Z.B,
         derived electron density/velocity

adapter.cheap_L(P,E,B):
  count cheap_L += 1 before dispatch
  evaluate only inverse/H interpolation, existing wall projection and
    the two Lorentz terms in L
  do not evaluate velocity gradients, convection, pressure transport,
    pressure force, pdot, CGL, or the full ion RHS
  count wall_projection += 1 and wall_normal_initial_solves += N
    immediately before project_wall, but use += 0 when T has zero rows

adapter.impulse(P, integral_E, integral_B):
  apply one wall projection to
    q_i n_i H integral_E
    + (q_i/m_i) mom_i cross QB*integral_B
  do not apply the physical-B magnitude floor to an integrated magnetic
    impulse that is temporarily near zero
  count cheap_L += 1 and the wall counters

adapter.endpoint(P,Z):
  count endpoint_check += 1 before dispatch
  evaluate FieldBatch once to recover G, electric field, electron density,
    electron velocity, and compatibility data
  FieldBatch itself counts field_rhs += 1

independent_full_ion_check(P,Z):
  count endpoint_full_ion_RHS += 1 and endpoint_check += 1 before dispatch
  evaluate one complete production-equivalent stage on an isolated context
  compare D(P)+A_m(P)+A_p(P)+L(P,E(P,Z),B) with its ion RHS
  also charge its full_hydro_eval separately
```

Required counters are:

```text
actual_outer_calls, full_hydro_eval, cheap_D, cheap_L, field_rhs,
wall_projection, wall_normal_initial_solves, quadrature_contributions,
endpoint_check, endpoint_full_ion_RHS
```

All rejected outer iterations are counted. The normal-solve count records the
initial componentwise solve used for each spatial sample; production
`project_wall` does not expose refinement solves, so that distinction is
reported rather than guessed. It covers only these adapter ion projections.
FieldBatch normal-system, SVD, diagonal-system, and solve work is exported in
its own `field_batch_work` ledger.

The generic callback kernel and the production adapter both observe the same
attempt, including callbacks that raise. Every attempt is counted immediately
before dispatch. When their snapshots are combined, only duplicate observations
of the same actual full-hydro, cheap-D/L, outer, or field call take the maximum
rather than being added twice. The endpoint full-ion check remains a separate
`endpoint_full_ion_RHS` charge. Harness-only diagnostic stages use a separate
work ledger and are never folded into candidate counters.

## C1 centered impulse

Unknown state is `Pc = (P0+P1)/2`. Set the initial fixed-point guess `Pc=P0`.

```text
for outer_call in [0,max_iterations):
  count actual_outer_calls += 1
  count full_hydro_eval += 1 before callback dispatch
  D,A_m,A_p = adapter.macro(Pc,Z0)

  z = Z0
  stages = []
  integral_E = 0
  integral_B = 0
  for micro in [0,M):
    k = [None,None,None,None]
    stages_z = [z, None, None, None]
    for r in [0,1,2,3]:
      z_r = RK4 evaluation state from z and previous k
      if r==0: z_0 = current z
      if r==1 or 2: z_r = z + (h/2) k0 or (h/2) k1
      if r==3: z_r = z + h k2
      count field_rhs += 1 before dispatch
      G_r,E_r,B_r = adapter.field(Pc,z_r)
      k[r] = G_r
      integral_E += h*w[r]*E_r
      integral_B += h*w[r]*B_r
      count quadrature_contributions += 1
      retain distinct stage record (r=1 and r=2 are not aliased)
      append the stage record immediately so a later callback failure retains it
    z = z + h*(k0+2*k1+2*k2+k3)/6

  count cheap_L += 1 before impulse dispatch
  record guessed P and advanced z as the attempted impulse inputs
  P1 = assemble(
    n1   = n0   + H_t D
    mom1 = mom0 + H_t A_m + adapter.impulse(Pc,integral_E,integral_B)
    p1   = p0   + H_t A_p
  )
  Pmid = (P0+P1)/2
  residual = Pc-Pmid
  metrics = physical-scale normalized max residual
  append immutable iteration record
  if metrics.converged:
    endpoint = adapter.endpoint(P1,z)
    trial = immutable(C1,P1,z,endpoint,...)
    if endpoint_check: run independent_full_ion_check(P1,z)
    return trial
  Pc = Pmid
fail with C1Convergence and retain all iterations/stages
```

The first field stage is `G(Pc,Z0)`. The integrated electric/magnetic impulses
use the exact RK4 stage weights over all `M` microsteps. Only one wall
projection is applied to the interval-averaged electromagnetic response because
`Pc` is fixed. C1 is second order in `H_t` plus fourth order in the microstep and
is not a fourth-order macro method. No exact nonlinear energy identity is
claimed.

## C2 midpoint-frozen hydro

Unknown state is the actual microgrid midpoint `Pc=P(H_t/2)`. Set `Pc=P0`.

```text
for outer_call in [0,max_iterations):
  count actual_outer_calls += 1
  count full_hydro_eval += 1 before callback dispatch
  A_m,A_p = adapter.macro(Pc,Z0)  [D is not used here]

  p,z = P0,Z0
  midpoint = None
  stages = []
  for micro in [0,M):
    for r in [0,1,2,3]:
      y_p,y_z = RK4 stage state from current p,z and prior rates
      count cheap_D += 1, field_rhs += 1, cheap_L += 1
        immediately before each corresponding dispatch
      D_r = adapter.cheap_density_rate(y_p)
      G_r,E_r,B_r = adapter.field(y_p,y_z)
      L_r = adapter.cheap_L(y_p,E_r,B_r)
      rate_p = (D_r, A_m+L_r, A_p)
      rate_z = G_r
      retain immutable stage record immediately, including on later failure
      count quadrature_contributions += 1 after a successful field stage
    advance p,z by h/6*(k0+2k1+2k2+k3) componentwise
    if micro+1 == M/2: midpoint = copy(p)

  residual = Pc-midpoint
  metrics = physical-scale normalized max residual
  append immutable iteration record
  if metrics.converged:
    endpoint = adapter.endpoint(p,z)
    trial = immutable(C2,p,z,midpoint,endpoint,...)
    if endpoint_check: run independent_full_ion_check(p,z)
    return trial
  Pc = midpoint
fail with C2Convergence and retain all iterations/stages
```

Every outer sweep performs exactly `4*M` cheap `D` evaluations, `4*M` cheap
`L` evaluations, and `4*M` real `G` evaluations. It performs one full hydro
evaluation for the frozen macro acceleration/pressure rate. It never performs
`4*M` full hydro evaluations.

The stage history owns, for each of `4*M` rows, the RK4 state `(P,Z)`, rates
`(Pdot,Zdot)`, and `(E,B)` values. C1 marks `Pdot` unavailable because a
prescribed centroid path has no physical derivative to export; its NPZ writer
omits a fake all-zero P-RHS array and stores an availability mask. C2 marks
the live rate available. Report the summed actual array bytes as
`stage_history_bytes`; this is history storage, not a measured allocator peak.

C2 preserves each actual coupled microstage on its cheap trajectory. Its own
frozen-hydro quadrature is second-order coupling with fourth-order micro RK4.
It coincides with coupled RK4 when `A_m=A_p=0`. It is not the synchronous T1
fixed point for general nonzero hydro terms, and its grid response is not an FK
particle-phase result.

## Residual normalization

The fixed-point P state uses these component scales:

```text
n_i:        ref.ni
mom_i:      m_i ref.ni ref.velocity
p_i:        ref.pressure
```

The distinct Z layout uses:

```text
p_perp_e:      ref.pressure
p_parallel_e:  ref.pressure
B:             ref.magnetic
```

For component `j`, the fixed-point normalization against the declared
physical scale is

```text
denominator_j = max(scale_j, 1e-300)
normalized_j  = abs(guess_j-target_j) / denominator_j
normalized    = max_j normalized_j
```

Candidate/oracle component errors use
`max(scale_j, abs(oracle_j), 1e-300)` as the denominator. P and Z retain
separate shape-checked scale arrays. No density norm is used for pressure or
magnetic components, and no global maximum crosses physical units.
Report raw and normalized values, scales, and the actual largest increment.
The actual increment is `max(abs(target-guess))`, evaluated elementwise before
any reduction; it is never the difference between the two maximum norms.
Convergence requires the normalized next-iteration residual to be at most
`rtol`.

## Endpoint reconstruction and diagnostics

`adapter.endpoint` recomputes massless-electron density, velocity,
compatibility momentum, electric field, current, and wall residuals from the
candidate endpoint. No accepted ledger or cache is changed.

```text
state_from_packed(P,Z,field=None):
  require P shape (q,5*Nc), Z shape (q,2*Nc+Nb)
  require finite positive q_i/e
  derive retained electron density n_e_hat=(q_i/e)n_i_hat
  use physical FieldBatch density only to build compatibility momentum
  validate all retained arrays, extents, Hermiticity, and B
  validate sampled physical ion density and pressure floors
  expose the FieldBatch identity as the research cache version/tag
```

The endpoint path therefore works for K>0 closed-RZ states without storing
physical sampled electron density in a retained Fourier slot. Optional full
source checks use blockwise `(q,Nc)`, `(q,3*Nc)`, `(q,Nc)` packing rather than
concatenating flattened field buffers.

The trial exposes:

- complete macro endpoint P/Z;
- midpoint P and fixed-point/endpoint residuals;
- integrated E and B impulses;
- optional complete RK4 stages and stage E/B;
- endpoint FieldBatch checks and reconstructed electron compatibility fields;
- raw scalar power/energy observations where available.

The harness may compute initial/final `state_totals` and per-stage
`P_i = q_i n_i u_i dot H E`, field-energy rate, and electron-energy rate for
each window's converged final outer waveform. These are instantaneous
diagnostic samples and RK4 stage increments, not a production energy or power
ledger. Stages from rejected or earlier fixed-point trial iterations are
retained as research evidence but excluded from the stage-energy rows. The
electron linear pressure-energy rate is

```text
2*pi * sum_cells(Zdot_perp_e[K,cell] + 0.5*Zdot_parallel_e[K,cell])
```

Only toroidal zero mode `K` enters this linear Fourier integral. The magnetic
rate remains the separate bilinear Parseval sum over all retained modes with
factor `2*pi`.
Every attempted harness diagnostic full stage is charged separately in
`diagnostic_work`; initial/final energy stages and closed-RZ hard audits are
not hidden inside candidate counters. A nonzero raw residual is retained.

The closed-RZ audit keeps the exact original hard/observation budget flags.
It does not promote observation failures into hard checks. Closed runs retain
component errors, constraints, and raw energy but do not assign a phase from a
flattened RZ-plus-toroidal vector. Periodic runs retain separate target and
highest-mode sentinel histories with maximum/final gain; if both labels select
the same mode, the sentinel is explicitly non-independent.

## Independent controls

Generic callback kernels are public so tests can use non-FOMHD equations.

Cold Hall control:

```text
u' = i a b
b' = i a u + i beta b
```

The independent helper uses the exponential-integrator convention
`phi_n(z)=sum_k z^k/(k+n)!`; at zero, `phi_n(0)=1/n!`.

Compare RK4 and both candidates with `scipy.linalg.expm` or an independent
closed-form 2x2 exponential. Check:

- the exact continuous field map is unitary;
- actual repeated RK4 has the documented finite-step gain;
- beta -> 0 recovers the Cayley-like cold limit;
- C1 removes the persistent step-centred macro gain but can retain phase error;
- C2 exactly reproduces coupled RK4 when the frozen hydro callback is zero.
- T0 agrees with the augmented exponential at zero, tiny positive/negative
  `beta*H_t`, both sides of the series branch, and ordinary arguments.

Full spatial split control:

```text
compare D+A_m+L with a complete production ion RHS at nonuniform periodic
and closed RZ states
check Fourier mode/transpose conventions
check the same A_m/A_p under different valid B metadata
use scale-based tolerances and retain both real/imaginary errors
```

Failure controls cover outer-cap rollback, unchanged accepted arrays and
generation/cache counters, exact `4*M` workload counts, and convergence under
`h/2` and `H_t/2` refinements.

## Harness

`tools/warp_multirate_ion.py` accepts CPU or `cuda:0` only, writes a new output
directory, and refuses GPU execution without `SLURM_JOB_ID`.

```text
case: periodic (default 33 points, amplitude 1e-4, mode 1 plus sentinel)
      closed (8 or 12 points, amplitude 0.02, nonuniform 0.1)
method: C1 or C2
controls: --h --M --macros --rtol --max-iterations --seed-mode --sentinel
          --macros is bounded to [1,32]
```

`--h` is the microstep. The harness passes `H_t=M*h` to each candidate window.
For `macros` windows, both candidate and synchronous reference advance
`macros*M*h`. `report.json.time_contract` records start/end time, microstep,
macro interval, steps per macro, macro count, and the common total duration
directly from controls.

An independent same-`h` synchronous RK4 oracle uses the complete production
stage, not only the `G` producer. Waveform/full-T1 reference execution is
optional and uses the existing production scheduler. Candidate and reference
use the same fresh isolated-context protocol: one cold execution, the requested
warmups, and the requested synchronized repetitions. Per-execution counters
remain in the timing record, and campaign work is reported with multiplier
`1 + warmups + repetitions` for each algorithm. Fixture setup and output I/O
are excluded. CPU timing is not a CPU/GPU speedup claim.

`status=completed` and `execution_completed=true` describe execution only.
`scientific_qualified=false` remains until nonzero-hydro refinement evidence
supports the formal second-order macro design and the parent declares the
comparison qualified.

Products are:

```text
input.json, provenance.json, report.json, arrays.npz
```

`report.json` records status, the independent time contract, exact candidate
and diagnostic counters, iteration residuals with macro-window index/physical
time, failures, final-outer stage diagnostics, field-batch work, matched timing
scope, qualification state, and explicit non-claims. `arrays.npz` retains
short-scope full trial waveforms/stages with macro-window index/physical time,
P-RHS availability, stage E/B, separate `p_scales`/`z_scales`, endpoint Z, and
periodic target/sentinel histories.

## Bounded smoke and acceptance

The worker runs cheap unit tests plus one short periodic endpoint smoke and one
short closed endpoint smoke. The parent selects and runs the scientific matrix.
No 18-case matrix is launched here.

Minimum acceptance commands:

```text
PYTHONPATH=<scratch>/src:<source>/src:<source>/tests/warp:<source>/tools \
  <source>/.venv/bin/python -W error -m pytest -q \
  <scratch>/tests/warp/test_time_multirate.py

same PYTHONPATH, then:
  <source>/.venv/bin/python <scratch>/tools/warp_multirate_ion.py \
    --case periodic --method C1 --device cpu --h <bounded-h> --M 4 \
    --macros 2 --endpoint-check --output <new-periodic-directory>
  <source>/.venv/bin/python <scratch>/tools/warp_multirate_ion.py \
    --case closed --method C1 --device cpu --h <bounded-h> --M 2 \
    --macros 1 --endpoint-check --output <new-closed-directory>
```

Passing these commands proves only the measured unit and short periodic/closed
smoke scope. Execution status does not imply scientific qualification. It does
not prove production acceptance, an interval-work budget, measured
nonzero-hydro second order, asymptotic work reduction, nonlinear energy
conservation, or GPU performance.
