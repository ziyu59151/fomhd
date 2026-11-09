# Massless field-only implementation pseudocode

This record is an A1 grid-feedback seam, not a time scheme. It evaluates the
massless electron/field Z response at one independent P/Z state and never
claims a complete ion RHS, production diagnostics, T2-E, C1/C2, or a
waveform sweep.

## A1-WR-050-FIELD-01: Ownership and result record

```text
evaluate(state,tag,context,alias=false):
  require tag step/generation/time equals state in the field failure boundary
  request fixed wall cache
  derived=DeriveElectron(state,context,ion_transport=false)
  trial=_copy_state(state, derived.electron_density_hat,
                    derived.compatibility_momentum)
  CheckRetainedMassless(trial)
  CheckInitialDBAndFullWallCurrent(derived,trial.B)

  electron=ElectronCGLB(derived,trial,tag)
  electron_magnetic_forces=1
  if backend.name == "cuda":  # exact _warp_magnetic_force dispatch condition
    warp_magnetic_kernel_evaluations+=1
  E0=MasslessOhmSource(electron,derived)
  reaction,wall_records=ApplyActualFieldWallCache(E0)
  E=E0+reaction

  field_rates.electron_perp=FFT(electron.pdot_perp)
  field_rates.electron_parallel=FFT(electron.pdot_parallel)
  field_rates.B=-C*E

  assemble field-only residual records and hard gates
  return FieldFrame(
      kind=massless_field_only,
      complete_stage=false,
      owned trial_state, field_rates, actual cache,
      ion=derived.ion, electron, BN, bhat, current, E0,
      owner_context=context,
      ion_em_terms=none,
      explicit tag, residuals, checks, actual WorkCounts)
```

There is no `.rhs.fluid[0]`, ion ledger, fake ion pressure rate, or call to
`finish_fluid`/full `check_stage`. `derive_electron` is called once for the
base field evaluation. Input arrays are not mutated; all outputs belong to
`context.compute`. The context retains ownership of parameters, geometry,
operators, and the immutable fixed wall cache; callers keep it alive while the
returned cache record is in use. `owner_context` is the exact producing context
object, including its parameter/operator/geometry/backend identities. No
global freeze or context mutation is introduced.

`electron_magnetic_forces` is the logical electron force evaluation and is one
for every base/alias field evaluation on either backend. The separate
`warp_magnetic_kernel_evaluations` counts actual CUDA kernel dispatches: one
electron dispatch per evaluated field frame plus one ion dispatch per explicit
EM-rate call. CPU retains zero Warp dispatches and does not masquerade its
fallback as a kernel. Alias aggregation adds the alternate frame's logical
electron force and backend-specific Warp count without executing another
diagnostic kernel.

## A1-WR-050-FIELD-02: Hard field checks

```text
hard checks:
  finite independent state, trial compatibility fields, E/J/ue, and Z rates
  independent P/Z Hermitian spectra and compatibility momentum Hermitian form
  exact derived electron density/momentum compatibility
  positive physical densities; current nodal B above B_min
  DB*B=0 and every full wall-current row=0 before Ohm solve
  current J=Cdag*B/mu0
  electron flux Fe=(q_i Fi-J)/e and continuity D*Fe
  D*J=0
  C*E+Bdot=0 and DB*Bdot=0
  every original Ohm wall row and E-E0-reaction=0
  electron CGL-B pressure-rate definitions
```

Residuals use the existing `diagnostics.measure` budgets and formulas.
`theta_consistency` remains available and nonhard exactly as in the full
massless stage. A hard failure retains the explicit tag and available partial
arrays without adding unavailable full-stage claims.

## A1-WR-050-FIELD-03: Field-only alias

```text
if alias:
  updated=N->2N+1, spectral_context+1, history samples retained
  alternate_tag=alias(stage identity copied except spectral_context)
  alternate=evaluate_field_only(trial,alternate_tag,updated)
  charge one additional derive/electron-CGL/Ohm/wall/field evaluation
  compare alternate vs base electron_perp, electron_parallel, and B
  use legacy alias weights, units, Fourier normalization and hard budgets
  do not call legacy full-ion _alias_probe
```

The base record stores the alternate frame and its explicit extra work. It does
not relabel the alias evaluation as a production stage.

## A1-WR-050-FIELD-04: Ion electromagnetic and feedback grid rates

```text
electromagnetic_rate(frame,context):
  require context is frame.owner_context before any force/work
  require parameter/operator/geometry/backend identities and tag/cache/wall
  require frame source/tag/generation and shape identity
  magnetic_raw=WarpMagneticForce(...)  # actual CUDA output is flat N*3*Nc
  if magnetic_raw is present:
    magnetic=magnetic_raw.reshape(N,3,Nc)  # owning view; no copy/transfer
  else:
    magnetic=q_i*n_i*u_i cross B_N        # CPU producer is already (N,3,Nc)
  E=inverse_physical(frame.cache.E)
  electric=q_i*n_i*H*E
  projected=ProjectWall(frame.ion.wall,magnetic+electric)
  frame.ion_em_terms=EMTerms(
    electric=electric,
    magnetic=magnetic.reshape(-1),
    projected=projected.value,
    reaction=projected.reaction,
    multiplier=projected.multiplier,
    multiplier_residual=projected.multiplier_residual,
    tag=frame.tag)
  return unchanged (zero_density, FFT(projected.value), zero_pressure)

feedback_rate(frame,context):
  em=electromagnetic_rate(frame,context)
  return (-frame.ion.divergence_hat, em.momentum,
          zero_pressure)
```

All arrays are retained SI complex128 arrays. The projected EM momentum has
units kg/(m^2*s^2); feedback density is 1/(m^3*s). The density term is the
already-derived linear ion flux divergence from `build_fluid`, not a second ion
transport evaluation. A production full derivative minus this feedback rate is
the explicitly booked transport contribution.

`ion_em_terms` starts as `None` when the field frame is created. One
`electromagnetic_rate` call fills it from that call's exact padded nodal
producer and wall projection; diagnostics must consume those retained arrays,
not reconstruct them from retained momentum. `electric`, `magnetic`,
`projected`, and `reaction` use the padded `[sample][R,phi,Z][cell]` nodal force
layout and kg/(m^2*s^2) units. `multiplier` and `multiplier_residual` retain the
wall-projection work shape, and `tag` is the owning field-stage tag.

The record and its backend-owned arrays are read-only to consumers and live as
long as the frame/context owning them. The producer may be called again by a
later explicitly requested EM or feedback operation; reading an existing
record never retriggers magnetic-force or wall-projection work.

The base frame owns its original context. A legitimate padded alias frame owns
the exact copied/updated context on which that alias was produced and exposes
it as `alias_frame.owner_context`; checks must use that object. A same-shaped
foreign context is rejected with `Failure("FieldOwnership")` before charges,
forces, projections, or counters are evaluated. The original owner context
remains valid and unchanged after rejection.

The established Warp magnetic-force kernel returns one flat complex128 array of
extent `N*3*Nc`, while the CPU expression returns `(N,3,Nc)`. The real CUDA path
must reshape the kernel output as an owning view of that same array before
adding it to the padded electric force; it must not transfer to host, copy
needlessly, or retain a borrowed temporary. The public `ion_em_terms.magnetic`
contract remains flat `N*3*Nc`, matching `electric`, `projected`, and
`reaction`.
