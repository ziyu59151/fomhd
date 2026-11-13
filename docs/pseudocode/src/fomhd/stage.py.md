# Stage implementation pseudocode

Uniform reference field normalization uses the same physical magnetic
reconstruction as stage evaluation.

## A0-PY-STAGE-00: Model dispatch

`electron_inertia=0` dispatches `evaluate_stage` and `initialize` to
`massless.py`. It dispatches `retain_failure_evidence` to the massless wrapper,
which reuses `_retain_common_failure_evidence` without recursive public
dispatch. An absent value is read as `0`. The explicit `electron_inertia=1`
path below is unchanged.

## A0-PY-STAGE-01: StageResult ownership

`evaluate_stage` returns a `SimpleNamespace` with every field consumed by the
time and diagnostic layers initialized, including `tag`, `rhs`, `cache`, `BN`,
`bhat`, `rho`, `current`, `fJ0`, `ELdot`, `phidot`, `ampere`, `AE`, two
`fluid` records, two `terms`, two `ledger` records, `source_id`, `residuals`,
and diagnostic-owned `checks`, `boundary`, `momentum`, and `totals`. Arrays are
owned by the result until returned; fluid contexts borrow no temporary sample
array after the call.

```text
EvaluateStage(state,tag,context,alias=false):
  construct fully initialized StageResult with source_id next unique value
  require tag.step/generation/time exactly equal state identity
  CheckRetained(state)
  BN=inverse(sample_apply(QB,state.B,modes),physical_real)
  bhat=BN/|BN|; require |BN|>B_min and bhat normalized
  rho=0, current=0, fJ0=0
  for species 0,1:
    fluid[species]=BuildFluid(...)
    BuildNonElectric(fluid[species],...)
    projected=ProjectWall(fluid.wall,fluid.non_electric)
    fJ0+=Forward(I*(q/m)*projected)
    rho+=q*state.fluid[species].n
    current+=q*fluid.flux_hat
  cache=Electrostatic(rho,tag,context)
  Transverse(result,context)             # produces materialized AE and E
  for species:
    FinishFluid(result,species,context)
    ledger[species]=PowerLedger(result,species,context)
  ELdot=-Longitudinal(current)/epsilon0
  phidot=solve every Lphi row for -DJ*current/epsilon0
  ampere=transpose(C)*state.B-mu0*(current-longitudinal(current))
  diagnostics module, imported locally:
    result.boundary=boundary_ledger(state,result,context)
    result.totals=state_totals(state,result,context)
    result.momentum=momentum(state,result,context)
    check_stage(state,result,context)  # mutates checks, raises on gates
  if alias:
    evaluate alternate N=2*N+1 with new spectral id and alias stage tag
    append alias checks and compare all fluid and B RHS components
    gate alias checks
  on Failure:
    retain_failure_evidence; re-raise
  on backend numerical error:
    wrap as Failure("NumericalLibrary"), retain evidence, re-raise
  return result
```

Diagnostics imports are local to avoid a `diagnostics -> physics -> stage`
cycle. All stage arrays remain on the selected backend. No CPU fallback is
introduced.

## A0-PY-STAGE-02: Failure evidence

```text
retain_failure_evidence(failure,tag,context,state,result,prefix):
  set failure.stage if absent
  append missing completed checks by full diagnostic identity
  copy existing residuals and E into failure.partial_arrays
  copy state B,n,mom,perp,parallel with unique prefix keys
  append required stage checks with available=False,passed=False,
    reason="not run: <failure code>"
```

Partial arrays are evidence only and never become a committed state.

## A0-PY-STAGE-03: Initialize

```text
initialize(state,context,cumulative,equilibrium,captured=None):
  require generation counter not exhausted
  advance next_generation to at least state.generation+1
  tag=(state.step,state.generation,next_trial,spectral_id,state.time,
       restart if step>0 else initial)
  stage=EvaluateStage(state,tag,alias=true)
  append setup checks converted to diagnostic records
  check initial divB and Ampere using the original matrices
  if equilibrium:
    gate ion/electron continuity,momentum,both pressures,B RHS
  always gate initial divB and Ampere
  set cumulative baseline,last_step,last_time
  return Committed(state,cache,accepted=true)
  on Failure:
    retain code/tag/checks/partial arrays, do not mutate cumulative
```

The diagnostic implementation behind initial gates is dependency-owned; this
stage records the required checks through that API.

## A0-PY-STAGE-04: ClosedReference

```text
closed_reference(context,n_i,p_i,p_e,B_reference,velocity=0):
  require finite n_i>0,p_i>0,p_e>0,B_reference>0,finite(velocity)
  allocate state
  electron number density=n_i*abs(q_i/q_e)  # exact native charge symmetry
  set zero-mode pressures:
    ion perp=p_i
    electron perp=parallel=p_e
  if velocity and K>0:
    set +1 and -1 toroidal momentum modes with amplitude
      mass*n*velocity/2
  construct harmonic B from transpose(topology_flux_row)/MB
  scale it so QB*B has the requested maximum magnitude
  state.time=start_time,generation=1,step=0
  return owned state
```

No state array may alias the helper harmonic buffer. Negative/nonfinite
reference inputs raise `Failure("Fixture", ...)` before allocation arithmetic.

## A0-PY-STAGE-05 - Verification comparison contract

Native-oracle comparisons are dimensional and elementwise:

```text
require all(isfinite(actual)) and all(isfinite(expected))
allowed_i = 64*eps*abs(expected_i) + roundoff_factor*eps*reference
require abs(actual_i-expected_i) <= allowed_i
```

`reference` is the native-unit term scale, not `1` and not a unitless floor.
For a non-cancelling array its own maximum magnitude is the initial reference.
For a zero or cancellation-sensitive array, callers supply the physical
reference (pressure rate, momentum rate, power term sum, source-current term, or
field-solve source scale). Cancellation-sensitive comparisons are explicit and
separate from ordinary relative comparisons.

The test suite mutates small nonzero pressure, momentum-rate, and power arrays,
including the former three-component kinetic-density error, and requires each
mutation to fail. The stress oracle uses zero velocity, inhomogeneous density,
anisotropic electron pressure, nonzero transverse forcing, and an independent
dense KKT check.
