# Runtime diagnostics

## A0-PY-DIAG-00: Model dispatch

For `electron_inertia=0`, the public `check_stage`, `check_step`,
`state_totals`, `boundary_ledger`, `momentum` and `stage_requirements` entry
points dispatch to `massless_checks.py`. Those records explicitly mark omitted
finite-charge, electron-inertia, electron-kinetic and longitudinal-electric
concepts unavailable and observe-only. The finite path below is unchanged.

If a massless stage result already carries `totals`, `boundary`, or `momentum`,
the common entry point returns that object unchanged. Only an uncached request
calls the matching public helper from the refreshed `massless_checks.py`.

## A0-PY-DIAG-01 - Budget and record primitives

```text
budget(context,key):
  require key exists in context.p.budgets
  return immutable Budget otherwise Failure("MissingBudget")

measure(out,name,residual,weights,unit,tag,context,population="",
        samples=false,component=""):
  if wall residual is empty and no wall entities/rows:
    require six wall budgets with matching units and norms
    append UNAVAILABLE, observe-only records for all/inner/wall and both norms
    return
  require residual and weights nonempty and residual.size divisible by weight.size
  derive the entity-space wall distance or use exact local adapter distances
  require every weight finite and positive
  for region in all,inner,wall; norm in linf,weighted_L2:
    require budget key name[.population].norm with exact unit and norm
    select active residual entries whose location region matches
    nonfinite selected entry => UNAVAILABLE with worst index
    no selected entry => UNAVAILABLE, observe-only, reason="empty region"
    linf raw = device max(abs(residual))
    weighted_L2 raw = sqrt(angular * sum(weight*abs(residual)^2))
    weighted reference uses the same region, weight, mode and angular measure
    threshold = atol + rtol*reference
    append host scalar record; no residual array is copied to host

scalar_check(out,name,raw,unit,tag,context,population=""):
  require matching value budget
  raw is available only when finite
  threshold = atol+rtol*reference
```

## A0-PY-DIAG-02 - Fixed registry and gates

```text
stage_requirements(context,tag):
  emit finite, hermitian_rhs, source_identity, memory_bound, minimum_B (value)
  emit each global residual at all/inner/wall x linf/weighted_L2
  for ion and electron emit domain minima and all local records
  emit electron-only parallel minimum, CGL source and parallel wall flux

gate(records,required=[]):
  reject duplicate full identities
  reject every hard record that is unavailable or failed
  attach all records to Failure("HardGate")
  after gate, require every named record in required

check_stage(state,result,context):
  check state/result finite and Hermitian using backend reductions
  check stage/source ownership, memory cap, B/n/face-n/pressure floors
  evaluate Poisson, Gauss, transverse divergence, divB, Ampere and CT-incidence
  per species evaluate continuity, full wall rows, actual wall fluxes, power
    pairings, theta identity and electron CGL source identity
  evaluate total current response using the stage response matrix
  call gate with the fixed stage requirement registry
```

## A0-PY-DIAG-03 - Totals, boundary and momentum

```text
state_totals(state,result,context):
  for every sample/mode and active cell reduce:
    number[a] += 2*pi*M0/N * Re(n_a)
    kinetic[a] += 2*pi*M0/N * |pi_a|^2/(2*m_a*n_a)
  ion internal = p_i/(gamma_i-1)
  electron internal = p_perp + p_parallel/2
  magnetic energy uses MB, longitudinal energy uses M1*epsilon0/2
  total excludes transverse electric energy
  topology=ell*B[k=0], physical flux=sum(av*Bphi[k=0])

boundary_ledger(state,result,context):
  recover physical B/E samples
  for every phi sample and wall:
    interpolate prescribed normal-B/tangential-E traces
    recover density, pressure, velocity and actual normal flux
    accumulate actual number, kinetic, internal and field energy flux

momentum(state,result,context):
  use max(N,3) diagnostic angle samples
  total R/phi/Z momentum includes ion+electron+epsilon0*EL cross BN
  convert to Cartesian samples and integrate M0
  include wall reaction without counting it as physical outward flux
  integrate wall magnetic/kinetic/pressure/electrostatic traction and z torque
```

## A0-PY-DIAG-04 - Stage and step gates

```text
check_step(left,base,mid_state,mid,right,end,cumulative,context,
           prescribed_source=None,quadrature=None):
  normalize quadrature to [(evaluated_state,evaluated_stage,weight)]
  default for midpoint is [(mid_state,mid,dt)]
  require stage tags and endpoint times match the scheduler
  form integrated number flux/current from weighted stage fluxes
  test per-species continuity, charge continuity, RC_flux, RC_mom and deltaC
  test longitudinal current displacement, CT update, current divB,
    incremental and baseline divB, topology increment
  integrate ledger, boundary, momentum and field-power rates by the same
    quadrature weights; never call base/endpoint quadrature extra work
  test number, charge, species kinetic, internal, magnetic, longitudinal,
    field exchange, total energy, Cartesian momentum and angular momentum
  gate every fixed step requirement

Field exchange uses the unchanged W budget. The quadrature sum is an interval
energy in J, so divide it by dt before recording `field_exchange` as average
power in W. A small-dt injected-power regression must fail the W gate.

For RK4 quadrature weights are dt/6, dt/3, dt/3, dt/6 for k1..k4.
The base case also supplies the first evaluated stage but is not added twice.

The source-aware branch validates rate-layout Hermitian/finite arrays, full wall
rows and divergence-preserving B before use. Interval source rates use the same
quadrature weights; midpoint passes only its midpoint source to interval checks.

Totals, norm selection and boundary/momentum work remain backend operations.
Per-cell values are never copied to host; only final reductions and explicit
wall/trace record scalars cross the device boundary.

All physical-state sample recovery defaults to `physical=true`, matching the
native inverse contract. Complex linear-algebra paths must request
`physical=false` explicitly; diagnostics never omit the imaginary-residue check
by relying on an implicit default.
```
