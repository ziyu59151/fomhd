# Time integration and acceptance transaction

## A0-PY-TIME-00 - Model-aware stages

The midpoint and RK4 transactions are shared. `evaluate_stage` and
`core.combine` dispatch by `electron_inertia`; massless candidates are
canonicalized before evaluation. `diagnostics.check_step` dispatches interval
checks to `massless_checks.check_step`. A non-null source callback on the
massless branch raises `ManufacturedSource` before the callback or trial is
consumed. `_check_dt` remains at the finite runtime hook point before any
candidate allocation, and `time_step_report` remains attached to accepted and
rejected results.

## A0-PY-TIME-01 - Midpoint

```text
advance(committed,cumulative,context,source=None):
  require committed.accepted and step<nsteps
  require committed/time/cumulative identities agree with scheduler
  trial=fresh(context.next_trial)
  base_tag=(step,generation,trial,t,base)
  mid_tag=(step,generation+1,trial,t_mid,midpoint)
  end_tag=(step+1,generation+2,trial,t_end,endpoint)
  base=evaluate_stage(left,base_tag,context,false)
  report=assess_runtime(stage_frequency_estimate(base,context),context,step)
  append the scalar report to checks and the run log
  if report fails and dt_guard=error: retain base evidence and reject
  mid_rhs=base.rhs + optional source(left,base_tag)
  midpoint=combine(left,mid_rhs,dt/2,mid_tag,context)
  mid=evaluate_stage(midpoint,mid_tag,context,true)
  candidate=combine(left,mid.rhs+optional source(midpoint,mid_tag),dt,end_tag)
  end=evaluate_stage(candidate,end_tag,context,false)
  quadrature=[(midpoint,mid,dt)]
  interval=interval_sources(left,candidate,quadrature,trial)
  result=check_step(... quadrature=quadrature)
  gate(base/mid/end records plus interval result)
  pending=clone(cumulative)
  add weighted ledger, boundary and endpoint identity exactly once
  synchronize backend
  finalize by replacing committed state/cache and replacing cumulative
  return accepted result

Failure before finalize:
  retain every available base/mid/end state, cache, check and residual
  for RK4, retain each derivative stage immediately after evaluation:
    k1/base, k2/midpoint, k3/midpoint_2, k4/station, then endpoint
  retain source callback payloads separately
  mark requirement phases not reached NOT_RUN
  preserve committed/cumulative unchanged
```

## A0-PY-TIME-02 - Default explicit RK4

```text
advance_rk4:
  k1_tag=(n,g1,trial,t,base)
  k2_tag=(n,g2,trial,t_mid,midpoint)
  k3_tag=(n,g3,trial,t_mid,midpoint_2)
  k4_tag=(n,g4,trial,t_end,station)
  k1=evaluate_stage(left,k1_tag)
  y2=combine(left,k1.rhs+source(left),dt/2,k2_tag)
  k2=evaluate_stage(y2,k2_tag,alias=true)
  y3=combine(left,k2.rhs+source(y2),dt/2,k3_tag)
  k3=evaluate_stage(y3,k3_tag,alias=true)
  y4=combine(left,k3.rhs+source(y3),dt,k4_tag)
  k4=evaluate_stage(y4,k4_tag)
  rhs4=(k1.rhs+2*k2.rhs+2*k3.rhs+k4.rhs)/6
  candidate=combine(left,rhs4,dt,end_tag)
  end=evaluate_stage(candidate,end_tag)
  quadrature=[(left,k1,dt/6),(y2,k2,dt/3),
              (y3,k3,dt/3),(y4,k4,dt/6)]
  weighted number flux/current/ledger/boundary/momentum are the only interval
    and cumulative contributions
  run the same check_step and one endpoint commit as midpoint

No implicit electron solve is used. Default integrator is RK4; midpoint runs
only when requested explicitly by configuration, a test, or a historical tool.
```

## A0-PY-TIME-03 - Failure evidence and counter safety

```text
fresh(counter):
  reject UINT64_MAX before increment

append_missing(target,source):
  compare full record identity including tag and all classification fields
  append only records not already present

retain_failure(failure,...):
  attach earliest available stage tag
  add completed stage checks
  copy residuals/E and state arrays under base/midpoint/endpoint/source prefixes
  add unavailable NOT_RUN records for expected stages
```

`time.dt_guard` values and the three guard limits are parsed by runtime I/O, but
frequency assessment and the guard hook are parent-owned in
`src/fomhd/stability.py`.
