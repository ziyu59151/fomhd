# A1 T0 predictor time transaction

This module is the bounded A1-T0 scheduler. It does not implement T1
waveform iteration, sparse operators, slabs, or a generic solver framework.
It reuses the A0 stage evaluator, model-aware state canonicalization,
diagnostics, cumulative-ledger helper, failure-retention helper, transaction,
and backend synchronization.

## A1-PY-PREDICTOR-01 - Clocks and stage construction

```text
advance(committed,cumulative,context):
  require accepted state and scheduler/cumulative endpoint agreement
  preflight the complete predictor history before trial allocation
  validate every integer micro/RK clock for j=0..M-1 before changing counters
  H=context.p.dt; M=context.p.micro_steps; h=H/M
  t0=core.time_at_step(n)
  t1=core.time_at_step(n+1)
  allocate one trial/source identity; base generation is the committed generation
```

For micro index `j`, one-based RK stage `r=1..4`, and integer stage offsets
`(c_1,c_2,c_3,c_4)=(0,2,2,4)` in quarter-micro units:

```text
tau = H * ((4*j+c[r])/(4*M))
t_stage = core.microstage_time(n,j,r)
P_hat(t_stage) = P0 + tau*F0
```

The stage time and tangent use this same fraction. No micro time is formed by
repeated addition of `h`. The Python helper receives zero-based
`rk_stage=r-1`; stage tags and this mirror retain the one-based `r` in their
stage strings.

## A1-PY-PREDICTOR-01A - Candidate assembly failure boundary

```text
assemble_predictor_state(P,Z,tag):
  build raw candidate with exact step,generation,trial,spectral context,
    microstep, iteration, slab, and time copied from tag
  try canonical=canonicalize_state(raw)
  except Failure as failure:
    if failure.stage is null: failure.stage=tag
    retain_failure_evidence(failure,tag,context,raw,None,"candidate")
    raise the original failure
  except unexpected library error:
    failure=Failure("NumericalLibrary",message); failure.stage=tag
    retain_failure_evidence(failure,tag,context,raw,None,"candidate")
    raise failure chained from the library error
  stamp and return canonical
```

The outer transaction retains scheduled/completed-stage evidence but preserves
an already attached assembly tag. It uses an endpoint or base fallback only
when no inner boundary supplied a failure tag. Raw candidate B, ion n/momentum/
perp, electron perp/parallel, and compatibility fields remain evidence-only and
are never accepted.

## A1-PY-PREDICTOR-02 - One macro interval

```text
base=evaluate_stage(committed.state, base_tag, context, false)
required_full_stage_evaluations += 1
F0=base.rhs.fluid[0]
run the predictor-only micro-RK4 guard on base
for j in [0,M):
  for r in 1..4:
    if j==0 and r==1:
      stage=base
      state=P_hat(t_j) with Z0
    else:
      r==1: Zstage=Z_j
      r==2: Zstage=Z_j+(h/2)*G1
      r==3: Zstage=Z_j+(h/2)*G2
      r==4: Zstage=Z_j+h*G3
      alias=(r in (2,3)); pass alias only to evaluate_stage
      state=assemble_predictor_state(P_hat(t_stage),Zstage,tag,context)
      stage=evaluate_stage(state,tag,context,alias)
      required_full_stage_evaluations += 1
    append (state,stage,h*b[r]) to quadrature
    accumulate ion n/mom/perp with weight h*b[r]
  Z_{j+1}=Z_j+h*(G1+2G2+2G3+G4)/6
  k1 of the next microstep is newly evaluated on Z_{j+1}; RK4 is not FSAL
P_end=P0+sum(weights*F_stage)
candidate=assemble_predictor_state(P_end,Z_M,endpoint_tag,context)
end=evaluate_stage(candidate,endpoint_tag,context,false)
```

`b=(1,2,2,1)/6`. Every stage evaluates the complete existing model RHS. The
accepted ion endpoint consumes its three independent fields (density,
momentum, and perpendicular pressure) from all micro-stage histories; there is
no extra `H*F0` impulse. The tangent path is not the accepted ion endpoint.

The first microstep k1 reuses the complete base evaluation. The subsequent
microstep k1 is a new full evaluation because RK4 is not FSAL. `k2` and `k3`
share a clock and primary spectral context by design, but not state,
generation, source identity, or stage string.

## A1-PY-PREDICTOR-03 - Diagnostics, metadata and one commit

```text
quadrature has exactly 4*M entries, each stage used once
evidence slots are indexed once by unique prefix; retention keeps list order
interval=diagnostics.check_step(left,base,first_state,first_stage,
                                candidate,end,cumulative,context,
                                quadrature=quadrature)
gate(base, all micro stages, endpoint, interval checks)
pending=time._pending_cumulative(cumulative,quadrature,candidate,trial)
compute endpoint tangent correction and all accepted metadata
context.compute.synchronize()  # final explicit synchronization
time._commit(committed,cumulative,pending,candidate,end.cache,interval)
attach endpoint and full stage/resource metadata to accepted result
```

All shared flux, current, power, boundary, momentum, and ledger diagnostics
consume the same `(state,stage,h*b)` list. Diagnostics remain read-only.

Result metadata reports:

```text
coupling="predictor"
micro_steps=M
micro_dt=h
quadrature_weights=(1,2,2,1)/6
full_rhs_stage_evaluations=4*M
endpoint_full_evaluations=1
complete_stage_evaluations=4*M+1
alias_evaluations=2*M (one extra model evaluation per k2/k3)
total_model_evaluations=6*M+1
assembly canonicalization calls are reported separately from model evaluations
endpoint_tangent_correction[field]:
  raw=max_abs(P_end-P0-H*F0)
  scale=max(max_abs(P0+H*F0),physical_reference_field,1e-300)
  relative=raw/scale
```

The correction values are measurements, not convergence or equivalence
claims.

## A1-PY-PREDICTOR-04 - Failure transaction

```text
on any stage, endpoint, interval/check, pending-ledger, or synchronize failure:
  retain every completed state/stage with prefix predictor_jjj_rR
  retain every scheduled but not reached slot as UNAVAILABLE/NOT_RUN
  attach the earliest completed nonzero step and micro identity
  leave committed state/cache and cumulative ledgers unchanged
  do not retry, clip, reduce, project, or fall back to synchronous RK4
```

All final metadata and endpoint tangent-correction computations complete before
the one explicit backend synchronize. Commit immediately follows that
synchronization.

Trial/source counters may have advanced as in A0. Intermediate microsteps are
evidence only and are never accepted restart or checkpoint states.

## A1-PY-PREDICTOR-05 - Micro-only time guard

The predictor guard uses `h=H/M` and the physical remaining horizon
`(nsteps-n)*H=(nsteps-n)*M*h`. It calls the existing explicit RK4 estimate with
`micro_horizon_steps=(nsteps-n)*M`. The report and check are labeled
`micro-RK4 only`; they do not establish macro-coupling stability. Existing
`dt_guard=error|warn` behavior, limits, logging, and failure preservation are
reused.
