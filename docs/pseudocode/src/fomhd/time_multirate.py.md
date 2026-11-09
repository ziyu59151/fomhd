# Massless A1-WR accepted macro transaction

Status: bounded implementation contract for the selected few-node WR scheme.
The generic solver, physical adapter, and mixed-interval ledger are prerequisites;
this module owns acceptance, cumulative commit, and failure rollback.

## Controls and identity

```text
[time]
  coupling = "multirate_wr"
  dt = H
  nsteps = number of macro intervals
  integrator = "rk4"
  micro_steps = M
  ion_nodes = 3 or 5
  waveform_rtol > 0
  waveform_max_iterations positive integer
  wr_initial = "tangent" or "constant"
  0 < wr_relaxation <= 1

Require massless electrons and no manufactured-source callback.
Reject multirate-only controls on every legacy coupling.
```

The semantic hash and checkpoint identity include coupling, `M`, `s`, `rtol`,
iteration cap, initializer, and relaxation. Existing A0/T0/T1 schemas and
checkpoint field sets remain unchanged.

## One macro transaction

```text
advance(committed,cumulative,context):
  validate coupling/controls and clock before allocating a trial
  require source=None
  n = accepted left step; t0 = scheduler(n); t1 = scheduler(n+1); H = context.p.dt
  allocate exactly one trial
  evaluate the actual full base stage once at (n,t0) and apply the micro guard

  adapter = MasslessWRPhysicsAdapter(context,left,trial_id=trial,
                                     slab=0)
  physics = adapter.solve_window(P(left),Z(left),
                                 H,M,s,endpoint_tag_factory=late_endpoint_tag,
                                 initial=wr_initial,
                                 relaxation=wr_relaxation)
  after all coarse/fine callbacks, late_endpoint_tag allocates the next
  generation and returns (n+1,t1,"endpoint",trial)
  require endpoint state step/time/generation and cache tag match that tag

  bundle = multirate_wr_checks.accumulate_mixed_interval(
      physics.window,left,physics.endpoint.full,context)
  interval = bundle.interval
  require interval trial/version/left/right/source identities and endpoint
  quadrature = [(view.state,view,sample.weight)
                for view,sample in zip(bundle.fine_views,
                                       window.fine_samples)]

  step_result = massless_checks.check_step(
      left,base,quadrature[0].state,quadrature[0].stage,
      endpoint.canonical_state,endpoint,cumulative,context,
      quadrature=quadrature,mixed_interval=interval)
  gate base, accepted coarse, accepted fine, endpoint, and interval checks
  reject a duplicate cumulative trial
  pending = time._pending_cumulative(
      cumulative,quadrature,endpoint.canonical_state,trial,
      mixed_interval=interval)

  build all fallible coupling metadata and diagnostic-work attribution now
  metadata includes:
    delta_D and interval kinetic_path
    epsilon_K=(endpoint ion kinetic-base ion kinetic)-kinetic_path
    bundle.diagnostic_work and result.diagnostic_work
    copied actual residual_history records, final normalized maximum/raw norm,
    and endpoint residual scalar summary when available
    full base/coarse/endpoint attempts, successes, failures, and returned
    full ion/electron power-ledger values
    fine_ion_power_ledger_calls=0

  synchronize
  time._commit(committed,cumulative,pending,
               endpoint.canonical_state,endpoint.cache,step_result)
  set result.frame/committed.frame=endpoint
  mark endpoint production_accepted=true, trial_endpoint=false,
       interval_transaction_required=false only after commit
  assign the prebuilt coupling_metadata and return one accepted interval
```

Metadata/report/dataclass construction is fallible and therefore occurs before
`_commit`. After commit, only nonfallible container assignments and accepted
flags occur. The declared `context.p.dt=H` is retained for quadrature and clock
metadata; neighbor-time subtraction is never the interval authority.

The adapter's accepted-left callback tags remain `(n,t0)`. The endpoint
generation is allocated after all macro stages so it is the final generation
and restart cannot reuse staged generations. Only the full
endpoint is rebuilt at `(n+1,t1)`, with a cache tag matching the accepted
state. Candidate step/time are never patched after evaluation.

## Evidence and failure

```text
coupling_metadata:
  scheme controls, macro/endpoint clock, accepted node version, sweeps,
  exact producer counts, one shared trial/source IDs, mixed interval,
  density work coarse/fine and delta, kinetic path, no production rollback claim

failure:
  retain current actual base/adapter/coarse/fine/endpoint records,
  mixed bundle if available, attempted tag, operational counters, and arrays
  committed.state/cache/acceptance and cumulative ledger remain unchanged

base failure:
  metadata uses None for unavailable source identity
  preserve original exception code/message/stage/checks/partial arrays
  base attempt=1, success=0, failure=1; no full ledger value is claimed
  residual_history and final residual fields are explicit None, never a fake
  empty converged history

partial solver failure:
  copy completed actual residual_history records from Failure evidence
  report the completed final scalar metrics available at the failure boundary

later transaction failure:
  preserve actual residual_history already copied from the solved window
  do not replace available history with None merely because the later
  Failure has no solver history attribute
```

No fine full-ion stage or canonicalization path is evaluated for interval
diagnostics. Full ion/electron ledger values come from charged base/coarse/
endpoint producers; the fine producer path reports zero ion power-ledger calls.
