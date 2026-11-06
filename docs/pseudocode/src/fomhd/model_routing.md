# Massless model routing

## A0-PY-ROUTE-01: Common public entry points

An absent or explicit `electron_inertia=0` selects the default
`massless_quasineutral_cgl_b` branch. Explicit `electron_inertia=1` remains the
finite-mass CGL-B/Darwin branch. Choosing massless never sets the physical
electron mass to zero.

```text
stage.evaluate_stage:
  if electron_inertia == 0:
    return massless.evaluate_stage(state,tag,context,alias)
  otherwise retain the existing finite path unchanged

stage.initialize:
  if electron_inertia == 0:
    return massless.initialize(state,context,cumulative,equilibrium,captured)
  otherwise retain the existing finite path unchanged

stage.retain_failure_evidence:
  if electron_inertia == 0:
    return massless.retain_failure_evidence(...)
  otherwise use the existing finite implementation

core.combine:
  if electron_inertia == 0:
    return massless.combine(base,rhs,weight,tag,context)
  otherwise retain the existing finite combination
```

The massless failure adapter calls `_retain_common_failure_evidence`, not the
public stage dispatcher. This avoids recursion while preserving all prior
checks, residuals, and arrays.

## A0-PY-ROUTE-02: Time transaction

```text
advance:
  validate integrator
  evaluate the base stage through the selected model
  run the existing _check_dt hook before allocating a trial
  if massless and source is not null:
    raise Failure("ManufacturedSource", ...)
  midpoint/rk4 combination uses model-aware combine and canonicalization
  every candidate is canonicalized before its stage is evaluated
  interval validation dispatches to massless_checks.check_step
  retain the existing pending-ledger and one-endpoint commit transaction
  log the existing time_step_report after acceptance
  on guard error, preserve time_step_report in failure JSON and leave the
    committed state and cumulative ledger unchanged
```

Massless electron density and momentum are compatibility outputs only. No
midpoint, RK4, or restart path advances or consumes them as independent state.

The final refreshed physics dependency must expose
`state_totals`, `boundary_ledger`, `momentum`, `check_stage`,
`stage_requirements`, and `check_step`. Routing returns already-computed
`result.totals`, `result.boundary`, and `result.momentum` directly. It does not
rebuild the independently solved wall correction, momentum balance, or
boundary ledger.

## A0-PY-ROUTE-03: Configuration and budgets

```text
load_case:
  resolve omitted electron_inertia to 0 and preserve explicit values {0,1}
  choose the exact finite or massless budget registry by model
  require every registry entry with its specified unit, norm, gate and scale
  model metadata identifies electron_inertia=not_applicable for massless

load_initial:
  retain all required state keys, including placeholder n_e and mom_e
  massless initialize ignores their values as truth and derives canonical
    n_e and compatibility mom_e from independent ion state and B
  reject DB B=0, S J=0, wall constraints, or canonical-state failures
```

The electron collisionless budgets use dimensional density, momentum,
pressure, flux, current, field, energy, and power scales. Theta consistency is
registered as observe-only because its convergence remains a parent MMS task.

## A0-PY-ROUTE-04: Stage, interval, and unavailable records

The massless branch emits its own stage and interval checks. Common finite
Poisson, Gauss, Darwin-Ampere, electron-inertia, electron-kinetic, and
longitudinal-electric-energy concepts are not evaluated.

```text
diagnostics.check_step:
  if electron_inertia == 0:
    return massless_checks.check_step(...)
  otherwise retain the existing finite interval logic

Record unavailable concepts as:
  available=False
  hard=False
  reason="not_applicable: <reason>"
```

Zero energy values may be reported for omitted terms only together with the
not-applicable metadata. They are never presented as passing finite checks.

## A0-PY-ROUTE-05: Provenance, snapshots and restart

```text
model metadata:
  electron_inertia=1 -> model=finite_mass_cgl_b
  electron_inertia=0 -> model=massless_quasineutral_cgl_b
                       electron_inertia=not_applicable
                       electron_kinetic_energy=not_applicable
                       longitudinal_electric_energy=not_applicable
                       finite_charge_gauss=not_applicable

identity/cumulative/run manifest:
  materialize effective electron inertia in semantic identity
  serialize model applicability without converting unavailable to zero

checkpoint:
  encode independent arrays and canonical compatibility fields
  encode model, applicability and cumulative applicability

read_restart:
  read bounded checkpoint metadata
  reject a historical omitted-finite semantic identity before decode/upload
  compare checkpoint model/applicability with configured model
  reject either cross-model direction before decode_state or backend upload
  after acceptance, canonicalize compatibility fields from independent state

snapshot:
  serialize derived_phi only when available; otherwise emit explicit
    availability and reason metadata instead of a null object array
```

CLI initialization accepts both models. The selected model metadata is written
to the run, snapshots and checkpoints, and the existing accepted-state
transaction remains unchanged.

## A0-PY-ROUTE-06: Refreshed physics dependency

The integrated massless dependency is the reviewed R2 source. Species-qualified
massless budget keys use the configured species IDs. A rejected local
massless-combine candidate is retained through the common failure helper before
the canonicalization failure is re-raised; the accepted state and cumulative
ledger are not changed.

## A1-PY-ROUTE-01 - Time coupling dispatch

```text
time.advance:
  validate coupling and micro_steps before any scheduler mutation
  synchronous + micro_steps=1 -> unchanged A0 midpoint/RK4 path for either model
  predictor + electron_inertia=0 + rk4 -> local time_predictor dispatch
  all other combinations -> Failure("Config", exact combination reason)
  predictor source callback on massless path -> Failure("ManufacturedSource")
    before callback invocation, stage evaluation, or trial allocation
```

The massless independent variables remain `P=(ion n,mom,perp)` and
`Z=(electron perp,parallel,B)`. Compatibility electron density/momentum are
reconstructed by existing canonicalization at every assembled stage and at the
accepted endpoint. They are never advanced as independent predictor state.
