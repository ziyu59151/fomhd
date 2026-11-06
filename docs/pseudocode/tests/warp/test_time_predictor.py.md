# A1 T0 predictor tests

Source: `tests/warp/test_time_predictor.py`.

## A1-PY-TEST-PREDICTOR-01 - Algorithmic ODE scheduler

Toy ODE fixtures isolate the time scheduler and are labeled algorithmic tests,
not physical validation.

```text
fast oscillatory ion forcing over an integer cycle:
  endpoint ion impulse tends to zero as M increases
  compare against the nonzero endpoint-force shortcut

two-variable feedback F(P,Z), G(P,Z):
  record the fixed tangent P_hat at every stage
  distinguish same-time k2/k3 state, generation, source, and stage identity
  verify all ion n/momentum/perp stage increments
  verify Z uses classical RK4 h*(1,2,2,1)/6
  M=1 differs from an independently computed synchronous RK4 full-state step

transaction:
  4M full RHS stages, first k1 aliases base, next k1 does not
  4M quadrature entries use h*(1,2,2,1)/6 and one ledger commit
  late micro, endpoint, and synchronize failures retain NOT_RUN evidence
  intermediate/endpoint assembly canonicalization failures retain the exact
    candidate tag and raw B/independent fields under the candidate prefix
  an unexpected endpoint canonicalization library error is wrapped and retains
    the same endpoint candidate/tag evidence
  accepted state/cache/cumulative ledger remain unchanged for assembly failure
```

## A1-PY-TEST-PREDICTOR-02 - Guards, configuration, memory, clocks

```text
guard uses h=H/M and (nsteps-step)*M but labels micro-RK4 scope
invalid configuration/source is rejected before counters or state mutate
peak estimate and state/workspace cap grow conservatively with M
massless fixed-wall bound grows with q and w, contributes to preflight, and
  is zero for finite inertia; missing wall lists use w=0
retained predictor wall samples grow with M and w; dense per-stage S/LU storage
  is distinct from the once-counted shared fixed wall-cache bound
macro endpoints use time_at_step; micro clocks use integer-index fractions
unrepresentable/equal stage clocks are rejected before advance
```

## A1-PY-TEST-PREDICTOR-03 - Real massless fixture

```text
construct the existing small closed massless context
run a current-free equilibrium and a nontrivial compatible-field perturbation
require canonical endpoint electron density/momentum
require interval species continuity and CT hard checks to pass with 4M
  retained stages and distinct stage identities
```

Passing these checks is evidence for the stated tiny CPU fixture only. It does
not establish macro-coupling stability, global Tokamak accuracy, GPU behavior,
or performance.
