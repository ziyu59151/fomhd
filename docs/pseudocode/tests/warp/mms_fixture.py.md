# Discrete coupled MMS and smooth spatial verification

Stable step IDs: `A0-VERIFY-COUPLED` and `A0-VERIFY-SPATIAL-SMOOTH`.

## A0-PY-MMS-VERIFY-01 Temporal fixture

```text
Context:
  p = foundation_fixture.parameters(K=1)
  p.Nphi = 9
  p.dt_guard = "warn"
  geometry = rectangle(cells=4 by 4, R=1, Z=0, dR=dZ=0.2 m)
  production operators and selected CPU/CUDA backend

Trajectory.at(tag, derivative=false):
  u = closed equilibrium reference when derivative=false
  u = zero-owned rate state when derivative=true
  set u.step, u.generation, u.time from tag
  t = tag.time / 1e-8 s
  wave(f) = sin(f*t), or derivative=true: f*cos(f*t)/1e-8 s
  for each species and each K+/-1 density entity:
    envelope = .6 + .2*cos(R)*cos(Z)
    n      = .01 * electron_density_factor * reference_ni * envelope * wave(1)
    mom_y  = .5 * electron_momentum_factor * mass * reference_ni * .1 m/s
             * envelope * wave(1.2)
    perp   = .015 * species_pressure * envelope * wave(.7)
    electron parallel = .011 * electron_pressure * envelope * wave(.7)
  B = equilibrium_B * ((derivative ? 0 : 1) + .01*wave(.8))

ManufacturedSource(actual, tag, context):
  reference = Trajectory.at(tag)
  physical = production evaluate_stage(reference, tag, context)
  exact_dot = Trajectory.at(tag, derivative=true)
  source = exact_dot - physical.rhs
  record source independently; never derive source from `actual`
  return source
```

The source is therefore a prescribed test term, not feedback cancellation.
`actual` is used only later by the production interval diagnostic to evaluate
the numerical midpoint response.

## A0-PY-MMS-VERIFY-02 Temporal advancement and acceptance

```text
RunLevel(integrator, steps):
  dt = 1e-8 s / steps
  initial = exact trajectory at t=start_time
  production initialize(initial, equilibrium=true)
  retain a separate harness cold pre-estimate for context
  for step in 1..steps:
    result = production time.advance(committed, cumulative, context, source)
    require result.time_step_report is the real base-stage production report
    retain its complete report and base check tag
    require one source evaluation per derivative stage
    for each interval quadrature entry:
      evaluate production source_rates(source, numerical_state,
                                       physical_stage_at_numerical_state)
      integrate rate * declared quadrature weight exactly once
    require every hard stage and interval check passed
  compare every component to Trajectory.at(final tag)
  derive runtime_integrated from the retained production reports
  serialize every tagged production check to checks.jsonl, including
  raw/reference/threshold hex, availability, hardness, pass and reason
  retain physical cumulative integrals and source rate-time products
  on Failure, serialize Failure.checks and Failure.partial_arrays to sidecars

normalized component errors:
  n_i, n_e       divide by .02*reference_ni
  mom_i, mom_e   divide by .1*mass*reference_ni
  perp_i,e       divide by .03*species_pressure
  parallel_e     divide by .03*electron_pressure
  B              divide by .01*reference_B
  component error = L2(actual-exact)/(scale*sqrt(component_size))
  combined = hypot(all component errors)
```

Midpoint uses 8, 16 and 32 steps. RK4 uses 2, 4 and 8 steps so every
component remains above the `1e-11` roundoff gate while the observed successive
orders still measure fourth order. The production cumulative ledger retains
physical terms only; manufactured source integrals are reported separately.
The omitted-ion-pressure negative run is expected to pass physical interval
identities while failing the analytic trajectory by a large margin.
It retains all eight actual production time_step_report objects, not merely a
count or its raw time_step_estimate checks.

Raw sidecar aggregation counts the six normal level files and the one negative
level file exactly once: 73,820 + 7,592 = 81,412 records across seven files.

## A0-PY-MMS-VERIFY-03 Smooth spatial verification

```text
For N in {8,16}:
  rectangle cells=N by N on R=[1,2], Z=[0,1]
  build_spatial_operators(mode=+2)
  use compact exp(-1/(1-s^2)) bump centered at (1.5,.5), width=.42 m
  evaluate independent cylindrical analytic G, curl, divergence and DJ
  account R-face/cell/Z-face/vertex wall flags independently
  report inner error on R=[1.25,1.75], Z=[.25,.75]
  report wall error and total error with their own normalizations
  require C*G and DB*C residuals below registered roundoff bounds
  require positive inner residuals and convergence order greater than 3.5
  deliberately reverse the vertex curl term and require inner error >.1
```

No wall or global fourth-order claim is inferred from the interior order.

## A0-PY-MMS-VERIFY-04 CLI persistence

```text
argparse(device, output, integrator, no_spatial)
require device=cpu or device=cuda:N with existing SLURM_JOB_ID
create output only if it does not exist or is an empty directory
derive runtime ownership from the imported fomhd package/module files
require an explicit runtime root, when supplied, to match that checkout
hash every module in the imported runtime, its pyproject.toml/uv.lock, and
the actual executed harness files
archive that bounded source snapshot with its own destination hashes
run requested production MMS levels with synchronized timing
run the small spatial convergence scan only for device=cpu
write raw checks and ledgers as sidecars instead of discarding reductions
write Failure.partial_arrays to NPZ when supplied, preserving NaN and infinity
write Failure.checks to JSONL when supplied
write strict JSON through exclusive create
retain FAIL status, exception traceback and any completed level evidence
```

For a CUDA run, the temporal MMS levels use the requested device and the
spatial scan is not silently executed on CPU. A separate `--device cpu`
invocation provides the small spatial evidence.
