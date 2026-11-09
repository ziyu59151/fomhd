# Mixed WR interval ledger tests

## A1-PY-TEST-WR-LEDGER-01 - Frozen values and validation

```text
Real successful periodic s5 window:
  accumulate only final_input_version coarse/fine samples
  require exactly 12 ledger keys and 6 boundary keys
  require coarse and fine weight sums independently equal H
  require five exact SourceKeys and 4*M exact fine SourceKeys
  mutate a wrong trial, accepted-node version, source id, or tag copy:
    require Failure before any host ledger is returned
  inject NaN/inf/negative quadrature weight:
    require Failure and no cumulative consumption
```

## A1-PY-TEST-WR-LEDGER-02 - Density split and kinetic accounting

```text
Use a real s5 periodic window with nonuniform density/flux:
  compute D_s and D_hat from actual ion flux divergence
  require density_work_coarse != density_work_fine
  require delta_D = density_work_coarse-density_work_fine
  require the ledger convection explicitly includes Qf[sD_s]
  require K_path =
    electric_node+pressure_kinetic+wall+magnetic
    -convection+spectral_kinetic
    within a scale-aware arithmetic bound

Remove one required density/coupling term or add it twice in a controlled
real-integrator input; require the K-path or continuity diagnostic to reject.

Compute ion compression work directly as Qc[-integral(p_i theta_i)]. Compute
electron pressure internal work directly from Qf of the retained perpendicular
pressure rate plus half the retained parallel pressure rate. Require the two
species ledger entries to preserve those distinct appendix meanings.
```

## A1-PY-TEST-WR-LEDGER-03 - Real checker integration

```text
Small periodic and closed-wall K=1 windows:
  pass actual fine FieldFrame stages and full left/endpoint records to
    massless_checks.check_step(mixed_interval=...)
  require species continuity, CT, field exchange and total energy use the
    same raw formulas as the legacy checker
  require actual fine views have no ledger/full-ion RHS/boundary/momentum
  spy species-0 build_non_electric and require no fine transport

Pending:
  add one frozen interval to a deep-copied cumulative
  require each of 12+6 fields changes once, with no weight factor
  require exact accepted interval identity update
  reject a pending transaction whose trial differs from the interval source trial
  calling None path gives the original weighted quadrature result
```

## A1-PY-TEST-WR-LEDGER-04 - Fine work gates and diagnostic cost

```text
Build one real complete K=1 stage at the same fine P/Z as the final sample.
After the oracle is built, monkeypatch species-0 build_non_electric to reject
all calls.

Run mixed check_step with actual fine views:
  require all eleven electron/wall work records
  compare raw, reference, threshold, hard, available, passed, configured and
    producer arithmetic bound, sample/modal work and transform evidence
    exactly with the complete full-stage records
  retain wall producer arithmetic bounds

Require MixedIntervalBundle.diagnostic_work to count accumulator ledger and
pressure-only projection operations. Require check-result diagnostic_work to
count ledger calls, pressure projections, inverse transforms, wall metrics,
mode-envelope sets, gate calls, gate transforms and emitted records. Preserve
the same counts on a physical rejection after these checks.

Keep nonzero-D mixed identity and default None/check_stage behavior unchanged.
```

## A1-PY-TEST-WR-LEDGER-05 - Role-specific wall traction

```text
Constant-traction oracle:
  use independent nonzero ion ram/pressure and electron-pressure/Maxwell
    components with a nonzero wall B_phi
  hand-integrate cylindrical rotation, samples=max(N,3), radius torque
  require transport role equals only T_A
  require electromagnetic role equals only T_C
  require the supplied projection reaction to be identical across roles
  require the old combined result to differ

Real nonuniform closed-wall K=1:
  evaluate one actual full producer at nonzero nonuniform P/Z
  compare transport and electromagnetic roles with independently selected
    wall traces
  require both contributions measurable and the duplicated sum to differ
  retain earlier electron-work and nonzero-D interval regressions
```

## A1-PY-TEST-WR-LEDGER-06 - Scheduler clock identity

```text
Use start_time=1, H=2e-12, and synthetic macro steps 23->24:
  require window.H exactly equals context.p.dt
  build coarse/fine metadata weights that independently sum H
  validate left, interval and right clocks with core.time_at_step
  assert (t_right-t_left)-H exceeds the rejected old relative threshold
  accept only a 16-eps absolute-clock identity allowance
  reject a spoofed right clock by 1e-10
  reject window H when it differs from declared context H

Mixed check_step keeps dt=context.p.dt and does not replace it with the
floating difference of absolute endpoint clocks.
```
