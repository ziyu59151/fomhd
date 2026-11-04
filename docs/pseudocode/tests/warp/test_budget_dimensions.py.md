# Massless fixture SI reference checks

## A0-PY-TEST-BUDGET-DIMENSIONS-01 - Physical reference reconstruction

```text
construct support, routing and periodic Hall parameters independently
for the Hall budget-only probe, replace the reference length by 0.3 m;
  do not build operators or claim a physical run with that altered fixture
reconstruct pressure energy=P*volume, advective time=L/u, total charge=rho*volume
  and total ion momentum=(mi*ni*volume)*u
for every registered norm of each selected diagnostic:
  compare its reference to the corresponding physical quantity/rate
  check its declared units and require the diagnostic is present
check current density without another area division, divergence with one
  inverse length, wall field residual with two inverse lengths, and
  interval momentum as momentum rather than force
```

These tests validate fixture scales, not numerical solution accuracy. Solver
equations, configured relative tolerances and hard/soft flags are unchanged.
