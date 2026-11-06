# Foundation fixture

Source: `tests/warp/foundation_fixture.py`.

## A0-PY-TEST-FOUNDATION-FIXTURE-01 - Procedure and assertions

```text
parameters(K):
  build proton/electron SI Species with positive physical masses
  set fixed reference length/time/density/pressure/field and derive current
  set Nphi=4*K+1, FP64 rank tolerance, memory caps and explicit
    electron_inertia=1 for this historical finite-equation fixture
  register array linf/weighted-L2 and scalar value budgets with physical units
  distinguish hard algebraic/positivity gates from observational balances
  return a fresh Parameters object for each test; never share mutable budgets
```

Covered top-level functions and fixture classes:

- `parameters`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.
