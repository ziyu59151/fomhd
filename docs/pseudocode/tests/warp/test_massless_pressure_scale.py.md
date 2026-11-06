# Massless pressure-work arithmetic scale

Source: `tests/warp/test_massless_pressure_scale.py`.

## A1-PY-TEST-PRESSURE-SCALE-01 - Strict cancellation and corruption

```text
build HallCase(kdi=1), its independent eigenpair, real periodic context,
  and real initial Hall eigenmode at relative amplitude 2e-6
use one predictor microstep with H=one cycle/64 and initialize through the
  production massless stage transaction
snapshot every configured physical budget before advance
advance the strict hard-check transaction

require:
  all budgets byte-for-byte-equivalent in behavior after advance
  predictor_m000_r3 electron_pressure_adjoint exists
  it is available, hard, and passed
  residual is about 1.3e-24 W
  arithmetic reference exceeds 1e-8 W (the omitted pre-cancellation operands
    dominate the old 3.86e-14 P:grad u scale)
  threshold still contains 4096*eps*reference and is at least the residual
  pressure_pairing remains hard and its configured physical budget is unchanged
  electron_pressure_only_pairing and electron_pressure_truncation retain the
    original P:grad(u) scale, not pressure_adjoint_scale
  the unrelated configured threshold is still derived from its unchanged budget

for a uniform real Hall stage:
  corrupt only the pressure-adjoint work residual to 1e-3 W
  rerun production check_stage
  require HardGate failure, hard=True, passed=False, raw>threshold
```

This is diagnostic-scaling evidence, not a relaxation of any physical gate.
