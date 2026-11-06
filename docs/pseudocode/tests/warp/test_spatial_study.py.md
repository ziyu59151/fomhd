# Fixed-domain spatial study tests

Source: `tests/warp/test_spatial_study.py`.

## A1-PY-TEST-SPATIAL-PROJECTION-DETECT-01 - Independent reference negative

```text
measure the baseline rectangle/compact spatial case
replace production massless._constraint_project with a zero projection
require one uninstrumented case run to raise the existing HardGate

in the negative test only, wrap diagnostics.gate to record the original hard
  failure and allow the already-computed result to return; production and
  benchmark code are not instrumented

compare baseline and corrupted:
  force_exact and E_exact arrays must be elementwise identical and nonzero
  projected_full_force must be exactly zero under the mutant
  corrupted E0 must be zero, so its error is the negative exact E array
  corrupted projected-force and E0 relative weighted-L2 errors must be one
  baseline projected-force relative error must be lower
  require at least one original hard check in the recorded failure

Do not use the underpowered zero-velocity work residual alone as the oracle and
do not select a new physical acceptance threshold.
```

The separate pressure-scale negative corrupts pressure work directly and must
still fail the unchanged hard gate.
