# Workflow fixture

Source: `tests/warp/workflow_fixture.py`.

## A0-PY-TEST-WORKFLOW-FIXTURE-01 - Procedure and assertions

```text
Resolve FOMHD_NATIVE_ORACLE_DIR when set; otherwise use this checkout's
build/native-oracles. Native exports are test inputs, not runtime dependencies.
cumulative(): initialize all physical integrals and counters to zero.
from_oracle(device,stress): load safe NPZ, read native metadata, build matching
geometry/operators and upload every state array explicitly to Backend(device).
assert_dimensional(actual,expected,reference): require identical shapes and
maxabs(error)<=2e-9*maxabs(expected)+2048*eps*physical_reference;
never substitute a dimensionless unit floor for a small physical quantity.
```

Covered top-level functions and fixture classes:

- `cumulative`
- `from_oracle`
- `assert_dimensional`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.
