# Foundation native oracle

Source: `tests/warp/foundation_native_oracle.py`.

## A0-PY-TEST-FOUNDATION-NATIVE-ORACLE-01 - Procedure and assertions

```text
Resolve ORACLE_ROOT through the shared workflow fixture, then select ordinary
and stress NPZ exports produced independently by the native C++ exporter.
If an export is absent, mark this oracle-only comparison skipped explicitly.
Load with allow_pickle=False and rebuild Python geometry/operators from the
same fixed rectangular input.
Compare ordering, weights, interpolation/constraint matrices, each mode's
G/C/DB/DJ/Cdag/Lphi and signed wall rows; require equal shapes and the stated
foundation tolerance. This checks the port, not continuum convergence.
```

Covered top-level functions and fixture classes:

- `assert_array`
- `test_foundation_matches_independent_native_arrays`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.
