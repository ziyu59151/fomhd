# Checked linear residual refinement regressions

Stable step ID: `A0-VERIFY-LINEAR`.

Status: active. Targets: `tests/linear_refinement.cpp`,
`tests/linear_refinement_fixture.hpp`, and the direct-solve checks in
`tests/cuda.cpp`. Source pseudocode precedes the implementation.

All fixtures are host-owned FP64/complex128, with column-major matrices.
The small four-by-four fixture is the zero-RHS cancellation mechanism reduced
from the real six-by-six stepped disk; no captured NPZ is shipped.

```text
SmallHomogeneousRows(backend):
  build the fixed 4x4 reduction with opposite imaginary endpoint RHS values
  solve at rank_rtol=1e-12 and residual_rtol=64*4*epsilon(double)
  require preserved fixed endpoint solution components
  independently accumulate original residuals and denominators in long double
  require original componentwise defect <= the unchanged tolerance
  require returned residual equals backend(A*x)-b within rounding
  require 0<=refinement_steps<=5; do not demand equal CPU/CUDA iteration counts

ComplexKKT(backend):
  build fixed Hermitian 3-primal/1-constraint KKT with O(1) complex entries
  use a hand-derived complex solution and RHS with a zero constraint RHS
  use distinct positive row and column scales
  require known solution, original equations, rank spectrum, and input immutability

RejectionsAndZero(backend):
  nonsingular matrix and zero RHS -> exact zero solution/residual, zero corrections
  duplicate-row singular matrix -> NumericalError
  NaN/Inf in RHS or matrix, invalid scales/tolerance -> NumericalError
  deterministic complex inexact system with residual tolerance 1e-30 -> NumericalError
  no relaxed tolerance is permitted after the correction bound or stagnation

SteppedDisk():
  build the 6x6 stepped disk with existing physical test parameters and closed_reference
  initialize with normal hard gates, then advance exactly one accepted interval
  require accepted state step=1 and cumulative accepted_intervals=1
  inspect finite per-solve backward error and refinement counts

CapturedDisk(case_path):
  load the preserved real input through the production case/state loader
  initialize and advance one accepted interval with unchanged input budgets
  report per-solve refinement counts and maximum final backward error
  failure remains a failing test and is retained in logs
```

The CPU executable selects `numeric`, `disk`, or `case <path>`; by default it
runs numeric and manufactured disk checks. The existing CUDA `numeric` mode
runs the same direct numeric fixtures inside the user's Slurm allocation.
Runtime CUDA verification and the full suite belong to parent integration.
