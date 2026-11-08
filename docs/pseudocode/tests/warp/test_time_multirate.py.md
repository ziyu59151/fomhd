# `tests/warp/test_time_multirate.py` pseudocode

## Independent numerical controls

```text
cold Hall:
  build u'=i a b, b'=i a u+i beta b
  compare the independent T0 map helper with expm of the augmented constant map
  verify C1 beta=0 preserves |u|^2+|b|^2
  refine micro h at fixed H
  require the T0 macro gain to persist while the C1 norm defect shrinks
  retain the nonzero C1 phase error as a limitation
  require C2 with zero frozen hydro terms to equal coupled RK4
  compare T0 at zero, tiny positive/negative, branch-boundary, and ordinary
    beta*H with an independent augmented exponential
  require phi_n(0)=1/n! and beta=0 exact Cayley limit

callbacks:
  use tiny backend arrays and callbacks independent of FOMHD
  verify exact work counts, stage identities and endpoint residuals
  inject failures in full-hydro, field, cheap-D and cheap-L callbacks
  require attempted calls to be charged before dispatch
  fail after one completed stage and retain that stage plus the attempted input
  fail after one completed macro and retain prior windows, current attempt,
    parseable report.json, complex NPZ history, and unchanged caller inputs
```

## Adapter and ownership controls

```text
periodic real state:
  pack P,Z
  snapshot the complete production RHS before adapters can reuse buffers
  compare blockwise D + A_m + A_p + L with that complete ion RHS
  compare A_m/A_p under two different valid physical B metadata
  verify cheap L never calls full-hydro adapter

closed K=1 endpoint:
  verify every ion/electron retained n/mom/pressure extent
  verify retained electron n=(q_i/e)n_i and physical quasineutrality
  verify blockwise complete ion RHS packing and direct endpoint split
  require the FieldBatch identity as cache version
  reject nonpositive physical ion pressure without the optional full oracle

harness time:
  use M>1 and multiple macro windows
  require every candidate window H=M*h and candidate/reference total durations
    to equal macros*M*h

normalization:
  require separate P/Z physical scale shapes and component units
  use a mixed density/momentum/pressure regression whose global cross-unit
    norm would hide the pressure error
  require actual increment max(abs(target-guess)) when maxima are equal but
    entries change

serialization:
  recursively host-convert arrays/scalars before JSON
  retain complex backend arrays in NPZ, not as JSON payloads
  mark C1 P-RHS unavailable and omit fake zero derivative arrays

ownership:
  snapshot P0,Z0, accepted state, cumulative ledger, context generation,
    wall-cache counters and source counters
  run a successful callback case and a capped failure
  require every snapshot unchanged
  require failure counters/history and last P/Z retained

C2 cost:
  require exactly 4*M cheap D, cheap L and field RHS calls per completed
    outer sweep
  require 4*M stage records and a positive exact history-byte count

diagnostics:
  select only each completed window's converged final outer stages
  exclude prior trial iterations from energy-rate rows
  linear electron rate uses only K with
    2*pi*Re sum(M0*(p_perp+0.5*p_parallel))
  use unequal nonunit M0 and compare with an independent physical-sample
    K=1 quadrature
  keep magnetic Parseval all-mode
  separate periodic target and highest-mode sentinel histories
  declare target=sentinel non-independent
  omit flattened closed-RZ phase labels
  preserve exact hard/observation budget flags in closed audit
  report zero wall initial solves for zero-row T

timing:
  candidate and production reference each use one cold run, the same warmups,
    the same synchronized repetitions, and fresh isolated contexts
  retain per-execution counters and campaign multiplier 1+warmups+repetitions

qualification:
  completed execution is true independently of scientific qualification
  no nonzero-hydro refinement evidence means formal second order stays design

provenance:
  require hashes for all numerical src/fomhd/*.py files and the actual tool, test,
    imported local fixtures/tools, and corresponding pseudocode mirrors
  derive an installed source root from its repository markers; honor an explicit
    FOMHD_SOURCE_ROOT override; use declared source B only outside the repository
  report actual Git top-level/HEAD/branch/dirty separately from the declared be5
    baseline
  include request and worker evidence hashes only when those optional files exist
  use an empty temporary task root to verify missing optional receipts do not fail
  never describe the scratch task root as repository authority

CLI:
  accept --macros 1 and 32; reject 0 and 33 with a bounded-control error

refinement:
  use a nontrivial callback system
  compare fixed-window h/2 and H/2 errors against an independent RK4 or
    closed-form reference
  retain the measured ratios; do not infer an untested order
```

Additional integration regressions check that failed timed executions include
completed macro work, and that a later reference failure retains the completed
candidate endpoint and full stage evidence. Generic all-frequency warnings do
not replace the separately declared target-accuracy and stability assessment.
