# Wall projection refinement regressions

Source: `tests/warp/test_wall_refinement.py`.

## A0-PY-TEST-WALL-REFINEMENT-01 - Constant-pressure quiet trace

```text
for nodes in (6,9):
  build the production original-rectangle CPU Context and closed reference
  evaluate the real finite-inertia production stage
  project its electron non-electric plus electric force
  forward the projected samples and multiply by the quiet-study horizon
  install that momentum in a copy of the reference electron state
  call the unchanged full-state retained-spectrum and WallState checks
```

The pre-repair projector must fail the n=6 and n=9 constant-pressure cases
with `WallState`: its forward wall trace multiplied by the quiet horizon
exceeds the unchanged hard bound. The repaired projector must satisfy the same
checks without changing the bound.

## A0-PY-TEST-WALL-REFINEMENT-02 - Nonzero projection bookkeeping

```text
build an actual production rectangular CPU Context and wall factors
project a deterministic complex nonzero physical force
require the input array is unchanged
require nonzero projected force and reaction
require the forward full-wall trace is within the declared FP64 trace bound
project again and require idempotence within original-scale roundoff
independently check value = input + reaction
independently check reaction/winv = -T.H*multiplier
independently check multiplier_residual = S*multiplier - original_rhs
check weighted reaction work is zero within the existing roundoff scale
```

The tolerances use FP64 epsilon multiplied by the observed input, force, and
matrix operation scales. The test does not mock the solver or reproduce its
iteration control.

## A0-PY-TEST-WALL-REFINEMENT-03 - Empty wall

```text
build the existing periodic no-wall fixture
build wall factors for positive density
project a deterministic complex nonzero force with zero selected rows
require the result copies the input exactly
require zero reaction, multiplier, and multiplier residual
```

## A0-PY-TEST-WALL-REFINEMENT-04 - Unselected full-wall row

```text
build a production rectangular CPU Context and its selected-row wall factors
append an independent azimuthal trace row to T_full for a negative fixture
apply a nonzero force in that row which the selected-row solve leaves unchanged
require WallProjection with a full-wall-trace failure, not a returned candidate
```

The fixture deliberately invalidates the selected/full-row consistency. It
checks the public postcondition without changing the physical wall model.

These checks cover only the CPU production context and the named fixtures. They
do not imply CUDA execution or global A0 acceptance.
