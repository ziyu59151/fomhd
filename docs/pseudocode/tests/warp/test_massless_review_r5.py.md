# Massless wall-work bound regression pseudocode

## A0-PY-TEST-MASSLESS-R5-01 - Checked producer envelopes

```text
build a small massless closed-grid context with diagnostics observable
construct the amplified magnetic-field fixture and evaluate one stage
require ion/electron/current and aggregate records to pass
require aggregate arithmetic and compatibility bounds to contain both
  nonnegative species contributions
construct current and momentum perturbations using C, Cdag, I, T_full
  and the actual reaction; scale each to half its checked input allowance
require the stage and corresponding work records to pass
```

## A0-PY-TEST-MASSLESS-R5-02 - Input and operation-domain rejection

```text
double the current and momentum perturbation allowances and require stage
  rejection by the corresponding full-wall input contracts
call gamma_operation with negative, infinite, and count*eps>=1
  inputs and require Failure("WallWork")
require gamma_operation(0) to be zero and ordinary active-mode bounds finite
  and nonnegative
```

## A0-PY-TEST-MASSLESS-R5-03 - Reduced scalar bound validation

```text
call bound_real with +Inf, NaN, a negative scalar, and nonfinite imaginary
  components and require a
  finite-nonnegative WallWork failure
build a module-scoped ordinary stage; for each of ion, electron, current,
  and species aggregate, replace one scale/upstream/compatibility/arithmetic
  residual with each invalid scalar, run check_stage, and require Failure
restore the residual after each parameterized case
scale finite fixture fields so the weighted paired sum overflows while every
  input array remains finite
run wall_work_metrics with overflow warnings suppressed and require the same
  finite-nonnegative WallWork failure
evaluate zero-reaction and zero-producer controls and require zero work/scale
  with finite nonnegative bounds for ion, electron, current, and species
  aggregates; a compatibility allowance need not be zero
```

## A0-PY-TEST-MASSLESS-R5-04 - Active Fourier and observation separation

```text
construct a Hermitian +/-1 harmonic state in the closed-grid nullspace
compare modal work against direct sample Parseval work for each producer
mutate only sample-space flux and require modal work and bounds unchanged
require transform_difference to be retained as observation evidence
inject a nonzero retained reaction and require the physical work gate reject
```

All tests use CPU FP64/complex128 fixtures. They verify diagnostic contracts
and finite-input failure handling; they do not establish GPU, long-horizon,
full-spectrum, or production performance validity.
