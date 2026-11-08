# T2-E1 affine time-prefix module

## Scope

`time_affine.py` implements only the bounded E1 affine algebra described by
`docs/pseudocode/T2_WINDOW_PLAN.md`. It does not implement E2, residual
correction, nonlinear field iteration, production integration, or a GPU
performance claim.

The module is backend-neutral Python algebra. Every call receives `xp`
explicitly as NumPy or CuPy and uses that namespace for all array work. There
is no backend discovery, CPU fallback, host transfer, stream operation, or
synchronization inside the module.

## Representation and ownership

- Retained matrices, vectors, stage states, RHS values, and returned endpoints
  are `xp.complex128`.
- `h` is a finite real scalar. It is not a CuPy scalar because converting a
  device scalar with `float` could synchronize.
- All inputs are read-only. The functions allocate fresh output storage and do
  not mutate or alias inputs.
- Shape, dimension, and dtype checks use only array metadata. No array values
  are inspected or copied to the host.
- A dtype, shape, array-type, or scalar-domain error is raised before algebra.
- The caller owns backend synchronization and observation. The module returns
  queued CuPy work without synchronizing it.

The tests exercise write-disabled NumPy inputs and non-aliasing outputs. The CuPy branch uses the same explicit `xp` operations; current owned-Slurm
verification is recorded in A1_T2_E_VERIFICATION.md. The isolated worker had
CPU evidence only; it did not supply the GPU claim.

## `affine_prefix(A, b, z0, *, xp)`

Inputs:

- `A`: `(M, ..., d, d)`, with `M >= 1` and `d >= 1`;
- `b`: `(M, ..., d)`;
- `z0`: `(..., d)`.

Each real leaf is the affine map `z -> A[j] z + b[j]`. Composition follows
temporal order:

```text
COMPOSE((A_left, b_left), (A_right, b_right))
    = (A_right A_left, A_right b_left + b_right)
```

Let `P = next_power_of_two(M)`. Leaves `j = M..P-1` are exact identity maps:
`A=eye(d)` and `b=0`. The implementation performs:

1. one vectorized up-sweep that builds aggregate maps for every tree level;
2. one vectorized down-sweep that builds exclusive prefix maps from the
   identity at the root;
3. one inclusive leaf composition and one endpoint evaluation per real leaf.

Python loops are over `O(log P)` levels only. No Python loop visits time nodes.
All maps at a level are batched.

The returned endpoints have shape `(M, ..., d)`. Work is
`O(P d^3)` from matrix composition and `O(P d^2)` from matrix-vector products;
tree-level count is `O(log P)` (matrix-product cost still depends on d). Peak temporary algebra storage is `O(P d^2 + P d)`.

## `constant_prefix(A, b, z0, *, xp)`

Inputs:

- `A`: `(..., d, d)`, shared by every microstep;
- `b`: `(M, ..., d)`;
- `z0`: `(..., d)`.

The implementation stores one matrix power per tree level and aggregate
vectors. It never materializes `A[j]` or a prefix matrix for every time node.

For a left block of length `s`, its aggregate vector is combined with the right
block as

```text
b_parent = A^s b_left + b_right
```

where `A^s` is the power stored at that level. The down-sweep carries the input
state of each subtree. At each parent, the right child's input is the endpoint
of the left child:

```text
z_right = A^s z_parent + b_left
```

The left child receives `z_parent`. A leaf endpoint is then
`A z_leaf + b_leaf`. The padded leaves use zero aggregate vectors.

This is a work-efficient tree, not an `O(M log M)` full scan. Excluding
`O(log M)` power-squaring matrix products, there are fewer than `3P` batched
matrix-vector applications over the whole tree:

- `O(P d^2 + d^3 log P)` scalar work;
- `O(log P)` span;
- `O(P d)` aggregate-vector storage plus `O(d^2 log P)` stored powers.

All nodes at a time level are batched. The returned endpoints have shape
`(M, ..., d)`. Tests compare this specialization with both
`affine_prefix` and an independent sequential recurrence for
`M = 1, 3, 5, 17, 65`.

## `rk4_affine(L, forcing, h, *, xp)`

Inputs:

- `L`: `(..., d, d)`;
- `forcing`: `(M, 4, ..., d)`, with distinct source identities in the four
  second-axis slots;
- `h`: finite real step size.

The shared affine matrix is the retained fourth-order polynomial

```text
A = I + H + H^2/2 + H^3/6 + H^4/24,    H = h L.
```

The source vector is computed from a zero-left RK4 step:

```text
k1 = f1
k2 = L (h k1 / 2) + f2
k3 = L (h k2 / 2) + f3
k4 = L (h k3) + f4
b  = h (k1 + 2 k2 + 2 k3 + k4) / 6
```

No source slot is dropped. The function returns `(A, b)`, with `A` shaped like
`L` and `b` shaped `(M, ..., d)`.

## `rk4_stages(L, forcing, h, left, *, xp)`

`left` is the explicit per-microstep left state, shape `(M, ..., d)`. Each row
is evaluated independently; it is not assumed to be a sequential continuation
of the previous row.

The returned tuple is `(stages, rhs, endpoints)`:

- `stages`: `(M, 4, ..., d)`;
- `rhs`: `(M, 4, ..., d)`;
- `endpoints`: `(M, ..., d)`.

Stage zero is `left` and its RHS is `L left + f1`. The remaining states and RHS
use the standard nonzero-left RK4 recurrences. The endpoint uses the weighted
four-RHS sum with denominator six.

## Verification

Tests are loaded by absolute `importlib` spec, not by path shadowing. The
independent oracles use plain per-step recurrences and do not call
`affine_prefix`, `constant_prefix`, or a composition helper.

Coverage includes:

- noncommuting varying maps with nonzero `b` and `z0`;
- identity-padded boundaries `M = 1, 3, 5, 17, 65`;
- generic versus constant-map prefixes and independent sequential recurrence;
- batched tree-operation bounds proving no per-time-node Python loop;
- four source slots separately and all four together;
- an analytic two-block Hall rotation and a stable dense four-component linear
  block;
- full stage-state, RHS, and endpoint comparison;
- read-only inputs and non-aliasing outputs;
- invalid dtype, shape, dimension, and real-step scalar cases.

The exact test command and observed RED/GREEN results are recorded in
`VERIFY.md` and `LOGS/`.

## A1-PY-TAFF-01 / A1-PY-TAFF-02 verification mapping

`test_time_affine.py` checks chronological noncommuting composition, nonzero
four-stage forcing, analytic rotation, non-power-of-two tails, independent
sequential recurrence and CPU/GPU arrays. No production whole-space matrix
composition or end-to-end speed follows from the number of tree levels.
