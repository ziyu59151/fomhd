# `src/fomhd/core.py` implementation pseudocode

Model-aware combination reads an absent `electron_inertia` as `0`; only an
explicit `1` selects finite-fluid combination.

## A0-PY-CORE-01 State layout and allocation

1. `M=2K+1`; electron and ion density, momentum, and perpendicular pressure
   have `M*Nc` or `M*3Nc` complex128 entries. Electron parallel pressure is
   likewise `M*Nc`; native ion parallel pressure remains empty.
2. Magnetic field has `M*len(MB)` entries.
3. Reject `memory_limit==0`. Apply the same conservative native peak estimate
   and state/workspace cap before allocating on `Context.compute`.
4. On `cuda`, require a positive `device_memory_limit` and apply both peak and
   state/workspace bounds to that device cap before allocation. The CPU
   reference continues to use the host `memory_limit`.

## A0-PY-CORE-02 Retained Hermitian state

1. Require every allocated vector to have its native size, finite values, real
   k=0 value, and conjugate negative/positive modes within
   `512*eps*max(max_abs,1e-300)`.
2. For each species and mode, require momentum wall trace
   `T_full x` below
   `2048*eps*max(max_abs(mom), m*nref*vref,1e-300)`.
3. Finite-mass `combine` performs backend axpy on every native state member,
   stamps result identity from `StageTag`, then runs the complete retained
   check. Massless `combine` advances only independent state members and
   canonicalizes compatibility fields. If canonicalization rejects the raw
   candidate, the massless combine path retains that candidate and its stage
   tag through `_retain_common_failure_evidence` before re-raising.

## A0-PY-CORE-03 Spectrum/sample conversion

1. `mode_apply` and `sample_apply` validate layout and call backend matrix-vector
   products without changing ownership.
2. Retained inverse validates shape/input before FFT. Its default is physical:
   only an explicitly requested arbitrary complex trial may pass
   `physical=False`.
3. Physical output computes the native imaginary-residue limit
   `512*eps*max(max_abs(hat),reference,1e-300)`, invokes the backend
   physical-sample primitive, and reports the first nonreal index. On CUDA this
   primitive is a Warp FP64 kernel.
4. Forward transform delegates to the selected backend and preserves the
   retained mode layout.

## A0-PY-CORE-04 Reproducible time

1. `time_at_step` rejects `n>2^52`, nonpositive/nonfinite `dt`, or nonfinite
   result and computes `start_time+n*dt`.
2. `time_at_midpoint` computes `start_time+(n+0.5)*dt` and rejects it unless
   it lies strictly between adjacent scheduled step times.
3. No invalid time is repaired by clipping or changing `dt`.

## A0-PY-CORE-05 Memory plan

1. Use the exact native conservative counts for dense systems, constraint
   matrices, per-mode workspaces, and resident operator matrices.
2. Multiply entries by 16 bytes and use checked sums/products. This is a
   rejection estimate, not a measured allocator result.

## A1-PY-CORE-01 - Predictor state assembly and clocks

```text
assemble_predictor_state(P,Z,tag,context):
  require each independent P and Z member has the existing retained extent
  copy ion n/mom/perp from P
  copy electron perp/parallel and B from Z
  use harmless compatibility placeholders for electron n/mom
  canonicalize through the existing massless state path
  stamp step, generation, and time from tag
  do not mutate or alias P, Z, or the committed state

microstage_time(p,macro_step,micro_index,rk_stage_index):
  validate integer indices and 0<=rk_stage_index<4
  numerator=4*micro_index+rk_stage_offset[rk_stage_index]
  denominator=4*M
  offset: k1=0, k2=2, k3=2, k4=4
  t0=time_at_step(macro_step); t1=time_at_step(macro_step+1)
  H=p.dt
  t=t0+H*(numerator/denominator) using checked FMA
  require finite t and t0<=t<=t1
  require representable midpoints strictly ordered within each microstep
  do not form micro times by repeatedly adding h
```

The tangent displacement uses the same integer fraction, not a repeatedly
accumulated clock. The separate offset helper evaluates `H*numerator/denominator`
for the tangent rather than multiplying a repeatedly used `h`. A predictor
stage state has the tangent ion value and the corresponding RK4
electron-pressure/field state.

## A1-PY-CORE-02 - Predictor and waveform preflight memory

```text
predictor_history_entries(c):
  if coupling == synchronous: return 0
  require coupling in {predictor,waveform}
  require micro_steps is positive non-boolean integer
  full_state = checked(11*Nc+MB_size)
  retained = checked(4*M*full_state, plus conservative cache/rhs/stage counts)
  if waveform:
    ion_stage = checked(Mmodes*5*Nc)
    retained += checked(2*(4*M)*ion_stage)  # old/new complete ion paths
    retained += checked(2*ion_stage)        # old/new ion endpoints
  return retained

preflight_memory(c):
  reject missing/zero host cap; require positive CUDA cap on CUDA
  estimate peak including the predictor O(M) retained stage/quadrature history
  for massless contexts add the independent fixed-wall cache bound:
    q*(64*Ne*w + 128*w*w + 64*w + 32*Nb*w) bytes
    q=2K+1; w=len(geometry.walls); q and w are checked integers
    missing/None wall lists use w=0; finite inertia contributes zero
  for predictor or waveform massless contexts add per retained stage:
    16*N*w*w + 32*N*(w+3*Nc) complex128-entry allowance
    multiply by 4*M with checked products; this covers stage wall.samples S,
    CPU dense LU original/factors, and dense vector work per phi sample
  apply checked peak and state/workspace caps before any trial allocation
```

The predictor and waveform scheduler retain one sweep of all `4M` evaluated
full stages for the existing read-only interval diagnostics. Waveform adds the
old/new `4M` ion stage paths and both ion endpoints. The estimate does not grow
with `waveform_max_iterations`; prior sweeps retain scalar iteration metrics and
failure checks, not all stage arrays. It is not constant-memory or sparse.

The wall-cache term is a conservative static storage bound, not a measured
allocator pool, resident-set size, or performance claim. It never constructs a
wall matrix during estimation. The shared fixed wall-cache bound is counted once;
the per-stage wall-sample term is separate because every retained predictor
stage owns its own density-scaled matrices and factors.

Internal ion endpoint q_j storage aliases the corresponding reconstructed r1
stage; it is not an additional M+1-array history. Only the old/new final macro
endpoints are extra arrays in the memory plan.
