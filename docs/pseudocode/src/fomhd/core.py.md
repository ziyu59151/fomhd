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
