# `src/fomhd/core.py` implementation pseudocode

Model-aware combination reads an absent `electron_inertia` as `0`; only an
explicit `1` selects finite-fluid combination.

## A0-PY-CORE-01 State layout and allocation

1. `M=2K+1`; electron and ion density, momentum, and perpendicular pressure
   have `M*Nc` or `M*3Nc` complex128 entries. Electron parallel pressure is
   likewise `M*Nc`; native ion parallel pressure remains empty.
2. Magnetic field has `M*len(MB)` entries.
3. Before any cap check, allocation, retained-state check, source counter
   mutation, or physics work, require model/storage coherence:

   ```text
   availability = op.finite_mass_available  # bool
   if electron_inertia != 0 and availability is false:
     Failure("Model", "finite-mass electron model requires complete
                      finite-mass operator storage")
   ```

   A massless model accepts either the reduced selected-storage operators or
   the complete full-storage operator set; the latter remains the oracle.
   Non-boolean availability is `Failure("Model")`.
4. Reject `memory_limit==0`. Apply the selected conservative peak estimate and
   state/workspace cap before allocating on `Context.compute`.
5. On `cuda`, require a positive `device_memory_limit` and apply both peak and
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

1. `mode_apply` validates layout and calls backend matrix-vector products without
   changing ownership. Its implementation is unchanged.
2. `sample_apply` validates `N>0`, a two-dimensional matrix, and one-dimensional
   values of length `N*columns`, then makes one multiple-RHS backend call:
   `backend.mm(matrix, values.reshape(N,columns).T).T`. The first `.T` is an
   ordinary transpose, not a conjugate transpose. The transposed result is
   flattened through `backend.array`, yielding a contiguous backend-owned
   `(N*rows,)` result. Preserve the existing complex128 result and
   backend host/device checks; malformed sample layout fails with
   `Failure("Shape", "sample operator shape")`.
3. Retained inverse validates shape/input before FFT. Its default is physical:
   only an explicitly requested arbitrary complex trial may pass
   `physical=False`.
4. Physical output computes the native imaginary-residue limit
   `512*eps*max(max_abs(hat),reference,1e-300)`, invokes the backend
   physical-sample primitive, and reports the first nonreal index. On CUDA this
   primitive is a Warp FP64 kernel.
5. Forward transform delegates to the selected backend and preserves the
   retained mode layout.

## A0-PY-CORE-04 Reproducible time

1. `time_at_step` rejects `n>2^52`, nonpositive/nonfinite `dt`, or nonfinite
   result and computes `start_time+n*dt`.
2. `time_at_midpoint` computes `start_time+(n+0.5)*dt` and rejects it unless
   it lies strictly between adjacent scheduled step times.
3. No invalid time is repaired by clipping or changing `dt`.

## A0-PY-CORE-05 Memory plan

1. Use exact checked counts for finite dense systems, constraint matrices,
   sampled/vector workspaces, and the selected operator inventory.
2. Multiply entries by 16 bytes and use checked sums/products. This is a
   rejection estimate, not a measured allocator result.
3. Charge the finite-inertia dense system and full `Bc` workspace only when
   effective `electron_inertia` is 1:

   ```text
   system = (2K+1)*Ne + selected_Bc_rows
   finite_entries = 32*system^2 + 4*Bc_full.size
   ```

   For massless stages these dense finite-inertia allocations do not exist.
   Omit only `16*finite_entries`; retain the existing state, diagnostic,
   sample, predictor/history, resident shared-operator, and fixed-wall-cache
   terms. Do not remove or raise either host/device cap.
4. Select this term from the effective electron model, not from a zero-row
   `Bc` placeholder. The default finite formula, including its conservative
   `8*resident` term, and both cap checks remain unchanged.
5. For massless stages replace `8*resident` with the following measured-shape
   inventory and temporary bound:

   ```text
   persistent =
       mode G/C/DB/DJ/Cdag arrays
     + I/H/QB/T/T_full/DR/DZ/topology
     + b1r/b1z/b2r/b2z + PR/PZ filter matrices
     + M0/M1/MB/Rbar_r/Rbar_v
     + retained PR/PZ LU originals, factors and pivots
     + finite matrices/factors only when a full-storage massless oracle is used
     + wall caps and poloidal magnetic norm arrays for the aligned box

   wall_temporaries = Nb*W + Ne*W + 5*W*W
   projection_temporaries = 2*rows(T)*3*Nc + rows(T)^2
   matrix_temporaries = 16*max(wall_temporaries, projection_temporaries)

   massless_plan =
       16*(sample workspace + state/predictor history)
     + persistent
     + matrix_temporaries
     + fixed_wall_cache_bound
   ```

   The wall-cache `C` and `Cdag` members borrow the resident operator arrays
   and are not charged twice. The fixed-wall bound owns the retained extraction,
   weighted-adjoint, wall matrix and prepared-factor arrays. During construction
   one `Nb x W` curl-boundary temporary and one `Ne x W` thin-curl temporary are
   live; five `W x W` allowances cover the selected principal, two scaling
   expression buffers, the scaled matrix and its LU factors. The coupled
   projection bound covers its weighted matrix, exact normal and batched
   products. Factor originals can alias their source matrices; counting them is
   a stated conservative upper bound, not an allocator measurement.
   Also account for the diagnostic complex Cdag stack and its absolute-value
   output while abs(I) remains live, for retained abs(I)/abs(Cdag)/abs(T_full),
   and for I/DR/DZ conjugate-transpose work arrays. These phases are separate;
   use their maximum with wall/projection temporaries. At EAST32 the diagnostic
   assembly alone requires 491,204,256 bytes. Runtime/library memory is measured
   separately; array accounting is not a process RSS certificate.
6. Check every shape product, sum, byte conversion and cap comparison for
   arithmetic overflow. Build the inventory from the actual operator shapes
   and reject an incomplete G/C/DB/DJ/Cdag mode inventory with
   `Failure("Shape")`; never replace its temporary allowance with zero.
7. Treat `estimated_peak_bytes` as a conservative rejection plan. Report
   actual allocated stage/diagnostic/history lower bounds and measured RSS or
   allocator-pool observations separately; never use one as proof of another.

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
history_sample_count(c):
  samples = getattr(c, "_history_samples", c.N)
  require samples is a non-boolean integer and 1 <= samples <= c.N
  return samples

predictor_history_entries(c):
  if coupling == synchronous: return 0
  if coupling == multirate_wr:
    require positive non-boolean M/max_iterations and s in {3,5}
    use the separate conservative WR branch below
  require coupling in {predictor,waveform}
  require micro_steps is positive non-boolean integer
  samples = history_sample_count(c)
  full_state = checked(11*Nc+MB_size)
  retained = checked(4*M*full_state, plus conservative cache/rhs/stage counts)
  retained += checked(256*samples*(3*Nc+Ne+Nb))
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
    16*samples*w*w + 32*samples*(w+3*Nc) complex128-entry allowance
    multiply by 4*M with checked products; this covers stage wall.samples S,
    CPU dense LU original/factors, and dense vector work per phi sample
  apply checked peak and state/workspace caps before any trial allocation
```

`_history_samples` is an internal alias-view identity. The ordinary context has
no such attribute and therefore uses `N`. A massless alias view changes `N` to
its alternate spectral sample count but inherits the outer view's history
sample identity unchanged. A nested alias view propagates that same original
identity. Boolean, non-integer, nonpositive, and greater-than-`N` identities
fail before a history allowance is returned.

The predictor and waveform scheduler retain one sweep of all `4M` evaluated
full stages for the existing read-only interval diagnostics. Waveform adds the
old/new `4M` ion stage paths and both ion endpoints. The estimate does not grow
with `waveform_max_iterations`; prior sweeps retain scalar iteration metrics and
failure checks, not all stage arrays. It is not constant-memory or sparse.

The actual alias probe runs synchronously inside the outer stage before that
stage returns. Its shallow context copy shares the outer retained history and
changes only the one-stage sample workspace. The corrected `N=11` alias plan is
therefore the unchanged outer `N=5` plan plus the alias-stage sample delta:
`5,262,540,504 = 5,081,734,872 + 180,805,632` bytes at the EAST32 K1 waveform
fixture. The outer result and its canonical/temporary data remain live until the
alias call returns; the alias result and local stage temporaries are released
after the probe. The pre-existing multi-stage history allowance conservatively
dominates the additional one-stage transient overlap. This is a rejection-plan
identity, not an allocator trace, measured RSS, or performance claim.

The wall-cache term is a conservative static storage bound, not a measured
allocator pool, resident-set size, or performance claim. It never constructs a
wall matrix during estimation. The shared fixed wall-cache bound is counted once;
the per-stage wall-sample term is separate because every retained predictor
stage owns its own density-scaled matrices and factors.

Internal ion endpoint q_j storage aliases the corresponding reconstructed r1
stage; it is not an additional M+1-array history. Only the old/new final macro
endpoints are extra arrays in the memory plan.

## A1-PY-CORE-03 - Conservative multirate-WR retained history

```text
retained_full_stage =
  2*state + cache + 256*samples*(3*Nc+Ne+Nb)

coarse_count = 1+s*max_iterations
fine_count = 4*M*max_iterations
alias_count = 2*M*max_iterations
fixed_full_count = 2  # full base and accepted endpoint

history =
  (coarse_count+fine_count+fixed_full_count)*retained_full_stage
  + alias_count*retained_full_stage(alias_samples)
  + coarse_count*(3*P + 2*projection + wall multiplier storage)
  + fine_count*(2*(P+Z) + 4*EM vectors + wall multiplier storage)
  + final generic window P/Z/G/C and coarse-node arrays
  + retained wall history for full/fine/alias stages

alias_samples=max(history_samples,2*N+1)
```

All integer products and sums are checked. This is a conservative capacity
rejection estimate, not an OOM measurement. It explicitly charges every
retained outer sweep, tangent and `s` coarse frames, alias frames,
base/endpoint, generic window arrays, and wall records. A0/T0/T1 formulas are
unchanged. Actual host/device resource guards remain separate in the runner.
