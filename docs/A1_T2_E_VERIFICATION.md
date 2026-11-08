# T2-E electron/field implementation and evidence

Status: bounded E1/E2/E3 implementation, 8 November 2026. This is an
experimental fixed-ion field API, not a complete coupled runtime scheme.
The design authority is D070 and the A1 design in the sibling repository.
No physical model, explicit RK4 target, precision or existing acceptance gate
changed. T2-I, T2-C, FK, multi-GPU and production-scale qualification are pending.

## Implemented computation

`time_affine.py` implements actual up/down affine trees, shared small-block
powers and four-source RK4 reconstruction. Chronological composition is
`(A_later A_earlier, A_later b_earlier + b_later)`. Tail padding, noncommuting
blocks, nonzero sources and both midpoint stages have independent controls.

`time_field.py` solves the original fixed-ion RK4 window relations. Each residual
batch reads one old waveform; four stages remain causal inside each microstep.
A tree correction propagates defects between nodes, followed by a newly evaluated
true residual. There is no serial-window guess or final replay. One initial RHS,
all rejected trials, all derivative actions and an endpoint validity RHS are
charged. The selected residual already supplies all stage E/current/velocity.
Inputs, accepted state and cumulative ledgers remain unchanged on success or
failure. Endpoint validation uses supplied right-stage P, not an accepted ion
endpoint. Integration with T1 and its complete energy transaction is not done.

The E2 reference uses a periodic uniform Hall Fourier symbol; true residuals
retain nonuniform density, finite amplitude and nonlinear pressure/field terms.
The E3 reference freezes the real constrained spatial derivative and applies
truncated powers of the RK4 polynomial by batched, scale-normalized centered
finite differences. Degrees 1/2/4/8 were compared in bounded controls. Degree 1
is cheaper in short windows; it is not a general long-window choice. The two perturbation RHS batches
remain sequential in this version. Tiny directions can lose finite-difference
homogeneity; a capped perturbation prevents amplifying compatibility roundoff.
Truncation and this approximate derivative mean E3's tree is a specifically
defined approximate inverse, not an exact affine semigroup or quadratic Newton.
Original residual and independent sequential waveform comparisons govern it.

`field_batch.py` provides the actual massless quasineutral CGL-B/Hall/constrained
Ohm field RHS with an independent time axis. It preserves production operators,
physical Fourier conventions, density-dependent projections and fixed electric
wall correction. Numerical arrays stay on one explicit NumPy or CuPy backend;
there is no whole-state host fallback. Domain, rank, solve, CT, wall-current and
Ohm checks remain active. Immutable producer/cache identities include model,
operator, spectral and rank parameters. It does not evaluate a complete ion RHS.

## GPU scheduling and separate spatial improvements

Each GPU residual evaluates all microsteps together using batched products,
FFTs and solves. Tree nodes at the same level are independent numerical batches.
Host loops cover only tree levels, four RK stages, components/modes and bounded
refinement; no complete-microstep host loop exists in the new solver. The
sequential loop remains an explicitly identified oracle and timing baseline.

The wall normal matrix is assembled exactly from column-incidence pairs. If
all pairs are diagonal, its positive diagonal gives exact singular extrema and
elementwise solves; otherwise the dense normal path remains. The explicit
`--dense-normal` control preserves that comparison. All selected/full-row checks
remain. Dense trace actions and fixed global wall factors are still present.

The older `core.sample_apply` now uses one multiple-RHS product instead of
one host-issued matvec per physical phi sample. Its production-stage old/new
comparison is separate from time-algorithm measurements. No Warp synchronization
was removed without a new lifetime/error contract; no precision was reduced.

## References, measurements and limits

Evidence root:
`/home/lxzeng/runs/fomhd/a1-t2-electron-field__20261108T071000Z`.
Independent review and before/failed snapshots:
`/home/lxzeng/fomhd-design/history/2026-11-08/evidence/a1-t2-electron-field__20261108T071000Z`.
The Chinese design report is `A1/A1_T2_E_REPORT.md` under design implementation.
`RESULTS.json` and `INPUT_MATRIX.csv` in the run root enumerate final cases;
`gallery/a1-benchmark.html` uses the canonical A0 renderer, CSS and plot helpers.
Raw per-case JSON/NPZ and tracebacks retain exact numerical-source hashes.

CPU executes the new tree itself. Separate sequential time oracles, production
stage block comparisons, analytic Hall/source algebra, nonlinear varying-stage
ion-path probes and GPU same-algorithm tests prevent relying only on shared
CPU/GPU logic. CPU A0 is a numerical reference with its original error scope.
Checks include all nodes/stages/RHS, E/current/velocity, time quadrature, read-only
inputs, invalid endpoint, stale producer identity, failed-library propagation,
nonconvergence and allocation preflight. No new restart format exists.

Timings use two warmups and five synchronized repetitions; input setup, cold
window and raw samples are retained separately. Per-window comparisons use the
same GPU allocation, physical inputs and retained outputs. The fair baseline is
sequential **field-only** RK4 with the same real G, checks and E/J/u history. It
is deliberately cheaper than invoking a complete A0/T1 stage and discarding its
ion work; no eliminated ion/diagnostic work is called a temporal speedup.
Full coupled A0/T1/new-method end-to-end comparison belongs to T2-C.
CPU/GPU ratios are separate hardware/backend observations.

Whole-2T fixed-P observations use an independent CPU sequential trajectory.
The initial 1e-10 window tolerance misses the long-horizon 1e-8 field comparison;
1e-12 is tested without changing that gate. The long observation is one complete
execution per backend/method, not five-repeat performance qualification.
High spatial Hall modes have retained eight-round convergence failures.
Temporal refinement at fixed H and longer H at fixed h are separate cases.
Where same-h errors reach the h/h2 roundoff floor, temporal accuracy is unresolved.

An additional real-wall long-window control holds H fixed and halves h
(M256 to M512, n12, amplitude0.05, spatial mode3). Degree1 converges in the
coarser case but fails the finer case's second correction line search, on CPU
and GPU. Degree2 passes the fine case in five rounds on both backends. Degree4
fails a magnetic-divergence check inside the polynomial/JVP correction: a large
coefficient amplifies an extremely small nontangent residual component. Saved
correction replay localizes that representation limit; it does not demonstrate
physical instability. The failed full run is retained, no constraint is relaxed,
and no degree4 GPU/full-window fix is claimed. These stress runs have zero
warmups and one measured repetition; they are not repeated performance results.

Nsight Systems and CuPy memory hooks measure separately from ordinary timings.
Pool live/reserved peaks, CUDA requested-allocation peaks, async pool reservation
and host RSS have explicit scopes. Complete-process RSS was not separately
sampled for the original whole-2T scripts or historical failed cases; their
missing measurements are not filled from a different run. Driver overhead is not silently equated to
pool memory. Profiling includes a serial oracle and is not a speedup timing.
The failed profile/device setup from overlapping allocations is preserved;
sequential owned-allocation retries are separate successful evidence.

## Production scaling boundary

E3 covers the actual nonuniform constrained RZ producer at 8/12/16 nodes per
axis, K=1, Nphi=5, including stronger finite-amplitude and longer-window controls.
These are bounded tests, not a 1024-squared production certificate. Existing
operator construction, trace matrices and fixed wall factors remain dense.
An allocation-free shape analysis records their large-grid lower bounds and
O(M*dZ) history. No full-space propagator matrix is required by the tree.
At fixed physical H the explicit Hall step also decreases with mesh spacing;
M and reference propagation difficulty grow. Small-grid iteration counts do
not certify bounded rounds after that change.

The next spatial unit must construct sparse/direct local operators and wall
traces without first materializing their dense versions, replace global fixed
wall solves with a validated structured action, and bound/stream window history.
Derivative and long-propagation costs require separate measurements. Larger
meshes, full coupled acceptance, long-time RZ physics and A100 performance remain
unverified. Follow the dated report's concrete scaling sequence and stop gates.
