# A0 Warp migration and electron-model work

Status: active `main` implementation with bounded CPU/GPU verification,
2026-11-05. Native reference is commit 48beca7 on `reference/a0-native`. This
branch contains one active implementation under `src/fomhd/`; the native
implementation is not duplicated here.
The long finite-mass scan finished: nine cases reached the requested horizon,
and one case was rejected by a wall constraint. This is a development baseline;
massless global spatial accuracy and physical Tokamak validation remain open.

1. Preserve the native CPU/CUDA reference and its evidence (done).
2. Port the complete finite-mass A0 workflow to Warp/CuPy, retaining the CGL-B
   closure, compatible RZ/Fourier operators, closed staircase wall, dense
   rank-checked field solves, diagnostics, and explicit midpoint.
3. Compare independently produced native and Python CPU/GPU arrays; run
   manufactured/convergence/constraint/transaction tests and short GPU pilots.
   Promote to `main` only after review and equivalent-physics verification.
4. Derive and implement the optional massless, exactly quasineutral branch.
   Preserve physical electron mass metadata; do not set a divisor to zero.
   Reconstruct a compatible electron flux, constrained Ohm field, and CGL
   pressure evolution. Reject cross-model restart. Inapplicable finite-charge
   diagnostics are explicitly unavailable, never silently passing.
5. Add fully explicit RK4 as a control, frequency/finite-horizon dt guards,
   equal-horizon stability scans, frequency/cost tables, and reproducible plots.
   No implicit or semi-implicit electron method is part of this task.
6. After short pilots, submit useful overnight workloads within one owned
   Slurm GPU allocation and a few hours of runtime. Preserve failures.

Every model or implementation change updates design, mirrored implementation
pseudocode, source, tests, and affected provenance together. HYM is the primary
scientific reference; GTC is secondary. Benchmark labels distinguish numerical
verification, controlled theory comparisons, and experimental validation.
Performance tuning and higher-order boundary upgrades remain deferred.

The migrated workflow is now fully explicit RK4 by default. Existing midpoint
transactions, historical midpoint controls, and explicit
`time.integrator = "midpoint"` cases remain supported; equations and diagnostic
budgets did not change with the default.

The subsequent [fixed-domain and multi-period campaign](A0_SPATIAL_MODE_EVIDENCE.md)
records the Python-only RK4 mainline, independent continuum field comparisons,
momentum balance refinement and measured multi-cycle modes. It supersedes
earlier statements that those measurements have not yet been attempted; all
historical attempts and their original limits remain intact.

## Shared port interface (revision 1)

The package is `src/fomhd/`; invoke `uv run python -m fomhd` from any working
directory while selecting the exact checkout. `uv_build` packages this
src-layout project. Runtime provenance requires the imported package to resolve
under that checkout's `src/fomhd` and hashes the actual source plus
`pyproject.toml` and `uv.lock`. Names and numerical formulas follow the frozen
native reference interfaces. Module boundaries:

- `types.py`: dataclasses Species, References, Budget, Parameters, StageTag,
  FluidState, State, Context; Failure; namespace records for native outputs.
- `backend.py`: Backend(device='cpu'|'cuda:0'), xp, array, to_host, scalar,
  synchronize, mv, mm, linear, inverse_fft, forward_fft, derivative, axpy.
  `linear(A,b,row_scale,col_scale,rank_rtol,residual_rtol)` returns a record
  x, residual, singular_values, scaled_residual, backward_error,
  row_scale, column_scale, refinement_steps; at most five residual corrections.
- `geometry.py`: build_geometry(R,Z,polygon,options=None), build_rectangle,
  build_stepped_disk. Geometry entity maps/metadata are host setup data.
- `operators.py`: build_operators(geometry,K,options=None,backend=None),
  validate_operators, compatible transforms. Operators use backend-owned arrays.
- `core.py`: allocate_state, combine, check_retained, mode_apply,
  sample_apply, inverse, forward, time_at_step, time_at_midpoint,
  estimated_peak_bytes. Same argument order as native; Context.compute=Backend.
- `wall.py`, `fluid.py`, `fields.py`: native physics.hpp functions, same names
  and argument order. `fields.py` owns electrostatic, longitudinal, transverse.
- `stage.py`: evaluate_stage, initialize, retain_failure_evidence.
- `diagnostics.py`: native diagnostics.hpp functions plus boundary_ledger,
  state_totals, momentum. `check_stage` mutates result; gates raise Failure.
- `time.py`: explicit `advance`; RK4 is the default and midpoint remains an
  explicit option. Both preserve the native-compatible accepted/rejected
  transaction and evidence contract.
- `massless.py`, `massless_checks.py`: quasineutral state reconstruction,
  shared ion/electron/current fluxes, constrained Ohm field, pressure evolution,
  and original-equation/interval diagnostics. `stage.py` and `time.py` route the
  same external transaction to the selected electron model.
- `stability.py`: model-dependent frequency estimates and finite-horizon
  amplitude/phase guards; midpoint and explicit RK4 use their own polynomials.
- `io.py`, `__main__.py`: strict configuration, NPZ, fresh run directories,
  provenance, diagnostics, checkpoints/restart, failure evidence.

All vectors are ONE-DIMENSIONAL contiguous complex128 arrays exactly as native:
scalars [mode][cell], vectors [mode][component R/phi/Z][cell], samples replace
mode with phi sample, tensor [sample][row][column][cell]. Modes -K..K.
Matrices are ordinary two-dimensional arrays; use @, .T.conj(), .shape.
Geometry weights are real host arrays; operator weights are backend float64.
All stage fields and state arrays stay on the selected device. Only small
scalars and explicit I/O snapshots cross to host. Array ownership follows
CuPy; Warp shares real views using explicit stream and lifetime rules.
Pure CPU is an explicit NumPy reference backend, never a silent GPU fallback.

Parameters/Context/State/StageTag preserve native field names and constructor
order. StageResult and other result records use types.SimpleNamespace, with
all consumed fields initialized explicitly. Python reserved `parallel` is
valid; retain this name. Geometry tables expose keys, lookup, R, Z, boundary,
active_to_dense, dense_to_active, contains(i,j), find(i,j). Wall.component is
'r_face' or 'z_face'. Operators expose native names and shapes; factors are
backend implementation details. Setup independent-row selection is deterministic
and must match the native algorithm; all original constraint rows remain checked.

No source module may read C++ source, load a native binary, or rely on an oracle
at runtime. Tests may compare already exported native NPZ evidence selected by
`FOMHD_NATIVE_ORACLE_DIR`. Tests are under `tests/warp/`; mirrored pseudocode
is under `docs/pseudocode/src/fomhd/`. Source, comments, tests, logs, and this
implementation record are English.

## Acceptance and dt policy

`species.electron_inertia=0|1` defaults to 0, selecting the massless
quasineutral model. Explicit `1` selects finite electron inertia.
`time.integrator` selects rk4 (default) or midpoint; omitted configuration and
the pilot CLI default to RK4. Effective inertia and integrator are both
materialized in the restart semantic identity, so a historical checkpoint
created under the former omitted-finite default cannot resume as massless.
`time.dt_guard=error|warn` defaults to error. A warn override
is for controlled instability scans; finite/positive/linear-solver failures
remain fatal. Frequency checks use physical parameters and operator-resolution
rate estimates, including plasma, cyclotron, pressure/advection, and Hall scales
as applicable. These estimates are not a computed constrained eigenspectrum or
a proven upper bound on the nonlinear production operator.
Midpoint has |R(ix)|^2=1+x^4/4, so no nonzero imaginary-axis stability interval.
Report horizon-dependent amplitude/phase errors, not a fictitious stable CFL.
Initial configurable budgets: at least 32 samples per fastest cycle, amplitude
error <=1 percent and phase error <=0.1 rad over the requested horizon.

Warp 1.19.0 and CuPy CUDA13x 14.2.0 live only in the project uv environment.
Use FP64/complex128; CuPy supplies cuFFT and dense LU/SVD. Keep the dense
reference algorithm before separately considering scalability optimization.

## Verification receipts and limits

Receipt roots are under `/home/lxzeng/runs/fomhd/`. The immutable source hashes
in each run identify what was executed; the worktree's Git HEAD alone is not
sufficient for these pre-commit runs. Historical failures remain available.

| Check | Actual result | Receipt |
|---|---|---|
| Complete CPU suite including longitudinal controls and offline figures | 367 passed, 9 explicitly skipped GPU checks, warnings treated as errors | `warp-integration__20261105T0236/cpu-final-supplement.log` |
| Native CPU regression | 84/84 passed; native rebuild succeeded | same root, `native-build.log`, `native-ctest.log` |
| Independent CPU integration review | 200 targeted checks, four CLI cases, expected timeout exit 124; no open finding in reviewed scope | same root, `final-review/REPORT.md` |
| Corrected fixed-domain massless CLI | 22 completed: 13 pass, 9 measured; no scientific-acceptance verdict | same root, `massless-cpu-fixed-domain/report.json` |
| GPU backend, native parity, massless midpoint/RK4 and CLI/restart | 12 passed, including 9 CUDA cases and 3 CPU fixture/reference cases | `warp-final-gpu-r2__20261105T0330/pytest.log`, job 10812 |
| GPU local Hall response | Three kdi samples passed with real CUDA execution | `warp-final-gpu__20261105T0320/hall/report.json`, job 10811 |
| Narrow GPU memory check | Actual Warp kernels and FFT/LU tests: 2 passed, 1 deselected; memcheck 0 errors | `warp-final-gpu-r2__20261105T0330/memcheck.log` |
| GPU coupled temporal MMS | Midpoint combined orders 1.936/1.969; RK4 3.998/3.996; all required hard checks available and passed | `warp-overnight-retry__20261105T0132/mms/results.json`, job 10810 |
| Completed closed-wall finite-mass GPU scan | 9 cases completed, 1 WallState rejection after 7 of 8 requested steps; all 10 frequency estimates unavailable; observe-only budget exceedances retained | `warp-overnight-retry__20261105T0132/scan/summary.json`, job 10810 |
| Unforced finite-mass longitudinal control | 12 CPU and 12 CUDA cases pass the declared polynomial/diagnostic gates; coarse continuum errors remain large and explicitly reported | `warp-longitudinal-cpu__20261105T0403/report.json`, `warp-longitudinal-gpu__20261105T0406/results/report.json`, job 10813 |
| Same-input native/Warp endpoints | Four-step finite-mass comparison passed; timing scopes differ, so no speedup ratio | `native-warp-parity-cost__20261105/comparison.json` |
| Spatial review and corrections | Fixed physical MMS, 1/s field metadata, rounding-aware containment accepted; global accuracy still unresolved | `warp-integration__20261105T0236/spatial-review/REPORT.md`, `fix-addendum/ADDENDUM.md` |
| Independent offline plotting review | 12 focused checks; all 51 plotted samples agree with archived data; no open finding | `warp-integration__20261105T0236/plot-review/REPORT.md` |

The first GPU acceptance found a **test setup** error: a CPU-only nullspace
fixture implicitly converted CUDA arrays before evolution. The repaired test
uses the existing CPU initial-state generator and explicit upload; solver
runtime bytes are unchanged between jobs 10811 and 10812. The first memcheck
including dense operator assembly timed out at 240 s and has no clean verdict.
The passing retry is limited to the two named backend tests; it is not a full
solver memory, race, initialization or synchronization audit. Its singular-LU
warning belongs to the intentional rank-deficiency rejection control.

The closed-grid massless data do not establish fourth-order global accuracy.
The initial coarse sequence mixes changing regional quadrature samples and
boundary closures even on an exact rectangle. Independent refinement through
n=16 reduced global reconstructed-velocity and pressure-force errors to about
1.03% and 3.13%, respectively, but does not prove an asymptotic wall order.
The Ohm comparison shares discrete I/Q/truncation and verifies pressure-source
propagation and assembly, not independent continuum electric-field accuracy.
No pressure sign, evolution formula or physical tolerance was changed to
improve those results. Boundary-order upgrades remain deferred.

## Direct longitudinal time control

`tools/warp_longitudinal.py` evolves an unforced small-amplitude warm
two-species longitudinal eigenmode using the production finite-mass stage,
time transaction and diagnostics. A nine-point periodic spectral test
adapter supplies compatible derivatives, a mean-gauged Poisson operator,
and independent constraint rows with all full residuals retained. It does
not change the production closed-wall geometry. The reference frequency and
polarization are derived independently from the linearized equations; no
production Jacobian is used as the oracle. Only `dt_guard=warn` is overridden
for deliberately coarse steps; physical hard budgets remain unchanged.

Both CPU and GPU scans completed 12 cases at the same two-cycle horizon,
with 4, 8, 32, 64 and 128 steps for midpoint/RK4 and a half-amplitude control
for each method. The analytic frequency is 8.981112 MHz for these controlled
parameters. At four steps, measured amplitude factors are about 642.738
(midpoint) and 16.979 (RK4), agreeing with their numerical polynomials.
At 128 steps, the midpoint amplitude is about 1.001487, with complex continuum
error 0.020197; RK4 has complex continuum error about 9.73e-6.
Passing the polynomial comparison does not mean coarse-step continuum
accuracy or stable physical evolution. All initialization/interval hard
checks, warnings, source snapshots and numeric histories are retained.

Follow-up regressions retain a rejected attempt's structured timestep report
when provided and require complete, matching half-amplitude plot controls.
These changes affect evidence retention/validation, not the production
equations or time update. Archived scans pin their original measured sources;
the final plot pins the corrected offline script and preserves identical data.

This comparison verifies a local wave and finite-horizon numerical response,
not production-wall spatial accuracy, a full stability region, physical CGL
ordering, or Tokamak/ICE physics. The chosen wave frequency exceeds the
electron cyclotron frequency; its mathematical verification must not be
reported as validation of the CGL physical regime.

The separate overnight closed-wall midpoint case at eight steps rejects
with `WallState` after seven accepted steps. Its archived electron mode-zero
residual is 4.2686974e-31 against a 4.1424681e-31 limit. The completed scan has
nine accepted cases and this one rejected case. All ten frequency estimates
are unavailable under the estimator's sample/coherent-cycle requirements. These records do
not establish that Langmuir growth caused the rejection. The direct periodic
control above supplies a separate, attributable time-step experiment.
Observe-only time-step and momentum-budget exceedances also remain in the
scan. Its normalized residual maximum covers available linf/weighted_L2
checks with positive thresholds; it does not cover every diagnostic or turn
case completion into a scientific acceptance verdict.

## Overnight experiment and physical scales

Job 10810 used one RTX 5090, two CPU cores and a 16 GiB reservation with a
five-hour limit. It finished on 2026-11-05 at 05:30:50 Asia/Shanghai, after
14,333.05 s (about 3 h 59 min). After coupled temporal MMS, it scanned
midpoint/RK4 at the same final
time of five plasma periods with 8, 16, 80, 160 and 320 steps. The small
controlled case uses n=1e12 m^-3, P=1e-9 Pa and B=1e-5 T; these are **not
Tokamak parameters**. A measured pilot estimated 12,425 s for the scan plus
MMS/setup. The 14,000 s scan budget, partial failures, source snapshot and
generated figures remain in `warp-overnight-retry__20261105T0132/`. The two
short final acceptance allocations ran on a separately assigned spare GPU;
their runtime is not used for speed comparison.

The tool-generated frequency table is
`frequency-scales__20261105/frequencies.csv`. For deuterium, ne=1e19 m^-3 and
B=3 T it gives fci=22.879 MHz, fpe=28.393 GHz, fce=83.977 GHz and an
upper-hybrid estimate of 88.647 GHz. Ten samples of the 30th ion harmonic give
dt=1.457e-10 s; 32 samples of that upper-hybrid period give dt=3.525e-13 s.
These are characteristic scales and sampling estimates, not a universal
stability bound. In particular, comparing the bare electron cyclotron and
plasma frequencies alone does not identify the fastest resolved Darwin mode.
The finite-mass momentum equation retains electron bulk gyromotion through
the full Lorentz force; a gyrotropic pressure closure does not itself remove
that response. Kinetic gyrophase physics and higher cyclotron harmonics are
not implied. The transverse Darwin constraint changes the response with
wavenumber and propagation angle, so the usual upper-hybrid expression is
not automatically the exact frequency here. The current parallel longitudinal
control isolates the plasma branch. Its results do not determine the fastest
branch for every production grid, and its mathematical test does not validate
the physical ordering of CGL at arbitrary high frequencies.

The massless branch removes the finite-charge/Langmuir degree of freedom and
uses its quasineutral constraint at every stage. Its Hall high-k branch grows
without the electron-inertia saturation; the grid-dependent Hall estimate
therefore remains in the explicit-step guard. Neither branch is universally
preferable, and the switch does not introduce an implicit electron method.

Offline figures are generated by `tools/plots/plot_warp_verification.py`,
`plot_warp_longitudinal.py`, and `plot_warp_benchmarks.py`.
They preserve input hashes, plotted values, failed
scan outcomes and actual backend labels. Formal performance optimization,
multi-GPU scalability, kinetic ions/ICE and experimental validation are not
accepted by this record. No release tag is created for these limited tests.

The reviewed temporal/Hall and closed-grid figures are in
`warp-verification-figures-r3__20261105T0335/`, with PNG/PDF files,
`plot-data.json`, `provenance.json`, and an explicit `SCOPE.md`. The MMS export
retains the omitted-source negative-control outcome as well as the plotted
integrator data. All prior failures and earlier renders remain archived.
The direct longitudinal CUDA figure is in
`warp-longitudinal-figures-r2__20261105T0420/`, with the same PNG/PDF, data,
provenance and scope products. Plotting never reruns a solver.

The combined gallery at
`/home/lxzeng/runs/fomhd/a0-benchmark-figures__20261104T050738Z/index.html`
retains the original five native figures and appends separate finite-mass
and massless Warp groups. The 2026-11-05 appendix includes completed scan
histories, retained failure and observe-only records, explicit CPU/CUDA
labels, and source hashes. The original gallery is preserved as
`index-native-20261104.html`. This update only inspects and renders existing
solver runs; it does not change a physical model, numerical guard, or solver.
