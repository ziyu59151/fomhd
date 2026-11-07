# Controlled T0 temporal evidence

Status: controlled CPU baseline delivered; T1 awaits the user's decision.
Scientific protocol: design decision D064 and `A1/A1_T0_BENCHMARK.md`.
Numerical benchmark commit: `206e850c3220c021b03932e262e8e52c307f4a7e`.
The T0 accepted update is inherited from `93219d2`; the current commit changes
failure-record collection and the separate prescribed-ion control workspace.

## Result and scope

The fixed tangent ion predictor and complete microstage ion quadrature have a
reproducible positive baseline. At h=T/2048 over two target periods, mixed-seed
M4 and M8 satisfy the declared target-accuracy, high-mode-growth and hard-check
criteria. M8 is close to the 1% growth flag. M16/M32 still satisfy target vector
accuracy while amplifying the seeded high harmonic. Microstep refinement at
fixed H does not remove the principal macro coupling effects.

This is a periodic temporal-isolation fixture: nine spatial samples, target
spatial harmonic 1, high harmonic 4, toroidal K=0 and Nphi=1. It is not general
CAW-CFL, closed-wall, production-scale, multi-GPU or speedup validation.
CAW denotes the fast magnetosonic/compressional Alfven branch. Micro CFL refers
to the electron/field fast subsystem; ion macro CFL and complete coupled-map
amplification remain separate questions.

| M | H/T | Target-only maximum relative vector error | Mixed maximum high-mode norm gain | Mixed observation |
|---|---|---|---|---|
| 4 | 1/512 | 8.7278343e-05 | 1.00122823 | completed 2T |
| 8 | 1/256 | 0.00034911217 | 1.00984122 | completed 2T |
| 16 | 1/128 | 0.0013964397 | 1.08055114 | completed 2T |
| 32 | 1/64 | 0.0055859409 | 1.80499679 | completed 2T |
| 64 | 1/32 | 0.022356812 | 26.3430661 | accepted to 1.75T; rejected at 1.78125T |
| 128 | 1/16 | 0.089944291 | 9.38113364 | accepted to 0.4375T; rejected at 0.5T |
| 256 | 1/8 | initial state only; rejected | 1 | accepted to 0T; rejected at 0.125T |

A rejected requested window has undetermined full-window stability. Its
accepted prefix is retained and any observed prefix amplification remains
evidence. An initial-state gain of 1 after first-window rejection is not a
completed-step result. Target-only high modes use absolute norms because their
initial relative gain is undefined.

## Controls and references

All 24 final controls/trajectories use the same clean numerical commit,
unchanged budgets and dt_guard=error. Main M=4,8,16,32,64,128,256 uses both
seeds. Four synchronous references use h=T/2048 and T/4096 over 2T. Four T0
cross-controls preserve H=T/16 or T/8 while halving h. Two prescribed-ion
controls observe T/2 without advancing P or creating a coupled energy ledger.

The actual nonzero-harmonic production 9x9 Jacobian includes pressure/acoustic
coordinates. The fastest full/frozen frequencies are 1.62306353356e8 and
1.53261330288e8 rad/s. At h* their omega*h values are 0.03213/0.03034, well
inside the RK4 imaginary-axis interval. The generic estimate is separately
recorded. Old T/64 and T/128 warnings are not demonstrated micro instability.

The two Jacobian perturbation scales differ by at most 2.08263e-8 s^-1 in
absolute matrix-entry norm; this is not a relative matrix error. The incomplete
m=0 longitudinal probe is retained and masked. Independent cold Hall vectors
and eigenvalues define the analytic reference. Production Jacobians and the
complete T0 macro maps are local linear evidence, not independent references
or nonlinear trajectories. Map powers and actual seed transients are reported
separately from spectral radii.

Full synchronous high-mode maximum vector errors over 2T are 1.16841e-6 at h*
and 7.30139e-8 at h*/2. Frozen-ion errors over T/2 are 2.19294e-7 and1.37060e-8;
P stays exactly fixed. Only the waveform view is cropped to 0.3T.

## Preserved failures and engineering repairs

Large macro energy/ion kinetic balance rejections retain the original 8e-9 J
budget, candidate arrays, checks, accepted state and cumulative ledger. At
H=T/8 the first candidate's ion kinetic balance defect is about2.2046e-8 J.
This is a balance defect, not the kinetic energy itself.

The first attempted matrix is retained separately. Its frozen h*/2 diagnostic
incorrectly requested storage for a fictitious whole retained predictor
window and stopped before stepping. The corrected manual diagnostic declares
one-step workspace with deterministic absolute clocks. Long failed-window
exports also exposed repeated full-record list membership; cached identity
buckets preserve exact record equality/order. Outer bookkeeping can still
rescan history; whole-export linear scaling is not claimed.

The final matrix reruns every case at one reviewed commit. Sixteen comparable
T0 cases preserve bitwise-identical accepted histories across the record-merge
repair; retained rejection checks match in content/order. Necessary gates,
physical RHS, quadrature, budgets and transaction rollback are unchanged.

Completed files retain the explicit COMPLETED_WITH_SOFT_FAILURES label,
including nonapplicable massless channels, empty periodic wall regions and
momentum/divergence convergence observations. These are not relabeled as
universal diagnostic passes. Legacy closed-initialization and m=0 issues are
retained separately; they do not automatically block T1.

## Artifacts, cost and next layer

- [Offline A0-style HTML gallery](/home/lxzeng/runs/fomhd/a1-benchmark__20261106T163530Z/a1-benchmark.html)
- [Chinese report](/home/lxzeng/fomhd-design/mainline_design/implementation/A1/A1_T0_REPORT.md)
- [Exact matrix commands and results](/home/lxzeng/runs/fomhd/a1-t0-controlled__20261106T163530Z/matrix-final/MATRIX_STATUS.json)
- [Compact plot input](/home/lxzeng/runs/fomhd/a1-t0-controlled__20261106T163530Z/matrix-final/plot-data.json)
- [Accepted-trajectory and failure-record equivalence](/home/lxzeng/runs/fomhd/a1-t0-controlled__20261106T163530Z/matrix-final/REPAIR_EQUIVALENCE.json)
- [Prior exploratory gallery](/home/lxzeng/runs/fomhd/a1-benchmark__20261106T130923Z/a1-benchmark.html)

The gallery reuses the unchanged A0 renderer/CSS and English plot conventions.
It keeps seven figures across eight pages, including full-horizon micro error,
macro accuracy, complete-map high-mode evolution, fixed-H refinements, raw
energy checks, linear maps, qualification and diagnostic-inclusive costs.
The old 1T warning points are separate from the new 2T baseline.

At N microsteps T0 uses 6N+N/M complete model evaluations versus 7N for
synchronous RK4; both remain O(N). T0 retains 4M quadrature stages per window.
Elapsed observations include diagnostics and recording, with concurrent CPU
runs and unequal accepted horizons for rejected cases. No equal-accuracy
speedup or isolated measured memory peak is claimed. This round did not rerun
GPU scientific scope; earlier GPU evidence remains historical.

Verification: 169 targeted CPU tests plus independent changed-path review;
actual gallery/provenance and delivery checks are recorded in the run index.
T1 is recommended for user review: preserve this input/horizon/criteria matrix,
first measure improvement at the same H, then test whether the usable H range
expands. T1-T4 each require their own report and user decision.
