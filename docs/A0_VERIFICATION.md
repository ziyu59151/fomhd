# A0 implementation and verification record

Date: 2026-11-04. Scope: the small FP64/complex128 C++20 reference solver and
explicit single-GPU CUDA backend, with a fixed closed staircase conductor/slip
boundary. This is a development verification record, not a release or an
experimental tokamak validation. No commit, tag, or push was made.

The authoritative continuous/discrete model remains in the sibling design
repository. Source and mirrored implementation pseudocode are maintained here.
The current raw evidence directory is `tmp/a0-completion-20261104/`.
The earlier `tmp/a0-implementation-20261104/` 68-test report is historical and
is not the current acceptance result.

## Implemented and repaired behavior

The native path covers geometry and topology, compatible RZ/Fourier operators,
ion scalar pressure, finite-mass CGL electron pressure, the coupled Darwin
field solve, CT magnetic evolution, explicit midpoint transactions, diagnostics,
strict input loading, snapshots, restart, and failure evidence. CUDA executes
the numerical primitives with cuBLAS/cuFFT/cuSOLVER; pointwise fluid assembly
remains on the host. The auxiliary Python analyzer now handles accepted-run
convergence and complex frequency/growth fits.

This completion pass repaired failure-stage identity, preservation of completed
stage evidence, source/combine exception attribution, no-throw result delivery
after commit, checkpoint-pointer reuse checks, and Slurm pipeline exit receipts.
It also completed shared spatial MMS operator assembly, bounded block right
solves, and offline analysis with active staggered masks, physical quadrature,
SI units, original-byte provenance checks, and consistent refinement identity.

The checked CPU/CUDA linear solves reuse the original LU for at most five FP64
residual corrections. Rank and componentwise backward-error budgets are
unchanged. Nonfinite updates, stagnation, or an exhausted correction bound
still reject. Snapshots record `derived_*_refinement_steps`. The original
168-by-168 failing system now takes one correction: componentwise error
3.0933e-16 and independently accumulated original-equation error 3.3734e-16,
against the unchanged 2.3874e-12 budget. The real disk CLI snapshot records one
Darwin correction at initialization.

No production arbitrary forcing option was added. Coupled temporal MMS uses a
separate test-only source callback. Its analytic reference trajectory determines
the prescribed source independently of the numerical state; source increments
have their own accepted ledger and diagnostic accounting.

## Verification evidence

| Check | Result and scope | Evidence under current raw directory |
|---|---|---|
| CPU Release suite | 84/84 passed after final integration, including both local dispersion tests; 82.32 s elapsed | `cpu-ctest-integrated.log`, `cpu-ctest-integrated-detail.log` |
| Auxiliary analyzer | 45 behavioral tests pass, including malformed identity, dimensions, units, zero signal and SI-time-offset controls | `analysis-final45.log`, `final-review/` |
| Coupled temporal MMS | All evolved variables converge at order 1.9341–2.0007, with nonzero longitudinal/transverse E and physical B RHS | `coupled-final-current.json` |
| Source-accounting sensitivity | All 16 diagnostic derivatives and 16 scalar interval accounting differences checked against independent centered differences | `source-controls-final.json`, `source-sensitivity/` |
| Spatial MMS | N=8,16,32; analytic staggered G/C/DB/DJ in cylindrical coordinates; inner, wall and global errors reported separately | `spatial-final-output.log` |
| Transactions and failure output | Negative controls exercise pre-commit, stage, interval, synchronization, source, signal and post-commit output failures | `runtime-round3-tests.log`, CPU suite runtime tests |
| Host sanitizer | 7/7 changed native contract tests pass with ASan and UBSan, leak detection enabled | `host-sanitize-tests.log` |
| CUDA | Numeric, complete stage evolution and capacity tests pass on RTX 5090; numeric regression executes one correction | `gpu-final.json`, `gpu-*-final.log` |
| Compute Sanitizer | Numeric/stage memcheck and initcheck report zero errors; memcheck reports zero leaked allocations | `gpu-memcheck-*-final.log`, `gpu-initcheck-*-final.log` |
| Local theoretical response | 7/7 continuum two-fluid/CGL/Darwin comparisons pass, including stable and unstable anisotropic controls | `dispersion-integrated-full.json`, `dispersion-integrated-preflight.json`, `final-review/dispersion-independent-math.json` |

Spatial inner orders are G=3.866/3.645, C=3.866/3.660,
DB=3.862/3.674 and DJ=4.188/4.283. These measured pre-asymptotic rates support
the specified high-order interior behavior; they do not establish global
fourth-order accuracy of a staircase wall. The irregular-mask tests cover
modes -2,0,+2 and noncommuting transform inversion. A tall-block allocation
regression verifies that the public right-solve workspace bound includes the
actual simultaneous allocations. No explicit transform inverse is formed.

The missing-ion-pressure-source control fails MMS accuracy as intended.
Independent review also demonstrated that the earlier diagnostic tests could
miss omitted source energy or zeroed Cartesian-x/angular rates. Repaired tests
now reject both retained mutations. Physical diagnostic budgets were not made
hard merely to make these tests fail.

GPU work ran in user Slurm job 10738 with the ordinary Toolkit 13.3 build using
`-lineinfo`. An immediate A100 allocation attempt was unavailable; the RTX 5090
allocation succeeded. This is neither an A100-current-build claim nor a speedup
measurement. No NCU counters, administrative access, or public configuration
changes were used.

## End-to-end runs and timing

The complete CLI was run with one shared geometry/initial-data file set,
K=1, Nphi=9, velocity amplitude 10 m/s, and final time 1.6e-7 s. Eight, sixteen
and thirty-two steps all complete with production hard checks enabled. The
offline ion toroidal-momentum Richardson estimate is 1.8746; differences
5.9355e-25 and 1.6186e-25 kg/(m^2 s) exceed the unchanged 1e-26 roundoff guard.
This is an unforced finite-window consistency check, not an exact physical
wave solution.

Run root:
`/home/lxzeng/runs/fomhd/closed-reference-dt-scan__20261104T041402Z/`.
Each run retains the exact inputs, executable/source provenance, diagnostics,
snapshots, and restart state. CPU end-to-end times are about 1.11, 1.89, and
3.58 seconds, including setup and I/O: one repetition, no warmup, single-thread
numerical libraries. These are diagnostic baseline timings, not optimized
throughput or a formal clean-release benchmark.

Earlier short-window data at 1.6e-8 s are retained. Their ion-momentum errors
were below the declared guard and are correctly marked UNAVAILABLE. An even
earlier scan generated identical numerical NPZ arrays separately, producing
different ZIP timestamps/file hashes; strict input identity rejected that
comparison. The accepted scan copies the same original files. No failed
comparison was relabeled PASS. A complex observation fit on the final run
returns WARN for a nonstationary envelope, not a claimed plasma eigenfrequency.
The final analyzer also produced `convergence-reviewed.json` and
`momentum-fit-reviewed.json` beside the originals. They preserve those outcomes
and include the analyzer source hash and NumPy version.

The real 6-by-6 nonrectangular reference input also completes eight CLI steps:
`/home/lxzeng/runs/fomhd/closed-disk6-runtime__20261104T041728Z/`.
Its accepted initial snapshot includes the actual Darwin correction count.
This verifies the run/I/O path on a staircase disk, not a smooth LCFS equilibrium.

## Local theoretical confirmation

The integrated local dispersion executable passes seven cases: warm parallel,
perpendicular and oblique response, CGL firehose with a stable control, and
oblique CGL mirror-like response with its own stable control. It uses an
explicitly test-only collocated periodic spectral adapter and compares the
production stage derivative to an independent continuum two-fluid/CGL/Darwin
linear matrix. This confirms the local equations against their theoretical
reference; it does not test production Yee/LCFS spatial accuracy or agreement
with experiments.

The full 14-by-14 matrix and its constraint invariants are checked before
the physical eigenvalue comparison. Magnetic divergence and transverse Ampere
constraints have rank three. An orthonormal nullspace gives an 11-dimensional
admissible eigenproblem; thermodynamic zero modes remain, and no mode is
discarded by its frequency or growth. Independent review rebuilt all three
Cartesian Ampere rows plus divB, verified rank three, and checked every lifted
eigenvector against the original matrix and constraints.

The worst normalized matrix discrepancy is 1.7283e-9 against a 1e-5 limit;
two-amplitude sensitivity is 2.5206e-9 against 1e-6. The largest mode-local
eigenvalue error divided by its allowed error is 0.01044, below one. Original
matrix invariant residuals are at most 1.8206e-14 against 1e-8. All 588 production
stage evaluations and 45,864 applicable hard checks pass. Reversed Lorentz
force, reversed anisotropic stress, an actual reversed-curl operator, zeroed
slow branches, and broken invariant derivatives are rejected by their gates.

Production growth rates are 8430.1103997 s^-1 for the parallel firehose and
1888.6106846 s^-1 for the oblique mirror-like branch. Their stable controls have
maximum numerical growth 5.3606e-5 and 0.0015848 s^-1, both below the fixed
0.01 s^-1 classification threshold. The oblique fixture was corrected using an
independent CGL compressive-stiffness calculation: its negative eigenvalue is
-0.02145417 Pa. The earlier candidate had positive stiffness and is retained
as failed fixture evidence. Neither this correction nor the admissible-space
eigenproblem relaxes the acceptance tolerances. These are fluid CGL branches,
not kinetic firehose/mirror threshold predictions.

## External equilibrium and remaining scientific scope

The user-supplied benchmark priorities are represented by the static reference,
parallel/perpendicular/oblique local response, parallel/perpendicular CGL
compression, and stable/unstable anisotropic controls. A cylindrical screw-pinch
or tokamak global eigenmode is not silently substituted by the local test:
those require a compatible current-carrying equilibrium and a separate spatial,
boundary and mode-shape validation study.

The GTAW `gfile_circular` audit is retained at
`/home/lxzeng/runs/fomhd/gtaw-equilibrium-audit-20261104/report.json`.
It parses a 129-by-129 equilibrium with 101 boundary points and approximately
0.48 m boundary radius. Its supplied pressure profile is zero, and the file
alone does not determine species densities/pressure split/finite-mass flows,
electric field, or a verified COCOS convention. It is parsing/input-audit
evidence only; no experimental two-fluid initialization was fabricated.

The A0 implementation remains a dense small-domain reference with explicit
memory rejection. Long-time stability, global smooth-boundary convergence,
physical eigenmode time series, nonlinear saturation, kinetic/hybrid comparisons,
experimental validation, sparse scalability, multi-GPU execution and NCU tuning
are not established by this record. A1–A3 are separate development stages.

## Integrated acceptance and reproducibility

The bounded A0 reference implementation, its listed verification gates, and
local theoretical comparisons are accepted for the scope above. This does not
freeze a scientific release or close the remaining global validation work.
DeepSeek Flash workers supplied bounded implementations; repeated stream
disconnects left some work incomplete. The interaction layer and native review
agents completed integration, repairs and independent acceptance. A worker's
success message was never used as a substitute for the integrated tests.

The final integrated CPU commands were:

```bash
cmake --preset cpu-release -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build --preset cpu-release --parallel 4
ctest --test-dir build/cpu-release --output-on-failure --parallel 2
build/cpu-release/fomhd_local_dispersion --preflight
build/cpu-release/fomhd_local_dispersion
```

The final two commands also retain complete JSON from the actual integrated
executable, rather than relying on a staged worker binary. The 84-test suite
includes the analyzer's 45 behavioral tests as one CTest entry; those counts
must not be added as if they were separate CTest cases.

`acceptance-identity-before-doc-sync.json` records the measured executable
hashes and preserves both build manifests before this documentation sync.
The tested CUDA executable SHA-256 is
`36773cb66c39ab0ec50ab54013e38d2390706f5d1d25102f4a4a491bb77b2259`.
`final-acceptance.json` connects the final source/documentation snapshot to the
test logs and verifies that the measured native sources and binaries did not
change during this sync. Regenerating a manifest is not a new GPU test.
All earlier failed runs and negative controls remain in the raw evidence tree.
