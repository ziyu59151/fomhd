# FOMHD development and release policy

> Status: active A0 implementation record.
> The implementation is runnable and has CPU/GPU correctness evidence; it has
> not received production-scale scientific validation.
> Authority: the design repository decisions D045/D053–D060 and the A0
> implementation/discretization specifications.

Update this file and the matching pseudocode whenever the source tree, toolchain,
module contract, branch policy, or run artifact schema changes. All source
comments, tests, diagnostics, logs, and implementation records are written in
English. The pseudocode directory is an intentional exception because its
bilingual edition is maintained separately.

## 1. Language, build, and auxiliary environments

The active solver uses Python, NVIDIA Warp, and CuPy; NumPy/SciPy provide the
explicit CPU reference. The frozen C++20/CUDA baseline remains available at
`reference/a0-native`. Standard-library code and established numerical
libraries are preferred for the initial implementation. CUDA
correctness support is part of A0; performance tuning, fusion, reduced
precision, and NCU-based optimization remain a separate later phase.

CMake, Ninja, and CTest own native targets and dependencies. The current build
baseline is GNU 15.2.0, CPU Release/Debug configurations, FFTW, MKL, OpenSSL,
TOML++, nlohmann/json, bundled miniz, and CUDA Toolkit 13.3 for the CUDA target.
The build manifest records the executable hash, source snapshot, compiler, and
resolved library versions. No dependency manager downloads native libraries at
configure time.

| File or directory | Responsibility |
|---|---|
| `CMakeLists.txt`, `cmake/` | Native targets, compile flags, dependencies, and build manifest |
| `CMakePresets.json` | Shareable CPU/CUDA and Debug/Release configurations |
| `CMakeUserPresets.json` | Optional machine-local paths; never commit it |
| `pyproject.toml`, `uv.lock`, `.python-version` | Project-local Warp/CuPy, Python tests and auxiliary PyTorch/input/analysis tools |
| `build/<preset>/` | Ignored CMake cache, objects, binaries, and evidence |
| `.venv/` | Reconstructible project-local Python solver environment; never committed |

The native C++ solver does not start through uv and does not link LibTorch. PyTorch is
available only for reference, input processing, and offline analysis. Its CUDA
user-space libraries do not replace the public CUDA Toolkit or grant device
access. GPU scheduling and permission remain controlled by Slurm and the
server administrator.

## 2. Source tree and ownership

```text
fomhd/                     Warp/CuPy implementation and NumPy/SciPy CPU reference
tests/warp/                Actual-stage parity, model routing, MMS and runtime tests
include/fomhd/             Public interfaces and data types
src/core/                  State, tags, allocation, and failure ownership
src/geometry/              RZ mesh, masks, entities, weights, and topology
src/numerics/              Compatible operators, Fourier transforms, dense solves
src/fluid/                 Ion and finite-mass CGL electron equations
src/fields/                Darwin electric solve and CT magnetic update
src/boundary/              Fixed closed staircase wall constraints and ledgers
src/time/                  Stage evaluation and explicit midpoint transactions
src/diagnostics/            Read-only checks, budgets, regions, and totals
src/io/                    Configuration, NPZ/NPY, run metadata, checkpoints
src/cuda/                  cuBLAS/cuFFT/cuSOLVER adapters and required kernels
app/                       `fomhd` solver command-line program
tests/                     Unit, integration, runtime, V&V, and CUDA tests
tools/                     `fomhd_make_case`, offline analysis and equilibrium audit
jobs/                      User-owned Slurm launch scripts when needed
docs/pseudocode/            Implementation-level specification by source path
```

The A0–A3 line shares one set of spatial, field, fluid, and I/O modules. A1
changes time scheduling rather than copying A0. A2 and A3 add explicitly
registered consumers or producers with their own source ownership contracts.
Native RAII and Python backend owners manage allocations/library handles;
CuPy owns complex128 buffers shared as float64 views with Warp. Borrowed views cannot
outlive their owner. Numerical modules do not perform file I/O. The time layer
owns accepted/rejected state transactions, diagnostics return records, and the
I/O layer publishes records and checkpoints.

`RHS` means the complete semi-discrete evolution derivative: flux divergence,
pressure and magnetic forces, electric response, wall reaction, and other
terms required for a state update. It is not limited to an externally imposed
physical source term.

## 3. Pseudocode, source mapping, and traceability

The implementation chain is:

```text
continuous equations -> discrete algorithm -> implementation pseudocode
-> Warp/CuPy or native C++/CUDA source -> independent verification and validation evidence
```

Every pseudocode block specifies units, shapes, index order, interleaved
locations, ownership, host/device placement, library calls, synchronization,
failure retention, and independent tests. `docs/pseudocode/TRACEABILITY.tsv`
connects stable step IDs to source files, design authority, and test names. It
now records implementation and CPU/GPU verification status separately from the
remaining scientific-validation scope.

Pseudocode is an implementation specification and maintenance guide, not a
proof of correctness. Manufactured solutions, convergence, conservation,
restart equivalence, negative controls, and independent input checks remain
mandatory. Changes discovered by those checks must update the pseudocode and
source in the same change.

## 4. Branches, tags, and reproducibility

Short-lived task branches use `codex/<task>`. Verified changes merge into
`master`. After an A0, A1, A2, or A3 benchmark scope is explicitly frozen, create
an immutable tag such as `a1-v1.0.0` and, only if maintenance is needed, a
`release/a1` branch. A later boundary or physics change is a new commit on the
maintenance branch; the original tag remains unchanged.

A reproducible run records the source commit and dirty state, the complete
bounded source snapshot used by the build, executable SHA-256, compiler flags,
dynamic libraries, case/config/input hashes, semantic and operator fingerprints,
precision, backend, device, and Slurm job ID. A formal benchmark must use a
clean commit. Debug runs may be dirty only when the manifest retains the exact
patch or source snapshot.

Different versions use different clones or Git worktrees and build directories.
A running job uses a frozen executable or Python source checkout and matching
libraries; its source/build path is never overwritten while it runs. No commit, tag, or remote push is
created automatically by the solver workflow.

## 5. Run products and restart rules

Use `/home/lxzeng/runs/fomhd/` as the personal run root. Put a recognizable case
prefix first and the timestamp/unique suffix last, for example:

```text
/home/lxzeng/runs/fomhd/a0-wave_reference__20261104T120000Z__7c21/
```

Each run is single-writer and refuses a non-empty output directory. It stores:

- `inputs/`: exact case, geometry, initial state, source manifest, and resolved paths;
- `provenance/`: executable, build manifest, and verified source snapshot;
- `logs/`, `history.csv`, `checks.jsonl`, and `observations.jsonl`;
- `snapshots/` and `checkpoints/` with atomic publication and SHA-256 pointers;
- `failure/`: structured numerical/I/O metadata and, when available, a matching
  NPZ containing the rejected stage arrays.

A restart reads only an accepted checkpoint with matching semantic, geometry,
operator, schema, and time fingerprints. It creates a new run directory and
records a parent link; it never overwrites or replays the previous run. A
checkpoint is authoritative for recovery, not the number of CSV rows.

## 6. Verification and validation workflow

Use the smallest independent test that proves a change, then run the complete
required scope before reporting a result:

For Warp changes, run `OPENBLAS_NUM_THREADS=1 .venv/bin/python -W error -m
pytest -q tests/warp`, then the applicable CUDA tests with `FOMHD_TEST_GPU=1`
inside an owned Slurm job. Numerical source changes require actual-stage
analytical/MMS checks and exact source/lock provenance. The explicit RK4
control has four physical stages; algebraic field solves at those stages
do not make the time integrator implicit.

1. Configure and build the affected CPU target.
2. Run the focused regression or negative-control test.
3. Run the full CPU Release CTest suite.
4. For CUDA changes, run the CUDA numeric/stage comparison in the user's Slurm
   allocation and use Compute Sanitizer for memory and initialization checks.
5. Preserve commands, versions, raw output, and machine/job identity under a
   dated `tmp/` evidence directory.
6. Report verification (implementation contracts and numerical checks) separately
   from validation (agreement with physical or experimental reference data).

The current [A0 verification record](A0_VERIFICATION.md) includes 84/84 passing
CPU CTests, coupled temporal and staggered spatial manufactured solutions,
seven local two-fluid/CGL/Darwin theory comparisons, CPU/CUDA stage comparison,
and zero-error Compute Sanitizer memcheck/initcheck. The analyzer's 45 behavioral
tests are included in one CTest entry. Current raw evidence is retained under
`tmp/a0-completion-20261104/`; the earlier 68-test result is historical.
No experimental tokamak validation, production-scale equilibrium validation,
global fourth-order staircase-wall accuracy or optimized performance is claimed.
