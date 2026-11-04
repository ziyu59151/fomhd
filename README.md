# FOMHD — C++ / CUDA A0 solver

FOMHD is implemented as a C++20 solver with an explicit CUDA backend. A0
prioritizes a runnable reference implementation, numerical diagnostics,
benchmarkable verification, and reproducible run artifacts. Kernel fusion,
reduced precision, and profiler-driven tuning are deferred until a later
optimization task.

The repository contains the A0 implementation-level pseudocode in
[`docs/pseudocode`](docs/pseudocode/README.md), the corresponding C++/CUDA
source tree, and an English implementation record. The design repository
`/home/lxzeng/fomhd-design` remains the authority for the mathematical model
and A0 boundary/discretization decisions; it is not a runtime dependency.

The current branch is `codex/a0-implementation`. It contains a complete small
CPU solver, a CUDA backend, NPZ/NPY input and output, restart support, failure
evidence, and diagnostic contracts. It has not received a production-scale
experimental validation or an NCU optimization study.

## Build and test

The native build uses CMake, Ninja, C++20, FFTW, MKL, OpenSSL, and the bundled
TOML/JSON/NPZ support. The CUDA build uses Toolkit 13.3 with cuBLAS, cuFFT, and
cuSOLVER. Use the checked-in presets where possible:

```bash
cmake --preset cpu-release -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build --preset cpu-release --parallel 4
ctest --preset cpu-release --output-on-failure
```

The CUDA target is configured separately on a host with the required Toolkit:

```bash
cmake --preset cuda-release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build --preset cuda-release --parallel 4
```

GPU execution must run inside the user's own Slurm allocation. The CUDA path
accelerates the numerical primitives and FFT/linear algebra calls; the
pointwise fluid assembly remains host-side in A0. This is an implementation
baseline, not a claim of an all-device optimized solver.

## Python and PyTorch

`pyproject.toml`, `uv.lock`, and `.python-version` describe only auxiliary
Python tools for input generation, cross-format checks, and offline analysis.
The C++ solver does not require Python, LibTorch, or PyTorch at runtime. The
project-local `.venv` is isolated from the public server environment; it does
not grant or change GPU permissions.

Small [plotting scripts](tools/plots/README.md) render benchmark convergence,
local eigenmode comparisons, and accepted-run diagnostics as PNG/PDF with
plotted data and provenance. Install their optional dependencies with
`uv sync --frozen --group plot`; plotting remains an offline Python activity.

## Repository layout

```text
include/fomhd/       Public C++ types and interfaces
src/                 Geometry, operators, fluid, fields, boundary, time, I/O, CUDA
app/                 fomhd solver command-line program
tests/               Unit, integration, runtime, verification, and CUDA tests
tools/               fomhd_make_case, offline analysis and equilibrium audit tools
docs/pseudocode/     Implementation-level specification aligned with the source tree
cmake/               Build manifest and reproducibility helpers
build/               Ignored build trees
tmp/                 Ignored verification evidence and regenerated logs
```

Runtime directories are kept under `/home/lxzeng/runs/fomhd/`. Each run has a
distinct, tab-completion-friendly case prefix followed by a timestamp and
unique suffix. A run stores its exact inputs, source snapshot, executable
hash, resolved configuration, logs, diagnostic records, snapshots,
checkpoints, and failure evidence.

## Version policy

Development uses short-lived `codex/<task>` branches and merges into `master`
after review and verification. A stable `release/aN` branch and immutable
`aN-vX.Y.Z` tag are created only after that version's stated benchmark scope
has passed. A later boundary or physics change is a new maintenance change
with its own pseudocode, compatibility statement, and evidence; it does not
rewrite an earlier tag.

The source tree, build manifests, and documentation are versioned. Build
directories, virtual environments, run outputs, literature, and design
repository data are excluded from the source repository.

## Current evidence and limits

The current [A0 verification record](docs/A0_VERIFICATION.md) reports 84/84
integrated CPU CTests, coupled temporal and staggered spatial manufactured
solutions, restart/failure controls, CUDA numerical/stage checks, and zero-error
NVIDIA memcheck/initcheck. Seven local two-fluid/CGL/Darwin theory comparisons
pass, with independent reference matrices and stable/unstable controls.
Raw evidence is in `tmp/a0-completion-20261104/`.

The solver remains a dense small-domain reference. Interior spatial accuracy
does not establish global fourth-order accuracy at the staircase wall. Local
theory comparisons do not establish a global tokamak mode or experimental
validation. The GTAW G-EQDSK file is only an input audit. Optimization, sparse
scalability, long-time nonlinear studies and A1–A3 remain separate work.
