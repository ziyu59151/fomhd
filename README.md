# FOMHD — Warp/CuPy A0 solver and native reference

The current A0 development uses Python with NVIDIA Warp and CuPy on CUDA,
with an explicit NumPy/SciPy CPU reference. The C++20/CUDA implementation is
preserved at `reference/a0-native` (`48beca7`). A0
prioritizes a runnable reference implementation, numerical diagnostics,
benchmarkable verification, and reproducible run artifacts. Kernel fusion,
reduced precision, and profiler-driven tuning are deferred until a later
optimization task.

The repository contains the A0 implementation-level pseudocode in
[`docs/pseudocode`](docs/pseudocode/README.md), the corresponding C++/CUDA
source tree, mirrored Warp pseudocode, and an English implementation record. The design repository
`/home/lxzeng/fomhd-design` remains the authority for the mathematical model
and A0 boundary/discretization decisions; it is not a runtime dependency.

Changes reach `master` after independent review and verification.
[The migration record](docs/WARP_IMPLEMENTATION.md)
defines finite-mass equivalence, the optional quasineutral model, explicit
integrators, and acceptance scope. No production-scale experimental validation
or NCU optimization is claimed.

## Warp environment and checks

Use the project-local environment from a repository checkout:

```bash
uv sync --frozen --group dev --group plot
.venv/bin/python -m fomhd --help
```

The full verification suite uses independent arrays exported by the native
CPU implementation. After the native build below, generate them once into a
new directory (the exporter refuses overwrite):

```bash
mkdir -p build/native-oracles
build/cpu-release/fomhd_export_native build/native-oracles/native-oracle-4-k1.npz 4 1 0.01 0
build/cpu-release/fomhd_export_native build/native-oracles/native-oracle-stress.npz 4 1 0 1
OPENBLAS_NUM_THREADS=1 .venv/bin/python -W error -m pytest -q tests/warp
```

Set `FOMHD_NATIVE_ORACLE_DIR` to use another archived native dataset. The
solver itself never reads these test arrays.

The CUDA backend requires an owned Slurm allocation. It uses FP64/complex128
arrays, Warp kernels, and CuPy FFT/LU/SVD without silent CPU fallback. The
complete stage keeps numerical arrays on the selected device; geometry setup,
scalar diagnostics, and explicit I/O are host activities. This is a dense
correctness reference, without a performance advantage claim.

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

`pyproject.toml`, `uv.lock`, and `.python-version` manage the Warp/CuPy solver
and auxiliary input/analysis tools. Warp is pinned to 1.19.0 and CuPy CUDA13x
to 14.2.0. PyTorch remains an auxiliary library; the Warp solver does not use
autograd or require a PyTorch tensor path.
The C++ solver does not require Python, LibTorch, or PyTorch at runtime. The
project-local `.venv` is isolated from the public server environment; it does
not grant or change GPU permissions.

Small [plotting scripts](tools/plots/README.md) render benchmark convergence,
local eigenmode comparisons, and accepted-run diagnostics as PNG/PDF with
plotted data and provenance. Install their optional dependencies with
`uv sync --frozen --group plot`; plotting remains an offline Python activity.

## Repository layout

```text
fomhd/               Warp/CuPy solver, explicit CPU reference and command-line entry
include/fomhd/       Public C++ types and interfaces
src/                 Geometry, operators, fluid, fields, boundary, time, I/O, CUDA
app/                 fomhd solver command-line program
tests/               Unit, integration, runtime, verification, and CUDA tests
tests/warp/          Warp CPU/GPU parity, transactions, convergence and model checks
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

The [Warp implementation record](docs/WARP_IMPLEMENTATION.md) distinguishes
CPU/CUDA execution, finite/massless model checks, local Hall and longitudinal
waves, coupled temporal MMS, and unresolved global spatial accuracy. The
finite-mass longitudinal control demonstrates numerical coarse-step
amplification against an independent analytic mode and integrator polynomial;
it does not attribute a production-wall rejection to Langmuir instability.

The native [A0 verification record](docs/A0_VERIFICATION.md) reports 84/84
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
