# FOMHD A0 Python/Warp mainline

[Visual branch history](docs/BRANCH_GRAPH.md) · [Development policy](docs/development.md)

[Rectangular conducting-wall baseline](docs/benchmarks/rectangular-wall/README.md) ·
[Complete fields and comparison figures](docs/benchmarks/rectangular-wall/GALLERY.md)

The default development baseline uses E/B vectors on the grid-aligned RZ
conductor with the paired FOMHD Ampère closure and synchronous RK4. The
next acceptance is a reproducible GPU A0 with measured capacity and a
matched independent n_phi=1 compressional Alfvén mode. A2 passive particle
development follows that acceptance.

[Physical simulation prototype workflow](docs/RESEARCH_PROTOTYPE_WORKFLOW.md)
records how verification, independent physical comparisons, and retained known
issues determine prototype development priorities.

The A0 maintenance implementation is the FP64/complex128 Python solver under
[`src/fomhd/`](src/fomhd), using NVIDIA Warp 1.19.0, CuPy CUDA13x 14.2.0, and
an explicit NumPy/SciPy CPU reference. The fully explicit RK4 fluid path is
the default; `time.integrator = "midpoint"` remains available when a test or
study explicitly requests midpoint. The massless quasineutral branch is the
default: omitted `species.electron_inertia` resolves to `0`, while finite
electron inertia requires explicit `1`. The benchmark harness explicitly
retains `1` as a historical finite-electron local workload.

The C++20/CUDA solver baseline is `48beca7` on `reference/a0-native`. It is a
separate implementation branch and is not duplicated in this source tree. The
historical native verification record is in
[A0_VERIFICATION.md](docs/A0_VERIFICATION.md); the native implementation and
its native-only tests and generators remain on that reference branch.

The mathematical authority is the sibling repository
`/home/lxzeng/fomhd-design`: `DECISIONS`, `A0_IMPLEMENTATION_SPEC`, and
`A0_GLOBAL_DISCRETIZATION`. It is review context, not a build or runtime
dependency. The implementation-level pseudocode under
[`docs/pseudocode/src/fomhd`](docs/pseudocode/src/fomhd) is synchronized with
the active source and tests.

The scoped global fast-branch prototype evidence and limits are in
[A0_TOKAMAK_CAE_EVIDENCE.md](docs/A0_TOKAMAK_CAE_EVIDENCE.md); retained debugging
leads are in [A0_KNOWN_ISSUES.md](docs/A0_KNOWN_ISSUES.md).

## Environment and execution

Use the project-local environment from an exact checkout:

```bash
uv sync --frozen --group dev --group plot
uv run python -m fomhd --help
```

The package uses the conventional `src/` layout and the `uv_build` backend
pinned to the repository's uv version. Do not install the solver with global
pip. The project-local `.venv` is isolated from the public server environment
and does not grant or change GPU permissions.

Validate or run a case:

```bash
uv run python -m fomhd validate /absolute/path/to/case.toml \
  --output /home/lxzeng/runs/fomhd/a0-validate__20261105T120000Z
uv run python -m fomhd run /absolute/path/to/case.toml \
  --output /home/lxzeng/runs/fomhd/a0-case__20261105T120000Z
```

The solver has no native builder dependency. `tools/warp_benchmark.py`
generates its controlled closed-reference geometry, state, and case through
the production geometry, operator, stage, and I/O functions.

## Tests and independent references

The CPU suite runs with one BLAS thread and warnings treated as errors:

```bash
OPENBLAS_NUM_THREADS=1 \
FOMHD_NATIVE_ORACLE_DIR=/home/lxzeng/projects/fomhd/build/native-oracles \
uv run python -W error -m pytest -q
```

`FOMHD_NATIVE_ORACLE_DIR` is optional and contains already exported independent
native arrays used only by parity tests. The solver and ordinary benchmark case
generation never read native binaries or oracle arrays. Missing oracle-only
comparisons remain explicitly unavailable; required CPU tests do not depend on
the native build.

The CUDA backend requires an owned Slurm allocation. It uses FP64/complex128
arrays, Warp kernels, and CuPy FFT/LU/SVD without silent CPU fallback. A device,
stream, dtype, or shape mismatch is an error. No GPU acceptance or performance
claim is made by the CPU checks in this source checkout.

## Repository layout

```text
src/fomhd/                 active Warp/CuPy solver and CPU reference
tests/warp/                actual-stage, routing, MMS, runtime and parity tests
tests/test_analyze.py      offline run-analysis behavioral tests
tests/test_geqdsk.py       equilibrium-input parser tests
tools/                     Python benchmark, verification, analysis and plots
cases/                     case-generation and reference-fixture guidance
jobs/a0.slurm              user-owned frozen-checkout GPU launcher
docs/pseudocode/           bilingual specification mirrored under src/fomhd
```

Build products, virtual environments, run products, literature, and design
repository evidence stay out of the source tree. Personal runs are created
under `/home/lxzeng/runs/fomhd/` with a recognizable case prefix, UTC timestamp,
and unique suffix. Each run retains exact inputs, source and lock hashes,
diagnostics, snapshots, checkpoints, restart linkage, and failure evidence.

## Branch and version policy

Development uses short-lived `codex/<task>` branches and merges into `main`
after review and verification. The native reference remains on
`reference/a0-native`. Release branches and immutable tags are created only for
a stated, passed benchmark scope. Do not force-push, rewrite tags, or manifest
a different implementation merely to make two branches look equivalent.

Formal benchmark runs use a clean source commit. Debug runs may be dirty only
when provenance retains the exact source snapshot. A running job uses a frozen
source checkout and its matching lock and environment; those paths are not
overwritten while the job is active.

## Evidence and limits

[WARP_IMPLEMENTATION.md](docs/WARP_IMPLEMENTATION.md) records the migration
order and existing bounded CPU/GPU evidence. [A0_RUNTIME_GUIDANCE.md](docs/A0_RUNTIME_GUIDANCE.md)
records timestep and model guidance. [A0_VERIFICATION.md](docs/A0_VERIFICATION.md)
is a historical native-reference record, not evidence that the native tests
run from this branch.

Existing results remain historical and are not rewritten as new acceptance.
The solver is a dense small-domain correctness reference. Local analytical,
MMS, restart, conservation, and CPU/GPU comparisons do not establish global
tokamak accuracy, production equilibrium validation, optimized performance, or
all-branch scientific acceptance. Performance tuning, reduced precision,
autodiff, NCU work, and new physics are separate tasks.
