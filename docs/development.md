# FOMHD development and release policy

> Status: active A1 Python/Warp implementation record. The implementation is
> runnable with bounded CPU/GPU evidence; it has not received production-scale
> scientific validation.

A1 inherits A0 in the same source tree. `release/a0` at `9a38bf4` is the
independent A0 maintenance baseline; the retained local checkout is
`/home/lxzeng/.codex/worktrees/fomhd-a0-maintenance/fomhd`. Compare exact commits
with identical inputs and dependencies. T0 and T1 have bounded delivered
evidence; D070 adds bounded fixed-ion T2-E1/E2/E3 evidence. D068 revises the time route to T2-E field windows, T2-I ion prefixes, T2-C
integration and T3 scaling. Each unit needs CPU reference, GPU parity and
same-GPU performance attribution. The current branch is
`codex/a1-t2-electron-field`; no particle or complete coupled scheme is added. See
[A1_IMPLEMENTATION.md](A1_IMPLEMENTATION.md) and [ROADMAP.md](ROADMAP.md).

The mathematical authority is the sibling design repository:
`DECISIONS`, `A0_IMPLEMENTATION_SPEC`, and `A0_GLOBAL_DISCRETIZATION`. Design
materials are review context, not a build or runtime dependency. All active
source, comments, tests, diagnostics, logs, and implementation records are
English. The separately maintained bilingual pseudocode is the sole exception.

## 1. Implementation and environment

`main` has one solver implementation: Python with NVIDIA Warp kernels and CuPy
CUDA libraries under `src/fomhd/`. NumPy/SciPy exercise the explicit CPU
reference. The selected time path is fully explicit RK4 by default; midpoint
is retained for explicitly requested comparisons and historical controls.
FP64 state and complex128 retained arrays are requirements. There is no reduced
precision, autodiff, implicit electron update, or performance claim in ordinary
correctness work.

The C++20/CUDA implementation baseline is `48beca7` on `reference/a0-native`.
Native source, CMake targets, native-only tests, generators, and native
pseudocode remain on that branch. `main` does not carry a second source copy or
a mandatory native build dependency.

`pyproject.toml`, `uv.lock`, and `uv_build` own the project package and
auxiliary Python tools. Warp is pinned to 1.19.0 and CuPy CUDA13x to 14.2.0.
PyTorch is auxiliary and is not imported by the solver. Use only the
project-local environment:

```bash
uv sync --frozen --group dev --group plot
uv run python -m fomhd --help
```

Do not install the solver with global pip. Build products, virtual
environments, run products, literature, and design evidence stay outside the
source tree.

GPU placement follows data ownership. Warp kernels and CuPy arrays remain on
one selected device and stream; explicit synchronization occurs at declared
dependency and observation boundaries. Configuration, geometry setup, scalar
metadata, and file I/O are host activities. There is no silent CPU fallback.
GPU work runs only in the user's own Slurm allocation.

## 2. Source tree and ownership

```text
src/fomhd/                  active Warp/CuPy solver and NumPy/SciPy CPU reference
tests/warp/                 actual-stage parity, routing, MMS and runtime tests
tests/test_analyze.py       offline run-analysis tests
tests/test_geqdsk.py        equilibrium parser tests
tools/                      Python benchmark, verification, analysis and plots
jobs/                       user-owned Slurm launchers
cases/                      case-generation guidance
docs/pseudocode/src/fomhd/  bilingual implementation pseudocode
```

CuPy owns complex128 buffers. Warp receives explicit float64 component views
with documented order, layout, stream, and lifetime. Geometry tables and static
metadata are host setup. Numerical modules do not perform file I/O. The time
layer owns the accepted/rejected transaction; diagnostics are read-only; shared
physical fluxes and currents have one producer. The I/O layer publishes run
metadata, snapshots, checkpoints, and failure records.

`RHS` means the complete semi-discrete evolution derivative, not only an
external source term. Interfaces state units, shape, index order, stage
identity, ownership, host/device placement, synchronization, and failure
behavior.

## 3. Pseudocode, traceability, and provenance

The implementation chain is:

```text
continuous equations -> discrete algorithm -> implementation pseudocode
-> Python/Warp/CuPy source -> independent verification evidence
```

`docs/pseudocode/TRACEABILITY.tsv` maps active Python step IDs to the actual
`src/fomhd` file and its tests. Deleted native paths are not retained as broken
main-branch links; their historical mapping remains on
`reference/a0-native`. `SOURCE_BASELINE.json` records design-source authority
hashes without making the design repository a runtime dependency.

Run provenance is generated from the imported package, not from the current
working directory. The imported package must resolve under the exact
checkout's `src/fomhd`. Snapshots use the same `src/fomhd/...` relative paths
and hash the actual imported source plus `pyproject.toml` and `uv.lock`. A
runtime root that does not match the imported package is rejected.

## 4. Branches, tags, and reproducibility

Development uses short-lived `codex/<task>` branches and merges into `main`
after review and verification. The native reference is
`reference/a0-native`. Release branches and immutable tags are created only
after a stated benchmark scope has passed. Do not force-push, rewrite tags, or
manufacture history to make implementations look equivalent.

The A3-to-1.0 and later A4 roadmap does not create release branches or tags.
Documentation reviews use a `codex/<task>` branch and preserve existing A0/T0/T1
maintenance references. A3 release acceptance must name the actual model,
benchmarks, budgets and reproducible evidence before version 1.0 is created.

Formal benchmark runs use a clean source commit. Debug runs may be dirty only
when the manifest retains the exact source snapshot. A running job uses a
frozen Python/Warp checkout and matching lock and environment; those paths are
not overwritten while the job is active.

The Slurm launcher records the resolved project, Python executable, package and
lock hashes, Python/CuPy/Warp versions, case hash, Slurm job identity, visible
device, host, timestamps, and exit status. The solver run independently retains
its exact inputs and runtime-generated source snapshot.

## 5. Run products and restart rules

Use `/home/lxzeng/runs/fomhd/` as the personal run root. Put a recognizable case
prefix first and the UTC timestamp/unique suffix last:

```text
/home/lxzeng/runs/fomhd/a0-wave_reference__20261105T120000Z__7c21/
```

Each run is single-writer and refuses a non-empty output directory. It stores:

- `inputs/`: exact case, geometry, initial state, source manifest, and resolved paths;
- `provenance/`: runtime source snapshot, project metadata, lock, and versions;
- `logs/`, `history.csv`, `checks.jsonl`, and `observations.jsonl`;
- `snapshots/` and `checkpoints/` with atomic publication and SHA-256 pointers;
- `failure/`: structured numerical/I/O metadata and matching rejected arrays
  when available.

A restart reads only an accepted checkpoint with matching semantic, geometry,
operator, schema, and time fingerprints. It creates a new run directory and
records a parent link. It never overwrites or replays the previous run.

## 6. Verification and validation workflow

Use the smallest independent test that proves a change, then run the complete
required scope before reporting a result:

```bash
OPENBLAS_NUM_THREADS=1 \
FOMHD_NATIVE_ORACLE_DIR=/home/lxzeng/projects/fomhd/build/native-oracles \
uv run python -W error -m pytest -q
```

The oracle directory supplies already exported independent arrays only to
parity tests. Missing external oracle evidence is reported as unavailable; it
does not disable required core tests. Native arrays must not be regenerated by
Python and relabeled native.

For numerical source changes, run actual-stage analytical/MMS, transaction,
restart, and exact source/lock provenance checks. For CUDA changes, run the
applicable tests in the user's Slurm allocation and use Compute Sanitizer where
the task requires it. Preserve commands, versions, raw output, and job identity
under a dated run directory.

Report verification separately from scientific validation. The historical
native record in `A0_VERIFICATION.md` is not rerun by this branch. No current
check establishes experimental tokamak validation, production-scale
equilibrium validation, global staircase-wall fourth-order accuracy, or
optimized performance.

## Branch visualization

Open [the visual branch history](BRANCH_GRAPH.md) directly on GitHub. It is a
Mermaid view of real commit parents and every fetched branch head, refreshed
automatically on main changes and hourly for older branches. It is a bounded
snapshot, preceding its own graph-only commit. This does not require a browser
plugin or public GitHub Pages. GitHub Network is also linked there; private
repository availability depends on the account plan. VS Code Git Graph is an
editor extension, while gitgraph.js is a diagram library; neither installs a
new GitHub website interface.

For this checkout, `git graph` invokes `git log --graph --decorate --oneline
--all --date-order`. On another clone, install that repository-local alias:

```bash
git config alias.graph 'log --graph --decorate --oneline --all --date-order'
git fetch origin --prune
git graph
```

The native reference branch additionally retains the independent oracle
exporter imported unchanged from `d9390ac`; its solver baseline remains
`48beca7`. Python core tests need no native build. Set
`FOMHD_NATIVE_ORACLE_DIR` only for the optional independent parity tests.

## T2 development units and evidence boundary

Use `codex/a1-t2-electron-field` for the bounded E1/E2/E3 implementation, and
separate subsequent ion and integration branches with
one source package, synchronized pseudocode and exact benchmark commits. Preserve
`release/a0`, `release/a1-t0`, `release/a1-t1` and their independent checkouts.
Do not relabel documentation changes as new numerical evidence. Spatial backend
changes use separately verified commits so their benefits can be distinguished
from time-algorithm changes. No merge, release or tag is implied by this plan.
