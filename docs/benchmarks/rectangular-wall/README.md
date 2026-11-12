# Grid-aligned conducting wall benchmark

This is the D075 rectangular RZ baseline for FOMHD's electric and magnetic
vector fields. It pairs the half-link magnetic moment correction with the
Faraday curl and the existing conducting-wall constraints. The HYM evidence
motivates the geometry. The FOMHD closure, its accuracy and its verified
scope are given in the [boundary design](DESIGN.md) and
[benchmark report](REPORT.md).

[Complete comparison gallery](GALLERY.md) ·
[Frozen input identity](evidence/input-identity.json) ·
[Analytic and numerical evidence](evidence/)

The historical and repaired A0 calculations use the same physical input.
The nonlinear replay below covers axisymmetric quiet and pulse
evolution. The retained `m=0,±1` control checks modal algebra and
normal-current enforcement. A separate benchmark is needed for nonlinear
nonaxisymmetric mode accuracy and scale.

## Source revisions and exact replay inputs

| Source | Commit | Input directory | Accepted window |
| --- | --- | --- | --- |
| A0 before the paired curl repair | [`60a2252`](https://github.com/ziyu59151/fomhd/commit/60a2252fdce390780139bf18e7279fe850706188) | Same physical 14×20 quiet and pulse inputs as repaired A0 | 8 ns matched prefix |
| A0 paired curl repair | [`bc809bd`](https://github.com/ziyu59151/fomhd/commit/bc809bdd3aa5253e1ffa60c21e7dcf5e7c84d369) | [14×20 quiet](inputs/a0-repaired-14x20-quiet) and [14×20 pulse](inputs/a0-repaired-14x20-pulse) | 160 synchronous 50 ps steps, 8 ns |
| A1 paired curl repair | [`e8fb707`](https://github.com/ziyu59151/fomhd/commit/e8fb707507a8f7a018193e848c5420b79f2171ad) | [10×14 RK4 pulse](inputs/a1-repaired-10x14-sync-pulse) and [10×14 WR pulse](inputs/a1-repaired-10x14-wr3-pulse) | 64 synchronous steps or 16 waveform macros, 3.2 ns |

The case TOML and both input NPZ files preserve the archived bytes. The
descriptive source manifests now use the relative
[FreeGS equilibrium export](reference/freegs-equilibrium.npz), also preserved
byte for byte. The older trajectory used a longer run, and the comparisons
read its exact 8 ns prefix. Separate benchmark launches can report different
checkpoint hashes when the source manifest or output path changes. Compare
the numeric arrays and diagnostics at exact accepted times.

From the checkout that contains this benchmark directory, prepare a fresh
workspace for outputs and exact-source worktrees.

```sh
bench="$PWD/docs/benchmarks/rectangular-wall"
work="$(mktemp -d "${TMPDIR:-/tmp}/fomhd-rectangle.XXXXXX")"
git worktree add --detach "$work/a0-before" 60a2252fdce390780139bf18e7279fe850706188
git worktree add --detach "$work/a0" bc809bdd3aa5253e1ffa60c21e7dcf5e7c84d369
git worktree add --detach "$work/a1" e8fb707507a8f7a018193e848c5420b79f2171ad
for checkout in a0-before a0 a1; do
  (cd "$work/$checkout" && uv sync --frozen --group dev --group plot)
done
```

### Analytic CPU control

Run both operator revisions using the project virtual environment and one
BLAS thread. The generated JSON and figures go into the external workspace.

```sh
mkdir "$work/analytic"
for pair in a0-before:archived a0:repaired; do
  checkout="${pair%:*}"
  label="${pair#*:}"
  for script in analytic.py modal_checks.py cold_hall_spectrum.py; do
    (cd "$work/$checkout" &&
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --frozen python \
        "$bench/analytic/$script" "$label" "$work/analytic")
  done
done
(cd "$work/a0" && uv run --frozen python \
  "$bench/analytic/plot_analytic.py" "$work/analytic")
(cd "$work/a0" && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --frozen python \
  -m pytest -q tests/warp/test_rectangle_magnetic_flux.py)
```

The analytic driver samples exact flux differences, including both wall
cap half-segments. Its current RMS is an arithmetic sample RMS. The
FreeGS region summaries use cylindrical volume weights. The cold Hall
script is a frozen spatial linearization with a constrained spectrum.
Its RK4 bound applies to that linearization.

### GPU nonlinear replays

Use a user-owned Slurm GPU allocation on liustation. Each command writes to
a new directory. Keep the archived tolerances and timestep settings
when comparing either branch. The same device is needed for timing
comparisons.

```sh
for checkout in a0-before a0; do
  for variant in quiet pulse; do
    (cd "$work/$checkout" && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      uv run --frozen python -m fomhd run \
      "$bench/inputs/a0-repaired-14x20-$variant/case.toml" \
      --output "$work/$checkout-$variant")
  done
done
for variant in sync wr3; do
  (cd "$work/a1" && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    uv run --frozen python -m fomhd run \
    "$bench/inputs/a1-repaired-10x14-$variant-pulse/case.toml" \
    --output "$work/a1-$variant-pulse")
done
```

The expected accepted trajectories and same-time measurements are in the
[history tables and evidence JSON](evidence/). The GPU replay emits the
original state, accepted snapshots, and hard-constraint checks. The
gallery includes the complete fields and their same-time differences.

For a one-step CPU smoke check, stage a separate input with the CPU backend
and one scheduled step. On A1 WR this is one macro step. The helper
verifies the stored hashes before staging the alternate case.

```sh
(cd "$work/a0" && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --frozen python \
  "$bench/stage_case.py" "$bench/inputs/a0-repaired-14x20-pulse" "$work/cpu-input" \
  && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --frozen python -m fomhd run \
  "$work/cpu-input/case.toml" --output "$work/cpu-smoke")
```

The archived trajectories were run on RTX 5090 GPUs. Their timing and
memory figures characterize these cases and these source commits.
The full GPU performance changes belong to their own source revisions.
