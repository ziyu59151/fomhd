# Exploratory A0 figures

These are small, independent Python/Matplotlib scripts for inspecting existing
C++/CUDA solver evidence. They do not change solver state or rerun a simulation.
Keep case-specific figures here while the output schema is evolving. Extract a
shared plotting module only when repeated use identifies stable readers,
normalizations, and figure types. `plot_run.py` already reuses the checked
`tools/analyze.py` run reader instead of defining another NPZ schema.

## Environment

From the repository root, install the optional plotting dependency group into
the project-local environment:

```bash
uv sync --frozen --group plot
.venv/bin/python tools/plots/plot_run.py --help
```

`uv.lock` pins Matplotlib and its dependencies. No public server packages, GPU
access, PyTorch operations, or C++ runtime dependency are needed for plotting.

## Scripts and inputs

| Script | Required existing input | Figure content |
| --- | --- | --- |
| `plot_verification.py` | Coupled MMS JSON, spatial MMS log, reviewed three-run Richardson JSON | Temporal orders, interior spatial orders, interior/global contrast, actual CLI differences |
| `plot_local_modes.py` | Accepted seven-case local dispersion JSON | Theory/production spectra, growth controls, error budgets, four reconstructed magnetic eigenfunctions |
| `plot_run.py` | Accepted run with first/last snapshots, history, observations and checks | Native geometry and B-phi samples, Fourier velocity slice, number/energy drift, selected momentum response, diagnostic budget use |

Every `--output` must be a **new directory**. Each script saves PNG for browsing,
PDF for export, `plot-data.json`, `provenance.json`, and an English scope note.
The local-mode script also writes `eigenvalues.csv`. Provenance includes the
command arguments, input/script hashes, and package versions. Inputs remain
read-only; missing or unsupported evidence causes an explicit failure.

Example commands with the current evidence (choose a new output root):

```bash
plot_root=/home/lxzeng/runs/fomhd/a0-figures__my-study
evidence_dir=/home/lxzeng/projects/fomhd/tmp/a0-completion-20261104
scan_report=/home/lxzeng/runs/fomhd/closed-reference-dt-scan__20261104T041402Z/convergence-reviewed.json

.venv/bin/python tools/plots/plot_verification.py \
  --evidence "$evidence_dir" --scan "$scan_report" \
  --output "$plot_root/verification"

.venv/bin/python tools/plots/plot_local_modes.py \
  --input "$evidence_dir/dispersion-integrated-full.json" \
  --output "$plot_root/local-modes"

.venv/bin/python tools/plots/plot_run.py \
  --run /home/lxzeng/runs/fomhd/closed-reference-dt-scan__20261104T041402Z/run-steps32 \
  --output "$plot_root/closed-time-scan"
```

The first gallery is stored outside the source tree:
`/home/lxzeng/runs/fomhd/a0-benchmark-figures__20261104T050738Z/index.html`.
Its `README.md` identifies the final figures; early drafts are retained as
evidence. Its `reproduce.sh` uses frozen plotting sources and the original
input paths on this server to regenerate five figures in a new directory.

## Reading the figures

- Temporal MMS errors are coefficient RMS divided by prescribed test scales.
  Spatial errors are normalized maximum errors in the named region. Interior
  fourth-order behavior is not a global fourth-order staircase-wall claim.
  CLI Richardson differences compare numerical runs, not an exact solution.
- The local spectrum uses the convention `exp(lambda*t + i*k.x)`, with growth
  `Re(lambda)` and signed frequency `Im(lambda)/(2*pi)`. Its four magnetic maps
  are **t=0 local eigenfunction reconstructions with arbitrary amplitude**,
  not time-evolved snapshots or global tokamak eigenmodes. Cases are discrete
  parameter samples, not a continuous dispersion scan in wavenumber.
- Physical velocity is computed after Fourier reconstruction of both momentum
  and density. Staggered masks and native sampling locations remain separate;
  no spatial smoothing conceals the coarse mesh.
- Run plots preserve diagnostic `hard` versus `observe` status. An observation
  exceeding its budget is displayed, not silently converted to a passing gate.
  Exact zeros are retained in the data even when omitted on logarithmic axes.
  Short histories do not establish long-time stability or experimental validity.

Each script checks the numerical transformations it uses, including reported
orders, SI conversion, phase alignment, accepted step/time identity, and real
Fourier synthesis. Review still includes opening the rendered images: valid
numbers alone do not catch unreadable axis limits or overlapping annotations.
