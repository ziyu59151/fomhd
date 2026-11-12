# A0 rectangular-wall compressional Alfvén mode

The physical case is a synthetic deuterium equilibrium with stationary ions and electrons, uniform density $2\times10^{19}\,\mathrm{m}^{-3}$, uniform isotropic pressures and a vacuum toroidal guide field $B_\phi=30/R$ T. The major radius is 15 m. The full Hall/CGL-B solver uses positive pressures at total axis $\beta=10^{-5}$, retains $n_\phi=-1,0,+1$, and advances with synchronous RK4. The black frame in the RZ sections is the fixed rectangular conducting wall. The internal marker in the geometry array defines no material or electromagnetic interface.

The [repaired conducting wall](../rectangular-wall/README.md) supplies the spatial baseline. The [A0 performance replay](../a0-rk4-gpu/README.md) contains same-device implementation timing. The mode case measures physical frequency, polarization and structure through accepted evolution.

## Input matrix

| Replay name | Grid | Physical halfwidths (m) | Step (ps) | Planned steps | Planned time (ns) | Role |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| narrow-g12 | 12×12 | 0.60 | 195 | 3,840 | 748.8 | Fixed-box spatial comparison |
| narrow-g16 | 16×16 | 0.60 | 130 | 5,760 | 748.8 | Main perturbation |
| quiet-g16 | 16×16 | 0.60 | 130 | 5,760 | 748.8 | Same-time unperturbed control |
| half-g16 | 16×16 | 0.60 | 65 | 5,760 | 374.4 | Fixed-grid time-step check |
| wide-g20 | 20×20 | 0.75 | 130 | 5,760 | 748.8 | Shift both walls by 0.15 m at fixed 0.075 m cell spacing |
| cold-hall-g8 | 8×8 | 0.60 | 100 | 120 | 12.0 | Separate analytic axial $n_\phi=0$ control |

Each case has its original `case.toml`, `geometry.npz`, and `initial.npz`, with the exact mode data `seed.npz` where used. Run `sha256sum -c INPUTS.sha256` from this directory. The portable `source.json` replaces the archived absolute frozen-seed provenance path with its basename. This changes the manifest hash for the affected cases. The cases replay from the packaged initial arrays. The [input provenance table](INPUT_PROVENANCE.json) records the two manifest hashes and the identical physical input hashes. [Input checksums](INPUTS.sha256)

## Reproduction

Prepare the A0 solver checkouts at `803b332ce45750b8c6e759f7405ba76ed87ee213` for the toroidal runs and `bc809bdd3aa5253e1ffa60c21e7dcf5e7c84d369` for the axial control. Use Python 3.12, Warp 1.19.0, CuPy 14.2.0 and the NumPy/SciPy versions pinned by the checkout. Install with `uv sync --locked --group plot`. [Verified package source hashes](SOURCE_HASHES.json)

In a personal single-GPU Slurm allocation, replay each selected case using a new case directory outside the repository. The command keeps the input’s device, precision, step guard, validation, checkpoint and output settings. The case runner writes diagnostics to its normal `run/` and puts the additional timing records alongside it.

```bash
packet=/path/to/a0-rectangle-cae
archive=/path/to/rectangular-wall-checkout
optimized=/path/to/rk4-scaling-checkout
python=/path/to/.venv/bin/python
result=/path/to/new-analysis-tree
mkdir "$result"
cp -a "$packet/." "$result/"
mkdir "$result/cases"

bash "$packet/replay.sh" "$archive" cold-hall-g8 "$result/cases/cold-hall-g8" "$python"
for name in narrow-g12 narrow-g16 quiet-g16 half-g16 wide-g20; do
    bash "$packet/replay.sh" "$optimized" "$name" "$result/cases/$name" "$python"
done
```

The physical controls can run independently in separate allocations. Analyze the finished accepted snapshots on CPU. A full gallery regeneration uses these paths and run names.

```bash
cd "$result"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
PYTHONPATH="$archive/src:$archive" "$python" methods/analyze_cold_control.py \
    cases/cold-hall-g8 > cold-hall-short-comparison.json
export PYTHONPATH="$optimized/src:$optimized"
"$python" methods/analyze_mode.py cases/narrow-g12 analysis-r15-g12
"$python" methods/analyze_mode.py cases/narrow-g16 analysis-r15-g16 --quiet cases/quiet-g16
"$python" methods/analyze_mode.py cases/half-g16 analysis-r15-g16-half
"$python" methods/analyze_mode.py cases/wide-g20 analysis-r15-wide
"$python" check_receipts.py cases/narrow-g12 analysis-r15-g12/check-receipt.json
"$python" check_receipts.py cases/narrow-g16 analysis-r15-g16/check-receipt.json
"$python" check_receipts.py cases/quiet-g16 raw/quiet-g16/check-receipt.json
"$python" check_receipts.py cases/half-g16 analysis-r15-g16-half/check-receipt.json
"$python" check_receipts.py cases/wide-g20 analysis-r15-wide/check-receipt.json
"$python" compare_controls.py
"$python" render_gallery.py
```

## Independent reference and branch selection

The cold Hall reference uses analytic cylindrical derivatives, weighted two-dimensional quadrature, and a weak conducting-wall treatment with normal velocity, flux and current constraints. The represented gradient gauge is removed before whitening the magnetic mass. The narrow-box orders 12, 14 and 16 yield 6.162656, 6.170819 and 6.176967 MHz. The wider-box order-16 frequency is 4.938820 MHz. Both order-16 references are independently regenerated by the commands below. They are sampled into the bundled seed arrays. [Narrow-box refinement](reference/ritz-R15-n2e19-a0p6.json) · [Wider-box refinement](reference/ritz-R15-n2e19-a0p75.json)

For the exact branch selection, build the order-14 cold reference first and select its nearest strongly compressional frozen A0 mode. Then sample the order-16 reference for comparison. The frozen A0 mode uses the production interpolation, curl and wall reaction. Its frequency informs the seed and does not serve as the independent benchmark. The selected frozen branch, continuous reference and fully evolved state appear separately in the results.

```bash
mkdir -p reference-work
for order in 12 14 16; do
    "$python" methods/box_reference.py "reference-work/ritz-R15-n2e19-a0p6-p$order" \
        --order "$order" --width 0.60 --major-radius 15 --density 2e19
    "$python" methods/box_reference.py "reference-work/ritz-R15-n2e19-a0p75-p$order" \
        --order "$order" --width 0.75 --major-radius 15 --density 2e19
done
"$python" methods/discrete_cold.py reference-work/ritz-R15-n2e19-a0p6-p14.npz reference-work/narrow-g16 --grid 16
"$python" methods/prepare.py reference-work/new-g16 reference-work/ritz-R15-n2e19-a0p6-p16.npz \
    --grid 16 --dt 1.3e-10 --steps 5760 --stride 60 \
    --discrete reference-work/narrow-g16-tracked.npz --pair
"$python" methods/ritz_spatial_scales.py --references reference-work --output .
"$python" methods/hall_reference_shift.py reference-work/ritz-R15-n2e19-a0p6-p16.npz \
    hall-reference-shift.json
```

The [Hall continuation](hall-reference-shift.json) varies the Hall block at fixed traces in the reference only. All full A0 trajectories use the physical Hall term. Reference regeneration may change the eigenvector phase and container bytes. The exact-input replay is the path for byte-identical physics inputs. The [Ritz spatial-scale analysis](RITZ_SPATIAL_RESOLUTION.md) measures wavelengths and central-lobe widths directly from the reference fields, with numerical limits and estimates for a next fixed-box grid. The separate [reference current check](reference-current-convergence.json) includes order 18. The [frozen spatial review](independent-review/REVIEW.md) includes the unresolved narrow-box wall-collar shape difference, an independent 20×20 frozen diagnostic, and bounded nonsquare alternatives. The [analytic axial Hall reference](reference/AXIAL_HALL_REFERENCE.md) gives the equations for the separate short control.

For the supplemental current comparison, generate the narrow order-18 archive and compare the derivatives on common quadrature points.

```bash
"$python" methods/box_reference.py reference-work/ritz-R15-n2e19-a0p6-p18 \
    --order 18 --width 0.60 --major-radius 15 --density 2e19
"$python" methods/ritz_spatial_scales.py --references reference-work --include-order-18 \
    --frozen-narrow raw/frozen-support/discrete-r15-g12-tracked.npz \
    raw/frozen-support/discrete-r15-g16-tracked.npz \
    raw/frozen-support/g20-R15-n2e19-a0p6-p14-tracked.npz --output .
"$python" methods/reference_current_convergence.py --references reference-work \
    --output reference-current-convergence.json
PYTHONPATH="$result/methods:$PYTHONPATH" "$python" independent-review/wall_current_audit.py \
    --references reference-work --output independent-review/wall-current-weak-trace-audit.json
```

The [wall-current and momentum-trace check](independent-review/WALL_CURRENT_WEAK_TRACE.md) separates the toroidal derivative from the poloidal current that enters the cold Lorentz force.

## Results

The axial control accepted 120 steps through 12 ns. Its phase at the endpoint differs from the analytic trajectory by $-0.00870$ rad. The energy-scaled perturbation error is 6.34%. The analytic period is about 91.0 ns, so the short control measures phase and waveform over a fraction of one period. [Axial control plot](cold-hall-control.png) · [Archived samples](cold-hall-short-comparison.json) · [Independent recomputation](raw/cold-control-recomputed.json)

<!-- accepted-evolution -->

| Accepted case | Steps | Time (ns) | Cycles | Frequency (MHz) | Ritz (MHz) | Full-box parallel B overlap | CLI elapsed (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Narrow 12×12](analysis-r15-g12/summary.json) | 3,840 | 748.800 | 4.163 | 5.559849 | 6.176967 | 0.7840 | 44.55 |
| [Narrow 16×16](analysis-r15-g16/summary.json) | 5,760 | 748.800 | 4.230 | 5.648532 | 6.176967 | 0.7635 | 64.11 |
| [Narrow 16×16, half step](analysis-r15-g16-half/summary.json) | 5,760 | 374.400 | 2.115 | 5.648532 | 6.176967 | 0.7635 | 64.85 |
| [Wider 20×20](analysis-r15-wide/summary.json) | 5,760 | 748.800 | 3.405 | 4.547351 | 4.938820 | 0.8397 | 68.72 |

[A0 benchmark gallery](index.html) · [Physical comparison report](MODE_REPORT.md)

[Full perturbed and same-time quiet fields with their differences](analysis-r15-g16/total-fields.png) · [Mode structure](analysis-r15-g16/mode-structure.png) · [Polarization](analysis-r15-g16/polarization.png) · [Pressure sections](analysis-r15-g16/total-pressure.png) · [Phase and amplitude](analysis-r15-g16/phase-amplitude.png)

[Wide-box RZ sections](analysis-r15-wide/total-fields.png) · [Wide-box mode structure](analysis-r15-wide/mode-structure.png)

[Separate mesh, time-step and wall comparisons](refinement-time-wall-controls.png) · [Control data](control-comparison.json)
