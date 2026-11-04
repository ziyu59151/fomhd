# Accepted A0 run figures

## A0-PLOT-RUN — Native layouts and actual histories

Target: `tools/plots/plot_run.py`. A small independent Python/Matplotlib script,
using the existing `tools/analyze.py` run loader and snapshot/layout checks.
It reads a completed run and writes a new directory of English PNG/PDF figures,
plotted-data JSON and provenance. No GPU access or solver mutation is needed.

```text
load and verify the accepted run, input hashes and source snapshot
select first/last accepted snapshots by their recorded step/time
use existing cell/face/vertex masks and coordinates without resampling to cells
plot active cells and the input LCFS polygon separately
plot the final axisymmetric B_phi at its actual active vertex locations
reconstruct n_i and mom_i_phi from every retained Fourier mode at a central
  active Z slice: f(phi)=sum_k f_k exp(i*k*phi)
require small imaginary remainder and positive reconstructed density
compute u_i_phi(phi)=mom_i_phi(phi)/(m_i*n_i(phi)) in m/s
plot actual accepted relative number and total-energy drift from history.csv
plot actual complex ion toroidal momentum observation change, if present
group selected initial/endpoint checks by exact accepted step/time
plot max(abs(raw)/threshold) over existing applicable records for each step
  and distinguish skipped zero ratios from absent/unavailable diagnostics
retain hard versus observe semantics in both legend and data; an observe
  FAIL is shown, not promoted to a hard acceptance gate or hidden
```

Cell and staggered field masks must remain separate. Pixel/marker geometry is
the actual coarse layout; smooth contours must not imply additional resolution.
Fourier synthesis evaluates the stored truncated series, not a new simulation.
Numbers, energy and diagnostic ratios are dimensionless with explicit
normalization. Zero observation amplitude is unavailable, not an infinite
relative change. Histories are short-window checks and do not establish
long-time stability. Every input file used and both plotting/analyzer source
files are hashed in the output. Existing output directories are rejected.
