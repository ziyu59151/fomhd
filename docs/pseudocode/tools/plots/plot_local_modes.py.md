# Local A0 eigenmode figures

## A0-PLOT-LOCAL — Evidence, spectra and mode reconstruction

Target: `tools/plots/plot_local_modes.py`. This is a standalone, host-only
Matplotlib script for accepted `fomhd_local_dispersion` JSON. It does not run a
solver or define new physics. Inputs remain read-only; the output directory
must be new. English PNG/PDF figures and JSON provenance are written there.

```text
read the accepted seven-case dispersion JSON; require finite matrices/modes
verify each stored SI eigenvalue and polarization against its normalized value
check the plotted eigenvalues against the stored mode-local error allowance
plot positive-frequency warm branches and anisotropic growing/stable controls
  as discrete points, not a continuous wavenumber dispersion scan
plot each case's matrix, linearity and eigenvalue discrepancy / acceptance limit

select one nonzero positive-frequency magnetic branch for warm parallel and
  perpendicular cases (lowest-frequency branch above 10% of max magnetic norm)
select the largest-growth branch for each declared unstable CGL case
use the JSON array index, not selected_branch_mode (the latter precedes sorting)
align the analytic vector's dominant magnetic component to a positive real phase
align the production vector to the analytic vector by one global complex phase
divide both magnetic vectors by the same analytic magnetic-vector norm
render Re(delta B * exp(i*k.x)) at t=0 on a local Cartesian phase plane
plot all three magnetic components along wave phase with analytic lines and
  production markers; report gamma and signed Im(lambda)/(2*pi)
retain selected indices, scales, phase factors, plotted data and residual checks
```

Spatial amplitudes are arbitrary eigenvector normalizations. The normalized
phase plane is `k0*x/(2*pi), k0*z/(2*pi)` with B0 along z. These are reconstructed
local eigenfunctions of the measured production-response matrix, not evolved
snapshots, a global tokamak mode, or a time-integrated instability simulation.
Growth rates come from the response eigenvalues; no synthetic exponential time
series is presented as a numerical run. Exact zeros are not fabricated on log
axes. Reconstructing a known complex phase and comparing stored SI/normalized
vectors are checked before publication. Missing cases or unsupported input
fail explicitly; input and script hashes identify the output.
