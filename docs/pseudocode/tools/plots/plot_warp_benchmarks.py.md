# `tools/plots/plot_warp_benchmarks.py` implementation pseudocode

## A0-PY-PLOT-WARP-01 - Measured benchmark panels

```text
read --run benchmark.json, summary.json, and observations.npz
never synthesize, interpolate, or relabel a failed solve as success

pilot panel:
  plot only measured setup/init/first/steady values
  show measured steady half interdecile spread
  place failed cases in a separate status annotation

scan timing panel:
  plot measured per-step median only for accepted cases
  failed cases appear as named categorical annotations at a valid axes
  coordinate with their actual requested dt; no NaN scientific y is plotted

finite-horizon panel:
  plot measured final amplitude ratio and maximum normalized residual
  failed cases appear as named categorical annotations without fabricated
  residual metrics

frequency panel:
  plot signed measured angular frequency where availability says measured
  for a real standing wave, plot the unsigned angular frequency without
  inventing a propagation direction
  draw a horizontal dashed cold analytic omega_plasma reference
  label it approximate and not a continuum eigenmode validation
  annotate unavailable records with their dt; do not plot NaN y values

all-failed or empty run:
  keep a linear categorical failure axis and finish PNG/PDF generation
  never set a log y-axis that has no positive measured values

write static PNG and PDF into the requested run subdirectory
write the plotted source values to plot_data.json
```
