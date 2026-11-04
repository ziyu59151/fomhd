# A0 verification figures

## A0-PLOT-VERIFY — Measured convergence

Target: `tools/plots/plot_verification.py`. A standalone host-only Matplotlib
report for the accepted coupled MMS JSON, spatial MMS log, and reviewed CLI
Richardson report. Read-only inputs and a new output directory are required.

```text
require PASS coupled/CLI reports and expected 8,16,32 levels
read all eight evolved-field temporal errors and each positive dt
parse N,h and G,C,DB,DJ interior/wall/global errors from the spatial log
retain exact wall zeros; require positive plotted interior/global errors
recompute log(error ratio)/log(spacing ratio) and check stored temporal/spatial orders
require a time.dt three-run Richardson report with no analytic reference
require equal refinement ratios and differences above the recorded roundoff guard
verify the reported Richardson order from the two differences
select logarithmic axes before data-dependent limits to retain positive bounds
plot temporal MMS, interior spatial MMS, interior/global comparison and CLI differences
anchor slope-2/4 guides at declared measured coarse points
require every plotted error and guide to lie within positive y-axis limits
save PNG/PDF, exact plotted arrays, input/script hashes and English scope notes
```

Temporal errors are coefficient-vector RMS divided by the test's prescribed
component scale, `norm(error)/(scale*sqrt(size))`; they are dimensionless.
Spatial errors are maximum absolute error divided by the maximum analytic
reference in the named region (wall uses global normalization), also
dimensionless. h is the fixture spacing in meters. CLI differences use the
reviewed observable norm and unit; they are not exact-solution errors.
Finite, positive samples and decreasing spacings are checked before log axes
are used. No raw error or tolerance is changed for the plot. The output is a
four-panel summary, not a new convergence study or physical validation.
