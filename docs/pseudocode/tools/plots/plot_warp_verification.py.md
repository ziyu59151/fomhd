# Warp verification figures

## A0-PY-PLOT-VERIFY-01 Read measured evidence

1. Read a completed massless report and a passed coupled MMS report as host JSON.
2. Reject failed cases, duplicate case IDs and missing required groups. Preserve
   every massless case status in plot-data, including cases not drawn in these
   panels. Also preserve MMS group acceptance and the omitted-source negative
   control's detected/pass/physical-gate flags, errors and scope; reject an
   undetected or failed negative control.
3. For each Hall case, load the sibling numeric NPZ with pickle disabled,
   extract the two positive imaginary eigenvalues of the production matrix,
   and divide angular frequencies by 2*pi to obtain Hz. Plot them against the
   independent omega-minus/plus reference from the report.
4. Read midpoint/RK4 continuum errors and dt from actual massless time cases.
   Require a common physical final time within each refinement sequence.
5. Read the combined temporal MMS errors and dt for each integrator; compute
   descriptive orders as log(error ratio)/log(dt ratio). Use actual backend labels.
6. Read reconstruction velocity/divergence and compressible pressure-force
   relative L2 norms separately for all/inner/wall; preserve unit and entity count.
7. Require finite positive values for logarithmic curves, distinct dt/h samples,
   and valid units. No zero replacement, synthetic points, fitted mode or solver run.

## A0-PY-PLOT-VERIFY-02 Render and preserve

1. Refuse an existing output directory before reading inputs.
2. Draw two PNG/PDF figures: local Hall/time/coupled MMS, and three spatial
   operator panels. Label spatial results as measured with unresolved boundary order.
   Keep dt in seconds in plot-data; multiply by 1e9 for explicitly labeled ns axes.
   Use actual sample locations as x ticks without overlapping minor labels.
3. Save plotted numbers, case statuses, SHA256 hashes of reports/NPZ/script,
   package versions and command arguments. Use a new output directory.
4. State that local transverse Hall evidence is not full-spectrum, Tokamak or ICE
   validation; coupled MMS verifies time integration of forced discrete equations.
   No independent continuum E0 curve is drawn from shared discrete operators.
