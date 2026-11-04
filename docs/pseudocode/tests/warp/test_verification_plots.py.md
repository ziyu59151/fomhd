# Verification plotting checks

## A0-PY-TEST-PLOT-VERIFY-01 Data transformations and refusal

Exercise the actual plotting helpers with analytic complex eigenvalues and
literal step/error pairs. Check rad/s to Hz conversion, sorted positive branches,
observed order two, nonfinite/nonpositive rejection, duplicate refinement
rejection, unequal-horizon rejection, and existing-output refusal before any
input read. A missing NPZ must not be replaced by theoretical eigenvalues.
The smoke render on actual archived reports is separate from these unit checks.

Build a minimal complete pair of reports and a real numeric Hall NPZ. Collect
the data and require undrawn MMS negative-control and group statuses, all
massless case statuses and three input hashes to be preserved. An undetected
negative control must be refused, not dropped from figure provenance.
