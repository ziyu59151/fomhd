# Finite-mass longitudinal evidence figure

## A0-PY-PLOTLONG-01: Read completed evidence

Read a completed longitudinal report and every sibling numeric NPZ without
executing solver code. Refuse failed, duplicated, nonfinite, mixed-backend or
unequal-horizon evidence. Verify accepted step/time identity and history
length. Preserve every case summary and amplitude-sensitivity result,
including the half-amplitude controls that are not drawn in the main curves.
Require one distinct passing control for each integrator, its coarsest
same-step full/half-amplitude case pair, and the matching measured change.
Hash the report and all consumed histories.

## A0-PY-PLOTLONG-02: Render local time response

For each integrator, choose the finest full-amplitude history and plot its
electron density mode's real coefficient divided by the prescribed amplitude,
against time divided by the analytic period. Plot the independent cosine
on a fine display grid. Plot full-amplitude endpoint amplification and its
reported integrator-polynomial reference against omega times dt. Separately
plot continuum endpoint errors. Do not call amplification physical growth,
or polynomial agreement continuum accuracy. Use the recorded backend label.
Write PNG/PDF, exact plotted values, input/script hashes, and an English scope
note into a new directory; never modify inputs or replace previous figures.
