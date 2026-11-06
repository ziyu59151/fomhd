# Archived pressure-transport replay

Recreate the retained fixed-domain disk n20/wall pressure case with the exact
spatial-study builder, density, pressures, analytic profile and fixed harmonic
flux. Preserve strict historical failure separately. Change only power,
energy and momentum budget gates to observe, retaining existing tolerances;
for previously implicit scalar power gates use zero configured tolerance so
the original dynamic roundoff threshold remains. Nonfinite/positivity/solver/
model/current constraints stay hard. Initialize and advance a short actual RK4
run, storing raw initial checks and accepted energy components plus cumulative
physical boundary balance. No initial/source subtraction or new forcing.
