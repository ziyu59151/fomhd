# Native migration oracle exporter

## A0-EXPORT-NATIVE-01 - Independent native evidence

Scope: numerical verification evidence; no change to native evolution.

1. Parse NEW output NPZ, rectangular point count, K, perturbation amplitude.
2. Construct native test parameters and geometry/operators independently.
3. Construct closed_reference; optional stress case adds positive nonuniform
   density, nonuniform anisotropic pressure and retained toroidal pressure modes; evaluate full stage with alias audit; require
   every hard check available and passing. Advance one native midpoint step.
4. Export all geometry weights, topology, compatible/interpolation/constraint
   matrices, retained state, stage fields/RHS/ledgers, and accepted endpoint.
5. Convert native column-major matrices through operator(r,c) into C-order
   NPZ explicitly; vector ordering remains unchanged. Embed schema/config.
6. Refuse existing output and preserve nonzero exit on any failed gate.
