# `src/fomhd/ghost_reconstruction.py` implementation pseudocode

The host setup creates two FP64 line transfers from physical link means.
Lengths and coordinates are in metres. Each matrix is constant
across Fourier modes and RK stages. Ghost support is eliminated into the
physical rows at setup. A stage uses the same rows for its current magnetic
state and applies no new wall condition to tangential magnetic field.

```text
ghost_link_reconstruction(nodes, lower, upper, cylindrical, chart):
  normalize the coordinate by the uniform spacing, starting at the first wall
  integrate each real interval's normalized moments with dZ or R dR
  fit the first/last four intervals (three on a two-cell line)
  evaluate the fitted moments on two full ghost intervals outside each wall
  for the first/last cell and each wall point, use a six-link moment fit
    with both adjacent ghosts and four physical links
    (five links total on a two-cell line)
  for other cells, fit four physical links
  compose fitted rows with the ghost continuation rows
  if chart is supplied, right-solve the resulting rows by the graph chart
  return cell-point rows and the two actual wall-point rows

```

When an exterior radial interval would reach zero radius, mirror the radial
metric at that wall for auxiliary moments. True interval moments always use
the positive physical radius. Reject invalid shape, extent, nonfinite inputs,
or a singular solve. Stored poloidal components use their existing graph chart
inverse. The toroidal consumer uses these line transfers on the decharted
dual-area means of (F=R B_\phi) as specified in `ghost_toroidal.py.md`.

The local polynomial subspaces are reproduced to floating-point precision.
Use actual physical link averages in tests. A weighted mean and its value at
the geometric face are different even in the box interior. The old chart and
point reconstruction therefore require separate checks.
