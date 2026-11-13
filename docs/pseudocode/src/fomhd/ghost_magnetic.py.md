# `src/fomhd/ghost_magnetic.py` implementation pseudocode

`ghost_link_norm` runs once on the host at aligned-box operator setup. It
returns a real FP64 energy metric in the existing one-dimensional stored
magnetic chart. Inputs are increasing face coordinates and physical dual-link
endpoints in metres, a cylindrical weighting flag and the existing real
invertible graph chart. These inputs include both wall faces. The output is
shared by all stages and Fourier modes. Assembly rejects invalid support,
metrics or a singular reconstruction before a state can advance.

```text
h = box length / cell count
x = (face coordinate - first wall coordinate) / h
integrate exact zeroth, first and second moments on each physical interval
  use dZ axially and R dR radially, with the true R on physical intervals
W = diagonal physical zeroth moments
d = difference of neighboring link centroids in units of h
q = prefix_sum(W * (centroid_offset - weighted_mean(centroid_offset))) / d
H = W + D^T diag(q) D, where D is the first forward difference
L = difference of neighboring first slopes D/d
for each wall:
  take three actual intervals next to the wall in inward order
  continue their quadratic moment polynomial to the first two full ghost links
  evaluate both ghost interval averages with the selected metric continuation
  construct the difference of slopes across ghost2, ghost1 and the wall link
  calibrate its row to the physical curvature row on the exact quadratic means
  replace the adjacent L row, keeping the pullback to the three physical inputs
  on a two-cell line, average the calibrated rows from both sides
p = h/64 at each stencil center, times R there in the radial direction
multiply first and last p by 6/5
on a two-cell line use 3/2 instead for the shared endpoint stencil
H += L^T diag(p) L by accumulating the three-link stencil bands
return chart^(-T) H chart^(-1) using two solves, with symmetric roundoff cleanup
```

Both ghost values enter each boundary curvature row. The pullback uses the
transpose of the same continuation. Physical constant and affine data have
zero curvature, and all added weights are positive. The calibrated row equals
the corresponding physical three-link curvature in exact arithmetic. The
quadratic norm term supplies the accuracy improvement. The ghost rows are
composed at setup, so the resulting metric action at a stage needs only physical
magnetic state. No ghost contributes a physical volume, wall cap, energy
record, evolved degree of freedom or checkpoint field. The radial continuation
uses the true affine cylindrical metric when the two ghost intervals remain at
positive radius. Otherwise it mirrors the radial metric at the wall to keep
the auxiliary integration weights positive. This selection leaves the
calibrated physical row unchanged in exact arithmetic.

The independent regression checks a non-identity stored chart, constant and
affine moments, positive eigenvalues on a two-cell box near the axis, and
second-order radial derivative convergence in both near-wall layers. An axial
quadratic gives its exact derivative on every layer.

For the toroidal (F=R B_\phi) area means, `inverse_radius_weight` selects
unweighted polynomial moments with energy weights `length / mean_radius`.
Use `h/(64*R)` for the curvature coefficient before the endpoint factors.
This radial metric approximates the toroidal magnetic energy and keeps the
constant-F moment exact. It is separate from the cylindrical poloidal-link
moment rule. Reject simultaneous use of the two radial conventions.
