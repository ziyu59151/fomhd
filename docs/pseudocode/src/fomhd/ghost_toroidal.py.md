# `src/fomhd/ghost_toroidal.py` implementation pseudocode

The operator owns real FP64 line factors and two physical vertex vectors on
the selected backend. Setup takes full physical vertex coordinate lines in
metres and the radial/axial graph factors restricted to their interior
vertices. The graph chart leaves all true boundary vertices unchanged.

```text
build_ghost_toroidal(radial_nodes, axial_nodes, radial_chart, axial_chart, backend):
  compute the half/full dual-link bounds, lengths and radial mean Rbar
  set av = axial_length outer radial_length and h = 1/Rbar
  build uniform-moment ghost rows for cell and physical wall line values
  build axial ghost norm with length measure and radial norm with length/Rbar
  solve each interior graph line matrix against the identity
  U = inverse chart, acting with two line matrices only on interior vertices
  cache defect h-Uh and flux projection vector av/(av dot h)

toroidal_reconstruct(factors, values, backend):
  Braw = U Bstored + (h-Uh) * (av dot Bstored)/(av dot h)
  Favg = Rbar * Braw
  apply axial and radial midpoint rows and divide by the true cell radius

toroidal_trace(factors, values, backend):
  reuse the same Braw and Favg representation
  apply wall point rows normal to each wall and midpoint rows along its line
  divide radial wall results by wall radius and axial results by cell radius
  return radial wall values in (..., NZ, 2), then axial (..., 2, NR)

toroidal_weight(factors, values, backend):
  use the same Braw and Favg
  apply the axial and radial magnetic energy line norms to Favg
  multiply by Rbar, then apply U transpose plus the transposed rank-one term
```

`values` can have arbitrary leading batch axes. Its last axis is the
z-major flattened vertex dimension. `vertex_radius` and the two projection
vectors retain the two-dimensional vertex-grid shape. Reconstruction returns tesla values at
cell centers in the same batch layout. The norm result uses the original
stored vertex chart and contributes to the complete magnetic energy pairing.
The actions use the owning backend with no stage host transfers. The wall
reconstruction uses the decharted physical area means in both directions.
At a true wall vertex the chart is identity, but its half-area mean still
requires deaveraging in the normal direction.

The linear dechart selects the stored direct constant-F harmonic and the
graph-transformed zero-total-toroidal-flux complement. It matches a direct
harmonic plus Faraday increments. Arbitrary old nonharmonic initial fields
may have used a different convention for their net flux. The existing
accepted B arrays, fixed caps and relative Faraday update are unmodified.

If (E) is the dechart including the harmonic correction, its adjoint
fixes the area vector. Thus the composite metric maps the direct harmonic
exactly to the area vector, which is annihilated by the toroidal circulation
transpose. The new norm is positive definite since its two line norms are
positive definite and (E) is invertible on both the harmonic and the
zero-flux spaces.

The factors occupy (O(N_R^2+N_Z^2+N_RN_Z)) storage. Each batched action
uses (O(N_RN_Z(N_R+N_Z))) operations per field plus the scalar flux
projection. There is no stored global dense vertex-to-cell or magnetic
norm matrix. Invalid line support, chart, or vertex vector shape fails before
the new action can produce a result.
