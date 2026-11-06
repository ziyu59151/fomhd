# Grid convergence and mode-structure plots

1. Load every explicit report/selected-field pair and hash all inputs. Require
   the selected file to name the hash of the supplied report and to match its
   family/mode/physical references. Keep zero-frequency choices visible.
2. Plot selected frequency versus numerical cell count and continuation
   overlap/coverage versus count. Show grid labels and wall insets. Do not fit
   or report a convergence order; two points cannot establish one.
3. For adjacent grids, interpolate the previous unencoded complex eigenvector
   amplitude onto current centers with finite-common-support rules. Plot signed
   `Re(current-previous)` projected `B_parallel`, density, and velocity R/Z at
   actual cell centers. Render piecewise-constant per-cell rectangles as a
   masked `pcolormesh`; inactive/non-common cells are masked. Use a shared
   diverging, symmetric per-quantity scale and velocity-direction arrows rather
   than a smooth image that can hide coherent central or boundary-local spots.
   Before subtracting, fit one complex factor to the joint weighted nodal
   velocity/B vector, then apply that same factor to previous B, velocity, and
   density so their phase relation is preserved. Record the factor and common
   support. Identical modes under `-1` or `2j` must produce zero difference.
4. Treat selected eigenvector fields as unencoded physical amplitudes: use
   `Re(A)` for both n0 and n1 and label maps as a linear discrete eigenmode.
   Always render absolute selected structures for every supplied grid, including
   a one-grid invocation: signed `B_parallel`, signed density, and full
   three-component `|v|` with
   poloidal direction, flat actual cells, wall, and scales common across grids.
   Keep these visually and textually distinct from aligned differences and from
   actual evolved response maps.
   For unencoded n0 data, anchor magnetic and density at one common compression
   phase (dominant magnetic compression component real). Velocity may be in quadrature there, so choose
   one separate global velocity phase by making the dominant poloidal component
   real, apply that same factor to `vR` and `vZ`, and label the phase. Never
   present roundoff `Re(v)` at the compression phase as a velocity mode.
   Inter-grid subtraction continues to use one joint B/velocity alignment factor.
   For an optional matched-control response, reconstruct the physical phi=0
   field from retained response coefficients: mode 0 is `Re(A)`, mode 1 is
   `2 Re(A exp(i phi))`; label the encoded time-evolved response separately.
5. Measure actual evolved seed-subspace coherence from the saved weighted nodal
   vector `[sqrt(M0)*B/sqrt(mu0), sqrt(M0*rho)*v]`. Recover velocity from its
   five physical phase samples and use the already-retained `delta_B`
   coefficient. For n0 project the real evolved vector onto the real span of
   weighted `Re(seed)` and `Im(seed)`; for n1 project onto the complex weighted
   seed line and take the absolute phase-invariant inner product. Save per-time
   coherence, residual outside the span, and a compression-extremum summary.
   State that matched-control subtraction and numerical filtering are separate
   caveats.
6. Require sibling `case.toml`, `report.json`, and `input_mapping.npz` for a
   matched response. Match family, harmonic, numerical grid, coordinates, seed
   frequency, and both spectrum report and array hashes equal to the selected source
   before plotting or relabeling response fields. Reject missing or inconsistent
   provenance explicitly.
7. Save PNG and SVG plus an input-hash manifest. A missing optional response or
   an unavailable interpolation region is recorded visibly and never filled.
