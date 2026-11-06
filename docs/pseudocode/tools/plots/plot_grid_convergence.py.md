# Grid convergence and mode-structure plots

1. Load every explicit report/selected-field pair and hash all inputs. Require
   the selected file to name the hash of the supplied report and to match its
   family/mode/physical references. Keep zero-frequency choices visible.
2. Plot selected frequency versus grid resolution and continuation
   overlap/coverage versus resolution. Name resolutions as `N_R x N_Z`
   (`grid 20 x 20`), name the retained harmonic `n_phi`, and never use one
   overloaded `n` for both. Show wall insets. Do not fit or report a
   convergence order; two points cannot establish one.
3. For adjacent grids, interpolate the previous unencoded complex eigenvector
   amplitude onto current centers with finite-common-support rules. Plot signed
   `Re(current-previous)` projected `B_parallel`, density, and velocity R/Z at
   actual cell centers. Render piecewise-constant per-cell rectangles as a
   masked `pcolormesh`; inactive/non-common cells are masked. Use a shared
   diverging, symmetric per-quantity scale and velocity-direction arrows rather
   than a smooth image that can hide coherent central or boundary-local spots.
   Before subtracting, place the current complex B, velocity and density in
   its declared compression phase and unit shared quadratic norm, then fit
   one complex factor to the joint weighted nodal
   velocity/B vector, then apply that same factor to previous B, velocity, and
   density so their phase relation is preserved. Record the factor and common
   support. Identical modes under `-1` or `2j` must produce zero difference.
   Independent rescaling/rephasing of either member must not change a nonzero
   residual or the shared residual color scale across grid pairs.
4. Treat selected eigenvector fields as unencoded physical amplitudes and
   label maps as a linear discrete eigenmode. Remove arbitrary eigensolver
   scale with one shared divisor
   `sqrt(MB_magnetic_norm + M0_ion_kinetic_norm)` applied to B, velocity and
   density together; this quadratic normalization is not total wave energy.
   A global factor such as `2j` must not change the displayed shape.
   Always render absolute selected structures for every supplied grid, including
   a one-grid invocation: signed `B_parallel`, signed ion density (`delta n_i`),
   and full three-component ion velocity `|v|` with
   poloidal direction, flat actual cells, wall, and scales common across grids.
   Keep these visually and textually distinct from aligned differences and from
   actual evolved response maps.
   For unencoded `n_phi=0` data, anchor magnetic and density at one common compression
   phase (dominant magnetic compression component real). Velocity may be in quadrature there, so choose
   one separate global velocity phase by making the dominant poloidal component
   real, apply that same factor to `vR` and `vZ`, and label the phase. Never
   present roundoff `Re(v)` at the compression phase as a velocity mode.
   Inter-grid subtraction continues to use one joint B/velocity alignment factor.
   For `n_phi=1` use one declared compression phase for all fields, including velocity,
   retaining their relative polarization. This chooses the arbitrary toroidal
   angle origin for comparing selected linear modes; it does not rephase actual
   evolved responses. Record both phase and normalization in the manifest.
   For an optional matched-control response, reconstruct the physical phi=0
   field from retained response coefficients: mode 0 is `Re(A)`, mode 1 is
   `2 Re(A exp(i phi))`; label the encoded time-evolved response separately as
   nonlinear time evolution rather than as a linear selected mode.
5. Measure actual evolved seed-subspace coherence from the saved weighted nodal
   vector `[sqrt(M0)*B/sqrt(mu0), sqrt(M0*rho)*v]`. Recover velocity from its
   five physical phase samples and use the already-retained `delta_B`
   coefficient. For `n_phi=0` project the real evolved vector onto the real span of
   weighted `Re(seed)` and `Im(seed)`; for `n_phi=1` project onto the complex weighted
   seed line and take the absolute phase-invariant inner product. Save per-time
   coherence, residual outside the span, and a compression-extremum summary.
   State that matched-control subtraction and numerical filtering are separate
   caveats.
6. Require sibling `case.toml`, `report.json`, and `input_mapping.npz` for a
   matched response. Match family, harmonic, numerical grid, coordinates, seed
   frequency, and both spectrum report and array hashes equal to the selected source
   before plotting or relabeling response fields. Reject missing or inconsistent
   provenance explicitly.
7. Put a persistent purple corner banner `LIN | FOMHD discrete Jacobian` on
   every selected curve, absolute map, and aligned-delta figure. Mark those
   figures `not time evolved` without overlaying multi-line explanatory
   paragraphs. Keep normalization, phase, common-support, and no-convergence-
   order limitations in the JSON manifest and existing captions.
8. Save PNG and SVG plus an input-hash manifest. A missing optional response or
   an unavailable interpolation region is recorded visibly and never filled.
