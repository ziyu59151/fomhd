# Grid-mode comparison and branch export

This offline host tool reads production candidate exports. It does not rebuild
the dense Jacobian, call an eigensolver, evolve a state, or write an input
report/NPZ.

1. Resolve each input directory or NPZ with its sibling `report.json`. Hash both
   files and load only the small candidate/geometry arrays through
   `np.load(..., allow_pickle=False)`. Selectively reading an NPZ does not
   materialize saved Jacobian/eigenvector arrays.
2. Validate report family, toroidal harmonic, and physical reference identity
   across comparisons. A changed numerical grid or geometry digest is allowed;
   changed family, mode, species/references, `mu0`, model switch, or angular
   sampling is rejected before scoring.
3. Validate candidate arrays: eig indices `[c]`, frequency `[c]`, magnetic
   transfer ratio `[c]`, staggered-MB magnetic norm and ion kinetic norm `[c]`, velocity
   `[c,3,n]`, projected-node B `[c,3,n]`, density `[c,n]`, cells `R/Z/M0 [n]`,
   background node B `[3,n]`, and positive scalar `rho/mu0`.
4. For a circle, solve the existing independent cold constrained-wall reference
   at default order 8 and evaluate velocity/B at the exported cell centers.
   Score each candidate by the nodal diagnostic
   `rho*M0*|v|^2 + M0*|B|^2/mu0`. Report the absolute phase-invariant normalized
   complex overlap. This is strictly a structure metric, not physical total wave
   energy. Keep the candidate magnetic transfer ratio
   `sum(M0*|QB B|^2)/sum(MB*|B|^2)` visible because it differs from one.
5. For continuation, interpolate the previous selected complex velocity,
   projected-node B, and density with `LinearNDInterpolator` onto current cell
   centers. Use a target only when every interpolated component is finite and
   the target lies inside an actual active source-cell rectangle inferred from
   source center spacings. Convex-hull support alone can bridge an inactive
   staircase cell and is not sufficient. Report the linear finite count, active
   footprint intersection, common-cell count, and covered `M0/sum(M0)` with no
   extrapolation or hole filling. Score the same nodal energy overlap on that
   common support.
6. Rank cold-reference and continuation scores separately, report frequency,
   frequency ratio/delta, and top candidates. Store each metric's own top-score
   gap and ambiguity flag; never share one gap field across metrics. Mark close
   top-score ties as ambiguous and explicitly state that same-branch continuity
   is not asserted.
7. Select one candidate by finite positive frequency (Hz or MHz) or by exported
   eig index. Save its fields and provenance to a new selected NPZ; never mutate
   the source report or spectrum. `legacy-export` may reconstruct one context
   from an old spectrum, read its saved eigenvectors, and export a requested
   old branch without reconstructing the Jacobian.
8. Initialize continuation state to `None`, so ordinary scoring without a
   previous branch or selection never reads an unbound variable.
