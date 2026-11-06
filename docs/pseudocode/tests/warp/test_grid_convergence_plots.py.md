# Grid convergence plot regressions

Generate synthetic selected branches with explicit wall samples and known
signed physical deltas. Check projected `B_parallel`, density, and velocity
delta signs and common-support masks. Verify n0 uses `Re(A)` and encoded n1
response reconstruction uses the factor-two physical convention. Render PNG,
SVG, and an input-hash manifest, and verify changed report/selected bytes are
detected rather than silently accepted. Check flat-cell grids preserve each
actual value and explicitly mask inactive or non-common cells.
Keep the selected magnetic transfer ratio in provenance so structure-map
agreement is not presented as total-wave-energy agreement.

Construct synthetic n0 real/imaginary-span oscillations and an n1 complex-line
oscillation. Verify unit coherence and near-zero residual, then add an
orthogonal field and verify a finite residual appears. Include a mixed B/v
quadrature phase so raw joint overlap would fail while the declared span/line
projection succeeds.

Verify joint-alignment factors `-1` and `2j` give roundoff-zero B, density, and
velocity differences while preserving B/v phase relationships. Require the
absolute selected-structure figure with one grid and both absolute plus aligned
difference figures with two grids. Build synthetic response-run provenance
(`case.toml`, `report.json`, `input_mapping.npz`) and reject missing files,
wrong family/harmonic/grid, mismatched coordinates/frequency, and a spectrum
report or array hash that does not name the selected source.

Check a real n0 B/density with purely imaginary velocity: the compression-phase
velocity is near zero, but the declared global velocity quadrature snapshot is
nonzero and uses one consistent phase for both poloidal components.
Include toroidal velocity and require the displayed speed to include all three
components while retaining poloidal arrows.
