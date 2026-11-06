# A0 same-branch spatial refinement campaign

Protocol selected before the new results. The accepted prototype release stays
immutable. This campaign changes grid resolution and operational memory caps,
not the physical model, equations, wall closure or RK4 algorithm.

Compare circle nphi=0/1 grids 12,16,20,24,32, and EAST nphi=0/1 grids
12,16,20,24. Both retain Nphi=5, massless electrons, the same physical polygon,
profiles, boundary and source-free evolution as the prototype campaign.

First measure actual operator setup, conservative peak-memory plan, physical
storage RSS, and accepted-step cost in owned Slurm CPU allocations. Select
explicit CPU/CUDA allocations based on these observations; no silent backend
fallback or weakened memory/physical gates. The new campaign GPU ceiling is
eight GPU-hours, using bounded individual allocations. CPU spectral analysis
and plotting do not require GPU allocation.

Archive the complete baseline Jacobian/eigenpairs and all original candidates.
Track the previously selected cold-fast response by physical magnetic/velocity
structure, frequency neighborhood and independent cold reference. The QB-nodal overlap
is a structure metric, not the staggered MB magnetic-energy norm. Export and
report their magnetic quadratic-norm ratio; it can vary with grid and filter
visible small-scale structure. Keep the staggered MB norm independently. Interpolate
only for analysis on common supported cell positions, never to change the
solver or fill exterior/unsupported values. Record support coverage and top
competing candidates; ambiguity is a result, not a convergence pass.

Separate three questions: selected discrete eigenfrequency versus h, actual
RK4 evolution versus that eigenfrequency, and comparison with the independent
qualified cold Hall weak-wall reference. For EAST retain the relaxing
background and matched unperturbed run. A finer linear eigenmode image is
labelled as such; it is not substituted for a time-integrated result.

For selected feasible fine grids use matched control/perturbation evolution,
with existing actual-spectrum timestep policy and at least four measured
cycles when a frequency claim is made. Preserve short/budget-stopped results
as incomplete windows. Prior temporal/amplitude controls remain evidence;
repeat them only when a new failure or substantial mode change warrants it.

Render signed magnetic compression and density, velocity magnitude/direction,
input polygon and actual active cells, with correct n0/n1 amplitude semantics.
Use common normalized color scales for grid-to-grid shape comparisons; do not
smooth a grid-scale pattern into an apparent physical mode. Record staircase
area error separately. Two points or a branch switch do not establish an
order; report adjacent frequency changes and shape overlaps before deciding
whether the sequence is approaching a reproducible limit.

Actual response comparison must also examine early/late spatial coherence
against the seed. Use the real Re/Im eigenmode subspace for n0, since magnetic
and velocity components are in quadrature; a raw instantaneous joint-vector
overlap would falsely identify normal oscillation as shape change. n1 permits
a complex phase-invariant line/subspace comparison. This is an evolution
consistency diagnostic, not independent physical validation. Masked-area error
is reported with actual active-cell overlays; area alone does not establish
boundary-shape convergence.

The first grid24/grid32 K1 cost probes were cancelled during preparation and
retained as incomplete attempts. The identified implementation bottleneck was
repeated prefix SVD in independent constraint-row selection. A full normalized
matrix rank check can prove that every prefix is independent using singular
value interlacing; only then may the same ordered nonzero rows be returned
immediately. Rank-deficient/ambiguous cases retain the original prefix
algorithm and threshold. Verify exact selected rows, operators and accepted
states against the archived implementation before using this equivalent gate
for the scan. This does not change the physical/discrete boundary or equations.

Measured32-grid one-step costs imply approximately19-28 wall-clock hours
per complete spectral sector in the4-CPU setup, before additional eigenvalue
work. The32-grid full-spectrum tasks were deferred and their incomplete
logs preserved. This campaign completes circle12/16/20/24 and EAST12/16/20/24
first;32-grid preparation and one-step measurements are cost evidence only.

Initial audit, before interpreting the new spectra: circle12/16/20/24 have
112/208/316/448 active cells. Their mask area errors relative to the fixed
128-vertex polygon are -0.930%/+3.492%/+0.626%/-0.930%. With the original fixed
physical-flux initialization, the fitted MB-weighted magnetic amplitude errors
are +0.923%/-3.479%/-0.631%/+0.986%. Thus a raw grid-to-grid frequency comparison
includes changes in both boundary geometry and mapped background. These
simultaneous errors do not identify their individual frequency contributions;
do not divide primary frequencies by the fitted scale as an acceptance fix.

Reanalysis of the archived eight-cycle responses gives circle seed-subspace
coherence at least0.99999975, while EAST12 ends at0.940963 (n0) and0.917186
(n1). This establishes departure from the baseline seed span during EAST
evolution; background relaxation is a possible contributor, not a demonstrated
cause. New grid spectra/evolution remain separate evidence.

Implementation validation for the scan tools:643 CPU tests passed,9 skipped,
3 subtests passed;12 GPU gate tests passed with the existing expected singular
matrix negative-test warning. Three archived preparation comparisons returned
exactly the same rows, operators and accepted states. Source/report hashes,
raw attempts and detailed receipts reside in the personal run directory
`a0-grid-scan__20261106T010047Z`; these checks do not establish spatial
convergence or physical validation.

Absolute and adjacent-difference mode maps remove the eigensolver's arbitrary
complex scale using one shared staggered-magnetic plus ion-kinetic quadratic
normalization. The compression phase is declared and the same factor applies
to all physical components. This display-only normalization preserves relative
polarization and never changes physical response histories or raw spectra.
