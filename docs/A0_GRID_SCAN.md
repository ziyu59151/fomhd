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

## Spectrum evidence and final presentation

The table records working discrete candidates, not established same-branch
spatial convergence. Full nonlinear follow-ups and the remaining readiness
sectors have separate statuses in the live evidence archive.

| Circle enclosing grid | nphi=0 working frequency [MHz] | nphi=1 working frequency [MHz] |
|---|---:|---:|
|12|16.217947|16.476521|
|16|15.873736|16.059240|
|20|15.680648|16.287291|
|24|16.886549|17.092433|

The16-to20 changes are -1.2164%/+1.4201%. The corresponding continuation
overlaps are0.59423/0.70117; independent cold-reference overlaps are
0.60233/0.75880. Both rankings identify the same working candidates, but
frequency changes alone do not establish structural or spatial convergence.
At24 the nphi=1 continuation ranking selects index806, while the independent
cold-reference ranking selects index804 at17.352989MHz. Keep that disagreement
visible rather than treating the first selected image as an accepted branch.

EAST16 has no strong individual continuation of the archived EAST12 seeds.
The highest classified-candidate overlaps are0.21136/0.27410 at
8.405215/8.594070MHz (n0/n1). An independent audit of all853 positive-imaginary
n0 raw eigenpairs still finds maximum forward overlap0.21136. Nearby raw
mode5.625852MHz scores0.20842 and has parallel magnetic fraction0.16713,
below the unchanged0.5 classifier cutoff; the old selected fraction was
0.52794. Reversing interpolation changes the ranking but leaves all overlaps
low (maximum0.24250). These are unresolved spatial/polarization changes,
not an export/unit/layout or saved-eigendecomposition defect. Do not connect
the newly selected diagnostic probes into an accepted EAST branch curve.

EAST16 n0 K0/K1 fast-candidate frequencies agree to a maximum nearest relative
difference1.9241e-14, so that particular retained-sector check does not explain
the spatial discrepancy. All original spectra, excluded modes and failed
continuation evidence remain archived.

The user identified a recognizable two-lobe EAST20 n_phi=1 magnetic pattern
(saved index651,9.582281306MHz). A new follow-up compares24/28/32 with this
resolved20-grid direction, without requiring it to match the coarse12 seed.
Component overlaps distinguish magnetic compression from velocity, density
and the joint nodal metric. In the classified24-grid candidates the largest
magnetic-compression overlap is0.27986, at11.916495MHz; its joint overlap
is0.09440 on91.27% common M0 support. This is an emerging-structure hypothesis
and unresolved branch tracking, not a named-mode identification or convergence
pass. Excluded raw eigenpairs remain archived. Owned CPU28/32 full spectra are
running separately and do not consume additional GPU budget.

The curated report distinguishes teal NL nonlinear time evolution, purple LIN
production-Jacobian analysis, and independently formulated REF evidence.
Grid counts are N_R/N_Z, the harmonic is n_phi, and densities are n_i/n_e.
The report has numbered logical pages and keeps historical figures17/18;
detail reduction never deletes the full evidence. Prototype priorities follow
[RESEARCH_PROTOTYPE_WORKFLOW.md](RESEARCH_PROTOTYPE_WORKFLOW.md).
