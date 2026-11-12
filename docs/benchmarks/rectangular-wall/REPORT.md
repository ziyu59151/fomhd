# Rectangular conductor wall current

The repair addresses the poloidal magnetic derivatives that produce
toroidal current in the RZ box. It preserves the Hall/CGL-B equations,
the physical box, the input link chart, and the prescribed outward
normal magnetic flux. A0 exercised the closure before the matching
A1 synchronous RK4 and three-node waveform relaxation (WR) runs.

## Closure and constraints

For each physical link let w be its integrated measure, x its face
coordinate, and xbar its weighted centroid. The axial measure is dZ and
the radial measure is R dR. In the axial first layer, adjacent half- and
full-link means of an affine function are 3h/4 apart. The old adjoint
used h and its two graph actions reduced the limiting slope to 383/576.
The radial measure has the additional shift ell²/(12 Rbar) within each
link and must be included in the polynomial tests.

The old metric, electric measure, and topology cannot all stay fixed
while changing a row of Ampère independently. A symmetric magnetic
metric with a correction supported only at one wall also cannot
reproduce constants and linear functions at fixed electric measure and
unchanged deep-interior action. The correction applied to an affine
sample would have nonzero sum from the half link, while a symmetric
constant-preserving correction must sum to zero. The repair retains
the electric measure and adjusts the nearest-neighbor coefficients
of each full line. The positive norm is a diagonal link quadrature plus
a weighted first-difference Gram matrix. Its action sends constant
means to constant fluxes and affine means to face values up to a common
shift, which cancels in the derivative. The actual stored-chart norm
is obtained by congruence with the inverse graph factors. See the
[derivation and literature comparison](DESIGN.md).

The electric norm H_E is unchanged. A_h is assembled as
H_E^(-1) C_h* H_B. The same H_B pairs induction work with electric work
and defines the magnetic energy in the stage/interval ledger. C_h and
D_B,h keep the relative topological complex, so D_B,h C_h=0. The
wall-normal magnetic flux is a fixed cap contribution to total
divergence, with zero Faraday rate. The original magnetic inputs can
be reused byte for byte. The changed metric is included in the operator
fingerprint for restart identity. The legacy staircase and stored
toroidal harmonic have their original norm rules.

## Independent physical sampling

The constant axial, affine radial and affine axial tests sample
poloidal flux differences and install the outward wall cap through the
formal solver entry. The largest repaired toroidal current error on
the three meshes is 4.45e-6 A/m², including edges and corners. The
known nonzero affine probes reach O(1e6) A/m². A separate Faraday test
directly samples an analytic flux rate with zero tangential wall E.
It matches the computed induction rate and zero wall cap rate.

For the mixed vacuum field the table gives arithmetic sample RMS in
kA/m². The zero exact total current is recovered with a finite
truncation error on these smooth non-affine fields.

| Grid | First layer original → repaired | Second layer original → repaired | Corners original → repaired | Interior original → repaired |
|---|---:|---:|---:|---:|
| 7×6 | 411.36 → 4.784 | 88.92 → 1.643 | 23.60 → 7.002 | 1.416 → 0.5171 |
| 14×12 | 385.64 → 2.358 | 98.67 → 0.770 | 21.69 → 3.461 | 1.372 → 0.1276 |
| 28×24 | 375.64 → 1.169 | 101.60 → 0.382 | 20.77 → 1.719 | 0.944 → 0.0316 |

The Solov'ev field gives a nonzero current with the same repaired error
magnitudes. Its first-layer archived RMS rises from 315.76 to 350.22
kA/m². The repaired polynomial cases support first-order boundary and
second-order interior consistency of the two corrected derivatives.
This is a deliberate change from the old fourth-order Cartesian
interior stencil limit for those Ampère terms. Other spatial blocks
retain their own consistency scope.

The maximum sampled divergence including caps is 7.9e-12 T/m. In the
separate K=1 test with the actual normal-electric-field reaction, the
maximum D_B,h C_h matrix residual is 7.11e-14 m^-2, total divergence
after Faraday 1.66e-13 T/m, and the normal curl of B at the wall
3.29e-15 T/m. Relative magnetic-plus-electric work mismatch is below
4e-16 in all these tests. Normal current in the coupled model still
uses its existing wall reaction. The analytic current accuracy and nonlinear
evolution comparisons here are axisymmetric.

[Analytic figure](figures/mixed-vacuum-current-convergence.png) ·
[Archived analytic data](evidence/archived.json) ·
[Repaired analytic data](evidence/repaired.json) ·
[Modal closure checks](evidence/modal-repaired.json)

## Explicit time-step check

The largest eigenvalue of A_h C_h grows by 21.6%, 14.9% and 13.2%
on the three analytic meshes. A separate frozen K=0 axial cold Hall
linearization, using the production spatial and wall operators, gives
these RK4 imaginary-axis bounds. This tests the changed discretization
and the cold, pressure-free reference block.

| Grid | Original bound (ns) | Repaired bound (ns) |
|---|---:|---:|
| 6×6 | 6.558 | 6.161 |
| 8×8 | 3.794 | 3.617 |
| 10×10 | 2.462 | 2.354 |

[Archived cold Hall spectra](evidence/cold-archived.json) ·
[Repaired cold Hall spectra](evidence/cold-repaired.json)

## A0 FreeGS halo evolution

The physical inputs are the 200 kA single-null FreeGS equilibrium,
the same finite halo density and the same quiet and radial-pulse
initial arrays. The box, wall flux, and all initial array hashes match.
The old and new 14×20 runs used 50 ps synchronous RK4 steps, with
exact comparisons every 1.6 ns through 8 ns. All existing hard
checks passed at the compared endpoints and stages.

The archived first-layer current replay against the external current
and the new complete initial RHS give these first-layer RMS errors.

| Grid | Original (kA/m²) | Repaired (kA/m²) |
|---|---:|---:|
| 10×14 | 32.272 | 3.219 |
| 14×20 | 33.161 | 2.753 |
| 28×40 | 34.660 | 1.860 |

The 28×40 repaired result is an initial full RHS measurement. The
matched nonlinear evolution in this campaign is on 14×20. At 8 ns
the quiet first-layer RMS relative to the initial external GS current
is 33.114 kA/m² before repair and 2.746 kA/m² after repair.

| 8 ns measure | Original | Repaired |
|---|---:|---:|
| Quiet first-layer ion speed RMS (m/s) | 305.456 | 27.8805 |
| Quiet first-layer ion speed peak (m/s) | 1497.60 | 110.871 |
| Quiet first-layer nodal B drift peak (µT) | 166.52 | 7.00 |
| Quiet core ion speed RMS (m/s) | 208.013 | 263.286 |
| Quiet halo ion speed RMS (m/s) | 50.853 | 49.013 |
| Quiet whole-domain ion speed RMS (m/s) | 203.833 | 149.517 |
| Quiet total energy change from its own initial energy (J) | 2.139e-5 | 2.831e-7 |
| Outboard LCFS displacement, pulse minus quiet (mm) | 0.245179 | 0.245200 |
| Density-gradient centroid displacement, pulse minus quiet (mm) | 0.511939 | 0.513831 |

The quiet LCFS drift is 2.276 µm originally and 2.244 µm after repair.
The largest difference in matched pulse velocity response is 5.21 m/s
against a response of about 80.6 km/s. The corresponding nodal magnetic
gap is about 0.9 µT against 6.49 mT. The core residual grows on this grid even as the wall error falls.
The resulting static
equilibrium is still approximate, and this finite-window agreement
covers the selected pulse and plasma input.

The repaired quiet and pulse cases took 349 s each through 160 steps
on RTX 5090 cards, including setup and output. Instrumented operator
assembly was 0.48 s, process peak RSS 1.24 GiB, and sampled CuPy pool
reservation 254 MiB. The pool is only part of total device use. The
new shared complex128 line factors and float64 radial measure total
10,768 bytes on 14×20. All four successful A0/A1 allocations together
used 16 minutes 39 seconds of one-card time. The two first Slurm
launch attempts stopped before opening the solver because the
spooled script path was used as the campaign directory. Their logs
are retained. The relaunch changed only that launcher path.

[Matched runs and total-state sections](GALLERY.md#a0-freegs-matched-8-ns) ·
[Exact matched statistics](evidence/matched-8ns-matched-repair.json) ·
[8 ns displacement](figures/matched-displacement.png) ·
[28×40 repaired initial RHS](evidence/initial-28x40.json) ·
[FreeGS Ampère sections](figures/current-sections.png)

## A1 on the same spatial implementation

The paired A1 pulse runs used byte-identical 10×14 inputs and a
50 ps electronic/field microstep. Synchronous RK4 accepted 64 steps.
Three-node WR accepted 16 macros of 200 ps through 3.2 ns, with four
50 ps field microsteps per macro and the existing iteration tolerance
of 1e-12. Existing hard checks passed in both paths.

At 3.2 ns, the maximum WR-to-RK4 field differences are 3.31e-6 m/s
for ion velocity, 8.62e-14 T for the reconstructed magnetic field,
1.64e6 m^-3 for ion density, and 5.13e-16 Wb/rad for flux.
The largest velocity difference is 8.11e-10 of the maximum
synchronous change since the initial pulse. Both total energies
change by 4.10e-8 J relative to their respective initial states.
The synchronous and WR elapsed times are 84.8 s and 211.6 s,
respectively. The final WR macro uses four sweeps and 97 field
evaluations including the alias evaluation. The observed end-to-end
time is higher for this WR setting.

A one-step CPU A0-to-A1 synchronous port check on the same 10×14
pulse starts from exactly equal accepted arrays and differs at the
endpoint by at most 6.90e-20 of the declared physical scales.

[CPU port check](evidence/cpu-port.json) ·
[Matched A1 sections](GALLERY.md#a1-matched-pulse-at-32-ns) ·
[A1 numerical and constraint results](evidence/a1-3p2ns-summary.json)

## Reproduction and remaining scope

A0 commit `bc809bd` is on `codex/a0-rect-wall-curl` and A1 commit
`e8fb707` is on `codex/a1-rect-wall-curl`. A0 reports 697 CPU
passes and nine skips, and A1 reports 1241 passes and 31 skips.
Each suite also passes three subtests. Both wheels build.
The original source commits, case input hashes and shipped project locks
identify the actual runs. The tests cover the full
sampled poloidal affine conditions, the sampled Faraday rate, a
nonzero current and a mixed vacuum cancellation, the modal CT and
magnetic work identities, and the existing runtime tests. The
nonlinear matching here uses K=0. The accuracy of the separate
toroidal magnetic closure and larger multimode nonlinear evolution
requires its own physical tests. This campaign does not claim an
A1 speedup, a converged 28×40 trajectory, or a nonlinear stability
limit from the cold linearized calculation.

[Source revisions and input identity](evidence/input-identity.json) ·
[A0 test log](evidence/a0-cpu-suite.log) ·
[A1 test log](evidence/a1-cpu-suite.log) ·
[Analytic reproducibility](README.md#analytic-cpu-control)
