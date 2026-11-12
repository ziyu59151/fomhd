# A0 rectangular-wall compressional Alfvén benchmark

## Physical case and separate evidence

The synthetic equilibrium fills an axisymmetric box with a curl-free
toroidal guide field $B_\phi= B_0R_0/R$, spatially uniform density
$2\times10^{19}\,\mathrm{m}^{-3}$, stationary deuterium and constant
isotropic pressures. $R_0=15\,\mathrm{m}$ and $B_0=2\,\mathrm{T}$.
The solver retains $n_\phi=0,\pm1$ with five angular quadrature samples.
The positive pressures give total axis beta $10^{-5}$. The exterior
wall has the repaired relative E/B representation. The internal marker
in the geometry file carries no plasma boundary in this vacuum-field test.

Bradshaw's Chapter 4 develops cylindrical Hall compression branches
from the homogeneous circular theta-pinch. Chapter 5, Figure 5.3,
compares mode numbers and frequencies for a circular toroidal
cross-section. The circular examples motivate branch identification by
mode structure and polarization. Their frequencies belong to their
own geometry and equilibrium.
[Whales2 chapters 4 and 5](https://wrap.warwick.ac.uk/161400/1/WRAP_Theses_Bradshaw_2020.pdf#page=99)

The reference is a separate cold Hall Ritz solve in this same physical
box. It uses analytic cylindrical curl and volume quadrature, matching
homogeneous perturbation normal flux, zero normal current, zero normal
velocity, and the associated weak wall reaction. Polynomial order 12,
14 and 16 gives 6.162656, 6.170819 and 6.176967 MHz for the narrow box.
The last two orders give 0.998944 parallel magnetic shape overlap and
the p16 shifted-wall normal current residual is $3.19\times10^{-12}$
of the peak current. The wider box reference is 4.938820 MHz at p16.
The weak wall subspace and finite pressure in the full solver are stated
here so the comparison can be repeated exactly.

Scaling the Hall block alone from zero to its physical value in the
same narrow-box Ritz subspace shifts this mode from
6.178261 to
6.176967 MHz
(-0.0209%). Successive
whole-magnetic overlaps along the continuation exceed 0.9998. This
isolates the contribution of that block in the cold reference at
fixed normal traces. The accepted A0 trajectories all use the physical
Hall term. [Hall coefficient continuation](hall-reference-shift.json)

The initial perturbations use the matched frozen cold mode of the
production spatial operators. In each run the actual frequency and
structure are fitted from accepted RK4 snapshots. The selected
narrow-16 frozen mode is 5.648499 MHz. It is an input branch selection
and a numerical check on the evolution. The independent benchmark is
the separate Ritz frequency and field. The three evidence types have
separate labels in the gallery.

## Accepted runs

| Case | Step | Accepted time (ns) | Measured cycles | Fitted frequency (MHz) | Ritz frequency (MHz) | Parallel B overlap, full / ≥0.20 m from wall |
|---|---:|---:|---:|---:|---:|---:|
| Narrow 12×12 | 195 ps | 748.800 | 4.163 | 5.559849 | 6.176967 | 0.7840 / 0.9450 |
| Narrow 16×16 | 130 ps | 748.800 | 4.230 | 5.648532 | 6.176967 | 0.7635 / 0.9518 |
| Narrow 16×16 half step | 65 ps | 374.400 | 2.115 | 5.648532 | 6.176967 | 0.7635 / 0.9518 |
| Wider 20×20 | 130 ps | 748.800 | 3.405 | 4.547351 | 4.938820 | 0.8397 / 0.9632 |

The narrow 16×16 fit has phase RMS 1.59e-06 rad,
relative amplitude drift -5.06e-06, and
parallel magnetic fraction 0.999819.
The sampled independent field has parallel fraction
0.998861. The relative phase
of the ion-velocity and magnetic projections differs by
+1.1e-07 rad.
The component-resolved polarization data are in the mode summary. Frequency moves
+1.595% on fixed-box refinement.
The two accepted modes have common-support parallel shape overlap
0.9538 and interior overlap
0.9821. At 16×16 the
signed frequency difference from the independent solution remains
-8.555%. The full-box
shape has a larger difference than the interior. In the 0.20 m wall
collar, the parallel magnetic shape overlap is
0.6496. This region contains
64.9% of the accepted
parallel magnetic energy and
34.5% of the sampled
reference. The full-box velocity shape overlap is
0.6415. The branch and interior structure
have a reproducible comparison, with quantitative frequency, near-wall
magnetic shape and velocity-shape accuracy still unresolved.

Halving the step on 16×16 gives 49 common accepted
times through 374.400 ns. The maximum
relative magnetic perturbation gap is 4.7e-11,
the maximum relative velocity perturbation gap is
4.71e-11, and the phase gap is
4.7e-11 rad. The same-box discrepancy is
therefore much larger than the time-step sensitivity observed here.

The physical walls move from ±0.60 m to ±0.75 m around the box center
while the cell spacing remains 0.075 m. The full RK4 fitted frequency
changes -19.495% compared
with -20.045% in the
independent reference. This compares the tracked compression mode
across boxes at the same spacing and time step.

The quiet 16×16 run shares the physical input and times of the
perturbation. At the last common save its peak velocity drift is
9.42e-08 m/s and peak magnetic
drift is 1.33e-15 T. The gallery
shows absolute physical fields before the matched subtraction. The
cell-centered electric field is reconstructed through the normal
stage and H transfer from the accepted state. The absolute density is
around $2\times10^{19}\,\mathrm{m}^{-3}$. The difference column
subtracts the two scalar magnitudes where the row shows a magnitude.
The wider box has a separate 8-step CPU quiet
control through 1.04 ns. Wider and coarser
absolute plots use the initial static background.

The five full runs recorded 2,626,671, 3,939,951,
3,939,999, 3,939,951 and 3,939,951
passing hard checks, respectively, with all final endpoints accepted.
Check receipts retain measured divergence, wall current, normal flux,
pressure and electric work, energy interval and step-estimate values.

## Cold Hall short control and cost

The older axisymmetric axial cold Hall oracle supplies a separate
control. The repaired source completes 120 steps of 100 ps through
12 ns. At the endpoint its phase differs by -0.008697 rad from
the analytic value and the energy-scaled error is
6.34%. The reference frequency is
10.991395 MHz. The short control is too
short to fit an independent numerical frequency with the usual
multi-period criterion.

In the matched 16×16 pilot, archived and provisional optimized A0
used identical input bytes on two RTX 5090 allocations of the same
model. Full-step medians were
3.720 and
0.633 s.
The peak sampled CuPy reservations were
458.6 and
293.9 MiB.
The largest scaled primary-state difference after eight steps was
1.32e-20.
This pilot and the full nonlinear mode campaign preserve separate
source hashes. The long runs use the frozen numerical scaling source
identified in their provenance.

[Gallery](index.html) · [Mode and controls data](control-comparison.json) ·
[Source and reproduction commands](README.md) ·
[Independent branch review](independent-review/REVIEW.md) ·
[Run ledger](RUN_LEDGER.md) · [Archived campaign receipt](verified-campaign.json)

## Spatial scales of the selected mode and the next fixed-box grid

The independent cold Hall Ritz solution gives the complex `n_phi=1`
coefficient of δB∥ on the same 1.20 m × 1.20 m box as the narrow
A0 case. Since the equilibrium field is toroidal, the comparison uses
δBφ. The directional effective wavelength is

$$
k_i^{\rm rms} =
\left(\frac{\int R |\partial_i \delta B_\parallel|^2\,dR\,dZ}
{\int R |\delta B_\parallel|^2\,dR\,dZ}\right)^{1/2},
\qquad \lambda_i^{\rm eff}=2\pi/k_i^{\rm rms},\quad i\in\{R,Z\}.
$$

This derivative-based scale summarizes a standing pattern with several
lobes. The connected half-maximum width measures the central dominant
lobe along each coordinate through its antinode. Each interval ends at
the first point where the complex magnitude falls to half its peak.
Both scales come from the continuous Ritz representation and cylindrical
volume measure, without interpolation from the A0 grid. The script
checks the reconstructed polynomial against the independent field
evaluator at off-grid points.

| Independent Ritz order | λ_R, λ_Z [m] | Dominant half-maximum width R, Z [m] |
|---:|---:|---:|
| 12 | 1.496, 1.609 | 0.529, 0.559 |
| 14 | 1.512, 1.595 | 0.527, 0.548 |
| 16 | 1.523, 1.585 | 0.532, 0.547 |
| 18 | 1.530, 1.578 | 0.532, 0.545 |

The order-16 antinode is near `(R−R0,Z)=(0.0763,0) m`.
Restricting the directional norms to points at least 0.20 m from
the walls gives effective wavelengths of 1.538 and 1.569 m.
Orders 12 through 18 give a spread below 2.5% for each reported
scale. The directional first derivatives of δB∥ describe the main
pattern. The derivative accuracy of the full vector field and the
resolution of wall currents require their own checks.

| Grid | ΔR = ΔZ [m] | Cells per effective wavelength R, Z | Cells across half-maximum width R, Z | Evidence |
|---|---:|---:|---:|---|
| 12×12 | 0.1000 | 15.2, 15.8 | 5.32, 5.47 | Full RK4 run on the fixed box |
| 16×16 | 0.0750 | 20.3, 21.1 | 7.10, 7.30 | Full RK4 run on the fixed box |
| 20×20 | 0.0600 | 25.4, 26.4 | 8.87, 9.12 | Frozen spectrum on the fixed box |
| 24×24 | 0.0500 | 30.5, 31.7 | 10.64, 10.94 | Suggested fixed-box measurement |
| 32×32 | 0.0375 | 40.6, 42.3 | 14.19, 14.59 | Subsequent conditional measurement |

The 1.50 m wide box at 20×20 uses 0.075 m spacing. Its own
order-16 Ritz wavelengths are 1.900 m in R and 1.979 m in Z
(25.3, 26.4 cells), with widths of 0.665 m and 0.682 m
(8.87, 9.10 cells). It tests a change in the physical wall position
at fixed spacing. The fixed-box 20×20 row tests spatial refinement.

The same cuts quantify a shorter magnetic structure next to the
walls. Starting at each face, the order-16 distance to the first
half of its wall amplitude is 0.165 m at the inner radial face,
0.061 m at the outer radial face and 0.099 m at each axial face.
These distances span 2.20, 0.81 and 1.32 cells at 0.075 m, and
3.29, 1.21 and 1.98 cells at 0.050 m. The outer radial value
changes from 0.041 m at order 12 to 0.068 m at order 18.
It gives an additional resolution diagnostic for the magnetic
shape. Its convergence and the resolution of the current layer
require separate evidence.

The existing fixed-box frozen spectra track the compressional
branch from 5.560 MHz (12×12) to 5.648 MHz (16×16) and 5.666 MHz
(20×20). The change from 16×16 to 20×20 is 0.31%, while the
20×20 frequency remains 8.28% below the order-16 independent
frequency of 6.177 MHz. The central lobe has overlap near 0.95 at
12×12 and 16×16 on the fixed d≥0.20 m interior, and the full-box
overlap with the order-14 reference goes from 0.782 through 0.760
to 0.731. With the frozen fields bilinearly interpolated onto
exactly the same physical support and compared to the order-16
reference, the interior overlaps are 0.943, 0.936 and 0.920.
The overlaps over the common 0.05–0.20 m wall band are 0.674,
0.615 and 0.570. These comparisons support the identification
of the strongly compressional branch and its broad central
pattern. They show that the small change in frozen frequency
from 16×16 to 20×20 has yet to bring spatial convergence.
The whole-box shape, quantitative frequency and wall-current
convergence remain open.
Sampling the independent mode retains δB∥ overlap above 0.9996
before the full discrete evolution. Applying the production mapping
and current transfer gives collar J RMS differences of 52.4% and
34.8% on 12×12 and 16×16 relative to order 16, with 1.50% and
0.67% in the interior. The independent order-14 and order-16 J
fields themselves differ by 28.7% RMS in that collar, and orders
16 and 18 differ by 32.2%. The independent fraction of weighted
|J|² within 0.05 m of the wall grows from 32.1% to 68.3% over
orders 12 to 18. This check locates a derivative-sensitive region.
The Ritz sequence has yet to establish a converged physical
current-layer width. The order-18 current in the 0.05 m strips
is primarily toroidal, parallel to the equilibrium field. This
current component does not contribute to the cold linear Lorentz
force on this background. See the component and wall-trace audit
for the separate poloidal-current comparison.

Use 0.050 m spacing on the same physical box for the next fixed-box
comparison. This gives at least ten cells across the measured dominant
lobe even at the smallest width among the Ritz orders, and
increases the number of cells in the 0.20 m collar to four. The
0.05 m strip spans one cell at this spacing. Ten cells
is a planning target for this lobe. Use an initialized few-step
measurement for the time guard, memory and wall/current errors first.
Then compare the tracked mode frequency, interior and collar shape,
and local current with the fixed physical masks and independent
reference. Repeat at 0.0375 m if the fixed-box trend calls for a finer
comparison. A persistent collar error despite a resolved broad lobe
calls for an operator-level check on directly sampled fields before
committing another full-duration run.

At 0.050 m the existing guard extrapolates Δt≤69.4 ps, or at
least 10,784 steps for the same 748.8 ns observation window. At
0.0375 m it extrapolates Δt≤41.5 ps and at least 18,058 steps.
The current lean setup estimator gives 1.11 and 3.44 GiB
respectively. These are planning estimates from the current guard
and storage model. The separate guarded 24×24 nonzero-mode
Solov'ev control accepted three 15 ps steps at 1.085 s per warm
step, with 2,100 MiB sampled process VRAM, 637.6 MB of tracked
operator arrays and 0.910 s for assembly including inventory.
Using this per-step cost and the predicted step count as a rough
budget gives about 3.25 GPU-hours for the mode's 748.8 ns window
before any extra snapshot cost. The control establishes capacity
for this grid count on that input. The 24×24 mode itself still
requires the fixed-box checks above. The remaining composed
field operators and wall constraint work, together with the
smaller explicit Hall step, determine the cost of this refinement.

[Reference mode and cuts](ritz-spatial-scales.png) ·
[Scales and grid counts](ritz-spatial-scales.json) ·
[Measurement script](methods/ritz_spatial_scales.py) ·
[Frozen modes on common physical support](ritz-frozen-common-support.json) ·
[Reference mapping and current check](reference-mapping-audit.json) ·
[Ritz current convergence](reference-current-convergence.json) ·
[Weak-wall component and trace audit](independent-review/WALL_CURRENT_WEAK_TRACE.md) ·
[Guard and setup projections](next-grid-guard-projection.json)


The near-wall current differences also depend on current orientation.
In this cold linear toroidal guide field, only (J_R,J_Z) contribute
to the Lorentz and Hall coupling with the background. The mapped
A0 relative RMS of this perpendicular current in the 0.20 m collar
improves from 13.38% at 12×12 to 9.34% at 16×16, with 1.17% and
0.295% farther inside. In the independent reference alone the
perpendicular collar difference is 11.03% from order 14 to 16 and
9.76% from order 16 to 18, after common parallel-field phase and
scale alignment. The order-18 5 cm side-strip current is 97.5%
toroidal in current-squared measure. It is parallel to the guide field.
The physical lobe spacing recommendation follows the magnetic
compression scale. The derivatives and the strong-wall momentum
trace have a separate evidence limit.

[Weak cold wall and current orientation](independent-review/WALL_CURRENT_WEAK_TRACE.md)
