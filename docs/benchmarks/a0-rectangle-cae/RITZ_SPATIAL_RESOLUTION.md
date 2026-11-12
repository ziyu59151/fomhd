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
