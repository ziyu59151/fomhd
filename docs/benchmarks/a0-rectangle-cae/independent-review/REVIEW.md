# Independent checks of the $n_\phi=1$ rectangular Hall compression branch

The selected comparison has $R_0=15$ m, $R=14.4$–$15.6$ m, $Z=\pm0.6$ m, $B_\phi=30/R$ T, uniform deuterium number density $2\times10^{19}\,\mathrm{m}^{-3}$, and fixed conducting walls. The vacuum toroidal guide field gives zero equilibrium current and constant density and pressures. The independent spatial Ritz eigenproblem is cold Hall. Full A0 evolution uses a small positive pressure and its regular Hall/CGL-B path. The $\delta B_\parallel$ comparison uses the toroidal component with cylindrical $R$ volume weighting and a phase-invariant overlap.

## Same-box eigenvalue and shape

| Continuous Ritz order | Narrow box (MHz) | Wide box (MHz) |
| ---: | ---: | ---: |
| 12 | 6.162656 | 4.925015 |
| 14 | 6.170819 | 4.932865 |
| 16 | 6.176967 | 4.938820 |

The wide control changes both halfwidths from 0.60 to 0.75 m. The overlap between orders 14 and 16 is 0.998944 in the narrow box and 0.998774 in the wide box. The order-16 relative normal-current traces are $3.19\times10^{-12}$ and $3.31\times10^{-12}$, respectively. The narrow-box branch has magnetic compression fraction 0.996595. [Narrow box data](../reference/ritz-R15-n2e19-a0p6.json) · [Wide box data](../reference/ritz-R15-n2e19-a0p75.json)

| Frozen A0 grid | Matched frequency (MHz) | $\delta B_\parallel$ overlap against order 14 |
| ---: | ---: | ---: |
| 12×12 | 5.559820 | 0.7817 |
| 14×14 | 5.623094 | 0.7791 |
| 16×16 | 5.648499 | 0.7601 |
| 20×20 | 5.665783 | 0.7312 |

The 16→20 frequency difference is 0.31%. The 20×20 frequency is 8.18% below the order-14 result, and global $\delta B_\parallel$ overlap declines with this refinement. For a fixed 0.15 m wall collar at 20×20, the core overlap is 0.866 and the collar overlap is 0.602. The collar contains 0.543 of frozen A0 $\delta B_\parallel$ energy and 0.275 of the sampled reference. The same mismatch is visible on the coarser comparisons. The split locates a substantial part of the disagreement near the physical wall and leaves an interior discrepancy. The final mode comparison includes the whole box. [Collar data](R15-Bparallel-collars.json) · [Phase-aligned sections](R15-Bparallel-shapes.png) · [20×20 frozen spectrum](g20-R15-n2e19-a0p6-p14.json)

## Rectangular aspect ratio control

At fixed radial halfwidth 0.60 m and $R_0=15$ m, I tested axial halfwidths 0.40 and 0.80 m with gauge-clean order-12 Ritz references and matched equal-$h_R,h_Z$ A0 frozen Jacobians on four grids for each box. This tests whether separating the radial and axial length scales improves branch identification.

- At axial halfwidth 0.40 m, the lowest compressional reference is 7.080767 MHz. A0 shifts from 7.640 MHz at 12×8 to 7.991 MHz at 21×14, while overlap declines from 0.537 to 0.454. A higher tracked branch reaches 0.614 at the finest grid with a frequency error of about 14%.
- At axial halfwidth 0.80 m, the lowest compressional reference is 5.155077 MHz. The 18×24 A0 counterpart is 5.805428 MHz with overlap 0.463. A second branch is closer in frequency, 5.59408 MHz versus 5.39444 MHz, with overlap 0.563.

Neither change improves on the original square comparison. The square case provides the stronger bounded evolution test. The frozen A0 and Ritz evidence sets the accuracy limit of that test. Its selection alone establishes neither nonlinear persistence nor a converged quantitative mode benchmark. [Narrow rectangular data](nonsquare-R15-n2e19-a0p6-b0p4-results.json) · [Wide rectangular data](nonsquare-R15-n2e19-a0p6-b0p8-results.json) · [Equal-side specialization check](nonsquare-square-specialization.txt)
