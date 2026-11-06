# A0 prototype known issues

These observations remain evidence and debugging leads. They do not silently
change numerical acceptance criteria or block A1 solely because they exist.
D062 prioritizes reproducible global physical comparisons. Investigate when a
frequency differs by about 10%, a structure differs substantially, energy
imbalance approaches perturbation energy, or a run becomes unstable.

| ID | Observation and retained evidence | Interpretation / next useful trigger |
| --- | --- | --- |
| A0-01 | Fixed-rectangle momentum impulse defect was 1.80974e-21 kg m/s at n6: 20.12% of the nearly cancelling net impulse, 0.03456% of the sum of absolute species impulses. `A0_SPATIAL_MODE_EVIDENCE.md` retains runs and plots. | Boundary/interior coupling is a hypothesis; the original domain was a grid-aligned rectangle. Do not attribute it exclusively to curved staircase geometry. Compare physical traction and species scales when a global mode is affected. |
| A0-02 | Circle n20 shared electron pressure transport check: -6.26297e-8 W versus 1.46918e-17 W strict threshold; direct pressure adjoint was about 1e-22 W. Original hard failure remains archived. | Full shared flux/current transport, not simply the local algebraic adjoint, participates. This is an observe diagnostic in the new prototype cases; monitor cumulative energy relative to the perturbation signal. |
| A0-03 | Fourth-order interior formulas coexist with lower-order wall/interpolation closures and staircase geometry. Global fourth order has not been established. | Keep grid evidence; upgrade only with a demonstrated physical need, rather than assuming tiny residuals prove accuracy. |
| A0-04 | Cold Hall reference uses independent Fourier/Ritz fields and constrained weak-wall reactions. Its wall is not proven equivalent to the ordinary smooth conducting Hall boundary-value problem. | Report polynomial/quadrature/rank stabilization and mode overlap. Literature frequencies require identical parameters and boundaries for quantitative use. |
| A0-05 | Mapping a smooth EFIT or cold eigenfield onto the closed staircase may change B/current/velocity. A GS equilibrium alone does not prescribe separate species equilibria. | Store raw/mapped fields and corrections, initial RHS, and matched unperturbed controls. Do not subtract the equilibrium RHS or label background relaxation a growing mode. |
| A0-06 | Finite-inertia evidence is local, including periodic unforced longitudinal tests; no real Tokamak/global finite-inertia validation. | Retain explicit electron_inertia=1 historical cases. Through 1.0 the development priority is massless/quasineutral electrons. |

| A0-07 | EAST12 raw energy imbalance reaches 16.6%/24.1% of the base mode signal; common-background subtraction reduces this to 1.47e-5/2.71e-6, and half dt reduces the raw gap by about 32x. | Keep both scales visible. Real-geometry response frequencies are supported; long-time nonlinear energy accuracy is not established. |
| A0-08 | Circle fast-branch reference overlaps at grid16 are 0.604/0.652; frequency differences from order-8 cold reference are +5.86%/-4.37%. Slow candidates remain archived. | Two grids and a qualified weak-wall reference do not prove global convergence or identify every branch. User prioritized fast modes; slow classification is deferred. |
| A0-09 | The fixed-circle mask area and fixed-flux mapped B amplitude change nonmonotonically with grid. At12/16/20/24, fitted amplitude errors are +0.923%/-3.479%/-0.631%/+0.986%. | Raw frequency refinement includes geometry and input-mapping changes. Preserve raw frequencies and report both errors; no causal split or rescaled-frequency acceptance follows from this audit. |
| A0-10 | Archived EAST12 eight-cycle seed-subspace coherence falls to0.940963/0.917186 for n0/n1; circle12/16 stays at least0.99999975. | EAST response departs from its baseline linear seed. Track evolving structure and relaxing controls; do not present a single fitted frequency as proof of a stationary global eigenmode. |
| A0-11 | EAST12-to16 individual branch continuation is weak: best classified overlaps0.21136/0.27410; all853 positive-imaginary n0 raw eigenpairs still give only0.21136 forward maximum. The old parallel fraction0.52794 was near the0.5 classifier cutoff. | No strong converged EAST branch is established. Preserve nearby excluded modes, reverse-interpolation sensitivity and pending finer grids; do not change the classifier or silently connect different candidates. |
| A0-12 | Dense32-grid full spectra are estimated at19-28 wall-clock hours per sector in the measured4-CPU configuration; a K1 storage plan exceeds the RTX5090 memory cap. | This is a current implementation/analysis scaling limit. Retain32-grid setup/step probes; no32-grid frequency or actual-mode acceptance. It does not imply a Warp-versus-native speed claim. |

Nonfinite state, nonpositive density/pressure, failed numerical solves, broken
ownership/restart identity, and necessary model constraints still stop runs.
No energy residual is removed by adding a numerical wall reaction to physical
boundary power. All cumulative energy uses the actual accepted RK4 ledger.
