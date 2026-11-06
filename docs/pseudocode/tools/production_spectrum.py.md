# Production linear spectrum (CPU analysis)

This is a diagnostic of the production discrete RHS, not the independent cold
Hall reference. It neither replaces the evolution stage nor supplies a source.

1. Construct the requested physical background with the ordinary global-case
   builder. At one toroidal harmonic form independent variables: ion density,
   wall-tangent ion momentum, ion pressure, electron perpendicular/parallel
   pressure, and B in null(DB,S*Cdag[,topological-flux]). Normalize each block
   with its declared physical SI reference.
2. Use central differences of the complete production massless stage at
   epsilon=1e-5. Each n0 column is a real zero coefficient; n1 uses physical
   conjugate pairs epsilon/2, and its positive RHS coefficient is multiplied by
   two. Electron density/current compatibility remains derived by production.
3. Project the returned RHS onto the same basis. Save the entire scaled
   Jacobian, eigenspectrum, maximum absolute eigenvalue and maximum growth,
   positivity/background residual, and exact numerical-module hashes.
4. Identify finite positive-imaginary eigenvalues, reject large relative growth,
   and select the lowest with magnetic energy fraction >.1 and parallel magnetic
   fraction >.5. Keep zero modes and other candidates in metadata. Save the
   selected physical state direction and its actual eigen residual. Its use as
   an initial direction is distinct from the independent Fourier/Ritz oracle.
5. Repeat a coarser/half-difference-size calculation when a suspect eigenvalue
   or growth appears. A spectral radius is a baseline linear estimate, not a
   nonlinear stability theorem. A driver using it retains a stated margin,
   tracks departure from the background, and preserves conservative operator
   estimate warnings separately. It must inspect both n0 and n1 when K=1.
