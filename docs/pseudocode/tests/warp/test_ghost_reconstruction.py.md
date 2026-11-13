# `tests/warp/test_ghost_reconstruction.py` implementation pseudocode

1. Independently sample powers of the physical coordinate by Gauss quadrature
   over every link. Radial samples use the true cylindrical measure.
2. Apply a nonidentity line graph chart to the sampled fields. The production
   reconstruction must recover physical values at all centers and both walls
   from the stored chart, including constant fields and three-link support.
3. Build the formal two-ghost rectangle with Fourier cutoff 0 and 1. Install
   analytically integrated outward cap fluxes for a uniform axial field.
   Check center and wall values of the guide field plus toroidal harmonic.
4. Apply Faraday to an admissible relative electric vector and verify its
   increment preserves magnetic divergence with the same fixed caps. At the
   highest retained mode compare the modal Ampère and Faraday power pairings.
5. Apply the production velocity gradient to rigid toroidal rotation. Check
   the zero symmetric gradient and zero normal momentum trace on every wall.
