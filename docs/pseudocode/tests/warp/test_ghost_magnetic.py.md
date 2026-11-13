# `tests/warp/test_ghost_magnetic.py` implementation pseudocode

1. Build a uniform line with both boundary half-links and an independent
   representation of the existing graph filter. Scale the radial chart by
   the mean radii in the same orientation as the stored magnetic field.
2. Use Gaussian quadrature on each physical link to sample constant,
   affine and quadratic fields. Integrate with the physical cylindrical
   radius on radial links.
3. For a two-cell line, including a small positive lower radial wall,
   check metric symmetry, positive eigenvalues, exact constant action and
   exact affine derivative after transforming the norm back to link means.
4. Check the axial quadratic derivative throughout both a two-cell and an
   eight-cell line.
5. On radial lines of 8, 16 and 32 cells, require each refinement to reduce
   the quadratic derivative error in the first and second layer of both
   walls by a factor greater than 1/0.3.
