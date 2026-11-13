# `tests/warp/test_ghost_toroidal.py` implementation pseudocode

1. Independently construct a full rectangular vertex chart by applying
   nonidentity graph factors only to the interior rectangle. Integrate
   polynomial (F) over physical half/full areas and choose a zero-total-flux
   perturbation. Add a directly stored harmonic.
2. Reconstruct a batch of complex states and compare against the known
   polynomial and harmonic values at all cell centers and both sets of
   physical wall midpoint traces from the stored area averages.
3. Verify the weighted harmonic maps to the dual area vector. Verify the
   Hermitian adjoint and positive definiteness on a small vertex space.
4. On radial refinement with true cylindrical energy weight, check the
   reconstructed curl load from axial and radial polynomial variations,
   including the first two layers. Check the axial affine derivative to
   roundoff away from the old electric edge transform.
