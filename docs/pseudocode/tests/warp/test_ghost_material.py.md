# Rectangular material line verification

Source [material line tests](../../../../tests/warp/test_ghost_material.py).

1. Directly sample a constant and an affine function on two-, three-, and
   larger cell-center lines. Check derivatives at every center.
2. Test the discrete Green identity and the affine pressure force with
   independently specified physical wall traces and random complex values.
3. Independently extrapolate sampled values into both ghost layers and
   apply the centered interpolation rule at all faces. Check the small-line
   case and constant and affine reproduction at every face.
4. Reject inadequate line sizes and invalid derivative spacing.
5. With an actual rectangular geometry, verify rigid toroidal rotation,
   constant and axial affine pressure force including the wall traction,
   and the physical weighted work identity at variable density.
