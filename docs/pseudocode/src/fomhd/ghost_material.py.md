# Rectangular material line operators

Source [ghost material](../../../../src/fomhd/ghost_material.py).

Both routines build fixed host-owned `float64` coefficient arrays at setup.
Columns are physical cell centers in increasing coordinate order. The caller
owns device placement and applies the same coefficients to each stage's own
samples. There is no time-state storage or device synchronization here.
Require at least two cells and a finite positive spacing for the derivative.

1. `material_face_interpolation` extends the center line with two ghost
   centers per end. Starting at the lower end, the nearest ghost uses
   `2*f0 - f1` and the next uses `3*f0 - 2*f1`. Reverse the line at the upper
   end. Apply `(-1, 9, 9, -1)/16` to each consecutive four-center window.
   Eliminate ghost values in the resulting `(count + 1, count)` matrix.
   The first and last rows are the physical affine traces. Two physical cells
   suffice to construct both extensions and the single interior face row.
2. `material_line_derivative` builds a skew matrix with upper offsets
   `+1: 3/4` and `+2: -1/8`. Add `1/8` to each endpoint nearest-neighbor
   coefficient, with both additions on the same entry for two cells. This
   is the centered fourth-order skew operator with the adjacent endpoint
   updates and the sum of the three-center skew corrections collected into
   its bands. Add `(tau_upper* tau_upper - tau_lower* tau_lower)/2` with
   the affine wall rows, and divide by spacing. The result has units `1/m`
   for a spacing in meters. It differentiates constants and affine data
   exactly under uniform midpoint weights. The centered interior derivative
   is second order. Cylindrical weights and state-dependent trace correction
   belong to the caller.

Tests [material line checks](../../../../tests/warp/test_ghost_material.py).
