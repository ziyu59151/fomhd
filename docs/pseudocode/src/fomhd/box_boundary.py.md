# Grid-aligned outer wall source flow

Source: [box boundary](../../../../src/fomhd/box_boundary.py).
Design: [D075 RZ outer box](/home/lxzeng/fomhd-design/mainline_design/implementation/A0/A0_GLOBAL_DISCRETIZATION.md#d075-grid-aligned-rectangular-outer-wall).

1. `load_case` selects `fixed_outer_wall_static / rectangle_relative_v1`
   or `fixed_outer_wall_static / rectangle_ghost_v1` for the massless model.
   `build_aligned_box` places the physical faces
   half a spacing beyond the first and last cell centers in R and Z.
   It builds the existing full-cell topology on every center and retains a
   validated, strictly internal LCFS. The historical boundary selector
   continues to take its original mask and hash path.
2. The optional initial NPZ `wall_magnetic_flux` contains complex128 Fourier
   coefficients of outward flux in Wb/rad with shape
   `(2*K+1, number_of_wall_segments, 2)`. The final axis indexes the two
   halves of the segment in its vertex order. The wall list is r faces
   then z faces in geometry order. Absence means all-zero prescribed flux.
   Reject invalid shape, dtype, nonfinite values and broken Fourier symmetry.
   The raw file is covered by the source and restart identity.
3. Install immutable geometry-only trace rows and backend-owned cap values.
   Accumulate each half's flux at its corresponding dual vertex. At corners
   include both sides. Divide by the cylindrical vertex volume, then
   apply the existing T3 transformation. No new time-state component is
   allocated. Initial and stage checks use physical total-field divergence
   `DB @ B + cap`. Rate/increment checks keep homogeneous `DB @ Bdot`.
   The relative curl has zero tangential electric wall cochains, so the
   prescribed normal magnetic flux remains fixed. The separate normal
   current solve uses the checked fixed matrix, selected rows, and LU in
   `wall_cache.py` and checks every original row at each stage.
4. The actual electron RHS uses the wall pressure traction. Assemble the
   magnetic trace with the prescribed normal field mean and tangential
   values from the adjacent field DOFs in the legacy closure or from the
   two-ghost reconstruction at physical wall locations in the ghost closure.
   Along the wall use the center transfer. Across the wall use the end
   transfer on the same clipped dual means as the cell reconstruction.
   Evaluate the pressure tensor on
   that trace, multiply by surface measure, and apply the negative weighted
   transpose of the density-weighted velocity trace. The electron
   gradient/strain stage remains the volume term. The wall normal data,
   density and pressure trace are the same for force, momentum and energy.
   The diagnostic residual includes the corresponding outward work.
5. RHS and diagnostics run on the selected array backend. Configuration
   and flux validation are on the host at setup. For restart, the setup is
   repeated from the hashed original input before the accepted state is
   loaded. RHS checks own failures and the existing time transaction
   alone commits state, cache or integrated work.

Tests: [formal entry](../../../../tests/warp/test_rectangular_wall_entry.py)
and [physical cap/pressure traces](../../../../tests/warp/test_rectangle_magnetic_flux.py).
