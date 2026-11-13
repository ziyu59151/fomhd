# Rectangular conductor with two-layer ghost continuation

Select the new massless Hall/CGL-B spatial closure with

```toml
[boundary]
kind = "fixed_outer_wall_static"
potential = 0.0
version = "rectangle_ghost_v1"
```

The physical walls are at the outer RZ faces, half a cell from the
nearest center. The LCFS is an internal marker. The same initial NPZ
format supplies the stored magnetic field and the fixed normal flux on
the two halves of each wall segment. Archived `rectangle_relative_v1`
configurations still select the original paired closure. Start a new run
from the same initial file for a comparison. The two closures have
distinct operator and restart identities.

HYM's two exterior layers and stage refresh provide the implementation
pattern. TKIN applies point formulas at physical wall nodes. FOMHD fits
the physical half/full-link moments and uses two exterior intervals for
magnetic reconstruction. Two affine exterior centers supply the material
face interpolation. The resulting rows act on each RK4 stage's own
fields and densities. The two-layer values are eliminated at setup.
The fitted one-sided magnetic rows and the paired curvature norm have
separate effects. The relative Faraday complex supplies zero tangential
electric trace and preserves the initial normal flux. The new magnetic
norm pairs the same Faraday operator with Ampère and magnetic work.

Use `core.reconstruct_magnetic` on stored magnetic fields and rates. For
the ghost closure, the sparse `QB` holds the poloidal transfer, and the
common reconstruction combines that transfer with the factored toroidal
dual-area deaverage. The wall trace uses the same dechart and the physical
evaluation locations. See the [operator pseudocode](pseudocode/src/fomhd/operators.py.md)
and [case and restart pseudocode](pseudocode/src/fomhd/io.py.md).

The axisymmetric analytic check uses the existing 2 T cold Hall axial
reference on a fixed 1.2 m by 1.2 m conducting box, with 1 Pa positive
initial pressures in the solver. It checks uniform background, two
quadratures, and the unchanged guarded time step at three spatial
resolutions. The unprojected toroidal phase is also useful for separating
the sampled Ampère error from the initialization projection. A sampled
continuous field can violate the discrete normal-current condition. The
independent quarter-phase check records this rejection and the applied
compatibility correction separately. The multi-cycle replay starts at the
poloidal phase and uses byte-identical physical inputs to the legacy run.

The mathematical definition and source-reading record are in the sibling
design repository, `A0_GLOBAL_DISCRETIZATION.md`, section D078. The
same-box [axial Hall reference](benchmarks/a0-rectangle-cae/reference/AXIAL_HALL_REFERENCE.md)
defines the analytic standing wave and its boundary conditions.
