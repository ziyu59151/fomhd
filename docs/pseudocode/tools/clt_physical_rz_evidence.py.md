# `tools/clt_physical_rz_evidence.py` pseudocode

## Scope

Load two explicit, already-produced initial-state/rigid-test evidence bundles:

1. valid repair3 structured-intercept evidence, including the corrected positive-M0
   rigid-rotation strain;
2. old-formal self-circle initialization at N8/N12, including the full formal RHS for
   ion momentum, perpendicular electron pressure, and toroidal magnetic field.

This module never calls a solver, stage, time step, GPU, or network. It hashes and
copies explicit evidence only. It does not treat nonzero initial RHS as an evolved
state or compare R0/R2 candidates.

## Repair3 rigid evidence

Inputs:

```text
--structured-json
--structured-npz
--structured-producer
--structured-status
```

Require the repair3 STATUS schema/status, zero time steps, stage not called, and
STATUS-bound producer/JSON/NPZ hashes. Require the JSON probe hash to equal the
producer input.

For N8/N12/N16, load only:

```text
staircase_old_cell_R/Z
staircase_old_symmetric_strain_magnitude
staircase_candidate_cell_R/Z
positive_rigid_symmetric_strain
```

The positive strain magnitude is

```text
sqrt(sum(abs(positive_rigid_symmetric_strain[0])**2, axes=(tensor axes)))
```

The pair is explicitly labeled **old staircase vs corrected positive-M0 candidate**.
It is valid repair3 evidence but remains a failed physical rigid-rotation gate.

For each N8/N12/N16 grid, derive the true circle from that R3 fixture rather than
from the formal self-circle:

1. take the regular cell-centre extents from the R3 staircase/candidate arrays;
2. expand each axis by half the fixture spacing;
3. validate the spacing against `rz_trends.cases[*].h_m`;
4. require the R/Z extents to form the same square domain across all grids;
5. set the circle centre to the domain centre and its radius to the domain half-width.

Record the derived domain, centre, radius, and `h` with each grid. Draw the old
computational wall as exact axis-aligned cell edges at `centre ± h/2` wherever the
cross-cell neighbour is absent. Do not use a finite-mask contour, which can introduce
diagonal corner connectors.

## Old-formal initial RHS

Inputs:

```text
--old-formal-root
--old-formal-audit-json
--old-formal-status
--old-formal-producer
--old-formal-implementation
```

Require the equilibrium-audit STATUS schema/status and:

```text
STATUS.artifacts.staircase_source.sha256 == implementation hash
STATUS.artifacts.mapping_json.sha256 == aggregate mapping JSON hash
producer imports prepare_staircase_equilibrium
```

For g8/g12:

1. Validate `source_manifest.input_hashes` against the current `case.toml`,
   `geometry.npz`, and `initial.npz`.
2. Validate the per-grid mapping input hashes and the aggregate mapping input hashes.
3. Load `initial_rhs.npz` with `allow_pickle=False`; reject object/nonfinite arrays.
4. Use the positive `n_i` initial-state support to identify active cells and draw the
   old staircase wall as exact active-cell edges at `centre ± h/2` with missing
   neighbours. Do not infer support from the RHS or use a finite-mask contour.
5. Plot raw, unnormalized SI RHS values:

```text
|rhs(mom_i)|            kg/(m^2*s^2)
rhs(p_perp_e)          Pa/s
rhs(B_phi)             T/s
```

`mom_i` is the L2 magnitude over its three cell-centered vector components.
`p_perp_e` stays at cell centers. `B_phi` stays at its existing vertex positions.
No RHS difference, initial-state subtraction, fake perturbation, or smoothing is
applied.

Color limits are shared across N8/N12 for each physical field:

```text
mom magnitude: [0, max(N8,N12)]
signed p_perp: [-max|.|, +max|.|] over both grids
signed B_phi:  [-max|.|, +max|.|] over both grids
```

## Return

Return a small dictionary containing:

- figure-ready raw arrays and shared physical limits;
- exact source paths and SHA-256 values;
- byte-copy source/destination pairs for gallery provenance;
- explicit scope, units, true-wall/staircase roles, and no-evolution flags.
