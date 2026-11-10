# `tools/plots/render_clt_physical_rz.py` pseudocode

Render two ordinary PNG/SVG figures through the existing A1 Matplotlib helpers.

## Figure 6: repair3 rigid-rotation strain

Rows: N8, N12, N16. Columns:

```text
old staircase rigid-strain magnitude
corrected positive-M0 candidate rigid-strain magnitude
positive-M0 magnitude minus old-staircase magnitude
```

Use one shared magnitude color limit for the old/candidate columns across all grids
and one symmetric difference limit. Draw the true R3 circle from the fixture-derived
centre/radius recorded by the evidence loader. Draw the old computational wall only
from supplied axis-aligned active-cell edge segments at `centre ± h/2`; do not trace a
finite-mask contour. Label all quantities as velocity-gradient symmetric strain in
`s^-1`, with the physical gate **failed**, not accepted.

## Figure 7: self-circle full initial formal RHS

Rows: N8, N12. Columns:

```text
|rhs(mom_i)| in kg/(m^2*s^2)
rhs(p_perp_e) in Pa/s
rhs(B_phi) in T/s
```

Apply the shared limits from the evidence module across rows. Draw true LCFS and the
old staircase computational wall from exact active-cell edge segments. Label `t=0`,
`time_steps=0`, raw SI, no field differencing or normalization, and “initial formal
RHS, not an evolved state.”
