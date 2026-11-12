# Rectangular wall comparison gallery

The historical FreeGS control and the paired wall repair share the same physical inputs. The A1 time methods also share one initial pulse and one field microstep. The plots reproduce the accepted comparisons for these cases.

## Analytic current and refinement

![Solov’ev exact, historical and repaired current](figures/solovev-current-sections.png)

The upper panels show the total current on a common scale. The lower panels show the error and a magnified repaired scale.

![Mixed vacuum refinement](figures/mixed-vacuum-current-convergence.png)

## A0 FreeGS matched 8 ns

Nonlinear accepted quiet and pulse states on the same physical box, with one fixed outboard cut and exact common saved times.

### 14×20 repaired mesh · total states and difference

The first four columns show the accepted fields in physical units. The right column subtracts the quiet endpoint from the pulse endpoint. The magnitude rows display the norm of the vector difference.

![Complete state overview](figures/total-state-01-overview.png)

Absolute panels share a physical color scale for each row. The last column uses its own scale. Orange is the traced LCFS (quiet LCFS in the difference). The solid black line is the physical PEC in this run. A dotted purple line marks the alternate wall.

![Ion velocity components](figures/total-state-01-velocity.png)

Absolute panels share a physical color scale for each row. The last column uses its own scale. Orange is the traced LCFS (quiet LCFS in the difference). The solid black line is the physical PEC in this run. A dotted purple line marks the alternate wall.

![Magnetic field components](figures/total-state-01-magnetic.png)

Absolute panels share a physical color scale for each row. The last column uses its own scale. Orange is the traced LCFS (quiet LCFS in the difference). The solid black line is the physical PEC in this run. A dotted purple line marks the alternate wall.

### 14×20 legacy mesh · total states and difference

The first four columns show the accepted fields in physical units. The right column subtracts the quiet endpoint from the pulse endpoint. The magnitude rows display the norm of the vector difference.

![Complete state overview](figures/total-state-02-overview.png)

Absolute panels share a physical color scale for each row. The last column uses its own scale. Orange is the traced LCFS (quiet LCFS in the difference). The solid black line is the physical PEC in this run. A dotted purple line marks the alternate wall.

![Ion velocity components](figures/total-state-02-velocity.png)

Absolute panels share a physical color scale for each row. The last column uses its own scale. Orange is the traced LCFS (quiet LCFS in the difference). The solid black line is the physical PEC in this run. A dotted purple line marks the alternate wall.

![Magnetic field components](figures/total-state-02-magnetic.png)

Absolute panels share a physical color scale for each row. The last column uses its own scale. Orange is the traced LCFS (quiet LCFS in the difference). The solid black line is the physical PEC in this run. A dotted purple line marks the alternate wall.

### Rectangular wall curl

Accepted quiet and pulse responses on the same mesh with the legacy and repaired operators.

![Initial positions versus the continuous source](figures/initial-position.png)

Signed position errors at the fixed physical cut. The source flux and initial density profile are evaluated continuously from the supplied external spline.

![Magnetic and density displacements](figures/matched-displacement.png)

Solid curves subtract the matched quiet run from the pulse displacement. Dashed curves show the quiet displacement from its own time zero. All grids use the exact common saved times.

![14×20 legacy: Quiet minus its own initial state](figures/sections-01-quiet-change.png)

Actual saved fields at 8 ns. Symmetric SI scales are shared across grids. The original LCFS and 0.2 m physical collar edge mark the static analysis regions.

![14×20 repaired: Quiet minus its own initial state](figures/sections-02-quiet-change.png)

Actual saved fields at 8 ns. Symmetric SI scales are shared across grids. The original LCFS and 0.2 m physical collar edge mark the static analysis regions.

![14×20 legacy: Pulse minus its own initial state](figures/sections-01-pulse-change.png)

Actual saved fields at 8 ns. Symmetric SI scales are shared across grids. The original LCFS and 0.2 m physical collar edge mark the static analysis regions.

![14×20 repaired: Pulse minus its own initial state](figures/sections-02-pulse-change.png)

Actual saved fields at 8 ns. Symmetric SI scales are shared across grids. The original LCFS and 0.2 m physical collar edge mark the static analysis regions.

![14×20 legacy: Pulse minus matched quiet](figures/sections-01-pulse-minus-quiet.png)

Actual saved fields at 8 ns. Symmetric SI scales are shared across grids. The original LCFS and 0.2 m physical collar edge mark the static analysis regions.

![14×20 repaired: Pulse minus matched quiet](figures/sections-02-pulse-minus-quiet.png)

Actual saved fields at 8 ns. Symmetric SI scales are shared across grids. The original LCFS and 0.2 m physical collar edge mark the static analysis regions.

### Toroidal current

Accepted quiet-state Ampère current relative to the common external Grad–Shafranov profile.

![Toroidal current error](figures/current-sections.png)

Legacy and repaired panels share one signed scale at each time. Orange marks the original LCFS.

## A1 matched pulse at 3.2 ns

Synchronous RK4 and three-node waveform relaxation use byte-identical pulse data and the same field microstep.

### Matched pulse

Absolute fields and signed WR minus synchronous differences at the common accepted endpoint.

![Physical RZ pulse sections](figures/pulse-sections.png)

Absolute panels use common scales. The difference panels have independent signed scales. Orange marks the original LCFS.
