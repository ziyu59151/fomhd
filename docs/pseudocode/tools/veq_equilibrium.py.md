# `tools/veq_equilibrium.py` implementation pseudocode

## VEQ-PREP-00 Scope and import boundary

This module is a NumPy-only preprocessing data contract. Importing it must not
import `veqpy`, CuPy, Warp, or establish any execution context. Only the CLI
entry imports the pinned external solver.

The input is one native VEQ snapshot plus declared source metadata. The output
is an immutable `VeqEquilibriumSnapshot` containing physical SI arrays on the
native `(r,theta)` evaluation grid, the native snapshot, and explicit
conversion metadata. No FOMHD field, RHS, Ampere solve, Ohm solve, or time state
is reconstructed by this module.

## VEQ-PREP-01 Source identity and units

Record:

- source commit `2b64c9a1bffddc25c176bd52231bc3b481715c6c`;
- tree `ac8635c629d100db8998f9904793dbf697373eb7`;
- archive SHA-256;
- VEQ `mu0=4e-7*pi H/m`;
- `Psi` in Wb, `psi=Psi/(2*pi)` in Wb/rad;
- native `P` and `p0` in Pa;
- `F=R*B_phi` in T m and `B0=F_edge/R0`;
- raw `P_psin=dP/d(psin)` in Pa;
- raw `FF_psin=F*dF/d(psin)` in `(T m)^2`;
- physical `P_psi` in Pa/(Wb/rad).

Do not label raw coordinate derivatives with physical-flux units. Keep raw
source arrays and final physical derivatives as distinct fields.

## VEQ-PREP-02 Flux orientation and native pressure

VEQ `psin` increases from axis to edge. For the FOMHD positive-interior family
use

```text
psi_edge = surface_mean(Psi_edge)/(2*pi)
psi_F    = psi_edge - Psi/(2*pi)
```

so `psi_F=0` on the wall and is positive inside. Verify the selected sign from
the exported `B` and current, not from naming alone.

Retain native `P_native` and native `p0_native`. If a target pressure reference
is selected, form the explicit additive offset

```text
p_offset  = p_edge_target - surface_mean(P_native_edge)
p_total   = P_native + p_offset
```

The offset changes no GS gradient or current. Never replace the native profile
or native `p0`; store both and record the offset.

For the selected constant-F family:

```text
F(psi_F)         = F_edge
Fprime(psi_F)    = 0
p(psi_F)         = p_edge_target + pprime_F*psi_F
pprime_F         = +1e4 Pa/(Wb/rad)
```

The physical VEQ mapping is `P_psi=-pprime_F`.

## VEQ-PREP-03 Derived physical fields

Given native `R,Z,theta,r`, a radial differentiation matrix, and periodic theta
samples, form `B` from `psi_F`:

```text
psi_R = derivative of psi_F with respect to R
psi_Z = derivative with respect to Z
B_R   = -psi_Z/R
B_Z   =  psi_R/R
B_phi = F/R
```

The `(r,theta)` map is singular at `r=0`: all axis theta samples describe the
same physical point. Exclude that duplicated row from valid collocation and
banded residual samples. The flux function is known to be stationary at its
axis, so set `grad(psi_F)=0` exactly there. For every other field derivative,
form the regular physical-gradient limit from the first valid radial rings by
extrapolating their surface-mean Cartesian gradient components to `r=0`; do not
copy a first-ring gradient and do not set arbitrary non-flux derivatives to
zero.

For the selected constant-F family construct the **profile-background**
candidate separately:

```text
J_profile_R = J_profile_Z = 0
J_profile_phi = R*pprime_F
```

Never label this expression as the actual current represented by reconstructed
`B`. Compute actual Ampere current from the reconstructed field:

```text
J_ampere_R = -1/mu0 * dB_phi/dZ
J_ampere_phi = 1/mu0 * (dB_R/dZ - dB_Z/dR)
J_ampere_Z = 1/mu0 * (1/R)*d(R*B_phi)/dR
```

and integrate `J_ampere_phi` over the physical RZ Jacobian for the actual
Ampere current. Retain the solver's native profile `jphi` separately from both.
Do not inject the profile current into reconstructed `B`.

Construct two electron-flow diagnostics:

```text
u_e_profile = -J_profile/(e*n)
u_e_ampere  = -J_ampere/(e*n)
```

For `n=2e19 m^-3`, ions at rest, `p_i=0.6p`, and isotropic electrons:

```text
u_i = 0
E_profile = grad(p_i)/(e*n)
```

Only the selected constant-F family is constructed. Do not infer a general
electron field-parallel flow or CGL pressure classification.

## VEQ-PREP-04 Residuals and held-out samples

Compute the independent strong GS operator directly from `psi_F` derivatives:

```text
G_strong = psi_RR - psi_R/R + psi_ZZ
           + mu0*R^2*pprime_F
           + F*Fprime
```

with `Fprime=0` for this family. Use a different evaluation grid from the
solver representation grid. Report maximum absolute and RMS residual over held
out interior and near-wall bands separately from the solver's projected raw
residual.

At `r=1`, compute actual `B.n` using the smooth Fourier tangent collocated at
each sampled θ:

```text
t=(dR/dtheta,dZ/dtheta)
n=(t_Z,-t_R)/|t|
```

Check that this is the outward orientation. Do not use the chord tangent from
vertex `i` to `i+1` as the normal at vertex `i`. A polygon-segment midpoint
diagnostic may be reported separately, but must not be mixed with the vertex
sample.

For one fixed solve, evaluate on at least three independent output grids and
repeat the strong-GS and wall diagnostics. This distinguishes differentiation/
resampling floor from solver-representation under-resolution. Do not add new
solver degree searches.

Report the singular axis theta row as one excluded physical sample, alongside
the number of valid radial rows, the raw theta spread of axis fields, and the
reconstructed axis `B_pol`. These diagnostics are separate from the
interior/near-wall residuals over valid collocation samples.

## VEQ-PREP-05 Immutable output

Freeze all arrays and records. Write one NPZ with native and converted arrays
and one JSON with source identity, units, signs, offsets, solver status,
projected residuals, independent strong residuals, wall flux, and refinement
metadata. The native snapshot remains embedded and recoverable.
