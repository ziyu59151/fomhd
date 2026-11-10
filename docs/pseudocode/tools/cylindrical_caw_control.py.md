# Linearized cylindrical CAW reference pseudocode

## Scope

This is an independent analytic reference for the linearized massless,
quasineutral Hall/CGL-B problem in a straight circular cylinder:

- uniform guide field `B0`, density `n0`, ion pressure `pi0`, and isotropic
  electron pressure `pe0`;
- no parallel variation (`k_parallel=0`);
- axisymmetric `n=0`;
- perfectly conducting, impermeable wall at `r=a`.

The expressions are solutions of the linearized equations, not a full
finite-amplitude nonlinear system. This is not a finite-`R0` toroidal
eigenmode solver and does not analytically cover `n=1`. The module imports no
FOMHD geometry, state, or RHS code.

## Constants and validation

Defaults:

```text
a=0.5 m, B0=2 T, n0=2e19 m^-3,
mi=2*1.67262192595e-27 kg, pi0=600 Pa, pe0=400 Pa,
gamma=5/3, mu0=1.25663706127e-6 H/m,
e=1.602176634e-19 C, A=1e-3
```

The documented production defaults are copied as numeric literals:
`mu0=1.25663706127e-6 H/m` and `mi=2*MP` with
`MP=1.67262192595e-27 kg`. `DEFAULTS` is a mapping whose values are valid SI
inputs. The dataclass `CylinderCAW` stores the same fields and accepts
explicit caller values. Require positive finite `a`, `B0`, `n0`, `mi`,
`gamma`, `mu0`, and `e`; nonnegative finite `pi0`, `pe0`; and
`0 <= A <= 1e-3`. Reject nonfinite time and coordinates.

## Linearized analytic solution

```text
j11 = first positive zero of J1
kappa = j11/a
cfast = sqrt((B0^2/mu0 + gamma*pi0 + 2*pe0)/(mi*n0))
omega = kappa*cfast
period = 2*pi/omega
vA = B0/sqrt(mu0*mi*n0)
C = cos(omega*t)
S = sin(omega*t)
J1r = J1(kappa*r)
J0r = J0(kappa*r)
q0 = -kappa*A*vA/omega * J0r * S

u_r = A*vA*J1r*C
theta = kappa*A*vA*J0r*C
delta_n = n0*q0
delta_pi = gamma*pi0*q0
delta_pe_perp = 2*pe0*q0
delta_pe_parallel = pe0*q0
delta_B_guide = B0*q0
```

At `r=0`, `J1(kappa*r)/r -> kappa/2`; sample returns this finite ratio.
Velocity direction is radial from local center `(R0,0)`. At the axis, where
the radial direction is undefined but velocity is zero, use `(1,0)`
deterministically. `u_Z`, `u_phi`, and poloidal flux perturbation are zero.

`sample(R,Z,time,R0=3)` broadcasts R/Z, computes local
`r=sqrt((R-R0)^2+Z^2)` and `theta=atan2(Z,R-R0)`, rejects points outside the
circular cylinder beyond roundoff, and returns SI physical fields.

`seed_velocity(R,Z,R0,a,amplitude,...)` constructs a reference from the
explicit SI parameters and returns an `(N,3)` array ordered `(u_R,u_phi,u_Z)`.
At `t=0`, only the radial component is nonzero.

## Independent analytic checks

Using analytic time derivatives, check at sampled points and times:

```text
dn_dt + n0*theta = 0
dpi_dt + gamma*pi0*theta = 0
dpe_perp_dt + 2*pe0*theta = 0
dpe_parallel_dt + pe0*theta = 0
dBguide_dt + B0*theta = 0
```

Also report wall values:

```text
J1(kappa*a) = 0
u_r(a,t) = 0
delta_B_guide(a,t) = constant along the wall
delta_p(a,t) = constant along the wall
J_n = 0
E_t = 0
```

The analytic z derivative vanishes, so `b b:grad(u)=0`; massless
quasineutral continuity gives `div(u_e)=div(u_i)=theta`. These are reference
identities, not assertions about the production discrete RHS.

## Projections and frequency fit

Given physical ion velocity samples `u` with shape `(N,3)` or a physical
velocity vector, area weights `M0`, coordinates, and local center `R0`:

```text
u_radial = ((R-R0)*u_R + Z*u_Z)/r
basis = J1(kappa*r)
profile_coefficient = sum(M0*u_radial*basis)
                    / sum(M0*basis^2)
fractional_amplitude = profile_coefficient/vA
```

At the axis use the analytic finite `u_radial` limit through the radial
direction convention. For scalar density, pressure, or guide-B perturbations,
project against `J0(kappa*r)` with the same `M0` weighting:

```text
scalar_coefficient = sum(M0*value*J0)
                   / sum(M0*J0^2)
```

`fit_angular_frequency(times, values, omega_guess)` uses only the supplied
actual time samples. For each candidate frequency it solves
`value ~ c*cos(omega*t)+d*sin(omega*t)+offset` by linear least squares and
minimizes the residual over a bounded frequency interval around
`omega_guess`. Return the fitted frequency, coefficients, sample count, and
RMSE. No extrapolated samples are created.

## Declared large-aspect comparison

`large_aspect_comparison` evaluates the linearized straight-cylinder
reference at fixed `a` for `R0/a = 6,15,30,60`. It reports:

- straight-cylinder frequency and period;
- `a/R0` as the declared curvature/guide-field variation scale;
- `omega/Omega_ci` as the Hall frequency ratio;
- `kappa*rho_i` as the radial FLR ordering parameter.
- the caller-supplied `mi`, `mu0`, and `gamma` used in that comparison.

These are comparison scales, not a full toroidal frequency prediction. No
global fourth-order method order is assumed or claimed, and the finite
amplitude nonlinear problem is not represented.

## Tests

1. Exact `J1/J0` phases and overtone-free periodicity at multiple times.
2. Analytic continuity and CGL pressure source identities.
3. Wall root, boundary flux, normal current, and tangential E conditions.
4. Axis regularity of `J1/r`, physical radial conversion, and seed ordering.
5. `M0`-weighted `J1`/`J0` projections on off-center samples.
6. Frequency recovery from actual sampled times only.
7. Large-aspect sequence reporting and scope statements.
8. Invalid amplitude, nonfinite time, and outside-cylinder rejection.
