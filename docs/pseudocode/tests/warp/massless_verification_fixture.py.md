# `massless_verification_fixture.py` pseudocode

## A0-PY-MASSLESS-VERIFY-01 Purpose and boundary

Build test-only periodic Cartesian data for the actual `fomhd` runtime:

- physical units are SI;
- `context.N = Nphi = 4*K+1`, with `K=0` for the parallel Hall fixture;
- the grid stores `Ncells` independent z samples while the cylindrical radius is
  `R=1e30`, suppressing cylindrical curvature to roundoff;
- no wall rows exist, so the native closed-wall fixture is represented by a
  boundary-free periodic domain;
- solver formulas are imported from the source snapshot and are never
  reimplemented in this fixture.

The independent oracle uses the cold transverse Hall/ion-inertia equations, not
the production stage or its diagnostic helpers.

## A0-PY-MASSLESS-VERIFY-02 Periodic operators

1. Create entity tables with keys `(j, 0)` and coordinates
   `(R=1e30, z=2*pi*j/Ncells)`.
2. Build the periodic Fourier derivative matrix `S` from the native
   `periodic_fixture.hpp` coefficients, scaled by
   `k0*Ncells/(2*pi)`.
3. For parallel propagation set `DR=0`, `DZ=S`; include a zero azimuthal
   derivative for `K=0`.
4. Assemble gradient, curl, divergence, `DJ=-G^H`, and `Cdag=C^H` with the
   native component ordering `[R, phi, Z]`.
5. Set identity `I`, `H`, and `QB`; set empty wall traces and empty
   wall-projection matrices.
6. Verify the adapter against a plane wave and the exact identities `C G=0`,
   `DJ=-adjoint(G)`, and `DB C=0` before using it.

## A0-PY-MASSLESS-VERIFY-03 Uniform equilibrium and perturbations

1. Set ion and electron density `n0`, isotropic ion/electron pressure `p0`,
   `ui=0`, and `B=(0,0,B0)`.
2. Require a nonzero positive `B0`, positive ion/electron masses and charges,
   and `electron_inertia=0`.
3. Perturb one of `ui_R`, `ui_phi`, `B_R`, `B_phi` by an amplitude times
   `cos(2*pi*z/L)`.

## A0-PY-MASSLESS-VERIFY-04 Independent cold transverse matrix

For `e=|q_e|`, singly charged ions, and equal density:

- `A = k B0/sqrt(mu0 mi ne)`;
- `H = k^2 B0/(mu0 e ne)`;
- `omega_minus=(sqrt(H^2+4*A^2)-H)/2`;
- `omega_plus=(sqrt(H^2+4*A^2)+H)/2`.

Construct the four-equation complex coefficient matrix from:

- ion acceleration from `J x B0`;
- electron velocity `ue=ui-J/(e ne)`;
- `E=-ue x B0`;
- Faraday `dB/dt=-curl(E)`.

Return the matrix, positive branch frequencies, and the corresponding
eigenvectors. Reject a negative Hall-sign control through its incorrect matrix
signs, polarization and time response. Its unordered frequency set is unchanged.

## A0-PY-MASSLESS-VERIFY-05 Production linearization

1. Evaluate the real `fomhd.stage.evaluate_stage`.
2. For each perturbation direction and each of two amplitudes, evaluate the
   positive and negative perturbation about the same equilibrium.
3. Project the differing real sample fields onto the cosine mode.
4. Form the centered derivative and normalize rows by momentum scale for ion
   velocity and field scale for `B`.
5. Require the two-amplitude matrices to agree, and compare both matrices with
   the independent complex matrix using relative `L2` error outside a
   documented roundoff floor.
6. Record eigenvalues and hard production diagnostics. Hard diagnostics are
   never disabled to make this fixture pass.

## A0-PY-MASSLESS-VERIFY-06 Time evidence

1. Build the physical cosine/sine realization of the largest positive
   independent eigenvector.
2. Set `dt = cycles*(2*pi/omega)/steps`, `source=None`, and
   `context.p.dt_guard="warn"`.
3. The production operator-L1 guard estimate is intentionally more
   conservative than the independently matched retained eigenfrequency; record
   its warning on every bounded run while keeping all physical stage and
   interval gates fatal.
4. Compare the final modal vector both with the exact midpoint/RK4
   amplification polynomial and with the exact continuum exponential.
5. The polynomial comparison detects an incorrect transaction implementation.
   The continuum error is the quantity expected to decrease with step size.
6. Reduce retained `B` and ion momentum with the owning backend's `isfinite`
   and `all`. Transfer only the explicitly declared final-mode capture through
   `Backend.to_host`; never use implicit NumPy conversion on CuPy state.
7. Convert host-valued perturbations to the active `xp` before mutating device
   state.

## A0-PY-MASSLESS-VERIFY-07 Spatial evidence

1. Build the actual closed rectangle and `K=1, Nphi=5` source operators without
   replacing their RZ/Fourier matrices. Use cell-centered coordinates with fixed
   physical bounds `R=[1,2]`, `Z=[0,1]`, and a fixed `0.2 m` wall band at every
   mesh level.
2. Prescribe a positive non-axisymmetric density, smooth pressure bump with
   zero wall variation, toroidal ion velocity, and uniform wall-compatible
   axial magnetic field. Density and velocity modulation use the fixed polygon
   Z endpoints, not the grid's first and last cell centers.
3. Evaluate the production `velocity_gradient`, `gradient_adjoint`,
   `_constraint_project`, and `_massless_e0` paths.
4. Compare `theta=tr(G ue)` with explicit analytic derivatives. For isotropic
   pressure the continuum adjoint target is `-grad(p)`, not `+grad(p)`.
5. Compute pressure norms with cell weights `M0`; report `linf`, weighted
   relative L2, and entity count independently for all/interior/wall.
6. Convert retained Ohm fields to physical edge samples and measure against the
   independently assembled pressure-source target with edge weights `M1`.
   The target shares discrete I, Q and truncation and therefore does not prove
   independent continuum electric-field accuracy. Apply masks to
   the actual R-face, toroidal, and Z-face entities. Record empty regions as
   unavailable rather than copying a global error.
7. Prescribe a smooth compact-support edge flux and a nonzero current generated
   through the production `Cdag` from a smooth magnetic field. Form
   `Fe=(qi Fi-J)/e`, reconstruct with the actual `QHFe/ne`, and compare the
   resulting velocity, theta, and wall-normal flux with explicit continuum
   targets. This is a reconstruction test, not a nodal-velocity assertion.
8. Run at least three fixed-domain mesh levels and calculate orders from the
   weighted norms. Positive converged global/interior errors are evidence;
   nonconvergent wall norms remain reported limitations.
9. Evaluate the repaired production `theta_consistency` using the full padded
   divergence of `Fe/nf`. Compare it with explicit analytic derivatives.
10. Evaluate the actual `K=1` variable-density `QHFe/ne` velocity and compare
   with sample-wise component broadcasting. This exposes flattened-density
   reuse across different phi samples.
11. Require the variable-density velocity and padded-theta regressions to remain
    at roundoff; retain the pre-repair failures as separate historical evidence.
12. Delta theta and its field/L-infinity unit metadata use 1/s; relative norms
    are dimensionless. An unnormalized weighted L2 also carries sqrt(volume).

## A0-PY-MASSLESS-VERIFY-08 Diagnostic negative control

1. Use a real nonzero-field stage: uniform equilibrium plus a `2e-6 B0`
   transverse `Bx` perturbation.
2. Double its retained `E0`, recompute residuals, and evaluate the production
   hard drift-Ohm gate.
3. Preserve raw `linf` value, unit, hard threshold, and pass/fail result.
   Reject a roundoff-only control.

SI reference contract: current and charge references are already densities;
use Jref for wall/current definitions, Jref/L for current divergence, rhoref
for quasineutrality, Eref/L**2 for SCdagCE, and P*u*L**2 for advective
power. Interval charge is rhoref*L**3 and interval momentum is mi*ni*u*L**3.
Relative tolerances and hard/soft flags are independent of these units.
