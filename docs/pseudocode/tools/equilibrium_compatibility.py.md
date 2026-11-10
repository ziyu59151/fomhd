# `tools/equilibrium_compatibility.py` implementation pseudocode

## EQ-AUD-01 Callback and units

1. Define an independent continuous field callback
   `field(R,Z) -> mapping[str, ndarray]` on physical SI coordinates.
2. Require at least `psi, B_R, B_phi, B_Z, E_R, E_phi, E_Z, n, u_i_R,
   u_i_phi, u_i_Z, p_i, p_e_perp, p_e_parallel, pprime, Fprime`. Optional
   advertised `J_R, J_phi, J_Z` are comparison values only.
3. Use the project's default SI species constants in an immutable record.
   The callback owns all normalized-equation transformations.
4. Reject nonfinite values or invalid density/pressure before evaluating
   residuals. No solver stage, source RHS, or scripted zero is used as input.

## EQ-AUD-02 Independent sampled derivatives

1. Evaluate the callback at `(R,Z)` and central offsets in R and Z.
2. Obtain first derivatives by central differences for B, E, psi, number
   density, pressures, and velocities. Obtain second psi derivatives required
   by Grad-Shafranov (GS) with central second differences.
3. Keep the complete offset and derivative expression in the report so a
   derivative order is observed, not assumed.
4. Derive physical cylindrical curl components:
   `curl_R=-(dB_phi/dZ)`,
   `curl_phi=dB_R/dZ-dB_Z/dR`,
   `curl_Z=B_phi/R+dB_phi/dR`.
5. Derive `div(B)=dB_R/dR+B_R/R+dB_Z/dZ`.

## EQ-AUD-03 Actual current and electron flow

1. Set `J_actual=curl(B)/mu0`; never substitute the advertised
   `R*pprime` as the current.
2. Set `n_e=abs(q_i)*n/abs(q_e)`.
3. Derive electron flow from the actual Ampere current:
   `u_e=(q_i*n*u_i-J_actual)/(e*n_e)`.
   When supplied current derivatives are used,
   `d u_e/dx=d u_i/dx-dJ/dx/(q_i*n)+q_i*J*dn/dx/(q_i*n)^2`.
4. If the callback supplies advertised current, report
   `J_advertised-J_actual` separately in A/m^2.
5. Build the local orthonormal cylindrical vector gradient for axisymmetric
   fields, retaining the `-u_phi/R` and `+u_R/R` connection terms.
   The axisymmetric `Z,phi` connection is zero: uniform axial translation has
   zero strain.
6. Set `theta=trace(grad u_e)` and
   `s=b b:grad(u_e)`, `b=B/|B|`.

## EQ-AUD-04 Continuous equation residuals

1. Ampere residual is `J_actual-curl(B)/mu0`, retained as a construction
   check; advertised-current discrepancy is the physical diagnostic.
2. GS residual:
   `psi_RR-psi_R/R+psi_ZZ+mu0*R^2*pprime+F*Fprime`.
3. Ion number continuity:
   `(1/R)d(R*n*u_i_R)/dR+d(n*u_i_Z)/dZ`.
4. Electron number continuity uses derived `n_e,u_e`.
5. Ion force:
   `q_i*n*(E+u_i x B)-grad(p_i)`.
6. Electron force:
   `q_e*n_e*(E+u_e x B)-grad(p_e)` using isotropic
   `p_e=p_e_perp=p_e_parallel` for the compatibility audit.
   Reject locally anisotropic electron pressure before this force/Ohm path.
   The bounded auditor may evaluate CGL anisotropy only with the isotropic
   force/Ohm result explicitly marked unavailable; it must not average
   perpendicular and parallel tensor gradients silently.
7. Static ion pressure residual:
   `u_i.grad(p_i)+gamma_i*p_i*theta_i`.
8. Static electron CGL residuals:
   `u_e.grad(p_perp)-p_perp*(s-2*theta)` and
   `u_e.grad(p_parallel)+p_parallel*(theta+2*s)`.
9. Ohm residual:
   `E+u_e x B+grad(p_e)/(e*n_e)`.
10. Static Faraday residual is `curl(E)`.
11. Report vector/Rayleigh norms in raw SI units and against explicit local
    equation scales. Do not infer global equilibrium from one aggregate.

## EQ-AUD-05 Wall trace

1. Sample the callback on the specified circle and use its actual normal.
2. Report `psi`, `B.n`, `n_i u_i.n`, `n_e u_e.n`, and `J_actual.n`.
3. Report the poloidal electric tangent `E.t`.
4. Retain raw values and separate field/flux scales. The initial total normal
   magnetic flux is an input, not something forced to a canned zero.

## EQ-AUD-06 Refinement

1. Evaluate the same held-out points for at least three derivative spacings.
2. Compare selected current, strain/CGL, and force residuals with an
   independent analytic derivative reference where one exists.
3. Report each error and observed order. A zero must come from evaluated
   equations and actual current, not from a hard-coded diagnostic.
4. Include a varying-`F`/field-parallel-flow control whose CGL parallel and
   perpendicular responses are nonzero. Mark it as a dynamics control, not a
   solved GS equilibrium.

## EQ-AUD-07 Scope

The module is reusable continuous-physics compatibility tooling. Parent task
owns classification. It does not replace the source stage, Ohm solve, time
method, boundary algorithm, or global model acceptance.
