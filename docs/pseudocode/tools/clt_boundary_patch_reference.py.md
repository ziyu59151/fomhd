> Scope: independent local CPU reference only; not a selectable solver boundary.

# `tools/clt_boundary_patch_reference.py` implementation pseudocode

## A050-LOCAL-BP-01 Geometry and units

1. Define immutable `TriangleGeometry(r0, a, b)` in SI lengths. Require all
   three values to be finite and positive.
2. Use `O=(r0,0)`, `A=(r0+a,0)`, and `B=(r0,b)`. The triangle is
   `0<=x/a`, `0<=z/b`, and `x/a+z/b<=1`, with `x=R-r0`, `z=Z`.
3. Set the right-handed loop to `O->A->B->O`. On `A->B`, the unit tangent is
   `t=(-a,b)/L`, the outward unit normal is `n=(b,a)/L`, and `L=hypot(a,b)`.
4. Define area `ab/2`, cylindrical volume per radian
   `r0*ab/2+a^2*b/6`, signed wall distance `(b*x+a*z-a*b)/L`, and the wall
   tangent coordinate as projection from `A` onto `t`.
5. Keep all physical fields as caller callables of physical `(R,Z)`. No
   helper imports the polygon backend or the source solver.

## A050-LOCAL-BP-02 Physical edge sampling and electric trace

1. Obtain `m` Gauss-Legendre nodes and weights on `[0,1]`.
2. Sample `E_R(R,Z), E_Z(R,Z)` on:
   `OA: (r0+a*u,0)`,
   `OB: (r0,b*u)`,
   `AB: (r0+a*(1-u),b*u)`.
3. Set `eR=integral_OA E_R ds/a` and
   `eZ=integral_OB E_Z ds/b`. These remain axis-component averages; do not
   replace either by a wall-normal component.
4. Set `lambda=integral_AB E.t ds`, namely
   `integral_0^1 (-a*E_R+b*E_Z) du`. It has units of field times length.
5. Set circulation `Gamma=a*eR-b*eZ+lambda` and cylindrical curl
   `omega_phi=-Gamma/(ab/2)`. The minus sign is physical for the selected
   `(R,Z)` orientation.
6. Define `wR=(1-z/b,x/b)`, `wZ=(z/a,1-x/a)`, and
   `wL=(-z/(ab),x/(ab))`. Reconstruct
   `E=eR*wR+eZ*wZ+lambda*wL`.
7. Keep nonzero `lambda` as an explicit boundary/auxiliary value. A zero
   electric moment does not prescribe the current moment to zero.
8. Admit nonzero manufactured tangent traces for algebra checks. Keep the
   physical ideal-conductor condition `E_t=0` as a separate production rule;
   the manufactured tests do not redefine it.

## A050-LOCAL-BP-03 Potential, metric, and paired power

1. `sample_potential` reads callable potential values at `O,A,B`.
   `p1_gradient` reconstructs the affine gradient from those three values.
   For the wall-zero homogeneous P1 subset with `phi_A=phi_B=0`, it gives
   `(-phi_O/a,-phi_O/b)`. Do not claim higher-polynomial reproduction.
2. Form the exact homogeneous 2-by-2 cylindrical Gram from the selected
   formulas for `integral (r0+x) w_i.w_j dx dz`.
3. Form the 3-by-3 lifted Gram with the same collapsed-square Gauss map:
   `x=a*s`, `z=b*(1-s)*t`,
   `dA=a*b*(1-s) ds dt`, weight `(r0+x)*dA`.
4. Mark metric arrays read-only. Expose symmetric `apply`, `solve`, `pair`,
   and `adjoint`; these act only on the tiny local matrix.
5. Lump only the homogeneous metric with
   `D_ii=(M*v)_i/v_i`, `v=(b,a)`. This preserves the normal axis trace but
   is not assumed to preserve smooth energy or arbitrary action.
6. Sample both physical electric and current callables. Include both wall line
   moments in `x^T M y`; separately report direct-sampled power and the
   omission experiment. Do not silently drop either wall lift.

## A050-LOCAL-BP-04 Magnetic divergence by physical flux

1. On each edge use its actual outward normal and physical length:
   `OA: n=(0,-1)`, `OB: n=(-1,0)`, `AB: n=t_rot=(b,a)/L`.
2. Integrate `B.n*R ds` independently on all three edges.
3. Divide the signed sum by `integral R dA`. Preserve the wall normal flux.
4. For constant `B_Z`, the nonzero wall trace cancels the nonzero `OA` trace.
   For constant `B_R`, expect
   `average divergence=3*B_R/(3*r0+a)`.

## A050-LOCAL-BP-05 Affine pressure and gradient identity

1. Sample callable pressure at the shared vertices `O,A,B`, then reconstruct
   the unique affine pressure. Its gradient is constant.
2. Define the physical pressure force on the fluid as the negative gradient
   integral, `F_fluid=-integral grad(p) (r0+x) dx dz`. Constant pressure has
   zero force.
3. For a supplied velocity and divergence callable, compute
   `gradient_pairing=integral grad(p).u dV`,
   `pressure_divergence=integral p div(u) dV`, and
   `boundary=sum_edges integral p (u.n) R ds`.
4. Retain this positive-gradient relation separately as
   `pressure_gradient_identity`:
   `gradient_pairing+pressure_divergence=boundary`. Do not reverse its sign or
   add a duplicate API to represent the physical fluid force.
5. Define `+integral p n R ds` as the force of fluid on the wall. The wall on
   the fluid is its negative. Return wall and support reactions separately.
6. This is a local gradient/weak-identity oracle only. It neither supplies nor claims
   acceptance of the source solver's pressure RHS.

## A050-LOCAL-BP-06 Reciprocal-R tangent-current control

1. Use `E=E0*n`, with `E0` in V/m and `n=(b,a)/L`; this is constant and normal
   to `AB`, so its wall moment is zero.
2. Use `J=J0*(a,-b)/R`, with `J0` in A/m so `J` is in A/m^2. This vector is
   tangent to `AB` and has zero cylindrical divergence.
3. Sample and reconstruct `J` through the formal lifted entries. Retain its
   nonzero current wall moment even though the electric wall moment is zero.
4. Compare reconstructed `E.J` and weighted field error with direct physical
   quadrature at no fewer than three refinement levels. Report observed order
   and omitted-current-moment bias; do not impose exactness on reciprocal `R`
   outside the two-axis reconstruction space.
5. This is an independent local prototype, not a global Ohm/species/current
   acceptance.

## A050-LOCAL-BP-07 Scope failures and verification

1. Reject invalid geometry, quadrature orders, coefficient shapes, and field
   outputs with explicit errors.
2. Test analytic constants, a nonzero-curl field, homogeneous and curved
   zero-wall potentials, exact and lifted metrics, wall-moment omission,
   magnetic flux, affine pressure, constant electric power, and the
   reciprocal-`R` tangent-current control.
3. Use independent quadrature in tests and probe; use the stored triangle
   receipt only as provenance.
4. Report conditioning, refinement, operations, measured CPU cost, and all
   missing global contracts. Passing this local unit does not integrate
   species/current/Ohm closure, global assembly, or RK4.
5. Do not claim an implemented FOMHD rigid-rotation gradient/strain check. That
   check is assigned to another worker.
