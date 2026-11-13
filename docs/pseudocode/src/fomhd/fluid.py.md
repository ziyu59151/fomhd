# Two-fluid implementation pseudocode

## A0-PY-FLUID-00 - Layout and ownership

Retained arrays are one-dimensional complex128: scalars
`[mode][cell]`, vectors `[mode][component][cell]`, and tensors
`[sample][row][column][cell]`. Samples replace modes. Components are
`(R,phi,Z)`. The returned `FluidContext` owns primitive samples, flux,
gradient, pressure tensors, wall factors, source terms, and ledgers. All
backend operations use the selected `context.compute`; no NumPy/CuPy host
fallback is permitted.

## A0-PY-FLUID-01: BuildFluid

```text
BuildFluid(state,species,BN,bhat,tag,context):
  require species in 0..1 and all state extents match the retained layout
  require finite(state) and finite(BN,bhat)
  require each sampled |BN| > B_min
  require each sampled sum(conj(bhat)*bhat) == 1 within 2e-10
  n=inverse(state.n,physical_real)
  pi=inverse(state.mom,physical_real)
  p_perp=inverse(state.perp,physical_real)
  p_parallel = species==1 ? inverse(state.parallel,physical_real) : empty
  require n>n_min, p_perp>p_min,
          and p_parallel>parallel_min for the electron
  u=pi/(mass*n)
  Fhat=apply I to state.mom/mass once
  F=inverse(Fhat,physical_real)
  nf=apply I separately to each copied scalar component of n
  require nf>face_min before every division
  divF=DivSamples(F)
  divFhat=DJ*Fhat
  require Forward(divF)==divFhat within 2048*eps*max(1,scale)
  P=isotropic(p_perp) if ion
    else p_perp*I + (p_parallel-p_perp)*bhat*bhat
  uu[a,b]=u[a]*u[b]
  raw_moment=mass*n*uu+P
  wall=BuildWallContext(n,species,tag,context)
  return owned {n,pi,u,F,Fhat,nf,divF,divFhat,P,raw_moment,wall,tag}
```

Any shape, finite, positivity, magnetic-domain, normalized-direction, or
division identity failure raises `Failure`; no clipping is applied.

## A0-PY-FLUID-02: VelocityGradient and adjoint

The rectangular ghost closure uses backend-owned `material_radial_derivative`
and `material_axial_derivative` from spatial setup. Both act on physical center
values in increasing coordinate order. The input remains `[sample][component]
[cell]` with R as the fastest cell index. A cylindrical point gradient needs
no extra `-x/R` term in its radial derivative. The two azimuthal connection
terms below still apply.

For each end face of either coordinate line, let `tau` be its two-center
affine interpolation and `tau_n(x) = tau(n*x)/tau(n)`. Add
`sign * area_per_radian/M0_near * (tau_n(x)-tau(x))` at the adjacent center.
Evaluate its coefficients as `3*(n_next-n_near)/(4*tau(n))` multiplying
`x_near-x_next`. Use current-stage densities. The signed area ratio is
`±R_wall/(R_near*h_R)` radially and `±1/h_Z` axially. Both sides at each
corner contribute to their own coordinate derivative. The transpose uses
the same scalar on each end, scattering to its near and next centers with
opposite signs after applying `M0` to the input. The caller still pairs the
electron gradient with the existing density-weighted wall pressure traction.
The shared face flux, conservative divergence and electric transfer are
assembled separately from this material gradient.

```text
VelocityGradient(fluid,x,context):
  require x and density/face-density extents
  for a rectangular ghost closure, apply both material derivatives and
    their current-stage trace lifts instead of the radial and axial
    reconstruction below; use the same Fourier and connection terms
  for component beta in R,phi,Z:
    nx=n*x[beta]
    V_R=reshape(sample_apply(I_R,nx)/nf_R, (N,3,Nc))
    V_Z=reshape(sample_apply(I_Z,nx)/nf_Z, (N,3,Nc))
    QR=D_R V_R-x[beta]/Rc
    Qphi=DphiFull(x[beta])/Rc
    QZ=D_Z V_Z
    assemble Q[beta,:]
  Q[R,phi]-=x[phi]/Rc
  Q[phi,phi]+=x[R]/Rc
  return owned Q

GradientAdjoint(fluid,P,context):
  for row a:
    W[a,b]=M0*P[a,b]
    for a rectangular ghost closure, use both material transposes and
      their current-stage trace-lift transposes instead of the radial
      and axial reconstruction transposes below
    radialNodes=n*I_R.H(D_R.H(W[a,R])/nf_R)
    axialNodes=n*I_Z.H(D_Z.H(W[a,Z])/nf_Z)
    azDerivative=DphiFull(W[a,phi]/Rc)
    v[a]=radialNodes+axialNodes-azDerivative
    plus the R-weighted cylindrical connection terms from GC12
    v[a]/=M0
  return owned v
```

`DphiFull` is the actual full-`N` derivative matrix; its adjoint is the same
backend primitive applied in the reverse inner-product chain. No formal
reciprocal derivative is substituted for an operator adjoint.

## A0-PY-FLUID-03: BuildNonElectric

```text
BuildNonElectric(fluid,species,BN,bhat,context):
  Q=VelocityGradient(fluid,fluid.u)
  theta=R,R+phi,phi+Z,Z
  strain=sum_ab bhat[a]*Q[a,b]*bhat[b]
  adv=GradientAdjoint(fluid,u tensor pi/mass)
  gradF[a]=sum_b Q[a,b]*(pi[b]/mass)
  A=.5*m*(-adv+gradF+u*divF)
  pressureForce=GradientAdjoint(fluid,P)
  magneticForce=q*n*cross(u,BN)
    on CUDA: unpack every _warp_array result into (view,storage_owner)
              launch FP64 views on current external stream and synchronize;
              all four storage owners remain alive through the synchronization
    on explicit CPU backend: evaluate the same formula with xp
  r0=-A+pressureForce+magneticForce
  pressureFlux(p)=I(p)/nf*F
  ion:
    pdot=-Div(pressureFlux(p_perp))-(gamma-1)*p_perp*theta
  electron:
    pdot_perp=-Div(flux(p_perp))+p_perp*(strain-theta)
    pdot_parallel=-Div(flux(p_parallel))-2*p_parallel*strain
  store all intermediate sampled terms with species/tag identity
```

The ion momentum contains no second electron-pressure force. The magnetic
force is `q*n*cross(u,BN)` with the native component order.
The CUDA path synchronizes its explicit Warp stream before returning the owned
pointwise force array; it never transfers the force to a CPU fallback.

## A0-PY-FLUID-04: ElectricResponse and FinishFluid

```text
ElectricResponse(fluid,species,Ehat,context):
  require Ehat retained edge shape and finite
  E=inverse(Ehat,physical_real=false)
  force=q*n*sample_apply(H,E)
  return ProjectWall(fluid.wall,force).value

FinishFluid(stage,species,context):
  E=inverse(stage.cache.E,physical_real)
  electricForce=q*n*H(E)
  total=fluid.non_electric+electricForce
  projection=ProjectWall(fluid.wall,total)
  rhs.n=-fluid.divFhat
  rhs.mom=Forward(projection.value)
  rhs.perp=Forward(fluid.pdot_perp)
  electron rhs.parallel=Forward(fluid.pdot_parallel)
  stage.terms[species]=electricForce,projection,reaction,lambda,residual
  stage.residuals[wall_fluid_species]=sample_apply(T_full,projection.value)
```

The final projection contains all forces and is the sole momentum RHS
producer. Physical-real conversion is done only for the finished field and
RHS, after the complex electric-response action has been formed.

## A0-PY-FLUID-05: PowerLedger

All physical-rate integrations use `2*pi/N * M0` on nodes and
`2*pi/N * M1` on edges. The implementation records electric node/edge power,
pressure kinetic/internal power, magnetic and wall power, convection pairing,
CGL internal rates, and retained-versus-sampled kinetic and pressure spectral
transfer defects. A physical ledger value with an imaginary residue above the
roundoff bound raises `Failure("Hermitian", ...)`.

The kinetic-density term
`m/2 * |u|^2 * ndot` is subtracted once, from the R component slot only before
the three-component integral. The same once-only rule applies to the sampled
and convective density terms; broadcasting it across all components triples
the recorded work in compressible flow.
