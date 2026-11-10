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

```text
VelocityGradient(fluid,x,context):
  require x and density/face-density extents
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
# Polygon-dual fluid actions

## Scope and dispatch

This pseudocode covers only the `polygon_dual_v1` branches of
`src/fomhd/fluid.py`. The legacy staircase branch remains the default and its
actions are unchanged. Dispatch uses `geometry.kind`, with
`op.geometry_kind` as the fallback and `staircase_v1` as the final default.

All retained state is mode-major complex128. The retained sample backend is
CPU or CuPy; scalar reductions are backend operations and no whole-stage host
copy is introduced. Geometry tables, measures, interpolation, and operators
are supplied by the polygon mesh implementation.

`op.face_normals_backend` and `op.cell_radius_backend` are borrowed immutable
backend float64 arrays. Fluid actions consume them directly and never upload
the corresponding host geometry during a stage. Reading
`geometry.face_normals` or `geometry.cells.R` is only the bounded fallback for
synthetic tests whose operator lacks those persistent arrays. The parent
memory planner owns the persistent-array accounting.

The polygon layouts are:

```text
E = [normal-face samples Nf, toroidal cell samples Nc]
B = [toroidal vertex samples Nv, tangential-face samples Nf]
```

The scalar transfer is `scalar_I = [S; identity(Nc)]`, where `S` maps cell
scalars to normal-face centroids. The vector transfer `I` maps three physical
cell components to `Ne = Nf + Nc` edge components and is used only for
physical vectors. In particular, repeated scalar values use `scalar_I`, never
`I` repeated three times. `scalar_I` is a real-contract operator matrix but is
consumed with its existing dtype; fluid does not cast it to float64.

## Scalar flux and divergence

For a retained scalar vector `s`, the sampled face scalar is

```text
nf = scalar_I s
F_hat = I (pi / m)
F = inverse(F_hat)
div_hat = DJ F_hat
div = Dface F_n + Dphi(F_phi) / R_c
```

`Dface` acts from the `Nf` normal-face rows to cells. `Dphi` is the complete
periodic toroidal derivative and `R_c = M0 / area` is the positive cell
radius. The existing retained/sampled divergence identity check applies
without modification.

## Velocity gradient

Let `n` be sampled density, `nf` the scalar transfer of density, `u_q` a
physical velocity component, and

```text
W_n = diag(n),  W_f = diag(1 / nf),
F_R = diag(n_R / nf),  F_Z = diag(n_Z / nf).
```

Only the first `Nf` rows of `scalar_I` and the face normals contribute to
normal derivatives. For component `q`, the three coordinate columns are

```text
(G u)_q,R = Dface F_R S W_n u_q - u_q / R_c
(G u)_q,phi = Dphi(u_q) / R_c
(G u)_q,Z = Dface F_Z S W_n u_q
```

After those three actions, the original cylindrical connections are retained:

```text
G_R,phi -= u_phi / R_c
G_phi,phi += u_R / R_c
```

The pressure tensor and force pairing continue to use the existing `I`, `H`,
and `M0` weighted adjoint conventions.

## Exact weighted adjoint

The adjoint is derived from the same actions rather than from an independently
guessed force. For tensor `P`, define `P_qc` and the unweighted transpose by

```text
<P, Gx>_M0 = <G^T (M0 P), x>
```

The stored weighted adjoint is `(G^T (M0 P)) / M0`. For row/component `q`:

```text
normal_q =
    W_n S_R^T (n_R / nf) Dface^T (M0 P_q,R)
  + W_n S_R^T (n_Z / nf) Dface^T (M0 P_q,Z)

toroidal_q = Dphi^T (M0 P_q,phi / R_c)

y_q = (normal_q - M0 P_q,R / R_c - toroidal_q) / M0
```

`S_R` is the first `Nf` rows of `S`; toroidal rows are inserted as zero
normal-flux rows. The transpose of the radial `-u_q / R_c` term is the
`-M0 P_q,R / R_c` term above.

The cylindrical connections have the further adjoint contributions

```text
q = R:   +P_phi,phi / R_c
q = phi: -P_R,phi / R_c
q = Z:    0
```

The periodic spectral derivative has the existing antisymmetric discrete
transpose behavior, so its reverse action is applied with the same backend
derivative and the sign used by the staircase path. All products are dense
matrix actions or backend elementwise operations already present in the
fluid stage.

## Verification

The focused tests use a two-triangle non-axis finite-volume fixture with a
shared slanted face, an explicitly specified scalar transfer, face areas,
normals, cell measures, and boundary incidence. They independently check:

1. scalar transfer uses `[S; I]`;
2. normal-face plus toroidal divergence;
3. constant and linear cylindrical gradient actions under variable density;
4. the weighted gradient adjoint for complex, multi-mode toroidal data; and
5. the unchanged staircase dispatch.

This fixture verifies algebraic operator contracts only. Its synthetic
`Dface`, projection rows, and normals do not establish physical polynomial
consistency on the real polygon mesh.

A full vacuum RHS is not available until the independent polygon mesh backend
is present. That integration is reported separately and is not replaced by a
second fluid model or an RHS offset. Constant/linear `G u` recovery and the
constant-pressure wall projection on the actual mesh remain parent integration
gates.
