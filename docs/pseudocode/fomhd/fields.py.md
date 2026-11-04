# Darwin field implementation pseudocode

## A0-PY-FIELD-01: Electrostatic

```text
Electrostatic(rho,tag,context):
  require retained scalar shape, finite, and Hermitian
  allocate owned phi, EL, per-mode Poisson residuals, solve records
  for k=-K..K:
    rhs=rho[k]/epsilon0
    linear=backend.linear(Lphi[k],rhs,row_scale,col_scale,rank_rtol,tol)
    phi[k]=linear.x
    poisson_residual[k]=Lphi[k]*phi[k]-rhs
    EL[k]=-(G[k]*phi[k])
  require physical Hermitian phi and EL
  return FieldCache(tag,phi,EL,...)
```

Every `Nc` row is solved and checked. The Dirichlet potential trace does not
remove a body-charge row. A failed rank, finite, or Hermitian condition is
fatal.

## A0-PY-FIELD-02: LongitudinalProjection

```text
Longitudinal(current,tag,context):
  for k:
    rhs=DJ[k]*current[k]
    psi=linear(Lphi[k],rhs,...).x
    result[k]=-(G[k]*psi)
  return result
Transverse(current,tag,context)=current-Longitudinal(...)
```

The input current remains unchanged. The projection uses physical scales and
all retained modes.

## A0-PY-FIELD-03: TransverseDarwin

```text
Transverse(stage,context):
  require prepared fluid contexts and finite fJ0,EL
  AEAction(Ehat):
    result=0
    for species:
      force=ElectricResponse(fluid,Ehat) # complex preserved
      result+=Forward(I*(q/m)*force)
    return result
  AE=materialize(AEAction, M*Ne square)
  curl=block_diag(M1*Cdag[k]*C[k]/mu0)  # Cdag already contains MB
  A=curl+M1_repeated*AE
  require M1_repeated*AE is Hermitian
  top_rhs=-M1_repeated*(fJ0+AEAction(EL))
  KKT=[[A,Bc.H],[Bc,0]]
  solution=backend.linear(KKT,[top_rhs,0],scaled rows/columns,...)
  ET=solution[:M*Ne], lambda=solution[M*Ne:]
  require finite and Hermitian ET
  stage.AE=AE
  stage.cache.ET=ET
  stage.cache.E=EL+ET
  stage.cache.multiplier=lambda
  rhs_B=-C*ET
  residuals: KKT,dynamic,strong dynamic,full Bc divergence,DB*rhs_B
```

The selected `Bc` rows define the multiplier representation only.
`Bc_full*ET` is always checked, including rows not selected for elimination.
On a linear solve failure, the code attempts to report scaled singular values
and raises `Failure("DarwinRank", ...)`; it does not return a partial field.

## A0-PY-FIELD-04: Current consistency and magnetic RHS

`stage.py` verifies

```text
sum_species (q/m)*Forward(I*rhs.mom)
  == fJ0 + AE*cache.E
```

and `rhs.B=-C*cache.ET`. It also forms `ELdot=-Longitudinal(current)/epsilon0`,
`phidot=Lphi^-1(-DJ*current/epsilon0)`, and the initial Ampere residual
`C.H*B-mu0*Transverse(current)`. These diagnostics do not add independently
evolved field state.
