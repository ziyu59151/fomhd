# Massless quasineutral electron implementation pseudocode

The branch accepts an absent `electron_inertia` as the default `0` and rejects
explicit `1` rather than mixing finite and massless ownership.

## A0-PY-MASSLESS-01: Model and ownership contract

This branch represents collisionless ions, two independent electron CGL-B
pressures, and `B`. The physical electron mass in `Species.mass` remains
positive metadata and a compatibility-storage factor. It is never used as an
electron inertia term or set to zero.

For positive `e=-q_e` and positive `q_i`, the exact constraints are

```text
n_e = q_i n_i/e
J = C^H B/mu_0
F_e = (q_i F_i-J)/e
u_e = D_n^-1 Q H F_e
Q = 1-W T^H (T W T^H)^-1 T
W = D_n M_N^-1
```

`canonicalize_state` returns a new `State` with the input step, generation,
time, ion arrays, electron pressures, and `B` preserved by value. It derives
electron density and the compatibility field
`mom_e=m_e FFT(n_e u_e)`. These two electron fields are outputs only: no
stage, pressure, Ohm, diagnostic, or time code inverse-transforms `mom_e` to
obtain `u_e`.

All retained arrays have layout `[mode][entity]`, modes are `-K..K`, samples
have `N` physical azimuthal values, and vectors use R/phi/Z components.
Derived samples use the complete padded `N`-sample transform and only the
final material RHS is truncated to `-K..K`.

## A0-PY-MASSLESS-02: Canonical state and shared electron production

```text
derive_electron(state, context, *, ion_transport=true):
  require ion_transport is exactly boolean
  validate q_i>0, q_e<0, e=-q_e, m_e>0, state shapes and finite values
  BN=inverse(QB*B); bhat=BN/|BN|; require |BN|>B_min
  ion=BuildFluid(state.fluid[0],0,BN,bhat,context)
  if ion_transport:
    BuildNonElectric(ion,...)
  # false skips only this ion transport call; derive electron fields below
  J=Cdag*B/mu_0
  Fe=(q_i*ion.flux_hat-J)/e
  ne_hat=(q_i/e)*ion density spectrum
  ne=inverse(ne_hat)
  nf_component=Interpolate(ne)
  Fe_sample=inverse(Fe)
  for each azimuthal sample:
    W=diag(tile(ne,3))*diag(tile(M0,3))^-1
    A=T W T^H
    require A full rank on the deterministically selected T rows
    qhf=H*Fe
    qhf=Q*qhf
    ue=reshape(qhf,[N,3,Nc])/reshape(ne,[N,1,Nc])
    refine Q solve boundedly on the same factors
    require T_full*(ne*ue)=0 at roundoff scale
  ue_hat=FFT(ne*ue)
  return ion,J,Fe,ne,nf,ue,ue_hat,updated compatibility momentum
```

The keyword is keyword-only and defaults to true. `canonicalize_state` keeps its
existing default call and field input/output contract. A field-only caller can
derive once with `ion_transport=false`, then pass its compatibility outputs to
`_copy_state`; it must not perform a second derivation.

The retained component layout is R-block, phi-block, Z-block, but the
sample-space division is performed after reshaping to `[N,3,Nc]`. A flat
density sequence must never be tiled across samples. The wall response matrix
uses all selected wall rows. Its rank is checked with the configured rank
tolerance and a checked dense factorization. The full `T_full` constraint on
`ne*ue` is checked separately. Compatibility `mom_e` is derived-validation
storage, so the shared retained-state checker is applied to the independent
state with electron momentum replaced by zero; the compatibility field is
checked by its definition and Hermitian spectrum.

### Exact diagonal wall-normal specialization

```text
Q(nodal,density):
  validate positive current-stage density and finite nodal values
  T=current operator-selected wall rows
  winv=tile(current_density/M0,3)
  if every nonzero column of T occurs in at most one row:
    diagonal=sum(winv*abs(T)^2) over columns, samples batched
    largest=max(diagonal); smallest=min(diagonal)
    solve, bounded refinement, residuals, and full T_full check as below
  else:
    build each dense scaled normal and use checked SVD/LU fallback
```

For the diagonal branch, positivity makes the diagonal equal the exact normal
eigenvalues in ascending and descending order, so the configured relative
rank criterion is unchanged. The structural test uses the actual selected
`T`; it is not enabled from geometry alone. The fast path does not import
`FieldBatch`, cache a mutable operator, or retain a previous stage's density.
Dense coupled rows retain the original per-sample checked solve and full-wall
residual behavior.

## A0-PY-MASSLESS-03: Electron pressure and field

```text
electron_pressure(ue,Fe,ne,nf,Pperp,Pparallel,bhat):
  Ge=VelocityGradient(ne,nf,ue)
  theta=tr(Ge); strain=bhat bhat:Ge
  Pe=perp*I+(parallel-perp)*bhat bhat
  ge=Ge^H Pe
  pressure_flux=(I perp)/(I ne)*Fe
  parallel_flux=(I parallel)/(I ne)*Fe
  perp_dot=-Div(pressure_flux)
             +perp*(strain-theta)
  parallel_dot=-Div(parallel_flux)-2*parallel*strain
  theta_consistency=full_divergence(Fe/nf)
  theta_consistency_hat=FFT(theta_consistency)
  theta_defect=theta-theta_consistency
```

The pressure force is evaluated before interpolation to current edges. The
unconstrained Ohm field is

```text
node_source=ge/e-D_n*(u_e cross B_N)
E0=I*(Q*node_source reshaped[N,3,Nc]/ne reshaped[N,1,Nc])
```

`Q` acts on both the pressure and magnetic source terms.

## A0-PY-MASSLESS-04: Closed-wall electric-field correction

Before the first massless wall solve, request the context-private fixed wall
system cache. Cache construction and all per-stage lookups are inside the
stage failure boundary so a new setup error retains model evidence.

For each retained Fourier mode, with fixed operators

```text
W=M1^-1 S^H
T=C W
Qw=Cdag T
Aw=S Qw
```

The implementation forms only the thin products `C@W` (`Nb x walls`),
`Cdag@T` (`Ne x walls`), and `S@Qw` (`walls x walls`). It never materializes
`Cdag@C` (`Ne x Ne`) merely to form `Aw`.

```text
prepare-once, reuse:
  S = signed wall-normal edge extraction, all original wall rows
  W = M1^-1 S^H
  Aw = S*Cdag*C*W
  check Aw is Hermitian at the retained roundoff scale
  rows=IndependentRows(Aw,rank_rtol)  # deterministic increasing indices
  gauge_rows=all rows not selected
  if rows:
    prepared=PrepareLinear(Aw[rows,rows],rhs-independent physical scales)
  else:
    prepared=none

per nonzero electric-field spectrum:
  rhs=-S*Cdag*C*E0[mode]
  if rows:
    lambda_selected=SolvePrepared(prepared,rhs[rows])
  else:
    lambda=0
  lambda[gauge_rows]=0           # explicit principal-row gauge
  require Aw*lambda-rhs=0 on every original row
  E[mode]=E0[mode]+W*lambda
```

The cache stores `S`, `Aw`, selected/gauge rows, the principal scaled system,
its singular spectrum and LU, metadata, and exact persistent-array byte
accounting. Repeated right-hand sides reuse these fixed objects and perform
only thin right-hand-side products, prepared solves, full-row checks, and the
final `W` product.

The cache key is scoped to fixed operators, not content-scanned every stage:

```text
(operator object identity, geometry object identity, backend object identity,
 backend name/device, parameter-set/reference identities and reference
 length/electric values, rank_rtol, operator K/Ne/Nb/Nc,
 precomputed operator/rule/geometry hashes when present)
```

Physical sample count `N` and `spectral_id` are absent because the wall factors
operate only on retained Fourier-mode blocks. A shallow-copied alias context
therefore safely shares the same fixed factors, including counters on the
sequential host path. Operator matrices must be immutable after construction.
Any operator/geometry/backend/reference/rank/parameter-set change or
stale/replaced identity hash invalidates the cache. `_constraint_project`
remains density-dependent and is not memoized.

The former per-mode formulation remains the algebraic oracle:

```text
Aw_dense=S*Cdag*C*M1^-1*S^H
rhs_dense=-S*Cdag*C*E0[mode]
```

The solve reports rank, selected rows, omitted gauge coordinates, scaled
backward error, and full-row residual. The correction is algebraic; it is not
an implicit electron advance. `Bdot=-C E` uses the complete corrected field.

## A0-PY-MASSLESS-05: Stage assembly

```text
evaluate_stage(state,tag,context,alias=false):
  require tag identity equals input state
  canonical=canonicalize_state(state,context)
  CheckRetainedMassless(canonical)
  derive electron fields and reject incompatible B
  build ion non-electric terms
  build electron pressure terms from direct Fe and ue
  compute E0 and constrained E
  FinishFluid(result,ion,context)
  rhs.fluid[1].n=(q_i/e)*rhs.fluid[0].n
  rhs.fluid[1].mom=0                      # compatibility storage only
  rhs.fluid[1].perp=FFT(perp_dot)
  rhs.fluid[1].parallel=FFT(parallel_dot)
  rhs.B=-C*E
  cache.E=E; cache.ET=E; cache.E0=E0; cache.wall_reaction=E-E0
  cache.phi=None; cache.EL=None
  rho=0 exactly; current=J; ELdot=phidot=None
  populate massless checks, totals, boundary, momentum, ledgers
  return fully initialized StageResult(canonical_state=canonical)
```

No finite-mass Poisson, Gauss, Darwin, or Ampere evaluator is called. Source
payloads that request independent electron density/momentum evolution fail
as unsupported rather than being silently discarded.

## A0-PY-MASSLESS-06: Initialization, combining, and failure evidence

`initialize` evaluates the canonical initial state and hard-checks:

```text
DB*B=0
S*J=0
n_e=q_i*n_i/e
J=Cdag*B
Fe=(q_i Fi-J)/e
T_full*u_e=0
positive density and pressure domains
all algebraic branch checks
optional static RHS budgets
```

An incompatible initial current is rejected; it is never projected. The
committed state is the canonical state and the cumulative baseline uses the
same record structure as the parent stage.

Compatibility tolerances are computed from the `DB*B` and
`Cdag*B/mu0` operator-product term scales and a bounded floating-point
operation count. The observed current is not used as the reference for its
own wall residual. A separate negative fixture verifies incompatible-current
rejection.

`combine` advances the ion variables, electron pressures, and `B`, leaves the
electron compatibility density/momentum outside the independent DOF set, and
canonicalizes the resulting midpoint/endpoint. `retain_failure_evidence`
retains prior checks, residuals, fields, and arrays without calling the public
stage dispatch function recursively, then adds missing massless-specific
unavailable records.

### Runtime integration adaptation

The routing integration imports `stage._retain_common_failure_evidence` and
calls that non-dispatching helper before adding missing massless-specific
records. It must not import or call public `stage.retain_failure_evidence`.

Before canonicalizing a combined candidate, `combine` retains the raw candidate
under the `candidate` evidence prefix when canonicalization raises a `Failure`.
The preserved evidence includes the candidate stage identity, B, both densities,
momenta, and pressures. The original exception is re-raised without changing
the accepted state or ledger.

### A1-PY-MASSLESS-ALIAS-01 - Alias identity adaptation

```text
alias_probe(state,tag,context,result):
  copy context with alternate padded sample count and spectral id
  history_samples = history_sample_count(context)
  copy._history_samples = history_samples
  alternate_tag.step,generation,trial,time = tag values
  alternate_tag.stage = "alias"
  alternate_tag.micro_step = tag.micro_step
  alternate_tag.iteration = tag.iteration
  alternate_tag.slab = tag.slab
  evaluate and append alias diagnostics with the preserved identity
```

The alternate spectral context remains distinct, but the macrostep, microstep,
iteration, and slab ownership of the stage being probed must not be reset to
synchronous defaults.

The alternate `N=2N+1` changes only the one-stage sample workspace. The retained
`4M` sweep remains owned by the outer context, so the alias copy inherits the
validated outer history sample identity. A nested alias inherits the original
identity rather than multiplying or resetting it. The fixed wall factorization
cache key remains unchanged. This accounting correction does not alter any
physical state, alias diagnostic, cap, or numerical operation.


## A050 polygon wall dispatch

The polygon layout uses actual normal extraction and the shared weighted thin-SVD velocity projection. Magnetic alias groups are [B_phi(vertex), B_t(face)] with layout labels; the existing diagnostic budget components are selected explicitly. Physical Ohm, QN/current reconstruction and one-producer stage contracts are unchanged. See [polygon_wall](polygon_wall.py.md) and [polygon_projection](polygon_projection.py.md).
