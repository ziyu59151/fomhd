# Massless diagnostics implementation pseudocode

An absent `electron_inertia` is treated as `0`; an explicit value must resolve
to the selected model before massless diagnostic applicability is reported.

## A0-PY-MASSLESS-CHECK-01: Applicability and totals

Stage and interval records use the runtime's existing
`measure`, `scalar_check`, `gate`, and record dictionaries. Applicability is
explicit metadata:

```text
model=massless_quasineutral_cgl_b
electron_inertia=not_applicable
electron_kinetic_energy=not_applicable
electron_kinetic_transport=not_applicable
electron_mechanical_stress=not_applicable
finite_charge_electric_stress=not_applicable
spectral_kinetic_transfer=not_applicable
longitudinal_electric_energy=not_applicable
finite_charge_gauss=not_applicable
```

Unavailable concepts produce records with `available=False`, `hard=False`,
and a reason. They never produce a passed finite-mass check.

`state_totals` integrates:

```text
number=[integral(n_i),integral(n_e)]
kinetic=[integral(m_i n_i |u_i|^2/2),0]
ion_internal=integral(p_i/(gamma_i-1))
electron_perp=integral(p_e_perp)
electron_parallel=integral(p_e_parallel/2)
magnetic=integral(|B|^2/(2 mu_0))
longitudinal=0
energy=ion kinetic+ion internal+electron pressures+magnetic
```

The returned record keeps every runtime totals key and adds applicability
metadata. No electron kinetic or `epsilon0 E_L^2/2` term is added.

## A0-PY-MASSLESS-CHECK-02: Stage checks

```text
check_stage(state,result,context):
  ledger[0]=finite ion PowerLedger
  ledger[1]=massless electron work:
    electric_node=0, not_applicable
    electric_edge=q_e<F_e,actual E>_M1
    pressure_kinetic=<u_e,g_e>_M0
    pressure_internal=integral(PK pdot_perp+PK pdot_parallel/2)
    pressure_internal_full=integral(pdot_perp+pdot_parallel/2)
    pressure_truncation=retained-full
    perp_rate/parallel_rate=actual retained pressure RHS
    magnetic=<u_e,q_e n_e u_e cross B>_M0
    wall=q_e<F_e,E-E0>_M1
    wall_absolute_scale=ang_integral(abs(q_e F_e (E-E0)))_M1
    spectral_kinetic=0, not_applicable
    node_edge_reconstruction_difference is spatial, not energy transfer
  boundary=actual wall number/energy traces
  totals=state_totals
  momentum=Cartesian ion mechanical momentum and actual wall traction
  hard-check:
    finite/Hermitian RHS
    quasineutrality rho=0
    electron n compatibility and zero compatibility momentum RHS
    J=Cdag B/mu0
    Fe=(q_i Fi-J)/e
    T_full Fi=0, T_full Fe=0, T_full J=0
    all original wall-correction rows
    DB rhs.B=0 and DJ J=0
    CT consistency
    full padded pressure adjoint and retained-pressure truncation
      pressure_adjoint_scale =
        integral(
          abs(pdot_perp) + 0.5*abs(pdot_parallel)
          + sum_components abs(conj(u)*pressure_force)
          + sum_components abs(P*grad(u)))
      retain the physical residual, configured physical budget, hard status,
      and 4096*eps scale factor; all operands are pre-cancellation backend arrays
      use pressure_adjoint_scale only for electron_pressure_adjoint and the
      duplicate pressure_pairing residual
    pressure tensor moment and CGL source identities
    corrected-field wall reaction work checked separately for Fi, Fe, and J
      each reference = ang_integral(abs(q F Er)) in watts
    Ohm residual E-E0-E_reaction=0
    actual-field pairing retained pressure internal rate-q_e<Fe,E>=0
  non-hard, convergence-only:
    theta_consistency=full padded divergence(Fe/nf)
    theta_defect=theta-theta_consistency
  unavailable:
    finite-charge Gauss, longitudinal displacement current,
    electron inertia work/energy
```

The pressure-adjoint scale includes the pressure-rate operands that can cancel
before the retained pressure work residual is formed. The signed near-zero
result is not used as its own scale, and no background-energy floor or
host-side fallback is introduced. The configured physical threshold remains
separate and unchanged.

The pre-existing `pressure_scale=integral(abs(P:grad(u)))` remains the
reference for `electron_electric_pairing`, `electric_pairing`,
`electron_pressure_only_pairing`, and electron pressure truncation. The repair
does not broaden or change any of those unrelated references or configured
thresholds.

Theta consistency applies the full-sample cylindrical divergence to
`Fe/nf`; its Fourier projection is stored separately. It is not replaced by
`DJ*Fe/ne` and is not gated as an algebraic zero. Hard cancellation checks use
the sum of absolute pressure-gradient or charged-flux/reaction terms as their
dimensional scale; the near-zero signed result is not used as its own
tolerance scale.

Boundary traces extract the wall-normal scalar interpolation row
`r=I[edge][direction]`. Every velocity component uses
`momentum_component @ r / (mass * density_trace)`; for ions the stored
material momentum is divided by `m_i`, while electrons use direct padded
`ne*ue`. Electron velocity never consumes retained compatibility momentum.
Electron kinetic transport is absent.
Ion internal transport uses `1/(gamma_i-1)` and electron parallel pressure
uses the `1/2` factor. The wall-normal traction is

```text
T n = [m_i n_i u_i u_i + p_i I + P_e
       + |B|^2 I/(2 mu0) - B B/mu0] n
n = sigma e_direction
P_e = p_perp I + (p_parallel-p_perp) bhat bhat
bhat = B/|B|
```

The signed normal multiplies every stress term. Electron anisotropic stress
uses `bhat` even though magnetic stress uses physical `B`. There is no
electron ram or electric Maxwell stress. Linear and angular momentum are
rotated to Cartesian before integration.

For wall-reaction work, `Er=M1^-1 S^H lambda`, so `lambda` already carries
the edge mass matrix factor. The reference therefore must not multiply lambda
by `M1` again. Retain sample-space work as observational transform evidence:

```text
sample_work = (2 pi/N) Re sum_samples,edges conj(q F_s) Er_s M1_e
transform_difference = sample_work - modal_work
```

Neither sample-space work nor its difference from modal work sets the hard
acceptance threshold. The configured dimensional budget belongs to the
retained-mode check below.

The authoritative hard-check scalar uses the retained Parseval form, avoiding
an unverified FFT operation count:

```text
gamma(p) = p eps/(1-p eps)
xhat_i=qi Fi_hat
xhat_e=qe Fe_hat
xhat_J=J_hat
a_i,k=(|qi|/mi) |I| |mom_i,k|
a_J,k=|Cdag_k| |B_k|/mu0
a_e,k=a_i,k+a_J,k
d_i,direct_k=gamma(8*3*Nc+4) a_i,k
d_J,direct_k=gamma(8*Nb+2) a_J,k
d_e,direct_k=(1+gamma(10))(d_i,direct_k+d_J,direct_k)
             +gamma(10)a_e,k

L_pi,k=2048 eps max(maxabs(mom_i,k),
                   mi ni_ref uref, 1e-300)
require maxabs(T_full mom_i,k)<=L_pi,k
L_J=4096 eps Nb max_k(maxabs(Cdag_k) maxabs(B_k)/mu0)
require maxabs(S Jhat_k)<=L_J for all original rows

b_i,k=abs(S).T (|qi|/mi L_pi,k
       + gamma(8*3*Nc)(|qi|/mi) abs(T_full)abs(mom_i,k))
b_J,k=abs(S).T L_J
b_e,k=(1+gamma(10))(b_i,k+b_J,k)
d_i,k=d_i,direct_k+b_i,k
d_J,k=d_J,direct_k+b_J,k
d_e,k=d_e,direct_k+b_e,k
w_s=2 pi Re sum_k,e M1_e conj(xhat_s,k,e) Erhat_k,e
A_mode,s=2 pi sum_k,e M1_e |xhat_s,k,e||Erhat_k,e|
B_up,s=2 pi sum_k,e M1_e |Erhat_k,e| d_s,direct,k,e
B_compat,s=2 pi sum_k,e M1_e |Erhat_k,e| b_s,k,e
B_sum,s=gamma(10*M*Ne+8) A_mode,s
B_work,s=(1+gamma(2))(B_up,s+B_compat,s+B_sum,s)
threshold_s=max(configured_atol+configured_rtol*configured_reference,
                B_work,s)
```

All producer envelopes have units A/m^2 before pairing and watts after
pairing. `compatibility_bound` is the `B_compat` watt contribution from the
already-checked input contracts; `arithmetic_bound` is the full allowance and
includes direct producer error, checked-input uncertainty, and retained
bilinear-sum rounding. The electron mass is not used.

For aggregate species work, use

```text
B_species=(1+gamma(2))(B_ion+B_electron)
          +gamma(2)(abs(w_ion)+abs(w_electron))
```

so the aggregate is never below the sum of nonnegative component bounds.
After each device weighted reduction, and again after species aggregation,
validate `A_mode`, `B_up`, `B_compat`, `B_sum`, `B_work`, and every stored
bound as finite and nonnegative. Validate both complex components before
real conversion, and repeat validation for `scale`, `upstream_bound`,
`compatibility_bound`, and `arithmetic_bound` when constructing each producer
and aggregate evidence record. A nonfinite or negative bound raises a
`WallWork` failure before it can become a threshold. Reject nonfinite or
negative gamma counts. Retain sample-space work under a separate key and
expose `sample_work-modal_work` as observational transform evidence; that
difference is never added to work acceptance.

## A0-PY-MASSLESS-CHECK-03: Interval transaction

```text
check_step(left,base,midpoint_state,midpoint,right,end,...):
  use the supplied explicit quadrature weights
  verify:
    species continuity for ion and canonical electron density
    electron density derivative from ion continuity
    zero charge change
    DJ J=0
    full-wall J and species flux constraints
    B change+integral(C E)=0
    DB(Bright-Bleft)=0
    ion kinetic/internal balances
    electron perp/parallel pressure balances
    total magnetic+material energy balance without electron KE or E_L
    actual interval delta-P plus quadrature wall traction in kg*m/s
    interval total number/charge, ion kinetic/internal, electron
      perpendicular/parallel, magnetic and total-energy balances
    interval wall current and both species wall fluxes
  populate runtime-compatible interval number_flux,current,quadrature,
    cumulative ledger keys, totals, and applicability
```

The result is accepted only by the existing time transaction after all hard
gates pass. Diagnostics do not mutate the trial state, cumulative ledger, or
committed state. Applicability metadata is attached to stage totals, ledgers,
boundary records, interval records, and the cumulative ledger.
