# Mixed coarse-ion/fine-field interval ledger

Status: bounded CPU implementation pseudocode for the selected A1-WR-050
`F=A+C`, density-`D`-in-fine-`C` partition. This record does not claim full
advance, restart, GPU, or scientific acceptance.

Design authority:
`mainline_design/numerics/algorithm_design/A1/A1_WR_050.md` and the integration
appendix of the independent WR discrete review.

## A1-PY-WR-LEDGER-01 - Frozen interval records

```text
LedgerIntegral:
  exactly the 12 cumulative ledger keys
  each value is one finite host float in integrated SI work [J]

BoundaryIntegral:
  number=(ion,electron) [1]
  kinetic=(ion,electron) [J]
  internal_perp=(ion,electron) [J]
  internal_parallel=(ion,electron) [J]
  field_energy [J]
  total_energy [J]

SourceKey:
  immutable tuple(source_id,step,generation,trial,spectral_context,
                  time.hex(),stage,micro_step,iteration,slab)

MixedInterval:
  trial,node_version,left_step,right_step,left_time,right_time
  ledger=(ion,electron)
  boundary
  outward [kg*m/s]
  outward_torque [kg*m^2/s]
  projection_reaction [kg*m/s]
  projection_torque [kg*m^2/s]
  density_work_coarse [J]
  density_work_fine [J]
  kinetic_path [J]
  coarse_sources=(SourceKey,weight)
  fine_sources=(SourceKey,weight)

MixedIntervalBundle:
  interval
  fine_views
  diagnostic_work

MixedDiagnosticWork:
  electron_ledger_calls
  electron_pressure_only_projections
  electron_ledger_inverse_transforms
  wall_work_metric_calls
  wall_work_inverse_transforms
  wall_work_mode_envelope_sets
  electron_work_gate_calls
  electron_gate_transforms
  work_gate_records

No array, accepted flag, restart field or cumulative-schema field is stored in
the interval. Freeze dataclasses preserve metadata shape but do not freeze the
referenced fine-view arrays.
```

## A1-PY-WR-LEDGER-02 - Source and quadrature validation

```text
accumulate_mixed_interval(window,left_state,endpoint,context):
  require exactly one accepted OLD version, not mapped endpoint
  require s in {3,5}; H=window.H
  require window.H exactly equals declared context.p.dt
  require s window.coarse_samples and 4*M window.fine_samples
  require coarse/fine order and distinct source keys
  require every sample identity.iteration=window.final_input_version
  validate left and right macro identities with core.time_at_step(p,n),
    independently for actual left, interval left, right and endpoint clocks
  allow only 16 eps times max(|scheduled|,|start|,|dt|) clock identity
    roundoff; never infer H by subtracting adjacent absolute clocks

  Qc_l=H*window.Q[-1,l]
  Qf_jr=h*beta_r for the declared RK4 endpoint weights beta=(1,2,2,1)/6
  require each independent sum(abs(weight))=H

  for coarse l:
    require sample record.tag equals frozen sample identity/tag
    require record.full.source_id nonzero
    require recorded packed RHS equals sample.A+C
    require sample.P is immutable accepted P_l
    retain full ion producer frame and A projection from CoarseRecord

  for fine j,r:
    require sample.record.tag/source/frame tag and input P/Z all match
    require frame.ion_em_terms is present and belongs to frame.tag
    require actual feedback.n/mom equal retained density/EM producer
    never call ion transport or construct a full fine StageResult
    charge each accumulator _electron_ledger call, pressure-only projection
      and inverse transform in MixedIntervalBundle.diagnostic_work
```

## A1-PY-WR-LEDGER-03 - Exact ledger assembly

```text
At coarse nodes:
  f_A=-ion.convection+ion.pressure_force
  r_A=actual A projection value supplied with the same coarse call
  w_A=actual A projection reaction
  A_m=FFT(r_A), A_p=FFT(ion.pdot_perp)
  D_s=-ion.divergence

At fine nodes:
  f_E=frame.ion_em_terms.electric
  f_B=frame.ion_em_terms.magnetic
  r_C=frame.ion_em_terms.projected
  w_C=frame.ion_em_terms.reaction
  C_m=FFT(r_C), D_hat=-frame.ion.divergence_hat

Ion:
  electric_node=Qf[<u,f_E>]
  electric_edge=Qf[q_i<Flux,E>_edges]
  pressure_kinetic=Qc[<u,pressure_force>]
  pressure_internal=Qc[-integral(p_i theta_i)]
  magnetic=Qf[<u,f_B>]
  wall=Qc[<u,w_A>]+Qf[<u,w_C>]
  density_work_coarse=Qc[integral(sD_s)]
  density_work_fine=Qf[integral(sD_hat)]
  delta_D=density_work_coarse-density_work_fine
  convection=Qc[legacy ion ledger.convection]-delta_D
  perp_rate=pressure_internal
  parallel_rate=0
  spectral_kinetic=Qc[S_A]+Qf[S_C]
  spectral_perp=Qc[S_p]
  spectral_parallel=0

  S_A=<u,T(A_m)-r_A>
  S_C=<u,T(C_m)-r_C>-integral(s(T(D_hat)-D_s))
  S_p=integral((T(A_p)-ion.pdot_perp)/(gamma_i-1))

  kinetic_path=Qc[<u,T(A_m)>]+Qf[<u,T(C_m)>-integral(sT(D_hat))]

Electron:
  e=massless_checks._electron_ledger(actual fine frame)
  every ledger key is the corresponding Qf[e_key]
  preserve retained-full perpendicular/parallel pressure RHS meanings
```

## A1-PY-WR-LEDGER-04 - Boundary, traction, and fine view

```text
Reuse _wall_trace_data and factor its nonnegative scalar summands.

Finite/wall stages:
  Boundary.number=(Qf[b_N,i],Qf[b_N,e])
  Boundary.kinetic=(Qc[b_K,i],0)
  Boundary.internal_perp=(Qc[b_U,i],Qf[b_perp,e])
  Boundary.internal_parallel=(0,Qf[b_para,e])
  Boundary.field_energy=Qf[b_field]
  Boundary.total_energy=Qc[b_E,i]+Qf[b_E,e+b_field]

  t_A=m_i n_i u_i(u_i.n)+p_i n
  t_C=P_e n+(|B|^2 n/2-B(B.n))/mu0
  outward=Qc[wall_integral(rotate(t_A))]
          +Qf[wall_integral(rotate(t_C))]
  torque uses actual R_wall, with samples=max(N,3)
  projection_reaction=Qc[cell_integral(rotate(w_A))]
                       +Qf[cell_integral(rotate(w_C))]
  _momentum_integrals(...,traction_role):
    transport source selects ONLY t_A and the supplied coarse reaction w_A
    electromagnetic source selects ONLY t_C and the supplied fine reaction w_C
    never add t_A+t_C at either role and never subtract reaction from traction

FineStageView exposes only:
  state, fluid=(ion,electron), cache, current,
  rhs.B, tag, source_id, checks, kind/complete_stage
No fake ion ledger, full ion RHS, boundary, momentum, rates, or transport.
```

## A1-PY-WR-LEDGER-05 - Checker and pending consumption

```text
massless_checks.check_step(...,mixed_interval=None):
  None => original midpoint/quadrature path byte-for-byte unchanged
  mixed => validate interval against left/right/context
    keep dt=context.p.dt for weights, denominators and physical rates
    validate endpoint steps/times through direct scheduler clocks
    read number flux/current/divergence/wall flux, B power, CT and
      quasineutrality from actual fine views
    replace only 12-key ledger, six boundary fields, outward traction/torque
    read the same ledger electric edges in field exchange
    reuse original raw kinetic/internal/energy gates and tolerances
    do not subtract delta_D or kinetic_path from a physical gate
    run original electron work, pressure-adjoint, and wall-work gates
      through the gate helper also used by full check_stage
    return actual checker-only MixedDiagnosticWork counts and attach them to
      accepted or rejected interval evidence
    report delta_D, kinetic path, and metadata consistency separately

time._pending_cumulative(...,mixed_interval=None):
  None => original loop unchanged
  mixed => deep-copy cumulative, add each frozen integrated field exactly once
    no H multiplication, fine-stage loop, or reconstructed mixed values
    preserve duplicate-trial, finite and restart-key checks
```

## A1-PY-WR-LEDGER-06 - Verification

```text
Focused tests:
  exact 12/6 field sets and frozen metadata
  each independent quadrature spans H
  s5 nonzero density quadrature is moved into fine continuity
  K_path self-identity including density work
  one-term omission/double count changes evidence
  wrong trial/node/source/version/tag and NaN/inf weight reject
  mixed check_step with real periodic and closed-wall K=1 frames
  no full ion transport call in fine frames
  None checker and pending behavior unchanged
  independent nonzero t_A/t_C vector and torque ownership
  real nonuniform K=1 role split differs from duplicated full traction

CPU only; no GPU, Slurm, install, Git operation, or source promotion.
```
