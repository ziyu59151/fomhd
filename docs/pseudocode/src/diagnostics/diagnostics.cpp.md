# 只读运行诊断

> 状态：active；目标 `src/diagnostics/diagnostics.cpp`。依据 08 AC1/AC5、09 DG-A0/DG2、GC8–16。  
> 依赖：[类型](../../include/fomhd/types.hpp.md)、[fields](../fields/darwin.cpp.md)、[wall](../boundary/wall.cpp.md)。同步 time/io/tests/离线分析；最近同步：2026-11-03。

## A0-DIAG-01 — RecordAndGate

```text
Measure(name,residual,location,representation,region,budget,tag):
  require unique record key; budget has matching unit and norm
  if required source absent:
    return {status=UNAVAILABLE,raw_value=null,reason=missing_source}
  if any nonfinite: return FAIL with first/worst index and nonfinite count
  select active entities in region; never include dense padding/ghost
  if region empty: return UNAVAILABLE(reason="empty region")
  if a local periodic verification adapter has no wall entities and both wall
     matrices have zero rows, an empty wall_* residual has all six regional/norm
     records UNAVAILABLE, non-hard and not passed, reason="no boundary entities"
  otherwise an empty or mismatched residual/weight shape is an error
  if weighted_L2:
    raw=SpectralNorm or SampleNorm(residual,corresponding M)
    reference=norm(same-space constant reference_field,same M/region)
  if linf: raw=max(abs(residual)); reference=budget.reference
  if value: raw=scalar; reference=budget.reference
  threshold=atol+rtol*reference
  status=PASS if abs(raw)<=threshold else FAIL
  return record(raw,reference,threshold,worst entity/mode/sample,unit,...)
Gate(records,required_keys):
  reject duplicate keys or a missing required record (hard or observe)
  reject unless every required hard record is PASS
  retain observe FAIL/UNAVAILABLE with reason; they do not reject evolution
```

线性解使用 dense 中 `max(norm(rhs),norm(reference))` 的专门判据。L2 单位包含测度，例如电场点值 V/m，其三维积分范数为 V m^0.5；JSON 同时保存 quantity_unit/value_unit，不能用无量纲“relative”隐匿分母。极小值类检查直接比较实际值与严格下限，记录位置与单位，不把极小值当残差套本公式。

## A0-DIAG-02 — CheckInitial / CheckStage

```text
CheckInitial(U,stage,ops,budgets,kind):
  include stage.checks once (already contains CheckStage and alias records)
  include geometry topology/support/weight/distance results
  include separate CG,DBC,adjoint,GC8,I/H,wall rank and Gauss row count checks
  initial_divB=DB*U.B; initial_topology=ell*U.B[k=0]
  require initial Gauss,divB,transverse Ampere,wall and loop checks
  if kind==equilibrium: measure every evolutionary RHS against equilibrium budgets
  baseline=own(step=U.step,time=U.time,start_time,initial_divB,initial_topology,
    energy=StateTotals(U,stage.cache,stage.samples),momentum=stage.momentum_frame,
    geometry_hash,operator_hash,region definitions)
  return records,baseline

CheckStage(U,S,ops,budgets):
  CheckRetained and sampled positive-domain extrema (before unsafe divisions)
  check exact tag/source ownership and source registry, no duplicate electron
  record original Poisson/KKT equations, full DJ*ET, field boundary traces
  RG=DJ*EL-rho/epsilon0; RB=DB*B
  RA=Cdag*B-mu0*TransverseProject(J)
  for species:
    R_stage_charge=q*S.RHS[a].n+DJ*S.sources[a].current
    record full T*pi,full T*pidot,actual wall F and pressure fluxes
    record original S*lambda-Tr,projected-r+Winv*T'*lambda
    record electric edge-node mismatch,wall/magnetic work and pressure pairing
    record theta-DivSamples(V*u),convection pairing and local CGL internal source
  include S.current_response.records once and retain its actual residual map
  record finite/Hermitian RHS and DB*S.RHS.B
  return records,owning selected raw_residuals; all regions use the same physical masks
```

`CheckStage` 检查已有实际结果；诊断需要的投影使用同一静态 Poisson 算子和明确标记的诊断求解，不能用另一套场替换 S.cache。求解次数与 cost 单列。B 的 v 空间权用 `wv`；其 region 由 v 的实际坐标判定。

## A0-DIAG-03 — StateTotals

```text
StateTotals(U,cache,samples):
  for species a at each sample/cell:
    K_a=|pi_a|^2/(2*m_a*n_a)
    internal_i=p_i/(gamma_i-1)
    internal_e_perp=p_perp_e; internal_e_parallel=p_parallel_e/2
  each material energy = (2*pi/N)*sum_l,c M0[c]*density[l,c]
  E_B=(1/(2*mu0))*SpectralNorm(B,MB)^2
  E_L=(epsilon0/2)*SpectralNorm(EL,M1)^2
  E_total=sum(material components)+E_B+E_L     # 无 ET^2
  number_a=2*pi*sum_c M0[c]*real(nhat_a[k=0,c])
  mass_a=m_a*number_a; charge_a=q_a*number_a
  topology=ell*Bhat[k=0]; physical_flux=sum_v av[v]*Bphi[k=0,v]
  return {K[ion/electron],Ui,Uperp,Uparallel,EB=E_B,EL=E_L,total=E_total,
    number[species],mass[species],charge[species],topology,physical_flux}
```

拓扑通量是一个极向截面的环向磁通，不再乘 2*pi；体积分才乘环向角。能量不先把场插到节点再换权。总数可用物理样本积分独立复算；积分电荷允许非零，不强制中性。

## A0-DIAG-04 — CheckStepAndLedger

```text
CheckStep(left,base,midU,mid,right,end,interval,original_baseline,ctx):
  require interval and all stage identities match scheduler; mid weight==dt
  dt=ctx.config.time.dt; EL_L=base.cache.EL; EL_R=end.cache.EL
  rho_L=base.sources.total_rho; rho_R=end.sources.total_rho
  for species:
    RC_a=(right.n-left.n)/dt+mid.sources[a].number_divergence
    boundary_flux=sum independently over actual wall F,with area and angular weights
    total_balance=Number(right)-Number(left)+dt*boundary_flux
    integral_balance=right.n-left.n+DJ*interval[a].integrated_number_flux
  RC_flux=(rho_R-rho_L)/dt+sum_a q_a*mid.sources[a].number_divergence
  RC_mom=(rho_R-rho_L)/dt+DJ*mid.sources.total_current
  deltaC=DJ*mid.sources.total_current-sum_a q_a*mid.sources[a].number_divergence
  RL=LongitudinalProject(mid.sources.total_current)+epsilon0*(EL_R-EL_L)/dt
  delta_divB=DB*(B_R-B_L); topology_delta=ell*(B_R[k=0]-B_L[k=0])
  CT_residual=(B_R-B_L)/dt+C*mid.cache.ET
  record original_baseline.initial_divB,current RB,delta_divB independently
  record long_term_divB_change=DB*right.B-original_baseline.initial_divB
  reference existing end.checks for endpoint constraints; do not duplicate their keys
  record per-species/global number/charge balances with separate step-check names

  TL=StateTotals(left,base.cache,base.samples)
  TR=StateTotals(right,end.cache,end.samples)
  for species:
    L=mid.ledger[a]; W_electric=dt*L.rates.electric_power_edge
    W_pressure_internal=dt*L.rates.pressure_power_internal; W_wall=dt*L.rates.wall_power
    FK=mid.boundary_ledger[a].kinetic_advective_flux
    RK=TR.K[a]-TL.K[a]+dt*FK-dt*(L.rates.electric_power_edge
       +L.rates.pressure_power_kinetic+L.rates.wall_power+L.rates.magnetic_power)
    record RK and each term; convection/pressure pairing defects remain separate
  FUi=mid.boundary_ledger[ion].internal_transport_flux
  FUperp=mid.boundary_ledger[electron].perp_internal_transport_flux
  FUparallel=mid.boundary_ledger[electron].parallel_internal_transport_flux
  RUi=TR.Ui-TL.Ui+dt*FUi-dt*mid.ledger[ion].internal_rates.ion_internal_rate
  RUperp=TR.Uperp-TL.Uperp+dt*FUperp-dt*mid.ledger[electron].internal_rates.electron_perp_rate
  RUparallel=TR.Uparallel-TL.Uparallel+dt*FUparallel-dt*mid.ledger[electron].internal_rates.electron_parallel_rate
  PB=real(2*pi*sum_mode,entity(MB*conj(midU.B)*mid.RHS.B))/mu0
  PL=epsilon0*real(2*pi*sum_mode,entity(M1*conj(mid.cache.EL)*mid.field_derivatives.ELdot))
  record RBenergy=TR.EB-TL.EB-dt*PB, RLenergy=TR.EL-TL.EL-dt*PL
  field_power=PB+PL
  record field_power+sum_species(mid.ledger[a].rates.electric_power_edge)
    +mid.boundary_ledger.field_energy_flux
  total_energy_residual=TR.total-TL.total+dt*mid.boundary_ledger.total_energy_flux
  # 此模型无外源/额外状态修改；Wext=0、deltaEalg=0 必须由操作计数为零确认
  record spectral_transfer_defect separately,not as a compensating physical source
  record momentum residual=end.momentum_frame.P_cartesian-base.momentum_frame.P_cartesian
    +dt*mid.momentum_frame.outward_momentum_flux
  record angular residual=end.momentum_frame.Lz-base.momentum_frame.Lz
    +dt*mid.momentum_frame.outward_torque_z
  increment=dt*mid.ledger rates and boundary/field rates + accepted-step counters only
  for every step record: set time_s=right.time,interval={id,t_left,t_right,quadrature,source_refs}
  # 阶段记录 time_s 来自 StageTag，没有区间积分时 interval=null
  return StepDiagnostics{records,increment,totals=TR,owning selected raw_residuals,
    stats=merge(base.stats,mid.stats,end.stats)}; never mutate cumulative ledger
```

逐物种能量差保留有符号量；总量的相消不能替代分项通过。数值谱/时间误差已在 raw residual 中，不能把每步能量差直接定义成 `deltaEalg` 再强行闭账。若未来加入真正状态滤波，必须记录该操作实际前后能差并更新模型/算法选择。

### Test-only manufactured source branch

The optional source-aware overload preserves the ordinary eight-argument API.
For a supplied midpoint source, subtract S_n from the number difference quotient,
its charge sum from RC, `S_EL` from the longitudinal-field difference quotient,
and `dt*S_B` from the magnetic increment before CT/divergence checks. Subtract
`dt` times the independently evaluated source number, topology, energy and
momentum rates from their corresponding interval balances. The physical-stage
field-exchange check remains unforced. See [source rates](source.cpp.md). An
omitted source must trigger the ordinary hard conservation gates; it is never
defined from the observed conservation defect.

## A0-DIAG-05 — MomentumAndStress

```text
MomentumAndStress(U,samples,cache,boundary,ledger,ops):
  Ndiag=max(Nphi,3); angular_weight=2*pi/Ndiag
  if Nphi==1: require K==0; broadcast the single cylindrical sample/trace to 3 angles
  else: reuse actual Nphi samples/traces without additional interpolation
  EL_samples=InverseRetained(cache.EL,Ndiag,true); ELn=H*EL_samples
  BN=samples.BN (broadcast if Nphi==1)
  P_cartesian=0; Lz=0; outward_momentum_flux=0; outward_torque_z=0
  for diagnostic sample phi=2*pi*l/Ndiag and cell c:
    gD=epsilon0*cross(ELn,BN)
    ptotal=pi_i+pi_e+gD
    p_cart=(ptotal_R*cos(phi)-ptotal_phi*sin(phi),
            ptotal_R*sin(phi)+ptotal_phi*cos(phi),ptotal_Z)
    P_cartesian+=M0[c]*angular_weight*p_cart
    Lz+=M0[c]*angular_weight*Rc[c]*ptotal_phi
    similarly integrate ledger.terms.wall_reaction in Cartesian basis and its z torque
  for each wall segment/diagnostic sample using boundary traces:
    sigmaB=(Bb outer Bb-|Bb|^2*I/2)/mu0
    sigmaD=sigmaB+epsilon0*(ELb outer ELb-|ELb|^2*I/2
               +ELb outer ETb+ETb outer ELb-dot(ELb,ETb)*I)
    traction=sum_species(m*nb*ub*un+Pwall*normal)-sigmaD*normal
    outward_momentum_flux+=area*angular_weight*Cartesian(traction,phi)
    outward_torque_z+=area*angular_weight*Rwall*traction_phi
  return MomentumFrame with these totals,quadrature_N=Ndiag and reconstruction label
```

字段 `momentum_reconstruction="H/QB nodal quadrature"` 标记该诊断包含场转移误差；它不是本离散全域精确动量定理。不能把数值反力与连续应力两套边界账重复加一次。物理动量账和数值反力定位账分列，按声明的观测预算判定，不以零能流推导零冲量。

## A0-DIAG-06 — MakeSnapshotAndCost

```text
MakeSnapshot(committed,frame,context):
  require accepted and cache tag.stage in {initial,restart,endpoint}
  require cache tag generation/time/step matches committed evolution
  require frame accepted-state identity matches committed; raw maps retain their own tags
  samples=RecoverSamples(committed.evolution,context.Nphi)  # 只读重新派生 u/BN，不重解场/RHS
  synchronize backend; copy into separate owning host buffers
  scatter active arrays into rectangular slots,fill inactive slots canonical zero
  save all evolution plus EL/ET/phi from cache,u/BN from samples
  extrema=min/max over valid recovered samples,with actual entity/sample locations
  save extrema and selected residual maps owned by frame; missing requested map -> Failure
  mark caches derived,include masks/entity IDs,units,tag,layout,hashes
  no working lambda/ghost is labeled evolved state
```

基础计时使用 steady clock；GPU 区间前后在相同 stream 记录事件并同步读取，单独记录 H2D/D2H、setup、FFT、LU、stage、diagnostics、I/O。benchmark 记录 warmup、重复次数、规模、FP64、设备和精确输入；warmup 使用副本，不累计物理步。没有 NCU 门，也不自动做 kernel 调优。离线 `AnalyzeConvergence` 见 tools；缺参考输出 UNAVAILABLE，不伪造收敛阶次。
