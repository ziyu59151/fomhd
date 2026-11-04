# 同阶段计算与源所有权

> 状态：active；目标 `src/time/stage.cpp`。依据 GC13–15、StageSources/IntervalSources 合同。  
> 依赖：[state](../core/state.cpp.md)、[fluid](../fluid/fluid.cpp.md)、[fields](../fields/darwin.cpp.md)、[diagnostics](../diagnostics/diagnostics.cpp.md)。同步 time/tests；最近同步：2026-11-03。

## A0-STAGE-01 — EvaluateStage

调用者提供只读演化态、明确 StageTag、静态算子、运行时、可写独占工作区。返回 StageResult 自己拥有 RHS/cache/ledger 及仍需消费的采样量；函数返回后借用不得指向已回收的临时栈/arena。

```text
EvaluateStage(U,tag,ctx,enable_alias_probe):
  require tag.generation==U.generation and tag.time==U.time
  require exact geometry/layout/operator/boundary identities
  U=BindStageView(U,tag)                       # 只重绑只读描述器，非状态修改
  TRY CheckRetained(U)
  samples=TRY RecoverSamples(U,ctx.Nphi)       # 先检查再作除法
  for a in registered_species ordered by id:
    fluid[a]=TRY BuildFluidContext(a,U,samples,operators)
    nonE[a]=TRY BuildNonElectricRHS(fluid[a],samples.BN,samples.bhat)
    source[a]={rho=q*n_hat,number_flux=fluid[a].Fhat,current=q*fluid[a].Fhat,
      number_divergence=fluid[a].divFhat,raw_second_moment=fluid[a].raw_moment,
      boundary_number_flux=fluid[a].boundary_number_flux,
      boundary_number_flux_integral=fluid[a].boundary_number_flux_integral,
      response_action_ref=(ElectricMomentumAction,immutable_context=fluid[a]),
      producer_kind=fluid,population_id=a.id,tag,source_version=NewSourceVersion()}
  sources=TRY AggregateStageSources(source,tag)
  electrostatic=TRY SolveElectrostaticDarwin(sources.total_rho,tag)
  fJ0,AE_action=TRY BuildCurrentResponse(fluid,nonE)
  transverse=TRY SolveTransverseDarwin(fJ0,AE_action,electrostatic.EL,tag)
  cache={phi=electrostatic.phi,EL=electrostatic.EL,ET=transverse.ET,E=transverse.E,
    solve_results=[electrostatic.solve_results,transverse.result],tag}
  for a:
    rhs[a],terms[a]=TRY FinishFluidRHS(fluid[a],nonE[a],cache.E)
    ledger[a]=TRY StagePowerLedger(fluid[a],nonE[a],terms[a],rhs[a],cache)
  rhs.B=TRY BuildMagneticRHSCT(cache.ET)
  current_response=TRY CheckCurrentDerivative(rhs.mom_i,rhs.mom_e,fJ0,AE_action,cache.E)
  boundary_ledger=TRY ActualBoundaryTracesAndFluxes(U.B,samples,fluid,nonE,cache,operators,tag)
  field_derivatives=TRY FieldDiagnosticDerivatives(sources.total_current,U.B)
  momentum_frame=TRY MomentumAndStress(U,samples,cache,boundary_ledger,ledger,operators)
  result=own(tag,RHS=rhs,cache,samples,fluid,nonE,terms,sources,ledger,
    boundary_ledger,field_derivatives,current_response,momentum_frame,stats)
  result.checks,result.raw_residuals=CheckStage(U,result,operators,budgets)
  TRY Gate(result.checks,required_stage_keys(tag))
  if enable_alias_probe:
    alias=TRY AliasProbe(U,result)             # alternate invocation disables recursion
    result.checks+=alias.records; result.alias_tag=alias.tag
    TRY Gate(result.checks,required_stage_and_alias_keys(tag,alias.tag))
  return immutable result
on any failure:
  add module/tag,already-produced records and available partial arrays to Failure
  append completed records by full identity without duplicates
  mark not-yet-computed checks NOT_RUN with cause; no missing numeric value becomes 0
  retain trial_* B/n/momentum/pressure arrays, available residuals and trial_E
  return Failure; committed state/cumulative ledger are untouched
```

alias 场重解使用自己的 sampling context/工作区；既有主阶段不能被覆盖。记录总 solve 数含 alias 成本，并将物理阶段与诊断求解次数分列。固定 geometry 结构可复用；当前 n、u、P、壁因子与 AE 每阶段重建。

## A0-STAGE-02 — AggregateStageSources

```text
AggregateStageSources(items,expected_tag):
  require items.keys exactly equal registered active fluid population IDs
  seen=empty set; total_rho=0; total_current=0
  for id in sorted registry:
    item=items[id]
    require item.tag==expected_tag, geometry/layout/operator identity equal
    require item.producer_kind==fluid for A0
    for role in {rho_stage,J_mom_stage,number_flux,raw_second_moment,response}:
      key=(id,role,tag,source_version); require key not in seen; insert
    require current consumer shares same immutable flux owner/version as density
    total_rho+=item.rho
    total_current+=item.current
  return immutable StageSources(items,total_rho,total_current,seen,
    id=NewStageSourceSetId(),tag=expected_tag)
```

source_version 由生产者生成，派生和聚合不可另改。`StageSources` 不是 `IntervalSources` 的别名；A1 必须传实际积分权，A2 粒子不注册为活动源，A3 新群加入注册表前必须具备专门的瞬时/区间接口与无双计检查。

## A0-STAGE-03 — InitialStateAcceptance

```text
ValidateInitialState(U,context,kind):
  tag.stage=restart if kind==non_equilibrium_restart else initial
  tag.step=U.step; tag.time_s=U.time; tag.state_generation=U.generation
  tag.trial_id=NextTrialId(); tag.spectral_context_id=context.id
  tag.window_id=tag.iteration_id=0           # 不再次分配状态 generation
  initial=TRY EvaluateStage(U,tag,context,enable_alias_probe=true)
  records,baseline=CheckInitial(U,initial,operators,budgets,kind)
  # CheckInitial 已按 kind 加入全部演化量平衡门，或给出 not claimed；不重复追加同名记录
  TRY Gate(records,required_initial_keys(kind,tag,initial.alias_tag))
  bind accepted U descriptors to tag without changing numerical arrays
  frame=own(accepted_state_identity=tag,records,baseline.energy,initial.momentum_frame,
    initial.stats,initial.raw_residuals,initial_baseline=baseline)
  return CommittedState(move(U),initial.cache),frame
on any failure after EvaluateStage:
  retain initial tag,completed records,state arrays and raw residuals
  mark required records not reached as NOT_RUN without duplicating existing keys
  do not mutate cumulative baseline,last step,or committed state
```

一般非平衡初态也必须满足磁散度、Ampere、Gauss 与壁约束。外部 MHD 总压力/总电流不被自动拆成两个流体。没有 burn-in、强制准中性、初始化磁投影或额外 Ohm 求解器。
