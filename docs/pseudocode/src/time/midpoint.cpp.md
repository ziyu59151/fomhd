# A0 同细步显式中点与提交事务

> 状态：active；目标 `src/time/midpoint.cpp`。依据主规格 §8、GC16、09 时间账。  
> 依赖：[stage](stage.cpp.md)、[state](../core/state.cpp.md)、[diagnostics](../diagnostics/diagnostics.cpp.md)、[run IO](../io/run.cpp.md)。最近同步：2026-11-03。

## A0-TIME-01 — AdvanceOneStep

```text
AdvanceOneStep(committed,cumulative,dt,context):
  require dt>0 finite; require committed.accepted
  require committed.step<config.time.nsteps
  trial=NextTrialId(); n=committed.step; t=committed.time
  require t==TimeAtStep(n); tm=TimeAtMidpoint(n); tr=TimeAtStep(n+1)
  require finite(t,tm,tr) and t<tm<tr
  # 所有临时量独立拥有，committed/cumulative 在整个 prepare 阶段只读
  base_tag=(n,base,t,committed.generation,0,0,trial,main_context)
  base=TRY EvaluateStage(committed.evolution,base_tag,context,false)
  mid_tag=(n,midpoint,tm,NewGeneration(),0,0,trial,main_context)
  midU=TRY CombineEvolution(committed,base.RHS,dt/2,mid_tag,mid_buffer)
  mid=TRY EvaluateStage(midU,mid_tag,context,true)
  end_tag=(n+1,endpoint,tr,NewGeneration(),0,0,trial,main_context)
  candidate=TRY CombineEvolution(committed,mid.RHS,dt,end_tag,candidate_buffer)
  end=TRY EvaluateStage(candidate,end_tag,context,false)
  intervals=TRY MakeA0Interval(committed,candidate,mid.sources,dt,trial)
  step_diag=CheckStep(committed.evolution,base,midU,mid,candidate,end,intervals,cumulative.baseline,context)
  checks=base.checks+mid.checks+end.checks+step_diag.records   # mid 含 alias 检查
  TRY Gate(checks,required_step_keys(base.tag,mid.tag,mid.alias_tag,end.tag,intervals.id))
  pending=PrepareCommit(candidate,end,checks,step_diag,intervals,cumulative)
  return FinalizeA0Step(committed,cumulative,pending)   # 唯一修改点；返回预分配结果
on Failure:
  preserve last accepted state and cumulative bytes; no physical ledger add
  keep the failing stage tag; if whole-step Gate fails,use interval tag
  merge base/midpoint/endpoint records without duplicate identities
  retain base_*,midpoint_*,endpoint_* state arrays for each available state
  retain residuals under prefix.<key> and E under prefix_E
  mark requirements of phases not reached as NOT_RUN with the first failure code
  return RejectedResult(trial,Failure,available candidate/partial stages)

NumericalError from ledger accumulation or pre-commit device synchronization
enters the same evidence path as Failure:
  convert to Failure(code=NumericalLibrary)
  retain base/midpoint/endpoint arrays and checks
  retain the completed StepResult checks and interval flux/current arrays
  set stage=interval when no earlier stage failed
  committed state and cumulative ledger remain unchanged

add_source(rhs,state,stage_tag,saved_source,available):
  callback=TRY source(state,stage_tag,context)
  available=true
  TRY check_source(callback)
  on Failure:
    annotate the Failure with stage_tag when it has no stage
    rethrow
  on NumericalError:
    convert to Failure(NumericalLibrary) with stage=stage_tag
  return rhs+callback

combine_stage(base,rhs,weight,stage_tag):
  TRY combine(base,rhs,weight,stage_tag,context)
  on Failure:
    annotate with stage_tag when missing
  on NumericalError:
    convert to Failure(NumericalLibrary) with stage=stage_tag

Evidence retention uses the explicit source-available flags. An empty or malformed
source B is retained when the callback returned and source validation failed;
B.empty() is not a source-presence test.

The interval tag is the fallback only for check_step, ledger accumulation, or
pre-commit synchronization. Source callbacks use base/midpoint, and combine
uses the target state tag (midpoint or endpoint).
```

调度器在 host 固定使用 `TimeAtStep(n)=fma(double(n),dt,start_time)`、`TimeAtMidpoint(n)=fma(double(n)+0.5,dt,start_time)`；要求 nsteps≤2^52，确保所需整数和半整数可精确表示。时间分辨率不足返回错误。restart 沿用同规则，不重设时间原点。演化权、区间求积及残差的 dt 始终使用 config.time.dt，不能换成两个舍入时间标签的差。时间标签差只作浮点一致性核验：误差不得超过 `4*epsilon*max(abs(t_left),abs(t_right),dt)`。端点求得的 E 是新缓存；它不能替换/修正 candidate.B。

## A0-TIME-02 — MakeA0Interval

```text
MakeA0Interval(left,right,mid_sources,dt,trial):
  require right.time-left.time agrees with declared fixed dt under time arithmetic check
  require midpoint tag time exactly matches scheduler midpoint expression
  quadrature=[{source_id=mid_sources.id,weight_seconds=dt}]
  for species a:
    integral_F=dt*mid_sources[a].number_flux
    integral_J=q_a*integral_F
    boundary_charge_integral=dt*q_a*mid_sources[a].boundary_number_flux_integral
    record left/right rho from respective states
    check right.n-left.n+DJ*integral_F on all rows
  id=(runtime_uuid,trial,left.step,right.step)
  return IntervalSources(id,accepted=false,quadrature,...)
```

base/endpoint/alias 的账一律不在求积列表中。场瞬时电流、数通量和区间积分三者带独立量纲；不能将 `dt*J` 再传到瞬时 Darwin 输入。每个物种单独验连续性，之后才汇总总电荷。

## A0-TIME-03 — PrepareCommit / FinalizeA0Step

```text
PrepareCommit(candidate,end,checks,step_diag,interval,cumulative):
  require end.cache.tag generation/time corresponds to candidate
  require no interval key previously committed; interval.accepted==false
  allocate pending cumulative copy and add step_diag.increment once,check finite sums
  pending.cumulative.last_interval_id=interval.id; store its endpoint step/time
  frame=own(accepted_state_identity=end.tag,step_diag.totals,end.momentum_frame,step_diag.stats,
    selected copies of end.raw_residuals and step_diag.raw_residuals)
  pending.evolution=move(candidate); pending.cache=move(end.cache)
  pending.result={accepted=false,step=pending.evolution.step,time=pending.evolution.time,
    records=move(checks),interval=move(interval),diagnostic_frame=move(frame)}
  require all device tasks synchronized successfully
  return owning pending_bundle; all operations that can allocate/fail occur here

FinalizeA0Step(committed,cumulative,pending):
  # 单线程内存事务；只做 noexcept moves/swaps，不能在中途写文件或分配
  swap committed.evolution,pending.evolution
  swap committed.cache,pending.cache
  swap cumulative,pending.cumulative
  committed.accepted=true; pending.result.accepted=true; pending.result.interval.accepted=true
  static_assert(is_nothrow_move_constructible_v<StepResult>)
  return noexcept move(pending.result)
```

提交之后发生 I/O 失败：内存中的新状态仍是 accepted，记录 `io_failure_after_commit` 和最后持久 checkpoint，不假装回滚到旧物理步；停止下一步，尽力写失败回执。restart 权威是完整原子 checkpoint，不是 history 行数。进程在内存提交期间崩溃只恢复上一个完整 checkpoint，不从半写日志推断状态。

## A0-TIME-04 — RunLoopAndFailure

```text
while committed.step<nsteps:
  result=AdvanceOneStep(...)
  if rejected:
    WriteFailure(runtime_phase=numerical_trial,last_committed,result)
    mark run failed; exit numerical_error
  WriteAcceptedDiagnostics(result)
  if step divisible by snapshot_stride or final:
    WriteSnapshot(MakeSnapshot(committed,result.diagnostic_frame,context))
  if step divisible by checkpoint_stride or final: WriteCheckpoint(committed,cumulative,context)
mark run completed only after final output success
```

A0 失败后不自动减步/重试；用户以新运行目录显式修改算例再运行。可捕获终止请求只在安全阶段边界停下：先保留 accepted 状态并尝试 checkpoint，再退出；signal handler 只设置原子标志，不调用 CUDA/文件 I/O。不能捕获硬崩溃后声称最后一步已持久化。

提交后输出失败记录 `runtime_phase=post_commit_output` 与
`accepted_trial=true`；数值试算拒绝记录 `runtime_phase=numerical_trial` 与
`accepted_trial=false`。两种记录都保存最后一个完整 durable checkpoint。

测试 `TIME-01..04` 覆盖中点阶次、三个阶段失败回滚、重复账与提交后 I/O 故障。显式中点在纯虚轴没有非零绝对稳定区，有限时窗、相位和增长预算须实际核验。
