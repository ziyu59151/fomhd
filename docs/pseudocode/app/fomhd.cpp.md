# 可执行程序入口

> Status: active; target `app/fomhd.cpp`. The `fomhd` executable and its CLI tests are implemented; the command scope is still limited to A0.依据主规格 §3/8/9 与本目录完整接口。  
> 依赖：[config](../src/io/config.cpp.md)、[run](../src/io/run.cpp.md)、[stage](../src/time/stage.cpp.md)、[midpoint](../src/time/midpoint.cpp.md)。最近同步：2026-11-04。

## A0-APP-01 — CLI

```text
main(argc,argv):
  accept exactly:
    fomhd validate <case.toml> [--output <new-dir>]
    fomhd run <case.toml> [--output <new-dir>] [--restart <checkpoint.npz>]
    fomhd inspect <run-dir>
    fomhd --help / --version
  reject unknown/repeated option or extra positional argument
  if help/version: print usage/build identity; return 0
  if inspect: return Inspect(path) without GPU initialization
  install SIGINT/SIGTERM flag handlers before case loading, initialization,
    and accepted-state output
  runtime_phase=pre_commit_input
  config=TRY LoadCase(case_path,declared_overrides)
  runtime=TRY CreateCpuRuntime or CreateCudaRuntime according to config
  run=TRY CreateRun(config,inputs,build_manifest)
  geometry=TRY PolygonAndMask -> EnumerateAndValidateTopology -> Metrics -> DistanceAndHash
  operators=TRY LowOrderMatrices -> GraphFiltersAndTransforms -> FixedTransfers
  TRY OperatorAcceptanceAndTopology
  context=borrow(config,runtime,geometry,operators,budgets),with fresh runtime_uuid
  run.manifest.operator_rule_hash=operators.operator_rule_hash
  run.manifest.matrix_fingerprint=operators.matrix_fingerprint
  if restart:
    committed,cumulative,initial_frame,parent_linkage=TRY LoadRestart(checkpoint,config,context)
    deferred_parent=parent_linkage
  else:
    U=TRY LoadA0State(initial.npz,geometry,layout,config)
    committed,initial_frame=TRY ValidateInitialState(U,context,config.time.initial_kind)
    cumulative=zero typed ledger,last_interval_id=null,baseline=move(initial_frame.initial_baseline)
  accepted=committed.step; phase=5; runtime_phase=post_commit_output
  if deferred_parent: run.manifest.parent=deferred_parent
  write initial_frame records/history once under the accepted initial/restart step identity
  WriteSnapshot(MakeSnapshot(committed,initial_frame,context))
  WriteCheckpoint(committed,cumulative,context)
  if validate:
    mark run.status="validated_initial_state"; return 0
  TRY RunLoop(committed,cumulative,config)
  mark completed after final output; return 0
catch Failure:
  if run exists: WriteFailure with runtime_phase,last accepted identity and partial context
  else: print structured input/runtime failure to stderr
  return mapped nonzero exit code
catch std::bad_alloc / std::exception:
  convert to MemoryLimit / InternalFailure; retain first cause and runtime_phase;
  never report success
```

run loop 调用 advance 前设 `runtime_phase=numerical_trial`，返回后设
`post_commit_output`。可捕获终止标志在下一步前处理：已有同一步完整 checkpoint 时
不重复写入，否则尝试 checkpoint；随后标记 `stopped_at_accepted_boundary` 并返回 130。

退出码：0 完成该命令的范围；2 CLI/输入/schema；3 backend/resource；4 数值/正域/约束拒绝；5 文件 I/O；6 内部未预期错误；130 安全停止。`validate` 成功只表示指定初态与初始化检查通过，不代表时间演化或科学 benchmark 通过。

## A0-APP-02 — RuntimeOwnership

app 拥有 CaseConfig/Runtime/GeometryOperators/RunWriter/CommittedState/CumulativeLedger；time 借用它们并独占试算工作区；diagnostics 只读。错误上抛回 app 时，局部 RAII 先安全结束设备任务再释放。`run` 目录建立失败不会触碰已存在内容；初始化数值失败仍保存已读取的配置、算子检查与失败实体，便于诊断。

读取 restart 元数据及兼容性可在 CreateRun 前做只读预检；真正重建/检查成功后才生成新已接受初始 checkpoint。不要因为命令叫 run 就把未经验收的初态标 accepted。
