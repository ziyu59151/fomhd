# 运行目录、日志、原子输出与 restart

> 状态：active；目标 `src/io/run.cpp`。依据主规格 §9.1 与工程约定。  
> 依赖：[config](config.cpp.md)、[NPZ](npz.cpp.md)、[midpoint](../time/midpoint.cpp.md)。同步 app/tests；最近同步：2026-11-03。

## A0-RUN-01 — CreateRun

```text
CreateRun(config,input_bytes,build_manifest):
  path=config.io.output_dir
  require path absent or an existing empty directory owned by this run's caller
  create directory if absent; create lock with O_CREAT|O_EXCL,mode=0600
  if lock or any preexisting content exists: fail RunDirectoryNotEmpty
  create inputs,provenance,logs,snapshots,checkpoints,failure subdirectories
  copy exact input config,geometry,initial,source manifest bytes into inputs
  verify copied SHA256 equals pre-load hash; input changed during copy -> fail
  save resolved config and explicit CLI overrides separately
  run_id=case_name+'__'+UTC_timestamp+'__'+random_unique_suffix
  manifest.status="initializing"; accepted_step=null; durable_checkpoint=null
  manifest includes input hashes,design baseline hashes,source commit+dirty snapshot,
    binary SHA256,build flags,loaded native library versions,backend/precision,
    device UUID/model,Slurm job ID if present,units/layout,budgets,source identities
  AtomicJSON(run.json,manifest)
  open logs/run.log,history.csv,checks.jsonl in single-writer mode
```

默认推荐 `~/runs/fomhd/a0-<case>__<UTC>__<suffix>/`，但可执行程序不硬编码家目录。路径先做 canonical parent 校验，不跟随运行产物的意外 symlink。原输入不修改；大型数据可用显式不可变内容存储引用，只有已验证内容存在且可追溯才允许省去重复拷贝。

构建来源通过生成的 `build_manifest.json` 供程序读取，不在运行时执行任意 shell/Git 命令。dirty 调试二进制必须随构建保存相对基线的 patch 和涉及的未跟踪源码或完整源码快照；只有 commit+dirty 标志的正式 benchmark 拒绝。快照限项目源码/配置，不包含环境、文献、凭据或个人文件。

## A0-RUN-02 — AtomicPublish

```text
AtomicPublish(target,writer,replace_allowed):
  require target's parent belongs to current run and has no second writer
  if not replace_allowed: require target absent
  create unique temp sibling with exclusive creation in same filesystem
  phase="temp_created"
  TRY writer(temp)                         # NPZ/JSON，完成后关闭其库句柄
  fd=open(temp,O_RDONLY); fsync(fd); close(fd)
  validate temp format/hash/expected metadata
  if not replace_allowed: require target still absent under run lock
  rename(temp,target)                      # Linux 同目录原子 rename
  phase="renamed"
  fd=open(parent,O_DIRECTORY|O_RDONLY); fsync(fd); close(fd)
  return target_hash
on error before rename:
  preserve temp as unpublished if created; leave old target intact; return Failure(phase)
on error after rename:
  return Failure(phase="renamed_not_confirmed_durable",target_may_be_published=true)
  # target 可见但目录持久化未确认；temp 已不存在，不承诺回滚/旧指针仍在
```

唯一 step 文件从不覆盖；只有本运行 `run.json/latest.json` 这类指针允许原子替换。单写者锁不是对其他用户文件的权限声明。fsync/rename 错误必须传播；rename 后的目录 fsync 失败属于持久性未确认，不承诺旧指针还在。进程崩溃留下 temp 可由 inspect 标记为 incomplete，不读取为 restart。

## A0-RUN-03 — DiagnosticsAndFailure

```text
WriteAcceptedDiagnostics(result):
  require result.accepted; refuse duplicate step in this writer
  append DiagnosticRecords to checks.jsonl,one finite/explicit-null JSON per line
  append exactly one fixed-column history.csv row
  snapshots include every Poisson/Darwin solve residual, scales, singular values,
    backward error and dimensionless refinement_steps count (0 through 5)
  flush streams,check errors; periodically fsync at checkpoint boundaries
  update run manifest last_accepted_step/time

WriteFailure(last_committed,rejected_or_io_error):
  record accepted=false for numerical trial; retain last_committed separately
  application supplies runtime_phase={pre_commit_input,numerical_trial,post_commit_output}
  accepted_trial==true only for post_commit_output
  record last_durable_checkpoint from the manifest before writing failure output
  metadata includes original module/code/tag,raw values,budgets,library info,
    completed checks,NOT_RUN checks,available arrays,last durable checkpoint
  preserve NaN/Inf as typed strings in JSON and original FP bits in failure NPZ
  write failure/unique_trial_id.json and .npz atomically if IO still available
  if write itself fails: log stderr and preserve original numerical error + IO error
  manifest.status="failed" or "io_failed_after_commit"; no successful completion claim
```

history 固定列在 writer 初始化时由 schema 生成：step/time/dt、每物种 number/mass/charge、七项能量、total、RG/RB/RA/RL/RC、壁与功率、solve_count、elapsed_s；每列的单位/公式/缺失值规则写 manifest。详细数组检查在 checks/snapshot，不把变动字段偷偷加到 CSV 的中间行。

## A0-RUN-04 — CheckpointAndRestart

```text
WriteCheckpoint(committed,cumulative,context):
  require accepted and cache tag.stage in {initial,restart,endpoint}
  require cache tag generation/time/step matches committed evolution
  if manifest durable pointer names this step:
    require checkpoints/latest.json exists
    require pointer file/sha256/step/time/accepted identities equal manifest
    require checkpoint file exists and hash matches
    return
  synchronize backend
  arrays=ten evolution arrays + step,time,dt,start_time,schema/layout/model,
    config_hash,semantic_hash,geometry_hash,operator_rule_hash,matrix_fingerprint,accepted=true,
    original_run_id,baseline_divB=cumulative.baseline.initial_divB,
    cumulative_json (including last_interval_id,endpoint identity and scalar baseline metadata)
  hash=AtomicPublish(restart_<step>.npz,WriteNPZ,false)
  latest={file,sha256,step,time,accepted:true,schema_version}
  AtomicPublish(checkpoints/latest.json,WriteJSON,true)
  update manifest.durable_checkpoint only after both succeed

LoadRestart(checkpoint,new_case,context):
  read NPZ exact checkpoint schema; require accepted==true
  require no failure marker,model A0,known layout/schema
  validate whole-file hash if pointer/parent manifest supplied
  require semantic_hash matches new resolved case
  require context was built from the recorded/copied original geometry
  require geometry_hash,operator_rule_hash and matrix_fingerprint match context
  if actual matrix fingerprint differs: refuse exact-replay; a new comparison run is separate
  require step<=new_case.nsteps; finite time/dt and scheduler time rule consistent
  parse cumulative_json with duplicate-key rejection; validate finite typed sums/baseline
  if last_interval_id is null:
    require step==baseline.step==0 and time==baseline.time==start_time
    require all accepted-interval sums/counters are zero
  else: require stored last interval ends at checkpoint step/time
  baseline.initial_divB=unpack(baseline_divB); validate shape/units/finite values
  unpack evolution; ignore derived E/phi even if present in optional snapshot namespace
  assign fresh runtime generation and restart tag
  committed,frame=ValidateInitialState(evolution,context,kind="non_equilibrium_restart")
  discard frame.initial_baseline; retain original checkpoint cumulative.baseline
  # 不把一般运行中状态重新当成静平衡；约束与正域仍全部验收
  parent_linkage={checkpoint_path,checkpoint_sha256,original_config_hash,original_run_id}
  # 不创建输出目录；app 的唯一 RunWriter 负责保存 parent_linkage
  load cumulative once; do not replay already accepted powers
  return committed,cumulative,frame,parent_linkage
```

严格 restart 不支持换 K、N、网格、dt、物种、单位、精度或物理边界。只改输出路径、保存频率、最终停止步与显示计时不改变演化；保留原 config_hash 并记录新 manifest。恢复的初始诊断基线（初始 divB、拓扑/能量/时间原点）来自 checkpoint 中 cumulative 的 baseline 字段，不能重置成 restart 时刻而掩盖长期漂移。

## A0-RUN-05 — Inspect

只读检查 run.json、checkpoint pointer/hash、CSV 行连续性与 JSONL 可解析性；最后一行被截断则报告 interrupted，不补写数据。显示 last accepted 与 last durable checkpoint 的区别、失败原因、源版本、hard gate 状态。inspect 不创建 CUDA context、不重算物理、不修改或“修复”运行目录。
