# 公共数据类型

> 状态：active；目标 `include/fomhd/types.hpp`。依据 A0S1、GC9–GC16、09 诊断合同。  
> 依赖：[总约定](../../README.md)；变更同步所有消费者、NPZ schema 和 tests。最近同步：2026-11-03。

## A0-TYPE-01 — 数组与索引

```text
Real = binary64; Complex = pair(Real re, Real im)
EntityId = size_t; Mode = signed int64; StepId, Generation, TrialId = uint64
Location = {Cell, RFace, ZFace, Vertex}
Space = {Host, Device}; Backend = {CPU, CUDA}

EntityTable: keys[(i,j)], dense_to_active[], active_to_dense[], R[], Z[], boundary[]
  # Cell 的键为 (i,j)，RFace 为 (e,j)，ZFace 为 (i,f)，Vertex 为 (e,f)
IndexSpace: cells, rfaces, zfaces, vertices; NR,NZ; geometry_hash
FourierLayout: K, M=2*K+1, Nphi odd>=4*K+1, global_phase_origin=0
  stored_mode(s)=int64(s)-K
  offset(s,entity,nentity)=checked_add(checked_mul(s,nentity),entity)
  # 标量紧凑谱 [M][nentity]，同一 mode 内 entity 连续
  # 物理工作数组 [Nphi][nentity]；三分量为三个独立数组，不做 AoS
OwnedArray<T>: move-only owning buffer, shape, strides, Space, completion_event
View<T>: borrowed pointer, shape,strides,Space,owner_generation
ConstView<T>: read-only borrow; 不能转回可写指针
QuantityDescriptor: name,unit,location,population_id,representation,
  geometry_hash,operator_hash,layout_version,stage_tag,valid_entity_set
```

`representation` 仅为 `retained_spectrum / padded_samples / full_sample_spectrum`。不要用相同类型的裸指针隐式混用。复数文件不直接 memcpy 一个未知 ABI 的 `Complex`。标准库 `std::complex<double>` 可作为 host 算术表示，FFT/BLAS/磁盘适配见各分册。

磁场 `(BR[z],Bphi[v],BZ[r])`；棱量 `(ER[r],Ephi[c],EZ[z])`；流体三分量全部 `[c]`。磁散度在 `[v]`，电荷散度在 `[c]`。`M0[c]`、`M1=(r,c,z)`、`MB=(z,v,r)` 为正实权；所有 ghost/非活跃槽不属于这些紧凑空间。

## A0-TYPE-02 — 演化状态与阶段

```text
SpeciesConstants: id, mass_kg>0, charge_C; ion.gamma>1
IonFluidState: n[c], mom[3][c], p[c]                         # 每项带 M 个模
ElectronFluidState: n[c], mom[3][c], p_perp[c], p_parallel[c]
FieldState: B_R[z], B_phi[v], B_Z[r]                       # 只推进 B
EvolutionState: ion,electron,field; time_s, step, generation, descriptor
FieldCache: phi[c], EL[edge], ET[edge], E[edge], solve_results, StageTag
CommittedState: EvolutionState owned; FieldCache owned; accepted=true
StageTag: step, stage={initial,base,midpoint,endpoint,restart,alias}, time_s,
  state_generation, window_id,iteration_id,trial_id,spectral_context_id
  # A0 window_id=iteration_id=0；trial_id 每次试算唯一；alias 不能冒作主阶段
RHS: 与 EvolutionState 相同的数组，单位为对应状态单位/s；不含 E/phi
StageWorkspace: padded primitive arrays, gradients, tensors, r0, wall_factors,
  matrix/FFT workspace, candidate buffers; 不能别名 committed 的可写存储
StageResult: immutable tag,RHS,cache,samples,fluid,nonE,terms,sources,ledger,
  boundary_ledger,field_derivatives,current_response,momentum_frame,
  checks,raw_residuals,alias_tag,stats
  # owns all of these or an arena containing them; pointers fixed only after stable allocation
```

`n` m^-3，`mom` kg m^-2 s^-1，`p` Pa，B T，E V/m，phi V。`u=mom/(m*n)` 是临时派生，压力直接存储。stage 内 `n` 决定的响应保持固定；不能在一个 LU 组装的列之间改变 `n` 或壁反力因子。

伪代码短名 `tag.time/tag.generation` 分别表示 `time_s/state_generation`；`U.B`、`RHS.mom_i/mom_e` 分别映射 field 的三数组和 ion/electron.mom，不是额外存储。`ctx` 之外的算子、常数、预算须由显式传入的运行上下文借用；不得实现成可变全局状态。StageResult 移动时保持 arena 地址稳定，source 的 response 闭包不捕获将失效的局部变量。

## A0-TYPE-03 — 唯一源与账

```text
FluidStageSource:
  population_id, tag, source_version, producer_kind="fluid"
  rho=q*n; number_flux=I*(mom/m); current=q*number_flux
  number_divergence=DJ*number_flux
  raw_second_moment[a,b]=m*n*u[a]*u[b]+P[a,b]    # Pa, sampled nodes
  boundary_number_flux[wall,sample]; boundary_number_flux_integral; response_action_ref
StageSources:
  tag,geometry_hash,operator_hash,spectral_layout
  populations: sorted map {ion.id -> FluidStageSource, electron.id -> ...}
  total_rho,total_current                       # 中央聚合一次
IntervalSources:
  t_left,t_right,window_id,iteration_id,trial_id,accepted
  quadrature=[{stage_source_id,weight_seconds}]
  per_species {rho_left,rho_right,integrated_number_flux,integrated_current,
               boundary_charge_integral}
  # integrated_current A*s/m^2；除 dt 才是平均电流
StageLedger:
  tag,rates {electric_power_node,electric_power_edge,pressure_power_kinetic,
    pressure_power_internal,magnetic_power,wall_power,convection_pairing},
  internal_rates {ion_internal_rate | electron_perp_rate,electron_parallel_rate},
  spectral_transfer_defect {kinetic,internal_components},raw_power_arrays,
  owned context/nonE/finished_force/RHS references in StageResult arena
MomentumFrame: P_cartesian[3],Lz,outward_momentum_flux[3],outward_torque_z,
  numerical_wall_reaction[3],numerical_wall_torque_z,quadrature_N,reconstruction
CumulativeLedger: last_interval_id; typed sums of accepted interval increments;
  immutable original baseline {start_time,step,initial_divB,topology,energy,momentum}
DiagnosticFrame: accepted_state_identity,totals,momentum,stats,owning raw_residuals
AcceptedResult: accepted,step,time,records,accepted IntervalSources,DiagnosticFrame
```

所有 map 必须与登记的两个物种集合完全一致，ID 不相同；禁止同一群另以 electron/total 再登记。源键 `(population_id,role,tag,source_version)` 唯一。`response_action_ref` 仅借用当前阶段不可变上下文，不能保存到下一阶段/restart；函数闭包按显式上下文翻译，不需要大型虚类框架。

## A0-TYPE-04 — 结果、预算与资源

```text
Result<T> = T | Failure
Failure: code,module,step_id,tag,field,population,entity,mode,sample,
  raw_value,unit,reference,budget,library_status,last_committed_step,
  partial_results,reason
Budget: atol>=0,rtol>=0,reference>0,unit,norm={weighted_L2,linf,value},
  gate={hard,observe}; finite all; 指定检查不得缺预算
DiagnosticRecord:
  unique_key=(run,step,stage,generation,trial,iteration,spectral_context,
              name,population,region,norm,check_scope)
  quantity_unit,value_unit,location,geometry_hash,operator_hash,
  time_s,interval_optional {id,t_start,t_end,quadrature,source_refs},
  raw_value,reference_scale,atol,rtol,status,gate,worst_index,reason
status={PASS,FAIL,WARN,NOT_RUN,UNAVAILABLE}
LinearSolveResult: solution,raw_residual,scaled_residual,constraint_residual,
  factor_info,solve_info,rank,singular_values,dimensions,scales,backend,tag
Runtime: backend; device_id; stream; fft/blas/solver handles; MemoryBudget;
  no copy; destroy after all arrays and pending jobs finish
```

时间/阶段检验是标签全字段比较（必要的 stage 转换由 time 显式创建），不是比较浮点时间是否“差不多”。所有必需 hard 检查必须 PASS；缺数据不能用 0 代替。CUDA 析构不抛异常，正常路径必须先显式同步并检查；异常清理记录第二错误而不覆盖第一错误。host/device 声明、失败分类见 [backend](../../src/cuda/backend.cu.md)。
