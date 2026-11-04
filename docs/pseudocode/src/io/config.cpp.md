# 配置 schema 与输入验证

> 状态：active；目标 `src/io/config.cpp`。依据主规格 §3/9.1，未发布的 A0 v1 schema 具体化。  
> 依赖：[类型](../../include/fomhd/types.hpp.md)、[NPZ](npz.cpp.md)、[构建](../../CMakeLists.txt.md)。同步 app/restart/cases/tests；最近同步：2026-11-03。

## A0-IO-01 — ConfigSchema

TOML 用 toml++ 3.4.0 的 `toml::parse` 消费已保存的原始输入字节并传入来源路径，捕获 `toml::parse_error` 并保留位置。项目只接受下表字段；所有科学参数均必填，没有示例数值充当默认物理输入。解析器负责 TOML 重复键和语法拒绝，代码逐表检查未知键、类型和范围。

| 表 | 字段、类型与限制 |
|---|---|
| 顶层 | `schema_version=1`、`model="A0"`、`units="SI"`、`layout_version=1`、`case_name` 非空安全文件名前缀 |
| `species` | `ion_id,electron_id` 非空不同字符串；`qi,mi,qe,me,gamma_i` double；mi/me>0、qi>0、qe<0、`gamma_i>1`；qe 是带符号电子电荷，精确数值与来源显式记录 |
| `solver.constants` | `epsilon0_Fpm,mu0_Hpm` 正有限 double；属于同一套 SI 常数来源，记录版本，不隐造无量纲值或设为 1 |
| `grid` | `R0,Z0,hR,hZ` double；`NR,NZ,K,Nphi` integer；hR/hZ>0，NR/NZ>=2，K>=0，Nphi 奇数且>=4K+1 |
| `time` | `start_time,dt` double；dt>0；`nsteps` uint64，0≤nsteps≤2^52；`initial_kind="equilibrium"` 或 `"non_equilibrium"` |
| `boundary` | `kind="fixed_lcfs_static"`、`potential=0.0`、`version="staircase_relative_v1"`；其余边界值不接受 |
| `solver` | `backend="cpu"/"cuda"`、`precision="fp64"`、`complex_precision="complex128"`、`memory_limit_bytes`、`device_memory_limit_bytes` 正整数、`rank_rtol` 在 (0,1)、`device_ordinal` 非负整数（CPU 要求 0）、`cpu_threads` 正整数 |
| `solver.references` | `L_m,t_s,n_i_m3,n_e_m3,u_mps,E_Vpm,B_T,J_Apm2,rho_Cpm3,p_Pa` 全部正有限 double；单位由键固定 |
| `budgets.domain` | `n_min_i,n_min_e,n_face_min_i,n_face_min_e,p_min_i,p_min_perp_e,p_min_parallel_e,B_min` 非负有限 double |
| `budgets.checks.<name>` | 该检查的 `atol,rtol,reference,unit,norm,gate`；所有 name 从下文注册表展开；绝对容差>=0，相对>=0，参考>0 |
| `io` | `geometry_file,initial_file,output_dir` 字符串；`snapshot_stride,checkpoint_stride` 正整数；`max_input_bytes,max_uncompressed_bytes,max_entries,max_header_bytes` 正整数；`source_manifest_file` 字符串；`write_residual_arrays` bool |
| `diagnostics` | `inner_distance_m` 正 double；`alias_policy="initial_midpoint_restart"`；`timing_repetitions` 正 integer；`timing_warmup` 非负 integer；`mode_observations` 数组，元素为 `{field,component,mode,entity_key=[i,j]}` |

`nsteps=0` 是初始化/IO smoke 的合法范围，不是演化 benchmark。CLI 的 `--output` 可以替换 io.output_dir，`--restart` 另提供 checkpoint；记录原始 TOML、覆盖项和规范化最终配置。没有物理参数的隐式环境变量覆盖。

## A0-IO-02 — BudgetRegistry

每个 vector/array 检查显式生成 `.<norm>` 两项（`weighted_L2`、`linf`）；每物种检查再生成 `.ion/.electron`，区域为 `.all/.inner/.wall`。初始化几何/计数类用 `value`。只要求适用的区域记录，若 inner 为空则该区域诊断为 observe/UNAVAILABLE，而不是空集合 PASS。

上述展开只适用于真正局部成立的数组残差。`electric_pairing/pressure_pairing/convection_pairing` 是全域加权内积之差，唯一硬门为 `region=all,norm=value` 的有符号标量；`gradient_adjoint/transfer_adjoint` 同样只比较完整空间的双线性内积。不得把其功率密度逐点、在 inner 或 wall 子区强制相消，子区功率仅 observe，未设计界面项就不注册子区守恒硬门。总能量/总数/动量等全局预算也只使用全域标量（动量三个分量分别 value）；实际壁通量数组与局部连续性残差才按相应实体区域展开。

```text
required hard checks:
  finite,hermitian,imaginary,grid_consistency,geometry_topology,positive_weights,
  CG,DBC,divergence_theorem,transfer_adjoint,gradient_adjoint,
  poisson_linear,gauss,transverse_linear,transverse_divergence,
  initial_divB,initial_ampere,CT_increment,CT_update,topology_increment,
  wall_state,wall_rhs,wall_multiplier,wall_number_flux,wall_pressure_flux,
  stage_continuity,current_response,electric_pairing,pressure_pairing,
  magnetic_work,wall_work,convection_pairing,theta_identity,cgl_source,
  species_continuity,total_number_balance,total_charge_balance,
  RC_flux,RC_mom,deltaC,RL,alias_each_evolution_component,
  source_identity,memory_bound
required records,gate explicitly chosen by case:
  current_divB,current_ampere,total_energy,ion_kinetic,ion_internal,
  electron_kinetic,electron_perp,electron_parallel,field_energy,
  spectral_transfer_defect,wall_pressure,wall_traction,momentum,angular_momentum,
  topology_vs_physical_flux,geometry_distance
equilibrium initial kind additionally requires hard checks for:
  equilibrium_n_i,mom_i,p_i,n_e,mom_e,p_perp_e,p_parallel_e,B
```

结构/类型/有限值/来源检查的真值门用固定布尔语义，不要求给一个无意义的数值 reference。其余每种检查必须有单位匹配的预算；省略就 `MissingBudget`。共同预算可由输入生成工具展开成显式表，运行时不默认套同一个 1e-12。线性块与物理观测预算分开；不能因 rhs=0 把尺度设为零。

## A0-IO-03 — LoadCase

```text
LoadCase(path,cli_overrides):
  read exact raw bytes with max configuration byte cap; hash and retain bytes
  table=toml::parse(string_view(raw_bytes),string_view(path))  # 不再次打开可能变化的原文件
  visit every table key recursively; reject unknown keys and wrong TOML node types
  read required keys with exact value<T>() checks; no implicit int-to-bool/string casts
  finite check EVERY floating field including all budget leaves
  validate schema/model/units/enums,charge/mass/steps/grid/precision and stride rules
  resolve input paths relative to case file directory; output path explicit
  load geometry headers and compare nodes to declared grid with grid_consistency budget
  expand applicable budget keys by population,field,region,norm registry
  check source_manifest includes provider name/version,units,conversion method,
    input hashes,continuous-residual status and optional curve approximation error
  compute checked dimensions and memory plan before array allocations
  resolved_json=canonical sorted-key representation of exact parsed config+overrides
  config_hash=SHA256(resolved_json bytes)
  semantic_hash=SHA256(same config excluding only output path,strides,nsteps,
    timing options and provenance display paths)
  return immutable config,raw input bytes,resolved config,hashes
```

semantic_hash 保留 dt、start_time、geometry/source 数据哈希、物种、K/N、边界、精度、预算、算法版本；restart 允许改变停止步和输出位置，不允许偷偷换物理、时间或诊断门。backend 切换默认不兼容；CPU/CUDA 对照单独从相同初态启动，不冒称逐位 restart。

## A0-IO-04 — InputErrorAndProvenance

解析失败在输出目录创建前向 stderr 返回带路径/行列的错误；不打印进程环境或凭据。外部 provider 的 UNVERIFIED 可作为来源状态，但不能免除本地 hard 初态验收。内容哈希消费原始文件字节；canonical config 哈希是另一个字段，两者不混用。

JSON 使用 nlohmann/json 3.12.0，序列化前逐值检查有限性；`dump` 不得把 NaN 自动转换为 null 后伪装有效记录。失败原值用 `"NaN"/"+Inf"/"-Inf"` 加类型标识，普通缺失值用 null+reason。JSON 重复 key 在 restart/manifest 读取时用 SAX 回调的对象键集合拒绝；不能默认接受后键覆盖前键。

本实现使用 `json::parse` 的事件回调维护对象键栈：object_start 压入空集合，key 插入失败即抛 DuplicateKey，object_end 弹栈；其余事件返回 true。它实现上段的重复键拒绝，不能仅返回 false 静默丢弃重复成员。TOML 来源路径显式转 `std::string_view`，避免 C 字符串与 string/string_view 重载歧义。

## A0-IO-05 — DiagnosticKeyRegistry

固定注册表将记录按生产时机分组。表中的逗号列表是完整键名集合；下列 `expand` 生成 DiagnosticRecord 的 tuple key，不从已收到的记录倒推“必需哪些”，否则漏算会被漏检。每项的量纲由 producer 的输出 descriptor 和该物理残差公式确定，并与配置预算逐项比对。

| 组 | 名称 | 生产者与形态 |
|---|---|---|
| setup | grid_consistency,geometry_topology,positive_weights,CG,DBC,divergence_theorem,transfer_adjoint | mesh/operators；几何布尔或全域算子/标量残差，initial/restart 时绑定接受 tag 后记录 |
| stage_bool | finite,source_identity,memory_bound | 状态/源/峰值内存的真值；每阶段，包括 alias；无浮点容差 |
| stage_domain | domain.n_i,domain.n_e,domain.n_face_i,domain.n_face_e,domain.p_i,domain.p_perp_e,domain.p_parallel_e,domain.B | RecoverSamples 与消费者实际极小值，使用 budgets.domain 严格下限；每阶段 hard |
| stage_array | `hermitian,imaginary,poisson_linear,gauss,transverse_linear,transverse_divergence,wall_state,wall_rhs,wall_multiplier,wall_number_flux,wall_pressure_flux,stage_continuity,current_response,theta_identity,cgl_source,current_divB,current_ampere,CT_increment` | CheckStage；按数组描述器的分量/物种/有效实体生成，current_divB/current_ampere 门由 case 指定，其余 hard |
| stage_pair | gradient_adjoint,electric_pairing,pressure_pairing,magnetic_work,wall_work,convection_pairing | 全域有符号内积标量 value；物种分开；hard |
| stage_observe | spectral_transfer_defect,wall_pressure,wall_traction | 前者为全域分项率；后两者为真实壁迹；case 指定 gate |
| initial | initial_divB,initial_ampere | CheckInitial；仅 initial/restart hard，不在普通阶段要求这两个名字 |
| initial_observe | topology_vs_physical_flux,geometry_distance | CheckInitial；全域/几何标量；case 指定 gate |
| equilibrium | equilibrium_n_i,mom_i,p_i,n_e,mom_e,p_perp_e,p_parallel_e,B | 名字统一加 equilibrium_ 前缀（不重复添加）；仅 kind=equilibrium hard |
| step_array | species_continuity,RC_flux,RC_mom,deltaC,RL,CT_update | CheckStep；按其定义的完整空间残差生成，hard |
| step_scalar | total_number_balance,total_charge_balance,topology_increment | CheckStep；全域/逐物种标量 value；hard |
| step_energy | total_energy,ion_kinetic,ion_internal,electron_kinetic,electron_perp,electron_parallel,field_energy.B,field_energy.L,field_energy.exchange | CheckStep 的分项和交换式；value，case 指定 gate |
| step_momentum | momentum.x,momentum.y,momentum.z,angular_momentum | CheckStep；value，case 指定 gate |
| alias | alias_each_evolution_component | 比较主/加密采样 RHS 的14个标量分量，各 weighted_L2/linf，全域 hard |

```text
expand(group,tag):
  for name in fixed group list:
    get name's fixed producer,quantity unit,location and applicable species/components
    scalar/pair/Boolean -> region=all,norm=value (no automatic regional identities)
    local-array -> both norms,valid all/inner/wall regions of its own entity space
    matrix/linear-block -> corresponding full operator block only,not spatial subregions
    no species -> population=none; otherwise actual registered IDs,never a total duplicate
    if an inner region is empty: require its configured gate==observe; emit UNAVAILABLE key
    append field/component suffix to name when checking multiple arrays/components
    generate key with exact tag and name; duplicate generated key -> RegistryError
    bind threshold from matching budget or fixed Boolean/domain semantics
  return expected key/producer/gate/shape metadata

required_stage_keys(t)=expand(stage_bool+stage_domain+stage_array+stage_pair+stage_observe,t)
required_stage_and_alias_keys(t,a)=required_stage_keys(t)+required_stage_keys(a)+expand(alias,t)
required_initial_keys(kind,t,a)=required_stage_and_alias_keys(t,a)
  +expand(setup+initial+initial_observe,t)+(expand(equilibrium,t) if kind==equilibrium else [])
required_step_keys(base,mid,alias,end,interval_id)=required_stage_keys(base)
  +required_stage_and_alias_keys(mid,alias)+required_stage_keys(end)
  +expand(step_array+step_scalar+step_energy+step_momentum,end with check_scope=interval_id)
```

`field_energy` 和 `momentum` 的预算按上表组件展开；布尔/正域门不能被 case 改成 observe。CheckStage 从已完成的 state/domain/solve/flux/power 操作收集对应记录；未执行者返回带原因的 NOT_RUN，Gate 拒绝 hard 缺项。静态 setup 成果保存在 context.setup_checks，CheckInitial 只绑定标签并收集一次；不得每个普通阶段重新要求初态平衡。非平衡时可额外输出 equilibrium UNAVAILABLE，不属于该运行 required hard 集合。
