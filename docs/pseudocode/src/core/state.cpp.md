# 状态装载、有效域与组合

> 状态：active；目标 `src/core/state.cpp`。依据 A0S1、主规格 §3/4/8。  
> 依赖：[类型](../../include/fomhd/types.hpp.md)、[Fourier](../numerics/fourier.cpp.md)、[NPZ](../io/npz.cpp.md)。同步初态、time、restart、tests；最近同步：2026-11-03。

## A0-STATE-01 — CheckSizeAndAllocate

```text
CheckedBytes(shape,element_bytes,budget):
  count=1
  for d in shape:
    require(d>0 and count<=SIZE_MAX/d, BadShape)
    count*=d
  require(count<=SIZE_MAX/element_bytes, SizeOverflow)
  bytes=count*element_bytes
  require(bytes<=budget.remaining, MemoryLimit)
  return bytes

AllocateState(index,layout,backend,budget):
  enumerate 5 ion + 6 electron scalar cell arrays; B arrays sizes Nz,Nv,Nr
  reserve simultaneously-live committed,midpoint,candidate,RHS and caches
  include padded copies, full matrices, LU copies, workspace, IO staging
  require total checked estimate <= configured host/device caps
  allocate each owned array; zero initialize; on allocation error release owned buffers
  assign descriptors and return state with no valid physical stage yet
```

`MemoryBudget` 记峰值活跃分配，线性库 query 返回的空间也计入。零初始化只是防未初始化读取，不表示合法物理状态；正域校验前禁止用其求速度。无可写别名通过拥有者范围重叠检查，不只比较对象名。

## A0-STATE-02 — LoadA0State

```text
LoadA0State(npz,index,layout,config):
  decode exact required keys and on-disk shapes via ReadNPZ
  for each array and mode:
    for dense_slot in rectangular on-disk shape:
      if entity inactive: require value==0 (real and imaginary); skip
      else copy to [mode][active_id] using active_to_dense
  assign SI and global-phase descriptors from explicit schema
  require no legacy phi_B key, no pressure invariant / total-energy state
  assign time=start_time,step=0,generation=NewGeneration()
  TRY CheckRetained(state)
  return owned EvolutionState
```

NPZ v1 固定使用 `mom_i,mom_e`。外部 provider 若给速度，输入生成工具先在共同 padded 网格计算 `m*n*u`、forward 截谱并记录误差，输出本 schema；加载器不猜测 mom 是否其实是速度。支持导入速度的另一个 schema 须显式版本化，不能对原始 Fourier 系数逐模相乘。

## A0-STATE-03 — CheckRetained / RecoverSamples

```text
CheckRetained(U):
  require descriptors,geometry,layout,shape,space match expected
  for every active coefficient: require finite(re) and finite(im)
  for every scalar component: CheckHermitian(coefficients,budgets.hermitian)
  for species: check full_wall_T * mom at every retained mode, original full rows
  return checks

RecoverSamples(U,N):
  TRY CheckRetained(U)
  for all fluid arrays: x=InverseRetained(xhat,N,physical_real=true)
  BNhat=GatherBToNodes(U.B)                  # 固定线性转移
  BN=InverseRetained(BNhat,N,physical_real=true)
  for l,c:
    require ion.n>n_min_i, electron.n>n_min_e
    require pi_pressure>p_min_i, pe_perp>p_min_perp, pe_parallel>p_min_parallel
    Bmag=hypot(BN_R,BN_phi,BN_Z)
    require finite(Bmag) and Bmag>B_min
    for species a:
      u_a=pi_a/(m_a*n_a); require all finite
    b=BN/Bmag
  for species and each direction:
    nf=I_direction*n
    require finite(nf) and nf>n_face_min_a at ALL consumer edges, including wall
  return samples,primitive; 所有除法在相应正域检查之后
```

三压力、密度与 B 的预算是算例输入，不在这里发明下限。压力插值不作为密度分母，但非有限值必须拒绝；壁压力重构负值另记诊断，不能裁剪。padded 有效域与保留模共轭均要检查，谱系数正不表示物理密度正。

## A0-STATE-04 — CombineEvolution

```text
CombineEvolution(base,RHS,weight_s,target_tag,out):
  require descriptors compatible except declared target_tag
  require RHS.tag is stage expected by caller; require weight_s finite
  require out has no writable alias of base or RHS
  for each evolutionary component, mode, active entity:
    out = base + weight_s*RHS              # 不组合 E/phi/cache
  out.generation=target_tag.state_generation; out.time=target_tag.time_s
  out.step=target_tag.step
  set every out QuantityDescriptor.stage_tag=target_tag
  invalidate out.field_cache
  TRY CheckRetained(out)
  return out
```

`NewGeneration` 单调计数溢出即失败；试算代次可以跳号，不表示接受步增加。所有失败保留 base 的字节与累计账。内存交换提交见 [midpoint](../time/midpoint.cpp.md)，restart 不信任文件中的派生电场缓存。

## A0-STATE-05 — BindStageView

```text
BindStageView(U,tag):
  require U.step==tag.step and U.time==tag.time_s and U.generation==tag.state_generation
  make read-only array views sharing U owners; copy only descriptors
  set copied descriptors.stage_tag=tag; retain original owner generation and hashes
  return read-only stage view; never mutate U or its endpoint/initial cache
```

base 对同一个已提交状态重新求值时用此显式标签绑定，不要求把历史 endpoint 标签原地改成 base。试算数组标签由 CombineEvolution 写全。

## A0-STATE-06 — CreateCpuRuntime

```text
CreateCpuRuntime(config):
  require config.solver.cpu_threads==1       # 首版 FFTW/MKL sequential，没有未实现的线程开关
  initialize checked host MemoryBudget from config.memory_limit_bytes
  runtime.backend=CPU; stream/device handles absent
  runtime.host_fft=FFTW3 complex; runtime.host_linalg=MKL LAPACKE LP64 sequential
  record fftw_version,MKL_Get_Version_String,sizeof(lapack_int)==4
  initialize monotonic generation/trial/source counters; overflow -> Failure
  return owning Runtime; FFT plans/factors allocated lazily with counted workspace
```

CUDA runtime 同样先建立此 host 部分，供几何与静态约束选行；再附加独立的 device handles/budget。CPU 和 CUDA 分支均显式单 host 线程，不改系统线程设置；将来增加并行库后再版本化线程配置。
