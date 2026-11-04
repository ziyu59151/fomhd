# 环向 Fourier 适配

> 状态：active；目标 `src/numerics/fourier.cpp`。依据 A0S2、主规格 §5/6.2、GC11–15。  
> 依赖：[类型](../../include/fomhd/types.hpp.md)、[CUDA](../cuda/backend.cu.md)。同步全部非线性模块、NPZ、tests；最近同步：2026-11-03。

## A0-FFT-01 — PlansAndNormalization

标量工作区为 `complex[bin_or_sample][entity]`，offset=`l*E+c`。不同分量可用不同 E，逐分量建 plan，不混拼不同实体数。所有采样位于 `phi_l=2*pi*l/N`；外部偏移相位在 provider 一次转换。

```text
CreatePlans(N,E,backend):
  require N odd and N>=4*K+1; dimensions/products fit library integer and budget
  allocate separate input/output complex arrays [N,E]
  CPU:
    fftw_alloc_complex(N*E); FFTW_ESTIMATE (不让 planning 改输入)
    fftw_plan_many_dft(rank=1,n={N},howmany=E,
      in,inembed={N},istride=E,idist=1,
      out,onembed={N},ostride=E,odist=1,sign,FFTW_ESTIMATE)
    require plan!=nullptr for forward(-1),backward(+1)
  CUDA: same logical strides with cufftMakePlanMany, CUFFT_Z2Z; see backend
  record forward_scale=1/N; inverse_scale=1
  return RAII plan and owned workspace
```

stride 单位是一个复数，不是字节。只有合适 alignment 的缓冲才能复用 FFTW new-array execution；首版每 plan 绑定其分配的 work arrays，不将任意 std::vector 指针替进去。plan/cache 键包含 backend、device、N、E、strides、stream。K=0 时可选 N=1；实现必须允许一元素 Fourier 变换且导数为零。

## A0-FFT-02 — InverseRetained / ForwardRetained

```text
bin(k,N)=((k % N)+N)%N                     # signed calculation then checked cast
InverseRetained(hat[M,E],N,physical_real):
  fill bins[N,E] with complex zero
  for s in [0,M),c in [0,E): bins[bin(s-K,N),c]=hat[s,c]
  samples=ExecuteBackward(bins)            # 不缩放
  if physical_real:
    require Hermitian(hat) and max imaginary sample within imag budget
    return real(samples)                  # 检查成功才丢虚部
  return complex samples                  # 线性试向量、矩阵列必须走此分支

ForwardRetained(samples[N,E]):
  full=ExecuteForward(samples)/N
  for s,c: hat[s,c]=full[bin(s-K,N),c]
  return hat                              # 不强制 Hermitian，不对称化“修复”

CheckHermitian(hat):
  check imag(hat[K,:]) against budget
  for k in [1,K]: check hat[K-k,:]-conj(hat[K+k,:])
  report worst mode/entity and raw defects; never overwrite the two modes
```

真实状态及最终 RHS 另由调用者检查 Hermitian；一般复线性作用不能用这个要求拒绝非实试向量。映射外部采样原点 `delta_phi` 时，若 provider 的 FFT 系数基于 `phi_l-delta_phi`，全局系数为该系数乘 `exp(-i*k*delta_phi)`，记录转换约定并以已知正/负单模测试，不凭文件名猜原点。

## A0-FFT-03 — SampleDerivativeAndAdjoint

```text
DphiFull(samples[N,E]):
  full=ExecuteForward(samples)/N
  for bin_index in [0,N):
    k=bin_index if bin_index<=(N-1)/2 else bin_index-N
    for c: full[bin_index,c]*=i*k
  return ExecuteBackward(full)             # 使用全部 N 个 bin，不截回 ±K
DphiFullAdjoint(samples)=-DphiFull(samples)
DivSamples(F): return DR*FR + DphiFull(Fphi)/Rc + DZ*FZ
DivSamplesAdjoint(y):
  return adj(DR)*y, -DphiFull(y/Rc), adj(DZ)*y
```

这是普通 Euclidean adjoint；带物理权的伴随由消费者显式前乘/后除 M。R 与 phi 不相关，仍按写出的逆序运算。奇数 N 没有偶数 Nyquist 特例。恢复 u/b、压力张量、梯度/伴随和所有乘积始终在同一个 N 点空间，最终 RHS 才截谱。

## A0-FFT-04 — AliasProbe

```text
AliasProbe(U,main_stage_result):
  N2=checked(2*N+1)
  create separate context with same K,geometry,constants,time and new spectral_context_id
  alternate=EvaluateStage(U,alias_tag,context_N2,enable_alias_probe=false)
  for each evolutionary component:
    difference=alternate.RHS_hat-main_stage_result.RHS_hat
    compare retained weighted norms with budgets.alias[component]
  preserve both norms/fields where configured; no alternate data enters evolution
  return {records=alternate.checks+comparison records,tag=alias_tag}
    # distinct spectral_context/stage keys; comparison records use main tag with alias.* names
```

初始化、每个中点和重启执行该硬门；端点沿用普通阶段检查，下个中点仍重验。若要改变采样频率，需显式修改诊断配置和验收范围，不默默关掉非多项式误差门。测试 `FFT-01..04`：直接 DFT、非末轴多实体、单负模、全采样反伴随及 N/2N+1 非多项式对照。
