# CUDA 执行与资源合同

> 状态：active；目标 `src/cuda/backend.cu`，对应 CPU 逻辑在 numerics/fluid。CUDA execution and Compute Sanitizer memcheck/initcheck were verified under the user Slurm allocation; see the current V&V record.  
> 依赖：[类型](../../include/fomhd/types.hpp.md)、[Fourier](../numerics/fourier.cpp.md)、[dense](../numerics/dense.cpp.md)。同步 CMake/测试/diagnostics；最近同步：2026-11-03。

## A0-GPU-01 — RuntimeAndBuffers

```text
CreateCudaRuntime(config):
  host=TRY CreateCpuRuntime(config)
  require invoked under user's valid Slurm GPU allocation on this host
  query runtime/driver/device count using public CUDA API; record versions
  require requested ordinal exists within CUDA_VISIBLE_DEVICES mapping
  cudaSetDevice(ordinal); cudaGetDeviceProperties; check FP64 support
  cudaStreamCreateWithFlags(nonblocking)
  cublasCreate; cublasSetStream; cublasSetPointerMode(HOST)
  cusolverDnCreate; cusolverDnSetStream
  memory budget <= explicit cap and available allocation; no whole-host GPU probing
  return owning Runtime(host,backend=CUDA,device handles/budget)

AllocateDevice<T>(count): checked bytes; cudaMalloc; status check; register owner
Upload(host_owned,device_owned): cudaMemcpyAsync(H2D,stream); retain host until event done
Download(device_owned,host_owned): cudaMemcpyAsync(D2H,stream); synchronize before host read
Fence(): cudaStreamSynchronize(stream); failure invalidates dependent results
```

普通 host vector 可用于同步 copy；异步需要生命周期保障。首版为了清晰，跨 CPU/GPU 阶段边界直接 Fence，允许必要 host staging，不实现多 stream overlap/统一内存迁移优化。只有 successful Fence 后才回收/复用缓冲。错误保留第一个 CUDA status、调用、阶段和分配规模；不可偷偷把 CUDA 错误改为 CPU 成功。

## A0-GPU-02 — FFT

```text
cufftCreate(handle)
cufftSetAutoAllocation(handle,0)
n={N}; inembed={N}; onembed={N}             # 非空！启用显式 advanced layout
cufftMakePlanMany(handle,1,n,inembed,E,1,onembed,E,1,CUFFT_Z2Z,E,&work_bytes)
check queried bytes against memory budget; cudaMalloc(work_bytes)
cufftSetWorkArea(handle,work); cufftSetStream(handle,stream)
cufftExecZ2Z(handle,input,output,CUFFT_FORWARD or CUFFT_INVERSE)
check cufftResult; launch scale(output,1/N) only after forward
Fence before CPU reads or numerical decisions
```

FP64 用 `CUFFT_Z2Z`，不是单精度 `CUFFT_C2C`。`inembed/onembed=nullptr` 不用于本布局，避免 advanced stride 被忽略。输入每个分量 `[N,E]`；bin stride E、batch distance 1；双向都不归一化，逆向不再除 N。计划/工作区必须活到最后一次 transform 完成。

## A0-GPU-03 — BLASAndLU

```text
MatVec(A[m,n],x,y,op):
  if op==CUBLAS_OP_N: require length(x)==n and length(y)==m
  else if op in {CUBLAS_OP_C,CUBLAS_OP_T}: require length(x)==m and length(y)==n
  else: fail UnsupportedOperation
  require lda>=max(1,m)                    # m,n 始终描述未转置的原矩阵
  cublasZgemv(handle,op,m,n,&one,A,lda,x,1,&zero,y,1)
  op normal=N,adjoint=C (不是 T); check return
MatMul(A,B): cublasZgemm with explicit op,m,n,k,lda,ldb,ldc,one,zero

DenseLUFactor(A[n,n]):
  copy original into device LU
  cusolverDnZgetrf_bufferSize(handle,n,n,LU,n,&lwork)
  allocate lwork complex elements,ipiv[n] int,devInfo[1]
  cusolverDnZgetrf(handle,n,n,LU,n,work,ipiv,devInfo)
  download info,check factors finite using finite kernel; Fence
  require status success and info==0 else RETURN Failure before getrs
DenseLUSolve(factors,B[n,nrhs]):
  cusolverDnZgetrs(handle,CUBLAS_OP_N,n,nrhs,LU,n,ipiv,B,n,devInfo)
  download info; Fence; require info==0 and finite(B)
  check original residual via MatVec/MatMul; do not use LU as original A

CheckedLinearCuda(A,b,row_scale,col_scale,rank,tolerance):
  validate inputs/scales; preserve original and scaled matrices and RHS
  S=SingularValuesCuda(As); apply the unchanged relative rank threshold
  factors=DenseLUFactor(As); retain device factors,pivots,info through refinement
  y=DenseLUSolve(factors,bs)
  repeat the bounded SolveLinear loop in dense.cpp.md with at most 5 corrections:
    compute sr=MatVec(As,y)-bs using cuBLAS on the CUDA backend
    evaluate the existing componentwise backward error on the synchronized host
    stop on passing the unchanged tolerance; reject nonfinite/stagnating errors
    upload -sr, solve it with the SAME cuSOLVER LU and pivots, download delta
    update y with CUDA Axpy; reject nonfinite corrections or unchanged iterates
  rescale x, compute original residual through CUDA MatVec(A,x)-b, validate
  release device factors after their last correction; return refinement_steps
  never call the CPU linear solver or refactor to hide a CUDA solve failure
```

主元 int 为库内部 LAPACK 基数 1，原样传回；stride/lda 为元素数。LU work 长度是 complex 元素数，而 cufft 的 work_size 是字节，不能混用。源状态/原 A 不被原地分解。

## A0-GPU-04 — SVD

```text
SingularValuesCuda(A[m,n]):
  if m<n: X=adj(A); swap m,n else X=copy(A)
  query cusolverDnZgesvd_bufferSize(handle,m,n,&lwork)
  allocate work[lwork] complex,rwork[max(1,min(m,n)-1)] double,S[min(m,n)]
  allocate safe one-element dummy U,VT; ldu=ldvt=1; devInfo[1]
  cusolverDnZgesvd(handle,'N','N',m,n,X,m,S,dummyU,1,dummyVT,1,
                   work,lwork,rwork,devInfo)
  download info,S; Fence; require status success,info==0,finite nonnegative S
```

此 rwork 约定属于 cuSOLVER，不可复制到 CPU LAPACK zgesvd，后者至少 `5*min(m,n)` 个 double。几何/静态约束行选择首版在 host CPU 执行以便复现；这里仍定义完整 device 适配用于阶段系统需要的秩检查。所有库版本与该 API 约定记录在构建清单，实际 GPU 测试未完成前不标通过。

## A0-GPU-05 — ImplementedKernels

CUDA kernel 只翻译已定义的逐点算术或固定矩阵；矩阵运算优先 BLAS。没有 atomics/scatter 求和依赖。每个线程唯一写一个输出槽，线程越界直接 return。

```text
LinearIndexKernel(count,body):
  i=uint64(blockIdx.x)*blockDim.x+threadIdx.x
  if i<count: body(i)
launch blocks=ceil(count/256),threads=256,checked grid limit
after launch: cudaPeekAtLastError; before using validity decision Fence

kernels:
  axpy: out[i]=base[i]+alpha*rhs[i]
  pad: one thread per [N,E], derive k for bin; if abs(k)<=K copy hat[k+K,e] else 0
  gather: one thread per [M,E], read bins[bin(k,N),e]/N
  derivative: one thread per [N,E], multiply full bin by i*signed_k/N
  host: primitive recovery, fluid source assembly, weighting, finite checks,
        diagnostic reduction and matrix transpose/adjoint use synchronized host arrays
```

Native reference: domain checks run on synchronized host data before reciprocals or normalization. Fluid source assembly remains on the host; the CUDA backend accelerates the named numerical operations. No device fluid-source or validation kernel is claimed. Warp migration will separately implement device-resident source assembly and verify equivalence.

销毁顺序：Fence→释放阶段 arrays/workspace→销毁 FFT/BLAS/SOLVER handles→销毁 stream。测试在个人 Slurm 作业内执行 Compute Sanitizer memcheck/initcheck；只有采用共享内存或自定义同步后才对相应 kernel 加 racecheck/synccheck。运行失败日志不转成“未测试成功”。
