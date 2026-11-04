# 稠密线性代数与数值秩

> 状态：active；目标 `src/numerics/dense.cpp`。依据主规格 §6.2、GC6/10/15。  
> 依赖：[类型](../../include/fomhd/types.hpp.md)、[backend](../cuda/backend.cu.md)。同步 operators/fields/wall、构建与测试；最近同步：2026-11-03。

## A0-LA-01 — MaterializeAndScale

矩阵列主序 `A[row+col*lda]`，`lda=max(1,nrow)`。全部库尺寸先检查能转为 LP64 的 int；原矩阵、缩放矩阵、LU、右端、SVD 临时副本和 workspace 同时计入峰值内存。

```text
Materialize(action,nout,nin):
  CheckSizeAndAllocate(nout,nin)
  for j in [0,nin):
    e=zeros_complex(nin); e[j]=1
    y=TRY action(e)                       # 允许非 Hermitian，保留复值
    require y.size==nout and finite(y)
    A[:,j]=y
  return A

Scale(A,b,row_scale,col_scale):
  require all scales positive finite, lengths match rows/columns
  for r,c: As[r,c]=A[r,c]*col_scale[c]/row_scale[r]
  for r: bs[r]=b[r]/row_scale[r]
  require finite(As,bs); return As,bs
```

缩放是求解内部的单位处理，不修改物理数组、残差或模型常数。Poisson 列参考 `E_ref*L_ref`，行参考 `rho_ref/epsilon0`；横向系统见 fields；P 图矩阵无量纲；壁反力见 wall。原方程残差必须用原 A/b 重算。

## A0-LA-02 — CheckedLUFactor / CheckedLUSolve

CPU 基线绑定已存在的 MKL LAPACKE LP64 sequential：`mkl_lapacke.h`，`lapack_complex_double/MKL_Complex16`，不假设与 std::complex 二进制相同。GPU 绑定 cuSOLVER，见 backend。

```text
CheckedLUFactor(As,backend):
  require square, finite, n>0
  LU=deep_copy(As); allocate ipiv[n] in library integer type
  CPU: info=LAPACKE_zgetrf(LAPACK_COL_MAJOR,n,n,LU,lda,ipiv)
  CUDA: query workspace; cusolverDnZgetrf; copy info and synchronize
  require library status success, info==0, finite(LU)
  # info>0 是零主元；这里立即返回 Singular，绝不继续 getrs
  return owning factors {LU,ipiv,n,backend,As_reference,valid=true}

CheckedLUSolve(factors,bs):
  require factors.valid; bs rows==n; finite(bs)
  x=deep_copy(bs)
  CPU: info=LAPACKE_zgetrs(COL_MAJOR,'N',n,nrhs,LU,lda,ipiv,x,n)
  CUDA: cusolverDnZgetrs; copy info and synchronize
  require status success, info==0, finite(x)
  return x

SolveLinear(A,b,row_scale,col_scale,rank_rtol,residual_rtol):
  require square finite A, finite b, matching shapes
  require 0<rank_rtol<1 and positive finite residual_rtol
  As,bs=Scale(A,b,...); keep A,b,As,bs unchanged
  S=TRY SingularValues(As)
  require S.size==As.columns and S[0]>0 and S[-1]>rank_rtol*S[0]
  factors=TRY CheckedLUFactor(As)          # factor exactly once in this solve
  y=TRY CheckedLUSolve(factors,bs)
  steps=0; previous_error=infinity
  loop:
    require finite(y)
    sr=backend.MatVec(As,y)-bs; require finite(sr)
    scaled_residual=Norm(sr)
    error=max_r(abs(sr[r])/(abs(bs[r])+sum_j(abs(As[r,j])*abs(y[j]))))
      # Accumulate denominators in host long double as before.
      # For a zero denominator use abs(sr[r]); never add an absolute floor.
    require finite(error)
    if error<=residual_rtol: break
    require steps<5 and error<previous_error else reject
    previous_error=error
    delta=TRY CheckedLUSolve(factors,-sr)  # same factors and pivots; FP64
    next_y=backend.Axpy(y,delta,1)
    require finite(delta,next_y) and next_y!=y else reject
    y=next_y; steps+=1
  x=col_scale*y; require finite(x)
  residual=backend.MatVec(A,x)-b          # original equations, not LU storage
  require finite(residual,scaled_residual,error)
  return LinearResult(x,residual,S,scaled_residual,error,scales,steps)

LinearResult.refinement_steps:
  dimensionless host-owned count of completed residual corrections, in [0,5]
  zero when the first solution passes; no unsuccessful result is returned
  belongs to the same stage as its owning field cache
  accepted snapshots serialize this count next to each backward_error

```

主元为 LAPACK 的 1-based 索引，只传回同一库；若显示给用户则明确转换，不能减一后再交给 getrs。固定 geometry 分解可复用；阶段 n 变化后的 S 和 AE 系统不得复用旧 LU。批量右端 `nrhs` 的每列都是独立 RHS。

## A0-LA-03 — SingularValues / IndependentRows

```text
SingularValues(A):
  if min(m,n)==0: return empty singular_values
  X=deep_copy(A)
  if m<n: X=adj(X); swap m,n               # 统一满足 CUDA gesvd 的 m>=n
  CPU:
    allocate singular_values[n], rwork[max(1,5*n)]
    use LAPACKE_zgesvd_work(COL_MAJOR,'N','N',m,n,X,m,S,
      dummy_U,1,dummy_VT,1,query_work,lwork=-1,rwork)
    require query info==0 and finite positive workspace size within cap
    allocate work[ceil(real(query_work[0]))]
    restore X if needed; call same API with actual lwork
  CUDA: cusolverDnZgesvd_bufferSize/getrs-independent gesvd adapter
  require status success and info==0
  require S finite,nonnegative,nonincreasing within rounding bound
  return S

IndependentRows(B_full,rank_rtol):
  require 0<rank_rtol<1
  selected=[]; rank=0
  for r in original row order:
    candidate=stack(B_full[selected],B_full[r])
    normalize every nonzero candidate row by its Euclidean norm
    if row exactly zero: record dropped zero row; continue
    S=TRY SingularValues(candidate)
    new_rank=count(s>rank_rtol*S[0] for s in S)
    require new_rank in {rank,rank+1}, RankAmbiguous
    record row_id,S,threshold
    if new_rank==rank+1: append r; rank=new_rank
  return B_full[selected],selected,full_row_map,all_spectra
```

阈值是输入的数值秩判据，不是证明任意近相关行在数学上冗余。选出的子系统解必须回到完整原始约束验残差；不相容就失败，不靠删行让残差通过。`SVD info>0` 是未收敛，不能拿部分奇异值判秩。全域 `DJ` 理论行满秩，但仍记录实际结果；约束乘子只保留选中的行。

## A0-LA-04 — AdjointAndNormLeaves

```text
Adjoint(A)[r,c]=conj(A[c,r])
WeightedDot(x,y,w)=sum_j w[j]*conj(x[j])*y[j] in fixed order
WeightedNorm(x,w)=sqrt(max(0,real(WeightedDot(x,x,w))))
SpectralNorm(hat,w)=sqrt(2*pi*sum_mode,entity(w*abs(hat)^2))
SampleNorm(x,w)=sqrt((2*pi/N)*sum_sample,entity(w*abs(x)^2))
```

若内积出现超过舍入界的负自积/非有限值，先失败；`max(0,...)` 只处理已验明的负零/末位舍入。L∞ 为有效实体和样本/系数指定空间的最大模，不含求积权；两者 budget 单位分别记录。CPU 固定循环，GPU 首版 host 归约可作确定性参考，不声称跨设备逐位相等。
