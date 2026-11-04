# 兼容复形、插值与伴随

> 状态：active；目标 `src/numerics/operators.cpp`。依据 GC1–GC9、GC11。  
> 依赖：[mesh](../geometry/mesh.cpp.md)、[dense](dense.cpp.md)。同步 fluid/fields/wall/diagnostics；最近同步：2026-11-03。

## A0-OP-01 — LowOrderMatrices

输入只读 host geometry；构造 complex column-major 小矩阵并按 k 缓存。零初始化后只加以下条目；共享项用 `+=`，绝不覆盖相邻贡献。

```text
for c(i,j):
  b2r[c,r(i+1,j)]+=1; b2r[c,r(i,j)]-=1
  b2z[c,z(i,j+1)]+=1; b2z[c,z(i,j)]-=1
for r(e,j): b1r[r,v(e,j+1)]+=1; b1r[r,v(e,j)]-=1
for z(i,f): b1z[z,v(i,f)]+=1; b1z[z,v(i+1,f)]-=1
require b2r*b1r+b2z*b1z exactly integer zero
gR=-diag(1/lr)*adj(b2r); gZ=-diag(1/lz)*adj(b2z)

G2(k,f): return (gR*f, i*k*f/Rc, gZ*f)
C2(k,E):
  BR=(i*k*EZ-gZ*(Rc*Ephi))/Rz
  Bphi=-(adj(b1r)*(lr*ER)+adj(b1z)*(lz*EZ))/av
  BZ=(gR*(Rc*Ephi)-i*k*ER)/Rbar_r
  return BR,Bphi,BZ
DB2(k,B):
  return (adj(b1z)*(lz*Rz*BR)-adj(b1r)*(lr*Rbar_r*BZ)
          +i*k*av*Bphi)/wv
```

`*` 中向量同形连乘明确为逐点；矩阵乘用上式的矩阵位置判定。r/z/v 边界实体都保留；零切向 E/法向 B 迹已消元，不能再删壁行列。`G2` 的标量属于零 Dirichlet 势空间，不能用于一般流体的常量梯度。

## A0-OP-02 — GraphFiltersAndTransforms

```text
GraphP(entity_set,axis):
  P=identity(nentity)
  for p in entity_ID_order:
    q=lookup(key[p]+unit_integer_step(axis))
    if q absent or boundary[p] or boundary[q]: continue
    P[p,p]+=1/24; P[q,q]+=1/24
    P[p,q]-=1/24; P[q,p]-=1/24
  factor=CheckedLU(P,dimensionless_scales); preserve original P
  return P,factor
build PR_r,PZ_z,PR_v,PZ_v

T1(x): (PR_r*xR,xphi,PZ_z*xZ)
T1inv(x): (solve(PR_r,xR),xphi,solve(PZ_z,xZ))
T2(x): (PZ_z*xR,PR_v*(PZ_v*xphi),(PR_r*(Rbar_r*xZ))/Rbar_r)
T2inv(x): (solve(PZ_z,xR),solve(PZ_v,solve(PR_v,xphi)),
           solve(PR_r,Rbar_r*xZ)/Rbar_r)
T3(x): PR_v*(PZ_v*(Rbar_v*x))/Rbar_v
T3inv(x): solve(PZ_v,solve(PR_v,Rbar_v*x))/Rbar_v
```

注意 inverse 反序，mask 上 PR/PZ 不可交换。变换作用/solve 不修改输入，固定因子跨模复用。所有 inverse solve 检查原矩阵残差，不能固定迭代若干次代替。

```text
for k in [-K,K]:
  G[k]=Materialize(f -> T1(G2(k,f)))
  C[k]=Materialize(e -> T2(C2(k,T1inv(e))))
  DB[k]=Materialize(b -> T3(DB2(k,T2inv(b))))
  DJ[k]=-diag(1/M0)*adj(G[k])*diag(M1)
  Cdag[k]=diag(1/M1)*adj(C[k])*diag(MB)
  Lphi[k]=-DJ[k]*G[k]
  keep ALL Nc charge rows
  cache checked LU of scaled Lphi[k]; no constant-mode deletion for k=0
```

`Materialize` 每次一个复单位列，见 dense。M0/M1/MB 对同一位置实对角；复矩阵必须用 adj，不用普通 transpose。若总稠密存储超限立即失败，首版不静默换迭代解。

保存 `DR=DJ[0][:,Rblock]`、`DZ=DJ[0][:,Zblock]`，二者为实矩阵；完整采样环向散度直接 `Dphi/Rc`，不需要给 padded 全波数重复存矩阵。

## A0-OP-03 — FixedTransfers

```text
for r(e,j):
  if internal:
    four=[cell(e-2,j),cell(e-1,j),cell(e,j),cell(e+1,j)]
    if all four active and path contiguous: I_R[row,four]=[-1,9,9,-1]/16
    else: I_R[row,[cell(e-1,j),cell(e,j)]]=[1/2,1/2]
  else:
    I_R[row,[wall.cell_near,wall.cell_next]]=[3/2,-1/2]
for z(i,f): same rule in Z with cells f-2,f-1,f,f+1
I_phi=identity(Nc)
I=block_diag(I_R,I_phi,I_Z)                  # vector nodes -> edge
MN=concat(M0,M0,M0)
H=diag(1/MN)*adj(I)*diag(M1)                # edge -> vector nodes
wall_T_full=empty rows
for wall in deterministic (direction,entity_id) order:
  row=zero(3*Nc)
  row[normal_component node block]=sigma*I_direction[wall.face,:]
  append row; remember original wall ID
wall_T,selected_rows=IndependentRows(wall_T_full,rank_rtol)
```

三个支撑点也只用临面两点，不发明三点公式。固定 `I/H` 不随物种密度变化；`V(n)` 与 wall projection 才随阶段变化。T 检查用完整行，求解使用独立行。

```text
GatherBToNodes matrix Q_B:
  for c(i,j):
    BR: try z(i,j-1),z(i,j),z(i,j+1),z(i,j+2) weights [-1,9,9,-1]/16
    BZ: try r(i-1,j),r(i,j),r(i+1,j),r(i+2,j) same weights
    Bphi: try v(e,f) e=i-1..i+2,f=j-1..j+2, tensor-product weights
    use high-order stencil only if every supporting entity and every
      intervening dual cell in its rectangular support is active/connected
    otherwise BR=mean(z(i,j),z(i,j+1)); BZ=mean(r(i,j),r(i+1,j))
      Bphi=mean(v(i,j),v(i+1,j),v(i,j+1),v(i+1,j+1))
```

三个分量分别选择支撑，不从域外填 B。高阶转移不对 u/B 再做隐式滤波。定义 `Q_E=H` 用于诊断的节点 E；不替换场能的 M1/MB 积分。

## A0-OP-04 — OperatorAcceptanceAndTopology

```text
for k:
  record normalized norms C[k]*G[k] and DB[k]*C[k] separately
  record DJ[k]+diag(1/M0)*adj(G[k])*diag(M1)
  compare each against declared algebra budgets and original equation residuals
for deterministic x_node,y_edge:
  compare inner_MN(x,H*y) with inner_M1(I*x,y)
for deterministic j_edge at k=0:
  lhs=sum(M0*(DJ[0]*j))
  rhs=sum_wall(sigma*area_per_radian*j[wall.normal_edge])
  compare lhs,rhs (do not derive rhs from lhs)
ell2=concat(zeros(Nz),av,zeros(Nr))
ell=solve(adj(T2),conj(ell2)) conjugate-transposed back into row
  # 等价 ell*x=ell2*T2inv(x)；实际实现优先这个 action，勿显式 inverse
record ell*C[0]; require within budget
Bc_full=block_diag(DJ[k] for k=-K..K)
Bc,selected_constraint_rows=IndependentRows(Bc_full,config.solver.rank_rtol)
transverse_constraints=own(Bc_full,Bc,selected_constraint_rows)
operator_rule_hash=SHA256(rule_versions,layout,K,declared coefficients,geometry_hash)
matrix_fingerprint=SHA256(canonical stream of geometry IDs,shapes,weights,
  I,H,T_full,selected wall/constraint row IDs,PR/PZ,G,C,DB,DJ,Cdag,Lphi,Q_B,ell)
operator_hash=matrix_fingerprint
```

规则 hash 与实装矩阵 hash 分开保存：前者用于语义兼容，后者包含平台舍入实际字节，跨设备不能保证相同。canonical stream 按上述固定字段顺序写 UTF-8 名称长度/名称、little-endian uint64 维数/形状、列主序 binary64 实部/虚部；禁止非有限值，零值规范为正零。哈希消费原始未分解矩阵，不含 LU 工作区或阶段 n 相关响应；算法版本改变需更改 rule_versions。run manifest 与每份 checkpoint 均保存实际指纹，不仅保存规则名。`ell*B(k=0)` 与裸 `sum(av*Bphi)` 同时输出；前者是所选代数闭链，后者是物理求积估计。初态 Bn 为被消元的零迹，不把有限差分散度错误自动清零。
