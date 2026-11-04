# Darwin 场约束与 CT

> 状态：active；目标 `src/fields/darwin.cpp`。依据 A0F0、GC14/15、AC1/5。  
> 依赖：[fluid](../fluid/fluid.cpp.md)、[dense](../numerics/dense.cpp.md)、[Fourier](../numerics/fourier.cpp.md)。同步 stage/time/diagnostics；最近同步：2026-11-03。

## A0-FIELD-01 — SolveElectrostaticDarwin

```text
SolveElectrostaticDarwin(rho_hat,tag):
  require source tag,units,geometry,layout match
  for k in [-K,K]:
    rhs=rho_hat[k]/epsilon0
    phi[k]=solve using cached scaled LU of Lphi[k]=-DJ[k]*G[k]
    EL[k]=-G[k]*phi[k]
    check Lphi[k]*phi[k]-rhs and DJ[k]*EL[k]-rhs, ALL Nc rows
  check Hermitian(phi,EL) for physical call
  return phi,EL,solve_results

LongitudinalProject(x_hat):
  for k: psi[k]=solve(Lphi[k],DJ[k]*x[k]); result[k]=-G[k]*psi[k]
  return result
TransverseProject(x)=x-LongitudinalProject(x)
```

投影是 Helmholtz 分量/诊断，不修改传入电流。k=0 不删势点；Dirichlet 允许非零净体电荷。周期盒测试使用单独零均值 fixture，先检查净电荷相容，不减平均电荷伪造输入。

## A0-FIELD-02 — FrozenCurrentResponse

```text
BuildCurrentResponse(fluid_contexts,non_electric_terms):
  fJ0=zeros(all_retained_edges)
  for species a exactly once:
    r0_projected=ProjectMomentumRHS(ctx_a.wall,r0_a).projected
    dJ0_samples=(q_a/m_a)*I*r0_projected
    fJ0+=ForwardRetained(dJ0_samples)

  AE_action(Ehat):
    output=zeros(all_retained_edges)
    for species a exactly once:
      piE=ElectricMomentumAction(ctx_a,Ehat)    # 采样复值，冻结 n/wall
      output+=ForwardRetained((q_a/m_a)*I*piE)
    return output
  return fJ0,AE_action with borrowed immutable stage context
```

完整模式编号 `global_col=mode_index*Ne+edge_id`，edge 顺序 `(r,c,z)`。不采用只保留 k>=0 的压缩未知量，因为这里的完整复线性组装需要任意复列；实际物理解最后检查 Hermitian。AE 使用保留模→N 点→线性作用→保留模的 Galerkin 限制，复单位列不取 real。

## A0-FIELD-03 — SolveTransverseDarwin

```text
Bc_full,Bc,selected_rows=operators.transverse_constraints
  # 由 A0-OP-04 静态初始化一次并纳入指纹；选行只改变乘子表示
M1_all=repeat M1 for each retained mode

SolveTransverseDarwin(fJ0,AE_action,EL,tag):
  AE=Materialize(AE_action,M*Ne,M*Ne)
  curl_term=block_diag(adj(C[k])*diag(MB)*C[k]/mu0)
  A= curl_term + diag(M1_all)*AE
  check adjoint defect of diag(M1_all)*AE using algebra budget
  KKT=[[A,adj(Bc)],[Bc,zeros(nconstraint,nconstraint)]]
  btop=-M1_all*(fJ0+AE_action(EL)); b=concat(btop,zeros(nconstraint))
  E_column_scale=E_ref
  lambda_column_scale=(J_ref/t_ref)*L_ref^4
  first_row_scale[edge]=M1_all[edge]*(J_ref/t_ref)
  constraint_row_scale=E_ref/L_ref
  result=SolveLinear(KKT,b,scales,block_budgets,config.solver.rank_rtol,original_KKT_action)
  ET=result.solution[:M*Ne]; lambda=result.solution[M*Ne:]
  check Bc_full*ET (including dropped rows)
  original_dynamic=A*ET+adj(Bc)*lambda-btop
  check both dynamic and full divergence constraints with their own units
  require Hermitian(ET); E=EL+ET
  return ET,E,lambda,result
```

第一行单位为 A m/s（M1 乘电流变化率），约束行 V/m²；乘子单位 A m²/s。报告物理强形式时先除 M1，得到 A m^-2 s^-1。不可把两块原始数值混成无单位的单残差。

首版始终全保留模联合组装；环向对称也可以用这个正确但较贵的路径。没有实现逐模优化开关。系统是不定 saddle-point，LU 不假设 SPD。rank/memory/factor/residual 任何失败不返回假电场，也不覆盖 B。

## A0-FIELD-04 — MagneticRHSAndConsistency

```text
BuildMagneticRHSCT(ET):
  for k: Bdot[k]=-C[k]*ET[k]
  check DB[k]*Bdot[k] with CT-increment algebra budget
  return Bdot

CheckCurrentDerivative(actual_momdot_i,actual_momdot_e,fJ0,AE_action,E):
  Jdot_actual=(qi/mi)*I*actual_momdot_i+(qe/me)*I*actual_momdot_e
  Jdot_response=fJ0+AE_action(E)
  check Jdot_actual-Jdot_response
  return owning CurrentResponseCheck{Jdot_actual,Jdot_response,
    residual=Jdot_actual-Jdot_response,records,tag=E.tag}

FieldDiagnosticDerivatives(J,B):
  ELdot=-LongitudinalProject(J)/epsilon0
  phidot=solve(Lphi,-DJ*J/epsilon0) on all retained modes
  Ampere=Cdag*B-mu0*TransverseProject(J)
  return ELdot,phidot,Ampere
```

`ELdot` 用于阶段物理交换/约束诊断，不另独立推进 E。有限步用端点 EL 差与实际中点 J 检查纵向连续性。CT 保持初始 DBB 和闭链通量，不能声称将任意错误初值清到零。初始 transverse Ampere 也是必须验收的约束，解 E 本身不能修复它。对应测试 `FIELD-01..05`。
