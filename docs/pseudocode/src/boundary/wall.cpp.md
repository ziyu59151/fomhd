# 壁约束、反力与边界账

> 状态：active；目标 `src/boundary/wall.cpp`。依据 GC9/10/13–15 与固定封闭参考。  
> 依赖：[算子](../numerics/operators.cpp.md)、[线性代数](../numerics/dense.cpp.md)。同步 fluid/fields/diagnostics；最近同步：2026-11-03。

## A0-WALL-01 — BuildWallContext

```text
BuildWallContext(n_samples,m):
  require n valid; T=selected wall_T; keep T_full for checks
  for sample l:
    for component a,cell c: winv[a,c]=m*n[l,c]/M0[c]
    S= T*diag(winv)*adj(T)
    require finite S; SingularValues(scaled S) meets configured rank budget
    factor=CheckedLUFactor(scaled S)
    store factors,original S,winv,n generation,tag
  return immutable context
```

`W=diag(V/(m*n))` 的逆权在同一个物理采样点计算；不能对 Fourier 系数逐模除密度。小原型 N 个壁系统可以 host 分解；CUDA 后端若这样执行，拷贝与同步须显式记账，不能隐匿回退。首版统一后端矩阵 API 在指定设备分解，几何 T 的秩选择保持 host。

壁系统单位缩放：`force_ref=m*n_ref*u_ref/t_ref`，`s_ref=m*n_ref/(L_ref^3)`，`lambda_ref=force_ref/s_ref`；row_scale=`force_ref`、col_scale=`lambda_ref`。它们都是显式输入参考尺度派生，不用实际为零的力作分母。

## A0-WALL-02 — ProjectMomentumRHS

```text
ProjectMomentumRHS(ctx,r_samples):
  require ctx.tag/state_generation matches frozen stage coefficients
  for sample l:
    rhs=T*r[l]
    lambda=CheckedLUSolve(ctx.factor[l],rhs/row_scale)*col_scale
    reaction=-winv*adj(T)*lambda
    projected=r+reaction
    check S*lambda-rhs with original S
    check T_full*projected on original full wall rows
    preserve lambda,reaction,original_rhs,residuals
  return projected,reaction,lambda
```

允许 complex r，例如 AE 的单位向量响应。空 T 时为恒等映射（测试 fixture 可用）；固定 LCFS 必须有壁并在几何阶段核实。不要把 lambda 当作另一个需要推进的壁压力。输入 pi 若违背 Tpi=0 则拒绝，不调用此函数修剪状态。壁反力在允许速度上的功由 `inner_MN(u,reaction)` 实测。

## A0-WALL-03 — ActualBoundaryTracesAndFluxes

签名 `ActualBoundaryTracesAndFluxes(Bhat,samples,fluid,nonE,cache,operators,tag)`；输入均只读且同阶段。先由 cache 和 Bhat 恢复 E/B 的 N 点交错样本，再执行以下循环。输出拥有全部壁迹、逐壁通量及积分；不只返回总和。

```text
for each wall segment b, sample l:
  row=I_direction[b.face,:]; normal=sigma*unit(direction)
  for species:
    nb=row*n; require nb>n_face_min
    ub[a]=(row*pi[a])/(m*nb) for a=R,phi,Z
    Fn=sigma*F[direction][b.face]           # 实际生产的通量；不直接填 0
    un=dot(normal,ub); require Fn agrees with nb*un
    pb or pperpb,pparallelb=row*pressure
    if normal direction==R:
      Bb=(0,mean(Bphi at two endpoints),BZ[b.rface])
      ELb=(EL_R[b.rface],0,0); ETb=(ET_R[b.rface],0,0)
    if normal direction==Z:
      Bb=(BR[b.zface],mean(Bphi at two endpoints),0)
      ELb=(0,0,EL_Z[b.zface]); ETb=(0,0,ET_Z[b.zface])
    record Bn/Et as eliminated traces with provenance="prescribed relative complex"
    require norm(Bb)>B_min if electron wall tensor consumed
    bwall=Bb/norm(Bb)
    Pwall=pb*identity (ion), or pperpb*I+(pparallelb-pperpb)*bwall*bwall^T
    Kb=m*nb*|ub|^2/2; Ub=pb/(gamma-1) or pperpb+pparallelb/2
    material_energy_flux=(Kb+Ub)*un+dot(ub,Pwall*normal)
    pressure_transport_flux=sigma*nonE[species].pressure_fluxes[direction][b.face]
    compare it with (row*p)/nb*Fn           # 每个压力分别记，消费实际输运通量
    particle_number_flux=Fn
    momentum_flux=m*nb*ub*un+Pwall*normal
  Eb=ELb+ETb
  phi_dot_boundary=0                       # time-independent eliminated potential trace
  field_energy_flux=dot(cross(Eb,Bb)/mu0-epsilon0*phi_dot_boundary*ETb,normal)
  multiply each flux by wall.area_per_radian*2*pi/N and sum in wall ID order
return BoundaryLedger{tag,traces[wall,sample],per_species:
  {number_flux,kinetic_advective_flux=integral_wall(Kb*un),
   internal_transport_flux=integral_wall(pressure_transport_flux/(gamma-1)) if ion,
   perp_internal_transport_flux=integral_wall(perp_pressure_transport_flux) if electron,
   parallel_internal_transport_flux=integral_wall(parallel_pressure_transport_flux/2) if electron},
  field_energy_flux=integral_wall(field_energy_flux),
  total_energy_flux=integral_wall(sum_species(material_energy_flux)+field_energy_flux)}
```

从已消元零迹重建的 `Bn/Et` 是结构事实，不能冒充另一份独立壁测量。独立检查包括原 incidence 链、CT 增量和实际未消元壁反力残量。面法向 E 和切向 B 从实际自由场取得；零功率由这些量与实际 un 计算，不在账中硬编码 0。

边界牵引/数值反力分别保存。体节点反力积分先变成同一 Cartesian 基再累加，采用 diagnostics 的至少三点环向求积（包括 K=0/N=1 特例）；不因零能流宣称零动量交换。若墙面压力外推非正，记录原值与位置，并按配置的 wall_pressure gate 判定，不能隐式改正压力。

## A0-WALL-04 — ImmutableBoundary

每次阶段核对 geometry/operator/boundary hashes；场边界只有同一个常值势（规范 0）、ETt=0、初始 Bn=0。无独立 n/p/En/Bt/Jn 外部边值。旧 `ApplyIonFluidBoundary`、`ApplyElectronBoundary` 对应本文件的合法性检查和完整 RHS 消元；`FillFieldGhosts` 无独立物理未知量。首版矩阵作用不需要实体外 ghost；若输出调试 halo，只输出从已定义矩阵支撑重建的值，禁止回灌 RHS。

测试 `WALL-01..04` 覆盖 Tpi/Tpidot、壁零功、所有力统一投影、完整行秩复查与非法厚度几何。
