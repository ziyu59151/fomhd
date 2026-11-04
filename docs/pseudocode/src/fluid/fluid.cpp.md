# 两流体、柱坐标梯度与 RHS

> 状态：active；目标 `src/fluid/fluid.cpp`。依据 GC9–GC14、A0D1/2。  
> 依赖：[Fourier](../numerics/fourier.cpp.md)、[算子](../numerics/operators.cpp.md)、[wall](../boundary/wall.cpp.md)。同步 fields/stage/diagnostics；最近同步：2026-11-03。

## A0-FLUID-01 — PrimitiveAndFlux

输入同阶段 `U`，已通过 state 的有限值、正域和共轭检查；每个物种有自己的 n/m/q，两个物种共享 BN/b 和几何。输出数组归 StageWorkspace；source 中的 F 是不可变共享视图。

```text
BuildFluidContext(species,U,samples,operators):
  n,pi,u,pressures=samples of this species
  f=pi/m
  Fhat=I*(U.mom_hat/m)                     # 仅此生产一次；利用线性操作
  F=InverseRetained(Fhat,N,true)
  divFhat=DJ*Fhat                         # 密度 RHS 使用这个原值
  divF=DivSamples(F)
  require ForwardRetained(divF) agrees with divFhat within arithmetic budget
  for alpha in R,phi,Z: nf[alpha]=I_alpha*n; check nf before division
  for l,c,a,b:
    if ion: P[a,b]=p_i*(a==b)
    if electron: P[a,b]=p_perp*(a==b)+(p_parallel-p_perp)*bhat[a]*bhat[b]
    raw_moment[a,b]=m*n*u[a]*u[b]+P[a,b]
  wall=BuildWallContext(n,m)
  ndot_hat=-divFhat
  for wall b,sample l:
    boundary_number_flux[b,l]=sigma_b*F[direction_b][l,face_b]
  boundary_number_flux_integral=sum_b,l(area_b*2*pi/N*boundary_number_flux[b,l])
  return fixed context {n,pi,u,f,F,Fhat,divF,divFhat,ndot_hat,nf,P,raw_moment,
    wall,tag,boundary_number_flux,boundary_number_flux_integral}
```

`bhat` 这里是采样磁场单位矢量，不是 Fourier 帽标。压力张量存全 9 分量；惯性矩不能被热压力替代。磁力 `cross(u,BN)` 用 `(R,phi,Z)` 右手正交基：`(u_phi*B_Z-u_Z*B_phi, u_Z*B_R-u_R*B_Z, u_R*B_phi-u_phi*B_R)`。

## A0-FLUID-02 — VelocityGradient

固定阶段 n；任意输入向量 x 可为复数（用于伴随/响应核验），几何和 n 不变。以下矩阵仅作用 RZ 索引，Dphi 使用全部采样谱。

```text
VR(x)= (I_R*(n*x))/nf_R
VZ(x)= (I_Z*(n*x))/nf_Z
KR(x)= DR*VR(x)-x/Rc
KZ(x)= DZ*VZ(x)
Kphi(x)=DphiFull(x)/Rc

VelocityGradient(x):
  for a in [R,phi,Z]:
    Q[a,R]=KR(x[a]); Q[a,phi]=Kphi(x[a]); Q[a,Z]=KZ(x[a])
  Q[R,phi]-=x[phi]/Rc
  Q[phi,phi]+=x[R]/Rc
  return Q

KR_adj(y)=n*(adj(I_R)*((adj(DR)*y)/nf_R))-y/Rc
KZ_adj(y)=n*(adj(I_Z)*((adj(DZ)*y)/nf_Z))
Kphi_adj(y)=-DphiFull(y/Rc)

VelocityGradientWeightedAdjoint(P):
  for a,b: Q[a,b]=M0*P[a,b]               # same sample weights cancel
  for a: v[a]=KR_adj(Q[a,R])+Kphi_adj(Q[a,phi])+KZ_adj(Q[a,Z])
  v[R]+=Q[phi,phi]/Rc
  v[phi]-=Q[R,phi]/Rc
  for a: v[a]/=M0
  return v
```

这里 `adj` 是实际矩阵转置/共轭转置；不能用对 `1/n` 形式求导猜压力力。`theta=trace(Q)`，`strain=sum_ab bhat[a]*bhat[b]*Q[a,b]`。单独核查 `theta=DivSamples(V_R u_R,u_phi,V_Z u_Z)`。GC12 已含柱连接项，不再加旧柱坐标 g。

## A0-FLUID-03 — NonElectricAndPressureRHS

```text
BuildNonElectricRHS(ctx,BN,bhat):
  Q=VelocityGradient(u)
  theta=Q[R,R]+Q[phi,phi]+Q[Z,Z]
  strain=sum_ab(bhat[a]*bhat[b]*Q[a,b])
  for a,b: tensor_uf[a,b]=u[a]*f[b]
  adv_adj=VelocityGradientWeightedAdjoint(tensor_uf)
  for a:
    grad_times_f[a]=sum_b Q[a,b]*f[b]
    A[a]=(m/2)*(-adv_adj[a]+grad_times_f[a]+u[a]*divF)
  pressure_force=VelocityGradientWeightedAdjoint(P)
  magnetic_force=q*n*cross(u,BN)
  r0=-A+pressure_force+magnetic_force

  PressureTransport(p):
    for alpha: flux_p[alpha]=(I_alpha*p)/nf[alpha]*F[alpha]
    return DivSamples(flux_p),flux_p
  if ion:
    Tp,fp=PressureTransport(p_i)
    pdot_i=-Tp-(gamma_i-1)*p_i*theta
  if electron:
    Tperp,fperp=PressureTransport(p_perp)
    Tparallel,fparallel=PressureTransport(p_parallel)
    pdot_perp=-Tperp+p_perp*(strain-theta)
    pdot_parallel=-Tparallel-2*p_parallel*strain
  return NonElectricTerms{r0,A,pressure_force,magnetic_force,Q,theta,strain,
    pressure_derivatives,pressure_fluxes,tag=ctx.tag}
```

没有碰撞/热流/滤波/压力裁剪，离子动量不重复加电子压力。两物种通过共同场响应相互作用。电子内能源为 `pdot_perp_local+pdot_parallel_local/2=-sum(P*Q)`；离子内能源 `pdot_i_local/(gamma_i-1)=-p_i*theta`。动量中的压力项正号是梯度的加权伴随，对应连续负压力散度。

## A0-FLUID-04 — ElectricResponseAndFinish

```text
ElectricMomentumAction(ctx,Ehat):
  E_samples=InverseRetained(Ehat,N,physical_real=false)
  force=q*n*(H*E_samples)
  projected=ProjectMomentumRHS(ctx.wall,force).projected
  return projected                       # complex, sampled; 不取 real、不截谱

FinishFluidRHS(ctx,nonE,Ehat):
  require nonE.tag==ctx.tag
  E=InverseRetained(Ehat,N,true)
  electric_force=q*n*(H*E)
  total_raw=nonE.r0+electric_force
  pidot,wall_reaction,lambda=ProjectMomentumRHS(ctx.wall,total_raw)
  for momentum component: momdot_hat=ForwardRetained(pidot)
  for pressure component: pdot_hat=ForwardRetained(nonE.pressure_derivatives)
  ndot_hat=ctx.ndot_hat                    # 不重新构造密度通量
  CheckHermitian(all rhs); CheckFinite(all rhs)
  assign all RHS descriptors units=state_unit/s,stage_tag=ctx.tag
  check wall_T_full*momdot_hat, original wall multiplier equations
  return RHS, FinishedForceTerms{electric_force,pidot,ndot=-ctx.divF,
    wall_reaction,lambda,tag=ctx.tag}
```

电响应与最终动量都使用同一个壁投影，后者包含全部力，不仅电力。`ProjectMomentumRHS` 对固定 n 是线性的，因此 GC14 可从同一函数建立；不对 n、P(n) 或速度梯度另作时间求导。

## A0-FLUID-05 — StagePowerLedger

```text
StagePowerLedger(ctx,nonE,terms,RHS,cache):
  require all tags agree; E=InverseRetained(cache.E,N,true)
  record cell/sample arrays:
    electric_node=ctx.u dot terms.electric_force
    electric_projected=ctx.u dot ProjectMomentumRHS(ctx.wall,terms.electric_force).projected
    pressure_to_kinetic=ctx.u dot nonE.pressure_force
    pressure_to_internal=-sum_ab(ctx.P[a,b]*nonE.Q[a,b])
    magnetic=ctx.u dot nonE.magnetic_force
    wall=ctx.u dot terms.wall_reaction
    convection=ctx.u dot nonE.A-(m/2)*|ctx.u|^2*ctx.divF
  edge_electric=q*ctx.F dot E
  integrate node arrays using M0*2*pi/N,edge arrays using M1*2*pi/N
  rates={electric_power_node,electric_power_edge,pressure_power_kinetic,
    pressure_power_internal,magnetic_power,wall_power,convection_pairing}
  ion_internal_rate=integral_M0(-p_i*nonE.theta) if ion
  electron_perp_rate=integral_M0(p_perp*(nonE.strain-nonE.theta)) if electron
  electron_parallel_rate=integral_M0(-p_parallel*nonE.strain) if electron
  ndot_back=InverseRetained(RHS.n,N,true)
  pidot_back=InverseRetained(RHS.mom,N,true)
  Kdot_retained=ctx.u dot pidot_back-(m/2)*|ctx.u|^2*ndot_back
  Kdot_sample=ctx.u dot terms.pidot-(m/2)*|ctx.u|^2*terms.ndot
  Udot_retained=InverseRetained(RHS.p,N,true)/(gamma-1) if ion
    or separate InverseRetained(RHS.p_perp,N,true),InverseRetained(RHS.p_parallel,N,true)/2
  Udot_sample=nonE.pressure_derivatives with the same internal-energy factors
  spectral_transfer_defect=integral_M0(retained_rate-sample_rate),separate K and U
  return StageLedger(tag,rates,internal_rates,spectral_transfer_defect,
    owning references to ctx,nonE,terms,RHS and all raw power arrays)
```

三因子和非多项式会造成截谱功率差额；不把该差额填入“物理耗散”或声称完整算法严格保能。场能交换、状态能量与整步误差归 diagnostics。对应测试 `FLUID-01..05`。
