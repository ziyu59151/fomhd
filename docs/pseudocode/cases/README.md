# A0 算例与输入生成配方

> 状态：active；目标 `cases/`。配方不是已经验收的装置初态，也不是运行结果。  
> 依赖：[schema](../src/io/config.cpp.md)、[tests](../tests/README.md)、[算子](../src/numerics/operators.cpp.md)。最近同步：2026-11-03。

## A0-CASE-01 — DiscreteClosedReference

最小静止参考使用合法且壁支撑足够的极向矩形/平滑轮廓 mask；R>0、拓扑圆盘。输入显式指定网格、物种质量/电荷、三个压力、磁场幅值和预算，不把本配方的符号当成可运行数值。

```text
given positive constant ni,qi and qe<0: ne=-(qi/qe)*ni
mom_i=mom_e=0; p_i>0 constant; p_perp_e=p_parallel_e>0 constant
only k=0 nonzero
construct geometry/operators and ell per GC7/8
B_harmonic=diag(1/MB)*adj(ell) in B space
  # ell*C=0 -> Cdag*B=0；k=0 的该闭链仅在 Bphi 上，DB*B=0
rescale B_harmonic to requested nonzero reference magnetic amplitude
require nodal and wall |B| meets magnetization domain; else reject fixture
export fields using rectangular active mapping; preserve recipe and operator rule version
check full initialized RHS,Gauss,DB,Ampere,Tpi before labeling equilibrium
```

这是明确依赖离散算子的代数静止 fixture，不能用来证明空间精度或等同真实托卡马克平衡；精度另用解析 MMS。外部初值不被求解器自动投影成这个状态。

## A0-CASE-02 — NonEquilibriumSmoke

```text
start from DiscreteClosedReference density/pressure/B
choose K>=1 and explicit velocity amplitude U0,radial/vertical smooth bump f(R,Z)
u_i=u_e=(0,U0*f(R,Z)*cos(phi),0)
mom_i=mi*ni*u_i; mom_e=me*ne*u_e
# 两物种电流相消：J=0；纯 phi 流对 R/Z 壁法向迹为零
export retained spectrum exactly k=±1 and constants
set initial_kind=non_equilibrium; keep all constraint gates
run finite small nsteps with chosen dt and positive-domain budgets
```

非零流产生真实非平衡演化，可检查多阶段/场/RHS/IO 是否联通；仍不代表特定物理波或真实 LCFS benchmark。振幅与 dt 是显式实验参数，未测之前不宣称稳定。

## A0-CASE-03 — VerificationAndPhysicsCases

每个 case 文件夹保存 `README`（模型/目标/参考/预算/状态）、`case.toml`、小输入或可再生生成配方、`source_manifest.json`。大型平衡数据在运行输入/内容存储，不提交文献或大量结果。

三类验收分别登记：

- 解析/MMS/局部周期盒：验证算子、色散支、阶次和极化；周期 fixture 不是封闭 LCFS 生产输入。
- 固定阶梯边界的相容 RZ 初态：验证完整约束、边界与数值时窗；解析光滑 LCFS 到阶梯壁的误差单列。
- 真实装置/公开混合代码对比：需要同模型、剖面、物种和边界；A0 不含快粒子/动理学驱动，不能拿不同模型的增长率作一致性承诺。

具体装置数据和科学容差由每次研究输入提供；这不阻塞程序伪代码。A1 对比冻结同一几何/K/N/空间算子，先比较同物理误差，再记录成本。
