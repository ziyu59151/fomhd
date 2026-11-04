# 独立测试与验收规格

> Status: active; target test responsibilities are implemented across the unit, integration, runtime, verification, and CUDA test executables. This table remains the contract; actual results are recorded in the dated evidence directory.  
> 依赖：[完整步骤映射](../TRACEABILITY.tsv)、[算例](../cases/README.md)、[构建](../CMakeLists.txt.md)。最近同步：2026-11-04。

## A0-TEST-01 — HarnessAndGates

```text
for each registered test ID:
  read explicit fixture constants,seed,reference scales,budgets
  run independent expected calculation before calling tested implementation
  require nonempty observations and nonzero reference where appropriate
  save input,expected,actual,norms,budget,status,library/build identity
  if negative control: require expected failure code or defect above same gate
  unexpected exception,missing test or zero test count -> fail process
return nonzero if any test fails
```

算术恒等式用 `epsilon*operation_scale*condition_estimate` 的登记舍入界；科学观测由 case 给定预算。测试不得“自动选一个刚好通过的容差”。几何、算法、文件、CPU/CUDA 的 scope 分列；不把一个 fixture 的 PASS 汇总成全部科学通过。

| ID | 输入与独立参考 | 必需断言 / 负对照 |
|---|---|---|
| GEO-01 | 小矩形、L 型但宽度足够的圆盘、反向顺序多边形；手列实体数/面邻居 | 排序唯一、两方向面积/半棱、同一 mask；自交/孔洞/角点接触拒绝 |
| GEO-02 | 人工 2×3 cells，每格 V/R 与角子矩形独立积分 | av/wv/Rbar 精确匹配；故意把 Rbar_r 替 Rr 的壁误差被检测 |
| GEO-03 | 两条相距已知距离的矩形折线 | Hausdorff 区间覆盖真值；采样端点/窄支撑错误拒绝 |
| GEO-04 | 非对称 mask 与磁盘 rectangular arrays | active↔dense 双向无遗漏；无效槽非零拒绝 |
| OP-01 | 任意复 G 输入和任意复 E（非梯度） | 分别检验 CG=0 和 DBC=0；翻转一条 incidence 负对照必须触发 |
| OP-02 | noncommuting PR/PZ 的不规则 mask | T2inv(T2(x))=x；交换 inverse 顺序的负对照触发 |
| OP-03 | k=0 任意棱通量、手算 wall 面通量 | 全域散度定理、I/H 伴随；壁积分独立累加 |
| OP-04 | 独立解析光滑紧支撑 scalar/vector，远内域 3 档 h | G/C/DB 的解析导数误差和阶次；壁/全域另报，不主张全域四阶 |
| FFT-01 | 逐点直接 DFT，奇数 N=1/5/9，多实体不同幅相 | 正负模、非末轴、1/N、roundtrip；错轴/负数%负对照 |
| FFT-02 | 已知全采样高模 sin/cos，复随机向量 | Dphi 全 N 正确、反伴随；提前截±K被检测 |
| FFT-03 | 单模及三因子带限多项式独立卷积 | 先 pad 后乘积；错缩放/错 padding 被检测 |
| FFT-04 | 正 n 的倒数与 b 归一化，N/2N+1/4N+3 | 保存保留 RHS 差，不宣称有限 padding 精确 |
| LA-01 | 非零已知 complex x、独立 A*x、强制 pivot | 解与原方程；奇异 LU 返回后禁止 getrs |
| LA-02 | unitary*diag(3,2)*unitary 的长/宽复矩阵 | SVD=(3,2)，work 空间、转置与 info；失败不继续判秩 |
| LA-03 | 原始重复/零/近相关约束行 | 稳定按编号选行，全原行复查；删错行不被“通过”掩盖 |
| WALL-01 | 正密度、T 满行秩、从独立 nullspace 得到 pi | Tpi/Tpidot、W 正交反力、零功；给非法 pi 必须拒绝 |
| WALL-02 | 压力、电力、磁力、对流各给非零法向分量 | 完整 RHS 投影；只投电力负对照被检测 |
| WALL-03 | wall 插值实际 F 与独立无穿透迹 | 每物种数/压力/能流、角点逐段；故意密度/通量错层触发 |
| WALL-04 | 两流体不同 n/m/q 和环向变化 | 不共用错物种 W，S 奇异/边缘密度失效拒绝 |
| FLUID-01 | 全 3×3 解析柱梯度与随机张量 | Gu/Gu† 加权伴随；轴连接符号/额外 g 负对照 |
| FLUID-02 | pperp=pparallel、零速度梯度、指定压缩/剪切 | 张量各向同性、离子/CGL 压缩符号、local Udot=-P:Gu |
| FLUID-03 | 允许速度空间、非均匀 n、非零流 | 对流与密度动能配对、压力/电功、磁力/壁零功 |
| FLUID-04 | 对每个 RHS 字段比较纯 sampled 与 final retained 回升 | 截谱差额独立记录，错将其设为算法改能失败 |
| FLUID-05 | 密度/压力/B 最坏点在 padded 物理样本 | 先检查再除法、首失败位置、无 clipping |
| FIELD-01 | 独立已知 phi 构造算子级 b，以及非零净电荷 | 所有 Nc Gauss 行、k0 正常解、原残差；删边界电荷行失败 |
| FIELD-02 | 复试向量 a,b 与变环向 n | AE(a+i*b)=AE(a)+i AE(b)，weighted 自伴；错误 real/逐模解被检测 |
| FIELD-03 | 任意约束允许 ET 和 lambda 构造 KKT b | 不定 LU、完整约束复查；失效/过内存拒绝 |
| FIELD-04 | 同一实际 pidot 与 GC14 | Jdot 完整一致，忽略 wall 的旧 AE 负对照 |
| FIELD-05 | 非零初始 DB 仅用于 CT 测试（不绕过生产初态门） | DB 增量/闭链保持，不覆盖 B；保留原 DB 错误 |
| TIME-01 | 小已知线性 ODE/同空间高精度解，3 档 dt | 中点二阶，E/cache 不当演化量，有限时窗增长实测 |
| TIME-02 | base/midpoint/endpoint 注入不同失败 | committed/cumulative 字节和来源 ID 不变；部分检查标 NOT_RUN |
| TIME-03 | 常量已知 ledger + 非常量阶段 ledger | 仅 dt*mid；base/end/alias/拒绝不累计，重复 interval 拒绝 |
| TIME-04 | 提交后模拟 IO error、signal 安全点 | accepted 与 durable checkpoint 区分，restart 不读半写输出 |
| IO-01 | 每个 dtype/shape、Unicode、正负模 C++↔NumPy | 精确布局/值，NPZ 无 pickle，错误 endian/object/形状拒绝 |
| IO-02 | 缺/重复/未知 TOML/JSON key、ZIP 重复条目/截断/展开超限 | 指定错误码，不分配越预算，不覆盖已有文件 |
| IO-03 | checkpoint 发布各步骤注入错误 | latest 指向完整可读文件，旧 checkpoint 不变 |
| IO-04 | 同初态持续 N 步 vs N1+restart+(N-N1) | 状态/约束/账在预算内一致；不兼容语义 hash 拒绝 |
| GPU-01 | 与 CPU 相同非空数组/几何/阶段 | 逐模块 state/source/field/ledger 比较，明确 FP64/设备 |
| GPU-02 | 全核心小案例，个人 Slurm 作业 | memcheck/initcheck 通过；GPU unavailable 是 NOT_RUN 非 PASS |
| DIAG-01 | 人造错误电荷符号、过期源、错壁法向/尺度 | 独立诊断确实触发，不依赖总电荷相消 |
| DIAG-02 | 非零初始能/动量与零信号两类 | 不除零，缺参考 UNAVAILABLE，hard 缺项拒绝 |
| DIAG-03 | K=0,Nphi=1、非零纯环向动量及环向无关壁迹 | Px/Py 和横向合力为零；Lz 为独立解析积分；旧单角度 Cartesian 求积负对照触发 |
| DIAG-04 | 局部功率密度不相消但全域伴随内积成立的向量 | 全域 pairing hard PASS，子区 observe 非零不拒绝；observe FAIL 保留，缺 required 记录仍拒绝 |
| IO-05 | validate、run nsteps=0、accepted initial/restart cache | 均可写 snapshot/checkpoint；只创建一次目录；cache generation 不匹配拒绝 |
| IO-06 | 原始矩阵系数或哈希被改、原初始漂移基线存在 | restart 指纹不匹配拒绝；累计账与原 baseline 保留，不以恢复点重置 |
| AUX-01 | 人工快照 f_h=f_exact+h^4*g、3档h，同物理权与时间 | 离线分析恢复四阶；零差/错时刻/混模型拒绝或 UNAVAILABLE |
| AUX-02 | 解析复观测 `A0 exp((gamma-i*omega)t)`，采样满足 `abs(omega*dt)<pi` | 拟合恢复 omega/gamma 及符号；零幅度/不够样本/未登记时窗不作有效拟合 |

## A0-TEST-02 — ManufacturedSolutionsAndIntegration

算子级 `b=A*x` 仅证明求解器/组装自洽，不能证明离散精度。独立制造解使用解析函数和解析微分；在测试 harness 添加已知制造源并单独记账，生产 A0 配置不开放任意外源。

```text
SpatialMMS:
  use analytic compact bump inside a fixed distance from wall times exp(i*k*phi)
  evaluate scalar,vector and cylindrical analytic derivatives at each true staggered point
  for h,h/2,h/4: compare operator result on common physical interior region
  report h-error pairs,observed orders,wall/all errors and distance mapping

TemporalMMS:
  at fixed geometry/K/N/solve budget choose smooth manufactured U(t)
  independent analytic Udot(t); test forcing=Udot_reference-Rh(U_reference)
  compare mid-step solver at dt,dt/2,dt/4 at exactly common output times
  separate time error from fixed-space forced algebra; do not call it unforced plasma validation

FullA0Integration:
  use admissible non-equilibrium fixture from cases
  run initialization -> multiple accepted steps -> snapshot -> checkpoint -> restart
  require all hard checks,positive domain,unique source/ledger and nonempty history
  measure actual finite-window error vs finer dt reference; freeze comparison budgets beforehand
```

原生实现每完成一个模块就接对应 tests；接口无正式实现时不得注册永远 PASS 的占位测试。A1 后续重用这些空间核与相同输入，再增加窗口/迭代/区间电流检查。

The detailed supplementary contracts are [coupled MMS](coupled_verification.cpp.md),
[staggered spatial MMS](spatial_verification.cpp.md),
[source-accounting sensitivity](source_controls.cpp.md),
[linear refinement](linear_refinement.cpp.md), [transactions](transaction.cpp.md),
and [local dispersion](dispersion.cpp.md). The latter compares all 14 response
components and constraint invariants, then the 11-dimensional admissible
spectrum; its periodic adapter is test-only. The current evidence and remaining
scientific scope are recorded in [A0 verification](../../A0_VERIFICATION.md).

AUX 测试属于 tools 的辅助 Python 测试标签，使用源码项目 uv 环境及人工快照；不把 Python 作为原生库测试或求解器的运行依赖。构建与辅助测试的范围分别记录，缺辅助环境时该组为 NOT_RUN。
