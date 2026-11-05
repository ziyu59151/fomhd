# 离线收敛与模幅分析

> 状态：active；目标 `tools/analyze.py`，辅助 Python 不参与求解器。依据 09 诊断合同。  
> 依赖：[诊断](../src/diagnostics/diagnostics.cpp.md)、[run](../src/io/run.cpp.md)。最近同步：2026-11-03。

## A0-ANALYSIS-01 — AnalyzeConvergence

```text
read run manifests and selected snapshots with numpy.load(allow_pickle=False)
verify resolved identity without assuming universal Python/nlohmann float parity:
  validate original UTF-8 JSON
  lexically remove JSON whitespace only outside strings
  preserve key order,numeric spellings,escapes and Unicode code points
  require SHA256(minified original bytes)=manifest config_hash
  optionally also verify resolved_config_sha256 file-byte anchor when present
  a mismatch is UNAVAILABLE/fail-closed, never accepted as verified
require resolved input_hashes equal manifest input_hashes
require accepted data,matching model/physical case,units,observable,physical output times
require exactly 3 distinct grid sizes or time steps,exactly one independent scan variable
require descriptor h to equal the resolved value at scan_variable
require observable unit to equal the canonical SI unit for its physical field
derive norm units from field units and weights; reject inconsistent unit overrides
accept only time.dt, grid.hR, grid.hZ as supported refinement coordinates
require one physical-config projection outside grid,time,io,diagnostics
if time.dt: require the entire grid, start_time and initial_kind fixed;
  require nsteps*dt+start_time to align and initial file hashes identical
if grid refinement: require K,Nphi,dt,start_time,initial_kind and physical domain fixed
require the transverse grid spacing to stay fixed or refine at fixed aspect ratio
require time.dt to use same_grid_same_initial; an audit declaration cannot bypass identity
require same_grid_same_initial to have identical geometry and initial-file hashes
validate every declared raw resolved-file SHA256 anchor even when lexical config hash matches
compare fixed parameters exactly; compare positive h with relative tolerance only
if grid changes: require case_comparison.kind=audited_refinement with
  physical_case_id and initial_profile_comparison/evidence SHA-256
require common comparison operator/region fixed in physical coordinates
derive active cell mask from lcfs_R/Z with the solver's long-double certified
  orientation and half-open crossing; reject GeometryAmbiguous signs
never use inactive dense padding
fluid layouts: cell[mode,iz,ir]
B_R[mode,zface,rcell]=Z0+(f-1/2)hZ,R_nodes
B_phi[mode,zface,rface]=both half-grid coordinates
B_Z[mode,zcell,rface]=Z_nodes,R0+(e-1/2)hR
require absolute roundoff_floor; when observable_scale is supplied,
  effective_floor=max(roundoff_floor,observable_scale*roundoff_relative)
  record the definition and use effective_floor for every zero/roundoff decision
norm actual-reference using the declared definition:
  weighted_L2=sqrt(sum(weights*abs(value)^2))
  weighted_RMS=sqrt(sum(weights*abs(value)^2)/sum(weights))
  linf=max(abs(value))
weights.kind=a0_quadrature:
  cells use 2*pi*R*hR*hZ
  B_R uses 2*pi*R*hR*lz
  B_Z uses 2*pi*Rbar_r*hZ*lr
  B_phi uses 2*pi*wv from adjacent active cells
  weighted_L2 unit includes sqrt(volume), for example T*m^(3/2)
if analytic reference:
  evaluate analytic field at each actual field location and comparison time
  errors=declared_norm(actual-reference,specified physical weights)
  p01=log(error0/error1)/log(h0/h1); p12 similarly
else:
  map three levels to common observation grid with explicitly recorded interpolation
  differences=d01,d12; if refinement ratios equal r: p=log(d01/d12)/log(r)
  if not equal:
    let r01=h0/h1,r12=h1/h2
    solve d01/d12=expm1(p*log(r01))/(-expm1(-p*log(r12)))
      on specified admissible interval
    use only scale-free ratios; never subtract near-equal expm1 values at SI h
    bracketed bisection only if continuous sign bracket exists; otherwise UNAVAILABLE
  report Richardson estimate as assumption-dependent,not known exact error
zero/under-resolved difference relative to effective_floor -> UNAVAILABLE,
  not a nonpositive-order warning and not infinite order
output original norms,fit window,reference hashes,interpolation error and status
```

比较 h 时 dt 足够小且有独立时间误差核对；比较 dt 时固定 h/N/求解容差底。内域、壁带、全域三个区域分别计算，不因看见内域四阶宣称整个阶梯域四阶。

## A0-ANALYSIS-02 — ModeFit

```text
select explicit complex observable A(t),mode/component/entity or weighted projection
validate every observation step/time pair exactly against an accepted history row
validate field/component/entity active layout; allow definitions to share times
enforce strictly increasing time only within the selected projection;
  exact duplicates reject, no absolute SI-time floor for distinct samples
require saved cadence resolves target frequency and amplitude exceeds input floor
require fixed preregistered [t_start,t_end],>=3 valid samples
phase=unwrap(angle(A)); amplitude_log=log(abs(A))
center and scale t before least-squares fit phase=phase0-omega*t,
  amplitude_log=a0+gamma*t; convert slope/covariance back to physical time
evaluate model using t-reference_time to avoid cancellation or extrapolation overflow
report reference_time_s,amplitude_at_reference_time,phase_at_reference_time_rad
if extrapolated amplitude at t=0 is unrepresentable: amplitude=null, extrapolation=UNAVAILABLE
report omega=-phase_slope,gamma=amplitude_slope,residuals,window,amplitude,polarization
if multi-frequency beats/nonstationarity or unresolved cadence: WARN; no single-mode claim
compare independent temporal FFT as crosscheck; never modify evolution data
```

相位正负约定为系数 `exp(-i*omega*t)` 与空间 `exp(i*k*phi)`；不同约定先转换。有限窗拟合的不确定度与模型/插值误差分开；不能把数值衰减叫 Landau 阻尼。

输入工具若从速度生成 mom：pad n/u 到同一 N 点→逐点 `m*n*u`→forward→截谱→保存转换误差；不能 Fourier 系数逐项相乘。脚本在项目 uv 环境运行，输出单独 analysis 目录并记录输入 run/代码版本，不覆写原结果。

Accept native source hash strings and Python source records containing relative and sha256 fields. Validate matching relative paths and exact source snapshot hashes. Production resolved-file SHA256 anchors must verify before analysis.
