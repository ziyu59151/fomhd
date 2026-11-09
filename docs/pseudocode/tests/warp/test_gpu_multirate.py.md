# Complete A1-WR CPU/CUDA integration test pseudocode

```text
entry gate:
  run only when FOMHD_TEST_GPU=1
  construct no CUDA context unless SLURM_JOB_ID is also present
  default CPU-only invocation reports explicit skips

shared K0 complete fixture:
  reuse test_gpu_massless physical CPU/GPU context and host-transfer helpers
  reuse test_model_routing state and cumulative builders
  reuse test_time_multirate multirate case writer
  CPU preflight builds a duplicate CPU context through the same argument
    adapter and requires Parameters+geometry contract without touching CUDA
  context construction receives cpu.p and cpu.geometry, never the Context
  CUDA receives an independent deep copy of Parameters and the same dt value
  both contexts are checked to retain their dt across execution
  copy one nonzero host state into CPU and CUDA contexts
  preserve the fixture's original budgets
  controls: massless, RK4, multirate_wr, M=2, s=3/5, rtol=1e-10, cap=12

one/two complete macros:
  capture each actual WindowPhysicsResult from adapter.solve_window
  forbid whole-array host transfer during the CUDA transaction
  require owned CUDA Pnodes,Zcoarse,fineP/Z/G/C,Cintegrals,A,weights
  compare each independent P/Z component using
    scale=max(physical reference,max abs CPU,max abs GPU)
  compare all12 cumulative ledger keys and all6 boundary fields
  require matching accepted step/time/generation/cache/tag/trial
  assert both actual RK4 midpoint identities independently from stage zero:
    Z(rk1)=Z(stage0)+h/2*G(stage0)
    Z(rk2)=Z(stage0)+h/2*G(stage1)
  preserve the strict 2e-9 physical-component parity tolerance
  spy full power-ledger calls and bundle/result diagnostic work counts
  record stage.initialize setup counts separately for CPU and CUDA
  assert macro work against post-setup spy deltas
  retain setup+macro observed totals; never discard setup calls
  make no CPU/GPU-mutual-agreement claim beyond bounded parity

rollback/nonconvergence:
  accept one CUDA macro first
  inject actual fine, coarse, or endpoint Failure, and separately force
    nonconvergence with one-sweep cap
  synchronize only for post-transaction verification
  require original Failure, retained actual partial history/evidence,
    unchanged state/cache/cumulative, and no host fallback

CLI restart:
  use the valid hard massless fixture and fixed budget registry for case writing
  write one- and two-macro CUDA cases with all multirate controls
  run one macro to checkpoint, then run two macros with --restart
  run uninterrupted two macros independently
  compare state and all12/6 cumulative fields under the parent tmp base
  require cumulative interval count 2, never a reset
  reject a s3 checkpoint under a s5 case by semantic/control identity
```
