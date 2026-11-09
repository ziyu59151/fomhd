# Few-node multirate WR window tests

Status: bounded CPU behavioral test pseudocode for the selected A1-WR-050
window. These tests do not establish production physics or GPU acceptance.

Source mirror: `scratch/tests/warp/test_time_multirate_wr.py`.
Implementation mirror: `scratch/src/fomhd/time_multirate_wr.py`.

## A1-PY-TEST-WR-01 - Rule, layout and callback identities

```text
for s in {3,5}:
  construct nodes and Q once on CPU
  require endpoints, strict ordering, real nodes and cardinal partition unity
  for each monomial degree <= s-1 and each node_i:
    require sum_l Q_il*node_l^degree = node_i^(degree+1)/(degree+1)

two-component constant source, s=5, M=3, nonaligned coarse nodes:
  use nonzero complex P,Z,A,C and distinct G
  require analytic P_i=p0+H*node_i*(A+C), Z_i=z0+H*node_i*G
  require p0,z0 unchanged and returned arrays independent of inputs
  require tangent work=1 transport +1 field plus one sweep
  require exactly s transports and 4*M field/grid-response callbacks
  require work.complete_ion_evaluations is None

constant initializer:
  perform no tangent callbacks
  map constant P0, apply relaxation, and accept only the OLD second sweep
```

## A1-PY-TEST-WR-02 - Quadrature, refinement and independent oracles

```text
nonaligned M=3, s=5 cubic-force case:
  use the same actual G and C rates for both coarse state integrals
  require completed-step endpoint integrals to match the analytic cubic
  require Z_coarse-z0 = P-p0 when G=C and A=0
  require C integrals enter every density/momentum component
  require final effective weights have shape (M,4) and sum to H

scalar nonlinear coupled ODE:
  independently step the full P'=F(P,Z),Z'=G(P,Z) system with small RK4
  fix H,s and refine the field microstep from M=8 to M=32
  require measured target error to decrease
  require zero residual history offsets and exact callback scaling

two-component linear Hall case:
  form a coupled P/Z matrix with rotational and cross terms
  compare a small-H window endpoint with scipy.linalg.expm

s=3 versus s=5:
  use momentum nonlinearity on a common H,M
  require genuinely different converged coarse target endpoints
```

## A1-PY-TEST-WR-03 - Final sweep, OLD acceptance and work

```text
final samples:
  require 4*M identities in micro-major/RK-stage order
  require two middle Z stage states and identities distinct
  retain only FINAL-sweep full G,C histories and records

acceptance:
  use a relaxed nonlinear case whose converged residual is nonzero
  require P_endpoint to equal accepted OLD node, not mapped endpoint
  require mapped endpoint retained only as defect evidence
  require final input version to match OLD version

explicit componentwise norm:
  use a large old/mapped component with a small physical reference
  require max(reference,|old|,|mapped|) to set its convergence scale
  require raw defect/(rtol*scale)<=1 before relaxation
  use a tiny relaxed update with a large raw defect
  require that update does NOT imply convergence

fixed H,s, M=2 versus M=4:
  require transport callbacks per sweep independent of M
  require field/grid-response callbacks to scale from 4*2 to 4*4
```

## A1-PY-TEST-WR-04 - Failures, rollback and partial stages

```text
nonconvergence:
  require Failure("MultirateConvergence",...)
  preserve attempted counts, residual history, work, final current arrays
  leave p0,z0 unchanged

invalid controls:
  reject nonfinite/nonpositive H/rtol, bool/fractional/nonpositive M,
    wrong s, invalid iteration cap/relaxation/initializer
  require zero transport and field callbacks

physics callback Failure on first-sweep field stage r=2:
  tangent field and stages r=0,1 complete before failure
  preserve original FieldPhysics code, checks, stage and adapter partial array
  attempted_sweeps=1 while completed_residual_sweeps=0
  residual_history remains empty
  expose exactly two completed fine P,Z,G,C samples and identities/records
  expose attempted P,Z and r=2 identity for the failed invocation
  expose no uninitialized fine-array slots
  work field callbacks=4 and rk4_stages=3

reused producer scratch buffers:
  producer returns reusable G and C arrays and overwrites them each stage
  after four completed stages, overwrite both then fail in microstep1/rk0
  completed partial G/C samples remain exact stage values 1..4 and -10..-40
  completed producer record objects remain the original borrowed instances
  attempted identity remains microstep1/rk0

coarse-node producer failure:
  s5/M2 transport reuses one A scratch buffer and succeeds at nodes0,1
  overwrite A scratch and fail at coarse node2
  completed A remains exact values1,2; records are the original borrowed
    producer objects and identities are nodes0,1
  attempted identity/P/Z identify coarse node2 without extra transport calls
  original failure stage/check/adapter evidence remains present
  attempted sweep1/work counts include the failed transport invocation
```

## A1-PY-TEST-WR-05 - Execution scope

```text
CPU command uses the pinned NumPy/SciPy environment with one BLAS/OMP thread
  per process, no bytecode writes and pytest cache disabled
GPU/CuPy cases remain skipped unless the parent delegates an owned Slurm
  allocation; this suite launches no GPU work
```
