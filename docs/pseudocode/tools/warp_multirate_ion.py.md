# `tools/warp_multirate_ion.py` pseudocode

## Purpose

Run one bounded C1 or C2 research window, save provenance and raw evidence, and
compare against an independent complete-stage oracle. This tool does not commit
production state.

## Input and setup

```text
parse bounded CLI controls
require a new output directory
require SLURM_JOB_ID for CUDA

build periodic fixture:
  points default 33
  smooth spatial mode 1, amplitude 1e-4
  highest retained mode seed = amplitude*--sentinel
or build closed RZ fixture with points in {8,12}, amplitude 0.02,
  nonuniform 0.1 using tools.warp_t2_field.closed_inputs

for closed RZ, audit with the exact unchanged hard/observation flags
and retain both hard and observation failures without changing their policy

pack P0,Z0
compute separate physical scales:
  P: density, momentum, pressure
  Z: electron perpendicular/parallel pressure, magnetic field
write input.json before numerical work
resolve the source repository from an explicit FOMHD_SOURCE_ROOT, otherwise
  from this installed tool's repository root, otherwise from declared source B
require and hash every src/fomhd/*.py plus the actual tool, test, their imported
  local fixtures/tools, and corresponding pseudocode
report verified Git top-level/HEAD/branch/dirty from read-only subprocesses
name the declared be5 baseline separately from actual Git metadata
hash request and worker evidence only when present; absence is not an error
label the scratch task root as optional worker evidence, not repository authority
write provenance.json before numerical work

derive the independent time contract from controls:
  microstep = --h
  macro interval H_t = M*--h
  total duration = macros*M*--h
  record start/end time, H_t, microstep, steps per macro, and macro count
```

## Candidate execution

```text
for each cold/warmup/sample execution:
  create a fresh isolated context
  for macro in [0,macros):
    run time_multirate.run(method,isolated,P,Z,H_t,M,...)
  retain immutable trial fields and stage history
  use result endpoint as the next research input only

all failures:
  attach prior completed macro windows and the failed macro input
  write traceback, completed-window history, current completed stages,
    scalar iteration metadata, and actual attempted/last P/Z NPZ arrays
  host-convert CuPy arrays explicitly; complex arrays belong in NPZ
  write report with status=failed
  never create an accepted-product claim
```

## Independent references

```text
synchronous same-h oracle:
  on an isolated context, initialize the same endpoint-compatible state
  run M production synchronous RK4 advances with dt=h per macro

optional waveform/full-T1:
  run the existing production waveform scheduler on the same isolated state,
  H, M and reference controls

use one cold execution, the same warmups, and the same synchronized repeats
for both candidate and full production reference; use a fresh isolated context
for every execution

compare full candidate waveform and endpoint against the selected oracle
normalize each component with max(physical component scale, abs(oracle
  component), 1e-300); never use one cross-unit global norm
require candidate and reference total duration to equal macros*M*h
record per-execution counters and campaign multiplier 1+warmups+repetitions
for each algorithm; charge all rejected/outer/reference work
```

The oracle shares the production physical stage but not the candidate split
orchestration. FieldBatch-only continuation is not sufficient as the sole
oracle.

## Diagnostics

```text
record initial/final state totals where a complete stage is available
validate closed K>0 endpoint retained Q-mode extents and quasineutrality
require positive endpoint ion density/pressure and current cache identity
record P_i = q_i n_i u_i dot H E, field/electron instantaneous rates, and
  RK4 stage increments only for each window's converged final outer waveform
retain earlier trial stages only as research history, not a power ledger
electron linear rate:
  2*pi*Re sum_cells(M0[cell] * (Zdot_perp_e[K,cell]
    + 0.5*Zdot_parallel_e[K,cell]))
  using only toroidal zero mode K; magnetic Parseval remains all-mode
project target and highest retained sentinel separately on periodic 1D only
  store candidate/oracle histories, maximum/final gain, and independent flag
closed RZ: omit phase labels and retain component errors/constraints/raw energy
report exact counters, stage-history bytes, and separate FieldBatch work
count adapter normal solves only when wall T has rows; count wall work always
export stage/iteration macro-window index and physical time
mark C1 P-RHS unavailable instead of writing fake zero physical rates
report matched cold/warm synchronized timing with explicit scope
charge every attempted diagnostic full stage, including closed-RZ hard audits,
  in a separate diagnostic_work ledger
```

Timing does not become a speedup claim. Output I/O and fixture generation stay
outside the timed region. GPU code uses one device, one stream and explicit
synchronization at observation points.

## Products

New-scope references retain the generic all-frequency step report as a warning;
target accuracy, resolved-spectrum stability and high-mode contamination are
assessed separately. Original physical hard checks are preserved.
Every failed timed execution adds already completed macro work to the failed
macro's counters. Save the completed candidate endpoint and complete stage
history before reporting a later independent-reference or diagnostic failure.

```text
input.json
provenance.json
report.json
arrays.npz
traceback.txt only after failure
```

`report.json` always includes `scope="bounded research window"`,
`accepted_product=false`, and the missing interval-budget/production
transaction limitations. `status="completed"` and
`execution_completed=true` are separate from `scientific_qualified=false`.
Formal second-order macro behavior remains a design property until
nonzero-hydro refinement evidence is measured. Exit status is zero for
completed execution; it is not a scientific qualification.

`--macros` accepts the parent-bounded range `[1,32]`. The worker does not run
the parent scientific matrix or add another candidate.

The final two selected controls use C2 over 2T with M=128 and, at the same H,
M=256 and h/2. The CLI ceiling is 256 microsteps and 32 macros for these bounded
CPU controls, with the same 18 total configurations, host/device and total job
budgets. The existing 16-case records retain their original controls and hashes.
