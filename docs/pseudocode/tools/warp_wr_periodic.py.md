# Segmented periodic WR campaign runner pseudocode

## Scope

One CLI invocation runs one case and at most `segment_macros` complete accepted
macros of the existing periodic fixture. The global `total_microsteps` horizon
never changes with segmentation. No warmup is run in the correctness campaign.

Portable import layout:

```text
tools.warp_wr_periodic imports tools.warp_multirate_wr
tests import tools.warp_wr_periodic
standard project paths are src, tests/warp, and project root; tools is a
namespace package, not an additional PYTHONPATH entry
direct CLI execution derives the project root from __file__ before importing
the tools package
```

## Configuration

```text
validate device in {cpu,cuda:N}
validate coupling in {synchronous,predictor,waveform,multirate_wr}
validate wall_seconds<=1600 and positive M,s,h,total_microsteps,segment_macros
require total_microsteps%M==0
total_macros=total_microsteps//M

non-synchronous: dt=M*h; micro_steps=M; nsteps=total_macros
synchronous:     dt=h;   micro_steps=1; nsteps=total_microsteps
multirate_wr: ion_nodes=s; wr_initial; wr_relaxation="1
host cap=16GiB; device cap=8GiB
```

## Initialize or resume

```text
context,source_state,fixture_meta=periodic_inputs(...)
apply coupling/time/memory controls
preflight core.estimated_peak_bytes and declared window arrays

if no resume:
  cumulative=cumulative_record()
  committed=stage.initialize(source_state,context,cumulative,False)
else:
  read previous checkpoint.npz with allow_pickle=False
  verify accepted state, scheme/input JSON, source hashes, fixture identity,
         state/cumulative clocks, generation, source ids, and next counters
  state=io.decode_state(...); cumulative=io._cumulative_from_json(...)
  committed=restored accepted endpoint; do not run initialize/reset ledger

create NEW output directory or fail; copy accepted prefix history/snapshots
```

## Each campaign macro

```text
stop before next macro if wall/resource budget exhausted
synchronize; start monotonic timer
with scoped actual-producer spies:
  if synchronous:
    for M microsteps: result=time.advance(committed,cumulative,context)
  else:
    result=time.advance(committed,cumulative,context)
synchronize; measure elapsed/failure cost
require accepted endpoint, cumulative clock, finite package arrays
append one compact JSONL history row:
  clock, target/sentinel complex amplitudes, energies, hard/soft check summary,
  cumulative ledger JSON, coupling residual/work metadata, node count,
  actual producer/RHS/wall counts, memory, synchronized time
save source and selected accepted P/Z snapshots only
write checkpoint from io.encode_state + io._cumulative_json + strict metadata
```

No WindowResult or adapter evidence suite is retained across macros. Failed
trials leave the previous checkpoint/history untouched; partial arrays and
failure cost are written separately.

## Work spies

```text
wrap actual stage/full-ion/field/wall producers only for the advance scope
record calls and outcomes; never infer counts from S,M or guessed formulas
merge result-provided diagnostic work after ledger repair
coarse node callbacks are separate from fine field callbacks
```

## Outputs

```text
run.json, history.jsonl, snapshots.npz, checkpoint.npz
failure.json/failure_partial.npz only on an unaccepted attempt
```

Source manifest uses module identities and hashes only, without worker paths.

## Nonfinite metadata publication

```text
state and accepted endpoint arrays remain subject to finite hard gates
diagnostic/reference/bound metadata may legitimately contain +/-Inf or NaN
encode each nonfinite scalar as {value:null,nonfinite:"+Inf"|" -Inf"|"NaN"}
never serialize bare NaN/Infinity and never turn it into zero/pass
write failure evidence inside a secondary publication boundary
if evidence publication itself fails, retain that error on the original
Failure, write a minimal failure record when possible, and re-raise the
original Failure rather than a ValueError from the writer
```
