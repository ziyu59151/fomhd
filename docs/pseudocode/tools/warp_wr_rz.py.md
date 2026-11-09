# warp_wr_rz.py complete runtime pseudocode

## Scope

Run segmented, matched control/perturbed EAST K=1 complete accepted
transactions through the selected `fomhd` runtime. Import accepted package and
tool modules from `PYTHONPATH`; do not insert a worker source root.

`coupling=wr` is production `context.p.coupling="multirate_wr"`. It is not a
trial cursor and does not use the old direct adapter endpoint API.

## Configuration mapping

```text
wr:
  p.coupling=multirate_wr
  p.dt=M*h
  p.micro_steps=M
  p.ion_nodes=s
  p.wr_initial=tangent
  p.wr_relaxation=1
  p.waveform_rtol=--waveform-rtol (default 1e-10)
  p.nsteps=global macros

waveform:
  p.coupling=waveform
  p.dt=M*h
  p.micro_steps=M
  p.waveform_rtol=--waveform-rtol (default 1e-10)
  p.nsteps=global macros

synchronous:
  p.coupling=synchronous
  p.dt=h
  p.micro_steps=1
  p.nsteps=global macros*M
```

In every family one outer record spans `M*h`. Synchronous performs exactly M
complete `time.advance` calls per outer record; waveform and WR perform one.
Control and perturbed use identical clocks and global step counts.

## Accepted macro

For each ensemble:

1. Clone the accepted committed state/cache and deep-copy the cumulative ledger.
2. Keep the source version unchanged.
3. Wrap actual producer functions with a scoped counter:
   full stage/evaluation, ion non-electric/transport, electron derivation,
   field/eectron rate, wall construction/projection, and checked interval.
4. Call the ordinary `time.advance` for each required complete operation.
5. Require accepted results, finite checks, and no hard-gate failure. Compute
   the requested endpoint from `core.time_at_step(context.p, expected_step)`;
   never reconstruct it as `base.time + M*h`, which differs by FP64
   associativity on long prefixes.
6. Require cumulative `last_step/last_time` to equal that scheduler clock.

For WR, use `result.coupling_metadata` for final node version, sweeps, residual
history, mixed interval, exact fine/coarse/alias/endpoint work, and accepted
endpoint flags. Counts remain actual transaction counts, never constants.

Only after both control and perturbed succeed:

1. Commit both local cursors.
2. Save direct physical response and retained m=1 coefficients.
3. Append accepted checks and work history.
4. Write accepted cumulative checkpoints.

Failure leaves the original committed left state and cumulative ledger
unchanged. Copy left and referenced parent checkpoints under `failures/`.

## Accepted checkpoints

Use existing serialization primitives:

```text
io.encode_state / io.decode_state
io.encode_vertex / io.decode_vertex
io._cumulative_json / io._cumulative_from_json
io.parse_json / io._canonical
```

Each control/perturbed checkpoint stores:

- accepted state and cumulative ledger;
- step, time, generation, counters;
- source fingerprint;
- input mapping plus pinned gfile hash;
- exact scheme hash including mode, perturbation amplitude, and waveform
  tolerance;
- geometry hash;
- global requested macros and scheduler `nsteps`;
- reached outer records;
- parent checkpoint-chain SHA256.

`--resume` and `--resume-checkpoint` accept only complete accepted endpoints.
Reject mismatch before creating output where possible.

`--macros` is the immutable global outer horizon. `--segment-macros` bounds
one new output. A segment stops as `segment_complete` when the global horizon
remains; the next segment resumes from the checkpoint without resetting the
requested horizon.

## Outputs and claims

Write each accepted macro's:

- checks and working history;
- physical `(Nphi,3,Nc)` vectors and `(Nphi,Nc)` scalars plus prefixed raw
  state arrays;
- direct control/perturbed response and m=1 coefficients;
- accepted checkpoint and next-segment chain identity;
- configured waveform tolerance in checkpoint and snapshot metadata;
- synchronized timing and host/device memory samples;
- source/input/scheme/geometry provenance.

Statuses are `completed`, `segment_complete`, `budget_stopped`, or `failed`.
No speedup, convergence, scientific acceptance, or eigenmode claim is made.
