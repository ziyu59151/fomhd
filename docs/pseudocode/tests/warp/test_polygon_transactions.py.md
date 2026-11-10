# Native polygon transaction integration tests

## Scope

This is a test-only unit. It does not modify solver, runner, timestep, gate or
wall code. CPU runs use one numerical thread. CUDA cases are skipped unless
`FOMHD_TEST_GPU=1` and `SLURM_JOB_ID` are present; this worker does not execute
them.

Common physical case:

```text
mesh: polygon_dual_v1, N=8
baseline: vacuum
K=0, physical mode=0
amplitude=1e-4, smooth radial seed
synchronous: h=1e-12, M=1
WR: h=1e-10, M=4, s=3, waveform_rtol=1e-12
total accepted macro steps=2
```

The h difference reflects the actual demonstrated passing controls for two
different algorithms. It is not used to compare synchronous and WR numerical
values against each other. Within each backend/coupling pair, restart and
uninterrupted runs use identical h/M/s.

The amplitude is nonzero and the runner’s actual smooth momentum seed and
full wall projection are used. Empty callbacks, zero state or mock progression
do not count as success.

## Restart parity

For each coupling/backend:

1. Run a two-window uninterrupted control with `segment_steps=2`.
2. Run a two-total-window prefix with `segment_steps=1`.
3. Resume that accepted checkpoint in a new directory with
   `segment_steps=1`, keeping total target `steps=2`.
4. Require both runs to complete and checkpoint the same accepted endpoint.

Compare:

- native primary fields `n_i_modes`, `mom_i_modes`, `n_e_modes`,
  `mom_e_modes`, `B_phi_modes`, `B_t_modes`, `current_modes`;
- physical samples for B_N, ion velocity, densities, pressures and their
  component arrays;
- snapshot stage/cache frame identities, advance endpoint tags and coupling
  metadata;
- accepted step/time/interval count;
- checkpoint cumulative JSON recursively, `baseline_divB`, and all strict
  source/operator/geometry/input/scheme identity fields.

Same-backend numeric comparison follows the existing restart bound
`rtol=2e-9`, with scale-based absolute tolerance. Results are first restored to
NumPy via explicit host export before comparison.

## Rollback

For each coupling/backend:

1. Create the real runner context/state and initialize it.
2. Accept one complete macro window.
3. Capture state object identity, cache object identity/tag, primary native
   arrays, cumulative JSON and baseline divergence.
4. Install a failure after exactly two genuine evaluations in the next trial:
   - synchronous: wrap the actual imported `fomhd.time.evaluate_stage`;
   - WR: wrap the actual `fomhd.massless_field.evaluate`.
5. Call the existing `time.advance` once and require `Failure("Injected",...)`.

Assert:

- accepted state object and cache object are unchanged;
- state/time/generation and all primary arrays are unchanged;
- cumulative JSON and baseline divergence are unchanged;
- the failure retains its original stage, checks/partial arrays, and
  `coupling_metadata` where the WR contract supplies it;
- no trial interval is committed or balance duplicated.

Internal context trial/source/generation counters may advance and are not
required to roll back.

## Test selection

All tests are parametrized over `synchronous|wr` and `cpu|cuda:0`. No generic
xfail is permitted for a real-window failure; unexpected failures remain test
failures. The earlier `h=1e-12, M=2` WR failure is retained only as external
failed evidence and is not accepted transaction coverage. The CPU subset plus
the promoted runner restart test must pass before handoff. CUDA cases are
runtime-ready tests for the parent allocation, not worker evidence.

The rollback snapshot copies the cache-tag dictionary; it must not alias a
mutable StageTag record, otherwise an in-place mutation would escape detection.
