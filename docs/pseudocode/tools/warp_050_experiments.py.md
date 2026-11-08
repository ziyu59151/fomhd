# Bounded complete A0/T1 experiment harness

Source: `tools/warp_050_experiments.py`

Status: design contract for the scratch-only 0.5.0 Line A harness. The source
and focused tests must agree with this document.

## Authority and scope

- Use the selected read-only FOMHD worktree as the runtime source. Require
  `PYTHONPATH` to place its `src` directory before any editable installation.
- Reuse `tools.warp_tokamak_cae` for physical parameters, equilibrium
  background construction, smooth perturbation, exact production eigenvector
  perturbation, host transfer, and observations. Add no physical solver.
- `fomhd.time.advance` is the only macro transaction. A waveform transaction
  uses `coupling="waveform"`, `micro_steps=M`, `dt=M*h`, and the unchanged
  waveform defaults `waveform_rtol=1e-10` and
  `waveform_max_iterations=12`.
- A synchronous A0 comparison uses `coupling="synchronous"`, `micro_steps=1`,
  and macro `dt=h`. It therefore uses the same physical micro-step `h` and
  horizon as the waveform run, not waveform macro-step `H=M*h`.

## Validation

Validate all inputs before creating an output directory:

1. Accept geometry `east` or `circle`, enclosing nodes in
   `{12,16,24,32}`, `K` and mode in `{0,1}` with `mode <= K`, `Nphi=5`,
   ordinary CPU or `cuda:0`, synchronous or waveform coupling, positive
   micro-step count, positive finite horizon/h/wall/snapshot controls, and
   `wall_seconds <= 1600`.
2. A synchronous request requires `M=1`. Positive explicit `h` must divide the
   requested shared physical horizon into exactly `N` waveform macro intervals
   (`N*M*h`) or `N` synchronous intervals (`N*h`), within floating-point
   roundoff. Automatic `h` selects a provisional bound, computes the smallest
   `N`, then lowers `h` to make the same horizon exact.
3. Auto `h` obeys both
   `h <= .5/(1.5*max(initial_stage_frequency_estimate.omega_estimate))` and,
   when a target frequency is available, `h <= 1/(64*f_target)`. The operator
   estimate remains recorded but is not a new high-mode gate. If repeatable
   `--stability-spectrum-directory` arguments supply every retained sector,
   auto `h` instead obeys `h <= .5/(1.5*rho_max)` plus the separate target
   frequency budget. A selected archive frequency is resolved before auto `h`
   and supplies that target budget when no CLI target is present.
4. Build the A0 massless parameter registry and set host/device caps to 16 GiB
   and 8 GiB. Set `dt_guard="warn"` so historical global-high-mode accuracy
   warnings remain visible. Construct `OperatorOptions` directly with
   `include_finite_mass=False` when that source revision supports the field;
   do not build a default finite-mass operator context and discard it.
5. Require EAST to use the pinned G-EQDSK input. CPU uses one ordinary thread.
   CUDA requires an owned Slurm allocation and is not exercised by this task.

## Inputs and seed selection

1. Build the physical background with `warp_tokamak_cae.background`. Clone it
   into distinct control and perturbed states.
2. Validate and hash the optional archive's `report.json` and
   `spectrum.npz`, but read only `seed_n`, `seed_mom`, `seed_pi`, `seed_perp`,
   `seed_parallel`, `seed_B`, and the six background arrays. Never read,
   load, or diagonalize `jacobian`, `eigenvalues`, or `eigenvectors`.
3. Require report geometry/model identity to match the run and
   `report.physical_identity` to equal
   `warp_tokamak_cae.physical_identity(context)` exactly. Compare every archived
   background Fourier array with the current initial state; the maximum
   absolute discrepancy must not exceed `1e-10` times its physical scale
   (`B`, density, ion momentum, or pressure).
4. Compare archived numerical source hashes with the actual imported source
   files. Record and log mismatches explicitly; do not use them to excuse a
   physical-identity mismatch.
5. Use the report's selected eigenfrequency and hand the archived `seed_*`
   fields to `warp_tokamak_cae.discrete_perturb`. A supplied target frequency
   must identify that selected branch. Without an archive, call the existing
   smooth perturbation with `reference=None` and label it as a compressible
   impulse, not an exact eigenmode.
6. Persist a canonical seed selector in immutable
   `initial_provenance.seed_identity`: archive paths, spectrum/report SHA-256,
   archived source hashes, and the selected branch, or an explicit smooth-seed
   marker. Publish the restored selector in `case.spectrum_directory` and
   `seed_selector`, never merely echoing a resume CLI value.

## Complete stability spectra

`--stability-spectrum-directory` is repeatable and separate from the optional
selected-seed `--spectrum-directory`.

1. Require exactly sectors `0..K`: sector `0` for K=0 and both `0` and `1` for
   K=1. Reject duplicate or missing sectors and every kind/nodes/K mismatch.
2. Read each small report and the `eigenvalues` vector only. Record whether
   `jacobian` or `eigenvectors` are present in the NPZ key list but never load
   those arrays. Hash both report and spectrum files.
3. Require exact `physical_identity`, exact eigenvalue-derived spectral radius
   and maximum growth against the report, and agreement of all six archived
   baseline state arrays within `1e-10` of their declared physical scales.
   Reject a nonfinite archived/current background array or nonfinite computed
   discrepancy before any threshold comparison.
4. Preserve source-hash mismatches as explicit evidence. They do not bypass
   physical or baseline identity. Publish `rho_max` as the largest sector
   radius and label its scope as local discrete linear stability.
5. Target-frequency phase/amplitude assessment remains a separate
   `stability.assess_dt` record over the full requested physical horizon. Its
   `accepted` result is a hard pre-advance gate in fresh and resumed runs.
   Failure rejects an explicit `h`; it is never repaired by silently changing
   the requested step. It is not a nonlinear or all-high-mode stability proof.

## Time transaction and evidence

For each requested matched macro pair:

1. Check the shared wall budget before the pair. Never begin a pair that cannot
   keep both ensembles at the same accepted physical prefix.
2. Record the current operator estimate `omega_current*h` separately for both
   accepted states. `omega_current` comes from
   `stability.stage_frequency_estimate`; it is explicitly an estimate, not a
   rigorous nonnormal spectrum bound. Without complete spectra, a violation is
   a hard harness failure exactly as in the conservative policy. With complete
   spectra, the estimate remains an observation and the original local
   neighborhood guard is the hard pre-advance check:

   ```text
   ratio = max(Bmax/Bmax0, 1) * max(nmin0/nmin, 1)
   ratio <= 1.5
   rho_max * ratio * h_micro <= 0.5
   ```

   `Bmax0` and `nmin0` come from each ensemble's accepted initial baseline.
   Use micro `h`, never waveform macro `H`. Record current and initial
   pressure-per-density rates (ion, electron, and bulk) plus the advection rate
   from `stage_frequency_estimate`, with each current value divided by its
   initial value. These rate ratios are neighborhood observations only and
   introduce no acceptance threshold.
3. Assess the target frequency separately with the unchanged
   `stability.assess_dt` over the full micro-step horizon. Record this as
   target-mode accuracy evidence; do not rewrite historical thresholds.
4. Prepare a shallow committed proxy and copied scalar physical ledger for
   each ensemble. Call `fomhd.time.advance` exactly once per ensemble. Promote
   both macro endpoints only if both calls succeed, so a failure cannot leave
   one ensemble at a later physical time.
5. On success assert both ledger endpoints advanced by exactly one interval
   and both macro times agree. Append one scalar history record per ensemble
   and one JSON-safe coupling/iteration record containing counts taken from
   the actual `result.coupling_metadata`.
6. Keep power, energy, and momentum observe failures as raw failures in check
   streams. Existing positivity, CT, wall, model, and full-stage hard checks
   remain transactional and must not be relabeled.

The report distinguishes the requested horizon, the exact represented horizon,
the reached matched-prefix horizon, and whether a macro probe or wall budget
stopped the run. A probe never turns partial progress into completion.

## Accepted-pair continuation

`--resume-from PREVIOUS_OUTPUT_DIR` requires a new output directory. The
previous directory must contain the accepted-state-v2 `control.npz` and
`perturbed.npz`.

1. Validate both files against hashes in `report.json` and recursively validate
   their recorded parent-checkpoint hashes. Reject missing, corrupt, trial, or
   non-accepted files.
2. Require equal accepted step/time and shared controls, geometry/operator
   signatures, FP64/complex128 precision, physical identity, source hashes,
   waveform tolerances, micro `h`, macro `H`, total horizon, total steps,
   `start_time=0`, and initial/global provenance. Immutable perturbation
   identity includes mode, amplitude, and the initial perturbed-state hash;
   changed CLI values are rejected before model or cache reconstruction.
   An omitted selector restores the immutable selector metadata without reading
   the archive. An explicit selector must match its canonical paths and
   spectrum/report hashes and is checked before context or cache construction;
   a changed explicit selector is rejected.
   The strict runtime fingerprint covers every `src/fomhd/*.py` file including
   `energy.py`, the actual experiment runner, and
   `tools/warp_tokamak_cae.py`. Legacy archived-spectrum source hashes remain a
   separate mismatch-tolerant disclosure and are not the continuation identity.
3. Decode with `io.decode_state`, `io._upload_state`,
   `io._cumulative_from_json`, and `io.decode_vertex`. Complete all mismatch
   checks before reconstructing or promoting accepted solver state.
4. Restore the `next_trial`, `next_source`, and `next_generation` scheduler
   identities. Reconstruct the cache with a throwaway `cumulative_record()`
   through `stage.initialize`; discard that ledger and restore the accepted
   ledger exactly: baseline totals/momentum/divB/step/time, every integral, and
   accepted count.
5. Keep `p.start_time=0` and the original global time controls. Start the loop
   and history at the restored step, and keep snapshot names on global steps.
   `--probe-macros` limits additional macros in the current segment.
6. Promote only a matched accepted pair. Failed trial arrays remain separate;
   checkpoints always describe the last accepted pair. This is experimental
   continuation, not an official production restart claim.

## Streaming and persistence

- Write scalar history and checks immediately with flushing; do not retain a
  list of every full stage or spatial state.
- At macro multiples of `snapshot_every`, and always at the final accepted
  macro, write encoded accepted state and scalar observations under
  `snapshots/`. The number of spatial snapshots is bounded by
  `ceil(planned_macros/snapshot_every)+2`.
- On budget stop or failure, write the last accepted state and physical ledger
  for both ensembles and retain failing trial arrays under `failures/`.
  Failure arrays never become accepted state.
- The checkpoint is an explicit accepted-state NPZ with exact schema,
  provenance, encoded state, `cumulative_json`, and baseline divergence. It is
  not a production restart format and the report says so even though the
  accepted state and physical ledger are complete.
- Accepted-state-v2 checkpoints also record explicit source/trial counters,
  scheduler identities, initial/global provenance, compatibility signatures,
  strict runtime fingerprints, parent checkpoint hashes, stability policy, and
  each neighborhood baseline.

## Output

Create a new output directory and write:

- `report.json`: status, scope, resolved run plan, identities, source/archive
  hashes, counts, reached/requested horizons, and failures;
- `resolved.json`: all effective physical/time controls;
- `provenance.json` and a byte snapshot of the actual imported source;
- `input_mapping.npz`, `perturbation_mapping.npz`, and initial RHS arrays;
- per-ensemble history/check/coupling JSONL streams and initial checks;
- bounded `snapshots/`, final accepted `checkpoints/*.npz`, and separate
  `failures/*.npz`;
- `logs/run.log`, `logs/allocator.jsonl`, and `logs/failures.jsonl`.

Record actual `Nc`, `Ne`, `Nb`, a whole-array inventory with dtype/shape/bytes,
resident bytes, sampled host RSS/`ru_maxrss`, and CuPy allocator peak samples.
NumPy arrays are host-resident even when the selected backend is CUDA; CuPy
arrays are device-resident. Deduplicate by placement and object identity.
Use the supported device `used_bytes/total_bytes` and pinned `n_free_blocks`
interfaces. Publish device samples before optional pinned metadata; missing
measurements remain unavailable rather than becoming zeros.
Before CUDA construction, enforce the 8 GiB CuPy allocation-pool limit inside
the owned Slurm allocation. Sample the device used/free total separately from
pool bytes, and label sampling rather than an instrumented allocation peak.
Record context construction, background reconstruction, checkpoint validation,
throwaway-ledger source/cache reconstruction, total setup through initial
accepted states, and each synchronized advance separately. Advance timings
include completed and failed attempts. Streaming output is outside those
samples and inside total time. Extract only JSON-safe scalar/sequence metadata;
never embed dense state arrays in JSON.

The bounded spatial ladder permits 12, 16, 24, 32 and 40 enclosing nodes per
axis. The 40-node complete-A0 capacity probe uses the same input, retained modes,
microstep and observation window as the shorter spatial probes, with unchanged
16 GiB host, 8 GiB device and total job caps. This list is a campaign boundary;
it is not a universal production mesh restriction or a physical convergence claim.
