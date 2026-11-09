# Bounded 0.5.0 multirate gallery generator

## Scope

The generator is postprocessing only. It does not import the solver, launch GPU
work, modify run data, select a physics model, define an acceptance threshold, or
turn a return code into a physical result.

All transformations are bounded to:

- one explicit `--run-root`;
- the three canonical plot sources under `--source-root`;
- one new or marker-controlled `--output`.

## Scan and provenance

1. Create an in-memory input index.
2. Before decoding any artifact, hash its exact resolved path and record its
   byte count and role.
3. Decode JSON with non-finite constants rejected, JSONL line by line, and NPZ
   archives with `allow_pickle=False`.
4. Re-hash every indexed input after rendering. Fail if any source changed.
5. Export the resulting path/SHA-256 index as `input-index.json` and embed the
   same records in `reviewed-data.json`.

## Source schemas

### Completed nonlinear runs

Targeted directories:

- `optimized-east*/report.json`
- `retry-east*/report.json`
- `long-east*/report.json`

Used fields:

- `status`
- `case.nodes`, `case.active_cells`, `case.device`, `case.coupling`
- `failure.code`, `failure_code`, or `stop_reason`
- `logs/advance-timing.jsonl` records with `status=completed` and finite
  `elapsed_s`
- `memory-inventory.json.resident_bytes`
- `logs/perturbed-history.jsonl`: `time_s` and
  `energy.{total_J,field_J,baseline_total_J}`
- `snapshots/perturbed-step-*.npz`: `time_s`, `B_R`, `B_phi`, `B_Z`

Only completed reports enter cost or memory plots. Failed and in-progress
reports become availability notes. The field-state panel uses the maximum raw
absolute stored magnetic component and subtracts the first available snapshot;
this is a postprocessing definition, not a physical mode amplitude.

A `long-east*` report with `status=budget_stopped` is retained only as its
stored accepted prefix. Its perturbed history uses the actual global `time_s`
values and each record's original `baseline_total_J`. The note distinguishes
`reached_horizon_s`/`reached_macros` from
`requested_horizon_s`/`planned_macros`; it never claims full-horizon
completion. The series identity includes the report directory, `run_id`, and
`segment_probe.start_step`.

### Initial guesses

Primary source:

- `analysis/initial-guesses/summary.json`

Used fields:

- `cases[].{case_id,config}`
- `experiments[].{case_id,guess,backend,status,history}`
- `experiments[].{iterations_to_convergence,selected_correction_iterations_before_stop}`
- `experiments[].timing.median_s` and `attempted_elapsed_s`
- `experiments[].work.final_execution_producer_rhs_vectors`
- `experiments[].{failure,audit_G_caveat,timing_label}`
- `constant_tangent_pairs`
- `method.input_identity`, `method.failure_residuals`, and `limits`

All six cases and both backends are retained. Fixed-P records are field-window
verification, not coupled T1. Failed E3 audit-G reconstruction is kept distinct
from selected-iterate residuals. E2 backend fixture/input roundoff is named.
Raw per-case reports are only a fallback when the combined summary is absent.

### Multirate candidates

Targeted roots:

- `candidates-cpu-v2`
- `candidates-gpu`

The controlled row table is `multirate-matrix.json`. Each matrix row is matched
against its `input.json` and `report.json`; mismatched method, case, mode, `h`,
`M`, macros, or points are rejected. `candidates-cpu` is separately indexed as
failed historical evidence and never contributes a curve.

For each per-case `report.json`, optional `arrays.npz` and provenance are
hashed. Used fields:

- `status`, `candidate`, `case`, `mode`, `M`, `h`, `H`
- `time_contract.total_duration_s`
- `reference.{p_normalized,z_normalized}`
- `mode_comparison.target.{candidate_history,oracle_history}`
- `mode_comparison.sentinel.{candidate_history,oracle_history}` gains
- `returned_execution_counters.{full_hydro_eval,cheap_D,cheap_L,field_rhs}`
- `timing.campaign_multiplier_per_algorithm`

Series identity includes C1/C2, CPU/GPU, matrix experiment family, physical
case, target mode, points, duration, amplitude, nonuniformity, and the fixed
axis (`h` for an H scan or `H` for half-h refinement). Closed, target mode-4,
fixed-H half-h, and periodic mode-1 H scans never share a curve. The x-axis is
H in seconds. P and Z errors use separate panels. Target amplitude and phase
errors come from target final histories, including an exact zero phase.
Sentinel maximum/final gains are shown with the synchronous histories.
Counts are the returned execution verbatim; campaign multipliers are retained,
not removed, and do not become a net production-speedup claim.

### Fair primitive cost

Targeted roots:

- `fair-cost-cpu/{matrixid}/report.json`
- `fair-cost-gpu/{matrixid}/report.json`

The accepted report shape is the one emitted by
`tools/warp_multirate_cost.py`; it has no separate `schema` key. A completed
point requires:

- `source_input.json` equal to `report.source_input.config` and the controlled
  matrix row;
- matching candidate `input.json` when that file is present;
- `protocol` values equal to the matrix `h`, `H`, `M`, macros, and horizon;
- `protocol.device` equal to the selected CPU or CUDA root;
- `algorithm_controls_unchanged`, `horizon_unchanged`, and
  `matched_cost.same_protocol` all `true`;
- candidate and lean timings each with cold 1, warm-up 1, and 3 samples;
- finite candidate/lean medians plus actual `counters.{candidate,lean}` full
  hydro and field-RHS counts.

One `fair-primitive-cost` figure shows candidate and lean medians, full-hydro
counts, and field-RHS counts. CPU and GPU series are separate marker-only case
points. The figure states that it is not a production transaction, ledger,
restart, checkpoint, or FK solver result, and that hardware ratios are not
algorithm gains. Completed/failed/missing counts are written to availability
notes on every rescan.

When the matrix contains `GPU_coverage_revision.planned_gpu_cases`, candidate
availability distinguishes completed and pending planned GPU rows from matrix
rows that are not scheduled for GPU. The wider 18-case absence report remains
explicit instead of treating unscheduled rows as promised measurements.

### Long-RZ and full cost

Long-RZ figures activate only when a `long-east*/report.json` exposes an
`energy_history` array or scalar history with physical times, total energy, and
an archived baseline for every plotted point. Each continuation segment is a
separate series; different h, coupling, ensemble, or segment identity is never
connected. No baseline is replaced by a first total.

Full-cost support requires both `full-cost-baseline/RESULT.json` and
`full-cost-optimized/RESULT.json` with schema
`fomhd-bounded-full-advance-cost-v1`. It reads
`methods.{A0,T1}.measured_window_ms.median_ms`, `cold_total_ms`, and plan
identity. Legacy combined fabricated fields are ignored. Unmatched plans or
partial artifacts are unavailable.

### Repair 3 scientific separation

1. Parse both target components from each completed candidate report:
   `mode_comparison.candidate.{velocity_mode,magnetic_mode}` and the matching
   `oracle` fields. Compute amplitude error from normalized amplitude and wrap
   the phase difference into `[-pi, pi]`. Use report `candidate_p`/`candidate_z`
   values against their saved scales for full P/Z errors.
2. Build `multirate-target-error` only from periodic mode-1 fixed-h H-scan
   records whose total duration equals the matrix fixed quarter-T horizon.
   Panels are full P error, full Z error, target magnetic amplitude error, and
   target magnetic absolute phase error. Group by method, device, and physical
   comparison; connect only the ordered H scan. The panel limit is four series.
3. Put incompatible controls in separate figures:
   - target mode 4: velocity and magnetic amplitude/phase errors;
   - closed 8/12-node compatibility: full P/Z versus nodes, with the 64 ps
     scope in the caption;
   - fixed-H base/half-h pairs: P/Z and target component errors versus h;
   - 2T horizon pairs: P/Z and target component errors versus h, named as a
     different horizon from the quarter-T scan.
   Each panel is grouped by method and device, uses distinct CPU/GPU markers,
   and reports no convergence-order claim for fixed-H pairs.
4. For long-RZ reports, read both `logs/control-history.jsonl` and
   `logs/perturbed-history.jsonl`. Retain actual global `time_s`, `step`,
   `energy.total_J`, original `energy.baseline_total_J`, `balance_J`,
   `outward_J`, and `source_J` for each ensemble. Budget-stopped and failed
   reports contribute only stored accepted-prefix points and are never relabeled
   as full-horizon completion.
5. Pair `snapshots/control-step-*.npz` with
   `snapshots/perturbed-step-*.npz` by identical stored step and physical time.
   Read only `step`, `time_s`, `observed_BN`, and `observed_velocity`; compute
   absolute RMS control-perturbed differences in T and m/s. Do not subtract a
   first-segment response, interpolate, or reconstruct physics.
6. Render long-RZ as:
   - raw total energy minus each record's original archived baseline for control
     and perturbed ensembles;
   - energy balance, outward transfer, and source transfer for each ensemble;
   - matched RMS magnetic and velocity response at common snapshot times.
   Use short series labels with global step ranges; retain the full report ID,
   run ID, segment start, status, and report SHA-256 in the caption and raw input
   index. Independent segment curves remain unconnected.

## Figures and rendering

1. Build only plot-ready finite paired arrays.
2. Show initial-guess trajectories by backend/unit and pair summaries by case.
3. Show full P/Z error, target-mode error, sentinel gain, and per-method/device
   returned-execution work in controlled candidate figures.
4. Use marker-only series for independent runs, closed controls, half-h
   refinements, target modes, and candidate configurations.
5. Connect only ordered histories or an actual fixed-h H scan.
6. Never synthesize a point for a failed, missing, or in-progress source.
7. Write the canonical T2 reviewed-data schema.
8. Call `build_a1_t2_report.build` directly. That builder supplies canonical
   fonts, colors, Matplotlib configuration, PNG DPI, SVG output, A0 HTML
   rendering, CSS, and embedded SVGs.
9. Add one canonical availability page with explicit missing/failure notes.
10. Re-render through `build_a0_report.render_report`, which remains unchanged.
11. Run canonical CSS, English, embedded-SVG, artifact-link, and offline checks.

## Evidence types

- Analytic Hall controls are indexed from `analysis/hall-control/report.json`
  and named as independent linear controls.
- `nonlinear-energy-field` and `long-rz-energy` remain
  `nonlinear_evolution` figures.
- Candidate errors, fair costs, and full costs remain explicit
  `mixed_comparison` diagnostics with their stated non-claims.

## Controlled output

1. Reject output inside the run root or canonical source root.
2. Accept a new output path.
3. For an existing directory, require the generator's marker schema. Refuse to
   replace an unknown or historical gallery.
4. Build and verify completely in a sibling staging directory.
5. Rename a controlled old gallery to a temporary backup, promote staging, and
   remove the backup only after promotion succeeds.
6. Verify the promoted output again.

## Verification

- Focused schema, missing/failure, provenance, and output-guard tests.
- Real build against the current completed artifacts.
- Controlled regeneration of the same output.
- Re-hash all indexed inputs after rendering.
- Canonical `verify_output` plus independent English/offline/local-asset checks.
- Visual inspection of each PNG.

## Portable repository placement

The scratch paths are not part of the installed module contract. To promote the
scratch implementation into the source repository:

1. Copy `scratch/build_050_multirate_gallery.py` to
   `<repo>/tools/plots/build_050_multirate_gallery.py`.
2. Copy `scratch/tests/test_build_050_multirate_gallery.py` to
   `<repo>/tests/warp/test_build_050_multirate_gallery.py`.
3. Keep invoking the entrypoint with explicit `--run-root`, `--source-root`,
   and an `--output` outside both roots. Do not embed the scratch directory in
   the promoted tool or tests.
4. Run the copied tests from `<repo>` with the repository's normal pytest
   environment. The test locator accepts either the scratch `scratch/tests`
   layout or the promoted `tests/warp` layout.

## Final comparison and rendering boundaries

Keep the quarter-period fixed-h macro scan, fixed-H microstep sensitivity,
mode-four target, short closed-wall controls and two-period horizon separate.
Show both velocity and magnetic target components. A canonical figure has at
most four panels; split state errors from four target components and long-run
energy accounting from matched responses. Do not drop numeric source fields.
Join verified continuation segment plot series by method, h and ensemble using
original global time and baseline; identical boundary samples are deduplicated,
conflicting duplicate values are rejected. Full segment identities remain in
the caption and input index. Retain failed cost receipts while also indexing
explicit fixed-backend retries; only complete reports enter numeric cost plots.

The backend-parity panel reads the independently audited CPU/GPU summary. Plot
only explicitly compared endpoints, using its physical-component normalized
maxima for P and Z, and index the summary's exact raw input dependencies. It is
same-algorithm backend verification, not a timing ratio or an A0 truth claim.

Long-series joins require the existing independent long-RZ analysis and its
current raw-input/output SHA receipt, with empty chain integrity errors.
Require the analysis's resolved root to equal the explicit run root before
using its segment groups; another run's receipt cannot authorize these joins.
Without that evidence, reject multiple same-label long records rather than
silently connecting independent trajectories. This consumes the verified
analysis; it does not reconstruct the physical model. Enclosing R/Z grid nodes
are never called MPI ranks. Output must be disjoint from each input root in
both directions, including existing controlled ancestors, before any staging.

Final auxiliary plots use the same five physical families as the error views:
fixed-h quarter-window H scan, fixed-H h/2 sensitivity, target-mode control,
short closed-wall control, and separate2T horizon. Within each family combine
at most the two methods and two backends, with candidate/reference gains in
separate panels. Labels state actual duration; T means the matrix target period.
Fair-cost source_input preserves CPU/GPU origin exactly; scientific controls
match independently of device, oracle selector and warmup/repetition controls.
The actual run backend/protocol and remaining numerical controls must match.
