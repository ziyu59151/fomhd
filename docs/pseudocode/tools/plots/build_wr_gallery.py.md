# Current WR benchmark gallery builder

Status: bounded offline gallery implementation. Parent owns final scientific
interpretation. This builder never runs a solver, GPU, Slurm job, or search for
unlisted raw runs.

Source mirror: `scratch/gallery/tools/plots/build_wr_gallery.py`.

## WRGAL-01 - Manifest, provenance and output safety

```text
CLI: --manifest exact JSON, --source-root fomhd repo, --output new directory

manifest:
  schema=fomhd-wr-gallery-manifest-v1
  title/introduction/conclusions
  raw_root
  sources:
    cfl, microstep_reference, linear_selection,
    backend_comparison, parity_v3
    optional periodic_trial, periodic_comparison, matched_gpu_cost,
    production_precheck, boundary_precheck,
    reviewed_sections and energy_attribution
  availability: id/status/detail rows
  periodic_runs: id, label, run_json, history_jsonl, snapshots,
                 optional checkpoint
  rz_runs: id, label, backend, report, input_mapping, sample_index,
           four required named snapshots,
           optional paired control/perturbed middle snapshots,
           optional checkpoints

validation:
  require every fixed source except optional roles
  resolve relative paths against the manifest, never broad-search run roots
  hash every consumed input and retain path/hash/bytes/roles
  require nonfinite JSON rejection
  output must not exist
  output must not be an input, input ancestor/descendant, or under raw_root
  hash all inputs, canonical plot tools, and own script before rendering
  rehash after rendering and fail if any immutable input changed

periodic inventory:
  read run.json and record schema/status/total_macros
  hash and retain history.jsonl, snapshots.npz and optional checkpoint.npz
  do not infer a horizon from an empty running history

RZ reader:
  require report schema fomhd-warp-wr-rz-production-v2
  derive display status from actual report:
    completed -> completed, require reached == requested
    segment_complete -> segment, require reached < requested
    budget_stopped or partial -> partial, require reached < requested
    failed -> failure
    unsupported or inconsistent macro counts -> InputError
  read actual Nc/Nphi, R/Z/polygon mapping, and canonical Mapping construction
  decode required initial/final snapshots and optional middle pair once
  require control/perturbed step and time matching at every supplied stage
  require strictly increasing actual initial/middle/final times
  never synthesize a middle state or interval
  select one declared phi sample and require physical-real finite arrays
  retain report/mapping/snapshot/checkpoint paths and SHA-256 values
```

## WRGAL-02 - Actual-data figure construction

```text
base figures, each <=4 panels:

1 local_cfl
   source_kind=production_linearization
   local Jacobian max real rate, omega*h, RK4 spectral bound
   label finite nonuniform initial growth as nonnumeric-instability evidence

2 microstep_reference
   source_kind=independent_reference
   mode1 and mode16 h/h2 relative response and phase difference
   high mode16 is stability evidence only

3 analytic_selection
   source_kind=independent_reference
   s3/s5 Picard radius, target error and phase versus M
   narrow old Hall model, not production acceptance

4 short_backend_residual
   source_kind=verification
   actual CPU/GPU relative and absolute residual rows
   synthetic generic comparison is not physical precision evidence

5 generic_wr_parity
   source_kind=verification
   actual generic work, CPU time and CUDA time versus M
   serial synthetic scan; no time-algorithm acceleration claim

6 massless_scope
   source_kind=verification
   strict original K1 raw failure, positive periodic normalized residuals
   keep strict K1, soft consistency-only, and periodic positive hard gate separate

actual periodic trial, when supplied:
  require completed status, schema fomhd-wr-benchmark-v1,
  production_accepted=false, and a multi-row mode_history
  plot target mode1 component magnitudes, mode16 sentinel magnitudes,
  and actual normalized P/Z comparison rows
  label the result mathematical trial, never production acceptance
  normalized comparison x=0..n uses sorted reference names; caption states
  each explicit index -> reference-name mapping

periodic comparison, when supplied:
  require fomhd-periodic-summary-v2, complete cases, common nominal 2T
  nonlinear-mixture horizon, exact reference identity, hard_failed=0
  read target metrics only from target_mode_metrics.target.candidates
  and high-sentinel metrics separately from <sentinel>.candidates
  draw target mode1 complex-error/phase/amplitude and distinct mode16
  grid-high sentinel figures using named methods and M coordinates
  target/sentinel displayed phase values are signed and titles say signed
  target/sentinel device metadata derives from all included candidate schemes
  draw fixed-H pair target/sentinel complex and phase panels with M8 h and
  M16 h/2 labels; never infer macro instability from high-sentinel damping
  same-model synchronous/fixed-H controls use verification evidence class,
  not the independent-model reference badge

matched GPU cost, when supplied:
  require fomhd-matched-wr-cost-v1 and four accepted named methods
  plot actual median complete-advance run_seconds, not integer categories
  name A0 synchronous M1, T1 waveform M8, WR s3 M8 and WR s5 M8
  state one common-horizon warmup, three repeats, zero inside-run resets,
  and separately excluded reset/setup/context setup measurements
  apply COUNTER_ROLE_ERRATUM labels: raw full_ion_calls=ion transport
  producer, full_stage_calls=stage evaluator, full_rhs_calls=full RHS finish
  never present 144 -> 24 as a complete-RHS reduction

first-window boundary controls, when both prechecks are supplied:
  read every listed output path from direct output or command --output
  require each referenced output under that precheck's campaign root
  hash run.json plus history.jsonl/failure.json when present
  accept only one-macro partial first windows or failed hard-gate windows
  classify partial as acceptedprefix and converged hard failure as
  converged-hard-rejected, without assuming stage-specific failure names
  exclude A0/T1 rows from the WR ladder and distinguish base h/half h and s3/s5
  draw declared-M panels for corrected coarse/stage-evaluator/field counters
  and accepted advance_seconds or failed failure_seconds
  draw only explicit kinetic_energy and field_energy_B_interval failure ratios
  at effective base-h M=M*h/hbase with threshold 1
  accepted-prefix rows lack rejection samples and are not zero-filled
  base-h M64/M128 absent rows remain not_run; never infer a stability bound
  captions state first-window only and budget stop/unscanned scope

Every figure caption encodes:
  Question / Fixed / Scan / Reference / Pass scope

Only rows present in listed JSON are rendered. Missing/not-run/blocked data are
an explicit availability table and never zero/NaN placeholder plots.
```

## WRGAL-03 - Canonical renderer and output evidence

```text
load source-root canonical modules:
  build_a1_t2_report.py for data validation/plot/manifest/SVG helpers
  build_a0_report.py for report renderer
  build_a1_t0_report.py for style helper
  build_050_sections.py for hashing/checksum/self-contained verification

generate output:
  data.json, manifest.json, provenance.json, figure PNG/SVG, index.html,
  SHA256SUMS
inject availability table into canonical rendered HTML
embed every figure as base64 SVG; retain local SVG/data links
link the reviewed prior RZ sections unchanged
refuse HTTP/remote assets
write provenance only after rendering; custom pages link data.json

availability rows initially include:
  available present local audit/reference/parity evidence
  available completed 33-point 2T mathematical trial
  partial massless scope
  partial EAST12 first accepted RZ macro, 1 of 44 requested
  blocked full 2T periodic target/phase/energy
  blocked full-state periodic
  blocked complete EAST RZ campaign and middle/final common state
  not_run announced M8 A0/T1 and fixed-h h/h2 periodic cases until exact
          manifest paths and report evidence arrive
  not_run announced GPU16 tight 8-macro/80-ps RZ case until exact paths arrive
  blocked capacity measurements
  blocked matched-target same-GPU cost qualification

No old T0/T1/T2/C1/C2 data are rerendered or overwritten. No production pass,
phase/energy acceptance, capacity, matched cost or speedup is inferred.
```

## WRGAL-04 - Actual RZ cross sections

```text
for each explicit RZ run, render five pages with at most four panels:

1 B_phi:
   no middle: control/Delta initial and control/Delta final
   middle: Delta initial/middle/final plus final control
2 B_R and B_Z control/difference at final
3 u_phi:
   no middle: control/Delta initial and control/Delta final
   middle: Delta initial/middle/final plus final control
4 u_R and u_Z control/difference at final
5 n_i and p_i control/difference at final

plus one standard work/convergence page:
  actual work counters and control/perturbed normalized residual history
  residual panel is explicitly the first recorded macro only, not evidence
  of convergence over the complete accepted prefix

plot only the declared active-cell mapping through canonical _plot_cells
shared magnetic/delta symmetric scale and scalar sequential scale
phi sample, time, backend, nodes/s/M/h/H/Nc, units and limits in panel metadata
caption active-mask count from actual report Nc, never a fixed case count
page status and scope come from the report-derived completed/segment/partial/failure
no smoothing, interpolation, wall cells, or synthetic placeholder values
report accepted prefix versus global requested macros and hard/observe counts
work counters are mixed; never infer net GPU acceleration from them

finalize only after all custom PNG/SVG files exist and HTML embeds their SVG
```
