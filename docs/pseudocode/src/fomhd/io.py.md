# Configuration, NPZ, run directory and restart

## A0-PY-IO-00: Electron-model provenance

The budget registry is selected by `electron_inertia`: the existing finite
registry or the exact massless stage/interval registry. Run metadata and every
snapshot/checkpoint encode model and applicability:

```text
finite_mass_cgl_b
massless_quasineutral_cgl_b
```

Massless applicability marks electron inertia, electron kinetic energy,
finite-charge Gauss and longitudinal electric energy as
`not_applicable`. Restart compares both `physics_model` and canonical
applicability JSON before `decode_state` or backend upload and rejects either
cross-model direction.

Every species-qualified massless budget key is formed from the configured
`species[0].id` or `species[1].id`. This includes alias, equilibrium,
continuity, wall-interval, kinetic-energy, total-number, and electron work
population suffixes. Model-level names that do not carry a species population
remain unchanged.

## A0-PY-IO-01 - Strict configuration

```text
load_case(path,overrides):
  read bounded raw bytes; parse TOML with tomllib
  reject duplicate/unknown keys, wrong scalar types, nonfinite floats
  require schema/model/units/layout A0/SI/1
  species: distinct safe IDs, mass>0, ion charge/gamma valid,
    electron charge<0, electron_inertia in {0,1}, default 0
  time: dt>0, finite start, nsteps<=2^52, known initial_kind,
    integrator in {midpoint,rk4} default rk4,
    dt_guard in {error,warn} default error, optional
    minimum_samples_per_cycle=32, max_amplitude_error=0.01,
    max_phase_error_rad=0.1
  solver/backend/precision/reference constants/references strictly validated
  grid and fixed boundary strictly validated
  domain floors and complete budget registry strictly validated
  massless species-qualified registry keys use configured IDs
  diagnostics observations validated
  io limits, strides and paths validated
  source manifest must contain provider/version/units/conversion/input hashes/
    continuous residual status; manifest hashes equal actual file bytes
  load geometry NPZ with safe-key/dtype/zip limits
  build host geometry and selected-backend operators
  pass include_finite_mass=(effective electron_inertia != 0)
    after strict parsing; only the massless model selects omitted storage
  operator fingerprint includes the fixed-wall thin/prepared algorithm version;
    hash existing matrices only, never materialize curl-curl for a hash
    hash the explicit finite-storage availability marker for massless
    operators; omit Lphi/Bc records canonically instead of hashing None
  compute config_hash over canonical resolved config
  compute semantic_hash excluding only stop step, paths and display timing;
    include model, physics, effective electron_inertia, integrator,
    geometry/source hashes, budgets
  expose the three guard values on Parameters; frequency assessment and the
    runtime guard hook remain parent-owned in stability.py
```

## A0-PY-IO-02 - Safe NPZ state codec

```text
open_npz(path,limits,allowed):
  require regular non-symlink bounded file
  inspect ZIP central directory before NumPy:
    entry count, duplicate names, path traversal, encryption, method,
    compressed/expanded/header byte caps
  np.load(path,allow_pickle=False,mmap_mode=None)
  require exact key set and no object dtype
  enforce numeric dtype as declared by each schema

encode_state(state,geometry):
  for each scalar/vector mode/cell array:
    use backend-to-host snapshot or explicit CPU array
    require finite values
    scatter active entity [mode][component][active] into dense zero slots
    shape scalar=(M,NZ,NR), vector=(3,M,NZ,NR)
  scatter B_R/B_phi/B_Z on their own face/vertex grids

decode_state(arrays,geometry,K):
  require exact shapes/dtypes and finite values
  require inactive slots exactly zero
  gather active cells into owned contiguous 1-D complex128 state arrays

load_initial/read_restart ingress:
  keep the codec host-only
  upload every fluid array and B exactly once with backend.array to contiguous
    complex128 selected-device storage; no CPU fallback
  upload restart baseline_divB through the same selected backend
  massless read_restart canonicalizes derived electron density/momentum from
    independent state; initialization repeats the check before commit
```

## A0-PY-IO-03 - Fresh run and evidence

```text
check_json(check):
  serialize the existing finite/default record fields unchanged
  for only the explicit optional evidence allowlist
    (modal_work, sample_work, transform_difference, absolute_final_terms,
     upstream_bound, arithmetic_bound, compatibility_bound,
     configured_threshold):
    if the numeric attribute is present, serialize it with typed_value
  preserve unit and value_unit unchanged; these optional values use
    record.value_unit
  if an optional attribute is absent, emit no key so default records remain
    byte-compatible
  never serialize arbitrary record attributes, backend arrays or internal state

Run(case):
  reject symlink/nonempty output; create directory
  create exclusive lock and fixed subdirectories
  copy verified input bytes and source manifest
  require the imported package path is <checkout>/src/fomhd
  snapshot actual src/fomhd modules plus pyproject.toml and uv.lock
  reject snapshot paths or source files outside that exact checkout
  write resolved config, dependency versions, hashes and build provenance
  initialize run.json, history.csv, checks.jsonl, observations.jsonl, run.log

diagnostics(state,records,totals,momentum,elapsed):
  reject no records or duplicate/noncontiguous accepted step
  append check_json records and exactly one history row
  optional evidence nonfinite values use the existing typed-value policy
  update run manifest atomically
  never encode unavailable JSON values as successful physical zeros

failure(error,last_accepted,phase):
  distinguish pre-commit/input, rejected numerical trial and post-commit I/O
  preserve Failure code/stage/check_json evidence/partial arrays in JSON and NPZ
  preserve attached time_step_report in the failure JSON
  only the failure-partial NPZ validator permits raw NaN/Inf evidence; ordinary
    input, initial-state, and checkpoint readers always reject nonfinite arrays
  record last durable checkpoint and original first error
  failure NPZ is never accepted restart state
```

### A1-WR rejected-failure publication

```text
failure partial-array seam:
  preserve the original in-memory Error.partial_arrays unchanged
  classify every value as:
    safe numeric/string scalar or array-like evidence
    omitted structured evidence

  safe evidence:
    NumPy arrays, backend arrays converted through their host accessor, and
      finite/nonfinite numeric scalars or strings converted to host arrays
    reject object dtype, unsupported dtype, dataclass/module/backend graphs,
      and conversion errors
    preserve shape/dtype/content and allow nonfinite failure values

  omitted evidence:
    retain records/identities on the original Failure evidence attributes
    list each key with its original type and stable reason in failure JSON

  publish one NPZ only when at least one safe array remains
  JSON failure record includes:
    all partial keys, safe published keys/count, omitted keys/count
    partial file/hash and publication status
    JSON-safe Error.coupling_metadata when present
    JSON-safe Error.mixed_diagnostic_work when present, otherwise explicit
      accumulator/checker diagnostic-work values from coupling metadata
    exact Failure stage identity and rejected residual history

  accepted state and cumulative data are not read or mutated by publication
```

### A1-WR accepted coupling-metadata publication

```text
Run.time_coupling_evidence(metadata):
  called only after the accepted numerical advance and transaction commit
  owner: I/O publication boundary

  synchronize the accepted backend once
  convert metadata through strict JSON-safe compact conversion:
    dataclasses, mappings, tuples, scalars follow existing policy
    NumPy/CuPy real or integer arrays with at most 16 entries:
      host-copy once at this I/O boundary and preserve every element
    larger arrays:
      do not host-copy; retain dtype and shape summary
    object/unsupported arrays:
      retain explicit dtype/shape omission metadata
    no arbitrary object stringification

  store the safe value in run.json time_coupling_evidence
  do not mutate the metadata object, accepted state, cache or cumulative
  failures after accepted output publication use post_commit_output phase,
    not rollback or failed-before-commit labels
```

## A0-PY-IO-04 - Atomic checkpoint and restart

```text
atomic_publish(target,writer,replace=false,fault=None):
  create exclusive same-directory temp
  writer(temp); fsync file; validate; fault hooks for tests
  no-clobber publish (link) or atomic replace (os.replace)
  fsync parent directory
  propagate phase and whether target may already be published

snapshot:
  encode derived phi/E only when present
  otherwise write <name>_available=false and an explicit reason
  never place a Python None/object array in NPZ

checkpoint(committed,cumulative,context):
  require accepted state and matching endpoint cache
  require cumulative endpoint equals state endpoint
  if current durable pointer matches step: validate pointer and hash; reuse
  synchronize; encode state and cumulative JSON/baseline divB; atomic publish
  publish latest.json; publish run manifest pointer

read_restart(path,case):
  require exact checkpoint key set, accepted A0 schema/layout
  enforce file limit and NPZ safety
  compare semantic/model/integrator/geometry/operator fingerprints, dt/start
  if current massless storage is reduced and its fingerprint differs:
    reject an incompatible storage layout or changed operator before decode
  decode state; validate scheduler endpoint and cumulative baseline/counters
  validate latest hash when pointer names file
  return state/cumulative/parent/trial/source without creating output

inspect(run_dir):
  read only; parse run, contiguous history, JSONL integrity, checkpoint hash
  report interrupted temp/trailing lines without repairing files
```

The effective integrator and electron inertia are materialized in the restart
semantic identity. A historical checkpoint created with the former omitted
finite-electron default therefore cannot match or resume as the new massless
default. The 0.5 storage boundary is also explicit: a massless checkpoint made
with the former full operator set has a different fingerprint from the selected
reduced set. Restart detects that actual fingerprint difference and rejects the
legacy checkpoint as pre-0.5 full storage before state decode or upload. No
migration is attempted. A checkpoint and case made with the same selected
storage still restart normally.

The aligned-box operator fingerprint includes the affine-link magnetic norm
identity, both line matrices, and the transverse measures. Historical initial
arrays use the same sampling chart. Historical checkpoints with the old
operator fingerprint cannot resume with the revised norm.

Ambiguous legacy omitted-integrator checkpoints are rejected for the same
reason. run.json anchors the exact resolved.json bytes with
resolved_config_sha256, independent of JSON serialization ordering.

## A1-PY-IO-01 - Coupling configuration and semantic identity

```text
time allowed additions:
  coupling in {"synchronous","predictor","waveform","multirate_wr"},
    default "synchronous"
  micro_steps actual positive integer, default 1
  waveform_rtol positive finite non-boolean float, default 1e-10
  waveform_max_iterations positive non-boolean integer, default 12

cross-field validation:
  synchronous requires micro_steps==1
  predictor or waveform requires electron_inertia==0 and integrator=="rk4"
  waveform controls are accepted with waveform or multirate_wr
  ion_nodes, wr_initial, and wr_relaxation exist only for multirate_wr
  reject bool, fractional, zero, negative, or noninteger micro_steps
  reject nonfinite/nonpositive/bool waveform_rtol and invalid sweep limit
  reject unsupported combinations before Run directory or trial mutation

semantic time normalization:
  time.coupling=effective Parameters.coupling
  time.micro_steps=effective Parameters.micro_steps
  if waveform:
    time.waveform_rtol=effective Parameters.waveform_rtol
    time.waveform_max_iterations=effective Parameters.waveform_max_iterations
  if multirate_wr:
    time.waveform_rtol and time.waveform_max_iterations
    time.ion_nodes=effective Parameters.ion_nodes
    time.wr_initial=effective Parameters.wr_initial
    time.wr_relaxation=effective Parameters.wr_relaxation
```

Omitted defaults and explicit defaults produce the same resolved semantic
identity. Waveform keys are inserted into semantic identity only for waveform
coupling. Same-storage synchronous/predictor checkpoints remain valid;
pre-0.5 full-storage massless checkpoints do not cross the selected-storage
boundary and are rejected explicitly. `nsteps` remains the accepted
macro-interval count and `dt` remains the macro interval.

## A1-PY-IO-02 - Coupling manifest and restart compatibility

```text
run manifest:
  coupling=effective time coupling
  micro_steps=effective positive integer
  if waveform: waveform_rtol and waveform_max_iterations
  accepted predictor result evidence remains JSON-finite

checkpoint identity:
  write time_coupling and micro_steps scalar metadata for all couplings
  if waveform or multirate_wr: write waveform_rtol and waveform_max_iterations
  if multirate_wr: write ion_nodes, wr_initial, and wr_relaxation
  keep semantic_hash as the complete compatibility gate; waveform controls also
    enter waveform semantic identity

read_restart:
  require the exact waveform key set only for waveform checkpoints
  require time_coupling and micro_steps metadata for all couplings
  compare scheme, M, and waveform controls before semantic identity
    and before state decode/upload
  report the exact checkpoint/configured scheme or M mismatch
  legacy checkpoints without the added metadata are rejected explicitly
  synchronous scheme/M continues to resume; predictor resumes only under the
    same scheme, M, model, integrator, and semantic identity
  waveform and multirate_wr resume only under the same scheme, M, tolerance,
  sweep limit,
    model, integrator, and semantic identity
  multirate_wr additionally requires identical s, initializer, and relaxation
```

Only complete accepted macro endpoints are encoded. No partially iterated
waveform window is encoded; mid-microstage and old/new path states held for
iteration or interval quadrature are never written as restart candidates.
