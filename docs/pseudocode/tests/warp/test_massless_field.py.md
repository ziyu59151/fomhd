# Massless field-only test pseudocode

## FIELD-TESTS-01: Derivation switch

```text
spy BuildNonElectric:
  default DeriveElectron -> exactly one species-0 call
  ion_transport=false -> zero calls
  compare all derived electron arrays -> exact equality
```

## FIELD-TESTS-02: Independent field oracle

```text
for periodic nonuniform and closed-wall K=1 nonuniform fixtures:
  snapshot input bytes/identity
  full=massless.evaluate_stage(P,Z,tag)
  field=massless_field.evaluate(P,Z,tag)
  compare electron_perp,electron_parallel,B,E,J,ue,ne,me
  require at most 2e-12 max-relative error
  assert input unchanged and field record has no fake ion RHS/ledger
```

## FIELD-TESTS-03: Field-only alias and zero transport

```text
reject massless.build_non_electric, massless._alias_probe, and full check_stage
call field evaluate(alias=true)
require no rejected call
require padded alternate tag, owned cache, alias_B/perp/parallel checks
require explicit second field/derive/electron-CGL/wall evaluation count
```

## FIELD-TESTS-04: EM/feedback force split

```text
field frame creation leaves ion_em_terms=none
construct direct q_i n_i(u_i cross B_N + H E)
apply actual wall projection and FFT
compare exactly to electromagnetic_rate
compare to separately projected full-stage EM force
check full F - EM - projected transport within 8192 eps times full-F scale

require feedback density == -ion.divergence_hat == full ion density RHS
require EM/feedback pressure rates exactly zero
```

## FIELD-TESTS-05: Retained padded EM producer

```text
K=1 nonuniform closed fixture:
  assert ion_em_terms is none before electromagnetic_rate
  count project_wall calls, then request electromagnetic_rate once
  require exactly one projection and unchanged operation deltas
  reconstruct independent direct electric/magnetic and project_wall
  compare ion_em_terms electric/magnetic/projected/reaction and multipliers
  require IonRate momentum == FFT(retained projected)
  require ion_em_terms tag == frame tag
  snapshot input/record and verify reads preserve both
  require zero full transport and no extra producer call from diagnostics
```

## FIELD-TESTS-06: Actual CUDA producer layout on CPU

```text
K=1 nonuniform closed fixture:
  monkeypatch massless_field._warp_magnetic_force to return the direct force
  flattened to the actual CUDA layout (N*3*Nc,)
  request electromagnetic_rate once
  require no shape error and no host transfer/fallback
  require IonRate and wall projection to match independent direct force
  require ion_em_terms.magnetic/electric/projected/reaction remain flat
  require one magnetic producer and one wall projection work count
```

## FIELD-TESTS-07: Hard and failure contracts

## FIELD-TESTS-07: Electron/alias Warp kernel count

```text
CPU context:
  base evaluate -> electron_magnetic_forces=1, Warp kernel count=0

CPU proxy context with backend.name="cuda" but no real GPU:
  monkeypatch the exact massless electron _warp_magnetic_force dispatch
  base+alias evaluate -> dispatch called for base and padded alias
  aggregate electron_magnetic_forces=2 and Warp kernel count=2
  alias_frame Warp kernel count=1
  ion_magnetic_forces remains 0 because EM rate was not requested
  no ion transport/finish/full-ion RHS work
```

## FIELD-TESTS-08: Hard and failure contracts

## FIELD-TESTS-08: Frame ownership

```text
produce frame on owner context A
create same-shape context B with doubled ion charge
request electromagnetic_rate(frame,B)
  require FieldOwnership before magnetic/projection work
  require WorkCounts and ion_em_terms unchanged
  require owner A still produces the original direct-force result

produce alias frame on padded copied context
  require alias_frame.owner_context is the producing padded context
  require base context is rejected for alias EM
  require alias owner context is accepted and charged normally
```

## FIELD-TESTS-09: Original-budget hard and failure contracts

```text
periodic nonuniform fixture:
  retain positive original-budget hard checks
existing K0 fixture:
  retain positive original-budget hard checks

closed 4x4 K=1 uniform fixture:
  do not mutate rtol/atol, hard flags, inputs, or selected checks
  legacy massless.evaluate_stage raises HardGate
  new massless_field.evaluate raises HardGate
  both retain failing wall_electron_flux evidence at the same original
    raw=46277.802784546046 and threshold=20000
  do not relabel either rejection as a pass
  nonuniform K1 field equality remains configured diagnostic smoke only
  true production hard coverage remains pending a representative actual RZ case

incompatible current:
  require WallCurrent/InitialCompatibility and field-only evidence
injected constraint failure:
  require exact StageTag identity and retained partial input/trial arrays
distinct stage tags:
  require distinct owned cache records and Z arrays
```
