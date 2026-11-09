# Current WR benchmark gallery tests

Source mirror: `scratch/gallery/tests/test_build_wr_gallery.py`.

Source and builder locator:

```text
SOURCE_ROOT = Path(__file__).resolve().parents[2]
  for <repo>/tests/warp/test_build_wr_gallery.py this resolves to <repo>
MODULE_PATH = SOURCE_ROOT / tools/plots/build_wr_gallery.py

no private source-root path is embedded in the test
worker verification uses an isolated scratch overlay with the actual repo paths
```

Dated run-archive tests:

```text
wr_evidence_root fixture:
  read FOMHD_WR_EVIDENCE_ROOT
  skip all archive-dependent tests when unset or the directory is missing
  resolve the supplied root for the test duration

archive-dependent tests:
  real RZ reader matching
  RZ mismatch and middle validation
  real periodic/RZ gallery integration

pure schema/math/render tests:
  remain portable and use pytest tmp_path only
  no external run archive and no worker scratch manifest path
```

## WRGAL-TEST-01 - Real offline build

```text
portable tiny valid manifest with actual schema fields:
  build into a new temp directory using canonical source-root renderers
  require exactly six figures and each figure <=4 panels
  require PNG and SVG output; embed each SVG in HTML
  require no HTTP remote asset
  require data.json, manifest.json, provenance.json, index.html, SHA256SUMS
  require provenance hashes equal pre-build input hashes
  require input bytes unchanged
  require blocked availability row rendered explicitly
```

## WRGAL-TEST-02 - Scientific scope labels

```text
require figure metadata text contains Question, Fixed, Scan, Reference and
Pass scope
require strict K1 and periodic positive gate are shown as separate figure data
require generic serial/nonphysical and local-CFL limitations appear
require missing full periodic/EAST/capacity rows remain blocked
```

## WRGAL-TEST-03 - Trust boundaries

```text
require missing required source fails before output creation
require output under raw_root fails
require output that is an ancestor of an input fails
require existing output fails and retains its marker file unchanged
```

## WRGAL-TEST-04 - Actual periodic and RZ campaign data

```text
load the real completed 33-point 2T mathematical trial and
CPU EAST12 first-accepted-macro probe by exact manifest paths
require the physical gallery total is 13 figures
require actual periodic-trial and five RZ cross-section pages plus work page
require each expected PNG/SVG exists and its page appears in output manifest
require mathematical-trial, 1/44, report-derived segment status and
no-production-pass labels
require periodic trial report and RZ report in input provenance
```

## WRGAL-TEST-05 - RZ reader matching and optional checkpoints

```text
decode real initial/final control/perturbed snapshots when checkpoints are absent
require both stages have equal control/perturbed step and time
substitute a final perturbed snapshot for initial and require InputError

The negative test guards the bug where snapshot decoding was nested under
the optional checkpoint loop and only the last stage was matched.
```

## WRGAL-TEST-06 - Case generality and optional real middle state

```text
require completed/segment/partial/failure mapping from actual report status
require completed report reached == global requested macros
require completed with a short accepted prefix is rejected
require 16-cell caption metadata uses Nc=16 and contains no 113-cell constant

toroidal panel contracts:
  no middle -> control/Delta initial and control/Delta final
  middle -> Delta initial/middle/final followed by final control
other B_R/B_Z/u_R/u_Z/n_i/p_i pages -> final control/Delta pairs

middle manifest validation:
  reject only one of control_middle/perturbed_middle
  reject paired middle states that are not strictly between initial and final
  no synthetic middle state or interval is accepted
```

## WRGAL-TEST-07 - Final optional-source and reviewer-label contracts

```text
portable actual-schema fixtures add optional periodic_comparison and
matched_gpu_cost sources:
  require separate periodic-target-mode1, periodic-sentinel-mode16 and
  periodic-fixed-h-h2 figures
  require derived CPU/GPU device metadata on mixed candidate figures
  require signed phase titles and negative values remain signed
  require verification source_kind for same-model target/sentinel/fixed-H
  require named methods on meaningful M coordinates, including
  WR s3 M16 fixed-H h/2 with distinct M8/M16 values
  require nominal 2T nonlinear-mixture and grid-high-sentinel boundaries
  require matched cost labels A0/T1/WR s3/WR s5 and median run_seconds
  require warmup/reset/setup scope and the counter-role erratum
  reject any 144-to-24 complete-RHS interpretation

archive integration:
  require trial comparison caption maps every sorted x index to its name
  require RZ residual panel says first recorded macro and
  not evidence of whole-prefix convergence

these optional-source tests use tmp_path and do not require a final manifest
```

## WRGAL-TEST-08 - First-window boundary controls

```text
portable precheck fixtures provide direct output paths and command --output
referenced run/history/failure records:
  require first-window-boundary-work with four canonical panels
  require corrected coarse, stage-evaluator and field counter labels
  require acceptedprefix and converged-hard-rejected classification
  require no A0/T1 row in the WR ladder
  require energy rejection at effective base-h M=M*h/hbase and y=1 line
  require kinetic/field rejection series only where explicit checks exist
  require accepted-prefix rows not zero-filled
  require first-window, budget-stop/unscanned and no-stability-bound captions
  require every existing referenced record in provenance
```
