# `tools/plots/build_clt_intercept_gallery.py` pseudocode

## Scope

Build one offline scientific evidence gallery for:

```text
local geometry / sampling / paired-reference tests;
no complete evolution acceptance
```

The builder reads explicit source JSON paths and a pinned read-only FOMHD source
root. It never runs a solver, sets initial B to zero, modifies the source checkout,
uses a GPU or network, tunes performance, or claims global closure/acceptance.

## Command contract

```text
build_clt_intercept_gallery.py
  --geometry-probes <existing geometry-probes.json>
  --local-pairing <existing local_patch_probe_repair1.json>
  --local-producer <existing scratch/probe_local_patch.py>
  --local-implementation <existing scratch/repo/src/fomhd/boundary_patch.py>
  --local-status <existing worker-root STATUS.json>
  --structured-json <preserved repair3 JSON>
  --structured-npz <preserved repair3 NPZ>
  --structured-producer <preserved repair3 producer>
  --structured-status <preserved repair3 STATUS receipt>
  --old-formal-root <existing equilibrium-audit old-formal directory>
  --old-formal-audit <existing aggregate old-staircase mapping JSON>
  --old-formal-status <existing equilibrium-audit STATUS>
  --old-formal-producer <existing staircase mapping producer>
  --old-formal-implementation <existing staircase preparation helper>
  --source-root <read-only FOMHD checkout>
  --output <new non-existing directory>
```

Immediately select one thread for OMP, OpenBLAS, MKL, and NumExpr. Set
`sys.dont_write_bytecode=True`; use a temporary Matplotlib config under the new
output transaction.

Reject:

1. a missing/non-file input;
2. an existing output path;
3. an output inside the source tree;
4. an output equal to, containing, or contained by any input path;
5. a wrong JSON schema, nonfinite JSON value, missing required family, or source
   hash mismatch;
6. a geometry grid above 24 nodes per axis.

Render into a sibling temporary directory, verify all artifacts, then atomically
rename it to the requested new path. Do not leave a promoted partial output.

## Input validation

Geometry JSON: require the declared scope, SI units, exact circle/grid recipe text,
the eight declared cases, nodes in `{8,12,16,24}`, required grid/link/class/cost
fields, and finite numeric values. Verify the current `geometry.py` and
`boundary_geometry.py` hashes equal `source_files_sha256`.

Local-pairing JSON: require
`fomhd_local_intercept_patch_probe_v2_parent_repair1`,
`repair.corrected_sign_output=true`, the 20 local cases, the five declared
aspect-ratio refinements, and the reciprocal-current local control. Do not read
the v1 output.

Record SHA-256 and byte size for both JSON inputs, both geometry sources, the CLT
coefficient source, the canonical A0 report renderer, and the A1 plotting helper
source. Record the explicit local-data lineage inputs and validate the existing worker
STATUS receipt:

```text
STATUS.artifacts.probe.sha256
  == sha256(local producer)
STATUS.artifacts.implementation.sha256
  == sha256(local implementation)
STATUS.artifacts.corrected_json.sha256
  == sha256(local pairing JSON)
```

Require the producer AST to import `fomhd.boundary_patch`, require that module to
resolve to the explicit implementation under the producer's sibling `repo/src`, and
require the worker-root-relative STATUS artifact paths to match the explicit inputs.
Validate only these existing receipt fields; do not invent a new signature or use an
unrelated current-source hash to imply the old data lineage.

Repair3 physical evidence must use the preserved immutable bundle, not a mutable
worker-root `STATUS.json`. Validate its STATUS-bound JSON, NPZ, and producer hashes,
zero time steps, no stage call, and failed physical rigid-rotation gate.

Old-formal initial-RHS evidence must use the equilibrium-audit STATUS and aggregate
mapping selected by explicit paths. Validate the current shifted-TOML inputs through
the source manifests, per-grid mappings, aggregate mapping, producer import, and
STATUS-bound staircase implementation. Do not infer the formal RHS from an evolved
state or an earlier TOML receipt.

## Figure 1: exact RZ intercept geometry

Use the declared `nodes=12`, `phase_R=.17`, `phase_Z=-.119` recipe:

```text
h=1/(nodes-4)
theta=linspace(0,2*pi,128,endpoint=False)+.013
polygon=(3+.5*cos(theta), .5*sin(theta))
R=2.5+(arange(nodes)-2+.5+.17)*h
Z=-.5+(arange(nodes)-2+.5-.119)*h
geometry=build_geometry(R,Z,polygon)
intercept=build_boundary_geometry(geometry)
old=build_boundary_geometry(geometry,kind="staircase")
```

Require the intercept fingerprint to equal the supplied geometry JSON case
fingerprint. Plot unchanged active centres, true polygon, exact old staircase,
non-interior existing-face axis links, and type-5 R/Z directional metadata.
State that links are metadata over existing Cartesian face variables, not a new
normal-face operator. Do not invent field values.

## Figure 2: declared preprocessing records

Read only the eight supplied geometry cases. Plot boundary-link minimum divided by
the declared grid spacing and supplied per-case geometry elapsed time against nodes,
separated by declared R/Z phase. Labels say these are one-time preprocessing
records, not equal-accuracy performance measurements.

## Figure 3: local physical-field and current controls

Read the corrected local-pairing JSON. Plot:

- dimensionless known-field normal/tangent reconstruction errors versus the
  supplied `max_length_m`, grouped by recorded aspect ratio;
- relative curved physical-gradient error over the same recorded refinement;
- absolute retained-moment and omitted-moment reciprocal-R current power errors
  versus recorded `h_m`.

Retain the corrected pressure sign provenance. State that polynomial/known-field
exactness and the local reciprocal-R control are not global current, Ohm, boundary,
or synchronous-RK4 acceptance.

## Figure 4: near-tangent local conditioning

Plot supplied homogeneous and lifted local Gram condition numbers versus recorded
aspect ratio on log axes. This is a local algebra diagnostic, not a full Hall-system
CFL limit or global matrix condition.

## Figure 5: source-derived coefficient growth

Load the actual promoted `tools/clt_boundary_coefficients.py`. For both sides:

```text
offsets=[.50,2.49]
stencil=irregular_stencil(rule_for_offset, offset, side)
D_phys_p=weights_p/h**p
normalized_L1_p=sum(abs(D_phys_p))*h**p
                =sum(abs(weights_p))
```

The helper weights are dimensionless because its node coordinates are `x/h`.
Multiplying those weights directly by `h` or `h^2` is dimensionally wrong. The
normalized physical metric is invariant for every positive `h`; do not encode a
particular selected `h` in the plotted arrays.

Omit offsets below `.5h`; never clamp a denominator. Keep the exact `1.5h` branch
transition visible. Plot first- and second-derivative normalized L1 growth separately
and label it a coefficient-growth diagnostic, not a Hall-system CFL calculation.

Add one portable known-stencil test using `h=.125`, `.25`, and `1` for both derivative
orders. It must compare `sum(abs(weights/h^p))*h^p` against `sum(abs(weights))`.

## Figure 6: repair3 rigid-rotation strain

Use valid repair3 arrays for N8/N12/N16 and the exact preserved repair3 receipt.
Plot old staircase strain magnitude, corrected positive-M0 candidate magnitude, and
candidate-minus-old in `s^-1`. Retain one magnitude color limit across old/candidate
panels and one symmetric difference limit. Draw the true circle and old staircase
outline. Label the physical rigid-rotation gate as failed; this is not an evolved
state or acceptance.

## Figure 7: self-circle initial formal RHS

The gallery notation distinguishes the radius1m rigid fixture from the separate
radius0.5m geometry/equilibrium inputs; it must not define one universal radius.

Use the full formal initial RHS at `t=0` and zero time steps for N8/N12:

```text
|rhs(mom_i)|   kg/(m^2*s^2)
rhs(p_perp_e)  Pa/s
rhs(B_phi)     T/s
```

`mom_i` is the cell-centered L2 vector magnitude, `p_perp_e` remains cell-centered,
and `B_phi` remains at its existing vertex positions. Use raw SI values with no
subtraction, fake initial difference, normalization, or smoothing. Share limits
across N8/N12 separately for each physical field. Draw the true LCFS and old
staircase support as distinct geometry. Label the result an initial formal RHS, not
an evolved state.

## Optional comparison

No parent-supplied physical VEQ comparison arrays are consumed in this bounded run.
Do not read an invalid round, substitute another file, create a placeholder, or
wait. Mark the optional shared-physical VEQ comparison as deferred until the parent
supplies explicit data. Make no R0/R2 conclusion.

## Artifacts

In the new output directory write:

```text
index.html                         offline HTML, embedded SVGs, canonical A0 CSS
manifest.json                      A0 renderer manifest
data.json                          all plotted raw arrays with units/series labels
provenance.json                    scope, input/source hashes, output policy
sha256-manifest.json               explicit input/output SHA-256 records
provenance/local_data_probe.py     byte copy of exact corrected-JSON producer
provenance/local_data_boundary_patch.py
                                    byte copy of imported implementation
provenance/local_data_STATUS.json  byte copy of worker hash-binding receipt
provenance/local_pairing_probe_repair1.json
                                    byte copy of corrected local data
provenance/structured_repair3_*    byte copies of preserved repair3 lineage
provenance/old_formal_*            byte copies of corrected formal lineage
FIGNAME.png, FIGNAME.svg           seven ordinary scientific figures
```

Set copied provenance files read-only before atomic publication. Hash-check both the
originals and retained copies, include every copy as an explicit output record, and
retain these copies only in the gallery, never inside the business source checkout.
Freeze the final input/output manifests only after the current shifted-TOML formal
inputs and all explicit evidence hashes validate.

Render through the unchanged A0 `render_report` and A1 `configure_matplotlib`,
`style_axis`, `panel_label`, and `save_figure` helpers. Verify rendered `<style>`
contents equal the canonical A0 style block, replace each PNG image source with the
matching embedded SVG data URI, and reject external image URLs.

At completion record grid maxima, elapsed wall time, max RSS, figures=7, deferred
VEQ comparison, failed repair3 physical rigid gate, zero formal time steps,
solver/GPU/network/source-run flags false, and the exact scope sentence.

## Portable overlay tests and external run audit

Source-overlay tests may import `tools.plots.build_clt_intercept_gallery` and
`tools.clt_boundary_coefficients`, use only synthetic dictionaries and `tmp_path`, and
must not contain hardcoded run directories, worker directories, or a precondition that
the final gallery already exists.

The full artifact/hash audit remains outside the source overlay and runs against the
published gallery after rendering.
