# build_clt_boundary_report.py pseudocode

## Purpose and boundary

Build a bounded, offline A0-style gallery for paired native-polygon CLT
boundary runs. The builder reads completed accepted snapshots, actual initial
RHS/cache evidence, and the runner's explicit native geometry artifact. It writes a new
output directory only.

The builder never runs a solver, changes source or run evidence, invents
background or perturbation scales, interpolates a field between samples,
triangulates a Cartesian raster, reconstructs pressure or force terms, or
claims scientific success from a rendered image.

The worker deliverable is evidence-gated renderer code and synthetic tests.
It is not a real-data result.

## CLI and output safety

```text
--pair LABEL CONTROL_DIR PERTURBED_DIR   repeatable
--output NEWDIR
--source-root S                          optional canonical source root
--phi-index INDEX                        default 0
--wall-band FRACTION                     default 0.1 of minor radius
--normal-arrows                          optional native boundary normals
--no-orthogonal-phase                    disable the K>=1 extra phase
```

When `--source-root` is omitted, resolve the repository root as
`Path(__file__).resolve().parents[2]`. An explicit `--source-root` is used
directly. Do not consult environment variables, worker directories, old
checkouts, or polygon-runtime overlay fallbacks.

Resolve the output and reject:

1. an existing output path;
2. an output inside any pair directory or the source root;
3. duplicate labels;
4. an empty or malformed pair list.

Create the report in a temporary sibling directory and publish it with one
atomic rename. Remove the temporary directory on failure.

## Input contract

Each run directory must contain `report.json` and the snapshots named by its
`snapshots` list. `snapshot-every` and final accepted snapshots may omit
intermediate states; no missing state is synthesized.

From `report.json`, require:

- run status, `parameters`, physical major/minor radii, mesh, baseline, K,
  `Nphi`, node count, scheme and declared backend;
- native geometry fingerprint;
- the exact listed snapshot names;
- acceptance and diagnostic timing where present;
- failure receipt where the run failed.

Require an explicit runner-produced `geometry.npz` and verify its SHA256
against the report. Return a missing-file error when absent. Never reconstruct cell
polygons, centers or measures from `geometry_input`, generator grids, the
report fingerprint, or a guessed mesh.

Required `geometry.npz` keys:

```text
cell_polygon_points    (sum(vertices_per_cell),2) concatenated RZ
cell_polygon_offsets   (Nc+1,) integer offsets into cell_polygon_points
cell_polygon_counts    (Nc,) integer vertices per cell
cell_R                 (Nc,) native cell-centroid R, m
cell_Z                 (Nc,) native cell-centroid Z, m
M0                     (Nc,) native cell volume weights
physical_wall_distance (Nc,) true cell-wall distances, m
input_circle           (Ntrue,2) actual input contour
numerical_polygon      (Nnumerical,2) computational chord wall
face_coords_R          (Nf,) native face-center R, m
face_coords_Z          (Nf,) native face-center Z, m
face_normals           (Nf,2) native face normals
```

The polygon, offset/count, center, `M0`, physical-wall-distance, input-circle
and numerical-polygon keys are required. Face-center and normal keys are required only for
`--normal-arrows`. If an optional `geometry_fingerprint` scalar is present,
it must equal `report.json.geometry.fingerprint`.

Read:

- one actual cell polygon per active cell;
- native cell centers and measures;
- true cell-wall distances;
- actual input and computational wall coordinates;
- native face coordinates/normals for optional arrows.

Tests may inject an equivalent synthetic `Geometry`; production does not
substitute rectangles.

Each snapshot `metadata_json` must contain the actual:

- `time_s`;
- `step`;
- `units="SI"`;
- `sample_phis` with one finite angle per toroidal sample.

The physical SAMPLE arrays are required by exact name:

```text
B_N_samples                     (Nphi,3,Nc) T
ion_velocity_samples            (Nphi,3,Nc) m s^-1
ion_density_samples             (Nphi,Nc)   m^-3
ion_perp_pressure_samples       (Nphi,Nc)   Pa
electron_density_samples        (Nphi,Nc)   m^-3
electron_perp_pressure_samples  (Nphi,Nc)   Pa
electron_parallel_pressure_samples
                                (Nphi,Nc)   Pa
```

The latest runner payload names have been read directly. These physical
samples are never taken from modal arrays such as `mom_i_modes` or
`n_i_modes`. Complex storage
is accepted only when the selected physical sample is real within a strict
roundoff bound. Every selected value must be finite. Missing keys are reported
by their exact names.

Component order is `(R, phi, Z)`. The default selected `phi_index` is 0 and
its actual `sample_phis[index]` angle is recorded in radians. For K>=1, when
the optional extra phase is enabled, select the actual retained sample whose
angle is nearest to +90 degrees from the selected angle. Do not interpolate;
record the sampled angle and angular offset.

## Run and pair matching

Require a true initial snapshot with `step=0` and `time_s=0` in both runs of
every pair. A first saved nonzero-time state is not an initial state.

Compare reports for:

- identical source fingerprint;
- identical native geometry fingerprint and mesh;
- identical configuration/parameters after excluding only amplitude and an
  explicitly declared backend;
- identical Nphi, K, physical radii, micro/macro time controls and scheme.

Do not match by step alone. Load snapshot times and choose the latest common
physical time between control and perturbed runs. Reject a time mismatch or
missing common state.

Accept `completed`, `segment_completed`, `budget_stopped`, and a failed run
that still has an accepted snapshot. Preserve the report status, accepted
step/time, requested horizon, wall elapsed time, and failure receipt. Label a
failed or shortened comparison as a partial accepted-state comparison. The
renderer makes no terminal-horizon or scientific pass claim.

## Field figures

Field semantics:

```text
B_R, B_phi, B_Z             B_N_samples components
u_R, u_phi, u_Z             ion_velocity_samples components
n_i                         ion_density_samples
p_i                         ion_perp_pressure_samples
p_e_perp                    electron_perp_pressure_samples
p_e_parallel                electron_parallel_pressure_samples
```

For each representative pair and selected actual toroidal sample, create one
three-column figure for each of the eleven fields:

```text
(A) perturbed(t) - perturbed(0)
(B) control(t)   - control(0)
(C) perturbed(t) - control(t)
```

The three panels use one shared symmetric centered color scale and one
diverging colormap. Never autoscale a panel independently. Draw actual cell
polygons with `matplotlib.collections.PolyCollection`; do not use rectangle
patches or field interpolation. Overlay the actual input contour and
computational chord wall with distinct styles. Optionally draw arrows from
the explicit native `face_coords_R`/`face_coords_Z`/`face_normals` arrays.

Treat retained facet normals as physical outward normals. At a native cell
center, orient each matched facet normal so that
`dot(normal, facet_midpoint - cell_center) > 0`. Average the outward normals
of wall-adjacent cell facets and normalize each centroid arrow. Preserve the
actual polygon coordinates, field values and color scales.

The caption records:

- matched step and actual time;
- selected sample index, angle, K, Nphi and active-cell count;
- SI units and shared absolute range;
- absolute peak for all three differences;
- initial control background scale and initial pair seed scale;
- peak/background and peak/seed ratios, or `unavailable` when a denominator
  is exactly zero.

For B_R, B_phi and B_Z, also create separate raw control maps at true t=0 and
the matched final time. Use one absolute shared scale over all six raw
control panels so the toroidal B scale can be read without relying only on
differences.

Keep these toroidal identities independent in figure metadata:

- `n_phi`: the actual excited harmonic from `parameters.mode`;
- `k`: the retained harmonic cutoff from `parameters.K`;
- `n_phi_quadrature`: the actual angular sample count from
  `parameters.Nphi`.

Never substitute `K` for the excited harmonic. This distinction must hold for
K=1, mode=0, Nphi=5 and for the ordinary n_phi=1 case.

When K>=1 and the extra phase is enabled, repeat the bounded field and raw-B
figures for the nearest actual orthogonal phase. Do not synthesize phase
samples.

## Initial force/cache comparison

If `initial-rhs.npz` exists, hash it and show only explicit physical sample
keys:

```text
frame_ion_pressure_force_samples
frame_ion_magnetic_force_samples
frame_ion_electric_force_samples
frame_ion_convection_samples
frame_ion_wall_reaction_samples
frame_ion_net_force_samples
```

Each may be `(Nphi,3,Nc)` or `(3,Nc)` for the selected physical sample.
Plot component magnitude maps only when all six keys are present, finite and
shape-valid. If any are absent, list the exact missing keys and render no
placeholder zeros. Never derive pressure, magnetic, electric, wall or net
force from guessed arrays.

No GS flux contours are drawn unless an explicit retained GS field artifact
is supplied; the renderer does not invent a flux function.

## Fixed physical wall band

Compute `wall_width = wall_band_fraction * minor_radius` in metres. Using the
explicit `physical_wall_distance` array, define:

- wall band: `distance <= wall_width`;
- complement: `distance > wall_width`.

Use native cell measures `M0` as volume weights. For each region, field and
difference (perturbed change, control change, pair difference), record:

- sample count;
- total weighted volume;
- volume-weighted RMS;
- infinity norm.

If a region has no cells or zero total weight, mark every metric
`unavailable`, not zero. Include counts and volumes even when metrics are
unavailable.

## Figures, report and provenance

Use only:

- source `build_a0_report.py.render_report` for the exact A0 HTML/CSS;
- source A1-T0 `configure_matplotlib`, `style_axis`, `panel_label` and
  `save_figure` for plot style;
- source `build_050_sections.py.embed_svg_html` for self-contained SVG data.

Canonical renderer hashing covers exactly these four source files. The
renderer has no runtime dependency on `polygon_io.py`, `polygon_geometry.py`,
or another worker overlay; native plotting inputs come only from the strict
`geometry.npz` contract.

Write to the temporary output:

```text
index.html                  self-contained embedded-SVG A0 report
manifest.json               A0 manifest plus input hashes
data.json                   semantics, availability, metrics, figure records
provenance.json             run timing/failures and canonical hashes
figures/*.svg
figures/*.png
SHA256SUMS
```

Every caption, legend, note and title is English. Keep all coordinate axes in
SI metres and field colorbars in SI units. Include run status, failures,
accepted time, wall time, Nphi distinct quadrature samples, K, Nc, phi angle,
input SHA256 and canonical renderer SHA256.

Describe input provenance without claiming that the renderer inferred a data
origin class or scientific qualification. Write these exact generic fields to
`data.json` and the A0 manifest:

```text
input_origin = "caller-supplied native run reports and snapshot arrays; provenance hashes recorded"
scientific_acceptance = "not established by renderer"
```

Do not emit `real_data_result`. Make the scope state that the renderer reads,
validates and hashes caller-supplied runs and draws actual native polygon
cell data. State explicitly that physical wall accuracy and scientific
acceptance are not established. Do not describe real-data execution as
pending.

Before publishing, hash every consumed input again and fail if any changed.
Refuse remote or external image dependencies. The maximum default is 32
figures per pair set; reject a plan above that bound with an explicit message.

The report conclusion says only that the gallery renders supplied evidence
and does not establish numerical convergence, wall-boundary correctness or
scientific acceptance.

## Failure behavior

Return nonzero with an exact path/key/shape/time mismatch. Never:

- use the first saved nonzero-time state as t=0;
- match only by integer step;
- infer physical time from a filename;
- replace missing arrays with zeros;
- auto-scale one of the three comparison panels independently;
- silently drop a requested field or pair;
- call a synthetic fixture result a production result.
