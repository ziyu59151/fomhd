# `tools/analyze_clt_controls.py` bounded analysis

## Purpose and invocation

This helper reads completed saved control runs. It does not initialize the
solver, rerun evolution, create missing snapshots, or write into a run
directory.

```text
analyze_clt_controls.py
  --run RUN_DIR [--run RUN_DIR ...]
  --output NEW_JSON
  [--wall-width 0.05]
  [--continuous]
```

`--wall-width` is a finite positive physical distance in metres.
`--continuous` is opt-in because rebuilding the circular-equilibrium object
has a measurable cost. The output file must be new.

## Run inputs

Each run directory must contain:

```text
report.json
geometry.npz
snapshots/*.npz
initial-rhs.npz
```

Use only the file names listed by `report.json["snapshots"]`, or sorted files
under `snapshots/` when that list is absent. Parse each snapshot's
`metadata_json` to obtain the true `time_s` and frame identity. A missing or
non-finite true-initial or final time makes drift analysis unavailable; it is
not replaced by filename inference or zero.

Read physical sample arrays:

```text
B_N_samples                         (Nphi,3,Nc), T
ion_velocity_samples                (Nphi,3,Nc), m/s
ion_density_samples                 (Nphi,Nc), 1/m^3
ion_perp_pressure_samples           (Nphi,Nc), Pa
electron_perp_pressure_samples      (Nphi,Nc), Pa
electron_parallel_pressure_samples  (Nphi,Nc), Pa
```

Mode arrays such as `n_i_modes` are never substituted for these physical
sample arrays.

Geometry requires:

```text
M0                      (Nc,), positive per-radian cell volume
cell_R, cell_Z          (Nc,)
physical_wall_distance  (Nc,), nonnegative metres
```

## Run metadata

For every run report:

```text
run path and status
true initial/final snapshot step and time
scientific status
source fingerprint
geometry and operator fingerprints
K, Nphi, h, H, M, s
accepted step, accepted time, accepted intervals
```

Use report values first. Fall back only to the same metadata preserved in
snapshot headers where explicitly necessary. A directory whose name contains
`prepatch` is labelled:

```text
format_check_only_pre_s_repair_not_scientific_acceptance
```

No run receives a scientific acceptance label from this helper.

## Regions and metrics

With fixed physical width `w`:

```text
whole     = all finite cells
wall_band = physical_wall_distance <= w
interior  = physical_wall_distance > w
```

For a cell-scalar array `v(phi,c)`:

```text
weighted_l2 = sqrt((2*pi/Nphi) * sum_{phi,c in region} M0[c] |v|^2)
weighted_rms = sqrt(sum_{phi,c in region} M0[c] |v|^2 /
                    (Nphi * sum_{c in region} M0[c]))
linf = max_{phi,c in region} |v|
volume = 2*pi * sum_{c in region} M0[c]
```

For a vector sample `v(phi,q,c)`, report each Cartesian component separately
and define the group metric from

```text
|v(phi,c)|^2 = sum_q |v(phi,q,c)|^2.
```

The vector `linf` is explicitly the maximum absolute Cartesian component, not
the norm of the full vector. Do not form a mixed-unit maximum across B,
velocity, density, and pressure groups. Per-group maxima remain separate;
the cross-group aggregate is null with a mixed-units reason.

Each region reports count, volume, normalized weight, and either finite
metrics or:

```text
available=false, weighted_rms=null, linf=null
```

An empty wall band therefore has null metrics, never zero-filled metrics.

## Initial-to-final drift

For each group `B`, `ui`, `ni`, `pi`, `peperp`, `peparallel`:

```text
drift = final_physical_sample - true_initial_physical_sample
```

Report initial background metrics, final-region metrics, and a separate list
of per-snapshot maximum drifts. The maximum over snapshots is not the final
drift and is stored separately.

Group units are:

```text
B: T
ui: m/s
ni: 1/m^3
pi, peperp, peparallel: Pa
```

Background scale is the measured finite initial physical-sample RMS. Target
seed scale is reported separately:

```text
if A > 0: A * vA, only when vA is explicitly present
if A == 0: exactly zero
if A is absent but a scalar perturbation is present: use that scalar
otherwise unavailable with a reason
```

Store the scalar pressure perturbation separately even when it is zero; it
does not override a nonzero velocity seed. Do not floor or guess a scale.

## Optional continuous comparison

When `--continuous` is set and the campaign provides
`inputs/circular_gs/gs_equilibrium.json`, use
`tools.circular_equilibrium` as the independent continuous oracle:

- Prefer `report.init.baseline.equilibrium_parameters`, the declared per-run
  constants.
- Compare R0/a against the actual run major/minor radius.
- For legacy campaign constants without a declared per-run object, require
  R0/a/F0/n0 to match persisted baseline metadata; otherwise mark the
  comparison unavailable. Never compare a nearby run against the wrong
  aspect ratio.

- GS baseline: construct `build_gs_equilibrium` from saved parameters and the
  run's recorded GS degree, then call `sample_gs` at physical cell centers.
- Vacuum baseline: call `sample_vacuum` with saved parameters.
- Compare the mapped physical `B_N_samples` at true t0 with continuous
  `B_R/B_phi/B_Z`.

Report component and vector weighted RMS/Linf by region. Missing constants or
an unavailable oracle are explicit; no equilibrium API or continuous field is
invented. Report the run preprocess degree and external high-accuracy degree
separately; a numerical degree difference is explicit and does not silently
change model identity.

## Initial RHS force cache

If `initial-rhs.npz` contains all six explicit physical sample keys:

```text
frame_ion_pressure_force_samples
frame_ion_magnetic_force_samples
frame_ion_electric_force_samples
frame_ion_convection_samples
frame_ion_wall_reaction_samples
frame_ion_net_force_samples
```

report each vector's shape, finite count, nonzero count, absolute maximum,
and region metrics. Compute the saved-array closure defect:

```text
defect =
  net
  - (-convection + pressure + magnetic + electric + wall_reaction)
```

and report it by region. A nonzero net force is not called a physical
unbalance unless the retained wall reaction participates in this closure.
Missing keys are listed and never zero-filled.

Also inventory, when present, only these named diagnostic contracts:

```text
theta/strain/pdot_*:   (Nphi,Nc)
gradient:              (Nphi,3,3,Nc)
convection:            (Nphi,3,Nc)
```

Accept those exact shapes or a one-dimensional flat buffer of the same total
size and reshape it to the named contract. Never infer an arbitrary shape for
an unknown diagnostic array. Mask the cell axis once, retaining all
intermediate component axes.

Native RHS cochain arrays are not interpreted as physical B-dot vector
components without an actual QB reconstruction.

## Output

Emit strict JSON with:

```text
schema
generated timestamp
wall width
continuous requested
runs
scope statement
missing/unavailable evidence
```

The helper owns only the requested output file and performs no evolution.
