# Offline A0 analysis

`tools/analyze.py` reads accepted FOMHD run directories and writes one
no-clobber JSON report. It uses Python, NumPy, and the standard library only.
It never changes a run, snapshot, observation, or existing report.

Run it from the repository root with the project environment:

```bash
/home/lxzeng/projects/fomhd/.venv/bin/python tools/analyze.py --help
```

## Convergence

The convergence descriptor declares exactly one scan variable, a fixed physical
region, the observable unit and norm, and exactly three accepted levels. Use one
of these two independent methods:

- `reference.kind`: `plane_wave` or `constant`; the tool evaluates the
  independently declared analytic field at every actual snapshot location.
- `reference.kind`: `none`; the tool uses differences of three snapshots. Equal
  refinement ratios use the closed-form Richardson order. Unequal ratios use a
  bracketed bisection on the declared `order_interval`.

A complete analytic-descriptor skeleton is:

```json
{
  "schema_version": 1,
  "case_id": "a0-spatial-scan",
  "scan_variable": "grid.hR",
  "time_s": 1.0e-8,
  "roundoff_floor": 1.0e-14,
  "observable_scale": 1.0,
  "roundoff_relative": 1.0e-12,
  "time_tolerance": 1.0e-15,
  "case_comparison": {
    "kind": "audited_refinement",
    "physical_case_id": "a0-spatial-scan",
    "initial_profile_comparison": {
      "kind": "analytic_reference",
      "unit": "1/m^3",
      "evidence_sha256": "0000000000000000000000000000000000000000000000000000000000000000"
    }
  },
  "observable": {
    "name": "ion density perturbation",
    "key": "n_i",
    "component": null,
    "mode": -1,
    "unit": "1/m^3",
    "region": {"R": [0.2, 1.8], "Z": [0.0, 1.0]},
    "norm": {
      "kind": "weighted_RMS",
      "weights": {"kind": "uniform"}
    }
  },
  "levels": [
    {"run": "/path/to/coarse", "h": 0.4},
    {"run": "/path/to/medium", "h": 0.2},
    {"run": "/path/to/fine", "h": 0.1}
  ],
  "reference": {
    "kind": "plane_wave",
    "amplitude": 1.0,
    "k_R": 0.3,
    "k_Z": -0.2,
    "omega": 2.0,
    "phase": 0.0,
    "unit": "1/m^3"
  }
}
```

The supported scan variables are `time.dt`, `grid.hR`, and `grid.hZ`.
The descriptor `h` must equal the resolved value at `scan_variable`. The
resolved and manifest input hashes must agree. Config identity is fail-closed:

- it validates the original UTF-8 JSON;
- it removes JSON whitespace only outside strings, preserving numeric
  spellings such as `1e+20`, `1e-07`, and `-0.0`, escaped quotes/backslashes,
  Unicode code points, and key order;
- SHA-256 of those minified original bytes must equal `config_hash`, exactly
  reproducing the writer's `value.dump(2)` file and `value.dump()` hash;
- an optional `resolved_config_sha256` anchor (top-level, `provenance`, or
  `input_hashes.resolved`) additionally verifies the exact unresolved file
  bytes;
- a mismatch is rejected. Python parsed-value reserialization is never used to
  verify the hash.

Every declared raw-file anchor is checked, including when lexical verification
already succeeds. A valid raw-file anchor can also certify the resolved bytes
when a different JSON serializer produced the recorded config hash. Observable
units must match the canonical SI field units; labels cannot reinterpret a
density as pressure.
Norm units follow the field and weights. Weighted RMS and maximum norms retain
the field unit; unnormalized weighted L2 includes the square root of the weight
unit. An explicit norm unit must match this derived unit.

For `time.dt`, `same_grid_same_initial` is required, all initial and geometry-file hashes
must agree, the entire grid, start time and initial kind must stay fixed, and
`start_time + nsteps*dt` must align at every level. Fixed parameters use exact
comparisons; positive refinement coordinates use relative tolerance only.
For a spatial scan, the time step, start time, initial kind, Fourier truncation,
sampling count and physical domain remain fixed. The transverse spacing must
either stay fixed or refine with a fixed grid aspect ratio. For a grid scan with
different grids, `case_comparison.kind` must be `audited_refinement`; a
`same_grid_same_initial` declaration is rejected.

Snapshot layouts are physical, not dense cell rectangles:

- fluid scalar fields use `[mode, iz, ir]` on cell centers;
- `B_R` uses `[mode, zface, rcell]`;
- `B_phi` uses `[mode, zface, rface]`;
- `B_Z` uses `[mode, zcell, rface]`.

Active cells are derived from `lcfs_R/Z`; active faces and vertices require at
least one adjacent active cell. Inactive dense padding is never included in a
region or norm. Coordinates use `Z0 + (f-1/2)*hZ` for axial faces and
`R0 + (e-1/2)*hR` for radial faces.

The norm names have separate definitions:

- `weighted_L2` is the physical discrete norm
  `sqrt(sum(weights*abs(value)^2))`;
- `weighted_RMS` is the normalized root mean square
  `sqrt(sum(weights*abs(value)^2)/sum(weights))`;
- `linf` is `max(abs(value))`.

Every error and difference record includes the selected definition. Use the
same norm, region, and weights at every level.

For physical A0 integrals use `"weights": {"kind": "a0_quadrature"}`. This
includes the toroidal `2*pi` factor and the actual cell or face measure:
`R*hR*hZ`, `R*hR*lz`, `Rbar_r*hZ*lr`, or the vertex `wv`. A physical
`weighted_L2` report includes the volume power in its unit, such as
`T * m^(3/2)`. `uniform` and `radial` remain explicit synthetic-weight options.

For unknown solutions, set `"reference": {"kind": "none"}` and use
`h0 > h1 > h2`. Different source grids require an explicit common mapping:

```json
"mapping": {
  "kind": "bilinear",
  "comparison_grid": {
    "R": [0.2, 0.4, 0.6, 0.8, 1.0],
    "Z": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
  }
}
```

The mapping result records its source layout and interpolation status. Mapping
across inactive geometry is rejected until an explicit active-mask descriptor is
available. Supply `mapping.interpolation_error` only when that number comes from
an independent audit; otherwise it remains `UNAVAILABLE`. Unequal refinement
also requires `"order_interval": [minimum, maximum]`. A zero or roundoff-sized
error or difference produces `UNAVAILABLE`, never an infinite order.

When a representative `observable_scale` is supplied, the effective roundoff
level is

```text
max(roundoff_floor, observable_scale * roundoff_relative)
```

This relative guard catches tiny differences around a nonzero field scale, as
can occur in near-static time-step scans. `roundoff_relative` defaults to
`64 * machine epsilon` when omitted. The report records the supplied and
effective values. Unequal Richardson refinement is evaluated with scale-free
ratios `h0/h1` and `h1/h2`; SI-sized `h` values are never subtracted directly.

```bash
/home/lxzeng/projects/fomhd/.venv/bin/python tools/analyze.py convergence \
  analysis/a0-spatial/comparison.json \
  --output analysis/a0-spatial/convergence.json
```

For a time scan, set `scan_variable` to `time.dt`, fix the spatial grid and
solver-error floor, and change only `dt`. For a spatial scan, use a sufficiently
small independently checked `dt`. Report inner-domain, wall-band, and full-domain
descriptors separately.

## Complex mode fit

Configure observations in the case before running:

```toml
[diagnostics]
mode_observations = [
  {field = 'B', component = 'phi', mode = -1, entity_key = [0, 0]}
]
```

Fit the convention `A(t) = A0 exp(gamma t) exp(i(phase0 - omega t))`. The
target frequency, fit window, and amplitude floor are explicit:

```bash
/home/lxzeng/projects/fomhd/.venv/bin/python tools/analyze.py fit-mode \
  /home/lxzeng/runs/fomhd/example-run \
  --t-start 2.0e-7 --t-end 8.0e-7 \
  --target-frequency-hz 2.5e6 \
  --amplitude-floor 1.0e-12 \
  --field B --component phi --mode -1 --entity-key 0,0 \
  --output analysis/example-run/mode-Bphi.json
```

The fit rejects an unresolved target above Nyquist and a window with fewer than
three valid samples. It records cadence, all residuals, one-sigma least-squares
uncertainties, split-half stationarity, an FFT crosscheck, and a multi-tone
warning. The physical frequency is the negative NumPy signed FFT frequency
because NumPy's FFT kernel is `exp(-2*pi*i*f*t)`. A scalar projection cannot
establish vector polarization, so that item is explicitly `UNAVAILABLE`.

Every observation `(step,time)` must match an accepted `history.csv` row.
Definitions may share times; strict time ordering is checked only after one
field/component/mode/entity projection is selected. OLS centers and scales time
before forming `X^T X`, then converts the slope and covariance back to physical
seconds, so results do not depend on an arbitrary SI time offset.
The modeled signal is evaluated relative to `reference_time_s`; the report
records `amplitude_at_reference_time` and `phase_at_reference_time_rad`.
If extrapolating the amplitude to time zero would overflow or underflow, the
legacy `amplitude` field is null with `amplitude_extrapolation_status` set to
`UNAVAILABLE`. Finite in-window frequency and growth estimates remain usable.

The report includes verified input and source-snapshot hashes, the actual
`run.json`, resolved-config, history, observation, and snapshot hashes, and the
source-file hash table recorded by the solver. Keep analysis reports outside all
input run directories.
