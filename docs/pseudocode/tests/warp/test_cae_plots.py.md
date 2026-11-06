# Tokamak CAE plotting checks

## A0-PY-TEST-CAE-PLOTS-01 Selection and evidence retention

1. Exercise semantic-name parsing so newer revisions replace older revisions
   for the same `(kind,nodes,harmonic)`, while generic smoke names are excluded.
2. Keep an explicitly requested missing report, failed case or budget-stopped
   case in the receipt and HTML fragment. Missing arrays produce readable
   placeholder figures instead of deleting the case.
3. Refuse an existing output directory before opening missing input evidence.

## A0-PY-TEST-CAE-PLOTS-02 Synthetic rendering contract

Build a small conventional-output fixture with:

- three cell centers and a polygon;
- two probe cells, short two-step control and perturbed histories;
- node-aligned matched response arrays;
- five toroidal samples that exercise the zero-Fourier mean;
- one unavailable mode fit carrying a frequency-looking value.

Run the CLI on the fixture. Require both PNG/SVG pairs, the fixed
`cae-20261106` fragment id, relative links, and one SHA256 entry for every
report, TOML, JSON and NPZ byte stream read. Check that the `n_phi=0` history keeps
the nonzero five-phase mean, `n_phi=1` has real and imaginary histories, and the strong
compressional choice reports the exact synthetic time/probe/scale. Check that
failed or stopped solver status remains visible and an unavailable fit never
becomes a displayed numeric frequency.

Add a direct reconstruction regression:

```text
n_phi=0 coefficient = A
n_phi=1 coefficient = A/2
physical_at_phi(A, mode=0, phi=0) ==
physical_at_phi(A/2, mode=1, phi=0) == A
```

Use this to prevent A/2 from being presented as the physical `n_phi=1` spatial
field. Check ion-energy wording is kinetic plus internal and density scales are
number density in `m^-3`, not mass density. Name the plotted density
`delta n_i` and its ion-density background explicitly.

Inspect figure heading text for every solver status. Require the teal
`NL | FOMHD time evolution` banner, `device=...`, and
`solver status=completed|budget_stopped|failed|unavailable|initializing`.
Require `grid 20 x 20` and `n_phi=1` without relabeling the grid as `n=20`.
Check unavailability and initialization are not presented as completion.

Render an actual `n_phi=1` CLI overview fixture, not only the heading helper.
Require the imaginary retained-coefficient panel to render without a
`KeyError` and contain `Imaginary retained ion-density coefficient`.

## A0-PY-TEST-CAE-PLOTS-03 Actual short archive

When the known archive exists, collect the one-step
`east12-n0-cpu-smoke-r2` case on CPU and render it to a temporary new output.
Require its status and source hash to be retained, all present inputs to be
hashed, the unavailable mode-fit reason to remain explicit, and no frequency
to be invented.

## A0-PY-TEST-CAE-PLOTS-04 Compact nonlinear spatial map

1. Render a synthetic structure figure and inspect the actual plotted
   `QuadMesh` arrays. Require signed physical `delta B_parallel` and `delta n_i`
   values, total 3D speed, inactive-cell masks, and equal physical SI color
   limits without normalization.
2. Exercise an exactly zero signed field and require a finite masked mesh with
   a readable symmetric physical extent, not a scatter fallback.
3. Require exactly three map axes carrying timestamps, finite real colorbars
   with units, a solid black wall outline, a teal NL banner, and a concise
   case/grid/`n_phi`/`phi` heading. Reject repeated wall legends, probe rings,
   center dots, in-axis footnotes, and structure-figure status footers.
4. Inspect the real `Quiver` artist. Require normalized R/Z directions, a
   short dark style, arrow length bounded by a fraction of the cell spacing,
   and thinning on a high-count grid while retaining nonzero directions.
5. Require public receipt metadata to record the flat actual-cell rendering,
   no interpolation, physical field units, total-speed definition,
   direction-only arrows, and the separate density/velocity frame times.
6. Draw the figure and check tight bounds of every map and colorbar axis against
   the canvas, including its labels and visible ticks, to catch export clipping.
7. Remove or misalign an otherwise valid mapping; require both unavailable
   figure panels and `spatial_structure.available=false` with a concrete reason.
