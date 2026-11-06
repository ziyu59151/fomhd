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
