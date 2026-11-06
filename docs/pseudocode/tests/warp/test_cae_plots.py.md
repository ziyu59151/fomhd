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
report, TOML, JSON and NPZ byte stream read. Check that the `n=0` history keeps
the nonzero five-phase mean, `n=1` has real and imaginary histories, and the strong
compressional choice reports the exact synthetic time/probe/scale. Check that
failed or stopped solver status remains visible and an unavailable fit never
becomes a displayed numeric frequency.

Add a direct reconstruction regression:

```text
n0 coefficient = A
n1 coefficient = A/2
physical_at_phi(n0, 0, phi=0) == physical_at_phi(n1, 1, phi=0) == A
```

Use this to prevent A/2 from being presented as the physical `n=1` spatial
field. Check ion-energy wording is kinetic plus internal and density scales are
number density in `m^-3`, not mass density.

## A0-PY-TEST-CAE-PLOTS-03 Actual short archive

When the known archive exists, collect the one-step
`east12-n0-cpu-smoke-r2` case on CPU and render it to a temporary new output.
Require its status and source hash to be retained, all present inputs to be
hashed, the unavailable mode-fit reason to remain explicit, and no frequency
to be invented.
