# Fast-branch comparison and uncertainty summary

Read explicitly named archived cases only. Hash every consumed JSON/NPZ.
For each physical family/grid/harmonic/backend/variant, aggregate only available
probe frequencies; keep unavailable reasons and the spread across probes.
Compare observed frequency to the explicitly named discrete branch and, for
circle, the independent cold reference. Do not turn reference-selected mode
seeding into independent validation. Expose timestep and amplitude variants.

Compare matched delta-B arrays from base and halfdt (exact common cadence),
and base and twice halfamp. No silent time interpolation. Use cell cylindrical
weights R*hR*hZ, complex components, and report full-history relative L2 as
well as endpoint discrepancy. Compare energy imbalance to the quadratic mode
signal and to background total energy on separate axes.

Produce a small standalone summary table/JSON and two PNG/SVG figures. These
are offline diagnostics, not scientific pass gates. Missing/failed cases remain
listed rather than being discarded from plots or acceptance claims.

Visible matched-response titles use `n_phi` for the harmonic and explicitly
state the selected `12 x 12` enclosing grid. Number density is never represented
by that harmonic label. The comparison combines actual nonlinear histories
with separately named discrete frequencies and independent references; the
report gives this mixed provenance an explicit badge.

Backend comparisons require equal source hashes, prescribed physical geometry,
seed, amplitude, resolved parameters and timestep before selecting a common
accepted prefix. Exclude only measured background projection residuals from
that identity, since they differ by roundoff. Reject missing provenance.
Plot raw perturbed balance and the matched control-subtracted balance together;
the latter never replaces the former or changes an acceptance threshold.

Also render four matched magnetic-compression time traces, using archived
parallel_B coefficients and a base-selected cell and normalization. Overlay
CUDA only after the same provenance/cadence check. This is not a mode fit.
