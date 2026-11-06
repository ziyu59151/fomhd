# Production spectrum regressions

Use a small real circular context. Verify every constrained basis column obeys
its declared wall/divergence/flux rows. Verify physical state-direction packing
and n1 pair factors by a direct central RHS directional derivative and the
saved Jacobian acting on a deterministic vector. Never compare the Jacobian
only to a copy of its own assembly. Check finite eigenpairs and preserved
near-static candidates; do not impose a desired physical frequency.

Assert that the exact positive `memory_limit_bytes` value is passed as a
keyword. In a small n4 production spectrum integration, retain all original
arrays and check explicit candidate dimensions, finite complex fields, the
projected background field, cell coordinates/measures, wall samples, candidate
ordering by frequency, and hashable report provenance.

For a small `K=1`, n=1 circle export, verify `background_B_node` is the
zero-Fourier nodal background and equals projection of the retained center
sector `initial.B[K*Nb:(K+1)*Nb]`, not projection of the full multi-mode array.
Check candidate transfer-ratio metadata equals the nodal projected magnetic
quadratic norm divided by the true staggered-`MB` magnetic norm, and that the
selected report retains the ratio.
