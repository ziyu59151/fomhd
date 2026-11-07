# Controlled T0 gallery checks

## A1-PY-CHECK-T0-GALLERY-01 - Offline artifact verification

Read the compact input and built report without running a solver. Check seven
PNG/SVG pairs, eight report pages, embedded local SVGs, English visible text,
canonical A0 CSS, links and provenance hashes. Confirm completed-only endpoint
selection, strict exclusion of rejected candidate clocks, and accepted-horizon
labels. A synthetic boundary regression uses times before/at/after rejection.
Exercise fresh-output refusal and preserve the existing output unchanged.
Report each assertion group and exit nonzero on failure. Synthetic fixture
checks do not establish scientific correctness.
