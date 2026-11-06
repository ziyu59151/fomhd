# G-EQDSK reader tests

## A0-PY-TEST-GEQDSK-01: Analytic polynomial core

Write a minimal g-file-style numeric core for `psi = R^2 + 2 Z^2`.
Require exact analytic reconstruction of `B_R`, `B_Z`, and `B_phi`, plus the
declared boundary shape.

## A0-PY-TEST-GEQDSK-02: Truncation and unknown suffix

Remove bytes from the declared numeric core and require rejection. Append a
suffix to a non-EFIT header and require rejection instead of silent truncation
or acceptance.

## A0-PY-TEST-GEQDSK-03: Retained EFIT suffix

Append a synthetic labeled/comma-delimited suffix to an EFIT header. Require
the standard arrays to remain unchanged, the exact suffix bytes and SHA-256 to
be returned separately, and the format tag to be recorded.

When the retained read-only input is available, parse
`g048059.03650`, require its recorded SHA-256 and 129 x 129 core, sample the
magnetic axis, and exercise contour extraction at 0.8. Raw Matplotlib output
has two open boundary-touching paths and one closed axis-containing path at
that level; the public adapter must reject the open paths under its contract.
Skip only when the user-owned input is absent.
