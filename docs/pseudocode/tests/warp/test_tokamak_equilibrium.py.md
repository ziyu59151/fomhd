# External equilibrium adapter tests

## A0-PY-TEST-TOKAMAK-EQUILIBRIUM-01: Analytic polynomial samples

Write and parse a G-EQDSK-style grid for
`psi = (R - 1)^2 + Z^2`, with nonconstant F, total pressure and q profiles.
Hand-derived derivatives are `dpsi/dZ = 2Z` and
`dpsi/dR = 2(R - 1)`. Check two-dimensional query broadcasting, the exact
normalized flux, field signs and division by R, and profile values after a
nonzero psi-factor conversion.

## A0-PY-TEST-TOKAMAK-EQUILIBRIUM-02: Invalid point domains

Reject incompatible query shapes, nonfinite coordinates, coordinates outside
the source grid, normalized flux outside `[0, 1]`, and zero or nonfinite psi
factors.

## A0-PY-TEST-TOKAMAK-EQUILIBRIUM-03: Circular flux contour

At `psi_N = 0.8`, Matplotlib must return a simple circular contour around the
known axis. Check finite `N x 2` R-Z points, absence of a repeated closing
point, positive signed area, recorded level, and radius near
`sqrt(0.8)`. Check that an outward-touching contour is rejected as open, a
level beyond the sampled flux range is rejected as absent, and invalid levels
or nonfinite source data fail explicitly.
