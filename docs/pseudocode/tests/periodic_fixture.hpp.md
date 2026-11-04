# Test-only local periodic adapter

Target: `tests/periodic_fixture.hpp`. This adapter supplies the controlled local
Cartesian periodic limit used by [the dispersion test](dispersion.cpp.md).
It is not an input geometry, wall option or backend for production runs.

## A0-VERIFY-PERIODIC — Adapter ownership and algebra

The fixture owns its synthetic geometry, operators, material constants and
uniform admissible state. Values are FP64/complex128 in SI units and live on
the host. The phase chain has 9 or 11 points, K=0 and Nphi=1; all component
entity sets have the same size. Equal positive quadrature weights and identity
transfers make the fixture collocated. Empty wall constraints are explicit.

```text
construct the exactly real, skew spectral D from paired sine entries
scale DR and DZ by the declared physical k and propagation angle
assemble G, C, DB, DJ and Cdag with the signs in dispersion.cpp.md
construct the constant-potential gauge and identity transfer operators
check real/skew D, D*1, each Fourier derivative and every curl component
check CG, DBC, Cdag=adj(C) and DJ=-adj(G)
only after all algebra gates pass, expose the fixture to evaluate_stage
```

The radius is 1e30 m to suppress cylindrical curvature in this declared local
limit. The fixture stores all owned data for the entire stage evaluation; no
stage-borrowed view escapes its owner. Algebra failures stop the test before a
production stage is called. Tests retain full original-matrix and field
constraint checks, so the admissible eigenproblem cannot hide a broken stage.

CTest coverage: `verification_local_dispersion_preflight` and
`verification_local_dispersion`. Formulae, physical constants, perturbation
scales, eigenvalue matching and negative controls are specified once in the
linked dispersion pseudocode.
