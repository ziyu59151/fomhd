# `test_massless_verification.py` pseudocode

## A0-PY-TEST-MASSLESS-VERIFY-01 Procedure and assertions

1. Test the independent algebra first:
   - planted wrong-sign Hall model must be rejected;
   - high-k Hall branch must exceed low-k Hall branch;
   - the four-mode spectrum must have no Langmuir/plasma-frequency branch.
2. Build the actual CPU source stage on the periodic fixture.
3. Verify uniform equilibrium has zero momentum and field RHS.
4. Build the production transverse matrix at two amplitudes and compare it to
   the independent matrix.
5. Verify at least two scaled eigenvalues and branch ordering, while reporting
   all measured errors.
6. Run midpoint and RK4 at three positive step levels for equal final times.
7. Require the modal projection to match the exact midpoint/RK4 amplification
   polynomial. Separately require each refined error toward the exact
   continuum exponential to be below 0.35 times the previous error, with
   the method-specific final error ceilings in the test. This is an error
   reduction gate, not separate fitted order-two/order-four acceptance.
8. Record the production dt-guard warning on the bounded warn-mode runs; never
   relax a physical stage or interval gate.
9. Run actual closed-rectangle source operators with explicit smooth fields and
   independent analytic derivatives for dense, incompressible, positive-density,
   pressure/Ohm, retained-theta, and variable-density velocity controls.
10. Require sign-corrected pressure and region-specific Ohm norms, including
    empty-region handling.
11. Require actual `QHFe/ne` reconstruction from smooth flux plus nonzero
    operator-derived current to converge toward continuum velocity/divergence
    targets on at least three fixed-domain meshes.
12. Perturb `Bx`, double the resulting nonzero `E0`, and require the actual hard
    drift-Ohm gate to fail with raw value and threshold preserved.
13. Verify retained-state finiteness is reduced by the owning backend and that
    unconverted runtime arrays are rejected by NPZ output;
    object-dtype host arrays are rejected before file creation, both for
    ordinary output and explicit runtime failure capture.
14. Verify CLI refusal of CUDA outside Slurm and JSON schema without launching
    GPU work.
15. Compare sampled density on multiple meshes with the same analytic physical
    Z-domain modulation. Require delta-theta norm metadata to use 1/s.
16. For the deliberately wrong-sign Hall oracle, require a finite full-horizon
    production trajectory with hard checks available, then require order-one
    disagreement with that oracle. A reference-sign error does not require the
    correct production solver to reject a valid state. Do not use an incidental
    pressure-roundoff failure as the oracle comparison.

Every test names the production defect or harness contract it catches. Tests
that cannot run a requested CUDA case report it as unavailable; they do not
substitute a CPU pass.
