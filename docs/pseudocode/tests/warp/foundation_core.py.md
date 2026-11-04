# Foundation core

Source: `tests/warp/foundation_core.py`.

## A0-PY-TEST-FOUNDATION-CORE-01 - Procedure and assertions

```text
Build the small CPU Context using foundation parameters and real operators.
Allocate mode-major state; check scalar, three-component momentum, magnetic
and absent ion-parallel shapes; enforce memory caps.
Combine states into a new step/generation/time identity and check retained
Hermitian symmetry; reject incompatible arrays and nonphysical spectra.
Compare per-mode and per-sample matrix application with explicit products.
Invert a Hermitian retained spectrum and require exactly real physical samples;
allow complex trial vectors only through the explicit nonphysical path.
Check reproducible integer-step/midpoint times and reject invalid midpoint
representability or arguments.
```

Covered top-level functions and fixture classes:

- `context`
- `test_allocate_layout_and_memory_rejection`
- `test_combine_and_retained_hermitian_checks`
- `test_mode_and_sample_apply_shapes`
- `test_physical_inverse_accepts_hermitian_and_rejects_residue`
- `test_inverse_default_is_physical_and_rejects_imaginary_residue`
- `test_reproducible_times_and_invalid_midpoint`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.
