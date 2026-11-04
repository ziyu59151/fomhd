# Foundation types backend

Source: `tests/warp/foundation_types_backend.py`.

## A0-PY-TEST-FOUNDATION-TYPES-BACKEND-01 - Procedure and assertions

```text
Check dataclass field order, defaults, SI fixture constants and Failure payloads.
Construct explicit CPU Backend; require contiguous complex128 arrays.
In an isolated interpreter, import CPU backend without loading Warp/CuPy;
compile and run the actual Warp kernels on the explicit Warp CPU device.
Verify the bad-element flag begins at int32 maximum, complex storage views
share real/imaginary pairs, and reject invalid dtype/layout/device storage.
Compare axpy, analytic Fourier transforms and antiadjoint derivatives.
Solve a nonsymmetric complex system with row/column scaling; check residual,
rank spectrum, factor reuse and at most five refinement corrections.
Reject singular/invalid factors and stagnation; zero RHS has zero backward
error without eager zero division or warning dependence.
Check deterministic independent-row choices, device-owned spectrum handling,
and device-memory budgeting using narrow backend contract doubles.
These doubles verify API ownership, not actual CUDA execution.
```

Covered top-level functions and fixture classes:

- `test_native_field_order_and_defaults`
- `test_parameters_fixture_matches_native_constants`
- `test_failure_carries_code_and_message`
- `test_cpu_backend_is_explicit_and_preserves_complex_contiguous_storage`
- `test_cpu_import_does_not_eagerly_load_warp_or_cupy`
- `test_warp_kernels_compile_and_execute_on_warp_cpu_device`
- `test_physical_bad_buffer_starts_at_int32_max`
- `_DeviceArray`
- `_device_array`
- `test_cuda_kernel_storage_validation_and_float64_view`
- `test_axpy_and_analytic_fft`
- `test_derivative_is_antiadjoint`
- `test_scaled_solve_reuses_one_factor_and_checks_rank`
- `test_singular_and_invalid_solves_fail`
- `test_independent_rows_are_deterministic`
- `test_singular_lu_factorization_is_rejected_without_warning_mode`
- `test_zero_rhs_backward_error_has_zero_row_semantics_without_warning`
- `_NoWhereXp`
- `test_backward_error_uses_portable_safe_denominator`
- `_StagnatingBackend`
- `test_numpy_boolean_stagnation_is_reported_as_failure`
- `test_spectrum_checks_do_not_transfer_spectrum_to_host`
- `_CudaBudgetView`
- `test_gpu_like_allocation_consumes_device_memory_limit`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.
