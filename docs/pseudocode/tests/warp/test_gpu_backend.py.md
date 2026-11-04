# Test gpu backend

Source: `tests/warp/test_gpu_backend.py`.

## A0-PY-TEST-TEST-GPU-BACKEND-01 - Procedure and assertions

```text
Skip unless FOMHD_TEST_GPU=1; then require SLURM_JOB_ID and construct cuda:0.
On a nondefault CuPy stream execute actual shared-storage Warp axpy and
physical-real validation; compare known complex output and reject nonfinite
or nonreal physical values.
Round-trip the Fourier transform; solve an actual complex dense system and
check its residual; reject a singular GPU factor.
Build CPU and CUDA operators for the same mesh and compare the interpolation,
constraint, curl, divergence and potential matrices at declared tolerances.
Host transfers occur only at explicit comparison points.
```

Covered top-level functions and fixture classes:

- `gpu`
- `test_warp_kernel_and_nonreal_rejection`
- `test_gpu_fft_and_rank_checked_solve`
- `test_gpu_compatible_operators_match_cpu`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.
