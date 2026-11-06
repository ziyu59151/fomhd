# `src/fomhd/backend.py` implementation pseudocode

## A0-PY-BACK-01 Device selection

1. `Backend("cpu")` selects NumPy explicitly.
2. `Backend("cuda:N")` imports CuPy and Warp lazily. Warp kernels are defined and
   compiled only after a CUDA backend is requested.
3. CPU never falls back to CUDA. CUDA never routes state arrays through NumPy.
4. Every allocated numerical array is contiguous complex128. Geometry and
   operator weights are contiguous float64.
5. Before exposing CUDA storage to Warp, require the exact complex128 dtype,
   C-contiguous layout, and the same CUDA device as the backend. View the
   complex storage as float64 with shape `(n,2)` and retain that view until the
   stream operation completes.

## A0-PY-BACK-02 Basic algebra

1. `mv(A,x) = A @ x`; `mm(A,B) = A @ B`.
2. `axpy(a,b,alpha) = a + alpha*b`.
3. On CUDA, `axpy` launches a Warp FP64 kernel over two-component FP64 complex
   storage. It uses an explicit Warp stream wrapping the current CuPy stream.
   The launch is retained in that stream; no whole-array host transfer occurs.

## A0-PY-BACK-03 Fourier transform

1. Validate `K>=0`, `Nphi` odd, `Nphi>=4K+1`, and nonempty entities.
2. Retained mode `m` occupies transform bin `m mod Nphi`.
3. Inverse performs an unscaled backward FFT, matching FFTW sign and scaling.
4. Forward performs a forward FFT and divides retained bins by `Nphi`.
5. Derivative transforms the complete sample set, multiplies bin `q` by
   `i*q` (where bins above `(Nphi-1)/2` are negative), and inverse-transforms.
6. A physical-sample operation validates finiteness and imaginary residue in a
   Warp FP64 kernel on CUDA, writes zero imaginary parts, and raises `Failure`
   with the lowest violating index when any value is NaN, infinite, or exceeds
   the residue bound.
7. Initialize the atomic bad-index buffer to the maximum int32 value. An
   untouched buffer means success; an atomic minimum below the sentinel
   records the lowest invalid sample.

## A0-PY-BACK-04 Scaled dense solve

1. Require nonempty square finite `A`, matching vectors, strictly positive
   finite row/column scales, `0<rank_rtol<1`, and positive finite
   `residual_rtol`.
2. Form `As = diag(1/row_scale) A diag(column_scale)` and
   `bs = b/row_scale`.
3. Compute singular values in nonincreasing order. Reject an empty spectrum,
   zero largest value, or smallest value `<= rank_rtol*largest`. Use device
   reductions and endpoint scalars; do not copy the spectrum to the host.
4. Factor `As` once with LU and reuse that factor. CPU uses NumPy SVD and
   SciPy `lu_factor`/`lu_solve`; CUDA uses CuPy SVD and
   `cupyx.scipy.linalg.lu_factor`/`lu_solve`. On CPU, promote
   `LinAlgWarning` to failure and explicitly reject a zero diagonal pivot even
   when warnings are not globally fatal.
5. Evaluate componentwise scaled backward error
   `max_r |As y-bs|_r / (|bs_r| + sum_j |As_rj||y_j|)`. Replace a zero
   denominator by one before division, then explicitly select the numerator
   defect where the original denominator is zero. This is portable to CuPy
   14.2, which does not accept the `where=` argument used by `xp.divide`.
6. If the error exceeds tolerance, perform a residual correction using the
   same factor. Stop after at most five corrections. Reject nonfinite,
   stagnated, or nondecreasing residual sequences. Convert array equality to
   a host boolean with `bool(...)` so NumPy and CuPy scalar behavior agree.
7. Recover `x=diag(column_scale)y`, compute the original residual, and return
   a `SimpleNamespace` containing every field in the shared contract.

## A0-PY-BACK-05 Independent rows

1. Validate the selected backend array, finite contents, and
   `0<rank_rtol<1`.
2. Scan rows in index order and retain the indices of rows whose existing
   per-row `linalg.norm` is nonzero.
3. If there are no nonzero rows, return an empty index list without allocating
   a normalized matrix or calling SVD. If the number of nonzero rows exceeds
   the number of columns, skip the full-rank probe and go directly to the
   sequential fallback because full row rank is impossible.
4. Otherwise allocate one normalized matrix on the selected backend with
   `complex128` storage. For each retained row, compute the same per-row
   `linalg.norm` and division used by the sequential path, and place the
   normalized row in its original order.
5. Compute one SVD of the entire normalized matrix. The set is proved to have
   full row rank when every singular value is strictly greater than
   `rank_rtol*sigma_max`.
6. If proved, return all nonzero original row indices in order. For every
   prefix of those normalized rows, singular-value interlacing gives
   `sigma_min(prefix)>=sigma_min(full)` and
   `sigma_max(prefix)<=sigma_max(full)`. The full-rank result therefore implies
   that every prefix passes the existing strict test, so the old sequential
   algorithm would select every nonzero row.
7. If the probe does not prove full rank, process rows in original index order
   with the existing sequential algorithm unchanged. Recompute each per-row
   normalization, skip exactly zero rows, compute each prefix spectrum, increase
   rank only for a singular value strictly greater than
   `rank_rtol*largest`, reject a rank jump other than zero or one, and return
   the deterministic selected indices.
8. Keep validation and library failures visible. A failed probe is not treated
   as a negative result, and the fallback does not relax tolerances, reorder
   rows, pivot, or cache a result for another matrix. CUDA arrays and spectra
   remain device-resident while host code reads only the scalar decision.

Release the full-rank probe workspace before entering an unsuccessful-gate
fallback so it does not remain live alongside sequential SVD candidates.

## A0-PY-BACK-06 Prepared scaled solve

1. `prepare_linear(A,row_scale,column_scale,rank_rtol,residual_rtol)` performs
   the A0-PY-BACK-04 validation, forms the fixed scaled matrix, computes its
   singular spectrum, and factors it once with the selected library. It stores
   the original/scaled matrices, scales, tolerances, spectrum, LU factors,
   pivots, and factor validity for later right-hand sides.
2. `linear_prepared(prepared,rhs)` validates the right-hand side, forms only
   `rhs/row_scale`, and applies the same LU solve, componentwise backward-error
   test, and at-most-five residual corrections used by `linear`.
3. It returns a fresh `linear_record` with that right-hand side's `x`,
   residual, scaled residual, backward error, and refinement count. The cached
   spectrum, scales, and factorization replace every fixed-system
   reconstruction without changing any existing field or tolerance.
4. `linear` is the compatibility composition
   `linear_prepared(prepare_linear(...),rhs)`. Bad check input,
   ill-conditioning, singular LU, nonfinite solves, and stalled refinement
   remain explicit `Failure` paths; no fallback backend is attempted.
