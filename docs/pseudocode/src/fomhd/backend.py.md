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

1. Process rows in index order. Ignore exactly zero rows.
2. Normalize each candidate set by its independently computed `hypot` row norm.
3. Compute singular values of the normalized candidates on the selected
   backend. Increase rank only when one singular value exceeds
   `rank_rtol*largest`. Count the rank with a device reduction and read only
   that scalar result.
4. Reject a rank jump other than zero or one, and return the deterministic
   selected row indices.
