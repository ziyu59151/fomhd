"""Explicit NumPy/CuPy backend with Warp pointwise CUDA kernels."""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np

from .types import Failure, linear_record, lu_record


_PHYSICAL_BAD_SENTINEL = int(np.iinfo(np.int32).max)


class Backend:
    """Own numerical arrays for one explicit CPU or CUDA backend."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.name = "cpu" if device == "cpu" else "cuda"
        self._cuda = device != "cpu"
        self._warp = None
        self._wp_device = None
        self._warp_kernels = None
        if self._cuda:
            if not device.startswith("cuda:"):
                raise Failure("Backend", f"unsupported device {device!r}")
            try:
                import cupy as cp
                import warp as wp
            except Exception as error:
                raise Failure("Backend", f"CUDA backend unavailable: {error}") from error
            self.xp = cp
            self._warp = wp
            try:
                wp.init()
                self._wp_device = wp.get_device(device)
                wp.set_device(self._wp_device)
                cp.cuda.Device(self._wp_device.ordinal).use()
            except Exception as error:
                raise Failure("Backend", f"CUDA device setup failed: {error}") from error
            self._warp_kernels = self._make_warp_kernels()
        elif device == "cpu":
            self.xp = np
        else:
            raise Failure("Backend", f"unsupported device {device!r}")

    def _make_warp_kernels(self) -> SimpleNamespace:
        wp = self._warp

        @wp.kernel
        def axpy_kernel(
            base: wp.array(dtype=wp.vec2d),
            direction: wp.array(dtype=wp.vec2d),
            weight: wp.float64,
            out: wp.array(dtype=wp.vec2d),
        ):
            index = wp.tid()
            value = base[index] + weight * direction[index]
            out[index] = wp.vec2d(value[0], value[1])

        @wp.kernel
        def physical_kernel(
            source: wp.array(dtype=wp.vec2d),
            limit: wp.float64,
            out: wp.array(dtype=wp.vec2d),
            bad_index: wp.array(dtype=wp.int32),
        ):
            index = wp.tid()
            value = source[index]
            real = value[0]
            imag = value[1]
            if (
                not wp.isfinite(real)
                or not wp.isfinite(imag)
                or wp.abs(imag) > limit
            ):
                wp.atomic_min(bad_index, 0, index)
            out[index] = wp.vec2d(real, wp.float64(0.0))

        return SimpleNamespace(axpy=axpy_kernel, physical=physical_kernel)

    def array(self, value: Any, dtype: Any = None) -> Any:
        target = np.complex128 if dtype is None else dtype
        result = self.xp.asarray(value, dtype=target)
        return self.xp.ascontiguousarray(result)

    def real_array(self, value: Any) -> Any:
        return self.array(value, np.float64)

    def asarray(self, value: Any, dtype: Any = np.complex128) -> Any:
        result = self.xp.asarray(value, dtype=dtype)
        return self.xp.ascontiguousarray(result)

    def to_host(self, value: Any) -> np.ndarray:
        if self.is_csr(value):
            self._validate_csr(value)
            if self._cuda:
                return value.get(stream=self.xp.cuda.get_current_stream()).toarray()
            return value.toarray()
        self._require_array(value)
        return np.asarray(value) if not self._cuda else self.xp.asnumpy(value)

    def diagnostic_host(
        self, value: Any, expected_shape: tuple[int, ...]
    ) -> np.ndarray:
        """Read a caller-sized FP64 diagnostic pack in current-stream order."""
        self._require_array(value)
        if value.shape != expected_shape:
            raise Failure("Shape", "diagnostic pack has an unexpected shape")
        if value.dtype != self.xp.float64:
            raise Failure("Backend", "diagnostic pack must use float64")
        self._require_current_device(value)
        if self._cuda:
            return self.xp.asnumpy(
                value, stream=self.xp.cuda.get_current_stream(), blocking=True
            )
        return np.asarray(value)

    def csr(self, host_matrix: Any) -> Any:
        """Install a compact host CSR on the backend selected by the caller."""
        if self._cuda:
            from cupyx.scipy.sparse import csr_matrix

            matrix = csr_matrix(host_matrix)
        else:
            matrix = host_matrix
        if not self.is_csr(matrix):
            raise Failure("Backend", f"expected {self.name} CSR matrix")
        self._validate_csr(matrix)
        return matrix

    def is_csr(self, value: Any) -> bool:
        return getattr(value, "format", None) == "csr" and isinstance(
            value.data, self.xp.ndarray
        )

    def _validate_csr(self, matrix: Any) -> None:
        if matrix.ndim != 2:
            raise Failure("Shape", "CSR matrix must be two-dimensional")
        for array in (matrix.data, matrix.indices, matrix.indptr):
            self._require_array(array, 1)
            self._require_current_device(array)
            if not array.flags.c_contiguous:
                raise Failure("Backend", "CSR storage must be C-contiguous")
        if matrix.data.dtype != self.xp.complex128:
            raise Failure("Backend", "CSR values must use complex128")
        if (
            matrix.indices.dtype not in (self.xp.int32, self.xp.int64)
            or matrix.indptr.dtype != matrix.indices.dtype
        ):
            raise Failure("Backend", "CSR indices must share an int32 or int64 dtype")
        if (
            matrix.indices.size != matrix.data.size
            or matrix.indptr.size != matrix.shape[0] + 1
        ):
            raise Failure("Shape", "CSR buffer extents do not match the matrix")

    def scalar(self, value: Any) -> Any:
        if isinstance(value, (int, float, complex, np.number)):
            return value.item() if hasattr(value, "item") else value
        self._require_array(value)
        return value.item()

    def synchronize(self) -> None:
        if self._cuda:
            self.xp.cuda.get_current_stream().synchronize()

    def _require_array(self, value: Any, ndim: int | None = None) -> None:
        if value is None or not isinstance(value, self.xp.ndarray):
            raise Failure("Backend", f"expected {self.name} array")
        if ndim is not None and value.ndim != ndim:
            raise Failure("Shape", f"expected {ndim}D array, got shape {value.shape}")

    def _require_current_device(self, value: Any) -> None:
        if self._cuda:
            expected = int(self._wp_device.ordinal)
            if value.device.id != expected or self.xp.cuda.runtime.getDevice() != expected:
                raise Failure("Backend", f"expected storage on current backend cuda:{expected}")

    def _finite(self, value: Any) -> bool:
        self._require_array(value)
        return bool(self.xp.all(self.xp.isfinite(value)).item())

    def _require_finite(self, value: Any, message: str) -> None:
        if not self._finite(value):
            raise Failure("Finite", message)

    def _external_stream(self):
        wp = self._warp
        cp = self.xp
        return wp.Stream(
            device=self._wp_device,
            cuda_stream=cp.cuda.get_current_stream().ptr,
        )

    def _physical_bad_buffer(self) -> Any:
        return self.xp.full(
            (1,),
            _PHYSICAL_BAD_SENTINEL,
            dtype=self.xp.int32,
        )

    def _validate_kernel_storage(self, value: Any) -> None:
        if not isinstance(value, self.xp.ndarray):
            raise Failure("Kernel", f"expected {self.name} array")
        if value.dtype != self.xp.complex128:
            raise Failure("Kernel", "Warp kernel storage must be complex128")
        if not bool(value.flags.c_contiguous):
            raise Failure("Kernel", "Warp kernel storage must be C-contiguous")
        if self._cuda:
            actual = int(value.device.id)
            expected = int(self._wp_device.ordinal)
            if actual != expected:
                raise Failure(
                    "Kernel",
                    f"array device cuda:{actual} does not match backend cuda:{expected}",
                )

    def _kernel_storage_view(self, value: Any) -> Any:
        self._validate_kernel_storage(value)
        storage = value.view(self.xp.float64).reshape(-1, 2)
        if storage.size != 2 * value.size:
            raise Failure("Kernel", "complex storage view has an invalid size")
        return storage

    def _warp_array(self, value: Any) -> tuple[Any, Any]:
        storage = self._kernel_storage_view(value)
        warp_array = self._warp.array(
            storage,
            dtype=self._warp.vec2d,
            device=self._wp_device,
            copy=False,
        )
        return warp_array, storage

    def mv(self, matrix: Any, vector: Any) -> Any:
        self._require_array(vector)
        if self.is_csr(matrix):
            self._validate_csr(matrix)
            self._require_current_device(vector)
        else:
            self._require_array(matrix)
        if matrix.ndim != 2 or vector.ndim != 1 or matrix.shape[1] != vector.size:
            raise Failure("Shape", "matrix-vector shape mismatch")
        return self.asarray(matrix @ vector)

    def mm(self, left: Any, right: Any) -> Any:
        self._require_array(right)
        if self.is_csr(left):
            self._validate_csr(left)
            self._require_current_device(right)
        else:
            self._require_array(left)
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
            raise Failure("Shape", "matrix-matrix shape mismatch")
        return self.asarray(left @ right)

    def axpy(self, base: Any, direction: Any, weight: float) -> Any:
        self._require_array(base)
        self._require_array(direction)
        if base.shape != direction.shape:
            raise Failure("Shape", "axpy shape mismatch")
        if not math.isfinite(weight):
            raise Failure("Time", "nonfinite axpy weight")
        if base.size == 0:
            return self.array(base)
        if not self._cuda:
            return self.asarray(base + weight * direction)
        wp = self._warp
        source_a, owner_a = self._warp_array(base)
        source_b, owner_b = self._warp_array(direction)
        out = self.array(self.xp.empty_like(base))
        out_w, owner_out = self._warp_array(out)
        stream = self._external_stream()
        try:
            wp.launch(
                self._warp_kernels.axpy,
                dim=base.size,
                inputs=[source_a, source_b, float(weight)],
                outputs=[out_w],
                device=self._wp_device,
                stream=stream,
            )
            wp.synchronize_stream(stream)
        except Exception as error:
            raise Failure("Warp", f"axpy kernel failed: {error}") from error
        return out

    def physical_samples(self, values: Any, limit: float) -> Any:
        self._require_array(values)
        if not math.isfinite(limit) or limit < 0.0:
            raise Failure("Reference", "invalid physical-sample tolerance")
        if not self._cuda:
            host = np.asarray(values)
            if not np.all(np.isfinite(host)):
                raise Failure("Physical", "non-real physical sample contains nonfinite values")
            bad = np.flatnonzero(np.abs(host.imag) > limit)
            if bad.size:
                index = int(bad[0])
                raise Failure(
                    "Imaginary",
                    f"non-real physical sample at {index}; value={host[index]!r} "
                    f"limit={limit}",
                )
            return self.asarray(host.real)
        wp = self._warp
        source, owner_source = self._warp_array(values)
        out = self.array(self.xp.empty_like(values))
        out_w, owner_out = self._warp_array(out)
        bad = self._physical_bad_buffer()
        stream = self._external_stream()
        try:
            wp.launch(
                self._warp_kernels.physical,
                dim=values.size,
                inputs=[source, float(limit)],
                outputs=[out_w, bad],
                device=self._wp_device,
                stream=stream,
            )
            wp.synchronize_stream(stream)
            index = int(bad[0].item())
        except Exception as error:
            raise Failure("Warp", f"physical-sample kernel failed: {error}") from error
        if index < _PHYSICAL_BAD_SENTINEL:
            value = values[index].item()
            raise Failure(
                "Imaginary",
                f"non-real physical sample at {index}; value={value!r} "
                f"limit={limit}",
            )
        return out

    def inverse_fft(self, values: Any, entities: int, K: int, N: int) -> Any:
        self._validate_fourier(values, entities, K, N, modes=True)
        modes = 2 * K + 1
        retained = values.reshape(modes, entities)
        bins = self.xp.zeros((N, entities), dtype=self.xp.complex128)
        for stored in range(modes):
            bins[(stored - K) % N] = retained[stored]
        return self.asarray(self.xp.fft.ifft(bins, axis=0) * float(N)).reshape(-1)

    def forward_fft(self, values: Any, entities: int, K: int, N: int) -> Any:
        self._validate_fourier(values, entities, K, N, modes=False)
        samples = values.reshape(N, entities)
        full = self.xp.fft.fft(samples, axis=0) / float(N)
        retained = self.xp.empty((2 * K + 1, entities), dtype=self.xp.complex128)
        for stored in range(2 * K + 1):
            retained[stored] = full[(stored - K) % N]
        return self.asarray(retained).reshape(-1)

    def derivative(self, values: Any, entities: int, N: int) -> Any:
        if N <= 0 or entities <= 0:
            raise Failure("Fourier", "invalid derivative dimensions")
        if values.ndim != 1 or values.size != N * entities:
            raise Failure("Fourier", "derivative input shape mismatch")
        self._require_finite(values, "derivative input contains non-finite values")
        samples = values.reshape(N, entities)
        full = self.xp.fft.fft(samples, axis=0)
        signed = self.xp.arange(N, dtype=self.xp.float64)
        signed = self.xp.where(signed <= (N - 1) // 2, signed, signed - N)
        full = full * (1j * signed)[:, None] / float(N)
        return self.asarray(self.xp.fft.ifft(full, axis=0) * float(N)).reshape(-1)

    def _validate_fourier(
        self, values: Any, entities: int, K: int, N: int, modes: bool
    ) -> None:
        if K < 0 or N <= 0 or N % 2 == 0 or N < 4 * K + 1:
            raise Failure("Fourier", "Fourier requires K>=0 and odd N>=4*K+1")
        if entities <= 0:
            raise Failure("Fourier", "Fourier batch must contain at least one entity")
        expected = (2 * K + 1 if modes else N) * entities
        if values.ndim != 1 or values.size != expected:
            raise Failure("Fourier", "retained Fourier size mismatch")
        self._require_finite(values, "retained Fourier input is invalid")

    def singular_values(self, matrix: Any) -> Any:
        self._require_array(matrix, 2)
        self._require_finite(matrix, "SVD input contains non-finite values")
        if min(matrix.shape) == 0:
            return self.real_array([])
        values = self.asarray(self.xp.linalg.svd(matrix, compute_uv=False), np.float64)
        if not self._finite(values):
            raise Failure("Linear", "SVD returned an invalid singular value")
        if bool(self.xp.any(values < -1e-12).item()):
            raise Failure("Linear", "SVD returned an invalid singular value")
        values = self.xp.maximum(values, 0.0)
        if values.size > 1 and bool(
            self.xp.any(
                values[1:] > values[:-1] * (1.0 + 1e-12) + 1e-14
            ).item()
        ):
            raise Failure("Linear", "SVD singular values are not nonincreasing")
        return self.asarray(values, np.float64)

    def factor(self, matrix: Any) -> SimpleNamespace:
        self._require_array(matrix, 2)
        if matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
            raise Failure("Linear", "LU factorization requires a nonempty square matrix")
        self._require_finite(matrix, "LU factorization input contains non-finite values")
        if not self._cuda:
            from scipy.linalg import LinAlgWarning, lu_factor

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", LinAlgWarning)
                    factors, pivots = lu_factor(matrix, check_finite=False)
            except Exception as error:
                raise Failure(
                    "Linear", f"singular matrix in LU factorization: {error}"
                ) from error
            record = lu_record(
                original=self.array(matrix),
                factors=self.asarray(factors),
                pivots=self.asarray(pivots, np.int32),
                valid=True,
            )
        else:
            from cupyx.scipy.linalg import lu_factor

            try:
                factors, pivots = lu_factor(matrix)
            except Exception as error:
                raise Failure("Linear", f"singular matrix in LU factorization: {error}") from error
            record = lu_record(
                original=self.array(matrix),
                factors=self.asarray(factors),
                pivots=self.asarray(pivots, self.xp.int32),
                valid=True,
            )
        if bool(self.xp.any(self.xp.diag(record.factors) == 0.0).item()):
            raise Failure(
                "Linear", "singular matrix in LU factorization at a zero pivot"
            )
        if not self._finite(record.factors):
            raise Failure("Linear", "LU factors contain non-finite values")
        record.solve = lambda rhs: self.solve(record, rhs)
        return record

    def solve(self, factorization: SimpleNamespace, rhs: Any) -> Any:
        if factorization is None or not factorization.valid:
            raise Failure("Linear", "invalid LU factorization")
        self._require_array(rhs)
        if rhs.shape[0] != factorization.original.shape[0]:
            raise Failure("Linear", "LU right-hand side shape or value is invalid")
        self._require_finite(rhs, "LU right-hand side shape or value is invalid")
        if self._cuda:
            from cupyx.scipy.linalg import lu_solve

            try:
                result = lu_solve((factorization.factors, factorization.pivots), rhs)
            except Exception as error:
                raise Failure("Linear", f"LU solve failed: {error}") from error
        else:
            from scipy.linalg import lu_solve

            try:
                result = lu_solve(
                    (factorization.factors, factorization.pivots),
                    rhs,
                    check_finite=False,
                )
            except Exception as error:
                raise Failure("Linear", f"LU solve failed: {error}") from error
        result = self.asarray(result)
        self._require_finite(result, "LU solution contains non-finite values")
        return result

    def linear(
        self,
        matrix: Any,
        rhs: Any,
        row_scale: Any,
        column_scale: Any,
        rank_rtol: float,
        residual_rtol: float = 1e-10,
    ) -> SimpleNamespace:
        self._require_array(matrix, 2)
        self._require_array(rhs, 1)
        if rhs.size != matrix.shape[0]:
            raise Failure(
                "Linear", "checked solve shape, threshold, or tolerance is invalid"
            )
        prepared = self.prepare_linear(
            matrix,
            row_scale,
            column_scale,
            rank_rtol,
            residual_rtol,
        )
        return self.linear_prepared(prepared, rhs)

    def prepare_linear(
        self,
        matrix: Any,
        row_scale: Any,
        column_scale: Any,
        rank_rtol: float,
        residual_rtol: float = 1e-10,
    ) -> SimpleNamespace:
        """Validate and factor a fixed scaled dense system for later solves."""
        self._require_array(matrix, 2)
        self._require_array(row_scale, 1)
        self._require_array(column_scale, 1)
        n = matrix.shape[0]
        if (
            n == 0
            or n != matrix.shape[1]
            or row_scale.size != n
            or column_scale.size != matrix.shape[1]
            or not (0.0 < rank_rtol < 1.0)
            or not math.isfinite(residual_rtol)
            or residual_rtol <= 0.0
        ):
            raise Failure("Linear", "checked solve shape, threshold, or tolerance is invalid")
        self._require_finite(matrix, "checked solve matrix contains non-finite values")
        if bool(self.xp.any(self.xp.imag(row_scale) != 0.0).item()):
            raise Failure("Linear", "row scale must be real")
        if bool(self.xp.any(self.xp.imag(column_scale) != 0.0).item()):
            raise Failure("Linear", "column scale must be real")
        row_scale = self.asarray(self.xp.real(row_scale), np.float64)
        column_scale = self.asarray(self.xp.real(column_scale), np.float64)
        if bool(self.xp.any(row_scale <= 0.0).item()) or not self._finite(row_scale):
            raise Failure("Linear", "row scale must be finite and positive")
        if bool(self.xp.any(column_scale <= 0.0).item()) or not self._finite(column_scale):
            raise Failure("Linear", "column scale must be finite and positive")
        scaled_matrix = self.asarray(
            matrix * column_scale[None, :] / row_scale[:, None]
        )
        if not self._finite(scaled_matrix):
            raise Failure("Linear", "scaled linear system contains non-finite values")
        singular = self.singular_values(scaled_matrix)
        if (
            singular.size == 0
            or float(singular[0].item()) == 0.0
            or float(singular[-1].item())
            <= rank_rtol * float(singular[0].item())
        ):
            raise Failure("Linear", "scaled linear system is singular or ill-conditioned")
        factor = self.factor(scaled_matrix)
        return SimpleNamespace(
            original=self.asarray(matrix),
            scaled_matrix=scaled_matrix,
            row_scale=self.asarray(row_scale, np.float64),
            column_scale=self.asarray(column_scale, np.float64),
            rank_rtol=float(rank_rtol),
            residual_rtol=float(residual_rtol),
            singular_values=self.asarray(singular, np.float64),
            factor=factor,
        )

    def linear_prepared(
        self,
        prepared: SimpleNamespace,
        rhs: Any,
    ) -> SimpleNamespace:
        """Solve one right-hand side using an existing checked scaling and LU."""
        if prepared is None:
            raise Failure("Linear", "prepared scaled system is absent")
        try:
            original = prepared.original
            scaled_matrix = prepared.scaled_matrix
            row_scale = prepared.row_scale
            column_scale = prepared.column_scale
            singular = prepared.singular_values
            factor = prepared.factor
            residual_rtol = float(prepared.residual_rtol)
        except (AttributeError, TypeError, ValueError) as error:
            raise Failure("Linear", "prepared scaled system is invalid") from error
        self._require_array(scaled_matrix, 2)
        self._require_array(original, 2)
        self._require_array(row_scale, 1)
        self._require_array(column_scale, 1)
        self._require_array(singular, 1)
        self._require_array(rhs, 1)
        n = scaled_matrix.shape[0]
        if (
            n == 0
            or n != scaled_matrix.shape[1]
            or original.shape != scaled_matrix.shape
            or row_scale.size != n
            or column_scale.size != n
            or rhs.size != n
            or not math.isfinite(residual_rtol)
            or residual_rtol <= 0.0
            or factor is None
            or not getattr(factor, "valid", False)
        ):
            raise Failure(
                "Linear", "checked solve shape, threshold, or tolerance is invalid"
            )
        if not self._finite(scaled_matrix):
            raise Failure("Linear", "scaled linear system contains non-finite values")
        if not self._finite(row_scale) or bool(
            self.xp.any(row_scale <= 0.0).item()
        ):
            raise Failure("Linear", "row scale must be finite and positive")
        if not self._finite(column_scale) or bool(
            self.xp.any(column_scale <= 0.0).item()
        ):
            raise Failure("Linear", "column scale must be finite and positive")
        self._require_finite(rhs, "checked solve right-hand side contains non-finite values")
        scaled_rhs = self.asarray(rhs / row_scale)
        if not self._finite(scaled_rhs):
            raise Failure("Linear", "scaled linear system contains non-finite values")
        y = self.solve(factor, scaled_rhs)
        previous_error = math.inf
        refinement_steps = 0
        backward_error = math.inf
        scaled_residual = math.inf
        while True:
            residual_vector = self.asarray(scaled_matrix @ y - scaled_rhs)
            scaled_residual = float(self.xp.linalg.norm(residual_vector).item())
            scales = self.xp.abs(scaled_rhs) + self.xp.sum(
                self.xp.abs(scaled_matrix) * self.xp.abs(y)[None, :], axis=1
            )
            defects = self.xp.abs(residual_vector)
            safe_scales = self.xp.where(scales > 0.0, scales, 1.0)
            componentwise = self.xp.where(
                scales > 0.0, defects / safe_scales, defects
            )
            backward_error = float(self.xp.max(componentwise).item())
            if not math.isfinite(backward_error):
                raise Failure("Linear", "linear refinement backward error is not finite")
            if backward_error <= residual_rtol:
                break
            if refinement_steps == 5 or backward_error >= previous_error:
                raise Failure(
                    "Linear",
                    "scaled componentwise backward error exceeded its tolerance",
                )
            previous_error = backward_error
            correction = self.solve(factor, self.asarray(-residual_vector))
            following = self.asarray(y + correction)
            if not self._finite(following) or bool(self.xp.array_equal(following, y)):
                raise Failure("Linear", "linear refinement is non-finite or stagnated")
            y = following
            refinement_steps += 1
        x = self.asarray(column_scale * y)
        residual = self.asarray(original @ x - rhs)
        if (
            not self._finite(x)
            or not self._finite(residual)
            or not math.isfinite(scaled_residual)
            or not math.isfinite(backward_error)
            or backward_error > residual_rtol
        ):
            raise Failure(
                "Linear", "scaled componentwise backward error exceeded its tolerance"
            )
        return linear_record(
            x=x,
            residual=residual,
            singular_values=singular,
            scaled_residual=scaled_residual,
            backward_error=backward_error,
            row_scale=row_scale,
            column_scale=column_scale,
            refinement_steps=refinement_steps,
        )

    def independent_rows(self, matrix: Any, rank_rtol: float) -> list[int]:
        self._require_array(matrix, 2)
        if not (0.0 < rank_rtol < 1.0):
            raise Failure("Linear", "independent row input is invalid")
        self._require_finite(matrix, "independent row input is invalid")
        nonzero_rows: list[tuple[int, float]] = []
        for row in range(matrix.shape[0]):
            magnitude = float(self.xp.linalg.norm(matrix[row]).item())
            if magnitude == 0.0:
                continue
            nonzero_rows.append((row, magnitude))
            if len(nonzero_rows) > matrix.shape[1]:
                break
        if 0 < len(nonzero_rows) <= matrix.shape[1]:
            normalized = self.xp.empty(
                (len(nonzero_rows), matrix.shape[1]),
                dtype=self.xp.complex128,
            )
            for offset, (row, magnitude) in enumerate(nonzero_rows):
                normalized[offset] = matrix[row] / magnitude
            values = self.singular_values(normalized)
            if values.size and bool(
                self.xp.all(
                    values > rank_rtol * float(values[0].item())
                ).item()
            ):
                # A0-PY-BACK-05: singular-value interlacing proves that every
                # normalized prefix passes the same strict rank test.
                return [row for row, _ in nonzero_rows]
            del normalized, values
        selected: list[int] = []
        current_rank = 0
        for row in range(matrix.shape[0]):
            magnitude = float(self.xp.linalg.norm(matrix[row]).item())
            if magnitude == 0.0:
                continue
            candidate = self.xp.empty(
                (len(selected) + 1, matrix.shape[1]), dtype=self.xp.complex128
            )
            for offset, source in enumerate(selected):
                source_norm = float(self.xp.linalg.norm(matrix[source]).item())
                candidate[offset] = matrix[source] / source_norm
            candidate[len(selected)] = matrix[row] / magnitude
            values = self.singular_values(candidate)
            new_rank = (
                int(
                    self.xp.count_nonzero(
                        values > rank_rtol * float(values[0].item())
                    ).item()
                )
                if values.size
                else 0
            )
            if new_rank not in (current_rank, current_rank + 1):
                raise Failure("Linear", "independent-row rank is ambiguous")
            if new_rank == current_rank + 1:
                selected.append(row)
                current_rank = new_rank
        return selected

    def max_abs(self, value: Any) -> float:
        self._require_array(value)
        self._require_finite(value, "maximum norm requires finite input")
        return float(self.xp.max(self.xp.abs(value)).item()) if value.size else 0.0
