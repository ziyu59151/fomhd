"""Batched massless-electron/field right-hand side for the T2 fixed-P unit.

The implementation mirrors the selected A0 massless branch only.  It does not
evaluate the ion RHS, diagnostics, time integration, or any finite-inertia
field solve.
"""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

from fomhd.types import Failure
from fomhd.wall_cache import _build_wall_cache, _cache_key


_EPS = np.finfo(np.float64).eps


def _modes(context) -> int:
    return 2 * context.p.K + 1


def _electroweak_parameters(context) -> tuple[float, float]:
    ion = context.p.species[0]
    electron = context.p.species[1]
    e = -float(electron.charge)
    if float(ion.charge) <= 0.0:
        raise Failure("Model", "the primary ion charge must be positive")
    if not math.isfinite(e) or e <= 0.0:
        raise Failure("Model", "the electron charge must be negative")
    if not math.isfinite(float(electron.mass)) or float(electron.mass) <= 0.0:
        raise Failure("Model", "physical electron mass metadata must be positive")
    return e, float(ion.charge) / e


class FieldBatch:
    """Evaluate only the selected massless field RHS for independent time rows.

    P has shape ``(B, 2*K+1, 5*Nc)`` with density, radial momentum, azimuthal
    momentum, axial momentum, and ion pressure blocks.  Z has shape
    ``(B, 2*K+1, 2*Nc+Nb)`` with electron perpendicular pressure, electron
    parallel pressure, and magnetic-field blocks.  Every array is complex128
    on the context's selected NumPy or CuPy backend.
    """

    __slots__ = (
        "context",
        "backend",
        "xp",
        "cache",
        "_identity",
        "_wall_residual_records",
        "radius", "m0", "normal_columns", "normal_indices", "normal_values",
        "workspace_limit_bytes", "fixed_storage_bytes", "work", "diagonal_normal",
    )

    def __init__(self, context, *, workspace_limit_bytes=None, structured_normal=True) -> None:
        self.context = context
        self.backend = context.compute
        self.xp = context.compute.xp
        if int(getattr(context.p, "electron_inertia", 0)) != 0:
            raise Failure("Model", "massless branch requires electron_inertia=0")
        _electroweak_parameters(context)
        if context.N != context.p.Nphi:
            raise Failure(
                "Fourier",
                f"context sample count {context.N} differs from Nphi={context.p.Nphi}",
            )
        if context.N % 2 == 0 or context.N < 4 * context.p.K + 1:
            raise Failure(
                "Fourier",
                "Fourier requires odd Nphi >= 4*K+1",
            )
        nc = context.op.Nc
        if (
            context.op.Ne != (
                context.op.DR.shape[1] + nc + context.op.DZ.shape[1]
            )
            or context.op.Nb != context.op.MB.size
        ):
            raise Failure("Layout", "context edge or magnetic block layout is invalid")

        cache = getattr(context, "_wall_system_cache", None)
        if cache is None or cache.key != _cache_key(context):
            # This local immutable cache is owned by the producer.  It avoids
            # attaching state or changing build/solve/evaluation counters.
            cache = _build_wall_cache(context)
        self.cache = cache
        self._wall_residual_records = tuple(
            SimpleNamespace(
                mode=cached.mode,
                rank=len(cached.selected_rows),
                selected_rows=tuple(cached.selected_rows),
                gauge_rows=tuple(cached.gauge_rows),
            )
            for cached in cache.modes
        )
        self._identity = self._version(context)
        xp=self.xp
        self.radius=xp.asarray(context.geometry.cells.R,dtype=xp.float64)
        self.m0=xp.real(xp.asarray(context.op.M0)).astype(xp.float64)
        host_t=self.backend.to_host(xp.asarray(context.op.T))
        columns,indices,values=[],[],[]
        rows=host_t.shape[0]
        for column in np.flatnonzero(np.any(host_t != 0,axis=0)):
            nonzero=np.flatnonzero(host_t[:,column])
            for r in nonzero:
                for s in nonzero:
                    columns.append(column)
                    indices.append(r*rows+s)
                    values.append(host_t[r,column]*host_t[s,column].conjugate())
        self.normal_columns=xp.asarray(columns,dtype=xp.int64)
        self.normal_indices=xp.asarray(indices,dtype=xp.int64)
        self.normal_values=xp.asarray(values,dtype=xp.complex128)
        self.diagonal_normal=bool(structured_normal and all(i//rows==i%rows for i in indices)) if rows else False
        selected_limit=(context.p.device_memory_limit if self.backend.name=="cuda"
                        else context.p.memory_limit)
        self.workspace_limit_bytes=int(selected_limit if workspace_limit_bytes is None else workspace_limit_bytes)
        if self.workspace_limit_bytes<=0:
            raise ValueError("field workspace limit must be positive")
        arrays=[]
        for value in vars(context.op).values():
            arrays.extend(value if isinstance(value,list) else [value])
        self.fixed_storage_bytes=sum(int(getattr(v,"nbytes",0)) for v in arrays)+cache.storage_bytes
        self.work=dict(rhs_batches=0,rhs_vectors=0,normal_systems=0,
                       normal_svd_systems=0,normal_diagonal_systems=0,normal_solves=0,wall_solves=0)

    def workspace_estimate(self,batch):
        """Conservative live-buffer plan, separate from measured pool/RSS peaks."""
        c=self.context
        rows=c.op.T.shape[0]
        entries=len(self.normal_columns)
        parts=dict(fixed=self.fixed_storage_bytes,
                   physical_samples=160*batch*c.N*c.op.Nc*16,
                   retained=12*batch*(2*c.p.K+1)*(2*c.op.Nc+c.op.Nb)*16,
                   wall_normal_and_library_work=12*batch*c.N*rows*(1 if self.diagonal_normal else rows)*16,
                   wall_incidence_pairs=2*batch*c.N*entries*16)
        parts["total"]=sum(parts.values())
        return parts

    def weighted_normal(self,weights):
        """Exact incidence-pair assembly of T diag(weights) T*, with no W*d tile."""
        xp=self.xp
        batch,n,columns=weights.shape
        rows=self.context.op.T.shape[0]
        result=xp.zeros((batch*n,rows*rows),dtype=xp.complex128)
        values=weights.reshape(batch*n,columns)[:,self.normal_columns]*self.normal_values
        index=(xp.arange(batch*n)[:,None],self.normal_indices[None])
        # CuPy's atomic scatter supports float64, not complex128. Retain the
        # complex owner and accumulate both FP64 views without demotion.
        xp.add.at(result.real,index,values.real)
        xp.add.at(result.imag,index,values.imag)
        return result.reshape(batch,n,rows,rows)

    def _diagonal_normal(self,weights):
        xp=self.xp
        batch,n,columns=weights.shape
        rows=self.context.op.T.shape[0]
        diagonal=xp.zeros((batch*n,rows),dtype=xp.float64)
        values=weights.reshape(batch*n,columns)[:,self.normal_columns]*self.normal_values.real
        xp.add.at(diagonal,(xp.arange(batch*n)[:,None],self.normal_indices[None]//rows),values)
        return diagonal.reshape(batch,n,rows)

    def _normal_apply(self,normal,multiplier):
        return (normal*multiplier if normal.ndim==multiplier.ndim else
                self.xp.matmul(normal,multiplier[:,:,:,None])[:,:,:,0])

    @staticmethod
    def _version(context) -> tuple[Any, ...]:
        operation = context.op
        return (
            id(context),
            id(context.p),
            id(context.geometry),
            id(operation),
            id(context.compute),
            str(getattr(context.compute, "name", "")),
            str(getattr(context.compute, "device", "")),
            int(getattr(context, "spectral_id", 0)),
            str(getattr(context, "operator_hash", "")),
            str(getattr(context, "geometry_hash", "")),
            str(getattr(context, "rule_hash", "")),
            int(operation.K),
            int(operation.Nc),
            int(operation.Ne),
            int(operation.Nb),
            _cache_key(context),
            int(context.p.electron_inertia),
            float(context.p.mu0),
            float(context.p.B_min),
            tuple((float(s.mass),float(s.charge),float(s.n_min),float(s.face_min),
                   float(s.p_min),float(s.parallel_min)) for s in context.p.species),
        )

    def _require_identity(self) -> None:
        if self._version(self.context) != self._identity:
            raise Failure(
                "Identity",
                "FieldBatch operand version no longer matches its fixed snapshot",
            )

    def _require_array(self, value: Any, name: str) -> None:
        xp = self.xp
        if not isinstance(value, xp.ndarray):
            raise Failure("Backend", f"{name} must be a {self.backend.name} array")
        if value.dtype != xp.complex128:
            raise Failure("dtype", f"{name} must have complex128 dtype")
        if self.backend.name == "cuda":
            expected = int(self.backend._wp_device.ordinal)
            actual = int(value.device.id)
            if actual != expected:
                raise Failure(
                    "Backend",
                    f"{name} is on cuda:{actual}, expected cuda:{expected}",
                )

    def _validate_inputs(self, p_values: Any, z_values: Any):
        self._require_identity()
        self._require_array(p_values, "P")
        self._require_array(z_values, "Z")
        xp = self.xp
        nc = self.context.op.Nc
        nb = self.context.op.Nb
        modes = _modes(self.context)
        expected_p = (1, modes, 5 * nc)
        expected_z = (1, modes, 2 * nc + nb)
        if (
            p_values.ndim != 3
            or p_values.shape[1:] != expected_p[1:]
            or p_values.shape[0] <= 0
        ):
            raise Failure(
                "Shape",
                f"P layout must be (B,{modes},{5 * nc}) with B>0; "
                f"got {tuple(p_values.shape)}",
            )
        if (
            z_values.ndim != 3
            or z_values.shape[1:] != expected_z[1:]
            or z_values.shape[0] != p_values.shape[0]
        ):
            raise Failure(
                "Shape",
                f"Z layout must be (B,{modes},{2 * nc + nb}) and share B; "
                f"got {tuple(z_values.shape)}",
            )
        if not bool(xp.all(xp.isfinite(p_values)).item()):
            raise Failure("Finite", "P contains non-finite values")
        if not bool(xp.all(xp.isfinite(z_values)).item()):
            raise Failure("Finite", "Z contains non-finite values")
        estimate=self.workspace_estimate(len(p_values))
        if estimate["total"]>self.workspace_limit_bytes:
            failure=Failure("MemoryBudget",f"field batch plan {estimate['total']} exceeds {self.workspace_limit_bytes} bytes")
            failure.memory_estimate=estimate
            raise failure
        return p_values, z_values

    def _ifft(self, retained: Any, entities: int) -> Any:
        xp = self.xp
        batch, modes, actual = retained.shape
        if actual != entities:
            raise Failure("Shape", "retained Fourier entity extent changed")
        N = self.context.N
        K = self.context.p.K
        bins = xp.zeros((batch, N, entities), dtype=xp.complex128)
        index = (xp.arange(modes) - K) % N
        bins[:, index, :] = retained
        return xp.fft.ifft(bins, axis=1) * float(N)

    def _fft(self, samples: Any) -> Any:
        xp = self.xp
        batch, N, entities = samples.shape
        modes = _modes(self.context)
        K = self.context.p.K
        if N != self.context.N:
            raise Failure("Fourier", "sample/time extent differs from context")
        full = xp.fft.fft(samples, axis=1) / float(N)
        index = (xp.arange(modes) - K) % N
        return xp.ascontiguousarray(full[:, index, :])

    def _physical_samples(
        self,
        retained: Any,
        reference: float,
        name: str,
    ) -> Any:
        xp = self.xp
        scale = xp.max(xp.abs(retained), axis=(1, 2))
        limit = 512.0 * _EPS * xp.maximum(scale, max(float(reference), 1.0e-300))
        samples = self._ifft(retained, retained.shape[-1])
        imaginary = xp.max(xp.abs(xp.imag(samples)), axis=(1, 2))
        bad = imaginary > limit
        if bool(xp.any(bad).item()):
            item = int(xp.argmax(bad.astype(xp.int8)).item())
            raise Failure(
                "Imaginary",
                f"{name} batch item {item} is not physically real: "
                f"max_imag={float(imaginary[item].item()):.6e}, "
                f"limit={float(limit[item].item()):.6e}",
            )
        return xp.ascontiguousarray(
            xp.real(samples).astype(xp.complex128)
        )

    def _inverse_physical(
        self,
        retained: Any,
        entities: int,
        name: str,
        reference: float = 0.0,
    ) -> Any:
        if retained.shape[-1] != entities:
            raise Failure("Shape", f"{name} has an invalid entity extent")
        return self._physical_samples(retained, reference, name)

    def _mode_group(self, matrices, values: Any) -> Any:
        xp = self.xp
        batch, modes, columns = values.shape
        if len(matrices) != modes:
            raise Failure("Layout", "mode operator count differs from retained modes")
        rows = matrices[0].shape[0]
        output = xp.empty((batch, modes, rows), dtype=xp.complex128)
        for stored, matrix in enumerate(matrices):
            if matrix.shape[1] != columns:
                raise Failure("Shape", "mode operator columns do not match values")
            output[:, stored, :] = xp.matmul(
                values[:, stored, :], xp.conj(matrix.T)
                if self._conjugate_after_transpose
                else matrix.T
            )
        return output

    @property
    def _conjugate_after_transpose(self) -> bool:
        # Kept as an explicit switch for the one source convention that applies
        # ``I.T.conj()`` to the ion momentum flux.
        return bool(getattr(self, "_conjugate_mode", False))

    def _mode_group_regular(self, matrices, values: Any) -> Any:
        xp = self.xp
        batch, modes, columns = values.shape
        if len(matrices) != modes:
            raise Failure("Layout", "mode operator count differs from retained modes")
        rows = matrices[0].shape[0]
        output = xp.empty((batch, modes, rows), dtype=xp.complex128)
        for stored, matrix in enumerate(matrices):
            if matrix.shape[1] != columns:
                raise Failure("Shape", "mode operator columns do not match values")
            output[:, stored, :] = xp.matmul(values[:, stored, :], matrix.T)
        return output

    def _mode_group_conjugate(self, matrices, values: Any) -> Any:
        xp = self.xp
        batch, modes, columns = values.shape
        if len(matrices) != modes:
            raise Failure("Layout", "mode operator count differs from retained modes")
        rows = matrices[0].shape[0]
        output = xp.empty((batch, modes, rows), dtype=xp.complex128)
        for stored, matrix in enumerate(matrices):
            if matrix.shape[1] != columns:
                raise Failure("Shape", "mode operator columns do not match values")
            output[:, stored, :] = xp.matmul(
                values[:, stored, :], xp.conj(matrix.T)
            )
        return output

    def _same_mode_matrix(self, matrix: Any, values: Any) -> Any:
        xp = self.xp
        if values.shape[-1] != matrix.shape[1]:
            raise Failure("Shape", "mode matrix columns do not match values")
        return xp.matmul(values, matrix.T)

    def _same_mode_matrix_conjugate(self, matrix: Any, values: Any) -> Any:
        xp = self.xp
        if values.shape[-1] != matrix.shape[1]:
            raise Failure("Shape", "mode matrix columns do not match values")
        return xp.matmul(values, xp.conj(matrix.T))

    def _sample_matrix(self, matrix: Any, values: Any) -> Any:
        xp = self.xp
        if values.ndim != 3 or values.shape[1] != self.context.N:
            raise Failure("Shape", "sample operator requires a leading time batch")
        if values.shape[-1] != matrix.shape[1]:
            raise Failure("Shape", "sample matrix columns do not match values")
        return xp.matmul(values, matrix.T)

    def _derivative(self, values: Any) -> Any:
        xp = self.xp
        N = self.context.N
        if values.ndim != 3 or values.shape[1] != N:
            raise Failure("Fourier", "derivative requires a leading time batch")
        signed = xp.arange(N, dtype=xp.float64)
        signed = xp.where(signed <= (N - 1) // 2, signed, signed - N)
        full = xp.fft.fft(values, axis=1)
        full = full * (1j * signed)[None, :, None] / float(N)
        return xp.fft.ifft(full, axis=1) * float(N)

    def _require_positive(
        self,
        samples: Any,
        floor: float,
        name: str,
    ) -> None:
        xp = self.xp
        real = xp.real(samples)
        finite = xp.all(xp.isfinite(real), axis=(1, 2))
        positive = xp.all(real > float(floor), axis=(1, 2))
        bad = ~(finite & positive)
        if bool(xp.any(bad).item()):
            item = int(xp.argmax(bad.astype(xp.int8)).item())
            minimum = float(xp.min(real[item]).item())
            raise Failure(
                "Domain",
                f"{name} batch item {item} must exceed {float(floor):.6e}; "
                f"minimum={minimum:.6e}",
            )

    def _component_flux(self, scalar: Any) -> Any:
        xp = self.xp
        nc = self.context.op.Nc
        batch, N, actual = scalar.shape
        if N != self.context.N or actual != nc:
            raise Failure("Shape", "component-flux scalar layout is invalid")
        nodes = xp.repeat(scalar[:, :, None, :], 3, axis=2)
        return self._sample_matrix(
            self.context.op.I,
            nodes.reshape(batch, N, 3 * nc),
        )

    def _sampled_divergence(self, edges: Any) -> Any:
        xp = self.xp
        nc = self.context.op.Nc
        ne = self.context.op.Ne
        nr = self.context.op.DR.shape[1]
        nz = self.context.op.DZ.shape[1]
        if nr + nc + nz != ne:
            raise Failure("Layout", "edge component offsets are inconsistent")
        N = self.context.N
        if edges.ndim != 3 or edges.shape[1:] != (N, ne):
            raise Failure("Shape", "edge divergence layout is invalid")
        radial = self._sample_matrix(
            self.context.op.DR,
            edges[:, :, :nr],
        )
        azimuthal = self._derivative(edges[:, :, nr : nr + nc])
        axial = self._sample_matrix(
            self.context.op.DZ,
            edges[:, :, nr + nc :],
        )
        radius = self.radius
        return radial + axial + azimuthal / radius[None, None, :]

    def _velocity_gradient(
        self,
        density: Any,
        face_density: Any,
        velocity: Any,
        context=None,
    ) -> Any:
        del context
        xp = self.xp
        nc = self.context.op.Nc
        ne = self.context.op.Ne
        N = self.context.N
        batch = velocity.shape[0]
        if density.shape != (batch, N, nc):
            raise Failure("Shape", "velocity-gradient density layout is invalid")
        if face_density.shape != (batch, N, ne):
            raise Failure("Shape", "velocity-gradient face density layout is invalid")
        if velocity.shape != (batch, N, 3, nc):
            raise Failure("Shape", "velocity-gradient input layout is invalid")
        radius = self.radius
        gradient = xp.zeros(
            (batch, N, 3, 3, nc), dtype=xp.complex128
        )
        nr = self.context.op.DR.shape[1]
        axial_offset = nr + nc
        for component in range(3):
            component_values = velocity[:, :, component, :]
            derivative_phi = self._derivative(component_values)
            weighted = xp.tile(density * component_values, (1, 1, 3))
            interpolated = self._sample_matrix(self.context.op.I, weighted)
            radial_edges = interpolated[:, :, :nr] / face_density[:, :, :nr]
            axial_edges = (
                interpolated[:, :, axial_offset:] /
                face_density[:, :, axial_offset:]
            )
            radial = self._sample_matrix(self.context.op.DR, radial_edges)
            axial = self._sample_matrix(self.context.op.DZ, axial_edges)
            gradient[:, :, component, 0, :] = (
                radial - component_values / radius[None, None, :]
            )
            gradient[:, :, component, 1, :] = (
                derivative_phi / radius[None, None, :]
            )
            gradient[:, :, component, 2, :] = axial
        gradient[:, :, 0, 1, :] -= (
            velocity[:, :, 1, :] / radius[None, None, :]
        )
        gradient[:, :, 1, 1, :] += (
            velocity[:, :, 0, :] / radius[None, None, :]
        )
        return gradient

    def _gradient_adjoint(
        self,
        density: Any,
        face_density: Any,
        tensor: Any,
    ) -> Any:
        xp = self.xp
        nc = self.context.op.Nc
        ne = self.context.op.Ne
        N = self.context.N
        batch = tensor.shape[0]
        radius = self.radius
        m0 = self.m0
        nr = self.context.op.DR.shape[1]
        axial_offset = nr + nc
        result = xp.zeros((batch, N, 3, nc), dtype=xp.complex128)
        for row in range(3):
            weighted = m0[None, None, None, :] * tensor[:, :, row, :, :]
            radial_rhs = self._sample_matrix(
                xp.conj(self.context.op.DR.T),
                weighted[:, :, 0, :],
            )
            radial_edges = xp.zeros(
                (batch, N, ne), dtype=xp.complex128
            )
            radial_edges[:, :, :nr] = (
                radial_rhs / face_density[:, :, :nr]
            )
            radial_nodes = self._sample_matrix(
                xp.conj(self.context.op.I.T),
                radial_edges,
            ).reshape(batch, N, 3, nc)
            axial_rhs = self._sample_matrix(
                xp.conj(self.context.op.DZ.T),
                weighted[:, :, 2, :],
            )
            axial_edges = xp.zeros(
                (batch, N, ne), dtype=xp.complex128
            )
            axial_edges[:, :, axial_offset:] = (
                axial_rhs / face_density[:, :, axial_offset:]
            )
            axial_nodes = self._sample_matrix(
                xp.conj(self.context.op.I.T),
                axial_edges,
            ).reshape(batch, N, 3, nc)
            azimuthal_nodes = density * (
                radial_nodes[:, :, 0, :] + axial_nodes[:, :, 2, :]
            )
            azimuthal_rhs = weighted[:, :, 1, :] / radius[None, None, :]
            result[:, :, row, :] = (
                azimuthal_nodes
                - weighted[:, :, 0, :] / radius[None, None, :]
                - self._derivative(azimuthal_rhs)
            ) / m0[None, None, :]
            if row == 0:
                result[:, :, row, :] += (
                    tensor[:, :, 1, 1, :] / radius[None, None, :]
                )
            elif row == 1:
                result[:, :, row, :] -= (
                    tensor[:, :, 0, 1, :] / radius[None, None, :]
                )
        return result

    def _pressure_tensor(
        self,
        perp: Any,
        parallel: Any,
        bhat: Any,
    ) -> Any:
        xp = self.xp
        batch, N, nc = perp.shape
        direction = bhat.reshape(batch, N, 3, nc)
        identity = xp.eye(3, dtype=xp.complex128).reshape(
            1, 1, 3, 3, 1
        )
        tensor = perp[:, :, None, None, :] * identity
        tensor = tensor + (
            (parallel - perp)[:, :, None, None, :]
            * direction[:, :, :, None, :]
            * xp.conj(direction[:, :, None, :, :])
        )
        return tensor

    def _transport_flux(
        self,
        pressure: Any,
        face_density: Any,
        flux: Any,
    ) -> Any:
        pressure_edges = self._component_flux(pressure)
        return pressure_edges / face_density * flux

    def _constraint_project(
        self,
        source: Any,
        density: Any,
    ) -> SimpleNamespace:
        xp = self.xp
        nc = self.context.op.Nc
        N = self.context.N
        if (
            source.ndim == 4
            and source.shape[0:2] == (density.shape[0], N)
            and source.shape[2:] == (3, nc)
        ):
            source = source.reshape(source.shape[0], N, 3 * nc)
        if source.ndim != 3 or source.shape[2] != 3 * nc:
            raise Failure("Shape", "constraint source must have three components")
        batch = source.shape[0]
        matrix = xp.asarray(self.context.op.T, dtype=xp.complex128)
        if matrix.shape[0] == 0:
            zeros = xp.zeros((batch, N, 0), dtype=xp.complex128)
            return SimpleNamespace(
                value=source.copy(),
                residual=zeros,
                full_residual=xp.zeros((batch, 0), dtype=xp.complex128),
                rank_margin=xp.ones((batch, N), dtype=xp.float64),
            )
        m0 = self.m0
        density_real = xp.real(density)
        winv = xp.tile(density_real / m0[None, None, :], (1, 1, 3))
        matrix_h = xp.conj(matrix.T)
        normal = self._diagonal_normal(winv) if self.diagonal_normal else self.weighted_normal(winv)
        self.work["normal_systems"] += batch*N
        right = xp.matmul(source, matrix.T)

        length = float(self.context.p.ref.length)
        row_scale = 1.0 / length**3
        column_scale = 1.0 / length**3
        scaled = normal * (column_scale / row_scale)
        if self.diagonal_normal:
            largest=xp.max(scaled,axis=2)
            smallest=xp.min(scaled,axis=2)
            self.work["normal_diagonal_systems"] += batch*N
        else:
            singular = xp.linalg.svd(scaled, compute_uv=False)
            self.work["normal_svd_systems"] += batch*N
            largest = singular[:, :, 0]
            smallest = singular[:, :, -1]
        deficient = largest <= 0.0
        deficient |= smallest <= self.context.p.rank_rtol * largest
        if bool(xp.any(deficient).item()):
            flat = int(xp.argmax(deficient.reshape(-1).astype(xp.int8)).item())
            item, sample = divmod(flat, N)
            raise Failure(
                "WallRank",
                "selected wall reconstruction rows are deficient at "
                f"batch item {item}, time row {sample}",
            )

        def solve_normal(value):
            return (value/scaled if self.diagonal_normal else
                    xp.linalg.solve(scaled,value[:,:,:,None])[:,:,:,0])
        multiplier = solve_normal(right)
        self.work["normal_solves"] += batch*N
        residual = self._normal_apply(normal,multiplier) - right
        limit = self._constraint_limit(normal, multiplier, right, 512.0)
        active = xp.any(xp.abs(residual) > limit, axis=2)
        for _ in range(5):
            if not bool(xp.any(active).item()):
                break
            correction = solve_normal(-residual/row_scale)*column_scale
            self.work["normal_solves"] += batch*N
            multiplier = xp.where(
                active[:, :, None],
                multiplier + correction,
                multiplier,
            )
            residual = (
                self._normal_apply(normal,multiplier)
                - right
            )
            limit = self._constraint_limit(normal, multiplier, right, 512.0)
            active = xp.any(xp.abs(residual) > limit, axis=2)
        if bool(xp.any(active).item()):
            raise Failure(
                "WallProjection",
                "electron flux constraint refinement did not converge",
            )

        correction = winv * xp.matmul(
            matrix_h[None, None, :, :],
            multiplier[:, :, :, None],
        )[:, :, :, 0]
        output = source - correction
        final_residual = (
            self._normal_apply(normal,multiplier)
            - right
        )
        final_limit = self._constraint_limit(
            normal, multiplier, right, 4096.0
        )
        if bool(xp.any(xp.abs(final_residual) > final_limit).item()):
            raise Failure(
                "WallProjection",
                "electron flux constraint residual is too large",
            )

        full_matrix = xp.asarray(
            self.context.op.T_full, dtype=xp.complex128
        )
        full = xp.matmul(output, full_matrix.T)
        output_scale = xp.max(xp.abs(output), axis=(1, 2))
        source_scale = xp.max(xp.abs(source), axis=(1, 2))
        full_limit = (
            4096.0
            * _EPS
            * xp.maximum(xp.maximum(output_scale, source_scale), 1.0e-300)
        )
        full_scale = xp.max(xp.abs(full), axis=(1, 2))
        if bool(xp.any(full_scale > full_limit).item()):
            item = int(xp.argmax((full_scale > full_limit).astype(xp.int8)).item())
            raise Failure(
                "WallProjection",
                f"full wall trace rejects constrained velocity at batch item {item}",
            )
        return SimpleNamespace(
            value=output,
            residual=final_residual,
            full_residual=full,
            rank_margin=smallest / xp.maximum(largest, 1.0e-300),
        )

    def _constraint_limit(
        self,
        normal: Any,
        multiplier: Any,
        right: Any,
        factor: float,
    ) -> Any:
        xp = self.xp
        applied = self._normal_apply(normal,multiplier)
        scale = xp.maximum(
            xp.maximum(
                xp.max(xp.abs(right), axis=2),
                xp.max(xp.abs(applied), axis=2),
            ),
            1.0e-300,
        )
        return factor * _EPS * scale[:, :, None]

    def _wall_correction(self, e0: Any) -> SimpleNamespace:
        xp = self.xp
        batch = e0.shape[0]
        ne = self.context.op.Ne
        correction = xp.zeros((batch, 2 * self.context.p.K + 1, ne), dtype=xp.complex128)
        full_scale = xp.zeros(
            (batch, len(self.cache.modes)), dtype=xp.float64
        )
        for stored, cached in enumerate(self.cache.modes):
            if cached.extraction.shape[0] == 0:
                continue
            source = e0[:, stored, :]
            curled = xp.matmul(source, cached.C.T)
            curl_adjoint = xp.matmul(curled, cached.Cdag.T)
            rhs = -xp.matmul(curl_adjoint, cached.extraction.T)
            multiplier = xp.zeros_like(rhs)
            selected = list(cached.selected_rows)
            if cached.prepared is not None and selected:
                prepared = cached.prepared
                selected_rhs = rhs[:, selected]
                scaled_rhs = (
                    selected_rhs.T / prepared.row_scale[:, None]
                )
                y = self.backend.solve(prepared.factor, scaled_rhs)
                self.work["wall_solves"] += batch
                residual = prepared.scaled_matrix @ y - scaled_rhs
                scales = xp.abs(scaled_rhs) + xp.sum(
                    xp.abs(prepared.scaled_matrix)[:, :, None]
                    * xp.abs(y)[None, :, :],
                    axis=1,
                )
                safe_scales = xp.where(scales > 0.0, scales, 1.0)
                componentwise = xp.where(
                    scales > 0.0,
                    xp.abs(residual) / safe_scales,
                    xp.abs(residual),
                )
                active = xp.any(
                    componentwise > prepared.residual_rtol,
                    axis=0,
                )
                for _ in range(5):
                    if not bool(xp.any(active).item()):
                        break
                    correction_y = self.backend.solve(
                        prepared.factor, -residual
                    )
                    self.work["wall_solves"] += batch
                    y = xp.where(
                        active[None, :], y + correction_y, y
                    )
                    residual = prepared.scaled_matrix @ y - scaled_rhs
                    scales = xp.abs(scaled_rhs) + xp.sum(
                        xp.abs(prepared.scaled_matrix)[:, :, None]
                        * xp.abs(y)[None, :, :],
                        axis=1,
                    )
                    safe_scales = xp.where(scales > 0.0, scales, 1.0)
                    componentwise = xp.where(
                        scales > 0.0,
                        xp.abs(residual) / safe_scales,
                        xp.abs(residual),
                    )
                    active = xp.any(
                        componentwise > prepared.residual_rtol,
                        axis=0,
                    )
                if bool(xp.any(active).item()):
                    raise Failure(
                        "WallField",
                        "batched wall correction refinement did not converge",
                    )
                multiplier[:, selected] = (
                    prepared.column_scale[:, None] * y
                ).T
            residual = xp.matmul(multiplier, cached.matrix.T) - rhs
            rhs_scale = xp.maximum(
                xp.max(xp.abs(rhs), axis=1), 1.0e-300
            )
            relative = xp.max(xp.abs(residual), axis=1) / rhs_scale
            full_scale[:, stored] = relative
            if bool(xp.any(relative > 1.0e-8).item()):
                item = int(xp.argmax((relative > 1.0e-8).astype(xp.int8)).item())
                raise Failure(
                    "WallField",
                    "constrained electric-field correction fails an original "
                    f"wall row at batch item {item}, mode {cached.mode}",
                )
            correction[:, stored, :] = xp.matmul(
                multiplier, cached.weighted_adjoint.T
            )
        return SimpleNamespace(
            correction=correction,
            residual_scale=full_scale,
            records=self._wall_residual_records,
        )

    def _initial_constraints(self, b_values: Any, current: Any) -> SimpleNamespace:
        xp = self.xp
        modes = _modes(self.context)
        nb = self.context.op.Nb
        divergence = self._mode_group_regular(
            self.context.op.DB, b_values
        )
        div_scale = xp.zeros(b_values.shape[0], dtype=xp.float64)
        current_scale = xp.zeros(b_values.shape[0], dtype=xp.float64)
        wall_current_parts = []
        for stored in range(modes):
            db = xp.asarray(self.context.op.DB[stored], dtype=xp.complex128)
            cdag = xp.asarray(
                self.context.op.Cdag[stored], dtype=xp.complex128
            )
            b_scale = xp.max(xp.abs(b_values[:, stored, :]), axis=1)
            value_scale = xp.max(xp.abs(cdag), axis=(0, 1))
            div_scale = xp.maximum(
                div_scale,
                float(xp.max(xp.abs(db)).item()) * b_scale,
            )
            current_scale = xp.maximum(
                current_scale,
                value_scale * b_scale / float(self.context.p.mu0),
            )
            extraction = self.cache.modes[stored].extraction
            wall_current_parts.append(
                xp.matmul(current[:, stored, :], extraction.T)
            )
        divergence_bound = (
            4096.0 * _EPS * div_scale * max(nb, 1)
        )
        divergence_norm = xp.max(
            xp.abs(divergence), axis=(1, 2)
        )
        if bool(xp.any(divergence_norm > divergence_bound).item()):
            item = int(
                xp.argmax(
                    (divergence_norm > divergence_bound).astype(xp.int8)
                ).item()
            )
            raise Failure(
                "InitialCompatibility",
                f"initial magnetic field has nonzero divergence at batch item {item}",
            )
        wall_current = xp.concatenate(wall_current_parts, axis=1)
        current_bound = (
            4096.0 * _EPS * current_scale * max(nb, 1)
        )
        wall_current_norm = (xp.max(xp.abs(wall_current), axis=1)
                             if wall_current.shape[1] else xp.zeros(len(b_values)))
        if bool(xp.any(wall_current_norm > current_bound).item()):
            item = int(
                xp.argmax(
                    (wall_current_norm > current_bound).astype(xp.int8)
                ).item()
            )
            raise Failure(
                "WallCurrent",
                f"initial current violates the full closed-wall constraint "
                f"at batch item {item}",
            )
        return SimpleNamespace(
            divergence=divergence,
            divergence_max=divergence_norm,
            divergence_bound=divergence_bound,
            wall_current=wall_current,
            wall_current_max=wall_current_norm,
            wall_current_bound=current_bound,
        )

    def rhs(self, p_values: Any, z_values: Any) -> SimpleNamespace:
        """Return the massless Z RHS and selected derived/check quantities."""
        p_values, z_values = self._validate_inputs(p_values, z_values)
        self.work["rhs_batches"] += 1
        self.work["rhs_vectors"] += len(p_values)
        xp = self.xp
        context = self.context
        nc = context.op.Nc
        ne = context.op.Ne
        nb = context.op.Nb
        N = context.N
        modes = _modes(context)
        batch = p_values.shape[0]
        p_view = p_values.reshape(batch, modes, 5, nc)
        z_perp = z_values[:, :, :nc]
        z_parallel = z_values[:, :, nc : 2 * nc]
        b_values = z_values[:, :, 2 * nc : 2 * nc + nb]

        bn_spectral = self._same_mode_matrix(
            context.op.QB, b_values
        )
        bn = self._inverse_physical(
            bn_spectral, 3 * nc, "magnetic field"
        ).reshape(batch, N, 3, nc)
        magnitude = xp.sqrt(xp.sum(xp.abs(bn) ** 2, axis=2))
        magnetic_ok = xp.all(
            xp.isfinite(magnitude) & (magnitude > context.p.B_min),
            axis=(1, 2),
        )
        if not bool(xp.all(magnetic_ok).item()):
            item = int(
                xp.argmax((~magnetic_ok).astype(xp.int8)).item()
            )
            minimum = float(xp.min(magnitude[item]).item())
            raise Failure(
                "Magnetization",
                f"nodal B is below B_min at batch item {item}; minimum={minimum:.6e}",
            )
        bhat = bn / magnitude[:, :, None, :]

        electron_perp = self._inverse_physical(
            z_perp, nc, "electron perpendicular pressure"
        ).reshape(batch, N, nc)
        electron_parallel = self._inverse_physical(
            z_parallel, nc, "electron parallel pressure"
        ).reshape(batch, N, nc)
        self._require_positive(
            electron_perp,
            context.p.species[1].p_min,
            "electron perpendicular pressure",
        )
        self._require_positive(
            electron_parallel,
            context.p.species[1].parallel_min,
            "electron parallel pressure",
        )

        ion_density_hat = p_view[:, :, 0, :].reshape(batch, modes, nc)
        ion_momentum = p_view[:, :, 1:4, :].reshape(
            batch, modes, 3 * nc
        )
        ion_density = self._inverse_physical(
            ion_density_hat, nc, "ion density"
        ).reshape(batch, N, nc)
        self._require_positive(
            ion_density,
            context.p.species[0].n_min,
            "ion density",
        )
        ion_momentum_samples = self._inverse_physical(
            ion_momentum, 3 * nc, "ion momentum"
        ).reshape(batch, N, 3, nc)
        ion_face_density = self._component_flux(ion_density)
        self._require_positive(
            ion_face_density,
            context.p.species[0].face_min,
            "ion face density",
        )
        wall_trace = xp.matmul(
            ion_momentum_samples.reshape(batch, N, 3 * nc),
            xp.asarray(context.op.T_full, dtype=xp.complex128).T,
        )
        wall_norm = (xp.max(xp.abs(wall_trace), axis=(1, 2))
                     if wall_trace.shape[-1] else xp.zeros(batch))
        momentum_scale = xp.maximum(
            xp.max(xp.abs(ion_momentum_samples), axis=(1, 2, 3)),
            context.p.species[0].mass
            * context.p.ref.ni
            * context.p.ref.velocity,
        )
        momentum_limit = 4096.0 * _EPS * momentum_scale
        if bool(xp.any(wall_norm > momentum_limit).item()):
            item = int(xp.argmax((wall_norm > momentum_limit).astype(xp.int8)).item())
            raise Failure(
                "WallState",
                f"P momentum violates the full wall trace at batch item {item}",
            )

        ion_flux_hat = self._same_mode_matrix_conjugate(
            context.op.I,
            ion_momentum / float(context.p.species[0].mass),
        )
        current = self._mode_group_regular(context.op.Cdag, b_values) / float(
            context.p.mu0
        )
        initial_constraints = self._initial_constraints(b_values, current)

        electron_charge = float(context.p.species[1].charge)
        e = -electron_charge
        qi = float(context.p.species[0].charge)
        electron_flux_hat = (
            qi * ion_flux_hat - current
        ) / e
        ratio = qi / e
        electron_density_hat = ratio * ion_density_hat
        electron_density = self._inverse_physical(
            electron_density_hat, nc, "electron density"
        ).reshape(batch, N, nc)
        self._require_positive(
            electron_density,
            context.p.species[1].n_min,
            "derived electron density",
        )
        electron_face_density = self._component_flux(electron_density)
        self._require_positive(
            electron_face_density,
            context.p.species[1].face_min,
            "derived face electron density",
        )
        electron_flux = self._inverse_physical(
            electron_flux_hat, ne, "electron flux"
        ).reshape(batch, N, ne)
        h_flux = self._sample_matrix(
            context.op.H,
            electron_flux,
        )
        constraint = self._constraint_project(
            h_flux, electron_density
        )
        constrained_momentum = constraint.value.reshape(batch, N, 3, nc)
        electron_velocity = constrained_momentum / electron_density[:, :, None, :]

        gradient = self._velocity_gradient(
            electron_density,
            electron_face_density,
            electron_velocity,
        )
        theta = (
            gradient[:, :, 0, 0, :]
            + gradient[:, :, 1, 1, :]
            + gradient[:, :, 2, 2, :]
        )
        strain = xp.einsum(
            "tnaj,tnabj,tnbj->tnj",
            bhat,
            gradient,
            bhat,
        )
        pressure_tensor = self._pressure_tensor(
            electron_perp,
            electron_parallel,
            bhat,
        )
        pressure_force = self._gradient_adjoint(
            electron_density,
            electron_face_density,
            pressure_tensor,
        )
        magnetic_cross = xp.stack(
            (
                electron_velocity[:, :, 1, :] * bn[:, :, 2, :]
                - electron_velocity[:, :, 2, :] * bn[:, :, 1, :],
                electron_velocity[:, :, 2, :] * bn[:, :, 0, :]
                - electron_velocity[:, :, 0, :] * bn[:, :, 2, :],
                electron_velocity[:, :, 0, :] * bn[:, :, 1, :]
                - electron_velocity[:, :, 1, :] * bn[:, :, 0, :],
            ),
            axis=2,
        )
        magnetic_force = (
            electron_charge
            * xp.real(electron_density)[:, :, None, :]
            * magnetic_cross
        )

        pressure_flux = self._transport_flux(
            electron_perp,
            electron_face_density,
            electron_flux,
        )
        parallel_flux = self._transport_flux(
            electron_parallel,
            electron_face_density,
            electron_flux,
        )
        pdot_perp = (
            -self._sampled_divergence(pressure_flux)
            + electron_perp * (strain - theta)
        )
        pdot_parallel = (
            -self._sampled_divergence(parallel_flux)
            - 2.0 * electron_parallel * strain
        )

        ohm_source = (
            pressure_force / e
            - xp.real(electron_density)[:, :, None, :] * magnetic_cross
        )
        ohm_projection = self._constraint_project(
            ohm_source, electron_density
        )
        projected_velocity = (
            ohm_projection.value.reshape(batch, N, 3, nc)
            / electron_density[:, :, None, :]
        )
        e0 = self._fft(
            self._sample_matrix(context.op.I, projected_velocity.reshape(batch,N,3*nc))
        )
        wall = self._wall_correction(e0)
        electric = e0 + wall.correction

        perp_rhs = self._fft(pdot_perp)
        parallel_rhs = self._fft(pdot_parallel)
        b_rhs = -self._mode_group_regular(context.op.C, electric)
        rhs = xp.concatenate((perp_rhs, parallel_rhs, b_rhs), axis=2)

        finite_volume_velocity = electron_flux / electron_face_density
        theta_consistency = self._sampled_divergence(
            finite_volume_velocity
        )
        theta_defect = theta - theta_consistency
        theta_scale = xp.maximum(
            xp.max(xp.abs(theta), axis=(1, 2)),
            1.0e-300,
        )
        checks = SimpleNamespace(
            max_constraint_residual=float(
                xp.max(xp.abs(constraint.residual)).item() if constraint.residual.size else 0.
            ),
            max_full_wall_residual=float(
                xp.max(xp.abs(constraint.full_residual)).item() if constraint.full_residual.size else 0.
            ),
            max_ohm_constraint_residual=float(
                xp.max(xp.abs(ohm_projection.residual)).item() if ohm_projection.residual.size else 0.
            ),
            min_wall_rank_margin=float(
                xp.min(constraint.rank_margin).item()
            ),
            max_electric_wall_relative_residual=float(
                xp.max(wall.residual_scale).item()
            ),
            max_theta_relative_defect=float(
                xp.max(xp.abs(theta_defect) / theta_scale[:, None, None]).item()
            ),
            min_density_ratio=float(
                xp.min(
                    xp.real(electron_density)
                    / max(float(context.p.ref.ne), 1.0e-300)
                ).item()
            ),
            min_pressure_ratio=float(
                xp.min(
                    xp.minimum(
                        xp.real(electron_perp),
                        xp.real(electron_parallel),
                    )
                    / max(float(context.p.ref.pressure), 1.0e-300)
                ).item()
            ),
            min_magnetic_ratio=float(
                (xp.min(magnitude) / max(float(context.p.ref.magnetic), 1.0e-300)).item()
            ),
            max_initial_divergence=float(
                xp.max(initial_constraints.divergence_max).item()
            ),
            max_initial_wall_current=float(
                xp.max(initial_constraints.wall_current_max).item()
            ),
        )
        return SimpleNamespace(
            rhs=rhs,
            electric=electric,
            E0=e0,
            current=current.reshape(batch, modes * ne),
            velocity=electron_velocity.reshape(batch, N * 3 * nc),
            density=electron_density.reshape(batch, N * nc),
            electron_flux=electron_flux.reshape(batch, N * ne),
            gradient=gradient,
            pressure_force=pressure_force,
            magnetic_force=magnetic_force,
            pdot_perp=pdot_perp,
            pdot_parallel=pdot_parallel,
            perp_rhs=perp_rhs,
            parallel_rhs=parallel_rhs,
            b_rhs=b_rhs,
            constraint_residual=constraint.residual,
            full_wall_residual=constraint.full_residual,
            wall_rank_margin=constraint.rank_margin,
            ohm_constraint_residual=ohm_projection.residual,
            electric_wall_residual_scale=wall.residual_scale,
            wall_correction=wall.correction,
            wall_records=wall.records,
            initial_divergence=initial_constraints.divergence_max,
            initial_wall_current=initial_constraints.wall_current_max,
            theta=theta.reshape(batch, N * nc),
            strain=strain.reshape(batch, N * nc),
            theta_consistency=theta_consistency.reshape(batch, N * nc),
            theta_defect=theta_defect.reshape(batch, N * nc),
            checks=checks,
            identity=self._identity,
            scope=SimpleNamespace(
                field_rhs_only=True,
                ion_rhs=False,
                diagnostics=False,
                time_integration=False,
                full_acceptance=False,
                description=(
                    "bounded massless electron/field G producer; "
                    "not a full stage or scientific acceptance result"
                ),
            ),
        )


def _derivative(values, context):
    """Evaluate the retained signed-frequency azimuthal derivative."""
    producer = object.__new__(FieldBatch)
    producer.context = context
    producer.backend = context.compute
    producer.xp = context.compute.xp
    return producer._derivative(values)
