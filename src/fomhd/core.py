"""State allocation, retained spectra, and reproducible scheduling."""

from __future__ import annotations

import math
import sys
import ctypes
import ctypes.util
from typing import Any

import numpy as np

from .types import Context, Failure, FluidState, StageTag, State


_EPS = np.finfo(np.float64).eps
_LIBM = ctypes.CDLL(ctypes.util.find_library("m") or None)
_LIBM.fma.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
_LIBM.fma.restype = ctypes.c_double


def _fma(left: float, right: float, addend: float) -> float:
    return float(_LIBM.fma(left, right, addend))


def _checked_product(left: int, right: int, name: str) -> int:
    if left < 0 or right < 0:
        raise Failure("Memory", f"{name} size is negative")
    if left != 0 and right > sys.maxsize // left:
        raise Failure("Memory", f"{name} size product overflows")
    return left * right


def _checked_sum(left: int, right: int, name: str) -> int:
    if right > sys.maxsize - left:
        raise Failure("Memory", f"{name} size sum overflows")
    return left + right


def _require_memory(required: int, limit: int, name: str) -> None:
    if limit != 0 and required > limit:
        raise Failure(
            "Memory", f"{name} memory budget exceeded: required={required} limit={limit}"
        )


def slice_array(values: Any, offset: int, count: int) -> Any:
    if offset < 0 or count < 0 or offset + count > values.size:
        raise Failure("Shape", "slice outside array")
    return values[offset : offset + count]


def put(values: Any, offset: int, source: Any) -> None:
    if offset < 0 or offset + source.size > values.size:
        raise Failure("Shape", "write outside array")
    values[offset : offset + source.size] = source


def repeat_weights(weights: Any, count: int, backend: Any) -> Any:
    if count < 0:
        raise Failure("Shape", "negative weight repeat count")
    return backend.real_array(backend.xp.tile(weights, count))


def _massless_peak_bytes(c: Context, vector_entries: int) -> int:
    """Bound the live A0 massless operators, wall solves, and diagnostics."""
    name = "massless A0 memory plan"

    def product(left: int, right: int) -> int:
        return _checked_product(left, right, name)

    def add(left: int, right: int) -> int:
        return _checked_sum(left, right, name)

    op = c.op
    modes = 2 * c.p.K + 1
    mode_groups = (op.G, op.C, op.DB, op.DJ, op.Cdag)
    if any(len(group) != modes for group in mode_groups):
        raise Failure("Shape", "massless resident operator mode count mismatch")
    fixed_names = (
        "I", "H", "T", "T_full", "DR", "DZ", "topology_flux_row",
        "M0", "M1", "MB", "Rbar_r", "Rbar_v", "b1r", "b1z", "b2r", "b2z",
        "PR_r", "PZ_r", "PR_z", "PZ_z", "PR_v", "PZ_v",
        "wall_magnetic_divergence",
        "material_radial_derivative", "material_axial_derivative",
        "wall_toroidal_trace_indices",
    )
    resident = 0
    for matrix in (matrix for group in mode_groups for matrix in group):
        resident = add(resident, int(matrix.nbytes))
    for attribute in fixed_names:
        resident = add(resident, int(getattr(getattr(op, attribute), "nbytes", 0)))
    for matrix in (op.QB, op.wall_magnetic_reconstruction):
        if c.compute.is_csr(matrix):
            for array in (matrix.data, matrix.indices, matrix.indptr):
                resident = add(resident, int(array.nbytes))
        else:
            resident = add(resident, int(getattr(matrix, "nbytes", 0)))
    for axis in ("PR_r", "PZ_r", "PR_z", "PZ_z", "PR_v", "PZ_v"):
        factor = getattr(op, f"{axis}_factor")
        # Count originals even when they alias a retained transform.
        for array in (factor.original, factor.factors, factor.pivots):
            resident = add(resident, int(array.nbytes))
    if op.box_wall is not None:
        for array in vars(op.box_wall).values():
            resident = add(resident, int(getattr(array, "nbytes", 0)))
    if op.box_magnetic_norm is not None:
        for array in op.box_magnetic_norm[:-1]:
            resident = add(resident, int(array.nbytes))
    if op.ghost_toroidal is not None:
        for array in vars(op.ghost_toroidal).values():
            resident = add(resident, int(getattr(array, "nbytes", 0)))

    walls = len(c.geometry.walls)
    wall_square = product(walls, walls)
    cache_terms = add(
        add(product(64, product(op.Ne, walls)), product(128, wall_square)),
        add(product(64, walls), product(32, product(op.Nb, walls))),
    )
    cache = product(modes, cache_terms)
    # Hold the previous accepted frame and five RK4 evaluations, plus at most
    # one padded alias evaluation at a time. Include solve factors and vectors.
    retained_samples = add(product(6, c.N), add(product(2, c.N), 1))
    wall_per_sample = add(
        product(16, wall_square), product(32, add(walls, 3 * op.Nc))
    )
    wall_history = product(16, product(retained_samples, wall_per_sample))
    projection = product(
        16, add(product(2, product(op.T.shape[0], 3 * op.Nc)), wall_square)
    )
    cdag_entries = 0
    for matrix in op.Cdag:
        cdag_entries = add(cdag_entries, int(matrix.size))
    absolute = add(product(8, op.I.size), product(24, cdag_entries))
    absolute = add(absolute, product(8, op.T_full.size))
    adjoint = product(16, add(op.I.size, add(op.DR.size, op.DZ.size)))
    temporary = max(projection, absolute, adjoint)
    return add(add(resident, cache), add(wall_history, add(temporary, product(vector_entries, 16))))


def estimated_peak_bytes(c: Context) -> int:
    product = lambda left, right: _checked_product(left, right, "host memory plan")
    add = lambda left, right: _checked_sum(left, right, "host memory plan")
    modes = 2 * c.p.K + 1
    entries = product(256, product(c.N, add(add(3 * c.op.Nc, c.op.Ne), c.op.Nb)))
    entries = add(entries, product(32, product(modes, add(11 * c.op.Nc, c.op.Nb))))
    if not getattr(c.op, "finite_mass_available", True):
        return _massless_peak_bytes(c, entries)
    edges = product(modes, c.op.Ne)
    system = add(edges, c.op.Bc.shape[0])
    entries = add(entries, product(32, product(system, system)))
    entries = add(entries, product(4, c.op.Bc_full.size))
    resident = 0
    for group in (
        c.op.G,
        c.op.C,
        c.op.DB,
        c.op.DJ,
        c.op.Cdag,
        c.op.Lphi,
    ):
        for matrix in group:
            resident = add(resident, matrix.size)
    for matrix in (
        c.op.I,
        c.op.H,
        c.op.QB,
        c.op.T,
        c.op.T_full,
        c.op.PR_r,
        c.op.PZ_r,
        c.op.PR_z,
        c.op.PZ_z,
        c.op.PR_v,
        c.op.PZ_v,
    ):
        resident = add(resident, matrix.size)
    entries = add(entries, product(8, resident))
    return product(entries, 16)


def preflight_memory(c: Context) -> int:
    if int(getattr(c.p, "electron_inertia", 0)) != 0 and not getattr(
        c.op, "finite_mass_available", True
    ):
        raise Failure("Model", "finite-mass model requires complete operator storage")
    if c.p.memory_limit == 0:
        raise Failure("Memory", "a positive host memory cap is required")
    peak_bytes = estimated_peak_bytes(c)
    _require_memory(peak_bytes, c.p.memory_limit, "complete A0 peak storage plan")
    nc = c.op.Nc
    modes = 2 * c.p.K + 1
    entries = _checked_product(
        modes,
        _checked_sum(11 * nc, c.op.MB.size, "state"),
        "state",
    )
    workspace_bytes = _checked_product(entries, 16 * 20, "state/workspace")
    _require_memory(workspace_bytes, c.p.memory_limit, "state/workspace")
    if c.compute.name == "cuda":
        if c.p.device_memory_limit == 0:
            raise Failure("Memory", "a positive device memory cap is required")
        _require_memory(
            peak_bytes,
            c.p.device_memory_limit,
            "device complete A0 peak storage plan",
        )
        _require_memory(
            workspace_bytes,
            c.p.device_memory_limit,
            "device state/workspace",
        )

    return peak_bytes


def allocate_state(c: Context) -> State:
    preflight_memory(c)
    xp = c.compute.xp
    nc = c.op.Nc
    modes = 2 * c.p.K + 1

    def empty() -> Any:
        return c.compute.array(xp.empty(0, dtype=xp.complex128))

    def vector(count: int) -> Any:
        return c.compute.array(xp.zeros(modes * count, dtype=xp.complex128))

    ion = FluidState(
        n=vector(nc),
        mom=vector(3 * nc),
        perp=vector(nc),
        parallel=empty(),
    )
    electron = FluidState(
        n=vector(nc),
        mom=vector(3 * nc),
        perp=vector(nc),
        parallel=vector(nc),
    )
    return State(
        fluid=(ion, electron),
        B=vector(c.op.MB.size),
        step=0,
        generation=0,
        time=0.0,
    )


def _hermitian_check(values: Any, entities: int, c: Context, name: str) -> None:
    modes = 2 * c.p.K + 1
    if values.ndim != 1 or values.size != entities * modes:
        raise Failure("State", f"State: {name}")
    if not bool(c.compute.xp.all(c.compute.xp.isfinite(values)).item()):
        raise Failure("State", f"State: {name}")
    maximum = c.compute.max_abs(values)
    limit = 512.0 * _EPS * max(maximum, 1e-300)
    matrix = values.reshape(modes, entities)
    if bool(c.compute.xp.any(c.compute.xp.abs(matrix[c.p.K].imag) > limit).item()):
        raise Failure("Hermitian", f"Hermitian: {name}")
    for mode in range(1, c.p.K + 1):
        defect = c.compute.xp.abs(
            matrix[c.p.K - mode] - c.compute.xp.conj(matrix[c.p.K + mode])
        )
        if bool(c.compute.xp.any(defect > limit).item()):
            raise Failure("Hermitian", f"Hermitian: {name}")


def check_retained(state: State, c: Context) -> None:
    for species in range(2):
        fluid = state.fluid[species]
        _hermitian_check(fluid.n, c.op.Nc, c, "density")
        _hermitian_check(fluid.mom, 3 * c.op.Nc, c, "momentum")
        _hermitian_check(fluid.perp, c.op.Nc, c, "pressure")
        if species == 1:
            _hermitian_check(fluid.parallel, c.op.Nc, c, "parallel pressure")
        for mode in range(2 * c.p.K + 1):
            vector = fluid.mom[mode * 3 * c.op.Nc : (mode + 1) * 3 * c.op.Nc]
            reference = (
                c.p.species[species].mass
                * (c.p.ref.ne if species else c.p.ref.ni)
                * c.p.ref.velocity
            )
            limit = (
                2048.0
                * _EPS
                * max(c.compute.max_abs(vector), reference, 1e-300)
            )
            wall = c.compute.mv(c.op.T_full, vector)
            residual = c.compute.max_abs(wall)
            if residual > limit:
                raise Failure(
                    "WallState",
                    "input momentum violates full wall trace: "
                    f"species={c.p.species[species].id} mode={mode - c.p.K} "
                    f"residual={residual} limit={limit}",
                )
    _hermitian_check(state.B, c.op.MB.size, c, "B")


def combine(base: State, rhs: State, weight: float, tag: StageTag, c: Context) -> State:
    if int(getattr(c.p, "electron_inertia", 0)) == 0:
        from .massless import combine as combine_massless

        return combine_massless(base, rhs, weight, tag, c)
    if not math.isfinite(weight):
        raise Failure("Time", "nonfinite stage weight")
    fluid = []
    for species in range(2):
        left, right = base.fluid[species], rhs.fluid[species]
        fluid.append(
            FluidState(
                n=c.compute.axpy(left.n, right.n, weight),
                mom=c.compute.axpy(left.mom, right.mom, weight),
                perp=c.compute.axpy(left.perp, right.perp, weight),
                parallel=(
                    c.compute.axpy(left.parallel, right.parallel, weight)
                    if species
                    else c.compute.array(left.parallel)
                ),
            )
        )
    out = State(
        fluid=(fluid[0], fluid[1]),
        B=c.compute.axpy(base.B, rhs.B, weight),
        step=tag.step,
        generation=tag.generation,
        time=tag.time,
    )
    check_retained(out, c)
    return out


def mode_apply(matrices: list[Any], values: Any, backend: Any) -> Any:
    if not matrices:
        raise Failure("Shape", "empty mode operator")
    rows, columns = matrices[0].shape
    if values.ndim != 1 or values.size != len(matrices) * columns:
        raise Failure("Shape", "mode operator shape")
    modes = []
    for index, matrix in enumerate(matrices):
        if matrix.shape != (rows, columns):
            raise Failure("Shape", "mode operator shape")
        modes.append(backend.mv(matrix, values.reshape(len(matrices), columns)[index]))
    return backend.array(backend.xp.stack(modes)).reshape(-1)


def sample_apply(matrix: Any, values: Any, N: int, backend: Any) -> Any:
    if (
        N <= 0
        or matrix.ndim != 2
        or values.ndim != 1
        or values.size != N * matrix.shape[1]
    ):
        raise Failure("Shape", "sample operator shape")
    columns = matrix.shape[1]
    return backend.array(
        backend.mm(matrix, values.reshape(N, columns).T).T
    ).reshape(-1)


def reconstruct_magnetic(operators: Any, values: Any, backend: Any = None) -> Any:
    """Reconstruct mode-major magnetic components at cell centers on their backend."""
    backend = backend or operators.compute
    if values.ndim != 1 or values.size % operators.Nb != 0:
        raise Failure("Shape", "magnetic reconstruction input shape")
    modes = values.size // operators.Nb
    result = sample_apply(operators.QB, values, modes, backend)
    toroidal = getattr(operators, "ghost_toroidal", None)
    if toroidal is not None:
        from .ghost_toroidal import toroidal_reconstruct

        start = operators.PZ_z.shape[0]
        stop = start + operators.Rbar_v.size
        vertices = values.reshape(modes, operators.Nb)[:, start:stop]
        result.reshape(modes, 3, operators.Nc)[:, 1, :] = toroidal_reconstruct(
            toroidal, vertices, backend
        )
    return result


def inverse(
    values: Any,
    entities: int,
    c: Context,
    physical: bool = True,
    reference: float = 0.0,
) -> Any:
    if values.ndim != 1 or values.size != (2 * c.p.K + 1) * entities:
        raise Failure("Shape", "inverse FFT shape")
    samples = c.compute.inverse_fft(values, entities, c.p.K, c.N)
    if not physical:
        return samples
    if reference < 0.0 or not math.isfinite(reference):
        raise Failure("Reference", "invalid imaginary-residue reference")
    limit = 512.0 * _EPS * max(c.compute.max_abs(values), reference, 1e-300)
    return c.compute.physical_samples(samples, limit)


def forward(values: Any, entities: int, c: Context) -> Any:
    return c.compute.forward_fft(values, entities, c.p.K, c.N)


def time_at_step(p: Any, step: int) -> float:
    if step < 0 or step > (1 << 52) or not math.isfinite(p.dt) or p.dt <= 0.0:
        raise Failure("Time", "invalid step/dt")
    time = _fma(float(step), p.dt, p.start_time)
    if not math.isfinite(time):
        raise Failure("Time", "nonfinite scheduled time")
    return time


def time_at_midpoint(p: Any, step: int) -> float:
    midpoint = _fma(float(step) + 0.5, p.dt, p.start_time)
    if not (time_at_step(p, step) < midpoint < time_at_step(p, step + 1)):
        raise Failure("TimeResolution", "midpoint not representable")
    return midpoint
