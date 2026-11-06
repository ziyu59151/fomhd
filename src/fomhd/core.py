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


def predictor_history_entries(c: Context) -> int:
    """Conservatively count retained T0 predictor stage/quadrature state."""
    coupling = getattr(c.p, "coupling", "synchronous")
    if coupling == "synchronous":
        return 0
    if coupling != "predictor":
        raise Failure("Config", f"unknown time coupling: {coupling}")
    micro_steps = getattr(c.p, "micro_steps", None)
    if (
        isinstance(micro_steps, bool)
        or type(micro_steps) is not int
        or micro_steps <= 0
    ):
        raise Failure(
            "Config",
            "micro_steps must be a positive non-boolean integer",
        )
    modes = 2 * c.p.K + 1
    full_state = _checked_product(
        modes,
        _checked_sum(11 * c.op.Nc, c.op.MB.size, "predictor state"),
        "predictor state",
    )
    # Four independent RK4 stages per microstep retain both the evaluated
    # state and its RHS. The cache allowance is deliberately conservative;
    # this is a rejection estimate, not a measured allocator result.
    per_stage = _checked_sum(
        _checked_sum(full_state, full_state, "predictor history"),
        _checked_product(
            modes,
            _checked_sum(8 * c.op.Nc, c.op.Ne, "predictor cache"),
            "predictor cache",
        ),
        "predictor history",
    )
    per_stage = _checked_sum(
        per_stage,
        _checked_product(
            256,
            _checked_product(
                c.N,
                _checked_sum(
                    _checked_sum(
                        3 * c.op.Nc,
                        c.op.Ne,
                        "predictor sample workspace",
                    ),
                    c.op.Nb,
                    "predictor sample workspace",
                ),
                "predictor sample workspace",
            ),
            "predictor sample workspace",
        ),
        "predictor history",
    )
    history = _checked_product(
        _checked_product(4, micro_steps, "memory predictor microstage count"),
        per_stage,
        "memory predictor history",
    )
    return _checked_sum(
        history,
        predictor_wall_history_entries(c),
        "memory predictor history",
    )


def predictor_wall_history_entries(c: Context) -> int:
    """Count retained per-stage dense wall samples and their factors."""
    if int(getattr(c.p, "electron_inertia", 0)) != 0:
        return 0
    coupling = getattr(c.p, "coupling", "synchronous")
    if coupling == "synchronous":
        return 0
    if coupling != "predictor":
        raise Failure("Config", f"unknown time coupling: {coupling}")
    micro_steps = getattr(c.p, "micro_steps", None)
    if (
        isinstance(micro_steps, bool)
        or type(micro_steps) is not int
        or micro_steps <= 0
    ):
        raise Failure(
            "Config",
            "micro_steps must be a positive non-boolean integer",
        )
    walls = getattr(c.geometry, "walls", ())
    wall_count = len(walls) if walls is not None else 0
    if wall_count == 0:
        return 0
    # Each retained ion wall context stores one dense w-by-w S matrix and, on
    # CPU, a dense LU original/factor set per phi sample. The factor 16 is a
    # conservative complex128-entry allowance for those matrices and copies.
    matrix_entries = _checked_product(
        16,
        _checked_product(
            c.N,
            _checked_product(
                wall_count,
                wall_count,
                "predictor wall matrix history",
            ),
            "predictor wall matrix history",
        ),
        "predictor wall matrix history",
    )
    vector_entries = _checked_product(
        32,
        _checked_product(
            c.N,
            _checked_sum(
                wall_count,
                3 * c.op.Nc,
                "predictor wall vector history",
            ),
            "predictor wall vector history",
        ),
        "predictor wall vector history",
    )
    per_stage = _checked_sum(
        matrix_entries,
        vector_entries,
        "predictor wall matrix history",
    )
    return _checked_product(
        _checked_product(4, micro_steps, "predictor wall history"),
        per_stage,
        "predictor wall history",
    )


def fixed_wall_cache_bytes(c: Context) -> int:
    """Conservative static upper bound for the independent massless wall cache."""
    if int(getattr(c.p, "electron_inertia", 0)) != 0:
        return 0
    walls = getattr(c.geometry, "walls", ())
    wall_count = len(walls) if walls is not None else 0
    if wall_count == 0:
        return 0
    modes = 2 * c.p.K + 1
    per_mode = _checked_sum(
        _checked_product(
            64,
            _checked_product(
                c.op.Ne,
                wall_count,
                "fixed-wall cache storage",
            ),
            "fixed-wall cache storage",
        ),
        _checked_product(
            128,
            _checked_product(
                wall_count,
                wall_count,
                "fixed-wall cache storage",
            ),
            "fixed-wall cache storage",
        ),
        "fixed-wall cache storage",
    )
    per_mode = _checked_sum(
        per_mode,
        _checked_product(
            64, wall_count, "fixed-wall cache storage"
        ),
        "fixed-wall cache storage",
    )
    per_mode = _checked_sum(
        per_mode,
        _checked_product(
            32,
            _checked_product(
                c.op.Nb,
                wall_count,
                "fixed-wall cache storage",
            ),
            "fixed-wall cache storage",
        ),
        "fixed-wall cache storage",
    )
    return _checked_product(
        modes,
        per_mode,
        "fixed-wall cache storage",
    )


def estimated_peak_bytes(c: Context) -> int:
    product = lambda left, right: _checked_product(left, right, "host memory plan")
    add = lambda left, right: _checked_sum(left, right, "host memory plan")
    modes = 2 * c.p.K + 1
    edges = product(modes, c.op.Ne)
    system = add(edges, c.op.Bc.shape[0])
    entries = product(32, product(system, system))
    entries = add(entries, product(4, c.op.Bc_full.size))
    entries = add(
        entries,
        product(256, product(c.N, add(add(3 * c.op.Nc, c.op.Ne), c.op.Nb))),
    )
    entries = add(entries, product(32, product(modes, add(11 * c.op.Nc, c.op.Nb))))
    entries = add(entries, predictor_history_entries(c))
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
    return add(
        product(entries, 16),
        fixed_wall_cache_bytes(c),
    )


def _workspace_bytes(c: Context) -> int:
    modes = 2 * c.p.K + 1
    entries = _checked_product(
        modes,
        _checked_sum(11 * c.op.Nc, c.op.MB.size, "state"),
        "state",
    )
    bytes_value = _checked_product(entries, 16 * 20, "state/workspace")
    return _checked_sum(
        bytes_value,
        _checked_product(
            predictor_history_entries(c),
            16,
            "predictor history bytes",
        ),
        "state/workspace",
    )


def preflight_memory(c: Context) -> int:
    """Apply complete allocation caps before any trial state is allocated."""
    if c.p.memory_limit == 0:
        raise Failure("Memory", "a positive host memory cap is required")
    peak_bytes = estimated_peak_bytes(c)
    _require_memory(peak_bytes, c.p.memory_limit, "complete A0 peak storage plan")
    workspace_bytes = _workspace_bytes(c)
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
    samples = [
        backend.mv(matrix, values.reshape(N, matrix.shape[1])[sample])
        for sample in range(N)
    ]
    return backend.array(backend.xp.stack(samples)).reshape(-1)


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


def _microstage_fraction(
    p: Any,
    micro_step: int,
    rk_stage: int,
) -> tuple[int, int]:
    """Validate and return the common integer fraction for one microstage."""
    if (
        type(micro_step) is not int
        or type(rk_stage) is not int
        or rk_stage not in (0, 1, 2, 3)
    ):
        raise Failure("Time", "invalid predictor clock index")
    micro_steps = getattr(p, "micro_steps", None)
    if (
        isinstance(micro_steps, bool)
        or type(micro_steps) is not int
        or micro_steps <= 0
        or micro_step < 0
        or micro_step >= micro_steps
    ):
        raise Failure(
            "Config",
            "micro_steps must be a positive non-boolean integer",
        )
    denominator = 4 * micro_steps
    numerator = 4 * micro_step + (0, 2, 2, 4)[rk_stage]
    return numerator, denominator


def microstage_time(
    p: Any,
    macro_step: int,
    micro_step: int,
    rk_stage: int,
) -> float:
    """Return a deterministic T0 clock from integer microstage fractions."""
    if type(macro_step) is not int or macro_step < 0:
        raise Failure("Time", "invalid predictor clock index")
    numerator, denominator = _microstage_fraction(
        p, micro_step, rk_stage
    )
    t0 = time_at_step(p, macro_step)
    t1 = time_at_step(p, macro_step + 1)
    if numerator == 0:
        return t0
    if numerator == denominator:
        return t1
    value = _fma(float(p.dt), numerator / denominator, t0)
    if not math.isfinite(value) or not (t0 < value < t1):
        raise Failure(
            "TimeResolution",
            "predictor microstage midpoint is not representable",
        )
    return value


def microstage_offset(
    p: Any,
    micro_step: int,
    rk_stage: int,
) -> float:
    """Return the integer-fraction offset from a macro interval boundary."""
    numerator, denominator = _microstage_fraction(
        p, micro_step, rk_stage
    )
    value = _fma(float(p.dt), numerator / denominator, 0.0)
    if not math.isfinite(value):
        raise Failure(
            "TimeResolution",
            "predictor microstage offset is not representable",
        )
    return value
