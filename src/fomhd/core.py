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


def _history_sample_count(c: Context) -> int:
    limit = getattr(c, "N", None)
    samples = getattr(c, "_history_samples", limit)
    if (
        isinstance(limit, bool)
        or type(limit) is not int
        or limit <= 0
        or isinstance(samples, bool)
        or type(samples) is not int
        or samples <= 0
        or samples > limit
    ):
        raise Failure(
            "Memory",
            "history sample identity must be a positive non-boolean "
            "integer no greater than N",
        )
    return samples


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


def _multirate_wr_controls(c: Context) -> tuple[int, int, int, int]:
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
    maximum = getattr(c.p, "waveform_max_iterations", None)
    if isinstance(maximum, bool) or type(maximum) is not int or maximum <= 0:
        raise Failure(
            "Config",
            "waveform_max_iterations must be a positive non-boolean integer",
        )
    ion_nodes = getattr(c.p, "ion_nodes", None)
    if type(ion_nodes) is not int or ion_nodes not in (3, 5):
        raise Failure("Config", "ion_nodes must be 3 or 5")
    samples = _history_sample_count(c)
    return micro_steps, maximum, ion_nodes, samples


def _stage_history_entries(c: Context, samples: int) -> int:
    """Conservative per-retained-full-stage workspace/record bound."""

    modes = 2 * c.p.K + 1
    full_state = _checked_product(
        modes,
        _checked_sum(11 * c.op.Nc, c.op.MB.size, "retained stage state"),
        "retained stage state",
    )
    workspace = _checked_product(
        256,
        _checked_product(
            samples,
            _checked_sum(
                _checked_sum(
                    3 * c.op.Nc,
                    c.op.Ne,
                    "retained stage workspace",
                ),
                c.op.Nb,
                "retained stage workspace",
            ),
            "retained stage workspace",
        ),
        "retained stage workspace",
    )
    cache = _checked_product(
        modes,
        _checked_sum(8 * c.op.Nc, c.op.Ne, "retained stage cache"),
        "retained stage cache",
    )
    return _checked_sum(
        _checked_sum(
            2 * full_state,
            cache,
            "retained stage",
        ),
        workspace,
        "retained stage",
    )


def _wall_history_per_stage(c: Context, samples: int) -> int:
    walls = getattr(c.geometry, "walls", ())
    wall_count = len(walls) if walls is not None else 0
    if wall_count == 0:
        return 0
    matrix_entries = _checked_product(
        16,
        _checked_product(
            samples,
            _checked_product(
                wall_count,
                wall_count,
                "retained wall matrix",
            ),
            "retained wall matrix",
        ),
        "retained wall matrix",
    )
    vector_entries = _checked_product(
        32,
        _checked_product(
            samples,
            _checked_sum(
                wall_count,
                3 * c.op.Nc,
                "retained wall vector",
            ),
            "retained wall vector",
        ),
        "retained wall vector",
    )
    return _checked_sum(
        matrix_entries,
        vector_entries,
        "retained wall stage",
    )


def _multirate_wr_wall_history_entries(c: Context) -> int:
    micro_steps, maximum, ion_nodes, samples = _multirate_wr_controls(c)
    alias_samples = _checked_sum(
        2 * c.N,
        1,
        "alias sample count",
    )
    alias_samples = max(alias_samples, samples)
    coarse_count = _checked_sum(
        1,
        _checked_product(ion_nodes, maximum, "WR coarse count"),
        "WR coarse count",
    )
    fine_count = _checked_product(
        4,
        _checked_product(micro_steps, maximum, "WR fine count"),
        "WR fine count",
    )
    alias_count = _checked_product(
        2,
        _checked_product(micro_steps, maximum, "WR alias count"),
        "WR alias count",
    )
    non_alias_count = _checked_sum(
        _checked_sum(coarse_count, fine_count, "WR wall count"),
        2,
        "WR wall count",
    )
    non_alias = _checked_product(
        non_alias_count,
        _wall_history_per_stage(c, samples),
        "WR wall history",
    )
    alias = _checked_product(
        alias_count,
        _wall_history_per_stage(c, alias_samples),
        "WR alias wall history",
    )
    return _checked_sum(non_alias, alias, "WR wall history")


def multirate_wr_history_entries(c: Context) -> int:
    """Conservative retained-memory bound for all WR outer sweeps."""

    micro_steps, maximum, ion_nodes, samples = _multirate_wr_controls(c)
    alias_samples = max(
        samples,
        _checked_sum(2 * c.N, 1, "alias sample count"),
    )
    modes = 2 * c.p.K + 1
    p_entries = _checked_product(
        modes, 5 * c.op.Nc, "WR P entries"
    )
    z_entries = _checked_product(
        modes,
        _checked_sum(2 * c.op.Nc, c.op.Nb, "WR Z entities"),
        "WR Z entries",
    )
    walls = getattr(c.geometry, "walls", ())
    wall_count = len(walls) if walls is not None else 0
    vector_entries = _checked_product(
        samples, 3 * c.op.Nc, "WR vector entries"
    )
    coarse_count = _checked_sum(
        1,
        _checked_product(ion_nodes, maximum, "WR coarse count"),
        "WR coarse count",
    )
    fine_count = _checked_product(
        4,
        _checked_product(micro_steps, maximum, "WR fine count"),
        "WR fine count",
    )
    alias_count = _checked_product(
        2,
        _checked_product(micro_steps, maximum, "WR alias count"),
        "WR alias count",
    )
    stage_entries = _stage_history_entries(c, samples)
    alias_stage_entries = _stage_history_entries(c, alias_samples)
    full_stage_count = _checked_sum(
        coarse_count,
        _checked_sum(fine_count, 2, "WR full stage count"),
        "WR full stage count",
    )
    history = _checked_product(
        full_stage_count,
        stage_entries,
        "WR retained full stages",
    )
    history = _checked_sum(
        history,
        _checked_product(
            alias_count,
            alias_stage_entries,
            "WR retained alias stages",
        ),
        "WR retained stages",
    )
    coarse_extra = _checked_sum(
        _checked_product(
            3, p_entries, "WR coarse A/C/F storage"
        ),
        _checked_sum(
            _checked_product(
                2, vector_entries, "WR coarse projection storage"
            ),
            _checked_product(
                2, wall_count, "WR coarse multiplier storage"
            ),
            "WR coarse projection storage",
        ),
        "WR coarse record storage",
    )
    history = _checked_sum(
        history,
        _checked_product(
            coarse_count,
            coarse_extra,
            "WR coarse records",
        ),
        "WR coarse records",
    )
    fine_extra = _checked_sum(
        _checked_product(
            2,
            _checked_sum(p_entries, z_entries, "WR fine input/rate"),
            "WR fine input/rate",
        ),
        _checked_sum(
            _checked_product(
                4, vector_entries, "WR EM producer storage"
            ),
            _checked_product(
                2, wall_count, "WR EM multiplier storage"
            ),
            "WR EM producer storage",
        ),
        "WR fine record storage",
    )
    history = _checked_sum(
        history,
        _checked_product(
            fine_count,
            fine_extra,
            "WR fine records",
        ),
        "WR fine records",
    )
    window_entries = _checked_product(
        4,
        _checked_product(
            micro_steps,
            _checked_sum(
                _checked_sum(p_entries, p_entries, "WR window P"),
                _checked_sum(z_entries, z_entries, "WR window Z"),
                "WR window stages",
            ),
            "WR window stages",
        ),
        "WR window stages",
    )
    window_entries = _checked_sum(
        window_entries,
        _checked_product(
            16,
            _checked_product(
                ion_nodes,
                max(p_entries, z_entries),
                "WR coarse window",
            ),
            "WR coarse window",
        ),
        "WR window",
    )
    history = _checked_sum(
        history, window_entries, "WR window history"
    )
    return _checked_sum(
        history,
        _multirate_wr_wall_history_entries(c),
        "WR retained history",
    )


def predictor_history_entries(c: Context) -> int:
    """Conservatively count one retained A1 microstep sweep and path."""
    samples = _history_sample_count(c)
    coupling = getattr(c.p, "coupling", "synchronous")
    if coupling == "synchronous":
        return 0
    if coupling == "multirate_wr":
        return multirate_wr_history_entries(c)
    if coupling not in ("predictor", "waveform"):
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
                samples,
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
    history = _checked_sum(
        history,
        predictor_wall_history_entries(c),
        "memory predictor history",
    )
    if coupling == "waveform":
        ion_path = _checked_product(
            modes,
            _checked_product(
                5,
                c.op.Nc,
                "waveform ion path",
            ),
            "waveform ion path",
        )
        # Old and new stage paths plus their endpoints are retained in
        # addition to one complete evaluated sweep.
        history = _checked_sum(
            history,
            _checked_product(
                _checked_product(
                    2,
                    _checked_product(
                        4,
                        micro_steps,
                        "waveform ion stage paths",
                    ),
                    "waveform ion stage paths",
                ),
                ion_path,
                "waveform ion stage paths",
            ),
            "waveform ion stage paths",
        )
        history = _checked_sum(
            history,
            _checked_product(2, ion_path, "waveform ion endpoints"),
            "waveform ion endpoints",
        )
    return history


def predictor_wall_history_entries(c: Context) -> int:
    """Count retained per-stage dense wall samples and their factors."""
    samples = _history_sample_count(c)
    if int(getattr(c.p, "electron_inertia", 0)) != 0:
        return 0
    coupling = getattr(c.p, "coupling", "synchronous")
    if coupling == "synchronous":
        return 0
    if coupling == "multirate_wr":
        return _multirate_wr_wall_history_entries(c)
    if coupling not in ("predictor", "waveform"):
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
    # Each retained ion wall context stores dense matrices, factors, and
    # sample vectors. This helper preserves the historical predictor bound.
    per_stage = _wall_history_per_stage(c, samples)
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


def _require_model_storage(c: Context) -> None:
    """Reject operator storage that the selected electron model cannot use."""
    finite_mass_available = getattr(c.op, "finite_mass_available", True)
    if type(finite_mass_available) is not bool:
        raise Failure("Model", "operator finite-mass availability is invalid")
    if int(getattr(c.p, "electron_inertia", 0)) != 0 and not finite_mass_available:
        raise Failure(
            "Model",
            "finite-mass electron model requires complete finite-mass "
            "operator storage",
        )


def _array_storage_bytes(value: Any) -> int:
    if value is None:
        return 0
    entries = 1
    for extent in getattr(value, "shape", ()):
        entries = _checked_product(
            entries,
            int(extent),
            "operator resident storage",
        )
    return _checked_product(
        entries,
        int(getattr(value, "itemsize", 0)),
        "operator resident storage bytes",
    )


def _factor_storage_bytes(factor: Any) -> int:
    if factor is None:
        return 0
    total = 0
    # An original can alias its source matrix; counting both is a safe bound.
    for value in (factor.original, factor.factors, factor.pivots):
        total = _checked_sum(
            total,
            _array_storage_bytes(value),
            "operator factor storage",
        )
    return total


def massless_matrix_storage(c: Context) -> dict[str, int]:
    """Return the massless resident inventory and bounded live temporaries."""
    storage: dict[str, int] = {}
    modes = 2 * c.p.K + 1
    if any(
        len(getattr(c.op, name, ())) != modes
        for name in ("G", "C", "DB", "DJ", "Cdag")
    ):
        raise Failure("Shape", "massless resident operator mode count is inconsistent")

    def group(name: str, values: Any) -> None:
        total = 0
        for value in values:
            total = _checked_sum(
                total,
                _array_storage_bytes(value),
                f"{name} storage",
            )
        storage[name] = total

    group(
        "mode_matrices",
        (
            matrix
            for name in ("G", "C", "DB", "DJ", "Cdag")
            for matrix in getattr(c.op, name)
        ),
    )
    group(
        "fixed_matrices",
        (
            getattr(c.op, name, None)
            for name in (
                "I",
                "H",
                "QB",
                "T_full",
                "T",
                "DR",
                "DZ",
                "topology_flux_row",
                "b1r",
                "b1z",
                "b2r",
                "b2z",
            )
        ),
    )
    group(
        "filter_matrices",
        (
            getattr(c.op, name, None)
            for name in ("PR_r", "PZ_r", "PR_z", "PZ_z", "PR_v", "PZ_v")
        ),
    )
    group(
        "metric_vectors",
        (
            getattr(c.op, name, None)
            for name in ("M0", "M1", "MB", "Rbar_r", "Rbar_v")
        ),
    )
    box_wall = getattr(c.op, "box_wall", None)
    if box_wall is not None:
        group("rectangular_wall", (*vars(box_wall).values(), c.op.wall_magnetic_divergence))
    filter_factors = 0
    for name in ("PR_r", "PZ_r", "PR_z", "PZ_z", "PR_v", "PZ_v"):
        filter_factors = _checked_sum(
            filter_factors,
            _factor_storage_bytes(getattr(c.op, f"{name}_factor", None)),
            "filter factor storage",
        )
    storage["filter_factors"] = filter_factors

    # A massless context with the complete operator set remains a valid oracle.
    finite_oracle = 0
    for name in ("Lphi", "poisson_spectra"):
        for value in getattr(c.op, name, ()):
            finite_oracle = _checked_sum(
                finite_oracle,
                _array_storage_bytes(value),
                "finite oracle storage",
            )
    for value in (getattr(c.op, "Bc_full", None), getattr(c.op, "Bc", None)):
        finite_oracle = _checked_sum(
            finite_oracle,
            _array_storage_bytes(value),
            "finite oracle storage",
        )
    for factor in getattr(c.op, "phi_factor", ()):
        finite_oracle = _checked_sum(
            finite_oracle,
            _factor_storage_bytes(factor),
            "finite oracle factor storage",
        )
    storage["finite_storage_oracle"] = finite_oracle

    persistent = 0
    for name in (
        "mode_matrices",
        "fixed_matrices",
        "filter_matrices",
        "metric_vectors",
        "filter_factors",
        "finite_storage_oracle",
    ):
        persistent = _checked_sum(
            persistent,
            storage[name],
            "persistent operator storage",
        )
    if box_wall is not None:
        persistent = _checked_sum(
            persistent, storage["rectangular_wall"], "persistent rectangular wall storage"
        )
    storage["persistent"] = persistent

    walls = len(getattr(c.geometry, "walls", ()))
    # Wall setup simultaneously retains curl(Nb x W), thin-curl(Ne x W), and
    # five W x W scaling/principal/LU buffers before the cache publishes them.
    wall_temporaries = _checked_sum(
        _checked_product(
            c.op.Nb,
            walls,
            "wall-cache boundary temporaries",
        ),
        _checked_product(
            c.op.Ne,
            walls,
            "wall-cache edge temporaries",
        ),
        "wall-cache matrix temporaries",
    )
    wall_temporaries = _checked_sum(
        wall_temporaries,
        _checked_product(
            5,
            _checked_product(
                walls,
                walls,
                "wall-cache principal temporaries",
            ),
            "wall-cache principal temporaries",
        ),
        "wall-cache matrix temporaries",
    )
    selected = getattr(c.op, "T", None)
    selected_rows = (
        int(selected.shape[0])
        if selected is not None and len(selected.shape) > 0
        else 0
    )
    projection_temporaries = _checked_sum(
        _checked_product(
            2,
            _checked_product(
                selected_rows,
                3 * c.op.Nc,
                "constraint-projection temporaries",
            ),
            "constraint-projection temporaries",
        ),
        _checked_product(
            selected_rows,
            selected_rows,
            "constraint-projection normal",
        ),
        "constraint-projection temporaries",
    )
    spatial_temporaries = _checked_product(
        max(wall_temporaries, projection_temporaries),
        16,
        "matrix temporary bytes",
    )
    interpolation_entries = int(getattr(c.op.I, "size", 0))
    cdag_entries = 0
    for matrix in c.op.Cdag:
        cdag_entries = _checked_sum(cdag_entries, int(matrix.size), "Cdag stack")
    # massless_checks holds abs(I) while constructing the complex Cdag stack
    # and its real absolute-value output. Borrowing Cdag does not avoid this.
    storage["diagnostic_assembly_bytes"] = _checked_sum(
        _checked_product(8, interpolation_entries, "absolute interpolation"),
        _checked_product(24, cdag_entries, "diagnostic Cdag stack and absolute"),
        "diagnostic assembly",
    )
    absolute_entries = _checked_sum(
        _checked_sum(interpolation_entries, cdag_entries, "diagnostic arrays"),
        int(getattr(c.op.T_full, "size", 0)),
        "diagnostic arrays",
    )
    storage["diagnostic_retained_bytes"] = _checked_product(
        8, absolute_entries, "diagnostic absolute arrays"
    )
    adjoint_entries = interpolation_entries
    for name in ("DR", "DZ"):
        adjoint_entries = _checked_sum(
            adjoint_entries,
            int(getattr(getattr(c.op, name, None), "size", 0)),
            "gradient adjoint temporaries",
        )
    storage["adjoint_temporary_bytes"] = _checked_product(
        16, adjoint_entries, "gradient conjugate-transpose arrays"
    )
    storage["temporary_matrices"] = max(
        spatial_temporaries,
        storage["diagnostic_assembly_bytes"],
        storage["diagnostic_retained_bytes"],
        storage["adjoint_temporary_bytes"],
    )
    return storage


def estimated_peak_bytes(c: Context) -> int:
    product = lambda left, right: _checked_product(left, right, "host memory plan")
    add = lambda left, right: _checked_sum(left, right, "host memory plan")
    modes = 2 * c.p.K + 1
    edges = product(modes, c.op.Ne)
    entries = 0
    if int(getattr(c.p, "electron_inertia", 0)) != 0:
        system = add(edges, c.op.Bc.shape[0])
        entries = product(32, product(system, system))
        entries = add(entries, product(4, c.op.Bc_full.size))
    entries = add(
        entries,
        product(256, product(c.N, add(add(3 * c.op.Nc, c.op.Ne), c.op.Nb))),
    )
    entries = add(entries, product(32, product(modes, add(11 * c.op.Nc, c.op.Nb))))
    entries = add(entries, predictor_history_entries(c))
    if int(getattr(c.p, "electron_inertia", 0)) != 0:
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
        return add(product(entries, 16), fixed_wall_cache_bytes(c))

    storage = massless_matrix_storage(c)
    total = product(entries, 16)
    total = add(total, storage["persistent"])
    total = add(total, storage["temporary_matrices"])
    return add(total, fixed_wall_cache_bytes(c))


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
    _require_model_storage(c)
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
    _require_model_storage(c)
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
