"""Bounded research-only C1/C2 reduced-hydro ion multirate candidates.

This module deliberately does not own a production accepted/rejected
transaction.  It returns immutable research trials and isolates the one
optional full-stage endpoint check on a copied context. It does not implement
the absent production interval-work budget or same-accuracy acceptance.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable

import numpy as np

from fomhd import core
from fomhd.field_batch import FieldBatch
from fomhd.fluid import build_fluid, build_non_electric
from fomhd.stage import evaluate_stage
from fomhd.types import Failure, FluidState, StageTag, State
from fomhd.wall import build_wall, project_wall


_RK4_WEIGHTS = (1.0, 2.0, 2.0, 1.0)
_METHODS = frozenset(("C1", "C2"))
_DUPLICATE_OBSERVATION_COUNTERS = frozenset(
    (
        "actual_outer_calls",
        "full_hydro_eval",
        "cheap_D",
        "cheap_L",
        "field_rhs",
    )
)


class Counters:
    """Mutable call ledger for one bounded research attempt."""

    __slots__ = (
        "actual_outer_calls",
        "full_hydro_eval",
        "cheap_D",
        "cheap_L",
        "field_rhs",
        "wall_projection",
        "wall_normal_initial_solves",
        "quadrature_contributions",
        "endpoint_check",
        "endpoint_full_ion_RHS",
    )

    def __init__(self) -> None:
        for name in self.__slots__:
            setattr(self, name, 0)

    def snapshot(self) -> MappingProxyType:
        return MappingProxyType(
            {name: int(getattr(self, name)) for name in self.__slots__}
        )


def _combined_counters(*snapshots: MappingProxyType) -> MappingProxyType:
    """Merge two observations of the same research attempt.

    The callback kernel and production adapter observe the same attempted
    full-hydro, cheap-D/L, and field calls. Those duplicate counters use their
    maximum. Endpoint-check counters are present only in the adapter snapshot.
    """
    combined = {}
    for name in Counters.__slots__:
        values = [
            int(snapshot.get(name, 0))
            for snapshot in snapshots
        ]
        if (
            name not in _DUPLICATE_OBSERVATION_COUNTERS
            and sum(value != 0 for value in values) > 1
        ):
            raise Failure(
                "Counters",
                f"non-duplicate counter {name} observed in multiple snapshots",
            )
        combined[name] = max(values, default=0)
    return MappingProxyType(combined)


@dataclass(frozen=True, slots=True)
class MacroTerms:
    """Three P-shaped split terms from one hydro reconstruction."""

    density_rate: Any
    momentum_rate: Any
    pressure_rate: Any


@dataclass(frozen=True, slots=True)
class PhysicalScales:
    """Componentwise SI scales for the distinct P and Z layouts."""

    p: Any
    z: Any


@dataclass(frozen=True, slots=True)
class FieldTerms:
    """Real FieldBatch result plus the endpoint-compatible derived fields."""

    rhs: Any
    electric: Any
    e0: Any
    magnetic: Any
    density: Any
    velocity: Any
    current: Any
    checks: Any
    raw: Any


@dataclass(frozen=True, slots=True)
class StageRecord:
    """One distinct RK4 derivative stage."""

    outer: int
    micro_step: int
    rk_stage: int
    p: Any
    z: Any
    p_rhs: Any
    z_rhs: Any
    p_rhs_available: bool
    electric: Any
    magnetic: Any


@dataclass(frozen=True, slots=True)
class IterationRecord:
    """One complete outer sweep, retained even when rejected."""

    outer: int
    guess_p: Any
    endpoint_p: Any
    target_p: Any
    residual: Any
    residual_raw: float
    residual_normalized: float
    scale: float
    actual_increment_max: float
    converged: bool
    stage_count: int


@dataclass(frozen=True, slots=True)
class ResearchResult:
    """Immutable owning record for one completed research window."""

    method: str
    p: Any
    z: Any
    midpoint_p: Any
    endpoint_p: Any
    integrated_electric: Any
    integrated_magnetic: Any
    residual: Any
    residual_raw: float
    residual_normalized: float
    iterations: tuple[IterationRecord, ...]
    stages: tuple[StageRecord, ...]
    counters: MappingProxyType
    stage_history_bytes: int
    endpoint: FieldTerms | None = None
    state: State | None = None
    cache: Any = None
    endpoint_check: Any = None
    metadata: MappingProxyType = MappingProxyType({})


def _xp_for(value: Any, xp: Any = None):
    if xp is not None:
        return xp
    module = type(value).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy

        return cupy
    return np


def _copy(value: Any) -> Any:
    return value.copy()


def _max_abs(value: Any, xp: Any) -> float:
    if value is None or value.size == 0:
        return 0.0
    return float(xp.max(xp.abs(value)).item())


def _scalar(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise Failure("Finite", "nonfinite multirate scalar")
    return result


def _triple(value: Any, names: tuple[str, str, str]):
    if hasattr(value, names[0]):
        return tuple(getattr(value, name) for name in names)
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return tuple(value)
    raise Failure("Callbacks", f"expected a three-field {names} result")


def _pair(value: Any, names: tuple[str, str]):
    if hasattr(value, names[0]):
        return tuple(getattr(value, name) for name in names)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return tuple(value)
    raise Failure("Callbacks", f"expected a two-field {names} result")


def _residual_metrics(
    guess: Any,
    target: Any,
    scales: Any,
    xp: Any,
) -> tuple[Any, float, float, float, float]:
    if guess.shape != target.shape or guess.shape != scales.shape:
        raise Failure("Shape", "multirate residual/scales layout mismatch")
    residual = guess - target
    raw_values = xp.abs(residual)
    normalized_values = raw_values / xp.maximum(
        xp.abs(scales), 1.0e-300
    )
    raw = _max_abs(raw_values, xp)
    normalized = _max_abs(normalized_values, xp)
    scale = _scalar(xp.max(xp.maximum(xp.abs(scales), 1.0e-300)).item())
    increment = _max_abs(target - guess, xp)
    return residual, raw, normalized, scale, increment


def _stage_bytes(stages: tuple[StageRecord, ...]) -> int:
    total = 0
    seen = set()
    for stage in stages:
        for value in (
            stage.p,
            stage.z,
            stage.p_rhs,
            stage.z_rhs,
            stage.electric,
            stage.magnetic,
        ):
            interface = getattr(value, "__array_interface__", None)
            pointer = (
                int(interface["data"][0])
                if interface is not None and "data" in interface
                else 0
            )
            if pointer and pointer in seen:
                continue
            if pointer:
                seen.add(pointer)
            total += int(getattr(value, "nbytes", 0))
    return total


def _attach_failure(
    failure: Failure,
    *,
    counters: Counters,
    iterations: list[IterationRecord],
    stages: list[StageRecord],
    last_p: Any,
    last_z: Any,
) -> None:
    failure.research_history = tuple(iterations)
    failure.research_counters = counters.snapshot()
    failure.research_stages = tuple(stages)
    failure.last_attempt_p = None if last_p is None else _copy(last_p)
    failure.last_attempt_z = None if last_z is None else _copy(last_z)


def rk4_stability(argument: complex) -> complex:
    """Return the scalar RK4 stability polynomial."""
    value = complex(argument)
    return 1.0 + value + value**2 / 2.0 + value**3 / 6.0 + value**4 / 24.0


def _phi(z: complex, order: int) -> complex:
    """Return phi_n(z) with the z=0 continuous extension."""
    value = complex(z)
    if abs(value) < 1.0e-5:
        return complex(
            sum(
                value**power / math.factorial(power + order)
                for power in range(32)
            )
        )
    numerator = np.exp(value) - sum(
        value**power / math.factorial(power)
        for power in range(order)
    )
    return complex(numerator / value**order)


def cold_hall_t0_map(a: float, beta: float, H: float) -> np.ndarray:
    """Independent exact-fast integrated T0 map for normalized cold Hall.

    The returned state order is ``(u,b)`` for
    ``u'=i*a*b`` and ``b'=i*a*u+i*beta*b``.
    """
    values = (float(a), float(beta), float(H))
    if not all(math.isfinite(value) for value in values) or H <= 0.0:
        raise ValueError("cold Hall parameters must be finite and H positive")
    z = 1j * beta * H
    f0 = H * _phi(z, 1)
    f1 = H**2 * _phi(z, 2)
    f2 = H**3 * _phi(z, 3)
    return np.array(
        (
            (
                1.0 - a * a * f1,
                1j * a * (f0 - a * a * f2),
            ),
            (
                1j * a * f0,
                np.exp(z) - a * a * f1,
            ),
        ),
        dtype=np.complex128,
    )


def _validate_controls(
    p0: Any,
    z0: Any,
    H: float,
    M: int,
    scales: Any,
    rtol: float,
    max_iterations: int,
) -> None:
    if (
        not isinstance(H, (int, float))
        or isinstance(H, bool)
        or not math.isfinite(float(H))
        or float(H) <= 0.0
    ):
        raise Failure("Config", "H must be a positive finite non-boolean real")
    if type(M) is not int or M <= 0:
        raise Failure("Config", "M must be a positive non-boolean integer")
    if (
        not isinstance(rtol, (int, float))
        or isinstance(rtol, bool)
        or not math.isfinite(float(rtol))
        or float(rtol) <= 0.0
    ):
        raise Failure("Config", "rtol must be positive and finite")
    if type(max_iterations) is not int or max_iterations <= 0:
        raise Failure(
            "Config", "max_iterations must be a positive non-boolean integer"
        )
    if p0.ndim != 2 or z0.ndim != 2 or p0.shape[0] != z0.shape[0]:
        raise Failure("Shape", "P and Z must be 2D with a shared mode axis")
    if scales.shape != p0.shape:
        raise Failure("Shape", "P scales must have the P layout")


def centered_impulse_callbacks(
    macro: Callable[[Any], Any],
    field_rhs: Callable[[Any, Any], Any],
    impulse: Callable[[Any, Any, Any], Any],
    p0: Any,
    z0: Any,
    *,
    H: float,
    M: int,
    scales: Any,
    rtol: float = 1.0e-10,
    max_iterations: int = 12,
    keep_stages: bool = True,
    xp: Any = None,
) -> ResearchResult:
    """Run the generic C1 centered-impulse fixed-point callback kernel."""
    xp = _xp_for(p0, xp)
    _validate_controls(p0, z0, H, M, scales, rtol, max_iterations)
    counters = Counters()
    iterations: list[IterationRecord] = []
    all_stages: list[StageRecord] = []
    guess = _copy(p0)
    last_p = _copy(p0)
    last_z = _copy(z0)
    h = float(H) / M
    if not math.isfinite(h) or h <= 0.0:
        raise Failure("TimeResolution", "microstep is not representable")
    try:
        for outer in range(max_iterations):
            counters.actual_outer_calls += 1
            counters.full_hydro_eval += 1
            last_p, last_z = _copy(guess), _copy(z0)
            d_rate, momentum_rate, pressure_rate = _triple(
                macro(_copy(guess)),
                ("density_rate", "momentum_rate", "pressure_rate"),
            )
            if any(
                value.shape != p0.shape
                for value in (d_rate, momentum_rate, pressure_rate)
            ):
                raise Failure("Shape", "macro callback returned a non-P term")

            z = _copy(z0)
            # Generic callback layouts have no fixed E/B extents.  Allocate
            # from the first stage, including the real stage itself.
            integrated_electric = None
            integrated_magnetic = None
            for micro in range(M):
                rates_z = []
                for rk_stage in range(4):
                    if rk_stage == 0:
                        stage_z = _copy(z)
                    elif rk_stage == 1:
                        stage_z = z + (h / 2.0) * rates_z[0]
                    elif rk_stage == 2:
                        stage_z = z + (h / 2.0) * rates_z[1]
                    else:
                        stage_z = z + h * rates_z[2]
                    counters.field_rhs += 1
                    last_p, last_z = _copy(guess), _copy(stage_z)
                    rhs_z, electric, magnetic = _triple(
                        field_rhs(_copy(guess), _copy(stage_z)),
                        ("rhs", "electric", "magnetic"),
                    )
                    if rhs_z.shape != z0.shape:
                        raise Failure(
                            "Shape", "field callback returned an invalid Z layout"
                        )
                    if integrated_electric is None:
                        integrated_electric = xp.zeros_like(electric)
                        integrated_magnetic = xp.zeros_like(magnetic)
                    elif (
                        electric.shape != integrated_electric.shape
                        or magnetic.shape != integrated_magnetic.shape
                    ):
                        raise Failure(
                            "Shape", "field callback stage layout changed"
                        )
                    weight = h * _RK4_WEIGHTS[rk_stage] / 6.0
                    integrated_electric = (
                        integrated_electric + weight * electric
                    )
                    integrated_magnetic = (
                        integrated_magnetic + weight * magnetic
                    )
                    counters.quadrature_contributions += 1
                    rates_z.append(rhs_z)
                    if keep_stages:
                        all_stages.append(
                            StageRecord(
                                outer=outer,
                                micro_step=micro,
                                rk_stage=rk_stage,
                                p=_copy(guess),
                                z=_copy(stage_z),
                                p_rhs=None,
                                z_rhs=_copy(rhs_z),
                                p_rhs_available=False,
                                electric=_copy(electric),
                                magnetic=_copy(magnetic),
                            )
                        )
                z = z + (
                    h
                    * (
                        rates_z[0]
                        + 2.0 * rates_z[1]
                        + 2.0 * rates_z[2]
                        + rates_z[3]
                    )
                    / 6.0
                )
            if integrated_electric is None:
                raise Failure("Fields", "field callback produced no stage")

            counters.cheap_L += 1
            last_p, last_z = _copy(guess), _copy(z)
            em_increment = impulse(
                _copy(guess),
                _copy(integrated_electric),
                _copy(integrated_magnetic),
            )
            if em_increment.shape != p0.shape:
                raise Failure("Shape", "impulse callback returned a non-P term")
            endpoint_p = (
                p0
                + H * (d_rate + momentum_rate + pressure_rate)
                + em_increment
            )
            target_p = (p0 + endpoint_p) / 2.0
            residual, raw, normalized, scale, increment = _residual_metrics(
                guess, target_p, scales, xp
            )
            converged = normalized <= float(rtol)
            iterations.append(
                IterationRecord(
                    outer=outer,
                    guess_p=_copy(guess),
                    endpoint_p=_copy(endpoint_p),
                    target_p=_copy(target_p),
                    residual=_copy(residual),
                    residual_raw=raw,
                    residual_normalized=normalized,
                    scale=scale,
                    actual_increment_max=increment,
                    converged=converged,
                    stage_count=4 * M,
                )
            )
            last_p, last_z = _copy(endpoint_p), _copy(z)
            if converged:
                stages = tuple(all_stages) if keep_stages else tuple()
                return ResearchResult(
                    method="C1",
                    p=_copy(endpoint_p),
                    z=_copy(z),
                    midpoint_p=_copy(target_p),
                    endpoint_p=_copy(endpoint_p),
                    integrated_electric=_copy(integrated_electric),
                    integrated_magnetic=_copy(integrated_magnetic),
                    residual=_copy(residual),
                    residual_raw=raw,
                    residual_normalized=normalized,
                    iterations=tuple(iterations),
                    stages=stages,
                    counters=counters.snapshot(),
                    stage_history_bytes=_stage_bytes(stages),
                    metadata=MappingProxyType(
                        {
                            "scope": "bounded research C1",
                            "accepted_product": False,
                            "converged": True,
                        }
                    ),
                )
            guess = target_p
        raise Failure(
            "C1Convergence",
            f"C1 fixed point did not converge after {max_iterations} sweeps",
        )
    except Failure as failure:
        _attach_failure(
            failure,
            counters=counters,
            iterations=iterations,
            stages=all_stages,
            last_p=last_p,
            last_z=last_z,
        )
        raise
    except Exception as error:
        failure = Failure("NumericalLibrary", str(error))
        _attach_failure(
            failure,
            counters=counters,
            iterations=iterations,
            stages=all_stages,
            last_p=last_p,
            last_z=last_z,
        )
        raise failure from error


def frozen_midpoint_callbacks(
    macro: Callable[[Any], Any],
    density_rate: Callable[[Any], Any],
    lorentz: Callable[[Any, Any, Any], Any],
    field_rhs: Callable[[Any, Any], Any],
    p0: Any,
    z0: Any,
    *,
    H: float,
    M: int,
    scales: Any,
    rtol: float = 1.0e-10,
    max_iterations: int = 12,
    keep_stages: bool = True,
    xp: Any = None,
) -> ResearchResult:
    """Run the generic C2 midpoint-frozen-hydro callback kernel."""
    xp = _xp_for(p0, xp)
    _validate_controls(p0, z0, H, M, scales, rtol, max_iterations)
    if M < 2 or M % 2:
        raise Failure("Config", "C2 requires an even M >= 2")
    counters = Counters()
    iterations: list[IterationRecord] = []
    all_stages: list[StageRecord] = []
    guess = _copy(p0)
    last_p = _copy(p0)
    last_z = _copy(z0)
    h = float(H) / M
    if not math.isfinite(h) or h <= 0.0:
        raise Failure("TimeResolution", "microstep is not representable")
    try:
        for outer in range(max_iterations):
            counters.actual_outer_calls += 1
            counters.full_hydro_eval += 1
            last_p, last_z = _copy(guess), _copy(z0)
            frozen_momentum, frozen_pressure = _pair(
                macro(_copy(guess)),
                ("momentum_rate", "pressure_rate"),
            )
            if (
                frozen_momentum.shape != p0.shape
                or frozen_pressure.shape != p0.shape
            ):
                raise Failure("Shape", "C2 macro callback returned a non-P term")
            p, z = _copy(p0), _copy(z0)
            integrated_electric = None
            integrated_magnetic = None
            midpoint = None
            for micro in range(M):
                rates_p = []
                rates_z = []
                for rk_stage in range(4):
                    if rk_stage == 0:
                        stage_p, stage_z = _copy(p), _copy(z)
                    elif rk_stage == 1:
                        stage_p = p + (h / 2.0) * rates_p[0]
                        stage_z = z + (h / 2.0) * rates_z[0]
                    elif rk_stage == 2:
                        stage_p = p + (h / 2.0) * rates_p[1]
                        stage_z = z + (h / 2.0) * rates_z[1]
                    else:
                        stage_p = p + h * rates_p[2]
                        stage_z = z + h * rates_z[2]
                    counters.cheap_D += 1
                    last_p, last_z = _copy(stage_p), _copy(stage_z)
                    d_rate = density_rate(_copy(stage_p))
                    counters.field_rhs += 1
                    last_p, last_z = _copy(stage_p), _copy(stage_z)
                    rhs_z, electric, magnetic = _triple(
                        field_rhs(_copy(stage_p), _copy(stage_z)),
                        ("rhs", "electric", "magnetic"),
                    )
                    counters.cheap_L += 1
                    last_p, last_z = _copy(stage_p), _copy(stage_z)
                    l_rate = lorentz(
                        _copy(stage_p), _copy(electric), _copy(magnetic)
                    )
                    for value in (d_rate, l_rate, rhs_z):
                        expected = p0.shape if value is not rhs_z else z0.shape
                        if value.shape != expected:
                            raise Failure(
                                "Shape", "C2 callback returned an invalid stage term"
                            )
                    rhs_p = d_rate + frozen_momentum + l_rate + frozen_pressure
                    if integrated_electric is None:
                        integrated_electric = xp.zeros_like(electric)
                        integrated_magnetic = xp.zeros_like(magnetic)
                    weight = h * _RK4_WEIGHTS[rk_stage] / 6.0
                    integrated_electric = (
                        integrated_electric + weight * electric
                    )
                    integrated_magnetic = (
                        integrated_magnetic + weight * magnetic
                    )
                    rates_p.append(rhs_p)
                    rates_z.append(rhs_z)
                    counters.quadrature_contributions += 1
                    if keep_stages:
                        all_stages.append(
                            StageRecord(
                                outer=outer,
                                micro_step=micro,
                                rk_stage=rk_stage,
                                p=_copy(stage_p),
                                z=_copy(stage_z),
                                p_rhs=_copy(rhs_p),
                                z_rhs=_copy(rhs_z),
                                p_rhs_available=True,
                                electric=_copy(electric),
                                magnetic=_copy(magnetic),
                            )
                        )
                p = p + (
                    h
                    * (
                        rates_p[0]
                        + 2.0 * rates_p[1]
                        + 2.0 * rates_p[2]
                        + rates_p[3]
                    )
                    / 6.0
                )
                z = z + (
                    h
                    * (
                        rates_z[0]
                        + 2.0 * rates_z[1]
                        + 2.0 * rates_z[2]
                        + rates_z[3]
                    )
                    / 6.0
                )
                if micro + 1 == M // 2:
                    midpoint = _copy(p)
            if midpoint is None:
                raise Failure("Time", "C2 midpoint history is unavailable")
            residual, raw, normalized, scale, increment = _residual_metrics(
                guess, midpoint, scales, xp
            )
            converged = normalized <= float(rtol)
            iterations.append(
                IterationRecord(
                    outer=outer,
                    guess_p=_copy(guess),
                    endpoint_p=_copy(p),
                    target_p=_copy(midpoint),
                    residual=_copy(residual),
                    residual_raw=raw,
                    residual_normalized=normalized,
                    scale=scale,
                    actual_increment_max=increment,
                    converged=converged,
                    stage_count=4 * M,
                )
            )
            last_p, last_z = _copy(p), _copy(z)
            if converged:
                stages = tuple(all_stages) if keep_stages else tuple()
                return ResearchResult(
                    method="C2",
                    p=_copy(p),
                    z=_copy(z),
                    midpoint_p=_copy(midpoint),
                    endpoint_p=_copy(p),
                    integrated_electric=_copy(integrated_electric),
                    integrated_magnetic=_copy(integrated_magnetic),
                    residual=_copy(residual),
                    residual_raw=raw,
                    residual_normalized=normalized,
                    iterations=tuple(iterations),
                    stages=stages,
                    counters=counters.snapshot(),
                    stage_history_bytes=_stage_bytes(stages),
                    metadata=MappingProxyType(
                        {
                            "scope": "bounded research C2",
                            "accepted_product": False,
                            "converged": True,
                        }
                    ),
                )
            guess = midpoint
        raise Failure(
            "C2Convergence",
            f"C2 fixed point did not converge after {max_iterations} sweeps",
        )
    except Failure as failure:
        _attach_failure(
            failure,
            counters=counters,
            iterations=iterations,
            stages=all_stages,
            last_p=last_p,
            last_z=last_z,
        )
        raise
    except Exception as error:
        failure = Failure("NumericalLibrary", str(error))
        _attach_failure(
            failure,
            counters=counters,
            iterations=iterations,
            stages=all_stages,
            last_p=last_p,
            last_z=last_z,
        )
        raise failure from error


def _p_view(context, p: Any):
    return p.reshape(
        p.shape[0], 5, context.op.Nc
    )


def _p_full(
    context,
    *,
    density: Any | None = None,
    momentum: Any | None = None,
    pressure: Any | None = None,
) -> Any:
    xp = context.compute.xp
    q = 2 * context.p.K + 1
    nc = context.op.Nc
    zero_n = xp.zeros((q, nc), dtype=xp.complex128)
    zero_m = xp.zeros((q, 3 * nc), dtype=xp.complex128)
    zero_p = xp.zeros((q, nc), dtype=xp.complex128)
    return xp.concatenate(
        (
            zero_n if density is None else density,
            zero_m if momentum is None else momentum,
            zero_p if pressure is None else pressure,
        ),
        axis=1,
    )


def _ion_state(context, p: Any) -> FluidState:
    view = _p_view(context, p)
    q = 2 * context.p.K + 1
    nc = context.op.Nc
    return FluidState(
        n=view[:, 0, :].reshape(-1).copy(),
        mom=view[:, 1:4, :].reshape(q, 3 * nc).reshape(-1).copy(),
        perp=view[:, 4, :].reshape(-1).copy(),
        parallel=context.compute.xp.zeros(0, dtype=context.compute.xp.complex128),
    )


def _physical_b_from_retained(
    context, b_values: Any, *, require_magnetization: bool = True
):
    nc = context.op.Nc
    bn_spectral = context.compute.xp.matmul(
        b_values, context.op.QB.T
    )
    bn = core.inverse(bn_spectral.reshape(-1), 3 * nc, context, True)
    samples = bn.reshape(context.N, 3, nc)
    if not bool(context.compute.xp.all(context.compute.xp.isfinite(samples)).item()):
        raise Failure("Finite", "physical magnetic field is nonfinite")
    if not require_magnetization:
        return bn, None
    magnitude = context.compute.xp.sqrt(
        context.compute.xp.sum(context.compute.xp.abs(samples) ** 2, axis=1)
    )
    if not bool(
        context.compute.xp.all(magnitude > context.p.B_min).item()
    ):
        raise Failure("Magnetization", "macro metadata B is below B_min")
    bhat = (samples / magnitude[:, None, :]).reshape(-1)
    return bn, bhat


def _physical_b_field(context, z: Any):
    return _physical_b_from_retained(context, z[:, 2 * context.op.Nc :])


def _wall_initial_solve_count(context) -> int:
    """Count the one initial normal solve per sample only when T has rows."""
    rows = int(getattr(context.op.T, "shape", (0, 0))[0])
    return int(context.N) if rows else 0


def _full_context(context):
    isolated = copy.copy(context)
    isolated.next_generation = max(int(getattr(context, "next_generation", 1)), 2)
    isolated.next_trial = 1
    isolated.next_source = 1
    if hasattr(isolated, "_wall_system_cache"):
        del isolated._wall_system_cache
    if hasattr(isolated, "_wall_system_cache_builds"):
        del isolated._wall_system_cache_builds
    return isolated


def _state_from_packed(context, p: Any, z: Any, field: FieldTerms | None = None):
    xp = context.compute.xp
    nc = context.op.Nc
    q = 2 * context.p.K + 1
    if p.shape != (q, 5 * nc):
        raise Failure("Shape", "packed ion state has the wrong P layout")
    if z.shape != (q, 2 * nc + context.op.Nb):
        raise Failure("Shape", "packed ion state has the wrong Z layout")
    view = _p_view(context, p)
    zview = z.reshape(q, 2 * nc + context.op.Nb)
    electron = context.p.species[1]
    ion = context.p.species[0]
    ratio = float(ion.charge) / (-float(electron.charge))
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise Failure("Model", "ion/electron charge ratio must be positive")
    electron_n = (ratio * view[:, 0, :]).reshape(-1).copy()
    if field is None:
        electron_mom = xp.zeros(
            (q, 3 * nc), dtype=xp.complex128
        ).reshape(-1)
    else:
        density = field.density.reshape(context.N, nc)
        velocity = field.velocity.reshape(context.N, 3, nc)
        electron_mom = core.forward(
            (xp.real(density)[:, None, :] * velocity).reshape(-1),
            3 * nc,
            context,
        ) * float(electron.mass)
    state = State(
        fluid=(
            FluidState(
                n=view[:, 0, :].reshape(-1).copy(),
                mom=view[:, 1:4, :].reshape(-1).copy(),
                perp=view[:, 4, :].reshape(-1).copy(),
                parallel=xp.zeros(0, dtype=xp.complex128),
            ),
            FluidState(
                n=electron_n,
                mom=electron_mom,
                perp=zview[:, :nc].reshape(-1).copy(),
                parallel=zview[:, nc : 2 * nc].reshape(-1).copy(),
            ),
        ),
        B=zview[:, 2 * nc :].reshape(-1).copy(),
        step=0,
        generation=1,
        time=float(context.p.start_time),
    )
    core.check_retained(state, context)
    return state


def _validate_ion_endpoint(state: FluidState, context) -> None:
    """Validate positive physical ion density/pressure without a full RHS."""
    xp = context.compute.xp
    nc = context.op.Nc
    density = core.inverse(state.n, nc, context, True)
    pressure = core.inverse(state.perp, nc, context, True)
    if not bool(
        xp.all(xp.isfinite(xp.real(density)))
        and xp.all(xp.isfinite(xp.real(pressure)))
    ):
        raise Failure("Finite", "endpoint ion density/pressure is non-finite")
    ion = context.p.species[0]
    if not bool(xp.all(xp.real(density) > ion.n_min).item()):
        raise Failure("Domain", "endpoint ion density is below its floor")
    if not bool(xp.all(xp.real(pressure) > ion.p_min).item()):
        raise Failure("Domain", "endpoint ion pressure is below its floor")


class ReducedHydroAdapter:
    """Documented adapter over fixed production primitives.

    Private FieldBatch behavior is reused only through its public ``rhs``.
    Constraint projection is reused where the production split itself calls it.
    """

    def __init__(self, context) -> None:
        if int(getattr(context.p, "electron_inertia", 0)) != 0:
            raise Failure("Model", "multirate candidates require massless electrons")
        self.context = context
        self.xp = context.compute.xp
        self.field_batch = FieldBatch(context)
        self.counters = Counters()
        self._tag = StageTag(
            step=0,
            generation=1,
            trial=0,
            spectral_context=int(getattr(context, "spectral_id", 0)),
            time=float(context.p.start_time),
            stage="multirate_split",
        )

    def macro(
        self, p: Any, z: Any, *, include_density_rate: bool = True
    ) -> MacroTerms:
        """Evaluate the one allowed full hydro reconstruction."""
        self.counters.full_hydro_eval += 1
        bn, bhat = _physical_b_field(self.context, z)
        ion = build_fluid(
            _ion_state(self.context, p),
            0,
            bn,
            bhat,
            self._tag,
            self.context,
        )
        build_non_electric(ion, 0, bn, bhat, self.context)
        force = -ion.convection + ion.pressure_force
        self.counters.wall_projection += 1
        self.counters.wall_normal_initial_solves += (
            _wall_initial_solve_count(self.context)
        )
        projected = project_wall(ion.wall, force, self.context)
        if include_density_rate:
            density_rate = _p_full(
                self.context,
                density=-ion.divergence_hat.reshape(
                    2 * self.context.p.K + 1, self.context.op.Nc
                ),
            )
            self.counters.cheap_D += 1
        else:
            density_rate = _p_full(self.context)
        momentum_rate = _p_full(
            self.context,
            momentum=core.forward(
                projected.value, 3 * self.context.op.Nc, self.context
            ).reshape(2 * self.context.p.K + 1, 3 * self.context.op.Nc),
        )
        pressure_rate = _p_full(
            self.context,
            pressure=core.forward(
                ion.pdot_perp, self.context.op.Nc, self.context
            ).reshape(2 * self.context.p.K + 1, self.context.op.Nc),
        )
        return MacroTerms(density_rate, momentum_rate, pressure_rate)

    def field(self, p: Any, z: Any) -> FieldTerms:
        """Evaluate one real massless-electron/field RHS."""
        self.counters.field_rhs += 1
        raw = self.field_batch.rhs(p[None], z[None])
        nc = self.context.op.Nc
        return FieldTerms(
            rhs=raw.rhs[0],
            electric=raw.electric[0].reshape(-1),
            e0=raw.E0[0].reshape(-1),
            magnetic=z[:, 2 * nc :].copy(),
            density=raw.density[0],
            velocity=raw.velocity[0],
            current=raw.current[0],
            checks=raw.checks,
            raw=raw,
        )

    def density_rate(self, p: Any) -> Any:
        """Evaluate only the continuity term D(P)."""
        self.counters.cheap_D += 1
        ion = _ion_state(self.context, p)
        q = 2 * self.context.p.K + 1
        flux_hat = (
            (ion.mom / float(self.context.p.species[0].mass))
            .reshape(q, 3 * self.context.op.Nc)
            @ self.context.op.I.T.conj()
        ).reshape(-1)
        divergence = core.mode_apply(
            self.context.op.DJ, flux_hat, self.context.compute
        )
        return _p_full(
            self.context,
            density=-divergence.reshape(
                2 * self.context.p.K + 1, self.context.op.Nc
            ),
        )

    def lorentz(
        self,
        p: Any,
        electric: Any,
        magnetic: Any,
        *,
        require_magnetization: bool = True,
    ) -> Any:
        """Evaluate only L(P,E,B), including one wall projection."""
        self.counters.cheap_L += 1
        context = self.context
        xp = self.xp
        nc = context.op.Nc
        ne = context.op.Ne
        ion = _ion_state(context, p)
        density = core.inverse(ion.n, nc, context, True)
        momentum = core.inverse(ion.mom, 3 * nc, context, True).reshape(
            context.N, 3, nc
        )
        e_physical = core.inverse(
            electric,
            ne,
            context,
            True,
            context.p.ref.electric,
        ).reshape(context.N, ne)
        b_physical, _ = _physical_b_from_retained(
            context,
            magnetic,
            require_magnetization=require_magnetization,
        )
        b_physical = b_physical.reshape(context.N, 3, nc)
        interpolated = core.sample_apply(
            context.op.H,
            e_physical.reshape(-1),
            context.N,
            context.compute,
        ).reshape(context.N, 3, nc)
        qi = float(context.p.species[0].charge)
        mi = float(context.p.species[0].mass)
        electric_force = (
            qi * density.reshape(context.N, 1, nc) * interpolated
        )
        magnetic_force = (qi / mi) * xp.stack(
            (
                momentum[:, 1, :] * b_physical[:, 2, :]
                - momentum[:, 2, :] * b_physical[:, 1, :],
                momentum[:, 2, :] * b_physical[:, 0, :]
                - momentum[:, 0, :] * b_physical[:, 2, :],
                momentum[:, 0, :] * b_physical[:, 1, :]
                - momentum[:, 1, :] * b_physical[:, 0, :],
            ),
            axis=1,
        )
        wall = build_wall(
            density.reshape(-1),
            context.p.species[0],
            self._tag,
            context,
        )
        self.counters.wall_projection += 1
        self.counters.wall_normal_initial_solves += (
            _wall_initial_solve_count(context)
        )
        projected = project_wall(
            wall,
            (electric_force + magnetic_force).reshape(-1),
            context,
        )
        return _p_full(
            context,
            momentum=core.forward(
                projected.value, 3 * nc, context
            ).reshape(2 * context.p.K + 1, 3 * nc),
        )

    def impulse(self, p: Any, electric: Any, magnetic: Any) -> Any:
        """Apply the interval-integrated electromagnetic force once."""
        return self.lorentz(
            p,
            electric,
            magnetic,
            require_magnetization=False,
        )

    def endpoint(self, p: Any, z: Any):
        """Reconstruct and validate endpoint electron/field compatibility."""
        self.counters.endpoint_check += 1
        field = self.field(p, z)
        state = _state_from_packed(self.context, p, z, field)
        _validate_ion_endpoint(state.fluid[0], self.context)
        cache = SimpleNamespace(
            E=field.electric,
            E0=field.e0,
            current=field.current,
            density=field.density,
            velocity=field.velocity,
            checks=field.checks,
            tag=self._tag,
            version=field.raw.identity,
            scope="research endpoint reconstruction; not accepted production cache",
        )
        return field, state, cache

    def full_ion_rhs(self, p: Any, z: Any):
        """Evaluate a complete production-equivalent ion RHS on a copy context."""
        self.counters.full_hydro_eval += 1
        self.counters.endpoint_full_ion_RHS += 1
        self.counters.endpoint_check += 1
        local = _full_context(self.context)
        state = _state_from_packed(local, p, z)
        tag = StageTag(
            step=0,
            generation=1,
            trial=0,
            spectral_context=int(getattr(local, "spectral_id", 0)),
            time=float(local.p.start_time),
            stage="multirate_full_endpoint_check",
            iteration=0,
        )
        stage = evaluate_stage(state, tag, local, False)
        q = 2 * local.p.K + 1
        nc = local.op.Nc
        full = self.xp.empty(
            (q, 5 * nc), dtype=self.xp.complex128
        )
        full[:, :nc] = stage.rhs.fluid[0].n.reshape(q, nc)
        full[:, nc : 4 * nc] = stage.rhs.fluid[0].mom.reshape(q, 3 * nc)
        full[:, 4 * nc :] = stage.rhs.fluid[0].perp.reshape(q, nc)
        return full, stage

    def check_endpoint_split(self, result: ResearchResult):
        """Compare the endpoint split with one independent full ion RHS."""
        macro = self.macro(result.p, result.z)
        field = result.endpoint
        if field is None:
            field = self.field(result.p, result.z)
        lorentz = self.lorentz(result.p, field.electric, result.z[:, 2 * self.context.op.Nc :])
        split = (
            macro.density_rate
            + macro.momentum_rate
            + macro.pressure_rate
            + lorentz
        )
        full, stage = self.full_ion_rhs(result.p, result.z)
        difference = self.xp.abs(split - full)
        raw = _max_abs(difference, self.xp)
        p_scales = _physical_scales(self.context).p
        normalizer = self.xp.maximum(
            self.xp.maximum(
                self.xp.abs(p_scales),
                self.xp.abs(full),
            ),
            1.0e-300,
        )
        relative = _max_abs(
            difference / normalizer,
            self.xp,
        )
        return SimpleNamespace(
            max_abs=raw,
            relative=relative,
            passed=bool(relative <= 2.0e-9),
            split=split,
            full=full,
            stage=stage,
        )


def _physical_scales(context) -> PhysicalScales:
    """Return P/Z scales in their retained spectral component units."""
    nc = context.op.Nc
    nb = context.op.Nb
    q = 2 * context.p.K + 1
    ion = context.p.species[0]
    xp = context.compute.xp
    density = float(context.p.ref.ni)
    momentum = (
        float(ion.mass)
        * float(context.p.ref.ni)
        * float(context.p.ref.velocity)
    )
    pressure = float(context.p.ref.pressure)
    magnetic = float(context.p.ref.magnetic)
    values = (density, momentum, pressure, magnetic)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise Failure(
            "Scales", "P/Z physical reference scales must be positive and finite"
        )
    p_components = xp.concatenate(
        (
            xp.full((nc,), density, dtype=xp.float64),
            xp.full((3 * nc,), momentum, dtype=xp.float64),
            xp.full((nc,), pressure, dtype=xp.float64),
        )
    )
    z_components = xp.concatenate(
        (
            xp.full((2 * nc,), pressure, dtype=xp.float64),
            xp.full((nb,), magnetic, dtype=xp.float64),
        )
    )
    return PhysicalScales(
        p=xp.tile(p_components, (q, 1)),
        z=xp.tile(z_components, (q, 1)),
    )


def _field_batch_work(adapter: ReducedHydroAdapter):
    return MappingProxyType(
        {
            str(key): int(value)
            for key, value in adapter.field_batch.work.items()
        }
    )


def run(
    method: str,
    context,
    p0: Any,
    z0: Any,
    H: float,
    M: int,
    *,
    rtol: float = 1.0e-10,
    max_iterations: int = 12,
    keep_stages: bool = True,
    endpoint_check: bool = False,
) -> ResearchResult:
    """Run one bounded research window without a production commit."""
    method = str(method).upper()
    if method not in _METHODS:
        raise Failure("Config", f"unknown multirate method: {method}")
    adapter = ReducedHydroAdapter(context)
    scales = _physical_scales(context)
    if method == "C1":
        try:
            result = centered_impulse_callbacks(
                lambda p: adapter.macro(p, z0),
                adapter.field,
                adapter.impulse,
                p0,
                z0,
                H=H,
                M=M,
                scales=scales.p,
                rtol=rtol,
                max_iterations=max_iterations,
                keep_stages=keep_stages,
                xp=context.compute.xp,
            )
        except Failure as failure:
            failure.research_field_batch_work = _field_batch_work(adapter)
            failure.research_counters = _combined_counters(
                getattr(failure, "research_counters", MappingProxyType({})),
                adapter.counters.snapshot(),
            )
            raise
    else:
        try:
            result = frozen_midpoint_callbacks(
                lambda p: adapter.macro(
                    p, z0, include_density_rate=False
                ),
                adapter.density_rate,
                adapter.lorentz,
                adapter.field,
                p0,
                z0,
                H=H,
                M=M,
                scales=scales.p,
                rtol=rtol,
                max_iterations=max_iterations,
                keep_stages=keep_stages,
                xp=context.compute.xp,
            )
        except Failure as failure:
            failure.research_field_batch_work = _field_batch_work(adapter)
            failure.research_counters = _combined_counters(
                getattr(failure, "research_counters", MappingProxyType({})),
                adapter.counters.snapshot(),
            )
            raise
    base_counters = result.counters
    try:
        endpoint, state, cache = adapter.endpoint(result.p, result.z)
    except Failure as failure:
        failure.research_field_batch_work = _field_batch_work(adapter)
        _attach_failure(
            failure,
            counters=adapter.counters,
            iterations=list(result.iterations),
            stages=list(result.stages),
            last_p=result.p,
            last_z=result.z,
        )
        failure.research_counters = _combined_counters(
            base_counters, adapter.counters.snapshot()
        )
        raise
    result = replace(
        result,
        counters=_combined_counters(
            base_counters, adapter.counters.snapshot()
        ),
        endpoint=endpoint,
        state=state,
        cache=cache,
    )
    if endpoint_check:
        try:
            check = adapter.check_endpoint_split(result)
        except Failure as failure:
            failure.research_field_batch_work = _field_batch_work(adapter)
            _attach_failure(
                failure,
                counters=adapter.counters,
                iterations=list(result.iterations),
                stages=list(result.stages),
                last_p=result.p,
                last_z=result.z,
            )
            failure.research_counters = _combined_counters(
                base_counters, adapter.counters.snapshot()
            )
            raise
        result = replace(
            result,
            counters=_combined_counters(
                base_counters, adapter.counters.snapshot()
            ),
            endpoint_check=check,
        )
        if not check.passed:
            failure = Failure(
                "EndpointSplit",
                f"endpoint split relative residual {check.relative:.6e}",
            )
            failure.research_field_batch_work = _field_batch_work(adapter)
            _attach_failure(
                failure,
                counters=adapter.counters,
                iterations=list(result.iterations),
                stages=list(result.stages),
                last_p=result.p,
                last_z=result.z,
            )
            failure.research_counters = _combined_counters(
                base_counters, adapter.counters.snapshot()
            )
            raise failure
    metadata = dict(result.metadata)
    metadata["field_batch_work"] = dict(_field_batch_work(adapter))
    return replace(result, metadata=MappingProxyType(metadata))
