"""Few-node multirate waveform-relaxation window.

This module implements the selected A1-WR-050 mixed-quadrature system.  The
state arrays are retained in the caller-selected ``xp`` backend as
``complex128`` values.  ``theta`` in every callback is the normalized macro
fraction in ``[0, 1]``; physical time and particle/mode layouts remain owned by
the caller's transport and field adapters.  The field callback's second return
value is the grid response ``C`` (continuity plus Lorentz).

The solver owns only its trial arrays and identities.  Callback records are
borrowed objects.  Input ``p0`` and ``z0`` arrays are read-only by contract and
are never mutated.  A successful result owns the returned trial arrays.  A
failure leaves the inputs unchanged and exposes the current device arrays in
``Failure.partial_arrays``.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Literal

import numpy as np

from .types import Failure


_RK4_NODES = (0.0, 0.5, 0.5, 1.0)
_RK4_ENDPOINT_WEIGHTS = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)


@dataclass(frozen=True)
class WindowIdentity:
    """Immutable identity for one transport or field callback."""

    iteration: int
    micro_step: int | None
    rk_stage: int | None
    coarse_node: int | None
    role: str
    theta: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "micro_step": self.micro_step,
            "rk_stage": self.rk_stage,
            "coarse_node": self.coarse_node,
            "role": self.role,
            "theta": self.theta,
        }


@dataclass(frozen=True)
class FineSample:
    """One final-sweep RK4 stage; arrays are owned by the result."""

    identity: WindowIdentity
    P: Any
    Z: Any
    G: Any
    C: Any
    weight: float
    record: Any

    @property
    def grid_response(self) -> Any:
        return self.C


@dataclass(frozen=True)
class CoarseSample:
    """One final coarse-node sample and its transport call."""

    coarse_node: int
    theta: float
    P: Any
    Z: Any
    A: Any
    C_integral: Any
    identity: WindowIdentity
    record: Any


@dataclass(frozen=True)
class DenseSegment:
    """The microstep and local RK4 fraction used for one coarse node."""

    coarse_node: int
    micro_step: int
    x: float
    theta: float


@dataclass
class WindowResult:
    """Accepted few-node window and the final sweep that established it."""

    P: Any
    P_endpoint: Any
    mapped_nodes: Any
    mapped_endpoint: Any
    Z_endpoint: Any
    Z_coarse: Any
    residual: Any
    scaled_residual: Any
    normalized_residual: Any
    residual_scale: Any
    residual_node_linf: Any
    residual_component_linf: Any
    residual_component_normalized_max: Any
    residual_history: tuple[dict[str, Any], ...]
    coarse_nodes: np.ndarray
    lagrange: tuple[np.polynomial.Polynomial, ...]
    Q: np.ndarray
    coarse_A: Any
    C_integrals: Any
    coarse_samples: tuple[CoarseSample, ...]
    fine_P: Any
    fine_Z: Any
    fine_G: Any
    fine_C: Any
    fine_weights: Any
    fine_identities: tuple[WindowIdentity, ...]
    fine_records: tuple[Any, ...]
    fine_samples: tuple[FineSample, ...]
    dense_segments: tuple[DenseSegment, ...]
    initialization: dict[str, Any]
    work: dict[str, Any]
    iterations: int
    ion_inner_iterations: int
    final_input_version: int
    mapped_node_version: int
    H: float
    M: int
    s: int
    h: float
    rtol: float
    max_iterations: int
    relaxation: float
    initial: str
    p_scales: Any
    input_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)

    @property
    def sweeps(self) -> int:
        return self.iterations

    @property
    def accepted_node_version(self) -> int:
        return self.final_input_version

    @property
    def P_nodes(self) -> Any:
        return self.P

    @property
    def final_fine_samples(self) -> tuple[FineSample, ...]:
        return self.fine_samples

    @property
    def residual_max_scaled(self) -> float:
        return float(self.residual_history[-1]["scaled_max"])

    @property
    def endpoint_residual(self) -> Any:
        return self.residual[-1]

    @property
    def raw_norm(self) -> float:
        return float(self.residual_history[-1]["raw_linf"])

    @property
    def scaled_norm(self) -> float:
        return float(self.residual_history[-1]["scaled_linf"])

    @property
    def input_version(self) -> int:
        return self.final_input_version

    @property
    def fine_grid_response(self) -> Any:
        return self.fine_C

    @property
    def grid_response_integrals(self) -> Any:
        return self.C_integrals

    @property
    def component_scale(self) -> Any:
        return self.residual_scale


def _input_failure(message: str) -> Failure:
    return Failure("MultirateInput", message)


def _positive_real(value: Any, name: str) -> float:
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        raise _input_failure(f"{name} must be a finite positive real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise _input_failure(f"{name} must be a finite positive real scalar")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if not isinstance(value, numbers.Integral) or isinstance(value, bool):
        raise _input_failure(f"{name} must be a positive non-boolean integer")
    result = int(value)
    if result <= 0:
        raise _input_failure(f"{name} must be a positive non-boolean integer")
    return result


def _validate_controls(
    H: Any,
    M: Any,
    s: Any,
    rtol: Any,
    max_iterations: Any,
    relaxation: Any,
    initial: Any,
) -> tuple[float, int, int, float, int, float, str]:
    duration = _positive_real(H, "H")
    micro_steps = _positive_integer(M, "M")
    nodes = _positive_integer(s, "s")
    if nodes not in (3, 5):
        raise _input_failure("s must be 3 or 5")
    tolerance = _positive_real(rtol, "rtol")
    iteration_cap = _positive_integer(max_iterations, "max_iterations")
    relax = _positive_real(relaxation, "relaxation")
    if relax > 1.0:
        raise _input_failure("relaxation must not exceed 1")
    if not isinstance(initial, str) or initial not in ("tangent", "constant"):
        raise _input_failure("initial must be 'tangent' or 'constant'")
    micro_h = duration / micro_steps
    if not math.isfinite(micro_h) or micro_h <= 0.0:
        raise _input_failure("H/M is not a finite positive microstep")
    return duration, micro_steps, nodes, tolerance, iteration_cap, relax, initial


def _is_xp_array(value: Any, xp: Any) -> bool:
    try:
        return isinstance(value, xp.ndarray)
    except Exception:
        return False


def _all_finite(value: Any, xp: Any) -> bool:
    return bool(xp.all(xp.isfinite(value)))


def _validate_inputs(
    p0: Any,
    z0: Any,
    p_scales: Any,
    *,
    xp: Any,
    transport: Any,
    field: Any,
) -> Any:
    if xp is None or not hasattr(xp, "ndarray"):
        raise _input_failure("xp must be NumPy or CuPy")
    for name, value in (("p0", p0), ("z0", z0)):
        if not _is_xp_array(value, xp):
            raise _input_failure(f"{name} must be a {xp.__name__} array")
        if value.dtype != xp.complex128:
            raise _input_failure(f"{name} must have dtype xp.complex128")
        if value.ndim < 1 or value.size == 0:
            raise _input_failure(f"{name} must have a nonempty vector shape")
        if not _all_finite(value, xp):
            raise _input_failure(f"{name} contains non-finite values")
    if not callable(transport) or not callable(field):
        raise _input_failure("transport and field must be callable")
    if not _is_xp_array(p_scales, xp):
        raise _input_failure("p_scales must be a xp array")
    if not xp.issubdtype(p_scales.dtype, xp.floating):
        raise _input_failure("p_scales must have a real floating dtype")
    try:
        scales = xp.broadcast_to(p_scales, p0.shape)
    except ValueError as error:
        raise _input_failure("p_scales must broadcast to p0.shape") from error
    if not _all_finite(scales, xp) or not bool(xp.all(scales > 0.0)):
        raise _input_failure("p_scales must be finite and positive")
    return scales


def _require_callback_array(
    value: Any, shape: tuple[int, ...], name: str, xp: Any
) -> Any:
    if not _is_xp_array(value, xp):
        raise Failure("MultirateCallback", f"{name} must be a xp array")
    if value.dtype != xp.complex128:
        raise Failure("MultirateCallback", f"{name} must have dtype xp.complex128")
    if value.shape != shape:
        raise Failure(
            "MultirateCallback",
            f"{name} shape {value.shape} does not match {shape}",
        )
    if not _all_finite(value, xp):
        raise Failure("MultirateCallback", f"{name} contains non-finite values")
    return value


def _transport_result(result: Any, shape: tuple[int, ...], xp: Any) -> tuple[Any, Any]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise Failure(
            "MultirateCallback", "transport must return (A, record)"
        )
    A, record = result
    return _require_callback_array(A, shape, "transport A", xp), record


def _field_result(
    result: Any,
    p_shape: tuple[int, ...],
    z_shape: tuple[int, ...],
    xp: Any,
) -> tuple[Any, Any, Any]:
    if not isinstance(result, tuple) or len(result) != 3:
        raise Failure(
            "MultirateCallback", "field must return (G, C, record)"
        )
    G, C, record = result
    return (
        _require_callback_array(G, z_shape, "field G", xp),
        _require_callback_array(C, p_shape, "field C", xp),
        record,
    )


@lru_cache(maxsize=2)
def _gauss_lobatto_rule_cached(
    s: int,
) -> tuple[np.ndarray, tuple[np.polynomial.Polynomial, ...], np.ndarray]:
    if s not in (3, 5):
        raise _input_failure("s must be 3 or 5")
    roots = np.polynomial.legendre.Legendre.basis(s - 1).deriv().roots()
    roots = np.asarray(roots, dtype=np.complex128)
    if roots.shape != (s - 2,):
        raise RuntimeError("unexpected Gauss-Lobatto interior-root count")
    imaginary_tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(roots.real), initial=0.0)))
    )
    if float(np.max(np.abs(roots.imag), initial=0.0)) > imaginary_tolerance:
        raise RuntimeError("Gauss-Lobatto interior roots are not real")
    interior = np.sort((roots.real + 1.0) / 2.0)
    nodes = np.concatenate((np.array([0.0]), interior, np.array([1.0])))
    nodes.setflags(write=False)
    if not np.all(np.isfinite(nodes)) or not np.all(np.diff(nodes) > 0.0):
        raise RuntimeError("invalid Gauss-Lobatto nodes")

    lagrange: list[np.polynomial.Polynomial] = []
    for j in range(s):
        polynomial = np.polynomial.Polynomial([1.0])
        for k in range(s):
            if k != j:
                polynomial *= np.polynomial.Polynomial(
                    [-nodes[k], 1.0]
                ) / (nodes[j] - nodes[k])
        lagrange.append(polynomial)

    q = np.empty((s, s), dtype=np.float64)
    for i, endpoint in enumerate(nodes):
        for j, polynomial in enumerate(lagrange):
            q[i, j] = float(polynomial.integ()(float(endpoint)))
    q.setflags(write=False)
    return nodes, tuple(lagrange), q


def gauss_lobatto_rule(
    s: int,
) -> tuple[np.ndarray, tuple[np.polynomial.Polynomial, ...], np.ndarray]:
    """Return CPU ``nodes``, cardinal polynomials, and ``Q_il`` for ``s=3/5``."""

    if not isinstance(s, numbers.Integral) or isinstance(s, bool) or s not in (3, 5):
        raise _input_failure("s must be 3 or 5")
    return _gauss_lobatto_rule_cached(int(s))


def _lagrange_values(
    lagrange: tuple[np.polynomial.Polynomial, ...],
    frictions: np.ndarray,
) -> np.ndarray:
    values = np.empty(frictions.shape + (len(lagrange),), dtype=np.float64)
    for j, polynomial in enumerate(lagrange):
        values[..., j] = polynomial(frictions)
    return values


def _rk4_dense_weights(x: float) -> np.ndarray:
    return np.array(
        [
            x - 1.5 * x * x + (2.0 / 3.0) * x**3,
            x * x - (2.0 / 3.0) * x**3,
            x * x - (2.0 / 3.0) * x**3,
            -0.5 * x * x + (2.0 / 3.0) * x**3,
        ],
        dtype=np.float64,
    )


def _node_segments(
    nodes: np.ndarray, micro_steps: int
) -> tuple[tuple[int, int, float], ...]:
    segments = []
    for node, theta in enumerate(nodes):
        scaled = float(theta) * micro_steps
        micro_step = min(int(math.floor(scaled)), micro_steps - 1)
        x = scaled - micro_step
        if micro_step == micro_steps - 1 and math.isclose(
            x, 1.0, rel_tol=0.0, abs_tol=8.0 * np.finfo(float).eps
        ):
            x = 1.0
        segments.append((node, micro_step, float(x)))
    return tuple(segments)


@dataclass
class _FineTrial:
    P: Any
    Z: Any
    G: Any
    C: Any
    identities: tuple[WindowIdentity, ...]
    records: tuple[Any, ...]


@dataclass
class _PartialScan:
    iteration: int
    P: list[Any] = field(default_factory=list)
    Z: list[Any] = field(default_factory=list)
    G: list[Any] = field(default_factory=list)
    C: list[Any] = field(default_factory=list)
    identities: list[WindowIdentity] = field(default_factory=list)
    records: list[Any] = field(default_factory=list)
    attempted_identity: WindowIdentity | None = None
    attempted_P: Any | None = None
    attempted_Z: Any | None = None


@dataclass
class _PartialCoarse:
    iteration: int
    A: list[Any] = field(default_factory=list)
    identities: list[WindowIdentity] = field(default_factory=list)
    records: list[Any] = field(default_factory=list)
    attempted_identity: WindowIdentity | None = None
    attempted_P: Any | None = None
    attempted_Z: Any | None = None


def _scan_field(
    old_nodes: Any,
    z0: Any,
    *,
    iteration: int,
    micro_steps: int,
    micro_h: float,
    p_stage_basis: Any,
    field: Callable[..., Any],
    xp: Any,
    p_shape: tuple[int, ...],
    z_shape: tuple[int, ...],
    work: dict[str, Any],
    partial: _PartialScan,
) -> _FineTrial:
    p_stages = xp.tensordot(
        p_stage_basis, old_nodes, axes=([-1], [0])
    )
    fine_P = xp.empty((micro_steps, 4) + p_shape, dtype=xp.complex128)
    fine_Z = xp.empty((micro_steps, 4) + z_shape, dtype=xp.complex128)
    fine_G = xp.empty((micro_steps, 4) + z_shape, dtype=xp.complex128)
    fine_C = xp.empty((micro_steps, 4) + p_shape, dtype=xp.complex128)
    identities: list[WindowIdentity] = []
    records: list[Any] = []
    fine_P[...] = p_stages

    z_current = z0
    for micro_step in range(micro_steps):
        z_left = z_current
        for rk_stage, fraction in enumerate(_RK4_NODES):
            theta = (micro_step + fraction) / micro_steps
            if rk_stage == 0:
                z_stage = z_left
            elif rk_stage == 1:
                z_stage = z_left + (micro_h / 2.0) * fine_G[micro_step, 0]
            elif rk_stage == 2:
                z_stage = z_left + (micro_h / 2.0) * fine_G[micro_step, 1]
            else:
                z_stage = z_left + micro_h * fine_G[micro_step, 2]
            identity = WindowIdentity(
                iteration=int(iteration),
                micro_step=micro_step,
                rk_stage=rk_stage,
                coarse_node=None,
                role="field_rk4",
                theta=float(theta),
            )
            p_stage = p_stages[micro_step, rk_stage]
            partial.attempted_identity = identity
            partial.attempted_P = p_stage
            partial.attempted_Z = z_stage
            work["field_callbacks"] += 1
            work["field_calls"] += 1
            work["grid_L_evaluations"] += 1
            work["grid_L_calls"] += 1
            work["grid_response_evaluations"] += 1
            work["rk4_stages"] += 1
            work["rk4_stage_calls"] += 1
            raw_field = field(theta, p_stage, z_stage, identity)
            G, C, record = _field_result(
                raw_field,
                p_shape,
                z_shape,
                xp,
            )
            fine_Z[micro_step, rk_stage] = z_stage
            fine_G[micro_step, rk_stage] = G
            fine_C[micro_step, rk_stage] = C
            partial.P.append(p_stage)
            partial.Z.append(z_stage)
            partial.G.append(G.copy())
            partial.C.append(C.copy())
            partial.identities.append(identity)
            partial.records.append(record)
            identities.append(identity)
            records.append(record)
        z_current = z_left + micro_h * (
            fine_G[micro_step, 0]
            + 2.0 * fine_G[micro_step, 1]
            + 2.0 * fine_G[micro_step, 2]
            + fine_G[micro_step, 3]
        ) / 6.0
        if not _all_finite(z_current, xp):
            raise Failure(
                "MultirateCallback",
                f"field endpoint is non-finite in iteration {iteration}",
            )

    return _FineTrial(
        P=fine_P,
        Z=fine_Z,
        G=fine_G,
        C=fine_C,
        identities=tuple(identities),
        records=tuple(records),
    )


def _coarse_field_values(
    trial: _FineTrial,
    segments: tuple[tuple[int, int, float], ...],
    *,
    micro_h: float,
    p_shape: tuple[int, ...],
    z_shape: tuple[int, ...],
    dense_weights: Any,
    endpoint_weights: Any,
    xp: Any,
) -> tuple[Any, Any, Any]:
    micro_steps = trial.Z.shape[0]
    z_coarse = xp.empty((len(segments),) + z_shape, dtype=xp.complex128)
    c_coarse = xp.empty((len(segments),) + p_shape, dtype=xp.complex128)
    c_completed = xp.zeros(p_shape, dtype=xp.complex128)
    node_cursor = 0
    for micro_step in range(micro_steps):
        while (
            node_cursor < len(segments)
            and segments[node_cursor][1] == micro_step
        ):
            node, _, x = segments[node_cursor]
            weights = dense_weights[node]
            z_coarse[node] = trial.Z[micro_step, 0] + micro_h * xp.tensordot(
                weights, trial.G[micro_step], axes=([0], [0])
            )
            c_coarse[node] = c_completed + micro_h * xp.tensordot(
                weights, trial.C[micro_step], axes=([0], [0])
            )
            node_cursor += 1
        c_completed = c_completed + micro_h * xp.tensordot(
            endpoint_weights, trial.C[micro_step], axes=([0], [0])
        )
    if node_cursor != len(segments):
        raise Failure("MultirateInternal", "coarse-node segment partition failed")
    return z_coarse, c_coarse, trial.Z[micro_steps - 1, 0] + micro_h * (
        trial.G[micro_steps - 1, 0]
        + 2.0 * trial.G[micro_steps - 1, 1]
        + 2.0 * trial.G[micro_steps - 1, 2]
        + trial.G[micro_steps - 1, 3]
    ) / 6.0


def _transport_coarse(
    old_nodes: Any,
    z_coarse: Any,
    nodes: np.ndarray,
    segments: tuple[tuple[int, int, float], ...],
    *,
    iteration: int,
    transport: Callable[..., Any],
    xp: Any,
    p_shape: tuple[int, ...],
    work: dict[str, Any],
    partial: _PartialCoarse,
) -> tuple[Any, tuple[WindowIdentity, ...], tuple[Any, ...]]:
    coarse_A = xp.empty(old_nodes.shape, dtype=xp.complex128)
    identities = []
    records = []
    for i, theta in enumerate(nodes):
        _, micro_step, _ = segments[i]
        identity = WindowIdentity(
            iteration=int(iteration),
            micro_step=micro_step,
            rk_stage=None,
            coarse_node=i,
            role="transport_coarse",
            theta=float(theta),
        )
        partial.attempted_identity = identity
        partial.attempted_P = old_nodes[i].copy()
        partial.attempted_Z = z_coarse[i].copy()
        work["transport_callbacks"] += 1
        work["transport_calls"] += 1
        work["transport_coarse_evaluations"] += 1
        A, record = _transport_result(
            transport(float(theta), old_nodes[i], z_coarse[i], identity),
            p_shape,
            xp,
        )
        coarse_A[i] = A
        partial.A.append(A.copy())
        partial.identities.append(identity)
        partial.records.append(record)
        identities.append(identity)
        records.append(record)
    return coarse_A, tuple(identities), tuple(records)


def _map_nodes(
    p0: Any,
    old_nodes: Any,
    coarse_A: Any,
    c_integrals: Any,
    *,
    H: float,
    q_xp: Any,
    xp: Any,
) -> Any:
    mapped = xp.empty(old_nodes.shape, dtype=xp.complex128)
    mapped[0] = p0
    mapped[1:] = (
        p0
        + H
        * xp.tensordot(q_xp[1:], coarse_A, axes=([1], [0]))
        + c_integrals[1:]
    )
    return mapped


def _residual_metrics(
    old_nodes: Any,
    mapped: Any,
    physical_scales: Any,
    *,
    iteration: int,
    rtol: float,
    xp: Any,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    residual = old_nodes - mapped
    raw_abs = xp.abs(residual)
    old_magnitude = xp.max(xp.abs(old_nodes), axis=0)
    mapped_magnitude = xp.max(xp.abs(mapped), axis=0)
    component_scale = xp.maximum(
        physical_scales, xp.maximum(old_magnitude, mapped_magnitude)
    )
    scaled = raw_abs / component_scale
    normalized = raw_abs / (rtol * component_scale)
    scaled_linf = xp.max(scaled)
    endpoint_scaled = xp.max(scaled[-1])
    normalized_linf = xp.max(normalized)
    endpoint_normalized = xp.max(normalized[-1])
    raw_linf = xp.max(raw_abs)
    raw_l2 = xp.sqrt(xp.sum(raw_abs * raw_abs))
    scaled_l2 = xp.sqrt(xp.sum(scaled * scaled))
    normalized_l2 = xp.sqrt(xp.sum(normalized * normalized))
    if not _all_finite(residual, xp):
        raise Failure(
            "MultirateConvergence",
            f"non-finite residual in iteration {iteration}",
        )
    record = {
        "iteration": int(iteration),
        "raw_linf": float(raw_linf),
        "raw_l2": float(raw_l2),
        "scaled_linf": float(scaled_linf),
        "scaled_l2": float(scaled_l2),
        "scaled_max": float(scaled_linf),
        "normalized_linf": float(normalized_linf),
        "normalized_l2": float(normalized_l2),
        "normalized_max": float(normalized_linf),
        "scale_linf": float(xp.max(component_scale)),
        "endpoint_raw_linf": float(xp.max(raw_abs[-1])),
        "endpoint_scaled_linf": float(endpoint_scaled),
        "endpoint_normalized_linf": float(endpoint_normalized),
        "converged": False,
    }
    return residual, scaled, normalized, record


def _initial_nodes(
    initial: str,
    p0: Any,
    z0: Any,
    nodes: np.ndarray,
    H: float,
    *,
    transport: Callable[..., Any],
    field: Callable[..., Any],
    xp: Any,
    p_shape: tuple[int, ...],
    z_shape: tuple[int, ...],
    work: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    if initial == "constant":
        guessed = xp.broadcast_to(p0, (len(nodes),) + p_shape).copy()
        return guessed, {
            "kind": "constant",
            "transport_identity": None,
            "field_identity": None,
            "transport_record": None,
            "field_record": None,
            "A0": None,
            "G0": None,
            "C0": None,
            "L0": None,
            "F0": None,
        }

    identity = WindowIdentity(
        iteration=-1,
        micro_step=None,
        rk_stage=None,
        coarse_node=0,
        role="tangent_initialization",
        theta=0.0,
    )
    work["transport_callbacks"] += 1
    work["transport_calls"] += 1
    work["initial_transport_callbacks"] += 1
    A0, transport_record = _transport_result(
        transport(0.0, p0, z0, identity), p_shape, xp
    )
    work["field_callbacks"] += 1
    work["field_calls"] += 1
    work["grid_L_evaluations"] += 1
    work["grid_L_calls"] += 1
    work["grid_response_evaluations"] += 1
    work["initial_field_callbacks"] += 1
    G0, C0, field_record = _field_result(
        field(0.0, p0, z0, identity), p_shape, z_shape, xp
    )
    F0 = A0 + C0
    guessed = xp.empty((len(nodes),) + p_shape, dtype=xp.complex128)
    for i, theta in enumerate(nodes):
        guessed[i] = p0 + H * float(theta) * F0
    return guessed, {
        "kind": "tangent",
        "transport_identity": identity,
        "field_identity": identity,
        "transport_record": transport_record,
        "field_record": field_record,
        "A0": A0,
        "G0": G0,
        "C0": C0,
        "L0": C0,
        "F0": F0,
    }


def _work_snapshot(
    work: dict[str, Any], completed_sweeps: int, attempted_sweeps: int | None = None
) -> dict[str, Any]:
    result = dict(work)
    attempted = (
        int(completed_sweeps)
        if attempted_sweeps is None
        else int(attempted_sweeps)
    )
    result["sweeps"] = int(completed_sweeps)
    result["completed_residual_sweeps"] = int(completed_sweeps)
    result["attempted_sweeps"] = attempted
    result["complete_ion_evaluations"] = None
    result["complete_ion_evaluation_source"] = "parent adapter"
    result["transport_calls"] = int(work["transport_callbacks"])
    result["field_calls"] = int(work["field_callbacks"])
    result["grid_L_calls"] = int(work["grid_L_evaluations"])
    result["grid_response_evaluations"] = int(
        work["grid_response_evaluations"]
    )
    result["rk4_stage_calls"] = int(work["rk4_stages"])
    return result


def _attach_failure_evidence(
    failure: Failure,
    *,
    old_nodes: Any | None,
    mapped: Any | None,
    residual: Any | None,
    scaled_residual: Any | None,
    normalized_residual: Any | None,
    residual_scale: Any | None,
    z_coarse: Any | None,
    coarse_A: Any | None,
    trial: _FineTrial | None,
    partial_scan: _PartialScan | None,
    partial_coarse: _PartialCoarse | None,
    residual_history: list[dict[str, Any]],
    work: dict[str, Any],
    controls: dict[str, Any],
    attempted_sweeps: int,
    xp: Any,
) -> None:
    completed_sweeps = len(residual_history)
    snapshot = _work_snapshot(work, completed_sweeps, attempted_sweeps)
    partial: dict[str, Any] = {}
    for name, value in (
        ("P", old_nodes),
        ("mapped_P", mapped),
        ("residual", residual),
        ("scaled_residual", scaled_residual),
        ("normalized_residual", normalized_residual),
        ("residual_scale", residual_scale),
        ("Z_coarse", z_coarse),
        ("coarse_A", coarse_A),
    ):
        if value is not None:
            partial[name] = value
    if trial is not None:
        partial.update(
            {
                "fine_P": trial.P,
                "fine_Z": trial.Z,
                "fine_G": trial.G,
                "fine_C": trial.C,
                "fine_identities": trial.identities,
                "fine_records": trial.records,
            }
        )
    elif partial_scan is not None:
        completed = len(partial_scan.identities)
        if completed:
            partial.update(
                {
                    "fine_P": xp.stack(partial_scan.P, axis=0),
                    "fine_Z": xp.stack(partial_scan.Z, axis=0),
                    "fine_G": xp.stack(partial_scan.G, axis=0),
                    "fine_C": xp.stack(partial_scan.C, axis=0),
                    "partial_identities": tuple(partial_scan.identities),
                    "partial_records": tuple(partial_scan.records),
                    "completed_stage_count": completed,
                }
            )
        if partial_scan.attempted_identity is not None:
            partial.update(
                {
                    "partial_P": partial_scan.attempted_P,
                    "partial_Z": partial_scan.attempted_Z,
                    "partial_identity": partial_scan.attempted_identity,
                }
            )
    if partial_coarse is not None:
        completed_coarse = len(partial_coarse.identities)
        if completed_coarse:
            partial.update(
                {
                    "completed_coarse_A": xp.stack(
                        partial_coarse.A, axis=0
                    ),
                    "coarse_partial_identities": tuple(
                        partial_coarse.identities
                    ),
                    "coarse_partial_records": tuple(partial_coarse.records),
                    "completed_coarse_count": completed_coarse,
                }
            )
        if partial_coarse.attempted_identity is not None:
            partial.update(
                {
                    "attempted_coarse_P": partial_coarse.attempted_P,
                    "attempted_coarse_Z": partial_coarse.attempted_Z,
                    "attempted_coarse_identity": (
                        partial_coarse.attempted_identity
                    ),
                }
            )
    metadata = {
        "attempted_sweeps": int(attempted_sweeps),
        "completed_residual_sweeps": int(completed_sweeps),
        "residual_history": tuple(residual_history),
        "work": snapshot,
        "controls": dict(controls),
        "converged": False,
    }
    failure.attempted_sweeps = int(attempted_sweeps)
    failure.completed_residual_sweeps = int(completed_sweeps)
    failure.attempted_counts = snapshot
    failure.residual_history = tuple(residual_history)
    failure.completed_residual_history = tuple(residual_history)
    failure.residuals = tuple(residual_history)
    failure.work = snapshot
    failure.multirate_metadata = metadata
    if partial_scan is not None:
        failure.multirate_attempted_identity = partial_scan.attempted_identity
        failure.multirate_attempted_P = partial_scan.attempted_P
        failure.multirate_attempted_Z = partial_scan.attempted_Z
        failure.multirate_partial_identities = tuple(partial_scan.identities)
        failure.multirate_partial_records = tuple(partial_scan.records)
        if not hasattr(failure, "attempted_identity"):
            failure.attempted_identity = partial_scan.attempted_identity
        if not hasattr(failure, "completed_stage_count"):
            failure.completed_stage_count = len(partial_scan.identities)
        if not hasattr(failure, "partial_identities"):
            failure.partial_identities = tuple(partial_scan.identities)
        if not hasattr(failure, "partial_records"):
            failure.partial_records = tuple(partial_scan.records)
        if not hasattr(failure, "attempted_P"):
            failure.attempted_P = partial_scan.attempted_P
        if not hasattr(failure, "attempted_Z"):
            failure.attempted_Z = partial_scan.attempted_Z
        if partial_scan.attempted_identity is not None:
            failure.multirate_attempted_micro_step = (
                partial_scan.attempted_identity.micro_step
            )
            failure.multirate_attempted_rk_stage = (
                partial_scan.attempted_identity.rk_stage
            )
            if not hasattr(failure, "attempted_micro_step"):
                failure.attempted_micro_step = (
                    partial_scan.attempted_identity.micro_step
                )
            if not hasattr(failure, "attempted_rk_stage"):
                failure.attempted_rk_stage = (
                    partial_scan.attempted_identity.rk_stage
                )
    if partial_coarse is not None:
        failure.multirate_attempted_coarse_identity = (
            partial_coarse.attempted_identity
        )
        failure.multirate_attempted_coarse_P = partial_coarse.attempted_P
        failure.multirate_attempted_coarse_Z = partial_coarse.attempted_Z
        failure.multirate_coarse_partial_identities = tuple(
            partial_coarse.identities
        )
        failure.multirate_coarse_partial_records = tuple(
            partial_coarse.records
        )
        if not hasattr(failure, "attempted_coarse_identity"):
            failure.attempted_coarse_identity = (
                partial_coarse.attempted_identity
            )
        if not hasattr(failure, "attempted_coarse_P"):
            failure.attempted_coarse_P = partial_coarse.attempted_P
        if not hasattr(failure, "attempted_coarse_Z"):
            failure.attempted_coarse_Z = partial_coarse.attempted_Z
        if not hasattr(failure, "coarse_partial_identities"):
            failure.coarse_partial_identities = tuple(
                partial_coarse.identities
            )
        if not hasattr(failure, "coarse_partial_records"):
            failure.coarse_partial_records = tuple(partial_coarse.records)
    failure.multirate_partial_arrays = dict(partial)
    for name, value in partial.items():
        failure.partial_arrays.setdefault(name, value)


def solve_window(
    p0: Any,
    z0: Any,
    H: Any,
    M: Any,
    s: Any,
    transport: Callable[..., Any],
    field: Callable[..., Any],
    *,
    xp: Any,
    p_scales: Any,
    rtol: Any = 1.0e-10,
    max_iterations: Any = 12,
    relaxation: Any = 1.0,
    initial: Literal["tangent", "constant"] = "tangent",
) -> WindowResult:
    """Solve one selected few-node multirate waveform-relaxation window.

    ``transport(theta, P, Z, identity)`` returns ``(A, record)`` and
    ``field(theta, P, Z, identity)`` returns ``(G, C, record)``, where ``C`` is
    the grid response containing continuity and Lorentz terms.  Array arguments
    and outputs are ``xp.complex128`` and remain on the selected backend.  The
    field scan is a causal RK4 solve from ``z0``; the ion nodes are a single
    immutable coarse polynomial per outer sweep.

    The accepted result is the OLD node array that drove the final field scan.
    The freshly mapped nodes are returned separately as residual evidence.
    """

    duration, micro_steps, node_count, tolerance, iteration_cap, relax, start = (
        _validate_controls(
            H, M, s, rtol, max_iterations, relaxation, initial
        )
    )
    scales = _validate_inputs(
        p0, z0, p_scales, xp=xp, transport=transport, field=field
    )
    nodes, lagrange, q = gauss_lobatto_rule(node_count)
    p_shape = tuple(p0.shape)
    z_shape = tuple(z0.shape)
    micro_h = duration / micro_steps
    segments = _node_segments(nodes, micro_steps)

    stage_frictions = np.empty((micro_steps, 4), dtype=np.float64)
    for micro_step in range(micro_steps):
        for rk_stage, fraction in enumerate(_RK4_NODES):
            stage_frictions[micro_step, rk_stage] = (
                micro_step + fraction
            ) / micro_steps
    stage_basis_cpu = _lagrange_values(lagrange, stage_frictions)
    coarse_dense_cpu = np.stack(
        [_rk4_dense_weights(segment[2]) for segment in segments], axis=0
    )
    p_stage_basis = xp.asarray(stage_basis_cpu)
    q_xp = xp.asarray(q)
    dense_weights = xp.asarray(coarse_dense_cpu)
    endpoint_weights = xp.asarray(
        np.asarray(_RK4_ENDPOINT_WEIGHTS, dtype=np.float64)
    )
    fine_weights_cpu = (
        micro_h * np.asarray(_RK4_ENDPOINT_WEIGHTS, dtype=np.float64)
    )
    fine_weights = xp.broadcast_to(
        xp.asarray(fine_weights_cpu), (micro_steps, 4)
    ).copy()

    work: dict[str, Any] = {
        "transport_callbacks": 0,
        "field_callbacks": 0,
        "grid_L_evaluations": 0,
        "grid_response_evaluations": 0,
        "rk4_stages": 0,
        "transport_calls": 0,
        "field_calls": 0,
        "grid_L_calls": 0,
        "rk4_stage_calls": 0,
        "transport_coarse_evaluations": 0,
        "initial_transport_callbacks": 0,
        "initial_field_callbacks": 0,
        "sweeps": 0,
    }
    controls = {
        "H": duration,
        "M": micro_steps,
        "s": node_count,
        "h": micro_h,
        "rtol": tolerance,
        "max_iterations": iteration_cap,
        "relaxation": relax,
        "initial": start,
    }
    residual_history: list[dict[str, Any]] = []
    old_nodes: Any | None = None
    mapped: Any | None = None
    residual: Any | None = None
    scaled_residual: Any | None = None
    normalized_residual: Any | None = None
    residual_scale: Any | None = None
    z_coarse: Any | None = None
    c_integrals: Any | None = None
    coarse_A: Any | None = None
    trial: _FineTrial | None = None
    partial_scan: _PartialScan | None = None
    partial_coarse: _PartialCoarse | None = None
    attempted_sweeps = 0

    try:
        old_nodes, initialization = _initial_nodes(
            start,
            p0,
            z0,
            nodes,
            duration,
            transport=transport,
            field=field,
            xp=xp,
            p_shape=p_shape,
            z_shape=z_shape,
            work=work,
        )

        for iteration in range(iteration_cap):
            attempted_sweeps += 1
            trial = None
            z_coarse = None
            coarse_A = None
            mapped = None
            residual = None
            scaled_residual = None
            normalized_residual = None
            residual_scale = None
            c_integrals = None
            partial_scan = _PartialScan(iteration=iteration)
            partial_coarse = _PartialCoarse(iteration=iteration)
            trial = _scan_field(
                old_nodes,
                z0,
                iteration=iteration,
                micro_steps=micro_steps,
                micro_h=micro_h,
                p_stage_basis=p_stage_basis,
                field=field,
                xp=xp,
                p_shape=p_shape,
                z_shape=z_shape,
                work=work,
                partial=partial_scan,
            )
            partial_scan = None
            z_coarse, c_integrals, z_endpoint = _coarse_field_values(
                trial,
                segments,
                micro_h=micro_h,
                p_shape=p_shape,
                z_shape=z_shape,
                dense_weights=dense_weights,
                endpoint_weights=endpoint_weights,
                xp=xp,
            )
            coarse_A, transport_identities, transport_records = (
                _transport_coarse(
                    old_nodes,
                    z_coarse,
                    nodes,
                    segments,
                    iteration=iteration,
                    transport=transport,
                    xp=xp,
                    p_shape=p_shape,
                    work=work,
                    partial=partial_coarse,
                )
            )
            partial_coarse = None
            mapped = _map_nodes(
                p0,
                old_nodes,
                coarse_A,
                c_integrals,
                H=duration,
                q_xp=q_xp,
                xp=xp,
            )
            (
                residual,
                scaled_residual,
                normalized_residual,
                residual_record,
            ) = _residual_metrics(
                old_nodes, mapped, scales, iteration=iteration, rtol=tolerance, xp=xp
            )
            residual_scale = xp.maximum(
                scales,
                xp.maximum(
                    xp.max(xp.abs(old_nodes), axis=0),
                    xp.max(xp.abs(mapped), axis=0),
                ),
            )
            converged = residual_record["normalized_max"] <= 1.0
            residual_record["converged"] = bool(converged)
            residual_history.append(residual_record)
            work["sweeps"] = len(residual_history)

            if not converged:
                if iteration + 1 == iteration_cap:
                    failure = Failure(
                        "MultirateConvergence",
                        "multirate window did not converge after "
                        f"{iteration_cap} sweeps; normalized defect "
                        f"{residual_record['normalized_max']:.6e} exceeds 1",
                    )
                    raise failure
                next_nodes = old_nodes + relax * (mapped - old_nodes)
                next_nodes[0] = p0
                if not _all_finite(next_nodes, xp):
                    raise Failure(
                        "MultirateConvergence",
                        "relaxed ion nodes became non-finite",
                    )
                old_nodes = next_nodes
                continue

            raw_abs = xp.abs(residual)
            residual_node_linf = xp.max(raw_abs, axis=0)
            residual_component_linf = residual_node_linf
            residual_component_normalized_max = xp.max(
                xp.abs(normalized_residual), axis=0
            )
            coarse_samples = tuple(
                CoarseSample(
                    coarse_node=i,
                    theta=float(nodes[i]),
                    P=old_nodes[i],
                    Z=z_coarse[i],
                    A=coarse_A[i],
                    C_integral=c_integrals[i],
                    identity=transport_identities[i],
                    record=transport_records[i],
                )
                for i in range(node_count)
            )
            fine_samples = []
            for flat_index, identity in enumerate(trial.identities):
                micro_step = identity.micro_step
                rk_stage = identity.rk_stage
                assert micro_step is not None
                assert rk_stage is not None
                fine_samples.append(
                    FineSample(
                        identity=identity,
                        P=trial.P[micro_step, rk_stage],
                        Z=trial.Z[micro_step, rk_stage],
                        G=trial.G[micro_step, rk_stage],
                        C=trial.C[micro_step, rk_stage],
                        weight=float(fine_weights_cpu[rk_stage]),
                        record=trial.records[flat_index],
                    )
                )
            dense_segments = tuple(
                DenseSegment(
                    coarse_node=node,
                    micro_step=micro_step,
                    x=float(x),
                    theta=float(nodes[node]),
                )
                for node, micro_step, x in segments
            )
            return WindowResult(
                P=old_nodes,
                P_endpoint=old_nodes[-1],
                mapped_nodes=mapped,
                mapped_endpoint=mapped[-1],
                Z_endpoint=z_endpoint,
                Z_coarse=z_coarse,
                residual=residual,
                scaled_residual=scaled_residual,
                normalized_residual=normalized_residual,
                residual_scale=residual_scale,
                residual_node_linf=residual_node_linf,
                residual_component_linf=residual_component_linf,
                residual_component_normalized_max=(
                    residual_component_normalized_max
                ),
                residual_history=tuple(residual_history),
                coarse_nodes=nodes,
                lagrange=lagrange,
                Q=q,
                coarse_A=coarse_A,
                C_integrals=c_integrals,
                coarse_samples=coarse_samples,
                fine_P=trial.P,
                fine_Z=trial.Z,
                fine_G=trial.G,
                fine_C=trial.C,
                fine_weights=fine_weights,
                fine_identities=trial.identities,
                fine_records=trial.records,
                fine_samples=tuple(fine_samples),
                dense_segments=dense_segments,
                initialization=initialization,
                work=_work_snapshot(
                    work, len(residual_history), attempted_sweeps
                ),
                iterations=len(residual_history),
                ion_inner_iterations=0,
                final_input_version=iteration,
                mapped_node_version=iteration + 1,
                H=duration,
                M=micro_steps,
                s=node_count,
                h=micro_h,
                rtol=tolerance,
                max_iterations=iteration_cap,
                relaxation=relax,
                initial=start,
                p_scales=scales,
                input_shapes={"P": p_shape, "Z": z_shape},
            )
    except Failure as failure:
        _attach_failure_evidence(
            failure,
            old_nodes=old_nodes,
            mapped=mapped,
            residual=residual,
            scaled_residual=scaled_residual,
            normalized_residual=normalized_residual,
            residual_scale=residual_scale,
            z_coarse=z_coarse,
            coarse_A=coarse_A,
            trial=trial,
            partial_scan=partial_scan,
            partial_coarse=partial_coarse,
            residual_history=residual_history,
            work=work,
            controls=controls,
            attempted_sweeps=attempted_sweeps,
            xp=xp,
        )
        raise
    except Exception as error:
        failure = Failure("MultirateCallback", str(error))
        _attach_failure_evidence(
            failure,
            old_nodes=old_nodes,
            mapped=mapped,
            residual=residual,
            scaled_residual=scaled_residual,
            normalized_residual=normalized_residual,
            residual_scale=residual_scale,
            z_coarse=z_coarse,
            coarse_A=coarse_A,
            trial=trial,
            partial_scan=partial_scan,
            partial_coarse=partial_coarse,
            residual_history=residual_history,
            work=work,
            controls=controls,
            attempted_sweeps=attempted_sweeps,
            xp=xp,
        )
        raise failure from error


__all__ = [
    "CoarseSample",
    "DenseSegment",
    "FineSample",
    "WindowIdentity",
    "WindowResult",
    "gauss_lobatto_rule",
    "solve_window",
]
