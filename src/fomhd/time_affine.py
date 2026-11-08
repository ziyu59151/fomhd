"""Bounded T2-E1 affine RK4 time-prefix algebra.

All state and linear-operator arrays are ``xp.complex128``.  The explicit
``xp`` is either NumPy or CuPy; this module never imports or switches backends.
Inputs are read-only and no input is copied to the host for validation.
"""

from __future__ import annotations

import math
import numbers
from typing import Any


def _require_array(value: Any, name: str, xp: Any) -> None:
    if not isinstance(value, xp.ndarray):
        raise TypeError(f"{name} must be a xp.ndarray")
    if value.dtype != xp.complex128:
        raise TypeError(f"{name} must have dtype xp.complex128")


def _require_h(h: Any) -> float:
    if not isinstance(h, numbers.Real) or isinstance(h, bool):
        raise TypeError("h must be a finite real scalar")
    value = float(h)
    if not math.isfinite(value):
        raise ValueError("h must be a finite real scalar")
    return value


def _matvec(operator: Any, vector: Any, xp: Any) -> Any:
    """Apply a matrix to the last axis with normal leading-dimension broadcast."""
    return xp.matmul(operator, vector[..., None])[..., 0]


def _affine_shapes(A: Any, b: Any, z0: Any, xp: Any) -> tuple[int, tuple[int, ...], int]:
    _require_array(A, "A", xp)
    _require_array(b, "b", xp)
    _require_array(z0, "z0", xp)
    if A.ndim < 3:
        raise ValueError("A shape must be (M, ..., d, d)")
    steps = A.shape[0]
    d = A.shape[-1]
    if steps < 1 or d < 1 or A.shape[-2] != d:
        raise ValueError("A shape must contain a nonempty M and square d-by-d blocks")
    batch_shape = A.shape[1:-2]
    if b.shape != (steps,) + batch_shape + (d,):
        raise ValueError("b shape must be (M, ..., d) matching A")
    if z0.shape != batch_shape + (d,):
        raise ValueError("z0 shape must be (..., d) matching A")
    return steps, batch_shape, d


def _constant_shapes(A: Any, b: Any, z0: Any, xp: Any) -> tuple[int, tuple[int, ...], int]:
    _require_array(A, "A", xp)
    _require_array(b, "b", xp)
    _require_array(z0, "z0", xp)
    if A.ndim < 2:
        raise ValueError("A shape must be (..., d, d)")
    d = A.shape[-1]
    if d < 1 or A.shape[-2] != d:
        raise ValueError("A shape must contain square d-by-d blocks")
    batch_shape = A.shape[:-2]
    if b.ndim < 2 or b.shape[0] < 1:
        raise ValueError("b shape must contain a nonempty M time dimension")
    if b.shape[1:] != batch_shape + (d,):
        raise ValueError("b shape must be (M, ..., d) matching A")
    if z0.shape != batch_shape + (d,):
        raise ValueError("z0 shape must be (..., d) matching A")
    return b.shape[0], batch_shape, d


def _rk4_shapes(
    L: Any, forcing: Any, xp: Any
) -> tuple[int, tuple[int, ...], int]:
    _require_array(L, "L", xp)
    _require_array(forcing, "forcing", xp)
    if L.ndim < 2:
        raise ValueError("L shape must be (..., d, d)")
    d = L.shape[-1]
    if d < 1 or L.shape[-2] != d:
        raise ValueError("L shape must contain square d-by-d blocks")
    batch_shape = L.shape[:-2]
    if forcing.ndim < 3 or forcing.shape[0] < 1:
        raise ValueError("forcing shape must contain a nonempty M dimension")
    if forcing.shape[1] != 4:
        raise ValueError("forcing must have four RK4 source slots")
    if forcing.shape[-1] != d or forcing.shape[2:-1] != batch_shape:
        raise ValueError("forcing shape must match (M, 4, ..., d) from L")
    return forcing.shape[0], batch_shape, d


def _rk4_matrix(L: Any, h: float, xp: Any) -> Any:
    """Return I + hL + (hL)^2/2 + (hL)^3/6 + (hL)^4/24."""
    scaled = h * L
    scaled2 = xp.matmul(scaled, scaled)
    scaled4 = xp.matmul(scaled2, scaled2)
    identity = xp.eye(L.shape[-1], dtype=xp.complex128)
    return (
        identity
        + scaled
        + scaled2 / 2.0
        + xp.matmul(scaled2, scaled) / 6.0
        + scaled4 / 24.0
    )


def affine_prefix(A: Any, b: Any, z0: Any, *, xp: Any) -> Any:
    """Return all endpoints of varying affine maps ``z <- A[j] z + b[j]``.

    This is a balanced inclusive Blelloch-style up/down tree.  Composition is
    ``(A_later A_earlier, A_later b_earlier + b_later)``.  Padded leaves are
    exact identity maps.  Only tree-level operations are Python loops.
    """
    steps, batch_shape, d = _affine_shapes(A, b, z0, xp)
    padded = 1 << (steps - 1).bit_length()

    leaf_A = xp.zeros((padded,) + batch_shape + (d, d), dtype=xp.complex128)
    leaf_b = xp.zeros((padded,) + batch_shape + (d,), dtype=xp.complex128)
    leaf_A[:steps] = A
    leaf_b[:steps] = b
    if padded > steps:
        leaf_A[steps:] = xp.eye(d, dtype=xp.complex128)

    aggregates_A = [leaf_A]
    aggregates_b = [leaf_b]
    while aggregates_A[-1].shape[0] > 1:
        left_A = aggregates_A[-1][0::2]
        right_A = aggregates_A[-1][1::2]
        left_b = aggregates_b[-1][0::2]
        right_b = aggregates_b[-1][1::2]
        aggregates_A.append(xp.matmul(right_A, left_A))
        aggregates_b.append(_matvec(right_A, left_b, xp) + right_b)

    exclusive_A = xp.zeros((1,) + batch_shape + (d, d), dtype=xp.complex128)
    exclusive_b = xp.zeros((1,) + batch_shape + (d,), dtype=xp.complex128)
    exclusive_A[0] = xp.eye(d, dtype=xp.complex128)

    for level in range(len(aggregates_A) - 2, -1, -1):
        left_A = aggregates_A[level][0::2]
        left_b = aggregates_b[level][0::2]
        right_A = xp.matmul(left_A, exclusive_A)
        right_b = _matvec(left_A, exclusive_b, xp) + left_b
        child_count = aggregates_A[level].shape[0]
        child_A = xp.empty(
            (child_count,) + batch_shape + (d, d), dtype=xp.complex128
        )
        child_b = xp.empty(
            (child_count,) + batch_shape + (d,), dtype=xp.complex128
        )
        child_A[0::2] = exclusive_A
        child_A[1::2] = right_A
        child_b[0::2] = exclusive_b
        child_b[1::2] = right_b
        exclusive_A = child_A
        exclusive_b = child_b

    inclusive_A = xp.matmul(leaf_A, exclusive_A)
    inclusive_b = _matvec(leaf_A, exclusive_b, xp) + leaf_b
    endpoints = _matvec(inclusive_A, z0, xp) + inclusive_b
    return endpoints[:steps]


def constant_prefix(A: Any, b: Any, z0: Any, *, xp: Any) -> Any:
    """Return all endpoints for maps sharing the same matrix ``A``.

    The up sweep stores only block aggregates of vectors and the
    ``log2(P)`` powers of ``A``.  The down sweep carries each subtree's input
    state, so it uses powers and aggregate vectors without materializing one
    matrix per time node.
    """
    steps, batch_shape, d = _constant_shapes(A, b, z0, xp)
    padded = 1 << (steps - 1).bit_length()

    leaf_b = xp.zeros((padded,) + batch_shape + (d,), dtype=xp.complex128)
    leaf_b[:steps] = b
    aggregates_b = [leaf_b]
    powers = [A]
    while aggregates_b[-1].shape[0] > 1:
        power = powers[-1]
        left_b = aggregates_b[-1][0::2]
        right_b = aggregates_b[-1][1::2]
        aggregates_b.append(_matvec(power, left_b, xp) + right_b)
        if aggregates_b[-1].shape[0] > 1:
            powers.append(xp.matmul(power, power))

    state = z0[None]
    for level in range(len(aggregates_b) - 2, -1, -1):
        boundary = (
            _matvec(powers[level], state, xp)
            + aggregates_b[level][0::2]
        )
        child_count = aggregates_b[level].shape[0]
        child_state = xp.empty(
            (child_count,) + batch_shape + (d,), dtype=xp.complex128
        )
        child_state[0::2] = state
        child_state[1::2] = boundary
        state = child_state

    endpoints = _matvec(A, state, xp) + leaf_b
    return endpoints[:steps]


def rk4_stages(
    L: Any, forcing: Any, h: Any, left: Any, *, xp: Any
) -> tuple[Any, Any, Any]:
    """Return all four stage states, their RHS, and the window endpoints.

    ``forcing`` has shape ``(M, 4, ..., d)`` and ``left`` has shape
    ``(M, ..., d)``.  Stage slots correspond to RK4 source identities 1..4.
    """
    steps, batch_shape, d = _rk4_shapes(L, forcing, xp)
    step_size = _require_h(h)
    _require_array(left, "left", xp)
    if left.shape != (steps,) + batch_shape + (d,):
        raise ValueError("left shape must be (M, ..., d) matching forcing")

    states = xp.empty((steps, 4) + batch_shape + (d,), dtype=xp.complex128)
    rhs = xp.empty((steps, 4) + batch_shape + (d,), dtype=xp.complex128)

    states[:, 0] = left
    rhs[:, 0] = _matvec(L, left, xp) + forcing[:, 0]
    states[:, 1] = left + step_size * rhs[:, 0] / 2.0
    rhs[:, 1] = _matvec(L, states[:, 1], xp) + forcing[:, 1]
    states[:, 2] = left + step_size * rhs[:, 1] / 2.0
    rhs[:, 2] = _matvec(L, states[:, 2], xp) + forcing[:, 2]
    states[:, 3] = left + step_size * rhs[:, 2]
    rhs[:, 3] = _matvec(L, states[:, 3], xp) + forcing[:, 3]
    endpoints = left + step_size * (
        rhs[:, 0] + 2.0 * rhs[:, 1] + 2.0 * rhs[:, 2] + rhs[:, 3]
    ) / 6.0
    return states, rhs, endpoints


def rk4_affine(
    L: Any, forcing: Any, h: Any, *, xp: Any
) -> tuple[Any, Any]:
    """Return the shared RK4 matrix and each zero-left-stage affine source."""
    steps, batch_shape, d = _rk4_shapes(L, forcing, xp)
    step_size = _require_h(h)
    zero_left = xp.zeros((steps,) + batch_shape + (d,), dtype=xp.complex128)
    _, _, b = rk4_stages(L, forcing, step_size, zero_left, xp=xp)
    return _rk4_matrix(L, step_size, xp), b
