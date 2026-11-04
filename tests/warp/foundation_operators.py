from __future__ import annotations

import numpy as np
import pytest

from fomhd.backend import Backend
from fomhd.geometry import build_geometry, build_rectangle
from fomhd.operators import (
    OperatorOptions,
    apply_t1,
    apply_t1_inverse,
    apply_t2,
    apply_t2_inverse,
    apply_t2_inverse_block,
    apply_t3,
    apply_t3_inverse,
    build_gradient_operator,
    build_operators,
    validate_operators,
)


def rectangle():
    return build_rectangle(4, 3, 0.5, -0.2, 0.1, 0.1)


def ell():
    R = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    Z = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
    polygon = [
        (0.5, -0.3),
        (1.1, -0.3),
        (1.1, 0.0),
        (0.8, 0.0),
        (0.8, 0.4),
        (0.5, 0.4),
    ]
    return build_geometry(R, Z, polygon)


def deterministic(size: int, seed: int):
    i = np.arange(size)
    real = ((i * 13 + seed * 7) % 19) - 9.0
    imag = ((i * 5 + seed * 11) % 17) - 8.0
    return real / 10.0 + 1j * imag / 13.0


def test_integer_incidence_and_positive_weights() -> None:
    geometry = rectangle()
    operators = build_operators(
        geometry, 0, OperatorOptions(1e-10, 256 << 20), Backend("cpu")
    )
    assert np.max(np.abs(operators.b2r @ operators.b1r + operators.b2z @ operators.b1z)) == 0.0
    assert np.all(operators.M0 > 0.0)
    assert np.all(operators.M1 > 0.0)
    assert np.all(operators.MB > 0.0)
    assert operators.M0.dtype == np.float64
    assert operators.M1.dtype == np.float64
    assert operators.MB.dtype == np.float64


def test_compatible_complex_identities_and_divergence() -> None:
    geometry = rectangle()
    operators = build_operators(geometry, 1)
    for index, mode in enumerate((-1, 0, 1)):
        potential = operators.G[index].dtype.type(deterministic(operators.Nc, mode + 4))
        cg = operators.C[index] @ (operators.G[index] @ potential)
        assert np.max(np.abs(cg)) < 2e-12
        electric = deterministic(operators.Ne, mode + 9)
        dbc = operators.DB[index] @ (operators.C[index] @ electric)
        assert np.max(np.abs(dbc)) < 2e-12
    flux = deterministic(operators.M1.size, 3)
    divergence = operators.DJ[operators.K] @ flux
    volume = np.sum(operators.M0 * divergence)
    boundary = 0.0j
    nr, nc = len(geometry.r_faces.keys), len(geometry.cells.keys)
    for wall in geometry.walls:
        offset = 0 if wall.component == "r_face" else nr + nc
        boundary += wall.sign * wall.area_per_radian * flux[offset + wall.face]
    assert abs(volume - boundary) < 2e-13


def test_i_h_pairing() -> None:
    geometry = rectangle()
    operators = build_operators(geometry, 0)
    nodes = deterministic(3 * operators.Nc, 6)
    edges = deterministic(operators.M1.size, 12)
    node_weights = np.tile(operators.M0, 3)
    left = np.vdot(nodes, operators.H @ edges) if False else np.sum(
        np.conj(nodes) * node_weights * (operators.H @ edges)
    )
    right = np.sum(
        np.conj(operators.I @ nodes) * operators.M1 * edges
    )
    assert abs(left - right) < 2e-12


class _BackendXp:
    ndarray = np.ndarray
    complex128 = np.complex128
    float64 = np.float64

    def __init__(self) -> None:
        self.tile_calls = 0

    def __getattr__(self, name):
        return getattr(np, name)

    def tile(self, values, count):
        self.tile_calls += 1
        return np.tile(values, count)


def test_validation_uses_backend_tile_on_device_arrays() -> None:
    geometry = rectangle()
    backend = Backend("cpu")
    operators = build_operators(geometry, 0, backend=backend)
    wrapper = _BackendXp()
    backend.xp = wrapper
    checks = validate_operators(geometry, operators)
    assert len(checks) == 6
    assert wrapper.tile_calls == 1


class _TrackingBackend(Backend):
    def __init__(self) -> None:
        super().__init__("cpu")
        self.solve_calls = 0

    def solve(self, factorization, rhs):
        self.solve_calls += 1
        return super().solve(factorization, rhs)


def test_t2_inverse_block_defaults_to_operators_backend() -> None:
    geometry = rectangle()
    backend = _TrackingBackend()
    operators = build_operators(geometry, 0, backend=backend)
    backend.solve_calls = 0
    block = backend.array(np.eye(operators.Nb, dtype=np.complex128))
    transformed = apply_t2_inverse_block(
        operators,
        block,
        OperatorOptions(1e-10, 256 << 20),
    )
    assert transformed.shape == block.shape
    assert backend.solve_calls > 0


def test_noncommuting_transform_order_and_roundtrip() -> None:
    geometry = ell()
    operators = build_operators(geometry, 0)
    defect = operators.PR_v @ operators.PZ_v - operators.PZ_v @ operators.PR_v
    assert np.max(np.abs(defect)) > 1e-8
    x = deterministic(operators.MB.size, 4)
    for forward, backward in (
        (apply_t1, apply_t1_inverse),
        (apply_t2, apply_t2_inverse),
        (apply_t3, apply_t3_inverse),
    ):
        expected = (
            x[: operators.Ne]
            if forward is apply_t1
            else x[: operators.Nb]
            if forward is apply_t2
            else x[: operators.PR_v.shape[0]]
        )
        recovered = backward(operators, forward(operators, expected))
        assert np.max(np.abs(recovered - expected)) < 3e-12


def test_full_rows_and_k0_dirichlet_operator() -> None:
    geometry = rectangle()
    operators = build_operators(geometry, 1)
    assert len(operators.Lphi) == 3
    assert operators.Lphi[1].shape == (len(geometry.cells.keys),) * 2
    charge = deterministic(operators.Nc, 11)
    potential = operators.phi_factor[1].solve(charge)
    residual = operators.Lphi[1] @ potential - charge
    assert np.linalg.norm(residual) < 1e-11 * max(1.0, np.linalg.norm(charge))
    assert operators.T_full.shape[0] == len(geometry.walls)
    assert operators.T.shape[0] == len(operators.selected_wall_rows)
    assert operators.Bc_full.shape[0] == (2 * operators.K + 1) * operators.Nc
    assert operators.Bc.shape[0] == len(operators.selected_constraint_rows)


def test_gradient_inner_fourth_order() -> None:
    errors = []
    a, b = 3.0, 4.0
    r_lo, r_hi, z_lo, z_hi = 0.5, 1.0, -0.25, 0.25
    for level in range(3):
        nc = 8 << level
        hr = (r_hi - r_lo) / nc
        hz = (z_hi - z_lo) / (nc // 2)
        geometry = build_rectangle(
            nc + 1,
            nc // 2 + 1,
            r_lo + 0.5 * hr,
            z_lo + 0.5 * hz,
            hr,
            hz,
        )
        gradient = build_gradient_operator(geometry, 0)
        phi = np.sin(a * geometry.cells.R) * np.sin(b * geometry.cells.Z)
        actual = gradient @ phi
        max_error = 0.0
        nr, ncell = len(geometry.r_faces.keys), len(geometry.cells.keys)
        inner = 0.10
        for face in range(nr):
            R, Z = geometry.r_faces.R[face], geometry.r_faces.Z[face]
            if R - r_lo < inner or r_hi - R < inner or Z - z_lo < inner or z_hi - Z < inner:
                continue
            exact = a * np.cos(a * R) * np.sin(b * Z)
            max_error = max(max_error, abs(actual[face].real - exact))
        for face in range(len(geometry.z_faces.keys)):
            R, Z = geometry.z_faces.R[face], geometry.z_faces.Z[face]
            if R - r_lo < inner or r_hi - R < inner or Z - z_lo < inner or z_hi - Z < inner:
                continue
            exact = b * np.sin(a * R) * np.cos(b * Z)
            max_error = max(max_error, abs(actual[nr + ncell + face].real - exact))
        errors.append(max_error)
    assert np.log2(errors[0] / errors[1]) > 3.5
    assert np.log2(errors[1] / errors[2]) > 3.5


def test_operator_memory_and_rank_rejections() -> None:
    with pytest.raises(Exception, match="operator_matrices memory budget"):
        build_operators(rectangle(), 1, OperatorOptions(1e-10, 8))
    with pytest.raises(Exception, match="rank tolerance"):
        build_operators(rectangle(), 1, OperatorOptions(0.0, 256 << 20))
