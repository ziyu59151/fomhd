from __future__ import annotations

import importlib.util
import math
from functools import lru_cache
from pathlib import Path
import os

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "src/fomhd/time_affine.py"
REQUIRED_API = (
    "affine_prefix",
    "constant_prefix",
    "rk4_affine",
    "rk4_stages",
)


@pytest.mark.skipif(os.environ.get("FOMHD_TEST_GPU") != "1",reason="explicit owned Slurm GPU run required")
@pytest.mark.parametrize("length",[1,3,17,65])
def test_cuda_noncommuting_full_prefix_matches_independent_cpu(length):
    assert os.environ.get("SLURM_JOB_ID")
    import cupy as cp
    rng=np.random.default_rng(109)
    a=np.eye(3)[None]+.01*rng.normal(size=(length,3,3))
    a=a.astype(np.complex128)
    b=(rng.normal(size=(length,3))+1j*rng.normal(size=(length,3))).astype(np.complex128)
    start=np.array([.2+.3j,1.,-.4j])
    expected=sequential_affine(a,b,start)
    actual=load_time_affine().affine_prefix(cp.asarray(a),cp.asarray(b),cp.asarray(start),xp=cp)
    np.testing.assert_allclose(cp.asnumpy(actual),expected,rtol=1e-13,atol=1e-13)


@lru_cache(maxsize=1)
def load_time_affine():
    """Load the scratch module explicitly, without path shadowing."""
    assert MODULE_PATH.is_file(), f"feature module is absent: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location("t2_e1_time_affine", MODULE_PATH)
    assert spec is not None, f"importlib cannot create a spec for {MODULE_PATH}"
    assert spec.loader is not None, f"importlib spec has no loader for {MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [name for name in REQUIRED_API if not hasattr(module, name)]
    assert not missing, f"required E1 API is absent: {missing}"
    return module


def matvec(operator, vector):
    return np.matmul(operator, vector[..., None])[..., 0]


def sequential_affine(A, b, z0):
    """Independent affine recurrence; this must never call a prefix helper."""
    current = np.array(z0, copy=True)
    endpoints = np.empty((A.shape[0],) + z0.shape, dtype=np.complex128)
    for step in range(A.shape[0]):
        current = matvec(A[step], current) + b[step]
        endpoints[step] = current
    return endpoints


def sequential_rk4_recording(L, forcing, h, z0):
    """Independent per-step RK4 oracle with full stage history."""
    steps = forcing.shape[0]
    batch_shape = z0.shape[:-1]
    d = z0.shape[-1]
    states = np.empty((steps, 4) + batch_shape + (d,), dtype=np.complex128)
    rhs = np.empty_like(states)
    endpoints = np.empty_like(z0)[None].repeat(steps, axis=0)
    current = np.array(z0, copy=True)
    for step in range(steps):
        states[step, 0] = current
        rhs[step, 0] = matvec(L, current) + forcing[step, 0]
        states[step, 1] = current + h * rhs[step, 0] / 2.0
        rhs[step, 1] = matvec(L, states[step, 1]) + forcing[step, 1]
        states[step, 2] = current + h * rhs[step, 1] / 2.0
        rhs[step, 2] = matvec(L, states[step, 2]) + forcing[step, 2]
        states[step, 3] = current + h * rhs[step, 2]
        rhs[step, 3] = matvec(L, states[step, 3]) + forcing[step, 3]
        current = current + h * (
            rhs[step, 0]
            + 2.0 * rhs[step, 1]
            + 2.0 * rhs[step, 2]
            + rhs[step, 3]
        ) / 6.0
        endpoints[step] = current
    return states, rhs, endpoints


def independent_rk4_stages(L, forcing, h, left):
    """Evaluate each supplied microstep left state without prefix helpers."""
    steps = forcing.shape[0]
    states = np.empty((steps, 4) + left.shape[1:], dtype=np.complex128)
    rhs = np.empty_like(states)
    endpoints = np.empty_like(left)
    for step in range(steps):
        states[step, 0] = left[step]
        rhs[step, 0] = matvec(L, left[step]) + forcing[step, 0]
        states[step, 1] = left[step] + h * rhs[step, 0] / 2.0
        rhs[step, 1] = matvec(L, states[step, 1]) + forcing[step, 1]
        states[step, 2] = left[step] + h * rhs[step, 1] / 2.0
        rhs[step, 2] = matvec(L, states[step, 2]) + forcing[step, 2]
        states[step, 3] = left[step] + h * rhs[step, 2]
        rhs[step, 3] = matvec(L, states[step, 3]) + forcing[step, 3]
        endpoints[step] = left[step] + h * (
            rhs[step, 0]
            + 2.0 * rhs[step, 1]
            + 2.0 * rhs[step, 2]
            + rhs[step, 3]
        ) / 6.0
    return states, rhs, endpoints


def make_affine_case(rng, steps, batch_shape=(2,), d=2):
    A = rng.normal(size=(steps,) + batch_shape + (d, d))
    A = A + 1j * rng.normal(size=A.shape)
    b = rng.normal(size=(steps,) + batch_shape + (d,))
    b = b + 1j * rng.normal(size=b.shape)
    z0 = rng.normal(size=batch_shape + (d,))
    z0 = z0 + 1j * rng.normal(size=z0.shape)
    return A.astype(np.complex128), b.astype(np.complex128), z0.astype(np.complex128)


def make_rk4_case(rng, steps, batch_shape=(2,), d=4):
    L = rng.normal(size=batch_shape + (d, d))
    L = L + 1j * rng.normal(size=L.shape)
    forcing = rng.normal(size=(steps, 4) + batch_shape + (d,))
    forcing = forcing + 1j * rng.normal(size=forcing.shape)
    left = rng.normal(size=(steps,) + batch_shape + (d,))
    left = left + 1j * rng.normal(size=left.shape)
    return (
        L.astype(np.complex128),
        forcing.astype(np.complex128),
        left.astype(np.complex128),
    )


class RecordingXp:
    """NumPy namespace that counts matrix multiplications."""

    ndarray = np.ndarray
    complex128 = np.complex128

    def __init__(self):
        self.matmul_calls = 0

    def matmul(self, left, right):
        self.matmul_calls += 1
        return np.matmul(left, right)

    def __getattr__(self, name):
        return getattr(np, name)


def test_required_e1_api_is_present():
    module = load_time_affine()
    assert all(hasattr(module, name) for name in REQUIRED_API)


@pytest.mark.parametrize("steps", [1, 3, 5, 17, 65])
def test_affine_prefix_noncommuting_maps_and_padding(steps):
    rng = np.random.default_rng(3100 + steps)
    A, b, z0 = make_affine_case(rng, steps)
    if steps > 1:
        assert not np.array_equal(A[0] @ A[1], A[1] @ A[0])

    actual = load_time_affine().affine_prefix(A, b, z0, xp=np)
    expected = sequential_affine(A, b, z0)

    assert actual.shape == expected.shape
    assert actual.dtype == np.complex128
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("steps", [1, 3, 5, 17, 65])
def test_constant_prefix_matches_generic_tree_and_independent_oracle(steps):
    rng = np.random.default_rng(4100 + steps)
    _, b, z0 = make_affine_case(rng, steps)
    A_shared = (
        rng.normal(size=(2, 2, 2))
        + 1j * rng.normal(size=(2, 2, 2))
    ).astype(np.complex128)
    A_window = np.broadcast_to(A_shared[None], (steps,) + A_shared.shape)

    module = load_time_affine()
    actual = module.constant_prefix(A_shared, b, z0, xp=np)
    generic = module.affine_prefix(A_window, b, z0, xp=np)
    independent = sequential_affine(A_window, b, z0)

    assert actual.shape == independent.shape
    assert actual.dtype == np.complex128
    np.testing.assert_allclose(actual, generic, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual, independent, rtol=2e-13, atol=2e-13)


def test_generic_tree_batches_work_by_level_not_time_node():
    rng = np.random.default_rng(7100)
    steps = 65
    A, b, z0 = make_affine_case(rng, steps, batch_shape=(), d=2)
    xp = RecordingXp()

    load_time_affine().affine_prefix(A, b, z0, xp=xp)

    levels = math.ceil(math.log2(steps))
    assert xp.matmul_calls <= 4 * levels + 4


def test_constant_tree_batches_work_by_level_not_time_node():
    rng = np.random.default_rng(7200)
    steps = 65
    A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    _, b, z0 = make_affine_case(rng, steps, batch_shape=(), d=2)
    xp = RecordingXp()

    load_time_affine().constant_prefix(A.astype(np.complex128), b, z0, xp=xp)

    levels = math.ceil(math.log2(steps))
    assert xp.matmul_calls <= 4 * levels + 4


def test_rk4_affine_uses_all_distinct_sources():
    rng = np.random.default_rng(8100)
    L, forcing, _ = make_rk4_case(rng, steps=5)
    h = 0.17
    module = load_time_affine()

    A, b = module.rk4_affine(L, forcing, h, xp=np)

    assert A.shape == L.shape
    assert b.shape == forcing[:, 0].shape
    assert A.dtype == np.complex128
    assert b.dtype == np.complex128
    for source in range(4):
        forcing_without_source = forcing.copy()
        forcing_without_source[:, source] = 0.0
        _, omitted = module.rk4_affine(L, forcing_without_source, h, xp=np)
        assert np.max(np.abs(omitted - b)) > 1e-8


@pytest.mark.parametrize(
    "source",
    [0, 1, 2, 3],
    ids=["left", "midpoint-2", "midpoint-3", "right"],
)
def test_rk4_affine_single_source_matches_independent_oracle(source):
    rng = np.random.default_rng(8200 + source)
    L, _, z0_left = make_rk4_case(rng, steps=3, batch_shape=(1,), d=4)
    forcing = np.zeros((3, 4, 1, 4), dtype=np.complex128)
    forcing[:, source] = rng.normal(size=(3, 1, 4)) + 1j * rng.normal(
        size=(3, 1, 4)
    )
    z0 = z0_left[0]
    h = 0.11
    module = load_time_affine()

    A, b = module.rk4_affine(L, forcing, h, xp=np)
    actual = module.affine_prefix(
        np.broadcast_to(A[None], (3,) + A.shape), b, z0, xp=np
    )
    _, _, expected = sequential_rk4_recording(L, forcing, h, z0)

    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)


def test_rk4_affine_analytic_hall_two_block():
    h = 0.2
    H = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128)
    L = np.zeros((4, 4), dtype=np.complex128)
    L[:2, :2] = H
    L[2:, 2:] = 2.0 * H
    forcing = np.zeros((3, 4, 4), dtype=np.complex128)
    z0 = np.array([0.3 - 0.2j, 1.1 + 0.4j, -0.7 + 0.1j, 0.2 + 0.9j])
    module = load_time_affine()

    A, b = module.rk4_affine(L, forcing, h, xp=np)

    def stable_block(rate):
        x = h * rate
        even = 1.0 - x**2 / 2.0 + x**4 / 24.0
        odd = x - x**3 / 6.0
        return even * np.eye(2, dtype=np.complex128) + odd * H

    expected_A = np.zeros_like(L)
    expected_A[:2, :2] = stable_block(1.0)
    expected_A[2:, 2:] = stable_block(2.0)
    np.testing.assert_allclose(A, expected_A, rtol=2e-14, atol=2e-15)
    np.testing.assert_array_equal(b, np.zeros_like(b))

    actual = module.affine_prefix(
        np.broadcast_to(A[None], (3,) + A.shape), b, z0, xp=np
    )
    expected = np.stack([A @ z0, A @ A @ z0, A @ A @ A @ z0])
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_rk4_stages_match_independent_stable_four_component_oracle():
    rng = np.random.default_rng(9100)
    dense = rng.normal(size=(4, 4))
    L = (dense - dense.T).astype(np.complex128)
    forcing = rng.normal(size=(7, 4, 4))
    forcing = (forcing + 1j * rng.normal(size=forcing.shape)).astype(np.complex128)
    z0 = rng.normal(size=4) + 1j * rng.normal(size=4)
    z0 = z0.astype(np.complex128)
    h = 0.08
    module = load_time_affine()

    left = np.broadcast_to(z0[None], (7, 4)).copy()
    states, rhs, endpoints = module.rk4_stages(L, forcing, h, left, xp=np)
    expected_states, expected_rhs, expected_endpoints = independent_rk4_stages(
        L, forcing, h, left
    )

    assert states.shape == expected_states.shape
    assert rhs.shape == expected_rhs.shape
    assert endpoints.shape == expected_endpoints.shape
    np.testing.assert_allclose(states, expected_states, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(rhs, expected_rhs, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(endpoints, expected_endpoints, rtol=3e-13, atol=3e-13)


def test_rk4_affine_window_matches_independent_four_component_oracle():
    rng = np.random.default_rng(9200)
    dense = rng.normal(size=(4, 4))
    L = (dense - dense.T).astype(np.complex128)
    forcing = rng.normal(size=(9, 4, 4))
    forcing = (forcing + 1j * rng.normal(size=forcing.shape)).astype(np.complex128)
    z0 = (rng.normal(size=4) + 1j * rng.normal(size=4)).astype(np.complex128)
    h = 0.075
    module = load_time_affine()

    A, b = module.rk4_affine(L, forcing, h, xp=np)
    actual = module.affine_prefix(
        np.broadcast_to(A[None], (9,) + A.shape), b, z0, xp=np
    )
    _, _, expected = sequential_rk4_recording(L, forcing, h, z0)

    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)


def test_inputs_are_read_only_and_outputs_do_not_alias():
    rng = np.random.default_rng(10100)
    L, forcing, left = make_rk4_case(rng, steps=4, batch_shape=(2,), d=3)
    A, b, z0 = make_affine_case(rng, steps=4)
    originals = [L, forcing, left, A, b, z0]
    snapshots = [value.copy() for value in originals]
    for value in originals:
        value.setflags(write=False)

    module = load_time_affine()
    affine_out = module.affine_prefix(A, b, z0, xp=np)
    constant_out = module.constant_prefix(A[0], b, z0, xp=np)
    rk_A, rk_b = module.rk4_affine(L, forcing, 0.1, xp=np)
    stages, rhs, endpoints = module.rk4_stages(L, forcing, 0.1, left, xp=np)

    for source, snapshot in zip(originals, snapshots, strict=True):
        np.testing.assert_array_equal(source, snapshot)
    for output in (affine_out, constant_out, rk_A, rk_b, stages, rhs, endpoints):
        for source in originals:
            assert not np.shares_memory(output, source)
    before = [source.copy() for source in originals]
    affine_out[...] = -123.0
    constant_out[...] = -456.0
    rk_b[...] = -789.0
    stages[...] = -321.0
    rhs[...] = -654.0
    endpoints[...] = -987.0
    for source, snapshot in zip(originals, before, strict=True):
        np.testing.assert_array_equal(source, snapshot)


def test_affine_prefix_rejects_invalid_dtype_shape_and_dimension():
    module = load_time_affine()
    A = np.eye(2, dtype=np.complex128)[None]
    b = np.ones((1, 2), dtype=np.complex128)
    z0 = np.ones(2, dtype=np.complex128)

    with pytest.raises(TypeError, match="complex128"):
        module.affine_prefix(A.real.copy(), b, z0, xp=np)
    with pytest.raises(TypeError, match="complex128"):
        module.affine_prefix(A, b.astype(np.complex64), z0, xp=np)
    with pytest.raises(ValueError, match="shape"):
        module.affine_prefix(A[0], b, z0, xp=np)
    with pytest.raises(ValueError, match="shape"):
        module.affine_prefix(A, np.ones((1, 3), dtype=np.complex128), z0, xp=np)
    with pytest.raises(ValueError, match="shape"):
        module.affine_prefix(A, b, np.ones(3, dtype=np.complex128), xp=np)
    with pytest.raises(ValueError, match="nonempty"):
        module.affine_prefix(
            np.empty((0, 2, 2), dtype=np.complex128),
            np.empty((0, 2), dtype=np.complex128),
            z0,
            xp=np,
        )
    with pytest.raises(TypeError, match="xp.ndarray"):
        module.affine_prefix([[1.0, 0.0], [0.0, 1.0]], b, z0, xp=np)


def test_constant_and_rk4_reject_invalid_inputs():
    module = load_time_affine()
    L = np.eye(2, dtype=np.complex128)
    b = np.ones((3, 2), dtype=np.complex128)
    z0 = np.ones(2, dtype=np.complex128)
    forcing = np.ones((3, 4, 2), dtype=np.complex128)
    left = np.ones((3, 2), dtype=np.complex128)

    with pytest.raises(ValueError, match="shape"):
        module.constant_prefix(L, b[:, :1], z0, xp=np)
    with pytest.raises(TypeError, match="complex128"):
        module.constant_prefix(L.real.copy(), b, z0, xp=np)
    with pytest.raises(ValueError, match="four"):
        module.rk4_affine(L, forcing[:, :3], 0.1, xp=np)
    with pytest.raises(ValueError, match="finite"):
        module.rk4_affine(L, forcing, np.nan, xp=np)
    with pytest.raises(TypeError, match="real"):
        module.rk4_affine(L, forcing, 0.1 + 0.2j, xp=np)
    with pytest.raises(ValueError, match="shape"):
        module.rk4_stages(L, forcing, 0.1, left[:-1], xp=np)
    with pytest.raises(TypeError, match="complex128"):
        module.rk4_stages(L, forcing.astype(np.complex64), 0.1, left, xp=np)
