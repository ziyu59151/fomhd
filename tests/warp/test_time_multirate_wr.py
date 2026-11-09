"""Behavioral tests for the selected few-node multirate WR window."""

from __future__ import annotations

import importlib.util
import math
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

import fomhd
from fomhd.types import Failure, check_record


TASK_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = TASK_ROOT / "src/fomhd/time_multirate_wr.py"


@lru_cache(maxsize=1)
def load_module():
    assert MODULE_PATH.is_file(), f"feature module is absent: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location(
        "fomhd.time_multirate_wr", MODULE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(kind, **values):
    return SimpleNamespace(kind=kind, **values)


def _constant_callbacks(a, l, g):
    calls = {"transport": 0, "field": 0, "transport_theta": []}

    def transport(theta, p, z, identity):
        calls["transport"] += 1
        calls["transport_theta"].append(theta)
        assert p.dtype == np.complex128
        assert z.dtype == np.complex128
        return np.asarray(a, dtype=np.complex128), _record(
            "transport", identity=identity, theta=theta
        )

    def field(theta, p, z, identity):
        calls["field"] += 1
        assert p.dtype == np.complex128
        assert z.dtype == np.complex128
        return (
            np.asarray(g, dtype=np.complex128),
            np.asarray(l, dtype=np.complex128),
            _record("field", identity=identity, theta=theta),
        )

    return transport, field, calls


@pytest.mark.parametrize("s", [3, 5])
def test_lobatto_nodes_and_q_integrate_every_monomial(s):
    module = load_module()
    nodes, lagrange, q = module.gauss_lobatto_rule(s)

    assert nodes.dtype == np.float64
    assert q.dtype == np.float64
    assert nodes.shape == (s,)
    assert q.shape == (s, s)
    assert nodes[0] == 0.0
    assert nodes[-1] == 1.0
    assert np.all(np.diff(nodes) > 0.0)
    np.testing.assert_allclose(
        np.sum(np.stack([p(nodes) for p in lagrange], axis=0), axis=0),
        np.ones(s),
        rtol=0.0,
        atol=1.0e-14,
    )

    for degree in range(s):
        for i, endpoint in enumerate(nodes):
            exact = endpoint ** (degree + 1) / (degree + 1)
            weights = nodes**degree
            assert math.isclose(
                float(q[i] @ weights), exact, rel_tol=2.0e-15, abs_tol=2.0e-15
            )


def test_constant_source_uses_nonaligned_nodes_and_exact_callback_counts():
    module = load_module()
    p0 = np.array([0.2 - 0.1j, -0.4 + 0.3j], dtype=np.complex128)
    z0 = np.array([0.7 + 0.2j, -0.1 + 0.5j], dtype=np.complex128)
    p_before = p0.copy()
    z_before = z0.copy()
    a = np.array([0.3 + 0.2j, -0.1 + 0.4j], dtype=np.complex128)
    l = np.array([-0.2 + 0.5j, 0.6 - 0.3j], dtype=np.complex128)
    g = np.array([0.4 - 0.1j, -0.3 + 0.2j], dtype=np.complex128)
    transport, field, calls = _constant_callbacks(a, l, g)

    result = module.solve_window(
        p0,
        z0,
        H=0.9,
        M=3,
        s=5,
        transport=transport,
        field=field,
        xp=np,
        p_scales=np.ones_like(p0.real),
        rtol=1.0e-13,
        max_iterations=3,
    )

    nodes = result.coarse_nodes
    expected_p = p0[None] + 0.9 * nodes[:, None] * (a + l)[None]
    expected_z = z0[None] + 0.9 * nodes[:, None] * g[None]
    np.testing.assert_allclose(result.P, expected_p, rtol=3.0e-14, atol=3.0e-14)
    np.testing.assert_allclose(
        result.Z_coarse, expected_z, rtol=3.0e-14, atol=3.0e-14
    )
    np.testing.assert_allclose(
        result.Z_endpoint, z0 + 0.9 * g, rtol=3.0e-14, atol=3.0e-14
    )
    np.testing.assert_allclose(result.P_endpoint, result.P[-1])
    assert result.sweeps == 1
    assert result.iterations == 1
    assert result.ion_inner_iterations == 0
    assert result.accepted_node_version == 0
    assert result.residual_history[-1]["scaled_max"] <= 1.0e-13
    assert calls == {
        "transport": 1 + 5,
        "field": 1 + 4 * 3,
        "transport_theta": [0.0, *nodes.tolist()],
    }
    assert result.work["transport_callbacks"] == 6
    assert result.work["field_callbacks"] == 13
    assert result.work["grid_L_evaluations"] == 13
    assert result.work["rk4_stages"] == 12

    np.testing.assert_array_equal(p0, p_before)
    np.testing.assert_array_equal(z0, z_before)
    result.P[0, 0] = 99.0
    result.Z_coarse[0, 0] = -99.0
    np.testing.assert_array_equal(p0, p_before)
    np.testing.assert_array_equal(z0, z_before)


def test_constant_initializer_skips_tangent_callbacks_and_converges():
    module = load_module()
    p0 = np.array([0.1 - 0.2j], dtype=np.complex128)
    z0 = np.array([0.3 + 0.4j], dtype=np.complex128)
    a = np.array([0.5 + 0.1j], dtype=np.complex128)
    l = np.array([-0.2 + 0.3j], dtype=np.complex128)
    g = np.array([-0.1 - 0.2j], dtype=np.complex128)
    transport, field, calls = _constant_callbacks(a, l, g)

    result = module.solve_window(
        p0,
        z0,
        H=0.2,
        M=2,
        s=3,
        transport=transport,
        field=field,
        xp=np,
        p_scales=np.ones(1),
        initial="constant",
        rtol=1.0e-13,
    )

    expected = p0[None] + 0.2 * result.coarse_nodes[:, None] * (a + l)[None]
    np.testing.assert_allclose(result.P, expected, rtol=3.0e-14, atol=3.0e-14)
    assert result.initialization["kind"] == "constant"
    assert result.sweeps == 2
    assert calls["transport"] == 3 * 2
    assert calls["field"] == 4 * 2 * 2


def test_componentwise_scale_uses_old_and_mapped_state_magnitudes():
    module = load_module()
    p0 = np.array([100.0, 0.01], dtype=np.complex128)
    z0 = np.array([0.2 + 0.1j, -0.3 + 0.2j], dtype=np.complex128)
    c = np.array([1.0, 1.0e-4], dtype=np.complex128)

    def transport(theta, p, z, identity):
        del theta, p, z, identity
        return np.zeros(2, dtype=np.complex128), None

    def field(theta, p, z, identity):
        del theta, p, z, identity
        return np.zeros(2, dtype=np.complex128), c, None

    result = module.solve_window(
        p0,
        z0,
        H=10.0,
        M=1,
        s=3,
        transport=transport,
        field=field,
        xp=np,
        p_scales=np.array([1.0, 1.0]),
        initial="constant",
        rtol=0.5,
        max_iterations=1,
    )

    np.testing.assert_array_equal(result.P, np.broadcast_to(p0, (3, 2)))
    assert result.residual_history[-1]["normalized_max"] <= 1.0
    assert result.residual_history[-1]["raw_linf"] > 1.0


def test_small_relaxed_update_is_not_a_convergence_test():
    module = load_module()
    p0 = np.array([0.0], dtype=np.complex128)
    z0 = np.array([0.2 + 0.1j], dtype=np.complex128)

    def transport(theta, p, z, identity):
        del theta, p, z, identity
        return np.zeros_like(p0), None

    def field(theta, p, z, identity):
        del theta, p, z, identity
        return np.zeros_like(z0), np.ones_like(p0), None

    with pytest.raises(Failure) as caught:
        module.solve_window(
            p0,
            z0,
            H=0.5,
            M=1,
            s=3,
            transport=transport,
            field=field,
            xp=np,
            p_scales=np.ones(1),
            initial="constant",
            rtol=0.1,
            max_iterations=2,
            relaxation=0.01,
        )

    failure = caught.value
    assert failure.code == "MultirateConvergence"
    assert failure.attempted_sweeps == 2
    assert all(
        record["raw_linf"] > 0.1
        and record["normalized_max"] > 1.0
        for record in failure.residual_history
    )


def test_nonaligned_dense_l_quadrature_integrates_cubic_force():
    module = load_module()
    p0 = np.array([0.2 + 0.1j, -0.3 + 0.4j], dtype=np.complex128)
    z0 = np.array([-0.2 + 0.6j, 0.1 - 0.5j], dtype=np.complex128)

    def transport(theta, p, z, identity):
        del theta, z, identity
        return np.zeros_like(p), None

    def field(theta, p, z, identity):
        del p, z
        shape = (1.0 + theta + theta**2 + theta**3)
        forcing = shape * np.array([1.0 - 0.5j, -0.2 + 0.7j])
        return (
            forcing,
            forcing,
            _record("cubic", identity=identity, theta=theta),
        )

    result = module.solve_window(
        p0,
        z0,
        H=0.8,
        M=3,
        s=5,
        transport=transport,
        field=field,
        xp=np,
        p_scales=np.ones(2),
        rtol=1.0e-13,
        max_iterations=4,
    )

    integral = 1.0 + 0.5 + 1.0 / 3.0 + 0.25
    expected = p0 + 0.8 * integral * np.array(
        [1.0 - 0.5j, -0.2 + 0.7j], dtype=np.complex128
    )
    np.testing.assert_allclose(result.P_endpoint, expected, rtol=2.0e-14, atol=2.0e-14)
    expected_z = z0[None] + (result.P - p0[None])
    np.testing.assert_allclose(result.Z_coarse, expected_z, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(
        result.C_integrals[1:], result.P[1:] - p0[None], rtol=2.0e-14, atol=2.0e-14
    )
    assert result.fine_C.shape == (3, 4, 2)
    assert result.fine_C.dtype == np.complex128
    assert result.fine_samples[0].C.dtype == np.complex128
    assert result.fine_weights.shape == (3, 4)
    np.testing.assert_allclose(
        np.sum(result.fine_weights, axis=None), 0.8, rtol=0.0, atol=1.0e-15
    )
    assert result.sweeps == 2
    assert any(
        segment.micro_step in (0, 1, 2) and not math.isclose(segment.x, 0.0)
        and not math.isclose(segment.x, 1.0)
        for segment in result.dense_segments
    )


def _nonlinear_rhs(p, z):
    return -0.7 * p + 0.2 * z + 0.08 * p * p, -20.0 * z + 0.15 * p


def _independent_rk4(p0, z0, H, steps):
    p = complex(p0)
    z = complex(z0)
    h = H / steps
    for _ in range(steps):
        k1p, k1z = _nonlinear_rhs(p, z)
        k2p, k2z = _nonlinear_rhs(p + 0.5 * h * k1p, z + 0.5 * h * k1z)
        k3p, k3z = _nonlinear_rhs(p + 0.5 * h * k2p, z + 0.5 * h * k2z)
        k4p, k4z = _nonlinear_rhs(p + h * k3p, z + h * k3z)
        p += h * (k1p + 2.0 * k2p + 2.0 * k3p + k4p) / 6.0
        z += h * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0
    return p


def _nonlinear_callbacks():
    calls = {"transport": 0, "field": 0}

    def transport(theta, p, z, identity):
        del theta, z, identity
        calls["transport"] += 1
        return -0.7 * p, None

    def field(theta, p, z, identity):
        del theta
        calls["field"] += 1
        return (
            -20.0 * z + 0.15 * p,
            0.2 * z,
            _record("nonlinear", identity=identity),
        )

    return transport, field, calls


def test_nonlinear_mixed_target_refines_toward_small_step_rk4():
    module = load_module()
    p0 = np.array([0.35 + 0.08j], dtype=np.complex128)
    z0 = np.array([-0.1 + 0.04j], dtype=np.complex128)
    exact = _independent_rk4(p0[0], z0[0], 1.2, 32768)
    errors = []
    for micro_steps in (8, 32):
        transport, field, calls = _nonlinear_callbacks()
        result = module.solve_window(
            p0,
            z0,
            H=1.2,
            M=micro_steps,
            s=5,
            transport=transport,
            field=field,
            xp=np,
            p_scales=np.ones(1),
            rtol=2.0e-12,
            max_iterations=24,
            relaxation=0.8,
        )
        errors.append(float(abs(result.P_endpoint[0] - exact)))
        assert calls["field"] == 1 + 4 * micro_steps * result.sweeps
        assert calls["transport"] == 1 + 5 * result.sweeps

    assert errors[1] < errors[0]
    assert errors[1] < 5.0e-3


def test_linear_hall_window_matches_matrix_exponential_reference():
    module = load_module()
    full = np.array(
        [
            [0.7j, 0.0, 0.2, 0.0],
            [0.0, 0.8j, 0.0, 0.22],
            [-0.3, 0.0, 0.5j, 0.0],
            [0.0, -0.18, 0.0, 0.6j],
        ],
        dtype=np.complex128,
    )
    p0 = np.array([0.3 + 0.2j, -0.2 - 0.1j], dtype=np.complex128)
    z0 = np.array([0.6 - 0.1j, 0.15 + 0.25j], dtype=np.complex128)

    def transport(theta, p, z, identity):
        del theta, z, identity
        return full[:2, :2] @ p, None

    def field(theta, p, z, identity):
        del theta
        joined = np.concatenate((p, z))
        return (
            full[2:, :] @ joined,
            full[:2, 2:] @ z,
            _record("hall", identity=identity),
        )

    H = 1.0e-4
    result = module.solve_window(
        p0,
        z0,
        H=H,
        M=1,
        s=3,
        transport=transport,
        field=field,
        xp=np,
        p_scales=np.ones(2),
        rtol=1.0e-13,
        max_iterations=12,
    )
    exact = (expm(full * H) @ np.concatenate((p0, z0)))[:2]
    np.testing.assert_allclose(result.P_endpoint, exact, rtol=1.0e-12, atol=1.0e-12)


def test_s3_and_s5_produce_distinct_selected_targets():
    module = load_module()
    p0 = np.array([0.22 + 0.03j], dtype=np.complex128)
    z0 = np.array([0.12 - 0.02j], dtype=np.complex128)

    def transport(theta, p, z, identity):
        del theta, z, identity
        return p * p, None

    def field(theta, p, z, identity):
        del theta, p, z, identity
        return np.zeros_like(p0), np.zeros_like(p0), None

    endpoints = {}
    for s in (3, 5):
        result = module.solve_window(
            p0,
            z0,
            H=0.5,
            M=2,
            s=s,
            transport=transport,
            field=field,
            xp=np,
            p_scales=np.ones(1),
            rtol=1.0e-13,
            max_iterations=20,
            relaxation=0.8,
        )
        endpoints[s] = result.P_endpoint[0]

    assert abs(endpoints[3] - endpoints[5]) > 1.0e-7


def test_final_samples_keep_distinct_middle_z_stages_and_identities():
    module = load_module()
    p0 = np.array([0.2 - 0.1j], dtype=np.complex128)
    z0 = np.array([0.4 + 0.3j], dtype=np.complex128)
    seen = []

    def transport(theta, p, z, identity):
        del theta, z, identity
        return np.zeros_like(p), None

    def field(theta, p, z, identity):
        del theta, p
        seen.append((identity, z.copy()))
        return 1.7 * z, np.zeros_like(z), None

    result = module.solve_window(
        p0,
        z0,
        H=0.3,
        M=3,
        s=5,
        transport=transport,
        field=field,
        xp=np,
        p_scales=np.ones(1),
        rtol=1.0e-14,
        max_iterations=2,
    )

    final = [
        sample
        for sample in result.fine_samples
        if sample.identity.iteration == result.accepted_node_version
    ]
    assert len(final) == 12
    second = [sample for sample in final if sample.identity.rk_stage == 1]
    third = [sample for sample in final if sample.identity.rk_stage == 2]
    assert len(second) == len(third) == 3
    for left, middle in zip(second, third):
        assert left.identity.role == middle.identity.role == "field_rk4"
        assert left.identity.micro_step == middle.identity.micro_step
        assert left.identity.rk_stage != middle.identity.rk_stage
        assert not np.array_equal(left.Z, middle.Z)
    final_middles = [
        item
        for item in seen
        if item[0].iteration == result.accepted_node_version
        and item[0].rk_stage in (1, 2)
    ]
    assert len(final_middles) == 6
    assert all(item[1].shape == z0.shape for item in final_middles)


def test_converged_result_accepts_old_nodes_not_latest_mapped_nodes():
    module = load_module()
    p0 = np.array([0.25 + 0.02j], dtype=np.complex128)
    z0 = np.array([0.1 - 0.03j], dtype=np.complex128)

    def transport(theta, p, z, identity):
        del theta, z, identity
        return 0.4 * p * p, None

    def field(theta, p, z, identity):
        del theta, p, z, identity
        return np.zeros_like(p0), np.zeros_like(p0), None

    result = module.solve_window(
        p0,
        z0,
        H=0.4,
        M=2,
        s=5,
        transport=transport,
        field=field,
        xp=np,
        p_scales=np.ones(1),
        rtol=1.0e-2,
        max_iterations=20,
        relaxation=0.5,
    )

    assert result.P_endpoint[0] == result.P[-1, 0]
    assert abs(result.P_endpoint[0] - result.mapped_nodes[-1, 0]) > 1.0e-8
    assert np.max(np.abs(result.residual)) > 0.0
    assert result.final_input_version == result.accepted_node_version


def test_nonconvergence_raises_failure_with_current_evidence_and_no_rollback():
    module = load_module()
    p0 = np.array([0.5 + 0.2j], dtype=np.complex128)
    z0 = np.array([-0.1 + 0.3j], dtype=np.complex128)
    p_before = p0.copy()
    z_before = z0.copy()

    def transport(theta, p, z, identity):
        del theta, z, identity
        return -4.0 * p * p, None

    def field(theta, p, z, identity):
        del theta, p
        return z * z, z, None

    with pytest.raises(Failure) as caught:
        module.solve_window(
            p0,
            z0,
            H=0.8,
            M=1,
            s=3,
            transport=transport,
            field=field,
            xp=np,
            p_scales=np.ones(1),
            rtol=1.0e-14,
            max_iterations=1,
        )

    failure = caught.value
    assert failure.code == "MultirateConvergence"
    assert failure.attempted_sweeps == 1
    assert len(failure.residual_history) == 1
    assert failure.residual_history[-1]["scaled_max"] > 1.0e-14
    assert failure.partial_arrays["P"].shape == (3, 1)
    assert failure.partial_arrays["Z_coarse"].shape == (3, 1)
    assert failure.partial_arrays["fine_Z"].shape == (1, 4, 1)
    assert failure.work["field_callbacks"] == 1 + 4
    np.testing.assert_array_equal(p0, p_before)
    np.testing.assert_array_equal(z0, z_before)


def test_mid_scan_physics_failure_retains_only_completed_stage_evidence():
    module = load_module()
    p0 = np.array([0.4 + 0.2j], dtype=np.complex128)
    z0 = np.array([-0.1 + 0.3j], dtype=np.complex128)
    field_calls = 0

    def transport(theta, p, z, identity):
        del theta, p, z, identity
        return np.zeros(1, dtype=np.complex128), None

    def field(theta, p, z, identity):
        nonlocal field_calls
        field_calls += 1
        if identity.role == "field_rk4" and identity.rk_stage == 2:
            failure = Failure("FieldPhysics", "injected mid-scan physics guard")
            failure.stage = identity
            failure.checks = [
                check_record(
                    name="injected_guard",
                    tag=identity,
                    raw=2.0,
                    reference=1.0,
                    threshold=1.0,
                    passed=False,
                )
            ]
            failure.partial_arrays["adapter_evidence"] = np.array(
                [7.0 + 1.0j], dtype=np.complex128
            )
            raise failure
        return (
            np.zeros(1, dtype=np.complex128),
            np.zeros(1, dtype=np.complex128),
            SimpleNamespace(call=field_calls),
        )

    with pytest.raises(Failure) as caught:
        module.solve_window(
            p0,
            z0,
            H=0.2,
            M=2,
            s=3,
            transport=transport,
            field=field,
            xp=np,
            p_scales=np.ones(1),
            initial="tangent",
            max_iterations=2,
        )

    failure = caught.value
    assert failure.code == "FieldPhysics"
    assert failure.attempted_sweeps == 1
    assert failure.completed_residual_sweeps == 0
    assert failure.residual_history == ()
    assert failure.attempted_identity == failure.stage
    assert failure.attempted_identity.micro_step == 0
    assert failure.attempted_identity.rk_stage == 2
    assert failure.partial_arrays["fine_P"].shape == (2, 1)
    assert failure.partial_arrays["fine_Z"].shape == (2, 1)
    assert failure.partial_arrays["fine_G"].shape == (2, 1)
    assert failure.partial_arrays["fine_C"].shape == (2, 1)
    assert tuple(item.rk_stage for item in failure.partial_identities) == (0, 1)
    assert tuple(item.call for item in failure.partial_records) == (2, 3)
    assert failure.partial_arrays["partial_P"].shape == (1,)
    assert failure.partial_arrays["partial_Z"].shape == (1,)
    assert failure.work["field_callbacks"] == 4
    assert failure.work["rk4_stages"] == 3
    assert failure.work["attempted_sweeps"] == 1
    assert failure.partial_arrays["adapter_evidence"][0] == 7.0 + 1.0j
    assert failure.checks[0].name == "injected_guard"


def test_partial_scan_copies_reused_g_and_c_buffers_without_copying_records():
    module = load_module()
    p0 = np.array([0.4 + 0.2j], dtype=np.complex128)
    z0 = np.array([-0.1 + 0.3j], dtype=np.complex128)
    g_buffer = np.empty(1, dtype=np.complex128)
    c_buffer = np.empty(1, dtype=np.complex128)
    records = []

    def transport(theta, p, z, identity):
        del theta, p, z, identity
        return np.zeros(1, dtype=np.complex128), None

    def field(theta, p, z, identity):
        del theta, p, z
        if identity.role == "tangent_initialization":
            g_buffer[0] = 100.0
            c_buffer[0] = -100.0
        elif identity.micro_step == 1 and identity.rk_stage == 0:
            g_buffer[0] = 999.0
            c_buffer[0] = -999.0
            failure = Failure(
                "ScratchReuse",
                "caller reuses G/C scratch buffers then fails",
            )
            failure.stage = identity
            raise failure
        else:
            stage = 10 * int(identity.micro_step) + int(identity.rk_stage)
            g_buffer[0] = stage + 1.0
            c_buffer[0] = -10.0 * (stage + 1.0)
        record = SimpleNamespace(sequence=len(records), identity=identity)
        records.append(record)
        return g_buffer, c_buffer, record

    with pytest.raises(Failure) as caught:
        module.solve_window(
            p0,
            z0,
            H=0.2,
            M=2,
            s=3,
            transport=transport,
            field=field,
            xp=np,
            p_scales=np.ones(1),
            max_iterations=2,
        )

    failure = caught.value
    assert failure.code == "ScratchReuse"
    assert failure.attempted_identity.micro_step == 1
    assert failure.attempted_identity.rk_stage == 0
    assert failure.completed_stage_count == 4
    np.testing.assert_array_equal(
        failure.partial_arrays["fine_G"][:, 0],
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex128),
    )
    np.testing.assert_array_equal(
        failure.partial_arrays["fine_C"][:, 0],
        np.array([-10.0, -20.0, -30.0, -40.0], dtype=np.complex128),
    )
    assert tuple(item.sequence for item in failure.partial_records) == (
        1,
        2,
        3,
        4,
    )
    assert all(
        actual is expected
        for actual, expected in zip(
            failure.partial_records,
            records[1:5],
            strict=True,
        )
    )


def test_coarse_node_failure_retains_completed_and_attempted_transport_evidence():
    module = load_module()
    p0 = np.array([0.3 - 0.1j], dtype=np.complex128)
    z0 = np.array([0.2 + 0.4j], dtype=np.complex128)
    a_buffer = np.empty(1, dtype=np.complex128)
    transport_records = []
    coarse_attempts = []

    def transport(theta, p, z, identity):
        del theta, p, z
        record = SimpleNamespace(sequence=len(transport_records), identity=identity)
        transport_records.append(record)
        if identity.role == "tangent_initialization":
            a_buffer[0] = 0.0
            return a_buffer, record
        coarse_attempts.append(identity.coarse_node)
        if identity.coarse_node == 2:
            a_buffer[0] = 999.0
            failure = Failure("CoarsePhysics", "injected coarse-node failure")
            failure.stage = identity
            failure.checks = [
                check_record(
                    name="coarse_guard",
                    tag=identity,
                    raw=3.0,
                    reference=1.0,
                    threshold=1.0,
                    passed=False,
                )
            ]
            failure.partial_arrays["adapter_coarse"] = np.array(
                [11.0 + 2.0j], dtype=np.complex128
            )
            raise failure
        a_buffer[0] = float(identity.coarse_node + 1)
        return a_buffer, record

    def field(theta, p, z, identity):
        del theta, p, z, identity
        return (
            np.zeros(1, dtype=np.complex128),
            np.zeros(1, dtype=np.complex128),
            None,
        )

    with pytest.raises(Failure) as caught:
        module.solve_window(
            p0,
            z0,
            H=0.2,
            M=2,
            s=5,
            transport=transport,
            field=field,
            xp=np,
            p_scales=np.ones(1),
            max_iterations=2,
        )

    failure = caught.value
    assert failure.code == "CoarsePhysics"
    assert coarse_attempts == [0, 1, 2]
    assert failure.attempted_sweeps == 1
    assert failure.completed_residual_sweeps == 0
    assert failure.attempted_coarse_identity.coarse_node == 2
    assert failure.attempted_coarse_identity.role == "transport_coarse"
    np.testing.assert_array_equal(
        failure.partial_arrays["completed_coarse_A"][:, 0],
        np.array([1.0, 2.0], dtype=np.complex128),
    )
    assert tuple(
        item.coarse_node for item in failure.coarse_partial_identities
    ) == (0, 1)
    assert all(
        actual is expected
        for actual, expected in zip(
            failure.coarse_partial_records,
            transport_records[1:3],
            strict=True,
        )
    )
    np.testing.assert_array_equal(
        failure.attempted_coarse_P,
        np.broadcast_to(p0, (1,)),
    )
    np.testing.assert_array_equal(
        failure.attempted_coarse_Z,
        np.broadcast_to(z0, (1,)),
    )
    assert failure.work["transport_callbacks"] == 4
    assert failure.work["transport_coarse_evaluations"] == 3
    assert failure.partial_arrays["adapter_coarse"][0] == 11.0 + 2.0j
    assert failure.checks[0].name == "coarse_guard"


@pytest.mark.parametrize(
    "override",
    [
        {"H": 0.0},
        {"H": math.inf},
        {"M": 0},
        {"M": True},
        {"M": 2.0},
        {"s": 4},
        {"rtol": 0.0},
        {"rtol": math.nan},
        {"max_iterations": 0},
        {"max_iterations": True},
        {"relaxation": 0.0},
        {"relaxation": 1.1},
        {"initial": "linear"},
    ],
)
def test_invalid_controls_are_rejected_before_callbacks(override):
    module = load_module()
    p0 = np.ones(1, dtype=np.complex128)
    z0 = np.ones(1, dtype=np.complex128)
    transport, field, calls = _constant_callbacks(
        np.ones(1), np.ones(1), np.ones(1)
    )
    controls = {
        "H": 0.1,
        "M": 2,
        "s": 3,
        "transport": transport,
        "field": field,
        "xp": np,
        "p_scales": np.ones(1),
        "rtol": 1.0e-10,
        "max_iterations": 2,
        "relaxation": 1.0,
        "initial": "tangent",
    }
    controls.update(override)

    with pytest.raises(Failure):
        module.solve_window(p0, z0, **controls)

    assert calls["transport"] == 0
    assert calls["field"] == 0


def test_field_callbacks_scale_with_microsteps_while_transport_does_not():
    module = load_module()
    p0 = np.array([0.1 + 0.2j], dtype=np.complex128)
    z0 = np.array([0.3 - 0.1j], dtype=np.complex128)
    work = {}

    for micro_steps in (2, 4):
        transport, field, calls = _constant_callbacks(
            np.ones(1), 0.5 * np.ones(1), -0.2 * np.ones(1)
        )
        result = module.solve_window(
            p0,
            z0,
            H=0.25,
            M=micro_steps,
            s=3,
            transport=transport,
            field=field,
            xp=np,
            p_scales=np.ones(1),
        )
        work[micro_steps] = (result, calls)

    assert work[2][0].sweeps == work[4][0].sweeps == 1
    assert work[2][1]["transport"] == work[4][1]["transport"] == 4
    assert work[2][1]["field"] == 1 + 8
    assert work[4][1]["field"] == 1 + 16
