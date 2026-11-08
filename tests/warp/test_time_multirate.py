"""Independent and integration tests for bounded C1/C2 reduced-hydro splits."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm


SCRATCH_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(
    os.environ.get("FOMHD_SOURCE_ROOT", "/home/lxzeng/projects/fomhd")
)
SOURCE_ROOT = PROJECT_ROOT / "src"
TEST_ROOT = PROJECT_ROOT / "tests/warp"
TOOLS_ROOT = PROJECT_ROOT / "tools"
SCRATCH_SOURCE_ROOT = SCRATCH_ROOT / "src"
MODULE_PATH = SCRATCH_SOURCE_ROOT / "fomhd/time_multirate.py"
TOOL_PATH = SCRATCH_ROOT / "tools/warp_multirate_ion.py"
for path in (
    SCRATCH_SOURCE_ROOT,
    SOURCE_ROOT,
    TEST_ROOT,
    TOOLS_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _module():
    spec = importlib.util.spec_from_file_location(
        "time_multirate_under_test", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None, "missing scratch module"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tool_module():
    spec = importlib.util.spec_from_file_location(
        "warp_multirate_ion_under_test", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None, "missing scratch tool"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pack_state(context, state):
    xp = context.compute.xp
    nc = context.op.Nc
    q = 2 * context.p.K + 1
    p = xp.concatenate(
        (
            state.fluid[0].n.reshape(q, nc),
            state.fluid[0].mom.reshape(q, 3 * nc),
            state.fluid[0].perp.reshape(q, nc),
        ),
        axis=1,
    )
    z = xp.concatenate(
        (
            state.fluid[1].perp.reshape(q, nc),
            state.fluid[1].parallel.reshape(q, nc),
            state.B.reshape(q, context.op.Nb),
        ),
        axis=1,
    )
    return p, z


def _called_t0_map(a, beta, H):
    """Independent augmented-matrix oracle for the scalar T0 cold-Hall map."""
    augmented = np.zeros((4, 4), dtype=np.complex128)
    augmented[0, 1] = 1j * a
    augmented[1, 2] = 1j * a
    augmented[1, 1] = 1j * beta
    augmented[2, 3] = 1.0
    evolution = expm(augmented * H)
    return np.array(
        (
            (
                evolution[0, 0]
                + evolution[0, 2],
                evolution[0, 1] + 1j * a * evolution[0, 3],
            ),
            (
                evolution[1, 2],
                evolution[1, 1] + 1j * a * evolution[1, 3],
            ),
        )
    )


def _cold_hall_callbacks(module, a, beta, *, frozen):
    scale = 1.0
    if frozen:
        def macro(p):
            return np.zeros_like(p), np.zeros_like(p)

        def density_rate(p):
            return np.zeros_like(p)

        def lorentz(p, electric, magnetic):
            del electric
            return 1j * a * magnetic

        def field(p, z):
            rhs = 1j * a * p + 1j * beta * z
            return rhs, np.zeros_like(p), z

        return macro, density_rate, lorentz, field

    def macro(p):
        return (
            np.zeros_like(p),
            np.zeros_like(p),
            np.zeros_like(p),
        )

    def field(p, z):
        return 1j * a * p + 1j * beta * z, np.zeros_like(p), z

    def impulse(p, integral_electric, integral_magnetic):
        del p, integral_electric
        return 1j * a * integral_magnetic

    return macro, field, impulse


def test_t0_cold_hall_map_matches_independent_augmented_exponential():
    module = _module()
    a, beta, H = 0.37, -0.11, 0.23
    actual = np.asarray(module.cold_hall_t0_map(a, beta, H))
    expected = _called_t0_map(a, beta, H)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_c1_zero_beta_centered_impulse_is_unitary():
    module = _module()
    p0 = np.array([[0.7 + 0.2j]])
    z0 = np.array([[-0.3 + 0.4j]])
    macro, field, impulse = _cold_hall_callbacks(
        module, 1.3, 0.0, frozen=False
    )
    result = module.centered_impulse_callbacks(
        macro,
        field,
        impulse,
        p0,
        z0,
        H=0.4,
        M=3,
        scales=np.ones_like(p0),
        rtol=2e-14,
        max_iterations=20,
        keep_stages=True,
    )
    before = float(np.sum(np.abs(p0) ** 2 + np.abs(z0) ** 2))
    after = float(
        np.sum(np.abs(result.p) ** 2 + np.abs(result.z) ** 2)
    )
    assert after == pytest.approx(before, rel=0.0, abs=3e-13)


def test_t0_gain_persists_under_h_refinement_while_c1_norm_defect_shrinks():
    module = _module()
    a, beta, H = 0.1, 0.9, 0.5
    t0 = np.asarray(module.cold_hall_t0_map(a, beta, H))
    eigenvalues, eigenvectors = np.linalg.eig(t0)
    branch = int(np.argmax(np.abs(eigenvalues)))
    seed = eigenvectors[:, branch]
    seed /= np.linalg.norm(seed)
    steps = 16
    t0_gain = float(np.linalg.norm(np.linalg.matrix_power(t0, steps) @ seed))
    assert t0_gain > 1.0 + 1.0e-4

    p0 = np.array([[1.0 + 0.0j]])
    z0 = np.array([[0.0 + 0.0j]])
    macro, field, impulse = _cold_hall_callbacks(
        module, a, beta, frozen=False
    )
    defects = []
    for M in (1, 2, 4, 8):
        result = module.centered_impulse_callbacks(
            macro,
            field,
            impulse,
            p0,
            z0,
            H=H,
            M=M,
            scales=np.ones_like(p0),
            rtol=2e-14,
            max_iterations=50,
        )
        defects.append(
            abs(
                float(
                    np.sum(np.abs(result.p) ** 2 + np.abs(result.z) ** 2)
                    - 1.0
                )
            )
        )
    assert all(
        later < earlier
        for earlier, later in zip(defects[:-1], defects[1:])
    )
    assert defects[-1] < 1.0e-10

    result_h = module.centered_impulse_callbacks(
        macro,
        field,
        impulse,
        p0,
        z0,
        H=H,
        M=8,
        scales=np.ones_like(p0),
        rtol=2e-14,
        max_iterations=50,
    )
    exact = expm(
        np.array([[0.0, 1j * a], [1j * a, 1j * beta]])
        * H
    ) @ np.array([1.0 + 0.0j, 0.0 + 0.0j])
    phase_error = abs(
        np.angle(result_h.p[0, 0] / exact[0])
    )
    assert phase_error > 1.0e-8


def test_c2_with_zero_frozen_hydro_is_exact_coupled_rk4():
    module = _module()
    a, beta, H, M = 0.7, -0.2, 0.3, 4
    macro, density_rate, lorentz, field = _cold_hall_callbacks(
        module, a, beta, frozen=True
    )
    p0 = np.array([[0.4 + 0.1j]])
    z0 = np.array([[-0.2 + 0.3j]])
    result = module.frozen_midpoint_callbacks(
        macro,
        density_rate,
        lorentz,
        field,
        p0,
        z0,
        H=H,
        M=M,
        scales=np.ones_like(p0),
        rtol=2e-14,
        max_iterations=20,
        keep_stages=True,
    )

    h = H / M
    coupled = np.array(
        [[0.0, 1j * a], [1j * a, 1j * beta]],
        dtype=np.complex128,
    )
    state = np.array([p0[0, 0], z0[0, 0]], dtype=np.complex128)
    for _ in range(M):
        k1 = coupled @ state
        k2 = coupled @ (state + h * k1 / 2.0)
        k3 = coupled @ (state + h * k2 / 2.0)
        k4 = coupled @ (state + h * k3)
        state = state + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    np.testing.assert_allclose(
        [result.p[0, 0], result.z[0, 0]],
        state,
        rtol=3e-14,
        atol=3e-14,
    )
    assert len(result.stages) == 4 * M * len(result.iterations)
    assert result.counters["cheap_D"] == 4 * M * len(result.iterations)
    assert result.counters["cheap_L"] == 4 * M * len(result.iterations)
    assert result.counters["field_rhs"] == 4 * M * len(result.iterations)
    assert result.stage_history_bytes > 0
    assert all(stage.p_rhs_available for stage in result.stages)
    assert all(stage.p_rhs is not None for stage in result.stages)


def test_callback_failure_retains_history_and_does_not_mutate_inputs():
    module = _module()
    p0 = np.array([[1.0 + 0.0j]])
    z0 = np.array([[1.0 + 0.0j]])
    p_before = p0.copy()
    z_before = z0.copy()
    accepted = SimpleNamespace(
        state={"sentinel": (1, 2, 3)},
        accepted=True,
        cache={"version": 7},
    )
    accepted_before = copy.deepcopy(accepted)

    def macro(p):
        return (
            np.zeros_like(p),
            np.zeros_like(p),
            np.zeros_like(p),
        )

    def field(p, z):
        return -0.5j * p + 0.2j * z, np.zeros_like(p), z

    def impulse(p, integral_electric, integral_magnetic):
        del p, integral_electric
        return -0.5j * integral_magnetic

    with pytest.raises(Exception) as caught:
        module.centered_impulse_callbacks(
            macro,
            field,
            impulse,
            p0,
            z0,
            H=0.2,
            M=1,
            scales=np.ones_like(p0),
            rtol=1.0e-15,
            max_iterations=1,
        )
    failure = caught.value
    np.testing.assert_array_equal(p0, p_before)
    np.testing.assert_array_equal(z0, z_before)
    assert accepted == accepted_before
    assert len(failure.research_history) == 1
    assert failure.research_counters["actual_outer_calls"] == 1
    assert failure.last_attempt_p is not None
    assert failure.last_attempt_z is not None


def test_c1_failed_callbacks_are_charged_before_dispatch():
    module = _module()
    p0 = np.array([[1.0 + 0.0j]])
    z0 = np.array([[1.0 + 0.0j]])

    def macro(p):
        return (
            np.zeros_like(p),
            np.zeros_like(p),
            np.zeros_like(p),
        )

    def failed_field(p, z):
        del p, z
        raise RuntimeError("injected field failure")

    with pytest.raises(Exception) as caught:
        module.centered_impulse_callbacks(
            macro,
            failed_field,
            lambda *args: np.zeros_like(p0),
            p0,
            z0,
            H=0.2,
            M=2,
            scales=np.ones_like(p0),
            max_iterations=1,
        )
    counters = caught.value.research_counters
    assert counters["actual_outer_calls"] == 1
    assert counters["full_hydro_eval"] == 1
    assert counters["field_rhs"] == 1

    def field(p, z):
        return np.zeros_like(z), np.zeros_like(p), z

    def failed_impulse(p, electric, magnetic):
        del p, electric, magnetic
        raise RuntimeError("injected impulse failure")

    with pytest.raises(Exception) as caught:
        module.centered_impulse_callbacks(
            macro,
            field,
            failed_impulse,
            p0,
            z0,
            H=0.2,
            M=2,
            scales=np.ones_like(p0),
            max_iterations=1,
        )
    counters = caught.value.research_counters
    assert counters["full_hydro_eval"] == 1
    assert counters["field_rhs"] == 8
    assert counters["quadrature_contributions"] == 8
    assert counters["cheap_L"] == 1


def test_field_failure_retains_completed_stages_and_the_attempted_stage_input():
    module = _module()
    p0 = np.array([[1.0 + 0.0j]])
    z0 = np.array([[1.0 + 0.0j]])
    calls = 0

    def macro(p):
        return (
            np.zeros_like(p),
            np.zeros_like(p),
            np.zeros_like(p),
        )

    def field(p, z):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("fail second stage")
        return 0.25j * z, np.zeros_like(p), z

    with pytest.raises(Exception) as caught:
        module.centered_impulse_callbacks(
            macro,
            field,
            lambda *args: np.zeros_like(p0),
            p0,
            z0,
            H=0.4,
            M=2,
            scales=np.ones_like(p0),
            max_iterations=1,
        )
    failure = caught.value
    assert len(failure.research_stages) == 1
    assert failure.research_counters["field_rhs"] == 2
    np.testing.assert_allclose(
        failure.last_attempt_z,
        np.array([[1.0 + 0.025j]]),
    )
    np.testing.assert_array_equal(failure.last_attempt_p, p0)


def test_impulse_failure_records_the_advanced_z_input():
    module = _module()
    p0 = np.array([[1.0 + 0.0j]])
    z0 = np.array([[0.0 + 0.0j]])

    def macro(p):
        return (
            np.zeros_like(p),
            np.zeros_like(p),
            np.zeros_like(p),
        )

    def field(p, z):
        return 1j * (1.0 + z), np.zeros_like(p), z

    def impulse(p, electric, magnetic):
        del p, electric, magnetic
        raise RuntimeError("fail impulse")

    with pytest.raises(Exception) as caught:
        module.centered_impulse_callbacks(
            macro,
            field,
            impulse,
            p0,
            z0,
            H=0.4,
            M=1,
            scales=np.ones_like(p0),
            max_iterations=1,
        )
    h = 0.4
    k1 = 1j * (1.0 + z0)
    k2 = 1j * (1.0 + z0 + h * k1 / 2.0)
    k3 = 1j * (1.0 + z0 + h * k2 / 2.0)
    k4 = 1j * (1.0 + z0 + h * k3)
    expected = z0 + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    np.testing.assert_allclose(caught.value.last_attempt_z, expected)


def test_residual_increment_is_elementwise_not_difference_of_maxima():
    module = _module()
    guess = np.array([[1.0 + 0.0j, -1.0 + 0.0j]])
    target = np.array([[-1.0 + 0.0j, 1.0 + 0.0j]])
    _, _, _, _, increment = module._residual_metrics(
        guess,
        target,
        np.ones_like(guess),
        np,
    )
    assert increment == pytest.approx(2.0)


def test_c2_failed_cheap_callbacks_are_charged_before_dispatch():
    module = _module()
    p0 = np.array([[1.0 + 0.0j]])
    z0 = np.array([[1.0 + 0.0j]])

    def macro(p):
        return np.zeros_like(p), np.zeros_like(p)

    def failed_density(p):
        del p
        raise RuntimeError("injected density failure")

    with pytest.raises(Exception) as caught:
        module.frozen_midpoint_callbacks(
            macro,
            failed_density,
            lambda *args: np.zeros_like(p0),
            lambda p, z: (np.zeros_like(z), np.zeros_like(p), z),
            p0,
            z0,
            H=0.2,
            M=2,
            scales=np.ones_like(p0),
            max_iterations=1,
        )
    counters = caught.value.research_counters
    assert counters["full_hydro_eval"] == 1
    assert counters["cheap_D"] == 1
    assert counters["cheap_L"] == 0

    def failed_lorentz(p, electric, magnetic):
        del p, electric, magnetic
        raise RuntimeError("injected Lorentz failure")

    with pytest.raises(Exception) as caught:
        module.frozen_midpoint_callbacks(
            macro,
            lambda p: np.zeros_like(p),
            failed_lorentz,
            lambda p, z: (np.zeros_like(z), np.zeros_like(p), z),
            p0,
            z0,
            H=0.2,
            M=2,
            scales=np.ones_like(p0),
            max_iterations=1,
        )
    counters = caught.value.research_counters
    assert counters["full_hydro_eval"] == 1
    assert counters["cheap_D"] == 1
    assert counters["cheap_L"] == 1
    assert counters["field_rhs"] == 1


def test_real_adapter_failed_calls_are_charged(monkeypatch):
    module = _module()
    from test_massless_support import context, uniform_state

    context = context(K=0, hard_diagnostics=False)
    p, z = _pack_state(context, uniform_state(context))

    def failed(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected adapter failure")

    adapter = module.ReducedHydroAdapter(context)
    with monkeypatch.context() as patch:
        patch.setattr(module, "build_fluid", failed)
        with pytest.raises(RuntimeError, match="injected"):
            adapter.macro(p, z)
    assert adapter.counters.full_hydro_eval == 1

    adapter = module.ReducedHydroAdapter(context)
    with monkeypatch.context() as patch:
        patch.setattr(module.core, "mode_apply", failed)
        with pytest.raises(RuntimeError, match="injected"):
            adapter.density_rate(p)
    assert adapter.counters.cheap_D == 1

    adapter = module.ReducedHydroAdapter(context)
    with monkeypatch.context() as patch:
        patch.setattr(module.FieldBatch, "rhs", failed)
        with pytest.raises(RuntimeError, match="injected"):
            adapter.field(p, z)
    assert adapter.counters.field_rhs == 1

    adapter = module.ReducedHydroAdapter(context)
    field = module.ReducedHydroAdapter(context).field(p, z)
    with monkeypatch.context() as patch:
        patch.setattr(module, "project_wall", failed)
        with pytest.raises(RuntimeError, match="injected"):
            adapter.lorentz(p, field.electric, z[:, 2 * context.op.Nc :])
    assert adapter.counters.cheap_L == 1
    assert adapter.counters.wall_projection == 1
    assert adapter.counters.wall_normal_initial_solves == context.N


def test_closed_wall_initial_solves_are_counted_separately_from_fieldbatch():
    module = _module()
    from warp_t2_field import closed_inputs

    context, p, z, _ = closed_inputs(6, 0.02, 0.1, "cpu")
    adapter = module.ReducedHydroAdapter(context)
    adapter.macro(p, z)
    assert adapter.counters.wall_projection == 1
    assert adapter.counters.wall_normal_initial_solves == context.N
    assert adapter.field_batch.work["normal_systems"] >= 0


def test_zero_row_wall_projection_has_no_initial_normal_solve():
    module = _module()
    from warp_t2_field import periodic_inputs

    context, p, z, _ = periodic_inputs(
        9, 2.0e-6, 0.1, "cpu", spatial_mode=1
    )
    adapter = module.ReducedHydroAdapter(context)
    adapter.macro(p, z)
    assert context.op.T.shape[0] == 0
    assert adapter.counters.wall_projection == 1
    assert adapter.counters.wall_normal_initial_solves == 0


def test_physical_p_and_z_scales_have_separate_component_units():
    module = _module()
    from warp_t2_field import periodic_inputs

    context, p, z, _ = periodic_inputs(
        9, 2.0e-6, 0.1, "cpu", spatial_mode=1
    )
    scales = module._physical_scales(context)
    nc = context.op.Nc
    q = 2 * context.p.K + 1
    p_values = np.asarray(scales.p)
    z_values = np.asarray(scales.z)
    ion_mass = float(context.p.species[0].mass)

    assert p_values.shape == np.asarray(p).shape
    assert z_values.shape == np.asarray(z).shape
    np.testing.assert_allclose(
        p_values[:, :nc], float(context.p.ref.ni)
    )
    np.testing.assert_allclose(
        p_values[:, nc : 4 * nc],
        ion_mass
        * float(context.p.ref.ni)
        * float(context.p.ref.velocity),
    )
    np.testing.assert_allclose(
        p_values[:, 4 * nc :], float(context.p.ref.pressure)
    )
    np.testing.assert_allclose(
        z_values[:, : 2 * nc], float(context.p.ref.pressure)
    )
    np.testing.assert_allclose(
        z_values[:, 2 * nc :], float(context.p.ref.magnetic)
    )
    assert p_values.shape[0] == q


def test_harness_component_error_normalization_rejects_wrong_scale_layout():
    tool = _tool_module()
    expected = np.array([[1.0e12, 1.0e-20, 1.0e5]])
    actual = expected + np.array([[1.0e5, -2.0e-22, 2.0e3]])
    scales = np.array([[1.0e12, 1.0e-20, 1.0e5]])

    raw, normalized = tool._relative_error(actual, expected, scales)
    assert raw == pytest.approx(1.0e5)
    assert normalized == pytest.approx(2.0e-2)
    with pytest.raises(ValueError, match="scale shape"):
        tool._relative_error(
            np.array([[1.0, 2.0]]),
            np.array([[0.0, 0.0]]),
            np.ones((2, 1)),
        )


def test_stage_energy_diagnostics_select_only_final_outer_sweeps():
    tool = _tool_module()
    trial = SimpleNamespace(outer=0, marker="trial")
    final = SimpleNamespace(outer=1, marker="final")
    other_final = SimpleNamespace(outer=3, marker="other-final")
    windows = [
        SimpleNamespace(
            iterations=(
                SimpleNamespace(outer=0, converged=False),
                SimpleNamespace(outer=1, converged=True),
            ),
            stages=(trial, final),
        ),
        SimpleNamespace(
            iterations=(SimpleNamespace(outer=3, converged=True),),
            stages=(other_final,),
        ),
    ]
    selected = tool._final_outer_stages(windows)
    assert selected == [final, other_final]


def test_electron_power_uses_zero_mode_nonunit_cell_measure_quadrature():
    module = _module()
    tool = _tool_module()
    from fomhd import core
    from warp_t2_field import closed_inputs

    context, p, z, _ = closed_inputs(6, 0.02, 0.1, "cpu")
    nc = context.op.Nc
    q = 2 * context.p.K + 1
    cell_measure = np.linspace(0.75, 2.5, nc)
    context.op.M0 = np.asarray(cell_measure)
    z_rhs = np.zeros((q, 2 * nc + context.op.Nb), dtype=np.complex128)
    zero = context.p.K
    z_rhs[zero, :nc] = np.linspace(1.0, 2.0, nc)
    z_rhs[zero, nc : 2 * nc] = np.linspace(-0.5, 0.5, nc)
    z_rhs[zero + 1, :nc] = 1.0e6
    z_rhs[zero - 1, :nc] = 1.0e6
    perpendicular_samples = core.inverse(
        z_rhs[:, :nc].reshape(-1),
        nc,
        context,
        True,
    ).reshape(context.N, nc)
    parallel_samples = core.inverse(
        z_rhs[:, nc : 2 * nc].reshape(-1),
        nc,
        context,
        True,
    ).reshape(context.N, nc)
    physical_expected = 2.0 * math.pi * float(
        np.sum(
            cell_measure
            * np.mean(
                np.real(perpendicular_samples + 0.5 * parallel_samples),
                axis=0,
            )
        )
    )
    unweighted = 2.0 * math.pi * float(
        np.sum(
            np.mean(
                np.real(perpendicular_samples + 0.5 * parallel_samples),
                axis=0,
            )
        )
    )
    assert unweighted != pytest.approx(physical_expected)
    stage = SimpleNamespace(
        p=p,
        z=np.zeros_like(z),
        z_rhs=z_rhs,
        electric=np.zeros(
            q * context.op.Ne, dtype=np.complex128
        ),
        outer=0,
        micro_step=0,
        rk_stage=0,
    )
    rows = tool._stage_energy_rows(context, [stage], 1.0e-12)
    assert rows[0]["electron_energy_rate_instantaneous_W"] == pytest.approx(
        physical_expected
    )


def test_provenance_uses_task_requests_and_relocated_scratch_paths():
    tool = _tool_module()
    provenance = tool._provenance({"method": "C1"})
    files = set(provenance["files"])
    source_files = list((tool.SOURCE_ROOT / "fomhd").glob("*.py"))
    assert source_files
    required = {
        Path(tool.__file__).resolve(),
        tool.RESEARCH_PATH.resolve(),
        tool.TEST_PATH.resolve(),
        (tool.SOURCE_PROJECT / "tools/warp_t2_field.py").resolve(),
        (
            tool.SOURCE_PROJECT
            / "tests/warp/massless_verification_fixture.py"
        ).resolve(),
        (
            tool.SOURCE_PROJECT / "tests/warp/test_massless_support.py"
        ).resolve(),
        (
            tool.PSEUDOCODE_ROOT / "src/fomhd/time_multirate.md"
        ).resolve(),
        (
            tool.PSEUDOCODE_ROOT / "tests/warp/test_time_multirate.py.md"
        ).resolve(),
        (
            tool.PSEUDOCODE_ROOT / "tools/warp_multirate_ion.py.md"
        ).resolve(),
        (
            tool.SOURCE_PROJECT
            / "docs/pseudocode/tools/warp_t2_field.py.md"
        ).resolve(),
        (
            tool.SOURCE_PROJECT
            / "docs/pseudocode/tests/warp/massless_verification_fixture.py.md"
        ).resolve(),
        (
            tool.SOURCE_PROJECT
            / "docs/pseudocode/tests/warp/test_massless_support.py.md"
        ).resolve(),
        *(path.resolve() for path in source_files),
    }
    expected = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in required
    }
    assert {str(path) for path in required} <= files
    assert {path: provenance["files"][path] for path in expected} == expected
    for path in (
        tool.TASK_ROOT / "REQUEST.md",
        tool.TASK_ROOT / "REQUEST_REPAIR5.md",
        tool.TASK_ROOT / "LOGS/worker.log",
    ):
        if path.is_file():
            assert str(path.resolve()) in files
            assert (
                provenance["files"][str(path.resolve())]
                == hashlib.sha256(path.read_bytes()).hexdigest()
            )
    assert provenance["task_root"] == str(tool.TASK_ROOT)
    assert provenance["task_root_scope"].startswith("optional worker evidence")
    assert provenance["scratch_root"] == str(tool.SCRATCH_ROOT)
    assert provenance["source_commit"] == provenance["source_git"]["head"]
    assert provenance["source_branch"] == provenance["source_git"]["branch"]
    assert provenance["source_dirty"] == provenance["source_git"]["dirty"]
    assert (
        provenance["declared_baseline"]["commit"]
        == "be5f1c07913409bfd377f22e9ea9d69ac614b158"
    )


def test_provenance_allows_missing_optional_receipts(tmp_path, monkeypatch):
    tool = _tool_module()
    monkeypatch.setattr(tool, "TASK_ROOT", tmp_path)
    provenance = tool._provenance({"method": "C1"})
    files = set(provenance["files"])
    assert str((tmp_path / "REQUEST.md").resolve()) not in files
    assert str((tmp_path / "LOGS/worker.log").resolve()) not in files


def test_source_project_derives_installed_root_and_honors_override(tmp_path):
    tool = _tool_module()
    installed = tmp_path / "installed"
    (installed / "src/fomhd").mkdir(parents=True)
    (installed / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    derived, origin = tool._select_source_project(
        installed / "tools/warp_multirate_ion.py",
        None,
    )
    assert derived == installed.resolve()
    assert origin == "installed_source"
    override = tmp_path / "explicit"
    explicit, origin = tool._select_source_project(
        installed / "tools/warp_multirate_ion.py",
        str(override),
    )
    assert explicit == override.resolve()
    assert origin == "FOMHD_SOURCE_ROOT"
    fallback, origin = tool._select_source_project(
        tmp_path / "task/scratch/tools/warp_multirate_ion.py",
        None,
    )
    assert fallback == tool.DECLARED_SOURCE_B
    assert origin == "declared_source_b_fallback"


def test_jsonable_recursively_hosts_complex_array_scalars():
    tool = _tool_module()
    value = {
        "array": np.array([[1.0 + 2.0j]]),
        "scalar": np.complex128(3.0 - 4.0j),
    }
    encoded = json.dumps(
        tool._jsonable(value),
        allow_nan=False,
    )
    assert json.loads(encoded) == {
        "array": [[{"real": 1.0, "imag": 2.0}]],
        "scalar": {"real": 3.0, "imag": -4.0},
    }


def test_candidate_stage_arrays_signal_unavailable_c1_p_rhs():
    tool = _tool_module()
    context = SimpleNamespace(compute=SimpleNamespace(xp=np))
    p = np.zeros((1, 5), dtype=np.complex128)
    z = np.zeros((1, 3), dtype=np.complex128)
    stage = SimpleNamespace(
        p=p,
        z=z,
        p_rhs=None,
        p_rhs_available=False,
        z_rhs=z,
        electric=p,
        magnetic=z,
        outer=0,
        micro_step=0,
        rk_stage=0,
    )
    window = SimpleNamespace(
        stages=(stage,),
        iterations=(
            SimpleNamespace(
                outer=0,
                residual_raw=1.0,
                residual_normalized=0.5,
                scale=2.0,
                actual_increment_max=0.25,
                converged=False,
            ),
        ),
    )
    arrays = tool._candidate_arrays(
        [window],
        context,
        start_time=2.0,
        macro_H=0.5,
        microstep=0.125,
    )
    assert "candidate_stage_p_rhs" not in arrays
    assert not bool(arrays["candidate_stage_p_rhs_available"][0])
    assert arrays["candidate_stage_window"][0] == 0
    assert arrays["candidate_stage_time_s"][0] == pytest.approx(2.0)
    assert arrays["candidate_iteration_window"][0] == 0
    assert arrays["candidate_iteration_time_s"][0] == pytest.approx(2.5)


def test_candidate_macro_H_uses_microstep_times_M_for_multiple_windows(
    monkeypatch,
):
    tool = _tool_module()
    p0 = np.array([[1.0 + 0.0j]])
    z0 = np.array([[2.0 + 0.0j]])
    calls = []

    def fake_run(method, context, p, z, H, M, **kwargs):
        del method, context, M, kwargs
        calls.append(float(H))
        return SimpleNamespace(p=p.copy(), z=z.copy())

    monkeypatch.setattr(tool.research, "run", fake_run)
    config = {
        "method": "C1",
        "h": 0.125,
        "M": 4,
        "macros": 3,
        "rtol": 1.0e-10,
        "max_iterations": 2,
        "endpoint_check": False,
        "warmups": 0,
        "repetitions": 1,
    }
    windows, _ = tool._run_candidates(
        SimpleNamespace(compute=SimpleNamespace(xp=np)),
        p0,
        z0,
        0.125,
        config,
    )
    assert len(windows) == config["macros"]
    assert calls == [0.5] * (2 * config["macros"])
    timing = tool._simulation_timing(config)
    assert timing["microstep_h_s"] == 0.125
    assert timing["macro_H_s"] == 0.5
    assert timing["total_duration_s"] == 1.5
    assert timing["candidate_duration_s"] == timing["total_duration_s"]
    assert timing["reference_duration_s"] == timing["total_duration_s"]


def test_candidate_and_reference_timing_use_fresh_contexts_and_matched_protocol(
    monkeypatch,
):
    tool = _tool_module()
    p0 = np.array([[1.0 + 0.0j]])
    z0 = np.array([[2.0 + 0.0j]])
    context = SimpleNamespace(
        compute=SimpleNamespace(xp=np),
        p=SimpleNamespace(start_time=0.0),
    )
    captured = []

    def fake_run(method, local, p, z, H, M, **kwargs):
        del method, H, M, kwargs
        captured.append(local)
        counters = {
            name: 0 for name in tool.research.Counters.__slots__
        }
        counters["actual_outer_calls"] = 1
        return SimpleNamespace(
            p=p.copy(),
            z=z.copy(),
            counters=counters,
        )

    monkeypatch.setattr(tool.research, "run", fake_run)
    config = {
        "method": "C1",
        "h": 0.125,
        "M": 4,
        "macros": 3,
        "rtol": 1.0e-10,
        "max_iterations": 2,
        "endpoint_check": False,
        "warmups": 1,
        "repetitions": 2,
    }
    _, timing = tool._run_candidates(
        context,
        p0,
        z0,
        0.125,
        config,
    )
    assert timing["warmups"] == 1
    assert timing["repetitions"] == 2
    assert len(timing["executions"]) == 4
    assert timing["campaign_multiplier_per_algorithm"] == 4
    assert len({id(local) for local in captured}) == 4
    assert all(
        execution["counters"]["actual_outer_calls"] == 3
        for execution in timing["executions"]
    )


def test_reference_timing_uses_the_same_fresh_context_protocol(monkeypatch):
    tool = _tool_module()
    context = SimpleNamespace(
        compute=SimpleNamespace(xp=np),
        p=SimpleNamespace(start_time=0.0),
    )
    captured = []

    def fake_reference(local, p0, z0, **kwargs):
        del p0, z0, kwargs
        captured.append(local)
        return SimpleNamespace(
            frames=(object(), object()),
            cumulative=SimpleNamespace(
                last_trial=2,
                accepted_intervals=2,
            ),
        )

    monkeypatch.setattr(tool, "_run_reference", fake_reference)
    _, timing = tool._run_reference_timed(
        context,
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        h=0.1,
        M=2,
        macros=1,
        coupling="synchronous",
        rtol=1.0e-10,
        max_iterations=2,
        warmups=1,
        repetitions=2,
    )
    assert len(timing["executions"]) == 4
    assert len({id(local) for local in captured}) == 4
    assert timing["campaign_multiplier_per_algorithm"] == 4
    assert all(
        execution["counters"]["production_advances"] == 2
        for execution in timing["executions"]
    )


def test_hard_stage_audit_preserves_observation_policy():
    tool = _tool_module()
    from warp_t2_field import closed_inputs

    context, p, z, _ = closed_inputs(6, 0.02, 0.1, "cpu")
    key = next(
        key
        for key, budget in context.p.budgets.items()
        if getattr(budget, "hard", True) is False
    )
    before = {
        name: bool(getattr(budget, "hard", True))
        for name, budget in context.p.budgets.items()
    }
    audit = tool._hard_stage_audit(context, p, z, "repair3_audit")
    after = {
        name: bool(getattr(budget, "hard", True))
        for name, budget in context.p.budgets.items()
    }
    assert after == before
    assert audit["budget_hard_flags"] == before
    assert audit["budget_hard_flags"][key] is False


def test_mode_diagnostics_separate_periodic_sentinel_from_target():
    tool = _tool_module()
    from massless_verification_fixture import HallCase
    from warp_t2_field import periodic_inputs

    context, p, z, _ = periodic_inputs(
        9, 2.0e-6, 0.1, "cpu", spatial_mode=1
    )
    case = HallCase(points=9)
    window = SimpleNamespace(p=p, z=z)
    comparison = tool._mode_comparison(
        context,
        p,
        z,
        [window],
        SimpleNamespace(p=p, z=z, endpoints=[(p, z)]),
        target_mode=1,
        sentinel_mode=4,
        case=case,
    )
    assert comparison["sentinel"]["independent"] is True
    assert comparison["target"]["mode"] == 1
    assert comparison["sentinel"]["mode"] == 4
    assert comparison["target"]["candidate_history"]["final_gain"] >= 0.0
    assert comparison["sentinel"]["candidate_history"]["maximum_gain"] >= 0.0
    same = tool._mode_comparison(
        context,
        p,
        z,
        [window],
        SimpleNamespace(p=p, z=z, endpoints=[(p, z)]),
        target_mode=4,
        sentinel_mode=4,
        case=case,
    )
    assert same["sentinel"]["independent"] is False
    assert same["sentinel"]["candidate_history"] is not None


def test_failure_after_completed_macro_retains_parseable_history_and_inputs(
    monkeypatch,
    tmp_path,
):
    tool = _tool_module()
    from warp_t2_field import periodic_inputs

    context, p0, z0, _ = periodic_inputs(
        9, 2.0e-6, 0.1, "cpu", spatial_mode=1
    )
    p_before, z_before = p0.copy(), z0.copy()
    monkeypatch.setattr(
        tool,
        "build_case",
        lambda config: (context, p0, z0, None),
    )
    original_run = tool.research.run
    calls = 0
    first_result = None

    def fail_second_window(*args, **kwargs):
        nonlocal calls, first_result
        calls += 1
        if calls == 1:
            first_result = original_run(*args, **kwargs)
            return first_result
        first_p, first_z = args[2], args[3]
        failure = tool.Failure(
            "InjectedFailure",
            "fail after one completed macro",
        )
        failure.research_history = (
            SimpleNamespace(
                outer=0,
                guess_p=first_p.copy(),
                endpoint_p=first_p.copy(),
                target_p=first_p.copy(),
                residual=np.asarray(first_p).copy(),
                residual_raw=1.0,
                residual_normalized=0.5,
                scale=2.0,
                actual_increment_max=0.25,
                converged=False,
                stage_count=4,
            ),
        )
        failure.research_stages = ()
        failure.research_counters = {
            "actual_outer_calls": 1,
            "field_rhs": 1,
        }
        failure.last_attempt_p = np.asarray(first_p).copy()
        failure.last_attempt_z = np.asarray(first_z).copy()
        raise failure

    monkeypatch.setattr(tool.research, "run", fail_second_window)
    output = tmp_path / "failure-after-macro"
    config = {
        "case": "periodic",
        "device": "cpu",
        "method": "C1",
        "reference": "none",
        "points": 9,
        "amplitude": 2.0e-6,
        "nonuniform": 0.1,
        "h": 5.0e-12,
        "M": 2,
        "macros": 2,
        "rtol": 1.0e-9,
        "max_iterations": 4,
        "seed_mode": 1,
        "sentinel": 1.0e-4,
        "endpoint_check": False,
        "warmups": 0,
        "repetitions": 1,
    }
    metadata = tool.run(config, output)

    report = json.loads((output / "report.json").read_text())
    assert metadata["status"] == "failed"
    assert report["status"] == "failed"
    assert report["completed_window_count"] == 1
    assert report["completed_current_window_iterations"] == 1
    assert report["failed_window_index"] == 1
    assert report["counters"]["actual_outer_calls"] == (
        first_result.counters["actual_outer_calls"] + 1
    )
    assert report["field_batch_work"]["rhs_vectors"] >= 1
    assert len(report["iterations"]) == 3
    assert report["iterations"][0]["window"] == 0
    assert report["iterations"][1]["window"] == 0
    assert report["iterations"][2]["window"] == 1
    assert report["iterations"][2]["time_s"] == pytest.approx(
        report["time_contract"]["end_time_s"]
    )
    assert (output / "traceback.txt").exists()
    np.testing.assert_array_equal(p0, p_before)
    np.testing.assert_array_equal(z0, z_before)
    with np.load(output / "failure-arrays.npz") as arrays:
        assert "candidate_stage_p" in arrays
        assert "failure_iteration_residual" in arrays
        assert "last_attempt_p" in arrays
        assert "last_attempt_z" in arrays


def test_failure_after_one_stage_retains_parseable_report_and_npz(
    monkeypatch,
    tmp_path,
):
    tool = _tool_module()
    original_field = tool.research.ReducedHydroAdapter.field
    calls = 0

    def fail_second_stage(adapter, p, z):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second-stage failure")
        return original_field(adapter, p, z)

    monkeypatch.setattr(
        tool.research.ReducedHydroAdapter,
        "field",
        fail_second_stage,
    )
    output = tmp_path / "failure-after-stage"
    config = {
        "case": "periodic",
        "device": "cpu",
        "method": "C1",
        "reference": "none",
        "points": 9,
        "amplitude": 2.0e-6,
        "nonuniform": 0.1,
        "h": 5.0e-12,
        "M": 2,
        "macros": 1,
        "rtol": 1.0e-9,
        "max_iterations": 2,
        "seed_mode": 1,
        "sentinel": 1.0e-4,
        "endpoint_check": False,
        "warmups": 0,
        "repetitions": 1,
    }
    metadata = tool.run(config, output)

    report = json.loads((output / "report.json").read_text())
    assert metadata["status"] == "failed"
    assert report["completed_window_count"] == 0
    assert report["completed_current_window_stages"] == 1
    assert report["counters"]["field_rhs"] == 2
    assert (output / "traceback.txt").exists()
    with np.load(output / "failure-arrays.npz") as arrays:
        assert arrays["failure_stage_p"].shape[0] == 1
        assert arrays["failure_stage_p_rhs_available"].tolist() == [False]
        assert "failure_stage_p_rhs" not in arrays
        assert "last_attempt_p" in arrays
        assert "last_attempt_z" in arrays


def test_failed_window_stage_array_keeps_its_macro_window_and_time():
    tool = _tool_module()
    from warp_t2_field import periodic_inputs

    context, p, z, _ = periodic_inputs(
        9, 2.0e-6, 0.1, "cpu", spatial_mode=1
    )
    q = 2 * context.p.K + 1
    stage = SimpleNamespace(
        p=p,
        z=z,
        p_rhs=None,
        p_rhs_available=False,
        z_rhs=z,
        electric=np.zeros(
            q * context.op.Ne, dtype=np.complex128
        ),
        magnetic=z[:, 2 * context.op.Nc :],
        outer=0,
        micro_step=0,
        rk_stage=0,
    )
    error = SimpleNamespace(
        research_history=(),
        research_stages=(stage,),
    )
    arrays = tool._failure_history_arrays(
        error,
        context,
        start_time=0.0,
        window_index=1,
        macro_H=0.5,
        microstep=0.125,
    )
    assert arrays["failure_stage_window"][0] == 1
    assert arrays["failure_stage_time_s"][0] == pytest.approx(0.5)


def test_scientific_qualification_is_not_implied_by_execution_completion():
    tool = _tool_module()
    qualification = tool._qualification(
        {
            "p_normalized": 1.0e-12,
            "z_normalized": 1.0e-12,
        },
        endpoint_checks=[],
    )
    assert qualification["execution_completed"] is True
    assert qualification["scientific_qualified"] is False
    assert qualification["macro_order"] == "second-order design"
    assert qualification["nonzero_hydro_order_evidence"] is False


@pytest.mark.parametrize("order", (0, 1, 2, 3))
def test_phi_zero_limit_uses_factorial_order(order):
    module = _module()
    assert module._phi(0.0j, order) == pytest.approx(
        1.0 / math.factorial(order)
    )


@pytest.mark.parametrize(
    "beta_H",
    (0.0, 1.0e-12, -1.0e-12, 9.999e-6, 1.0e-5, 1.0001e-5),
)
def test_cold_hall_map_matches_independent_exponential_near_branch(beta_H):
    module = _module()
    a, H = 0.37, 0.02
    beta = beta_H / H
    actual = np.asarray(module.cold_hall_t0_map(a, beta, H))
    expected = _called_t0_map(a, beta, H)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12)


def test_cold_hall_beta_zero_matches_exact_cayley_limit():
    module = _module()
    a, H = 1.3, 2.0e-8
    actual = np.asarray(module.cold_hall_t0_map(a, 0.0, H))
    expected = expm(np.array([[0.0, 1j * a], [1j * a, 0.0]]) * H)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-13)


def test_closed_endpoint_retains_q_mode_layout_and_quasineutrality():
    module = _module()
    from fomhd import core
    from warp_t2_field import closed_inputs

    context, p, z, _ = closed_inputs(6, 0.02, 0.1, "cpu")
    adapter = module.ReducedHydroAdapter(context)
    field, state, cache = adapter.endpoint(p, z)
    q = 2 * context.p.K + 1
    nc = context.op.Nc
    ion = context.p.species[0]
    electron = context.p.species[1]
    ratio = float(ion.charge) / (-float(electron.charge))
    expected_electron_n = (
        ratio * np.asarray(p)[:, :nc].reshape(-1)
    )

    assert state.fluid[0].n.size == q * nc
    assert state.fluid[0].mom.size == q * 3 * nc
    assert state.fluid[0].perp.size == q * nc
    assert state.fluid[1].n.size == q * nc
    assert state.fluid[1].mom.size == q * 3 * nc
    assert state.fluid[1].perp.size == q * nc
    assert state.fluid[1].parallel.size == q * nc
    assert state.B.size == q * context.op.Nb
    np.testing.assert_allclose(state.fluid[1].n, expected_electron_n)

    ion_density = core.inverse(
        np.asarray(state.fluid[0].n), nc, context, True
    )
    electron_density = core.inverse(
        np.asarray(state.fluid[1].n), nc, context, True
    )
    charge_density = (
        float(ion.charge) * ion_density
        + float(electron.charge) * electron_density
    )
    neutral_scale = abs(float(ion.charge)) * max(
        float(np.max(np.abs(ion_density))), 1.0e-300
    )
    assert float(np.max(np.abs(charge_density))) <= 2.0e-12 * neutral_scale

    core.check_retained(state, context)
    assert cache.version == field.raw.identity
    assert cache.version == adapter.field_batch._identity
    assert cache.tag == adapter._tag


def test_closed_endpoint_split_uses_blockwise_complete_ion_rhs():
    module = _module()
    from warp_t2_field import closed_inputs

    context, p, z, _ = closed_inputs(6, 0.02, 0.1, "cpu")
    adapter = module.ReducedHydroAdapter(context)
    field, _, _ = adapter.endpoint(p, z)
    full, stage = adapter.full_ion_rhs(p, z)
    q = 2 * context.p.K + 1
    nc = context.op.Nc
    assert full.shape == p.shape
    np.testing.assert_allclose(
        full[:, :nc], np.asarray(stage.rhs.fluid[0].n).reshape(q, nc)
    )
    np.testing.assert_allclose(
        full[:, nc : 4 * nc],
        np.asarray(stage.rhs.fluid[0].mom).reshape(q, 3 * nc),
    )
    np.testing.assert_allclose(
        full[:, 4 * nc :],
        np.asarray(stage.rhs.fluid[0].perp).reshape(q, nc),
    )

    full_calls_before_check = adapter.counters.endpoint_full_ion_RHS
    checked = adapter.check_endpoint_split(
        SimpleNamespace(p=p, z=z, endpoint=field)
    )
    assert checked.passed
    assert checked.full.shape == p.shape
    assert (
        adapter.counters.endpoint_full_ion_RHS
        == full_calls_before_check + 1
    )


def test_endpoint_rejects_nonpositive_ion_pressure_without_full_oracle():
    module = _module()
    from fomhd import core
    from warp_t2_field import closed_inputs

    context, p, z, _ = closed_inputs(6, 0.02, 0.1, "cpu")
    nc = context.op.Nc
    q = 2 * context.p.K + 1
    negative_pressure = -np.ones((context.N, nc), dtype=np.complex128)
    p_bad = p.copy()
    p_bad[:, 4 * nc :] = core.forward(
        negative_pressure.reshape(-1), nc, context
    ).reshape(q, nc)
    adapter = module.ReducedHydroAdapter(context)
    with pytest.raises(Exception, match="pressure"):
        adapter.endpoint(p_bad, z)
    assert adapter.counters.endpoint_check == 1
    assert adapter.counters.endpoint_full_ion_RHS == 0


@pytest.mark.parametrize("macros", (1, 32))
def test_macro_limit_accepts_parent_range(monkeypatch, tmp_path, macros):
    tool = _tool_module()
    observed = {}

    def fake_run(config, output):
        observed.update(config)
        observed["output"] = output
        return {"status": "completed"}

    monkeypatch.setattr(tool, "run", fake_run)
    assert (
        tool.main(
            [
                "--method",
                "C1",
                "--h",
                "1e-3",
                "--M",
                "2",
                "--macros",
                str(macros),
                "--output",
                str(tmp_path / f"macro-{macros}"),
            ]
        )
        == 0
    )
    assert observed["macros"] == macros


@pytest.mark.parametrize("macros", (0, 33))
def test_macro_limit_rejects_values_outside_parent_range(macros):
    tool = _tool_module()
    with pytest.raises(SystemExit):
        tool.main(
            [
                "--method",
                "C1",
                "--h",
                "1e-3",
                "--M",
                "2",
                "--macros",
                str(macros),
                "--output",
                "/tmp/unused-multirate-macro-bound",
            ]
        )


def test_periodic_split_matches_full_production_ion_rhs_and_B_invariance():
    module = _module()
    from fomhd.massless import evaluate_stage
    from massless_verification_fixture import stage_tag
    from warp_t2_field import periodic_inputs, unpack_state

    context, p, z, _ = periodic_inputs(
        9, 2.0e-6, 0.1, "cpu", spatial_mode=1
    )
    state = unpack_state(context, np.asarray(p).copy(), np.asarray(z).copy())
    stage = evaluate_stage(state, stage_tag(state), context, False)
    q = 2 * context.p.K + 1
    full_p = np.empty_like(p)
    full_p[:, : context.op.Nc] = np.asarray(
        stage.rhs.fluid[0].n
    ).reshape(q, context.op.Nc)
    full_p[:, context.op.Nc : 4 * context.op.Nc] = np.asarray(
        stage.rhs.fluid[0].mom
    ).reshape(q, 3 * context.op.Nc)
    full_p[:, 4 * context.op.Nc :] = np.asarray(
        stage.rhs.fluid[0].perp
    ).reshape(q, context.op.Nc)
    adapter = module.ReducedHydroAdapter(context)
    assembled_full, _ = adapter.full_ion_rhs(p, z)
    assembled_scales = np.asarray(module._physical_scales(context).p)
    assembled_normalized = np.abs(assembled_full - full_p) / np.maximum(
        np.maximum(assembled_scales, np.abs(full_p)),
        1.0e-300,
    )
    assert float(np.max(assembled_normalized)) <= 2.0e-12
    macro = adapter.macro(p, z)

    field = adapter.field(p, z)
    nc = context.op.Nc
    electric = field.electric
    magnetic = z[:, 2 * nc :]
    lorentz = adapter.lorentz(p, electric, magnetic)
    split_p = (
        np.asarray(macro.density_rate)
        + np.asarray(macro.momentum_rate)
        + np.asarray(lorentz)
        + np.asarray(macro.pressure_rate)
    )
    p_scales = np.asarray(module._physical_scales(context).p)
    normalized = np.abs(split_p - full_p) / np.maximum(
        np.maximum(np.abs(p_scales), np.abs(full_p)),
        1.0e-300,
    )
    assert float(np.max(normalized)) <= 5.0e-10

    z_alt = z.copy()
    z_alt[:, 2 * nc :] *= 1.17
    alt = adapter.macro(p, z_alt)
    for name in ("density_rate", "momentum_rate", "pressure_rate"):
        left = np.asarray(getattr(macro, name))
        right = np.asarray(getattr(alt, name))
        local_scale = max(float(np.max(np.abs(left))), 1.0e-300)
        assert float(np.max(np.abs(left - right))) <= 2.0e-12 * local_scale


def test_adapter_does_not_touch_accepted_context_or_wall_counters():
    module = _module()
    from fomhd.wall_cache import get_wall_cache
    from test_massless_support import context, uniform_state

    context = context(K=0, hard_diagnostics=False)
    state = uniform_state(context)
    p, z = _pack_state(context, state)
    cache = get_wall_cache(context)
    before = {
        "generation": context.next_generation,
        "trial": context.next_trial,
        "source": context.next_source,
        "cache_builds": getattr(context, "_wall_system_cache_builds", 0),
        "cache_solves": cache.solves,
        "cache_evaluations": cache.evaluations,
    }
    adapter = module.ReducedHydroAdapter(context)
    adapter.macro(p, z)
    field = adapter.field(p, z)
    adapter.lorentz(p, field.electric, z[:, 2 * context.op.Nc :])
    after = {
        "generation": context.next_generation,
        "trial": context.next_trial,
        "source": context.next_source,
        "cache_builds": getattr(context, "_wall_system_cache_builds", 0),
        "cache_solves": cache.solves,
        "cache_evaluations": cache.evaluations,
    }
    assert after == before


def test_closed_rz_split_matches_full_production_ion_rhs():
    module = _module()
    from fomhd.massless import evaluate_stage
    from massless_verification_fixture import stage_tag
    from warp_t2_field import closed_inputs, unpack_state

    context, p, z, _ = closed_inputs(6, 0.02, 0.1, "cpu")
    state = unpack_state(context, np.asarray(p).copy(), np.asarray(z).copy())
    stage = evaluate_stage(state, stage_tag(state), context, False)
    q = 2 * context.p.K + 1
    full = np.empty_like(p)
    full[:, : context.op.Nc] = np.asarray(
        stage.rhs.fluid[0].n
    ).reshape(q, context.op.Nc)
    full[:, context.op.Nc : 4 * context.op.Nc] = np.asarray(
        stage.rhs.fluid[0].mom
    ).reshape(q, 3 * context.op.Nc)
    full[:, 4 * context.op.Nc :] = np.asarray(
        stage.rhs.fluid[0].perp
    ).reshape(q, context.op.Nc)
    adapter = module.ReducedHydroAdapter(context)
    macro = adapter.macro(p, z)
    field = adapter.field(p, z)
    lorentz = adapter.lorentz(
        p,
        field.electric,
        z[:, 2 * context.op.Nc :],
    )
    split = (
        np.asarray(macro.density_rate)
        + np.asarray(macro.momentum_rate)
        + np.asarray(lorentz)
        + np.asarray(macro.pressure_rate)
    )
    p_scales = np.asarray(module._physical_scales(context).p)
    normalized = np.abs(split - full) / np.maximum(
        np.maximum(np.abs(p_scales), np.abs(full)),
        1.0e-300,
    )
    assert float(np.max(normalized)) <= 2.0e-9


@pytest.mark.parametrize("method", ("C1", "C2"))
def test_real_periodic_candidates_converge_with_exact_work_counts(method):
    module = _module()
    from massless_verification_fixture import HallCase
    from warp_t2_field import periodic_inputs

    context, p, z, _ = periodic_inputs(
        9, 2.0e-6, 0.1, "cpu", spatial_mode=1
    )
    case = HallCase(points=9)
    h = 0.02 / (case.hall_rate * 16.0 / 0.9)
    before = {
        "generation": context.next_generation,
        "trial": context.next_trial,
        "source": context.next_source,
    }
    result = module.run(
        method,
        context,
        p,
        z,
        h,
        2,
        rtol=1.0e-9,
        max_iterations=4,
        keep_stages=True,
        endpoint_check=False,
    )
    outer = len(result.iterations)
    assert result.iterations[-1].converged
    assert result.counters["actual_outer_calls"] == outer
    assert result.counters["field_rhs"] == 4 * 2 * outer + 1
    if method == "C1":
        assert result.counters["cheap_D"] == outer
        assert result.counters["cheap_L"] == outer
        assert all(not stage.p_rhs_available for stage in result.stages)
        assert all(stage.p_rhs is None for stage in result.stages)
    else:
        assert result.counters["cheap_D"] == 4 * 2 * outer
        assert result.counters["cheap_L"] == 4 * 2 * outer
        assert all(stage.p_rhs_available for stage in result.stages)
        assert all(stage.p_rhs is not None for stage in result.stages)
    assert result.metadata["field_batch_work"]["rhs_vectors"] >= 1
    assert {
        "generation": context.next_generation,
        "trial": context.next_trial,
        "source": context.next_source,
    } == before


def test_failed_timing_includes_completed_macro_work():
    tool = _tool_module()
    error = tool.Failure('InjectedFailure', 'second macro')
    error.completed_windows = [SimpleNamespace(counters={'field_rhs': 7, 'full_hydro_eval': 3})]
    error.research_counters = {'field_rhs': 2, 'full_hydro_eval': 1}
    def fail():
        raise error
    with pytest.raises(tool.Failure):
        tool._timed(fail, np, 0, 1)
    counts = error.timing_executions[0]['counters']
    assert counts['field_rhs'] == 9
    assert counts['full_hydro_eval'] == 4


def test_late_reference_failure_preserves_completed_candidate(monkeypatch, tmp_path):
    tool = _tool_module()
    def fail_reference(*args, **kwargs):
        raise tool.Failure('InjectedReference', 'after candidate completed')
    monkeypatch.setattr(tool, '_run_reference_timed', fail_reference)
    cfg = dict(case='periodic', device='cpu', method='C1', reference='synchronous',
               points=9, amplitude=2e-6, nonuniform=.1, h=5e-12, M=2, macros=1,
               rtol=1e-9, max_iterations=4, seed_mode=1, sentinel=1e-4,
               endpoint_check=False, warmups=0, repetitions=1)
    output = tmp_path / 'late-failure'
    report = tool.run(cfg, output)
    assert report['failure_code'] == 'InjectedReference'
    assert report['completed_window_count'] == 1
    with np.load(output / 'failure-arrays.npz') as data:
        assert 'candidate_stage_p' in data
        assert data['candidate_p'].shape[0] >= 1
        assert 'candidate_z' in data


def test_cli_allows_bounded_fixed_H_half_h_control(monkeypatch, tmp_path):
    tool = _tool_module()
    seen = {}
    def capture(config, output):
        seen.update(config)
        return {'status': 'completed'}
    monkeypatch.setattr(tool, 'run', capture)
    assert tool.main(['--case', 'periodic', '--method', 'C2', '--points', '33',
                      '--h', '9.8973669428115e-11', '--M', '256', '--macros', '32',
                      '--output', str(tmp_path / 'refinement')]) == 0
    assert seen['M'] == 256 and seen['macros'] == 32
