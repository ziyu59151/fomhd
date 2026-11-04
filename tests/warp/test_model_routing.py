"""End-to-end model routing checks for finite-mass and massless electrons."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from foundation_fixture import parameters as foundation_parameters
from fomhd import io
from fomhd.__main__ import _context as command_context
from fomhd.__main__ import _run_command
from fomhd.backend import Backend
from fomhd.core import mode_apply
from fomhd.geometry import build_rectangle
from fomhd.operators import build_operators
from fomhd.stage import closed_reference, initialize
from fomhd.time import advance
from fomhd.types import Budget, Context, Failure, StageTag


MASSLESS_MODEL = "massless_quasineutral_cgl_b"
FINITE_MODEL = "finite_mass_cgl_b"


def _massless_budgets(p):
    ref = p.ref
    ion, electron = p.species
    ion_id = ion.id
    electron_id = electron.id
    density_rate = ref.ni / ref.time
    momentum_rate = (
        ion.mass * ref.ni * ref.velocity / ref.time
    )
    pressure_rate = ref.pressure / ref.time
    flux = ref.ni * ref.velocity
    length = ref.length
    current = ref.current

    budgets = {}

    def array(name, unit, reference, hard=True):
        for suffix, norm in (
            ("linf", "linf"),
            ("weighted_L2", "weighted_l2"),
        ):
            budgets[f"{name}.{suffix}"] = Budget(
                atol=0.0,
                rtol=2.0e-6,
                reference=reference,
                unit=unit,
                norm=norm,
                hard=hard,
            )

    def scalar(name, unit, reference, hard=True):
        budgets[f"{name}.value"] = Budget(
            atol=0.0,
            rtol=2.0e-6,
            reference=reference,
            unit=unit,
            norm="value",
            hard=hard,
        )

    for name, unit, reference in (
        ("quasineutrality", "C/m^3", ref.charge),
        (
            "electron_density_compatibility",
            "1/(m^3*s)",
            density_rate,
        ),
        (
            "electron_momentum_compatibility",
            "kg/(m^2*s^2)",
            electron.mass * density_rate * ref.velocity,
        ),
        ("current_definition", "A/m^2", current),
        ("electron_flux_definition", "1/(m^2*s)", flux),
        ("electron_flux_continuity", "1/(m^3*s)", density_rate),
        ("djj", "A/m^3", current / length),
        ("djj_interval", "A/m^3", current / length),
        (
            "ct_divergence",
            "T/(m*s)",
            ref.magnetic / (length * ref.time),
        ),
        ("ct_update", "T/s", ref.magnetic / ref.time),
        ("divB_update", "T/m", ref.magnetic / length),
        ("wall_current", "A/m^2", current),
        ("wall_current_interval", "A/m^2", current),
        ("wall_ion_flux", "1/(m^2*s)", flux),
        ("wall_electron_flux", "1/(m^2*s)", flux),
        (f"wall_ion_flux_interval.{ion_id}", "1/(m^2*s)", flux),
        (
            f"wall_electron_flux_interval.{electron_id}",
            "1/(m^2*s)",
            flux,
        ),
        (
            "wall_field_original_rows",
            "V/m^3",
            ref.electric / length**2,
        ),
        ("drift_ohm_residual", "V/m", ref.electric),
        ("electron_cgl_perp", "Pa/s", pressure_rate),
        ("electron_cgl_parallel", "Pa/s", pressure_rate),
        (
            "electron_pressure_adjoint",
            "W",
            ref.pressure * ref.velocity * length**2,
        ),
        (
            "electron_electric_pairing",
            "W",
            ref.pressure * ref.velocity * length**2,
        ),
        ("initial_divB", "T/m", ref.magnetic / length),
        ("wall_initial_current", "A/m^2", current),
        ("species_continuity", "1/(m^3*s)", density_rate),
        (f"species_continuity.{ion_id}", "1/(m^3*s)", density_rate),
        (f"species_continuity.{electron_id}", "1/(m^3*s)", density_rate),
        (
            "quasineutrality_interval",
            "C/(m^3*s)",
            ref.charge / ref.time,
        ),
        (f"alias_n.{ion_id}", "1/(m^3*s)", density_rate),
        (f"alias_n.{electron_id}", "1/(m^3*s)", density_rate),
        (f"alias_mom.{ion_id}", "kg/(m^2*s^2)", momentum_rate),
        (f"alias_mom.{electron_id}", "kg/(m^2*s^2)", momentum_rate),
        (f"alias_perp.{ion_id}", "Pa/s", pressure_rate),
        (f"alias_perp.{electron_id}", "Pa/s", pressure_rate),
        (f"alias_parallel.{electron_id}", "Pa/s", pressure_rate),
        ("alias_B", "T/s", ref.magnetic / ref.time),
        (f"equilibrium_n.{ion_id}", "1/(m^3*s)", density_rate),
        (f"equilibrium_n.{electron_id}", "1/(m^3*s)", density_rate),
        (f"equilibrium_mom.{ion_id}", "kg/(m^2*s^2)", momentum_rate),
        (
            f"equilibrium_mom.{electron_id}",
            "kg/(m^2*s^2)",
            momentum_rate,
        ),
        (f"equilibrium_perp.{ion_id}", "Pa/s", pressure_rate),
        (f"equilibrium_perp.{electron_id}", "Pa/s", pressure_rate),
        (f"equilibrium_parallel.{electron_id}", "Pa/s", pressure_rate),
        ("equilibrium_B", "T/s", ref.magnetic / ref.time),
        ("theta_consistency", "1/s", 1.0 / ref.time),
    ):
        array(name, unit, reference, name != "theta_consistency")

    power = ref.pressure * ref.velocity * length**2
    energy = ref.pressure * length**3
    magnetic_energy = ref.magnetic**2 * length**3 / p.mu0
    total_energy = energy + magnetic_energy
    force = momentum_rate * length**3
    for name, unit, reference in (
        (f"electron_pressure_adjoint.{electron_id}", "W", power),
        (f"electron_electric_pairing.{electron_id}", "W", power),
        (f"electron_magnetic_work.{electron_id}", "W", power),
        (f"pressure_pairing.{electron_id}", "W", power),
        (f"electric_pairing.{electron_id}", "W", power),
        (f"electron_pressure_only_pairing.{electron_id}", "W", power),
        (
            "wall_field_reaction_work",
            "W",
            current * ref.electric * length**3,
        ),
        (
            "wall_field_reaction_work_ion",
            "W",
            current * ref.electric * length**3,
        ),
        (
            "wall_field_reaction_work_electron",
            "W",
            current * ref.electric * length**3,
        ),
        (
            "wall_field_reaction_work_current",
            "W",
            current * ref.electric * length**3,
        ),
        (f"kinetic_energy.{ion_id}", "J", energy),
        ("ion_internal", "J", energy),
        ("electron_perp", "J", energy),
        ("electron_parallel", "J", energy),
        ("field_energy_B_interval", "J", magnetic_energy),
        ("field_exchange_interval", "W", total_energy / ref.time),
        ("total_energy_interval", "J", total_energy),
        ("total_charge_balance_interval", "C", ref.charge * length**3),
        (f"total_number_balance.{ion_id}", "1", ref.ni * length**3),
        (f"total_number_balance.{electron_id}", "1", ref.ne * length**3),
        ("momentum_balance_interval_x", "kg*m/s", force * ref.time),
        ("momentum_balance_interval_y", "kg*m/s", force * ref.time),
        ("momentum_balance_interval_z", "kg*m/s", force * ref.time),
        (
            "angular_momentum_interval",
            "kg*m^2/s",
            ion.mass * ref.ni * ref.velocity * length**4,
        ),
    ):
        scalar(name, unit, reference)
    return budgets


def _parameters(*, inertia, integrator="midpoint", nsteps=1, hard=False):
    p = foundation_parameters(0)
    p.electron_inertia = inertia
    p.integrator = integrator
    p.dt_guard = "warn"
    p.nsteps = nsteps
    p.dt = 2.0e-11
    p.budgets = (
        _massless_budgets(p) if inertia == 0 else p.budgets
    )
    if not hard:
        for budget in p.budgets.values():
            budget.hard = False
    return p


def _context(*, inertia, integrator="midpoint", nsteps=1, hard=False):
    p = _parameters(
        inertia=inertia,
        integrator=integrator,
        nsteps=nsteps,
        hard=hard,
    )
    backend = Backend("cpu")
    geometry = build_rectangle(6, 6, 0.5, -0.5, 0.2, 0.2)
    operators = build_operators(geometry, 0, backend=backend)
    return Context(
        p=p,
        geometry=geometry,
        op=operators,
        compute=backend,
        N=p.Nphi,
        next_generation=2,
    )


def _wall_extraction(context):
    nc = context.op.Nc
    nr = len(context.geometry.r_faces.keys)
    result = np.zeros(
        (len(context.geometry.walls), context.op.Ne),
        dtype=np.complex128,
    )
    for row, wall in enumerate(context.geometry.walls):
        edge = (
            wall.face
            if wall.component == "r_face"
            else nr + nc + wall.face
        )
        result[row, edge] = wall.sign
    return result


def _compatible_magnetic_field(context, *, seed=7):
    rng = np.random.default_rng(seed)
    result = np.zeros_like(np.asarray(context.op.Cdag[0]), dtype=np.float64)
    del result
    field = np.zeros(context.op.Nb, dtype=np.complex128)
    constraints = np.concatenate(
        (
            _wall_extraction(context) @ np.asarray(context.op.Cdag[0]),
            np.asarray(context.op.DB[0]),
        ),
        axis=0,
    )
    _, singular, vh = np.linalg.svd(constraints, full_matrices=True)
    null_count = int(
        np.count_nonzero(singular <= context.p.rank_rtol * singular[0])
    )
    if null_count == 0:
        raise AssertionError("fixture has no wall-compatible magnetic mode")
    basis = vh[-null_count:].T.conj()
    coefficients = rng.standard_normal(null_count)
    coefficients[int(np.argmax(np.linalg.norm(
        np.asarray(context.op.Cdag[0]) @ basis, axis=0
    )))] += 1.0
    field[:] = basis @ coefficients
    field *= context.p.ref.magnetic / np.max(np.abs(field))
    return context.compute.array(field)


def _state(context, *, compatible=True):
    nc = context.op.Nc
    state = SimpleNamespace(
        fluid=(
            SimpleNamespace(
                n=np.zeros(nc, dtype=np.complex128),
                mom=np.zeros(3 * nc, dtype=np.complex128),
                perp=np.zeros(nc, dtype=np.complex128),
                parallel=np.zeros(0, dtype=np.complex128),
            ),
            SimpleNamespace(
                n=np.zeros(nc, dtype=np.complex128),
                mom=np.zeros(3 * nc, dtype=np.complex128),
                perp=np.zeros(nc, dtype=np.complex128),
                parallel=np.zeros(nc, dtype=np.complex128),
            ),
        ),
        B=(
            _compatible_magnetic_field(context)
            if compatible
            else context.compute.array(
                np.random.default_rng(31).standard_normal(context.op.Nb)
                * context.p.ref.magnetic
            )
        ),
        step=0,
        generation=1,
        time=context.p.start_time,
    )
    state.fluid[0].n[:] = context.p.ref.ni
    state.fluid[1].n[:] = -(
        context.p.species[0].charge / context.p.species[1].charge
    ) * context.p.ref.ni
    state.fluid[0].perp[:] = context.p.ref.pressure
    state.fluid[1].perp[:] = context.p.ref.pressure
    state.fluid[1].parallel[:] = context.p.ref.pressure
    return state


def _cumulative():
    names = (
        "electric_node",
        "electric_edge",
        "pressure_kinetic",
        "pressure_internal",
        "magnetic",
        "wall",
        "convection",
        "perp_rate",
        "parallel_rate",
        "spectral_kinetic",
        "spectral_perp",
        "spectral_parallel",
    )
    return SimpleNamespace(
        baseline=None,
        integrals=[
            SimpleNamespace(**dict.fromkeys(names, 0.0)) for _ in range(2)
        ],
        boundary_integrals=SimpleNamespace(
            number=[0.0, 0.0],
            kinetic=[0.0, 0.0],
            internal_perp=[0.0, 0.0],
            internal_parallel=[0.0, 0.0],
            field_energy=0.0,
            total_energy=0.0,
        ),
        accepted_intervals=0,
        last_trial=0,
        last_step=0,
        last_time=0.0,
    )


def _copy_state(comitted):
    state = comitted.state
    return {
        "step": state.step,
        "generation": state.generation,
        "time": state.time,
        "B": np.asarray(state.B).copy(),
        "ion_n": np.asarray(state.fluid[0].n).copy(),
        "electron_n": np.asarray(state.fluid[1].n).copy(),
        "electron_mom": np.asarray(state.fluid[1].mom).copy(),
    }


def test_nonzero_compatible_current_fixture_and_untrusted_compatibility_momentum():
    from fomhd.massless import canonicalize_state, evaluate_stage
    from fomhd.types import StageTag

    context = _context(inertia=0)
    initial = _state(context)
    initial.fluid[1].n.fill(-3.0 * context.p.ref.ne)
    initial.fluid[1].mom.fill(7.0)
    canonical = canonicalize_state(initial, context)
    tag = StageTag(0, 1, 1, 0, 0.0, "routing_test")
    first = evaluate_stage(canonical, tag, context)
    corrupted = copy.deepcopy(canonical)
    corrupted.fluid[1].mom.fill(1.0e8)
    second = evaluate_stage(corrupted, tag, context)

    np.testing.assert_array_equal(first.rhs.fluid[1].mom, 0.0)
    np.testing.assert_allclose(
        first.rhs.fluid[0].n,
        second.rhs.fluid[0].n,
        rtol=2.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        first.cache.E,
        second.cache.E,
        rtol=2.0e-12,
        atol=0.0,
    )
    current_norm = float(np.max(np.abs(first.current)))
    current_term = float(
        np.max(np.abs(np.asarray(context.op.Cdag[0])))
        * np.max(np.abs(np.asarray(canonical.B)))
        / context.p.mu0
    )
    roundoff = 8192.0 * np.finfo(np.float64).eps * current_term
    assert current_norm > 1.0e4 * roundoff
    assert current_norm > 1.0e-6 * current_term
    curl = mode_apply(context.op.C, first.cache.E, context.compute)
    curl_norm = float(np.max(np.abs(curl)))
    curl_term = float(
        np.max(np.abs(np.asarray(context.op.C[0])))
        * np.max(np.abs(np.asarray(first.cache.E)))
    )
    assert curl_norm > 1.0e4 * 8192.0 * np.finfo(np.float64).eps * curl_term
    np.testing.assert_allclose(
        first.rhs.fluid[1].n,
        context.p.species[0].charge
        / (-context.p.species[1].charge)
        * first.rhs.fluid[0].n,
        rtol=2.0e-12,
        atol=0.0,
    )
    assert first.totals.kinetic[1] == 0.0
    assert first.totals.longitudinal == 0.0
    assert (
        first.totals.applicability.electron_kinetic_energy
        == "not_applicable"
    )
    assert float(np.max(np.abs(
        mode_apply(context.op.DJ, first.current, context.compute)
    ))) <= 1.0e-9 * max(current_norm, 1.0e-300)


@pytest.mark.parametrize("inertia", [1, 0], ids=["finite", "massless"])
@pytest.mark.parametrize("integrator", ["midpoint", "rk4"])
def test_actual_cpu_midpoint_and_rk4_commit_once(inertia, integrator):
    context = _context(
        inertia=inertia, integrator=integrator, nsteps=1
    )
    state = (
        closed_reference(
            context,
            context.p.ref.ni,
            context.p.ref.pressure,
            context.p.ref.pressure,
            context.p.ref.magnetic,
        )
        if inertia
        else _state(context)
    )
    cumulative = _cumulative()
    committed = initialize(state, context, cumulative, False)
    original = committed.state
    result = advance(committed, cumulative, context)

    assert committed.state is not original
    assert committed.state.step == 1
    assert cumulative.accepted_intervals == 1
    assert cumulative.last_step == 1
    assert result.accepted
    assert result.interval.accepted
    assert result.interval.weight == pytest.approx(context.p.dt)
    assert result.totals.energy == pytest.approx(
        result.totals.kinetic[0]
        + result.totals.ion_internal
        + result.totals.electron_perp
        + result.totals.electron_parallel
        + result.totals.magnetic
    )
    if inertia == 0:
        np.testing.assert_array_equal(
            result.frame.rhs.fluid[1].mom, 0.0
        )
        np.testing.assert_allclose(
            committed.state.fluid[1].n,
            context.p.species[0].charge
            / (-context.p.species[1].charge)
            * committed.state.fluid[0].n,
            rtol=2.0e-12,
            atol=0.0,
        )
        assert result.interval.applicability.finite_charge_gauss == (
            "not_applicable"
        )


def test_guard_error_preserves_commit_and_warn_publishes_report():
    context = _context(inertia=0, nsteps=1)
    context.p.dt_guard = "error"
    context.p.minimum_samples_per_cycle = 1.0e12
    state = _state(context)
    cumulative = _cumulative()
    committed = initialize(state, context, cumulative, False)
    old_state = _copy_state(committed)
    old_cumulative = copy.deepcopy(cumulative)

    with pytest.raises(Exception) as error:
        advance(committed, cumulative, context)
    assert getattr(error.value, "code", "") == "TimeStepGuard"
    current_state = _copy_state(committed)
    for name, value in old_state.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(current_state[name], value)
        else:
            assert current_state[name] == value
    assert committed.state.step == 0
    assert cumulative.accepted_intervals == 0
    assert vars(cumulative)["last_trial"] == old_cumulative.last_trial
    assert error.value.time_step_report["action"] == "error"

    warned = _context(inertia=0, nsteps=1)
    warned.p.minimum_samples_per_cycle = 1.0e12
    warned.p.dt_guard = "warn"
    warned_cumulative = _cumulative()
    warned_committed = initialize(
        _state(warned), warned, warned_cumulative, False
    )
    result = advance(warned_committed, warned_cumulative, warned)
    assert result.accepted
    assert result.time_step_report["action"] == "warn"
    json.dumps(io._json_value(result.time_step_report), allow_nan=False)


def test_massless_source_hook_is_rejected_without_commit():
    context = _context(inertia=0, nsteps=1)
    cumulative = _cumulative()
    committed = initialize(_state(context), context, cumulative, False)
    state = committed.state

    def source(_state, _tag):
        return SimpleNamespace(
            fluid=tuple(
                SimpleNamespace(
                    n=np.ones_like(fluid.n),
                    mom=np.ones_like(fluid.mom),
                    perp=np.ones_like(fluid.perp),
                    parallel=np.ones_like(fluid.parallel),
                )
                for fluid in state.fluid
            ),
            B=np.ones_like(state.B),
        )

    with pytest.raises(Exception) as error:
        advance(committed, cumulative, context, source)
    assert getattr(error.value, "code", "") == "ManufacturedSource"
    assert committed.state.step == 0
    assert cumulative.accepted_intervals == 0


def _toml_value(value):
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return str(value).lower()
    return repr(value)


def _write_case(
    root,
    p,
    geometry,
    state,
    integrator,
    nsteps,
    *,
    dt_guard="warn",
    minimum_samples_per_cycle=32.0,
):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / "initial.npz", **io.encode_state(state, geometry))
    polygon = np.asarray(
        [[point.R, point.Z] for point in geometry.polygon_input],
        dtype=np.float64,
    )
    np.savez(
        root / "geometry.npz",
        R_nodes=np.asarray(geometry.R, dtype=np.float64),
        Z_nodes=np.asarray(geometry.Z, dtype=np.float64),
        lcfs_R=polygon[:, 0],
        lcfs_Z=polygon[:, 1],
    )
    lines = [
        "schema_version=1",
        'model="A0"',
        'units="SI"',
        "layout_version=1",
        'case_name="routing_fixture"',
        "[species]",
        f'ion_id="{p.species[0].id}"',
        f'electron_id="{p.species[1].id}"',
        f"qi={p.species[0].charge!r}",
        f"mi={p.species[0].mass!r}",
        f"qe={p.species[1].charge!r}",
        f"me={p.species[1].mass!r}",
        f"gamma_i={p.species[0].gamma!r}",
        f"electron_inertia={p.electron_inertia}",
        "[grid]",
        f"R0={float(geometry.R[0])!r}",
        f"Z0={float(geometry.Z[0])!r}",
        f"hR={float(geometry.R[1] - geometry.R[0])!r}",
        f"hZ={float(geometry.Z[1] - geometry.Z[0])!r}",
        f"NR={int(geometry.R.size)}",
        f"NZ={int(geometry.Z.size)}",
        f"K={p.K}",
        f"Nphi={p.Nphi}",
        "[time]",
        f"start_time={p.start_time!r}",
        f"dt={p.dt!r}",
        f"nsteps={nsteps}",
        'initial_kind="non_equilibrium"',
        f'integrator="{integrator}"',
        f'dt_guard="{dt_guard}"',
        f"minimum_samples_per_cycle={minimum_samples_per_cycle!r}",
        "max_amplitude_error=0.01",
        "max_phase_error_rad=0.1",
        "[boundary]",
        'kind="fixed_lcfs_static"',
        "potential=0.0",
        'version="staircase_relative_v1"',
        "[solver]",
        'backend="cpu"',
        'precision="fp64"',
        'complex_precision="complex128"',
        f"memory_limit_bytes={p.memory_limit}",
        f"device_memory_limit_bytes={p.device_memory_limit}",
        f"rank_rtol={p.rank_rtol!r}",
        "device_ordinal=0",
        "cpu_threads=1",
        "[solver.constants]",
        f"epsilon0_Fpm={p.epsilon0!r}",
        f"mu0_Hpm={p.mu0!r}",
        'version="routing-fixture"',
        "[solver.references]",
        f"L_m={p.ref.length!r}",
        f"t_s={p.ref.time!r}",
        f"n_i_m3={p.ref.ni!r}",
        f"n_e_m3={p.ref.ne!r}",
        f"u_mps={p.ref.velocity!r}",
        f"E_Vpm={p.ref.electric!r}",
        f"B_T={p.ref.magnetic!r}",
        f"J_Apm2={p.ref.current!r}",
        f"rho_Cpm3={p.ref.charge!r}",
        f"p_Pa={p.ref.pressure!r}",
        "[budgets.domain]",
        f"n_min_i={p.species[0].n_min!r}",
        f"n_min_e={p.species[1].n_min!r}",
        f"n_face_min_i={p.species[0].face_min!r}",
        f"n_face_min_e={p.species[1].face_min!r}",
        f"p_min_i={p.species[0].p_min!r}",
        f"p_min_perp_e={p.species[1].p_min!r}",
        f"p_min_parallel_e={p.species[1].parallel_min!r}",
        f"B_min={p.B_min!r}",
    ]
    for name, budget in p.budgets.items():
        norm = (
            budget.norm.value
            if hasattr(budget.norm, "value")
            else str(budget.norm)
        )
        if norm == "weighted_l2":
            norm = "weighted_L2"
        lines.extend(
            [
                f'[budgets.checks."{name}"]',
                f"atol={budget.atol!r}",
                f"rtol={budget.rtol!r}",
                f"reference={budget.reference!r}",
                f'unit="{budget.unit}"',
                f'norm="{norm}"',
                f'gate="{"hard" if budget.hard else "observe"}"',
            ]
        )
    lines.extend(
        [
            "[diagnostics]",
            f"inner_distance_m={p.inner_distance!r}",
            'alias_policy="initial_midpoint_restart"',
            "timing_repetitions=1",
            "timing_warmup=0",
            "mode_observations=[]",
            "[io]",
            'geometry_file="geometry.npz"',
            'initial_file="initial.npz"',
            'output_dir="unused"',
            "snapshot_stride=1",
            "checkpoint_stride=1",
            "max_input_bytes=16777216",
            "max_uncompressed_bytes=33554432",
            "max_entries=4096",
            "max_header_bytes=1048576",
            'source_manifest_file="source.json"',
            "write_residual_arrays=true",
        ]
    )
    case = root / "case.toml"
    case.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    manifest = {
        "provider_name": "routing_fixture",
        "provider_version": "1",
        "units": "SI",
        "conversion_method": "direct complex128 state encoding",
        "input_hashes": {
            "case": digest(case),
            "geometry": digest(root / "geometry.npz"),
            "initial": digest(root / "initial.npz"),
        },
        "continuous_residual_status": "UNVERIFIED",
    }
    (root / "source.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return case


def _latest_checkpoint(run):
    pointer = json.loads(
        (run / "checkpoints/latest.json").read_text(encoding="utf-8")
    )
    return run / "checkpoints" / pointer["file"]


def _rewrite_npz(source, target, **changes):
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays.update(changes)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **arrays)
    return target


@pytest.mark.parametrize(
    "ion_id,electron_id",
    [
        ("ion", "e"),
        ("D", "electron"),
        ("D", "e"),
    ],
)
def test_massless_cli_uses_configured_species_ids_in_budget_registry(
    tmp_path, ion_id, electron_id
):
    context = _context(inertia=0, nsteps=1, hard=True)
    context.p.species[0].id = ion_id
    context.p.species[1].id = electron_id
    context.p.budgets = _massless_budgets(context.p)
    context.p.dt = 2.0e-13
    context.p.dt_guard = "warn"
    case_path = _write_case(
        tmp_path / f"{ion_id}-{electron_id}",
        context.p,
        context.geometry,
        _state(context),
        "midpoint",
        1,
    )
    output = tmp_path / f"{ion_id}-{electron_id}" / "run"
    assert _run_command(
        ["run", str(case_path), "--output", str(output)]
    ) == 0
    manifest = json.loads(
        (output / "run.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert manifest["last_accepted_step"] == 1
    records = [
        json.loads(line)
        for line in (output / "checks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    populations = {record["population"] for record in records}
    assert {ion_id, electron_id} <= populations
    assert any(
        record["name"] == "alias_n"
        and record["population"] == electron_id
        for record in records
    )


def test_rejected_massless_combine_retains_raw_candidate_without_commit():
    context = _context(inertia=0, nsteps=1, hard=False)
    state = _state(context)
    rng = np.random.default_rng(901)
    momentum = rng.normal(size=state.fluid[0].mom.size).astype(
        np.complex128
    )
    constraint = np.asarray(context.op.T_full)
    momentum -= (
        np.linalg.pinv(constraint) @ (constraint @ momentum)
    )
    momentum *= context.p.species[0].mass * context.p.ref.ni
    state.fluid[0].mom[:] = momentum

    cumulative = _cumulative()
    captured = {}
    committed = initialize(
        state, context, cumulative, False, captured
    )
    rate = captured["rhs"].fluid[0].n.real
    density = committed.state.fluid[0].n.real
    context.p.dt = 4.0 * float(
        np.min(density[rate < 0] / -rate[rate < 0])
    )
    accepted_state = _copy_state(committed)
    accepted_ledger = copy.deepcopy(cumulative)

    with pytest.raises(Failure) as error:
        advance(committed, cumulative, context)

    failure = error.value
    assert failure.code == "Domain"
    assert failure.stage.stage == "midpoint"
    assert failure.stage.step == 0
    assert failure.stage.trial == context.next_trial - 1
    assert failure.stage.time == pytest.approx(context.p.dt / 2.0)
    for suffix in (
        "B",
        "n_i",
        "momentum_i",
        "p_i",
        "n_e",
        "momentum_e",
        "p_e",
        "p_parallel_e",
    ):
        assert f"candidate_{suffix}" in failure.partial_arrays
    candidate_density = np.asarray(
        failure.partial_arrays["candidate_n_i"]
    )
    assert np.any(candidate_density.real <= 0.0)
    current_state = _copy_state(committed)
    for name, value in accepted_state.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(current_state[name], value)
        else:
            assert current_state[name] == value
    assert cumulative.accepted_intervals == (
        accepted_ledger.accepted_intervals
    )
    assert cumulative.last_step == accepted_ledger.last_step
    assert cumulative.last_time == accepted_ledger.last_time
    assert cumulative.last_trial == accepted_ledger.last_trial


def test_failure_evidence_uses_common_helper_without_public_dispatch_recursion(
    monkeypatch,
):
    import fomhd.stage as stage

    fake = ModuleType("fomhd.massless")

    def retain(failure, tag, context, state, result, prefix):
        return stage._retain_common_failure_evidence(
            failure, tag, context, state, result, prefix
        )

    fake.retain_failure_evidence = retain
    fake.stage_requirements = lambda context, tag: []
    monkeypatch.setitem(sys.modules, "fomhd.massless", fake)
    monkeypatch.setattr(
        stage._diagnostics(), "stage_requirements", lambda context, tag: []
    )
    failure = Failure("Injected", "routing seam")
    tag = StageTag(0, 1, 1, 0, 0.0, "test")
    context = SimpleNamespace(
        p=SimpleNamespace(electron_inertia=0),
    )
    stage.retain_failure_evidence(
        failure, tag, context, None, None, "routing"
    )
    assert failure.stage == tag
    assert failure.checks == []


def test_massless_diagnostics_preserve_returned_ledgers(monkeypatch):
    import fomhd.diagnostics as diagnostics

    fake = ModuleType("fomhd.massless_checks")

    def duplicate(*args, **kwargs):
        raise AssertionError("cached massless ledger was recomputed")

    fake.state_totals = duplicate
    fake.boundary_ledger = duplicate
    fake.momentum = duplicate
    monkeypatch.setitem(sys.modules, "fomhd.massless_checks", fake)
    totals = object()
    boundary = object()
    momentum = object()
    result = SimpleNamespace(
        totals=totals,
        boundary=boundary,
        momentum=momentum,
    )
    context = SimpleNamespace(
        p=SimpleNamespace(electron_inertia=0),
    )
    assert diagnostics.state_totals(None, result, context) is totals
    assert diagnostics.boundary_ledger(None, result, context) is boundary
    assert diagnostics.momentum(None, result, context) is momentum


def test_massless_stage_result_and_wall_correction_are_returned_unchanged(
    monkeypatch,
):
    import fomhd.stage as stage

    wall_correction = object()
    boundary = object()
    momentum = object()
    result = SimpleNamespace(
        cache=SimpleNamespace(wall_correction=wall_correction),
        boundary=boundary,
        momentum=momentum,
    )
    committed = object()
    fake = ModuleType("fomhd.massless")
    fake.evaluate_stage = lambda *args, **kwargs: result
    fake.initialize = lambda *args, **kwargs: committed
    monkeypatch.setitem(sys.modules, "fomhd.massless", fake)
    context = SimpleNamespace(
        p=SimpleNamespace(electron_inertia=0),
    )
    assert stage.evaluate_stage(None, None, context) is result
    assert stage.initialize(None, context, None, False) is committed
    assert result.cache.wall_correction is wall_correction
    assert result.boundary is boundary
    assert result.momentum is momentum


def test_finite_guard_error_persists_full_report_without_commit(tmp_path):
    context = _context(inertia=1, nsteps=1, hard=True)
    state = closed_reference(
        context,
        context.p.ref.ni,
        context.p.ref.pressure,
        context.p.ref.pressure,
        context.p.ref.magnetic,
    )
    case_path = _write_case(
        tmp_path / "input",
        context.p,
        context.geometry,
        state,
        "midpoint",
        1,
        dt_guard="error",
        minimum_samples_per_cycle=1.0e12,
    )
    output = tmp_path / "run"
    assert _run_command(
        ["run", str(case_path), "--output", str(output)]
    ) == 4

    manifest = json.loads(
        (output / "run.json").read_text(encoding="utf-8")
    )
    assert manifest["last_accepted_step"] == 0
    pointer = json.loads(
        (output / "checkpoints/latest.json").read_text(encoding="utf-8")
    )
    assert pointer["step"] == 0
    failure_path = next((output / "failure").glob("failure_*.json"))
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["code"] == "TimeStepGuard"
    assert failure["runtime_phase"] == "numerical_trial"
    assert failure["accepted_trial"] is False
    report = failure["time_step_report"]
    assert report["action"] == "error"
    assert report["step"] == 0
    assert report["reasons"]
    assert report["estimate"]["electron_inertia"] == 1
    json.dumps(report, allow_nan=False)


def test_cli_massless_init_run_restarts_guard_report_and_predecode_model_rejection(
    tmp_path, monkeypatch
):
    root = tmp_path / "massless"
    context = _context(inertia=0, nsteps=1)
    context.p.budgets = {
        name: Budget(
            atol=budget.atol,
            rtol=1.0e-5,
            reference=budget.reference,
            unit=budget.unit,
            norm=budget.norm,
            hard=budget.hard,
        )
        for name, budget in _massless_budgets(context.p).items()
    }
    state = _state(context)
    state.fluid[1].n.fill(123.0)
    state.fluid[1].mom.fill(-456.0)
    case_path = _write_case(
        root, context.p, context.geometry, state, "midpoint", 1
    )

    output1 = root / "run1"
    assert _run_command(
        ["run", str(case_path), "--output", str(output1)]
    ) == 0
    checkpoint1 = _latest_checkpoint(output1)
    with np.load(checkpoint1, allow_pickle=False) as archive:
        np.testing.assert_allclose(
            archive["n_e"],
            context.p.species[0].charge
            / (-context.p.species[1].charge)
            * archive["n_i"],
            rtol=2.0e-12,
            atol=0.0,
        )
        assert not np.allclose(archive["mom_e"], -456.0)
    manifest1 = json.loads(
        (output1 / "run.json").read_text(encoding="utf-8")
    )
    assert manifest1["model_variant"] == MASSLESS_MODEL
    assert (
        manifest1["model_applicability"]["electron_kinetic_energy"]
        == "not_applicable"
    )
    records = [
        json.loads(line)
        for line in (output1 / "checks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    for unavailable in (
        "electron_kinetic_energy",
        "longitudinal_electric_energy",
        "finite_charge_gauss",
    ):
        matches = [
            record for record in records if record["name"] == unavailable
        ]
        assert matches
        assert all(record["status"] == "UNAVAILABLE" for record in matches)
    assert not any(
        record["name"] in ("gauss", "poisson_linear")
        and record["status"] == "PASS"
        for record in records
    )
    estimates = [
        json.loads(line)
        for line in (output1 / "time_estimates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert estimates and estimates[0]["action"] == "warn"

    guard_case = _write_case(
        root / "guard-error",
        context.p,
        context.geometry,
        state,
        "midpoint",
        1,
        dt_guard="error",
        minimum_samples_per_cycle=1.0e12,
    )
    guard_output = root / "guard-error-run"
    assert _run_command(
        ["run", str(guard_case), "--output", str(guard_output)]
    ) == 4
    guard_manifest = json.loads(
        (guard_output / "run.json").read_text(encoding="utf-8")
    )
    assert guard_manifest["last_accepted_step"] == 0
    guard_failure_path = next(
        (guard_output / "failure").glob("failure_*.json")
    )
    guard_failure = json.loads(
        guard_failure_path.read_text(encoding="utf-8")
    )
    assert guard_failure["code"] == "TimeStepGuard"
    assert guard_failure["runtime_phase"] == "numerical_trial"
    assert guard_failure["time_step_report"]["action"] == "error"

    case_path = _write_case(
        root, context.p, context.geometry, state, "midpoint", 2
    )
    output2 = root / "run2"
    assert _run_command(
        [
            "run",
            str(case_path),
            "--output",
            str(output2),
            "--restart",
            str(checkpoint1),
        ]
    ) == 0
    checkpoint2 = _latest_checkpoint(output2)

    case_path = _write_case(
        root, context.p, context.geometry, state, "midpoint", 3
    )
    output3 = root / "run3"
    assert _run_command(
        [
            "run",
            str(case_path),
            "--output",
            str(output3),
            "--restart",
            str(checkpoint2),
        ]
    ) == 0
    checkpoint3 = _latest_checkpoint(output3)
    loaded = io.load_case(case_path, {"output": str(root / "unused")})
    saved = io.read_restart(checkpoint3, loaded)
    assert saved.state.step == 3
    assert saved.cumulative.baseline.step == 0
    assert saved.cumulative.accepted_intervals == 3
    np.testing.assert_allclose(
        saved.state.fluid[1].n,
        loaded.parameters.species[0].charge
        / (-loaded.parameters.species[1].charge)
        * saved.state.fluid[0].n,
        rtol=2.0e-12,
        atol=0.0,
    )
    with np.load(checkpoint3, allow_pickle=False) as archive3:
        corrupt_density = archive3["n_e"].copy()
        corrupt_momentum = archive3["mom_e"].copy()
        for i, j in loaded.geometry.cells.keys:
            corrupt_density[:, j, i] = 777.0
            corrupt_momentum[:, :, j, i] = -999.0
        corrupt_changes = {
            "n_e": corrupt_density,
            "mom_e": corrupt_momentum,
        }
    corrupted_derived = _rewrite_npz(
        checkpoint3,
        root / "corrupt-derived" / checkpoint3.name,
        **corrupt_changes,
    )
    restored = io.read_restart(corrupted_derived, loaded)
    np.testing.assert_allclose(
        restored.state.fluid[1].n,
        loaded.parameters.species[0].charge
        / (-loaded.parameters.species[1].charge)
        * restored.state.fluid[0].n,
        rtol=2.0e-12,
        atol=0.0,
    )
    encoded = io.encode_state(restored.state, loaded.geometry)
    with np.load(checkpoint3, allow_pickle=False) as archive3:
        for name in (
            "n_i",
            "mom_i",
            "p_i",
            "p_perp_e",
            "p_parallel_e",
            "B_R",
            "B_phi",
            "B_Z",
        ):
            np.testing.assert_array_equal(encoded[name], archive3[name])
        assert not np.allclose(encoded["mom_e"], -999.0)

    tampered = _rewrite_npz(
        checkpoint3,
        root / "tampered" / checkpoint3.name,
        physics_model=np.asarray(FINITE_MODEL),
    )
    tampered_applicability = _rewrite_npz(
        checkpoint3,
        root / "tampered-applicability" / checkpoint3.name,
        model_applicability=np.asarray(
            json.dumps(
                {
                    "model": FINITE_MODEL,
                    "electron_inertia": 1,
                    "electron_kinetic_energy": "applicable",
                    "longitudinal_electric_energy": "applicable",
                    "finite_charge_gauss": "applicable",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
    )
    with np.load(checkpoint3, allow_pickle=False) as archive:
        cumulative_arrays = {
            key: archive[key] for key in archive.files
        }
    cumulative_record_value = json.loads(
        str(cumulative_arrays["cumulative_json"].item())
    )
    cumulative_record_value["applicability"] = {
        "model": FINITE_MODEL,
        "electron_inertia": 1,
    }
    tampered_cumulative = _rewrite_npz(
        checkpoint3,
        root / "tampered-cumulative" / checkpoint3.name,
        cumulative_json=np.asarray(
            json.dumps(
                cumulative_record_value,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
    )

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("state decode ran before model metadata rejection")

    monkeypatch.setattr(io, "decode_state", forbidden_decode)
    with pytest.raises(ValueError, match="model/applicability mismatch"):
        io.read_restart(tampered, loaded)
    with pytest.raises(ValueError, match="model/applicability mismatch"):
        io.read_restart(tampered_applicability, loaded)
    with pytest.raises(ValueError, match="cumulative applicability mismatch"):
        io.read_restart(tampered_cumulative, loaded)
    monkeypatch.undo()

    finite_p = foundation_parameters(0)
    finite_p.dt = context.p.dt
    finite_p.integrator = "midpoint"
    finite_p.dt_guard = "warn"
    finite_geometry = build_rectangle(6, 6, 0.5, -0.5, 0.2, 0.2)
    finite_backend = Backend("cpu")
    finite_operators = build_operators(
        finite_geometry, 0, backend=finite_backend
    )
    finite_context = Context(
        finite_p,
        finite_geometry,
        finite_operators,
        finite_backend,
        finite_p.Nphi,
    )
    finite_state = closed_reference(
        finite_context,
        finite_p.ref.ni,
        finite_p.ref.pressure,
        finite_p.ref.pressure,
        finite_p.ref.magnetic,
    )
    finite_case_path = _write_case(
        root / "finite",
        finite_p,
        finite_geometry,
        finite_state,
        "midpoint",
        1,
    )
    finite_case = io.load_case(
        finite_case_path, {"output": str(root / "finite-unused")}
    )
    finite_runtime_context = command_context(finite_case)
    finite_cumulative = _cumulative()
    finite_committed = initialize(
        finite_state,
        finite_runtime_context,
        finite_cumulative,
        False,
    )
    finite_run = io.Run(finite_case)
    try:
        finite_run.checkpoint(
            finite_committed,
            finite_cumulative,
            finite_runtime_context,
        )
        finite_checkpoint = _latest_checkpoint(finite_run.path())
    finally:
        finite_run.close()

    monkeypatch.setattr(io, "decode_state", forbidden_decode)
    with pytest.raises(ValueError, match="model/applicability mismatch"):
        io.read_restart(checkpoint3, finite_case)
    with pytest.raises(ValueError, match="model/applicability mismatch"):
        io.read_restart(finite_checkpoint, loaded)


def test_snapshot_serializes_unavailable_phi_without_null_array(tmp_path):
    context = _context(inertia=0, nsteps=0, hard=False)
    state = _state(context)
    cumulative = _cumulative()
    captured = {}
    committed = initialize(state, context, cumulative, False, captured)
    frame = SimpleNamespace(**captured)
    context.p.budgets = _massless_budgets(context.p)
    case_path = _write_case(
        tmp_path / "input",
        context.p,
        context.geometry,
        state,
        "midpoint",
        0,
    )
    case = io.load_case(
        case_path, {"output": str(tmp_path / "snapshot-run")}
    )
    committed.context = command_context(case)
    run = io.Run(case)
    try:
        run.snapshot(committed, frame)
        path = run.path() / "snapshots/state_0000000000.npz"
        with np.load(path, allow_pickle=False) as archive:
            assert not archive["derived_phi_available"].item()
            assert "not_applicable" in str(
                archive["derived_phi_reason"].item()
            )
            assert "derived_phi" not in archive.files
            assert archive["physics_model"].item() == MASSLESS_MODEL
    finally:
        run.close()
