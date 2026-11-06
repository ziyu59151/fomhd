import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fomhd.backend import Backend
from fomhd.core import (
    allocate_state,
    forward,
    inverse,
    mode_apply,
    sample_apply,
)
from fomhd.fields import electrostatic, longitudinal, transverse
from fomhd.fluid import (
    _magnetic_force_kernel,
    _warp_magnetic_force,
    build_fluid,
    build_non_electric,
    divergence,
    electric_response,
    gradient_adjoint,
    velocity_gradient,
)
from fomhd.geometry import build_rectangle
from fomhd.stage import closed_reference, evaluate_stage, initialize
from fomhd.types import (
    Budget,
    Context,
    Failure,
    Norm,
    Parameters,
    References,
    Species,
    StageTag,
)
from fomhd.wall import build_wall, project_wall


_FLOAT_EPS = np.finfo(np.float64).eps
_RELATIVE_ROUNDOFF_ULPS = 64.0
_REFERENCE_ROUNDOFF_ULPS = 16.0


def parameters(k=1):
    species = [
        Species("ion", 1.67262192595e-27, 1.602176634e-19, 5.0 / 3.0, 0, 0, 0, 0),
        Species("electron", 9.1093837139e-31, -1.602176634e-19, 0, 0, 0, 0, 0),
    ]
    ref = References(
        length=0.2,
        time=1.0e-6,
        ni=1.0e12,
        ne=1.0e12,
        velocity=1.0,
        electric=1.0,
        magnetic=1.0e-5,
        current=1.0,
        charge=1.0e-7,
        pressure=1.0e-9,
    )
    p = Parameters(
        species=species,
        epsilon0=8.8541878188e-12,
        mu0=1.25663706127e-6,
        B_min=1.0e-12,
        dt=1.0e-9,
        start_time=0.0,
        rank_rtol=1.0e-12,
        nsteps=1,
        K=k,
        Nphi=4 * k + 1,
        memory_limit=512 << 20,
        device_memory_limit=512 << 20,
        inner_distance=0.02,
        ref=ref,
        budgets={},
        electron_inertia=1,
    )
    response = sum(s.charge * s.charge * ref.ni / s.mass for s in species)
    p.ref.current = ref.time * ref.electric * (
        response + 1.0 / (p.mu0 * ref.length * ref.length)
    )

    length = ref.length
    time = ref.time
    density = ref.ni
    pressure = ref.pressure
    electric = ref.electric
    magnetic = ref.magnetic
    force = species[0].mass * density * ref.velocity / time
    power = pressure * length**3 / time
    energy = pressure * length**3 + magnetic**2 * length**3 / p.mu0

    def array(name, unit, scale, hard=True):
        for label, norm in (
            ("linf", Norm.linf),
            ("weighted_L2", Norm.weighted_l2),
        ):
            p.budgets[f"{name}.{label}"] = Budget(
                atol=0.0,
                rtol=1.0e-8,
                reference=scale,
                unit=unit,
                norm=norm,
                hard=hard,
            )

    def scalar(name, unit, scale, hard=True):
        p.budgets[f"{name}.value"] = Budget(
            atol=0.0,
            rtol=1.0e-8,
            reference=scale,
            unit=unit,
            norm=Norm.value,
            hard=hard,
        )

    for name in ("gauss", "poisson_linear", "transverse_divergence"):
        array(name, "V/m^2", electric / length)
    array("longitudinal_projection", "A/m^3", p.ref.current / length)
    array("darwin_dynamic", "A/(m^2*s)", p.ref.current / time)
    for name in ("current_divB", "current_ampere"):
        array(name, "T/m", magnetic / length, False)
    for name in (
        "initial_divB",
        "initial_ampere",
        "divB_update",
        "long_term_divB_change",
    ):
        array(name, "T/m", magnetic / length)
    array("CT_increment", "T/(m*s)", magnetic / (length * time))
    for name in ("CT_update", "equilibrium_B", "alias_B"):
        array(name, "T/s", magnetic / time)
    array("current_response", "A/(m^2*s)", p.ref.current / time)
    for name in ("RC_flux", "RC_mom", "deltaC"):
        array(name, "C/(m^3*s)", p.ref.charge / time)
    array("RL", "A/m^2", p.ref.current)
    for name in (
        "ion_internal",
        "electron_perp",
        "electron_parallel",
        "field_energy_B",
        "field_energy_L",
        "total_energy",
    ):
        scalar(name, "J", energy, False)
    scalar("field_exchange", "W", energy / time, False)
    scalar("topology_increment", "Wb", magnetic * length**2)
    scalar("total_charge_balance", "C", p.ref.charge * length**3)
    for name in ("momentum_x", "momentum_y", "momentum_z"):
        scalar(
            name,
            "kg*m/s",
            species[0].mass * density * length**3 * ref.velocity,
            False,
        )
    scalar(
        "angular_momentum",
        "kg*m^2/s",
        species[0].mass * density * length**4 * ref.velocity,
        False,
    )
    for i, sp in enumerate(species):
        ident = f".{sp.id}"
        for name in (
            "stage_continuity",
            "species_continuity",
            "alias_n",
            "equilibrium_n",
        ):
            array(name + ident, "1/(m^3*s)", density / time)
        for name in (
            "wall_rhs",
            "wall_multiplier",
            "alias_mom",
            "equilibrium_mom",
        ):
            array(name + ident, "kg/(m^2*s^2)", force)
        array("wall_state" + ident, "kg/(m^2*s)", force * time)
        array("theta_identity" + ident, "1/s", 1.0 / time)
        array("wall_number_flux" + ident, "1/(m^2*s)", density * ref.velocity)
        array("wall_perp_flux" + ident, "Pa*m/s", pressure * ref.velocity)
        if i:
            array("wall_parallel_flux" + ident, "Pa*m/s", pressure * ref.velocity)
        for name in ("alias_perp", "equilibrium_perp"):
            array(name + ident, "Pa/s", pressure / time)
        if i:
            for name in ("cgl_source", "alias_parallel", "equilibrium_parallel"):
                array(name + ident, "Pa/s", pressure / time)
        for name in (
            "electric_pairing",
            "pressure_pairing",
            "magnetic_work",
            "wall_work",
            "convection_pairing",
        ):
            scalar(name + ident, "W", power)
        scalar("spectral_transfer_defect" + ident, "W", power, False)
        scalar("total_number_balance" + ident, "1", density * length**3)
        scalar("kinetic_energy" + ident, "J", energy, False)
    return p


def _weighted_dot(xp, left, right, weights):
    return xp.sum(weights * xp.conj(left) * right)


from workflow_fixture import ORACLE_ROOT

ORACLE = ORACLE_ROOT / "native-oracle-4-k1.npz"
STRESS_ORACLE = ORACLE_ROOT / "native-oracle-stress.npz"


def _oracle_fixture(oracle_path=ORACLE):
    with np.load(oracle_path, allow_pickle=False) as source:
        data = {key: source[key].copy() for key in source.files}
    metadata = json.loads(str(data["metadata_json"].item()))
    points = int(metadata["points"])
    p = parameters(int(metadata["K"]))
    p.dt = float(metadata["dt"])
    p.Nphi = int(metadata["Nphi"])
    backend = Backend("cpu")
    geometry = build_rectangle(points, points, 1.0, 0.0, 0.2, 0.2)
    vertices = geometry.vertices
    segments = [
        (
            np.array([vertices.R[wall.vertices[0]], vertices.Z[wall.vertices[0]]]),
            np.array([vertices.R[wall.vertices[1]], vertices.Z[wall.vertices[1]]]),
        )
        for wall in geometry.walls
    ]
    for table in (
        geometry.cells,
        geometry.r_faces,
        geometry.z_faces,
        geometry.vertices,
    ):
        distances = []
        for radius, axial in zip(table.R, table.Z):
            point = np.array([radius, axial])
            distances.append(
                min(
                    np.linalg.norm(
                        point
                        - (start + np.clip(np.dot(point - start, end - start) / np.dot(end - start, end - start), 0.0, 1.0) * (end - start))
                    )
                    for start, end in segments
                )
            )
        table.wall_distance = np.asarray(distances, dtype=np.float64)
    context = Context(p, geometry, SimpleNamespace(), backend, p.Nphi)
    op = SimpleNamespace(
        K=p.K,
        Nc=len(geometry.cells.keys),
        Ne=data["M1"].size,
        Nb=data["MB"].size,
        M0=data["M0"].copy(),
        M1=data["M1"].copy(),
        MB=data["MB"].copy(),
        I=data["I"].copy(),
        H=data["H"].copy(),
        QB=data["QB"].copy(),
        DR=data["DR"].copy(),
        DZ=data["DZ"].copy(),
        T=data["T"].copy(),
        T_full=data["T_full"].copy(),
        Bc=data["Bc"].copy(),
        Bc_full=data["Bc_full"].copy(),
        topology_flux_row=data["topology_flux_row"].copy(),
        G=[data[f"G_{i}"].copy() for i in range(2 * p.K + 1)],
        C=[data[f"C_{i}"].copy() for i in range(2 * p.K + 1)],
        DB=[data[f"DB_{i}"].copy() for i in range(2 * p.K + 1)],
        DJ=[data[f"DJ_{i}"].copy() for i in range(2 * p.K + 1)],
        Cdag=[data[f"Cdag_{i}"].copy() for i in range(2 * p.K + 1)],
        Lphi=[data[f"Lphi_{i}"].copy() for i in range(2 * p.K + 1)],
        PR_r=data["PR_r"].copy(),
        PR_z=data["PR_z"].copy(),
        PR_v=data["PR_v"].copy(),
        PZ_r=data["PZ_r"].copy(),
        PZ_z=data["PZ_z"].copy(),
        PZ_v=data["PZ_v"].copy(),
        setup_checks=[],
    )
    context.op = op
    state = allocate_state(context)
    state.fluid[0].n = data["state_0_n"].copy()
    state.fluid[0].mom = data["state_0_mom"].copy()
    state.fluid[0].perp = data["state_0_perp"].copy()
    state.fluid[0].parallel = data["state_0_parallel"].copy()
    state.fluid[1].n = data["state_1_n"].copy()
    state.fluid[1].mom = data["state_1_mom"].copy()
    state.fluid[1].perp = data["state_1_perp"].copy()
    state.fluid[1].parallel = data["state_1_parallel"].copy()
    state.B = data["state_B"].copy()
    state.step = 0
    state.generation = 1
    state.time = 0.0
    data["_metadata"] = metadata
    return context, op, state, data


def _assert_native_close(
    actual,
    expected,
    name,
    reference=None,
    roundoff_factor=_REFERENCE_ROUNDOFF_ULPS,
):
    """Compare native-unit values without a unitless absolute floor.

    The relative term follows the measured quantity. The reference term is
    only the roundoff floor for the corresponding physical term scale.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape, name
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise AssertionError(f"{name}: nonfinite actual or expected value")
    if expected.size == 0:
        return
    magnitude = np.abs(expected)
    expected_scale = float(np.max(magnitude))
    reference = expected_scale if reference is None else float(reference)
    if not np.isfinite(reference) or reference < 0.0:
        raise AssertionError(f"{name}: invalid reference {reference}")
    allowed = (
        _RELATIVE_ROUNDOFF_ULPS * _FLOAT_EPS * magnitude
        + roundoff_factor * _FLOAT_EPS * reference
    )
    excess = np.abs(actual - expected) - allowed
    if np.any(excess > 0.0):
        worst = int(np.argmax(excess))
        raise AssertionError(
            f"{name}: error={abs(actual.flat[worst] - expected.flat[worst])} "
            f"allowed={allowed.flat[worst]} expected={expected.flat[worst]} "
            f"reference={reference}"
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("pressure", 1.0e-9),
        ("momentum_rate", 1.0e-20),
        ("power", 1.0e-23),
    ],
)
def test_native_comparison_rejects_zeroed_small_nonzero_arrays(name, value):
    expected = np.array([value, 1.5 * value], dtype=np.complex128)
    with pytest.raises(AssertionError):
        _assert_native_close(np.zeros_like(expected), expected, name)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("pressure", 1.0e-9),
        ("momentum_rate", 1.0e-20),
        ("power", 1.0e-23),
    ],
)
def test_native_comparison_rejects_relative_errors_hidden_by_unit_floor(name, value):
    expected = np.array([value, -0.5 * value], dtype=np.complex128)
    actual = 0.8 * expected
    with pytest.raises(AssertionError):
        _assert_native_close(actual, expected, name)


@pytest.mark.parametrize("bad_side", ["actual", "expected"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_native_comparison_rejects_nonfinite_values(bad_side, bad_value):
    expected = np.array([1.0, 2.0], dtype=np.complex128)
    actual = np.array([1.0, 2.0], dtype=np.complex128)
    if bad_side == "actual":
        actual[0] = bad_value
    else:
        expected[0] = bad_value
    with pytest.raises(AssertionError):
        _assert_native_close(actual, expected, "nonfinite")


def _cell_integral(context, values):
    values = np.asarray(values).reshape(context.N, -1, context.op.Nc)
    weights = (
        2.0
        * np.pi
        / context.N
        * np.real(np.asarray(context.op.M0))
    )
    return np.sum(values * weights[None, None, :])


def _fJ0_term_reference(context, result):
    reference = 0.0
    for species in range(2):
        physical = context.p.species[species]
        term = forward(
            sample_apply(
                context.op.I,
                result.fluid[species].non_electric
                * physical.charge
                / physical.mass,
                context.N,
                context.compute,
            ),
            context.op.Ne,
            context,
        )
        reference += float(np.max(np.abs(term)))
    return reference


def _zero_reference(context, name):
    ref = context.p.ref
    density_scale = max(ref.ni, ref.ne)
    speed_scale = ref.velocity
    pressure_scale = ref.pressure
    force_scale = pressure_scale / ref.length
    gradient_scale = 1.0 / ref.time
    current_rate = ref.current / ref.time
    magnetic_rate = ref.magnetic / ref.time
    if name.startswith("rhs_"):
        if name.endswith("_mom"):
            return force_scale
        if name.endswith(("_perp", "_parallel")):
            return pressure_scale / ref.time
        if name.endswith("_n"):
            return density_scale / ref.time
        if name.endswith("_B"):
            return magnetic_rate
    if name.endswith("_n") or name == "rho":
        return abs(context.p.species[0].charge) * density_scale
    if name == "current":
        return abs(context.p.species[0].charge) * density_scale * speed_scale
    if name == "phi":
        return ref.electric * ref.length
    if name in ("EL", "ET", "E"):
        return ref.electric
    if name == "ELdot":
        return ref.electric / ref.time
    if name.endswith("_u"):
        return speed_scale
    if "_flux" in name:
        return density_scale * speed_scale
    if name.endswith("_nf"):
        return density_scale
    if name.endswith(("_tensor", "_perp", "_parallel")):
        return pressure_scale
    if name.endswith(("_gradient", "_theta", "_strain")):
        return gradient_scale
    if "_convection" in name:
        return pressure_scale * speed_scale / ref.length
    if name.endswith(("_pressure_force", "_non_electric")):
        return force_scale
    if "_pdot_" in name:
        return pressure_scale / ref.time
    if name.endswith("_mom"):
        return context.p.species[0].mass * density_scale * speed_scale
    if name.startswith("residual_"):
        if name in ("residual_poisson_linear", "residual_gauss"):
            return ref.electric / ref.length
        if "longitudinal_projection" in name:
            return ref.current / ref.length
        if "darwin_dynamic" in name or "current_response" in name:
            return current_rate
        if "ampere" in name or "divB" in name:
            return ref.magnetic / ref.length
        if "CT_" in name:
            return ref.magnetic / (ref.length * ref.time)
        if "continuity" in name:
            return density_scale / ref.time
        if "wall_state" in name:
            return context.p.species[0].mass * density_scale * speed_scale
        if "wall_multiplier" in name or "wall_rhs" in name:
            return force_scale
        if "wall_number_flux" in name:
            return density_scale * speed_scale
        if "wall_perp_flux" in name or "wall_parallel_flux" in name:
            return pressure_scale * speed_scale
        if "theta_identity" in name:
            return gradient_scale
        if "cgl_source" in name:
            return pressure_scale / ref.time
        return current_rate
    if name.startswith("ledger_"):
        return pressure_scale * ref.length**3 / ref.time
    if name.endswith("_B"):
        return ref.magnetic
    return max(1.0e-300, abs(context.p.species[0].charge) * density_scale)


def _assert_oracle_array(
    context,
    actual,
    oracle,
    name,
    reference=None,
    roundoff_factor=_REFERENCE_ROUNDOFF_ULPS,
):
    expected = oracle[name]
    expected_scale = float(np.max(np.abs(expected))) if expected.size else 0.0
    if reference is None:
        reference = expected_scale if expected_scale > 0.0 else _zero_reference(
            context, name
        )
    _assert_native_close(
        actual,
        expected,
        name,
        reference=reference,
        roundoff_factor=roundoff_factor,
    )


def _physical_samples(context, values, entities):
    return forward(
        np.asarray(values, dtype=np.complex128).reshape(-1),
        entities,
        context,
    )


def _forced_field_state(context, oracle):
    nc = context.op.Nc
    nphi = context.N
    phi = 2.0 * np.pi * np.arange(nphi, dtype=np.float64) / nphi
    radial = np.asarray(context.geometry.cells.R, dtype=np.float64)
    axial = np.asarray(context.geometry.cells.Z, dtype=np.float64)
    axial_unit = (axial - axial.min()) / (axial.max() - axial.min())
    shape = (
        1.0
        + 0.08 * np.cos(phi)[:, None]
        + 0.04 * np.sin(2.0 * phi)[:, None] * axial_unit[None, :]
        + 0.03 * radial[None, :] / radial.max()
    )
    ion_density = context.p.ref.ni * shape
    electron_density = ion_density * (
        1.0 + 0.012 * np.cos(phi + 0.3)[:, None]
    )
    state = allocate_state(context)
    state.B = oracle["state_B"].copy()
    state.step = 0
    state.generation = 1
    state.time = 0.0
    state.fluid[0].n = _physical_samples(context, ion_density, nc)
    state.fluid[0].mom = np.zeros_like(state.fluid[0].mom)
    state.fluid[0].perp = _physical_samples(
        context,
        context.p.ref.pressure
        * (1.0 + 0.04 * np.cos(phi)[:, None])
        * np.ones((nphi, nc), dtype=np.float64),
        nc,
    )
    state.fluid[1].n = _physical_samples(context, electron_density, nc)
    electron_velocity = np.zeros((nphi, 3, nc), dtype=np.complex128)
    electron_velocity[:, 1, :] = (
        0.03 + 0.012 * np.cos(phi)[:, None] + 0.008 * axial_unit[None, :]
    )
    electron_momentum = (
        context.p.species[1].mass
        * electron_density[:, None, :]
        * electron_velocity
    )
    state.fluid[1].mom = _physical_samples(context, electron_momentum, 3 * nc)
    electron_perp = context.p.ref.pressure * (
        1.0
        + 0.12 * np.cos(phi)[:, None]
        + 0.05 * axial_unit[None, :]
    )
    electron_parallel = 1.7 * context.p.ref.pressure * (
        1.0
        + 0.09 * np.sin(phi)[:, None]
        - 0.03 * axial_unit[None, :]
    )
    state.fluid[1].perp = _physical_samples(context, electron_perp, nc)
    state.fluid[1].parallel = _physical_samples(context, electron_parallel, nc)
    return state


def test_wall_projection_is_constrained_idempotent_and_conserves_no_work():
    context, op, state, oracle = _oracle_fixture()
    xp = context.compute.xp
    nc = op.Nc
    density = inverse(state.fluid[0].n, nc, context, True)
    rng = np.random.default_rng(4)
    r = xp.asarray(
        rng.standard_normal(context.N * 3 * nc)
        + 1j * rng.standard_normal(context.N * 3 * nc),
        dtype=xp.complex128,
    )
    tag = StageTag(0, 1, 1, 0, 0.0, "test")
    wall = build_wall(density, context.p.species[0], tag, context)
    projection = project_wall(wall, r, context)
    full_rows = sample_apply(
        op.T_full, projection.value, context.N, context.compute
    )
    assert float(xp.max(xp.abs(full_rows))) < 2.0e-10
    again = project_wall(wall, projection.value, context)
    assert float(xp.max(xp.abs(again.value - projection.value))) < 2.0e-10
    work = xp.sum(
        xp.conj(projection.value)
        * projection.reaction
        / xp.tile(wall.samples[0].winv, context.N)
    )
    native_bound = (
        1.0e-12
        * float(xp.sum(xp.abs(r) ** 2))
        / float(xp.min(wall.samples[0].winv))
    )
    assert abs(complex(work)) < max(2.0e-9, native_bound)


def test_wall_nonuniform_density_preserves_component_cell_order_and_full_rows():
    context, op, state, oracle = _oracle_fixture()
    nc = op.Nc
    phi = 2.0 * np.pi * np.arange(context.N, dtype=np.float64) / context.N
    axial = np.asarray(context.geometry.cells.Z, dtype=np.float64)
    axial = (axial - axial.min()) / (axial.max() - axial.min())
    density = context.p.ref.ni * (
        1.0
        + 0.09 * np.cos(phi)[:, None]
        + 0.04 * np.sin(2.0 * phi)[:, None] * axial[None, :]
        + 0.03 * np.arange(nc, dtype=np.float64)[None, :] / nc
    )
    tag = StageTag(0, 1, 1, 0, 0.0, "test")
    wall = build_wall(density.reshape(-1), context.p.species[0], tag, context)
    expected_winv = (
        context.p.species[0].mass
        * np.tile(density, (1, 3))
        / np.tile(np.real(np.asarray(op.M0)), 3)[None, :]
    )
    for sample, record in enumerate(wall.samples):
        np.testing.assert_allclose(
            np.asarray(record.winv),
            expected_winv[sample],
            rtol=2.0e-13,
            atol=0.0,
        )

    rng = np.random.default_rng(41)
    force = rng.standard_normal(context.N * 3 * nc) + 1j * rng.standard_normal(
        context.N * 3 * nc
    )
    projection = project_wall(wall, force, context)
    assert np.max(np.abs(sample_apply(op.T_full, projection.value, context.N, context.compute))) < 2.0e-10
    second = project_wall(wall, projection.value, context)
    assert np.max(np.abs(second.value - projection.value)) < 2.0e-10
    work = np.sum(
        np.conj(projection.value)
        * projection.reaction
        / np.concatenate([sample.winv for sample in wall.samples])
    )
    bound = 1.0e-12 * np.sum(np.abs(force) ** 2) / np.min(expected_winv)
    assert abs(complex(work)) < max(2.0e-9, bound)
    other = rng.standard_normal(force.size) + 1j * rng.standard_normal(force.size)
    other_projection = project_wall(wall, other, context)
    metric_inverse = np.concatenate([sample.winv for sample in wall.samples])
    left = np.sum(np.conj(projection.value) * other / metric_inverse)
    right = np.sum(np.conj(force) * other_projection.value / metric_inverse)
    assert abs(complex(left - right)) < 2.0e-9 * max(1.0, abs(complex(left)))


def test_wall_rejects_nonpositive_density_before_division():
    context, op, state, oracle = _oracle_fixture()
    density = inverse(state.fluid[0].n, op.Nc, context, True)
    density[0] = -1.0
    with pytest.raises(Failure) as error:
        build_wall(
            density,
            context.p.species[0],
            StageTag(0, 1, 1, 0, 0.0, "test"),
            context,
        )
    assert error.value.code in {"Domain", "FluidDomain"}


def test_velocity_gradient_weighted_adjoint_and_trace_identity():
    context, op, state, oracle = _oracle_fixture()
    xp = context.compute.xp
    tag = StageTag(0, 1, 1, 0, 0.0, "test")
    bn, bhat = oracle["BN"], oracle["bhat"]
    fluid = build_fluid(state.fluid[0], 0, bn, bhat, tag, context)
    rng = np.random.default_rng(8)
    x = xp.asarray(
        rng.standard_normal(fluid.u.size) + 1j * rng.standard_normal(fluid.u.size),
        dtype=xp.complex128,
    )
    tensor = xp.asarray(
        rng.standard_normal(context.N * 9 * op.Nc)
        + 1j
        * rng.standard_normal(context.N * 9 * op.Nc),
        dtype=xp.complex128,
    )
    gradient = velocity_gradient(fluid, x, context)
    adjoint = gradient_adjoint(fluid, tensor, context)
    m0 = xp.real(xp.asarray(op.M0)).astype(xp.float64)
    tensor_weights = xp.tile(m0, context.N * 9)
    momentum_weights = xp.tile(m0, context.N * 3)
    lhs = xp.sum(tensor_weights * xp.conj(gradient) * tensor)
    rhs = xp.sum(momentum_weights * xp.conj(x) * adjoint)
    assert abs(complex(lhs - rhs)) <= 3.0e-9 * max(1.0, abs(complex(lhs)))

    gradient_u = velocity_gradient(fluid, fluid.u, context)
    nc = op.Nc
    trace = (
        gradient_u.reshape(context.N, 3, 3, nc)[:, 0, 0]
        + gradient_u.reshape(context.N, 3, 3, nc)[:, 1, 1]
        + gradient_u.reshape(context.N, 3, 3, nc)[:, 2, 2]
    ).reshape(-1)
    nu = fluid.u.reshape(context.N, 3, nc) * fluid.n.reshape(
        context.N, 1, nc
    )
    interpolated = sample_apply(
        op.I, nu.reshape(-1), context.N, context.compute
    ).reshape(context.N, op.Ne)
    nf = fluid.nf.reshape(context.N, op.Ne)
    nr = op.DR.shape[1]
    radial_trace = interpolated[:, :nr] / nf[:, :nr]
    azimuthal_trace = (
        interpolated[:, nr : nr + nc] / nf[:, nr : nr + nc]
    )
    axial_trace = interpolated[:, nr + nc :] / nf[:, nr + nc :]
    assert radial_trace.shape == (context.N, nr)
    assert azimuthal_trace.shape == (context.N, nc)
    assert axial_trace.shape == (context.N, op.DZ.shape[1])
    velocity_edges = xp.concatenate(
        (radial_trace, azimuthal_trace, axial_trace), axis=1
    )
    velocity_trace = divergence(velocity_edges.reshape(-1), context)
    assert float(xp.max(xp.abs(trace - velocity_trace))) < 2.0e-11


def test_pressure_tensor_cgl_sources_and_invalid_magnetic_input():
    context, op, state, oracle = _oracle_fixture()
    xp = context.compute.xp
    tag = StageTag(0, 1, 1, 0, 0.0, "test")
    bn, bhat = oracle["BN"], oracle["bhat"]
    electron = build_fluid(
        state.fluid[1], 1, bn, bhat, tag, context
    )
    build_non_electric(electron, 1, bn, bhat, context)
    nc = op.Nc
    tensor = electron.tensor.reshape(context.N, 9, nc)
    perp = electron.perp.reshape(context.N, nc)
    parallel = electron.parallel.reshape(context.N, nc)
    direction = bhat.reshape(context.N, 3, nc)
    assert np.allclose(
        tensor[:, 0, :],
        perp + (parallel - perp) * np.real(direction[:, 0, :]) ** 2,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert np.allclose(
        tensor[:, 4, :],
        perp + (parallel - perp) * np.real(direction[:, 1, :]) ** 2,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    local_source = (
        electron.perp * (electron.strain - electron.theta)
        - electron.parallel * electron.strain
    )
    contraction = xp.sum(
        electron.tensor.reshape(context.N, 3, 3, nc)
        * electron.gradient.reshape(context.N, 3, 3, nc),
        axis=(1, 2),
    )
    assert float(xp.max(xp.abs(local_source.reshape(context.N, nc) + contraction))) < 2.0e-12

    bad_bhat = bhat.copy()
    bad_bhat[nc] = 0.0
    with pytest.raises(Failure) as error:
        build_fluid(state.fluid[0], 0, bn, bad_bhat, tag, context)
    assert error.value.code in {"MagneticDirection", "Magnetization"}


def test_electrostatic_and_longitudinal_cover_all_rows():
    context, op, state, oracle = _oracle_fixture()
    xp = context.compute.xp
    nc = op.Nc
    rng = np.random.default_rng(15)
    physical = rng.standard_normal((context.N, nc))
    charge = forward(physical.reshape(-1), nc, context)
    tag = StageTag(0, 1, 1, 0, 0.0, "test")
    cache = electrostatic(charge, tag, context)
    for mode, matrix in enumerate(op.Lphi):
        phi = cache.phi[mode * nc : (mode + 1) * nc]
        residual = (
            matrix @ phi
            - charge[mode * nc : (mode + 1) * nc] / context.p.epsilon0
        )
        rhs_scale = max(
            1.0,
            float(
                xp.max(
                    xp.abs(
                        charge[mode * nc : (mode + 1) * nc]
                        / context.p.epsilon0
                    )
                )
            ),
        )
        assert float(xp.max(xp.abs(residual))) < context.p.rank_rtol * rhs_scale
    gauss = mode_apply(op.DJ, cache.EL, context.compute) - charge / context.p.epsilon0
    assert float(xp.max(xp.abs(gauss))) < context.p.rank_rtol * rhs_scale

    random_current = xp.asarray(
        rng.standard_normal((2 * context.p.K + 1) * op.Ne)
        + 1j * rng.standard_normal((2 * context.p.K + 1) * op.Ne),
        dtype=xp.complex128,
    )
    projected = longitudinal(random_current, context)
    transverse_part = random_current - projected
    divergence = mode_apply(op.DJ, transverse_part, context.compute)
    assert float(xp.max(xp.abs(divergence))) < 2.0e-9


def test_materialized_response_is_complex_linear_and_weighted_self_adjoint():
    context, op, state, oracle = _oracle_fixture()
    xp = context.compute.xp
    bn, bhat = oracle["BN"], oracle["bhat"]
    stage = type("Stage", (), {})()
    stage.tag = StageTag(0, 1, 1, 0, 0.0, "test")
    stage.fluid = [
        build_fluid(state.fluid[a], a, bn, bhat, stage.tag, context)
        for a in range(2)
    ]
    modes = 2 * context.p.K + 1
    stage.fJ0 = xp.zeros(modes * op.Ne, dtype=xp.complex128)
    stage.rho = sum(
        context.p.species[a].charge * state.fluid[a].n for a in range(2)
    )
    stage.cache = electrostatic(stage.rho, stage.tag, context)
    stage.residuals = {}
    stage.rhs = type("Rhs", (), {})()
    stage.rhs.B = xp.zeros(modes * op.Nb, dtype=xp.complex128)
    transverse(stage, context)

    size = modes * op.Ne
    rng = np.random.default_rng(23)
    x = xp.asarray(
        rng.standard_normal(size) + 1j * rng.standard_normal(size),
        dtype=xp.complex128,
    )
    y = xp.asarray(
        rng.standard_normal(size) + 1j * rng.standard_normal(size),
        dtype=xp.complex128,
    )
    alpha = 0.4 - 0.7j
    beta = -0.2 + 0.3j
    linearity = stage.AE @ (alpha * x + beta * y) - (
        alpha * (stage.AE @ x) + beta * (stage.AE @ y)
    )
    assert float(xp.max(xp.abs(linearity))) < 2.0e-10 * float(xp.max(xp.abs(stage.AE @ x)))

    direct = xp.zeros(size, dtype=xp.complex128)
    for species in range(2):
        force = electric_response(stage.fluid[species], species, x, context)
        edges = sample_apply(
            op.I,
            force
            * context.p.species[species].charge
            / context.p.species[species].mass,
            context.N,
            context.compute,
        )
        direct += forward(edges, op.Ne, context)
    assert float(xp.max(xp.abs(stage.AE @ x - direct))) < 2.0e-12 * max(
        1.0, float(xp.max(xp.abs(direct)))
    )
    m1 = xp.tile(
        xp.real(xp.asarray(op.M1)).astype(xp.float64),
        modes,
    )
    left = _weighted_dot(xp, x, stage.AE @ y, m1)
    right = _weighted_dot(xp, stage.AE @ x, y, m1)
    assert abs(complex(left - right)) < 2.0e-9 * max(1.0, abs(complex(left)))
    full_divergence = op.Bc_full @ stage.cache.ET
    assert float(xp.max(xp.abs(full_divergence))) < 2.0e-9


def test_forced_transverse_solve_matches_independent_direct_kkt():
    context, op, state, oracle = _oracle_fixture()
    tag = StageTag(0, 1, 1, 0, 0.0, "forced")
    state = _forced_field_state(context, oracle)
    result = evaluate_stage(state, tag, context, alias=False)

    assert np.max(np.abs(result.fluid[0].u)) < 2.0e-12
    assert np.max(np.abs(result.fluid[1].u.reshape(context.N, 3, op.Nc)[:, 1, :])) > 1.0e-2
    assert np.ptp(np.real(result.fluid[1].n)) > 1.0e11
    assert np.max(np.abs(result.fluid[1].parallel - result.fluid[1].perp)) > 1.0e-10
    assert np.max(np.abs(result.fJ0)) > 1.0e-12
    assert np.max(np.abs(result.cache.ET)) > 1.0e-18

    modes = 2 * context.p.K + 1
    edge_size = modes * op.Ne
    constraints = op.Bc.shape[0]
    system_size = edge_size + constraints
    m1_base = np.real(np.asarray(op.M1))
    m1 = np.tile(m1_base, modes)
    curl = np.zeros((edge_size, edge_size), dtype=np.complex128)
    for stored in range(modes):
        block = (
            m1_base[:, None]
            * (np.asarray(op.Cdag[stored]) @ np.asarray(op.C[stored]))
            / context.p.mu0
        )
        curl[
            stored * op.Ne : (stored + 1) * op.Ne,
            stored * op.Ne : (stored + 1) * op.Ne,
        ] = block
    dynamic = curl + m1[:, None] * np.asarray(result.AE)
    top_rhs = -m1 * (
        np.asarray(result.fJ0)
        + np.asarray(result.AE) @ np.asarray(result.cache.EL)
    )
    kkt = np.zeros((system_size, system_size), dtype=np.complex128)
    kkt[:edge_size, :edge_size] = dynamic
    kkt[:edge_size, edge_size:] = np.asarray(op.Bc).T.conj()
    kkt[edge_size:, :edge_size] = np.asarray(op.Bc)
    rhs = np.concatenate((top_rhs, np.zeros(constraints, dtype=np.complex128)))
    expected = np.linalg.solve(kkt, rhs)
    actual = np.concatenate((result.cache.ET, result.cache.multiplier))
    _assert_native_close(
        actual,
        expected,
        "forced_direct_kkt",
        reference=max(float(np.max(np.abs(expected))), float(np.max(np.abs(rhs)))),
        roundoff_factor=64.0,
    )
    full_divergence = np.asarray(op.Bc_full) @ np.asarray(result.cache.ET)
    _assert_native_close(
        full_divergence,
        np.zeros_like(full_divergence),
        "forced_full_divergence",
        reference=float(np.max(np.abs(top_rhs))),
    )


def test_power_ledger_subtracts_kinetic_density_term_once():
    context, op, state, oracle = _oracle_fixture()
    state = _forced_field_state(context, oracle)
    result = evaluate_stage(
        state,
        StageTag(0, 1, 1, 0, 0.0, "power"),
        context,
        alias=False,
    )
    species = 1
    fluid = result.fluid[species]
    terms = result.terms[species]
    u = np.asarray(fluid.u).reshape(context.N, 3, op.Nc)
    pidot = np.asarray(
        inverse(result.rhs.fluid[species].mom, 3 * op.Nc, context, True)
    ).reshape(context.N, 3, op.Nc)
    force = np.asarray(terms.force).reshape(context.N, 3, op.Nc)
    ndot = np.asarray(
        inverse(result.rhs.fluid[species].n, op.Nc, context, True)
    ).reshape(context.N, op.Nc)
    divergence = np.asarray(fluid.divergence).reshape(context.N, op.Nc)
    density_term = (
        0.5
        * context.p.species[species].mass
        * np.sum(np.abs(u) ** 2, axis=1)
    )
    assert np.max(np.abs(divergence)) > 1.0e-5
    assert np.max(np.abs(ndot)) > 1.0e5
    assert np.max(density_term) > 1.0e-34

    retained = np.conj(u) * pidot
    retained[:, 0, :] -= density_term * ndot
    sampled = np.conj(u) * force
    sampled[:, 0, :] += density_term * divergence
    expected_spectral_kinetic = np.real(
        _cell_integral(context, retained - sampled)
    )
    _assert_native_close(
        result.ledger[species].spectral_kinetic,
        expected_spectral_kinetic,
        "compressible_spectral_kinetic",
    )
    convection_base = np.conj(u) * np.asarray(
        fluid.convection
    ).reshape(context.N, 3, op.Nc)
    convection_pairing = np.array(convection_base, copy=True)
    divergence_product = density_term[:, None, :] * divergence[:, None, :]
    convection_pairing[:, 0, :] -= density_term * divergence
    convection_term_scale = abs(
        _cell_integral(context, np.abs(convection_base))
    ) + abs(_cell_integral(context, np.abs(divergence_product)))
    expected_convection = np.real(
        _cell_integral(context, convection_pairing)
    )
    _assert_native_close(
        result.ledger[species].convection,
        expected_convection,
        "compressible_convection_pairing",
        reference=convection_term_scale,
    )
    buggy_retained = np.conj(u) * pidot - density_term[:, None, :] * ndot[:, None, :]
    buggy_sampled = np.conj(u) * force + density_term[:, None, :] * divergence[:, None, :]
    buggy_spectral = np.real(_cell_integral(context, buggy_retained - buggy_sampled))
    buggy_convection = np.real(
        _cell_integral(
            context,
            np.conj(u) * np.asarray(fluid.convection).reshape(context.N, 3, op.Nc)
            - density_term[:, None, :] * divergence[:, None, :],
        )
    )
    expected_old_convection = np.real(
        _cell_integral(
            context,
            np.conj(u) * np.asarray(fluid.convection).reshape(context.N, 3, op.Nc)
            - density_term[:, None, :] * divergence[:, None, :],
        )
    )
    _assert_native_close(
        buggy_convection,
        expected_old_convection,
        "triple_density_convection_fixture",
    )
    with pytest.raises(AssertionError):
        _assert_native_close(
            buggy_spectral,
            expected_spectral_kinetic,
            "triple_density_spectral_mutation",
        )
    with pytest.raises(AssertionError):
        _assert_native_close(
            buggy_convection,
            expected_convection,
            "triple_density_convection_mutation",
            reference=convection_term_scale,
        )


def test_nontrivial_stage_and_initialize_reject_invalid_closed_reference():
    context, op, state, oracle = _oracle_fixture()
    xp = context.compute.xp
    tag = StageTag(
        step=state.step,
        generation=state.generation,
        trial=1,
        spectral_context=0,
        time=state.time,
        stage="integration",
    )
    result = evaluate_stage(state, tag, context, alias=False)
    assert result.source_id != 0
    assert result.rhs.fluid[0].n.shape == state.fluid[0].n.shape
    modes = 2 * context.p.K + 1
    assert result.cache.E.shape[0] == modes * op.Ne
    assert result.AE.shape == (
        modes * op.Ne,
        modes * op.Ne,
    )
    actual = None
    for species in range(2):
        species_actual = sample_apply(
            op.I,
            result.rhs.fluid[species].mom,
            modes,
            context.compute,
        ) * context.p.species[species].charge / context.p.species[species].mass
        actual = species_actual if actual is None else actual + species_actual
    expected = result.fJ0 + result.AE @ result.cache.E
    assert float(xp.max(xp.abs(actual - expected))) < 2.0e-9
    assert result.rhs.B.shape == state.B.shape
    for mode, matrix in enumerate(op.DB):
        block = result.rhs.B[mode * op.Nb : (mode + 1) * op.Nb]
        assert float(xp.max(xp.abs(matrix @ block))) < 2.0e-9

    stale = StageTag(
        step=state.step,
        generation=state.generation,
        trial=2,
        spectral_context=0,
        time=state.time + context.p.dt,
        stage="stale",
    )
    with pytest.raises(Failure) as stale_error:
        evaluate_stage(state, stale, context, alias=False)
    assert stale_error.value.code == "StaleStage"
    assert stale_error.value.stage == stale
    assert "trial_B" in stale_error.value.partial_arrays
    assert "trial_n_i" in stale_error.value.partial_arrays
    assert stale_error.value.checks

    cumulative = type("Cumulative", (), {})()
    with pytest.raises(Failure) as error:
        closed_reference(
            context,
            -1.0,
            context.p.ref.pressure,
            context.p.ref.pressure,
            context.p.ref.magnetic,
        )
    assert error.value.code in {"Fixture", "Domain"}
    initialized = initialize(state, context, cumulative, equilibrium=False)
    assert initialized.accepted


@pytest.mark.skipif(not ORACLE.exists(), reason="native oracle is unavailable")
def test_physics_matches_independent_native_oracle():
    context, op, state, oracle = _oracle_fixture()
    context.op = op
    reference = closed_reference(
        context,
        context.p.ref.ni,
        context.p.ref.pressure,
        context.p.ref.pressure,
        context.p.ref.magnetic,
        velocity=1.0e-2,
    )
    for species, ident in enumerate(("0", "1")):
        for name in ("n", "mom", "perp", "parallel"):
            _assert_oracle_array(
                context,
                getattr(reference.fluid[species], name),
                oracle,
                f"state_{ident}_{name}",
            )
    _assert_oracle_array(context, reference.B, oracle, "state_B")

    tag = StageTag(0, 1, 1, 0, 0.0, "initial")
    result = evaluate_stage(state, tag, context, alias=False)

    _assert_oracle_array(context, result.BN, oracle, "BN")
    _assert_oracle_array(context, result.bhat, oracle, "bhat")
    _assert_oracle_array(context, result.rho, oracle, "rho")
    _assert_oracle_array(context, result.current, oracle, "current")
    _assert_oracle_array(
        context,
        result.fJ0,
        oracle,
        "fJ0",
        reference=_fJ0_term_reference(context, result),
        roundoff_factor=256.0,
    )
    _assert_oracle_array(context, result.cache.phi, oracle, "phi")
    _assert_oracle_array(context, result.cache.EL, oracle, "EL")
    transverse_reference = _zero_reference(context, "ET")
    _assert_oracle_array(
        context, result.cache.ET, oracle, "ET", reference=transverse_reference
    )
    _assert_oracle_array(
        context, result.cache.E, oracle, "E", reference=transverse_reference
    )
    _assert_oracle_array(context, result.AE, oracle, "AE")
    _assert_oracle_array(context, result.ELdot, oracle, "ELdot")
    for species, ident in enumerate(("0", "1")):
        fluid = result.fluid[species]
        suffix = f"fluid_{ident}_"
        for name in (
            "u",
            "flux",
            "flux_hat",
            "nf",
            "tensor",
            "gradient",
            "theta",
            "strain",
            "convection",
            "pressure_force",
            "non_electric",
            "pdot_perp",
            "pdot_parallel",
        ):
            _assert_oracle_array(
                context,
                getattr(fluid, name),
                oracle,
                suffix + name,
            )
        for name in ("n", "mom", "perp", "parallel"):
            oracle_name = f"rhs_{ident}_{name}"
            _assert_oracle_array(
                context,
                getattr(result.rhs.fluid[species], name),
                oracle,
                oracle_name,
                reference=_zero_reference(context, oracle_name),
            )
        ledger = getattr(result.ledger[species], "__dict__")
        expected_ledger = oracle[f"ledger_{ident}"]
        actual_ledger = np.asarray(
            [
                ledger["electric_node"],
                ledger["electric_edge"],
                ledger["pressure_kinetic"],
                ledger["pressure_internal"],
                ledger["magnetic"],
                ledger["wall"],
                ledger["convection"],
                ledger["perp_rate"],
                ledger["parallel_rate"],
                ledger["spectral_kinetic"],
                ledger["spectral_perp"],
                ledger["spectral_parallel"],
            ]
        )
        _assert_native_close(
            actual_ledger,
            expected_ledger,
            f"ledger_{ident}",
            reference=_zero_reference(context, f"ledger_{ident}"),
        )
    _assert_oracle_array(
        context,
        result.rhs.B,
        oracle,
        "rhs_B",
        reference=_zero_reference(context, "rhs_B"),
    )
    for name, residual in result.residuals.items():
        oracle_name = f"residual_{name}"
        if oracle_name in oracle:
            _assert_oracle_array(
                context,
                residual,
                oracle,
                oracle_name,
                reference=_zero_reference(context, oracle_name),
            )


@pytest.mark.skipif(not STRESS_ORACLE.exists(), reason="stress oracle is unavailable")
def test_physics_matches_native_stress_oracle():
    context, op, state, oracle = _oracle_fixture(STRESS_ORACLE)
    assert oracle["_metadata"]["stress"] is True
    assert oracle["_metadata"]["velocity"] == 0.0
    assert oracle["_metadata"]["dt"] == 1.0e-12

    result = evaluate_stage(
        state,
        StageTag(0, 1, 1, 0, 0.0, "initial"),
        context,
        alias=False,
    )
    assert np.max(np.abs(result.fJ0)) > 1.0
    assert np.max(np.abs(result.cache.ET)) > 1.0e-7
    assert np.ptp(np.real(result.fluid[0].n)) > 1.0e11
    assert np.max(np.abs(result.fluid[1].parallel - result.fluid[1].perp)) > 1.0e-10

    _assert_oracle_array(context, result.BN, oracle, "BN")
    _assert_oracle_array(context, result.bhat, oracle, "bhat")
    _assert_oracle_array(context, result.rho, oracle, "rho")
    _assert_oracle_array(context, result.current, oracle, "current")
    _assert_oracle_array(
        context,
        result.fJ0,
        oracle,
        "fJ0",
        reference=_fJ0_term_reference(context, result),
        roundoff_factor=256.0,
    )
    _assert_oracle_array(context, result.cache.phi, oracle, "phi")
    _assert_oracle_array(context, result.cache.EL, oracle, "EL")
    transverse_scale = float(np.max(np.abs(oracle["ET"])))
    _assert_oracle_array(
        context,
        result.cache.ET,
        oracle,
        "ET",
        reference=transverse_scale,
        roundoff_factor=128.0,
    )
    _assert_oracle_array(
        context,
        result.cache.E,
        oracle,
        "E",
        reference=transverse_scale,
        roundoff_factor=128.0,
    )
    _assert_oracle_array(context, result.AE, oracle, "AE")
    _assert_oracle_array(context, result.ELdot, oracle, "ELdot")

    for species, ident in enumerate(("0", "1")):
        fluid = result.fluid[species]
        for name in (
            "u",
            "flux",
            "flux_hat",
            "nf",
            "tensor",
            "gradient",
            "theta",
            "strain",
            "convection",
            "pressure_force",
            "non_electric",
            "pdot_perp",
            "pdot_parallel",
        ):
            _assert_oracle_array(context, getattr(fluid, name), oracle, f"fluid_{ident}_{name}")
        for name in ("n", "mom", "perp", "parallel"):
            oracle_name = f"rhs_{ident}_{name}"
            _assert_oracle_array(
                context,
                getattr(result.rhs.fluid[species], name),
                oracle,
                oracle_name,
                reference=_zero_reference(context, oracle_name),
            )
        ledger = result.ledger[species]
        actual_ledger = np.asarray(
            [
                ledger.electric_node,
                ledger.electric_edge,
                ledger.pressure_kinetic,
                ledger.pressure_internal,
                ledger.magnetic,
                ledger.wall,
                ledger.convection,
                ledger.perp_rate,
                ledger.parallel_rate,
                ledger.spectral_kinetic,
                ledger.spectral_perp,
                ledger.spectral_parallel,
            ]
        )
        _assert_native_close(
            actual_ledger,
            oracle[f"ledger_{ident}"],
            f"ledger_{ident}",
            reference=_zero_reference(context, f"ledger_{ident}"),
        )

    _assert_oracle_array(
        context,
        result.rhs.B,
        oracle,
        "rhs_B",
        reference=_zero_reference(context, "rhs_B"),
    )
    for name, residual in result.residuals.items():
        oracle_name = f"residual_{name}"
        if oracle_name in oracle:
            _assert_oracle_array(
                context,
                residual,
                oracle,
                oracle_name,
                reference=_zero_reference(context, oracle_name),
            )


def test_warp_magnetic_force_kernel_compiles_and_matches_formula():
    os.environ["WARP_CACHE_PATH"] = str(
        Path(__file__).resolve().parents[2] / "warp-cache"
    )
    import warp as wp

    wp.init()
    kernel = _magnetic_force_kernel(wp)
    velocity = np.array(
        [1.0 + 1.0j, 2.0 - 1.0j, 3.0 + 0.5j, -1.0 + 0.2j, 0.3j, -2.0],
        dtype=np.complex128,
    )
    magnetic = np.array(
        [0.5 - 0.1j, 1.0 + 0.2j, -0.3 + 0.4j, 0.2 + 0.5j, -0.4, 0.8 - 0.2j],
        dtype=np.complex128,
    )
    density = np.array([2.0 + 0.1j, 3.0 - 0.2j], dtype=np.complex128)
    charge = -1.7
    interleaved = lambda values: np.ascontiguousarray(values).view(
        np.float64
    ).reshape(-1, 2)
    u = wp.array(interleaved(velocity), dtype=wp.vec2d, device="cpu")
    b = wp.array(interleaved(magnetic), dtype=wp.vec2d, device="cpu")
    n = wp.array(interleaved(density), dtype=wp.vec2d, device="cpu")
    out = wp.zeros(velocity.size, dtype=wp.vec2d, device="cpu")
    wp.launch(
        kernel,
        dim=density.size,
        inputs=[u, b, n, np.float64(charge), np.int32(1)],
        outputs=[out],
        device="cpu",
    )
    actual = out.numpy().reshape(-1, 2).view(np.complex128).reshape(-1)
    expected = np.empty_like(velocity)
    for sample in range(2):
        vector = sample * 3
        ur, uphi, uz = velocity[vector : vector + 3]
        br, bphi, bz = magnetic[vector : vector + 3]
        expected[vector : vector + 3] = charge * density[sample] * np.array(
            [
                uphi * bz - uz * bphi,
                uz * br - ur * bz,
                ur * bphi - uphi * br,
            ]
        )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)


def test_warp_magnetic_force_unpacks_and_retains_array_storage():
    os.environ["WARP_CACHE_PATH"] = str(
        Path(__file__).resolve().parents[2] / "warp-cache"
    )
    import warp as wp

    wp.init()
    backend = Backend("cpu")
    backend.name = "cuda"
    backend._warp = wp
    backend._wp_device = wp.get_device("cpu")
    backend._external_stream = lambda: None
    context = SimpleNamespace(
        N=2,
        op=SimpleNamespace(Nc=1),
        compute=backend,
    )
    velocity = np.array(
        [1.0 + 1.0j, 2.0 - 1.0j, 3.0 + 0.5j, -1.0 + 0.2j, 0.3j, -2.0],
        dtype=np.complex128,
    )
    magnetic = np.array(
        [0.5 - 0.1j, 1.0 + 0.2j, -0.3 + 0.4j, 0.2 + 0.5j, -0.4, 0.8 - 0.2j],
        dtype=np.complex128,
    )
    density = np.array([2.0 + 0.1j, 3.0 - 0.2j], dtype=np.complex128)
    charge = -1.7
    actual = _warp_magnetic_force(velocity, magnetic, density, charge, context)
    expected = np.empty_like(velocity)
    for sample in range(2):
        vector = sample * 3
        ur, uphi, uz = velocity[vector : vector + 3]
        br, bphi, bz = magnetic[vector : vector + 3]
        expected[vector : vector + 3] = charge * density[sample] * np.array(
            [
                uphi * bz - uz * bphi,
                uz * br - ur * bz,
                ur * bphi - uphi * br,
            ]
        )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)
