"""Independent fixtures and budgets for the massless-electron branch."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from fomhd.backend import Backend
from fomhd.core import allocate_state, forward
from fomhd.geometry import build_rectangle
from fomhd.operators import build_operators
from fomhd.types import (
    Budget,
    Context,
    Norm,
    Parameters,
    References,
    Species,
)


def parameters(K: int = 0, *, hard_diagnostics: bool = True) -> Parameters:
    ion = Species(
        id="ion",
        mass=1.67262192595e-27,
        charge=1.602176634e-19,
        gamma=5.0 / 3.0,
        n_min=0.0,
        face_min=0.0,
        p_min=0.0,
        parallel_min=0.0,
    )
    electron = Species(
        id="electron",
        mass=9.1093837139e-31,
        charge=-1.602176634e-19,
        gamma=0.0,
        n_min=0.0,
        face_min=0.0,
        p_min=0.0,
        parallel_min=0.0,
    )
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
        species=(ion, electron),
        epsilon0=8.8541878188e-12,
        mu0=1.25663706127e-6,
        B_min=1.0e-12,
        dt=2.0e-10,
        start_time=0.0,
        rank_rtol=1.0e-12,
        nsteps=1,
        K=K,
        Nphi=4 * K + 1,
        memory_limit=512 << 20,
        device_memory_limit=512 << 20,
        inner_distance=0.05,
        ref=ref,
        electron_inertia=0,
    )
    response = sum(
        species.charge * species.charge * ref.ni / species.mass
        for species in p.species
    )
    p.ref.current = p.ref.time * p.ref.electric * (
        response + 1.0 / (p.mu0 * ref.length**2)
    )

    def array(
        name: str,
        unit: str,
        reference: float,
        hard: bool = hard_diagnostics,
    ) -> None:
        for suffix, norm in (
            ("linf", Norm.linf),
            ("weighted_L2", Norm.weighted_l2),
        ):
            p.budgets[f"{name}.{suffix}"] = Budget(
                atol=0.0,
                rtol=2.0e-8,
                reference=reference,
                unit=unit,
                norm=norm,
                hard=hard,
            )

    def scalar(
        name: str,
        unit: str,
        reference: float,
        hard: bool = hard_diagnostics,
    ) -> None:
        p.budgets[f"{name}.value"] = Budget(
            atol=0.0,
            rtol=2.0e-8,
            reference=reference,
            unit=unit,
            norm=Norm.value,
            hard=hard,
        )

    density_rate = ref.ni / ref.time
    momentum_rate = ion.mass * ref.ni * ref.velocity / ref.time
    pressure_rate = ref.pressure / ref.time
    flux = ref.ni * ref.velocity
    current = p.ref.current
    length = ref.length
    for name, unit, scale in (
        ("quasineutrality", "C/m^3", ref.charge),
        ("electron_density_compatibility", "1/(m^3*s)", density_rate),
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
        ("ct_divergence", "T/(m*s)", ref.magnetic / (length * ref.time)),
        ("ct_update", "T/s", ref.magnetic / ref.time),
        ("divB_update", "T/m", ref.magnetic / length),
        ("wall_current_interval", "A/m^2", current),
        ("wall_ion_flux_interval.ion", "1/(m^2*s)", flux),
        (
            "wall_electron_flux_interval.electron",
            "1/(m^2*s)",
            flux,
        ),
        ("initial_divB", "T/m", ref.magnetic / length),
        ("wall_initial_current", "A/m^2", current),
        ("wall_current", "A/m^2", current),
        ("wall_ion_flux", "1/(m^2*s)", flux),
        ("wall_electron_flux", "1/(m^2*s)", flux),
        ("wall_field_original_rows", "V/m^3", ref.electric / length**2),
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
        ("theta_consistency", "1/s", 1.0 / ref.time),
        ("species_continuity", "1/(m^3*s)", density_rate),
        ("quasineutrality_interval", "C/(m^3*s)", ref.charge / ref.time),
    ):
        array(name, unit, scale, hard_diagnostics and name != "theta_consistency")
    p.budgets["theta_consistency.linf"].hard = False
    p.budgets["theta_consistency.weighted_L2"].hard = False

    for name, unit, scale in (
        ("minimum_density", "1/m^3", ref.ni),
        ("minimum_electron_density", "1/m^3", ref.ne),
        ("electric_power", "W", ref.pressure * ref.velocity * length**2),
        ("pressure_power", "W", ref.pressure * ref.velocity * length**2),
        ("pressure_energy", "J", ref.pressure * length**3),
        ("electron_perp_energy", "J", ref.pressure * length**3),
        ("electron_parallel_energy", "J", ref.pressure * length**3),
        ("magnetic_energy", "J", ref.magnetic**2 * length**3 / p.mu0),
        ("total_energy", "J", ref.pressure * length**3),
        ("ion_kinetic_energy", "J", ref.pressure * length**3),
        ("momentum_balance", "kg*m/s^2", momentum_rate * length**3),
        ("kinetic_energy.ion", "J", ref.pressure * length**3),
        ("ion_internal", "J", ref.pressure * length**3),
        ("electron_perp", "J", ref.pressure * length**3),
        ("electron_parallel", "J", ref.pressure * length**3),
        ("field_energy_B_interval", "J", ref.pressure * length**3),
        (
            "field_exchange_interval",
            "W",
            ref.pressure * length**3 / ref.time,
        ),
        ("total_energy_interval", "J", ref.pressure * length**3),
        ("total_charge_balance_interval", "C", ref.charge * length**3),
        ("total_number_balance.ion", "1", ref.ni * length**3),
        ("total_number_balance.electron", "1", ref.ne * length**3),
        (
            "momentum_balance_interval_x",
            "kg*m/s",
            ion.mass * ref.ni * ref.velocity * length**3,
        ),
        (
            "momentum_balance_interval_y",
            "kg*m/s",
            ion.mass * ref.ni * ref.velocity * length**3,
        ),
        (
            "momentum_balance_interval_z",
            "kg*m/s",
            ion.mass * ref.ni * ref.velocity * length**3,
        ),
        (
            "angular_momentum_interval",
            "kg*m^2/s",
            ion.mass * ref.ni * ref.velocity * length**4,
        ),
        (
            "wall_field_reaction_work",
            "W",
            current * ref.electric * length**3,
        ),
    ):
        scalar(name, unit, scale)
    for species in (ion, electron):
        population = "." + species.id
        array("alias_n" + population, "1/(m^3*s)", density_rate)
        array(
            "alias_mom" + population,
            "kg/(m^2*s^2)",
            momentum_rate,
        )
        array("alias_perp" + population, "Pa/s", pressure_rate)
        array("species_continuity" + population, "1/(m^3*s)", density_rate)
    array("alias_parallel.electron", "Pa/s", pressure_rate)
    array("alias_B", "T/s", ref.magnetic / ref.time)
    scalar(
        "electron_pressure_adjoint.electron",
        "W",
        ref.pressure * ref.velocity * length**2,
    )
    scalar(
        "electron_electric_pairing.electron",
        "W",
        ref.pressure * ref.velocity * length**2,
    )
    scalar(
        "electron_magnetic_work.electron",
        "W",
        ref.pressure * ref.velocity * length**2,
    )
    scalar(
        "pressure_pairing.electron",
        "W",
        ref.pressure * ref.velocity * length**2,
    )
    scalar(
        "electric_pairing.electron",
        "W",
        ref.pressure * ref.velocity * length**2,
    )
    scalar(
        "electron_pressure_only_pairing.electron",
        "W",
        ref.pressure * ref.velocity * length**2,
    )
    for name in (
        "wall_field_reaction_work_ion",
        "wall_field_reaction_work_electron",
        "wall_field_reaction_work_current",
    ):
        scalar(
            name,
            "W",
            ref.pressure * ref.velocity * length**2,
        )
    return p


def context(
    K: int = 0,
    *,
    hard_diagnostics: bool = True,
    nodes_r: int = 4,
    nodes_z: int = 4,
) -> Context:
    p = parameters(K, hard_diagnostics=hard_diagnostics)
    hr = 0.6 / (nodes_r - 1)
    hz = 0.6 / (nodes_z - 1)
    geometry = build_rectangle(nodes_r, nodes_z, 1.0, 0.0, hr, hz)
    backend = Backend("cpu")
    operators = build_operators(geometry, K, backend=backend)
    return Context(
        p=p,
        geometry=geometry,
        op=operators,
        compute=backend,
        N=p.Nphi,
        next_generation=2,
    )


def _mode_block(values: np.ndarray, stored: int, entities: int) -> np.ndarray:
    start = stored * entities
    return values[start : start + entities]


def _wall_extraction(context: Context, mode: int) -> np.ndarray:
    nc = context.op.Nc
    nr = len(context.geometry.r_faces.keys)
    walls = context.geometry.walls
    result = np.zeros((len(walls), context.op.Ne), dtype=np.complex128)
    for row, wall in enumerate(walls):
        edge = (
            wall.face
            if wall.component == "r_face"
            else nr + nc + wall.face
        )
        result[row, edge] = wall.sign
    return result


def compatible_magnetic_field(
    context: Context, *, seed: int = 7, scale: float = 1.0e-5
) -> np.ndarray:
    """Build divergence-free B in the null space of the full wall current."""
    rng = np.random.default_rng(seed)
    result = np.zeros(
        (2 * context.p.K + 1) * context.op.MB.size,
        dtype=np.complex128,
    )
    for mode in range(context.p.K + 1):
        stored = mode + context.p.K
        cdag = np.asarray(context.op.Cdag[stored])
        S = _wall_extraction(context, stored)
        db = np.asarray(context.op.DB[stored])
        constraints = np.concatenate((S @ cdag, db), axis=0)
        _, singular, vh = np.linalg.svd(constraints, full_matrices=True)
        threshold = context.p.rank_rtol * singular[0]
        null_count = int(np.count_nonzero(singular <= threshold))
        if null_count == 0:
            if mode == 0:
                raise AssertionError(
                    "fixture wall-current operator has no zero-mode null vector"
                )
            continue
        current_norms = [
            float(np.max(np.abs(cdag @ vector)))
            for vector in vh[-null_count:]
        ]
        selected = int(np.argmax(current_norms))
        coefficients = rng.standard_normal(null_count)
        if current_norms[selected] == 0.0:
            raise AssertionError("fixture null space has no nonzero current")
        coefficients[selected] += 1.0
        potential = (
            vh[-null_count:].T.conj() @ coefficients
        )
        mode_field = potential
        maximum = float(np.max(np.abs(mode_field)))
        if maximum == 0.0:
            raise AssertionError("fixture constructed a zero magnetic field")
        mode_field /= maximum
        _mode_block(result, stored, context.op.MB.size)[:] = mode_field
        if mode:
            _mode_block(
                result,
                context.p.K - mode,
                context.op.MB.size,
            )[:] = np.conj(mode_field)
    global_maximum = max(
        float(np.max(np.abs(_mode_block(result, stored, context.op.MB.size))))
        for stored in range(2 * context.p.K + 1)
    )
    result *= scale / global_maximum
    return context.compute.array(result)


def current_free_magnetic_field(context: Context, *, seed: int = 17):
    """Return a nonzero divergence-free zero-mode field in ker(Cdag)."""
    if context.p.K != 0:
        raise AssertionError("current-free fixture is axisymmetric")
    nc = context.op.Nc
    nb = context.op.MB.size
    constraints = np.concatenate(
        (
            np.asarray(context.op.Cdag[0]),
            np.asarray(context.op.DB[0]),
        ),
        axis=0,
    )
    _, singular, vh = np.linalg.svd(constraints, full_matrices=True)
    null_count = int(
        np.count_nonzero(
            singular <= context.p.rank_rtol * singular[0]
        )
    )
    if null_count == 0:
        raise AssertionError("fixture has no current-free magnetic vector")
    qb = np.asarray(context.op.QB)
    harmonic = np.conj(
        np.asarray(context.op.topology_flux_row).reshape(-1)
    ) / np.real(np.asarray(context.op.MB))
    harmonic *= context.p.ref.magnetic / float(
        np.max(np.abs(qb @ harmonic))
    )
    correction = np.linalg.lstsq(
        constraints, -(constraints @ harmonic), rcond=None
    )[0]
    B_mode = harmonic + correction
    if float(np.max(np.abs(qb @ B_mode))) == 0.0:
        raise AssertionError("fixture has no QB-visible current-free B")
    B_mode *= context.p.ref.magnetic / float(np.max(np.abs(B_mode)))
    result = np.zeros(
        (2 * context.p.K + 1) * nb, dtype=np.complex128
    )
    _mode_block(result, context.p.K, nb)[:] = B_mode
    return context.compute.array(result)


def wall_incompatible_current_field(context: Context, *, seed: int = 41):
    """Construct DBB=0 with a deliberately nonzero full-wall current."""
    if context.p.K != 0:
        raise AssertionError("incompatible-current fixture is axisymmetric")
    rng = np.random.default_rng(seed)
    electric_mode = rng.standard_normal(context.op.Ne)
    B_mode = np.asarray(context.op.C[0]) @ electric_mode
    B_mode *= context.p.ref.magnetic / float(np.max(np.abs(B_mode)))
    result = np.zeros(
        (2 * context.p.K + 1) * context.op.MB.size,
        dtype=np.complex128,
    )
    _mode_block(result, context.p.K, context.op.MB.size)[:] = B_mode
    return context.compute.array(result)


def uniform_state(context: Context, *, compatible: bool = True):
    state = allocate_state(context)
    nc = context.op.Nc
    zero = context.p.K * nc
    for species in range(2):
        state.fluid[species].n[zero : zero + nc] = context.p.ref.ni
        state.fluid[species].perp[zero : zero + nc] = context.p.ref.pressure
        if species == 1:
            state.fluid[species].parallel[zero : zero + nc] = (
                context.p.ref.pressure
            )
    state.B = (
        compatible_magnetic_field(context)
        if compatible
        else context.compute.array(
            np.random.default_rng(31).standard_normal(state.B.size)
            * context.p.ref.magnetic
        )
    )
    state.time = context.p.start_time
    state.step = 0
    state.generation = 1
    return state


def equilibrium_state(context: Context):
    state = uniform_state(context)
    state.B = current_free_magnetic_field(context)
    return state


def perturb_only_compatibility_momentum(state, context: Context):
    changed = type(state)(
        fluid=(
            type(state.fluid[0])(
                n=state.fluid[0].n.copy(),
                mom=state.fluid[0].mom.copy(),
                perp=state.fluid[0].perp.copy(),
                parallel=state.fluid[0].parallel.copy(),
            ),
            type(state.fluid[1])(
                n=state.fluid[1].n.copy(),
                mom=state.fluid[1].mom.copy(),
                perp=state.fluid[1].perp.copy(),
                parallel=state.fluid[1].parallel.copy(),
            ),
        ),
        B=state.B.copy(),
        step=state.step,
        generation=state.generation,
        time=state.time,
    )
    changed.fluid[1].mom += (
        context.p.species[1].mass
        * context.p.ref.ne
        * context.p.ref.velocity
        * 1.0e-3
    )
    return changed


def stage_tag(state):
    from fomhd.types import StageTag

    return StageTag(
        step=state.step,
        generation=state.generation,
        trial=1,
        spectral_context=0,
        time=state.time,
        stage="test",
    )


def weighted_dot(context: Context, left, right, weights) -> complex:
    xp = context.compute.xp
    return complex(
        xp.sum(
            xp.asarray(weights, dtype=xp.float64)
            * xp.conj(xp.asarray(left))
            * xp.asarray(right)
        ).item()
    )


def budget_names() -> set[str]:
    return {
        "quasineutrality",
        "electron_density_compatibility",
        "electron_momentum_compatibility",
        "current_definition",
        "electron_flux_definition",
        "electron_flux_continuity",
        "djj",
        "ct_divergence",
        "wall_current",
        "wall_ion_flux",
        "wall_electron_flux",
        "wall_field_original_rows",
        "wall_field_reaction_work",
        "drift_ohm_residual",
        "electron_cgl_perp",
        "electron_cgl_parallel",
        "electron_pressure_adjoint",
        "electron_electric_pairing",
        "theta_consistency",
    }
