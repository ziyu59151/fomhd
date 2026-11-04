"""Independent periodic Hall fixture for the massless source stage.

The production stage and its diagnostics remain the system under test.  This
module only constructs test geometry/state data and an independent cold
four-field continuation oracle.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from fomhd import diagnostics as diagnostics_module
from fomhd.backend import Backend
from fomhd import massless_checks
from fomhd.core import allocate_state, forward, inverse, mode_apply, sample_apply
from fomhd.fluid import (
    _component_flux,
    _sampled_divergence,
    gradient_adjoint,
    velocity_gradient,
)
from fomhd.geometry import EntityTable, Point2, build_geometry
from fomhd.massless import _constraint_project, _derive_electron, _massless_e0
from fomhd.operators import Operators, build_operators
from fomhd.stage import closed_reference, evaluate_stage, initialize
from fomhd.time import advance
from fomhd.types import (
    Budget,
    Context,
    Failure,
    Norm,
    Parameters,
    References,
    Species,
    StageTag,
    State,
    cumulative_record,
)


MU0 = 1.25663706127e-6
E_CHARGE = 1.602176634e-19
MI = 1.67262192595e-27
ME = 9.1093837139e-31
EPSILON0 = 8.8541878188e-12
R_PERIODIC = 1.0e30


@dataclass(frozen=True)
class HallCase:
    """Physical parameters for one parallel cold Hall mode."""

    kdi: float = 1.0
    points: int = 9
    identity: str = ""
    n0: float = 1.0e16
    B0: float = 0.1
    pressure: float = 0.4

    def __post_init__(self) -> None:
        if not math.isfinite(self.kdi) or self.kdi <= 0.0:
            raise ValueError("kdi must be positive and finite")
        if self.points < 9 or self.points % 2 == 0:
            raise ValueError("periodic fixture requires an odd point count >= 9")
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (self.n0, self.B0, self.pressure)
        ):
            raise ValueError("Hall equilibrium values must be positive")

    @property
    def di(self) -> float:
        return math.sqrt(MI / (MU0 * self.n0 * E_CHARGE**2))

    @property
    def k(self) -> float:
        return self.kdi / self.di

    @property
    def wavelength(self) -> float:
        return 2.0 * math.pi / self.k

    @property
    def v_alfven(self) -> float:
        return self.B0 / math.sqrt(MU0 * MI * self.n0)

    @property
    def alfven_rate(self) -> float:
        return self.k * self.v_alfven

    @property
    def hall_rate(self) -> float:
        return self.k * self.k * self.B0 / (MU0 * E_CHARGE * self.n0)

    @property
    def omega_plasma(self) -> float:
        return math.sqrt(self.n0 * E_CHARGE**2 / (EPSILON0 * ME))


@dataclass(frozen=True)
class HallFrequencies:
    """Positive Hall branches and the scale that separates them."""

    omega_minus: float
    omega_plus: float
    alfven_rate: float
    hall_rate: float
    omega_plasma: float


@dataclass(frozen=True)
class HallTimeResult:
    """Observed finite-step result and its independent polynomial oracle."""

    integrator: str
    requested_steps: int
    accepted_steps: int
    dt: float
    final_time: float
    omega: float
    amplification: complex
    actual_mode: np.ndarray
    expected_mode: np.ndarray
    relative_error: float
    continuum_relative_error: float
    amplitude_ratio: float
    phase_error_rad: float
    continuum_amplitude_ratio: float
    continuum_phase_error_rad: float
    initial_mode_error: float
    finite: bool
    hard_checks_available: bool
    source_is_none: bool
    diagnostics: tuple[Any, ...]
    dt_reports: tuple[Any, ...]
    failure_code: str | None
    failure_message: str
    warning: str = ""


@dataclass(frozen=True)
class SpatialEvidence:
    """Operator-level closed-grid smooth-field measurements."""

    minimum_density: float
    maximum_density: float
    theta_relative_error: dict[str, float]
    full_theta_relative_error: dict[str, float]
    retained_delta_relative_error: dict[str, float]
    wrong_djfe_relative_error: dict[str, float]
    pressure_force_error: dict[str, float]
    pressure_norms: dict[str, dict[str, Any]]
    ohm_error: dict[str, float]
    ohm_norms: dict[str, dict[str, Any]]
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class ReconstructionEvidence:
    """Actual QHFe/ne reconstruction from smooth closed-domain flux/current."""

    nodes: int
    h: float
    minimum_density: float
    maximum_density: float
    maximum_abs_current: float
    current_relation_error: float
    velocity_norms: dict[str, dict[str, Any]]
    theta_norms: dict[str, dict[str, Any]]
    delta_theta_norms: dict[str, dict[str, Any]]
    maximum_wall_flux: float
    relative_wall_flux: float
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class DensityLayoutEvidence:
    """Variable-density electron velocity reconstruction error."""

    relative_error: float
    linf_error: float
    density_min: float
    density_max: float
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class DriftOhmEvidence:
    """Behavior of the drift-Ohm residual under a perturbed solved field."""

    electric_field_linf: float
    baseline_linf: float
    doubled_linf: float
    hard_threshold: float
    unit: str
    gate_failed: bool
    arrays: dict[str, np.ndarray]


def hall_frequencies(case: HallCase) -> HallFrequencies:
    """Return the independent cold two-branch frequency formula."""
    a = case.alfven_rate
    h = case.hall_rate
    root = math.hypot(h, 2.0 * a)
    return HallFrequencies(
        omega_minus=0.5 * (root - h),
        omega_plus=0.5 * (root + h),
        alfven_rate=a,
        hall_rate=h,
        omega_plasma=case.omega_plasma,
    )


def cold_transverse_matrix(
    case: HallCase, *, hall_sign: int = 1
) -> np.ndarray:
    """Return the independent complex matrix in normalized state order.

    State order is ``(ui_x/va, ui_y/va, Bx/B0, By/B0)``.  The signs follow
    the electron flux ``Fe=Fi-J/e``, ``E=-ue x B0``, and Faraday's law.
    """
    if hall_sign not in (-1, 1):
        raise ValueError("hall_sign must be +1 or -1")
    a = case.alfven_rate
    h = float(hall_sign) * case.hall_rate
    matrix = np.zeros((4, 4), dtype=np.complex128)
    matrix[0, 2] = 1j * a
    matrix[1, 3] = 1j * a
    matrix[2, 0] = 1j * a
    matrix[2, 3] = -h
    matrix[3, 1] = 1j * a
    matrix[3, 2] = h
    return matrix


def _entity_table(name: str, points: int, wavelength: float) -> EntityTable:
    table = EntityTable(name=name)
    table.keys = [(index, 0) for index in range(points)]
    table.dense_to_active = np.arange(points, dtype=np.int64)
    table.active_to_dense = np.arange(points, dtype=np.int64)
    table.R = np.full(points, R_PERIODIC, dtype=np.float64)
    table.Z = (
        np.arange(points, dtype=np.float64)
        * wavelength
        / float(points)
    )
    table.boundary = np.zeros(points, dtype=np.uint8)
    table.lookup = {key: index for index, key in enumerate(table.keys)}
    return table


def periodic_derivative(points: int, k: float) -> np.ndarray:
    """Native periodic Fourier derivative for the first retained wave."""
    derivative = np.zeros((points, points), dtype=np.complex128)
    half = (points - 1) // 2
    coefficients = np.zeros(half + 1, dtype=np.longdouble)
    for distance in range(1, half + 1):
        total = np.longdouble(0.0)
        for mode in range(1, half + 1):
            argument = (
                2.0
                * np.longdouble(np.pi)
                * mode
                * distance
                / np.longdouble(points)
            )
            total += np.longdouble(mode) * np.sin(argument)
        coefficients[distance] = (
            np.longdouble(4.0)
            * np.longdouble(np.pi)
            * total
            / np.longdouble(points * points)
        )
    scale = k * points / (2.0 * math.pi)
    for row in range(points):
        for distance in range(1, half + 1):
            forward = (row + distance) % points
            backward = (row - distance) % points
            value = scale * float(coefficients[distance])
            derivative[row, forward] = value
            derivative[row, backward] = -value
    return derivative


def _periodic_geometry(
    points: int, wavelength: float
) -> SimpleNamespace:
    cells = _entity_table("cells", points, wavelength)
    r_faces = _entity_table("r_faces", points, wavelength)
    z_faces = _entity_table("z_faces", points, wavelength)
    vertices = _entity_table("vertices", points, wavelength)
    return SimpleNamespace(
        R=np.array([R_PERIODIC], dtype=np.float64),
        Z=np.array([0.0, wavelength], dtype=np.float64),
        active=np.ones(points, dtype=np.uint8),
        cells=cells,
        r_faces=r_faces,
        z_faces=z_faces,
        vertices=vertices,
        M0=np.ones(points, dtype=np.float64),
        M1=np.ones(3 * points, dtype=np.float64),
        MB=np.ones(3 * points, dtype=np.float64),
        lr=np.ones(points, dtype=np.float64),
        lz=np.ones(points, dtype=np.float64),
        av=np.ones(points, dtype=np.float64),
        wv=np.ones(points, dtype=np.float64),
        Rbar_r=np.full(points, R_PERIODIC, dtype=np.float64),
        Rbar_v=np.full(points, R_PERIODIC, dtype=np.float64),
        cell_wall_distance=np.full(points, np.inf, dtype=np.float64),
        walls=[],
        wall_polyline=[],
    )


def _periodic_operators(
    geometry: SimpleNamespace,
    points: int,
    k: float,
    backend: Backend,
) -> Operators:
    nc = points
    ne = 3 * nc
    nb = 3 * nc
    derivative = periodic_derivative(points, k)
    radial = np.zeros((points, points), dtype=np.complex128)
    azimuthal = np.zeros((points, points), dtype=np.complex128)
    axial = derivative
    gradient = np.zeros((ne, nc), dtype=np.complex128)
    gradient[nc : 2 * nc] = azimuthal
    gradient[2 * nc :] = axial
    curl = np.zeros((nb, ne), dtype=np.complex128)
    curl[0:nc, nc : 2 * nc] = -axial
    curl[nc : 2 * nc, 0:nc] = axial
    curl[nc : 2 * nc, 2 * nc :] = -radial
    curl[2 * nc :, nc : 2 * nc] = radial
    divergence = np.zeros((nc, ne), dtype=np.complex128)
    divergence[:, 0:nc] = radial
    divergence[:, 2 * nc :] = axial
    current_divergence = -gradient.T.conj()
    identity_edges = np.eye(ne, dtype=np.complex128)
    identity_cells = np.eye(nc, dtype=np.complex128)
    identity_vectors = np.eye(ne, dtype=np.complex128)
    empty_wall = np.zeros((0, ne), dtype=np.complex128)

    op = Operators(
        K=0,
        compute=backend,
        Nc=nc,
        Ne=ne,
        Nb=nb,
        M0=backend.real_array(geometry.M0),
        M1=backend.real_array(geometry.M1),
        MB=backend.real_array(geometry.MB),
        Rbar_r=backend.real_array(geometry.Rbar_r),
        Rbar_v=backend.real_array(geometry.Rbar_v),
        b1r=np.zeros((nc, nc), dtype=np.complex128),
        b1z=np.zeros((nc, nc), dtype=np.complex128),
        b2r=np.zeros((nc, nc), dtype=np.complex128),
        b2z=np.zeros((nc, nc), dtype=np.complex128),
        PR_r=backend.asarray(identity_cells),
        PZ_r=backend.asarray(identity_cells),
        PR_z=backend.asarray(identity_cells),
        PZ_z=backend.asarray(identity_cells),
        PR_v=backend.asarray(identity_cells),
        PZ_v=backend.asarray(identity_cells),
        G=[backend.asarray(gradient)],
        C=[backend.asarray(curl)],
        DB=[backend.asarray(divergence)],
        DJ=[backend.asarray(current_divergence)],
        Cdag=[backend.asarray(curl.T.conj())],
        Lphi=[backend.asarray(identity_cells)],
        I=backend.asarray(identity_edges),
        H=backend.asarray(identity_edges),
        QB=backend.asarray(identity_vectors),
        DR=backend.asarray(radial),
        DZ=backend.asarray(axial),
        T_full=backend.asarray(empty_wall),
        T=backend.asarray(empty_wall),
        selected_wall_rows=[],
        Bc_full=backend.asarray(current_divergence),
        Bc=backend.asarray(current_divergence),
        selected_constraint_rows=list(range(nc)),
        topology_flux_row=backend.asarray(
            np.zeros((1, ne), dtype=np.complex128)
        ),
        setup_checks=[],
        poisson_spectra=[],
    )
    for name in ("PR_r", "PZ_r", "PR_z", "PZ_z", "PR_v", "PZ_v"):
        setattr(op, f"{name}_factor", backend.factor(getattr(op, name)))
    return op


def _add_array_budget(
    budgets: dict[str, Budget],
    name: str,
    unit: str,
    reference: float,
) -> None:
    for suffix, norm in (
        ("linf", Norm.linf),
        ("weighted_L2", Norm.weighted_l2),
    ):
        budgets[f"{name}.{suffix}"] = Budget(
            atol=0.0,
            rtol=2.0e-8,
            reference=reference,
            unit=unit,
            norm=norm,
            hard=name != "theta_consistency",
        )


def _add_scalar_budget(
    budgets: dict[str, Budget],
    name: str,
    unit: str,
    reference: float,
) -> None:
    budgets[f"{name}.value"] = Budget(
        atol=0.0,
        rtol=2.0e-8,
        reference=reference,
        unit=unit,
        norm=Norm.value,
        hard=True,
    )


def _parameters(case: HallCase) -> Parameters:
    ion = Species(
        id="ion",
        mass=MI,
        charge=E_CHARGE,
        gamma=5.0 / 3.0,
    )
    electron = Species(
        id="electron",
        mass=ME,
        charge=-E_CHARGE,
        gamma=0.0,
    )
    reference_frequency = math.hypot(case.alfven_rate, case.hall_rate)
    ref = References(
        length=1.0,
        time=1.0 / reference_frequency,
        ni=case.n0,
        ne=case.n0,
        velocity=case.v_alfven,
        electric=case.B0 * case.v_alfven,
        magnetic=case.B0,
        current=case.B0 / (MU0 * case.wavelength),
        charge=E_CHARGE * case.n0,
        pressure=case.pressure,
    )
    parameters = Parameters(
        species=(ion, electron),
        epsilon0=EPSILON0,
        mu0=MU0,
        B_min=1.0e-12,
        dt=1.0e-12,
        start_time=0.0,
        rank_rtol=1.0e-12,
        nsteps=1,
        K=0,
        Nphi=1,
        memory_limit=512 << 20,
        device_memory_limit=512 << 20,
        inner_distance=0.0,
        ref=ref,
        electron_inertia=0,
    )
    density_rate = case.n0 / ref.time
    momentum_rate = MI * case.n0 * case.v_alfven / ref.time
    pressure_rate = case.pressure / ref.time
    massless_current = ref.current
    flux = case.n0 * case.v_alfven
    length = ref.length
    for name, unit, scale in (
        ("quasineutrality", "C/m^3", ref.charge),
        ("electron_density_compatibility", "1/(m^3*s)", density_rate),
        ("electron_momentum_compatibility", "kg/(m^2*s^2)",
         ME * density_rate * case.v_alfven),
        ("current_definition", "A/m^2", massless_current),
        ("electron_flux_definition", "1/(m^2*s)", flux),
        ("electron_flux_continuity", "1/(m^3*s)", density_rate),
        ("djj", "A/m^3", massless_current / length),
        ("djj_interval", "A/m^3", massless_current / length),
        ("ct_divergence", "T/(m*s)", case.B0 / (length * ref.time)),
        ("ct_update", "T/s", case.B0 / ref.time),
        ("divB_update", "T/m", case.B0 / length),
        ("wall_current_interval", "A/m^2", massless_current),
        ("wall_ion_flux_interval.ion", "1/(m^2*s)", flux),
        ("wall_electron_flux_interval.electron", "1/(m^2*s)", flux),
        ("initial_divB", "T/m", case.B0 / length),
        ("wall_initial_current", "A/m^2", massless_current),
        ("wall_current", "A/m^2", massless_current),
        ("wall_ion_flux", "1/(m^2*s)", flux),
        ("wall_electron_flux", "1/(m^2*s)", flux),
        ("wall_field_original_rows", "V/m^3",
         ref.electric / length**2),
        ("drift_ohm_residual", "V/m", ref.electric),
        ("electron_cgl_perp", "Pa/s", pressure_rate),
        ("electron_cgl_parallel", "Pa/s", pressure_rate),
        ("electron_pressure_adjoint", "W",
         case.pressure * case.v_alfven * length**2),
        ("electron_electric_pairing", "W",
         case.pressure * case.v_alfven * length**2),
        ("theta_consistency", "1/s", 1.0 / ref.time),
        ("species_continuity", "1/(m^3*s)", density_rate),
        ("quasineutrality_interval", "C/(m^3*s)", ref.charge / ref.time),
    ):
        _add_array_budget(parameters.budgets, name, unit, scale)
    for name, unit, scale in (
        ("minimum_density", "1/m^3", case.n0),
        ("minimum_electron_density", "1/m^3", case.n0),
        ("electric_power", "W",
         case.pressure * case.v_alfven * length**2),
        ("pressure_power", "W",
         case.pressure * case.v_alfven * length**2),
        ("pressure_energy", "J", case.pressure * length**3),
        ("electron_perp_energy", "J", case.pressure * length**3),
        ("electron_parallel_energy", "J", case.pressure * length**3),
        ("magnetic_energy", "J", case.B0**2 * length**3 / MU0),
        ("total_energy", "J", case.pressure * length**3),
        ("ion_kinetic_energy", "J", case.pressure * length**3),
        ("momentum_balance", "kg*m/s^2", momentum_rate * length**3),
        ("kinetic_energy.ion", "J", case.pressure * length**3),
        ("ion_internal", "J", case.pressure * length**3),
        ("electron_perp", "J", case.pressure * length**3),
        ("electron_parallel", "J", case.pressure * length**3),
        ("field_energy_B_interval", "J", case.pressure * length**3),
        ("field_exchange_interval", "W",
         case.pressure * length**3 / ref.time),
        ("total_energy_interval", "J", case.pressure * length**3),
        ("total_charge_balance_interval", "C", ref.charge * length**3),
        ("total_number_balance.ion", "1", case.n0 * length**3),
        ("total_number_balance.electron", "1", case.n0 * length**3),
        ("momentum_balance_interval_x", "kg*m/s",
         momentum_rate * length**3 * ref.time),
        ("momentum_balance_interval_y", "kg*m/s",
         momentum_rate * length**3 * ref.time),
        ("momentum_balance_interval_z", "kg*m/s",
         momentum_rate * length**3 * ref.time),
        ("angular_momentum_interval", "kg*m^2/s",
         MI * case.n0 * case.v_alfven * length**4),
        ("wall_field_reaction_work", "W",
         massless_current * ref.electric * length**3),
    ):
        _add_scalar_budget(parameters.budgets, name, unit, scale)
    for population in ("ion", "electron"):
        _add_array_budget(
            parameters.budgets,
            f"alias_n.{population}",
            "1/(m^3*s)",
            density_rate,
        )
        _add_array_budget(
            parameters.budgets,
            f"alias_mom.{population}",
            "kg/(m^2*s^2)",
            momentum_rate,
        )
        _add_array_budget(
            parameters.budgets,
            f"alias_perp.{population}",
            "Pa/s",
            pressure_rate,
        )
        _add_array_budget(
            parameters.budgets,
            f"species_continuity.{population}",
            "1/(m^3*s)",
            density_rate,
        )
    _add_array_budget(
        parameters.budgets, "alias_parallel.electron", "Pa/s", pressure_rate
    )
    _add_array_budget(
        parameters.budgets, "alias_B", "T/s", case.B0 / ref.time
    )
    for name in (
        "pressure_pairing.electron",
        "electric_pairing.electron",
        "electron_magnetic_work.electron",
        "electron_pressure_only_pairing.electron",
        "wall_field_reaction_work_ion",
        "wall_field_reaction_work_electron",
        "wall_field_reaction_work_current",
    ):
        _add_scalar_budget(
            parameters.budgets,
            name,
            "W",
            case.pressure * case.v_alfven * length**2,
        )
    return parameters


def build_periodic_context(
    case: HallCase, *, device: str = "cpu"
) -> Context:
    """Build the test-only periodic geometry and real source context."""
    backend = Backend(device)
    geometry = _periodic_geometry(case.points, case.wavelength)
    operators = _periodic_operators(geometry, case.points, case.k, backend)
    parameters = _parameters(case)
    return Context(
        p=parameters,
        geometry=geometry,
        op=operators,
        compute=backend,
        N=parameters.Nphi,
        next_generation=2,
    )


def uniform_state(context: Context, case: HallCase) -> State:
    """Create the uniform positive-pressure current-free equilibrium."""
    state = allocate_state(context)
    nc = context.op.Nc
    for species in range(2):
        state.fluid[species].n[:] = case.n0
        state.fluid[species].perp[:] = case.pressure
        if species == 1:
            state.fluid[species].parallel[:] = case.pressure
    state.B[2 * nc : 3 * nc] = case.B0
    state.step = 0
    state.generation = 1
    state.time = context.p.start_time
    return state


def stage_tag(state: State, stage: str = "verification") -> StageTag:
    return StageTag(
        step=state.step,
        generation=state.generation,
        trial=1,
        spectral_context=0,
        time=state.time,
        stage=stage,
    )


def evaluate(
    state: State, context: Context, *, alias: bool = False
) -> Any:
    return evaluate_stage(
        state,
        stage_tag(state, "dispersion"),
        context,
        alias,
    )


def copy_state(state: State) -> State:
    return copy.deepcopy(state)


def cosine_field(
    case: HallCase, amplitude: float
) -> np.ndarray:
    coordinate = (
        2.0
        * math.pi
        * np.arange(case.points, dtype=np.float64)
        / float(case.points)
    )
    return amplitude * np.cos(coordinate)


def perturb_state(
    state: State,
    case: HallCase,
    variable: str,
    amplitude: float,
    context: Context,
) -> State:
    """Perturb one transverse physical variable by a cosine mode."""
    changed = copy_state(state)
    nc = case.points
    field = cosine_field(case, amplitude)
    xp = context.compute.xp
    if variable == "ui_x":
        scale = MI * case.n0 * case.v_alfven
        changed.fluid[0].mom[:nc] += xp.asarray(scale * field)
    elif variable == "ui_y":
        scale = MI * case.n0 * case.v_alfven
        changed.fluid[0].mom[nc : 2 * nc] += xp.asarray(scale * field)
    elif variable == "Bx":
        changed.B[:nc] += xp.asarray(case.B0 * field)
    elif variable == "By":
        changed.B[nc : 2 * nc] += xp.asarray(case.B0 * field)
    else:
        raise ValueError(f"unknown transverse variable: {variable}")
    return changed


def project_component(
    values: np.ndarray,
    component: int,
    case: HallCase,
) -> complex:
    """Return the unit-cosine continuation coefficient of one component."""
    if not isinstance(values, np.ndarray):
        raise TypeError("project_component requires explicit host conversion")
    samples = np.asarray(values, dtype=np.complex128).reshape(3, case.points)
    if np.max(np.abs(samples.imag)) > (
        4096.0 * np.finfo(np.float64).eps * max(np.max(np.abs(samples)), 1.0)
    ):
        raise ValueError("periodic modal projection requires real physical samples")
    real = samples[component].real
    coordinate = (
        2.0
        * math.pi
        * np.arange(case.points, dtype=np.float64)
        / float(case.points)
    )
    cosine = 2.0 * np.dot(real, np.cos(coordinate)) / case.points
    sine = 2.0 * np.dot(real, np.sin(coordinate)) / case.points
    return complex(
        cosine,
        -sine,
    )


def production_transverse_matrix(
    case: HallCase,
    *,
    relative_amplitudes: tuple[float, float] = (2.0e-6, 4.0e-6),
    device: str = "cpu",
) -> tuple[np.ndarray, tuple[np.ndarray, ...], dict[str, Any]]:
    """Differentiate the actual species stage for the four Hall fields."""
    context = build_periodic_context(case, device=device)
    equilibrium = uniform_state(context, case)
    variables = ("ui_x", "ui_y", "Bx", "By")
    matrices = []
    evidence: dict[str, Any] = {"diagnostics": []}
    for relative in relative_amplitudes:
        matrix = np.zeros((4, 4), dtype=np.complex128)
        for column, variable in enumerate(variables):
            plus = evaluate(
                perturb_state(
                    equilibrium, case, variable, relative, context
                ),
                context,
            )
            minus = evaluate(
                perturb_state(
                    equilibrium, case, variable, -relative, context
                ),
                context,
            )
            evidence["diagnostics"].extend(plus.checks)
            evidence["diagnostics"].extend(minus.checks)
            for row, (field, component) in enumerate(
                (("momentum", 0), ("momentum", 1), ("B", 0), ("B", 1))
            ):
                if field == "momentum":
                    plus_values = context.compute.to_host(
                        plus.rhs.fluid[0].mom
                    )
                    minus_values = context.compute.to_host(
                        minus.rhs.fluid[0].mom
                    )
                    projection = (
                        project_component(plus_values, component, case)
                        - project_component(minus_values, component, case)
                    ) / (2.0 * relative)
                    row_scale = MI * case.n0 * case.v_alfven
                else:
                    plus_values = context.compute.to_host(plus.rhs.B)
                    minus_values = context.compute.to_host(minus.rhs.B)
                    projection = (
                        project_component(plus_values, component, case)
                        - project_component(minus_values, component, case)
                    ) / (2.0 * relative)
                    row_scale = case.B0
                matrix[row, column] = projection / row_scale
        matrices.append(matrix)
    return matrices[0], tuple(matrices), evidence


def _branch_eigenpair(
    case: HallCase, *, hall_sign: int = 1
) -> tuple[np.ndarray, np.ndarray, complex]:
    matrix = cold_transverse_matrix(case, hall_sign=hall_sign)
    values, vectors = np.linalg.eig(matrix)
    positive = [
        index
        for index, value in enumerate(values)
        if value.imag > 0.0
    ]
    if not positive:
        raise ValueError("cold Hall matrix has no positive-frequency branch")
    index = max(positive, key=lambda item: float(values[item].imag))
    eigenvalue = complex(values[index])
    vector = np.asarray(vectors[:, index], dtype=np.complex128).copy()
    pivot = int(np.argmax(np.abs(vector)))
    vector *= np.exp(-1j * np.angle(vector[pivot]))
    vector /= np.max(np.abs(vector))
    return vector, matrix, eigenvalue


def initial_hall_eigenmode(
    case: HallCase,
    *,
    relative_amplitude: float = 2.0e-6,
    hall_sign: int = 1,
    device: str = "cpu",
    context: Context | None = None,
) -> tuple[State, np.ndarray, complex]:
    """Create a real cosine/sine realization of one independent eigenmode."""
    if not math.isfinite(relative_amplitude) or relative_amplitude <= 0.0:
        raise ValueError("relative_amplitude must be positive and finite")
    vector, _, eigenvalue = _branch_eigenpair(case, hall_sign=hall_sign)
    if context is None:
        context = build_periodic_context(case, device=device)
    state = uniform_state(context, case)
    coordinate = (
        2.0
        * math.pi
        * np.arange(case.points, dtype=np.float64)
        / float(case.points)
    )
    shape = (
        np.real(vector)[:, None] * np.cos(coordinate)[None, :]
        - np.imag(vector)[:, None] * np.sin(coordinate)[None, :]
    )
    nc = case.points
    scale = MI * case.n0 * case.v_alfven
    xp = context.compute.xp
    state.fluid[0].mom[:nc] += (
        xp.asarray(relative_amplitude * scale * shape[0])
    )
    state.fluid[0].mom[nc : 2 * nc] += (
        xp.asarray(relative_amplitude * scale * shape[1])
    )
    state.B[:nc] += xp.asarray(relative_amplitude * case.B0 * shape[2])
    state.B[nc : 2 * nc] += xp.asarray(
        relative_amplitude * case.B0 * shape[3]
    )
    return state, vector, eigenvalue


def _integrator_polynomial(integrator: str, argument: complex) -> complex:
    if integrator == "midpoint":
        return 1.0 + argument + 0.5 * argument**2
    if integrator == "rk4":
        return (
            1.0
            + argument
            + 0.5 * argument**2
            + argument**3 / 6.0
            + argument**4 / 24.0
        )
    raise ValueError(f"unsupported integrator: {integrator}")


def _state_transverse_mode(
    state: State,
    case: HallCase,
    context: Context,
) -> np.ndarray:
    nc = case.points
    momentum = context.compute.to_host(state.fluid[0].mom)
    magnetic = context.compute.to_host(state.B)
    velocity_scale = MI * case.n0 * case.v_alfven
    return np.asarray(
        [
            project_component(momentum, 0, case) / velocity_scale,
            project_component(momentum, 1, case) / velocity_scale,
            project_component(magnetic, 0, case) / case.B0,
            project_component(magnetic, 1, case) / case.B0,
        ],
        dtype=np.complex128,
    )


def _runtime_finite(context: Context, state: State) -> bool:
    """Reduce retained device state with its owning backend."""
    xp = context.compute.xp
    return bool(
        xp.all(xp.isfinite(state.B)).item()
        and xp.all(xp.isfinite(state.fluid[0].mom)).item()
    )


def run_hall_time_case(
    case: HallCase,
    *,
    integrator: str,
    steps_per_cycle: int,
    cycles: int,
    relative_amplitude: float,
    oracle_hall_sign: int = 1,
    device: str = "cpu",
) -> HallTimeResult:
    """Advance the real source transaction and compare its retained eigenmode."""
    if integrator not in ("midpoint", "rk4"):
        raise ValueError("integrator must be midpoint or rk4")
    if steps_per_cycle < 32 or cycles < 1:
        raise ValueError("time fixture requires >=32 steps/cycle and >=1 cycle")
    vector, _, eigenvalue = _branch_eigenpair(
        case, hall_sign=oracle_hall_sign
    )
    omega = float(eigenvalue.imag)
    final_time = cycles * 2.0 * math.pi / omega
    steps = steps_per_cycle * cycles
    dt = final_time / float(steps)
    context = build_periodic_context(case, device=device)
    initial, initial_vector, initial_value = initial_hall_eigenmode(
        case,
        relative_amplitude=relative_amplitude,
        hall_sign=oracle_hall_sign,
        context=context,
    )
    initial_vector = vector
    initial_value = eigenvalue
    context.p.integrator = integrator
    context.p.nsteps = steps
    context.p.dt = dt
    # The production estimator uses an operator L1 scale, which is much more
    # conservative than the independently matched retained eigenfrequency.
    # Keep its warning, never its physical gates, visible for this bounded run.
    context.p.dt_guard = "warn"
    cumulative = cumulative_record()
    committed = initialize(initial, context, cumulative, equilibrium=False)
    initial_mode = _state_transverse_mode(committed.state, case, context)
    initial_reference = relative_amplitude * initial_vector
    initial_mode_error = float(
        np.linalg.norm(initial_mode - initial_reference)
        / max(np.linalg.norm(initial_reference), 1.0e-300)
    )
    diagnostics: list[Any] = []
    accepted = 0
    source_is_none = True
    failure_code = None
    failure_message = ""
    warning = ""
    dt_reports: list[Any] = []
    for _ in range(steps):
        try:
            result = advance(committed, cumulative, context, None)
        except Failure as failure:
            failure_code = failure.code
            failure_message = str(failure)
            diagnostics.extend(failure.checks)
            break
        diagnostics.extend(result.checks)
        if result.time_step_report is not None:
            dt_reports.append(result.time_step_report)
        accepted += 1
    if failure_code is None and dt_reports:
        last_report = dt_reports[-1]
        if not last_report.get("accepted", False):
            warning = "; ".join(last_report.get("reasons", ()))
    if failure_code is None and not dt_reports:
        raise RuntimeError("production time advance did not emit a dt report")
    actual = _state_transverse_mode(committed.state, case, context)
    amplification = _integrator_polynomial(
        integrator, initial_value * dt
    )
    expected = (
        relative_amplitude
        * amplification**accepted
        * initial_vector
    )
    relative_error = float(
        np.linalg.norm(actual - expected)
        / max(np.linalg.norm(expected), 1.0e-300)
    )
    projected = np.vdot(initial_vector, actual)
    reference = np.vdot(initial_vector, expected)
    amplitude_ratio = float(abs(projected) / max(abs(reference), 1.0e-300))
    phase_error = float(np.angle(projected / reference))
    continuum = (
        relative_amplitude
        * np.exp(initial_value * final_time)
        * initial_vector
    )
    continuum_relative_error = float(
        np.linalg.norm(actual - continuum)
        / max(np.linalg.norm(continuum), 1.0e-300)
    )
    continuum_projected = np.vdot(initial_vector, actual)
    continuum_reference = np.vdot(initial_vector, continuum)
    continuum_amplitude_ratio = float(
        abs(continuum_projected)
        / max(abs(continuum_reference), 1.0e-300)
    )
    continuum_phase_error = float(
        np.angle(continuum_projected / continuum_reference)
    )
    hard = [record for record in diagnostics if record.hard]
    hard_ok = failure_code is None and bool(hard) and all(
        record.available and record.passed for record in hard
    )
    finite = bool(
        np.all(np.isfinite(actual))
        and _runtime_finite(context, committed.state)
    )
    return HallTimeResult(
        integrator=integrator,
        requested_steps=steps,
        accepted_steps=accepted,
        dt=dt,
        final_time=final_time,
        omega=omega,
        amplification=amplification,
        actual_mode=actual,
        expected_mode=expected,
        relative_error=relative_error,
        continuum_relative_error=continuum_relative_error,
        amplitude_ratio=amplitude_ratio,
        phase_error_rad=phase_error,
        continuum_amplitude_ratio=continuum_amplitude_ratio,
        continuum_phase_error_rad=continuum_phase_error,
        initial_mode_error=initial_mode_error,
        finite=finite,
        hard_checks_available=hard_ok,
        source_is_none=source_is_none,
        diagnostics=tuple(diagnostics),
        dt_reports=tuple(dt_reports),
        failure_code=failure_code,
        failure_message=failure_message,
        warning=warning,
    )


def _closed_context(
    nodes: int, *, K: int = 1, device: str = "cpu"
) -> Context:
    if nodes < 4:
        raise ValueError("closed spatial fixture requires at least 4 cells")
    case = HallCase(kdi=1.0)
    parameters = _parameters(case)
    parameters.K = K
    parameters.Nphi = 4 * K + 1
    parameters.inner_distance = 0.2
    h = 1.0 / float(nodes)
    geometry = build_geometry(
        1.0 + (np.arange(nodes, dtype=np.float64) + 0.5) * h,
        (np.arange(nodes, dtype=np.float64) + 0.5) * h,
        [
            Point2(1.0, 0.0),
            Point2(2.0, 0.0),
            Point2(2.0, 1.0),
            Point2(1.0, 1.0),
        ],
    )
    backend = Backend(device)
    operators = build_operators(
        geometry,
        K,
        backend=backend,
    )
    return Context(
        p=parameters,
        geometry=geometry,
        op=operators,
        compute=backend,
        N=parameters.Nphi,
        next_generation=2,
    )


def _region_masks(context: Context) -> dict[str, np.ndarray]:
    distances = np.asarray(context.geometry.cell_wall_distance)
    return {
        "all": np.ones(distances.size, dtype=bool),
        "inner": distances > context.p.inner_distance,
        "wall": distances <= context.p.inner_distance,
    }


def _relative_region_errors(
    actual: np.ndarray,
    expected: np.ndarray,
    context: Context,
    *,
    reference_scale: float | None = None,
) -> dict[str, float]:
    nc = context.op.Nc
    residual = np.asarray(actual).reshape(context.N, -1, nc)
    reference = np.asarray(expected).reshape(context.N, -1, nc)
    denominator = max(
        float(np.linalg.norm(reference))
        if reference_scale is None
        else reference_scale,
        1.0e-300,
    )
    result = {}
    for name, mask in _region_masks(context).items():
        if not np.any(mask):
            result[name] = math.nan
            continue
        local_actual = residual[:, :, mask]
        local_reference = reference[:, :, mask]
        result[name] = float(
            np.linalg.norm(local_actual - local_reference)
            / denominator
        )
    return result


def _region_norms(
    actual: np.ndarray,
    expected: np.ndarray,
    masks: dict[str, np.ndarray],
    weights: np.ndarray,
    *,
    unit: str,
    relative_reference: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Return weighted norms and explicit availability for each region."""
    residual = np.asarray(actual).reshape(
        context_rows(actual), -1, context_columns(actual)
    )
    reference = np.asarray(expected).reshape(residual.shape)
    scale_reference = (
        reference
        if relative_reference is None
        else np.asarray(relative_reference).reshape(residual.shape)
    )
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weight.size != residual.shape[2]:
        raise ValueError("region weights do not match cell entities")
    output: dict[str, dict[str, Any]] = {}
    for name, mask in masks.items():
        selected = np.asarray(mask, dtype=bool).reshape(-1)
        count = int(np.count_nonzero(selected))
        if count == 0:
            output[name] = {
                "available": False,
                "entities": 0,
                "linf": None,
                "weighted_relative_l2": None,
                "unit": unit,
                "reason": "empty region",
            }
            continue
        local_residual = (
            residual[:, :, selected]
            - reference[:, :, selected]
        )
        local_reference = scale_reference[:, :, selected]
        local_weight = weight[selected]
        difference_l2 = math.sqrt(
            float(
                np.sum(
                    np.abs(local_residual) ** 2
                    * local_weight[None, None, :]
                )
            )
        )
        reference_l2 = math.sqrt(
            float(
                np.sum(
                    np.abs(local_reference) ** 2
                    * local_weight[None, None, :]
                )
            )
        )
        output[name] = {
            "available": True,
            "entities": count,
            "linf": float(np.max(np.abs(local_residual))),
            "weighted_relative_l2": (
                difference_l2 / reference_l2
                if reference_l2 > 0.0
                else (0.0 if difference_l2 == 0.0 else math.inf)
            ),
            "unit": unit,
            "reference_weighted_l2": reference_l2,
            "difference_weighted_l2": difference_l2,
            "reason": "",
        }
    return output


def context_rows(values: np.ndarray) -> int:
    return int(np.asarray(values).shape[0])


def context_columns(values: np.ndarray) -> int:
    shape = np.asarray(values).shape
    if len(shape) != 3:
        raise ValueError("region norm input must have shape [N,C,Nc]")
    return int(shape[2])


def _edge_masks(context: Context) -> dict[str, np.ndarray]:
    distances = []
    for table in (
        context.geometry.r_faces,
        context.geometry.cells,
        context.geometry.z_faces,
    ):
        distances.extend(
            diagnostics_module._entity_wall_distances(table, context)
        )
    values = np.asarray(distances, dtype=np.float64)
    if values.size != context.op.Ne:
        raise ValueError("closed edge region mask shape mismatch")
    return {
        "all": np.ones(values.size, dtype=bool),
        "inner": values > context.p.inner_distance,
        "wall": values <= context.p.inner_distance,
    }


def _pressure_bump_derivatives(
    radial: np.ndarray,
    axial: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a wall-zero pressure bump and its analytic R, phi, Z derivatives."""
    r = np.asarray(radial, dtype=np.float64).reshape(1, -1)
    z = np.asarray(axial, dtype=np.float64).reshape(1, -1)
    angle = np.asarray(phi, dtype=np.float64).reshape(-1, 1)
    sx = np.sin(math.pi * (r - 1.0))
    sz = np.sin(math.pi * z)
    modulation = 0.4 * (0.16 * np.cos(angle) + 0.05 * np.sin(angle))
    radial_derivative = (
        modulation
        * 2.0
        * math.pi
        * sx
        * np.cos(math.pi * (r - 1.0))
        * sz**2
    )
    azimuthal_derivative = (
        0.4
        * (-0.16 * np.sin(angle) + 0.05 * np.cos(angle))
        * sx**2
        * sz**2
    )
    axial_derivative = (
        modulation * sx**2 * 2.0 * math.pi * sz * np.cos(math.pi * z)
    )
    value = modulation * sx**2 * sz**2
    shape = (angle.shape[0], max(r.shape[1], z.shape[1]))
    return (
        np.broadcast_to(value, shape),
        np.broadcast_to(radial_derivative, shape),
        np.broadcast_to(azimuthal_derivative, shape),
        np.broadcast_to(axial_derivative, shape),
    )


def _independent_constraint(
    nodal: np.ndarray,
    density: np.ndarray,
    context: Context,
) -> np.ndarray:
    """Apply the design's T projection without calling its source helper."""
    xp = context.compute.xp
    nc = context.op.Nc
    m0 = xp.real(xp.asarray(context.op.M0)).astype(xp.float64)
    T = xp.asarray(context.op.T, dtype=xp.complex128)
    T_h = xp.conj(T.T)
    values = xp.asarray(nodal, dtype=xp.complex128).reshape(
        context.N, 3 * nc
    )
    density_values = xp.real(xp.asarray(density)).reshape(context.N, nc)
    output = xp.empty_like(values)
    if T.shape[0] == 0:
        return xp.asarray(values).reshape(-1)
    for sample in range(context.N):
        winv = xp.tile(density_values[sample] / m0, 3)
        weighted = T * winv[None, :]
        normal = weighted @ T_h
        rhs = T @ values[sample]
        multiplier = xp.linalg.lstsq(
            normal,
            rhs,
            rcond=context.p.rank_rtol,
        )[0]
        output[sample] = values[sample] - winv * (T_h @ multiplier)
    return output.reshape(-1)


def _cross_velocity_magnetic(
    velocity: np.ndarray,
    magnetic: np.ndarray,
    context: Context,
) -> np.ndarray:
    xp = context.compute.xp
    nc = context.op.Nc
    u = xp.asarray(velocity, dtype=xp.complex128).reshape(
        context.N, 3, nc
    )
    b = xp.asarray(magnetic, dtype=xp.complex128).reshape(
        context.N, 3, nc
    )
    return xp.stack(
        (
            u[:, 1, :] * b[:, 2, :] - u[:, 2, :] * b[:, 1, :],
            u[:, 2, :] * b[:, 0, :] - u[:, 0, :] * b[:, 2, :],
            u[:, 0, :] * b[:, 1, :] - u[:, 1, :] * b[:, 0, :],
        ),
        axis=1,
    ).reshape(-1)


def closed_spatial_evidence(
    *,
    nodes: int = 5,
    compressible: bool,
    device: str = "cpu",
) -> SpatialEvidence:
    """Measure smooth reconstruction on the actual closed RZ/Fourier operators."""
    context = _closed_context(nodes, K=1, device=device)
    xp = context.compute.xp
    nc = context.op.Nc
    nr = len(context.geometry.r_faces.keys)
    z_min = min(point.Z for point in context.geometry.polygon)
    z_max = max(point.Z for point in context.geometry.polygon)
    z_length = z_max - z_min
    phi = 2.0 * math.pi * np.arange(context.N) / float(context.N)
    cell_r = np.asarray(context.geometry.cells.R, dtype=np.float64)
    cell_z = np.asarray(context.geometry.cells.Z, dtype=np.float64)
    cosine_phi = np.cos(phi)[:, None]
    sine_phi = np.sin(phi)[:, None]
    cosine_twophi = np.cos(2.0 * phi)[:, None]
    sine_twophi = np.sin(2.0 * phi)[:, None]
    z_phase = (
        2.0 * math.pi * (cell_z[None, :] - z_min) / z_length
    )
    density_modulation = (
        1.0 + 0.18 * cosine_phi + 0.07 * sine_phi
    )
    density = 1.0e16 * density_modulation * (
        1.0 + 0.10 * np.cos(z_phase)
    )
    (
        pressure_variation,
        radial_pressure,
        azimuthal_pressure,
        z_derivative,
    ) = _pressure_bump_derivatives(cell_r, cell_z, phi)
    pressure = 0.4 + pressure_variation
    velocity_scale = 7.0e4
    axial_modulation = 1.0 + 0.15 * np.cos(z_phase)
    if compressible:
        toroidal_profile = (
            cosine_phi + 0.30 * cosine_twophi + 0.14 * sine_phi
        )
    else:
        toroidal_profile = np.ones_like(cosine_phi)
    u_phi = velocity_scale * toroidal_profile * axial_modulation
    velocity = np.zeros(
        (context.N, 3, nc), dtype=np.complex128
    )
    velocity[:, 1, :] = u_phi
    face_density = _component_flux(
        xp.asarray(density.reshape(-1)), context
    )
    face_density = context.compute.to_host(face_density).reshape(
        context.N, context.op.Ne
    )
    flux = np.zeros(
        (context.N, context.op.Ne), dtype=np.complex128
    )
    flux[:, nr : nr + nc] = (
        face_density[:, nr : nr + nc] * u_phi
    )
    quotient = flux / face_density
    fluid = SimpleNamespace(
        n=xp.asarray(density.reshape(-1)),
        nf=xp.asarray(face_density.reshape(-1)),
        u=xp.asarray(velocity.reshape(-1)),
    )
    gradient = velocity_gradient(
        fluid, fluid.u, context
    )
    gradient = context.compute.to_host(gradient).reshape(
        context.N, 3, 3, nc
    )
    theta = (
        gradient[:, 0, 0, :]
        + gradient[:, 1, 1, :]
        + gradient[:, 2, 2, :]
    )
    if compressible:
        exact_theta = velocity_scale * axial_modulation / cell_r[None, :] * (
            -sine_phi
            - 0.60 * sine_twophi
            + 0.14 * cosine_phi
        )
    else:
        exact_theta = np.zeros_like(theta)
    theta_scale = max(
        abs(exact_theta).max(),
        velocity_scale / float(np.min(cell_r)),
    )
    theta_error = float(
        np.linalg.norm(theta - exact_theta) / theta_scale
    )

    quotient_full_theta = _sampled_divergence(
        xp.asarray(quotient.reshape(-1)), context
    )
    quotient_full_theta = context.compute.to_host(
        quotient_full_theta
    ).reshape(context.N, nc)
    full_theta_error = float(
        np.linalg.norm(quotient_full_theta - exact_theta) / theta_scale
    )
    # The repaired source applies the full padded divergence to Fe/nf.  Keep
    # this operator-level call synchronized with `_electron_fluid`.
    retained_theta = quotient_full_theta.copy()
    retained_delta = theta - retained_theta
    theta_errors = _relative_region_errors(
        theta / theta_scale,
        exact_theta / theta_scale,
        context,
        reference_scale=1.0,
    )
    full_theta_errors = _relative_region_errors(
        quotient_full_theta / theta_scale,
        exact_theta / theta_scale,
        context,
        reference_scale=1.0,
    )
    retained_delta_errors = _relative_region_errors(
        retained_delta / theta_scale,
        np.zeros_like(exact_theta),
        context,
        reference_scale=1.0,
    )
    flux_hat = forward(
        xp.asarray(flux.reshape(-1)), context.op.Ne, context
    )
    retained_flux_divergence = inverse(
        mode_apply(context.op.DJ, flux_hat, context.compute),
        nc,
        context,
        True,
    )
    retained_flux_divergence = context.compute.to_host(
        retained_flux_divergence
    ).reshape(context.N, nc)
    wrong_djfe = retained_flux_divergence / density
    wrong_errors = _relative_region_errors(
        wrong_djfe / theta_scale,
        exact_theta / theta_scale,
        context,
        reference_scale=1.0,
    )

    tensor = np.zeros(
        (context.N, 3, 3, nc), dtype=np.complex128
    )
    for component in range(3):
        tensor[:, component, component, :] = pressure_variation
    pressure_force = gradient_adjoint(
        fluid, xp.asarray(tensor.reshape(-1)), context
    )
    pressure_force = context.compute.to_host(pressure_force).reshape(
        context.N, 3, nc
    )
    azimuthal_pressure = azimuthal_pressure / cell_r[None, :]
    analytic_pressure_force = -np.stack(
        (
            radial_pressure,
            azimuthal_pressure,
            z_derivative,
        ),
        axis=1,
    )
    pressure_norms = _region_norms(
        pressure_force,
        analytic_pressure_force,
        _region_masks(context),
        context.compute.to_host(context.op.M0),
        unit="Pa/m",
    )
    pressure_errors = {
        name: (
            float(values["weighted_relative_l2"])
            if values["available"]
            else math.nan
        )
        for name, values in pressure_norms.items()
    }

    magnetic = np.zeros(
        (context.N, 3, nc), dtype=np.complex128
    )
    magnetic[:, 2, :] = 0.1
    electron_charge = float(context.p.species[1].charge)
    e = -electron_charge
    cross_actual = xp.asarray(
        _cross_velocity_magnetic(velocity.reshape(-1), magnetic.reshape(-1), context)
    ).reshape(context.N, 3, nc)
    electron = SimpleNamespace(
        pressure_force=xp.asarray(pressure_force.reshape(-1))
    )
    derived = SimpleNamespace(
        electron_density=xp.asarray(density.reshape(-1)),
        electron_velocity=xp.asarray(velocity.reshape(-1)),
        BN=xp.asarray(magnetic.reshape(-1)),
    )
    actual_e0 = context.compute.to_host(
        _massless_e0(electron, derived, context)
    )

    independent_source = xp.asarray(
        analytic_pressure_force / e,
        dtype=np.complex128,
    )
    independent_source -= xp.asarray(density[:, None, :]) * cross_actual
    projected = _independent_constraint(
        independent_source.reshape(-1),
        density.reshape(-1),
        context,
    ).reshape(context.N, 3, nc)
    projected_velocity = projected / xp.asarray(
        density[:, None, :]
    )
    edge_velocity = sample_apply(
        context.op.I,
        xp.asarray(projected_velocity.reshape(-1)),
        context.N,
        context.compute,
    )
    edge_velocity = xp.asarray(edge_velocity).reshape(
        context.N, context.op.Ne
    )
    expected_e0 = forward(
        edge_velocity.reshape(-1), context.op.Ne, context
    )
    expected_e0 = context.compute.to_host(expected_e0)
    actual_e0_edges = context.compute.to_host(
        inverse(
            context.compute.asarray(actual_e0),
            context.op.Ne,
            context,
            True,
            context.p.ref.electric,
        )
    )
    expected_e0_edges = context.compute.to_host(
        inverse(
            context.compute.asarray(expected_e0),
            context.op.Ne,
            context,
            True,
            context.p.ref.electric,
        )
    )
    ohm_norms = _region_norms(
        actual_e0_edges.reshape(context.N, 1, context.op.Ne),
        expected_e0_edges.reshape(context.N, 1, context.op.Ne),
        _edge_masks(context),
        context.compute.to_host(context.op.M1),
        unit="V/m",
    )
    ohm_regions = {
        name: (
            float(values["weighted_relative_l2"])
            if values["available"]
            else math.nan
        )
        for name, values in ohm_norms.items()
    }
    return SpatialEvidence(
        minimum_density=float(xp.min(density).item()),
        maximum_density=float(xp.max(density).item()),
        theta_relative_error=theta_errors,
        full_theta_relative_error=full_theta_errors,
        retained_delta_relative_error=retained_delta_errors,
        wrong_djfe_relative_error=wrong_errors,
        pressure_force_error=pressure_errors,
        pressure_norms=pressure_norms,
        ohm_error=ohm_regions,
        ohm_norms=ohm_norms,
        arrays={
            "density": density.reshape(-1),
            "pressure": pressure.reshape(-1),
            "velocity": velocity.reshape(-1),
            "flux": flux.reshape(-1),
            "theta": theta.reshape(-1),
            "theta_exact": exact_theta.reshape(-1),
            "theta_retained": retained_theta.reshape(-1),
            "pressure_force": pressure_force.reshape(-1),
            "pressure_force_exact": analytic_pressure_force.reshape(-1),
            "E0": actual_e0,
            "E0_exact": expected_e0,
        },
    )


def reconstruction_evidence(
    *,
    nodes: int = 4,
    device: str = "cpu",
) -> ReconstructionEvidence:
    """Drive the actual QHFe/ne path from smooth flux and nonzero current."""
    context = _closed_context(nodes, K=1, device=device)
    xp = context.compute.xp
    nc = context.op.Nc
    nr = len(context.geometry.r_faces.keys)
    nz = len(context.geometry.z_faces.keys)
    phi = 2.0 * math.pi * np.arange(context.N) / float(context.N)
    cell_r = np.asarray(context.geometry.cells.R, dtype=np.float64)
    cell_z = np.asarray(context.geometry.cells.Z, dtype=np.float64)
    density = (
        1.0e16
        * (1.0 + 0.18 * np.cos(phi)[:, None]
           + 0.07 * np.sin(phi)[:, None])
        * (1.0 + 0.10 * np.cos(2.0 * math.pi * cell_z))[None, :]
    )
    face_density_device = _component_flux(
        xp.asarray(density.reshape(-1)), context
    )
    face_density = context.compute.to_host(
        face_density_device
    ).reshape(context.N, context.op.Ne)

    velocity_scale = 7.0e4
    edge_velocity = np.zeros(
        (context.N, context.op.Ne), dtype=np.complex128
    )
    edge_tables = (
        (context.geometry.r_faces, 1.0, 0),
        (context.geometry.cells, 0.0, nr),
        (context.geometry.z_faces, 0.5, nr + nc),
    )
    for table, factor, offset in edge_tables:
        x = np.asarray(table.R, dtype=np.float64) - 1.0
        z = np.asarray(table.Z, dtype=np.float64)
        profile = (
            velocity_scale
            * factor
            * np.sin(math.pi * x) ** 2
            * np.sin(math.pi * z) ** 2
        )
        edge_velocity[:, offset : offset + table.R.size] = (
            profile[None, :]
        )
    flux_samples = face_density * edge_velocity
    flux_hat = forward(
        xp.asarray(flux_samples.reshape(-1)),
        context.op.Ne,
        context,
    )

    magnetic_edges = np.zeros(
        (context.N, context.op.Nb), dtype=np.complex128
    )
    vertex_count = len(context.geometry.vertices.keys)
    magnetic_tables = (
        (context.geometry.z_faces, 0),
        (context.geometry.vertices, nz),
        (context.geometry.r_faces, nz + vertex_count),
    )
    for table, offset in magnetic_tables:
        x = np.asarray(table.R, dtype=np.float64) - 1.0
        z = np.asarray(table.Z, dtype=np.float64)
        profile = (
            np.sin(math.pi * x) ** 2
            * np.sin(math.pi * z) ** 2
            * (1.0 + 0.20 * np.cos(phi))[:, None]
        )
        magnetic_edges[
            :, offset : offset + table.R.size
        ] = profile
    magnetic_edges[:, nz : nz + vertex_count] += 0.05
    magnetic_hat = forward(
        xp.asarray(magnetic_edges.reshape(-1)),
        context.op.Nb,
        context,
    )
    current_hat = mode_apply(
        context.op.Cdag, magnetic_hat, context.compute
    ) / context.p.mu0
    ion_flux_hat = (
        -float(context.p.species[1].charge) * flux_hat + current_hat
    ) / float(context.p.species[0].charge)
    relation = (
        float(context.p.species[0].charge) * ion_flux_hat - current_hat
    ) / (-float(context.p.species[1].charge)) - flux_hat
    current_relation_error = float(
        np.linalg.norm(context.compute.to_host(relation))
        / max(float(np.linalg.norm(context.compute.to_host(flux_hat))), 1.0e-300)
    )
    current_samples = context.compute.to_host(
        inverse(
            current_hat,
            context.op.Ne,
            context,
            True,
            context.p.ref.current,
        )
    )

    electron_flux_samples = inverse(
        flux_hat,
        context.op.Ne,
        context,
        True,
    )
    h_flux = sample_apply(
        context.op.H,
        electron_flux_samples,
        context.N,
        context.compute,
    )
    constrained = _constraint_project(
        h_flux,
        xp.asarray(density.reshape(-1)),
        context,
    )
    density_device = xp.asarray(density.reshape(-1))
    velocity_device = constrained / xp.tile(
        xp.asarray(density[:, None, :]), (1, 3, 1)
    ).reshape(-1)
    velocity = context.compute.to_host(velocity_device).reshape(
        context.N, 3, nc
    )
    expected_velocity = np.zeros_like(velocity)
    expected_velocity[:, 0, :] = (
        velocity_scale
        * np.sin(math.pi * (cell_r - 1.0))[None, :] ** 2
        * np.sin(math.pi * cell_z)[None, :] ** 2
    )
    expected_velocity[:, 2, :] = 0.5 * expected_velocity[:, 0, :]

    fluid = SimpleNamespace(
        n=density_device,
        nf=face_density_device,
        u=velocity_device,
    )
    gradient = context.compute.to_host(
        velocity_gradient(fluid, velocity_device, context)
    ).reshape(context.N, 3, 3, nc)
    theta = (
        gradient[:, 0, 0, :]
        + gradient[:, 1, 1, :]
        + gradient[:, 2, 2, :]
    )
    exact_theta = np.broadcast_to(
        velocity_scale
        * (
            2.0
            * math.pi
            * np.sin(math.pi * (cell_r - 1.0))
            * np.cos(math.pi * (cell_r - 1.0))
            * np.sin(math.pi * cell_z) ** 2
            + np.sin(math.pi * (cell_r - 1.0)) ** 2
            * np.sin(math.pi * cell_z) ** 2
            / cell_r
            + math.pi
            * np.sin(math.pi * (cell_r - 1.0)) ** 2
            * np.sin(math.pi * cell_z)
            * np.cos(math.pi * cell_z)
        )[None, :],
        (context.N, nc),
    )
    quotient = flux_samples / face_density
    quotient_theta = context.compute.to_host(
        _sampled_divergence(xp.asarray(quotient.reshape(-1)), context)
    ).reshape(context.N, nc)
    delta_theta = theta - quotient_theta
    masks = _region_masks(context)
    weights = context.compute.to_host(context.op.M0)
    velocity_norms = _region_norms(
        velocity, expected_velocity, masks, weights, unit="m/s"
    )
    theta_norms = _region_norms(
        theta.reshape(context.N, 1, nc),
        exact_theta.reshape(context.N, 1, nc),
        masks,
        weights,
        unit="1/s",
    )
    delta_theta_norms = _region_norms(
        delta_theta.reshape(context.N, 1, nc),
        np.zeros((context.N, 1, nc), dtype=np.complex128),
        masks,
        weights,
        unit="1/s",
        relative_reference=exact_theta.reshape(context.N, 1, nc),
    )

    wall_flux_values = context.compute.to_host(
        sample_apply(
            context.op.T_full,
            constrained,
            context.N,
            context.compute,
        )
    ).reshape(context.N, len(context.geometry.walls))
    maximum_wall_flux = float(np.max(np.abs(wall_flux_values)))
    constrained_linf = float(
        np.max(np.abs(context.compute.to_host(constrained)))
    )
    relative_wall_flux = maximum_wall_flux / max(
        constrained_linf, 1.0e-300
    )

    return ReconstructionEvidence(
        nodes=nodes,
        h=1.0 / float(nodes),
        minimum_density=float(np.min(np.real(density))),
        maximum_density=float(np.max(np.real(density))),
        maximum_abs_current=float(np.max(np.abs(current_samples))),
        current_relation_error=current_relation_error,
        velocity_norms=velocity_norms,
        theta_norms=theta_norms,
        delta_theta_norms=delta_theta_norms,
        maximum_wall_flux=maximum_wall_flux,
        relative_wall_flux=relative_wall_flux,
        arrays={
            "density": density.reshape(-1),
            "flux": flux_samples.reshape(-1),
            "current": current_samples.reshape(-1),
            "velocity": velocity.reshape(-1),
            "expected_velocity": expected_velocity.reshape(-1),
            "theta": theta.reshape(-1),
            "exact_theta": exact_theta.reshape(-1),
            "delta_theta": delta_theta.reshape(-1),
            "wall_flux": wall_flux_values.reshape(-1),
        },
    )


def variable_density_velocity_evidence(
    *,
    nodes: int = 6,
    device: str = "cpu",
) -> DensityLayoutEvidence:
    """Exercise the real K=1 variable-density QHFe/ne reconstruction."""
    context = _closed_context(nodes, K=1, device=device)
    xp = context.compute.xp
    nc = context.op.Nc
    n0 = 1.0e16
    state = closed_reference(context, n0, 0.4, 0.4, 0.1)
    phi = 2.0 * math.pi * np.arange(context.N) / float(context.N)
    density = n0 * (1.0 + 0.2 * np.cos(phi))[:, None] * np.ones(
        (1, nc)
    )
    pressure = 0.4 * (1.0 + 0.1 * np.cos(phi))[:, None] * np.ones(
        (1, nc)
    )
    velocity = np.zeros(
        (context.N, 3, nc), dtype=np.complex128
    )
    velocity[:, 0, :] = 1.0e4 * (
        1.0 + 0.1 * np.cos(phi)
    )[:, None]
    velocity[:, 1, :] = 3.0e4 * (
        1.0 + 0.2 * np.sin(phi)
    )[:, None]
    velocity[:, 2, :] = 2.0e4 * (
        1.0 + 0.1 * np.sin(2.0 * phi)
    )[:, None]
    for species in range(2):
        state.fluid[species].n[:] = forward(
            xp.asarray(density.reshape(-1)), nc, context
        )
        state.fluid[species].perp[:] = forward(
            xp.asarray(pressure.reshape(-1)), nc, context
        )
        if species == 1:
            state.fluid[species].parallel[:] = state.fluid[species].perp
    state.fluid[0].mom[:] = forward(
        xp.asarray((MI * density[:, None, :] * velocity).reshape(-1)),
        3 * nc,
        context,
    )
    derived = _derive_electron(state, context)
    physical_density = context.compute.to_host(
        derived.electron_density
    ).reshape(context.N, nc)
    electron_flux = inverse(
        derived.electron_flux_hat,
        context.op.Ne,
        context,
        True,
    )
    h_flux = sample_apply(
        context.op.H,
        electron_flux,
        context.N,
        context.compute,
    )
    constrained = _constraint_project(
        h_flux,
        derived.electron_density,
        context,
    )
    constrained = context.compute.to_host(constrained).reshape(
        context.N, 3, nc
    )
    denominator = np.repeat(
        physical_density[:, None, :],
        3,
        axis=1,
    )
    expected = constrained / denominator
    actual = context.compute.to_host(
        derived.electron_velocity
    ).reshape(context.N, 3, nc)
    residual = actual - expected
    relative = float(
        np.linalg.norm(residual)
        / max(float(np.linalg.norm(expected)), 1.0e-300)
    )
    return DensityLayoutEvidence(
        relative_error=relative,
        linf_error=float(np.max(np.abs(residual))),
        density_min=float(np.min(np.real(physical_density))),
        density_max=float(np.max(np.real(physical_density))),
        arrays={
            "density": physical_density.reshape(-1),
            "actual_velocity": actual.reshape(-1),
            "expected_velocity": expected.reshape(-1),
            "residual": residual.reshape(-1),
            "electron_flux": context.compute.to_host(
                derived.electron_flux
            ).reshape(-1),
        },
    )


def drift_ohm_tautology_evidence(
    *,
    device: str = "cpu",
) -> DriftOhmEvidence:
    """Double a nonzero solved E0 and evaluate the production hard gate."""
    case = HallCase(kdi=1.0)
    context = build_periodic_context(case, device=device)
    state = uniform_state(context, case)
    state = perturb_state(state, case, "Bx", 2.0e-6, context)
    result = evaluate(state, context)
    baseline_checks = [
        record
        for record in result.checks
        if record.name == "drift_ohm_residual"
    ]
    baseline = massless_checks._stage_residuals(
        result.canonical_state,
        result,
        context,
    )
    original = context.compute.to_host(result.E0)
    xp = context.compute.xp
    result.E0 = xp.asarray(original * 2.0)
    perturbed = massless_checks._stage_residuals(
        result.canonical_state,
        result,
        context,
    )
    result.residuals = perturbed
    residual = context.compute.to_host(
        perturbed["drift_ohm_residual"]
    )
    baseline_value = float(
        np.max(np.abs(context.compute.to_host(
            baseline["drift_ohm_residual"]
        )))
    )
    doubled_value = float(np.max(np.abs(residual)))
    and_record = next(
        record
        for record in baseline_checks
        if record.region == "all"
        and getattr(record.norm, "value", str(record.norm)) == "linf"
    )
    threshold = float(and_record.threshold)
    result.checks = []
    failure = None
    try:
        massless_checks.check_stage(
            result.canonical_state,
            result,
            context,
        )
    except Failure as error:
        failure = error
    doubled_checks = [
        record
        for record in getattr(failure, "checks", ())
        if record.name == "drift_ohm_residual"
        and record.region == "all"
        and getattr(record.norm, "value", str(record.norm)) == "linf"
    ]
    gate_failed = bool(
        failure is not None
        and doubled_checks
        and not doubled_checks[0].passed
    )
    return DriftOhmEvidence(
        electric_field_linf=float(np.max(np.abs(original))),
        baseline_linf=baseline_value,
        doubled_linf=doubled_value,
        hard_threshold=threshold,
        unit=and_record.unit,
        gate_failed=gate_failed,
        arrays={
            "E0": np.asarray(original),
            "changed_field": np.asarray(original),
            "baseline_residual": np.asarray(
                context.compute.to_host(
                    baseline["drift_ohm_residual"]
                )
            ),
            "perturbed_residual": residual,
            "doubled_field": np.asarray(original) * 2.0,
        },
    )
