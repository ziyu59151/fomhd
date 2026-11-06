"""Public records shared by the A0 numerical foundation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from typing import Any


class Norm(str, Enum):
    linf = "linf"
    weighted_l2 = "weighted_l2"
    value = "value"


@dataclass
class Species:
    id: str = ""
    mass: float = 0.0
    charge: float = 0.0
    gamma: float = 0.0
    n_min: float = 0.0
    face_min: float = 0.0
    p_min: float = 0.0
    parallel_min: float = 0.0


@dataclass
class References:
    length: float = 0.0
    time: float = 0.0
    ni: float = 0.0
    ne: float = 0.0
    velocity: float = 0.0
    electric: float = 0.0
    magnetic: float = 0.0
    current: float = 0.0
    charge: float = 0.0
    pressure: float = 0.0


@dataclass
class Budget:
    atol: float = 0.0
    rtol: float = 0.0
    reference: float = 0.0
    unit: str = ""
    norm: Norm = Norm.linf
    hard: bool = True


@dataclass
class Parameters:
    species: tuple[Species, Species] = field(
        default_factory=lambda: (Species(), Species())
    )
    epsilon0: float = 0.0
    mu0: float = 0.0
    B_min: float = 0.0
    dt: float = 0.0
    start_time: float = 0.0
    rank_rtol: float = 0.0
    nsteps: int = 0
    K: int = 0
    Nphi: int = 0
    memory_limit: int = 0
    device_memory_limit: int = 0
    inner_distance: float = 0.0
    ref: References = field(default_factory=References)
    budgets: dict[str, Budget] = field(default_factory=dict)
    electron_inertia: int = 0
    integrator: str = "rk4"
    dt_guard: str = "error"
    minimum_samples_per_cycle: float = 32.0
    max_amplitude_error: float = 0.01
    max_phase_error_rad: float = 0.1


@dataclass
class StageTag:
    step: int = 0
    generation: int = 0
    trial: int = 0
    spectral_context: int = 0
    time: float = 0.0
    stage: str = ""


@dataclass
class FluidState:
    n: Any = field(default_factory=list)
    mom: Any = field(default_factory=list)
    perp: Any = field(default_factory=list)
    parallel: Any = field(default_factory=list)


@dataclass
class State:
    fluid: tuple[FluidState, FluidState] = field(
        default_factory=lambda: (FluidState(), FluidState())
    )
    B: Any = field(default_factory=list)
    step: int = 0
    generation: int = 0
    time: float = 0.0


@dataclass
class Context:
    p: Parameters
    geometry: Any
    op: Any
    compute: Any
    N: int
    spectral_id: int = 0
    next_generation: int = 1
    next_trial: int = 1
    next_source: int = 1
    geometry_hash: str = ""
    operator_hash: str = ""
    rule_hash: str = ""


class Failure(RuntimeError):
    """Numerical or contract failure with native-compatible evidence."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.checks: list[SimpleNamespace] = []
        self.stage: StageTag | None = None
        self.partial_arrays: dict[str, Any] = {}


def check_record(**values: Any) -> SimpleNamespace:
    defaults = {
        "name": "",
        "population": "",
        "region": "all",
        "unit": "",
        "value_unit": "",
        "reason": "",
        "component": "",
        "producer": "diagnostics",
        "tag": StageTag(),
        "raw": 0.0,
        "reference": 0.0,
        "threshold": 0.0,
        "norm": Norm.linf,
        "hard": True,
        "available": True,
        "passed": False,
        "worst": 0,
        "selected": 0,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def linear_record(**values: Any) -> SimpleNamespace:
    defaults = {
        "x": None,
        "residual": None,
        "singular_values": None,
        "scaled_residual": 0.0,
        "backward_error": 0.0,
        "row_scale": None,
        "column_scale": None,
        "refinement_steps": 0,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def lu_record(**values: Any) -> SimpleNamespace:
    defaults = {
        "original": None,
        "factors": None,
        "pivots": None,
        "valid": False,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def operator_check_record(**values: Any) -> SimpleNamespace:
    defaults = {
        "name": "",
        "unit": "",
        "mode": 0,
        "residual": 0.0,
        "scale": 0.0,
        "roundoff_bound": 0.0,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def field_cache_record(**values: Any) -> SimpleNamespace:
    defaults = {
        "phi": None,
        "EL": None,
        "ET": None,
        "E": None,
        "multiplier": None,
        "poisson_residual": None,
        "tag": StageTag(),
        "poisson_solves": [],
        "transverse_solve": None,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def wall_sample_record(**values: Any) -> SimpleNamespace:
    defaults = {"S": None, "lu": None, "winv": None, "row_scale": 0.0, "col_scale": 0.0}
    defaults.update(values)
    return SimpleNamespace(**defaults)


def wall_context_record(**values: Any) -> SimpleNamespace:
    defaults = {"samples": [], "tag": StageTag()}
    defaults.update(values)
    return SimpleNamespace(**defaults)


def projection_record(**values: Any) -> SimpleNamespace:
    defaults = {
        "value": None,
        "reaction": None,
        "multiplier": None,
        "multiplier_residual": None,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def fluid_context_record(**values: Any) -> SimpleNamespace:
    names = (
        "n",
        "mom",
        "u",
        "perp",
        "parallel",
        "nf",
        "flux",
        "flux_hat",
        "divergence",
        "divergence_hat",
        "tensor",
        "raw_moment",
        "gradient",
        "theta",
        "strain",
        "convection",
        "pressure_force",
        "magnetic_force",
        "non_electric",
        "pdot_perp",
        "pdot_parallel",
        "pressure_flux",
        "parallel_flux",
        "wall",
    )
    defaults = {name: None for name in names}
    defaults["wall"] = wall_context_record()
    defaults.update(values)
    return SimpleNamespace(**defaults)


def fluid_terms_record(**values: Any) -> SimpleNamespace:
    defaults = {
        "electric": None,
        "force": None,
        "reaction": None,
        "multiplier": None,
        "multiplier_residual": None,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def ledger_record(**values: Any) -> SimpleNamespace:
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
    defaults = {name: 0.0 for name in names}
    defaults.update(values)
    return SimpleNamespace(**defaults)


def wall_trace_record(**values: Any) -> SimpleNamespace:
    defaults = {
        "B": [0.0, 0.0, 0.0],
        "EL": [0.0, 0.0, 0.0],
        "ET": [0.0, 0.0, 0.0],
        "velocity": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "density": [0.0, 0.0],
        "normal_flux": [0.0, 0.0],
        "perp": [0.0, 0.0],
        "parallel": [0.0, 0.0],
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def boundary_ledger_record(**values: Any) -> SimpleNamespace:
    defaults = {
        "traces": [],
        "number": [0.0, 0.0],
        "kinetic": [0.0, 0.0],
        "internal_perp": [0.0, 0.0],
        "internal_parallel": [0.0, 0.0],
        "field_energy": 0.0,
        "total_energy": 0.0,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def momentum_record(**values: Any) -> SimpleNamespace:
    defaults = {
        "value": [0.0, 0.0, 0.0],
        "outward": [0.0, 0.0, 0.0],
        "wall_reaction": [0.0, 0.0, 0.0],
        "angular": 0.0,
        "outward_torque": 0.0,
        "wall_torque": 0.0,
        "angular_samples": 0,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def totals_record(**values: Any) -> SimpleNamespace:
    defaults = {
        "number": [0.0, 0.0],
        "kinetic": [0.0, 0.0],
        "ion_internal": 0.0,
        "electron_perp": 0.0,
        "electron_parallel": 0.0,
        "magnetic": 0.0,
        "longitudinal": 0.0,
        "energy": 0.0,
        "topology": 0.0,
        "physical_flux": 0.0,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def stage_result_record(**values: Any) -> SimpleNamespace:
    defaults = {
        "tag": StageTag(),
        "rhs": State(),
        "cache": field_cache_record(),
        "BN": None,
        "bhat": None,
        "rho": None,
        "current": None,
        "fJ0": None,
        "ELdot": None,
        "phidot": None,
        "ampere": None,
        "AE": None,
        "fluid": (fluid_context_record(), fluid_context_record()),
        "terms": (fluid_terms_record(), fluid_terms_record()),
        "ledger": (ledger_record(), ledger_record()),
        "boundary": boundary_ledger_record(),
        "momentum": momentum_record(),
        "totals": totals_record(),
        "checks": [],
        "residuals": {},
        "source_id": 0,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def baseline_record(**values: Any) -> SimpleNamespace:
    defaults = {
        "divB": None,
        "totals": totals_record(),
        "momentum": momentum_record(),
        "time": 0.0,
        "step": 0,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def cumulative_record(**values: Any) -> SimpleNamespace:
    defaults = {
        "baseline": baseline_record(),
        "integrals": (ledger_record(), ledger_record()),
        "boundary_integrals": boundary_ledger_record(),
        "accepted_intervals": 0,
        "last_trial": 0,
        "last_step": 0,
        "last_time": 0.0,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def interval_sources_record(**values: Any) -> SimpleNamespace:
    defaults = {
        "number_flux": (None, None),
        "current": (None, None),
        "source_id": 0,
        "trial": 0,
        "left_step": 0,
        "right_step": 0,
        "weight": 0.0,
        "left_time": 0.0,
        "right_time": 0.0,
        "accepted": False,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def committed_record(**values: Any) -> SimpleNamespace:
    defaults = {"state": State(), "cache": field_cache_record(), "accepted": False}
    defaults.update(values)
    return SimpleNamespace(**defaults)


def step_result_record(**values: Any) -> SimpleNamespace:
    defaults = {
        "checks": [],
        "totals": totals_record(),
        "momentum": momentum_record(),
        "interval": interval_sources_record(),
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)
