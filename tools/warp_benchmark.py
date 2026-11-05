#!/usr/bin/env python3
"""Bounded finite-mass FOMHD pilot and explicit-integrator scan harness.

The harness uses the production case loader, closed-reference constructor,
initialization transaction, and explicit time transaction. It never replaces a
complete solver stage with a synthetic model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import signal
import shutil
import socket
import subprocess
import sys
import time
import tomllib
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SOURCE_ROOT = SCRIPT_REPO_ROOT / "src"
if SCRIPT_SOURCE_ROOT.is_dir() and str(SCRIPT_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_SOURCE_ROOT))


SCHEMA = "fomhd-warp-benchmark-v1"
INITIAL_DENSITY_FRACTION = 1.0e-6
EXPECTED_N = 1.0e12
EXPECTED_P = 1.0e-9
EXPECTED_B = 1.0e-5
EXPECTED_U = 1.0


class BudgetExhausted(RuntimeError):
    """The user-provided wall-clock budget expired."""


def _exception_chain(error: BaseException) -> list[BaseException]:
    """Return explicit cause/context exceptions in outer-to-root order."""
    ordered = []
    seen = set()

    def visit(current: BaseException | None) -> None:
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        ordered.append(current)
        visit(current.__cause__)
        visit(current.__context__)

    visit(error)
    return ordered


def _find_budget_exhaustion(error: BaseException) -> BudgetExhausted | None:
    for item in _exception_chain(error):
        if isinstance(item, BudgetExhausted):
            return item
    return None


@contextmanager
def _hard_wallclock(seconds: float):
    """Interrupt Python solver work at the user wall-clock boundary."""
    previous = signal.getsignal(signal.SIGALRM)

    def expire(signum, frame):
        raise BudgetExhausted(
            f"wall-clock budget of {seconds:g} s expired"
        )

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    kind: str
    grid: int
    K: int
    dt: float
    nsteps: int
    integrator: str
    target_time_s: float


@dataclass(frozen=True)
class ScanPlan:
    omega_plasma_rad_s: float
    target_time_s: float
    cases: tuple[CaseSpec, ...]
    assessments: tuple[dict[str, Any], ...]


@dataclass
class GridInputs:
    grid: int
    K: int
    directory: Path
    geometry_path: Path
    initial_path: Path
    initial_sha256: str
    geometry_sha256: str
    bootstrap_case: Path
    metadata: dict[str, Any]


def _solver_modules():
    from fomhd import core, io, stage
    from fomhd import stability
    from fomhd import time as solver_time
    from fomhd.types import Context, Failure

    return SimpleNamespace(
        core=core,
        io=io,
        stage=stage,
        stability=stability,
        solver_time=solver_time,
        Context=Context,
        Failure=Failure,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _typed(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _typed(value.item())
    if isinstance(value, complex):
        return {
            "real": _typed(value.real),
            "imag": _typed(value.imag),
            "type": "complex128",
        }
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return {"value": "NaN", "type": "float64"}
        return {
            "value": "+Inf" if value > 0 else "-Inf",
            "type": "float64",
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _typed(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_typed(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): _typed(item)
            for key, item in vars(value).items()
            if key != "traces"
        }
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        json.dump(
            _typed(value),
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")


def _write_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_root() -> Path:
    import fomhd

    package_root = Path(fomhd.__file__).resolve().parent
    if package_root.name != "fomhd" or package_root.parent.name != "src":
        raise RuntimeError(f"fomhd is not imported from src/fomhd: {package_root}")
    return package_root.parents[1]


def _source_hashes(root: Path) -> dict[str, str]:
    package = root / "src" / "fomhd"
    records = {}
    for path in sorted(package.glob("*.py")):
        records[f"src/fomhd/{path.name}"] = _sha256(path)
    for name in ("pyproject.toml", "uv.lock"):
        path = root / name
        if path.is_file():
            records[name] = _sha256(path)
    for relative, path in _tool_source_paths().items():
        records[relative] = _sha256(path)
    return records


def _tool_source_paths() -> dict[str, Path]:
    benchmark = Path(__file__).resolve()
    plot = benchmark.parent / "plots" / "plot_warp_benchmarks.py"
    if not plot.is_file():
        raise FileNotFoundError(f"plotting CLI source is missing: {plot}")
    return {
        "tools/warp_benchmark.py": benchmark,
        "tools/plots/plot_warp_benchmarks.py": plot,
    }


def _archive_tool_sources(output: Path) -> None:
    for relative, source in _tool_source_paths().items():
        target = output / "provenance" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_bytes = source.read_bytes()
        if target.read_bytes() != source_bytes:
            raise ValueError(f"archived tool source mismatch: {relative}")
        if _sha256(target) != hashlib.sha256(source_bytes).hexdigest():
            raise ValueError(f"archived tool source hash mismatch: {relative}")


def _git_provenance(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    toplevel = run("rev-parse", "--show-toplevel")
    if toplevel is None or Path(toplevel).resolve() != root.resolve():
        return {
            "commit": None,
            "status": None,
            "reason": "source is an archived snapshot, not a repository root",
        }
    return {"commit": run("rev-parse", "HEAD"), "status": run("status", "--short")}


def _dependency_provenance() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": _package_version("scipy"),
        "matplotlib": _package_version("matplotlib"),
        "cupy-cuda13x": _package_version("cupy-cuda13x"),
        "warp-lang": _package_version("warp-lang"),
        "pytest": _package_version("pytest"),
    }


def _base_provenance(command: str, argv: list[str], root: Path) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "initializing",
        "command": command,
        "argv": argv,
        "started_utc": _utc_now(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "dependencies": _dependency_provenance(),
        "source_root": str(root),
        "source_hashes": _source_hashes(root),
        "git": _git_provenance(root),
        "model": {
            "id": "A0",
            "variant": "finite_mass",
            "electron_inertia": 1,
            "precision": "float64/complex128",
            "scope": "controlled numerical benchmark; not experimental validation",
        },
    }


def _create_output(path: Path) -> Path:
    output = path.expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"output already exists; refusing overwrite: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError(
            f"output parent must be an existing real directory: {output.parent}"
        )
    try:
        output.mkdir(mode=0o700)
    except FileExistsError as error:
        raise FileExistsError(
            f"output already exists; refusing overwrite: {output}"
        ) from error
    return output


def _parse_int_list(text: str, name: str, *, minimum: int = 1) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be comma-separated integers") from error
    if not values or any(value < minimum for value in values):
        raise ValueError(f"{name} values must be >= {minimum}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate values")
    return values


def _parse_integrators(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values or any(value not in ("midpoint", "rk4") for value in values):
        raise ValueError("integrators must contain midpoint and/or rk4")
    if len(set(values)) != len(values):
        raise ValueError("integrators contains duplicates")
    return values


def _normalise_device(device: str) -> tuple[str, int]:
    if device == "cpu":
        return "cpu", 0
    match = re.fullmatch(r"cuda:(\d+)", device)
    if match is None:
        raise ValueError("device must be cpu or cuda:N")
    return "cuda", int(match.group(1))


def build_scan_plan(
    *,
    grid: int,
    K: int,
    cycles: float,
    nsteps_values: tuple[int, ...],
    integrators: tuple[str, ...],
    frequency_reference: dict[str, Any],
    minimum_samples: float = 32.0,
    amplitude_budget: float = 0.01,
    phase_budget: float = 0.1,
) -> ScanPlan:
    """Build the equal-final-horizon scan without constructing solver state."""
    if grid < 2 or K != 1:
        raise ValueError("benchmark grid must be >=2 and K must be 1")
    if not math.isfinite(cycles) or cycles <= 0:
        raise ValueError("cycles must be positive and finite")
    omega = float(frequency_reference["omega_plasma"])
    if not math.isfinite(omega) or omega <= 0:
        raise ValueError("invalid plasma-frequency reference")
    target = cycles * 2.0 * math.pi / omega
    if not math.isfinite(target) or target <= 0:
        raise ValueError("invalid target horizon")
    cases = []
    assessments = []
    for integrator in integrators:
        for nsteps in nsteps_values:
            dt = target / nsteps
            logical = dt * nsteps
            if abs(logical - target) > 32.0 * np.spacing(target):
                raise ArithmeticError("dt and nsteps do not preserve the target horizon")
            case_id = f"scan-{integrator}-n{nsteps:06d}-g{grid}-k{K}"
            cases.append(
                CaseSpec(
                    case_id=case_id,
                    kind="scan",
                    grid=grid,
                    K=K,
                    dt=dt,
                    nsteps=nsteps,
                    integrator=integrator,
                    target_time_s=target,
                )
            )
            from fomhd import stability

            report = stability.assess_dt(
                integrator,
                omega,
                dt,
                nsteps,
                minimum_samples=minimum_samples,
                amplitude_budget=amplitude_budget,
                phase_budget=phase_budget,
            )
            report["case_id"] = case_id
            report["logical_final_time_s"] = logical
            report["target_final_time_s"] = target
            report["relative_time_error"] = abs(logical - target) / target
            report["scope"] = (
                "cold scalar plasma-frequency estimate; explicit-method "
                "finite-horizon assessment, not a continuum validation"
            )
            assessments.append(report)
    return ScanPlan(omega, target, tuple(cases), tuple(assessments))


def hermite_density_pair(
    density: float, envelope: np.ndarray, amplitude: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return Hermitian +k and -k electron-density Fourier coefficients."""
    env = np.asarray(envelope, dtype=np.float64)
    if not math.isfinite(density) or density <= 0:
        raise ValueError("density must be positive and finite")
    if not math.isfinite(amplitude) or amplitude <= 0 or amplitude >= 0.5:
        raise ValueError("density perturbation fraction must be in (0, 0.5)")
    if env.ndim != 1 or not np.isfinite(env).all() or np.any(env < 0):
        raise ValueError("envelope must be a finite nonnegative vector")
    if np.max(env, initial=0.0) > 1.0 + 1e-14:
        raise ValueError("envelope must be bounded by one")
    positive = (
        0.5 * float(amplitude) * float(density) * env.astype(np.complex128)
    )
    return positive, np.conj(positive)


def _budget_checks() -> dict[str, dict[str, Any]]:
    """Return production finite-model check metadata with physical scales."""
    ion_mass = 1.67262192595e-27
    ion_charge = 1.602176634e-19
    electron_mass = 9.1093837139e-31
    electron_charge = -1.602176634e-19
    length = 0.2
    time_scale = 1.0e-6
    density = EXPECTED_N
    pressure = EXPECTED_P
    magnetic = EXPECTED_B
    mu0 = 1.25663706127e-6
    response = (
        ion_charge * ion_charge * density / ion_mass
        + electron_charge * electron_charge * density / electron_mass
    )
    current = time_scale * (
        response + 1.0 / (mu0 * length * length)
    )
    charge = 1.0e-7
    velocity = EXPECTED_U
    force = ion_mass * density * velocity / time_scale
    power = pressure * length**3 / time_scale
    energy = pressure * length**3 + magnetic * magnetic * length**3 / mu0
    checks: dict[str, dict[str, Any]] = {}

    def array(
        key: str,
        unit: str,
        reference: float,
        hard: bool = True,
    ) -> None:
        for norm in ("linf", "weighted_L2"):
            checks[f"{key}.{norm}"] = {
                "unit": unit,
                "norm": norm,
                "reference": reference,
                "hard": hard,
            }

    def value(
        key: str,
        unit: str,
        reference: float,
        hard: bool = True,
    ) -> None:
        checks[f"{key}.value"] = {
            "unit": unit,
            "norm": "value",
            "reference": reference,
            "hard": hard,
        }

    for key in ("gauss", "poisson_linear", "transverse_divergence"):
        array(key, "V/m^2", 1.0 / length)
    array("longitudinal_projection", "A/m^3", current / length)
    array("darwin_dynamic", "A/(m^2*s)", current / time_scale)
    for key in ("current_divB", "current_ampere"):
        array(key, "T/m", magnetic / length, False)
    for key in (
        "initial_divB",
        "initial_ampere",
        "divB_update",
        "long_term_divB_change",
    ):
        array(key, "T/m", magnetic / length)
    array("CT_increment", "T/(m*s)", magnetic / (length * time_scale))
    for key in ("CT_update", "equilibrium_B", "alias_B"):
        array(key, "T/s", magnetic / time_scale)
    array("current_response", "A/(m^2*s)", current / time_scale)
    for key in ("RC_flux", "RC_mom", "deltaC"):
        array(key, "C/(m^3*s)", charge / time_scale)
    array("RL", "A/m^2", current)
    for key in (
        "ion_internal",
        "electron_perp",
        "electron_parallel",
        "field_energy_B",
        "field_energy_L",
        "total_energy",
    ):
        value(key, "J", energy, False)
    value("field_exchange", "W", energy / time_scale, False)
    value("topology_increment", "Wb", magnetic * length * length)
    value("total_charge_balance", "C", charge * length**3)
    for key in ("momentum_x", "momentum_y", "momentum_z"):
        value(
            key,
            "kg*m/s",
            ion_mass * density * length**3 * velocity,
            False,
        )
    value(
        "angular_momentum",
        "kg*m^2/s",
        ion_mass * density * length**4 * velocity,
        False,
    )
    for suffix in (".ion", ".electron"):
        for key in (
            "stage_continuity",
            "species_continuity",
            "alias_n",
            "equilibrium_n",
        ):
            array(key + suffix, "1/(m^3*s)", density / time_scale)
        for key in (
            "wall_rhs",
            "wall_multiplier",
            "alias_mom",
            "equilibrium_mom",
        ):
            array(key + suffix, "kg/(m^2*s^2)", force)
        array("wall_state" + suffix, "kg/(m^2*s)", force * time_scale)
        array("theta_identity" + suffix, "1/s", 1.0 / time_scale)
        array(
            "wall_number_flux" + suffix,
            "1/(m^2*s)",
            density * velocity,
        )
        array("wall_perp_flux" + suffix, "Pa*m/s", pressure * velocity)
        if suffix == ".electron":
            array(
                "wall_parallel_flux" + suffix,
                "Pa*m/s",
                pressure * velocity,
            )
        for key in ("alias_perp", "equilibrium_perp"):
            array(key + suffix, "Pa/s", pressure / time_scale)
        if suffix == ".electron":
            for key in (
                "cgl_source",
                "alias_parallel",
                "equilibrium_parallel",
            ):
                array(key + suffix, "Pa/s", pressure / time_scale)
        for key in (
            "electric_pairing",
            "pressure_pairing",
            "magnetic_work",
            "wall_work",
            "convection_pairing",
        ):
            value(key + suffix, "W", power)
        value(
            "spectral_transfer_defect" + suffix,
            "W",
            power,
            False,
        )
        value(
            "total_number_balance" + suffix,
            "1",
            density * length**3,
        )
        value("kinetic_energy" + suffix, "J", energy, False)
    return checks


def _python_case_template() -> str:
    """Render a complete finite-mass case using only project-owned data."""
    mu0 = 1.25663706127e-6
    response = (
        1.602176634e-19**2 * EXPECTED_N / 1.67262192595e-27
        + 1.602176634e-19**2 * EXPECTED_N / 9.1093837139e-31
    )
    current = 1.0e-6 * (response + 1.0 / (mu0 * 0.2 * 0.2))
    values = {
        "schema_version": 1,
        "model": "A0",
        "units": "SI",
        "layout_version": 1,
        "case_name": "python-template",
    }
    lines = [f"{key} = {json.dumps(value) if isinstance(value, str) else value}"
             for key, value in values.items()]
    lines.extend(
        [
            "[species]",
            'ion_id = "ion"',
            'electron_id = "electron"',
            "qi = 1.602176634e-19",
            "mi = 1.67262192595e-27",
            "qe = -1.602176634e-19",
            "me = 9.1093837139e-31",
            "gamma_i = 1.6666666666666667",
            "electron_inertia = 1",
            "[grid]",
            "R0 = 1.0",
            "Z0 = 0.0",
            "hR = 0.2",
            "hZ = 0.2",
            "NR = 4",
            "NZ = 4",
            "K = 1",
            "Nphi = 5",
            "[time]",
            "start_time = 0.0",
            "dt = 2.0e-9",
            "nsteps = 1",
            'initial_kind = "non_equilibrium"',
            'integrator = "rk4"',
            'dt_guard = "warn"',
            "minimum_samples_per_cycle = 32.0",
            "max_amplitude_error = 0.01",
            "max_phase_error_rad = 0.1",
            "[boundary]",
            'kind = "fixed_lcfs_static"',
            "potential = 0.0",
            'version = "staircase_relative_v1"',
            "[solver]",
            'backend = "cpu"',
            'precision = "fp64"',
            'complex_precision = "complex128"',
            f"memory_limit_bytes = {512 << 20}",
            f"device_memory_limit_bytes = {512 << 20}",
            "rank_rtol = 1.0e-12",
            "device_ordinal = 0",
            "cpu_threads = 1",
            "[solver.constants]",
            "epsilon0_Fpm = 8.8541878188e-12",
            f"mu0_Hpm = {mu0!r}",
            'version = "A0-SI-v1"',
            "[solver.references]",
            "L_m = 0.2",
            "t_s = 1.0e-6",
            "n_i_m3 = 1.0e12",
            "n_e_m3 = 1.0e12",
            "u_mps = 1.0",
            "E_Vpm = 1.0",
            "B_T = 1.0e-5",
            f"J_Apm2 = {current!r}",
            "rho_Cpm3 = 1.0e-7",
            "p_Pa = 1.0e-9",
            "[budgets.domain]",
            "n_min_i = 0.0",
            "n_min_e = 0.0",
            "n_face_min_i = 0.0",
            "n_face_min_e = 0.0",
            "p_min_i = 0.0",
            "p_min_perp_e = 0.0",
            "p_min_parallel_e = 0.0",
            "B_min = 1.0e-12",
        ]
    )
    for name, check in _budget_checks().items():
        lines.extend(
            [
                f'[budgets.checks."{name}"]',
                "atol = 0.0",
                "rtol = 1.0e-9",
                f"reference = {check['reference']!r}",
                f"unit = {json.dumps(check['unit'])}",
                f"norm = {json.dumps(check['norm'])}",
                f"gate = {json.dumps('hard' if check['hard'] else 'observe')}",
            ]
        )
    lines.extend(
        [
            "[diagnostics]",
            "inner_distance_m = 0.05",
            'alias_policy = "initial_midpoint_restart"',
            "timing_repetitions = 1",
            "timing_warmup = 0",
            "mode_observations = []",
            "[io]",
            'geometry_file = "geometry.npz"',
            'initial_file = "initial.npz"',
            'output_dir = "run"',
            "snapshot_stride = 1",
            "checkpoint_stride = 1",
            "max_input_bytes = 16777216",
            "max_uncompressed_bytes = 33554432",
            "max_entries = 4096",
            "max_header_bytes = 1048576",
            'source_manifest_file = "source_manifest.json"',
            "write_residual_arrays = false",
        ]
    )
    return "\n".join(lines) + "\n"


def _replace_top_key(text: str, key: str, literal: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(key)}\s*=\s*.*$")
    replacement = f"{key} = {literal}"
    result, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise ValueError(f"top-level TOML key is missing: {key}")
    return result


def _set_section_key(
    text: str, section: str, key: str, literal: str
) -> str:
    lines = text.splitlines()
    header = f"[{section}]"
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == header)
    except StopIteration as error:
        raise ValueError(f"TOML section is missing: {section}") from error
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].lstrip().startswith("["):
            end = index
            break
    key_pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for index in range(start + 1, end):
        if key_pattern.match(lines[index]):
            lines[index] = f"{key} = {literal}"
            return "\n".join(lines) + "\n"
    lines.insert(start + 1, f"{key} = {literal}")
    return "\n".join(lines) + "\n"


def _render_case(
    template: str,
    *,
    case_name: str,
    grid: int,
    K: int,
    dt: float,
    nsteps: int,
    integrator: str,
    dt_guard: str,
    backend: str,
    device_ordinal: int,
    geometry_path: str,
    initial_path: str,
    source_path: str,
) -> str:
    text = _replace_top_key(template, "case_name", repr(case_name))
    text = _set_section_key(text, "species", "electron_inertia", "1")
    text = _set_section_key(text, "grid", "K", str(K))
    text = _set_section_key(text, "grid", "NR", str(grid))
    text = _set_section_key(text, "grid", "NZ", str(grid))
    text = _set_section_key(text, "grid", "Nphi", str(4 * K + 1))
    text = _set_section_key(text, "time", "dt", format(dt, ".17g"))
    text = _set_section_key(text, "time", "nsteps", str(nsteps))
    text = _set_section_key(
        text, "time", "initial_kind", repr("non_equilibrium")
    )
    text = _set_section_key(text, "time", "integrator", repr(integrator))
    text = _set_section_key(text, "time", "dt_guard", repr(dt_guard))
    text = _set_section_key(text, "solver", "backend", repr(backend))
    text = _set_section_key(text, "solver", "device_ordinal", str(device_ordinal))
    text = _set_section_key(text, "io", "geometry_file", repr(geometry_path))
    text = _set_section_key(text, "io", "initial_file", repr(initial_path))
    text = _set_section_key(text, "io", "source_manifest_file", repr(source_path))
    text = _set_section_key(text, "io", "snapshot_stride", str(max(1, nsteps)))
    text = _set_section_key(text, "io", "checkpoint_stride", str(max(1, nsteps)))
    text = _set_section_key(text, "io", "write_residual_arrays", "false")
    text = _set_section_key(text, "diagnostics", "mode_observations", "[]")
    return text


def _relative_path(target: Path, parent: Path) -> str:
    return os.path.relpath(target, parent)


def _source_manifest(
    case_path: Path,
    geometry_path: Path,
    initial_path: Path,
    *,
    grid: int,
    initial_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "provider_name": "warp_benchmark_closed_reference_density_seed",
        "provider_version": "v1",
        "units": "SI",
        "conversion_method": (
            "Python-generated A0 case schema; production closed_reference with "
            "zero flow; Hermitian electron-density k=+/-1 seed under a smooth "
            "spatial envelope; unchanged ion density, positive pressures, and "
            "discrete harmonic magnetic field."
        ),
        "continuous_residual_status": (
            "Controlled numerical initial condition; not a tokamak equilibrium "
            "or experimental validation."
        ),
        "initial_metadata": initial_metadata,
        "input_hashes": {
            "case": _sha256(case_path),
            "geometry": _sha256(geometry_path),
            "initial": _sha256(initial_path),
        },
    }


def _write_geometry(path: Path, geometry: Any) -> None:
    polygon = np.asarray(
        [(point.R, point.Z) for point in geometry.polygon_input],
        dtype=np.float64,
    )
    arrays = {
        "R_nodes": np.asarray(geometry.R, dtype=np.float64),
        "Z_nodes": np.asarray(geometry.Z, dtype=np.float64),
        "lcfs_R": np.ascontiguousarray(polygon[:, 0]),
        "lcfs_Z": np.ascontiguousarray(polygon[:, 1]),
    }
    _write_npz(path, arrays)


def _smooth_envelope(geometry: Any) -> tuple[np.ndarray, tuple[int, int]]:
    keys = list(geometry.cells.keys)
    r = np.asarray(geometry.cells.R, dtype=np.float64)
    z = np.asarray(geometry.cells.Z, dtype=np.float64)
    if r.size != len(keys) or z.size != len(keys):
        raise ValueError("cell coordinate table does not match active cells")
    r_mid = 0.5 * (float(np.min(r)) + float(np.max(r)))
    z_mid = 0.5 * (float(np.min(z)) + float(np.max(z)))
    r_scale = max(float(np.max(r)) - float(np.min(r)), 1.0)
    z_scale = max(float(np.max(z)) - float(np.min(z)), 1.0)
    dr = (r - r_mid) / r_scale
    dz = (z - z_mid) / z_scale
    envelope = np.cos(np.pi * dr) ** 2 * np.cos(np.pi * dz) ** 2
    envelope = np.clip(envelope, 1.0e-12, 1.0)
    selected = int(np.argmax(envelope))
    return envelope, keys[selected]


def _seed_electron_density(
    state: Any,
    context: Any,
    *,
    density: float,
    amplitude: float,
) -> dict[str, Any]:
    envelope, selected_key = _smooth_envelope(context.geometry)
    positive, negative = hermite_density_pair(density, envelope, amplitude)
    nc = int(context.op.Nc)
    K = int(context.p.K)
    state.fluid[1].n[(K + 1) * nc : (K + 2) * nc] = context.compute.array(
        positive, np.complex128
    )
    state.fluid[1].n[(K - 1) * nc : K * nc] = context.compute.array(
        negative, np.complex128
    )
    coefficient = complex(positive[np.argmax(envelope)])
    return {
        "definition": (
            "Hermitian electron-density Fourier seed; physical fractional "
            "peak is amplitude*envelope*cos(phi+phase)"
        ),
        "density_fraction_peak": float(amplitude),
        "mode": 1,
        "selected_entity_key": [int(selected_key[0]), int(selected_key[1])],
        "selected_fourier_coefficient": coefficient,
        "envelope_max": float(np.max(envelope)),
        "envelope_min": float(np.min(envelope)),
        "hermitian_defect": float(np.max(np.abs(negative - np.conj(positive)))),
        "ion_density_changed": False,
        "pressures_changed": False,
        "momentum_zero_flow": True,
        "initial_current_zero": True,
        "magnetic_field_changed": False,
        "ampere_validation": (
            "zero flow with unchanged harmonic B; checked by the production "
            "initialize initial_ampere diagnostic"
        ),
        "initial_charge": (
            "finite electron-density perturbation; electrostatic potential "
            "is solved by the production finite-Poisson initialize path"
        ),
    }


def _initial_physical_metadata(
    state: Any, context: Any, core: Any, seed: dict[str, Any]
) -> dict[str, Any]:
    nc = int(context.op.Nc)
    electron = core.inverse(state.fluid[1].n, nc, context, True)
    ion = core.inverse(state.fluid[0].n, nc, context, True)
    electron_perp = core.inverse(state.fluid[1].perp, nc, context, True)
    electron_parallel = core.inverse(
        state.fluid[1].parallel, nc, context, True
    )
    ion_perp = core.inverse(state.fluid[0].perp, nc, context, True)
    momentum = [
        core.inverse(fluid.mom, 3 * nc, context, True) for fluid in state.fluid
    ]
    xp = context.compute.xp

    def minimum(value: Any) -> float:
        return float(xp.real(xp.min(value)).item())

    return {
        **seed,
        "electron_density_min_m3": minimum(electron),
        "electron_density_max_m3": float(xp.real(xp.max(electron)).item()),
        "ion_density_min_m3": minimum(ion),
        "ion_density_max_m3": float(xp.real(xp.max(ion)).item()),
        "electron_perp_pressure_min_Pa": minimum(electron_perp),
        "electron_parallel_pressure_min_Pa": minimum(electron_parallel),
        "ion_pressure_min_Pa": minimum(ion_perp),
        "maximum_abs_initial_momentum": max(
            float(xp.max(xp.abs(value)).item())
            for value in momentum
        ),
    }


def _build_grid_inputs(
    output: Path,
    template: str,
    *,
    grid: int,
    K: int,
    first_case: CaseSpec,
    backend: str,
    device_ordinal: int,
) -> GridInputs:
    modules = _solver_modules()
    from fomhd.geometry import build_rectangle

    directory = output / "inputs" / f"grid{grid}-k{K}"
    directory.mkdir(parents=True)
    geometry_path = directory / "geometry.npz"
    geometry = build_rectangle(grid, grid, 1.0, 0.0, 0.2, 0.2)
    _write_geometry(geometry_path, geometry)

    bootstrap_dir = directory / "bootstrap"
    bootstrap_dir.mkdir()
    bootstrap_case = bootstrap_dir / "case.toml"
    bootstrap_initial = bootstrap_dir / "bootstrap_initial.npz"
    _write_npz(
        bootstrap_initial,
        {"bootstrap": np.asarray(True)},
    )
    bootstrap_text = _render_case(
        template,
        case_name=f"bootstrap-g{grid}-k{K}",
        grid=grid,
        K=K,
        dt=first_case.dt,
        nsteps=0,
        integrator=first_case.integrator,
        dt_guard="error" if first_case.kind == "pilot" else "warn",
        backend=backend,
        device_ordinal=device_ordinal,
        geometry_path=_relative_path(geometry_path, bootstrap_dir),
        initial_path=_relative_path(bootstrap_initial, bootstrap_dir),
        source_path="source_manifest.json",
    )
    _write_text(bootstrap_case, bootstrap_text)
    _write_json(
        bootstrap_dir / "source_manifest.json",
        _source_manifest(
            bootstrap_case,
            geometry_path,
            bootstrap_initial,
            grid=grid,
            initial_metadata={"bootstrap_seed": True},
        ),
    )
    bootstrap = modules.io.load_case(bootstrap_case)
    context = modules.Context(
        bootstrap.parameters,
        bootstrap.geometry,
        bootstrap.operators,
        bootstrap.compute,
        bootstrap.parameters.Nphi,
    )
    state = modules.stage.closed_reference(
        context, EXPECTED_N, EXPECTED_P, EXPECTED_P, EXPECTED_B, velocity=0.0
    )
    seed = _seed_electron_density(
        state,
        context,
        density=EXPECTED_N,
        amplitude=INITIAL_DENSITY_FRACTION,
    )
    metadata = _initial_physical_metadata(state, context, modules.core, seed)
    encoded = modules.io.encode_state(state, bootstrap.geometry)
    initial_path = directory / "initial.npz"
    _write_npz(initial_path, encoded)
    metadata.update(
        {
            "controlled_parameters": {
                "n_m3": EXPECTED_N,
                "electron_pressure_Pa": EXPECTED_P,
                "ion_pressure_Pa": EXPECTED_P,
                "magnetic_reference_T": EXPECTED_B,
                "scope": "controlled numerical parameters, not a tokamak",
            },
            "input_sha256": _sha256(initial_path),
            "geometry_sha256": _sha256(geometry_path),
        }
    )
    _write_json(directory / "initial_metadata.json", metadata)

    final_bootstrap = _render_case(
        template,
        case_name=f"bootstrap-g{grid}-k{K}",
        grid=grid,
        K=K,
        dt=first_case.dt,
        nsteps=0,
        integrator=first_case.integrator,
        dt_guard="error" if first_case.kind == "pilot" else "warn",
        backend=backend,
        device_ordinal=device_ordinal,
        geometry_path=_relative_path(geometry_path, bootstrap_dir),
        initial_path=_relative_path(initial_path, bootstrap_dir),
        source_path="source_manifest.json",
    )
    _write_text(bootstrap_case, final_bootstrap)
    _write_json(
        bootstrap_dir / "source_manifest.json",
        _source_manifest(
            bootstrap_case,
            geometry_path,
            initial_path,
            grid=grid,
            initial_metadata=metadata,
        ),
    )
    modules.io.load_case(bootstrap_case)
    return GridInputs(
        grid=grid,
        K=K,
        directory=directory,
        geometry_path=geometry_path,
        initial_path=initial_path,
        initial_sha256=_sha256(initial_path),
        geometry_sha256=_sha256(geometry_path),
        bootstrap_case=bootstrap_case,
        metadata=metadata,
    )


def _write_case(
    output: Path,
    template: str,
    inputs: GridInputs,
    spec: CaseSpec,
    *,
    backend: str,
    device_ordinal: int,
) -> Path:
    case_dir = output / "cases" / spec.case_id
    case_dir.mkdir(parents=True)
    case_path = case_dir / "case.toml"
    text = _render_case(
        template,
        case_name=spec.case_id,
        grid=spec.grid,
        K=spec.K,
        dt=spec.dt,
        nsteps=spec.nsteps,
        integrator=spec.integrator,
        dt_guard="error" if spec.kind == "pilot" else "warn",
        backend=backend,
        device_ordinal=device_ordinal,
        geometry_path=_relative_path(inputs.geometry_path, case_dir),
        initial_path=_relative_path(inputs.initial_path, case_dir),
        source_path="source_manifest.json",
    )
    _write_text(case_path, text)
    _write_json(
        case_dir / "source_manifest.json",
        _source_manifest(
            case_path,
            inputs.geometry_path,
            inputs.initial_path,
            grid=spec.grid,
            initial_metadata=inputs.metadata,
        ),
    )
    return case_path


def _rms(value: Any, backend: Any) -> float:
    if int(getattr(value, "size", 0)) == 0:
        return 0.0
    xp = backend.xp
    return float(backend.scalar(xp.sqrt(xp.mean(xp.abs(value) ** 2))))


def _check_records(checks: list[Any]) -> list[dict[str, Any]]:
    records = []
    for check in checks:
        records.append(
            {
                "name": check.name,
                "population": check.population,
                "component": getattr(check, "component", ""),
                "producer": getattr(check, "producer", "diagnostics"),
                "region": check.region,
                "norm": str(check.norm),
                "available": bool(check.available),
                "hard": bool(check.hard),
                "passed": bool(check.passed),
                "raw": check.raw,
                "threshold": check.threshold,
                "reference": check.reference,
                "unit": check.unit,
                "reason": getattr(check, "reason", ""),
                "stage": _typed(check.tag),
            }
        )
    return records


POSITIVITY_CHECK_NAMES = frozenset(
    {"minimum_n", "minimum_face_n", "minimum_perp", "minimum_parallel", "minimum_B"}
)


def _normalised_residual(check: Any) -> float | None:
    norm = str(check.norm).lower()
    threshold = float(check.threshold)
    if (
        not check.available
        or norm not in ("linf", "weighted_l2")
        or not math.isfinite(threshold)
        or threshold <= 0.0
    ):
        return None
    return abs(float(check.raw)) / threshold


def _diagnostic_summary(checks: list[Any]) -> dict[str, Any]:
    """Separate upper-bound residual ratios from lower-bound domain checks."""
    evaluated = []
    excluded = 0
    peak = None
    peak_ratio = None
    positivity = {}
    actual_failures = []
    for check in checks:
        ratio = _normalised_residual(check)
        if ratio is None:
            excluded += 1
        else:
            evaluated.append(check)
            if math.isfinite(ratio) and (peak_ratio is None or ratio > peak_ratio):
                peak_ratio = ratio
                peak = check
        name = str(check.name)
        if name in POSITIVITY_CHECK_NAMES and check.available:
            value = float(check.raw)
            positivity[name] = (
                value if name not in positivity else min(positivity[name], value)
            )
        if not check.available:
            status = "UNAVAILABLE"
        elif not check.passed:
            status = "FAIL" if check.hard else "OBSERVE_FAIL"
        else:
            status = None
        if status is not None:
            actual_failures.append(
                {
                    "name": name,
                    "population": check.population,
                    "component": getattr(check, "component", ""),
                    "region": check.region,
                    "norm": str(check.norm),
                    "raw": check.raw,
                    "threshold": check.threshold,
                    "hard": bool(check.hard),
                    "status": status,
                    "reason": getattr(check, "reason", ""),
                    "stage": getattr(check, "tag", None),
                }
            )
    peak_record = None
    if peak is not None:
        peak_record = {
            "name": peak.name,
            "population": peak.population,
            "component": getattr(peak, "component", ""),
            "region": peak.region,
            "norm": str(peak.norm),
            "raw": peak.raw,
            "threshold": peak.threshold,
            "ratio": peak_ratio,
            "stage": getattr(peak, "tag", None),
        }
    hard_failed = sum(
        1
        for item in actual_failures
        if item["hard"] and item["status"] == "FAIL"
    )
    hard_unavailable = sum(
        1
        for item in actual_failures
        if item["hard"] and item["status"] == "UNAVAILABLE"
    )
    return {
        "max_normalised_residual": peak_ratio,
        "peak_residual_check": peak_record,
        "residual_normalisation": {
            "selection": (
                "available linf/weighted_L2 checks with positive finite "
                "threshold; exact raw/threshold ratio"
            ),
            "evaluated_count": len(evaluated),
            "excluded_count": excluded,
            "scope": (
                "upper-bound residual diagnostics only; lower-bound positivity "
                "and dimensionless/truth checks are excluded"
            ),
        },
        "positivity": positivity,
        "hard_failed_count": hard_failed,
        "hard_unavailable_count": hard_unavailable,
        "observe_failed_count": sum(
            1 for item in actual_failures if item["status"] == "OBSERVE_FAIL"
        ),
        "actual_failure_statuses": actual_failures,
    }


def _collect(
    result: Any,
    state: Any,
    context: Any,
    *,
    selected_cell: int,
    elapsed: float,
) -> dict[str, Any]:
    backend = context.compute
    frame = getattr(result, "frame", result)
    checks = list(result.checks)
    diagnostic_summary = _diagnostic_summary(checks)
    nc = int(context.op.Nc)
    K = int(context.p.K)
    coefficient = state.fluid[1].n[(K + 1) * nc + selected_cell]
    coefficient_value = complex(backend.scalar(coefficient))
    cache = frame.cache
    observed = {
        "step": int(state.step),
        "time_s": float(state.time),
        "step_elapsed_s": float(elapsed),
        "ne_hat_real": coefficient_value.real,
        "ne_hat_imag": coefficient_value.imag,
        "E_rms_V_per_m": _rms(cache.E, backend),
        "E_energy_J": float(frame.totals.longitudinal),
        "ion_density_rms_m3": _rms(frame.fluid[0].n, backend),
        "electron_density_rms_m3": _rms(frame.fluid[1].n, backend),
        "ion_velocity_rms_m_per_s": _rms(frame.fluid[0].u, backend),
        "electron_velocity_rms_m_per_s": _rms(frame.fluid[1].u, backend),
        "number_i": float(frame.totals.number[0]),
        "number_e": float(frame.totals.number[1]),
        "total_energy_J": float(frame.totals.energy),
        "max_normalised_residual": diagnostic_summary[
            "max_normalised_residual"
        ],
        "hard_failure_count": diagnostic_summary["hard_failed_count"],
        "hard_unavailable_count": diagnostic_summary[
            "hard_unavailable_count"
        ],
        "observe_failed_count": diagnostic_summary["observe_failed_count"],
        "positivity": diagnostic_summary["positivity"],
        "diagnostic_summary": diagnostic_summary,
        "accepted": True,
    }
    return observed


def _state_arrays(state: Any, geometry: Any, io: Any, prefix: str) -> dict[str, Any]:
    arrays = {
        f"{prefix}__{key}": value
        for key, value in io.encode_state(state, geometry).items()
    }
    arrays[f"{prefix}__step"] = np.asarray(state.step, dtype=np.uint64)
    arrays[f"{prefix}__generation"] = np.asarray(
        state.generation, dtype=np.uint64
    )
    arrays[f"{prefix}__time"] = np.asarray(state.time, dtype=np.float64)
    return arrays


def _serialise_failure(error: BaseException, accepted_step: int | None) -> dict[str, Any]:
    record = {
        "error_type": type(error).__name__,
        "message": str(error),
        "accepted_step": accepted_step,
        "exception_chain": [
            {
                "type": type(item).__name__,
                "code": getattr(item, "code", None),
                "message": str(item),
            }
            for item in _exception_chain(error)
        ],
        "code": getattr(error, "code", None),
        "stage": getattr(error, "stage", None),
        "checks": _check_records(getattr(error, "checks", [])),
        "partial_array_keys": sorted(getattr(error, "partial_arrays", {}).keys()),
        "nonfinite_arrays_preserved": False,
    }
    return record


def _write_failure_arrays(
    path: Path,
    error: BaseException,
    backend: Any | None,
    state: Any | None,
    geometry: Any,
    io: Any,
) -> list[str]:
    arrays = {}
    partial = getattr(error, "partial_arrays", {})
    if backend is not None:
        for key, value in partial.items():
            try:
                arrays[f"failure__{key}"] = np.asarray(backend.to_host(value))
            except Exception as conversion_error:
                arrays[f"failure_conversion_error__{key}"] = np.asarray(
                    str(conversion_error)
                )
    if state is not None:
        arrays.update(_state_arrays(state, geometry, io, "last_accepted"))
    arrays["failure_code"] = np.asarray(getattr(error, "code", type(error).__name__))
    arrays["failure_message"] = np.asarray(str(error))
    _write_npz(path, arrays)
    return sorted(arrays)


def _backend_device_ordinal(backend: Any) -> int | None:
    device = str(backend.device)
    if device == "cpu":
        return None
    match = re.fullmatch(r"cuda:(\d+)", device)
    if match is None:
        raise ValueError(f"invalid backend device identity: {device!r}")
    return int(match.group(1))


def _device_provenance(context: Any, core: Any) -> dict[str, Any]:
    backend = context.compute
    ordinal = _backend_device_ordinal(backend)
    record = {
        "requested": backend.device,
        "actual": backend.device,
        "device_ordinal": ordinal,
        "backend_name": backend.name,
        "fallback_used": False,
        "estimated_peak_bytes": int(core.estimated_peak_bytes(context)),
        "device_memory_limit_bytes": int(context.p.device_memory_limit),
    }
    if backend.name == "cuda":
        if ordinal is None:
            raise ValueError("CUDA backend has no device ordinal")
        cp = backend.xp
        import warp

        with cp.cuda.Device(ordinal):
            properties = cp.cuda.runtime.getDeviceProperties(ordinal)
            free_memory, total_memory = cp.cuda.runtime.memGetInfo()
            stream_pointer = int(cp.cuda.get_current_stream().ptr)
        record.update(
            {
                "name": str(properties["name"]),
                "pci_bus_id": int(properties["pciBusID"]),
                "total_memory_bytes": int(total_memory),
                "free_memory_bytes_at_probe": int(free_memory),
                "warp_version": str(warp.__version__),
                "cupy_version": str(cp.__version__),
                "stream": stream_pointer,
            }
        )
    else:
        record.update(
            {
                "name": platform.processor() or "CPU",
                "total_memory_bytes": None,
                "free_memory_bytes_at_probe": None,
            }
        )
    return record


def _empty_cumulative(start_time: float) -> Any:
    return SimpleNamespace(
        baseline=None,
        integrals=[
            SimpleNamespace(
                electric_node=0.0,
                electric_edge=0.0,
                pressure_kinetic=0.0,
                pressure_internal=0.0,
                magnetic=0.0,
                wall=0.0,
                convection=0.0,
                perp_rate=0.0,
                parallel_rate=0.0,
                spectral_kinetic=0.0,
                spectral_perp=0.0,
                spectral_parallel=0.0,
            )
            for _ in range(2)
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
        last_time=float(start_time),
    )


def _write_case_tables(
    case_dir: Path,
    observations: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    with (case_dir / "history.csv").open("x", encoding="utf-8", newline="") as handle:
        if observations:
            writer = csv.DictWriter(handle, fieldnames=list(observations[0]))
            writer.writeheader()
            writer.writerows(observations)
    with (case_dir / "events.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "phase",
                "status",
                "wall_s",
                "time_s",
                "error_type",
                "error_code",
                "message",
            ],
        )
        writer.writeheader()
        writer.writerows(events)


def _timing_summary(timings: list[float], setup_s: float, initial_s: float) -> dict[str, Any]:
    record = {
        "setup_s": float(setup_s),
        "initial_transaction_s": float(initial_s),
        "first_step_s": None,
        "steady_median_s": None,
        "steady_half_interdecile_spread_s": None,
        "accepted_timed_steps": len(timings),
    }
    if not timings:
        return record
    record["first_step_s"] = float(timings[0])
    if len(timings) > 1:
        steady = np.asarray(timings[1:], dtype=np.float64)
        record["steady_median_s"] = float(np.median(steady))
        record["steady_half_interdecile_spread_s"] = float(
            0.5 * (np.percentile(steady, 90) - np.percentile(steady, 10))
        )
    return record


def _run_case(
    case_path: Path,
    spec: CaseSpec,
    *,
    deadline: float,
    case_started: float,
) -> dict[str, Any]:
    modules = _solver_modules()
    case_dir = case_path.parent
    observations: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    states: dict[str, Any] = {}
    checks_initial: list[dict[str, Any]] = []
    checks_final: list[dict[str, Any]] = []
    timings: list[float] = []
    setup_s = float("nan")
    initial_s = float("nan")
    accepted_step = 0
    context = None
    committed = None
    run = None
    failure = None
    final_observation = None
    diagnostic_events: list[dict[str, Any]] = []
    runtime_time_reports: list[dict[str, Any]] = []
    case = None
    phase = "pre_commit_input"
    try:
        if time.monotonic() >= deadline:
            raise BudgetExhausted("wall-clock budget exhausted before case setup")
        started = time.perf_counter()
        case = modules.io.load_case(case_path)
        case.compute.synchronize()
        setup_s = time.perf_counter() - started
        backend = case.compute
        device = _device_provenance(
            modules.Context(
                case.parameters,
                case.geometry,
                case.operators,
                case.compute,
                case.parameters.Nphi,
            ),
            modules.core,
        )
        artifact_started = time.perf_counter()
        run = modules.io.Run(case)
        artifact_setup_s = time.perf_counter() - artifact_started
        started = time.perf_counter()
        state = modules.io.load_initial(case)
        context = modules.Context(
            case.parameters,
            case.geometry,
            case.operators,
            case.compute,
            case.parameters.Nphi,
        )
        cumulative = _empty_cumulative(case.parameters.start_time)
        captured = {}
        committed = modules.stage.initialize(
            state, context, cumulative, False, captured
        )
        context.compute.synchronize()
        initial_s = time.perf_counter() - started
        phase = "numerical_trial"
        initial_frame = SimpleNamespace(**captured)
        checks_initial = _check_records(initial_frame.checks)
        _write_json(case_dir / "initial_checks.json", checks_initial)
        selected = case.source["initial_metadata"]["selected_entity_key"]
        selected_cell = case.geometry.cells.find(int(selected[0]), int(selected[1]))
        initial_observation = _collect(
            initial_frame,
            committed.state,
            context,
            selected_cell=selected_cell,
            elapsed=initial_s,
        )
        observations.append(initial_observation)
        diagnostic_events.append(
            {
                "step": 0,
                "time_s": committed.state.time,
                **initial_observation["diagnostic_summary"],
            }
        )
        states.update(
            _state_arrays(committed.state, case.geometry, modules.io, "initial")
        )
        snapshot_steps = {
            int(round(value))
            for value in np.linspace(0, spec.nsteps, num=min(8, spec.nsteps + 1))
        }
        snapshot_steps.add(0)
        snapshot_steps.add(spec.nsteps)
        events.append(
            {
                "step": 0,
                "phase": "initialize",
                "status": "accepted",
                "wall_s": initial_s,
                "time_s": committed.state.time,
                "error_type": "",
                "error_code": "",
                "message": "",
            }
        )
        while committed.state.step < spec.nsteps:
            if time.monotonic() >= deadline:
                raise BudgetExhausted(
                    f"wall-clock budget exhausted at accepted step {committed.state.step}"
                )
            step_started = time.perf_counter()
            result = modules.solver_time.advance(committed, cumulative, context)
            context.compute.synchronize()
            elapsed = time.perf_counter() - step_started
            timings.append(elapsed)
            accepted_step = int(committed.state.step)
            observation = _collect(
                result,
                committed.state,
                context,
                selected_cell=selected_cell,
                elapsed=elapsed,
            )
            observations.append(observation)
            diagnostic_events.append(
                {
                    "step": accepted_step,
                    "time_s": committed.state.time,
                    **observation["diagnostic_summary"],
                }
            )
            final_observation = observation
            report = getattr(result, "time_step_report", None)
            if report is not None:
                runtime_time_reports.append(_typed(report))
                time_estimate = getattr(run, "time_estimate", None)
                if callable(time_estimate):
                    time_estimate(report)
            if accepted_step in snapshot_steps:
                states.update(
                    _state_arrays(
                        committed.state,
                        case.geometry,
                        modules.io,
                        f"snapshot_{accepted_step:010d}",
                    )
                )
            events.append(
                {
                    "step": accepted_step,
                    "phase": "advance",
                    "status": "accepted",
                    "wall_s": elapsed,
                    "time_s": committed.state.time,
                    "error_type": "",
                    "error_code": "",
                    "message": "",
                }
            )
        final_time_error = abs(
            float(committed.state.time) - float(spec.target_time_s)
        ) / max(abs(float(spec.target_time_s)), np.finfo(float).tiny)
        if final_time_error > 64.0 * np.finfo(float).eps:
            raise modules.Failure(
                "TimeHorizon",
                f"accepted final time differs from target by {final_time_error:g}",
            )
        checks_final = _check_records(result.checks)
        _write_json(case_dir / "final_checks.json", checks_final)
        states.update(
            _state_arrays(committed.state, case.geometry, modules.io, "final")
        )
        artifacts_started = time.perf_counter()
        phase = "post_commit_output"
        run.diagnostics(
            committed.state,
            result.checks,
            result.totals,
            result.momentum,
            time.monotonic() - case_started,
        )
        run.snapshot(committed, result.frame)
        run.checkpoint(committed, cumulative, context)
        run.status("completed")
        artifact_write_s = time.perf_counter() - artifacts_started
        status = "accepted"
        error_record = None
    except BudgetExhausted as error:
        failure = error
        status = "budget_exhausted"
        error_record = _serialise_failure(error, accepted_step)
        events.append(
            {
                "step": accepted_step,
                "phase": "wallclock_budget",
                "status": "budget_exhausted",
                "wall_s": "",
                "time_s": committed.state.time if committed is not None else "",
                "error_type": type(error).__name__,
                "error_code": "",
                "message": str(error),
            }
        )
        if committed is not None and context is not None:
            try:
                run.checkpoint(committed, cumulative, context)
            except Exception as checkpoint_error:
                error_record["checkpoint_error"] = str(checkpoint_error)
        if case is not None:
            try:
                raw_keys = _write_failure_arrays(
                    case_dir / "failed_raw.npz",
                    error,
                    case.compute,
                    committed.state if committed is not None else None,
                    case.geometry,
                    modules.io,
                )
                error_record["raw_array_keys"] = raw_keys
                error_record["nonfinite_arrays_preserved"] = bool(
                    getattr(error, "partial_arrays", {})
                )
            except Exception as raw_error:
                error_record["raw_array_write_error"] = str(raw_error)
        _write_json(case_dir / "failure.json", error_record)
        if run is not None:
            try:
                run.status("budget_exhausted")
            except Exception:
                pass
    except Exception as error:
        failure = error
        budget_root = _find_budget_exhaustion(error)
        status = "budget_exhausted" if budget_root is not None else "failed"
        error_record = _serialise_failure(error, accepted_step)
        if budget_root is not None:
            error_record["wrapped_budget_exhausted"] = True
            error_record["budget_message"] = str(budget_root)
        events.append(
            {
                "step": accepted_step,
                "phase": "solver",
                "status": status,
                "wall_s": "",
                "time_s": committed.state.time if committed is not None else "",
                "error_type": type(error).__name__,
                "error_code": getattr(error, "code", ""),
                "message": str(error),
            }
        )
        if committed is not None and context is not None:
            try:
                run.checkpoint(committed, cumulative, context)
            except Exception as checkpoint_error:
                error_record["checkpoint_error"] = str(checkpoint_error)
        if run is not None:
            try:
                run.failure(
                    error,
                    accepted_step,
                    phase,
                )
            except Exception as recording_error:
                error_record["native_failure_recording_error"] = str(
                    recording_error
                )
            else:
                if budget_root is not None:
                    try:
                        run.status("budget_exhausted")
                        error_record["nested_run_status"] = "budget_exhausted"
                    except Exception as status_error:
                        error_record["nested_run_status_error"] = str(
                            status_error
                        )
        if case is not None:
            try:
                raw_keys = _write_failure_arrays(
                    case_dir / "failed_raw.npz",
                    error,
                    case.compute,
                    committed.state if committed is not None else None,
                    case.geometry,
                    modules.io,
                )
                error_record["raw_array_keys"] = raw_keys
                error_record["nonfinite_arrays_preserved"] = bool(
                    getattr(error, "partial_arrays", {})
                )
            except Exception as raw_error:
                error_record["raw_array_write_error"] = str(raw_error)
        _write_json(case_dir / "failure.json", error_record)
    finally:
        if run is not None:
            run.close()

    if final_observation is None and observations:
        final_observation = observations[-1]
    peak_ratios = [
        event["max_normalised_residual"]
        for event in diagnostic_events
        if event["max_normalised_residual"] is not None
    ]
    run_peak = max(peak_ratios) if peak_ratios else None
    run_positivity = {}
    for event in diagnostic_events:
        for name, value in event["positivity"].items():
            run_positivity[name] = min(
                value, run_positivity.get(name, value)
            )
    actual_failure_statuses = [
        {"step": event["step"], **item}
        for event in diagnostic_events
        for item in event["actual_failure_statuses"]
    ]
    runtime_guard = {
        "status": "AVAILABLE" if runtime_time_reports else "UNAVAILABLE",
        "source": "actual result.time_step_report from fomhd.time.advance",
        "scope": (
            "operator-resolution estimate and explicit-method finite-horizon "
            "action; not the cold scalar preassessment"
        ),
        "cold_preassessment": "benchmark.json.scan.cold_reference_preassessment",
        "reports": runtime_time_reports,
    }
    rejection_report = None
    if failure is not None:
        rejection_report = {
            "status": status,
            "failure": error_record,
            "time_step_report": getattr(failure, "time_step_report", None)
            or (
                getattr(context, "last_dt_report", None)
                if context is not None
                else None
            ),
            "accepted_diagnostic_events": diagnostic_events,
            "scope": (
                "rejected or budget-terminated trial evidence; reports are "
                "not promoted to accepted time-step records"
            ),
        }
    timing = _timing_summary(timings, setup_s, initial_s)
    result_record = {
        "case_id": spec.case_id,
        "kind": spec.kind,
        "status": status,
        "integrator": spec.integrator,
        "grid": spec.grid,
        "K": spec.K,
        "dt_s": spec.dt,
        "nsteps_requested": spec.nsteps,
        "target_time_s": spec.target_time_s,
        "accepted_step": accepted_step,
        "dt_guard_policy": "error" if spec.kind == "pilot" else "warn",
        "accepted_time_s": (
            float(committed.state.time) if committed is not None else None
        ),
        "final_time_relative_error": locals().get("final_time_error"),
        "timing": timing,
        "artifact_setup_s": locals().get("artifact_setup_s"),
        "artifact_write_s": locals().get("artifact_write_s"),
        "device": locals().get("device"),
        "failure": error_record,
        "observation_count": len(observations),
        "runtime_guard": runtime_guard,
        "max_normalised_residual_selection": (
            "peak over initial and every accepted step using available "
            "linf/weighted_L2 checks with positive finite thresholds"
        ),
        "max_normalised_residual": run_peak,
        "peak_residual_check": (
            max(
                (
                    event["peak_residual_check"]
                    for event in diagnostic_events
                    if event["peak_residual_check"] is not None
                ),
                key=lambda item: item["ratio"],
                default=None,
            )
        ),
        "positivity_minima": run_positivity,
        "actual_failure_statuses": actual_failure_statuses,
        "initial_input_sha256": (
            _sha256(case.initial_path) if case is not None else None
        ),
        "geometry_sha256": (
            _sha256(case.geometry_path) if case is not None else None
        ),
    }
    _write_case_tables(case_dir, observations, events)
    _write_json(case_dir / "diagnostic_events.json", diagnostic_events)
    _write_json(case_dir / "time_step_reports.json", runtime_guard)
    if rejection_report is not None:
        _write_json(case_dir / "rejection_report.json", rejection_report)
    if states:
        _write_npz(case_dir / "states.npz", states)
    _write_json(case_dir / "result.json", result_record)
    return {
        "record": result_record,
        "observations": observations,
        "states": states,
        "failure": failure,
        "budget_exhausted": status == "budget_exhausted",
    }


def _frequency_analysis(
    times: np.ndarray,
    signal: np.ndarray,
    *,
    requested_cycles: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    time_s = np.asarray(times, dtype=np.float64).reshape(-1)
    samples = np.asarray(signal, dtype=np.complex128).reshape(-1)
    arrays = {
        "time_s": time_s,
        "signal_real": samples.real,
        "signal_imag": samples.imag,
        "sample_interval_s": np.diff(time_s),
        "frequency_hz": np.empty(0, dtype=np.float64),
        "spectrum_real": np.empty(0, dtype=np.float64),
        "spectrum_imag": np.empty(0, dtype=np.float64),
        "spectrum_abs": np.empty(0, dtype=np.float64),
        "raw_spectrum_abs": np.empty(0, dtype=np.float64),
        "detrended_spectrum_abs": np.empty(0, dtype=np.float64),
    }

    def unavailable(reason: str, **values: Any):
        return {
            "status": "UNAVAILABLE",
            "reason": reason,
            "samples": int(time_s.size),
            "requested_cycles": float(requested_cycles),
            "sample_source": "actual accepted samples only",
            "scope": (
                "coherent-signal estimate only; no Langmuir physics inferred "
                "from an unresolved or aperiodic signal"
            ),
            **values,
        }, arrays

    if (
        time_s.size != samples.size
        or time_s.size < 2
        or not np.isfinite(time_s).all()
        or not np.isfinite(samples).all()
    ):
        return unavailable(
            "need at least two finite matched accepted samples"
        )
    intervals = np.diff(time_s)
    if not np.isfinite(intervals).all() or np.any(intervals <= 0):
        return unavailable("actual sample times are not strictly increasing")
    sample_dt = float(np.median(intervals))
    cadence_defect = (
        float(np.max(np.abs(intervals - sample_dt))) / sample_dt
    )
    arrays["sample_interval_s"] = intervals
    if cadence_defect > 1.0e-6:
        return unavailable(
            "actual sample cadence is not adequately uniform",
            median_cadence_s=sample_dt,
            cadence_relative_defect=cadence_defect,
        )
    duration = float(time_s[-1] - time_s[0])
    if not math.isfinite(duration) or duration <= 0:
        return unavailable("actual sample duration is not positive finite")
    raw_spectrum = np.fft.fft(samples - np.mean(samples))
    raw_frequencies = np.fft.fftfreq(time_s.size, d=sample_dt)
    arrays["frequency_hz"] = raw_frequencies
    arrays["spectrum_real"] = raw_spectrum.real
    arrays["spectrum_imag"] = raw_spectrum.imag
    arrays["spectrum_abs"] = np.abs(raw_spectrum)
    arrays["raw_spectrum_abs"] = np.abs(raw_spectrum)
    if time_s.size < 16:
        return unavailable(
            "need at least 16 finite accepted samples",
            sample_dt_s=sample_dt,
            cadence_relative_defect=cadence_defect,
            duration_s=duration,
            resolution_hz=1.0 / duration,
            nyquist_hz=0.5 / sample_dt,
        )

    design = np.column_stack(
        (np.ones(time_s.size, dtype=np.float64), time_s - time_s[0])
    )
    coefficients, *_ = np.linalg.lstsq(design, samples, rcond=None)
    detrended = samples - design @ coefficients
    total_power = float(np.sum(np.abs(detrended) ** 2))
    raw_power = float(np.sum(np.abs(samples - np.mean(samples)) ** 2))
    if (
        not math.isfinite(total_power)
        or total_power <= np.finfo(float).tiny
        or total_power <= 1.0e-12 * max(raw_power, np.finfo(float).tiny)
    ):
        return unavailable(
            "detrended signal has no resolvable oscillatory power"
        )

    spectrum = np.fft.fft(detrended)
    frequencies = np.fft.fftfreq(time_s.size, d=sample_dt)
    candidate = np.abs(frequencies) > 0
    indices = np.flatnonzero(candidate)
    selected = int(indices[np.argmax(np.abs(spectrum[candidate]))])
    signed_hz = float(frequencies[selected])
    peak_hz = abs(signed_hz)
    if not math.isfinite(peak_hz) or peak_hz <= 0:
        return unavailable("no nonzero finite spectral peak")
    resolution_hz = 1.0 / duration
    nyquist_hz = 0.5 / sample_dt
    observed_cycles = peak_hz * duration
    samples_per_cycle = time_s.size / observed_cycles if observed_cycles else 0
    peak_fraction = (
        float(np.abs(spectrum[selected]) ** 2)
        / max(float(np.sum(np.abs(spectrum[candidate]) ** 2)), np.finfo(float).tiny)
    )
    phase = 2.0 * np.pi * peak_hz * (time_s - time_s[0])
    two_sided_model = np.column_stack(
        (np.exp(1j * phase), np.exp(-1j * phase))
    )
    components, *_ = np.linalg.lstsq(two_sided_model, detrended, rcond=None)
    fitted = two_sided_model @ components
    coherence = 1.0 - float(np.sum(np.abs(detrended - fitted) ** 2)) / total_power
    positive_power = float(abs(components[0]) ** 2)
    negative_power = float(abs(components[1]) ** 2)
    component_total = positive_power + negative_power
    positive_fraction = (
        positive_power / component_total if component_total > 0 else 0.0
    )
    negative_fraction = (
        negative_power / component_total if component_total > 0 else 0.0
    )
    smaller_fraction = min(positive_fraction, negative_fraction)
    if smaller_fraction >= 0.10:
        mode_character = "real_standing_wave"
        propagation_sign = "undetermined"
        reported_signed_hz = None
    elif positive_fraction > negative_fraction:
        mode_character = "directional_complex"
        propagation_sign = "positive"
        reported_signed_hz = peak_hz
    else:
        mode_character = "directional_complex"
        propagation_sign = "negative"
        reported_signed_hz = -peak_hz
    dominant = (
        detrended.real
        if np.sum(detrended.real**2) >= np.sum(detrended.imag**2)
        else detrended.imag
    )
    zero_crossings = int(np.count_nonzero(np.diff(np.signbit(dominant))))
    split = time_s.size // 2
    half_frequencies = []
    for half, values in enumerate((detrended[:split], detrended[split:])):
        half_spectrum = np.fft.fft(values)
        half_frequency = np.fft.fftfreq(values.size, d=sample_dt)
        half_candidate = np.abs(half_frequency) > 0
        half_index = np.flatnonzero(half_candidate)[
            np.argmax(np.abs(half_spectrum[half_candidate]))
        ]
        half_frequencies.append(float(half_frequency[half_index]))
    half_difference = abs(abs(half_frequencies[0]) - abs(half_frequencies[1]))

    arrays["frequency_hz"] = frequencies
    arrays["spectrum_real"] = spectrum.real
    arrays["spectrum_imag"] = spectrum.imag
    arrays["spectrum_abs"] = np.abs(spectrum)
    arrays["detrended_spectrum_abs"] = np.abs(spectrum)
    common = {
        "samples": int(time_s.size),
        "sample_dt_s": sample_dt,
        "cadence_relative_defect": cadence_defect,
        "duration_s": duration,
        "resolution_hz": resolution_hz,
        "nyquist_hz": nyquist_hz,
        "signed_fft_peak_hz": signed_hz,
        "peak_power_fraction": peak_fraction,
        "coherence_r_squared": coherence,
        "fitted_model": (
            "a+ exp(+i*omega*t) + a- exp(-i*omega*t), after linear "
            "complex detrending; both Fourier components fitted"
        ),
        "positive_component_power_fraction": positive_fraction,
        "negative_component_power_fraction": negative_fraction,
        "mode_character": mode_character,
        "propagation_sign": propagation_sign,
        "dominant_component_zero_crossings": zero_crossings,
        "observed_cycles": observed_cycles,
        "samples_per_observed_cycle": samples_per_cycle,
        "half_record_frequencies_hz": half_frequencies,
        "half_record_frequency_difference_hz": half_difference,
    }
    if observed_cycles < 2.0:
        return unavailable(
            "accepted span covers fewer than two measured cycles", **common
        )
    if samples_per_cycle < 8.0:
        return unavailable(
            "actual cadence provides fewer than eight samples per measured cycle",
            **common,
        )
    if peak_fraction < 0.15:
        return unavailable(
            "spectral power is not concentrated in a coherent oscillation",
            **common,
        )
    if coherence < 0.60:
        return unavailable(
            "complex exponential fit lacks coherent oscillation",
            **common,
        )
    if zero_crossings < 3:
        return unavailable(
            "detrended dominant signal lacks a repeating oscillation",
            **common,
        )
    if half_difference > max(3.0 * resolution_hz, 0.25 * peak_hz):
        return unavailable(
            "first and second half records disagree on the spectral peak",
            **common,
        )
    angular = 2.0 * math.pi * peak_hz
    amplitude_ratio = float(
        np.max(np.abs(samples[-1:]))
        / max(np.max(np.abs(samples[:1])), np.finfo(float).tiny)
    )
    return {
        "status": "MEASURED",
        "method": (
            "linear-detrended nonzero-bin FFT, followed by a two-sided "
            "complex-exponential fit, half-record agreement, and "
            "oscillation checks"
        ),
        "signed_frequency_hz": reported_signed_hz,
        "fft_peak_frequency_hz": signed_hz,
        "angular_frequency_rad_s": angular,
        "requested_cycles": float(requested_cycles),
        "final_amplitude_ratio": amplitude_ratio,
        "uncertainty_hz_one_bin": resolution_hz,
        "sample_source": "actual accepted samples only",
        "scope": (
            "finite-horizon coherent-signal estimate; not an exact continuum "
            "eigenfrequency and not Langmuir validation; real standing waves "
            "have no determined propagation sign"
        ),
        **common,
    }, arrays


def _frequency_fit(
    times: np.ndarray,
    signal: np.ndarray,
    *,
    requested_cycles: float,
) -> dict[str, Any]:
    report, _arrays = _frequency_analysis(
        times, signal, requested_cycles=requested_cycles
    )
    return report


def _write_observations(
    path: Path,
    all_observations: dict[str, list[dict[str, Any]]],
    frequency_arrays: dict[str, dict[str, np.ndarray]] | None = None,
) -> None:
    arrays = {}
    for case_id, observations in all_observations.items():
        fields = (
            "step",
            "time_s",
            "step_elapsed_s",
            "ne_hat_real",
            "ne_hat_imag",
            "E_rms_V_per_m",
            "E_energy_J",
            "ion_density_rms_m3",
            "electron_density_rms_m3",
            "ion_velocity_rms_m_per_s",
            "electron_velocity_rms_m_per_s",
            "number_i",
            "number_e",
            "total_energy_J",
            "max_normalised_residual",
            "hard_failure_count",
            "hard_unavailable_count",
            "observe_failed_count",
        )
        for field in fields:
            values = [item[field] for item in observations]
            dtype = (
                np.int64
                if field
                in (
                    "step",
                    "hard_failure_count",
                    "hard_unavailable_count",
                    "observe_failed_count",
                )
                else np.float64
            )
            arrays[f"{case_id}.{field}"] = np.asarray(values, dtype=dtype)
    for case_id, case_arrays in (frequency_arrays or {}).items():
        for field, values in case_arrays.items():
            arrays[f"{case_id}.frequency_{field}"] = np.asarray(values)
    _write_npz(path, arrays)


def _summary_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in cases:
        timing = item.get("timing", {})
        failure = item.get("failure") or {}
        rows.append(
            {
                "case_id": item["case_id"],
                "kind": item["kind"],
                "status": item["status"],
                "integrator": item["integrator"],
                "grid": item["grid"],
                "K": item["K"],
                "dt_s": item["dt_s"],
                "nsteps": item["nsteps_requested"],
                "target_time_s": item["target_time_s"],
                "accepted_step": item["accepted_step"],
                "accepted_time_s": item.get("accepted_time_s"),
                "final_time_relative_error": item.get(
                    "final_time_relative_error"
                ),
                "setup_s": timing.get("setup_s"),
                "initial_s": timing.get("initial_transaction_s"),
                "first_step_s": timing.get("first_step_s"),
                "steady_median_s": timing.get("steady_median_s"),
                "steady_spread_s": timing.get(
                    "steady_half_interdecile_spread_s"
                ),
                "max_normalised_residual": item.get("max_normalised_residual"),
                "failure_code": failure.get("code"),
                "failure_message": failure.get("message"),
            }
        )
    return rows


def _write_results_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    rows = _summary_rows(cases)
    with path.open("x", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("case_id,status\n")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pilot_specs(
    grids: tuple[int, ...],
    K: int,
    integrator: str,
    steps: int,
    omega: float,
    samples_per_cycle: float,
) -> tuple[CaseSpec, ...]:
    dt = 2.0 * math.pi / omega / samples_per_cycle
    return tuple(
        CaseSpec(
            case_id=f"pilot-{integrator}-steps{steps}-g{grid}-k{K}",
            kind="pilot",
            grid=grid,
            K=K,
            dt=dt,
            nsteps=steps,
            integrator=integrator,
            target_time_s=dt * steps,
        )
        for grid in grids
    )


def _frequency_reference(
    template_case: Path, modules: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    with template_case.open("rb") as handle:
        document = tomllib.load(handle)
    species = document["species"]
    references = document["solver"]["references"]
    constants = document["solver"]["constants"]
    scales = modules.stability.frequency_scales(
        ne=references["n_e_m3"],
        ni=references["n_i_m3"],
        B=references["B_T"],
        mi=species["mi"],
        me=species["me"],
        qi=species["qi"],
        qe=species["qe"],
        gamma_i=species["gamma_i"],
        epsilon0=constants["epsilon0_Fpm"],
        mu0=constants["mu0_Hpm"],
    )
    return scales, document


def _run_command_impl(arguments: argparse.Namespace, deadline: float) -> int:
    backend, device_ordinal = _normalise_device(arguments.device)
    if arguments.K != 1:
        raise ValueError("this harness currently requires K=1")
    grids = (
        _parse_int_list(arguments.grids, "grids", minimum=2)
        if arguments.command == "pilot"
        else (int(arguments.grid),)
    )
    nsteps_values = (
        _parse_int_list(arguments.nsteps, "nsteps", minimum=1)
        if arguments.command == "scan"
        else (int(arguments.steps),)
    )
    integrators = (
        _parse_integrators(arguments.integrators)
        if arguments.command == "scan"
        else (arguments.integrator,)
    )
    output = _create_output(Path(arguments.output))
    started = time.monotonic()
    root = _source_root()
    modules = _solver_modules()
    provenance = _base_provenance(arguments.command, sys.argv[1:], root)
    provenance.update(
        {
            "run_id": f"{arguments.command}__{output.name}__{uuid.uuid4().hex[:8]}",
            "output": str(output),
            "requested_device": arguments.device,
            "backend": backend,
            "device_ordinal": device_ordinal,
            "time_budget_s": arguments.time_budget_s,
            "time_budget_enforcement": (
                "SIGALRM preempts Python and NumPy solver work at the deadline; "
                "bounded evidence cleanup occurs after preemption and is not "
                "additional numerical work"
            ),
            "grids": list(grids),
            "K": arguments.K,
            "integrators": list(integrators),
            "nsteps": list(nsteps_values),
            "initial_condition": {
                "kind": "closed_reference_hermitian_electron_density_seed",
                "zero_flow": True,
                "density_fraction_peak": INITIAL_DENSITY_FRACTION,
                "density_modes": [-1, 1],
                "ion_density_changed": False,
                "pressures_changed": False,
                "magnetic_field_changed": False,
            },
        }
    )
    _archive_tool_sources(output)
    provenance["tool_source_archive"] = "provenance/tools"
    _write_json(output / "benchmark.json", provenance)
    _write_json(
        output / "input_metadata.json",
        {
            "controlled": provenance["initial_condition"],
            "reference_scope": (
                "n=1e12 m^-3, P=1e-9 Pa, B=1e-5 T are controlled numerical "
                "parameters, not tokamak values"
            ),
        },
    )
    cases = []
    all_observations = {}
    all_frequency_arrays = {}
    command_error = None
    status = "completed"
    try:
        template = _python_case_template()
        template_case = output / "work" / "python-template" / "case.toml"
        _write_text(template_case, template)
        builder_source = Path(__file__).resolve()
        provenance["case_builder"] = {
            "kind": "python",
            "path": str(builder_source),
            "sha256": _sha256(builder_source),
            "entry_point": "_python_case_template",
            "geometry": "fomhd.geometry.build_rectangle",
            "operators": "fomhd.io.load_case -> fomhd.operators.build_operators",
            "initial_state": "fomhd.stage.closed_reference",
            "state_io": "fomhd.io.encode_state and run-owned source manifest",
        }
        frequency_reference, _template_document = _frequency_reference(
            template_case, modules
        )
        omega = float(frequency_reference["omega_plasma"])
        provenance["time"] = {
            "omega_plasma_rad_s": omega,
            "omega_plasma_definition": "hypot(electron plasma frequency, ion plasma frequency)",
            "omega_plasma_scope": (
                "cold approximate reference; not an exact discrete or continuum eigenfrequency"
            ),
            "frequency_scales": frequency_reference,
        }
        if arguments.command == "scan":
            plan = build_scan_plan(
                grid=int(arguments.grid),
                K=int(arguments.K),
                cycles=float(arguments.cycles),
                nsteps_values=nsteps_values,
                integrators=integrators,
                frequency_reference=frequency_reference,
                minimum_samples=float(arguments.minimum_samples),
                amplitude_budget=float(arguments.amplitude_budget),
                phase_budget=float(arguments.phase_budget),
            )
            specs = plan.cases
            provenance["scan"] = {
                "target_time_s": plan.target_time_s,
                "cycles": float(arguments.cycles),
                "equal_final_time": True,
                "cold_reference_preassessment": list(plan.assessments),
                "runtime_guard_source": (
                    "per-case time_step_reports.json from result.time_step_report"
                ),
                "dt_guard_policy": "warn",
                "hard_gate_policy": "unchanged and fatal",
            }
            cold_preassessments = {
                item["case_id"]: item for item in plan.assessments
            }
        else:
            specs = _pilot_specs(
                grids,
                int(arguments.K),
                arguments.integrator,
                int(arguments.steps),
                omega,
                float(arguments.samples_per_cycle),
            )
            provenance["pilot"] = {
                "samples_per_cycle": float(arguments.samples_per_cycle),
                "dt_s": specs[0].dt,
                "warmup": "first timed actual transaction",
            }
        by_grid = {}
        for spec in specs:
            if spec.grid not in by_grid:
                by_grid[spec.grid] = _build_grid_inputs(
                    output,
                    template,
                    grid=spec.grid,
                    K=spec.K,
                    first_case=spec,
                    backend=backend,
                    device_ordinal=device_ordinal,
                )
        case_paths = {}
        for spec in specs:
            case_paths[spec.case_id] = _write_case(
                output,
                template,
                by_grid[spec.grid],
                spec,
                backend=backend,
                device_ordinal=device_ordinal,
            )
        for index, spec in enumerate(specs):
            if time.monotonic() >= deadline:
                raise BudgetExhausted(
                    f"wall-clock budget exhausted before case {spec.case_id}"
                )
            case_started = time.monotonic()
            outcome = _run_case(
                case_paths[spec.case_id],
                spec,
                deadline=deadline,
                case_started=case_started,
            )
            record = outcome["record"]
            if arguments.command == "scan":
                record["cold_time_preassessment"] = cold_preassessments.get(
                    spec.case_id
                )
                observations = outcome["observations"]
                series = np.asarray(
                    [
                        complex(item["ne_hat_real"], item["ne_hat_imag"])
                        for item in observations
                    ],
                    dtype=np.complex128,
                )
                times = np.asarray(
                    [item["time_s"] for item in observations], dtype=np.float64
                )
                fit, frequency_arrays = _frequency_analysis(
                    times, series, requested_cycles=float(arguments.cycles)
                )
                fit["omega_plasma_reference_rad_s"] = omega
                fit["relative_reference_difference"] = (
                    abs(fit["angular_frequency_rad_s"] - omega) / omega
                    if fit["status"] == "MEASURED"
                    else None
                )
                record["frequency"] = fit
                _write_json(case_paths[spec.case_id].parent / "result.json", record)
                all_frequency_arrays[spec.case_id] = frequency_arrays
            cases.append(record)
            all_observations[spec.case_id] = outcome["observations"]
            print(
                f"[{index + 1}/{len(specs)}] {spec.case_id}: "
                f"{record['status']} at step {record['accepted_step']}",
                flush=True,
            )
            if outcome["budget_exhausted"]:
                raise BudgetExhausted(
                    f"wall-clock budget preempted case {spec.case_id}: "
                    f"{outcome['failure']}"
                ) from outcome["failure"]
        if not cases:
            raise RuntimeError("no cases were executed")
    except BudgetExhausted as error:
        status = "budget_exhausted"
        command_error = _serialise_failure(error, None)
        _write_text(output / "logs" / "traceback.txt", traceback.format_exc())
    except Exception as error:
        status = "failed"
        command_error = _serialise_failure(error, None)
        _write_text(output / "logs" / "traceback.txt", traceback.format_exc())
    summaries = []
    for record in cases:
        summary_case = {
            key: value for key, value in record.items() if key != "failure"
        }
        failure = record.get("failure")
        if failure is not None:
            summary_case["failure"] = {
                "error_type": failure.get("error_type"),
                "code": failure.get("code"),
                "message": failure.get("message"),
                "wrapped_budget_exhausted": failure.get(
                    "wrapped_budget_exhausted", False
                ),
            }
        summaries.append(summary_case)
    if cases:
        _write_results_csv(output / "cases.csv", cases)
    if all_observations:
        _write_observations(
            output / "observations.npz",
            all_observations,
            all_frequency_arrays,
        )
    summary = {
        "schema": SCHEMA,
        "status": status,
        "run_id": provenance["run_id"],
        "command": arguments.command,
        "model_variant": "finite_mass",
        "requested_device": arguments.device,
        "backend": backend,
        "cases": summaries,
        "case_count": len(cases),
        "failure": command_error,
        "wallclock_s": time.monotonic() - started,
        "provenance_file": "benchmark.json",
        "observations_file": "observations.npz" if all_observations else None,
        "note": (
            "Only actual solver outcomes are stored. Failed cases are not "
            "replaced or interpolated."
        ),
    }
    _write_json(output / "summary.json", summary)
    provenance.update(
        {
            "status": status,
            "finished_utc": _utc_now(),
            "wallclock_s": summary["wallclock_s"],
            "case_count": len(cases),
            "failure": command_error,
        }
    )
    _write_json(output / "benchmark.json", provenance)
    _write_text(
        output / "logs" / "benchmark.log",
        json.dumps(_typed(summary), indent=2, allow_nan=False) + "\n",
    )
    if command_error:
        print(f"benchmark {status}: {command_error['message']}", file=sys.stderr)
        return 124 if status == "budget_exhausted" else 2
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run finite-mass FOMHD pilots or equal-final-horizon explicit "
            "integrator scans and retain raw outcomes."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--output", required=True)
    pilot.add_argument("--device", default="cpu", help="cpu or cuda:N")
    pilot.add_argument("--grids", default="4", help="comma-separated NR=NZ values")
    pilot.add_argument("--K", type=int, default=1)
    pilot.add_argument("--integrator", choices=("midpoint", "rk4"), default="rk4")
    pilot.add_argument("--steps", type=int, default=4)
    pilot.add_argument("--samples-per-cycle", type=float, default=64.0)
    pilot.add_argument("--time-budget-s", type=float, default=300.0)

    scan = subparsers.add_parser("scan")
    scan.add_argument("--output", required=True)
    scan.add_argument("--device", default="cpu", help="cpu or cuda:N")
    scan.add_argument("--grid", type=int, default=4)
    scan.add_argument("--K", type=int, default=1)
    scan.add_argument("--cycles", type=float, default=2.0)
    scan.add_argument("--nsteps", default="16,32")
    scan.add_argument("--integrators", default="midpoint,rk4")
    scan.add_argument("--minimum-samples", type=float, default=32.0)
    scan.add_argument("--amplitude-budget", type=float, default=0.01)
    scan.add_argument("--phase-budget", type=float, default=0.1)
    scan.add_argument("--time-budget-s", type=float, default=300.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "pilot":
        if arguments.steps < 1:
            raise ValueError("--steps must be positive")
        if (
            not math.isfinite(arguments.samples_per_cycle)
            or arguments.samples_per_cycle <= 0
        ):
            raise ValueError("--samples-per-cycle must be positive and finite")
    else:
        if arguments.grid < 2:
            raise ValueError("--grid must be >= 2")
    if (
        not math.isfinite(arguments.time_budget_s)
        or arguments.time_budget_s <= 0
    ):
        raise ValueError("--time-budget-s must be positive and finite")
    deadline = time.monotonic() + float(arguments.time_budget_s)
    with _hard_wallclock(float(arguments.time_budget_s)):
        return _run_command_impl(arguments, deadline)


if __name__ == "__main__":
    raise SystemExit(main())
