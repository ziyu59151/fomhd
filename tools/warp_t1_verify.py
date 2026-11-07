#!/usr/bin/env python3
"""Bounded A1-T1 waveform evidence against the preserved controlled T0 case."""

from __future__ import annotations

import argparse
import cmath
import copy
import gzip
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import traceback
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def _bootstrap_import_paths() -> Path:
    """Make direct script execution import tools/fomhd without PYTHONPATH."""
    candidate_root = Path(__file__).resolve().parents[1]
    candidates = []
    source_override = os.environ.get("FOMHD_SOURCE_ROOT")
    if source_override:
        candidates.append(Path(source_override))
    candidates.extend(
        (
            candidate_root,
            Path.cwd(),
            Path("/home/lxzeng/projects/fomhd"),
        )
    )
    source_root = next(
        (
            path.resolve()
            for path in candidates
            if (path / "src" / "fomhd" / "time.py").is_file()
        ),
        None,
    )
    for path in reversed(
        tuple(dict.fromkeys(path.resolve() for path in candidates))
    ):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    if source_root is not None:
        for path in (
            source_root / "tests" / "warp",
            source_root / "src",
            source_root,
        ):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        return source_root
    return candidate_root


SOURCE_ROOT = _bootstrap_import_paths()

from fomhd.stage import initialize
from fomhd.time import advance
from fomhd.types import Failure, cumulative_record
from tools import warp_t0_controlled as t0

SCHEMA_VERSION = 1
VALID_DEVICES = ("cpu", "cuda:0")
VALID_MICRO_PER_PERIOD = (2048, 4096)
VALID_SEEDS = ("target", "mixed")
MEMORY_CAP_BYTES = t0.MEMORY_CAP_BYTES
TRANSVERSE_INDEXES = t0.TRANSVERSE_INDEXES
DEFAULT_T0_BASELINE_ROOT = Path(
    "/home/lxzeng/runs/fomhd/a1-t0-controlled__20261106T163530Z/"
    "matrix-final/cases"
)
T1_FILES = (
    "tools/warp_t1_verify.py",
    "docs/pseudocode/tools/warp_t1_verify.py.md",
    "tests/warp/test_warp_t1_verify.py",
    "docs/pseudocode/tests/warp/test_warp_t1_verify.py.md",
)
BASELINE_ARRAY_KEYS = (
    "target_coeff_history",
    "high_coeff_history",
    "mode_m1_coeff_history",
    "mode_m4_coeff_history",
    "initial_ion_n",
    "initial_ion_momentum",
    "initial_ion_perp",
    "initial_electron_perp",
    "initial_electron_parallel",
    "initial_B",
    "accepted_ion_n",
    "accepted_ion_momentum",
    "accepted_ion_perp",
    "accepted_electron_perp",
    "accepted_electron_parallel",
    "accepted_B",
    "totals_number_ion_history",
    "totals_number_electron_history",
    "totals_kinetic_ion_history",
    "totals_kinetic_electron_history",
    "totals_ion_internal_J_history",
    "totals_electron_perp_J_history",
    "totals_electron_parallel_J_history",
    "totals_magnetic_J_history",
    "totals_total_energy_J_history",
)
SCHEDULE_FLOAT_KEYS = (
    "omega_target_rad_s",
    "target_period_s",
    "horizon_s",
    "micro_dt_s",
    "macro_dt_s",
)
SCHEDULE_INTEGER_KEYS = (
    "micro_per_period",
    "micro_steps",
    "cycles",
    "points",
    "total_micro_steps",
    "macro_steps",
)
SCOPE = (
    "Bounded A1-T1 periodic HallFixture(kdi=1, points=9), K=0, Nphi=1, "
    "using the real production state/ledger and one-macro-at-a-time waveform "
    "advance. This is not a production restart, closed-wall physical "
    "validation, asymptotic-stability proof, or performance claim."
)


def _finite(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_device(device: str) -> None:
    if device not in VALID_DEVICES:
        raise ValueError(f"device must be one of {VALID_DEVICES}")


def schedule(
    *,
    coupling: str,
    micro_per_period: int,
    micro_steps: int,
    cycles: int,
    points: int,
) -> dict[str, Any]:
    """Resolve the exact T0/T1 clocks from the reusable controlled helper."""
    return t0.schedule(
        coupling=coupling,
        micro_per_period=micro_per_period,
        micro_steps=micro_steps,
        cycles=cycles,
        points=points,
    )


def normalize_controls(
    *,
    device: str,
    M: int,
    micro_per_period: int,
    cycles: int,
    seed: str,
    rtol: float,
    max_iterations: int,
    points: int = 9,
) -> dict[str, Any]:
    """Validate all candidate controls and resolve the T0-compatible clock."""
    _validate_device(device)
    M = _positive_integer("M", M)
    if (
        isinstance(micro_per_period, bool)
        or type(micro_per_period) is not int
        or micro_per_period not in VALID_MICRO_PER_PERIOD
    ):
        raise ValueError(
            f"micro_per_period must be one of {VALID_MICRO_PER_PERIOD}"
        )
    cycles = _positive_integer("cycles", cycles)
    if seed not in VALID_SEEDS:
        raise ValueError(f"seed must be one of {VALID_SEEDS}")
    rtol = _finite("waveform_rtol", rtol, positive=True)
    max_iterations = _positive_integer(
        "waveform_max_iterations", max_iterations
    )
    if isinstance(points, bool) or type(points) is not int or points != 9:
        raise ValueError("points must be 9 for the preserved Hall fixture")
    clocks = schedule(
        coupling="predictor",
        micro_per_period=micro_per_period,
        micro_steps=M,
        cycles=cycles,
        points=points,
    )
    return {
        "device": device,
        "M": M,
        "micro_per_period": micro_per_period,
        "cycles": cycles,
        "seed": seed,
        "points": points,
        "coupling": "waveform",
        "integrator": "rk4",
        "dt_guard": "error",
        "waveform_rtol": rtol,
        "waveform_max_iterations": max_iterations,
        "memory_limit": MEMORY_CAP_BYTES,
        "device_memory_limit": MEMORY_CAP_BYTES,
        "schedule": clocks,
    }


def _default_baseline_root() -> Path:
    value = os.environ.get("FOMHD_T0_BASELINE_ROOT")
    return Path(value) if value else DEFAULT_T0_BASELINE_ROOT


def resolve_t0_baseline(path: str | Path | None, controls: Mapping[str, Any]) -> Path:
    if path is not None:
        candidate = Path(path)
    else:
        candidate = _default_baseline_root() / (
            f"t0-h{controls['micro_per_period']}-M{controls['M']}-"
            f"{controls['seed']}"
        )
    candidate = candidate.resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"T0 baseline directory does not exist: {candidate}")
    return candidate


def _require_fresh(output: str | Path) -> Path:
    answer = Path(output)
    if answer.exists():
        raise FileExistsError(f"refusing overwrite of existing output: {answer}")
    return answer


def _combined_hash(first: str, second: str) -> str:
    return hashlib.sha256(f"{first}:{second}".encode("ascii")).hexdigest()


def _load_t0_baseline(
    path: str | Path,
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one preserved exact-clock T0 case and load evidence arrays."""
    root = Path(path).resolve()
    case_path = root / "case.json"
    history_path = root / "history.npz"
    if not case_path.is_file() or not history_path.is_file():
        raise ValueError(
            "T0 baseline must contain case.json and history.npz"
        )
    try:
        case = json.loads(case_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read T0 baseline case: {error}") from error

    saved_controls = case.get("controls", {})
    expected_controls = {
        "seed": controls["seed"],
        "micro_per_period": controls["micro_per_period"],
        "micro_steps": controls["M"],
        "cycles": controls["cycles"],
        "points": controls["points"],
    }
    for name, expected in expected_controls.items():
        if saved_controls.get(name) != expected:
            raise ValueError(
                f"T0 baseline {name} mismatch: "
                f"{saved_controls.get(name)!r} != {expected!r}"
            )
    physical = case.get("physical", {})
    for name, expected in (("kdi", 1.0), ("K", 0), ("Nphi", 1)):
        actual = physical.get(name)
        if actual is None or not math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError(f"T0 baseline physical {name} mismatch")

    saved_schedule = case.get("schedule", {})
    clocks = controls["schedule"]
    for name in SCHEDULE_INTEGER_KEYS:
        if saved_schedule.get(name) != clocks.get(name):
            raise ValueError(
                f"T0 baseline schedule {name} mismatch: "
                f"{saved_schedule.get(name)!r} != {clocks.get(name)!r}"
            )
    for name in SCHEDULE_FLOAT_KEYS:
        if not math.isclose(
            float(saved_schedule.get(name, math.nan)),
            float(clocks[name]),
            rel_tol=2.0e-14,
            abs_tol=0.0,
        ):
            raise ValueError(f"T0 baseline schedule {name} mismatch")

    selected = {}
    with np.load(history_path, allow_pickle=False) as history:
        required = (
            "accepted_time_s",
            "target_coeff_history",
            "high_coeff_history",
        )
        missing = [name for name in required if name not in history.files]
        if missing:
            raise ValueError(
                f"T0 baseline history lacks required arrays: {missing}"
            )
        for name in ("accepted_time_s",) + BASELINE_ARRAY_KEYS:
            if name in history.files:
                selected[name] = np.asarray(history[name]).copy()

    saved_times = np.asarray(selected["accepted_time_s"], dtype=np.float64)
    expected_times = (
        float(clocks.get("start_time_s", 0.0))
        + np.arange(clocks["macro_steps"] + 1, dtype=np.float64)
        * float(clocks["macro_dt_s"])
    )
    if (
        saved_times.ndim != 1
        or saved_times.size < 1
        or saved_times.size > expected_times.size
    ):
        raise ValueError(
            "T0 baseline clock must be a nonempty original candidate prefix"
        )
    prefix_expected = expected_times[: saved_times.size]
    clock_abs_error = float(
        np.max(np.abs(saved_times - prefix_expected))
    )
    clock_tolerance = 4.0e-15 * max(
        abs(float(prefix_expected[-1])), 1.0
    )
    if clock_abs_error > clock_tolerance:
        raise ValueError(
            "T0 baseline clock prefix mismatch: "
            f"max |delta t|={clock_abs_error:.17g} s"
        )
    accepted_prefix_steps = int(saved_times.size - 1)
    saved_accepted = case.get("counts", {}).get(
        "accepted_macro_steps",
        case.get("accepted_macro_steps"),
    )
    if (
        saved_accepted is not None
        and int(saved_accepted) != accepted_prefix_steps
    ):
        raise ValueError(
            "T0 baseline accepted count differs from its clock prefix: "
            f"{saved_accepted} != {accepted_prefix_steps}"
        )
    if np.asarray(selected["target_coeff_history"]).shape[0] != saved_times.size:
        raise ValueError("T0 target modal history is not clock aligned")
    if np.asarray(selected["high_coeff_history"]).shape[0] != saved_times.size:
        raise ValueError("T0 high modal history is not clock aligned")
    for name, value in selected.items():
        if name.startswith("accepted_") and name != "accepted_time_s":
            if np.asarray(value).shape[0] != accepted_prefix_steps:
                raise ValueError(
                    f"T0 {name} history is not aligned to accepted endpoints"
                )

    case_hash = t0._sha256(case_path)
    history_hash = t0._sha256(history_path)
    return {
        "path": str(root),
        "case_json_path": str(case_path),
        "history_npz_path": str(history_path),
        "case_json_sha256": case_hash,
        "history_npz_sha256": history_hash,
        "combined_sha256": _combined_hash(case_hash, history_hash),
        "clock_alignment": {
            "aligned": True,
            "original_prefix": True,
            "complete_window": saved_times.size == expected_times.size,
            "sample_count": int(saved_times.size),
            "matched_macro_endpoints": accepted_prefix_steps,
            "expected_macro_steps": int(clocks["macro_steps"]),
            "future_macro_endpoints_unavailable": int(
                clocks["macro_steps"] - accepted_prefix_steps
            ),
            "maximum_absolute_error_s": clock_abs_error,
            "absolute_tolerance_s": clock_tolerance,
            "relative_tolerance": 0.0,
        },
        "case": case,
        "arrays": selected,
        "budgets": case.get("budgets", {}),
        "source": case.get("source", {}),
    }


def _synchronous_reference_path(
    baseline: Mapping[str, Any],
    controls: Mapping[str, Any],
) -> Path:
    return Path(baseline["path"]).parent / (
        f"sync-h{controls['micro_per_period']}-{controls['seed']}"
    )


def _synchronous_reference_input(
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: reference[key]
        for key in (
            "path",
            "case_json_path",
            "history_npz_path",
            "case_json_sha256",
            "history_npz_sha256",
            "combined_sha256",
            "identity",
            "compatibility",
            "metadata",
        )
    }


def _validate_synchronous_reference_identity(
    case: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    controls: Mapping[str, Any],
) -> None:
    saved_controls = case.get("controls", {})
    for name, expected in (
        ("coupling", "synchronous"),
        ("seed", controls["seed"]),
        ("points", controls["points"]),
    ):
        if saved_controls.get(name) != expected:
            raise ValueError(
                f"synchronous reference {name} differs: "
                f"{saved_controls.get(name)!r} != {expected!r}"
            )
    if not math.isclose(
        float(case.get("schedule", {}).get("horizon_s", math.nan)),
        float(controls["schedule"]["horizon_s"]),
        rel_tol=2.0e-12,
        abs_tol=0.0,
    ):
        raise ValueError("synchronous reference horizon differs")
    reference_case = t0._periodic_case(controls["points"])
    expected_identity = t0._case_identity(
        reference_case,
        controls["seed"],
    )
    if case.get("physical") != expected_identity["physical"]:
        raise ValueError("synchronous reference physical identity differs")
    if case.get("discrete") != expected_identity["discrete"]:
        raise ValueError("synchronous reference discrete identity differs")
    expected_seed = t0._seed_identity(
        t0.build_seed(points=controls["points"], seed=controls["seed"])
    )
    t0._validate_seed_identity(case.get("seed", {}), expected_seed)
    times = np.asarray(arrays.get("accepted_time_s", ()), dtype=np.float64)
    if (
        times.ndim != 1
        or times.size < 2
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
        or not math.isclose(times[0], 0.0, rel_tol=0.0, abs_tol=1.0e-15)
        or not math.isclose(
            float(times[-1]),
            float(controls["schedule"]["horizon_s"]),
            rel_tol=2.0e-12,
            abs_tol=0.0,
        )
    ):
        raise ValueError(
            "synchronous reference clocks do not span zero through the horizon"
        )


def _load_synchronous_reference(
    path: str | Path,
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the sibling fine-h synchronous case through the T0 verifier."""
    directory = Path(path).resolve()
    case_path = directory / "case.json"
    history_path = directory / "history.npz"
    if not case_path.is_file() or not history_path.is_file():
        raise ValueError(
            "synchronous reference must contain case.json and history.npz"
        )
    case = json.loads(case_path.read_text(encoding="utf-8"))
    compatibility = {
        "commit_mismatch_accepted": False,
        "numerical_source_hashes_equal": False,
        "reason": "t0._load_matched_reference validated all identities",
    }
    try:
        arrays, metadata = t0._load_matched_reference(
            directory,
            seed=controls["seed"],
            horizon=controls["schedule"]["horizon_s"],
            points=controls["points"],
        )
    except ValueError as error:
        compatible_errors = {
            "reference-case source commit differs",
            "matched reference numerical runtime source hashes differ",
        }
        if str(error) not in compatible_errors:
            raise
        with np.load(history_path, allow_pickle=False) as history:
            arrays = {
                name: np.asarray(history[name]).copy()
                for name in history.files
            }
        _validate_synchronous_reference_identity(case, arrays, controls)
        missing = [
            name
            for name in (
                "accepted_time_s",
                "target_coeff_history",
                "high_coeff_history",
            )
            if name not in arrays
        ]
        if missing:
            raise ValueError(
                f"synchronous reference history lacks arrays: {missing}"
            ) from error
        metadata = {
            "available": True,
            "path": str(directory),
            "status": case.get("status"),
            "validated_by": (
                "t0._load_matched_reference plus explicit non-source identity "
                "revalidation for the preserved synchronous baseline"
            ),
        }
        saved_hashes = case.get("source", {}).get("source_hashes", {})
        current_hashes = t0._runtime_source_hashes()
        different_files = [
            name
            for name in sorted(set(saved_hashes) | set(current_hashes))
            if saved_hashes.get(name) != current_hashes.get(name)
        ]
        saved_commit = (
            case.get("source", {}).get("source_git", {}).get("commit")
        )
        current_commit = t0._source_git().get("commit")
        compatibility = {
            "commit_mismatch_accepted": True,
            "numerical_source_hashes_equal": not different_files,
            "source_identity_mismatch_accepted": True,
            "different_source_files": different_files,
            "saved_source_commit": saved_commit,
            "current_source_commit": current_commit,
            "reason": (
                "preserved synchronous input/seed/physical/discrete/clock "
                "identity revalidated; source-version differences are retained "
                "explicitly because T1 integration changes runtime files"
            ),
        }
    if arrays is None:
        raise ValueError("synchronous reference arrays are unavailable")
    case_hash = t0._sha256(case_path)
    history_hash = t0._sha256(history_path)
    identity = {
        "coupling": case.get("controls", {}).get("coupling"),
        "controls": case.get("controls", {}),
        "schedule": case.get("schedule", {}),
        "physical": case.get("physical", {}),
        "discrete": case.get("discrete", {}),
        "seed": case.get("seed", {}),
        "status": case.get("status"),
        "source": case.get("source", {}),
    }
    return {
        "available": True,
        "path": str(directory),
        "case_json_path": str(case_path),
        "history_npz_path": str(history_path),
        "case_json_sha256": case_hash,
        "history_npz_sha256": history_hash,
        "combined_sha256": _combined_hash(case_hash, history_hash),
        "identity": identity,
        "compatibility": compatibility,
        "metadata": metadata,
        "case": case,
        "arrays": arrays,
    }


def _proposed_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _library_versions() -> dict[str, Any]:
    result = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    try:
        import scipy

        result["scipy"] = scipy.__version__
    except Exception:
        result["scipy"] = "unavailable"
    return result


def _add_t1_source_hashes(
    hashes: dict[str, str],
    source_root: Path,
) -> dict[str, str]:
    for relative in (
        "src/fomhd/time_waveform.py",
        "docs/pseudocode/src/fomhd/time_waveform.py.md",
    ):
        path = Path(source_root) / relative
        if path.is_file():
            hashes[relative] = t0._sha256(path)
    return hashes


def _build_provenance(
    argv: Sequence[str],
    device: str,
) -> dict[str, Any]:
    """Return source Git/dirty and all numerical/T1 file hashes."""
    hashes = dict(t0._runtime_source_hashes())
    _add_t1_source_hashes(hashes, t0.ROOT)
    root = _proposed_root()
    for relative in T1_FILES:
        path = root / relative
        if path.is_file():
            hashes[relative] = t0._sha256(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "bounded A1-T1 waveform evidence",
        "argv": list(argv),
        "utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
        },
        "selected_device": device,
        "precision": "float64/complex128",
        "source_git": t0._source_git(),
        "file_hashes": hashes,
        "library_versions": _library_versions(),
        "scope": SCOPE,
    }


def _verify_backend(context: Any, requested: str) -> dict[str, Any]:
    actual = getattr(context.compute, "device", None)
    name = getattr(context.compute, "name", None)
    expected_name = "cpu" if requested == "cpu" else "cuda"
    if actual != requested or name != expected_name:
        raise Failure(
            "BackendIdentity",
            f"requested {requested}, constructed {actual}/{name}; no fallback",
        )
    return {
        "requested": requested,
        "actual": actual,
        "backend_name": name,
        "fallback_used": False,
    }


_OWNING_STATE_FIELDS = ("n", "mom", "perp", "parallel")


def _seed_state_for_backend(
    built: Mapping[str, Any],
    context: Any,
) -> tuple[Any, dict[str, Any]]:
    """Keep CPU seed construction separate from selected-backend ownership."""
    backend = context.compute
    selected_device = getattr(backend, "device", None)
    if selected_device == "cpu":
        return built["state"], {
            "construction": "t0.build_seed CPU/NumPy default context",
            "selected_device": selected_device,
            "explicit_upload": False,
            "uploaded_field_count": 0,
            "uploaded_fields": [],
            "dtype": "complex128",
            "runtime_cpu_fallback": False,
        }
    if selected_device != "cuda:0":
        raise Failure(
            "SeedUpload",
            f"unsupported selected seed device: {selected_device!r}",
        )

    uploaded = copy.deepcopy(built["state"])
    uploaded_fields = []
    for species, fluid in enumerate(uploaded.fluid):
        for name in _OWNING_STATE_FIELDS:
            value = getattr(fluid, name)
            try:
                selected = backend.array(value, dtype=np.complex128)
            except Exception as error:
                raise Failure(
                    "SeedUpload",
                    f"fluid[{species}].{name} upload failed: {error}",
                ) from error
            expected_xp = getattr(backend, "xp", None)
            if expected_xp is not None and not isinstance(
                selected, expected_xp.ndarray
            ):
                raise Failure(
                    "SeedUpload",
                    f"fluid[{species}].{name} did not upload to {selected_device}",
                )
            setattr(fluid, name, selected)
            uploaded_fields.append(f"fluid[{species}].{name}")
    try:
        uploaded.B = backend.array(uploaded.B, dtype=np.complex128)
    except Exception as error:
        raise Failure("SeedUpload", f"B upload failed: {error}") from error
    expected_xp = getattr(backend, "xp", None)
    if expected_xp is not None and not isinstance(
        uploaded.B, expected_xp.ndarray
    ):
        raise Failure(
            "SeedUpload",
            f"B did not upload to {selected_device}",
        )
    uploaded_fields.append("B")
    return uploaded, {
        "construction": "t0.build_seed CPU/NumPy default context",
        "selected_device": selected_device,
        "explicit_upload": True,
        "uploaded_field_count": len(uploaded_fields),
        "uploaded_fields": uploaded_fields,
        "dtype": "complex128",
        "state_identity": {
            "step": int(uploaded.step),
            "generation": int(uploaded.generation),
            "time": float(uploaded.time),
        },
        "runtime_cpu_fallback": False,
    }


def _prepare_runtime(
    controls: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> SimpleNamespace:
    """Build the real fixture, state and cumulative ledger without advancing."""
    case = t0._periodic_case(controls["points"])
    context = t0.build_periodic_context(case, device=controls["device"])
    budgets_before = t0._jsonable(t0._budget_snapshot(context.p))
    context.p.memory_limit = controls["memory_limit"]
    context.p.device_memory_limit = controls["device_memory_limit"]
    context.p.coupling = "waveform"
    context.p.micro_steps = controls["M"]
    context.p.waveform_rtol = controls["waveform_rtol"]
    context.p.waveform_max_iterations = controls["waveform_max_iterations"]
    context.p.integrator = "rk4"
    context.p.dt = controls["schedule"]["macro_dt_s"]
    context.p.nsteps = controls["schedule"]["macro_steps"]
    context.p.dt_guard = "error"
    built = t0.build_seed(
        points=controls["points"],
        seed=controls["seed"],
    )
    built["context"] = context
    seed_state, seed_upload = _seed_state_for_backend(built, context)
    built["state"] = seed_state
    cumulative = cumulative_record()
    captured: dict[str, Any] = {}
    committed = initialize(
        seed_state,
        context,
        cumulative,
        equilibrium=False,
        captured=captured,
    )
    budgets_after = t0._jsonable(t0._budget_snapshot(context.p))
    if budgets_after != budgets_before:
        raise Failure(
            "BudgetMutation",
            "original solver budgets or hard flags changed during setup",
        )
    prepared = SimpleNamespace(
        controls=controls,
        baseline=baseline,
        schedule=controls["schedule"],
        case=case,
        context=context,
        seed=built,
        raw=seed_state,
        seed_upload=seed_upload,
        committed=committed,
        cumulative=cumulative,
        initial_checks=list(captured.get("checks", ())),
        initial_totals=captured.get("totals"),
        budgets=budgets_before,
        mode_specs=t0._mode_specs(built, case),
    )
    prepared.backend = _verify_backend(context, controls["device"])
    return prepared


def _validate_prepared_runtime(
    prepared: SimpleNamespace,
    controls: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    p = prepared.context.p
    expected = {
        "coupling": "waveform",
        "micro_steps": controls["M"],
        "waveform_rtol": controls["waveform_rtol"],
        "waveform_max_iterations": controls["waveform_max_iterations"],
        "integrator": "rk4",
        "dt": controls["schedule"]["macro_dt_s"],
        "nsteps": controls["schedule"]["macro_steps"],
        "dt_guard": "error",
        "memory_limit": controls["memory_limit"],
        "device_memory_limit": controls["device_memory_limit"],
    }
    for name, value in expected.items():
        actual = getattr(p, name, None)
        if isinstance(value, float):
            if not math.isclose(float(actual), value, rel_tol=0.0, abs_tol=0.0):
                raise Failure(
                    "Config", f"runtime {name} differs: {actual!r} != {value!r}"
                )
        elif actual != value:
            raise Failure(
                "Config", f"runtime {name} differs: {actual!r} != {value!r}"
            )
    if t0._jsonable(t0._budget_snapshot(p)) != baseline["budgets"]:
        raise Failure(
            "BudgetMutation",
            "runtime budgets differ from the preserved T0 baseline",
        )
    saved_seed = baseline["case"].get("seed")
    if saved_seed:
        t0._validate_seed_identity(
            saved_seed,
            t0._seed_identity(prepared.seed),
        )
    backend = _verify_backend(prepared.context, controls["device"])
    prepared.backend = backend
    return {
        "backend": backend,
        "original_budgets": t0._budget_snapshot(p),
        "seed_identity_equal_to_t0": True,
    }


def _capture_endpoint(
    state: Any,
    context: Any,
    case: Any,
    harmonics: Iterable[int],
) -> dict[str, Any]:
    """Explicit accepted-diagnostic host transfer; never called during a stage."""
    mode9 = {}
    transverse = {}
    for harmonic in harmonics:
        value = t0.project_state(state, context, case, harmonic)
        mode9[harmonic] = value
        transverse[harmonic] = value[list(TRANSVERSE_INDEXES)]
    state_arrays = {
        name.removeprefix("endpoint_"): np.asarray(value).copy()
        for name, value in t0._state_arrays(state, context, "endpoint").items()
    }
    return {
        "mode9": mode9,
        "transverse": transverse,
        "state_arrays": state_arrays,
    }


def _totals_to_dict(totals: Any) -> dict[str, float]:
    return t0._totals_arrays(totals)


def _snapshot_cumulative(
    cumulative: Any,
    committed: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        t0._cumulative_arrays(cumulative, committed),
        t0._cumulative_metadata(cumulative, committed),
    )


def _capture_seed_arrays(prepared: SimpleNamespace) -> dict[str, np.ndarray]:
    injected = getattr(prepared, "raw_state_arrays", None)
    if injected is not None:
        return {
            f"seed_{name}": np.asarray(value).copy()
            for name, value in injected.items()
        }
    return {
        f"seed_{name.removeprefix('raw_')}": np.asarray(value).copy()
        for name, value in t0._state_arrays(
            prepared.raw,
            prepared.context,
            "raw",
        ).items()
    }


def _extract_metadata(record: Any) -> dict[str, Any] | None:
    if record is None:
        return None
    value = getattr(record, "coupling_metadata", None)
    if value is None and isinstance(record, Mapping):
        value = record.get("coupling_metadata")
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {"value": value}


def _write_iteration_record(
    stream: Any,
    *,
    macro_index: int,
    accepted_before: int,
    scheduled_reject_time: float,
    status: str,
    metadata: Mapping[str, Any] | None,
    failure_code: str | None = None,
) -> dict[str, Any]:
    payload = {
        "macro_index": macro_index,
        "accepted_before": accepted_before,
        "status": status,
        "scheduled_reject_time_s": scheduled_reject_time,
        "failure_code": failure_code,
        "metadata_available": metadata is not None,
        "metadata": t0._jsonable(metadata) if metadata is not None else None,
    }
    stream.write(json.dumps(payload, sort_keys=True) + "\n")
    stream.flush()
    return payload


def _iterations_from_metadata(metadata: Mapping[str, Any] | None) -> list[Any]:
    if not metadata:
        return []
    value = metadata.get("iterations", ())
    return list(value) if isinstance(value, (list, tuple)) else []


def _iteration_identifier(item: Any, fallback: int) -> int:
    if isinstance(item, Mapping):
        for name in ("iteration", "sweep", "iteration_index", "index"):
            value = item.get(name)
            if type(value) is int and not isinstance(value, bool):
                return value
    return fallback


def _recursive_key_values(value: Any, pattern: re.Pattern[str]) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                name = str(key)
                if (
                    pattern.search(name.lower())
                    and not isinstance(child, (Mapping, list, tuple))
                    and not isinstance(child, (bool, np.bool_))
                ):
                    try:
                        number = float(child)
                    except (TypeError, ValueError):
                        number = math.nan
                    if math.isfinite(number):
                        found.setdefault(name, []).append(number)
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif hasattr(item, "__dict__"):
            visit(vars(item))

    visit(value)
    return found


def _sum_metadata_key(
    metadata_records: Sequence[Mapping[str, Any]],
    key: str,
) -> int | None:
    total = 0
    available = False
    for metadata in metadata_records:
        value = metadata.get(key)
        if type(value) is int and not isinstance(value, bool):
            total += value
            available = True
    return total if available else None


def _iteration_summary(
    metadata_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts = []
    identifiers = []
    for metadata in metadata_records:
        items = _iterations_from_metadata(metadata)
        counts.append(len(items))
        identifiers.extend(
            _iteration_identifier(item, index)
            for index, item in enumerate(items)
        )
    identifier_range = (
        [min(identifiers), max(identifiers)] if identifiers else None
    )
    path_values = _recursive_key_values(
        list(metadata_records),
        re.compile(
            r"path.*(?:relative|normalized)|(?:relative|normalized).*path"
        ),
    )
    endpoint_values = _recursive_key_values(
        list(metadata_records),
        re.compile(
            r"endpoint.*(?:relative|normalized)|"
            r"(?:relative|normalized).*endpoint"
        ),
    )
    normalization_semantics = (
        "normalized fixed-point update using the per-field scale "
        "max(background/physical reference, max|old|, max|new|); "
        "this uses actual-field scaling and is not a small-signal response "
        "or relative-error threshold"
    )

    def value_range(values: Mapping[str, list[float]]) -> dict[str, Any]:
        flat = [item for group in values.values() for item in group]
        if not flat:
            return {
                "available": False,
                "minimum": None,
                "maximum": None,
                "by_key": {},
                "semantics": normalization_semantics,
            }
        return {
            "available": True,
            "minimum": float(min(flat)),
            "maximum": float(max(flat)),
            "by_key": {
                key: {
                    "minimum": float(min(group)),
                    "maximum": float(max(group)),
                }
                for key, group in sorted(values.items())
            },
            "semantics": normalization_semantics,
        }

    return {
        "available": bool(metadata_records),
        "metadata_records": len(metadata_records),
        "per_macro": counts,
        "minimum_per_macro": min(counts) if counts else None,
        "maximum_per_macro": max(counts) if counts else None,
        "total": int(sum(counts)),
        "identifier_range": identifier_range,
        "path_relative": value_range(path_values),
        "endpoint_relative": value_range(endpoint_values),
    }


def _count_summary(metadata_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = (
        "complete_stage_evaluations",
        "full_rhs_stage_evaluations",
        "endpoint_full_evaluations",
        "alias_evaluations",
        "total_model_evaluations",
        "retained_quadrature_entries",
        "assemble_canonicalization_calls",
    )
    result = {
        name: _sum_metadata_key(metadata_records, name)
        for name in names
    }
    extra = _recursive_key_values(
        list(metadata_records),
        re.compile(r".*model.*evaluation.*"),
    )
    result["all_model_evaluation_field_maxima"] = {
        name: float(max(values))
        for name, values in sorted(extra.items())
    }
    return result


def _memory_summary(
    metadata_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values = _recursive_key_values(
        list(metadata_records),
        re.compile(r".*(memory|storage|peak|bytes).*"),
    )
    maxima = {
        name: float(max(group))
        for name, group in sorted(values.items())
        if group
    }
    return {
        "reported_estimates": maxima,
        "available": bool(maxima),
        "semantics": (
            "maximum scalar values reported in actual coupling metadata; "
            "not an independently measured process peak"
        ),
    }


def _array_comparison(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    unit: str,
) -> dict[str, Any]:
    observed = np.asarray(actual, dtype=np.complex128)
    expected = np.asarray(reference, dtype=np.complex128)
    if observed.shape != expected.shape:
        raise ValueError(
            f"comparison shape mismatch: {observed.shape} != {expected.shape}"
        )
    difference = np.abs(observed - expected)
    denominator = np.maximum(np.abs(expected), 1.0e-300)
    relative = difference / denominator
    return {
        "unit": unit,
        "maximum_absolute_error": float(np.max(difference))
        if difference.size
        else 0.0,
        "maximum_relative_error": float(np.max(relative))
        if relative.size
        else 0.0,
        "final_absolute_error": float(np.max(difference[-1]))
        if difference.size
        else 0.0,
        "final_relative_error": float(np.max(relative[-1]))
        if relative.size
        else 0.0,
        "zero_reference_semantics": (
            "relative error uses a 1e-300 floor; absolute error remains primary"
        ),
    }


def _canonicalize_initial_array(
    actual: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove one archived singleton time axis, never any other axis."""
    observed = np.asarray(actual, dtype=np.complex128)
    expected = np.asarray(reference, dtype=np.complex128)
    if observed.shape == expected.shape:
        return observed, expected
    if (
        expected.ndim == observed.ndim + 1
        and expected.shape[0] == 1
        and expected.shape[1:] == observed.shape
    ):
        return observed, expected[0]
    raise ValueError(
        "initial comparison shape mismatch after singleton-axis check: "
        f"{observed.shape} != {expected.shape}"
    )


def _vector_history_comparison(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    relative_defined: bool,
) -> dict[str, Any]:
    """Compare each complex vector row by Euclidean norm, never max component."""
    observed = np.asarray(actual, dtype=np.complex128)
    expected = np.asarray(reference, dtype=np.complex128)
    if observed.shape != expected.shape:
        raise ValueError(
            f"vector comparison shape mismatch: "
            f"{observed.shape} != {expected.shape}"
        )
    if observed.ndim != 2:
        raise ValueError("vector comparison histories must be two-dimensional")
    absolute = np.linalg.norm(observed - expected, axis=1)
    expected_norm = np.linalg.norm(expected, axis=1)
    observed_norm = np.linalg.norm(observed, axis=1)
    if relative_defined:
        if np.any(expected_norm <= 0.0):
            raise ValueError(
                "relative vector error is undefined for a zero reference row"
            )
        relative = absolute / expected_norm
        maximum_relative = float(np.max(relative)) if relative.size else 0.0
        final_relative = float(relative[-1]) if relative.size else 0.0
    else:
        relative = None
        maximum_relative = None
        final_relative = None
    return {
        "relative_defined": bool(relative_defined),
        "maximum_absolute_vector_error": float(np.max(absolute))
        if absolute.size
        else 0.0,
        "final_absolute_vector_error": float(absolute[-1])
        if absolute.size
        else 0.0,
        "maximum_relative_vector_error": maximum_relative,
        "final_relative_vector_error": final_relative,
        "maximum_absolute_norm_difference": float(
            np.max(np.abs(observed_norm - expected_norm))
        )
        if absolute.size
        else 0.0,
        "final_absolute_norm_difference": float(
            abs(observed_norm[-1] - expected_norm[-1])
        )
        if absolute.size
        else 0.0,
        "final_actual_norm": float(observed_norm[-1])
        if absolute.size
        else 0.0,
        "final_reference_norm": float(expected_norm[-1])
        if absolute.size
        else 0.0,
        "definition": (
            "each row uses ||actual-reference||_2 / ||reference||_2"
            if relative_defined
            else "absolute vector errors only; relative ratio is undefined"
        ),
    }


_STATE_UNITS = {
    "ion_n": "1/m^3",
    "ion_momentum": "kg/(m^2*s)",
    "ion_perp": "Pa",
    "electron_perp": "Pa",
    "electron_parallel": "Pa",
    "B": "T",
}


def _align_baseline_prefix(
    candidate_times: np.ndarray,
    baseline: Mapping[str, Any],
    key: str,
    *,
    accepted_rows: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Match the candidate prefix covered by a possibly rejected T0 history."""
    arrays = baseline["arrays"]
    if key not in arrays or key == "accepted_time_s":
        return None
    candidate = np.asarray(candidate_times, dtype=np.float64)
    reference_times = np.asarray(
        arrays["accepted_time_s"], dtype=np.float64
    )
    values = np.asarray(arrays[key])
    tolerance = 4.0e-15 * max(abs(float(reference_times[-1])), 1.0)
    if np.any(np.diff(candidate) < -tolerance):
        raise ValueError("candidate clocks are not monotone")
    covered = candidate <= reference_times[-1] + tolerance
    if not np.array_equal(covered, np.cumprod(covered, dtype=bool)):
        raise ValueError("candidate times exceed the T0 prefix before its end")
    candidate_indexes = []
    row_indexes = []
    for candidate_index in np.flatnonzero(covered):
        moment = float(candidate[candidate_index])
        matches = np.flatnonzero(
            np.isclose(
                reference_times,
                moment,
                rtol=2.0e-13,
                atol=tolerance,
            )
        )
        if matches.size != 1:
            raise ValueError(
                f"candidate time {moment:.17g} is not uniquely represented "
                "inside the T0 baseline prefix"
            )
        reference_index = int(matches[0])
        row_index = reference_index - 1 if accepted_rows else reference_index
        if row_index < 0 or row_index >= values.shape[0]:
            raise ValueError(
                f"T0 {key} history is not aligned to its clocks"
            )
        candidate_indexes.append(int(candidate_index))
        row_indexes.append(row_index)
    matched_count = len(candidate_indexes)
    coverage = {
        "baseline_sample_count": int(reference_times.size),
        "candidate_sample_count": int(candidate.size),
        "matched_sample_count": matched_count,
        "future_samples_unavailable": int(candidate.size - matched_count),
        "baseline_rows_are_accepted_only": accepted_rows,
        "baseline_prefix_only": reference_times.size
        < baseline["clock_alignment"]["expected_macro_steps"] + 1,
        "comparison_is_prefix_only": matched_count < candidate.size,
        "future_endpoints_compared": False,
        "semantics": (
            "only exact common clocks are compared; candidate endpoints after "
            "the preserved T0 rejected prefix remain retained but have no T0 "
            "reference"
        ),
    }
    return (
        candidate[np.asarray(candidate_indexes, dtype=np.int64)].copy(),
        values[np.asarray(row_indexes, dtype=np.int64)].copy(),
        coverage,
    )


def _baseline_comparisons(
    *,
    times: np.ndarray,
    target: np.ndarray,
    high: np.ndarray,
    initial_state_arrays: Mapping[str, np.ndarray],
    state_arrays: Mapping[str, np.ndarray],
    totals_by_name: Mapping[str, np.ndarray],
    baseline: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result: dict[str, Any] = {
        "path": baseline["path"],
        "combined_sha256": baseline["combined_sha256"],
        "clock_alignment": baseline["clock_alignment"],
    }
    arrays_to_save = {}
    target_match = _align_baseline_prefix(
        times, baseline, "target_coeff_history"
    )
    high_match = _align_baseline_prefix(
        times, baseline, "high_coeff_history"
    )
    if target_match is not None:
        _, saved_target, target_coverage = target_match
        arrays_to_save["t0_target_coeff_history"] = saved_target
        target_history_comparison = t0._accuracy_history(
            target[: saved_target.shape[0]],
            saved_target,
        )
        result["target_modal"] = {
            **target_history_comparison,
            "relative_modal_vector_error": float(
                np.max(
                    target_history_comparison["relative_vector_error"]
                )
            ),
            "maximum_relative_modal_vector_error": float(
                np.max(
                    target_history_comparison["relative_vector_error"]
                )
            ),
            "coverage": target_coverage,
            "definition": (
                "maximum row ||actual-reference||_2 / ||reference||_2"
            ),
        }
    if high_match is not None:
        _, saved_high, high_coverage = high_match
        arrays_to_save["t0_high_coeff_history"] = saved_high
        result["high_modal"] = {
            "coverage": high_coverage,
            "matched_reference_retained": True,
            "future_reference_available": not high_coverage[
                "comparison_is_prefix_only"
            ],
        }

    initial_comparison = {}
    for name, actual in initial_state_arrays.items():
        key = f"initial_{name}"
        if key not in baseline["arrays"]:
            continue
        expected = np.asarray(baseline["arrays"][key]).copy()
        arrays_to_save[f"t0_{key}"] = expected
        canonical_actual, canonical_expected = (
            _canonicalize_initial_array(actual, expected)
        )
        initial_comparison[name] = _array_comparison(
            canonical_actual,
            canonical_expected,
            unit=_STATE_UNITS.get(name, ""),
        )
    result["initial_state"] = initial_comparison

    state_comparison = {}
    for name, actual in state_arrays.items():
        key = f"accepted_{name}"
        match = _align_baseline_prefix(
            np.asarray(times[1:], dtype=np.float64),
            baseline,
            key,
            accepted_rows=True,
        )
        if match is None:
            continue
        _, expected, coverage = match
        arrays_to_save[f"t0_{key}"] = expected
        state_comparison[name] = _array_comparison(
            np.asarray(actual)[: expected.shape[0]],
            expected,
            unit=_STATE_UNITS.get(name, ""),
        )
        state_comparison[name]["coverage"] = coverage
    result["accepted_state"] = state_comparison

    totals_comparison = {}
    for name, actual in totals_by_name.items():
        key = f"totals_{name}_history"
        match = _align_baseline_prefix(times, baseline, key)
        if match is None:
            continue
        _, expected, coverage = match
        arrays_to_save[f"t0_{key}"] = expected
        totals_comparison[name] = _array_comparison(
            np.asarray(actual)[: expected.shape[0]],
            expected,
            unit="J" if name.endswith("_J") else "",
        )
        totals_comparison[name]["coverage"] = coverage
    result["totals"] = totals_comparison
    result["coverage"] = {
        **baseline["clock_alignment"],
        "candidate_macro_endpoints": int(times.size - 1),
        "future_endpoints_compared": False,
        "future_endpoints_fabricated": False,
    }
    result["semantics"] = (
        "T0 fixed-tangent production baseline at exactly matching accepted "
        "macro times; candidate endpoints beyond a preserved T0 rejected "
        "prefix are retained without a fabricated T0 reference"
    )
    return result, arrays_to_save


def _synchronous_comparisons(
    *,
    times: np.ndarray,
    target: np.ndarray,
    high: np.ndarray,
    seed: str,
    reference: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays = reference["arrays"]
    matched_target_times, matched_target = t0.align_reference_history(
        times,
        arrays,
        "target_coeff_history",
    )
    matched_high_times, matched_high = t0.align_reference_history(
        times,
        arrays,
        "high_coeff_history",
    )
    if not np.array_equal(matched_target_times, matched_high_times):
        raise ValueError("synchronous target/high clocks do not align")
    report = {
        **_synchronous_reference_input(reference),
        "available": True,
        "reference_source": "synchronous_fine_h_rk4",
        "matched_time_count": int(matched_target_times.size),
        "target": _vector_history_comparison(
            target,
            matched_target,
            relative_defined=True,
        ),
        "high": _vector_history_comparison(
            high,
            matched_high,
            relative_defined=(seed == "mixed"),
        ),
        "semantics": (
            "actual synchronous fine-h RK4 vectors evaluated at the same "
            "accepted T1 macro clocks"
        ),
    }
    return report, {
        "sync_target_coeff_history": matched_target,
        "sync_high_coeff_history": matched_high,
    }


def _phi(z: complex, order: int) -> complex:
    """Stable phi_order(z)=(exp(z)-Taylor_(order-1))/z^order."""
    value = complex(z)
    if order <= 0:
        raise ValueError("phi order must be positive")
    if abs(value) < 1.0e-3:
        total = 0.0j
        term = 1.0 / math.factorial(order)
        for index in range(64):
            total += term
            power = order + index
            term *= value / float(power + 1)
            if abs(term) <= 2.0e-17 * max(abs(total), 1.0e-300):
                break
        return complex(total)
    partial = sum(
        value**index / math.factorial(index)
        for index in range(order)
    )
    return complex((cmath.exp(value) - partial) / value**order)


def exact_t0_map(a: float, beta: float, H: float) -> np.ndarray:
    """Exact fast-system T0 map in A1_T0_LINEAR_GAIN's frozen-path limit."""
    a = _finite("a", a)
    beta = _finite("beta", beta)
    H = _finite("H", H)
    if H < 0.0:
        raise ValueError("H must be nonnegative")
    z = 1j * beta * H
    f0 = H * _phi(z, 1)
    f1 = H**2 * _phi(z, 2)
    f2 = H**3 * _phi(z, 3)
    return np.asarray(
        [
            [1.0 - a * a * f1, 1j * a * (f0 - a * a * f2)],
            [1j * a * f0, cmath.exp(z) - a * a * f1],
        ],
        dtype=np.complex128,
    )


def _model_rhs(
    a: float,
    beta: float,
    ion: complex,
    field: complex,
) -> tuple[complex, complex]:
    return 1j * a * field, 1j * a * ion + 1j * beta * field


def _relative_update(
    old: np.ndarray,
    new: np.ndarray,
) -> dict[str, float]:
    old_value = np.asarray(old, dtype=np.complex128)
    new_value = np.asarray(new, dtype=np.complex128)
    raw = float(np.max(np.abs(new_value - old_value)))
    scale = max(
        float(np.max(np.abs(old_value))),
        float(np.max(np.abs(new_value))),
        1.0e-300,
    )
    return {
        "raw_max_abs": raw,
        "scale": scale,
        "relative_max_abs": raw / scale,
    }


def linear_waveform_sweeps(
    a: float,
    beta: float,
    H: float,
    M: int,
    seed: Sequence[complex],
    max_iterations: int,
    *,
    rtol: float = 1.0e-10,
) -> dict[str, Any]:
    """Independent full-4M-stage reconstruction for the selected 2x2 model."""
    a = _finite("a", a)
    beta = _finite("beta", beta)
    H = _finite("H", H, positive=True)
    M = _positive_integer("M", M)
    max_iterations = _positive_integer("max_iterations", max_iterations)
    rtol = _finite("rtol", rtol, positive=True)
    initial = np.asarray(seed, dtype=np.complex128)
    if initial.shape != (2,) or not np.all(np.isfinite(initial)):
        raise ValueError("seed must be a finite complex two-vector")
    accepted = initial.copy()
    h = H / float(M)
    nodes = (0.0, 0.5, 0.5, 1.0)
    paths = np.empty((M, 4, 2), dtype=np.complex128)
    paths[:, :, 1] = initial[1]
    initial_ion_derivative = 1j * a * initial[1]
    for micro in range(M):
        for stage, offset in enumerate(nodes):
            paths[micro, stage, 0] = (
                initial[0] + (micro + offset) * h * initial_ion_derivative
            )
    endpoint = np.asarray(
        [initial[0] + H * initial_ion_derivative, initial[1]],
        dtype=np.complex128,
    )
    previous_z_observations = np.full(
        (M, 4),
        initial[1],
        dtype=np.complex128,
    )
    iterations = []
    converged = False
    for iteration in range(max_iterations):
        old_paths = paths.copy()
        old_endpoint = endpoint.copy()
        new_paths = np.empty_like(old_paths)
        z_stage_observations = np.empty((M, 4), dtype=np.complex128)
        ion = accepted[0]
        field = accepted[1]
        for micro in range(M):
            old_ion = old_paths[micro, :, 0]
            z_stage = np.empty(4, dtype=np.complex128)
            ion_rhs = np.empty(4, dtype=np.complex128)
            field_rhs = np.empty(4, dtype=np.complex128)
            z_stage[0] = field
            ion_rhs[0], field_rhs[0] = _model_rhs(
                a, beta, old_ion[0], z_stage[0]
            )
            for stage in range(1, 4):
                if stage == 1:
                    z_stage[stage] = field + 0.5 * h * field_rhs[0]
                elif stage == 2:
                    z_stage[stage] = field + 0.5 * h * field_rhs[1]
                else:
                    z_stage[stage] = field + h * field_rhs[2]
                ion_rhs[stage], field_rhs[stage] = _model_rhs(
                    a, beta, old_ion[stage], z_stage[stage]
                )
            z_stage_observations[micro, :] = z_stage
            new_paths[micro, :, 0] = (
                ion,
                ion + 0.5 * h * ion_rhs[0],
                ion + 0.5 * h * ion_rhs[1],
                ion + h * ion_rhs[2],
            )
            ion += h * (
                ion_rhs[0]
                + 2.0 * ion_rhs[1]
                + 2.0 * ion_rhs[2]
                + ion_rhs[3]
            ) / 6.0
            field += h * (
                field_rhs[0]
                + 2.0 * field_rhs[1]
                + 2.0 * field_rhs[2]
                + field_rhs[3]
            ) / 6.0
        new_paths[:, :, 1] = z_stage_observations
        new_endpoint = np.asarray([ion, field], dtype=np.complex128)
        ion_path_metric = _relative_update(
            old_paths[:, :, 0],
            new_paths[:, :, 0],
        )
        ion_endpoint_metric = _relative_update(
            old_endpoint[:1],
            new_endpoint[:1],
        )
        z_observation_metric = _relative_update(
            previous_z_observations,
            z_stage_observations,
        )
        record = {
            "iteration": iteration,
            "path": {
                "ion": dict(ion_path_metric),
                "z_stages_observation": dict(z_observation_metric),
            },
            "endpoint": {"ion": dict(ion_endpoint_metric)},
            "z_observation": dict(z_observation_metric),
            "converged": (
                ion_path_metric["relative_max_abs"] <= rtol
                and ion_endpoint_metric["relative_max_abs"] <= rtol
            ),
            "convergence_criteria": (
                "ion_stage_u_path_and_ion_endpoint"
            ),
        }
        iterations.append(record)
        paths = new_paths
        endpoint = new_endpoint
        if record["converged"]:
            converged = True
            break
        previous_z_observations = z_stage_observations
    return {
        "model": {
            "u_rhs": "i*a*b",
            "b_rhs": "i*a*u+i*beta*b",
            "method": "full four-stage reconstruction on each of M microsteps",
        },
        "a": a,
        "beta": beta,
        "H": H,
        "M": M,
        "seed": initial,
        "rtol": rtol,
        "max_iterations": max_iterations,
        "stopping_metric": "ion_stage_u_path_and_ion_endpoint",
        "z_observations_are_diagnostic_only": True,
        "converged": converged,
        "sweeps": len(iterations),
        "iterations": iterations,
        "path": paths,
        "z_stage_path": z_stage_observations,
        "endpoint": endpoint,
    }


def _write_input(
    output: Path,
    *,
    controls: Mapping[str, Any],
    baseline: Mapping[str, Any],
    synchronous_reference: Mapping[str, Any],
    provenance: Mapping[str, Any],
    argv: Sequence[str],
) -> dict[str, Any]:
    baseline_input = {
        key: baseline[key]
        for key in (
            "path",
            "case_json_path",
            "history_npz_path",
            "case_json_sha256",
            "history_npz_sha256",
            "combined_sha256",
            "clock_alignment",
        )
    }
    baseline_input["original_budgets"] = baseline["budgets"]
    baseline_input["source"] = baseline["source"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "A1-T1 waveform verification input",
        "argv": list(argv),
        "controls": controls,
        "schedule": controls["schedule"],
        "t0_predictor_baseline": baseline_input,
        "synchronous_reference": _synchronous_reference_input(
            synchronous_reference
        ),
        "source": {
            "source_git": provenance["source_git"],
            "file_hashes": provenance["file_hashes"],
            "selected_device": provenance["selected_device"],
            "precision": "float64/complex128",
        },
        "semantics": SCOPE,
    }
    t0._write_json(output / "input.json", payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=VALID_DEVICES, default="cpu")
    parser.add_argument("--M", type=int, default=4)
    parser.add_argument(
        "--micro-per-period",
        type=int,
        choices=VALID_MICRO_PER_PERIOD,
        default=2048,
    )
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--seed", choices=VALID_SEEDS, default="target")
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--points", type=int, default=9)
    parser.add_argument("--t0-baseline", type=Path)
    parser.add_argument("--progress-every", type=int, default=0)
    return parser


def run_case(
    output: str | Path,
    *,
    device: str = "cpu",
    M: int = 4,
    micro_per_period: int = 2048,
    cycles: int = 2,
    seed: str = "target",
    rtol: float = 1.0e-10,
    max_iterations: int = 12,
    points: int = 9,
    t0_baseline: str | Path | None = None,
    progress_every: int = 0,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run one bounded T1 case and retain complete accepted/failure evidence."""
    controls = normalize_controls(
        device=device,
        M=M,
        micro_per_period=micro_per_period,
        cycles=cycles,
        seed=seed,
        rtol=rtol,
        max_iterations=max_iterations,
        points=points,
    )
    output = _require_fresh(output)
    baseline_path = resolve_t0_baseline(t0_baseline, controls)
    baseline = _load_t0_baseline(baseline_path, controls)
    synchronous_reference = _load_synchronous_reference(
        _synchronous_reference_path(baseline, controls),
        controls,
    )
    output.mkdir(parents=True, exist_ok=False)
    invocation = list(argv) if argv is not None else [str(Path(__file__))]
    provenance = _build_provenance(invocation, device)
    provenance["synchronous_reference"] = _synchronous_reference_input(
        synchronous_reference
    )
    _write_input(
        output,
        controls=controls,
        baseline=baseline,
        synchronous_reference=synchronous_reference,
        provenance=provenance,
        argv=invocation,
    )
    t0._write_json(output / "provenance.json", provenance)

    expected_steps = int(controls["schedule"]["macro_steps"])
    macro_dt = float(controls["schedule"]["macro_dt_s"])
    start_time = float(controls["schedule"].get("start_time_s", 0.0))
    high_harmonic = (points - 1) // 2
    harmonics = (1, high_harmonic)
    mode9_history: dict[int, list[np.ndarray]] = {
        harmonic: [] for harmonic in harmonics
    }
    transverse_history: dict[int, list[np.ndarray]] = {
        harmonic: [] for harmonic in harmonics
    }
    state_history: dict[str, list[np.ndarray]] = {}
    totals_history: list[dict[str, float]] = []
    accepted_times: list[float] = []
    metadata_records: list[Mapping[str, Any]] = []
    prepared = None
    accepted = 0
    failure: Exception | None = None
    failure_traceback = ""
    scheduled_reject_time = start_time + macro_dt
    total_started = time.perf_counter()
    advance_started = total_started
    check_accumulator = t0._CheckAccumulator()
    check_count = 0

    checks_path = output / "checks.jsonl.gz"
    iterations_path = output / "iterations.jsonl"
    with gzip.open(checks_path, "xt", encoding="utf-8") as checks_stream:
        with iterations_path.open("x", encoding="utf-8") as iteration_stream:
            try:
                prepared = _prepare_runtime(controls, baseline)
                _validate_prepared_runtime(
                    prepared, controls, baseline
                )
                check_count += t0._stream_checks(
                    checks_stream,
                    "initial",
                    prepared.initial_checks,
                    check_accumulator,
                )
                if prepared.initial_totals is None:
                    raise RuntimeError("initial production totals unavailable")
                initial_capture = _capture_endpoint(
                    prepared.committed.state,
                    prepared.context,
                    prepared.case,
                    harmonics,
                )
                for harmonic in harmonics:
                    mode9_history[harmonic].append(
                        initial_capture["mode9"][harmonic]
                    )
                    transverse_history[harmonic].append(
                        initial_capture["transverse"][harmonic]
                    )
                for name, value in initial_capture["state_arrays"].items():
                    state_history.setdefault(f"initial_{name}", []).append(value)
                totals_history.append(
                    _totals_to_dict(prepared.initial_totals)
                )
                accepted_times.append(float(prepared.committed.state.time))
                prepared.context.compute.synchronize()
                advance_started = time.perf_counter()
                for macro in range(expected_steps):
                    scheduled_reject_time = start_time + (macro + 1) * macro_dt
                    try:
                        result = advance(
                            prepared.committed,
                            prepared.cumulative,
                            prepared.context,
                            None,
                        )
                    except Exception as error:
                        metadata = _extract_metadata(error)
                        _write_iteration_record(
                            iteration_stream,
                            macro_index=macro,
                            accepted_before=accepted,
                            scheduled_reject_time=scheduled_reject_time,
                            status="failed",
                            metadata=metadata,
                            failure_code=getattr(error, "code", type(error).__name__),
                        )
                        raise
                    metadata = _extract_metadata(result)
                    _write_iteration_record(
                        iteration_stream,
                        macro_index=macro,
                        accepted_before=accepted,
                        scheduled_reject_time=scheduled_reject_time,
                        status="accepted",
                        metadata=metadata,
                    )
                    if metadata is not None:
                        metadata_records.append(metadata)
                    check_count += t0._stream_checks(
                        checks_stream,
                        f"macro_{macro}",
                        getattr(result, "checks", ()),
                        check_accumulator,
                    )
                    accepted += 1
                    capture = _capture_endpoint(
                        prepared.committed.state,
                        prepared.context,
                        prepared.case,
                        harmonics,
                    )
                    accepted_times.append(float(prepared.committed.state.time))
                    for harmonic in harmonics:
                        mode9_history[harmonic].append(
                            capture["mode9"][harmonic]
                        )
                        transverse_history[harmonic].append(
                            capture["transverse"][harmonic]
                        )
                    for name, value in capture["state_arrays"].items():
                        state_history.setdefault(
                            f"accepted_{name}", []
                        ).append(value)
                    totals_history.append(_totals_to_dict(result.totals))
                    if progress_every > 0 and (
                        accepted == 1
                        or accepted == expected_steps
                        or accepted % progress_every == 0
                    ):
                        print(
                            f"accepted {accepted}/{expected_steps} macro steps",
                            file=sys.stderr,
                        )
                prepared.context.compute.synchronize()
            except Exception as error:
                failure_traceback = traceback.format_exc()
                if isinstance(error, Failure):
                    failure = error
                else:
                    failure = Failure("NumericalLibrary", str(error))
                check_count += t0._stream_checks(
                    checks_stream,
                    "failure",
                    getattr(failure, "checks", ()),
                    check_accumulator,
                )
    total_elapsed = time.perf_counter() - total_started
    advance_elapsed = max(0.0, time.perf_counter() - advance_started)
    check_summary = check_accumulator.finish()

    times = np.asarray(accepted_times, dtype=np.float64)
    mode9_arrays = {
        harmonic: np.asarray(values, dtype=np.complex128).reshape(-1, 9)
        for harmonic, values in mode9_history.items()
    }
    transverse_arrays = {
        harmonic: np.asarray(values, dtype=np.complex128).reshape(-1, 4)
        for harmonic, values in transverse_history.items()
    }
    target = transverse_arrays[1]
    high = transverse_arrays[high_harmonic]
    arrays: dict[str, Any] = {
        "accepted_time_s": times,
        "target_coeff_history": target,
        "high_coeff_history": high,
        "high_abs_amplitude_history": np.abs(high),
        "high_norm_history": np.linalg.norm(high, axis=1),
        **{
            f"mode_m{harmonic}_coeff_history": values
            for harmonic, values in mode9_arrays.items()
        },
        **{
            name: np.asarray(values)
            for name, values in state_history.items()
        },
    }
    totals_by_name: dict[str, np.ndarray] = {}
    if totals_history:
        for name in totals_history[0]:
            values = np.asarray(
                [record[name] for record in totals_history],
                dtype=np.float64,
            )
            totals_by_name[name] = values
            arrays[f"totals_{name}_history"] = values

    cumulative_metadata: dict[str, Any] = {"available": False}
    partial_names: list[str] = []
    partial_arrays: dict[str, np.ndarray] = {}
    if prepared is not None:
        cumulative_arrays, cumulative_metadata = _snapshot_cumulative(
            prepared.cumulative,
            prepared.committed,
        )
        arrays.update(cumulative_arrays)
        arrays.update(_capture_seed_arrays(prepared))
        if times.size:
            arrays["seed_state9"] = mode9_arrays[1][0]
            arrays["seed_target_coeff"] = target[0]
            arrays["seed_high_coeff"] = high[0]

    if failure is not None:
        for name, value in getattr(failure, "partial_arrays", {}).items():
            array = t0._array(value)
            partial_names.append(str(name))
            if array is not None:
                partial_arrays[str(name)] = array
                arrays[f"partial_{name}"] = array

    accepted_history_raw_path = output / "accepted-history-raw.npz"
    t0._save_npz(accepted_history_raw_path, arrays)

    accuracy = None
    reference_specs: dict[int, dict[str, Any]] = {}
    if prepared is not None:
        reference_specs = dict(prepared.mode_specs)
        if high_harmonic not in reference_specs:
            high_matrix, _ = t0._independent_matrix_for_k(
                prepared.case, high_harmonic * prepared.case.k
            )
            high_value, high_vector = t0.independent_whistler(high_matrix)
            embedded = np.zeros(9, dtype=np.complex128)
            embedded[list(TRANSVERSE_INDEXES)] = high_vector
            reference_specs[high_harmonic] = {
                "amplitude": 0.0,
                "vector": embedded,
                "transverse_vector": high_vector,
                "eigenvalue": high_value,
            }
    independent_references = (
        t0._references_at(reference_specs, times)
        if reference_specs and times.size
        else {}
    )
    per_harmonic = {}
    if independent_references:
        expected_target = independent_references[1][
            :, list(TRANSVERSE_INDEXES)
        ]
        target_accuracy = t0._accuracy_history(target, expected_target)
        arrays["target_reference_history"] = expected_target
        arrays["target_vector_error_history"] = target_accuracy[
            "relative_vector_error"
        ]
        arrays["target_amplitude_drift_history"] = target_accuracy[
            "amplitude_drift"
        ]
        arrays["target_phase_error_unwrapped_history"] = target_accuracy[
            "phase_error_unwrapped_rad"
        ]
        for harmonic in harmonics:
            actual_mode = mode9_arrays[harmonic]
            expected_mode = independent_references[harmonic]
            difference = np.linalg.norm(actual_mode - expected_mode, axis=1)
            denominator = np.linalg.norm(expected_mode, axis=1)
            relative_defined = harmonic == 1 or seed == "mixed"
            relative = None
            if relative_defined:
                if np.any(denominator <= 0.0):
                    raise ValueError(
                        f"m={harmonic} independent reference vector is zero; "
                        "relative error is unavailable"
                    )
                relative = difference / denominator
            per_harmonic[str(harmonic)] = {
                "maximum_absolute_error": float(np.max(difference)),
                "maximum_relative_error": (
                    float(np.max(relative))
                    if relative is not None and relative.size
                    else None
                ),
                "relative_defined": relative_defined,
            }
            arrays[f"mode_m{harmonic}_reference_history"] = expected_mode
            if relative is not None:
                arrays[f"mode_m{harmonic}_vector_error_history"] = relative
        qualification = t0.qualify_accuracy(
            vector_error=float(np.max(target_accuracy["relative_vector_error"])),
            amplitude_drift=float(np.max(target_accuracy["amplitude_drift"])),
            phase_error_rad=float(
                np.max(np.abs(target_accuracy["phase_error_unwrapped_rad"]))
            ),
        )
        full_success = failure is None and accepted == expected_steps
        if not full_success:
            qualification["joint_qualified"] = False
            qualification["partial"] = True
        accuracy = {
            **qualification,
            "partial": not full_success,
            "maximum_relative_vector_error": float(
                np.max(target_accuracy["relative_vector_error"])
            ),
            "maximum_amplitude_drift": float(
                np.max(target_accuracy["amplitude_drift"])
            ),
            "maximum_absolute_phase_error_rad": float(
                np.max(np.abs(target_accuracy["phase_error_unwrapped_rad"]))
            ),
            "final_relative_vector_error": float(
                target_accuracy["relative_vector_error"][-1]
            ),
            "final_amplitude_drift": float(
                target_accuracy["amplitude_drift"][-1]
            ),
            "final_phase_error_unwrapped_rad": float(
                target_accuracy["phase_error_unwrapped_rad"][-1]
            ),
            "reference_source": (
                "independent cold transverse Hall eigenvalue/eigenvector "
                "mode_specs, not the production Jacobian"
            ),
        }

    synchronous_comparison = {}
    synchronous_arrays = {}
    synchronous_high = None
    if times.size:
        synchronous_comparison, synchronous_arrays = (
            _synchronous_comparisons(
                times=times,
                target=target,
                high=high,
                seed=seed,
                reference=synchronous_reference,
            )
        )
        arrays.update(synchronous_arrays)
        synchronous_high = synchronous_arrays["sync_high_coeff_history"]

    predictor_high_match = (
        _align_baseline_prefix(times, baseline, "high_coeff_history")
        if times.size
        else None
    )
    if high.size:
        high_growth = t0._growth_summary(
            high,
            times,
            reference_history=synchronous_high,
            initial_ratio_defined=(seed == "mixed"),
        )
        if synchronous_high is not None:
            high_growth["matched_synchronous"]["reference_source"] = (
                "synchronous_fine_h_rk4"
            )
            high_growth["matched_synchronous"]["reference_identity"] = (
                _synchronous_reference_input(synchronous_reference)
            )
        if seed == "target" and synchronous_high is not None:
            high_growth["matched_synchronous"].update(
                {
                    "relative_modal_vector_error": None,
                    "maximum_relative_modal_vector_error": None,
                    "relative_values_undefined_reason": (
                        "target-only high harmonic is numerical noise; no "
                        "amplitude or relative ratio against noise is defined"
                    ),
                }
            )
        if predictor_high_match is not None:
            _, matched_high, coverage = predictor_high_match
            actual_prefix = high[: matched_high.shape[0]]
            predictor_report = _vector_history_comparison(
                actual_prefix,
                matched_high,
                relative_defined=(seed == "mixed"),
            )
            predictor_report.update(
                {
                    "reference_source": "t0_predictor_prefix",
                    "coverage": coverage,
                    "future_reference_available": not coverage[
                        "comparison_is_prefix_only"
                    ],
                }
            )
            if seed == "target":
                predictor_report.update(
                    {
                        "relative_modal_vector_error": None,
                        "amplitude_ratio_to_t0_predictor": None,
                        "maximum_amplitude_ratio_to_t0_predictor": None,
                        "final_amplitude_ratio_to_t0_predictor": None,
                        "relative_values_undefined_reason": (
                            "target-only high harmonic is numerical noise"
                        ),
                    }
                )
            else:
                actual_norms = np.linalg.norm(actual_prefix, axis=1)
                reference_norms = np.linalg.norm(matched_high, axis=1)
                ratios = actual_norms / reference_norms
                predictor_report.update(
                    {
                        "relative_modal_vector_error": predictor_report[
                            "maximum_relative_vector_error"
                        ],
                        "amplitude_ratio_to_t0_predictor": float(
                            actual_norms[-1] / reference_norms[-1]
                        ),
                        "maximum_amplitude_ratio_to_t0_predictor": float(
                            np.max(ratios)
                        ),
                        "final_amplitude_ratio_to_t0_predictor": float(
                            ratios[-1]
                        ),
                    }
                )
            high_growth["matched_t0_predictor"] = predictor_report
            if coverage["comparison_is_prefix_only"]:
                high_growth["future_t0_predictor_reference"] = {
                    "available": False,
                    "reason": (
                        "candidate advanced beyond the preserved T0 predictor "
                        "rejected prefix"
                    ),
                    "unavailable_accepted_endpoints": coverage[
                        "future_samples_unavailable"
                    ],
                }
        high_growth["initial_absolute_max_amplitude"] = float(
            np.max(np.abs(high[0]))
        )
        high_growth["seeded_norm_ratio_semantics"] = (
            "ratio to initialized high-harmonic norm; undefined for target-only "
            "because its absolute noise amplitude is retained instead"
        )
        ratio_history = high_growth.get("amplitude_ratio_history")
        if ratio_history is not None:
            arrays["high_amplitude_ratio_history"] = np.asarray(ratio_history)
    else:
        high_growth = {
            "available": False,
            "reason": "no accepted state was initialized",
            "maximum_amplitude_ratio": None,
            "final_amplitude_ratio": None,
        }

    baseline_comparison = {}
    saved_baseline_arrays = {}
    initial_state_arrays = {
        name.removeprefix("initial_"): np.asarray(values[0])
        for name, values in state_history.items()
        if name.startswith("initial_") and values
    }
    accepted_state_arrays = {
        name.removeprefix("accepted_"): np.asarray(values)
        for name, values in state_history.items()
        if name.startswith("accepted_")
    }
    if times.size:
        baseline_comparison, saved_baseline_arrays = _baseline_comparisons(
            times=times,
            target=target,
            high=high,
            initial_state_arrays=initial_state_arrays,
            state_arrays=accepted_state_arrays,
            totals_by_name=totals_by_name,
            baseline=baseline,
        )
        arrays.update(saved_baseline_arrays)

    iteration_metadata_for_summary = list(metadata_records)
    failure_metadata = _extract_metadata(failure) if failure is not None else None
    if failure_metadata is not None:
        iteration_metadata_for_summary.append(failure_metadata)
    iteration_summary = _iteration_summary(iteration_metadata_for_summary)
    counts_summary = _count_summary(iteration_metadata_for_summary)
    memory_summary = _memory_summary(iteration_metadata_for_summary)
    t0._save_npz(output / "history.npz", arrays)

    if failure is not None or accepted != expected_steps:
        status = "UNDETERMINED"
    elif check_summary["hard_failed"]:
        status = "FAILED_HARD_CHECKS"
    elif check_summary["soft_failed"]:
        status = "COMPLETED_WITH_SOFT_FAILURES"
    else:
        status = "COMPLETED"
    final_endpoint_available = status.startswith("COMPLETED")
    case_identity = (
        t0._case_identity(prepared.case, seed)
        if prepared is not None
        else {"physical": None, "discrete": None}
    )
    high_max_ratio = high_growth.get("maximum_amplitude_ratio")
    high_final_ratio = high_growth.get("final_amplitude_ratio")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "case_id": (
            f"t1_waveform_mpp{micro_per_period}_M{M}_{seed}"
        ),
        "scope": SCOPE,
        "controls": controls,
        "schedule": controls["schedule"],
        "source": provenance,
        "physical": case_identity["physical"],
        "discrete": case_identity["discrete"],
        "backend": (
            getattr(prepared, "backend", None)
            if prepared is not None
            else {"requested": device, "actual": None, "fallback_used": False}
        ),
        "budgets": (
            t0._jsonable(t0._budget_snapshot(prepared.context.p))
            if prepared is not None
            else baseline["budgets"]
        ),
        "seed_initialization": (
            prepared.seed_upload
            if prepared is not None
            else {"available": False}
        ),
        "original_budgets_preserved": prepared is not None,
        "accepted_macro_steps": accepted,
        "expected_macro_steps": expected_steps,
        "horizon_s": controls["schedule"]["horizon_s"],
        "last_accepted_time_s": float(times[-1]) if times.size else None,
        "final_endpoint_available": final_endpoint_available,
        "invented_rejected_endpoint": False,
        "target_max_relative_vector_error": (
            accuracy.get("maximum_relative_vector_error")
            if accuracy is not None
            else None
        ),
        "target_phase": {
            "maximum_absolute_unwrapped_error_rad": (
                accuracy.get("maximum_absolute_phase_error_rad")
                if accuracy is not None
                else None
            ),
            "final_error_rad": (
                accuracy.get("final_phase_error_unwrapped_rad")
                if accuracy is not None
                else None
            ),
        },
        "target_amplitude_drift": (
            accuracy.get("maximum_amplitude_drift")
            if accuracy is not None
            else None
        ),
        "accuracy": accuracy,
        "per_harmonic_independent_error": per_harmonic,
        "high_frequency_growth": {
            **high_growth,
            "maximum_seeded_norm_ratio": high_max_ratio,
            "final_seeded_norm_ratio": high_final_ratio,
            "target_only_ratio_undefined": seed == "target",
        },
        "synchronous_reference": synchronous_comparison,
        "t0_predictor_baseline": baseline_comparison,
        "iteration_controls": {
            "waveform_rtol": controls["waveform_rtol"],
            "waveform_max_iterations": controls["waveform_max_iterations"],
        },
        "iterations": iteration_summary,
        "counts": {
            "advance_calls": accepted + (1 if failure is not None else 0),
            "accepted_macro_steps": accepted,
            "raw_check_records": check_count,
            "model_calls": counts_summary,
        },
        "memory": memory_summary,
        "check_summary": check_summary,
        "energy": {
            "raw_checks_file": checks_path.name,
            "original_budgets": (
                t0._jsonable(t0._budget_snapshot(prepared.context.p))
                if prepared is not None
                else baseline["budgets"]
            ),
            "semantics": (
                "all raw hard/soft and J-unit checks streamed unchanged; no "
                "scientific threshold was relaxed"
            ),
        },
        "cumulative": cumulative_metadata,
        "timing": {
            "total_wall_seconds": total_elapsed,
            "advance_wall_seconds": advance_elapsed,
        },
        "failure": (
            {
                "failure_code": getattr(failure, "code", type(failure).__name__),
                "failure_file": "failure.json",
                "failure_arrays_file": "failure.npz",
            }
            if failure is not None
            else None
        ),
        "artifacts": {
            "input_json": "input.json",
            "case_json": "case.json",
            "history_npz": "history.npz",
            "accepted_history_raw_npz": accepted_history_raw_path.name,
            "checks_jsonl_gzip": checks_path.name,
            "iterations_jsonl": iterations_path.name,
            "provenance_json": "provenance.json",
        },
        "limitations": [
            "periodic temporal-isolation fixture only",
            "m is a spatial harmonic, not toroidal K",
            "independent cold REF does not cover ion-pressure branches",
            "finite accepted endpoints do not prove continuous-time or asymptotic stability",
            "CPU and CUDA require separate measured cases",
        ],
    }

    if failure is not None:
        failure_arrays = {
            name: value
            for name, value in arrays.items()
            if (
                name.startswith("partial_")
                or name.startswith("accepted_")
                or name.startswith("initial_")
                or name.startswith("totals_")
                or name.startswith("cumulative_")
                or name in ("accepted_time_s", "target_coeff_history", "high_coeff_history")
            )
        }
        for name, value in partial_arrays.items():
            failure_arrays[f"partial_{name}"] = value
        t0._save_npz(output / "failure.npz", failure_arrays)
        failure_payload = {
            "case_id": summary["case_id"],
            "status": "UNDETERMINED",
            "failure_code": getattr(failure, "code", type(failure).__name__),
            "failure_type": type(failure).__name__,
            "message": str(failure),
            "traceback": failure_traceback,
            "failure_stage": t0._jsonable(getattr(failure, "stage", None)),
            "checks": t0._jsonable(getattr(failure, "checks", ())),
            "coupling_metadata": t0._jsonable(
                getattr(failure, "coupling_metadata", None)
            ),
            "partial_array_names": sorted(partial_names),
            "accepted_macro_steps": accepted,
            "expected_macro_steps": expected_steps,
            "scheduled_rejected_time_s": scheduled_reject_time,
            "last_accepted_time_s": float(times[-1]) if times.size else None,
            "final_endpoint_available": False,
            "invented_rejected_endpoint": False,
            "cumulative": cumulative_metadata,
        }
        t0._write_json(output / "failure.json", failure_payload)
    t0._write_json(output / "case.json", summary)
    for name in (
        "input_json",
        "history_npz",
        "accepted_history_raw_npz",
        "checks_jsonl_gzip",
        "iterations_jsonl",
        "provenance_json",
        "case_json",
    ):
        path = output / summary["artifacts"][name]
        summary["artifacts"][f"{name}_bytes"] = path.stat().st_size
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(arguments)
    invocation = [str(Path(__file__).resolve()), *arguments]
    try:
        summary = run_case(
            args.output,
            device=args.device,
            M=args.M,
            micro_per_period=args.micro_per_period,
            cycles=args.cycles,
            seed=args.seed,
            rtol=args.rtol,
            max_iterations=args.max_iterations,
            points=args.points,
            t0_baseline=args.t0_baseline,
            progress_every=args.progress_every,
            argv=invocation,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "accepted_macro_steps": summary["accepted_macro_steps"],
                "expected_macro_steps": summary["expected_macro_steps"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"].startswith("COMPLETED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
