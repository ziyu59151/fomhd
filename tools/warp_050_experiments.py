#!/usr/bin/env python3
"""Bounded complete A0/T1 0.5.0 experiment harness (scratch-only)."""

from __future__ import annotations

import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import copy
import hashlib
import json
import math
import resource
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

SCRATCH_ROOT = Path(__file__).resolve().parents[1]
HOST_MEMORY_CAP = 16 << 30
DEVICE_MEMORY_CAP = 8 << 30
MAX_WALL_SECONDS = 1600.0
VALID_NODES = (12, 16, 24, 32, 40)
SEED_NAMES = (
    "seed_n",
    "seed_mom",
    "seed_pi",
    "seed_perp",
    "seed_parallel",
    "seed_B",
)
BACKGROUND_NAMES = (
    "background_B",
    "background_n",
    "background_mom",
    "background_pi",
    "background_perp",
    "background_parallel",
)
EIGENPAIR_NAMES = ("jacobian", "eigenvalues", "eigenvectors")

import fomhd  # noqa: E402
from fomhd import core, diagnostics, io, stability, stage  # noqa: E402
from fomhd.energy import summarize_energy  # noqa: E402
from fomhd.operators import OperatorOptions  # noqa: E402
from fomhd.time import advance  # noqa: E402
from fomhd.types import StageTag, cumulative_record  # noqa: E402
from tools import warp_tokamak_cae as cae  # noqa: E402

physical_identity = cae.physical_identity


def source_identity() -> dict[str, Any]:
    """Return and validate the imported runtime source identity."""
    package = Path(fomhd.__file__).resolve()
    source_root = package.parents[2]
    expected = os.environ.get("FOMHD_SOURCE_ROOT")
    if expected is not None and Path(expected).resolve() != source_root:
        raise RuntimeError(
            "imported fomhd source does not match FOMHD_SOURCE_ROOT: "
            f"imported={source_root} expected={Path(expected).resolve()}"
        )
    return {
        "source_root": str(source_root),
        "fomhd_package": str(package),
        "time_module": str(Path(sys.modules["fomhd.time"].__file__).resolve()),
        "python": sys.executable,
        "expected_source_root": None if expected is None else str(Path(expected).resolve()),
        "validated": expected is None or Path(expected).resolve() == source_root,
    }


def _finite_positive(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be positive and finite") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive non-boolean integer")
    return value


def _positive_integer_argument(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        ) from error
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _h_argument(text: str) -> float | str:
    if text == "auto":
        return text
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("h must be positive or auto") from error
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("h must be positive or auto")
    return value


def _wall_argument(text: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("wall seconds must be finite") from error
    if not math.isfinite(value) or value <= 0.0 or value > MAX_WALL_SECONDS:
        raise argparse.ArgumentTypeError(
            f"wall seconds must be in (0, {MAX_WALL_SECONDS:g}]"
        )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("circle", "east"), required=True)
    parser.add_argument("--nodes", type=int, choices=VALID_NODES, required=True)
    parser.add_argument("--K", type=int, choices=(0, 1), required=True)
    parser.add_argument("--mode", type=int, choices=(0, 1), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--equilibrium", type=Path)
    parser.add_argument(
        "--coupling", choices=("synchronous", "waveform"), required=True
    )
    parser.add_argument("--micro-steps", type=int, default=1)
    parser.add_argument("--horizon-s", type=float, required=True)
    parser.add_argument("--h", type=_h_argument, default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=_wall_argument, default=MAX_WALL_SECONDS)
    parser.add_argument("--probe-macros", type=_positive_integer_argument)
    parser.add_argument(
        "--snapshot-every", type=_positive_integer_argument, default=16
    )
    parser.add_argument("--target-frequency-hz", type=float)
    parser.add_argument("--spectrum-directory", type=Path)
    parser.add_argument(
        "--stability-spectrum-directory",
        dest="stability_spectrum_directories",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--amplitude", type=float, default=1.0e-4)
    return parser


def plan_time(
    horizon_s: float,
    h_s: float,
    micro_steps: int,
    coupling: str,
) -> dict[str, Any]:
    """Resolve an exact physical horizon for a fixed physical micro-step."""
    horizon = _finite_positive(horizon_s, "horizon_s")
    h = _finite_positive(h_s, "h_s")
    micro_steps = _positive_integer(micro_steps, "micro_steps")
    if coupling not in ("synchronous", "waveform"):
        raise ValueError("coupling must be synchronous or waveform")
    if coupling == "synchronous":
        micro_steps = 1
        macro_dt = h
    else:
        macro_dt = micro_steps * h
    ratio = horizon / macro_dt
    nsteps = int(round(ratio))
    if nsteps < 1 or abs(ratio - nsteps) > 1.0e-9 * max(1.0, nsteps):
        raise ValueError(
            "horizon_s must be an exact integer number of "
            f"{coupling} macro intervals: horizon/macro_dt={ratio:.17g}"
        )
    return {
        "coupling": coupling,
        "h_s": h,
        "micro_steps": micro_steps,
        "dt_s": macro_dt,
        "nsteps": nsteps,
        "horizon_s": macro_dt * nsteps,
        "micro_steps_total": nsteps * micro_steps,
    }


def snapshot_steps(nsteps: int, snapshot_every: int) -> list[int]:
    """Return bounded macro snapshot indices, including the final step."""
    nsteps = _positive_integer(nsteps, "nsteps")
    snapshot_every = _positive_integer(snapshot_every, "snapshot_every")
    values = list(range(snapshot_every, nsteps + 1, snapshot_every))
    if not values or values[-1] != nsteps:
        values.append(nsteps)
    return values


def _validate_args(args: Any, *, allow_fixture_nodes: bool) -> dict[str, Any]:
    kind = str(args.kind)
    if kind not in ("circle", "east"):
        raise ValueError("kind must be circle or east")
    nodes = int(args.nodes)
    if nodes not in VALID_NODES and not (allow_fixture_nodes and nodes == 4):
        raise ValueError(f"nodes must be one of {VALID_NODES}")
    K = int(args.K)
    mode = int(args.mode)
    if K not in (0, 1) or mode not in (0, 1) or mode > K:
        raise ValueError("K and mode must be 0/1 with mode <= K")
    coupling = str(args.coupling)
    if coupling not in ("synchronous", "waveform"):
        raise ValueError("coupling must be synchronous or waveform")
    micro_steps = _positive_integer(args.micro_steps, "micro_steps")
    if coupling == "synchronous" and micro_steps != 1:
        raise ValueError("synchronous coupling requires micro_steps=1")
    horizon = _finite_positive(args.horizon_s, "horizon_s")
    wall_seconds = _finite_positive(args.wall_seconds, "wall_seconds")
    if wall_seconds > MAX_WALL_SECONDS:
        raise ValueError("wall_seconds must not exceed 1600")
    snapshot_every = _positive_integer(args.snapshot_every, "snapshot_every")
    probe_macros = getattr(args, "probe_macros", None)
    if probe_macros is not None:
        probe_macros = _positive_integer(probe_macros, "probe_macros")
    target = getattr(args, "target_frequency_hz", None)
    if target is not None:
        target = _finite_positive(target, "target_frequency_hz")
    amplitude = float(getattr(args, "amplitude", 1.0e-4))
    if not math.isfinite(amplitude) or not 0.0 < amplitude <= 1.0e-2:
        raise ValueError("amplitude must be finite in (0, 0.01]")
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"output must be a NEW path: {output}")
    resume_from = getattr(args, "resume_from", None)
    if resume_from is not None:
        resume_from = Path(resume_from).resolve()
        if not resume_from.is_dir():
            raise ValueError(f"resume_from must be an existing directory: {resume_from}")
    stability_directories = [
        Path(value).resolve()
        for value in getattr(args, "stability_spectrum_directories", [])
    ]
    h_requested = getattr(args, "h", "auto")
    if h_requested != "auto":
        h_requested = _finite_positive(h_requested, "h")
    if kind == "east" and getattr(args, "equilibrium", None) is None:
        raise ValueError("EAST requires --equilibrium")
    return {
        "kind": kind,
        "nodes": nodes,
        "K": K,
        "mode": mode,
        "device": str(getattr(args, "device", "cpu")),
        "coupling": coupling,
        "micro_steps": micro_steps,
        "horizon_s": horizon,
        "h_requested": h_requested,
        "output": output.resolve(),
        "wall_seconds": wall_seconds,
        "probe_macros": probe_macros,
        "snapshot_every": snapshot_every,
        "target_frequency_hz": target,
        "amplitude": amplitude,
        "equilibrium": getattr(args, "equilibrium", None),
        "spectrum_directory": getattr(args, "spectrum_directory", None),
        "stability_spectrum_directories": stability_directories,
        "resume_from": resume_from,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _collection_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    """Legacy archive source list; mismatches are disclosed, not gates."""
    root = Path(fomhd.__file__).resolve().parent
    return {
        path.name: _sha256_file(path)
        for path in sorted(root.glob("*.py"))
        if path.name != "energy.py"
    }


def _runtime_fingerprint() -> dict[str, Any]:
    """Hash every runtime source file used by strict experimental continuation."""
    package = Path(fomhd.__file__).resolve().parent
    files = {
        f"src/fomhd/{path.name}": _sha256_file(path)
        for path in sorted(package.glob("*.py"))
    }
    files[f"runner/{Path(__file__).name}"] = _sha256_file(Path(__file__))
    helper = Path(cae.__file__).resolve()
    files[f"seed_helper/{helper.name}"] = _sha256_file(helper)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "algorithm": "sha256-files-v1",
        "files": files,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _git_head(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def _array_identity(value: Any, context: Any) -> dict[str, Any] | None:
    cupy_module = sys.modules.get("cupy")
    cupy_array = getattr(cupy_module, "ndarray", None)
    if isinstance(value, np.ndarray):
        host = np.asarray(value)
    elif cupy_array is not None and isinstance(value, cupy_array):
        host = np.asarray(cae.host(value, context))
    else:
        return None
    return {
        "dtype": str(host.dtype),
        "shape": list(host.shape),
        "sha256": hashlib.sha256(
            np.ascontiguousarray(host).tobytes()
        ).hexdigest(),
    }


def _object_array_signature(value: Any, context: Any) -> dict[str, Any]:
    return {
        name: identity
        for name, child in sorted(vars(value).items())
        if (identity := _array_identity(child, context)) is not None
    }


def _compatibility_identity(
    context: Any,
    controls: dict[str, Any],
    plan: dict[str, Any],
    *,
    seed_sha256: str,
) -> dict[str, Any]:
    target = controls.get("target_frequency_hz")
    return {
        "case": {
            "kind": controls["kind"],
            "nodes": int(controls["nodes"]),
            "K": int(controls["K"]),
            "mode": int(controls["mode"]),
            "device": controls["device"],
            "Nphi": int(context.N),
            "electron_inertia": int(context.p.electron_inertia),
            "integrator": context.p.integrator,
            "amplitude": float(controls["amplitude"]),
        },
        "time": {
            "coupling": plan["coupling"],
            "micro_steps": int(plan["micro_steps"]),
            "h_s": float(plan["h_s"]),
            "dt_s": float(plan["dt_s"]),
            "start_time": float(context.p.start_time),
            "nsteps": int(plan["nsteps"]),
            "horizon_s": float(plan["horizon_s"]),
            "waveform_rtol": float(context.p.waveform_rtol),
            "waveform_max_iterations": int(
                context.p.waveform_max_iterations
            ),
            "target_frequency_hz": target,
        },
        "precision": {
            "real": str(context.compute.xp.dtype("float64")),
            "complex": str(context.compute.xp.dtype("complex128")),
        },
        "physical_identity": cae.physical_identity(context),
        "perturbation": {
            "mode": int(controls["mode"]),
            "amplitude": float(controls["amplitude"]),
            "initial_perturbed_sha256": str(seed_sha256),
        },
        "geometry_arrays": _object_array_signature(
            context.geometry, context
        ),
        "operator_arrays": _object_array_signature(context.op, context),
        "source_hashes": _source_hashes(),
        "runtime_fingerprint": _runtime_fingerprint(),
    }


def _source_snapshot(output: Path) -> dict[str, Any]:
    identity = source_identity()
    package = Path(identity["fomhd_package"]).parent
    destination = output / "provenance" / "source" / "src" / "fomhd"
    destination.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for path in sorted(package.glob("*.py")):
        target = destination / path.name
        shutil.copy2(path, target)
        relative = str(target.relative_to(output))
        if _sha256_file(path) != _sha256_file(target):
            raise RuntimeError(f"source snapshot mismatch: {relative}")
        hashes[relative] = _sha256_file(path)
    harness_target = output / "provenance" / "tools" / Path(__file__).name
    harness_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), harness_target)
    hashes[str(harness_target.relative_to(output))] = _sha256_file(Path(__file__))
    pseudocode = (
        SCRATCH_ROOT / "docs" / "pseudocode" / "tools"
        / (Path(__file__).name + ".md")
    )
    if pseudocode.is_file():
        target = output / "provenance" / "docs" / "pseudocode" / "tools" / pseudocode.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pseudocode, target)
        hashes[str(target.relative_to(output))] = _sha256_file(pseudocode)
    return {
        **identity,
        "git_head": _git_head(Path(identity["source_root"])),
        "files": hashes,
        "archive_source_hashes": _source_hashes(),
        "runtime_fingerprint": _runtime_fingerprint(),
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "nbytes": int(value.nbytes),
        }
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable(child)
            for key, child in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(value), allow_nan=False) + "\n")
        stream.flush()


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}\n"
        )
        stream.flush()


@contextmanager
def _explicit_massless_operator_options():
    option_names = {field.name for field in fields(OperatorOptions)}
    supported = "include_finite_mass" in option_names
    original = cae.build_operators

    def configured(geometry, K, options=None, backend=None):
        settings = dict(options or {})
        if supported:
            settings["include_finite_mass"] = False
        return original(geometry, K, settings, backend)

    cae.build_operators = configured
    try:
        yield {
            "name": "include_finite_mass",
            "requested_value": False,
            "applied": supported,
            "source_capability": supported,
            "note": (
                "passed directly to operator construction"
                if supported
                else "selected source lacks OperatorOptions.include_finite_mass"
            ),
        }
    finally:
        cae.build_operators = original


def _make_context(
    kind: str,
    nodes: int,
    K: int,
    device: str,
    equilibrium: Path | None,
    memory_limit_bytes: int,
):
    cae_device = "cuda" if device == "cuda:0" else "cpu"
    with _explicit_massless_operator_options() as option_record:
        context, meta, eq = cae.make_context(
            kind,
            nodes,
            K,
            cae_device,
            equilibrium,
            memory_limit_bytes=memory_limit_bytes,
        )
    context.p.memory_limit = HOST_MEMORY_CAP
    context.p.device_memory_limit = DEVICE_MEMORY_CAP
    context._warp_050_operator_option = option_record
    return context, meta, eq


def _clone_state(source: Any, context: Any) -> Any:
    value = core.allocate_state(context)
    value.B = source.B.copy()
    for species in range(2):
        for name in ("n", "mom", "perp", "parallel"):
            setattr(
                value.fluid[species],
                name,
                getattr(source.fluid[species], name).copy(),
            )
    value.step = int(source.step)
    value.generation = int(source.generation)
    value.time = float(source.time)
    return value


def _archive_paths(directory: Path) -> tuple[Path, Path]:
    directory = Path(directory)
    if directory.is_dir():
        return directory / "spectrum.npz", directory / "report.json"
    if directory.name == "spectrum.npz":
        return directory, directory.with_name("report.json")
    if directory.name == "report.json":
        return directory.with_name("spectrum.npz"), directory
    raise ValueError("spectrum directory must contain report.json and spectrum.npz")


def _seed_selector_identity(
    directory: Path | None,
    *,
    archive: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if directory is None and archive is None:
        return {"kind": "smooth"}
    if archive is None:
        spectrum_path, report_path = _archive_paths(Path(directory))
        spectrum_path = spectrum_path.resolve()
        report_path = report_path.resolve()
        if not spectrum_path.is_file() or not report_path.is_file():
            raise ValueError(
                "spectrum archive is missing report.json or spectrum.npz"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        archive = {
            "spectrum_path": str(spectrum_path),
            "report_path": str(report_path),
            "spectrum_sha256": _sha256_file(spectrum_path),
            "report_sha256": _sha256_file(report_path),
            "report_source_hashes": report.get("source_hashes"),
            "selected": report.get("selected"),
        }
    spectrum_path = Path(archive["spectrum_path"]).resolve()
    report_path = Path(archive["report_path"]).resolve()
    source_hashes = archive["report_source_hashes"]
    selected = archive["selected"]
    if not isinstance(source_hashes, dict) or not isinstance(selected, dict):
        raise ValueError("spectrum archive selector identity is invalid")
    return {
        "kind": "spectrum_archive",
        "directory": str(spectrum_path.parent),
        "spectrum_path": str(spectrum_path),
        "report_path": str(report_path),
        "spectrum_sha256": str(archive["spectrum_sha256"]),
        "report_sha256": str(archive["report_sha256"]),
        "source_hashes": _jsonable(source_hashes),
        "selected": _jsonable(selected),
    }


def _max_abs_discrepancy(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(
            f"spectrum background shape mismatch: {actual.shape} != {expected.shape}"
        )
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise ValueError("spectrum background identity contains nonfinite values")
    discrepancy = float(np.max(np.abs(actual - expected), initial=0.0))
    if not math.isfinite(discrepancy):
        raise ValueError("spectrum background identity discrepancy is nonfinite")
    return discrepancy


def read_archive_seed(
    directory: Path,
    context: Any,
    initial: Any,
    *,
    kind: str,
    nodes: int,
    K: int,
    mode: int,
    target_frequency_hz: float | None,
) -> dict[str, Any]:
    """Read only archived seed/background arrays; never touch eigenpairs."""
    spectrum_path, report_path = _archive_paths(Path(directory))
    if not spectrum_path.is_file() or not report_path.is_file():
        raise ValueError("spectrum archive is missing report.json or spectrum.npz")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for name, expected in (
        ("kind", kind),
        ("nodes", nodes),
        ("K", K),
        ("mode", mode),
    ):
        if report.get(name) != expected:
            raise ValueError(
                f"spectrum {name} mismatch: archive={report.get(name)!r} "
                f"requested={expected!r}"
            )
    selected = report.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("spectrum report has no selected branch")
    selected_frequency = _finite_positive(
        selected.get("frequency_hz"), "selected.frequency_hz"
    )
    if target_frequency_hz is not None and not math.isclose(
        selected_frequency,
        float(target_frequency_hz),
        rel_tol=1.0e-10,
        abs_tol=1.0e-9,
    ):
        raise ValueError(
            "target frequency does not identify the archived selected branch"
        )
    current_identity = cae.physical_identity(context)
    if report.get("physical_identity") != current_identity:
        raise ValueError("spectrum physical identity mismatch")

    with np.load(spectrum_path, allow_pickle=False) as archive:
        missing = [
            name
            for name in SEED_NAMES + BACKGROUND_NAMES
            if name not in archive.files
        ]
        if missing:
            raise ValueError(
                "spectrum archive is missing arrays: " + ", ".join(missing)
            )
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in SEED_NAMES + BACKGROUND_NAMES
        }
        files = set(archive.files)
    fields = {
        "n": arrays["seed_n"],
        "mom": arrays["seed_mom"],
        "pi": arrays["seed_pi"],
        "perp": arrays["seed_perp"],
        "parallel": arrays["seed_parallel"],
        "B": arrays["seed_B"],
    }
    expected_shapes = {
        "n": (context.op.Nc,),
        "mom": (3 * context.op.Nc,),
        "pi": (context.op.Nc,),
        "perp": (context.op.Nc,),
        "parallel": (context.op.Nc,),
        "B": (context.op.Nb,),
    }
    for name, value in fields.items():
        if value.shape != expected_shapes[name]:
            raise ValueError(
                f"seed_{name} shape mismatch: {value.shape} != "
                f"{expected_shapes[name]}"
            )
        if not np.issubdtype(value.dtype, np.complexfloating):
            raise ValueError(f"seed_{name} must be complex")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"seed_{name} contains nonfinite values")

    current = {
        "B": cae.host(initial.B, context),
        "n": cae.host(initial.fluid[0].n, context),
        "mom": cae.host(initial.fluid[0].mom, context),
        "pi": cae.host(initial.fluid[0].perp, context),
        "perp": cae.host(initial.fluid[1].perp, context),
        "parallel": cae.host(initial.fluid[1].parallel, context),
    }
    scales = {
        "B": context.p.ref.magnetic,
        "n": context.p.ref.ni,
        "mom": (
            context.p.species[0].mass
            * context.p.ref.ni
            * context.p.ref.velocity
        ),
        "pi": context.p.ref.pressure,
        "perp": context.p.ref.pressure,
        "parallel": context.p.ref.pressure,
    }
    discrepancies = {}
    for name, scale in scales.items():
        discrepancy = _max_abs_discrepancy(
            arrays[f"background_{name}"], current[name]
        )
        discrepancies[name] = {
            "maximum_absolute": discrepancy,
            "physical_scale": float(scale),
            "scaled": discrepancy / float(scale),
            "threshold": 1.0e-10,
        }
        if discrepancy > 1.0e-10 * float(scale):
            raise ValueError(
                f"spectrum background field mismatch: {name} "
                f"max_abs={discrepancy:.17g} scale={scale:.17g}"
            )

    actual_hashes = _source_hashes()
    report_hashes = report.get("source_hashes")
    if not isinstance(report_hashes, dict):
        raise ValueError("spectrum report source_hashes must be an object")
    mismatch = {
        name: {
            "archive": report_hashes.get(name),
            "actual": actual_hashes.get(name),
        }
        for name in sorted(set(report_hashes) | set(actual_hashes))
        if report_hashes.get(name) != actual_hashes.get(name)
    }
    return {
        "directory": str(Path(directory).resolve()),
        "spectrum_path": str(spectrum_path.resolve()),
        "report_path": str(report_path.resolve()),
        "spectrum_bytes": spectrum_path.stat().st_size,
        "report_bytes": report_path.stat().st_size,
        "spectrum_sha256": _sha256_file(spectrum_path),
        "report_sha256": _sha256_file(report_path),
        "selected": _jsonable(selected),
        "fields": fields,
        "background_discrepancies": discrepancies,
        "report_source_hashes": report_hashes,
        "actual_source_hashes": actual_hashes,
        "source_hash_match": not mismatch,
        "source_hash_mismatches": mismatch,
        "eigenpair_arrays_present": sorted(
            name for name in EIGENPAIR_NAMES if name in files
        ),
        "eigenpair_arrays_read": [],
        "radius_scope": (
            "the archive supplies one selected n=1 branch/radius; it is not "
            "a full K=1 spectral-radius bound"
        ),
    }


def _baseline_arrays(initial: Any, context: Any) -> dict[str, np.ndarray]:
    return {
        "B": cae.host(initial.B, context),
        "n": cae.host(initial.fluid[0].n, context),
        "mom": cae.host(initial.fluid[0].mom, context),
        "pi": cae.host(initial.fluid[0].perp, context),
        "perp": cae.host(initial.fluid[1].perp, context),
        "parallel": cae.host(initial.fluid[1].parallel, context),
    }


def _baseline_scales(context: Any) -> dict[str, float]:
    return {
        "B": float(context.p.ref.magnetic),
        "n": float(context.p.ref.ni),
        "mom": float(
            context.p.species[0].mass
            * context.p.ref.ni
            * context.p.ref.velocity
        ),
        "pi": float(context.p.ref.pressure),
        "perp": float(context.p.ref.pressure),
        "parallel": float(context.p.ref.pressure),
    }


def read_stability_spectra(
    directories: list[Path],
    context: Any,
    initial: Any,
    *,
    kind: str,
    nodes: int,
    K: int,
) -> dict[str, Any]:
    """Validate one complete retained-sector spectrum without loading eigenpairs."""
    if not directories:
        raise ValueError("at least one stability spectrum directory is required")
    current_identity = cae.physical_identity(context)
    baseline = _baseline_arrays(initial, context)
    scales = _baseline_scales(context)
    required = set(range(int(K) + 1))
    seen_modes: set[int] = set()
    entries: list[dict[str, Any]] = []
    files_read: set[str] = set()
    for directory in directories:
        spectrum_path, report_path = _archive_paths(Path(directory))
        if not spectrum_path.is_file() or not report_path.is_file():
            raise ValueError(
                "stability spectrum is missing report.json or spectrum.npz"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for name, expected in (
            ("kind", kind),
            ("nodes", int(nodes)),
            ("K", int(K)),
        ):
            if report.get(name) != expected:
                raise ValueError(
                    f"stability spectrum {name} mismatch: "
                    f"archive={report.get(name)!r} requested={expected!r}"
                )
        mode = report.get("mode")
        if isinstance(mode, bool) or type(mode) is not int or mode not in required:
            raise ValueError(
                f"stability spectrum mode must be one of {sorted(required)}"
            )
        if mode in seen_modes:
            raise ValueError(f"duplicate stability spectrum sector: {mode}")
        if report.get("physical_identity") != current_identity:
            raise ValueError("stability spectrum physical identity mismatch")
        reported_radius = report.get("spectral_radius_rad_s")
        reported_growth = report.get("maximum_growth_per_s")
        if type(reported_radius) is not float or not math.isfinite(reported_radius):
            raise ValueError(
                "stability report spectral_radius_rad_s must be finite"
            )
        if type(reported_growth) is not float or not math.isfinite(reported_growth):
            raise ValueError(
                "stability report maximum_growth_per_s must be finite"
            )

        with np.load(spectrum_path, allow_pickle=False) as archive:
            names = set(archive.files)
            required_arrays = {"eigenvalues"} | {
                f"background_{name}" for name in baseline
            }
            missing = sorted(required_arrays - names)
            if missing:
                raise ValueError(
                    "stability spectrum is missing arrays: " + ", ".join(missing)
                )
            eigenvalues = np.array(archive["eigenvalues"], copy=True)
            files_read.add("eigenvalues")
            discrepancies = {}
            for name, expected in baseline.items():
                array_name = f"background_{name}"
                saved = np.array(archive[array_name], copy=True)
                files_read.add(array_name)
                discrepancy = _max_abs_discrepancy(saved, expected)
                scale = scales[name]
                discrepancies[name] = {
                    "maximum_absolute": discrepancy,
                    "physical_scale": scale,
                    "scaled": discrepancy / scale,
                    "threshold": 1.0e-10,
                }
                if discrepancy > 1.0e-10 * scale:
                    raise ValueError(
                        f"stability spectrum baseline mismatch: {name} "
                        f"max_abs={discrepancy:.17g} scale={scale:.17g}"
                    )

        if eigenvalues.dtype != np.dtype(np.complex128):
            raise ValueError("stability eigenvalues must have complex128 dtype")
        if eigenvalues.ndim != 1 or eigenvalues.size == 0:
            raise ValueError("stability eigenvalues must be a nonempty vector")
        if not np.all(np.isfinite(eigenvalues)):
            raise ValueError("stability eigenvalues contain nonfinite values")
        if report.get("size") != int(eigenvalues.size):
            raise ValueError(
                "stability spectrum size mismatch: "
                f"report={report.get('size')!r} array={eigenvalues.size}"
            )
        radius = float(np.max(np.abs(eigenvalues)))
        maximum_growth = float(np.max(eigenvalues.real))
        if reported_radius != radius:
            raise ValueError(
                "stability spectral radius mismatch: "
                f"report={reported_radius!r} eigenvalues={radius!r}"
            )
        if reported_growth != maximum_growth:
            raise ValueError(
                "stability maximum growth mismatch: "
                f"report={reported_growth!r} eigenvalues={maximum_growth!r}"
            )

        actual_hashes = _source_hashes()
        report_hashes = report.get("source_hashes")
        if not isinstance(report_hashes, dict):
            raise ValueError("stability report source_hashes must be an object")
        source_mismatch = {
            name: {
                "archive": report_hashes.get(name),
                "actual": actual_hashes.get(name),
            }
            for name in sorted(set(report_hashes) | set(actual_hashes))
            if report_hashes.get(name) != actual_hashes.get(name)
        }
        background_metadata = report.get("background_arrays", {})
        if background_metadata and not isinstance(background_metadata, dict):
            raise ValueError("stability background_arrays must be an object")
        for name in baseline:
            declared = background_metadata.get(f"background_{name}")
            if declared is None:
                continue
            if not isinstance(declared, dict):
                raise ValueError(
                    f"stability background_arrays.background_{name} "
                    "must be an object"
                )
            expected_shape = list(baseline[name].shape)
            if declared.get("shape") is not None and declared["shape"] != expected_shape:
                raise ValueError(
                    f"stability background metadata shape mismatch: {name}"
                )
            if (
                declared.get("dtype") is not None
                and declared["dtype"] != str(baseline[name].dtype)
            ):
                raise ValueError(
                    f"stability background metadata dtype mismatch: {name}"
                )

        seen_modes.add(mode)
        entries.append(
            {
                "directory": str(Path(directory).resolve()),
                "spectrum_path": str(spectrum_path.resolve()),
                "report_path": str(report_path.resolve()),
                "spectrum_sha256": _sha256_file(spectrum_path),
                "report_sha256": _sha256_file(report_path),
                "mode": mode,
                "size": int(eigenvalues.size),
                "eigenvalues_dtype": str(eigenvalues.dtype),
                "spectral_radius_rad_s": radius,
                "maximum_growth_per_s": maximum_growth,
                "background_discrepancies": discrepancies,
                "source_hash_match": not source_mismatch,
                "source_hash_mismatches": source_mismatch,
                "large_eigenpair_arrays_present": sorted(
                    {"jacobian", "eigenvectors"} & names
                ),
                "eigenvalue_vector_read": True,
                "eigenpair_arrays_read": [],
            }
        )
    missing_modes = sorted(required - seen_modes)
    if missing_modes:
        raise ValueError(
            "stability spectra missing retained sectors: "
            + ", ".join(str(mode) for mode in missing_modes)
        )
    rho_max = max(item["spectral_radius_rad_s"] for item in entries)
    return {
        "spectra": sorted(entries, key=lambda item: item["mode"]),
        "complete_sectors": sorted(seen_modes),
        "rho_max_rad_s": rho_max,
        "h_cap_s": 0.5 / (1.5 * rho_max),
        "eigenpair_arrays_read": [],
        "files_read": sorted(files_read),
        "source_hash_mismatch_recorded": any(
            not item["source_hash_match"] for item in entries
        ),
        "scope": (
            "local discrete linear spectral radius and neighborhood guard; "
            "not a nonlinear stability proof"
        ),
    }


def _probe_estimates(states: dict[str, Any], context: Any) -> dict[str, Any]:
    values = {}
    for label, state in states.items():
        result = stage.evaluate_stage(
            state,
            StageTag(
                step=int(state.step),
                generation=int(state.generation),
                trial=0,
                spectral_context=0,
                time=float(state.time),
                stage=f"050_{label}_frequency_probe",
            ),
            context,
            False,
        )
        values[label] = stability.stage_frequency_estimate(result, context)
    maximum = max(values.values(), key=lambda item: item["omega_estimate"])
    return {
        "by_ensemble": values,
        "omega_estimate": float(maximum["omega_estimate"]),
        "scope": (
            "operator-resolution estimate; not a rigorous nonnormal "
            "full-spectrum or nonlinear bound"
        ),
    }


def _resolve_time_plan(
    controls: dict[str, Any],
    context: Any,
    states: dict[str, Any],
    *,
    stability_radius_rad_s: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    requested = controls["h_requested"]
    probes = _probe_estimates(states, context)
    if requested != "auto":
        plan = plan_time(
            controls["horizon_s"],
            float(requested),
            controls["micro_steps"],
            controls["coupling"],
        )
        provisional = float(requested)
    else:
        omega = probes["omega_estimate"]
        if not math.isfinite(omega) or omega <= 0.0:
            raise ValueError("automatic h requires a positive frequency estimate")
        if stability_radius_rad_s is None:
            provisional = 0.5 / (1.5 * omega)
        else:
            provisional = 0.5 / (1.5 * float(stability_radius_rad_s))
        target = controls["target_frequency_hz"]
        if target is not None:
            provisional = min(provisional, 1.0 / (64.0 * target))
        if controls["coupling"] == "synchronous":
            planned = max(1, int(math.ceil(controls["horizon_s"] / provisional)))
            h = controls["horizon_s"] / planned
        else:
            planned = max(
                1,
                int(
                    math.ceil(
                        controls["horizon_s"]
                        / (controls["micro_steps"] * provisional)
                    )
                ),
            )
            h = controls["horizon_s"] / (controls["micro_steps"] * planned)
        plan = plan_time(
            controls["horizon_s"],
            h,
            controls["micro_steps"],
            controls["coupling"],
        )
        requested = "auto"
    plan["h_provisional_s"] = provisional
    plan["h_policy"] = (
        "parent-selected validated micro h"
        if requested != "auto"
        else (
            (
                "micro h <= 0.5/(1.5*rho_max), with target period/64 "
                "when a target is supplied"
            )
            if stability_radius_rad_s is not None
            else (
                "micro h <= min(0.5/(1.5*max initial stage omega estimate), "
                "target period/64 when target supplied)"
            )
        )
    )
    plan["stability_radius_rad_s"] = stability_radius_rad_s
    plan["stability_cap_s"] = (
        None
        if stability_radius_rad_s is None
        else 0.5 / (1.5 * float(stability_radius_rad_s))
    )
    if (
        plan["stability_cap_s"] is not None
        and plan["h_s"] > plan["stability_cap_s"]
    ):
        raise ValueError(
            "resolved stability h cap exceeded: "
            f"h={plan['h_s']:.17g} cap={plan['stability_cap_s']:.17g}"
        )
    plan["operator_estimate_used_for_automatic_h"] = bool(
        requested == "auto" and stability_radius_rad_s is None
    )
    return plan, probes


def _clone_experiment_start(experiment: dict[str, Any], context: Any):
    committed = SimpleNamespace(
        state=experiment["state"],
        cache=experiment["frame"].cache,
        accepted=True,
    )
    return committed, copy.deepcopy(experiment["ledger"])


def _check_counts(records: list[Any], state: dict[str, int]) -> None:
    for record in records:
        if not getattr(record, "available", True):
            state["unavailable"] += 1
        elif getattr(record, "passed", False):
            state["pass"] += 1
        elif getattr(record, "hard", True):
            state["hard_fail"] += 1
        else:
            state["observe_fail"] += 1


def _check_record(record: Any) -> dict[str, Any]:
    return _jsonable(io._check_json(record))


def _append_checks(path: Path, records: list[Any], counts: dict[str, int]) -> None:
    for record in records:
        _append_jsonl(path, _check_record(record))
    _check_counts(records, counts)


def _scalar_history(experiment: dict[str, Any], envelope: dict[str, Any]) -> dict[str, Any]:
    frame = experiment["frame"]
    ledger = experiment["ledger"]
    energy = summarize_energy(
        frame.totals,
        0,
        baseline_total_J=experiment["baseline_energy"],
        outward_J=ledger.boundary_integrals.total_energy,
        source_J=0.0,
    )
    return {
        "time_s": float(experiment["state"].time),
        "step": int(experiment["state"].step),
        "generation": int(experiment["state"].generation),
        "energy": energy,
        "momentum": _jsonable(frame.momentum.value),
        "accepted_intervals": int(ledger.accepted_intervals),
        "last_trial": int(ledger.last_trial),
        "runtime_envelope": envelope,
    }


def _ledger_json(ledger: Any) -> dict[str, Any]:
    return _jsonable(io._cumulative_json(ledger))


def _state_sha256(state: Any, context: Any) -> str:
    return _collection_sha256(io.encode_state(state, context.geometry))


def _guard_baseline(frame: Any, context: Any) -> dict[str, float]:
    xp = context.compute.xp
    cells = int(context.op.Nc)
    magnetic = float(
        context.compute.scalar(
            xp.max(
                xp.sqrt(
                    xp.sum(
                        xp.abs(frame.BN.reshape(context.N, 3, cells)) ** 2,
                        axis=1,
                    )
                )
            )
        )
    )
    density = float(
        context.compute.scalar(xp.min(xp.real(frame.fluid[0].n)))
    )
    if (
        not math.isfinite(magnetic)
        or magnetic <= 0.0
        or not math.isfinite(density)
        or density <= 0.0
    ):
        raise ValueError("guard baseline requires finite positive B and density")
    return {
        "magnetic_max_T": magnetic,
        "density_min": density,
    }


def _write_state_npz(
    path: Path,
    experiment: dict[str, Any],
    context: Any,
    kind: str,
    *,
    run_id: str,
    compatibility: dict[str, Any],
    initial_provenance: dict[str, Any],
    parent_checkpoints: list[dict[str, str]],
    guard_baseline: dict[str, float],
    stability: dict[str, Any] | None,
) -> str:
    arrays = io.encode_state(experiment["state"], context.geometry)
    metadata = {
        "schema": "fomhd-warp-050-accepted-state-v2",
        "kind": kind,
        "accepted": True,
        "step": int(experiment["state"].step),
        "generation": int(experiment["state"].generation),
        "time_s": float(experiment["state"].time),
        "dt_s": float(context.p.dt),
        "micro_h_s": float(
            context.p.dt / context.p.micro_steps
            if context.p.coupling == "waveform"
            else context.p.dt
        ),
        "coupling": context.p.coupling,
        "micro_steps": int(context.p.micro_steps),
        "nsteps": int(context.p.nsteps),
        "start_time_s": float(context.p.start_time),
        "resolved_horizon_s": float(context.p.dt * context.p.nsteps),
        "waveform_rtol": float(context.p.waveform_rtol),
        "waveform_max_iterations": int(context.p.waveform_max_iterations),
        "time_plan": {
            "coupling": context.p.coupling,
            "h_s": float(
                context.p.dt / context.p.micro_steps
                if context.p.coupling == "waveform"
                else context.p.dt
            ),
            "micro_steps": int(context.p.micro_steps),
            "dt_s": float(context.p.dt),
            "nsteps": int(context.p.nsteps),
            "horizon_s": float(context.p.dt * context.p.nsteps),
        },
        "run_id": run_id,
        "physical_identity": cae.physical_identity(context),
        "source_hashes": _source_hashes(),
        "runtime_fingerprint": _runtime_fingerprint(),
        "source_identity": source_identity(),
        "cumulative_json": _ledger_json(experiment["ledger"]),
        "compatibility": compatibility,
        "initial_provenance": initial_provenance,
        "parent_checkpoints": parent_checkpoints,
        "guard_baseline": guard_baseline,
        "stability": stability,
        "scheduler": {
            "state_generation": int(experiment["state"].generation),
            "last_trial": int(getattr(context, "next_trial", 1)) - 1,
            "last_source": int(getattr(context, "next_source", 1)) - 1,
            "next_trial": int(getattr(context, "next_trial", 1)),
            "next_source": int(getattr(context, "next_source", 1)),
            "next_generation": int(getattr(context, "next_generation", 1)),
        },
        "production_restart": False,
        "restart_note": (
            "complete accepted state, ledger, scheduler counters and provenance; "
            "this explicit scratch NPZ remains an experimental continuation "
            "format, not a production restart fixture"
        ),
    }
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, allow_nan=False) + "\n"
    )
    arrays["cumulative_json"] = np.asarray(
        json.dumps(metadata["cumulative_json"], allow_nan=False)
    )
    arrays["baseline_divB"] = io.encode_vertex(
        experiment["ledger"].baseline.divB,
        context.geometry,
        context.p.K,
    )
    arrays["last_source_id"] = np.asarray(
        metadata["scheduler"]["last_source"], dtype=np.uint64
    )
    arrays["last_trial_id"] = np.asarray(
        metadata["scheduler"]["last_trial"], dtype=np.uint64
    )
    arrays["state_step"] = np.asarray(
        experiment["state"].step, dtype=np.uint64
    )
    arrays["state_generation"] = np.asarray(
        experiment["state"].generation, dtype=np.uint64
    )
    arrays["state_time"] = np.asarray(
        experiment["state"].time, dtype=np.float64
    )
    arrays["accepted"] = np.asarray(True, dtype=np.bool_)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return _sha256_file(path)


def _write_snapshot(
    path: Path,
    experiment: dict[str, Any],
    context: Any,
    envelope: dict[str, Any],
) -> str:
    arrays = io.encode_state(experiment["state"], context.geometry)
    observed = cae.observation(
        experiment["state"],
        experiment["frame"],
        experiment["ledger"],
        context,
        experiment["baseline_energy"],
    )
    arrays.update(
        {
            "step": np.asarray(experiment["state"].step, dtype=np.uint64),
            "generation": np.asarray(experiment["state"].generation, dtype=np.uint64),
            "time_s": np.asarray(experiment["state"].time, dtype=np.float64),
            "ledger_json": np.asarray(
                json.dumps(_ledger_json(experiment["ledger"]), allow_nan=False)
            ),
            "runtime_envelope_json": np.asarray(
                json.dumps(_jsonable(envelope), allow_nan=False)
            ),
            "observed_BN": np.asarray(observed["BN"]),
            "observed_density": np.asarray(observed["density"]),
            "observed_velocity": np.asarray(observed["velocity"]),
            "observation_json": np.asarray(
                json.dumps(
                    _jsonable(
                        {
                            key: observed[key]
                            for key in (
                                "time_s",
                                "step",
                                "energy",
                                "energy_raw",
                                "momentum",
                            )
                        }
                    ),
                    allow_nan=False,
                )
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return _sha256_file(path)


def _write_failure(
    path: Path,
    error: BaseException | dict[str, Any],
    context: Any,
    *,
    label: str,
    step: int,
) -> list[str]:
    if isinstance(error, dict):
        error_info = error
        partial = {}
    else:
        error_info = {
            "code": getattr(error, "code", type(error).__name__),
            "message": str(error),
            "traceback": getattr(
                error,
                "_warp050_traceback",
                __import__("traceback").format_exc(),
            ),
            "stage": _jsonable(getattr(error, "stage", None)),
        }
        partial = getattr(error, "partial_arrays", {}) or {}
    arrays = {
        f"failure__{name}": np.asarray(cae.host(value, context))
        for name, value in partial.items()
    }
    arrays["failure_json"] = np.asarray(
        json.dumps(_jsonable(error_info), allow_nan=False)
    )
    arrays["label"] = np.asarray(label)
    arrays["attempted_after_step"] = np.asarray(step, dtype=np.uint64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return sorted(arrays)


def _rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _ru_maxrss_bytes() -> int | None:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    return value if value > 0 else None


def _sampled_peak(samples: list[dict[str, Any]], name: str) -> int | None:
    values = [
        int(sample[name])
        for sample in samples
        if sample.get(name) is not None
    ]
    return max(values) if values else None


def _memory_sample(context: Any, elapsed_s: float) -> dict[str, Any]:
    value = {
        "elapsed_s": elapsed_s,
        "rss_bytes": _rss_bytes(),
        "ru_maxrss_bytes": _ru_maxrss_bytes(),
        "backend": context.compute.name,
    }
    if context.compute.name == "cuda":
        try:
            import cupy

            pool = cupy.get_default_memory_pool()
            value.update(
                {
                    "device_pool_used_bytes": int(pool.used_bytes()),
                    "device_pool_total_bytes": int(pool.total_bytes()),
                }
            )
            pinned = cupy.get_default_pinned_memory_pool()
            value["pinned_pool_free_blocks"] = int(pinned.n_free_blocks())
            free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
            value["device_wide_used_bytes"] = int(total_bytes - free_bytes)
            value["device_wide_total_bytes"] = int(total_bytes)
        except Exception as error:
            value["device_allocator_error"] = str(error)
    return value


def _array_inventory(context: Any, experiments: dict[str, dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    cupy_module = sys.modules.get("cupy")
    cupy_array = getattr(cupy_module, "ndarray", None)

    def visit(value: Any, path: str, depth: int = 0) -> None:
        if depth > 32:
            return
        if isinstance(value, np.ndarray):
            placement = "host"
        elif cupy_array is not None and isinstance(value, cupy_array):
            placement = "device"
        else:
            placement = None
        if placement is not None:
            identity = (placement, id(value))
            if identity in seen:
                return
            seen.add(identity)
            records.append(
                {
                    "path": path,
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "bytes": int(value.nbytes),
                    "placement": placement,
                }
            )
            return
        if value is None or isinstance(value, (str, bytes, int, float, complex, bool)):
            return
        identity = ("object", id(value))
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}", depth + 1)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]", depth + 1)
        elif hasattr(value, "__dict__"):
            for key, child in vars(value).items():
                if key not in ("compute", "backend"):
                    visit(child, f"{path}.{key}", depth + 1)

    visit(context.geometry, "context.geometry")
    visit(context.op, "context.operators")
    for label, experiment in experiments.items():
        visit(experiment["state"], f"experiments.{label}.state")
        visit(experiment["frame"], f"experiments.{label}.frame")
        visit(experiment["ledger"], f"experiments.{label}.ledger")
    host = sum(item["bytes"] for item in records if item["placement"] == "host")
    device = sum(item["bytes"] for item in records if item["placement"] == "device")
    return {
        "array_count": len(records),
        "host_resident_bytes": host,
        "device_resident_bytes": device,
        "resident_bytes": host + device,
        "deduplication": (
            "placement and object identity within retained "
            "context/state/frame roots"
        ),
        "arrays": records,
    }


def _target_accuracy(
    target_frequency_hz: float | None,
    h_s: float,
    total_micro_steps: int,
) -> dict[str, Any]:
    if target_frequency_hz is None:
        return {
            "available": False,
            "reason": "no target frequency supplied or archived",
        }
    report = stability.assess_dt(
        "rk4",
        2.0 * math.pi * float(target_frequency_hz),
        float(h_s),
        int(total_micro_steps),
    )
    report["target_frequency_hz"] = float(target_frequency_hz)
    report["scope"] = (
        "target-frequency scalar RK4 accuracy over the full horizon; "
        "separate from grid-high mode stability"
    )
    return report


def _require_target_accuracy(report: dict[str, Any]) -> None:
    if report.get("available") is False:
        return
    if report.get("accepted") is not True:
        reasons = ", ".join(str(reason) for reason in report.get("reasons", []))
        raise ValueError(
            "target-frequency accuracy gate failed"
            + (f": {reasons}" if reasons else "")
        )


def _envelope(experiment: dict[str, Any], context: Any) -> dict[str, Any]:
    estimate = stability.stage_frequency_estimate(experiment["frame"], context)
    h = (
        context.p.dt / context.p.micro_steps
        if context.p.coupling == "waveform"
        else context.p.dt
    )
    value = float(estimate["omega_estimate"] * h)
    return {
        "omega_current_rad_s": float(estimate["omega_estimate"]),
        "h_s": float(h),
        "omega_h": value,
        "limit": 0.5,
        "accepted": bool(math.isfinite(value) and value <= 0.5),
        "scope": estimate["scope"],
    }


def _neighborhood_guard(
    frame: Any,
    context: Any,
    *,
    baseline_magnetic_max_T: float,
    baseline_density_min: float,
    baseline_rate_estimate: dict[str, Any],
    rho_max_rad_s: float,
    micro_h_s: float,
) -> dict[str, Any]:
    """Evaluate the original resolved-spectrum local-neighborhood guard."""
    xp = context.compute.xp
    cells = int(context.op.Nc)
    bmax = float(
        context.compute.scalar(
            xp.max(
                xp.sqrt(
                    xp.sum(
                        xp.abs(
                            frame.BN.reshape(context.N, 3, cells)
                        )
                        ** 2,
                        axis=1,
                    )
                )
            )
        )
    )
    nmin = float(context.compute.scalar(xp.min(xp.real(frame.fluid[0].n))))
    baseline_b = _finite_positive(
        baseline_magnetic_max_T, "baseline_magnetic_max_T"
    )
    baseline_n = _finite_positive(
        baseline_density_min, "baseline_density_min"
    )
    rho = _finite_positive(rho_max_rad_s, "rho_max_rad_s")
    h = _finite_positive(micro_h_s, "micro_h_s")
    if not math.isfinite(bmax) or not math.isfinite(nmin) or nmin <= 0.0:
        raise ValueError("runtime neighborhood guard requires finite positive fields")
    current_rate_estimate = stability.stage_frequency_estimate(frame, context)
    observed_rates = {}
    for name, field in (
        ("ion_pressure_per_density", "ion_pressure_rate"),
        ("electron_pressure_per_density", "electron_pressure_rate"),
        ("bulk_pressure_per_density", "bulk_pressure_rate"),
        ("advection", "advection_rate"),
    ):
        initial = float(baseline_rate_estimate[field])
        current = float(current_rate_estimate[field])
        if (
            not math.isfinite(initial)
            or initial < 0.0
            or not math.isfinite(current)
            or current < 0.0
        ):
            raise ValueError(
                f"runtime neighborhood rate observation is invalid: {field}"
            )
        observed_rates[name] = {
            "initial_rad_s": initial,
            "current_rad_s": current,
            "relative_to_initial": (
                current / initial if initial > 0.0 else None
            ),
            "difference_rad_s": current - initial,
        }
    magnetic_ratio = max(bmax / baseline_b, 1.0)
    density_ratio = max(baseline_n / nmin, 1.0)
    ratio = magnetic_ratio * density_ratio
    rho_ratio_h = rho * ratio * h
    reasons = []
    if ratio > 1.5:
        reasons.append("nonlinear-neighborhood ratio exceeds 1.5")
    if rho_ratio_h > 0.5:
        reasons.append("rho_max*ratio*micro_h exceeds 0.5")
    return {
        "baseline_magnetic_max_T": baseline_b,
        "current_magnetic_max_T": bmax,
        "magnetic_ratio": magnetic_ratio,
        "baseline_density_min": baseline_n,
        "current_density_min": nmin,
        "density_ratio": density_ratio,
        "ratio": ratio,
        "ratio_limit": 1.5,
        "rho_max_rad_s": rho,
        "micro_h_s": h,
        "rho_ratio_h": rho_ratio_h,
        "rho_ratio_h_limit": 0.5,
        "observed_rates_relative_to_initial": observed_rates,
        "rate_observation_scope": (
            "stage_frequency_estimate observations only; no acceptance threshold"
        ),
        "accepted": not reasons,
        "reasons": reasons,
        "scope": (
            "local linear spectral radius with a runtime neighborhood ratio; "
            "not a nonlinear stability proof"
        ),
    }


def _initial_rhs_record(frame: Any, context: Any) -> dict[str, Any]:
    values = {"B": frame.rhs.B}
    for species, label in enumerate(("i", "e")):
        for name in ("n", "mom", "perp", "parallel"):
            values[f"{name}_{label}"] = getattr(frame.rhs.fluid[species], name)
    return {
        name: {
            "maximum_absolute": float(
                context.compute.xp.max(context.compute.xp.abs(value)).item()
            )
            if value.size
            else 0.0,
            "shape": list(value.shape),
        }
        for name, value in values.items()
    }


def _equilibrium_checks(records: list[Any]) -> list[dict[str, Any]]:
    return [
        _check_record(record)
        for record in records
        if str(record.name).startswith("equilibrium_")
    ]


def _measure_equilibrium(frame: Any, context: Any) -> list[Any]:
    """Record equilibrium RHS residuals without gating a non-equilibrium seed."""
    records: list[Any] = []
    tag = frame.tag
    for species in range(2):
        diagnostics.measure(
            records,
            "equilibrium_n",
            frame.rhs.fluid[species].n,
            context.op.M0,
            "1/(m^3*s)",
            tag,
            context,
            context.p.species[species].id,
        )
        diagnostics.measure(
            records,
            "equilibrium_perp",
            frame.rhs.fluid[species].perp,
            context.op.M0,
            "Pa/s",
            tag,
            context,
            context.p.species[species].id,
        )
        diagnostics.measure(
            records,
            "equilibrium_mom",
            frame.rhs.fluid[species].mom,
            context.compute.xp.tile(
                context.compute.xp.real(
                    context.compute.xp.asarray(context.op.M0)
                ).astype(context.compute.xp.float64),
                3,
            ),
            "kg/(m^2*s^2)",
            tag,
            context,
            context.p.species[species].id,
        )
    diagnostics.measure(
        records,
        "equilibrium_parallel",
        frame.rhs.fluid[1].parallel,
        context.op.M0,
        "Pa/s",
        tag,
        context,
        context.p.species[1].id,
    )
    diagnostics.measure(
        records,
        "equilibrium_B",
        frame.rhs.B,
        context.op.MB,
        "T/s",
        tag,
        context,
    )
    for record in records:
        record.hard = False
        if record.available:
            record.reason = (
                "observed initial non-equilibrium residual; not an acceptance gate"
            )
    return records


def _validate_parent_checkpoints(
    parents: list[dict[str, str]], seen: set[Path] | None = None
) -> None:
    seen = set() if seen is None else seen
    for descriptor in parents:
        if not isinstance(descriptor, dict):
            raise ValueError("resume parent checkpoint descriptor is invalid")
        path = Path(str(descriptor.get("path", ""))).resolve()
        expected = descriptor.get("sha256")
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"resume parent checkpoint hash mismatch: {path}")
        if path in seen:
            continue
        seen.add(path)
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
        nested = metadata.get("parent_checkpoints", [])
        if not isinstance(nested, list):
            raise ValueError("resume parent checkpoint chain is invalid")
        _validate_parent_checkpoints(nested, seen)


def _read_resume_metadata(
    resume_from: Path, label: str
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray], str]:
    directory = Path(resume_from)
    report_path = directory / "report.json"
    if not report_path.is_file():
        raise ValueError("resume_from is missing report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint = directory / "checkpoints" / f"{label}.npz"
    declared = report.get(label, {}).get("checkpoint")
    actual_sha = _sha256_file(checkpoint) if checkpoint.is_file() else None
    declared_path = (
        Path(str(declared.get("path", "")))
        if isinstance(declared, dict)
        else Path("")
    )
    if not declared_path.is_absolute():
        declared_path = directory / declared_path
    if (
        not isinstance(declared, dict)
        or declared_path.resolve() != checkpoint.resolve()
        or declared.get("sha256") != actual_sha
    ):
        raise ValueError(f"resume checkpoint file hash mismatch: {label}")
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    if "metadata_json" not in arrays:
        raise ValueError("resume checkpoint metadata_json is missing")
    metadata = json.loads(str(np.asarray(arrays["metadata_json"]).item()))
    if metadata.get("schema") != "fomhd-warp-050-accepted-state-v2":
        raise ValueError("resume checkpoint schema mismatch")
    if metadata.get("accepted") is not True:
        raise ValueError("resume checkpoint is not accepted state")
    if "accepted" not in arrays or not np.asarray(arrays["accepted"]).item():
        raise ValueError("resume checkpoint is not accepted state")
    for name, dtype in (
        ("state_step", np.uint64),
        ("state_generation", np.uint64),
        ("state_time", np.float64),
        ("last_trial_id", np.uint64),
        ("last_source_id", np.uint64),
    ):
        if name not in arrays:
            raise ValueError(f"resume checkpoint missing {name}")
        value = np.asarray(arrays[name])
        if value.shape != () or value.dtype != np.dtype(dtype):
            raise ValueError(f"resume checkpoint {name} has invalid dtype or shape")
    if int(arrays["state_step"].item()) != metadata.get("step"):
        raise ValueError("resume checkpoint step metadata mismatch")
    if int(arrays["state_generation"].item()) != metadata.get("generation"):
        raise ValueError("resume checkpoint generation metadata mismatch")
    if float(arrays["state_time"].item()) != metadata.get("time_s"):
        raise ValueError("resume checkpoint time metadata mismatch")
    if "cumulative_json" not in arrays:
        raise ValueError("resume checkpoint cumulative_json is missing")
    cumulative_text = str(np.asarray(arrays["cumulative_json"]).item())
    if metadata.get("cumulative_json") != json.loads(cumulative_text):
        raise ValueError("resume checkpoint cumulative metadata mismatch")
    scheduler = metadata.get("scheduler")
    if not isinstance(scheduler, dict):
        raise ValueError("resume checkpoint scheduler metadata is missing")
    for name in ("last_trial", "last_source"):
        if int(arrays[f"{name}_id"].item()) != scheduler.get(name):
            raise ValueError(f"resume checkpoint {name} metadata mismatch")
    if (
        scheduler.get("next_trial") != int(arrays["last_trial_id"].item()) + 1
        or scheduler.get("next_source") != int(arrays["last_source_id"].item()) + 1
        or scheduler.get("next_generation", 0)
        <= int(arrays["state_generation"].item())
        or scheduler.get("state_generation")
        != int(arrays["state_generation"].item())
    ):
        raise ValueError("resume checkpoint scheduler counters are inconsistent")
    parents = metadata.get("parent_checkpoints")
    if not isinstance(parents, list):
        raise ValueError("resume checkpoint parent_checkpoints is missing")
    _validate_parent_checkpoints(parents)
    return checkpoint, metadata, arrays, actual_sha


def _resume_plan(
    metadata: dict[str, Any], controls: dict[str, Any]
) -> dict[str, Any]:
    saved = metadata.get("time_plan")
    if not isinstance(saved, dict):
        raise ValueError("resume checkpoint time_plan is missing")
    try:
        plan = {
            "coupling": str(saved["coupling"]),
            "h_s": float(saved["h_s"]),
            "micro_steps": int(saved["micro_steps"]),
            "dt_s": float(saved["dt_s"]),
            "nsteps": int(saved["nsteps"]),
            "horizon_s": float(saved["horizon_s"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("resume checkpoint time_plan is invalid") from error
    if (
        plan["coupling"] != controls["coupling"]
        or plan["micro_steps"] != controls["micro_steps"]
        or plan["horizon_s"] != controls["horizon_s"]
        or (
            controls["h_requested"] != "auto"
            and float(controls["h_requested"]) != plan["h_s"]
        )
    ):
        raise ValueError("resume time configuration mismatch")
    recomputed = plan_time(
        plan["horizon_s"],
        plan["h_s"],
        plan["micro_steps"],
        plan["coupling"],
    )
    if any(
        recomputed[key] != plan[key]
        for key in (
            "coupling",
            "h_s",
            "micro_steps",
            "dt_s",
            "nsteps",
            "horizon_s",
        )
    ):
        raise ValueError("resume time plan is inconsistent")
    return plan


def _validate_resume_seed_identity(
    control: dict[str, Any],
    perturbed: dict[str, Any],
    controls: dict[str, Any],
) -> dict[str, Any]:
    if control.get("initial_provenance") != perturbed.get("initial_provenance"):
        raise ValueError("resume matched-pair initial provenance mismatch")
    provenance = control.get("initial_provenance")
    seed_identity = (
        provenance.get("seed_identity")
        if isinstance(provenance, dict)
        else None
    )
    if not isinstance(seed_identity, dict):
        raise ValueError("resume immutable seed identity is missing")
    if controls["mode"] != seed_identity.get("mode"):
        raise ValueError("resume perturbation mode mismatch")
    if float(controls["amplitude"]) != float(seed_identity.get("amplitude")):
        raise ValueError("resume perturbation amplitude mismatch")
    seed_sha256 = seed_identity.get("initial_perturbed_sha256")
    if (
        not isinstance(seed_sha256, str)
        or len(seed_sha256) != 64
        or any(character not in "0123456789abcdef" for character in seed_sha256)
    ):
        raise ValueError("resume immutable seed identity is invalid")
    selector = seed_identity.get("selector")
    if (
        not isinstance(selector, dict)
        or selector.get("kind") not in ("smooth", "spectrum_archive")
    ):
        raise ValueError("resume seed selector identity is missing")
    explicit_selector = controls["spectrum_directory"]
    if explicit_selector is not None:
        requested = _seed_selector_identity(Path(explicit_selector))
        if requested != selector:
            raise ValueError("resume seed selector mismatch")
    return seed_identity


def _load_resume_experiments(
    resume_from: Path,
    context: Any,
    controls: dict[str, Any],
    stability_check: dict[str, Any] | None = None,
    preloaded: dict[str, Any] | None = None,
    seed_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loaded = (
        {
            label: _read_resume_metadata(Path(resume_from), label)
            for label in ("control", "perturbed")
        }
        if preloaded is None
        else preloaded
    )
    control = loaded["control"][1]
    perturbed = loaded["perturbed"][1]
    for name in (
        "physical_identity",
        "source_hashes",
        "runtime_fingerprint",
    ):
        if control.get(name) != perturbed.get(name):
            raise ValueError(f"resume matched-pair {name} mismatch")
    if control.get("compatibility") != perturbed.get("compatibility"):
        raise ValueError("resume matched-pair compatibility mismatch")
    if seed_identity is None:
        seed_identity = _validate_resume_seed_identity(
            control, perturbed, controls
        )
    seed_sha256 = seed_identity["initial_perturbed_sha256"]
    runtime_fingerprint = _runtime_fingerprint()
    if control.get("runtime_fingerprint") != runtime_fingerprint:
        raise ValueError("resume runtime fingerprint mismatch")
    if control.get("run_id") != perturbed.get("run_id"):
        raise ValueError("resume matched-pair run identity mismatch")
    control_scheduler = dict(control.get("scheduler", {}))
    perturbed_scheduler = dict(perturbed.get("scheduler", {}))
    control_scheduler.pop("state_generation", None)
    perturbed_scheduler.pop("state_generation", None)
    if control_scheduler != perturbed_scheduler:
        raise ValueError("resume matched-pair scheduler identity mismatch")
    if control.get("parent_checkpoints") != perturbed.get("parent_checkpoints"):
        raise ValueError("resume matched-pair parent chain mismatch")
    if control.get("step") != perturbed.get("step") or control.get(
        "time_s"
    ) != perturbed.get("time_s"):
        raise ValueError("resume matched-pair accepted endpoint mismatch")

    plan = _resume_plan(control, controls)
    context.p.coupling = plan["coupling"]
    context.p.micro_steps = plan["micro_steps"]
    context.p.dt = plan["dt_s"]
    context.p.nsteps = plan["nsteps"]
    context.p.start_time = 0.0
    context.p.dt_guard = "warn"
    context.p.waveform_rtol = 1.0e-10
    context.p.waveform_max_iterations = 12

    compatibility = _compatibility_identity(
        context, controls, plan, seed_sha256=seed_sha256
    )
    if control.get("compatibility") != compatibility:
        raise ValueError("resume controls/geometry/operator/source mismatch")
    if control.get("physical_identity") != cae.physical_identity(context):
        raise ValueError("resume physical identity mismatch")
    if control.get("source_hashes") != _source_hashes():
        raise ValueError("resume source fingerprint mismatch")
    stored_stability = control.get("stability")
    if stability_check is not None:
        if stored_stability is None:
            raise ValueError("resume stability policy mismatch")
        stored_files = [
            (entry["spectrum_sha256"], entry["report_sha256"])
            for entry in stored_stability.get("spectra", [])
        ]
        checked_files = [
            (entry["spectrum_sha256"], entry["report_sha256"])
            for entry in stability_check["spectra"]
        ]
        if (
            stored_files != checked_files
            or stored_stability.get("rho_max_rad_s")
            != stability_check.get("rho_max_rad_s")
        ):
            raise ValueError("resume stability spectrum identity mismatch")
    stability = stability_check or stored_stability

    cumulative_values = {}
    reconstructed_states = {}
    for label, (_, metadata, arrays, _) in loaded.items():
        cumulative = io._cumulative_from_json(metadata["cumulative_json"])
        if "baseline_divB" not in arrays:
            raise ValueError("resume checkpoint baseline_divB is missing")
        cumulative.baseline.divB = context.compute.array(
            io.decode_vertex(
                arrays["baseline_divB"],
                context.geometry,
                context.p.K,
            ),
            np.complex128,
        )
        state = io.decode_state(arrays, context.geometry, context.p.K)
        io._upload_state(state, context.compute)
        state.step = int(arrays["state_step"].item())
        state.generation = int(arrays["state_generation"].item())
        state.time = float(arrays["state_time"].item())
        if (
            state.step > context.p.nsteps
            or state.time != core.time_at_step(context.p, state.step)
            or state.generation == (1 << 64) - 1
        ):
            raise ValueError("resume scheduler identity mismatch")
        if (
            cumulative.baseline.step > state.step
            or cumulative.baseline.time
            != core.time_at_step(context.p, cumulative.baseline.step)
            or cumulative.last_step != state.step
            or cumulative.last_time != state.time
            or cumulative.accepted_intervals
            != state.step - cumulative.baseline.step
        ):
            raise ValueError("resume baseline or interval identity mismatch")
        if cumulative.accepted_intervals and cumulative.last_trial == 0:
            raise ValueError("resume accepted interval trial is missing")
        scheduler = metadata["scheduler"]
        for name in ("next_trial", "next_source", "next_generation"):
            value = scheduler.get(name)
            if type(value) is not int or not 0 < value < (1 << 64) - 1:
                raise ValueError(f"resume scheduler {name} is invalid")
        if (
            int(arrays["last_trial_id"].item())
            < cumulative.last_trial
            or int(arrays["last_source_id"].item()) < 0
        ):
            raise ValueError("resume scheduler counters are inconsistent")
        cumulative_values[label] = cumulative
        reconstructed_states[label] = state

    control_cumulative = cumulative_values["control"]
    perturbed_cumulative = cumulative_values["perturbed"]
    if (
        control_cumulative.baseline.step
        != perturbed_cumulative.baseline.step
        or control_cumulative.baseline.time
        != perturbed_cumulative.baseline.time
        or control_cumulative.accepted_intervals
        != perturbed_cumulative.accepted_intervals
    ):
        raise ValueError("resume matched-pair ledger identity mismatch")

    scheduler = control["scheduler"]
    context.next_trial = int(scheduler["next_trial"])
    context.next_source = int(scheduler["next_source"])
    context.next_generation = max(
        int(scheduler["next_generation"]),
        max(state.generation for state in reconstructed_states.values()) + 1,
    )
    source_before = context.next_source
    trial_before = context.next_trial
    reconstruction_started = time.monotonic()
    experiments: dict[str, dict[str, Any]] = {}
    resume_checks: dict[str, list[Any]] = {}
    for label, state in reconstructed_states.items():
        state.generation += 1
        validation_ledger = cumulative_record()
        captured: dict[str, Any] = {}
        committed = stage.initialize(
            state,
            context,
            validation_ledger,
            False,
            captured,
        )
        frame = SimpleNamespace(**captured)
        ledger = cumulative_values[label]
        if committed.state.step != ledger.last_step:
            raise ValueError("resume cache reconstruction changed accepted step")
        experiments[label] = {
            "state": committed.state,
            "frame": frame,
            "ledger": ledger,
            "baseline_energy": float(ledger.baseline.totals.energy),
            "checks": {
                "pass": 0,
                "observe_fail": 0,
                "hard_fail": 0,
                "unavailable": 0,
            },
            "initial_rhs": _initial_rhs_record(frame, context),
            "equilibrium_checks": _equilibrium_checks(
                _measure_equilibrium(frame, context)
            ),
            "coupling_records": 0,
            "actual_model_evaluations": 0,
            "iterations": 0,
            "guard_baseline": loaded[label][1]["guard_baseline"],
        }
        resume_checks[label] = list(frame.checks)
    context.compute.synchronize()
    reconstruction = {
        "elapsed_s": time.monotonic() - reconstruction_started,
        "source_evaluations": context.next_source - source_before,
        "trial_identities": context.next_trial - trial_before,
        "scope": (
            "throwaway-ledger stage.initialize cache reconstruction; "
            "accepted cumulative ledger restored without reinitialization"
        ),
    }
    input_checkpoints = [
        {
            "label": label,
            "path": str(loaded[label][0]),
            "sha256": loaded[label][3],
        }
        for label in ("control", "perturbed")
    ]
    return {
        "experiments": experiments,
        "resume_checks": resume_checks,
        "plan": plan,
        "metadata": control,
        "initial_provenance": control["initial_provenance"],
        "parent_checkpoints": input_checkpoints,
        "run_id": control["run_id"],
        "stability": stability,
        "start_step": int(control["step"]),
        "input_checkpoints": input_checkpoints,
        "reconstruction": reconstruction,
        "compatibility": compatibility,
    }


def run(args: Any, *, context_factory=None) -> int:
    """Run one bounded experiment and return 0 complete/2 stopped or failed."""
    controls = _validate_args(args, allow_fixture_nodes=context_factory is not None)
    output: Path = controls["output"]
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "logs" / "run.log"
    allocator_path = output / "logs" / "allocator.jsonl"
    failures_path = output / "logs" / "failures.jsonl"
    report_path = output / "report.json"
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema": "fomhd-warp-050-experiment-v2",
        "status": "initializing",
        "scope": (
            "complete nonlinear A0/T1 matched control and perturbed response; "
            "not experimental validation, a full nonnormal stability proof, "
            "or a production restart claim"
        ),
        "model": {
            "name": "fomhd massless-electron two-fluid A0/T1",
            "electron_inertia": 0,
            "integrator": "rk4",
            "Nphi": 5,
            "prototype_policy": "prototype_observe_power_energy_momentum",
        },
        "case": {key: value for key, value in controls.items() if key not in ("output",)},
        "requested_horizon_s": controls["horizon_s"],
        "budget_stopped": False,
        "source": _source_snapshot(output),
        "environment": {
            "cpu_threads": 1,
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "memory_caps": {
            "host_bytes": HOST_MEMORY_CAP,
            "device_bytes": DEVICE_MEMORY_CAP,
        },
    }
    _write_json(report_path, report)
    _log(log_path, f"output initialized at {output}")
    context = None
    experiments: dict[str, dict[str, Any]] = {}
    run_id: str | None = None
    compatibility: dict[str, Any] | None = None
    initial_provenance: dict[str, Any] | None = None
    parent_checkpoints: list[dict[str, str]] = []
    stability: dict[str, Any] | None = None
    resume_metadata = None
    resume_seed_identity = None
    try:
        if controls["resume_from"] is not None:
            resume_metadata = {
                label: _read_resume_metadata(
                    Path(controls["resume_from"]), label
                )
                for label in ("control", "perturbed")
            }
            control_metadata = resume_metadata["control"][1]
            perturbed_metadata = resume_metadata["perturbed"][1]
            if (
                control_metadata.get("initial_provenance")
                != perturbed_metadata.get("initial_provenance")
            ):
                raise ValueError(
                    "resume matched-pair initial provenance mismatch"
                )
            provenance = control_metadata["initial_provenance"]
            immutable_seed = provenance.get("seed_identity")
            selector = (
                immutable_seed.get("selector")
                if isinstance(immutable_seed, dict)
                else None
            )
            if (
                not isinstance(selector, dict)
                or selector.get("kind") not in ("smooth", "spectrum_archive")
            ):
                raise ValueError("resume seed selector identity is missing")
            report["seed_selector"] = _jsonable(selector)
            if selector["kind"] == "spectrum_archive":
                report["case"]["spectrum_directory"] = selector.get(
                    "directory"
                )
                selected = selector.get("selected")
                report["selected_frequency_hz"] = _finite_positive(
                    selected.get("frequency_hz")
                    if isinstance(selected, dict)
                    else None,
                    "selected.frequency_hz",
                )
            else:
                report["case"]["spectrum_directory"] = None
                report["selected_frequency_hz"] = provenance.get(
                    "target_frequency_hz"
                )
            resume_seed_identity = _validate_resume_seed_identity(
                control_metadata,
                perturbed_metadata,
                controls,
            )
        if controls["device"] == "cuda:0" and not os.environ.get("SLURM_JOB_ID"):
            raise ValueError("CUDA requires an owned Slurm allocation")
        if controls["device"] == "cuda:0":
            import cupy

            cupy.cuda.Device(0).use()
            cupy.get_default_memory_pool().set_limit(size=DEVICE_MEMORY_CAP)
        selected_factory = context_factory or _make_context
        context_started = time.monotonic()
        context, meta, eq = selected_factory(
            controls["kind"],
            controls["nodes"],
            controls["K"],
            controls["device"],
            controls["equilibrium"],
            HOST_MEMORY_CAP,
        )
        context.compute.synchronize()
        report["context_construction_s"] = time.monotonic() - context_started
        context.p.memory_limit = HOST_MEMORY_CAP
        context.p.device_memory_limit = DEVICE_MEMORY_CAP
        context.p.Nphi = 5
        context.N = 5
        context.p.electron_inertia = 0
        context.p.integrator = "rk4"
        context.p.dt_guard = "warn"
        report["operator_construction"] = getattr(
            context,
            "_warp_050_operator_option",
            {
                "name": "include_finite_mass",
                "requested_value": False,
                "applied": "include_finite_mass"
                in {field.name for field in fields(OperatorOptions)},
                "note": "injected fixture context",
            },
        )
        if controls["kind"] == "east" and eq is not None:
            equilibrium_sha = controls["equilibrium"]
            if equilibrium_sha is not None:
                meta["equilibrium_sha256"] = _sha256_file(Path(equilibrium_sha))
        report["case"].update(_jsonable(meta))
        report["actual_grid"] = {
            "Nc": int(context.op.Nc),
            "Ne": int(context.op.Ne),
            "Nb": int(context.op.Nb),
            "Nphi": int(context.N),
        }
        background_started = time.monotonic()
        initial, background_arrays, background_mapping = cae.background(
            context, meta, eq
        )
        report["background_reconstruction_s"] = (
            time.monotonic() - background_started
        )
        report["case"]["background_mapping"] = _jsonable(background_mapping)
        np.savez_compressed(
            output / "input_mapping.npz",
            **background_arrays,
            R=context.geometry.cells.R,
            Z=context.geometry.cells.Z,
            polygon=np.array(
                [[point.R, point.Z] for point in context.geometry.polygon]
            ),
        )

        if controls["resume_from"] is None:
            seed = _clone_state(initial, context)
            archive = None
            selector = _seed_selector_identity(None)
            if controls["spectrum_directory"] is not None:
                archive = read_archive_seed(
                    Path(controls["spectrum_directory"]),
                    context,
                    initial,
                    kind=controls["kind"],
                    nodes=controls["nodes"],
                    K=controls["K"],
                    mode=controls["mode"],
                    target_frequency_hz=controls["target_frequency_hz"],
                )
                selector = _seed_selector_identity(
                    Path(controls["spectrum_directory"]),
                    archive=archive,
                )
                choice = {
                    "fields": archive.pop("fields"),
                    "frequency_hz": float(archive["selected"]["frequency_hz"]),
                    "index": int(archive["selected"]["index"]),
                }
                cae.discrete_perturb(
                    seed,
                    context,
                    meta,
                    controls["mode"],
                    controls["amplitude"],
                    choice,
                )
                report["selected_frequency_hz"] = float(
                    choice["frequency_hz"]
                )
                report["archive"] = archive
                _append_jsonl(
                    output / "logs" / "archive_source_mismatch.jsonl",
                    {
                        "source_hash_match": archive["source_hash_match"],
                        "mismatches": archive["source_hash_mismatches"],
                        "note": (
                            "shared numerical equivalence is tested separately; "
                            "archive seed is accepted only on physical identity "
                            "and background agreement"
                        ),
                    },
                )
            else:
                cae.perturb(
                    seed,
                    context,
                    meta,
                    controls["mode"],
                    controls["amplitude"],
                    None,
                )
                report["selected_frequency_hz"] = controls[
                    "target_frequency_hz"
                ]
            report["case"]["spectrum_directory"] = selector.get("directory")
            report["seed_selector"] = _jsonable(selector)
            np.savez_compressed(
                output / "perturbation_mapping.npz",
                **context.seed_arrays,
            )
            report["seed"] = _jsonable(meta)

            if controls["stability_spectrum_directories"]:
                stability = read_stability_spectra(
                    controls["stability_spectrum_directories"],
                    context,
                    initial,
                    kind=controls["kind"],
                    nodes=controls["nodes"],
                    K=controls["K"],
                )
                report["stability"] = stability
                for entry in stability["spectra"]:
                    _append_jsonl(
                        output / "logs" / "stability_source_mismatch.jsonl",
                        {
                            "mode": entry["mode"],
                            "source_hash_match": entry["source_hash_match"],
                            "mismatches": entry["source_hash_mismatches"],
                            "note": (
                                "equivalent source optimizations are reviewed "
                                "separately; physical and baseline identity "
                                "remain hard requirements"
                            ),
                        },
                    )

            target = (
                report.get("selected_frequency_hz")
                if report.get("selected_frequency_hz") is not None
                else controls["target_frequency_hz"]
            )
            plan, probes = _resolve_time_plan(
                {**controls, "target_frequency_hz": target},
                context,
                {"control": initial, "perturbed": seed},
                stability_radius_rad_s=(
                    None
                    if stability is None
                    else stability["rho_max_rad_s"]
                ),
            )
            report["initial_frequency_estimate"] = _jsonable(probes)
            report["time_plan"] = plan
            report["planned_macros"] = plan["nsteps"]
            report["requested_h_s"] = controls["h_requested"]
            report["h_s"] = plan["h_s"]
            report["resolved_horizon_s"] = plan["horizon_s"]
            context.p.coupling = controls["coupling"]
            context.p.micro_steps = plan["micro_steps"]
            context.p.dt = plan["dt_s"]
            context.p.nsteps = plan["nsteps"]
            context.p.start_time = 0.0
            context.p.dt_guard = "warn"
            context.p.waveform_rtol = 1.0e-10
            context.p.waveform_max_iterations = 12
            report["target_accuracy"] = _target_accuracy(
                target,
                plan["h_s"],
                plan["micro_steps_total"],
            )
            report["target_frequency_hz"] = target
            _require_target_accuracy(report["target_accuracy"])
            report["step_policy"] = {
                "historical_assess_dt": "unchanged; dt_guard=warn",
                "hard_runtime_envelope": (
                    "omega_current*h <= 0.5"
                    if stability is None
                    else "not applied when complete spectra are present"
                ),
                "estimated_operator_warning": "recorded; not a high-mode gate",
                "resolved_spectrum_policy": (
                    "complete retained sectors cap micro h at "
                    "0.5/(1.5*rho_max)"
                    if stability is not None
                    else "conservative operator-resolution policy"
                ),
                "runtime_neighborhood_guard": (
                    "max(Bmax/Bmax0,1)*max(nmin0/nmin,1)<=1.5 and "
                    "rho_max*ratio*micro_h<=0.5"
                    if stability is not None
                    else "not applicable without complete spectra"
                ),
                "micro_h": plan["h_s"],
                "macro_dt": plan["dt_s"],
                "t1_relation": (
                    "p.dt = p.micro_steps * h"
                    if controls["coupling"] == "waveform"
                    else "synchronous A0 macro dt = h"
                ),
                "estimate_scope": probes["scope"],
                "target_accuracy_is_separate": True,
                "a0_comparison": (
                    "synchronous A0 must use the same h and physical horizon; "
                    "not waveform macro H=M*h"
                ),
                "parent_controls": "h/2 and amplitude controls remain required",
                "no_silent_window_or_mode_reduction": True,
            }
            _write_json(output / "resolved.json", asdict(context.p))

            run_id = uuid.uuid4().hex
            report["run_id"] = run_id
            initial_control_sha256 = _state_sha256(initial, context)
            initial_perturbed_sha256 = _state_sha256(seed, context)
            compatibility = _compatibility_identity(
                context,
                controls,
                plan,
                seed_sha256=initial_perturbed_sha256,
            )
            initial_provenance = _jsonable(
                {
                    "run_id": run_id,
                    "origin_output": str(output),
                    "global_start_time_s": 0.0,
                    "global_horizon_s": plan["horizon_s"],
                    "global_nsteps": plan["nsteps"],
                    "global_time_plan": plan,
                    "target_frequency_hz": target,
                    "mode": int(controls["mode"]),
                    "amplitude": float(controls["amplitude"]),
                    "seed_identity": {
                        "mode": int(controls["mode"]),
                        "amplitude": float(controls["amplitude"]),
                        "initial_perturbed_sha256": initial_perturbed_sha256,
                        "selector": selector,
                    },
                    "source_hashes": _source_hashes(),
                    "runtime_fingerprint": _runtime_fingerprint(),
                    "physical_identity": cae.physical_identity(context),
                    "initial_frequency_estimate": probes,
                    "stability_spectra": (
                        []
                        if stability is None
                        else [
                            {
                                "mode": entry["mode"],
                                "spectrum_sha256": entry["spectrum_sha256"],
                                "report_sha256": entry["report_sha256"],
                                "spectral_radius_rad_s": entry[
                                    "spectral_radius_rad_s"
                                ],
                            }
                            for entry in stability["spectra"]
                        ]
                    ),
                    "initial_control_sha256": initial_control_sha256,
                    "initial_perturbed_sha256": initial_perturbed_sha256,
                }
            )
            parent_checkpoints = []

            for label, state in (("control", initial), ("perturbed", seed)):
                ledger = cumulative_record()
                captured: dict[str, Any] = {}
                committed = stage.initialize(
                    _clone_state(state, context),
                    context,
                    ledger,
                    False,
                    captured,
                )
                frame = SimpleNamespace(**captured)
                equilibrium_records = _measure_equilibrium(frame, context)
                rhs_arrays = io.encode_state(frame.rhs, context.geometry)
                np.savez_compressed(
                    output / f"{label}-initial-rhs.npz",
                    **rhs_arrays,
                )
                checks_path = output / "logs" / f"{label}-checks.jsonl"
                counts = {
                    "pass": 0,
                    "observe_fail": 0,
                    "hard_fail": 0,
                    "unavailable": 0,
                }
                _append_checks(
                    checks_path,
                    list(frame.checks) + equilibrium_records,
                    counts,
                )
                initial_rhs = _initial_rhs_record(frame, context)
                equilibrium = _equilibrium_checks(equilibrium_records)
                _write_json(
                    output / "logs" / f"{label}-initial.json",
                    {
                        "rhs_norms": initial_rhs,
                        "equilibrium_residual_checks": equilibrium,
                        "equilibrium_residual_acceptance_gate": False,
                        "equilibrium_residual_note": (
                            "recorded raw residuals; a perturbed initial "
                            "state is not required to be an equilibrium"
                        ),
                    },
                )
                experiments[label] = {
                    "state": committed.state,
                    "frame": frame,
                    "ledger": ledger,
                    "baseline_energy": frame.totals.energy,
                    "checks": counts,
                    "initial_rhs": initial_rhs,
                    "equilibrium_checks": equilibrium,
                    "coupling_records": 0,
                    "actual_model_evaluations": 0,
                    "iterations": 0,
                    "guard_baseline": _guard_baseline(frame, context),
                }
                experiment = experiments[label]
                envelope = _envelope(experiment, context)
                history = _scalar_history(experiment, envelope)
                report.setdefault(label, {})
                report[label].update(
                    {
                        "accepted_steps": 0,
                        "time_s": float(experiment["state"].time),
                        "ledger_accepted_intervals": 0,
                        "checks": dict(counts),
                        "initial_runtime_envelope": envelope,
                        "initial_rhs_norms": initial_rhs,
                        "equilibrium_residual_checks": equilibrium,
                        "soft_budget_conclusion": (
                            "not an acceptance claim; raw observe failures are "
                            "retained"
                        ),
                    }
                )
                _append_jsonl(
                    output / "logs" / f"{label}-history.jsonl", history
                )
                _append_jsonl(
                    allocator_path,
                    _memory_sample(context, time.monotonic() - started),
                )
        else:
            stability_check = None
            if controls["stability_spectrum_directories"]:
                stability_check = read_stability_spectra(
                    controls["stability_spectrum_directories"],
                    context,
                    initial,
                    kind=controls["kind"],
                    nodes=controls["nodes"],
                    K=controls["K"],
                )
            resume_started = time.monotonic()
            resumed = _load_resume_experiments(
                controls["resume_from"],
                context,
                controls,
                stability_check=stability_check,
                preloaded=resume_metadata,
                seed_identity=resume_seed_identity,
            )
            report["resume_validation_s"] = time.monotonic() - resume_started
            experiments = resumed["experiments"]
            plan = dict(resumed["plan"])
            plan.update(
                {
                    "micro_steps_total": (
                        plan["nsteps"] * plan["micro_steps"]
                    ),
                    "h_provisional_s": None,
                    "h_policy": "restored exact accepted global time plan",
                    "stability_radius_rad_s": (
                        None
                        if resumed["stability"] is None
                        else resumed["stability"]["rho_max_rad_s"]
                    ),
                    "stability_cap_s": (
                        None
                        if resumed["stability"] is None
                        else resumed["stability"]["h_cap_s"]
                    ),
                    "operator_estimate_used_for_automatic_h": False,
                }
            )
            probes = resumed["initial_provenance"][
                "initial_frequency_estimate"
            ]
            stability = resumed["stability"]
            run_id = resumed["run_id"]
            compatibility = resumed["compatibility"]
            initial_provenance = resumed["initial_provenance"]
            parent_checkpoints = resumed["parent_checkpoints"]
            report["run_id"] = run_id
            if stability is not None:
                report["stability"] = stability
            report["initial_frequency_estimate"] = probes
            report["time_plan"] = plan
            report["planned_macros"] = plan["nsteps"]
            report["requested_h_s"] = controls["h_requested"]
            report["h_s"] = plan["h_s"]
            report["resolved_horizon_s"] = plan["horizon_s"]
            target = resumed["initial_provenance"].get(
                "target_frequency_hz"
            )
            report["target_accuracy"] = _target_accuracy(
                target,
                plan["h_s"],
                plan["micro_steps_total"],
            )
            report["target_frequency_hz"] = target
            _require_target_accuracy(report["target_accuracy"])
            report["step_policy"] = {
                "historical_assess_dt": "unchanged; dt_guard=warn",
                "hard_runtime_envelope": (
                    "omega_current*h <= 0.5"
                    if stability is None
                    else "not applied when complete spectra are present"
                ),
                "estimated_operator_warning": (
                    "recorded in original provenance; no new high-mode gate"
                ),
                "resolved_spectrum_policy": (
                    "restored complete-sector rho_max policy"
                    if stability is not None
                    else "restored conservative operator-resolution policy"
                ),
                "runtime_neighborhood_guard": (
                    "max(Bmax/Bmax0,1)*max(nmin0/nmin,1)<=1.5 and "
                    "rho_max*ratio*micro_h<=0.5"
                    if stability is not None
                    else "not applicable without complete spectra"
                ),
                "micro_h": plan["h_s"],
                "macro_dt": plan["dt_s"],
                "target_accuracy_is_separate": True,
                "no_silent_window_or_mode_reduction": True,
            }
            _write_json(output / "resolved.json", asdict(context.p))
            report["continuation"] = {
                "resume_from": str(Path(controls["resume_from"])),
                "start_step": resumed["start_step"],
                "start_time_s": float(
                    experiments["control"]["state"].time
                ),
                "input_checkpoints": resumed["input_checkpoints"],
                "parent_checkpoints": parent_checkpoints,
                "source_cache_reconstruction": resumed["reconstruction"],
                "policy": (
                    "restore accepted pair and cumulative ledger; reconstruct "
                    "cache on a throwaway ledger; no time-zero restart"
                ),
                "probe_macros_scope": (
                    "additional accepted macros in this segment"
                ),
            }
            for label, experiment in experiments.items():
                checks_path = output / "logs" / f"{label}-checks.jsonl"
                counts = experiment["checks"]
                _append_checks(
                    checks_path,
                    list(resumed["resume_checks"][label]),
                    counts,
                )
                rhs_arrays = io.encode_state(
                    experiment["frame"].rhs, context.geometry
                )
                np.savez_compressed(
                    output / f"{label}-resume-rhs.npz",
                    **rhs_arrays,
                )
                _write_json(
                    output / "logs" / f"{label}-resume.json",
                    {
                        "accepted_step": int(experiment["state"].step),
                        "accepted_time_s": float(experiment["state"].time),
                        "accepted_generation": int(
                            experiment["state"].generation
                        ),
                        "ledger": _ledger_json(experiment["ledger"]),
                        "rhs_norms": experiment["initial_rhs"],
                        "equilibrium_residual_checks": experiment[
                            "equilibrium_checks"
                        ],
                    },
                )
                envelope = _envelope(experiment, context)
                history = _scalar_history(experiment, envelope)
                report.setdefault(label, {})
                report[label].update(
                    {
                        "accepted_steps": int(experiment["state"].step),
                        "time_s": float(experiment["state"].time),
                        "ledger_accepted_intervals": int(
                            experiment["ledger"].accepted_intervals
                        ),
                        "checks": dict(counts),
                        "initial_runtime_envelope": envelope,
                        "initial_rhs_norms": experiment["initial_rhs"],
                        "equilibrium_residual_checks": experiment[
                            "equilibrium_checks"
                        ],
                        "soft_budget_conclusion": (
                            "not an acceptance claim; raw observe failures are "
                            "retained"
                        ),
                    }
                )
                _append_jsonl(
                    output / "logs" / f"{label}-history.jsonl", history
                )
                _append_jsonl(
                    allocator_path,
                    _memory_sample(context, time.monotonic() - started),
                )

        context.compute.synchronize()
        report["setup_through_initial_states_s"] = time.monotonic() - started
        report["setup_cost_breakdown"] = {
            "context_construction_s": report.get("context_construction_s"),
            "background_reconstruction_s": report.get(
                "background_reconstruction_s"
            ),
            "resume_validation_s": report.get("resume_validation_s"),
            "source_cache_reconstruction_s": (
                None
                if report.get("continuation") is None
                else report["continuation"]["source_cache_reconstruction"][
                    "elapsed_s"
                ]
            ),
            "total_setup_s": report["setup_through_initial_states_s"],
            "scope": (
                "context, input reconstruction, checkpoint validation, "
                "throwaway-cache reconstruction and initial diagnostics"
            ),
        }
        segment_start = int(experiments["control"]["state"].step)
        effective_limit = plan["nsteps"]
        if controls["probe_macros"] is not None:
            effective_limit = min(
                effective_limit,
                segment_start + controls["probe_macros"],
            )
        snapshots = set(snapshot_steps(plan["nsteps"], controls["snapshot_every"]))
        stop_reason = None
        failure_info = None
        reached = segment_start
        report["segment_probe"] = {
            "start_step": segment_start,
            "effective_macro_limit": effective_limit,
            "additional_macro_limit": controls["probe_macros"],
            "scope": (
                "macro limit is global endpoint; --probe-macros limits "
                "additional macros in this segment"
            ),
        }
        while reached < effective_limit:
            elapsed = time.monotonic() - started
            if elapsed >= controls["wall_seconds"]:
                stop_reason = "wall_budget"
                break
            envelope_values = {
                label: _envelope(experiment, context)
                for label, experiment in experiments.items()
            }
            _append_jsonl(
                output / "logs" / "runtime-envelope.jsonl",
                {
                    "before_macro": reached + 1,
                    "values": envelope_values,
                },
            )
            if stability is None:
                failed = {
                    label: value
                    for label, value in envelope_values.items()
                    if not value["accepted"]
                }
                if failed:
                    failure_info = {
                        "code": "RuntimeStabilityEnvelope",
                        "message": "omega_current*h exceeded 0.5",
                        "before_macro": reached + 1,
                        "ensembles": failed,
                    }
                    _append_jsonl(failures_path, failure_info)
                    break

            if stability is not None:
                micro_h = (
                    context.p.dt / context.p.micro_steps
                    if context.p.coupling == "waveform"
                    else context.p.dt
                )
                neighborhood = {
                    label: _neighborhood_guard(
                        experiment["frame"],
                        context,
                        baseline_magnetic_max_T=experiment[
                            "guard_baseline"
                        ]["magnetic_max_T"],
                        baseline_density_min=experiment[
                            "guard_baseline"
                        ]["density_min"],
                        baseline_rate_estimate=probes["by_ensemble"][label],
                        rho_max_rad_s=stability["rho_max_rad_s"],
                        micro_h_s=micro_h,
                    )
                    for label, experiment in experiments.items()
                }
                _append_jsonl(
                    output / "logs" / "stability-neighborhood.jsonl",
                    {
                        "before_macro": reached + 1,
                        "values": neighborhood,
                        "scope": stability["scope"],
                    },
                )
                neighborhood_failed = {
                    label: value
                    for label, value in neighborhood.items()
                    if not value["accepted"]
                }
                if neighborhood_failed:
                    failure_info = {
                        "code": "ResolvedSpectrumNeighborhoodGuard",
                        "message": (
                            "resolved-spectrum local-neighborhood guard "
                            "exceeded"
                        ),
                        "before_macro": reached + 1,
                        "ensembles": neighborhood_failed,
                        "scope": (
                            "local linear/neighborhood guard only; not a "
                            "nonlinear stability proof"
                        ),
                    }
                    _append_jsonl(failures_path, failure_info)
                    break

            trial: dict[str, Any] = {}
            trial_failure: dict[str, BaseException | dict[str, Any]] = {}
            for label, experiment in experiments.items():
                committed, ledger = _clone_experiment_start(experiment, context)
                context.compute.synchronize()
                advance_started = time.monotonic()
                try:
                    result = advance(committed, ledger, context)
                    context.compute.synchronize()
                    _append_jsonl(output / "logs" / "advance-timing.jsonl", {
                        "ensemble": label, "macro": reached + 1,
                        "elapsed_s": time.monotonic() - advance_started,
                        "status": "completed",
                        "scope": "synchronized complete advance; state copy and output excluded",
                    })
                    trial[label] = {
                        "state": committed.state,
                        "frame": result.frame,
                        "ledger": ledger,
                        "result": result,
                    }
                except Exception as error:
                    try:
                        context.compute.synchronize()
                    except Exception:
                        pass
                    _append_jsonl(
                        output / "logs" / "advance-timing.jsonl",
                        {
                            "ensemble": label,
                            "macro": reached + 1,
                            "elapsed_s": time.monotonic() - advance_started,
                            "status": "failed",
                            "scope": (
                                "synchronized failed attempt including solver "
                                "transaction; state copy and output excluded"
                            ),
                        },
                    )
                    error._warp050_traceback = (
                        __import__("traceback").format_exc()
                    )
                    trial_failure[label] = error
            if trial_failure:
                for label, error in trial_failure.items():
                    error_info = {
                        "code": getattr(error, "code", type(error).__name__),
                        "message": str(error),
                        "traceback": getattr(
                            error, "_warp050_traceback", ""
                        ),
                        "stage": _jsonable(getattr(error, "stage", None)),
                        "coupling_metadata": _jsonable(
                            getattr(error, "coupling_metadata", None)
                        ),
                    }
                    path = (
                        output
                        / "failures"
                        / f"{label}-macro-{reached + 1:08d}.npz"
                    )
                    _write_failure(
                        path,
                        error,
                        context,
                        label=label,
                        step=reached,
                    )
                    _append_jsonl(
                        failures_path,
                        {
                            "ensemble": label,
                            "macro": reached + 1,
                            **error_info,
                        },
                    )
                    error_checks = list(getattr(error, "checks", []) or [])
                    if error_checks:
                        _append_checks(
                            output / "logs" / f"{label}-checks.jsonl",
                            error_checks,
                            experiments[label]["checks"],
                        )
                    _append_jsonl(
                        output / "logs" / f"{label}-coupling.jsonl",
                        {
                            "macro": reached + 1,
                            "available": getattr(
                                error, "coupling_metadata", None
                            )
                            is not None,
                            "accepted": False,
                            "metadata": error_info["coupling_metadata"],
                        },
                    )
                failure_info = {
                    "code": "MatchedMacroFailure",
                    "message": "one or both matched transactions failed",
                    "ensembles": sorted(trial_failure),
                    "accepted_prefix_macros": reached,
                }
                break

            next_step = reached + 1
            for label, experiment in experiments.items():
                value = trial[label]
                if int(value["ledger"].accepted_intervals) != next_step:
                    raise RuntimeError(
                        f"{label} ledger did not advance exactly once per macro"
                    )
                if int(value["state"].step) != next_step:
                    raise RuntimeError(f"{label} state step mismatch")
                experiment["state"] = value["state"]
                experiment["frame"] = value["frame"]
                experiment["ledger"] = value["ledger"]
                result = value["result"]
                _append_checks(
                    output / "logs" / f"{label}-checks.jsonl",
                    list(result.checks),
                    experiment["checks"],
                )
                metadata = _jsonable(getattr(result, "coupling_metadata", None))
                iterations = (
                    list(metadata.get("iterations", []))
                    if isinstance(metadata, dict)
                    else []
                )
                actual_calls = (
                    int(metadata.get("actual_model_call_count", 0))
                    if isinstance(metadata, dict)
                    else 0
                )
                experiment["actual_model_evaluations"] += actual_calls
                experiment["iterations"] += len(iterations)
                experiment["coupling_records"] += 1
                _append_jsonl(
                    output / "logs" / f"{label}-coupling.jsonl",
                    {
                        "macro": next_step,
                        "available": metadata is not None,
                        "accepted": bool(result.accepted),
                        "metadata": metadata,
                    },
                )
                _append_jsonl(
                    output / "logs" / f"{label}-history.jsonl",
                    _scalar_history(experiment, envelope_values[label]),
                )
                report[label].update(
                    {
                        "accepted_steps": next_step,
                        "time_s": float(experiment["state"].time),
                        "ledger_accepted_intervals": int(
                            experiment["ledger"].accepted_intervals
                        ),
                        "checks": dict(experiment["checks"]),
                        "actual_model_evaluations": experiment[
                            "actual_model_evaluations"
                        ],
                        "iteration_records": experiment["iterations"],
                        "coupling_records": experiment["coupling_records"],
                    }
                )
                if next_step in snapshots:
                    path = (
                        output
                        / "snapshots"
                        / f"{label}-step-{next_step:08d}.npz"
                    )
                    digest = _write_snapshot(
                        path, experiment, context, envelope_values[label]
                    )
                    report[label].setdefault("snapshots", []).append(
                        {
                            "step": next_step,
                            "path": str(path.relative_to(output)),
                            "sha256": digest,
                        }
                    )
            reached = next_step
            _append_jsonl(allocator_path, _memory_sample(context, time.monotonic() - started))
            _log(
                log_path,
                f"accepted matched macro {reached}/{plan['nsteps']} "
                f"time={experiments['control']['state'].time:.17g}s",
            )
            if reached == plan["nsteps"]:
                break
            if (
                controls["probe_macros"] is not None
                and reached >= effective_limit
            ):
                stop_reason = "macro_probe"
                break
            if time.monotonic() - started >= controls["wall_seconds"]:
                stop_reason = "wall_budget"
                break

        for label, experiment in experiments.items():
            if (
                reached > 0
                and reached not in {
                    item["step"] for item in report[label].get("snapshots", [])
                }
            ):
                path = output / "snapshots" / f"{label}-step-{reached:08d}.npz"
                envelope = _envelope(experiment, context)
                digest = _write_snapshot(path, experiment, context, envelope)
                report[label].setdefault("snapshots", []).append(
                    {
                        "step": reached,
                        "path": str(path.relative_to(output)),
                        "sha256": digest,
                    }
                )
            report[label]["accepted_steps"] = int(experiment["state"].step)
            report[label]["time_s"] = float(experiment["state"].time)
            report[label]["ledger_accepted_intervals"] = int(
                experiment["ledger"].accepted_intervals
            )
            report[label]["checks"] = dict(experiment["checks"])

        report["stop_reason"] = stop_reason
        report["budget_stopped"] = bool(
            failure_info is None
            and (stop_reason is not None or reached < plan["nsteps"])
        )
        report["reached_macros"] = reached
        report["reached_horizon_s"] = (
            float(experiments["control"]["state"].time) if experiments else 0.0
        )
        if failure_info is not None:
            report["status"] = "failed"
            report["failure"] = _jsonable(failure_info)
        elif reached == plan["nsteps"] and not report["budget_stopped"]:
            report["status"] = "completed"
        else:
            report["status"] = "budget_stopped"
            if stop_reason is None:
                report["stop_reason"] = "incomplete_macro_limit"
        report["coupling"] = {
            "mode": controls["coupling"],
            "actual_model_evaluations": sum(
                item["actual_model_evaluations"] for item in experiments.values()
            ),
            "iteration_records": sum(
                item["iterations"] for item in experiments.values()
            ),
            "transaction_policy": (
                "one production advance call per ensemble per accepted macro; "
                "both macro endpoints promoted only as a matched pair"
            ),
        }
        report["check_summary"] = {
            label: dict(experiment["checks"])
            for label, experiment in experiments.items()
        }
        report["observe_budget_conclusion"] = (
            "raw power/energy/momentum observe failures are retained; "
            "this harness does not claim those soft budgets pass"
        )
        report["restart"] = {
            "production_restart": False,
            "experimental_continuation": True,
            "resume_from": (
                None
                if controls["resume_from"] is None
                else str(controls["resume_from"])
            ),
            "format": "fomhd-warp-050-accepted-state-v2",
            "reason": (
                "matched accepted pair, exact cumulative ledger and scheduler "
                "identities are available for the bounded scratch continuation "
                "workflow; this is not an official production restart claim"
            ),
        }
    except Exception as error:
        info = {
            "code": getattr(error, "code", type(error).__name__),
            "message": str(error),
            "traceback": __import__("traceback").format_exc(),
            "stage": _jsonable(getattr(error, "stage", None)),
        }
        report["status"] = "failed"
        report["failure"] = info
        report["stop_reason"] = "failure"
        report["budget_stopped"] = False
        report["reached_macros"] = min(
            (
                int(experiment["state"].step)
                for experiment in experiments.values()
            ),
            default=0,
        )
        report["reached_horizon_s"] = (
            float(context.p.dt * report["reached_macros"])
            if context is not None
            else 0.0
        )
        _append_jsonl(failures_path, info)
        if context is not None and getattr(error, "partial_arrays", None):
            _write_failure(
                output / "failures" / "top-level.npz",
                error,
                context,
                label="top-level",
                step=int(
                    experiments.get("control", {})
                    .get("state", SimpleNamespace(step=0))
                    .step
                ),
            )
    finally:
        if context is not None:
            for label, experiment in experiments.items():
                checkpoint = output / "checkpoints" / f"{label}.npz"
                try:
                    report.setdefault(label, {})
                    report[label]["accepted_steps"] = int(experiment["state"].step)
                    report[label]["time_s"] = float(experiment["state"].time)
                    report[label]["ledger_accepted_intervals"] = int(
                        experiment["ledger"].accepted_intervals
                    )
                    digest = _write_state_npz(
                        checkpoint,
                        experiment,
                        context,
                        controls["kind"],
                        run_id=str(run_id),
                        compatibility=compatibility or {},
                        initial_provenance=initial_provenance or {},
                        parent_checkpoints=parent_checkpoints,
                        guard_baseline=experiment["guard_baseline"],
                        stability=stability,
                    )
                    report[label]["checkpoint"] = {
                        "path": str(checkpoint.relative_to(output)),
                        "sha256": digest,
                        "production_restart": False,
                        "schema": "fomhd-warp-050-accepted-state-v2",
                    }
                except Exception as error:
                    _append_jsonl(
                        failures_path,
                        {
                            "checkpoint_error": str(error),
                            "ensemble": label,
                        },
                    )
            inventory = _array_inventory(context, experiments)
            _write_json(output / "memory-inventory.json", inventory)
            samples = []
            if allocator_path.is_file():
                samples = [
                    json.loads(line)
                    for line in allocator_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            report["memory"] = {
                "Nc": int(context.op.Nc),
                "Ne": int(context.op.Ne),
                "Nb": int(context.op.Nb),
                "inventory_file": "memory-inventory.json",
                "array_count": inventory["array_count"],
                "host_resident_bytes": inventory["host_resident_bytes"],
                "device_resident_bytes": inventory["device_resident_bytes"],
                "ru_maxrss_bytes": _ru_maxrss_bytes(),
                "sampled_peak_rss_bytes": _sampled_peak(samples, "rss_bytes"),
                "sampled_peak_device_pool_used_bytes": _sampled_peak(
                    samples, "device_pool_used_bytes"
                ),
                "sampled_peak_device_pool_total_bytes": _sampled_peak(
                    samples, "device_pool_total_bytes"
                ),
                "sampled_peak_device_wide_used_bytes": _sampled_peak(
                    samples, "device_wide_used_bytes"
                ),
                "sampling_scope": (
                    "RSS and allocator samples at setup/accepted macros; "
                    "ru_maxrss is process high-water mark; device-wide values "
                    "include CUDA/Warp/library storage and are not per-process counters"
                ),
            }
        if experiments:
            report["check_summary"] = {
                label: dict(experiment["checks"])
                for label, experiment in experiments.items()
            }
            report["coupling"] = {
                "mode": controls["coupling"],
                "actual_model_evaluations": sum(
                    item["actual_model_evaluations"]
                    for item in experiments.values()
                ),
                "iteration_records": sum(
                    item["iterations"] for item in experiments.values()
                ),
                "transaction_policy": (
                    "one production advance call per ensemble per accepted "
                    "macro; both endpoints promoted only as a matched pair"
                ),
            }
        report["elapsed_s"] = time.monotonic() - started
        _write_json(report_path, report)
        _log(log_path, f"status={report['status']} elapsed_s={report['elapsed_s']:.6f}")
        print(json.dumps(_jsonable(report), indent=2, allow_nan=False), flush=True)
    return 0 if report["status"] == "completed" else 2


def main() -> None:
    args = _parser().parse_args()
    try:
        code = run(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()
