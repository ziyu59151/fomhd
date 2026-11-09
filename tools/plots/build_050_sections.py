#!/usr/bin/env python3
"""Build bounded real-RZ sections from existing 0.5.0 run evidence."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

for thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[thread_variable] = "1"

import numpy as np

sys.dont_write_bytecode = True

SCHEMA = "fomhd-050-real-rz-sections-v1"
MAX_FIGURES = 80
REAL_RTOL = 1e-10
REAL_ATOL = 1e-12
TIME_RTOL = 1e-12
TIME_ATOL = 1e-30
VALID_TERMINAL = {"completed", "budget_stopped"}
SPATIAL_RUN_RE = re.compile(r"^(?:optimized|retry)-east(?:16|24|32|40)-")


class InputError(ValueError):
    """Raised when an input artifact does not support the bounded sections."""


class UnsafeOutputError(ValueError):
    """Raised when the requested output path is not safe."""


class ProvenanceError(ValueError):
    """Raised when exact input or output provenance cannot be verified."""


@dataclass(frozen=True)
class FieldSpec:
    key: str
    title: str
    family: str
    component: int | None
    unit: str
    signed: bool


FIELDS = (
    FieldSpec("B_R", "Radial magnetic field", "magnetic", 0, "T", True),
    FieldSpec("B_phi", "Toroidal magnetic field", "magnetic", 1, "T", True),
    FieldSpec("B_Z", "Vertical magnetic field", "magnetic", 2, "T", True),
    FieldSpec("u_R", "Radial ion velocity", "velocity", 0, "m s$^{-1}$", True),
    FieldSpec("u_phi", "Toroidal ion velocity", "velocity", 1, "m s$^{-1}$", True),
    FieldSpec("u_Z", "Vertical ion velocity", "velocity", 2, "m s$^{-1}$", True),
    FieldSpec("n_i", "Ion density", "density", None, "m$^{-3}$", False),
)
FIELD_BY_KEY = {field.key: field for field in FIELDS}


@dataclass(frozen=True)
class Mapping:
    R: np.ndarray
    Z: np.ndarray
    polygon: np.ndarray
    radial_edges: np.ndarray
    axial_edges: np.ndarray

    @property
    def nc(self) -> int:
        return int(self.R.size)

    @property
    def vertices(self) -> np.ndarray:
        return cell_patch_vertices(
            self.R, self.Z, self.radial_edges, self.axial_edges
        )


@dataclass(frozen=True)
class Snapshot:
    path: Path
    sha256: str
    step: int
    time_s: float
    magnetic: np.ndarray
    velocity: np.ndarray
    density: np.ndarray


@dataclass(frozen=True)
class SnapshotPair:
    control: Snapshot
    perturbed: Snapshot

    @property
    def step(self) -> int:
        return self.control.step

    @property
    def time_s(self) -> float:
        return self.control.time_s


@dataclass
class RunEvidence:
    name: str
    directory: Path
    report: dict[str, Any]
    report_sha256: str
    status: str
    nodes: int | None
    device: str
    mapping: Mapping | None
    nphi: int | None
    pairs: list[SnapshotPair]
    note: str | None = None


@dataclass
class LongMethod:
    name: str
    directories: list[Path]
    mapping: Mapping
    nphi: int
    pairs: list[SnapshotPair]
    status: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise InputError(f"non-finite JSON constant is not allowed: {value}")


def load_json_object(path: Path, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError) as error:
        raise InputError(f"cannot read {where} {path}: {error}") from error
    if not isinstance(value, dict):
        raise InputError(f"{where} must contain a JSON object: {path}")
    return value


class InputIndex:
    """Hash each consumed input exactly once and retain all roles."""

    def __init__(self) -> None:
        self._records: dict[Path, dict[str, Any]] = {}

    def file(self, path: Path, role: str) -> Path:
        resolved = path.resolve()
        if not resolved.is_file():
            raise InputError(f"missing required file: {resolved}")
        record = self._records.get(resolved)
        if record is None:
            record = {
                "path": str(resolved),
                "bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
                "roles": [],
            }
            self._records[resolved] = record
        if role not in record["roles"]:
            record["roles"].append(role)
            record["roles"].sort()
        return resolved

    def json(self, path: Path, role: str) -> dict[str, Any]:
        resolved = self.file(path, role)
        return load_json_object(resolved, role)

    def npz(self, path: Path, role: str) -> dict[str, np.ndarray]:
        resolved = self.file(path, role)
        try:
            with np.load(resolved, allow_pickle=False) as archive:
                return {key: np.asarray(archive[key]) for key in archive.files}
        except (OSError, ValueError) as error:
            raise InputError(f"invalid NPZ {resolved}: {error}") from error

    def records(self) -> list[dict[str, Any]]:
        return [self._records[path].copy() for path in sorted(self._records)]


def _required_array(
    arrays: dict[str, np.ndarray], key: str, path: Path
) -> np.ndarray:
    if key not in arrays:
        raise InputError(f"missing NPZ key {key!r}: {path}")
    return np.asarray(arrays[key])


def axis_boundaries(centers: np.ndarray, axis: str) -> np.ndarray:
    unique = np.unique(np.asarray(centers, dtype=np.float64))
    if unique.ndim != 1 or unique.size < 2:
        raise InputError(f"cannot derive {axis} cell extents from fewer than two centers")
    if not np.isfinite(unique).all():
        raise InputError(f"{axis} centers contain non-finite values")
    differences = np.diff(unique)
    if not np.all(differences > 0.0):
        raise InputError(f"{axis} cell extents are not strictly ordered")
    edges = np.empty(unique.size + 1, dtype=np.float64)
    edges[0] = unique[0] - 0.5 * differences[0]
    edges[-1] = unique[-1] + 0.5 * differences[-1]
    edges[1:-1] = 0.5 * (unique[:-1] + unique[1:])
    if not np.all(np.diff(edges) > 0.0):
        raise InputError(f"{axis} midpoint cell boundaries are not ordered")
    return edges


def _cell_half_extents(
    centers: np.ndarray, edges: np.ndarray, axis: str
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(centers)
    indices = np.searchsorted(unique, centers)
    if np.any(indices >= unique.size) or not np.all(
        np.isclose(unique[np.minimum(indices, unique.size - 1)], centers, rtol=0.0, atol=1e-12)
    ):
        raise InputError(f"{axis} centers do not belong to their derived unique grid")
    return edges[indices], edges[indices + 1]


def cell_patch_vertices(
    R: np.ndarray,
    Z: np.ndarray,
    radial_edges: np.ndarray,
    axial_edges: np.ndarray,
) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    if R.ndim != 1 or Z.ndim != 1 or R.shape != Z.shape or R.size == 0:
        raise InputError("active-cell R and Z must be equal nonempty vectors")
    if not np.isfinite(R).all() or not np.isfinite(Z).all():
        raise InputError("active-cell centers contain non-finite coordinates")
    r0, r1 = _cell_half_extents(R, np.asarray(radial_edges), "R")
    z0, z1 = _cell_half_extents(Z, np.asarray(axial_edges), "Z")
    vertices = np.empty((R.size, 4, 2), dtype=np.float64)
    vertices[:, 0, 0] = r0
    vertices[:, 0, 1] = z0
    vertices[:, 1, 0] = r1
    vertices[:, 1, 1] = z0
    vertices[:, 2, 0] = r1
    vertices[:, 2, 1] = z1
    vertices[:, 3, 0] = r0
    vertices[:, 3, 1] = z1
    return vertices


def load_mapping(path: Path, inputs: InputIndex, role: str) -> Mapping:
    arrays = inputs.npz(path, role)
    R = np.asarray(_required_array(arrays, "R", path), dtype=np.float64)
    Z = np.asarray(_required_array(arrays, "Z", path), dtype=np.float64)
    polygon = np.asarray(_required_array(arrays, "polygon", path), dtype=np.float64)
    if R.ndim != 1 or Z.ndim != 1 or R.shape != Z.shape or R.size == 0:
        raise InputError(f"R and Z must be equal nonempty vectors: {path}")
    if not np.isfinite(R).all() or not np.isfinite(Z).all():
        raise InputError(f"R or Z contains non-finite values: {path}")
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise InputError(f"polygon must have shape (at least 3, 2): {path}")
    if not np.isfinite(polygon).all():
        raise InputError(f"polygon contains non-finite values: {path}")
    closed = np.vstack([polygon, polygon[0]])
    area = 0.5 * float(
        np.sum(
            closed[:-1, 0] * closed[1:, 1]
            - closed[1:, 0] * closed[:-1, 1]
        )
    )
    if not math.isfinite(area) or abs(area) <= np.finfo(np.float64).tiny:
        raise InputError(f"polygon has zero or non-finite area: {path}")
    radial_edges = axis_boundaries(R, "R")
    axial_edges = axis_boundaries(Z, "Z")
    cell_patch_vertices(R, Z, radial_edges, axial_edges)
    return Mapping(R, Z, polygon, radial_edges, axial_edges)


def _physical_real(
    values: np.ndarray, name: str, path: Path, *, rtol: float = REAL_RTOL, atol: float = REAL_ATOL
) -> np.ndarray:
    values = np.asarray(values)
    if values.size == 0:
        raise InputError(f"{name} has no samples at physical phi=0: {path}")
    if not np.isfinite(values.real).all():
        raise InputError(f"{name} contains non-finite values at physical phi=0: {path}")
    if np.iscomplexobj(values):
        if not np.isfinite(values.imag).all():
            raise InputError(f"{name} contains non-finite imaginary values: {path}")
        imaginary = float(np.max(np.abs(values.imag), initial=0.0))
        scale = float(np.max(np.abs(values.real), initial=0.0))
        tolerance = atol + rtol * scale
        if imaginary > tolerance:
            raise InputError(
                f"{name} is not physical-real at phi=0: max|imag|={imaginary:.6e} "
                f"> tolerance={tolerance:.6e}: {path}"
            )
    return np.asarray(values.real, dtype=np.float64).copy()


def decode_vector_samples(
    samples: np.ndarray,
    name: str,
    nphi: int,
    nc: int,
    path: Path,
    *,
    rtol: float = REAL_RTOL,
    atol: float = REAL_ATOL,
) -> np.ndarray:
    expected = nphi * 3 * nc
    values = np.asarray(samples)
    if values.ndim != 1 or values.size != expected:
        raise InputError(
            f"{name} shape mismatch: got {values.shape}, expected ({expected},) "
            f"for Nphi={nphi}, 3 components, Nc={nc}: {path}"
        )
    if not np.isfinite(values.real).all() or (
        np.iscomplexobj(values) and not np.isfinite(values.imag).all()
    ):
        raise InputError(f"{name} contains non-finite samples: {path}")
    return _physical_real(
        values.reshape(nphi, 3, nc)[0], name, path, rtol=rtol, atol=atol
    )


def decode_scalar_samples(
    samples: np.ndarray,
    name: str,
    nphi: int,
    nc: int,
    path: Path,
    *,
    rtol: float = REAL_RTOL,
    atol: float = REAL_ATOL,
) -> np.ndarray:
    expected = nphi * nc
    values = np.asarray(samples)
    if values.ndim != 1 or values.size != expected:
        raise InputError(
            f"{name} shape mismatch: got {values.shape}, expected ({expected},) "
            f"for Nphi={nphi}, Nc={nc}: {path}"
        )
    if not np.isfinite(values.real).all() or (
        np.iscomplexobj(values) and not np.isfinite(values.imag).all()
    ):
        raise InputError(f"{name} contains non-finite samples: {path}")
    return _physical_real(
        values.reshape(nphi, nc)[0], name, path, rtol=rtol, atol=atol
    )


def _scalar_int(arrays: dict[str, np.ndarray], key: str, path: Path) -> int:
    value = _required_array(arrays, key, path)
    if value.shape != ():
        raise InputError(f"{key} must be scalar: {path}")
    try:
        result = int(value.item())
    except (TypeError, ValueError, OverflowError) as error:
        raise InputError(f"{key} must be an integer: {path}") from error
    if result < 0 or result != float(value.item()):
        raise InputError(f"{key} must be a nonnegative integer: {path}")
    return result


def _scalar_float(arrays: dict[str, np.ndarray], key: str, path: Path) -> float:
    value = _required_array(arrays, key, path)
    if value.shape != ():
        raise InputError(f"{key} must be scalar: {path}")
    try:
        result = float(value.item())
    except (TypeError, ValueError, OverflowError) as error:
        raise InputError(f"{key} must be a finite number: {path}") from error
    if not math.isfinite(result) or result < 0.0:
        raise InputError(f"{key} must be finite and nonnegative: {path}")
    return result


def read_snapshot(
    path: Path, mapping: Mapping, nphi: int, inputs: InputIndex
) -> Snapshot:
    arrays = inputs.npz(path, "matched physical-RZ snapshot")
    for key in (
        "step",
        "time_s",
        "observed_BN",
        "observed_density",
        "observed_velocity",
    ):
        _required_array(arrays, key, path)
    step = _scalar_int(arrays, "step", path)
    time_s = _scalar_float(arrays, "time_s", path)
    magnetic = decode_vector_samples(
        arrays["observed_BN"], "observed_BN", nphi, mapping.nc, path
    )
    velocity = decode_vector_samples(
        arrays["observed_velocity"], "observed_velocity", nphi, mapping.nc, path
    )
    density = decode_scalar_samples(
        arrays["observed_density"], "observed_density", nphi, mapping.nc, path
    )
    return Snapshot(
        path=path.resolve(),
        sha256=sha256_file(path),
        step=step,
        time_s=time_s,
        magnetic=magnetic,
        velocity=velocity,
        density=density,
    )


def _read_ensemble(
    run: Path, ensemble: str, mapping: Mapping, nphi: int, inputs: InputIndex
) -> dict[int, Snapshot]:
    directory = run / "snapshots"
    pattern = re.compile(rf"^{re.escape(ensemble)}-step-(\d+)\.npz$")
    snapshots: dict[int, Snapshot] = {}
    paths = sorted(
        path
        for path in directory.glob(f"{ensemble}-step-*.npz")
        if pattern.fullmatch(path.name)
    )
    if not paths:
        raise InputError(f"no {ensemble} snapshots found in {directory}")
    for path in paths:
        filename_step = int(pattern.fullmatch(path.name).group(1))
        snapshot = read_snapshot(path, mapping, nphi, inputs)
        if snapshot.step != filename_step:
            raise InputError(
                f"snapshot filename/state step mismatch: {path.name} stores {snapshot.step}"
            )
        if snapshot.step in snapshots:
            raise InputError(f"duplicate {ensemble} step {snapshot.step}: {run}")
        snapshots[snapshot.step] = snapshot
    return snapshots


def match_snapshots(
    control: dict[int, Snapshot],
    perturbed: dict[int, Snapshot],
    *,
    rel_tol: float = TIME_RTOL,
    abs_tol: float = TIME_ATOL,
) -> list[SnapshotPair]:
    control_steps = set(control)
    perturbed_steps = set(perturbed)
    if control_steps != perturbed_steps:
        missing_perturbed = sorted(control_steps - perturbed_steps)
        missing_control = sorted(perturbed_steps - control_steps)
        raise InputError(
            "control/perturbed cadence mismatch: "
            f"control-only={missing_perturbed}, perturbed-only={missing_control}"
        )
    pairs: list[SnapshotPair] = []
    for step in sorted(control_steps):
        left, right = control[step], perturbed[step]
        if not math.isclose(
            left.time_s, right.time_s, rel_tol=rel_tol, abs_tol=abs_tol
        ):
            raise InputError(
                f"control/perturbed time mismatch at step {step}: "
                f"{left.time_s:.17g} != {right.time_s:.17g}"
            )
        pairs.append(SnapshotPair(left, right))
    return pairs


def read_matched_pairs(
    run: Path, mapping: Mapping, nphi: int, inputs: InputIndex
) -> list[SnapshotPair]:
    return match_snapshots(
        _read_ensemble(run, "control", mapping, nphi, inputs),
        _read_ensemble(run, "perturbed", mapping, nphi, inputs),
    )


def require_compatible_grid(
    reference: Mapping,
    candidate: Mapping,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-14,
) -> None:
    if reference.R.shape != candidate.R.shape:
        raise InputError(
            f"active-grid shape mismatch: {reference.R.shape} != {candidate.R.shape}"
        )
    for name, left, right in (
        ("R", reference.R, candidate.R),
        ("Z", reference.Z, candidate.Z),
        ("polygon", reference.polygon, candidate.polygon),
    ):
        if left.shape != right.shape or not np.allclose(
            left, right, rtol=rtol, atol=atol, equal_nan=False
        ):
            raise InputError(f"active-grid {name} mismatch")


def snapshot_arrays(snapshot: Snapshot) -> dict[str, np.ndarray]:
    return {
        "B_R": snapshot.magnetic[0],
        "B_phi": snapshot.magnetic[1],
        "B_Z": snapshot.magnetic[2],
        "u_R": snapshot.velocity[0],
        "u_phi": snapshot.velocity[1],
        "u_Z": snapshot.velocity[2],
        "n_i": snapshot.density,
    }


def field_values(snapshot: Snapshot, field: FieldSpec) -> np.ndarray:
    return snapshot_arrays(snapshot)[field.key]


def field_delta(pair: SnapshotPair, field: FieldSpec) -> np.ndarray:
    return field_values(pair.perturbed, field) - field_values(pair.control, field)


def symmetric_limit(*arrays: np.ndarray) -> float:
    maximum = 0.0
    for array in arrays:
        values = np.asarray(array, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            maximum = max(maximum, float(np.max(np.abs(finite))))
    if maximum == 0.0:
        return float(np.finfo(np.float64).tiny)
    return maximum


def symmetric_limits(*arrays: np.ndarray) -> tuple[float, float]:
    limit = symmetric_limit(*arrays)
    return -limit, limit


def sequential_limits(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise InputError("cannot normalize an array with no finite values")
    low, high = float(np.min(finite)), float(np.max(finite))
    if low == high:
        padding = max(abs(low) * 1e-12, np.finfo(np.float64).eps)
        low -= padding
        high += padding
    return low, high


def _safe_output(run_root: Path, source_root: Path, output: Path) -> Path:
    run_root = run_root.resolve()
    source_root = source_root.resolve()
    output = output.resolve()
    if output.exists():
        raise UnsafeOutputError(f"refusing existing output path: {output}")
    for label, root in (("run", run_root), ("source", source_root)):
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise UnsafeOutputError(
                f"output must be outside the {label} root and its ancestors: {output}"
            )
    if not run_root.is_dir():
        raise InputError(f"run root is not a directory: {run_root}")
    if not source_root.is_dir():
        raise InputError(f"source root is not a directory: {source_root}")
    return output


def _node_count(report: dict[str, Any]) -> int | None:
    case = report.get("case")
    if isinstance(case, dict) and isinstance(case.get("nodes"), int):
        return case["nodes"]
    seed = report.get("seed")
    if isinstance(seed, dict) and isinstance(seed.get("nodes"), int):
        return seed["nodes"]
    return None


def _nphi(report: dict[str, Any]) -> int:
    actual = report.get("actual_grid")
    if isinstance(actual, dict) and isinstance(actual.get("Nphi"), int):
        value = actual["Nphi"]
    else:
        model = report.get("model")
        value = model.get("Nphi") if isinstance(model, dict) else None
    if not isinstance(value, int) or value < 1:
        raise InputError("report does not provide a positive actual_grid.Nphi/model.Nphi")
    return value


def _nc(report: dict[str, Any]) -> int | None:
    actual = report.get("actual_grid")
    value = actual.get("Nc") if isinstance(actual, dict) else None
    return value if isinstance(value, int) and value > 0 else None


def _device(report: dict[str, Any]) -> str:
    case = report.get("case")
    if isinstance(case, dict) and isinstance(case.get("device"), str):
        return case["device"]
    return "not recorded"


def classify_spatial_runs(
    run_root: Path, inputs: InputIndex
) -> list[RunEvidence]:
    candidates = sorted(
        path
        for path in run_root.iterdir()
        if path.is_dir() and SPATIAL_RUN_RE.match(path.name)
    )
    result: list[RunEvidence] = []
    for run in candidates:
        report_path = run / "report.json"
        if not report_path.is_file():
            result.append(
                RunEvidence(
                    run.name,
                    run,
                    {},
                    "",
                    "missing-report",
                    None,
                    "not recorded",
                    None,
                    None,
                    [],
                    f"No-data: {run.name} has no report.json.",
                )
            )
            continue
        report = inputs.json(report_path, "0.5.0 spatial-probe report")
        report_hash = sha256_file(report_path)
        status = str(report.get("status", "unrecorded"))
        nodes = _node_count(report)
        device = _device(report)
        if status != "completed":
            failure = report.get("failure")
            if isinstance(failure, dict):
                code = str(failure.get("code", "unrecorded"))
                message = str(failure.get("message", "no message recorded"))
            else:
                code = "unrecorded"
                message = str(report.get("stop_reason", "no failure detail"))
            result.append(
                RunEvidence(
                    run.name,
                    run,
                    report,
                    report_hash,
                    status,
                    nodes,
                    device,
                    None,
                    None,
                    [],
                    f"No-data: {run.name} status={status}; failure={code}: {message}. "
                    "No numeric evolution sections were inferred.",
                )
            )
            continue
        mapping = load_mapping(
            run / "input_mapping.npz", inputs, f"{run.name} active-cell mapping"
        )
        nphi = _nphi(report)
        recorded_nc = _nc(report)
        if recorded_nc is not None and recorded_nc != mapping.nc:
            raise InputError(
                f"{run.name} actual_grid.Nc={recorded_nc} but mapping stores {mapping.nc}"
            )
        pairs = read_matched_pairs(run, mapping, nphi, inputs)
        result.append(
            RunEvidence(
                run.name,
                run,
                report,
                report_hash,
                status,
                nodes,
                device,
                mapping,
                nphi,
                pairs,
            )
        )
    return result


def _report_final(report: dict[str, Any], label: str) -> tuple[int, float]:
    section = report.get(label)
    if not isinstance(section, dict):
        raise InputError(f"report lacks {label} continuation endpoint")
    step = section.get("accepted_steps")
    time_s = section.get("time_s")
    if isinstance(step, bool) or not isinstance(step, int):
        raise InputError(f"report {label}.accepted_steps must be an integer")
    if isinstance(time_s, bool) or not isinstance(time_s, (int, float)):
        raise InputError(f"report {label}.time_s must be numeric")
    return step, float(time_s)


def verify_continuation(
    current: Path,
    parent: Path,
    current_report: dict[str, Any],
    parent_report: dict[str, Any],
    run_root: Path,
    inputs: InputIndex,
) -> None:
    continuation = current_report.get("continuation")
    if not isinstance(continuation, dict):
        raise InputError(f"{current.name} does not record a continuation object")
    resume = continuation.get("resume_from")
    if not isinstance(resume, str):
        raise InputError(f"{current.name} continuation.resume_from is missing")
    recorded_parent = Path(resume).resolve()
    expected_parent = parent.resolve()
    if recorded_parent != expected_parent:
        raise InputError(
            f"{current.name} recorded parent {recorded_parent} != {expected_parent}"
        )
    if not recorded_parent.is_relative_to(run_root.resolve()):
        raise InputError(f"{current.name} parent escapes run root: {recorded_parent}")
    start_step = continuation.get("start_step")
    start_time = continuation.get("start_time_s")
    if isinstance(start_step, bool) or not isinstance(start_step, int):
        raise InputError(f"{current.name} continuation.start_step is invalid")
    if isinstance(start_time, bool) or not isinstance(start_time, (int, float)):
        raise InputError(f"{current.name} continuation.start_time_s is invalid")
    for label in ("control", "perturbed"):
        parent_step, parent_time = _report_final(parent_report, label)
        if start_step != parent_step or not math.isclose(
            float(start_time),
            parent_time,
            rel_tol=TIME_RTOL,
            abs_tol=TIME_ATOL,
        ):
            raise InputError(
                f"{current.name} start {start_step}/{float(start_time):.17g} "
                f"does not equal parent {label} {parent_step}/{parent_time:.17g}"
            )
    checkpoints = continuation.get("parent_checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 2:
        raise InputError(f"{current.name} continuation.parent_checkpoints is invalid")
    labels = set()
    for record in checkpoints:
        if not isinstance(record, dict):
            raise InputError(f"{current.name} parent checkpoint entry is invalid")
        label = record.get("label")
        path_text = record.get("path")
        sha256 = record.get("sha256")
        if label not in {"control", "perturbed"} or label in labels:
            raise InputError(f"{current.name} parent checkpoint label is invalid")
        labels.add(label)
        expected = (parent / "checkpoints" / f"{label}.npz").resolve()
        if not isinstance(path_text, str) or Path(path_text).resolve() != expected:
            raise InputError(f"{current.name} parent {label} checkpoint path mismatch")
        if not isinstance(sha256, str) or sha256_file(expected) != sha256:
            raise InputError(f"{current.name} parent {label} checkpoint SHA256 mismatch")
        inputs.file(expected, f"{current.name} verified continuation parent checkpoint")
    if labels != {"control", "perturbed"}:
        raise InputError(f"{current.name} does not record both parent checkpoints")
    if current_report.get("actual_grid") != parent_report.get("actual_grid"):
        raise InputError(f"{current.name} actual_grid differs from its recorded parent")
    current_mapping = load_mapping(
        current / "input_mapping.npz", inputs, f"{current.name} continuation mapping"
    )
    parent_mapping = load_mapping(
        parent / "input_mapping.npz", inputs, f"{parent.name} origin mapping"
    )
    require_compatible_grid(parent_mapping, current_mapping)


def discover_continuation_chain(
    run_root: Path, root_name: str, inputs: InputIndex
) -> list[Path]:
    root = (run_root / root_name).resolve()
    if not root.is_dir():
        raise InputError(f"missing long-chain root: {root}")
    chain = [root]
    seen = {root}
    while True:
        parent = chain[-1]
        parent_report = inputs.json(
            parent / "report.json", f"{parent.name} chain report"
        )
        candidates: list[Path] = []
        prefix = root_name.rsplit("-", 1)[0]
        for child in sorted(run_root.glob(f"{prefix}-*")):
            if not child.is_dir() or child.resolve() in seen:
                continue
            report_path = child / "report.json"
            if not report_path.is_file():
                continue
            report = inputs.json(report_path, f"{child.name} candidate chain report")
            continuation = report.get("continuation")
            if not isinstance(continuation, dict):
                continue
            resume = continuation.get("resume_from")
            if not isinstance(resume, str) or Path(resume).resolve() != parent:
                continue
            if report.get("status") not in VALID_TERMINAL:
                continue
            verify_continuation(
                child, parent, report, parent_report, run_root, inputs
            )
            candidates.append(child.resolve())
        candidates = sorted(set(candidates))
        if not candidates:
            return chain
        if len(candidates) != 1:
            names = ", ".join(path.name for path in candidates)
            raise InputError(
                f"ambiguous valid continuation children after {parent.name}: {names}"
            )
        chain.append(candidates[0])
        seen.add(candidates[0])
        if len(chain) > 20:
            raise InputError("continuation chain exceeds the bounded depth")


def collect_chain_pairs(
    directories: list[Path],
    nphi: int,
    inputs: InputIndex,
) -> tuple[Mapping, list[SnapshotPair]]:
    origin_mapping = load_mapping(
        directories[0] / "input_mapping.npz", inputs, "long-chain origin mapping"
    )
    by_time: list[SnapshotPair] = []
    for directory in directories:
        report = inputs.json(directory / "report.json", f"{directory.name} chain report")
        mapping = load_mapping(
            directory / "input_mapping.npz", inputs, f"{directory.name} chain mapping"
        )
        require_compatible_grid(origin_mapping, mapping)
        recorded_nphi = _nphi(report)
        if recorded_nphi != nphi:
            raise InputError(
                f"{directory.name} Nphi={recorded_nphi} differs from long chain Nphi={nphi}"
            )
        by_time.extend(read_matched_pairs(directory, mapping, nphi, inputs))
    by_time.sort(key=lambda pair: pair.time_s)
    unique: list[SnapshotPair] = []
    for pair in by_time:
        if unique and math.isclose(
            unique[-1].time_s, pair.time_s, rel_tol=TIME_RTOL, abs_tol=TIME_ATOL
        ):
            if (
                unique[-1].control.sha256 != pair.control.sha256
                or unique[-1].perturbed.sha256 != pair.perturbed.sha256
            ):
                raise InputError(
                    f"conflicting long-chain snapshots at time {pair.time_s:.17g}"
                )
            continue
        unique.append(pair)
    return origin_mapping, unique


def _long_directories(
    run_root: Path,
    *,
    root_name: str,
    with_continuation: bool,
    inputs: InputIndex,
) -> list[Path]:
    return (
        discover_continuation_chain(run_root, root_name, inputs)
        if with_continuation
        else [(run_root / root_name).resolve()]
    )


def load_long_methods(run_root: Path, inputs: InputIndex) -> list[LongMethod]:
    t1_directories = discover_continuation_chain(
        run_root, "long-east12-waveform-probe", inputs
    )
    t1_report = inputs.json(
        t1_directories[0] / "report.json", "T1 origin report"
    )
    nphi = _nphi(t1_report)
    t1_mapping, t1_pairs = collect_chain_pairs(t1_directories, nphi, inputs)
    a0_directories = _long_directories(
        run_root,
        root_name="long-east12-synchronous",
        with_continuation=False,
        inputs=inputs,
    )
    half_directories = discover_continuation_chain(
        run_root, "long-east12-synchronous-halfh", inputs
    )
    methods = [
        LongMethod("T1 waveform", t1_directories, t1_mapping, nphi, t1_pairs, "budget_stopped")
    ]
    for name, directories, status in (
        ("A0 same h", a0_directories, "budget_stopped"),
        ("A0 half h", half_directories, "budget_stopped"),
    ):
        mapping, pairs = collect_chain_pairs(directories, nphi, inputs)
        require_compatible_grid(t1_mapping, mapping)
        methods.append(LongMethod(name, directories, mapping, nphi, pairs, status))
    return methods


def find_pair_at_time(
    pairs: list[SnapshotPair], time_s: float
) -> SnapshotPair | None:
    for pair in pairs:
        if math.isclose(
            pair.time_s, time_s, rel_tol=TIME_RTOL, abs_tol=TIME_ATOL
        ):
            return pair
    return None


def common_long_pairs(
    methods: list[LongMethod],
) -> list[tuple[float, list[SnapshotPair]]]:
    reference = methods[0]
    common: list[tuple[float, list[SnapshotPair]]] = []
    for pair in reference.pairs:
        matched = [pair]
        for method in methods[1:]:
            candidate = find_pair_at_time(method.pairs, pair.time_s)
            if candidate is None:
                break
            matched.append(candidate)
        if len(matched) == len(methods):
            common.append((pair.time_s, matched))
    if len(common) < 3:
        raise InputError(
            f"long chain has only {len(common)} saved common times; at least three are required"
        )
    return common


def select_early_middle_final(
    common: list[tuple[float, list[SnapshotPair]]]
) -> list[tuple[str, float, list[SnapshotPair]]]:
    indices = sorted({0, len(common) // 2, len(common) - 1})
    labels = ("early", "middle", "final")
    return [
        (label, common[index][0], common[index][1])
        for label, index in zip(labels, indices, strict=True)
    ]


def load_canonical(source_root: Path) -> tuple[Any, Any, dict[str, Path]]:
    paths = {
        "a1_t2": source_root / "tools/plots/build_a1_t2_report.py",
        "a0": source_root / "tools/plots/build_a0_report.py",
        "a1_t0": source_root / "tools/plots/build_a1_t0_report.py",
    }
    for path in paths.values():
        if not path.is_file():
            raise InputError(f"missing canonical source: {path}")
    spec = importlib.util.spec_from_file_location("sections_a1_t2", paths["a1_t2"])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {paths['a1_t2']}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    a0, t0 = module.load_sources(source_root)
    return a0, t0, paths


def _plot_cells(
    axis: Any,
    mapping: Mapping,
    values: np.ndarray,
    *,
    norm: Any,
    cmap: str,
    title: str,
    colorbar_label: str,
    t0: Any,
) -> Any:
    from matplotlib.collections import PolyCollection
    from matplotlib.ticker import ScalarFormatter

    collection = PolyCollection(
        mapping.vertices,
        array=np.ma.masked_invalid(np.asarray(values, dtype=np.float64)),
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        linewidths=0.0,
        antialiaseds=False,
    )
    axis.add_collection(collection)
    closed = np.vstack([mapping.polygon, mapping.polygon[0]])
    axis.plot(
        closed[:, 0],
        closed[:, 1],
        color="#202020",
        linewidth=1.15,
        label="Wall polygon",
        zorder=4,
    )
    axis.set_xlim(float(np.min(closed[:, 0])), float(np.max(closed[:, 0])))
    axis.set_ylim(float(np.min(closed[:, 1])), float(np.max(closed[:, 1])))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"$R$ (m)")
    axis.set_ylabel(r"$Z$ (m)")
    axis.set_title(title)
    axis.legend(loc="lower left", frameon=True, framealpha=0.85)
    t0.style_axis(axis)
    colorbar = axis.figure.colorbar(collection, ax=axis, fraction=0.046, pad=0.035)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 4))
    colorbar.formatter = formatter
    colorbar.update_ticks()
    colorbar.set_label(colorbar_label)
    return collection


def _save_figure(figure: Any, figures_dir: Path, name: str, t0: Any, plt: Any) -> None:
    figure.tight_layout()
    t0.save_figure(figure, figures_dir, name)
    plt.close(figure)


def _figure_record(
    number: int,
    *,
    title: str,
    caption: str,
    image: str,
    svg: str,
    panels: list[tuple[str, str]],
    status: str,
    device: str,
    components: list[str],
    grid: int,
    nphi: int,
    n_phi_quadrature: int,
    question: str,
    controls: str,
    criterion: str,
    source_kind: str = "nonlinear_evolution",
) -> dict[str, Any]:
    return {
        "number": number,
        "title": title,
        "caption": caption,
        "source_kind": source_kind,
        "components": components,
        "status": status,
        "device": device,
        "question": question,
        "controls": controls,
        "criterion": criterion,
        "grid": {"n_r": grid, "n_z": grid},
        "n_phi": 1,
        "k": 1,
        "n_phi_quadrature": n_phi_quadrature,
        "image": image,
        "svg": svg,
        "data": "data.json",
        "provenance": "provenance.json",
        "panels": [
            {"letter": letter, "description": description}
            for letter, description in panels
        ],
    }


def _plot_control_delta(
    *,
    pair: SnapshotPair,
    mapping: Mapping,
    field: FieldSpec,
    delta_limits: tuple[float, float],
    output: Path,
    name: str,
    t0: Any,
    plt: Any,
) -> list[tuple[str, str]]:
    from matplotlib.colors import Normalize, TwoSlopeNorm

    control = field_values(pair.control, field)
    delta = field_delta(pair, field)
    low, high = sequential_limits(control)
    dlow, dhigh = delta_limits
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    _plot_cells(
        axes[0],
        mapping,
        control,
        norm=Normalize(vmin=low, vmax=high),
        cmap="cividis",
        title=f"Control: {field.title}",
        colorbar_label=field.unit,
        t0=t0,
    )
    _plot_cells(
        axes[1],
        mapping,
        delta,
        norm=TwoSlopeNorm(vmin=dlow, vcenter=0.0, vmax=dhigh),
        cmap="RdBu_r",
        title=f"Delta: {field.title}",
        colorbar_label=field.unit,
        t0=t0,
    )
    t0.panel_label(axes[0], "a")
    t0.panel_label(axes[1], "b")
    _save_figure(figure, output / "figures", name, t0, plt)
    time_text = f"{pair.time_s:.9e} s"
    return [
        ("a", f"Control {field.key} at matched step {pair.step}, t={time_text}."),
        (
            "b",
            f"Perturbed minus control {field.key} at the same matched step and time; "
            f"centered shared limits [{dlow:.6e}, {dhigh:.6e}].",
        ),
    ]


def plot_final_fields(
    *,
    pair: SnapshotPair,
    mapping: Mapping,
    fields: tuple[FieldSpec, ...],
    delta_limits: dict[str, tuple[float, float]],
    output: Path,
    prefix: str,
    t0: Any,
    plt: Any,
) -> list[tuple[str, list[tuple[str, str]], list[str]]]:
    from matplotlib.colors import Normalize, TwoSlopeNorm

    figure, axes = plt.subplots(len(fields), 2, figsize=(10.4, 3.75 * len(fields)))
    axes = np.atleast_2d(axes)
    panel_records: list[tuple[str, str]] = []
    letters = "abcdefgh"
    for row, field in enumerate(fields):
        control = field_values(pair.control, field)
        delta = field_delta(pair, field)
        control_low, control_high = sequential_limits(control)
        dlow, dhigh = delta_limits[field.key]
        _plot_cells(
            axes[row, 0],
            mapping,
            control,
            norm=Normalize(vmin=control_low, vmax=control_high),
            cmap="cividis",
            title=f"Control: {field.title}",
            colorbar_label=field.unit,
            t0=t0,
        )
        _plot_cells(
            axes[row, 1],
            mapping,
            delta,
            norm=TwoSlopeNorm(vmin=dlow, vcenter=0.0, vmax=dhigh),
            cmap="RdBu_r",
            title=f"Delta: {field.title}",
            colorbar_label=field.unit,
            t0=t0,
        )
        left_letter, right_letter = letters[2 * row : 2 * row + 2]
        t0.panel_label(axes[row, 0], left_letter)
        t0.panel_label(axes[row, 1], right_letter)
        panel_records.extend(
            (
                (
                    left_letter,
                    f"Control {field.key} at matched step {pair.step}, "
                    f"t={pair.time_s:.9e} s.",
                ),
                (
                    right_letter,
                    f"Perturbed minus control {field.key}; centered limits "
                    f"[{dlow:.6e}, {dhigh:.6e}].",
                ),
            )
        )
    _save_figure(figure, output / "figures", prefix, t0, plt)
    return [(prefix, panel_records, [field.key for field in fields])]


def plot_delta_times(
    *,
    first: SnapshotPair,
    last: SnapshotPair,
    mapping: Mapping,
    fields: tuple[FieldSpec, ...],
    output: Path,
    prefix: str,
    t0: Any,
    plt: Any,
) -> list[tuple[str, list[tuple[str, str]], list[str]]]:
    from matplotlib.colors import TwoSlopeNorm

    figure, axes = plt.subplots(len(fields), 2, figsize=(10.4, 3.75 * len(fields)))
    axes = np.atleast_2d(axes)
    panel_records: list[tuple[str, str]] = []
    letters = "abcdefgh"
    for row, field in enumerate(fields):
        first_delta = field_delta(first, field)
        last_delta = field_delta(last, field)
        limit = symmetric_limit(first_delta, last_delta)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        _plot_cells(
            axes[row, 0],
            mapping,
            first_delta,
            norm=norm,
            cmap="RdBu_r",
            title=f"First: d{field.key}",
            colorbar_label=field.unit,
            t0=t0,
        )
        _plot_cells(
            axes[row, 1],
            mapping,
            last_delta,
            norm=norm,
            cmap="RdBu_r",
            title=f"Final: d{field.key}",
            colorbar_label=field.unit,
            t0=t0,
        )
        left_letter, right_letter = letters[2 * row : 2 * row + 2]
        t0.panel_label(axes[row, 0], left_letter)
        t0.panel_label(axes[row, 1], right_letter)
        panel_records.extend(
            (
                (
                    left_letter,
                    f"First matched saved delta {field.key}, "
                    f"t={first.time_s:.9e} s, centered limit +/-{limit:.6e}.",
                ),
                (
                    right_letter,
                    f"Final matched saved delta {field.key}, "
                    f"t={last.time_s:.9e} s, same centered limit.",
                ),
            )
        )
    _save_figure(figure, output / "figures", prefix, t0, plt)
    return [(prefix, panel_records, [field.key for field in fields])]


def plot_long_common(
    *,
    label: str,
    time_s: float,
    pairs: list[SnapshotPair],
    mapping: Mapping,
    field: FieldSpec,
    actual_limits: tuple[float, float],
    difference_limits: tuple[float, float],
    output: Path,
    prefix: str,
    t0: Any,
    plt: Any,
) -> list[tuple[str, list[tuple[str, str]], list[str]]]:
    from matplotlib.colors import Normalize, TwoSlopeNorm

    t1, a0, halfh = (pair.control for pair in pairs)
    actual = [field_values(t1, field), field_values(a0, field), field_values(halfh, field)]
    differences = [
        field_values(t1, field) - field_values(a0, field),
        field_values(t1, field) - field_values(halfh, field),
    ]
    actual_low, actual_high = actual_limits
    diff_low, diff_high = difference_limits

    actual_figure, actual_axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    for axis, snapshot, title, letter in zip(
        actual_axes,
        (t1, a0, halfh),
        ("T1 waveform", "A0 same h", "A0 half h"),
        "abc",
        strict=True,
    ):
        _plot_cells(
            axis,
            mapping,
            field_values(snapshot, field),
            norm=Normalize(vmin=actual_low, vmax=actual_high),
            cmap="cividis",
            title=f"{title}: {field.key}",
            colorbar_label=field.unit,
            t0=t0,
        )
        t0.panel_label(axis, letter)
    actual_name = f"{prefix}_actual"
    _save_figure(actual_figure, output / "figures", actual_name, t0, plt)

    difference_figure, difference_axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    for axis, values, title, letter in zip(
        difference_axes,
        differences,
        ("T1 minus A0 same h", "T1 minus A0 half h"),
        "ab",
        strict=True,
    ):
        _plot_cells(
            axis,
            mapping,
            values,
            norm=TwoSlopeNorm(vmin=diff_low, vcenter=0.0, vmax=diff_high),
            cmap="RdBu_r",
            title=title,
            colorbar_label=field.unit,
            t0=t0,
        )
        t0.panel_label(axis, letter)
    difference_name = f"{prefix}_difference"
    _save_figure(difference_figure, output / "figures", difference_name, t0, plt)

    time_text = f"{time_s:.9e} s"
    return [
        (
            actual_name,
            [
                ("a", f"T1 waveform control {field.key} at {label} common time {time_text}."),
                ("b", f"A0 same-h control {field.key} at the same saved time."),
                ("c", f"A0 half-h control {field.key} at the same saved time."),
            ],
            [field.key],
        ),
        (
            difference_name,
            [
                (
                    "a",
                    f"T1 minus A0 same-h {field.key}; shared centered limits "
                    f"[{diff_low:.6e}, {diff_high:.6e}].",
                ),
                (
                    "b",
                    f"T1 minus A0 half-h {field.key}; same limits. Raw FP64 "
                    "cancellation limits the interpretation.",
                ),
            ],
            [field.key],
        ),
    ]


def _field_capacity_notes(
    pairs: list[SnapshotPair],
) -> dict[str, dict[str, float]]:
    first, last = pairs[0], pairs[-1]
    epsilon = float(np.finfo(np.float64).eps)
    result: dict[str, dict[str, float]] = {}
    for field in FIELDS:
        control_last = field_values(last.control, field)
        perturbed_last = field_values(last.perturbed, field)
        delta_last = perturbed_last - control_last
        background = float(
            max(
                np.max(np.abs(control_last), initial=0.0),
                np.max(np.abs(perturbed_last), initial=0.0),
            )
        )
        delta_scale = float(np.max(np.abs(delta_last), initial=0.0))
        stored_floor = epsilon * background
        limit = symmetric_limit(field_delta(first, field), delta_last)
        result[field.key] = {
            "background_max_abs": background,
            "delta_max_abs_final": delta_scale,
            "stored_subtraction_roundoff_floor": stored_floor,
            "roundoff_floor_over_delta": (
                stored_floor / delta_scale if delta_scale > 0.0 else float("inf")
            ),
            "shared_final_delta_limit": limit,
        }
    return result


def _long_common_limits(
    selected: list[tuple[str, float, list[SnapshotPair]]],
    field: FieldSpec,
) -> tuple[tuple[float, float], tuple[float, float]]:
    actual_arrays: list[np.ndarray] = []
    difference_arrays: list[np.ndarray] = []
    for _, _, pairs in selected:
        controls = [pair.control for pair in pairs]
        actual_arrays.extend(field_values(snapshot, field) for snapshot in controls)
        t1 = field_values(controls[0], field)
        difference_arrays.extend(
            (t1 - field_values(controls[1], field), t1 - field_values(controls[2], field))
        )
    return (
        sequential_limits(np.concatenate(actual_arrays)),
        symmetric_limits(*difference_arrays),
    )


def _provenance_outputs(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"provenance.json", "SHA256SUMS"}:
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def embed_svg_html(html_text: str, root: Path) -> str:
    pattern = re.compile(r'src="([^"]+\.png)"')

    def replace(match: re.Match[str]) -> str:
        relative = unquote(match.group(1))
        svg_path = root / Path(relative).with_suffix(".svg")
        if not svg_path.is_file():
            raise InputError(f"cannot embed missing SVG: {svg_path}")
        encoded = base64.b64encode(svg_path.read_bytes()).decode("ascii")
        return f'src="data:image/svg+xml;base64,{encoded}"'

    rendered = pattern.sub(replace, html_text)
    remaining = [
        value
        for value in re.findall(r'src="([^"]+)"', rendered)
        if not value.startswith("data:")
    ]
    if remaining:
        raise ProvenanceError(f"HTML retains external image sources: {remaining[:3]}")
    return rendered


def _all_figure_names(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in (root / "figures").glob("*.png")
    )


def write_checksums(root: Path) -> Path:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            records.append(
                f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
            )
    checksum_path = root / "SHA256SUMS"
    checksum_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    return checksum_path


def verify_output(root: Path, *, expected_figures: int) -> dict[str, Any]:
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise ProvenanceError("missing SHA256SUMS")
    listed: set[str] = set()
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        pieces = line.split("  ", 1)
        if len(pieces) != 2:
            raise ProvenanceError(f"invalid SHA256SUMS line {line_number}")
        expected, relative = pieces
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ProvenanceError(f"output checksum mismatch: {relative}")
        if relative in listed:
            raise ProvenanceError(f"duplicate output checksum: {relative}")
        listed.add(relative)
    expected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if listed != expected:
        raise ProvenanceError(
            f"checksum coverage mismatch: missing={sorted(expected - listed)}, "
            f"extra={sorted(listed - expected)}"
        )
    figures = _all_figure_names(root)
    if len(figures) != expected_figures:
        raise ProvenanceError(
            f"figure count {len(figures)} != expected {expected_figures}"
        )
    if len(figures) > MAX_FIGURES:
        raise ProvenanceError(f"figure count {len(figures)} exceeds {MAX_FIGURES}")
    html_path = root / "sections.html"
    html_text = html_path.read_text(encoding="utf-8")
    if 'data:image/svg+xml;base64,' not in html_text:
        raise ProvenanceError("HTML does not embed SVG image data")
    if re.search(r'src="(?!data:)', html_text):
        raise ProvenanceError("HTML still depends on an external image source")
    provenance = load_json_object(root / "provenance.json", "output provenance")
    if not provenance.get("input_sha256") or not provenance.get("output_sha256"):
        raise ProvenanceError("provenance lacks exact input/output checksums")
    return {
        "html": str(html_path),
        "html_bytes": html_path.stat().st_size,
        "figures": len(figures),
        "png_files": len(figures),
        "svg_files": len(list((root / "figures").glob("*.svg"))),
        "input_files": len(provenance["input_sha256"]),
        "output_files": len(provenance["output_sha256"]),
        "sha256sums": str(checksum_path),
    }


def build(run_root: Path, source_root: Path, output: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    source_root = source_root.resolve()
    output = _safe_output(run_root, source_root, output)
    inputs = InputIndex()
    inputs.file(Path(__file__).resolve(), "cross-section producer source")
    a0, t0, canonical_paths = load_canonical(source_root)
    for name, path in canonical_paths.items():
        inputs.file(path, f"canonical {name} renderer/style")
    plt = t0.configure_matplotlib()

    spatial = classify_spatial_runs(run_root, inputs)
    completed_spatial = [run for run in spatial if run.status == "completed"]
    methods = load_long_methods(run_root, inputs)
    common = common_long_pairs(methods)
    selected_common = select_early_middle_final(common)
    final_times = [method.pairs[-1].time_s for method in methods]
    if max(final_times) - min(final_times) > TIME_RTOL * max(final_times):
        raise InputError(f"long method final times do not match: {final_times}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    )
    figure_counter = 0
    figure_meta: list[dict[str, Any]] = []
    data: dict[str, Any] = {
        "schema": SCHEMA,
        "run_root": str(run_root),
        "source_root": str(source_root),
        "coordinate_scope": (
            "Real-Cartesian R-Z active-center cells at physical toroidal phi=0; "
            "no Fourier reconstruction, triangulation, smoothing, or pressure inference."
        ),
        "availability": [
            {
                "run": run.name,
                "status": run.status,
                "nodes": run.nodes,
                "note": run.note,
            }
            for run in spatial
        ],
        "runs": {},
        "long_chain": {},
        "figures": figure_meta,
    }

    def add_figure(
        *,
        title: str,
        caption: str,
        base_name: str,
        panel_records: list[tuple[str, str]],
        status: str,
        device: str,
        components: list[str],
        grid: int,
        nphi: int,
        question: str,
        controls: str,
        criterion: str,
        source_kind: str = "nonlinear_evolution",
    ) -> None:
        nonlocal figure_counter
        figure_counter += 1
        image = f"figures/{base_name}.png"
        svg = f"figures/{base_name}.svg"
        record = _figure_record(
            figure_counter,
            title=title,
            caption=caption,
            image=image,
            svg=svg,
            panels=panel_records,
            status=status,
            device=device,
            components=components,
            grid=grid,
            nphi=nphi,
            n_phi_quadrature=nphi,
            question=question,
            controls=controls,
            criterion=criterion,
            source_kind=source_kind,
        )
        figure_meta.append(record)

    try:
        (temporary / "figures").mkdir()
        for run in completed_spatial:
            assert run.mapping is not None
            assert run.nphi is not None
            assert run.nodes is not None
            assert run.pairs
            first, final = run.pairs[0], run.pairs[-1]
            limits = _field_capacity_notes(run.pairs)
            data["runs"][run.name] = {
                "status": run.status,
                "nodes": run.nodes,
                "active_cells": run.mapping.nc,
                "nphi": run.nphi,
                "device": run.device,
                "first_step": first.step,
                "first_time_s": first.time_s,
                "final_step": final.step,
                "final_time_s": final.time_s,
                "field_precision": limits,
            }

            magnetic_groups = ((FIELD_BY_KEY["B_R"], FIELD_BY_KEY["B_phi"]), (FIELD_BY_KEY["B_Z"],))
            for index, fields in enumerate(magnetic_groups, start=1):
                prefix = f"fig_{figure_counter + 1:03d}_{run.name}_magnetic_time"
                records = plot_delta_times(
                    first=first,
                    last=final,
                    mapping=run.mapping,
                    fields=fields,
                    output=temporary,
                    prefix=prefix,
                    t0=t0,
                    plt=plt,
                )
                for _, panels, components in records:
                    add_figure(
                        title=f"{run.name}: magnetic perturbation, first versus final",
                        caption=(
                            f"Matched saved control/perturbed snapshots for physical phi=0. "
                            f"Time range {first.time_s:.9e} to {final.time_s:.9e} s. "
                            "Each component uses shared centered limits across the two times."
                        ),
                        base_name=prefix,
                        panel_records=panels,
                        status="completed nonlinear spatial probe; capacity response only",
                        device=run.device,
                        components=components,
                        grid=run.nodes,
                        nphi=run.nphi,
                        question="How do the saved magnetic perturbation structures differ between the first and final common snapshots?",
                        controls="Matched control and perturbed snapshots; shared normalization across times.",
                        criterion="Direct signed sections on active cells; no convergence or eigenmode claim.",
                    )

            final_groups = (
                (FIELD_BY_KEY["B_R"], FIELD_BY_KEY["B_phi"]),
                (FIELD_BY_KEY["B_Z"], FIELD_BY_KEY["u_R"]),
                (FIELD_BY_KEY["u_phi"], FIELD_BY_KEY["u_Z"]),
                (FIELD_BY_KEY["n_i"],),
            )
            for index, fields in enumerate(final_groups, start=1):
                prefix = f"fig_{figure_counter + 1:03d}_{run.name}_final_fields_{index}"
                records = plot_final_fields(
                    pair=final,
                    mapping=run.mapping,
                    fields=fields,
                    delta_limits={
                        key: (
                            -limits[key]["shared_final_delta_limit"],
                            limits[key]["shared_final_delta_limit"],
                        )
                        for key in limits
                    },
                    output=temporary,
                    prefix=prefix,
                    t0=t0,
                    plt=plt,
                )
                for _, panels, components in records:
                    add_figure(
                        title=f"{run.name}: final control and perturbation sections",
                        caption=(
                            f"Final matched saved sections at step {final.step}, "
                            f"t={final.time_s:.9e} s. Delta is perturbed minus control. "
                            "Active-center cells use actual R/Z extent; the wall polygon is overlaid."
                        ),
                        base_name=prefix,
                        panel_records=panels,
                        status="completed nonlinear spatial probe; one saved final time",
                        device=run.device,
                        components=components,
                        grid=run.nodes,
                        nphi=run.nphi,
                        question="What are the saved control fields and the perturbation response on the actual R-Z cells?",
                        controls="Matched control/perturbed run pair; signed deltas use centered normalization.",
                        criterion="Direct stored physical-phi sections in SI; no inferred pressure or smoothing.",
                    )

        for method in methods:
            final = method.pairs[-1]
            limits = _field_capacity_notes(method.pairs)
            data["long_chain"].setdefault("methods", {})[method.name] = {
                "directories": [str(path) for path in method.directories],
                "status": method.status,
                "active_cells": method.mapping.nc,
                "nphi": method.nphi,
                "final_step": final.step,
                "final_time_s": final.time_s,
                "field_precision": limits,
            }
            final_groups = (
                (FIELD_BY_KEY["B_R"], FIELD_BY_KEY["B_phi"]),
                (FIELD_BY_KEY["B_Z"], FIELD_BY_KEY["u_R"]),
                (FIELD_BY_KEY["u_phi"], FIELD_BY_KEY["u_Z"]),
                (FIELD_BY_KEY["n_i"],),
            )
            slug = method.name.lower().replace(" ", "_")
            for index, fields in enumerate(final_groups, start=1):
                prefix = f"fig_{figure_counter + 1:03d}_long_{slug}_final_fields_{index}"
                records = plot_final_fields(
                    pair=final,
                    mapping=method.mapping,
                    fields=fields,
                    delta_limits={
                        key: (
                            -limits[key]["shared_final_delta_limit"],
                            limits[key]["shared_final_delta_limit"],
                        )
                        for key in limits
                    },
                    output=temporary,
                    prefix=prefix,
                    t0=t0,
                    plt=plt,
                )
                for _, panels, components in records:
                    add_figure(
                        title=f"Long RZ {method.name}: final control and perturbation",
                        caption=(
                            f"Budget-stopped matched chain endpoint at step {final.step}, "
                            f"t={final.time_s:.9e} s. This is a bounded retained-time prefix."
                        ),
                        base_name=prefix,
                        panel_records=panels,
                        status="budget-stopped bounded long-chain prefix; not a convergence result",
                        device="cuda:0",
                        components=components,
                        grid=12,
                        nphi=method.nphi,
                        question="What does the retained final common-time response look like for this method?",
                        controls="Matched control/perturbed pair and verified continuation mapping.",
                        criterion="Stored real-RZ sections only; no interpolation or eigenmode claim.",
                    )

        common_limits: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
            field.key: _long_common_limits(selected_common, field)
            for field in (FIELD_BY_KEY["B_phi"], FIELD_BY_KEY["u_phi"])
        }
        selected_summary = []
        for label, time_s, pairs in selected_common:
            selected_summary.append(
                {"label": label, "time_s": time_s, "actual_common_saved_time": True}
            )
            for field in (FIELD_BY_KEY["B_phi"], FIELD_BY_KEY["u_phi"]):
                actual_limits, difference_limits = common_limits[field.key]
                prefix = (
                    f"fig_{figure_counter + 1:03d}_long_common_{label}_{field.key}"
                )
                records = plot_long_common(
                    label=label,
                    time_s=time_s,
                    pairs=pairs,
                    mapping=methods[0].mapping,
                    field=field,
                    actual_limits=actual_limits,
                    difference_limits=difference_limits,
                    output=temporary,
                    prefix=prefix,
                    t0=t0,
                    plt=plt,
                )
                for suffix, panels, components in records:
                    is_difference = suffix.endswith("_difference")
                    add_figure(
                        title=(
                            f"Long RZ {label} common-time {field.key} sections"
                            if not is_difference
                            else f"Long RZ {label} common-time {field.key} method differences"
                        ),
                        caption=(
                            f"Only actual saved common time {time_s:.9e} s is used. "
                            "T1, A0 same-h, and A0 half-h share the verified 12-node mapping."
                        ),
                        base_name=suffix,
                        panel_records=panels,
                        status="budget-stopped bounded common-time comparison; no convergence claim",
                        device="cuda:0",
                        components=components,
                        grid=12,
                        nphi=methods[0].nphi,
                        question="How do retained T1 and matched A0 controls compare on the same grid and time?",
                        controls="Strict recorded-parent chain checkpoints, mapping, and exact saved times.",
                        criterion=(
                            "Visible raw sections and differences; large-background cancellation "
                            "limits precision and does not identify a physical eigenmode."
                        ),
                        source_kind="mixed_comparison",
                    )
        data["long_chain"]["common_times"] = selected_summary
        data["long_chain"]["common_time_count"] = len(common)

        if figure_counter > MAX_FIGURES:
            raise InputError(
                f"bounded build produced {figure_counter} figures, limit {MAX_FIGURES}"
            )

        no_data_notes = [
            run.note for run in spatial if run.note is not None
        ]
        no_data_notes.extend(
            (
                "T1 uses a verified continuation prefix through 2.017595307918 target periods; "
                "the original longer plan was not executed.",
                "The failed long-east12-waveform-segment1 branch contributed no numeric panels.",
                "Large backgrounds are subtracted in FP64; stored-sample cancellation floors "
                "are recorded but are not solver error bounds.",
                "Saved physical-phi pressure observations are absent; this bounded gallery "
                "does not reconstruct pressure from the retained state arrays.",
                "The N40 completed A0 probe is a bounded response/capacity probe, not an eigenmode.",
                "No spatial convergence, temporal convergence, or nonlinear stability conclusion "
                "is claimed.",
            )
        )
        data["precision_scope"] = {
            "stored_dtype": "complex128 observations; FP64 real sections at phi=0",
            "cancellation_floor": "eps64 * max(|control|, |perturbed|)",
            "warning": "The floor is a stored subtraction scale, not a solver-error certificate.",
        }
        data["available_figure_slots"] = MAX_FIGURES
        _write_json(temporary / "data.json", data)
        _write_json(
            temporary / "provenance.json",
            {
                "schema": f"{SCHEMA}-provenance",
                "status": "pending final output hash collection",
            },
        )

        pages = [
            {
                "id": "short-probes",
                "title": "Short 16/24/32/40 spatial probes",
                "introduction": (
                    "Completed probes contribute first/final magnetic perturbations and "
                    "final control/delta sections. The rows below do not interpolate snapshots."
                ),
                "figures": [
                    record
                    for record in figure_meta
                    if "long" not in record["title"].lower()
                ],
                "notes": no_data_notes,
            },
            {
                "id": "long-rz",
                "title": "Matched 12-node long-RZ chain",
                "introduction": (
                    "The retained T1, A0 same-h, and A0 half-h prefixes are compared only "
                    "at actual saved common times on a verified common grid."
                ),
                "figures": [
                    record
                    for record in figure_meta
                    if "long" in record["title"].lower()
                ],
                "notes": [
                    "All long methods are budget-stopped bounded prefixes.",
                    "Parent checkpoint hashes, steps, times, actual grids, and mappings were verified.",
                ],
            },
        ]
        manifest = {
            "title": "FOMHD 0.5.0 real R-Z cross sections",
            "introduction": (
                "English, offline evidence gallery generated from saved 0.5.0 probes. "
                "Physical toroidal phi index zero is plotted directly; no Fourier transform, "
                "triangulation, smoothing, or solver execution is used."
            ),
            "notation": [
                {"symbol": r"$R$", "meaning": "Cartesian major radius in metres"},
                {"symbol": r"$Z$", "meaning": "Cartesian vertical coordinate in metres"},
                {"symbol": "control", "meaning": "Unperturbed nonlinear run ensemble"},
                {"symbol": "delta", "meaning": "Perturbed minus control at one matched step/time"},
                {"symbol": "shared limits", "meaning": "Identical colour normalization for compared panels"},
            ],
            "archive": {"label": "Output directory", "path": "."},
            "pages": pages,
            "conclusions": (
                "These figures expose retained spatial response structures without claiming "
                "an eigenmode, convergence, or physical instability. Failed and preflight-"
                "rejected probes remain explicit no-data records."
            ),
            "known_issues": [
                {"label": "Compact build data and precision notes", "path": "data.json"},
                {"label": "Exact input and output SHA256 provenance", "path": "provenance.json"},
            ],
        }
        manifest_path = temporary / "manifest.json"
        _write_json(manifest_path, manifest)
        html_path = temporary / "sections.html"
        rendered = a0.render_report(manifest, manifest_path, html_path)
        html_path.write_text(embed_svg_html(rendered, temporary), encoding="utf-8")

        output_records = _provenance_outputs(temporary)
        provenance = {
            "schema": f"{SCHEMA}-provenance",
            "run_root": str(run_root),
            "source_root": str(source_root),
            "input_sha256": inputs.records(),
            "output_sha256": output_records,
            "figure_count": figure_counter,
            "html_self_contained": True,
            "excluded_from_self_hash": ["provenance.json", "SHA256SUMS"],
        }
        _write_json(temporary / "provenance.json", provenance)
        write_checksums(temporary)
        summary = verify_output(temporary, expected_figures=figure_counter)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    summary["output"] = str(output)
    summary["html"] = str(output / "sections.html")
    summary["sha256sums"] = str(output / "SHA256SUMS")
    summary["requested_output_was_new"] = True
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build(args.run_root, args.source_root, args.output)
    except (InputError, UnsafeOutputError, ProvenanceError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
