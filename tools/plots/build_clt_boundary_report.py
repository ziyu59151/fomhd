#!/usr/bin/env python3
"""Build an offline A0-style gallery for native-polygon CLT boundary runs."""

from __future__ import annotations

import argparse
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable


for thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[thread_variable] = "1"

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402


SCHEMA = "fomhd-clt-boundary-gallery-v1"
INPUT_ORIGIN = (
    "caller-supplied native run reports and snapshot arrays; "
    "provenance hashes recorded"
)
SCIENTIFIC_ACCEPTANCE = "not established by renderer"
DATA_SCOPE = (
    "Reads, validates and hashes caller-supplied native run reports, "
    "geometry.npz and physical snapshot arrays; renders actual native polygon "
    "cells and paired physical-time differences. The renderer does not "
    "establish physical wall accuracy or scientific acceptance."
)
MAX_FIGURES = 32
TIME_RTOL = 1.0e-12
TIME_ATOL = 1.0e-30
PHYSICAL_REAL_RTOL = 1.0e-10
PHYSICAL_REAL_ATOL = 1.0e-12
ALLOWED_STATUSES = {
    "completed",
    "segment_completed",
    "budget_stopped",
    "failed",
}


class GalleryError(ValueError):
    """Raised when an input cannot support an honest gallery."""


@dataclass(frozen=True)
class FieldSpec:
    key: str
    title: str
    array_key: str
    component: int | None
    unit: str
    signed: bool


FIELDS = (
    FieldSpec("B_R", "Radial magnetic field", "B_N_samples", 0, "T", True),
    FieldSpec("B_phi", "Toroidal magnetic field", "B_N_samples", 1, "T", True),
    FieldSpec("B_Z", "Vertical magnetic field", "B_N_samples", 2, "T", True),
    FieldSpec(
        "u_R",
        "Radial ion velocity",
        "ion_velocity_samples",
        0,
        "m s^-1",
        True,
    ),
    FieldSpec(
        "u_phi",
        "Toroidal ion velocity",
        "ion_velocity_samples",
        1,
        "m s^-1",
        True,
    ),
    FieldSpec(
        "u_Z",
        "Vertical ion velocity",
        "ion_velocity_samples",
        2,
        "m s^-1",
        True,
    ),
    FieldSpec(
        "n_i",
        "Ion density",
        "ion_density_samples",
        None,
        "m^{-3}",
        False,
    ),
    FieldSpec(
        "p_i",
        "Ion perpendicular pressure",
        "ion_perp_pressure_samples",
        None,
        "Pa",
        False,
    ),
    FieldSpec(
        "p_e_perp",
        "Electron perpendicular pressure",
        "electron_perp_pressure_samples",
        None,
        "Pa",
        False,
    ),
    FieldSpec(
        "p_e_parallel",
        "Electron parallel pressure",
        "electron_parallel_pressure_samples",
        None,
        "Pa",
        False,
    ),
    FieldSpec(
        "n_e",
        "Electron density",
        "electron_density_samples",
        None,
        "m^{-3}",
        False,
    ),
)
FIELD_BY_KEY = {field.key: field for field in FIELDS}
MAGNETIC_FIELDS = tuple(FIELD_BY_KEY[name] for name in ("B_R", "B_phi", "B_Z"))

FORCE_KEYS = {
    "pressure": "frame_ion_pressure_force_samples",
    "magnetic": "frame_ion_magnetic_force_samples",
    "electric": "frame_ion_electric_force_samples",
    "convection": "frame_ion_convection_samples",
    "wall_reaction": "frame_ion_wall_reaction_samples",
    "net": "frame_ion_net_force_samples",
}

ARRAY_SEMANTICS = {
    "B_N_samples": {
        "shape": ["Nphi", 3, "Nc"],
        "components": ["R", "phi", "Z"],
        "unit": "T",
        "meaning": "Actual physical magnetic field samples.",
    },
    "ion_velocity_samples": {
        "shape": ["Nphi", 3, "Nc"],
        "components": ["R", "phi", "Z"],
        "unit": "m s^-1",
        "meaning": "Actual physical ion velocity samples.",
    },
    "ion_density_samples": {
        "shape": ["Nphi", "Nc"],
        "unit": "m^{-3}",
        "meaning": "Actual physical ion density samples.",
    },
    "ion_perp_pressure_samples": {
        "shape": ["Nphi", "Nc"],
        "unit": "Pa",
        "meaning": "Actual physical ion perpendicular pressure samples.",
    },
    "electron_density_samples": {
        "shape": ["Nphi", "Nc"],
        "unit": "m^{-3}",
        "meaning": "Actual physical electron density samples.",
    },
    "electron_perp_pressure_samples": {
        "shape": ["Nphi", "Nc"],
        "unit": "Pa",
        "meaning": "Actual physical electron perpendicular pressure samples.",
    },
    "electron_parallel_pressure_samples": {
        "shape": ["Nphi", "Nc"],
        "unit": "Pa",
        "meaning": "Actual physical electron parallel pressure samples.",
    },
    **{
        key: {
            "shape": ["Nphi", 3, "Nc"],
            "unit": "N m^-3",
            "meaning": (
                "Explicit retained initial ion force/cache vector. It is never "
                "reconstructed from guessed arrays."
            ),
        }
        for key in FORCE_KEYS.values()
    },
}


@dataclass(frozen=True)
class PairSpec:
    label: str
    control: Path
    perturbed: Path


@dataclass(frozen=True)
class Geometry:
    polygons: tuple[np.ndarray, ...]
    contour: np.ndarray
    chord_wall: np.ndarray
    centers: np.ndarray
    volumes: np.ndarray
    wall_distances: np.ndarray
    face_coords: np.ndarray
    face_normals: np.ndarray
    fingerprint: str

    @property
    def nc(self) -> int:
        return len(self.polygons)


@dataclass(frozen=True)
class SnapshotHeader:
    path: Path
    sha256: str
    step: int
    time_s: float
    metadata: dict[str, Any]
    sample_phis: np.ndarray


@dataclass(frozen=True)
class RunEvidence:
    directory: Path
    report: dict[str, Any]
    report_sha256: str
    parameters: dict[str, Any]
    impact: str
    source_fingerprint: str | None
    geometry_fingerprint: str
    geometry_sha256: str
    nc: int
    nphi: int
    K: int
    nodes: int
    status: str
    snapshots: tuple[SnapshotHeader, ...]


@dataclass(frozen=True)
class MatchedPair:
    label: str
    control: RunEvidence
    perturbed: RunEvidence
    control_initial: SnapshotHeader
    perturbed_initial: SnapshotHeader
    control_final: SnapshotHeader
    perturbed_final: SnapshotHeader
    matched_time_s: float
    matched_step: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise GalleryError(f"non-finite JSON constant is not allowed: {value}")


def load_json_object(path: Path, where: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (OSError, json.JSONDecodeError) as error:
        raise GalleryError(f"cannot read {where} {path}: {error}") from error
    if not isinstance(value, dict):
        raise GalleryError(f"{where} must contain a JSON object: {path}")
    return value


def load_snapshot_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load one non-pickle NPZ snapshot for tests and explicit callers."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as error:
        raise GalleryError(f"invalid snapshot NPZ {path}: {error}") from error


def _required_object(mapping: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise GalleryError(f"{where}.{key} must be an object")
    return value


def _required_text(mapping: dict[str, Any], key: str, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GalleryError(f"{where}.{key} must be nonempty text")
    return value


def _required_positive_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GalleryError(f"{where} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise GalleryError(f"{where} must be a finite positive number")
    return result


def _required_int(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GalleryError(f"{where} must be an integer >= {minimum}")
    return value


class InputIndex:
    """Hash each immutable input once and retain every consuming role."""

    def __init__(self) -> None:
        self._records: dict[Path, dict[str, Any]] = {}

    def file(self, path: Path, role: str) -> Path:
        resolved = path.resolve()
        if not resolved.is_file() or resolved.is_symlink():
            raise GalleryError(f"missing regular input file: {resolved}")
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

    def records(self) -> list[dict[str, Any]]:
        return [self._records[path].copy() for path in sorted(self._records)]

    def verify_unchanged(self) -> None:
        for path, record in self._records.items():
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise GalleryError(
                    f"input changed during rendering: {path} "
                    f"(expected {record['sha256']})"
                )


def _safe_relative_child(root: Path, value: str, where: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise GalleryError(f"{where} must be a relative path below {root}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise GalleryError(f"{where} escapes {root}") from error
    return path


def read_snapshot_header(path: Path, inputs: InputIndex, role: str) -> SnapshotHeader:
    resolved = inputs.file(path, role)
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            if "metadata_json" not in archive.files:
                raise GalleryError(
                    f"snapshot metadata_json key is missing: {resolved}"
                )
            metadata_value = np.asarray(archive["metadata_json"])
    except (OSError, ValueError) as error:
        raise GalleryError(f"invalid snapshot NPZ {resolved}: {error}") from error
    if metadata_value.shape != ():
        raise GalleryError(f"metadata_json must be scalar text: {resolved}")
    try:
        metadata = json.loads(str(metadata_value), parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise GalleryError(f"invalid metadata_json in {resolved}: {error}") from error
    if not isinstance(metadata, dict):
        raise GalleryError(f"metadata_json must contain an object: {resolved}")
    for key in ("time_s", "step", "units", "sample_phis"):
        if key not in metadata:
            raise GalleryError(
                f"snapshot metadata_json is missing required key {key!r}: {resolved}"
            )
    if metadata["units"] != "SI":
        raise GalleryError(
            f"snapshot metadata_json.units must be 'SI', got "
            f"{metadata['units']!r}: {resolved}"
        )
    step = _required_int(metadata["step"], f"{resolved}.metadata_json.step")
    try:
        time_s = float(metadata["time_s"])
        sample_phis = np.asarray(metadata["sample_phis"], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise GalleryError(f"invalid snapshot metadata values: {resolved}") from error
    if not math.isfinite(time_s):
        raise GalleryError(f"snapshot time_s is not finite: {resolved}")
    if sample_phis.ndim != 1 or sample_phis.size == 0:
        raise GalleryError(
            f"snapshot metadata_json.sample_phis must be a nonempty vector: {resolved}"
        )
    if not np.isfinite(sample_phis).all():
        raise GalleryError(f"sample_phis contains non-finite values: {resolved}")
    if np.unique(sample_phis).size != sample_phis.size:
        raise GalleryError(f"sample_phis values are not distinct: {resolved}")
    return SnapshotHeader(
        path=resolved,
        sha256=inputs._records[resolved]["sha256"],
        step=step,
        time_s=time_s,
        metadata=metadata,
        sample_phis=sample_phis,
    )


def load_run(directory: Path, inputs: InputIndex, role: str) -> RunEvidence:
    directory = directory.resolve()
    report_path = directory / "report.json"
    inputs.file(report_path, f"{role} report")
    report = load_json_object(report_path, f"{role} report")
    parameters = _required_object(report, "parameters", f"{role} report")
    geometry = _required_object(report, "geometry", f"{role} report")
    status = _required_text(report, "status", f"{role} report")
    if status not in ALLOWED_STATUSES:
        raise GalleryError(
            f"{role} report status {status!r} is not an accepted snapshot state"
        )
    K = _required_int(parameters.get("K"), f"{role}.parameters.K")
    nphi = _required_int(
        parameters.get("Nphi"), f"{role}.parameters.Nphi", minimum=1
    )
    if nphi != 4 * K + 1:
        raise GalleryError(
            f"{role} Nphi={nphi} does not equal 4*K+1 for K={K}"
        )
    nc = _required_int(
        geometry.get("Nc"), f"{role}.geometry.Nc", minimum=1
    )
    nodes = _required_int(
        parameters.get("nodes"), f"{role}.parameters.nodes", minimum=1
    )
    geometry_fingerprint = _required_text(
        geometry, "fingerprint", f"{role}.geometry"
    )
    geometry_sha256 = _required_text(
        geometry, "sha256", f"{role}.geometry"
    )
    if not re.fullmatch(r"[0-9a-f]{64}", geometry_sha256):
        raise GalleryError(f"{role}.geometry.sha256 must be lowercase SHA256")
    provenance = report.get("provenance")
    source_fingerprint = None
    if isinstance(provenance, dict):
        value = provenance.get("source_fingerprint")
        if isinstance(value, str) and value:
            source_fingerprint = value
    raw_snapshots = report.get("snapshots")
    if not isinstance(raw_snapshots, list) or not raw_snapshots:
        raise GalleryError(f"{role}.snapshots must be a nonempty array")
    snapshot_root = directory / "snapshots"
    snapshots: list[SnapshotHeader] = []
    for index, raw_name in enumerate(raw_snapshots):
        if not isinstance(raw_name, str) or not raw_name:
            raise GalleryError(f"{role}.snapshots[{index}] must be path text")
        path = _safe_relative_child(
            snapshot_root, raw_name, f"{role}.snapshots[{index}]"
        )
        snapshots.append(
            read_snapshot_header(path, inputs, f"{role} snapshot {raw_name}")
        )
    times = [snapshot.time_s for snapshot in snapshots]
    steps = [snapshot.step for snapshot in snapshots]
    if len(times) != len(set(times)):
        raise GalleryError(f"{role} has duplicate snapshot physical times")
    if len(steps) != len(set(steps)):
        raise GalleryError(f"{role} has duplicate snapshot steps")
    return RunEvidence(
        directory=directory,
        report=report,
        report_sha256=inputs._records[report_path.resolve()]["sha256"],
        parameters=dict(parameters),
        impact=str(parameters.get("amplitude")),
        source_fingerprint=source_fingerprint,
        geometry_fingerprint=geometry_fingerprint,
        geometry_sha256=geometry_sha256,
        nc=nc,
        nphi=nphi,
        K=K,
        nodes=nodes,
        status=status,
        snapshots=tuple(snapshots),
    )


def _initial_snapshot(run: RunEvidence) -> SnapshotHeader:
    candidates = [
        snapshot
        for snapshot in run.snapshots
        if snapshot.step == 0
        and abs(snapshot.time_s) <= TIME_ATOL
    ]
    if len(candidates) != 1:
        raise GalleryError(
            f"run {run.directory} lacks exactly one true initial snapshot "
            "with step=0 and time_s=0; the first saved point is not accepted"
        )
    return candidates[0]


def _common_time(
    control: RunEvidence, perturbed: RunEvidence
) -> tuple[SnapshotHeader, SnapshotHeader]:
    matches: list[tuple[SnapshotHeader, SnapshotHeader]] = []
    for left in control.snapshots:
        for right in perturbed.snapshots:
            if math.isclose(
                left.time_s,
                right.time_s,
                rel_tol=TIME_RTOL,
                abs_tol=TIME_ATOL,
            ):
                matches.append((left, right))
    if not matches:
        raise GalleryError(
            f"physical time mismatch between {control.directory} and "
            f"{perturbed.directory}; no common saved time exists"
        )
    noninitial = [
        pair
        for pair in matches
        if pair[0].step != 0 or abs(pair[0].time_s) > TIME_ATOL
    ]
    if not noninitial:
        raise GalleryError(
            f"physical time mismatch between {control.directory} and "
            f"{perturbed.directory}; only t=0 is common"
        )
    return max(noninitial, key=lambda pair: pair[0].time_s)


def _compatibility_error(
    control: RunEvidence, perturbed: RunEvidence
) -> str | None:
    ignored = {"amplitude", "device", "output"}
    left = {
        key: value
        for key, value in control.parameters.items()
        if key not in ignored
    }
    right = {
        key: value
        for key, value in perturbed.parameters.items()
        if key not in ignored
    }
    if left != right:
        differing = sorted(set(left) | set(right))
        details = [
            f"{key}: {left.get(key)!r} != {right.get(key)!r}"
            for key in differing
            if left.get(key) != right.get(key)
        ]
        return "; ".join(details)
    if control.geometry_fingerprint != perturbed.geometry_fingerprint:
        return (
            "geometry fingerprint mismatch "
            f"{control.geometry_fingerprint!r} != "
            f"{perturbed.geometry_fingerprint!r}"
        )
    if (
        control.source_fingerprint is not None
        and perturbed.source_fingerprint is not None
        and control.source_fingerprint != perturbed.source_fingerprint
    ):
        return "source fingerprint mismatch"
    if (control.nc, control.nphi, control.K) != (
        perturbed.nc,
        perturbed.nphi,
        perturbed.K,
    ):
        return "mesh or retained-mode shape mismatch"
    return None


def match_pairs(specs: Iterable[PairSpec], inputs: InputIndex) -> list[MatchedPair]:
    matched: list[MatchedPair] = []
    labels: set[str] = set()
    for spec in specs:
        label = spec.label.strip()
        if not label:
            raise GalleryError("pair label must be nonempty")
        if label in labels:
            raise GalleryError(f"duplicate pair label: {label}")
        labels.add(label)
        control = load_run(spec.control, inputs, f"pair {label} control")
        perturbed = load_run(spec.perturbed, inputs, f"pair {label} perturbed")
        mismatch = _compatibility_error(control, perturbed)
        if mismatch:
            raise GalleryError(f"incompatible runs in pair {label}: {mismatch}")
        control_initial = _initial_snapshot(control)
        perturbed_initial = _initial_snapshot(perturbed)
        if not np.array_equal(
            control_initial.sample_phis, perturbed_initial.sample_phis
        ):
            raise GalleryError(
                f"toroidal sample angles differ in pair {label}"
            )
        control_final, perturbed_final = _common_time(control, perturbed)
        if control_final.sample_phis.shape[0] != control.nphi:
            raise GalleryError(
                f"pair {label} metadata sample_phis count "
                f"{control_final.sample_phis.shape[0]} != Nphi {control.nphi}"
            )
        if not np.array_equal(
            control_final.sample_phis, perturbed_final.sample_phis
        ):
            raise GalleryError(
                f"matched snapshot toroidal angles differ in pair {label}"
            )
        matched.append(
            MatchedPair(
                label=label,
                control=control,
                perturbed=perturbed,
                control_initial=control_initial,
                perturbed_initial=perturbed_initial,
                control_final=control_final,
                perturbed_final=perturbed_final,
                matched_time_s=float(control_final.time_s),
                matched_step=control_final.step,
            )
        )
    if not matched:
        raise GalleryError("at least one --pair is required")
    return matched


def _physical_values(
    arrays: dict[str, np.ndarray],
    key: str,
    expected_shape: tuple[int, ...],
    path: Path,
) -> np.ndarray:
    if key not in arrays:
        raise GalleryError(f"missing physical SAMPLE array {key!r}: {path}")
    value = np.asarray(arrays[key])
    if value.shape != expected_shape:
        raise GalleryError(
            f"{key} shape {value.shape} != {expected_shape}: {path}"
        )
    if value.dtype.kind not in "biufc":
        raise GalleryError(f"{key} must be numeric, got {value.dtype}: {path}")
    if not np.isfinite(value).all():
        raise GalleryError(f"{key} contains non-finite values: {path}")
    if np.iscomplexobj(value):
        real = value.real.astype(np.float64, copy=False)
        scale = max(float(np.max(np.abs(real), initial=0.0)), 1.0)
        tolerance = PHYSICAL_REAL_ATOL + PHYSICAL_REAL_RTOL * scale
        if np.any(np.abs(value.imag) > tolerance):
            raise GalleryError(
                f"{key} selected samples are not physical-real within "
                f"tolerance: {path}"
            )
        return real.copy()
    return value.astype(np.float64, copy=False)


def extract_field(
    arrays: dict[str, np.ndarray], field: FieldSpec, phi_index: int
) -> np.ndarray:
    metadata_value = np.asarray(arrays.get("metadata_json", ""))
    try:
        metadata = json.loads(str(metadata_value))
    except json.JSONDecodeError:
        metadata = {}
    if isinstance(metadata, dict) and "sample_phis" in metadata:
        nphi = np.asarray(metadata["sample_phis"]).size
    else:
        shape = np.asarray(arrays.get(field.array_key, [])).shape
        if len(shape) not in (2, 3):
            raise GalleryError(
                f"cannot determine Nphi from {field.array_key!r}"
            )
        nphi = shape[0]
    if phi_index < 0 or phi_index >= nphi:
        raise GalleryError(
            f"phi_index {phi_index} is outside Nphi sample range [0,{nphi - 1}]"
        )
    shape = np.asarray(arrays.get(field.array_key, [])).shape
    if len(shape) == 3:
        nc = shape[2]
        expected = (nphi, 3, nc)
    elif len(shape) == 2:
        nc = shape[1]
        expected = (nphi, nc)
    else:
        raise GalleryError(f"invalid shape for {field.array_key!r}: {shape}")
    values = _physical_values(
        arrays, field.array_key, expected, Path("loaded snapshot")
    ).reshape(nphi, *expected[1:])[phi_index]
    if field.component is not None:
        return np.asarray(values[field.component], dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def field_differences(
    control_initial: dict[str, np.ndarray],
    control_final: dict[str, np.ndarray],
    perturbed_initial: dict[str, np.ndarray],
    perturbed_final: dict[str, np.ndarray],
    field: FieldSpec,
    phi_index: int,
) -> dict[str, np.ndarray]:
    ci = extract_field(control_initial, field, phi_index)
    cf = extract_field(control_final, field, phi_index)
    pi = extract_field(perturbed_initial, field, phi_index)
    pf = extract_field(perturbed_final, field, phi_index)
    return {
        "perturbed_initial": pi,
        "control_initial": ci,
        "perturbed_final": pf,
        "control_final": cf,
        "perturbed_change": pf - pi,
        "control_change": cf - ci,
        "pair_difference": pf - cf,
    }


def shared_absolute_limit(*values: np.ndarray) -> float:
    result = 0.0
    for value in values:
        array = np.asarray(value, dtype=np.float64)
        if not np.isfinite(array).all():
            raise GalleryError("cannot set a color limit from non-finite values")
        result = max(result, float(np.max(np.abs(array), initial=0.0)))
    return result


def _weights_metrics(
    values: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected = values[mask]
    selected_weights = weights[mask]
    total = float(np.sum(selected_weights))
    if selected.size == 0 or total <= 0.0:
        return {
            "available": False,
            "rms": None,
            "linf": None,
        }
    rms = math.sqrt(
        float(np.sum(selected_weights * selected * selected) / total)
    )
    return {
        "available": True,
        "rms": rms,
        "linf": float(np.max(np.abs(selected), initial=0.0)),
    }


def wall_band_metrics(
    geometry: Geometry,
    differences: dict[str, dict[str, np.ndarray]],
    width_m: float,
) -> dict[str, Any]:
    if not math.isfinite(width_m) or width_m <= 0.0:
        raise GalleryError("physical wall-band width must be positive and finite")
    distances = np.asarray(geometry.wall_distances, dtype=np.float64)
    weights = np.asarray(geometry.volumes, dtype=np.float64)
    if (
        distances.shape != (geometry.nc,)
        or weights.shape != (geometry.nc,)
        or not np.isfinite(distances).all()
        or not np.isfinite(weights).all()
        or np.any(distances < 0.0)
        or np.any(weights <= 0.0)
    ):
        raise GalleryError("native wall distances or volume weights are invalid")
    regions = {
        "wall_band": distances <= width_m,
        "complement": distances > width_m,
    }
    result: dict[str, Any] = {
        "definition": "cell_wall_distance <= wall_width, native cell M0 weights",
        "width_m": float(width_m),
        "regions": {},
    }
    for region_name, mask in regions.items():
        count = int(np.count_nonzero(mask))
        volume = float(np.sum(weights[mask]))
        fields: dict[str, Any] = {}
        for field_key, values in differences.items():
            fields[field_key] = {
                "perturbed_change": _weights_metrics(
                    values["perturbed_change"], weights, mask
                ),
                "control_change": _weights_metrics(
                    values["control_change"], weights, mask
                ),
                "pair_difference": _weights_metrics(
                    values["pair_difference"], weights, mask
                ),
            }
        result["regions"][region_name] = {
            "count": count,
            "volume_m3": volume,
            "fields": fields,
        }
    return result


def _resolve_source_root(source_root: Path | None) -> Path:
    root = (
        Path(__file__).resolve().parents[2]
        if source_root is None
        else Path(source_root)
    )
    root = root.resolve()
    required = (
        "tools/plots/build_a0_report.py",
        "tools/plots/build_a1_t0_report.py",
        "tools/plots/build_a1_t2_report.py",
        "tools/plots/build_050_sections.py",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise GalleryError(f"source root is missing canonical files: {missing}")
    return root


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise GalleryError(f"cannot load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_canonical(
    source_root: Path, inputs: InputIndex
) -> tuple[Any, Any, Any, dict[str, Path]]:
    paths = {
        "build_a0_report.py": source_root / "tools/plots/build_a0_report.py",
        "build_a1_t0_report.py": source_root
        / "tools/plots/build_a1_t0_report.py",
        "build_a1_t2_report.py": source_root
        / "tools/plots/build_a1_t2_report.py",
        "build_050_sections.py": source_root
        / "tools/plots/build_050_sections.py",
    }
    for name, path in paths.items():
        inputs.file(path, f"canonical {name}")
    a1_t2 = _load_module("clt_a1_t2_canonical", paths["build_a1_t2_report.py"])
    a0, t0 = a1_t2.load_sources(source_root)
    sections = _load_module(
        "clt_sections_canonical", paths["build_050_sections.py"]
    )
    for function in ("embed_svg_html", "write_checksums"):
        if not callable(getattr(sections, function, None)):
            raise GalleryError(
                f"canonical build_050_sections.py lacks {function}"
            )
    return a0, t0, sections, paths


def _extract_polygon(points: Any, where: str) -> np.ndarray:
    try:
        result = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise GalleryError(f"{where} must contain numeric R/Z points") from error
    if (
        result.ndim != 2
        or result.shape[1] != 2
        or result.shape[0] < 3
        or not np.isfinite(result).all()
    ):
        raise GalleryError(f"{where} must be a finite (N,2) contour")
    return result


def load_geometry_for_run(
    run: RunEvidence,
    inputs: InputIndex,
    *,
    require_face_arrows: bool,
) -> Geometry:
    path = run.directory / "geometry.npz"
    if not path.is_file():
        raise GalleryError(
            f"missing explicit geometry.npz for run {run.directory}; "
            "native cell polygons and weights will not be reconstructed"
        )
    resolved = inputs.file(
        path, f"run {run.directory.name} explicit native geometry"
    )
    actual_sha256 = inputs._records[resolved]["sha256"]
    if actual_sha256 != run.geometry_sha256:
        raise GalleryError(
            f"geometry.npz SHA256 mismatch for {run.directory}: "
            f"{actual_sha256} != {run.geometry_sha256}"
        )
    arrays = load_snapshot_arrays(path)
    required = (
        "cell_polygon_points",
        "cell_polygon_offsets",
        "cell_polygon_counts",
        "cell_R",
        "cell_Z",
        "M0",
        "physical_wall_distance",
        "input_circle",
        "numerical_polygon",
    )
    missing = [key for key in required if key not in arrays]
    if missing:
        raise GalleryError(
            f"geometry.npz is missing required keys {missing}: {path}"
        )
    flat = np.asarray(arrays["cell_polygon_points"], dtype=np.float64)
    offsets = np.asarray(arrays["cell_polygon_offsets"])
    if (
        flat.ndim != 2
        or flat.shape[1] != 2
        or flat.shape[0] < 3
        or not np.isfinite(flat).all()
    ):
        raise GalleryError(
            f"cell_polygon_points must be finite (N,2): {path}"
        )
    if (
        offsets.ndim != 1
        or offsets.shape != (run.nc + 1,)
        or offsets.dtype.kind not in "iu"
    ):
        raise GalleryError(
            f"cell_polygon_offsets must be integer ({run.nc + 1},): {path}"
        )
    integer_offsets = offsets.astype(np.int64)
    if (
        integer_offsets[0] != 0
        or integer_offsets[-1] != flat.shape[0]
        or np.any(np.diff(integer_offsets) < 3)
    ):
        raise GalleryError(
            f"cell_polygon_offsets are not a valid variable-cell boundary: {path}"
        )
    counts = np.asarray(arrays["cell_polygon_counts"])
    if (
        counts.shape != (run.nc,)
        or counts.dtype.kind not in "iu"
        or not np.array_equal(counts.astype(np.int64), np.diff(integer_offsets))
    ):
        raise GalleryError(
            f"cell_polygon_counts do not match the offsets: {path}"
        )
    polygons = tuple(
        flat[start:stop].copy()
        for start, stop in zip(integer_offsets[:-1], integer_offsets[1:])
    )
    cell_R = np.asarray(arrays["cell_R"], dtype=np.float64)
    cell_Z = np.asarray(arrays["cell_Z"], dtype=np.float64)
    volumes = np.asarray(arrays["M0"], dtype=np.float64)
    distances = np.asarray(
        arrays["physical_wall_distance"], dtype=np.float64
    )
    for name, value in (
        ("cell_R", cell_R),
        ("cell_Z", cell_Z),
        ("M0", volumes),
        ("physical_wall_distance", distances),
    ):
        if value.shape != (run.nc,) or not np.isfinite(value).all():
            raise GalleryError(
                f"geometry.npz {name} must be finite ({run.nc},): {path}"
            )
    contour = _extract_polygon(
        arrays["input_circle"], f"{path}.input_circle"
    )
    chord_wall = _extract_polygon(
        arrays["numerical_polygon"], f"{path}.numerical_polygon"
    )
    if "geometry_fingerprint" in arrays:
        stored = np.asarray(arrays["geometry_fingerprint"])
        if stored.shape != () or str(stored) != run.geometry_fingerprint:
            raise GalleryError(
                f"geometry.npz fingerprint {stored!r} != report "
                f"{run.geometry_fingerprint!r}: {path}"
            )
    face_keys = ("face_coords_R", "face_coords_Z", "face_normals")
    present_faces = [key in arrays for key in face_keys]
    if any(present_faces) and not all(present_faces):
        raise GalleryError(
            f"geometry.npz face center coordinates and normals must be supplied "
            f"together: {path}"
        )
    if require_face_arrows and not all(present_faces):
        raise GalleryError(
            f"--normal-arrows requires geometry.npz keys face_coords_R, "
            f"face_coords_Z and face_normals: {path}"
        )
    if all(present_faces):
        face_R = np.asarray(arrays["face_coords_R"], dtype=np.float64)
        face_Z = np.asarray(arrays["face_coords_Z"], dtype=np.float64)
        face_normals = np.asarray(arrays["face_normals"], dtype=np.float64)
        if (
            face_R.ndim != 1
            or face_Z.shape != face_R.shape
            or not np.isfinite(face_R).all()
            or not np.isfinite(face_Z).all()
        ):
            raise GalleryError(
                f"face_coords_R and face_coords_Z must be finite equal vectors: {path}"
            )
        face_coords = np.column_stack((face_R, face_Z))
        if (
            face_normals.shape != (face_coords.shape[0], 2)
            or not np.isfinite(face_normals).all()
        ):
            raise GalleryError(
                f"face_normals must be finite (Nf,2): {path}"
            )
    else:
        face_coords = np.empty((0, 2), dtype=np.float64)
        face_normals = np.empty((0, 2), dtype=np.float64)
    return Geometry(
        polygons=polygons,
        contour=contour,
        chord_wall=chord_wall,
        centers=np.column_stack((cell_R, cell_Z)),
        volumes=volumes,
        wall_distances=distances,
        face_coords=face_coords,
        face_normals=face_normals,
        fingerprint=run.geometry_fingerprint,
    )


def _validate_geometry(geometry: Geometry, nc: int) -> None:
    if len(geometry.polygons) != nc:
        raise GalleryError(
            f"geometry has {len(geometry.polygons)} cell polygons != Nc {nc}"
        )
    for index, polygon in enumerate(geometry.polygons):
        if (
            polygon.ndim != 2
            or polygon.shape[1] != 2
            or polygon.shape[0] < 3
            or not np.isfinite(polygon).all()
        ):
            raise GalleryError(
                f"cell polygon {index} is not a finite polygon with >=3 vertices"
            )
    _extract_polygon(geometry.contour, "geometry.input_circle")
    _extract_polygon(geometry.chord_wall, "geometry.numerical_polygon")
    if (
        geometry.centers.shape != (nc, 2)
        or geometry.volumes.shape != (nc,)
        or geometry.wall_distances.shape != (nc,)
        or not np.isfinite(geometry.centers).all()
        or not np.isfinite(geometry.volumes).all()
        or not np.isfinite(geometry.wall_distances).all()
        or np.any(geometry.volumes <= 0.0)
        or np.any(geometry.wall_distances < 0.0)
    ):
        raise GalleryError("native geometry centers, measures or distances are invalid")
    if (
        geometry.face_coords.ndim != 2
        or geometry.face_coords.shape[1:] != (2,)
        or geometry.face_normals.shape != (geometry.face_coords.shape[0], 2)
        or not np.isfinite(geometry.face_coords).all()
        or not np.isfinite(geometry.face_normals).all()
    ):
        raise GalleryError("native face coordinates or normals are invalid")


def _same_geometry(left: Geometry, right: Geometry) -> bool:
    if left.fingerprint != right.fingerprint or left.nc != right.nc:
        return False
    if len(left.polygons) != len(right.polygons):
        return False
    return all(
        np.array_equal(a, b)
        for a, b in (
            (left.contour, right.contour),
            (left.chord_wall, right.chord_wall),
            (left.centers, right.centers),
            (left.volumes, right.volumes),
            (left.wall_distances, right.wall_distances),
            (left.face_coords, right.face_coords),
            (left.face_normals, right.face_normals),
        )
    ) and all(
        np.array_equal(a, b)
        for a, b in zip(left.polygons, right.polygons)
    )


def _phase_records(
    pair: MatchedPair, phi_index: int, orthogonal_phase: bool
) -> list[dict[str, Any]]:
    angles = pair.control_final.sample_phis
    if phi_index < 0 or phi_index >= angles.size:
        raise GalleryError(
            f"phi_index {phi_index} is outside Nphi sample range "
            f"[0,{angles.size - 1}]"
        )
    phase_records = [
        {
            "index": phi_index,
            "angle_rad": float(angles[phi_index]),
            "source": "requested --phi-index",
            "selection_note": "selected actual physical sample; no interpolation",
            "orthogonal_offset_rad": 0.0,
        }
    ]
    if pair.control.K >= 1 and orthogonal_phase and angles.size > 1:
        target = float(angles[phi_index] + 0.5 * np.pi)
        circular = np.abs(
            np.angle(np.exp(1j * (angles - target)))
        )
        candidates = [index for index in range(angles.size) if index != phi_index]
        index = min(candidates, key=lambda item: (circular[item], item))
        sampled = float(angles[index])
        offset = target - sampled
        phase_records.append(
            {
                "index": index,
                "angle_rad": sampled,
                "source": "nearest actual retained sample to +90 degrees",
                "selection_note": (
                    "nearest sampled orthogonal phase; no interpolation or "
                    "synthetic phase value"
                ),
                "orthogonal_offset_rad": offset,
            }
        )
    return phase_records


def _limits(geometry: Geometry) -> tuple[float, float, float, float]:
    coords = np.vstack(
        [geometry.contour, geometry.chord_wall, *geometry.polygons]
    )
    return (
        float(np.min(coords[:, 0])),
        float(np.max(coords[:, 0])),
        float(np.min(coords[:, 1])),
        float(np.max(coords[:, 1])),
    )


def _centroid_boundary_normals(
    geometry: Geometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Map explicit face normals to wall-adjacent native cell centers."""
    edges: dict[tuple[float, float, float, float], list[Any]] = {}
    decimals = 10
    for cell, polygon in enumerate(geometry.polygons):
        closed = np.vstack((polygon, polygon[0]))
        for start, stop in zip(closed[:-1], closed[1:]):
            a = (round(float(start[0]), decimals), round(float(start[1]), decimals))
            b = (round(float(stop[0]), decimals), round(float(stop[1]), decimals))
            key = (*a, *b) if a <= b else (*b, *a)
            edges.setdefault(key, []).append((cell, start, stop))
    rmin, rmax, zmin, zmax = _limits(geometry)
    tolerance = 1.0e-8 * max(rmax - rmin, zmax - zmin, 1.0)
    accum = np.zeros((geometry.nc, 2), dtype=np.float64)
    touched = np.zeros(geometry.nc, dtype=bool)
    for records in edges.values():
        if len(records) != 1:
            continue
        cell, start, stop = records[0]
        midpoint = 0.5 * (start + stop)
        distances = np.linalg.norm(geometry.face_coords - midpoint, axis=1)
        face = int(np.argmin(distances))
        if distances[face] > tolerance:
            raise GalleryError(
                f"no explicit native face center matches cell {cell} boundary"
            )
        normal = geometry.face_normals[face].copy()
        if np.dot(normal, midpoint - geometry.centers[cell]) < 0.0:
            normal = -normal
        accum[cell] += normal
        touched[cell] = True
    lengths = np.linalg.norm(accum, axis=1)
    if not np.any(touched):
        raise GalleryError("no wall-adjacent native cell normals are available")
    if np.any(touched & (lengths <= 0.0)):
        raise GalleryError("native cell wall normals are degenerate")
    accum[touched] /= lengths[touched, None]
    return touched, accum


def _draw_cells(
    axis: Any,
    geometry: Geometry,
    values: np.ndarray,
    *,
    norm: Any,
    cmap: str,
    title: str,
    t0: Any,
    normal_arrows: bool,
) -> Any:
    from matplotlib.collections import PolyCollection  # noqa: PLC0415

    if np.asarray(values).shape != (geometry.nc,):
        raise GalleryError(
            f"map values shape {np.asarray(values).shape} != ({geometry.nc},)"
        )
    collection = PolyCollection(
        geometry.polygons,
        array=np.asarray(values, dtype=np.float64),
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        linewidths=0.0,
        antialiaseds=False,
    )
    axis.add_collection(collection)
    contour = np.vstack((geometry.contour, geometry.contour[0]))
    axis.plot(
        contour[:, 0],
        contour[:, 1],
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        label="Input contour",
        zorder=4,
    )
    chord = np.vstack((geometry.chord_wall, geometry.chord_wall[0]))
    if chord.size:
        axis.plot(
            chord[:, 0],
            chord[:, 1],
            color="#f8fafc",
            linewidth=1.6,
            label="Computational chord wall",
            zorder=5,
        )
        axis.plot(
            chord[:, 0],
            chord[:, 1],
            color="#7c2d12",
            linewidth=0.7,
            zorder=6,
        )
    if normal_arrows:
        if geometry.face_coords.size == 0:
            raise GalleryError(
                "--normal-arrows requested but native face arrays are absent"
            )
        width = max(
            _limits(geometry)[1] - _limits(geometry)[0],
            _limits(geometry)[3] - _limits(geometry)[2],
        )
        scale = 0.04 * width
        touched, cell_normals = _centroid_boundary_normals(geometry)
        axis.quiver(
            geometry.centers[touched, 0],
            geometry.centers[touched, 1],
            cell_normals[touched, 0],
            cell_normals[touched, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0 / scale,
            color="#111827",
            width=0.003,
            zorder=7,
        )
    rmin, rmax, zmin, zmax = _limits(geometry)
    pad_r = 0.03 * max(rmax - rmin, 1.0e-12)
    pad_z = 0.03 * max(zmax - zmin, 1.0e-12)
    axis.set_xlim(rmin - pad_r, rmax + pad_r)
    axis.set_ylim(zmin - pad_z, zmax + pad_z)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("R (m)")
    axis.set_ylabel("Z (m)")
    axis.set_title(title)
    axis.legend(loc="lower left", frameon=True, framealpha=0.88)
    t0.style_axis(axis)
    return collection


def _common_norm(limit: float) -> tuple[Any, list[float], str]:
    from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: PLC0415

    if limit > 0.0:
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        return norm, [-limit, limit], "exact zero-centered shared limits"
    # No displayed value differs from zero. A visible unit plotting range does
    # not create field values; the caption records the exact-zero condition.
    return Normalize(vmin=-1.0, vmax=1.0), [-1.0, 1.0], "all three maps are exactly zero"


def _ratio(value: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return value / denominator


def _force_magnitude(
    arrays: dict[str, np.ndarray], key: str, nphi: int, nc: int, phi_index: int
) -> np.ndarray:
    if key not in arrays:
        raise GalleryError(f"missing force-cache array {key!r}")
    value = _physical_values(
        arrays, key, (nphi, 3, nc), Path("initial-rhs.npz")
    )
    selected = value[phi_index]
    return np.sqrt(np.sum(selected * selected, axis=0))


def _load_initial_force(
    pair: MatchedPair,
    inputs: InputIndex,
    phi_index: int,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]] | None]:
    paths = {
        "control": pair.control.directory / "initial-rhs.npz",
        "perturbed": pair.perturbed.directory / "initial-rhs.npz",
    }
    present = {name: path.is_file() for name, path in paths.items()}
    result: dict[str, Any] = {
        "status": "not_supplied",
        "available": False,
        "paths": {name: str(path) for name, path in paths.items()},
        "missing_keys": [],
        "components": {key: None for key in FORCE_KEYS.values()},
        "note": (
            "initial-rhs.npz was not supplied; no force component was "
            "recomputed or zero-filled"
        ),
    }
    if not any(present.values()):
        return result, None
    arrays_by_run: dict[str, dict[str, np.ndarray]] = {}
    missing: set[str] = set()
    for name, path in paths.items():
        if not present[name]:
            missing.update(FORCE_KEYS.values())
            continue
        inputs.file(path, f"pair {pair.label} {name} initial RHS force cache")
        arrays = load_snapshot_arrays(path)
        arrays_by_run[name] = arrays
        missing.update(key for key in FORCE_KEYS.values() if key not in arrays)
    for name, arrays in arrays_by_run.items():
        for key in FORCE_KEYS.values():
            if key in arrays:
                try:
                    _force_magnitude(
                        arrays, key, pair.control.nphi, pair.control.nc, phi_index
                    )
                except GalleryError:
                    missing.add(key)
    result["missing_keys"] = sorted(missing)
    result["status"] = (
        "available" if not missing and len(arrays_by_run) == 2 else "partial"
    )
    result["available"] = result["status"] == "available"
    if not result["available"]:
        result["note"] = (
            "explicit force-cache keys are missing or shape-invalid: "
            + ", ".join(result["missing_keys"])
        )
        return result, None
    component_values: dict[str, Any] = {}
    magnitudes: dict[str, dict[str, np.ndarray]] = {
        "control": {},
        "perturbed": {},
    }
    for component_name, key in FORCE_KEYS.items():
        component_values[key] = {
            run_name: {
                "array_key": key,
                "phi_index": phi_index,
                "shape": list(np.asarray(arrays[key]).shape),
            }
            for run_name, arrays in arrays_by_run.items()
        }
        for run_name, arrays in arrays_by_run.items():
            magnitudes[run_name][component_name] = _force_magnitude(
                arrays, key, pair.control.nphi, pair.control.nc, phi_index
            )
    result["components"] = component_values
    result["note"] = (
        "Six explicit physical SAMPLE force arrays were supplied for both runs; "
        "map magnitudes are derived only from those retained vectors."
    )
    return result, magnitudes


def _run_summary(run: RunEvidence) -> dict[str, Any]:
    acceptance = run.report.get("acceptance")
    diag = run.report.get("diag")
    horizon = run.report.get("horizon")
    failure = run.report.get("failure")
    accepted_time = (
        acceptance.get("accepted_time_s") if isinstance(acceptance, dict) else None
    )
    accepted_step = (
        acceptance.get("accepted_step") if isinstance(acceptance, dict) else None
    )
    wall_elapsed = diag.get("elapsed_s") if isinstance(diag, dict) else None
    failure_view = None
    if isinstance(failure, dict):
        failure_view = {
            "code": failure.get("code"),
            "message": failure.get("message"),
            "last_accepted_step": failure.get("last_accepted_step"),
        }
    return {
        "directory": str(run.directory),
        "status": run.status,
        "scope_label": _scope_label(run.status),
        "accepted_step": accepted_step,
        "accepted_time_s": accepted_time,
        "wall_elapsed_s": wall_elapsed,
        "requested_target_step": (
            acceptance.get("target_step")
            if isinstance(acceptance, dict)
            else None
        ),
        "requested_horizon_s": (
            horizon.get("expected_global_horizon_s")
            if isinstance(horizon, dict)
            else None
        ),
        "failure": failure_view,
        "device": run.parameters.get("device"),
        "report_sha256": run.report_sha256,
    }


def _scope_label(status: str) -> str:
    return {
        "completed": "completed requested horizon",
        "segment_completed": "accepted segment only; global horizon incomplete",
        "budget_stopped": "budget-stopped at last accepted state",
        "failed": "failed after last accepted state; partial scope",
    }[status]


def _pair_scope(pair: MatchedPair) -> str:
    return (
        f"Matched at physical t={pair.matched_time_s:.12e} s, step "
        f"{pair.matched_step}. Control: {_scope_label(pair.control.status)}. "
        f"Perturbed: {_scope_label(pair.perturbed.status)}. "
        "No terminal-horizon or scientific pass claim is made."
    )


def _excited_harmonic(pair: MatchedPair) -> int:
    control = pair.control.parameters.get("mode")
    perturbed = pair.perturbed.parameters.get("mode")
    if isinstance(control, bool) or not isinstance(control, int) or control < 0:
        raise GalleryError(
            "actual control parameters.mode must be a nonnegative integer"
        )
    if perturbed != control:
        raise GalleryError("control and perturbed parameters.mode differ")
    if control > pair.control.K:
        raise GalleryError(
            f"actual excited harmonic mode={control} exceeds retained "
            f"K={pair.control.K}"
        )
    return control


def _run_provenance_text(run: RunEvidence) -> str:
    acceptance = run.report.get("acceptance")
    diag = run.report.get("diag")
    accepted_step = (
        acceptance.get("accepted_step") if isinstance(acceptance, dict) else None
    )
    accepted_time = (
        acceptance.get("accepted_time_s") if isinstance(acceptance, dict) else None
    )
    wall_elapsed = diag.get("elapsed_s") if isinstance(diag, dict) else None
    failure = run.report.get("failure")
    if isinstance(failure, dict):
        failure_text = (
            f"{failure.get('code')}: {failure.get('message')}; "
            f"last accepted step={failure.get('last_accepted_step')}"
        )
    else:
        failure_text = "none recorded"
    source = run.source_fingerprint or "unavailable"
    return (
        f"Provenance for {run.directory.name}: status={run.status}; "
        f"source_sha256={source}; report_sha256={run.report_sha256}; "
        f"accepted_step={accepted_step}; accepted_time_s={accepted_time}; "
        f"wall_elapsed_s={wall_elapsed}; failure={failure_text}."
    )


def _figure_record(
    number: int,
    *,
    pair: MatchedPair,
    phase: dict[str, Any],
    title: str,
    caption: str,
    base: str,
    components: list[str],
    panels: list[dict[str, Any]],
    question: str,
    controls: str,
    criterion: str,
) -> dict[str, Any]:
    devices = sorted(
        {
            str(pair.control.parameters.get("device")),
            str(pair.perturbed.parameters.get("device")),
        }
    )
    return {
        "number": number,
        "title": title,
        "caption": caption,
        "source_kind": "nonlinear_evolution",
        "components": components,
        "status": _pair_scope(pair),
        "device": " / ".join(devices),
        "question": question,
        "controls": controls,
        "criterion": criterion,
        "grid": {"n_r": pair.control.nodes, "n_z": pair.control.nodes},
        "n_phi": _excited_harmonic(pair),
        "k": pair.control.K,
        "n_phi_quadrature": pair.control.nphi,
        "image": f"figures/{base}.png",
        "svg": f"figures/{base}.svg",
        "data": "data.json",
        "provenance": "provenance.json",
        "panels": panels,
        "pair": pair.label,
        "phi_index": phase["index"],
        "phi_angle_rad": phase["angle_rad"],
    }


def _plot_field_figure(
    *,
    number: int,
    pair: MatchedPair,
    geometry: Geometry,
    phase: dict[str, Any],
    field: FieldSpec,
    differences: dict[str, np.ndarray],
    figures_dir: Path,
    t0: Any,
    plt: Any,
    normal_arrows: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from matplotlib.ticker import ScalarFormatter  # noqa: PLC0415

    arrays = [
        differences["perturbed_change"],
        differences["control_change"],
        differences["pair_difference"],
    ]
    limit = shared_absolute_limit(*arrays)
    norm, limits, limit_note = _common_norm(limit)
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.4))
    titles = (
        "Perturbed(t) - perturbed(0)",
        "Control(t) - control(0)",
        "Perturbed(t) - control(t)",
    )
    collections = []
    for axis, values, title in zip(axes, arrays, titles):
        collection = _draw_cells(
            axis,
            geometry,
            values,
            norm=norm,
            cmap="RdBu_r",
            title=f"{field.key}: {title}",
            t0=t0,
            normal_arrows=normal_arrows,
        )
        collections.append(collection)
    for axis, letter in zip(axes, "ABC"):
        t0.panel_label(axis, letter)
    colorbar = figure.colorbar(
        collections[-1],
        ax=axes.tolist(),
        fraction=0.026,
        pad=0.025,
    )
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 4))
    colorbar.formatter = formatter
    colorbar.update_ticks()
    colorbar.set_label(
        f"Physical field ({field.unit}), one shared symmetric scale"
    )
    base = (
        f"fig_{number:03d}_{_safe_name(pair.label)}_"
        f"phi{phase['index']:03d}_{_safe_name(field.key)}"
    )
    figure.subplots_adjust(
        left=0.055, right=0.84, bottom=0.14, top=0.88, wspace=0.18
    )
    t0.save_figure(figure, figures_dir, base)
    plt.close(figure)
    peak_values = {
        "perturbed_change": float(
            np.max(np.abs(differences["perturbed_change"]), initial=0.0)
        ),
        "control_change": float(
            np.max(np.abs(differences["control_change"]), initial=0.0)
        ),
        "pair_difference": float(
            np.max(np.abs(differences["pair_difference"]), initial=0.0)
        ),
    }
    background = float(
        np.max(np.abs(differences["control_initial"]), initial=0.0)
    )
    seed = float(
        np.max(
            np.abs(
                differences["perturbed_initial"]
                - differences["control_initial"]
            ),
            initial=0.0,
        )
    )
    peak = max(peak_values.values())
    caption = (
        f"{field.title} in {field.unit}. True t=0 subtraction for pair "
        f"{pair.label}, matched t={pair.matched_time_s:.9e} s at step "
        f"{pair.matched_step}; phi index {phase['index']}, angle "
        f"{phase['angle_rad']:.9e} rad; n_phi={_excited_harmonic(pair)}, "
        f"K={pair.control.K}, Nphi="
        f"{pair.control.nphi} distinct samples, Nc={geometry.nc}. "
        f"Shared absolute range [{limits[0]:.9e}, {limits[1]:.9e}] {field.unit}; "
        f"{limit_note}. Absolute peaks: perturbed change "
        f"{peak_values['perturbed_change']:.9e}, control change "
        f"{peak_values['control_change']:.9e}, pair difference "
        f"{peak_values['pair_difference']:.9e}. Initial control background "
        f"{background:.9e}; initial pair seed {seed:.9e}; peak/background "
        f"{_fmt_ratio(_ratio(peak, background))}; peak/seed "
        f"{_fmt_ratio(_ratio(peak, seed))} {field.unit}. Native polygon cells "
        "are shown without interpolation. Input contour and Computational "
        "chord wall are overlaid."
    )
    panels = [
        {
            "letter": letter,
            "description": description,
            "formula": formula,
            "limits": limits,
            "unit": field.unit,
            "max_abs": maximum,
        }
        for letter, description, formula, maximum in (
            (
                "A",
                "Perturbed final minus perturbed true initial.",
                "perturbed(t) - perturbed(0)",
                peak_values["perturbed_change"],
            ),
            (
                "B",
                "Control final minus control true initial.",
                "control(t) - control(0)",
                peak_values["control_change"],
            ),
            (
                "C",
                "Perturbed final minus control final at the same time.",
                "perturbed(t) - control(t)",
                peak_values["pair_difference"],
            ),
        )
    ]
    record = _figure_record(
        number,
        pair=pair,
        phase=phase,
        title=f"{pair.label}: {field.title} comparisons",
        caption=caption,
        base=base,
        components=[field.key],
        panels=panels,
        question=(
            f"How do perturbed change, control change and their matched "
            f"difference compare for {field.key}?"
        ),
        controls=(
            "True physical t=0; latest common saved physical time; one shared "
            "symmetric color scale."
        ),
        criterion="Renderer evidence only; no convergence or physics pass.",
    )
    data = {
        "title": field.title,
        "unit": field.unit,
        "array_key": field.array_key,
        "component": field.component,
        "phi_index": phase["index"],
        "phi_angle_rad": phase["angle_rad"],
        "shared_absolute_limit": limit,
        "panels": panels,
        "difference_check": {
            "perturbed_change_max_abs": peak_values["perturbed_change"],
            "control_change_max_abs": peak_values["control_change"],
            "pair_difference_max_abs": peak_values["pair_difference"],
            "initial_control_background_max_abs": background,
            "initial_pair_seed_max_abs": seed,
            "peak_over_background": _ratio(peak, background),
            "peak_over_seed": _ratio(peak, seed),
        },
        "figure": f"figures/{base}.svg",
    }
    return record, data


def _plot_raw_b_figure(
    *,
    number: int,
    pair: MatchedPair,
    geometry: Geometry,
    phase: dict[str, Any],
    control_initial: dict[str, np.ndarray],
    control_final: dict[str, np.ndarray],
    figures_dir: Path,
    t0: Any,
    plt: Any,
    normal_arrows: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from matplotlib.ticker import ScalarFormatter  # noqa: PLC0415

    values = {
        "initial": {
            field.key: extract_field(control_initial, field, phase["index"])
            for field in MAGNETIC_FIELDS
        },
        "final": {
            field.key: extract_field(control_final, field, phase["index"])
            for field in MAGNETIC_FIELDS
        },
    }
    limit = shared_absolute_limit(
        *(value for group in values.values() for value in group.values())
    )
    norm, limits, limit_note = _common_norm(limit)
    figure, axes = plt.subplots(3, 2, figsize=(10.8, 11.8))
    collections = []
    for row, field in enumerate(MAGNETIC_FIELDS):
        for column, (time_name, time_text) in enumerate(
            (("initial", "true t=0"), ("final", "matched final"))
        ):
            collection = _draw_cells(
                axes[row, column],
                geometry,
                values[time_name][field.key],
                norm=norm,
                cmap="RdBu_r",
                title=f"Control {field.key} at {time_text}",
                t0=t0,
                normal_arrows=normal_arrows,
            )
            collections.append(collection)
    panel_letters = iter("ABCDEF")
    for row in range(3):
        for column in range(2):
            t0.panel_label(axes[row, column], next(panel_letters))
    colorbar = figure.colorbar(
        collections[-1],
        ax=axes.ravel().tolist(),
        fraction=0.026,
        pad=0.025,
    )
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 4))
    colorbar.formatter = formatter
    colorbar.update_ticks()
    colorbar.set_label(
        "Control magnetic field (T), one shared absolute symmetric scale"
    )
    base = (
        f"fig_{number:03d}_{_safe_name(pair.label)}_"
        f"phi{phase['index']:03d}_raw_control_B"
    )
    figure.subplots_adjust(
        left=0.065, right=0.82, bottom=0.055, top=0.96,
        hspace=0.20, wspace=0.08,
    )
    t0.save_figure(figure, figures_dir, base)
    plt.close(figure)
    caption = (
        f"Raw control magnetic fields B_R, B_phi and B_Z in T for "
        f"{pair.label}. True t=0 is compared with matched final "
        f"t={pair.matched_time_s:.9e} s, step {pair.matched_step}, phi index "
        f"{phase['index']} at {phase['angle_rad']:.9e} rad. One shared absolute "
        f"symmetric range [{limits[0]:.9e}, {limits[1]:.9e}] T supports the "
        f"toroidal field scale; {limit_note}. n_phi={_excited_harmonic(pair)}, "
        f"Nphi={pair.control.nphi} distinct samples, K={pair.control.K}, "
        f"Nc={geometry.nc}. Actual native polygon "
        "cells are rendered without interpolation."
    )
    panel_letters = iter("ABCDEF")
    panels = [
        {
            "letter": next(panel_letters),
            "description": (
                f"Control {field.key} at "
                f"{'true t=0' if column == 0 else 'matched final'}."
            ),
            "limits": limits,
            "unit": "T",
        }
        for row, field in enumerate(MAGNETIC_FIELDS)
        for column in "AB"
    ]
    record = _figure_record(
        number,
        pair=pair,
        phase=phase,
        title=f"{pair.label}: raw control magnetic fields",
        caption=caption,
        base=base,
        components=[field.key for field in MAGNETIC_FIELDS],
        panels=panels,
        question="What is the absolute raw magnetic scale at initial and final time?",
        controls="One shared absolute symmetric scale across all six control maps.",
        criterion="Raw evidence only; no eigenvector or mode-amplitude claim.",
    )
    data = {
        "shared_absolute_limit": limit,
        "panels": panels,
        "figure": f"figures/{base}.svg",
        "components": [field.key for field in MAGNETIC_FIELDS],
    }
    return record, data


def _plot_force_figure(
    *,
    number: int,
    pair: MatchedPair,
    geometry: Geometry,
    phase: dict[str, Any],
    magnitudes: dict[str, dict[str, np.ndarray]],
    figures_dir: Path,
    t0: Any,
    plt: Any,
    normal_arrows: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from matplotlib.colors import Normalize  # noqa: PLC0415

    names = list(FORCE_KEYS)
    figure, axes = plt.subplots(6, 2, figsize=(10.8, 21.5))
    panel_records = []
    panel_letters = iter("ABCDEFGHIJKL")
    for row, name in enumerate(names):
        limit = shared_absolute_limit(
            magnitudes["control"][name], magnitudes["perturbed"][name]
        )
        norm = Normalize(vmin=0.0, vmax=max(limit, 1.0e-300))
        for column, run_name in enumerate(("control", "perturbed")):
            _draw_cells(
                axes[row, column],
                geometry,
                magnitudes[run_name][name],
                norm=norm,
                cmap="viridis",
                title=f"{run_name.title()} |F_{name}| at true t=0",
                t0=t0,
                normal_arrows=normal_arrows,
            )
            letter = next(panel_letters)
            t0.panel_label(axes[row, column], letter)
            panel_records.append(
                {
                    "letter": letter,
                    "description": (
                        f"{run_name.title()} ion {name} force magnitude at "
                        f"true t=0 from the explicit retained vector."
                    ),
                    "unit": "N m^-3",
                }
            )
    base = (
        f"fig_{number:03d}_{_safe_name(pair.label)}_"
        f"phi{phase['index']:03d}_initial_force_cache"
    )
    figure.subplots_adjust(
        left=0.055, right=0.985, bottom=0.025, top=0.985,
        hspace=0.24, wspace=0.10,
    )
    t0.save_figure(figure, figures_dir, base)
    plt.close(figure)
    caption = (
        f"Actual initial ion force/cache components in machine momentum-density "
        f"units N m^-3 for pair {pair.label}, true t=0, phi index "
        f"{phase['index']} at {phase['angle_rad']:.9e} rad. Panels use the six "
        "explicit retained vector arrays; map magnitude is "
        "sqrt(F_R^2+F_phi^2+F_Z^2). No term was reconstructed and no missing "
        "term was filled with zero. Each component row has its own shared "
        "absolute scale between control and perturbed runs."
    )
    record = _figure_record(
        number,
        pair=pair,
        phase=phase,
        title=f"{pair.label}: retained initial ion force/cache",
        caption=caption,
        base=base,
        components=list(FORCE_KEYS.values()),
        panels=panel_records,
        question="Which explicit initial force/cache vectors were retained?",
        controls="Both runs at true t=0; no derived pressure or missing-term fill.",
        criterion="Availability and scale evidence only; no net-force pass claim.",
    )
    data = {
        "figure": f"figures/{base}.svg",
        "components": list(FORCE_KEYS.values()),
    }
    return record, data


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return result.strip("_") or "item"


def _fmt_ratio(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.9e}"


def _write_json(path: Path, value: Any) -> None:
    try:
        text = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            ensure_ascii=True,
        )
    except (TypeError, ValueError) as error:
        raise GalleryError(f"cannot serialize JSON {path}: {error}") from error
    path.write_text(text + "\n", encoding="utf-8")


def _safe_output(
    output: Path,
    pairs: tuple[PairSpec, ...],
    source_root: Path,
) -> Path:
    target = output.expanduser().resolve()
    if target.exists():
        raise GalleryError(f"output must be a new output directory: {target}")
    source = source_root.resolve()
    protected = [spec.control.resolve() for spec in pairs] + [
        spec.perturbed.resolve() for spec in pairs
    ]
    protected.append(source)
    for root in protected:
        if target.is_relative_to(root):
            raise GalleryError(f"output {target} is inside input root {root}")
    if not target.name:
        raise GalleryError("output directory name is empty")
    return target


def _manifests(
    *,
    data: dict[str, Any],
    figure_records: list[dict[str, Any]],
    page_records: list[dict[str, Any]],
    input_sha256: list[dict[str, Any]],
    canonical_sha256: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    notation = [
        {
            "symbol": "t",
            "meaning": "Actual physical time in seconds from snapshot metadata.",
        },
        {
            "symbol": "Nphi",
            "meaning": "Number of distinct retained toroidal sample angles.",
        },
        {
            "symbol": "K",
            "meaning": "Maximum retained toroidal harmonic index.",
        },
        {
            "symbol": "Nc",
            "meaning": "Number of actual active native polygon cells.",
        },
        {
            "symbol": "Delta",
            "meaning": (
                "Paired subtraction at true initial and matched final physical "
                "times; no interpolation."
            ),
        },
    ]
    conclusions = (
        "This evidence-gated gallery renders caller-supplied arrays only. It "
        "does not establish numerical convergence, physical wall accuracy or "
        "scientific acceptance."
    )
    manifest = {
        "schema": SCHEMA,
        "title": "Native polygon CLT boundary gallery",
        "introduction": (
            "Offline side-by-side maps of control, perturbed and paired native "
            "polygon-cell evidence. All displayed values come from explicit "
            "snapshots; no raster interpolation or missing-term fill is used."
        ),
        "notation": notation,
        "archive": {"path": "data.json", "label": "Machine-readable gallery data"},
        "pages": page_records,
        "conclusions": conclusions,
        "known_issues": [],
        "input_sha256": input_sha256,
        "canonical_renderer_sha256": canonical_sha256,
        "figure_count": len(figure_records),
        "input_origin": INPUT_ORIGIN,
        "scientific_acceptance": SCIENTIFIC_ACCEPTANCE,
    }
    return manifest, data


def build(
    pairs: Iterable[PairSpec],
    output: Path,
    *,
    source_root: Path | None = None,
    wall_band_fraction: float = 0.1,
    phi_index: int = 0,
    orthogonal_phase: bool = True,
    normal_arrows: bool = False,
) -> dict[str, Any]:
    """Validate inputs, render a new gallery atomically, and return its data."""
    specs = tuple(
        PairSpec(
            label=str(spec.label),
            control=Path(spec.control),
            perturbed=Path(spec.perturbed),
        )
        for spec in pairs
    )
    if not specs:
        raise GalleryError("at least one --pair is required")
    if (
        not math.isfinite(wall_band_fraction)
        or wall_band_fraction <= 0.0
    ):
        raise GalleryError("--wall-band must be a finite positive fraction")
    root = _resolve_source_root(source_root)
    target = _safe_output(Path(output), specs, root)
    inputs = InputIndex()
    producer = Path(__file__).resolve()
    inputs.file(producer, "gallery producer source")
    matched = match_pairs(specs, inputs)
    phase_sets = [
        _phase_records(pair, phi_index, orthogonal_phase) for pair in matched
    ]
    maximum_figures = sum(
        len(phases) * (len(FIELDS) + 1) + 1 for phases in phase_sets
    )
    if maximum_figures > MAX_FIGURES:
        raise GalleryError(
            f"input would require more than {MAX_FIGURES} figures "
            f"({maximum_figures}); reduce --pair inputs or disable the extra "
            "orthogonal phase"
        )
    for pair in matched:
        _initial_snapshot(pair.control)
        _initial_snapshot(pair.perturbed)
    a0, t0, sections, canonical_paths = _load_canonical(root, inputs)
    plt = t0.configure_matplotlib()

    geometries: list[Geometry] = []
    for pair in matched:
        geometry = load_geometry_for_run(
            pair.control, inputs, require_face_arrows=normal_arrows
        )
        perturbed_geometry = load_geometry_for_run(
            pair.perturbed, inputs, require_face_arrows=normal_arrows
        )
        _validate_geometry(geometry, pair.control.nc)
        _validate_geometry(perturbed_geometry, pair.perturbed.nc)
        if geometry.fingerprint != pair.control.geometry_fingerprint:
            raise GalleryError(
                f"geometry fingerprint {geometry.fingerprint!r} != report "
                f"{pair.control.geometry_fingerprint!r} for pair {pair.label}"
            )
        if not _same_geometry(geometry, perturbed_geometry):
            raise GalleryError(
                f"explicit geometry.npz arrays differ within pair {pair.label}"
            )
        geometries.append(geometry)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
    )
    try:
        figures_dir = temporary / "figures"
        figures_dir.mkdir()
        figure_records: list[dict[str, Any]] = []
        page_records: list[dict[str, Any]] = []
        pair_data: dict[str, Any] = {}
        all_figure_names: list[str] = []
        number = 0

        for pair, geometry, phases in zip(
            matched, geometries, phase_sets, strict=True
        ):
            control_initial = load_snapshot_arrays(pair.control_initial.path)
            control_final = load_snapshot_arrays(pair.control_final.path)
            perturbed_initial = load_snapshot_arrays(pair.perturbed_initial.path)
            perturbed_final = load_snapshot_arrays(pair.perturbed_final.path)
            nphi = pair.control.nphi
            nc = pair.control.nc
            expected_dims = {
                "B_N_samples": (nphi, 3, nc),
                "ion_velocity_samples": (nphi, 3, nc),
                "ion_density_samples": (nphi, nc),
                "ion_perp_pressure_samples": (nphi, nc),
                "electron_density_samples": (nphi, nc),
                "electron_perp_pressure_samples": (nphi, nc),
                "electron_parallel_pressure_samples": (nphi, nc),
            }
            for snapshot_name, snapshot_arrays in (
                ("control initial", control_initial),
                ("control final", control_final),
                ("perturbed initial", perturbed_initial),
                ("perturbed final", perturbed_final),
            ):
                for key, shape in expected_dims.items():
                    if key not in snapshot_arrays:
                        raise GalleryError(
                            f"{snapshot_name} is missing physical SAMPLE array "
                            f"{key!r}"
                        )
                    _physical_values(
                        snapshot_arrays,
                        key,
                        shape,
                        pair.control_initial.path,
                    )

            primary_phase = phases[0]
            phase_fields: dict[str, Any] = {}
            primary_differences: dict[str, dict[str, np.ndarray]] = {}
            pair_figures: list[dict[str, Any]] = []
            page_figures: list[dict[str, Any]] = []
            for phase in phases:
                differences_for_phase: dict[str, dict[str, np.ndarray]] = {}
                fields_for_phase: dict[str, Any] = {}
                for field in FIELDS:
                    differences = field_differences(
                        control_initial,
                        control_final,
                        perturbed_initial,
                        perturbed_final,
                        field,
                        phase["index"],
                    )
                    differences_for_phase[field.key] = differences
                    number += 1
                    record, field_data = _plot_field_figure(
                        number=number,
                        pair=pair,
                        geometry=geometry,
                        phase=phase,
                        field=field,
                        differences=differences,
                        figures_dir=figures_dir,
                        t0=t0,
                        plt=plt,
                        normal_arrows=normal_arrows,
                    )
                    figure_records.append(record)
                    page_figures.append(record)
                    pair_figures.append(record)
                    all_figure_names.append(record["svg"])
                    fields_for_phase[field.key] = field_data
                number += 1
                raw_record, raw_data = _plot_raw_b_figure(
                    number=number,
                    pair=pair,
                    geometry=geometry,
                    phase=phase,
                    control_initial=control_initial,
                    control_final=control_final,
                    figures_dir=figures_dir,
                    t0=t0,
                    plt=plt,
                    normal_arrows=normal_arrows,
                )
                figure_records.append(raw_record)
                page_figures.append(raw_record)
                pair_figures.append(raw_record)
                all_figure_names.append(raw_record["svg"])
                phase["fields"] = fields_for_phase
                phase["raw_control_B"] = raw_data
                if phase is primary_phase:
                    phase_fields = fields_for_phase
                    primary_differences = differences_for_phase

            widths = [
                _required_positive_number(
                    pair.control.parameters.get("minor_radius"),
                    f"pair {pair.label}.parameters.minor_radius",
                )
                for _ in (0,)
            ]
            width_m = wall_band_fraction * widths[0]
            band = wall_band_metrics(geometry, primary_differences, width_m)

            force_data, force_magnitudes = _load_initial_force(
                pair, inputs, primary_phase["index"]
            )
            force_notes = []
            if force_magnitudes is not None:
                number += 1
                force_record, force_figure_data = _plot_force_figure(
                    number=number,
                    pair=pair,
                    geometry=geometry,
                    phase=primary_phase,
                    magnitudes=force_magnitudes,
                    figures_dir=figures_dir,
                    t0=t0,
                    plt=plt,
                    normal_arrows=normal_arrows,
                )
                figure_records.append(force_record)
                page_figures.append(force_record)
                pair_figures.append(force_record)
                all_figure_names.append(force_record["svg"])
                force_data["figure"] = force_figure_data["figure"]
                force_notes.append(
                    "Initial force/cache vectors were read from explicit arrays "
                    "in initial-rhs.npz."
                )
            else:
                force_notes.append(
                    "Initial force/cache unavailable: "
                    + (
                        "initial-rhs.npz was not supplied."
                        if force_data["status"] == "not_supplied"
                        else "missing explicit keys "
                        + ", ".join(force_data["missing_keys"])
                        + "."
                    )
                )

            band_region = band["regions"]["wall_band"]
            band_note = (
                f"Fixed physical wall band width {width_m:.9e} m "
                f"({wall_band_fraction:g} of minor radius). "
                + (
                    f"Unavailable: no cells lie in the band "
                    f"(count={band_region['count']}, "
                    f"volume={band_region['volume_m3']:.9e} m^3)."
                    if band_region["count"] == 0
                    else f"{band_region['count']} cells, volume "
                    f"{band_region['volume_m3']:.9e} m^3."
                )
            )
            no_gs_note = (
                "No GS flux contours are drawn: no explicit retained GS field "
                "artifact was supplied, and none was invented."
            )
            notes = [
                _pair_scope(pair),
                _run_provenance_text(pair.control),
                _run_provenance_text(pair.perturbed),
                band_note,
                *force_notes,
                no_gs_note,
                (
                    "All physical arrays are selected at actual sampled toroidal "
                    "angles. No Cartesian raster interpolation or full-cell "
                    "rectangle substitution was used."
                ),
                (
                    "This page is renderer evidence only; it makes no convergence, "
                    "wall-correctness or scientific acceptance claim."
                ),
            ]
            page_records.append(
                {
                    "id": _safe_name(pair.label),
                    "title": f"Pair: {pair.label}",
                    "introduction": _pair_scope(pair),
                    "notes": notes,
                    "figures": page_figures,
                }
            )
            pair_data[pair.label] = {
                "status": _pair_scope(pair),
                "matched_step": pair.matched_step,
                "matched_time_s": pair.matched_time_s,
                "n_phi": _excited_harmonic(pair),
                "K": pair.control.K,
                "Nphi_distinct": pair.control.nphi,
                "Nc": pair.control.nc,
                "geometry_fingerprint": geometry.fingerprint,
                "control": _run_summary(pair.control),
                "perturbed": _run_summary(pair.perturbed),
                "phases": phases,
                "fields": phase_fields,
                "wall_band": band,
                "initial_force_cache": force_data,
                "figures": [record["svg"] for record in pair_figures],
                "notes": notes,
            }

        input_records = inputs.records()
        canonical_sha256 = {
            name: sha256_file(path) for name, path in canonical_paths.items()
        }
        data: dict[str, Any] = {
            "schema": SCHEMA,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "input_origin": INPUT_ORIGIN,
            "scientific_acceptance": SCIENTIFIC_ACCEPTANCE,
            "scope": DATA_SCOPE,
            "figure_count": len(figure_records),
            "figures": all_figure_names,
            "figure_records": figure_records,
            "input_sha256": input_records,
            "canonical_sha256": canonical_sha256,
            "array_semantics": ARRAY_SEMANTICS,
            "force_array_names": FORCE_KEYS,
            "pairs": pair_data,
        }
        manifest, data = _manifests(
            data=data,
            figure_records=figure_records,
            page_records=page_records,
            input_sha256=input_records,
            canonical_sha256=canonical_sha256,
        )
        _write_json(temporary / "manifest.json", manifest)
        _write_json(temporary / "data.json", data)
        provenance = {
            "schema": f"{SCHEMA}-provenance",
            "producer": str(producer),
            "producer_sha256": sha256_file(producer),
            "canonical_sha256": canonical_sha256,
            "input_sha256": input_records,
            "run_status_and_timing": {
                label: {
                    "control": pair["control"],
                    "perturbed": pair["perturbed"],
                }
                for label, pair in pair_data.items()
            },
            "figure_count": len(figure_records),
        }
        _write_json(temporary / "provenance.json", provenance)
        rendered = a0.render_report(
            manifest,
            temporary / "manifest.json",
            temporary / "index.html",
        )
        embedded = sections.embed_svg_html(rendered, temporary)
        (temporary / "index.html").write_text(embedded, encoding="utf-8")
        if re.search(r'src="(?!data:)', embedded):
            raise GalleryError("HTML retains a non-embedded external image source")
        if "Native polygon cells" not in embedded:
            raise GalleryError("HTML lacks the native-polygon evidence caption")
        inputs.verify_unchanged()
        sections.write_checksums(temporary)
        os.replace(temporary, target)
        return data
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("LABEL", "CONTROL_DIR", "PERTURBED_DIR"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--phi-index", type=int, default=0)
    parser.add_argument("--wall-band", type=float, default=0.1)
    parser.add_argument("--normal-arrows", action="store_true")
    parser.add_argument(
        "--no-orthogonal-phase",
        action="store_true",
        help="Do not add the nearest sampled +90 degree phase for K>=1.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        build(
            [
                PairSpec(label, Path(control), Path(perturbed))
                for label, control, perturbed in arguments.pair
            ],
            arguments.output,
            source_root=arguments.source_root,
            wall_band_fraction=arguments.wall_band,
            phi_index=arguments.phi_index,
            orthogonal_phase=not arguments.no_orthogonal_phase,
            normal_arrows=arguments.normal_arrows,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
