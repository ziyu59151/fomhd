#!/usr/bin/env python3
"""Offline A0 convergence and complex mode analysis.

The tool reads solver-produced run directories without modifying them.  All
report paths are no-clobber outputs and reports include the provenance hashes
that were actually checked.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
PHASE_CONVENTION = "exp(-i*omega*t)"
OBSERVABLE_UNITS = {
    "n_i": "1/m^3",
    "n_e": "1/m^3",
    "p_i": "Pa",
    "p_perp_e": "Pa",
    "p_parallel_e": "Pa",
    "mom_i": "kg/(m^2*s)",
    "mom_e": "kg/(m^2*s)",
    "B": "T",
}
NORM_DEFINITIONS = {
    "weighted_L2": "sqrt(sum(weights * abs(value)^2))",
    "weighted_RMS": "sqrt(sum(weights * abs(value)^2) / sum(weights))",
    "linf": "max(abs(value))",
}


class AnalysisError(ValueError):
    """Input data or descriptor cannot support the requested analysis."""


def _reject_constant(value: str) -> None:
    raise AnalysisError(f"non-finite JSON number is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnalysisError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except AnalysisError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read JSON {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise AnalysisError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _minify_original_json(raw: bytes) -> bytes:
    """Validate JSON, then remove whitespace only outside JSON strings."""

    try:
        text = raw.decode("utf-8")
        json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except AnalysisError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot validate original UTF-8 JSON: {exc}") from exc
    output: list[str] = []
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
            output.append(character)
        elif character not in " \t\r\n":
            output.append(character)
    return "".join(output).encode("utf-8")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise AnalysisError(f"{name} must be finite")
    return result


def _positive(value: Any, name: str) -> float:
    result = _number(value, name)
    if result <= 0.0:
        raise AnalysisError(f"{name} must be positive")
    return result


def _same(a: Any, b: Any, tolerance: float = 1.0e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def _safe_existing(path: Path, name: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise AnalysisError(f"{name} is missing: {path}") from exc
    if not resolved.is_file() and not resolved.is_dir():
        raise AnalysisError(f"{name} is not a regular path: {path}")
    return resolved


def _safe_relative(root: Path, relative: Any, name: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise AnalysisError(f"{name} must be a non-empty relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise AnalysisError(f"{name} escapes its provenance root: {relative}")
    return root / candidate


def _verified_hash(
    path: Path, expected: Any, records: dict[str, str], name: str
) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise AnalysisError(f"{name} lacks a SHA-256 provenance hash")
    actual = _sha256(_safe_existing(path, name))
    if actual != expected:
        raise AnalysisError(f"{name} SHA-256 mismatch: {path}")
    records[str(path)] = actual


@dataclass(frozen=True)
class RunData:
    directory: Path
    manifest: dict[str, Any]
    resolved: dict[str, Any]
    paths: dict[str, Path]
    hashes: dict[str, str]
    source_files: dict[str, str]
    history: list[dict[str, str]]
    config_hash_verification: dict[str, Any]

    @property
    def model(self) -> str:
        return str(self.manifest.get("model"))

    @property
    def units(self) -> str:
        return str(self.manifest.get("units"))


def load_run(run_dir: str | Path) -> RunData:
    """Load and verify one accepted runtime directory with schema v1."""

    directory = _safe_existing(Path(run_dir), "run directory")
    if not directory.is_dir():
        raise AnalysisError(f"run path is not a directory: {directory}")
    paths = {
        "run.json": directory / "run.json",
        "inputs/resolved.json": directory / "inputs/resolved.json",
        "inputs/case.toml": directory / "inputs/case.toml",
        "inputs/geometry.npz": directory / "inputs/geometry.npz",
        "inputs/initial.npz": directory / "inputs/initial.npz",
        "inputs/source_manifest.json": directory / "inputs/source_manifest.json",
        "history.csv": directory / "history.csv",
        "observations.jsonl": directory / "observations.jsonl",
    }
    for name, path in paths.items():
        _safe_existing(path, name)

    manifest = _read_json(paths["run.json"])
    resolved = _read_json(paths["inputs/resolved.json"])
    if not isinstance(manifest, dict) or not isinstance(resolved, dict):
        raise AnalysisError("run metadata must be JSON objects")
    if manifest.get("schema_version") != 1:
        raise AnalysisError("unsupported run schema_version")
    if resolved.get("schema_version", 1) != 1:
        raise AnalysisError("unsupported resolved-config schema_version")
    if manifest.get("model") != "A0" or resolved.get("model") != "A0":
        raise AnalysisError("run and resolved config must both declare model A0")
    if manifest.get("units") != "SI" or resolved.get("units") != "SI":
        raise AnalysisError("run and resolved config must both use units SI")
    resolved_raw = paths["inputs/resolved.json"].read_bytes()
    resolved_config_sha256 = hashlib.sha256(resolved_raw).hexdigest()
    minified_resolved = _minify_original_json(resolved_raw)
    minified_config_hash = hashlib.sha256(minified_resolved).hexdigest()
    anchors = []
    for container, key in ((manifest, "resolved_config_sha256"),
                           (manifest.get("provenance"), "resolved_config_sha256"),
                           (manifest.get("input_hashes"), "resolved")):
        if isinstance(container, dict) and key in container:
            anchors.append(container[key])
    if any(anchor != resolved_config_sha256 for anchor in anchors):
        raise AnalysisError("resolved-file SHA-256 does not match its manifest anchor")
    anchored_resolved_hash = anchors[0] if anchors else None
    if minified_config_hash == manifest.get("config_hash"):
        config_hash_verification = {
            "status": "PASS",
            "method": "lexical_minification_of_original_utf8",
            "portable_across_json_serializers": True,
            "resolved_config_sha256": resolved_config_sha256,
            "minified_config_sha256": minified_config_hash,
            "note": "Whitespace was removed outside strings only; token spellings and key order were preserved.",
        }
    elif anchored_resolved_hash is not None:
        if anchored_resolved_hash != resolved_config_sha256:
            raise AnalysisError("resolved-file SHA-256 does not match its manifest anchor")
        config_hash_verification = {
            "status": "PASS",
            "method": "resolved_file_sha256",
            "portable_across_json_serializers": True,
            "resolved_config_sha256": resolved_config_sha256,
            "minified_config_sha256": minified_config_hash,
        }
    else:
        raise AnalysisError(
            "config_hash does not match lexical minification of the original "
            "resolved UTF-8 JSON and no resolved-file SHA-256 anchor is present"
        )
    if resolved.get("input_hashes") != manifest.get("input_hashes"):
        raise AnalysisError("resolved and manifest input_hashes disagree")

    status = str(manifest.get("status", ""))
    if status in {"initializing", "failed", "failed_numerical_trial", "io_failed"}:
        raise AnalysisError(f"run has no analyzable accepted result: status={status}")
    if "io_failed" in status:
        raise AnalysisError(f"run output is not provenance-complete: status={status}")
    if manifest.get("last_accepted_step") is None:
        raise AnalysisError("run has no accepted step")

    input_hashes = manifest.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise AnalysisError("run manifest lacks input_hashes")
    input_paths = {
        "case": paths["inputs/case.toml"],
        "geometry": paths["inputs/geometry.npz"],
        "initial": paths["inputs/initial.npz"],
        "source": paths["inputs/source_manifest.json"],
    }
    hashes: dict[str, str] = {}
    for name, path in input_paths.items():
        _verified_hash(path, input_hashes.get(name), hashes, f"input {name}")

    build = manifest.get("build")
    if not isinstance(build, dict):
        raise AnalysisError("run manifest lacks build provenance")
    source_files = build.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise AnalysisError("run manifest lacks measured source-file hashes")
    source_root = directory / "provenance/source"
    for relative, expected in source_files.items():
        source_path = _safe_relative(source_root, relative, "source snapshot")
        _verified_hash(source_path, expected, hashes, f"source snapshot {relative}")

    columns = str(manifest.get("history_columns", "")).split(",")
    if not columns or any(not column for column in columns):
        raise AnalysisError("run manifest lacks a valid history_columns schema")
    try:
        with paths["history.csv"].open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != columns:
                raise AnalysisError("history.csv header does not match run.json")
            history = list(reader)
    except AnalysisError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise AnalysisError(f"cannot parse history.csv: {exc}") from exc
    if not history:
        raise AnalysisError("history.csv has no accepted rows")
    previous_step: int | None = None
    for row in history:
        try:
            step = int(row["step"])
            time_value = float(row["time_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AnalysisError("history.csv row has an invalid step/time") from exc
        if not math.isfinite(time_value):
            raise AnalysisError("history.csv contains a non-finite accepted time")
        if previous_step is not None and step != previous_step + 1:
            raise AnalysisError("history.csv accepted steps are not contiguous")
        previous_step = step
    try:
        manifest_step = int(manifest["last_accepted_step"])
        manifest_time = float(manifest["last_accepted_time"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError("run manifest has invalid last accepted identity") from exc
    if history[-1]["step"] != str(manifest_step) or not _same(
        history[-1]["time_s"], manifest_time
    ):
        raise AnalysisError("run manifest and history.csv disagree on last accepted data")

    for name in (
        "run.json",
        "inputs/resolved.json",
        "history.csv",
        "observations.jsonl",
    ):
        hashes[str(paths[name])] = _sha256(paths[name])
    return RunData(
        directory=directory,
        manifest=manifest,
        resolved=resolved,
        paths=paths,
        hashes=hashes,
        source_files={str(k): str(v) for k, v in source_files.items()},
        history=history,
        config_hash_verification=config_hash_verification,
    )


def _case_identity(run: RunData) -> tuple[Any, Any, Any, Any]:
    source = run.manifest.get("source", {})
    return (
        run.resolved.get("case_name"),
        run.resolved.get("species"),
        run.resolved.get("boundary"),
        (
            source.get("provider_name"),
            source.get("provider_version"),
        )
        if isinstance(source, dict)
        else None,
    )


def _resolved_value(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise AnalysisError(f"scan variable is absent from resolved config: {path}")
        value = value[component]
    return value


def _physical_config_hash(run: RunData) -> str:
    projection = {
        key: value
        for key, value in run.resolved.items()
        if key
        not in {
            "grid",
            "time",
            "io",
            "diagnostics",
            "resolved_paths",
            "input_hashes",
        }
    }
    return _canonical_hash(projection)


def _provenance(run: RunData) -> dict[str, Any]:
    return {
        "run_directory": str(run.directory),
        "run_id": run.manifest.get("run_id"),
        "manifest_status": run.manifest.get("status"),
        "verified": True,
        "verified_hashes": dict(sorted(run.hashes.items())),
        "source_files": dict(sorted(run.source_files.items())),
        "run_manifest_sha256": run.hashes[str(run.paths["run.json"])],
        "resolved_config_sha256": run.hashes[
            str(run.paths["inputs/resolved.json"])
        ],
        "history_csv_sha256": run.hashes[str(run.paths["history.csv"])],
        "observations_jsonl_sha256": run.hashes[
            str(run.paths["observations.jsonl"])
        ],
        "config_hash": run.manifest.get("config_hash"),
        "config_hash_replayed": run.config_hash_verification["status"] == "PASS",
        "config_hash_verification": run.config_hash_verification,
        "semantic_hash": run.manifest.get("semantic_hash"),
        "geometry_hash": run.manifest.get("geometry_hash"),
        "config_hash_verified": run.config_hash_verification["status"] == "PASS",
    }


def _normalize_region(observable: dict[str, Any]) -> dict[str, list[float]]:
    region = observable.get("region")
    if not isinstance(region, dict):
        raise AnalysisError("observable.region must be an explicit physical region")
    result: dict[str, list[float]] = {}
    for axis in ("R", "Z"):
        bounds = region.get(axis)
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise AnalysisError(f"observable.region.{axis} must be [minimum, maximum]")
        low = _number(bounds[0], f"region {axis} minimum")
        high = _number(bounds[1], f"region {axis} maximum")
        if low >= high:
            raise AnalysisError(f"observable.region.{axis} must have minimum < maximum")
        result[axis] = [low, high]
    return result


def _snapshot_time(path: Path) -> float:
    try:
        with np.load(path, allow_pickle=False) as data:
            return float(np.asarray(data["time"]).item())
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise AnalysisError(f"cannot read snapshot time from {path}: {exc}") from exc


def _select_snapshot(
    run: RunData,
    level: dict[str, Any],
    target: float | None,
    time_tolerance: float,
) -> Path:
    explicit = level.get("snapshot")
    if explicit is not None:
        snapshot = _safe_relative(run.directory, explicit, "snapshot")
        if target is not None and not _same(
            _snapshot_time(snapshot), target, time_tolerance
        ):
            raise AnalysisError("level snapshot time differs from comparison time")
        return _safe_existing(snapshot, "snapshot")
    candidates = sorted((run.directory / "snapshots").glob("state_*.npz"))
    if not candidates:
        raise AnalysisError(f"run has no accepted state snapshot: {run.directory}")
    if target is None:
        times = [_snapshot_time(path) for path in candidates]
        if max(times) - min(times) > time_tolerance:
            raise AnalysisError(
                "multiple snapshot times require an explicit common comparison time"
            )
        return candidates[-1]
    matches = [
        path
        for path in candidates
        if _same(_snapshot_time(path), target, time_tolerance)
    ]
    if len(matches) != 1:
        raise AnalysisError(
            f"expected one snapshot at time {target}, found {len(matches)}"
        )
    return matches[0]


@dataclass(frozen=True)
class FieldLayout:
    key: str
    role: str
    R: np.ndarray
    Z: np.ndarray
    active: np.ndarray
    quadrature: np.ndarray
    component: int | None


def _inside_polygon(R: float, Z: float, pr: np.ndarray, pz: np.ndarray) -> bool:
    # Match the certified half-open predicate used by geometry/mesh.cpp.
    cr, cz = np.longdouble(R), np.longdouble(Z)
    inside = False
    for index in range(len(pr)):
        a_r, a_z = np.longdouble(pr[index]), np.longdouble(pz[index])
        b_r, b_z = np.longdouble(pr[(index + 1) % len(pr)]), np.longdouble(pz[(index + 1) % len(pz)])
        if (a_z <= cz < b_z) or (b_z <= cz < a_z):
            tie = ((a_r == b_r and cr == a_r) or
                   (a_z == b_z and cz == a_z) or
                   (cr == a_r and cz == a_z) or (cr == b_r and cz == b_z))
            side = np.longdouble(0) if tie else (b_r - a_r) * (cz - a_z) - (b_z - a_z) * (cr - a_r)
            bound = 16 * np.finfo(np.longdouble).eps * (
                (abs(b_r) + abs(a_r)) * (abs(cz) + abs(a_z)) +
                (abs(b_z) + abs(a_z)) * (abs(cr) + abs(a_r)))
            if not tie and (not np.isfinite(side) or abs(side) <= bound):
                raise AnalysisError("GeometryAmbiguous: orientation sign cannot be certified")
            if (b_z > a_z and side > 0.0) or (b_z < a_z and side < 0.0):
                inside = not inside
    return inside


def _field_layout(run: RunData, observable: dict[str, Any]) -> FieldLayout:
    key = observable.get("key")
    if not isinstance(key, str) or not key:
        raise AnalysisError("observable.key must name a snapshot array")
    component = observable.get("component")
    component_index: int | None = None
    if key in {"mom_i", "mom_e"}:
        names = {"R": 0, "phi": 1, "Z": 2}
        if isinstance(component, str) and component in names:
            component_index = names[component]
        elif isinstance(component, int) and not isinstance(component, bool):
            component_index = component
        else:
            raise AnalysisError(f"{key} requires component R, phi, Z, or 0/1/2")
        if component_index not in {0, 1, 2}:
            raise AnalysisError(f"{key} component is outside 0, 1, 2")
        role = "cell"
    elif key in {"n_i", "n_e", "p_i", "p_perp_e", "p_parallel_e"}:
        if component not in {None, "scalar"}:
            raise AnalysisError(f"{key} is scalar and must use component null")
        role = "cell"
    elif key == "B":
        component_map = {"R": "B_R", "phi": "B_phi", "Z": "B_Z"}
        if component not in component_map:
            raise AnalysisError("B requires component R, phi, or Z")
        key = component_map[component]
        role = {"B_R": "radial_x_zface", "B_phi": "rface_x_zface", "B_Z": "rface_x_zcell"}[key]
        component_index = None
    elif key in {"B_R", "B_phi", "B_Z"}:
        if component is not None:
            raise AnalysisError("explicit B component key must use component null")
        role = {"B_R": "radial_x_zface", "B_phi": "rface_x_zface", "B_Z": "rface_x_zcell"}[key]
    else:
        raise AnalysisError(f"unsupported snapshot observable key: {key}")

    try:
        with np.load(run.paths["inputs/geometry.npz"], allow_pickle=False) as geometry:
            R = np.asarray(geometry["R_nodes"], dtype=np.float64)
            Z = np.asarray(geometry["Z_nodes"], dtype=np.float64)
            polygon_R = np.asarray(geometry["lcfs_R"], dtype=np.float64)
            polygon_Z = np.asarray(geometry["lcfs_Z"], dtype=np.float64)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise AnalysisError(f"cannot read physical geometry: {exc}") from exc
    if (
        R.ndim != 1
        or Z.ndim != 1
        or polygon_R.ndim != 1
        or polygon_Z.ndim != 1
        or polygon_R.shape != polygon_Z.shape
        or len(R) < 2
        or len(Z) < 2
        or len(polygon_R) < 3
        or np.any(np.diff(R) <= 0.0)
        or np.any(np.diff(Z) <= 0.0)
    ):
        raise AnalysisError("physical geometry coordinates are invalid")
    h_r = float(R[1] - R[0])
    h_z = float(Z[1] - Z[0])
    cells = np.asarray(
        [
            [_inside_polygon(float(r), float(z), polygon_R, polygon_Z) for r in R]
            for z in Z
        ],
        dtype=bool,
    )
    if not np.any(cells):
        raise AnalysisError("physical geometry contains no active cells")

    factor = 2.0 * math.pi
    if role == "cell":
        active = cells
        quadrature = factor * R[None, :] * h_r * h_z * np.ones((len(Z), 1))
        coords_R, coords_Z = R, Z
    elif role == "radial_x_zface":
        coords_R = R
        coords_Z = np.zeros(len(Z) + 1, dtype=np.float64) + Z[0] + (
            np.arange(len(Z) + 1) - 0.5
        ) * h_z
        active = np.zeros((len(Z) + 1, len(R)), dtype=bool)
        quadrature = np.zeros_like(active, dtype=np.float64)
        for f in range(len(Z) + 1):
            lower = cells[f - 1] if f > 0 else np.zeros(len(R), dtype=bool)
            upper = cells[f] if f < len(Z) else np.zeros(len(R), dtype=bool)
            active[f] = lower | upper
            both = lower & upper
            lz = np.where(both, h_z, 0.5 * h_z)
            quadrature[f] = factor * R * h_r * lz
    elif role == "rface_x_zcell":
        coords_R = R[0] + (np.arange(len(R) + 1) - 0.5) * h_r
        coords_Z = Z
        active = np.zeros((len(Z), len(R) + 1), dtype=bool)
        quadrature = np.zeros_like(active, dtype=np.float64)
        for e in range(len(R) + 1):
            left = cells[:, e - 1] if e > 0 else np.zeros(len(Z), dtype=bool)
            right = cells[:, e] if e < len(R) else np.zeros(len(Z), dtype=bool)
            active[:, e] = left | right
            both = left & right
            lr = np.where(both, h_r, 0.5 * h_r)
            face_R = float(coords_R[e])
            cell_R = np.asarray(
                [
                    R[e - 1] if left[j] else R[e] if right[j] else face_R
                    for j in range(len(Z))
                ]
            )
            radial = np.where(both, face_R, 0.5 * (face_R + cell_R))
            quadrature[:, e] = factor * radial * h_z * lr
    else:
        coords_R = R[0] + (np.arange(len(R) + 1) - 0.5) * h_r
        coords_Z = np.zeros(len(Z) + 1, dtype=np.float64) + Z[0] + (
            np.arange(len(Z) + 1) - 0.5
        ) * h_z
        active = np.zeros((len(Z) + 1, len(R) + 1), dtype=bool)
        quadrature = np.zeros_like(active, dtype=np.float64)
        for f in range(len(Z) + 1):
            for e in range(len(R) + 1):
                adjacent = []
                for dj in (-1, 0):
                    for di in (-1, 0):
                        j, i = f + dj, e + di
                        if 0 <= j < len(Z) and 0 <= i < len(R) and cells[j, i]:
                            adjacent.append((i, j))
                if not adjacent:
                    continue
                active[f, e] = True
                for i, j in adjacent:
                    r_cell = float(R[i])
                    r_half = float(coords_R[e])
                    r_lo = min(r_cell, r_half)
                    r_hi = max(r_cell, r_half)
                    quadrature[f, e] += factor * (
                        (r_hi * r_hi - r_lo * r_lo) * h_z / 4.0
                    )
    return FieldLayout(
        key=key,
        role=role,
        R=coords_R,
        Z=coords_Z,
        active=active,
        quadrature=quadrature,
        component=component_index,
    )


def _snapshot_field(
    run: RunData,
    snapshot_path: Path,
    observable: dict[str, Any],
) -> tuple[np.ndarray, FieldLayout, float, int, dict[str, Any]]:
    layout = _field_layout(run, observable)
    key = layout.key
    mode = observable.get("mode", 0)
    if isinstance(mode, bool) or not isinstance(mode, int):
        raise AnalysisError("observable.mode must be an integer")

    snapshot_hash = _sha256(snapshot_path)
    try:
        with np.load(snapshot_path, allow_pickle=False) as data:
            required = {
                "accepted",
                "schema_version",
                "layout_version",
                "model_id",
                "config_hash",
                "semantic_hash",
                "geometry_hash",
                "step",
                "time",
                key,
            }
            missing = sorted(required.difference(data.files))
            if missing:
                raise AnalysisError(
                    f"snapshot lacks required fields: {', '.join(missing)}"
                )
            if not bool(np.asarray(data["accepted"]).item()):
                raise AnalysisError("snapshot is not accepted data")
            if int(np.asarray(data["schema_version"]).item()) != 1:
                raise AnalysisError("snapshot has unsupported schema_version")
            if int(np.asarray(data["layout_version"]).item()) != int(
                run.manifest["layout_version"]
            ):
                raise AnalysisError("snapshot layout_version differs from run")
            if str(np.asarray(data["model_id"]).item()) != run.model:
                raise AnalysisError("snapshot model differs from run")
            if str(np.asarray(data["config_hash"]).item()) != run.manifest.get(
                "config_hash"
            ):
                raise AnalysisError("snapshot config_hash differs from run")
            if str(np.asarray(data["semantic_hash"]).item()) != run.manifest.get(
                "semantic_hash"
            ):
                raise AnalysisError("snapshot semantic_hash differs from run")
            if str(np.asarray(data["geometry_hash"]).item()) != run.manifest.get(
                "geometry_hash"
            ):
                raise AnalysisError("snapshot geometry_hash differs from run")
            time_value = float(np.asarray(data["time"]).item())
            step = int(np.asarray(data["step"]).item())
            array = np.array(data[key], copy=True)
    except AnalysisError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise AnalysisError(f"cannot read snapshot {snapshot_path}: {exc}") from exc

    if layout.component is not None:
        if array.ndim != 4:
            raise AnalysisError("observable.component is outside the array layout")
        array = array[layout.component]
    if array.ndim != 3:
        raise AnalysisError(
            "snapshot observable must use [mode, iz, ir] layout after component selection"
        )
    try:
        K = int(run.resolved["grid"]["K"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError("resolved config lacks integer grid.K") from exc
    mode_index = mode + K
    if mode_index < 0 or mode_index >= array.shape[0]:
        raise AnalysisError("observable.mode is outside the snapshot mode layout")
    array = np.asarray(array[mode_index], dtype=np.complex128)
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise AnalysisError("snapshot observable contains non-finite values")
    if array.shape != layout.active.shape:
        raise AnalysisError(
            f"snapshot {key} shape {array.shape} disagrees with "
            f"{layout.role} shape {layout.active.shape}"
        )

    history_match = any(
        int(row["step"]) == step and float(row["time_s"]) == time_value
        for row in run.history
    )
    if not history_match:
        raise AnalysisError("snapshot is not an accepted history row")
    metadata = {
        "path": str(snapshot_path),
        "sha256": snapshot_hash,
        "accepted": True,
        "step": step,
        "time_s": time_value,
        "layout": "[mode, iz, ir]",
        "layout_role": layout.role,
        "mode": mode,
        "component": layout.component,
        "active_entities": int(np.sum(layout.active)),
    }
    return array, layout, time_value, step, metadata


def _weights(values: dict[str, Any], layout: FieldLayout) -> np.ndarray:
    kind = values.get("kind")
    if kind == "uniform":
        return np.ones_like(layout.active, dtype=np.float64)
    if kind == "radial":
        power = _number(values.get("power", 1.0), "radial weight power")
        if power < 0.0:
            raise AnalysisError("radial weight power must be nonnegative")
        return np.broadcast_to(
            np.power(layout.R, power)[None, :], layout.active.shape
        ).astype(np.float64)
    if kind == "a0_quadrature":
        return layout.quadrature
    raise AnalysisError(
        "norm.weights.kind must be 'uniform', 'radial', or 'a0_quadrature'"
    )


def _region_mask(
    R: np.ndarray, Z: np.ndarray, region: dict[str, list[float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    RR, ZZ = np.meshgrid(R, Z)
    mask = (
        (RR >= region["R"][0])
        & (RR <= region["R"][1])
        & (ZZ >= region["Z"][0])
        & (ZZ <= region["Z"][1])
    )
    if not np.any(mask):
        raise AnalysisError("fixed physical comparison region contains no grid points")
    return mask, RR, ZZ


def _norm(
    values: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    kind: str,
) -> tuple[float, int]:
    selected = np.asarray(values[mask])
    if kind == "linf":
        return float(np.max(np.abs(selected))), int(selected.size)
    if kind not in NORM_DEFINITIONS:
        raise AnalysisError(
            "norm.kind must be 'weighted_L2', 'weighted_RMS', or 'linf'"
        )
    selected_weights = np.asarray(weights[mask], dtype=np.float64)
    total_weight = float(np.sum(selected_weights))
    if total_weight <= 0.0 or not math.isfinite(total_weight):
        raise AnalysisError("norm weights must have positive finite total")
    weighted_square = float(
        np.sum(selected_weights * np.abs(selected) ** 2)
    )
    if kind == "weighted_L2":
        value = math.sqrt(weighted_square)
    else:
        value = math.sqrt(weighted_square / total_weight)
    return value, int(selected.size)


def _analytic(
    reference: dict[str, Any],
    RR: np.ndarray,
    ZZ: np.ndarray,
    time_value: float,
) -> np.ndarray:
    kind = reference.get("kind")
    amplitude = _number(reference.get("amplitude"), "reference.amplitude")
    phase = _number(reference.get("phase", 0.0), "reference.phase")
    if kind == "plane_wave":
        k_r = _number(reference.get("k_R"), "reference.k_R")
        k_z = _number(reference.get("k_Z"), "reference.k_Z")
        omega = _number(reference.get("omega", 0.0), "reference.omega")
        return (amplitude * np.exp(1j * (k_r * RR + k_z * ZZ - omega * time_value + phase))).astype(
            np.complex128
        )
    if kind == "constant":
        return np.full(
            RR.shape, amplitude * np.exp(1j * phase), dtype=np.complex128
        )
    raise AnalysisError("reference.kind must be 'plane_wave', 'constant', or 'none'")


def _interp_axis(
    values: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    axis: int,
) -> np.ndarray:
    real = np.apply_along_axis(
        lambda column: np.interp(target, source, column.real), axis, values
    )
    imag = np.apply_along_axis(
        lambda column: np.interp(target, source, column.imag), axis, values
    )
    return real + 1j * imag


def _map_bilinear(
    values: np.ndarray,
    source_R: np.ndarray,
    source_Z: np.ndarray,
    target_R: np.ndarray,
    target_Z: np.ndarray,
) -> np.ndarray:
    if (
        target_R[0] < source_R[0]
        or target_R[-1] > source_R[-1]
        or target_Z[0] < source_Z[0]
        or target_Z[-1] > source_Z[-1]
    ):
        raise AnalysisError("comparison grid lies outside a source snapshot grid")
    along_R = _interp_axis(values, source_R, target_R, axis=1)
    return _interp_axis(along_R, source_Z, target_Z, axis=0)


def _order_record(
    value: float | None,
    status: str,
    reason: str = "",
    **extra: Any,
) -> dict[str, Any]:
    result = {"value": value, "status": status}
    if reason:
        result["reason"] = reason
    result.update(extra)
    return result


def _bisect_order(
    h0: float,
    h1: float,
    h2: float,
    difference_ratio: float,
    interval: list[float],
) -> tuple[float | None, str]:
    low = _number(interval[0], "order_interval minimum")
    high = _number(interval[1], "order_interval maximum")
    if low >= high:
        raise AnalysisError("order_interval must have minimum < maximum")
    ratio_01 = h0 / h1
    ratio_12 = h1 / h2
    if ratio_01 <= 1.0 or ratio_12 <= 1.0:
        raise AnalysisError("unequal Richardson requires h0 > h1 > h2")
    log_01 = math.log(ratio_01)
    log_12 = math.log(ratio_12)

    def residual(p: float) -> float:
        # Work in scale-free refinement ratios. Subtracting nearly equal
        # dimensionless h**p values at SI sizes loses all significant digits.
        numerator = math.expm1(p * log_01)
        denominator = -math.expm1(-p * log_12)
        if denominator == 0.0:
            return math.nan
        return numerator / denominator - difference_ratio

    points = np.linspace(low, high, 513)
    previous_p = float(points[0])
    previous_f = residual(previous_p)
    for raw_p in points[1:]:
        p = float(raw_p)
        f = residual(p)
        if math.isfinite(previous_f) and math.isfinite(f):
            if abs(previous_f) <= 1.0e-13:
                return previous_p, "continuous sign bracket"
            if abs(f) <= 1.0e-13:
                return p, "continuous sign bracket"
            if previous_f * f < 0.0:
                left, right = previous_p, p
                for _ in range(100):
                    middle = 0.5 * (left + right)
                    fm = residual(middle)
                    if abs(fm) <= 1.0e-13 or right - left <= 1.0e-13:
                        return middle, "continuous sign bracket"
                    if residual(left) * fm <= 0.0:
                        right = middle
                    else:
                        left = middle
                return 0.5 * (left + right), "continuous sign bracket"
        previous_p, previous_f = p, f
    return None, "no continuous sign bracket in the declared admissible interval"


def _write_report(result: dict[str, Any], output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    temporary = output.parent / f".{output.name}.tmp.{os.getpid()}.{id(result)}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite existing analysis: {output}"
            ) from None
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _output_is_in_inputs(output: Path, inputs: list[Path]) -> bool:
    target = output.resolve()
    for item in inputs:
        root = item.resolve()
        if target == root or root in target.parents:
            return True
    return False


def analyze_convergence(
    descriptor: str | Path | dict[str, Any],
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze analytic errors or three-level Richardson convergence."""

    if isinstance(descriptor, dict):
        value = descriptor
        base = Path.cwd()
        descriptor_path: Path | None = None
    else:
        descriptor_path = _safe_existing(Path(descriptor), "comparison descriptor")
        value = _read_json(descriptor_path)
        base = descriptor_path.parent
    if not isinstance(value, dict):
        raise AnalysisError("comparison descriptor must be a JSON object")
    if value.get("schema_version") != 1:
        raise AnalysisError("comparison descriptor schema_version must be 1")
    scan_variable = value.get("scan_variable")
    if not isinstance(scan_variable, str) or not scan_variable.strip():
        raise AnalysisError("exactly one non-empty scan variable is required")
    if scan_variable not in ("time.dt", "grid.hR", "grid.hZ"):
        raise AnalysisError("supported scan variables are time.dt, grid.hR, and grid.hZ")
    case_id = value.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise AnalysisError("comparison descriptor requires case_id")
    levels = value.get("levels")
    if not isinstance(levels, list) or len(levels) != 3:
        raise AnalysisError(
            "convergence requires exactly three levels; adjacent-triple analysis is not supported"
        )
    floor = _number(value.get("roundoff_floor"), "roundoff_floor")
    if floor < 0.0:
        raise AnalysisError("roundoff_floor must be nonnegative")
    observable_scale_value = value.get("observable_scale")
    if observable_scale_value is None:
        roundoff_guard = {
            "status": "UNAVAILABLE",
            "reason": "no observable_scale was supplied; only the absolute floor applies",
            "absolute_floor": floor,
            "effective_norm": floor,
        }
        effective_floor = floor
    else:
        observable_scale = _number(
            observable_scale_value, "observable_scale"
        )
        if observable_scale < 0.0:
            raise AnalysisError("observable_scale must be nonnegative")
        roundoff_relative = _number(
            value.get("roundoff_relative", 64.0 * np.finfo(np.float64).eps),
            "roundoff_relative",
        )
        if roundoff_relative < 0.0:
            raise AnalysisError("roundoff_relative must be nonnegative")
        relative_floor = observable_scale * roundoff_relative
        effective_floor = max(floor, relative_floor)
        roundoff_guard = {
            "status": "PASS",
            "observable_scale": observable_scale,
            "roundoff_relative": roundoff_relative,
            "relative_floor": relative_floor,
            "absolute_floor": floor,
            "effective_norm": effective_floor,
            "definition": "max(roundoff_floor, observable_scale * roundoff_relative)",
        }
    time_tolerance = _positive(
        value.get("time_tolerance", 1.0e-12), "time_tolerance"
    )
    target_time = (
        None
        if value.get("time_s") is None
        else _number(value.get("time_s"), "time_s")
    )
    observable = value.get("observable")
    if not isinstance(observable, dict):
        raise AnalysisError("comparison descriptor requires observable metadata")
    observable_name = observable.get("name")
    observable_unit = observable.get("unit")
    if not isinstance(observable_name, str) or not observable_name:
        raise AnalysisError("observable.name must be non-empty")
    if not isinstance(observable_unit, str) or not observable_unit:
        raise AnalysisError("observable.unit must be non-empty")
    expected_unit = _observable_unit({"field": observable.get("key")})
    if expected_unit is None or observable_unit != expected_unit:
        raise AnalysisError("observable.unit differs from the physical field unit")
    region = _normalize_region(observable)
    norm_descriptor = observable.get("norm")
    if not isinstance(norm_descriptor, dict):
        raise AnalysisError("observable.norm must be explicit")
    norm_kind = norm_descriptor.get("kind")
    if norm_kind not in NORM_DEFINITIONS:
        raise AnalysisError("observable.norm.kind is unsupported")
    weight_descriptor = norm_descriptor.get("weights", {"kind": "uniform"})
    if not isinstance(weight_descriptor, dict):
        raise AnalysisError("observable.norm.weights must be an object")
    declared_norm_unit = norm_descriptor.get("unit")
    if declared_norm_unit is not None and (
        not isinstance(declared_norm_unit, str) or not declared_norm_unit
    ):
        raise AnalysisError("observable.norm.unit must be non-empty when supplied")
    if (
        norm_kind == "weighted_L2"
        and weight_descriptor.get("kind") == "a0_quadrature"
    ):
        norm_unit = f"{observable_unit} * m^(3/2)"
    elif norm_kind == "weighted_L2" and weight_descriptor.get("kind") == "radial":
        power = _number(weight_descriptor.get("power", 1.0), "radial weight power")
        norm_unit = (f"{observable_unit} * m^({power / 2:g})"
                     if power != 0.0 else observable_unit)
    else:
        norm_unit = observable_unit
    if declared_norm_unit is not None and declared_norm_unit != norm_unit:
        raise AnalysisError("observable.norm.unit differs from the derived norm unit")

    reference = value.get("reference")
    if not isinstance(reference, dict):
        raise AnalysisError("comparison descriptor requires reference metadata")
    reference_kind = reference.get("kind")
    if reference_kind not in {"none", "plane_wave", "constant"}:
        raise AnalysisError("reference.kind is unsupported")
    if reference_kind != "none" and reference.get("unit") != observable_unit:
        raise AnalysisError("analytic reference unit differs from observable unit")

    loaded: list[
        tuple[
            RunData,
            dict[str, Any],
            np.ndarray,
            FieldLayout,
            float,
            int,
            dict[str, Any],
            dict[str, Any],
        ]
    ] = []
    refinements: list[float] = []
    run_directories: set[Path] = set()
    observed_times: list[float] = []
    expected_identity: tuple[Any, Any, Any, Any] | None = None
    for index, level in enumerate(levels):
        if not isinstance(level, dict) or not isinstance(level.get("run"), str):
            raise AnalysisError(f"level {index} lacks a run directory")
        run_path = Path(level["run"])
        if not run_path.is_absolute():
            run_path = base / run_path
        run = load_run(run_path)
        if run.directory in run_directories:
            raise AnalysisError("convergence levels must use distinct run directories")
        run_directories.add(run.directory)
        identity = _case_identity(run)
        if expected_identity is None:
            expected_identity = identity
            if identity[0] != case_id:
                raise AnalysisError(
                    f"run case identity {identity[0]!r} differs from case_id {case_id!r}"
                )
        elif identity != expected_identity:
            raise AnalysisError("runs have incompatible model/case/source identity")
        if run.model != "A0" or run.units != "SI":
            raise AnalysisError("runs must all use model A0 and units SI")
        h = _positive(level.get("h"), f"level {index} refinement")
        resolved_scan = _number(
            _resolved_value(run.resolved, scan_variable),
            f"level {index} resolved {scan_variable}",
        )
        if not math.isclose(h, resolved_scan, rel_tol=1.0e-12, abs_tol=0.0):
            raise AnalysisError(
                f"level {index} scan value {h!r} differs from resolved "
                f"{scan_variable}={resolved_scan!r}"
            )
        if any(math.isclose(h, previous, rel_tol=1.0e-14, abs_tol=0.0) for previous in refinements):
            raise AnalysisError("refinement levels must be distinct")
        refinements.append(h)
        snapshot_path = _select_snapshot(
            run, level, target_time, time_tolerance
        )
        (
            field,
            layout,
            time_value,
            step,
            snapshot_metadata,
        ) = _snapshot_field(run, snapshot_path, observable)
        observed_times.append(time_value)
        mask, RR, ZZ = _region_mask(layout.R, layout.Z, region)
        mask &= layout.active
        if not np.any(mask):
            raise AnalysisError(
                "fixed physical comparison region contains no active entities"
            )
        weights = _weights(weight_descriptor, layout)
        level_metadata = {
            "index": index,
            "h": h,
            "scan_variable": scan_variable,
            "scan_value": h,
            "time_s": time_value,
            "step": step,
            "snapshot": snapshot_metadata,
            "provenance": _provenance(run),
            "observable": {
                "name": observable_name,
                "key": observable.get("key"),
                "component": observable.get("component"),
                "mode": observable.get("mode", 0),
                "unit": observable_unit,
            },
        }
        loaded.append(
            (
                run,
                level,
                field,
                layout,
                time_value,
                step,
                level_metadata,
                {"mask": mask, "RR": RR, "ZZ": ZZ, "weights": weights},
            )
        )

    if target_time is None and max(observed_times) - min(observed_times) > time_tolerance:
        raise AnalysisError("all levels must use the same physical output time")

    physical_hashes = {_physical_config_hash(entry[0]) for entry in loaded}
    if len(physical_hashes) != 1:
        raise AnalysisError(
            "runs have incompatible physical case configuration outside the scan"
        )
    def require_fixed(section: str, key: str, label: str | None = None) -> None:
        values = [
            entry[0].resolved[section][key]
            for entry in loaded
        ]
        first = values[0]
        for value in values[1:]:
            if value != first:
                raise AnalysisError(
                    f"{label or f'{section}.{key}'} cannot drift outside scan_variable"
                )

    if scan_variable == "time.dt":
        for key in sorted(loaded[0][0].resolved["grid"]):
            require_fixed("grid", key, f"grid.{key}")
        require_fixed("time", "start_time")
        require_fixed("time", "initial_kind")
    elif scan_variable.startswith("grid."):
        require_fixed("grid", "K")
        require_fixed("grid", "Nphi")
        require_fixed("time", "dt")
        require_fixed("time", "start_time")
        require_fixed("time", "initial_kind")
        transverse_key = "hZ" if scan_variable == "grid.hR" else "hR"
        transverse = [
            float(entry[0].resolved["grid"][transverse_key]) for entry in loaded
        ]
        scanned = [float(_resolved_value(entry[0].resolved, scan_variable))
                   for entry in loaded]
        fixed_transverse = all(x == transverse[0] for x in transverse[1:])
        fixed_aspect = all(math.isclose(
            x / h, transverse[0] / scanned[0], rel_tol=1.0e-10, abs_tol=0.0)
            for x, h in zip(transverse[1:], scanned[1:]))
        if not fixed_transverse and not fixed_aspect:
            raise AnalysisError(
                "transverse spacing must stay fixed or preserve the grid aspect ratio"
            )
        domains = {
            (
                float(entry[0].resolved["grid"]["R0"]),
                float(entry[0].resolved["grid"]["R0"])
                + (int(entry[0].resolved["grid"]["NR"]) - 1)
                * float(entry[0].resolved["grid"]["hR"]),
                float(entry[0].resolved["grid"]["Z0"]),
                float(entry[0].resolved["grid"]["Z0"])
                + (int(entry[0].resolved["grid"]["NZ"]) - 1)
                * float(entry[0].resolved["grid"]["hZ"]),
            )
            for entry in loaded
        }
        reference_domain = next(iter(domains))
        if any(
            not all(
                _same(value, reference, 1.0e-10)
                for value, reference in zip(domain, reference_domain)
            )
            for domain in domains
        ):
            raise AnalysisError(
                "spatial refinement must preserve the physical grid domain bounds"
            )
    case_comparison = value.get("case_comparison")
    if not isinstance(case_comparison, dict):
        raise AnalysisError("comparison descriptor requires case_comparison metadata")
    comparison_kind = case_comparison.get("kind")
    if scan_variable == "time.dt" and comparison_kind != "same_grid_same_initial":
        raise AnalysisError("time.dt requires same_grid_same_initial comparison")
    initial_hashes = {
        entry[0].manifest["input_hashes"]["initial"] for entry in loaded
    }
    if comparison_kind == "same_grid_same_initial":
        if not all(
            np.array_equal(entry[3].R, loaded[0][3].R)
            and np.array_equal(entry[3].Z, loaded[0][3].Z)
            for entry in loaded[1:]
        ):
            raise AnalysisError(
                "different grids require an audited_refinement declaration"
            )
        if len({entry[0].manifest["input_hashes"]["geometry"] for entry in loaded}) != 1:
            raise AnalysisError("same-grid levels must use identical boundary geometry")
        if len(initial_hashes) != 1:
            raise AnalysisError(
                "same-grid time levels must use identical initial physical content"
            )
        end_times = {
            float(entry[0].resolved["time"]["start_time"])
            + float(entry[0].resolved["time"]["dt"])
            * int(entry[0].resolved["time"]["nsteps"])
            for entry in loaded
        }
        if max(end_times) - min(end_times) > time_tolerance:
            raise AnalysisError(
                "dependent nsteps must align all levels at the same final time"
            )
    elif comparison_kind == "audited_refinement":
        if case_comparison.get("physical_case_id") != case_id:
            raise AnalysisError("audited refinement physical_case_id differs from case_id")
        declaration = case_comparison.get("initial_profile_comparison")
        if not isinstance(declaration, dict):
            raise AnalysisError(
                "audited refinement requires an initial_profile_comparison declaration"
            )
        if declaration.get("unit") != observable_unit:
            raise AnalysisError(
                "initial profile comparison unit differs from observable unit"
            )
        evidence = declaration.get("evidence_sha256")
        if not isinstance(evidence, str) or len(evidence) != 64:
            raise AnalysisError(
                "audited refinement requires a SHA-256 evidence declaration"
            )
    else:
        raise AnalysisError(
            "case_comparison.kind must be same_grid_same_initial or audited_refinement"
        )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "convergence",
        "analysis_code_sha256": _sha256(Path(__file__)),
        "numpy_version": np.__version__,
        "status": "PASS",
        "case_id": case_id,
        "scan_variable": scan_variable,
        "observable": {
            "name": observable_name,
            "key": observable.get("key"),
            "component": observable.get("component"),
            "mode": observable.get("mode", 0),
            "unit": observable_unit,
            "region": region,
            "norm": {
                "kind": norm_kind,
                "weights": weight_descriptor,
                "definition": NORM_DEFINITIONS[norm_kind],
                "unit": norm_unit,
                "normalization": "sum(weights)"
                if norm_kind == "weighted_RMS"
                else "none",
            },
        },
        "comparison_time_s": observed_times[0],
        "roundoff_floor": floor,
        "roundoff_guard": roundoff_guard,
        "levels": [entry[6] for entry in loaded],
        "warnings": [],
        "limitations": [],
    }

    if reference_kind != "none":
        method = "analytic_reference"
        result["method"] = method
        result["reference"] = {
            "kind": reference_kind,
            "unit": observable_unit,
            "declaration_sha256": _canonical_hash(reference),
            "parameters": reference,
        }
        errors: list[float] = []
        for _, _, field, layout, time_value, _, metadata, geometry in loaded:
            mask = geometry["mask"]
            RR = geometry["RR"]
            ZZ = geometry["ZZ"]
            exact = _analytic(reference, RR, ZZ, time_value)
            error_value, count = _norm(
                field - exact,
                mask,
                geometry["weights"],
                norm_kind,
            )
            errors.append(error_value)
            metadata["norm"] = {
                "value": error_value,
                "unit": norm_unit,
                "points": count,
            }
        orders: dict[str, Any] = {}
        valid = True
        for label, left, right in (("p01", 0, 1), ("p12", 1, 2)):
            e0, e1 = errors[left], errors[right]
            if e0 <= effective_floor or e1 <= effective_floor:
                valid = False
                orders[label] = _order_record(
                    None,
                    "UNAVAILABLE",
                    "an error is at or below the effective roundoff guard",
                    errors=[e0, e1],
                )
            else:
                h_left = refinements[left]
                h_right = refinements[right]
                orders[label] = _order_record(
                    math.log(e0 / e1) / math.log(h_left / h_right),
                    "PASS",
                    errors=[e0, e1],
                    refinement=[h_left, h_right],
                )
        result["errors"] = errors
        result["orders"] = orders
        if not valid:
            result["status"] = "UNAVAILABLE"
            result["reason"] = "one or more empirical orders cannot be reported"
        result["limitations"].append(
            "Analytic orders apply only to the explicitly declared observable and physical region."
        )
    else:
        result["reference"] = {"kind": "none"}
        fields = [entry[2] for entry in loaded]
        source_layouts = [entry[3] for entry in loaded]
        source_R = [layout.R for layout in source_layouts]
        source_Z = [layout.Z for layout in source_layouts]
        grids_equal = all(
            np.array_equal(source_R[0], R)
            and np.array_equal(source_Z[0], Z)
            and np.array_equal(source_layouts[0].active, layout.active)
            for R, Z, layout in zip(
                source_R[1:], source_Z[1:], source_layouts[1:]
            )
        )
        mapping_descriptor = value.get("mapping")
        mapped_fields: list[np.ndarray]
        if mapping_descriptor is not None:
            if not isinstance(mapping_descriptor, dict):
                raise AnalysisError("mapping must be an object")
            if mapping_descriptor.get("kind") != "bilinear":
                raise AnalysisError("mapping.kind must be 'bilinear'")
            grid = mapping_descriptor.get("comparison_grid")
            if not isinstance(grid, dict):
                raise AnalysisError("mapping.comparison_grid is required")
            try:
                target_R = np.asarray(grid["R"], dtype=np.float64)
                target_Z = np.asarray(grid["Z"], dtype=np.float64)
            except (KeyError, TypeError, ValueError) as exc:
                raise AnalysisError("invalid comparison grid coordinates") from exc
            if (
                target_R.ndim != 1
                or target_Z.ndim != 1
                or len(target_R) < 2
                or len(target_Z) < 2
                or np.any(np.diff(target_R) <= 0.0)
                or np.any(np.diff(target_Z) <= 0.0)
            ):
                raise AnalysisError("comparison grid axes must be strictly increasing")
            coordinates_identical = grids_equal and all(
                np.array_equal(target_R, R) and np.array_equal(target_Z, Z)
                for R, Z in zip(source_R, source_Z)
            )
            if coordinates_identical:
                mapped_fields = fields
                target_mask = loaded[0][7]["mask"]
            else:
                if any(not np.all(layout.active) for layout in source_layouts):
                    raise AnalysisError(
                        "bilinear mapping across inactive geometry requires an explicit active-mask descriptor"
                    )
                mapped_fields = [
                    _map_bilinear(field, R, Z, target_R, target_Z)
                    for field, R, Z in zip(fields, source_R, source_Z)
                ]
                target_mask, _, _ = _region_mask(target_R, target_Z, region)
            target_layout = FieldLayout(
                key=source_layouts[0].key,
                role=source_layouts[0].role,
                R=target_R,
                Z=target_Z,
                active=np.ones(target_mask.shape, dtype=bool),
                quadrature=np.zeros(target_mask.shape, dtype=np.float64),
                component=source_layouts[0].component,
            )
            if (
                weight_descriptor.get("kind") == "a0_quadrature"
                and not coordinates_identical
            ):
                raise AnalysisError(
                    "a0_quadrature requires identical active grids; mapped runs need explicit comparison quadrature"
                )
            target_weights = (
                loaded[0][7]["weights"]
                if coordinates_identical
                else _weights(weight_descriptor, target_layout)
            )
            interpolation = {
                "kind": "bilinear",
                "comparison_grid": {
                    "R": target_R.tolist(),
                    "Z": target_Z.tolist(),
                },
                "source_layout": "[mode, iz, ir]",
            }
            if coordinates_identical:
                interpolation["interpolation_error"] = {
                    "status": "PASS",
                    "value": 0.0,
                    "unit": norm_unit,
                    "reason": "comparison coordinates coincide with source coordinates",
                }
            elif mapping_descriptor.get("interpolation_error") is not None:
                supplied = _number(
                    mapping_descriptor.get("interpolation_error"),
                    "mapping.interpolation_error",
                )
                if supplied < 0.0:
                    raise AnalysisError("mapping.interpolation_error must be nonnegative")
                interpolation["interpolation_error"] = {
                    "status": "PASS",
                    "value": supplied,
                    "unit": norm_unit,
                    "source": "user_declared",
                }
            else:
                interpolation["interpolation_error"] = {
                    "status": "UNAVAILABLE",
                    "value": None,
                    "unit": norm_unit,
                    "reason": "no independent interpolation error was supplied",
                }
        else:
            if not grids_equal:
                raise AnalysisError(
                    "different grids require an explicit comparison-grid mapping"
                )
            mapped_fields = fields
            target_mask = loaded[0][7]["mask"]
            target_weights = loaded[0][7]["weights"]
            interpolation = {
                "kind": "identity",
                "source_layout": "[mode, iz, ir]",
                "interpolation_error": {
                    "status": "PASS",
                    "value": 0.0,
                    "unit": norm_unit,
                    "reason": "all levels use identical physical coordinates",
                },
            }
        difference_01, count_01 = _norm(
            mapped_fields[0] - mapped_fields[1],
            target_mask,
            target_weights,
            norm_kind,
        )
        difference_12, count_12 = _norm(
            mapped_fields[1] - mapped_fields[2],
            target_mask,
            target_weights,
            norm_kind,
        )
        differences = [difference_01, difference_12]
        result["mapping"] = interpolation
        result["differences"] = {
            "d01": {"value": difference_01, "unit": norm_unit, "points": count_01},
            "d12": {"value": difference_12, "unit": norm_unit, "points": count_12},
        }
        h0, h1, h2 = refinements
        if not (h0 > h1 > h2):
            raise AnalysisError("refinement levels must be ordered h0 > h1 > h2")
        if (
            difference_01 <= effective_floor
            or difference_12 <= effective_floor
        ):
            result["status"] = "UNAVAILABLE"
            result["reason"] = "difference is at or below the effective roundoff guard"
            result["orders"] = {
                "richardson": _order_record(None, "UNAVAILABLE", result["reason"])
            }
            result["richardson_estimate"] = {"status": "UNAVAILABLE", "value": None}
        else:
            ratio = float(difference_01 / difference_12)
            if _same(h0 / h1, h1 / h2, 1.0e-9):
                refinement_ratio = h0 / h1
                result["method"] = "richardson"
                result["refinement_relation"] = "equal"
                order = math.log(ratio) / math.log(refinement_ratio)
                order_status = "PASS"
                order_reason = ""
            else:
                result["method"] = "richardson"
                result["refinement_relation"] = "unequal"
                interval = value.get("order_interval")
                if not isinstance(interval, list) or len(interval) != 2:
                    raise AnalysisError(
                        "unequal refinement requires order_interval [minimum, maximum]"
                    )
                order, order_reason = _bisect_order(h0, h1, h2, ratio, interval)
                order_status = "PASS" if order is not None else "UNAVAILABLE"
            if order is not None and order <= 0.0:
                order_status = "WARN"
                result["status"] = "WARN"
                result["warnings"].append(
                    "Richardson order is nonpositive; differences do not decrease under refinement"
                )
            result["orders"] = {
                "richardson": _order_record(
                    order,
                    order_status,
                    order_reason,
                    difference_ratio=ratio,
                )
            }
            if order is None:
                result["status"] = "UNAVAILABLE"
                result["reason"] = order_reason
                result["richardson_estimate"] = {
                    "status": "UNAVAILABLE",
                    "value": None,
                }
            else:
                if not math.isfinite(order):
                    result["status"] = "UNAVAILABLE"
                    result["reason"] = "Richardson order is non-finite"
                    result["richardson_estimate"] = {
                        "status": "UNAVAILABLE",
                        "value": None,
                    }
                else:
                    ratio_power = (h1 / h2) ** order
                    if order <= 0.0:
                        estimate = None
                        estimate_status = "UNAVAILABLE"
                    elif math.isclose(ratio_power, 1.0):
                        estimate = None
                        estimate_status = "UNAVAILABLE"
                    else:
                        estimate = difference_12 / abs(ratio_power - 1.0)
                        estimate_status = "PASS"
                    result["richardson_estimate"] = {
                        "status": estimate_status,
                        "kind": "leading_difference_estimate",
                        "leading_error": estimate,
                        "unit": norm_unit,
                        "assumption": "smooth asymptotic expansion with leading power p",
                    }
        result["limitations"].append(
            "Richardson order and extrapolation are asymptotic estimates, not known exact errors."
        )

    if value.get("interpretation") is not None:
        result["interpretation"] = value["interpretation"]
    if descriptor_path is not None:
        result["descriptor"] = {
            "path": str(descriptor_path),
            "sha256": _sha256(descriptor_path),
        }

    if output is not None:
        output_path = Path(output)
        inputs = [entry[0].directory for entry in loaded]
        if descriptor_path is not None:
            inputs.append(descriptor_path)
        if _output_is_in_inputs(output_path, inputs):
            raise AnalysisError("analysis output must not be placed in an input run")
        _write_report(result, output_path)
    return result


def _read_observations(run: RunData) -> tuple[list[dict[str, Any]], str]:
    definitions = run.resolved.get("diagnostics", {}).get("mode_observations", [])
    if not isinstance(definitions, list):
        raise AnalysisError("resolved diagnostics.mode_observations must be a list")
    rows: list[dict[str, Any]] = []
    try:
        lines = run.paths["observations.jsonl"].read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AnalysisError(f"cannot read observations.jsonl: {exc}") from exc
    accepted_pairs = {
        (int(row["step"]), float(row["time_s"])) for row in run.history
    }
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
            if not isinstance(row, dict):
                raise AnalysisError("row is not an object")
            step = int(row["step"])
            time_value = _number(row["time"], "observation time")
            real = _number(row["real"], "observation real component")
            imag = _number(row["imag"], "observation imaginary component")
            definition = row["definition"]
        except AnalysisError as exc:
            raise AnalysisError(
                f"malformed observations.jsonl line {line_number}: {exc}"
            ) from exc
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AnalysisError(
                f"malformed observations.jsonl line {line_number}: {exc}"
            ) from exc
        if not any(definition == declared for declared in definitions):
            raise AnalysisError(
                f"observation definition on line {line_number} is absent from resolved config"
            )
        if not any(
            accepted_step == step and accepted_time == time_value
            for accepted_step, accepted_time in accepted_pairs
        ):
            raise AnalysisError(
                f"observation line {line_number} lacks a matching accepted history step/time"
            )
        field_name = definition.get("field")
        field_component = definition.get("component")
        layout = _field_layout(
            run,
            {
                "key": field_name,
                "component": (
                    None
                    if field_name not in {"mom_i", "mom_e", "B"}
                    or field_component == "scalar"
                    else field_component
                ),
                "mode": definition.get("mode", 0),
            },
        )
        try:
            ix = int(definition["entity_key"][0])
            iy = int(definition["entity_key"][1])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise AnalysisError("observation entity_key is invalid") from exc
        if (
            iy < 0
            or iy >= layout.active.shape[0]
            or ix < 0
            or ix >= layout.active.shape[1]
            or not layout.active[iy, ix]
        ):
            raise AnalysisError("observation entity is inactive or outside the physical grid")
        try:
            mode = int(definition["mode"])
            K = int(run.resolved["grid"]["K"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AnalysisError("observation mode identity is invalid") from exc
        if abs(mode) > K:
            raise AnalysisError("observation mode lies outside the retained spectrum")
        rows.append(
            {
                "step": step,
                "time": time_value,
                "value": complex(real, imag),
                "definition": definition,
            }
        )
    if not rows:
        raise AnalysisError("observations.jsonl contains no complex observations")
    return rows, _sha256(run.paths["observations.jsonl"])


def _select_definition(
    rows: list[dict[str, Any]],
    field: str | None,
    component: str | None,
    mode: int | None,
    entity_key: list[int] | None,
) -> dict[str, Any]:
    definitions: list[dict[str, Any]] = []
    for row in rows:
        if row["definition"] not in definitions:
            definitions.append(row["definition"])
    if all(value is None for value in (field, component, mode, entity_key)):
        if len(definitions) != 1:
            raise AnalysisError(
                "multiple mode observations require field/component/mode/entity_key"
            )
        return definitions[0]
    selected = {
        key: value
        for key, value in {
            "field": field,
            "component": component,
            "mode": mode,
            "entity_key": entity_key,
        }.items()
        if value is not None
    }
    matches = [
        definition
        for definition in definitions
        if all(definition.get(key) == value for key, value in selected.items())
    ]
    if len(matches) != 1:
        raise AnalysisError("requested complex observation definition is not unique")
    return matches[0]


def _observable_unit(definition: dict[str, Any]) -> str | None:
    field = definition.get("field")
    if not isinstance(field, str):
        return None
    if field in OBSERVABLE_UNITS:
        return OBSERVABLE_UNITS[field]
    if field.startswith("B_"):
        return OBSERVABLE_UNITS["B"]
    if field.startswith("mom_"):
        return OBSERVABLE_UNITS["mom_i"]
    return None


def _linear_fit(
    times: np.ndarray, values: np.ndarray
) -> tuple[float, float, np.ndarray, np.ndarray, float]:
    center = float(np.mean(times))
    time_scale = float(np.max(np.abs(times - center)))
    if time_scale == 0.0:
        raise AnalysisError("linear fit requires more than one distinct time")
    scaled_time = (times - center) / time_scale
    design = np.column_stack((np.ones(len(times)), scaled_time))
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coefficients
    slope = float(coefficients[1]) / time_scale
    intercept = float(coefficients[0]) - slope * center
    return intercept, slope, residual, design, time_scale


def _uncertainty(
    residual: np.ndarray, design: np.ndarray, time_scale: float
) -> float:
    if len(residual) <= 2:
        return math.inf
    dof = len(residual) - 2
    variance = float(np.dot(residual, residual)) / dof
    covariance = variance * np.linalg.pinv(design.T @ design)
    return math.sqrt(max(float(covariance[1, 1]), 0.0)) / time_scale


def _fft_crosscheck(
    times: np.ndarray, values: np.ndarray, target_frequency_hz: float
) -> dict[str, Any]:
    cadence = float(np.median(np.diff(times)))
    count = len(times)
    nfft = 1
    while nfft < max(8 * count, 1024):
        nfft *= 2
    centered = values * np.exp(-0.5j * np.angle(values[0]))
    spectrum = np.fft.fft(centered, n=nfft)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.fftfreq(nfft, d=cadence)
    peak = int(np.argmax(power))
    physical_frequency = -float(frequencies[peak])
    resolution = 1.0 / (count * cadence)
    tolerance = max(0.1 * abs(target_frequency_hz), 2.0 * resolution)
    return {
        "status": "PASS"
        if abs(physical_frequency - target_frequency_hz) <= tolerance
        else "WARN",
        "physical_frequency_hz": physical_frequency,
        "angular_frequency_rad_s": 2.0 * math.pi * physical_frequency,
        "numpy_signed_frequency_hz": float(frequencies[peak]),
        "resolution_hz": resolution,
        "agreement_tolerance_hz": tolerance,
        "convention": "physical frequency is the negative NumPy FFT frequency for exp(-i*omega*t)",
    }


def _residual_spectrum_ratio(
    times: np.ndarray, residual: np.ndarray, signal: np.ndarray
) -> float:
    cadence = float(np.median(np.diff(times)))
    nfft = 1
    while nfft < max(8 * len(times), 1024):
        nfft *= 2
    residual_power = np.abs(np.fft.fft(residual, n=nfft))
    signal_power = np.abs(np.fft.fft(signal, n=nfft))
    signal_peak = float(np.max(signal_power))
    return (
        float(np.max(residual_power)) / signal_peak
        if signal_peak > 0.0
        else math.inf
    )


def fit_mode(
    run_dir: str | Path,
    *,
    t_start: float | None,
    t_end: float | None,
    amplitude_floor: float,
    target_frequency_hz: float,
    field: str | None = None,
    component: str | None = None,
    mode: int | None = None,
    entity_key: list[int] | None = None,
    cadence_relative_tolerance: float = 1.0e-6,
    residual_phase_tolerance: float = 0.1,
    residual_log_amplitude_tolerance: float = 0.05,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Fit A(t)=A0 exp(gamma*t) exp(i(phase0-omega*t)) to observations."""

    if t_start is None or t_end is None:
        raise AnalysisError("an explicit [t_start, t_end] window is required")
    start = _number(t_start, "t_start")
    end = _number(t_end, "t_end")
    if start >= end:
        raise AnalysisError("fit window requires t_start < t_end")
    floor = _positive(amplitude_floor, "amplitude_floor")
    target = _number(target_frequency_hz, "target_frequency_hz")
    cadence_tolerance = _positive(
        cadence_relative_tolerance, "cadence_relative_tolerance"
    )
    phase_tolerance = _positive(residual_phase_tolerance, "residual_phase_tolerance")
    log_tolerance = _positive(
        residual_log_amplitude_tolerance, "residual_log_amplitude_tolerance"
    )

    run = load_run(run_dir)
    rows, observations_hash = _read_observations(run)
    definition = _select_definition(rows, field, component, mode, entity_key)
    selected = [
        row
        for row in rows
        if row["definition"] == definition and start <= row["time"] <= end
    ]
    if len(selected) < 3:
        raise AnalysisError("fit window must contain at least three observations")
    selected.sort(key=lambda row: row["time"])
    if any(
        left["time"] == right["time"]
        for left, right in zip(selected, selected[1:])
    ):
        raise AnalysisError("selected projection has duplicate observation times")
    times = np.asarray([row["time"] for row in selected], dtype=np.float64)
    signal = np.asarray([row["value"] for row in selected], dtype=np.complex128)
    if np.any(np.diff(times) <= 0.0):
        raise AnalysisError("fit samples must have strictly increasing time")
    intervals = np.diff(times)
    cadence = float(np.median(intervals))
    if not math.isfinite(cadence) or cadence <= 0.0:
        raise AnalysisError("fit sample cadence must be positive and finite")
    if np.max(np.abs(intervals - cadence)) > cadence_tolerance * cadence:
        raise AnalysisError("fit samples do not have a consistent saved cadence")
    if abs(target) >= 0.5 / cadence:
        raise AnalysisError(
            "saved cadence does not resolve the target frequency (Nyquist criterion)"
        )

    amplitude = np.abs(signal)
    valid = amplitude >= floor
    excluded = [
        {"time_s": float(t), "amplitude": float(a)}
        for t, a, is_valid in zip(times, amplitude, valid)
        if not is_valid
    ]
    if int(np.sum(valid)) < 3:
        raise AnalysisError("fewer than three samples exceed the amplitude floor")
    fit_times = times[valid]
    fit_signal = signal[valid]
    fit_amplitude = amplitude[valid]
    phase = np.unwrap(np.angle(fit_signal))
    log_amplitude = np.log(fit_amplitude)
    reference_time = float(np.mean(fit_times))
    relative_times = fit_times - reference_time
    phase_reference, phase_slope, phase_residual, phase_design, phase_scale = _linear_fit(
        relative_times, phase
    )
    log_reference, growth, log_residual, log_design, log_scale = _linear_fit(
        relative_times, log_amplitude
    )
    phase0 = phase_reference - phase_slope * reference_time
    log_a0 = log_reference - growth * reference_time
    # Extrapolation to t=0 may overflow even for finite data in the fit window.
    amplitude_at_zero = (math.exp(log_a0) if
        math.log(np.nextafter(0.0, 1.0)) <= log_a0 <= math.log(sys.float_info.max)
        else None)
    omega = -phase_slope
    frequency = omega / (2.0 * math.pi)
    phase_uncertainty = _uncertainty(phase_residual, phase_design, phase_scale)
    growth_uncertainty = _uncertainty(log_residual, log_design, log_scale)
    modeled_signal = np.exp(log_reference + growth * relative_times) * np.exp(
        1j * (phase_reference - omega * relative_times)
    )
    complex_residual = fit_signal - modeled_signal
    fft_crosscheck = _fft_crosscheck(fit_times, fit_signal, target)
    residual_peak_ratio = _residual_spectrum_ratio(
        fit_times, complex_residual, fit_signal
    )

    warnings: list[str] = []
    if excluded:
        warnings.append(
            f"{len(excluded)} sample(s) below the explicit amplitude floor were excluded"
        )
    phase_rms = math.sqrt(float(np.mean(phase_residual**2)))
    log_rms = math.sqrt(float(np.mean(log_residual**2)))
    if phase_rms > phase_tolerance:
        warnings.append("single-mode phase residual exceeds the declared tolerance")
    if log_rms > log_tolerance:
        warnings.append(
            "single-mode log-amplitude residual exceeds the declared tolerance"
        )
    if residual_peak_ratio > 0.2:
        warnings.append("residual spectrum indicates a multi-tone component")
    if fft_crosscheck["status"] != "PASS":
        warnings.append("FFT crosscheck disagrees with the target frequency")
    if abs(frequency - target) > fft_crosscheck["agreement_tolerance_hz"]:
        warnings.append("fitted frequency disagrees with the declared target")

    split = len(fit_times) // 2
    nonstationary = False
    split_metadata: dict[str, Any] = {"status": "UNAVAILABLE"}
    if split >= 3 and len(fit_times) - split >= 3:
        first_phase = np.unwrap(np.angle(fit_signal[:split]))
        second_phase = np.unwrap(np.angle(fit_signal[split:]))
        first_log = np.log(fit_amplitude[:split])
        second_log = np.log(fit_amplitude[split:])
        _, slope_1, _, _, _ = _linear_fit(fit_times[:split], first_phase)
        _, slope_2, _, _, _ = _linear_fit(fit_times[split:], second_phase)
        _, growth_1, _, _, _ = _linear_fit(
            fit_times[:split], first_log
        )
        _, growth_2, _, _, _ = _linear_fit(
            fit_times[split:], second_log
        )
        frequency_1 = -slope_1 / (2.0 * math.pi)
        frequency_2 = -slope_2 / (2.0 * math.pi)
        growth_threshold = max(0.05, 0.25 * abs(growth))
        frequency_threshold = max(3.0 * fft_crosscheck["resolution_hz"], 0.05 * abs(target))
        nonstationary = (
            abs(growth_2 - growth_1) > growth_threshold
            or abs(frequency_2 - frequency_1) > frequency_threshold
        )
        split_metadata = {
            "status": "WARN" if nonstationary else "PASS",
            "first_half": {
                "frequency_hz": frequency_1,
                "growth_rate_per_s": growth_1,
            },
            "second_half": {
                "frequency_hz": frequency_2,
                "growth_rate_per_s": growth_2,
            },
            "growth_difference_per_s": growth_2 - growth_1,
            "frequency_difference_hz": frequency_2 - frequency_1,
        }
        if nonstationary:
            warnings.append(
                "nonstationary envelope or frequency detected between fit halves"
            )
    else:
        warnings.append("too few samples for a split-half stationarity check")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "mode_fit",
        "analysis_code_sha256": _sha256(Path(__file__)),
        "numpy_version": np.__version__,
        "status": "WARN" if warnings else "PASS",
        "phase_convention": PHASE_CONVENTION,
        "formula": "A(t)=A0*exp(gamma*t)*exp(i*(phase0-omega*t))",
        "observable": {
            "definition": definition,
            "unit": _observable_unit(definition),
            "selection": {
                "field": field,
                "component": component,
                "mode": mode,
                "entity_key": entity_key,
            },
        },
        "fit_window": {"start_s": start, "end_s": end},
        "sample_cadence_s": cadence,
        "amplitude_floor": floor,
        "target_frequency_hz": target,
        "sample_count": int(np.sum(valid)),
        "excluded_low_amplitude": excluded,
        "frequency_hz": frequency,
        "angular_frequency_rad_s": omega,
        "phase_intercept_rad": phase0,
        "growth_rate_per_s": growth,
        "log_amplitude_intercept": log_a0,
        "amplitude": amplitude_at_zero,
        "amplitude_extrapolation_status": "UNAVAILABLE" if amplitude_at_zero is None else "PASS",
        "reference_time_s": reference_time,
        "amplitude_at_reference_time": math.exp(log_reference),
        "phase_at_reference_time_rad": phase_reference,
        "uncertainty": {
            "frequency_hz_1sigma": phase_uncertainty / (2.0 * math.pi),
            "angular_frequency_rad_s_1sigma": phase_uncertainty,
            "growth_rate_per_s_1sigma": growth_uncertainty,
            "scope": "linear least-squares statistical estimate; model/interpolation error is separate",
        },
        "samples": {
            "time_s": fit_times.tolist(),
            "real": fit_signal.real.tolist(),
            "imag": fit_signal.imag.tolist(),
            "amplitude": fit_amplitude.tolist(),
        },
        "residuals": {
            "phase_rad": phase_residual.tolist(),
            "log_amplitude": log_residual.tolist(),
            "complex_real": complex_residual.real.tolist(),
            "complex_imag": complex_residual.imag.tolist(),
            "phase_rms_rad": phase_rms,
            "log_amplitude_rms": log_rms,
            "residual_peak_to_signal_peak": residual_peak_ratio,
        },
        "warnings": warnings,
        "fft_crosscheck": fft_crosscheck,
        "stationarity": split_metadata,
        "polarization": {
            "status": "UNAVAILABLE",
            "reason": "a scalar mode-observation projection does not define a vector polarization",
        },
        "provenance": _provenance(run),
        "limitations": [
            "A numerical decay is not interpreted as Landau damping.",
            "Single-mode values are not claimed when warnings report unresolved structure.",
        ],
    }
    result["provenance"]["observations_jsonl_sha256_actual"] = observations_hash
    if output is not None:
        output_path = Path(output)
        inputs = [run.directory]
        if _output_is_in_inputs(output_path, inputs):
            raise AnalysisError("analysis output must not be placed in an input run")
        _write_report(result, output_path)
    return result


def _entity_key(value: str) -> list[int]:
    try:
        parts = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("entity key must be two integers: R,Z") from exc
    if len(parts) != 2 or any(part < 0 for part in parts):
        raise argparse.ArgumentTypeError("entity key must be two nonnegative integers")
    return parts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only A0 convergence and mode analysis"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convergence = subparsers.add_parser(
        "convergence", help="analyze a comparison descriptor"
    )
    convergence.add_argument("descriptor", type=Path)
    convergence.add_argument("--output", type=Path, required=True)

    fit = subparsers.add_parser("fit-mode", help="fit one complex mode observation")
    fit.add_argument("run", type=Path)
    fit.add_argument("--t-start", type=float, required=True)
    fit.add_argument("--t-end", type=float, required=True)
    fit.add_argument("--amplitude-floor", type=float, required=True)
    fit.add_argument("--target-frequency-hz", type=float, required=True)
    fit.add_argument("--field")
    fit.add_argument("--component")
    fit.add_argument("--mode", type=int)
    fit.add_argument("--entity-key", type=_entity_key)
    fit.add_argument("--cadence-tolerance", type=float, default=1.0e-6)
    fit.add_argument("--residual-phase-tolerance", type=float, default=0.1)
    fit.add_argument(
        "--residual-log-amplitude-tolerance", type=float, default=0.05
    )
    fit.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "convergence":
            analyze_convergence(arguments.descriptor, output=arguments.output)
        else:
            fit_mode(
                arguments.run,
                t_start=arguments.t_start,
                t_end=arguments.t_end,
                amplitude_floor=arguments.amplitude_floor,
                target_frequency_hz=arguments.target_frequency_hz,
                field=arguments.field,
                component=arguments.component,
                mode=arguments.mode,
                entity_key=arguments.entity_key,
                cadence_relative_tolerance=arguments.cadence_tolerance,
                residual_phase_tolerance=arguments.residual_phase_tolerance,
                residual_log_amplitude_tolerance=arguments.residual_log_amplitude_tolerance,
                output=arguments.output,
            )
    except (AnalysisError, FileExistsError, OSError) as exc:
        print(f"analyze: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
