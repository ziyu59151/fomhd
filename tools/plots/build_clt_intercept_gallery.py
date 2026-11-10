#!/usr/bin/env python3
"""Build a bounded offline CLT intercept evidence gallery."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

sys.dont_write_bytecode = True
for _thread_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_name] = "1"

# Set the bounded thread counts before loading numerical libraries.
import numpy as np  # noqa: E402

OVERLAY_ROOT = Path(__file__).resolve().parents[2]
if str(OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(OVERLAY_ROOT))

from tools.clt_physical_rz_evidence import (  # noqa: E402
    load_old_formal_evidence,
    load_repair3_rigid_evidence,
)
from tools.plots.render_clt_physical_rz import (  # noqa: E402
    FIGURE6,
    FIGURE7,
    render_physical_rz_figures,
)


SCOPE = (
    "local geometry / sampling / paired-reference tests; "
    "no complete evolution acceptance"
)
GRID_LIMIT = 24
MAX_WALL_SECONDS = 180.0
MAX_RSS_BYTES = 4 << 30
LOCAL_SCHEMA = "fomhd_local_intercept_patch_probe_v2_parent_repair1"
GALLERY_SCHEMA = "fomhd_clt_intercept_gallery_v1"
FIGURE_NAMES = (
    "fig01_rz_intercept_geometry",
    "fig02_geometry_preprocess",
    "fig03_local_field_current",
    "fig04_gram_conditioning",
    "fig05_clt_coefficient_growth",
    FIGURE6,
    FIGURE7,
)
HEX_RE = re.compile(r"[0-9a-f]{64}")
STYLE_RE = re.compile(r"<style>(.*?)</style>", re.DOTALL)


class InputError(ValueError):
    """Raised when an input, source identity, or output path is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_constant(value: str) -> Any:
    raise InputError(f"nonfinite JSON constant is not allowed: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_json_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise InputError(f"cannot read JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise InputError(f"JSON root must be an object: {path}")
    return value


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{where} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise InputError(f"{where} must be finite")
    return number


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputError(f"{where} must be an object")
    return value


def _items(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise InputError(f"{where} must be an array")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{where} must be nonempty text")
    return value


def validate_geometry_probes(
    data: dict[str, Any],
    source_root: Path,
) -> dict[str, Any]:
    """Validate the supplied geometry receipts and pinned source identities."""

    source_root = source_root.resolve()
    required = {
        "scope",
        "units",
        "polygon_recipe",
        "grid_recipe",
        "source_files_sha256",
        "script_sha256",
        "elapsed_wall_seconds",
        "cases",
    }
    if not required.issubset(data):
        raise InputError(
            f"geometry input is missing keys: {sorted(required - set(data))}"
        )
    _text(data["scope"], "geometry.scope")
    if data["units"] != "SI, cylindrical surface/volume measures per radian":
        raise InputError("geometry.units differs from the declared SI convention")
    if data["polygon_recipe"] != (
        "128 circle samples, R0=3m, a=0.5m, angle_offset=0.013rad"
    ):
        raise InputError("geometry.polygon_recipe differs from the supplied circle")
    if "h=1/(nodes-4)" not in _text(
        data["grid_recipe"], "geometry.grid_recipe"
    ):
        raise InputError("geometry.grid_recipe lacks the declared h formula")
    if not HEX_RE.fullmatch(_text(data["script_sha256"], "geometry.script_sha256")):
        raise InputError("geometry.script_sha256 is not SHA-256")
    _finite(data["elapsed_wall_seconds"], "geometry.elapsed_wall_seconds")

    hashes = _object(data["source_files_sha256"], "geometry.source_files_sha256")
    for relative in ("src/fomhd/geometry.py", "src/fomhd/boundary_geometry.py"):
        expected = _text(hashes.get(relative), f"geometry hash {relative}")
        path = source_root / relative
        if not path.is_file():
            raise InputError(f"required geometry source is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise InputError(f"geometry source SHA-256 mismatch: {relative}")

    cases = _items(data["cases"], "geometry.cases")
    if len(cases) != 8:
        raise InputError("geometry.cases must contain the eight declared cases")
    combos: set[tuple[int, float]] = set()
    for index, raw in enumerate(cases):
        where = f"geometry.cases[{index}]"
        case = _object(raw, where)
        nodes = case.get("nodes")
        if isinstance(nodes, bool) or not isinstance(nodes, int):
            raise InputError(f"{where}.nodes must be integer")
        if nodes not in (8, 12, 16, 24) or nodes > GRID_LIMIT:
            raise InputError(f"{where}.nodes exceeds the declared grid family")
        h_m = _finite(case.get("h_m"), f"{where}.h_m")
        phase_r = _finite(
            case.get("grid_phase_R_h"), f"{where}.grid_phase_R_h"
        )
        phase_z = _finite(
            case.get("grid_phase_Z_h"), f"{where}.grid_phase_Z_h"
        )
        if not math.isclose(h_m, 1.0 / (nodes - 4), rel_tol=0.0, abs_tol=1e-15):
            raise InputError(f"{where}.h_m does not equal 1/(nodes-4)")
        if not math.isclose(phase_z, -0.7 * phase_r, abs_tol=1e-15):
            raise InputError(f"{where} phase Z does not follow the declared shift")
        combo = (nodes, phase_r)
        if combo in combos:
            raise InputError(f"duplicate geometry case: {combo}")
        combos.add(combo)
        for key in (
            "boundary_link_min_over_h",
            "elapsed_wall_seconds",
            "polygon_area_m2",
            "dual_covered_area_m2",
        ):
            if _finite(case.get(key), f"{where}.{key}") < 0.0:
                raise InputError(f"{where}.{key} must be nonnegative")
        if case.get("status") != "geometry_constructed_only":
            raise InputError(f"{where}.status is not the declared geometry-only state")
        if not HEX_RE.fullmatch(_text(
            case.get("geometry_fingerprint"),
            f"{where}.geometry_fingerprint",
        )):
            raise InputError(f"{where}.geometry_fingerprint is not SHA-256")
        for class_key in ("r_classes", "z_classes"):
            classes = _object(case.get(class_key), f"{where}.{class_key}")
            for key in classes:
                if key not in {"1", "2", "4", "5"}:
                    raise InputError(f"{where}.{class_key} has unknown class {key}")
                value = classes[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise InputError(f"{where}.{class_key}.{key} is invalid")
    expected_combos = {
        (nodes, phase) for nodes in (8, 12, 16, 24) for phase in (0.0, 0.17)
    }
    if combos != expected_combos:
        raise InputError("geometry cases do not match the declared grid/shift scan")
    return data


def validate_local_pairing(data: dict[str, Any]) -> dict[str, Any]:
    """Validate only the corrected local-pairing repair input."""

    if data.get("schema") != LOCAL_SCHEMA:
        raise InputError(f"local-pairing schema must be {LOCAL_SCHEMA}")
    repair = _object(data.get("repair"), "local.repair")
    if repair.get("corrected_sign_output") is not True:
        raise InputError("local-pairing repair does not assert corrected pressure sign")
    if not _items(data.get("not_accepted"), "local.not_accepted"):
        raise InputError("local.not_accepted must retain explicit scope exclusions")
    cases = _items(data.get("cases"), "local.cases")
    if len(cases) != 20:
        raise InputError("local.cases must contain the 20 declared local cases")
    for index, raw in enumerate(cases):
        where = f"local.cases[{index}]"
        case = _object(raw, where)
        for key in ("aspect_ratio", "max_length_m", "a_m", "b_m"):
            if _finite(case.get(key), f"{where}.{key}") <= 0.0:
                raise InputError(f"{where}.{key} must be positive")
        potential = _object(case.get("potential"), f"{where}.potential")
        electric = _object(case.get("electric"), f"{where}.electric")
        metric = _object(case.get("metric"), f"{where}.metric")
        for mapping, key in (
            (potential, "curved_gradient_relative_linf"),
            (electric, "normal_reconstruction_linf"),
            (electric, "tangent_reconstruction_linf"),
            (metric, "homogeneous_condition"),
            (metric, "lifted_condition"),
        ):
            if _finite(mapping.get(key), f"{where}.{key}") < 0.0:
                raise InputError(f"{where}.{key} must be nonnegative")
    refinement = _object(data.get("refinement"), "local.refinement")
    expected_levels = {"0.01", "0.1", "1.0", "10.0", "100.0"}
    if set(refinement) != expected_levels:
        raise InputError("local.refinement levels differ from the declared aspects")
    reciprocal = _object(
        data.get("reciprocal_current_control"),
        "local.reciprocal_current_control",
    )
    if reciprocal.get("status") != (
        "independent_local_prototype_not_global_Ohm_acceptance"
    ):
        raise InputError("reciprocal-current control has a different scope status")
    reciprocal_cases = _items(
        reciprocal.get("cases"), "local.reciprocal_current_control.cases"
    )
    if len(reciprocal_cases) != 3:
        raise InputError("reciprocal-current control requires three refinement levels")
    h_values: list[float] = []
    for index, raw in enumerate(reciprocal_cases):
        where = f"local.reciprocal_current_control.cases[{index}]"
        case = _object(raw, where)
        for key in (
            "h_m",
            "retained_moment_power_error",
            "omitted_moment_power_error",
            "relative_weighted_field_error",
        ):
            if _finite(case.get(key), f"{where}.{key}") <= 0.0:
                raise InputError(f"{where}.{key} must be positive")
        h_values.append(float(case["h_m"]))
    if not all(first > second for first, second in zip(h_values, h_values[1:])):
        raise InputError("reciprocal-current h_m values must decrease")

    receipt = _object(data.get("provenance_receipt"), "local.provenance_receipt")
    receipt_path = Path(_text(receipt.get("path"), "local.provenance_receipt.path"))
    expected_hash = _text(
        receipt.get("sha256"), "local.provenance_receipt.sha256"
    )
    if not receipt_path.is_file() or sha256_file(receipt_path) != expected_hash:
        raise InputError("local provenance receipt is missing or hash-mismatched")
    return data


def validate_local_lineage(
    local_json: Path,
    producer: Path,
    implementation: Path,
    status_path: Path,
) -> dict[str, Any]:
    """Validate the existing worker STATUS binding for corrected local data."""

    local_json = local_json.resolve()
    producer = producer.resolve()
    implementation = implementation.resolve()
    status_path = status_path.resolve()
    for path in (local_json, producer, implementation, status_path):
        if not path.is_file():
            raise InputError(f"local-data lineage input is missing: {path}")

    status = _load_json(status_path)
    if status.get("schema") != "fomhd-clt-local-pairing-worker-status-v1":
        raise InputError("local-data STATUS schema is not the worker receipt")
    if status.get("status") != "complete_bounded_parent_repair1":
        raise InputError("local-data STATUS is not the corrected repair1 receipt")
    worker_root = Path(
        _text(status.get("worker_root"), "local-data STATUS.worker_root")
    ).resolve()
    artifacts = _object(status.get("artifacts"), "local-data STATUS.artifacts")

    bindings = (
        ("implementation", implementation, "implementation"),
        ("probe", producer, "producer"),
        ("corrected_json", local_json, "corrected JSON"),
    )
    hashes: dict[str, str] = {}
    for key, explicit_path, label in bindings:
        artifact = _object(artifacts.get(key), f"local-data STATUS.artifacts.{key}")
        relative = Path(_text(
            artifact.get("path"),
            f"local-data STATUS.artifacts.{key}.path",
        ))
        resolved_artifact = (worker_root / relative).resolve()
        if relative.is_absolute() or resolved_artifact != explicit_path:
            raise InputError(f"{label} path differs from local-data STATUS")
        expected_hash = _text(
            artifact.get("sha256"),
            f"local-data STATUS.artifacts.{key}.sha256",
        )
        actual_hash = sha256_file(explicit_path)
        if actual_hash != expected_hash:
            raise InputError(f"{label} hash differs from local-data STATUS")
        hashes[key] = actual_hash

    expected_implementation = (
        producer.parent / "repo/src/fomhd/boundary_patch.py"
    ).resolve()
    if implementation != expected_implementation:
        raise InputError("local implementation is not the producer's imported module")
    try:
        tree = ast.parse(producer.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as error:
        raise InputError(f"cannot parse local producer: {error}") from error
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    if "fomhd.boundary_patch" not in imports:
        raise InputError("local producer does not import fomhd.boundary_patch")

    return {
        "local_json": str(local_json),
        "local_json_sha256": hashes["corrected_json"],
        "producer": str(producer),
        "producer_sha256": hashes["probe"],
        "producer_import": "fomhd.boundary_patch",
        "implementation": str(implementation),
        "implementation_sha256": hashes["implementation"],
        "status": str(status_path),
        "status_sha256": sha256_file(status_path),
    }


def validate_geometry_probe_script(
    geometry_path: Path,
    data: dict[str, Any],
) -> Path:
    """Verify the supplied probe-script hash beside the geometry JSON."""

    script = geometry_path.resolve().parent / "probe_boundary_geometry.py"
    if not script.is_file():
        raise InputError(f"geometry probe script is missing: {script}")
    if sha256_file(script) != data["script_sha256"]:
        raise InputError("geometry probe script SHA-256 differs from its JSON")
    return script


def validate_output(
    input_paths: list[Path],
    source_root: Path,
    output: Path,
) -> None:
    """Require a new output outside inputs and the read-only source tree."""

    output = output.resolve()
    source_root = source_root.resolve()
    if output.exists():
        raise InputError(f"refusing existing output path: {output}")
    if not output.parent.is_dir():
        raise InputError(f"output parent directory is missing: {output.parent}")
    if output.is_relative_to(source_root):
        raise InputError("output must remain outside the read-only source tree")
    for raw in input_paths:
        path = raw.resolve()
        if (
            output == path
            or output.is_relative_to(path)
            or path.is_relative_to(output)
        ):
            raise InputError(f"output contains or is contained by input: {path}")


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _source_modules(source_root: Path) -> tuple[ModuleType, ModuleType]:
    source = source_root.resolve()
    sys.path.insert(0, str(source / "src"))
    geometry_module = __import__("fomhd.geometry", fromlist=["build_geometry"])
    boundary_module = __import__(
        "fomhd.boundary_geometry",
        fromlist=["build_boundary_geometry"],
    )
    return geometry_module, boundary_module


def _style_modules(source_root: Path) -> tuple[ModuleType, ModuleType]:
    a0 = _load_module(
        "clt_gallery_a0_renderer",
        source_root / "tools/plots/build_a0_report.py",
    )
    a1 = _load_module(
        "clt_gallery_a1_style",
        source_root / "tools/plots/build_a1_t0_report.py",
    )
    if sha256_file(source_root / "tools/plots/build_a0_report.py") != (
        a1.CANONICAL_HASHES["renderer"]
    ):
        raise InputError("A0 renderer hash differs from the reviewed A1 style source")
    return a0, a1


def _source_commit(source_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(source_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "path": str(source_root),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--short"),
    }


def normalized_derivative_l1(
    weights: Any,
    h_m: float,
    order: int,
) -> float:
    """Return ``||D_phys||_1*h^p`` for dimensionless source weights."""

    h_value = _finite(h_m, "h_m")
    if h_value <= 0.0:
        raise InputError("h_m must be positive")
    if order not in (1, 2):
        raise InputError("derivative order must be 1 or 2")
    dimensionless = np.asarray(weights, dtype=np.float64)
    if dimensionless.ndim != 1 or not np.all(np.isfinite(dimensionless)):
        raise InputError("derivative weights must be a finite 1-D array")
    physical = dimensionless / h_value**order
    return float(np.sum(np.abs(physical)) * h_value**order)


def build_source_profile(
    coefficients: ModuleType,
    source_path: Path,
) -> dict[str, Any]:
    """Evaluate promoted dimensionless source weights for Figure 5."""

    source_path = source_path.resolve()
    four_offsets = np.linspace(0.5, 1.49, 199)
    five_offsets = np.linspace(1.5, 2.49, 199)
    offsets = np.concatenate((four_offsets, five_offsets))
    panels: dict[str, Any] = {"a": {"series": []}, "b": {"series": []}}
    for side in ("upper", "lower"):
        first: list[float] = []
        second: list[float] = []
        rule: list[str] = []
        for offset in offsets:
            offset_value = float(offset)
            selected = coefficients.irregular_rule_for_offset(offset_value)
            stencil = coefficients.irregular_stencil(
                selected,
                offset_value,
                side=side,
            )
            first.append(float(np.sum(np.abs(stencil.first_derivative_weights))))
            second.append(float(np.sum(np.abs(stencil.second_derivative_weights))))
            rule.append(selected)
        panels["a"]["series"].append(
            {
                "side": side,
                "offset_h": offsets.tolist(),
                "normalized_l1": first,
                "rule": rule,
            }
        )
        panels["b"]["series"].append(
            {
                "side": side,
                "offset_h": offsets.tolist(),
                "normalized_l1": second,
                "rule": rule,
            }
        )
    return {
        "source_file": "tools/clt_boundary_coefficients.py",
        "source_sha256": sha256_file(source_path),
        "normalization": (
            "||D_phys||_1 * h^p == ||dimensionless weights||_1"
        ),
        "dropped_offsets": "below 0.5h omitted, not clamped",
        "branch_threshold_h": 1.5,
        "panels": panels,
    }


def _figure1(
    geometry_data: dict[str, Any],
    source_root: Path,
) -> dict[str, Any]:
    geometry_module, boundary_module = _source_modules(source_root)
    case = next(
        item
        for item in geometry_data["cases"]
        if item["nodes"] == 12 and math.isclose(item["grid_phase_R_h"], 0.17)
    )
    nodes = 12
    h_m = 1.0 / (nodes - 4)
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False) + 0.013
    polygon = np.column_stack(
        (3.0 + 0.5 * np.cos(theta), 0.5 * np.sin(theta))
    )
    R = 2.5 + (np.arange(nodes) - 2.0 + 0.5 + 0.17) * h_m
    Z = -0.5 + (
        np.arange(nodes) - 2.0 + 0.5 - 0.17 * 0.7
    ) * h_m
    geometry = geometry_module.build_geometry(R, Z, polygon)
    intercept = boundary_module.build_boundary_geometry(geometry)
    staircase = boundary_module.build_boundary_geometry(
        geometry,
        kind="staircase",
    )
    if intercept.fingerprint != case["geometry_fingerprint"]:
        raise InputError("rebuilt Figure 1 geometry fingerprint differs from input")
    if len(geometry.cells.keys) != case["Nc"]:
        raise InputError("rebuilt Figure 1 cell count differs from input")

    active = np.asarray(geometry.active, dtype=bool)
    class_shape = intercept.r_class.shape
    active_dense = active.reshape(class_shape, order="F")
    active_R = np.broadcast_to(np.asarray(geometry.R)[:, None], class_shape)
    active_Z = np.broadcast_to(np.asarray(geometry.Z)[None, :], class_shape)

    def centers(mask: np.ndarray) -> tuple[list[float], list[float]]:
        return (
            np.asarray(active_R[mask], dtype=np.float64).tolist(),
            np.asarray(active_Z[mask], dtype=np.float64).tolist(),
        )

    active_R_values, active_Z_values = centers(active_dense)
    irregular_mask = active_dense & (
        (intercept.r_class == 2) | (intercept.z_class == 2)
    )
    irregular_R, irregular_Z = centers(irregular_mask)
    dropped_R_mask = active_dense & (intercept.r_class == 5)
    dropped_Z_mask = active_dense & (intercept.z_class == 5)
    dropped_R_R, dropped_R_Z = centers(dropped_R_mask)
    dropped_Z_R, dropped_Z_Z = centers(dropped_Z_mask)

    links = {"R": [], "Z": []}
    for link in intercept.axis_links:
        if link.interior:
            continue
        links[link.axis].append(
            {
                "origin_R_m": link.endpoint_R_m[0],
                "origin_Z_m": link.endpoint_Z_m[0],
                "wall_R_m": link.endpoint_R_m[1],
                "wall_Z_m": link.endpoint_Z_m[1],
                "component": link.physical_component,
                "length_m": link.length_m,
            }
        )
    wall_R = np.append(intercept.polygon_R_m, intercept.polygon_R_m[0])
    wall_Z = np.append(intercept.polygon_Z_m, intercept.polygon_Z_m[0])
    old_R = np.asarray(staircase.wall_polyline_R_m)
    old_Z = np.asarray(staircase.wall_polyline_Z_m)
    return {
        "case_index": geometry_data["cases"].index(case),
        "fingerprint": intercept.fingerprint,
        "nodes": nodes,
        "h_m": h_m,
        "Nc": len(geometry.cells.keys),
        "active_R_m": active_R_values,
        "active_Z_m": active_Z_values,
        "irregular_R_m": irregular_R,
        "irregular_Z_m": irregular_Z,
        "dropped_R_direction_R_m": dropped_R_R,
        "dropped_R_direction_Z_m": dropped_R_Z,
        "dropped_Z_direction_R_m": dropped_Z_R,
        "dropped_Z_direction_Z_m": dropped_Z_Z,
        "true_wall_R_m": wall_R.tolist(),
        "true_wall_Z_m": wall_Z.tolist(),
        "staircase_R_m": old_R.tolist(),
        "staircase_Z_m": old_Z.tolist(),
        "axis_links": links,
        "metadata_only": (
            "Directional classes and existing-face links are metadata over unchanged "
            "active centres and Cartesian faces; no field state is invented."
        ),
    }


def _figure2(geometry_data: dict[str, Any]) -> dict[str, Any]:
    series = []
    for phase in (0.0, 0.17):
        values = sorted(
            (
                item
                for item in geometry_data["cases"]
                if math.isclose(item["grid_phase_R_h"], phase)
            ),
            key=lambda item: item["nodes"],
        )
        series.append(
            {
                "phase_R_h": phase,
                "nodes": [item["nodes"] for item in values],
                "link_min_over_h": [
                    item["boundary_link_min_over_h"] for item in values
                ],
                "elapsed_wall_seconds": [
                    item["elapsed_wall_seconds"] for item in values
                ],
            }
        )
    return {
        "series": series,
        "grid_recipe": geometry_data["grid_recipe"],
        "record_scope": geometry_data["scope"],
    }


def _figure3(local_data: dict[str, Any]) -> dict[str, Any]:
    cases = sorted(
        local_data["cases"],
        key=lambda case: (case["max_length_m"], case["aspect_ratio"]),
    )
    normal_positive = [
        case for case in cases
        if case["electric"]["normal_reconstruction_linf"] > 0.0
    ]
    tangent_positive = [
        case for case in cases
        if case["electric"]["tangent_reconstruction_linf"] > 0.0
    ]
    aspect_series = []
    for aspect in sorted({case["aspect_ratio"] for case in cases}):
        rows = sorted(
            (case for case in cases if case["aspect_ratio"] == aspect),
            key=lambda case: case["max_length_m"],
        )
        aspect_series.append(
            {
                "aspect_ratio": aspect,
                "max_length_m": [case["max_length_m"] for case in rows],
                "curved_gradient_relative_linf": [
                    case["potential"]["curved_gradient_relative_linf"]
                    for case in rows
                ],
            }
        )
    reciprocal = local_data["reciprocal_current_control"]["cases"]
    return {
        "corrected_sign_output": local_data["repair"]["corrected_sign_output"],
        "not_accepted": local_data["not_accepted"],
        "normal_reconstruction": {
            "max_length_m": [case["max_length_m"] for case in normal_positive],
            "value": [
                case["electric"]["normal_reconstruction_linf"]
                for case in normal_positive
            ],
            "exact_zero_cases_omitted": len(cases) - len(normal_positive),
        },
        "tangent_reconstruction": {
            "max_length_m": [case["max_length_m"] for case in tangent_positive],
            "value": [
                case["electric"]["tangent_reconstruction_linf"]
                for case in tangent_positive
            ],
            "exact_zero_cases_omitted": len(cases) - len(tangent_positive),
        },
        "curved_gradient_by_aspect": aspect_series,
        "reciprocal_power": {
            "h_m": [case["h_m"] for case in reciprocal],
            "retained_moment_power_error": [
                case["retained_moment_power_error"] for case in reciprocal
            ],
            "omitted_moment_power_error": [
                case["omitted_moment_power_error"] for case in reciprocal
            ],
            "relative_weighted_field_error": [
                case["relative_weighted_field_error"] for case in reciprocal
            ],
            "observed_orders": local_data["reciprocal_current_control"][
                "observed_weighted_field_error_order"
            ],
            "omitted_moment_bias_persists": local_data[
                "reciprocal_current_control"
            ]["omitted_moment_bias_persists"],
        },
    }


def _figure4(local_data: dict[str, Any]) -> dict[str, Any]:
    cases = local_data["cases"]
    return {
        "aspect_ratio": [case["aspect_ratio"] for case in cases],
        "max_length_m": [case["max_length_m"] for case in cases],
        "homogeneous_condition": [
            case["metric"]["homogeneous_condition"] for case in cases
        ],
        "lifted_condition": [
            case["metric"]["lifted_condition"] for case in cases
        ],
    }


def build_plot_data(
    geometry_data: dict[str, Any],
    local_data: dict[str, Any],
    source_root: Path,
    rigid_evidence: dict[str, Any],
    formal_rhs: dict[str, Any],
) -> dict[str, Any]:
    """Build every plotted array without reading undeclared scientific data."""

    coefficient_path = source_root / "tools/clt_boundary_coefficients.py"
    coefficients = _load_module(
        "clt_gallery_coefficients",
        coefficient_path,
    )
    return {
        "schema": f"{GALLERY_SCHEMA}_plot_data",
        "scope": SCOPE,
        "units": {
            "geometry": "m",
            "geometry_time": "s",
            "link_over_h": "dimensionless",
            "field_reconstruction": "dimensionless L-infinity",
            "power": "W/rad",
            "condition_number": "dimensionless",
            "normalized_physical_coefficient_l1": (
                "dimensionless ||D_phys||_1*h^p"
            ),
        },
        "figures": {
            FIGURE_NAMES[0]: _figure1(geometry_data, source_root),
            FIGURE_NAMES[1]: _figure2(geometry_data),
            FIGURE_NAMES[2]: _figure3(local_data),
            FIGURE_NAMES[3]: _figure4(local_data),
            FIGURE_NAMES[4]: build_source_profile(coefficients, coefficient_path),
            FIGURE6: rigid_evidence["figure"],
            FIGURE7: formal_rhs["figure"],
        },
    }


def _style_axis_and_label(a1: ModuleType, axis: Any, letter: str) -> None:
    a1.style_axis(axis)
    a1.panel_label(axis, letter)


def render_figures(
    plot_data: dict[str, Any],
    output: Path,
    a1: ModuleType,
) -> None:
    """Render the five ordinary PNG/SVG figures through A1 shared helpers."""

    plt = a1.configure_matplotlib()
    from matplotlib.ticker import NullFormatter

    fig1 = plot_data["figures"][FIGURE_NAMES[0]]
    figure, axis = plt.subplots(figsize=(7.6, 6.4), constrained_layout=True)
    axis.scatter(
        fig1["active_R_m"],
        fig1["active_Z_m"],
        s=13,
        color=a1.PALE_GRAY,
        label="unchanged active cell centres",
        zorder=2,
    )
    axis.scatter(
        fig1["irregular_R_m"],
        fig1["irregular_Z_m"],
        s=34,
        facecolors="none",
        edgecolors=a1.BLUE,
        linewidths=1.0,
        label="retained irregular metadata",
        zorder=4,
    )
    for link in fig1["axis_links"]["R"]:
        axis.plot(
            [link["origin_R_m"], link["wall_R_m"]],
            [link["origin_Z_m"], link["wall_Z_m"]],
            color=a1.BLUE,
            linewidth=0.75,
            alpha=0.75,
            zorder=3,
        )
    for link in fig1["axis_links"]["Z"]:
        axis.plot(
            [link["origin_R_m"], link["wall_R_m"]],
            [link["origin_Z_m"], link["wall_Z_m"]],
            color=a1.ORANGE,
            linewidth=0.75,
            alpha=0.70,
            zorder=3,
        )
    axis.plot([], [], color=a1.BLUE, label="E_R face-to-wall links (metadata)")
    axis.plot([], [], color=a1.ORANGE, label="E_Z face-to-wall links (metadata)")
    axis.scatter(
        fig1["dropped_R_direction_R_m"],
        fig1["dropped_R_direction_Z_m"],
        marker="v",
        s=38,
        color=a1.BLUE,
        edgecolors="white",
        linewidths=0.4,
        label="R-direction dropped metadata",
        zorder=6,
    )
    axis.scatter(
        fig1["dropped_Z_direction_R_m"],
        fig1["dropped_Z_direction_Z_m"],
        marker="^",
        s=38,
        color=a1.ORANGE,
        edgecolors="white",
        linewidths=0.4,
        label="Z-direction dropped metadata",
        zorder=6,
    )
    axis.plot(
        fig1["true_wall_R_m"],
        fig1["true_wall_Z_m"],
        color=a1.ACCENT,
        linewidth=2.0,
        label="true 128-vertex wall",
        zorder=5,
    )
    axis.plot(
        fig1["staircase_R_m"],
        fig1["staircase_Z_m"],
        color=a1.DARK,
        linewidth=1.1,
        linestyle="--",
        label="old staircase identity",
        zorder=5,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.set_title("Circle12, phase 0.17: exact wall, staircase, links, classes")
    _style_axis_and_label(a1, axis, "a")
    axis.legend(loc="upper right", fontsize=6.6, framealpha=0.92)
    a1.save_figure(figure, output, FIGURE_NAMES[0])
    plt.close(figure)

    fig2 = plot_data["figures"][FIGURE_NAMES[1]]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    markers = ("o", "s")
    for marker, series in zip(markers, fig2["series"]):
        label = f"R phase {series['phase_R_h']:.2f}h"
        axes[0].plot(
            series["nodes"],
            series["link_min_over_h"],
            marker=marker,
            color=a1.BLUE if series["phase_R_h"] == 0.0 else a1.ORANGE,
            label=label,
        )
        axes[1].plot(
            series["nodes"],
            series["elapsed_wall_seconds"],
            marker=marker,
            color=a1.BLUE if series["phase_R_h"] == 0.0 else a1.ORANGE,
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("declared enclosing nodes per axis")
    axes[0].set_ylabel("minimum boundary link / h")
    axes[0].set_title("Minimum existing-face link")
    axes[1].set_xlabel("declared enclosing nodes per axis")
    axes[1].set_ylabel("supplied geometry elapsed time [s]")
    axes[1].set_title("Recorded one-time preprocessing")
    for letter, axis in zip(("a", "b"), axes):
        _style_axis_and_label(a1, axis, letter)
        axis.legend()
    a1.save_figure(figure, output, FIGURE_NAMES[1])
    plt.close(figure)

    fig3 = plot_data["figures"][FIGURE_NAMES[2]]
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), constrained_layout=True)
    normal = fig3["normal_reconstruction"]
    tangent = fig3["tangent_reconstruction"]
    refinement_ticks = (0.125, 0.25, 0.5, 1.0)
    axes[0].scatter(
        normal["max_length_m"],
        normal["value"],
        color=a1.BLUE,
        marker="o",
        label="normal E",
    )
    axes[0].scatter(
        tangent["max_length_m"],
        tangent["value"],
        color=a1.ORANGE,
        marker="s",
        label="tangent E",
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xticks(refinement_ticks)
    axes[0].set_xticklabels(("0.125", "0.25", "0.5", "1.0"))
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_xlabel("declared max_length [m]")
    axes[0].set_ylabel("known-field reconstruction L∞")
    axes[0].set_title("Known-field trace reconstruction")
    axes[0].legend()
    colors = (a1.BLUE, a1.ORANGE, a1.ACCENT, a1.DARK, a1.GRAY)
    for color, series in zip(colors, fig3["curved_gradient_by_aspect"]):
        axes[1].plot(
            series["max_length_m"],
            series["curved_gradient_relative_linf"],
            marker="o",
            color=color,
            label=f"aspect={series['aspect_ratio']:g}",
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xticks(refinement_ticks)
    axes[1].set_xticklabels(("0.125", "0.25", "0.5", "1.0"))
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xlabel("declared max_length [m]")
    axes[1].set_ylabel("curved-gradient relative L∞")
    axes[1].set_title("Physical-gradient reconstruction")
    axes[1].legend(fontsize=6.6)
    reciprocal = fig3["reciprocal_power"]
    axes[2].plot(
        reciprocal["h_m"],
        reciprocal["retained_moment_power_error"],
        marker="o",
        color=a1.BLUE,
        label="retained-moment error",
    )
    axes[2].plot(
        reciprocal["h_m"],
        reciprocal["omitted_moment_power_error"],
        marker="s",
        color=a1.ORANGE,
        label="omitted-moment error",
    )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("local control h [m]")
    axes[2].set_ylabel("absolute power error [W/rad]")
    axes[2].set_title("Reciprocal-R current control")
    axes[2].legend()
    for letter, axis in zip(("a", "b", "c"), axes):
        _style_axis_and_label(a1, axis, letter)
    a1.save_figure(figure, output, FIGURE_NAMES[2])
    plt.close(figure)

    fig4 = plot_data["figures"][FIGURE_NAMES[3]]
    figure, axis = plt.subplots(figsize=(6.3, 4.2), constrained_layout=True)
    axis.scatter(
        fig4["aspect_ratio"],
        fig4["homogeneous_condition"],
        color=a1.BLUE,
        marker="o",
        label="homogeneous local Gram",
        s=34,
    )
    axis.scatter(
        fig4["aspect_ratio"],
        fig4["lifted_condition"],
        color=a1.ORANGE,
        marker="s",
        label="lifted local Gram",
        s=34,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("declared local aspect ratio")
    axis.set_ylabel("condition number")
    axis.set_title("Near-tangent local Gram conditioning")
    _style_axis_and_label(a1, axis, "a")
    axis.legend()
    a1.save_figure(figure, output, FIGURE_NAMES[3])
    plt.close(figure)

    fig5 = plot_data["figures"][FIGURE_NAMES[4]]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), constrained_layout=True)
    for axis, panel, letter in zip(axes, ("a", "b"), ("a", "b")):
        for color, series in zip(
            (a1.BLUE, a1.ORANGE),
            fig5["panels"][panel]["series"],
        ):
            axis.plot(
                series["offset_h"],
                series["normalized_l1"],
                color=color,
                linewidth=1.2,
                linestyle="-" if series["side"] == "upper" else "--",
                label=f"{series['side']} side",
            )
        axis.axvline(
            fig5["branch_threshold_h"],
            color=a1.DARK,
            linestyle=":",
            linewidth=1.0,
            label="1.5h branch threshold",
        )
        axis.set_yscale("log")
        axis.set_xlabel("retained boundary offset / h")
        axis.set_ylabel("||D_phys||_1 * h^p (dimensionless)")
        axis.set_title(
            "First derivative" if panel == "a" else "Second derivative"
        )
        _style_axis_and_label(a1, axis, letter)
        axis.legend(fontsize=7)
    a1.save_figure(figure, output, FIGURE_NAMES[4])
    plt.close(figure)
    render_physical_rz_figures(
        plot_data["figures"][FIGURE6],
        plot_data["figures"][FIGURE7],
        output,
        plt,
        a1,
    )


def _figure_record(
    number: int,
    name: str,
    title: str,
    caption: str,
    source_kind: str,
    *,
    components: list[str] | None = None,
    panels: list[tuple[str, str]] | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "number": number,
        "title": title,
        "caption": caption,
        "image": f"{name}.png",
        "svg": f"{name}.svg",
        "data": "data.json",
        "provenance": "provenance.json",
        "source_kind": source_kind,
    }
    if components:
        result["components"] = components
    if panels:
        result["panels"] = [
            {"letter": letter, "description": description}
            for letter, description in panels
        ]
    if status:
        result["status"] = status
    return result


def build_manifest(plot_data: dict[str, Any]) -> dict[str, Any]:
    """Build an A0 renderer manifest with explicit local-evidence scope."""

    fig1 = _figure_record(
        1,
        FIGURE_NAMES[0],
        "Exact circle12 RZ intercept geometry",
        (
            "The true wall, old staircase identity, unchanged active centres, "
            "existing-face links, and directional CLT classes are rebuilt from the "
            "promoted geometry helpers. Links and classes are metadata over the "
            "existing Cartesian layout; no field or evolved state is plotted."
        ),
        "mixed_comparison",
        components=[
            "exact 128-vertex wall",
            "old staircase identity",
            "existing-face link metadata",
            "directional CLT point-class metadata",
        ],
        panels=[("a", "RZ section at nodes=12 and R phase=0.17.")],
        status="Geometry construction and metadata only",
    )
    fig2 = _figure_record(
        2,
        FIGURE_NAMES[1],
        "Declared link scale and preprocessing records",
        (
            "Both panels use only the eight supplied geometry-probe records. "
            "Minimum boundary-link length is normalized by the declared h. Elapsed "
            "time is a one-time preprocessing observation, not an equal-accuracy "
            "performance claim."
        ),
        "verification",
        panels=[
            ("a", "Minimum non-interior axis-link length divided by h."),
            ("b", "Supplied per-case geometry construction elapsed time."),
        ],
        status="Declared geometry records only",
    )
    fig3 = _figure_record(
        3,
        FIGURE_NAMES[2],
        "Known fields, physical gradients, and reciprocal-R power errors",
        (
            "Known-field reconstruction and curved-gradient errors use the corrected "
            "local-pairing JSON. Exact zero error points are omitted from logarithmic "
            "display without clamping. The reciprocal-R control retains both "
            "retained-moment and omitted-moment power errors. These are local checks, "
            "not global current/Ohm closure or boundary acceptance."
        ),
        "verification",
        panels=[
            ("a", "Normal and tangent electric-field reconstruction L-infinity."),
            ("b", "Curved physical-gradient relative L-infinity by aspect ratio."),
            ("c", "Absolute reciprocal-R current power errors."),
        ],
        status="Corrected local prototype; pressure sign repair applied",
    )
    fig4 = _figure_record(
        4,
        FIGURE_NAMES[3],
        "Near-tangent local Gram conditioning",
        (
            "The supplied homogeneous and lifted local Gram condition numbers are "
            "shown against their recorded aspect ratio. This is a local algebra "
            "diagnostic, not a complete Hall-system CFL bound or global matrix "
            "condition claim."
        ),
        "verification",
        panels=[("a", "Local condition number versus aspect ratio.")],
        status="Local coefficient algebra only",
    )
    fig5 = _figure_record(
        5,
        FIGURE_NAMES[4],
        "Source-derived CLT coefficient growth",
        (
            "The promoted CLT coefficient helper is evaluated for both sides at "
            "retained offsets [0.5h, 2.5h). Helpers return dimensionless weights "
            "for nodes_h=x/h, so physical coefficients are weights/h^p and the "
            "plotted ||D_phys||_1*h^p equals ||weights||_1 for every positive h. "
            "Points below 0.5h are omitted, not denominator-clamped. This is a "
            "coefficient-growth diagnostic, not a full Hall-system CFL calculation."
        ),
        "verification",
        panels=[
            ("a", "First-derivative L1 weight scale."),
            ("b", "Second-derivative L1 weight scale."),
        ],
        status="Source coefficient diagnostic; no time-step recommendation",
    )
    fig6 = _figure_record(
        6,
        FIGURE6,
        "Repair3 rigid rotation: old staircase versus corrected positive-M0",
        (
            "Valid repair3 evidence is shown for N8/N12/N16 at t=0. Columns are "
            "old staircase symmetric strain magnitude, corrected positive-M0 "
            "candidate magnitude, and candidate-minus-old. Strain is in s^-1. "
            "The true circle and domain are derived from each R3 fixture; the old "
            "computational wall is drawn as exact active-cell outer edges. "
            "The physical rigid-rotation gate remains failed; this is not physical "
            "acceptance or an evolved state."
        ),
        "verification",
        components=[
            "old staircase rigid-rotation strain",
            "corrected positive-M0 candidate strain",
            "explicit old-to-positive difference",
        ],
        panels=[
            ("a", "N8 old staircase."),
            ("b", "N8 positive-M0 candidate."),
            ("c", "N8 candidate minus old."),
            ("d", "N12 old staircase."),
            ("e", "N12 positive-M0 candidate."),
            ("f", "N12 candidate minus old."),
            ("g", "N16 old staircase."),
            ("h", "N16 positive-M0 candidate."),
            ("i", "N16 candidate minus old."),
        ],
        status="Valid repair3 arrays; physical gate failed",
    )
    fig7 = _figure_record(
        7,
        FIGURE7,
        "Self-circle full initial formal staircase RHS",
        (
            "Raw SI formal initial RHS at t=0 and zero time steps for N8/N12: "
            "|rhs(mom_i)| in kg/(m^2*s^2), rhs(p_perp_e) in Pa/s, and rhs(B_phi) "
            "in T/s. Color limits are shared across grids for each field. The true "
            "LCFS and exact old staircase cell-edge wall are distinguished. No field "
            "difference, subtraction, normalization, or evolved-state interpretation "
            "is applied."
        ),
        "verification",
        components=[
            "formally initialized staircase state",
            "full stage RHS arrays",
            "true LCFS versus old staircase support",
        ],
        panels=[
            ("a", "N8 ion-momentum RHS magnitude."),
            ("b", "N8 perpendicular electron-pressure RHS."),
            ("c", "N8 toroidal magnetic-field RHS."),
            ("d", "N12 ion-momentum RHS magnitude."),
            ("e", "N12 perpendicular electron-pressure RHS."),
            ("f", "N12 toroidal magnetic-field RHS."),
        ],
        status="Initial formal RHS only; no step executed",
    )
    pages = [
        {
            "id": "scope-and-rz",
            "title": "Scope and exact RZ intercept construction",
            "introduction": (
                "This gallery is limited to local geometry, sampling, and paired "
                "reference tests. It does not present a complete evolved state."
            ),
            "notes": [
                SCOPE,
                (
                    "The physical input is the same 128-vertex circle with R0=3m, "
                    "a=0.5m, and theta offset=0.013rad."
                ),
                (
                    "The rebuilt fingerprint matches the supplied geometry-probe "
                    "case before any figure is rendered."
                ),
            ],
            "figures": [fig1],
        },
        {
            "id": "geometry-records",
            "title": "Declared geometry preprocessing",
            "introduction": (
                "The gallery reads the supplied preprocessing records without "
                "introducing another grid scale or rerunning a performance study."
            ),
            "notes": [
                (
                    "The declared grid family stops at 24 nodes per axis and both "
                    "recorded shifts are shown."
                ),
                "Normal B initial data are neither set nor plotted.",
            ],
            "figures": [fig2],
        },
        {
            "id": "local-paired-controls",
            "title": "Corrected local paired-reference controls",
            "introduction": (
                "Known-field, physical-gradient, reciprocal-R current, and local "
                "conditioning checks remain separate from global solver assembly."
            ),
            "notes": [
                (
                    "The corrected pressure sign receipt is required; the v1 JSON "
                    "is not used for scientific claims."
                ),
                (
                    "Polynomial/known-field exactness does not establish missing "
                    "global closure, synchronous RK4, or complete boundary physics."
                ),
            ],
            "figures": [fig3, fig4],
        },
        {
            "id": "source-coefficients",
            "title": "Source-derived coefficient growth",
            "introduction": (
                "The promoted coefficient helper exposes actual nonuniform weights "
                "as a conditioning diagnostic rather than a time-step prescription."
            ),
            "notes": [
                "Dropped points below 0.5h are omitted from this coefficient plot.",
                (
                    "L1*h^order growth is not a full Hall-system CFL bound and does "
                    "not select a production time step."
                ),
            ],
            "figures": [fig5],
        },
        {
            "id": "physical-rz-initial-evidence",
            "title": "Physical RZ initial and rigid evidence",
            "introduction": (
                "These cross-sections use existing initial/rigid evidence only. "
                "They do not represent evolved states or fake initial perturbations."
            ),
            "notes": [
                (
                    "Repair3 is valid corrected positive-M0 evidence, but its "
                    "rigid-rotation physical gate remains failed. Its true circle "
                    "is derived from the R3 cell domain, not the formal self-circle."
                ),
                (
                    "Old-formal RHS arrays are raw SI values at t=0 with zero time "
                    "steps; color limits are shared across N8/N12 for each field."
                ),
                (
                    "Old computational walls are exact active-cell edges with missing "
                    "neighbours, not finite-mask contours. The true circle is "
                    "distinguished from those edges, and no R0/R2 conclusion is used."
                ),
            ],
            "figures": [fig6, fig7],
        },
        {
            "id": "limitations",
            "title": "Optional operator map and acceptance boundary",
            "introduction": (
                "An optional shared-physical VEQ comparison is deferred until the "
                "parent supplies that data."
            ),
            "notes": [
                (
                    "The added physical RZ panels use explicit repair3 rigid and "
                    "old-formal initial-RHS evidence only."
                ),
                (
                    "No global backend, source stage, time step, GPU run, network, "
                    "or scientific acceptance is claimed."
                ),
                (
                    "Optional VEQ shared-physical cross-comparison remains future work "
                    "and consumes no unprovided data."
                ),
                "No R0/R2 comparison conclusion is used.",
                (
                    "The parent owns acceptance and independent reruns. This gallery "
                    "contains local evidence and explicit unit/scope labels only."
                ),
            ],
            "figures": [],
        },
    ]
    return {
        "title": "CLT local intercept evidence gallery",
        "introduction": (
            "This offline English gallery reuses the canonical A0 renderer, CSS, "
            "evidence badges, and A1 Matplotlib style. It covers local geometry, "
            "sampling, paired-reference, and coefficient diagnostics only."
        ),
        "notation": [
            {
                "symbol": "R, Z",
                "meaning": "Cylindrical major-radius and axial coordinates in metres.",
            },
            {
                "symbol": "R0, a",
                "meaning": (
                    "Circle centre R=3m; a=.5m for geometry/equilibrium inputs, "
                    "a=1m for the separate rigid-rotation fixture."
                ),
            },
            {"symbol": "h", "meaning": "Declared uniform R/Z cell-centre spacing."},
            {"symbol": "Nc", "meaning": "Number of unchanged active structured cells."},
            {"symbol": "L1", "meaning": "Sum of absolute derivative-stencil weights."},
            {
                "symbol": "VER / MIX",
                "meaning": (
                    "Controlled verification evidence; explicitly separated mixed "
                    "local components."
                ),
            },
        ],
        "archive": {
            "path": "data.json",
            "label": "Raw plotted arrays, units, series labels, and source scope",
        },
        "pages": pages,
        "conclusions": (
            "The gallery records exact local intercept geometry, existing-face link "
            "scale, corrected paired-reference errors, local near-tangent "
            "conditioning, and source-derived coefficient growth. It does not "
            "establish complete evolution, global closure, a full Hall CFL limit, or "
            "scientific acceptance."
        ),
        "known_issues": [
            {
                "path": "provenance.json",
                "label": "Input/source/renderer hashes, scope, and output policy",
            },
            {
                "path": "sha256-manifest.json",
                "label": "Independent input/output SHA-256 records",
            },
        ],
    }


def embed_svgs(rendered: str, output: Path) -> str:
    for name in FIGURE_NAMES:
        payload = base64.b64encode((output / f"{name}.svg").read_bytes()).decode(
            "ascii"
        )
        needle = f'src="{name}.png"'
        if rendered.count(needle) != 1:
            raise RuntimeError(f"expected one rendered image source for {name}")
        rendered = rendered.replace(
            needle,
            f'src="data:image/svg+xml;base64,{payload}"',
        )
    if rendered.count('src="data:image/svg+xml;base64,') != len(FIGURE_NAMES):
        raise RuntimeError("not every gallery image is embedded SVG")
    return rendered


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _input_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "role": role,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _write_outputs(
    temp: Path,
    geometry_path: Path,
    local_path: Path,
    producer_path: Path,
    implementation_path: Path,
    local_status_path: Path,
    source_root: Path,
    geometry_data: dict[str, Any],
    local_data: dict[str, Any],
    lineage: dict[str, Any],
    rigid_evidence: dict[str, Any],
    formal_rhs: dict[str, Any],
    plot_data: dict[str, Any],
    a0: ModuleType,
    a1: ModuleType,
    started: float,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    builder_path = Path(__file__).resolve()
    receipt = local_data["provenance_receipt"]
    geometry_script = validate_geometry_probe_script(
        geometry_path,
        geometry_data,
    )
    inputs = [
        _input_record(geometry_path, "geometry probes"),
        _input_record(geometry_script, "independent geometry probe script"),
        _input_record(local_path, "corrected local-pairing repair1"),
        _input_record(producer_path, "corrected local-data producer"),
        _input_record(
            implementation_path,
            "producer-imported local boundary patch",
        ),
        _input_record(local_status_path, "existing local-data STATUS binding"),
        _input_record(
            Path(receipt["path"]),
            "independent local quadrature provenance receipt",
        ),
        _input_record(source_root / "src/fomhd/geometry.py", "geometry builder"),
        _input_record(
            source_root / "src/fomhd/boundary_geometry.py",
            "promoted boundary geometry",
        ),
        _input_record(
            source_root / "tools/clt_boundary_coefficients.py",
            "promoted CLT coefficient helper",
        ),
        _input_record(
            source_root / "tools/plots/build_a0_report.py",
            "canonical A0 renderer and CSS",
        ),
        _input_record(
            source_root / "tools/plots/build_a1_t0_report.py",
            "A1 Matplotlib style helpers",
        ),
        *[
            _input_record(Path(item["path"]), item["role"])
            for item in rigid_evidence["inputs"] + formal_rhs["inputs"]
        ],
    ]
    source_receipt = _source_commit(source_root)
    source_hashes_before = {
        item["path"]: item["sha256"] for item in inputs
    }
    input_hashes_before = {
        str(geometry_path.resolve()): sha256_file(geometry_path),
        str(local_path.resolve()): sha256_file(local_path),
    }
    provenance_directory = temp / "provenance"
    provenance_directory.mkdir()
    retained_local = (
        (producer_path, "local_data_probe.py"),
        (implementation_path, "local_data_boundary_patch.py"),
        (local_status_path, "local_data_STATUS.json"),
        (local_path, "local_pairing_probe_repair1.json"),
    )
    physical_copies = tuple(
        (Path(item["source"]), item["copy"])
        for item in rigid_evidence["copies"] + formal_rhs["copies"]
    )
    for source, name in retained_local + physical_copies:
        destination = provenance_directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destination.chmod(0o444)
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"retained local-data copy hash mismatch: {source}")
    _write_json(temp / "data.json", plot_data)
    render_figures(plot_data, temp, a1)
    manifest = build_manifest(plot_data)
    manifest_path = temp / "manifest.json"
    _write_json(manifest_path, manifest)

    provisional_provenance = {
        "schema": f"{GALLERY_SCHEMA}_provenance",
        "status": "rendering",
        "scope": SCOPE,
    }
    provisional_hash_manifest = {
        "schema": f"{GALLERY_SCHEMA}_sha256_manifest",
        "status": "rendering",
        "scope": {
            "complete_evolution_acceptance": False,
            "optional_slot_6": "deferred_veq_parent_data",
        },
        "inputs": inputs,
        "outputs": [],
    }
    _write_json(temp / "provenance.json", provisional_provenance)
    _write_json(temp / "sha256-manifest.json", provisional_hash_manifest)
    html_path = temp / "index.html"
    rendered = a0.render_report(manifest, manifest_path, html_path)
    rendered = embed_svgs(rendered, temp)
    html_path.write_text(rendered, encoding="utf-8")
    if a1.style_text(html_path) != a1.style_text(
        source_root / "tools/plots/build_a0_report.py"
    ):
        raise RuntimeError("rendered HTML CSS differs from canonical A0 style")
    if "http://" in rendered.casefold() or "https://" in rendered.casefold():
        raise RuntimeError("offline gallery contains an external URL")

    elapsed = time.monotonic() - started
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if elapsed > MAX_WALL_SECONDS:
        raise RuntimeError(f"wall budget exceeded: {elapsed:.3f}s")
    if max_rss > MAX_RSS_BYTES:
        raise RuntimeError(f"memory budget exceeded: {max_rss} bytes")

    hashed_artifacts = [
        html_path,
        manifest_path,
        temp / "data.json",
        *[
            temp / f"{name}.{suffix}"
            for name in FIGURE_NAMES
            for suffix in ("png", "svg")
        ],
        *[
            provenance_directory / name
            for _, name in retained_local + physical_copies
        ],
    ]
    provenance = {
        "schema": f"{GALLERY_SCHEMA}_provenance",
        "status": "complete",
        "generated_utc": datetime.now(UTC).isoformat(),
        "scope": SCOPE,
        "source": source_receipt,
        "inputs": inputs,
        "builder": {
            "path": str(builder_path),
            "sha256": sha256_file(builder_path),
        },
        "canonical_reuse": {
            "a0_renderer": str(
                (source_root / "tools/plots/build_a0_report.py").resolve()
            ),
            "a0_renderer_sha256": a1.CANONICAL_HASHES["renderer"],
            "a1_style_helpers": str(
                (source_root / "tools/plots/build_a1_t0_report.py").resolve()
            ),
            "css_matches_a0": True,
            "matplotlib_helpers": [
                "configure_matplotlib",
                "style_axis",
                "panel_label",
                "save_figure",
            ],
        },
        "policy": {
            "figures": len(FIGURE_NAMES),
            "optional_slot_6": "deferred_veq_parent_data",
            "complete_evolution_acceptance": False,
            "solver_used": False,
            "gpu_used": False,
            "network_used": False,
            "new_dependencies": False,
            "source_checkout_modified_by_builder": False,
            "required_source_hashes_stable_during_render": True,
            "initial_B_set_to_zero": False,
            "global_backend_or_time_step_selected": False,
            "grid_nodes_max": max(case["nodes"] for case in geometry_data["cases"]),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "elapsed_wall_seconds": elapsed,
            "max_rss_bytes": max_rss,
            "raw_arrays": "data.json",
            "optional_slot_6_reason": (
                "Optional shared-physical VEQ comparison awaits parent-supplied data."
            ),
        },
        "local_not_accepted": local_data["not_accepted"],
        "local_data_lineage": {
            **lineage,
            "retained_copies": [
                {
                    "source": str(source.resolve()),
                    "copy": f"provenance/{name}",
                    "sha256": sha256_file(source),
                }
                for source, name in retained_local
            ],
        },
        "physical_rz_evidence": {
            "repair3_rigid": rigid_evidence["lineage"],
            "old_formal_initial_rhs": formal_rhs["lineage"],
            "retained_copies": [
                {
                    "source": str(source.resolve()),
                    "copy": f"provenance/{name}",
                    "sha256": sha256_file(source),
                }
                for source, name in physical_copies
            ],
            "scope": (
                "initial RHS and rigid tests only; no evolved state or solver call"
            ),
        },
        "artifacts": [_record(path, temp) for path in hashed_artifacts],
    }
    _write_json(temp / "provenance.json", provenance)
    outputs = [_record(path, temp) for path in hashed_artifacts]
    outputs.append(_record(temp / "provenance.json", temp))
    hash_manifest = {
        "schema": f"{GALLERY_SCHEMA}_sha256_manifest",
        "status": "complete",
        "scope": {
            "complete_evolution_acceptance": False,
            "optional_slot_6": "deferred_veq_parent_data",
        },
        "inputs": inputs,
        "outputs": outputs,
        "self_sha256_omitted": True,
    }
    _write_json(temp / "sha256-manifest.json", hash_manifest)

    for item in inputs:
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"input changed during rendering: {path}")
    for relative, digest in source_hashes_before.items():
        if (
            relative not in input_hashes_before
            and sha256_file(Path(relative)) != digest
        ):
            raise RuntimeError(f"source changed during rendering: {relative}")
    if any(
        sha256_file(Path(path)) != digest
        for path, digest in input_hashes_before.items()
    ):
        raise RuntimeError("source JSON changed during rendering")
    if sha256_file(builder_path) != provenance["builder"]["sha256"]:
        raise RuntimeError("builder changed during rendering")
    return {
        "figures": len(FIGURE_NAMES),
        "scope": SCOPE,
        "source_commit": source_receipt["commit"],
    }


def build(
    geometry_path: Path,
    local_path: Path,
    local_producer: Path,
    local_implementation: Path,
    local_status: Path,
    structured_json: Path,
    structured_npz: Path,
    structured_producer: Path,
    structured_status: Path,
    old_formal_root: Path,
    old_formal_audit: Path,
    old_formal_status: Path,
    old_formal_producer: Path,
    old_formal_implementation: Path,
    source_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Validate, render, verify, and atomically publish a new gallery."""

    started = time.monotonic()
    geometry_path = geometry_path.resolve()
    local_path = local_path.resolve()
    local_producer = local_producer.resolve()
    local_implementation = local_implementation.resolve()
    local_status = local_status.resolve()
    structured_json = structured_json.resolve()
    structured_npz = structured_npz.resolve()
    structured_producer = structured_producer.resolve()
    structured_status = structured_status.resolve()
    old_formal_root = old_formal_root.resolve()
    old_formal_audit = old_formal_audit.resolve()
    old_formal_status = old_formal_status.resolve()
    old_formal_producer = old_formal_producer.resolve()
    old_formal_implementation = old_formal_implementation.resolve()
    source_root = source_root.resolve()
    output = output.resolve()
    for path in (
        geometry_path,
        local_path,
        local_producer,
        local_implementation,
        local_status,
        structured_json,
        structured_npz,
        structured_producer,
        structured_status,
        old_formal_audit,
        old_formal_status,
        old_formal_producer,
        old_formal_implementation,
    ):
        if not path.is_file():
            raise InputError(f"required input is not a file: {path}")
    if not old_formal_root.is_dir():
        raise InputError(f"old-formal evidence root is missing: {old_formal_root}")
    validate_output(
        [
            geometry_path,
            local_path,
            local_producer,
            local_implementation,
            local_status,
            structured_json,
            structured_npz,
            structured_producer,
            structured_status,
            old_formal_root,
            old_formal_audit,
            old_formal_status,
            old_formal_producer,
            old_formal_implementation,
        ],
        source_root,
        output,
    )

    geometry_data = _load_json(geometry_path)
    local_data = _load_json(local_path)
    validate_geometry_probes(geometry_data, source_root)
    validate_local_pairing(local_data)
    lineage = validate_local_lineage(
        local_path,
        local_producer,
        local_implementation,
        local_status,
    )
    rigid_evidence = load_repair3_rigid_evidence(
        structured_json,
        structured_npz,
        structured_producer,
        structured_status,
    )
    formal_rhs = load_old_formal_evidence(
        old_formal_root,
        old_formal_audit,
        old_formal_status,
        old_formal_producer,
        old_formal_implementation,
    )
    a0, a1 = _style_modules(source_root)
    plot_data = build_plot_data(
        geometry_data,
        local_data,
        source_root,
        rigid_evidence,
        formal_rhs,
    )

    temp = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    old_mpconfig = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = str(temp / ".mplconfig")
    try:
        result = _write_outputs(
            temp,
            geometry_path,
            local_path,
            local_producer,
            local_implementation,
            local_status,
            source_root,
            geometry_data,
            local_data,
            lineage,
            rigid_evidence,
            formal_rhs,
            plot_data,
            a0,
            a1,
            started,
        )
        shutil.rmtree(temp / ".mplconfig", ignore_errors=True)
        for name in FIGURE_NAMES:
            for suffix in ("png", "svg"):
                path = temp / f"{name}.{suffix}"
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"missing figure: {path}")
        temp.replace(output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    finally:
        if old_mpconfig is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = old_mpconfig
    result["output"] = str(output)
    result["elapsed_wall_seconds"] = time.monotonic() - started
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-probes", type=Path, required=True)
    parser.add_argument("--local-pairing", type=Path, required=True)
    parser.add_argument("--local-producer", type=Path, required=True)
    parser.add_argument("--local-implementation", type=Path, required=True)
    parser.add_argument("--local-status", type=Path, required=True)
    parser.add_argument("--structured-json", type=Path, required=True)
    parser.add_argument("--structured-npz", type=Path, required=True)
    parser.add_argument("--structured-producer", type=Path, required=True)
    parser.add_argument("--structured-status", type=Path, required=True)
    parser.add_argument("--old-formal-root", type=Path, required=True)
    parser.add_argument("--old-formal-audit", type=Path, required=True)
    parser.add_argument("--old-formal-status", type=Path, required=True)
    parser.add_argument("--old-formal-producer", type=Path, required=True)
    parser.add_argument("--old-formal-implementation", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build(
            args.geometry_probes,
            args.local_pairing,
            args.local_producer,
            args.local_implementation,
            args.local_status,
            args.structured_json,
            args.structured_npz,
            args.structured_producer,
            args.structured_status,
            args.old_formal_root,
            args.old_formal_audit,
            args.old_formal_status,
            args.old_formal_producer,
            args.old_formal_implementation,
            args.source_root,
            args.output,
        )
    except (InputError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
