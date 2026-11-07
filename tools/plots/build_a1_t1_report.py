#!/usr/bin/env python3
"""Build an offline A0-style gallery for bounded A1 T1 benchmark data."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import itertools
import json
import math
import os
import re
import shutil
import sys
import tempfile
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BLUE = "#0072B2"
ORANGE = "#D55E00"
TEAL = "#00796B"
DARK = "#37474F"
GRAY = "#6B7280"
PALE_GRAY = "#B0BEC5"
T0_PRIMARY_COUPLING = "predictor"
T1_PRIMARY_COUPLING = "waveform"

FIGURE_NAMES = (
    "fig01_linear_seed_maps",
    "fig02_t0_t1_matrix",
    "fig03_high_gain_history",
    "fig04_iteration_residuals",
    "fig05_energy_residuals",
    "fig06_refinement_tolerance",
    "fig07_work_storage_gpu",
)

A0_HTML_DEFAULT = Path(
    "/home/lxzeng/runs/fomhd/a0-benchmark-figures__20261104T050738Z/index.html"
)
STYLE_RE = re.compile(r"<style>(.*?)</style>", re.DOTALL)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
HEX_RE = re.compile(r"[0-9a-fA-F]{64}")
COMMIT_RE = re.compile(r"[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?")

CASE_KEYS = {
    "id",
    "coupling",
    "M",
    "seed",
    "micro_per_period",
    "rtol",
    "completed",
    "status",
    "time_over_T",
    "target_error",
    "high_norm_ratio",
    "high_absolute_norm",
    "total_energy_residual_J",
    "kinetic_energy_residual_J",
    "energy_time_over_T",
    "max_target_error",
    "max_high_gain",
    "final_high_gain",
    "sweeps_min",
    "sweeps_max",
    "model_calls",
    "estimated_peak_bytes",
    "elapsed_s",
    "iterations",
    "failure_message",
    "rejected_time_over_T",
}

LINEAR_KEYS = {
    "M",
    "positive_branch_modulus",
    "eigenmode_power_gain",
    "actual_seed_final_gain",
    "operator_power_norm",
    "production_gain",
}


class InputError(ValueError):
    """Raised when compact data or a referenced source violates the contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise InputError(f"nonfinite JSON constant is not allowed: {value}")


def _unique_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def load_json(path: Path, where: str) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object_pairs,
        )
    except json.JSONDecodeError as error:
        raise InputError(f"{where} is not valid JSON: {error}") from error


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputError(f"{where} must be an object")
    return value


def _items(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise InputError(f"{where} must be an array")
    return value


def _text(mapping: dict[str, Any], key: str, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{where}.{key} must be nonempty text")
    if CJK_RE.search(value):
        raise InputError(f"{where}.{key} must be English")
    return value


def _text_value(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{where} must be nonempty text")
    if CJK_RE.search(value):
        raise InputError(f"{where} must be English")
    return value


def _nullable_text(mapping: dict[str, Any], key: str, where: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    return _text(mapping, key, where)


def _bool(mapping: dict[str, Any], key: str, where: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise InputError(f"{where}.{key} must be true or false")
    return value


def _integer(mapping: dict[str, Any], key: str, where: str, minimum: int = 1) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InputError(f"{where}.{key} must be an integer >= {minimum}")
    return value


def _number(
    value: Any,
    where: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{where} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise InputError(f"{where} must be finite")
    if minimum is not None and result < minimum:
        raise InputError(f"{where} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise InputError(f"{where} must be <= {maximum}")
    return result


def _nullable_number(
    mapping: dict[str, Any],
    key: str,
    where: str,
    *,
    minimum: float | None = None,
) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    return _number(value, f"{where}.{key}", minimum=minimum)


def _numbers(
    mapping: dict[str, Any],
    key: str,
    where: str,
    *,
    expected: int | None = None,
    minimum: float | None = None,
) -> list[float]:
    raw = _items(mapping.get(key), f"{where}.{key}")
    if expected is not None and len(raw) != expected:
        raise InputError(
            f"{where}.{key} has length {len(raw)}; expected {expected}"
        )
    return [
        _number(value, f"{where}.{key}[{index}]", minimum=minimum)
        for index, value in enumerate(raw)
    ]


def _nullable_numbers(
    mapping: dict[str, Any],
    key: str,
    where: str,
    *,
    expected: int,
    minimum: float | None = None,
) -> list[float] | None:
    value = mapping.get(key)
    if value is None:
        return None
    raw = _items(value, f"{where}.{key}")
    if raw and expected and len(raw) != expected:
        raise InputError(
            f"{where}.{key} has length {len(raw)}; expected 0 or {expected}"
        )
    return [
        _number(item, f"{where}.{key}[{index}]", minimum=minimum)
        for index, item in enumerate(raw)
    ]


def _nondecreasing(values: list[float], where: str) -> None:
    if any(right < left for left, right in itertools.pairwise(values)):
        raise InputError(f"{where} must be nondecreasing")


def _require_keys(mapping: dict[str, Any], keys: set[str], where: str) -> None:
    missing = sorted(keys.difference(mapping))
    if missing:
        raise InputError(f"{where} is missing required keys: {', '.join(missing)}")


def _validate_protocol(data: dict[str, Any]) -> dict[str, Any]:
    protocol = _object(data.get("protocol"), "protocol")
    keys = {
        "T_s",
        "h_s",
        "cycles",
        "rtol",
        "max_iterations",
        "target_error_limit",
        "phase_limit_rad",
        "amplitude_limit",
        "growth_limit",
        "energy_budget_J",
    }
    _require_keys(protocol, keys, "protocol")
    if _number(protocol["T_s"], "protocol.T_s", minimum=0.0) <= 0.0:
        raise InputError("protocol.T_s must be positive")
    if _number(protocol["h_s"], "protocol.h_s", minimum=0.0) <= 0.0:
        raise InputError("protocol.h_s must be positive")
    if _integer(protocol, "cycles", "protocol") != 2:
        raise InputError("protocol.cycles must be 2")
    if _number(protocol["rtol"], "protocol.rtol", minimum=0.0) <= 0.0:
        raise InputError("protocol.rtol must be positive")
    if _integer(protocol, "max_iterations", "protocol") != 12:
        raise InputError("protocol.max_iterations must be 12")
    expected = {
        "target_error_limit": 0.01,
        "phase_limit_rad": 0.1,
        "amplitude_limit": 0.01,
        "growth_limit": 1.01,
        "energy_budget_J": 8.0e-9,
    }
    for key, expected_value in expected.items():
        value = _number(protocol[key], f"protocol.{key}", minimum=0.0)
        if not math.isclose(value, expected_value, rel_tol=0.0, abs_tol=0.0):
            raise InputError(f"protocol.{key} must be {expected_value!r}")
    return protocol


def _validate_linear(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, raw in enumerate(_items(data.get("linear"), "linear")):
        where = f"linear[{index}]"
        row = _object(raw, where)
        _require_keys(row, LINEAR_KEYS, where)
        M = _integer(row, "M", where)
        if M in seen:
            raise InputError(f"duplicate linear M={M}")
        seen.add(M)
        for key in LINEAR_KEYS.difference({"M"}):
            if _number(row[key], f"{where}.{key}", minimum=0.0) <= 0.0:
                raise InputError(f"{where}.{key} must be positive")
        rows.append(row)
    missing = {4, 8, 16, 32}.difference(seen)
    if missing:
        raise InputError(
            f"linear is missing required M rows: {sorted(missing)}"
        )
    return rows


def _validate_iterations(
    case: dict[str, Any],
    where: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    keys = {
        "macro_index",
        "iteration",
        "path_max",
        "endpoint_max",
        "total_energy_residual_J",
        "kinetic_energy_residual_J",
    }
    for index, raw in enumerate(_items(case.get("iterations"), f"{where}.iterations")):
        row_where = f"{where}.iterations[{index}]"
        row = _object(raw, row_where)
        _require_keys(row, keys, row_where)
        macro = _integer(row, "macro_index", row_where, minimum=0)
        iteration = _integer(row, "iteration", row_where, minimum=0)
        key = (macro, iteration)
        if key in seen:
            raise InputError(f"{where} has duplicate iteration keys")
        seen.add(key)
        _number(row["path_max"], f"{row_where}.path_max", minimum=0.0)
        _number(row["endpoint_max"], f"{row_where}.endpoint_max", minimum=0.0)
        _nullable_number(row, "total_energy_residual_J", row_where)
        _nullable_number(row, "kinetic_energy_residual_J", row_where)
        rows.append(row)
    return rows


def _validate_cases(data: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(_items(data.get("cases"), "cases")):
        where = f"cases[{index}]"
        case = _object(raw, where)
        _require_keys(case, CASE_KEYS, where)
        case_id = _text(case, "id", where)
        if case_id in seen_ids:
            raise InputError(f"duplicate case id: {case_id}")
        seen_ids.add(case_id)

        coupling = _text(case, "coupling", where)
        if coupling not in {"predictor", "waveform", "synchronous"}:
            raise InputError(f"{where}.coupling is invalid")
        seed = _text(case, "seed", where)
        if seed not in {"target", "mixed"}:
            raise InputError(f"{where}.seed is invalid")
        _integer(case, "M", where)
        micro = _integer(case, "micro_per_period", where)
        if micro not in {2048, 4096}:
            raise InputError(f"{where}.micro_per_period must be 2048 or 4096")
        rtol = _nullable_number(case, "rtol", where, minimum=0.0)
        if rtol is not None and rtol <= 0.0:
            raise InputError(f"{where}.rtol must be positive or null")
        completed = _bool(case, "completed", where)
        _text(case, "status", where)

        times = _numbers(case, "time_over_T", where, minimum=0.0)
        _nondecreasing(times, f"{where}.time_over_T")
        count = len(times)
        _numbers(case, "target_error", where, expected=count, minimum=0.0)
        _nullable_numbers(
            case,
            "high_norm_ratio",
            where,
            expected=count,
            minimum=0.0,
        )
        _nullable_numbers(
            case,
            "high_absolute_norm",
            where,
            expected=count,
            minimum=0.0,
        )

        energy_times = _numbers(
            case,
            "energy_time_over_T",
            where,
            minimum=0.0,
        )
        _nondecreasing(energy_times, f"{where}.energy_time_over_T")
        energy_count = len(energy_times)
        _numbers(
            case,
            "total_energy_residual_J",
            where,
            expected=energy_count,
        )
        _numbers(
            case,
            "kinetic_energy_residual_J",
            where,
            expected=energy_count,
        )

        for key in (
            "max_target_error",
            "max_high_gain",
            "final_high_gain",
            "sweeps_min",
            "sweeps_max",
            "model_calls",
            "estimated_peak_bytes",
            "elapsed_s",
        ):
            _nullable_number(case, key, where, minimum=0.0)
        _nullable_text(case, "failure_message", where)
        rejected_time = _nullable_number(
            case,
            "rejected_time_over_T",
            where,
            minimum=0.0,
        )
        if completed and rejected_time is not None:
            raise InputError(f"{where}.rejected_time_over_T must be null if completed")
        if not completed and rejected_time is None:
            raise InputError(f"{where}.rejected_time_over_T is required if incomplete")

        if seed == "target":
            ratio = case["high_norm_ratio"]
            if ratio:
                raise InputError(
                    f"{where}.high_norm_ratio must be null or empty for target-only seed"
                )
            if case["max_high_gain"] is not None:
                raise InputError(
                    f"{where}.max_high_gain must be null for target-only seed"
                )
            if case["final_high_gain"] is not None:
                raise InputError(
                    f"{where}.final_high_gain must be null for target-only seed"
                )

        _validate_iterations(case, where)
        cases.append(case)
    return cases


def _validate_gpu(data: dict[str, Any]) -> dict[str, Any]:
    gpu = _object(data.get("gpu"), "gpu")
    _require_keys(gpu, {"status", "notes", "parity_rows"}, "gpu")
    _text(gpu, "status", "gpu")
    for index, note in enumerate(_items(gpu.get("notes"), "gpu.notes")):
        _text_value(note, f"gpu.notes[{index}]")
    seen: set[tuple[int, str]] = set()
    for index, raw in enumerate(_items(gpu.get("parity_rows"), "gpu.parity_rows")):
        where = f"gpu.parity_rows[{index}]"
        row = _object(raw, where)
        keys = {
            "M",
            "accepted_macros",
            "relative_state_error",
            "relative_mode_error",
            "device",
        }
        _require_keys(row, keys, where)
        M = _integer(row, "M", where)
        _integer(row, "accepted_macros", where, minimum=0)
        _number(
            row["relative_state_error"],
            f"{where}.relative_state_error",
            minimum=0.0,
        )
        _number(
            row["relative_mode_error"],
            f"{where}.relative_mode_error",
            minimum=0.0,
        )
        device = _text(row, "device", where)
        key = (M, device)
        if key in seen:
            raise InputError(f"duplicate GPU parity row for M={M}, device={device}")
        seen.add(key)
    return gpu


def _validate_provenance(data: dict[str, Any]) -> dict[str, Any]:
    provenance = _object(data.get("provenance"), "provenance")
    keys = {
        "source_commit",
        "source_hashes",
        "evidence_root",
        "baseline_root",
        "notes",
    }
    _require_keys(provenance, keys, "provenance")
    commit = _text(provenance, "source_commit", "provenance")
    if not COMMIT_RE.fullmatch(commit):
        raise InputError("provenance.source_commit must be a Git SHA")
    source_hashes = _object(
        provenance.get("source_hashes"),
        "provenance.source_hashes",
    )
    for path, digest in source_hashes.items():
        _text_value(path, f"provenance.source_hashes key {path!r}")
        _text_value(digest, f"provenance.source_hashes[{path!r}]")
        if not HEX_RE.fullmatch(digest):
            raise InputError(
                f"provenance.source_hashes[{path!r}] must be a SHA-256 digest"
            )
    _text(provenance, "evidence_root", "provenance")
    _text(provenance, "baseline_root", "provenance")
    for index, note in enumerate(_items(provenance.get("notes"), "provenance.notes")):
        _text_value(note, f"provenance.notes[{index}]")
    return provenance


def validate_data(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and return the fixed protocol without changing scientific values."""
    data = _object(data, "data")
    required = {
        "protocol",
        "t0_gallery",
        "linear",
        "cases",
        "gpu",
        "provenance",
    }
    _require_keys(data, required, "data")
    protocol = _validate_protocol(data)
    gallery = _text(data, "t0_gallery", "data")
    if not Path(gallery).is_absolute():
        raise InputError("data.t0_gallery must be an absolute path")
    _validate_linear(data)
    if not _validate_cases(data):
        raise InputError("cases must not be empty")
    _validate_gpu(data)
    _validate_provenance(data)
    return protocol


def resolve_local_path(value: str, base: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _t0_data_path(gallery: Path) -> Path:
    if gallery.is_file():
        if gallery.suffix.casefold() == ".json":
            return gallery
        return gallery.parent / "data.json"
    if gallery.is_dir():
        return gallery / "data.json"
    raise InputError(f"t0_gallery does not exist: {gallery}")


def _validate_t0_case(raw: Any, where: str) -> dict[str, Any]:
    case = _object(raw, where)
    for key in ("M", "seed", "coupling", "micro_per_period"):
        if key not in case:
            raise InputError(f"{where} is missing {key}")
    _integer(case, "M", where)
    seed = _text(case, "seed", where)
    if seed not in {"target", "mixed"}:
        raise InputError(f"{where}.seed is invalid")
    _text(case, "coupling", where)
    micro = _integer(case, "micro_per_period", where)
    if micro not in {2048, 4096}:
        raise InputError(f"{where}.micro_per_period is invalid")
    for key in ("time_over_T", "high_norm_ratio", "high_absolute_norm"):
        value = case.get(key)
        if value is None:
            continue
        values = _items(value, f"{where}.{key}")
        for index, item in enumerate(values):
            _number(item, f"{where}.{key}[{index}]", minimum=0.0)
    return case


def load_t0_data(gallery: Path) -> tuple[dict[str, Any], Path]:
    data_path = _t0_data_path(gallery)
    if not data_path.is_file():
        raise InputError(f"preserved T0 data does not exist: {data_path}")
    raw = load_json(data_path, "preserved T0 data")
    data = _object(raw, "preserved T0 data")
    if not isinstance(data.get("cases"), list):
        raise InputError("preserved T0 data.cases must be an array")
    for index, case in enumerate(data["cases"]):
        _validate_t0_case(case, f"preserved T0 data.cases[{index}]")
    return data, data_path.resolve()


def load_module(name: str, path: Path) -> Any:
    if not path.is_file():
        raise InputError(f"required source module does not exist: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def style_text(path: Path) -> bytes:
    match = STYLE_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise InputError(f"no HTML style block found in {path}")
    return match.group(1).encode("utf-8")


def _resolve_reference_roots(
    data: dict[str, Any],
    data_path: Path,
) -> tuple[Path, Path, Path]:
    gallery = Path(data["t0_gallery"]).resolve()
    evidence = resolve_local_path(
        data["provenance"]["evidence_root"],
        data_path.parent,
    )
    baseline = resolve_local_path(
        data["provenance"]["baseline_root"],
        data_path.parent,
    )
    for label, path in (
        ("T0 gallery", gallery),
        ("evidence root", evidence),
        ("baseline root", baseline),
    ):
        if not path.exists():
            raise InputError(f"{label} does not exist: {path}")
    return gallery, evidence, baseline


def _chinese_report_candidates(
    gallery: Path,
    evidence: Path,
    baseline: Path,
) -> list[Path]:
    roots = {
        gallery.parent.resolve(),
        evidence.resolve(),
        baseline.resolve(),
    }
    patterns = (
        "*chinese*.html",
        "*chinese*/*.html",
        "*chinese*/*/*.html",
        "zh*.html",
        "zh*/*.html",
    )
    found: set[Path] = set()
    for root in roots:
        for pattern in patterns:
            found.update(
                path.resolve()
                for path in root.glob(pattern)
                if path.is_file()
            )
    return sorted(found)


def external_references(
    gallery: Path,
    evidence: Path,
    baseline: Path,
) -> list[tuple[str, Path]]:
    references: list[tuple[str, Path]] = [
        ("Preserved T0 gallery and exact T0 data", gallery),
        ("Original evidence root; preserved report paths remain read-only", evidence),
        ("Original baseline root", baseline),
    ]
    chinese = _chinese_report_candidates(gallery, evidence, baseline)
    if chinese:
        references.append(("Preserved Chinese report reference", chinese[0]))
    return references


def manifest_references(
    references: list[tuple[str, Path]],
    output: Path,
) -> list[dict[str, str]]:
    grouped: dict[Path, list[str]] = {}
    for label, path in references:
        grouped.setdefault(path.resolve(), []).append(label)
    return [
        {
            "label": "; ".join(labels),
            "path": os.path.relpath(path, output.resolve()),
        }
        for path, labels in grouped.items()
    ]


def case_H_over_T(case: dict[str, Any]) -> float:
    return float(case.get("H_over_T", case["M"] / case["micro_per_period"]))


def accepted_indices(case: dict[str, Any], times: list[float]) -> list[int]:
    limit = case.get("rejected_time_over_T")
    if case.get("completed", False) or limit is None:
        return list(range(len(times)))
    return [index for index, value in enumerate(times) if value < limit]


def case_gain_history(case: dict[str, Any]) -> tuple[list[float], list[float]]:
    ratios = case.get("high_norm_ratio")
    if case.get("seed") != "mixed" or not ratios:
        return [], []
    times = [float(value) for value in case["time_over_T"]]
    indices = accepted_indices(case, times)
    return (
        [times[index] for index in indices],
        [float(ratios[index]) for index in indices],
    )


def accepted_horizon(case: dict[str, Any], time_key: str = "time_over_T") -> float:
    times = [float(value) for value in case.get(time_key, [])]
    indices = accepted_indices(case, times)
    return max((times[index] for index in indices), default=0.0)


def is_primary_rtol(case: dict[str, Any], protocol: dict[str, Any]) -> bool:
    return case.get("rtol") == protocol["rtol"]


def primary_cases(
    data: dict[str, Any],
    coupling: str = T1_PRIMARY_COUPLING,
) -> list[dict[str, Any]]:
    return [
        case
        for case in data["cases"]
        if case["coupling"] == coupling
        and is_primary_rtol(case, data["protocol"])
    ]


def select_case(
    cases: list[dict[str, Any]],
    M: int,
    seed: str,
    micro: int,
    *,
    coupling: str,
) -> dict[str, Any] | None:
    matches = [
        case
        for case in cases
        if case["M"] == M
        and case["seed"] == seed
        and case["micro_per_period"] == micro
        and case["coupling"] == coupling
    ]
    return sorted(
        matches,
        key=lambda case: (not case.get("completed", False), case.get("id", "")),
    )[0] if matches else None


def _case_color(M: int) -> str:
    return {4: TEAL, 16: BLUE, 32: ORANGE}.get(M, GRAY)


def _positive_xy(
    x_values: list[float] | tuple[float, ...],
    y_values: list[float] | tuple[float, ...],
) -> tuple[list[float], list[float]]:
    x_out: list[float] = []
    y_out: list[float] = []
    for x_value, y_value in zip(x_values, y_values, strict=True):
        x_value, y_value = float(x_value), float(y_value)
        if x_value > 0.0 and y_value > 0.0:
            x_out.append(x_value)
            y_out.append(y_value)
    return x_out, y_out


def render_figures(
    data: dict[str, Any],
    t0_data: dict[str, Any],
    output: Path,
    t0_module: Any,
) -> None:
    protocol = data["protocol"]
    plt = t0_module.configure_matplotlib()
    style_axis = t0_module.style_axis
    panel_label = t0_module.panel_label
    save_figure = t0_module.save_figure

    linear_rows = sorted(data["linear"], key=lambda row: row["M"])
    chosen = [row for row in linear_rows if row["M"] in {4, 8, 16, 32}]
    fig, axis = plt.subplots(figsize=(9.4, 5.0), layout="constrained")
    M_values = [row["M"] for row in chosen]
    series = (
        ("positive_branch_modulus", "REF per-step eigenbranch modulus", BLUE, "o", "-"),
        ("eigenmode_power_gain", "REF modulus raised to N", TEAL, "s", "--"),
        (
            "actual_seed_final_gain",
            "REF exact-fast T0 seeded G^N v",
            DARK,
            "^",
            "-",
        ),
        ("operator_power_norm", "REF operator power norm", GRAY, "D", ":"),
        ("production_gain", "NL observed production gain", ORANGE, "X", "--"),
    )
    for key, label, color, marker, linestyle in series:
        x_values, y_values = _positive_xy(
            M_values,
            [row[key] for row in chosen],
        )
        axis.plot(
            x_values,
            y_values,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.2,
            label=label,
        )
    axis.axhline(1.0, color=GRAY, linestyle=":", linewidth=1.0)
    axis.axhline(protocol["growth_limit"], color=ORANGE, linestyle="--", linewidth=1.0)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(M_values, labels=[str(value) for value in M_values])
    axis.minorticks_off()
    axis.set_xlabel("Microsteps per macro interval M")
    axis.set_ylabel("Map modulus or seeded-vector gain")
    axis.set_title(
        "Exact-fast T0 seed map versus production action; finite 2T transverse fixture"
    )
    axis.legend(loc="best", ncol=2)
    axis.text(
        0.02,
        0.03,
        "A finite two-period pass is not asymptotic stability.",
        transform=axis.transAxes,
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": PALE_GRAY, "pad": 2},
    )
    style_axis(axis)
    panel_label(axis, "a")
    save_figure(fig, output, FIGURE_NAMES[0])
    plt.close(fig)

    t0_cases = t0_data["cases"]
    t0_micro = int(t0_data.get("protocol", {}).get("micro_per_period", 2048))
    primary = primary_cases(data)
    matrix_M = sorted(
        {
            case["M"]
            for case in primary
            if 4 <= case["M"] <= 256 and case["micro_per_period"] == 2048
        }
        | {
            case["M"]
            for case in t0_cases
            if 4 <= case["M"] <= 256
            and case["micro_per_period"] == t0_micro
            and case["coupling"] == T0_PRIMARY_COUPLING
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), layout="constrained")
    for title, seed, metric, threshold_key in (
        ("Target maximum relative error", "target", "max_target_error", "target_error_limit"),
        ("High-mode maximum gain", "mixed", "max_high_gain", "growth_limit"),
    ):
        axis = axes[0] if seed == "target" else axes[1]
        for label, cases, micro, coupling, color, linestyle in (
            (
                "T0 preserved matrix",
                t0_cases,
                t0_micro,
                T0_PRIMARY_COUPLING,
                GRAY,
                "--",
            ),
            (
                "T1 waveform primary rtol",
                primary,
                2048,
                T1_PRIMARY_COUPLING,
                BLUE,
                "-",
            ),
        ):
            completed_x: list[float] = []
            completed_y: list[float] = []
            for M in matrix_M:
                case = select_case(
                    cases,
                    M,
                    seed,
                    micro,
                    coupling=coupling,
                )
                if case is None:
                    continue
                value = case.get(metric)
                if value is None:
                    continue
                x_value, y_value = _positive_xy(
                    [case_H_over_T(case)],
                    [value],
                )
                if not x_value:
                    continue
                if case.get("completed", False):
                    completed_x.extend(x_value)
                    completed_y.extend(y_value)
                else:
                    horizon = accepted_horizon(case)
                    axis.plot(
                        x_value,
                        y_value,
                        color=ORANGE,
                        marker="X",
                        markersize=7,
                        linestyle="none",
                    )
                    axis.annotate(
                        f"M={M}: {horizon:.3g}T prefix",
                        (x_value[0], y_value[0]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=6.2,
                        color="#9A3412",
                    )
            if completed_x:
                axis.plot(
                    completed_x,
                    completed_y,
                    color=color,
                    linestyle=linestyle,
                    marker="o" if label.startswith("T1") else "s",
                    markersize=4,
                    label=label,
                )
        axis.axhline(protocol[threshold_key], color=ORANGE, linestyle="--", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("Macro interval H/T")
        axis.set_ylabel("Relative error or gain" if seed == "target" else "Gain ratio")
        axis.set_title(title)
        axis.legend(loc="best")
        style_axis(axis)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.suptitle(
        "T0 predictor and T1 waveform over M=4..256; rejected crosses are prefixes",
        fontsize=10.5,
    )
    save_figure(fig, output, FIGURE_NAMES[1])
    plt.close(fig)

    shown_M = (4, 16, 32)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.8, 6.8),
        sharex="col",
        layout="constrained",
    )
    for column, M in enumerate(shown_M):
        top = axes[0, column]
        bottom = axes[1, column]
        for version, cases, micro, coupling, linestyle in (
            ("T0", t0_cases, t0_micro, T0_PRIMARY_COUPLING, "--"),
            ("T1", data["cases"], 2048, T1_PRIMARY_COUPLING, "-"),
        ):
            case = select_case(
                cases,
                M,
                "mixed",
                micro,
                coupling=coupling,
            )
            if case is None:
                continue
            times, gains = case_gain_history(case)
            if times:
                top.plot(
                    times,
                    gains,
                    color=TEAL if version == "T1" else GRAY,
                    linestyle=linestyle,
                    linewidth=1.25,
                    label=version,
                )
                zero_x, zero_y = _positive_xy(
                    times,
                    [abs(gain - 1.0) for gain in gains],
                )
                bottom.semilogy(
                    zero_x,
                    zero_y,
                    color=TEAL if version == "T1" else GRAY,
                    linestyle=linestyle,
                    linewidth=1.2,
                    label=version,
                )
                if not case.get("completed", False):
                    top.plot(
                        times[-1],
                        gains[-1],
                        color=ORANGE,
                        marker="X",
                        linestyle="none",
                    )
                    top.annotate(
                        f"prefix {times[-1]:.3g}T",
                        (times[-1], gains[-1]),
                        textcoords="offset points",
                        xytext=(3, 5),
                        fontsize=6.2,
                    )
            else:
                top.text(
                    0.03,
                    0.92,
                    f"{version} gain unavailable",
                    transform=top.transAxes,
                    fontsize=7,
                    va="top",
                )
        top.axhline(1.01, color=ORANGE, linestyle="--", linewidth=1)
        top.set_title(f"M={M}")
        top.set_ylabel("Seeded high-mode gain")
        top.legend(loc="best")
        style_axis(top)
        panel_label(top, "abc"[column])
        bottom.axhline(0.01, color=ORANGE, linestyle="--", linewidth=1)
        bottom.set_xlabel("Accepted time t/T")
        bottom.set_ylabel("abs(gain - 1)")
        bottom.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
        bottom.legend(loc="best")
        style_axis(bottom)
        panel_label(bottom, "def"[column])
    fig.suptitle(
        (
            "High-mode gain histories: T0 predictor dashed, T1 waveform solid; "
            f"actual h_s/T_s=1/2048, h_s={protocol['h_s']:.7g} s, "
            f"T_s={protocol['T_s']:.7g} s"
        ),
        fontsize=10.5,
    )
    save_figure(fig, output, FIGURE_NAMES[2])
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.9), layout="constrained")
    for axis, M, letter in zip(axes, shown_M, "abc", strict=True):
        case = select_case(
            primary,
            M,
            "target",
            2048,
            coupling=T1_PRIMARY_COUPLING,
        )
        if case is None:
            axis.text(0.5, 0.5, "Primary target case unavailable", ha="center", va="center")
            axis.axis("off")
            continue
        groups: dict[int, list[dict[str, Any]]] = {}
        for row in case["iterations"]:
            groups.setdefault(row["macro_index"], []).append(row)
        if not groups:
            axis.text(
                0.5,
                0.5,
                "Iteration diagnostics unavailable",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        else:
            first_macro = min(groups)
            worst_macro = max(
                groups,
                key=lambda macro: max(row["endpoint_max"] for row in groups[macro]),
            )
            selected = (first_macro,) if first_macro == worst_macro else (
                first_macro,
                worst_macro,
            )
            for macro in selected:
                rows = sorted(groups[macro], key=lambda row: row["iteration"])
                x_values = [row["iteration"] for row in rows]
                for key, marker, color, label in (
                    ("path_max", "o", BLUE, "path maximum"),
                    ("endpoint_max", "s", TEAL, "endpoint maximum"),
                ):
                    y_values = [row[key] for row in rows]
                    visible = [
                        index
                        for index, value in enumerate(y_values)
                        if float(value) > 0.0
                    ]
                    axis.semilogy(
                        [x_values[index] for index in visible],
                        [y_values[index] for index in visible],
                        color=color,
                        marker=marker,
                        linestyle="-" if macro == first_macro else "--",
                        linewidth=1.0,
                        label=label if macro == first_macro else None,
                    )
            axis.text(
                0.02,
                0.04,
                f"first macro={first_macro}; worst macro={worst_macro}",
                transform=axis.transAxes,
                fontsize=6.8,
                bbox={"facecolor": "white", "edgecolor": PALE_GRAY, "pad": 2},
            )
        rtol = case.get("rtol")
        if rtol is not None:
            axis.axhline(float(rtol), color=ORANGE, linestyle="--", label="rtol")
        else:
            axis.text(
                0.98,
                0.96,
                "rtol unavailable",
                ha="right",
                va="top",
                transform=axis.transAxes,
                fontsize=7,
            )
        axis.set_xlabel("Nonlinear sweep iteration")
        axis.set_ylabel("Residual maximum")
        axis.set_title(f"T1 waveform M={M}; first and worst macro")
        axis.legend(loc="best")
        style_axis(axis)
        panel_label(axis, letter)
    fig.suptitle(
        "Path and endpoint residuals by sweep; same-time RK4 midpoint targets are distinct",
        fontsize=10.5,
    )
    save_figure(fig, output, FIGURE_NAMES[3])
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.0), layout="constrained")
    for M in shown_M:
        for seed, linestyle in (("target", "-"), ("mixed", "--")):
            case = select_case(
                primary,
                M,
                seed,
                2048,
                coupling=T1_PRIMARY_COUPLING,
            )
            if case is None or not case["energy_time_over_T"]:
                continue
            times = [float(value) for value in case["energy_time_over_T"]]
            indices = accepted_indices(case, times)
            for axis, key in zip(
                axes[:2],
                ("total_energy_residual_J", "kinetic_energy_residual_J"),
                strict=True,
            ):
                xs = [times[index] for index in indices]
                ys = [abs(float(case[key][index])) for index in indices]
                zero_x, zero_y = _positive_xy(xs, ys)
                axis.semilogy(
                    zero_x,
                    zero_y,
                    color=_case_color(M),
                    linestyle=linestyle,
                    linewidth=0.9,
                    label=f"M={M}, {seed}",
                )
        case = select_case(
            primary,
            M,
            "mixed",
            2048,
            coupling=T1_PRIMARY_COUPLING,
        )
        if case is None or not case["iterations"]:
            continue
        macro_energy: dict[int, tuple[float, float]] = {}
        for row in case["iterations"]:
            total = row["total_energy_residual_J"]
            kinetic = row["kinetic_energy_residual_J"]
            if total is None or kinetic is None:
                continue
            old_total, old_kinetic = macro_energy.get(
                row["macro_index"],
                (0.0, 0.0),
            )
            macro_energy[row["macro_index"]] = (
                max(old_total, abs(float(total))),
                max(old_kinetic, abs(float(kinetic))),
            )
        if macro_energy:
            macros = sorted(macro_energy)
            axes[2].semilogy(
                macros,
                [macro_energy[macro][0] for macro in macros],
                color=_case_color(M),
                marker="o",
                linestyle="-",
                label=f"M={M} total",
            )
            axes[2].semilogy(
                macros,
                [macro_energy[macro][1] for macro in macros],
                color=_case_color(M),
                marker="^",
                linestyle="--",
                label=f"M={M} kinetic",
            )
    for axis, title in zip(
        axes,
        (
            "T1 waveform accepted final total-energy histories",
            "T1 waveform accepted final kinetic-energy histories",
            "T1 waveform observed intermediate-sweep maxima",
        ),
        strict=True,
    ):
        axis.axhline(
            protocol["energy_budget_J"],
            color=ORANGE,
            linestyle="--",
            linewidth=1,
            label="unchanged budget",
        )
        axis.set_yscale("log")
        axis.set_ylabel("abs(raw residual) [J]")
        axis.set_title(title)
        axis.legend(loc="best", fontsize=6.8, ncol=2)
        style_axis(axis)
    axes[0].set_xlabel("Accepted time t/T")
    axes[1].set_xlabel("Accepted time t/T")
    axes[2].set_xlabel("Macro index")
    for axis, letter in zip(axes, "abc", strict=True):
        panel_label(axis, letter)
    fig.suptitle(
        "Raw energy residuals; intermediate trials are observed, only final windows gate",
        fontsize=10.5,
    )
    save_figure(fig, output, FIGURE_NAMES[4])
    plt.close(fig)

    controls = [
        case
        for case in data["cases"]
        if any(
            math.isclose(case_H_over_T(case), target, rel_tol=0.0, abs_tol=1.0e-15)
            for target in (1.0 / 16.0, 1.0 / 8.0)
        )
    ]
    rtol_keys = sorted(
        {case["rtol"] for case in controls},
        key=lambda value: (value is not None, value if value is not None else -1.0),
    )
    x_positions = {key: index for index, key in enumerate(rtol_keys)}
    fig = plt.figure(figsize=(12.6, 6.2), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(4.0, 1.0))
    axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]))
    status_axis = fig.add_subplot(grid[1, :])
    for axis, seed, metric, threshold in (
        (axes[0], "target", "max_target_error", protocol["target_error_limit"]),
        (axes[1], "mixed", "max_high_gain", protocol["growth_limit"]),
    ):
        grouped: dict[tuple[float, int, str], list[tuple[float, float]]] = {}
        incomplete: list[str] = []
        for case in (item for item in controls if item["seed"] == seed):
            key = x_positions[case["rtol"]]
            value = case[metric]
            h_value = case_H_over_T(case)
            color = TEAL if math.isclose(h_value, 1 / 16, abs_tol=1e-15) else BLUE
            marker = "o" if case["micro_per_period"] == 2048 else "s"
            if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
                if not case["completed"]:
                    incomplete.append(
                        f"M={case['M']}, {case['coupling']}, "
                        f"{'h*' if case['micro_per_period'] == 2048 else 'h*/2'}, "
                        f"rtol={case['rtol']}: {case['status']} at "
                        f"{accepted_horizon(case):.3g}T; metric unavailable"
                    )
                continue
            if case["completed"]:
                grouped.setdefault(
                    (h_value, case["micro_per_period"], case["coupling"]),
                    [],
                ).append((key, float(value)))
            else:
                axis.plot(
                    key,
                    value,
                    color=ORANGE,
                    marker="X",
                    markersize=7,
                    linestyle="none",
                )
                incomplete.append(
                    f"M={case['M']}, {case['coupling']}, "
                    f"{'h*' if case['micro_per_period'] == 2048 else 'h*/2'}, "
                    f"rtol={case['rtol']}: {case['status']} at "
                    f"{accepted_horizon(case):.3g}T; plotted metric is prefix-only"
                )
        for (h_value, micro, coupling), points in grouped.items():
            points.sort()
            color = TEAL if math.isclose(h_value, 1 / 16, abs_tol=1e-15) else BLUE
            linestyle = {
                "predictor": "-",
                "waveform": ":",
                "synchronous": "--",
            }[coupling]
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=color,
                marker="o" if micro == 2048 else "s",
                linestyle=linestyle,
                label=(
                    f"H/T={h_value:.5g}, "
                    f"{'h*' if micro == 2048 else 'h*/2'}, {coupling}"
                ),
            )
        axis.axhline(threshold, color=ORANGE, linestyle="--", linewidth=1)
        axis.set_yscale("log")
        axis.set_xticks(
            list(x_positions.values()),
            labels=[
                "primary" if key == protocol["rtol"] else (
                    "null" if key is None else f"{key:.2g}"
                )
                for key in rtol_keys
            ],
        )
        axis.set_xlabel("Relative tolerance rtol")
        axis.set_ylabel("Maximum relative error" if seed == "target" else "Maximum gain")
        axis.set_title(
            "Target error controls" if seed == "target" else "High-seed gain controls"
        )
        axis.legend(loc="best", fontsize=6.7)
        style_axis(axis)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    status_axis.axis("off")
    incomplete_text = (
        "\n".join(incomplete)
        if incomplete
        else "No fixed-H incomplete control is present in the supplied data."
    )
    status_axis.text(
        0.0,
        1.0,
        textwrap.fill(incomplete_text, 155),
        transform=status_axis.transAxes,
        va="top",
        fontsize=6.8,
        color="#9A3412",
    )
    panel_label(status_axis, "c")
    fig.suptitle(
        "Fixed-H h* and h*/2 controls at H=T/16 and H=T/8; incomplete points stay explicit",
        fontsize=10.5,
    )
    save_figure(fig, output, FIGURE_NAMES[5])
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.2), layout="constrained")
    work_cases = [
        case
        for case in primary
        if case["micro_per_period"] == 2048
    ]
    legend_seen: set[tuple[str, str]] = set()
    for case in sorted(work_cases, key=lambda item: (item["M"], item["seed"])):
        color = TEAL if case["seed"] == "target" else ORANGE
        marker = {
            "predictor": "o",
            "waveform": "s",
            "synchronous": "^",
        }[case["coupling"]]
        key = (case["coupling"], case["seed"])
        label = f"{case['coupling']}, {case['seed']}"
        first_for_label = key not in legend_seen
        legend_seen.add(key)
        if case["model_calls"] is not None:
            axes[0].plot(
                case["M"],
                case["model_calls"],
                color=color,
                marker=marker,
                linestyle="none",
                label=label if first_for_label else None,
            )
        if case["estimated_peak_bytes"] is not None:
            axes[1].plot(
                case["M"],
                case["estimated_peak_bytes"],
                color=color,
                marker=marker,
                linestyle="none",
                label=label if first_for_label else None,
            )
    for axis, title, ylabel in (
        (axes[0], "Supplied model calls", "Model calls"),
        (axes[1], "Conservative storage estimate", "Estimated peak bytes"),
    ):
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("Microsteps per macro interval M")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend(loc="best", fontsize=7)
        style_axis(axis)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")

    gpu = data["gpu"]
    parity = gpu["parity_rows"]
    axes[2].axis("off")
    if parity:
        table = axes[2].table(
            cellText=[
                [
                    str(row["M"]),
                    str(row["accepted_macros"]),
                    f"{row['relative_state_error']:.3g}",
                    f"{row['relative_mode_error']:.3g}",
                    row["device"],
                ]
                for row in parity
            ],
            colLabels=["M", "Macros", "State rel.", "Mode rel.", "Device"],
            loc="upper center",
            cellLoc="center",
            colLoc="center",
            bbox=(0.0, 0.22, 1.0, 0.68),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.8)
    axes[2].text(
        0.0,
        0.02 if parity else 0.1,
        textwrap.fill(
            "GPU status: "
            + gpu["status"]
            + (
                ""
                if parity
                else ". No parity rows were supplied; bounded GPU parity is unavailable."
            )
            + " "
            + " ".join(gpu["notes"]),
            65,
        ),
        transform=axes[2].transAxes,
        va="bottom",
        fontsize=6.8,
    )
    axes[2].set_title("Bounded GPU parity")
    panel_label(axes[2], "c")
    axes[0].text(
        0.02,
        0.04,
        "CPU elapsed time is diagnostic context, not a final T1 performance verdict.",
        transform=axes[0].transAxes,
        fontsize=6.5,
        bbox={"facecolor": "white", "edgecolor": PALE_GRAY, "pad": 2},
    )
    fig.suptitle(
        "Work and storage observations; no speedup, measured-peak, or T3 claim",
        fontsize=10.5,
    )
    save_figure(fig, output, FIGURE_NAMES[6])
    plt.close(fig)


def _availability_notes(data: dict[str, Any]) -> list[str]:
    cases = data["cases"]
    target_ratio_unavailable = sum(
        case["seed"] == "target"
        and (case["high_norm_ratio"] is None or not case["high_norm_ratio"])
        for case in cases
    )
    mixed_ratio_unavailable = sum(
        case["seed"] == "mixed"
        and (case["high_norm_ratio"] is None or not case["high_norm_ratio"])
        for case in cases
    )
    energy_unavailable = sum(
        not case["energy_time_over_T"]
        or not case["total_energy_residual_J"]
        or not case["kinetic_energy_residual_J"]
        for case in cases
    )
    iteration_energy_unavailable = sum(
        row["total_energy_residual_J"] is None
        or row["kinetic_energy_residual_J"] is None
        for case in cases
        for row in case["iterations"]
    )
    rejected = [case["id"] for case in cases if not case["completed"]]
    return [
        (
            f"{target_ratio_unavailable} target-only case(s) have no relative "
            "high-mode gain; null/empty remains unavailable."
        ),
        (
            f"{mixed_ratio_unavailable} mixed-seed case(s) have unavailable gain "
            "histories; no denominator or zero is invented."
        ),
        (
            f"{energy_unavailable} case(s) have empty energy histories and "
            f"{iteration_energy_unavailable} intermediate residual entr(ies) are null."
        ),
        (
            "Rejected/incomplete case IDs: " + ", ".join(rejected)
            if rejected
            else "No case is marked rejected or incomplete in the supplied data."
        ),
        (
            f"GPU parity rows supplied: {len(data['gpu']['parity_rows'])}; "
            f"status: {data['gpu']['status']}."
        ),
    ]


def build_manifest(
    data: dict[str, Any],
    output: Path,
    references: list[tuple[str, Path]],
    t0_module: Any,
) -> dict[str, Any]:
    protocol = data["protocol"]
    cases = data["cases"]
    completed = sum(case["completed"] for case in cases)
    rejected = len(cases) - completed

    fig1 = t0_module.figure_record(
        1,
        FIGURE_NAMES[0],
        "Exact-fast T0 seed maps versus production action",
        (
            "REF marks the independently supplied exact-fast T0 seeded-vector "
            "gain; REF also marks the per-step eigenbranch modulus, its Nth power, "
            "and operator power norm. NL marks the observed production gain. These are distinct "
            "objects in this periodic transverse model. A finite two-period pass "
            "is not an asymptotic stability result."
        ),
        "mixed_comparison",
        components=[
            "REF exact-fast T0 seeded map",
            "REF independent macro-map modulus and power",
            "REF operator power norm",
            "NL observed production gain",
        ],
        panels=[
            ("a", "Per-step modulus and finite-window seeded gains for M=4..32."),
        ],
        metadata={
            "device": "Supplied CPU FP64 map data",
            "status": "Finite two-period transverse fixture only",
        },
    )
    fig2 = t0_module.figure_record(
        2,
        FIGURE_NAMES[1],
        "T0 versus primary T1 over the original M=4..256 matrix",
        (
            "NL denotes supplied nonlinear T0 and T1 histories. T1 points use "
            "only waveform cases with rtol equal to the fixed primary protocol "
            "value. "
            "Tolerance variants are reserved for Figure 6. A rejected cross is "
            "an accepted-prefix observation with its actual horizon; it is not a "
            "full-window pass."
        ),
        "mixed_comparison",
        components=[
            "NL T0 predictor nonlinear matrix",
            "NL T1 waveform primary nonlinear matrix",
        ],
        panels=[
            ("a", "Completed maximum target error and rejected-prefix crosses."),
            ("b", "Completed maximum high-mode gain and rejected-prefix crosses."),
        ],
        metadata={
            "device": "Supplied CPU FP64 records",
            "status": f"{completed} completed; {rejected} rejected/incomplete",
        },
    )
    fig3 = t0_module.figure_record(
        3,
        FIGURE_NAMES[2],
        "High-mode gain histories for M=4, 16, and 32",
        (
            "NL means supplied nonlinear time evolution. T0 predictor is dashed "
            "and the primary T1 waveform is solid. The upper row retains the full "
            "gain range, while the lower deviation row keeps the fixed 1.01 "
            "criterion visible beside M=32 growth near 1.8. Actual supplied h/T "
            "and T values are stated on the figure; rejected histories stop at "
            "their accepted prefix."
        ),
        "nonlinear_evolution",
        panels=[
            ("a", "T0/T1 high-mode gain history for M=4."),
            ("b", "T0/T1 high-mode gain history for M=16."),
            ("c", "T0/T1 high-mode gain history for M=32."),
            ("d", "Absolute gain deviation and the 1% criterion for M=4."),
            ("e", "Absolute gain deviation and the 1% criterion for M=16."),
            ("f", "Absolute gain deviation and the 1% criterion for M=32."),
        ],
        metadata={
            "device": "Supplied CPU FP64 records",
            "status": "Completed histories and accepted prefixes only",
        },
    )
    fig4 = t0_module.figure_record(
        4,
        FIGURE_NAMES[3],
        "Iteration path and endpoint residuals",
        (
            "VER identifies iterative solver diagnostics from the primary T1 "
            "waveform cases. For each M the first and worst macro are labeled and "
            "the supplied rtol is drawn. Path and endpoint maxima remain separate. "
            "Same-time RK4 midpoint targets are distinct from the converged "
            "discrete target, which is synchronous fine RK4 under the same "
            "canonicalization."
        ),
        "verification",
        panels=[
            ("a", "M=4 first/worst macro path and endpoint residuals."),
            ("b", "M=16 first/worst macro path and endpoint residuals."),
            ("c", "M=32 first/worst macro path and endpoint residuals."),
        ],
        metadata={
            "device": "Supplied CPU FP64 diagnostics",
            "status": "Iteration evidence; unavailable entries stay unavailable",
        },
    )
    fig5 = t0_module.figure_record(
        5,
        FIGURE_NAMES[4],
        "Raw energy residuals and intermediate-sweep budgets",
        (
            "NL marks accepted final nonlinear windows from primary T1 waveform "
            "cases; VER-like intermediate maxima are placed in a separate panel. "
            "All values remain raw joules against the unchanged supplied energy "
            "budget. Intermediate trial residuals are observations, not accepted "
            "gates; only a completed converged final window can pass the final "
            "energy check."
        ),
        "mixed_comparison",
        components=[
            "NL final total and ion-kinetic energy histories",
            "observed intermediate-sweep residual maxima",
            "UNCHANGED supplied energy budget",
        ],
        panels=[
            ("a", "Accepted final total-energy residual magnitude."),
            ("b", "Accepted final ion-kinetic residual magnitude."),
            ("c", "Observed intermediate-sweep maxima by macro."),
        ],
        metadata={
            "device": "Supplied CPU FP64 records",
            "status": "Final-window gate separate from observed intermediate trials",
        },
    )
    fig6 = t0_module.figure_record(
        6,
        FIGURE_NAMES[5],
        "Fixed-H half-step and tolerance controls",
        (
            "VER denotes bounded numerical controls. The panels compare target "
            "error and mixed-seed high gain at H=T/16 and H=T/8 for h* and h*/2 "
            "and for supplied tolerance variants. Incomplete cases are named "
            "below the panels; any prefix-only numeric value is an open cross and "
            "never a completed point."
        ),
        "verification",
        panels=[
            ("a", "Target maximum error across supplied tolerance choices."),
            ("b", "Mixed-seed maximum gain across supplied tolerance choices."),
            ("c", "Incomplete control annotations with accepted horizons."),
        ],
        metadata={
            "device": "Supplied CPU FP64 records",
            "status": f"{rejected} rejected/incomplete case(s) retained",
        },
    )
    fig7 = t0_module.figure_record(
        7,
        FIGURE_NAMES[6],
        "Work, conservative storage, and bounded GPU parity",
        (
            "VER denotes supplied work/parity observations. Model calls are "
            "copied from the data, and storage is the conservative estimate, not "
            "a measured peak. CPU wall time is not a final A1 performance verdict. "
            "Absent GPU rows are explicitly unavailable, and cross-slab "
            "time-parallel work is deferred T3."
        ),
        "mixed_comparison",
        components=[
            "supplied model-call counts",
            "conservative peak-byte estimates",
            "bounded CPU/GPU parity rows or unavailable notes",
        ],
        panels=[
            ("a", "Model calls versus macro size."),
            ("b", "Conservative estimated peak bytes."),
            ("c", "Bounded parity table or explicit unavailable note."),
        ],
        metadata={
            "device": "CPU records; GPU only when supplied in parity_rows",
            "status": data["gpu"]["status"],
        },
    )

    provenance_notes = data["provenance"]["notes"]
    pages = [
        {
            "id": "scope-protocol",
            "title": "Scope and fixed protocol",
            "introduction": (
                "This English gallery presents a controlled periodic nine-point "
                "temporal fixture. It makes no global-wall, CAW, large-production, "
                "or asymptotic-stability claim. The finite 2T window is a bounded "
                "algorithm observation, not a stability proof."
            ),
            "notes": [
                (
                    f"Fixed protocol: cycles={protocol['cycles']}, "
                    f"rtol={protocol['rtol']:.3g}, "
                    f"max_iterations={protocol['max_iterations']}, target error "
                    f"{protocol['target_error_limit']:.3g}, phase "
                    f"{protocol['phase_limit_rad']:.3g} rad, amplitude "
                    f"{protocol['amplitude_limit']:.3g}, growth "
                    f"{protocol['growth_limit']:.3g}, and energy budget "
                    f"{protocol['energy_budget_J']:.3g} J."
                ),
                (
                    "REF labels the independent Hall-map eigen, singular, and "
                    "seed quantities. NL labels actual production trajectory gains "
                    "and nonlinear histories. LIN is reserved for a production-RHS "
                    "Jacobian and is not used in this gallery. VER labels numerical "
                    "controls. No blanket acceptance claim follows from a link or "
                    "finite window."
                ),
                (
                    "No blanket acceptance statement is made: completed and "
                    "rejected records keep their supplied status."
                ),
            ],
        },
        {
            "id": "linear-seed-map",
            "title": "Linear and exact seed maps",
            "introduction": (
                "The first figure distinguishes one-step spectral information, a "
                "powered eigenmode estimate, exact seeded-vector action, operator "
                "norm, and the observed nonlinear production gain."
            ),
            "notes": [
                (
                    "A modulus below one does not prove finite-window nonlinear "
                    "contraction or asymptotic stability."
                ),
                (
                    "The exact-fast T0 seeded and spectral quantities are REF; "
                    "independent map quantities are REF; observed production gains "
                    "are NL. No production-RHS Jacobian LIN result is presented."
                ),
            ],
            "figures": [fig1],
        },
        {
            "id": "t0-t1-matrix",
            "title": "T0 and T1 matrix comparison",
            "introduction": (
                "The preserved T0 matrix provides the baseline for the original "
                "M=4..256 context. Primary T1 points use waveform coupling, the "
                "fixed rtol, and the h* cadence; tolerance variants remain separate."
            ),
            "notes": [
                (
                    "Rejected crosses annotate accepted prefix horizons. They do "
                    "not assert a full-window criterion pass."
                ),
                (
                    "The T0 overlay is read from data.json associated with the "
                    "preserved t0_gallery path and is never rewritten."
                ),
                (
                    "No CAW interpretation is assigned to the transverse Hall "
                    "temporal fixture."
                ),
            ],
            "figures": [fig2],
        },
        {
            "id": "high-mode-history",
            "title": "High-mode histories",
            "introduction": (
                "Three macro sizes compare supplied T0 predictor and primary T1 "
                "waveform mixed-seed gain histories, with a lower deviation row "
                "around 1.01."
            ),
            "notes": [
                (
                    "Target-only cases have an undefined high-seed-relative gain "
                    "and keep null/empty gain values rather than manufactured zeros."
                ),
                (
                    "Large M=32 gain is kept in the upper row; the lower row makes "
                    "the unchanged 1% criterion visible."
                ),
            ],
            "figures": [fig3],
        },
        {
            "id": "iteration-residuals",
            "title": "Iteration path and endpoint residuals",
            "introduction": (
                "Primary T1 waveform sweep diagnostics show how path and endpoint "
                "convergence maxima behave inside the first and worst supplied macros."
            ),
            "notes": [
                (
                    "The converged discrete target is synchronous fine RK4 under "
                    "the same canonicalization; same-time RK4 midpoints are distinct."
                ),
                (
                    "The supplied rtol is a reference, not a replacement for "
                    "endpoint or energy acceptance records."
                ),
            ],
            "figures": [fig4],
        },
        {
            "id": "energy-budget",
            "title": "Raw residual and budget evidence",
            "introduction": (
                "Primary T1 waveform final-window histories and intermediate-sweep "
                "maxima are kept visually separate against the unchanged raw energy "
                "budget."
            ),
            "notes": [
                (
                    "Only a completed converged final window is eligible for the "
                    "final energy gate. Intermediate trials are reported as observed."
                ),
                (
                    "Small background-relative residuals do not imply small "
                    "high-seed-relative accumulated error."
                ),
            ],
            "figures": [fig5],
        },
        {
            "id": "refinement-controls",
            "title": "Half-step and tolerance controls",
            "introduction": (
                "The fixed H=T/16 and H=T/8 comparisons isolate h* versus h*/2 "
                "and retain supplied tolerance variants and incomplete cases."
            ),
            "notes": _availability_notes(data),
            "figures": [fig6],
        },
        {
            "id": "work-gpu-references",
            "title": "Work, storage, GPU parity, and references",
            "introduction": (
                "Model calls and conservative storage estimates are presentation "
                "facts. CPU wall time is not a final performance verdict, and T3 "
                "cross-slab time-parallel scheduling is deferred."
            ),
            "notes": [
                *provenance_notes,
                (
                    f"{len(data['provenance']['source_hashes'])} source-file "
                    "hashes and the compact-input SHA-256 are recorded in "
                    "provenance.json."
                ),
            ],
            "figures": [fig7],
        },
    ]

    return {
        "title": "A1 T1 controlled temporal evidence: periodic nine-point fixture",
        "introduction": (
            "This English gallery reuses the unchanged A0 renderer, page layout, "
            "evidence badges, CSS, and English presentation style. It compares "
            "supplied waveform-coupled T1 records with the preserved T0 predictor "
            "matrix for a periodic "
            "nine-point controlled temporal fixture. The report does not run the "
            "solver or modify a tolerance. It makes no global-wall, CAW, "
            "large-production, speedup, or asymptotic-stability claim."
        ),
        "notation": [
            {"symbol": "T", "meaning": "Supplied target temporal period."},
            {"symbol": "h*", "meaning": "Supplied fine reference microstep."},
            {"symbol": "H", "meaning": "Macro interval H=Mh."},
            {"symbol": "M", "meaning": "Microsteps in one macro interval."},
            {"symbol": "rtol", "meaning": "Relative iterative convergence tolerance."},
            {"symbol": "NL / LIN / REF / VER", "meaning": (
                "Nonlinear evolution; local production-RHS Jacobian (not used "
                "here); independent Hall-map reference; numerical verification."
            )},
        ],
        "archive": {
            "path": "data.json",
            "label": "Byte-preserved compact T1 data; exact hashes are in provenance.json",
        },
        "pages": pages,
        "conclusions": (
            f"The supplied records contain {completed} completed and {rejected} "
            "rejected/incomplete cases. No blanket acceptance statement is made: "
            "the gallery preserves rejected prefixes, null availability, the "
            "finite 2T observational scope, and the separation between T0 baseline "
            "and primary-versus-tolerance T1 comparisons."
        ),
        "known_issues": manifest_references(references, output),
    }


def embed_svgs(rendered: str, output: Path) -> str:
    for name in FIGURE_NAMES:
        payload = base64.b64encode((output / f"{name}.svg").read_bytes()).decode("ascii")
        data_uri = f"data:image/svg+xml;base64,{payload}"
        needle = f'src="{name}.png"'
        if rendered.count(needle) != 1:
            raise RuntimeError(f"expected one rendered image source for {name}")
        rendered = rendered.replace(needle, f'src="{data_uri}"')
    embedded = 'src="data:image/svg+xml;base64,'
    if rendered.count(embedded) != len(FIGURE_NAMES):
        raise RuntimeError("not all figures were embedded as SVG")
    if any(f'src="{name}.png"' in rendered for name in FIGURE_NAMES):
        raise RuntimeError("a raster figure source remains in the HTML")
    return rendered


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=True, indent=2) + "\n")


def output_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def build(
    data_path: Path,
    output: Path,
    renderer_path: Path,
    a0_html_path: Path,
    style_paths: tuple[Path, Path],
) -> dict[str, Any]:
    data_path, output = Path(data_path).resolve(), Path(output).resolve()
    renderer_path, a0_html_path = Path(renderer_path).resolve(), Path(a0_html_path).resolve()
    style_paths = tuple(path.resolve() for path in style_paths)
    if output.exists():
        raise FileExistsError(f"refusing existing output path: {output}")
    if not data_path.is_file():
        raise InputError(f"compact data is not a file: {data_path}")
    if renderer_path == output or data_path == output:
        raise InputError("output cannot replace an input path")

    data_hash = sha256_file(data_path)
    data = _object(load_json(data_path, "compact data"), "data")
    protocol = validate_data(data)
    gallery, evidence, baseline = _resolve_reference_roots(data, data_path)
    t0_data, t0_data_path = load_t0_data(gallery)
    t0_data_hash = sha256_file(t0_data_path)
    gallery_hash = sha256_file(gallery) if gallery.is_file() else None
    references = external_references(gallery, evidence, baseline)

    a1_t0_path = renderer_path.with_name("build_a1_t0_report.py")
    renderer = load_module("canonical_a0_renderer", renderer_path)
    t0_module = load_module("canonical_a1_t0_report", a1_t0_path)
    for function in (
        "configure_matplotlib",
        "style_axis",
        "panel_label",
        "save_figure",
        "figure_record",
        "check_canonical_files",
    ):
        if not callable(getattr(t0_module, function, None)):
            raise InputError(f"A1 T0 module does not expose {function}")
    if not callable(getattr(renderer, "render_report", None)):
        raise InputError("canonical A0 renderer does not expose render_report")
    canonical_hashes = t0_module.check_canonical_files(
        renderer_path,
        a0_html_path,
        style_paths,
    )
    builder_path = Path(__file__).resolve()
    builder_hash = sha256_file(builder_path)

    if sha256_file(data_path) != data_hash:
        raise RuntimeError("compact input changed during validation")
    if sha256_file(t0_data_path) != t0_data_hash:
        raise RuntimeError("preserved T0 data changed during validation")
    if sha256_file(renderer_path) != canonical_hashes["renderer"]:
        raise RuntimeError("canonical renderer changed during validation")
    if sha256_file(builder_path) != builder_hash:
        raise RuntimeError("builder changed during validation")

    output.mkdir(parents=True, exist_ok=False)
    old_mplconfig = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(prefix="a1-t1-matplotlib-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        try:
            render_figures(data, t0_data, output, t0_module)
        finally:
            if old_mplconfig is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = old_mplconfig

    for name in FIGURE_NAMES:
        for suffix in ("png", "svg"):
            path = output / f"{name}.{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"figure output is missing or empty: {path}")
    shutil.copyfile(data_path, output / "data.json")

    manifest = build_manifest(data, output, references, t0_module)
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    provenance_path = output / "provenance.json"
    _write_json(
        provenance_path,
        {
            "schema_version": 1,
            "status": "rendering",
            "compact_input_sha256": data_hash,
        },
    )

    html_path = output / "a1-benchmark.html"
    rendered = renderer.render_report(manifest, manifest_path, html_path)
    rendered = embed_svgs(rendered, output)
    _write_text(html_path, rendered)

    if sha256_file(data_path) != data_hash:
        raise RuntimeError("compact input changed during rendering")
    if sha256_file(t0_data_path) != t0_data_hash:
        raise RuntimeError("preserved T0 data changed during rendering")
    if sha256_file(renderer_path) != canonical_hashes["renderer"]:
        raise RuntimeError("canonical renderer changed during rendering")
    if sha256_file(builder_path) != builder_hash:
        raise RuntimeError("builder changed during rendering")
    if style_text(html_path) != style_text(a0_html_path):
        raise RuntimeError("rendered HTML CSS differs from canonical A0 CSS")

    hashed_outputs = [
        html_path,
        manifest_path,
        output / "data.json",
        *[
            output / f"{name}.{suffix}"
            for name in FIGURE_NAMES
            for suffix in ("png", "svg")
        ],
    ]
    provenance = {
        "schema_version": 1,
        "status": "complete",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_commit": data["provenance"]["source_commit"],
        "source_hashes": data["provenance"]["source_hashes"],
        "evidence_root": str(evidence),
        "baseline_root": str(baseline),
        "notes": data["provenance"]["notes"],
        "compact_input": {
            "path": str(data_path),
            "sha256": data_hash,
            "bytes": data_path.stat().st_size,
        },
        "preserved_t0": {
            "gallery": str(gallery),
            "gallery_sha256": gallery_hash,
            "data": str(t0_data_path),
            "data_sha256": t0_data_hash,
            "bytes": t0_data_path.stat().st_size,
        },
        "external_references": [
            {"label": label, "path": str(path)}
            for label, path in references
        ],
        "canonical_sources": {
            "renderer_path": str(renderer_path),
            "renderer_sha256": canonical_hashes["renderer"],
            "a0_html_path": str(a0_html_path),
            "a0_html_sha256": canonical_hashes["a0_html"],
            "plot_tokamak_cae_path": str(style_paths[0]),
            "plot_tokamak_cae_sha256": canonical_hashes["plot_tokamak_cae"],
            "plot_cae_comparison_path": str(style_paths[1]),
            "plot_cae_comparison_sha256": canonical_hashes["plot_cae_comparison"],
        },
        "builder": {
            "path": str(builder_path),
            "sha256": builder_hash,
        },
        "protocol_snapshot": protocol,
        "render_policy": {
            "renderer_reused_unchanged": True,
            "css_matches_a0_byte_for_byte": True,
            "figures": len(FIGURE_NAMES),
            "png_dpi": 160,
            "svg_fonttype": "none",
            "source_runs_executed": False,
            "runtime_network_used": False,
            "solver_used": False,
            "raw_input_overwritten": False,
            "scientific_thresholds_modified": False,
            "time_parallel_t3_executed": False,
        },
        "limitations": [
            "Controlled periodic nine-point temporal fixture only.",
            "No global-wall, CAW, large-production, speedup, measured-peak, or asymptotic-stability claim.",
            "CPU elapsed time is diagnostic context, not a final A1 performance verdict.",
            "Cross-slab time-parallel work is deferred T3.",
            *data["provenance"]["notes"],
        ],
        "copied_data": output_record(output / "data.json", output),
        "outputs": [output_record(path, output) for path in hashed_outputs],
    }
    _write_json(provenance_path, provenance)
    return {
        "output": str(output),
        "html_bytes": html_path.stat().st_size,
        "figures": len(FIGURE_NAMES),
        "pages": len(manifest["pages"]),
        "completed_cases": sum(case["completed"] for case in data["cases"]),
        "rejected_cases": sum(not case["completed"] for case in data["cases"]),
        "gpu_parity_rows": len(data["gpu"]["parity_rows"]),
        "input_sha256": data_hash,
        "renderer_sha256": canonical_hashes["renderer"],
        "builder_sha256": builder_hash,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--renderer", type=Path)
    parser.add_argument("--a0-html", type=Path, default=A0_HTML_DEFAULT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        print(f"error: refusing existing output path: {output}", file=sys.stderr)
        return 2
    data_path = args.data.resolve()
    if not data_path.is_file():
        print(f"error: compact data is not a file: {data_path}", file=sys.stderr)
        return 2
    source_root = (
        args.source_root
        or (
            Path(os.environ["FOMHD_SOURCE_ROOT"])
            if "FOMHD_SOURCE_ROOT" in os.environ
            else Path(__file__).resolve().parents[2]
        )
    ).resolve()
    renderer_path = (
        args.renderer or source_root / "tools/plots/build_a0_report.py"
    ).resolve()
    a0_html_path = args.a0_html.resolve()
    style_paths = (
        (source_root / "tools/plots/plot_tokamak_cae.py").resolve(),
        (source_root / "tools/plots/plot_cae_comparison.py").resolve(),
    )
    try:
        result = build(
            data_path,
            output,
            renderer_path,
            a0_html_path,
            style_paths,
        )
    except (InputError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
