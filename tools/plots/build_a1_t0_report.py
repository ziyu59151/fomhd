#!/usr/bin/env python3
"""Build an offline A0-style report for the controlled A1 T0 baseline."""

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
import textwrap
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ACCENT = "#00796B"
BLUE = "#0072B2"
ORANGE = "#D55E00"
DARK = "#37474F"
GRAY = "#6B7280"
PALE_GRAY = "#B0BEC5"

FIGURE_NAMES = (
    "fig01_resolved_frequencies",
    "fig02_micro_response",
    "fig03_target_accuracy",
    "fig04_high_mode_history",
    "fig05_fixed_H_refinement",
    "fig06_energy_residuals",
    "fig07_linear_and_costs",
)

A0_HTML_DEFAULT = Path(
    "/home/lxzeng/runs/fomhd/a0-benchmark-figures__20261104T050738Z/index.html"
)
CANONICAL_HASHES = {
    "renderer": "8d043646d669b0854b4acb27bf494eb704cb40c146eae841914dd7955a4597a5",
    "plot_tokamak_cae": "7cdb2fcfb77ab987d492b2a45f82c25a48209ffb7059647f8c700ce300b4068e",
    "plot_cae_comparison": (
        "3ef32bf13e78632fa80adbe71d45abbacb9de0b253e417cfecdf8ab34c1c850f"
    ),
}

CASE_KEYS = {
    "id",
    "coupling",
    "seed",
    "M",
    "micro_per_period",
    "H_over_T",
    "completed",
    "status",
    "time_over_T",
    "target_error",
    "target_phase_rad",
    "target_amplitude_ratio",
    "high_norm_ratio",
    "high_absolute_norm",
    "high_projection_ratio",
    "energy_time_over_T",
    "total_energy_residual_J",
    "kinetic_energy_residual_J",
    "energy_budget_J",
    "max_target_error",
    "max_phase_rad",
    "max_amplitude_drift",
    "max_high_gain",
    "final_high_gain",
    "target_accuracy_pass",
    "high_growth_pass",
    "hard_checks_pass",
    "rejected_time_over_T",
    "failure_message",
    "rejected_total_energy_residual_J",
    "rejected_kinetic_energy_residual_J",
    "elapsed_s",
    "estimated_peak_bytes",
    "model_calls",
}

STYLE_RE = re.compile(r"<style>(.*?)</style>", re.DOTALL)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
HEX_RE = re.compile(r"[0-9a-fA-F]{64}")


class InputError(ValueError):
    """Raised when supplied compact data or provenance violates the contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _nullable_bool(mapping: dict[str, Any], key: str, where: str) -> bool | None:
    value = mapping.get(key)
    if value is not None and not isinstance(value, bool):
        raise InputError(f"{where}.{key} must be true, false, or null")
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
    return (
        None
        if value is None
        else _number(value, f"{where}.{key}", minimum=minimum)
    )


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
        raise InputError(f"{where}.{key} must contain {expected} values")
    values = [
        _number(value, f"{where}.{key}[{index}]", minimum=minimum)
        for index, value in enumerate(raw)
    ]
    return values


def _nullable_numbers(
    mapping: dict[str, Any],
    key: str,
    where: str,
    *,
    expected: int | None = None,
    minimum: float | None = None,
) -> list[float] | None:
    if mapping.get(key) is None:
        return None
    return _numbers(
        mapping,
        key,
        where,
        expected=expected,
        minimum=minimum,
    )


def _nondecreasing(values: list[float], where: str) -> None:
    if any(right < left for left, right in itertools.pairwise(values)):
        raise InputError(f"{where} must be nondecreasing")


def _require_keys(mapping: dict[str, Any], keys: set[str], where: str) -> None:
    missing = sorted(keys.difference(mapping))
    if missing:
        raise InputError(f"{where} is missing required keys: {', '.join(missing)}")


def _protocol(data: dict[str, Any], where: str) -> dict[str, Any]:
    protocol = _object(data.get("protocol"), where)
    keys = {
        "T_s",
        "h_s",
        "cycles",
        "points",
        "target_mode",
        "high_mode",
        "micro_per_period",
        "target_amplitude",
        "high_amplitude",
        "target_error_limit",
        "amplitude_limit",
        "phase_limit_rad",
        "growth_limit",
    }
    _require_keys(protocol, keys, where)
    _number(protocol["T_s"], f"{where}.T_s", minimum=0.0)
    _number(protocol["h_s"], f"{where}.h_s", minimum=0.0)
    _integer(protocol, "cycles", where)
    _integer(protocol, "points", where)
    _integer(protocol, "target_mode", where)
    _integer(protocol, "high_mode", where)
    _integer(protocol, "micro_per_period", where)
    for key in (
        "target_amplitude",
        "high_amplitude",
        "target_error_limit",
        "amplitude_limit",
        "phase_limit_rad",
        "growth_limit",
    ):
        _number(protocol[key], f"{where}.{key}", minimum=0.0)
    expected = {
        "cycles": 2,
        "points": 9,
        "target_mode": 1,
        "high_mode": 4,
        "micro_per_period": 2048,
    }
    for key, value in expected.items():
        if protocol[key] != value:
            raise InputError(f"{where}.{key} must be {value} for this protocol")
    expected_float = {
        "target_amplitude": 2.0e-6,
        "high_amplitude": 2.0e-8,
        "target_error_limit": 0.01,
        "amplitude_limit": 0.01,
        "phase_limit_rad": 0.1,
        "growth_limit": 1.01,
    }
    for key, value in expected_float.items():
        if not math.isclose(
            float(protocol[key]),
            value,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise InputError(f"{where}.{key} must be {value!r}")
    return protocol


def _spectrum(data: dict[str, Any], where: str) -> dict[str, Any]:
    spectrum = _object(data.get("spectrum"), where)
    keys = {
        "m",
        "full_omega_rad_s",
        "micro_omega_rad_s",
        "acoustic_omega_rad_s",
        "generic_omega_rad_s",
        "chosen_h_s",
        "rk4_y_limit",
        "candidate_steps_per_period",
        "zero_mode_note",
        "full_jacobian_epsilon_max_abs_difference_s_inv",
    }
    _require_keys(spectrum, keys, where)
    modes = _numbers(spectrum, "m", where)
    if len(modes) != 4 or modes != [1.0, 2.0, 3.0, 4.0]:
        raise InputError(f"{where}.m must be [1, 2, 3, 4]")
    for key in (
        "full_omega_rad_s",
        "micro_omega_rad_s",
        "acoustic_omega_rad_s",
    ):
        _numbers(spectrum, key, where, expected=4)
    _number(
        spectrum["generic_omega_rad_s"],
        f"{where}.generic_omega_rad_s",
        minimum=0.0,
    )
    _number(spectrum["chosen_h_s"], f"{where}.chosen_h_s", minimum=0.0)
    _number(spectrum["rk4_y_limit"], f"{where}.rk4_y_limit", minimum=0.0)
    steps = _numbers(spectrum, "candidate_steps_per_period", where, minimum=1.0)
    if not {64.0, 128.0, 2048.0}.issubset(steps):
        raise InputError(
            f"{where}.candidate_steps_per_period must include 64, 128, 2048"
        )
    _text(spectrum, "zero_mode_note", where)
    _number(
        spectrum["full_jacobian_epsilon_max_abs_difference_s_inv"],
        f"{where}.full_jacobian_epsilon_max_abs_difference_s_inv",
        minimum=0.0,
    )
    return spectrum


def _micro_controls(data: dict[str, Any], where: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    keys = {
        "id",
        "label",
        "system",
        "micro_per_period",
        "time_over_T",
        "real_component",
        "reference_real_component",
        "norm_ratio",
        "vector_error",
        "completed",
        "status",
        "omega_rad_s",
    }
    for index, raw in enumerate(_items(data.get("micro_controls"), where)):
        item_where = f"{where}[{index}]"
        item = _object(raw, item_where)
        _require_keys(item, keys, item_where)
        _text(item, "id", item_where)
        _text(item, "label", item_where)
        system = _text(item, "system", item_where)
        if system not in {"full", "frozen"}:
            raise InputError(f"{item_where}.system must be full or frozen")
        _integer(item, "micro_per_period", item_where)
        times = _numbers(item, "time_over_T", item_where, minimum=0.0)
        _nondecreasing(times, f"{item_where}.time_over_T")
        if not times:
            raise InputError(f"{item_where}.time_over_T must not be empty")
        _numbers(item, "real_component", item_where, expected=len(times))
        _numbers(
            item,
            "reference_real_component",
            item_where,
            expected=len(times),
        )
        _numbers(item, "norm_ratio", item_where, expected=len(times))
        _numbers(
            item,
            "vector_error",
            item_where,
            expected=len(times),
            minimum=0.0,
        )
        _bool(item, "completed", item_where)
        _text(item, "status", item_where)
        _number(item["omega_rad_s"], f"{item_where}.omega_rad_s", minimum=0.0)
        records.append(item)
    return records


def _cases(data: dict[str, Any], where: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(_items(data.get("cases"), where)):
        item_where = f"{where}[{index}]"
        case = _object(raw, item_where)
        _require_keys(case, CASE_KEYS, item_where)
        _text(case, "id", item_where)
        coupling = _text(case, "coupling", item_where)
        if coupling not in {"synchronous", "predictor"}:
            raise InputError(f"{item_where}.coupling is invalid")
        seed = _text(case, "seed", item_where)
        if seed not in {"target", "mixed"}:
            raise InputError(f"{item_where}.seed is invalid")
        _integer(case, "M", item_where)
        _integer(case, "micro_per_period", item_where)
        _number(case["H_over_T"], f"{item_where}.H_over_T", minimum=0.0)
        completed = _bool(case, "completed", item_where)
        _text(case, "status", item_where)
        times = _numbers(case, "time_over_T", item_where, minimum=0.0)
        _nondecreasing(times, f"{item_where}.time_over_T")
        count = len(times)
        _numbers(
            case,
            "target_error",
            item_where,
            expected=count,
            minimum=0.0,
        )
        _numbers(case, "target_phase_rad", item_where, expected=count)
        _numbers(
            case,
            "target_amplitude_ratio",
            item_where,
            expected=count,
            minimum=0.0,
        )
        _nullable_numbers(case, "high_norm_ratio", item_where, expected=count)
        _numbers(
            case,
            "high_absolute_norm",
            item_where,
            expected=count,
            minimum=0.0,
        )
        _nullable_numbers(case, "high_projection_ratio", item_where, expected=count)
        energy_times = _numbers(
            case,
            "energy_time_over_T",
            item_where,
            minimum=0.0,
        )
        _nondecreasing(energy_times, f"{item_where}.energy_time_over_T")
        energy_count = len(energy_times)
        _numbers(
            case,
            "total_energy_residual_J",
            item_where,
            expected=energy_count,
        )
        _numbers(
            case,
            "kinetic_energy_residual_J",
            item_where,
            expected=energy_count,
        )
        _number(case["energy_budget_J"], f"{item_where}.energy_budget_J", minimum=0.0)
        for key in (
            "max_target_error",
            "max_amplitude_drift",
            "max_high_gain",
            "final_high_gain",
        ):
            _nullable_number(case, key, item_where, minimum=0.0)
        _nullable_number(case, "max_phase_rad", item_where)
        _nullable_bool(case, "target_accuracy_pass", item_where)
        _nullable_bool(case, "high_growth_pass", item_where)
        hard_pass = _bool(case, "hard_checks_pass", item_where)
        rejected_time = _nullable_number(
            case,
            "rejected_time_over_T",
            item_where,
            minimum=0.0,
        )
        _nullable_text(case, "failure_message", item_where)
        _nullable_number(
            case,
            "rejected_total_energy_residual_J",
            item_where,
        )
        _nullable_number(
            case,
            "rejected_kinetic_energy_residual_J",
            item_where,
        )
        _number(case["elapsed_s"], f"{item_where}.elapsed_s", minimum=0.0)
        _nullable_number(
            case,
            "estimated_peak_bytes",
            item_where,
            minimum=0.0,
        )
        _nullable_number(case, "model_calls", item_where, minimum=0.0)
        if not completed and rejected_time is None and not hard_pass:
            raise InputError(
                f"{item_where} is rejected but has no rejected_time_over_T"
            )
        records.append(case)
    return records


def _linear(data: dict[str, Any], where: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    keys = {
        "m",
        "M",
        "H_over_T",
        "rho",
        "rho_over_2T",
        "seed_gain_max",
        "seed_gain_final",
        "omega_H_rad",
        "phase_alias",
    }
    for index, raw in enumerate(_items(data.get("linear"), where)):
        item_where = f"{where}[{index}]"
        item = _object(raw, item_where)
        _require_keys(item, keys, item_where)
        _integer(item, "m", item_where)
        _integer(item, "M", item_where)
        for key in (
            "H_over_T",
            "rho",
            "rho_over_2T",
            "seed_gain_max",
            "seed_gain_final",
            "omega_H_rad",
        ):
            _number(item[key], f"{item_where}.{key}", minimum=0.0)
        _bool(item, "phase_alias", item_where)
        records.append(item)
    return records


def _legacy(data: dict[str, Any], where: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    keys = {
        "label",
        "micro_per_period",
        "M",
        "H_over_T",
        "status",
        "warning",
        "target_error",
    }
    for index, raw in enumerate(_items(data.get("legacy"), where)):
        item_where = f"{where}[{index}]"
        item = _object(raw, item_where)
        _require_keys(item, keys, item_where)
        _text(item, "label", item_where)
        _integer(item, "micro_per_period", item_where)
        _integer(item, "M", item_where)
        _number(item["H_over_T"], f"{item_where}.H_over_T", minimum=0.0)
        _text(item, "status", item_where)
        _text(item, "warning", item_where)
        _nullable_number(item, "target_error", item_where, minimum=0.0)
        records.append(item)
    return records


def validate_data(data: dict[str, Any]) -> dict[str, Any]:
    """Validate the compact-data contract without altering scientific values."""
    required = {
        "protocol",
        "source_commit",
        "scope",
        "input_provenance",
        "spectrum",
        "micro_controls",
        "cases",
        "linear",
        "legacy",
        "recommendation",
    }
    _require_keys(data, required, "data")
    protocol = _protocol(data, "protocol")
    _text(data, "source_commit", "data")
    _text(data, "scope", "data")
    _spectrum(data, "spectrum")
    if not _micro_controls(data, "micro_controls"):
        raise InputError("micro_controls must not be empty")
    if not _cases(data, "cases"):
        raise InputError("cases must not be empty")
    if not _linear(data, "linear"):
        raise InputError("linear must not be empty")
    _legacy(data, "legacy")
    _text(data, "recommendation", "data")
    inputs = _items(data.get("input_provenance"), "input_provenance")
    for index, raw in enumerate(inputs):
        where = f"input_provenance[{index}]"
        item = _object(raw, where)
        _require_keys(item, {"path", "sha256", "role"}, where)
        _text(item, "path", where)
        digest = _text(item, "sha256", where)
        if not HEX_RE.fullmatch(digest):
            raise InputError(f"{where}.sha256 must be a SHA-256 hex digest")
        _text(item, "role", where)
    spectrum = data["spectrum"]
    if not math.isclose(
        float(protocol["h_s"]),
        float(protocol["T_s"]) / protocol["micro_per_period"],
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ):
        raise InputError("protocol.h_s must equal protocol.T_s / 2048")
    if not math.isclose(
        float(spectrum["chosen_h_s"]),
        float(protocol["h_s"]),
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ):
        raise InputError("spectrum.chosen_h_s must equal protocol.h_s")
    if not math.isclose(
        float(spectrum["rk4_y_limit"]),
        2.0 * math.sqrt(2.0),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise InputError("spectrum.rk4_y_limit must be 2*sqrt(2)")
    return protocol


def resolve_local_path(text: str, base: Path) -> Path:
    path = Path(text)
    return (path if path.is_absolute() else base / path).resolve()


def verify_input_provenance(
    data: dict[str, Any],
    data_path: Path,
) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for index, raw in enumerate(data["input_provenance"]):
        where = f"input_provenance[{index}]"
        item = _object(raw, where)
        path = resolve_local_path(item["path"], data_path.parent)
        if not path.is_file():
            raise InputError(f"{where} local file does not exist: {path}")
        actual = sha256_file(path)
        if actual != item["sha256"].casefold():
            raise InputError(f"{where} SHA-256 mismatch: {path}")
        verified.append(
            {
                "path": str(path),
                "sha256": actual,
                "role": item["role"],
            }
        )
    return verified


def load_renderer(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("canonical_a0_renderer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load renderer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def style_text(path: Path) -> bytes:
    match = STYLE_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise InputError(f"no HTML style block found in {path}")
    return match.group(1).encode("utf-8")


def check_canonical_files(
    renderer_path: Path,
    a0_html_path: Path,
    style_paths: tuple[Path, Path],
) -> dict[str, str]:
    for path in (renderer_path, a0_html_path, *style_paths):
        if not path.is_file():
            raise InputError(f"required canonical file does not exist: {path}")
    hashes = {
        "renderer": sha256_file(renderer_path),
        "a0_html": sha256_file(a0_html_path),
        "plot_tokamak_cae": sha256_file(style_paths[0]),
        "plot_cae_comparison": sha256_file(style_paths[1]),
    }
    reviewed = {
        key: hashes[key]
        for key in ("renderer", "plot_tokamak_cae", "plot_cae_comparison")
    }
    if reviewed != CANONICAL_HASHES:
        raise InputError(
            "canonical renderer or A0 style source differs from the reviewed hashes"
        )
    if style_text(renderer_path) != style_text(a0_html_path):
        raise InputError("canonical renderer CSS does not match A0 HTML byte-for-byte")
    return hashes


def configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return plt


def style_axis(axis: Any) -> None:
    axis.grid(True, which="both", alpha=0.18)
    axis.tick_params(labelsize=7.5)


def panel_label(axis: Any, letter: str) -> None:
    axis.text(
        -0.025,
        1.015,
        f"({letter})",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        clip_on=False,
    )


def save_figure(fig: Any, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=160, facecolor="white")
    fig.savefig(output / f"{name}.svg", facecolor="white")


def _accepted_indices(case: dict[str, Any], times: list[float]) -> list[int]:
    limit = case["rejected_time_over_T"]
    if case["completed"] or limit is None:
        return list(range(len(times)))
    return [index for index, time in enumerate(times) if time < limit]


def accepted_horizon(case: dict[str, Any]) -> float:
    times = case["time_over_T"]
    return max((times[index] for index in _accepted_indices(case, times)), default=0.0)


def _series(case: dict[str, Any], key: str, indices: list[int]) -> list[float]:
    values = case[key]
    return [float(values[index]) for index in indices]


def target_accuracy_cases(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only completed target-seed cases with all final metrics."""
    return [
        case
        for case in data["cases"]
        if case["seed"] == "target"
        and case["completed"]
        and case["max_target_error"] is not None
        and case["max_phase_rad"] is not None
        and case["max_amplitude_drift"] is not None
    ]


def accepted_history(
    case: dict[str, Any],
    time_key: str,
    value_key: str,
) -> tuple[list[float], list[float]]:
    """Return only recorded accepted samples strictly before a rejection time."""
    times = [float(value) for value in case[time_key]]
    indices = _accepted_indices(case, times)
    return (
        [times[index] for index in indices],
        _series(case, value_key, indices),
    )


def _case_style(case: dict[str, Any], chosen_h_micro: int) -> tuple[str, str, str]:
    half = case["micro_per_period"] == 2 * chosen_h_micro
    if case["coupling"] == "synchronous":
        color = DARK if case["seed"] == "target" else GRAY
        return color, ":", "^"
    color = ACCENT if case["seed"] == "target" else ORANGE
    if half:
        color = BLUE if case["seed"] == "target" else PALE_GRAY
    return color, "--" if half else "-", "s" if half else "o"


def render_figures(data: dict[str, Any], output: Path) -> None:
    import numpy as np

    protocol = data["protocol"]
    spectrum = data["spectrum"]
    chosen_h_micro = protocol["micro_per_period"]

    plt = configure_matplotlib()

    modes = np.asarray(spectrum["m"], dtype=float)
    full = np.asarray(spectrum["full_omega_rad_s"], dtype=float)
    frozen = np.asarray(spectrum["micro_omega_rad_s"], dtype=float)
    acoustic = np.asarray(spectrum["acoustic_omega_rad_s"], dtype=float)
    generic = float(spectrum["generic_omega_rad_s"])
    steps = np.asarray(spectrum["candidate_steps_per_period"], dtype=float)
    threshold = float(spectrum["rk4_y_limit"])
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), layout="constrained")
    fig.suptitle("Resolved modal frequencies and RK4 interval position", fontsize=10.5)
    axes[0].plot(modes, full, color=ACCENT, marker="o", label="Full production spectrum")
    axes[0].plot(
        modes,
        frozen,
        color=BLUE,
        marker="s",
        linestyle="--",
        label="Frozen-ion electron-field spectrum",
    )
    axes[0].plot(
        modes,
        acoustic,
        color=ORANGE,
        marker="^",
        linestyle="-.",
        label="Acoustic branch",
    )
    axes[0].axhline(
        generic,
        color=GRAY,
        linestyle=":",
        linewidth=1.2,
        label="Generic operator estimate (not an eigenfrequency)",
    )
    axes[0].set_yscale("log")
    axes[0].set_xticks(modes, labels=[str(int(mode)) for mode in modes])
    axes[0].set_xlabel("Spatial harmonic m")
    axes[0].set_ylabel("Angular frequency [rad/s]")
    axes[0].set_title("Discrete spectra and estimate")
    axes[0].legend(loc="best")

    candidate_h = protocol["T_s"] / steps
    for values, color, marker, linestyle, label in (
        (full, ACCENT, "o", "-", "Full spectrum maximum"),
        (frozen, BLUE, "s", "--", "Frozen-ion maximum"),
        (acoustic, ORANGE, "^", "-.", "Acoustic branch maximum"),
    ):
        ratio = float(np.max(values)) * candidate_h / threshold
        axes[1].plot(
            steps,
            ratio,
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
    axes[1].plot(
        steps,
        generic * candidate_h / threshold,
        color=GRAY,
        marker="x",
        linestyle=":",
        label="Generic estimate",
    )
    axes[1].axhline(1.0, color=ORANGE, linewidth=1.0, linestyle="--")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xticks(
        [64, 128, 2048],
        labels=["64\nold warning", "128\nold warning", "2048\nchosen h*"],
    )
    axes[1].minorticks_off()
    axes[1].set_xlabel("Microsteps per target period")
    axes[1].set_ylabel("omega*h / rk4_y_limit")
    axes[1].set_title("Distance from the RK4 imaginary-axis limit")
    axes[1].legend(loc="best")
    axes[1].text(
        0.02,
        0.03,
        spectrum["zero_mode_note"],
        transform=axes[1].transAxes,
        fontsize=7,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": PALE_GRAY, "pad": 2},
    )
    for axis, letter in zip(axes, ("a", "b"), strict=True):
        style_axis(axis)
        panel_label(axis, letter)
    save_figure(fig, output, FIGURE_NAMES[0])
    plt.close(fig)

    controls = data["micro_controls"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), layout="constrained")
    fig.suptitle("High-frequency micro response and vector error", fontsize=10.5)
    for index, control in enumerate(controls):
        full_system = control["system"] == "full"
        color = ACCENT if full_system else BLUE
        linestyle = "-" if control["micro_per_period"] == chosen_h_micro else "--"
        label = (
            f"{'Full' if full_system else 'Frozen'}, "
            f"{control['micro_per_period']} per T"
        )
        times = np.asarray(control["time_over_T"], dtype=float)
        real = np.asarray(control["real_component"], dtype=float)
        reference = np.asarray(control["reference_real_component"], dtype=float)
        error = np.asarray(control["vector_error"], dtype=float)
        zoom = times <= 0.3 + 8.0 * sys.float_info.epsilon
        axes[0].plot(
            times[zoom],
            real[zoom],
            color=color,
            linestyle=linestyle,
            marker="o" if index == 0 else None,
            markersize=2.5,
            label=label,
        )
        axes[0].plot(
            times[zoom],
            reference[zoom],
            color="black",
            linestyle=":",
            linewidth=0.8,
        )
        visible_error = error > 0.0
        axes[1].semilogy(
            times[visible_error],
            error[visible_error],
            color=color,
            linestyle=linestyle,
            label=label,
        )
    axes[0].plot([], [], color="black", linestyle=":", label="Micro-run reference")
    axes[0].set_xlim(0.0, 0.3)
    axes[0].set_xlabel("Time t/T (first 0.3T shown)")
    axes[0].set_ylabel("Real high-frequency component")
    axes[0].set_title("Waveform response")
    axes[0].legend(loc="best")
    axes[1].set_xlabel("Time t/T (full supplied horizon)")
    axes[1].set_ylabel("Relative vector error")
    axes[1].set_title("Error exposes coincident responses")
    axes[1].legend(loc="best")
    for axis, letter in zip(axes, ("a", "b"), strict=True):
        style_axis(axis)
        panel_label(axis, letter)
    save_figure(fig, output, FIGURE_NAMES[1])
    plt.close(fig)

    target_cases = target_accuracy_cases(data)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.9), layout="constrained")
    fig.suptitle("Completed target-mode accuracy versus macro interval", fontsize=10.5)
    grouped_targets: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for case in target_cases:
        grouped_targets.setdefault(
            (case["coupling"], case["micro_per_period"]),
            [],
        ).append(case)
    for (coupling, micro), cases in sorted(grouped_targets.items()):
        representative = cases[0]
        color, linestyle, marker = _case_style(representative, chosen_h_micro)
        label = (
            f"{coupling}, "
            f"{'h*' if micro == chosen_h_micro else 'h*/2'}"
        )
        common = {
            "color": color,
            "linestyle": linestyle,
            "marker": marker,
            "markersize": 4.5,
        }
        cases = sorted(cases, key=lambda item: item["H_over_T"])
        for axis, key in zip(
            axes,
            ("max_target_error", "max_phase_rad", "max_amplitude_drift"),
            strict=True,
        ):
            axis.plot(
                [case["H_over_T"] for case in cases],
                [abs(float(case[key])) for case in cases],
                label=label,
                **common,
            )
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Macro interval H/T")
        axis.legend(loc="best")
        style_axis(axis)
    axes[0].set_yscale("log")
    axes[0].axhline(protocol["target_error_limit"], color=ORANGE, linestyle="--")
    axes[0].set_ylabel("Maximum relative vector error")
    axes[0].set_title("Vector error; criterion 1%")
    axes[1].set_yscale("symlog", linthresh=protocol["phase_limit_rad"])
    axes[1].axhline(protocol["phase_limit_rad"], color=ORANGE, linestyle="--")
    axes[1].axhline(-protocol["phase_limit_rad"], color=ORANGE, linestyle="--")
    axes[1].set_ylabel("Maximum absolute phase [rad]")
    axes[1].set_title("Phase; criterion 0.1 rad")
    axes[2].set_yscale("log")
    axes[2].axhline(protocol["amplitude_limit"], color=ORANGE, linestyle="--")
    axes[2].set_ylabel("Maximum relative amplitude drift")
    axes[2].set_title("Amplitude; criterion 1%")
    rejected_count = sum(
        not case["completed"] and case["seed"] == "target"
        for case in data["cases"]
    )
    if rejected_count:
        fig.supxlabel(
            f"{rejected_count} rejected target-seed window(s) omitted; "
            "see Figure 6 and the data record.",
            fontsize=7,
            color="#9A3412",
        )
    for axis, letter in zip(axes, ("a", "b", "c"), strict=True):
        panel_label(axis, letter)
    save_figure(fig, output, FIGURE_NAMES[2])
    plt.close(fig)

    high_cases = [
        case
        for case in data["cases"]
        if case["seed"] == "mixed" and case["high_absolute_norm"]
    ]
    fig, (axis, gain_zoom) = plt.subplots(1, 2, figsize=(14.0, 5.4), gridspec_kw={"width_ratios": (1.7, 1.0)}, layout="constrained")
    fig.suptitle(
        "Absolute seeded high-mode norm in accepted nonlinear histories",
        fontsize=10.5,
    )
    for case in sorted(high_cases, key=lambda item: (item["M"], item["seed"])):
        times = np.asarray(case["time_over_T"], dtype=float)
        accepted = _accepted_indices(case, case["time_over_T"])
        if not accepted:
            continue
        norm = np.asarray(_series(case, "high_absolute_norm", accepted), dtype=float)
        plot_times = times[accepted]
        color, linestyle, marker = _case_style(case, chosen_h_micro)
        if case["coupling"] == "predictor":
            styles = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 2, 1, 2)))
            selected = (4, 8, 16, 32, 64, 128, 256, 512).index(case["M"])
            linestyle = styles[selected]
        label = (
            f"M={case['M']}, {case['coupling']}, "
            f"{'h*' if case['micro_per_period'] == chosen_h_micro else 'h*/2'}"
        )
        positive = norm > 0.0
        axis.semilogy(
            plot_times[positive],
            norm[positive],
            color=color,
            linestyle=linestyle,
            linewidth=1.1,
            label=label,
        )
        if not case["completed"] and norm[-1] > 0.0:
            axis.plot(
                plot_times[-1],
                norm[-1],
                color=ORANGE,
                marker="X",
                markersize=7,
                linestyle="none",
                label=f"last accepted, M={case['M']}",
            )
    axis.set_xlabel("Accepted nonlinear time t/T")
    axis.set_ylabel("Absolute high-mode four-component norm")
    axis.legend(loc="best", ncol=2)
    style_axis(axis)
    panel_label(axis, "a")
    for case in high_cases:
        if case["micro_per_period"] != chosen_h_micro:
            continue
        if case["coupling"] == "predictor" and case["M"] not in (4, 8, 16):
            continue
        if case["high_norm_ratio"] is None:
            continue
        indices = _accepted_indices(case, case["time_over_T"])
        gain = _series(case, "high_norm_ratio", indices)
        color = {4: ACCENT, 8: BLUE, 16: ORANGE}.get(case["M"], GRAY)
        label = f"M={case['M']}, max={max(gain):.6g}" if case["coupling"] == "predictor" else "Synchronous"
        gain_zoom.plot([case["time_over_T"][i] for i in indices], gain, color=color, label=label)
    gain_zoom.axhline(protocol["growth_limit"], color=GRAY, linestyle="--", label="1% growth criterion")
    gain_zoom.set_xlabel("Accepted nonlinear time t/T")
    gain_zoom.set_ylabel("Seeded high-mode norm / initial norm")
    gain_zoom.set_title("Small macros and the fixed 1% criterion")
    gain_zoom.legend(loc="best", fontsize=7)
    style_axis(gain_zoom)
    panel_label(gain_zoom, "b")
    save_figure(fig, output, FIGURE_NAMES[3])
    plt.close(fig)

    cross_cases = [
        case
        for case in data["cases"]
        if case["coupling"] == "predictor"
        and case["micro_per_period"] in {chosen_h_micro, 2 * chosen_h_micro}
        and (
            math.isclose(case["H_over_T"], 1.0 / 16.0, rel_tol=0.0, abs_tol=1.0e-15)
            or math.isclose(
                case["H_over_T"],
                1.0 / 8.0,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        )
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.9), layout="constrained")
    fig.suptitle("Fixed-H microstep-refinement comparison", fontsize=10.5)
    fixed_h = (1.0 / 16.0, 1.0 / 8.0)
    positions = {value: index for index, value in enumerate(fixed_h)}
    cross_points: dict[tuple[str, bool, str], list[tuple[float, float]]] = {}
    for case in cross_cases:
        if not case["completed"]:
            continue
        position = min(positions, key=lambda value: abs(value - case["H_over_T"]))
        x_value = positions[position] + (-0.08 if case["seed"] == "target" else 0.08)
        half = case["micro_per_period"] == 2 * chosen_h_micro
        color = BLUE if half else ACCENT
        marker = "s" if half else "o"
        label = "h*/2" if half else "h*"
        y_values = {
            "max_target_error": abs(float(case["max_target_error"])),
            "max_phase_rad": abs(float(case["max_phase_rad"])),
            "max_amplitude_drift": abs(float(case["max_amplitude_drift"])),
        }
        for axis, key in zip(
            axes,
            ("max_target_error", "max_phase_rad", "max_amplitude_drift"),
            strict=True,
        ):
            cross_points.setdefault((case["seed"], half, key), []).append(
                (x_value, y_values[key])
            )
            axis.annotate(
                f"2T, {case['elapsed_s']:.3g} s",
                (x_value, y_values[key]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=6.5,
            )
    for (seed, half, key), points in cross_points.items():
        axis = axes[
            ("max_target_error", "max_phase_rad", "max_amplitude_drift").index(key)
        ]
        color = (BLUE if half else ACCENT) if seed == "target" else (GRAY if half else ORANGE)
        marker = "s" if half else "o"
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color=color,
            marker=marker,
            linestyle="none",
            label=f"{seed}, {'h*/2' if half else 'h*'}",
        )
    for axis in axes:
        axis.set_xticks([0, 1], labels=["H=T/16", "H=T/8"])
        axis.set_xlim(-0.35, 1.35)
        axis.legend(loc="lower right")
        style_axis(axis)
    axes[0].set_yscale("log")
    axes[0].axhline(protocol["target_error_limit"], color=ORANGE, linestyle="--")
    axes[0].set_ylabel("Maximum relative vector error")
    axes[0].set_title("Target vector error")
    axes[1].set_yscale("symlog", linthresh=protocol["phase_limit_rad"])
    axes[1].axhline(protocol["phase_limit_rad"], color=ORANGE, linestyle="--")
    axes[1].axhline(-protocol["phase_limit_rad"], color=ORANGE, linestyle="--")
    axes[1].set_ylabel("Maximum absolute phase [rad]")
    axes[1].set_title("Target phase")
    axes[2].set_yscale("log")
    axes[2].axhline(protocol["amplitude_limit"], color=ORANGE, linestyle="--")
    axes[2].set_ylabel("Maximum relative amplitude drift")
    axes[2].set_title("Amplitude drift")
    rejected_cross = [case for case in cross_cases if not case["completed"]]
    if rejected_cross:
        rejected_text = "; ".join(
            (
                f"{case['seed']}, M={case['M']} ({'h*/2' if case['micro_per_period'] == 2 * chosen_h_micro else 'h*'}) "
                f"rejected at {case['rejected_time_over_T']:.3g}T"
            )
            for case in rejected_cross
        )
        fig.supxlabel(
            textwrap.fill(f"Rejected candidates are not final points: {rejected_text}.", 150),
            fontsize=7,
            color="#9A3412",
        )
    for axis, letter in zip(axes, ("a", "b", "c"), strict=True):
        panel_label(axis, letter)
    save_figure(fig, output, FIGURE_NAMES[4])
    plt.close(fig)

    residual_cases = [
        case
        for case in data["cases"]
        if case["coupling"] == "predictor"
        and case["micro_per_period"] in {chosen_h_micro, 2 * chosen_h_micro}
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), layout="constrained")
    fig.suptitle(
        "Raw energy residuals divided by the unchanged supplied budget",
        fontsize=10.5,
    )
    legend_seen: set[str] = set()
    for case in residual_cases:
        times = case["energy_time_over_T"]
        limit = case["rejected_time_over_T"]
        indices = list(range(len(times)))
        if not case["completed"] and limit is not None:
            indices = _accepted_indices(case, times)
        if not indices:
            continue
        color, linestyle, marker = _case_style(case, chosen_h_micro)
        cadence = (
            "h*/2"
            if case["micro_per_period"] == 2 * chosen_h_micro
            else "h*"
        )
        label = f"{case['coupling']} {case['seed']} {cadence}"
        legend_label = None if label in legend_seen else label
        legend_seen.add(label)
        for axis, key in zip(
            axes,
            ("total_energy_residual_J", "kinetic_energy_residual_J"),
            strict=True,
        ):
            normalized = np.abs(
                np.asarray(_series(case, key, indices), dtype=float)
            ) / float(case["energy_budget_J"])
            positive = normalized > 0.0
            axis.semilogy(
                np.asarray(times)[indices][positive],
                normalized[positive],
                color=color,
                linestyle=linestyle,
                linewidth=0.9,
                alpha=0.72,
                label=legend_label,
            )
    if any(not case["completed"] for case in residual_cases):
        for axis in axes:
            axis.plot(
                [],
                [],
                color=ORANGE,
                marker="X",
                markersize=7,
                linestyle="none",
                label="rejected candidate",
            )
    for case in residual_cases:
        if case["completed"]:
            continue
        for axis, key in zip(
            axes,
            (
                "rejected_total_energy_residual_J",
                "rejected_kinetic_energy_residual_J",
            ),
            strict=True,
        ):
            rejected = case[key]
            if rejected is None:
                continue
            axis.plot(
                case["rejected_time_over_T"],
                abs(float(rejected)) / float(case["energy_budget_J"]),
                color=ORANGE,
                marker="X",
                markersize=7,
                linestyle="none",
            )
    for axis, title in zip(
        axes,
        ("Total-energy residual", "Kinetic-energy residual"),
        strict=True,
    ):
        axis.set_yscale("log")
        axis.axhline(1.0, color=ORANGE, linestyle="--")
        axis.set_xlabel("Accepted residual time t/T")
        axis.set_ylabel("abs(raw residual) / supplied budget")
        axis.set_title(title)
        axis.legend(loc="best", ncol=2, fontsize=7)
        style_axis(axis)
    for axis, letter in zip(axes, ("a", "b"), strict=True):
        panel_label(axis, letter)
    save_figure(fig, output, FIGURE_NAMES[5])
    plt.close(fig)

    fig = plt.figure(figsize=(15.0, 9.2), layout="constrained")
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 0.08, 1.75),
        height_ratios=(1.0, 1.0),
    )
    radius_axis = fig.add_subplot(grid[0, 0])
    gain_axis = fig.add_subplot(grid[1, 0])
    qualification_axis = fig.add_subplot(grid[0, 2])
    cost_axis = fig.add_subplot(grid[1, 2])
    qualification_axis.axis("off")
    cost_axis.axis("off")
    fig.suptitle("Linear interval maps, nonlinear qualification, and costs", fontsize=10.5)
    linear_rows = sorted(data["linear"], key=lambda item: (item["m"], item["M"]))
    for mode in sorted({item["m"] for item in linear_rows}):
        rows = [item for item in linear_rows if item["m"] == mode]
        color = {1: ACCENT, 2: BLUE, 3: DARK, 4: ORANGE}[mode]
        x_values = [item["H_over_T"] for item in rows]
        radius_axis.plot(
            x_values,
            [item["rho"] for item in rows],
            color=color,
            marker="o",
            linestyle="-",
            label=f"m={mode}: one-step radius",
        )
        radius_axis.plot(
            x_values,
            [
                item["rho_over_2T"] if item["rho_over_2T"] > 0.0 else np.nan
                for item in rows
            ],
            color=color,
            marker="s",
            linestyle="--",
            label=f"m={mode}: radius over 2T",
        )
        gain_axis.plot(
            x_values,
            [item["seed_gain_max"] for item in rows],
            color=color,
            marker="o",
            linestyle="-",
            label=f"m={mode}: maximum seed gain",
        )
        gain_axis.plot(
            x_values,
            [item["seed_gain_final"] for item in rows],
            color=color,
            marker="s",
            linestyle="--",
            label=f"m={mode}: final seed gain",
        )
    for axis in (radius_axis, gain_axis):
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Macro interval H/T")
        axis.set_yscale("log")
        axis.legend(loc="best")
        style_axis(axis)
    radius_axis.axhline(1.0, color=GRAY, linestyle=":")
    radius_axis.set_ylabel("LIN map radius")
    radius_axis.set_title("LIN | FOMHD macro-map growth")
    gain_axis.axhline(protocol["growth_limit"], color=ORANGE, linestyle="--")
    gain_axis.set_ylabel("LIN seed gain")
    gain_axis.set_title("LIN prediction; 1% amplification line")
    panel_label(radius_axis, "a")
    panel_label(gain_axis, "b")

    primary_cases = [
        case
        for case in data["cases"]
        if case["coupling"] == "predictor"
        and case["micro_per_period"] == chosen_h_micro
    ]
    h_values = sorted({case["H_over_T"] for case in primary_cases})

    def case_for(h_value: float, seed: str) -> dict[str, Any] | None:
        return next(
            (
                case
                for case in primary_cases
                if math.isclose(case["H_over_T"], h_value, rel_tol=0.0, abs_tol=1e-15)
                and case["seed"] == seed
            ),
            None,
        )

    def qualification(case: dict[str, Any] | None, key: str) -> str:
        if case is None:
            return "missing"
        if not case["completed"]:
            return "undetermined"
        value = case[key]
        return "not assessed" if value is None else "pass" if value else "fail"

    qualification_rows = []
    for h_value in h_values:
        target = case_for(h_value, "target")
        mixed = case_for(h_value, "mixed")
        qualification_rows.append(
            [
                f"{h_value:.5g}",
                qualification(target, "target_accuracy_pass"),
                qualification(mixed, "target_accuracy_pass"),
                qualification(target, "high_growth_pass"),
                qualification(mixed, "high_growth_pass"),
                (
                    "pass"
                    if target is not None and target["hard_checks_pass"]
                    else "rejected"
                    if target is not None and not target["completed"]
                    else "fail"
                    if target is not None
                    else "missing"
                ),
                (
                    "pass"
                    if mixed is not None and mixed["hard_checks_pass"]
                    else "rejected"
                    if mixed is not None and not mixed["completed"]
                    else "fail"
                    if mixed is not None
                    else "missing"
                ),
            ]
        )
    qualification_table = qualification_axis.table(
        cellText=qualification_rows or [["no h* predictor cases", *(["-"] * 6)]],
        colLabels=[
            "H/T",
            "Target\naccuracy",
            "Mixed\ntarget accuracy",
            "Target\nhigh growth",
            "Mixed\nhigh growth",
            "Target\nhard checks",
            "Mixed\nhard checks",
        ],
        loc="upper center",
        cellLoc="center",
        colLoc="center",
        bbox=(0.0, 0.0, 1.0, 1.0),
    )
    qualification_table.auto_set_font_size(False)
    qualification_table.set_fontsize(7)
    qualification_axis.set_title(
        "Observed qualification at h* (rejected windows: undetermined)",
        fontsize=9.5,
        pad=5,
    )
    panel_label(qualification_axis, "c")

    cost_rows = []
    for h_value in h_values:
        target = case_for(h_value, "target")
        mixed = case_for(h_value, "mixed")
        cost_rows.append(
            [
                (
                    f"{h_value:.5g} (rejected)"
                    if (
                        target is not None
                        and not target["completed"]
                    )
                    or (
                        mixed is not None
                        and not mixed["completed"]
                    )
                    else f"{h_value:.5g}"
                ),
                "-" if target is None else f"{target['elapsed_s']:.3g} / {accepted_horizon(target):.3g}T",
                "-" if mixed is None else f"{mixed['elapsed_s']:.3g} / {accepted_horizon(mixed):.3g}T",
                (
                    "-"
                    if target is None or target["model_calls"] is None
                    else f"{target['model_calls']:.3g}"
                ),
                (
                    "-"
                    if mixed is None or mixed["model_calls"] is None
                    else f"{mixed['model_calls']:.3g}"
                ),
                (
                    "-"
                    if target is None or target["estimated_peak_bytes"] is None
                    else f"{target['estimated_peak_bytes'] / 2**20:.3g}"
                ),
                (
                    "-"
                    if mixed is None or mixed["estimated_peak_bytes"] is None
                    else f"{mixed['estimated_peak_bytes'] / 2**20:.3g}"
                ),
            ]
        )
    cost_table = cost_axis.table(
        cellText=cost_rows or [["no h* predictor cases", *(["-"] * 6)]],
        colLabels=[
            "H/T",
            "Target\ns / horizon",
            "Mixed\ns / horizon",
            "Target\ncalls",
            "Mixed\ncalls",
            "Target\nMiB est.",
            "Mixed\nMiB est.",
        ],
        loc="upper center",
        cellLoc="center",
        colLoc="center",
        bbox=(0.0, 0.0, 1.0, 1.0),
    )
    cost_table.auto_set_font_size(False)
    cost_table.set_fontsize(7)
    cost_axis.set_title(
        "Diagnostic-inclusive run cost; memory is an estimate, not peak",
        fontsize=9.5,
        pad=5,
    )
    panel_label(cost_axis, "d")
    save_figure(fig, output, FIGURE_NAMES[6])
    plt.close(fig)


def figure_record(
    number: int,
    name: str,
    title: str,
    caption: str,
    source_kind: str,
    *,
    components: list[str] | None = None,
    panels: list[tuple[str, str]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
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
        record["components"] = components
    if panels:
        record["panels"] = [
            {"letter": letter, "description": description}
            for letter, description in panels
        ]
    if metadata:
        record.update(metadata)
    return record


def refinement_notes(data: dict[str, Any]) -> list[str]:
    """Retain both seeds and partial windows beside the completed-point plot."""
    notes = []
    for case in data["cases"]:
        if case["coupling"] != "predictor" or not any(
            math.isclose(case["H_over_T"], value, abs_tol=1e-15)
            for value in (1 / 16, 1 / 8)
        ):
            continue
        end = accepted_horizon(case)
        target = case["max_target_error"]
        gain = case["max_high_gain"]
        error_text = "unavailable" if target is None else f"{target:.6g}"
        gain_text = "undefined for unseeded high mode" if gain is None else f"{gain:.6g}"
        notes.append(
            f"{case['seed']}, H/T={case['H_over_T']:.5g}, "
            f"{case['micro_per_period']} steps/T: {case['status']}, accepted to "
            f"{end:.6g}T; {'full-window' if case['completed'] else 'prefix-only'} "
            f"target error {error_text}, maximum high-mode gain {gain_text}. "
            "Raw accepted/rejected energy checks are separate in Figure 6."
        )
    return notes


def build_manifest(data: dict[str, Any]) -> dict[str, Any]:
    protocol = data["protocol"]
    cases = data["cases"]
    completed = sum(case["completed"] for case in cases)
    rejected = len(cases) - completed
    legacy = data["legacy"]
    criteria = (
        f"{protocol['target_error_limit']:.3g} vector error, "
        f"{protocol['amplitude_limit']:.3g} amplitude drift, "
        f"{protocol['phase_limit_rad']:.3g} rad phase, and a "
        f"{protocol['growth_limit']:.3g} high-mode growth flag"
    )
    fig1 = figure_record(
        1,
        FIGURE_NAMES[0],
        "Resolved frequencies and old/chosen interval position",
        (
            "Full production-spectrum, frozen-ion electron-field, and acoustic "
            "branch values come from the supplied compact data. The generic "
            "operator estimate is shown separately and is not a measured "
            "eigenfrequency. Panel (b) divides omega*h by the supplied RK4 "
            "imaginary-axis limit for old 64/128 steps per target period and "
            "the chosen h*. The excluded zero mode and its caveat remain in "
            "data.json."
        ),
        "mixed_comparison",
        components=[
            "full production-RHS spectrum",
            "frozen-ion electron-field spectrum",
            "acoustic branch",
            "generic operator-resolution estimate",
        ],
        panels=[
            ("a", "Angular frequency versus spatial harmonic."),
            ("b", "RK4 interval position for candidate microsteps."),
        ],
        metadata={
            "n_phi": 0,
            "k": 0,
            "device": "CPU FP64/complex128",
            "status": "Discrete spectra plus a separate generic estimate",
        },
    )
    fig2 = figure_record(
        2,
        FIGURE_NAMES[1],
        "High-mode micro response and full-horizon error",
        (
            "The VER badge means a controlled numerical comparison, not a new "
            "nonlinear acceptance result. Panel (a) shows the actual and supplied "
            "micro-run reference real components only through 0.3T. Panel (b) "
            "uses the complete supplied vector-error horizon so coincident "
            "waveforms cannot hide a discrepancy. Frozen controls are distinct "
            "from full synchronous mixed-state sampling."
        ),
        "verification",
        panels=[
            ("a", "Real component over the visible first 0.3T."),
            ("b", "Relative vector error over the full supplied horizon."),
        ],
        metadata={
            "device": "CPU FP64/complex128",
            "status": "Micro controls; the first 0.3T is a visual zoom only",
        },
    )
    fig3 = figure_record(
        3,
        FIGURE_NAMES[2],
        "Completed target-mode accuracy versus H/T",
        (
            "Only completed target-seed endpoints from the compact data are "
            "shown; rejected cases retain no zero or invented final metric. "
            "Predictor and synchronous coupling use different markers, and h* "
            "is separated from h*/2. Horizontal lines are the criteria fixed "
            "before the scan. VER denotes verification metrics rather than a "
            "stability guarantee."
        ),
        "verification",
        panels=[
            ("a", "Maximum target vector error."),
            ("b", "Maximum absolute target phase error."),
            ("c", "Maximum target amplitude drift."),
        ],
        metadata={
            "device": "CPU FP64/complex128",
            "status": f"{completed} completed cases; {rejected} rejected windows",
        },
    )
    fig4 = figure_record(
        4,
        FIGURE_NAMES[3],
        "Absolute seeded high-mode norm in nonlinear histories",
        (
            "The NL badge denotes accepted nonlinear time histories, not a "
            "linear-map reconstruction. Plotted values use the supplied absolute "
            "high-mode norm, avoiding a near-zero target-only denominator. "
            "Rejected runs stop at their last accepted sample and carry distinct "
            "markers. Linear interval predictions appear separately in Figure 7."
        ),
        "nonlinear_evolution",
        panels=[
            ("a", "Absolute high-mode norm versus accepted t/T."),
            ("b", "Small-M normalized gain and the fixed 1% criterion."),
        ],
        metadata={
            "device": "CPU FP64/complex128",
            "status": "Completed histories and accepted prefixes of rejected cases",
        },
    )
    fig5 = figure_record(
        5,
        FIGURE_NAMES[4],
        "Fixed-H microstep refinement at T/16 and T/8",
        (
            "Both fixed macro intervals compare h* with h*/2 using the same "
            "predeclared criteria. Annotations give the supplied single-run "
            "elapsed time. Rejected windows are not plotted as final points; "
            "their accepted-prefix horizon remains in data.json. This does not "
            "turn different accepted prefixes into a like-for-like final metric."
        ),
        "verification",
        panels=[
            ("a", "Maximum target vector error."),
            ("b", "Maximum absolute target phase error."),
            ("c", "Maximum target amplitude drift."),
        ],
        metadata={
            "device": "CPU FP64/complex128",
            "status": "Fixed-H completed points; completion context in labels",
        },
    )
    fig6 = figure_record(
        6,
        FIGURE_NAMES[5],
        "Energy and kinetic raw residuals against the unchanged budget",
        (
            "The NL badge identifies actual nonlinear histories. Plotted curves "
            "divide the absolute raw residual at accepted samples by the "
            "unchanged supplied energy budget; exact residuals remain in "
            "data.json, while exact zeros are not drawn on the logarithmic axes. "
            "The unit line marks the budget, not a rescaled tolerance. Cross "
            "markers are rejected candidates at their recorded rejection times "
            "and do not imply accepted endpoints."
        ),
        "nonlinear_evolution",
        panels=[
            ("a", "Total-energy residual magnitude relative to budget."),
            ("b", "Kinetic-energy residual magnitude relative to budget."),
        ],
        metadata={
            "device": "CPU FP64/complex128",
            "status": "Accepted prefixes plus distinct rejected candidates",
        },
    )
    fig7 = figure_record(
        7,
        FIGURE_NAMES[6],
        "Linear interval maps, observed qualification, and run cost",
        (
            "Panels (a) and (b) are LIN map predictions and are not nonlinear "
            "observations or independent references. Panels (c) and (d) "
            "summarize the supplied nonlinear run records and costs. A rejected "
            "requested window is marked undetermined. Elapsed time includes "
            "diagnostics and recording, so it is not an equal-accuracy speedup; "
            "memory is the supplied allocation estimate, not measured peak."
        ),
        "mixed_comparison",
        components=[
            "LIN FOMHD macro-map radius and seed gain",
            "nonlinear completion and accuracy qualification",
            "diagnostic-inclusive cost observations",
        ],
        panels=[
            ("a", "LIN one-step radius and radius over 2T."),
            ("b", "LIN maximum and final seed gain."),
            ("c", "Accuracy, amplification flag, and hard-check qualification."),
            ("d", "Elapsed time, model calls, and estimated peak bytes."),
        ],
        metadata={
            "device": "CPU FP64/complex128",
            "status": "LIN predictions remain separate from nonlinear qualification",
        },
    )

    legacy_notes = []
    for row in legacy:
        metric = (
            ""
            if row["target_error"] is None
            else f" Supplied target error: {row['target_error']:.6g}."
        )
        legacy_notes.append(
            f"{row['label']} (microsteps per target period "
            f"{row['micro_per_period']}, M={row['M']}, H/T={row['H_over_T']:.6g}; "
            f"status {row['status']}). Warning: {row['warning']}.{metric}"
        )

    pages = [
        {
            "id": "scope-protocol",
            "title": "Scope and fixed protocol",
            "introduction": (
                "This gallery isolates a controlled periodic, low-order time "
                "algorithm baseline. It does not claim general fast-magnetosonic, "
                "closed-wall, production-scale, or all-parameter validation."
            ),
            "notes": [
                (
                    "Question: how does the controlled T0 macro-step baseline "
                    "compare with synchronous fine stepping for target accuracy, "
                    "seeded high-mode amplification, and unchanged hard checks?"
                ),
                (
                    f"Fixed: periodic {protocol['points']}-point geometry, "
                    f"toroidal K=0 and N_phi=1, target m={protocol['target_mode']}, "
                    f"high m={protocol['high_mode']}, target amplitude "
                    f"{protocol['target_amplitude']:.3g}, high seed "
                    f"{protocol['high_amplitude']:.3g}, and a "
                    f"{protocol['cycles']}T horizon."
                ),
                (
                    f"Scan: M=4, 8, 16, 32, 64, 128, 256 at a common "
                    f"h*=T/{protocol['micro_per_period']}, plus fixed-H "
                    "H=T/16 and H=T/8 half-microstep controls."
                ),
                (
                    "Reference structure: nonlinear accepted histories, linear "
                    "Jacobian/map predictions labeled LIN, and only independently "
                    "formulated calculations labeled REF."
                ),
                f"Predeclared criteria: {criteria}.",
                (
                    "Frozen-ion electron-field controls occupy T/2 and are "
                    "distinct from the full nonlinear T0 acceptance evolution."
                ),
            ],
        },
        {
            "id": "resolved-frequency",
            "title": "Resolved frequencies and microstep position",
            "introduction": (
                "The first figure keeps measured discrete branches, the acoustic "
                "branch, and the generic operator estimate visually separate."
            ),
            "notes": [
                (
                    "CAW, when used elsewhere in A0 material, means the fast "
                    "magnetosonic branch only; it is not a synonym for a Hall or "
                    "whistler branch."
                ),
                (
                    f"Maximum absolute entry difference across the two full-Jacobian difference "
                    "scales: "
                    f"{data['spectrum']['full_jacobian_epsilon_max_abs_difference_s_inv']:.4g} s^-1."
                ),
            ],
            "figures": [fig1],
        },
        {
            "id": "micro-response",
            "title": "Actual high-frequency micro controls",
            "introduction": (
                "Waveform agreement is shown together with a separate full-horizon "
                "error panel, so visual overlap is not mistaken for zero error."
            ),
            "notes": [
                (
                    "The waveform panel deliberately zooms the first 0.3T; the "
                    "reported vector-error panel uses every supplied control sample."
                ),
                (
                    "Frozen-ion controls consume production electron-field rates "
                    "without a coupled transaction. Full synchronous controls are "
                    "accepted nonlinear trajectories sampled at micro cadence."
                ),
            ] + [
                f"{item['label']}, {item['micro_per_period']} steps/T: "
                f"{item['status']}; {'completed' if item['completed'] else 'partial'}; "
                f"observed through {max(item['time_over_T'], default=0):.6g}T."
                for item in data["micro_controls"]
            ],
            "figures": [fig2],
        },
        {
            "id": "target-accuracy",
            "title": "Target-mode trajectory accuracy",
            "introduction": (
                "Completed target endpoints are compared with the criteria fixed "
                "before the scan. Hard-rejected requested windows remain "
                "undetermined and are excluded from these final points."
            ),
            "notes": [
                (
                    "Vector error, phase, and amplitude drift are evaluated "
                    "separately for every completed target-seed case."
                ),
                (
                    "A completed run that misses an accuracy criterion remains a "
                    "completed observed result; it is not relabeled as rejected."
                ),
                (
                    "The high m=4 macro phase can alias at H=T/16 and H=T/8 and "
                    "is not used to infer that mode's frequency."
                ),
            ],
            "figures": [fig3],
        },
        {
            "id": "high-mode",
            "title": "Seeded high-mode histories",
            "introduction": (
                "Absolute high-mode norms avoid an undefined near-zero ratio for "
                "target-only initial states and preserve accepted prefixes of "
                "rejected runs."
            ),
            "notes": [
                (
                    "The >1% growth criterion is a check flag. Transient gain, "
                    "cumulative growth, and final gain remain separate source "
                    "values."
                ),
                (
                    "Linear map predictions are not overlaid as if they were "
                    "nonlinear trajectory observations."
                ),
            ],
            "figures": [fig4],
        },
        {
            "id": "fixed-H-and-energy",
            "title": "Fixed-H refinement and raw energy residuals",
            "introduction": (
                "The fixed macro interval tests microstep refinement directly. "
                "The energy view retains the unchanged budget and marks rejected "
                "candidates separately."
            ),
            "notes": [
                (
                    "Series with different elapsed prefixes are not treated as "
                    "equivalent final 2T metrics."
                ),
                (
                    "The figure reports raw residual magnitudes relative to the "
                    "supplied budget; it does not redefine or widen a solver gate."
                ),
            ] + refinement_notes(data),
            "figures": [fig5, fig6],
        },
        {
            "id": "linear-qualification-cost",
            "title": "Linear maps, observed qualification, and cost",
            "introduction": (
                "LIN radius and seed-gain predictions answer a local linear "
                "question. They are placed beside, not combined with, nonlinear "
                "accuracy and hard-check outcomes."
            ),
            "notes": [
                (
                    "A spectral radius below one is not a finite-window nonlinear "
                    "stability guarantee; transient and repeated-map effects stay "
                    "visible in the supplied source fields."
                ),
                (
                    "Elapsed time is for one run and includes diagnostic and "
                    "recording work. It does not support an equal-accuracy speedup "
                    "claim."
                ),
                (
                    "Estimated peak bytes are not measured resident memory or a "
                    "measured process high-water mark."
                ),
            ],
            "figures": [fig7],
        },
        {
            "id": "legacy-and-next-question",
            "title": "Legacy warnings and the next bounded question",
            "introduction": (
                "Old 64/128-step warning points stay separate from the new main "
                "scan. Historical labels and their original warnings are not "
                "rewritten as new data."
            ),
            "notes": (
                [*legacy_notes]
                or ["No legacy warning rows were supplied."]
            )
            + [
                (
                    "Reusable baseline: exact compact-data, script, renderer, "
                    "style, and source-file hashes are recorded in provenance.json."
                ),
                (
                    "Next T1 question: at the same H, is T1 more accurate; and "
                    "does it enlarge the usable H range under these unchanged "
                    "criteria?"
                ),
                (
                    f"A0 presentation source is the canonical renderer at commit "
                    f"scope {data['source_commit']}; no future implementation is "
                    "claimed on this page."
                ),
            ],
        },
    ]

    return {
        "title": "A1 controlled T0 baseline: periodic temporal evidence",
        "introduction": (
            "This English gallery presents the recorded results for "
            "a controlled periodic temporal benchmark. It reuses the unchanged "
            "A0 renderer, page layout, evidence badges, and CSS. NL means nonlinear "
            "time evolution, LIN means a frozen production-Jacobian or macro-map "
            "calculation, REF is reserved for an independently formulated "
            "reference, and VER means verification. The report does not rerun the "
            "solver or modify any scientific threshold."
        ),
        "notation": [
            {"symbol": "T", "meaning": "Target-wave period fixed by the protocol."},
            {"symbol": "h*", "meaning": "Common microstep T/2048."},
            {"symbol": "H", "meaning": "Macro interval containing M microsteps."},
            {"symbol": "M", "meaning": "Microsteps per macro interval, H=Mh."},
            {"symbol": "K, N_phi", "meaning": "Toroidal cutoff K=0 and toroidal sample count 1."},
            {"symbol": "m", "meaning": "Spatial harmonic on the periodic 9-point ring."},
            {
                "symbol": "NL / LIN / REF / VER",
                "meaning": (
                    "Nonlinear evolution; frozen production Jacobian or map; "
                    "independent formulation; verification evidence."
                ),
            },
        ],
        "archive": {
            "path": "data.json",
            "label": "Copied compact plot data; provenance.json records input and output hashes",
        },
        "pages": pages,
        "conclusions": data["recommendation"],
        "known_issues": [
            {
                "path": "provenance.json",
                "label": "Exact input, script, renderer, style, and output hashes",
            }
        ],
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
    data_hash = sha256_file(data_path)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    data = _object(data, "data")
    protocol = validate_data(data)
    verified_inputs = verify_input_provenance(data, data_path)
    canonical_hashes = check_canonical_files(
        renderer_path,
        a0_html_path,
        style_paths,
    )
    builder_path = Path(__file__).resolve()
    builder_hash = sha256_file(builder_path)
    renderer = load_renderer(renderer_path)
    if not callable(getattr(renderer, "render_report", None)):
        raise InputError("canonical renderer does not expose render_report")

    output.mkdir(parents=True, exist_ok=False)
    old_mplconfig = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(prefix="a1-t0-matplotlib-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        try:
            render_figures(data, output)
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

    manifest = build_manifest(data)
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    provisional_provenance = {
        "schema_version": 1,
        "status": "rendering",
        "input": {"sha256": data_hash},
        "renderer": {"sha256": canonical_hashes["renderer"]},
    }
    provenance_path = output / "provenance.json"
    _write_json(provenance_path, provisional_provenance)

    html_path = output / "a1-benchmark.html"
    rendered = renderer.render_report(manifest, manifest_path, html_path)
    rendered = embed_svgs(rendered, output)
    _write_text(html_path, rendered)

    if sha256_file(renderer_path) != canonical_hashes["renderer"]:
        raise RuntimeError("canonical A0 renderer changed during rendering")
    if sha256_file(data_path) != data_hash:
        raise RuntimeError("compact input changed during rendering")
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
        "source_commit": data["source_commit"],
        "scope": data["scope"],
        "compact_input": {
            "path": str(data_path.resolve()),
            "sha256": data_hash,
            "bytes": data_path.stat().st_size,
        },
        "verified_input_provenance": verified_inputs,
        "canonical_sources": {
            "renderer_path": str(renderer_path.resolve()),
            "renderer_sha256": canonical_hashes["renderer"],
            "a0_html_path": str(a0_html_path.resolve()),
            "a0_html_sha256": canonical_hashes["a0_html"],
            "plot_tokamak_cae_path": str(style_paths[0].resolve()),
            "plot_tokamak_cae_sha256": canonical_hashes["plot_tokamak_cae"],
            "plot_cae_comparison_path": str(style_paths[1].resolve()),
            "plot_cae_comparison_sha256": canonical_hashes[
                "plot_cae_comparison"
            ],
        },
        "builder": {
            "path": str(builder_path),
            "sha256": builder_hash,
        },
        "render_policy": {
            "renderer_reused_unchanged": True,
            "css_matches_a0_byte_for_byte": True,
            "figures": len(FIGURE_NAMES),
            "png_dpi": 160,
            "svg_fonttype": "none",
            "source_runs_executed": False,
            "runtime_network_used": False,
            "solver_used": False,
            "scientific_thresholds_modified": False,
        },
        "protocol_snapshot": protocol,
        "copied_data": output_record(output / "data.json", output),
        "outputs": [output_record(path, output) for path in hashed_outputs],
    }
    _write_json(provenance_path, provenance)
    return {
        "output": str(output),
        "html_bytes": html_path.stat().st_size,
        "figures": len(FIGURE_NAMES),
        "pages": len(manifest["pages"]),
        "verified_inputs": len(verified_inputs),
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
