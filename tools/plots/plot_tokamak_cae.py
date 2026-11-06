#!/usr/bin/env python3
"""Render archived Tokamak CAE response cases without running a solver."""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import textwrap
import tomllib

import numpy as np


GALLERY_ID = "cae-20261106"
EVOLUTION_IDENTITY = "NL | FOMHD time evolution"
EVOLUTION_ACCENT = "#00796B"
PHASES = 2 * np.pi * np.arange(5) / 5
CASE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SEMANTIC_RE = re.compile(
    r"^(?P<kind>circle|east)(?P<nodes>\d+)-n(?P<mode>[01])-"
    r"(?P<variant>.+)-r(?P<revision>\d+)$"
)
ENERGY_KEYS = ("ion_J", "electron_J", "field_J", "total_J")
ENERGY_LABELS = {
    "ion_J": "Ion total energy (kinetic+internal)",
    "electron_J": "Electron energy",
    "field_J": "Field energy",
    "total_J": "Total energy",
    "balance_J": "Energy balance",
    "outward_J": "Outward energy",
    "source_J": "Source energy",
}
ENERGY_COLORS = {
    "ion_J": "#0072B2",
    "electron_J": "#D55E00",
    "field_J": "#009E73",
    "total_J": "#111111",
    "balance_J": "#CC79A7",
    "outward_J": "#7F7F7F",
    "source_J": "#E69F00",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def select_default_cases(run_root: Path) -> list[str]:
    """Select the greatest revision for each semantic kind/grid/harmonic."""
    selected: dict[tuple[str, int, int], tuple[int, str]] = {}
    kind_order = {"circle": 0, "east": 1}
    for path in run_root.iterdir():
        match = SEMANTIC_RE.fullmatch(path.name)
        if match is None or not path.is_dir():
            continue
        if not (path / "report.json").is_file():
            continue
        key = (
            match.group("kind"),
            int(match.group("nodes")),
            int(match.group("mode")),
        )
        candidate = (
            int(match.group("revision")),
            path.name,
        )
        current = selected.get(key)
        if current is None or candidate > current:
            selected[key] = candidate
    return [
        name for _, (_, name) in sorted(
            selected.items(),
            key=lambda item: (
                kind_order[item[0][0]],
                item[0][1],
                item[0][2],
                item[1][1],
            ),
        )
    ]


def mode_history(values, mode: int):
    """Return a retained toroidal coefficient without dropping n_phi=0."""
    array = np.asarray(values)
    if array.ndim < 2 or array.shape[1] != 5:
        raise ValueError("Signal must have exactly five toroidal samples")
    if mode not in (0, 1):
        raise ValueError(
            "Only retained harmonics n_phi=0 or n_phi=1 are supported"
        )
    if mode == 0:
        zero = np.mean(array, axis=1)
        if np.iscomplexobj(zero):
            scale = max(float(np.max(np.abs(zero), initial=0.0)), 1.0)
            if float(np.max(np.abs(zero.imag), initial=0.0)) > 1e-10 * scale:
                raise ValueError(
                    "n_phi=0 zero-Fourier mean has a non-negligible "
                    "imaginary part"
                )
            return zero.real
        return zero
    weights = np.exp(-1j * mode * PHASES)
    coefficient = np.tensordot(array, weights, axes=([1], [0])) / 5
    return np.asarray(coefficient)


def physical_at_phi(coefficient, mode: int, phi: float = 0.0):
    """Reconstruct the real physical field from its retained coefficient."""
    if mode not in (0, 1):
        raise ValueError(
            "Only retained harmonics n_phi=0 or n_phi=1 are supported"
        )
    if not math.isfinite(float(phi)):
        raise ValueError("phi must be finite")
    coefficient = np.asarray(coefficient)
    if mode == 0:
        return np.real(coefficient)
    return 2 * np.real(coefficient * np.exp(1j * mode * float(phi)))


def _read_npz(data: bytes, path: str) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }


def _array_summary(arrays: dict[str, np.ndarray]) -> dict:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(
                np.all(np.isfinite(value))
                if value.dtype.kind in "fciub"
                else True
            ),
        }
        for name, value in sorted(arrays.items())
    }


def _cell_count(arrays: dict) -> int:
    candidates = []
    mapping = arrays.get("input_mapping.npz")
    if mapping is not None and "R" in mapping:
        candidates.append(("input R", int(np.asarray(mapping["R"]).size)))
    matched = arrays.get("matched-response.npz")
    if matched is not None and "delta_B" in matched:
        candidates.append(("delta_B", int(matched["delta_B"].shape[-1])))
    if matched is not None and "delta_n" in matched:
        candidates.append(("delta_n", int(matched["delta_n"].shape[-1])))
    for relative in ("control-signals.npz", "perturbed-signals.npz"):
        signal = arrays.get(relative)
        if signal is None:
            continue
        for key, divisor in (
            ("density", 5),
            ("BN", 15),
            ("velocity", 15),
        ):
            if key not in signal or np.asarray(signal[key]).ndim != 2:
                continue
            width = np.asarray(signal[key]).shape[1]
            if width % divisor == 0:
                candidates.append((f"{relative}:{key}", width // divisor))
    if not candidates:
        raise ValueError("cannot infer active cell count from supplied arrays")
    counts = {count for _, count in candidates}
    if len(counts) != 1 or next(iter(counts)) <= 0:
        raise ValueError(f"cell-count disagreement: {candidates}")
    return next(iter(counts))


def _reshape_nodal_signals(arrays: dict) -> dict:
    """Normalize documented flat nodal rows to time, phase, component, cell."""
    cells = _cell_count(arrays)
    layouts = {}
    for relative in ("control-signals.npz", "perturbed-signals.npz"):
        signal = arrays.get(relative)
        if signal is None:
            continue
        for key, samples, components in (
            ("density", 5, 1),
            ("BN", 5, 3),
            ("velocity", 5, 3),
        ):
            if key not in signal:
                continue
            array = np.asarray(signal[key])
            expected = (samples, components, cells)
            if array.ndim == 2:
                width = samples * components * cells
                if array.shape[1] != width:
                    raise ValueError(
                        f"{relative}:{key} flat width {array.shape[1]} != "
                        f"{width}"
                    )
                normalized = array.reshape((array.shape[0], *expected))
            elif array.ndim == 4 and array.shape[1:] == expected:
                normalized = array
            elif components == 1 and array.ndim == 3 and array.shape[1:] == (
                samples, cells
            ):
                normalized = array[:, :, None, :]
            else:
                raise ValueError(
                    f"{relative}:{key} shape {array.shape} does not match "
                    f"cells={cells}"
                )
            signal[key] = normalized
            layouts[f"{relative}:{key}"] = {
                "original_shape": list(array.shape),
                "normalized_shape": list(normalized.shape),
            }
        if "density" in signal:
            signal["density"] = np.asarray(signal["density"])[:, :, 0, :]
    return {"cells": cells, "layouts": layouts}


def _parsed_history(value, label: str) -> dict:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty JSON list")
    records = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"{label}[{index}] is not an object")
        time_s = _finite(row.get("time_s"))
        if time_s is None:
            raise ValueError(f"{label}[{index}] has no finite time_s")
        energy = row.get("energy")
        if not isinstance(energy, dict):
            raise ValueError(f"{label}[{index}] has no energy object")
        records.append({
            "time_s": time_s,
            "step": row.get("step"),
            "energy": {
                key: _finite(energy[key])
                for key in (
                    *ENERGY_KEYS,
                    "balance_J",
                    "relative_total_change",
                    "outward_J",
                    "source_J",
                )
            },
        })
    times = np.asarray([row["time_s"] for row in records])
    if np.any(np.diff(times) < 0):
        raise ValueError(f"{label} times are not ordered")
    return {"records": records, "time_s": times}


def _frequency_summary(report: dict) -> dict:
    fits = report.get("mode_fits", [])
    if not isinstance(fits, list):
        return {
            "available": False,
            "unavailable_reasons": ["mode_fits is not a list"],
            "fits": [],
        }
    available = []
    reasons = []
    compact = []
    for row in fits:
        if not isinstance(row, dict):
            reasons.append("invalid mode-fit record")
            continue
        fit = row.get("fit")
        fit = fit if isinstance(fit, dict) else {}
        available_flag = fit.get("available") is True
        if not available_flag:
            reason = str(fit.get("reason") or "fit marked unavailable")
            reasons.append(reason)
            compact.append({
                "cell": row.get("cell"),
                "available": False,
                "reason": reason,
            })
            continue
        frequency_hz = _finite(fit.get("frequency_hz"))
        omega = _finite(fit.get("omega_rad_s"))
        if frequency_hz is None and omega is not None:
            frequency_hz = omega / (2 * math.pi)
        if frequency_hz is None:
            reasons.append("available fit has no finite frequency")
            compact.append({
                "cell": row.get("cell"),
                "available": False,
                "reason": "available fit has no finite frequency",
            })
            continue
        entry = {
            "cell": row.get("cell"),
            "R_m": _finite(row.get("R_m")),
            "Z_m": _finite(row.get("Z_m")),
            "available": True,
            "frequency_hz": frequency_hz,
            "omega_rad_s": omega,
            "growth_rate_per_s": _finite(fit.get("growth_rate_per_s")),
            "sample_count": fit.get("sample_count"),
            "interval_count": fit.get("interval_count"),
            "measured_cycles": _finite(
                fit.get("complete_measured_cycles",
                        fit.get("measured_cycles"))
            ),
            "samples_per_cycle": _finite(fit.get("samples_per_cycle")),
        }
        available.append(entry)
        compact.append(entry)
    return {
        "available": bool(available),
        "selected": available[0] if available else None,
        "unavailable_reasons": sorted(set(reasons)),
        "fits": compact,
    }


def _compression_selection(arrays: dict, signals: dict) -> dict:
    matched = arrays.get("matched-response.npz")
    control = signals.get("control-signals.npz")
    if matched is None:
        return {
            "available": False,
            "reason": "matched-response.npz is unavailable",
        }
    for name in (
        "time_s", "delta_n", "probe_cells",
    ):
        if name not in matched:
            return {
                "available": False,
                "reason": f"matched-response.npz lacks {name}",
            }
    try:
        mode = signals.get("mode")
        if mode is None:
            return {
                "available": False,
                "reason": "physical toroidal mode is unavailable",
            }
        for signal in (
            control,
            signals.get("perturbed-signals.npz"),
        ):
            if signal is None or "density" not in signal:
                return {
                    "available": False,
                    "reason": "control or perturbed ion-density signal is unavailable",
                }
        coefficient = mode_history(matched["delta_n"], int(mode))
        control_density = np.asarray(control["density"])
        if control_density.shape[0] == 0:
            return {
                "available": False,
                "reason": "control ion density has no initial sample",
            }
        baseline = np.abs(mode_history(control_density, 0)[0, :])
        baseline = baseline[np.isfinite(baseline) & (baseline > 0)]
        if not baseline.size:
            return {
                "available": False,
                "reason": "initial control ion-density scale is zero or nonfinite",
            }
        scale = float(np.median(baseline))
        probe_cells = np.asarray(matched["probe_cells"], dtype=int)
        cells = np.asarray(matched["delta_n"]).shape[-1]
        if probe_cells.ndim != 1 or not probe_cells.size:
            return {
                "available": False,
                "reason": "probe_cells is empty or not one-dimensional",
            }
        if np.any(probe_cells < 0) or np.any(probe_cells >= cells):
            return {
                "available": False,
                "reason": "probe_cells contains an out-of-range cell",
            }
        if coefficient.shape[-1] != cells:
            return {
                "available": False,
                "reason": "ion-density coefficient and probe count disagree",
            }
        probe_values = coefficient[:, probe_cells]
        magnitude = np.abs(probe_values)
        if not np.any(np.isfinite(magnitude)):
            return {
                "available": False,
                "reason": "compressional probe signal is nonfinite",
            }
        flat = int(np.nanargmax(magnitude))
        time_index, probe_index = np.unravel_index(flat, magnitude.shape)
        probe_cell = int(probe_cells[probe_index])
        coefficient_value = probe_values[time_index, probe_index]
        physical_value = physical_at_phi(
            coefficient_value,
            int(mode),
            phi=0.0,
        )
        time_s = float(np.asarray(matched["time_s"], dtype=float)[time_index])
        result = {
            "available": True,
            "time_index": int(time_index),
            "time_s": time_s,
            "probe_cell": probe_cell,
            "probe_index": int(probe_index),
            "scale_m-3": scale,
            "scale_definition": (
                "median initial control number density across cell centers"
            ),
            "compressional_fraction": float(abs(coefficient_value) / scale),
            "compressional_fraction_definition": (
                "retained matched-density coefficient magnitude / "
                "initial number-density scale"
            ),
            "delta_n_phi0_m-3": float(physical_value),
            "delta_n_phi0_fraction": float(abs(physical_value) / scale),
        }
        input_mapping = arrays.get("input_mapping.npz")
        if input_mapping is not None and {
            "R", "Z"
        } <= set(input_mapping):
            R = np.asarray(input_mapping["R"])
            Z = np.asarray(input_mapping["Z"])
            if R.shape == Z.shape == (cells,):
                result["probe_R_m"] = float(R[probe_cell])
                result["probe_Z_m"] = float(Z[probe_cell])
        return result
    except (TypeError, ValueError, IndexError) as error:
        return {
            "available": False,
            "reason": f"compressional selection failed: {error}",
        }


def _case_metadata(report: dict, config: dict, name: str) -> dict:
    case = report.get("case")
    case = case if isinstance(case, dict) else {}
    name_match = re.search(r"-n([01])-", name)
    mode = config.get("nphi", case.get("nphi"))
    if mode is None and name_match:
        mode = int(name_match.group(1))
    source_hashes = report.get("source_hashes")
    source_hashes = source_hashes if isinstance(source_hashes, dict) else {}
    source_bundle = None
    if source_hashes:
        source_bundle = _sha256(
            json.dumps(
                source_hashes,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    equilibrium_source = case.get("sha256")
    return {
        "backend": report.get("device", "unavailable"),
        "kind": case.get("kind", config.get("kind")),
        "nodes": case.get("nodes", config.get("nodes")),
        "K": case.get("K", config.get("K")),
        "mode": mode,
        "Nphi": case.get("Nphi", config.get("Nphi")),
        "active_cells": case.get("active_cells"),
        "electron_inertia": case.get(
            "electron_inertia", config.get("electron_inertia")
        ),
        "profile_assumption": case.get("profile_assumption"),
        "convention": case.get("convention"),
        "wall": case.get("wall"),
        "shot": case.get("shot"),
        "equilibrium_time_s": _finite(case.get("time_s")),
        "psiN_wall": _finite(case.get("psiN_wall")),
        "source_url": case.get("source_url"),
        "equilibrium_source_sha256": equilibrium_source,
        "source_bundle_sha256": source_bundle,
        "source_hashes": source_hashes,
    }


def collect_case(run_root: Path, name: str) -> dict:
    """Read one case once, retaining missing and malformed evidence."""
    if CASE_RE.fullmatch(name) is None:
        raise ValueError(f"Invalid case directory name: {name!r}")
    case_dir = run_root / name
    result = {
        "name": name,
        "case_dir": str(case_dir),
        "inputs": [],
        "errors": [],
        "warnings": [],
        "solver_status": "unavailable",
        "_arrays": {},
    }
    if not case_dir.is_dir():
        result["errors"].append("case directory is missing")
        return result

    inputs = {}

    def read(relative: str, role: str) -> bytes | None:
        path = case_dir / relative
        if not path.is_file():
            result["warnings"].append(f"{relative} is missing")
            return None
        data = path.read_bytes()
        inputs[relative] = {
            "case": name,
            "path": relative,
            "role": role,
            "size_bytes": len(data),
            "sha256": _sha256(data),
        }
        return data

    report = None
    report_data = read("report.json", "report")
    if report_data is None:
        result["errors"].append("report.json is missing")
    else:
        try:
            report = json.loads(report_data)
            if not isinstance(report, dict):
                raise ValueError("top-level JSON is not an object")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            result["errors"].append(f"report.json is invalid: {error}")
            report = None
    if report is not None:
        result["solver_status"] = str(report.get("status", "unavailable"))
        failure = report.get("failure")
        result["failure"] = failure if isinstance(failure, dict) else None
        result["control"] = report.get("control")
        result["perturbed"] = report.get("perturbed")
        result["scope"] = report.get("scope")
        result["frequency"] = _frequency_summary(report)
    else:
        result["failure"] = None
        result["frequency"] = {
            "available": False,
            "selected": None,
            "unavailable_reasons": ["report.json is unavailable"],
            "fits": [],
        }

    config = {}
    config_data = read("case.toml", "configuration")
    if config_data is not None:
        try:
            config = tomllib.loads(config_data.decode())
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
            result["warnings"].append(f"case.toml is invalid: {error}")
    result.update(_case_metadata(
        report if report is not None else {},
        config,
        name,
    ))

    histories = {}
    for relative, key in (
        ("control-history.json", "control_history"),
        ("perturbed-history.json", "perturbed_history"),
    ):
        raw = read(relative, "history")
        if raw is None:
            continue
        try:
            histories[key] = _parsed_history(json.loads(raw), relative)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            result["warnings"].append(f"{relative} is invalid: {error}")

    arrays = {}
    for relative, role in (
        ("input_mapping.npz", "input mapping"),
        ("perturbation_mapping.npz", "seed mapping"),
        ("control-signals.npz", "control signals"),
        ("perturbed-signals.npz", "perturbed signals"),
        ("matched-response.npz", "matched response"),
    ):
        raw = read(relative, role)
        if raw is None:
            continue
        try:
            arrays[relative] = _read_npz(raw, relative)
        except (OSError, ValueError) as error:
            result["warnings"].append(f"{relative} is invalid: {error}")

    result["inputs"] = sorted(
        inputs.values(), key=lambda item: item["path"]
    )
    result["array_shapes"] = {
        relative: _array_summary(value)
        for relative, value in sorted(arrays.items())
    }
    try:
        result["signal_layout"] = _reshape_nodal_signals(arrays)
    except (IndexError, TypeError, ValueError) as error:
        result["signal_layout"] = {
            "cells": None,
            "layouts": {},
            "error": str(error),
        }
        result["warnings"].append(f"nodal signal layout: {error}")
    result["_arrays"] = arrays
    result["_histories"] = histories
    if "control_history" in histories:
        result["control_history"] = histories["control_history"]["records"]
    if "perturbed_history" in histories:
        result["perturbed_history"] = histories["perturbed_history"]["records"]
    signal_arrays = {
        "control-signals.npz": arrays.get("control-signals.npz"),
        "perturbed-signals.npz": arrays.get("perturbed-signals.npz"),
        "mode": result.get("mode"),
    }
    result["compression"] = _compression_selection(arrays, {
        key: value for key, value in signal_arrays.items() if value is not None
    })
    return result


def _energy(history: dict | None, key: str):
    if history is None:
        return None, None
    times = np.asarray(history["time_s"], dtype=float)
    values = np.asarray([
        np.nan if row["energy"].get(key) is None
        else float(row["energy"][key])
        for row in history["records"]
    ])
    return times, values


def _relative(values) -> np.ndarray:
    if values is None:
        return None
    result = np.full(values.shape, np.nan)
    if values.size and values[0] != 0 and np.isfinite(values[0]):
        result = (values - values[0]) / abs(values[0])
    return result


def _symlog(axis, values_list, zero_name: str = "zero") -> None:
    finite = np.concatenate([
        np.abs(np.asarray(values)[np.isfinite(values)])
        for values in values_list
        if values is not None and np.asarray(values).size
    ]) if values_list else np.array([])
    positive = finite[finite > 0]
    if positive.size:
        axis.set_yscale(
            "symlog",
            linthresh=max(float(np.min(positive)) * 10, 1e-300),
        )
    axis.axhline(0, color="0.65", linewidth=0.7, linestyle=":")
    axis.text(
        0.01, 0.02, f"Symlog; {zero_name} line shown",
        transform=axis.transAxes, fontsize=7.5, color="0.35",
    )


def _style_axis(axis) -> None:
    axis.grid(True, which="both", alpha=0.18)
    axis.tick_params(labelsize=7.5)


def _legend_if_present(axis, **kwargs) -> None:
    if axis.get_legend_handles_labels()[0]:
        axis.legend(**kwargs)


def _case_title(data: dict) -> str:
    nodes = data.get("nodes")
    grid = f"{nodes} x {nodes}" if nodes is not None else "unavailable"
    device = data.get("backend") or "unavailable"
    parts = [
        data["name"],
        f"device={device}",
        f"grid {grid}",
        f"K={data.get('K')}",
        f"n_phi={data.get('mode')}",
    ]
    return " | ".join(str(part) for part in parts)


def _evolution_banner(fig, data: dict) -> None:
    status = str(data.get("solver_status", "unavailable"))
    device = str(data.get("backend") or "unavailable")
    detail = f"device={device} | solver status={status}"
    if status != "completed":
        detail += " | no claim of completed evolution"
    banner = fig.text(
        0.008, 0.995, f"{EVOLUTION_IDENTITY}\n{detail}",
        ha="left", va="top", fontsize=8.5, fontweight="bold",
        color=EVOLUTION_ACCENT,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": EVOLUTION_ACCENT,
            "linewidth": 1.2,
        },
        zorder=100,
    )
    banner.set_gid("fomhd-nonlinear-banner")


def _figure_heading(
    fig,
    data: dict,
    subtitle: str,
    *,
    width: int = 145,
) -> None:
    _evolution_banner(fig, data)
    fig.suptitle(
        _case_title(data) + "\n" + textwrap.fill(subtitle, width=width),
        fontsize=11.5,
        y=0.955,
    )
    status = f"Solver status: {data['solver_status']}"
    if data.get("failure"):
        status += (
            " — "
            + str(data["failure"].get("code", "failure"))
            + ": "
            + str(data["failure"].get("message", ""))
        )
    fig.text(
        0.5, 0.005,
        status
        + ". Offline response figure; figure generation is not scientific acceptance.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#B2182B" if data["solver_status"] not in {
            "completed", "budget_stopped"
        } else "#444444",
    )


def _plot_mode_probe(axis, data: dict, component: str) -> None:
    histories = data.get("_histories", {})
    control = histories.get("control_history")
    perturbed = histories.get("perturbed_history")
    arrays = data.get("_arrays", {})
    matched = arrays.get("matched-response.npz")
    mode = data.get("mode")
    scale = data.get("compression", {}).get("scale_m-3")
    if (
        control is None
        or perturbed is None
        or matched is None
        or mode not in (0, 1)
        or scale is None
        or "delta_n" not in matched
        or "probe_cells" not in matched
    ):
        reason = data.get("compression", {}).get(
            "reason", "probe histories are unavailable"
        )
        axis.text(0.5, 0.5, reason, transform=axis.transAxes,
                  ha="center", va="center", fontsize=9)
        axis.set_axis_off()
        return
    signals = arrays.get("perturbed-signals.npz")
    control_signals = arrays.get("control-signals.npz")
    if (
        signals is None
        or control_signals is None
        or "density" not in signals
        or "density" not in control_signals
    ):
        axis.text(0.5, 0.5, "ion-density signal arrays are unavailable",
                  transform=axis.transAxes, ha="center", va="center")
        axis.set_axis_off()
        return
    try:
        original = mode_history(signals["density"], mode)
        changed = original - original[0]
        matched_density = mode_history(matched["delta_n"], mode)
    except (TypeError, ValueError) as error:
        axis.text(0.5, 0.5, str(error), transform=axis.transAxes,
                  ha="center", va="center")
        axis.set_axis_off()
        return
    probes = np.asarray(matched["probe_cells"], dtype=int)
    selected = data.get("compression", {}).get("probe_cell")
    t_original = np.asarray(signals.get("time_s", []), dtype=float)
    t_matched = np.asarray(matched["time_s"], dtype=float)
    for probe in probes:
        if probe < 0 or probe >= original.shape[-1]:
            continue
        strong = int(probe) == selected
        alpha = 1.0 if strong else 0.24
        width = 2.0 if strong else 0.8
        series = {
            "original": original[:, probe],
            "changed": changed[:, probe],
            "matched": matched_density[:, probe],
        }
        traces = [("-", series["original"], t_original),
                  ("--", series["changed"], t_original),
                  (":", series["matched"], t_matched)]
        if mode == 0:
            index = {"real": 0, "changed": 1, "matched": 2}[component]
            traces = traces[index:index+1]
        for style, values, times in traces:
            if values.size != times.size:
                continue
            plotted = (
                np.real(values) if component != "imag" else np.imag(values)
            ) / scale
            axis.plot(
                times * 1e9,
                plotted,
                linestyle=style,
                color="#0072B2" if strong else "0.4",
                alpha=alpha,
                linewidth=width,
                marker="." if strong else None,
            )
    if mode != 0 or component != "real":axis.axhline(0, color="0.7", linewidth=0.7)
    axis.set_xlabel("Time [ns]")
    if mode == 1:
        density_axis_label = (
            "retained ion-density coefficient (A/2) / initial n_i"
        )
    else:
        density_axis_label = {
            "real": "ion density n_i / initial n_i",
            "changed": "ion-density change delta n_i / initial n_i",
            "matched": "matched-control delta n_i / initial n_i",
        }[component]
    axis.set_ylabel(
        ("Imaginary " if component == "imag" else "Real ")
        + density_axis_label
    )
    axis.set_title(
        (f"n_phi=1 retained ion-density coefficient (A/2): {component}"
         if mode == 1
         else "n_phi=0 original ion density n_i (background retained)"
         if component == "real"
         else "n_phi=0 change delta n_i from perturbed initial"
         if component == "changed"
         else "n_phi=0 matched-control delta n_i (separate signal scale)"),
        fontsize=9,
    )
    if component != "imag":
        if component == "real":
            axis.plot([], [], "k-", label="Original perturbed ion density")
        if mode == 1 or component == "changed":
            axis.plot([], [], "k--", label="Change delta n_i from perturbed initial")
        if mode == 1 or component == "matched":
            axis.plot([], [], "k:", label="Matched-control delta n_i")
        axis.legend(fontsize=7, loc="best")
    _style_axis(axis)


def _render_overview(data: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 4, figsize=(19, 10))
    fig.subplots_adjust(
        left=0.04, right=0.985, bottom=0.075, top=0.86,
        wspace=0.22, hspace=0.32,
    )
    histories = data.get("_histories", {})
    control = histories.get("control_history")
    perturbed = histories.get("perturbed_history")
    arrays = data.get("_arrays", {})
    matched = arrays.get("matched-response.npz")

    axis = axes[0, 0]
    absolute_values = []
    for label, history, style in (
        ("control", control, "--"),
        ("perturbed", perturbed, "-"),
    ):
        if history is None:
            continue
        for key in ENERGY_KEYS:
            times, values = _energy(history, key)
            if values is None or not np.any(np.isfinite(values)):
                continue
            axis.plot(
                times * 1e9,
                values,
                style,
                color=ENERGY_COLORS[key],
                linewidth=2.0 if key == "total_J" else 1.0,
                label=f"{ENERGY_LABELS[key]} ({label})",
            )
            absolute_values.append(values)
    if absolute_values:
        _symlog(axis, absolute_values)
    axis.set(xlabel="Time [ns]", ylabel="Energy [J]",
             title="Absolute energy components: conservation scale")
    if axis.get_legend_handles_labels()[0]:
        _legend_if_present(axis, fontsize=6.4, ncol=1, loc="best")
    else:
        axis.text(0.5, 0.5, "energy histories unavailable",
                  transform=axis.transAxes, ha="center", va="center")
    _style_axis(axis)

    axis = axes[0, 1]
    relative_values = []
    for label, history, style in (
        ("control", control, "--"),
        ("perturbed", perturbed, "-"),
    ):
        if history is None:
            continue
        for key in ENERGY_KEYS[:-1]:
            times, values = _energy(history, key)
            relative = _relative(values)
            if relative is None or not np.any(np.isfinite(relative)):
                continue
            axis.plot(
                times * 1e9, relative, style,
                color=ENERGY_COLORS[key],
                linewidth=0.9,
                label=f"Δ{ENERGY_LABELS[key]} ({label})",
            )
            relative_values.append(relative)
        times, supplied = _energy(history, "relative_total_change")
        if supplied is not None and np.any(np.isfinite(supplied)):
            axis.plot(
                times * 1e9, supplied, style,
                color=ENERGY_COLORS["total_J"],
                linewidth=2.0,
                label=f"Supplied total relative change ({label})",
            )
            relative_values.append(supplied)
    if relative_values:
        _symlog(axis, relative_values)
    axis.set(
        xlabel="Time [ns]",
        ylabel="Change from each history's first baseline",
        title="Relative baseline changes",
    )
    _legend_if_present(axis, fontsize=6.1, ncol=1, loc="best")
    _style_axis(axis)

    axis = axes[0, 2]
    signal_values = []
    for label, history, style in (
        ("control", control, "--"),
        ("perturbed", perturbed, "-"),
    ):
        if history is None:
            continue
        for key in ("balance_J", "outward_J", "source_J"):
            times, values = _energy(history, key)
            if values is None or not np.any(np.isfinite(values)):
                continue
            axis.plot(
                times * 1e9, values, style,
                color=ENERGY_COLORS[key],
                linewidth=1.0,
                label=f"{ENERGY_LABELS[key]} ({label})",
            )
            signal_values.append(values)
    if matched is not None:
        matched_time = np.asarray(matched.get("time_s", []), dtype=float)
        for key, label, color, style in (
            ("perturbation_magnetic_J", "Matched perturbation magnetic",
             "#009E73", "-"),
            ("perturbation_ion_kinetic_J", "Matched perturbation ion kinetic",
             "#0072B2", ":"),
        ):
            if key in matched:
                values = np.asarray(matched[key], dtype=float)
                if values.size == matched_time.size:
                    axis.plot(
                        matched_time * 1e9, values, style,
                        color=color, linewidth=1.6, label=label,
                    )
                    signal_values.append(values)
    if signal_values:
        _symlog(axis, signal_values)
    axis.set(
        xlabel="Time [ns]", ylabel="Energy [J]",
        title="Balance versus matched perturbation energy: signal scale",
    )
    if axis.get_legend_handles_labels()[0]:
        _legend_if_present(axis, fontsize=6.3, loc="best")
    else:
        axis.text(0.5, 0.5, "matched perturbation comparison unavailable",
                  transform=axis.transAxes, ha="center", va="center")
    _style_axis(axis)

    mode = data.get("mode")
    if mode == 0:
        _plot_mode_probe(axes[0, 3], data, "real")
        _plot_mode_probe(axes[1, 0], data, "changed")
    else:
        _plot_mode_probe(axes[0, 3], data, "real")
        _plot_mode_probe(axes[1, 0], data, "imag")

    frequency = data.get("frequency", {})
    axis = axes[1, 1]
    if mode != 0:axis.set_axis_off()
    lines = [
        "Frequency evidence",
        "",
        "Measured frequency:",
    ]
    if frequency.get("available") and frequency.get("selected"):
        selected = frequency["selected"]
        lines.extend([
            f"  {selected['frequency_hz']:.6g} Hz",
            f"  cell {selected.get('cell')}",
            f"  measured cycles {selected.get('measured_cycles')}",
            "Source: report.mode_fits with available=true",
        ])
    else:
        lines.append("  UNAVAILABLE")
        reasons = frequency.get("unavailable_reasons", [])
        lines.extend(f"  {reason}" for reason in reasons[:5])
        lines.append("No numeric pilot frequency is inferred.")
    if mode == 0:
        _plot_mode_probe(axis, data, "matched")
    else:
        axis.text(0.02, 0.97, "\n".join(lines), va="top", ha="left",
                  fontsize=8.5, family="monospace", transform=axis.transAxes)

    axis = axes[1, 2]
    axis.set_axis_off()
    compression = data.get("compression", {})
    profile = textwrap.fill(str(data.get("profile_assumption")), width=54)
    lines = [
        "Evidence ledger",
        "",
        f"Source bundle: {str(data.get('source_bundle_sha256'))[:16]}",
        f"Equilibrium source: "
        f"{str(data.get('equilibrium_source_sha256'))[:16]}",
        f"Profile: {profile}",
    ]
    if compression.get("available"):
        lines.extend([
            "",
            "Strong compressional ion-density signal:",
            f"  t = {compression['time_s']:.12g} s",
            f"  probe = {compression['probe_cell']}",
            f"  scale = {compression['scale_m-3']:.12g} m^-3",
            f"  retained coefficient fraction = "
            f"{compression['compressional_fraction']:.6g}",
        ])
    else:
        lines.extend([
            "",
            "Compressional time unavailable:",
            f"  {compression.get('reason')}",
        ])
    if data.get("warnings"):
        lines.extend(["", "Unavailable inputs:"])
        lines.extend(
            textwrap.fill(f"  {item}", width=58)
            for item in data["warnings"][:5]
        )
    axis.text(0.01, 0.98, "\n".join(lines), va="top", ha="left",
              fontsize=7.3, family="monospace",
              transform=axis.transAxes, wrap=True)

    axes[1, 3].set_axis_off()
    axes[1, 3].text(
        0.02, 0.98,
        "Field meanings\n\n"
        "Solid: perturbed original ion density\n"
        "Dashed: delta n_i from perturbed initial\n"
        "Dotted: matched-control delta n_i\n\n"
        "n_phi=0 uses the five-phase zero-Fourier mean.\n"
        "n_phi=1 real/imaginary use the retained\n"
        "toroidal Fourier coefficient (A/2).\n\n"
        "Histories remain separate; no interpolation\n"
        "or planned-period frequency is introduced.",
        va="top", ha="left", fontsize=8.5, transform=axes[1, 3].transAxes,
    )
    _figure_heading(
        fig,
        data,
        "Energy, balance, probe histories, and evidence availability",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=170, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def _matched_shape(data: dict):
    arrays = data.get("_arrays", {})
    matched = arrays.get("matched-response.npz")
    compression = data.get("compression", {})
    mode = data.get("mode")
    if matched is None or not compression.get("available"):
        return None
    if mode not in (0, 1):
        return None
    try:
        density_coefficient = mode_history(matched["delta_n"], mode)
        velocity_coefficient = mode_history(matched["delta_velocity"], mode)
        return {
            "delta_B_parallel": physical_at_phi(
                matched["parallel_B"], mode, phi=0.0
            ),
            "delta_n": physical_at_phi(
                density_coefficient, mode, phi=0.0
            ),
            "delta_velocity": physical_at_phi(
                velocity_coefficient, mode, phi=0.0
            ),
            "time_s": np.asarray(matched["time_s"], dtype=float),
            "probe_cells": np.asarray(matched["probe_cells"], dtype=int),
            "time_index": int(compression["time_index"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _plot_cells(
    axis,
    R,
    Z,
    polygon,
    values,
    label,
    cmap="viridis",
    *,
    signed=False,
):
    values = np.asarray(values)
    if values.shape != R.shape:
        raise ValueError("cell values and coordinates disagree")
    finite = np.isfinite(values)
    if not np.any(finite):
        axis.text(0.5, 0.5, "No finite nodal values",
                  transform=axis.transAxes, ha="center", va="center")
        return
    vmax = float(np.max(np.abs(values[finite])))
    if not math.isfinite(vmax) or vmax <= 0:
        image = axis.scatter(R, Z, c="0.7", marker="s", s=55,
                             edgecolor="0.2", linewidth=0.35)
        axis.text(0.5, 0.05, "Exact zero field",
                  transform=axis.transAxes, ha="center", va="bottom")
    else:
        image = axis.scatter(
            R, Z, c=values, cmap=cmap,
            vmin=-vmax if signed else 0,
            vmax=vmax,
            marker="s", s=55, edgecolor="0.25", linewidth=0.35,
        )
    closed = np.vstack([polygon, polygon[0]]) if polygon.shape[0] else polygon
    axis.plot(closed[:, 0], closed[:, 1], "--", color="#D55E00",
              linewidth=1.3, label="Input wall polygon")
    axis.scatter(R, Z, s=8, color="white", edgecolor="0.2", linewidth=0.25)
    axis.set_aspect("equal", adjustable="box")
    axis.set(xlabel="R [m]", ylabel="Z [m]", title=label)
    axis.legend(fontsize=7, loc="best")
    axis.text(
        0.01, 0.01,
        "Cell-center scatter; solver boundary is a staircase. No interpolation.",
        transform=axis.transAxes, fontsize=6.8, color="0.35", va="bottom",
    )
    if vmax > 0:
        axis.figure.colorbar(image, ax=axis, shrink=0.78, label=label)


def _render_structure(data: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.subplots_adjust(
        left=0.065, right=0.96, bottom=0.065, top=0.85,
        wspace=0.20, hspace=0.22,
    )
    compression = data.get("compression", {})
    arrays = data.get("_arrays", {})
    mapping = arrays.get("input_mapping.npz")
    shape = _matched_shape(data)
    if (
        mapping is None
        or "R" not in mapping
        or "Z" not in mapping
        or "polygon" not in mapping
    ):
        reason = "input_mapping.npz lacks R, Z, or polygon"
    elif shape is None:
        reason = compression.get(
            "reason", "matched node-aligned perturbation arrays are unavailable"
        )
    else:
        R = np.asarray(mapping["R"], dtype=float)
        Z = np.asarray(mapping["Z"], dtype=float)
        polygon = np.asarray(mapping["polygon"], dtype=float)
        index = shape["time_index"]
        if (
            R.ndim != 1
            or Z.shape != R.shape
            or polygon.ndim != 2
            or polygon.shape[1:] != (2,)
            or shape["delta_B_parallel"].shape[-1] != R.size
        ):
            reason = "matched arrays are not aligned to the stored cell count"
        else:
            reason = None
            b_values = shape["delta_B_parallel"][index]
            n_values = shape["delta_n"][index]
            velocity_index = int(np.argmax(np.max(np.linalg.norm(shape["delta_velocity"], axis=1), axis=1)))
            velocity_time_ns = float(shape["time_s"][velocity_index] * 1e9)
            v_values = np.sqrt(
                np.sum(np.abs(shape["delta_velocity"][velocity_index]) ** 2, axis=0)
            )
            _plot_cells(
                axes[0, 0], R, Z, polygon, b_values,
                r"Signed physical $\delta B_\parallel(\phi=0)$ [T]",
                "RdBu_r", signed=True,
            )
            _plot_cells(
                axes[0, 1], R, Z, polygon, n_values,
                r"Signed physical ion density $\delta n_i(\phi=0)$ [m$^{-3}$]",
                "RdBu_r", signed=True,
            )
            _plot_cells(
                axes[1, 0], R, Z, polygon, v_values,
                r"Physical ion velocity $|\delta v_i(\phi=0)|$ [m/s]", "magma",
            )
            axes[1, 0].set_title(
                rf"Ion velocity magnitude $|\delta v_i(\phi=0)|$, "
                f"t={velocity_time_ns:.5g} ns"
            )
            axis = axes[1, 1]
            vectors = np.asarray(shape["delta_velocity"][velocity_index])
            vr = vectors[0]
            vz = vectors[2]
            lengths = np.hypot(vr, vz)
            keep = np.isfinite(vr) & np.isfinite(vz) & (lengths > 0)
            if np.any(keep):
                normalized_r = np.zeros_like(vr)
                normalized_z = np.zeros_like(vz)
                normalized_r[keep] = vr[keep] / lengths[keep]
                normalized_z[keep] = vz[keep] / lengths[keep]
                image = axis.scatter(
                    R, Z, c=lengths, cmap="plasma", marker="s", s=55,
                    edgecolor="0.25", linewidth=0.3,
                )
                axis.quiver(
                    R[keep], Z[keep], normalized_r[keep], normalized_z[keep],
                    angles="xy", scale_units="xy", scale=8.0,
                    color="#111111", width=0.004,
                )
                fig.colorbar(image, ax=axis, shrink=0.78,
                             label=r"$|\delta v_i|$ [m/s]")
                axis.text(
                    0.01, 0.01,
                    "Directions normalized; color is amplitude.",
                    transform=axis.transAxes, fontsize=6.8,
                    color="0.35", va="bottom",
                )
            else:
                axis.text(0.5, 0.5, "Zero poloidal velocity polarization",
                          transform=axis.transAxes, ha="center", va="center")
            closed = np.vstack([polygon, polygon[0]])
            axis.plot(closed[:, 0], closed[:, 1], "--", color="#D55E00",
                      linewidth=1.3, label="Input wall polygon")
            axis.set_aspect("equal", adjustable="box")
            axis.set(
                xlabel="R [m]", ylabel="Z [m]",
                title=(
                    rf"Ion velocity polarization at $\phi=0$, "
                    f"t={velocity_time_ns:.5g} ns"
                ),
            )
            axis.legend(fontsize=7, loc="best")

            evidence_axis = axes.flat
            for axis in evidence_axis:
                probe = int(compression["probe_cell"])
                axis.scatter(
                    [R[probe]], [Z[probe]], s=180,
                    facecolors="none", edgecolors="#D55E00",
                    linewidths=2.0,
                )

    if reason is not None:
        for axis in axes.flat:
            axis.text(0.5, 0.5, reason, transform=axis.transAxes,
                      ha="center", va="center", fontsize=10, wrap=True)
            axis.set_axis_off()
    if compression.get("available"):
        detail = "Signed field and ion-velocity structure at phi=0"
    else:
        detail = "Unavailable compressional structure evidence"
    _figure_heading(fig, data, detail, width=86)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=170, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def _case_public(data: dict) -> dict:
    excluded = {"_arrays", "_histories", "inputs"}
    result = {
        key: _json_safe(value)
        for key, value in data.items()
        if key not in excluded
    }
    energy = {}
    for label in ("control_history", "perturbed_history"):
        history = data.get("_histories", {}).get({
            "control_history": "control_history",
            "perturbed_history": "perturbed_history",
        }[label])
        if history is None:
            energy[label] = None
            continue
        energy[label] = {
            "time_s": history["time_s"].tolist(),
            **{
                key: [
                    row["energy"].get(key)
                    for row in history["records"]
                ]
                for key in (
                    *ENERGY_KEYS,
                    "balance_J",
                    "relative_total_change",
                    "outward_J",
                    "source_J",
                )
            },
        }
    result["energy_series"] = energy
    result["field_meanings"] = {
        "original": "perturbed original field",
        "change_from_initial": "perturbed field minus its initial value",
        "matched_control_difference": "perturbed field minus control",
        "physical_structure": (
            "physical_at_phi: Re(c) for n_phi=0; "
            "2 Re(c exp(i*n_phi*phi)) for n_phi=1"
        ),
        "physical_structure_phi_rad": 0.0,
        "density_species": "ion",
        "velocity_species": "ion",
        "retained_amplitude": (
            "the retained n_phi=1 toroidal coefficient is A/2; "
            "the phi=0 physical field applies the factor 2"
        ),
    }
    result["figure_identity"] = {
        "label": EVOLUTION_IDENTITY,
        "accent": EVOLUTION_ACCENT,
        "device": data.get("backend", "unavailable"),
        "solver_status": data["solver_status"],
        "completed_evolution_claim": data["solver_status"] == "completed",
    }
    result["figure_captions"] = {
        "overview": (
            "Energy, balance, and ion-density probe histories; unavailable "
            "frequency evidence remains explicit."
        ),
        "structure": (
            "Signed physical delta B_parallel, signed ion density delta n_i, "
            "ion velocity magnitude, and ion velocity polarization at phi=0. "
            "Maps use node-aligned cell centers; the solver boundary is a "
            "staircase and no interpolation is applied. Staggered seed B is "
            "not plotted."
        ),
    }
    result["input_hashes"] = {
        item["path"]: item["sha256"]
        for item in data.get("inputs", [])
    }
    return result


def _fragment(cases: list[dict]) -> str:
    sections = []
    for data in cases:
        name = html.escape(data["name"])
        overview_png = f"overview/{data['name']}.png"
        overview_svg = f"overview/{data['name']}.svg"
        structure_png = f"structure/{data['name']}.png"
        structure_svg = f"structure/{data['name']}.svg"
        failure = data.get("failure")
        failure_text = (
            "None recorded"
            if not failure else
            html.escape(
                f"{failure.get('code', 'failure')}: "
                f"{failure.get('message', '')}"
            )
        )
        compression = data.get("compression", {})
        if compression.get("available"):
            compression_text = (
                f"Strong ion-density compression at t={compression['time_s']:.12g} s, "
                f"probe cell {compression['probe_cell']} "
                f"(R={compression.get('probe_R_m')}, "
                f"Z={compression.get('probe_Z_m')} m), "
                f"scale {compression['scale_m-3']:.12g} m^-3, "
                f"retained coefficient fraction "
                f"{compression['compressional_fraction']:.6g}, "
                f"delta n_i(phi=0) "
                f"{compression['delta_n_phi0_m-3']:.6g} m^-3."
            )
        else:
            compression_text = (
                "Strong ion-density compressional time unavailable: "
                + html.escape(str(compression.get("reason")))
            )
        frequency = data.get("frequency", {})
        if frequency.get("available") and frequency.get("selected"):
            selected = frequency["selected"]
            frequency_text = (
                f"Report mode-fit frequency {selected['frequency_hz']:.6g} Hz "
                f"at cell {selected.get('cell')}; measured cycles "
                f"{selected.get('measured_cycles')}."
            )
        else:
            reasons = "; ".join(
                html.escape(reason)
                for reason in frequency.get("unavailable_reasons", [])
            ) or "no available mode-fit record"
            frequency_text = "Measured frequency unavailable: " + reasons
        equilibrium = (
            f"Equilibrium source {html.escape(str(data.get('equilibrium_source_sha256')))}; "
            f"assumption {html.escape(str(data.get('profile_assumption')))}; "
            f"convention {html.escape(str(data.get('convention')))}."
        )
        errors = data.get("errors", [])
        warnings = data.get("warnings", [])
        missing = "; ".join(html.escape(item) for item in [*errors, *warnings])
        sections.append(f"""
<article id="{GALLERY_ID}-{name}" class="cae-case">
  <h3>{name}</h3>
  <p class="cae-status"><strong>Status:</strong> {html.escape(data['solver_status'])}
     | <strong>Backend:</strong> {html.escape(str(data.get('backend')))}
     | <strong>N_R x N_Z / K / n_phi:</strong>
     {html.escape(str(data.get('nodes')))} x {html.escape(str(data.get('nodes')))} /
     {html.escape(str(data.get('K')))} / {html.escape(str(data.get('mode')))}
     | <strong>source bundle:</strong>
     <code>{html.escape(str(data.get('source_bundle_sha256'))[:16])}</code></p>
  <p><strong>Failure:</strong> {failure_text}</p>
  <p><strong>Comparison:</strong> {compression_text} {frequency_text}</p>
  <p><strong>Source and assumptions:</strong> {equilibrium}</p>
  <p><strong>Energy panels:</strong> Ion total energy (kinetic+internal),
     electron energy, field energy, total energy, supplied relative total change,
     balance, outward and source energy are read separately for the control and
     perturbed histories.</p>
  <p><strong>Evidence notes:</strong> {missing or 'All expected artifact roles were read.'}</p>
  <div class="cae-figures">
    <figure><a href="{overview_svg}"><img src="{overview_png}"
      alt="Energy, balance, and mode-probe histories for {name}"></a>
      <figcaption>Overview: energy, balance, ion-density probe histories, and
      fit availability.</figcaption></figure>
    <figure><a href="{structure_svg}"><img src="{structure_png}"
      alt="Matched perturbation magnetic field, ion density, ion velocity,
      and polarization for {name}"></a>
      <figcaption>Structure: signed physical delta B-parallel and ion number
      density delta n_i, physical ion velocity magnitude, and ion velocity
      polarization at phi=0.
      Ion-density compression and ion-velocity panels use their separately
      labeled strongest times.</figcaption></figure>
  </div>
</article>""")
    return f"""<section id="{GALLERY_ID}" class="cae-gallery">
<style>
#{GALLERY_ID} {{ --cae-ink:#17212b; --cae-rule:#bac4ce; color:var(--cae-ink); }}
#{GALLERY_ID} .cae-case {{ border-top:1px solid var(--cae-rule); padding:1rem 0; }}
#{GALLERY_ID} .cae-status {{ font-variant-numeric: tabular-nums; }}
#{GALLERY_ID} .cae-figures {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:1rem; }}
#{GALLERY_ID} figure {{ margin:0; }}
#{GALLERY_ID} img {{ width:100%; height:auto; border:1px solid var(--cae-rule); }}
#{GALLERY_ID} figcaption {{ font-size:.9rem; }}
</style>
<h2>Tokamak CAE response cases</h2>
<p>NL | FOMHD time evolution. Offline plots from supplied histories and
node-aligned response arrays. Solid is the
perturbed original ion-density field, dashed is delta n_i from perturbed
initial, and dotted is matched-control delta n_i. The n_phi=0 history retains the five-phase
zero-Fourier mean. Spatial structure reconstructs physical fields at phi=0:
Re(c) for n_phi=0 and 2 Re(c exp(i*n_phi*phi)) for n_phi=1, where the retained
n_phi=1 coefficient is A/2.
Ion-density compression selects the structure time. Frequencies appear only when the
source report marks a mode fit available; planned periods are never used as
measurements. Figures are not final scientific acceptance.</p>
{''.join(sections)}
</section>
"""


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases",
        help="comma-separated direct child case directory names",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run root is not a directory: {run_root}")

    if args.cases is None:
        names = select_default_cases(run_root)
        selection_mode = "default_latest_semantic_revision"
    else:
        names = []
        for item in args.cases.split(","):
            name = item.strip()
            if not name:
                raise ValueError("--cases contains an empty directory name")
            if CASE_RE.fullmatch(name) is None:
                raise ValueError(f"Invalid case directory name: {name!r}")
            if name not in names:
                names.append(name)
        selection_mode = "explicit"
    if not names:
        raise ValueError(
            "No formal named cases were found; use --cases for explicit names"
        )

    cases = [collect_case(run_root, name) for name in names]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "overview").mkdir()
    (output / "structure").mkdir()
    (output / "receipts").mkdir()

    old_cache = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(prefix="fomhd-cae-mpl-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            for data in cases:
                _render_overview(
                    data, output / "overview" / data["name"]
                )
                _render_structure(
                    data, output / "structure" / data["name"]
                )
                data["figures"] = {
                    "overview_png": f"overview/{data['name']}.png",
                    "overview_svg": f"overview/{data['name']}.svg",
                    "structure_png": f"structure/{data['name']}.png",
                    "structure_svg": f"structure/{data['name']}.svg",
                }
        finally:
            if old_cache is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = old_cache

    fragment = _fragment(cases)
    (output / "fragment.html").write_text(fragment)
    inputs = sorted(
        [
            item
            for data in cases
            for item in data.get("inputs", [])
        ],
        key=lambda item: (item["case"], item["path"]),
    )
    receipt = {
        "schema": "fomhd-tokamak-cae-gallery-receipt-v1",
        "gallery_id": GALLERY_ID,
        "generation_status": "completed",
        "scientific_acceptance": False,
        "solver_executed": False,
        "scope": (
            "Offline visualization of supplied Tokamak response cases; "
            "not final scientific acceptance."
        ),
        "run_root": str(run_root),
        "output": str(output),
        "selection": {
            "mode": selection_mode,
            "cases": names,
            "default_policy": (
                "greatest -rN revision for each circle/east, grid-size, "
                "and n_phi=0/1 key; generic smoke names and superseded revisions "
                "are excluded"
            ),
        },
        "command": [str(Path(sys.argv[0]))] + (
            list(argv) if argv is not None else sys.argv[1:]
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "host_device_placement": "host files; CPU-only rendering",
        },
        "script_sha256": _sha256(Path(__file__).read_bytes()),
        "inputs": inputs,
        "input_count": len(inputs),
        "cases": [_case_public(data) for data in cases],
    }
    _write_json(output / "receipts/report.json", receipt)
    print(json.dumps({
        "output": str(output),
        "gallery_id": GALLERY_ID,
        "cases": names,
        "input_count": len(inputs),
        "figure_pairs": 2 * len(cases),
        "scientific_acceptance": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
