#!/usr/bin/env python3
"""Build the bounded 0.5.0 multirate gallery from reviewed run artifacts."""

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
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

for thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_variable, "1")

import numpy as np

sys.dont_write_bytecode = True

GALLERY_SCHEMA = "fomhd-050-multirate-gallery-v1"
MARKER_NAME = ".fomhd-050-multirate-gallery.json"
RUN_REPORT_SCHEMA = "fomhd-warp-050-experiment-v1"
INITIAL_GUESS_ROOTS = (
    "initial-guesses-cpu",
    "initial-guesses-cpu-e3",
    "initial-guesses-gpu-fixed",
)
CANDIDATE_ROOTS = ("candidates-cpu-v2", "candidates-gpu")
LEGACY_CANDIDATE_ROOT = "candidates-cpu"
FAIR_COST_ROOTS = (
    ("fair-cost-cpu", "CPU", "cpu"),
    ("fair-cost-gpu", "GPU", "cuda:0"),
    ("fair-cost-gpu-fixed", "GPU", "cuda:0"),
)
MATRIX_FILE = "multirate-matrix.json"
CANONICAL_FILES = (
    "tools/plots/build_a0_report.py",
    "tools/plots/build_a1_t0_report.py",
    "tools/plots/build_a1_t2_report.py",
)
URL_RE = re.compile(r"(?:https?|ftp)://", re.IGNORECASE)
ASSET_RE = re.compile(r'(?:src|href)="([^"]+)"')


class InputError(ValueError):
    """Raised when a source artifact does not support a bounded transform."""


class UnsafeOutputError(ValueError):
    """Raised when an output path is not a controlled gallery target."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise InputError(f"non-finite JSON constant is not allowed: {value}")


class InputIndex:
    """Hash and decode each artifact exactly when it is first consumed."""

    def __init__(self) -> None:
        self._records: dict[Path, dict[str, Any]] = {}

    def file(self, path: Path, role: str) -> Path:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"input file does not exist: {resolved}")
        record = self._records.get(resolved)
        if record is None:
            record = {
                "path": str(resolved),
                "sha256": sha256_file(resolved),
                "bytes": resolved.stat().st_size,
                "roles": [],
            }
            self._records[resolved] = record
        if role not in record["roles"]:
            record["roles"].append(role)
            record["roles"].sort()
        return resolved

    def json(self, path: Path, role: str) -> dict[str, Any]:
        resolved = self.file(path, role)
        try:
            value = json.loads(
                resolved.read_text(encoding="utf-8"),
                parse_constant=_reject_constant,
            )
        except json.JSONDecodeError as error:
            raise InputError(f"invalid JSON input {resolved}: {error}") from error
        if not isinstance(value, dict):
            raise InputError(f"JSON input must be an object: {resolved}")
        return value

    def jsonl(self, path: Path, role: str) -> list[dict[str, Any]]:
        resolved = self.file(path, role)
        records = []
        for line_number, line in enumerate(
            resolved.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line, parse_constant=_reject_constant)
            except json.JSONDecodeError as error:
                raise InputError(
                    f"invalid JSONL input {resolved}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise InputError(
                    f"JSONL record must be an object: {resolved}:{line_number}"
                )
            records.append(value)
        return records

    def npz(self, path: Path, role: str, keys: tuple[str, ...]) -> dict[str, Any]:
        resolved = self.file(path, role)
        try:
            with np.load(resolved, allow_pickle=False) as archive:
                return {key: np.asarray(archive[key]) for key in keys if key in archive}
        except (OSError, ValueError) as error:
            raise InputError(f"invalid NPZ input {resolved}: {error}") from error

    def records(self) -> list[dict[str, Any]]:
        return [self._records[path].copy() for path in sorted(self._records)]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _series(
    label: str,
    x: list[float],
    y: list[float],
    *,
    marker: str | None = None,
    linestyle: str = "-",
) -> dict[str, Any]:
    if not x or len(x) != len(y):
        raise InputError(f"series {label!r} has no plot-ready paired values")
    if not all(math.isfinite(value) for value in (*x, *y)):
        raise InputError(f"series {label!r} contains a non-finite value")
    result: dict[str, Any] = {"label": label, "x": x, "y": y, "linestyle": linestyle}
    if marker is not None:
        result["marker"] = marker
    return result


def _panel(
    title: str,
    xlabel: str,
    ylabel: str,
    series: list[dict[str, Any]],
    *,
    xscale: str | None = None,
    yscale: str | None = None,
) -> dict[str, Any]:
    if not series:
        raise InputError(f"panel {title!r} has no observed series")
    panel: dict[str, Any] = {
        "title": title,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "series": series,
    }
    if xscale is not None:
        panel["xscale"] = xscale
    if yscale is not None:
        panel["yscale"] = yscale
    return panel


def _figure(
    figure_id: str,
    title: str,
    source_kind: str,
    panels: list[dict[str, Any]],
    *,
    caption: str,
    status: str,
    device: str,
    question: str,
    controls: str,
    criterion: str,
    components: list[str] | None = None,
) -> dict[str, Any]:
    figure: dict[str, Any] = {
        "id": figure_id,
        "title": title,
        "source_kind": source_kind,
        "caption": caption,
        "status": status,
        "device": device,
        "question": question,
        "controls": controls,
        "criterion": criterion,
        "panels": panels,
    }
    if components is not None:
        figure["components"] = components
    return figure


def _bounded_figures(figures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for figure in figures:
        panels = figure["panels"]
        if len(panels) <= 4:
            result.append(figure)
            continue
        if len(panels) == 6 and panels[0]["title"].startswith("Full P"):
            groups = [
                ("", "Target components", panels[2:]),
                ("-state", "Full state errors", panels[:2]),
            ]
        else:
            groups = [
                (
                    "" if index == 0 else f"-part{index + 1}",
                    f"Part {index + 1}",
                    panels[index * 3 : index * 3 + 3],
                )
                for index in range((len(panels) + 2) // 3)
            ]
        for suffix, label, selected in groups:
            result.append(
                {
                    **figure,
                    "id": figure["id"] + suffix,
                    "title": figure["title"] + ": " + label,
                    "panels": selected,
                }
            )
    return result


def _join_plot_series(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in series:
        label = item["label"]
        group = grouped.setdefault(label, {"template": item, "points": {}})
        for x, y in zip(item["x"], item["y"], strict=True):
            previous = group["points"].get(x)
            if previous is not None and not math.isclose(
                previous, y, rel_tol=1e-13, abs_tol=1e-24
            ):
                raise InputError(f"Conflicting continuation plot boundary: {label}")
            group["points"][x] = y
    result = []
    for group in grouped.values():
        points = sorted(group["points"].items())
        result.append(
            {
                **group["template"],
                "x": [x for x, _ in points],
                "y": [y for _, y in points],
            }
        )
    return result


def _history_records(
    run: Path, inputs: InputIndex, name: str, role: str
) -> list[dict[str, Any]]:
    path = run / "logs" / name
    return inputs.jsonl(path, role) if path.is_file() else []


def _energy_points(history: list[dict[str, Any]]) -> list[dict[str, float]]:
    points = []
    first_field = None
    for record in history:
        time_value = _number(record.get("time_s"))
        step = _integer(record.get("step"))
        energy = _mapping(record.get("energy"))
        total = _number(energy.get("total_J"))
        baseline = _number(energy.get("baseline_total_J"))
        field = _number(energy.get("field_J"))
        if time_value is None or total is None or baseline is None:
            continue
        if field is not None and first_field is None:
            first_field = field
        point = {
            "time_s": time_value,
            "step": step,
            "total_delta_J": total - baseline,
            "balance_J": _number(energy.get("balance_J")),
            "outward_J": _number(energy.get("outward_J")),
            "source_J": _number(energy.get("source_J")),
            "field_delta_J": (
                field - first_field
                if field is not None and first_field is not None
                else None
            ),
        }
        points.append(point)
    return points


def _report_energy_points(report: dict[str, Any]) -> list[dict[str, float]]:
    history = report.get("energy_history")
    if not isinstance(history, list):
        return []
    converted = []
    for item in history:
        if not isinstance(item, dict):
            continue
        energy = _mapping(item.get("energy"))
        converted.append(
            {
                "time_s": item.get("time_s"),
                "step": item.get("step"),
                "energy": energy if energy else item,
            }
        )
    return _energy_points(converted)


def _field_points(run: Path, inputs: InputIndex) -> list[dict[str, float]]:
    snapshots = sorted((run / "snapshots").glob("perturbed-step-*.npz"))
    values = []
    for path in snapshots:
        arrays = inputs.npz(
            path,
            "perturbed nonlinear field snapshot",
            ("time_s", "B_R", "B_phi", "B_Z"),
        )
        time_value = _number(np.asarray(arrays.get("time_s", [])).item())
        components = [
            arrays[name]
            for name in ("B_R", "B_phi", "B_Z")
            if name in arrays and arrays[name].size
        ]
        if time_value is None or not components:
            continue
        amplitude = max(float(np.max(np.abs(value))) for value in components)
        if math.isfinite(amplitude):
            values.append({"time_s": time_value, "maximum_abs_B_T": amplitude})
    baseline = values[0]["maximum_abs_B_T"] if values else None
    if baseline is None:
        return []
    return [
        {
            "time_s": point["time_s"],
            "field_amplitude_delta_T": point["maximum_abs_B_T"] - baseline,
        }
        for point in values
    ]


def _matched_response_points(run: Path, inputs: InputIndex) -> list[dict[str, float]]:
    keys = ("step", "time_s", "observed_BN", "observed_velocity")

    def read(ensemble: str) -> dict[int, dict[str, Any]]:
        snapshots = {}
        for path in sorted((run / "snapshots").glob(f"{ensemble}-step-*.npz")):
            arrays = inputs.npz(
                path,
                f"{ensemble} matched long-RZ snapshot",
                keys,
            )
            step = _integer(int(np.asarray(arrays.get("step", [])).item()))
            time_value = _number(np.asarray(arrays.get("time_s", [])).item())
            if step is None or time_value is None:
                continue
            snapshots[step] = {
                "time_s": time_value,
                "observed_BN": np.asarray(arrays.get("observed_BN", [])),
                "observed_velocity": np.asarray(arrays.get("observed_velocity", [])),
            }
        return snapshots

    control = read("control")
    points = []
    for step, perturbed in sorted(read("perturbed").items()):
        matched = control.get(step)
        if matched is None or not math.isclose(
            matched["time_s"],
            perturbed["time_s"],
            rel_tol=1e-12,
            abs_tol=1e-30,
        ):
            continue
        values = {}
        for name, key in (
            ("rms_delta_B_T", "observed_BN"),
            ("rms_delta_v_m_s", "observed_velocity"),
        ):
            left = matched[key]
            right = perturbed[key]
            if left.shape != right.shape or not left.size:
                break
            difference = np.abs(right - left)
            value = float(np.sqrt(np.mean(difference * difference)))
            if not math.isfinite(value):
                break
            values[name] = value
        if len(values) == 2:
            points.append(
                {
                    "step": step,
                    "time_s": perturbed["time_s"],
                    **values,
                }
            )
    return points


def _scan_runs(
    run_root: Path, inputs: InputIndex
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    complete = []
    long_reports = []
    notes = []
    for directory in sorted(run_root.iterdir()):
        if not directory.is_dir():
            continue
        name = directory.name
        if name == "base-east16-waveform":
            notes.append(
                "Historical base-east16-waveform: excluded from the current matched "
                "A0/T1 figure and retained in its separate comparison scope."
            )
            continue
        if not (name.startswith(("optimized-east", "retry-east", "long-east"))):
            continue
        report_path = directory / "report.json"
        if not report_path.is_file():
            continue
        report = inputs.json(report_path, f"run report {name}")
        status = str(report.get("status", "unknown"))
        case = _mapping(report.get("case"))
        is_long = name.startswith("long-east")
        accepted_prefix = is_long and status == "budget_stopped"
        record = {
            "id": name,
            "path": report_path,
            "report_sha256": sha256_file(report_path),
            "status": status,
            "run_id": report.get("run_id"),
            "segment_start_step": _integer(
                _nested(report, "segment_probe", "start_step")
            ),
            "resume_from": _nested(report, "restart", "resume_from"),
            "requested_horizon_s": _number(report.get("requested_horizon_s")),
            "reached_horizon_s": _number(report.get("reached_horizon_s")),
            "planned_macros": _integer(report.get("planned_macros")),
            "reached_macros": _integer(report.get("reached_macros")),
            "coupling": str(
                case.get("coupling") or _nested(report, "coupling", "mode") or ""
            ),
            "nodes": _integer(case.get("nodes")),
            "active_cells": _integer(case.get("active_cells")),
            "device": str(case.get("device", "unavailable")),
            "h_s": _number(report.get("h_s")),
            "ensemble": str(
                report.get("ensemble") or ("perturbed" if is_long else "not recorded")
            ),
            "report": report,
            "timing_median_s": None,
            "resident_bytes": None,
            "energy_points": [],
            "control_energy_points": [],
            "matched_response_points": [],
            "field_points": [],
        }
        if is_long:
            long_reports.append(record)
            if accepted_prefix:
                reached = record["reached_horizon_s"]
                requested = record["requested_horizon_s"]
                notes.append(
                    f"Long-RZ {name}: budget_stopped accepted prefix; reached "
                    f"{reached:.6g} s ({record['reached_macros']} macros), requested "
                    f"{requested:.6g} s ({record['planned_macros']} macros); not a "
                    "complete full-horizon run."
                )
        if status != "completed" and not accepted_prefix:
            failure = _mapping(report.get("failure"))
            reason = (
                failure.get("code")
                or report.get("failure_code")
                or report.get("stop_reason")
                or "not recorded"
            )
            notes.append(
                f"Run {name}: {status}; failure/reason {reason}. "
                "No timing or memory point is plotted."
            )
            if not is_long:
                continue
        timing_path = directory / "logs/advance-timing.jsonl"
        if timing_path.is_file():
            timing = inputs.jsonl(timing_path, f"complete advance timing {name}")
            values = [
                elapsed
                for item in timing
                if item.get("status") == "completed"
                and (elapsed := _number(item.get("elapsed_s"))) is not None
            ]
            record["timing_median_s"] = _median(values)
        memory_path = directory / "memory-inventory.json"
        if memory_path.is_file():
            memory = inputs.json(memory_path, f"device memory inventory {name}")
            record["resident_bytes"] = _number(memory.get("resident_bytes"))
        if record["resident_bytes"] is None:
            record["resident_bytes"] = _number(
                _nested(report, "memory", "device_resident_bytes")
            )
        history = _history_records(
            directory,
            inputs,
            "perturbed-history.jsonl",
            f"perturbed scalar history {name}",
        )
        record["energy_points"] = _energy_points(history)
        if is_long:
            control_history = _history_records(
                directory,
                inputs,
                "control-history.jsonl",
                f"control scalar history {name}",
            )
            record["control_energy_points"] = _energy_points(control_history)
            report_history = _report_energy_points(report)
            if report_history and not record["energy_points"]:
                if record["ensemble"] == "control":
                    record["control_energy_points"] = report_history
                else:
                    record["energy_points"] = report_history
            record["matched_response_points"] = _matched_response_points(
                directory, inputs
            )
        else:
            record["field_points"] = _field_points(directory, inputs)
        if not is_long:
            complete.append(record)
    return complete, long_reports, notes


def _cost_figure(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [
        run
        for run in runs
        if run["timing_median_s"] is not None
        and run["resident_bytes"] is not None
        and run["active_cells"] is not None
        and run["nodes"] is not None
    ]
    if not usable:
        return None

    def grouped(key: str, ylabel: str, title: str) -> dict[str, Any]:
        panels = []
        for coupling in sorted({run["coupling"] for run in usable}):
            selected = [run for run in usable if run["coupling"] == coupling]
            panels.append(
                _series(
                    f"{coupling}: {len(selected)} complete run(s)",
                    [float(run["active_cells"]) for run in selected],
                    [float(run[key]) for run in selected],
                    marker="o",
                    linestyle="None",
                )
            )
        return _panel(
            title,
            "Active spatial cells",
            ylabel,
            panels,
            xscale="log",
            yscale="log" if key in {"timing_median_s", "resident_bytes"} else "linear",
        )

    timing = grouped(
        "timing_median_s",
        "Median synchronized complete advance (s)",
        "Complete nonlinear advance timing",
    )
    memory = grouped(
        "resident_bytes",
        "Retained resident device memory (bytes)",
        "Inspected device-resident arrays",
    )
    size_series = []
    for coupling in sorted({run["coupling"] for run in usable}):
        selected = [run for run in usable if run["coupling"] == coupling]
        size_series.append(
            _series(
                coupling,
                [float(run["active_cells"]) for run in selected],
                [float(run["nodes"]) for run in selected],
                marker="s",
                linestyle="None",
            )
        )
    size = _panel(
        "Declared spatial size",
        "Active spatial cells",
        "Enclosing R/Z nodes per axis",
        size_series,
        xscale="log",
    )
    names = ", ".join(sorted(run["id"] for run in usable))
    return _figure(
        "complete-spatial-cost",
        "Completed A0/T1 spatial cost and device memory",
        "mixed_comparison",
        [timing, memory, size],
        caption=(
            f"Completed runs: {names}. Timing uses synchronized complete advance "
            "records from advance-timing.jsonl; state copies and output are excluded "
            "by the recorded scope. Memory is the retained array inventory, not a "
            "device-wide allocator high-water mark. Discrete runs are marker-only; "
            "failed or in-progress reports are excluded."
        ),
        status="Completed run records only; no speedup, scaling, or physics pass is inferred.",
        device="CUDA device reports",
        question="What are the observed spatial size, complete advance time, and retained memory?",
        controls="Matched completed records grouped by reported coupling.",
        criterion="Display only finite timing and memory values from completed reports.",
        components=[
            "synchronized nonlinear advances",
            "retained device-resident arrays",
            "reported active spatial cells",
        ],
    )


def _energy_figure(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    energy_runs = [run for run in runs if len(run["energy_points"]) >= 2]
    if not energy_runs:
        return None

    total_series = []
    field_energy_series = []
    for index, run in enumerate(energy_runs):
        label = f"{run['nodes']}x{run['nodes']} grid, {run['coupling']}"
        marker = ("o", "s", "^", "D")[index % 4]
        total_series.append(
            _series(
                label,
                [point["time_s"] for point in run["energy_points"]],
                [point["total_delta_J"] for point in run["energy_points"]],
                marker=marker,
            )
        )
        field_energy_series.append(
            _series(
                label,
                [point["time_s"] for point in run["energy_points"]],
                [point["field_delta_J"] for point in run["energy_points"]],
                marker=marker,
            )
        )
    panels = [
        _panel(
            "Total energy minus archived baseline",
            "Physical time (s)",
            "Total energy change (J)",
            total_series,
            yscale="symlog",
        ),
        _panel(
            "Raw field energy minus first scalar record",
            "Physical time (s)",
            "Field energy change (J)",
            field_energy_series,
            yscale="symlog",
        ),
    ]
    field_runs = [run for run in energy_runs if len(run["field_points"]) >= 2]
    if field_runs:
        field_amplitude_series = []
        for index, run in enumerate(field_runs):
            field_amplitude_series.append(
                _series(
                    f"{run['nodes']}x{run['nodes']} grid, {run['coupling']}",
                    [point["time_s"] for point in run["field_points"]],
                    [point["field_amplitude_delta_T"] for point in run["field_points"]],
                    marker=("o", "s", "^", "D")[index % 4],
                )
            )
        panels.append(
            _panel(
                "Maximum stored magnetic-component absolute value: change",
                "Physical time (s)",
                "Change of max stored component abs (T)",
                field_amplitude_series,
                yscale="symlog",
            )
        )
    names = ", ".join(run["id"] for run in energy_runs)
    return _figure(
        "nonlinear-energy-field",
        "Short-window nonlinear energy and field-state changes",
        "nonlinear_evolution",
        panels,
        caption=(
            f"Perturbed scalar histories and NPZ field snapshots from {names}. "
            "The first panel subtracts the archived baseline_total_J in each record; "
            "the latter panels subtract the first available raw field value. These "
            "are diagnostics, not a production energy ledger."
        ),
        status="Short completed windows only; raw diagnostics and not a production ledger.",
        device="CUDA device reports",
        question="What raw energy and field-state changes are visible over the completed physical windows?",
        controls="Matched control/perturbed nonlinear records; the plotted histories are explicit perturbed records.",
        criterion="Plot only stored scalar histories and snapshots without interpolation.",
    )


def _scan_initial_guesses(
    run_root: Path, inputs: InputIndex
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    summary_path = run_root / "analysis/initial-guesses/summary.json"
    if summary_path.is_file():
        summary = inputs.json(summary_path, "combined initial-guess summary")
        cases = {
            str(case.get("case_id")): case
            for case in summary.get("cases", [])
            if isinstance(case, dict) and case.get("case_id") is not None
        }
        records = []
        for experiment in summary.get("experiments", []):
            if not isinstance(experiment, dict):
                continue
            case_id = str(experiment.get("case_id") or "unrecorded")
            case = _mapping(cases.get(case_id))
            config = _mapping(case.get("config"))
            status = str(experiment.get("status", "unknown"))
            timing = _mapping(experiment.get("timing"))
            elapsed = _number(timing.get("median_s"))
            if elapsed is None:
                elapsed = _number(experiment.get("attempted_elapsed_s"))
            history = []
            for item in experiment.get("history", []):
                if not isinstance(item, dict):
                    continue
                iteration = _integer(item.get("iteration"))
                trial = _integer(item.get("trial"))
                residual = _number(item.get("residual"))
                if iteration is None or residual is None:
                    continue
                history.append(
                    {
                        "iteration": iteration,
                        "trial": trial,
                        "residual": residual,
                        "lambda": _number(item.get("lambda")),
                    }
                )
            work = _mapping(experiment.get("work"))
            backend = str(experiment.get("backend", "unrecorded")).upper()
            failure = _mapping(experiment.get("failure"))
            records.append(
                {
                    "backend": backend,
                    "case_id": case_id,
                    "case_short": case_id.removeprefix("e2-n33-").removeprefix(
                        "e3-n12-"
                    ),
                    "unit": str(config.get("unit", "unrecorded")),
                    "M": _integer(config.get("M")),
                    "points": _integer(config.get("points")),
                    "device": backend,
                    "guess": str(experiment.get("guess", "unknown")),
                    "status": status,
                    "history": history,
                    "iterations": _integer(experiment.get("iterations_to_convergence"))
                    or _integer(
                        experiment.get("selected_correction_iterations_before_stop")
                    ),
                    "elapsed_s": elapsed,
                    "timing_label": experiment.get("timing_label"),
                    "rhs_vectors": _integer(
                        work.get("final_execution_producer_rhs_vectors")
                    ),
                    "failure_code": failure.get("code"),
                    "audit_G_caveat": experiment.get("audit_G_caveat"),
                }
            )
        pairs = [
            pair
            for pair in summary.get("constant_tangent_pairs", [])
            if isinstance(pair, dict)
        ]
        notes = [
            (
                "Initial guesses: the combined summary is fixed-P field-window "
                "verification, not a coupled T1 result."
            )
        ]
        method = _mapping(summary.get("method"))
        input_identity = str(method.get("input_identity") or "")
        if "E2" in input_identity:
            notes.append(
                "Initial-guess E2 CPU/GPU saved inputs differ at backend fixture "
                "construction; initial-Z roundoff is measured, and P bytes were "
                "not saved."
            )
        failed = [record for record in records if record["status"] == "failed"]
        if failed:
            names = ", ".join(
                f"{record['backend']}/{record['case_id']}/{record['guess']}"
                for record in failed
            )
            notes.append(
                f"Initial-guess failed records: {names}. These retain "
                "FieldWindowConvergence evidence and are not solutions."
            )
        if any(
            "audit" in str(record.get("audit_G_caveat") or "").lower()
            for record in failed
        ):
            notes.append(
                "Failed E3 audit final G reconstruction is not an accepted true "
                "residual; selected-iterate history is kept distinct."
            )
        for limit in summary.get("limits", []):
            notes.append(f"Initial-guess limit: {limit}")
        return records, pairs, notes

    records = []
    notes = []
    for root_name in INITIAL_GUESS_ROOTS:
        root = run_root / root_name
        if not root.is_dir():
            continue
        suite_path = root / "suite.json"
        suite = (
            inputs.json(suite_path, f"initial-guess suite {root_name}")
            if suite_path.is_file()
            else {}
        )
        for report_path in sorted(root.glob("*/report.json")):
            report = inputs.json(
                report_path, f"initial-guess case {_relative(report_path, run_root)}"
            )
            case = _mapping(report.get("case"))
            case_id = str(case.get("id") or report_path.parent.name)
            device = str(report.get("device") or suite.get("device") or "unavailable")
            for guess_name in ("constant", "tangent"):
                guess = _mapping(report.get(guess_name))
                history = [
                    item
                    for item in guess.get("history", [])
                    if isinstance(item, dict)
                    and _integer(item.get("iteration")) is not None
                    and _number(item.get("residual")) is not None
                ]
                failure = _mapping(guess.get("failure"))
                timing = _mapping(guess.get("timing"))
                elapsed = _number(timing.get("median_s"))
                if elapsed is None:
                    elapsed = _number(guess.get("attempted_elapsed_s"))
                work = _mapping(guess.get("work"))
                record = {
                    "suite": root_name,
                    "case_id": case_id,
                    "device": device,
                    "guess": guess_name,
                    "status": str(guess.get("status", "unknown")),
                    "stress": bool(case.get("stress", False)),
                    "M": _integer(case.get("M")),
                    "iterations": _integer(guess.get("iterations")),
                    "history": history,
                    "elapsed_s": elapsed,
                    "rhs_vectors": _integer(
                        work.get("final_execution_producer_rhs_vectors")
                    ),
                    "failure_code": failure.get("code"),
                }
                records.append(record)
                if record["status"] == "failed":
                    notes.append(
                        f"Initial-guess failed record: {root_name}/{case_id}/{guess_name}; "
                        f"code {record['failure_code'] or 'not recorded'}."
                    )
    if any(record["suite"] == "initial-guesses-gpu-fixed" for record in records):
        notes.append(
            "Fixed-P initial-guess scope: these are field-window verification runs, "
            "not a coupled T1 candidate."
        )
    return records, [], notes


def _guess_figure(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    with_history = [record for record in records if len(record["history"]) >= 2]
    if not with_history:
        return None
    successful = next(
        (
            record
            for record in with_history
            if record["status"] == "completed"
            and not record["stress"]
            and record["suite"] == "initial-guesses-cpu"
        ),
        None,
    )
    failed = next(
        (
            record
            for record in with_history
            if record["status"] == "failed" and record["suite"] == "initial-guesses-cpu"
        ),
        None,
    )
    panels = []
    if successful is not None:
        series = []
        for record in with_history:
            if (
                record["suite"] == successful["suite"]
                and record["case_id"] == successful["case_id"]
            ):
                series.append(
                    _series(
                        f"{record['guess']} guess",
                        [float(item["iteration"]) for item in record["history"]],
                        [float(item["residual"]) for item in record["history"]],
                        marker="o" if record["guess"] == "constant" else "s",
                    )
                )
        panels.append(
            _panel(
                f"Residual trajectory: {successful['case_id']}",
                "Correction iteration",
                "Reported residual",
                series,
                yscale="log",
            )
        )
    if failed is not None:
        series = []
        for record in with_history:
            if record["case_id"] != failed["case_id"] or record["status"] != "failed":
                continue
            if record["suite"] not in {
                "initial-guesses-cpu",
                "initial-guesses-gpu-fixed",
            }:
                continue
            series.append(
                _series(
                    f"{record['device']} {record['guess']} ({record['failure_code']})",
                    [float(item["iteration"]) for item in record["history"]],
                    [float(item["residual"]) for item in record["history"]],
                    marker="^" if record["guess"] == "constant" else "D",
                )
            )
        panels.append(
            _panel(
                f"Failed residual trajectory: {failed['case_id']}",
                "Correction iteration",
                "Reported residual",
                series,
                yscale="log",
            )
        )
    timed = [record for record in with_history if record["elapsed_s"] is not None]
    if timed:
        series = []
        for guess_name in ("constant", "tangent"):
            selected = [record for record in timed if record["guess"] == guess_name]
            if not selected:
                continue
            series.append(
                _series(
                    guess_name,
                    [
                        float(record["iterations"] or len(record["history"]) - 1)
                        for record in selected
                    ],
                    [float(record["elapsed_s"]) for record in selected],
                    marker="o" if guess_name == "constant" else "s",
                    linestyle="None",
                )
            )
        panels.append(
            _panel(
                "Corrections and elapsed time",
                "Correction iterations",
                "Median or attempted elapsed time (s)",
                series,
                yscale="log",
            )
        )
    if not panels:
        return None
    failed_count = sum(record["status"] == "failed" for record in records)
    return _figure(
        "initial-guess",
        "Fixed-P initial-guess convergence, timing, and failures",
        "verification",
        panels,
        caption=(
            "Representative case reports retain the reported residual histories for "
            "constant and tangent guesses. The fixed-P guess is a field-window "
            "verification, not a coupled T1 candidate. Failed attempts are shown "
            "with their recorded failure code; attempted elapsed times are not "
            "performance measurements. Timing observations are marker-only."
        ),
        status=(
            f"Raw case reports include {failed_count} failed guess record(s); "
            "fixed-P is not a coupled T1."
        ),
        device="CPU and fixed-P CUDA cases",
        question="How do field-window initial guesses converge, time, and fail?",
        controls="Constant and built-in tangent guesses against each case's independent sequential same-h reference.",
        criterion="Reported original residual histories and per-guess statuses; no coupled acceptance is inferred.",
    )


def _summary_guess_figures(
    records: list[dict[str, Any]], pairs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    figures = []
    trajectory_panels = []
    for backend in ("CPU", "GPU"):
        for unit in ("E2", "E3"):
            selected = [
                record
                for record in records
                if record.get("backend") == backend
                and record.get("unit") == unit
                and len(record["history"]) >= 2
            ]
            if not selected:
                continue
            series = []
            for record in selected:
                status = record["status"]
                series.append(
                    _series(
                        f"{record['case_short']} {record['guess']} "
                        f"{'FAIL' if status == 'failed' else 'OK'}",
                        list(range(len(record["history"]))),
                        [item["residual"] for item in record["history"]],
                        marker="o" if record["guess"] == "constant" else "s",
                    )
                )
            trajectory_panels.append(
                _panel(
                    f"{backend} {unit} recorded residual histories",
                    "Recorded attempt index",
                    "Reported residual",
                    series,
                    yscale="log",
                )
            )
    if trajectory_panels:
        figures.append(
            _figure(
                "initial-guess-trajectories",
                "Fixed-P initial-guess residual histories",
                "verification",
                trajectory_panels,
                caption=(
                    "All recorded constant and tangent attempts in the combined "
                    "summary are retained, including retained difficult failures "
                    "and backtracking trials. Fixed-P field-window verification is "
                    "not a coupled T1 result. Cases: "
                    + ", ".join(dict.fromkeys(record["case_id"] for record in records))
                    + "."
                ),
                status="Exploratory field-window histories; failures are not solutions.",
                device="CPU and fixed-P CUDA summary records",
                question="How did constant and tangent guesses evolve and where did they fail?",
                controls=(
                    "Each curve is one backend, case, and guess; no curve spans "
                    "cases or backends."
                ),
                criterion=(
                    "Use the summary's recorded attempt histories without treating "
                    "a failed residual as an accepted state."
                ),
            )
        )

    case_ids = list(
        dict.fromkeys(
            [str(pair.get("case_id")) for pair in pairs]
            + [record["case_id"] for record in records]
        )
    )
    case_index = {case_id: index for index, case_id in enumerate(case_ids)}
    summary_panels = []
    if pairs:
        series = []
        for backend in sorted({str(pair.get("backend", "")).upper() for pair in pairs}):
            selected = [
                pair
                for pair in pairs
                if str(pair.get("backend", "")).upper() == backend
                and pair.get("case_id") in case_index
                and _number(pair.get("initial_residual_constant_over_tangent"))
                is not None
            ]
            if not selected:
                continue
            series.append(
                _series(
                    f"{backend} initial residual ratio",
                    [float(case_index[pair["case_id"]]) for pair in selected],
                    [
                        float(pair["initial_residual_constant_over_tangent"])
                        for pair in selected
                    ],
                    marker="o" if backend == "CPU" else "s",
                    linestyle="None",
                )
            )
        if series:
            summary_panels.append(
                _panel(
                    "Constant/tangent initial residual ratio",
                    "Case index",
                    "Constant / tangent initial residual",
                    series,
                )
            )
    iterations_series = []
    for backend in ("CPU", "GPU"):
        for guess in ("constant", "tangent"):
            selected = [
                record
                for record in records
                if record.get("backend") == backend
                and record["guess"] == guess
                and record["status"] == "completed"
                and record.get("iterations") is not None
            ]
            if selected:
                iterations_series.append(
                    _series(
                        f"{backend} {guess}",
                        [float(case_index[record["case_id"]]) for record in selected],
                        [float(record["iterations"]) for record in selected],
                        marker="o" if guess == "constant" else "s",
                        linestyle="None",
                    )
                )
    if iterations_series:
        summary_panels.append(
            _panel(
                "Converged correction iterations",
                "Case index",
                "Correction iterations",
                iterations_series,
            )
        )
    elapsed_series = []
    for backend in ("CPU", "GPU"):
        for guess in ("constant", "tangent"):
            selected = [
                record
                for record in records
                if record.get("backend") == backend
                and record["guess"] == guess
                and record.get("elapsed_s") is not None
                and record["elapsed_s"] > 0.0
            ]
            if selected:
                elapsed_series.append(
                    _series(
                        f"{backend} {guess} (median; failed=attempted)",
                        [float(case_index[record["case_id"]]) for record in selected],
                        [record["elapsed_s"] for record in selected],
                        marker="o" if guess == "constant" else "s",
                        linestyle="None",
                    )
                )
    if elapsed_series:
        summary_panels.append(
            _panel(
                "Median successful timing or failed attempt",
                "Case index",
                "Elapsed time (s)",
                elapsed_series,
                yscale="symlog",
            )
        )
    if summary_panels:
        mapping = ", ".join(
            f"{index}={case_id}" for index, case_id in enumerate(case_ids)
        )
        figures.append(
            _figure(
                "initial-guess-pairs",
                "Constant/tangent pairs, iterations, timing, and failures",
                "verification",
                summary_panels,
                caption=(
                    f"Case index: {mapping}. All six cases and both backends are "
                    "retained. Failed attempts use attempted elapsed time only and "
                    "are never interpreted as successful solves."
                ),
                status="Pair summaries preserve difficult failures; fixed-P is not coupled T1.",
                device="CPU and fixed-P CUDA summary records",
                question="Which initial-guess pairs converged, and how did timing or failure differ?",
                controls="Case/backend/guess identities are retained; discrete points are not connected.",
                criterion="Use only summary pair scalars and recorded timing fields.",
            )
        )
    return figures


def _guess_figures(
    records: list[dict[str, Any]], pairs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if any("backend" in record for record in records):
        return _summary_guess_figures(records, pairs)
    figure = _guess_figure(records)
    return [figure] if figure is not None else []


def _candidate_question(row: dict[str, Any]) -> str:
    row_id = str(row.get("id", ""))
    case = str(row.get("case", "unrecorded"))
    mode = _integer(row.get("seed_mode"))
    points = _integer(row.get("points"))
    if case == "closed":
        return f"closed n={points} control"
    if case != "periodic":
        return f"{case} control"
    if row_id.endswith("-halfh"):
        return "fixed H half h"
    if mode != 1 or "target" in row_id.lower():
        return f"target mode {mode}"
    if re.search(r"-M\d+$", row_id):
        return "fixed h H scan"
    return f"matrix row {row_id}"


def _candidate_group_key(record: dict[str, Any]) -> tuple[Any, ...]:
    question = record["question"]
    if question == "fixed h H scan":
        control = ("h", record["h"])
    elif question == "fixed H half h":
        control = ("H", record["H"])
    else:
        control = ("h,H", record["h"], record["H"])
    return (
        record["method"],
        record["device"],
        question,
        record["case"],
        record["mode"],
        record["points"],
        record["duration_s"],
        record["amplitude"],
        record["nonuniform"],
        control,
    )


def _candidate_series_label(
    record: dict[str, Any], *, quantity: str | None = None
) -> str:
    if record["question"] == "fixed h H scan":
        detail = f"fixed-h H scan, mode{record['mode']}"
    elif record["question"] == "fixed H half h":
        detail = f"fixed-H half-h, mode{record['mode']}"
    elif record["case"] == "closed":
        detail = f"closed, n={record['points']}"
    else:
        detail = f"target mode{record['mode']}"
    label = f"{record['method']} {record['device']}: {detail}, duration={record['duration_s']:.6g} s"
    return f"{label} {quantity}" if quantity else label


def _candidate_line_style(records: list[dict[str, Any]]) -> str:
    if (
        len({record["H"] for record in records}) > 1
        and records[0]["question"] == "fixed h H scan"
    ):
        return "--" if records[0]["method"] == "C2" else "-"
    return "None"


def _candidate_marker(record: dict[str, Any], base: str) -> str:
    if record["device"] == "GPU":
        return {"o": "^", "s": "D", "^": "P", "D": "X"}.get(base, "X")
    return base


def _history_number(history: dict[str, Any], key: str) -> float | None:
    value = _number(history.get(key))
    if value is not None:
        return value
    values = history.get(key)
    if isinstance(values, list):
        for item in reversed(values):
            number = _number(item)
            if number is not None:
                return number
    return None


def _component_errors(
    candidate: dict[str, Any], oracle: dict[str, Any]
) -> tuple[float | None, float | None]:
    candidate_amplitude = _number(candidate.get("normalized_amplitude"))
    oracle_amplitude = _number(oracle.get("normalized_amplitude"))
    candidate_phase = _number(candidate.get("phase_rad"))
    oracle_phase = _number(oracle.get("phase_rad"))
    amplitude_error = None
    phase_error = None
    if candidate_amplitude is not None and oracle_amplitude not in (None, 0.0):
        amplitude_error = abs(candidate_amplitude / oracle_amplitude - 1.0)
    if candidate_phase is not None and oracle_phase is not None:
        difference = candidate_phase - oracle_phase
        phase_error = abs(math.atan2(math.sin(difference), math.cos(difference)))
    return amplitude_error, phase_error


def _validate_matrix_row(
    row: dict[str, Any], actual: dict[str, Any], *, source: str
) -> None:
    input_style = "method" in actual and "seed_mode" in actual
    required = {
        "method": "method" if input_style else "candidate",
        "case": "case",
        "points": "points",
        "h": "h",
        "M": "M",
        "macros": "macros",
        "seed_mode": "seed_mode" if input_style else "mode",
    }
    for expected_key, actual_key in required.items():
        expected = row.get(expected_key)
        observed = actual.get(actual_key)
        if expected_key in {"h"}:
            left = _number(expected)
            right = _number(observed)
            matches = (
                left is not None
                and right is not None
                and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-30)
            )
        elif expected_key in {"M", "macros", "points", "seed_mode"}:
            matches = _integer(expected) == _integer(observed)
        else:
            matches = str(expected) == str(observed)
        if not matches:
            raise InputError(
                f"candidate matrix mismatch for {source}.{actual_key}: "
                f"matrix={expected!r} actual={observed!r}"
            )


def _scan_candidates(
    run_root: Path, inputs: InputIndex
) -> tuple[list[dict[str, Any]], list[str]]:
    records = []
    notes = []
    matrix_path = run_root / MATRIX_FILE
    if not matrix_path.is_file():
        return records, [
            (
                "Multirate candidate comparisons: unavailable. Controlled matrix "
                "multirate-matrix.json is absent."
            )
        ]
    matrix = inputs.json(matrix_path, "controlled multirate matrix")
    rows = [row for row in matrix.get("rows", []) if isinstance(row, dict)]
    if not rows:
        return records, [
            (
                "Multirate candidate comparisons: unavailable. The controlled matrix "
                "contains no rows."
            )
        ]
    amplitude = _number(matrix.get("amplitude"))
    nonuniform = _number(matrix.get("nonuniform"))
    fixed_periodic_horizon = _number(matrix.get("fixed_periodic_horizon_s"))
    target_period = _number(matrix.get("target_T_s"))
    for root_name in CANDIDATE_ROOTS:
        root = run_root / root_name
        device = "GPU" if root_name.endswith("gpu") else "CPU"
        missing = []
        completed_ids = []
        for row in rows:
            row_id = str(row.get("id") or "unrecorded")
            directory = root / row_id
            report_path = directory / "report.json"
            input_path = directory / "input.json"
            provenance_path = directory / "provenance.json"
            if input_path.is_file():
                input_config = inputs.json(
                    input_path, f"candidate input {root_name}/{row_id}"
                )
                _validate_matrix_row(
                    row, input_config, source=f"{root_name}/{row_id}/input.json"
                )
            if provenance_path.is_file():
                inputs.json(
                    provenance_path,
                    f"candidate provenance {root_name}/{row_id}",
                )
            if not report_path.is_file():
                missing.append(row_id)
                continue
            report = inputs.json(report_path, f"candidate report {root_name}/{row_id}")
            arrays_path = directory / "arrays.npz"
            if arrays_path.is_file():
                inputs.file(
                    arrays_path,
                    f"candidate arrays {root_name}/{row_id}",
                )
            _validate_matrix_row(
                row, report, source=f"{root_name}/{row_id}/report.json"
            )
            h = _number(row.get("h"))
            M = _integer(row.get("M"))
            macros = _integer(row.get("macros"))
            if h is None or M is None or macros is None:
                raise InputError(f"candidate matrix row is incomplete: {row_id}")
            H = h * M
            duration = H * macros
            for key, expected in (
                ("H", H),
                ("duration", duration),
            ):
                observed = (
                    _number(report.get("H"))
                    if key == "H"
                    else _number(_nested(report, "time_contract", "total_duration_s"))
                )
                if observed is None or not math.isclose(
                    observed, expected, rel_tol=1e-12, abs_tol=1e-30
                ):
                    raise InputError(
                        f"candidate time contract mismatch {root_name}/{row_id}.{key}"
                    )
            status = str(report.get("status", "unknown"))
            failure = _mapping(report.get("failure"))
            failure_code = failure.get("code") or report.get("failure_code")
            if status != "completed":
                notes.append(
                    f"Candidate {root_name}/{row_id}: {status}; failure/reason "
                    f"{failure_code or 'not recorded'}. No numeric point is plotted."
                )
                continue
            reference = _mapping(report.get("reference"))
            mode_comparison = _mapping(report.get("mode_comparison"))
            if device == "GPU" and not mode_comparison:
                notes.append(
                    f"Completed GPU {row_id} has no rerun same-backend oracle; "
                    "its oracle-error curves are absent. Same-algorithm CPU/GPU "
                    "differences are shown separately in the backend verification panel."
                )
            target = _mapping(mode_comparison.get("target"))
            target_candidate = _mapping(target.get("candidate_history"))
            target_oracle = _mapping(target.get("oracle_history"))
            velocity_amplitude_error, velocity_phase_error = _component_errors(
                _mapping(_nested(mode_comparison, "candidate", "velocity_mode")),
                _mapping(_nested(mode_comparison, "oracle", "velocity_mode")),
            )
            magnetic_amplitude_error, magnetic_phase_error = _component_errors(
                _mapping(_nested(mode_comparison, "candidate", "magnetic_mode")),
                _mapping(_nested(mode_comparison, "oracle", "magnetic_mode")),
            )
            if velocity_amplitude_error is None:
                candidate_amplitude = _history_number(
                    target_candidate, "final_amplitude"
                )
                oracle_amplitude = _history_number(target_oracle, "final_amplitude")
                if candidate_amplitude is not None and oracle_amplitude not in (
                    None,
                    0.0,
                ):
                    velocity_amplitude_error = abs(
                        candidate_amplitude / oracle_amplitude - 1.0
                    )
            if velocity_phase_error is None:
                candidate_phase = _history_number(target_candidate, "phases_rad")
                oracle_phase = _history_number(target_oracle, "phases_rad")
                if candidate_phase is not None and oracle_phase is not None:
                    difference = candidate_phase - oracle_phase
                    velocity_phase_error = abs(
                        math.atan2(math.sin(difference), math.cos(difference))
                    )
            sentinel = _mapping(mode_comparison.get("sentinel"))
            sentinel_candidate = _mapping(sentinel.get("candidate_history"))
            sentinel_oracle = _mapping(sentinel.get("oracle_history"))
            returned = _mapping(report.get("returned_execution_counters"))
            timing = _mapping(report.get("timing"))
            record = {
                "id": row_id,
                "root": root_name,
                "question": _candidate_question(row),
                "case": str(row["case"]),
                "method": str(row["method"]),
                "device": device,
                "status": status,
                "mode": _integer(row.get("seed_mode")),
                "points": _integer(row.get("points")),
                "h": h,
                "H": H,
                "M": M,
                "macros": macros,
                "duration_s": duration,
                "fixed_periodic_horizon_s": fixed_periodic_horizon,
                "target_period_s": target_period,
                "amplitude": amplitude,
                "nonuniform": nonuniform,
                "p_error": _number(reference.get("p_normalized")),
                "z_error": _number(reference.get("z_normalized")),
                "target_velocity_amplitude_error": velocity_amplitude_error,
                "target_velocity_phase_error": velocity_phase_error,
                "target_magnetic_amplitude_error": magnetic_amplitude_error,
                "target_magnetic_phase_error": magnetic_phase_error,
                "sentinel_candidate_max_gain": _number(
                    sentinel_candidate.get("maximum_gain")
                ),
                "sentinel_candidate_final_gain": _number(
                    sentinel_candidate.get("final_gain")
                ),
                "sentinel_sync_max_gain": _number(sentinel_oracle.get("maximum_gain")),
                "sentinel_sync_final_gain": _number(sentinel_oracle.get("final_gain")),
                "full_hydro_eval": _number(returned.get("full_hydro_eval")),
                "cheap_D": _number(returned.get("cheap_D")),
                "cheap_L": _number(returned.get("cheap_L")),
                "field_rhs": _number(returned.get("field_rhs")),
                "campaign_multiplier": _number(
                    timing.get("campaign_multiplier_per_algorithm")
                ),
            }
            records.append(record)
            completed_ids.append(row_id)
            if not returned:
                notes.append(
                    f"Candidate {root_name}/{row_id}: returned_execution_counters "
                    "are absent; no work-count point is plotted."
                )
        if missing:
            notes.append(
                f"Candidate matrix {device}: missing report.json for "
                f"{', '.join(missing)}. No numeric points are interpolated."
            )
        notes.append(
            f"Candidate matrix {device}: {len(completed_ids)}/{len(rows)} "
            "report.json files completed and passed the matrix/time-contract checks."
        )
        if device == "GPU":
            revision = _mapping(matrix.get("GPU_coverage_revision"))
            planned = [
                str(row_id)
                for row_id in revision.get("planned_gpu_cases", [])
                if str(row_id) in {str(row.get("id")) for row in rows}
            ]
            if planned:
                completed_set = set(completed_ids)
                completed_planned = [
                    row_id for row_id in planned if row_id in completed_set
                ]
                pending_planned = [
                    row_id for row_id in planned if row_id not in completed_set
                ]
                outside_completed = sorted(completed_set - set(planned))
                unscheduled = len(rows) - len(set(planned) | completed_set)
                notes.append(
                    "Candidate matrix GPU planned coverage: "
                    f"{len(completed_planned)}/{len(planned)} completed; completed "
                    f"{', '.join(completed_planned) or 'none'}; pending "
                    f"{', '.join(pending_planned) or 'none'}. {unscheduled} matrix "
                    "row(s) are not scheduled for GPU. "
                    f"Completed beyond the initial subset: {', '.join(outside_completed) or 'none'}."
                )
    legacy_root = run_root / LEGACY_CANDIDATE_ROOT
    for report_path in sorted(legacy_root.glob("*/report.json")):
        report = inputs.json(
            report_path,
            f"historical candidate failure {_relative(report_path, run_root)}",
        )
        traceback_path = report_path.parent / "traceback.txt"
        if traceback_path.is_file():
            inputs.file(
                traceback_path,
                f"historical candidate traceback {_relative(traceback_path, run_root)}",
            )
        reason = report.get("failure_code") or _mapping(report.get("failure")).get(
            "code"
        )
        notes.append(
            f"Historical {_relative(report_path.parent, run_root)}: "
            f"status={report.get('status', 'unknown')}, failure/reason "
            f"{reason or 'not recorded'}; excluded from candidates-cpu-v2 evidence."
        )
    return records, notes


def _candidate_figures(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    usable = [record for record in records if record["status"] == "completed"]
    figures = []
    notes = []

    def same(value: float | None, target: float | None) -> bool:
        return (
            value is not None
            and target is not None
            and math.isclose(value, target, rel_tol=1e-12, abs_tol=1e-30)
        )

    def comparison_series(
        selected: list[dict[str, Any]],
        metric: str,
        x_key: str,
        scope: str,
        marker: str,
    ) -> list[dict[str, Any]]:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for record in selected:
            if record[metric] is None:
                continue
            key = (
                record["method"],
                record["device"],
                record["case"],
                record["mode"],
                record["points"],
                record["duration_s"],
                record["h"] if x_key == "H" else None,
                record["H"] if x_key == "h" else None,
                record["amplitude"],
                record["nonuniform"],
            )
            groups.setdefault(key, []).append(record)
        result = []
        for group in sorted(groups, key=str):
            group_records = sorted(groups[group], key=lambda item: item[x_key])
            label_scope = (
                f"closed n={group_records[0]['points']}, 64 ps"
                if group_records[0]["case"] == "closed"
                else scope
            )
            result.append(
                _series(
                    f"{group_records[0]['method']} "
                    f"{group_records[0]['device']}: {label_scope}",
                    [record[x_key] for record in group_records],
                    [record[metric] for record in group_records],
                    marker=_candidate_marker(group_records[0], marker),
                    linestyle=(
                        "-"
                        if len(group_records) > 1 and x_key in {"H", "h"}
                        else "None"
                    ),
                )
            )
        return result

    all_metrics = (
        (
            "p_error",
            "Full P component error",
            "P",
            "Maximum normalized P error",
            "log",
        ),
        (
            "z_error",
            "Full Z component error",
            "Z",
            "Maximum normalized Z error",
            "log",
        ),
        (
            "target_velocity_amplitude_error",
            "Target velocity amplitude error",
            "velocity amplitude",
            "Absolute relative amplitude error",
            "symlog",
        ),
        (
            "target_velocity_phase_error",
            "Target velocity absolute phase error",
            "velocity phase",
            "Absolute wrapped phase error (rad)",
            "linear",
        ),
        (
            "target_magnetic_amplitude_error",
            "Target magnetic amplitude error",
            "magnetic amplitude",
            "Absolute relative amplitude error",
            "symlog",
        ),
        (
            "target_magnetic_phase_error",
            "Target magnetic absolute phase error",
            "magnetic phase",
            "Absolute wrapped phase error (rad)",
            "linear",
        ),
    )

    def target_panels(
        selected: list[dict[str, Any]],
        x_key: str,
        xlabel: str,
        scope: str,
        metric_names: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        panels = []
        for metric, title, quantity, ylabel, scale in all_metrics:
            if metric not in metric_names:
                continue
            marker = "s" if "Z" in quantity else "o"
            series = comparison_series(selected, metric, x_key, scope, marker)
            if series:
                panels.append(
                    _panel(
                        title,
                        xlabel,
                        ylabel,
                        series,
                        xscale="log" if x_key in {"H", "h"} else "linear",
                        yscale=scale,
                    )
                )
        return panels

    quarter_records = [
        record
        for record in usable
        if record["case"] == "periodic"
        and record["mode"] == 1
        and record["question"] == "fixed h H scan"
        and same(record["duration_s"], record["fixed_periodic_horizon_s"])
    ]
    panels = target_panels(
        quarter_records,
        "H",
        "Macro interval H (s)",
        "quarter-T fixed-h H scan",
        (
            "p_error",
            "z_error",
            "target_magnetic_amplitude_error",
            "target_magnetic_phase_error",
        ),
    )
    if panels:
        figures.append(
            _figure(
                "multirate-target-error",
                "Quarter-T fixed-h scan: full P/Z and magnetic target error",
                "mixed_comparison",
                panels,
                caption=(
                    "This figure contains only periodic mode-1 fixed-h H scans at "
                    "the common quarter-T duration. Full P/Z errors are component "
                    "maxima against stored scales. Magnetic target amplitude and "
                    "phase use mode_comparison.candidate/oracle magnetic_mode, not "
                    "the velocity-only target history. CPU circles and GPU triangles "
                    "distinguish overlapping backend curves. Closed, target-mode 4, "
                    "fixed-H half-h, and 2T horizon records are separate figures."
                ),
                status=(
                    "Completed candidate fields only; no same-accuracy speedup or "
                    "physical pass is inferred."
                ),
                device="C1/C2 and CPU/CUDA candidate series",
                question=(
                    "How do full and magnetic target errors vary with H in the "
                    "common quarter-T fixed-h scan?"
                ),
                controls=(
                    "Periodic mode 1, n=33, amplitude=0.02, nonuniform=0.1, and "
                    "the matrix fixed quarter-T duration."
                ),
                criterion=(
                    "Use stored component errors and connect only the ordered H scan."
                ),
                components=[
                    "multirate candidate",
                    "same-h synchronous reference",
                    "velocity and magnetic target modes",
                ],
            )
        )

    target4_records = [
        record
        for record in usable
        if record["case"] == "periodic" and "target" in record["id"].lower()
    ]
    panels = target_panels(
        target4_records,
        "H",
        "Macro interval H (s)",
        "target-mode control",
        tuple(metric[0] for metric in all_metrics),
    )
    if panels:
        figures.append(
            _figure(
                "multirate-target4-control",
                "Target-mode control: velocity and magnetic components",
                "mixed_comparison",
                panels,
                caption=(
                    "Target-mode 4 controls are shown independently from the "
                    "quarter-T mode-1 H scan. Velocity and magnetic amplitude/phase "
                    "errors use each component's stored candidate and synchronous "
                    "mode_comparison fields; a favorable velocity component does not "
                    "hide an unfavorable magnetic component."
                ),
                status="Component-resolved control only; no physical pass is inferred.",
                device="CPU and CUDA target-mode reports",
                question="Did the target-mode control preserve both velocity and magnetic components?",
                controls="Exact case, method, device, target mode, H, and duration remain separate.",
                criterion="Read both stored component errors without scalar-vector substitution.",
                components=[
                    "multirate target-mode candidate",
                    "synchronous target-mode reference",
                    "velocity and magnetic components",
                ],
            )
        )

    closed_records = [record for record in usable if record["case"] == "closed"]
    panels = target_panels(
        closed_records,
        "points",
        "Poloidal points (nodes)",
        "closed compatibility control",
        ("p_error", "z_error"),
    )
    if panels:
        figures.append(
            _figure(
                "multirate-closed-control",
                "Short closed-wall compatibility: full P/Z by nodes",
                "mixed_comparison",
                panels,
                caption=(
                    "Closed controls n=8 and n=12 cover only the stored 64 ps "
                    "compatibility window. They are not periodic quarter-T points, "
                    "wall-phase evidence, a long-time trajectory, or an order test."
                ),
                status="64 ps compatibility only; no production or physical claim.",
                device="CPU and CUDA closed controls",
                question="What full P/Z error accompanied each short closed control?",
                controls="Closed n=8/n=12 and 64 ps scope remain explicit.",
                criterion="Plot full P/Z against nodes without joining to periodic H scans.",
                components=[
                    "closed control candidate",
                    "synchronous closed reference",
                    "full P and Z components",
                ],
            )
        )

    pair_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in usable:
        if record["case"] != "periodic" or record["mode"] != 1:
            continue
        key = (
            record["method"],
            record["device"],
            record["points"],
            record["duration_s"],
            record["H"],
            record["amplitude"],
            record["nonuniform"],
        )
        pair_groups.setdefault(key, []).append(record)
    fixed_h_quarter_records = []
    two_t_records = []
    for group_records in pair_groups.values():
        if not any(record["question"] == "fixed H half h" for record in group_records):
            continue
        duration = group_records[0]["duration_s"]
        if same(duration, group_records[0]["fixed_periodic_horizon_s"]):
            fixed_h_quarter_records.extend(group_records)
        elif same(
            duration,
            (
                2.0 * group_records[0]["target_period_s"]
                if group_records[0]["target_period_s"] is not None
                else None
            ),
        ):
            two_t_records.extend(group_records)

    panels = target_panels(
        fixed_h_quarter_records,
        "h",
        "Microstep h (s)",
        "fixed-H h and h/2 control",
        tuple(metric[0] for metric in all_metrics),
    )
    if panels:
        figures.append(
            _figure(
                "multirate-fixed-h-halfh",
                "Fixed-H microstep pair: errors versus h",
                "mixed_comparison",
                panels,
                caption=(
                    "Each connected pair keeps H and the quarter-T horizon fixed while "
                    "plotting h and h/2 for C1/C2. The two-point display is a sensitivity "
                    "control and is not an empirical convergence-order estimate."
                ),
                status="Fixed-H sensitivity control; no order claim.",
                device="CPU and CUDA fixed-H pairs",
                question="How do full and target component errors change when h is halved at fixed H?",
                controls="Only a base and its same-H half-h sibling share a curve.",
                criterion="Use actual h from each controlled report; do not infer a slope.",
                components=[
                    "base microstep candidate",
                    "same-H half-h candidate",
                    "full P/Z and target components",
                ],
            )
        )

    panels = target_panels(
        two_t_records,
        "h",
        "Microstep h (s)",
        "2T horizon fixed-H h and h/2",
        tuple(metric[0] for metric in all_metrics),
    )
    if panels:
        figures.append(
            _figure(
                "multirate-2t-horizon",
                "2T horizon control: errors versus h",
                "mixed_comparison",
                panels,
                caption=(
                    "These curves are the separate C2 2T-horizon base and fixed-H "
                    "half-h pair. They are not part of the quarter-T mode-1 H scan; "
                    "the horizon label is retained explicitly."
                ),
                status="2T horizon control only; no convergence-order or full-run claim.",
                device="C2 CPU and CUDA records",
                question="What component errors were stored for the separate 2T horizon pair?",
                controls="Only C2 mode 1 at the 2T duration, with H fixed across h and h/2.",
                criterion="Use the stored 2T duration and h pair; do not merge with quarter-T.",
                components=[
                    "C2 2T base candidate",
                    "fixed-H half-h candidate",
                    "full P/Z and target components",
                ],
            )
        )

    # Backend-only long runs need no same-backend h/2 sibling to retain work.
    for record in usable:
        if (
            record["case"] == "periodic"
            and record["mode"] == 1
            and record["target_period_s"] is not None
            and same(record["duration_s"], 2 * record["target_period_s"])
            and record not in two_t_records
        ):
            two_t_records.append(record)
    families = [
        ("main", "Fixed-h H scan", "H", "Macro interval H (s)", quarter_records),
        (
            "half-h",
            "Fixed-H h and h/2",
            "h",
            "Microstep h (s)",
            fixed_h_quarter_records,
        ),
        ("target", "Target-mode control", "H", "Macro interval H (s)", target4_records),
        (
            "closed",
            "Short closed-wall control",
            "points",
            "Enclosing R/Z nodes per axis",
            closed_records,
        ),
        ("2t", "Separate 2T horizon", "h", "Microstep h (s)", two_t_records),
    ]
    covered = {
        (record["root"], record["id"])
        for *_, selected in families
        for record in selected
    }
    for index, record in enumerate(usable):
        if (record["root"], record["id"]) not in covered:
            families.append(
                (
                    f"other-{index}",
                    record["question"],
                    "H",
                    "Macro interval H (s)",
                    [record],
                )
            )
    for family_id, title, x_key, xlabel, selected in families:
        if not selected:
            continue
        duration = ", ".join(
            f"{value:.6g} s" for value in sorted({r["duration_s"] for r in selected})
        )
        scope = (
            "fixed-h H scan" if family_id == "main" else title
        ) + f", duration={duration}"
        control = (
            f"Actual observation duration: {duration}. Method/backend, grid, mode, "
            "amplitude, background and fixed h/H controls remain separate curves."
        )
        gain_panels = []
        for metric, metric_title, marker in (
            ("sentinel_candidate_max_gain", "Candidate high-mode maximum gain", "o"),
            ("sentinel_sync_max_gain", "Synchronous high-mode maximum gain", "s"),
            ("sentinel_candidate_final_gain", "Candidate high-mode final gain", "o"),
            ("sentinel_sync_final_gain", "Synchronous high-mode final gain", "s"),
        ):
            series = comparison_series(selected, metric, x_key, scope, marker)
            if series:
                gain_panels.append(
                    _panel(
                        metric_title,
                        xlabel,
                        "Reported mode amplitude gain",
                        series,
                        xscale="log" if x_key in {"H", "h"} else "linear",
                    )
                )
        if gain_panels:
            figures.append(
                _figure(
                    "multirate-mode-gain"
                    + ("" if family_id == "main" else "-" + family_id),
                    title + ": high-mode gains",
                    "mixed_comparison",
                    gain_panels,
                    caption=(
                        "Candidate and synchronous gains are separate stored observations. "
                        "Nonuniform nonlinear physical gain is not automatically numerical growth. "
                        "The target period T and this observation duration are distinct; "
                        + control
                    ),
                    status="Descriptive gain controls, not a physical qualification.",
                    device="Separate CPU and CUDA curves",
                    question="How does the candidate high-mode response compare with its synchronous response?",
                    controls=control,
                    criterion="Use each recorded gain; do not mix physical horizons.",
                    components=[
                        "candidate high-mode gain",
                        "synchronous high-mode gain",
                    ],
                )
            )
        work_panels = []
        for metric, metric_title, quantity, marker in (
            ("full_hydro_eval", "Complete transport evaluations", "Full hydro", "o"),
            ("cheap_D", "Density response evaluations", "Cheap D", "s"),
            ("cheap_L", "Lorentz response evaluations", "Cheap L", "^"),
            ("field_rhs", "Electron/field RHS evaluations", "Field RHS", "D"),
        ):
            series = comparison_series(
                selected, metric, x_key, scope + " " + quantity, marker
            )
            if series:
                work_panels.append(
                    _panel(
                        metric_title,
                        xlabel,
                        "Counted operations",
                        series,
                        xscale="log" if x_key in {"H", "h"} else "linear",
                        yscale="log",
                    )
                )
        if work_panels:
            figures.append(
                _figure(
                    "multirate-work-" + family_id,
                    title + ": counted work",
                    "mixed_comparison",
                    work_panels,
                    caption=(
                        "Actual returned-execution counters, including charged endpoint debug work. "
                        "Campaign cold/warmup/repetition multipliers are not applied to this count. "
                        "No full-ion or FK saving is inferred from transport counts alone. "
                        + control
                    ),
                    status="Operation counts only; no performance or particle-operation claim.",
                    device="Separate CPU and CUDA curves",
                    question="Which operations remain after reducing complete transport nodes?",
                    controls=control,
                    criterion="Retain every available returned-execution count without normalization.",
                    components=[
                        "complete transport",
                        "density/Lorentz response",
                        "field RHS",
                    ],
                )
            )
    if not figures:
        notes.append(
            "Multirate candidate comparisons: unavailable. No completed candidate "
            "report with the controlled matrix schema is present."
        )
    elif any(record.get("campaign_multiplier") not in (None, 1.0) for record in usable):
        notes.append(
            "Candidate returned-execution counts are not divided or multiplied by "
            "the recorded cold/warmup/sample campaign multiplier."
        )
    return figures, notes


def _scan_fair_costs(
    run_root: Path, inputs: InputIndex
) -> tuple[list[dict[str, Any]], list[str]]:
    matrix_path = run_root / MATRIX_FILE
    if not matrix_path.is_file():
        return [], [
            (
                "Fair primitive cost: unavailable. Controlled matrix "
                "multirate-matrix.json is absent."
            )
        ]
    matrix = inputs.json(matrix_path, "controlled multirate matrix")
    rows = [row for row in matrix.get("rows", []) if isinstance(row, dict)]
    if not rows:
        return [], [
            "Fair primitive cost: unavailable. The controlled matrix contains no rows."
        ]

    records = []
    notes = []
    for root_name, device, protocol_device in FAIR_COST_ROOTS:
        root = run_root / root_name
        missing = []
        completed = 0
        for row in rows:
            row_id = str(row.get("id") or "unrecorded")
            directory = root / row_id
            report_path = directory / "report.json"
            source_input_path = directory / "source_input.json"
            if not report_path.is_file():
                missing.append(row_id)
                continue
            try:
                report = inputs.json(
                    report_path, f"fair primitive cost {root_name}/{row_id}"
                )
                if source_input_path.is_file():
                    inputs.file(
                        source_input_path,
                        f"fair cost source input {root_name}/{row_id}",
                    )
                arrays_path = directory / "arrays.npz"
                if arrays_path.is_file():
                    inputs.file(
                        arrays_path,
                        f"fair cost arrays {root_name}/{row_id}",
                    )
                traceback_path = directory / "traceback.txt"
                if traceback_path.is_file():
                    inputs.file(
                        traceback_path,
                        f"fair cost traceback {root_name}/{row_id}",
                    )
                status = str(report.get("status", "unknown"))
                failure = _mapping(report.get("failure"))
                if status != "completed":
                    notes.append(
                        f"Fair primitive cost {device}/{row_id}: {status}; "
                        f"failure/reason {failure.get('code') or report.get('failure_code') or 'not recorded'}. "
                        "No numeric point is plotted."
                    )
                    continue
                if not source_input_path.is_file():
                    raise InputError("source_input.json is absent")
                source_input = inputs.json(
                    source_input_path, f"fair cost source input {root_name}/{row_id}"
                )
                source_config = _mapping(_nested(report, "source_input", "config"))
                if source_input != source_config:
                    raise InputError(
                        "source_input.json does not match source_input.config"
                    )
                _validate_matrix_row(
                    row,
                    source_config,
                    source=f"{root_name}/{row_id}/source_input.json",
                )
                candidate_root = (
                    "candidates-cpu-v2" if device == "CPU" else "candidates-gpu"
                )
                candidate_input_path = run_root / candidate_root / row_id / "input.json"
                if candidate_input_path.is_file():
                    candidate_input = inputs.json(
                        candidate_input_path,
                        f"fair cost candidate input {candidate_root}/{row_id}",
                    )
                    non_algorithm_keys = {
                        "device",
                        "reference",
                        "warmups",
                        "repetitions",
                    }
                    if {
                        k: v
                        for k, v in candidate_input.items()
                        if k not in non_algorithm_keys
                    } != {
                        k: v
                        for k, v in source_config.items()
                        if k not in non_algorithm_keys
                    }:
                        raise InputError(
                            "candidate input.json does not match fair source input"
                        )
                protocol = _mapping(report.get("protocol"))
                actual_config = _mapping(
                    _nested(report, "source_input", "actual_config")
                )
                if (
                    actual_config.get("device") != protocol_device
                    or actual_config.get("warmups") != protocol.get("warmups")
                    or actual_config.get("repetitions") != protocol.get("repetitions")
                    or {
                        k: v
                        for k, v in actual_config.items()
                        if k not in {"device", "warmups", "repetitions"}
                    }
                    != {
                        k: v
                        for k, v in source_config.items()
                        if k not in {"device", "warmups", "repetitions"}
                    }
                ):
                    raise InputError(
                        "actual config backend/protocol or scientific controls mismatch"
                    )
                h = _number(row.get("h"))
                M = _integer(row.get("M"))
                macros = _integer(row.get("macros"))
                if h is None or M is None or macros is None:
                    raise InputError("candidate matrix row is incomplete")
                expected_horizon = h * M * macros
                checks = {
                    "actual_h_s": math.isclose(
                        _number(protocol.get("actual_h_s")) or math.nan,
                        h,
                        rel_tol=1e-12,
                        abs_tol=1e-30,
                    ),
                    "actual_H_s": math.isclose(
                        _number(protocol.get("actual_H_s")) or math.nan,
                        h * M,
                        rel_tol=1e-12,
                        abs_tol=1e-30,
                    ),
                    "actual_M": _integer(protocol.get("actual_M")) == M,
                    "actual_macros": _integer(protocol.get("actual_macros")) == macros,
                    "actual_horizon_s": math.isclose(
                        _number(protocol.get("actual_horizon_s")) or math.nan,
                        expected_horizon,
                        rel_tol=1e-12,
                        abs_tol=1e-30,
                    ),
                    "device": protocol.get("device") == protocol_device,
                    "algorithm_controls_unchanged": protocol.get(
                        "algorithm_controls_unchanged"
                    )
                    is True,
                    "horizon_unchanged": protocol.get("horizon_unchanged") is True,
                }
                failed_checks = [name for name, passed in checks.items() if not passed]
                if failed_checks:
                    raise InputError("protocol mismatch: " + ", ".join(failed_checks))
                if _integer(protocol.get("warmups")) != 1:
                    raise InputError(f"warmups={protocol.get('warmups')}, expected 1")
                if _integer(protocol.get("repetitions")) != 3:
                    raise InputError(
                        f"repetitions={protocol.get('repetitions')}, expected 3"
                    )
                diagnostics = _mapping(report.get("diagnostics"))
                if "pre" not in diagnostics or "post" not in diagnostics:
                    raise InputError("matched diagnostics pre/post scope is absent")
                matched = _mapping(report.get("matched_cost"))
                candidate_timing = _mapping(matched.get("candidate"))
                lean_timing = _mapping(matched.get("lean_reference"))
                if matched.get("same_protocol") is not True:
                    raise InputError("matched_cost.same_protocol is not true")
                if matched.get("speedup_claim") is not False:
                    raise InputError("matched_cost.speedup_claim is not false")
                for role, timing in (
                    ("candidate", candidate_timing),
                    ("lean_reference", lean_timing),
                ):
                    if (
                        _integer(timing.get("warmups")) != 1
                        or _integer(timing.get("repetitions")) != 3
                        or _integer(timing.get("cold_executions")) != 1
                    ):
                        raise InputError(
                            f"matched_cost.{role} is not cold 1, warmup 1, 3 samples"
                        )
                counters = _mapping(report.get("counters"))
                candidate_counters = _mapping(counters.get("candidate"))
                lean_counters = _mapping(counters.get("lean"))
                record = {
                    "id": row_id,
                    "root": root_name,
                    "device": device,
                    "method": str(row.get("method")),
                    "case": str(row.get("case")),
                    "h": h,
                    "H": h * M,
                    "M": M,
                    "macros": macros,
                    "duration_s": expected_horizon,
                    "candidate_median_s": _number(candidate_timing.get("median_s")),
                    "lean_median_s": _number(lean_timing.get("median_s")),
                    "candidate_full_hydro": _number(
                        candidate_counters.get("full_hydro_eval")
                    ),
                    "lean_full_hydro": _number(lean_counters.get("full_hydro_eval")),
                    "candidate_field_rhs": _number(candidate_counters.get("field_rhs")),
                    "lean_field_rhs": _number(lean_counters.get("field_rhs")),
                }
                required = (
                    "candidate_median_s",
                    "lean_median_s",
                    "candidate_full_hydro",
                    "lean_full_hydro",
                    "candidate_field_rhs",
                    "lean_field_rhs",
                )
                invalid = [
                    name
                    for name in required
                    if record[name] is None
                    or (name.endswith("_s") and record[name] <= 0.0)
                    or (not name.endswith("_s") and record[name] < 0.0)
                ]
                if invalid:
                    raise InputError(
                        "non-finite or invalid measured fields: " + ", ".join(invalid)
                    )
                records.append(record)
                completed += 1
            except InputError as error:
                notes.append(
                    f"Fair primitive cost {device}/{row_id}: invalid; {error}. "
                    "No numeric point is plotted."
                )
        if missing:
            notes.append(
                f"Fair primitive cost {device}: {completed} completed finite of "
                f"{len(rows)}; missing report.json for {', '.join(missing)}. "
                "No numeric points are interpolated."
            )
        else:
            notes.append(
                f"Fair primitive cost {device}: {completed} completed finite of "
                f"{len(rows)}."
            )
    return records, notes


def _fair_cost_figure(
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not records:
        return None
    case_ids = sorted({str(record["id"]) for record in records})
    case_index = {case_id: index for index, case_id in enumerate(case_ids)}
    panels = []
    for title, quantity, metrics in (
        (
            "Candidate and lean median research-window time",
            "median",
            ("candidate_median_s", "lean_median_s"),
        ),
        (
            "Full hydro evaluations",
            "full hydro",
            ("candidate_full_hydro", "lean_full_hydro"),
        ),
        (
            "Field RHS calls",
            "field RHS",
            ("candidate_field_rhs", "lean_field_rhs"),
        ),
    ):
        series = []
        for device in ("CPU", "GPU"):
            for method in ("C1", "C2"):
                for metric, role in zip(metrics, ("candidate", "lean")):
                    selected = [
                        record
                        for record in records
                        if record["device"] == device and record["method"] == method
                    ]
                    if not selected:
                        continue
                    series.append(
                        _series(
                            f"{device} {method} {role} {quantity}",
                            [float(case_index[record["id"]]) for record in selected],
                            [record[metric] for record in selected],
                            marker=(
                                ("^" if role == "candidate" else "D")
                                if device == "GPU"
                                else ("o" if role == "candidate" else "s")
                            ),
                            linestyle="None",
                        )
                    )
        if series:
            panels.append(
                _panel(
                    title,
                    "Case index (mapping in caption)",
                    (
                        "Median synchronized research window (s)"
                        if quantity == "median"
                        else "Actual returned-execution count"
                    ),
                    series,
                    yscale="log",
                )
            )
    if not panels:
        return None
    mapping = ", ".join(f"{index}={case_id}" for index, case_id in enumerate(case_ids))
    return _figure(
        "fair-primitive-cost",
        "Fair candidate versus lean synchronous primitive cost",
        "mixed_comparison",
        panels,
        caption=(
            f"Case index: {mapping}. Each point is one fair-cost report generated "
            "by warp_multirate_cost.py from the same built fixture, model, "
            "primitives, diagnostics, horizon, h, M, and macros. Reports require "
            "cold 1, warmup 1, 3 samples and exact candidate/lean protocol identity. "
            "Candidate and lean full-hydro and field-RHS counters are the actual "
            "returned counts. CPU and GPU series remain separate and are not "
            "interpreted as a hardware ratio or algorithm gain. This is a bounded "
            "research-window measurement, not a production transaction, ledger, "
            "restart, checkpoint, or FK solver result, and it is not an FK "
            "acceleration claim."
        ),
        status=(
            "Matched research-window measurements only; not a production "
            "transaction or FK acceleration claim."
        ),
        device="Separate CPU and CUDA fair-cost reports",
        question=(
            "What measured candidate and lean synchronous research-window medians "
            "and actual full-hydro/field counts were observed?"
        ),
        controls=(
            "Exact source input, fixture, model, primitives, diagnostics, horizon, "
            "h, M, macros, cold/warmup/sample protocol, and device are required to "
            "match within each report."
        ),
        criterion=(
            "Plot only finite completed reports with same_protocol=true; marker-only "
            "case points are not connected."
        ),
        components=[
            "candidate measured complete research window",
            "lean synchronous measured complete research window",
            "actual full-hydro and field-RHS counts",
        ],
    )


def _verified_long_groups(run_root: Path, inputs: InputIndex) -> dict[str, str]:
    analysis_path = run_root / "analysis/long-rz/analysis.json"
    receipt_path = run_root / "analysis/long-rz/VERIFY_CURRENT.json"
    if not analysis_path.is_file() or not receipt_path.is_file():
        return {}
    receipt = inputs.json(receipt_path, "independent continuation identity receipt")
    analysis = inputs.json(analysis_path, "independent continuation identity analysis")
    if analysis.get("root") != str(run_root.resolve()):
        raise InputError("Unverified long continuation analysis root")
    expected = _nested(receipt, "outputs", analysis_path.name, "sha256")
    if (
        receipt.get("read_issues")
        or analysis.get("read_issues")
        or expected != sha256_file(analysis_path)
    ):
        raise InputError("Unverified long continuation analysis")
    for path, metadata in receipt.get("inputs", {}).items():
        raw = Path(path)
        inputs.file(raw, "verified continuation raw dependency")
        if sha256_file(raw) != metadata.get("sha256"):
            raise InputError(f"Unverified long continuation dependency: {raw}")
    groups = {}
    for name, chain in analysis.get("chains", {}).items():
        if chain.get("integrity_errors") != []:
            raise InputError(f"Unverified long continuation chain: {name}")
        for segment in chain.get("segments", []):
            if segment in groups:
                raise InputError(f"Ambiguous long continuation segment: {segment}")
            groups[segment] = name
    return groups


def _long_plot_method(record: dict[str, Any]) -> str:
    if record["coupling"] == "waveform":
        return "T1"
    return "A0 h/2" if "halfh" in record["id"] else "A0 h"


def _long_figure(
    long_reports: list[dict[str, Any]], verified_groups: dict[str, str] | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    cohorts: dict[str, list[str]] = {}
    for record in long_reports:
        if any(
            record[key]
            for key in (
                "energy_points",
                "control_energy_points",
                "matched_response_points",
            )
        ):
            cohorts.setdefault(_long_plot_method(record), []).append(record["id"])
    for ids in cohorts.values():
        if len(ids) > 1:
            declared = {(verified_groups or {}).get(name) for name in ids}
            if None in declared or len(declared) != 1:
                raise InputError("Unverified independent long records cannot be joined")
    energy_series = []
    balance_series = {"control": [], "perturbed": []}
    magnetic_response_series = []
    velocity_response_series = []
    for record in long_reports:
        method = _long_plot_method(record)
        for ensemble, key in (
            ("control", "control_energy_points"),
            ("perturbed", "energy_points"),
        ):
            points = list(record[key])
            if len(points) < 2:
                continue
            label = f"{method} {ensemble}"
            energy_series.append(
                _series(
                    label,
                    [point["time_s"] for point in points],
                    [point["total_delta_J"] for point in points],
                    marker="o",
                )
            )
            for quantity, component, marker in (
                ("balance", "balance_J", "o"),
                ("outward", "outward_J", "s"),
                ("source", "source_J", "^"),
            ):
                if all(point.get(component) is not None for point in points):
                    balance_series[ensemble].append(
                        _series(
                            f"{method} {ensemble} {quantity}",
                            [point["time_s"] for point in points],
                            [point[component] for point in points],
                            marker=marker,
                        )
                    )
        response_points = record["matched_response_points"]
        if response_points:
            magnetic_response_series.append(
                _series(
                    f"{method} matched",
                    [point["time_s"] for point in response_points],
                    [point["rms_delta_B_T"] for point in response_points],
                    marker="s",
                )
            )
            velocity_response_series.append(
                _series(
                    f"{method} matched",
                    [point["time_s"] for point in response_points],
                    [point["rms_delta_v_m_s"] for point in response_points],
                    marker="^",
                )
            )
    energy_series = _join_plot_series(energy_series)
    balance_series = {
        key: _join_plot_series(value) for key, value in balance_series.items()
    }
    magnetic_response_series = _join_plot_series(magnetic_response_series)
    velocity_response_series = _join_plot_series(velocity_response_series)
    panels = []
    if energy_series:
        panels.append(
            _panel(
                "Raw energy minus original archived baseline",
                "Physical time (s)",
                "Energy change (J)",
                energy_series,
                yscale="symlog",
            )
        )
    if balance_series["control"]:
        panels.append(
            _panel(
                "Control energy balance and transfers",
                "Physical time (s)",
                "Energy (J)",
                balance_series["control"],
                yscale="symlog",
            )
        )
    if balance_series["perturbed"]:
        panels.append(
            _panel(
                "Perturbed energy balance and transfers",
                "Physical time (s)",
                "Energy (J)",
                balance_series["perturbed"],
                yscale="symlog",
            )
        )
    if magnetic_response_series:
        panels.append(
            _panel(
                "Matched RMS magnetic response",
                "Physical time (s)",
                "RMS control-perturbed observed_BN (T)",
                magnetic_response_series,
                yscale="log",
            )
        )
    if velocity_response_series:
        panels.append(
            _panel(
                "Matched RMS velocity response",
                "Physical time (s)",
                "RMS control-perturbed observed_velocity (m/s)",
                velocity_response_series,
                yscale="log",
            )
        )
    if not panels:
        return None, (
            "Long-RZ diagnostics: unavailable. No long-east report exposes a "
            "two-point stored prefix with physical time and its original archived "
            "baseline."
        )
    return (
        _figure(
            "long-rz-energy",
            "Long-RZ accepted prefixes: energy accounting and matched response",
            "nonlinear_evolution",
            panels,
            caption=(
                "Control and perturbed histories use their stored global time and "
                "step. Raw energy is total_J minus each record's original "
                "baseline_total_J; no baseline is reset at a segment start. The "
                "balance panels retain stored balance_J, outward_J, and source_J "
                "separately. Matched response is the absolute RMS difference between "
                "control and perturbed observed_BN (T) and observed_velocity (m/s) "
                "snapshots at the same physical step/time; no first-segment response "
                "is subtracted and no physics is recomputed. Budget-stopped and failed "
                "reports contribute only stored accepted prefixes, never a full-horizon "
                "claim. Full segment identities and report hashes: "
                + "; ".join(
                    f"{record['id']}: run_id={record['run_id'] or 'not recorded'}, "
                    f"start_step={record['segment_start_step']}, h={record['h_s']} s, "
                    f"coupling={record['coupling'] or 'not recorded'}, "
                    f"status={record['status']}, "
                    f"sha256={record['report_sha256']}"
                    for record in long_reports
                )
                + "."
            ),
            status=(
                "Long-RZ accepted prefixes/segments only; raw diagnostics are not "
                "a production ledger."
            ),
            device="Reported long-RZ device",
            question=(
                "What raw long-RZ diagnostics are available within each accepted "
                "prefix or continuation segment?"
            ),
            controls=(
                "Exact run_id, start_step, h, coupling, and ensemble remain separate; "
                "budget_stopped does not mean the requested full horizon completed. "
                "Plot lines join matching method/h/ensemble accepted segments at global time; raw report and snapshot paths "
                "remain in input-index.json."
            ),
            criterion=(
                "Require stored global times, original archived baselines, and "
                "same-step control/perturbed snapshot pairs."
            ),
        ),
        None,
    )


def _full_cost(run_root: Path, inputs: InputIndex) -> tuple[dict[str, Any] | None, str]:
    baseline_path = run_root / "full-cost-baseline/RESULT.json"
    optimized_path = run_root / "full-cost-optimized/RESULT.json"
    if not baseline_path.is_file() and not optimized_path.is_file():
        return None, (
            "Shared matched full cost: unavailable. Baseline and optimized "
            "harness RESULT.json artifacts are absent."
        )
    by_role: dict[str, dict[str, Any]] = {}
    if baseline_path.is_file():
        by_role["baseline"] = inputs.json(
            baseline_path, "shared full-cost baseline result"
        )
    if optimized_path.is_file():
        by_role["optimized"] = inputs.json(
            optimized_path, "shared full-cost optimized result"
        )
    if set(by_role) != {"baseline", "optimized"}:
        return None, (
            "Shared matched full cost: unavailable. Both baseline and optimized "
            "RESULT.json artifacts are required."
        )
    for role, result in by_role.items():
        if result.get("schema") != "fomhd-bounded-full-advance-cost-v1":
            return None, (
                f"Shared matched full cost: unavailable. {role} RESULT.json does "
                "not use the generated harness schema."
            )
        if result.get("status") != "completed":
            return None, (
                f"Shared matched full cost: unavailable. {role} RESULT.json "
                f"status is {result.get('status', 'unknown')}, not completed."
            )
    method_data = {}
    for role, result in by_role.items():
        methods = _mapping(result.get("methods"))
        if set(methods) != {"A0", "T1"}:
            return None, (
                f"Shared matched full cost: unavailable. {role} RESULT.json "
                "must contain exact A0 and T1 method summaries."
            )
        method_data[role] = {}
        for method in ("A0", "T1"):
            summary = _mapping(methods.get(method))
            median_ms = _number(_nested(summary, "measured_window_ms", "median_ms"))
            cold_ms = _number(summary.get("cold_total_ms"))
            count = _integer(_nested(summary, "measured_window_ms", "count"))
            if median_ms is None or count is None or count <= 0:
                return None, (
                    f"Shared matched full cost: unavailable. {role} {method} lacks "
                    "a completed measured-window median."
                )
            method_data[role][method] = {
                "median_ms": median_ms,
                "cold_ms": cold_ms,
                "plan": _mapping(summary.get("plan")),
            }
    plan_keys = (
        "coupling",
        "micro_steps",
        "h_s",
        "dt_s",
        "macros",
        "horizon_s",
    )
    for method in ("A0", "T1"):
        baseline_plan = method_data["baseline"][method]["plan"]
        optimized_plan = method_data["optimized"][method]["plan"]
        if any(baseline_plan.get(key) != optimized_plan.get(key) for key in plan_keys):
            return None, (
                f"Shared matched full cost: unavailable. Baseline/optimized "
                f"{method} plans do not match exactly."
            )
    baseline_scales = by_role["baseline"].get("physical_scales")
    optimized_scales = by_role["optimized"].get("physical_scales")
    if (
        baseline_scales is not None
        and optimized_scales is not None
        and baseline_scales != optimized_scales
    ):
        return None, (
            "Shared matched full cost: unavailable. Baseline/optimized physical "
            "scales do not match."
        )
    panels = []
    for key, title, ylabel in (
        (
            "median_ms",
            "Measured full-advance window",
            "Median synchronized full window (ms)",
        ),
        (
            "cold_ms",
            "Cold total including state copy and initialization",
            "Cold total elapsed time (ms)",
        ),
    ):
        series = []
        for role in ("baseline", "optimized"):
            values = [method_data[role][method][key] for method in ("A0", "T1")]
            if any(value is None for value in values):
                continue
            series.append(
                _series(
                    f"{role} source",
                    [0.0, 1.0],
                    values,
                    marker="o" if role == "baseline" else "s",
                    linestyle="None",
                )
            )
        if series:
            panels.append(
                _panel(
                    title,
                    "Method index (A0=0, T1=1)",
                    ylabel,
                    series,
                    yscale="log",
                )
            )
    if not panels:
        return None, (
            "Shared matched full cost: unavailable. Exact RESULT.json artifacts "
            "do not expose finite complete-advance cost fields."
        )
    return (
        _figure(
            "shared-complete-cost",
            "Shared matched complete-advance cost",
            "mixed_comparison",
            panels,
            caption=(
                "Values come from the generated harness RESULT.json schema: "
                "methods.{A0,T1}.measured_window_ms.median_ms and cold_total_ms. "
                "A0 and T1 plans match exactly across baseline/optimized sources. "
                "The figure does not establish a same-accuracy speedup."
            ),
            status="Matched source-pair cost only; no speedup claim.",
            device="Runtime selected by the full-advance harness",
            question="What measured full-window and cold costs were observed for A0 and T1?",
            controls=(
                "Same method plan, horizon, h, and physical scales in separate "
                "baseline and optimized source runs."
            ),
            criterion="Require completed RESULT.json artifacts with measured-window counts.",
            components=[
                "baseline source complete advance",
                "optimized source complete advance",
                "A0 and T1 methods",
            ],
        ),
        "",
    )


def _backend_parity_figure(run_root: Path, inputs: InputIndex) -> dict[str, Any] | None:
    path = run_root / "analysis/multirate/summary.json"
    if not path.is_file():
        return None
    summary = inputs.json(path, "independent same-algorithm CPU/GPU analysis")
    for raw_path, expected in summary.get("input_file_sha256", {}).items():
        raw = Path(raw_path)
        if not raw.is_absolute():
            raw = run_root / raw
        inputs.file(raw, "independent backend analysis raw dependency")
        if sha256_file(raw) != expected:
            raise InputError(f"Backend summary dependency changed: {raw}")
    rows = [
        row
        for row in summary.get("CPU_GPU_parity", [])
        if all(
            _nested(row, "endpoints", field, "status") == "compared"
            for field in ("candidate_p", "candidate_z")
        )
    ]
    if not rows:
        return None
    series = [
        _series(
            label,
            list(range(len(rows))),
            [row["endpoints"][field]["maximum_normalized"] for row in rows],
            marker=marker,
            linestyle="None",
        )
        for field, label, marker in (
            ("candidate_p", "Ion state", "o"),
            ("candidate_z", "Electron/field state", "s"),
        )
    ]
    return _figure(
        "multirate-backend-parity",
        "Same-algorithm CPU/GPU endpoint differences",
        "verification",
        [
            _panel(
                "Physical-component scaled endpoint difference",
                "Case index",
                "Maximum normalized difference",
                series,
                yscale="log",
            )
        ],
        caption="; ".join(f"{i}: {row['case_id']}" for i, row in enumerate(rows))
        + ". Component scales and complete final-stage checks are retained in the independent summary; input roundoff is measured separately.",
        status="Backend verification only; no timing or physical qualification claim.",
        device="CPU and CUDA",
        question="Does the GPU reproduce the same candidate algorithm?",
        controls="Same physical parameters, discrete candidate, precision and final time.",
        criterion="Use physical-scale tolerances; do not require bitwise equality or identify A0 with analytic truth.",
    )


def collect_gallery_data(run_root: Path, source_root: Path) -> dict[str, Any]:
    """Rescan the bounded run root and return plot-ready reviewed data."""
    run_root = run_root.resolve()
    source_root = source_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"run root is not a directory: {run_root}")
    inputs = InputIndex()
    complete_runs, long_reports, availability = _scan_runs(run_root, inputs)
    figures = []
    cost = _cost_figure(complete_runs)
    if cost is not None:
        figures.append(cost)
    energy = _energy_figure(complete_runs)
    if energy is not None:
        figures.append(energy)
    guesses, guess_pairs, guess_notes = _scan_initial_guesses(run_root, inputs)
    availability.extend(guess_notes)
    figures.extend(_guess_figures(guesses, guess_pairs))
    candidates, candidate_notes = _scan_candidates(run_root, inputs)
    availability.extend(candidate_notes)
    candidate_figures, candidate_availability = _candidate_figures(candidates)
    figures.extend(candidate_figures)
    availability.extend(candidate_availability)
    parity = _backend_parity_figure(run_root, inputs)
    if parity is not None:
        figures.append(parity)
    fair_costs, fair_cost_notes = _scan_fair_costs(run_root, inputs)
    availability.extend(fair_cost_notes)
    fair_cost_figure = _fair_cost_figure(fair_costs)
    if fair_cost_figure is not None:
        figures.append(fair_cost_figure)
    long_figure, long_note = _long_figure(
        long_reports, _verified_long_groups(run_root, inputs)
    )
    if long_figure is not None:
        figures.append(long_figure)
    if long_note is not None:
        availability.append(long_note)
    full_cost_figure, full_cost_note = _full_cost(run_root, inputs)
    if full_cost_figure is not None:
        figures.append(full_cost_figure)
    if full_cost_note:
        availability.append(full_cost_note)
    availability.append(
        "Physical qualification: not inferred from return code, timing, or status."
    )
    hall_path = run_root / "analysis/hall-control/report.json"
    if hall_path.is_file():
        hall = inputs.json(hall_path, "analytic Hall control report")
        availability.append(
            "Evidence types remain separate: analysis/hall-control/report.json is "
            f"an analytic linear control ({hall.get('scope') or 'scope not recorded'}); "
            "nonlinear-energy-field and long-rz-energy diagnostic figures "
            "are nonlinear evolution records, not analytic controls."
        )
    for name in CANONICAL_FILES:
        path = source_root / name
        if path.is_file():
            inputs.file(path, "canonical renderer or style source")
    if not figures:
        raise InputError("no plot-ready completed source section is available")
    figures = _bounded_figures(figures)
    evidence_types: dict[str, list[str]] = {}
    for figure in figures:
        evidence_types.setdefault(str(figure["source_kind"]), []).append(
            str(figure["id"])
        )
    return {
        "title": "Bounded FOMHD 0.5.0 and multirate evidence gallery",
        "introduction": (
            "Offline figures generated by rescanning the supplied run root. Every "
            "numeric series comes from an exact hashed input; unavailable sections "
            "are listed without placeholder values."
        ),
        "conclusions": (
            "This gallery is a postprocessing view of available artifacts. It does "
            "not establish a physical pass, coupled T1 acceptance, same-accuracy "
            "speedup, or production restart."
        ),
        "archive": {
            "label": "Run root",
            "path": str(run_root),
        },
        "notation": [
            {"symbol": "H", "meaning": "Macro coupling interval."},
            {"symbol": "h", "meaning": "Micro integration step."},
            {"symbol": "NL", "meaning": "Nonlinear evolution source badge."},
            {"symbol": "VER", "meaning": "Focused verification source badge."},
            {"symbol": "MIX", "meaning": "Explicit components shown together."},
            {
                "symbol": "max|B_component|",
                "meaning": "Maximum absolute stored magnetic component/coefficient; not vector-field magnitude.",
            },
        ],
        "known_issues": [],
        "figures": figures,
        "evidence_types": evidence_types,
        "availability_notes": availability,
        "input_index": inputs.records(),
        "source_roots": {
            "run_root": str(run_root),
            "source_root": str(source_root),
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_output_target(output: Path, *, run_root: Path, source_root: Path) -> bool:
    """Return whether an existing controlled gallery must be replaced."""
    output = output.resolve()
    run_root = run_root.resolve()
    source_root = source_root.resolve()
    if (
        output == run_root
        or output.is_relative_to(run_root)
        or run_root.is_relative_to(output)
    ):
        raise UnsafeOutputError("output must be outside the run root")
    if (
        output == source_root
        or output.is_relative_to(source_root)
        or source_root.is_relative_to(output)
    ):
        raise UnsafeOutputError("output must be outside the canonical source root")
    if not output.exists():
        return False
    if not output.is_dir():
        raise UnsafeOutputError(f"existing output is not a directory: {output}")
    marker = output / MARKER_NAME
    if not marker.is_file():
        raise UnsafeOutputError(
            f"refusing to replace an existing uncontrolled directory: {output}"
        )
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UnsafeOutputError(
            f"invalid controlled-gallery marker: {marker}"
        ) from error
    if value.get("schema") != GALLERY_SCHEMA:
        raise UnsafeOutputError(f"unrecognized controlled-gallery marker: {marker}")
    return True


def _verify_assets(
    gallery: Path, renderer_path: Path, t2: Any, data: dict[str, Any]
) -> None:
    t2.verify_output(data, gallery, renderer_path)
    html = (gallery / "index.html").read_text(encoding="utf-8")
    if URL_RE.search(html):
        raise RuntimeError("rendered HTML contains a remote asset")
    if t2.CJK_RE.search(html):
        raise RuntimeError("rendered HTML contains non-English CJK text")
    for link in ASSET_RE.findall(html):
        if link.startswith(("#", "data:", "mailto:")) or "://" in link:
            continue
        target = (gallery / link).resolve()
        if not target.exists():
            raise RuntimeError(f"rendered HTML has a missing local asset: {link}")


def _write_availability(path: Path, notes: list[str]) -> None:
    lines = [
        "# Data availability and limits",
        "",
        "Every listed numeric series is traceable through `input-index.json`.",
        "No unavailable section receives a placeholder numeric series.",
        "",
    ]
    lines.extend(f"- {note}" for note in notes)
    lines.extend(
        [
            "",
            "Return-code completion, timing, and raw diagnostics do not establish",
            "coupled T1 acceptance, physical qualification, or a performance claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _verify_inputs_unchanged(records: list[dict[str, Any]]) -> None:
    for record in records:
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"input changed during rendering: {path}")


def _augment_manifest(
    gallery: Path,
    source_root: Path,
    data: dict[str, Any],
) -> None:
    manifest_path = gallery / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["archive"] = {
        "label": "Exact input path and SHA-256 index",
        "path": "input-index.json",
    }
    manifest["known_issues"] = [
        {
            "label": "Data availability and limits",
            "path": "availability.md",
        }
    ]
    manifest["pages"].append(
        {
            "id": "data-availability",
            "title": "Data availability and limits",
            "introduction": (
                "Missing sections are named explicitly; no numeric placeholder "
                "series are inserted."
            ),
            "notes": data["availability_notes"],
            "figures": [],
        }
    )
    manifest["schema"] = GALLERY_SCHEMA
    manifest["canonical_source_root"] = str(source_root)
    manifest["source_roots"] = data["source_roots"]
    manifest["input_index_sha256"] = sha256_file(gallery / "input-index.json")
    _write_json(manifest_path, manifest)
    _write_json(gallery / "gallery-manifest.json", manifest)
    t2 = _load_module(
        "a050_gallery_t2_renderer",
        source_root / "tools/plots/build_a1_t2_report.py",
    )
    a0, _ = t2.load_sources(source_root)
    html = a0.render_report(manifest, manifest_path, gallery / "index.html")
    html = t2.embed_svgs(html, data["figures"], gallery)
    (gallery / "index.html").write_text(html, encoding="utf-8")


def _write_provenance(
    gallery: Path,
    entrypoint: Path,
    source_root: Path,
    data: dict[str, Any],
) -> None:
    outputs = {}
    for path in sorted(gallery.iterdir()):
        if path.is_file() and path.name != "provenance.json":
            outputs[path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    canonical = {
        name: {
            "path": str((source_root / name).resolve()),
            "sha256": sha256_file(source_root / name),
        }
        for name in CANONICAL_FILES
    }
    _write_json(
        gallery / "provenance.json",
        {
            "schema": GALLERY_SCHEMA,
            "generated_utc": datetime.now(UTC).isoformat(),
            "entrypoint": {
                "path": str(entrypoint.resolve()),
                "sha256": sha256_file(entrypoint),
            },
            "canonical_sources": canonical,
            "source_roots": data["source_roots"],
            "input_index": data["input_index"],
            "outputs": outputs,
        },
    )


def _swap_controlled_output(staging: Path, output: Path, existing: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if existing:
        backup = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.backup-", dir=output.parent)
        )
        backup.rmdir()
        output.rename(backup)
    try:
        staging.rename(output)
    except Exception:
        if backup is not None and backup.exists() and not output.exists():
            backup.rename(output)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def build_gallery(run_root: Path, source_root: Path, output: Path) -> dict[str, Any]:
    """Build into a staging directory and replace only a controlled gallery."""
    run_root = run_root.resolve()
    source_root = source_root.resolve()
    output = output.resolve()
    if not (source_root / "tools/plots").is_dir():
        raise FileNotFoundError(f"canonical source root is invalid: {source_root}")
    existing = ensure_output_target(
        output,
        run_root=run_root,
        source_root=source_root,
    )
    data = collect_gallery_data(run_root, source_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.staging-", dir=output.parent
    ) as temporary:
        staging_root = Path(temporary)
        data_path = staging_root / "reviewed-data.json"
        _write_json(data_path, data)
        (staging_root / "input-index.json").write_text(
            json.dumps(data["input_index"], ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        staging = staging_root / "gallery"
        t2 = _load_module(
            "a050_gallery_t2_builder",
            source_root / "tools/plots/build_a1_t2_report.py",
        )
        t2.build(data_path, staging, source_root)
        shutil.copyfile(data_path, staging / "reviewed-data.json")
        shutil.copyfile(staging_root / "input-index.json", staging / "input-index.json")
        _write_availability(staging / "availability.md", data["availability_notes"])
        _augment_manifest(staging, source_root, data)
        _verify_inputs_unchanged(data["input_index"])
        renderer_path = source_root / "tools/plots/build_a0_report.py"
        _verify_assets(staging, renderer_path, t2, data)
        _write_json(
            staging / MARKER_NAME,
            {
                "schema": GALLERY_SCHEMA,
                "title": data["title"],
                "run_root": str(run_root),
                "source_root": str(source_root),
            },
        )
        _write_provenance(
            staging,
            Path(__file__).resolve(),
            source_root,
            data,
        )
        _swap_controlled_output(staging, output, existing)
    _verify_inputs_unchanged(data["input_index"])
    t2_final = _load_module(
        "a050_gallery_t2_final_verifier",
        source_root / "tools/plots/build_a1_t2_report.py",
    )
    _verify_assets(
        output,
        source_root / "tools/plots/build_a0_report.py",
        t2_final,
        data,
    )
    return {
        "status": "ok",
        "output": str(output),
        "figures": len(data["figures"]),
        "availability_notes": len(data["availability_notes"]),
        "inputs": len(data["input_index"]),
        "input_index_sha256": sha256_file(output / "input-index.json"),
        "reviewed_data_sha256": sha256_file(output / "reviewed-data.json"),
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "index_sha256": sha256_file(output / "index.html"),
        "replaced_existing": existing,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_gallery(args.run_root, args.source_root, args.output)
    except (
        FileNotFoundError,
        InputError,
        UnsafeOutputError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
