#!/usr/bin/env python3
"""Plot only measured FOMHD benchmark outcomes."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _series(summary: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [case for case in summary.get("cases", []) if case.get("kind") == key]


def _timing_values(case: dict[str, Any]) -> dict[str, Any]:
    timing = case.get("timing") or {}
    return {
        "case_id": case.get("case_id"),
        "status": case.get("status"),
        "setup_s": timing.get("setup_s"),
        "initial_s": timing.get("initial_transaction_s"),
        "first_step_s": timing.get("first_step_s"),
        "steady_median_s": timing.get("steady_median_s"),
        "steady_spread_s": timing.get("steady_half_interdecile_spread_s"),
    }


def _scan_values(case: dict[str, Any]) -> dict[str, Any]:
    frequency = case.get("frequency") or {}
    return {
        "case_id": case.get("case_id"),
        "status": case.get("status"),
        "integrator": case.get("integrator"),
        "dt_s": case.get("dt_s"),
        "nsteps": case.get("nsteps_requested"),
        "accepted_step": case.get("accepted_step"),
        "steady_median_s": (case.get("timing") or {}).get("steady_median_s"),
        "max_normalised_residual": case.get("max_normalised_residual"),
        "frequency_status": frequency.get("status"),
        "signed_frequency_hz": frequency.get("signed_frequency_hz"),
        "angular_frequency_rad_s": frequency.get("angular_frequency_rad_s"),
        "final_amplitude_ratio": frequency.get("final_amplitude_ratio"),
        "omega_plasma_reference_rad_s": frequency.get(
            "omega_plasma_reference_rad_s"
        ),
        "failure_code": (case.get("failure") or {}).get("code"),
    }


def _method_style(method: str) -> tuple[str, str]:
    return ("C0", "o") if method == "midpoint" else ("C3", "s")


def _set_dt_limits(ax: Any, cases: list[dict[str, Any]]) -> None:
    values = [
        float(case["dt_s"])
        for case in cases
        if _finite(case.get("dt_s")) and float(case["dt_s"]) > 0
    ]
    if values:
        low = min(values)
        high = max(values)
        ax.set_xlim(low / 1.8, high * 1.8)


def _annotate_failed_cases(
    ax: Any,
    failed: list[dict[str, Any]],
    *,
    color: str = "C3",
    label: str = "failed case (no y value)",
) -> None:
    positions = [
        float(case["dt_s"])
        for case in failed
        if _finite(case.get("dt_s")) and float(case["dt_s"]) > 0
    ]
    if not positions:
        return
    transform = ax.get_xaxis_transform()
    ax.plot(
        positions,
        [0.06] * len(positions),
        transform=transform,
        color=color,
        marker="x",
        linestyle="None",
        label=label,
        clip_on=False,
    )
    for index, case in enumerate(failed):
        ax.text(
            float(case["dt_s"]),
            0.10 + 0.035 * (index % 5),
            str(case.get("case_id", "failed"))[:22],
            transform=transform,
            rotation=90,
            va="bottom",
            ha="center",
            fontsize=5,
            color=color,
        )


def _empty(ax: Any, text: str) -> None:
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="0.35",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_pilot(ax: Any, cases: list[dict[str, Any]]) -> None:
    if not cases:
        _empty(ax, "No pilot cases in this run")
        ax.set_title("(a) Pilot timing (measured)")
        return
    labels = []
    positions = []
    phases = (
        ("setup_s", "setup", "C0"),
        ("initial_s", "initial", "C2"),
        ("first_step_s", "first actual step", "C1"),
        ("steady_median_s", "steady median", "C4"),
    )
    for case_index, case in enumerate(cases):
        values = _timing_values(case)
        base = len(labels)
        for phase_index, (key, label, color) in enumerate(phases):
            value = values[key]
            if not _finite(value) or value <= 0:
                continue
            position = base + phase_index
            positions.append(
                (position, float(value), case, color, label, values)
            )
        labels.extend(
            [
                f"{case.get('case_id', 'case')}\n" + "setup",
                "initial",
                "first",
                "steady",
            ]
        )
    for position, value, _case, color, label, values in positions:
        ax.bar(position, value, color=color, alpha=0.82, label=None)
        if label == "steady median" and _finite(values["steady_spread_s"]):
            spread = float(values["steady_spread_s"])
            ax.errorbar(
                position,
                value,
                yerr=spread,
                fmt="none",
                ecolor="black",
                elinewidth=0.8,
                capsize=2,
            )
    if not positions:
        _empty(ax, "Pilot cases have no measured timing and are not plotted")
        failed = [case for case in cases if case.get("status") != "accepted"]
        if failed:
            _annotate_failed_cases(ax, failed)
        ax.set_title("(a) Pilot timing (measured)")
        return
    ax.set_yscale("log")
    ax.set_ylabel("seconds (log scale)")
    ax.grid(True, axis="y", alpha=0.25)
    shown = {item[4] for item in positions}
    handles = [
        ax.bar([], [], color=color, label=label)
        for _key, label, color in phases
        if label in shown
    ]
    if handles:
        ax.legend(handles=handles, fontsize=7, frameon=False)
    ax.set_xticks(range(0, len(labels), 4))
    case_labels = [
        f"{case.get('case_id', 'case')}\n{case.get('status', '')}"
        for case in cases
    ]
    ax.set_xticklabels(case_labels, rotation=20, ha="right", fontsize=7)
    failed = [case for case in cases if case.get("status") != "accepted"]
    if failed:
        ax.text(
            0.01,
            0.98,
            "failed cases retained; unavailable timing not fabricated",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            color="C3",
        )
    ax.set_title("(a) Pilot timing (measured)")


def _plot_cost(ax: Any, cases: list[dict[str, Any]]) -> None:
    if not cases:
        _empty(ax, "No scan cases in this run")
        ax.set_title("(b) Timed scan cost (measured)")
        return
    for method in ("midpoint", "rk4"):
        selected = [case for case in cases if case.get("integrator") == method]
        color, marker = _method_style(method)
        accepted = [
            (_scan_values(case), case)
            for case in selected
            if case.get("status") == "accepted"
            and _finite(_scan_values(case)["steady_median_s"])
        ]
        if accepted:
            ax.scatter(
                [item[0]["dt_s"] for item in accepted],
                [item[0]["steady_median_s"] for item in accepted],
                color=color,
                marker=marker,
                label=method,
            )
        failed = [case for case in selected if case.get("status") != "accepted"]
        _annotate_failed_cases(ax, failed, label=f"{method} failure")
    has_measured_cost = any(
        case.get("status") == "accepted"
        and _finite(_scan_values(case)["steady_median_s"])
        for case in cases
    )
    ax.set_xscale("log")
    if has_measured_cost:
        ax.set_yscale("log")
        ax.set_ylabel("median accepted step (s)")
    else:
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.set_ylabel("failed cases (categorical)")
    ax.set_xlabel("dt (s)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    _set_dt_limits(ax, cases)
    failed_count = sum(case.get("status") != "accepted" for case in cases)
    if failed_count:
        ax.text(
            0.01,
            0.03,
            f"{failed_count} failed case(s); no interpolated success",
            transform=ax.transAxes,
            fontsize=7,
            color="C3",
        )
    ax.set_title("(b) Timed scan cost (measured)")


def _plot_residual(ax: Any, cases: list[dict[str, Any]]) -> None:
    if not cases:
        _empty(ax, "No scan cases in this run")
        ax.set_title("(c) Maximum normalized diagnostic residual")
        return
    for method in ("midpoint", "rk4"):
        selected = [
            case
            for case in cases
            if case.get("integrator") == method
            and case.get("status") == "accepted"
        ]
        values = [
            _scan_values(case)
            for case in selected
            if _finite(_scan_values(case)["max_normalised_residual"])
        ]
        if not values:
            continue
        color, marker = _method_style(method)
        ax.scatter(
            [value["dt_s"] for value in values],
            [value["max_normalised_residual"] for value in values],
            color=color,
            marker=marker,
            label=method,
        )
    failed = [case for case in cases if case.get("status") != "accepted"]
    _annotate_failed_cases(
        ax, failed, label="failure (no residual substitution)"
    )
    has_measured_residual = any(
        case.get("status") == "accepted"
        and _finite(_scan_values(case)["max_normalised_residual"])
        for case in cases
    )
    ax.set_xscale("log")
    if has_measured_residual:
        ax.set_yscale("log")
        ax.set_ylabel("raw / threshold (max)")
    else:
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.set_ylabel("failed cases (categorical)")
    ax.set_xlabel("dt (s)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    _set_dt_limits(ax, cases)
    failed_count = sum(case.get("status") != "accepted" for case in cases)
    if failed_count:
        ax.text(
            0.01,
            0.03,
            f"{failed_count} failed case(s); no residual substituted",
            transform=ax.transAxes,
            fontsize=7,
            color="C3",
        )
    ax.set_title("(c) Maximum normalized diagnostic residual")


def _plot_frequency(ax: Any, cases: list[dict[str, Any]]) -> None:
    if not cases:
        _empty(ax, "No scan cases in this run")
        ax.set_title("(d) Measured signal frequency")
        return
    measured = [
        _scan_values(case)
        for case in cases
        if (_scan_values(case)["frequency_status"] == "MEASURED")
    ]
    reference = None
    for value in measured:
        if _finite(value["omega_plasma_reference_rad_s"]):
            reference = float(value["omega_plasma_reference_rad_s"])
            break
    if reference is None:
        reference = float(cases[0].get("frequency", {}).get(
            "omega_plasma_reference_rad_s", np.nan
        ))
    if _finite(reference) and reference > 0:
        ax.axhline(
            reference,
            color="0.25",
            linestyle="--",
            linewidth=1.0,
            label=r"cold $\omega_p$ reference (approximate)",
        )
    for method in ("midpoint", "rk4"):
        selected = [value for value in measured if value["integrator"] == method]
        if not selected:
            continue
        color, marker = _method_style(method)
        ax.scatter(
            [value["dt_s"] for value in selected],
            [value["angular_frequency_rad_s"] for value in selected],
            color=color,
            marker=marker,
            label=f"{method} measured",
        )
    unavailable = [
        _scan_values(case)
        for case in cases
        if (_scan_values(case)["frequency_status"] != "MEASURED")
    ]
    if unavailable:
        _annotate_failed_cases(
            ax,
            [
                {
                    "dt_s": item["dt_s"],
                    "case_id": item["case_id"],
                }
                for item in unavailable
            ],
            color="C7",
            label="frequency unavailable",
        )
    ax.set_xscale("log")
    if measured or (_finite(reference) and reference > 0):
        ax.set_yscale("log")
    else:
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
    ax.set_xlabel("dt (s)")
    ax.set_ylabel(r"measured angular frequency (rad/s)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    _set_dt_limits(ax, cases)
    unavailable_count = sum(
        _scan_values(case)["frequency_status"] != "MEASURED"
        for case in cases
    )
    if unavailable_count:
        ax.text(
            0.01,
            0.03,
            f"{unavailable_count} frequency estimate(s) UNAVAILABLE",
            transform=ax.transAxes,
            fontsize=7,
            color="0.35",
        )
    ax.set_title("(d) Measured signal frequency")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--output")
    arguments = parser.parse_args(argv)
    run_dir = Path(arguments.run_dir).expanduser().absolute()
    summary_path = run_dir / "summary.json"
    benchmark_path = run_dir / "benchmark.json"
    observations_path = run_dir / "observations.npz"
    for path in (summary_path, benchmark_path):
        if not path.is_file():
            raise FileNotFoundError(f"benchmark artifact is missing: {path}")
    output = (
        Path(arguments.output).expanduser().absolute()
        if arguments.output
        else run_dir / "plots"
    )
    png = output / "warp_benchmark_summary.png"
    pdf = output / "warp_benchmark_summary.pdf"
    plot_data_path = output / "plot_data.json"
    if os.path.lexists(png) or os.path.lexists(pdf) or os.path.lexists(plot_data_path):
        raise FileExistsError(f"plot output already exists; refusing overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _load_json(summary_path)
    benchmark = _load_json(benchmark_path)
    observations = (
        np.load(observations_path, allow_pickle=False)
        if observations_path.is_file()
        else None
    )
    pilot = _series(summary, "pilot")
    scan = _series(summary, "scan")
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.8), constrained_layout=True)
    _plot_pilot(axes[0, 0], pilot)
    _plot_cost(axes[0, 1], scan)
    _plot_residual(axes[1, 0], scan)
    _plot_frequency(axes[1, 1], scan)
    model = benchmark.get("model", {})
    figure.suptitle(
        "FOMHD finite-mass benchmark: actual solver outcomes only\n"
        f"device={benchmark.get('requested_device')}  "
        f"model={model.get('variant')}  precision={model.get('precision')}",
        fontsize=11,
    )
    figure.savefig(png, dpi=180)
    figure.savefig(pdf)
    plt.close(figure)
    plot_data = {
        "source_summary": str(summary_path),
        "source_benchmark": str(benchmark_path),
        "source_observations": (
            str(observations_path) if observations is not None else None
        ),
        "observation_arrays": (
            sorted(observations.files) if observations is not None else []
        ),
        "pilot": [_timing_values(case) for case in pilot],
        "scan": [_scan_values(case) for case in scan],
        "theoretical_references": {
            "cold_omega_plasma_rad_s": benchmark.get("time", {}).get(
                "omega_plasma_rad_s"
            ),
            "label": (
                "cold approximate plasma frequency; not exact continuum "
                "validation or a fitted production eigenmode"
            ),
        },
        "rules": [
            "measured values are not interpolated",
            "failed cases are not replaced by successful values",
            "unavailable frequencies remain unavailable",
        ],
    }
    with plot_data_path.open("x", encoding="utf-8") as handle:
        json.dump(plot_data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(png)
    print(pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
