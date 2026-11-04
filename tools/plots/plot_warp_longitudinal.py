#!/usr/bin/env python3
"""Plot archived unforced longitudinal time controls without running a solver."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


def collect(path: Path) -> dict:
    hashes = {}

    def read(source):
        content = source.read_bytes()
        hashes[str(source.resolve())] = hashlib.sha256(content).hexdigest()
        return content

    report = json.loads(read(path))
    cases = report["cases"]
    if report["status"] != "completed" or not report["passed"] or not cases:
        raise ValueError("A completed passing control report is required")
    if len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("Duplicate case identity")
    controls = report.get("amplitude_sensitivity", [])
    if (len(controls) != 2 or {row["integrator"] for row in controls} != {"midpoint", "rk4"}
            or any(not row["passed"] for row in controls)):
        raise ValueError("One passing amplitude-sensitivity control per integrator is required")
    device = cases[0]["device"]
    horizon, omega = cases[0]["target_time_s"], cases[0]["omega_rad_s"]
    if not all(math.isfinite(x) and x > 0 for x in (horizon, omega)):
        raise ValueError("Invalid physical horizon or frequency")
    histories = {}
    for row in cases:
        numeric = [row[key] for key in (
            "dt_s", "relative_amplitude", "amplitude_ratio", "polynomial_amplitude_ratio")]
        if not all(math.isfinite(x) and x > 0 for x in numeric):
            raise ValueError("Invalid positive plot quantity")
        error = row["continuum_relative_error"]
        if not math.isfinite(error) or error < 0:
            raise ValueError("Invalid continuum error")
        if (not row["passed"] or not row["hard_pass"] or row["device"] != device
                or row["integrator"] not in ("midpoint", "rk4")
                or row["accepted_steps"] != row["steps"] or row["steps"] <= 0
                or not math.isclose(row["omega_rad_s"], omega, rel_tol=1e-12, abs_tol=0)
                or any(not math.isclose(value, horizon, rel_tol=1e-12, abs_tol=0)
                       for value in (row["target_time_s"], row["accepted_time_s"],
                                     row["steps"]*row["dt_s"]))):
            raise ValueError("Failed, mixed-backend or unequal-horizon evidence")
        raw = (path.parent/(row["case_id"]+".npz")).resolve()
        if raw.parent != path.parent.resolve():
            raise ValueError("Histories must be sibling evidence files")
        with np.load(io.BytesIO(read(raw)), allow_pickle=False) as arrays:
            times, mode = arrays["time_s"], arrays["mode"]
        if (times.shape != (row["steps"]+1,) or mode.shape != (row["steps"]+1, 7)
                or not np.all(np.isfinite(times)) or not np.all(np.isfinite(mode))
                or not np.allclose(times, np.arange(row["steps"]+1)*row["dt_s"],
                                   rtol=1e-12, atol=0)):
            raise ValueError("History and accepted step/time identity disagree")
        histories[row["case_id"]] = {
            "cycles": (times*omega/(2*math.pi)).tolist(),
            "electron_real": (mode[:, 3].real/row["relative_amplitude"]).tolist(),
        }
    amplitude = max(row["relative_amplitude"] for row in cases)
    fine, curves = {}, {}
    for method in ("midpoint", "rk4"):
        rows = sorted([row for row in cases if row["integrator"] == method
                       and row["relative_amplitude"] == amplitude], key=lambda row: row["dt_s"])
        if not rows or len({row["steps"] for row in rows}) != len(rows):
            raise ValueError("Missing or duplicate full-amplitude steps")
        half = [row for row in cases if row["integrator"] == method
                and row["relative_amplitude"] != amplitude]
        base = rows[-1]
        if (len(half) != 1 or half[0]["relative_amplitude"] != amplitude/2
                or half[0]["steps"] != base["steps"]):
            raise ValueError("Each amplitude control requires a same-step full/half pair")
        measured_change = abs(base["amplitude_ratio"]-half[0]["amplitude_ratio"])/base["amplitude_ratio"]
        control = next(row for row in controls if row["integrator"] == method)
        if not math.isclose(control["relative_amplitude_change"], measured_change,
                            rel_tol=1e-10, abs_tol=1e-15):
            raise ValueError("Amplitude-control summary disagrees with its case pair")
        fine[method] = dict(histories[rows[0]["case_id"]], case_id=rows[0]["case_id"])
        curves[method] = {
            "omega_dt": [omega*row["dt_s"] for row in rows],
            "amplification": [row["amplitude_ratio"] for row in rows],
            "polynomial": [row["polynomial_amplitude_ratio"] for row in rows],
            "continuum_error": [row["continuum_relative_error"] for row in rows],
        }
    display_cycles = np.linspace(0, horizon*omega/(2*math.pi), 501)
    return dict(input_sha256=hashes, device=device, cases=cases, histories=fine,
                amplitude_sensitivity=report["amplitude_sensitivity"], curves=curves,
                analytic=dict(cycles=display_cycles.tolist(),
                              electron_real=np.cos(2*math.pi*display_cycles).tolist()))


def render(data: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    analytic = data["analytic"]
    axes[0].plot(analytic["cycles"], analytic["electron_real"], "k-", label="Continuum")
    for index, method in enumerate(("midpoint", "rk4")):
        history, curve = data["histories"][method], data["curves"][method]
        axes[0].plot(history["cycles"], history["electron_real"],
                     "o" if index == 0 else "+", color=f"C{index}",
                     markersize=4, markevery=4, label=f"{method}: finest dt")
        axes[1].loglog(curve["omega_dt"], curve["polynomial"], color=f"C{index}",
                       label=f"{method}: polynomial")
        axes[1].loglog(curve["omega_dt"], curve["amplification"], "o",
                       color=f"C{index}", markerfacecolor="none", markersize=8,
                       label=f"{method}: measured")
        errors = np.asarray(curve["continuum_error"])
        positive = errors > 0
        axes[2].loglog(np.asarray(curve["omega_dt"])[positive], errors[positive],
                       "o-", color=f"C{index}", label=method)
    axes[0].set(xlabel="Time / analytic period", ylabel="Electron mode / initial amplitude",
                title="Unforced longitudinal oscillation")
    axes[1].axhline(1, color="0.6", linestyle=":")
    axes[1].set(xlabel=r"$\omega\,\Delta t$", ylabel="Endpoint amplitude / initial amplitude",
                title="Numerical amplification, equal final time")
    axes[2].set(xlabel=r"$\omega\,\Delta t$", ylabel="Relative complex endpoint error",
                title="Error against the continuum mode")
    for axis in axes:
        axis.grid(True, which="both", alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle(f"Finite-mass local time verification ({data['device']}): periodic test adapter")
    for extension in ("png", "pdf"):
        fig.savefig(output/f"longitudinal-time-control.{extension}", dpi=180)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    data = collect(args.input)
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output/"mpl-cache"))
    render(data, args.output)
    import matplotlib
    provenance = dict(input_sha256=data["input_sha256"],
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      command=sys.argv if argv is None else argv, python=sys.version,
                      numpy=np.__version__, matplotlib=matplotlib.__version__)
    (args.output/"plot-data.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    (args.output/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    (args.output/"SCOPE.md").write_text(
        "# Figure scope\n\nUnforced finite-mass longitudinal mode of the specified "
        "two-fluid/CGL equations on a periodic test adapter. The waveforms use "
        "the finest full-amplitude histories. Coarse amplification is numerical, "
        "not physical growth. Polynomial agreement does not imply continuum "
        "accuracy. All cases and half-amplitude controls remain in plot-data.json; "
        "zero errors are retained even when omitted on logarithmic axes. "
        "No closed-wall attribution, physical CGL ordering, full stability region, "
        "Tokamak/ICE, performance or experimental validation is claimed. "
        "This script did not execute a solver.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
