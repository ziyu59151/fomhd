#!/usr/bin/env python3
"""Plot archived local Hall, temporal MMS and closed-grid operator evidence."""
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


def require_positive(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Logarithmic plot values must be finite and positive")
    return result


def positive_frequencies_hz(values: np.ndarray) -> list[float]:
    values = np.asarray(values)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("Expected four finite transverse eigenvalues")
    positive = sorted(float(value.imag / (2 * math.pi))
                      for value in values if value.imag > 0)
    if len(positive) != 2:
        raise ValueError("Expected two positive-frequency transverse branches")
    return positive


def refinement(rows: list[dict]) -> dict:
    if len(rows) < 2:
        raise ValueError("At least two refinement levels are required")
    rows = sorted(rows, key=lambda row: require_positive(row["dt_s"]), reverse=True)
    dt = [require_positive(row["dt_s"]) for row in rows]
    errors = [require_positive(row["error"]) for row in rows]
    horizon = require_positive(rows[0]["final_time_s"])
    if len(set(dt)) != len(dt):
        raise ValueError("Duplicate time steps in refinement")
    if any(not math.isclose(require_positive(row["final_time_s"]), horizon,
                            rel_tol=1e-12, abs_tol=0) for row in rows):
        raise ValueError("Refinement requires an equal physical horizon")
    orders = [math.log(a / b) / math.log(x / y)
              for a, b, x, y in zip(errors[:-1], errors[1:], dt[:-1], dt[1:])]
    return {"dt_s": dt, "error": errors, "orders": orders, "final_time_s": horizon}


def collect(massless_path: Path, mms_path: Path) -> dict:
    inputs = {}

    def read(path):
        content = path.read_bytes()
        inputs[str(path.resolve())] = hashlib.sha256(content).hexdigest()
        return content

    massless = json.loads(read(massless_path))
    mms = json.loads(read(mms_path))
    cases = massless["cases"]
    if len({case["case"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case identity")
    failed = [case["case"] for case in cases if case["status"] not in ("pass", "measured")]
    if failed or not mms["passed"] or mms["status"] != "PASS":
        raise ValueError(f"Failed verification input cannot be omitted: {failed}")
    negative = mms["omitted_ion_pressure_source_control"]
    if not negative["passed"] or not negative["detected"]:
        raise ValueError("Failed MMS negative control cannot be omitted")
    result = {"input_sha256": inputs, "massless_backend": massless["backend_requested"],
              "mms_backend": mms["device"], "scientific_acceptance": massless["scientific_acceptance"],
              "case_statuses": {case["case"]: case["status"] for case in cases},
              "mms_status": {"status": mms["status"], "passed": mms["passed"],
                             "groups": {name: group["passed"] for name, group in mms["results"].items()},
                             "negative_control": {key: negative[key] for key in (
                                 "case", "passed", "detected", "physical_gates_passed", "integrator",
                                 "steps", "minimum_detected_error", "errors", "scope") if key in negative}},
              "hall": [], "massless_time": {}, "mms_time": {}, "spatial": {}}
    for case in cases:
        if case["kind"] != "controlled_local_parallel_hall":
            continue
        raw = (massless_path.parent / case["raw_file"]).resolve()
        if raw.parent != massless_path.parent.resolve():
            raise ValueError("Hall arrays must be sibling evidence files")
        with np.load(io.BytesIO(read(raw)), allow_pickle=False) as arrays:
            measured = positive_frequencies_hz(arrays["production_eigenvalues"])
        result["hall"].append({"kdi": require_positive(case["kdi"]),
                               "theory_hz": [require_positive(case[key]) / (2 * math.pi)
                                             for key in ("omega_minus", "omega_plus")],
                               "measured_hz": measured})
    result["hall"].sort(key=lambda row: row["kdi"])
    if not result["hall"] or len({row["kdi"] for row in result["hall"]}) != len(result["hall"]):
        raise ValueError("Missing or duplicated Hall samples")
    for method in ("midpoint", "rk4"):
        rows = [{"dt_s": case["dt_s"], "error": case["continuum_relative_error"],
                 "final_time_s": case["final_time_s"]}
                for case in cases if case["kind"] == "real_time_source_none"
                and case["integrator"] == method]
        result["massless_time"][method] = refinement(rows)
        source = mms["results"][method]
        if source["device"] != mms["device"] or not source["passed"]:
            raise ValueError("MMS backend or acceptance mismatch")
        result["mms_time"][method] = refinement([
            {"dt_s": case["dt_s"], "error": case["errors"]["combined"],
             "final_time_s": case["duration_s"]} for case in source["cases"]])
    for field, kind, key, unit in (
        ("Velocity reconstruction", "closed_qhfe_reconstruction", "velocity_norms", "m/s"),
        ("Velocity divergence", "closed_qhfe_reconstruction", "theta_norms", "1/s"),
        ("Pressure force", "closed_rectangle_operator_consistency", "pressure_norms", "Pa/m"),
    ):
        levels = sorted([case for case in cases if case["kind"] == kind
                         and (kind != "closed_rectangle_operator_consistency" or case["compressible"])],
                        key=lambda case: case["nodes"])
        if len(levels) < 2 or len({case["nodes"] for case in levels}) != len(levels):
            raise ValueError("Missing or duplicate spatial levels")
        regions = {}
        for region in ("all", "inner", "wall"):
            rows = []
            for level in levels:
                norm = level[key][region]
                if not norm["available"] or norm["unit"] != unit or norm["entities"] <= 0:
                    raise ValueError(f"Unavailable or invalid spatial norm: {field}/{region}")
                rows.append({"nodes": level["nodes"], "h_m": 1 / require_positive(level["nodes"]),
                             "error": require_positive(norm["weighted_relative_l2"]),
                             "entities": norm["entities"], "field_unit": unit})
            regions[region] = rows
        result["spatial"][field] = regions
    return result


def render(data: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    hall = data["hall"]
    for index, label in enumerate(("Lower branch", "Upper branch")):
        x = [row["kdi"] for row in hall]
        axes[0].loglog(x, [row["theory_hz"][index] for row in hall],
                       color=f"C{index}", label=f"{label}: theory")
        axes[0].loglog(x, [row["measured_hz"][index] for row in hall],
                       "o", color=f"C{index}", markerfacecolor="none", markersize=9,
                       label=f"{label}: measured")
    axes[0].set(xlabel=r"$k d_i$", ylabel="Frequency (Hz)",
                title=f"Local transverse Hall response ({data['massless_backend']})")
    for axis, group, title in (
        (axes[1], "massless_time", f"Massless eigenmode ({data['massless_backend']})"),
        (axes[2], "mms_time", f"Finite-mass coupled MMS ({data['mms_backend']})"),
    ):
        for method, values in data[group].items():
            order = ", ".join(f"{value:.2f}" for value in values["orders"])
            axis.loglog(np.asarray(values["dt_s"]) * 1e9, values["error"], "o-",
                        label=f"{method}; p={order}")
        ticks = sorted({dt * 1e9 for values in data[group].values() for dt in values["dt_s"]})
        axis.set_xticks(ticks, labels=[f"{value:.3g}" for value in ticks])
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set(xlabel="Time step (ns)", ylabel="Relative error", title=title)
    for axis in axes:
        axis.grid(True, which="both", alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Numerical verification: local theory and equal-horizon refinement")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"local-and-temporal.{extension}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for axis, (field, regions) in zip(axes, data["spatial"].items()):
        for region, rows in regions.items():
            axis.loglog([row["h_m"] for row in rows], [row["error"] for row in rows],
                        "o-", label=region)
        ticks = sorted({row["h_m"] for rows in regions.values() for row in rows})
        axis.set_xticks(ticks, labels=[f"{value:.3g}" for value in ticks])
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set(xlabel="Grid spacing (m)", ylabel="Weighted relative L2 error", title=field)
        axis.grid(True, which="both", alpha=0.2)
        axis.legend()
    fig.suptitle(f"Closed-grid operator measurements ({data['massless_backend']}): boundary order unresolved")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"spatial-operators.{extension}", dpi=180)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--massless", type=Path, required=True)
    parser.add_argument("--mms", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    data = collect(args.massless, args.mms)
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output / "mpl-cache"))
    render(data, args.output)
    import matplotlib
    provenance = {"input_sha256": data["input_sha256"], "command": sys.argv if argv is None else argv,
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "numpy": np.__version__, "matplotlib": matplotlib.__version__, "python": sys.version}
    (args.output / "plot-data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.output / "SCOPE.md").write_text(
        "# Figure scope\n\nLocal transverse Hall and forced discrete temporal MMS are numerical "
        "verification, not full-spectrum, Tokamak, ICE or experimental validation. "
        "Backend labels are taken from the input reports. Spatial curves are measurements "
        "on a fixed rectangle with changing regional quadrature samples; they do not "
        "establish a formal boundary order. E0 with shared discrete operators is not "
        "plotted as independent continuum field accuracy. Massless case statuses and "
        "MMS group/omitted-source control status remain in plot-data.json. "
        "No solver was run by this script.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
