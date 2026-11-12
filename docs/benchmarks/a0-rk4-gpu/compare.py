"""Compare two completed A0 CLI runs using accepted states and saved hard checks."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


FIELDS = (
    "n_i", "n_e", "mom_i", "mom_e", "p_i", "p_perp_e", "p_parallel_e",
    "B_R", "B_phi", "B_Z", "derived_E",
)
INPUTS = ("case.toml", "geometry.npz", "initial.npz")
CHECK_KEYS = (
    "name", "population", "component", "region", "step", "stage", "norm",
    "unit", "status",
)
CHECK_NAMES = (
    "ct_divergence", "divB_update", "wall_ion_flux", "wall_electron_flux",
    "wall_current", "wall_field_original_rows", "field_exchange_interval",
    "wall_field_reaction_work", "total_energy_interval",
)


def hard_checks(run: Path) -> list[dict]:
    with (run / "checks.jsonl").open() as source:
        return [row for line in source if (row := json.loads(line))["gate"] == "hard"]


def check_identity(row: dict) -> tuple:
    return tuple(row[key] for key in CHECK_KEYS)


def check_usage(rows: list[dict]) -> dict:
    usage = {}
    for name in CHECK_NAMES:
        positive_budget = [
            abs(row["raw"]) / row["threshold"]
            for row in rows if row["name"] == name and row["threshold"] > 0
        ]
        if positive_budget:
            usage[name] = max(positive_budget)
    return usage


def compare(left: Path, right: Path) -> dict:
    for run in (left, right):
        status = json.loads((run / "run.json").read_text())["status"]
        assert status == "completed", f"Run is incomplete: {run}"
    input_hashes = {}
    for name in INPUTS:
        original = (left / "inputs" / name).read_bytes()
        assert original == (right / "inputs" / name).read_bytes(), f"Input mismatch: {name}"
        input_hashes[name] = hashlib.sha256(original).hexdigest()

    left_end = sorted((left / "snapshots").glob("state_*.npz"))[-1]
    right_end = sorted((right / "snapshots").glob("state_*.npz"))[-1]
    with np.load(left_end, allow_pickle=False) as x, np.load(right_end, allow_pickle=False) as y:
        assert x["step"] == y["step"] and x["time"] == y["time"], "Endpoint mismatch"
        assert x["accepted"] and y["accepted"], "Endpoint must be accepted"
        fields = {}
        for name in FIELDS:
            assert x[name].shape == y[name].shape, f"Shape mismatch: {name}"
            fields[name] = {
                "left_max_abs": float(np.max(np.abs(x[name]), initial=0)),
                "max_abs_delta": float(np.max(np.abs(x[name] - y[name]), initial=0)),
            }
        endpoint = {"step": int(x["step"]), "time_s": float(x["time"])}

    left_checks, right_checks = hard_checks(left), hard_checks(right)
    mismatches = [
        {"index": index, "left": check_identity(a), "right": check_identity(b)}
        for index, (a, b) in enumerate(zip(left_checks, right_checks))
        if check_identity(a) != check_identity(b)
    ]
    threshold_differences = [
        (a["threshold"], b["threshold"])
        for a, b in zip(left_checks, right_checks)
        if check_identity(a) == check_identity(b) and a["threshold"] != b["threshold"]
    ]
    left_statuses = Counter(row["status"] for row in left_checks)
    right_statuses = Counter(row["status"] for row in right_checks)
    checks_pass = (
        len(left_checks) == len(right_checks)
        and not mismatches
        and left_statuses == right_statuses == {"PASS": len(left_checks)}
        and bool(left_checks)
    )
    return {
        "input_sha256": input_hashes,
        "endpoint": endpoint,
        "fields": fields,
        "hard_checks": {
            "same_and_all_pass": checks_pass,
            "left_status_counts": left_statuses,
            "right_status_counts": right_statuses,
            "mismatched_common_row_count": len(mismatches),
            "first_mismatch": mismatches[0] if mismatches else None,
            "threshold_float_difference_count": len(threshold_differences),
            "max_threshold_relative_difference": max(
                (abs(a - b) / max(abs(a), abs(b)) for a, b in threshold_differences),
                default=0.0,
            ),
            "left_max_abs_raw_per_positive_threshold": check_usage(left_checks),
            "right_max_abs_raw_per_positive_threshold": check_usage(right_checks),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_run", type=Path)
    parser.add_argument("candidate_run", type=Path)
    args = parser.parse_args()
    report = compare(args.original_run, args.candidate_run)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["hard_checks"]["same_and_all_pass"] else 1)
