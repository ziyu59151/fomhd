"""Summarize one or more raw measurement JSONL files as a JSON table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["event"] == "start" and rows[-1]["event"] == "exit", path
    assembly = next((row for row in rows if row["event"] == "operator_assembly"), None)
    initialization = next((row for row in rows if row["event"] == "initialization"), None)
    steps = [row for row in rows if row["event"] == "step"]
    device_path = path.with_name(path.stem + "-device.csv")
    if path.name == "metrics.jsonl":
        device_path = path.parent / "device-samples.csv"
    device_samples = [
        int(line.rsplit(",", 1)[1]) for line in device_path.read_text().splitlines()
    ]
    return {
        "allocation": rows[0]["allocation"],
        "source": rows[0]["source"],
        "exit_code": rows[-1]["code"],
        "complete_wall_s": rows[-1]["elapsed_s"],
        "assembly_s_including_inventory": assembly["elapsed_assembly_s"] if assembly else None,
        "initial_state_build_s": initialization["elapsed_initial_s"] if initialization else None,
        "warm_step_median_s": statistics.median(
            row["elapsed_step_s"] for row in steps[1:]
        ) if len(steps) > 1 else None,
        "step_measurements": len(steps),
        "hard_checks": sum(row["hard"] for row in steps),
        "hard_failures": [name for row in steps for name in row["hard_failures"]],
        "tracked_operator_B": sum(
            member["bytes"] for member in assembly["members"].values()
        ) if assembly else None,
        "host_rss_B": max(row["host_peak_rss_B"] for row in rows),
        "process_gpu_memory_samples": len(device_samples),
        "sampled_process_gpu_peak_MiB": max(device_samples, default=None),
        "sampled_cupy_pool_peak_live_B": max(
            (row["cupy_pool_live_B"] for row in rows if "cupy_pool_live_B" in row), default=None
        ),
        "sampled_cupy_pool_peak_reserved_B": max(
            (row["cupy_pool_reserved_B"] for row in rows if "cupy_pool_reserved_B" in row),
            default=None,
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps({str(path): summarize(path) for path in args.metrics}, indent=2,
                     allow_nan=False))
