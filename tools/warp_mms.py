#!/usr/bin/env python3
"""Run coupled discrete MMS verification and persist one JSON record."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path


PROPOSAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROPOSAL_ROOT / "src"))
sys.path.insert(0, str(PROPOSAL_ROOT / "tests" / "warp"))

from mms_fixture import (  # noqa: E402
    MIDPOINT_LEVELS,
    RK4_LEVELS,
    aggregate_raw_checks,
    archive_source_snapshot,
    build_identity,
    omitted_ion_pressure_control,
    prepare_output_directory,
    require_device,
    run_verification,
    write_failure_arrays,
    write_failure_checks,
    write_json,
)

def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda:N; CUDA requires an existing SLURM_JOB_ID",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new or empty result directory",
    )
    parser.add_argument(
        "--integrator",
        choices=("midpoint", "rk4", "all"),
        default="all",
    )
    parser.add_argument(
        "--no-spatial",
        action="store_true",
        help="skip the small independent spatial convergence scan",
    )
    return parser.parse_args()


def main():
    arguments = _arguments()
    output = prepare_output_directory(arguments.output).resolve()
    started = time.perf_counter()
    record = {
        "case": "A0 Warp coupled temporal MMS and spatial verification",
        "status": "RUNNING",
        "arguments": {
            "device": arguments.device,
            "output": str(output),
            "integrator": arguments.integrator,
            "spatial": not arguments.no_spatial,
        },
        "synchronization": {
            "method": "Backend.synchronize after each accepted level and stage",
            "timing_clock": "perf_counter",
        },
        "build_identity": build_identity(
            PROPOSAL_ROOT,
            cli_path=Path(__file__),
        ),
    }
    try:
        record["source_snapshot"] = archive_source_snapshot(
            output / "source_snapshot",
            harness_root=PROPOSAL_ROOT,
            cli_path=Path(__file__),
        )
        device = require_device(arguments.device)
        integrators = (
            ("midpoint", "rk4")
            if arguments.integrator == "all"
            else (arguments.integrator,)
        )
        record["device"] = device
        use_spatial = not arguments.no_spatial and device == "cpu"
        record["arguments"]["spatial_executed"] = use_spatial
        record["arguments"]["spatial_note"] = (
            "spatial convergence runs on CPU so CUDA verification never "
            "silently falls back; run the same CLI with --device cpu"
        )
        record["results"] = {}
        for integrator in integrators:
            level_started = time.perf_counter()
            result = run_verification(
                integrator,
                device=device,
                levels=(
                    MIDPOINT_LEVELS
                    if integrator == "midpoint"
                    else RK4_LEVELS
                ),
                spatial=use_spatial,
                raw_root=output / "raw",
            )
            result["elapsed_s"] = time.perf_counter() - level_started
            result["synchronized_timing"] = True
            record["results"][integrator] = result
        record["omitted_ion_pressure_source_control"] = (
            omitted_ion_pressure_control(
                device,
                raw_root=output / "raw" / "negative",
            )
        )
        record["raw_check_aggregate"] = aggregate_raw_checks(
            record["results"],
            record["omitted_ion_pressure_source_control"],
        )
        record["elapsed_s"] = time.perf_counter() - started
        record["passed"] = bool(
            all(result["passed"] for result in record["results"].values())
            and record["omitted_ion_pressure_source_control"]["passed"]
        )
        record["status"] = "PASS" if record["passed"] else "FAIL"
        exit_code = 0 if record["passed"] else 1
    except Exception as error:
        record["elapsed_s"] = time.perf_counter() - started
        record["passed"] = False
        record["status"] = "FAIL"
        failure_directory = output / "failure"
        failure_arrays = write_failure_arrays(
            failure_directory,
            error,
            "cli_failure",
        )
        raw_checks = write_failure_checks(
            failure_directory,
            error,
            "cli_failure",
        )
        stage = getattr(error, "stage", None)
        record["failure"] = {
            "type": type(error).__name__,
            "code": getattr(error, "code", type(error).__name__),
            "message": str(error),
            "stage": None
            if stage is None
            else {
                "name": stage.stage,
                "step": stage.step,
                "generation": stage.generation,
                "trial": stage.trial,
                "time_s": stage.time,
            },
            "partial_arrays": sorted(
                getattr(error, "partial_arrays", {})
            ),
            "failure_arrays": failure_arrays,
            "raw_checks": raw_checks,
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    write_json(output / "results.json", record)
    print(output / "results.json")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
