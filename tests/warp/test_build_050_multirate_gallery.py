from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = next(
    path
    for path in (
        Path(__file__).parents[1] / "build_050_multirate_gallery.py",
        Path(__file__).parents[2] / "tools/plots/build_050_multirate_gallery.py",
    )
    if path.is_file()
)


@pytest.fixture
def gallery_module():
    spec = importlib.util.spec_from_file_location(
        "build_050_multirate_gallery", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def complete_run(root: Path, name: str, *, nodes: int, coupling: str) -> None:
    case = {
        "id": name,
        "kind": "east",
        "nodes": nodes,
        "K": 1,
        "mode": 1,
        "device": "cuda:0",
        "coupling": coupling,
        "micro_steps": 1 if coupling == "synchronous" else 4,
        "active_cells": nodes * 10,
        "Nphi": 5,
    }
    run = root / name
    write_json(
        run / "report.json",
        {
            "schema": "fomhd-warp-050-experiment-v1",
            "status": "completed",
            "scope": "test control and perturbed response",
            "case": case,
            "h_s": 1e-11,
            "reached_horizon_s": 2e-11,
            "elapsed_s": 3.0,
            "time_plan": {"micro_steps": case["micro_steps"]},
            "memory": {"device_resident_bytes": 1000 + nodes},
        },
    )
    (run / "logs").mkdir()
    (run / "logs/advance-timing.jsonl").write_text(
        json.dumps(
            {
                "ensemble": "control",
                "macro": 1,
                "elapsed_s": float(nodes),
                "status": "completed",
                "scope": "synchronized complete advance",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        run / "memory-inventory.json",
        {
            "array_count": 2,
            "host_resident_bytes": 0,
            "device_resident_bytes": 1000 + nodes,
            "resident_bytes": 1000 + nodes,
            "arrays": [],
        },
    )
    (run / "logs/perturbed-history.jsonl").write_text(
        json.dumps(
            {
                "time_s": 0.0,
                "step": 0,
                "energy": {
                    "total_J": 10.0,
                    "baseline_total_J": 10.0,
                    "field_J": 5.0,
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "time_s": 1e-11,
                "step": 1,
                "energy": {
                    "total_J": 10.5,
                    "baseline_total_J": 10.0,
                    "field_J": 5.25,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def initial_guess_report(
    root: Path,
    suite: str,
    case_id: str,
    *,
    failed: bool,
    device: str = "cpu",
) -> None:
    status = "failed" if failed else "completed"
    history = [
        {"iteration": 0, "trial": 0, "residual": 1.0, "lambda": 0.0},
        {"iteration": 1, "trial": 1, "residual": 1e-3, "lambda": 1.0},
    ]
    guess = {
        "status": status,
        "guess": "constant",
        "solution": not failed,
        "history": history,
        "iterations": 1,
        "initial_residual": 1.0,
        "residual_evaluations": 2,
        "work": {
            "final_execution_producer_rhs_vectors": 20,
            "reference_action_vectors": 10,
            "jvp_rhs_vectors": 0,
        },
    }
    if failed:
        guess.update(
            {
                "failure": {"code": "FieldWindowConvergence"},
                "attempted_elapsed_s": 0.5,
                "timing_label": "stress: no performance claim after failure",
            }
        )
    else:
        guess["timing"] = {"median_s": 0.25}
    write_json(
        root / suite / case_id / "report.json",
        {
            "case": {
                "id": case_id,
                "unit": "E2",
                "points": 33,
                "M": 32,
                "spatial_mode": 16,
                "stress": failed,
            },
            "status": "completed",
            "device": device,
            "harness_completed": True,
            "constant": guess,
            "tangent": {**guess, "guess": "tangent"},
        },
    )


def write_matrix(root: Path, rows: list[dict]) -> None:
    write_json(
        root / "multirate-matrix.json",
        {
            "purpose": "test controlled matrix",
            "configuration_count": len(rows),
            "fixed_periodic_horizon_s": 8e-3,
            "target_T_s": 1.0,
            "amplitude": 0.02,
            "nonuniform": 0.1,
            "rows": rows,
        },
    )


def candidate_report(
    root: Path,
    row: dict,
    *,
    device: str,
    status: str = "completed",
    full_hydro: int = 11,
    target_phase_candidate: float = 0.01,
    target_phase_oracle: float = 0.0,
    target_magnetic_amplitude_candidate: float = 1.01,
    target_magnetic_amplitude_oracle: float = 1.0,
    target_magnetic_phase_candidate: float = 0.0,
    target_magnetic_phase_oracle: float = 0.0,
) -> None:
    directory = root / row["id"]
    config = {
        "case": row["case"],
        "device": "cpu" if device == "CPU" else "cuda:0",
        "method": row["method"],
        "reference": "synchronous",
        "points": row["points"],
        "amplitude": 0.02,
        "nonuniform": 0.1,
        "h": row["h"],
        "M": row["M"],
        "macros": row["macros"],
        "seed_mode": row["seed_mode"],
    }
    write_json(directory / "input.json", config)
    write_json(directory / "provenance.json", {"config": config, "files": {}})
    if status == "missing":
        return
    h = float(row["h"])
    H = h * int(row["M"])
    duration = H * int(row["macros"])

    def history(amplitude: float, phases: list[float]) -> dict:
        return {
            "amplitudes": [0.9 * amplitude, amplitude],
            "phases_rad": phases,
            "final_amplitude": amplitude,
            "maximum_amplitude": 1.02 * amplitude,
            "final_gain": 1.0,
            "maximum_gain": 1.02,
        }

    mode_comparison = None
    if row["case"] == "periodic":
        mode_comparison = {
            "amplitude_ratio": 1.01,
            "phase_error_rad": target_phase_candidate - target_phase_oracle,
            "candidate": {
                "velocity_mode": {
                    "normalized_amplitude": 1.01,
                    "phase_rad": target_phase_candidate,
                },
                "magnetic_mode": {
                    "normalized_amplitude": target_magnetic_amplitude_candidate,
                    "phase_rad": target_magnetic_phase_candidate,
                },
            },
            "oracle": {
                "velocity_mode": {
                    "normalized_amplitude": 1.0,
                    "phase_rad": target_phase_oracle,
                },
                "magnetic_mode": {
                    "normalized_amplitude": target_magnetic_amplitude_oracle,
                    "phase_rad": target_magnetic_phase_oracle,
                },
            },
            "target": {
                "mode": row["seed_mode"],
                "candidate_history": history(1.01, [0.0, target_phase_candidate]),
                "oracle_history": history(1.0, [0.0, target_phase_oracle]),
            },
            "sentinel": {
                "mode": row["points"] // 2,
                "independent": True,
                "candidate_history": {
                    **history(0.5, [0.0, 0.0]),
                    "maximum_gain": 1.04,
                    "final_gain": 0.70,
                },
                "oracle_history": {
                    **history(0.8, [0.0, 0.0]),
                    "maximum_gain": 1.07,
                    "final_gain": 0.65,
                },
            },
        }
    report = {
        "status": status,
        "candidate": row["method"],
        "case": row["case"],
        "points": row["points"],
        "mode": row["seed_mode"],
        "h": h,
        "M": int(row["M"]),
        "H": H,
        "macros": int(row["macros"]),
        "device": config["device"],
        "time_contract": {"total_duration_s": duration},
        "reference": {
            "p_normalized": 2e-5,
            "z_normalized": 3e-5,
        },
        "mode_comparison": mode_comparison,
        "returned_execution_counters": {
            "full_hydro_eval": full_hydro,
            "cheap_D": full_hydro + 1,
            "cheap_L": full_hydro + 2,
            "field_rhs": full_hydro + 3,
        },
        "counters": {
            "full_hydro_eval": 1,
            "cheap_D": 1,
            "cheap_L": 1,
            "field_rhs": 1,
        },
        "timing": {"campaign_multiplier_per_algorithm": 2},
    }
    if status == "failed":
        report["failure_code"] = "TimeStepGuard"
    write_json(directory / "report.json", report)


def guess_summary(root: Path) -> list[str]:
    case_ids = [
        "e2-n33-m128-low-mode1",
        "e2-n33-m32-mode16",
        "e2-n33-m128-mode16-difficult",
        "e3-n12-m16-p1-mode1",
        "e3-n12-m512-h0.5-p1-mode3-difficult",
        "e3-n12-m512-h0.5-p2-mode3-difficult",
    ]
    cases = []
    experiments = []
    pairs = []
    for index, case_id in enumerate(case_ids):
        unit = "E2" if case_id.startswith("e2") else "E3"
        stress = "difficult" in case_id
        cases.append(
            {
                "case_id": case_id,
                "config": {
                    "id": case_id,
                    "unit": unit,
                    "points": 33 if unit == "E2" else 12,
                    "M": 128 if unit == "E2" else 512,
                    "amplitude": 0.02,
                    "nonuniform": 0.1,
                    "spatial_mode": 1,
                },
            }
        )
        for backend in ("cpu", "gpu"):
            statuses = {}
            for guess_index, guess in enumerate(("constant", "tangent")):
                failed = stress and ("e2-" in case_id or "p1-" in case_id)
                statuses[guess] = "failed" if failed else "completed"
                history = [
                    {
                        "iteration": step,
                        "trial": 0,
                        "residual": 10.0 ** (-step - guess_index),
                        "lambda": 0.0,
                    }
                    for step in range(3)
                ]
                experiments.append(
                    {
                        "case_id": case_id,
                        "guess": guess,
                        "backend": backend,
                        "status": statuses[guess],
                        "history": history,
                        "selected_iterate_history": history,
                        "iterations_to_convergence": (None if failed else 2),
                        "selected_correction_iterations_before_stop": 2,
                        "residual_evaluations": 3,
                        "initial_residual": history[0]["residual"],
                        "last_selected_iterate_residual": history[-1]["residual"],
                        "last_trial_residual": history[-1]["residual"],
                        "timing": None if failed else {"median_s": 0.5 + index / 10},
                        "attempted_elapsed_s": 0.75 if failed else None,
                        "timing_label": (
                            "failed attempt is not a benchmark"
                            if failed
                            else "measured"
                        ),
                        "failure": (
                            {"code": "FieldWindowConvergence"} if failed else None
                        ),
                        "audit_G_caveat": (
                            "E3 audit final G is reconstructed and is not an "
                            "accepted true residual."
                            if failed and unit == "E3"
                            else None
                        ),
                        "work": {"final_execution_producer_rhs_vectors": 1000 + index},
                    }
                )
            pairs.append(
                {
                    "case_id": case_id,
                    "backend": backend,
                    "constant_status": statuses["constant"],
                    "tangent_status": statuses["tangent"],
                    "initial_residual_constant_over_tangent": 10.0 + index,
                    "constant_iterations": 2,
                    "tangent_iterations": 2,
                }
            )
    write_json(
        root / "analysis/initial-guesses/summary.json",
        {
            "schema_version": 1,
            "scope": "fixed-P field-window, not coupled T1",
            "method": {
                "failure_residuals": (
                    "The audit final G reconstruction is not an accepted true residual."
                ),
                "input_identity": (
                    "E2 saved inputs differ at backend fixture construction; "
                    "the initial-Z roundoff is measured."
                ),
            },
            "cases": cases,
            "experiments": experiments,
            "constant_tangent_pairs": pairs,
            "limits": ["No coupled acceptance is established."],
        },
    )
    return case_ids


def cost_result(*, baseline: bool) -> dict:
    shift = 0.0 if baseline else -1.0
    return {
        "schema": "fomhd-bounded-full-advance-cost-v1",
        "status": "completed",
        "scope": "test complete advance",
        "claim_policy": {
            "time_parallel_claim": False,
            "same_gpu_t1_algorithm_gain_claim": False,
        },
        "methods": {
            "A0": {
                "plan": {
                    "label": "A0",
                    "coupling": "synchronous",
                    "micro_steps": 1,
                    "h_s": 1e-11,
                    "dt_s": 1e-11,
                    "macros": 8,
                    "horizon_s": 8e-11,
                },
                "measured_window_ms": {"count": 5, "median_ms": 10.0 + shift},
                "cold_total_ms": 12.0 + shift,
            },
            "T1": {
                "plan": {
                    "label": "T1",
                    "coupling": "waveform",
                    "micro_steps": 4,
                    "h_s": 1e-11,
                    "dt_s": 4e-11,
                    "macros": 2,
                    "horizon_s": 8e-11,
                },
                "measured_window_ms": {"count": 5, "median_ms": 8.0 + shift},
                "cold_total_ms": 10.0 + shift,
            },
        },
        "physical_scales": {"rho": 1.0, "B": 1.0},
    }


def fair_cost_report(
    root: Path,
    row: dict,
    *,
    device: str,
    status: str = "completed",
    warmups: int = 1,
    repetitions: int = 3,
    candidate_median_s: float = 1.0,
    lean_median_s: float = 0.5,
    candidate_full_hydro: int = 10,
    lean_full_hydro: int = 12,
    candidate_field_rhs: int = 20,
    lean_field_rhs: int = 24,
) -> None:
    device_value = "cpu" if device == "CPU" else "cuda:0"
    config = {
        "case": row["case"],
        "device": device_value,
        "method": row["method"],
        "reference": "synchronous",
        "points": row["points"],
        "amplitude": 0.02,
        "nonuniform": 0.1,
        "h": row["h"],
        "M": row["M"],
        "macros": row["macros"],
        "seed_mode": row["seed_mode"],
        "rtol": 1e-10,
        "max_iterations": 12,
        "sentinel": 1e-4,
        "endpoint_check": True,
        "warmups": 0,
        "repetitions": 1,
    }
    directory = root / f"fair-cost-{device.lower()}" / row["id"]
    write_json(directory / "source_input.json", config)
    h = float(row["h"])
    M = int(row["M"])
    macros = int(row["macros"])
    report = {
        "status": status,
        "scope": "matched research-window cost; no acceleration claim",
        "source_input": {
            "path": str(root / "candidates-cpu-v2" / row["id"] / "input.json"),
            "sha256": "0" * 64,
            "config": config,
            "actual_config": dict(
                config, device=device_value, warmups=warmups, repetitions=repetitions
            ),
        },
        "provenance": {"files": {}},
        "protocol": {
            "actual_h_s": h,
            "actual_H_s": h * M,
            "actual_M": M,
            "actual_macros": macros,
            "actual_horizon_s": h * M * macros,
            "source_horizon_s": h * M * macros,
            "algorithm_controls_unchanged": True,
            "horizon_unchanged": True,
            "device": device_value,
            "warmups": warmups,
            "repetitions": repetitions,
            "fresh_context_per_execution": True,
            "synchronized_timer_boundaries": True,
        },
        "diagnostics": {"pre": {"scope": "same fixture"}, "post": {}},
        "matched_cost": {
            "candidate": {
                "median_s": candidate_median_s,
                "warmups": warmups,
                "repetitions": repetitions,
                "cold_executions": 1,
            },
            "lean_reference": {
                "median_s": lean_median_s,
                "warmups": warmups,
                "repetitions": repetitions,
                "cold_executions": 1,
            },
            "same_protocol": True,
            "speedup_claim": False,
            "scope": "same-CPU or same-GPU algorithm research-window cost",
        },
        "counters": {
            "candidate": {
                "full_hydro_eval": candidate_full_hydro,
                "field_rhs": candidate_field_rhs,
            },
            "lean": {
                "full_hydro_eval": lean_full_hydro,
                "field_rhs": lean_field_rhs,
            },
        },
        "limitations": [
            "research-window cost only; ratios are not acceleration claims",
            "no accepted transaction, ledger, restart, checkpoint, or FK solver",
        ],
    }
    if status == "failed":
        report["failure_code"] = "CUDARuntimeError"
    write_json(directory / "report.json", report)


def test_input_index_retains_exact_paths_and_sha256(
    gallery_module, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    complete_run(
        run_root, "optimized-east16-synchronous", nodes=16, coupling="synchronous"
    )

    data = gallery_module.collect_gallery_data(run_root, tmp_path / "source")

    report = (run_root / "optimized-east16-synchronous/report.json").resolve()
    indexed = {Path(item["path"]): item for item in data["input_index"]}
    assert report in indexed
    assert indexed[report]["sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    assert indexed[report]["bytes"] == report.stat().st_size


def test_completed_cost_excludes_failed_run_and_failures_stay_explicit(
    gallery_module, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    complete_run(
        run_root, "optimized-east16-synchronous", nodes=16, coupling="synchronous"
    )
    failed = run_root / "optimized-east24-waveform"
    write_json(
        failed / "report.json",
        {
            "schema": "fomhd-warp-050-experiment-v1",
            "status": "failed",
            "case": {
                "kind": "east",
                "nodes": 24,
                "coupling": "waveform",
                "micro_steps": 4,
                "active_cells": 240,
            },
            "failure": {"code": "CUDARuntimeError"},
        },
    )

    data = gallery_module.collect_gallery_data(run_root, tmp_path / "source")
    cost = next(f for f in data["figures"] if f["id"] == "complete-spatial-cost")
    x_values = {
        value
        for panel in cost["panels"]
        for series in panel["series"]
        for value in series["x"]
    }
    assert 240.0 not in x_values
    assert any(
        "optimized-east24-waveform" in note and "failed" in note
        for note in data["availability_notes"]
    )


def test_initial_guess_failures_and_fixed_p_scope_are_explicit(
    gallery_module, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    initial_guess_report(
        run_root,
        "initial-guesses-cpu",
        "e2-n33-m32-mode16",
        failed=False,
    )
    initial_guess_report(
        run_root,
        "initial-guesses-cpu",
        "e2-n33-m128-mode16-difficult",
        failed=True,
    )
    initial_guess_report(
        run_root,
        "initial-guesses-gpu-fixed",
        "e2-n33-m128-mode16-difficult",
        failed=True,
        device="cuda:0",
    )

    data = gallery_module.collect_gallery_data(run_root, tmp_path / "source")
    guess_figure = next(f for f in data["figures"] if f["id"] == "initial-guess")
    text = json.dumps(guess_figure)
    assert "not a coupled T1" in text
    assert "FieldWindowConvergence" in text
    timing_series = guess_figure["panels"][-1]["series"]
    assert all(series["linestyle"] == "None" for series in timing_series)
    assert any("failed" in note for note in data["availability_notes"])


def test_missing_future_sections_are_unavailable_without_numeric_series(
    gallery_module, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    complete_run(
        run_root, "optimized-east16-synchronous", nodes=16, coupling="synchronous"
    )

    data = gallery_module.collect_gallery_data(run_root, tmp_path / "source")

    assert all(
        "placeholder" not in json.dumps(figure).lower() for figure in data["figures"]
    )
    notes = "\n".join(data["availability_notes"])
    assert "Multirate candidate comparisons: unavailable" in notes
    assert "Long-RZ diagnostics: unavailable" in notes
    assert "Shared matched full cost: unavailable" in notes


def test_candidate_series_never_mix_method_control_or_target_mode(
    gallery_module, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    rows = [
        {
            "id": "periodic-C1-M16",
            "case": "periodic",
            "method": "C1",
            "points": 33,
            "h": 1e-3,
            "M": 2,
            "macros": 4,
            "seed_mode": 1,
        },
        {
            "id": "periodic-C1-M32",
            "case": "periodic",
            "method": "C1",
            "points": 33,
            "h": 1e-3,
            "M": 4,
            "macros": 2,
            "seed_mode": 1,
        },
        {
            "id": "periodic-C1-H32-halfh",
            "case": "periodic",
            "method": "C1",
            "points": 33,
            "h": 5e-4,
            "M": 8,
            "macros": 2,
            "seed_mode": 1,
        },
        {
            "id": "periodic-C1-target4",
            "case": "periodic",
            "method": "C1",
            "points": 33,
            "h": 1e-3,
            "M": 4,
            "macros": 2,
            "seed_mode": 4,
        },
        {
            "id": "periodic-C2-M16",
            "case": "periodic",
            "method": "C2",
            "points": 33,
            "h": 1e-3,
            "M": 2,
            "macros": 4,
            "seed_mode": 1,
        },
        {
            "id": "closed8-C1",
            "case": "closed",
            "method": "C1",
            "points": 8,
            "h": 1e-4,
            "M": 2,
            "macros": 4,
            "seed_mode": 1,
        },
        {
            "id": "periodic-C1-M128",
            "case": "periodic",
            "method": "C1",
            "points": 33,
            "h": 1e-3,
            "M": 8,
            "macros": 1,
            "seed_mode": 1,
        },
    ]
    write_matrix(run_root, rows)
    cpu = run_root / "candidates-cpu-v2"
    gpu = run_root / "candidates-gpu"
    candidate_report(
        cpu, rows[0], device="CPU", full_hydro=11, target_phase_candidate=0.0
    )
    candidate_report(
        cpu, rows[1], device="CPU", full_hydro=12, target_phase_candidate=0.0
    )
    candidate_report(
        cpu, rows[2], device="CPU", full_hydro=13, target_phase_candidate=0.0
    )
    candidate_report(
        cpu, rows[3], device="CPU", full_hydro=14, target_phase_candidate=0.0
    )
    candidate_report(
        cpu, rows[4], device="CPU", full_hydro=15, target_phase_candidate=0.0
    )
    candidate_report(cpu, rows[5], device="CPU", full_hydro=16)
    candidate_report(
        gpu, rows[0], device="GPU", full_hydro=21, target_phase_candidate=0.0
    )
    candidate_report(cpu, rows[6], device="CPU", status="missing")
    write_json(
        run_root / "candidates-cpu/periodic-C1-M32/report.json",
        {
            "status": "failed",
            "candidate": "C1",
            "failure_code": "TimeStepGuard",
        },
    )

    data = gallery_module.collect_gallery_data(run_root, tmp_path / "source")

    figure_ids = {figure["id"] for figure in data["figures"]}
    assert "multirate-target-error" in figure_ids
    assert "multirate-mode-gain" in figure_ids
    assert "multirate-work-main" in figure_ids
    target = next(
        figure for figure in data["figures"] if figure["id"] == "multirate-target-error"
    )
    panels = {panel["title"]: panel for panel in target["panels"]}
    assert any("P" in title for title in panels)
    assert any("Z" in title for title in panels)
    assert "Target magnetic amplitude error" in panels
    assert "Target magnetic absolute phase error" in panels
    phase_panel = next(
        panel
        for title, panel in panels.items()
        if "magnetic" in title.lower() and "phase" in title.lower()
    )
    assert all(
        value == 0.0 for series in phase_panel["series"] for value in series["y"]
    )
    assert all(len(panel["series"]) <= 4 for panel in target["panels"])
    assert not any(
        token in series["label"]
        for panel in target["panels"]
        for series in panel["series"]
        for token in ("closed8", "closed12", "H32-halfh", "target4", "2T")
    )

    all_series = [
        series
        for figure in data["figures"]
        if figure["id"].startswith("multirate-")
        for panel in figure["panels"]
        for series in panel["series"]
    ]
    for series in all_series:
        assert not ("C1" in series["label"] and "C2" in series["label"])
        assert not ("CPU" in series["label"] and "GPU" in series["label"])
    hscan = next(
        series
        for series in all_series
        if "C1 CPU" in series["label"]
        and "fixed-h H scan" in series["label"]
        and "P" in series["label"]
    )
    assert hscan["x"] == [2e-3, 4e-3]

    work = next(
        figure for figure in data["figures"] if figure["id"] == "multirate-work-main"
    )
    hydro = next(
        series
        for panel in work["panels"]
        for series in panel["series"]
        if "C1 CPU" in series["label"]
        and "fixed-h H scan" in series["label"]
        and "Full hydro" in series["label"]
    )
    assert hydro["y"] == [11.0, 12.0]
    notes = "\n".join(data["availability_notes"])
    assert "periodic-C1-M128" in notes and "missing" in notes
    assert "candidates-cpu" in notes and "TimeStepGuard" in notes


def test_initial_guess_summary_preserves_all_pairs_and_failure_caveats(
    gallery_module, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    case_ids = guess_summary(run_root)

    data = gallery_module.collect_gallery_data(run_root, tmp_path / "source")

    figure_ids = {figure["id"] for figure in data["figures"]}
    assert "initial-guess-trajectories" in figure_ids
    assert "initial-guess-pairs" in figure_ids
    pair_figure = next(
        figure for figure in data["figures"] if figure["id"] == "initial-guess-pairs"
    )
    assert len(pair_figure["panels"][0]["series"]) == 2
    assert all(
        len(series["x"]) == 6 and len(series["y"]) == 6
        for series in pair_figure["panels"][0]["series"]
    )
    text = json.dumps(
        [
            figure
            for figure in data["figures"]
            if figure["id"].startswith("initial-guess-")
        ]
    )
    for case_id in case_ids:
        assert case_id in text
    notes = "\n".join(data["availability_notes"])
    assert "not a coupled T1" in notes
    assert "audit final G" in notes and "not an accepted true residual" in notes
    assert "backend fixture construction" in notes


def test_full_cost_uses_exact_harness_result_schema(
    gallery_module, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    write_json(
        run_root / "full-cost-baseline/RESULT.json",
        cost_result(baseline=True),
    )
    write_json(
        run_root / "full-cost-optimized/RESULT.json",
        cost_result(baseline=False),
    )
    write_json(
        run_root / "full-cost-baseline/optimized/RESULT.json",
        {
            "baseline": {"complete_cost_s": 1.0},
            "optimized": {"complete_cost_s": 2.0},
        },
    )

    data = gallery_module.collect_gallery_data(run_root, tmp_path / "source")

    figure = next(f for f in data["figures"] if f["id"] == "shared-complete-cost")
    measured = figure["panels"][0]
    assert measured["title"] == "Measured full-advance window"
    by_label = {series["label"]: series for series in measured["series"]}
    assert by_label["baseline source"]["y"] == [10.0, 8.0]
    assert by_label["optimized source"]["y"] == [9.0, 7.0]


def test_long_rz_segments_keep_separate_baselines_and_curves(
    gallery_module, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    for name, h_s, coupling, ensemble, times, totals, baselines in (
        (
            "long-east16-segment-a",
            1e-9,
            "waveform",
            "control",
            [0.0, 1e-8],
            [11.0, 18.0],
            [10.0, 15.0],
        ),
        (
            "long-east16-segment-b",
            2e-9,
            "synchronous",
            "perturbed",
            [2e-8, 3e-8],
            [22.0, 23.0],
            [20.0, 20.0],
        ),
    ):
        write_json(
            run_root / name / "report.json",
            {
                "status": "completed",
                "case": {
                    "nodes": 16,
                    "active_cells": 200,
                    "device": "cuda:0",
                    "coupling": coupling,
                },
                "h_s": h_s,
                "ensemble": ensemble,
                "energy_history": [
                    {
                        "time_s": time_s,
                        "total_J": total,
                        "baseline_total_J": baseline,
                    }
                    for time_s, total, baseline in zip(times, totals, baselines)
                ],
            },
        )

    data = gallery_module.collect_gallery_data(run_root, tmp_path / "source")

    figure = next(f for f in data["figures"] if f["id"] == "long-rz-energy")
    series = figure["panels"][0]["series"]
    by_ensemble = {
        next(name for name in ("control", "perturbed") if name in item["label"]): item
        for item in series
    }
    assert by_ensemble["control"]["y"] == [1.0, 3.0]
    assert by_ensemble["perturbed"]["y"] == [2.0, 3.0]
    assert by_ensemble["control"]["x"] == [0.0, 1e-8]
    assert by_ensemble["perturbed"]["x"] == [2e-8, 3e-8]
    assert "long-east16-segment-a" in figure["caption"]
    assert "long-east16-segment-b" in figure["caption"]


def test_output_guard_rejects_uncontrolled_existing_directory(
    gallery_module, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    source_root = tmp_path / "source"
    run_root.mkdir()
    source_root.mkdir()
    output = tmp_path / "historical-gallery"
    output.mkdir()
    (output / "index.html").write_text("historical", encoding="utf-8")

    with pytest.raises(gallery_module.UnsafeOutputError):
        gallery_module.ensure_output_target(
            output,
            run_root=run_root,
            source_root=source_root,
        )


def test_fair_cost_uses_real_report_schema_and_keeps_devices_separate(
    gallery_module, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    row = {
        "id": "periodic-C1-M16",
        "case": "periodic",
        "method": "C1",
        "points": 33,
        "h": 1e-3,
        "M": 2,
        "macros": 4,
        "seed_mode": 1,
    }
    write_matrix(run_root, [row])
    write_json(
        run_root / "candidates-cpu-v2" / row["id"] / "input.json",
        {
            "case": "periodic",
            "device": "cpu",
            "method": "C1",
            "reference": "synchronous",
            "points": 33,
            "amplitude": 0.02,
            "nonuniform": 0.1,
            "h": 1e-3,
            "M": 2,
            "macros": 4,
            "seed_mode": 1,
            "rtol": 1e-10,
            "max_iterations": 12,
            "sentinel": 1e-4,
            "endpoint_check": True,
            "warmups": 0,
            "repetitions": 1,
        },
    )
    fair_cost_report(
        run_root,
        row,
        device="CPU",
        candidate_median_s=1.0e-3,
        lean_median_s=5.0e-4,
        candidate_full_hydro=10,
        lean_full_hydro=12,
        candidate_field_rhs=20,
        lean_field_rhs=24,
    )
    fair_cost_report(
        run_root,
        row,
        device="GPU",
        candidate_median_s=2.0e-3,
        lean_median_s=1.5e-3,
        candidate_full_hydro=11,
        lean_full_hydro=13,
        candidate_field_rhs=21,
        lean_field_rhs=25,
    )

    data = gallery_module.collect_gallery_data(run_root, tmp_path / "source")

    figures = [f for f in data["figures"] if f["id"] == "fair-primitive-cost"]
    assert len(figures) == 1
    figure = figures[0]
    panels = {panel["title"]: panel for panel in figure["panels"]}
    median = panels["Candidate and lean median research-window time"]
    by_label = {series["label"]: series for series in median["series"]}
    assert by_label["CPU C1 candidate median"]["y"] == [1.0e-3]
    assert by_label["CPU C1 lean median"]["y"] == [5.0e-4]
    assert by_label["GPU C1 candidate median"]["y"] == [2.0e-3]
    assert by_label["GPU C1 lean median"]["y"] == [1.5e-3]
    hydro = panels["Full hydro evaluations"]
    assert {series["label"]: series["y"] for series in hydro["series"]} == {
        "CPU C1 candidate full hydro": [10.0],
        "CPU C1 lean full hydro": [12.0],
        "GPU C1 candidate full hydro": [11.0],
        "GPU C1 lean full hydro": [13.0],
    }
    field = panels["Field RHS calls"]
    assert {series["label"]: series["y"] for series in field["series"]} == {
        "CPU C1 candidate field RHS": [20.0],
        "CPU C1 lean field RHS": [24.0],
        "GPU C1 candidate field RHS": [21.0],
        "GPU C1 lean field RHS": [25.0],
    }
    text = json.dumps(figure)
    assert "cold 1, warmup 1, 3 samples" in text
    assert "not a production transaction" in text
    assert "not an FK acceleration claim" in text
    assert all(
        not ("CPU" in series["label"] and "GPU" in series["label"])
        for panel in figure["panels"]
        for series in panel["series"]
    )


def test_fair_cost_failed_invalid_and_missing_reports_are_explicit(
    gallery_module, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    rows = [
        {
            "id": "periodic-C1-M16",
            "case": "periodic",
            "method": "C1",
            "points": 33,
            "h": 1e-3,
            "M": 2,
            "macros": 4,
            "seed_mode": 1,
        },
        {
            "id": "periodic-C1-M32",
            "case": "periodic",
            "method": "C1",
            "points": 33,
            "h": 1e-3,
            "M": 4,
            "macros": 2,
            "seed_mode": 1,
        },
    ]
    write_matrix(run_root, rows)
    complete_run(
        run_root,
        "optimized-east16-synchronous",
        nodes=16,
        coupling="synchronous",
    )
    fair_cost_report(run_root, rows[0], device="CPU", status="failed")
    fair_cost_report(run_root, rows[1], device="GPU", warmups=2)

    data = gallery_module.collect_gallery_data(run_root, tmp_path / "source")

    assert not any(f["id"] == "fair-primitive-cost" for f in data["figures"])
    notes = "\n".join(data["availability_notes"])
    assert "Fair primitive cost CPU/periodic-C1-M16: failed" in notes
    assert "CUDARuntimeError" in notes
    assert "GPU/periodic-C1-M32: invalid" in notes
    assert "warmups=2" in notes
    assert "Fair primitive cost CPU: 0 completed finite of 2" in notes
    assert "Fair primitive cost GPU: 0 completed finite of 2" in notes
    assert "missing report.json for periodic-C1-M32" in notes
    assert "missing report.json for periodic-C1-M16" in notes


def test_gpu_availability_uses_planned_coverage_revision(
    gallery_module, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    rows = [
        {
            "id": "periodic-C1-M16",
            "case": "periodic",
            "method": "C1",
            "points": 33,
            "h": 1e-3,
            "M": 2,
            "macros": 4,
            "seed_mode": 1,
        },
        {
            "id": "periodic-C1-M32",
            "case": "periodic",
            "method": "C1",
            "points": 33,
            "h": 1e-3,
            "M": 4,
            "macros": 2,
            "seed_mode": 1,
        },
    ]
    write_matrix(run_root, rows)
    matrix_path = run_root / "multirate-matrix.json"
    matrix = json.loads(matrix_path.read_text())
    matrix["GPU_coverage_revision"] = {
        "planned_gpu_cases": ["periodic-C1-M16"],
    }
    write_json(matrix_path, matrix)
    candidate_report(
        run_root / "candidates-gpu",
        rows[0],
        device="GPU",
    )

    data = gallery_module.collect_gallery_data(run_root, tmp_path / "source")

    notes = "\n".join(data["availability_notes"])
    assert "GPU planned coverage: 1/1 completed" in notes
    assert "periodic-C1-M16" in notes
    assert "1 matrix row(s) are not scheduled for GPU" in notes


def test_budget_stopped_long_prefix_uses_global_times_and_archived_baseline(
    gallery_module, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    report = {
        "status": "budget_stopped",
        "run_id": "prefix-a",
        "case": {
            "nodes": 12,
            "active_cells": 113,
            "device": "cuda:0",
            "coupling": "waveform",
        },
        "h_s": 2.0e-9,
        "requested_horizon_s": 2.8e-6,
        "reached_horizon_s": 1.6e-8,
        "planned_macros": 341,
        "reached_macros": 2,
        "segment_probe": {"start_step": 0},
        "restart": {"resume_from": None},
    }
    write_json(run_root / "long-east12-waveform-probe/report.json", report)
    history = [
        {
            "time_s": 0.0,
            "energy": {
                "total_J": 11.0,
                "field_J": 5.0,
                "baseline_total_J": 10.0,
            },
        },
        {
            "time_s": 8.0e-9,
            "energy": {
                "total_J": 12.0,
                "field_J": 6.0,
                "baseline_total_J": 10.0,
            },
        },
        {
            "time_s": 1.6e-8,
            "energy": {
                "total_J": 15.0,
                "field_J": 7.0,
                "baseline_total_J": 10.0,
            },
        },
    ]
    (run_root / "long-east12-waveform-probe/logs").mkdir(parents=True)
    (run_root / "long-east12-waveform-probe/logs/perturbed-history.jsonl").write_text(
        "\n".join(json.dumps(item) for item in history) + "\n",
        encoding="utf-8",
    )

    data = gallery_module.collect_gallery_data(run_root, tmp_path / "source")

    figure = next(f for f in data["figures"] if f["id"] == "long-rz-energy")
    series = figure["panels"][0]["series"]
    assert len(series) == 1
    assert series[0]["x"] == [0.0, 8.0e-9, 1.6e-8]
    assert series[0]["y"] == [1.0, 2.0, 5.0]
    assert series[0]["label"] == "T1 perturbed"
    assert "run_id=prefix-a" in figure["caption"]
    assert "start_step=0" in figure["caption"]
    assert "budget_stopped" in figure["caption"]
    assert "start_step=0" in json.dumps(figure)
    notes = "\n".join(data["availability_notes"])
    assert "reached 1.6e-08 s (2 macros)" in notes
    assert "requested 2.8e-06 s (341 macros)" in notes
    assert "not a complete full-horizon run" in notes


def test_target_figure_separates_quarter_t_scan_and_exposes_magnetic_target4(
    gallery_module, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    scan_row = {
        "id": "periodic-C1-M16",
        "case": "periodic",
        "method": "C1",
        "points": 33,
        "h": 1e-3,
        "M": 2,
        "macros": 4,
        "seed_mode": 1,
    }
    target4_row = {
        "id": "periodic-C1-target4",
        "case": "periodic",
        "method": "C1",
        "points": 33,
        "h": 2e-3,
        "M": 2,
        "macros": 2,
        "seed_mode": 4,
    }
    write_matrix(run_root, [scan_row, target4_row])
    cpu = run_root / "candidates-cpu-v2"
    gpu = run_root / "candidates-gpu"
    candidate_report(cpu, scan_row, device="CPU")
    candidate_report(
        cpu,
        target4_row,
        device="CPU",
        target_phase_candidate=0.03315576464973205,
        target_magnetic_amplitude_candidate=1.0307229690123894,
        target_magnetic_phase_candidate=0.10704171648011618,
    )
    candidate_report(gpu, scan_row, device="GPU")

    data = gallery_module.collect_gallery_data(run_root, tmp_path / "source")

    main = next(
        figure for figure in data["figures"] if figure["id"] == "multirate-target-error"
    )
    assert all(len(panel["series"]) <= 4 for panel in main["panels"])
    assert "target4" not in json.dumps(main)
    assert {
        series["label"].split(":")[0]
        for panel in main["panels"]
        for series in panel["series"]
    } == {"C1 CPU", "C1 GPU"}

    target4 = next(
        figure
        for figure in data["figures"]
        if figure["id"] == "multirate-target4-control"
    )
    panels = {panel["title"]: panel for panel in target4["panels"]}
    velocity_phase = panels["Target velocity absolute phase error"]["series"][0]["y"]
    magnetic_amplitude = panels["Target magnetic amplitude error"]["series"][0]["y"]
    magnetic_phase = panels["Target magnetic absolute phase error"]["series"][0]["y"]
    assert velocity_phase == pytest.approx([0.03315576464973205])
    assert magnetic_amplitude == pytest.approx([0.030722969012389356])
    assert magnetic_phase == pytest.approx([0.10704171648011618])


def test_long_rz_reads_both_ensembles_and_matched_snapshot_rms(
    gallery_module, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    run = run_root / "long-east12-waveform-segment1"
    report = {
        "status": "failed",
        "run_id": "segment-run",
        "case": {
            "nodes": 12,
            "active_cells": 113,
            "device": "cuda:0",
            "coupling": "waveform",
        },
        "h_s": 2.0e-9,
        "requested_horizon_s": 2.8e-6,
        "reached_horizon_s": 3.0e-8,
        "planned_macros": 341,
        "reached_macros": 2,
        "segment_probe": {"start_step": 10},
        "restart": {"resume_from": None},
    }
    write_json(run / "report.json", report)

    def history(total_values: list[float], baseline: float) -> list[dict]:
        return [
            {
                "time_s": time_s,
                "step": step,
                "energy": {
                    "total_J": total,
                    "baseline_total_J": baseline,
                    "balance_J": balance,
                    "outward_J": outward,
                    "source_J": source,
                },
            }
            for time_s, step, total, balance, outward, source in zip(
                [1.0e-8, 2.0e-8, 3.0e-8],
                [10, 11, 12],
                total_values,
                [0.0, -1.0, -2.0],
                [0.0, -0.75, -1.5],
                [0.0, -0.25, -0.5],
            )
        ]

    logs = run / "logs"
    logs.mkdir(parents=True)
    (logs / "control-history.jsonl").write_text(
        "\n".join(json.dumps(item) for item in history([20.0, 21.0, 22.0], 19.0))
        + "\n",
        encoding="utf-8",
    )
    (logs / "perturbed-history.jsonl").write_text(
        "\n".join(json.dumps(item) for item in history([30.0, 31.0, 32.0], 28.0))
        + "\n",
        encoding="utf-8",
    )

    snapshots = run / "snapshots"
    snapshots.mkdir()
    for step, time_s, control_b, perturbed_b, control_v, perturbed_v in (
        (11, 2.0e-8, [3.0, 0.0], [5.0, 0.0], [10.0, 0.0], [13.0, 0.0]),
        (12, 3.0e-8, [3.0, 0.0], [9.0, 0.0], [10.0, 0.0], [14.0, 0.0]),
    ):
        for ensemble, observed_b, observed_v in (
            ("control", control_b, control_v),
            ("perturbed", perturbed_b, perturbed_v),
        ):
            np.savez(
                snapshots / f"{ensemble}-step-{step:08d}.npz",
                step=np.asarray(step, dtype=np.uint64),
                time_s=np.asarray(time_s, dtype=np.float64),
                observed_BN=np.asarray(observed_b, dtype=np.complex128),
                observed_velocity=np.asarray(observed_v, dtype=np.complex128),
            )

    data = gallery_module.collect_gallery_data(run_root, tmp_path / "source")

    figure = next(f for f in data["figures"] if f["id"] == "long-rz-energy")
    panels = {
        panel["title"]: panel
        for f in data["figures"]
        if f["id"].startswith("long-rz-energy")
        for panel in f["panels"]
    }
    raw = panels["Raw energy minus original archived baseline"]
    assert len(raw["series"]) == 2
    assert {series["label"]: series["y"] for series in raw["series"]} == {
        "T1 control": [1.0, 2.0, 3.0],
        "T1 perturbed": [2.0, 3.0, 4.0],
    }
    control_balance = panels["Control energy balance and transfers"]
    assert {series["label"]: series["y"] for series in control_balance["series"]} == {
        "T1 control balance": [0.0, -1.0, -2.0],
        "T1 control outward": [0.0, -0.75, -1.5],
        "T1 control source": [0.0, -0.25, -0.5],
    }
    assert panels["Matched RMS magnetic response"]["series"][0]["y"] == pytest.approx(
        [math.sqrt(2.0), 3.0 * math.sqrt(2.0)]
    )
    assert panels["Matched RMS velocity response"]["series"][0]["y"] == pytest.approx(
        [3.0 / math.sqrt(2.0), 2.0 * math.sqrt(2.0)]
    )
    assert "long-east12-waveform-segment1" in figure["caption"]
    assert "run_id=segment-run" in figure["caption"]
    assert "sha256=" in figure["caption"]
    assert "run_id=segment-run" not in json.dumps(raw["series"])


def test_canonical_panel_split_preserves_target_and_long_components(gallery_module):
    figures = [
        {
            "id": "target",
            "title": "Target",
            "panels": [
                {"title": title}
                for title in (
                    "Full P component error",
                    "Full Z component error",
                    "velocity amplitude",
                    "velocity phase",
                    "magnetic amplitude",
                    "magnetic phase",
                )
            ],
        },
        {
            "id": "long",
            "title": "Long",
            "panels": [{"title": str(index)} for index in range(5)],
        },
    ]
    split = gallery_module._bounded_figures(figures)
    assert all(len(figure["panels"]) <= 4 for figure in split)
    assert sum(len(figure["panels"]) for figure in split) == 11
    assert {p["title"] for f in split if f["id"] == "target" for p in f["panels"]} == {
        "velocity amplitude",
        "velocity phase",
        "magnetic amplitude",
        "magnetic phase",
    }


def test_continuation_plot_join_preserves_global_time_and_rejects_conflict(
    gallery_module,
):
    records = [
        {"label": "T1 control", "x": [0.0, 2.0], "y": [3.0, 5.0]},
        {"label": "T1 control", "x": [2.0, 4.0], "y": [5.0, 9.0]},
    ]
    joined = gallery_module._join_plot_series(records)
    assert joined[0]["x"] == [0.0, 2.0, 4.0]
    assert joined[0]["y"] == [3.0, 5.0, 9.0]
    records[1]["y"][0] = 7.0
    with pytest.raises(gallery_module.InputError, match="Conflicting"):
        gallery_module._join_plot_series(records)


def test_backend_parity_plot_uses_only_compared_physical_endpoints(
    gallery_module, tmp_path
):
    write_json(
        tmp_path / "analysis/multirate/summary.json",
        {
            "CPU_GPU_parity": [
                {
                    "case_id": "C2",
                    "endpoints": {
                        "candidate_p": {
                            "status": "compared",
                            "maximum_normalized": 2e-15,
                        },
                        "candidate_z": {
                            "status": "compared",
                            "maximum_normalized": 3e-16,
                        },
                    },
                },
                {"case_id": "pending", "endpoints": {}},
            ],
            "input_file_sha256": {},
        },
    )
    figure = gallery_module._backend_parity_figure(
        tmp_path, gallery_module.InputIndex()
    )
    assert figure["source_kind"] == "verification"
    assert figure["panels"][0]["series"][0]["y"] == [2e-15]
    assert figure["panels"][0]["series"][1]["y"] == [3e-16]
    assert "pending" not in figure["caption"]


@pytest.mark.parametrize("nested_input", ["run", "source"])
def test_controlled_output_cannot_be_ancestor_of_input(
    gallery_module, tmp_path, nested_input
):
    output = tmp_path / "controlled"
    output.mkdir()
    write_json(
        output / gallery_module.MARKER_NAME, {"schema": gallery_module.GALLERY_SCHEMA}
    )
    run = output / "run" if nested_input == "run" else tmp_path / "elsewhere-run"
    source = (
        output / "source" if nested_input == "source" else tmp_path / "elsewhere-source"
    )
    run.mkdir()
    source.mkdir()
    with pytest.raises(gallery_module.UnsafeOutputError):
        gallery_module.ensure_output_target(output, run_root=run, source_root=source)
    assert run.is_dir() and source.is_dir()


def test_independent_nonoverlapping_long_runs_are_not_joined(gallery_module):
    records = []
    for index in range(2):
        records.append(
            {
                "id": f"long-east12-waveform-independent{index}",
                "coupling": "waveform",
                "run_id": str(index),
                "segment_start_step": 2 * index,
                "h_s": 1.0,
                "status": "budget_stopped",
                "report_sha256": str(index),
                "control_energy_points": [],
                "matched_response_points": [],
                "energy_points": [
                    {"time_s": float(j), "step": j, "total_delta_J": float(j)}
                    for j in (2 * index, 2 * index + 1)
                ],
            }
        )
    with pytest.raises(gallery_module.InputError, match="Unverified"):
        gallery_module._long_figure(records)


def test_verified_continuation_receipt_uses_basename_and_raw_hashes(
    gallery_module, tmp_path
):
    folder = tmp_path / "analysis/long-rz"
    folder.mkdir(parents=True)
    raw = tmp_path / "raw.json"
    raw.write_text("original")
    analysis = folder / "analysis.json"
    write_json(
        analysis,
        {
            "root": str(tmp_path.resolve()),
            "read_issues": [],
            "chains": {"T1": {"segments": ["first", "second"], "integrity_errors": []}},
        },
    )
    write_json(
        folder / "VERIFY_CURRENT.json",
        {
            "read_issues": [],
            "inputs": {str(raw): {"sha256": gallery_module.sha256_file(raw)}},
            "outputs": {
                "analysis.json": {"sha256": gallery_module.sha256_file(analysis)}
            },
        },
    )
    assert gallery_module._verified_long_groups(
        tmp_path, gallery_module.InputIndex()
    ) == {"first": "T1", "second": "T1"}
    payload = json.loads(analysis.read_text())
    receipt_path = folder / "VERIFY_CURRENT.json"
    receipt = json.loads(receipt_path.read_text())
    payload["root"] = str(tmp_path / "other-run")
    write_json(analysis, payload)
    receipt["outputs"]["analysis.json"]["sha256"] = gallery_module.sha256_file(analysis)
    write_json(receipt_path, receipt)
    with pytest.raises(gallery_module.InputError, match="root"):
        gallery_module._verified_long_groups(tmp_path, gallery_module.InputIndex())
    payload["root"] = str(tmp_path.resolve())
    write_json(analysis, payload)
    receipt["outputs"]["analysis.json"]["sha256"] = gallery_module.sha256_file(analysis)
    write_json(receipt_path, receipt)
    raw.write_text("changed")
    with pytest.raises(gallery_module.InputError, match="Unverified"):
        gallery_module._verified_long_groups(tmp_path, gallery_module.InputIndex())


def test_gpu_fair_cost_accepts_cpu_origin_but_not_changed_physics(
    gallery_module, tmp_path
):
    row = {
        "id": "periodic-C2-M64",
        "case": "periodic",
        "method": "C2",
        "points": 33,
        "h": 1e-3,
        "M": 2,
        "macros": 4,
        "seed_mode": 1,
    }
    write_matrix(tmp_path, [row])
    fair_cost_report(tmp_path, row, device="GPU")
    old = tmp_path / "fair-cost-gpu" / row["id"]
    config = json.loads((old / "source_input.json").read_text())
    config["device"] = "cpu"
    report = json.loads((old / "report.json").read_text())
    report["source_input"]["config"] = config
    write_json(old / "source_input.json", config)
    write_json(old / "report.json", report)
    write_json(
        tmp_path / "candidates-gpu" / row["id"] / "input.json",
        dict(config, device="cuda:0", reference="none"),
    )
    records, notes = gallery_module._scan_fair_costs(
        tmp_path, gallery_module.InputIndex()
    )
    assert len(records) == 1 and records[0]["device"] == "GPU", notes
    report["source_input"]["actual_config"]["rtol"] = 1e-3
    write_json(old / "report.json", report)
    records, notes = gallery_module._scan_fair_costs(
        tmp_path, gallery_module.InputIndex()
    )
    assert records == []
    assert any("actual" in note for note in notes)


def test_auxiliary_work_and_gains_separate_observation_horizons(
    gallery_module, tmp_path
):
    short = {
        "id": "periodic-C2-M16",
        "case": "periodic",
        "method": "C2",
        "points": 33,
        "h": 1e-3,
        "M": 2,
        "macros": 4,
        "seed_mode": 1,
    }
    long = dict(short, id="periodic-C2-2T-M16", macros=32)
    write_matrix(tmp_path, [short, long])
    path = tmp_path / "multirate-matrix.json"
    matrix = json.loads(path.read_text())
    matrix["target_T_s"] = 0.032
    write_json(path, matrix)
    candidate_report(tmp_path / "candidates-cpu-v2", short, device="CPU", full_hydro=11)
    candidate_report(tmp_path / "candidates-cpu-v2", long, device="CPU", full_hydro=99)
    data = gallery_module.collect_gallery_data(tmp_path, tmp_path / "source")
    work = {
        f["id"]: f for f in data["figures"] if f["id"].startswith("multirate-work-")
    }
    short_values = work["multirate-work-main"]["panels"][0]["series"][0]["y"]
    long_values = work["multirate-work-2t"]["panels"][0]["series"][0]["y"]
    assert short_values == [11.0] and long_values == [99.0]
    gains = [f for f in data["figures"] if f["id"].startswith("multirate-mode-gain")]
    assert len(gains) == 2 and all(len(f["panels"]) <= 4 for f in gains)
    assert all(len(p["series"]) <= 4 for f in gains for p in f["panels"])
    assert "0.008" in work["multirate-work-main"]["caption"]
    assert "0.064" in work["multirate-work-2t"]["caption"]


def test_stored_magnetic_component_statistic_is_not_vector_magnitude(
    gallery_module, tmp_path
):
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    for i, scale in enumerate((1.0, 2.0)):
        np.savez(
            snapshots / f"perturbed-step-{i:08d}.npz",
            time_s=float(i),
            B_R=np.array([3.0]) * scale,
            B_phi=np.array([4.0]) * scale,
            B_Z=np.array([0.0]),
        )
    points = gallery_module._field_points(tmp_path, gallery_module.InputIndex())
    assert [p["field_amplitude_delta_T"] for p in points] == [0.0, 4.0]
