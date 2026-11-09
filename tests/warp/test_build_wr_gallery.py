"""Portable behavioral tests for the bounded current WR gallery builder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = SOURCE_ROOT / "tools/plots/build_wr_gallery.py"


def load_builder():
    assert MODULE_PATH.is_file(), f"gallery builder is absent: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location("wr_gallery_builder", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def wr_evidence_root():
    value = os.environ.get("FOMHD_WR_EVIDENCE_ROOT")
    if not value:
        pytest.skip("FOMHD_WR_EVIDENCE_ROOT is not set")
    root = Path(value).expanduser()
    if not root.is_dir():
        pytest.skip(f"FOMHD_WR_EVIDENCE_ROOT is missing: {root}")
    return root.resolve()


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _tiny_inputs(root: Path):
    raw = root / "raw"
    raw.mkdir()
    cfl = {
        "rows": [
            {
                "case": "uniform",
                "subsystem": "full",
                "max_real_s_inv": 1.0e-4,
                "omega_h": 0.2,
                "imaginary_axis_RK4_bound": 2.8284271247461903,
                "linear_spectral_RK4_max_abs": 1.0,
            },
            {
                "case": "nonuniform",
                "subsystem": "fixed",
                "max_real_s_inv": 2.0e5,
                "omega_h": 0.4,
                "imaginary_axis_RK4_bound": 2.8284271247461903,
                "linear_spectral_RK4_max_abs": 1.0001,
            },
        ],
        "scope": "tiny local audit",
    }
    micro = {
        "rows": [
            {
                "mode": 1,
                "reference_h_h2_relative": 2.0e-10,
                "phase_difference_rad": -1.0e-11,
            },
            {
                "mode": 16,
                "reference_h_h2_relative": 0.5,
                "phase_difference_rad": -0.1,
            },
        ],
        "scope": "tiny reference response",
    }
    linear = {
        "rows": [
            {"s": 3, "M": 1, "outer_picard_radius": 1e-5, "error": 1e-5, "phase": -1e-5},
            {"s": 3, "M": 2, "outer_picard_radius": 2e-5, "error": 2e-5, "phase": -2e-5},
            {"s": 5, "M": 1, "outer_picard_radius": 1e-6, "error": 2e-6, "phase": -2e-6},
            {"s": 5, "M": 2, "outer_picard_radius": 2e-6, "error": 3e-6, "phase": -3e-6},
        ],
        "scope": "tiny analytic selection",
    }
    backend = [
        {"name": "P", "relative": 1e-15, "abs_error": 1e-20},
        {"name": "Z", "relative": 2e-15, "abs_error": 2e-20},
    ]
    parity = {
        "wr": {
            "cases": [
                {
                    "case": "nonlinear",
                    "s": 3,
                    "M": 3,
                    "iterations": 2,
                    "work": {"field_callbacks": 13, "transport_callbacks": 4},
                    "timing": {"cpu_seconds": 1e-4, "cuda_seconds": 2e-4},
                },
                {
                    "case": "hall",
                    "s": 5,
                    "M": 8,
                    "iterations": 2,
                    "work": {"field_callbacks": 33, "transport_callbacks": 6},
                    "timing": {"cpu_seconds": 2e-4, "cuda_seconds": 3e-4},
                },
            ]
        },
        "massless": {
            "strict_k1_failure_cpu": {
                "primitive": {"raw": 32241.27, "threshold": 20000.0},
                "legacy_full_cpu": {"raw": 32241.27, "threshold": 20000.0},
            },
            "strict_k1_failure_cuda": {
                "primitive": {"raw": 46277.80, "threshold": 20000.0}
            },
            "cases": [
                {
                    "fixture": "periodic_nonuniform",
                    "scope": "positive_hard_gate",
                    "comparisons": [
                        {"name": "ue", "normalized_error": 1e-12},
                        {"name": "B", "normalized_error": 2e-12},
                    ],
                }
            ],
            "gate_scope_complete": False,
        },
        "gate_scope_complete": False,
        "status": "numerical_consistency_passed",
    }
    reviewed = root / "reviewed-sections.html"
    reviewed.write_text("<!doctype html><title>reviewed sections</title>\n")
    paths = {
        "cfl": raw / "cfl.json",
        "microstep_reference": raw / "micro.json",
        "linear_selection": raw / "linear.json",
        "backend_comparison": raw / "backend.json",
        "parity_v3": raw / "parity.json",
    }
    for name, value in (
        ("cfl", cfl),
        ("microstep_reference", micro),
        ("linear_selection", linear),
        ("backend_comparison", backend),
        ("parity_v3", parity),
    ):
        _write_json(paths[name], value)
    manifest = {
        "schema": "fomhd-wr-gallery-manifest-v1",
        "title": "Tiny WR gallery",
        "introduction": "Offline gallery contract test.",
        "conclusions": "Only listed data are shown.",
        "raw_root": str(raw),
        "sources": {name: str(path) for name, path in paths.items()}
        | {"reviewed_sections": str(reviewed)},
        "availability": [
            {"id": "full_periodic_2t", "status": "blocked", "detail": "not supplied"},
            {"id": "east_rz", "status": "blocked", "detail": "not supplied"},
        ],
    }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, paths


def _optional_inputs(root: Path):
    components = [1.0, 2.0, 3.0, 4.0, 5.0, 0.0]

    def metrics(scale: float):
        return {
            "absolute_component_abs_error": [
                scale * value for value in components
            ],
            "amplitude_ratio": 1.0 + scale,
            "history_proxy_amplitude_ratio": 1.0,
            "history_proxy_phase_rad": 0.0,
            "history_proxy_vector_relative_error": 0.0,
            "normalized_difference_norm": scale,
            "phase_rad": -scale / 10.0,
            "phase_within_reported_budget": True,
            "vector_relative_error": scale,
            "vector_within_reported_budget": True,
        }

    schemes = {
        "sync": {"coupling": "synchronous", "device": "cpu", "M": 1, "s": 3, "h": 1.0},
        "m4": {"coupling": "multirate_wr", "device": "cpu", "M": 4, "s": 3, "h": 1.0},
        "m8": {"coupling": "multirate_wr", "device": "cpu", "M": 8, "s": 3, "h": 1.0},
        "gpu_m8": {"coupling": "multirate_wr", "device": "gpu", "M": 8, "s": 3, "h": 1.0},
        "s5": {"coupling": "multirate_wr", "device": "cpu", "M": 8, "s": 5, "h": 1.0},
        "t1": {"coupling": "waveform", "device": "cpu", "M": 8, "s": 3, "h": 1.0},
        "fixed": {"coupling": "multirate_wr", "device": "cpu", "M": 16, "s": 3, "h": 0.5},
    }
    roles = {
        "sync": "matched_synchronous_reference",
        "m4": "multirate_wr_s3_M4_candidate",
        "m8": "multirate_wr_s3_M8_candidate",
        "gpu_m8": "gpu_multirate_wr_s3_M8_candidate",
        "s5": "multirate_wr_s5_M8_candidate",
        "t1": "waveform_T1_candidate",
        "fixed": "fixed_H_half_h_cost_and_endpoint_candidate",
    }
    cases = {
        ident: {
            "role": roles[ident],
            "status": "complete",
            "scheme": scheme,
            "clock": {
                "completed_macros": 1,
                "total_macros": 1,
                "progress_fraction": 1.0,
                "progress_microsteps": scheme["M"],
                "step": 1,
                "time": 2.0,
            },
            "diagnostics": {"checks": {"hard_failed": 0}},
        }
        for ident, scheme in schemes.items()
    }
    candidate_ids = ("m4", "m8", "gpu_m8", "s5", "t1", "fixed")
    periodic = {
        "schema": "fomhd-periodic-summary-v2",
        "scope": {
            "nominal_horizon": "2T campaign label; nonlinear mixture",
            "comparison_policy": "no endpoint subtraction across different final time",
        },
        "manifest": {"reference_case": "sync"},
        "metric_contract": {
            "absolute_component_order": [
                "u_R",
                "u_phi",
                "u_Z",
                "B_R",
                "B_phi",
                "B_Z",
            ],
            "no_mixed_unit_absolute_norm": True,
            "budget_status": "metric only; not a universal scientific pass",
        },
        "contracts": {
            "all_comparison_sources_admitted": True,
            "all_complete_physical_contracts_exact": True,
            "cpu_source_arrays_exact": True,
            "physical_parameter_mismatches": {},
        },
        "cases": cases,
        "comparisons": {
            ident: {
                "available": True,
                "join": {
                    "joined": True,
                    "progress_fraction": 1.0,
                    "time": 2.0,
                },
                "physical_parameter_mismatches": [],
            }
            for ident in candidate_ids
        },
        "target_mode_metrics": {
            "target": {
                "mode": 1,
                "reference_physical_evolution": {"normalized_norm_change": 0.0},
                "candidates": {
                    ident: metrics((index + 1) * 1e-6)
                    for index, ident in enumerate(candidate_ids)
                },
            },
            "sentinel": {
                "mode": 16,
                "reference_physical_evolution": {"normalized_norm_change": 0.0},
                "candidates": {
                    ident: metrics((index + 1) * 1e-3)
                    for index, ident in enumerate(candidate_ids)
                },
            },
        },
    }

    def cost_method(method, micro_steps, ion_nodes, median):
        samples = [median - 0.1, median, median + 0.1]
        return {
            "status": "accepted",
            "identity": {
                "method": method,
                "micro_steps": micro_steps,
                "ion_nodes": ion_nodes,
            },
            "context_setup_seconds": 0.05,
            "ledger_resets_inside_physical_run": 0,
            "warmup": {
                "run_seconds": median,
                "reset_seconds": 0.01,
                "setup_seconds": 0.02,
                "total_seconds": median + 0.03,
            },
            "run_seconds": {
                "samples": samples,
                "median": median,
                "minimum": samples[0],
                "maximum": samples[-1],
            },
        }

    cost = {
        "schema": "fomhd-matched-wr-cost-v1",
        "status": "completed",
        "config": {
            "device": "cuda:0",
            "points": 33,
            "horizon_microsteps": 8,
            "horizon_seconds": 1.0e-9,
            "warmup": 1,
            "repeats": 3,
        },
        "limits": {
            "warmup": 1,
            "repeats": 3,
            "warmup_semantics": "one synchronized full-horizon warmup",
        },
        "physical_input_consistent": True,
        "all_initial_inputs_nontrivial": True,
        "work_counter_policy": "raw counters remain separate and are never summed",
        "speedup_claim": None,
        "methods": [
            cost_method("synchronous", 1, None, 3.0),
            cost_method("waveform", 8, None, 8.0),
            cost_method("multirate_wr", 8, 3, 4.0),
            cost_method("multirate_wr", 8, 5, 5.0),
        ],
    }
    periodic_path = root / "periodic-comparison.json"
    cost_path = root / "matched-gpu-cost.json"
    _write_json(periodic_path, periodic)
    _write_json(cost_path, cost)
    return periodic_path, cost_path


def _boundary_inputs(root: Path):
    precheck_root = root / "precheck"
    analysis = precheck_root / "analysis"
    analysis.mkdir(parents=True)

    def write_case(name, coupling, micro_steps, h, outcome, work, seconds, checks):
        output = precheck_root / "periodic" / name
        output.mkdir(parents=True)
        run = {
            "schema": "fomhd-periodic-wr-campaign-v1",
            "status": "partial" if outcome == "acceptedprefix" else "failed",
            "completed_macros": 1 if outcome == "acceptedprefix" else 0,
            "total_macros": 128,
            "stop_reason": "segment" if outcome == "acceptedprefix" else None,
            "identity": {
                "scheme": {
                    "schema": "fomhd-periodic-wr-campaign-v1",
                    "device": "cpu",
                    "coupling": coupling,
                    "M": micro_steps,
                    "s": 3 if "s5" not in name else 5,
                    "h": h,
                    "total_microsteps": 4096,
                }
            },
        }
        _write_json(output / "run.json", run)
        if outcome == "acceptedprefix":
            history = {
                "completed_macros": 1,
                "work": work,
                "timing": {"advance_seconds": seconds},
                "checks": {"hard_failed": 0},
            }
            (output / "history.jsonl").write_text(json.dumps(history) + "\n")
        else:
            (output / "history.jsonl").write_text("")
            failure = {
                "code": "HardGate",
                "failure_seconds": seconds,
                "work": work,
                "checks": {"failures": checks},
                "coupling_metadata": {
                    "residual_history": [{"converged": True}],
                },
            }
            _write_json(output / "failure.json", failure)
        return output

    def work(coarse, stages, fields):
        return {
            "coarse_node_calls": coarse,
            "full_stage_calls": stages,
            "field_evaluations": fields,
            "full_ion_calls": 2 * stages,
            "full_rhs_calls": stages,
        }

    base_a0 = write_case(
        "cpu-precheck33-synchronous-M1-s3",
        "synchronous",
        1,
        1.0,
        "acceptedprefix",
        work(0, 5, 0),
        0.05,
        [],
    )
    base_s3 = write_case(
        "cpu-precheck33-multirate_wr-M8-s3",
        "multirate_wr",
        8,
        1.0,
        "acceptedprefix",
        work(10, 12, 97),
        0.5,
        [],
    )
    reject_s3 = write_case(
        "cpu-precheck33-multirate_wr-M16-s3",
        "multirate_wr",
        16,
        1.0,
        "converged-hard-rejected",
        work(13, 15, 257),
        1.5,
        [{"name": "field_energy_B_interval", "hard": True, "raw": -1.2e-8, "threshold": 8e-9}],
    )
    reject_s5 = write_case(
        "cpu-precheck33-multirate_wr-M32-s5",
        "multirate_wr",
        32,
        1.0,
        "converged-hard-rejected",
        work(21, 23, 513),
        2.5,
        [{"name": "field_energy_B_interval", "hard": True, "raw": -2.0e-8, "threshold": 8e-9}],
    )
    half_s3 = write_case(
        "cpu-fixedH-boundary-halfh-M32-s3",
        "multirate_wr",
        32,
        0.5,
        "converged-hard-rejected",
        work(13, 15, 513),
        2.6,
        [{"name": "kinetic_energy", "hard": True, "raw": -2.2e-8, "threshold": 8e-9}],
    )
    half_s5 = write_case(
        "cpu-fixedH-boundary-halfh-M64-s5",
        "multirate_wr",
        64,
        0.5,
        "acceptedprefix",
        work(21, 23, 1025),
        5.4,
        [],
    )
    production = [
        {
            "name": base_a0.name,
            "returncode": 0,
            "status": "partial",
            "output": str(base_a0),
            "advance_seconds": 0.05,
        },
        {
            "name": base_s3.name,
            "returncode": 0,
            "status": "partial",
            "output": str(base_s3),
            "advance_seconds": 0.5,
        },
        {
            "name": reject_s3.name,
            "returncode": 4,
            "status": "failed",
            "output": str(reject_s3),
            "advance_seconds": 0.0,
        },
        {
            "name": reject_s5.name,
            "returncode": 4,
            "status": "failed",
            "output": str(reject_s5),
            "advance_seconds": 0.0,
        },
    ]
    boundary = [
        {
            "name": half_s3.name,
            "returncode": 4,
            "status": "failed",
            "command": ["python", "warp_wr_periodic.py", "--output", str(half_s3)],
        },
        {
            "name": half_s5.name,
            "returncode": 0,
            "status": "partial",
            "command": ["python", "warp_wr_periodic.py", "--output", str(half_s5)],
        },
    ]
    production_path = analysis / "cpu-production-precheck.json"
    boundary_path = analysis / "fixedH-boundary-precheck.json"
    _write_json(production_path, production)
    _write_json(boundary_path, boundary)
    return production_path, boundary_path


def test_builds_self_contained_canonical_gallery_with_actual_input_hashes(tmp_path):
    builder = load_builder()
    manifest, inputs = _tiny_inputs(tmp_path)
    before = {name: path.read_bytes() for name, path in inputs.items()}
    output = tmp_path / "gallery-v1"

    result = builder.build_gallery(manifest, output, SOURCE_ROOT)

    assert result["status"] == "ok"
    assert result["figures"] == 6
    html = (output / "index.html").read_text()
    assert html.count("data:image/svg+xml;base64,") == 6
    assert "http://" not in html and "https://" not in html
    assert 'id="availability"' in html
    assert "blocked" in html
    assert "reviewed sections" in html or "reviewed-sections.html" in html
    for required in ("data.json", "manifest.json", "provenance.json", "SHA256SUMS"):
        assert (output / required).is_file()
    assert len(list(output.glob("*.png"))) == 6
    assert len(list(output.glob("*.svg"))) == 6

    provenance = json.loads((output / "provenance.json").read_text())
    by_path = {item["path"]: item for item in provenance["inputs"]}
    for role, path in inputs.items():
        expected = hashlib.sha256(before[role]).hexdigest()
        assert by_path[str(path.resolve())]["sha256"] == expected
        assert path.read_bytes() == before[role]
    assert "build_wr_gallery.py" in provenance["source_hashes"]
    assert "tools/plots/build_a0_report.py" in provenance["source_hashes"]


def test_figure_contracts_show_question_fixed_scan_reference_and_pass_scope(tmp_path):
    builder = load_builder()
    manifest, _ = _tiny_inputs(tmp_path)
    output = tmp_path / "gallery"
    builder.build_gallery(manifest, output, SOURCE_ROOT)
    html = (output / "index.html").read_text()
    data = json.loads((output / "data.json").read_text())

    assert len(data["figures"]) == 6
    assert all(1 <= len(figure["panels"]) <= 4 for figure in data["figures"])
    for text in ("Question:", "Fixed:", "Scan:", "Reference:", "Pass scope:"):
        assert text in html
    assert "strict original k1" in html.casefold()
    assert "periodic" in html
    assert "serial synthetic" in html
    assert "local linearization" in html.casefold()


def test_missing_required_source_fails_before_output_creation(tmp_path):
    builder = load_builder()
    manifest, _ = _tiny_inputs(tmp_path)
    value = json.loads(manifest.read_text())
    value["sources"]["parity_v3"] = str(tmp_path / "missing.json")
    manifest.write_text(json.dumps(value))
    output = tmp_path / "missing-output"

    with pytest.raises(builder.InputError):
        builder.build_gallery(manifest, output, SOURCE_ROOT)
    assert not output.exists()


def test_optional_wall_response_addendum_is_hashed_and_linked(tmp_path):
    builder = load_builder()
    manifest, _ = _tiny_inputs(tmp_path)
    wall = tmp_path / "wall-response.html"
    wall.write_text("<!doctype html><title>wall response</title>\n")
    value = json.loads(manifest.read_text())
    value["sources"]["wall_response_addendum"] = str(wall)
    manifest.write_text(json.dumps(value, indent=2))
    output = tmp_path / "gallery-wall"

    result = builder.build_gallery(manifest, output, SOURCE_ROOT)

    assert result["figures"] == 6
    data = json.loads((output / "data.json").read_text())
    issue = next(
        item
        for item in data["known_issues"]
        if item["label"].startswith("Wall-adjacent response audit")
    )
    assert issue["path"] == str(wall.resolve())
    html = (output / "index.html").read_text()
    assert "Wall-adjacent response audit" in html
    assert "raw states, initialization and shared-scale maps" in html
    assert "wall-response.html" in html
    provenance = json.loads((output / "provenance.json").read_text())
    record = next(
        item
        for item in provenance["inputs"]
        if item["path"] == str(wall.resolve())
    )
    assert "wall_response_addendum" in record["roles"]


def test_output_inside_raw_root_is_rejected(tmp_path):
    builder = load_builder()
    manifest, _ = _tiny_inputs(tmp_path)
    output = tmp_path / "raw/gallery"

    with pytest.raises(builder.UnsafeOutputError):
        builder.build_gallery(manifest, output, SOURCE_ROOT)
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path):
    builder = load_builder()
    manifest, _ = _tiny_inputs(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("keep\n")

    with pytest.raises(FileExistsError):
        builder.build_gallery(manifest, output, SOURCE_ROOT)
    assert marker.read_text() == "keep\n"
    assert sorted(path.name for path in output.iterdir()) == ["marker.txt"]


def test_optional_periodic_comparison_and_matched_cost_figures_are_explicit(
    tmp_path,
):
    builder = load_builder()
    manifest, _ = _tiny_inputs(tmp_path)
    periodic_path, cost_path = _optional_inputs(tmp_path)
    value = json.loads(manifest.read_text())
    value["sources"]["periodic_comparison"] = str(periodic_path)
    value["sources"]["matched_gpu_cost"] = str(cost_path)
    manifest.write_text(json.dumps(value, indent=2))

    builder.build_gallery(manifest, tmp_path / "gallery", SOURCE_ROOT)

    data = json.loads((tmp_path / "gallery/data.json").read_text())
    figures = {figure["id"]: figure for figure in data["figures"]}
    assert {
        "periodic-target-mode1",
        "periodic-sentinel-mode16",
        "periodic-fixed-h-h2",
        "matched-gpu-cost-8h",
    } <= set(figures)

    target = figures["periodic-target-mode1"]
    sentinel = figures["periodic-sentinel-mode16"]
    fixed = figures["periodic-fixed-h-h2"]
    assert target["source_kind"] == "verification"
    assert sentinel["source_kind"] == "verification"
    assert fixed["source_kind"] == "verification"
    assert target["device"] == "CPU and GPU"
    assert "signed phase error" in target["panels"][1]["title"].casefold()
    assert "signed phase error" in sentinel["panels"][1]["title"].casefold()
    assert all(value < 0.0 for value in target["panels"][1]["series"][0]["y"])
    target_labels = {
        series["label"]
        for panel in target["panels"]
        for series in panel["series"]
    }
    assert "CPU WR s3 M16 fixed-H h/2" in target_labels
    assert "GPU WR s3 M8" in target_labels
    assert target["panels"][0]["xlabel"] == "M microsteps per advance"
    assert "nominal 2T" in target["controls"]
    assert "nonlinear mixture" in target["controls"]
    assert "2T" in target["controls"] or "2T" in target["caption"]
    assert "grid-high sentinel" in sentinel["controls"].casefold()
    assert "not macro instability" in sentinel["controls"].casefold()
    assert {
        tuple(panel["series"][0]["x"])
        for panel in fixed["panels"]
    } == {(8.0, 16.0)}

    cost = figures["matched-gpu-cost-8h"]
    cost_labels = {series["label"] for series in cost["panels"][0]["series"]}
    assert cost_labels == {
        "A0 synchronous M1",
        "T1 waveform M8",
        "WR s3 M8",
        "WR s5 M8",
    }
    cost_text = f"{cost['controls']} {cost['caption']}".casefold()
    for required in (
        "warmup",
        "reset_seconds",
        "setup_seconds",
        "median complete-advance",
        "ion transport producer",
        "stage evaluator",
        "full rhs finish",
        "not a complete-rhs reduction",
    ):
        assert required in cost_text


def test_first_window_boundary_controls_use_referenced_records(tmp_path):
    builder = load_builder()
    manifest, _ = _tiny_inputs(tmp_path)
    production_path, boundary_path = _boundary_inputs(tmp_path)
    value = json.loads(manifest.read_text())
    value["sources"]["production_precheck"] = str(production_path)
    value["sources"]["boundary_precheck"] = str(boundary_path)
    manifest.write_text(json.dumps(value, indent=2))

    builder.build_gallery(manifest, tmp_path / "gallery", SOURCE_ROOT)

    output = tmp_path / "gallery"
    data = json.loads((output / "data.json").read_text())
    figures = {figure["id"]: figure for figure in data["figures"]}
    assert {"first-window-boundary-work", "first-window-energy-rejection"} <= set(
        figures
    )
    work = figures["first-window-boundary-work"]
    assert len(work["panels"]) == 4
    assert work["title"] == "First-window macro-step acceptance controls"
    labels = {
        series["label"]
        for panel in work["panels"]
        for series in panel["series"]
    }
    assert "base h WR s3" in labels
    assert "base h WR s5" in labels
    assert "fixed H h/2 WR s3" in labels
    assert all("A0" not in label and "T1" not in label for label in labels)
    assert "Stage evaluator calls" in {
        panel["title"] for panel in work["panels"]
    }
    assert "first-window" in work["caption"].casefold()
    assert "unscanned" in work["caption"].casefold()
    assert "converged-hard-rejected" in work["caption"]
    assert "acceptedprefix" in work["caption"]

    energy = figures["first-window-energy-rejection"]
    panel = energy["panels"][0]
    assert energy["title"] == "First-window energy-gate rejection control"
    assert panel["xlabel"] == "Effective base-h M = M*h/h_base"
    assert panel["horizontal_lines"][0]["y"] == 1.0
    assert {series["label"] for series in panel["series"]} == {
        "base h WR s3 field energy",
        "base h WR s5 field energy",
        "fixed H h/2 WR s3 ion kinetic",
    }
    assert "not zero-filled" in energy["caption"]
    assert "no stability bound" in energy["caption"].casefold()
    assert "not a real spatial wall-boundary" in work["caption"]
    assert "not a real spatial wall-boundary" in energy["caption"]

    provenance = json.loads((output / "provenance.json").read_text())
    paths = {Path(record["path"]) for record in provenance["inputs"]}
    referenced = set()
    for source in (production_path, boundary_path):
        for row in json.loads(source.read_text()):
            output_path = row.get("output")
            if output_path is None:
                command = row["command"]
                output_path = command[command.index("--output") + 1]
            output_path = Path(output_path)
            for name in ("run.json", "history.jsonl", "failure.json"):
                candidate = output_path / name
                if candidate.exists():
                    referenced.add(candidate.resolve())
    assert referenced <= paths


def _real_rz_entry(run_root: Path):
    root = run_root / "rz/cpu-production-probe12-M4-s3"
    return {
        "id": "east12-probe",
        "label": "EAST12 one-macro probe",
        "backend": "cpu",
        "report": str(root / "report.json"),
        "input_mapping": str(root / "input_mapping.npz"),
        "sample_index": 0,
        "snapshots": {
            "control_initial": str(root / "snapshots/control-initial.npz"),
            "control_final": str(root / "snapshots/control-macro-000001.npz"),
            "perturbed_initial": str(root / "snapshots/perturbed-initial.npz"),
            "perturbed_final": str(root / "snapshots/perturbed-macro-000001.npz"),
        },
        "checkpoints": {
            "control": str(root / "checkpoints/control.npz"),
            "perturbed": str(root / "checkpoints/perturbed.npz"),
        },
    }


def _read_real_rz(builder, run_root: Path, entry: dict):
    sections = builder._module(
        "wr_gallery_test_sections",
        SOURCE_ROOT / "tools/plots/build_050_sections.py",
    )
    return builder._read_rz_runs(
        {"rz_runs": [entry]},
        run_root,
        builder.InputIndex(),
        sections,
    )[0]


def test_real_rz_reader_matches_both_stages_without_optional_checkpoints(
    wr_evidence_root,
):
    builder = load_builder()
    entry = _real_rz_entry(wr_evidence_root)
    entry.pop("checkpoints")

    run = _read_real_rz(builder, wr_evidence_root, entry)

    assert run["reached_macros"] == 1
    for stage in ("initial", "final"):
        control = run["snapshots"][f"control_{stage}"]
        perturbed = run["snapshots"][f"perturbed_{stage}"]
        assert control["step"] == perturbed["step"]
        assert control["time_s"] == pytest.approx(perturbed["time_s"])


def test_real_rz_reader_rejects_mismatched_snapshot_times(wr_evidence_root):
    builder = load_builder()
    entry = _real_rz_entry(wr_evidence_root)
    entry["snapshots"]["perturbed_initial"] = entry["snapshots"]["perturbed_final"]

    with pytest.raises(
        builder.InputError,
        match="initial control/perturbed snapshots are not matched",
    ):
        _read_real_rz(builder, wr_evidence_root, entry)


@pytest.mark.parametrize(
    "report_status,reached,requested,expected",
    [
        ("completed", 8, 8, "completed"),
        ("segment_complete", 1, 44, "segment"),
        ("budget_stopped", 2, 44, "partial"),
        ("failed", 3, 8, "failure"),
    ],
)
def test_rz_status_is_derived_from_report(report_status, reached, requested, expected):
    builder = load_builder()
    report = {
        "status": report_status,
        "reached_macros": reached,
        "global_requested_macros": requested,
    }

    assert builder._rz_status(report) == expected


def test_rz_completed_status_rejects_inconsistent_macro_count():
    builder = load_builder()
    report = {
        "status": "completed",
        "reached_macros": 7,
        "global_requested_macros": 8,
    }

    with pytest.raises(builder.InputError, match="completed report has not reached"):
        builder._rz_status(report)


def test_rz_panel_layout_uses_real_middle_without_fake_interval():
    builder = load_builder()

    assert builder._rz_panel_layout(["B_phi"], False) == [
        ("initial", "B_phi", False),
        ("initial", "B_phi", True),
        ("final", "B_phi", False),
        ("final", "B_phi", True),
    ]
    assert builder._rz_panel_layout(["B_phi"], True) == [
        ("initial", "B_phi", True),
        ("middle", "B_phi", True),
        ("final", "B_phi", True),
        ("final", "B_phi", False),
    ]
    assert builder._rz_panel_layout(["B_R", "B_Z"], True) == [
        ("final", "B_R", False),
        ("final", "B_R", True),
        ("final", "B_Z", False),
        ("final", "B_Z", True),
    ]


def test_rz_caption_uses_actual_active_cell_count_and_report_status():
    builder = load_builder()
    run = {
        "backend": "cuda",
        "label": "GPU16 tight probe",
        "nodes": 16,
        "s": 3,
        "M": 8,
        "h": 1.0e-11,
        "H": 8.0e-11,
        "status_label": "completed",
        "reached_macros": 8,
        "requested_macros": 8,
        "sample_index": 0,
        "nphi": 5,
        "nc": 16,
    }

    caption = builder._rz_caption(run, "B_phi")

    assert "16-cell active mask" in caption
    assert "113-cell" not in caption
    assert "status=completed" in caption
    assert "8 of 8 requested macros" in caption


def test_rz_middle_snapshots_must_be_supplied_as_a_matched_pair(wr_evidence_root):
    builder = load_builder()
    entry = _real_rz_entry(wr_evidence_root)
    entry["snapshots"]["control_middle"] = entry["snapshots"]["control_final"]

    with pytest.raises(builder.InputError, match="middle snapshots must be supplied together"):
        _read_real_rz(builder, wr_evidence_root, entry)


def test_rz_middle_snapshots_must_lie_between_initial_and_final(wr_evidence_root):
    builder = load_builder()
    entry = _real_rz_entry(wr_evidence_root)
    entry["snapshots"]["control_middle"] = entry["snapshots"]["control_final"]
    entry["snapshots"]["perturbed_middle"] = entry["snapshots"]["perturbed_final"]

    with pytest.raises(builder.InputError, match="strictly ordered initial/middle/final"):
        _read_real_rz(builder, wr_evidence_root, entry)


def test_real_periodic_trial_and_rz_probe_render_actual_physical_pages(
    tmp_path,
    wr_evidence_root,
):
    builder = load_builder()
    manifest = {
        "schema": "fomhd-wr-gallery-manifest-v1",
        "title": "Physical-run gallery test",
        "introduction": "Actual periodic trial and one RZ macro.",
        "conclusions": "Partial horizon only.",
        "raw_root": str(wr_evidence_root),
        "sources": {
            "cfl": str(wr_evidence_root / "analysis/periodic-frequency-audit.json"),
            "microstep_reference": str(wr_evidence_root / "analysis/microstep-reference-response.json"),
            "linear_selection": str(wr_evidence_root / "analysis/linear-selection.json"),
            "backend_comparison": str(wr_evidence_root / "analysis/short-periodic-backend-comparison.json"),
            "parity_v3": str(wr_evidence_root / "analysis/parity-gpu-v3.json"),
            "periodic_trial": str(wr_evidence_root / "analysis/cpu-trial33-M32-2T.json"),
        },
        "periodic_runs": [
            {
                "id": "periodic-probe-running",
                "label": "Periodic probe (running)",
                "run_json": str(wr_evidence_root / "periodic/cpu-production-probe-M32-s3/run.json"),
                "history_jsonl": str(wr_evidence_root / "periodic/cpu-production-probe-M32-s3/history.jsonl"),
                "snapshots": str(wr_evidence_root / "periodic/cpu-production-probe-M32-s3/snapshots.npz"),
            }
        ],
            "rz_runs": [_real_rz_entry(wr_evidence_root)],
        "availability": [
            {"id": "periodic_math_trial", "status": "available", "detail": "mathematical trial"},
            {"id": "rz_first_macro", "status": "partial", "detail": "1 of 44 requested macros"},
            {"id": "middle_rz", "status": "blocked", "detail": "not supplied"},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    output = tmp_path / "physical-gallery"

    result = builder.build_gallery(manifest_path, output, SOURCE_ROOT)

    assert result["figures"] == 13
    data = json.loads((output / "data.json").read_text())
    evidence = data["rz_run_evidence"][0]
    assert evidence["report_status"] == "segment_complete"
    assert evidence["status"] == "segment"
    figures = {figure["id"]: figure for figure in data["figures"]}
    trial_caption = figures["periodic-2t-trial"]["caption"]
    trial = json.loads(
        (wr_evidence_root / "analysis/cpu-trial33-M32-2T.json").read_text()
    )
    comparison_names = sorted(trial["comparisons"])
    assert "Sorted comparison index mapping" in trial_caption
    for index, name in enumerate(comparison_names):
        assert f"{index} -> {name}" in trial_caption
    work = figures["rz-work-checks"]
    work_text = f"{work['controls']} {work['caption']}".casefold()
    assert "first recorded macro" in work_text
    assert "not evidence of whole-prefix convergence" in work_text
    rendered_manifest = json.loads((output / "manifest.json").read_text())
    ids = {page["id"] for page in rendered_manifest["pages"]}
    expected = {
        "periodic-2t-trial",
        "rz-magnetic-toroidal-initial-final",
        "rz-magnetic-poloidal-final",
        "rz-velocity-toroidal-initial-final",
        "rz-velocity-poloidal-final",
        "rz-density-pressure-final",
        "rz-work-checks",
    }
    assert expected <= ids
    for name in expected:
        assert (output / f"{name}.png").stat().st_size > 0
        assert (output / f"{name}.svg").stat().st_size > 0
    html = (output / "index.html").read_text()
    assert "mathematical trial" in html.casefold()
    assert "1 / 44" in html or "1 of 44" in html
    assert "no production pass" in html.casefold()
    assert "partial-horizon probe" not in html.casefold()
    provenance = json.loads((output / "provenance.json").read_text())
    paths = {item["path"] for item in provenance["inputs"]}
    assert str((wr_evidence_root / "analysis/cpu-trial33-M32-2T.json").resolve()) in paths
    assert str((wr_evidence_root / "rz/cpu-production-probe12-M4-s3/report.json").resolve()) in paths
