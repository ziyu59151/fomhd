"""Focused tests for the bounded saved-run control analysis helper."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools.analyze_clt_controls import AnalysisError, analyze_run, main


FORCE_KEYS = {
    "pressure": "frame_ion_pressure_force_samples",
    "magnetic": "frame_ion_magnetic_force_samples",
    "electric": "frame_ion_electric_force_samples",
    "convection": "frame_ion_convection_samples",
    "wall_reaction": "frame_ion_wall_reaction_samples",
    "net": "frame_ion_net_force_samples",
}


def _snapshot(
    path,
    time_s,
    *,
    b_values,
    density_values,
    step,
    nphi=1,
    K=0,
):
    metadata = {
        "layout_kind": "polygon_dual_v1",
        "step": step,
        "time_s": time_s,
        "mode_axis": list(range(-K, K + 1)),
        "sample_axis": "uniform toroidal physical samples",
        "units": "SI",
    }
    arrays = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "n_i_modes": np.asarray(
            [
                [1000.0 * (index + 1) + 2000.0j, -7.0j * (index + 1)]
                for index in range(2 * K + 1)
            ]
        ),
        "B_N_samples": np.asarray(b_values, dtype=np.complex128),
        "ion_velocity_samples": np.zeros(
            (nphi, 3, 2), dtype=np.complex128
        ),
        "ion_density_samples": np.asarray(
            density_values, dtype=np.complex128
        ),
        "ion_perp_pressure_samples": np.full(
            (nphi, 2), 3.0 + 0.0j, dtype=np.complex128
        ),
        "electron_perp_pressure_samples": np.full(
            (nphi, 2), 2.0 + 0.0j, dtype=np.complex128
        ),
        "electron_parallel_pressure_samples": np.full(
            (nphi, 2), 4.0 + 0.0j, dtype=np.complex128
        ),
    }
    np.savez(path, **arrays)


def _write_run(
    root: Path,
    *,
    name="control",
    distances=(0.01, 0.2),
    include_initial=True,
    force_cache=True,
    nphi=1,
    K=0,
    amplitude=0.0,
    vA_m_s=None,
    scalar_pressure_perturbation=None,
    equilibrium_parameters=None,
    major_radius=3.0,
    minor_radius=0.5,
):
    run = root / name
    snapshots_dir = run / "snapshots"
    snapshots_dir.mkdir(parents=True)
    geometry = {
        "M0": np.asarray([1.0, 3.0], dtype=np.float64),
        "cell_R": np.asarray([2.0, 3.0], dtype=np.float64),
        "cell_Z": np.asarray([0.1, -0.2], dtype=np.float64),
        "physical_wall_distance": np.asarray(distances, dtype=np.float64),
        "geometry_fingerprint": np.asarray("geometry-fixture"),
        "layout_kind": np.asarray("polygon_dual_v1"),
    }
    np.savez(run / "geometry.npz", **geometry)
    snapshot_names = []
    if include_initial:
        initial_name = "step-0000000000.npz"
        _snapshot(
            snapshots_dir / initial_name,
            0.0,
            b_values=np.zeros((nphi, 3, 2)),
            density_values=np.broadcast_to(
                np.asarray([[2.0, 4.0]]), (nphi, 2)
            ),
            step=0,
            nphi=nphi,
            K=K,
        )
        snapshot_names.append(initial_name)
    final_name = "step-0000000001.npz"
    _snapshot(
        snapshots_dir / final_name,
        1.0e-12,
        b_values=np.broadcast_to(
            np.asarray([[[1.0, 0.0], [2.0, 0.0], [0.0, 4.0]]]),
            (nphi, 3, 2),
        ),
        density_values=np.broadcast_to(
            np.asarray([[2.0, 4.0]]), (nphi, 2)
        ),
        step=1,
        nphi=nphi,
        K=K,
    )
    snapshot_names.append(final_name)

    force_values = {
        "pressure": 1.0,
        "magnetic": 2.0,
        "electric": 3.0,
        "convection": 4.0,
        "wall_reaction": 5.0,
        "net": 0.0,
    }
    rhs = {}
    if force_cache:
        for name, scale in force_values.items():
            value = np.zeros((nphi, 3, 2), dtype=np.float64)
            value[:, 0, 0] = scale
            rhs[FORCE_KEYS[name]] = value
        scalar = np.broadcast_to(
            np.asarray([[-1.0, 2.0]], dtype=np.complex128),
            (nphi, 2),
        )
        vector = np.broadcast_to(
            np.asarray([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
            (nphi, 3, 2),
        )
        tensor = np.arange(
            nphi * 3 * 3 * 2, dtype=np.float64
        ).reshape(nphi, 3, 3, 2)
        for name in (
            "frame_ion_theta",
            "frame_ion_strain",
            "frame_electron_theta",
            "frame_electron_strain",
            "frame_ion_pdot_perp",
            "frame_ion_pdot_parallel",
            "frame_electron_pdot_perp",
            "frame_electron_pdot_parallel",
        ):
            rhs[name] = scalar.reshape(-1)
        rhs["frame_ion_convection"] = vector
        rhs["frame_ion_gradient"] = tensor.reshape(-1)
        np.savez(run / "initial-rhs.npz", **rhs)

    report = {
        "status": "completed",
        "scientific_status": "pending_parent_validation",
        "parameters": {
            "K": K,
            "Nphi": nphi,
            "micro_h": 1.0e-12,
            "macro_h": 1.0e-12,
            "micro_count": 1,
            "ion_nodes": 3,
            "nodes": 8,
            "amplitude": amplitude,
            "seed": "smooth",
            "major_radius": major_radius,
            "minor_radius": minor_radius,
        },
        "horizon": {
            "micro_h_s": 1.0e-12,
            "macro_h_s": 1.0e-12,
            "M": 1,
            "segment_steps": 1,
        },
        "geometry": {
            "kind": "polygon_dual_v1",
            "fingerprint": "geometry-fixture",
            "operator_fingerprint": "operator-fixture",
            "Nc": 2,
        },
        "acceptance": {
            "accepted_step": 1,
            "accepted_time_s": 1.0e-12,
            "accepted_intervals": 1,
        },
        "provenance": {
            "source_fingerprint": "source-fixture",
            "source_manifest": {},
        },
        "init": {
            "baseline": {
                "baseline": "gs",
                "gs_degree": 10,
                **(
                    {"equilibrium_parameters": equilibrium_parameters}
                    if equilibrium_parameters is not None
                    else {}
                ),
            },
            "perturbation": {
                "seed": "smooth",
                "amplitude": amplitude,
                "vA_m_s": vA_m_s,
                "scalar_pressure_perturbation": (
                    scalar_pressure_perturbation
                ),
            },
        },
        "snapshots": snapshot_names,
    }
    (run / "report.json").write_text(
        json.dumps(report, sort_keys=True), encoding="utf-8"
    )
    return run


def _region(report, name, group):
    return report["drift"]["regions"][name]["groups"][group]


def test_vector_weighted_rms_uses_cell_weights_and_components_once(tmp_path):
    run = _write_run(tmp_path)

    result = analyze_run(run, wall_width=0.05)

    assert result["analysis_status"] == "available"
    whole_b = _region(result, "whole", "B")
    assert whole_b["weighted_rms"] == pytest.approx(
        math.sqrt(53.0 / 4.0), rel=0.0, abs=1e-15
    )
    assert whole_b["linf"] == pytest.approx(4.0)
    assert whole_b["components"]["R"]["weighted_rms"] == pytest.approx(0.5)
    assert whole_b["components"]["phi"]["weighted_rms"] == pytest.approx(
        1.0
    )
    assert whole_b["components"]["Z"]["weighted_rms"] == pytest.approx(
        math.sqrt(12.0)
    )
    assert _region(result, "wall_band", "B")["weighted_rms"] == pytest.approx(
        math.sqrt(5.0)
    )
    assert _region(result, "interior", "B")["weighted_rms"] == pytest.approx(
        4.0
    )


def test_torus_volume_and_physical_l2_are_independent_of_nphi(tmp_path):
    run = _write_run(tmp_path, nphi=5, K=1)

    result = analyze_run(run, wall_width=0.05)

    assert result["regions"]["whole"]["volume_m3"] == pytest.approx(
        8.0 * math.pi
    )
    whole_b = _region(result, "whole", "B")
    assert whole_b["volume_m3"] == pytest.approx(8.0 * math.pi)
    assert whole_b["weighted_l2"] == pytest.approx(
        math.sqrt((2.0 * math.pi / 5.0) * 5.0 * 53.0)
    )
    assert whole_b["weighted_rms"] == pytest.approx(
        math.sqrt(53.0 / 4.0)
    )
    assert whole_b["linf_definition"] == (
        "maximum absolute Cartesian component; not vector norm"
    )


def test_true_zero_drift_is_available_and_mode_arrays_are_ignored(tmp_path):
    run = _write_run(tmp_path)

    result = analyze_run(run, wall_width=0.05)

    density = _region(result, "whole", "ni")
    assert density["available"] is True
    assert density["weighted_rms"] == 0.0
    assert density["linf"] == 0.0
    assert result["drift"]["snapshots"][-1]["groups"]["B"] > 0.0
    assert result["drift"]["snapshots"][-1]["maximum_abs"] is None
    assert "mixed units" in result["drift"][
        "snapshots"
    ][-1]["maximum_abs_reason"]
    assert result["drift"]["maximum_abs"] is None
    assert "mixed units" in result["drift"]["maximum_abs_reason"]
    assert result["drift"]["final"]["maximum_abs"] is None
    assert "mixed units" in result["drift"]["final"]["maximum_abs_reason"]


def test_missing_true_initial_is_unavailable_not_zero_filled(tmp_path):
    run = _write_run(tmp_path, include_initial=False)

    result = analyze_run(run, wall_width=0.05)

    assert result["analysis_status"] == "unavailable"
    assert result["drift"]["available"] is False
    assert result["drift"]["reason"]
    assert result["drift"]["final"] is None


def test_empty_wall_band_has_null_metrics(tmp_path):
    run = _write_run(tmp_path, distances=(0.2, 0.3))

    result = analyze_run(run, wall_width=0.05)

    band = _region(result, "wall_band", "B")
    assert band["available"] is False
    assert band["count"] == 0
    assert band["weighted_rms"] is None
    assert band["linf"] is None


def test_rhs_force_closure_and_frame_diagnostics_are_explicit(tmp_path):
    run = _write_run(tmp_path)

    result = analyze_run(run, wall_width=0.05)

    forces = result["initial_rhs"]["forces"]
    assert forces["available"] is True
    assert forces["missing_keys"] == []
    assert forces["components"]["pressure"]["nonzero_count"] == 1
    defect = forces["sum_defect"]
    assert defect["available"] is True
    assert defect["regions"]["whole"]["linf"] == pytest.approx(7.0)
    diagnostics = result["initial_rhs"]["frame_diagnostics"]
    assert diagnostics["frame_ion_theta"]["available"] is True
    assert diagnostics["frame_electron_pdot_parallel"]["available"] is True


def test_flat_named_frame_diagnostics_reshape_known_contracts(tmp_path):
    run = _write_run(tmp_path, nphi=5, K=1)

    result = analyze_run(run, wall_width=0.05)

    diagnostics = result["initial_rhs"]["frame_diagnostics"]
    theta = diagnostics["frame_ion_theta"]
    gradient = diagnostics["frame_ion_gradient"]
    convection = diagnostics["frame_ion_convection"]
    assert theta["available"] is True
    assert theta["shape"] == [5, 2]
    assert gradient["available"] is True
    assert gradient["shape"] == [5, 3, 3, 2]
    assert len(gradient["regions"]["whole"]["components"]) == 9
    assert convection["available"] is True
    assert convection["shape"] == [5, 3, 2]
    assert len(convection["regions"]["whole"]["components"]) == 3


def test_optional_continuous_and_missing_force_cache_are_explicit(tmp_path):
    run = _write_run(tmp_path, force_cache=False)

    result = analyze_run(run, wall_width=0.05, continuous=False)

    assert result["continuous"]["available"] is False
    assert "not requested" in result["continuous"]["reason"]
    assert result["initial_rhs"]["forces"]["available"] is False
    assert set(result["initial_rhs"]["forces"]["missing_keys"]) == set(
        FORCE_KEYS.values()
    )


def test_continuous_comparison_uses_physical_vector_axis(
    tmp_path, monkeypatch
):
    declared = {
        "R0_m": 3.0,
        "a_m": 0.5,
        "F0_T_m": 6.0,
        "n0_m^-3": 2.0e19,
        "p_edge_Pa": 1000.0,
        "pprime_Pa_per_Wb_rad": 1.0e4,
        "alpha_ion_fraction": 0.6,
    }
    run = _write_run(tmp_path, equilibrium_parameters=declared)
    constants = tmp_path / "inputs" / "circular_gs" / "gs_equilibrium.json"
    constants.parent.mkdir(parents=True)
    constants.write_text(
        json.dumps(
            {
                "parameters": {
                    "R0_m": 30.0,
                    "a_m": 0.5,
                    "F0_T_m": 6.0,
                    "n0_m^-3": 2.0e19,
                    "p_edge_Pa": 1000.0,
                    "pprime_Pa_per_Wb_rad": 1.0e4,
                    "alpha_ion_fraction": 0.6,
                    "degree": 12,
                }
            }
        ),
        encoding="utf-8",
    )
    called = {}

    def build(**values):
        called["build"] = values
        return SimpleNamespace()

    def sample(model, radius, axial):
        called["shape"] = (radius.shape, axial.shape)
        zeros = np.zeros(radius.shape)
        return {
            "B_R": zeros,
            "B_phi": zeros,
            "B_Z": zeros,
        }

    monkeypatch.setitem(
        sys.modules,
        "tools.circular_equilibrium",
        SimpleNamespace(
            build_gs_equilibrium=build,
            sample_gs=sample,
            sample_vacuum=lambda *args, **kwargs: None,
        ),
    )

    result = analyze_run(run, wall_width=0.05, continuous=True)

    assert called["build"]["degree"] == 10
    assert called["build"]["R0"] == 3.0
    assert called["shape"] == ((2,), (2,))
    assert result["continuous"]["available"] is True
    assert result["continuous"]["regions"]["whole"]["linf"] == 0.0
    assert result["continuous"]["parameter_source"] == (
        "run.init.baseline.equilibrium_parameters"
    )
    assert result["continuous"]["preprocess_degree"] == 10
    assert result["continuous"]["external_degree"] == 12
    assert result["continuous"]["degree_difference"] == -2


def test_legacy_continuous_constants_must_match_actual_run_geometry(
    tmp_path, monkeypatch
):
    run = _write_run(tmp_path, major_radius=30.0, minor_radius=0.5)
    constants = tmp_path / "inputs" / "circular_gs" / "gs_equilibrium.json"
    constants.parent.mkdir(parents=True)
    constants.write_text(
        json.dumps(
            {
                "parameters": {
                    "R0_m": 3.0,
                    "a_m": 0.5,
                    "F0_T_m": 6.0,
                    "n0_m^-3": 2.0e19,
                    "p_edge_Pa": 1000.0,
                    "pprime_Pa_per_Wb_rad": 1.0e4,
                    "alpha_ion_fraction": 0.6,
                    "degree": 12,
                }
            }
        ),
        encoding="utf-8",
    )

    def forbidden(**values):
        raise AssertionError("mismatched legacy constants must not build")

    monkeypatch.setitem(
        sys.modules,
        "tools.circular_equilibrium",
        SimpleNamespace(
            build_gs_equilibrium=forbidden,
            sample_gs=lambda *args: None,
            sample_vacuum=lambda *args, **kwargs: None,
        ),
    )

    result = analyze_run(run, wall_width=0.05, continuous=True)

    assert result["continuous"]["available"] is False
    assert "mismatch" in result["continuous"]["reason"]


def test_velocity_seed_scale_takes_precedence_over_zero_pressure(tmp_path):
    run = _write_run(
        tmp_path,
        amplitude=0.25,
        vA_m_s=2.0e5,
        scalar_pressure_perturbation=0.0,
    )

    result = analyze_run(run, wall_width=0.05)

    target = result["target_seed_scale"]
    assert target["kind"] == "A_times_vA"
    assert target["value"] == pytest.approx(5.0e4)
    assert target["unit"] == "m/s"
    assert target["scalar_pressure_perturbation"] == 0.0


def test_cli_writes_new_strict_json_and_labels_prepatch_format_check(tmp_path):
    run = _write_run(tmp_path, name="cpu-prepatch-gs-N8")
    output = tmp_path / "analysis.json"

    assert main(
        [
            "--run",
            str(run),
            "--wall-width",
            "0.05",
            "--output",
            str(output),
        ]
    ) == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    analyzed = data["runs"][0]

    assert data["schema"] == "clt_control_analysis_v1"
    assert analyzed["scope"]["format_check_only"] is True
    assert analyzed["scope"]["scientific_acceptance"] is False
    assert analyzed["target_seed_scale"]["value"] == 0.0
    with pytest.raises(AnalysisError, match="already exists"):
        main(
            [
                "--run",
                str(run),
                "--wall-width",
                "0.05",
                "--output",
                str(output),
            ]
        )
