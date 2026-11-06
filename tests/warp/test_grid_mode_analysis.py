"""Behavioral checks for grid candidate export analysis."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.grid_mode_analysis import (
    _reference_for_spectrum,
    _rank,
    analyze,
    continuation_score,
    energy_overlap,
    interpolate_previous,
    load_spectrum,
    load_selected,
    save_selected,
    select_candidate,
    validate_candidate_export,
    validate_comparable,
)
from tools.cold_hall_reference import MP


def _identity():
    return {
        "species": [
            {"id": "ion", "mass": 2.0, "charge": 1.0, "gamma": 5/3,
             "n_min": 0.0, "face_min": 0.0, "p_min": 0.0, "parallel_min": 0.0},
            {"id": "electron", "mass": 1/1836, "charge": -1.0, "gamma": 0.0,
             "n_min": 0.0, "face_min": 0.0, "p_min": 0.0, "parallel_min": 0.0},
        ],
        "references": {
            "length": 1.0, "time": 1.0, "ni": 1.0, "ne": 1.0,
            "velocity": 1.0, "electric": 1.0, "magnetic": 1.0,
            "current": 1.0, "charge": 1.0, "pressure": 1.0,
        },
        "mu0": 2.0, "epsilon0": 1.0, "B_min": 1e-10,
        "rank_rtol": 1e-12, "electron_inertia": 0, "Nphi": 5,
        "geometry_sha256": "grid-dependent",
    }


def _report(kind="circle", mode=0):
    return {
        "kind": kind,
        "nodes": 4,
        "K": max(mode, 0),
        "mode": mode,
        "physical_identity": _identity(),
        "candidates": [
            {"index": 5, "frequency_hz": 1e6,
             "magnetic_transfer_ratio": 0.5,
             "MB_magnetic_norm": 4.0, "M0_ion_kinetic_norm": 2.0},
            {"index": 9, "frequency_hz": 2e6,
             "magnetic_transfer_ratio": 0.75,
             "MB_magnetic_norm": 8.0, "M0_ion_kinetic_norm": 4.0},
        ],
    }


def _arrays():
    velocity = np.zeros((2, 3, 4), dtype=complex)
    magnetic = np.zeros((2, 3, 4), dtype=complex)
    density = np.zeros((2, 4), dtype=complex)
    velocity[0, 0] = [1, 2, 3, 4]
    velocity[0, 2] = [4, 3, 2, 1]
    magnetic[0, 1] = 0.5
    density[0] = [1, 0, -1, 0]
    velocity[1] = velocity[0] * 2
    magnetic[1] = magnetic[0] * 2
    density[1] = density[0] * 2
    return {
        "candidate_indices": np.array([5, 9], dtype=np.int64),
        "candidate_frequency_hz": np.array([1e6, 2e6]),
        "candidate_magnetic_transfer_ratio": np.array([0.5, 0.75]),
        "candidate_MB_magnetic_norm": np.array([4.0, 8.0]),
        "candidate_M0_ion_kinetic_norm": np.array([2.0, 4.0]),
        "candidate_velocity": velocity,
        "candidate_B_node": magnetic,
        "candidate_density": density,
        "R": np.array([0.0, 1.0, 0.0, 1.0]),
        "Z": np.array([0.0, 0.0, 1.0, 1.0]),
        "M0": np.array([1.0, 2.0, 3.0, 4.0]),
        "background_B_node": np.tile(np.array([[0.0], [1.0], [0.0]]), (1, 4)),
        "rho": np.array(2.0),
        "mu0": np.array(2.0),
        "wall_R": np.array([0.0, 1.0, 1.0, 0.0]),
        "wall_Z": np.array([0.0, 0.0, 1.0, 1.0]),
    }


def _current_spectrum(velocity, magnetic, density):
    return {
        "arrays": {
            "candidate_velocity": velocity[None],
            "candidate_B_node": magnetic[None],
            "candidate_density": density[None],
            "R": np.array([0.2, 2.0]),
            "Z": np.array([0.2, 2.0]),
            "M0": np.array([1.0, 3.0]),
            "rho": np.array(2.0),
            "mu0": np.array(2.0),
        }
    }


def test_analytic_nodal_overlap_is_phase_and_scale_invariant():
    candidate_velocity = np.array([[1+2j, 2-1j], [0.5j, -3], [1, 4j]])
    candidate_magnetic = np.array([[0.2-0.1j, 1], [-0.5j, 2], [1+1j, -1]])
    reference_velocity = candidate_velocity * (0.3-1.7j)
    reference_magnetic = candidate_magnetic * (0.3-1.7j)
    candidate = {"velocity": candidate_velocity, "B_node": candidate_magnetic}
    reference = {"velocity": reference_velocity, "B_node": reference_magnetic}
    result = energy_overlap(
        candidate, reference, np.array([1.0, 3.0]), 2.0, 4.0
    )
    assert result["normalized_overlap"] == pytest.approx(1.0, abs=1e-14)


def test_partial_and_disjoint_interpolation_use_finite_common_support_only():
    previous = {
        "R": np.array([0.0, 1.0, 0.0, 1.0]),
        "Z": np.array([0.0, 0.0, 1.0, 1.0]),
        "velocity": np.ones((3, 4), dtype=complex),
        "B_node": np.ones((3, 4), dtype=complex) * (1+1j),
        "density": np.ones(4, dtype=complex) * (2-1j),
    }
    current = _current_spectrum(
        np.ones((3, 2), dtype=complex),
        np.ones((3, 2), dtype=complex) * (1+1j),
        np.ones(2, dtype=complex) * (2-1j),
    )
    partial = interpolate_previous(
        previous, current["arrays"]["R"], current["arrays"]["Z"],
        current["arrays"]["M0"],
    )
    np.testing.assert_array_equal(partial["mask"], [True, False])
    assert partial["covered_M0_fraction"] == pytest.approx(0.25)
    score = continuation_score(
        previous, current, current["arrays"]["M0"], 2.0, 2.0, 0
    )
    assert score["normalized_overlap"] == pytest.approx(1.0)
    assert score["common_cell_count"] == 1
    current["arrays"]["R"] = np.array([10.0, 11.0])
    current["arrays"]["Z"] = np.array([10.0, 11.0])
    disjoint = interpolate_previous(
        previous, current["arrays"]["R"], current["arrays"]["Z"],
        current["arrays"]["M0"],
    )
    assert not np.any(disjoint["mask"])
    assert disjoint["covered_M0_fraction"] == 0.0
    assert disjoint["common_cell_count"] == 0


def test_interpolation_does_not_fill_an_inactive_source_cell():
    coordinates = np.array([
        (r, z) for z in (0.0, 1.0, 2.0) for r in (0.0, 1.0, 2.0)
        if not (r == 1.0 and z == 1.0)
    ])
    previous = {
        "R": coordinates[:, 0],
        "Z": coordinates[:, 1],
        "velocity": np.tile(
            (coordinates[:, 0] + 2j * coordinates[:, 1])[None, :], (3, 1)
        ),
        "B_node": np.tile(
            (1 + coordinates[:, 0] + 2j * coordinates[:, 1])[None, :], (3, 1)
        ),
        "density": coordinates[:, 0] + 2j * coordinates[:, 1],
    }
    interpolated = interpolate_previous(
        previous,
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 3.0]),
    )
    np.testing.assert_array_equal(
        interpolated["linear_finite_mask"], [True, True]
    )
    np.testing.assert_array_equal(
        interpolated["active_source_footprint"], [True, False]
    )
    np.testing.assert_array_equal(interpolated["mask"], [True, False])
    assert interpolated["covered_M0_fraction"] == pytest.approx(0.25)


def test_rank_ambiguity_is_metric_specific():
    candidates = [
        {
            "eigen_index": 1,
            "frequency_hz": 1.0,
            "normalized_overlap": 0.900,
            "continuation_overlap": 0.900,
        },
        {
            "eigen_index": 2,
            "frequency_hz": 2.0,
            "normalized_overlap": 0.899,
            "continuation_overlap": 0.100,
        },
    ]
    _rank(candidates, "normalized_overlap")
    _rank(candidates, "continuation_overlap")
    assert candidates[0]["normalized_overlap_ambiguous"] is True
    assert candidates[0]["continuation_overlap_ambiguous"] is False
    assert candidates[0]["normalized_overlap_top_score_gap"] == pytest.approx(0.001)
    assert candidates[0]["continuation_overlap_top_score_gap"] == pytest.approx(0.8)


@pytest.mark.parametrize("change,message", [
    ("family", "family"),
    ("mode", "mode"),
    ("reference", "reference"),
])
def test_comparison_rejects_changed_identity(change, message):
    left = _report()
    right = _report()
    if change == "family":
        right["kind"] = "east"
    elif change == "mode":
        right["mode"] = 1
    else:
        right["physical_identity"] = json.loads(json.dumps(right["physical_identity"]))
        right["physical_identity"]["references"]["magnetic"] = 2.0
    with pytest.raises(ValueError, match=message):
        validate_comparable([left, right])


def test_candidate_export_dimensions_are_explicit():
    arrays = _arrays()
    validate_candidate_export(arrays, _report())
    arrays["candidate_B_node"] = arrays["candidate_B_node"][:, :2]
    with pytest.raises(ValueError, match=r"candidate_B_node.*shape"):
        validate_candidate_export(arrays, _report())
    arrays = _arrays()
    arrays["candidate_velocity"] = arrays["candidate_velocity"][:, :2, :]
    with pytest.raises(ValueError, match=r"candidate_velocity.*shape"):
        validate_candidate_export(arrays, _report())


def test_frequency_index_selection_and_selected_export_preserve_inputs(tmp_path):
    source = tmp_path / "grid"
    source.mkdir()
    np.savez(source / "spectrum.npz", **_arrays())
    report = _report()
    (source / "report.json").write_text(json.dumps(report) + "\n")
    spectrum_before = hashlib.sha256((source / "spectrum.npz").read_bytes()).hexdigest()
    report_before = hashlib.sha256((source / "report.json").read_bytes()).hexdigest()
    spectrum = load_spectrum(source)
    ordinal, choice = select_candidate(spectrum, frequency_hz=1.0e6, tolerance_hz=1.0)
    assert ordinal == 0 and choice["eigen_index"] == 5
    ordinal, choice = select_candidate(spectrum, index=9)
    assert ordinal == 1 and choice["frequency_hz"] == 2.0e6
    saved = save_selected(tmp_path / "selected.npz", spectrum, 0)
    selected = load_selected(Path(saved["path"]))
    assert selected["velocity"].shape == (3, 4)
    assert selected["B_node"].shape == (3, 4)
    assert selected["density"].shape == (4,)
    assert float(selected["magnetic_transfer_ratio"]) == 0.5
    assert float(selected["MB_magnetic_norm"]) == 4.0
    assert float(selected["M0_ion_kinetic_norm"]) == 2.0
    with pytest.raises(FileExistsError):
        save_selected(tmp_path / "selected.npz", spectrum, 0)
    assert hashlib.sha256((source / "spectrum.npz").read_bytes()).hexdigest() == spectrum_before
    assert hashlib.sha256((source / "report.json").read_bytes()).hexdigest() == report_before


def test_analyze_ordinary_scoring_needs_no_previous_branch(tmp_path):
    source = tmp_path / "grid"
    source.mkdir()
    np.savez(source / "spectrum.npz", **_arrays())
    (source / "report.json").write_text(json.dumps(_report(kind="east")) + "\n")
    result = analyze([source], reference_order=2)
    assert result["selection"] is None
    assert len(result["grids"]) == 1
    assert result["grids"][0]["candidate_count"] == 2
    assert "continuation_overlap" not in result["grids"][0]["candidates"][0]


def test_cold_reference_solution_cache_is_evaluated_at_each_grid_cells():
    report = _report()
    report["physical_identity"]["references"].update({
        "length": 1.0,
        "magnetic": 3.4,
        "ni": 3.423e18,
    })
    report["physical_identity"]["species"][0]["mass"] = 2 * MP
    report["physical_identity"]["mu0"] = 1.25663706127e-6
    first = {
        "report": report,
        "arrays": {
            "R": np.array([2.7, 3.3, 2.7, 3.3]),
            "Z": np.array([-0.3, -0.3, 0.3, 0.3]),
        },
    }
    second = {
        "report": report,
        "arrays": {
            "R": np.array([2.7, 3.0, 3.3, 2.7, 3.0, 3.3, 2.7, 3.0, 3.3]),
            "Z": np.array([-0.3, -0.3, -0.3, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3]),
        },
    }
    cache = {}
    fields4, meta4 = _reference_for_spectrum(first, 2, cache)
    fields9, meta9 = _reference_for_spectrum(second, 2, cache)
    assert fields4["velocity"].shape == (3, 4)
    assert fields9["velocity"].shape == (3, 9)
    assert fields4["B_node"].shape == (3, 4)
    assert fields9["B_node"].shape == (3, 9)
    assert meta4 == meta9
