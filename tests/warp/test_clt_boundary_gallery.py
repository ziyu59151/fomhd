"""Behavioral tests for the native-polygon CLT boundary gallery."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = SOURCE_ROOT / "tools/plots/build_clt_boundary_report.py"


def load_builder():
    assert MODULE_PATH.is_file(), f"gallery builder is absent: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location("clt_gallery_builder", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def builder():
    return load_builder()


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _fixture_geometry(module):
    polygons = (
        np.array([[2.0, 0.0], [3.0, 0.0], [2.8, 0.45], [2.0, 0.5]]),
        np.array([[3.0, 0.0], [4.0, 0.0], [3.8, 0.5], [3.0, 0.4]]),
        np.array([[2.0, 0.5], [2.8, 0.45], [3.0, 0.9], [2.5, 1.1]]),
        np.array([[3.0, 0.4], [3.8, 0.5], [3.5, 1.0], [3.0, 0.9]]),
        np.array([[2.5, 1.1], [3.0, 0.9], [3.5, 1.0], [3.0, 1.4]]),
        np.array([[3.5, 1.0], [3.8, 0.5], [4.0, 0.0], [3.8, 1.3]]),
    )
    centers = np.array([np.mean(polygon, axis=0) for polygon in polygons])
    contour_angle = np.linspace(0.0, 2.0 * np.pi, 41)
    contour = np.column_stack(
        (3.0 + np.cos(contour_angle), np.sin(contour_angle))
    )
    face_pairs = []
    face_normals = []
    for polygon in polygons:
        closed = np.vstack((polygon, polygon[0]))
        for start, stop in zip(closed[:-1], closed[1:]):
            tangent = stop - start
            normal = np.array([-tangent[1], tangent[0]])
            face_pairs.append(np.vstack((start, stop)))
            face_normals.append(normal / np.linalg.norm(normal))
    face_coords = np.mean(np.stack(face_pairs), axis=1)
    face_normals = np.stack(face_normals)
    return SimpleNamespace(
        nc=len(polygons),
        polygons=polygons,
        contour=contour,
        chord_wall=contour.copy(),
        centers=centers,
        volumes=np.array([0.21, 0.22, 0.20, 0.18, 0.16, 0.19]),
        wall_distances=np.array([0.16, 0.20, 0.25, 0.18, 0.12, 0.14]),
        face_coords=face_coords,
        face_normals=face_normals,
        fingerprint="synthetic-native-polygon-fixture",
    )


def _write_geometry(path: Path, geometry):
    flat = np.concatenate(geometry.polygons, axis=0)
    offsets = np.zeros(len(geometry.polygons) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(polygon) for polygon in geometry.polygons])
    np.savez_compressed(
        path,
        cell_polygon_points=flat,
        cell_polygon_offsets=offsets,
        cell_polygon_counts=np.diff(offsets),
        cell_R=geometry.centers[:, 0],
        cell_Z=geometry.centers[:, 1],
        M0=geometry.volumes,
        physical_wall_distance=geometry.wall_distances,
        input_circle=geometry.contour,
        numerical_polygon=geometry.chord_wall,
        face_coords_R=geometry.face_coords[:, 0],
        face_coords_Z=geometry.face_coords[:, 1],
        face_normals=geometry.face_normals,
        geometry_fingerprint=np.asarray(geometry.fingerprint),
    )


def _sample_arrays(nphi: int, nc: int, run: str, final: bool):
    phi = np.arange(nphi, dtype=np.float64)
    cell = np.arange(nc, dtype=np.float64)
    control_bias = 0.0 if run == "control" else 0.03
    final_bias = 0.0 if run == "control" else 0.17
    wave = np.cos(2.0 * np.pi * phi[:, None] / nphi + cell[None, :] / 7.0)
    radial = np.sin(np.pi * cell[None, :] / max(nc, 1))
    vectors = np.empty((nphi, 3, nc), dtype=np.float64)
    offsets = np.array([0.10, 2.00, -0.20])
    for component in range(3):
        vectors[:, component, :] = (
            offsets[component]
            + control_bias
            + final_bias
            + (0.05 + 0.02 * component) * wave
            + (0.01 * (component + 1)) * radial
            + (0.04 if final else 0.0) * (component + 1) * cell[None, :] / nc
        )
    velocity = np.empty_like(vectors)
    for component in range(3):
        velocity[:, component, :] = (
            (component + 1) * 10.0
            + control_bias * 1.0e3
            + final_bias * 1.0e3
            + (3.0 + component) * wave
            + (0.2 + 0.1 * component) * float(final)
            + 0.03 * float(final) * cell[None, :] / nc
        )
    density = (
        2.0e19
        + control_bias * 2.0e18
        + final_bias * 2.0e18
        + 4.0e16 * float(final)
        + (2.0e17 * (1.0 + cell[None, :] / nc)) * (1.0 + 0.2 * wave)
    )
    electron_density = 1.01 * density
    pressure_i = (
        600.0
        + control_bias * 60.0
        + final_bias * 60.0
        + 0.8 * float(final)
        + (3.0 + cell[None, :]) * (1.0 + 0.1 * wave)
    )
    pressure_perp = (
        400.0
        + control_bias * 40.0
        + final_bias * 40.0
        + 0.6 * float(final)
        + (2.0 + cell[None, :]) * (1.0 + 0.08 * wave)
    )
    pressure_parallel = (
        390.0
        + control_bias * 39.0
        + final_bias * 39.0
        + 0.5 * float(final)
        + (1.0 + cell[None, :]) * (1.0 - 0.06 * wave)
    )
    return {
        "B_N_samples": vectors,
        "ion_velocity_samples": velocity,
        "ion_density_samples": density,
        "ion_perp_pressure_samples": pressure_i,
        "electron_density_samples": electron_density,
        "electron_perp_pressure_samples": pressure_perp,
        "electron_parallel_pressure_samples": pressure_parallel,
    }


def _make_run(
    root: Path,
    module,
    geometry,
    name: str,
    *,
    K: int = 0,
    mode: int | None = None,
    time_mismatch: bool = False,
    missing_t0: bool = False,
    nonfinite: bool = False,
    missing_phi: bool = False,
    force_cache: bool = False,
):
    run_dir = root / name
    snapshot_dir = run_dir / "snapshots"
    snapshot_dir.mkdir(parents=True)
    _write_geometry(run_dir / "geometry.npz", geometry)
    nphi = 4 * K + 1
    nc = len(geometry.polygons)
    run_kind = "control" if "control" in name else "perturbed"
    amplitudes = {"control": 0.0, "perturbed": 1.0e-4}
    parameters = {
        "K": K,
        "mode": K if mode is None else mode,
        "Nphi": nphi,
        "mesh": "polygon",
        "baseline": "vacuum",
        "nodes": 2,
        "major_radius": 3.0,
        "minor_radius": 1.0,
        "amplitude": amplitudes[run_kind],
        "micro_h": 1.0e-8,
        "macro_h": 1.0e-8,
        "M": 1,
        "ion_nodes": 3,
        "coupling": "synchronous",
        "device": "cpu",
        "steps": 10,
        "segment_steps": 10,
        "grid_shift_R": 0.0,
        "grid_shift_Z": 0.0,
        "horizon_s": 1.0e-7,
    }
    phi = 2.0 * np.pi * np.arange(nphi, dtype=np.float64) / nphi
    final_time = 1.0e-6 if not time_mismatch else 1.1e-6
    snapshots = []
    times = [(0, 0.0, False), (10, final_time, True)]
    if missing_t0:
        times = times[1:]
    for index, (step, time_s, final) in enumerate(times):
        arrays = _sample_arrays(nphi, nc, run_kind, final)
        if nonfinite and final:
            arrays["ion_velocity_samples"][0, 1, 0] = np.nan
        metadata = {
            "layout_kind": "polygon_dual_v1",
            "step": step,
            "generation": index + 1,
            "time_s": time_s,
            "units": "SI",
            "B_native_order": ["B_phi_vertices", "B_t_faces"],
        }
        if not missing_phi:
            metadata["sample_phis"] = phi.tolist()
        arrays["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, allow_nan=False)
        )
        filename = f"step-{step:010d}.npz"
        np.savez_compressed(snapshot_dir / filename, **arrays)
        snapshots.append(filename)
    report = {
        "status": "completed",
        "scientific_status": "pending_parent_validation",
        "parameters": parameters,
        "scheme_parameters": {
            key: parameters[key]
            for key in (
                "K",
                "mode",
                "Nphi",
                "amplitude",
                "coupling",
                "micro_h",
                "macro_h",
                "M",
                "ion_nodes",
                "grid_shift_R",
                "grid_shift_Z",
            )
        },
        "geometry": {
            "kind": "polygon_dual_v1",
            "fingerprint": geometry.fingerprint,
            "sha256": hashlib.sha256(
                (run_dir / "geometry.npz").read_bytes()
            ).hexdigest(),
            "Nc": nc,
        },
        "geometry_input": {
            "polygon": geometry.contour.tolist(),
            "generator_R": [2.25, 2.75, 3.25, 3.75],
            "generator_Z": [0.25, 0.75, 1.25, 1.75],
            "grid_shift_R": 0.0,
            "grid_shift_Z": 0.0,
        },
        "snapshots": snapshots,
        "acceptance": {
            "target_step": 10,
            "accepted_step": 10,
            "accepted_time_s": final_time,
            "accepted_intervals": 10,
            "complete_global_horizon": True,
        },
        "provenance": {
            "source_fingerprint": "synthetic-source-fingerprint",
            "device": "cpu",
        },
        "diag": {"elapsed_s": 1.25, "snapshot_count": len(snapshots)},
    }
    _write_json(run_dir / "report.json", report)
    if force_cache:
        force = np.empty((nphi, 3, nc), dtype=np.float64)
        force[...] = np.arange(nc, dtype=np.float64)[None, None, :]
        force[:, 1, :] += 2.0
        force[:, 2, :] -= 0.5
        np.savez_compressed(
            run_dir / "initial-rhs.npz",
            frame_ion_pressure_force_samples=force * 1.0,
            frame_ion_magnetic_force_samples=force * 2.0,
            frame_ion_electric_force_samples=force * 3.0,
            frame_ion_convection_samples=force * 4.0,
            frame_ion_wall_reaction_samples=force * 5.0,
            frame_ion_net_force_samples=force * 6.0,
        )
    return run_dir


def _build_fixture(
    tmp_path: Path,
    module,
    *,
    K: int = 0,
    mode: int | None = None,
    control_options=None,
    perturbed_options=None,
    one_pair: bool = True,
):
    geometry = _fixture_geometry(module)
    control = _make_run(
        tmp_path,
        module,
        geometry,
        "pair-control" if one_pair else "control",
        K=K,
        mode=mode,
        **(control_options or {}),
    )
    perturbed = _make_run(
        tmp_path,
        module,
        geometry,
        "pair-perturbed" if one_pair else "perturbed",
        K=K,
        mode=mode,
        **(perturbed_options or {}),
    )
    output = tmp_path / "gallery"
    module.build(
        [module.PairSpec("representative", control, perturbed)],
        output,
        source_root=SOURCE_ROOT,
    )
    return output, geometry, control, perturbed


def test_outward_oblique_facet_normals_are_not_flipped(builder):
    polygon = np.array([[2.0, 0.0], [3.0, 0.2], [2.4, 0.8]])
    center = np.mean(polygon, axis=0)
    face_coords = []
    face_normals = []
    closed = np.vstack((polygon, polygon[0]))
    for start, stop in zip(closed[:-1], closed[1:]):
        tangent = stop - start
        normal = np.array([-tangent[1], tangent[0]])
        midpoint = 0.5 * (start + stop)
        if np.dot(normal, midpoint - center) < 0.0:
            normal = -normal
        normal /= np.linalg.norm(normal)
        face_coords.append(midpoint)
        face_normals.append(normal)
    face_coords = np.asarray(face_coords)
    face_normals = np.asarray(face_normals)
    expected = np.sum(face_normals, axis=0)
    expected /= np.linalg.norm(expected)
    geometry = builder.Geometry(
        polygons=(polygon.copy(),),
        contour=np.array(
            [[1.9, -0.1], [3.1, 0.1], [2.4, 0.9], [1.9, -0.1]]
        ),
        chord_wall=polygon.copy(),
        centers=center.reshape(1, 2),
        volumes=np.array([0.2]),
        wall_distances=np.array([0.1]),
        face_coords=face_coords,
        face_normals=face_normals,
        fingerprint="oblique-outward-fixture",
    )
    touched, normals = builder._centroid_boundary_normals(geometry)
    assert touched.tolist() == [True]
    assert np.dot(normals[0], expected) > 0.99
    for cell, midpoint, normal in zip(
        range(geometry.nc), face_coords, face_normals
    ):
        assert cell == 0
        assert np.dot(normal, midpoint - geometry.centers[cell]) > 0.0
    np.testing.assert_array_equal(geometry.polygons[0], polygon)
    np.testing.assert_array_equal(geometry.face_normals, face_normals)


def test_actual_polygon_cells_share_one_scale_and_use_true_zero_differences(
    builder, tmp_path
):
    output, geometry, _, _ = _build_fixture(tmp_path, builder)
    touched, cell_normals = builder._centroid_boundary_normals(geometry)
    assert np.all(touched)
    np.testing.assert_allclose(np.linalg.norm(cell_normals, axis=1), 1.0)
    data = json.loads((output / "data.json").read_text())
    pair = data["pairs"]["representative"]
    for field in pair["fields"].values():
        assert len(field["panels"]) == 3
        assert len({tuple(panel["limits"]) for panel in field["panels"]}) == 1
        assert field["panels"][2]["formula"] == (
            "perturbed(t) - control(t)"
        )
        assert field["difference_check"]["perturbed_change_max_abs"] > 0.0
        assert field["difference_check"]["control_change_max_abs"] > 0.0
        assert field["difference_check"]["pair_difference_max_abs"] > 0.0
    control0 = builder.load_snapshot_arrays(
        next(
            path
            for path in sorted((tmp_path / "pair-control/snapshots").glob("*.npz"))
            if "0000000000" in path.name
        )
    )
    control_t = builder.load_snapshot_arrays(
        next(
            path
            for path in sorted((tmp_path / "pair-control/snapshots").glob("*.npz"))
            if "0000000010" in path.name
        )
    )
    perturbed0 = builder.load_snapshot_arrays(
        next(
            path
            for path in sorted((tmp_path / "pair-perturbed/snapshots").glob("*.npz"))
            if "0000000000" in path.name
        )
    )
    perturbed_t = builder.load_snapshot_arrays(
        next(
            path
            for path in sorted((tmp_path / "pair-perturbed/snapshots").glob("*.npz"))
            if "0000000010" in path.name
        )
    )
    field = builder.FIELD_BY_KEY["B_R"]
    differences = builder.field_differences(
        control0, control_t, perturbed0, perturbed_t, field, 0
    )
    np.testing.assert_allclose(
        differences["perturbed_change"],
        perturbed_t["B_N_samples"][0, 0] - perturbed0["B_N_samples"][0, 0],
    )
    np.testing.assert_allclose(
        differences["control_change"],
        control_t["B_N_samples"][0, 0] - control0["B_N_samples"][0, 0],
    )
    np.testing.assert_allclose(
        differences["pair_difference"],
        perturbed_t["B_N_samples"][0, 0] - control_t["B_N_samples"][0, 0],
    )
    assert np.any(
        differences["perturbed_change"] != differences["control_change"]
    )
    index = (output / "index.html").read_text()
    assert "Native polygon cells" in index
    assert "Input contour" in index
    assert "Computational chord wall" in index


def test_initial_rhs_force_cache_panels_use_explicit_arrays(builder, tmp_path):
    output, _, _, _ = _build_fixture(
        tmp_path,
        builder,
        control_options={"force_cache": True},
        perturbed_options={"force_cache": True},
    )
    data = json.loads((output / "data.json").read_text())
    force = data["pairs"]["representative"]["initial_force_cache"]
    assert force["available"] is True
    assert force["missing_keys"] == []
    assert set(force["components"]) == set(builder.FORCE_KEYS.values())
    assert any(name.endswith("_force_cache.svg") for name in data["figures"])


def test_missing_force_cache_is_explicit_and_never_zero_filled(builder, tmp_path):
    output, _, _, _ = _build_fixture(tmp_path, builder)
    data = json.loads((output / "data.json").read_text())
    force = data["pairs"]["representative"]["initial_force_cache"]
    assert force["available"] is False
    assert force["status"] == "not_supplied"
    assert all(value is None for value in force["components"].values())
    notes = "\n".join(
        json.loads((output / "manifest.json").read_text())["pages"][0]["notes"]
    )
    assert "initial-rhs.npz" in notes


@pytest.mark.parametrize("case,match", [
    ("time_mismatch", "physical time mismatch"),
    ("missing_t0", "true initial snapshot"),
])
def test_bad_time_matching_is_refused(builder, tmp_path, case, match):
    geometry = _fixture_geometry(builder)
    control = _make_run(tmp_path, builder, geometry, "control")
    options = {case: True}
    perturbed = _make_run(
        tmp_path, builder, geometry, "perturbed", **options
    )
    with pytest.raises(builder.GalleryError, match=match):
        builder.build(
            [builder.PairSpec("bad-time", control, perturbed)],
            tmp_path / "gallery",
            source_root=SOURCE_ROOT,
        )


def test_nonfinite_and_missing_phi_are_refused(builder, tmp_path):
    geometry = _fixture_geometry(builder)
    control = _make_run(tmp_path, builder, geometry, "control")
    perturbed = _make_run(
        tmp_path, builder, geometry, "perturbed", nonfinite=True
    )
    with pytest.raises(builder.GalleryError, match="non-finite"):
        builder.build(
            [builder.PairSpec("bad-field", control, perturbed)],
            tmp_path / "gallery-nonfinite",
            source_root=SOURCE_ROOT,
        )
    coherent = _make_run(
        tmp_path, builder, geometry, "coherent-control", missing_phi=True
    )
    bad_metadata = _make_run(
        tmp_path, builder, geometry, "coherent-perturbed", missing_phi=True
    )
    with pytest.raises(builder.GalleryError, match="sample_phis"):
        builder.build(
            [builder.PairSpec("missing-phi", coherent, bad_metadata)],
            tmp_path / "gallery-phi",
            source_root=SOURCE_ROOT,
        )


def test_missing_geometry_npz_is_refused_without_mesh_guess(builder, tmp_path):
    geometry = _fixture_geometry(builder)
    control = _make_run(tmp_path, builder, geometry, "control")
    perturbed = _make_run(tmp_path, builder, geometry, "perturbed")
    (perturbed / "geometry.npz").unlink()
    with pytest.raises(
        builder.GalleryError, match="missing explicit geometry.npz"
    ):
        builder.build(
            [builder.PairSpec("missing-geometry", control, perturbed)],
            tmp_path / "gallery-missing-geometry",
            source_root=SOURCE_ROOT,
        )
    assert not (tmp_path / "gallery-missing-geometry").exists()


def test_geometry_sha_mismatch_is_refused(builder, tmp_path):
    geometry = _fixture_geometry(builder)
    control = _make_run(tmp_path, builder, geometry, "control")
    perturbed = _make_run(tmp_path, builder, geometry, "perturbed")
    report_path = control / "report.json"
    report = json.loads(report_path.read_text())
    report["geometry"]["sha256"] = "0" * 64
    _write_json(report_path, report)
    with pytest.raises(builder.GalleryError, match="geometry.npz SHA256 mismatch"):
        builder.build(
            [builder.PairSpec("bad-geometry-sha", control, perturbed)],
            tmp_path / "gallery-bad-geometry-sha",
            source_root=SOURCE_ROOT,
        )
    assert not (tmp_path / "gallery-bad-geometry-sha").exists()


def test_wall_band_empty_is_unavailable_and_pressures_are_included(
    builder, tmp_path
):
    output, _, _, _ = _build_fixture(tmp_path, builder)
    data = json.loads((output / "data.json").read_text())
    pair = data["pairs"]["representative"]
    assert {"p_i", "p_e_perp", "p_e_parallel"} <= set(pair["fields"])
    band = pair["wall_band"]
    assert band["width_m"] == pytest.approx(0.1)
    assert band["regions"]["wall_band"]["count"] == 0
    assert band["regions"]["wall_band"]["volume_m3"] == 0.0
    for field in band["regions"]["wall_band"]["fields"].values():
        for difference in field.values():
            assert difference["rms"] is None
            assert difference["linf"] is None
    assert band["regions"]["complement"]["count"] == 6
    assert band["regions"]["complement"]["volume_m3"] > 0.0


def test_k1_adds_actual_sampled_orthogonal_phase_without_interpolation(
    builder, tmp_path
):
    output, _, _, _ = _build_fixture(tmp_path, builder, K=1, mode=0)
    data = json.loads((output / "data.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    phases = data["pairs"]["representative"]["phases"]
    assert len(phases) == 2
    assert phases[0]["index"] == 0
    assert phases[0]["angle_rad"] == 0.0
    nearest_quarter = 2.0 * np.pi / 5.0
    assert phases[1]["index"] == 1
    assert phases[1]["angle_rad"] == pytest.approx(nearest_quarter)
    assert phases[1]["orthogonal_offset_rad"] == pytest.approx(
        np.pi / 2.0 - nearest_quarter
    )
    assert "no interpolation" in phases[1]["selection_note"].casefold()
    assert "real_data_result" not in data
    assert data["input_origin"] == (
        "caller-supplied native run reports and snapshot arrays; "
        "provenance hashes recorded"
    )
    assert data["scientific_acceptance"] == "not established by renderer"
    assert "caller-supplied" in data["scope"]
    assert "native polygon" in data["scope"]
    assert "physical wall accuracy" in data["scope"]
    assert "pending" not in data["scope"].casefold()
    assert manifest["input_origin"] == data["input_origin"]
    assert (
        manifest["scientific_acceptance"]
        == data["scientific_acceptance"]
    )
    assert {
        (figure["n_phi"], figure["k"], figure["n_phi_quadrature"])
        for figure in manifest["pages"][0]["figures"]
    } == {(0, 1, 5)}


def test_excited_harmonic_one_remains_distinct_from_cutoff_and_quadrature(
    builder, tmp_path
):
    output, _, _, _ = _build_fixture(tmp_path, builder, K=1, mode=1)
    manifest = json.loads((output / "manifest.json").read_text())
    figure_metadata = {
        (figure["n_phi"], figure["k"], figure["n_phi_quadrature"])
        for figure in manifest["pages"][0]["figures"]
    }
    assert figure_metadata == {(1, 1, 5)}


def test_a0_css_embedding_input_hashes_and_native_svg_labels(builder, tmp_path):
    output, _, _, _ = _build_fixture(tmp_path, builder)
    index = (output / "index.html").read_text()
    assert "<style>" in index
    assert ":root{color-scheme:light;font-family:system-ui,sans-serif" in index
    assert 'src="data:image/svg+xml;base64,' in index
    assert not re.search(r'src="(?!data:)', index)
    data = json.loads((output / "data.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert data["figure_count"] == len(data["figures"])
    assert data["figure_count"] <= 32
    assert data["input_sha256"]
    assert manifest["input_sha256"] == data["input_sha256"]
    expected_canonical = {
        "build_a0_report.py",
        "build_a1_t0_report.py",
        "build_a1_t2_report.py",
        "build_050_sections.py",
    }
    assert set(data["canonical_sha256"]) == expected_canonical
    assert builder._resolve_source_root(None) == SOURCE_ROOT
    assert any(
        record["path"].endswith("geometry.npz")
        for record in data["input_sha256"]
    )
    svg_paths = sorted((output / "figures").glob("*.svg"))
    assert svg_paths
    for path in svg_paths:
        text = path.read_text()
        assert "Input contour" in text
        assert "Computational chord wall" in text
        assert "m s" in text or "(T)" in text or "(Pa)" in text or "m^{-3}" in text
        assert not re.search(r"[\u4e00-\u9fff]", text)
    sums = (output / "SHA256SUMS").read_text()
    assert "index.html" in sums
    assert "manifest.json" in sums
    assert "data.json" in sums


def test_existing_output_and_more_than_32_planned_figures_are_refused(
    builder, tmp_path
):
    geometry = _fixture_geometry(builder)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(builder.GalleryError, match="new output"):
        builder.build(
            [builder.PairSpec("one", tmp_path, tmp_path)],
            existing,
            source_root=SOURCE_ROOT,
        )

    runs = [
        (
            _make_run(tmp_path, builder, geometry, f"k1-control-{index}", K=1),
            _make_run(
                tmp_path, builder, geometry, f"k1-perturbed-{index}", K=1
            ),
        )
        for index in range(2)
    ]
    with pytest.raises(builder.GalleryError, match="more than 32 figures"):
        builder.build(
            [
                builder.PairSpec(f"pair-{index}", control, perturbed)
                for index, (control, perturbed) in enumerate(runs)
            ],
            tmp_path / "too-many",
            source_root=SOURCE_ROOT,
        )
