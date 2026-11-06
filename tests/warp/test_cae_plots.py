"""Behavioral checks for the offline Tokamak CAE response gallery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.plots import plot_tokamak_cae as cae_plots
from tools.plots.plot_tokamak_cae import (
    main,
    mode_history,
    physical_at_phi,
    select_default_cases,
)


REAL_ROOT = Path(
    "/home/lxzeng/runs/fomhd/a0-tokamak-cae__20261105T154538Z"
)


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def _fixture(
    root: Path,
    *,
    status: str = "budget_stopped",
    mode: int = 0,
) -> Path:
    case = root / f"circle4-n{mode}-test-r2"
    case.mkdir(parents=True)
    times = np.array([0.0, 1.0e-9, 2.0e-9])
    control_energy = [
        {
            "time_s": float(t),
            "step": index,
            "energy": {
                "ion_J": 10.0 + index,
                "electron_J": 20.0 + index,
                "field_J": 1000.0 + index,
                "total_J": 1030.0 + 3 * index,
                "balance_J": 0.0,
                "relative_total_change": index * 1.0e-4,
                "outward_J": 0.0,
                "source_J": 0.0,
            },
        }
        for index, t in enumerate(times)
    ]
    perturbed_energy = [
        {
            "time_s": float(t),
            "step": index,
            "energy": {
                "ion_J": 11.0 + 2 * index,
                "electron_J": 21.0,
                "field_J": 1001.0 - index,
                "total_J": 1033.0 + index,
                "balance_J": float((index - 1) * 1.0e-8),
                "relative_total_change": (index - 1) * 1.0e-9,
                "outward_J": index * 1.0e-12,
                "source_J": 0.0,
            },
        }
        for index, t in enumerate(times)
    ]
    _json(case / "control-history.json", control_energy)
    _json(case / "perturbed-history.json", perturbed_energy)

    density = np.full((3, 5, 3), 100.0, dtype=np.complex128)
    density[:, 1:, :] = 999.0
    control_density = np.full_like(density, 100.0)
    velocity = np.zeros((3, 5, 3, 3), dtype=np.complex128)
    velocity[:, :, 0, :] = 1.0 + np.arange(3)
    velocity[:, :, 2, :] = 0.5
    bn = np.zeros((3, 5, 3, 3), dtype=np.complex128)
    bn[:, :, 1, :] = 2.0
    np.savez(
        case / "control-signals.npz",
        time_s=times,
        density=control_density.reshape(3, 5 * 3),
        velocity=np.zeros_like(velocity).reshape(3, 5 * 3 * 3),
        BN=np.zeros_like(bn).reshape(3, 5 * 3 * 3),
        B=np.zeros_like(bn),
    )
    np.savez(
        case / "perturbed-signals.npz",
        time_s=times,
        density=density.reshape(3, 5 * 3),
        velocity=velocity.reshape(3, 5 * 3 * 3),
        BN=bn.reshape(3, 5 * 3 * 3),
        B=bn,
    )
    delta_n = np.zeros((3, 5, 3), dtype=np.float64)
    delta_n[0, :, 0] = 2.0
    delta_n[1, :, 0] = 3.0
    delta_n[2, :, 2] = 5.0
    matched_velocity = np.ones((3, 5, 3, 3), dtype=np.complex128)
    matched_velocity[0] *= 0.1
    matched_velocity[1] *= 0.2
    np.savez(
        case / "matched-response.npz",
        time_s=times,
        delta_B=np.ones((3, 3, 3), dtype=np.complex128),
        delta_n=delta_n,
        delta_velocity=matched_velocity,
        parallel_B=np.ones((3, 3)),
        probe_cells=np.array([0, 2]),
        perturbation_magnetic_J=np.array([1.0, 2.0, 4.0]),
        perturbation_ion_kinetic_J=np.array([0.5, 0.7, 1.0]),
    )
    np.savez(case / "perturbation_mapping.npz",
             raw_B=np.ones(7, dtype=np.complex128),
             mapped_B=np.ones(7, dtype=np.complex128),
             raw_velocity=np.ones((3, 3), dtype=np.complex128),
             mapped_velocity=np.ones((3, 3), dtype=np.complex128))
    np.savez(case / "input_mapping.npz",
             R=np.array([1.5, 1.7, 1.9]),
             Z=np.array([0.0, 0.1, -0.1]),
             polygon=np.array([[1.0, -1.0], [2.0, -1.0],
                               [2.0, 1.0], [1.0, 1.0]]),
             background_raw_B=np.ones(7, dtype=np.complex128),
             background_mapped_B=np.ones(7, dtype=np.complex128))
    (case / "case.toml").write_text(
        'kind = "circle"\n'
        "nodes = 4\n"
        f"nphi = {mode}\n"
        "K = 0\n"
        "Nphi = 5\n"
    )
    _json(case / "report.json", {
        "status": status,
        "device": "cpu",
        "failure": {"code": "BudgetStop", "message": "synthetic short prefix"},
        "case": {
            "kind": "circle",
            "nodes": 4,
            "K": 0,
            "Nphi": 5,
            "active_cells": 3,
            "profile_assumption": "uniform synthetic plasma",
        },
        "source_hashes": {"tools/warp_tokamak_cae.py": "a" * 64},
        "mode_fits": [{
            "cell": 2,
            "fit": {
                "available": False,
                "reason": "fewer than nine samples",
                "frequency_hz": 987654321.0,
            },
        }],
    })
    return case


def _structure_data(*, grid: int = 3, zero_parallel_b: bool = False) -> dict:
    coordinates = 1.0 + np.arange(grid, dtype=float)
    rr, zz = np.meshgrid(coordinates, coordinates)
    R = rr.ravel()
    Z = zz.ravel()
    active = np.ones(R.size, dtype=bool)
    if grid == 3:
        active[(R == 2.0) & (Z == 2.0)] = False
    times = np.array([0.0, 1.0e-9, 2.0e-9])
    parallel_b = np.zeros((times.size, R.size))
    if not zero_parallel_b:
        parallel_b[-1] = R - 2.0 * Z
    delta_n = np.zeros((times.size, 5, R.size))
    for phase in range(5):
        delta_n[-1, phase] = R + 2.0 * Z
    delta_velocity = np.zeros((times.size, 5, 3, R.size))
    delta_velocity[1, :, 0, :] = 0.3
    delta_velocity[1, :, 1, :] = 0.4
    delta_velocity[1, :, 2, :] = 1.2
    return {
        "name": f"structure-grid{grid}-n0-test-r1",
        "backend": "cpu",
        "nodes": grid,
        "K": 0,
        "mode": 0,
        "solver_status": "completed",
        "compression": {
            "available": True,
            "time_index": 2,
            "time_s": 2.0e-9,
            "probe_cell": 0,
            "reason": None,
        },
        "_arrays": {
            "input_mapping.npz": {
                "R": R[active],
                "Z": Z[active],
                "polygon": np.array([
                    [0.5, 0.5],
                    [grid + 0.5, 0.5],
                    [grid + 0.5, grid + 0.5],
                    [0.5, grid + 0.5],
                ]),
            },
            "matched-response.npz": {
                "time_s": times,
                "parallel_B": parallel_b[:, active],
                "delta_n": delta_n[:, :, active],
                "delta_velocity": delta_velocity[:, :, :, active],
                "probe_cells": np.array([0, R.size - 1]),
            },
        },
    }


def _retained_structure_figure(data: dict, tmp_path: Path, monkeypatch):
    import matplotlib.pyplot as plt

    plt.close("all")
    retained = {}

    def retain(figure):
        retained["figure"] = figure

    monkeypatch.setattr(plt, "close", retain)
    cae_plots._render_structure(data, tmp_path / "structure")
    return retained["figure"]


def _map_axes(figure) -> list:
    return [
        axis
        for axis in figure.axes
        if axis.get_xlabel() == "R [m]"
        and axis.get_ylabel() == "Z [m]"
    ]


def _quad_meshes(figure) -> list:
    return [
        item
        for axis in _map_axes(figure)
        for item in axis.collections
        if type(item).__name__ == "QuadMesh"
    ]


def test_default_selection_uses_latest_semantic_revision(tmp_path):
    for name in (
        "circle12-n0-cpu-pilot-r1",
        "circle12-n0-cpu-pilot-r2",
        "circle12-n1-cpu-pilot-r3",
        "east12-n0-cpu-smoke-r1",
        "east12-n0-cpu-smoke-r2",
        "circle-cpu-smoke-r5",
        "spectrum-circle12-k1-n1-r1",
    ):
        (tmp_path / name).mkdir()
        _json(tmp_path / name / "report.json", {"status": "failed"})

    assert select_default_cases(tmp_path) == [
        "circle12-n0-cpu-pilot-r2",
        "circle12-n1-cpu-pilot-r3",
        "east12-n0-cpu-smoke-r2",
    ]


def test_mode_history_retains_the_nonzero_zero_fourier_mean():
    samples = np.zeros((2, 5, 2), dtype=np.complex128)
    samples[:, 0, :] = 2.0
    samples[:, 1:, :] = 100.0
    actual = mode_history(samples, 0)
    np.testing.assert_array_equal(actual, np.full((2, 2), 80.4))


def test_mode_history_one_retains_complex_coefficient():
    samples = np.exp(1j * np.arange(5)[None, :, None] * 2 * np.pi / 5)
    samples = np.broadcast_to(samples, (3, 5, 2)).copy()
    actual = mode_history(samples, 1)
    np.testing.assert_allclose(actual, np.ones((3, 2), dtype=complex))


def test_n0_once_and_n1_half_coefficient_give_same_phi0_physics():
    amplitude = 5.0
    n0 = physical_at_phi(amplitude, 0, phi=0.0)
    n1 = physical_at_phi(amplitude / 2, 1, phi=0.0)
    np.testing.assert_allclose(n0, amplitude)
    np.testing.assert_allclose(n1, amplitude)
    np.testing.assert_allclose(n0, n1)


@pytest.mark.parametrize(
    "status", ["completed", "budget_stopped", "failed", "unavailable",
               "initializing"]
)
def test_evolution_figure_names_linearity_device_and_solver_status(status):
    from matplotlib.figure import Figure

    data = {
        "name": "circle20-n1-device-test-r1",
        "backend": "cuda",
        "nodes": 20,
        "K": 1,
        "mode": 1,
        "solver_status": status,
    }
    fig = Figure()
    cae_plots._figure_heading(fig, data, "Evidence summary")
    visible = "\n".join(text.get_text() for text in fig.texts)

    assert "NL | FOMHD time evolution" in visible
    assert "device=cuda" in visible
    assert f"solver status={status}" in visible
    assert "grid 20 x 20" in visible
    assert "n_phi=1" in visible
    if status == "completed":
        assert "no claim of completed evolution" not in visible
    else:
        assert "no claim of completed evolution" in visible
    banner = next(
        text for text in fig.texts
        if "NL | FOMHD time evolution" in text.get_text()
    )
    assert banner.get_color().lower() == "#00796b"


def test_n1_overview_renders_imaginary_retained_coefficient(tmp_path):
    run_root = tmp_path / "runs"
    case = _fixture(run_root, mode=1)
    output = tmp_path / "figures"

    assert main([
        "--run-root", str(run_root), "--output", str(output),
        "--cases", case.name,
    ]) == 0

    overview = (output / "overview" / f"{case.name}.svg").read_text()
    assert "n_phi=1" in overview
    assert "retained ion-density coefficient (A/2): imag" in overview
    assert "Imaginary retained ion-density coefficient" in overview


def test_synthetic_cli_preserves_stop_and_hashes_every_input(tmp_path):
    run_root = tmp_path / "runs"
    case = _fixture(run_root)
    output = tmp_path / "figures"

    assert main(["--run-root", str(run_root), "--output", str(output),
                 "--cases", case.name]) == 0

    receipt = json.loads((output / "receipts/report.json").read_text())
    assert receipt["gallery_id"] == "cae-20261106"
    assert receipt["scientific_acceptance"] is False
    assert receipt["cases"][0]["solver_status"] == "budget_stopped"
    assert receipt["cases"][0]["compression"]["time_s"] == pytest.approx(2e-9)
    assert receipt["cases"][0]["compression"]["probe_cell"] == 2
    assert receipt["cases"][0]["compression"]["scale_m-3"] == 100.0
    assert "scale_kg_m-3" not in receipt["cases"][0]["compression"]
    assert receipt["cases"][0]["compression"]["delta_n_phi0_m-3"] == 5.0
    assert receipt["cases"][0]["frequency"]["available"] is False
    spatial = receipt["cases"][0]["spatial_structure"]
    assert spatial["rendering"]["geometry"] == "flat actual cells"
    assert spatial["rendering"]["inactive_cells"] == "masked"
    assert spatial["rendering"]["interpolation"] is False
    assert spatial["density_time_s"] == pytest.approx(2.0e-9)
    assert spatial["velocity_time_s"] == pytest.approx(2.0e-9)
    assert spatial["fields"]["delta_B_parallel"]["unit"] == "T"
    assert spatial["fields"]["delta_n_i"]["unit"] == "m^-3"
    assert spatial["fields"]["ion_speed"]["unit"] == "m/s"
    assert "three velocity components" in (
        spatial["fields"]["ion_speed"]["definition"]
    )
    assert spatial["arrows"] == (
        "normalized reconstructed R/Z ion velocity direction only"
    )
    assert {
        item["role"] for item in receipt["inputs"]
        if item["path"].endswith("report.json")
    } == {"report"}
    read_names = {
        item["path"].rsplit("/", 1)[-1] for item in receipt["inputs"]
    }
    assert read_names == {
        "report.json", "case.toml", "control-history.json",
        "perturbed-history.json", "control-signals.npz",
        "perturbed-signals.npz", "matched-response.npz",
        "perturbation_mapping.npz", "input_mapping.npz",
    }
    for item in receipt["inputs"]:
        path = case / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        assert item["size_bytes"] == path.stat().st_size
    for stem in ("overview", "structure"):
        assert (output / stem / f"{case.name}.png").is_file()
        assert (output / stem / f"{case.name}.svg").is_file()
    overview_svg = (output / "overview" / f"{case.name}.svg").read_text()
    structure_svg = (output / "structure" / f"{case.name}.svg").read_text()
    for svg in (overview_svg, structure_svg):
        assert "NL | FOMHD time evolution" in svg
        assert "device=cpu" in svg
        assert "solver status=budget_stopped" in svg
        assert "n_phi=0" in svg
    assert "n_i" in structure_svg
    fragment = (output / "fragment.html").read_text()
    assert 'id="cae-20261106"' in fragment
    assert "budget_stopped" in fragment
    assert "fewer than nine samples" in fragment
    assert "987654321" not in fragment
    assert "Ion total energy (kinetic+internal)" in fragment
    assert "kg/m^3" not in fragment
    assert "m^-3" in fragment
    assert "A/2" in fragment
    assert "2 Re(c exp(i*n_phi*phi))" in (
        receipt["cases"][0]["field_meanings"]["physical_structure"]
    )
    assert "zero-Fourier" in fragment


def test_missing_explicit_case_gets_visible_placeholders(tmp_path):
    run_root = tmp_path / "runs"
    run_root.mkdir()
    output = tmp_path / "figures"

    assert main(["--run-root", str(run_root), "--output", str(output),
                 "--cases", "missing-case-r1"]) == 0

    receipt = json.loads((output / "receipts/report.json").read_text())
    assert receipt["cases"][0]["solver_status"] == "unavailable"
    assert receipt["cases"][0]["errors"]
    assert (output / "overview/missing-case-r1.png").is_file()
    assert (output / "structure/missing-case-r1.svg").is_file()


def test_existing_output_is_refused_before_input_read(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("unchanged")

    with pytest.raises(FileExistsError):
        main(["--run-root", str(tmp_path / "absent"),
              "--output", str(tmp_path)])

    assert marker.read_text() == "unchanged"


def test_real_short_case_smoke(tmp_path):
    if not (REAL_ROOT / "east12-n0-cpu-smoke-r2/report.json").is_file():
        pytest.skip("real short CPU case is unavailable")
    output = tmp_path / "actual"
    assert main([
        "--run-root", str(REAL_ROOT), "--output", str(output),
        "--cases", "east12-n0-cpu-smoke-r2",
    ]) == 0
    receipt = json.loads((output / "receipts/report.json").read_text())
    case = receipt["cases"][0]
    assert case["solver_status"] == "completed"
    assert case["backend"] == "cpu"
    assert case["mode"] == 0
    assert case["equilibrium_source_sha256"].startswith("df8f073b")
    assert case["compression"]["available"] is True
    assert case["compression"]["time_s"] == pytest.approx(8.318055109186038e-11)
    assert case["compression"]["probe_cell"] == 60
    assert case["compression"]["scale_m-3"] == pytest.approx(2e19)
    assert case["frequency"]["available"] is False
    assert receipt["input_count"] == 9


def test_structure_has_three_flat_physical_maps_and_arrow_directions(
    tmp_path, monkeypatch
):
    data = _structure_data()
    figure = _retained_structure_figure(data, tmp_path, monkeypatch)
    axes = _map_axes(figure)

    assert len(axes) == 3
    meshes = _quad_meshes(figure)
    assert len(meshes) == 3
    assert {mesh.cmap.name for mesh in meshes} == {"RdBu_r", "viridis"}
    b_mesh = next(
        mesh for mesh in meshes
        if mesh.cmap.name == "RdBu_r"
        and "parallel" in mesh.axes.get_title().lower()
    )
    n_mesh = next(
        mesh for mesh in meshes
        if mesh.cmap.name == "RdBu_r" and mesh is not b_mesh
    )
    speed_mesh = next(mesh for mesh in meshes if mesh.cmap.name == "viridis")

    expected_mask = np.zeros((3, 3), dtype=bool)
    expected_mask[1, 1] = True
    for mesh in meshes:
        assert np.array_equal(
            np.ma.getmaskarray(mesh.get_array()), expected_mask
        )
    expected_b = np.array([
        [-1.0, 0.0, 1.0],
        [-3.0, np.nan, -1.0],
        [-5.0, -4.0, -3.0],
    ])
    expected_n = np.array([
        [3.0, 4.0, 5.0],
        [5.0, np.nan, 7.0],
        [7.0, 8.0, 9.0],
    ])
    expected_speed = np.full((3, 3), 1.3)
    expected_speed[1, 1] = np.nan
    np.testing.assert_allclose(
        np.ma.array(np.ma.getdata(b_mesh.get_array()), mask=expected_mask),
        np.ma.array(expected_b, mask=expected_mask),
    )
    np.testing.assert_allclose(
        np.ma.array(np.ma.getdata(n_mesh.get_array()), mask=expected_mask),
        np.ma.array(expected_n, mask=expected_mask),
    )
    np.testing.assert_allclose(
        np.ma.array(np.ma.getdata(speed_mesh.get_array()), mask=expected_mask),
        np.ma.array(expected_speed, mask=expected_mask),
    )
    assert (b_mesh.norm.vmin, b_mesh.norm.vmax) == (-5.0, 5.0)
    assert (n_mesh.norm.vmin, n_mesh.norm.vmax) == (-9.0, 9.0)
    assert (speed_mesh.norm.vmin, speed_mesh.norm.vmax) == (0.0, 1.3)
    for mesh in meshes:
        assert mesh.get_clim()[1] > 1.0

    titles = [axis.get_title() for axis in axes]
    assert any(
        "parallel" in title.lower() and "t=2 ns" in title
        for title in titles
    )
    assert any(
        "delta n_i" in title and "t=2 ns" in title for title in titles
    )
    speed_title = speed_mesh.axes.get_title()
    assert "total 3d ion speed" in speed_title.lower()
    assert "normalized r/z" in speed_title.lower()
    assert "t=1 ns" in speed_title

    quiver = next(
        item
        for axis in axes
        for item in axis.collections
        if type(item).__name__ == "Quiver"
    )
    assert quiver.pivot == "middle"
    np.testing.assert_allclose(
        np.hypot(quiver.U, quiver.V), np.ones_like(quiver.U)
    )
    assert 0 < 1.0 / quiver.scale <= 0.4
    facecolor = np.asarray(quiver.get_facecolor())[:, :3]
    assert np.max(facecolor) <= 0.3

    for axis in axes:
        assert axis.get_legend() is None
        assert list(axis.texts) == []
        assert all(
            type(item).__name__ in {"QuadMesh", "Quiver"}
            for item in axis.collections
        )
        outlined = [
            line for line in axis.lines
            if line.get_linestyle() == "-"
            and line.get_color() in {
                "black", "#000000", (0.0, 0.0, 0.0)
            }
        ]
        assert outlined

    figure_text = "\n".join(text.get_text() for text in figure.texts)
    assert "NL | FOMHD time evolution" in figure_text
    assert figure_text.count("NL | FOMHD time evolution") == 1
    assert "grid 3 x 3" in figure_text
    assert "n_phi=0" in figure_text
    assert "phi=0" in figure_text
    assert "Offline response figure" not in figure_text
    assert "Solver status:" not in figure_text
    assert "Input wall polygon" not in figure_text


def test_zero_signed_field_remains_a_finite_physical_mesh(
    tmp_path, monkeypatch
):
    data = _structure_data(zero_parallel_b=True)
    figure = _retained_structure_figure(data, tmp_path, monkeypatch)
    b_mesh = next(
        mesh for mesh in _quad_meshes(figure)
        if "parallel" in mesh.axes.get_title().lower()
    )

    mask = np.ma.getmaskarray(b_mesh.get_array())
    values = np.ma.getdata(b_mesh.get_array())[~mask]
    assert np.all(np.isfinite(values))
    assert np.all(values == 0.0)
    assert np.count_nonzero(mask) == 1
    assert (b_mesh.norm.vmin, b_mesh.norm.vmax) == (-1.0, 1.0)
    assert all(
        type(item).__name__ != "PathCollection"
        for item in b_mesh.axes.collections
    )


def test_structure_colorbars_and_labels_fit_export_canvas(tmp_path, monkeypatch):
    figure = _retained_structure_figure(_structure_data(), tmp_path, monkeypatch)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis in figure.axes:
        bounds = axis.get_tightbbox(renderer)
        assert bounds.x0 >= -1
        assert bounds.x1 <= figure.bbox.width + 1
        assert bounds.y0 >= -1
        assert bounds.y1 <= figure.bbox.height + 1


@pytest.mark.parametrize("mapping_error", ["missing", "misaligned"])
def test_structure_receipt_matches_unavailable_mapping(
    tmp_path, monkeypatch, mapping_error
):
    data = _structure_data()
    if mapping_error == "missing":
        del data["_arrays"]["input_mapping.npz"]
    else:
        mapping = data["_arrays"]["input_mapping.npz"]
        mapping["Z"] = mapping["Z"][:-1]
    figure = _retained_structure_figure(data, tmp_path, monkeypatch)
    assert not _quad_meshes(figure)
    spatial = cae_plots._case_public(data)["spatial_structure"]
    assert spatial["available"] is False
    assert spatial["reason"]
    assert "mapping" in spatial["reason"] or "aligned" in spatial["reason"]


def test_structure_thins_and_bounds_arrows_on_a_large_grid(
    tmp_path, monkeypatch
):
    data = _structure_data(grid=20)
    figure = _retained_structure_figure(data, tmp_path, monkeypatch)
    quiver = next(
        item
        for axis in _map_axes(figure)
        for item in axis.collections
        if type(item).__name__ == "Quiver"
    )

    count = len(quiver.get_offsets())
    assert 0 < count < 400
    np.testing.assert_allclose(
        np.hypot(quiver.U, quiver.V), np.ones_like(quiver.U)
    )
    assert 0 < 1.0 / quiver.scale <= 0.4
