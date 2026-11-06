"""Behavioral checks for grid convergence and flat-cell structure plots."""
import json
from pathlib import Path

import numpy as np
import pytest

from tools.grid_mode_analysis import save_selected, sha256_path
from tools.plots.plot_grid_convergence import (
    flat_cell_grid,
    load_branch,
    physical_at_phi,
    render,
    response_seed_coherence,
    response_fields,
    selected_structure_fields,
    signed_structure_delta,
    structural_real,
) 


def _identity(geometry="a"):
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
        "geometry_sha256": geometry,
    }


def _write_branch(root: Path, label: str, nodes: int, frequency_hz: float,
                  R: np.ndarray, Z: np.ndarray, velocity: np.ndarray,
                  magnetic: np.ndarray, density: np.ndarray) -> tuple[Path, Path]:
    directory = root / label
    directory.mkdir()
    report = {
        "kind": "circle",
        "nodes": nodes,
        "K": 1,
        "mode": 1,
        "physical_identity": _identity(label),
        "candidates": [{
            "index": 3,
            "frequency_hz": frequency_hz,
            "growth_rate_per_s": 0.0,
            "parallel_fraction": 1.0,
            "magnetic_fraction": 0.5,
            "magnetic_transfer_ratio": 0.7,
            "MB_magnetic_norm": 4.0,
            "M0_ion_kinetic_norm": 2.0,
        }],
    }
    report_path = directory / "report.json"
    report_path.write_text(json.dumps(report) + "\n")
    nc = R.size
    spectrum = {
        "source": str(directory),
        "report": report,
        "arrays": {
            "candidate_indices": np.array([3], dtype=np.int64),
            "candidate_frequency_hz": np.array([frequency_hz]),
            "candidate_magnetic_transfer_ratio": np.array([0.7]),
            "candidate_MB_magnetic_norm": np.array([4.0]),
            "candidate_M0_ion_kinetic_norm": np.array([2.0]),
            "candidate_velocity": velocity[None],
            "candidate_B_node": magnetic[None],
            "candidate_density": density[None],
            "R": R,
            "Z": Z,
            "M0": np.ones(nc),
            "background_B_node": np.tile(
                np.array([[0.0], [1.0], [0.0]]), (1, nc)
            ),
            "rho": np.array(2.0),
            "mu0": np.array(2.0),
            "wall_R": np.array([-0.5, 1.5, 1.5, -0.5]),
            "wall_Z": np.array([-0.5, -0.5, 1.5, 1.5]),
        },
        "hashes": {
            "report_sha256": sha256_path(report_path),
            "spectrum_sha256": "1" * 64,
        },
    }
    selected_path = directory / "selected.npz"
    save_selected(selected_path, spectrum, 0)
    return report_path, selected_path


def _response_arrays(seed_B: np.ndarray, seed_velocity: np.ndarray,
                     seed_density: np.ndarray) -> dict:
    times = np.linspace(0.0, 0.3, 4)
    phases = np.exp(2j * np.pi * np.arange(5) / 5)
    angles = np.linspace(0.0, 1.0, times.size)
    coefficient_B = np.stack([
        np.exp(1j * angle) * seed_B / 2 for angle in angles
    ])
    coefficient_velocity = np.stack([
        np.exp(1j * angle) * seed_velocity / 2 for angle in angles
    ])
    density_coefficient = np.stack([
        np.exp(1j * angle) * seed_density / 2 for angle in angles
    ])
    velocity_samples = np.empty(
        (times.size, 5, 3, seed_B.shape[-1]), dtype=complex
    )
    density_samples = np.empty(
        (times.size, 5, seed_B.shape[-1]), dtype=complex
    )
    for time_index in range(times.size):
        for phase_index, phase in enumerate(phases):
            velocity_samples[time_index, phase_index] = (
                coefficient_velocity[time_index] * phase
                + coefficient_velocity[time_index].conj() / phase
            )
            density_samples[time_index, phase_index] = (
                density_coefficient[time_index] * phase
                + density_coefficient[time_index].conj() / phase
            )
    return {
        "time_s": times,
        "delta_B": coefficient_B,
        "delta_n": density_samples,
        "delta_velocity": velocity_samples,
        "parallel_B": coefficient_B[:, 1],
    }


def _write_response_provenance(
    run: Path,
    selected_path: Path,
    selected_report_path: Path,
    R: np.ndarray,
    Z: np.ndarray,
    *,
    kind: str = "circle",
    nodes: int = 8,
    mode: int = 1,
    frequency_hz: float = 1.0e6,
    mapping_R: np.ndarray | None = None,
    report_hash: str | None = None,
    spectrum_hash: str | None = None,
    include_case: bool = True,
    include_report: bool = True,
    include_mapping: bool = True,
) -> None:
    with np.load(selected_path, allow_pickle=False) as archive:
        selected_hash = str(archive["report_sha256"])
        selected_spectrum_hash = str(archive["spectrum_sha256"])
        K = int(archive["K"])
    if mapping_R is None:
        mapping_R = R
    if report_hash is None:
        report_hash = selected_hash
    if spectrum_hash is None:
        spectrum_hash = selected_spectrum_hash
    if include_case:
        (run / "case.toml").write_text(
            f'kind = "{kind}"\n'
            f"nodes = {nodes}\n"
            f"nphi = {mode}\n"
            f"K = {K}\n"
            "Nphi = 5\n"
            "electron_inertia = 0\n"
        )
    if include_report:
        (run / "report.json").write_text(json.dumps({
            "case": {"kind": kind, "nodes": nodes, "K": K},
            "selected_frequency_hz": frequency_hz,
            "spectra": [{
                "path": str(selected_report_path.parent),
                "report_sha256": report_hash,
                "spectrum_sha256": spectrum_hash,
                "mode": mode,
            }],
        }) + "\n")
    if include_mapping:
        np.savez_compressed(
            run / "input_mapping.npz", R=mapping_R, Z=Z,
            polygon=np.array([[-0.5, -0.5], [1.5, -0.5],
                              [1.5, 1.5], [-0.5, 1.5]]),
        )


def test_signed_flat_cell_deltas_use_actual_unencoded_amplitudes():
    previous = {
        "velocity": np.zeros((3, 4), dtype=complex),
        "B_node": np.zeros((3, 4), dtype=complex),
        "density": np.zeros(4, dtype=complex),
        "R": np.array([0.0, 1.0, 0.0, 1.0]),
        "Z": np.array([0.0, 0.0, 1.0, 1.0]),
        "M0": np.ones(4),
    }
    current = {
        "R": np.array([0.0, 1.0, 0.0, 1.0]),
        "Z": np.array([0.0, 0.0, 1.0, 1.0]),
        "M0": np.ones(4),
        "rho": np.array(2.0),
        "mu0": np.array(2.0),
        "background_B_node": np.tile(np.array([[0.0], [1.0], [0.0]]), (1, 4)),
    }
    fields = {
        "velocity": np.zeros((3, 4), dtype=complex),
        "B_node": np.zeros((3, 4), dtype=complex),
        "density": np.full(4, -4+2j),
    }
    fields["velocity"][0] = 2+3j
    fields["velocity"][2] = -3+1j
    fields["B_node"][1] = 5-2j
    delta = signed_structure_delta(previous, current, fields)
    np.testing.assert_allclose(delta["delta_B_parallel"], 5)
    np.testing.assert_allclose(delta["delta_n"], -4)
    np.testing.assert_allclose(delta["delta_vR"], 2)
    np.testing.assert_allclose(delta["delta_vZ"], -3)
    assert delta["covered_M0_fraction"] == 1.0
    assert structural_real(np.array([1+2j]))[0] == 1.0
    assert physical_at_phi(np.array([1+2j]), 1)[0] == 2.0
    assert physical_at_phi(np.array([3+2j]), 0)[0] == 3.0
    edge_R, edge_Z, grid = flat_cell_grid(
        np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 2.0, 3.0]),
    )
    assert edge_R.shape == (3,) and edge_Z.shape == (3,)
    np.testing.assert_allclose(grid[0], [1.0, 2.0])
    np.testing.assert_allclose(grid[1, 0], 3.0)
    assert np.isnan(grid[1, 1])


@pytest.mark.parametrize("factor", [-1.0, 2.0j])
def test_structure_delta_aligns_one_joint_complex_eigenvector_factor(factor):
    R = np.array([0.0, 1.0, 0.0, 1.0])
    Z = np.array([0.0, 0.0, 1.0, 1.0])
    base_velocity = np.array([
        [1+0.2j, 2-0.1j, 0.4j, -1+0.3j],
        [0.3-0.2j, -0.5j, 1+0.1j, 2],
        [-0.4j, 0.7, 1.2-0.3j, -0.8j],
    ], dtype=complex)
    base_B = np.array([
        [0.2+0.1j, -0.4j, 0.6, 0.1-0.2j],
        [1+0.3j, 0.2-0.1j, -0.7j, 0.4],
        [-0.2+0.5j, 0.3j, 0.8, -0.6j],
    ], dtype=complex)
    base_density = np.array([0.3-0.1j, -0.7+0.2j, 0.5j, 1+0.2j])
    previous = {
        "R": R,
        "Z": Z,
        "M0": np.ones(4),
        "velocity": base_velocity,
        "B_node": base_B,
        "density": base_density,
    }
    current = {
        "R": R,
        "Z": Z,
        "M0": np.ones(4),
        "rho": np.array(2.0),
        "mu0": np.array(2.0),
        "background_B_node": np.tile(
            np.array([[0.0], [1.0], [0.0]]), (1, 4)
        ),
    }
    fields = {
        "velocity": factor * base_velocity,
        "B_node": factor * base_B,
        "density": factor * base_density,
    }
    delta = signed_structure_delta(previous, current, fields)
    assert delta["alignment_factor"] == pytest.approx(factor, abs=1e-13)
    np.testing.assert_allclose(delta["delta_B_parallel"], 0, atol=1e-13)
    np.testing.assert_allclose(delta["delta_n"], 0, atol=1e-13)
    np.testing.assert_allclose(delta["delta_velocity"], 0, atol=1e-13)
    assert delta["covered_M0_fraction"] == 1.0


def test_one_grid_renders_absolute_selected_structure(tmp_path):
    R = np.array([0.0, 1.0, 0.0, 1.0])
    Z = np.array([0.0, 0.0, 1.0, 1.0])
    velocity = np.zeros((3, 4), dtype=complex)
    velocity[0] = [1, 2, 3, 4]
    velocity[2] = [-1, 0, 1, 2]
    magnetic = np.zeros((3, 4), dtype=complex)
    magnetic[1] = [0.5, 1.0, -0.5, 2.0]
    density = np.array([1.0, -1.0, 2.0, -2.0], dtype=complex)
    branch = _write_branch(
        tmp_path, "grid4", 4, 1.0e6, R, Z, velocity, magnetic, density
    )
    output = tmp_path / "figures-one"
    result = render([branch], output)
    assert (output / "grid_mode_selected_structure.png").is_file()
    assert (output / "grid_mode_selected_structure.svg").is_file()
    assert not (output / "grid_mode_structure_deltas.png").exists()
    structure = result["figures"]["structure"]
    assert structure["selected_files"]
    assert "needs at least two grids" in structure["delta_reason"]


def test_n0_velocity_panel_uses_declared_quadrature_phase():
    selected = {
        "velocity": np.zeros((3, 4), dtype=complex),
        "B_node": np.zeros((3, 4), dtype=complex),
        "density": np.array([1.0, -2.0, 3.0, -4.0], dtype=complex),
        "background_B_node": np.tile(
            np.array([[0.0], [1.0], [0.0]]), (1, 4)
        ),
    }
    selected["B_node"][1] = [0.5, 1.0, -0.5, 2.0]
    selected["velocity"][0] = [1j, 2j, 3j, 4j]
    selected["velocity"][1] = [2j, 2j, 2j, 2j]
    selected["velocity"][2] = [-2j, -1j, 1j, 2j]
    fields = selected_structure_fields(selected, mode=0)
    assert fields["compression_phase_rad"] == pytest.approx(0.0)
    assert fields["velocity_phase_rad"] == pytest.approx(-np.pi / 2)
    assert np.max(np.abs(fields["compression_velocity"])) < 1e-14
    assert np.max(fields["speed"]) > 0
    np.testing.assert_allclose(fields["speed"], np.sqrt([9, 9, 14, 24]))
    np.testing.assert_allclose(fields["velocity_snapshot"][0], [1, 2, 3, 4])
    np.testing.assert_allclose(fields["velocity_snapshot"][2], [-2, -1, 1, 2])
    np.testing.assert_allclose(fields["B_parallel"], [0.5, 1.0, -0.5, 2.0])
    np.testing.assert_allclose(fields["density"], [1.0, -2.0, 3.0, -4.0])


@pytest.mark.parametrize("corruption", [
    "missing_case",
    "missing_report",
    "missing_mapping",
    "family",
    "mode",
    "grid",
    "coordinates",
    "frequency",
    "report_hash",
    "spectrum_hash",
])
def test_response_rejects_missing_or_mismatched_provenance(tmp_path, corruption):
    R = np.array([0.0, 1.0, 0.0, 1.0])
    Z = np.array([0.0, 0.0, 1.0, 1.0])
    velocity = np.zeros((3, 4), dtype=complex)
    velocity[0] = [1, 2, 3, 4]
    velocity[2] = [-2, -1, 1, 2]
    magnetic = np.zeros((3, 4), dtype=complex)
    magnetic[1] = [0.5, 1.0, -0.5, 2.0]
    density = np.array([1.0, -1.0, 2.0, -2.0], dtype=complex)
    branch = _write_branch(
        tmp_path, "grid4", 4, 1.0e6, R, Z, velocity, magnetic, density
    )
    run = tmp_path / f"response-{corruption}"
    run.mkdir()
    response_path = run / "matched-response.npz"
    np.savez_compressed(
        response_path, **_response_arrays(magnetic, velocity, density)
    )
    kwargs = {
        "kind": "circle",
        "nodes": 4,
        "mode": 1,
        "frequency_hz": 1.0e6,
    }
    if corruption == "missing_case":
        kwargs["include_case"] = False
    elif corruption == "missing_report":
        kwargs["include_report"] = False
    elif corruption == "missing_mapping":
        kwargs["include_mapping"] = False
    elif corruption == "family":
        kwargs["kind"] = "east"
    elif corruption == "mode":
        kwargs["mode"] = 0
    elif corruption == "grid":
        kwargs["nodes"] = 3
    elif corruption == "coordinates":
        kwargs["mapping_R"] = R + 0.25
    elif corruption == "frequency":
        kwargs["frequency_hz"] = 2.0e6
    elif corruption == "report_hash":
        kwargs["report_hash"] = "0" * 64
    else:
        kwargs["spectrum_hash"] = "0" * 64
    _write_response_provenance(
        run, branch[1], branch[0], R, Z, **kwargs
    )
    with pytest.raises(ValueError, match="provenance"):
        render(
            [branch], tmp_path / f"figures-{corruption}",
            response_path=response_path, response_grid=4,
        )


def test_encoded_response_n1_uses_physical_factor_two():
    coefficient = 1+2j
    phi = 2*np.pi*np.arange(5)/5
    density_samples = np.array([
        physical_at_phi(np.array([coefficient]), 1, angle)[0] for angle in phi
    ])
    velocity_samples = np.array([
        physical_at_phi(np.full(3, coefficient), 1, angle) for angle in phi
    ])
    response = {
        "arrays": {
            "time_s": np.array([0.25]),
            "delta_n": np.tile(density_samples[None, :, None], (1, 1, 2)),
            "delta_velocity": np.tile(
                velocity_samples[None, :, :, None], (1, 1, 1, 2)
            ),
            "delta_B": np.full((1, 3, 2), coefficient, dtype=complex),
            "parallel_B": np.full((1, 2), coefficient, dtype=complex),
        }
    }
    branch = {
        "mode": 1,
        "selected": {
            "R": np.array([0.0, 1.0]),
            "Z": np.array([0.0, 1.0]),
        },
    }
    fields = response_fields(response, branch, 0)
    np.testing.assert_allclose(fields["delta_B_parallel"], 2.0)
    np.testing.assert_allclose(fields["delta_n"], 2.0)
    np.testing.assert_allclose(fields["delta_vR"], 2.0)
    np.testing.assert_allclose(fields["delta_vZ"], 2.0)
    assert fields["time_s"] == 0.25


def test_n0_response_coherence_uses_real_seed_span():
    times = np.arange(5, dtype=float) * 0.1
    angles = np.linspace(0.0, 1.2, times.size)
    seed_B = np.zeros((3, 2), dtype=complex)
    seed_velocity = np.zeros((3, 2), dtype=complex)
    seed_B[1, 0] = 1.0
    seed_B[2, 0] = 0.5j
    seed_velocity[0, 0] = 1.0 - 0.2j
    seed_velocity[2, 0] = -0.3j
    coefficient_B = np.stack([
        np.cos(angle) * seed_B.real + np.sin(angle) * seed_B.imag
        for angle in angles
    ])
    coefficient_velocity = np.stack([
        np.cos(angle) * seed_velocity.real + np.sin(angle) * seed_velocity.imag
        for angle in angles
    ])
    coefficient_B[-1] = 0
    coefficient_velocity[-1] = 0
    coefficient_B[-1, :, 1] = np.array([3.0, 0.0, 0.0])
    density_coefficient = np.cos(angles)[:, None] * np.ones((1, 2))
    response = {
        "arrays": {
            "time_s": times,
            "delta_B": coefficient_B,
            "delta_n": np.tile(
                density_coefficient[:, None, :], (1, 5, 1)
            ),
            "delta_velocity": np.tile(
                coefficient_velocity[:, None, :, :], (1, 5, 1, 1)
            ),
        }
    }
    branch = {
        "mode": 0,
        "selected": {
            "B_node": seed_B,
            "velocity": seed_velocity,
            "density": np.array([2.0, 0.0]),
            "M0": np.array([1.0, 2.0]),
            "rho": np.array(1.0),
            "mu0": np.array(1.0),
        },
    }
    result = response_seed_coherence(response, branch)
    np.testing.assert_allclose(result["coherence"][:-1], 1.0, atol=1e-14)
    np.testing.assert_allclose(result["residual_fraction"][:-1], 0.0, atol=1e-14)
    assert result["coherence"][-1] == pytest.approx(0.0, abs=1e-14)
    assert result["residual_fraction"][-1] == pytest.approx(1.0, abs=1e-14)
    assert result["summary"]["seed_subspace_rank"] == 2
    assert result["summary"]["mode_rule"] == "real span(Re seed, Im seed)"


def test_n1_response_coherence_uses_complex_seed_line_and_fourier_quadrature():
    times = np.arange(5, dtype=float) * 0.1
    angles = np.linspace(0.0, 1.4, times.size)
    phases = np.exp(2j * np.pi * np.arange(5) / 5)
    seed_B = np.zeros((3, 2), dtype=complex)
    seed_velocity = np.zeros((3, 2), dtype=complex)
    seed_B[0, 0] = 0.7 + 0.4j
    seed_B[2, 0] = -0.2 + 0.8j
    seed_velocity[0, 0] = 1.1 - 0.6j
    seed_velocity[1, 0] = 0.3j
    coefficient_B = np.stack([
        np.exp(1j * angle) * seed_B / 2 for angle in angles
    ])
    coefficient_B[-1] = 0
    coefficient_B[-1, :, 1] = np.array([0.0, 2.0, 0.0])
    coefficient_velocity = np.stack([
        np.exp(1j * angle) * seed_velocity / 2 for angle in angles
    ])
    coefficient_velocity[-1] = 0
    velocity_samples = np.empty((times.size, 5, 3, 2), dtype=complex)
    for time_index, coefficient in enumerate(coefficient_velocity):
        for phase_index, phase in enumerate(phases):
            velocity_samples[time_index, phase_index] = (
                coefficient * phase + coefficient.conj() / phase
            )
    density_coefficient = np.exp(1j * angles)[:, None] * np.array([[1.0, 0.0]])
    density_samples = np.tile(
        density_coefficient[:, None, :], (1, 5, 1)
    ).astype(complex)
    for time_index in range(times.size):
        for phase_index, phase in enumerate(phases):
            density_samples[time_index, phase_index] = (
                density_coefficient[time_index] * phase
                + density_coefficient[time_index].conj() / phase
            )
    response = {
        "arrays": {
            "time_s": times,
            "delta_B": coefficient_B,
            "delta_n": density_samples,
            "delta_velocity": velocity_samples,
        }
    }
    branch = {
        "mode": 1,
        "selected": {
            "B_node": seed_B,
            "velocity": seed_velocity,
            "density": np.array([1.0, 0.0]),
            "M0": np.array([1.0, 2.0]),
            "rho": np.array(1.0),
            "mu0": np.array(1.0),
        },
    }
    result = response_seed_coherence(response, branch)
    np.testing.assert_allclose(result["coherence"][:-1], 1.0, atol=1e-13)
    np.testing.assert_allclose(result["residual_fraction"][:-1], 0.0, atol=1e-13)
    assert result["coherence"][-1] == pytest.approx(0.0, abs=1e-13)
    assert result["residual_fraction"][-1] == pytest.approx(1.0, abs=1e-13)
    assert result["summary"]["seed_subspace_rank"] == 1
    assert result["summary"]["mode_rule"] == "phase-invariant complex seed line"


def test_render_writes_hash_manifest_and_detects_changed_report(tmp_path):
    R0 = np.array([0.0, 1.0, 0.0, 1.0])
    Z0 = np.array([0.0, 0.0, 1.0, 1.0])
    R1 = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    Z1 = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
    velocity0 = np.zeros((3, 4), dtype=complex)
    velocity0[0] = R0
    velocity0[2] = Z0
    velocity1 = np.zeros((3, 9), dtype=complex)
    velocity1[0] = R1 + 0.1
    velocity1[2] = Z1 - 0.1
    magnetic0 = np.zeros((3, 4), dtype=complex)
    magnetic1 = np.zeros((3, 9), dtype=complex)
    magnetic1[1] = 1.0 - R1 * Z1
    density0 = np.zeros(4, dtype=complex)
    density1 = -0.2 + R1 - Z1
    first = _write_branch(
        tmp_path, "grid4", 4, 0.0, R0, Z0, velocity0, magnetic0, density0
    )
    second = _write_branch(
        tmp_path, "grid8", 8, 1.0e6, R1, Z1, velocity1, magnetic1, density1
    )
    with np.load(second[1], allow_pickle=False) as archive:
        seed_B = np.array(archive["B_node"])
        seed_velocity = np.array(archive["velocity"])
        seed_density = np.array(archive["density"])
    times = np.linspace(0.0, 0.3, 4)
    phases = np.exp(2j * np.pi * np.arange(5) / 5)
    angles = np.linspace(0.0, 1.0, times.size)
    coefficient_B = np.stack([
        np.exp(1j * angle) * seed_B / 2 for angle in angles
    ])
    coefficient_velocity = np.stack([
        np.exp(1j * angle) * seed_velocity / 2 for angle in angles
    ])
    density_coefficient = np.stack([
        np.exp(1j * angle) * seed_density / 2 for angle in angles
    ])
    velocity_samples = np.empty((times.size, 5, 3, R1.size), dtype=complex)
    density_samples = np.empty((times.size, 5, R1.size), dtype=complex)
    for time_index in range(times.size):
        for phase_index, phase in enumerate(phases):
            velocity_samples[time_index, phase_index] = (
                coefficient_velocity[time_index] * phase
                + coefficient_velocity[time_index].conj() / phase
            )
            density_samples[time_index, phase_index] = (
                density_coefficient[time_index] * phase
                + density_coefficient[time_index].conj() / phase
            )
    response_run = tmp_path / "response-run"
    response_run.mkdir()
    response_path = response_run / "matched-response.npz"
    np.savez_compressed(
        response_path,
        time_s=times,
        delta_B=coefficient_B,
        delta_n=density_samples,
        delta_velocity=velocity_samples,
        parallel_B=coefficient_B[:, 1],
    )
    _write_response_provenance(
        response_run, second[1], second[0], R1, Z1,
        nodes=8, mode=1, frequency_hz=1.0e6,
    )
    output = tmp_path / "figures"
    result = render(
        [first, second], output, response_path=response_path, response_grid=8
    )
    assert result["mode"] == 1
    assert (output / "grid_mode_convergence.png").is_file()
    assert (output / "grid_mode_convergence.svg").is_file()
    assert (output / "grid_mode_selected_structure.png").is_file()
    assert (output / "grid_mode_selected_structure.svg").is_file()
    assert (output / "grid_mode_structure_deltas.png").is_file()
    assert (output / "grid_mode_structure_deltas.svg").is_file()
    assert (output / "matched_response_structure.png").is_file()
    assert (output / "matched_response_structure.svg").is_file()
    assert (output / "response_seed_coherence.png").is_file()
    assert (output / "response_seed_coherence.svg").is_file()
    assert (output / "response_seed_coherence.npz").is_file()
    manifest = json.loads((output / "input_hashes.json").read_text())
    assert len(manifest["branches"]) == 2
    assert all(len(item["selected_sha256"]) == 64 for item in manifest["branches"])
    assert manifest["branches"][0]["frequency_hz"] == 0.0
    assert manifest["response"]["sha256"] == sha256_path(response_path)
    assert manifest["response"]["provenance"]["validated"]["family"] == "circle"
    assert all(
        len(value) == 64
        for value in manifest["response"]["provenance"]["hashes"].values()
    )
    assert len(
        manifest["figures"]["structure"]["alignment_factors"]
    ) == 1
    assert len(
        manifest["figures"]["structure"]["selected_structure_phases"]
    ) == 2
    assert (
        result["figures"]["response"]["seed_coherence"]["summary"]["minimum_coherence"]
        > 0.99
    )
    first[0].write_text(first[0].read_text() + "\n")
    with pytest.raises(ValueError, match="report bytes"):
        load_branch(first[0], first[1])
