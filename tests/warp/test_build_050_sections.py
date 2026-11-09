from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = next(
    path
    for path in (
        Path(__file__).parents[1] / "build_050_sections.py",
        Path(__file__).parents[2] / "tools/plots/build_050_sections.py",
    )
    if path.is_file()
)


@pytest.fixture
def sections_module():
    spec = importlib.util.spec_from_file_location("build_050_sections", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def mapping(module, R: np.ndarray, Z: np.ndarray, polygon: np.ndarray):
    return module.Mapping(
        R=np.asarray(R, dtype=float),
        Z=np.asarray(Z, dtype=float),
        polygon=np.asarray(polygon, dtype=float),
        radial_edges=module.axis_boundaries(R, "R"),
        axial_edges=module.axis_boundaries(Z, "Z"),
    )


def snapshot(module, step: int, time_s: float, vectors: np.ndarray, density: np.ndarray):
    return module.Snapshot(
        path=Path(f"snapshot-{step}.npz"),
        sha256=f"hash-{step}",
        step=step,
        time_s=time_s,
        magnetic=vectors,
        velocity=vectors.copy(),
        density=density,
    )


def test_nonrectangular_active_cells_have_no_bounding_grid_bridge(sections_module):
    R = np.array([0.5, 1.5, 0.5, 1.5, 0.5])
    Z = np.array([0.5, 0.5, 1.5, 1.5, 2.5])
    polygon = np.array(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]]
    )
    cells = mapping(sections_module, R, Z, polygon)
    assert cells.vertices.shape == (5, 4, 2)
    assert np.allclose(cells.vertices[:, 0, 0], R - 0.5)
    assert np.allclose(cells.vertices[:, 2, 0], R + 0.5)
    assert np.allclose(cells.vertices[:, 0, 1], Z - 0.5)
    assert np.allclose(cells.vertices[:, 2, 1], Z + 0.5)
    assert len(cells.vertices) < 2 * 3


def test_vector_sample_layout_selects_phi_zero(sections_module):
    nphi, nc = 5, 3
    samples = np.zeros(nphi * 3 * nc, dtype=np.complex128)
    shaped = samples.reshape(nphi, 3, nc)
    shaped[0, 0] = [1, 2, 3]
    shaped[0, 1] = [4, 5, 6]
    shaped[0, 2] = [7, 8, 9]
    shaped[1] = 100.0
    values = sections_module.decode_vector_samples(
        samples, "observed_BN", nphi, nc, Path("sample.npz")
    )
    assert np.array_equal(values, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


def test_control_perturbed_delta_sign(sections_module):
    nc = 2
    control_vectors = np.full((3, nc), 2.0)
    perturbed_vectors = np.full((3, nc), 5.0)
    density = np.ones(nc)
    pair = sections_module.SnapshotPair(
        snapshot(sections_module, 1, 1e-11, control_vectors, density),
        snapshot(sections_module, 1, 1e-11, perturbed_vectors, density),
    )
    assert np.array_equal(
        sections_module.field_delta(pair, sections_module.FIELD_BY_KEY["B_phi"]),
        np.full(nc, 3.0),
    )


def test_delta_maps_receive_centered_shared_limit(sections_module):
    first = np.array([-1.0, 0.0, 1.0])
    second = np.array([-4.0, 2.0, 0.0])
    assert sections_module.symmetric_limits(first, second) == (-4.0, 4.0)
    assert sections_module.symmetric_limits(first, second) == sections_module.symmetric_limits(
        second, first
    )


def test_time_mismatch_is_refused(sections_module):
    vectors = np.zeros((3, 1))
    density = np.ones(1)
    control = {
        4: snapshot(sections_module, 4, 1.0, vectors, density),
    }
    perturbed = {
        4: snapshot(sections_module, 4, 1.1, vectors, density),
    }
    with pytest.raises(sections_module.InputError, match="time mismatch"):
        sections_module.match_snapshots(control, perturbed)


def test_step_set_mismatch_is_refused(sections_module):
    vectors = np.zeros((3, 1))
    density = np.ones(1)
    control = {
        4: snapshot(sections_module, 4, 1.0, vectors, density),
        8: snapshot(sections_module, 8, 2.0, vectors, density),
    }
    perturbed = {
        4: snapshot(sections_module, 4, 1.0, vectors, density),
    }
    with pytest.raises(sections_module.InputError, match="cadence mismatch"):
        sections_module.match_snapshots(control, perturbed)


def test_grid_mismatch_is_refused(sections_module):
    polygon = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0]])
    left = mapping(
        sections_module,
        np.array([0.5, 1.5]),
        np.array([0.5, 1.5]),
        polygon,
    )
    right = mapping(
        sections_module,
        np.array([0.5, 1.6]),
        np.array([0.5, 1.5]),
        polygon,
    )
    with pytest.raises(sections_module.InputError, match="R mismatch"):
        sections_module.require_compatible_grid(left, right)


def test_existing_output_and_input_nested_outputs_are_refused(
    sections_module, tmp_path
):
    run_root = tmp_path / "run"
    source_root = tmp_path / "source"
    run_root.mkdir()
    source_root.mkdir()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(sections_module.UnsafeOutputError, match="existing"):
        sections_module._safe_output(run_root, source_root, existing)
    with pytest.raises(sections_module.UnsafeOutputError, match="outside"):
        sections_module._safe_output(run_root, source_root, run_root / "output")
    with pytest.raises(sections_module.UnsafeOutputError, match="outside"):
        sections_module._safe_output(run_root, source_root, source_root / "output")


def test_physical_real_tolerance_is_enforced(sections_module):
    nphi, nc = 5, 2
    samples = np.zeros(nphi * 3 * nc, dtype=np.complex128)
    samples.reshape(nphi, 3, nc)[0] = 1.0 + 1.0j
    with pytest.raises(sections_module.InputError, match="physical-real"):
        sections_module.decode_vector_samples(
            samples, "observed_BN", nphi, nc, Path("bad.npz")
        )
    samples.reshape(nphi, 3, nc)[0].imag = 1e-15
    values = sections_module.decode_vector_samples(
        samples, "observed_BN", nphi, nc, Path("tolerated.npz")
    )
    assert np.array_equal(values, np.ones((3, nc)))
