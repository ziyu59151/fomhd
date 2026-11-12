"""A0 formal aligned-box wall entry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fomhd import io
from fomhd.backend import Backend
from fomhd.geometry import (
    Point2,
    build_aligned_box,
    build_rectangle,
)
from fomhd.operators import build_operators
from fomhd.types import Context, Failure
from test_model_routing import _parameters, _state, _write_case


R_NODES = np.array([0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
Z_NODES = np.array([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5])
PHYSICAL_CORNERS = {
    (0.4, -0.6),
    (1.6, -0.6),
    (1.6, 0.6),
    (0.4, 0.6),
}
HISTORICAL_LCFS = [
    (0.5, -0.5),
    (1.5, -0.5),
    (1.5, 0.5),
    (0.5, 0.5),
]
MARKER_LCFS = [
    (0.54, -0.04),
    (0.58, -0.04),
    (0.58, 0.04),
    (0.54, 0.04),
]


def _aligned_box(lcfs=MARKER_LCFS):
    return build_aligned_box(R_NODES, Z_NODES, lcfs)


def test_aligned_box_has_full_cell_extents_corners_and_every_cell_active() -> None:
    geometry = _aligned_box()

    np.testing.assert_array_equal(geometry.active, 1)
    assert len(geometry.cells.keys) == R_NODES.size * Z_NODES.size
    assert geometry.cell_id(0, 0) == 0
    assert geometry.cell_id(R_NODES.size - 1, Z_NODES.size - 1) == 35

    r_boundary = geometry.r_faces.R[geometry.r_faces.boundary != 0]
    z_boundary = geometry.z_faces.Z[geometry.z_faces.boundary != 0]
    assert min(r_boundary) == pytest.approx(0.4)
    assert max(r_boundary) == pytest.approx(1.6)
    assert min(z_boundary) == pytest.approx(-0.6)
    assert max(z_boundary) == pytest.approx(0.6)

    polyline = {
        (round(point.R, 12), round(point.Z, 12))
        for point in geometry.wall_polyline
    }
    assert PHYSICAL_CORNERS <= polyline
    assert len(geometry.walls) == 4 * R_NODES.size


def test_lcfs_is_a_marker_and_never_masks_aligned_box_cells() -> None:
    geometry = _aligned_box(MARKER_LCFS)

    assert geometry.lcfs_polygon == [
        Point2(R, Z) for R, Z in MARKER_LCFS
    ]
    assert geometry.polygon == [
        Point2(0.4, -0.6),
        Point2(1.6, -0.6),
        Point2(1.6, 0.6),
        Point2(0.4, 0.6),
    ]
    np.testing.assert_array_equal(geometry.active, 1)


def test_legacy_rectangle_keeps_its_center_vertex_mask_and_no_lcfs_marker() -> None:
    geometry = build_rectangle(4, 3, 0.5, -0.2, 0.1, 0.1)

    assert geometry.lcfs_polygon is None
    assert np.count_nonzero(geometry.active) == 6
    assert len(geometry.cells.keys) == 6


@pytest.mark.parametrize(
    ("R", "Z", "lcfs", "message"),
    [
        (
            np.array([0.5, 0.7, 0.95]),
            Z_NODES[:3],
            MARKER_LCFS,
            "not uniform",
        ),
        (
            np.array([0.05, 0.25, 0.45]),
            Z_NODES[:3],
            MARKER_LCFS,
            "R<=0",
        ),
        (
            R_NODES[:3],
            Z_NODES[:3],
            [(0.5, -0.1), (0.7, 0.1), (0.7, -0.1), (0.5, 0.1)],
            "self-intersecting",
        ),
        (
            R_NODES[:3],
            Z_NODES[:3],
            [(0.5, -0.1), (0.8, -0.1), (0.8, 0.1), (0.5, 0.1)],
            "strictly inside",
        ),
    ],
    ids=("irregular", "nonpositive-wall", "self-crossing", "outside-wall"),
)
def test_aligned_box_rejects_invalid_grid_and_lcfs(
    R, Z, lcfs, message
) -> None:
    with pytest.raises(Failure, match=message):
        build_aligned_box(R, Z, lcfs)


def _rewrite_source(case_path: Path) -> None:
    root = case_path.parent
    source_path = root / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["input_hashes"]["case"] = io.sha256_file(case_path)
    source["input_hashes"]["geometry"] = io.sha256_file(root / "geometry.npz")
    source_path.write_text(json.dumps(source), encoding="utf-8")


def _set_geometry(case_path: Path, lcfs) -> None:
    polygon = np.asarray(lcfs, dtype=np.float64)
    np.savez(
        case_path.parent / "geometry.npz",
        R_nodes=R_NODES,
        Z_nodes=Z_NODES,
        lcfs_R=polygon[:, 0],
        lcfs_Z=polygon[:, 1],
    )
    _rewrite_source(case_path)


def _set_boundary(case_path: Path, kind: str, version: str, potential=0.0) -> None:
    lines = case_path.read_text(encoding="utf-8").splitlines()
    replacements = {
        "kind": f'kind="{kind}"',
        "version": f'version="{version}"',
        "potential": f"potential={potential!r}",
    }
    for index, line in enumerate(lines):
        key = line.partition("=")[0].strip()
        if key in replacements:
            lines[index] = replacements[key]
    case_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _rewrite_source(case_path)


def _formal_case(tmp_path):
    parameters = _parameters(
        inertia=0, integrator="rk4", nsteps=0, hard=True
    )
    geometry = _aligned_box(HISTORICAL_LCFS)
    compute = Backend("cpu")
    operators = build_operators(geometry, 0, backend=compute)
    context = Context(
        p=parameters,
        geometry=geometry,
        op=operators,
        compute=compute,
        N=parameters.Nphi,
    )
    case_path = _write_case(
        tmp_path,
        parameters,
        geometry,
        _state(context),
        "rk4",
        0,
    )
    _set_geometry(case_path, HISTORICAL_LCFS)
    _set_boundary(
        case_path, "fixed_outer_wall_static", "rectangle_relative_v1"
    )
    return case_path


def _digest_for(geometry_hash_input: str, version: str) -> str:
    return hashlib.sha256((geometry_hash_input + version).encode()).hexdigest()


def test_formal_rectangle_identity_is_distinct_from_same_historical_input(
    tmp_path,
) -> None:
    case_path = _formal_case(tmp_path)
    current = io.load_case(case_path)

    assert np.all(current.geometry.active == 1)
    assert current.geometry.lcfs_polygon == [
        Point2(R, Z) for R, Z in HISTORICAL_LCFS
    ]
    assert current.geometry_path.name == "geometry.npz"
    assert current.geometry_hash == _digest_for(
        current.input_hashes["geometry"], "rectangle_relative_v1"
    )

    _set_boundary(case_path, "fixed_lcfs_static", "staircase_relative_v1")
    historical = io.load_case(case_path)

    assert historical.input_hashes["geometry"] == current.input_hashes["geometry"]
    assert historical.geometry.lcfs_polygon is None
    assert not np.all(historical.geometry.active == 1)
    assert historical.geometry_hash == _digest_for(
        historical.input_hashes["geometry"], "half_open_ray_certified_v2"
    )
    assert historical.geometry_hash != current.geometry_hash
    assert historical.semantic_hash != current.semantic_hash


def test_formal_rectangle_rejects_lcfs_touching_an_outer_wall(tmp_path) -> None:
    case_path = _formal_case(tmp_path)
    _set_geometry(
        case_path,
        [(-0.2, -0.2), (1.0, -0.2), (1.0, 0.2), (-0.2, 0.2)],
    )

    with pytest.raises(Failure, match="strictly inside"):
        io.load_case(case_path)


def test_formal_rectangle_rejects_nonzero_potential(tmp_path) -> None:
    case_path = _formal_case(tmp_path)
    _set_boundary(
        case_path,
        "fixed_outer_wall_static",
        "rectangle_relative_v1",
        potential=1.0,
    )

    with pytest.raises(ValueError, match="unsupported boundary"):
        io.load_case(case_path)
