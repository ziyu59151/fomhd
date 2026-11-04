from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fomhd.geometry import build_rectangle
from fomhd.operators import build_operators


from workflow_fixture import ORACLE_ROOT

ORACLE = ORACLE_ROOT / "native-oracle-4-k1.npz"
STRESS_ORACLE = ORACLE_ROOT / "native-oracle-stress.npz"


def assert_array(actual, expected, label: str) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape, (
        f"{label} shape: actual={actual.shape} expected={expected.shape}"
    )
    if expected.size == 0:
        return
    scale = max(1.0, float(np.max(np.abs(expected))))
    error = float(np.max(np.abs(actual - expected)))
    assert error <= 2e-11 * scale, f"{label}: max_abs_error={error} scale={scale}"


@pytest.mark.parametrize("oracle", [ORACLE, STRESS_ORACLE])
def test_foundation_matches_independent_native_arrays(oracle: Path) -> None:
    if not oracle.exists():
        pytest.skip("native oracle not available")
    native = np.load(oracle, allow_pickle=False)
    geometry = build_rectangle(4, 4, 1.0, 0.0, 0.2, 0.2)
    operators = build_operators(geometry, 1)

    assert_array(geometry.R, native["R"], "R")
    assert_array(geometry.Z, native["Z"], "Z")
    assert_array(geometry.cells.R, native["cell_R"], "cell_R")
    assert_array(geometry.cells.Z, native["cell_Z"], "cell_Z")
    for name in ("M0", "M1", "MB"):
        assert_array(getattr(operators, name), native[name], name)
    assert_array(geometry.wv, native["wv"], "wv")
    for name in (
        "PR_r",
        "PZ_r",
        "PR_z",
        "PZ_z",
        "PR_v",
        "PZ_v",
        "I",
        "H",
        "QB",
        "T_full",
        "T",
        "Bc_full",
        "Bc",
        "DR",
        "DZ",
        "topology_flux_row",
    ):
        assert_array(getattr(operators, name), native[name], name)
    for stored, mode in enumerate((-1, 0, 1)):
        for name in ("G", "C", "DB", "DJ", "Cdag", "Lphi"):
            assert_array(
                getattr(operators, name)[stored],
                native[f"{name}_{stored}"],
                f"{name}[{mode}]",
            )

    wall_edges = []
    wall_signs = []
    nr = len(geometry.r_faces.keys)
    nc = len(geometry.cells.keys)
    for wall in geometry.walls:
        wall_edges.append(
            wall.face if wall.component == "r_face" else nr + nc + wall.face
        )
        wall_signs.append(wall.sign)
    assert_array(wall_edges, native["wall_edges"], "wall_edges")
    assert_array(wall_signs, native["wall_signs"], "wall_signs")
