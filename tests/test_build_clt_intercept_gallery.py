from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import clt_boundary_coefficients
from tools import clt_physical_rz_evidence
from tools.plots import build_clt_intercept_gallery as gallery


def test_local_schema_rejection_uses_only_supplied_object() -> None:
    with pytest.raises(gallery.InputError, match="schema"):
        gallery.validate_local_pairing({"schema": "wrong"})


def test_output_overwrite_and_source_containment_use_tmp_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    input_path = tmp_path / "input.json"
    input_path.write_text("{}\n", encoding="utf-8")
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(gallery.InputError, match="existing"):
        gallery.validate_output([input_path], source, existing)
    with pytest.raises(gallery.InputError, match="source tree"):
        gallery.validate_output([input_path], source, source / "new-output")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lineage_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    worker = tmp_path / "worker"
    producer = worker / "scratch/probe_local_patch.py"
    implementation = worker / "scratch/repo/src/fomhd/boundary_patch.py"
    local_json = worker / "scratch/local_patch_probe_repair1.json"
    status = worker / "STATUS.json"
    implementation.parent.mkdir(parents=True)
    producer.parent.mkdir(parents=True, exist_ok=True)
    producer.write_text(
        "from fomhd.boundary_patch import TriangleGeometry\n",
        encoding="utf-8",
    )
    implementation.write_text("class TriangleGeometry: pass\n", encoding="utf-8")
    local_json.write_text('{"schema": "local"}\n', encoding="utf-8")
    status.write_text(
        json.dumps(
            {
                "schema": "fomhd-clt-local-pairing-worker-status-v1",
                "status": "complete_bounded_parent_repair1",
                "worker_root": str(worker),
                "artifacts": {
                    "implementation": {
                        "path": "scratch/repo/src/fomhd/boundary_patch.py",
                        "sha256": _hash(implementation),
                    },
                    "probe": {
                        "path": "scratch/probe_local_patch.py",
                        "sha256": _hash(producer),
                    },
                    "corrected_json": {
                        "path": "scratch/local_patch_probe_repair1.json",
                        "sha256": _hash(local_json),
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return producer, implementation, local_json, status


def test_local_lineage_uses_existing_status_binding(tmp_path: Path) -> None:
    producer, implementation, local_json, status = _lineage_fixture(tmp_path)
    lineage = gallery.validate_local_lineage(
        local_json,
        producer,
        implementation,
        status,
    )
    assert lineage["producer_import"] == "fomhd.boundary_patch"
    assert lineage["producer_sha256"] == _hash(producer)
    assert lineage["implementation_sha256"] == _hash(implementation)

    receipt = json.loads(status.read_text(encoding="utf-8"))
    receipt["artifacts"]["implementation"]["sha256"] = "0" * 64
    status.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    with pytest.raises(gallery.InputError, match="STATUS"):
        gallery.validate_local_lineage(
            local_json,
            producer,
            implementation,
            status,
        )


@pytest.mark.parametrize(
    ("rule", "offset_h"),
    [
        ("four_point", 0.75),
        ("five_point", 2.25),
    ],
)
@pytest.mark.parametrize("side", ("upper", "lower"))
@pytest.mark.parametrize(("order", "attribute"), [
    (1, "first_derivative_weights"),
    (2, "second_derivative_weights"),
])
def test_known_stencil_normalization_is_h_invariant(
    rule: str,
    offset_h: float,
    side: str,
    order: int,
    attribute: str,
) -> None:
    stencil = clt_boundary_coefficients.irregular_stencil(
        rule,
        offset_h,
        side=side,
    )
    weights = np.asarray(getattr(stencil, attribute), dtype=np.float64)
    expected = float(np.sum(np.abs(weights)))
    for h_m in (0.125, 0.25, 1.0):
        actual = gallery.normalized_derivative_l1(weights, h_m, order)
        assert actual == pytest.approx(expected, rel=0.0, abs=2e-14)


def test_source_profile_reports_dimensionless_weights() -> None:
    profile = gallery.build_source_profile(
        clt_boundary_coefficients,
        Path(clt_boundary_coefficients.__file__),
    )
    assert profile["normalization"] == (
        "||D_phys||_1 * h^p == ||dimensionless weights||_1"
    )
    for panel in ("a", "b"):
        for series in profile["panels"][panel]["series"]:
            assert "normalized_l1" in series
            assert "scaled_l1" not in series
            assert np.all(np.asarray(series["normalized_l1"]) > 0.0)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_repair3_rigid_evidence_checks_status_and_strain(tmp_path: Path) -> None:
    json_path = tmp_path / "repair3.json"
    npz_path = tmp_path / "repair3.npz"
    producer = tmp_path / "probe.py"
    status = tmp_path / "STATUS.json"
    _write_json(
        json_path,
        {
            "schema": "coordinated_structured_intercept_probe_repair3_v1",
            "status": "REPAIR3_SOURCE_BUILDER_ACCEPTED_COUPLED_SYSTEM_INFEASIBLE",
            "scientific_rejection": False,
            "candidate_complete": False,
            "physical_acceptance": False,
            "stage": {"called": False, "time_steps_run": 0},
            "source": {"probe_sha256": "PENDING"},
            "rz_trends": {"cases": []},
        },
    )
    producer.write_text("from fomhd.backend import Backend\n", encoding="utf-8")
    arrays = {}
    trend_cases = []
    for nodes in (8, 12, 16):
        prefix = f"circle-N{nodes}"
        h_m = 2.0 / nodes
        axis_R = 2.0 + (np.arange(nodes, dtype=np.float64) + 0.5) * h_m
        axis_Z = -1.0 + (np.arange(nodes, dtype=np.float64) + 0.5) * h_m
        grid_R, grid_Z = np.meshgrid(axis_R, axis_Z)
        cells_R = grid_R.reshape(-1)
        cells_Z = grid_Z.reshape(-1)
        arrays[f"{prefix}__staircase_old_cell_R"] = cells_R
        arrays[f"{prefix}__staircase_old_cell_Z"] = cells_Z
        arrays[f"{prefix}__staircase_candidate_cell_R"] = arrays[
            f"{prefix}__staircase_old_cell_R"
        ]
        arrays[f"{prefix}__staircase_candidate_cell_Z"] = arrays[
            f"{prefix}__staircase_old_cell_Z"
        ]
        arrays[f"{prefix}__staircase_old_symmetric_strain_magnitude"] = np.arange(
            nodes * nodes, dtype=np.float64
        )
        strain = np.zeros((5, 3, 3, nodes * nodes), dtype=np.complex128)
        strain[0, 0, 0, :] = (nodes / 8.0) * np.arange(nodes * nodes)
        arrays[f"{prefix}__positive_rigid_symmetric_strain"] = strain
        trend_cases.append({"nodes": nodes, "h_m": h_m})
    np.savez(npz_path, **arrays)
    probe_hash = _hash(producer)
    document = json.loads(json_path.read_text(encoding="utf-8"))
    document["source"]["probe_sha256"] = probe_hash
    document["rz_trends"]["cases"] = trend_cases
    _write_json(json_path, document)
    _write_json(
        status,
        {
            "schema": "fomhd-coordinated-structured-intercept-repair3-status-v1",
            "status": "complete_repair3_source_builder_accepted_coupled_infeasible",
            "rigid_rotation": {
                "physical_gate_passed": False,
                "staircase_boundary_strain": [0.1, 0.2, 0.3],
                "positive_candidate_boundary_strain": [0.2, 0.3, 0.4],
            },
            "artifacts": {
                "json": {"sha256": _hash(json_path)},
                "npz": {"sha256": _hash(npz_path)},
                "probe": {"sha256": probe_hash},
            },
        },
    )
    loaded = clt_physical_rz_evidence.load_repair3_rigid_evidence(
        json_path,
        npz_path,
        producer,
        status,
    )
    assert loaded["figure"]["physical_gate_passed"] is False
    assert loaded["figure"]["grids"]["N8"]["old_magnitude"][0] == pytest.approx(0.0)
    assert loaded["figure"]["grids"]["N16"]["positive_magnitude"][-1] > 0.0
    n8 = loaded["figure"]["grids"]["N8"]
    assert n8["wall_center_R_m"] == pytest.approx(3.0)
    assert n8["wall_center_Z_m"] == pytest.approx(0.0)
    assert n8["wall_radius_m"] == pytest.approx(1.0)
    assert n8["domain_R_m"] == pytest.approx([2.0, 4.0])
    assert n8["domain_Z_m"] == pytest.approx([-1.0, 1.0])
    radii = np.hypot(
        np.asarray(n8["true_R_m"]) - 3.0,
        np.asarray(n8["true_Z_m"]),
    )
    assert np.allclose(radii, 1.0, rtol=0.0, atol=2e-14)
    assert n8["old_wall_segments_m"]
    assert all(
        segment[0] == pytest.approx(segment[2])
        or segment[1] == pytest.approx(segment[3])
        for segment in n8["old_wall_segments_m"]
    )


def test_active_cell_wall_segments_follow_missing_neighbor_edges() -> None:
    R = np.array([0.0, 1.0])
    Z = np.array([0.0, 1.0])
    active = np.array(
        [
            [True, True],
            [False, True],
        ]
    )
    segments = clt_physical_rz_evidence.cell_edge_wall_segments(R, Z, active)
    assert segments
    assert all(
        segment[0] == pytest.approx(segment[2])
        or segment[1] == pytest.approx(segment[3])
        for segment in segments
    )
    corners = {
        (
            round(segment[0], 12),
            round(segment[1], 12),
            round(segment[2], 12),
            round(segment[3], 12),
        )
        for segment in segments
    }
    assert (-0.5, -0.5, 0.5, -0.5) in corners
    assert (0.5, 0.5, 0.5, 1.5) in corners


def test_old_formal_evidence_has_shared_physical_limits(
    tmp_path: Path,
) -> None:
    root = tmp_path / "old-formal"
    producer = tmp_path / "probe_staircase_mapping.py"
    implementation = tmp_path / "tools/prepare_staircase_equilibrium.py"
    status = tmp_path / "STATUS.json"
    aggregate = tmp_path / "old_staircase_mapping.json"
    producer.write_text(
        "from prepare_staircase_equilibrium import build_and_run\n",
        encoding="utf-8",
    )
    implementation.parent.mkdir()
    implementation.write_text("def build_and_run(): pass\n", encoding="utf-8")
    cases = []
    for nodes in (8, 12):
        grid = root / f"g{nodes}"
        grid.mkdir(parents=True)
        R = 2.5 + (np.arange(nodes) + 0.5) / nodes
        Z = -0.5 + (np.arange(nodes) + 0.5) / nodes
        np.savez(
            grid / "geometry.npz",
            R_nodes=R,
            Z_nodes=Z,
            lcfs_R=np.array([2.5, 3.5, 3.5, 2.5]),
            lcfs_Z=np.array([-0.5, -0.5, 0.5, 0.5]),
        )
        np.savez(
            grid / "initial.npz",
            n_i=np.ones((1, nodes, nodes), dtype=np.complex128),
        )
        mom_i = np.zeros((3, 1, nodes, nodes), dtype=np.complex128)
        mom_i[0, 0] = np.arange(nodes * nodes, dtype=np.float64).reshape(
            nodes, nodes
        )
        B_phi = np.zeros((1, nodes + 1, nodes + 1), dtype=np.complex128)
        B_phi[0, 0, 0] = -3.0
        B_phi[0, -1, -1] = 3.0
        np.savez(
            grid / "initial_rhs.npz",
            mom_i=mom_i,
            p_perp_e=np.full((1, nodes, nodes), 2.0, dtype=np.complex128),
            B_phi=nodes * B_phi,
        )
        (grid / "case.toml").write_text(f"NR={nodes}\n", encoding="utf-8")
        source_manifest = {
            "input_hashes": {
                "case": _hash(grid / "case.toml"),
                "geometry": _hash(grid / "geometry.npz"),
                "initial": _hash(grid / "initial.npz"),
            }
        }
        _write_json(grid / "source_manifest.json", source_manifest)
        mapping = {
            "nodes": nodes,
            "input_hashes": {
                **source_manifest["input_hashes"],
                "source_manifest": _hash(grid / "source_manifest.json"),
            },
        }
        _write_json(grid / "mapping.json", mapping)
        cases.append(mapping)
    _write_json(
        aggregate,
        {
            "schema": "fomhd_equilibrium_old_staircase_mapping_v1",
            "scope": (
                "formal old structured staircase initialization only; no time step"
            ),
            "limits": {"K": 0, "Nphi": 5, "time_steps": 0},
            "cases": cases,
        },
    )
    _write_json(
        status,
        {
            "schema": "fomhd-equilibrium-audit-worker-status-v1",
            "status": "complete_bounded_parent_review_corrected",
            "artifacts": {
                "staircase_source": {"sha256": _hash(implementation)},
                "mapping_json": {"sha256": _hash(aggregate)},
            },
        },
    )
    loaded = clt_physical_rz_evidence.load_old_formal_evidence(
        root,
        aggregate,
        status,
        producer,
        implementation,
    )
    limits = loaded["figure"]["shared_limits"]
    assert limits["mom_i_abs"][0] == pytest.approx(0.0)
    assert limits["B_phi"][0] == pytest.approx(-36.0)
    assert limits["B_phi"][1] == pytest.approx(36.0)
