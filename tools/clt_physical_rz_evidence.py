"""Read-only loaders for the two physical RZ initial/rigid evidence bundles."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


class PhysicalEvidenceError(ValueError):
    """Raised when explicit physical-evidence inputs violate their receipt."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhysicalEvidenceError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise PhysicalEvidenceError(f"JSON root must be an object: {path}")
    return value


def _require(path: Path) -> None:
    if not path.is_file():
        raise PhysicalEvidenceError(f"required evidence file is missing: {path}")


def _npz(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    before = _sha256(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = required - set(archive.files)
            if missing:
                raise PhysicalEvidenceError(
                    f"{path.name} is missing arrays: {sorted(missing)}"
                )
            arrays = {name: np.array(archive[name], copy=True) for name in required}
    except (OSError, ValueError) as error:
        raise PhysicalEvidenceError(f"cannot load NPZ evidence: {path}") from error
    if _sha256(path) != before:
        raise PhysicalEvidenceError(f"NPZ changed during read: {path}")
    for name, array in arrays.items():
        if array.dtype.hasobject or (
            array.dtype.kind in "fc" and not np.all(np.isfinite(array))
        ):
            raise PhysicalEvidenceError(
                f"{path.name}:{name} is not finite numeric data"
            )
    return arrays


def _copy(source: Path, name: str) -> dict[str, str]:
    return {"source": str(source.resolve()), "copy": name}


def _input(path: Path, role: str) -> dict[str, str]:
    return {"path": str(path.resolve()), "role": role}


def _uniform_axis(values: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise PhysicalEvidenceError(f"{name} coordinates are not finite")
    unique = np.unique(values)
    if unique.size < 2:
        raise PhysicalEvidenceError(f"{name} coordinates do not span a grid")
    spacing = np.diff(unique)
    if not np.allclose(spacing, spacing[0], rtol=1e-12, atol=1e-12):
        raise PhysicalEvidenceError(f"{name} coordinates are not uniformly spaced")
    return unique, float(spacing[0])


def cell_edge_wall_segments(
    R_centres: np.ndarray,
    Z_centres: np.ndarray,
    active_mask: np.ndarray,
) -> list[tuple[float, float, float, float]]:
    """Return the exact old-staircase wall as active-cell outer edges."""

    R, h_R = _uniform_axis(R_centres, "R")
    Z, h_Z = _uniform_axis(Z_centres, "Z")
    mask = np.asarray(active_mask, dtype=bool)
    if mask.shape != (Z.size, R.size):
        raise PhysicalEvidenceError("active-cell mask shape does not match its axes")
    if not np.any(mask):
        raise PhysicalEvidenceError("active-cell mask is empty")
    half_R = 0.5 * h_R
    half_Z = 0.5 * h_Z
    segments: list[tuple[float, float, float, float]] = []
    for j, z in enumerate(Z):
        for i, r in enumerate(R):
            if not mask[j, i]:
                continue
            if i == 0 or not mask[j, i - 1]:
                segments.append(
                    (float(r - half_R), float(z - half_Z),
                     float(r - half_R), float(z + half_Z))
                )
            if i == R.size - 1 or not mask[j, i + 1]:
                segments.append(
                    (float(r + half_R), float(z - half_Z),
                     float(r + half_R), float(z + half_Z))
                )
            if j == 0 or not mask[j - 1, i]:
                segments.append(
                    (float(r - half_R), float(z - half_Z),
                     float(r + half_R), float(z - half_Z))
                )
            if j == Z.size - 1 or not mask[j + 1, i]:
                segments.append(
                    (float(r - half_R), float(z + half_Z),
                     float(r + half_R), float(z + half_Z))
                )
    return segments


def _active_cell_wall_segments(
    R_centres: np.ndarray,
    Z_centres: np.ndarray,
) -> list[tuple[float, float, float, float]]:
    R, inverse_R = np.unique(np.asarray(R_centres, dtype=np.float64), return_inverse=True)
    Z, inverse_Z = np.unique(np.asarray(Z_centres, dtype=np.float64), return_inverse=True)
    mask = np.zeros((Z.size, R.size), dtype=bool)
    mask[inverse_Z, inverse_R] = True
    return cell_edge_wall_segments(R, Z, mask)


def _circle_coordinates(
    center_R_m: float,
    center_Z_m: float,
    radius_m: float,
) -> tuple[list[float], list[float]]:
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    return (
        (center_R_m + radius_m * np.cos(theta)).tolist(),
        (center_Z_m + radius_m * np.sin(theta)).tolist(),
    )


def load_repair3_rigid_evidence(
    json_path: Path,
    npz_path: Path,
    producer: Path,
    status_path: Path,
) -> dict[str, Any]:
    """Load valid repair3 old/positive-M0 rigid-rotation evidence."""

    json_path, npz_path, producer, status_path = (
        path.resolve() for path in (json_path, npz_path, producer, status_path)
    )
    for path in (json_path, npz_path, producer, status_path):
        _require(path)
    status = _json(status_path)
    if status.get("schema") != (
        "fomhd-coordinated-structured-intercept-repair3-status-v1"
    ):
        raise PhysicalEvidenceError("repair3 STATUS schema mismatch")
    if status.get("status") != (
        "complete_repair3_source_builder_accepted_coupled_infeasible"
    ):
        raise PhysicalEvidenceError("repair3 STATUS is not the accepted repair3 result")
    artifacts = status.get("artifacts")
    if not isinstance(artifacts, dict):
        raise PhysicalEvidenceError("repair3 STATUS has no artifacts")
    expected = {
        "json": json_path,
        "npz": npz_path,
        "probe": producer,
    }
    for key, path in expected.items():
        try:
            recorded = artifacts[key]["sha256"]
        except (KeyError, TypeError) as error:
            raise PhysicalEvidenceError(f"repair3 STATUS lacks {key} hash") from error
        if _sha256(path) != recorded:
            raise PhysicalEvidenceError(f"repair3 {key} hash differs from STATUS")

    document = _json(json_path)
    if document.get("schema") != "coordinated_structured_intercept_probe_repair3_v1":
        raise PhysicalEvidenceError("repair3 JSON schema mismatch")
    if document.get("physical_acceptance") is not False:
        raise PhysicalEvidenceError(
            "repair3 JSON must retain failed physical acceptance"
        )
    stage = document.get("stage")
    if not isinstance(stage, dict) or stage.get("called") is not False:
        raise PhysicalEvidenceError("repair3 rigid evidence must not call the stage")
    if stage.get("time_steps_run") != 0:
        raise PhysicalEvidenceError("repair3 rigid evidence must have zero time steps")
    source = document.get("source")
    if not isinstance(source, dict) or source.get("probe_sha256") != _sha256(producer):
        raise PhysicalEvidenceError("repair3 JSON probe hash mismatch")
    rigid_summary = status.get("rigid_rotation")
    if not isinstance(rigid_summary, dict):
        raise PhysicalEvidenceError("repair3 STATUS lacks rigid-rotation summary")
    if rigid_summary.get("physical_gate_passed") is not False:
        raise PhysicalEvidenceError("repair3 rigid-summary gate must remain failed")

    required: set[str] = set()
    for nodes in (8, 12, 16):
        prefix = f"circle-N{nodes}"
        required.update(
            {
                f"{prefix}__staircase_old_cell_R",
                f"{prefix}__staircase_old_cell_Z",
                f"{prefix}__staircase_candidate_cell_R",
                f"{prefix}__staircase_candidate_cell_Z",
                f"{prefix}__staircase_old_symmetric_strain_magnitude",
                f"{prefix}__positive_rigid_symmetric_strain",
            }
        )
    arrays = _npz(npz_path, required)
    trend_cases = document.get("rz_trends", {}).get("cases")
    if not isinstance(trend_cases, list):
        raise PhysicalEvidenceError("repair3 JSON lacks RZ trend cases")
    trend_h = {}
    for case in trend_cases:
        if not isinstance(case, dict):
            raise PhysicalEvidenceError("repair3 RZ trend case is not an object")
        nodes = case.get("nodes")
        h_m = case.get("h_m")
        if (
            isinstance(nodes, bool)
            or not isinstance(nodes, int)
            or isinstance(h_m, bool)
            or not isinstance(h_m, (int, float))
            or not math.isfinite(float(h_m))
        ):
            raise PhysicalEvidenceError("repair3 RZ trend metadata is invalid")
        trend_h[nodes] = float(h_m)
    wall_signature = None
    grids: dict[str, Any] = {}
    magnitude_max = 0.0
    difference_max = 0.0
    for nodes in (8, 12, 16):
        prefix = f"circle-N{nodes}"
        old = arrays[f"{prefix}__staircase_old_symmetric_strain_magnitude"].real
        strain = arrays[f"{prefix}__positive_rigid_symmetric_strain"]
        old_R = np.asarray(arrays[f"{prefix}__staircase_old_cell_R"], dtype=np.float64)
        old_Z = np.asarray(arrays[f"{prefix}__staircase_old_cell_Z"], dtype=np.float64)
        candidate_R = np.asarray(
            arrays[f"{prefix}__staircase_candidate_cell_R"], dtype=np.float64
        )
        candidate_Z = np.asarray(
            arrays[f"{prefix}__staircase_candidate_cell_Z"], dtype=np.float64
        )
        if not np.array_equal(old_R, candidate_R) or not np.array_equal(
            old_Z, candidate_Z
        ):
            raise PhysicalEvidenceError(
                f"N{nodes} staircase/candidate cell coordinates differ"
            )
        if strain.shape[2:] != (3, old.size) or strain[0].shape != (3, 3, old.size):
            raise PhysicalEvidenceError(f"N{nodes} positive strain shape mismatch")
        R, h_R = _uniform_axis(old_R, f"N{nodes} R")
        Z, h_Z = _uniform_axis(old_Z, f"N{nodes} Z")
        expected_h = trend_h.get(nodes)
        if expected_h is None or not math.isclose(
            h_R, expected_h, rel_tol=1e-12, abs_tol=1e-12
        ) or not math.isclose(h_Z, expected_h, rel_tol=1e-12, abs_tol=1e-12):
            raise PhysicalEvidenceError(
                f"N{nodes} cell spacing differs from RZ trend metadata"
            )
        domain_R = (float(R[0] - 0.5 * h_R), float(R[-1] + 0.5 * h_R))
        domain_Z = (float(Z[0] - 0.5 * h_Z), float(Z[-1] + 0.5 * h_Z))
        radius_candidates = (
            0.5 * (domain_R[1] - domain_R[0]),
            0.5 * (domain_Z[1] - domain_Z[0]),
        )
        if not math.isclose(
            radius_candidates[0],
            radius_candidates[1],
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise PhysicalEvidenceError(
                f"N{nodes} R3 circle domain is not square"
            )
        center_R = 0.5 * (domain_R[0] + domain_R[1])
        center_Z = 0.5 * (domain_Z[0] + domain_Z[1])
        radius = 0.5 * sum(radius_candidates)
        signature = (center_R, center_Z, radius)
        if wall_signature is None:
            wall_signature = signature
        elif not np.allclose(signature, wall_signature, rtol=1e-10, atol=1e-10):
            raise PhysicalEvidenceError("R3 circle wall differs across grids")
        true_R, true_Z = _circle_coordinates(center_R, center_Z, radius)
        wall_segments = _active_cell_wall_segments(old_R, old_Z)
        positive = np.sqrt(np.sum(np.abs(strain[0]) ** 2, axis=(0, 1)))
        difference = positive - old
        magnitude_max = max(magnitude_max, float(np.max(old)), float(np.max(positive)))
        difference_max = max(difference_max, float(np.max(np.abs(difference))))
        grids[f"N{nodes}"] = {
            "old_R_m": arrays[f"{prefix}__staircase_old_cell_R"].tolist(),
            "old_Z_m": arrays[f"{prefix}__staircase_old_cell_Z"].tolist(),
            "candidate_R_m": arrays[
                f"{prefix}__staircase_candidate_cell_R"
            ].tolist(),
            "candidate_Z_m": arrays[
                f"{prefix}__staircase_candidate_cell_Z"
            ].tolist(),
            "old_magnitude": old.tolist(),
            "positive_magnitude": positive.tolist(),
            "difference": difference.tolist(),
            "true_R_m": true_R,
            "true_Z_m": true_Z,
            "h_R_m": h_R,
            "h_Z_m": h_Z,
            "domain_R_m": list(domain_R),
            "domain_Z_m": list(domain_Z),
            "wall_center_R_m": center_R,
            "wall_center_Z_m": center_Z,
            "wall_radius_m": radius,
            "old_wall_segments_m": wall_segments,
            "old_boundary_max_s_inv": float(
                rigid_summary["staircase_boundary_strain"][
                    (nodes // 4) - 2
                ]
            ),
            "positive_boundary_max_s_inv": float(
                rigid_summary["positive_candidate_boundary_strain"][
                    (nodes // 4) - 2
                ]
            ),
        }
    if wall_signature is None:
        raise PhysicalEvidenceError("repair3 evidence has no circle wall")
    true_R, true_Z = _circle_coordinates(*wall_signature)
    return {
        "figure": {
            "units": {"strain": "s^-1", "length": "m"},
            "normalization": "raw symmetric velocity-gradient strain; no rescaling",
            "physical_gate_passed": bool(rigid_summary["physical_gate_passed"]),
            "time_steps": 0,
            "true_R_m": true_R,
            "true_Z_m": true_Z,
            "wall_derivation": (
                "R3 cell-centre domain expanded by half the fixture h from "
                "rz_trends; circle centred on the domain with r=half-width"
            ),
            "shared_limits": {
                "magnitude": [0.0, magnitude_max],
                "difference": [-difference_max, difference_max],
            },
            "grids": grids,
        },
        "lineage": {
            "json": str(json_path),
            "json_sha256": _sha256(json_path),
            "npz": str(npz_path),
            "npz_sha256": _sha256(npz_path),
            "producer": str(producer),
            "producer_sha256": _sha256(producer),
            "status": str(status_path),
            "status_sha256": _sha256(status_path),
        },
        "inputs": [
            _input(json_path, "repair3 structured evidence JSON"),
            _input(npz_path, "repair3 structured evidence NPZ"),
            _input(producer, "repair3 evidence producer"),
            _input(status_path, "repair3 STATUS receipt"),
        ],
        "copies": [
            _copy(json_path, "structured_repair3.json"),
            _copy(npz_path, "structured_repair3.npz"),
            _copy(producer, "structured_repair3_producer.py"),
            _copy(status_path, "structured_repair3_STATUS.json"),
        ],
    }


def _edges(nodes: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 1 or nodes.size < 2 or not np.all(np.diff(nodes) > 0.0):
        raise PhysicalEvidenceError("old-formal grid coordinates are invalid")
    edges = np.empty(nodes.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (nodes[:-1] + nodes[1:])
    edges[0] = nodes[0] - (edges[1] - nodes[0])
    edges[-1] = nodes[-1] + (nodes[-1] - edges[-2])
    return edges


def load_old_formal_evidence(
    root: Path,
    audit_json: Path,
    status_path: Path,
    producer: Path,
    implementation: Path,
) -> dict[str, Any]:
    """Load N8/N12 full formal initial staircase RHS evidence."""

    root, audit_json, status_path, producer, implementation = (
        path.resolve()
        for path in (root, audit_json, status_path, producer, implementation)
    )
    for path in (audit_json, status_path, producer, implementation):
        _require(path)
    status = _json(status_path)
    if status.get("schema") != "fomhd-equilibrium-audit-worker-status-v1":
        raise PhysicalEvidenceError("equilibrium STATUS schema mismatch")
    if status.get("status") != "complete_bounded_parent_review_corrected":
        raise PhysicalEvidenceError("equilibrium STATUS is not the corrected audit")
    artifacts = status.get("artifacts")
    if not isinstance(artifacts, dict):
        raise PhysicalEvidenceError("equilibrium STATUS has no artifacts")
    if artifacts["staircase_source"]["sha256"] != _sha256(implementation):
        raise PhysicalEvidenceError(
            "old-formal implementation hash differs from STATUS"
        )
    if artifacts["mapping_json"]["sha256"] != _sha256(audit_json):
        raise PhysicalEvidenceError("old-formal aggregate JSON differs from STATUS")
    expected_implementation = producer.parent / (
        "tools/prepare_staircase_equilibrium.py"
    )
    if implementation != expected_implementation.resolve():
        raise PhysicalEvidenceError(
            "old-formal implementation path is not producer import"
        )
    tree = ast.parse(producer.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    if "prepare_staircase_equilibrium" not in imports:
        raise PhysicalEvidenceError("old-formal producer import is missing")

    aggregate = _json(audit_json)
    if aggregate.get("schema") != "fomhd_equilibrium_old_staircase_mapping_v1":
        raise PhysicalEvidenceError("old-formal aggregate schema mismatch")
    if aggregate.get("scope") != (
        "formal old structured staircase initialization only; no time step"
    ):
        raise PhysicalEvidenceError("old-formal aggregate scope mismatch")
    limits = aggregate.get("limits")
    if not isinstance(limits, dict) or limits.get("time_steps") != 0:
        raise PhysicalEvidenceError("old-formal data must have zero time steps")
    if limits.get("K") != 0 or limits.get("Nphi") != 5:
        raise PhysicalEvidenceError("old-formal K/Nphi contract mismatch")
    aggregate_cases = {
        int(case["nodes"]): case for case in aggregate.get("cases", [])
    }

    grids: dict[str, Any] = {}
    copies = [
        _copy(audit_json, "old_formal_audit.json"),
        _copy(status_path, "old_formal_STATUS.json"),
        _copy(producer, "old_formal_producer.py"),
        _copy(implementation, "old_formal_implementation.py"),
    ]
    inputs = [
        _input(audit_json, "old-formal aggregate audit JSON"),
        _input(status_path, "old-formal equilibrium STATUS"),
        _input(producer, "old-formal producer"),
        _input(implementation, "old-formal producer implementation"),
    ]
    signed_max = {"p_perp_e": 0.0, "B_phi": 0.0}
    mom_max = 0.0
    for nodes in (8, 12):
        grid = root / f"g{nodes}"
        required_paths = {
            name: grid / name
            for name in (
                "case.toml",
                "geometry.npz",
                "initial.npz",
                "initial_rhs.npz",
                "source_manifest.json",
                "mapping.json",
            )
        }
        for path in required_paths.values():
            _require(path)
            copies.append(_copy(path, f"old_formal/g{nodes}/{path.name}"))
            inputs.append(_input(path, f"old-formal g{nodes} {path.name}"))
        source_manifest = _json(required_paths["source_manifest.json"])
        recorded = source_manifest["input_hashes"]
        for key, name in (
            ("case", "case.toml"),
            ("geometry", "geometry.npz"),
            ("initial", "initial.npz"),
        ):
            if _sha256(required_paths[name]) != recorded[key]:
                raise PhysicalEvidenceError(f"g{nodes} {name} input hash mismatch")
        mapping = _json(required_paths["mapping.json"])
        if mapping["input_hashes"]["source_manifest"] != _sha256(
            required_paths["source_manifest.json"]
        ):
            raise PhysicalEvidenceError(f"g{nodes} source-manifest hash mismatch")
        aggregate_case = aggregate_cases.get(nodes)
        if aggregate_case is None or (
            aggregate_case.get("input_hashes") != mapping.get("input_hashes")
        ):
            raise PhysicalEvidenceError(f"g{nodes} aggregate mapping mismatch")

        rhs = _npz(
            required_paths["initial_rhs.npz"],
            {"mom_i", "p_perp_e", "B_phi"},
        )
        initial = _npz(required_paths["initial.npz"], {"n_i"})
        geometry = _npz(
            required_paths["geometry.npz"],
            {"R_nodes", "Z_nodes", "lcfs_R", "lcfs_Z"},
        )
        momentum = rhs["mom_i"]
        if momentum.shape != (3, 1, nodes, nodes):
            raise PhysicalEvidenceError(f"g{nodes} mom_i shape mismatch")
        if np.max(np.abs(np.imag(rhs["mom_i"]))) > 1e-12:
            raise PhysicalEvidenceError(f"g{nodes} mom_i is not a real physical RHS")
        if np.max(np.abs(np.imag(rhs["p_perp_e"]))) > 1e-12:
            raise PhysicalEvidenceError(f"g{nodes} p_perp_e is not real")
        if np.max(np.abs(np.imag(rhs["B_phi"]))) > 1e-12:
            raise PhysicalEvidenceError(f"g{nodes} B_phi is not real")
        mom_abs = np.sqrt(np.sum(np.abs(momentum[:, 0].real) ** 2, axis=0))
        p_perp = rhs["p_perp_e"][0].real
        B_phi = rhs["B_phi"][0].real
        support = initial["n_i"][0].real != 0.0
        if B_phi.shape != (nodes + 1, nodes + 1):
            raise PhysicalEvidenceError(f"g{nodes} B_phi shape mismatch")
        mom_max = max(mom_max, float(np.max(mom_abs)))
        signed_max["p_perp_e"] = max(
            signed_max["p_perp_e"], float(np.max(np.abs(p_perp)))
        )
        signed_max["B_phi"] = max(
            signed_max["B_phi"], float(np.max(np.abs(B_phi)))
        )
        R = geometry["R_nodes"]
        Z = geometry["Z_nodes"]
        wall_segments = cell_edge_wall_segments(R, Z, support)
        grids[f"N{nodes}"] = {
            "R_centres_m": R.tolist(),
            "Z_centres_m": Z.tolist(),
            "R_edges_m": _edges(R).tolist(),
            "Z_edges_m": _edges(Z).tolist(),
            "mom_i_abs": mom_abs.tolist(),
            "p_perp_e": p_perp.tolist(),
            "B_phi": B_phi.tolist(),
            "active_mask": support.tolist(),
            "old_wall_segments_m": wall_segments,
            "true_R_m": geometry["lcfs_R"].tolist(),
            "true_Z_m": geometry["lcfs_Z"].tolist(),
            "initial_rhs_npz_sha256": _sha256(required_paths["initial_rhs.npz"]),
        }
    return {
        "figure": {
            "time_s": 0.0,
            "time_steps": 0,
            "normalization": "raw SI formal initial RHS; no subtraction or rescaling",
            "units": {
                "mom_i_abs": "kg/(m^2*s^2)",
                "p_perp_e": "Pa/s",
                "B_phi": "T/s",
            },
            "shared_limits": {
                "mom_i_abs": [0.0, mom_max],
                "p_perp_e": [
                    -signed_max["p_perp_e"],
                    signed_max["p_perp_e"],
                ],
                "B_phi": [-signed_max["B_phi"], signed_max["B_phi"]],
            },
            "grids": grids,
        },
        "lineage": {
            "aggregate_json": str(audit_json),
            "aggregate_json_sha256": _sha256(audit_json),
            "status": str(status_path),
            "status_sha256": _sha256(status_path),
            "producer": str(producer),
            "producer_sha256": _sha256(producer),
            "implementation": str(implementation),
            "implementation_sha256": _sha256(implementation),
        },
        "inputs": inputs,
        "copies": copies,
    }
