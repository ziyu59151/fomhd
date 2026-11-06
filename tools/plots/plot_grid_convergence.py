#!/usr/bin/env python3
"""Plot selected grid-mode convergence and piecewise-constant cell structure."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from tools.grid_mode_analysis import (
    comparison_identity,
    interpolate_previous,
    load_selected,
    sha256_path,
    validate_comparable,
)


def read_npz(path: Path,
             names: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        selected = archive.files if names is None else names
        missing = [name for name in selected if name not in archive]
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
        return {name: np.array(archive[name], copy=True) for name in selected}


def load_branch(report_path: Path, selected_path: Path) -> dict:
    report_path = report_path.resolve()
    selected_path = selected_path.resolve()
    report = json.loads(report_path.read_text())
    selected = load_selected(selected_path)
    report_hash = sha256_path(report_path)
    if str(selected["report_sha256"]) != report_hash:
        raise ValueError("selected branch does not name the supplied report bytes")
    if str(selected["kind"]) != str(report["kind"]) or int(selected["mode"]) != int(report["mode"]):
        raise ValueError("selected branch family/mode differs from its report")
    identity = report["physical_identity"]
    rho = float(identity["species"][0]["mass"]) * float(identity["references"]["ni"])
    if not math.isclose(float(selected["rho"]), rho, rel_tol=1e-13, abs_tol=0.0):
        raise ValueError("selected branch rho differs from its report reference")
    if not math.isclose(float(selected["mu0"]), float(identity["mu0"]),
                        rel_tol=1e-13, abs_tol=0.0):
        raise ValueError("selected branch mu0 differs from its report")
    candidates = report.get("candidates", [])
    matches = [
        item for item in candidates
        if int(item["index"]) == int(selected["candidate_index"])
    ]
    if len(matches) != 1 or not math.isclose(
        float(matches[0]["frequency_hz"]),
        float(selected["candidate_frequency_hz"]),
        rel_tol=1e-12, abs_tol=1e-9,
    ):
        raise ValueError("selected branch candidate is absent from its report")
    return {
        "report_path": str(report_path),
        "selected_path": str(selected_path),
        "report": report,
        "selected": selected,
        "nodes": int(report["nodes"]),
        "mode": int(report["mode"]),
        "frequency_hz": float(selected["candidate_frequency_hz"]),
        "hashes": {
            "report_sha256": report_hash,
            "selected_sha256": selected["sha256"],
        },
    }


def structural_real(field: np.ndarray) -> np.ndarray:
    """Return the unencoded eigenvector real amplitude for n0 or n1."""
    return np.real(np.asarray(field))


def physical_at_phi(coefficient: np.ndarray, mode: int, phi: float = 0.0) -> np.ndarray:
    """Reconstruct an encoded physical state coefficient at physical phi."""
    if mode not in (0, 1):
        raise ValueError("only modes 0 and 1 are retained")
    if mode == 0:
        return np.real(coefficient)
    return 2.0 * np.real(coefficient * np.exp(1j * mode * float(phi)))


def mode_coefficient(values: np.ndarray, mode: int, axis: int = 0) -> np.ndarray:
    """Extract one retained coefficient from five physical phi samples."""
    values = np.asarray(values)
    axis = int(axis)
    if axis < 0:
        axis += values.ndim
    if values.ndim < 1 or not 0 <= axis < values.ndim or values.shape[axis] != 5:
        raise ValueError("five retained Fourier samples are required")
    phases = np.exp(-1j * mode * np.arange(5) * 2 * np.pi / 5) / 5
    return np.tensordot(values, phases, axes=([axis], [0]))


def flat_cell_grid(R: np.ndarray, Z: np.ndarray,
                   values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place cell values on their actual rectangular grid with inactive cells masked."""
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    values = np.asarray(values, dtype=float)
    if R.ndim != 1 or Z.shape != R.shape or values.shape != R.shape:
        raise ValueError("cell R, Z, and values must have one entry per active cell")
    if not np.isfinite(R).all() or not np.isfinite(Z).all():
        raise ValueError("cell coordinates must be finite")
    unique_R = np.unique(R)
    unique_Z = np.unique(Z)
    if unique_R.size < 2 or unique_Z.size < 2:
        raise ValueError("flat-cell display needs a two-dimensional cell grid")
    r_index = {value: index for index, value in enumerate(unique_R)}
    z_index = {value: index for index, value in enumerate(unique_Z)}
    grid = np.full((unique_Z.size, unique_R.size), np.nan)
    occupied = set()
    for r, z, value in zip(R, Z, values):
        key = (z_index[z], r_index[r])
        if key in occupied:
            raise ValueError("duplicate active cell center in flat-cell display")
        occupied.add(key)
        grid[key] = value
    dR = np.diff(unique_R)
    dZ = np.diff(unique_Z)
    if np.any(dR <= 0) or np.any(dZ <= 0):
        raise ValueError("cell coordinate grids must be strictly increasing")
    edge_R = np.concatenate((
        [unique_R[0] - 0.5 * np.median(dR)],
        unique_R[:-1] + 0.5 * dR,
        [unique_R[-1] + 0.5 * np.median(dR)],
    ))
    edge_Z = np.concatenate((
        [unique_Z[0] - 0.5 * np.median(dZ)],
        unique_Z[:-1] + 0.5 * dZ,
        [unique_Z[-1] + 0.5 * np.median(dZ)],
    ))
    return edge_R, edge_Z, grid


def signed_structure_delta(previous: dict, current: dict,
                           current_fields: dict) -> dict:
    """Return signed common-support deltas of unencoded eigenvector amplitudes."""
    interpolated = interpolate_previous(
        previous, current["R"], current["Z"], current["M0"]
    )
    mask = interpolated["mask"]
    rho = float(current["rho"])
    mu0 = float(current["mu0"])
    previous_joint = _weighted_vector(
        interpolated["B_node"][:, mask],
        interpolated["velocity"][:, mask],
        np.asarray(current["M0"])[mask],
        rho,
        mu0,
    )
    current_joint = _weighted_vector(
        current_fields["B_node"][:, mask],
        current_fields["velocity"][:, mask],
        np.asarray(current["M0"])[mask],
        rho,
        mu0,
    )
    previous_norm_squared = np.vdot(previous_joint, previous_joint)
    alignment_factor = (
        np.vdot(previous_joint, current_joint) / previous_norm_squared
        if previous_norm_squared > 0
        else 1.0 + 0.0j
    )
    delta_velocity = np.full_like(current_fields["velocity"], np.nan + 0j)
    delta_magnetic = np.full_like(current_fields["B_node"], np.nan + 0j)
    delta_density = np.full(current_fields["density"].shape, np.nan + 0j)
    delta_velocity[:, mask] = (
        current_fields["velocity"][:, mask]
        - alignment_factor * interpolated["velocity"][:, mask]
    )
    delta_magnetic[:, mask] = (
        current_fields["B_node"][:, mask]
        - alignment_factor * interpolated["B_node"][:, mask]
    )
    delta_density[mask] = (
        current_fields["density"][mask]
        - alignment_factor * interpolated["density"][mask]
    )
    background = np.asarray(current["background_B_node"], dtype=complex)
    magnitude = np.linalg.norm(background, axis=0)
    bhat = np.zeros_like(background)
    nonzero = magnitude > 0
    bhat[:, nonzero] = background[:, nonzero] / magnitude[nonzero]
    delta_B_parallel = np.sum(
        structural_real(delta_magnetic) * bhat.real, axis=0
    )
    delta_B_parallel[~mask] = np.nan
    velocity_real = structural_real(delta_velocity)
    density_real = structural_real(delta_density)
    density_real[~mask] = np.nan
    return {
        "mask": mask,
        "covered_M0_fraction": interpolated["covered_M0_fraction"],
        "interpolation_reason": interpolated["reason"],
        "alignment_factor": complex(alignment_factor),
        "alignment_factor_real": float(np.real(alignment_factor)),
        "alignment_factor_imag": float(np.imag(alignment_factor)),
        "alignment_factor_abs": float(abs(alignment_factor)),
        "delta_B_parallel": delta_B_parallel,
        "delta_n": density_real,
        "delta_vR": velocity_real[0],
        "delta_vZ": velocity_real[2],
        "delta_velocity": velocity_real,
    }


def _wall_outline(axis, selected: dict) -> None:
    if "wall_R" not in selected or "wall_Z" not in selected:
        axis.text(0.99, 0.02, "wall unavailable", transform=axis.transAxes,
                  ha="right", va="bottom", fontsize=7)
        return
    wall_R = np.asarray(selected["wall_R"])
    wall_Z = np.asarray(selected["wall_Z"])
    axis.plot(
        np.append(wall_R, wall_R[0]), np.append(wall_Z, wall_Z[0]),
        color="black", linewidth=1.3, zorder=6,
    )


def _flat_panel(axis, R: np.ndarray, Z: np.ndarray, normalized: np.ndarray,
                selected: dict, title: str, colorbar_label: str,
                velocity: np.ndarray | None = None, *,
                signed: bool = True, cmap: str = "RdBu_r") -> None:
    edge_R, edge_Z, grid = flat_cell_grid(R, Z, normalized)
    finite = np.isfinite(normalized)
    if np.any(finite) and signed:
        bound = max(1.0, float(np.nanmax(np.abs(normalized))))
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    elif np.any(finite):
        norm = Normalize(
            vmin=0.0, vmax=max(1.0, float(np.nanmax(normalized)))
        )
    else:
        norm = (
            Normalize(vmin=-1.0, vmax=1.0)
            if signed else Normalize(vmin=0.0, vmax=1.0)
        )
    mesh = axis.pcolormesh(
        edge_R, edge_Z, np.ma.masked_invalid(grid),
        cmap=cmap, norm=norm, shading="flat", rasterized=False,
    )
    _wall_outline(axis, selected)
    if velocity is not None:
        active = np.isfinite(normalized)
        u = np.asarray(velocity[0], dtype=float)[active]
        w = np.asarray(velocity[2], dtype=float)[active]
        speed = np.hypot(u, w)
        shown = speed > 0
        if np.any(shown):
            axis.quiver(
                R[active][shown], Z[active][shown],
                u[shown] / speed[shown], w[shown] / speed[shown],
                angles="xy", pivot="middle", color="0.12",
                width=0.004, headwidth=3.3, headlength=4.1, zorder=7,
            )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.set_title(title, fontsize=9)
    colorbar = axis.figure.colorbar(mesh, ax=axis, fraction=0.046, pad=0.035)
    colorbar.set_label(colorbar_label, fontsize=7)
    colorbar.ax.tick_params(labelsize=6)


def _save_figure(fig, output: Path, stem: str) -> list[str]:
    paths = [output / f"{stem}.png", output / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=170, facecolor="white", bbox_inches="tight")
    fig.savefig(paths[1], facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return [str(path) for path in paths]


def render_convergence(branches: list[dict], output: Path) -> dict:
    branches = sorted(branches, key=lambda item: item["nodes"])
    if len({item["nodes"] for item in branches}) != len(branches):
        raise ValueError("selected branches require unique numerical grid counts")
    mode = branches[0]["mode"]
    validate_comparable([item["report"] for item in branches])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    nodes = np.asarray([item["nodes"] for item in branches])
    frequencies = np.asarray([item["frequency_hz"] / 1e6 for item in branches])
    axes[0].plot(nodes, frequencies, marker="o", color="0.15")
    for x, y in zip(nodes, frequencies):
        axes[0].annotate(f"n={int(x)}", (x, y), xytext=(0, 7),
                         textcoords="offset points", ha="center", fontsize=7)
    axes[0].set_xlabel("numerical grid n (cells across)")
    axes[0].set_ylabel("selected frequency [MHz]")
    axes[0].set_title(f"Linear branch frequency, nphi={mode}")
    axes[0].grid(alpha=0.25)
    pair_nodes = []
    overlaps = []
    coverage = []
    for previous, current in zip(branches[:-1], branches[1:]):
        previous_fields = {
            "velocity": previous["selected"]["velocity"],
            "B_node": previous["selected"]["B_node"],
            "density": previous["selected"]["density"],
            "R": previous["selected"]["R"],
            "Z": previous["selected"]["Z"],
            "M0": previous["selected"]["M0"],
        }
        current_fields = {
            "velocity": current["selected"]["velocity"],
            "B_node": current["selected"]["B_node"],
            "density": current["selected"]["density"],
        }
        interpolated = interpolate_previous(
            previous_fields, current["selected"]["R"],
            current["selected"]["Z"], current["selected"]["M0"],
        )
        from tools.grid_mode_analysis import _energy_inner
        score = _energy_inner(
            current_fields, interpolated, current["selected"]["M0"],
            float(current["selected"]["rho"]), float(current["selected"]["mu0"]),
            interpolated["mask"],
        )
        pair_nodes.append(current["nodes"])
        overlaps.append(np.nan if score["normalized_overlap"] is None else score["normalized_overlap"])
        coverage.append(interpolated["covered_M0_fraction"])
    if pair_nodes:
        axes[1].plot(pair_nodes, overlaps, marker="o", color="#1f77b4")
        axes[2].plot(pair_nodes, coverage, marker="s", color="#d95f02")
    else:
        axes[1].text(0.5, 0.5, "needs at least two grids", ha="center",
                     va="center", transform=axes[1].transAxes)
        axes[2].text(0.5, 0.5, "needs at least two grids", ha="center",
                     va="center", transform=axes[2].transAxes)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("numerical grid n")
    axes[1].set_ylabel("common-support normalized overlap")
    axes[1].set_title("Continuation overlap")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xlabel("numerical grid n")
    axes[2].set_ylabel("covered M0 fraction")
    axes[2].set_title("Common finite support")
    fig.suptitle(
        "Selected grid branches; no convergence order inferred from these points",
        fontsize=11,
    )
    fig.tight_layout()
    return {"files": _save_figure(fig, output, "grid_mode_convergence")}


def selected_structure_fields(selected: dict, mode: int = 0) -> dict:
    """Return absolute selected structure at declared linear phases."""
    velocity_complex = np.asarray(selected["velocity"], dtype=complex)
    magnetic_complex = np.asarray(selected["B_node"], dtype=complex)
    density_complex = np.asarray(selected["density"], dtype=complex)
    if mode == 0:
        if np.max(np.abs(magnetic_complex)) > 0:
            source = magnetic_complex
        else:
            source = density_complex
        dominant = int(np.argmax(np.abs(source)))
        compression_phase = (
            -float(np.angle(source.reshape(-1)[dominant]))
            if np.abs(source.reshape(-1)[dominant]) > 0 else 0.0
        )
        compression_factor = np.exp(1j * compression_phase)
        poloidal = velocity_complex[[0, 2]].reshape(-1)
        dominant_velocity = int(np.argmax(np.abs(poloidal)))
        velocity_phase = (
            -float(np.angle(poloidal[dominant_velocity]))
            if np.abs(poloidal[dominant_velocity]) > 0 else 0.0
        )
        velocity_factor = np.exp(1j * velocity_phase)
    else:
        compression_phase = 0.0
        velocity_phase = 0.0
        compression_factor = 1.0 + 0.0j
        velocity_factor = 1.0 + 0.0j
    velocity_snapshot = np.real(velocity_complex * velocity_factor)
    compression_velocity = np.real(velocity_complex * compression_factor)
    magnetic = np.real(magnetic_complex * compression_factor)
    density = np.real(density_complex * compression_factor)
    background = np.asarray(selected["background_B_node"], dtype=complex)
    magnitude = np.linalg.norm(background, axis=0)
    bhat = np.zeros_like(background)
    nonzero = magnitude > 0
    bhat[:, nonzero] = background[:, nonzero] / magnitude[nonzero]
    return {
        "B_parallel": np.sum(magnetic * bhat.real, axis=0),
        "density": density,
        "speed": np.linalg.norm(velocity_snapshot, axis=0),
        "velocity": velocity_snapshot,
        "velocity_snapshot": velocity_snapshot,
        "compression_velocity": compression_velocity,
        "compression_phase_rad": compression_phase,
        "velocity_phase_rad": velocity_phase,
        "compression_factor_real": float(np.real(compression_factor)),
        "compression_factor_imag": float(np.imag(compression_factor)),
        "velocity_factor_real": float(np.real(velocity_factor)),
        "velocity_factor_imag": float(np.imag(velocity_factor)),
    }


def render_selected_structure(branches: list[dict], output: Path) -> dict:
    """Render absolute selected eigenmode structure for every supplied grid."""
    branches = sorted(branches, key=lambda item: item["nodes"])
    fields = [
        selected_structure_fields(item["selected"], item["mode"])
        for item in branches
    ]
    keys = ("B_parallel", "density", "speed")
    scales = {
        key: max(
            (float(np.nanmax(np.abs(item[key])))
             for item in fields if np.any(np.isfinite(item[key]))),
            default=0.0,
        )
        for key in keys
    }
    fig, axes = plt.subplots(
        len(branches), 3, figsize=(11.5, 3.5 * len(branches)), squeeze=False
    )
    for row, (branch, item) in enumerate(zip(branches, fields)):
        for column, key in enumerate(keys):
            scale = scales[key]
            normalized = item[key] / scale if scale > 0 else item[key] * 0.0
            if key == "speed":
                phase_title = (
                    f", vphase={item['velocity_phase_rad']:.4g} rad"
                    if branch["mode"] == 0 else ", phi=0"
                )
                _flat_panel(
                    axes[row, column], branch["selected"]["R"],
                    branch["selected"]["Z"], normalized,
                    branch["selected"],
                    f"absolute selected n={branch['nodes']}: |v|{phase_title}",
                    "normalized |v| at declared velocity phase",
                    velocity=item["velocity"], signed=False, cmap="viridis",
                )
            else:
                if branch["mode"] == 0:
                    phase_title = (
                        f", comp_phase={item['compression_phase_rad']:.4g} rad"
                    )
                    label = (
                        "normalized B_parallel at compression phase"
                        if key == "B_parallel"
                        else "normalized density at compression phase"
                    )
                else:
                    phase_title = ", phi=0"
                    label = (
                        "normalized Re(A_B) . b0"
                        if key == "B_parallel" else "normalized Re(A_n)"
                    )
                _flat_panel(
                    axes[row, column], branch["selected"]["R"],
                    branch["selected"]["Z"], normalized,
                    branch["selected"],
                    f"absolute selected n={branch['nodes']}: {key}{phase_title}",
                    label,
                )
    if branches[0]["mode"] == 0:
        phase_note = (
            "n0 per-grid compression and global velocity quadrature phases "
            "are declared in panel labels and manifest"
        )
    else:
        phase_note = "n1 selected eigenvector amplitudes at phi=0; Re(A)"
    fig.suptitle(
        "Absolute selected linear discrete eigenmode; not time evolved\n"
        f"{phase_note}; flat actual cells; scales common across grids",
        fontsize=11,
    )
    fig.tight_layout()
    return {
        "files": _save_figure(fig, output, "grid_mode_selected_structure"),
        "phases": [
            {
                "nodes": branch["nodes"],
                "compression_phase_rad": item["compression_phase_rad"],
                "velocity_phase_rad": item["velocity_phase_rad"],
            }
            for branch, item in zip(branches, fields)
        ],
    }


def render_structure(branches: list[dict], output: Path) -> dict:
    branches = sorted(branches, key=lambda item: item["nodes"])
    selected_render = render_selected_structure(branches, output)
    if len(branches) < 2:
        return {
            "selected_files": selected_render["files"],
            "selected_structure_phases": selected_render["phases"],
            "delta_files": [],
            "delta_reason": "needs at least two grids for aligned differences",
        }
    deltas = []
    for previous, current in zip(branches[:-1], branches[1:]):
        previous_fields = {
            "velocity": previous["selected"]["velocity"],
            "B_node": previous["selected"]["B_node"],
            "density": previous["selected"]["density"],
            "R": previous["selected"]["R"],
            "Z": previous["selected"]["Z"],
            "M0": previous["selected"]["M0"],
        }
        current_fields = {
            "velocity": current["selected"]["velocity"],
            "B_node": current["selected"]["B_node"],
            "density": current["selected"]["density"],
        }
        delta = signed_structure_delta(
            previous_fields, current["selected"], current_fields
        )
        deltas.append(delta)
    keys = ("delta_B_parallel", "delta_n", "delta_vR", "delta_vZ")
    scales = {
        key: max(
            (float(np.nanmax(np.abs(delta[key])))
             for delta in deltas if np.any(np.isfinite(delta[key]))),
            default=0.0,
        )
        for key in keys
    }
    fig, axes = plt.subplots(
        len(deltas), 4, figsize=(14.5, 3.6 * len(deltas)), squeeze=False
    )
    labels = {
        "delta_B_parallel": "normalized delta B_parallel",
        "delta_n": "normalized delta n",
        "delta_vR": "normalized delta vR",
        "delta_vZ": "normalized delta vZ",
    }
    for row, (previous, current, delta) in enumerate(
        zip(branches[:-1], branches[1:], deltas)
    ):
        selected = current["selected"]
        for column, key in enumerate(keys):
            scale = scales[key]
            normalized = (
                delta[key] / scale if scale > 0 and np.isfinite(scale)
                else delta[key] * 0.0
            )
            normalized[~delta["mask"]] = np.nan
            _flat_panel(
                axes[row, column], selected["R"], selected["Z"], normalized,
                selected,
                f"{current['nodes']} grid: {key}",
                labels[key],
                velocity=delta["delta_velocity"] if column >= 2 else None,
            )
        axes[row, 0].text(
            0.01, 0.01,
            f"coverage={delta['covered_M0_fraction']:.3f}",
            transform=axes[row, 0].transAxes, fontsize=7,
            ha="left", va="bottom", color="0.15",
        )
    fig.suptitle(
        "Jointly aligned linear discrete eigenmode: signed common-support delta\n"
        "one weighted B/v factor applied to B, v, and density; flat actual cells",
        fontsize=11,
    )
    fig.tight_layout()
    return {
        "selected_files": selected_render["files"],
        "selected_structure_phases": selected_render["phases"],
        "delta_files": _save_figure(fig, output, "grid_mode_structure_deltas"),
        "alignment_factors": [
            {
                "previous_nodes": previous["nodes"],
                "current_nodes": current["nodes"],
                "real": delta["alignment_factor_real"],
                "imag": delta["alignment_factor_imag"],
                "magnitude": delta["alignment_factor_abs"],
                "covered_M0_fraction": delta["covered_M0_fraction"],
            }
            for previous, current, delta in zip(
                branches[:-1], branches[1:], deltas
            )
        ],
    }


def load_response(path: Path) -> dict:
    path = path.resolve()
    run = path.parent
    siblings = {
        "case": run / "case.toml",
        "report": run / "report.json",
        "input_mapping": run / "input_mapping.npz",
    }
    missing_files = [str(item) for item in siblings.values() if not item.is_file()]
    if missing_files:
        raise ValueError(
            "matched-response provenance files missing: " + ", ".join(missing_files)
        )
    arrays = read_npz(path)
    required = ("time_s", "delta_B", "delta_n", "delta_velocity", "parallel_B")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError("matched response lacks " + ", ".join(missing))
    return {
        "path": str(path),
        "arrays": arrays,
        "sha256": sha256_path(path),
        "provenance": {
            "case": tomllib.loads(siblings["case"].read_text()),
            "report": json.loads(siblings["report"].read_text()),
            "input_mapping": read_npz(siblings["input_mapping"], ("R", "Z")),
            "hashes": {
                "case_sha256": sha256_path(siblings["case"]),
                "report_sha256": sha256_path(siblings["report"]),
                "input_mapping_sha256": sha256_path(siblings["input_mapping"]),
            },
            "paths": {name: str(value) for name, value in siblings.items()},
        },
    }


def validate_response_provenance(response: dict, branch: dict) -> dict:
    """Match a response run to the selected production spectrum and cells."""
    provenance = response.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("matched-response provenance is absent")
    case = provenance.get("case")
    report = provenance.get("report")
    mapping = provenance.get("input_mapping")
    if not isinstance(case, dict) or not isinstance(report, dict):
        raise ValueError("matched-response provenance metadata is malformed")
    if not isinstance(mapping, dict) or "R" not in mapping or "Z" not in mapping:
        raise ValueError("matched-response provenance mapping lacks R/Z")
    selected = branch["selected"]
    report_case = report.get("case")
    if not isinstance(report_case, dict):
        raise ValueError("matched-response report provenance lacks case metadata")
    metadata = {"case": case, "report case": report_case}
    for label, values in metadata.items():
        for key in ("kind", "nodes", "K"):
            if key not in values:
                raise ValueError(
                    f"matched-response provenance metadata missing {label}.{key}"
                )
    if "nphi" not in case:
        raise ValueError("matched-response provenance metadata missing case.nphi")
    expected_kind = str(selected["kind"])
    expected_mode = int(branch["mode"])
    expected_nodes = int(branch["nodes"])
    expected_K = int(selected["K"])
    checks = {
        "case family": (str(case.get("kind")), expected_kind),
        "report family": (str(report_case.get("kind")), expected_kind),
        "case harmonic": (int(case.get("nphi")), expected_mode),
        "case grid": (int(case.get("nodes")), expected_nodes),
        "report grid": (int(report_case.get("nodes")), expected_nodes),
        "case K": (int(case.get("K")), expected_K),
        "report K": (int(report_case.get("K")), expected_K),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"matched-response provenance mismatch in {label}: "
                f"{actual!r} != {expected!r}"
            )
    for name in ("R", "Z"):
        actual = np.asarray(mapping[name], dtype=float)
        expected = np.asarray(selected[name], dtype=float)
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, rtol=0, atol=1e-12
        ):
            raise ValueError(
                f"matched-response provenance mismatch in {name} cell coordinates"
            )
    selected_frequency = report.get("selected_frequency_hz")
    if selected_frequency is None or not math.isclose(
        float(selected_frequency),
        float(selected["candidate_frequency_hz"]),
        rel_tol=1e-12,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "matched-response provenance mismatch in selected seed frequency"
        )
    spectra = report.get("spectra")
    if not isinstance(spectra, list):
        raise ValueError("matched-response report provenance lacks spectra")
    selected_report_hash = str(selected["report_sha256"])
    selected_spectrum_hash = str(selected["spectrum_sha256"])
    matching_sources = [
        item for item in spectra
        if str(item.get("report_sha256")) == selected_report_hash
        and str(item.get("spectrum_sha256")) == selected_spectrum_hash
        and int(item.get("mode", -1)) == expected_mode
    ]
    if not matching_sources:
        raise ValueError(
            "matched-response provenance mismatch in selected source spectrum hashes"
        )
    validated = {
        "family": expected_kind,
        "harmonic": expected_mode,
        "grid": expected_nodes,
        "K": expected_K,
        "selected_frequency_hz": float(selected_frequency),
        "selected_report_sha256": selected_report_hash,
        "selected_spectrum_sha256": selected_spectrum_hash,
    }
    provenance["validated"] = validated
    return validated


def _weighted_vector(B: np.ndarray, velocity: np.ndarray,
                     M0: np.ndarray, rho: float, mu0: float) -> np.ndarray:
    B = np.asarray(B, dtype=complex)
    velocity = np.asarray(velocity, dtype=complex)
    M0 = np.asarray(M0, dtype=float)
    if B.shape != velocity.shape or B.shape != (3, M0.size):
        raise ValueError("weighted B and velocity require shape [3,cell]")
    root_M0 = np.sqrt(M0)
    return np.concatenate((
        root_M0[None, :] * B / math.sqrt(mu0),
        math.sqrt(rho) * root_M0[None, :] * velocity,
    ), axis=1).reshape(-1)


def _projection(coherence_values: np.ndarray, seed: np.ndarray,
                mode: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Project weighted actual vectors onto a real n0 span or complex n1 line."""
    coherence = np.full(coherence_values.shape[0], np.nan)
    residual = np.full(coherence_values.shape[0], np.nan)
    if mode == 0:
        columns = np.column_stack((np.real(seed), np.imag(seed)))
        _, singular, right = np.linalg.svd(columns, full_matrices=False)
        if singular.size == 0 or singular[0] == 0:
            return coherence, residual, 0
        rank = int(np.sum(singular > 1e-12 * singular[0]))
        basis = columns @ right[:rank].conj().T
        q, _ = np.linalg.qr(basis, mode="reduced")
        for index, value in enumerate(coherence_values):
            actual = np.real(value)
            norm = np.linalg.norm(actual)
            if norm == 0 or not np.isfinite(norm):
                continue
            projected = q @ (q.T @ actual)
            coherence[index] = np.linalg.norm(projected) / norm
            residual[index] = np.linalg.norm(actual - projected) / norm
        return coherence, residual, rank
    norm_seed = np.linalg.norm(seed)
    if norm_seed == 0:
        return coherence, residual, 0
    for index, value in enumerate(coherence_values):
        norm = np.linalg.norm(value)
        if norm == 0 or not np.isfinite(norm):
            continue
        projected = seed * (np.vdot(seed, value) / np.vdot(seed, seed))
        coherence[index] = abs(np.vdot(seed, value)) / (norm_seed * norm)
        residual[index] = np.linalg.norm(value - projected) / norm
    return coherence, residual, 1


def _compression_extrema(series: np.ndarray) -> tuple[list[int], int, int]:
    series = np.asarray(series, dtype=float)
    finite = np.flatnonzero(np.isfinite(series))
    if finite.size == 0:
        return [], -1, -1
    minimum = int(finite[np.argmin(series[finite])])
    maximum = int(finite[np.argmax(series[finite])])
    if finite.size < 3:
        return sorted(set((minimum, maximum))), minimum, maximum
    values = series[finite]
    derivative = np.diff(values)
    extrema = []
    for index in range(1, derivative.size):
        if derivative[index - 1] * derivative[index] < 0:
            extrema.append(int(finite[index]))
    if not extrema:
        extrema = sorted(set((minimum, maximum)))
    return extrema, extrema[0], extrema[-1]


def response_seed_coherence(response: dict, branch: dict) -> dict:
    """Measure actual evolved fields outside the declared seed subspace."""
    arrays = response["arrays"]
    times = np.asarray(arrays["time_s"], dtype=float)
    B_coefficient = np.asarray(arrays["delta_B"], dtype=complex)
    density_samples = np.asarray(arrays["delta_n"], dtype=complex)
    velocity_samples = np.asarray(arrays["delta_velocity"], dtype=complex)
    selected = branch["selected"]
    M0 = np.asarray(selected["M0"], dtype=float)
    rho = float(selected["rho"])
    mu0 = float(selected["mu0"])
    nc = M0.size
    if B_coefficient.shape != (times.size, 3, nc):
        raise ValueError("response delta_B must have shape [time,3,cell]")
    if density_samples.shape != (times.size, 5, nc):
        raise ValueError("response delta_n must have shape [time,5,cell]")
    if velocity_samples.shape != (times.size, 5, 3, nc):
        raise ValueError("response delta_velocity must have shape [time,5,3,cell]")
    velocity_coefficient = mode_coefficient(
        velocity_samples, branch["mode"], axis=1
    )
    seed = _weighted_vector(
        selected["B_node"], selected["velocity"], M0, rho, mu0
    )
    actual = np.stack([
        _weighted_vector(B_coefficient[index], velocity_coefficient[index],
                         M0, rho, mu0)
        for index in range(times.size)
    ])
    coherence, residual, rank = _projection(actual, seed, branch["mode"])
    density_coefficient = mode_coefficient(
        density_samples, branch["mode"], axis=1
    )
    density_phi0 = physical_at_phi(density_coefficient, branch["mode"])
    seed_density = np.abs(np.asarray(selected["density"]))
    compression_cell = int(np.argmax(seed_density))
    compression = density_phi0[:, compression_cell]
    extrema, early_extremum, late_extremum = _compression_extrema(compression)
    finite = np.isfinite(coherence) & np.isfinite(residual)
    summary = {
        "available": bool(np.any(finite)),
        "mode_rule": (
            "real span(Re seed, Im seed)" if branch["mode"] == 0
            else "phase-invariant complex seed line"
        ),
        "seed_subspace_rank": rank,
        "sample_count": int(np.sum(finite)),
        "minimum_coherence": (
            float(np.min(coherence[finite])) if np.any(finite) else None
        ),
        "mean_coherence": (
            float(np.mean(coherence[finite])) if np.any(finite) else None
        ),
        "maximum_residual_fraction": (
            float(np.max(residual[finite])) if np.any(finite) else None
        ),
        "compression_cell": compression_cell,
        "compression_extremum_indices": extrema,
        "early_compression_extremum_index": early_extremum,
        "late_compression_extremum_index": late_extremum,
        "background_caveat": (
            "matched-control subtraction removes the common background evolution, "
            "but this diagnostic does not prove absence of residual background modes"
        ),
        "numerical_filter_caveat": (
            "no temporal or spatial projection/filter is applied; residual outside "
            "the seed span includes numerical-filter and unresolved-mode content"
        ),
    }
    return {
        "time_s": times,
        "coherence": coherence,
        "residual_fraction": residual,
        "compression_phi0": compression,
        "summary": summary,
    }


def render_seed_coherence(result: dict, output: Path) -> dict:
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    axes[0].plot(result["time_s"], result["coherence"], color="#1f77b4")
    axes[0].set_ylabel("seed-span coherence")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[1].plot(result["time_s"], result["residual_fraction"], color="#d95f02")
    axes[1].set_ylabel("residual outside seed span")
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel("time [s]")
    axes[1].grid(alpha=0.25)
    summary = result["summary"]
    for index in (
        summary["early_compression_extremum_index"],
        summary["late_compression_extremum_index"],
    ):
        if index >= 0:
            axes[0].axvline(result["time_s"][index], color="0.55",
                            linewidth=0.8, linestyle="--")
            axes[1].axvline(result["time_s"][index], color="0.55",
                            linewidth=0.8, linestyle="--")
    note = (
        f"{summary['mode_rule']}; rank={summary['seed_subspace_rank']}\n"
        "weighted nodal [sqrt(M0) B/sqrt(mu0), sqrt(M0 rho) v]\n"
        "matched-control and numerical-filter caveats retained"
    )
    axes[0].text(0.01, 0.02, note, transform=axes[0].transAxes,
                 fontsize=7.5, va="bottom", ha="left")
    fig.suptitle("Actual evolved seed-subspace coherence", fontsize=11)
    fig.tight_layout()
    files = _save_figure(fig, output, "response_seed_coherence")
    data_path = output / "response_seed_coherence.npz"
    np.savez_compressed(
        data_path,
        time_s=result["time_s"],
        coherence=result["coherence"],
        residual_fraction=result["residual_fraction"],
        compression_phi0=result["compression_phi0"],
        mode_rule=np.asarray(summary["mode_rule"]),
        seed_subspace_rank=np.asarray(summary["seed_subspace_rank"]),
    )
    return {
        "files": files,
        "data_path": str(data_path),
        "data_sha256": sha256_path(data_path),
        "summary": summary,
    }


def response_fields(response: dict, branch: dict, time_index: int) -> dict:
    arrays = response["arrays"]
    times = np.asarray(arrays["time_s"], dtype=float)
    if times.ndim != 1 or times.size == 0 or not np.isfinite(times).all():
        raise ValueError("response time_s must be finite and nonempty")
    index = int(time_index)
    if index < 0:
        index += times.size
    if not 0 <= index < times.size:
        raise ValueError("response time index is outside the saved history")
    mode = branch["mode"]
    nc = np.asarray(branch["selected"]["R"]).size
    density_samples = np.asarray(arrays["delta_n"])
    velocity_samples = np.asarray(arrays["delta_velocity"])
    parallel = np.asarray(arrays["parallel_B"])
    if density_samples.ndim != 3 or density_samples.shape[1:] != (5, nc):
        raise ValueError("response delta_n must have shape [time,5,cell]")
    if velocity_samples.ndim != 4 or velocity_samples.shape[1:] != (5, 3, nc):
        raise ValueError("response delta_velocity must have shape [time,5,3,cell]")
    if parallel.shape != times.shape + (nc,):
        raise ValueError("response parallel_B must have shape [time,cell]")
    density = physical_at_phi(mode_coefficient(density_samples[index], mode), mode)
    velocity = physical_at_phi(
        mode_coefficient(velocity_samples[index], mode), mode
    )
    magnetic_parallel = physical_at_phi(parallel[index], mode)
    return {
        "time_s": float(times[index]),
        "delta_B_parallel": magnetic_parallel,
        "delta_n": density,
        "delta_vR": velocity[0],
        "delta_vZ": velocity[2],
        "delta_velocity": velocity,
    }


def render_response(response: dict | None, branches: list[dict], output: Path,
                    response_grid: int | None, time_index: int) -> dict:
    if response is None:
        return {"files": [], "reason": "no matched response supplied"}
    nc = np.asarray(response["arrays"]["parallel_B"]).shape[-1]
    choices = [
        branch for branch in branches
        if np.asarray(branch["selected"]["R"]).size == nc
        and (response_grid is None or branch["nodes"] == response_grid)
    ]
    if len(choices) != 1:
        raise ValueError("response grid must identify exactly one selected branch")
    branch = choices[0]
    validate_response_provenance(response, branch)
    fields = response_fields(response, branch, time_index)
    keys = ("delta_B_parallel", "delta_n", "delta_vR", "delta_vZ")
    scales = {
        key: max(float(np.nanmax(np.abs(fields[key]))), 0.0)
        for key in keys
    }
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.7))
    labels = {
        "delta_B_parallel": "normalized physical delta B_parallel",
        "delta_n": "normalized physical delta n",
        "delta_vR": "normalized physical delta vR",
        "delta_vZ": "normalized physical delta vZ",
    }
    for column, key in enumerate(keys):
        scale = scales[key]
        normalized = fields[key] / scale if scale > 0 else fields[key] * 0.0
        _flat_panel(
            axes[column], branch["selected"]["R"], branch["selected"]["Z"],
            normalized, branch["selected"],
            f"response: {key}", labels[key],
            velocity=fields["delta_velocity"] if column >= 2 else None,
        )
    fig.suptitle(
        f"Matched-control time-evolved response at t={fields['time_s']:.6g} s, "
        f"physical phi=0, nphi={branch['mode']}\n"
        "encoded state convention: n0=Re(A), n1 physical=2Re(A_retained); "
        "piecewise-constant actual cells",
        fontsize=11,
    )
    fig.tight_layout()
    coherence = response_seed_coherence(response, branch)
    return {
        "files": _save_figure(fig, output, "matched_response_structure"),
        "seed_coherence": render_seed_coherence(coherence, output),
    }


def render(branch_pairs: list[tuple[Path, Path]], output: Path,
           response_path: Path | None = None, response_grid: int | None = None,
           response_time_index: int = -1) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    branches = [load_branch(report, selected) for report, selected in branch_pairs]
    if not branches:
        raise ValueError("at least one report/selected-field branch is required")
    validate_comparable([item["report"] for item in branches])
    identities = []
    for branch in branches:
        expected = comparison_identity(branch["report"])
        embedded = json.loads(str(branch["selected"]["comparison_identity_json"]))
        if embedded != expected:
            raise ValueError("selected branch physical identity differs from its report")
        identities.append(expected)
    response = load_response(response_path) if response_path is not None else None
    figures = {
        "convergence": render_convergence(branches, output),
        "structure": render_structure(branches, output),
        "response": render_response(
            response, branches, output, response_grid, response_time_index
        ),
    }
    manifest = {
        "schema": "fomhd_grid_convergence_plots_v1",
        "scope": (
            "selected linear discrete eigenmodes; piecewise-constant actual-cell "
            "maps; no smoothing and no convergence-order inference"
        ),
        "mode": branches[0]["mode"],
        "comparison_identity": identities[0],
        "field_convention": (
            "selected eigenvector maps use Re(A) for n0 and n1; "
            "matched response uses encoded physical phi reconstruction"
        ),
        "overlap_meaning": (
            "structure comparison only; not physical total wave energy"
        ),
        "branches": [
            {
                "nodes": item["nodes"],
                "frequency_hz": item["frequency_hz"],
                "magnetic_transfer_ratio": float(
                    item["selected"]["magnetic_transfer_ratio"]
                ),
                "MB_magnetic_norm": float(item["selected"]["MB_magnetic_norm"]),
                "M0_ion_kinetic_norm": float(
                    item["selected"]["M0_ion_kinetic_norm"]
                ),
                "report_path": item["report_path"],
                "selected_path": item["selected_path"],
                **item["hashes"],
            }
            for item in branches
        ],
        "response": None if response is None else {
            "path": response["path"],
            "sha256": response["sha256"],
            "time_index": int(response_time_index),
            "grid": response_grid,
            "provenance": {
                "paths": response["provenance"]["paths"],
                "hashes": response["provenance"]["hashes"],
                "validated": response["provenance"]["validated"],
            },
        },
        "figures": figures,
    }
    manifest_path = output / "input_hashes.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_path(manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--branch", action="append", nargs=2, metavar=("REPORT", "SELECTED"),
        required=True, help="explicit report.json and selected-branch NPZ pair",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--response-grid", type=int)
    parser.add_argument("--response-time-index", type=int, default=-1)
    args = parser.parse_args(argv)
    branch_pairs = [(Path(report), Path(selected)) for report, selected in args.branch]
    result = render(
        branch_pairs, args.output, response_path=args.response,
        response_grid=args.response_grid,
        response_time_index=args.response_time_index,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
