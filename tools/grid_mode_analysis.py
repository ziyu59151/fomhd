#!/usr/bin/env python3
"""Compare production grid eigenmode candidates without rebuilding the Jacobian."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.interpolate import LinearNDInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from fomhd.core import reconstruct_magnetic
from tools.cold_hall_reference import ReferenceCase, evaluate_mode, solve_reference


CANDIDATE_NAMES = (
    "candidate_indices",
    "candidate_frequency_hz",
    "candidate_magnetic_transfer_ratio",
    "candidate_MB_magnetic_norm",
    "candidate_M0_ion_kinetic_norm",
    "candidate_velocity",
    "candidate_B_node",
    "candidate_density",
    "R",
    "Z",
    "M0",
    "background_B_node",
    "rho",
    "mu0",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_input(source: Path) -> tuple[Path, Path]:
    source = source.resolve()
    if source.is_dir():
        return source / "spectrum.npz", source / "report.json"
    if source.suffix == ".npz":
        return source, source.with_name("report.json")
    if source.name == "report.json":
        return source.with_name("spectrum.npz"), source
    raise ValueError(f"expected a spectrum directory, spectrum.npz, or report.json: {source}")


def read_npz(path: Path, names: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        selected = archive.files if names is None else names
        missing = [name for name in selected if name not in archive]
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
        return {name: np.array(archive[name], copy=True) for name in selected}


def validate_candidate_export(arrays: dict[str, np.ndarray], report: dict) -> None:
    missing = [name for name in CANDIDATE_NAMES if name not in arrays]
    if missing:
        raise ValueError(
            "legacy or incomplete spectrum export lacks "
            + ", ".join(missing)
            + "; use legacy-export for one old selected branch"
        )
    indices = np.asarray(arrays["candidate_indices"])
    frequencies = np.asarray(arrays["candidate_frequency_hz"])
    transfer_ratio = np.asarray(arrays["candidate_magnetic_transfer_ratio"])
    mb_norm = np.asarray(arrays["candidate_MB_magnetic_norm"])
    kinetic_norm = np.asarray(arrays["candidate_M0_ion_kinetic_norm"])
    velocity = np.asarray(arrays["candidate_velocity"])
    magnetic = np.asarray(arrays["candidate_B_node"])
    density = np.asarray(arrays["candidate_density"])
    R = np.asarray(arrays["R"])
    Z = np.asarray(arrays["Z"])
    M0 = np.asarray(arrays["M0"])
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError("candidate_indices must be a one-dimensional integer array")
    count = indices.size
    if frequencies.shape != (count,) or frequencies.dtype.kind not in "fc":
        raise ValueError("candidate_frequency_hz must have shape [candidate]")
    for name, value in (
        ("candidate_magnetic_transfer_ratio", transfer_ratio),
        ("candidate_MB_magnetic_norm", mb_norm),
        ("candidate_M0_ion_kinetic_norm", kinetic_norm),
    ):
        if value.shape != (count,) or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"{name} must be a finite nonnegative [candidate] array")
    if velocity.ndim != 3 or velocity.shape[0] != count or velocity.shape[1] != 3:
        raise ValueError("candidate_velocity must have shape [candidate,3,cell]")
    nc = velocity.shape[2]
    expected = {
        "candidate_B_node": (count, 3, nc),
        "candidate_density": (count, nc),
        "R": (nc,),
        "Z": (nc,),
        "M0": (nc,),
        "background_B_node": (3, nc),
    }
    for name, shape in expected.items():
        if np.asarray(arrays[name]).shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {np.asarray(arrays[name]).shape}")
    for name in ("rho", "mu0"):
        value = np.asarray(arrays[name])
        if value.ndim != 0 or not np.isfinite(value).all() or float(value) <= 0:
            raise ValueError(f"{name} must be a finite positive scalar")
    for name in ("candidate_velocity", "candidate_B_node", "candidate_density",
                 "background_B_node"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} contains nonfinite values")
    for name in ("candidate_frequency_hz", "R", "Z", "M0"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} contains nonfinite values")
    if np.any(M0 < 0) or not np.any(M0 > 0):
        raise ValueError("M0 must be nonnegative with positive total measure")
    if len(set(indices.tolist())) != count:
        raise ValueError("candidate eigen indices must be unique")
    if count and np.any(np.diff(frequencies) < 0):
        raise ValueError("candidates must be ordered by frequency")
    reported = report.get("candidates")
    if not isinstance(reported, list) or len(reported) != count:
        raise ValueError("report candidate metadata disagrees with exported candidate count")
    for j, item in enumerate(reported):
        if int(item["index"]) != int(indices[j]):
            raise ValueError("report candidate index disagrees with candidate_indices")
        if not math.isclose(float(item["frequency_hz"]), float(frequencies[j]),
                            rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("report candidate frequency disagrees with candidate_frequency_hz")
        if not math.isclose(
            float(item["magnetic_transfer_ratio"]),
            float(transfer_ratio[j]), rel_tol=1e-12, abs_tol=1e-15,
        ):
            raise ValueError("report magnetic transfer ratio disagrees with its export")
    for name in ("wall_R", "wall_Z"):
        if name in arrays and np.asarray(arrays[name]).ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")


def load_spectrum(source: Path) -> dict:
    spectrum_path, report_path = resolve_input(source)
    report = json.loads(report_path.read_text())
    arrays = read_npz(spectrum_path, CANDIDATE_NAMES + ("wall_R", "wall_Z"))
    validate_candidate_export(arrays, report)
    return {
        "source": str(source.resolve()),
        "spectrum_path": str(spectrum_path),
        "report_path": str(report_path),
        "report": report,
        "arrays": arrays,
        "hashes": {
            "spectrum_sha256": sha256_path(spectrum_path),
            "report_sha256": sha256_path(report_path),
        },
    }


def comparison_identity(report: dict) -> dict:
    identity = report.get("physical_identity")
    if not isinstance(identity, dict):
        raise ValueError("report has no physical_identity")
    required = ("species", "references", "mu0", "epsilon0", "B_min",
                "electron_inertia", "Nphi")
    missing = [name for name in required if name not in identity]
    if missing:
        raise ValueError("physical_identity lacks " + ", ".join(missing))
    return {
        "family": report.get("kind"),
        "mode": report.get("mode"),
        "species": identity["species"],
        "references": identity["references"],
        "mu0": identity["mu0"],
        "epsilon0": identity["epsilon0"],
        "B_min": identity["B_min"],
        "electron_inertia": identity["electron_inertia"],
        "Nphi": identity["Nphi"],
    }


def validate_comparable(reports: list[dict]) -> dict:
    if not reports:
        raise ValueError("at least one spectrum report is required")
    baseline = comparison_identity(reports[0])
    for report in reports[1:]:
        current = comparison_identity(report)
        for key in ("family", "mode"):
            if current[key] != baseline[key]:
                raise ValueError(f"changed comparison {key}: {current[key]!r} != {baseline[key]!r}")
        for key in ("species", "references", "mu0", "epsilon0", "B_min",
                    "electron_inertia", "Nphi"):
            if current[key] != baseline[key]:
                raise ValueError(f"changed background reference identity: {key}")
    return baseline


def _energy_inner(candidate: dict, reference: dict, M0: np.ndarray,
                  rho: float, mu0: float, mask: np.ndarray | None = None) -> dict:
    velocity = np.asarray(candidate["velocity"], dtype=complex)
    magnetic = np.asarray(candidate["B_node"], dtype=complex)
    reference_velocity = np.asarray(reference["velocity"], dtype=complex)
    reference_magnetic = np.asarray(reference["B_node"], dtype=complex)
    if mask is None:
        mask = np.ones(velocity.shape[-1], dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (velocity.shape[-1],):
        raise ValueError("structure-overlap mask must have one entry per cell")
    weight = np.asarray(M0, dtype=float)[mask]
    candidate_energy = (
        rho * np.sum(weight * np.sum(np.abs(velocity[:, mask]) ** 2, axis=0))
        + np.sum(weight * np.sum(np.abs(magnetic[:, mask]) ** 2, axis=0)) / mu0
    )
    reference_energy = (
        rho * np.sum(weight * np.sum(np.abs(reference_velocity[:, mask]) ** 2, axis=0))
        + np.sum(weight * np.sum(np.abs(reference_magnetic[:, mask]) ** 2, axis=0)) / mu0
    )
    inner = (
        rho * np.sum(weight * np.sum(velocity[:, mask].conj() * reference_velocity[:, mask], axis=0))
        + np.sum(weight * np.sum(magnetic[:, mask].conj() * reference_magnetic[:, mask], axis=0)) / mu0
    )
    denominator = math.sqrt(float(candidate_energy) * float(reference_energy))
    return {
        "candidate_structure_norm_squared": float(candidate_energy),
        "reference_structure_norm_squared": float(reference_energy),
        "inner_complex": complex(inner),
        "normalized_overlap": None if denominator == 0 else float(abs(inner) / denominator),
    }


def energy_overlap(candidate: dict, reference: dict, M0: np.ndarray,
                   rho: float, mu0: float) -> dict:
    """Return the phase-invariant nodal velocity/B structure overlap."""
    result = _energy_inner(candidate, reference, M0, rho, mu0)
    inner = result.pop("inner_complex")
    result["inner_real"] = float(inner.real)
    result["inner_imag"] = float(inner.imag)
    return result


def candidate_fields(spectrum: dict, ordinal: int) -> dict:
    arrays = spectrum["arrays"]
    return {
        "velocity": arrays["candidate_velocity"][ordinal],
        "B_node": arrays["candidate_B_node"][ordinal],
        "density": arrays["candidate_density"][ordinal],
    }


def interpolate_previous(previous: dict, R: np.ndarray, Z: np.ndarray,
                         M0: np.ndarray) -> dict:
    """Interpolate fields only inside finite active source-cell footprints."""
    previous_velocity = np.asarray(previous["velocity"], dtype=complex)
    previous_magnetic = np.asarray(previous["B_node"], dtype=complex)
    previous_density = np.asarray(previous["density"], dtype=complex)
    previous_R = np.asarray(previous["R"], dtype=float)
    previous_Z = np.asarray(previous["Z"], dtype=float)
    finite = (
        np.isfinite(previous_R) & np.isfinite(previous_Z)
        & np.all(np.isfinite(previous_velocity), axis=0)
        & np.all(np.isfinite(previous_magnetic), axis=0)
        & np.isfinite(previous_density)
    )
    points = np.column_stack((previous_R[finite], previous_Z[finite]))
    targets = np.column_stack((np.asarray(R), np.asarray(Z)))
    velocity = np.full((3, targets.shape[0]), np.nan + 0j, dtype=complex)
    magnetic = np.full((3, targets.shape[0]), np.nan + 0j, dtype=complex)
    density = np.full(targets.shape[0], np.nan + 0j, dtype=complex)
    reason = None
    if points.shape[0] < 3:
        reason = "fewer than three finite previous cell centers"
    else:
        try:
            for component in range(3):
                velocity[component] = LinearNDInterpolator(
                    points, previous_velocity[component, finite]
                )(targets)
                magnetic[component] = LinearNDInterpolator(
                    points, previous_magnetic[component, finite]
                )(targets)
            density = LinearNDInterpolator(
                points, previous_density[finite]
            )(targets)
        except Exception as error:
            reason = f"linear interpolation unavailable: {type(error).__name__}: {error}"
    mask = (
        np.isfinite(R) & np.isfinite(Z)
        & np.all(np.isfinite(velocity), axis=0)
        & np.all(np.isfinite(magnetic), axis=0)
        & np.isfinite(density)
    )
    linear_finite_mask = mask.copy()
    active_source_footprint = np.zeros(targets.shape[0], dtype=bool)
    finite_R = previous_R[finite]
    finite_Z = previous_Z[finite]
    unique_R = np.unique(finite_R)
    unique_Z = np.unique(finite_Z)
    if unique_R.size >= 2 and unique_Z.size >= 2:
        spacing_R = float(np.median(np.diff(unique_R)))
        spacing_Z = float(np.median(np.diff(unique_Z)))
        tolerance_R = 1e-10 * max(abs(spacing_R), 1.0)
        tolerance_Z = 1e-10 * max(abs(spacing_Z), 1.0)
        order_R = np.argsort(finite_R)
        order_Z = np.argsort(finite_Z)
        sorted_R = finite_R[order_R]
        sorted_Z = finite_Z[order_Z]
        active = set(zip(finite_R.tolist(), finite_Z.tolist()))
        for index, (r, z) in enumerate(targets):
            if not np.isfinite(r) or not np.isfinite(z):
                continue
            source_r = sorted_R[np.argmin(np.abs(sorted_R - r))]
            source_z = sorted_Z[np.argmin(np.abs(sorted_Z - z))]
            active_source_footprint[index] = (
                (source_r, source_z) in active
                and abs(r - source_r) <= 0.5 * spacing_R + tolerance_R
                and abs(z - source_z) <= 0.5 * spacing_Z + tolerance_Z
            )
    mask &= active_source_footprint
    if np.any(linear_finite_mask & ~active_source_footprint):
        footprint_reason = "convex-hull interpolation excluded outside active source cells"
        reason = footprint_reason if reason is None else reason + "; " + footprint_reason
    total_measure = float(np.sum(M0))
    covered_measure = float(np.sum(M0[mask]))
    return {
        "velocity": velocity,
        "B_node": magnetic,
        "density": density,
        "mask": mask,
        "linear_finite_mask": linear_finite_mask,
        "active_source_footprint": active_source_footprint,
        "covered_M0_fraction": covered_measure / total_measure if total_measure > 0 else 0.0,
        "common_cell_count": int(np.sum(mask)),
        "reason": reason,
    }


def continuation_score(previous: dict, current: dict, M0: np.ndarray,
                       rho: float, mu0: float, ordinal: int) -> dict:
    fields = candidate_fields(current, ordinal)
    interpolated = interpolate_previous(previous, current["arrays"]["R"],
                                        current["arrays"]["Z"], M0)
    score = _energy_inner(fields, interpolated, M0, rho, mu0, interpolated["mask"])
    return {
        "covered_M0_fraction": interpolated["covered_M0_fraction"],
        "common_cell_count": interpolated["common_cell_count"],
        "interpolation_available": bool(np.any(interpolated["mask"])),
        "interpolation_reason": interpolated["reason"],
        "normalized_overlap": score["normalized_overlap"],
        "common_support_structure_fraction": (
            score["candidate_structure_norm_squared"]
            / max(
                _energy_inner(fields, fields, M0, rho, mu0)[
                    "candidate_structure_norm_squared"
                ],
                np.finfo(float).tiny,
            )
        ),
        "_interpolated": interpolated,
    }


def _reference_for_spectrum(spectrum: dict, order: int,
                            reference_cache: dict) -> tuple[dict, dict]:
    report = spectrum["report"]
    if report["kind"] != "circle":
        return {}, {
            "available": False,
            "reason": "independent circular cold reference applies only to kind=circle",
        }
    identity = report["physical_identity"]
    case = ReferenceCase(mode=int(report["mode"]), order=order)
    checks = {
        "length": (identity["references"]["length"], case.a),
        "magnetic": (identity["references"]["magnetic"], case.B0),
        "density": (identity["references"]["ni"], case.density),
        "mass": (identity["species"][0]["mass"], case.mass),
        "mu0": (identity["mu0"], 1.25663706127e-6),
    }
    for name, (actual, expected) in checks.items():
        if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=0.0):
            return {}, {
                "available": False,
                "reason": f"circle background {name} differs from the standard cold reference",
            }
    cache_key = (case.mode, order)
    if cache_key not in reference_cache:
        reference_cache[cache_key] = solve_reference(case)
    reference = reference_cache[cache_key]
    fields = evaluate_mode(reference, spectrum["arrays"]["R"], spectrum["arrays"]["Z"])
    return {
        "velocity": np.asarray(fields["velocity"]).T,
        "B_node": np.asarray(fields["b"]).T,
    }, {
        "available": True,
        "model": reference["model"],
        "order": order,
        "reference_frequency_hz": reference["frequency_hz"],
        "metric": "nodal rho*M0*|v|^2 + M0*|B|^2/mu0",
    }


def _rank(candidates: list[dict], key: str) -> None:
    ordered = sorted(
        (item for item in candidates if item.get(key) is not None),
        key=lambda item: item[key],
        reverse=True,
    )
    for rank, item in enumerate(ordered, 1):
        item[f"{key}_rank"] = rank
    gap = None
    if len(ordered) > 1:
        gap = ordered[0][key] - ordered[1][key]
    if ordered:
        ordered[0][f"{key}_top_score_gap"] = gap
        ordered[0][f"{key}_ambiguous"] = (
            gap is not None and gap <= max(1e-6, 0.01 * ordered[0][key])
        )


def _top(candidates: list[dict], key: str, limit: int = 8) -> list[dict]:
    rows = sorted(
        (item for item in candidates if item.get(key) is not None),
        key=lambda item: item[key],
        reverse=True,
    )[:limit]
    return [
        {
            "eigen_index": item["eigen_index"],
            "frequency_hz": item["frequency_hz"],
            "score": item[key],
            "rank": item[f"{key}_rank"],
        }
        for item in rows
    ]


def select_candidate(spectrum: dict, index: int | None = None,
                     frequency_hz: float | None = None,
                     tolerance_hz: float = 1.0) -> tuple[int, dict]:
    arrays = spectrum["arrays"]
    if (index is None) == (frequency_hz is None):
        raise ValueError("choose exactly one candidate index or frequency")
    if index is not None:
        matches = np.flatnonzero(arrays["candidate_indices"] == int(index))
    else:
        frequency_hz = float(frequency_hz)
        if not math.isfinite(frequency_hz) or frequency_hz <= 0:
            raise ValueError("selected frequency must be finite and positive")
        difference = np.abs(arrays["candidate_frequency_hz"] - frequency_hz)
        nearest = float(np.min(difference))
        if nearest > tolerance_hz:
            raise ValueError(
                f"no candidate frequency within {tolerance_hz:g} Hz of {frequency_hz:g} Hz"
            )
        matches = np.flatnonzero(difference <= tolerance_hz)
    if len(matches) != 1:
        raise ValueError(f"candidate selection is ambiguous or absent ({len(matches)} matches)")
    ordinal = int(matches[0])
    return ordinal, {
        "eigen_index": int(arrays["candidate_indices"][ordinal]),
        "candidate_ordinal": ordinal,
        "frequency_hz": float(arrays["candidate_frequency_hz"][ordinal]),
        "magnetic_transfer_ratio": float(
            arrays["candidate_magnetic_transfer_ratio"][ordinal]
        ),
        "MB_magnetic_norm": float(arrays["candidate_MB_magnetic_norm"][ordinal]),
        "M0_ion_kinetic_norm": float(
            arrays["candidate_M0_ion_kinetic_norm"][ordinal]
        ),
    }


def selected_payload(spectrum: dict, ordinal: int) -> dict:
    arrays = spectrum["arrays"]
    report = spectrum["report"]
    fields = candidate_fields(spectrum, ordinal)
    payload = {
        "format_version": np.asarray(1, dtype=np.int64),
        "kind": np.asarray(report["kind"]),
        "nodes": np.asarray(report["nodes"], dtype=np.int64),
        "K": np.asarray(report["K"], dtype=np.int64),
        "mode": np.asarray(report["mode"], dtype=np.int64),
        "candidate_ordinal": np.asarray(ordinal, dtype=np.int64),
        "candidate_index": np.asarray(arrays["candidate_indices"][ordinal], dtype=np.int64),
        "candidate_frequency_hz": np.asarray(arrays["candidate_frequency_hz"][ordinal]),
        "magnetic_transfer_ratio": np.asarray(
            arrays["candidate_magnetic_transfer_ratio"][ordinal]
        ),
        "MB_magnetic_norm": np.asarray(
            arrays["candidate_MB_magnetic_norm"][ordinal]
        ),
        "M0_ion_kinetic_norm": np.asarray(
            arrays["candidate_M0_ion_kinetic_norm"][ordinal]
        ),
        "velocity": fields["velocity"],
        "B_node": fields["B_node"],
        "density": fields["density"],
        "R": arrays["R"],
        "Z": arrays["Z"],
        "M0": arrays["M0"],
        "background_B_node": arrays["background_B_node"],
        "rho": arrays["rho"],
        "mu0": arrays["mu0"],
        "report_sha256": np.asarray(spectrum["hashes"]["report_sha256"]),
        "spectrum_sha256": np.asarray(spectrum["hashes"]["spectrum_sha256"]),
        "comparison_identity_json": np.asarray(
            json.dumps(comparison_identity(report), sort_keys=True)
        ),
        "field_convention": np.asarray(
            "unencoded linear eigenvector physical amplitude; use Re(A) for n0 and n1"
        ),
        "metric_scope": np.asarray(
            "nodal diagnostic; distinct from production staggered-MB energy"
        ),
    }
    for name in ("wall_R", "wall_Z"):
        if name in arrays:
            payload[name] = arrays[name]
    return payload


def save_selected(path: Path, spectrum: dict, ordinal: int,
                  overwrite: bool = False) -> dict:
    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"selected output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = selected_payload(spectrum, ordinal)
    handle = tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.stem + ".", suffix=".npz", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(handle, **payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "eigen_index": int(payload["candidate_index"]),
        "frequency_hz": float(payload["candidate_frequency_hz"]),
    }


def load_selected(path: Path) -> dict:
    path = path.resolve()
    arrays = read_npz(path)
    required = (
        "kind", "nodes", "mode", "candidate_index", "candidate_frequency_hz",
        "magnetic_transfer_ratio", "MB_magnetic_norm", "M0_ion_kinetic_norm",
        "velocity", "B_node", "density", "R", "Z", "M0",
        "background_B_node", "rho", "mu0", "report_sha256", "spectrum_sha256",
        "comparison_identity_json",
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError("selected branch lacks " + ", ".join(missing))
    velocity = arrays["velocity"]
    magnetic = arrays["B_node"]
    if velocity.ndim != 2 or velocity.shape[0] != 3 or magnetic.shape != velocity.shape:
        raise ValueError("selected velocity/B_node must have shape [3,cell]")
    nc = velocity.shape[1]
    for name in ("density", "R", "Z", "M0"):
        if arrays[name].shape != (nc,):
            raise ValueError(f"selected {name} must have shape [cell]")
    if arrays["background_B_node"].shape != (3, nc):
        raise ValueError("selected background_B_node must have shape [3,cell]")
    for name in ("velocity", "B_node", "density", "background_B_node"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"selected {name} contains nonfinite values")
    for name in ("magnetic_transfer_ratio", "MB_magnetic_norm",
                 "M0_ion_kinetic_norm"):
        if np.asarray(arrays[name]).ndim != 0 or not np.isfinite(arrays[name]):
            raise ValueError(f"selected {name} must be a finite scalar")
    return {
        **arrays,
        "path": str(path),
        "sha256": sha256_path(path),
    }


def validate_previous(selected: dict, spectra: list[dict]) -> None:
    if not spectra:
        return
    try:
        identity = json.loads(str(selected["comparison_identity_json"]))
    except json.JSONDecodeError as error:
        raise ValueError("selected branch comparison identity is invalid JSON") from error
    synthetic = {
        "kind": str(selected["kind"]),
        "mode": int(selected["mode"]),
        "physical_identity": {
            "species": identity["species"],
            "references": identity["references"],
            "mu0": identity["mu0"],
            "epsilon0": identity["epsilon0"],
            "B_min": identity["B_min"],
            "electron_inertia": identity["electron_inertia"],
            "Nphi": identity["Nphi"],
        },
    }
    validate_comparable([spectra[0]["report"], synthetic])
    if not math.isclose(float(selected["rho"]), spectra[0]["arrays"]["rho"],
                        rel_tol=1e-14, abs_tol=0):
        raise ValueError("selected branch rho differs from current spectra")
    if not math.isclose(float(selected["mu0"]), spectra[0]["arrays"]["mu0"],
                        rel_tol=1e-14, abs_tol=0):
        raise ValueError("selected branch mu0 differs from current spectra")


def _selected_previous(selected: dict) -> dict:
    return {
        "velocity": selected["velocity"],
        "B_node": selected["B_node"],
        "density": selected["density"],
        "R": selected["R"],
        "Z": selected["Z"],
        "M0": selected["M0"],
    }


def analyze(sources: list[Path], reference_order: int = 8,
            previous_path: Path | None = None, select_grid: int | None = None,
            select_index: int | None = None,
            select_frequency_hz: float | None = None,
            frequency_tolerance_hz: float = 1.0,
            output: Path | None = None, save_selected_path: Path | None = None,
            overwrite: bool = False) -> dict:
    if not 2 <= reference_order <= 12:
        raise ValueError("reference order must be in [2,12]")
    if frequency_tolerance_hz <= 0:
        raise ValueError("frequency tolerance must be positive")
    if previous_path is not None and (
        select_index is not None or select_frequency_hz is not None
    ):
        raise ValueError("use either a previous selected branch or a new selection, not both")
    spectra = [load_spectrum(source) for source in sources]
    spectra.sort(key=lambda item: (int(item["report"]["nodes"]), item["source"]))
    identity = validate_comparable([item["report"] for item in spectra])
    selected = load_selected(previous_path) if previous_path is not None else None
    previous = None
    selection = None
    if selected is not None:
        validate_previous(selected, spectra)
        previous = _selected_previous(selected)
        selection = {
            "source": "previous_selected",
            "path": str(Path(previous_path).resolve()),
            "sha256": selected["sha256"],
            "eigen_index": int(selected["candidate_index"]),
            "frequency_hz": float(selected["candidate_frequency_hz"]),
        }
    elif select_grid is not None or select_index is not None or select_frequency_hz is not None:
        if select_index is None and select_frequency_hz is None:
            raise ValueError("selection needs a candidate index or frequency")
        choices = [item for item in spectra if int(item["report"]["nodes"]) == select_grid]
        if select_grid is None and len(spectra) == 1:
            choices = spectra
        if len(choices) != 1:
            raise ValueError("--select-grid must identify exactly one loaded spectrum")
        ordinal, choice = select_candidate(
            choices[0], index=select_index, frequency_hz=select_frequency_hz,
            tolerance_hz=frequency_tolerance_hz,
        )
        options = {
            "vector": None,
            "R": choices[0]["arrays"]["R"],
            "Z": choices[0]["arrays"]["Z"],
            "M0": choices[0]["arrays"]["M0"],
            **candidate_fields(choices[0], ordinal),
        }
        previous = options
        selection = {
            "source": "current_grid",
            "spectrum": choices[0]["source"],
            "nodes": int(choices[0]["report"]["nodes"]),
            **choice,
        }
        if save_selected_path is not None:
            selection["saved"] = save_selected(
                save_selected_path, choices[0], ordinal, overwrite
            )
    elif save_selected_path is not None:
        raise ValueError("--save-selected requires a candidate selection")
    reference_cache = {}
    grids = []
    for spectrum in spectra:
        report = spectrum["report"]
        reference, reference_report = _reference_for_spectrum(
            spectrum, reference_order, reference_cache
        )
        candidates = []
        for ordinal, (index, frequency) in enumerate(zip(
            spectrum["arrays"]["candidate_indices"],
            spectrum["arrays"]["candidate_frequency_hz"],
        )):
            fields = candidate_fields(spectrum, ordinal)
            row = {
                "candidate_ordinal": ordinal,
                "eigen_index": int(index),
                "frequency_hz": float(frequency),
                "frequency_mhz": float(frequency) / 1e6,
                "magnetic_transfer_ratio": float(
                    spectrum["arrays"]["candidate_magnetic_transfer_ratio"][ordinal]
                ),
                "MB_magnetic_norm": float(
                    spectrum["arrays"]["candidate_MB_magnetic_norm"][ordinal]
                ),
                "M0_ion_kinetic_norm": float(
                    spectrum["arrays"]["candidate_M0_ion_kinetic_norm"][ordinal]
                ),
            }
            if reference:
                row.update(energy_overlap(
                    fields, reference, spectrum["arrays"]["M0"],
                    float(spectrum["arrays"]["rho"]),
                    float(spectrum["arrays"]["mu0"]),
                ))
            if previous is not None:
                branch_frequency = selection["frequency_hz"]
                row["frequency_ratio_to_branch"] = (
                    float(frequency) / branch_frequency if branch_frequency else None
                )
                row["frequency_delta_hz"] = float(frequency) - branch_frequency
                continuation = continuation_score(
                    previous, spectrum, spectrum["arrays"]["M0"],
                    float(spectrum["arrays"]["rho"]),
                    float(spectrum["arrays"]["mu0"]), ordinal,
                )
                continuation.pop("_interpolated")
                row["continuation"] = continuation
                row["continuation_overlap"] = continuation["normalized_overlap"]
            if reference_report.get("available"):
                row["frequency_ratio_to_cold_reference"] = (
                    float(frequency) / reference_report["reference_frequency_hz"]
                )
            candidates.append(row)
        _rank(candidates, "normalized_overlap")
        _rank(candidates, "continuation_overlap")
        continuation_top = _top(candidates, "continuation_overlap")
        cold_top = _top(candidates, "normalized_overlap")
        grids.append({
            "source": spectrum["source"],
            "spectrum_path": spectrum["spectrum_path"],
            "report_path": spectrum["report_path"],
            "kind": report["kind"],
            "nodes": report["nodes"],
            "K": report["K"],
            "mode": report["mode"],
            "K_user_selected": report.get("K_user_selected"),
            "candidate_count": len(candidates),
            "hashes": spectrum["hashes"],
            "independent_cold_reference": reference_report,
            "candidates": candidates,
            "top_continuation_candidates": continuation_top,
            "top_cold_reference_candidates": cold_top,
            "branch_assertion": "not asserted; parent must inspect frequency and overlap evidence",
        })
    result = {
        "schema": "fomhd_grid_mode_analysis_v1",
        "scope": (
            "host-only nodal diagnostic of exported production eigenvectors; "
            "not an evolution, convergence-order fit, or same-branch assertion"
        ),
        "metric": (
            "rho*M0*|v|^2 + M0*|B_node|^2/mu0 on actual cells; "
            "phase-invariant structure metric only, not physical total wave energy; "
            "candidate magnetic_transfer_ratio exposes the nodal-vs-staggered-MB difference"
        ),
        "field_convention": (
            "candidate and selected fields are unencoded physical eigenvector "
            "amplitudes; structural displays use Re(A) for both n0 and n1"
        ),
        "ambiguity_threshold": "top-score gap <= max(1e-6, 0.01*top score)",
        "overlap_meaning": (
            "normalized structure comparison; do not interpret as energy conservation "
            "or physical total wave-energy agreement"
        ),
        "comparison_identity": identity,
        "selection": selection,
        "grids": grids,
    }
    if output is not None:
        output = output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        report_path = output / "grid_mode_analysis.json"
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(report_path)
        result["analysis_path"] = str(report_path)
        result["analysis_sha256"] = sha256_path(report_path)
    return result


def export_legacy(source: Path, destination: Path,
                  frequency_hz: float | None = None, index: int | None = None,
                  memory_limit_bytes: int = 24 << 30,
                  equilibrium: Path | None = None,
                  overwrite: bool = False) -> dict:
    """Export one saved old branch without rebuilding its Jacobian."""
    spectrum_path, report_path = resolve_input(source)
    report = json.loads(report_path.read_text())
    with np.load(spectrum_path, allow_pickle=False) as archive:
        files = set(archive.files)
        if all(name in files for name in CANDIDATE_NAMES):
            raise ValueError("spectrum already contains candidate exports")
    has_eigenvectors = "eigenvectors" in files
    names = ["velocity_basis", "magnetic_basis"]
    if has_eigenvectors:
        names.extend(("eigenvectors", "background_B"))
    else:
        names.extend(("seed_n", "seed_mom", "seed_B"))
    available = read_npz(
        spectrum_path, tuple(names),
    )
    reported = report.get("candidates", [])
    if index is None:
        if frequency_hz is None:
            raise ValueError("legacy export needs an eigen index or frequency")
        matches = [item for item in reported if abs(float(item["frequency_hz"]) - frequency_hz) <= 1.0]
        if len(matches) != 1:
            raise ValueError("legacy frequency does not identify exactly one reported candidate")
        index = int(matches[0]["index"])
    matches = [item for item in reported if int(item["index"]) == int(index)]
    if len(matches) != 1:
        raise ValueError("legacy eigen index is absent from the report candidates")
    from tools.production_spectrum import direction
    from tools.warp_tokamak_cae import host, make_context, physical_identity

    context, context_meta, context_eq = make_context(
        report["kind"], int(report["nodes"]), int(report["K"]),
        "cpu", equilibrium, memory_limit_bytes=memory_limit_bytes,
    )
    current_identity = physical_identity(context)
    if "physical_identity" in report and report["physical_identity"] != current_identity:
        raise ValueError("legacy context physical identity differs from its report")
    if "physical_identity" not in report:
        report = dict(report)
        report["physical_identity"] = current_identity
    nc = context.op.Nc
    if has_eigenvectors:
        fields = direction(
            available["eigenvectors"][:, int(index)], context,
            available["velocity_basis"], available["magnetic_basis"],
        )
        nb = context.op.Nb
        background_field = available["background_B"][
            int(report["K"]) * nb : (int(report["K"]) + 1) * nb
        ]
    else:
        selected_report = report.get("selected")
        if not isinstance(selected_report, dict) or int(selected_report["index"]) != int(index):
            raise ValueError("legacy file has eigenvectors only for its reported selected seed")
        from tools.warp_tokamak_cae import background
        fields = {
            "n": available["seed_n"],
            "mom": available["seed_mom"],
            "B": available["seed_B"],
        }
        initial, _, _ = background(context, context_meta, context_eq)
        background_field = host(initial.B, context)[
            int(report["K"]) * context.op.Nb:
            (int(report["K"]) + 1) * context.op.Nb
        ]
    bnode = reconstruct_magnetic(context.op, fields["B"]).reshape(3, nc)
    nodal_magnetic = float(np.sum(np.abs(bnode) ** 2 * host(context.op.M0, context)))
    staggered_magnetic = float(np.sum(
        np.abs(fields["B"]) ** 2 * np.asarray(context.op.MB).real
    ))
    magnetic_norm = float(
        np.sum(np.abs(fields["B"]) ** 2 * np.asarray(context.op.MB).real)
        / context.p.mu0
    )
    rho = context.p.species[0].mass * context.p.ref.ni
    kinetic_norm = float(
        np.sum(np.abs(fields["mom"].reshape(3, nc)) ** 2 * context.op.M0) / rho
    )
    spectrum = {
        "source": str(Path(source).resolve()),
        "report": report,
        "arrays": {
            "candidate_indices": np.asarray([index], dtype=np.int64),
            "candidate_frequency_hz": np.asarray([float(matches[0]["frequency_hz"])]),
            "candidate_magnetic_transfer_ratio": np.asarray([
                nodal_magnetic / max(staggered_magnetic, 1e-300)
            ]),
            "candidate_MB_magnetic_norm": np.asarray([magnetic_norm]),
            "candidate_M0_ion_kinetic_norm": np.asarray([kinetic_norm]),
            "candidate_velocity": (fields["mom"].reshape(3, nc) / (
                context.p.species[0].mass * context.p.ref.ni
            ))[None],
            "candidate_B_node": bnode[None],
            "candidate_density": fields["n"].reshape(1, nc),
            "R": host(context.geometry.cells.R, context).reshape(nc),
            "Z": host(context.geometry.cells.Z, context).reshape(nc),
            "M0": host(context.op.M0, context).real.reshape(nc),
            "background_B_node": reconstruct_magnetic(context.op, background_field).reshape(3, nc),
            "rho": np.asarray(rho),
            "mu0": np.asarray(context.p.mu0),
            "wall_R": np.asarray([point.R for point in context.geometry.polygon]),
            "wall_Z": np.asarray([point.Z for point in context.geometry.polygon]),
        },
        "hashes": {
            "spectrum_sha256": sha256_path(spectrum_path),
            "report_sha256": sha256_path(report_path),
        },
    }
    return save_selected(destination, spectrum, 0, overwrite)


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _frequency_arguments(parser: argparse.ArgumentParser, prefix: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{prefix}-frequency-hz", type=float)
    group.add_argument(f"--{prefix}-frequency-mhz", type=float)


def _selected_frequency(args: argparse.Namespace, prefix: str) -> float | None:
    hz = getattr(args, f"{prefix}_frequency_hz")
    mhz = getattr(args, f"{prefix}_frequency_mhz")
    return hz if hz is not None else None if mhz is None else 1e6 * mhz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analysis = commands.add_parser("analyze", help="score and optionally select exported candidates")
    analysis.add_argument("spectra", nargs="+", type=Path)
    analysis.add_argument("--output", type=Path)
    analysis.add_argument("--previous", type=Path)
    analysis.add_argument("--select-grid", type=int)
    analysis.add_argument("--select-index", type=int)
    _frequency_arguments(analysis, "select")
    analysis.add_argument("--frequency-tolerance-hz", type=float, default=1.0)
    analysis.add_argument("--save-selected", type=Path)
    analysis.add_argument("--reference-order", type=int, default=8)
    analysis.add_argument("--overwrite", action="store_true")
    legacy = commands.add_parser("legacy-export", help="export one branch from an old spectrum")
    legacy.add_argument("spectrum", type=Path)
    legacy.add_argument("--output", type=Path, required=True)
    legacy.add_argument("--select-index", type=int)
    _frequency_arguments(legacy, "select")
    legacy.add_argument("--equilibrium", type=Path)
    legacy.add_argument("--memory-gib", type=positive_int, default=24)
    legacy.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze":
        result = analyze(
            args.spectra,
            reference_order=args.reference_order,
            previous_path=args.previous,
            select_grid=args.select_grid,
            select_index=args.select_index,
            select_frequency_hz=_selected_frequency(args, "select"),
            frequency_tolerance_hz=args.frequency_tolerance_hz,
            output=args.output,
            save_selected_path=args.save_selected,
            overwrite=args.overwrite,
        )
    else:
        frequency = _selected_frequency(args, "select")
        if (args.select_index is None) == (frequency is None):
            parser.error("legacy-export requires exactly one index or frequency")
        result = export_legacy(
            args.spectrum, args.output, frequency_hz=frequency,
            index=args.select_index, memory_limit_bytes=args.memory_gib << 30,
            equilibrium=args.equilibrium, overwrite=args.overwrite,
        )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
