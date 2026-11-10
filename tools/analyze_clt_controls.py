#!/usr/bin/env python3
"""Analyze saved CLT control runs without rerunning solver evolution."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "clt_control_analysis_v1"
FORCE_KEYS = {
    "pressure": "frame_ion_pressure_force_samples",
    "magnetic": "frame_ion_magnetic_force_samples",
    "electric": "frame_ion_electric_force_samples",
    "convection": "frame_ion_convection_samples",
    "wall_reaction": "frame_ion_wall_reaction_samples",
    "net": "frame_ion_net_force_samples",
}
FRAME_CONTRACTS = {
    "frame_ion_theta": (),
    "frame_ion_strain": (),
    "frame_electron_theta": (),
    "frame_electron_strain": (),
    "frame_ion_pdot_perp": (),
    "frame_ion_pdot_parallel": (),
    "frame_electron_pdot_perp": (),
    "frame_electron_pdot_parallel": (),
    "frame_ion_gradient": (3, 3),
    "frame_electron_gradient": (3, 3),
    "frame_ion_convection": (3,),
    "frame_electron_convection": (3,),
}
GROUPS = {
    "B": ("B_N_samples", 3, "T"),
    "ui": ("ion_velocity_samples", 3, "m/s"),
    "ni": ("ion_density_samples", None, "1/m^3"),
    "pi": ("ion_perp_pressure_samples", None, "Pa"),
    "peperp": ("electron_perp_pressure_samples", None, "Pa"),
    "peparallel": ("electron_parallel_pressure_samples", None, "Pa"),
}
COMPONENTS = ("R", "phi", "Z")


class AnalysisError(RuntimeError):
    """Stable saved-input or output-contract failure."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisError(f"{label} must contain a JSON object")
    return value


def _load_npz(path: Path, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError) as error:
        raise AnalysisError(f"cannot read {label} {path}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as error:
        raise AnalysisError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _array(
    arrays: dict[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    if name not in arrays:
        raise AnalysisError(f"{label} is missing explicit array {name}")
    value = np.asarray(arrays[name])
    if value.shape != shape:
        raise AnalysisError(
            f"{label} {name} shape {value.shape} != {shape}"
        )
    if not np.isfinite(value).all():
        raise AnalysisError(f"{label} {name} contains non-finite values")
    return value


def _positive_finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"{name} must be numeric") from error
    if not math.isfinite(result) or result <= 0.0:
        raise AnalysisError(f"{name} must be finite and positive")
    return result


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise AnalysisError(f"{name} must be a non-negative integer")
    return int(value)


def _regions(
    distances: np.ndarray, width: float
) -> dict[str, np.ndarray]:
    return {
        "whole": np.ones(distances.shape, dtype=bool),
        "wall_band": distances <= width,
        "interior": distances > width,
    }


def _metrics(
    values: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(values)[..., mask]
    selected_weights = weights[mask]
    count = int(np.count_nonzero(mask))
    total_weight = int(selected.shape[0]) * float(np.sum(selected_weights))
    volume = 2.0 * math.pi * float(np.sum(selected_weights))
    result: dict[str, Any] = {
        "available": bool(count and total_weight > 0.0),
        "count": count,
        "volume_m3": volume,
        "weighted_rms": None,
        "weighted_l2": None,
        "linf": None,
    }
    if not result["available"]:
        return result
    magnitude_squared = np.sum(np.abs(selected) ** 2, axis=0)
    weighted_squared = float(
        np.sum(selected_weights[None, :] * magnitude_squared)
    )
    result["weighted_l2"] = math.sqrt(
        2.0
        * math.pi
        / float(selected.shape[0])
        * weighted_squared
    )
    result["weighted_rms"] = math.sqrt(weighted_squared / total_weight)
    result["linf"] = float(np.max(np.abs(selected), initial=0.0))
    result["linf_definition"] = (
        "maximum absolute Cartesian component; not vector norm"
        if selected.ndim > 2
        else "maximum absolute value"
    )
    return result


def _composition_metrics(
    values: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    value = np.asarray(values)
    result = _metrics(value, weights, mask)
    if value.ndim < 2 or value.shape[-1] != weights.size:
        raise AnalysisError("diagnostic array does not end in the cell axis")
    component_count = int(np.prod(value.shape[1:-1]))
    if component_count <= 1:
        return result
    shaped = value.reshape(
        value.shape[0], component_count, value.shape[-1]
    )
    result["components"] = [
        _metrics(shaped[:, index, :], weights, mask)
        for index in range(component_count)
    ]
    return result


def _vector_metrics(
    values: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    result = _metrics(values, weights, mask)
    result["components"] = {
        component: _metrics(values[:, index, :], weights, mask)
        for index, component in enumerate(COMPONENTS)
    }
    return result


def _parse_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays = _load_npz(path, "snapshot")
    if "metadata_json" not in arrays:
        raise AnalysisError(f"snapshot {path} lacks metadata_json")
    try:
        metadata = json.loads(str(arrays["metadata_json"].item()))
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise AnalysisError(
            f"snapshot {path} metadata_json is invalid: {error}"
        ) from error
    if not isinstance(metadata, dict):
        raise AnalysisError(f"snapshot {path} metadata must be an object")
    time_s = metadata.get("time_s")
    try:
        time_value = float(time_s)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"snapshot {path} has no finite time_s") from error
    if not math.isfinite(time_value):
        raise AnalysisError(f"snapshot {path} has non-finite time_s")
    metadata["time_s"] = time_value
    return metadata, arrays


def _snapshot_paths(run: Path, report: dict[str, Any]) -> list[Path]:
    names = report.get("snapshots")
    if isinstance(names, list) and names:
        path = run / "snapshots"
        result = [path / str(name) for name in names]
    else:
        result = sorted((run / "snapshots").glob("*.npz"))
    missing = [str(path) for path in result if not path.is_file()]
    if missing:
        raise AnalysisError("missing snapshot files: " + ", ".join(missing))
    if not result:
        raise AnalysisError("run has no snapshots")
    return result


def _ordered_snapshots(
    paths: list[Path],
) -> list[tuple[Path, dict[str, Any], dict[str, np.ndarray]]]:
    loaded = []
    for path in paths:
        metadata, arrays = _parse_snapshot(path)
        loaded.append((path, metadata, arrays))
    loaded.sort(
        key=lambda item: (
            float(item[1]["time_s"]),
            int(item[1].get("step", -1)),
        )
    )
    return loaded


def _sample_values(
    arrays: dict[str, np.ndarray],
    *,
    key: str,
    vector: bool,
    nphi: int,
    nc: int,
    label: str,
) -> np.ndarray:
    shape = (nphi, 3, nc) if vector else (nphi, nc)
    return _array(arrays, key, shape, label)


def _group_drift(
    snapshots: list[tuple[Path, dict[str, Any], dict[str, np.ndarray]]],
    *,
    nphi: int,
    nc: int,
    weights: np.ndarray,
    region_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    initial = snapshots[0]
    final = snapshots[-1]
    initial_values = {
        group: _sample_values(
            initial[2],
            key=key,
            vector=vector == 3,
            nphi=nphi,
            nc=nc,
            label=f"true initial {group}",
        )
        for group, (key, vector, _unit) in GROUPS.items()
    }
    final_values = {
        group: _sample_values(
            final[2],
            key=key,
            vector=vector == 3,
            nphi=nphi,
            nc=nc,
            label=f"final {group}",
        )
        for group, (key, vector, _unit) in GROUPS.items()
    }
    drift_regions: dict[str, Any] = {}
    for region_name, mask in region_masks.items():
        groups: dict[str, Any] = {}
        for group, (_key, vector, unit) in GROUPS.items():
            delta = final_values[group] - initial_values[group]
            if vector == 3:
                metric = _vector_metrics(delta, weights, mask)
                background = _vector_metrics(initial_values[group], weights, mask)
            else:
                metric = _metrics(delta, weights, mask)
                background = _metrics(initial_values[group], weights, mask)
            metric["unit"] = unit
            metric["background"] = background
            groups[group] = metric
        drift_regions[region_name] = {
            "count": int(np.count_nonzero(mask)),
            "volume_m3": 2.0 * math.pi * float(np.sum(weights[mask])),
            "groups": groups,
        }

    snapshot_maxima = []
    for path, metadata, arrays in snapshots[1:]:
        group_maxima = {}
        snapshot_maximum = 0.0
        for group, (key, vector, _unit) in GROUPS.items():
            values = _sample_values(
                arrays,
                key=key,
                vector=vector == 3,
                nphi=nphi,
                nc=nc,
                label=f"snapshot {group}",
            )
            value = float(
                np.max(np.abs(values - initial_values[group]), initial=0.0)
            )
            group_maxima[group] = value
            snapshot_maximum = max(snapshot_maximum, value)
        snapshot_maxima.append(
            {
                "file": path.name,
                "step": int(metadata.get("step", -1)),
                "time_s": float(metadata["time_s"]),
                "maximum_abs": None,
                "maximum_abs_reason": (
                    "mixed units across groups; use per-group maxima"
                ),
                "groups": group_maxima,
            }
        )

    final_group_maxima = {}
    for group, (_key, vector, _unit) in GROUPS.items():
        final_group_maxima[group] = float(
            np.max(
                np.abs(final_values[group] - initial_values[group]),
                initial=0.0,
            )
        )
    return {
        "available": True,
        "initial": {
            "file": initial[0].name,
            "step": int(initial[1].get("step", -1)),
            "time_s": float(initial[1]["time_s"]),
        },
        "final": {
            "file": final[0].name,
            "step": int(final[1].get("step", -1)),
            "time_s": float(final[1]["time_s"]),
            "maximum_abs": None,
            "maximum_abs_reason": (
                "mixed units across groups; use per-group maxima"
            ),
            "groups": final_group_maxima,
        },
        "maximum_abs": None,
        "maximum_abs_reason": (
            "mixed units across groups; use per-group maxima"
        ),
        "snapshots": snapshot_maxima,
        "regions": drift_regions,
    }


def _seed_scale(report: dict[str, Any]) -> dict[str, Any]:
    parameters = report.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    init = report.get("init")
    if not isinstance(init, dict):
        init = {}
    perturbation = init.get("perturbation")
    if not isinstance(perturbation, dict):
        perturbation = {}
    amplitude = parameters.get("amplitude", perturbation.get("amplitude"))
    seed = parameters.get("seed", perturbation.get("seed"))
    scalar = perturbation.get("scalar_pressure_perturbation")
    vA = perturbation.get("vA_m_s", parameters.get("vA_m_s"))
    result: dict[str, Any] = {
        "seed": seed,
        "amplitude_A": amplitude,
        "velocity_scale_vA_m_s": vA,
        "scalar_pressure_perturbation": scalar,
        "unit": None,
        "value": None,
        "available": False,
        "source": "report.parameters/report.init.perturbation",
    }
    if scalar is not None:
        scalar_value = float(scalar)
        if not math.isfinite(scalar_value):
            raise AnalysisError("scalar pressure perturbation is non-finite")
        result["scalar_pressure_perturbation"] = scalar_value
    if amplitude is not None:
        value = float(amplitude)
        if not math.isfinite(value) or value < 0.0:
            raise AnalysisError("seed amplitude must be finite and nonnegative")
        if value == 0.0:
            result.update(
                kind="A_times_vA",
                value=0.0,
                unit="m/s",
                available=True,
                reason="zero amplitude gives an exactly zero seed scale",
            )
        elif vA is not None:
            velocity = float(vA)
            if not math.isfinite(velocity) or velocity < 0.0:
                raise AnalysisError("vA must be finite and nonnegative")
            result.update(
                kind="A_times_vA",
                value=value * velocity,
                unit="m/s",
                available=True,
            )
        else:
            result["reason"] = (
                "nonzero amplitude is persisted but vA is not; no scale was guessed"
            )
    elif scalar is not None:
        result.update(
            kind="scalar_pressure_perturbation",
            value=scalar_value,
            unit="Pa",
            available=True,
        )
    else:
        result["reason"] = "seed amplitude and scalar perturbation are both absent"
    return result


def _force_region_offsets(
    forces: dict[str, np.ndarray],
) -> tuple[dict[str, Any], np.ndarray]:
    components = {}
    for name, key in FORCE_KEYS.items():
        value = forces[name]
        components[name] = {
            "array_key": key,
            "shape": list(value.shape),
            "finite_count": int(np.count_nonzero(np.isfinite(value))),
            "nonzero_count": int(np.count_nonzero(np.abs(value) > 0.0)),
            "absolute_max": float(np.max(np.abs(value), initial=0.0)),
        }
    defect = (
        forces["net"]
        - (
            -forces["convection"]
            + forces["pressure"]
            + forces["magnetic"]
            + forces["electric"]
            + forces["wall_reaction"]
        )
    )
    return components, np.asarray(defect)


def _initial_rhs(
    run: Path,
    *,
    nphi: int,
    nc: int,
    weights: np.ndarray,
    region_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    path = run / "initial-rhs.npz"
    result: dict[str, Any] = {
        "file": str(path),
        "exists": path.is_file(),
        "forces": {
            "available": False,
            "status": "not_supplied",
            "missing_keys": [],
            "components": {},
            "sum_defect": {"available": False},
            "note": (
                "No force was reconstructed; a nonzero net force is not "
                "called physical unbalance without wall reaction."
            ),
        },
        "frame_diagnostics": {},
    }
    if not path.is_file():
        result["forces"]["missing_keys"] = list(FORCE_KEYS.values())
        result["forces"]["note"] = (
            "initial-rhs.npz is absent; no force component was zero-filled"
        )
        for name in FRAME_CONTRACTS:
            result["frame_diagnostics"][name] = {
                "available": False,
                "reason": "initial-rhs.npz absent",
            }
        return result

    arrays = _load_npz(path, "initial RHS")
    missing = [key for key in FORCE_KEYS.values() if key not in arrays]
    result["forces"]["missing_keys"] = missing
    if not missing:
        forces = {
            name: _array(
                arrays,
                key,
                (nphi, 3, nc),
                "initial force cache",
            )
            for name, key in FORCE_KEYS.items()
        }
        components, defect = _force_region_offsets(forces)
        for name, component in components.items():
            component["regions"] = {
                region: _vector_metrics(forces[name], weights, mask)
                for region, mask in region_masks.items()
            }
        result["forces"].update(
            available=True,
            status="available",
            components=components,
        )
        result["forces"]["sum_defect"] = {
            "available": True,
            "formula": (
                "net - (-convection + pressure + magnetic + electric + "
                "wall_reaction)"
            ),
            "regions": {
                region: _vector_metrics(defect, weights, mask)
                for region, mask in region_masks.items()
            },
        }
    else:
        result["forces"]["status"] = "partial"
        result["forces"]["components"] = {
            name: {
                "array_key": key,
                "available": key in arrays,
            }
            for name, key in FORCE_KEYS.items()
        }

    for name, components in FRAME_CONTRACTS.items():
        if name not in arrays:
            result["frame_diagnostics"][name] = {
                "available": False,
                "reason": "key absent",
            }
            continue
        value = np.asarray(arrays[name])
        expected = (nphi, *components, nc)
        if value.shape != expected:
            if value.ndim == 1 and value.size == int(np.prod(expected)):
                value = value.reshape(expected)
            else:
                result["frame_diagnostics"][name] = {
                    "available": False,
                    "reason": (
                        f"shape {value.shape} does not match named contract "
                        f"{expected}"
                    ),
                }
                continue
        if not np.isfinite(value).all():
            result["frame_diagnostics"][name] = {
                "available": False,
                "reason": "non-finite values",
            }
            continue
        result["frame_diagnostics"][name] = {
            "available": True,
            "shape": list(value.shape),
            "regions": {
                region: _composition_metrics(value, weights, mask)
                for region, mask in region_masks.items()
            },
        }
    return result


def _find_equilibrium_constants(run: Path) -> tuple[Path, dict[str, Any]] | None:
    for parent in (run.parent, *run.parent.parents):
        path = parent / "inputs" / "circular_gs" / "gs_equilibrium.json"
        if path.is_file():
            return path, _load_json(path, "continuous equilibrium constants")
    return None


def _continuous_comparison(
    run: Path,
    report: dict[str, Any],
    initial: dict[str, np.ndarray],
    *,
    nphi: int,
    nc: int,
    radius: np.ndarray,
    axial: np.ndarray,
    weights: np.ndarray,
    region_masks: dict[str, np.ndarray],
    requested: bool,
) -> dict[str, Any]:
    if not requested:
        return {
            "available": False,
            "reason": "continuous comparison not requested",
        }
    located = _find_equilibrium_constants(run)
    if located is None:
        return {
            "available": False,
            "reason": "inputs/circular_gs/gs_equilibrium.json is absent",
        }
    path, document = located
    external = document.get("parameters")
    if not isinstance(external, dict):
        return {"available": False, "reason": "equilibrium parameters are absent"}
    init = report.get("init")
    baseline = init.get("baseline") if isinstance(init, dict) else None
    if not isinstance(baseline, dict):
        return {"available": False, "reason": "run baseline metadata is absent"}
    kind = baseline.get("baseline")
    report_parameters = report.get("parameters")
    if not isinstance(report_parameters, dict):
        report_parameters = {}
    declared = baseline.get("equilibrium_parameters")
    if declared is not None and not isinstance(declared, dict):
        return {
            "available": False,
            "reason": "declared equilibrium_parameters is not an object",
        }

    def same(left: Any, right: Any) -> bool:
        return math.isclose(
            float(left), float(right), rel_tol=1.0e-12, abs_tol=0.0
        )

    if declared is not None:
        parameters = declared
        parameter_source = "run.init.baseline.equilibrium_parameters"
        legacy_validated = True
    else:
        parameters = external
        parameter_source = "campaign legacy constants"
        actual_r0 = report_parameters.get("major_radius")
        actual_a = report_parameters.get("minor_radius")
        if actual_r0 is None or actual_a is None:
            return {
                "available": False,
                "reason": "actual run major/minor radius were not persisted",
            }
        if not (
            same(parameters.get("R0_m"), actual_r0)
            and same(parameters.get("a_m"), actual_a)
        ):
            return {
                "available": False,
                "reason": (
                    "legacy constants mismatch actual run major/minor radius"
                ),
            }
        actual_f0 = baseline.get("F0_T_m", baseline.get("F0"))
        actual_n0 = baseline.get("n0_m^-3", baseline.get("n0"))
        if actual_f0 is None or actual_n0 is None:
            return {
                "available": False,
                "reason": (
                    "legacy constants cannot be validated against persisted "
                    "run F0/n0 baseline values"
                ),
            }
        if not (
            same(parameters.get("F0_T_m"), actual_f0)
            and same(parameters.get("n0_m^-3"), actual_n0)
        ):
            return {
                "available": False,
                "reason": "legacy constants mismatch persisted run F0/n0",
            }
        legacy_validated = True

    actual_r0 = report_parameters.get("major_radius")
    actual_a = report_parameters.get("minor_radius")
    if declared is not None:
        if actual_r0 is None or actual_a is None:
            return {
                "available": False,
                "reason": "actual run major/minor radius were not persisted",
            }
        if not (
            same(parameters.get("R0_m"), actual_r0)
            and same(parameters.get("a_m"), actual_a)
        ):
            return {
                "available": False,
                "reason": (
                    "declared equilibrium parameters mismatch actual run "
                    "major/minor radius"
                ),
            }

    external_degree = (
        int(external["degree"]) if "degree" in external else None
    )
    preprocess_degree = (
        int(baseline.get("gs_degree", external_degree))
        if kind == "gs"
        and (
            baseline.get("gs_degree") is not None
            or external_degree is not None
        )
        else None
    )
    try:
        from tools.circular_equilibrium import (
            build_gs_equilibrium,
            sample_gs,
            sample_vacuum,
        )

        if kind == "gs":
            model = build_gs_equilibrium(
                R0=float(parameters["R0_m"]),
                a=float(parameters["a_m"]),
                F0=float(parameters["F0_T_m"]),
                n0=float(parameters["n0_m^-3"]),
                p_edge=float(parameters["p_edge_Pa"]),
                pprime=float(parameters["pprime_Pa_per_Wb_rad"]),
                alpha=float(parameters["alpha_ion_fraction"]),
                degree=preprocess_degree,
            )
            continuous = sample_gs(model, radius, axial)
        elif kind == "vacuum":
            continuous = sample_vacuum(
                radius,
                axial,
                F0=float(parameters["F0_T_m"]),
                n0=float(parameters["n0_m^-3"]),
                pressure=float(parameters["p_edge_Pa"]),
                alpha=float(parameters["alpha_ion_fraction"]),
            )
        else:
            return {
                "available": False,
                "reason": f"baseline {kind!r} has no continuous comparison",
            }
    except Exception as error:
        return {
            "available": False,
            "reason": f"continuous comparison failed: {type(error).__name__}: {error}",
        }

    mapped = _array(
        initial,
        "B_N_samples",
        (nphi, 3, nc),
        "mapped true-initial magnetic samples",
    )
    expected = np.stack(
        (
            np.asarray(continuous["B_R"]),
            np.asarray(continuous["B_phi"]),
            np.asarray(continuous["B_Z"]),
        ),
        axis=0,
    )[None, :, :]
    difference = mapped - expected
    return {
        "available": True,
        "baseline": kind,
        "constants_file": str(path),
        "constants_sha256": _sha256(path),
        "model_constants": parameters,
        "parameter_source": parameter_source,
        "legacy_constants_validated": legacy_validated,
        "preprocess_degree": preprocess_degree,
        "external_degree": external_degree,
        "degree_difference": (
            None
            if preprocess_degree is None or external_degree is None
            else preprocess_degree - external_degree
        ),
        "degree_note": (
            "Run preprocess degree and external high-accuracy reference "
            "degree are reported separately; they need not be identical."
        ),
        "regions": {
            region: _vector_metrics(difference, weights, mask)
            for region, mask in region_masks.items()
        },
    }


def _geometry(
    run: Path, report: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = run / "geometry.npz"
    arrays = _load_npz(path, "geometry")
    cell_r = np.asarray(arrays.get("cell_R"))
    cell_z = np.asarray(arrays.get("cell_Z"))
    if cell_r.ndim != 1 or cell_z.shape != cell_r.shape or cell_r.size == 0:
        raise AnalysisError("geometry cell_R/cell_Z are absent or inconsistent")
    nc = int(cell_r.size)
    weights = _array(arrays, "M0", (nc,), "geometry")
    weights = np.asarray(weights, dtype=np.float64)
    distances = _array(
        arrays, "physical_wall_distance", (nc,), "geometry"
    )
    distances = np.asarray(distances, dtype=np.float64)
    if not np.isfinite(cell_r).all() or not np.isfinite(cell_z).all():
        raise AnalysisError("geometry cell coordinates contain non-finite values")
    if np.any(weights <= 0.0):
        raise AnalysisError("geometry M0 must be positive")
    if np.any(distances < 0.0):
        raise AnalysisError("physical_wall_distance must be nonnegative")
    fingerprint = (
        str(arrays["geometry_fingerprint"].item())
        if "geometry_fingerprint" in arrays
        else None
    )
    report_geometry = report.get("geometry")
    if not isinstance(report_geometry, dict):
        report_geometry = {}
    summary = {
        "file": str(path),
        "sha256": _sha256(path),
        "fingerprint": fingerprint,
        "reported_fingerprint": report_geometry.get("fingerprint"),
        "operator_fingerprint": report_geometry.get("operator_fingerprint"),
        "layout_kind": (
            str(arrays["layout_kind"].item())
            if "layout_kind" in arrays
            else None
        ),
        "Nc": nc,
        "volume_m3": 2.0 * math.pi * float(np.sum(weights)),
        "cell_R_range_m": [
            float(np.min(cell_r)),
            float(np.max(cell_r)),
        ],
        "cell_Z_range_m": [
            float(np.min(cell_z)),
            float(np.max(cell_z)),
        ],
        "wall_distance_range_m": [
            float(np.min(distances)),
            float(np.max(distances)),
        ],
    }
    return summary, weights, distances, np.asarray(cell_r), np.asarray(cell_z)


def _metadata_summary(report: dict[str, Any]) -> dict[str, Any]:
    parameters = report.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    horizon = report.get("horizon")
    if not isinstance(horizon, dict):
        horizon = {}
    acceptance = report.get("acceptance")
    if not isinstance(acceptance, dict):
        acceptance = {}
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    return {
        "status": report.get("status"),
        "scientific_status": report.get("scientific_status"),
        "source_fingerprint": provenance.get("source_fingerprint"),
        "K": parameters.get("K"),
        "Nphi": parameters.get("Nphi"),
        "grid_nodes": parameters.get("nodes"),
        "h_s": horizon.get("micro_h_s", parameters.get("micro_h")),
        "H_s": horizon.get("macro_h_s", parameters.get("macro_h")),
        "M": horizon.get("M", parameters.get("micro_count")),
        "s": parameters.get("ion_nodes"),
        "accepted_step": acceptance.get("accepted_step"),
        "accepted_time_s": acceptance.get("accepted_time_s"),
        "accepted_intervals": acceptance.get("accepted_intervals"),
        "target_step": acceptance.get("target_step"),
    }


def _analyze_run(
    run: Path, *, wall_width: float, continuous: bool
) -> dict[str, Any]:
    report_path = run / "report.json"
    report = _load_json(report_path, "run report")
    geometry_summary, weights, distances, cell_r, cell_z = _geometry(run, report)
    snapshots = _ordered_snapshots(_snapshot_paths(run, report))
    initial = snapshots[0]
    final = snapshots[-1]
    initial_time = float(initial[1]["time_s"])
    final_time = float(final[1]["time_s"])
    if not math.isfinite(initial_time) or not math.isfinite(final_time):
        raise AnalysisError("true initial or final snapshot time is non-finite")
    if initial_time != 0.0:
        raise AnalysisError(
            f"true initial snapshot time is {initial_time}, not t0=0"
        )
    metadata = _metadata_summary(report)
    nphi_value = initial[2].get("B_N_samples")
    if nphi_value is None:
        raise AnalysisError("true initial snapshot lacks B_N_samples")
    nphi = int(np.asarray(nphi_value).shape[0])
    report_nphi = metadata.get("Nphi")
    if report_nphi is not None and int(report_nphi) != nphi:
        raise AnalysisError(
            f"report Nphi {report_nphi} != snapshot Nphi {nphi}"
        )
    nc = weights.size
    masks = _regions(distances, wall_width)

    drift = _group_drift(
        snapshots,
        nphi=nphi,
        nc=nc,
        weights=weights,
        region_masks=masks,
    )
    drift["background_scale"] = {
        group: {
            "unit": unit,
            "whole": (
                drift["regions"]["whole"]["groups"][group]["background"]
            ),
        }
        for group, (_key, _vector, unit) in GROUPS.items()
    }
    initial_rhs = _initial_rhs(
        run,
        nphi=nphi,
        nc=nc,
        weights=weights,
        region_masks=masks,
    )
    continuous_result = _continuous_comparison(
        run,
        report,
        initial[2],
        nphi=nphi,
        nc=nc,
        radius=cell_r,
        axial=cell_z,
        weights=weights,
        region_masks=masks,
        requested=continuous,
    )
    prepatch = "prepatch" in run.name.lower()
    return {
        "directory": str(run),
        "analysis_status": "available",
        "scope": {
            "format_check_only": prepatch,
            "label": (
                "format_check_only_pre_s_repair_not_scientific_acceptance"
                if prepatch
                else "bounded_saved_run_analysis_not_scientific_acceptance"
            ),
            "scientific_acceptance": False,
            "statement": (
                "Saved physical samples are analyzed without rerunning "
                "evolution; this helper cannot establish scientific acceptance."
            ),
        },
        "report": {
            "path": str(report_path),
            "sha256": _sha256(report_path),
            "status": report.get("status"),
            "scientific_status": report.get("scientific_status"),
            "source_fingerprint": metadata["source_fingerprint"],
        },
        "metadata": {
            **metadata,
            "Nphi": nphi,
            "Nc": nc,
            "initial_snapshot": {
                "file": initial[0].name,
                "step": int(initial[1].get("step", -1)),
                "time_s": initial_time,
            },
            "final_snapshot": {
                "file": final[0].name,
                "step": int(final[1].get("step", -1)),
                "time_s": final_time,
            },
        },
        "geometry": geometry_summary,
        "wall_width_m": wall_width,
        "regions": {
            name: {
                "count": int(np.count_nonzero(mask)),
                "volume_m3": (
                    2.0 * math.pi * float(np.sum(weights[mask]))
                ),
            }
            for name, mask in masks.items()
        },
        "drift": drift,
        "target_seed_scale": _seed_scale(report),
        "initial_rhs": initial_rhs,
        "continuous": continuous_result,
        "cochain_interpretation": {
            "physical_Bdot_components_available": False,
            "reason": (
                "Native RHS cochains are not labeled as physical B-dot "
                "components without an actual QB reconstruction."
            ),
        },
        "cell_centers": {
            "R_m": cell_r.tolist(),
            "Z_m": cell_z.tolist(),
        },
    }


def analyze_run(
    run: str | Path, *, wall_width: float = 0.05, continuous: bool = False
) -> dict[str, Any]:
    """Analyze one saved run; input failures are explicit unavailable output."""
    path = Path(run)
    width = _positive_finite(wall_width, "wall width")
    try:
        return _analyze_run(path, wall_width=width, continuous=continuous)
    except AnalysisError as error:
        prepatch = "prepatch" in path.name.lower()
        return {
            "directory": str(path),
            "analysis_status": "unavailable",
            "scope": {
                "format_check_only": prepatch,
                "label": (
                    "format_check_only_pre_s_repair_not_scientific_acceptance"
                    if prepatch
                    else "bounded_saved_run_analysis_not_scientific_acceptance"
                ),
                "scientific_acceptance": False,
            },
            "error": str(error),
            "drift": {
                "available": False,
                "reason": str(error),
                "final": None,
            },
            "continuous": {
                "available": False,
                "reason": "run analysis unavailable",
            },
        }


def analyze_runs(
    runs: list[str | Path],
    *,
    wall_width: float = 0.05,
    continuous: bool = False,
) -> dict[str, Any]:
    width = _positive_finite(wall_width, "wall width")
    if not runs:
        raise AnalysisError("at least one --run is required")
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wall_width_m": width,
        "continuous_requested": bool(continuous),
        "runs": [
            analyze_run(
                run,
                wall_width=width,
                continuous=continuous,
            )
            for run in runs
        ],
        "scope_statement": (
            "Read-only analysis of persisted physical samples. No evolution "
            "was rerun and missing values were not synthesized."
        ),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            value,
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=Path,
        help="saved run directory; may be repeated",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wall-width", type=float, default=0.05)
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="build the saved circular equilibrium and compare true t0",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise AnalysisError(f"output already exists: {args.output}")
    result = analyze_runs(
        args.run,
        wall_width=args.wall_width,
        continuous=args.continuous,
    )
    _write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
