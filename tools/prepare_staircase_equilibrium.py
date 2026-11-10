"""Formal old-staircase mapping for the frozen self-circle equilibrium."""

from __future__ import annotations

import hashlib
import json
import math
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fomhd import core, io, stage
from fomhd.backend import Backend
from fomhd.geometry import Point2, build_geometry
from fomhd.operators import build_operators
from fomhd.types import Context, Failure, cumulative_record

if __package__:
    from .circular_equilibrium import (
        ELEMENTARY_CHARGE,
        _evaluate_flux,
        sample_gs,
    )
    from .circular_equilibrium import (
        MU0 as ARCHIVED_MU0,
    )
    from .warp_tokamak_cae import host, physical_parameters, project_B
else:
    from circular_equilibrium import (
        ELEMENTARY_CHARGE,
        _evaluate_flux,
        sample_gs,
    )
    from circular_equilibrium import (
        MU0 as ARCHIVED_MU0,
    )
    from tools.warp_tokamak_cae import host, physical_parameters, project_B

DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parents[1]
MEMORY_LIMIT = 4 << 30


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _circle(model, count=128):
    angle = 2.0 * np.pi * np.arange(count, dtype=np.float64) / count
    return np.column_stack(
        (
            model.R0 + model.a * np.cos(angle),
            model.a * np.sin(angle),
        )
    )


def _grid(model, nodes):
    lo = np.array((model.R0 - model.a, -model.a), dtype=np.float64)
    hi = np.array((model.R0 + model.a, model.a), dtype=np.float64)
    spacing = (hi - lo) / nodes
    r = lo[0] + (np.arange(nodes, dtype=np.float64) + 0.5) * spacing[0]
    z = lo[1] + (np.arange(nodes, dtype=np.float64) + 0.5) * spacing[1]
    return lo, hi, spacing, r, z


def _write_toml(
    destination: Path,
    *,
    case_name: str,
    nodes: int,
    spacing,
    grid_lo,
    reference_parameters,
    source_root: Path,
):
    case_template = Path(source_root) / "cases/massless-example.toml"
    source_text = case_template.read_text(encoding="utf-8")
    values = {
        "": {
            "case_name": json.dumps(case_name),
        },
        "[grid]": {
            "R0": repr(float(grid_lo[0]) + float(spacing[0]) / 2.0),
            "Z0": repr(float(grid_lo[1]) + float(spacing[1]) / 2.0),
            "hR": repr(float(spacing[0])),
            "hZ": repr(float(spacing[1])),
            "NR": str(nodes),
            "NZ": str(nodes),
            "K": "0",
            "Nphi": "5",
        },
        "[time]": {
            "start_time": "0.0",
            "dt": repr(1.0e-9),
            "nsteps": "0",
            "initial_kind": '"non_equilibrium"',
            "integrator": '"rk4"',
        },
        "[solver]": {
            "backend": '"cpu"',
            "memory_limit_bytes": str(MEMORY_LIMIT),
            "device_memory_limit_bytes": str(MEMORY_LIMIT),
            "cpu_threads": "1",
        },
        "[solver.references]": {
            "L_m": repr(reference_parameters.ref.length),
            "t_s": repr(reference_parameters.ref.time),
            "n_i_m3": repr(reference_parameters.ref.ni),
            "n_e_m3": repr(reference_parameters.ref.ne),
            "u_mps": repr(reference_parameters.ref.velocity),
            "E_Vpm": repr(reference_parameters.ref.electric),
            "B_T": repr(reference_parameters.ref.magnetic),
            "J_Apm2": repr(reference_parameters.ref.current),
            "rho_Cpm3": repr(reference_parameters.ref.charge),
            "p_Pa": repr(reference_parameters.ref.pressure),
        },
        "[budgets.domain]": {
            "B_min": repr(reference_parameters.B_min),
        },
        "[diagnostics]": {
            "inner_distance_m": repr(0.15 * reference_parameters.ref.length),
        },
        "[io]": {
            "geometry_file": '"geometry.npz"',
            "initial_file": '"initial.npz"',
            "source_manifest_file": '"source_manifest.json"',
            "output_dir": '"run"',
            "snapshot_stride": "1",
            "checkpoint_stride": "1",
            "max_input_bytes": str(MEMORY_LIMIT // 16),
            "max_uncompressed_bytes": str(MEMORY_LIMIT // 16),
            "max_entries": "2000000",
            "max_header_bytes": str(1 << 20),
            "write_residual_arrays": "true",
        },
    }
    for check_name, budget in reference_parameters.budgets.items():
        values[f'[budgets.checks."{check_name}"]'] = {
            "reference": repr(budget.reference),
        }

    section = ""
    output = []
    for line in source_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
        key = None
        if "=" in line and not stripped.startswith("#"):
            key = line.split("=", 1)[0].strip()
        if key in values.get(section, {}):
            comment = line.split("#", 1)[1] if "#" in line else ""
            line = f"{key} = {values[section][key]}"
            if comment:
                line += "  # " + comment
        output.append(line)
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")


def _build_context(model, nodes):
    polygon = _circle(model)
    lo, hi, spacing, r_centers, z_centers = _grid(model, nodes)
    geometry = build_geometry(
        r_centers,
        z_centers,
        [Point2(*point) for point in polygon],
        {"memory_cap_bytes": MEMORY_LIMIT, "grid_rtol": 1.0e-12},
    )
    pressure = model.p_edge + model.pprime * float(
        sample_gs(model, model.R0, 0.0)["psi"]
    )
    parameters = physical_parameters(
        0,
        model.n0,
        abs(model.F0 / model.R0),
        model.a,
        pressure,
    )
    parameters.memory_limit = MEMORY_LIMIT
    parameters.device_memory_limit = MEMORY_LIMIT
    backend = Backend("cpu")
    operators = build_operators(
        geometry,
        0,
        {
            "memory_cap_bytes": MEMORY_LIMIT,
            "rank_rtol": parameters.rank_rtol,
            "include_finite_mass": False,
        },
        backend,
    )
    context = Context(parameters, geometry, operators, backend, 5)
    return context, polygon, (lo, hi, spacing)


def _map_state(model, context):
    geometry = context.geometry
    operators = context.op
    nc = operators.Nc
    state = core.allocate_state(context)
    cells = sample_gs(
        model,
        np.asarray(geometry.cells.R),
        np.asarray(geometry.cells.Z),
    )
    state.fluid[0].n[:nc] = np.full(nc, model.n0)
    state.fluid[1].n[:nc] = np.full(nc, model.n0)
    state.fluid[0].perp[:nc] = cells["p_ion"]
    state.fluid[1].perp[:nc] = cells["p_e_perp"]
    state.fluid[1].parallel[:nc] = cells["p_e_parallel"]
    state.fluid[0].mom[:] = 0.0
    state.fluid[1].mom[:] = 0.0

    def magnetic(table):
        r = np.asarray(table.R)
        z = np.asarray(table.Z)
        _, psi_r, psi_z, _, _, _ = _evaluate_flux(model, r, z)
        return -psi_z / r, model.F0 / r, psi_r / r

    radial, _, _ = magnetic(geometry.z_faces)
    _, toroidal, _ = magnetic(geometry.vertices)
    _, _, vertical = magnetic(geometry.r_faces)
    table_radius_offset = max(
        float(
            np.max(
                np.hypot(
                    np.asarray(geometry.z_faces.R) - model.R0,
                    np.asarray(geometry.z_faces.Z),
                )
                - model.a
            )
        ),
        float(
            np.max(
                np.hypot(
                    np.asarray(geometry.vertices.R) - model.R0,
                    np.asarray(geometry.vertices.Z),
                )
                - model.a
            )
        ),
        float(
            np.max(
                np.hypot(
                    np.asarray(geometry.r_faces.R) - model.R0,
                    np.asarray(geometry.r_faces.Z),
                )
                - model.a
            )
        ),
    )
    raw = np.concatenate((radial, toroidal, vertical)).astype(np.complex128)
    shell = SimpleNamespace(
        p=context.p,
        op=operators,
        geometry=geometry,
        compute=context.compute,
    )
    flux = float(np.real(host(operators.topology_flux_row, context).reshape(-1) @ raw))
    corrected, projection = project_B(raw, shell, 0, flux)
    state.B[: operators.Nb] = context.compute.asarray(corrected)
    state.step = 0
    state.generation = 1
    state.time = 0.0
    return state, {
        "raw_B": raw,
        "corrected_B": corrected,
        "sampling_ranges": {
            "B_R": [float(np.min(radial)), float(np.max(radial))],
            "B_phi": [float(np.min(toroidal)), float(np.max(toroidal))],
            "B_Z": [float(np.min(vertical)), float(np.max(vertical))],
        },
        "projection_changes": {
            "B_R": {
                "max_abs": float(np.max(np.abs(corrected[: radial.size] - radial))),
                "relative_linf": float(
                    np.max(np.abs(corrected[: radial.size] - radial))
                    / max(float(np.max(np.abs(radial))), 1.0e-300)
                ),
            },
            "B_phi": {
                "max_abs": float(
                    np.max(
                        np.abs(
                            corrected[radial.size : radial.size + toroidal.size]
                            - toroidal
                        )
                    )
                ),
                "relative_linf": float(
                    np.max(
                        np.abs(
                            corrected[radial.size : radial.size + toroidal.size]
                            - toroidal
                        )
                    )
                    / max(float(np.max(np.abs(toroidal))), 1.0e-300)
                ),
            },
            "B_Z": {
                "max_abs": float(
                    np.max(np.abs(corrected[radial.size + toroidal.size :] - vertical))
                ),
                "relative_linf": float(
                    np.max(np.abs(corrected[radial.size + toroidal.size :] - vertical))
                    / max(float(np.max(np.abs(vertical))), 1.0e-300)
                ),
            },
        },
        "projection": projection,
        "target_topology_flux": flux,
        "maximum_entity_outside_circle_m": float(max(table_radius_offset, 0.0)),
    }


def _actual_phi(model, r, z):
    _, psi_r, _, psi_rr, _, psi_zz = _evaluate_flux(model, r, z)
    return -(psi_rr + psi_zz - psi_r / r) / (ARCHIVED_MU0 * r)


def _continuous_electron_reference(model, context, canonical=None):
    geometry = context.geometry
    r = np.asarray(geometry.cells.R)
    z = np.asarray(geometry.cells.Z)
    jphi = _actual_phi(model, r, z)
    ue_phi = -jphi / (ELEMENTARY_CHARGE * model.n0)
    expected_momentum = model.n0 * context.p.species[1].mass * ue_phi
    output = {
        "units": {
            "J_phi": "A/m^2",
            "u_e_phi": "m/s",
            "electron_momentum_density": "kg/(m^2*s)",
        },
        "actual_J_phi_range": [float(np.min(jphi)), float(np.max(jphi))],
        "expected_u_e_phi_range": [
            float(np.min(ue_phi)),
            float(np.max(ue_phi)),
        ],
        "expected_electron_momentum_density_range": [
            float(np.min(expected_momentum)),
            float(np.max(expected_momentum)),
        ],
        "declared_velocity_scale_m_per_s": context.p.ref.velocity,
        "declared_momentum_scale": (
            model.n0 * context.p.species[1].mass * context.p.ref.velocity
        ),
    }
    if canonical is not None:
        nc = context.op.Nc
        modes = 2 * context.p.K + 1
        compact = np.asarray(canonical.fluid[1].mom).reshape(modes, 3, nc)[context.p.K]
        actual_complex = compact[1]
        actual = np.real(actual_complex)
        difference = actual - expected_momentum
        output["canonical_compatibility_momentum_phi_range"] = [
            float(np.min(actual)),
            float(np.max(actual)),
        ]
        output["canonical_minus_continuous_max_abs"] = float(np.max(np.abs(difference)))
        output["canonical_compatibility_imaginary_max_abs"] = float(
            np.max(np.abs(np.imag(actual_complex)))
        )
        output["canonical_minus_continuous_relative_to_physical_scale"] = float(
            np.max(np.abs(difference))
            / max(float(np.max(np.abs(expected_momentum))), 1.0e-300)
        )
        output["classification"] = (
            "diagnostic comparison only; canonical compatibility momentum is "
            "not a replacement physical electron-momentum evolution variable"
        )
    return output


def _staircase_wall_mapping(model, context):
    geometry = context.geometry
    records = []
    electric_scale = model.alpha * model.pprime / (ELEMENTARY_CHARGE * model.n0)
    for wall in geometry.walls:
        if wall.component == "r_face":
            table = geometry.r_faces
            normal = np.array((float(wall.sign), 0.0))
        else:
            table = geometry.z_faces
            normal = np.array((0.0, float(wall.sign)))
        r = float(table.R[wall.face])
        z = float(table.Z[wall.face])
        radius = math.hypot(r - model.R0, z)
        smooth_normal = np.array((r - model.R0, z)) / radius
        tangent = np.array((-normal[1], normal[0]))
        smooth_tangent = np.array((-smooth_normal[1], smooth_normal[0]))
        _, psi_r, psi_z, _, _, _ = _evaluate_flux(model, r, z)
        b = np.array((-psi_z / r, model.F0 / r, psi_r / r))
        e = np.array(
            (
                electric_scale * psi_r,
                0.0,
                electric_scale * psi_z,
            )
        )
        smooth_r = model.R0 + model.a * smooth_normal[0]
        smooth_z = model.a * smooth_normal[1]
        _, smooth_psi_r, smooth_psi_z, _, _, _ = _evaluate_flux(
            model, smooth_r, smooth_z
        )
        smooth_b = np.array(
            (
                -smooth_psi_z / smooth_r,
                model.F0 / smooth_r,
                smooth_psi_r / smooth_r,
            )
        )
        smooth_e = np.array(
            (
                electric_scale * smooth_psi_r,
                0.0,
                electric_scale * smooth_psi_z,
            )
        )
        records.append(
            {
                "outside_circle_m": max(radius - model.a, 0.0),
                "staircase_normal_B": float(b[0] * normal[0] + b[2] * normal[1]),
                "smooth_normal_B": float(
                    smooth_b[0] * smooth_normal[0] + smooth_b[2] * smooth_normal[1]
                ),
                "staircase_tangent_E": float(e[0] * tangent[0] + e[2] * tangent[1]),
                "smooth_tangent_E": float(
                    smooth_e[0] * smooth_tangent[0] + smooth_e[2] * smooth_tangent[1]
                ),
            }
        )
    outside = np.array([row["outside_circle_m"] for row in records])
    bstair = np.array([row["staircase_normal_B"] for row in records])
    bsmooth = np.array([row["smooth_normal_B"] for row in records])
    estair = np.array([row["staircase_tangent_E"] for row in records])
    esmooth = np.array([row["smooth_tangent_E"] for row in records])
    return {
        "count": len(records),
        "maximum_outside_circle_m": float(np.max(outside)),
        "mean_outside_circle_m": float(np.mean(outside)),
        "staircase_normal_B": {
            "max_abs": float(np.max(np.abs(bstair))),
            "rms": float(np.sqrt(np.mean(bstair**2))),
        },
        "smooth_normal_B": {
            "max_abs": float(np.max(np.abs(bsmooth))),
            "rms": float(np.sqrt(np.mean(bsmooth**2))),
        },
        "staircase_tangent_E": {
            "max_abs": float(np.max(np.abs(estair))),
            "rms": float(np.sqrt(np.mean(estair**2))),
        },
        "smooth_tangent_E": {
            "max_abs": float(np.max(np.abs(esmooth))),
            "rms": float(np.sqrt(np.mean(esmooth**2))),
        },
        "classification": (
            "staircase wall/domain-mapping mismatch; extension diagnostics do "
            "not imply the continuous circle input is dynamically inconsistent"
        ),
    }


def _write_source_manifest(destination, case_path, geometry_path, initial_path):
    document = {
        "provider_name": "equilibrium_audit_frozen_self_circle",
        "provider_version": "1",
        "units": "SI",
        "conversion_method": (
            "physical sample_gs at old R/Z entities; input-only minimum-MB "
            "compatibility projection preserving topology flux"
        ),
        "input_hashes": {
            "case": _sha256(case_path),
            "geometry": _sha256(geometry_path),
            "initial": _sha256(initial_path),
        },
        "continuous_residual_status": (
            "evaluated by independent equilibrium-audit compatibility probe; "
            "polynomial Ampere discrepancy retained"
        ),
    }
    destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _region_weights(context, band_factor=1.0):
    geometry = context.geometry
    band = max(
        float(np.min(np.diff(np.asarray(geometry.R, dtype=np.float64)))),
        float(np.min(np.diff(np.asarray(geometry.Z, dtype=np.float64)))),
    )
    distance = np.asarray(geometry.cell_wall_distance, dtype=np.float64)
    wall = distance <= band_factor * band
    return wall, ~wall, np.asarray(context.op.M0, dtype=np.float64)


def _state_change(before, after):
    output = {}
    for key in sorted(before):
        if key not in after:
            continue
        a = np.asarray(before[key])
        b = np.asarray(after[key])
        if a.shape != b.shape or not np.issubdtype(a.dtype, np.number):
            continue
        difference = b - a
        original_scale = float(np.max(np.abs(a), initial=0.0))
        output[key] = {
            "max_abs_change": float(np.max(np.abs(difference), initial=0.0)),
            "relative_linf_change": (
                None
                if original_scale <= 1.0e-300
                else float(np.max(np.abs(difference), initial=0.0) / original_scale)
            ),
        }
    return output


def _region_summary(state, context):
    wall, interior, weights = _region_weights(context)
    output = {}
    nc = context.op.Nc
    modes = 2 * context.p.K + 1
    center_mode = context.p.K
    compact_fields = {
        "n_i": (state.fluid[0].n, 1),
        "mom_i": (state.fluid[0].mom, 3),
        "p_i": (state.fluid[0].perp, 1),
        "n_e": (state.fluid[1].n, 1),
        "mom_e": (state.fluid[1].mom, 3),
        "p_perp_e": (state.fluid[1].perp, 1),
        "p_parallel_e": (state.fluid[1].parallel, 1),
    }
    for key, (value, components) in compact_fields.items():
        compact = np.asarray(value).reshape(modes, components, nc)[center_mode]
        for region, mask in (("wall_band", wall), ("interior", interior)):
            values = np.asarray(compact)[:, mask]
            value_weights = weights[mask]
            if values.size == 0:
                output[f"{key}.{region}"] = 0.0
                continue
            output[f"{key}.{region}"] = float(
                np.sqrt(
                    np.sum(
                        value_weights[None, :]
                        * np.sum(np.abs(values) ** 2, axis=0)[None, :]
                    )
                    / max(float(np.sum(value_weights)), 1.0e-300)
                )
            )
    geometry = context.geometry
    nz = int(geometry.z_faces.R.size)
    nv = int(geometry.vertices.R.size)
    nr = int(geometry.r_faces.R.size)
    b = np.asarray(host(state.B, context)).reshape(modes, -1)[center_mode]
    tables = (
        ("B_R", geometry.z_faces, b[:nz]),
        ("B_phi", geometry.vertices, b[nz : nz + nv]),
        ("B_Z", geometry.r_faces, b[nz + nv : nz + nv + nr]),
    )
    band = max(
        float(np.min(np.diff(np.asarray(geometry.R, dtype=np.float64)))),
        float(np.min(np.diff(np.asarray(geometry.Z, dtype=np.float64)))),
    )
    for key, table, values in tables:
        distance = np.abs(
            np.hypot(
                np.asarray(table.R, dtype=np.float64) - 3.0,
                np.asarray(table.Z, dtype=np.float64),
            )
            - 0.5
        )
        for region, mask in (
            ("wall_band", distance <= band),
            ("interior", distance > band),
        ):
            selected = np.asarray(values)[mask]
            output[f"{key}.{region}"] = (
                0.0
                if selected.size == 0
                else float(np.sqrt(np.mean(np.abs(selected) ** 2)))
            )
    return output


def build_and_run(
    output_root: Path,
    model,
    nodes: int,
    *,
    source_root: Path | None = None,
):
    source_root = (
        DEFAULT_SOURCE_ROOT if source_root is None else Path(source_root).resolve()
    )
    if not (source_root / "cases/massless-example.toml").is_file():
        raise ValueError("source_root must contain cases/massless-example.toml")
    if not (source_root / "src/fomhd").is_dir():
        raise ValueError("source_root must contain src/fomhd")
    output = Path(output_root) / f"g{nodes}"
    output.mkdir(parents=True, exist_ok=True)
    context, polygon, grid = _build_context(model, nodes)
    lo, hi, spacing = grid
    geometry_path = output / "geometry.npz"
    np.savez_compressed(
        geometry_path,
        R_nodes=np.asarray(context.geometry.R),
        Z_nodes=np.asarray(context.geometry.Z),
        lcfs_R=polygon[:, 0],
        lcfs_Z=polygon[:, 1],
    )
    state, mapping = _map_state(model, context)
    initial_path = output / "initial.npz"
    initial_arrays = io.encode_state(state, context.geometry)
    np.savez_compressed(initial_path, **initial_arrays)
    case_path = output / "case.toml"
    _write_toml(
        case_path,
        case_name=f"self-circle-old-staircase-g{nodes}",
        nodes=nodes,
        spacing=spacing,
        grid_lo=lo,
        reference_parameters=context.p,
        source_root=source_root,
    )
    source_path = output / "source_manifest.json"
    _write_source_manifest(source_path, case_path, geometry_path, initial_path)
    result = {
        "nodes": nodes,
        "grid": {
            "lo": lo.tolist(),
            "hi": hi.tolist(),
            "hR_m": float(spacing[0]),
            "hZ_m": float(spacing[1]),
            "cells": int(context.op.Nc),
            "r_faces": int(context.geometry.r_faces.R.size),
            "z_faces": int(context.geometry.z_faces.R.size),
            "vertices": int(context.geometry.vertices.R.size),
        },
        "mapping": {
            "projection": mapping["projection"],
            "sampling_ranges": mapping["sampling_ranges"],
            "projection_changes": mapping["projection_changes"],
            "target_topology_flux_Wb_per_rad": mapping["target_topology_flux"],
            "maximum_entity_outside_circle_m": mapping[
                "maximum_entity_outside_circle_m"
            ],
            "staircase_vs_smooth_wall": _staircase_wall_mapping(model, context),
            "raw_B_fingerprint": hashlib.sha256(mapping["raw_B"].tobytes()).hexdigest(),
            "corrected_B_fingerprint": hashlib.sha256(
                mapping["corrected_B"].tobytes()
            ).hexdigest(),
        },
        "input_hashes": {
            "case": _sha256(case_path),
            "geometry": _sha256(geometry_path),
            "initial": _sha256(initial_path),
            "source_manifest": _sha256(source_path),
        },
        "continuous_electron_reference": _continuous_electron_reference(model, context),
    }
    try:
        case = io.load_case(case_path)
        loaded = io.load_initial(case)
        formal_context = Context(
            case.parameters,
            case.geometry,
            case.operators,
            case.compute,
            5,
        )
        original = io.encode_state(loaded, case.geometry)
        ledger = cumulative_record()
        captured = {}
        committed = stage.initialize(
            loaded,
            formal_context,
            ledger,
            False,
            captured,
        )
        canonical = io.encode_state(committed.state, case.geometry)
        initialized_stage = SimpleNamespace(**captured)
        probe_stage = stage.evaluate_stage(
            committed.state,
            initialized_stage.tag,
            formal_context,
            False,
        )
        rhs = io.encode_state(probe_stage.rhs, case.geometry)
        np.savez_compressed(output / "original.npz", **original)
        np.savez_compressed(output / "canonical.npz", **canonical)
        np.savez_compressed(output / "initial_rhs.npz", **rhs)
        result.update(
            {
                "status": "completed",
                "load_and_initialize": "passed",
                "canonical_changes": _state_change(original, canonical),
                "canonical_electron_comparison": _continuous_electron_reference(
                    model, context, committed.state
                ),
                "rhs_region_scales": _region_summary(probe_stage.rhs, formal_context),
                "initial_rhs_maxima": {
                    key: float(np.max(np.abs(value), initial=0.0))
                    for key, value in rhs.items()
                },
                "checks": {
                    "count": len(probe_stage.checks),
                    "hard_failures": sum(
                        bool(check.hard and not check.passed)
                        for check in probe_stage.checks
                    ),
                },
                "operator_hash": case.operator_hash,
                "semantic_hash": case.semantic_hash,
            }
        )
    except Failure as failure:
        result.update(
            {
                "status": "hard_failure",
                "failure": {
                    "code": failure.code,
                    "message": str(failure),
                    "stage": (
                        None if failure.stage is None else failure.stage.__dict__
                    ),
                    "partial_array_keys": sorted(failure.partial_arrays),
                },
            }
        )
        np.savez_compressed(
            output / "failure_arrays.npz",
            **{
                key: value
                for key, value in failure.partial_arrays.items()
                if isinstance(value, (np.ndarray, int, float, complex))
            },
        )
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        result.update(
            {
                "status": "earlier_failure",
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            }
        )
    (output / "mapping.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result
