#!/usr/bin/env python3
"""Measure pressure-force and massless Ohm fields on fixed RZ domains."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEST_FIXTURE_DIR = ROOT / "tests" / "warp"


def _resolve_source_parent(root: Path) -> Path:
    """Use ``src`` only when the planned package layout is actually present."""
    root = Path(root).resolve()
    return root / "src" if (root / "src" / "fomhd").is_dir() else root


PACKAGE_PARENT = _resolve_source_parent(ROOT)
for _path in (PACKAGE_PARENT, TEST_FIXTURE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fomhd  # noqa: E402
from fomhd import massless, stage  # noqa: E402
from fomhd.backend import Backend  # noqa: E402
from fomhd.core import forward, inverse, sample_apply  # noqa: E402
from fomhd.fluid import _component_flux, gradient_adjoint  # noqa: E402
from fomhd.geometry import Point2, build_geometry, point_segment_distance  # noqa: E402
from fomhd.massless import _massless_e0, _wall_correction  # noqa: E402
from fomhd.operators import OperatorOptions, build_operators  # noqa: E402
from fomhd.types import Context, Failure, StageTag  # noqa: E402
import foundation_fixture  # noqa: E402
import massless_verification_fixture  # noqa: E402
import mms_fixture  # noqa: E402
from massless_verification_fixture import HallCase, _parameters  # noqa: E402
from mms_fixture import require_device  # noqa: E402


PACKAGE_DIR = Path(fomhd.__file__).resolve().parent
DEFAULT_MEMORY_CAP = 8 << 30
P0 = 0.4
PRESSURE_AMPLITUDE = 0.04
ELECTRON_DENSITY = 1.0e16
TOROIDAL_MAGNETIC = 0.1
REFERENCE_WIDTH = 0.15
NODES_MIN = 4
NODES_MAX = 32
SHAPES = ("rectangle", "disk")
PROFILES = ("compact", "wall")
OWN_HARNESS_PATHS = (
    ROOT / "tools" / "warp_spatial_study.py",
    ROOT / "tests" / "warp" / "test_spatial_study.py",
    ROOT / "docs" / "pseudocode" / "tools" / "warp_spatial_study.py.md",
)
DEFAULT_FIXTURE_PATHS = tuple(
    Path(module.__file__).resolve()
    for module in (
        massless_verification_fixture,
        mms_fixture,
        foundation_fixture,
    )
    if getattr(module, "__file__", None)
)


@dataclass(frozen=True)
class FixedGeometry:
    """Production geometry and reference-domain metadata for one study mesh."""

    nodes: int
    shape: str
    h: float
    geometry: Any
    active_cells: int
    cell_mask_area_m2: float
    toroidal_volume_m3: float
    polygon_area_m2: float
    analytic_target_area_m2: float
    hausdorff_lower_m: float | None
    hausdorff_upper_m: float | None


@dataclass(frozen=True)
class SpatialCaseResult:
    """Host evidence from one geometry/profile measurement."""

    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


def fixed_geometry(nodes: int, shape: str) -> FixedGeometry:
    """Build a fixed physical RZ reference domain using production geometry."""
    if isinstance(nodes, bool) or not isinstance(nodes, int):
        raise Failure("StudyInput", "nodes must be an integer")
    if not NODES_MIN <= nodes <= NODES_MAX:
        raise Failure(
            "StudyInput",
            f"nodes must be in [{NODES_MIN}, {NODES_MAX}]",
        )
    selected = str(shape).lower()
    if selected not in SHAPES:
        raise Failure("StudyInput", f"unsupported shape {shape!r}")

    h = 1.0 / float(nodes)
    radial = 1.0 + (np.arange(nodes, dtype=np.float64) + 0.5) * h
    axial = (np.arange(nodes, dtype=np.float64) + 0.5) * h
    if selected == "rectangle":
        polygon = (
            Point2(1.0, 0.0),
            Point2(2.0, 0.0),
            Point2(2.0, 1.0),
            Point2(1.0, 1.0),
        )
        target_area = 1.0
    else:
        angle = 2.0 * math.pi * np.arange(128, dtype=np.float64) / 128.0
        polygon = tuple(
            Point2(
                1.5 + 0.45 * math.cos(float(value)),
                0.5 + 0.45 * math.sin(float(value)),
            )
            for value in angle
        )
        target_area = math.pi * 0.45**2

    geometry = build_geometry(radial, axial, polygon)
    active_cells = len(geometry.cells.keys)
    lower = getattr(geometry.diagnostics, "hausdorff_lower_m", None)
    upper = getattr(geometry.diagnostics, "hausdorff_upper_m", None)
    return FixedGeometry(
        nodes=nodes,
        shape=selected,
        h=h,
        geometry=geometry,
        active_cells=active_cells,
        cell_mask_area_m2=float(active_cells) * h * h,
        toroidal_volume_m3=2.0 * math.pi * float(np.sum(geometry.M0)),
        polygon_area_m2=abs(float(geometry.diagnostics.signed_polygon_area)),
        analytic_target_area_m2=target_area,
        hausdorff_lower_m=None if lower is None else float(lower),
        hausdorff_upper_m=None if upper is None else float(upper),
    )


def _reference_boundary_distance(
    r: Any,
    z: Any,
    shape: str,
) -> np.ndarray:
    radial, axial = np.broadcast_arrays(
        np.asarray(r, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
    )
    selected = str(shape).lower()
    if selected == "rectangle":
        return np.minimum.reduce(
            (
                radial - 1.0,
                2.0 - radial,
                axial,
                1.0 - axial,
            )
        )
    if selected == "disk":
        return 0.45 - np.hypot(radial - 1.5, axial - 0.5)
    raise Failure("StudyInput", f"unsupported shape {shape!r}")


def fixed_region_masks(
    r: Any,
    z: Any,
    shape: str,
    width: float = REFERENCE_WIDTH,
) -> dict[str, np.ndarray]:
    """Partition samples using the fixed reference boundary, not the staircase."""
    if not math.isfinite(width) or width < 0.0:
        raise Failure("StudyInput", "region width must be finite and non-negative")
    distance = _reference_boundary_distance(r, z, shape)
    inner = distance > float(width)
    return {
        "all": np.ones(distance.shape, dtype=bool),
        "inner": inner,
        "wall": ~inner,
    }


def analytic_pressure(
    r: Any,
    z: Any,
    phi: Any,
    profile: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pressure variation and exact cylindrical gradient components."""
    radial, axial, angle = np.broadcast_arrays(
        np.asarray(r, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
        np.asarray(phi, dtype=np.float64),
    )
    if np.any(radial <= 0.0):
        raise Failure("StudyInput", "analytic pressure requires R > 0")
    selected = str(profile).lower()
    if selected == "compact":
        radius_square = 0.25**2
        q = (
            (radial - 1.5) ** 2 + (axial - 0.5) ** 2
        ) / radius_square
        inside = q < 1.0
        safe = np.where(inside, 1.0 - q, 1.0)
        profile_value = np.where(
            inside,
            np.exp(1.0 - 1.0 / safe),
            0.0,
        )
        derivative_q = np.where(
            inside,
            -profile_value / (safe * safe),
            0.0,
        )
        derivative_r = derivative_q * 2.0 * (radial - 1.5) / radius_square
        derivative_z = derivative_q * 2.0 * (axial - 0.5) / radius_square
    elif selected == "wall":
        disk_square = 0.45**2
        envelope = np.sin(math.pi * (radial - 1.0)) * np.sin(
            math.pi * axial
        )
        boundary_factor = 1.0 - (
            (radial - 1.5) ** 2 + (axial - 0.5) ** 2
        ) / disk_square
        profile_value = (envelope * boundary_factor) ** 2
        envelope_r = math.pi * np.cos(
            math.pi * (radial - 1.0)
        ) * np.sin(math.pi * axial)
        envelope_z = (
            math.pi
            * np.sin(math.pi * (radial - 1.0))
            * np.cos(math.pi * axial)
        )
        factor_r = -2.0 * (radial - 1.5) / disk_square
        factor_z = -2.0 * (axial - 0.5) / disk_square
        derivative_r = (
            2.0
            * envelope
            * boundary_factor
            * (envelope_r * boundary_factor + envelope * factor_r)
        )
        derivative_z = (
            2.0
            * envelope
            * boundary_factor
            * (envelope_z * boundary_factor + envelope * factor_z)
        )
    else:
        raise Failure("StudyInput", f"unsupported pressure profile {profile!r}")

    cosine = np.cos(angle)
    sine = np.sin(angle)
    variation = PRESSURE_AMPLITUDE * profile_value * cosine
    gradient_r = PRESSURE_AMPLITUDE * derivative_r * cosine
    gradient_phi_over_r = (
        -PRESSURE_AMPLITUDE * profile_value * sine / radial
    )
    gradient_z = PRESSURE_AMPLITUDE * derivative_z * cosine
    return variation, gradient_r, gradient_phi_over_r, gradient_z


def _expanded_region_mask(
    selection: Any,
    components: int,
    entities: int,
) -> np.ndarray:
    values = np.asarray(selection, dtype=bool)
    if values.shape == (entities,):
        return np.repeat(values[None, :], components, axis=0)
    if values.shape == (components, entities):
        return values.copy()
    raise Failure(
        "Shape",
        f"region mask must have shape ({entities},) or "
        f"({components}, {entities}), got {values.shape}",
    )


def weighted_norms(
    actual: Any,
    reference: Any,
    entity_weights: Any,
    regions: Mapping[str, Any],
    *,
    phi_weight: float,
    unit: str,
    component_names: Sequence[str] = ("R", "phi", "Z"),
) -> dict[str, dict[str, Any]]:
    """Return physical quadrature norms with explicit region availability."""
    measured = np.asarray(actual, dtype=np.complex128)
    expected = np.asarray(reference, dtype=np.complex128)
    if measured.ndim != 3 or measured.shape != expected.shape:
        raise Failure("Shape", "weighted norms require matching [N,C,E] arrays")
    samples, components, entities = measured.shape
    weights = np.asarray(entity_weights, dtype=np.float64)
    if weights.shape != (entities,):
        raise Failure("Shape", "weighted norm entity weights do not match")
    if (
        np.any(~np.isfinite(weights))
        or np.any(weights <= 0.0)
        or not math.isfinite(phi_weight)
        or phi_weight <= 0.0
    ):
        raise Failure("Weight", "quadrature weights must be finite and positive")
    if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(expected)):
        raise Failure("Finite", "weighted norm input contains non-finite values")
    if len(component_names) != components:
        raise Failure("Shape", "component names do not match norm input")

    difference = measured - expected
    absolute_difference = np.abs(difference)
    absolute_reference = np.abs(expected)

    def summarize(selection: np.ndarray) -> dict[str, Any]:
        value_count = int(samples * np.count_nonzero(selection))
        selected_entities = [int(value) for value in np.count_nonzero(
            selection, axis=1
        )]
        if value_count == 0:
            empty_components = {
                name: {
                    "available": False,
                    "reason": "empty region",
                    "entity_count": 0,
                    "value_count": 0,
                    "weight_sum": 0.0,
                    "linf": None,
                    "weighted_rms": None,
                    "weighted_l2": None,
                    "reference_weighted_l2": None,
                    "difference_weighted_l2": None,
                    "relative_weighted_l2": None,
                    "relative_reason": "empty region",
                }
                for name in component_names
            }
            return {
                "available": False,
                "reason": "empty region",
                "unit": unit,
                "metric_units": {"linf": unit, "weighted_rms": unit, "weighted_l2": f"({unit}) m^(3/2)", "relative_weighted_l2": "1"},
                "counts": {
                    "phi_samples": samples,
                    "components": components,
                    "entity_count": 0,
                    "entity_count_by_component": selected_entities,
                    "value_count": 0,
                },
                "phi_weight": float(phi_weight),
                "weight_sum": 0.0,
                "linf": None,
                "weighted_rms": None,
                "weighted_l2": None,
                "reference_weighted_l2": None,
                "difference_weighted_l2": None,
                "relative_weighted_l2": None,
                "relative_reason": "empty region",
                "components": empty_components,
            }

        broadcast_selection = selection[None, :, :]
        weight_field = np.broadcast_to(
            weights[None, None, :], measured.shape
        ) * broadcast_selection
        difference_square = float(
            np.sum(absolute_difference**2 * weight_field)
        )
        reference_square = float(
            np.sum(absolute_reference**2 * weight_field)
        )
        selected_weight_sum = float(np.sum(weights * selection))
        weighted_l2 = math.sqrt(phi_weight * difference_square)
        reference_l2 = math.sqrt(phi_weight * reference_square)
        weighted_rms = math.sqrt(
            difference_square / (samples * selected_weight_sum)
        )
        if reference_l2 > 0.0:
            relative = weighted_l2 / reference_l2
            relative_reason = ""
        else:
            relative = None
            relative_reason = "zero reference weighted L2"
        selected_values = absolute_difference[
            np.broadcast_to(broadcast_selection, measured.shape)
        ]
        component_records = {}
        for component, name in enumerate(component_names):
            component_selection = selection[component]
            component_values = absolute_difference[:, component, :][
                :, component_selection
            ]
            component_weights = weights[component_selection]
            component_difference_square = float(
                np.sum(
                    component_values**2
                    * component_weights[None, :]
                )
            )
            component_reference_square = float(
                np.sum(
                    absolute_reference[:, component, :][
                        :, component_selection
                    ]
                    ** 2
                    * component_weights[None, :]
                )
            )
            component_l2 = math.sqrt(
                phi_weight * component_difference_square
            )
            component_reference_l2 = math.sqrt(
                phi_weight * component_reference_square
            )
            component_records[name] = {
                "available": bool(
                    selected_entities[component] > 0
                ),
                "reason": (
                    ""
                    if selected_entities[component] > 0
                    else "empty component region"
                ),
                "entity_count": selected_entities[component],
                "value_count": int(
                    samples * selected_entities[component]
                ),
                "weight_sum": float(
                    np.sum(component_weights)
                ),
                "linf": (
                    float(np.max(component_values))
                    if component_values.size
                    else None
                ),
                "weighted_rms": (
                    math.sqrt(
                        component_difference_square
                        / (samples * float(np.sum(component_weights)))
                    )
                    if component_values.size
                    else None
                ),
                "weighted_l2": (
                    component_l2 if component_values.size else None
                ),
                "reference_weighted_l2": (
                    component_reference_l2
                    if component_values.size
                    else None
                ),
                "difference_weighted_l2": (
                    component_l2 if component_values.size else None
                ),
                "relative_weighted_l2": (
                    component_l2 / component_reference_l2
                    if component_values.size
                    and component_reference_l2 > 0.0
                    else None
                ),
                "relative_reason": (
                    ""
                    if component_values.size
                    and component_reference_l2 > 0.0
                    else (
                        "empty component region"
                        if not component_values.size
                        else "zero component reference weighted L2"
                    )
                ),
            }
        return {
            "available": True,
            "reason": "",
            "unit": unit,
                "metric_units": {"linf": unit, "weighted_rms": unit, "weighted_l2": f"({unit}) m^(3/2)", "relative_weighted_l2": "1"},
            "counts": {
                "phi_samples": samples,
                "components": components,
                "entity_count": int(
                    np.count_nonzero(np.any(selection, axis=0))
                ),
                "entity_count_by_component": selected_entities,
                "value_count": value_count,
            },
            "phi_weight": float(phi_weight),
            "weight_sum": selected_weight_sum,
            "linf": float(np.max(selected_values)),
            "weighted_rms": weighted_rms,
            "weighted_l2": weighted_l2,
            "reference_weighted_l2": reference_l2,
            "difference_weighted_l2": weighted_l2,
            "relative_weighted_l2": relative,
            "relative_reason": relative_reason,
            "components": component_records,
        }

    return {
        str(name): summarize(
            _expanded_region_mask(selection, components, entities)
        )
        for name, selection in regions.items()
    }


def _isotropic_tensor(
    pressure: np.ndarray,
    context: Context,
) -> Any:
    samples, cells = pressure.shape
    tensor = np.zeros(
        (samples, 3, 3, cells), dtype=np.complex128
    )
    for component in range(3):
        tensor[:, component, component, :] = pressure
    return context.compute.asarray(tensor.reshape(-1))


def _distance_to_numerical_wall(
    r: np.ndarray,
    z: np.ndarray,
    geometry: Any,
) -> np.ndarray:
    result = np.full(r.shape, np.inf, dtype=np.float64)
    for wall in geometry.walls:
        first, second = wall.vertices
        a = Point2(
            float(geometry.vertices.R[first]),
            float(geometry.vertices.Z[first]),
        )
        b = Point2(
            float(geometry.vertices.R[second]),
            float(geometry.vertices.Z[second]),
        )
        for index, (radial, axial) in enumerate(zip(r.flat, z.flat)):
            result.flat[index] = min(
                result.flat[index],
                point_segment_distance(
                    Point2(float(radial), float(axial)),
                    a,
                    b,
                ),
            )
    return result


def _geometry_record(fixed: FixedGeometry, context: Context) -> dict[str, Any]:
    geometry = context.geometry
    return {
        "shape": fixed.shape,
        "nodes": fixed.nodes,
        "h_m": fixed.h,
        "R_bounds_m": [1.0, 2.0],
        "Z_bounds_m": [0.0, 1.0],
        "grid_cells": fixed.nodes * fixed.nodes,
        "active_cells": fixed.active_cells,
        "active_fraction": (
            fixed.active_cells / float(fixed.nodes * fixed.nodes)
        ),
        "cell_mask_area_m2": fixed.cell_mask_area_m2,
        "analytic_target_area_m2": fixed.analytic_target_area_m2,
        "cell_mask_area_relative_to_analytic": (
            fixed.cell_mask_area_m2 / fixed.analytic_target_area_m2 - 1.0
        ),
        "fixed_polygon_vertices": len(geometry.polygon),
        "fixed_polygon_area_m2": fixed.polygon_area_m2,
        "toroidal_volume_m3": fixed.toroidal_volume_m3,
        "toroidal_volume_definition": "2*pi*sum(M0)",
        "hausdorff_lower_m": fixed.hausdorff_lower_m,
        "hausdorff_upper_m": fixed.hausdorff_upper_m,
        "numerical_wall_segments": len(geometry.walls),
        "region_partition": (
            "fixed reference boundary distance; staircase distance is metadata"
        ),
        "inner_distance_m": REFERENCE_WIDTH,
    }


def _physical_vector_samples(
    retained: Any,
    entities: int,
    context: Context,
) -> np.ndarray:
    samples = context.compute.to_host(
        inverse(
            retained,
            entities,
            context,
            True,
            context.p.ref.electric,
        )
    )
    return np.asarray(samples).reshape(context.N, entities)


def _staggered_vector_norms(
    actual: np.ndarray,
    reference: np.ndarray,
    entity_weights: np.ndarray,
    regions: Mapping[str, np.ndarray],
    component_slices: Mapping[str, slice],
    *,
    phi_weight: float,
    unit: str,
) -> dict[str, dict[str, Any]]:
    """Measure the flat production edge layout and each physical component."""
    aggregate = weighted_norms(
        actual[:, None, :],
        reference[:, None, :],
        entity_weights,
        regions,
        phi_weight=phi_weight,
        unit=unit,
        component_names=("staggered",),
    )
    for region_name in regions:
        component_records = {}
        for component_name, selected_slice in component_slices.items():
            component_norms = weighted_norms(
                actual[:, selected_slice][:, None, :],
                reference[:, selected_slice][:, None, :],
                entity_weights[selected_slice],
                {
                    name: region[selected_slice]
                    for name, region in regions.items()
                },
                phi_weight=phi_weight,
                unit=unit,
                component_names=(component_name,),
            )
            component_records[component_name] = component_norms[region_name]
        aggregate[region_name]["components"] = component_records
        aggregate[region_name]["counts"]["components"] = len(component_slices)
        aggregate[region_name]["layout"] = (
            "staggered production M1 order: R faces, cell centers, Z faces"
        )
    return aggregate


def _correction_rows(
    correction_record: Any,
    context: Context,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    residuals = []
    right_hand_sides = []
    selected = []
    gauge_rows = []
    relative = 0.0
    for record in correction_record.modes:
        residual = np.asarray(
            context.compute.to_host(record.full_residual),
            dtype=np.complex128,
        )
        rhs = np.asarray(
            context.compute.to_host(record.rhs),
            dtype=np.complex128,
        )
        residuals.append(residual)
        right_hand_sides.append(rhs)
        row_mask = np.zeros(rhs.size, dtype=bool)
        row_mask[np.asarray(record.selected_rows, dtype=np.int64)] = True
        selected.append(row_mask)
        gauge_rows.extend(int(value) for value in record.gauge_rows)
        relative = max(
            relative,
            float(np.max(np.abs(residual)))
            / max(float(np.max(np.abs(rhs))), 1.0e-300),
        )
    residual_array = np.stack(residuals)
    rhs_array = np.stack(right_hand_sides)
    selected_array = np.stack(selected)
    metadata = {
        "mode_count": len(correction_record.modes),
        "rank": int(correction_record.rank),
        "rows_checked": int(residual_array.size),
        "original_row_residual_linf": float(
            np.max(np.abs(residual_array))
        ),
        "relative_original_row_residual_linf": relative,
        "selected_rows_per_mode": [
            len(record.selected_rows)
            for record in correction_record.modes
        ],
        "gauge_rows": gauge_rows,
        "residual_role": (
            "algebraic original-row correction residual; not continuum error"
        ),
    }
    return residual_array, rhs_array, selected_array, metadata


def spatial_case(
    context: Context,
    shape: str,
    profile: str,
) -> SpatialCaseResult:
    """Measure one fixed-domain pressure-force and massless Ohm case."""
    selected_shape = str(shape).lower()
    selected_profile = str(profile).lower()
    if selected_shape not in SHAPES:
        raise Failure("StudyInput", f"unsupported shape {shape!r}")
    if selected_profile not in PROFILES:
        raise Failure("StudyInput", f"unsupported profile {profile!r}")
    if context.p.K != 1 or context.N != 5 or context.op.K != 1:
        raise Failure("StudyContext", "spatial case requires K=1 and Nphi=5")
    if not hasattr(context, "fixed_geometry"):
        raise Failure("StudyContext", "context lacks fixed geometry metadata")
    fixed: FixedGeometry = context.fixed_geometry
    if fixed.geometry is not context.geometry or fixed.shape != selected_shape:
        raise Failure("StudyContext", "fixed geometry metadata does not match context")

    xp = context.compute.xp
    cells = context.op.Nc
    entities = context.op.Ne
    radial_cells = len(context.geometry.r_faces.keys)
    axial_cells = len(context.geometry.z_faces.keys)
    phi = 2.0 * math.pi * np.arange(context.N, dtype=np.float64) / context.N
    phi_weight = 2.0 * math.pi / context.N

    cell_r = np.asarray(context.geometry.cells.R, dtype=np.float64)
    cell_z = np.asarray(context.geometry.cells.Z, dtype=np.float64)
    pressure_variation, radial_gradient, azimuthal_gradient, axial_gradient = (
        analytic_pressure(
            cell_r[None, :],
            cell_z[None, :],
            phi[:, None],
            selected_profile,
        )
    )
    pressure_full = P0 + pressure_variation
    force_exact = -np.stack(
        (radial_gradient, azimuthal_gradient, axial_gradient),
        axis=1,
    )

    density = np.full(
        (context.N, cells), ELECTRON_DENSITY, dtype=np.complex128
    )
    density_device = context.compute.asarray(density.reshape(-1))
    face_density = _component_flux(density_device, context)
    velocity = np.zeros(
        (context.N, 3, cells), dtype=np.complex128
    )
    fluid = SimpleNamespace(
        n=density_device,
        nf=face_density,
        u=context.compute.asarray(velocity.reshape(-1)),
    )

    tensor_full = _isotropic_tensor(pressure_full, context)
    tensor_delta = _isotropic_tensor(pressure_variation, context)
    tensor_constant = _isotropic_tensor(
        np.full_like(pressure_variation, P0), context
    )
    raw_full_device = gradient_adjoint(fluid, tensor_full, context)
    raw_delta_device = gradient_adjoint(fluid, tensor_delta, context)
    raw_constant_device = gradient_adjoint(
        fluid, tensor_constant, context
    )
    projected_full_device = massless._constraint_project(
        raw_full_device, density_device, context
    )
    constant_projected_device = massless._constraint_project(
        raw_constant_device, density_device, context
    )
    projected_delta_device = (
        projected_full_device - constant_projected_device
    )
    raw_full = context.compute.to_host(raw_full_device).reshape(
        context.N, 3, cells
    )
    raw_delta = context.compute.to_host(raw_delta_device).reshape(
        context.N, 3, cells
    )
    raw_constant = context.compute.to_host(
        raw_constant_device
    ).reshape(context.N, 3, cells)
    projected_full = context.compute.to_host(
        projected_full_device
    ).reshape(context.N, 3, cells)
    constant_projected = context.compute.to_host(
        constant_projected_device
    ).reshape(context.N, 3, cells)
    projected_delta = context.compute.to_host(
        projected_delta_device
    ).reshape(context.N, 3, cells)

    state = stage.closed_reference(context, ELECTRON_DENSITY, P0, P0, TOROIDAL_MAGNETIC)
    nb = context.op.Nb
    harmonic = state.B[context.p.K*nb:(context.p.K+1)*nb]
    fixed_flux = TOROIDAL_MAGNETIC * math.log(2.0)
    measured_flux = context.compute.scalar((context.op.topology_flux_row @ harmonic)[0])
    harmonic *= fixed_flux / measured_flux
    pressure_hat = forward(context.compute.asarray(pressure_full.reshape(-1)), cells, context)
    state.fluid[1].perp = pressure_hat.copy()
    state.fluid[1].parallel = pressure_hat.copy()
    tag = StageTag(state.step, state.generation, 1, 0, state.time, "spatial_evidence")
    frame = massless.evaluate_stage(state, tag, context, False)
    magnetic = context.compute.to_host(frame.BN).reshape(context.N, 3, cells)
    e0_retained = frame.E0
    electric_retained = frame.cache.E
    correction_retained = frame.cache.wall_reaction
    correction_record = frame.cache.wall_correction
    context.compute.synchronize()
    e0 = _physical_vector_samples(e0_retained, entities, context)
    electric = _physical_vector_samples(
        electric_retained, entities, context
    )
    correction = _physical_vector_samples(
        correction_retained, entities, context
    )

    edge_r = np.concatenate(
        (
            np.asarray(context.geometry.r_faces.R, dtype=np.float64),
            cell_r,
            np.asarray(context.geometry.z_faces.R, dtype=np.float64),
        )
    )
    edge_z = np.concatenate(
        (
            np.asarray(context.geometry.r_faces.Z, dtype=np.float64),
            cell_z,
            np.asarray(context.geometry.z_faces.Z, dtype=np.float64),
        )
    )
    (
        edge_pressure,
        edge_radial_gradient,
        edge_azimuthal_gradient,
        edge_axial_gradient,
    ) = analytic_pressure(
        edge_r[None, :],
        edge_z[None, :],
        phi[:, None],
        selected_profile,
    )
    elementary_charge = -float(context.p.species[1].charge)
    electric_exact = np.empty(
        (context.N, entities), dtype=np.float64
    )
    radial_slice = slice(0, radial_cells)
    center_slice = slice(radial_cells, radial_cells + cells)
    axial_slice = slice(radial_cells + cells, entities)
    electric_exact[:, radial_slice] = (
        -edge_radial_gradient[:, radial_slice]
        / (elementary_charge * ELECTRON_DENSITY)
    )
    electric_exact[:, center_slice] = (
        -edge_azimuthal_gradient[:, center_slice]
        / (elementary_charge * ELECTRON_DENSITY)
    )
    electric_exact[:, axial_slice] = (
        -edge_axial_gradient[:, axial_slice]
        / (elementary_charge * ELECTRON_DENSITY)
    )

    cell_regions = fixed_region_masks(
        cell_r, cell_z, selected_shape
    )
    radial_regions = fixed_region_masks(
        np.asarray(context.geometry.r_faces.R),
        np.asarray(context.geometry.r_faces.Z),
        selected_shape,
    )
    center_regions = fixed_region_masks(cell_r, cell_z, selected_shape)
    axial_regions = fixed_region_masks(
        np.asarray(context.geometry.z_faces.R),
        np.asarray(context.geometry.z_faces.Z),
        selected_shape,
    )
    edge_regions = {
        name: np.concatenate(
            (
                radial_regions[name],
                center_regions[name],
                axial_regions[name],
            )
        )
        for name in ("all", "inner", "wall")
    }
    edge_weights = np.asarray(context.compute.to_host(context.op.M1), dtype=np.float64)
    cell_weights = np.asarray(context.compute.to_host(context.op.M0), dtype=np.float64)
    component_slices = {
        "R": radial_slice,
        "phi": center_slice,
        "Z": axial_slice,
    }

    force_norms = {
        "raw_full_force": weighted_norms(
            raw_full,
            force_exact,
            cell_weights,
            cell_regions,
            phi_weight=phi_weight,
            unit="Pa/m",
        ),
        "raw_delta_force": weighted_norms(
            raw_delta,
            force_exact,
            cell_weights,
            cell_regions,
            phi_weight=phi_weight,
            unit="Pa/m",
        ),
        "projected_full_force": weighted_norms(
            projected_full,
            force_exact,
            cell_weights,
            cell_regions,
            phi_weight=phi_weight,
            unit="Pa/m",
        ),
        "constant_projected_force": weighted_norms(
            constant_projected,
            np.zeros_like(force_exact),
            cell_weights,
            cell_regions,
            phi_weight=phi_weight,
            unit="Pa/m",
        ),
        "projected_delta_force": weighted_norms(
            projected_delta,
            force_exact,
            cell_weights,
            cell_regions,
            phi_weight=phi_weight,
            unit="Pa/m",
        ),
    }
    force_norms["raw_full_force"].update(
        {
            "continuum_pass_fail_eligible": False,
            "interpretation": (
                "raw gradient-adjoint output contains the discrete weak "
                "boundary lift and is not pointwise -grad(p)"
            ),
        }
    )
    for name in (
        "raw_delta_force",
        "projected_full_force",
        "constant_projected_force",
        "projected_delta_force",
    ):
        force_norms[name]["continuum_pass_fail_eligible"] = False
        force_norms[name]["interpretation"] = (
            "reported continuum comparison; no scientific threshold selected"
        )

    ohm_norms = {
        "E0": _staggered_vector_norms(
            e0,
            electric_exact,
            edge_weights,
            edge_regions,
            component_slices,
            phi_weight=phi_weight,
            unit="V/m",
        ),
        "E": _staggered_vector_norms(
            electric,
            electric_exact,
            edge_weights,
            edge_regions,
            component_slices,
            phi_weight=phi_weight,
            unit="V/m",
        ),
        "E_correction": _staggered_vector_norms(
            correction,
            np.zeros_like(correction),
            edge_weights,
            edge_regions,
            component_slices,
            phi_weight=phi_weight,
            unit="V/m",
        ),
    }

    residual, rhs, selected_rows, correction_metadata = _correction_rows(
        correction_record, context
    )
    actual_wall_slots = np.concatenate(
        (
            np.asarray(
                context.geometry.r_faces.boundary, dtype=bool
            ),
            np.zeros(cells, dtype=bool),
            np.asarray(
                context.geometry.z_faces.boundary, dtype=bool
            ),
        )
    )
    off_wall = ~actual_wall_slots
    correction_difference = electric - e0
    correction_scale = max(
        float(np.max(np.abs(correction))),
        float(np.max(np.abs(correction_difference))),
        1.0e-300,
    )
    confined_limit = 4096.0 * np.finfo(np.float64).eps * correction_scale
    off_wall_linf = max(
        float(np.max(np.abs(correction[:, off_wall]), initial=0.0)),
        float(
            np.max(
                np.abs(correction_difference[:, off_wall]),
                initial=0.0,
            )
        ),
    )
    inner = edge_regions["inner"]
    inner_correction_linf = float(
        np.max(np.abs(correction[:, inner]), initial=0.0)
    )
    inner_difference_linf = float(
        np.max(
            np.abs(correction_difference[:, inner]),
            initial=0.0,
        )
    )
    difference_identity_linf = float(
        np.max(np.abs(correction_difference - correction))
    )
    correction_metadata.update(
        {
            "actual_wall_slot_count": int(
                np.count_nonzero(actual_wall_slots)
            ),
            "off_wall_linf": off_wall_linf,
            "confined_limit": confined_limit,
            "confined_to_actual_wall_slots": bool(
                off_wall_linf <= confined_limit
            ),
            "inner_correction_linf": inner_correction_linf,
            "inner_field_difference_linf": inner_difference_linf,
            "E_minus_E0_identity_linf": difference_identity_linf,
        }
    )

    cell_reference_distance = _reference_boundary_distance(
        cell_r, cell_z, selected_shape
    )
    edge_reference_distance = _reference_boundary_distance(
        edge_r, edge_z, selected_shape
    )
    cell_numerical_distance = np.asarray(
        context.geometry.cell_wall_distance, dtype=np.float64
    )
    edge_numerical_distance = _distance_to_numerical_wall(
        edge_r,
        edge_z,
        context.geometry,
    )
    edge_component_index = np.concatenate(
        (
            np.zeros(radial_cells, dtype=np.uint8),
            np.ones(cells, dtype=np.uint8),
            np.full(axial_cells, 2, dtype=np.uint8),
        )
    )

    arrays = {
        "phi_rad": phi,
        "cell_r_m": cell_r,
        "cell_z_m": cell_z,
        "edge_r_m": edge_r,
        "edge_z_m": edge_z,
        "edge_component_index": edge_component_index,
        "cell_reference_boundary_distance_m": cell_reference_distance,
        "edge_reference_boundary_distance_m": edge_reference_distance,
        "cell_numerical_wall_distance_m": cell_numerical_distance,
        "edge_numerical_wall_distance_m": edge_numerical_distance,
        "cell_region_all": cell_regions["all"],
        "cell_region_inner": cell_regions["inner"],
        "cell_region_wall": cell_regions["wall"],
        "edge_region_all": edge_regions["all"],
        "edge_region_inner": edge_regions["inner"],
        "edge_region_wall": edge_regions["wall"],
        "actual_wall_E_slot": actual_wall_slots,
        "pressure_variation_Pa": pressure_variation,
        "pressure_full_Pa": pressure_full,
        "pressure_edge_variation_Pa": edge_pressure,
        "B_node_mode_vector_T": context.compute.to_host(state.B),
        "B_cell_samples_T": magnetic,
        "raw_full_force_Pa_per_m": raw_full,
        "raw_delta_force_Pa_per_m": raw_delta,
        "raw_constant_force_Pa_per_m": raw_constant,
        "projected_full_force_Pa_per_m": projected_full,
        "constant_projected_force_Pa_per_m": constant_projected,
        "projected_delta_force_Pa_per_m": projected_delta,
        "force_exact_Pa_per_m": force_exact,
        "raw_full_force_error_Pa_per_m": raw_full - force_exact,
        "raw_delta_force_error_Pa_per_m": raw_delta - force_exact,
        "projected_full_force_error_Pa_per_m": (
            projected_full - force_exact
        ),
        "constant_projected_force_error_Pa_per_m": (
            constant_projected
        ),
        "projected_delta_force_error_Pa_per_m": (
            projected_delta - force_exact
        ),
        "E0_V_per_m": e0,
        "E_V_per_m": electric,
        "E_exact_V_per_m": electric_exact,
        "E_correction_V_per_m": correction,
        "E0_error_V_per_m": e0 - electric_exact,
        "E_error_V_per_m": electric - electric_exact,
        "E_correction_error_V_per_m": correction,
        "wall_correction_residual": residual,
        "wall_correction_rhs": rhs,
        "wall_correction_selected_rows": selected_rows,
    }

    metadata = {
        "label": (
            f"{selected_shape}_n{fixed.nodes}_{selected_profile}"
        ),
        "shape": selected_shape,
        "profile": selected_profile,
        "nodes": fixed.nodes,
        "h_m": fixed.h,
        "K": int(context.p.K),
        "Nphi": int(context.N),
        "density_m3": ELECTRON_DENSITY,
        "toroidal_magnetic_field_T": TOROIDAL_MAGNETIC,
        "magnetic_field_source": (
            "discrete harmonic B normalized to fixed toroidal flux; complete massless stage"
        ),
        "prescribed_ion_velocity_m_per_s": 0.0,
        "derived_electron_velocity_max_m_per_s": float(xp.max(xp.abs(frame.fluid[1].u)).item()),
        "fixed_toroidal_flux_Wb": fixed_flux,
        "full_stage_hard_checks_pass": all(c.available and c.passed for c in frame.checks if c.hard),
        "full_stage_check_count": len(frame.checks),
        "phi_weight_rad": phi_weight,
        "geometry": _geometry_record(fixed, context),
        "force": force_norms,
        "ohm": {
            "E0": ohm_norms["E0"],
            "E": ohm_norms["E"],
            "E_correction": ohm_norms["E_correction"],
            "correction": correction_metadata,
            "reference_path": (
                "direct analytic evaluation at production sample locations; "
                "no I/Q/H/G/C application"
            ),
            "continuum_role": (
                "spatial measurement only; no accuracy threshold or order "
                "acceptance selected"
            ),
        },
        "status": "measured",
    }
    return SpatialCaseResult(metadata=metadata, arrays=arrays)


def build_context(
    nodes: int,
    shape: str,
    device: str = "cpu",
    memory_cap: int = DEFAULT_MEMORY_CAP,
) -> Context:
    """Build one K=1, Nphi=5 fixed-domain production context."""
    if isinstance(memory_cap, bool) or not isinstance(memory_cap, int):
        raise Failure("Memory", "memory cap must be an integer byte count")
    if memory_cap <= 0:
        raise Failure("Memory", "memory cap must be positive")
    normalized_device = require_device(device)
    parameters = _parameters(HallCase(kdi=1))
    parameters.K = 1
    parameters.Nphi = 5
    parameters.inner_distance = REFERENCE_WIDTH
    parameters.memory_limit = memory_cap
    parameters.device_memory_limit = memory_cap
    backend = Backend(normalized_device)
    fixed = fixed_geometry(nodes, shape)
    operators = build_operators(
        fixed.geometry,
        parameters.K,
        OperatorOptions(
            rank_rtol=parameters.rank_rtol,
            memory_cap_bytes=memory_cap,
        ),
        backend,
    )
    context = Context(
        p=parameters,
        geometry=fixed.geometry,
        op=operators,
        compute=backend,
        N=parameters.Nphi,
        next_generation=2,
    )
    context.fixed_geometry = fixed
    backend.synchronize()
    return context


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            _json_safe(payload),
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    temporary.replace(destination)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, complex):
        return {"real": _json_safe(value.real), "imag": _json_safe(value.imag)}
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def snapshot_source(
    destination: Path,
    *,
    package_dir: Path | None = None,
    fixture_paths: Sequence[Path] | None = None,
    harness_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Copy and hash the imported solver package, fixtures, and harness."""
    package_root = (
        PACKAGE_DIR
        if package_dir is None
        else Path(package_dir).resolve()
    )
    fixtures = tuple(
        Path(path).resolve()
        for path in (
            DEFAULT_FIXTURE_PATHS
            if fixture_paths is None
            else fixture_paths
        )
    )
    harness = tuple(
        Path(path).resolve()
        for path in (
            OWN_HARNESS_PATHS
            if harness_paths is None
            else harness_paths
        )
    )
    if not package_root.is_dir():
        raise Failure("SourceSnapshot", f"package directory absent: {package_root}")
    package_files = sorted(package_root.rglob("*.py"))
    missing = [
        str(path)
        for path in (*package_files, *fixtures, *harness)
        if not path.is_file()
    ]
    if missing:
        raise Failure("SourceSnapshot", f"snapshot input is absent: {missing}")

    snapshot_root = Path(destination)
    snapshot_root.mkdir(parents=True, exist_ok=False)
    entries = []

    def archive(source: Path, relative: Path, role: str) -> None:
        target = snapshot_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_hash = _sha256(source)
        destination_hash = _sha256(target)
        if source_hash != destination_hash:
            raise Failure(
                "SourceSnapshot",
                f"snapshot hash mismatch for {source}",
            )
        entries.append(
            {
                "role": role,
                "source_path": str(source.resolve()),
                "snapshot_path": str(relative),
                "size_bytes": target.stat().st_size,
                "sha256": source_hash,
            }
        )

    for path in package_files:
        archive(
            path,
            Path("fomhd") / path.relative_to(package_root),
            "imported_solver_package",
        )
    for path in fixtures:
        archive(path, Path("fixtures") / path.name, "imported_fixture")
    for path in harness:
        try:
            relative = path.relative_to(ROOT.resolve())
        except ValueError:
            relative = Path(path.name)
        archive(path, Path("harness") / relative, "verification_harness")

    manifest = {
        "schema": "a0-fixed-spatial-source-v1",
        "source_root": str(ROOT.resolve()),
        "package_parent": str(PACKAGE_PARENT.resolve()),
        "package_dir": str(package_root),
        "fixture_paths": [str(path) for path in fixtures],
        "harness_paths": [str(path) for path in harness],
        "file_count": len(entries),
        "files": entries,
        "scope": (
            "actual imported Python solver package, imported fixtures, and "
            "own verification harness"
        ),
    }
    manifest_path = snapshot_root / "SOURCE_MANIFEST.json"
    _write_json(manifest_path, manifest)
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }


def _save_npz(path: Path, arrays: Mapping[str, Any]) -> str:
    cleaned: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise Failure(
                "Output",
                f"array {name} has object dtype",
            )
        cleaned[str(name)] = array
    if not cleaned:
        raise Failure("Output", "case produced no arrays")
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **cleaned)
    return _sha256(Path(path))


def _software_record(device: str) -> dict[str, Any]:
    distributions = {}
    for distribution in (
        "numpy",
        "scipy",
        "warp-lang",
        "cupy-cuda13x",
        "pytest",
    ):
        try:
            distributions[distribution] = importlib.metadata.version(
                distribution
            )
        except importlib.metadata.PackageNotFoundError:
            distributions[distribution] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "distributions": distributions,
        "precision": "float64/complex128",
        "device": device,
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def _failure_record(
    label: str,
    error: Exception,
    elapsed_s: float,
) -> dict[str, Any]:
    return {
        "case": label,
        "status": "error",
        "elapsed_s": elapsed_s,
        "failure_code": getattr(error, "code", type(error).__name__),
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
        "partial_array_names": sorted(
            getattr(error, "partial_arrays", {}) or {}
        ),
    }


def _prepare_output(path: Path) -> Path:
    destination = Path(path)
    if destination.exists():
        if not destination.is_dir():
            raise Failure("Output", "output path exists and is not a directory")
        if any(destination.iterdir()):
            raise Failure("Output", "output directory must be new or empty")
    else:
        destination.mkdir(parents=True)
    return destination


def _expand_tokens(values: Sequence[str]) -> list[str]:
    expanded = []
    for value in values:
        expanded.extend(
            token.strip()
            for token in str(value).split(",")
            if token.strip()
        )
    return expanded


def _unique(values: Sequence[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--nodes",
        nargs="+",
        default=["6,8,12,16"],
        help="grid cell counts in each direction, comma or space separated",
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["rectangle,disk"],
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["compact,wall"],
    )
    parser.add_argument(
        "--memory-cap-gib",
        type=float,
        default=8.0,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    started = time.perf_counter()
    nodes_tokens = _expand_tokens(arguments.nodes)
    shapes = tuple(
        str(value).lower() for value in _expand_tokens(arguments.shapes)
    )
    profiles = tuple(
        str(value).lower() for value in _expand_tokens(arguments.profiles)
    )
    try:
        nodes = _unique([int(value) for value in nodes_tokens])
    except ValueError as error:
        print(f"invalid node list: {error}", file=sys.stderr)
        return 2
    if not nodes or any(
        value < NODES_MIN or value > NODES_MAX for value in nodes
    ):
        print(
            f"nodes must be in [{NODES_MIN}, {NODES_MAX}]",
            file=sys.stderr,
        )
        return 2
    if not shapes or any(value not in SHAPES for value in shapes):
        print(f"shapes must be selected from {SHAPES}", file=sys.stderr)
        return 2
    if not profiles or any(value not in PROFILES for value in profiles):
        print(f"profiles must be selected from {PROFILES}", file=sys.stderr)
        return 2
    if (
        not math.isfinite(arguments.memory_cap_gib)
        or arguments.memory_cap_gib <= 0.0
    ):
        print("memory cap must be positive and finite", file=sys.stderr)
        return 2
    try:
        device = require_device(arguments.device)
        output = _prepare_output(arguments.output)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 2

    memory_cap = int(arguments.memory_cap_gib * (1 << 30))
    report: dict[str, Any] = {
        "schema": "a0-fixed-spatial-study-v1",
        "status": "running",
        "backend_requested": arguments.device,
        "device": device,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "arguments": {
            "nodes": nodes,
            "shapes": list(shapes),
            "profiles": list(profiles),
            "memory_cap_gib": arguments.memory_cap_gib,
            "memory_cap_bytes": memory_cap,
            "output": str(output),
        },
        "scope": {
            "included": [
                "fixed-domain pressure-force against independent analytic gradient",
                "massless Ohm field against independent analytic charge pressure force",
                "rectangle and fixed-128-vertex disk staircase masks",
                "K=1 Nphi=5 actual closed RZ/Fourier production operators",
                "inner/wall/all regional physical quadrature norms",
            ],
            "excluded": [
                "scientific tolerance selection",
                "global convergence-order claim from one grid",
                "finite-inertia dynamic momentum-floor identification",
                "temporal integration, nonlinear equilibrium evolution, and A1 work",
                "GPU execution unless explicitly run inside an owned Slurm allocation",
            ],
        },
        "scientific_acceptance": "not_assessed",
        "software": _software_record(device),
        "cases": [],
        "failures": [],
    }
    try:
        report["source_snapshot"] = snapshot_source(
            output / "source"
        )
    except Exception as error:
        report["failures"].append(
            _failure_record(
                "source_snapshot",
                error,
                time.perf_counter() - started,
            )
        )
        report["status"] = "failed"
        report["elapsed_s"] = time.perf_counter() - started
        try:
            _write_json(output / "report.json", report)
        except Exception as report_error:
            print(
                f"failed to write report: {report_error}",
                file=sys.stderr,
            )
        return 1

    for node_count in nodes:
        for shape in shapes:
            setup_started = time.perf_counter()
            try:
                context = build_context(
                    node_count,
                    shape,
                    device,
                    memory_cap,
                )
                context.compute.synchronize()
            except Exception as error:
                report["failures"].append(
                    _failure_record(
                        f"setup_{shape}_n{node_count}",
                        error,
                        time.perf_counter() - setup_started,
                    )
                )
                _write_json(output / "report.json", report)
                continue
            setup_record = {
                "nodes": node_count,
                "shape": shape,
                "elapsed_s": time.perf_counter() - setup_started,
                "geometry": _geometry_record(
                    context.fixed_geometry, context
                ),
            }
            report.setdefault("geometry_builds", []).append(setup_record)
            _write_json(output / "report.json", report)
            for profile in profiles:
                label = f"{shape}_n{node_count}_{profile}"
                case_started = time.perf_counter()
                try:
                    result = spatial_case(context, shape, profile)
                    array_path = output / f"{label}.npz"
                    array_hash = _save_npz(array_path, result.arrays)
                    case = _json_safe(result.metadata)
                    case.update(
                        {
                            "elapsed_s": (
                                time.perf_counter() - case_started
                            ),
                            "arrays_file": array_path.name,
                            "arrays_sha256": array_hash,
                        }
                    )
                    report["cases"].append(case)
                except Exception as error:
                    failure_arrays = mms_fixture.write_failure_arrays(output, error, label+"-failure")
                    failure_checks = mms_fixture.write_failure_checks(output, error, label+"-failure")
                    report["failures"].append(
                        _failure_record(
                            label,
                            error,
                            time.perf_counter() - case_started,
                        )
                    )
                    report["failures"][-1].update(arrays=failure_arrays, checks=failure_checks)
                _write_json(output / "report.json", report)
            del context

    request_count = len(nodes) * len(shapes) * len(profiles)
    report["requested_case_count"] = request_count
    report["measured_case_count"] = len(report["cases"])
    report["failed_case_count"] = len(report["failures"])
    report["execution_completed"] = not report["failures"]
    report["status"] = (
        "measured" if report["execution_completed"] else "failed"
    )
    report["elapsed_s"] = time.perf_counter() - started
    report["scope"]["successful_execution_meaning"] = (
        "requested quantities were measured; accuracy is not accepted"
    )
    try:
        _write_json(output / "report.json", report)
    except Exception as error:
        print(
            f"failed to write report: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print(output / "report.json")
    return 0 if report["execution_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
