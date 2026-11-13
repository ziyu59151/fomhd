"""Factored conservative toroidal reconstruction and magnetic pairing."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ghost_magnetic import ghost_link_norm
from .ghost_reconstruction import ghost_link_reconstruction
from .types import Failure


@dataclass(frozen=True)
class GhostToroidal:
    radial_inverse: Any
    axial_inverse: Any
    radial_norm: Any
    axial_norm: Any
    radial_rows: Any
    axial_rows: Any
    radial_wall_rows: Any
    axial_wall_rows: Any
    radial_radius: Any
    radial_center_radius: Any
    radial_wall_radius: Any
    vertex_radius: Any
    harmonic_defect: Any
    flux_projection: Any


def _intervals(nodes):
    edges = np.r_[nodes[0], (nodes[:-1] + nodes[1:]) / 2, nodes[-1]]
    return edges[:-1], edges[1:]


def _interior_inverse(chart, size):
    if chart.shape != (size, size) or not np.all(np.isfinite(chart)):
        raise Failure("Operators", "invalid toroidal graph line chart")
    try:
        return np.linalg.solve(chart, np.eye(size))
    except np.linalg.LinAlgError as error:
        raise Failure("Operators", "singular toroidal graph line chart") from error


def _interior_map(values, radial, axial):
    result = values.copy()
    result[..., 1:-1, 1:-1] = axial @ values[..., 1:-1, 1:-1] @ radial.T
    return result


def build_ghost_toroidal(radial_nodes, axial_nodes, radial_chart, axial_chart, backend):
    """Build the fixed ghost factors from full vertex lines and interior charts."""
    rlo, rhi = _intervals(radial_nodes)
    zlo, zhi = _intervals(axial_nodes)
    radial_size, axial_size = radial_nodes.size, axial_nodes.size
    radial_rows, radial_wall_rows = ghost_link_reconstruction(radial_nodes, rlo, rhi, False, None)
    axial_rows, axial_wall_rows = ghost_link_reconstruction(axial_nodes, zlo, zhi, False, None)
    radial_norm = ghost_link_norm(
        radial_nodes, rlo, rhi, False, np.eye(radial_size), inverse_radius_weight=True
    )
    axial_norm = ghost_link_norm(axial_nodes, zlo, zhi, False, np.eye(axial_size))
    radial_inverse = _interior_inverse(radial_chart, radial_size - 2)
    axial_inverse = _interior_inverse(axial_chart, axial_size - 2)
    radial_radius = (rhi + rlo) / 2
    vertex_radius = np.broadcast_to(radial_radius, (axial_size, radial_size)).copy()
    harmonic = 1.0 / vertex_radius
    harmonic_defect = harmonic - _interior_map(harmonic, radial_inverse, axial_inverse)
    area = (zhi - zlo)[:, None] * (rhi - rlo)
    flux_projection = area / np.sum(area * harmonic)
    return GhostToroidal(
        radial_inverse=backend.real_array(radial_inverse),
        axial_inverse=backend.real_array(axial_inverse),
        radial_norm=backend.real_array(radial_norm),
        axial_norm=backend.real_array(axial_norm),
        radial_rows=backend.real_array(radial_rows),
        axial_rows=backend.real_array(axial_rows),
        radial_wall_rows=backend.real_array(radial_wall_rows),
        axial_wall_rows=backend.real_array(axial_wall_rows),
        radial_radius=backend.real_array(radial_radius),
        radial_center_radius=backend.real_array((radial_nodes[:-1] + radial_nodes[1:]) / 2),
        radial_wall_radius=backend.real_array(radial_nodes[[0, -1]]),
        vertex_radius=backend.real_array(vertex_radius),
        harmonic_defect=backend.real_array(harmonic_defect),
        flux_projection=backend.real_array(flux_projection),
    )


def _toroidal_raw(factors, values, backend):
    size = factors.vertex_radius.size
    if values.ndim == 0 or values.shape[-1] != size:
        raise Failure("Shape", "toroidal magnetic input has an invalid vertex extent")
    stored = values.reshape(values.shape[:-1] + factors.vertex_radius.shape)
    raw = _interior_map(stored, factors.radial_inverse, factors.axial_inverse)
    coefficient = backend.xp.sum(stored * factors.flux_projection, axis=(-2, -1))
    raw += coefficient[..., None, None] * factors.harmonic_defect
    return raw


def toroidal_reconstruct(factors, values, backend):
    """Return physical center tesla values from batched stored vertex fields."""
    average = _toroidal_raw(factors, values, backend) * factors.radial_radius
    centers = factors.axial_rows @ average @ factors.radial_rows.T
    centers /= factors.radial_center_radius
    return centers.reshape(values.shape[:-1] + (-1,))


def toroidal_trace(factors, values, backend):
    """Return physical tesla traces on radial walls then axial walls."""
    average = _toroidal_raw(factors, values, backend) * factors.radial_radius
    radial = factors.axial_rows @ average @ factors.radial_wall_rows.T
    radial /= factors.radial_wall_radius
    axial = factors.axial_wall_rows @ average @ factors.radial_rows.T
    axial /= factors.radial_center_radius
    return radial, axial


def toroidal_weight(factors, values, backend):
    """Apply the positive toroidal magnetic norm in the stored vertex chart."""
    average = _toroidal_raw(factors, values, backend) * factors.radial_radius
    load = factors.axial_norm @ average @ factors.radial_norm.T
    load *= factors.radial_radius
    result = _interior_map(load, factors.radial_inverse.T, factors.axial_inverse.T)
    coefficient = backend.xp.sum(load * factors.harmonic_defect, axis=(-2, -1))
    result += coefficient[..., None, None] * factors.flux_projection
    return result.reshape(values.shape)
