"""Render initial/rigid physical RZ evidence with the shared A1 Matplotlib style."""

from __future__ import annotations

from typing import Any

import numpy as np

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


FIGURE6 = "fig06_repair3_rigid_rotation_strain"
FIGURE7 = "fig07_self_circle_initial_formal_rhs"


def _dense_cells(
    R: list[float],
    Z: list[float],
    values: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R_array = np.asarray(R, dtype=np.float64)
    Z_array = np.asarray(Z, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    R_unique = np.unique(R_array)
    Z_unique = np.unique(Z_array)
    grid = np.full((Z_unique.size, R_unique.size), np.nan, dtype=np.float64)
    for R_value, Z_value, value in zip(R_array, Z_array, value_array):
        i = int(np.searchsorted(R_unique, R_value))
        j = int(np.searchsorted(Z_unique, Z_value))
        grid[j, i] = value
    return R_unique, Z_unique, grid


def _draw_staircase_wall(axis: Any, data: dict[str, Any]) -> None:
    segments = np.asarray(data["old_wall_segments_m"], dtype=np.float64)
    if segments.ndim != 2 or segments.shape[1] != 4 or not len(segments):
        raise ValueError("old staircase wall segments are missing or malformed")
    lines = segments.reshape(-1, 2, 2)
    axis.add_collection(
        LineCollection(
            lines,
            colors=["#37474F"],
            linewidths=0.8,
            linestyles="--",
            label="old staircase cell edges",
        )
    )
    axis.update_datalim(lines.reshape(-1, 2))
    axis.autoscale_view()


def _draw_walls(axis: Any, data: dict[str, Any]) -> None:
    axis.plot(
        data["true_R_m"],
        data["true_Z_m"],
        color="#00796B",
        linewidth=1.5,
        label="R3 true circle",
    )
    _draw_staircase_wall(axis, data)


def render_physical_rz_figures(
    rigid: dict[str, Any],
    formal_rhs: dict[str, Any],
    output: Any,
    plt: Any,
    a1: Any,
) -> None:
    """Render Figures 6 and 7 without changing any supplied array."""

    figure, axes = plt.subplots(3, 3, figsize=(10.0, 9.2), constrained_layout=True)
    magnitude_norm = Normalize(*rigid["shared_limits"]["magnitude"])
    difference_norm = Normalize(*rigid["shared_limits"]["difference"])
    letters = iter("abcdefghi")
    mesh_old = None
    mesh_positive = None
    mesh_difference = None
    for row, key in enumerate(("N8", "N12", "N16")):
        grid = rigid["grids"][key]
        R, Z, old = _dense_cells(
            grid["old_R_m"],
            grid["old_Z_m"],
            grid["old_magnitude"],
        )
        _, _, positive = _dense_cells(
            grid["candidate_R_m"],
            grid["candidate_Z_m"],
            grid["positive_magnitude"],
        )
        _, _, difference = _dense_cells(
            grid["candidate_R_m"],
            grid["candidate_Z_m"],
            grid["difference"],
        )
        panels = (
            (old, "viridis", magnitude_norm, "old staircase"),
            (positive, "viridis", magnitude_norm, "positive-M0 candidate"),
            (difference, "coolwarm", difference_norm, "positive-M0 minus old"),
        )
        for column, (values, cmap, norm, title) in enumerate(panels):
            axis = axes[row, column]
            mesh = axis.pcolormesh(
                R,
                Z,
                np.ma.masked_invalid(values),
                shading="nearest",
                cmap=cmap,
                norm=norm,
            )
            _draw_walls(axis, grid)
            axis.set_aspect("equal")
            axis.set_xlabel("R [m]")
            if column == 0:
                axis.set_ylabel(f"{key}: Z [m]")
            axis.set_title(title if row == 0 else f"{key}: {title}", fontsize=8)
            a1.style_axis(axis)
            a1.panel_label(axis, next(letters))
            if (row, column) == (0, 0):
                mesh_old = mesh
                axis.legend(fontsize=6.5, loc="upper right")
            elif (row, column) == (0, 1):
                mesh_positive = mesh
            elif (row, column) == (0, 2):
                mesh_difference = mesh
    figure.colorbar(
        mesh_old,
        ax=axes[:, 0],
        shrink=0.82,
        label="symmetric strain [s^-1]",
    )
    figure.colorbar(
        mesh_positive,
        ax=axes[:, 1],
        shrink=0.82,
        label="symmetric strain [s^-1]",
    )
    figure.colorbar(
        mesh_difference,
        ax=axes[:, 2],
        shrink=0.82,
        label="strain difference [s^-1]",
    )
    figure.suptitle(
        "Repair3 rigid rotation: old staircase vs corrected positive-M0 "
        "(physical gate failed)",
        fontsize=10,
    )
    a1.save_figure(figure, output, FIGURE6)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(10.2, 6.0), constrained_layout=True)
    field_specs = (
        ("mom_i_abs", "|rhs(mom_i)| [kg/(m^2 s^2)]", "viridis", "mom_i_abs"),
        ("p_perp_e", "rhs(p_perp_e) [Pa/s]", "coolwarm", "p_perp_e"),
        ("B_phi", "rhs(B_phi) [T/s]", "coolwarm", "B_phi"),
    )
    letters = iter("abcdef")
    meshes: dict[str, Any] = {}
    for row, key in enumerate(("N8", "N12")):
        grid = formal_rhs["grids"][key]
        active = np.asarray(grid["active_mask"], dtype=bool)
        for column, (field, label, cmap, limit_key) in enumerate(field_specs):
            axis = axes[row, column]
            if field == "B_phi":
                values = np.asarray(grid[field], dtype=np.float64)
                vertex_mask = np.zeros(values.shape, dtype=bool)
                vertex_mask[:-1, :-1] |= active
                vertex_mask[1:, :-1] |= active
                vertex_mask[:-1, 1:] |= active
                vertex_mask[1:, 1:] |= active
                mesh = axis.pcolormesh(
                    np.asarray(grid["R_edges_m"]),
                    np.asarray(grid["Z_edges_m"]),
                    np.ma.masked_where(~vertex_mask, values),
                    shading="nearest",
                    cmap=cmap,
                    norm=Normalize(*formal_rhs["shared_limits"][limit_key]),
                )
            else:
                values = np.asarray(grid[field], dtype=np.float64)
                mesh = axis.pcolormesh(
                    np.asarray(grid["R_centres_m"]),
                    np.asarray(grid["Z_centres_m"]),
                    np.ma.masked_where(~active, values),
                    shading="nearest",
                    cmap=cmap,
                    norm=Normalize(*formal_rhs["shared_limits"][limit_key]),
                )
            axis.plot(
                grid["true_R_m"],
                grid["true_Z_m"],
                color="#00796B",
                linewidth=1.4,
                label="true LCFS",
            )
            _draw_staircase_wall(axis, grid)
            axis.set_aspect("equal")
            axis.set_xlabel("R [m]")
            if column == 0:
                axis.set_ylabel(f"{key}: Z [m]")
            axis.set_title(label, fontsize=8)
            a1.style_axis(axis)
            a1.panel_label(axis, next(letters))
            if row == 0:
                meshes[field] = mesh
            if (row, column) == (0, 0):
                axis.legend(fontsize=6.2, loc="upper right")
    for column, (field, label, _, _) in enumerate(field_specs):
        figure.colorbar(
            meshes[field],
            ax=axes[:, column],
            shrink=0.82,
            label=label,
        )
    figure.suptitle(
        "Self-circle full initial formal staircase RHS "
        "(t=0, zero time steps; true LCFS vs old staircase)",
        fontsize=10,
    )
    a1.save_figure(figure, output, FIGURE7)
    plt.close(figure)
