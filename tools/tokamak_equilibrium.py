"""Read-only CPU sampling of externally supplied tokamak equilibrium data.

The scalar functions in this module do not solve an equilibrium or map fields
onto the FOMHD state. They create new float64 NumPy arrays and never mutate
``eq``; callers own the returned arrays and their provenance.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline


class _FluxContour(np.ndarray):
    """R-Z polygon carrying the normalized flux level it was extracted from."""

    def __new__(cls, polygon, flux_level):
        value = np.asarray(polygon, dtype=np.float64).view(cls)
        value.flux_level = float(flux_level)
        return value

    def __array_finalize__(self, source):
        if source is not None:
            self.flux_level = getattr(source, "flux_level", None)


def _source_grid(eq):
    try:
        R = np.asarray(eq["R"], dtype=np.float64)
        Z = np.asarray(eq["Z"], dtype=np.float64)
        psi = np.asarray(eq["psi"], dtype=np.float64)
        psi_axis = float(eq["psi_axis"])
        psi_boundary = float(eq["psi_boundary"])
        magnetic_axis = np.asarray(eq["magnetic_axis"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("equilibrium is missing scalar or array fields") from error

    if R.ndim != 1 or Z.ndim != 1 or R.size < 2 or Z.size < 2:
        raise ValueError("equilibrium R and Z must be one-dimensional grids")
    if (
        not np.isfinite(R).all()
        or not np.isfinite(Z).all()
        or not np.isfinite(psi).all()
    ):
        raise ValueError("equilibrium source arrays must be finite")
    if np.any(R <= 0.0) or np.any(np.diff(R) <= 0.0) or np.any(np.diff(Z) <= 0.0):
        raise ValueError("equilibrium source grid must be positive and increasing")
    if psi.shape != (Z.size, R.size):
        raise ValueError("equilibrium psi must have shape (len(Z), len(R))")
    if (
        not np.isfinite(psi_axis)
        or not np.isfinite(psi_boundary)
        or psi_axis == psi_boundary
    ):
        raise ValueError("equilibrium flux normalization must be finite and nonzero")
    if magnetic_axis.shape != (2,) or not np.isfinite(magnetic_axis).all():
        raise ValueError("equilibrium magnetic axis must be a finite R-Z point")
    if not (
        R[0] <= magnetic_axis[0] <= R[-1]
        and Z[0] <= magnetic_axis[1] <= Z[-1]
    ):
        raise ValueError("equilibrium magnetic axis is outside the source grid")
    return R, Z, psi, psi_axis, psi_boundary, magnetic_axis


def _profiles(eq, size):
    result = {}
    for output_name, source_name in (
        ("F", "F"),
        ("pressure_Pa", "pressure"),
        ("q", "q"),
    ):
        try:
            values = np.asarray(eq[source_name], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"equilibrium is missing valid {source_name} profile") from error
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError(
                f"equilibrium {source_name} profile must be a finite len(R) array"
            )
        result[output_name] = values
    return result


def _query_grid(R, Z, source_R, source_Z, psi_factor):
    try:
        factor = float(psi_factor)
    except (TypeError, ValueError) as error:
        raise ValueError("psi_factor must be an explicit finite nonzero number") from error
    if not np.isfinite(factor) or factor == 0.0:
        raise ValueError("psi_factor must be an explicit finite nonzero number")

    try:
        query_R, query_Z = np.broadcast_arrays(
            np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("query R and Z cannot be broadcast to one shape") from error
    if not np.isfinite(query_R).all() or not np.isfinite(query_Z).all():
        raise ValueError("query R and Z coordinates must be finite")
    inside = (
        (query_R >= source_R[0])
        & (query_R <= source_R[-1])
        & (query_Z >= source_Z[0])
        & (query_Z <= source_Z[-1])
    )
    if not inside.all():
        raise ValueError("query R and Z coordinates are outside the source grid")
    return factor, query_R, query_Z


def sample_equilibrium(eq, R, Z, *, psi_factor):
    """Sample point fields from a G-EQDSK-style equilibrium.

    ``R`` and ``Z`` are broadcast to one shape and interpreted in metres.
    ``psi_normalized`` and ``q`` are dimensionless; ``F`` is T m;
    ``pressure_Pa`` is total MHD pressure in pascals. The returned ``B_R``,
    ``B_phi`` and ``B_Z`` are in tesla when the source obeys those units.
    ``psi_factor`` is the explicit caller-selected conversion to Wb/rad and
    must be finite and nonzero. Query points must be on the source grid and
    their normalized flux must remain in ``[0, 1]``.
    """

    source_R, source_Z, source_psi, psi_axis, psi_boundary, _ = _source_grid(eq)
    profiles = _profiles(eq, source_R.size)
    factor, query_R, query_Z = _query_grid(
        R, Z, source_R, source_Z, psi_factor
    )

    spline = RectBivariateSpline(
        source_Z,
        source_R,
        source_psi,
        kx=min(3, source_Z.size - 1),
        ky=min(3, source_R.size - 1),
    )
    flat_R = query_R.ravel()
    flat_Z = query_Z.ravel()
    psi = spline.ev(flat_Z, flat_R)
    dpsi_dZ = spline.ev(flat_Z, flat_R, dx=1, dy=0)
    dpsi_dR = spline.ev(flat_Z, flat_R, dx=0, dy=1)

    psi_normalized = (psi - psi_axis) / (psi_boundary - psi_axis)
    tolerance = 32.0 * np.finfo(np.float64).eps
    if np.any(psi_normalized < -tolerance) or np.any(
        psi_normalized > 1.0 + tolerance
    ):
        raise ValueError("query normalized flux is outside [0, 1]")
    psi_normalized = np.clip(psi_normalized, 0.0, 1.0)

    profile_x = np.linspace(0.0, 1.0, source_R.size)
    output = {
        "psi_normalized": psi_normalized.reshape(query_R.shape),
        "B_R": (-factor * dpsi_dZ / flat_R).reshape(query_R.shape),
        "B_phi": np.interp(
            psi_normalized, profile_x, profiles["F"]
        ).reshape(query_R.shape)
        / query_R,
        "B_Z": (factor * dpsi_dR / flat_R).reshape(query_R.shape),
        "pressure_Pa": np.interp(
            psi_normalized, profile_x, profiles["pressure_Pa"]
        ).reshape(query_R.shape),
        "F": np.interp(psi_normalized, profile_x, profiles["F"]).reshape(
            query_R.shape
        ),
        "q": np.interp(psi_normalized, profile_x, profiles["q"]).reshape(
            query_R.shape
        ),
    }
    return output


def _signed_area(polygon):
    R = polygon[:, 0]
    Z = polygon[:, 1]
    return 0.5 * np.sum(R * np.roll(Z, -1) - np.roll(R, -1) * Z)


def closed_flux_contour(eq, level=0.8):
    """Extract the unique closed source contour containing the magnetic axis.

    The result is a finite counter-clockwise ``N x 2`` R-Z polygon in metres
    whose dimensionless ``flux_level`` attribute records the requested
    normalized source flux. The level identifies a contour from the input data
    only; it is not asserted to be the last closed flux surface. Matplotlib is
    imported lazily and used only for input preparation. The caller owns the
    returned array.
    """

    R, Z, psi, psi_axis, psi_boundary, magnetic_axis = _source_grid(eq)
    try:
        flux_level = float(level)
    except (TypeError, ValueError) as error:
        raise ValueError("contour level must be a finite value in [0, 1]") from error
    if not np.isfinite(flux_level) or not 0.0 <= flux_level <= 1.0:
        raise ValueError("contour level must be a finite value in [0, 1]")

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.path import Path

    psi_normalized = (psi - psi_axis) / (psi_boundary - psi_axis)
    figure = Figure()
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    contour_set = axes.contour(
        R, Z, psi_normalized, levels=[flux_level]
    )
    point_scale = max(
        abs(R[0]),
        abs(R[-1]),
        abs(Z[0]),
        abs(Z[-1]),
        1.0,
    )
    closure_tolerance = 32.0 * np.finfo(np.float64).eps * point_scale
    containing = []
    open_fragments = 0
    for segment in contour_set.allsegs[0]:
        points = np.asarray(segment, dtype=np.float64)
        if points.size == 0:
            continue
        if not np.isfinite(points).all():
            raise ValueError("Matplotlib returned a nonfinite flux contour")
        if len(points) < 3 or not np.allclose(
            points[0],
            points[-1],
            rtol=0.0,
            atol=closure_tolerance,
        ):
            open_fragments += 1
            continue
        polygon = points[:-1]
        if np.unique(polygon, axis=0).shape[0] < 3:
            raise ValueError("Matplotlib returned a degenerate flux contour")
        if Path(polygon).contains_point(magnetic_axis):
            containing.append(polygon)

    if not containing:
        raise ValueError("no closed flux contour contains the magnetic axis" + ("; open fragments only" if open_fragments else ""))
    if len(containing) != 1:
        raise ValueError("multiple closed flux contours contain the magnetic axis")

    polygon = containing[0].copy()
    area = _signed_area(polygon)
    area_tolerance = (
        64.0
        * polygon.shape[0]
        * np.finfo(np.float64).eps
        * max(abs(polygon[:, 0]).max(), abs(polygon[:, 1]).max(), 1.0) ** 2
    )
    if not np.isfinite(area) or abs(area) <= area_tolerance:
        raise ValueError("flux contour has zero or nonfinite signed area")
    if area < 0.0:
        polygon = polygon[::-1].copy()
    return _FluxContour(polygon, flux_level)
