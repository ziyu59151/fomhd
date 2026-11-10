"""Dense clipped-polygon operators for the A050 ``polygon_dual_v1`` layout."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .backend import Backend
from .operators import OperatorOptions, Operators
from .polygon_geometry import PolygonGeometry
from .types import Failure, operator_check_record


_EPS = np.finfo(np.float64).eps


@dataclass
class PolygonOperatorOptions:
    rank_rtol: float = 1.0e-10
    memory_cap_bytes: int = 512 << 20
    max_condition: float = 1.0e8
    scalar_degree: int = 1
    max_support_cells: int = 24
    max_support_faces: int = 32


def _checked_product(left: int, right: int, name: str) -> int:
    if left < 0 or right < 0:
        raise Failure("Memory", f"{name} size is negative")
    if left and right > sys.maxsize // left:
        raise Failure("Memory", f"{name} size product overflows")
    return left * right


def _checked_sum(left: int, right: int, name: str) -> int:
    if right > sys.maxsize - left:
        raise Failure("Memory", f"{name} size sum overflows")
    return left + right


def _options(
    value: PolygonOperatorOptions | OperatorOptions | dict[str, Any] | None,
) -> PolygonOperatorOptions:
    if value is None:
        return PolygonOperatorOptions()
    if isinstance(value, PolygonOperatorOptions):
        return value
    if isinstance(value, OperatorOptions):
        return PolygonOperatorOptions(
            rank_rtol=value.rank_rtol,
            memory_cap_bytes=value.memory_cap_bytes,
        )
    try:
        return PolygonOperatorOptions(**value)
    except (TypeError, ValueError) as error:
        raise Failure("Operators", f"invalid polygon operator options: {error}") from error


def _validate_options(options: PolygonOperatorOptions) -> None:
    if not isinstance(options.memory_cap_bytes, int) or options.memory_cap_bytes < 0:
        raise Failure("Memory", "operator memory cap must be a non-negative integer")
    if not 0.0 < options.rank_rtol < 1.0:
        raise Failure("Operators", "operator rank tolerance is invalid")
    if not math.isfinite(options.max_condition) or options.max_condition <= 1.0:
        raise Failure("Operators", "maximum reconstruction condition is invalid")
    if options.scalar_degree not in (1, 2):
        raise Failure("Operators", "scalar reconstruction degree must be one or two")
    if options.max_support_cells < 3 or options.max_support_cells > 24:
        raise Failure("Operators", "scalar cell support must lie in [3,24]")
    if options.max_support_faces < 3 or options.max_support_faces > 32:
        raise Failure("Operators", "QB face support must lie in [3,32]")


def _estimate_bytes(
    geometry: PolygonGeometry,
    modes: int,
    options: PolygonOperatorOptions,
) -> int:
    nf = len(geometry.faces.keys)
    nc = len(geometry.cells.keys)
    nv = len(geometry.vertices.keys)
    ne = nf + nc
    nb = nv + nf
    mode_entries = _checked_product(
        modes,
        _checked_sum(
            _checked_sum(
                _checked_product(ne, nc, "polygon operator"),
                _checked_product(nb, ne, "polygon operator"),
                "polygon operator",
            ),
            _checked_sum(
                _checked_sum(
                    _checked_product(nv, nb, "polygon operator"),
                    _checked_product(nc, ne, "polygon operator"),
                    "polygon operator",
                ),
                _checked_product(ne, nb, "polygon operator"),
                "polygon operator",
            ),
            "polygon operator",
        ),
        "polygon operator",
    )
    fixed_entries = _checked_sum(
        _checked_sum(
            _checked_product(nf, nc, "polygon operator"),
            _checked_product(ne, nc, "polygon operator"),
            "polygon operator",
        ),
        _checked_sum(
            _checked_sum(
                _checked_product(3, _checked_product(nc, ne, "polygon operator"), "polygon operator"),
                _checked_product(2, _checked_product(nc, ne, "polygon operator"), "polygon operator"),
                "polygon operator",
            ),
            _checked_product(
                3,
                _checked_product(nc, nb, "polygon operator"),
                "polygon operator",
            ),
            "polygon operator",
        ),
        "polygon operator",
    )
    fixed_entries = _checked_sum(
        fixed_entries,
        _checked_sum(
            _checked_product(nf, nv, "polygon operator"),
            _checked_product(nc, nf, "polygon operator"),
            "polygon operator",
        ),
        "polygon operator",
    )
    fixed_entries = _checked_sum(
        fixed_entries,
        _checked_product(
            2,
            _checked_product(len(geometry.walls), 3 * nc, "polygon operator"),
            "polygon operator",
        ),
        "polygon operator",
    )
    sparse_entries = _checked_sum(
        int(geometry.b1.nnz),
        int(geometry.b2.nnz),
        "polygon operator",
    )
    lsq_work = _checked_product(
        nf + nc,
        _checked_product(options.max_support_faces, 8, "polygon operator"),
        "polygon operator",
    )
    bytes_ = _checked_product(
        _checked_sum(fixed_entries, mode_entries, "polygon operator"),
        16,
        "polygon operator",
    )
    bytes_ = _checked_sum(
        bytes_,
        _checked_product(sparse_entries, 32, "polygon operator"),
        "polygon operator",
    )
    bytes_ = _checked_sum(
        bytes_,
        _checked_product(lsq_work, 16, "polygon operator"),
        "polygon operator",
    )
    # Both the host construction and the backend-owned result coexist at peak.
    return _checked_product(2, bytes_, "polygon operator")


def _basis(points: np.ndarray, degree: int) -> np.ndarray:
    columns = [np.ones(points.shape[0], dtype=np.float64), points[:, 0], points[:, 1]]
    if degree == 2:
        columns.extend((points[:, 0] ** 2, points[:, 0] * points[:, 1], points[:, 1] ** 2))
    return np.column_stack(columns)


def _scaled_equal(left: float, right: float, scale: float = 0.0) -> bool:
    magnitude = max(
        abs(left),
        abs(right),
        abs(scale),
        np.finfo(np.float64).tiny,
    )
    return abs(left - right) <= 4096.0 * _EPS * magnitude


def _regular_cell_geometry(
    geometry: PolygonGeometry,
    cell: int,
    hr: float,
    hz: float,
) -> bool:
    polygon = geometry.cell_polygons[cell]
    if polygon.shape != (4, 2):
        return False
    coordinate_scale = max(
        float(np.max(np.abs(polygon))),
        abs(float(geometry.generator_R[cell])),
        abs(float(geometry.generator_Z[cell])),
        hr,
        hz,
    )
    for edge in range(4):
        vector = polygon[(edge + 1) % 4] - polygon[edge]
        length = float(np.linalg.norm(vector))
        if not length > 0.0:
            return False
        if not (
            _scaled_equal(float(vector[0]), 0.0, length)
            or _scaled_equal(float(vector[1]), 0.0, length)
        ):
            return False
    span_r = float(np.ptp(polygon[:, 0]))
    span_z = float(np.ptp(polygon[:, 1]))
    if not _scaled_equal(span_r, hr, max(span_r, hr)):
        return False
    if not _scaled_equal(span_z, hz, max(span_z, hz)):
        return False
    full_area = hr * hz
    if not _scaled_equal(float(geometry.area_c[cell]), full_area, full_area):
        return False
    return (
        _scaled_equal(
            float(geometry.cells.R[cell]),
            float(geometry.generator_R[cell]),
            coordinate_scale,
        )
        and _scaled_equal(
            float(geometry.cells.Z[cell]),
            float(geometry.generator_Z[cell]),
            coordinate_scale,
        )
    )


def _is_regular_two_owner_face(
    geometry: PolygonGeometry,
    face: int,
) -> bool:
    """Certify a full axis-aligned two-cell Cartesian facet geometrically."""
    if face < 0 or face >= geometry.face_cells.shape[0]:
        return False
    first_cell, second_cell = (
        int(geometry.face_cells[face, 0]),
        int(geometry.face_cells[face, 1]),
    )
    if first_cell < 0 or second_cell < 0:
        return False
    hr = float(geometry.R[1] - geometry.R[0])
    hz = float(geometry.Z[1] - geometry.Z[0])
    if not (
        _regular_cell_geometry(geometry, first_cell, hr, hz)
        and _regular_cell_geometry(geometry, second_cell, hr, hz)
    ):
        return False

    first = np.array(
        (geometry.cells.R[first_cell], geometry.cells.Z[first_cell]),
        dtype=np.float64,
    )
    second = np.array(
        (geometry.cells.R[second_cell], geometry.cells.Z[second_cell]),
        dtype=np.float64,
    )
    separation = second - first
    length = float(np.linalg.norm(separation))
    if not length > 0.0:
        return False
    radial = _scaled_equal(float(separation[0]), 0.0, length)
    axial = _scaled_equal(float(separation[1]), 0.0, length)
    if radial == axial:
        return False
    actual_midpoint = np.array(
        (geometry.faces.R[face], geometry.faces.Z[face]),
        dtype=np.float64,
    )
    centroid_midpoint = 0.5 * (first + second)
    coordinate_scale = max(
        float(np.max(np.abs(actual_midpoint))),
        float(np.max(np.abs(centroid_midpoint))),
        hr,
        hz,
    )
    if any(
        not _scaled_equal(
            float(actual_midpoint[index]),
            float(centroid_midpoint[index]),
            coordinate_scale,
        )
        for index in range(2)
    ):
        return False
    normal = geometry.face_normals[face]
    alignment = float(np.dot(normal, separation / length))
    if not _scaled_equal(alignment, 1.0, 1.0):
        return False
    expected_length = hz if axial else hr
    return _scaled_equal(
        float(geometry.face_lengths[face]),
        expected_length,
        expected_length,
    )


def _scalar_reconstruction(
    geometry: PolygonGeometry,
    settings: PolygonOperatorOptions,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    nf = len(geometry.faces.keys)
    nc = len(geometry.cells.keys)
    regular = np.zeros(nf, dtype=bool)
    if settings.scalar_degree == 1:
        for face in range(nf):
            regular[face] = _is_regular_two_owner_face(geometry, int(face))

    reports: list[dict[str, Any]] = [{} for _ in range(nf)]
    weights = np.zeros((nf, nc), dtype=np.float64)
    polynomial_faces = np.flatnonzero(~regular)
    if polynomial_faces.size:
        cell_points = np.column_stack((geometry.cells.R, geometry.cells.Z))
        face_points = np.column_stack((geometry.faces.R, geometry.faces.Z))
        polynomial_reports, polynomial_weights = _local_weights(
            cell_points,
            face_points[polynomial_faces],
            settings.scalar_degree,
            settings.max_support_cells,
            settings.max_condition,
            settings.rank_rtol,
            "scalar",
        )
        for offset, (face, report) in enumerate(
            zip(polynomial_faces, polynomial_reports)
        ):
            report["query"] = int(face)
            report["kind"] = "polynomial"
            reports[int(face)] = report
            weights[int(face)] = polynomial_weights[offset]
    for face in np.flatnonzero(regular):
        owners = geometry.face_cells[int(face)]
        first, second = int(owners[0]), int(owners[1])
        weights[face, first] = 0.5
        weights[face, second] = 0.5
        reports[int(face)] = {
            "query": int(face),
            "kind": "regular_two_owner",
            "support": 2,
            "rank": 2,
            "condition": 1.0,
            "indices": np.array((first, second), dtype=np.int64),
        }
    return reports, weights, regular


def _local_weights(
    samples: np.ndarray,
    queries: np.ndarray,
    degree: int,
    max_support: int,
    max_condition: float,
    rank_rtol: float,
    label: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if samples.ndim != 2 or samples.shape[1] != 2 or queries.ndim != 2 or queries.shape[1] != 2:
        raise Failure("Shape", f"{label} reconstruction coordinates are invalid")
    if samples.shape[0] < degree + 2:
        raise Failure("Reconstruction", f"{label} has insufficient local samples")
    tree = cKDTree(samples)
    terms = 3 if degree == 1 else 6
    reports: list[dict[str, Any]] = []
    weights: list[np.ndarray] = []
    limit = min(max_support, samples.shape[0])
    for query_id, query in enumerate(queries):
        count = min(limit, max(4, terms + 1))
        accepted = False
        for _ in range(8):
            distances, indices = tree.query(query, k=count)
            indices = np.atleast_1d(indices).astype(np.int64)
            distances = np.atleast_1d(distances).astype(np.float64)
            indices = np.unique(indices)
            scale = float(np.max(distances)) if distances.size else 0.0
            if not scale > 0.0:
                if count == limit:
                    raise Failure(
                        "Reconstruction",
                        f"{label} query {query_id} has no certified local support",
                    )
                count = min(limit, count + 1)
                continue
            scaled = (samples[indices] - query) / scale
            matrix = _basis(scaled, degree)
            column_scale = np.linalg.norm(matrix, axis=0)
            if np.any(column_scale == 0.0):
                if count == limit:
                    raise Failure(
                        "Reconstruction",
                        f"{label} query {query_id} has a rank-deficient local support",
                    )
                count = min(limit, count + 2)
                continue
            matrix = matrix / column_scale
            singular = np.linalg.svd(matrix, compute_uv=False)
            if singular[0] == 0.0:
                condition = math.inf
                rank = 0
            else:
                condition = float(singular[0] / singular[-1])
                rank = int(np.count_nonzero(singular > rank_rtol * singular[0]))
            if rank == terms and condition <= max_condition:
                inverse = np.linalg.pinv(matrix, rcond=rank_rtol)
                local = np.zeros(samples.shape[0], dtype=np.float64)
                local[indices] = inverse[0] / column_scale[0]
                weights.append(local)
                reports.append(
                    {
                        "query": int(query_id),
                        "support": int(indices.size),
                        "rank": rank,
                        "condition": condition,
                        "indices": indices.copy(),
                    }
                )
                accepted = True
                break
            if count == limit:
                break
            count = min(limit, max(count + 2, int(math.ceil(1.5 * count))))
        if not accepted:
            raise Failure(
                "Reconstruction",
                f"{label} query {query_id} is rank deficient or exceeds condition "
                f"{max_condition:.3e} within support {limit}",
            )
    return reports, np.vstack(weights)


def _qb_poloidal_weights(
    samples: np.ndarray,
    tangents: np.ndarray,
    queries: np.ndarray,
    max_support: int,
    max_condition: float,
    rank_rtol: float,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Fit the single face measurement Bt=t_R*BR+t_Z*BZ as a linear vector."""
    tree = cKDTree(samples)
    limit = min(max_support, samples.shape[0])
    reports: list[dict[str, Any]] = []
    radial_rows: list[np.ndarray] = []
    axial_rows: list[np.ndarray] = []
    for query_id, query in enumerate(queries):
        count = min(limit, 8)
        accepted = False
        for _ in range(8):
            distances, indices = tree.query(query, k=count)
            indices = np.unique(np.atleast_1d(indices).astype(np.int64))
            distances = np.atleast_1d(distances).astype(np.float64)
            scale = float(np.max(distances)) if distances.size else 0.0
            if not scale > 0.0:
                if count == limit:
                    break
                count = min(limit, count + 2)
                continue
            local_points = (samples[indices] - query) / scale
            tangent = tangents[indices]
            x = local_points[:, 0]
            z = local_points[:, 1]
            matrix = np.column_stack(
                (
                    tangent[:, 0],
                    tangent[:, 0] * x,
                    tangent[:, 0] * z,
                    tangent[:, 1],
                    tangent[:, 1] * x,
                    tangent[:, 1] * z,
                )
            )
            column_scale = np.linalg.norm(matrix, axis=0)
            if np.any(column_scale == 0.0):
                if count == limit:
                    break
                count = min(limit, max(count + 2, int(math.ceil(1.5 * count))))
                continue
            matrix = matrix / column_scale
            singular = np.linalg.svd(matrix, compute_uv=False)
            rank = int(np.count_nonzero(singular > rank_rtol * singular[0]))
            condition = float(singular[0] / singular[-1])
            if rank == 6 and condition <= max_condition:
                inverse = np.linalg.pinv(matrix, rcond=rank_rtol)
                radial = np.zeros(samples.shape[0], dtype=np.float64)
                axial = np.zeros(samples.shape[0], dtype=np.float64)
                radial[indices] = inverse[0] / column_scale[0]
                axial[indices] = inverse[3] / column_scale[3]
                radial_rows.append(radial)
                axial_rows.append(axial)
                reports.append(
                    {
                        "query": int(query_id),
                        "support": int(indices.size),
                        "rank": rank,
                        "condition": condition,
                        "indices": indices.copy(),
                    }
                )
                accepted = True
                break
            if count == limit:
                break
            count = min(limit, max(count + 2, int(math.ceil(1.5 * count))))
        if not accepted:
            raise Failure(
                "Reconstruction",
                f"QB_Bt query {query_id} is rank deficient or exceeds condition "
                f"{max_condition:.3e} within support {limit}",
            )
    return reports, np.vstack(radial_rows), np.vstack(axial_rows)


def _dense_diagonal(values: np.ndarray) -> np.ndarray:
    return np.diag(np.asarray(values, dtype=np.float64))


def _immutable_real(backend: Backend, values: Any) -> Any:
    result = backend.real_array(values)
    if backend.name == "cpu":
        result.flags.writeable = False
    return result


def _check(
    backend: Backend,
    records: list[SimpleNamespace],
    name: str,
    residual: Any,
    scale: float,
    terms: int,
    unit: str,
) -> None:
    value = backend.max_abs(residual)
    bound = 512.0 * _EPS * max(1, terms) * max(scale, 1.0)
    if not math.isfinite(value) or not math.isfinite(scale) or value > bound:
        raise Failure(
            "Operators",
            f"polygon operator identity failed: {name} residual={value:.3e} "
            f"bound={bound:.3e}",
        )
    records.append(
        operator_check_record(
            name=name,
            unit=unit,
            mode=0,
            residual=value,
            scale=scale,
            roundoff_bound=bound,
        )
    )


def _legacy_fields(
    geometry: PolygonGeometry,
    backend: Backend,
) -> dict[str, Any]:
    nf = len(geometry.faces.keys)
    nc = len(geometry.cells.keys)
    nv = len(geometry.vertices.keys)
    empty = np.zeros((0, 0), dtype=np.complex128)
    fields: dict[str, Any] = {
        "b1r": backend.asarray(geometry.b1.toarray()),
        "b1z": backend.asarray(np.zeros((0, nv), dtype=np.complex128)),
        "b2r": backend.asarray(geometry.b2.toarray()),
        "b2z": backend.asarray(np.zeros((nc, 0), dtype=np.complex128)),
        "PR_r": backend.asarray(empty),
        "PZ_r": backend.asarray(empty),
        "PR_z": backend.asarray(empty),
        "PZ_z": backend.asarray(empty),
        "PR_v": backend.asarray(empty),
        "PZ_v": backend.asarray(empty),
    }
    for name in ("PR_r", "PZ_r", "PR_z", "PZ_z", "PR_v", "PZ_v"):
        fields[f"{name}_factor"] = None
    return fields


def build_polygon_operators(
    geometry: PolygonGeometry,
    K: int,
    options: PolygonOperatorOptions | OperatorOptions | dict[str, Any] | None = None,
    backend: Backend | None = None,
) -> Operators:
    """Build the bounded dense polygon_dual_v1 operator oracle."""
    settings = _options(options)
    _validate_options(settings)
    if type(K) is not int or K < 0:
        raise Failure("Operators", "polygon operator K must be a non-negative integer")
    if not isinstance(geometry, PolygonGeometry) or getattr(geometry, "kind", None) != "polygon_dual_v1":
        raise Failure("Operators", "polygon operators require a polygon_dual_v1 geometry")
    backend = backend or Backend("cpu")
    if not isinstance(backend, Backend):
        raise Failure("Backend", "polygon operators require an explicit Backend")
    modes = 2 * K + 1
    nf = len(geometry.faces.keys)
    nc = len(geometry.cells.keys)
    nv = len(geometry.vertices.keys)
    ne = nf + nc
    nb = nv + nf
    estimate = _estimate_bytes(geometry, modes, settings)
    if settings.memory_cap_bytes and estimate > settings.memory_cap_bytes:
        raise Failure(
            "Memory",
            "operator_matrices memory budget exceeded: "
            f"requested={estimate} limit={settings.memory_cap_bytes} "
            f"shapes=(Nf={nf},Nc={nc},Nv={nv},K={K})",
        )

    face_owner_counts = np.bincount(geometry.face_cells[:, 0], minlength=nc)
    if int(np.max(face_owner_counts)) > settings.max_support_faces:
        raise Failure(
            "Operators",
            f"cell face support {int(np.max(face_owner_counts))} exceeds "
            f"{settings.max_support_faces}",
        )
    cell_points = np.column_stack((geometry.cells.R, geometry.cells.Z))
    face_points = np.column_stack((geometry.faces.R, geometry.faces.Z))
    vertex_points = np.column_stack((geometry.vertices.R, geometry.vertices.Z))
    scalar_reports, S_host, regular_faces = _scalar_reconstruction(
        geometry,
        settings,
    )

    phi_reports, qb_phi = _local_weights(
        vertex_points,
        cell_points,
        1,
        settings.max_support_cells,
        settings.max_condition,
        settings.rank_rtol,
        "QB_Bphi",
    )
    qb_face_reports, br_face_weights, bz_face_weights = _qb_poloidal_weights(
        face_points,
        geometry.face_tangents,
        cell_points,
        settings.max_support_faces,
        settings.max_condition,
        settings.rank_rtol,
    )

    g_host = -np.diag(1.0 / geometry.face_distances) @ geometry.b2.toarray().T
    curvature = geometry.M0 / geometry.area_c
    i_k = 1j * K
    G_host = np.zeros((ne, nc), dtype=np.complex128)
    G_host[:nf] = g_host
    G_host[nf:] = np.diag(i_k / curvature)

    C_host = np.zeros((nb, ne), dtype=np.complex128)
    C_host[:nv, :nf] = (
        -np.diag(1.0 / geometry.av) @ geometry.b1.toarray().T @ _dense_diagonal(geometry.face_distances)
    )
    C_host[nv:, :nf] = np.diag(-i_k / geometry.Rbar_r)
    C_host[nv:, nf:] = (
        np.diag(1.0 / geometry.Rbar_r)
        @ g_host
        @ _dense_diagonal(curvature)
    )

    DB_host = np.zeros((nv, nb), dtype=np.complex128)
    DB_host[:, :nv] = np.diag(i_k * geometry.av / geometry.wv)
    DB_host[:, nv:] = (
        -np.diag(1.0 / geometry.wv)
        @ geometry.b1.toarray().T
        @ _dense_diagonal(geometry.face_distances * geometry.Rbar_r)
    )

    M1 = np.concatenate(
        (geometry.Rbar_r * geometry.face_lengths * geometry.face_distances, geometry.M0)
    )
    MB = np.concatenate(
        (geometry.wv, geometry.Rbar_r * geometry.face_lengths * geometry.face_distances)
    )
    G_modes: list[Any] = []
    C_modes: list[Any] = []
    DB_modes: list[Any] = []
    DJ_modes: list[Any] = []
    Cdag_modes: list[Any] = []
    for mode in range(-K, K + 1):
        mode_k = 1j * mode
        g = G_host.copy()
        g[nf:] = np.diag(mode_k / curvature)
        c = C_host.copy()
        c[nv:, :nf] = np.diag(-mode_k / geometry.Rbar_r)
        db = DB_host.copy()
        db[:, :nv] = np.diag(mode_k * geometry.av / geometry.wv)
        dj = -np.diag(1.0 / geometry.M0) @ g.conj().T @ _dense_diagonal(M1)
        cdag = np.diag(1.0 / M1) @ c.conj().T @ _dense_diagonal(MB)
        G_modes.append(backend.asarray(g))
        C_modes.append(backend.asarray(c))
        DB_modes.append(backend.asarray(db))
        DJ_modes.append(backend.asarray(dj))
        Cdag_modes.append(backend.asarray(cdag))

    operators = Operators(
        K=K,
        Nc=nc,
        Ne=ne,
        Nb=nb,
        M0=backend.real_array(geometry.M0),
        M1=backend.real_array(M1),
        MB=backend.real_array(MB),
        Rbar_r=backend.real_array(geometry.Rbar_r),
        Rbar_v=backend.real_array(geometry.Rbar_v),
        G=G_modes,
        C=C_modes,
        DB=DB_modes,
        DJ=DJ_modes,
        Cdag=Cdag_modes,
        compute=backend,
        finite_mass_available=False,
        finite_mass_unavailable_reason=(
            "polygon_dual_v1 first unit is massless-only; no Poisson/Darwin storage"
        ),
        omitted_finite_mass=(
            "Lphi",
            "phi_factor",
            "poisson_spectra",
            "Bc_full",
            "Bc",
            "selected_constraint_rows",
        ),
        finite_mass_omitted_bytes=0,
    )

    S = backend.real_array(S_host)
    operators.S = S
    operators.scalar_I = backend.asarray(np.vstack((S_host, np.eye(nc))))
    I_host = np.zeros((ne, 3 * nc), dtype=np.complex128)
    I_host[:nf, :nc] = (
        geometry.face_normals[:, 0][:, None] * S_host
    )
    I_host[nf:, nc : 2 * nc] = np.eye(nc)
    I_host[:nf, 2 * nc :] = (
        geometry.face_normals[:, 1][:, None] * S_host
    )
    operators.I = backend.asarray(I_host)
    node_weights = np.tile(geometry.M0, 3)
    H_host = (
        np.diag(1.0 / node_weights)
        @ I_host.conj().T
        @ _dense_diagonal(M1)
    )
    operators.H = backend.asarray(H_host)

    QB_host = np.zeros((3 * nc, nb), dtype=np.complex128)
    QB_host[:nc, nv:] = br_face_weights
    QB_host[nc : 2 * nc, :nv] = qb_phi
    QB_host[2 * nc :, nv:] = bz_face_weights
    operators.QB = backend.asarray(QB_host)

    boundary_faces = np.flatnonzero(geometry.face_cells[:, 1] < 0)
    T_full_host = I_host[boundary_faces, :]
    operators.T_full = backend.asarray(T_full_host)
    operators.selected_wall_rows = backend.independent_rows(
        operators.T_full, settings.rank_rtol
    )
    operators.T = backend.asarray(T_full_host[operators.selected_wall_rows])
    operators.Dface = operators.DJ[K][:, :nf]
    operators.DR = operators.DJ[K][:, :nf]
    operators.DZ = backend.asarray(np.zeros((nc, 0), dtype=np.complex128))
    operators.curvature_radius = backend.real_array(geometry.cells.R)
    operators.topology_flux_row = backend.asarray(
        np.concatenate((geometry.av, np.zeros(nf)))[None, :]
    )
    empty_columns = modes * ne
    operators.Bc_full = backend.asarray(np.zeros((0, empty_columns), dtype=np.complex128))
    operators.Bc = operators.Bc_full
    operators.selected_constraint_rows = []
    legacy = _legacy_fields(geometry, backend)
    for name, value in legacy.items():
        setattr(operators, name, value)

    conditions = [
        report["condition"]
        for report in scalar_reports + phi_reports + qb_face_reports
    ]
    operators.reconstruction = SimpleNamespace(
        degree=settings.scalar_degree,
        max_support_cells=max(report["support"] for report in scalar_reports),
        min_support_cells=min(report["support"] for report in scalar_reports),
        min_rank=min(report["rank"] for report in scalar_reports),
        max_condition=max(float(value) for value in conditions),
        min_condition=min(float(value) for value in conditions),
        regular_faces=int(np.count_nonzero(regular_faces)),
        polynomial_faces=int(nf - np.count_nonzero(regular_faces)),
        kinds=(
            ("regular_two_owner", "polynomial")
            if settings.scalar_degree == 1
            else ("polynomial",)
        ),
        reports=scalar_reports,
    )
    operators.qb_diagnostics = SimpleNamespace(
        max_support_faces=max(
            report["support"] for report in qb_face_reports
        ),
        max_support_vertices=max(report["support"] for report in phi_reports),
        max_condition=max(
            float(report["condition"])
            for report in phi_reports + qb_face_reports
        ),
        reports=phi_reports + qb_face_reports,
    )
    operators.kind = "polygon_dual_v1"
    operators.geometry_kind = "polygon_dual_v1"
    operators.matrix_layout_version = "polygon_dual_v1"
    operators.Nf = nf
    operators.Nv = nv
    operators.face_normals_backend = _immutable_real(backend, geometry.face_normals)
    operators.cell_radius_backend = _immutable_real(backend, geometry.cells.R)

    records: list[SimpleNamespace] = []
    for index, mode in enumerate(range(-K, K + 1)):
        cg = backend.mm(operators.C[index], operators.G[index])
        dbc = backend.mm(operators.DB[index], operators.C[index])
        cg_scale = backend.max_abs(operators.C[index]) * backend.max_abs(
            operators.G[index]
        )
        dbc_scale = backend.max_abs(operators.DB[index]) * backend.max_abs(
            operators.C[index]
        )
        _check(backend, records, f"CG_m{mode}", cg, cg_scale, ne, "1/m^2")
        _check(backend, records, f"DB_C_m{mode}", dbc, dbc_scale, nb, "1/m^2")
        gradient_adjoint = backend.asarray(
            operators.M0[:, None] * operators.DJ[index]
            + operators.G[index].T.conj() * operators.M1[None, :]
        )
        _check(
            backend,
            records,
            f"gradient_divergence_adjoint_m{mode}",
            gradient_adjoint,
            backend.max_abs(operators.G[index]) * backend.max_abs(operators.M1),
            2,
            "m^2",
        )
    transfer = backend.asarray(
        backend.xp.tile(operators.M0, 3)[:, None] * operators.H
        - operators.I.T.conj() * operators.M1[None, :]
    )
    _check(
        backend,
        records,
        "transfer_adjoint",
        transfer,
        backend.max_abs(operators.I) * backend.max_abs(operators.M1),
        2,
        "m^3",
    )
    topology = backend.mm(
        operators.topology_flux_row, operators.C[K]
    )
    _check(
        backend,
        records,
        "topology_curl",
        topology,
        backend.max_abs(operators.topology_flux_row)
        * backend.max_abs(operators.C[K]),
        nb,
        "m",
    )
    operators.setup_checks = records
    return operators
