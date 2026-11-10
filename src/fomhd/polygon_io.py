"""Native cochain I/O for the selected ``polygon_dual_v1`` layout."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
from scipy import sparse


LAYOUT_KIND = "polygon_dual_v1"
LAYOUT_VERSION = 1
LAYOUT_SCHEMA = f"{LAYOUT_KIND}:v1"
GEOMETRY_RECONSTRUCTION_VERSION = 1

_REQUIRED_GEOMETRY = (
    "cells",
    "faces",
    "vertices",
    "generator_R",
    "generator_Z",
    "cell_polygons",
    "face_vertices",
    "face_cells",
    "face_normals",
    "face_tangents",
    "face_lengths",
    "face_distances",
    "M0",
    "M1",
    "MB",
    "av",
    "wv",
    "Rbar_v",
    "b1",
    "b2",
)


def _require_geometry(geometry):
    kind = getattr(geometry, "kind", None)
    if kind != LAYOUT_KIND:
        raise ValueError(
            f"native polygon I/O requires kind={LAYOUT_KIND!r}, got {kind!r}"
        )
    missing = [name for name in _REQUIRED_GEOMETRY if not hasattr(geometry, name)]
    if missing:
        raise ValueError(
            "polygon geometry is missing required native entities: "
            + ", ".join(missing)
        )
    counts = (
        len(geometry.cells.R),
        len(geometry.faces.R),
        len(geometry.vertices.R),
    )
    if any(type(value) is not int or value <= 0 for value in counts):
        raise ValueError("polygon geometry entity tables must be nonempty")
    expected = {
        "face_vertices": (counts[1], 2),
        "face_cells": (counts[1], 2),
        "face_normals": (counts[1], 2),
        "face_tangents": (counts[1], 2),
        "face_lengths": (counts[1],),
        "face_distances": (counts[1],),
        "M1": (counts[1] + counts[0],),
        "MB": (counts[2] + counts[1],),
        "av": (counts[2],),
        "wv": (counts[2],),
        "Rbar_v": (counts[2],),
    }
    for name, shape in expected.items():
        value = np.asarray(getattr(geometry, name))
        if value.shape != shape:
            raise ValueError(
                f"polygon geometry {name} shape {value.shape} != {shape}"
            )
    if len(geometry.cell_polygons) != counts[0]:
        raise ValueError("cell_polygons must have one entry per cell")
    for name in (
        "generator_R",
        "generator_Z",
        "M0",
    ):
        if np.asarray(getattr(geometry, name)).shape != (counts[0],):
            raise ValueError(f"polygon geometry {name} has the wrong shape")
    return counts


def _jsonable(value):
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, complex):
        if value.imag != 0:
            raise ValueError("complex geometry identity is unsupported")
        return float(value.real)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "R") and hasattr(value, "Z"):
        return [float(value.R), float(value.Z)]
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [_jsonable(item) for item in value.ravel()]
        return np.asarray(value).tolist()
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return str(value)


def _canonical_json(value):
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _hash_numeric(digest, label, value):
    array = np.asarray(value)
    if array.dtype == object or not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"polygon geometry {label} is not numeric")
    if np.issubdtype(array.dtype, np.inexact) and not np.isfinite(array).all():
        raise ValueError(f"polygon geometry {label} contains nonfinite values")
    canonical = np.ascontiguousarray(
        array.astype(array.dtype.newbyteorder("<"), copy=False)
    )
    digest.update(label.encode("ascii"))
    digest.update(repr(canonical.shape).encode("ascii"))
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(canonical.tobytes(order="C"))


def _hash_bytes(digest, label, value):
    raw = value if isinstance(value, bytes) else bytes(value)
    array = np.frombuffer(raw, dtype=np.uint8)
    digest.update(label.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))


def _hash_sparse(digest, label, value):
    if sparse.issparse(value):
        matrix = value.tocsr(copy=True)
        matrix.sort_indices()
        _hash_numeric(digest, f"{label}.shape", np.asarray(matrix.shape))
        _hash_numeric(digest, f"{label}.indptr", matrix.indptr)
        _hash_numeric(digest, f"{label}.indices", matrix.indices)
        _hash_numeric(digest, f"{label}.data", matrix.data)
        return
    _hash_numeric(digest, label, value)


def _entity_keys(table, expected):
    keys = getattr(table, "keys", None)
    if keys is None:
        return list(range(expected))
    if len(keys) != expected:
        raise ValueError("polygon entity key count does not match its table")
    return list(keys)


def _polygon_array(geometry):
    polygon = getattr(geometry, "polygon_input", None)
    if polygon is None:
        polygon = getattr(geometry, "polygon", None)
    if polygon is None:
        raise ValueError("polygon geometry input contour is missing")
    return np.asarray(
        [[point.R, point.Z] for point in polygon], dtype=np.float64
    )


def _cell_key_table(geometry, count):
    keys = _entity_keys(geometry.cells, count)
    try:
        result = np.asarray(keys, dtype=np.int64)
    except (TypeError, ValueError):
        return _canonical_json(keys)
    if result.shape != (count, 2):
        return _canonical_json(keys)
    return result


def native_entity_metadata(geometry):
    nc, nf, nv = _require_geometry(geometry)
    return {
        "layout_kind": LAYOUT_KIND,
        "layout_schema": LAYOUT_SCHEMA,
        "reconstruction_version": GEOMETRY_RECONSTRUCTION_VERSION,
        "cell_count": nc,
        "face_count": nf,
        "vertex_count": nv,
        "cell_keys": _jsonable(_entity_keys(geometry.cells, nc)),
        "face_keys": _jsonable(_entity_keys(geometry.faces, nf)),
        "vertex_keys": _jsonable(_entity_keys(geometry.vertices, nv)),
        "cell_frames": ["polygon_area_centroid"],
        "face_frames": ["plane_normal_facet_midpoint"],
        "vertex_frames": ["orthogonal_dual_vertex"],
        "E_layout": ["normal_faces", "phi_cells"],
        "B_layout": ["phi_vertices", "tangential_faces"],
    }


def geometry_fingerprint(geometry):
    nc, nf, nv = _require_geometry(geometry)
    digest = hashlib.sha256()
    digest.update(LAYOUT_KIND.encode("ascii"))
    digest.update(LAYOUT_SCHEMA.encode("ascii"))
    digest.update(
        repr(GEOMETRY_RECONSTRUCTION_VERSION).encode("ascii")
    )
    _hash_numeric(digest, "polygon_input", _polygon_array(geometry))
    polygon = getattr(geometry, "polygon", None)
    if polygon is not None:
        _hash_numeric(
            digest,
            "polygon",
            np.asarray(
                [[point.R, point.Z] for point in polygon], dtype=np.float64
            ),
        )
    for index, cell in enumerate(geometry.cell_polygons):
        value = np.asarray(cell, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != 2 or value.shape[0] < 3:
            raise ValueError(f"cell_polygons[{index}] is malformed")
        _hash_numeric(digest, f"cell_polygon.{index}", value)
    _hash_numeric(digest, "cell_keys", _cell_key_table(geometry, nc))
    _hash_bytes(
        digest, "face_keys", _canonical_json(_entity_keys(geometry.faces, nf))
    )
    _hash_bytes(
        digest,
        "vertex_keys",
        _canonical_json(_entity_keys(geometry.vertices, nv)),
    )
    for name in (
        "generator_R",
        "generator_Z",
        "face_vertices",
        "face_cells",
        "face_normals",
        "face_tangents",
        "face_lengths",
        "face_distances",
        "M0",
        "M1",
        "MB",
        "av",
        "wv",
        "Rbar_v",
    ):
        _hash_numeric(digest, name, getattr(geometry, name))
    _hash_sparse(digest, "b1", geometry.b1)
    _hash_sparse(digest, "b2", geometry.b2)
    _hash_numeric(digest, "cell_R", geometry.cells.R)
    _hash_numeric(digest, "cell_Z", geometry.cells.Z)
    _hash_numeric(digest, "face_R", geometry.faces.R)
    _hash_numeric(digest, "face_Z", geometry.faces.Z)
    _hash_numeric(digest, "vertex_R", geometry.vertices.R)
    _hash_numeric(digest, "vertex_Z", geometry.vertices.Z)
    return digest.hexdigest()


def _mode_count(K):
    if type(K) is not int or K < 0:
        raise ValueError("retained mode index K must be a nonnegative integer")
    return 2 * K + 1


def _host_array(value, name):
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be an explicit host numpy ndarray")
    return value


def _compact_complex(value, shape, name):
    array = _host_array(value, name)
    if array.dtype != np.dtype(np.complex128):
        raise ValueError(f"{name} must have complex128 dtype")
    expected = int(np.prod(shape))
    if array.size != expected:
        raise ValueError(
            f"{name} shape mismatch: got size {array.size}, expected {expected}"
        )
    if array.shape != shape and array.ndim != 1:
        raise ValueError(
            f"{name} shape mismatch: got {array.shape}, expected {shape}"
        )
    return array.reshape(shape)


def _validate_complex(
    value,
    shape,
    modes,
    K,
    name,
    *,
    mode_order=True,
):
    array = _host_array(value, name)
    if array.dtype != np.dtype(np.complex128):
        raise ValueError(f"{name} must have complex128 dtype")
    if array.shape != shape or array.size != modes * int(np.prod(shape[1:])):
        raise ValueError(f"{name} shape mismatch: got {array.shape}, expected {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    if not mode_order:
        return array
    matrix = array.reshape(modes, -1)
    scale = max(float(np.max(np.abs(matrix), initial=0.0)), 1.0e-300)
    tolerance = 512.0 * np.finfo(np.float64).eps * scale
    if np.any(np.abs(matrix[K].imag) > tolerance):
        raise ValueError(f"{name} violates mode order: central mode is not real")
    for mode in range(1, K + 1):
        defect = np.abs(matrix[K - mode] - np.conj(matrix[K + mode]))
        if np.any(defect > tolerance):
            raise ValueError(
                f"{name} violates mode order: modes -{mode} and +{mode} "
                "are not conjugates"
            )
    return array


def _metadata_arrays(geometry, fingerprint):
    nc, nf, nv = geometry.cells.R.size, geometry.faces.R.size, geometry.vertices.R.size
    metadata = native_entity_metadata(geometry)
    arrays = {
        "layout_schema": np.asarray(LAYOUT_SCHEMA),
        "geometry_fingerprint": np.asarray(fingerprint),
        "native_entity_metadata": np.asarray(
            json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
        "cell_count": np.asarray(nc, dtype=np.int64),
        "face_count": np.asarray(nf, dtype=np.int64),
        "vertex_count": np.asarray(nv, dtype=np.int64),
        "cell_entity_order": np.arange(nc, dtype=np.int64),
        "face_entity_order": np.arange(nf, dtype=np.int64),
        "vertex_entity_order": np.arange(nv, dtype=np.int64),
        "generator_R": np.asarray(geometry.generator_R, dtype=np.float64).copy(),
        "generator_Z": np.asarray(geometry.generator_Z, dtype=np.float64).copy(),
    }
    cell_keys = _cell_key_table(geometry, nc)
    if isinstance(cell_keys, np.ndarray):
        arrays["cell_entity_keys"] = cell_keys
    return arrays


def _verify_metadata(arrays, geometry, K):
    for name in (
        "layout_schema",
        "geometry_fingerprint",
        "native_entity_metadata",
    ):
        if name not in arrays:
            raise ValueError(f"native checkpoint {name} is missing")
    schema = np.asarray(arrays["layout_schema"])
    if schema.shape != () or str(schema) != LAYOUT_SCHEMA:
        raise ValueError(
            f"native layout_schema mismatch: got {schema!r}, "
            f"expected {LAYOUT_SCHEMA!r}"
        )
    fingerprint = np.asarray(arrays["geometry_fingerprint"])
    expected = geometry_fingerprint(geometry)
    if fingerprint.shape != () or str(fingerprint) != expected:
        raise ValueError(
            "native geometry fingerprint mismatch; layout, topology, "
            "entity order, normal, incidence or measure changed"
        )
    nc, nf, nv = _require_geometry(geometry)
    for name, count in (
        ("cell_count", nc),
        ("face_count", nf),
        ("vertex_count", nv),
    ):
        if name not in arrays:
            raise ValueError(f"native entity metadata {name} is missing")
        value = np.asarray(arrays[name])
        if value.shape != () or int(value) != count:
            raise ValueError(f"native entity metadata {name} mismatch")
    try:
        stored = json.loads(str(np.asarray(arrays["native_entity_metadata"])))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("native_entity_metadata is not valid JSON") from error
    if stored != native_entity_metadata(geometry):
        raise ValueError("native_entity_metadata does not match geometry")
    modes = _mode_count(K)
    return modes, nc, nf, nv


def encode_state(state, geometry):
    """Encode one native graph state into compact mode-major cochains."""
    nc, nf, nv = _require_geometry(geometry)
    if len(state.fluid) != 2:
        raise ValueError("state must contain ion and electron fluids")
    ion, electron = state.fluid
    ion_n = _host_array(ion.n, "n_i")
    if ion_n.ndim == 2 and ion_n.shape[1] == nc:
        modes = ion_n.shape[0]
    elif ion_n.ndim == 1 and ion_n.size % nc == 0:
        modes = ion_n.size // nc
    else:
        raise ValueError("n_i shape mismatch")
    if modes < 1 or modes % 2 != 1:
        raise ValueError("n_i must use an odd mode-major shape")
    K = modes // 2
    if state.step < 0 or state.generation < 0 or not np.isfinite(state.time):
        raise ValueError("state identity is invalid")

    arrays = {
        "n_i": _validate_complex(
            _compact_complex(ion.n, (modes, nc), "n_i"),
            (modes, nc),
            modes,
            K,
            "n_i",
        ),
        "mom_i": _validate_complex(
            _compact_complex(ion.mom, (modes, 3, nc), "mom_i"),
            (modes, 3, nc),
            modes,
            K,
            "mom_i",
        ),
        "p_i": _validate_complex(
            _compact_complex(ion.perp, (modes, nc), "p_i"),
            (modes, nc),
            modes,
            K,
            "p_i",
        ),
        "n_e": _validate_complex(
            _compact_complex(electron.n, (modes, nc), "n_e"),
            (modes, nc),
            modes,
            K,
            "n_e",
        ),
        "mom_e": _validate_complex(
            _compact_complex(electron.mom, (modes, 3, nc), "mom_e"),
            (modes, 3, nc),
            modes,
            K,
            "mom_e",
        ),
        "p_perp_e": _validate_complex(
            _compact_complex(electron.perp, (modes, nc), "p_perp_e"),
            (modes, nc),
            modes,
            K,
            "p_perp_e",
        ),
        "p_parallel_e": _validate_complex(
            _compact_complex(
                electron.parallel, (modes, nc), "p_parallel_e"
            ),
            (modes, nc),
            modes,
            K,
            "p_parallel_e",
        ),
    }
    if np.asarray(ion.parallel).size != 0:
        raise ValueError("ion parallel pressure must be empty in this layout")

    b_host = _host_array(state.B, "B state")
    if b_host.dtype != np.dtype(np.complex128):
        raise ValueError("B state must have complex128 dtype")
    if b_host.size != modes * (nv + nf):
        raise ValueError("B state extent mismatch")
    b = _validate_complex(
        b_host.reshape(modes, nv + nf),
        (modes, nv + nf),
        modes,
        K,
        "B state",
    )
    arrays["B_phi_native"] = b[:, :nv].copy()
    arrays["B_t_native"] = b[:, nv:].copy()
    arrays.update(_metadata_arrays(geometry, geometry_fingerprint(geometry)))
    return arrays


def decode_state(arrays, geometry, K):
    """Decode compact native cochains after strict identity validation."""
    from .types import FluidState, State

    modes, nc, nf, nv = _verify_metadata(arrays, geometry, K)
    required = (
        "n_i",
        "mom_i",
        "p_i",
        "n_e",
        "mom_e",
        "p_perp_e",
        "p_parallel_e",
        "B_phi_native",
        "B_t_native",
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"native checkpoint field is missing: {missing[0]}")
    values = {
        "n_i": _validate_complex(
            arrays["n_i"], (modes, nc), modes, K, "n_i"
        ),
        "mom_i": _validate_complex(
            arrays["mom_i"], (modes, 3, nc), modes, K, "mom_i"
        ),
        "p_i": _validate_complex(
            arrays["p_i"], (modes, nc), modes, K, "p_i"
        ),
        "n_e": _validate_complex(
            arrays["n_e"], (modes, nc), modes, K, "n_e"
        ),
        "mom_e": _validate_complex(
            arrays["mom_e"], (modes, 3, nc), modes, K, "mom_e"
        ),
        "p_perp_e": _validate_complex(
            arrays["p_perp_e"], (modes, nc), modes, K, "p_perp_e"
        ),
        "p_parallel_e": _validate_complex(
            arrays["p_parallel_e"], (modes, nc), modes, K, "p_parallel_e"
        ),
        "B_phi_native": _validate_complex(
            arrays["B_phi_native"],
            (modes, nv),
            modes,
            K,
            "B_phi_native",
        ),
        "B_t_native": _validate_complex(
            arrays["B_t_native"],
            (modes, nf),
            modes,
            K,
            "B_t_native",
        ),
    }
    fluid = (
        FluidState(
            n=values["n_i"].reshape(-1).copy(),
            mom=values["mom_i"].reshape(-1).copy(),
            perp=values["p_i"].reshape(-1).copy(),
            parallel=np.zeros(0, dtype=np.complex128),
        ),
        FluidState(
            n=values["n_e"].reshape(-1).copy(),
            mom=values["mom_e"].reshape(-1).copy(),
            perp=values["p_perp_e"].reshape(-1).copy(),
            parallel=values["p_parallel_e"].reshape(-1).copy(),
        ),
    )
    return State(
        fluid=fluid,
        B=np.concatenate(
            (values["B_phi_native"], values["B_t_native"]), axis=1
        ).reshape(-1).copy(),
        step=0,
        generation=1,
        time=0.0,
    )


def encode_vertex(vector, geometry, K):
    nc, nf, nv = _require_geometry(geometry)
    del nc, nf
    modes = _mode_count(K)
    value = _compact_complex(vector, (modes, nv), "vertex cochain")
    return _validate_complex(
        value, (modes, nv), modes, K, "vertex cochain"
    ).copy()


def decode_vertex(array, geometry, K):
    nc, nf, nv = _require_geometry(geometry)
    del nc, nf
    modes = _mode_count(K)
    value = _compact_complex(array, (modes, nv), "vertex cochain")
    return (
        _validate_complex(
            value, (modes, nv), modes, K, "vertex cochain"
        )
        .reshape(-1)
        .copy()
    )


def _operator_metadata(operators):
    metadata = {
        "layout_kind": LAYOUT_KIND,
        "layout_version": LAYOUT_VERSION,
        "geometry_reconstruction_version": GEOMETRY_RECONSTRUCTION_VERSION,
        "matrix_layout_version": getattr(
            operators, "matrix_layout_version", None
        ),
        "reconstruction_version": getattr(
            operators, "reconstruction_version", None
        ),
    }
    for owner_name in ("reconstruction", "qb_diagnostics"):
        owner = getattr(operators, owner_name, None)
        if owner is None:
            metadata[owner_name] = None
            continue
        metadata[owner_name] = {
            name: _jsonable(value)
            for name, value in vars(owner).items()
            if name != "reports"
            and not isinstance(value, (np.ndarray, sparse.spmatrix))
        }
    for name in (
        "scalar_degree",
        "max_support_cells",
        "max_support_faces",
        "max_condition",
    ):
        if hasattr(operators, name):
            metadata[name] = _jsonable(getattr(operators, name))
    return metadata


def _hash_operator_value(digest, label, value, backend):
    if sparse.issparse(value):
        _hash_sparse(digest, label, value)
        return
    if backend is not None:
        value = backend.to_host(value)
    if not isinstance(value, np.ndarray):
        raise ValueError(
            f"operator {label} requires an explicit backend.to_host export"
        )
    _hash_numeric(digest, label, value)


def operator_fingerprint(operators, backend=None):
    """Hash declared layout plus the actual bounded operator payloads."""
    geometry = getattr(operators, "geometry", None)
    if geometry is not None:
        attached = geometry_fingerprint(geometry)
    else:
        attached = getattr(operators, "geometry_fingerprint", None)
    if not attached:
        raise ValueError(
            "polygon operators must attach geometry or geometry_fingerprint"
        )
    digest = hashlib.sha256()
    digest.update(b"polygon-operator-payload-v2")
    digest.update(LAYOUT_KIND.encode("ascii"))
    digest.update(str(LAYOUT_VERSION).encode("ascii"))
    digest.update(str(attached).encode("ascii"))
    metadata = _operator_metadata(operators)
    _hash_bytes(digest, "operator_metadata", _canonical_json(metadata))
    matrix = getattr(operators, "matrix_fingerprint", None)
    if matrix:
        _hash_bytes(
            digest, "matrix_fingerprint", str(matrix).encode("ascii")
        )
    for name in ("G", "C", "DB", "DJ", "Cdag"):
        group = getattr(operators, name, None)
        if group is None:
            raise ValueError(f"operator group {name} is missing")
        if not isinstance(group, (list, tuple)):
            group = (group,)
        for index, value in enumerate(group):
            _hash_operator_value(
                digest, f"{name}.{index}", value, backend
            )
    for name in (
        "I",
        "H",
        "S",
        "scalar_I",
        "QB",
        "T",
        "T_full",
        "M0",
        "M1",
        "MB",
        "topology_flux_row",
    ):
        if not hasattr(operators, name):
            raise ValueError(f"operator payload {name} is missing")
        _hash_operator_value(
            digest, name, getattr(operators, name), backend
        )
    selected = getattr(operators, "selected_wall_rows", None)
    if selected is None:
        raise ValueError("operator selected_wall_rows is missing")
    _hash_numeric(
        digest,
        "selected_wall_rows",
        np.asarray(selected, dtype=np.int64),
    )
    return digest.hexdigest()


def checkpoint_layout_metadata(geometry):
    """Return the scalar metadata required by a native accepted checkpoint."""
    return SimpleNamespace(
        layout_schema=LAYOUT_SCHEMA,
        geometry_fingerprint=geometry_fingerprint(geometry),
        native_entity_metadata=native_entity_metadata(geometry),
    )
