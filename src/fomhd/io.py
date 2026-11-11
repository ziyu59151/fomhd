"""Strict A0 Python configuration, state I/O, run provenance and restart."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import sys
import time as time_module
import tomllib
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .types import Failure


SCHEMA_VERSION = 1
LAYOUT_VERSION = 1
MODEL_ID = "A0"
FINITE_MODEL = "finite_mass_cgl_b"
MASSLESS_MODEL = "massless_quasineutral_cgl_b"
RULE_HASH_TEXT = "A0-compatible-operators-v2"
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
SAFE_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")


def _model_applicability(parameters):
    """Return JSON-safe model applicability shared by restart and provenance."""
    if int(getattr(parameters, "electron_inertia", 0)) == 0:
        return {
            "model": MASSLESS_MODEL,
            "electron_inertia": "not_applicable",
            "electron_kinetic_energy": "not_applicable",
            "electron_kinetic_transport": "not_applicable",
            "electron_mechanical_stress": "not_applicable",
            "finite_charge_electric_stress": "not_applicable",
            "longitudinal_electric_energy": "not_applicable",
            "finite_charge_gauss": "not_applicable",
            "spectral_kinetic_transfer": "not_applicable",
        }
    return {
        "model": FINITE_MODEL,
        "electron_inertia": 1,
        "electron_kinetic_energy": "applicable",
        "longitudinal_electric_energy": "applicable",
        "finite_charge_gauss": "applicable",
    }


@dataclass
class Limits:
    max_input_bytes: int = 256 << 20
    max_expanded_bytes: int = 512 << 20
    max_entries: int = 4096
    max_header_bytes: int = 1 << 20


def _norm_text(norm):
    if hasattr(norm, "value"):
        norm = norm.value
    value = str(norm).lower()
    if value.endswith("weighted_l2"):
        return "weighted_L2"
    if value.endswith("linf"):
        return "linf"
    if value.endswith("value"):
        return "value"
    return str(norm)


class PublicationError(OSError):
    def __init__(self, phase, temporary, published, message):
        super().__init__(f"{phase}: {message}; temporary={temporary}")
        self.phase = phase
        self.temporary = Path(temporary)
        self.published = bool(published)


@dataclass
class Restart:
    state: object
    cumulative: object
    parent: dict
    trial: int
    source: int


def _checked_add(left, right, name):
    out = left + right
    if out < 0 or out > (1 << 63):
        raise ValueError(f"{name} exceeds the configured integer range")
    return out


def _require_finite_json(value, path="$"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite JSON number at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_json(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_json(child, f"{path}[{index}]")
    return value


def typed_value(value):
    value = float(value)
    if math.isfinite(value):
        return value
    text = "NaN" if math.isnan(value) else ("+Inf" if value > 0 else "-Inf")
    return {"value": text, "type": "float64"}


def _json_value(value):
    if isinstance(value, complex):
        if abs(value.imag) > 1e-12 * max(1.0, abs(value.real)):
            raise ValueError("complex value in JSON metadata")
        return typed_value(value.real)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        return typed_value(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_json_value(child) for child in value]
    if isinstance(value, dict):
        return {str(key): _json_value(child) for key, child in value.items()}
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_value(child)
            for key, child in vars(value).items()
            if key != "traces"
        }
    return value


def _canonical(value):
    value = _json_value(value)
    _require_finite_json(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"input is not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path, cap=4 << 20):
    path = Path(path)
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode) or status.st_size > cap:
        raise ValueError(f"input is not a bounded regular file: {path}")
    data = path.read_bytes()
    if len(data) > cap:
        raise ValueError(f"cannot read bounded input: {path}")
    return data.decode("utf-8")


def write_text(path, text):
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _write_atomic_text(path, text):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size:
        raise ValueError("atomic JSON writer requires a caller-owned empty file")
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def parse_json(text):
    if len(text.encode("utf-8")) > 4 << 20:
        raise ValueError("JSON input exceeds the configuration byte cap")
    value = json.loads(
        text,
        object_pairs_hook=_pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON token: {token}")
        ),
    )
    _require_finite_json(value)
    return value


def read_json(path):
    return parse_json(read_text(path))


def _keys(table, expected, name):
    if not isinstance(table, dict):
        raise ValueError(f"{name} must be a table")
    wanted = set(expected)
    actual = set(table)
    unknown = sorted(actual - wanted)
    missing = sorted(wanted - actual)
    if unknown:
        raise ValueError(f"unknown key {name}.{unknown[0]}")
    if missing:
        raise ValueError(f"missing key {name}.{missing[0]}")


def _real(table, key, positive=False):
    value = table[key]
    if type(value) is not float:
        raise ValueError(f"{key} must be a TOML float")
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{key} is outside its finite domain")
    return value


def _integer(table, key, positive=False):
    value = table[key]
    if type(value) is not int or value < 0:
        raise ValueError(f"{key} must be a nonnegative integer")
    if positive and value == 0:
        raise ValueError(f"{key} must be positive")
    return value


def _string(table, key):
    value = table[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _safe_name(value):
    return bool(SAFE_NAME.fullmatch(value))


def _finite_budget_registry(parameters):
    specs = {}

    def array(key, unit, hard=True):
        specs[f"{key}.linf"] = (unit, "linf", hard)
        specs[f"{key}.weighted_L2"] = (unit, "weighted_L2", hard)

    def value(key, unit, hard=True):
        specs[f"{key}.value"] = (unit, "value", hard)

    for key in ("poisson_linear", "gauss", "transverse_divergence"):
        array(key, "V/m^2")
    array("darwin_dynamic", "A/(m^2*s)")
    array("longitudinal_projection", "A/m^3")
    for key in (
        "initial_divB",
        "initial_ampere",
        "divB_update",
        "long_term_divB_change",
    ):
        array(key, "T/m")
    for key in ("current_divB", "current_ampere"):
        array(key, "T/m", False)
    array("CT_increment", "T/(m*s)")
    array("current_response", "A/(m^2*s)")
    for key in ("CT_update", "equilibrium_B", "alias_B"):
        array(key, "T/s")
    for key in ("RC_flux", "RC_mom", "deltaC"):
        array(key, "C/(m^3*s)")
    array("RL", "A/m^2")
    value("topology_increment", "Wb")
    value("total_charge_balance", "C")
    for key in (
        "ion_internal",
        "electron_perp",
        "electron_parallel",
        "field_energy_B",
        "field_energy_L",
        "total_energy",
    ):
        value(key, "J", False)
    value("field_exchange", "W", False)
    for key in ("momentum_x", "momentum_y", "momentum_z"):
        value(key, "kg*m/s", False)
    value("angular_momentum", "kg*m^2/s", False)
    for a in range(2):
        suffix = f".{parameters.species[a].id}"
        for key in (
            "stage_continuity",
            "species_continuity",
            "equilibrium_n",
            "alias_n",
        ):
            array(key + suffix, "1/(m^3*s)")
        for key in (
            "wall_rhs",
            "wall_multiplier",
            "equilibrium_mom",
            "alias_mom",
        ):
            array(key + suffix, "kg/(m^2*s^2)")
        array("wall_state" + suffix, "kg/(m^2*s)")
        array("theta_identity" + suffix, "1/s")
        array("wall_number_flux" + suffix, "1/(m^2*s)")
        array("wall_perp_flux" + suffix, "Pa*m/s")
        if a:
            array("wall_parallel_flux" + suffix, "Pa*m/s")
        for key in ("equilibrium_perp", "alias_perp"):
            array(key + suffix, "Pa/s")
        if a:
            for key in (
                "cgl_source",
                "equilibrium_parallel",
                "alias_parallel",
            ):
                array(key + suffix, "Pa/s")
        for key in (
            "electric_pairing",
            "pressure_pairing",
            "magnetic_work",
            "wall_work",
            "convection_pairing",
        ):
            value(key + suffix, "W")
        value("spectral_transfer_defect" + suffix, "W", False)
        value("total_number_balance" + suffix, "1")
        value("kinetic_energy" + suffix, "J", False)
    return specs


def _massless_budget_registry(parameters):
    """Return the exact observe/hard registry emitted by the massless branch."""
    ion_id = parameters.species[0].id
    electron_id = parameters.species[1].id
    specs = {}

    def array(key, unit, hard=True):
        specs[f"{key}.linf"] = (unit, "linf", hard)
        specs[f"{key}.weighted_L2"] = (unit, "weighted_L2", hard)

    def value(key, unit, hard=True):
        specs[f"{key}.value"] = (unit, "value", hard)

    for key, unit, hard in (
        ("quasineutrality", "C/m^3", True),
        ("electron_density_compatibility", "1/(m^3*s)", True),
        ("electron_momentum_compatibility", "kg/(m^2*s^2)", True),
        ("current_definition", "A/m^2", True),
        ("electron_flux_definition", "1/(m^2*s)", True),
        ("electron_flux_continuity", "1/(m^3*s)", True),
        ("djj", "A/m^3", True),
        ("djj_interval", "A/m^3", True),
        ("ct_divergence", "T/(m*s)", True),
        ("ct_update", "T/s", True),
        ("divB_update", "T/m", True),
        ("wall_current", "A/m^2", True),
        ("wall_current_interval", "A/m^2", True),
        ("wall_ion_flux", "1/(m^2*s)", True),
        ("wall_electron_flux", "1/(m^2*s)", True),
        (f"wall_ion_flux_interval.{ion_id}", "1/(m^2*s)", True),
        (
            f"wall_electron_flux_interval.{electron_id}",
            "1/(m^2*s)",
            True,
        ),
        ("wall_field_original_rows", "V/m^3", True),
        ("drift_ohm_residual", "V/m", True),
        ("electron_cgl_perp", "Pa/s", True),
        ("electron_cgl_parallel", "Pa/s", True),
        ("theta_consistency", "1/s", False),
        ("electron_pressure_adjoint", "W", True),
        ("electron_electric_pairing", "W", True),
        ("initial_divB", "T/m", True),
        ("wall_initial_current", "A/m^2", True),
        ("species_continuity", "1/(m^3*s)", True),
        (f"species_continuity.{ion_id}", "1/(m^3*s)", True),
        (f"species_continuity.{electron_id}", "1/(m^3*s)", True),
        ("quasineutrality_interval", "C/(m^3*s)", True),
        (f"alias_n.{ion_id}", "1/(m^3*s)", True),
        (f"alias_n.{electron_id}", "1/(m^3*s)", True),
        (f"alias_mom.{ion_id}", "kg/(m^2*s^2)", True),
        (f"alias_mom.{electron_id}", "kg/(m^2*s^2)", True),
        (f"alias_perp.{ion_id}", "Pa/s", True),
        (f"alias_perp.{electron_id}", "Pa/s", True),
        (f"alias_parallel.{electron_id}", "Pa/s", True),
        ("alias_B", "T/s", True),
        (f"equilibrium_n.{ion_id}", "1/(m^3*s)", True),
        (f"equilibrium_n.{electron_id}", "1/(m^3*s)", True),
        (f"equilibrium_mom.{ion_id}", "kg/(m^2*s^2)", True),
        (
            f"equilibrium_mom.{electron_id}",
            "kg/(m^2*s^2)",
            True,
        ),
        (f"equilibrium_perp.{ion_id}", "Pa/s", True),
        (f"equilibrium_perp.{electron_id}", "Pa/s", True),
        (f"equilibrium_parallel.{electron_id}", "Pa/s", True),
        ("equilibrium_B", "T/s", True),
    ):
        array(key, unit, hard)
    for key, unit in (
        (f"electron_pressure_adjoint.{electron_id}", "W"),
        (f"electron_electric_pairing.{electron_id}", "W"),
        (f"electron_magnetic_work.{electron_id}", "W"),
        (f"pressure_pairing.{electron_id}", "W"),
        (f"electric_pairing.{electron_id}", "W"),
        (f"electron_pressure_only_pairing.{electron_id}", "W"),
        ("wall_field_reaction_work", "W"),
        ("wall_field_reaction_work_ion", "W"),
        ("wall_field_reaction_work_electron", "W"),
        ("wall_field_reaction_work_current", "W"),
        (f"kinetic_energy.{ion_id}", "J"),
        ("ion_internal", "J"),
        ("electron_perp", "J"),
        ("electron_parallel", "J"),
        ("field_energy_B_interval", "J"),
        ("field_exchange_interval", "W"),
        ("total_energy_interval", "J"),
        ("total_charge_balance_interval", "C"),
        (f"total_number_balance.{ion_id}", "1"),
        (f"total_number_balance.{electron_id}", "1"),
        ("momentum_balance_interval_x", "kg*m/s"),
        ("momentum_balance_interval_y", "kg*m/s"),
        ("momentum_balance_interval_z", "kg*m/s"),
        ("angular_momentum_interval", "kg*m^2/s"),
    ):
        value(key, unit)
    return specs


def _budget_registry(parameters):
    if int(getattr(parameters, "electron_inertia", 0)) == 0:
        return _massless_budget_registry(parameters)
    return _finite_budget_registry(parameters)


def _operator_fingerprint(operators, backend):
    existing = getattr(operators, "matrix_fingerprint", None)
    if existing:
        return str(existing)
    digest = hashlib.sha256()
    digest.update(RULE_HASH_TEXT.encode("ascii"))
    groups = (
        "G",
        "C",
        "DB",
        "DJ",
        "Cdag",
        "Lphi",
    )
    for name in groups:
        matrix_group = getattr(operators, name)
        for matrix in matrix_group:
            array = np.asarray(backend.to_host(matrix))
            digest.update(f"{name}:{array.shape}:".encode("ascii"))
            digest.update(np.ascontiguousarray(array).tobytes())
    for name in (
        "I",
        "H",
        "QB",
        "T_full",
        "T",
        "Bc_full",
        "Bc",
        "DR",
        "DZ",
        "topology_flux_row",
        "PR_r",
        "PZ_r",
        "PR_z",
        "PZ_z",
        "PR_v",
        "PZ_v",
    ):
        value = getattr(operators, name)
        array = np.asarray(backend.to_host(value))
        digest.update(f"{name}:{array.shape}:".encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    for name in ("M0", "M1", "MB"):
        array = np.asarray(backend.to_host(getattr(operators, name)))
        digest.update(f"{name}:{array.shape}:".encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    for name in ("selected_wall_rows", "selected_constraint_rows"):
        digest.update(
            (name + ":" + ",".join(map(str, getattr(operators, name)))).encode(
                "ascii"
            )
        )
    return digest.hexdigest()


def _zip_header_length(handle, version):
    if version[0] == 1:
        raw = handle.read(2)
        return struct.unpack("<H", raw)[0] if len(raw) == 2 else 0
    raw = handle.read(4)
    return struct.unpack("<I", raw)[0] if len(raw) == 4 else 0


def _inspect_npz(path, limits):
    path = Path(path)
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode) or status.st_size > limits.max_input_bytes:
        raise ValueError(f"NPZ input is not a bounded regular file: {path}")
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        if len(infos) > limits.max_entries:
            raise ValueError("NPZ entry count exceeds the configured limit")
        seen = set()
        expanded = 0
        names = []
        for info in infos:
            if info.is_dir():
                raise ValueError("directory entries are not allowed in NPZ")
            if info.flag_bits & 1:
                raise ValueError("encrypted NPZ entries are not allowed")
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError("unsupported ZIP compression method")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ValueError("symbolic-link NPZ entries are not allowed")
            name = info.filename
            if name in seen:
                raise ValueError("duplicate NPZ entry name")
            seen.add(name)
            if (
                not name.endswith(".npy")
                or name == ".npy"
                or "/" in name
                or "\\" in name
            ):
                raise ValueError(f"unsafe NPZ entry: {name}")
            key = name[:-4]
            if not SAFE_KEY.fullmatch(key):
                raise ValueError(f"unsafe NPZ key: {key}")
            expanded = _checked_add(expanded, info.file_size, "expanded NPZ bytes")
            if expanded > limits.max_expanded_bytes:
                raise ValueError(
                    "NPZ expanded data exceeds the configured byte limit"
                )
            with archive.open(info) as handle:
                magic = handle.read(8)
                if len(magic) != 8 or magic[:6] != b"\x93NUMPY":
                    raise ValueError(f"invalid NPY magic in {name}")
                header = _zip_header_length(handle, (magic[6], magic[7]))
                if (
                    header <= 0
                    or header > limits.max_header_bytes
                    or header > info.file_size
                ):
                    raise ValueError(f"invalid NPY header length in {name}")
            names.append(key)
        return names


def read_npz(path, limits=None, allowed_keys=None, *, allow_nonfinite=False):
    limits = limits or Limits()
    keys = _inspect_npz(path, limits)
    if allowed_keys is not None and set(keys) != set(allowed_keys):
        unknown = sorted(set(keys) - set(allowed_keys))
        missing = sorted(set(allowed_keys) - set(keys))
        if unknown:
            raise ValueError(f"NPZ entry is not present in the key whitelist: {unknown[0]}")
        raise ValueError(f"NPZ is missing required key: {missing[0]}")
    try:
        with np.load(Path(path), allow_pickle=False, mmap_mode=None) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    except (ValueError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError(f"unsafe or malformed NPZ: {error}") from error
    if set(arrays) != set(keys):
        raise ValueError("NPZ key set changed while reading")
    for key, array in arrays.items():
        if array.dtype.hasobject:
            raise ValueError(f"object dtype is not allowed: {key}")
        if array.dtype.kind not in "fcbiuU":
            raise ValueError(f"unsupported NPZ dtype: {key}:{array.dtype}")
        if (
            not allow_nonfinite
            and array.dtype.kind in "fc"
            and not np.isfinite(array).all()
        ):
            raise ValueError(f"NPZ array contains nonfinite values: {key}")
    return arrays


def _npz_write(path, arrays, *, allow_nonfinite=False):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size:
        raise ValueError("NPZ temp writer requires an empty regular file")
    with path.open("r+b") as handle:
        handle.seek(0)
        handle.truncate()
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    read_npz(
        path,
        Limits(
            max_input_bytes=max(1, path.stat().st_size),
            max_expanded_bytes=max(
                path.stat().st_size,
                sum(int(v.nbytes) for v in arrays.values())
                + 256 * len(arrays),
            ),
            max_entries=max(1, len(arrays)),
            max_header_bytes=1 << 20,
        ),
        set(arrays),
        allow_nonfinite=allow_nonfinite,
    )


def atomic_publish(target, writer, replace=False, fault=None):
    target = Path(target)
    parent = target.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("output parent must be a real directory")
    if not replace and os.path.lexists(target):
        raise ValueError("target already exists")
    temporary = parent / (
        f"{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    phase = "temp_created"
    published = False
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
        )
        os.close(descriptor)
        if fault:
            fault(phase)
        writer(temporary)
        phase = "written"
        if fault:
            fault(phase)
        descriptor = os.open(temporary, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        digest = sha256_file(temporary)
        phase = "validated"
        if fault:
            fault(phase)
        if replace:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
            published = True
            temporary.unlink()
        published = True
        phase = "renamed"
        if fault:
            fault(phase)
        descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        phase = "directory_synced"
        if fault:
            fault(phase)
        return digest
    except Exception as error:
        raise PublicationError(phase, temporary, published, str(error)) from error


def _publish_json(target, value, replace=False):
    value = _json_value(value)
    _require_finite_json(value)

    def writer(path):
        _write_atomic_text(
            path, json.dumps(value, indent=2, allow_nan=False) + "\n"
        )

    return atomic_publish(target, writer, replace)


def _entity_keys(table):
    keys = getattr(table, "keys", None)
    if keys is None:
        raise ValueError("entity table has no keys")
    return list(keys)


def _host_array(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    get = getattr(value, "get", None)
    if callable(get):
        return np.asarray(get())
    return np.asarray(value)


def _scatter(vector, table, modes, ny, nx, components=1):
    vector = _host_array(vector).reshape(-1)
    keys = _entity_keys(table)
    if vector.size != modes * components * len(keys):
        raise ValueError("compact state extent mismatch")
    if components == 1:
        dense = np.zeros((modes, ny, nx), dtype=np.complex128)
    else:
        dense = np.zeros((components, modes, ny, nx), dtype=np.complex128)
    for component in range(components):
        for mode in range(modes):
            for active, key in enumerate(keys):
                value = vector[(mode * components + component) * len(keys) + active]
                if value != 0 and not np.isfinite(value):
                    raise ValueError("state contains nonfinite values")
                i, j = key
                if components == 1:
                    dense[mode, j, i] = value
                else:
                    dense[component, mode, j, i] = value
    return dense


def _gather(array, table, modes, ny, nx, components=1):
    keys = _entity_keys(table)
    expected = (
        (modes, ny, nx)
        if components == 1
        else (components, modes, ny, nx)
    )
    if tuple(array.shape) != expected:
        raise ValueError(
            f"state array shape mismatch: got {array.shape}, expected {expected}"
        )
    if array.dtype != np.dtype(np.complex128):
        raise ValueError("state array must have complex128 dtype")
    if not np.isfinite(array).all():
        raise ValueError("state contains nonfinite values")
    active = {key for key in keys}
    if components == 1:
        for mode in range(modes):
            for j in range(ny):
                for i in range(nx):
                    if (i, j) not in active and array[mode, j, i] != 0:
                        raise ValueError("nonzero inactive slot in state")
    else:
        for component in range(components):
            for mode in range(modes):
                for j in range(ny):
                    for i in range(nx):
                        if (i, j) not in active and array[component, mode, j, i] != 0:
                            raise ValueError("nonzero inactive slot in state")
    out = np.empty(modes * components * len(keys), dtype=np.complex128)
    for component in range(components):
        for mode in range(modes):
            for active, key in enumerate(keys):
                i, j = key
                value = (
                    array[mode, j, i]
                    if components == 1
                    else array[component, mode, j, i]
                )
                out[(mode * components + component) * len(keys) + active] = value
    return out


def encode_state(state, geometry):
    nc = len(_entity_keys(geometry.cells))
    modes = int(state.fluid[0].n.size) // nc
    ny, nx = len(geometry.Z), len(geometry.R)
    arrays = {}
    for a, suffix in enumerate(("i", "e")):
        fluid = state.fluid[a]
        arrays[f"n_{suffix}"] = _scatter(fluid.n, geometry.cells, modes, ny, nx)
        arrays[f"mom_{suffix}"] = _scatter(
            fluid.mom, geometry.cells, modes, ny, nx, 3
        )
        if a == 0:
            arrays["p_i"] = _scatter(fluid.perp, geometry.cells, modes, ny, nx)
        else:
            arrays["p_perp_e"] = _scatter(
                fluid.perp, geometry.cells, modes, ny, nx
            )
            arrays["p_parallel_e"] = _scatter(
                fluid.parallel, geometry.cells, modes, ny, nx
            )
    nz = len(_entity_keys(geometry.z_faces))
    nv = len(_entity_keys(geometry.vertices))
    nr = len(_entity_keys(geometry.r_faces))
    nb = nz + nv + nr
    b = _host_array(state.B).reshape(-1)
    if b.size != modes * nb:
        raise ValueError("compact B state extent mismatch")
    radial = np.empty(modes * nz, dtype=np.complex128)
    poloidal = np.empty(modes * nv, dtype=np.complex128)
    vertical = np.empty(modes * nr, dtype=np.complex128)
    for mode in range(modes):
        radial[mode * nz : (mode + 1) * nz] = b[
            mode * nb : mode * nb + nz
        ]
        poloidal[mode * nv : (mode + 1) * nv] = b[
            mode * nb + nz : mode * nb + nz + nv
        ]
        vertical[mode * nr : (mode + 1) * nr] = b[
            mode * nb + nz + nv : (mode + 1) * nb
        ]
    arrays["B_R"] = _scatter(
        radial, geometry.z_faces, modes, ny + 1, nx
    )
    arrays["B_phi"] = _scatter(
        poloidal, geometry.vertices, modes, ny + 1, nx + 1
    )
    arrays["B_Z"] = _scatter(
        vertical, geometry.r_faces, modes, ny, nx + 1
    )
    return arrays


def decode_state(arrays, geometry, K):
    from .types import FluidState, State

    ny, nx = len(geometry.Z), len(geometry.R)
    modes = 2 * int(K) + 1
    fluid = []
    for a, suffix in enumerate(("i", "e")):
        fluid.append(
            FluidState(
                n=_gather(arrays[f"n_{suffix}"], geometry.cells, modes, ny, nx),
                mom=_gather(
                    arrays[f"mom_{suffix}"], geometry.cells, modes, ny, nx, 3
                ),
                perp=_gather(
                    arrays["p_i" if a == 0 else "p_perp_e"],
                    geometry.cells,
                    modes,
                    ny,
                    nx,
                ),
                parallel=(
                    np.zeros(0, dtype=np.complex128)
                    if a == 0
                    else _gather(
                        arrays["p_parallel_e"],
                        geometry.cells,
                        modes,
                        ny,
                        nx,
                    )
                ),
            )
        )
    radial = _gather(
        arrays["B_R"], geometry.z_faces, modes, ny + 1, nx
    )
    poloidal = _gather(
        arrays["B_phi"], geometry.vertices, modes, ny + 1, nx + 1
    )
    vertical = _gather(
        arrays["B_Z"], geometry.r_faces, modes, ny, nx + 1
    )
    nz = len(_entity_keys(geometry.z_faces))
    nv = len(_entity_keys(geometry.vertices))
    nr = len(_entity_keys(geometry.r_faces))
    nb = nz + nv + nr
    B = np.empty(modes * nb, dtype=np.complex128)
    for mode in range(modes):
        B[mode * nb : mode * nb + nz] = radial[mode * nz : (mode + 1) * nz]
        B[mode * nb + nz : mode * nb + nz + nv] = poloidal[
            mode * nv : (mode + 1) * nv
        ]
        B[mode * nb + nz + nv : (mode + 1) * nb] = vertical[
            mode * nr : (mode + 1) * nr
        ]
    return State(fluid=tuple(fluid), B=B, step=0, generation=1, time=0.0)


def encode_vertex(vector, geometry, K):
    modes = 2 * int(K) + 1
    return _scatter(
        vector,
        geometry.vertices,
        modes,
        len(geometry.Z) + 1,
        len(geometry.R) + 1,
    )


def decode_vertex(array, geometry, K):
    return _gather(
        array,
        geometry.vertices,
        2 * int(K) + 1,
        len(geometry.Z) + 1,
        len(geometry.R) + 1,
    )


def _upload_state(state, backend):
    """Move a host codec state to the selected solver backend exactly once."""
    if backend is None or not callable(getattr(backend, "array", None)):
        raise ValueError("selected backend must provide array()")
    for fluid in state.fluid:
        fluid.n = backend.array(_host_array(fluid.n), np.complex128)
        fluid.mom = backend.array(_host_array(fluid.mom), np.complex128)
        fluid.perp = backend.array(_host_array(fluid.perp), np.complex128)
        if fluid.parallel is None:
            fluid.parallel = backend.array(
                np.zeros(0, dtype=np.complex128), np.complex128
            )
        else:
            fluid.parallel = backend.array(
                _host_array(fluid.parallel), np.complex128
            )
    state.B = backend.array(_host_array(state.B), np.complex128)
    return state


def _build_parameters(species, solver, grid, time, budgets, diagnostics):
    from .types import Budget, Parameters, References, Species

    species_values = [
        Species(
            id=species["ion_id"],
            mass=species["mi"],
            charge=species["qi"],
            gamma=species["gamma_i"],
            n_min=budgets["domain"]["n_min_i"],
            face_min=budgets["domain"]["n_face_min_i"],
            p_min=budgets["domain"]["p_min_i"],
            parallel_min=0.0,
        ),
        Species(
            id=species["electron_id"],
            mass=species["me"],
            charge=species["qe"],
            gamma=0.0,
            n_min=budgets["domain"]["n_min_e"],
            face_min=budgets["domain"]["n_face_min_e"],
            p_min=budgets["domain"]["p_min_perp_e"],
            parallel_min=budgets["domain"]["p_min_parallel_e"],
        ),
    ]
    references = References(
        length=solver["references"]["L_m"],
        time=solver["references"]["t_s"],
        ni=solver["references"]["n_i_m3"],
        ne=solver["references"]["n_e_m3"],
        velocity=solver["references"]["u_mps"],
        electric=solver["references"]["E_Vpm"],
        magnetic=solver["references"]["B_T"],
        current=solver["references"]["J_Apm2"],
        charge=solver["references"]["rho_Cpm3"],
        pressure=solver["references"]["p_Pa"],
    )
    registry = {
        key: Budget(
            atol=value["atol"],
            rtol=value["rtol"],
            reference=value["reference"],
            unit=value["unit"],
            norm=value["norm"],
            hard=value["gate"] == "hard",
        )
        for key, value in budgets["checks"].items()
    }
    parameters_value = Parameters(
        species=tuple(species_values),
        epsilon0=solver["constants"]["epsilon0_Fpm"],
        mu0=solver["constants"]["mu0_Hpm"],
        B_min=budgets["domain"]["B_min"],
        dt=time["dt"],
        start_time=time["start_time"],
        rank_rtol=solver["rank_rtol"],
        nsteps=time["nsteps"],
        K=grid["K"],
        Nphi=grid["Nphi"],
        memory_limit=solver["memory_limit_bytes"],
        device_memory_limit=solver["device_memory_limit_bytes"],
        inner_distance=diagnostics["inner_distance_m"],
        ref=references,
        budgets=registry,
        electron_inertia=species.get("electron_inertia", 0),
        integrator=time.get("integrator", "rk4"),
        dt_guard=time.get("dt_guard", "error"),
        minimum_samples_per_cycle=time.get(
            "minimum_samples_per_cycle", 32.0
        ),
        max_amplitude_error=time.get("max_amplitude_error", 0.01),
        max_phase_error_rad=time.get("max_phase_error_rad", 0.1),
    )
    return parameters_value


def load_case(path, overrides=None, backend=None):
    """Load the exact A0 v1 schema and construct geometry/operators."""
    overrides = overrides or {}
    path = Path(path).absolute()
    raw = read_text(path)
    case_path = path
    try:
        document = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"TOML input: {error}") from error
    top = {
        "schema_version",
        "model",
        "units",
        "layout_version",
        "case_name",
        "species",
        "grid",
        "time",
        "boundary",
        "solver",
        "budgets",
        "io",
        "diagnostics",
    }
    _keys(document, top, "case")
    if (
        _integer(document, "schema_version") != SCHEMA_VERSION
        or _integer(document, "layout_version") != LAYOUT_VERSION
        or _string(document, "model") != MODEL_ID
        or _string(document, "units") != "SI"
    ):
        raise ValueError("expected A0 schema/layout 1; units must be SI")
    case_name = _string(document, "case_name")
    if not _safe_name(case_name):
        raise ValueError("invalid case_name")

    species = document["species"]
    species_keys = ["ion_id", "electron_id", "qi", "mi", "qe", "me", "gamma_i"]
    if "electron_inertia" in species:
        species_keys.append("electron_inertia")
    _keys(species, species_keys, "species")
    for name in ("ion_id", "electron_id"):
        if not _safe_name(_string(species, name)):
            raise ValueError("invalid species identifier")
    if species["ion_id"] == species["electron_id"]:
        raise ValueError("species identifiers must differ")
    for name in ("mi", "me", "qi", "gamma_i"):
        _real(species, name, True)
    _real(species, "qe")
    if species["qe"] >= 0 or species["gamma_i"] <= 1:
        raise ValueError("invalid species constants")
    inertia = species.get("electron_inertia", 0)
    if type(inertia) is not int or inertia not in (0, 1):
        raise ValueError("electron_inertia must be 0 or 1")

    solver = document["solver"]
    _keys(
        solver,
        {
            "backend",
            "precision",
            "complex_precision",
            "memory_limit_bytes",
            "device_memory_limit_bytes",
            "rank_rtol",
            "device_ordinal",
            "cpu_threads",
            "constants",
            "references",
        },
        "solver",
    )
    backend_name = _string(solver, "backend")
    if backend_name not in ("cpu", "cuda"):
        raise ValueError("unsupported backend")
    if (
        _string(solver, "precision") != "fp64"
        or _string(solver, "complex_precision") != "complex128"
        or _integer(solver, "cpu_threads", True) != 1
    ):
        raise ValueError("unsupported precision/thread policy")
    device = _integer(solver, "device_ordinal")
    if backend_name == "cpu" and device != 0:
        raise ValueError("CPU device ordinal must be zero")
    memory = _integer(solver, "memory_limit_bytes", True)
    device_memory = _integer(solver, "device_memory_limit_bytes", True)
    rank_rtol = _real(solver, "rank_rtol", True)
    if rank_rtol >= 1:
        raise ValueError("rank_rtol must be below one")
    constants = solver["constants"]
    _keys(constants, {"epsilon0_Fpm", "mu0_Hpm", "version"}, "constants")
    _real(constants, "epsilon0_Fpm", True)
    _real(constants, "mu0_Hpm", True)
    if not _string(constants, "version"):
        raise ValueError("constants version is required")
    references = solver["references"]
    _keys(
        references,
        {
            "L_m",
            "t_s",
            "n_i_m3",
            "n_e_m3",
            "u_mps",
            "E_Vpm",
            "B_T",
            "J_Apm2",
            "rho_Cpm3",
            "p_Pa",
        },
        "references",
    )
    for name in references:
        _real(references, name, True)

    time = document["time"]
    allowed_time = {"start_time", "dt", "nsteps", "initial_kind"}
    for optional in (
        "integrator",
        "dt_guard",
        "minimum_samples_per_cycle",
        "max_amplitude_error",
        "max_phase_error_rad",
    ):
        if optional in time:
            allowed_time.add(optional)
    _keys(time, allowed_time, "time")
    if "nsteps" in overrides:
        time["nsteps"] = overrides["nsteps"]
    _real(time, "dt", True)
    _real(time, "start_time")
    _integer(time, "nsteps")
    if time["nsteps"] > 1 << 52:
        raise ValueError("too many time steps")
    if _string(time, "initial_kind") not in (
        "equilibrium",
        "non_equilibrium",
    ):
        raise ValueError("unknown initial kind")
    if time.get("integrator", "rk4") not in ("midpoint", "rk4"):
        raise ValueError("unknown time integrator")
    if time.get("dt_guard", "error") not in ("error", "warn"):
        raise ValueError("unknown dt_guard policy")
    for name, positive in (
        ("minimum_samples_per_cycle", True),
        ("max_amplitude_error", True),
        ("max_phase_error_rad", True),
    ):
        if name in time:
            _real(time, name, positive)

    boundary = document["boundary"]
    _keys(boundary, {"kind", "potential", "version"}, "boundary")
    boundary_kind = _string(boundary, "kind")
    boundary_version = _string(boundary, "version")
    if _real(boundary, "potential") != 0 or (
        boundary_kind,
        boundary_version,
    ) not in {
        ("fixed_lcfs_static", "staircase_relative_v1"),
        ("fixed_outer_wall_static", "rectangle_relative_v1"),
    }:
        raise ValueError("unsupported boundary")

    if boundary_version == "rectangle_relative_v1" and inertia != 0:
        raise ValueError("rectangle_relative_v1 requires massless electrons")

    grid = document["grid"]
    _keys(grid, {"R0", "Z0", "hR", "hZ", "NR", "NZ", "K", "Nphi"}, "grid")
    nr = _integer(grid, "NR", True)
    nz = _integer(grid, "NZ", True)
    K = _integer(grid, "K")
    nphi = _integer(grid, "Nphi", True)
    if (
        nr < 2
        or nz < 2
        or K > 1000
        or nphi > 100000
        or nphi % 2 == 0
        or nphi < 4 * K + 1
    ):
        raise ValueError("invalid grid or Fourier resolution")
    for name in ("R0", "Z0"):
        _real(grid, name)
    for name in ("hR", "hZ"):
        _real(grid, name, True)

    budgets = document["budgets"]
    _keys(budgets, {"domain", "checks"}, "budgets")
    domain = budgets["domain"]
    _keys(
        domain,
        {
            "n_min_i",
            "n_min_e",
            "n_face_min_i",
            "n_face_min_e",
            "p_min_i",
            "p_min_perp_e",
            "p_min_parallel_e",
            "B_min",
        },
        "domain",
    )
    for name in domain:
        if _real(domain, name) < 0:
            raise ValueError("negative domain floor")
    checks = budgets["checks"]
    if not isinstance(checks, dict):
        raise ValueError("checks must be a table")
    # Build parameter shell first, then enforce the full fixed registry.
    parameters = _build_parameters(
        species, solver, grid, time, budgets, document["diagnostics"]
    )
    specs = _budget_registry(parameters)
    unknown = sorted(set(checks) - set(specs))
    if unknown:
        raise ValueError(f"unknown budget {unknown[0]}")
    missing = sorted(set(specs) - set(checks))
    if missing:
        raise ValueError(f"missing budget {missing[0]}")
    for name, spec in specs.items():
        value = checks[name]
        _keys(value, {"atol", "rtol", "reference", "unit", "norm", "gate"}, name)
        atol = _real(value, "atol")
        rtol = _real(value, "rtol")
        reference = _real(value, "reference", True)
        unit, norm, required_hard = spec
        gate = _string(value, "gate")
        if (
            atol < 0
            or rtol < 0
            or _string(value, "unit") != unit
            or _string(value, "norm") != norm
            or not name.endswith(f".{norm}")
            or gate not in ("hard", "observe")
            or (required_hard and gate != "hard")
        ):
            raise ValueError(f"invalid budget unit/norm/gate: {name}")

    diagnostics = document["diagnostics"]
    _keys(
        diagnostics,
        {
            "inner_distance_m",
            "alias_policy",
            "timing_repetitions",
            "timing_warmup",
            "mode_observations",
        },
        "diagnostics",
    )
    _real(diagnostics, "inner_distance_m", True)
    if _string(diagnostics, "alias_policy") != "initial_midpoint_restart":
        raise ValueError("unsupported alias policy")
    _integer(diagnostics, "timing_repetitions", True)
    _integer(diagnostics, "timing_warmup")
    if not isinstance(diagnostics["mode_observations"], list):
        raise ValueError("mode_observations must be an array")
    for observation in diagnostics["mode_observations"]:
        _keys(
            observation,
            {"field", "component", "mode", "entity_key"},
            "observation",
        )
        if (
            type(observation["mode"]) is not int
            or abs(observation["mode"]) > K
            or not isinstance(observation["entity_key"], list)
            or len(observation["entity_key"]) != 2
        ):
            raise ValueError("invalid mode observation")

    paths = document["io"]
    _keys(
        paths,
        {
            "geometry_file",
            "initial_file",
            "output_dir",
            "snapshot_stride",
            "checkpoint_stride",
            "max_input_bytes",
            "max_uncompressed_bytes",
            "max_entries",
            "max_header_bytes",
            "source_manifest_file",
            "write_residual_arrays",
        },
        "io",
    )
    if overrides.get("output") is not None:
        paths["output_dir"] = str(Path(overrides["output"]).absolute())

    def resolve(name):
        value = Path(_string(paths, name))
        return (case_path.parent / value).resolve()

    output_value = Path(paths["output_dir"])
    output = (
        output_value.resolve()
        if output_value.is_absolute()
        else (case_path.parent / output_value).resolve()
    )
    geometry_path = resolve("geometry_file")
    initial_path = resolve("initial_file")
    source_path = resolve("source_manifest_file")
    snapshot_stride = _integer(paths, "snapshot_stride", True)
    checkpoint_stride = _integer(paths, "checkpoint_stride", True)
    if type(paths["write_residual_arrays"]) is not bool:
        raise ValueError("write_residual_arrays must be boolean")
    limits = Limits(
        _integer(paths, "max_input_bytes", True),
        _integer(paths, "max_uncompressed_bytes", True),
        _integer(paths, "max_entries", True),
        _integer(paths, "max_header_bytes", True),
    )
    if limits.max_expanded_bytes > memory // 4:
        raise ValueError(
            "input expanded cap exceeds one quarter of the host memory budget"
        )
    source = read_json(source_path)
    for key in (
        "provider_name",
        "provider_version",
        "units",
        "conversion_method",
        "input_hashes",
        "continuous_residual_status",
    ):
        if key not in source:
            raise ValueError(f"source manifest missing {key}")
    if source["units"] != "SI":
        raise ValueError("source manifest units must be SI")
    input_hashes = {
        "case": _sha256_bytes(raw.encode("utf-8")),
        "geometry": sha256_file(geometry_path),
        "initial": sha256_file(initial_path),
        "source": sha256_file(source_path),
    }
    recorded = source["input_hashes"]
    if not isinstance(recorded, dict):
        raise ValueError("source input_hashes must be a table")
    for required in ("case", "geometry", "initial"):
        if required not in recorded:
            raise ValueError(f"source input_hashes missing {required}")
    for key, digest in recorded.items():
        if key not in input_hashes or digest != input_hashes[key]:
            raise ValueError(f"source input hash mismatch: {key}")

    geometry_arrays = read_npz(
        geometry_path, limits, {"R_nodes", "Z_nodes", "lcfs_R", "lcfs_Z"}
    )
    if any(
        geometry_arrays[name].dtype != np.dtype(np.float64)
        for name in ("R_nodes", "Z_nodes", "lcfs_R", "lcfs_Z")
    ):
        raise ValueError("geometry arrays must use float64 dtype")
    R = np.asarray(geometry_arrays["R_nodes"], dtype=np.float64)
    Z = np.asarray(geometry_arrays["Z_nodes"], dtype=np.float64)
    lcfs_R = np.asarray(geometry_arrays["lcfs_R"], dtype=np.float64)
    lcfs_Z = np.asarray(geometry_arrays["lcfs_Z"], dtype=np.float64)
    if R.ndim != 1 or Z.ndim != 1 or lcfs_R.ndim != 1 or lcfs_Z.ndim != 1:
        raise ValueError("geometry arrays must be one dimensional")
    if R.size != nr or Z.size != nz or lcfs_R.size != lcfs_Z.size:
        raise ValueError("geometry dimensions disagree with configuration")
    for index, value in enumerate(R):
        expected = float(index) * grid["hR"] + grid["R0"]
        if abs(value - expected) > 64 * sys.float_info.epsilon * max(
            abs(value), abs(grid["hR"])
        ):
            raise ValueError("R grid configuration mismatch")
    for index, value in enumerate(Z):
        expected = float(index) * grid["hZ"] + grid["Z0"]
        if abs(value - expected) > 64 * sys.float_info.epsilon * max(
            abs(value), abs(grid["Z0"]), abs(grid["hZ"])
        ):
            raise ValueError("Z grid configuration mismatch")
    polygon = np.column_stack((lcfs_R, lcfs_Z))
    from . import geometry as geometry_module
    from . import operators as operators_module

    geometry_options = {"memory_cap_bytes": memory, "grid_rtol": 1e-12}
    if boundary_version == "rectangle_relative_v1":
        geometry = geometry_module.build_aligned_box(
            R, Z, polygon, geometry_options
        )
    else:
        geometry = geometry_module.build_geometry(
            R, Z, polygon, geometry_options
        )
    if backend is None:
        from .backend import Backend

        backend = Backend(
            "cpu" if backend_name == "cpu" else f"cuda:{device}"
        )
    operators = operators_module.build_operators(
        geometry,
        K,
        {"rank_rtol": rank_rtol, "memory_cap_bytes": memory},
        backend,
    )
    geometry_identity = (
        boundary_version
        if boundary_version == "rectangle_relative_v1"
        else "half_open_ray_certified_v2"
    )
    geometry_hash = _sha256_bytes(
        (input_hashes["geometry"] + geometry_identity).encode()
    )
    if boundary_version == "rectangle_relative_v1":
        from .box_boundary import install_normal_flux

        initial_arrays = _initial_arrays(initial_path, limits, rectangle=True)
        flux = initial_arrays.get("wall_magnetic_flux")
        if flux is None:
            flux = np.zeros((2 * K + 1, len(geometry.walls), 2), dtype=np.complex128)
        install_normal_flux(operators, geometry, flux)
        del initial_arrays
    fingerprint = _operator_fingerprint(operators, backend)
    resolved = {
        **document,
        "resolved_paths": {
            "geometry": str(geometry_path),
            "initial": str(initial_path),
            "source": str(source_path),
            "output": str(output),
        },
        "input_hashes": input_hashes,
    }
    config_hash = _sha256_bytes(_canonical(resolved).encode())
    semantic = json.loads(json.dumps(document))
    # Effective defaults are part of restart identity; an omitted historical
    # midpoint choice must not silently become RK4 after a default change.
    semantic["species"]["electron_inertia"] = parameters.electron_inertia
    semantic["time"]["integrator"] = parameters.integrator
    semantic["time"].pop("nsteps", None)
    for name in (
        "output_dir",
        "snapshot_stride",
        "checkpoint_stride",
        "geometry_file",
        "initial_file",
        "source_manifest_file",
    ):
        semantic["io"].pop(name, None)
    for name in ("timing_repetitions", "timing_warmup"):
        semantic["diagnostics"].pop(name, None)
    # The source manifest contains the case-file hash. Excluding only that
    # circular file hash while hashing the manifest content keeps stop-step
    # and path-only restart changes semantic-neutral without hiding source
    # provenance changes.
    semantic["input_hashes"] = {
        "geometry": input_hashes["geometry"],
        "initial": input_hashes["initial"],
    }
    semantic_source = json.loads(json.dumps(source))
    semantic_source["input_hashes"].pop("case", None)
    semantic["source"] = semantic_source
    semantic["operator_hash"] = str(fingerprint)
    semantic["geometry_hash"] = geometry_hash
    semantic_hash = _sha256_bytes(_canonical(semantic).encode())
    return SimpleNamespace(
        parameters=parameters,
        geometry=geometry,
        operators=operators,
        limits=limits,
        case_path=case_path,
        geometry_path=geometry_path,
        initial_path=initial_path,
        source_path=source_path,
        output=output,
        name=case_name,
        backend=backend_name,
        initial_kind=time["initial_kind"],
        raw=raw,
        config_hash=config_hash,
        semantic_hash=semantic_hash,
        geometry_hash=geometry_hash,
        operator_hash=str(fingerprint),
        device=device,
        snapshot_stride=snapshot_stride,
        checkpoint_stride=checkpoint_stride,
        residual_arrays=paths["write_residual_arrays"],
        resolved=resolved,
        source=source,
        input_hashes=input_hashes,
        compute=backend,
    )


INITIAL_STATE_KEYS = frozenset(
    {
        "n_i",
        "mom_i",
        "p_i",
        "n_e",
        "mom_e",
        "p_perp_e",
        "p_parallel_e",
        "B_R",
        "B_phi",
        "B_Z",
    }
)


def _initial_arrays(path, limits, *, rectangle):
    allowed = INITIAL_STATE_KEYS
    if rectangle and "wall_magnetic_flux" in _inspect_npz(path, limits):
        allowed = allowed | {"wall_magnetic_flux"}
    return read_npz(path, limits, allowed)


def load_initial(case):
    arrays = _initial_arrays(
        case.initial_path, case.limits,
        rectangle=case.geometry.lcfs_polygon is not None,
    )
    state = decode_state(arrays, case.geometry, case.parameters.K)
    _upload_state(state, case.compute)
    state.time = case.parameters.start_time
    return state


def _identity(arrays, committed, case, run_id, k, compute):
    arrays["step"] = np.asarray(committed.state.step, dtype=np.uint64)
    arrays["generation"] = np.asarray(committed.state.generation, dtype=np.uint64)
    arrays["time"] = np.asarray(committed.state.time, dtype=np.float64)
    arrays["dt"] = np.asarray(case.parameters.dt, dtype=np.float64)
    arrays["start_time"] = np.asarray(case.parameters.start_time, dtype=np.float64)
    arrays["accepted"] = np.asarray(True, dtype=np.bool_)
    arrays["schema_version"] = np.asarray(SCHEMA_VERSION, dtype=np.uint64)
    arrays["layout_version"] = np.asarray(LAYOUT_VERSION, dtype=np.uint64)
    arrays["model_id"] = np.asarray(MODEL_ID)
    applicability = _model_applicability(case.parameters)
    arrays["physics_model"] = np.asarray(applicability["model"])
    arrays["model_applicability"] = np.asarray(_canonical(applicability))
    for name in ("config_hash", "semantic_hash", "geometry_hash"):
        arrays[name] = np.asarray(getattr(case, name))
    arrays["operator_rule_hash"] = np.asarray(
        _sha256_bytes(RULE_HASH_TEXT.encode())
    )
    arrays["matrix_fingerprint"] = np.asarray(case.operator_hash)
    arrays["original_run_id"] = np.asarray(run_id)


def _check_accepted(committed):
    if not getattr(committed, "accepted", False):
        raise ValueError("output requires an accepted state")
    cache_tag = getattr(committed.cache, "tag", None)
    if cache_tag is None:
        raise ValueError("output requires a matching accepted cache")
    if (
        cache_tag.step != committed.state.step
        or cache_tag.generation != committed.state.generation
        or cache_tag.time != committed.state.time
        or cache_tag.stage not in ("initial", "restart", "endpoint")
    ):
        raise ValueError("output requires a matching accepted cache")


OPTIONAL_CHECK_EVIDENCE = (
    "modal_work",
    "sample_work",
    "transform_difference",
    "absolute_final_terms",
    "upstream_bound",
    "arithmetic_bound",
    "compatibility_bound",
    "configured_threshold",
)


def _check_json(check):
    result = {
        "name": check.name,
        "population": check.population,
        "component": getattr(check, "component", ""),
        "producer": getattr(check, "producer", "diagnostics"),
        "region": check.region,
        "selected_count": int(getattr(check, "selected", 0)),
        "step": check.tag.step,
        "generation": check.tag.generation,
        "trial": check.tag.trial,
        "spectral_context": check.tag.spectral_context,
        "time": typed_value(check.tag.time),
        "stage": check.tag.stage,
        "raw": typed_value(check.raw),
        "reference": typed_value(check.reference),
        "threshold": typed_value(check.threshold),
        "unit": check.unit,
        "value_unit": getattr(check, "value_unit", check.unit),
        "norm": _norm_text(check.norm),
        "gate": "hard" if check.hard else "observe",
        "status": (
            "UNAVAILABLE"
            if not check.available
            else ("PASS" if check.passed else "FAIL")
        ),
        "reason": getattr(check, "reason", ""),
        "worst_index": int(getattr(check, "worst", 0)),
    }
    for name in OPTIONAL_CHECK_EVIDENCE:
        value = getattr(check, name, None)
        if value is not None:
            result[name] = typed_value(value)
    return result


def _step_name(prefix, step, suffix):
    return f"{prefix}{int(step):010d}{suffix}"


def _cumulative_json(cumulative):
    boundary = cumulative.boundary_integrals
    result = {
        "baseline": {
            "totals": {
                name: _json_value(getattr(cumulative.baseline.totals, name))
                for name in _TOTALS_KEYS
            },
            "momentum": {
                name: _json_value(getattr(cumulative.baseline.momentum, name))
                for name in _MOMENTUM_KEYS
            },
            "step": int(cumulative.baseline.step),
            "time": float(cumulative.baseline.time),
        },
        "integrals": [_json_value(ledger) for ledger in cumulative.integrals],
        "boundary_integrals": {
            "number": _json_value(boundary.number),
            "kinetic": _json_value(boundary.kinetic),
            "internal_perp": _json_value(boundary.internal_perp),
            "internal_parallel": _json_value(boundary.internal_parallel),
            "field_energy": float(boundary.field_energy),
            "total_energy": float(boundary.total_energy),
        },
        "accepted_intervals": int(cumulative.accepted_intervals),
        "last_trial": int(cumulative.last_trial),
        "last_step": int(cumulative.last_step),
        "last_time": float(cumulative.last_time),
    }
    applicability = getattr(cumulative, "applicability", None)
    if applicability is not None:
        result["applicability"] = _json_value(applicability)
    return result


def _copy_captured(source, target, expected, label):
    source = Path(source)
    target = Path(target)
    if sha256_file(source) != expected:
        raise ValueError(f"captured input hash mismatch: {label}")
    shutil.copyfile(source, target)
    if sha256_file(target) != expected:
        raise ValueError(f"provenance copy hash mismatch: {label}")


def _archive_source_snapshot(directory, build):
    records = build.get("versions", {}).get("source_files", {})
    if not records:
        raise ValueError("build source snapshot is missing")
    project_root = Path(__file__).resolve().parents[2]
    for module, record in records.items():
        if not isinstance(record, dict):
            record = {"path": record, "sha256": record}
        if "relative" in record:
            relative = Path(record["relative"])
        elif module == "fomhd":
            relative = Path("src/fomhd/__init__.py")
        elif "/" in module:
            relative = Path(module)
        else:
            parts = module.split(".")
            if not parts or parts[0] != "fomhd":
                raise ValueError(f"invalid source module identity: {module}")
            relative = Path("src", *parts).with_suffix(".py")
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid source snapshot path: {relative}")
        if not (
            relative in {Path("pyproject.toml"), Path("uv.lock")}
            or relative.is_relative_to(Path("src/fomhd"))
        ):
            raise ValueError(f"source snapshot path is outside src/fomhd: {relative}")
        source = Path(record["path"]).resolve()
        if not source.is_relative_to(project_root):
            raise ValueError(f"source snapshot source is outside checkout: {source}")
        target = directory / "provenance/source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_captured(
            source,
            target,
            record["sha256"],
            f"source:{module}",
        )


class Run:
    def __init__(
        self, case, build_manifest=None, require_project_snapshot=True
    ):
        self.case = case
        self.directory = Path(case.output)
        if self.directory.is_symlink():
            raise ValueError(f"RunDirectoryNotEmpty: {self.directory}")
        if self.directory.exists():
            if not self.directory.is_dir() or any(self.directory.iterdir()):
                raise ValueError(f"RunDirectoryNotEmpty: {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock_fd = os.open(
            self.directory / "lock",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        try:
            for name in (
                "inputs",
                "provenance",
                "logs",
                "snapshots",
                "checkpoints",
                "failure",
            ):
                (self.directory / name).mkdir()
            for key, source, target in (
                ("case", case.case_path, self.directory / "inputs/case.toml"),
                (
                    "geometry",
                    case.geometry_path,
                    self.directory / "inputs/geometry.npz",
                ),
                (
                    "initial",
                    case.initial_path,
                    self.directory / "inputs/initial.npz",
                ),
                (
                    "source",
                    case.source_path,
                    self.directory / "inputs/source_manifest.json",
                ),
            ):
                _copy_captured(
                    source,
                    target,
                    case.input_hashes[key],
                    key,
                )
            _publish_json(
                self.directory / "inputs/resolved.json", case.resolved
            )
            stamp = time_module.strftime(
                "%Y%m%dT%H%M%SZ", time_module.gmtime()
            )
            self.run_id = f"{case.name}__{stamp}__{uuid.uuid4().hex[:8]}"
            build = _load_build_manifest(
                build_manifest,
                require_project_snapshot=require_project_snapshot,
            )
            _archive_source_snapshot(self.directory, build)
            _publish_json(self.directory / "provenance/build.json", build)
            self.manifest = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "status": "initializing",
                "model": MODEL_ID,
                "model_variant": _model_applicability(
                    case.parameters
                )["model"],
                "model_applicability": _model_applicability(
                    case.parameters
                ),
                "units": "SI",
                "layout_version": LAYOUT_VERSION,
                "integrator": getattr(
                    case.parameters, "integrator", "rk4"
                ),
                "time_guard": {
                    "policy": getattr(case.parameters, "dt_guard", "error"),
                    "minimum_samples_per_cycle": getattr(
                        case.parameters, "minimum_samples_per_cycle", 32.0
                    ),
                    "max_amplitude_error": getattr(
                        case.parameters, "max_amplitude_error", 0.01
                    ),
                    "max_phase_error_rad": getattr(
                        case.parameters, "max_phase_error_rad", 0.1
                    ),
                },
                "config_hash": case.config_hash,
                "resolved_config_sha256": sha256_file(self.directory / "inputs/resolved.json"),
                "semantic_hash": case.semantic_hash,
                "geometry_hash": case.geometry_hash,
                "matrix_fingerprint": case.operator_hash,
                "input_hashes": case.input_hashes,
                "source": case.source,
                "build": build,
                "backend": case.backend,
                "precision": "fp64/complex128",
                "last_accepted_step": None,
                "last_accepted_time": None,
                "durable_checkpoint": None,
                "history_columns": (
                    "step,time_s,number_i,number_e,kinetic_i_J,kinetic_e_J,"
                    "ion_internal_J,electron_perp_J,electron_parallel_J,"
                    "magnetic_J,longitudinal_J,total_energy_J,topology_Wb,"
                    "physical_flux_Wb,Lz_kg_m2_per_s,elapsed_s"
                ),
            }
            job = os.environ.get("SLURM_JOB_ID")
            if job:
                self.manifest["slurm_job_id"] = job
            self._publish_manifest()
            write_text(
                self.directory / "history.csv",
                self.manifest["history_columns"] + "\n",
            )
            for name in ("checks.jsonl", "observations.jsonl"):
                write_text(self.directory / name, "")
            write_text(self.directory / "logs/run.log", "Run initialized.\n")
            self.last_history = None
        except Exception:
            os.close(self.lock_fd)
            self.lock_fd = -1
            raise

    def close(self):
        if getattr(self, "lock_fd", -1) >= 0:
            os.close(self.lock_fd)
            self.lock_fd = -1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        return self.directory

    def _publish_manifest(self):
        _publish_json(self.directory / "run.json", self.manifest, True)

    def parent(self, parent):
        self.manifest["parent"] = _json_value(parent)
        self._publish_manifest()

    def status(self, status):
        self.manifest["status"] = status
        self._publish_manifest()
        with (self.directory / "logs/run.log").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(status + "\n")
            handle.flush()

    def time_estimate(self, report):
        """Persist the scalar time-step guard report for this accepted trial."""
        with (self.directory / "time_estimates.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(_json_value(report), allow_nan=False) + "\n")
            handle.flush()

    def diagnostics(self, state, checks, totals, momentum, elapsed):
        if self.last_history is not None and state.step != self.last_history + 1:
            raise ValueError("duplicate or noncontiguous accepted history")
        if not checks:
            raise ValueError("accepted result has no diagnostic records")
        with (self.directory / "checks.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            for check in checks:
                handle.write(
                    json.dumps(_check_json(check), allow_nan=False) + "\n"
                )
            handle.flush()
        values = [
            state.step,
            state.time,
            totals.number[0],
            totals.number[1],
            totals.kinetic[0],
            totals.kinetic[1],
            totals.ion_internal,
            totals.electron_perp,
            totals.electron_parallel,
            totals.magnetic,
            totals.longitudinal,
            totals.energy,
            totals.topology,
            totals.physical_flux,
            momentum.angular,
            float(elapsed),
        ]
        if not all(
            math.isfinite(float(value))
            for value in values
            if not isinstance(value, int)
        ):
            raise ValueError("accepted diagnostics are nonfinite")
        with (self.directory / "history.csv").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(
                ",".join(
                    str(int(value))
                    if index in (0,)
                    else format(float(value), ".17g")
                    if not isinstance(value, int)
                    else str(value)
                    for index, value in enumerate(values)
                )
                + "\n"
            )
            handle.flush()
        observations = self.case.resolved["diagnostics"].get(
            "mode_observations", []
        )
        if observations:
            arrays = encode_state(state, self.case.geometry)
            modes = 2 * int(self.case.parameters.K) + 1
            for observation in observations:
                field = str(observation["field"])
                component = str(observation["component"])
                if field == "B":
                    field = f"B_{component}"
                if field not in arrays:
                    raise ValueError(f"unknown observation field: {field}")
                array = arrays[field]
                component_index = 0
                if array.ndim == 4:
                    if component == "R":
                        component_index = 0
                    elif component == "phi":
                        component_index = 1
                    elif component == "Z":
                        component_index = 2
                    else:
                        raise ValueError("unknown observation vector component")
                elif component not in ("scalar", field.rsplit("_", 1)[-1]):
                    raise ValueError("scalar observation requires scalar component")
                i, j = map(int, observation["entity_key"])
                ny, nx = array.shape[-2:]
                if not (0 <= i < nx and 0 <= j < ny):
                    raise ValueError("observation entity outside grid")
                k = int(observation["mode"]) + int(self.case.parameters.K)
                if array.ndim == 4:
                    value = array[component_index, k, j, i]
                else:
                    value = array[k, j, i]
                with (self.directory / "observations.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step": int(state.step),
                                "time": float(state.time),
                                "definition": observation,
                                "real": float(value.real),
                                "imag": float(value.imag),
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
        self.last_history = int(state.step)
        self.manifest["last_accepted_step"] = int(state.step)
        self.manifest["last_accepted_time"] = float(state.time)
        self._publish_manifest()

    def snapshot(self, committed, frame=None):
        _check_accepted(committed)
        arrays = encode_state(committed.state, self.case.geometry)
        _identity(
            arrays,
            committed,
            self.case,
            self.run_id,
            self.case.parameters.K,
            self.case.compute,
        )
        if frame is not None:
            if frame.tag != committed.cache.tag:
                raise ValueError("snapshot frame is stale")
            modes = 2 * int(self.case.parameters.K) + 1
            for name, value, entities, reason in (
                (
                    "derived_E",
                    frame.cache.E,
                    int(self.case.operators.Ne),
                    "not applicable to this model",
                ),
                (
                    "derived_phi",
                    frame.cache.phi,
                    int(self.case.operators.Nc),
                    "not_applicable: massless quasineutral branch has no "
                    "electrostatic potential",
                ),
            ):
                arrays[f"{name}_available"] = np.asarray(
                    value is not None, dtype=np.bool_
                )
                if value is None:
                    arrays[f"{name}_reason"] = np.asarray(reason)
                else:
                    arrays[name] = _host_array(value).reshape(modes, entities)
            if self.case.residual_arrays:
                for key, value in frame.residuals.items():
                    arrays[f"derived_residual_{key}"] = _host_array(value)
            solves = list(getattr(frame.cache, "poisson_solves", []))
            transverse = getattr(frame.cache, "transverse_solve", None)
            if transverse is not None:
                solves.append(transverse)
            for index, solve in enumerate(solves):
                prefix = (
                    "derived_darwin"
                    if transverse is not None and index == len(solves) - 1
                    else f"derived_poisson_{index}"
                )
                arrays[f"{prefix}_singular_values"] = _host_array(
                    solve.singular_values
                )
                arrays[f"{prefix}_row_scale"] = _host_array(solve.row_scale)
                arrays[f"{prefix}_column_scale"] = _host_array(
                    solve.column_scale
                )
                arrays[f"{prefix}_physical_residual"] = _host_array(
                    solve.residual
                )
                arrays[f"{prefix}_backward_error"] = np.asarray(
                    solve.backward_error, dtype=np.float64
                )
                arrays[f"{prefix}_refinement_steps"] = np.asarray(
                    solve.refinement_steps, dtype=np.uint64
                )
        elif self.case.residual_arrays:
            raise ValueError(
                "requested residual arrays require an accepted diagnostic frame"
            )
        target = self.directory / "snapshots" / _step_name(
            "state_", committed.state.step, ".npz"
        )
        atomic_publish(target, lambda path: _npz_write(path, arrays))

    def checkpoint(self, committed, cumulative, context):
        _check_accepted(committed)
        if (
            cumulative.last_step != committed.state.step
            or cumulative.last_time != committed.state.time
        ):
            raise ValueError("checkpoint ledger endpoint mismatch")
        durable = self.manifest.get("durable_checkpoint")
        if durable and durable.get("step") == committed.state.step:
            pointer = read_json(self.directory / "checkpoints/latest.json")
            if pointer != durable:
                raise ValueError("durable checkpoint pointer does not match manifest")
            path = self.directory / "checkpoints" / durable["file"]
            if sha256_file(path) != durable["sha256"]:
                raise ValueError("durable checkpoint metadata does not match file")
            return
        context.compute.synchronize()
        arrays = encode_state(committed.state, self.case.geometry)
        _identity(
            arrays,
            committed,
            self.case,
            self.run_id,
            self.case.parameters.K,
            context.compute,
        )
        cumulative_value = _cumulative_json(cumulative)
        arrays["cumulative_json"] = np.asarray(_canonical(cumulative_value))
        arrays["baseline_divB"] = encode_vertex(
            cumulative.baseline.divB,
            self.case.geometry,
            context.p.K,
        )
        arrays["last_source_id"] = np.asarray(
            int(getattr(context, "next_source", 1)) - 1, dtype=np.uint64
        )
        arrays["last_trial_id"] = np.asarray(
            int(getattr(context, "next_trial", 1)) - 1, dtype=np.uint64
        )
        path = self.directory / "checkpoints" / _step_name(
            "restart_", committed.state.step, ".npz"
        )
        digest = atomic_publish(path, lambda p: _npz_write(p, arrays))
        latest = {
            "file": path.name,
            "sha256": digest,
            "step": int(committed.state.step),
            "time": float(committed.state.time),
            "accepted": True,
            "schema_version": SCHEMA_VERSION,
        }
        for name in ("history.csv", "checks.jsonl"):
            descriptor = os.open(
                self.directory / name, os.O_RDONLY | os.O_NOFOLLOW
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        _publish_json(self.directory / "checkpoints/latest.json", latest, True)
        self.manifest["durable_checkpoint"] = latest
        self._publish_manifest()

    def failure(self, error, accepted_step, phase="pre_commit_input"):
        labels = {
            "pre_commit_input": (
                "pre_commit_or_input",
                "failed_before_commit",
                False,
            ),
            "numerical_trial": (
                "rejected_numerical_trial",
                "failed_numerical_trial",
                False,
            ),
            "post_commit_output": (
                "output_after_commit",
                "io_failed_after_commit",
                True,
            ),
        }
        if phase not in labels:
            raise ValueError(f"unknown runtime phase: {phase}")
        failure_class, status, accepted_trial = labels[phase]
        identity = datetime_stamp = time_module.strftime(
            "%Y%m%dT%H%M%S", time_module.gmtime()
        )
        record = {
            "error": str(error),
            "last_accepted_step": accepted_step,
            "accepted_trial": accepted_trial,
            "runtime_phase": phase,
            "failure_class": failure_class,
            "last_durable_checkpoint": self.manifest.get("durable_checkpoint"),
        }
        if isinstance(error, Failure):
            record["code"] = error.code
            report = getattr(error, "time_step_report", None)
            if report is not None:
                record["time_step_report"] = _json_value(report)
            record["checks"] = [
                _check_json(check) for check in getattr(error, "checks", [])
            ]
            record["stage"] = _json_value(getattr(error, "stage", None))
            partial = getattr(error, "partial_arrays", {})
            record["partial_array_keys"] = sorted(partial)
            record["partial_array_count"] = len(partial)
            if partial:
                arrays = {
                    key: _host_array(value) for key, value in partial.items()
                }
                partial_path = (
                    self.directory
                    / "failure"
                    / f"{failure_class}__{uuid.uuid4().hex}.npz"
                )
                try:
                    digest = atomic_publish(
                        partial_path,
                        lambda path: _npz_write(
                            path, arrays, allow_nonfinite=True
                        ),
                    )
                    record["partial_file"] = partial_path.name
                    record["partial_sha256"] = digest
                    record["partial_published"] = True
                except Exception as partial_error:
                    record["partial_published"] = False
                    record["partial_write_error"] = str(partial_error)
            else:
                record["partial_published"] = False
        if isinstance(error, PublicationError):
            record["publication_phase"] = error.phase
            record["target_may_be_published"] = error.published
            record["temporary"] = str(error.temporary)
        stamp = time_module.time_ns()
        target = self.directory / "failure" / f"failure_{stamp}_{uuid.uuid4().hex}.json"
        _publish_json(target, record)
        self.manifest["failure"] = record
        self.status(status)
        return record


def _load_build_manifest(
    build_manifest, require_project_snapshot=True
):
    source_files = {}
    package_dir = Path(__file__).resolve().parent
    if package_dir.name != "fomhd" or package_dir.parent.name != "src":
        raise ValueError("project package must be imported from src/fomhd")
    project_dir = package_dir.parents[1]
    for filename in sorted(package_dir.glob("*.py")):
        if not filename.is_file():
            continue
        relative = f"src/fomhd/{filename.name}"
        source_files[relative] = {
            "path": str(filename),
            "relative": relative,
            "sha256": sha256_file(filename),
        }
    for filename in ("pyproject.toml", "uv.lock"):
        path = project_dir / filename
        if not path.is_file():
            if require_project_snapshot:
                raise ValueError(
                    f"build source snapshot is missing {filename}"
                )
            continue
        source_files[filename] = {
            "path": str(path.resolve()),
            "relative": filename,
            "sha256": sha256_file(path),
        }
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "fomhd_model": MODEL_ID,
        "layout_version": LAYOUT_VERSION,
        "rule_hash": _sha256_bytes(RULE_HASH_TEXT.encode()),
        "source_files": source_files,
    }
    try:
        import cupy

        versions["cupy"] = cupy.__version__
    except ImportError:
        versions["cupy"] = "not imported"
    try:
        import warp

        versions["warp"] = getattr(warp, "__version__", "unknown")
    except ImportError:
        versions["warp"] = "not imported"
    if build_manifest is None:
        return {"versions": versions, "status": "runtime-generated"}
    if isinstance(build_manifest, (str, Path)):
        value = read_json(build_manifest)
        value["manifest_sha256"] = sha256_file(build_manifest)
        return value
    return {**dict(build_manifest), "versions": versions}


def _array_scalar(array, dtype):
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError("metadata value must be a scalar")
    if value.dtype != np.dtype(dtype):
        raise ValueError(
            f"metadata dtype mismatch: got {value.dtype}, expected {dtype}"
        )
    return value.item()


def _array_text(array):
    value = np.asarray(array)
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError("metadata value must be a Unicode scalar")
    text = str(value.item())
    if "\x00" in text:
        raise ValueError("metadata contains an embedded NUL")
    return text


_CUMULATIVE_KEYS = {
    "baseline",
    "integrals",
    "boundary_integrals",
    "accepted_intervals",
    "last_trial",
    "last_step",
    "last_time",
}
_BASELINE_KEYS = {"totals", "momentum", "step", "time"}
_TOTALS_KEYS = {
    "number",
    "kinetic",
    "ion_internal",
    "electron_perp",
    "electron_parallel",
    "magnetic",
    "longitudinal",
    "energy",
    "topology",
    "physical_flux",
}
_MOMENTUM_KEYS = {
    "value",
    "outward",
    "wall_reaction",
    "angular",
    "outward_torque",
    "wall_torque",
    "angular_samples",
}
_LEDGER_KEYS = {
    "electric_node",
    "electric_edge",
    "pressure_kinetic",
    "pressure_internal",
    "magnetic",
    "wall",
    "convection",
    "perp_rate",
    "parallel_rate",
    "spectral_kinetic",
    "spectral_perp",
    "spectral_parallel",
}
_BOUNDARY_KEYS = {
    "number",
    "kinetic",
    "internal_perp",
    "internal_parallel",
    "field_energy",
    "total_energy",
}


def _exact_fields(value, fields, name):
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} has an invalid field set")
    return value


def _json_float(value, name):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return value


def _json_uint(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _json_vector(value, length, name):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must have {length} entries")
    for index, item in enumerate(value):
        _json_float(item, f"{name}[{index}]")
    return value


def _validate_totals(value):
    _exact_fields(value, _TOTALS_KEYS, "cumulative baseline totals")
    _json_vector(value["number"], 2, "totals.number")
    _json_vector(value["kinetic"], 2, "totals.kinetic")
    for name in _TOTALS_KEYS - {"number", "kinetic"}:
        _json_float(value[name], f"totals.{name}")


def _validate_momentum(value):
    _exact_fields(value, _MOMENTUM_KEYS, "cumulative baseline momentum")
    for name in ("value", "outward", "wall_reaction"):
        _json_vector(value[name], 3, f"momentum.{name}")
    for name in ("angular", "outward_torque", "wall_torque"):
        _json_float(value[name], f"momentum.{name}")
    _json_uint(value["angular_samples"], "momentum.angular_samples")


def _cumulative_from_json(value):
    if (
        not isinstance(value, dict)
        or not _CUMULATIVE_KEYS.issubset(value)
        or set(value) - (_CUMULATIVE_KEYS | {"applicability"})
    ):
        raise ValueError("cumulative has an invalid field set")
    baseline = _exact_fields(value["baseline"], _BASELINE_KEYS, "baseline")
    _validate_totals(baseline["totals"])
    _validate_momentum(baseline["momentum"])
    _json_uint(baseline["step"], "baseline.step")
    _json_float(baseline["time"], "baseline.time")
    integrals = value["integrals"]
    if not isinstance(integrals, list) or len(integrals) != 2:
        raise ValueError("cumulative integrals must contain exactly two species")
    parsed_integrals = []
    for species, ledger in enumerate(integrals):
        _exact_fields(ledger, _LEDGER_KEYS, f"integrals[{species}]")
        for name in _LEDGER_KEYS:
            _json_float(ledger[name], f"integrals[{species}].{name}")
        parsed_integrals.append(SimpleNamespace(**ledger))
    boundary = _exact_fields(
        value["boundary_integrals"], _BOUNDARY_KEYS, "boundary_integrals"
    )
    for name in ("number", "kinetic", "internal_perp", "internal_parallel"):
        _json_vector(boundary[name], 2, f"boundary_integrals.{name}")
    for name in ("field_energy", "total_energy"):
        _json_float(boundary[name], f"boundary_integrals.{name}")
    accepted_intervals = _json_uint(
        value["accepted_intervals"], "accepted_intervals"
    )
    last_trial = _json_uint(value["last_trial"], "last_trial")
    last_step = _json_uint(value["last_step"], "last_step")
    last_time = _json_float(value["last_time"], "last_time")
    if accepted_intervals == 0:
        if (
            baseline["step"] != 0
            or last_step != 0
            or last_time != baseline["time"]
            or last_trial != 0
            or any(ledger[name] != 0.0 for ledger in integrals for name in _LEDGER_KEYS)
            or any(
                (boundary[name][component] != 0.0
                 for name in ("number", "kinetic", "internal_perp", "internal_parallel")
                 for component in range(2))
            )
            or boundary["field_energy"] != 0.0
            or boundary["total_energy"] != 0.0
        ):
            raise ValueError(
                "initial checkpoint cumulative ledger must be complete and zero"
            )
    return SimpleNamespace(
        baseline=SimpleNamespace(
            totals=SimpleNamespace(**baseline["totals"]),
            momentum=SimpleNamespace(**baseline["momentum"]),
            step=baseline["step"],
            time=baseline["time"],
        ),
        integrals=parsed_integrals,
        boundary_integrals=SimpleNamespace(**boundary),
        accepted_intervals=accepted_intervals,
        last_trial=last_trial,
        last_step=last_step,
        last_time=last_time,
        applicability=value.get("applicability"),
    )


def read_restart(path, case):
    path = Path(path)
    allowed = {
        "n_i",
        "mom_i",
        "p_i",
        "n_e",
        "mom_e",
        "p_perp_e",
        "p_parallel_e",
        "B_R",
        "B_phi",
        "B_Z",
        "step",
        "generation",
        "time",
        "dt",
        "start_time",
        "accepted",
        "schema_version",
        "layout_version",
        "model_id",
        "physics_model",
        "model_applicability",
        "config_hash",
        "semantic_hash",
        "geometry_hash",
        "operator_rule_hash",
        "matrix_fingerprint",
        "original_run_id",
        "cumulative_json",
        "baseline_divB",
        "last_source_id",
        "last_trial_id",
    }
    arrays = read_npz(path, case.limits, allowed)
    if not _array_scalar(arrays["accepted"], np.bool_):
        raise ValueError("checkpoint was not an accepted A0 state")
    if (
        _array_scalar(arrays["schema_version"], np.uint64) != SCHEMA_VERSION
        or _array_scalar(arrays["layout_version"], np.uint64) != LAYOUT_VERSION
        or _array_text(arrays["model_id"]) != MODEL_ID
    ):
        raise ValueError("checkpoint schema/model mismatch")
    expected_applicability = _model_applicability(case.parameters)
    if (
        _array_text(arrays["physics_model"])
        != expected_applicability["model"]
        or parse_json(_array_text(arrays["model_applicability"]))
        != expected_applicability
    ):
        raise ValueError(
            "restart electron model/applicability mismatch: "
            f"checkpoint={_array_text(arrays['physics_model'])}, "
            f"configured={expected_applicability['model']}"
        )
    expected = {
        "semantic_hash": case.semantic_hash,
        "geometry_hash": case.geometry_hash,
        "matrix_fingerprint": case.operator_hash,
        "operator_rule_hash": _sha256_bytes(RULE_HASH_TEXT.encode()),
    }
    for name, value in expected.items():
        if _array_text(arrays[name]) != value:
            raise ValueError(f"restart {name} mismatch")
    if (
        _array_scalar(arrays["dt"], np.float64) != case.parameters.dt
        or _array_scalar(arrays["start_time"], np.float64)
        != case.parameters.start_time
    ):
        raise ValueError("restart time configuration mismatch")
    cumulative = _cumulative_from_json(
        parse_json(_array_text(arrays["cumulative_json"]))
    )
    stored_applicability = cumulative.applicability
    if int(getattr(case.parameters, "electron_inertia", 0)) == 0:
        if stored_applicability != expected_applicability:
            raise ValueError(
                "restart cumulative applicability mismatch"
            )
    elif stored_applicability not in (None, expected_applicability):
        raise ValueError("restart cumulative applicability mismatch")
    state = decode_state(arrays, case.geometry, case.parameters.K)
    _upload_state(state, case.compute)
    state.step = _array_scalar(arrays["step"], np.uint64)
    state.time = _array_scalar(arrays["time"], np.float64)
    state.generation = _array_scalar(arrays["generation"], np.uint64)
    if int(getattr(case.parameters, "electron_inertia", 0)) == 0:
        from .massless import canonicalize_state
        from .types import Context

        context = Context(
            case.parameters,
            case.geometry,
            case.operators,
            case.compute,
            case.parameters.Nphi,
        )
        state = canonicalize_state(state, context)
    from .core import time_at_step

    if (
        state.step > case.parameters.nsteps
        or state.time != time_at_step(case.parameters, state.step)
        or state.generation == (1 << 64) - 1
    ):
        raise ValueError("restart scheduler identity mismatch")
    state.generation += 1
    cumulative.baseline.divB = case.compute.array(
        decode_vertex(
            arrays["baseline_divB"], case.geometry, case.parameters.K
        ),
        np.complex128,
    )
    if (
        cumulative.baseline.step > state.step
        or cumulative.baseline.time
        != time_at_step(case.parameters, cumulative.baseline.step)
        or cumulative.last_step != state.step
        or cumulative.last_time != state.time
        or cumulative.accepted_intervals
        != state.step - cumulative.baseline.step
    ):
        raise ValueError("restart baseline or interval identity mismatch")
    if (
        cumulative.accepted_intervals == 0
        and (
            cumulative.last_trial != 0
            or cumulative.integrals
            and any(
                abs(float(value)) > 0
                for ledger in cumulative.integrals
                for value in vars(ledger).values()
                if isinstance(value, (int, float))
            )
        )
    ):
        raise ValueError("initial checkpoint contains nonzero cumulative ledger")
    if cumulative.accepted_intervals and cumulative.last_trial == 0:
        raise ValueError("restart interval trial is missing")
    trial = _array_scalar(arrays["last_trial_id"], np.uint64)
    source = _array_scalar(arrays["last_source_id"], np.uint64)
    if (
        trial == (1 << 64) - 1
        or source == (1 << 64) - 1
        or trial < cumulative.last_trial
    ):
        raise ValueError("restart counters are invalid")
    digest = sha256_file(path)
    latest_path = path.parent / "latest.json"
    if latest_path.exists():
        latest = read_json(latest_path)
        if (
            latest.get("file") == path.name
            and latest.get("sha256") != digest
        ):
            raise ValueError("checkpoint file hash does not match durable pointer")
    parent = {
        "checkpoint": str(path.absolute()),
        "sha256": digest,
        "original_run_id": _array_text(arrays["original_run_id"]),
        "original_config_hash": _array_text(arrays["config_hash"]),
    }
    return Restart(state=state, cumulative=cumulative, parent=parent,
                   trial=trial, source=source)


def inspect(run_dir):
    run_dir = Path(run_dir)
    result = read_json(run_dir / "run.json")
    history = read_text(run_dir / "history.csv", 64 << 20)
    checks = read_text(run_dir / "checks.jsonl", 512 << 20)
    interrupted = bool(history and not history.endswith("\n")) or bool(
        checks and not checks.endswith("\n")
    )
    count = 0
    for line in checks.splitlines():
        if not line:
            continue
        try:
            parse_json(line)
            count += 1
        except Exception:
            interrupted = True
            break
    rows = history.splitlines()
    last = None
    for line in rows[1:] if rows else []:
        if not line:
            continue
        try:
            step_text, separator, _ = line.partition(",")
            step = int(step_text)
            if separator != "," or (last is not None and step != last + 1):
                interrupted = True
        except ValueError:
            interrupted = True
            break
        last = step
    durable = result.get("durable_checkpoint")
    if durable is not None:
        path = run_dir / "checkpoints" / durable["file"]
        if sha256_file(path) != durable["sha256"]:
            raise ValueError("durable checkpoint hash mismatch")
    for entry in run_dir.rglob("*"):
        if ".tmp." in entry.name:
            interrupted = True
    result["inspection"] = {
        "interrupted_output": interrupted,
        "diagnostic_records": count,
        "last_history_step": last,
    }
    return result
