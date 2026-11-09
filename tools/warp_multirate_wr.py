#!/usr/bin/env python3
"""Bounded CPU/CuPy benchmark for the selected massless A1-WR window."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import platform
import resource
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
for runtime_path in (
    SOURCE_ROOT / "src",
    SOURCE_ROOT / "tests" / "warp",
    SOURCE_ROOT / "tools",
):
    if str(runtime_path) not in sys.path:
        sys.path.insert(0, str(runtime_path))

from fomhd import massless, massless_field, stage  # noqa: E402
from fomhd import multirate_wr_physics as wr  # noqa: E402
from fomhd import time_multirate_wr  # noqa: E402
from fomhd.types import Failure, StageTag  # noqa: E402
from wr_benchmark_fixture import periodic_inputs  # noqa: E402


REQUIRED = (
    "case",
    "points",
    "K",
    "amplitude",
    "nonuniform",
    "seed_mode",
    "sentinel",
    "h",
    "M",
    "s",
    "macros",
    "device",
    "coupling",
    "rtol",
    "max_iterations",
    "wall_seconds",
)
DEFAULT_H = 1.9794733885623e-10
NOMINAL_TARGET_T = 4.05396149977559e-7
MAX_SNAPSHOT_BYTES = 64 << 20
RK4_NODES = (0.0, 0.5, 0.5, 1.0)
HOST_RSS_LIMIT_BYTES = 16 << 30
DEVICE_USED_LIMIT_BYTES = 8 << 30


def _positive_real(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive real")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real")
    return result


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive non-boolean integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive non-boolean integer")
    return int(value)


def validate_config(config):
    """Validate one exact runner configuration without building a context."""

    if not isinstance(config, dict):
        raise ValueError("input must be a JSON object")
    keys = set(config)
    missing = sorted((set(REQUIRED) - {"h"}) - keys)
    unknown = sorted(keys - set(REQUIRED))
    if missing:
        raise ValueError("missing controls: " + ", ".join(missing))
    if unknown:
        raise ValueError("unknown controls: " + ", ".join(unknown))
    if config["case"] not in ("periodic", "east"):
        raise ValueError("case must be periodic or east")
    points = _positive_integer(config["points"], "points")
    if config["case"] == "periodic" and points < 9:
        raise ValueError("periodic points must be at least nine")
    if points < 4:
        raise ValueError("points must be at least four")
    if points % 2 == 0 and config["case"] == "periodic":
        raise ValueError("periodic points must be odd")
    K = config["K"]
    if isinstance(K, bool) or not isinstance(K, int) or K < 0:
        raise ValueError("K must be a nonnegative integer")
    if config["case"] == "periodic" and K != 0:
        raise ValueError(
            "the extracted periodic physical fixture currently exposes K=0; "
            "real K=1 is the parent-owned EAST integration"
        )
    if config["case"] == "east" and K != 1:
        raise ValueError("EAST reuse requires actual K=1")
    for name in ("amplitude", "nonuniform", "sentinel"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if (
        isinstance(config["seed_mode"], bool)
        or not isinstance(config["seed_mode"], int)
    ):
        raise ValueError("seed_mode must be an integer")
    seed_mode = int(config["seed_mode"])
    if seed_mode < 1 or seed_mode > max(1, points // 2):
        raise ValueError("seed_mode must lie in [1, points//2]")
    h = _positive_real(config.get("h", DEFAULT_H), "h")
    M = _positive_integer(config["M"], "M")
    if config["s"] not in (3, 5):
        raise ValueError("s must be 3 or 5")
    _positive_integer(config["macros"], "macros")
    if config["device"] not in ("cpu", "cuda:0"):
        raise ValueError("device must be cpu or cuda:0")
    if config["device"] == "cuda:0" and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("CUDA runs require an owned Slurm allocation")
    if config["coupling"] not in ("wr", "synchronous"):
        raise ValueError("coupling must be wr or synchronous")
    _positive_real(config["rtol"], "rtol")
    _positive_integer(config["max_iterations"], "max_iterations")
    wall_seconds = _positive_real(config["wall_seconds"], "wall_seconds")
    if wall_seconds > 1600.0:
        raise ValueError("wall_seconds must not exceed 1600")
    resolved = dict(config)
    resolved["seed_mode"] = seed_mode
    resolved["h"] = float(h)
    resolved["H"] = float(h) * M
    resolved["nominal_target_T"] = NOMINAL_TARGET_T
    return resolved


def _read_json(path):
    raw = Path(path).read_bytes()

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=reject_duplicates), raw


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_paths():
    """Return every runtime source/document path relative to the source root."""

    return {
        "src/fomhd/multirate_wr_physics.py": Path(wr.__file__),
        "src/fomhd/massless_field.py": Path(massless_field.__file__),
        "src/fomhd/massless.py": Path(massless.__file__),
        "src/fomhd/time_multirate_wr.py": Path(time_multirate_wr.__file__),
        "src/fomhd/stage.py": Path(stage.__file__),
        "tests/warp/wr_benchmark_fixture.py": Path(
            SOURCE_ROOT / "tests" / "warp" / "wr_benchmark_fixture.py"
        ),
        "tests/warp/massless_verification_fixture.py": Path(
            SOURCE_ROOT / "tests" / "warp" / "massless_verification_fixture.py"
        ),
        "tests/warp/test_multirate_wr_physics.py": Path(
            SOURCE_ROOT / "tests" / "warp" / "test_multirate_wr_physics.py"
        ),
        "tests/warp/test_warp_multirate_wr.py": Path(
            SOURCE_ROOT / "tests" / "warp" / "test_warp_multirate_wr.py"
        ),
        "tools/warp_multirate_wr.py": Path(__file__),
        "docs/pseudocode/src/fomhd/multirate_wr_physics.py.md": Path(
            SOURCE_ROOT
            / "docs"
            / "pseudocode"
            / "src"
            / "fomhd"
            / "multirate_wr_physics.py.md"
        ),
        "docs/pseudocode/tools/warp_multirate_wr.py.md": Path(
            SOURCE_ROOT / "docs" / "pseudocode" / "tools" / "warp_multirate_wr.py.md"
        ),
        "docs/pseudocode/tests/warp/test_multirate_wr_physics.py.md": Path(
            SOURCE_ROOT
            / "docs"
            / "pseudocode"
            / "tests"
            / "warp"
            / "test_multirate_wr_physics.py.md"
        ),
        "docs/pseudocode/tests/warp/test_warp_multirate_wr.py.md": Path(
            SOURCE_ROOT
            / "docs"
            / "pseudocode"
            / "tests"
            / "warp"
            / "test_warp_multirate_wr.py.md"
        ),
    }


def _source_hashes():
    paths = _source_paths()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError("runtime source is absent: " + ", ".join(missing))
    return {name: _sha256(path) for name, path in paths.items()}


def _host_memory():
    rss = None
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        rss = pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        pass
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        peak *= 1
    else:
        peak *= 1024
    return {"host_rss_bytes": rss, "host_peak_rss_bytes": peak}


def _check_resource_sample(sample):
    """Hard-gate actual host RSS and allocated device bytes."""

    host = sample.get("host_rss_bytes")
    if host is not None and int(host) > HOST_RSS_LIMIT_BYTES:
        failure = Failure(
            "ResourceLimit",
            f"host RSS {int(host)} bytes exceeds {HOST_RSS_LIMIT_BYTES}",
        )
        failure.memory_sample = dict(sample)
        raise failure
    device = sample.get("device_used_bytes")
    if device is not None and int(device) > DEVICE_USED_LIMIT_BYTES:
        failure = Failure(
            "ResourceLimit",
            f"device used {int(device)} bytes exceeds "
            f"{DEVICE_USED_LIMIT_BYTES}",
        )
        failure.memory_sample = dict(sample)
        raise failure
    return sample


def _memory_sample(context, name, samples):
    value = {"name": name, "monotonic_s": time.monotonic(), **_host_memory()}
    xp = context.compute.xp if context is not None else np
    if xp is not np:
        pool = xp.get_default_memory_pool()
        value.update(
            device_used_bytes=int(pool.used_bytes()),
            device_reserved_bytes=int(pool.total_bytes()),
            device_limit_bytes=DEVICE_USED_LIMIT_BYTES,
        )
    samples.append(value)
    _check_resource_sample(value)
    return value


def _to_numpy(value, context):
    xp = context.compute.xp
    return np.asarray(value) if xp is np else xp.asnumpy(value)


def _window_mode_amplitudes(context, p, z, target_mode, sentinel_mode):
    nc = context.op.Nc
    nb = context.op.Nb

    def component_amplitudes(block, mode):
        values = _to_numpy(block, context)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[0] != 1:
            raise ValueError("bounded periodic fixture expects one stored mode row")
        samples = values[0]
        return complex(np.fft.fft(samples)[mode % samples.size] / samples.size)

    p_host, z_host = _to_numpy(p, context), _to_numpy(z, context)
    blocks = {
        "ion_n": p_host[:, :nc],
        "ion_mom_R": p_host[:, nc : 2 * nc],
        "ion_mom_phi": p_host[:, 2 * nc : 3 * nc],
        "ion_mom_Z": p_host[:, 3 * nc : 4 * nc],
        "ion_pi": p_host[:, 4 * nc : 5 * nc],
        "electron_perp": z_host[:, :nc],
        "electron_parallel": z_host[:, nc : 2 * nc],
        "B_R": z_host[:, 2 * nc : 2 * nc + nb // 3],
        "B_phi": z_host[:, 2 * nc + nb // 3 : 2 * nc + 2 * nb // 3],
        "B_Z": z_host[:, 2 * nc + 2 * nb // 3 :],
    }
    result = {}
    for mode_name, mode in (("target", target_mode), ("sentinel", sentinel_mode)):
        result[mode_name] = {
            "mode": int(mode),
            "components": {
                name: component_amplitudes(block, mode)
                for name, block in blocks.items()
            },
        }
    return result


def _check_summary(checks):
    rows = []
    for check in checks:
        rows.append(
            {
                "name": str(getattr(check, "name", "")),
                "population": str(getattr(check, "population", "")),
                "unit": str(getattr(check, "unit", "")),
                "hard": bool(getattr(check, "hard", True)),
                "available": bool(getattr(check, "available", True)),
                "passed": bool(getattr(check, "passed", False)),
                "raw": _jsonable(getattr(check, "raw", None)),
                "reference": _jsonable(getattr(check, "reference", None)),
                "threshold": _jsonable(getattr(check, "threshold", None)),
            }
        )
    return {
        "count": len(rows),
        "failed_hard": sum(
            row["hard"] and (not row["available"] or not row["passed"])
            for row in rows
        ),
        "failed_observations": sum(
            not row["hard"] and (not row["available"] or not row["passed"])
            for row in rows
        ),
        "checks": rows,
    }


def _window_evidence(context, result):
    work = result.work.snapshot()
    fine = tuple(result.window.fine_samples)
    residual_maxima = {}
    for record in getattr(result, "fine_records", ()):
        for name, value in record.residual_max.items():
            residual_maxima[name] = max(residual_maxima.get(name, 0.0), value)
    return {
        "sweeps": int(result.window.sweeps),
        "accepted_node_version": int(result.window.accepted_node_version),
        "s": int(result.window.s),
        "M": int(result.window.M),
        "h": float(result.window.h),
        "H": float(result.window.H),
        "rtol": float(result.window.rtol),
        "residual_max_scaled": float(result.window.residual_max_scaled),
        "raw_norm": float(result.window.raw_norm),
        "scaled_norm": float(result.window.scaled_norm),
        "residual_history": list(result.window.residual_history),
        "endpoint_residual": _jsonable(result.window.endpoint_residual),
        "work": work,
        "source": result.source,
        "clock": result.clock,
        "identity_evidence": result.identity_evidence,
        "caller_unchanged": bool(result.caller_unchanged),
        "ledger_unchanged": bool(result.ledger_unchanged),
        "fine_stage_count": len(fine),
        "record_counts": {
            "accepted_coarse": len(result.coarse_records),
            "accepted_fine": len(result.fine_records),
            "all_coarse_callbacks": len(result.all_coarse_records),
            "all_fine_callbacks": len(result.all_fine_records),
        },
        "stage_output_shapes": _stage_output_shapes(result),
        "fine_residual_max": residual_maxima,
        "electric_exchange_W": [
            {
                "real": None
                if record.electric_exchange_W is None
                else float(record.electric_exchange_W.real),
                "imag": None
                if record.electric_exchange_W is None
                else float(record.electric_exchange_W.imag),
                "error": record.electric_exchange_error,
            }
            for record in getattr(result, "fine_records", ())
        ],
        "endpoint": {
            "tag": asdict(result.endpoint.tag),
            "checks": _check_summary(result.endpoint.checks),
            "production_accepted": bool(result.endpoint.production_accepted),
            "trial_endpoint": bool(result.endpoint.trial_endpoint),
            "interval_transaction_required": bool(
                result.endpoint.interval_transaction_required
            ),
            "source_id": int(result.endpoint.full.source_id),
            "totals": _object_shapes(result.endpoint.totals),
        },
        "version_evidence": result.version_evidence,
    }


def _stage_output_shapes(result):
    coarse = []
    for record in result.coarse_records:
        projection = record.transport_projection
        coarse.append(
            {
                "tag": asdict(record.tag),
                "A": _object_shapes(record.A),
                "C": _object_shapes(record.C),
                "F": _object_shapes(record.F),
                "transport_projection_value": _object_shapes(projection.value),
                "transport_projection_reaction": _object_shapes(
                    projection.reaction
                ),
                "transport_projection_multiplier": _object_shapes(
                    projection.multiplier
                ),
                "transport_projection_multiplier_residual": _object_shapes(
                    projection.multiplier_residual
                ),
                "split_relative_error": float(record.split_relative_error),
            }
        )
    fine = []
    for record in result.fine_records:
        terms = record.frame.ion_em_terms
        fine.append(
            {
                "tag": asdict(record.tag),
                "P": _object_shapes(record.input_P),
                "Z": _object_shapes(record.input_Z),
                "G": _object_shapes(record.G),
                "C": _object_shapes(record.C),
                "electric_force": None
                if terms is None
                else _object_shapes(terms.electric),
                "magnetic_force": None
                if terms is None
                else _object_shapes(terms.magnetic),
                "projected_force": None
                if terms is None
                else _object_shapes(terms.projected),
                "wall_reaction": None
                if terms is None
                else _object_shapes(terms.reaction),
                "multiplier": None
                if terms is None
                else _object_shapes(terms.multiplier),
                "multiplier_residual": None
                if terms is None
                else _object_shapes(terms.multiplier_residual),
            }
        )
    return {"coarse": coarse, "fine": fine}


def _stage_identities(result):
    rows = []
    for record in getattr(result, "coarse_records", ()):
        rows.append(
            {
                "kind": "coarse",
                "tag": asdict(record.tag),
                "window_identity": record.identity.as_dict(),
                "source_id": int(record.full.source_id),
                "split_relative_error": float(record.split_relative_error),
            }
        )
    for record in getattr(result, "fine_records", ()):
        rows.append(
            {
                "kind": "fine",
                "tag": asdict(record.tag),
                "window_identity": record.identity.as_dict(),
                "source_id": int(record.frame.source_id),
                "alias_field_evaluations": int(record.alias_field_evaluations),
                "cache_object_id": int(record.cache_object_id),
                "cache_storage_bytes": int(record.cache_storage_bytes),
            }
        )
    return rows


def _object_shapes(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _object_shapes(item) for key, item in value.items()}
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {
            str(key): _object_shapes(item)
            for key, item in vars(value).items()
        }
    if isinstance(value, (tuple, list)):
        return [_object_shapes(item) for item in value]
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {
            "shape": list(shape),
            "dtype": str(getattr(value, "dtype", "")),
            "bytes": int(getattr(value, "nbytes", 0)),
        }
    if isinstance(value, (int, float, bool, str)) or value is None:
        return _jsonable(value)
    return {"type": type(value).__name__}


def _jsonable(value):
    if isinstance(value, complex) or isinstance(value, np.complexfloating):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _add_arrays(arrays, prefix, value, context):
    if value is None:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _add_arrays(arrays, f"{prefix}_{key}", item, context)
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _add_arrays(arrays, f"{prefix}_{index}", item, context)
        return
    if hasattr(value, "__dict__") and not isinstance(value, type):
        for key, item in vars(value).items():
            _add_arrays(arrays, f"{prefix}_{key}", item, context)
        return
    if hasattr(value, "shape") and bool(getattr(value, "flags", SimpleNamespace(c_contiguous=True)).c_contiguous):
        arrays[prefix] = _to_numpy(value, context).copy()
    elif isinstance(value, (int, float, bool, np.generic)):
        arrays[prefix] = np.asarray(value)


def _encode_arrays(arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    if len(payload) > MAX_SNAPSHOT_BYTES:
        raise Failure(
            "BenchmarkSnapshot",
            f"array snapshot exceeds {MAX_SNAPSHOT_BYTES} bytes",
        )
    return {
        "arrays_npz_base64": base64.b64encode(payload).decode("ascii"),
        "array_bytes": int(len(payload)),
        "array_sha256": hashlib.sha256(payload).hexdigest(),
        "array_names": sorted(arrays),
    }


def _reference_run(context, p0, z0, h, M, macros, factor):
    xp = context.compute.xp
    p, z = p0.copy(), z0.copy()
    H = float(h) * int(M)
    steps_per_macro = int(M) * int(factor)
    sub_h = H / steps_per_macro
    start = float(context.p.start_time)
    windows = []
    stages = []
    total_full = 0
    alias_evaluations = 0
    for macro in range(int(macros)):
        kp, kz = [], []
        for sub in range(steps_per_macro):
            kp, kz = [], []
            for rk, fraction in enumerate(RK4_NODES):
                if rk == 0:
                    sp, sz = p, z
                elif rk == 1:
                    sp, sz = p + 0.5 * sub_h * kp[0], z + 0.5 * sub_h * kz[0]
                elif rk == 2:
                    sp, sz = p + 0.5 * sub_h * kp[1], z + 0.5 * sub_h * kz[1]
                else:
                    sp, sz = p + sub_h * kp[2], z + sub_h * kz[2]
                when = start + macro * H + (sub + fraction) * sub_h
                generation = int(context.next_generation)
                tag = StageTag(
                    step=macro,
                    generation=generation,
                    trial=int(context.next_trial),
                    spectral_context=context.spectral_id,
                    time=when,
                    stage="reference_full_rk4",
                    micro_step=sub,
                    iteration=macro,
                    slab=macro,
                )
                context.next_generation += 1
                context.next_trial += 1
                raw = wr.state_from_packed(
                    context,
                    sp,
                    sz,
                    step=tag.step,
                    generation=tag.generation,
                    time=tag.time,
                )
                alias = rk in (1, 2)
                alias_evaluations += int(alias)
                full = stage.evaluate_stage(raw, tag, context, alias)
                total_full += 1
                p_rate = wr.pack_stage_rhs(full, 0, context)
                _, z_rate = wr.pack_stage_rhs(full, 1, context)
                kp.append(p_rate)
                kz.append(z_rate)
                stages.append(
                    {
                        "tag": asdict(tag),
                        "stage": f"macro={macro},sub={sub},rk={rk}",
                    }
                )
            p = p + sub_h * (kp[0] + 2 * kp[1] + 2 * kp[2] + kp[3]) / 6.0
            z = z + sub_h * (kz[0] + 2 * kz[1] + 2 * kz[2] + kz[3]) / 6.0
        endpoint_time = start + (macro + 1) * H
        endpoint_tag = StageTag(
            step=macro,
            generation=int(context.next_generation),
            trial=int(context.next_trial),
            spectral_context=context.spectral_id,
            time=endpoint_time,
            stage="reference_endpoint",
            micro_step=steps_per_macro - 1,
            iteration=macro,
            slab=macro,
        )
        context.next_generation += 1
        context.next_trial += 1
        endpoint_state = wr.state_from_packed(
            context,
            p,
            z,
            step=endpoint_tag.step,
            generation=endpoint_tag.generation,
            time=endpoint_tag.time,
        )
        endpoint = stage.evaluate_stage(
            endpoint_state, endpoint_tag, context, False
        )
        total_full += 1
        p, z = wr.pack_state(context, endpoint.canonical_state)
        windows.append(
            {
                "macro": macro,
                "end_time_s": endpoint_time,
                "checks": _check_summary(endpoint.checks),
                "source_id": int(endpoint.source_id),
            }
        )
    return {
        "P": p,
        "Z": z,
        "windows": windows,
        "stage_identities": stages,
        "full_stage_calls": total_full,
        "field_rhs_calls": total_full,
        "endpoint_check_calls": int(macros),
        "alias_evaluations": alias_evaluations,
        "diagnostics_scope": (
            "correctness-only full-RHS RK4 with alias at RK4 middle stages; "
            "not a production transaction performance measurement"
        ),
    }


def _comparison(context, left_p, left_z, right_p, right_z):
    p_scale, z_scale = wr.physical_scales(context)
    xp = context.compute.xp
    return {
        "p_max_abs": float(xp.max(xp.abs(left_p - right_p)).item()),
        "p_normalized": float(
            xp.max(xp.abs(left_p - right_p) / p_scale).item()
        ),
        "z_max_abs": float(xp.max(xp.abs(left_z - right_z)).item()),
        "z_normalized": float(
            xp.max(xp.abs(left_z - right_z) / z_scale).item()
        ),
    }


def _build_periodic(config):
    context, state, metadata = periodic_inputs(
        int(config["points"]),
        float(config["amplitude"]),
        float(config["nonuniform"]),
        seed_mode=int(config["seed_mode"]),
        sentinel=float(config["sentinel"]),
        device=config["device"],
    )
    return context, state, metadata


def _run_candidate(config, deadline, arrays, memory_samples):
    context, state, fixture_metadata = _build_periodic(config)
    p_source, z_source = wr.pack_state(context, state)
    p, z = p_source.copy(), z_source.copy()
    start = float(context.p.start_time)
    H = float(config["H"])
    windows = []
    identities = []
    window_evidence = []
    mode_history = []
    last_result = None
    _add_arrays(arrays, "source_P", p_source, context)
    _add_arrays(arrays, "source_Z", z_source, context)
    _memory_sample(context, "candidate_start", memory_samples)
    try:
        for macro in range(int(config["macros"])):
            if time.monotonic() > deadline:
                raise TimeoutError("wall-clock benchmark budget expired")
            base_time = start + macro * H
            state.time = base_time
            adapter = wr.MasslessWRPhysicsAdapter(context, state, slab=macro)
            result = adapter.solve_window(
                p,
                z,
                H=H,
                M=int(config["M"]),
                s=int(config["s"]),
                rtol=float(config["rtol"]),
                max_iterations=int(config["max_iterations"]),
                initial="tangent",
            )
            p, z = result.endpoint.P, result.endpoint.Z
            state = result.endpoint.full.canonical_state
            state.time = base_time + H
            end_time = state.time
            windows.append(
                {
                    "macro": macro,
                    "start_time_s": base_time,
                    "end_time_s": end_time,
                    "sweeps": int(result.window.sweeps),
                    "work": result.work.snapshot(),
                }
            )
            window_evidence.append(_window_evidence(context, result))
            identities.extend(_stage_identities(result))
            mode_history.append(
                {
                    "macro": macro,
                    "time_s": end_time,
                    "amplitudes": _window_mode_amplitudes(
                        context,
                        p,
                        z,
                        int(config["seed_mode"]),
                        int(config["points"]) // 2,
                    ),
                }
            )
            if macro < 2 or macro >= int(config["macros"]) - 2:
                _add_arrays(arrays, f"accepted_macro_{macro:04d}_P", p, context)
                _add_arrays(arrays, f"accepted_macro_{macro:04d}_Z", z, context)
            if macro == int(config["macros"]) - 1:
                _add_arrays(arrays, "final_window_P_nodes", result.window.P, context)
                _add_arrays(arrays, "final_window_Z_coarse", result.window.Z_coarse, context)
                _add_arrays(arrays, "final_window_fine_P", result.window.fine_P, context)
                _add_arrays(arrays, "final_window_fine_Z", result.window.fine_Z, context)
                _add_arrays(arrays, "final_window_fine_G", result.window.fine_G, context)
                _add_arrays(arrays, "final_window_fine_C", result.window.fine_C, context)
                _add_arrays(arrays, "endpoint_totals", result.endpoint.totals, context)
            last_result = result
            _memory_sample(context, f"candidate_macro_{macro}", memory_samples)
    except Exception as error:
        error.partial_candidate = {
            "completed_macros": [window["macro"] for window in windows],
            "windows": list(windows),
            "window_evidence": list(window_evidence),
            "stage_identities": list(identities),
            "mode_history": list(mode_history),
        }
        raise
    _add_arrays(arrays, "accepted_P", p, context)
    _add_arrays(arrays, "accepted_Z", z, context)
    return {
        "context": context,
        "source_P": p_source,
        "source_Z": z_source,
        "P": p,
        "Z": z,
        "windows": windows,
        "last_result": last_result,
        "window_evidence": window_evidence,
        "stage_identities": identities,
        "mode_history": mode_history,
        "fixture": fixture_metadata,
    }


def _run_reference(config, factor, deadline, arrays, memory_samples, label):
    context, state, metadata = _build_periodic(config)
    p0, z0 = wr.pack_state(context, state)
    _memory_sample(context, f"{label}_start", memory_samples)
    if time.monotonic() > deadline:
        raise TimeoutError("wall-clock benchmark budget expired")
    result = _reference_run(
        context,
        p0,
        z0,
        float(config["h"]),
        int(config["M"]),
        int(config["macros"]),
        factor,
    )
    _add_arrays(arrays, f"reference_{label}_P", result["P"], context)
    _add_arrays(arrays, f"reference_{label}_Z", result["Z"], context)
    _memory_sample(context, f"{label}_end", memory_samples)
    return {
        "context": context,
        "P": result["P"],
        "Z": result["Z"],
        "windows": result["windows"],
        "stage_identities": result["stage_identities"],
        "full_stage_calls": result["full_stage_calls"],
        "field_rhs_calls": result["field_rhs_calls"],
        "endpoint_check_calls": result["endpoint_check_calls"],
        "alias_evaluations": result["alias_evaluations"],
        "diagnostics_scope": result["diagnostics_scope"],
        "factor": factor,
        "fixture": metadata,
    }


def _east_interface(config):
    return {
        "status": "unsupported_east",
        "scope": (
            "parent owns the pinned EAST campaign, full transaction integration, "
            "and actual spectrum dispatch"
        ),
        "pinned_input": "g048059.03650",
        "expected_location": "parent-owned <source-root>/inputs/g048059.03650",
        "pinned_sha256": "df8f073bbad202d799b70d0bc2479c9cbcc2d64b729c1019f2575b14de961842",
        "K": 1,
        "Nphi": 5,
        "context_factory": (
            "warp_050_experiments._make_context(kind='east', nodes=config.points, "
            "K=1, device=config.device, equilibrium=pinned_input, memory_limit_bytes)"
        ),
        "background_seed_helpers": [
            "warp_tokamak_cae.background(context,meta,eq)",
            "warp_tokamak_cae.perturb(state,context,meta,mode,amplitude,reference)",
        ],
        "mapping_requirements": [
            "record equilibrium contour and operator mapping",
            "record nonzero equilibrium residual",
            "do not infer h from nominal target period",
            "parent supplies h and spectra",
        ],
        "requested_config": config,
    }


def _atomic_write_new(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing overwrite of existing output: {path}")
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(report), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing overwrite of existing output: {path}"
            ) from error
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _failure_payload(failure):
    return {
        "type": type(failure).__name__,
        "code": str(getattr(failure, "code", "BenchmarkFailure")),
        "message": str(failure),
        "traceback": traceback.format_exc(),
        "adapter_work": _jsonable(getattr(failure, "adapter_work", None)),
        "attempted_sweeps": int(getattr(failure, "attempted_sweeps", 0)),
        "completed_residual_sweeps": int(
            getattr(failure, "completed_residual_sweeps", 0)
        ),
        "source_delta": int(getattr(failure, "adapter_source_delta", 0)),
    }


def run_paths(input_path, output_path):
    """Execute one exact JSON input into one new output JSON file."""

    output = Path(output_path).expanduser().resolve()
    if output.exists():
        return 3
    raw_config, raw_bytes = _read_json(Path(input_path).expanduser().resolve())
    config = validate_config(raw_config)
    source_hashes = _source_hashes()
    if config["case"] == "east":
        report = {
            "schema": "fomhd-wr-benchmark-v1",
            "status": "unsupported_east",
            "source_root": str(SOURCE_ROOT),
            "input_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "config": raw_config,
            "resolved_config": config,
            "source_hashes": source_hashes,
            "production_accepted": False,
            "east_interface": _east_interface(config),
            "limitations": [
                "no EAST context or evolution was run",
                "parent owns pinned-input mapping, h, spectra, and campaign",
            ],
        }
        _atomic_write_new(output, report)
        return 2

    started = time.perf_counter()
    deadline = time.monotonic() + float(config["wall_seconds"])
    arrays = {}
    memory_samples = []
    completed = []
    report = {
        "schema": "fomhd-wr-benchmark-v1",
        "status": "running",
        "source_root": str(SOURCE_ROOT),
        "input_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "config": raw_config,
        "resolved_config": config,
        "source_hashes_before": source_hashes,
        "production_accepted": False,
        "scope": (
            "bounded A1-WR physics adapter and controlled CPU benchmark; "
            "no production accepted transaction"
        ),
    }
    try:
        candidate = None
        if config["coupling"] == "wr":
            candidate = _run_candidate(
                config, deadline, arrays, memory_samples
            )
            completed.append(
                {
                    "kind": "candidate",
                    "macros": int(config["macros"]),
                    "end_time_s": candidate["windows"][-1]["end_time_s"],
                }
            )
        same_h = _run_reference(
            config, 1, deadline, arrays, memory_samples, "same_h"
        )
        completed.append(
            {
                "kind": "same_h_reference",
                "macros": int(config["macros"]),
                "end_time_s": same_h["windows"][-1]["end_time_s"],
            }
        )
        half_h = _run_reference(
            config, 2, deadline, arrays, memory_samples, "half_h"
        )
        completed.append(
            {
                "kind": "half_h_reference",
                "macros": int(config["macros"]),
                "end_time_s": half_h["windows"][-1]["end_time_s"],
            }
        )
        comparisons = {
            "same_h_vs_half_h": _comparison(
                same_h["context"],
                same_h["P"],
                same_h["Z"],
                half_h["P"],
                half_h["Z"],
            )
        }
        if candidate is not None:
            comparisons.update(
                accepted_vs_same_h=_comparison(
                    candidate["context"],
                    candidate["P"],
                    candidate["Z"],
                    same_h["P"],
                    same_h["Z"],
                ),
                accepted_vs_half_h=_comparison(
                    candidate["context"],
                    candidate["P"],
                    candidate["Z"],
                    half_h["P"],
                    half_h["Z"],
                ),
            )
            arrays["final_window_basis_Q"] = np.asarray(
                candidate["last_result"].window.Q
            )
        encoded = _encode_arrays(arrays)
        source_after = _source_hashes()
        if source_after != source_hashes:
            raise Failure("Provenance", "runtime source changed during benchmark")
        report.update(
            status="completed",
            source_hashes=source_hashes,
            source_hashes_after=source_after,
            completed_windows=completed,
            references={
                "same_h": {
                    key: value
                    for key, value in same_h.items()
                    if key not in ("context", "P", "Z")
                },
                "half_h": {
                    key: value
                    for key, value in half_h.items()
                    if key not in ("context", "P", "Z")
                },
            },
            comparisons=comparisons,
            window_evidence=[] if candidate is None else candidate["window_evidence"],
            stage_identities=(
                same_h["stage_identities"] + half_h["stage_identities"]
                if candidate is None
                else candidate["stage_identities"]
                + same_h["stage_identities"]
                + half_h["stage_identities"]
            ),
            mode_history=[] if candidate is None else candidate["mode_history"],
            fixture_metadata={
                "candidate": None if candidate is None else candidate["fixture"],
                "same_h": same_h["fixture"],
                "half_h": half_h["fixture"],
            },
            memory_samples=memory_samples,
            timing={
                "total_elapsed_s": float(time.perf_counter() - started),
                "wall_budget_s": float(config["wall_seconds"]),
                "timer": "time.perf_counter and time.monotonic",
            },
            limitations=[
                "no GPU/CUDA or Slurm work was performed by this invocation",
                "periodic K=0 fixture; EAST K=1 is parent-owned",
                "not a production cumulative ledger or accepted state",
                "full mixed-ledger transaction and restart remain parent integration",
            ],
            **encoded,
        )
    except Exception as error:
        failure_arrays = dict(arrays)
        for name, value in getattr(error, "partial_arrays", {}).items():
            if hasattr(value, "shape"):
                failure_arrays[f"failure_{name}"] = value
        encoded = _encode_arrays(failure_arrays)
        report.update(
            status="failed",
            source_root=str(SOURCE_ROOT),
            completed_windows=completed,
            failure=_failure_payload(error),
            partial_candidate=_jsonable(
                getattr(error, "partial_candidate", None)
            ),
            memory_samples=memory_samples,
            timing={
                "elapsed_s": float(time.perf_counter() - started),
                "wall_budget_s": float(config["wall_seconds"]),
            },
            **encoded,
        )
        _atomic_write_new(output, report)
        return 1
    _atomic_write_new(output, report)
    return 0


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        return run_paths(args.input, args.output)
    except FileExistsError as error:
        print(str(error), file=sys.stderr)
        return 3
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"input error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
