#!/usr/bin/env python3
"""Bounded native-circular real-wall experiment runner (CPU/GPU transaction)."""

from __future__ import annotations

import argparse
import hashlib
import io as io_module
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time as wall_clock
import traceback
import zipfile
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from fomhd import core, io, stage, time as time_module  # noqa: E402
from fomhd.types import (  # noqa: E402
    Context,
    StageTag,
    cumulative_record,
)


GIB = 1 << 30
HOST_MEMORY_CAP = 32 * GIB
DEVICE_MEMORY_CAP = 16 * GIB
MAX_MEMORY_FRACTION = 0.75
MAX_WALL_SECONDS = 1700.0
MAX_SNAPSHOTS = 256
CHECKPOINT_SCHEMA = "clt_boundary_complete_accepted_v1"
NATIVE_LAYOUT_SCHEMA = "polygon_dual_v1:v1"
DEFAULT_N0 = 2.0e19
DEFAULT_B0 = 2.0
DEFAULT_LENGTH = 0.5
DEFAULT_PRESSURE = 1000.0


class RunnerError(RuntimeError):
    """Runner contract error with stable machine-readable code."""

    def __init__(self, message, code="Runner"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ControlPlan:
    K: int
    mode: int
    Nphi: int
    mesh: str
    baseline: str
    nodes: int
    major_radius: float
    minor_radius: float
    amplitude: float
    seed: str
    micro_h: float
    macro_h: float
    micro_count: int
    steps: int
    segment_steps: int
    coupling: str
    device: str
    wall_seconds: float
    snapshot_every: int
    ion_nodes: int
    waveform_rtol: float
    grid_shift_R: float
    grid_shift_Z: float
    horizon_s: float


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=12)
    parser.add_argument("--mesh", choices=("polygon", "staircase"), default="polygon")
    parser.add_argument("--baseline", choices=("vacuum", "gs"), default="vacuum")
    parser.add_argument("--major-radius", type=float, default=3.0)
    parser.add_argument("--minor-radius", type=float, default=0.5)
    parser.add_argument("--K", type=int, choices=(0, 1), default=0)
    parser.add_argument("--mode", type=int, choices=(0, 1), default=0)
    parser.add_argument("--amplitude", type=float, default=1.0e-4)
    parser.add_argument(
        "--seed",
        choices=("smooth", "cylindrical-caw"),
        default="smooth",
    )
    parser.add_argument(
        "--h",
        type=float,
        default=1.0e-8,
        help="explicit microstep; this default is not a CFL claim",
    )
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--s", type=int, choices=(3, 5), default=3)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--segment-steps", type=int)
    parser.add_argument(
        "--coupling", choices=("synchronous", "wr"), default="synchronous"
    )
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=1500.0)
    parser.add_argument("--snapshot-every", type=int, default=64)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--waveform-rtol",
        type=float,
        default=1.0e-12,
        help="explicit new-run WR tolerance; old WR history is unchanged",
    )
    parser.add_argument(
        "--grid-shift-R",
        type=float,
        default=0.0,
        help="generator R shift in fractional h_R units",
    )
    parser.add_argument(
        "--grid-shift-Z",
        type=float,
        default=0.0,
        help="generator Z shift in fractional h_Z units",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="build geometry and estimate operator storage without allocation",
    )
    return parser.parse_args(argv)


def _positive_finite(value, name):
    if type(value) is bool:
        raise RunnerError(f"{name} must be a finite positive number", "Config")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RunnerError(f"{name} must be a finite positive number", "Config")
    return result


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise RunnerError(
            f"{name} must be a positive non-boolean integer", "Config"
        )
    return value


def _nonnegative_finite(value, name):
    if type(value) is bool:
        raise RunnerError(
            f"{name} must be a finite nonnegative number", "Config"
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RunnerError(
            f"{name} must be a finite nonnegative number", "Config"
        )
    return result


def validate_controls(args):
    nodes = _positive_integer(args.nodes, "nodes")
    if nodes < 4:
        raise RunnerError("nodes must be at least 4", "Config")
    major = _positive_finite(args.major_radius, "major radius")
    minor = _positive_finite(args.minor_radius, "minor radius")
    if minor >= major:
        raise RunnerError(
            "minor radius must be positive and smaller than major radius",
            "Config",
        )
    if args.K not in (0, 1):
        raise RunnerError("K must be 0 or 1", "Config")
    if args.mode not in (0, 1):
        raise RunnerError("mode must be 0 or 1", "Config")
    if args.mode > args.K:
        raise RunnerError(
            f"K={args.K} does not retain requested mode {args.mode}",
            "Config",
        )
    amplitude = _nonnegative_finite(args.amplitude, "amplitude")
    if amplitude > 0.01:
        raise RunnerError(
            "amplitude must be a nonnegative fractional value no greater than 0.01",
            "Config",
        )
    if getattr(args, "seed", None) not in ("smooth", "cylindrical-caw"):
        raise RunnerError(
            "seed must be smooth or cylindrical-caw", "Config"
        )
    if args.seed == "cylindrical-caw" and args.mode != 0:
        raise RunnerError(
            "cylindrical-caw is a mode 0 standing-wave initial control",
            "Config",
        )
    micro_h = _positive_finite(args.h, "h")
    micro_count = _positive_integer(args.M, "M")
    if args.coupling == "synchronous" and micro_count != 1:
        raise RunnerError(
            "synchronous coupling requires micro_steps M=1", "Config"
        )
    steps = _positive_integer(args.steps, "steps")
    segment = (
        steps
        if args.segment_steps is None
        else _positive_integer(args.segment_steps, "segment-steps")
    )
    if segment > steps:
        raise RunnerError(
            "segment-steps must not exceed global steps", "Config"
        )
    if args.s not in (3, 5):
        raise RunnerError("s must be 3 or 5", "Config")
    waveform_rtol = _positive_finite(args.waveform_rtol, "waveform-rtol")
    grid_shift_R = float(args.grid_shift_R)
    grid_shift_Z = float(args.grid_shift_Z)
    for name, value in (
        ("grid-shift-R", grid_shift_R),
        ("grid-shift-Z", grid_shift_Z),
    ):
        if not math.isfinite(value) or not -0.5 <= value <= 0.5:
            raise RunnerError(
                f"{name} must be finite in [-0.5,0.5] generator spacings",
                "Config",
            )
    wall_seconds = _positive_finite(args.wall_seconds, "wall seconds")
    if wall_seconds > MAX_WALL_SECONDS:
        raise RunnerError(
            "wall-seconds must be no greater than 1700 seconds", "Config"
        )
    snapshot_every = _positive_integer(
        args.snapshot_every, "snapshot-every"
    )
    snapshot_upper_bound = (
        (steps + snapshot_every - 1) // snapshot_every + 2
    )
    if snapshot_upper_bound > MAX_SNAPSHOTS:
        raise RunnerError(
            f"snapshot schedule could create {snapshot_upper_bound} states, "
            f"exceeding the bounded count {MAX_SNAPSHOTS}",
            "Config",
        )
    if args.coupling not in ("synchronous", "wr"):
        raise RunnerError("unknown coupling", "Config")
    if args.device == "cuda:0" and not os.environ.get("SLURM_JOB_ID"):
        raise RunnerError(
            "cuda:0 requires an owned SLURM_JOB_ID allocation", "Allocation"
        )
    macro_h = micro_h * (micro_count if args.coupling == "wr" else 1)
    return ControlPlan(
        K=args.K,
        mode=args.mode,
        Nphi=4 * args.K + 1,
        mesh=args.mesh,
        baseline=args.baseline,
        nodes=nodes,
        major_radius=major,
        minor_radius=minor,
        amplitude=amplitude,
        seed=args.seed,
        micro_h=micro_h,
        macro_h=macro_h,
        micro_count=micro_count,
        steps=steps,
        segment_steps=segment,
        coupling=args.coupling,
        device=args.device,
        wall_seconds=wall_seconds,
        snapshot_every=snapshot_every,
        ion_nodes=args.s,
        waveform_rtol=waveform_rtol,
        grid_shift_R=grid_shift_R,
        grid_shift_Z=grid_shift_Z,
        horizon_s=steps * macro_h,
    )


def validate_output(path):
    path = Path(path)
    if path.is_symlink():
        raise RunnerError("output must not be a symlink", "Output")
    if path.exists():
        raise RunnerError(
            f"output must be a new directory (NEWDIR), got {path}",
            "Output",
        )
    if not path.name:
        raise RunnerError("output directory name is empty", "Output")
    return path


def _host_meminfo_available():
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def _host_cgroup_available():
    maximum = Path("/sys/fs/cgroup/memory.max")
    current = Path("/sys/fs/cgroup/memory.current")
    if not maximum.is_file() or not current.is_file():
        return None
    maximum_text = maximum.read_text(encoding="ascii").strip()
    if maximum_text == "max":
        return None
    return max(int(maximum_text) - int(current.read_text(encoding="ascii")), 0)


def host_available_memory():
    values = [
        value
        for value in (_host_meminfo_available(), _host_cgroup_available())
        if value is not None
    ]
    if not values:
        raise RunnerError("host available-memory probe is unavailable", "Memory")
    return min(values)


def device_available_memory(device):
    if device == "cpu":
        return None
    try:
        import cupy as cp
    except Exception as error:
        raise RunnerError(
            f"CUDA memory probe unavailable: {error}", "Memory"
        ) from error
    free, _ = cp.cuda.Device(int(device.split(":", 1)[1])).mem_info
    return int(free)


def check_memory_caps(
    *,
    host_available_bytes,
    device_available_bytes,
    host_cap_bytes=HOST_MEMORY_CAP,
    device_cap_bytes=DEVICE_MEMORY_CAP,
):
    result = {
        "host_available_bytes": int(host_available_bytes),
        "host_cap_bytes": int(host_cap_bytes),
        "device_available_bytes": (
            None
            if device_available_bytes is None
            else int(device_available_bytes)
        ),
        "device_cap_bytes": int(device_cap_bytes),
        "maximum_fraction": MAX_MEMORY_FRACTION,
    }
    host_fraction = host_cap_bytes / host_available_bytes
    result["host_fraction"] = host_fraction
    if host_fraction > MAX_MEMORY_FRACTION:
        raise RunnerError(
            "host cap exceeds 75% of memory available inside the allocation",
            "Memory",
        )
    if device_available_bytes is not None:
        device_fraction = device_cap_bytes / device_available_bytes
        result["device_fraction"] = device_fraction
        if device_fraction > MAX_MEMORY_FRACTION:
            raise RunnerError(
                "device cap exceeds 75% of memory available inside the allocation",
                "Memory",
            )
    return result


def build_parameters(args, plan):
    from tools.warp_tokamak_cae import physical_parameters

    parameters = physical_parameters(
        plan.K,
        DEFAULT_N0,
        DEFAULT_B0,
        DEFAULT_LENGTH,
        DEFAULT_PRESSURE,
    )
    parameters.K = plan.K
    parameters.Nphi = plan.Nphi
    parameters.memory_limit = HOST_MEMORY_CAP
    parameters.device_memory_limit = DEVICE_MEMORY_CAP
    parameters.electron_inertia = 0
    parameters.integrator = "rk4"
    parameters.dt_guard = "error"
    parameters.start_time = 0.0
    parameters.dt = plan.macro_h
    parameters.nsteps = plan.steps
    parameters.coupling = (
        "multirate_wr" if plan.coupling == "wr" else "synchronous"
    )
    parameters.micro_steps = plan.micro_count
    parameters.waveform_rtol = plan.waveform_rtol
    if parameters.coupling == "multirate_wr":
        parameters.ion_nodes = plan.ion_nodes
        parameters.wr_initial = "constant"
        parameters.wr_relaxation = 1.0
    else:
        parameters.ion_nodes = None
        parameters.wr_initial = None
        parameters.wr_relaxation = None
    time_module.validate_time_controls(parameters)
    return parameters


def _circle_polygon(major, minor, nodes):
    count = max(32, 4 * int(nodes))
    angle = np.arange(count, dtype=np.float64) * (2.0 * np.pi / count)
    return np.column_stack(
        (
            major + minor * np.cos(angle),
            minor * np.sin(angle),
        )
    )


def _generator_grid(plan):
    lo_R = plan.major_radius - plan.minor_radius
    hi_R = plan.major_radius + plan.minor_radius
    lo_Z = -plan.minor_radius
    hi_Z = plan.minor_radius
    h_R = (hi_R - lo_R) / plan.nodes
    h_Z = (hi_Z - lo_Z) / plan.nodes
    R = lo_R + (np.arange(plan.nodes, dtype=np.float64) + 0.5) * h_R
    Z = lo_Z + (np.arange(plan.nodes, dtype=np.float64) + 0.5) * h_Z
    R += plan.grid_shift_R * h_R
    Z += plan.grid_shift_Z * h_Z
    return R, Z


def _import_polygon_backend():
    try:
        from fomhd import polygon_geometry, polygon_operators
    except ImportError as error:
        raise RunnerError(
            "polygon geometry/operator integration is not available: "
            f"{error}",
            "IntegrationUnavailable",
        ) from error
    return polygon_geometry, polygon_operators


def _legacy_geometry_fingerprint(geometry):
    digest = hashlib.sha256()
    digest.update(b"staircase_v1")
    for label, table in (
        ("cells", geometry.cells),
        ("radial_faces", geometry.r_faces),
        ("vertical_faces", geometry.z_faces),
        ("vertices", geometry.vertices),
    ):
        keys = np.asarray(getattr(table, "keys", ()), dtype=np.int64)
        digest.update(label.encode("ascii"))
        digest.update(repr(keys.shape).encode("ascii"))
        digest.update(keys.tobytes())
        digest.update(np.asarray(table.R, dtype=np.float64).tobytes())
        digest.update(np.asarray(table.Z, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _geometry_kind(geometry):
    return getattr(geometry, "kind", "staircase_v1")


def _memory_setup(plan):
    host_available = host_available_memory()
    device_available = device_available_memory(plan.device)
    memory = check_memory_caps(
        host_available_bytes=host_available,
        device_available_bytes=device_available,
    )
    operator_cap = (
        DEVICE_MEMORY_CAP if plan.device != "cpu" else HOST_MEMORY_CAP
    )
    return memory, operator_cap


def _build_geometry(args, plan):
    polygon = _circle_polygon(
        plan.major_radius, plan.minor_radius, plan.nodes
    )
    R, Z = _generator_grid(plan)
    geometry_options = {"memory_cap_bytes": HOST_MEMORY_CAP}
    if args.mesh == "polygon":
        polygon_geometry, _ = _import_polygon_backend()
        from fomhd.geometry import Point2

        geometry = polygon_geometry.build_polygon_geometry(
            R,
            Z,
            [Point2(float(radius), float(axial)) for radius, axial in polygon],
            geometry_options,
        )
        if _geometry_kind(geometry) != "polygon_dual_v1":
            raise RunnerError("geometry returned a non-native layout", "Layout")
        from fomhd import polygon_io

        geometry_fingerprint = polygon_io.geometry_fingerprint(geometry)
    else:
        from fomhd.geometry import Point2, build_geometry

        geometry = build_geometry(
            R,
            Z,
            [Point2(float(radius), float(axial)) for radius, axial in polygon],
            geometry_options,
        )
        geometry_fingerprint = _legacy_geometry_fingerprint(geometry)
    return geometry, geometry_fingerprint, polygon, R, Z


def _estimated_operator_storage(geometry, plan, operator_cap, rank_rtol):
    if _geometry_kind(geometry) == "polygon_dual_v1":
        _, polygon_operators = _import_polygon_backend()
        settings = polygon_operators.PolygonOperatorOptions(
            rank_rtol=rank_rtol,
            memory_cap_bytes=operator_cap,
        )
        estimate = polygon_operators._estimate_bytes(
            geometry, 2 * plan.K + 1, settings
        )
        source = "fomhd.polygon_operators._estimate_bytes"
    else:
        from fomhd.operators import _estimate_operator_bytes

        estimate = _estimate_operator_bytes(
            geometry, plan.K, include_finite_mass=False
        )
        source = "fomhd.operators._estimate_operator_bytes"
    return int(estimate), source


def _geometry_summary(geometry):
    diagonal = getattr(geometry, "diagnostics", None)
    metric_names = (
        "polygon_area",
        "polygon_M0",
        "coverage_area_relative",
        "coverage_M0_relative",
        "min_cell_area",
        "min_M0",
        "min_face_length",
        "min_face_distance",
        "min_vertex_area",
        "min_vertex_volume",
        "input_generators",
        "retained_generators",
        "removed_generators",
    )
    metrics = {}
    if diagonal is not None:
        for name in metric_names:
            value = getattr(diagonal, name, None)
            if value is not None:
                metrics[name] = float(value)
    faces = getattr(geometry, "faces", getattr(geometry, "r_faces"))
    return {
        "kind": _geometry_kind(geometry),
        "Nc": int(len(geometry.cells.R)),
        "Nf": int(len(faces.R)),
        "Nv": int(len(geometry.vertices.R)),
        "metrics": metrics,
    }


def _preflight_only(args, plan, output, started):
    report = {
        "status": "preflight_initializing",
        "scientific_status": "pending_parent_validation",
        "created_at_unix": wall_clock.time(),
        "parameters": asdict(plan),
        "scheme_parameters": scheme_parameters(plan),
        "output": str(output),
    }
    try:
        memory, operator_cap = _memory_setup(plan)
        parameters = build_parameters(args, plan)
        geometry, fingerprint, polygon, R, Z = _build_geometry(args, plan)
        geometry_artifact = write_geometry_artifact(
            output,
            SimpleNamespace(geometry=geometry, op=None, compute=None),
            {"geometry_fingerprint": fingerprint},
        )
        estimate, estimate_source = _estimated_operator_storage(
            geometry,
            plan,
            operator_cap,
            parameters.rank_rtol,
        )
        if estimate > operator_cap:
            raise RunnerError(
                "operator_matrices memory budget exceeded before allocation: "
                f"estimated={estimate} limit={operator_cap}",
                "Memory",
            )
        report.update(
            {
                "status": "preflight_complete",
                "geometry": {
                    **_geometry_summary(geometry),
                    "fingerprint": fingerprint,
                    **geometry_artifact,
                },
                "preflight": {
                    "estimated_operator_bytes": estimate,
                    "estimated_operator_gib": estimate / GIB,
                    "operator_cap_bytes": operator_cap,
                    "operator_headroom_bytes": operator_cap - estimate,
                    "estimate_source": estimate_source,
                    "operators_allocated": False,
                    "state_allocated": False,
                    "memory_probe": memory,
                },
                "input_fingerprint": input_identity(
                    plan,
                    geometry_fingerprint=fingerprint,
                    baseline=plan.baseline,
                    mesh=plan.mesh,
                    nodes=plan.nodes,
                ),
                "provenance": {
                    "source_fingerprint": _source_fingerprint(),
                    "source_manifest": source_fingerprint_manifest(),
                    "runner": str(Path(__file__).resolve()),
                    "device": plan.device,
                    },
                    "geometry_input": {
                    "polygon": np.asarray(polygon).tolist(),
                    "generator_R": np.asarray(R).tolist(),
                    "generator_Z": np.asarray(Z).tolist(),
                    "grid_shift_R": plan.grid_shift_R,
                    "grid_shift_Z": plan.grid_shift_Z,
                },
            }
        )
        report["elapsed_s"] = wall_clock.monotonic() - started
        write_strict_json(output / "report.json", report)
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = {
            "code": getattr(error, "code", type(error).__name__),
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        report["elapsed_s"] = wall_clock.monotonic() - started
        write_strict_json(output / "report.json", report)
        return 2


def make_context(args):
    plan = validate_controls(args)
    memory, operator_cap = _memory_setup(plan)
    parameters = build_parameters(args, plan)
    from fomhd.backend import Backend

    backend = Backend(plan.device)
    geometry, geometry_fingerprint, polygon, R, Z = _build_geometry(args, plan)
    operator_options = {
        "memory_cap_bytes": operator_cap,
        "rank_rtol": parameters.rank_rtol,
    }
    if _geometry_kind(geometry) == "polygon_dual_v1":
        _, polygon_operators = _import_polygon_backend()
        operators = polygon_operators.build_polygon_operators(
            geometry, plan.K, operator_options, backend
        )
    else:
        from fomhd.operators import build_operators

        operators = build_operators(
            geometry,
            plan.K,
            {**operator_options, "include_finite_mass": False},
            backend,
        )
    if operators.finite_mass_available:
        raise RunnerError(
            "this runner requires massless-only operator storage", "Model"
        )
    if not hasattr(operators, "geometry"):
        operators.geometry = geometry
    operator_fingerprint = io._operator_fingerprint(operators, backend)
    context = Context(
        p=parameters,
        geometry=geometry,
        op=operators,
        compute=backend,
        N=parameters.Nphi,
        next_generation=2,
    )
    context.geometry_hash = geometry_fingerprint
    context.operator_hash = operator_fingerprint
    context.clt_plan = plan
    metadata = {
        "layout_kind": _geometry_kind(geometry),
        "layout_schema": (
            NATIVE_LAYOUT_SCHEMA
            if _geometry_kind(geometry) == "polygon_dual_v1"
            else io.LAYOUT_VERSION
        ),
        "geometry_fingerprint": geometry_fingerprint,
        "operator_fingerprint": operator_fingerprint,
        "mesh": args.mesh,
        "baseline": args.baseline,
        "nodes": plan.nodes,
        "Nphi": plan.Nphi,
        "K": plan.K,
        "mode": plan.mode,
        "seed": plan.seed,
        "micro_h": plan.micro_h,
        "macro_h": plan.macro_h,
        "M": plan.micro_count,
        "s": plan.ion_nodes,
        "waveform_rtol": plan.waveform_rtol,
        "grid_shift_R": plan.grid_shift_R,
        "grid_shift_Z": plan.grid_shift_Z,
        "steps": plan.steps,
        "segment_steps": plan.segment_steps,
        "coupling": plan.coupling,
        "horizon_s": plan.horizon_s,
        "memory": memory,
        "input_polygon": polygon,
        "generator_R": R,
        "generator_Z": Z,
        "F0_T_m": DEFAULT_B0 * plan.major_radius,
    }
    metadata["scheme_parameters"] = scheme_parameters(plan)
    metadata["input_fingerprint"] = _input_fingerprint(context, metadata)
    return context, metadata


def _copy_state(state, context):
    result = core.allocate_state(context)
    result.B[...] = state.B
    for species in range(2):
        for name in ("n", "mom", "perp", "parallel"):
            target = getattr(result.fluid[species], name)
            source = getattr(state.fluid[species], name)
            if target.size:
                target[...] = source
    result.step = int(state.step)
    result.generation = int(state.generation)
    result.time = float(state.time)
    return result


def _initial_complex(context, value):
    """Export one host initialization field to the retained backend dtype."""
    return context.compute.array(value, np.complex128)


def _fill_fluid_control(state, context, density, p_i, p_e_perp, p_e_parallel):
    nc = context.op.Nc
    modes = 2 * context.p.K + 1
    K = context.p.K
    density = _initial_complex(context, density)
    p_i = _initial_complex(context, p_i)
    p_e_perp = _initial_complex(context, p_e_perp)
    p_e_parallel = _initial_complex(context, p_e_parallel)
    for species in range(2):
        fluid = state.fluid[species]
        fluid.n.reshape(modes, nc)[K] = density
        fluid.perp.reshape(modes, nc)[K] = p_i if species == 0 else p_e_perp
        if species == 1:
            fluid.parallel.reshape(modes, nc)[K] = p_e_parallel
    return state


def _native_b_blocks(context):
    geometry = context.geometry
    modes = 2 * context.p.K + 1
    if _geometry_kind(geometry) == "polygon_dual_v1":
        nv = len(geometry.vertices.R)
        nf = len(geometry.faces.R)
        return "polygon", modes, nv, nf, 0, nv
    nv = len(geometry.vertices.R)
    nz = len(geometry.z_faces.R)
    return "staircase", modes, nv, 0, nz, nz + nv


def _vacuum_state(context):
    plan = context.clt_plan
    state = core.allocate_state(context)
    pressure_i = 0.6 * DEFAULT_PRESSURE
    pressure_e = 0.4 * DEFAULT_PRESSURE
    density = np.full(context.op.Nc, DEFAULT_N0, dtype=np.float64)
    _fill_fluid_control(
        state,
        context,
        density,
        np.full(context.op.Nc, pressure_i),
        np.full(context.op.Nc, pressure_e),
        np.full(context.op.Nc, pressure_e),
    )
    kind, modes, nv, nf, first, second = _native_b_blocks(context)
    if kind == "polygon":
        B_phi = _initial_complex(
            context, DEFAULT_B0 * plan.major_radius / context.geometry.Rbar_v
        )
        B = state.B.reshape(modes, nv + nf)
        B[context.p.K] = 0.0
        B[context.p.K, :nv] = B_phi
    else:
        B_phi = _initial_complex(
            context,
            DEFAULT_B0
            * plan.major_radius
            / np.asarray(context.geometry.Rbar_v),
        )
        B = state.B.reshape(modes, context.op.Nb)
        B[context.p.K] = 0.0
        B[context.p.K, first:second] = B_phi
    raw_radius = np.asarray(context.geometry.vertices.R, dtype=np.float64)
    mapped_radius = np.asarray(context.geometry.Rbar_v, dtype=np.float64)
    raw_B_phi = DEFAULT_B0 * plan.major_radius / raw_radius
    mapped_B_phi = DEFAULT_B0 * plan.major_radius / mapped_radius
    state.generation = 1
    state.time = 0.0
    state.step = 0
    return state, {
        "baseline": "vacuum",
        "equilibrium_parameters": {
            "R0_m": float(plan.major_radius),
            "a_m": float(plan.minor_radius),
            "F0_T_m": float(DEFAULT_B0 * plan.major_radius),
            "n0_m^-3": float(DEFAULT_N0),
            "p_edge_Pa": float(DEFAULT_PRESSURE),
            "pprime_Pa_per_Wb_rad": 0.0,
            "alpha_ion_fraction": 0.6,
            "degree": 0,
        },
        "raw_B_phi": raw_B_phi,
        "mapped_B_phi": mapped_B_phi,
        "raw_minus_mapped_B_phi": raw_B_phi - mapped_B_phi,
        "B_poloidal_raw": np.zeros(
            nf
            if kind == "polygon"
            else len(context.geometry.z_faces.R)
            + len(context.geometry.r_faces.R),
            dtype=np.float64,
        ),
    }


def _gs_state(context):
    plan = context.clt_plan
    try:
        from tools.circular_equilibrium import build_gs_equilibrium, sample_gs
    except ImportError as error:
        raise RunnerError(
            "circular equilibrium integration is not available: "
            f"{error}",
            "IntegrationUnavailable",
        ) from error
    model = build_gs_equilibrium(
        R0=plan.major_radius,
        a=plan.minor_radius,
        F0=DEFAULT_B0 * plan.major_radius,
        n0=DEFAULT_N0,
        p_edge=DEFAULT_PRESSURE,
        pprime=1.0e4,
        alpha=0.6,
        degree=10,
    )
    state = core.allocate_state(context)
    cells = sample_gs(
        model,
        np.asarray(context.geometry.cells.R),
        np.asarray(context.geometry.cells.Z),
    )
    vertex_R = np.asarray(context.geometry.vertices.R)
    vertex_Z = np.asarray(context.geometry.vertices.Z)
    vertex_inside = (
        (vertex_R - plan.major_radius) ** 2 + vertex_Z**2
        <= plan.minor_radius**2 * (1.0 + 32.0 * np.finfo(np.float64).eps)
    )
    vertex_samples = sample_gs(
        model,
        vertex_R[vertex_inside],
        vertex_Z[vertex_inside],
    )
    _fill_fluid_control(
        state,
        context,
        cells["n"],
        cells["p_ion"],
        cells["p_e_perp"],
        cells["p_e_parallel"],
    )
    kind, modes, nv, nf, first, second = _native_b_blocks(context)
    if kind != "polygon":
        # The old-grid GS control is C @ Aphi + the current-free harmonic.
        # Continuum derivatives are sampled only at active cell centers.
        B = state.B.reshape(modes, context.op.Nb)
        B[context.p.K] = 0.0
        nr = len(context.geometry.r_faces.R)
        aphi = np.zeros(context.op.Ne, dtype=np.complex128)
        aphi[nr : nr + context.op.Nc] = (
            np.asarray(cells["psi"], dtype=np.complex128)
            / np.asarray(context.geometry.cells.R)
        )
        old_bpol = np.asarray(
            context.compute.to_host(
                context.compute.mv(
                    context.op.C[context.p.K],
                    context.compute.array(aphi, np.complex128),
                )
            ),
            dtype=np.complex128,
        )
        topology = np.asarray(
            context.compute.to_host(context.op.topology_flux_row),
            dtype=np.complex128,
        ).reshape(-1)
        magnetic_weight = np.asarray(
            context.compute.to_host(context.op.MB), dtype=np.float64
        )
        harmonic = (
            DEFAULT_B0
            * plan.major_radius
            * topology.conj()
            / magnetic_weight
        )
        old_B = old_bpol + harmonic
        B[context.p.K] = _initial_complex(context, old_B)
        face_R = np.asarray(context.geometry.cells.R)
        faces = {
            "B_R": cells["B_R"],
            "B_Z": cells["B_Z"],
            "B_phi": cells["B_phi"],
        }
        discrete_tangent = np.zeros(nf)
        raw_tangent = np.zeros(nf)
        raw_normal = np.zeros(nf)
        old_mapping = "C@Aphi + F0*conj(topology_flux_row)/MB"
        project_B_used = False
    else:
        face_R = np.asarray(context.geometry.faces.R)
        face_Z = np.asarray(context.geometry.faces.Z)
        faces = sample_gs(model, face_R, face_Z)
        normals = np.asarray(context.geometry.face_normals)
        tangents = np.asarray(context.geometry.face_tangents)
        poloidal = np.column_stack((faces["B_R"], faces["B_Z"]))
        raw_tangent = np.sum(poloidal * tangents, axis=1)
        raw_normal = np.sum(poloidal * normals, axis=1)
        B = state.B.reshape(modes, nv + nf)
        B[context.p.K] = 0.0
        B_phi = _initial_complex(
            context,
            DEFAULT_B0 * plan.major_radius / context.geometry.Rbar_v,
        )
        B[context.p.K, :nv] = B_phi
        normal_gradient = np.asarray(
            context.compute.to_host(
                context.op.G[context.p.K][:nf, :]
            ),
            dtype=np.complex128,
        )
        discrete_tangent = np.real(
            normal_gradient @ np.asarray(cells["psi"], dtype=np.complex128)
        ) / face_R
        B[context.p.K, nv:] = _initial_complex(
            context, discrete_tangent
        )
        old_mapping = None
        project_B_used = False
    state.generation = 1
    state.time = 0.0
    state.step = 0
    return state, {
        "baseline": "gs",
        "equilibrium_parameters": {
            "R0_m": model.R0,
            "a_m": model.a,
            "F0_T_m": model.F0,
            "n0_m^-3": model.n0,
            "p_edge_Pa": model.p_edge,
            "pprime_Pa_per_Wb_rad": model.pprime,
            "alpha_ion_fraction": model.alpha,
            "degree": model.degree,
        },
        "gs_degree": model.degree,
        "gs_solve_residual": model.solve_residual,
        "vertex_polygon_mask": vertex_inside,
        "raw_B_phi_valid": vertex_samples["B_phi"],
        "mapped_B_phi_valid": (
            DEFAULT_B0 * plan.major_radius
            / np.asarray(context.geometry.Rbar_v)[vertex_inside]
        ),
        "raw_minus_mapped_B_phi_valid": vertex_samples["B_phi"]
        - (
            DEFAULT_B0
            * plan.major_radius
            / np.asarray(context.geometry.Rbar_v)[vertex_inside]
        ),
        "B_poloidal_raw": np.column_stack((cells["B_R"], cells["B_Z"])),
        "B_t_raw_continuous": raw_tangent,
        "B_n_raw_continuous": raw_normal,
        "B_t_discrete": discrete_tangent,
        "old_staircase_mapping": old_mapping,
        "B_n_interpretation": (
            "discrete tangential control from curl of cell flux; "
            "continuous normal samples are diagnostics, not subtracted or "
            "forced as a projected state"
        ),
        "project_B_used": project_B_used,
    }


def initial_state(context, metadata):
    if metadata["baseline"] == "vacuum":
        return _vacuum_state(context)
    return _gs_state(context)


def _wall_distance(geometry, count):
    value = getattr(geometry, "cell_wall_distance", None)
    if callable(value):
        value = value()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != count or not np.isfinite(array).all() or np.any(array <= 0):
        raise RunnerError(
            "geometry true cell-wall distance is unavailable or nonpositive",
            "Geometry",
        )
    return array


def _project_full_momentum(momentum, context, mass, density):
    operator = np.asarray(
        context.compute.to_host(context.op.T_full), dtype=np.complex128
    )
    M0 = np.asarray(
        context.compute.to_host(context.op.M0), dtype=np.float64
    )
    weight = np.tile(M0 / (mass * density), 3)
    weighted = operator / weight[None, :]
    constraint = operator @ weighted.T
    target = operator @ momentum
    multiplier = np.linalg.lstsq(
        constraint, target, rcond=context.p.rank_rtol
    )[0]
    projected = momentum - weighted.T @ multiplier
    residual = float(np.max(np.abs(operator @ projected), initial=0.0))
    return projected, residual


def _cylindrical_caw_velocity(context):
    try:
        from tools.cylindrical_caw_control import seed_velocity
    except ImportError as error:
        raise RunnerError(
            "cylindrical-caw seed requires tools.cylindrical_caw_control: "
            f"{error}",
            "IntegrationUnavailable",
        ) from error
    if not callable(seed_velocity):
        raise RunnerError(
            "tools.cylindrical_caw_control.seed_velocity is not callable",
            "IntegrationUnavailable",
        )
    plan = context.clt_plan
    R = np.asarray(context.geometry.cells.R, dtype=np.float64)
    Z = np.asarray(context.geometry.cells.Z, dtype=np.float64)
    values = {
        "R0": plan.major_radius,
        "a": plan.minor_radius,
        "amplitude": plan.amplitude,
        "B0": DEFAULT_B0,
        "B_0": DEFAULT_B0,
        "n0": DEFAULT_N0,
        "n_0": DEFAULT_N0,
        "mi": context.p.species[0].mass,
        "pi0": 0.6 * DEFAULT_PRESSURE,
        "pe0": 0.4 * DEFAULT_PRESSURE,
        "p_i0": 0.6 * DEFAULT_PRESSURE,
        "p_e0": 0.4 * DEFAULT_PRESSURE,
        "gamma": context.p.species[0].gamma,
        "mu0": context.p.mu0,
        "e": abs(context.p.species[1].charge),
        "time": 0.0,
    }
    positional = [R, Z]
    keywords = {}
    signature = inspect.signature(seed_velocity)
    for name, parameter in signature.parameters.items():
        if name in ("R", "Z"):
            continue
        if name in values:
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                positional.append(values[name])
            else:
                keywords[name] = values[name]
            continue
        if (
            parameter.default is inspect.Parameter.empty
            and parameter.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ):
            raise RunnerError(
                "cylindrical_caw_control.seed_velocity has unsupported "
                f"required parameter {name!r}",
                "IntegrationUnavailable",
            )
    try:
        velocity = seed_velocity(*positional, **keywords)
    except Exception as error:
        raise RunnerError(
            f"cylindrical-caw seed helper failed: {error}",
            "IntegrationUnavailable",
        ) from error
    raw = np.asarray(velocity)
    if raw.dtype.kind == "c":
        if np.max(np.abs(raw.imag), initial=0.0) > 512.0 * np.finfo(np.float64).eps * max(
            np.max(np.abs(raw.real), initial=0.0), 1.0e-300
        ):
            raise RunnerError(
                "cylindrical-caw seed velocity returned a complex field",
                "Seed",
            )
        raw = raw.real
    raw = np.asarray(raw, dtype=np.float64)
    expected = (context.op.Nc, 3)
    if raw.shape != expected:
        raise RunnerError(
            f"cylindrical-caw seed shape {raw.shape} != {expected}",
            "Seed",
        )
    if not np.isfinite(raw).all():
        raise RunnerError(
            "cylindrical-caw seed velocity contains nonfinite values",
            "Seed",
        )
    return raw, {
        "helper": "tools.cylindrical_caw_control.seed_velocity",
        "sample_count": int(context.op.Nc),
    }


def apply_default_perturbation(state, context):
    plan = context.clt_plan
    seeded = _copy_state(state, context)
    geometry = context.geometry
    R = np.asarray(geometry.cells.R, dtype=np.float64)
    Z = np.asarray(geometry.cells.Z, dtype=np.float64)
    nc = context.op.Nc
    modes = 2 * context.p.K + 1
    ion = seeded.fluid[0]
    density = np.asarray(
        context.compute.to_host(ion.n), dtype=np.complex128
    ).reshape(modes, nc)[context.p.K].real
    momentum = np.zeros((modes, 3, nc), dtype=np.complex128)
    if plan.seed == "smooth":
        radius = np.hypot(R - plan.major_radius, Z)
        safe = np.maximum(radius, np.finfo(np.float64).eps)
        direction_R = (R - plan.major_radius) / safe
        direction_Z = Z / safe
        wall_distance = _wall_distance(geometry, nc)
        profile = np.clip(
            1.0 - (radius / wall_distance) ** 2, 0.0, 1.0
        ) ** 2
        speed = plan.amplitude * context.p.ref.velocity * profile
        velocity_cells = np.zeros((nc, 3), dtype=np.float64)
        velocity_cells[:, 0] = speed * direction_R
        velocity_cells[:, 2] = speed * direction_Z
        stored_modes = (context.p.K,) if plan.mode == 0 else (
            context.p.K - 1,
            context.p.K + 1,
        )
        coefficient = 1.0 if plan.mode == 0 else 0.5
        seed_evidence = {
            "seed": "smooth",
            "profile": "axisymmetric radial disk (1-(r/wall)^2)^2",
        }
    else:
        if plan.mode != 0:
            raise RunnerError(
                "cylindrical-caw is a mode 0 initial control", "Config"
            )
        velocity_cells, seed_evidence = _cylindrical_caw_velocity(context)
        stored_modes = (context.p.K,)
        coefficient = 1.0
        seed_evidence.update(
            {
                "seed": "cylindrical-caw",
                "standing_caw_t0": True,
                "scalar_pressure_perturbation": 0.0,
                "magnetic_perturbation": 0.0,
                "continuous_limit": "STRAIGHT-cylinder reference only",
            }
        )
    for stored in stored_modes:
        momentum[stored] = (
            coefficient
            * context.p.species[0].mass
            * density[None, :]
            * velocity_cells.T
        )
    seed_evidence["vA_m_s"] = float(context.p.ref.velocity)
    raw = momentum.reshape(-1)
    projected = np.empty_like(raw)
    residuals = {}
    for stored in range(modes):
        start = stored * 3 * nc
        stop = (stored + 1) * 3 * nc
        projected[start:stop], residuals[str(stored - context.p.K)] = (
            _project_full_momentum(
                raw[start:stop],
                context,
                context.p.species[0].mass,
                density,
            )
        )
    ion.mom[...] = context.compute.array(projected, np.complex128)
    return seeded, {
        **seed_evidence,
        "physical_mode": plan.mode,
        "amplitude": plan.amplitude,
        "raw_mom_i": raw,
        "projected_mom_i": projected,
        "projected_minus_raw": projected - raw,
        "wall_trace_residual_by_mode": residuals,
    }


def _host(context, value):
    return np.asarray(context.compute.to_host(value))


def host_state_for_codec(state, context):
    """Explicitly export a complete state at declared I/O boundaries."""
    from fomhd.types import FluidState, State

    fluid = []
    for source in state.fluid:
        fluid.append(
            FluidState(
                n=np.asarray(_host(context, source.n), dtype=np.complex128),
                mom=np.asarray(
                    _host(context, source.mom), dtype=np.complex128
                ),
                perp=np.asarray(
                    _host(context, source.perp), dtype=np.complex128
                ),
                parallel=np.asarray(
                    _host(context, source.parallel), dtype=np.complex128
                ),
            )
        )
    return State(
        fluid=tuple(fluid),
        B=np.asarray(_host(context, state.B), dtype=np.complex128),
        step=int(state.step),
        generation=int(state.generation),
        time=float(state.time),
    )


def _physical_samples(value, shape, context, name):
    array = np.asarray(_host(context, value), dtype=np.complex128)
    if array.size != int(np.prod(shape)):
        raise RunnerError(
            f"snapshot sample {name} has size {array.size}, expected {int(np.prod(shape))}",
            "Snapshot",
        )
    array = array.reshape(shape)
    if not np.isfinite(array).all():
        raise RunnerError(
            f"snapshot sample {name} contains nonfinite values", "Snapshot"
        )
    return array.copy()


def _initial_force_samples(context, frame):
    shape = (int(context.N), 3, int(context.op.Nc))
    sources = {
        "frame_ion_pressure_force_samples": frame.fluid[0].pressure_force,
        "frame_ion_magnetic_force_samples": frame.fluid[0].magnetic_force,
        "frame_ion_electric_force_samples": frame.terms[0].electric,
        "frame_ion_convection_samples": frame.fluid[0].convection,
        "frame_ion_wall_reaction_samples": frame.terms[0].reaction,
        "frame_ion_net_force_samples": frame.terms[0].force,
    }
    result = {}
    for name, value in sources.items():
        if value is None:
            raise RunnerError(
                f"initial force evidence {name} is unavailable", "Snapshot"
            )
        result[name] = _physical_samples(value, shape, context, name)
    return result


def _snapshot_payload(state, frame, context):
    modes = 2 * context.p.K + 1
    nc = context.op.Nc
    nphi = int(context.N)
    host_state = host_state_for_codec(state, context)
    arrays = dict(io.encode_state(host_state, context.geometry))
    coefficient_names = {
        "n_i": "n_i_modes",
        "n_e": "n_e_modes",
        "mom_i": "mom_i_modes",
        "mom_e": "mom_e_modes",
        "p_i": "ion_perp_pressure_modes",
        "p_perp_e": "electron_perp_pressure_modes",
        "p_parallel_e": "electron_parallel_pressure_modes",
    }
    for source, target in coefficient_names.items():
        arrays[target] = arrays.pop(source)
    arrays["n_i_modes"] = host_state.fluid[0].n.reshape(modes, nc)
    arrays["mom_i_modes"] = host_state.fluid[0].mom.reshape(
        modes, 3, nc
    )
    arrays["ion_perp_pressure_modes"] = host_state.fluid[0].perp.reshape(
        modes, nc
    )
    arrays["n_e_modes"] = host_state.fluid[1].n.reshape(modes, nc)
    arrays["mom_e_modes"] = host_state.fluid[1].mom.reshape(
        modes, 3, nc
    )
    arrays["electron_perp_pressure_modes"] = host_state.fluid[1].perp.reshape(
        modes, nc
    )
    arrays["electron_parallel_pressure_modes"] = (
        host_state.fluid[1].parallel.reshape(modes, nc)
    )
    if "B_phi_native" in arrays:
        arrays["B_phi_modes"] = arrays.pop("B_phi_native")
        arrays["B_t_modes"] = arrays.pop("B_t_native")
    else:
        for name in ("B_R", "B_phi", "B_Z"):
            arrays[f"{name}_modes"] = arrays.pop(name)
    if frame is None or getattr(frame, "BN", None) is None:
        raise RunnerError("snapshot requires actual stage B_N samples", "Snapshot")
    if getattr(frame, "fluid", None) is None:
        raise RunnerError(
            "snapshot requires actual stage fluid samples", "Snapshot"
        )
    bn_samples = _physical_samples(
        frame.BN, (nphi, 3, nc), context, "B_N"
    )
    ion = frame.fluid[0]
    electron = frame.fluid[1]
    ion_density = _physical_samples(
        ion.n, (nphi, nc), context, "ion_density"
    )
    ion_velocity = _physical_samples(
        ion.u, (nphi, 3, nc), context, "ion_velocity"
    )
    ion_pressure = _physical_samples(
        ion.perp, (nphi, nc), context, "ion_perp_pressure"
    )
    electron_density = _physical_samples(
        electron.n, (nphi, nc), context, "electron_density"
    )
    electron_perp = _physical_samples(
        electron.perp, (nphi, nc), context, "electron_perp_pressure"
    )
    electron_parallel = _physical_samples(
        electron.parallel,
        (nphi, nc),
        context,
        "electron_parallel_pressure",
    )
    arrays["B_N_samples"] = bn_samples
    arrays["B_N_R_samples"] = bn_samples[:, 0].copy()
    arrays["B_N_phi_samples"] = bn_samples[:, 1].copy()
    arrays["B_N_Z_samples"] = bn_samples[:, 2].copy()
    arrays["ion_velocity_samples"] = ion_velocity
    arrays["ion_velocity_R_samples"] = ion_velocity[:, 0].copy()
    arrays["ion_velocity_phi_samples"] = ion_velocity[:, 1].copy()
    arrays["ion_velocity_Z_samples"] = ion_velocity[:, 2].copy()
    arrays["ion_density_samples"] = ion_density
    arrays["electron_density_samples"] = electron_density
    arrays["ion_perp_pressure_samples"] = ion_pressure
    arrays["electron_perp_pressure_samples"] = electron_perp
    arrays["electron_parallel_pressure_samples"] = electron_parallel
    if getattr(frame, "current", None) is not None:
        arrays["current_modes"] = np.asarray(
            _host(context, frame.current), dtype=np.complex128
        ).reshape(modes, context.op.Ne)
    sample_phis = 2.0 * np.pi * np.arange(nphi, dtype=np.float64) / nphi
    metadata = {
        "layout_kind": _geometry_kind(context.geometry),
        "step": int(state.step),
        "generation": int(state.generation),
        "time_s": float(state.time),
        "seed": context.clt_plan.seed,
        "seed_scope": (
            "STRAIGHT-cylinder CAW approach; not exact finite-R truth"
            if context.clt_plan.seed == "cylindrical-caw"
            else "smooth radial initial velocity"
        ),
        "units": "SI",
        "sample_phis": sample_phis.tolist(),
        "sample_axis": "uniform toroidal physical samples phi=2*pi*j/Nphi",
        "sample_units": {
            "B_N_samples": "T",
            "ion_velocity_samples": "m/s",
            "ion_density_samples": "1/m^3",
            "electron_density_samples": "1/m^3",
            "ion_perp_pressure_samples": "Pa",
            "electron_perp_pressure_samples": "Pa",
            "electron_parallel_pressure_samples": "Pa",
        },
        "mode_axis": list(range(-context.p.K, context.p.K + 1)),
        "mode_units": {
            "n_i_modes": "1/m^3",
            "n_e_modes": "1/m^3",
            "mom_i_modes": "kg/(m^2*s)",
            "mom_e_modes": "kg/(m^2*s)",
            "ion_perp_pressure_modes": "Pa",
            "electron_perp_pressure_modes": "Pa",
            "electron_parallel_pressure_modes": "Pa",
            "B_phi_modes": "T",
            "B_t_modes": "T",
        },
        "B_mode_order": ["B_phi_vertices", "B_t_faces"],
        "current_mode_order": "native E=[normal faces,phi cells]",
        "frame_tag": (
            None
            if frame is None
            else asdict(frame.tag) if not isinstance(frame.tag, dict) else frame.tag
        ),
    }
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, allow_nan=False)
    )
    return arrays


def _frame_arrays(context, frame):
    result = {}
    for owner_name, owner in (
        ("cache", getattr(frame, "cache", None)),
        ("ion", getattr(frame, "fluid", (None, None))[0]),
        ("electron", getattr(frame, "fluid", (None, None))[1]),
    ):
        if owner is None:
            continue
        for name, value in vars(owner).items():
            if isinstance(value, np.ndarray):
                result[f"{owner_name}_{name}"] = value
                continue
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                try:
                    result[f"{owner_name}_{name}"] = _host(context, value)
                except (TypeError, ValueError):
                    pass
    return result


def _atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".tmp.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _deterministic_npz(path, arrays):
    """Write content-stable NPZ for geometry identity across frozen roots."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".tmp.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            with zipfile.ZipFile(
                handle,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as archive:
                for name in sorted(arrays):
                    payload = io_module.BytesIO()
                    np.lib.format.write_array(
                        payload, np.asarray(arrays[name]), allow_pickle=False
                    )
                    info = zipfile.ZipInfo(
                        f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
                    )
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o600 << 16
                    archive.writestr(info, payload.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timed_output(context, writer):
    _synchronize(context)
    started = wall_clock.monotonic()
    value = writer()
    _synchronize(context)
    return value, wall_clock.monotonic() - started


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".tmp.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_geometry_artifact(output, context, metadata):
    """Write one renderable geometry artifact and return its receipt."""
    existing = getattr(context, "_geometry_artifact", None)
    if existing is not None:
        return existing
    geometry = context.geometry
    nc = int(
        context.op.Nc
        if getattr(context, "op", None) is not None
        else len(geometry.cells.R)
    )
    cell_R = np.asarray(geometry.cells.R, dtype=np.float64)
    cell_Z = np.asarray(geometry.cells.Z, dtype=np.float64)
    if getattr(context, "op", None) is not None:
        M0 = np.asarray(_host(context, context.op.M0), dtype=np.float64)
    else:
        M0 = np.asarray(geometry.M0, dtype=np.float64)
    wall_distance = np.asarray(
        geometry.cell_wall_distance, dtype=np.float64
    )
    input_circle = np.asarray(
        [[point.R, point.Z] for point in geometry.polygon_input],
        dtype=np.float64,
    )
    if _geometry_kind(geometry) == "polygon_dual_v1":
        polygons = [
            np.asarray(value, dtype=np.float64)
            for value in geometry.cell_polygons
        ]
        offsets = np.zeros(nc + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([value.shape[0] for value in polygons])
        polygon_points = np.concatenate(polygons, axis=0)
        faces = geometry.faces
        face_vertices = np.asarray(geometry.face_vertices, dtype=np.int64)
        face_normals = np.asarray(geometry.face_normals, dtype=np.float64)
        face_tangents = np.asarray(geometry.face_tangents, dtype=np.float64)
        face_lengths = np.asarray(geometry.face_lengths, dtype=np.float64)
        face_volumes = np.asarray(
            geometry.Rbar_r
            * geometry.face_lengths
            * geometry.face_distances,
            dtype=np.float64,
        )
        face_area_per_radian = np.asarray(
            geometry.Rbar_r * geometry.face_lengths, dtype=np.float64
        )
        numerical = getattr(geometry, "wall_polyline", geometry.polygon)
    else:
        polygon_points = np.empty((0, 2), dtype=np.float64)
        offsets = np.zeros(nc + 1, dtype=np.int64)
        faces = geometry.r_faces
        nf = len(faces.R)
        face_vertices = np.empty((0, 2), dtype=np.int64)
        face_normals = np.empty((0, 2), dtype=np.float64)
        face_tangents = np.empty((0, 2), dtype=np.float64)
        face_lengths = np.empty(0, dtype=np.float64)
        if getattr(context, "op", None) is None:
            face_volumes = np.empty(0, dtype=np.float64)
        else:
            face_volumes = np.asarray(
                _host(context, context.op.M1)[:nf], dtype=np.float64
            )
        face_area_per_radian = np.empty(0, dtype=np.float64)
        numerical = geometry.polygon
    numerical_polygon = np.asarray(
        [[point.R, point.Z] for point in numerical], dtype=np.float64
    )
    arrays = {
        "cell_polygon_points": polygon_points,
        "cell_polygon_offsets": offsets,
        "cell_polygon_counts": np.diff(offsets),
        "cell_R": cell_R,
        "cell_Z": cell_Z,
        "M0": M0,
        "physical_wall_distance": wall_distance,
        "input_circle": input_circle,
        "numerical_polygon": numerical_polygon,
        "face_vertices": face_vertices,
        "face_coords_R": np.asarray(faces.R, dtype=np.float64),
        "face_coords_Z": np.asarray(faces.Z, dtype=np.float64),
        "face_normals": face_normals,
        "face_tangents": face_tangents,
        "face_lengths": face_lengths,
        "face_area_per_radian": face_area_per_radian,
        "face_volumes": face_volumes,
        "geometry_fingerprint": np.asarray(metadata["geometry_fingerprint"]),
        "layout_kind": np.asarray(_geometry_kind(geometry)),
    }
    path = Path(output) / "geometry.npz"
    digest = _deterministic_npz(path, arrays)
    receipt = {
        "file": path.name,
        "path": str(path),
        "sha256": digest,
        "geometry_fingerprint": metadata["geometry_fingerprint"],
        "layout_kind": _geometry_kind(geometry),
    }
    context._geometry_artifact = receipt
    return receipt


def _finite_json(value, path="$"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite JSON number at {path}")
    if isinstance(value, np.generic):
        return _finite_json(value.item(), path)
    if isinstance(value, np.ndarray):
        raise ValueError(f"array leaked into JSON at {path}")
    if isinstance(value, dict):
        return {str(key): _finite_json(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item, f"{path}[{index}]") for index, item in enumerate(value)]
    return value


def _evidence_json(value):
    """Summarize arrays for JSON while retaining full arrays in explicit NPZ."""
    if isinstance(value, np.ndarray):
        numeric = value.dtype.kind in "biufc"
        finite = value[np.isfinite(value)] if numeric else value
        if numeric and finite.size:
            magnitude = np.abs(finite) if value.dtype.kind == "c" else finite
            minimum = float(np.min(magnitude))
            maximum = float(np.max(magnitude))
        else:
            minimum = maximum = None
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "minimum": minimum,
            "maximum": maximum,
            "finite_count": (
                int(np.count_nonzero(np.isfinite(value)))
                if numeric
                else int(value.size)
            ),
            "size": int(value.size),
        }
    if isinstance(value, np.generic):
        return _evidence_json(value.item())
    if isinstance(value, dict):
        return {str(key): _evidence_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_evidence_json(item) for item in value]
    return value


def _coupling_metadata_json(metadata):
    if metadata is None:
        return None
    keys = (
        "coupling",
        "micro_steps",
        "ion_nodes",
        "waveform_rtol",
        "waveform_max_iterations",
        "wr_initial",
        "wr_relaxation",
        "macro_step",
        "left_step",
        "right_step",
        "left_time",
        "right_time",
        "trial",
        "node_version",
        "sweeps",
        "waveform_residual_scaled",
        "waveform_raw_norm",
        "residual_history",
        "final_normalized_residual_max",
        "final_raw_norm",
        "endpoint_residual_summary",
        "work",
        "bundle_diagnostic_work",
        "result_diagnostic_work",
        "base_source_id",
        "endpoint_source_id",
        "base_ion_kinetic_J",
        "endpoint_ion_kinetic_J",
        "epsilon_K_J",
        "coarse_generations",
        "fine_generations",
        "interval_attribution",
        "endpoint",
        "guard_scope",
        "rollback",
        "scope",
    )
    return _evidence_json(
        {name: metadata[name] for name in keys if name in metadata}
    )


def _finalize_timing(timing):
    timing["total_s"] = float(timing["total_s"])
    timing["attributed_s"] = float(
        sum(
            timing.get(name, 0.0)
            for name in (
                "setup_s",
                "initialization_s",
                "resume_evaluation_s",
                "advance_s",
                "output_s",
            )
        )
    )
    timing["unattributed_s"] = max(
        timing["total_s"] - timing["attributed_s"], 0.0
    )
    timing["unattributed_scope"] = (
        "validation, final bookkeeping and final JSON publication time not "
        "assigned to setup, initialization, resume, advance or explicit "
        "output boundaries"
    )
    return timing


def write_strict_json(path, value):
    cleaned = _finite_json(value)
    text = json.dumps(
        cleaned,
        sort_keys=True,
        indent=2,
        allow_nan=False,
        ensure_ascii=True,
    ) + "\n"
    _atomic_text(path, text)


def read_checkpoint_arrays(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise RunnerError("resume checkpoint must be a regular file", "Checkpoint")
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
    except (OSError, ValueError) as error:
        raise RunnerError(
            f"checkpoint schema cannot be read: {error}", "Checkpoint"
        ) from error
    for name in ("checkpoint_schema", "complete_accepted"):
        if name not in arrays:
            raise RunnerError(
                f"checkpoint schema {name} is missing; legacy state was not guessed",
                "Checkpoint",
            )
    schema = np.asarray(arrays["checkpoint_schema"])
    if schema.shape != () or str(schema) != CHECKPOINT_SCHEMA:
        raise RunnerError("checkpoint schema mismatch", "Checkpoint")
    accepted = np.asarray(arrays["complete_accepted"])
    if accepted.shape != () or not bool(accepted):
        raise RunnerError(
            "checkpoint is not a complete accepted endpoint", "Checkpoint"
        )
    return arrays


def _array_text(value):
    array = np.asarray(value)
    if array.shape != ():
        raise RunnerError("checkpoint scalar metadata has wrong shape", "Checkpoint")
    if array.dtype.kind in ("U", "S"):
        return str(array)
    raise RunnerError("checkpoint text metadata has wrong dtype", "Checkpoint")


def _array_scalar(value, dtype):
    array = np.asarray(value)
    if array.shape != ():
        raise RunnerError("checkpoint scalar metadata has wrong shape", "Checkpoint")
    return dtype(array)


_SOURCE_MANIFEST_PATHS = (
    "tools/warp_clt_boundary.py",
    "src/fomhd/io.py",
    "src/fomhd/polygon_io.py",
    "src/fomhd/stage.py",
    "src/fomhd/time.py",
    "src/fomhd/polygon_geometry.py",
    "src/fomhd/polygon_operators.py",
    "tools/circular_equilibrium.py",
    "tools/cylindrical_caw_control.py",
)


def source_fingerprint_manifest():
    """Relative-path source hashes, independent of the frozen checkout root."""
    return {
        relative: (
            hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            if (ROOT / relative).is_file()
            else None
        )
        for relative in _SOURCE_MANIFEST_PATHS
    }


def _source_fingerprint():
    manifest = source_fingerprint_manifest()
    return hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def scheme_parameters(plan):
    """Semantic time/placement controls that must survive restart."""
    return {
        "K": plan.K,
        "mode": plan.mode,
        "Nphi": plan.Nphi,
        "amplitude": plan.amplitude,
        "seed": plan.seed,
        "coupling": plan.coupling,
        "micro_h": plan.micro_h,
        "macro_h": plan.macro_h,
        "M": plan.micro_count,
        "s": plan.ion_nodes,
        "waveform_rtol": plan.waveform_rtol,
        "grid_shift_R": plan.grid_shift_R,
        "grid_shift_Z": plan.grid_shift_Z,
    }


def input_identity(
    plan,
    *,
    geometry_fingerprint,
    baseline=None,
    mesh=None,
    nodes=None,
):
    identity = {
        **scheme_parameters(plan),
        "major_radius": plan.major_radius,
        "minor_radius": plan.minor_radius,
        "F0_T_m": DEFAULT_B0 * plan.major_radius,
        "geometry_fingerprint": geometry_fingerprint,
    }
    if baseline is not None:
        identity["baseline"] = baseline
    if mesh is not None:
        identity["mesh"] = mesh
    if nodes is not None:
        identity["nodes"] = nodes
    return hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _input_fingerprint(context, metadata):
    return input_identity(
        context.clt_plan,
        geometry_fingerprint=metadata["geometry_fingerprint"],
        baseline=metadata["baseline"],
        mesh=metadata["mesh"],
        nodes=metadata["nodes"],
    )


def checkpoint_metadata(context, metadata):
    plan = context.clt_plan
    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "layout_schema": (
            NATIVE_LAYOUT_SCHEMA
            if _geometry_kind(context.geometry) == "polygon_dual_v1"
            else "staircase_v1"
        ),
        "geometry_fingerprint": metadata["geometry_fingerprint"],
        "geometry_file": metadata["geometry_file"],
        "geometry_file_sha256": metadata["geometry_file_sha256"],
        "operator_fingerprint": metadata["operator_fingerprint"],
        "source_fingerprint": _source_fingerprint(),
        "input_fingerprint": _input_fingerprint(context, metadata),
        "scheme": plan.coupling,
        "scheme_parameters": json.dumps(
            scheme_parameters(plan),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "micro_steps": plan.micro_count,
        "ion_nodes": plan.ion_nodes,
        "mesh": metadata["mesh"],
        "baseline": metadata["baseline"],
        "dt": plan.macro_h,
        "start_time": context.p.start_time,
        "nsteps": plan.steps,
    }


def write_checkpoint(path, committed, cumulative, context, metadata):
    host_state = host_state_for_codec(committed.state, context)
    arrays = dict(io.encode_state(host_state, context.geometry))
    identity = checkpoint_metadata(context, metadata)
    for name, value in identity.items():
        if name == "checkpoint_schema":
            continue
        arrays[name] = np.asarray(value)
    arrays["checkpoint_schema"] = np.asarray(CHECKPOINT_SCHEMA)
    arrays["complete_accepted"] = np.asarray(True)
    arrays["step"] = np.asarray(committed.state.step, dtype=np.uint64)
    arrays["generation"] = np.asarray(
        committed.state.generation, dtype=np.uint64
    )
    arrays["time"] = np.asarray(committed.state.time, dtype=np.float64)
    arrays["next_generation"] = np.asarray(
        context.next_generation, dtype=np.uint64
    )
    arrays["next_trial"] = np.asarray(context.next_trial, dtype=np.uint64)
    arrays["next_source"] = np.asarray(context.next_source, dtype=np.uint64)
    arrays["cumulative_json"] = np.asarray(
        io._canonical(io._cumulative_json(cumulative))
    )
    arrays["baseline_divB"] = io.encode_vertex(
        _host(context, cumulative.baseline.divB),
        context.geometry,
        context.p.K,
    )
    digest = _atomic_npz(path, arrays)
    pointer = {
        "file": Path(path).name,
        "sha256": digest,
        "step": int(committed.state.step),
        "time_s": float(committed.state.time),
        "complete_accepted": True,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
    }
    write_strict_json(Path(path).parent / "latest.json", pointer)
    return pointer


def load_checkpoint(path, context, metadata):
    arrays = read_checkpoint_arrays(path)
    expected = checkpoint_metadata(context, metadata)
    text_fields = (
        "layout_schema",
        "geometry_fingerprint",
        "operator_fingerprint",
        "source_fingerprint",
        "input_fingerprint",
        "geometry_file",
        "geometry_file_sha256",
        "scheme",
        "scheme_parameters",
        "mesh",
        "baseline",
    )
    for name in text_fields:
        if name not in arrays:
            raise RunnerError(
                f"checkpoint schema field {name} is missing", "Checkpoint"
            )
        if _array_text(arrays[name]) != str(expected[name]):
            raise RunnerError(f"checkpoint {name} mismatch", "Checkpoint")
    if _array_scalar(arrays["micro_steps"], np.uint64) != expected["micro_steps"]:
        raise RunnerError("checkpoint micro_steps mismatch", "Checkpoint")
    if _array_scalar(arrays["ion_nodes"], np.uint64) != expected["ion_nodes"]:
        raise RunnerError("checkpoint ion_nodes mismatch", "Checkpoint")
    saved_nsteps = int(_array_scalar(arrays["nsteps"], np.uint64))
    accepted_step = int(_array_scalar(arrays["step"], np.uint64))
    if saved_nsteps < accepted_step or saved_nsteps > expected["nsteps"]:
        raise RunnerError(
            "checkpoint nsteps does not bound the accepted endpoint "
            "or requested resume target",
            "Checkpoint",
        )
    if _array_scalar(arrays["dt"], np.float64) != expected["dt"]:
        raise RunnerError("checkpoint dt mismatch", "Checkpoint")
    if _array_scalar(arrays["start_time"], np.float64) != expected["start_time"]:
        raise RunnerError("checkpoint start_time mismatch", "Checkpoint")
    state = io.decode_state(arrays, context.geometry, context.p.K)
    io._upload_state(state, context.compute)
    state.step = int(_array_scalar(arrays["step"], np.uint64))
    state.generation = int(_array_scalar(arrays["generation"], np.uint64))
    state.time = float(_array_scalar(arrays["time"], np.float64))
    expected_time = context.p.start_time + state.step * context.p.dt
    if state.time != expected_time:
        raise RunnerError(
            "checkpoint scheduler endpoint does not match dt", "Checkpoint"
        )
    cumulative = io._cumulative_from_json(
        io.parse_json(_array_text(arrays["cumulative_json"]))
    )
    baseline = io.decode_vertex(
        arrays["baseline_divB"], context.geometry, context.p.K
    )
    cumulative.baseline.divB = context.compute.array(
        baseline, np.complex128
    )
    if (
        cumulative.last_step != state.step
        or cumulative.last_time != state.time
        or cumulative.accepted_intervals
        != state.step - cumulative.baseline.step
    ):
        raise RunnerError(
            "checkpoint cumulative ledger endpoint mismatch", "Checkpoint"
        )
    context.next_generation = int(
        _array_scalar(arrays["next_generation"], np.uint64)
    )
    context.next_trial = int(_array_scalar(arrays["next_trial"], np.uint64))
    context.next_source = int(_array_scalar(arrays["next_source"], np.uint64))
    if (
        context.next_generation <= state.generation
        or context.next_trial < cumulative.last_trial
        or context.next_source < 1
    ):
        raise RunnerError("checkpoint counters are invalid", "Checkpoint")
    restart_tag = StageTag(
        step=state.step,
        generation=state.generation,
        trial=context.next_trial,
        spectral_context=context.spectral_id,
        time=state.time,
        stage="restart",
    )
    frame = stage.evaluate_stage(state, restart_tag, context, False)
    committed = SimpleNamespace(
        state=state, cache=frame.cache, accepted=True
    )
    return committed, cumulative, frame, arrays


def _save_failure(output, error, committed, context, metadata):
    tag = getattr(error, "stage", None)
    if tag is None and committed is not None:
        tag = StageTag(
            step=committed.state.step,
            generation=committed.state.generation,
            trial=getattr(context, "next_trial", 0),
            spectral_context=getattr(context, "spectral_id", 0),
            time=committed.state.time,
            stage="failure",
        )
    checks = [
        io._check_json(check)
        for check in (getattr(error, "checks", None) or [])
    ]
    if not checks:
        checks.append(
            {
                "name": getattr(error, "code", type(error).__name__),
                "stderr": str(error),
                "tag": (
                    None
                    if tag is None
                    else {
                        "step": int(tag.step),
                        "generation": int(tag.generation),
                        "trial": int(tag.trial),
                        "spectral_context": int(tag.spectral_context),
                        "time_s": float(tag.time),
                        "stage": str(tag.stage),
                    }
                ),
                "status": "FAIL",
            }
        )
    receipt = {
        "status": "failed",
        "code": getattr(error, "code", type(error).__name__),
        "message": str(error),
        "name": getattr(error, "code", type(error).__name__),
        "tag": (
            None
            if tag is None
            else {
                "step": int(tag.step),
                "generation": int(tag.generation),
                "trial": int(tag.trial),
                "spectral_context": int(tag.spectral_context),
                "time_s": float(tag.time),
                "stage": str(tag.stage),
            }
        ),
        "traceback": traceback.format_exc(),
        "checks": checks,
        "coupling_metadata": _coupling_metadata_json(
            getattr(error, "coupling_metadata", None)
        ),
        "last_accepted_step": (
            None if committed is None else int(committed.state.step)
        ),
        "checkpoint_retained": (
            output / "checkpoints" / "restart.npz"
        ).is_file(),
    }
    write_strict_json(output / "failure.json", receipt)
    arrays = {
        "code": np.asarray(receipt["code"]),
        "message": np.asarray(str(error)),
        "tag_json": np.asarray(
            json.dumps(receipt["tag"], sort_keys=True, allow_nan=False)
        ),
        "checks_json": np.asarray(
            json.dumps(checks, sort_keys=True, allow_nan=False)
        ),
    }
    if committed is not None:
        host_state = host_state_for_codec(committed.state, context)
        for name, value in io.encode_state(
            host_state, context.geometry
        ).items():
            arrays[f"accepted_{name}"] = value
    for name, value in (
        getattr(error, "partial_arrays", None) or {}
    ).items():
        try:
            arrays[f"partial_{name}"] = _host(context, value)
        except (TypeError, ValueError):
            continue
    _atomic_npz(output / "failure.npz", arrays)
    return receipt


def _peak_rss_bytes():
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _synchronize(context):
    if context is not None:
        context.compute.synchronize()


def _pool_snapshot_from_object(pool):
    return {
        "used_bytes": int(pool.used_bytes()),
        "reserved_bytes": int(pool.total_bytes()),
    }


def _device_pool_snapshot(device):
    if device == "cpu":
        return {"used_bytes": 0, "reserved_bytes": 0}
    import cupy as cp

    pool = cp.get_default_memory_pool()
    return _pool_snapshot_from_object(pool)


def _record_pool_sample(device, tracker, label):
    tracker.setdefault("attempted_sampling_points", []).append(label)
    try:
        _synchronize(tracker.get("_context"))
        sample = _device_pool_snapshot(device)
    except Exception as error:
        tracker["final_available"] = False
        tracker.setdefault("sampling_errors", []).append(
            {
                "label": label,
                "code": type(error).__name__,
                "message": str(error),
            }
        )
        return None
    tracker["used_bytes_final"] = sample["used_bytes"]
    tracker["reserved_bytes_final"] = sample["reserved_bytes"]
    tracker["final_available"] = True
    tracker["sample_successes"] = int(tracker.get("sample_successes", 0)) + 1
    tracker["sampled_peak_used_bytes"] = max(
        tracker.get("sampled_peak_used_bytes", 0),
        sample["used_bytes"],
    )
    tracker["sampled_peak_reserved_bytes"] = max(
        tracker.get("sampled_peak_reserved_bytes", 0),
        sample["reserved_bytes"],
    )
    tracker.setdefault("sampling_points", []).append(label)
    return sample


def _pool_report(tracker):
    successes = int(tracker.get("sample_successes", 0))
    final_available = bool(tracker.get("final_available", False))
    errors = list(tracker.get("sampling_errors", []))
    peak_used = (
        int(tracker.get("sampled_peak_used_bytes", 0))
        if successes
        else None
    )
    peak_reserved = (
        int(tracker.get("sampled_peak_reserved_bytes", 0))
        if successes
        else None
    )
    used_final = (
        int(tracker.get("used_bytes_final", 0))
        if final_available
        else None
    )
    reserved_final = (
        int(tracker.get("reserved_bytes_final", 0))
        if final_available
        else None
    )
    return {
        "available": successes > 0 and not errors,
        "used_bytes_final": used_final,
        "reserved_bytes_final": reserved_final,
        "free_bytes_final": (
            reserved_final - used_final
            if used_final is not None and reserved_final is not None
            else None
        ),
        "reserved_bytes_meaning": "CuPy MemoryPool.total_bytes()",
        "sampled_peak_used_bytes": peak_used,
        "sampled_peak_reserved_bytes": peak_reserved,
        "sampling_points": list(tracker.get("sampling_points", [])),
        "attempted_sampling_points": list(
            tracker.get("attempted_sampling_points", [])
        ),
        "sampling_errors": errors,
        "peak_scope": (
            "sampled at setup, initialization/resume completion and complete "
            "step boundaries only; in-stage temporary peaks are unsampled"
        ),
    }


def _tag_dict(tag):
    if tag is None:
        return None
    return {
        "step": int(tag.step),
        "generation": int(tag.generation),
        "trial": int(tag.trial),
        "spectral_context": int(tag.spectral_context),
        "time_s": float(tag.time),
        "stage": str(tag.stage),
    }


def _check_records(frame):
    if frame is None:
        return []
    return [io._check_json(check) for check in frame.checks]


def run(args):
    started = wall_clock.monotonic()
    plan = validate_controls(args)
    output = validate_output(args.output)
    output.mkdir(parents=True)
    if args.preflight_only:
        return _preflight_only(args, plan, output, started)
    (output / "snapshots").mkdir()
    (output / "checkpoints").mkdir()
    report = {
        "status": "initializing",
        "scientific_status": "pending_parent_validation",
        "created_at_unix": wall_clock.time(),
        "parameters": asdict(plan),
        "output": str(output),
        "snapshots": [],
        "checkpoints": [],
        "raw_checks": [],
        "counters": {
            "initializations": 0,
            "advance_calls": 0,
            "accepted_windows": 0,
            "failed_windows": 0,
            "snapshots": 0,
            "checkpoints_written": 0,
        },
    }
    write_strict_json(output / "report.json", report)
    context = None
    committed = None
    cumulative = None
    metadata = None
    frame = None
    timing = {
        "setup_s": 0.0,
        "initialization_s": 0.0,
        "resume_evaluation_s": 0.0,
        "advance_s": 0.0,
        "output_s": 0.0,
    }
    pool_tracker = {
        "used_bytes_final": 0,
        "reserved_bytes_final": 0,
        "sampled_peak_used_bytes": 0,
        "sampled_peak_reserved_bytes": 0,
        "sampling_points": [],
    }
    try:
        context, metadata = make_context(args)
        geometry_artifact = write_geometry_artifact(
            output, context, metadata
        )
        metadata["geometry_file"] = geometry_artifact["file"]
        metadata["geometry_file_sha256"] = geometry_artifact["sha256"]
        _synchronize(context)
        timing["setup_s"] = wall_clock.monotonic() - started
        pool_tracker["_context"] = context
        _record_pool_sample(plan.device, pool_tracker, "setup_complete")
        report["provenance"] = {
            "source_fingerprint": _source_fingerprint(),
            "source_manifest": source_fingerprint_manifest(),
            "runner": str(Path(__file__).resolve()),
            "python": sys.version,
            "numpy": np.__version__,
            "device": plan.device,
            "memory_probe": metadata["memory"],
            "budget_policy": {
                "hard_gates": "preserved from physical_parameters",
                "observe_policy": "preserved from physical_parameters",
            },
        }
        report["geometry"] = {
            "kind": _geometry_kind(context.geometry),
            "fingerprint": metadata["geometry_fingerprint"],
            "operator_fingerprint": metadata["operator_fingerprint"],
            "file": metadata["geometry_file"],
            "path": str(output / metadata["geometry_file"]),
            "sha256": metadata["geometry_file_sha256"],
            "Nc": int(context.op.Nc),
            "Nf": int(
                getattr(
                    context.op,
                    "Nf",
                    len(getattr(context.geometry, "faces", context.geometry.r_faces).R),
                )
            ),
            "Nv": len(context.geometry.vertices.R),
        }
        report["horizon"] = {
            "micro_h_s": plan.micro_h,
            "macro_h_s": plan.macro_h,
            "M": plan.micro_count,
            "global_steps": plan.steps,
            "segment_steps": plan.segment_steps,
            "expected_global_horizon_s": plan.horizon_s,
            "claim": "physically matched integration times; no time-parallel claim",
        }
        if args.resume is not None:
            _synchronize(context)
            resume_started = wall_clock.monotonic()
            try:
                committed, cumulative, frame, _ = load_checkpoint(
                    args.resume, context, metadata
                )
            finally:
                _synchronize(context)
                timing["resume_evaluation_s"] = (
                    wall_clock.monotonic() - resume_started
                )
                _record_pool_sample(
                    plan.device,
                    pool_tracker,
                    "resume_evaluation_complete",
                )
            report["resume"] = {
                "path": str(Path(args.resume).resolve()),
                "re_evaluated_stage": True,
                "cumulative_reset": False,
            }
        else:
            base, initialization = initial_state(context, metadata)
            seeded, perturbation = apply_default_perturbation(base, context)
            cumulative = cumulative_record()
            captured = {}
            _synchronize(context)
            initialize_started = wall_clock.monotonic()
            try:
                committed = stage.initialize(
                    seeded, context, cumulative, False, captured
                )
            finally:
                _synchronize(context)
                timing["initialization_s"] = (
                    wall_clock.monotonic() - initialize_started
                )
                _record_pool_sample(
                    plan.device,
                    pool_tracker,
                    "initialization_complete",
                )
            frame = SimpleNamespace(**captured)
            report["counters"]["initializations"] += 1
            report["init"] = {
                "elapsed_s": timing["initialization_s"],
                "baseline": _evidence_json(initialization),
                "perturbation": _evidence_json(perturbation),
                "tag": _tag_dict(frame.tag),
                "checks": _check_records(frame),
            }
            report["raw_checks"].extend(
                {"phase": "initial", **record}
                for record in report["init"]["checks"]
            )
            rhs_host = host_state_for_codec(frame.rhs, context)
            rhs_arrays = dict(io.encode_state(rhs_host, context.geometry))
            for prefix, values in (
                ("initialization", initialization),
                ("perturbation", perturbation),
            ):
                for name, value in values.items():
                    if isinstance(value, np.ndarray):
                        rhs_arrays[f"{prefix}_{name}"] = value
            rhs_arrays.update(
                {
                    f"frame_{name}": value
                    for name, value in _frame_arrays(context, frame).items()
                }
            )
            rhs_arrays.update(_initial_force_samples(context, frame))
            rhs_digest, elapsed = _timed_output(
                context,
                lambda: _atomic_npz(output / "initial-rhs.npz", rhs_arrays),
            )
            timing["output_s"] += elapsed
            report["init"]["rhs_file"] = "initial-rhs.npz"
            report["init"]["rhs_sha256"] = rhs_digest

        snapshot_name = f"step-{committed.state.step:010d}.npz"

        def write_initial_outputs():
            _atomic_npz(
                output / "snapshots" / snapshot_name,
                _snapshot_payload(committed.state, frame, context),
            )
            return write_checkpoint(
                output / "checkpoints" / "restart.npz",
                committed,
                cumulative,
                context,
                metadata,
            )

        checkpoint, elapsed = _timed_output(
            context, write_initial_outputs
        )
        timing["output_s"] += elapsed
        report["snapshots"].append(snapshot_name)
        report["counters"]["snapshots"] += 1
        report["checkpoints"] = [checkpoint]
        report["counters"]["checkpoints_written"] += 1

        segment_stop = min(plan.steps, committed.state.step + plan.segment_steps)
        last_snapshot_step = committed.state.step
        report["status"] = "running"
        report["segment"] = {
            "first_unaccepted_step": int(committed.state.step),
            "target_step": int(segment_stop),
        }
        while committed.state.step < segment_stop:
            if wall_clock.monotonic() - started >= plan.wall_seconds:
                report["status"] = "budget_stopped"
                report["stop_reason"] = "wall_seconds before next window"
                break
            _synchronize(context)
            advance_started = wall_clock.monotonic()
            report["counters"]["advance_calls"] += 1
            try:
                result = time_module.advance(committed, cumulative, context)
            except Exception as error:
                _synchronize(context)
                timing["advance_s"] += (
                    wall_clock.monotonic() - advance_started
                )
                report["counters"]["failed_windows"] += 1
                report["failure"] = _save_failure(
                    output, error, committed, context, metadata
                )
                report["status"] = "failed"
                break
            _synchronize(context)
            elapsed = wall_clock.monotonic() - advance_started
            timing["advance_s"] += elapsed
            _record_pool_sample(
                plan.device,
                pool_tracker,
                f"step_{committed.state.step}_complete",
            )
            report["counters"]["accepted_windows"] += 1
            report.setdefault("advance", []).append(
                {
                    "step": int(committed.state.step),
                    "time_s": float(committed.state.time),
                    "elapsed_s": elapsed,
                    "tag": _tag_dict(result.frame.tag),
                    "coupling_metadata": _coupling_metadata_json(
                        getattr(result, "coupling_metadata", None)
                    ),
                    "checks": [
                        io._check_json(check) for check in result.checks
                    ],
                }
            )
            report["raw_checks"].extend(
                {
                    "phase": f"advance:{committed.state.step}",
                    **io._check_json(check),
                }
                for check in result.checks
            )
            frame = result.frame
            should_snapshot = (
                committed.state.step % plan.snapshot_every == 0
                or committed.state.step == segment_stop
            )
            snapshot_name = f"step-{committed.state.step:010d}.npz"

            def write_accepted_outputs():
                if should_snapshot:
                    _atomic_npz(
                        output / "snapshots" / snapshot_name,
                        _snapshot_payload(committed.state, frame, context),
                    )
                return write_checkpoint(
                    output / "checkpoints" / "restart.npz",
                    committed,
                    cumulative,
                    context,
                    metadata,
                )

            checkpoint, elapsed = _timed_output(
                context, write_accepted_outputs
            )
            timing["output_s"] += elapsed
            if should_snapshot:
                report["snapshots"].append(snapshot_name)
                report["counters"]["snapshots"] += 1
                last_snapshot_step = committed.state.step
            report["checkpoints"] = [checkpoint]
            report["counters"]["checkpoints_written"] += 1
        if (
            report["status"] in ("running", "budget_stopped")
            and committed.state.step != last_snapshot_step
        ):
            snapshot_name = f"step-{committed.state.step:010d}.npz"
            _, elapsed = _timed_output(
                context,
                lambda: _atomic_npz(
                    output / "snapshots" / snapshot_name,
                    _snapshot_payload(committed.state, frame, context),
                ),
            )
            timing["output_s"] += elapsed
            report["snapshots"].append(snapshot_name)
            report["counters"]["snapshots"] += 1
        if report["status"] == "running":
            report["status"] = (
                "completed"
                if committed.state.step == plan.steps
                else "segment_completed"
            )
        report["acceptance"] = {
            "target_step": plan.steps,
            "accepted_step": int(committed.state.step),
            "accepted_time_s": float(committed.state.time),
            "accepted_intervals": int(cumulative.accepted_intervals),
            "complete_global_horizon": committed.state.step == plan.steps,
            "complete_requested_segment": (
                committed.state.step == segment_stop
            ),
        }
        _record_pool_sample(plan.device, pool_tracker, "final_complete")
        timing["total_s"] = wall_clock.monotonic() - started
        _finalize_timing(timing)
        report["timing"] = timing
        report["diag"] = {
            "host_peak_rss_bytes": _peak_rss_bytes(),
            "host_vmhwm_scope": "actual process resident high-water mark",
            "device_pool": _pool_report(pool_tracker),
            "elapsed_s": timing["total_s"],
            "snapshot_count": len(report["snapshots"]),
            "checkpoint_count": report["counters"]["checkpoints_written"],
        }
        report["limitations"] = [
            "This run does not establish scientific success.",
            "No time-parallel claim is made for waveform relaxation.",
            "GS discrete mapping requires the promoted equilibrium/mesh pair.",
            "Device pool peaks are boundary-sampled; in-stage temporary peaks are unsampled.",
            "Regular-S/adjoint physical consistency remains parent-owned.",
        ]
        write_strict_json(output / "report.json", report)
        return 0 if report["status"] in ("completed", "segment_completed") else 2
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = {
            "code": getattr(error, "code", type(error).__name__),
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        if context is not None:
            try:
                _save_failure(output, error, committed, context, metadata or {})
            except Exception as evidence_error:
                report["failure"]["evidence_error"] = str(evidence_error)
            _record_pool_sample(plan.device, pool_tracker, "failure_complete")
        timing["total_s"] = wall_clock.monotonic() - started
        _finalize_timing(timing)
        report["timing"] = timing
        report["diag"] = {
            "host_peak_rss_bytes": _peak_rss_bytes(),
            "host_vmhwm_scope": "actual process resident high-water mark",
            "device_pool": _pool_report(pool_tracker),
            "elapsed_s": timing["total_s"],
        }
        write_strict_json(output / "report.json", report)
        return 2


def main(argv=None):
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
