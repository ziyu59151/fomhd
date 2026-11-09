#!/usr/bin/env python3
"""Run segmented nonuniform EAST K=1 complete WR/multirate transactions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np

try:
    import fomhd
    from fomhd import (
        core,
        fluid,
        io,
        massless,
        massless_checks,
        massless_field,
        stage,
        wall,
    )
    from fomhd import time as time_module
    from fomhd import time_multirate, time_waveform
    from fomhd.time import advance
    from fomhd.types import Failure, StageTag, cumulative_record
    from tools import warp_050_experiments as legacy
    from tools import warp_tokamak_cae as cae
except ImportError as error:
    raise RuntimeError(
        "warp_wr_rz requires accepted fomhd complete-runtime source and "
        "tools on PYTHONPATH"
    ) from error


DEFAULT_H_S = 2.046399005804572e-9
DEFAULT_EQUILIBRIUM = Path(
    "/home/lxzeng/runs/fomhd/a0-tokamak-cae__20261105T154538Z/"
    "inputs/g048059.03650"
)
VALID_NODES = (12, 16, 24, 32, 40)
FIXTURE_NODES = (4, 6)
COUPLINGS = ("wr", "synchronous", "waveform")
DEFAULT_WAVEFORM_RTOL = 1.0e-10
WR_MAX_ITERATIONS = 12
WR_RELAXATION = 1.0
SCHEMA = "fomhd-warp-wr-rz-production-v2"
CHECKPOINT_SCHEMA = "fomhd-warp-wr-rz-accepted-checkpoint-v1"


class RunnerError(ValueError):
    """Raised when a runner input, identity, or checkpoint contract is invalid."""


@dataclass(frozen=True)
class Config:
    nodes: int
    device: str
    h: float
    M: int
    s: int
    macros: int
    segment_macros: int
    coupling: str
    output: Path
    equilibrium: Path
    wall_seconds: float
    snapshot_every: int
    amplitude: float
    waveform_rtol: float
    resume: Path | None
    resume_checkpoint: Path | None
    resume_root: Path | None


def _positive_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise RunnerError(f"{name} must be a positive integer") from error
    if isinstance(value, bool) or result <= 0:
        raise RunnerError(f"{name} must be a positive integer")
    return result


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RunnerError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0.0:
        raise RunnerError(f"{name} must be finite and positive")
    return result


def _resume_root(
    resume: Path | None, resume_checkpoint: Path | None
) -> Path | None:
    if resume is not None and resume_checkpoint is not None:
        raise RunnerError("--resume and --resume-checkpoint are mutually exclusive")
    value = resume if resume is not None else resume_checkpoint
    if value is None:
        return None
    path = Path(value).resolve()
    if path.is_dir():
        root = path
    elif path.name == "latest.json":
        root = path.parent.parent
    elif path.suffix == ".npz" and path.parent.name == "checkpoints":
        root = path.parent.parent
    else:
        raise RunnerError(
            "resume checkpoint must be a prior run directory or "
            "a checkpoint path inside its checkpoints directory"
        )
    for name in ("control.npz", "perturbed.npz", "latest.json"):
        if not (root / "checkpoints" / name).is_file():
            raise RunnerError(f"resume checkpoint is missing {name}: {root}")
    if not (root / "report.json").is_file():
        raise RunnerError(f"resume directory is missing report.json: {root}")
    return root


def _scheme(config: Config) -> dict[str, Any]:
    context_coupling = {
        "wr": "multirate_wr",
        "synchronous": "synchronous",
        "waveform": "waveform",
    }[config.coupling]
    effective_macro = float(config.M) * config.h
    micro_steps = 1 if config.coupling == "synchronous" else config.M
    global_nsteps = (
        config.macros * config.M
        if config.coupling == "synchronous"
        else config.macros
    )
    return {
        "algorithm_coupling": config.coupling,
        "context_coupling": context_coupling,
        "kind": "east",
        "mode": 1,
        "amplitude": config.amplitude,
        "h_s": config.h,
        "advance_dt_s": (
            config.h if config.coupling == "synchronous" else effective_macro
        ),
        "M": config.M,
        "s": config.s,
        "micro_steps": micro_steps,
        "effective_macro_s": effective_macro,
        "global_nsteps": global_nsteps,
        "global_requested_macros": config.macros,
        "dt_s": config.h if config.coupling == "synchronous" else effective_macro,
        "ion_nodes": config.s if config.coupling == "wr" else None,
        "wr_initial": "tangent" if config.coupling == "wr" else None,
        "wr_relaxation": WR_RELAXATION if config.coupling == "wr" else None,
        "waveform_rtol": (
            config.waveform_rtol
            if config.coupling in ("wr", "waveform")
            else None
        ),
        "waveform_max_iterations": (
            WR_MAX_ITERATIONS
            if config.coupling in ("wr", "waveform")
            else None
        ),
    }


def _scheme_sha256(config: Config) -> str:
    return hashlib.sha256(
        json.dumps(
            _scheme(config), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _scalar(arrays: dict[str, np.ndarray], name: str, path: Path) -> Any:
    if name not in arrays:
        raise RunnerError(f"checkpoint is missing {name}: {path}")
    value = np.asarray(arrays[name])
    if value.shape != ():
        raise RunnerError(f"checkpoint {name} is not scalar: {path}")
    return value.item()


def _checkpoint_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as error:
        raise RunnerError(f"invalid checkpoint {path}: {error}") from error


def _source_fingerprint() -> dict[str, Any]:
    value = legacy._runtime_fingerprint()
    files = dict(value.get("files", {}))
    files["runner/warp_wr_rz.py"] = legacy._sha256_file(
        Path(__file__).resolve()
    )
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "algorithm": "sha256-files-v1",
        "files": files,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _preflight_resume(config: Config) -> int:
    if config.resume_root is None:
        return 0
    source_fingerprint = _source_fingerprint()["sha256"]
    scheme_hash = _scheme_sha256(config)
    reached_values = []
    for label in ("control", "perturbed"):
        path = config.resume_root / "checkpoints" / f"{label}.npz"
        arrays = _checkpoint_arrays(path)
        if not bool(_scalar(arrays, "accepted", path)):
            raise RunnerError(f"resume checkpoint was not accepted: {path}")
        if int(_scalar(arrays, "global_macros", path)) != config.macros:
            raise RunnerError(
                "resume global macros mismatch: "
                f"{_scalar(arrays, 'global_macros', path)} != {config.macros}"
            )
        if str(_scalar(arrays, "scheme_hash", path)) != scheme_hash:
            raise RunnerError("resume scheme mismatch")
        if str(_scalar(arrays, "source_fingerprint", path)) != source_fingerprint:
            raise RunnerError("resume source fingerprint mismatch")
        expected_nsteps = _scheme(config)["global_nsteps"]
        if int(_scalar(arrays, "global_nsteps", path)) != expected_nsteps:
            raise RunnerError("resume global nsteps mismatch")
        reached_values.append(int(_scalar(arrays, "reached_macros", path)))
    if reached_values[0] != reached_values[1]:
        raise RunnerError("resume control/perturbed reached-macro mismatch")
    if not 0 <= reached_values[0] <= config.macros:
        raise RunnerError("resume reached macros are outside the global horizon")
    return reached_values[0]


def validate_config(
    args: Any,
    *,
    allow_fixture_nodes: bool = False,
    environ: dict[str, str] | None = None,
) -> Config:
    """Validate one segment without creating its output directory."""

    environ = os.environ if environ is None else environ
    output = Path(args.output).resolve()
    if output.exists():
        raise RunnerError(f"output must be a NEW output path: {output}")
    device = str(args.device)
    if device not in ("cpu", "cuda:0"):
        raise RunnerError("device must be cpu or cuda:0")
    if device == "cuda:0" and not str(environ.get("SLURM_JOB_ID", "")).strip():
        raise RunnerError("cuda:0 requires an owned SLURM_JOB_ID allocation")
    nodes = _positive_int(args.nodes, "nodes")
    allowed_nodes = VALID_NODES + (FIXTURE_NODES if allow_fixture_nodes else ())
    if nodes not in allowed_nodes:
        raise RunnerError(
            f"nodes must be one of {VALID_NODES}; fixture nodes 4/6 require "
            "the explicit test API"
        )
    coupling = str(args.coupling)
    if coupling not in COUPLINGS:
        raise RunnerError(f"coupling must be one of {COUPLINGS}")
    M = _positive_int(args.M, "M")
    s = _positive_int(args.s, "s")
    if s not in (3, 5):
        raise RunnerError("s must be 3 or 5")
    macros = _positive_int(args.macros, "macros")
    segment_macros = (
        macros
        if getattr(args, "segment_macros", None) is None
        else _positive_int(args.segment_macros, "segment_macros")
    )
    if segment_macros > macros:
        raise RunnerError("segment_macros must not exceed global macros")
    h = _positive_float(args.h, "h")
    wall_seconds = _positive_float(args.wall_seconds, "wall_seconds")
    if wall_seconds > legacy.MAX_WALL_SECONDS:
        raise RunnerError(
            f"wall_seconds must not exceed {legacy.MAX_WALL_SECONDS:g}"
        )
    snapshot_every = _positive_int(args.snapshot_every, "snapshot_every")
    amplitude = _positive_float(args.amplitude, "amplitude")
    if amplitude > 1.0e-2:
        raise RunnerError("amplitude must not exceed 1e-2")
    waveform_rtol = _positive_float(
        getattr(args, "waveform_rtol", DEFAULT_WAVEFORM_RTOL),
        "waveform_rtol",
    )
    equilibrium = Path(args.equilibrium).resolve()
    if not equilibrium.is_file():
        raise RunnerError(f"pinned EAST equilibrium is missing: {equilibrium}")
    if legacy._sha256_file(equilibrium) != cae.EAST_SHA:
        raise RunnerError("pinned EAST equilibrium SHA256 mismatch")
    resume = getattr(args, "resume", None)
    resume_checkpoint = getattr(args, "resume_checkpoint", None)
    resume_path = None if resume is None else Path(resume)
    checkpoint_path = (
        None if resume_checkpoint is None else Path(resume_checkpoint)
    )
    root = _resume_root(resume_path, checkpoint_path)
    config = Config(
        nodes=nodes,
        device=device,
        h=h,
        M=M,
        s=s,
        macros=macros,
        segment_macros=segment_macros,
        coupling=coupling,
        output=output,
        equilibrium=equilibrium,
        wall_seconds=wall_seconds,
        snapshot_every=snapshot_every,
        amplitude=amplitude,
        waveform_rtol=waveform_rtol,
        resume=resume_path,
        resume_checkpoint=checkpoint_path,
        resume_root=root,
    )
    _preflight_resume(config)
    return config


def deadline_reached(elapsed_s: float, limit_s: float) -> bool:
    return float(elapsed_s) >= float(limit_s)


def validate_matched_records(left: dict[str, Any], right: dict[str, Any]) -> None:
    if int(left["step"]) != int(right["step"]):
        raise RunnerError(f"step mismatch: {left['step']} != {right['step']}")
    if not math.isclose(
        float(left["time_s"]),
        float(right["time_s"]),
        rel_tol=1e-14,
        abs_tol=1e-30,
    ):
        raise RunnerError(
            f"clock mismatch: {left['time_s']:.17g} != {right['time_s']:.17g}"
        )
    for key, message in (
        ("geometry_sha256", "geometry mismatch"),
        ("active_cells", "active-grid mismatch"),
        ("nphi", "Nphi mismatch"),
    ):
        if left.get(key) != right.get(key):
            raise RunnerError(f"{message}: {left.get(key)} != {right.get(key)}")


def _now_counter() -> int:
    return time.perf_counter_ns()


def _sync(context: Any) -> None:
    context.compute.synchronize()


def _host(value: Any, context: Any) -> np.ndarray:
    return np.asarray(cae.host(value, context))


def _geometry_sha256(context: Any) -> str:
    return str(cae.physical_identity(context)["geometry_sha256"])


def _physical_samples(frame: Any, context: Any) -> dict[str, np.ndarray]:
    nc = int(context.op.Nc)
    nphi = int(context.N)

    def vector(value: Any, name: str) -> np.ndarray:
        samples = _host(value, context).reshape(nphi, 3, nc)
        if not np.isfinite(samples).all():
            raise RunnerError(f"{name} contains non-finite physical samples")
        return samples

    def scalar(value: Any, name: str) -> np.ndarray:
        samples = _host(value, context).reshape(nphi, nc)
        if not np.isfinite(samples).all():
            raise RunnerError(f"{name} contains non-finite physical samples")
        return samples

    return {
        "BN": vector(frame.BN, "BN"),
        "u_i": vector(frame.fluid[0].u, "u_i"),
        "n_i": scalar(frame.fluid[0].n, "n_i"),
        "p_i": scalar(frame.fluid[0].perp, "p_i"),
        "p_e_perp": scalar(frame.fluid[1].perp, "p_e_perp"),
        "p_e_parallel": scalar(frame.fluid[1].parallel, "p_e_parallel"),
    }


def _mode_coefficient(samples: np.ndarray, mode: int) -> np.ndarray:
    nphi = int(samples.shape[0])
    phase = np.exp(
        -1j * float(mode) * 2.0 * np.pi * np.arange(nphi) / nphi
    )
    return np.einsum("p...,p->...", samples, phase) / nphi


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _snapshot_arrays(
    frame: Any,
    state: Any,
    context: Any,
    *,
    step: int,
    time_s: float,
    macro: int,
    checks: tuple[Any, ...],
) -> dict[str, np.ndarray]:
    raw_state = io.encode_state(state, context.geometry)
    arrays = {
        f"state_{name}": value for name, value in raw_state.items()
    }
    arrays.update(_physical_samples(frame, context))
    arrays.update(
        {
            "step": np.asarray(step, dtype=np.uint64),
            "time_s": np.asarray(time_s, dtype=np.float64),
            "macro": np.asarray(macro, dtype=np.uint64),
            "accepted": np.asarray(True, dtype=np.bool_),
            "waveform_rtol": np.asarray(
                context.p.waveform_rtol, dtype=np.float64
            ),
            "checks_json": np.asarray(
                json.dumps(legacy._jsonable(list(checks)), allow_nan=False)
            ),
        }
    )
    return arrays


def _save_snapshot(
    output: Path,
    label: str,
    frame: Any,
    state: Any,
    context: Any,
    *,
    step: int,
    time_s: float,
    macro: int,
    checks: tuple[Any, ...],
    name: str,
) -> Path:
    path = output / "snapshots" / f"{label}-{name}.npz"
    _write_npz(
        path,
        _snapshot_arrays(
            frame,
            state,
            context,
            step=step,
            time_s=time_s,
            macro=macro,
            checks=checks,
        ),
    )
    return path


def _input_hash(
    background_arrays: dict[str, np.ndarray], context: Any
) -> str:
    arrays = {name: np.asarray(value) for name, value in background_arrays.items()}
    arrays.update(
        {
            "R": np.asarray(context.geometry.cells.R),
            "Z": np.asarray(context.geometry.cells.Z),
            "polygon": np.asarray(
                [[point.R, point.Z] for point in context.geometry.polygon]
            ),
        }
    )
    mapping_hash = legacy._collection_sha256(arrays)
    return hashlib.sha256(
        f"{mapping_hash}\0{cae.EAST_SHA}".encode("utf-8")
    ).hexdigest()


def _checkpoint_payload(
    committed: SimpleNamespace,
    cumulative: Any,
    context: Any,
    config: Config,
    *,
    reached_macros: int,
    source_fingerprint: str,
    input_hash: str,
    chain_parent_sha256: str,
) -> dict[str, np.ndarray]:
    arrays = io.encode_state(committed.state, context.geometry)
    arrays.update(
        {
            "cumulative_json": np.asarray(
                io._canonical(io._cumulative_json(cumulative))
            ),
            "baseline_divB": io.encode_vertex(
                cumulative.baseline.divB,
                context.geometry,
                context.p.K,
            ),
            "step": np.asarray(committed.state.step, dtype=np.uint64),
            "time": np.asarray(committed.state.time, dtype=np.float64),
            "generation": np.asarray(
                committed.state.generation, dtype=np.uint64
            ),
            "accepted": np.asarray(True, dtype=np.bool_),
            "checkpoint_schema": np.asarray(CHECKPOINT_SCHEMA),
            "source_fingerprint": np.asarray(source_fingerprint),
            "input_hash": np.asarray(input_hash),
            "scheme_hash": np.asarray(_scheme_sha256(config)),
            "geometry_hash": np.asarray(_geometry_sha256(context)),
            "waveform_rtol": np.asarray(
                config.waveform_rtol, dtype=np.float64
            ),
            "global_nsteps": np.asarray(
                _scheme(config)["global_nsteps"], dtype=np.uint64
            ),
            "global_macros": np.asarray(config.macros, dtype=np.uint64),
            "reached_macros": np.asarray(
                reached_macros, dtype=np.uint64
            ),
            "context_coupling": np.asarray(context.p.coupling),
            "micro_steps": np.asarray(
                context.p.micro_steps, dtype=np.uint64
            ),
            "next_trial": np.asarray(
                context.next_trial, dtype=np.uint64
            ),
            "next_generation": np.asarray(
                context.next_generation, dtype=np.uint64
            ),
            "next_source": np.asarray(
                context.next_source, dtype=np.uint64
            ),
            "chain_parent_sha256": np.asarray(chain_parent_sha256),
        }
    )
    return arrays


def _write_checkpoint(
    output: Path,
    label: str,
    committed: SimpleNamespace,
    cumulative: Any,
    context: Any,
    config: Config,
    *,
    reached_macros: int,
    source_fingerprint: str,
    input_hash: str,
    chain_parent_sha256: str,
) -> dict[str, Any]:
    path = output / "checkpoints" / f"{label}.npz"
    _write_npz(
        path,
        _checkpoint_payload(
            committed,
            cumulative,
            context,
            config,
            reached_macros=reached_macros,
            source_fingerprint=source_fingerprint,
            input_hash=input_hash,
            chain_parent_sha256=chain_parent_sha256,
        ),
    )
    return {
        "path": str(path.relative_to(output)),
        "sha256": legacy._sha256_file(path),
        "step": int(committed.state.step),
        "time_s": float(committed.state.time),
        "accepted": True,
        "reached_macros": reached_macros,
    }


def _read_checkpoint(
    path: Path,
    context: Any,
    config: Config,
    *,
    source_fingerprint: str,
    input_hash: str,
) -> dict[str, Any]:
    arrays = _checkpoint_arrays(path)
    expected = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "source_fingerprint": source_fingerprint,
        "input_hash": input_hash,
        "scheme_hash": _scheme_sha256(config),
        "geometry_hash": _geometry_sha256(context),
    }
    for name, value in expected.items():
        if str(_scalar(arrays, name, path)) != value:
            raise RunnerError(f"checkpoint {name} mismatch: {path}")
    if int(_scalar(arrays, "global_macros", path)) != config.macros:
        raise RunnerError("checkpoint global macros mismatch")
    if int(_scalar(arrays, "global_nsteps", path)) != _scheme(config)[
        "global_nsteps"
    ]:
        raise RunnerError("checkpoint global nsteps mismatch")
    if not bool(_scalar(arrays, "accepted", path)):
        raise RunnerError("checkpoint state is not accepted")
    state = io.decode_state(arrays, context.geometry, context.p.K)
    io._upload_state(state, context.compute)
    state.step = int(_scalar(arrays, "step", path))
    state.time = float(_scalar(arrays, "time", path))
    state.generation = int(_scalar(arrays, "generation", path))
    state = massless.canonicalize_state(state, context)
    tag = StageTag(
        step=state.step,
        generation=state.generation,
        trial=int(_scalar(arrays, "next_trial", path)) - 1,
        spectral_context=int(context.spectral_id),
        time=state.time,
        stage="restart",
    )
    frame = stage.evaluate_stage(state, tag, context, False)
    state = frame.canonical_state
    cumulative = io._cumulative_from_json(
        io.parse_json(str(_scalar(arrays, "cumulative_json", path)))
    )
    cumulative.baseline.divB = context.compute.array(
        io.decode_vertex(arrays["baseline_divB"], context.geometry, context.p.K),
        np.complex128,
    )
    if (
        cumulative.last_step != state.step
        or cumulative.last_time != state.time
        or cumulative.accepted_intervals != state.step
    ):
        raise RunnerError("checkpoint cumulative endpoint mismatch")
    if state.time != core.time_at_step(context.p, state.step):
        raise RunnerError("checkpoint scheduler time mismatch")
    context.next_trial = int(_scalar(arrays, "next_trial", path))
    context.next_generation = int(_scalar(arrays, "next_generation", path))
    context.next_source = int(_scalar(arrays, "next_source", path))
    committed = SimpleNamespace(
        state=state,
        cache=frame.cache,
        accepted=True,
    )
    return {
        "committed": committed,
        "frame": frame,
        "cumulative": cumulative,
        "reached_macros": int(_scalar(arrays, "reached_macros", path)),
        "sha256": legacy._sha256_file(path),
    }


def _initialise_ensemble(
    label: str,
    source_state: Any,
    context: Any,
    output: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    cumulative = cumulative_record()
    captured: dict[str, Any] = {}
    committed = stage.initialize(
        legacy._clone_state(source_state, context),
        context,
        cumulative,
        False,
        captured,
    )
    frame = SimpleNamespace(**captured)
    equilibrium_records = legacy._measure_equilibrium(frame, context)
    counts = {
        "pass": 0,
        "observe_fail": 0,
        "hard_fail": 0,
        "unavailable": 0,
    }
    all_checks = list(frame.checks) + equilibrium_records
    legacy._append_checks(
        output / "logs" / f"{label}-checks.jsonl",
        all_checks,
        counts,
    )
    equilibrium = legacy._equilibrium_checks(equilibrium_records)
    initial_rhs = legacy._initial_rhs_record(frame, context)
    legacy._write_json(
        output / "logs" / f"{label}-initial.json",
        {
            "rhs_norms": initial_rhs,
            "equilibrium_residual_checks": equilibrium,
            "equilibrium_residual_acceptance_gate": False,
            "stationary_equilibrium_claim": False,
            "note": (
                "raw mapped-EAST initial defects are retained; the background "
                "is not asserted to be a stationary numerical equilibrium"
            ),
        },
    )
    _write_npz(
        output / "initial_states" / f"{label}.npz",
        io.encode_state(committed.state, context.geometry),
    )
    _save_snapshot(
        output,
        label,
        frame,
        committed.state,
        context,
        step=int(committed.state.step),
        time_s=float(committed.state.time),
        macro=0,
        checks=tuple(all_checks),
        name="initial",
    )
    report["ensembles"][label] = {
        "step": int(committed.state.step),
        "time_s": float(committed.state.time),
        "geometry_sha256": _geometry_sha256(context),
        "active_cells": int(context.op.Nc),
        "nphi": int(context.N),
        "production_accepted": True,
        "initial_rhs_norms": initial_rhs,
        "equilibrium_residual_checks": equilibrium,
    }
    return {
        "committed": committed,
        "frame": frame,
        "cumulative": cumulative,
    }


@contextmanager
def _producer_spy():
    counts: Counter[str] = Counter()
    targets = (
        (time_module, "evaluate_stage", "stage_evaluations"),
        (time_multirate, "evaluate_stage", "stage_evaluations"),
        (time_waveform, "evaluate_stage", "stage_evaluations"),
        (massless, "build_non_electric", "full_ion_calls"),
        (massless, "build_fluid", "transport_producer_calls"),
        (massless, "_derive_electron", "field_electron_calls"),
        (massless, "build_wall", "wall_projections"),
        (massless_field, "evaluate", "field_electron_calls"),
        (massless_field, "feedback_rate", "electron_feedback_calls"),
        (massless_checks, "check_step", "checked_interval_calls"),
        (wall, "build_wall", "wall_projections"),
        (wall, "project_wall", "wall_projections"),
    )
    originals: list[tuple[Any, str, Any]] = []

    try:
        for module, name, counter_name in targets:
            function = getattr(module, name, None)
            if not callable(function):
                continue
            original = function
            originals.append((module, name, original))

            def make_wrapper(fn, label):
                def wrapper(*args, **kwargs):
                    counts[label] += 1
                    return fn(*args, **kwargs)

                return wrapper

            setattr(module, name, make_wrapper(original, counter_name))
        yield counts
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)


def _zero_work() -> dict[str, int]:
    return {
        "complete_advance_calls": 0,
        "stage_evaluations": 0,
        "full_ion_calls": 0,
        "field_electron_calls": 0,
        "electron_feedback_calls": 0,
        "transport_producer_calls": 0,
        "wall_projections": 0,
        "checked_interval_calls": 0,
        "coarse_transport_calls": 0,
        "fine_full_stage_calls": 0,
        "alias_field_evaluations": 0,
        "endpoint_reconstruction_calls": 0,
        "accepted_endpoint_full_stage_calls": 0,
    }


def _merge_work(total: dict[str, int], work: dict[str, Any]) -> None:
    for name, value in work.items():
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            total[name] = total.get(name, 0) + int(value)


def _advance_ensemble(
    context: Any,
    ensemble: dict[str, Any],
    config: Config,
    advance_callable: Callable[..., Any],
) -> dict[str, Any]:
    committed = SimpleNamespace(
        state=ensemble["committed"].state,
        cache=ensemble["committed"].cache,
        accepted=True,
    )
    cumulative = copy.deepcopy(ensemble["cumulative"])
    count = config.M if config.coupling == "synchronous" else 1
    checks: list[Any] = []
    metadata: list[Any] = []
    work = _zero_work()
    work["complete_advance_calls"] = count
    started = _now_counter()
    with _producer_spy() as counts:
        for _ in range(count):
            _sync(context)
            result = advance_callable(committed, cumulative, context)
            _sync(context)
            if not bool(getattr(result, "accepted", False)):
                raise RunnerError("advance did not return an accepted result")
            checks.extend(result.checks)
            result_metadata = getattr(result, "coupling_metadata", None)
            if result_metadata is not None:
                metadata.append(legacy._jsonable(result_metadata))
                raw_work = dict(result_metadata.get("work", {}))
                _merge_work(work, raw_work)
                if config.coupling == "wr":
                    full = int(raw_work.get("full_stage_calls", 0)) + int(
                        raw_work.get("base_full_stage_calls", 0)
                    )
                    work["full_ion_calls"] = max(
                        work["full_ion_calls"], full
                    )
                    work["transport_producer_calls"] = max(
                        work["transport_producer_calls"],
                        int(raw_work.get("coarse_transport_calls", 0)),
                    )
                    work["field_electron_calls"] = max(
                        work["field_electron_calls"],
                        int(raw_work.get("field_evaluations", 0)),
                    )
                    work["accepted_endpoint_full_stage_calls"] = max(
                        work["accepted_endpoint_full_stage_calls"],
                        int(raw_work.get("endpoint_full_stage_calls", 0)),
                    )
        _merge_work(work, counts)
    elapsed_ns = _now_counter() - started
    work["runner_synchronized_ns"] = int(elapsed_ns)
    expected_step = int(ensemble["committed"].state.step) + count
    expected_time = core.time_at_step(context.p, expected_step)
    if committed.state.step != expected_step or committed.state.time != expected_time:
        raise RunnerError("advance result does not match the requested clock")
    if (
        cumulative.last_step != committed.state.step
        or cumulative.last_time != committed.state.time
        or cumulative.accepted_intervals != committed.state.step
    ):
        raise RunnerError("accepted cumulative ledger does not match the clock")
    return {
        "committed": SimpleNamespace(
            state=committed.state,
            cache=committed.cache,
            accepted=True,
        ),
        "frame": result.frame,
        "cumulative": cumulative,
        "checks": tuple(checks),
        "coupling_metadata": metadata,
        "work": work,
        "step": int(committed.state.step),
        "time_s": float(committed.state.time),
        "geometry_sha256": _geometry_sha256(context),
        "active_cells": int(context.op.Nc),
        "nphi": int(context.N),
    }


def _response_arrays(
    control: dict[str, Any],
    perturbed: dict[str, Any],
    context: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    left = _physical_samples(control["frame"], context)
    right = _physical_samples(perturbed["frame"], context)
    delta = {
        "delta_BN": right["BN"] - left["BN"],
        "delta_u_i": right["u_i"] - left["u_i"],
        "delta_n_i": right["n_i"] - left["n_i"],
        "delta_p_i": right["p_i"] - left["p_i"],
        "delta_p_e_perp": right["p_e_perp"] - left["p_e_perp"],
        "delta_p_e_parallel": right["p_e_parallel"] - left["p_e_parallel"],
    }
    n1 = {
        "n1_BN": _mode_coefficient(delta["delta_BN"], 1),
        "n1_u_i": _mode_coefficient(delta["delta_u_i"], 1),
        "n1_n_i": _mode_coefficient(delta["delta_n_i"], 1),
        "n1_p_i": _mode_coefficient(delta["delta_p_i"], 1),
    }
    arrays = {**delta, **n1}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RunnerError("matched physical response is non-finite")
    return arrays, {
        "matched": True,
        "mode": 1,
        "step": control["step"],
        "time_s": control["time_s"],
        "n1_density_max_abs_per_m3": float(np.max(np.abs(n1["n1_n_i"]))),
        "n1_magnetic_max_abs_T": float(np.max(np.abs(n1["n1_BN"]))),
        "n1_velocity_max_abs_m_per_s": float(np.max(np.abs(n1["n1_u_i"]))),
        "delta_density_max_abs_per_m3": float(
            np.max(np.abs(delta["delta_n_i"]))
        ),
        "note": (
            "Direct physical perturbed-minus-control response and retained m=1 "
            "coefficient; not a background-normalized error metric."
        ),
    }


def _check_counts(checks: list[Any]) -> dict[str, int]:
    counts = Counter()
    for check in checks:
        if not getattr(check, "available", True):
            counts["unavailable"] += 1
        elif getattr(check, "passed", False):
            counts["pass"] += 1
        elif getattr(check, "hard", True):
            counts["hard_fail"] += 1
        else:
            counts["observe_fail"] += 1
    return dict(counts)


def _write_failure_evidence(
    path: Path,
    error: BaseException,
    context: Any,
    *,
    label: str,
    step: int,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "failure_json": np.asarray(
            json.dumps(
                {
                    "code": getattr(error, "code", type(error).__name__),
                    "message": str(error),
                    "stage": legacy._jsonable(getattr(error, "stage", None)),
                    "coupling_metadata": legacy._jsonable(
                        getattr(error, "coupling_metadata", None)
                    ),
                    "skip_keys": [],
                },
                allow_nan=False,
            )
        ),
        "label": np.asarray(label),
        "attempted_after_step": np.asarray(step, dtype=np.uint64),
    }
    skipped = []
    partial = getattr(error, "partial_arrays", {}) or {}
    for name, value in partial.items():
        try:
            if isinstance(value, np.ndarray):
                arrays[f"failure__{name}"] = np.asarray(value)
            elif type(value).__module__.startswith("cupy"):
                arrays[f"failure__{name}"] = _host(value, context)
            elif np.isscalar(value) and not isinstance(value, (str, bytes)):
                arrays[f"failure__{name}"] = np.asarray(value)
            else:
                skipped.append(str(name))
        except Exception:
            skipped.append(str(name))
    report = json.loads(str(arrays["failure_json"].item()))
    report["skip_keys"] = skipped
    arrays["failure_json"] = np.asarray(
        json.dumps(report, allow_nan=False)
    )
    _write_npz(path, arrays)


def _record_provenance(
    output: Path,
    source: dict[str, Any],
    config: Config,
    source_fingerprint: dict[str, Any],
) -> None:
    inputs = []
    for role, path in (
        ("runner", Path(__file__).resolve()),
        (
            "runner_pseudocode",
            Path(__file__).resolve().parents[1]
            / "docs/pseudocode/tools/warp_wr_rz.py.md",
        ),
    ):
        if not path.is_file():
            continue
        digest = legacy._sha256_file(path)
        inputs.append({"role": role, "path": str(path), "sha256": digest})
        target = output / "provenance" / "inputs" / role / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    legacy._write_json(
        output / "provenance.json",
        {
            "source_snapshot": source,
            "source_fingerprint": source_fingerprint,
            "inputs": inputs,
            "gfile": {
                "path": str(config.equilibrium),
                "sha256": cae.EAST_SHA,
            },
            "scope": (
                "complete accepted macro transaction with mixed WR work; "
                "scientific acceptance remains parent work"
            ),
        },
    )


def _base_report(config: Config) -> dict[str, Any]:
    scheme = _scheme(config)
    return {
        "schema": SCHEMA,
        "status": "initializing",
        "scope": (
            "actual nonuniform EAST K=1 complete accepted A1-WR transaction; "
            "not experimental validation, speedup, or convergence proof"
        ),
        "production_accepted": False,
        "coupling": config.coupling,
        "case": {
            **asdict(config),
            "output": str(config.output),
            "equilibrium": str(config.equilibrium),
            "resume": None if config.resume_root is None else str(config.resume_root),
            "resume_checkpoint": None,
            "kind": "east",
            "K": 1,
            "mode": 1,
            "electron_inertia": 0,
        },
        "source_plan": scheme,
        "global_requested_macros": config.macros,
        "segment_macros": config.segment_macros,
        "reached_macros": 0,
        "environment": {
            "cpu_threads": 1,
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "memory_caps": {
            "host_bytes": legacy.HOST_MEMORY_CAP,
            "device_bytes": legacy.DEVICE_MEMORY_CAP,
        },
        "ensembles": {},
        "macros": [],
        "work_total": _zero_work(),
        "check_counts": {
            "pass": 0,
            "observe_fail": 0,
            "hard_fail": 0,
            "unavailable": 0,
        },
        "claim_boundary": {
            "speedup_claimed": False,
            "convergence_claimed": False,
            "physical_eigenmode_claimed": False,
            "scientific_acceptance_claimed": False,
        },
    }


def run(
    args: Any,
    *,
    context_factory: Callable[..., Any] | None = None,
    advance_callable: Callable[..., Any] | None = None,
    allow_fixture_nodes: bool = False,
) -> dict[str, Any]:
    """Run one accepted checkpoint segment."""

    config = validate_config(args, allow_fixture_nodes=allow_fixture_nodes)
    output = config.output
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise RunnerError(f"output must be a NEW output path: {output}") from error
    started = time.monotonic()
    report = _base_report(config)
    context: Any | None = None
    ensembles: dict[str, dict[str, Any]] = {}
    source_fingerprint_value = _source_fingerprint()
    source_fingerprint = source_fingerprint_value["sha256"]
    input_hash: str | None = None
    chain_parent = ""
    macro_in_progress: int | None = None
    ensemble_in_progress: str | None = None
    try:
        source = legacy._source_snapshot(output)
        report["source"] = source
        report["source_fingerprint"] = source_fingerprint_value
        legacy._write_json(output / "report.json", report)
        if config.device == "cuda:0":
            import cupy

            cupy.cuda.Device(0).use()
            cupy.get_default_memory_pool().set_limit(
                size=legacy.DEVICE_MEMORY_CAP
            )
        selected_factory = context_factory or legacy._make_context
        context, meta, eq = selected_factory(
            "east",
            config.nodes,
            1,
            config.device,
            config.equilibrium,
            legacy.HOST_MEMORY_CAP,
        )
        _sync(context)
        context.p.memory_limit = legacy.HOST_MEMORY_CAP
        context.p.device_memory_limit = legacy.DEVICE_MEMORY_CAP
        context.p.Nphi = 5
        context.N = 5
        context.p.electron_inertia = 0
        context.p.integrator = "rk4"
        context.p.dt_guard = "warn"
        context.p.start_time = 0.0
        context.p.K = 1
        context.p.nsteps = _scheme(config)["global_nsteps"]
        context.p.dt = _scheme(config)["dt_s"]
        context.p.micro_steps = _scheme(config)["micro_steps"]
        if config.coupling == "wr":
            context.p.coupling = "multirate_wr"
            context.p.ion_nodes = config.s
            context.p.wr_initial = "tangent"
            context.p.wr_relaxation = WR_RELAXATION
            context.p.waveform_rtol = config.waveform_rtol
            context.p.waveform_max_iterations = WR_MAX_ITERATIONS
        elif config.coupling == "waveform":
            context.p.coupling = "waveform"
            context.p.ion_nodes = None
            context.p.wr_initial = None
            context.p.wr_relaxation = None
            context.p.waveform_rtol = config.waveform_rtol
            context.p.waveform_max_iterations = WR_MAX_ITERATIONS
        else:
            context.p.coupling = "synchronous"
            context.p.ion_nodes = None
            context.p.wr_initial = None
            context.p.wr_relaxation = None
        time_module.validate_time_controls(context.p)
        report["case"].update(legacy._jsonable(meta))
        report["case"]["equilibrium_sha256"] = legacy._sha256_file(
            config.equilibrium
        )
        report["actual_grid"] = {
            "Nc": int(context.op.Nc),
            "Ne": int(context.op.Ne),
            "Nb": int(context.op.Nb),
            "Nphi": int(context.N),
        }
        report["physical_identity"] = legacy._jsonable(
            cae.physical_identity(context)
        )
        parameter_record = legacy._jsonable(asdict(context.p))
        report["model_metadata"] = {
            key: parameter_record[key]
            for key in (
                "species",
                "epsilon0",
                "mu0",
                "B_min",
                "rank_rtol",
                "K",
                "Nphi",
                "electron_inertia",
                "integrator",
                "coupling",
                "micro_steps",
                "ion_nodes",
                "wr_initial",
                "wr_relaxation",
                "waveform_rtol",
                "ref",
            )
            if key in parameter_record
        }
        report["budgets"] = parameter_record.get("budgets", {})
        legacy._write_json(output / "resolved.json", asdict(context.p))
        initial, background_arrays, background_mapping = cae.background(
            context, meta, eq
        )
        report["case"]["background_mapping"] = legacy._jsonable(
            background_mapping
        )
        _write_npz(
            output / "input_mapping.npz",
            {
                **background_arrays,
                "R": np.asarray(context.geometry.cells.R),
                "Z": np.asarray(context.geometry.cells.Z),
                "polygon": np.asarray(
                    [[point.R, point.Z] for point in context.geometry.polygon]
                ),
            },
        )
        input_hash = _input_hash(background_arrays, context)
        seed = legacy._clone_state(initial, context)
        cae.perturb(
            seed,
            context,
            meta,
            mode=1,
            amplitude=config.amplitude,
            reference=None,
        )
        mapped_velocity = np.asarray(context.seed_arrays["mapped_velocity"])
        if not np.isfinite(mapped_velocity).all() or not np.any(
            mapped_velocity != 0.0
        ):
            raise RunnerError("K=1 perturbation mapping is zero or non-finite")
        _write_npz(
            output / "perturbation_mapping.npz",
            {
                name: np.asarray(value)
                for name, value in context.seed_arrays.items()
            },
        )
        report["seed"] = legacy._jsonable(meta)
        report["initial_perturbation"] = {
            "mode": 1,
            "amplitude": config.amplitude,
            "mapped_velocity_max_abs_m_per_s": float(
                np.max(np.abs(mapped_velocity))
            ),
            "nonzero": True,
        }
        _record_provenance(
            output,
            source,
            config,
            source_fingerprint_value,
        )
        if config.resume_root is None:
            ensembles["control"] = _initialise_ensemble(
                "control", initial, context, output, report
            )
            ensembles["perturbed"] = _initialise_ensemble(
                "perturbed", seed, context, output, report
            )
            report["reached_macros"] = 0
        else:
            try:
                parent_report = json.loads(
                    (config.resume_root / "report.json").read_text(
                        encoding="utf-8"
                    )
                )
            except (OSError, json.JSONDecodeError) as error:
                raise RunnerError(f"cannot read resume report: {error}") from error
            report["parent"] = {
                "directory": str(config.resume_root),
                "report_sha256": legacy._sha256_file(
                    config.resume_root / "report.json"
                ),
                "source_accepted_macros": int(
                    parent_report.get("reached_macros", 0)
                ),
            }
            controls = _read_checkpoint(
                config.resume_root / "checkpoints/control.npz",
                context,
                config,
                source_fingerprint=source_fingerprint,
                input_hash=input_hash,
            )
            perturbed = _read_checkpoint(
                config.resume_root / "checkpoints/perturbed.npz",
                context,
                config,
                source_fingerprint=source_fingerprint,
                input_hash=input_hash,
            )
            if controls["reached_macros"] != perturbed["reached_macros"]:
                raise RunnerError("resume control/perturbed macro mismatch")
            ensembles["control"] = controls
            ensembles["perturbed"] = perturbed
            report["reached_macros"] = controls["reached_macros"]
            report["parent"].update(
                {
                    "control_checkpoint_sha256": controls["sha256"],
                    "perturbed_checkpoint_sha256": perturbed["sha256"],
                }
            )
            chain_parent = legacy._sha256_file(
                config.resume_root / "checkpoints/latest.json"
            )
        latest: dict[str, Any] = {}
        if config.resume_root is None:
            for label, ensemble in ensembles.items():
                report["ensembles"][label].update(
                    {
                        "production_accepted": True,
                        "geometry_sha256": _geometry_sha256(context),
                    }
                )
        else:
            for label, ensemble in ensembles.items():
                report["ensembles"][label] = {
                    "step": int(ensemble["committed"].state.step),
                    "time_s": float(ensemble["committed"].state.time),
                    "geometry_sha256": _geometry_sha256(context),
                    "active_cells": int(context.op.Nc),
                    "nphi": int(context.N),
                    "production_accepted": True,
                    "checkpoint_sha256": ensemble["sha256"],
                }
        if report["reached_macros"]:
            for label, ensemble in ensembles.items():
                latest[label] = _write_checkpoint(
                    output,
                    label,
                    ensemble["committed"],
                    ensemble["cumulative"],
                    context,
                    config,
                    reached_macros=report["reached_macros"],
                    source_fingerprint=source_fingerprint,
                    input_hash=input_hash,
                    chain_parent_sha256=chain_parent,
                )
        legacy._write_json(output / "report.json", report)
        selected_advance = advance_callable or advance
        segment_start = report["reached_macros"]
        segment_end = min(
            config.macros, segment_start + config.segment_macros
        )
        all_checks: list[Any] = []
        while report["reached_macros"] < segment_end:
            elapsed = time.monotonic() - started
            if deadline_reached(elapsed, config.wall_seconds):
                report["status"] = "budget_stopped"
                report["stop_reason"] = "wall_deadline_before_next_macro"
                break
            if _source_fingerprint()["sha256"] != source_fingerprint:
                raise Failure("SourceMutation", "runtime source changed during segment")
            macro_in_progress = report["reached_macros"] + 1
            local: dict[str, dict[str, Any]] = {}
            for label in ("control", "perturbed"):
                ensemble_in_progress = label
                local[label] = _advance_ensemble(
                    context,
                    ensembles[label],
                    config,
                    selected_advance,
                )
            control, perturbed = local["control"], local["perturbed"]
            validate_matched_records(control, perturbed)
            macro_work = _zero_work()
            _merge_work(macro_work, control["work"])
            _merge_work(macro_work, perturbed["work"])
            _merge_work(report["work_total"], macro_work)
            macro_checks = list(control["checks"]) + list(perturbed["checks"])
            all_checks.extend(macro_checks)
            counts = _check_counts(macro_checks)
            for name, value in counts.items():
                report["check_counts"][name] += value
            report["checks"] = report["check_counts"]
            for check in macro_checks:
                legacy._append_jsonl(
                    output / "logs/accepted-checks.jsonl",
                    legacy._check_record(check),
                )
            response_arrays, response = _response_arrays(
                control, perturbed, context
            )
            _write_npz(
                output / "responses" / f"macro-{macro_in_progress:06d}.npz",
                response_arrays,
            )
            report["reached_macros"] = macro_in_progress
            for label, result in local.items():
                ensembles[label].update(result)
                if macro_in_progress % config.snapshot_every == 0:
                    _save_snapshot(
                        output,
                        label,
                        result["frame"],
                        result["committed"].state,
                        context,
                        step=result["step"],
                        time_s=result["time_s"],
                        macro=macro_in_progress,
                        checks=tuple(result["checks"]),
                        name=f"macro-{macro_in_progress:06d}",
                    )
                latest[label] = _write_checkpoint(
                    output,
                    label,
                    result["committed"],
                    result["cumulative"],
                    context,
                    config,
                    reached_macros=macro_in_progress,
                    source_fingerprint=source_fingerprint,
                    input_hash=input_hash,
                    chain_parent_sha256=chain_parent,
                )
                report["ensembles"][label].update(
                    {
                        "step": result["step"],
                        "time_s": result["time_s"],
                        "production_accepted": True,
                        "checkpoint_sha256": latest[label]["sha256"],
                    }
                )
            report["ledger"] = legacy._jsonable(
                io._cumulative_json(ensembles["control"]["cumulative"])
            )
            metadata = {
                "macro": macro_in_progress,
                "step": control["step"],
                "time_s": control["time_s"],
                "production_accepted": True,
                "work": macro_work,
                "checks": counts,
                "control_coupling_metadata": control["coupling_metadata"],
                "perturbed_coupling_metadata": perturbed["coupling_metadata"],
                "coupling_metadata": (
                    control["coupling_metadata"][-1]
                    if control["coupling_metadata"]
                    else None
                ),
                "residual_history": (
                    control["coupling_metadata"][-1].get("residual_history", [])
                    if control["coupling_metadata"]
                    else []
                ),
                "response": response,
            }
            report["macros"].append(metadata)
            legacy._append_jsonl(output / "logs/history.jsonl", metadata)
            sample = legacy._memory_sample(
                context, time.monotonic() - started
            )
            legacy._append_jsonl(output / "logs/memory.jsonl", sample)
            if (
                sample.get("rss_bytes") is not None
                and int(sample["rss_bytes"]) > legacy.HOST_MEMORY_CAP
            ):
                report["status"] = "budget_stopped"
                report["stop_reason"] = "host_memory_cap"
                break
            if (
                sample.get("device_pool_used_bytes") is not None
                and int(sample["device_pool_used_bytes"])
                > legacy.DEVICE_MEMORY_CAP
            ):
                report["status"] = "budget_stopped"
                report["stop_reason"] = "device_memory_cap"
                break
            latest_meta = {
                "schema": CHECKPOINT_SCHEMA,
                "reached_macros": report["reached_macros"],
                "global_requested_macros": config.macros,
                "global_nsteps": _scheme(config)["global_nsteps"],
                "scheme_hash": _scheme_sha256(config),
                "source_fingerprint": source_fingerprint,
                "input_hash": input_hash,
                "geometry_hash": _geometry_sha256(context),
                "parent_checkpoint_sha256": chain_parent,
                "checkpoints": latest,
            }
            legacy._write_json(output / "checkpoints/latest.json", latest_meta)
            macro_in_progress = None
            ensemble_in_progress = None
            legacy._write_json(output / "report.json", report)
        if report.get("status") == "initializing":
            if report["reached_macros"] >= config.macros:
                report["status"] = "completed"
            elif report["reached_macros"] >= segment_end:
                report["status"] = "segment_complete"
            else:
                report["status"] = "budget_stopped"
        report["production_accepted"] = True
        report["reached_horizon_s"] = (
            report["reached_macros"] * _scheme(config)["effective_macro_s"]
        )
        report["elapsed_s"] = time.monotonic() - started
        report["source_fingerprint_end"] = _source_fingerprint()
        if (
            report["source_fingerprint_end"]["sha256"]
            != source_fingerprint
        ):
            raise Failure("SourceMutation", "runtime source changed during run")
    except Exception as error:
        report["status"] = "failed"
        report["production_accepted"] = False
        report["failure"] = {
            "code": getattr(error, "code", type(error).__name__),
            "message": str(error),
            "traceback": traceback.format_exc(),
            "macro": macro_in_progress,
            "ensemble": ensemble_in_progress,
        }
        failure_dir = output / "failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        for label, ensemble in ensembles.items():
            try:
                _write_npz(
                    failure_dir / f"left-{label}.npz",
                    _checkpoint_payload(
                        ensemble["committed"],
                        ensemble["cumulative"],
                        context,
                        config,
                        reached_macros=report.get("reached_macros", 0),
                        source_fingerprint=source_fingerprint,
                        input_hash=input_hash or "",
                        chain_parent_sha256=chain_parent,
                    ),
                )
            except Exception:
                pass
        if config.resume_root is not None:
            for label in ("control", "perturbed"):
                source_checkpoint = (
                    config.resume_root / "checkpoints" / f"{label}.npz"
                )
                if source_checkpoint.is_file():
                    shutil.copy2(
                        source_checkpoint,
                        failure_dir / f"parent-{label}.npz",
                    )
        failure_json = output / "failures" / "failure.json"
        legacy._write_json(failure_json, report["failure"])
        if context is not None:
            _write_failure_evidence(
                failure_json.with_suffix(".npz"),
                error,
                context,
                label=ensemble_in_progress or "top-level",
                step=macro_in_progress or 0,
            )
    finally:
        report["elapsed_s"] = time.monotonic() - started
        try:
            legacy._write_json(output / "report.json", report)
        except Exception:
            pass
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--h", type=float, default=DEFAULT_H_S)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--s", type=int, choices=(3, 5), required=True)
    parser.add_argument("--macros", type=int, required=True)
    parser.add_argument("--segment-macros", type=int)
    parser.add_argument("--coupling", choices=COUPLINGS, default="wr")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--equilibrium", type=Path, default=DEFAULT_EQUILIBRIUM
    )
    parser.add_argument("--wall-seconds", type=float, default=1600.0)
    parser.add_argument("--snapshot-every", type=int, default=1)
    parser.add_argument("--amplitude", type=float, default=1.0e-4)
    parser.add_argument(
        "--waveform-rtol", type=float, default=DEFAULT_WAVEFORM_RTOL
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run(args)
    except RunnerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(Path(args.output).resolve()),
                "production_accepted": report["production_accepted"],
                "global_requested_macros": report["global_requested_macros"],
                "reached_macros": report["reached_macros"],
                "failure": report.get("failure"),
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if report["status"] in ("completed", "segment_complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
