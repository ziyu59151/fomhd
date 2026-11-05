#!/usr/bin/env python3
"""Fixed-domain finite-inertia momentum and quiet-reference measurements."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src" if (ROOT / "src/fomhd").exists() else ROOT),
               str(ROOT / "tests/warp")]

from foundation_fixture import parameters
from workflow_fixture import cumulative
from mms_fixture import require_device, serialize_check_record, write_failure_arrays, write_failure_checks
from fomhd import core, io, stage
from fomhd.backend import Backend
from fomhd.geometry import Point2, build_geometry
from fomhd.operators import build_operators
from fomhd.time import advance
from fomhd.types import Context, Failure, StageTag


# Fixed physical toroidal flux for B_phi = 1e-5 T m / R on the old rectangle.
FIXED_FLUX = 1e-5 * .6 * math.log(1.5 / .9)
HORIZON = 5.567242158847295e-7
SCOPE = ("Finite-mass fixed-domain numerical controls; not Tokamak equilibrium "
         "validation or proof that a staircase causes the previous momentum floor.")


def build_context(nodes, shape, *, original=False, device="cpu"):
    if not 3 <= nodes <= 32 or shape not in ("rectangle", "disk"):
        raise ValueError("nodes must be in [3,32] and shape rectangle or disk")
    lo_r, lo_z, length = (.9, -.1, .6) if original else (1., 0., 1.)
    h = length / nodes
    if shape == "rectangle":
        polygon = [Point2(lo_r, lo_z), Point2(lo_r+length, lo_z),
                   Point2(lo_r+length, lo_z+length), Point2(lo_r, lo_z+length)]
    else:
        polygon = [Point2(1.5+.45*math.cos(a), .5+.45*math.sin(a))
                   for a in np.arange(128)*2*math.pi/128]
    geometry = build_geometry(lo_r+(np.arange(nodes)+.5)*h,
                              lo_z+(np.arange(nodes)+.5)*h, polygon)
    backend = Backend(require_device(device))
    p = parameters(1)
    p.integrator, p.dt_guard = "rk4", "error"
    p.memory_limit = p.device_memory_limit = 8 << 30
    operators = build_operators(geometry, p.K, {"memory_cap_bytes": 8 << 30,
                               "rank_rtol": p.rank_rtol}, backend)
    return Context(p, geometry, operators, backend, p.Nphi)


def reference(context):
    state = stage.closed_reference(context, 1e12, 1e-9, 1e-9, 1e-5)
    nb, K = context.op.Nb, context.p.K
    harmonic = state.B[K*nb:(K+1)*nb]
    flux = context.compute.scalar((context.op.topology_flux_row @ harmonic)[0])
    harmonic *= FIXED_FLUX / flux
    return state


def integrate(vector, context):
    """Reduce physical cylindrical vectors into Cartesian total force/momentum."""
    xp = context.compute.xp
    nc, N = context.op.Nc, context.N
    v = xp.real(vector).reshape(N, 3, nc)
    phi = 2*math.pi*xp.arange(N)/N
    c, s = xp.cos(phi)[:, None], xp.sin(phi)[:, None]
    cart = xp.stack((v[:, 0]*c-v[:, 1]*s, v[:, 0]*s+v[:, 1]*c, v[:, 2]), axis=1)
    w = 2*math.pi/N*xp.asarray(context.op.M0)
    return context.compute.to_host(xp.sum(cart*w[None, None, :], axis=(0, 2))).real


def endpoint_balance(state, frame, context):
    nc, N, backend = context.op.Nc, context.N, context.compute
    xp = backend.xp

    def node_e(retained):
        return core.sample_apply(context.op.H, core.inverse(retained, context.op.Ne,
                                 context, True), N, backend).reshape(N, 3, nc)

    el, eldot = node_e(frame.cache.EL), node_e(frame.ELdot)
    bn = frame.BN.reshape(N, 3, nc)
    bdot = core.inverse(core.sample_apply(context.op.QB, frame.rhs.B,
                        2*context.p.K+1, backend), 3*nc, context, True).reshape(N, 3, nc)
    cross = lambda a, b: xp.cross(a, b, axisa=1, axisb=1, axisc=1)
    field_rate = integrate(context.p.epsilon0*(cross(eldot, bn)+cross(el, bdot)), context)
    fluid_rate = [integrate(core.inverse(f.mom, 3*nc, context, True), context)
                  for f in frame.rhs.fluid]
    total_rate = sum(fluid_rate)+field_rate
    outward = np.asarray(frame.momentum.outward)
    return dict(total_rate_N=total_rate.tolist(),
                fluid_rate_N=[v.tolist() for v in fluid_rate],
                field_rate_N=field_rate.tolist(), outward_N=outward.tolist(),
                balance_defect_N=(total_rate+outward).tolist(),
                numerical_wall_reaction_N=list(frame.momentum.wall_reaction),
                momentum_kg_m_per_s=list(frame.momentum.value),
                check_counts=dict(Counter("pass" if c.passed else "hard_fail" if c.hard
                                          else "observe_fail" for c in frame.checks)))


def evaluate(state, context):
    return stage.evaluate_stage(state, StageTag(state.step, state.generation, 1, 0,
                                state.time, "spatial_evidence"), context, False)


def static_control(context, shape):
    state = reference(context)
    nc, K, amplitude = context.op.Nc, context.p.K, 1e-10
    for mode in (-1, 1):
        for field in (state.fluid[1].perp, state.fluid[1].parallel):
            field[(K+mode)*nc:(K+mode+1)*nc] = amplitude/2
    frame = evaluate(state, context)
    report = endpoint_balance(state, frame, context)
    area = float(np.sum(np.asarray(context.geometry.M0) / np.asarray(context.geometry.cells.R)))
    reference_area = 1.0 if shape == "rectangle" else math.pi*.45**2
    report.update(area_h_m2=area, reference_area_m2=reference_area,
                  analytic_outward_x_N=math.pi*amplitude*area,
                  reference_outward_x_N=math.pi*amplitude*reference_area,
                  pressure_amplitude_Pa=amplitude)
    return report


def evolve(context, *, steps, horizon=HORIZON, quiet=False):
    if steps < 1 or not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("positive steps and finite positive horizon are required")
    p, backend, nc, K = context.p, context.compute, context.op.Nc, context.p.K
    p.nsteps, p.dt = steps, horizon/steps
    state = reference(context)
    if not quiet:
        r, z = np.asarray(context.geometry.cells.R), np.asarray(context.geometry.cells.Z)
        envelope = np.cos(math.pi*(r-1.2))**2 * np.cos(math.pi*(z-.2))**2
        for mode in (-1, 1):
            state.fluid[1].n[(K+mode)*nc:(K+mode+1)*nc] = backend.asarray(.5e6*envelope)
    ledger, captured = cumulative(), {}
    initial = io.encode_state(state, context.geometry)
    committed = stage.initialize(state, context, ledger, False, captured)
    first_frame = SimpleNamespace(**captured)
    initial_balance = endpoint_balance(committed.state, first_frame, context)
    initial_rhs = io.encode_state(first_frame.rhs, context.geometry)
    history = [list(first_frame.momentum.value)]
    times, intervals, reports, checks = [0.], [], [], []
    failure = None
    partial_arrays = {}
    started = time.monotonic()
    for _ in range(steps):
        try:
            result = advance(committed, ledger, context)
        except Failure as error:
            failure = dict(code=error.code, message=str(error), stage=vars(error.stage) if getattr(error, "stage", None) else None)
            partial_arrays = {"failure__"+str(k): backend.to_host(v) for k, v in (getattr(error, "partial_arrays", {}) or {}).items()}
            rejected = getattr(error, "time_step_report", None)
            if rejected is not None:
                reports.append(rejected)
            checks.extend(serialize_check_record(c) for c in error.checks)
            break
        history.append(list(result.frame.momentum.value))
        times.append(committed.state.time)
        selected = [c for c in result.checks if c.name == "momentum_x"]
        if len(selected) != 1:
            raise ValueError("each accepted interval must have one momentum_x record")
        intervals.append(float(selected[0].raw))
        checks.extend(serialize_check_record(c) for c in result.checks if not c.passed)
        r = result.time_step_report
        reports.append({key: r[key] for key in ("accepted", "omega_dt", "samples_per_cycle", "reasons")})
    backend.synchronize()
    final_frame = evaluate(committed.state, context)
    final = io.encode_state(committed.state, context.geometry)
    drift = 0.
    scales = {"n_i": 1e12, "n_e": 1e12, "p_i": 1e-9,
              "p_perp_e": 1e-9, "p_parallel_e": 1e-9,
              "mom_i": p.species[0].mass*1e12*p.ref.velocity,
              "mom_e": p.species[1].mass*1e12*p.ref.velocity,
              "B_R": 1e-5, "B_phi": 1e-5, "B_Z": 1e-5}
    field_drifts = {key: float(np.max(np.abs(final[key]-initial[key]), initial=0))/scale
                    for key, scale in scales.items()}
    drift = max(field_drifts.values())
    arrays = {"time_s": np.asarray(times), "momentum": np.asarray(history),
              "momentum_x_interval": np.asarray(intervals)}
    arrays.update({f"initial__{k}": v for k, v in initial.items()})
    arrays.update({f"initial_rhs__{k}": v for k, v in initial_rhs.items()})
    arrays.update({f"final__{k}": v for k, v in final.items()})
    arrays.update(partial_arrays)
    report = dict(status="failed" if failure else "completed", failure=failure,
                  accepted_steps=committed.state.step, requested_steps=steps,
                  horizon_s=horizon, dt_s=p.dt, quiet=quiet, fixed_flux_Wb=FIXED_FLUX,
                  elapsed_s=time.monotonic()-started, initial_balance=initial_balance,
                  final_balance=endpoint_balance(committed.state, final_frame, context),
                  signed_cumulative_momentum_x_defect=float(np.sum(intervals)),
                  maximum_relative_state_drift=drift, field_drift_over_declared_scale=field_drifts,
                  drift_scales=scales, time_step_reports=reports,
                  diagnostic_failures=checks)
    return report, arrays


def snapshot(output):
    import fomhd
    package = Path(fomhd.__file__).resolve().parent
    paths = [(p, Path("src/fomhd") / p.name) for p in sorted(package.glob("*.py"))]
    paths += [(ROOT/r, Path(r)) for r in ("tools/warp_momentum_study.py",
              "tests/warp/foundation_fixture.py", "tests/warp/workflow_fixture.py",
              "tests/warp/mms_fixture.py", "docs/pseudocode/tools/warp_momentum_study.py.md")]
    hashes = {}
    for source, relative in paths:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=("static", "evolve", "quiet"), default="static")
    parser.add_argument("--nodes", default="4,6,8,12,16")
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--horizon", type=float, default=HORIZON)
    args = parser.parse_args()
    if args.device == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        parser.error("CUDA evidence requires an owned Slurm allocation")
    require_device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(scope=SCOPE, device=args.device, mode=args.mode,
                   slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                   hashes=snapshot(args.output), cases=[])
    for n in [int(v) for v in args.nodes.split(",")]:
        for shape in (("rectangle", "disk") if args.mode == "static" else ("rectangle",)):
            name = f"{shape}-n{n}"
            started = time.monotonic()
            try:
                context = build_context(n, shape, original=args.mode != "static", device=args.device)
                if args.mode == "static":
                    result = dict(status="completed", **static_control(context, shape))
                else:
                    result, arrays = evolve(context, steps=args.steps, horizon=args.horizon,
                                            quiet=args.mode == "quiet")
                    np.savez_compressed(args.output/f"{name}.npz", **arrays)
                result.update(active_cells=context.op.Nc, h_m=(1. if args.mode == "static" else .6)/n)
                del context
            except Exception as error:
                result = dict(status="failed", failure=dict(code=getattr(error, "code", type(error).__name__),
                                                             message=str(error)))
                result["failure_arrays"] = write_failure_arrays(args.output, error, name+"-failure")
                result["failure_checks"] = write_failure_checks(args.output, error, name+"-failure")
            result.update(name=name, nodes=n, shape=shape, total_elapsed_s=time.monotonic()-started)
            summary["cases"].append(result)
            (args.output/"report.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
            print(json.dumps({k:result[k] for k in ("name", "status", "total_elapsed_s")}), flush=True)
    return int(any(c["status"] != "completed" for c in summary["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
