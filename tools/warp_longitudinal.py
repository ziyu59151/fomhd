#!/usr/bin/env python3
"""Unforced local finite-mass longitudinal time control, with a periodic adapter."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests/warp")]

from foundation_fixture import parameters  # noqa: E402
from massless_verification_fixture import _periodic_geometry, _periodic_operators  # noqa: E402
from mms_fixture import require_device, serialize_check_record, write_failure_arrays, write_failure_checks  # noqa: E402
from workflow_fixture import cumulative  # noqa: E402
from fomhd.backend import Backend  # noqa: E402
from fomhd.core import allocate_state  # noqa: E402
from fomhd.stage import initialize  # noqa: E402
from fomhd.time import advance  # noqa: E402
from fomhd.types import Context, Failure  # noqa: E402


SCOPE = (
    "Unforced small-amplitude parallel longitudinal mode of the finite-mass "
    "two-fluid/CGL equations, using a test-only periodic spectral adapter. "
    "Production stage, time transaction and hard diagnostics are unchanged. "
    "Not closed-wall spatial accuracy, a full stability region, Tokamak/ICE "
    "validation, a validation of physical CGL ordering, or a performance comparison."
)
POLYNOMIAL_TOLERANCE = 1e-5


def polynomial(integrator, z):
    result = 1 + z + z*z/2
    if integrator == "rk4":
        result += z**3/6 + z**4/24
    return result


def build_context(device="cpu"):
    """Return device-owned operators and an independent SI continuum mode."""
    backend = Backend(require_device(device))
    p = parameters(0)
    p.electron_inertia = 1
    p.Nphi = 1
    p.inner_distance = 0.0
    points, k, density, pressure, magnetic = 9, 1.0, 1e12, 1e-9, 1e-5
    geometry = _periodic_geometry(points, 2*math.pi/k)
    op = _periodic_operators(geometry, points, k, backend)
    xp = backend.xp
    gauge = xp.full((points, points), k*k/points, dtype=xp.complex128)
    op.Lphi = [op.G[0].T.conj() @ op.G[0] + gauge]
    op.selected_constraint_rows = backend.independent_rows(op.Bc_full, p.rank_rtol)
    op.Bc = backend.asarray(op.Bc_full[op.selected_constraint_rows, :])
    op.topology_flux_row[:] = 0
    op.topology_flux_row[0, 2*points:] = 1/points
    context = Context(p, geometry, op, backend, 1)

    mi, me = [s.mass for s in p.species]
    q = p.species[0].charge
    wi2 = density*q*q/(p.epsilon0*mi)
    we2 = density*q*q/(p.epsilon0*me)
    a = wi2 + k*k*(5/3)*pressure/(mi*density)
    b = we2 + k*k*3*pressure/(me*density)
    omega2 = (a+b+math.sqrt((a-b)**2+4*wi2*we2))/2
    omega = math.sqrt(omega2)
    speed = math.sqrt(we2)/k
    ion_density = wi2/(a-omega2)
    vector = np.array([ion_density, -omega*ion_density/(k*speed),
                       (5/3)*ion_density, 1., -omega/(k*speed), 1., 3.], complex)
    wave = backend.asarray(np.exp(2j*math.pi*np.arange(points)/points))
    defects = {
        "plane_wave": backend.scalar(xp.max(xp.abs(op.DZ @ wave-1j*k*wave)))/k,
        "constant_derivative": backend.scalar(xp.max(xp.abs(op.DZ.sum(axis=1))))/k,
        "CG": backend.scalar(xp.max(xp.abs(op.C[0] @ op.G[0])))/(k*k),
        "DC": backend.scalar(xp.max(xp.abs(op.DB[0] @ op.C[0])))/(k*k),
        "mean_gauge": backend.scalar(xp.max(xp.abs(op.Lphi[0].sum(axis=1)-k*k)))/(k*k),
        "derivative_skew": backend.scalar(xp.max(xp.abs(op.DZ+op.DZ.T.conj())))/k,
        "current_adjoint": backend.scalar(xp.max(xp.abs(op.DJ[0]+op.G[0].T.conj())))/k,
        "curl_adjoint": backend.scalar(xp.max(xp.abs(op.Cdag[0]-op.C[0].T.conj())))/k,
    }
    maximum = max(float(value) for value in defects.values())
    if maximum > 1e-12 or len(op.selected_constraint_rows) != points-1:
        raise Failure("Fixture", "periodic finite adapter identities failed")
    return context, dict(density=density, pressure=pressure, magnetic=magnetic,
                         k=k, speed=speed, omega=omega, vector=vector, wave=wave,
                         adapter_defects=defects, adapter_max_defect=maximum)


def project(state, context, mode, *, derivative=False):
    """Reduce seven mode coefficients on-device; transfer only seven scalars."""
    n = context.op.Nc
    mi, me = [s.mass for s in context.p.species]
    density, pressure, speed = mode["density"], mode["pressure"], mode["speed"]
    fields = [state.fluid[0].n/density,
              state.fluid[0].mom[2*n:]/(mi*density*speed),
              state.fluid[0].perp/pressure, state.fluid[1].n/density,
              state.fluid[1].mom[2*n:]/(me*density*speed),
              state.fluid[1].perp/pressure, state.fluid[1].parallel/pressure]
    xp = context.compute.xp
    coefficients = []
    for index, field in enumerate(fields):
        if not derivative and index in (0, 2, 3, 5, 6):
            field = field-1
        coefficients.append(context.compute.scalar(2*xp.mean(field*mode["wave"].conj())))
    return np.asarray(coefficients, dtype=np.complex128)


def run_case(integrator, steps, *, cycles=2, amplitude=1e-7, device="cpu"):
    """Run one bounded local control and retain complete real diagnostics."""
    if integrator not in ("midpoint", "rk4"):
        raise ValueError("integrator must be midpoint or rk4")
    if not isinstance(steps, int) or isinstance(steps, bool) or not 4 <= steps <= 4096:
        raise ValueError("steps must be an integer in [4, 4096]")
    if not math.isfinite(cycles) or not 1 <= cycles <= 4:
        raise ValueError("cycles must be in [1, 4]")
    if not math.isfinite(amplitude) or not 0 < amplitude <= 1e-6:
        raise ValueError("amplitude must be in (0, 1e-6]")
    context, mode = build_context(device)
    p, backend = context.p, context.compute
    p.integrator, p.nsteps, p.dt_guard = integrator, steps, "warn"
    horizon = cycles*2*math.pi/mode["omega"]
    p.dt = horizon/steps
    factor = polynomial(integrator, 1j*mode["omega"]*p.dt)
    if amplitude*abs(factor)**steps > 1e-3:
        raise ValueError("predicted amplification leaves the bounded linear control")
    state = allocate_state(context)
    state.generation = 1
    n = context.op.Nc
    xp = backend.xp
    for species, offset in ((0, 0), (1, 3)):
        v = mode["vector"]
        wave = mode["wave"]
        state.fluid[species].n[:] = mode["density"]*(1+amplitude*xp.real(v[offset]*wave))
        state.fluid[species].mom[2*n:] = (
            p.species[species].mass*mode["density"]*mode["speed"]
            *amplitude*xp.real(v[offset+1]*wave))
        state.fluid[species].perp[:] = mode["pressure"]*(1+amplitude*xp.real(v[offset+2]*wave))
        if species:
            state.fluid[species].parallel[:] = mode["pressure"]*(1+amplitude*xp.real(v[6]*wave))
    state.B[2*n:] = mode["magnetic"]
    neutral_defect = float(abs(backend.scalar(xp.mean(state.fluid[0].n-state.fluid[1].n)))/mode["density"])
    if neutral_defect > 32*np.finfo(float).eps:
        raise Failure("Fixture", "periodic initial total charge is nonzero")
    ledger, captured = cumulative(), {}
    committed = initialize(state, context, ledger, False, captured)
    initial = project(committed.state, context, mode)
    prescribed = amplitude*mode["vector"]
    initial_error = np.linalg.norm(initial-prescribed)/np.linalg.norm(prescribed)
    rhs = project(captured["rhs"], context, mode, derivative=True)
    rhs_reference = 1j*mode["omega"]*prescribed
    rhs_error = np.linalg.norm(rhs-rhs_reference)/np.linalg.norm(rhs_reference)
    records = list(captured["checks"])
    initial_hard = sum(record.hard for record in records)
    history, times, reports = [initial], [committed.state.time], []
    failure = None
    started = time.monotonic()
    for _ in range(steps):
        try:
            result = advance(committed, ledger, context, None)
        except Failure as error:
            records.extend(error.checks)
            rejected_report = getattr(error, "time_step_report", None)
            if rejected_report is not None:
                reports.append(rejected_report)
            failure = error
            break
        records.extend(result.checks)
        reports.append(result.time_step_report)
        history.append(project(committed.state, context, mode))
        times.append(committed.state.time)
    backend.synchronize()
    actual = history[-1]
    accepted = committed.state.step
    expected = prescribed*factor**accepted
    continuum = prescribed*np.exp(1j*mode["omega"]*committed.state.time)
    polynomial_error = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    hard = [record for record in records if record.hard]
    hard_pass = bool(hard) and all(record.available and record.passed for record in hard)
    projection = np.vdot(prescribed, actual)/np.vdot(prescribed, prescribed).real
    passed = (failure is None and accepted == steps and hard_pass
              and initial_error < 1e-7 and rhs_error < POLYNOMIAL_TOLERANCE
              and polynomial_error < POLYNOMIAL_TOLERANCE)
    summary = dict(integrator=integrator, device=device, steps=steps, accepted_steps=accepted,
                   dt_s=p.dt, target_time_s=horizon, accepted_time_s=committed.state.time,
                   omega_rad_s=mode["omega"], frequency_Hz=mode["omega"]/(2*math.pi),
                   relative_amplitude=amplitude, adapter_max_defect=mode["adapter_max_defect"],
                   initial_neutrality_relative_defect=neutral_defect,
                   initial_mode_relative_error=float(initial_error), rhs_mode_relative_error=float(rhs_error),
                   polynomial_relative_error=polynomial_error, polynomial_tolerance=POLYNOMIAL_TOLERANCE,
                   continuum_relative_error=float(np.linalg.norm(actual-continuum)/np.linalg.norm(continuum)),
                   amplitude_ratio=float(abs(projection)), polynomial_amplitude_ratio=float(abs(factor)**accepted),
                   wrapped_continuum_phase_error_rad=float(np.angle(projection*np.exp(-1j*mode["omega"]*committed.state.time))),
                   initial_hard_records=initial_hard, hard_records=len(hard), hard_pass=hard_pass,
                   dt_warnings=sum(not report["accepted"] for report in reports),
                   evolution_wall_seconds=time.monotonic()-started, passed=bool(passed),
                   failure=None if failure is None else dict(code=failure.code, message=str(failure)))
    return dict(summary=summary, history=np.asarray(history), times=np.asarray(times),
                diagnostics=records, time_step_reports=reports, failure=failure)


def run_scan(output, *, device="cpu", steps=(4, 8, 32, 64, 128), cycles=2):
    """Write a fresh, fully attributed scan including half-amplitude controls."""
    require_device(device)
    steps = list(steps)
    if not steps or any(not isinstance(count, int) or isinstance(count, bool)
                        or not 4 <= count <= 4096 for count in steps):
        raise ValueError("steps must be a nonempty list of integers in [4, 4096]")
    if len(set(steps)) != len(steps):
        raise ValueError("step counts must be unique")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources = list((ROOT/"fomhd").glob("*.py")) + [Path(__file__).resolve()]
    sources += [ROOT/name for name in (
        "tests/warp/foundation_fixture.py", "tests/warp/workflow_fixture.py",
        "tests/warp/massless_verification_fixture.py", "tests/warp/mms_fixture.py",
        "pyproject.toml", "uv.lock")]
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = output/"source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    provenance = dict(source_hashes=hashes, device=device,
                      slurm_job_id=os.environ.get("SLURM_JOB_ID"), command=sys.argv)
    (output/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    (output/"SCOPE.md").write_text(SCOPE+"\n")
    results = []
    for integrator in ("midpoint", "rk4"):
        for count, amplitude in [(count, 1e-7) for count in steps]+[(min(steps), 5e-8)]:
            name = f"{integrator}-n{count}-a{amplitude:.0e}"
            try:
                result = run_case(integrator, count, cycles=cycles, amplitude=amplitude, device=device)
            except Failure as error:
                write_failure_arrays(output, error, stem=name+"-initial-failure")
                write_failure_checks(output, error, stem=name+"-initial-failure")
                (output/(name+"-failure.json")).write_text(json.dumps(dict(code=error.code, message=str(error)), indent=2)+"\n")
                raise
            summary = result["summary"]
            summary["case_id"] = name
            results.append(summary)
            np.savez_compressed(output/(name+".npz"), time_s=result["times"], mode=result["history"])
            with gzip.open(output/(name+"-checks.json.gz"), "wt", encoding="utf-8") as stream:
                json.dump([serialize_check_record(record) for record in result["diagnostics"]], stream, allow_nan=False)
            (output/(name+"-dt.json")).write_text(json.dumps(result["time_step_reports"], indent=2, allow_nan=False)+"\n")
            if result["failure"] is not None:
                write_failure_arrays(output, result["failure"], stem=name+"-failure")
            report = dict(scope=SCOPE, status="running", cases=results)
            (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            print(json.dumps(summary), flush=True)
    sensitivity = []
    for integrator in ("midpoint", "rk4"):
        base, half = [row for row in results if row["integrator"] == integrator and row["steps"] == min(steps)]
        error = abs(base["amplitude_ratio"]-half["amplitude_ratio"])/base["amplitude_ratio"]
        sensitivity.append(dict(integrator=integrator, relative_amplitude_change=error, passed=error < 1e-5))
    report.update(status="completed", amplitude_sensitivity=sensitivity,
                  passed=all(row["passed"] for row in results) and all(row["passed"] for row in sensitivity))
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cycles", type=float, default=2.)
    parser.add_argument("--steps", default="4,8,32,64,128")
    args = parser.parse_args()
    steps = [int(value) for value in args.steps.split(",")]
    if not steps or len(set(steps)) != len(steps):
        parser.error("steps must be a nonempty unique list")
    report = run_scan(args.output, device=args.device, steps=steps, cycles=args.cycles)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
