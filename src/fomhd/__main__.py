"""A0 command line entry point: check, run, restart and inspect."""

from __future__ import annotations

import inspect
import os
import signal
import sys
import time
import traceback
from types import SimpleNamespace

from . import io, time as time_module
from .stage import initialize
from .types import Context, Failure


USAGE = """Usage: python -m fomhd check CASE [--output DIR]
       python -m fomhd validate CASE [--output DIR]
       python -m fomhd run CASE [--output DIR] [--restart CHECKPOINT]
       python -m fomhd restart CHECKPOINT CASE [--output DIR]
       python -m fomhd inspect RUN_DIR
       python -m fomhd --help | --version
"""
VERSION = "FOMHD 0.1.0 A0 Python/Warp FP64 complex128"
STOP = False


def _stop(signum, frame):
    global STOP
    STOP = True


def _identity(error, committed=None):
    if committed is not None:
        return committed.state.step
    return None


def _context(case):
    if getattr(case.parameters, "electron_inertia", 0) not in (0, 1):
        raise Failure("Config", "electron_inertia must be 0 or 1")
    backend = case.compute
    return Context(
        case.parameters,
        case.geometry,
        case.operators,
        backend,
        case.parameters.Nphi,
    )


def _empty_cumulative(case):
    return SimpleNamespace(
        baseline=None,
        integrals=[
            SimpleNamespace(
                electric_node=0.0,
                electric_edge=0.0,
                pressure_kinetic=0.0,
                pressure_internal=0.0,
                magnetic=0.0,
                wall=0.0,
                convection=0.0,
                perp_rate=0.0,
                parallel_rate=0.0,
                spectral_kinetic=0.0,
                spectral_perp=0.0,
                spectral_parallel=0.0,
            )
            for _ in range(2)
        ],
        boundary_integrals=SimpleNamespace(
            number=[0.0, 0.0],
            kinetic=[0.0, 0.0],
            internal_perp=[0.0, 0.0],
            internal_parallel=[0.0, 0.0],
            field_energy=0.0,
            total_energy=0.0,
        ),
        accepted_intervals=0,
        last_trial=0,
        last_step=0,
        last_time=case.parameters.start_time,
    )


def _initialize_once(state, context, cumulative, equilibrium):
    captured_map = {}
    if len(inspect.signature(initialize).parameters) >= 5:
        value = initialize(state, context, cumulative, equilibrium, captured_map)
    else:
        value = initialize(state, context, cumulative, equilibrium)
    captured = SimpleNamespace(**captured_map)
    if isinstance(value, tuple):
        if len(value) == 2:
            return value[0], value[1], captured
        if len(value) == 3:
            return value
    return value, cumulative, captured


def _write_initial(run, committed, captured, elapsed):
    frame = captured
    if frame is None:
        if hasattr(committed, "frame"):
            frame = committed.frame
        else:
            raise Failure(
                "DiagnosticFrame",
                "initialize did not return the accepted diagnostic frame",
            )
    run.diagnostics(
        committed.state,
        frame.checks,
        frame.totals,
        frame.momentum,
        elapsed,
    )
    run.snapshot(committed, frame)
    run.checkpoint(committed, committed.cumulative, committed.context)


def _normalise_committed(value, context, cumulative):
    committed = value
    if not hasattr(committed, "state"):
        raise Failure("Time", "initialize did not return a committed state")
    if not hasattr(committed, "accepted"):
        committed.accepted = True
    committed.context = context
    committed.cumulative = cumulative
    return committed


def _parse_check_or_run(command, args):
    if not args:
        raise ValueError(USAGE)
    case = args[0]
    output = None
    restart = None
    seen = set()
    index = 1
    while index < len(args):
        option = args[index]
        if option in seen or index + 1 >= len(args):
            raise ValueError("duplicate or incomplete option")
        seen.add(option)
        if option == "--output":
            output = args[index + 1]
        elif option == "--restart" and command == "run":
            restart = args[index + 1]
        else:
            raise ValueError(f"unknown option: {option}")
        index += 2
    return case, output, restart


def _parse_restart(args):
    if len(args) < 2:
        raise ValueError(USAGE)
    checkpoint, case = args[0], args[1]
    output = None
    if len(args) == 4 and args[2] == "--output":
        output = args[3]
    elif len(args) != 2:
        raise ValueError("unknown or incomplete restart option")
    return checkpoint, case, output


def _run_command(argv):
    global STOP
    if argv in (["--help"], ["-h"]):
        print(USAGE, end="")
        return 0
    if argv == ["--version"]:
        print(VERSION)
        return 0
    if not argv:
        raise ValueError(USAGE)
    command = argv[0]
    if command == "inspect":
        if len(argv) != 2:
            raise ValueError(USAGE)
        print(
            io.json.dumps(
                io.inspect(argv[1]),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    if command not in ("check", "validate", "run", "restart"):
        raise ValueError(USAGE)
    restart_path = None
    if command == "restart":
        restart_path, case_path, output = _parse_restart(argv[1:])
        command = "run"
    else:
        case_path, output, restart_path = _parse_check_or_run(
            command, argv[1:]
        )

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    STOP = False
    phase = "pre_commit_input"
    exit_code = 2
    run = None
    committed = None
    accepted = None
    restart_parent = None
    start = time.monotonic()
    try:
        overrides = {"output": output} if output is not None else {}
        case = io.load_case(case_path, overrides)
        exit_code = 3
        context = _context(case)
        exit_code = 5
        run = io.Run(case)
        cumulative = _empty_cumulative(case)
        exit_code = 4
        if restart_path:
            saved = io.read_restart(restart_path, case)
            context.next_trial = saved.trial + 1
            context.next_source = saved.source + 1
            restart_parent = saved.parent
            state = saved.state
            validation = _empty_cumulative(case)
            committed, _discarded, captured = _initialize_once(
                state, context, validation, False
            )
            cumulative = saved.cumulative
        else:
            state = io.load_initial(case)
            committed, cumulative, captured = _initialize_once(
                state,
                context,
                cumulative,
                case.initial_kind == "equilibrium",
            )
        committed = _normalise_committed(committed, context, cumulative)
        accepted = committed.state.step
        phase = "post_commit_output"
        exit_code = 5
        if restart_parent is not None:
            run.parent(restart_parent)
        try:
            _write_initial(
                run, committed, captured, time.monotonic() - start
            )
        except Exception:
            raise
        if command in ("check", "validate"):
            run.status("validated_initial_state")
            print(f"Validated initial state: {run.path()}")
            return 0
        run.status("running")
        while committed.state.step < case.parameters.nsteps:
            if STOP:
                run.checkpoint(committed, cumulative, context)
                run.status("stopped_at_accepted_boundary")
                return 130
            phase = "numerical_trial"
            exit_code = 4
            result = time_module.advance(committed, cumulative, context)
            accepted = committed.state.step
            phase = "post_commit_output"
            exit_code = 5
            if getattr(result, "time_step_report", None) is not None:
                run.time_estimate(result.time_step_report)
            if getattr(result, "coupling_metadata", None) is not None:
                run.time_coupling_evidence(result.coupling_metadata)
            run.diagnostics(
                committed.state,
                result.checks,
                result.totals,
                result.momentum,
                time.monotonic() - start,
            )
            if (
                committed.state.step % case.snapshot_stride == 0
                or committed.state.step == case.parameters.nsteps
            ):
                frame = None
                if case.residual_arrays:
                    frame = committed.frame
                    if frame is None:
                        raise Failure(
                            "DiagnosticFrame",
                            "residual snapshot requires an accepted stage frame",
                        )
                run.snapshot(committed, frame)
            if (
                committed.state.step % case.checkpoint_stride == 0
                or committed.state.step == case.parameters.nsteps
            ):
                run.checkpoint(committed, cumulative, context)
        context.compute.synchronize()
        run.status("completed")
        print(f"Completed {committed.state.step} steps: {run.path()}")
        return 0
    except Exception as error:
        print(f"FOMHD error: {error}", file=sys.stderr)
        if run is not None:
            try:
                run.failure(error, accepted, phase)
            except Exception as recording_error:
                print(
                    f"Failure recording also failed: {recording_error}",
                    file=sys.stderr,
                )
        elif os.environ.get("FOMHD_TRACEBACK"):
            traceback.print_exc()
        return exit_code
    finally:
        if run is not None:
            run.close()


def main():
    return _run_command(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
