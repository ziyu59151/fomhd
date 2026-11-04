"""Exercise failure receipts, safe-stop checkpointing, and Slurm receipts."""

import argparse
import json
import os
import pathlib
import re
import shutil
import signal
import stat
import subprocess
import time


def wait_until(predicate, description, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {description}")


def replace_number(text, key, value):
    result, count = re.subn(
        rf"(?m)^(\s*{re.escape(key)}\s*=\s*)\d+\s*$", rf"\g<1>{value}", text
    )
    assert count == 1, f"could not set {key}"
    return result


def replace_bool(text, key, value):
    result, count = re.subn(
        rf"(?m)^(\s*{re.escape(key)}\s*=\s*)(?:true|false)\s*$",
        rf"\g<1>{str(value).lower()}",
        text,
    )
    assert count == 1, f"could not set {key}"
    return result


def replace_in_section(text, section, key, value):
    pattern = rf"(?ms)(^\[{re.escape(section)}\]\n)(.*?)(?=^\[)"

    def update(match):
        body, count = re.subn(
            rf"(?m)^({re.escape(key)}\s*=\s*)[^\n]+",
            rf"\g<1>{value}",
            match.group(2),
        )
        assert count == 1, f"could not set {section}.{key}"
        return match.group(1) + body

    result, count = re.subn(pattern, update, text)
    assert count == 1, f"could not find {section}"
    return result


def latest_failure(run):
    records = sorted((run / "failure").glob("failure_*.json"))
    assert len(records) == 1, f"expected one failure receipt, found {len(records)}"
    return json.loads(records[0].read_text())


def run_status(pid):
    try:
        status = pathlib.Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return None
    match = re.search(r"^State:\s+(\S+)", status, re.MULTILINE)
    return match.group(1) if match else None


parser = argparse.ArgumentParser()
parser.add_argument("--build", type=pathlib.Path, required=True)
parser.add_argument("--slurm", type=pathlib.Path, required=True)
parser.add_argument(
    "--only",
    choices=(
        "all",
        "precommit",
        "numerical",
        "postcommit",
        "checkpoint_failure",
        "initial_signal",
        "restart_parent",
        "safe_stop",
        "slurm",
    ),
    default="all",
)
args = parser.parse_args()
build = args.build.resolve()
slurm = args.slurm.resolve()
testing = build / "testing"
testing.mkdir(exist_ok=True)
work = testing / f"runtime-failures-{time.time_ns()}-{args.only}"
assert not work.exists(), f"evidence directory already exists: {work}"
work.mkdir()
(work / "commands.log").write_text("")
print(f"Evidence: {work}", flush=True)


def execute(*command, expected=0, env=None):
    result = subprocess.run(
        [str(value) for value in command],
        text=True,
        capture_output=True,
        env=env,
    )
    with (work / "commands.log").open("a") as log:
        log.write(
            f"{command}\nexit={result.returncode}\n{result.stdout}{result.stderr}\n"
        )
    assert result.returncode == expected, result.stdout + result.stderr
    return result


def make_case(name, steps=4, velocity=0.01, k=1):
    path = work / name
    arguments = [build / "fomhd_make_case", path, steps, velocity]
    if k != 1:
        arguments += [1e-9, k]
    execute(*arguments)
    return path / "case.toml"


def test_precommit():
    case = make_case("precommit-input")
    run = work / "precommit-run"
    result = execute(
        build / "fomhd",
        "run",
        case,
        "--output",
        run,
        "--restart",
        work / "missing-restart.npz",
        expected=4,
    )
    assert "cannot open NPZ input" in result.stderr
    record = latest_failure(run)
    assert record["runtime_phase"] == "pre_commit_input"
    assert record["failure_class"] == "pre_commit_or_input"
    assert record["accepted_trial"] is False
    assert record["last_accepted_step"] is None
    assert record["last_durable_checkpoint"] is None
    assert json.loads((run / "run.json").read_text())["status"] == "failed_before_commit"


def test_postcommit():
    case = make_case("postcommit-output", steps=100)
    text = case.read_text()
    text = replace_number(text, "snapshot_stride", 100)
    text = replace_number(text, "checkpoint_stride", 1)
    text = replace_bool(text, "write_residual_arrays", False)
    case.write_text(text)
    run = work / "postcommit-run"
    stdout_path = work / "postcommit.stdout"
    stderr_path = work / "postcommit.stderr"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            [str(build / "fomhd"), "run", str(case), "--output", str(run)],
            text=True,
            stdout=stdout,
            stderr=stderr,
        )
        snapshots = run / "snapshots"

        def started():
            try:
                return json.loads((run / "run.json").read_text())["status"] == "running"
            except (FileNotFoundError, json.JSONDecodeError):
                return False

        wait_until(started, "run loop")
        os.kill(process.pid, signal.SIGSTOP)
        wait_until(lambda: run_status(process.pid) == "T", "process stop")
        snapshots.chmod(0o555)
        os.kill(process.pid, signal.SIGCONT)
        try:
            assert process.wait(timeout=60) == 5
        finally:
            snapshots.chmod(0o755)
            if process.poll() is None:
                process.kill()
                process.wait()
    record = latest_failure(run)
    assert record["runtime_phase"] == "post_commit_output"
    assert record["failure_class"] == "output_after_commit"
    assert record["accepted_trial"] is True
    last_accepted = record["last_accepted_step"]
    assert last_accepted == 100
    assert record["last_durable_checkpoint"]["step"] == last_accepted - 1
    assert json.loads((run / "run.json").read_text())["status"] == "io_failed_after_commit"


def test_numerical():
    case = make_case("numerical-rejection", steps=100)
    text = case.read_text()
    text = replace_in_section(
        text, "budgets.checks.'CT_update.linf'", "rtol", "0.0"
    )
    case.write_text(text)
    run = work / "numerical-run"
    result = execute(build / "fomhd", "run", case, "--output", run, expected=4)
    assert "hard check CT_update" in result.stderr
    record = latest_failure(run)
    assert record["runtime_phase"] == "numerical_trial"
    assert record["failure_class"] == "rejected_numerical_trial"
    assert record["accepted_trial"] is False
    assert record["last_accepted_step"] == 0
    assert record["last_durable_checkpoint"]["step"] == 0
    assert record["stage"]["stage"] == "interval"
    assert json.loads((run / "run.json").read_text())["status"] == "failed_numerical_trial"


def test_checkpoint_failure():
    case = make_case("checkpoint-failure", steps=100)
    text = case.read_text()
    text = replace_number(text, "snapshot_stride", 100)
    text = replace_number(text, "checkpoint_stride", 1)
    text = replace_bool(text, "write_residual_arrays", False)
    case.write_text(text)
    run = work / "checkpoint-failure-run"
    with (work / "checkpoint-failure.stdout").open("w") as stdout, (
        work / "checkpoint-failure.stderr"
    ).open("w") as stderr:
        process = subprocess.Popen(
            [str(build / "fomhd"), "run", str(case), "--output", str(run)],
            text=True,
            stdout=stdout,
            stderr=stderr,
        )

        def started():
            try:
                return json.loads((run / "run.json").read_text())["status"] == "running"
            except (FileNotFoundError, json.JSONDecodeError):
                return False

        wait_until(started, "run loop with initial checkpoint")
        pointer_path = run / "checkpoints/latest.json"
        previous_pointer = json.loads(pointer_path.read_text())
        previous_file = run / "checkpoints" / previous_pointer["file"]
        previous_bytes = previous_file.read_bytes()
        os.kill(process.pid, signal.SIGSTOP)
        wait_until(lambda: run_status(process.pid) == "T", "process stop")
        (run / "checkpoints").chmod(0o555)
        os.kill(process.pid, signal.SIGCONT)
        try:
            assert process.wait(timeout=60) == 5
        finally:
            (run / "checkpoints").chmod(0o755)
            if process.poll() is None:
                process.kill()
                process.wait()
    record = latest_failure(run)
    assert record["runtime_phase"] == "post_commit_output"
    assert record["accepted_trial"] is True
    assert record["last_accepted_step"] == 1
    assert record["last_durable_checkpoint"] == previous_pointer
    assert json.loads(pointer_path.read_text()) == previous_pointer
    assert previous_file.read_bytes() == previous_bytes


def test_initial_signal():
    case = make_case("initial-signal", steps=100, k=2)
    run = work / "initial-signal-run"
    with (work / "initial-signal.stdout").open("w") as stdout, (
        work / "initial-signal.stderr"
    ).open("w") as stderr:
        process = subprocess.Popen(
            [str(build / "fomhd"), "run", str(case), "--output", str(run)],
            text=True,
            stdout=stdout,
            stderr=stderr,
        )
        wait_until(lambda: (run / "run.json").is_file(), "initializing run")
        os.kill(process.pid, signal.SIGSTOP)
        wait_until(lambda: run_status(process.pid) == "T", "initial process stop")
        process.send_signal(signal.SIGTERM)
        os.kill(process.pid, signal.SIGCONT)
        try:
            status_code = process.wait(timeout=60)
            assert status_code == 130, f"initial signal exit={status_code}"
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    manifest = json.loads((run / "run.json").read_text())
    assert manifest["status"] == "stopped_at_accepted_boundary"
    assert manifest["last_accepted_step"] == 0
    assert manifest["durable_checkpoint"]["step"] == 0


def test_restart_parent():
    case = make_case("restart-parent", steps=1, k=2)
    source_run = work / "restart-source-run"
    execute(build / "fomhd", "run", case, "--output", source_run)
    pointer = json.loads((source_run / "checkpoints/latest.json").read_text())
    restart = source_run / "checkpoints" / pointer["file"]
    run = work / "restart-parent-run"
    with (work / "restart-parent.stdout").open("w") as stdout, (
        work / "restart-parent.stderr"
    ).open("w") as stderr:
        process = subprocess.Popen(
            [
                str(build / "fomhd"),
                "run",
                str(case),
                "--output",
                str(run),
                "--restart",
                str(restart),
            ],
            text=True,
            stdout=stdout,
            stderr=stderr,
        )
        run_json = run / "run.json"
        wait_until(run_json.is_file, "restart run initialization")
        os.kill(process.pid, signal.SIGSTOP)
        wait_until(lambda: run_status(process.pid) == "T", "restart process stop")
        saved = run / "run.json.review-saved"
        run_json.rename(saved)
        run_json.mkdir()
        (run_json / "blocker").write_text("block atomic replacement\n")
        os.kill(process.pid, signal.SIGCONT)
        try:
            status_code = process.wait(timeout=60)
            assert status_code == 5, f"restart parent exit={status_code}"
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            shutil.rmtree(run_json)
            saved.rename(run_json)
    record = latest_failure(run)
    assert record["runtime_phase"] == "post_commit_output"
    assert record["failure_class"] == "output_after_commit"
    assert record["accepted_trial"] is True
    assert record["last_accepted_step"] == 1


def test_safe_stop():
    case = make_case("safe-stop", steps=100)
    text = case.read_text()
    text = replace_number(text, "snapshot_stride", 100)
    text = replace_number(text, "checkpoint_stride", 100)
    text = replace_bool(text, "write_residual_arrays", False)
    case.write_text(text)
    run = work / "safe-stop-run"
    stdout_path = work / "safe-stop.stdout"
    stderr_path = work / "safe-stop.stderr"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            [str(build / "fomhd"), "run", str(case), "--output", str(run)],
            text=True,
            stdout=stdout,
            stderr=stderr,
        )
        history = run / "history.csv"

        def accepted_two():
            try:
                return len(history.read_text().splitlines()) >= 3
            except FileNotFoundError:
                return False

        wait_until(accepted_two, "two accepted steps")
        process.send_signal(signal.SIGTERM)
        try:
            assert process.wait(timeout=60) == 130
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    manifest = json.loads((run / "run.json").read_text())
    assert manifest["status"] == "stopped_at_accepted_boundary"
    latest = manifest["durable_checkpoint"]
    assert latest is not None
    assert latest["step"] == manifest["last_accepted_step"]
    pointer = json.loads((run / "checkpoints/latest.json").read_text())
    assert pointer["step"] == manifest["last_accepted_step"]
    assert not list((run / "failure").glob("failure_*.json"))

    current_case = make_case("safe-stop-current-checkpoint", steps=100)
    text = current_case.read_text()
    text = replace_number(text, "snapshot_stride", 100)
    text = replace_number(text, "checkpoint_stride", 1)
    text = replace_bool(text, "write_residual_arrays", False)
    current_case.write_text(text)
    current_run = work / "safe-stop-current-run"
    with (work / "safe-stop-current.stdout").open("w") as stdout, (
        work / "safe-stop-current.stderr"
    ).open("w") as stderr:
        process = subprocess.Popen(
            [
                str(build / "fomhd"),
                "run",
                str(current_case),
                "--output",
                str(current_run),
            ],
            text=True,
            stdout=stdout,
            stderr=stderr,
        )
        current_history = current_run / "history.csv"

        def current_accepted_two():
            try:
                return len(current_history.read_text().splitlines()) >= 3
            except FileNotFoundError:
                return False

        wait_until(current_accepted_two, "two accepted steps with current checkpoint")
        process.send_signal(signal.SIGTERM)
        try:
            assert process.wait(timeout=60) == 130
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    current_manifest = json.loads((current_run / "run.json").read_text())
    assert current_manifest["durable_checkpoint"]["step"] == current_manifest[
        "last_accepted_step"
    ]
    assert not list((current_run / "failure").glob("failure_*.json"))


def test_slurm():
    stubs = work / "stubs"
    stubs.mkdir()
    stub = stubs / "srun"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "${SRUN_STUB_LOG}"\n'
        'exit "${SRUN_STUB_EXIT:-7}"\n'
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    case = make_case("slurm-case")
    output = work / "slurm-output" / "run"
    output.parent.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{stubs}:{env['PATH']}",
            "FOMHD_BIN": str(build / "fomhd"),
            "FOMHD_CASE": str(case),
            "FOMHD_OUTPUT": str(output),
            "SRUN_STUB_LOG": str(work / "srun-calls.log"),
            "SRUN_STUB_EXIT": "7",
        }
    )
    result = execute("bash", slurm, expected=7, env=env)
    log = (output.parent / "run.launch.log").read_text()
    assert "exit_code=7" in log, result.stdout + result.stderr
    assert "program_exit_code=7" in log
    assert "tee_exit_code=0" in log
    assert re.search(r"(?m)^finished_utc=\d{4}-\d\d-\d\dT", log)
    assert "run" in (work / "srun-calls.log").read_text()

    tee = stubs / "tee"
    tee.write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in\n'
        '  *.stdout.log) exit 9 ;;\n'
        '  *) exec /usr/bin/tee "$@" ;;\n'
        "esac\n"
    )
    tee.chmod(tee.stat().st_mode | stat.S_IXUSR)
    tee_output = work / "slurm-tee-failure" / "run"
    tee_output.parent.mkdir()
    env["FOMHD_OUTPUT"] = str(tee_output)
    env["SRUN_STUB_EXIT"] = "0"
    result = execute("bash", slurm, expected=9, env=env)
    tee_log = (tee_output.parent / "run.launch.log").read_text()
    assert "program_exit_code=0" in tee_log, result.stdout + result.stderr
    assert "tee_exit_code=9" in tee_log
    assert "exit_code=9" in tee_log


tests = {
    "precommit": test_precommit,
    "numerical": test_numerical,
    "postcommit": test_postcommit,
    "checkpoint_failure": test_checkpoint_failure,
    "initial_signal": test_initial_signal,
    "restart_parent": test_restart_parent,
    "safe_stop": test_safe_stop,
    "slurm": test_slurm,
}
selected = tests if args.only == "all" else {args.only: tests[args.only]}
receipt = {}
for name, test in selected.items():
    test()
    receipt[name] = "PASS"
    (work / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print("PASS: " + ", ".join(receipt))
