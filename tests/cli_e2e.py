"""Exercise real output and restart paths, retaining evidence under the build tree."""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import tempfile

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--build", type=pathlib.Path, required=True)
a = p.parse_args()
build = a.build.resolve()
(build / "testing").mkdir(exist_ok=True)
work = pathlib.Path(tempfile.mkdtemp(prefix="cli-", dir=build / "testing"))
print(f"Evidence: {work}", flush=True)


def command(*args, expected=0):
    result = subprocess.run([str(x) for x in args], text=True, capture_output=True)
    with (work / "commands.log").open("a") as log:
        log.write(f"{args}\nexit={result.returncode}\n{result.stdout}{result.stderr}\n")
    assert result.returncode == expected, result.stdout + result.stderr
    return result.stdout


command(build / "fomhd_make_case", work / "input")
case = work / "input/case.toml"
original = case.read_text()
command(build / "fomhd", "validate", case, "--output", work / "validate")
command(build / "fomhd", "run", case, "--output", work / "full")
case.write_text(re.sub(r"nsteps\s*=\s*4\b", "nsteps = 2", original))
command(build / "fomhd", "run", case, "--output", work / "first")
latest = json.loads((work / "first/checkpoints/latest.json").read_text())
case.write_text(original)
command(build / "fomhd", "run", case, "--output", work / "second", "--restart",
        work / "first/checkpoints" / latest["file"])


def checkpoint(run):
    directory = work / run / "checkpoints"
    pointer = json.loads((directory / "latest.json").read_text())
    return np.load(directory / pointer["file"], allow_pickle=False)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


with checkpoint("full") as full, checkpoint("second") as split:
    for key in ("n_i", "mom_i", "p_i", "n_e", "mom_e", "p_perp_e", "p_parallel_e",
                "B_R", "B_phi", "B_Z", "baseline_divB", "step", "time"):
        np.testing.assert_array_equal(full[key], split[key], err_msg=key)
    left, right = (json.loads(str(x["cumulative_json"])) for x in (full, split))
    # Trial identifiers may differ after revalidation; physical integrals must not.
    left.pop("last_trial")
    right.pop("last_trial")
    assert left == right, "restart changed the cumulative physical ledger"
for run in ("validate", "full", "first", "second"):
    report = json.loads(command(build / "fomhd", "inspect", work / run))
    assert not report["inspection"]["interrupted_output"]
    assert report["inspection"]["diagnostic_records"] > 0
    records = [json.loads(line) for line in (work / run / "checks.jsonl").read_text().splitlines()]
    assert records
command(build / "fomhd", "run", case, "--output", work / "full", expected=5)

# Exercise durable rejected-stage evidence after the run writer has initialized.
failure_input = work / "failure_input"
command(build / "fomhd_make_case", failure_input)
initial_path = failure_input / "initial.npz"
with np.load(initial_path, allow_pickle=False) as source:
    arrays = {key: source[key] for key in source.files}
pressure = arrays["p_i"]
zero_mode = pressure.shape[0] // 2
active = np.argwhere(pressure[zero_mode] != 0)
assert active.size > 0, "fixture has no active zero-mode pressure cells"
iz, ir = active[0]
pressure[zero_mode, iz, ir] = -1.0
np.savez(initial_path, **arrays)
source_manifest_path = failure_input / "source_manifest.json"
source_manifest = json.loads(source_manifest_path.read_text())
source_manifest["input_hashes"]["initial"] = sha256(initial_path)
source_manifest_path.write_text(json.dumps(source_manifest, indent=2) + "\n")
failure_run = work / "failure"
command(build / "fomhd", "run", failure_input / "case.toml",
        "--output", failure_run, expected=4)
failure_records = sorted((failure_run / "failure").glob("failure_*.json"))
assert len(failure_records) == 1, "failure metadata was not published"
failure_record = json.loads(failure_records[0].read_text())
assert failure_record["runtime_phase"] == "pre_commit_input"
assert failure_record["failure_class"] == "pre_commit_or_input"
assert failure_record["accepted_trial"] is False
assert failure_record["code"] == "Domain"
assert "pressure" in failure_record["error"].lower()
assert failure_record["last_durable_checkpoint"] is None
assert failure_record["stage"]["stage"] == "initial"
assert failure_record["checks"]
assert any(check["status"] == "UNAVAILABLE" for check in failure_record["checks"])
assert failure_record["partial_published"] is True
partial_path = failure_run / "failure" / failure_record["partial_file"]
assert partial_path.is_file()
assert sha256(partial_path) == failure_record["partial_sha256"]
with np.load(partial_path, allow_pickle=False) as partial:
    assert "trial_n_i" in partial.files
    assert "trial_p_i" in partial.files
print("PASS: real CLI initialization, evolution, restart equivalence, inspection, no-clobber")
