from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest


PROPOSAL_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROPOSAL_ROOT / "tools" / "warp_benchmark.py"
PLOT_SCRIPT = PROPOSAL_ROOT / "tools" / "plots" / "plot_warp_benchmarks.py"
FOMHD_ROOT = Path(
    os.environ.get("FOMHD_REPO_ROOT", Path(__file__).resolve().parents[2])
).resolve()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness():
    return _load("warp_benchmark_under_test", SCRIPT)


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(FOMHD_ROOT / "src"), str(PROPOSAL_ROOT)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["MPLCONFIGDIR"] = "/tmp/fomhd-warp-benchmark-mpl"
    env["OPENBLAS_NUM_THREADS"] = "1"
    return env


def test_tool_source_paths_resolve_executed_files(harness):
    paths = harness._tool_source_paths()
    assert paths["tools/warp_benchmark.py"] == SCRIPT.resolve()
    assert paths["tools/plots/plot_warp_benchmarks.py"] == PLOT_SCRIPT.resolve()
    hashes = harness._source_hashes(FOMHD_ROOT)
    for relative, path in paths.items():
        assert hashes[relative] == hashlib.sha256(path.read_bytes()).hexdigest()
    package_source = FOMHD_ROOT / "src" / "fomhd" / "time.py"
    assert hashes["src/fomhd/time.py"] == hashlib.sha256(
        package_source.read_bytes()
    ).hexdigest()


def test_import_has_no_run_side_effect(harness, tmp_path):
    assert harness.SCHEMA == "fomhd-warp-benchmark-v1"
    assert not list(tmp_path.iterdir())


def test_pilot_cli_default_integrator_is_rk4(harness, tmp_path):
    arguments = harness._parser().parse_args(
        ["pilot", "--output", str(tmp_path / "pilot")]
    )
    assert arguments.integrator == "rk4"


def test_benchmark_provenance_declares_historical_finite_scope(harness):
    provenance = harness._base_provenance("scan", [], PROPOSAL_ROOT)
    assert provenance["model"]["electron_inertia"] == 1
    assert provenance["model"]["scope"].startswith(
        "historical finite-electron"
    )


def test_scan_plan_preserves_equal_final_time(harness):
    plan = harness.build_scan_plan(
        grid=4,
        K=1,
        cycles=2.0,
        nsteps_values=(16, 32, 128),
        integrators=("midpoint", "rk4"),
        frequency_reference={"omega_plasma": 5.64e7},
    )
    assert len(plan.cases) == 6
    for case in plan.cases:
        assert case.dt * case.nsteps == pytest.approx(
            plan.target_time_s, rel=2e-15, abs=0.0
        )
        assert case.target_time_s == plan.target_time_s
    assert {
        item["relative_time_error"] for item in plan.assessments
    } == {0.0}


def test_density_seed_is_hermitian_and_positive(harness):
    envelope = np.linspace(0.2, 1.0, 9, dtype=np.float64)
    positive, negative = harness.hermite_density_pair(
        1.0e12, envelope, 1.0e-6
    )
    np.testing.assert_array_equal(negative, np.conj(positive))
    samples = np.fft.ifft(np.array([negative, np.zeros_like(positive), positive]))
    fractional = np.real(samples[0]) / 1.0e12
    assert (1.0 + fractional).min() > 0.0
    assert fractional.max() <= 1.0e-6 * (1.0 + 1.0e-12)


def test_failure_arrays_preserve_nan_and_last_accepted_state(harness, tmp_path):
    from fomhd.types import Failure

    failure = Failure("ForcedFailure", "forced test failure")
    failure.partial_arrays = {
        "bad_nan_inf": np.array([np.nan, np.inf], dtype=np.float64)
    }

    class Backend:
        def to_host(self, value):
            return np.asarray(value)

    backend = Backend()
    backend.array = lambda value: np.asarray(value)
    state = type(
        "State",
        (),
        {"step": 7, "generation": 3, "time": 2.5e-7},
    )()

    class IO:
        @staticmethod
        def encode_state(state, geometry):
            return {"n_e": np.asarray([1.0 + 0.0j])}

    path = tmp_path / "failed_raw.npz"
    keys = harness._write_failure_arrays(
        path, failure, backend, state, object(), IO
    )
    assert "failure_code" in keys
    assert "last_accepted__step" in keys
    assert "last_accepted__n_e" in keys
    with np.load(path, allow_pickle=False) as archive:
        assert str(archive["failure_message"]) == "forced test failure"
        assert int(archive["last_accepted__step"]) == 7
        assert np.isnan(archive["failure__bad_nan_inf"][0])
        assert np.isinf(archive["failure__bad_nan_inf"][1])


def test_budget_classifier_traverses_actual_solver_failure(harness):
    from fomhd.types import Failure

    try:
        raise harness.BudgetExhausted("wall-clock budget expired")
    except harness.BudgetExhausted as budget:
        wrapped = Failure("Memory", "AE materialization failed")
        wrapped.__cause__ = budget
    root = harness._find_budget_exhaustion(wrapped)
    assert isinstance(root, harness.BudgetExhausted)
    assert harness._exception_chain(wrapped)[-1] is root


def test_wrapped_budget_updates_nested_run_status(tmp_path):
    output = tmp_path / "wrapped-budget"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "scan",
            "--output",
            str(output),
            "--device",
            "cpu",
            "--grid",
            "4",
            "--K",
            "1",
            "--cycles",
            "2",
            "--nsteps",
            "64",
            "--integrators",
            "midpoint",
            "--time-budget-s",
            "1",
        ],
        cwd=FOMHD_ROOT,
        env=_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 124, completed.stdout + completed.stderr
    summary = json.loads((output / "summary.json").read_text())
    case = summary["cases"][0]
    assert summary["status"] == "budget_exhausted"
    assert case["status"] == "budget_exhausted"
    assert case["failure"]["wrapped_budget_exhausted"] is True
    case_dir = output / "cases" / case["case_id"]
    result = json.loads((case_dir / "result.json").read_text())
    assert result["failure"]["wrapped_budget_exhausted"] is True
    assert (case_dir / "failed_raw.npz").is_file()
    assert (case_dir / "rejection_report.json").is_file()
    nested = json.loads((case_dir / "run" / "run.json").read_text())
    assert nested["status"] == "budget_exhausted"
    assert "wall-clock budget of 1 s expired" in nested["failure"]["error"]


def test_source_snapshot_does_not_inherit_parent_git_identity(tmp_path, harness):
    parent = tmp_path / "unrelated-repository"
    parent.mkdir()
    subprocess.run(["git", "init", "--quiet", str(parent)], check=True)
    snapshot = parent / "run" / "source"
    snapshot.mkdir(parents=True)
    (parent / "unrelated.txt").write_text("unrelated worktree data\n")
    identity = harness._git_provenance(snapshot)
    assert identity["commit"] is None
    assert identity["status"] is None


def test_actual_checkout_retains_git_status(tmp_path, harness):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / "solver.py").write_text("# Uncommitted source\n")
    identity = harness._git_provenance(tmp_path)
    assert identity["commit"] is None
    assert "solver.py" in identity["status"]


def test_diagnostic_summary_excludes_positivity_and_zero_threshold(harness):
    from types import SimpleNamespace

    def check(name, raw, threshold, norm, *, available=True, hard=True):
        return SimpleNamespace(
            name=name,
            population="electron",
            component="",
            region="all",
            norm=norm,
            raw=raw,
            threshold=threshold,
            reference=threshold,
            unit="1",
            available=available,
            hard=hard,
            passed=(
                available
                and (
                    raw > threshold
                    if name.startswith("minimum_")
                    else raw <= threshold
                )
            ),
            reason="",
        )

    summary = harness._diagnostic_summary(
        [
            check("minimum_perp", 1.0e-9, 0.0, "value"),
            check("minimum_B", 2.0e-5, 1.0e-12, "value"),
            check("poisson_linear", 2.0, 4.0, "linf"),
            check("gauss", 3.0, 2.0, "weighted_L2"),
            check(
                "darwin_dynamic",
                0.0,
                0.0,
                "linf",
                available=False,
            ),
            check("time_step_estimate", 0.5, 1.0, "value"),
        ]
    )
    assert summary["max_normalised_residual"] == pytest.approx(1.5)
    assert summary["residual_normalisation"]["evaluated_count"] == 2
    assert summary["residual_normalisation"]["excluded_count"] == 4
    assert summary["positivity"]["minimum_perp"] == pytest.approx(1.0e-9)
    assert summary["positivity"]["minimum_B"] == pytest.approx(2.0e-5)
    assert summary["hard_failed_count"] == 1
    assert summary["hard_unavailable_count"] == 1


def test_output_creation_refuses_overwrite(harness, tmp_path):
    output = harness._create_output(tmp_path / "unique")
    assert output.is_dir()
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        harness._create_output(output)


@pytest.mark.parametrize(
    ("kind", "expected_guard"),
    (("pilot", "error"), ("scan", "warn")),
)
def test_rendered_case_uses_dt_guard_only_for_scan(
    harness, kind, expected_guard
):
    template = """case_name = 'template'
[species]
electron_inertia = 1
[grid]
K = 1
NR = 4
NZ = 4
Nphi = 5
[time]
dt = 1e-9
nsteps = 1
initial_kind = 'equilibrium'
[solver]
backend = 'cpu'
device_ordinal = 0
[io]
geometry_file = 'geometry.npz'
initial_file = 'initial.npz'
source_manifest_file = 'source.json'
snapshot_stride = 1
checkpoint_stride = 1
write_residual_arrays = true
[diagnostics]
mode_observations = []
"""
    text = harness._render_case(
        template,
        case_name=f"{kind}-case",
        grid=4,
        K=1,
        dt=1e-9,
        nsteps=1,
        integrator="midpoint",
        dt_guard=expected_guard,
        backend="cpu",
        device_ordinal=0,
        geometry_path="geometry.npz",
        initial_path="initial.npz",
        source_path="source.json",
    )
    assert tomllib.loads(text)["time"]["dt_guard"] == expected_guard


def test_frequency_fit_marks_short_records_unavailable(harness):
    result, arrays = harness._frequency_analysis(
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 1.0, 1.0], dtype=np.complex128),
        requested_cycles=2.0,
    )
    assert result["status"] == "UNAVAILABLE"
    assert arrays["time_s"].shape == (3,)


def test_device_provenance_uses_nonzero_backend_ordinal(harness):
    from types import SimpleNamespace

    current_device = [0]
    queried = []

    class Device:
        def __init__(self, ordinal):
            self.ordinal = int(ordinal)
            self.previous = None

        def __enter__(self):
            self.previous = current_device[0]
            current_device[0] = self.ordinal
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            current_device[0] = self.previous

    class Cuda:
        runtime = None

        @staticmethod
        def get_current_stream():
            queried.append(("stream", current_device[0], None))
            return SimpleNamespace(ptr=4000 + current_device[0])

    Cuda.Device = Device

    class Runtime:
        @staticmethod
        def getDeviceProperties(ordinal):
            queried.append(("properties", current_device[0], ordinal))
            return {"name": b"mock-cuda-device", "pciBusID": 99}

        @staticmethod
        def memGetInfo():
            queried.append(("memory", current_device[0], None))
            return (1234, 5678)

    Cuda.runtime = Runtime()
    fake_xp = SimpleNamespace(
        __version__="mock-cupy",
        cuda=Cuda(),
    )
    backend = SimpleNamespace(
        name="cuda",
        device="cuda:3",
        xp=fake_xp,
    )
    parameters = SimpleNamespace(device_memory_limit=9876)
    assert not hasattr(parameters, "device_ordinal")
    context = SimpleNamespace(compute=backend, p=parameters)
    core = SimpleNamespace(
        estimated_peak_bytes=lambda _context: 5555
    )
    record = harness._device_provenance(context, core)
    assert queried == [
        ("properties", 3, 3),
        ("memory", 3, None),
        ("stream", 3, None),
    ]
    assert current_device == [0]
    assert record["device_ordinal"] == 3
    assert record["total_memory_bytes"] == 5678
    assert record["free_memory_bytes_at_probe"] == 1234
    assert record["stream"] == 4003


def test_frequency_fit_measures_actual_samples_shape(harness):
    frequency_hz = 2.5e6
    times = np.arange(64, dtype=np.float64) * 2.0e-8
    signal = np.exp(-2j * np.pi * frequency_hz * times)
    result, arrays = harness._frequency_analysis(
        times, signal, requested_cycles=2.0
    )
    assert result["status"] == "MEASURED"
    assert result["propagation_sign"] == "negative"
    assert result["mode_character"] == "directional_complex"
    assert abs(result["signed_frequency_hz"] + frequency_hz) < 2.0e5
    assert result["angular_frequency_rad_s"] == pytest.approx(
        2.0 * math.pi * frequency_hz, abs=2.0 * math.pi * 2.0e5
    )
    assert arrays["spectrum_abs"].shape == (64,)
    assert arrays["frequency_hz"].shape == (64,)


@pytest.mark.parametrize(
    ("phase", "signal_kind"),
    ((0.0, "cosine"), (math.pi / 5.0, "phase_shifted_sine")),
)
def test_frequency_measures_real_standing_wave_without_sign(
    harness, phase, signal_kind
):
    frequency_hz = 2.5e6
    samples_per_period = 16
    times = np.arange(256, dtype=np.float64) / (
        samples_per_period * frequency_hz
    )
    signal = np.cos(2.0 * np.pi * frequency_hz * times + phase)
    result, arrays = harness._frequency_analysis(
        times, signal.astype(np.complex128), requested_cycles=2.0
    )
    assert result["status"] == "MEASURED"
    assert result["mode_character"] == "real_standing_wave"
    assert result["propagation_sign"] == "undetermined"
    assert result["signed_frequency_hz"] is None
    assert result["angular_frequency_rad_s"] == pytest.approx(
        2.0 * math.pi * frequency_hz,
        abs=2.0 * math.pi * 2.0 * result["resolution_hz"],
    )
    assert result["samples_per_observed_cycle"] == pytest.approx(
        16.0, abs=0.1
    )
    assert arrays["spectrum_abs"].shape == (256,)
    assert signal_kind in ("cosine", "phase_shifted_sine")


def test_frequency_measures_complex_rotating_mode_with_known_sign(harness):
    frequency_hz = 2.5e6
    samples_per_period = 16
    times = np.arange(256, dtype=np.float64) / (
        samples_per_period * frequency_hz
    )
    signal = np.exp(2j * np.pi * frequency_hz * times)
    result, _arrays = harness._frequency_analysis(
        times, signal, requested_cycles=2.0
    )
    assert result["status"] == "MEASURED"
    assert result["mode_character"] == "directional_complex"
    assert result["propagation_sign"] == "positive"
    assert result["signed_frequency_hz"] > 0.0
    assert abs(result["signed_frequency_hz"] - frequency_hz) < 1.0e3


def test_frequency_rejects_monotonic_growth(harness):
    times = np.arange(64, dtype=np.float64) * 2.0e-8
    signal = np.exp(times / times[-1] * 4.0).astype(np.complex128)
    result, _arrays = harness._frequency_analysis(
        times, signal, requested_cycles=2.0
    )
    assert result["status"] == "UNAVAILABLE"
    assert "cycle" in result["reason"] or "coherent" in result["reason"]


def test_frequency_rejects_aperiodic_failure_tail(harness):
    times = np.arange(64, dtype=np.float64) * 2.0e-8
    frequency_hz = 2.5e6
    signal = np.exp(-2j * np.pi * frequency_hz * times)
    signal[44:] = np.exp(
        2j * np.pi * 8.0e6 * (times[44:] - times[44])
    )
    result, _arrays = harness._frequency_analysis(
        times, signal, requested_cycles=2.0
    )
    assert result["status"] == "UNAVAILABLE"


def test_all_failed_plot_does_not_require_successful_y_values(tmp_path):
    run = tmp_path / "failed-run"
    run.mkdir()
    summary = {
        "status": "completed",
        "cases": [
            {
                "case_id": "failed-case",
                "kind": "scan",
                "status": "failed",
                "integrator": "midpoint",
                "dt_s": 1.0e-7,
                "nsteps_requested": 64,
                "max_normalised_residual": None,
                "timing": {"steady_median_s": None},
                "frequency": {
                    "status": "UNAVAILABLE",
                    "reason": "no resolved oscillation",
                },
            }
        ],
    }
    benchmark = {
        "model": {
            "variant": "finite_mass",
            "precision": "float64/complex128",
        },
        "requested_device": "cpu",
        "time": {"omega_plasma_rad_s": 5.64e7},
    }
    (run / "summary.json").write_text(json.dumps(summary))
    (run / "benchmark.json").write_text(json.dumps(benchmark))
    completed = subprocess.run(
        [sys.executable, str(PLOT_SCRIPT), str(run)],
        cwd=FOMHD_ROOT,
        env=_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (run / "plots" / "warp_benchmark_summary.png").stat().st_size > 0


def test_real_cpu_smoke_and_plot(tmp_path):
    output = tmp_path / "cpu-smoke"
    command = [
        sys.executable,
        str(SCRIPT),
        "scan",
        "--output",
        str(output),
        "--device",
        "cpu",
        "--grid",
        "4",
        "--K",
        "1",
        "--cycles",
        "2",
        "--nsteps",
        "2",
        "--integrators",
        "midpoint",
        "--time-budget-s",
        "180",
    ]
    completed = subprocess.run(
        command,
        cwd=FOMHD_ROOT,
        env=_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=210,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    with (output / "summary.json").open() as handle:
        summary = json.load(handle)
    with (output / "benchmark.json").open() as handle:
        benchmark = json.load(handle)
    assert summary["status"] == "completed"
    assert summary["requested_device"] == "cpu"
    assert len(summary["cases"]) == 1
    case = summary["cases"][0]
    assert case["status"] == "accepted"
    assert case["accepted_step"] == 2
    assert case["final_time_relative_error"] <= 64.0 * np.finfo(float).eps
    assert case["device"]["backend_name"] == "cpu"
    assert case["device"]["fallback_used"] is False
    assert case["device"]["total_memory_bytes"] is None
    assert case["frequency"]["status"] == "UNAVAILABLE"
    assert case["timing"]["first_step_s"] > 0.0
    for relative, delivered in (
        ("tools/warp_benchmark.py", SCRIPT),
        ("tools/plots/plot_warp_benchmarks.py", PLOT_SCRIPT),
    ):
        expected = hashlib.sha256(delivered.read_bytes()).hexdigest()
        assert benchmark["source_hashes"][relative] == expected
        archived = output / "provenance" / relative
        assert archived.read_bytes() == delivered.read_bytes()
        assert hashlib.sha256(archived.read_bytes()).hexdigest() == expected
    case_dir = output / "cases" / case["case_id"]
    assert (case_dir / "initial_checks.json").is_file()
    assert (case_dir / "final_checks.json").is_file()
    assert (case_dir / "states.npz").is_file()
    assert (case_dir / "run" / "checkpoints" / "latest.json").is_file()
    assert (case_dir / "time_step_reports.json").is_file()
    assert (case_dir / "run" / "time_estimates.jsonl").is_file()
    from tools.analyze import load_run
    loaded = load_run(case_dir / "run")
    assert loaded.source_files["src/fomhd/time.py"]
    assert loaded.config_hash_verification["status"] == "PASS"
    runtime_guard = json.loads(
        (case_dir / "time_step_reports.json").read_text()
    )
    assert runtime_guard["status"] == "AVAILABLE"
    assert len(runtime_guard["reports"]) == 2
    assert all("estimate" in report for report in runtime_guard["reports"])
    assert all("action" in report for report in runtime_guard["reports"])
    with np.load(output / "observations.npz", allow_pickle=False) as archive:
        prefix = case["case_id"]
        times = archive[f"{prefix}.time_s"]
        coefficient = archive[f"{prefix}.ne_hat_real"] + 1j * archive[
            f"{prefix}.ne_hat_imag"
        ]
        assert times.shape == (3,)
        assert coefficient.shape == (3,)
        assert np.isfinite(times).all()
        assert np.isfinite(coefficient).all()
        assert archive[f"{prefix}.hard_failure_count"].sum() == 0
        assert f"{prefix}.frequency_spectrum_abs" in archive.files
        assert f"{prefix}.frequency_frequency_hz" in archive.files
    completed_plot = subprocess.run(
        [
            sys.executable,
            str(PLOT_SCRIPT),
            str(output),
        ],
        cwd=FOMHD_ROOT,
        env=_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
        check=False,
    )
    assert completed_plot.returncode == 0, (
        completed_plot.stdout + completed_plot.stderr
    )
    assert (output / "plots" / "warp_benchmark_summary.png").stat().st_size > 0
    assert (output / "plots" / "warp_benchmark_summary.pdf").stat().st_size > 0
    plot_data = json.loads((output / "plots" / "plot_data.json").read_text())
    assert plot_data["scan"][0]["status"] == "accepted"
    assert plot_data["rules"]
