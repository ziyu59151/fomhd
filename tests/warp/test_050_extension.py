from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRATCH = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(
    os.environ.get(
        "FOMHD_SOURCE_ROOT",
        str(SCRATCH),
    )
).resolve()


def _tool():
    name = "scratch_warp_050_experiments"
    if name in sys.modules:
        return sys.modules[name]
    path = SCRATCH / "tools" / "warp_050_experiments.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def small_case():
    from tools.warp_tokamak_cae import background, make_context

    context, meta, equilibrium = make_context(
        "circle", 4, 1, "cpu", memory_limit_bytes=16 << 30
    )
    initial, _, _ = background(context, meta, equilibrium)
    return context, meta, initial


def _write_stability_archive(
    directory: Path,
    context,
    initial,
    *,
    mode: int,
    K: int = 1,
    eigenvalues: np.ndarray | None = None,
):
    from tools.warp_tokamak_cae import host, physical_identity

    directory.mkdir()
    if eigenvalues is None:
        eigenvalues = np.asarray(
            [1j * (2.0e6 + mode), 3.0e5 + 2.0e6j], dtype=np.complex128
        )
    arrays = {
        "eigenvalues": eigenvalues,
        "background_B": host(initial.B, context),
        "background_n": host(initial.fluid[0].n, context),
        "background_mom": host(initial.fluid[0].mom, context),
        "background_pi": host(initial.fluid[0].perp, context),
        "background_perp": host(initial.fluid[1].perp, context),
        "background_parallel": host(initial.fluid[1].parallel, context),
    }
    np.savez(directory / "spectrum.npz", **arrays)
    source_hashes = _tool()._source_hashes()
    report = {
        "kind": "circle",
        "nodes": 4,
        "K": K,
        "mode": mode,
        "size": int(eigenvalues.size),
        "spectral_radius_rad_s": float(np.max(np.abs(eigenvalues))),
        "maximum_growth_per_s": float(np.max(eigenvalues.real)),
        "physical_identity": physical_identity(context),
        "source_hashes": source_hashes,
    }
    (directory / "report.json").write_text(json.dumps(report) + "\n")
    return report


def _write_seed_archive(directory: Path, context, initial, *, frequency_hz: float):
    from tools.warp_tokamak_cae import host, physical_identity

    directory.mkdir()
    nc = context.op.Nc
    arrays = {
        "seed_n": np.zeros(nc, dtype=np.complex128),
        "seed_mom": np.zeros(3 * nc, dtype=np.complex128),
        "seed_pi": np.zeros(nc, dtype=np.complex128),
        "seed_perp": np.zeros(nc, dtype=np.complex128),
        "seed_parallel": np.zeros(nc, dtype=np.complex128),
        "seed_B": np.zeros(context.op.Nb, dtype=np.complex128),
    }
    arrays["seed_B"][:] = host(initial.B, context)
    arrays.update(
        {
            f"background_{name}": value
            for name, value in _tool()._baseline_arrays(initial, context).items()
        }
    )
    np.savez(directory / "spectrum.npz", **arrays)
    report = {
        "kind": "circle",
        "nodes": 4,
        "K": 0,
        "mode": 0,
        "physical_identity": physical_identity(context),
        "source_hashes": _tool()._source_hashes(),
        "selected": {
            "index": 0,
            "frequency_hz": frequency_hz,
            "growth_rate_per_s": 0.0,
        },
    }
    (directory / "report.json").write_text(json.dumps(report) + "\n")


def test_parser_accepts_repeatable_stability_and_resume(tmp_path):
    parser = _tool()._parser()
    args = parser.parse_args(
        [
            "--kind",
            "circle",
            "--nodes",
            "12",
            "--K",
            "1",
            "--mode",
            "0",
            "--coupling",
            "waveform",
            "--horizon-s",
            "1e-8",
            "--resume-from",
            str(tmp_path / "previous"),
            "--stability-spectrum-directory",
            str(tmp_path / "sector-0"),
            "--stability-spectrum-directory",
            str(tmp_path / "sector-1"),
            "--output",
            str(tmp_path / "new"),
        ]
    )
    assert args.resume_from == tmp_path / "previous"
    assert args.stability_spectrum_directories == [
        tmp_path / "sector-0",
        tmp_path / "sector-1",
    ]


def test_complete_stability_spectra_read_only_eigenvalues(tmp_path, small_case):
    context, _, initial = small_case
    sector_0 = tmp_path / "sector-0"
    sector_1 = tmp_path / "sector-1"
    report_0 = _write_stability_archive(
        sector_0, context, initial, mode=0
    )
    report_1 = _write_stability_archive(
        sector_1, context, initial, mode=1
    )
    result = _tool().read_stability_spectra(
        [sector_0, sector_1],
        context,
        initial,
        kind="circle",
        nodes=4,
        K=1,
    )
    assert result["complete_sectors"] == [0, 1]
    assert result["eigenpair_arrays_read"] == []
    assert result["rho_max_rad_s"] == max(
        report_0["spectral_radius_rad_s"],
        report_1["spectral_radius_rad_s"],
    )
    assert result["h_cap_s"] == pytest.approx(
        0.5 / (1.5 * result["rho_max_rad_s"]), rel=0, abs=0
    )
    assert {
        item["mode"] for item in result["spectra"]
    } == {0, 1}
    assert all(item["spectrum_sha256"] for item in result["spectra"])

    with pytest.raises(ValueError, match="missing retained sectors: 1"):
        _tool().read_stability_spectra(
            [sector_0],
            context,
            initial,
            kind="circle",
            nodes=4,
            K=1,
        )


def test_stability_radius_must_exactly_match_report(tmp_path, small_case):
    context, _, initial = small_case
    sector = tmp_path / "sector"
    report = _write_stability_archive(sector, context, initial, mode=0)
    report["spectral_radius_rad_s"] = math.nextafter(
        report["spectral_radius_rad_s"], math.inf
    )
    (sector / "report.json").write_text(json.dumps(report) + "\n")
    with pytest.raises(ValueError, match="spectral radius mismatch"):
        _tool().read_stability_spectra(
            [sector],
            context,
            initial,
            kind="circle",
            nodes=4,
            K=1,
        )


def test_stability_source_mismatch_is_recorded_not_excused(
    tmp_path, small_case
):
    context, _, initial = small_case
    sector = tmp_path / "sector"
    report = _write_stability_archive(sector, context, initial, mode=0)
    other = tmp_path / "other"
    _write_stability_archive(other, context, initial, mode=1)
    report["source_hashes"]["operators.py"] = "0" * 64
    (sector / "report.json").write_text(json.dumps(report) + "\n")
    result = _tool().read_stability_spectra(
        [sector, other],
        context,
        initial,
        kind="circle",
        nodes=4,
        K=1,
    )
    assert result["source_hash_mismatch_recorded"] is True
    assert result["spectra"][0]["source_hash_match"] is False


def test_stability_baseline_mismatch_is_rejected(tmp_path, small_case):
    context, _, initial = small_case
    sector = tmp_path / "sector"
    _write_stability_archive(sector, context, initial, mode=0)
    other = tmp_path / "other"
    _write_stability_archive(other, context, initial, mode=1)
    path = sector / "spectrum.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["background_n"][0] += 1.0e9
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="baseline mismatch"):
        _tool().read_stability_spectra(
            [sector, other],
            context,
            initial,
            kind="circle",
            nodes=4,
            K=1,
        )


@pytest.mark.parametrize("nonfinite_source", ["archive", "current"])
def test_stability_background_rejects_nonfinite_identity(
    tmp_path, small_case, nonfinite_source
):
    context, _, initial = small_case
    current = _tool()._clone_state(initial, context)
    sector_0 = tmp_path / "sector-0"
    sector_1 = tmp_path / "sector-1"
    _write_stability_archive(sector_0, context, initial, mode=0)
    _write_stability_archive(sector_1, context, initial, mode=1)

    if nonfinite_source == "archive":
        path = sector_0 / "spectrum.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True) for name in archive.files
            }
        arrays["background_n"][0] = np.nan
        np.savez(path, **arrays)
    else:
        current.fluid[0].n[0] = np.inf

    with pytest.raises(ValueError, match="nonfinite"):
        _tool().read_stability_spectra(
            [sector_0, sector_1],
            context,
            current,
            kind="circle",
            nodes=4,
            K=1,
        )


def test_resolved_spectrum_uses_rho_cap_not_operator_estimate(
    small_case,
):
    context, _, initial = small_case
    rho = 162_000_000.0
    cap = 0.5 / (1.5 * rho)
    controls = {
        "h_requested": "auto",
        "horizon_s": 3.8e-9,
        "micro_steps": 2,
        "coupling": "waveform",
        "target_frequency_hz": None,
    }
    plan, probes = _tool()._resolve_time_plan(
        controls,
        context,
        {"control": initial, "perturbed": initial},
        stability_radius_rad_s=rho,
    )
    assert plan["h_provisional_s"] == pytest.approx(cap, rel=0, abs=0)
    assert plan["h_s"] > 0.5 / (1.5 * probes["omega_estimate"])


def test_explicit_micro_h_above_resolved_cap_is_rejected(small_case):
    context, _, initial = small_case
    rho = 162_000_000.0
    h = (0.5 / (1.5 * rho)) * 1.1
    controls = {
        "h_requested": h,
        "horizon_s": 4 * 2 * h,
        "micro_steps": 2,
        "coupling": "waveform",
        "target_frequency_hz": None,
    }
    with pytest.raises(ValueError, match="stability h cap exceeded"):
        _tool()._resolve_time_plan(
            controls,
            context,
            {"control": initial, "perturbed": initial},
            stability_radius_rad_s=rho,
        )


def test_archive_selected_target_governs_automatic_h(
    tmp_path, small_case
):
    from tools.warp_tokamak_cae import background, make_context

    context, meta, equilibrium = make_context(
        "circle", 4, 0, "cpu", memory_limit_bytes=16 << 30
    )
    initial, _, _ = background(context, meta, equilibrium)
    frequency_hz = 2.0e8
    archive = tmp_path / "seed"
    _write_seed_archive(archive, context, initial, frequency_hz=frequency_hz)
    output = tmp_path / "output"

    assert (
        _tool().run(
            _run_args(
                output,
                probe=1,
                h="auto",
                horizon_s=1.0e-9,
                spectrum_directory=archive,
            ),
            context_factory=_small_context,
        )
        == 2
    )
    report = json.loads((output / "report.json").read_text())
    target_cap_s = 1.0 / (64.0 * frequency_hz)
    assert report["target_frequency_hz"] == frequency_hz
    assert report["time_plan"]["h_provisional_s"] == pytest.approx(
        target_cap_s, rel=0, abs=0
    )
    assert report["h_s"] <= target_cap_s
    assert report["target_accuracy"]["accepted"] is True


def test_target_accuracy_failure_prevents_fresh_advance(
    tmp_path, monkeypatch
):
    tool = _tool()
    h = 2.75e-10
    output = tmp_path / "target-failure"
    advance_calls = []

    def forbidden_advance(*args, **kwargs):
        advance_calls.append(1)
        raise AssertionError("advance must not run after target failure")

    monkeypatch.setattr(tool, "advance", forbidden_advance)
    assert (
        tool.run(
            _run_args(
                output,
                probe=1,
                h=h,
                target_frequency_hz=1.0 / (4.0 * h),
            ),
            context_factory=_small_context,
        )
        == 2
    )
    report = json.loads((output / "report.json").read_text())
    assert advance_calls == []
    assert report["reached_macros"] == 0
    assert "target-frequency accuracy gate failed" in report["failure"]["message"]
    assert report["requested_h_s"] == h
    assert report["h_s"] == h
    assert report["target_accuracy"]["horizon_s"] == pytest.approx(
        report["requested_horizon_s"], rel=0, abs=1.0e-18
    )


def test_target_accuracy_failure_prevents_resumed_advance(
    tmp_path, monkeypatch
):
    tool = _tool()
    prefix = tmp_path / "prefix"
    resumed = tmp_path / "resumed"
    target = 1.0e6
    assert (
        tool.run(
            _run_args(prefix, probe=2, target_frequency_hz=target),
            context_factory=_small_context,
        )
        == 2
    )

    real_target_accuracy = tool._target_accuracy
    advance_calls = []

    def rejected_target_accuracy(*args, **kwargs):
        report = real_target_accuracy(*args, **kwargs)
        report["accepted"] = False
        report["reasons"] = ["forced resumed target rejection"]
        return report

    def forbidden_advance(*args, **kwargs):
        advance_calls.append(1)
        raise AssertionError("advance must not run after target failure")

    monkeypatch.setattr(tool, "_target_accuracy", rejected_target_accuracy)
    monkeypatch.setattr(tool, "advance", forbidden_advance)
    assert (
        tool.run(
            _run_args(
                resumed,
                probe=2,
                resume_from=prefix,
                target_frequency_hz=target,
            ),
            context_factory=_small_context,
        )
        == 2
    )
    report = json.loads((resumed / "report.json").read_text())
    assert advance_calls == []
    assert report["reached_macros"] == 2
    assert report["target_accuracy"]["accepted"] is False
    assert "target-frequency accuracy gate failed" in report["failure"]["message"]
    assert report["target_accuracy"]["horizon_s"] == pytest.approx(
        report["requested_horizon_s"], rel=0, abs=1.0e-18
    )


def test_compatibility_identity_includes_perturbation_seed(
    small_case,
):
    context, _, _ = small_case
    tool = _tool()
    controls = {
        "kind": "circle",
        "nodes": 4,
        "K": 1,
        "mode": 0,
        "device": "cpu",
        "amplitude": 1.0e-4,
        "target_frequency_hz": None,
    }
    plan = tool.plan_time(1.0e-9, 1.0e-10, 1, "synchronous")
    identity = tool._compatibility_identity(
        context, controls, plan, seed_sha256="a" * 64
    )
    changed_amplitude = tool._compatibility_identity(
        context, {**controls, "amplitude": 5.0e-5}, plan, seed_sha256="a" * 64
    )
    changed_mode = tool._compatibility_identity(
        context, {**controls, "mode": 1}, plan, seed_sha256="a" * 64
    )
    changed_seed = tool._compatibility_identity(
        context, controls, plan, seed_sha256="b" * 64
    )
    assert changed_amplitude != identity
    assert changed_mode != identity
    assert changed_seed != identity


def test_resume_rejects_changed_amplitude_before_model_calls(
    tmp_path, monkeypatch
):
    tool = _tool()
    prefix = tmp_path / "prefix"
    resumed = tmp_path / "resumed"
    assert (
        tool.run(
            _run_args(prefix, probe=2),
            context_factory=_small_context,
        )
        == 2
    )

    model_calls = []

    def forbidden_model_call(*args, **kwargs):
        model_calls.append(1)
        raise AssertionError("model call must not run before identity rejection")

    monkeypatch.setattr(tool.stage, "initialize", forbidden_model_call)
    assert (
        tool.run(
            _run_args(
                resumed,
                probe=2,
                resume_from=prefix,
                amplitude=5.0e-5,
            ),
            context_factory=_small_context,
        )
        == 2
    )
    report = json.loads((resumed / "report.json").read_text())
    assert model_calls == []
    assert "amplitude" in report["failure"]["message"]


def test_resume_rejects_changed_seed_selector_before_context_calls(
    tmp_path, monkeypatch
):
    from tools.warp_tokamak_cae import background, make_context

    context, meta, equilibrium = make_context(
        "circle", 4, 0, "cpu", memory_limit_bytes=16 << 30
    )
    initial, _, _ = background(context, meta, equilibrium)
    tool = _tool()
    selected_archive = tmp_path / "selected-seed"
    changed_archive = tmp_path / "changed-seed"
    prefix = tmp_path / "prefix"
    resumed = tmp_path / "resumed"
    _write_seed_archive(
        selected_archive, context, initial, frequency_hz=2.0e8
    )
    _write_seed_archive(
        changed_archive, context, initial, frequency_hz=2.1e8
    )
    assert (
        tool.run(
            _run_args(
                prefix,
                probe=1,
                h="auto",
                horizon_s=1.0e-9,
                spectrum_directory=selected_archive,
            ),
            context_factory=_small_context,
        )
        == 2
    )
    original_identity = _metadata(
        prefix / "checkpoints" / "control.npz"
    )["initial_provenance"]["seed_identity"]

    context_calls = []
    model_calls = []

    def forbidden_context(*args, **kwargs):
        context_calls.append(1)
        raise AssertionError("context must not be built before selector rejection")

    def forbidden_model_call(*args, **kwargs):
        model_calls.append(1)
        raise AssertionError("model call must not run before selector rejection")

    monkeypatch.setattr(tool.stage, "initialize", forbidden_model_call)
    assert (
        tool.run(
            _run_args(
                resumed,
                probe=1,
                h="auto",
                horizon_s=1.0e-9,
                resume_from=prefix,
                spectrum_directory=changed_archive,
            ),
            context_factory=forbidden_context,
        )
        == 2
    )
    report = json.loads((resumed / "report.json").read_text())
    assert context_calls == []
    assert model_calls == []
    assert "resume seed selector mismatch" in report["failure"]["message"]
    assert report["case"]["spectrum_directory"] == original_identity[
        "selector"
    ]["directory"]
    assert report["seed_selector"] == original_identity["selector"]


def test_resume_omitted_selector_restores_original_metadata(
    tmp_path,
):
    from tools.warp_tokamak_cae import background, make_context

    context, meta, equilibrium = make_context(
        "circle", 4, 0, "cpu", memory_limit_bytes=16 << 30
    )
    initial, _, _ = background(context, meta, equilibrium)
    tool = _tool()
    selected_archive = tmp_path / "selected-seed"
    prefix = tmp_path / "prefix"
    resumed = tmp_path / "resumed"
    _write_seed_archive(
        selected_archive, context, initial, frequency_hz=2.0e8
    )
    assert (
        tool.run(
            _run_args(
                prefix,
                probe=1,
                h="auto",
                horizon_s=1.0e-9,
                spectrum_directory=selected_archive,
            ),
            context_factory=_small_context,
        )
        == 2
    )
    previous = _metadata(prefix / "checkpoints" / "control.npz")
    selector = previous["initial_provenance"]["seed_identity"]["selector"]
    shutil.rmtree(selected_archive)

    assert (
        tool.run(
            _run_args(
                resumed,
                probe=10,
                h="auto",
                horizon_s=1.0e-9,
                resume_from=prefix,
            ),
            context_factory=_small_context,
        )
        == 0
    )
    report = json.loads((resumed / "report.json").read_text())
    current = _metadata(resumed / "checkpoints" / "control.npz")
    assert report["status"] == "completed"
    assert report["case"]["spectrum_directory"] == selector["directory"]
    assert report["seed_selector"] == selector
    assert report["selected_frequency_hz"] == 2.0e8
    assert current["initial_provenance"] == previous["initial_provenance"]


def test_resume_rejects_changed_runtime_fingerprint_before_model_calls(
    tmp_path, monkeypatch
):
    tool = _tool()
    prefix = tmp_path / "prefix"
    assert (
        tool.run(
            _run_args(prefix, probe=2),
            context_factory=_small_context,
        )
        == 2
    )
    original = tool._runtime_fingerprint()
    runner_key = next(
        path
        for path in original["files"]
        if path.endswith("warp_050_experiments.py")
    )
    helper_key = next(
        path
        for path in original["files"]
        if path.endswith("warp_tokamak_cae.py")
    )
    mutations = {
        "runner": lambda value: value["files"].__setitem__(
            runner_key, "0" * 64
        ),
        "helper": lambda value: value["files"].__setitem__(
            helper_key, "1" * 64
        ),
        "runtime_hash": lambda value: value.__setitem__("sha256", "2" * 64),
    }
    model_calls = []

    def forbidden_model_call(*args, **kwargs):
        model_calls.append(1)
        raise AssertionError("model call must not run before runtime rejection")

    monkeypatch.setattr(tool.stage, "initialize", forbidden_model_call)
    for name, mutate in mutations.items():
        altered = copy.deepcopy(original)
        mutate(altered)
        monkeypatch.setattr(
            tool, "_runtime_fingerprint", lambda altered=altered: altered
        )
        output = tmp_path / f"resume-{name}"
        assert (
            tool.run(
                _run_args(output, probe=2, resume_from=prefix),
                context_factory=_small_context,
            )
            == 2
        )
        report = json.loads((output / "report.json").read_text())
        assert "runtime fingerprint mismatch" in report["failure"]["message"]
    assert model_calls == []


def test_neighborhood_guard_clamps_ratios_and_uses_micro_h(small_case):
    context, _, _ = small_case
    cells = context.op.Nc
    field = np.zeros((context.N, 3, cells), dtype=np.complex128)
    field[:, 0, :] = 1.0
    frame = SimpleNamespace(
        BN=field,
        fluid=(
            SimpleNamespace(
                n=np.full(cells, 0.8, dtype=np.complex128),
                perp=np.full(cells, 1.0, dtype=np.complex128),
                u=np.zeros((context.N, 3, cells), dtype=np.complex128),
            ),
            SimpleNamespace(
                n=np.full(cells, 1.0, dtype=np.complex128),
                perp=np.full(cells, 1.0, dtype=np.complex128),
                parallel=np.full(cells, 1.0, dtype=np.complex128),
                u=np.zeros((context.N, 3, cells), dtype=np.complex128),
            ),
        ),
    )
    baseline_rates = {
        "ion_pressure_rate": 10.0,
        "electron_pressure_rate": 20.0,
        "bulk_pressure_rate": 30.0,
        "advection_rate": 0.0,
    }
    accepted = _tool()._neighborhood_guard(
        frame,
        context,
        baseline_magnetic_max_T=1.0,
        baseline_density_min=1.0,
        baseline_rate_estimate=baseline_rates,
        rho_max_rad_s=100.0,
        micro_h_s=0.003,
    )
    assert accepted["ratio"] == pytest.approx(1.25)
    assert accepted["accepted"] is True
    assert "not a nonlinear stability proof" in accepted["scope"]
    rejected = _tool()._neighborhood_guard(
        frame,
        context,
        baseline_magnetic_max_T=1.0,
        baseline_density_min=1.0,
        baseline_rate_estimate=baseline_rates,
        rho_max_rad_s=100.0,
        micro_h_s=0.0041,
    )
    assert rejected["accepted"] is False
    assert rejected["rho_ratio_h"] == pytest.approx(0.5125)


def test_array_inventory_counts_host_numpy_and_device_proxy(monkeypatch):
    class FakeDeviceArray:
        def __init__(self, value):
            self.value = np.asarray(value)
            self.dtype = self.value.dtype
            self.shape = self.value.shape
            self.nbytes = self.value.nbytes

    monkeypatch.setitem(
        sys.modules, "cupy", SimpleNamespace(ndarray=FakeDeviceArray)
    )
    host = np.arange(4, dtype=np.float64)
    device = FakeDeviceArray(np.arange(3, dtype=np.complex128))
    context = SimpleNamespace(
        compute=SimpleNamespace(name="cuda"),
        geometry=SimpleNamespace(host=host),
        op=SimpleNamespace(device=device),
    )
    experiment = {
        "control": {
            "state": SimpleNamespace(host=host),
            "frame": SimpleNamespace(),
            "ledger": SimpleNamespace(),
        }
    }
    inventory = _tool()._array_inventory(context, experiment)
    assert inventory["host_resident_bytes"] == host.nbytes
    assert inventory["device_resident_bytes"] == device.nbytes
    assert sum(item["path"] == "context.geometry.host" for item in inventory["arrays"]) == 1


def test_peak_sample_is_unavailable_not_zero_for_missing_measurement():
    assert _tool()._sampled_peak([{"rss_bytes": None}, {}], "rss_bytes") is None


def test_parent_checkpoint_hash_mismatch_is_rejected(tmp_path):
    path = tmp_path / "parent.npz"
    np.savez(
        path,
        metadata_json=np.asarray(
            json.dumps({"parent_checkpoints": []})
        ),
    )
    with pytest.raises(ValueError, match="parent checkpoint hash mismatch"):
        _tool()._validate_parent_checkpoints(
            [{"path": str(path), "sha256": "0" * 64}]
        )


def test_runtime_fingerprint_covers_complete_runtime():
    fingerprint = _tool()._runtime_fingerprint()
    files = fingerprint["files"]
    assert len(fingerprint["sha256"]) == 64
    assert any(path.endswith("/energy.py") for path in files)
    assert any(path.endswith("warp_050_experiments.py") for path in files)
    assert any(path.endswith("warp_tokamak_cae.py") for path in files)


def _run_args(
    output: Path,
    *,
    probe: int,
    resume_from: Path | None = None,
    stability_directories: list[Path] | None = None,
    K: int = 0,
    mode: int = 0,
    h: float | str = 2.75e-10,
    horizon_s: float | None = None,
    target_frequency_hz: float | None = None,
    amplitude: float = 1.0e-4,
    spectrum_directory: Path | None = None,
):
    micro_steps = 2
    return SimpleNamespace(
        kind="circle",
        nodes=4,
        K=K,
        mode=mode,
        device="cpu",
        equilibrium=None,
        coupling="waveform",
        micro_steps=micro_steps,
        horizon_s=(
            4 * micro_steps * 2.75e-10 if horizon_s is None else horizon_s
        ),
        h=h,
        output=output,
        wall_seconds=180.0,
        probe_macros=probe,
        snapshot_every=2,
        target_frequency_hz=target_frequency_hz,
        spectrum_directory=spectrum_directory,
        stability_spectrum_directories=stability_directories or [],
        resume_from=resume_from,
        amplitude=amplitude,
    )


def _force_operator_envelope_failure(monkeypatch, tool, micro_h: float):
    real_estimate = tool.stability.stage_frequency_estimate

    def high_frequency_estimate(stage, context):
        estimate = dict(real_estimate(stage, context))
        estimate["omega_estimate"] = 1.0 / micro_h
        return estimate

    monkeypatch.setattr(
        tool.stability, "stage_frequency_estimate", high_frequency_estimate
    )


def _small_context(kind, nodes, K, device, equilibrium, memory_limit_bytes):
    from tools.warp_tokamak_cae import make_context

    return make_context(
        kind,
        4,
        K,
        device,
        equilibrium,
        memory_limit_bytes=memory_limit_bytes,
    )


def _metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["metadata_json"].item()))


def _ledger(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["cumulative_json"].item()))


def _assert_ledgers_close(left, right, path=""):
    if isinstance(left, dict):
        assert set(left) == set(right), path
        for key in left:
            if key == "last_trial":
                continue
            _assert_ledgers_close(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_ledgers_close(a, b, f"{path}[{index}]")
    elif isinstance(left, float):
        assert left == pytest.approx(right, rel=1.0e-11, abs=1.0e-12), path
    else:
        assert left == right, path


def test_two_plus_two_continuation_preserves_accepted_state_and_ledger(
    tmp_path, monkeypatch
):
    tool = _tool()
    prefix = tmp_path / "prefix"
    resumed = tmp_path / "resumed"
    uninterrupted = tmp_path / "uninterrupted"

    assert tool.run(
        _run_args(prefix, probe=2), context_factory=_small_context
    ) == 2
    original_hashes = {
        path.relative_to(prefix): _sha(path)
        for path in prefix.rglob("*")
        if path.is_file()
    }
    assert tool.run(
        _run_args(resumed, probe=2, resume_from=prefix),
        context_factory=_small_context,
    ) == 0
    assert tool.run(
        _run_args(uninterrupted, probe=4), context_factory=_small_context
    ) == 0
    assert original_hashes == {
        path.relative_to(prefix): _sha(path)
        for path in prefix.rglob("*")
        if path.is_file()
    }

    resumed_report = json.loads((resumed / "report.json").read_text())
    assert resumed_report["status"] == "completed"
    assert resumed_report["continuation"]["start_step"] == 2
    assert resumed_report["setup_cost_breakdown"][
        "source_cache_reconstruction_s"
    ] > 0.0
    history = [
        json.loads(line)
        for line in (resumed / "logs" / "control-history.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [item["step"] for item in history] == [2, 3, 4]
    assert (resumed / "snapshots" / "control-step-00000004.npz").is_file()

    for label in ("control", "perturbed"):
        with np.load(
            resumed / "checkpoints" / f"{label}.npz", allow_pickle=False
        ) as resumed_npz, np.load(
            uninterrupted / "checkpoints" / f"{label}.npz",
            allow_pickle=False,
        ) as reference_npz:
            for name in (
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
            ):
                scale = max(
                    float(np.max(np.abs(reference_npz[name]), initial=0.0)),
                    1.0,
                )
                np.testing.assert_allclose(
                    resumed_npz[name],
                    reference_npz[name],
                    rtol=1.0e-10,
                    atol=1.0e-10 * scale,
                )
        _assert_ledgers_close(
            _ledger(resumed / "checkpoints" / f"{label}.npz"),
            _ledger(uninterrupted / "checkpoints" / f"{label}.npz"),
            label,
        )

    previous = _metadata(prefix / "checkpoints" / "control.npz")
    current = _metadata(resumed / "checkpoints" / "control.npz")
    assert current["schema"] == "fomhd-warp-050-accepted-state-v2"
    assert current["initial_provenance"] == previous["initial_provenance"]
    assert current["scheduler"]["next_trial"] > previous["scheduler"]["next_trial"]
    assert current["scheduler"]["next_source"] > previous["scheduler"]["next_source"]
    assert current["scheduler"]["state_generation"] > previous["scheduler"]["state_generation"]
    assert current["parent_checkpoints"] == [
        {
            "label": label,
            "path": str(prefix / "checkpoints" / f"{label}.npz"),
            "sha256": _sha(prefix / "checkpoints" / f"{label}.npz"),
        }
        for label in ("control", "perturbed")
    ]

    failed = tmp_path / "failed"
    real_advance = tool.advance

    def fail_after_first_resumed_macro(committed, ledger, context):
        if committed.state.step >= 2:
            raise RuntimeError("synthetic resumed failure")
        return real_advance(committed, ledger, context)

    monkeypatch.setattr(tool, "advance", fail_after_first_resumed_macro)
    assert tool.run(
        _run_args(failed, probe=4, resume_from=prefix),
        context_factory=_small_context,
    ) == 2
    for label in ("control", "perturbed"):
        checkpoint = _metadata(failed / "checkpoints" / f"{label}.npz")
        assert checkpoint["step"] == 2
        assert checkpoint["accepted"] is True
        assert (failed / "failures" / f"{label}-macro-00000003.npz").is_file()
    assert (failed / "failures" / "perturbed-macro-00000003.npz").is_file()
    timing = [
        json.loads(line)
        for line in (failed / "logs" / "advance-timing.jsonl")
        .read_text()
        .splitlines()
    ]
    assert any(item["status"] == "failed" for item in timing)

    mismatch = tmp_path / "mismatch"
    mismatch_args = _run_args(mismatch, probe=2, resume_from=prefix)
    mismatch_args.h *= 1.000001
    assert tool.run(
        mismatch_args, context_factory=_small_context
    ) == 2
    mismatch_report = json.loads((mismatch / "report.json").read_text())
    assert "time configuration mismatch" in mismatch_report["failure"]["message"]

    corrupt = tmp_path / "corrupt"
    shutil.copytree(prefix, corrupt)
    checkpoint = corrupt / "checkpoints" / "control.npz"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")
    corrupt_output = tmp_path / "corrupt-output"
    assert tool.run(
        _run_args(corrupt_output, probe=2, resume_from=corrupt),
        context_factory=_small_context,
    ) == 2
    corrupt_report = json.loads(
        (corrupt_output / "report.json").read_text()
    )
    assert "checkpoint file hash mismatch" in corrupt_report["failure"][
        "message"
    ]


def test_complete_spectra_run_ignores_operator_envelope_failure(
    tmp_path, small_case, monkeypatch
):
    context, _, initial = small_case
    tool = _tool()
    sector = tmp_path / "sector-0"
    sector_1 = tmp_path / "sector-1"
    _write_stability_archive(sector, context, initial, mode=0)
    _write_stability_archive(sector_1, context, initial, mode=1)
    _force_operator_envelope_failure(monkeypatch, tool, micro_h=2.75e-10)

    output = tmp_path / "complete-spectra"
    assert (
        tool.run(
            _run_args(
                output,
                probe=1,
                stability_directories=[sector, sector_1],
                K=1,
            ),
            context_factory=_small_context,
        )
        == 2
    )

    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "budget_stopped"
    assert report["reached_macros"] == 1
    assert "failure" not in report
    assert report["step_policy"]["hard_runtime_envelope"] == (
        "not applied when complete spectra are present"
    )

    envelopes = [
        json.loads(line)
        for line in (output / "logs" / "runtime-envelope.jsonl")
        .read_text()
        .splitlines()
    ]
    assert envelopes[0]["values"]["control"]["accepted"] is False
    assert envelopes[0]["values"]["perturbed"]["accepted"] is False

    neighborhood = [
        json.loads(line)
        for line in (output / "logs" / "stability-neighborhood.jsonl")
        .read_text()
        .splitlines()
    ]
    assert neighborhood[0]["values"]["control"]["accepted"] is True
    assert neighborhood[0]["values"]["perturbed"]["accepted"] is True
    observed = neighborhood[0]["values"]["control"][
        "observed_rates_relative_to_initial"
    ]
    assert set(observed) == {
        "ion_pressure_per_density",
        "electron_pressure_per_density",
        "bulk_pressure_per_density",
        "advection",
    }
    assert all(
        math.isfinite(value["current_rad_s"])
        and math.isfinite(value["initial_rad_s"])
        and math.isfinite(value["relative_to_initial"])
        for value in observed.values()
    )
    assert "no acceptance threshold" in neighborhood[0]["values"]["control"][
        "rate_observation_scope"
    ]


def test_run_without_spectra_keeps_operator_envelope_hard(
    tmp_path, monkeypatch
):
    tool = _tool()
    _force_operator_envelope_failure(monkeypatch, tool, micro_h=2.75e-10)
    output = tmp_path / "conservative"

    assert (
        tool.run(
            _run_args(output, probe=1),
            context_factory=_small_context,
        )
        == 2
    )
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["reached_macros"] == 0
    assert report["failure"]["code"] == "RuntimeStabilityEnvelope"
    assert report["step_policy"]["hard_runtime_envelope"] == (
        "omega_current*h <= 0.5"
    )
    assert not (output / "logs" / "stability-neighborhood.jsonl").exists()


def test_bounded_forty_node_capacity_probe_is_explicitly_selectable(tmp_path):
    tool = _tool()
    args = tool._parser().parse_args([
        '--kind', 'east', '--nodes', '40', '--K', '1', '--mode', '1',
        '--coupling', 'synchronous', '--horizon-s', '8e-11',
        '--equilibrium', str(tmp_path / 'equilibrium'),
        '--output', str(tmp_path / 'capacity-probe')])
    assert args.nodes == 40
    assert tool.HOST_MEMORY_CAP == 16 << 30
    assert tool.DEVICE_MEMORY_CAP == 8 << 30
