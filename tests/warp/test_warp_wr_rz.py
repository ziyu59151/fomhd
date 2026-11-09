from __future__ import annotations

import importlib.util
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

SOURCE_ROOT = Path(
    os.environ.get("FOMHD_SOURCE_ROOT", Path(__file__).parents[2])
).resolve()
TOOLS_ROOT = Path(os.environ.get("FOMHD_TOOLS_ROOT", SOURCE_ROOT)).resolve()
EQUILIBRIUM = Path(
    "/home/lxzeng/runs/fomhd/a0-tokamak-cae__20261105T154538Z/"
    "inputs/g048059.03650"
)
MODULE_PATH = Path(__file__).parents[2] / "tools/warp_wr_rz.py"
TEST_H_S = 1.023199502902286e-9


def _configure_import_path() -> None:
    adapter = SOURCE_ROOT / "src/fomhd/multirate_wr_physics.py"
    transaction = SOURCE_ROOT / "src/fomhd/time_multirate.py"
    if not adapter.is_file() or not transaction.is_file():
        raise RuntimeError(
            "selected source root lacks the accepted WR transaction: "
            f"{SOURCE_ROOT}"
        )
    for path in (str(TOOLS_ROOT), str(SOURCE_ROOT), str(SOURCE_ROOT / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ["FOMHD_SOURCE_ROOT"] = str(SOURCE_ROOT)
    import fomhd
    import fomhd.multirate_wr_physics
    import fomhd.time_multirate

    imported_root = Path(fomhd.__file__).resolve().parents[2]
    paths = (
        Path(fomhd.multirate_wr_physics.__file__).resolve(),
        Path(fomhd.time_multirate.__file__).resolve(),
    )
    if imported_root != SOURCE_ROOT:
        raise RuntimeError(f"fomhd root {imported_root} != {SOURCE_ROOT}")
    if not all(path.is_relative_to(SOURCE_ROOT / "src") for path in paths):
        raise RuntimeError("WR runtime modules resolve outside the source root")


@pytest.fixture(scope="module")
def rz_module():
    _configure_import_path()
    spec = importlib.util.spec_from_file_location("warp_wr_rz", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def args(
    output: Path,
    *,
    coupling: str = "wr",
    h: float = TEST_H_S,
    M: int = 1,
    macros: int = 1,
    segment_macros: int | None = None,
    resume: Path | None = None,
    resume_checkpoint: Path | None = None,
    amplitude: float = 1.0e-4,
    waveform_rtol: float = 1.0e-10,
) -> Namespace:
    return Namespace(
        nodes=4,
        device="cpu",
        h=h,
        M=M,
        s=3,
        macros=macros,
        segment_macros=segment_macros,
        coupling=coupling,
        output=output,
        equilibrium=EQUILIBRIUM,
        wall_seconds=1600.0,
        snapshot_every=1,
        amplitude=amplitude,
        waveform_rtol=waveform_rtol,
        resume=resume,
        resume_checkpoint=resume_checkpoint,
    )


@pytest.fixture(scope="module")
def direct_wr(rz_module, tmp_path_factory):
    output = tmp_path_factory.mktemp("direct-wr") / "run"
    report = rz_module.run(
        args(output, M=1, macros=2, segment_macros=2),
        allow_fixture_nodes=True,
    )
    return output, report


@pytest.fixture(scope="module")
def segmented_wr(rz_module, tmp_path_factory):
    base = tmp_path_factory.mktemp("segmented-wr")
    first_output = base / "segment-1"
    resume_output = base / "segment-2"
    first = rz_module.run(
        args(first_output, M=1, macros=2, segment_macros=1),
        allow_fixture_nodes=True,
    )
    resumed = rz_module.run(
        args(
            resume_output,
            M=1,
            macros=2,
            segment_macros=1,
            resume_checkpoint=first_output / "checkpoints/latest.json",
        ),
        allow_fixture_nodes=True,
    )
    return first_output, first, resume_output, resumed


@pytest.fixture(scope="module")
def cross_family(rz_module, tmp_path_factory):
    root = tmp_path_factory.mktemp("cross-family")
    reports = {}
    for coupling in ("wr", "waveform", "synchronous"):
        output = root / coupling
        reports[coupling] = (
            output,
            rz_module.run(
                args(
                    output,
                    coupling=coupling,
                    h=0.5 * TEST_H_S,
                    M=2,
                    macros=1,
                ),
                allow_fixture_nodes=True,
            ),
        )
    return reports


def _checkpoint_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def test_complete_wr_two_macros_commit_production_accepted_evidence(direct_wr):
    output, report = direct_wr
    assert report["status"] == "completed"
    assert report["coupling"] == "wr"
    assert report["production_accepted"] is True
    assert report["global_requested_macros"] == 2
    assert report["reached_macros"] == 2
    assert report["source_plan"]["context_coupling"] == "multirate_wr"
    assert report["source_plan"]["global_nsteps"] == 2
    assert report["ensembles"]["control"]["step"] == 2
    assert report["ensembles"]["perturbed"]["step"] == 2
    assert (
        report["ensembles"]["control"]["time_s"]
        == report["ensembles"]["perturbed"]["time_s"]
    )
    assert report["work_total"]["fine_full_stage_calls"] == 0
    assert report["work_total"]["full_ion_calls"] > 0
    assert report["work_total"]["field_electron_calls"] > 0
    assert report["work_total"]["transport_producer_calls"] > 0
    assert report["work_total"]["wall_projections"] > 0
    assert report["checks"]["hard_fail"] == 0
    assert report["ledger"]["accepted_intervals"] == 2
    assert report["ledger"]["last_step"] == 2
    assert len(report["macros"]) == 2
    for macro in report["macros"]:
        assert macro["production_accepted"] is True
        assert macro["coupling_metadata"]["coupling"] == "multirate_wr"
        assert macro["coupling_metadata"]["endpoint"]["production_accepted"] is True
        assert macro["residual_history"]
    assert (output / "checkpoints/control.npz").is_file()
    assert (output / "checkpoints/perturbed.npz").is_file()
    assert (output / "checkpoints/latest.json").is_file()
    assert (output / "logs/accepted-checks.jsonl").is_file()
    assert (output / "logs/accepted-checks.jsonl").read_text(
        encoding="utf-8"
    ).strip()


def test_checkpoint_retains_all_twelve_ledger_and_six_boundary_fields(
    rz_module, direct_wr
):
    output, report = direct_wr
    from fomhd import io
    from fomhd.multirate_wr_checks import LEDGER_FIELDS

    arrays = _checkpoint_arrays(output / "checkpoints/control.npz")
    cumulative = io._cumulative_from_json(
        io.parse_json(str(arrays["cumulative_json"].item()))
    )
    assert len(LEDGER_FIELDS) == 12
    assert len(vars(cumulative.integrals[0])) == 12
    assert len(vars(cumulative.integrals[1])) == 12
    assert set(vars(cumulative.boundary_integrals)) == {
        "number",
        "kinetic",
        "internal_perp",
        "internal_parallel",
        "field_energy",
        "total_energy",
    }
    assert cumulative.accepted_intervals == 2
    assert cumulative.last_step == 2
    assert cumulative.last_time == report["ensembles"]["control"]["time_s"]
    assert int(arrays["accepted"].item()) == 1


def test_one_macro_plus_resume_matches_uninterrupted_two_macro_run(
    direct_wr, segmented_wr
):
    direct_output, direct_report = direct_wr
    first_output, first, resume_output, resumed = segmented_wr
    assert first["status"] == "segment_complete"
    assert first["global_requested_macros"] == 2
    assert first["reached_macros"] == 1
    assert resumed["status"] == "completed"
    assert resumed["reached_macros"] == 2
    assert resumed["parent"]["source_accepted_macros"] == 1
    assert resumed["parent"]["control_checkpoint_sha256"] == first["ensembles"][
        "control"
    ]["checkpoint_sha256"]

    direct_state = _checkpoint_arrays(direct_output / "checkpoints/control.npz")
    resumed_state = _checkpoint_arrays(resume_output / "checkpoints/control.npz")
    direct_time = direct_report["ensembles"]["control"]["time_s"]
    assert resumed["ensembles"]["control"]["time_s"] == direct_time
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
        np.testing.assert_allclose(
            resumed_state[name],
            direct_state[name],
            rtol=2.0e-12,
            atol=1.0e-15,
        )
    for name in ("step", "time", "generation", "source_fingerprint"):
        if name == "time":
            np.testing.assert_allclose(resumed_state[name], direct_state[name])
        else:
            assert resumed_state[name].item() == direct_state[name].item()


def test_resume_rejects_changed_global_scheme_before_work(rz_module, segmented_wr):
    first_output, _, _, _ = segmented_wr
    mismatch_output = first_output.parent / "bad-resume"
    with pytest.raises(rz_module.RunnerError, match="global|scheme|macros"):
        rz_module.run(
            args(
                mismatch_output,
                M=1,
                macros=3,
                segment_macros=1,
                resume=first_output,
            ),
            allow_fixture_nodes=True,
        )
    assert not mismatch_output.exists()
    amplitude_output = first_output.parent / "bad-amplitude-resume"
    with pytest.raises(rz_module.RunnerError, match="scheme"):
        rz_module.run(
            args(
                amplitude_output,
                M=1,
                macros=2,
                segment_macros=1,
                resume=first_output,
                amplitude=5.0e-5,
            ),
            allow_fixture_nodes=True,
        )
    assert not amplitude_output.exists()
    rtol_output = first_output.parent / "bad-rtol-resume"
    with pytest.raises(rz_module.RunnerError, match="scheme"):
        rz_module.run(
            args(
                rtol_output,
                M=1,
                macros=2,
                segment_macros=1,
                resume=first_output,
                waveform_rtol=1.0e-12,
            ),
            allow_fixture_nodes=True,
        )
    assert not rtol_output.exists()


def test_all_three_families_match_physical_horizon_and_step_semantics(
    cross_family,
):
    expected_time = TEST_H_S
    reports = {name: value[1] for name, value in cross_family.items()}
    for report in reports.values():
        assert report["status"] == "completed"
        assert report["ensembles"]["control"]["time_s"] == expected_time
        assert report["ensembles"]["perturbed"]["time_s"] == expected_time
        assert report["source_plan"]["effective_macro_s"] == expected_time
    assert reports["wr"]["ensembles"]["control"]["step"] == 1
    assert reports["waveform"]["ensembles"]["control"]["step"] == 1
    assert reports["synchronous"]["ensembles"]["control"]["step"] == 2
    assert reports["synchronous"]["work_total"]["complete_advance_calls"] == 4
    assert reports["waveform"]["work_total"]["complete_advance_calls"] == 2
    assert reports["wr"]["work_total"]["complete_advance_calls"] == 2


def test_reference_producer_counts_come_from_actual_scoped_calls(cross_family):
    sync = cross_family["synchronous"][1]
    t1 = cross_family["waveform"][1]
    wr = cross_family["wr"][1]
    for report in (sync, t1):
        assert report["work_total"]["full_ion_calls"] > 0
        assert report["work_total"]["field_electron_calls"] > 0
        assert report["work_total"]["transport_producer_calls"] > 0
        assert report["work_total"]["wall_projections"] > 0
        assert report["work_total"]["stage_evaluations"] > 0
    assert wr["work_total"]["fine_full_stage_calls"] == 0
    assert wr["work_total"]["coarse_transport_calls"] > 0
    assert wr["work_total"]["accepted_endpoint_full_stage_calls"] == 2


def test_injected_failure_preserves_resumed_left_checkpoint(
    rz_module, segmented_wr, tmp_path
):
    first_output, first, _, _ = segmented_wr
    original_control = first_output / "checkpoints/control.npz"
    original_perturbed = first_output / "checkpoints/perturbed.npz"
    before = {
        "control": original_control.read_bytes(),
        "perturbed": original_perturbed.read_bytes(),
    }
    failure_output = tmp_path / "failed-resume"
    calls = 0

    def failing_advance(*args, **kwargs):
        nonlocal calls
        calls += 1
        from fomhd.types import Failure

        raise Failure("InjectedFailure", "intentional accepted-resume failure")

    report = rz_module.run(
        args(
            failure_output,
            M=1,
            macros=2,
            segment_macros=1,
            resume=first_output,
        ),
        advance_callable=failing_advance,
        allow_fixture_nodes=True,
    )
    assert calls == 1
    assert report["status"] == "failed"
    assert report["failure"]["code"] == "InjectedFailure"
    assert original_control.read_bytes() == before["control"]
    assert original_perturbed.read_bytes() == before["perturbed"]
    assert (failure_output / "failures/left-control.npz").is_file()
    assert (failure_output / "failures/left-perturbed.npz").is_file()


def test_output_and_cuda_guards(rz_module, tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(rz_module.RunnerError, match="NEW output"):
        rz_module.validate_config(args(existing), allow_fixture_nodes=True)
    cuda = args(tmp_path / "cuda")
    cuda.device = "cuda:0"
    with pytest.raises(rz_module.RunnerError, match="SLURM_JOB_ID"):
        rz_module.validate_config(
            cuda, allow_fixture_nodes=True, environ={}
        )


def test_nonzero_east_k1_perturbation_and_physical_snapshot_layout(direct_wr):
    output, report = direct_wr
    with np.load(output / "perturbation_mapping.npz", allow_pickle=False) as seed:
        assert np.max(np.abs(seed["mapped_velocity"])) > 0.0
    nc = report["actual_grid"]["Nc"]
    with np.load(
        output / "snapshots/control-macro-000002.npz", allow_pickle=False
    ) as saved:
        assert saved["BN"].shape == (5, 3, nc)
        assert saved["u_i"].shape == (5, 3, nc)
        assert saved["n_i"].shape == (5, nc)
        assert saved["p_i"].shape == (5, nc)
        assert saved["p_e_perp"].shape == (5, nc)
        assert saved["p_e_parallel"].shape == (5, nc)
    response = report["macros"][-1]["response"]
    assert response["matched"] is True
    assert response["n1_density_max_abs_per_m3"] > 0.0


def test_eight_macro_segment_resume_uses_exact_core_scheduler_clock(
    rz_module, tmp_path
):
    from types import SimpleNamespace
    from fomhd import core

    root = tmp_path / "long-clock"
    first_output = root / "segment-1"
    resume_output = root / "segment-2"
    h = 0.25 * TEST_H_S
    macros = 8
    first = rz_module.run(
        args(
            first_output,
            h=h,
            M=4,
            macros=macros,
            segment_macros=4,
            amplitude=1.0e-5,
            waveform_rtol=1.0e-14,
        ),
        allow_fixture_nodes=True,
    )
    assert first["status"] == "segment_complete"
    assert first["reached_macros"] == 4
    resumed = rz_module.run(
        args(
            resume_output,
            h=h,
            M=4,
            macros=macros,
            segment_macros=4,
            resume=first_output,
            amplitude=1.0e-5,
            waveform_rtol=1.0e-14,
        ),
        allow_fixture_nodes=True,
    )
    expected_time = core.time_at_step(
        SimpleNamespace(dt=4.0 * h, start_time=0.0), macros
    )
    assert resumed["status"] == "completed"
    assert resumed["reached_macros"] == macros
    assert resumed["ensembles"]["control"]["step"] == macros
    assert resumed["ensembles"]["control"]["time_s"] == expected_time
    assert resumed["ensembles"]["perturbed"]["time_s"] == expected_time
    assert resumed["source_plan"]["waveform_rtol"] == 1.0e-14
    assert resumed["model_metadata"]["waveform_rtol"] == 1.0e-14
    with np.load(
        resume_output / "checkpoints/control.npz", allow_pickle=False
    ) as checkpoint:
        assert float(checkpoint["waveform_rtol"].item()) == 1.0e-14
    with np.load(
        resume_output / "snapshots/control-macro-000008.npz",
        allow_pickle=False,
    ) as snapshot:
        assert float(snapshot["waveform_rtol"].item()) == 1.0e-14
