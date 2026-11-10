"""Real native-polygon restart and rollback transaction tests."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fomhd.polygon_geometry")
pytest.importorskip("fomhd.polygon_operators")

from fomhd import massless_field, stage, time as time_module  # noqa: E402
from fomhd.types import Failure, cumulative_record  # noqa: E402
from tools import warp_clt_boundary as runner  # noqa: E402


RTOL = 2.0e-9
PRIMARY = (
    "n_i_modes",
    "mom_i_modes",
    "n_e_modes",
    "mom_e_modes",
    "B_phi_modes",
    "B_t_modes",
    "current_modes",
)
SAMPLES = (
    "B_N_samples",
    "B_N_R_samples",
    "B_N_phi_samples",
    "B_N_Z_samples",
    "ion_velocity_samples",
    "ion_velocity_R_samples",
    "ion_velocity_phi_samples",
    "ion_velocity_Z_samples",
    "ion_density_samples",
    "ion_perp_pressure_samples",
    "electron_density_samples",
    "electron_perp_pressure_samples",
    "electron_parallel_pressure_samples",
)
CHECKPOINT_IDENTITY = (
    "checkpoint_schema",
    "complete_accepted",
    "layout_schema",
    "geometry_fingerprint",
    "geometry_file",
    "geometry_file_sha256",
    "operator_fingerprint",
    "source_fingerprint",
    "input_fingerprint",
    "scheme",
    "scheme_parameters",
    "micro_steps",
    "ion_nodes",
    "mesh",
    "baseline",
    "dt",
    "start_time",
    "nsteps",
)


@pytest.fixture(params=("cpu", "cuda:0"))
def device(request):
    if request.param == "cuda:0" and (
        os.environ.get("FOMHD_TEST_GPU") != "1"
        or not os.environ.get("SLURM_JOB_ID")
    ):
        pytest.skip(
            "CUDA transactions require FOMHD_TEST_GPU=1 and SLURM_JOB_ID"
        )
    return request.param


def _arguments(output, coupling, device, steps, segment, resume=None):
    values = [
        "--nodes",
        "8",
        "--mesh",
        "polygon",
        "--baseline",
        "vacuum",
        "--major-radius",
        "3",
        "--minor-radius",
        "0.5",
        "--K",
        "0",
        "--mode",
        "0",
        "--amplitude",
        "1e-4",
        "--seed",
        "smooth",
        "--h",
        "1e-10" if coupling == "wr" else "1e-12",
        "--M",
        "4" if coupling == "wr" else "1",
        "--s",
        "3",
        "--steps",
        str(steps),
        "--segment-steps",
        str(segment),
        "--coupling",
        coupling,
        "--waveform-rtol",
        "1e-12",
        "--device",
        device,
        "--output",
        str(output),
        "--wall-seconds",
        "120",
        "--snapshot-every",
        "1",
    ]
    if resume is not None:
        values.extend(["--resume", str(resume)])
    return runner.parse_args(values)


def _run(args):
    status = runner.run(args)
    output = args.output
    report = json.loads((output / "report.json").read_text())
    if status != 0:
        pytest.fail(
            f"runner failed: status={status!r} report={report.get('failure')}"
        )
    return report


def _close(actual, expected, label):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    scale = max(float(np.max(np.abs(expected), initial=0.0)), 1.0e-300)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=RTOL,
        atol=RTOL * scale,
        err_msg=label,
    )


def _nested_close(actual, expected, path="$"):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected:
            _nested_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, value in enumerate(expected):
            _nested_close(actual[index], value, f"{path}[{index}]")
        return
    if isinstance(expected, (int, float)):
        _close(actual, expected, path)
        return
    assert actual == expected, path


def _stable_coupling_metadata(value):
    if value is None:
        return None
    result = dict(value)
    for name in ("base_source_id", "endpoint_source_id"):
        result.pop(name, None)
    return result


def _load_snapshot(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _host_state(context, state):
    return runner.host_state_for_codec(state, context)


def _capture_committed(context, committed, cumulative):
    host = _host_state(context, committed.state)
    return {
        "state_object": committed.state,
        "cache_object": committed.cache,
        "state_step": int(committed.state.step),
        "state_time": float(committed.state.time),
        "state_generation": int(committed.state.generation),
        "arrays": {
            "n_i": host.fluid[0].n.copy(),
            "mom_i": host.fluid[0].mom.copy(),
            "p_i": host.fluid[0].perp.copy(),
            "n_e": host.fluid[1].n.copy(),
            "mom_e": host.fluid[1].mom.copy(),
            "p_perp_e": host.fluid[1].perp.copy(),
            "p_parallel_e": host.fluid[1].parallel.copy(),
            "B": host.B.copy(),
        },
        "cache_tag": dict(vars(committed.cache.tag)),
        "cumulative_json": runner.io._canonical(
            runner.io._cumulative_json(cumulative)
        ),
        "baseline_divB": runner._host(
            context, cumulative.baseline.divB
        ).copy(),
    }


def _assert_capture_unchanged(context, committed, cumulative, capture):
    assert committed.state is capture["state_object"]
    assert committed.cache is capture["cache_object"]
    assert committed.state.step == capture["state_step"]
    assert committed.state.time == capture["state_time"]
    assert committed.state.generation == capture["state_generation"]
    host = _host_state(context, committed.state)
    actual = {
        "n_i": host.fluid[0].n,
        "mom_i": host.fluid[0].mom,
        "p_i": host.fluid[0].perp,
        "n_e": host.fluid[1].n,
        "mom_e": host.fluid[1].mom,
        "p_perp_e": host.fluid[1].perp,
        "p_parallel_e": host.fluid[1].parallel,
        "B": host.B,
    }
    for name, expected in capture["arrays"].items():
        np.testing.assert_array_equal(actual[name], expected)
    assert vars(committed.cache.tag) == capture["cache_tag"]
    assert runner.io._canonical(
        runner.io._cumulative_json(cumulative)
    ) == capture["cumulative_json"]
    np.testing.assert_array_equal(
        runner._host(context, cumulative.baseline.divB),
        capture["baseline_divB"],
    )


@pytest.mark.parametrize("coupling", ("synchronous", "wr"))
def test_real_polygon_restart_matches_uninterrupted(
    tmp_path, coupling, device
):
    root = tmp_path / f"{coupling}-{device.replace(':', '-')}"
    prefix = root / "prefix"
    continued = root / "continued"
    uninterrupted = root / "uninterrupted"

    _run(_arguments(prefix, coupling, device, 2, 1))
    _run(
        _arguments(
            continued,
            coupling,
            device,
            2,
            1,
            prefix / "checkpoints" / "restart.npz",
        )
    )
    _run(_arguments(uninterrupted, coupling, device, 2, 2))

    left = _load_snapshot(continued / "snapshots" / "step-0000000002.npz")
    right = _load_snapshot(
        uninterrupted / "snapshots" / "step-0000000002.npz"
    )
    for name in PRIMARY + SAMPLES:
        _close(left[name], right[name], name)

    left_meta = json.loads(str(left["metadata_json"]))
    right_meta = json.loads(str(right["metadata_json"]))
    assert left_meta["frame_tag"] == right_meta["frame_tag"]
    assert left_meta["generation"] == right_meta["generation"]
    assert left_meta["step"] == right_meta["step"] == 2
    assert left_meta["time_s"] == right_meta["time_s"]

    left_report = json.loads((continued / "report.json").read_text())
    right_report = json.loads((uninterrupted / "report.json").read_text())
    prefix_report = json.loads((prefix / "report.json").read_text())
    assert prefix_report["acceptance"]["accepted_intervals"] == 1
    assert prefix_report["acceptance"]["accepted_time_s"] == (
        4.0e-10 if coupling == "wr" else 1.0e-12
    )
    assert left_report["acceptance"]["accepted_intervals"] == 2
    assert right_report["acceptance"]["accepted_intervals"] == 2
    assert left_report["acceptance"]["accepted_time_s"] == (
        8.0e-10 if coupling == "wr" else 2.0e-12
    )
    assert (
        left_report["acceptance"]["accepted_time_s"]
        == right_report["acceptance"]["accepted_time_s"]
    )
    assert left_report["advance"][-1]["tag"] == (
        right_report["advance"][-1]["tag"]
    )
    assert _stable_coupling_metadata(
        left_report["advance"][-1]["coupling_metadata"]
    ) == _stable_coupling_metadata(
        right_report["advance"][-1]["coupling_metadata"]
    )

    left_ckpt = _load_checkpoint(
        continued / "checkpoints" / "restart.npz"
    )
    right_ckpt = _load_checkpoint(
        uninterrupted / "checkpoints" / "restart.npz"
    )
    for name in CHECKPOINT_IDENTITY:
        assert np.asarray(left_ckpt[name]).item() == np.asarray(
            right_ckpt[name]
        ).item(), name
    left_cumulative = json.loads(
        str(np.asarray(left_ckpt["cumulative_json"]).item())
    )
    right_cumulative = json.loads(
        str(np.asarray(right_ckpt["cumulative_json"]).item())
    )
    _nested_close(left_cumulative, right_cumulative, "cumulative")
    _close(
        left_ckpt["baseline_divB"],
        right_ckpt["baseline_divB"],
        "baseline_divB",
    )


def _context_for_rollback(tmp_path, coupling, device):
    args = _arguments(
        tmp_path / f"rollback-{coupling}-{device.replace(':', '-')}",
        coupling,
        device,
        2,
        2,
    )
    context, metadata = runner.make_context(args)
    base, _ = runner.initial_state(context, metadata)
    seeded, _ = runner.apply_default_perturbation(base, context)
    cumulative = cumulative_record()
    captured = {}
    committed = stage.initialize(
        seeded, context, cumulative, False, captured
    )
    first = time_module.advance(committed, cumulative, context)
    assert first.accepted and first.interval.accepted
    capture = _capture_committed(context, committed, cumulative)
    return context, committed, cumulative, capture


@pytest.mark.parametrize("coupling", ("synchronous", "wr"))
def test_real_polygon_trial_failure_after_two_evaluations_rolls_back(
    tmp_path, coupling, device, monkeypatch
):
    context, committed, cumulative, capture = _context_for_rollback(
        tmp_path, coupling, device
    )

    if coupling == "synchronous":
        original = time_module.evaluate_stage
        calls = 0

        def injected(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise Failure(
                    "Injected",
                    "synchronous failure after two genuine evaluations",
                )
            return original(*args, **kwargs)

        monkeypatch.setattr(time_module, "evaluate_stage", injected)
    else:
        original = massless_field.evaluate
        calls = 0

        def injected(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise Failure(
                    "Injected",
                    "WR failure after two genuine field evaluations",
                )
            return original(*args, **kwargs)

        monkeypatch.setattr(massless_field, "evaluate", injected)

    with pytest.raises(Failure) as raised:
        time_module.advance(committed, cumulative, context)
    failure = raised.value

    assert calls >= 3
    assert failure.code == "Injected"
    assert failure.stage is not None
    assert getattr(failure, "partial_arrays", {})
    _assert_capture_unchanged(context, committed, cumulative, capture)
    if coupling == "wr":
        metadata = getattr(failure, "coupling_metadata", None)
        assert metadata is not None
        assert metadata["rollback"] is True
        assert "work" in metadata
        assert "sweeps" in metadata
    else:
        assert getattr(failure, "coupling_metadata", None) is None
