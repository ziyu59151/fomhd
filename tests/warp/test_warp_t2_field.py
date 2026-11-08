"""Independent Hall phase and source checks for the bounded T2-E runner."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_RUNNER = None


def runner():
    global _RUNNER
    if _RUNNER is None:
        path = Path(__file__).resolve().parents[2] / "tools/warp_t2_field.py"
        assert path.is_file(), "scratch T2-E evidence runner is missing"
        spec = importlib.util.spec_from_file_location("scratch_warp_t2_field", path)
        _RUNNER = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_RUNNER)
        assert Path(_RUNNER.__file__).resolve() == path
    return _RUNNER


class FakeProducer:
    diagonal_normal = False

    def __init__(self, error, fail_at_call):
        self.error = error
        self.fail_at_call = fail_at_call
        self.calls = 0
        self.work = {"rhs_calls": 0}

    def rhs(self, p, z):
        self.calls += 1
        self.work["rhs_calls"] += 1
        if self.calls == self.fail_at_call:
            raise self.error
        zero = np.zeros_like(z)
        return SimpleNamespace(
            rhs=zero,
            checks=SimpleNamespace(),
            electric=zero,
            current=zero,
            velocity=zero,
        )

    def workspace_estimate(self, batch):
        return {"total": 0}


def injected_failure_run(tmp_path, monkeypatch, *, fail_at_call):
    r = runner()
    error = r.Failure("InjectedFailure", "injected producer failure")
    error.nodes = np.array([[[1.0 + 2.0j, 3.0 + 4.0j]]])
    producer = FakeProducer(error, fail_at_call)
    context = SimpleNamespace(
        op=SimpleNamespace(Nc=1, Nb=1),
        p=SimpleNamespace(K=0),
        N=1,
    )
    monkeypatch.setattr(
        r,
        "periodic_inputs",
        lambda *args, **kwargs: (
            context,
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            np.ones((1, 2)),
        ),
    )
    monkeypatch.setattr(r, "FieldBatch", lambda *args, **kwargs: producer)
    config = dict(unit="E2", device="cpu", method="serial", points=9, M=2,
                  amplitude=.02, nonuniform=.1, h_factor=1., warmups=0,
                  repetitions=1)
    result = r.run(config, tmp_path / "case")
    return result, json.loads((tmp_path / "case/report.json").read_text())


def assert_failure_report(tmp_path, expected_scope):
    report = json.loads((tmp_path / "case/report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_code"] == "InjectedFailure"
    assert report["failure_message"] == "injected producer failure"
    assert report["attempted_work_scope"] == expected_scope
    assert report["attempted_rhs_batches"] == 1
    assert report["attempted_rhs_vectors"] == 1
    assert report["attempted_producer_work"] == {"rhs_calls": 1}
    assert report["host_peak_rss_bytes"] > 0
    traceback_text = (tmp_path / "case/traceback.txt").read_text()
    assert "injected producer failure" in traceback_text
    assert "raise self.error" in traceback_text
    with np.load(tmp_path / "case/failure-window.npz") as arrays:
        np.testing.assert_array_equal(
            arrays["nodes"], np.array([[[1.0 + 2.0j, 3.0 + 4.0j]]])
        )


def test_oracle_failure_records_attempted_work_and_original_error(tmp_path, monkeypatch):
    _, report = injected_failure_run(tmp_path, monkeypatch, fail_at_call=1)
    assert report["attempted_work_scope"] == "oracle"
    assert_failure_report(tmp_path, "oracle")


def test_timed_failure_records_reset_counts_and_original_error(tmp_path, monkeypatch):
    _, report = injected_failure_run(tmp_path, monkeypatch, fail_at_call=10)
    assert report["attempted_work_scope"] == "timed"
    assert_failure_report(tmp_path, "timed")


def test_linear_case_preserves_hall_polarization_and_safe_step():
    case = runner().linear_inputs(33, 17)
    l, h, initial = case["L"], case["h"], case["start"]
    assert h*np.max(np.abs(np.linalg.eigvals(l))) <= .050000001
    vector = initial[1]
    np.testing.assert_allclose(l[1] @ vector, 1j*case["beta"][1]*vector, atol=1e-18)
    assert case["forcing"].shape == (17,4,33,4)


def test_linear_affine_tree_matches_independent_four_stage_oracle():
    r = runner()
    case = r.linear_inputs(9, 17, manufactured=True)
    actual = r.linear_solve(case, "prefix", np)
    expected = r.linear_solve(case, "serial", np)
    for a,b in zip(actual,expected):
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-15)
    assert np.max(np.abs(case["forcing"][:,1]-case["forcing"][:,2])) > 0


def test_periodic_nonuniform_field_uses_real_production_rhs():
    r = runner()
    c,p,z,background = r.periodic_inputs(9,.02,.2,"cpu")
    from fomhd.field_batch import FieldBatch
    from massless_verification_fixture import stage_tag
    from fomhd.massless import evaluate_stage
    state = r.unpack_state(c,p,z)
    production = evaluate_stage(state,stage_tag(state),c)
    actual = FieldBatch(c).rhs(p[None],z[None])
    expected = r.pack_state(c,production.rhs)[1]
    np.testing.assert_allclose(actual.rhs[0],expected,rtol=1e-10,atol=1e-6)
    assert np.max(np.abs(actual.electric)) > 0


def test_linear_run_writes_readable_report_and_full_waveform(tmp_path):
    config=dict(unit="E1",device="cpu",method="prefix",points=9,M=3,
                manufactured=True,h_factor=1.,warmups=0,repetitions=1)
    result=runner().run(config,tmp_path/"case")
    stored=json.loads((tmp_path/"case/report.json").read_text())
    assert stored["correct"] is True
    assert stored["host_peak_rss_bytes"] > 0
    with np.load(tmp_path/"case/arrays.npz") as arrays:
        assert arrays["nodes"].shape==(3,9,4)
        np.testing.assert_allclose(arrays["nodes"],arrays["oracle_nodes"],rtol=1e-12,atol=1e-15)


def test_smooth_closed_seed_has_same_physical_profile_and_full_constraints():
    from test_massless_support import context
    r=runner()
    assert hasattr(r,"smooth_closed_seed"),"smooth wall-compatible seed is missing"
    for n in (6,8):
        c=context(K=1,hard_diagnostics=False,nodes_r=n,nodes_z=n)
        b=r.smooth_closed_seed(c,.02)
        nb=c.op.Nb
        values=b.reshape(3,nb)
        np.testing.assert_allclose(values[0],values[2].conj(),rtol=1e-12,atol=1e-18)
        for stored in range(3):
            div=c.op.DB[stored]@values[stored]
            current=c.op.Cdag[stored]@values[stored]
            from fomhd.wall_cache import get_wall_cache
            wall=get_wall_cache(c).modes[stored].extraction@current
            assert np.max(np.abs(div))<1e-15
            assert np.max(np.abs(wall))<1e-15
        nodal=(c.op.QB@values[1]).reshape(3,c.op.Nc)
        assert np.min(np.linalg.norm(nodal,axis=0))>.2*c.p.ref.magnetic


def test_closed_profiles_are_anchored_to_fixed_physical_domain():
    r=runner()
    for points in (6,8):
        c,p,z,_=r.closed_inputs(points,.02,.1,"cpu")
        nc=c.op.Nc
        radial=np.asarray(c.geometry.cells.R)
        axial=np.asarray(c.geometry.cells.Z)
        profile=np.cos(2*np.pi*(radial-1.)/.6)*np.cos(2*np.pi*axial/.6)
        expected=c.p.ref.ni*(1+.1*profile)
        np.testing.assert_allclose(p[c.p.K,:nc],expected,rtol=2e-15,atol=0)


def test_periodic_high_mode_is_explicit_not_confused_with_grid_frequency():
    c,p,z,_=runner().periodic_inputs(33,.02,.1,"cpu",spatial_mode=16)
    nc=c.op.Nc
    coefficients=np.fft.fft(z[0,2*nc:3*nc])/nc
    assert abs(coefficients[16]-.001)<1e-16
    assert abs(coefficients[1])<1e-16
