"""Physical configuration and initial normalization checks."""
import numpy as np
import pytest
from tools.warp_tokamak_cae import circular_polygon, physical_parameters, encode_mode


def test_massless_physical_registry_observes_power_but_keeps_constraints():
    p=physical_parameters(0,3.423e18,3.4,1.,230.)
    assert p.electron_inertia==0 and p.Nphi==5 and p.integrator=='rk4'
    assert p.species[0].mass>3e-27
    assert not p.budgets['electron_pressure_adjoint.electron.value'].hard
    assert not p.budgets['electron_pressure_adjoint.linf'].hard
    assert p.budgets['wall_current.linf'].hard
    assert p.budgets['quasineutrality.linf'].hard


def test_mode_embedding_and_real_zero():
    np.testing.assert_allclose(encode_mode(np.array([2.]),0,1),[0,2,0])
    np.testing.assert_allclose(encode_mode(np.array([2j]),1,1),[-1j,0,1j])
    with pytest.raises(ValueError):encode_mode(np.array([2j]),0,0)
    p=circular_polygon()
    assert len(p)==128 and p[:,0].min()==pytest.approx(2)


@pytest.fixture(scope='module')
def discrete_case(tmp_path_factory):
    from tools.production_spectrum import compute
    from tools.warp_tokamak_cae import make_context,background
    root=tmp_path_factory.mktemp('discrete_case')
    report=compute('circle',4,0,0,root/'spectrum')
    # This intentionally names a fast branch; it does not exclude low candidates.
    freq=next(c['frequency_hz'] for c in report['candidates'] if c['frequency_hz']>1e6)
    context,meta,eq=make_context('circle',4,0,'cpu');state,_,_=background(context,meta,eq)
    return root,context,meta,state,freq


def test_resolved_spectrum_validates_actual_background(discrete_case):
    from tools.warp_tokamak_cae import load_discrete_spectra
    root,c,m,state,freq=discrete_case
    entries,choice=load_discrete_spectra([root/'spectrum'],c,m,0,freq,state)
    assert entries and choice['frequency_hz']==pytest.approx(freq)
    changed=state.fluid[0].n.copy();state.fluid[0].n[0]*=1.0001
    try:
        with pytest.raises(ValueError,match='background identity'):
            load_discrete_spectra([root/'spectrum'],c,m,0,freq,state)
    finally:state.fluid[0].n=changed
    old=c.p.species[0].mass;c.p.species[0].mass*=2
    try:
        with pytest.raises(ValueError,match='references/geometry identity'):
            load_discrete_spectra([root/'spectrum'],c,m,0,freq,state)
    finally:c.p.species[0].mass=old


def test_paired_run_preserves_initial_checks_and_raw_energies(discrete_case):
    import json
    from types import SimpleNamespace
    from tools.warp_tokamak_cae import run
    root,c,m,state,freq=discrete_case
    args=SimpleNamespace(kind='circle',nodes=4,mode=0,K=0,device='cpu',equilibrium=None,
                         reference_order=3,control_only=False,spectrum=[root/'spectrum'],seed_frequency=freq,
                         amplitude=1e-4,cycles=8.,horizon=.05/freq,steps=4,dt_divisor=1,wall_seconds=60.,output=root/'paired')
    assert run(args)==0
    report=json.loads((args.output/'report.json').read_text())
    assert report['target_periods']==pytest.approx(.05)
    history=json.loads((args.output/'perturbed-history.json').read_text())
    assert 'magnetic_J' in history[-1]['energy_raw']
    assert history[-1]['energy_raw']['kinetic_e_J']==0
    checks=[json.loads(line) for line in (args.output/'perturbed-checks.jsonl').read_text().splitlines()]
    assert any(r['stage']=='initial' for r in checks)
    assert (args.output/'perturbed-initial-rhs.npz').is_file()
    a=np.load(args.output/'perturbed-signals.npz');b=np.load(args.output/'control-signals.npz')
    np.testing.assert_array_equal(a['time_s'],b['time_s'])
    args.output=root/'excessive-dt';args.steps=1;args.horizon=1/freq
    assert run(args)==2
    failed=json.loads((args.output/'report.json').read_text())
    assert failed['control']['accepted_steps']==0
    assert 'stability guard' in failed['control']['failures'][0]['message']


def test_larger_grid_keeps_memory_gate_and_validates_cap():
    from tools.warp_tokamak_cae import make_context
    from fomhd.types import Failure
    with pytest.raises(ValueError,match='positive integer'):
        make_context('circle',32,0,'cpu',memory_limit_bytes=0)
    with pytest.raises(ValueError,match='nodes in'):
        make_context('circle',41,0,'cpu')
    with pytest.raises(Failure,match='memory'):
        make_context('circle',32,0,'cpu',memory_limit_bytes=1)


def test_050_grid40_reaches_geometry_under_unchanged_memory_cap(monkeypatch):
    from tools import warp_tokamak_cae as cae

    class GeometryReached(RuntimeError):
        pass

    def stop_before_allocation(*args, **kwargs):
        raise GeometryReached("bounded grid validation passed")

    monkeypatch.setattr(cae, "build_geometry", stop_before_allocation)
    with pytest.raises(GeometryReached):
        cae.make_context("circle", 40, 1, "cpu", memory_limit_bytes=8 << 30)
    with pytest.raises(ValueError, match="nodes"):
        cae.make_context("circle", 41, 1, "cpu", memory_limit_bytes=8 << 30)
