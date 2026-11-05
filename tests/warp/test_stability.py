"""Analytical guards catch a false imaginary-axis CFL and unit mistakes."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest


def module():
    path = Path(__file__).parents[2] / 'src' / 'fomhd' / 'stability.py'
    assert path.is_file(), 'stability guard has not been implemented'
    spec=importlib.util.spec_from_file_location('stability', path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_midpoint_growth_has_no_nonzero_imaginary_stable_interval():
    s=module()
    for x in [.001,.1,1,2]:
        z=s.amplification('midpoint',1j*x)
        assert abs(z)**2 == pytest.approx(1+x**4/4,rel=2e-15)
        m=s.temporal_metrics('midpoint',1.,x,1000)
        assert m['log_amplitude_error'] > 0
    assert not s.assess_dt('midpoint',1.,.2,3142)['accepted']


def test_rk4_boundary_and_finite_horizon_phase():
    s=module()
    assert abs(s.amplification('rk4',1j*2*np.sqrt(2))) == pytest.approx(1.)
    assert abs(s.amplification('rk4',3j)) > 1
    x=.05;steps=10000
    result=s.temporal_metrics('rk4',1.,x,steps)
    numerical=s.amplification('rk4',1j*x)**steps
    assert result['amplitude_relative_error'] == pytest.approx(abs(abs(numerical)-1),abs=1e-12)
    expected_phase=steps*(np.angle(s.amplification('rk4',1j*x))-x)
    assert result['phase_error_rad'] == pytest.approx(abs(expected_phase),abs=1e-13)
    assert s.assess_dt('midpoint',1.,2*np.pi/256,256*100)['accepted']
    assert not s.assess_dt('midpoint',1.,2*np.pi/32,32*100)['accepted']


def test_frequency_units_and_massless_hall_limit():
    s=module()
    f=s.frequency_scales(ne=1e19,ni=1e19,B=3.,mi=3.3435837768e-27,k=100.)
    assert f['fpe_Hz'] == pytest.approx(28.393e9,rel=2e-5)
    assert f['fci_Hz'] == pytest.approx(22.87913341158855e6,rel=1e-14)
    assert f['omega_whistler_finite'] < f['omega_ce']
    doubled=s.frequency_scales(ne=1e19,ni=1e19,B=3.,mi=3.3435837768e-27,k=200.)
    assert doubled['omega_whistler_massless'] == pytest.approx(4*f['omega_whistler_massless'])
    dense=s.frequency_scales(ne=4e19,ni=4e19,B=3.,mi=3.3435837768e-27,k=100.)
    assert dense['omega_pe'] == pytest.approx(2*f['omega_pe'])
    for args in [dict(ne=0),dict(B=-1),dict(k=np.nan),dict(me=0)]:
        with pytest.raises(ValueError): s.frequency_scales(**args)


def test_invalid_guard_configuration_is_rejected():
    s=module()
    for method,omega,dt,steps in [('implicit',1.,1.,1),('rk4',-1.,1.,1),('rk4',1.,0.,1),('rk4',1.,1.,-1)]:
        with pytest.raises(ValueError): s.assess_dt(method,omega,dt,steps)
    assert s.assess_dt('midpoint',0.,1.,0)['accepted']


def test_stage_frequency_estimates_follow_the_selected_model():
    from types import SimpleNamespace as NS
    from foundation_fixture import parameters
    from fomhd.backend import Backend
    from fomhd.types import Context

    p = parameters(1)
    backend = Backend('cpu')
    geometry = NS(cells=NS(R=np.array([1., 2.])))
    op = NS(Nc=2, DR=np.array([[1., -1.]]), DZ=np.array([[2., -2.]]))
    context = Context(p, geometry, op, backend, p.Nphi)
    fluids = []
    for number in (p.ref.ni, p.ref.ne):
        fluids.append(NS(n=np.full(p.Nphi*2, number),
                         u=np.zeros(p.Nphi*6),
                         perp=np.full(p.Nphi*2, p.ref.pressure),
                         parallel=np.full(p.Nphi*2, p.ref.pressure)))
    bn = np.zeros((p.Nphi, 3, 2)); bn[:, 1] = p.ref.magnetic
    stage = NS(fluid=fluids, BN=bn.reshape(-1))
    s = module()
    finite = s.stage_frequency_estimate(stage, context)
    assert finite['electron_inertia'] == 1
    assert finite['omega_estimate'] >= finite['omega_plasma']
    assert finite['omega_estimate'] >= finite['omega_ce']
    p.electron_inertia = 0
    massless = s.stage_frequency_estimate(stage, context)
    assert massless['active_limits'] == ['hall', 'ion_cyclotron', 'alfven', 'bulk_pressure', 'advection']
    assert massless['omega_estimate'] >= massless['omega_whistler_massless']
    # Low density and fine resolution can make the massless Hall limit stricter.
    assert massless['omega_estimate'] > finite['omega_estimate']
    assert massless['omega_plasma'] == finite['omega_plasma']
    op.DR *= 2; op.DZ *= 2
    finer = s.stage_frequency_estimate(stage, context)
    assert finer['omega_whistler_massless'] > massless['omega_whistler_massless']


def test_frequency_guard_rejects_and_warn_override_is_explicit():
    from types import SimpleNamespace as NS
    from foundation_fixture import parameters

    s = module()
    p = parameters(0)
    p.dt = 1.; p.nsteps = 100
    context = NS(p=p)
    estimate = {'omega_estimate': 1., 'electron_inertia': 1}
    report = s.assess_runtime(estimate, context, step=0)
    assert not report['accepted']
    assert report['action'] == 'error'
    p.dt_guard = 'warn'
    warning = s.assess_runtime(estimate, context, step=0)
    assert not warning['accepted']
    assert warning['action'] == 'warn'
    assert warning['estimate'] == estimate


def test_real_time_transaction_guard_and_warn_override():
    from fomhd.stage import initialize
    from fomhd.time import advance
    from fomhd.types import Failure
    from workflow_fixture import cumulative
    from foundation_fixture import parameters
    from fomhd.backend import Backend
    from fomhd.geometry import build_rectangle
    from fomhd.operators import build_operators
    from fomhd.types import Context
    from fomhd.stage import closed_reference

    p = parameters(1)
    backend = Backend('cpu')
    geometry = build_rectangle(4, 4, 1., 0., .2, .2)
    context = Context(p, geometry, build_operators(geometry, 1, backend=backend), backend, p.Nphi)
    initial = closed_reference(context, p.ref.ni, p.ref.pressure, p.ref.pressure, p.ref.magnetic)
    context.p.integrator = "midpoint"
    context.p.nsteps = 1_000_000
    ledger = cumulative()
    committed = initialize(initial, context, ledger, False)
    old_state = committed.state
    with pytest.raises(Failure) as error:
        advance(committed, ledger, context)
    assert error.value.code == 'TimeStepGuard'
    assert committed.state is old_state and ledger.accepted_intervals == 0
    assert not context.last_dt_report['accepted']
    context.p.dt_guard = 'warn'
    result = advance(committed, ledger, context)
    assert committed.state.step == 1 and ledger.accepted_intervals == 1
    assert result.time_step_report['action'] == 'warn'
    record = next(c for c in result.checks if c.name == 'time_step_estimate')
    assert not record.passed and not record.hard
