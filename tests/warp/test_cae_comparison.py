"""Frequency aggregation and complex control comparison regressions."""
import numpy as np
from tools.plots.plot_cae_comparison import summarize_probes,common_difference


def test_unavailable_probes_are_not_used_as_frequencies():
    a=summarize_probes({'mode_fits':[{'fit':{'available':False,'frequency_hz':99,'reason':'short'}},
       {'fit':{'available':True,'frequency_hz':10,'relative_residual':.01,'measured_cycles':8}},
       {'fit':{'available':True,'frequency_hz':12,'relative_residual':.02,'measured_cycles':7}}]})
    assert a['frequency_hz']==11 and a['probe_count']==2 and a['minimum_measured_cycles']==7


def test_complex_halfstep_and_halfamplitude_require_real_matching_cadence():
    times=np.arange(5.)
    a={'time_s':times,'delta_B':np.ones((5,3,2))+2j}
    b={'time_s':np.arange(9.)/2,'delta_B':np.ones((9,3,2))*.5+1j}
    assert common_difference(a,b,np.array([2.,3.]),stride=2,amplitude_scale=2)['relative_history_L2']==0
    b['time_s'][2]+=.01
    assert not common_difference(a,b,np.array([2.,3.]),stride=2,amplitude_scale=2)['available']


def test_backend_pairing_rejects_changed_physics_or_seed():
    from copy import deepcopy
    from tools.plots.plot_cae_comparison import backend_pairing
    report={'source_hashes':{'src/fomhd/core.py':'abc'},'amplitude':1e-4,
            'selected_frequency_hz':10.,'dt_s':.01,'case':{'kind':'circle','nodes':12,'K':0,'Nphi':5,
            'background_mapping':{'max_DB':1e-14}}}
    resolved={'electron_inertia':0,'dt':.01,'nsteps':100}
    other=deepcopy(report);other['case']['background_mapping']['max_DB']=2e-14
    assert backend_pairing(report,other,resolved,resolved)['available']
    for field in ('amplitude','selected_frequency_hz','dt_s'):
        changed=deepcopy(other);changed[field]*=2
        assert not backend_pairing(report,changed,resolved,resolved)['available']
    changed=deepcopy(other);changed['source_hashes']['src/fomhd/core.py']='def'
    assert not backend_pairing(report,changed,resolved,resolved)['available']
    assert not backend_pairing(report,other,resolved,dict(resolved,electron_inertia=1))['available']
    assert not backend_pairing({},other,resolved,resolved)['available']
