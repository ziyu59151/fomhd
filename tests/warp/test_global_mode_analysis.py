"""Independent real-mode and Fourier amplitude regressions."""
import numpy as np
import pytest
from tools.global_mode_analysis import fit_real_mode, matched_difference, physical_mode_coefficients


def test_real_zero_mode_with_offset_trend_and_growth():
    t=np.linspace(0,8e-6,1025)
    y=4+2e4*t+np.exp(-2e4*t)*(3*np.cos(2*np.pi*1e6*t)+.4*np.sin(2*np.pi*1e6*t))
    r=fit_real_mode(t,y)
    assert r['available']
    assert r['frequency_hz']==pytest.approx(1e6,rel=2e-6)
    assert r['growth_rate_per_s']==pytest.approx(-2e4,rel=2e-4)
    assert r['relative_residual']<1e-6


def test_short_record_and_cadence_are_not_claimed():
    t=np.linspace(0,2e-6,257)
    assert not fit_real_mode(t,np.cos(2*np.pi*1e6*t))['available']
    t[31]+=.1*(t[1]-t[0])
    with pytest.raises(ValueError,match='uniform'):
        fit_real_mode(t,np.cos(2*np.pi*1e6*t))


def test_n0_n1_have_equal_peak_physical_amplitude():
    for mode in (0,1):
        c=physical_mode_coefficients(np.array([.01]),mode)
        modes=np.arange(-mode,mode+1)
        assert sum(c[:,0]*np.exp(1j*modes*0)).real==pytest.approx(.01)
    with pytest.raises(ValueError):
        physical_mode_coefficients(np.array([1j]),0)


def test_background_subtraction_requires_same_times():
    t=np.arange(5.)
    np.testing.assert_array_equal(matched_difference(t,t+3,t,t+2),np.ones(5))
    with pytest.raises(ValueError):
        matched_difference(t,t,t+.1,t)
