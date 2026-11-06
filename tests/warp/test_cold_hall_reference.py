"""Independent identities and finite-frequency cold-wall reference checks."""
import numpy as np
import pytest

from tools.cold_hall_reference import (
    ReferenceCase, basis_fields, disk_quadrature, solve_reference,
)


@pytest.mark.parametrize("mode", [0, 1])
def test_potential_curl_is_solenoidal_and_preserves_wall_trace(mode):
    case = ReferenceCase(mode=mode, order=3)
    theta = 2*np.pi*(np.arange(101)+0.37)/101
    R = case.R0 + case.a*np.cos(theta)
    Z = case.a*np.sin(theta)
    fields = basis_fields(case, R, Z)
    normal = np.stack((np.cos(theta),np.zeros_like(theta),np.sin(theta)),axis=1)
    assert np.max(np.abs(fields['div_b'])) < 1e-12
    assert np.max(np.abs(np.einsum('qi,qij->qj',normal,fields['b']))) < 1e-12
    # The rational enrichment supplies nonzero, constant R*b_phi at the wall.
    np.testing.assert_allclose(R*fields['b'][:,1,-1],2.0,atol=2e-14)
    assert np.max(np.abs(np.einsum('qi,qij->qj',normal,fields['current'][:,:,-1:]))) < 1e-7


def test_physical_quadrature_volume_and_zero_perturbation_flux():
    case=ReferenceCase(mode=0,order=3)
    R,Z,w=disk_quadrature(case,18,80)
    np.testing.assert_allclose(np.sum(w),case.R0*np.pi*case.a**2,rtol=1e-14)
    fields=basis_fields(case,R,Z)
    flux=np.sum(fields['b'][:,1,:]*(w/R)[:,None],axis=0)
    assert np.max(np.abs(flux))<1e-12


@pytest.mark.parametrize("mode",[0,1])
def test_reference_has_finite_compressional_mode_and_energy_structure(mode):
    result=solve_reference(ReferenceCase(mode=mode,order=3))
    assert result['omega_rad_s']>0
    assert result['parallel_magnetic_fraction']>.5
    assert result['skew_relative_defect']<1e-12
    assert result['relative_eigen_residual']<1e-10
    assert result['independent_wall_current_relative_defect']<1e-8
    assert result['model']=='cold_hall_constrained_weak_wall'
    assert result['time_convention']=='exp(+i*omega*t)'
