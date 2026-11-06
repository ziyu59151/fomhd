"""Production linearization identity and constraint regressions."""
import importlib

import numpy as np
import pytest

from tools.production_spectrum import constrained_basis, compute, direction, pack_rhs, shifted
from tools.warp_tokamak_cae import make_context, background, host
from fomhd import stage
from fomhd.types import StageTag


def test_basis_and_actual_directional_rhs(tmp_path):
    result=compute('circle',4,0,0,tmp_path/'spectrum')
    data=np.load(tmp_path/'spectrum/spectrum.npz')
    context,meta,eq=make_context('circle',4,0,'cpu')
    initial,_,_=background(context,meta,eq)
    V,B=constrained_basis(context,0)
    assert np.max(abs(context.op.T@V))<1e-12
    assert np.max(abs(context.op.DB[0]@B))<1e-11
    nc=context.op.Nc
    vector=np.sin(np.arange(result['size'])+.1)
    fields=direction(vector,context,V,B)
    tag=StageTag(0,1,1,0,0.,'spectrum_test')
    epsilon=2e-6
    plus=stage.evaluate_stage(shifted(initial,fields,epsilon,context,0),tag,context,False)
    minus=stage.evaluate_stage(shifted(initial,fields,-epsilon,context,0),tag,context,False)
    measured=(pack_rhs(plus,context,V,B,0)-pack_rhs(minus,context,V,B,0))/(2*epsilon)
    predicted=data['jacobian']@vector
    assert np.linalg.norm(measured-predicted)/np.linalg.norm(measured)<1e-7
    assert result['spectral_radius_rad_s']>0
    assert len(result['candidates'])>1
    assert result['candidates'][0]['frequency_hz']<1e3
    assert result['source_hashes']
    assert all(len(value)==64 for value in result['source_hashes'].values())
    count=len(result['candidates'])
    nc=context.op.Nc
    assert data['candidate_indices'].shape==(count,)
    assert data['candidate_frequency_hz'].shape==(count,)
    assert data['candidate_magnetic_transfer_ratio'].shape==(count,)
    assert data['candidate_MB_magnetic_norm'].shape==(count,)
    assert data['candidate_M0_ion_kinetic_norm'].shape==(count,)
    assert data['candidate_velocity'].shape==(count,3,nc)
    assert data['candidate_B_node'].shape==(count,3,nc)
    assert data['candidate_density'].shape==(count,nc)
    assert data['background_B_node'].shape==(3,nc)
    assert data['R'].shape==data['Z'].shape==data['M0'].shape==(nc,)
    assert data['wall_R'].shape==data['wall_Z'].shape
    assert float(data['rho'])>0 and float(data['mu0'])>0
    nodal_magnetic=np.sum(np.abs(data['candidate_B_node'])**2*data['M0'][None,None,:],axis=(1,2))/float(data['mu0'])
    np.testing.assert_allclose(
        nodal_magnetic,
        data['candidate_magnetic_transfer_ratio']*data['candidate_MB_magnetic_norm'],
        rtol=1e-12,atol=1e-18,
    )
    for item in result['candidates']:
        assert np.isfinite(item['magnetic_transfer_ratio'])
        assert item['MB_magnetic_norm']>=0
        assert item['M0_ion_kinetic_norm']>=0


def test_compute_forwards_positive_memory_limit_keyword(tmp_path,monkeypatch):
    production=importlib.import_module('tools.production_spectrum')
    observed={}
    def capture(*args,**kwargs):
        observed.update(args=args,kwargs=kwargs)
        raise RuntimeError('stop after context argument capture')
    monkeypatch.setattr(production,'make_context',capture)
    with pytest.raises(RuntimeError,match='argument capture'):
        production.compute('circle',4,0,0,tmp_path/'spectrum',memory_limit_bytes=7<<30)
    assert observed['kwargs']['memory_limit_bytes']==7<<30


def test_k1_n1_background_export_uses_retained_magnetic_sector(tmp_path):
    output=tmp_path/'spectrum'
    result=compute('circle',4,1,1,output)
    context,meta,eq=make_context('circle',4,1,'cpu')
    initial,_,_=background(context,meta,eq)
    baseline=stage.evaluate_stage(
        initial,StageTag(0,1,1,0,0.,'k1_n1_background_export'),context,False
    )
    nc=context.op.Nc
    K=context.p.K
    retained=host(initial.B,context)[K*context.op.Nb:(K+1)*context.op.Nb]
    expected=(host(context.op.QB,context)@retained).reshape(3,nc)
    zero_fourier=host(baseline.BN,context).reshape(5,3,nc).mean(axis=0)
    with np.load(output/'spectrum.npz',allow_pickle=False) as data:
        assert data['background_B_node'].shape==(3,nc)
        np.testing.assert_allclose(data['background_B_node'],expected,rtol=0,atol=1e-12)
        np.testing.assert_allclose(data['background_B_node'],zero_fourier,rtol=0,atol=1e-12)
        count=len(result['candidates'])
        assert data['candidate_B_node'].shape==(count,3,nc)
        assert data['candidate_velocity'].shape==(count,3,nc)
        assert result['selected']['magnetic_transfer_ratio']==float(
            data['candidate_magnetic_transfer_ratio'][
                result['candidates'].index(result['selected'])
            ]
        )
