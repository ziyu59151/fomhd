"""Production linearization identity and constraint regressions."""
import numpy as np
from tools.production_spectrum import constrained_basis, compute, direction, pack_rhs, shifted
from tools.warp_tokamak_cae import make_context, background
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
