"""Run explicitly inside an owned Slurm allocation with FOMHD_TEST_GPU=1."""
import os
import numpy as np
import pytest
from fomhd.backend import Backend
from fomhd.geometry import build_rectangle
from fomhd.operators import build_operators
from fomhd.types import Failure

pytestmark=pytest.mark.skipif(os.environ.get('FOMHD_TEST_GPU')!='1',reason='explicit Slurm GPU test not requested')

@pytest.fixture(scope='module')
def gpu():
    assert os.environ.get('SLURM_JOB_ID'), 'GPU tests require an owned Slurm allocation'
    return Backend('cuda:0')


def test_warp_kernel_and_nonreal_rejection(gpu):
    cp=gpu.xp
    with cp.cuda.Stream(non_blocking=True):
        x=gpu.array([1+2j,3-4j,5+6j]);y=gpu.array([2-1j,-3+2j,1j])
        out=gpu.axpy(x,y,.25)
        np.testing.assert_allclose(gpu.to_host(out),[1.5+1.75j,2.25-3.5j,5+6.25j],rtol=0,atol=1e-15)
        for values in ([1+0j,2+1e-4j],[complex(np.nan,0),1],[1,complex(np.inf,0)]):
            with pytest.raises(Failure):gpu.physical_samples(gpu.array(values),1e-12)
        assert np.all(gpu.to_host(gpu.physical_samples(gpu.array([1,2+1e-16j]),1e-12)).imag==0)


def test_gpu_fft_and_rank_checked_solve(gpu):
    x=gpu.array([.1+.2j,3.,.1-.2j])
    np.testing.assert_allclose(gpu.to_host(gpu.forward_fft(gpu.inverse_fft(x,1,1,5),1,1,5)),gpu.to_host(x),rtol=1e-14,atol=1e-14)
    A=gpu.array([[2+1j,1-.5j],[.2j,3-2j]])
    b=gpu.array([1-2j,2+.3j]);sc=gpu.real_array([1,1])
    out=gpu.linear(A,b,sc,sc,1e-12)
    assert gpu.scalar(gpu.xp.max(gpu.xp.abs(A@out.x-b)))<1e-13
    with pytest.raises(Failure):gpu.factor(gpu.array([[1,2],[2,4]]))


def test_gpu_compatible_operators_match_cpu(gpu):
    g=build_rectangle(4,4,1.,0.,.2,.2)
    cpu=build_operators(g,1,backend=Backend('cpu'))
    op=build_operators(g,1,backend=gpu)
    for key in ['I','H','T','Bc','QB','DR','DZ']:
        np.testing.assert_allclose(gpu.to_host(getattr(op,key)),getattr(cpu,key),rtol=2e-12,atol=2e-12)
    for key in ['C','Cdag','DB','DJ','G','Lphi']:
        for x,y in zip(getattr(op,key),getattr(cpu,key)):
            np.testing.assert_allclose(gpu.to_host(x),y,rtol=2e-12,atol=2e-12)
