"""Independent checks of T2-E window scheduling, powers and failed trials."""

import importlib
import importlib.util

import numpy as np
import pytest

from fomhd.types import Failure


def module():
    assert importlib.util.find_spec("fomhd.time_field") is not None, "T2-E window algorithm is missing"
    return importlib.import_module("fomhd.time_field")


def sequential(power, b, start):
    value = start.copy()
    result = []
    for item in b:
        value = power(1, value[None])[0] + item
        result.append(value.copy())
    return np.stack(result)


@pytest.mark.parametrize("length", [1, 3, 17, 32, 65])
def test_operator_tree_keeps_all_prefixes_and_nonzero_start(length):
    m = module()
    a = np.array([[0.998, -.01], [.01, .998]], dtype=np.complex128)
    power = lambda n, v: v @ np.linalg.matrix_power(a, n).T
    rng = np.random.default_rng(71)
    b = (rng.normal(size=(length, 2)) + 1j*rng.normal(size=(length, 2))).astype(np.complex128)
    start = np.array([.3+1j, -.7j])
    old = b.copy()
    actual = m.operator_prefix(power, b, start, xp=np)
    np.testing.assert_allclose(actual, sequential(power, b, start), rtol=3e-14, atol=3e-14)
    np.testing.assert_array_equal(b, old)
    assert not np.shares_memory(actual, b)


def test_frozen_polynomial_power_matches_full_small_rk4_matrix():
    m = module()
    l = np.array([[.1, -2, .3], [2, -.2, .4], [0, -.1, -.3]], dtype=np.complex128)
    reference = m.PolynomialReference(lambda v: v @ l.T, .01, xp=np, max_degree=30)
    x = .01*l
    a = np.eye(3) + x + x@x/2 + x@x@x/6 + x@x@x@x/24
    vectors = np.array([[1., .2, -.4], [.3, -.7j, .1]], dtype=np.complex128)
    for length in [1, 2, 8, 32]:
        np.testing.assert_allclose(reference.power(length, vectors), vectors @ np.linalg.matrix_power(a, length).T, rtol=3e-14, atol=3e-14)
    assert reference.action_calls > 0
    assert reference.maximum_degree <= 30


def independent_window(p, z, h):
    nodes, stages, rates = [], [], []
    for j in range(len(p)):
        k1 = p[j, 0] + .4*z*z
        y2 = z+h*k1/2
        k2 = p[j, 1]+.4*y2*y2
        y3 = z+h*k2/2
        k3 = p[j, 2]+.4*y3*y3
        y4 = z+h*k3
        k4 = p[j, 3]+.4*y4*y4
        stages.append(np.stack([z, y2, y3, y4]))
        rates.append(np.stack([k1, k2, k3, k4]))
        z = z+h*(k1+2*k2+2*k3+k4)/6
        nodes.append(z.copy())
    return np.stack(nodes), np.stack(stages), np.stack(rates)


def test_real_nonlinear_residual_recovers_same_rk4_without_serial_initialization():
    m = module()
    p = (.1 + np.arange(17*4).reshape(17,4,1,1)*1e-4).astype(np.complex128)
    z0 = np.array([[.2]], dtype=np.complex128)
    old_p, old_z = p.copy(), z0.copy()
    calls = []
    def rhs(p_values, z):
        calls.append(len(z))
        return p_values + .4*z*z
    reference = m.PolynomialReference(lambda v: .16*v, .003, xp=np, max_degree=30)
    result = m.solve_window(rhs, p, z0, .003, reference, xp=np, scales=np.ones_like(z0.real), tolerance=1e-12)
    expected, stages, rates = independent_window(p, z0, .003)
    np.testing.assert_allclose(result.nodes, expected, rtol=0, atol=3e-12)
    np.testing.assert_allclose(result.stages, stages, rtol=0, atol=3e-12)
    np.testing.assert_allclose(result.rates, rates, rtol=0, atol=3e-12)
    assert result.history[-1]["residual"] <= 1e-12
    assert result.residual_evaluations >= 2
    assert calls[0] == 1
    assert calls[-1]==1
    assert all(n == 17 for n in calls[1:-1])
    np.testing.assert_array_equal(p, old_p)
    np.testing.assert_array_equal(z0, old_z)
    assert np.max(np.abs(result.stages[:, 1]-result.stages[:, 2])) > 0


def test_nonconvergence_keeps_accepted_input_and_residual_evidence():
    m = module()
    p = np.ones((17,4,1,1), dtype=np.complex128)
    z0 = np.array([[.2]], dtype=np.complex128)
    old = z0.copy()
    reference = m.PolynomialReference(lambda v: 0*v, .03, xp=np)
    with pytest.raises(Failure) as caught:
        m.solve_window(lambda p,z:p+z*z, p,z0,.03,reference,xp=np,scales=np.ones_like(z0.real),tolerance=1e-15,max_iterations=1)
    np.testing.assert_array_equal(z0, old)
    assert caught.value.code == "FieldWindowConvergence"
    assert caught.value.history


def test_bad_controls_are_rejected_before_rhs_calls():
    m = module()
    def bad_rhs(p,z):
        raise AssertionError("invalid controls reached RHS")
    for h in [0, -1, float("nan")]:
        with pytest.raises((Failure, ValueError)):
            m.solve_window(bad_rhs,np.ones((1,4,1,1),complex),np.ones((1,1),complex),h,None,xp=np,scales=np.ones((1,1)),tolerance=1e-10)


def test_directional_difference_does_not_magnify_constraint_roundoff():
    m=module()
    base=np.array([[1.,0.]],dtype=complex)
    def constrained_rhs(p,z):
        if np.max(np.abs(z[...,1]))>1e-12:
            raise Failure("Constraint","normal direction outside roundoff budget")
        return 2*z
    derivative=m.FrozenJVP(constrained_rhs,base,base,np.ones_like(base.real),xp=np)
    direction=np.array([[[1e-20,1e-20]]],dtype=complex)
    result=derivative(direction)
    assert np.all(np.isfinite(result))
    assert np.max(np.abs(result))<1e-18


def test_numerical_library_failure_is_not_accepted_by_line_search():
    m=module()
    calls=0
    def rhs(p,z):
        nonlocal calls
        calls+=1
        if calls==6:
            raise Failure("NumericalLibrary","injected device/solver error")
        return p+.4*z*z
    p=np.ones((3,4,1,1),complex)*.1
    z=np.array([[.2]],complex)
    reference=m.PolynomialReference(lambda v:.16*v,.01,xp=np)
    with pytest.raises(Failure) as caught:
        m.solve_window(rhs,p,z,.01,reference,xp=np,scales=np.ones((1,1)),tolerance=1e-12)
    assert caught.value.code=="NumericalLibrary"


def test_converged_window_still_checks_final_endpoint_validity():
    m=module()
    def rhs(p,z):
        if np.any(z.real<=0):
            raise Failure("Domain","negative pressure")
        return p
    path=np.array([-1.8,.1,-.1,-10.],complex).reshape(1,4,1,1)
    start=np.array([[1.]],complex)
    endpoint=np.array([[[-.9666666666666668]]],complex)
    reference=m.PolynomialReference(lambda v:0*v,1.,xp=np)
    with pytest.raises(Failure) as caught:
        m.solve_window(rhs,path,start,1.,reference,xp=np,scales=np.ones((1,1)),
                       tolerance=1e-12,initial=endpoint)
    assert caught.value.code=="Domain"
    assert caught.value.history
    assert caught.value.rhs_vectors==5


def test_window_memory_preflight_rejects_before_any_rhs():
    m=module()
    def bad_rhs(p,z):
        raise AssertionError("budget rejection reached numerical producer")
    with pytest.raises(Failure) as caught:
        m.solve_window(bad_rhs,np.ones((17,4,1,1),complex),np.ones((1,1),complex),.01,None,
                       xp=np,scales=np.ones((1,1)),tolerance=1e-10,workspace_limit_bytes=1)
    assert caught.value.code=="MemoryBudget"


def test_selected_stage_field_history_is_retained_without_extra_replay():
    from types import SimpleNamespace
    m=module()
    class RecordingRHS:
        def __call__(self,p,z):
            self.last=SimpleNamespace(electric=3*z,current=2*z,velocity=z.copy())
            return p+.1*z
    rhs=RecordingRHS()
    p=np.ones((3,4,1,1),complex)*.1
    z=np.array([[.2]],complex)
    reference=m.PolynomialReference(lambda v:.1*v,.01,xp=np)
    result=m.solve_window(rhs,p,z,.01,reference,xp=np,scales=np.ones((1,1)),tolerance=1e-12)
    assert hasattr(result,"stage_electric"),"selected field-stage history is missing"
    np.testing.assert_allclose(result.stage_electric,3*result.stages,rtol=1e-15,atol=1e-15)
    assert result.rhs_batches==1+4*result.residual_evaluations+1


@pytest.mark.skipif(__import__('os').environ.get('FOMHD_TEST_GPU')!='1',reason='owned Slurm GPU test required')
def test_cuda_window_matches_cpu_with_distinct_time_and_stage_ion_inputs():
    import os
    assert os.environ.get('SLURM_JOB_ID')
    from tools.warp_t2_field import closed_inputs,CountedRHS,serial_field
    from fomhd.field_batch import FieldBatch
    from fomhd.time_field import FrozenJVP,PolynomialReference,solve_window
    c,p,z,background=closed_inputs(4,.02,.1,'cpu')
    count,nc=5,c.op.Nc
    path=np.broadcast_to(p,(count,4)+p.shape).copy()
    for j in range(count):
        for stage in range(4):
            path[j,stage,:,:nc]*=1+.02*np.sin(.6*j+.4*stage)
            path[j,stage,:,nc:4*nc]*=1+.08*np.cos(.5*j+.7*stage)
    h=.05/(8/(.6/3)**2*c.p.ref.magnetic/(c.p.mu0*(-c.p.species[1].charge)*c.p.ref.ni*.9))
    cpu_rhs=CountedRHS(FieldBatch(c));fields={}
    cpu=serial_field(cpu_rhs,path,z,h,np,field_history=fields)
    gpu_c,_,_,_=closed_inputs(4,.02,.1,'cuda:0')
    xp=gpu_c.compute.xp
    gpu_rhs=CountedRHS(FieldBatch(gpu_c))
    p_gpu,z_gpu,background_gpu=xp.asarray(path),xp.asarray(z),xp.asarray(background)
    derivative=FrozenJVP(gpu_rhs,p_gpu[0,0],z_gpu,background_gpu,xp=xp)
    result=solve_window(gpu_rhs,p_gpu,z_gpu,h,PolynomialReference(derivative,h,xp=xp,max_degree=8),
                        xp=xp,scales=background_gpu*.02,tolerance=1e-12)
    for actual,expected in zip((result.nodes,result.stages,result.rates*h),(cpu[0],cpu[1],cpu[2]*h)):
        assert np.max(np.abs(xp.asnumpy(actual)-expected)/(background*.02))<1e-8
    for actual,key in ((result.stage_electric,'electric'),(result.stage_current,'current'),(result.stage_velocity,'velocity')):
        assert np.max(np.abs(xp.asnumpy(actual)-fields[key]))/max(np.max(np.abs(fields[key])),1e-300)<1e-8
    np.testing.assert_array_equal(xp.asnumpy(p_gpu),path)
    np.testing.assert_array_equal(xp.asnumpy(z_gpu),z)
