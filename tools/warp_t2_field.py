#!/usr/bin/env python3
"""Bounded fixed-ion T2-E CPU/CUDA windows, independent oracles and timings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT,ROOT/"src",ROOT/"tests/warp"):
    if str(location) not in sys.path:
        sys.path.insert(0,str(location))

from fomhd.backend import Backend
from fomhd.core import allocate_state, forward
from fomhd.field_batch import FieldBatch
from fomhd.time_affine import constant_prefix, rk4_affine, rk4_stages
from fomhd.time_field import FrozenJVP, PeriodicHallReference, PolynomialReference, rk4_batch, solve_window
from fomhd.types import Failure
from massless_verification_fixture import HallCase, build_periodic_context, uniform_state


def linear_inputs(points, count, *, manufactured=False, h_factor=1.):
    """Independent normalized Hall/CGL linear block, not nonlinear evolution."""
    case = HallCase(points=points)
    modes = np.fft.fftfreq(points)*points
    beta = case.hall_rate*modes**2
    h = .05/max(beta)*h_factor
    l = np.zeros((points,4,4),dtype=np.complex128)
    l[:,2,3], l[:,3,2] = -beta, beta
    start = np.zeros((points,4),dtype=np.complex128)
    start[1,2:], start[-1,2:] = [2e-6,-2e-6j],[2e-6,2e-6j]
    high = points//2
    start[high,2:],start[-high,2:] = [2e-8,-2e-8j],[2e-8,2e-8j]
    forcing = np.zeros((count,4,points,4),dtype=np.complex128)
    if manufactured:
        t = np.arange(count)[:,None,None,None]
        stage = np.array([1.,2.,3.,4.])[None,:,None,None]
        pattern = np.cos(np.arange(points))[None,None,:,None]
        forcing[:] = case.hall_rate*1e-8*(1+.03*t+.2*stage)*pattern
    return dict(L=l,h=h,start=start,forcing=forcing,beta=beta,
                normalization="electron pressures/p0 and transverse B/B0")


def linear_solve(case,method,xp):
    l,forcing,h,start = (case[k] for k in ("L","forcing","h","start"))
    if method == "prefix":
        a,b = rk4_affine(l,forcing,h,xp=xp)
        nodes = constant_prefix(a,b,start,xp=xp)
        left = xp.concatenate((start[None],nodes[:-1]),axis=0)
        stages,rates,_ = rk4_stages(l,forcing,h,left,xp=xp)
        return nodes,stages,rates
    nodes,stages,rates = [],[],[]
    z = start.copy()
    for f in forcing:
        k1 = xp.matmul(l,z[...,None])[...,0]+f[0]
        y2 = z+h*k1/2
        k2 = xp.matmul(l,y2[...,None])[...,0]+f[1]
        y3 = z+h*k2/2
        k3 = xp.matmul(l,y3[...,None])[...,0]+f[2]
        y4 = z+h*k3
        k4 = xp.matmul(l,y4[...,None])[...,0]+f[3]
        stages.append(xp.stack((z,y2,y3,y4)))
        rates.append(xp.stack((k1,k2,k3,k4)))
        z = z+h*(k1+2*k2+2*k3+k4)/6
        nodes.append(z)
    return xp.stack(nodes),xp.stack(stages),xp.stack(rates)


def pack_state(context,state):
    xp,nc,q = context.compute.xp,context.op.Nc,2*context.p.K+1
    p = xp.concatenate((state.fluid[0].n.reshape(q,nc),state.fluid[0].mom.reshape(q,3*nc),state.fluid[0].perp.reshape(q,nc)),axis=1)
    z = xp.concatenate((state.fluid[1].perp.reshape(q,nc),state.fluid[1].parallel.reshape(q,nc),state.B.reshape(q,context.op.Nb)),axis=1)
    return p,z


def unpack_state(context,p,z):
    state = allocate_state(context)
    xp,nc = context.compute.xp,context.op.Nc
    state.fluid[0].n = p[:,:nc].copy().reshape(-1)
    state.fluid[0].mom = p[:,nc:4*nc].copy().reshape(-1)
    state.fluid[0].perp = p[:,4*nc:].copy().reshape(-1)
    state.fluid[1].n = state.fluid[0].n.copy()
    state.fluid[1].perp = z[:,:nc].copy().reshape(-1)
    state.fluid[1].parallel = z[:,nc:2*nc].copy().reshape(-1)
    state.B = xp.ascontiguousarray(z[:,2*nc:]).reshape(-1)
    state.step,state.generation,state.time = 0,1,context.p.start_time
    return state


def periodic_inputs(points,amplitude,nonuniform,device,*,spatial_mode=1):
    case = HallCase(points=points)
    c = build_periodic_context(case,device=device)
    xp = c.compute.xp
    state = uniform_state(c,case)
    phase = xp.arange(points,dtype=xp.float64)*2*math.pi/points
    density = case.n0*(1+nonuniform*xp.cos(phase))
    state.fluid[0].n[:] = density
    state.fluid[1].n[:] = density
    carrier=spatial_mode*phase
    velocity = xp.stack((amplitude*case.v_alfven*xp.cos(carrier),
                         .7*amplitude*case.v_alfven*xp.sin(carrier),
                         .03*nonuniform*case.v_alfven*xp.sin(phase)))
    state.fluid[0].mom[:] = (velocity*density*float(c.p.species[0].mass)).reshape(-1)
    state.fluid[1].perp[:] = case.pressure*(1+.2*nonuniform*xp.cos(phase))
    state.fluid[1].parallel[:] = case.pressure*(1-.15*nonuniform*xp.cos(phase))
    state.B[:points] = amplitude*case.B0*xp.cos(carrier)
    state.B[points:2*points] = -amplitude*case.B0*xp.sin(carrier)
    p,z = pack_state(c,state)
    scales = xp.concatenate((xp.full(2*points,case.pressure),xp.full(3*points,case.B0)))[None]
    return c,p,z,scales


def smooth_closed_seed(c,amplitude,*,spatial_mode=1):
    """Smooth toroidal background plus compatible n_phi=1 curl perturbation.

    The fixed wall principal solve corrects current-normal trace without a
    volume null-space SVD. This is bounded fixture initialization, not a new
    production equilibrium or a spatial convergence certificate.
    """
    from fomhd.wall_cache import get_wall_cache
    cache=get_wall_cache(c)
    nr=len(c.geometry.r_faces.keys)
    nz=len(c.geometry.z_faces.keys)
    nv=len(c.geometry.vertices.keys)
    b=np.zeros((3,c.op.Nb),complex)
    vertex_r=np.asarray(c.geometry.vertices.R)
    b[1,nz:nz+nv]=c.p.ref.magnetic*1.3/vertex_r
    cell_r=np.asarray(c.geometry.cells.R)
    cell_z=np.asarray(c.geometry.cells.Z)
    shape=np.sin(math.pi*(cell_r-1.)/.6)**4*np.sin(math.pi*cell_z/.6)**4
    shape*=np.cos((spatial_mode-1)*math.pi*cell_z/.6)
    electric=np.zeros(c.op.Ne,complex)
    electric[nr:nr+c.op.Nc]=amplitude*c.p.ref.magnetic*.6*shape
    b[2]=c.op.C[2]@electric
    b[0]=b[2].conj()
    for stored,mode in enumerate(cache.modes):
        rhs=-mode.extraction@(mode.Cdag@b[stored])
        multiplier=np.zeros(len(rhs),complex)
        selected=list(mode.selected_rows)
        if selected:
            solution=c.compute.linear_prepared(mode.prepared,c.compute.array(rhs[selected]))
            multiplier[selected]=solution.x
        b[stored]+=mode.C@(mode.weighted_adjoint@multiplier)
        scale=max(np.max(np.abs(rhs)),1e-300)
        residual=mode.extraction@(mode.Cdag@b[stored])
        if np.max(np.abs(residual))>1e-8*scale+1e-18:
            raise Failure("InitialCompatibility","smooth fixture wall projection failed")
    return b.reshape(-1)


def closed_inputs(points,amplitude,nonuniform,device,*,spatial_mode=1):
    """Actual RZ wall operators and nonuniform compatible magnetic field."""
    from test_massless_support import context, uniform_state as closed_state
    from fomhd.operators import build_operators
    c = context(K=1,hard_diagnostics=False,nodes_r=points,nodes_z=points)
    c.p.memory_limit = c.p.device_memory_limit = 8 << 30
    state = closed_state(c,compatible=False)
    state.B = c.compute.array(smooth_closed_seed(c,amplitude,spatial_mode=spatial_mode))
    nc = c.op.Nc
    r = np.asarray(c.geometry.cells.R)
    z = np.asarray(c.geometry.cells.Z)
    phi = np.arange(c.N)*2*math.pi/c.N
    profile = np.cos(2*math.pi*(r-1.)/.6)*np.cos(2*math.pi*z/.6)
    density = c.p.ref.ni*(1+nonuniform*profile[None,:]*(1+.2*np.cos(phi[:,None])))
    state.fluid[0].n = forward(density.reshape(-1),nc,c)
    state.fluid[1].n = state.fluid[0].n.copy()
    for name,coefficient in (("perp",.3),("parallel",-.2)):
        pressure = c.p.ref.pressure*(1+coefficient*nonuniform*profile[None,:]*(1+.3*np.cos(phi[:,None])))
        setattr(state.fluid[1],name,forward(pressure.reshape(-1),nc,c))
    momentum = np.zeros((c.N,3,nc),complex)
    # Pure toroidal ion flow has no wall-normal R/Z component.
    momentum[:,1] = amplitude*c.p.ref.ni*c.p.species[0].mass*c.p.ref.velocity*(1+.2*np.cos(phi[:,None]))
    state.fluid[0].mom = forward(momentum.reshape(-1),3*nc,c)
    p0,z0 = pack_state(c,state)
    physical_scales = np.tile(np.r_[np.full(2*nc,c.p.ref.pressure),np.full(c.op.Nb,c.p.ref.magnetic)],(3,1))
    if device != "cpu":
        backend = Backend(device)
        c.compute = backend
        c.op = build_operators(c.geometry,c.p.K,backend=backend)
        p0,z0 = backend.array(p0),backend.array(z0)
        physical_scales = backend.real_array(physical_scales)
    return c,p0,z0,physical_scales


class CountedRHS:
    def __init__(self,producer):
        self.producer = producer
        self.calls = self.vectors = 0
        self.last = None
        self.check_extrema = {}

    def reset(self):
        self.calls = self.vectors = 0
        self.check_extrema.clear()
        for key in self.producer.work:
            self.producer.work[key] = 0

    def __call__(self,p,z):
        self.calls += 1
        self.vectors += len(z)
        self.last = self.producer.rhs(p,z)
        for key,value in vars(self.last.checks).items():
            bounds=self.check_extrema.setdefault(key,[value,value])
            bounds[0],bounds[1]=min(bounds[0],value),max(bounds[1],value)
        return self.last.rhs


def serial_field(rhs,path,start,h,xp,*,field_history=None):
    """Independent sequential original RK4 time oracle, validation only."""
    nodes,stages,rates = [],[],[]
    current = start.copy()
    fields=[]
    def evaluate(p,z):
        value=rhs(p,z)
        if field_history is not None:
            fields.append((rhs.last.electric,rhs.last.current,rhs.last.velocity))
        return value
    for p in path:
        k1 = evaluate(p[0:1],current[None])[0]
        y2 = current+h*k1/2
        k2 = evaluate(p[1:2],y2[None])[0]
        y3 = current+h*k2/2
        k3 = evaluate(p[2:3],y3[None])[0]
        y4 = current+h*k3
        k4 = evaluate(p[3:4],y4[None])[0]
        stages.append(xp.stack((current,y2,y3,y4)))
        rates.append(xp.stack((k1,k2,k3,k4)))
        current = current+h*(k1+2*k2+2*k3+k4)/6
        nodes.append(current)
    rhs(path[-1:,3],current[None])
    if field_history is not None:
        for index,key in enumerate(("electric","current","velocity")):
            data=xp.concatenate([v[index] for v in fields],axis=0)
            field_history[key]=data.reshape((len(path),4)+data.shape[1:])
    return xp.stack(nodes),xp.stack(stages),xp.stack(rates)


def synchronize(xp):
    if xp is not np:
        xp.cuda.get_current_stream().synchronize()


def timed(function,xp,warmups=2,repetitions=5):
    """Cold and warmed synchronized end-to-end windows, without output I/O."""
    synchronize(xp)
    begin = time.perf_counter()
    value = function()
    synchronize(xp)
    cold = time.perf_counter()-begin
    for _ in range(warmups):
        value = function()
    synchronize(xp)
    samples = []
    for _ in range(repetitions):
        begin = time.perf_counter()
        value = function()
        synchronize(xp)
        samples.append(time.perf_counter()-begin)
    return value,dict(cold_s=cold,samples_s=samples,median_s=float(np.median(samples)),
                      minimum_s=min(samples),maximum_s=max(samples),warmups=warmups,repetitions=repetitions,
                      scope="construction/guess/residual/correction/stage recovery/checks/sync; input fixture and output I/O separate")


def _host(value,xp):
    return np.asarray(value) if xp is np else xp.asnumpy(value)


def _save_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def run(config,output):
    if output.exists():
        raise ValueError("output must be a new directory")
    if config["device"] != "cpu" and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("CUDA runs require an owned Slurm allocation")
    output.mkdir(parents=True)
    _save_json(output/"input.json",config)
    metadata = dict(config=config,source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in list((ROOT/"src/fomhd").glob("*.py"))+[Path(__file__)]},
                    host=platform.node(),slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                    field_subproblem_only=True,full_coupled_acceptance="not_applicable",
                    outer_S="fixed supplied path",particle_rp_q="not_applicable")
    started = time.perf_counter()
    attempted_work_scope = None
    try:
        backend = Backend(config["device"])
        xp = backend.xp
        if xp is not np:
            xp.get_default_memory_pool().set_limit(size=8<<30)
            metadata["gpu"] = {str(k): (v.decode() if isinstance(v,bytes) else v)
                               for k,v in xp.cuda.runtime.getDeviceProperties(0).items()
                               if k in ("name","major","minor","totalGlobalMem","multiProcessorCount",
                                        "pciBusID","pciDeviceID","pciDomainID")}
        setup = time.perf_counter()
        kind,points,count = config["unit"],config["points"],config["M"]
        if kind == "E1":
            problem = linear_inputs(points,count,manufactured=config["manufactured"],h_factor=config["h_factor"])
            h = problem["h"]
            host_problem = problem.copy()
            for key in ("L","forcing","start"):
                problem[key] = backend.array(problem[key])
            synchronize(xp)
            metadata["input_setup_s"] = time.perf_counter()-setup
            oracle = linear_solve(host_problem,"serial",np)
            actual,timing = timed(lambda:linear_solve(problem,config["method"],xp),xp,
                                  config["warmups"],config["repetitions"])
            scale = 2e-6
            errors = [float(np.max(np.abs(_host(a,xp)-b)))/scale for a,b in zip(actual,oracle)]
            # Exact discrete continuation to a declared 2T linear observation.
            total = math.ceil(4*math.pi/(problem["beta"][1]*h))
            x = 1j*host_problem["beta"]*h
            factor = 1+x+x*x/2+x**3/6+x**4/24
            exact = np.exp(x*total)
            metadata.update(linear_2T_discrete_gain=float(abs(factor[-1]**total)),
                            linear_2T_phase_error=float(np.angle(factor[1]**total/exact[1])),
                            linear_2T_note="independent polynomial continuation, not a measured repeated nonlinear run",
                            errors=dict(nodes=errors[0],stages=errors[1],rates_times_h=errors[2]*h),
                            h=h,H=count*h,r_Z=None,r_space=None,
                            normalized_units=host_problem["normalization"],
                            correct=bool(max(errors[:2]) <= 1e-10 and errors[2]*h <= 1e-10))
            arrays = {"nodes":_host(actual[0],xp),"stages":_host(actual[1],xp),
                      "rates":_host(actual[2],xp),"oracle_nodes":oracle[0],"forcing":host_problem["forcing"]}
        else:
            build = periodic_inputs if kind == "E2" else closed_inputs
            c,p0,z0,background = build(points,config["amplitude"],config["nonuniform"],config["device"],spatial_mode=config.get("spatial_mode",1))
            producer = FieldBatch(c,structured_normal=not config.get("dense_normal",False))
            rhs = CountedRHS(producer)
            if kind == "E2":
                case = HallCase(points=points)
                omega = case.hall_rate*(points//2)**2/(1-config["nonuniform"])
            else:
                dr = .6/(points-1)
                # Conservative Hall scale; actual stability/refinement remains measured.
                omega = 8/dr**2*c.p.ref.magnetic/(c.p.mu0*(-c.p.species[1].charge)*c.p.ref.ni*(1-config["nonuniform"]))
            h = .05/omega*config["h_factor"]
            path = xp.broadcast_to(p0,(count,4)+p0.shape).copy()
            scales = background*max(config["amplitude"],.02 if kind=="E3" else 2e-6)
            synchronize(xp)
            metadata.update(input_setup_s=time.perf_counter()-setup,h=h,H=count*h,
                            omega_estimate=omega,omega_h=omega*h,active_cells=c.op.Nc,
                            magnetic_dofs=c.op.Nb,K=c.p.K,Nphi=c.N,
                            background_is_equilibrium=False if kind=="E3" or config["nonuniform"] else "uniform reference with perturbation",
                            fixed_path="constant supplied ion stage values; not a complete accepted outer trajectory")
            metadata["normal_representation"]="exact diagonal" if producer.diagonal_normal else "dense incidence Gram"

            def execute():
                if config["method"] == "serial":
                    fields={}
                    numerical=serial_field(rhs,path,z0,h,xp,field_history=fields)
                    return SimpleNamespace(nodes=numerical[0],stages=numerical[1],rates=numerical[2],
                                           stage_electric=fields["electric"],stage_current=fields["current"],stage_velocity=fields["velocity"])
                if kind == "E2":
                    reference = PeriodicHallReference(points,case.k,case.B0,case.n0,c.p.mu0,-c.p.species[1].charge,h,xp=xp)
                else:
                    derivative = FrozenJVP(rhs,p0,z0,background,xp=xp,eta=config["eta"])
                    reference = PolynomialReference(derivative,h,xp=xp,max_degree=config["degree"])
                result = solve_window(rhs,path,z0,h,reference,xp=xp,scales=scales,
                                      tolerance=config["tolerance"],max_iterations=8)
                result.reference_action_calls = reference.action_calls
                result.reference_action_vectors = reference.action_vectors
                result.jvp_calls = reference.action.calls if kind=="E3" else 0
                return result

            # An independent sequential time recurrence on this backend is outside solve/timing.
            attempted_work_scope = "oracle"
            oracle_begin = time.perf_counter()
            oracle_fields={}
            oracle = serial_field(rhs,path,z0,h,xp,field_history=oracle_fields)
            synchronize(xp)
            metadata["oracle_s"] = time.perf_counter()-oracle_begin
            oracle=tuple(_host(v,xp) for v in oracle)
            oracle_fields={k:_host(v,xp) for k,v in oracle_fields.items()}
            metadata["oracle_work"]=dict(rhs_batches=rhs.calls,rhs_vectors=rhs.vectors,
                                         producer=producer.work.copy())
            rhs.reset()
            attempted_work_scope = "timed"
            actual,timing = timed(execute,xp,config["warmups"],config["repetitions"])
            numerical = (actual.nodes,actual.stages,actual.rates)
            host_scales=_host(scales,xp)
            host_numerical=tuple(_host(v,xp) for v in numerical)
            errors = [float(np.max(np.abs(a-b)/(host_scales*(1/h if i==2 else 1)))) for i,(a,b) in enumerate(zip(host_numerical,oracle))]
            electric=_host(actual.stage_electric,xp)
            electric_scale=max(float(np.max(np.abs(oracle_fields["electric"]))),1e-300)
            electric_error=float(np.max(np.abs(electric-oracle_fields["electric"]))/electric_scale)
            metadata.update(errors=dict(nodes=errors[0],stages=errors[1],rates_times_h=errors[2]),
                            correct=max(errors) <= 1e-8 and electric_error<=1e-8,
                            qualification={"same_discrete_waveform_pass":max(errors)<=1e-8,"stage_electric_pass":electric_error<=1e-8,"temporal_refinement":"paired analysis required","full_coupled_acceptance":"not_applicable"},
                            stage_electric_relative_error=electric_error,checks=vars(rhs.last.checks),
                            checks_scope="last timed execution endpoint; every RHS independently enforces its hard checks",
                            check_extrema_all_timed_rhs=rhs.check_extrema,
                            measured_all_repetitions_rhs_batches=rhs.calls,measured_all_repetitions_rhs_vectors=rhs.vectors,
                            r_Z=None if config["method"]=="serial" else actual.iterations,
                            r_space="direct batched wall solves; polynomial derivative depth counted separately",
                            final_ledger_commit=False,history=[] if config["method"]=="serial" else actual.history)
            if config["method"]!="serial":
                metadata.update(residual_evaluations=actual.residual_evaluations,
                                window_rhs_vectors=actual.rhs_vectors,
                                reference_action_calls=actual.reference_action_calls,
                                reference_action_vectors=actual.reference_action_vectors,jvp_calls=actual.jvp_calls)
            weights=np.array([1.,2.,2.,1.])/6
            electric_integral=h*np.sum(electric*weights.reshape(1,4,1,1),axis=(0,1))
            arrays = dict(nodes=host_numerical[0],stages=host_numerical[1],rates=host_numerical[2],
                          electric=electric,current=_host(actual.stage_current,xp),velocity=_host(actual.stage_velocity,xp),
                          electric_time_integral=electric_integral,
                          oracle_nodes=oracle[0],oracle_electric=oracle_fields["electric"],
                          initial_Z=_host(z0,xp),fixed_P=_host(p0,xp),scales=host_scales)
            metadata["producer_work_all_repetitions"]=producer.work.copy()
            metadata["work_scope"]="cold + warmups + timed repetitions; excludes independent serial oracle and input setup"
            metadata["batch_workspace_plan"]=producer.workspace_estimate(count)
            if kind=="E2":
                carrier=config.get("spatial_mode",1)
                def coefficient(nodes):
                    transverse=nodes[:,0,2*points:3*points]-1j*nodes[:,0,3*points:4*points]
                    return np.fft.fft(transverse,axis=-1)[:,carrier]/points
                phase_actual,phase_ref=coefficient(host_numerical[0]),coefficient(oracle[0])
                metadata["target_mode_phase_error_rad"]=float(np.max(np.abs(np.angle(phase_actual/phase_ref))))
                arrays["target_mode"]=phase_actual
                arrays["oracle_target_mode"]=phase_ref
        metadata.update(status="completed",timing=timing)
        if xp is not np:
            pool = xp.get_default_memory_pool()
            metadata["device_memory"] = dict(live_pool_bytes=pool.used_bytes(),reserved_pool_bytes=pool.total_bytes(),
                                              note="end-of-case pool observation; not yet a total device peak")
        metadata["output_array_bytes"] = sum(a.nbytes for a in arrays.values())
        np.savez_compressed(output/"arrays.npz",**arrays)
    except Exception as error:
        metadata.update(status="failed",correct=False,failure_code=getattr(error,"code",type(error).__name__),
                        failure_message=str(error),history=getattr(error,"history",[]))
        (output/"traceback.txt").write_text(traceback.format_exc())
        if "rhs" in locals():
            metadata["attempted_rhs_batches"]=rhs.calls
            metadata["attempted_rhs_vectors"]=rhs.vectors
        if "producer" in locals():
            metadata["attempted_producer_work"]=producer.work.copy()
            metadata["attempted_work_scope"]=attempted_work_scope
        if hasattr(error,"nodes"):
            np.savez_compressed(output/"failure-window.npz",nodes=_host(error.nodes,xp))
    metadata["host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    metadata["total_elapsed_s"] = time.perf_counter()-started
    _save_json(output/"report.json",metadata)
    print(json.dumps({k:metadata.get(k) for k in ("status","correct","errors","r_Z","failure_code","total_elapsed_s")}),flush=True)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit",choices=("E1","E2","E3"),required=True)
    parser.add_argument("--device",choices=("cpu","cuda:0"),default="cpu")
    parser.add_argument("--method",choices=("prefix","serial"),default="prefix")
    parser.add_argument("--points",type=int,default=33)
    parser.add_argument("--M",type=int,default=32)
    parser.add_argument("--amplitude",type=float,default=.02)
    parser.add_argument("--nonuniform",type=float,default=.1)
    parser.add_argument("--spatial-mode",type=int,default=1)
    parser.add_argument("--h-factor",type=float,default=1.)
    parser.add_argument("--eta",type=float,default=1e-5)
    parser.add_argument("--degree",type=int,default=8)
    parser.add_argument("--tolerance",type=float,default=1e-10)
    parser.add_argument("--manufactured",action="store_true")
    parser.add_argument("--dense-normal",action="store_true",help="retain the exact dense normal control")
    parser.add_argument("--warmups",type=int,default=2)
    parser.add_argument("--repetitions",type=int,default=5)
    parser.add_argument("--output",type=Path,required=True)
    options=vars(parser.parse_args(argv))
    output = options.pop("output")
    if (options["M"]<1 or options["M"]>512 or options["points"]<4
            or options["nonuniform"]<0 or options["nonuniform"]>=.5
            or not 0<options["amplitude"]<=.1 or not 0<options["h_factor"]<=1
            or not 1<=options["degree"]<=30 or not 0<options["tolerance"]<=1e-10
            or not 1<=options["spatial_mode"]<=options["points"]//2
            or not 0<=options["warmups"]<=2 or not 1<=options["repetitions"]<=5):
        parser.error("controls exceed the selected bounded experiment")
    result = run(options,output)
    return 0 if result["status"]=="completed" and result["correct"] else 1


if __name__=="__main__":
    raise SystemExit(main())
