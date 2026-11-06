#!/usr/bin/env python3
"""Budgeted massless-electron global compressional-response experiments (SI)."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.linalg import lstsq

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from fomhd import core, io, massless, stage, stability
from fomhd.backend import Backend
from fomhd.geometry import Point2, build_geometry
from fomhd.operators import build_operators
from fomhd.time import advance
from fomhd.types import Budget, Context, Norm, Parameters, References, Species, StageTag, cumulative_record
from tools.cold_hall_reference import E, MP, MU0, ReferenceCase, disk_quadrature, evaluate_mode, solve_reference
from tools.global_mode_analysis import fit_real_mode, physical_mode_coefficients, fluctuation_energy

EAST_SHA='df8f073bbad202d799b70d0bc2479c9cbcc2d64b729c1019f2575b14de961842'
EAST_URL='https://raw.githubusercontent.com/Youjunhu/GTAW/7ae4929f2b4e56637f0462a9a186d92edb17f316/gfile/g048059.03650'


def circular_polygon():
    a=np.arange(128)*2*np.pi/128
    return np.column_stack((3+np.cos(a),np.sin(a)))


def physical_parameters(K,density,B,length,pressure):
    """SI reference scales; only prototype power/energy/momentum checks observe."""
    va=B/math.sqrt(MU0*2*MP*density);t=length/va
    ref=References(length=length,time=t,ni=density,ne=density,velocity=va,
                   electric=B*va,magnetic=B,current=B/(MU0*length),
                   charge=E*density,pressure=pressure)
    p=Parameters(species=(Species('ion',2*MP,E,5/3),Species('electron',9.1093837139e-31,-E,0)),
                 epsilon0=8.8541878188e-12,mu0=MU0,B_min=1e-10,
                 dt=1e-12,start_time=0,rank_rtol=1e-12,nsteps=1,K=K,Nphi=5,
                 memory_limit=24<<30,device_memory_limit=24<<30,inner_distance=length*.15,
                 ref=ref,electron_inertia=0,integrator='rk4',dt_guard='error')
    scales={'C/m^3':ref.charge,'1/(m^3*s)':density/t,'kg/(m^2*s^2)':2*MP*density*va/t,
            'A/m^2':ref.current,'1/(m^2*s)':density*va,'A/m^3':ref.current/length,
            'T/(m*s)':B/(length*t),'T/s':B/t,'T/m':B/length,
            'V/m^3':ref.electric/length**2,'V/m':ref.electric,'Pa/s':pressure/t,
            '1/s':1/t,'W':pressure*length**3/t,'J':(pressure+B*B/MU0)*length**3,
            'C/(m^3*s)':ref.charge/t,'C':ref.charge*length**3,'1':density*length**3,
            'kg*m/s':2*MP*density*va*length**3,'kg*m^2/s':2*MP*density*va*length**4}
    for key,(unit,norm,hard) in io._massless_budget_registry(p).items():
        observe=unit in ('W','J','kg*m/s','kg*m^2/s')
        p.budgets[key]=Budget(0,1e-9,scales[unit],unit,Norm.weighted_l2 if norm=="weighted_L2" else Norm(norm),hard and not observe)
    # Compatibility momentum uses physical electron mass, not ion inertia.
    for suffix in ('linf','weighted_L2'):
        p.budgets['electron_momentum_compatibility.'+suffix].reference*=p.species[1].mass/p.species[0].mass
    return p


def encode_mode(a,mode,K):
    if mode>K:raise ValueError('mode exceeds retained K')
    coefficients=physical_mode_coefficients(a,mode)
    out=np.zeros((2*K+1,np.asarray(a).size),complex)
    out[K-mode:K+mode+1]=coefficients
    return out.reshape(-1)


def host(value,context):
    return np.asarray(context.compute.to_host(value))


def project_B(value,context,stored,flux=None):
    """Explicit input-only minimum-MB-norm compatibility correction on CPU."""
    op=context.op;g=context.geometry
    nr=len(g.r_faces.R);nc=op.Nc
    edges=[w.face if w.component=='r_face' else nr+nc+w.face for w in g.walls]
    rows=[host(op.DB[stored],context),host(op.Cdag[stored],context)[edges]]
    rhs=[np.zeros(rows[0].shape[0],complex),np.zeros(len(edges),complex)]
    if flux is not None:
        rows.append(host(op.topology_flux_row,context).reshape(1,-1));rhs.append(np.array([flux]))
    A=np.vstack(rows);target=np.concatenate(rhs)
    w=np.sqrt(host(op.MB,context).real)
    scaled=A/w[None,:]
    norm=np.linalg.norm(scaled,axis=1);keep=norm>0
    scaled=scaled[keep]/norm[keep,None]
    residual=(target-A@value)[keep]/norm[keep]
    correction=lstsq(scaled,residual,cond=context.p.rank_rtol,lapack_driver='gelsd')[0]/w
    corrected=value+correction
    record=dict(relative_MB_correction=float(np.linalg.norm(correction*w)/max(np.linalg.norm(value*w),1e-300)),
                max_DB=float(np.max(np.abs(host(op.DB[stored],context)@corrected),initial=0)),
                max_wall_curl=float(np.max(np.abs(host(op.Cdag[stored],context)[edges]@corrected),initial=0)))
    return corrected,record


def make_context(kind,nodes,K,device,eq_path=None):
    if not 4<=nodes<=24 or K not in (0,1):raise ValueError('nodes in [4,24] and K=0/1 required')
    meta={'kind':kind,'nodes':nodes,'K':K,'Nphi':5,'electron_inertia':0,
          'wall':'fixed closed conductor with constrained weak-wall reactions',
          'diagnostic_policy':'prototype_observe_power_energy_momentum'}
    if kind=='circle':
        polygon=circular_polygon();n=3.423e18;B=3.4;L=1.;pressure=1e-4*B**2/(2*MU0)
        eq=None
        meta.update(R0_m=3.,a_m=1.,B0_T=B,density_m3=n,total_axis_beta=1e-4,
                    pressure_total_Pa=pressure,profile_assumption='uniform deuterium; pi=pe=ptotal/2; electron isotropic initially')
    else:
        from tools.geqdsk import read_geqdsk
        from tools.tokamak_equilibrium import closed_flux_contour
        if eq_path is None:raise ValueError('EAST requires the pinned G-EQDSK input')
        if hashlib.sha256(Path(eq_path).read_bytes()).hexdigest()!=EAST_SHA:
            raise ValueError('EAST input SHA256 mismatch')
        eq=read_geqdsk(eq_path);polygon=np.asarray(closed_flux_contour(eq,.8),dtype=float)
        n=2e19;B=abs(eq['central_R_B'][1]);L=(np.ptp(polygon[:,0])+np.ptp(polygon[:,1]))/4
        pressure=float(np.max(eq['pressure']))
        meta.update(shot=48059,time_s=3.65,source_url=EAST_URL,sha256=EAST_SHA,
                    psiN_wall=.8,psi_factor=-1.,density_m3=n,
                    profile_assumption='uniform deuterium density; EFIT p_total split equally; no imposed ion flow',
                    convention='B_R=-psi_Z/R, B_Z=psi_R/R with psi=-psi_EFIT; F/R unchanged; sign matches integrated header plasma current')
    lo=polygon.min(axis=0);hi=polygon.max(axis=0);h=(hi-lo)/nodes
    geometry=build_geometry(lo[0]+(np.arange(nodes)+.5)*h[0],lo[1]+(np.arange(nodes)+.5)*h[1],
                            [Point2(*a) for a in polygon])
    p=physical_parameters(K,n,B,L,pressure)
    backend=Backend("cuda:0" if device=="cuda" else device)
    op=build_operators(geometry,K,{'memory_cap_bytes':24<<30,'rank_rtol':p.rank_rtol},backend)
    context=Context(p,geometry,op,backend,5)
    meta.update(active_cells=op.Nc,h_R_m=float(h[0]),h_Z_m=float(h[1]),
                area_h_m2=float(np.sum(geometry.M0/geometry.cells.R)),
                volume_h_m3=float(2*np.pi*np.sum(geometry.M0)))
    return context,meta,eq


def background(context,meta,eq):
    p=context.p;g=context.geometry;op=context.op;nc=op.Nc
    state=stage.closed_reference(context,p.ref.ni,p.ref.pressure/2,p.ref.pressure/2,p.ref.magnetic)
    mapping={}
    if eq is None:
        R,Z,w=disk_quadrature(ReferenceCase(order=5))
        flux=float(np.sum(w*3.4*3/R**2))
        raw=host(state.B,context)[p.K*op.Nb:(p.K+1)*op.Nb]
        raw*=flux/float((host(op.topology_flux_row,context).reshape(-1)@raw).real)
        value=raw;mapping['flux_target_Wb']=flux
    else:
        from tools.tokamak_equilibrium import sample_equilibrium
        values=[]
        for table,component in ((g.z_faces,'B_R'),(g.vertices,'B_phi'),(g.r_faces,'B_Z')):
            values.extend(np.asarray(sample_equilibrium(eq,table.R,table.Z,psi_factor=meta['psi_factor'])[component]))
        raw=np.asarray(values,complex)
        flux=float((host(op.topology_flux_row,context).reshape(-1)@raw).real)
        value,correction=project_B(raw,context,p.K,flux)
        mapping.update(correction)
        profiles=sample_equilibrium(eq,g.cells.R,g.cells.Z,psi_factor=meta['psi_factor'])
        pressure=np.asarray(profiles['pressure_Pa'])/2
        for field in (state.fluid[0].perp,state.fluid[1].perp,state.fluid[1].parallel):
            field[p.K*nc:(p.K+1)*nc]=context.compute.asarray(pressure)
    state.B[p.K*op.Nb:(p.K+1)*op.Nb]=context.compute.asarray(value)
    return state,{'background_raw_B':raw,'background_mapped_B':value},mapping


def perturb(state,context,meta,mode,amplitude,reference):
    op=context.op;g=context.geometry;p=context.p;K=p.K;nc=op.Nc
    R,Z=np.asarray(g.cells.R),np.asarray(g.cells.Z)
    if reference is not None:
        fields=evaluate_mode(reference,R,Z)
        # A common phase makes the dominant compression real at t=0 for n=0.
        bphi=fields['b'][:,1];phase=np.exp(-1j*np.angle(bphi[np.argmax(np.abs(bphi))]))
        norm=max(np.max(np.abs(fields['b']))/p.ref.magnetic,np.max(np.abs(fields['velocity']))/p.ref.velocity,
                 np.max(np.abs(fields['div_velocity']))/reference['omega_rad_s'])
        scale=phase/norm
        v=fields['velocity']*scale
        lam=1j*reference['omega_rad_s']
        dn=-p.ref.ni*fields['div_velocity']*scale/lam
        dpi=-(5/3)*(p.ref.pressure/2)*fields['div_velocity']*scale/lam
        theta=fields['div_velocity']*scale
        ve=(fields['velocity']-fields['current']/(E*p.ref.ne))*scale
        strain=(ve[:,0]+1j*mode*ve[:,1])/R
        dpperp=(p.ref.pressure/2)*(strain-2*theta)/lam
        dppar=-(p.ref.pressure/2)*(theta+2*strain)/lam
        # Sample the gauge-invariant analytic B, not a sampled potential whose
        # continuous gauge cancellation need not survive a different discrete C.
        b=[]
        for table,component in ((g.z_faces,0),(g.vertices,1),(g.r_faces,2)):
            b.extend(evaluate_mode(reference,table.R,table.Z)['b'][:,component]*scale)
        raw_b=np.asarray(b)
        b,mapping=project_B(raw_b,context,K+mode,0. if mode==0 else None)
        context.seed_arrays={'raw_B':raw_b,'mapped_B':b.copy(),'raw_velocity':v.copy()}
        meta['seed']='independent cold Hall constrained-wall compressional candidate'
    else:
        x=(R-np.mean(R))/p.ref.length;z=(Z-np.mean(Z))/p.ref.length
        envelope=np.exp(-3*(x*x+z*z))
        v=np.column_stack((x*envelope,np.zeros(nc),z*envelope))*p.ref.velocity
        b=np.zeros(op.Nb,complex);dn=np.zeros(nc,complex)
        dpi=dpperp=dppar=np.zeros(nc,complex);mapping={}
        raw_b=b.copy();context.seed_arrays={'raw_B':raw_b,'mapped_B':b.copy(),'raw_velocity':v.copy()}
        meta['seed']='smooth poloidal compressional velocity impulse; not an exact eigenmode'
    # The n0 initial condition is a real standing phase, not two copies of A.
    if mode==0:
        v,b,dn,dpi,dpperp,dppar=[np.real(a) for a in (v,b,dn,dpi,dpperp,dppar)]
    # Input-only normal velocity correction. Constant reference density is exact
    # for both declared initial states; dynamic wall response is production code.
    T=host(op.T,context);m=np.tile(host(op.M0,context).real,3)
    nodal=v.T.reshape(-1)
    correction=lstsq(T/np.sqrt(m)[None,:],T@nodal,cond=p.rank_rtol)[0]/np.sqrt(m)
    vflat=nodal-correction
    context.seed_arrays['mapped_velocity']=vflat.reshape(3,nc).T
    mapping['relative_velocity_wall_correction']=float(np.linalg.norm(correction*np.sqrt(m))/max(np.linalg.norm(nodal*np.sqrt(m)),1e-300))
    base_n=host(state.fluid[0].n,context)[K*nc:(K+1)*nc].real
    state.fluid[0].mom+=context.compute.asarray(encode_mode(amplitude*p.species[0].mass*np.tile(base_n,3)*vflat,mode,K))
    state.fluid[0].n+=context.compute.asarray(encode_mode(amplitude*dn,mode,K))
    state.fluid[0].perp+=context.compute.asarray(encode_mode(amplitude*dpi,mode,K))
    state.fluid[1].perp+=context.compute.asarray(encode_mode(amplitude*dpperp,mode,K))
    state.fluid[1].parallel+=context.compute.asarray(encode_mode(amplitude*dppar,mode,K))
    state.B+=context.compute.asarray(encode_mode(amplitude*b,mode,K))
    meta['initial_perturbation_mapping']=mapping
    return state


def physical_identity(context):
    """Exact physical-reference and host-geometry identity for a linearization."""
    p=context.p
    digest=hashlib.sha256()
    for a in (context.geometry.R,context.geometry.Z,context.geometry.active,
              context.geometry.cells.R,context.geometry.cells.Z):
        digest.update(np.asarray(a).tobytes())
    return dict(species=[asdict(a) for a in p.species],references=asdict(p.ref),
                mu0=p.mu0,epsilon0=p.epsilon0,B_min=p.B_min,rank_rtol=p.rank_rtol,
                electron_inertia=p.electron_inertia,Nphi=context.N,geometry_sha256=digest.hexdigest())


def load_discrete_spectra(paths,context,meta,mode,frequency_hz,baseline):
    """Validate baseline spectra and optionally seed one explicitly named branch."""
    from scipy.linalg import eig
    from tools.production_spectrum import direction
    entries=[];chosen=None
    required=set(range(context.p.K+1))
    numerical={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/fomhd').glob('*.py') if p.name!='energy.py'}
    for directory in paths:
        report=json.loads((directory/'report.json').read_text())
        if any(report[k]!=v for k,v in [('kind',meta['kind']),('nodes',meta['nodes']),('K',context.p.K)]):
            raise ValueError('spectrum geometry/model identity mismatch')
        if report['source_hashes']!=numerical:raise ValueError('spectrum numerical source identity mismatch')
        with np.load(directory/'spectrum.npz',allow_pickle=False) as saved:
            for name,current,scale in [('B',baseline.B,context.p.ref.magnetic),
                                       ('n',baseline.fluid[0].n,context.p.ref.ni),
                                       ('mom',baseline.fluid[0].mom,context.p.species[0].mass*context.p.ref.ni*context.p.ref.velocity),
                                       ('pi',baseline.fluid[0].perp,context.p.ref.pressure),
                                       ('perp',baseline.fluid[1].perp,context.p.ref.pressure),
                                       ('parallel',baseline.fluid[1].parallel,context.p.ref.pressure)]:
                if 'background_'+name not in saved or saved['background_'+name].shape!=current.shape or np.max(abs(saved['background_'+name]-host(current,context)),initial=0)>1e-10*scale:
                    raise ValueError('spectrum physical background identity mismatch: '+name)
        if report.get('physical_identity')!=physical_identity(context):raise ValueError('spectrum physical references/geometry identity mismatch')
        required.discard(report['mode'])
        entries.append(dict(path=str(directory),report_sha256=hashlib.sha256((directory/'report.json').read_bytes()).hexdigest(),
                            spectrum_sha256=hashlib.sha256((directory/'spectrum.npz').read_bytes()).hexdigest(),
                            mode=report['mode'],radius=report['spectral_radius_rad_s'],maximum_growth=report['maximum_growth_per_s']))
        if report['mode']==mode and frequency_hz is not None:
            with np.load(directory/'spectrum.npz',allow_pickle=False) as data:
                values,vectors=eig(data['jacobian'])
                index=int(np.argmin(abs(values.imag-2*np.pi*frequency_hz)))
                value=values[index]
                if value.imag<=0 or abs(value.imag/(2*np.pi)-frequency_hz)>1e-5*frequency_hz:
                    raise ValueError('requested eigenfrequency is absent from the supplied spectrum')
                fields=direction(vectors[:,index],context,data['velocity_basis'],data['magnetic_basis'])
                chosen=dict(fields=fields,frequency_hz=float(value.imag/(2*np.pi)),growth_rate_per_s=float(value.real),index=index)
    if required:raise ValueError('resolved stability requires every retained nonnegative Fourier sector')
    return entries,chosen


def discrete_perturb(state,context,meta,mode,amplitude,choice):
    """An exact discrete linear direction; no independent-reference substitution."""
    fields=choice['fields'];p=context.p;nc=context.op.Nc
    bnode=(host(context.op.QB,context)@fields['B']).reshape(3,nc)
    phase=np.exp(-1j*np.angle(bnode[1,np.argmax(abs(bnode[1]))]))
    norm=max(np.max(abs(bnode))/p.ref.magnetic,np.max(abs(fields['mom']))/(p.species[0].mass*p.ref.ni*p.ref.velocity),
             np.max(abs(fields['n']))/p.ref.ni,*(np.max(abs(fields[k]))/p.ref.pressure for k in ('pi','perp','parallel')))
    physical={k:v*phase/norm for k,v in fields.items()}
    if mode==0:physical={k:v.real for k,v in physical.items()}
    for attr,key in [('n','n'),('mom','mom'),('perp','pi')]:
        a=getattr(state.fluid[0],attr);a+=context.compute.asarray(encode_mode(amplitude*physical[key],mode,p.K))
    state.fluid[1].perp+=context.compute.asarray(encode_mode(amplitude*physical['perp'],mode,p.K))
    state.fluid[1].parallel+=context.compute.asarray(encode_mode(amplitude*physical['parallel'],mode,p.K))
    state.B+=context.compute.asarray(encode_mode(amplitude*physical['B'],mode,p.K))
    context.seed_arrays={'raw_B':physical['B'],'mapped_B':physical['B'],
                         'raw_velocity':physical['mom'].reshape(3,nc).T/(p.species[0].mass*p.ref.ni),
                         'mapped_velocity':physical['mom'].reshape(3,nc).T/(p.species[0].mass*p.ref.ni)}
    meta['seed']='named production linear eigenmode; all lower-frequency candidates retained separately'
    meta['discrete_seed']={k:v for k,v in choice.items() if k!='fields'}
    return state


def snapshot_sources(output):
    paths=list((ROOT/'src/fomhd').glob('*.py'))+[Path(__file__),ROOT/'tools/cold_hall_reference.py',ROOT/'tools/global_mode_analysis.py',ROOT/'tools/production_spectrum.py']
    paths += [p for p in (ROOT/'tools/geqdsk.py',ROOT/'tools/tokamak_equilibrium.py') if p.exists()]
    hashes={}
    for path in paths:
        rel=path.relative_to(ROOT);target=output/'source'/rel
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def observation(state,frame,ledger,context,baseline):
    from fomhd.energy import summarize_energy
    nc=context.op.Nc
    return dict(time_s=state.time,step=state.step,
                energy=summarize_energy(frame.totals,0,baseline_total_J=baseline,
                                        outward_J=ledger.boundary_integrals.total_energy,source_J=0.),
                energy_raw=dict(kinetic_i_J=frame.totals.kinetic[0],kinetic_e_J=frame.totals.kinetic[1],
                                ion_internal_J=frame.totals.ion_internal,electron_perp_J=frame.totals.electron_perp,
                                electron_parallel_J=frame.totals.electron_parallel,magnetic_J=frame.totals.magnetic,
                                longitudinal_J=frame.totals.longitudinal),
                momentum=list(frame.momentum.value),
                B=host(state.B,context),BN=host(frame.BN,context),
                density=host(frame.fluid[0].n,context),velocity=host(frame.fluid[0].u,context))


def _write_report(path,report):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def run(args):
    if args.device=='cuda' and not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('CUDA requires an owned Slurm allocation')
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    report=dict(status='initializing',device=args.device,slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                source_hashes=snapshot_sources(output),scope='A0 background compressional response; no fast ions or experimental validation')
    _write_report(output/'report.json',report)
    all_runs={};context=None
    try:
        K=args.mode if args.K is None else args.K
        context,meta,eq=make_context(args.kind,args.nodes,K,args.device,args.equilibrium)
        reference=solve_reference(ReferenceCase(mode=args.mode,order=args.reference_order)) if args.kind=='circle' else None
        if reference is not None:
            meta['cold_reference']={k:v for k,v in reference.items() if isinstance(v,(str,int,float))}
        report.update(case=meta)
        initial,arrays,mapping=background(context,meta,eq)
        report['case']['background_mapping']=mapping
        np.savez_compressed(output/'input_mapping.npz',**arrays,R=context.geometry.cells.R,Z=context.geometry.cells.Z,
                            polygon=np.array([[q.R,q.Z] for q in context.geometry.polygon]))
        def clone(data):
            value=core.allocate_state(context)
            value.B=data.B.copy()
            for a in range(2):
                for key in ('n','mom','perp','parallel'):
                    setattr(value.fluid[a],key,getattr(data.fluid[a],key).copy())
            value.generation=1
            return value
        spectra,choice=load_discrete_spectra(args.spectrum,context,meta,args.mode,args.seed_frequency,initial) if args.spectrum else ([],None)
        seed=clone(initial)
        if not args.control_only:
            if choice is None:perturb(seed,context,meta,args.mode,args.amplitude,reference)
            else:discrete_perturb(seed,context,meta,args.mode,args.amplitude,choice)
            np.savez_compressed(output/'perturbation_mapping.npz',**context.seed_arrays)
        estimates=[]
        for data in ([initial] if args.control_only else [initial,seed]):
            probe=stage.evaluate_stage(data,StageTag(0,data.generation,1,0,0.,'cost_probe'),context,False)
            estimates.append(stability.stage_frequency_estimate(probe,context))
        estimate=max(estimates,key=lambda e:e['omega_estimate'])
        target_frequency=(choice['frequency_hz'] if choice is not None else reference['frequency_hz'] if reference is not None else context.p.ref.velocity/(2*np.pi*context.p.ref.length))
        horizon=args.cycles/target_frequency if args.horizon is None else args.horizon
        spectral_radius=max((entry['radius'] for entry in spectra),default=0.)
        if spectral_radius:
            # Accuracy follows the selected branch; stability covers all retained
            # baseline sectors with 50% rate headroom and rho*dt <= .5.
            dt=min(.5/(1.5*spectral_radius),1/(64*target_frequency))
            steps=max(1,int(math.ceil(horizon/dt)))
            context.p.dt_guard='warn'
        else:steps=max(1,int(math.ceil(horizon*estimate['omega_estimate']*40/(2*np.pi))))
        steps*=args.dt_divisor
        if args.steps is not None:steps=args.steps
        context.p.dt=horizon/steps;context.p.nsteps=steps
        while not stability.assess_dt('rk4',2*np.pi*target_frequency if spectra else estimate['omega_estimate'],context.p.dt,steps)['accepted'] and args.steps is None:
            steps*=2;context.p.nsteps=steps;context.p.dt=horizon/steps
        report.update(requested_steps=steps,horizon_s=horizon,dt_s=context.p.dt,initial_frequency_estimate=estimate,
                      target_periods=horizon*target_frequency,amplitude=args.amplitude,spectra=spectra,selected_frequency_hz=target_frequency,
                      timestep_policy='resolved baseline spectral stability and selected-mode accuracy; conservative estimate observe' if spectra else 'conservative operator-resolution estimate')
        (output/'resolved.json').write_text(json.dumps(asdict(context.p),indent=2,allow_nan=False)+'\n')
        # A harness configuration, distinct from the stricter verification loader.
        config={'kind':args.kind,'nodes':args.nodes,'nphi':args.mode,'K':K,'Nphi':5,
                'electron_inertia':0,'integrator':'rk4','amplitude':args.amplitude,
                'horizon_s':horizon,'steps':steps,'dt_s':context.p.dt,'diagnostic_policy':'prototype_observe'}
        comments={'kind':'Geometry family; circle or east; prototype response experiment.',
                  'nodes':'Number of enclosing cell centers per R/Z direction; staircase wall.',
                  'nphi':'Physical toroidal harmonic; 0 axisymmetric or 1 nonaxisymmetric.',
                  'K':'Largest retained Fourier mode; K=0/1; axisymmetric consistency may use K=1.',
                  'Nphi':'Angular quadrature points; fixed at five for both harmonics.',
                  'electron_inertia':'Dimensionless model switch; 0 massless/quasineutral, 1 finite; this harness requires 0.',
                  'integrator':'Time method; RK4; bounded local verification, global validation in progress.',
                  'amplitude':'Dimensionless physical field amplitude; n0 A once, n1 conjugate A/2.',
                  'horizon_s':'Common physical endpoint in seconds for control and perturbation.',
                  'steps':'Accepted RK4 intervals requested; guard remains hard.',
                  'dt_s':'Time step in seconds, from actual stage rate estimate and RK4 budgets.',
                  'diagnostic_policy':'Power, energy, momentum observe; positivity/model/solver constraints stop.'}
        (output/'case.toml').write_text('\n'.join('# '+comments[k]+' Validation: prototype response, not experimental validation.\n'+k+' = '+json.dumps(v) for k,v in config.items())+'\n')
        from contextlib import ExitStack
        experiments={}
        with ExitStack() as stack:
            for label,data in ([('control',initial)] if args.control_only else [('control',initial),('perturbed',seed)]):
                ledger=cumulative_record();captured={}
                committed=stage.initialize(clone(data),context,ledger,False,captured)
                frame=SimpleNamespace(**captured)
                np.savez_compressed(output/(label+'-initial-rhs.npz'),**io.encode_state(frame.rhs,context.geometry))
                experiments[label]=dict(committed=committed,ledger=ledger,baseline=frame.totals.energy,
                    history=[observation(committed.state,frame,ledger,context,frame.totals.energy)],
                    checks=Counter(),initial_checks=frame.checks,failures=[],elapsed_s=0.,
                    log=stack.enter_context((output/(label+'-checks.jsonl')).open('w')))
            def record(experiment,records):
                for c in records:
                    experiment['checks']['unavailable' if not c.available else 'pass' if c.passed else 'hard_fail' if c.hard else 'observe_fail']+=1
                    experiment['log'].write(json.dumps(io._check_json(c))+'\n')
            for experiment in experiments.values():record(experiment,experiment.pop('initial_checks'))
            for i in range(steps):
                if time.monotonic()-started>args.wall_seconds:
                    report['status']='budget_stopped';break
                stopped=False
                for label,experiment in experiments.items():
                    step_started=time.monotonic()
                    try:
                        if spectral_radius:
                            current=experiment['history'][-1]
                            ratio=max(float(np.max(np.linalg.norm(current['BN'].reshape(5,3,-1),axis=1)))/estimate['magnetic_max_T'],1.)
                            ratio*=max(context.p.ref.ni/float(np.min(current['density'].real)),1.)
                            if ratio>1.5 or spectral_radius*ratio*context.p.dt>.5:
                                raise ValueError('resolved-spectrum linear-neighborhood stability guard exceeded')
                        result=advance(experiment['committed'],experiment['ledger'],context)
                    except Exception as error:
                        experiment['failures'].append(dict(code=getattr(error,'code',type(error).__name__),message=str(error),step=experiment['committed'].state.step))
                        if getattr(error,'checks',None):record(experiment,error.checks)
                        payload={k:host(v,context) for k,v in (getattr(error,'partial_arrays',{}) or {}).items()}
                        if payload:np.savez_compressed(output/(label+'-failure.npz'),**payload)
                        stopped=True;break
                    record(experiment,result.checks)
                    experiment['history'].append(observation(experiment['committed'].state,result.frame,experiment['ledger'],context,experiment['baseline']))
                    experiment['elapsed_s']+=time.monotonic()-step_started
                if (i+1)%16==0:
                    print(json.dumps(dict(step=i+1,steps=steps,time_s=experiments['control']['committed'].state.time,elapsed_s=time.monotonic()-started)),flush=True)
                    for experiment in experiments.values():experiment['log'].flush()
                if stopped:break
            for label,experiment in experiments.items():
                history=experiment['history'];all_runs[label]=history
                records=[dict(time_s=h['time_s'],step=h['step'],energy=h['energy'],energy_raw=h['energy_raw'],momentum=h['momentum']) for h in history]
                (output/(label+'-history.json')).write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
                np.savez_compressed(output/(label+'-signals.npz'),time_s=np.array([h['time_s'] for h in history]),
                                    **{key:np.stack([h[key] for h in history]) for key in ('B','BN','density','velocity')})
                report[label]=dict(accepted_steps=experiment['committed'].state.step,elapsed_s=experiment['elapsed_s'],
                                   checks=dict(experiment['checks']),failures=experiment['failures'])
        _write_report(output/'report.json',report)
        if 'perturbed' in all_runs:
            c,s=all_runs['control'],all_runs['perturbed'];count=min(len(c),len(s))
            t=np.array([h['time_s'] for h in s[:count]])
            db=np.stack([s[j]['BN']-c[j]['BN'] for j in range(count)]).reshape(count,5,3,context.op.Nc)
            dn=np.stack([s[j]['density']-c[j]['density'] for j in range(count)]).reshape(count,5,context.op.Nc)
            du=np.stack([s[j]['velocity']-c[j]['velocity'] for j in range(count)]).reshape(count,5,3,context.op.Nc)
            if not np.array_equal(t,np.array([h['time_s'] for h in c[:count]])):
                raise ValueError('control and perturbed cadence mismatch')
            energy=[fluctuation_energy(s[j]['B'],c[j]['B'],s[j]['velocity'],c[j]['velocity'],c[j]['density'],context) for j in range(count)]
            phi=2*np.pi*np.arange(5)/5
            coefficient=np.einsum('tpsn,p->tsn',db,np.exp(-1j*args.mode*phi))/5
            b0=all_runs['control'][0]['BN'].reshape(5,3,-1).mean(axis=0).real
            bhat=b0/np.linalg.norm(b0,axis=0)
            parallel=np.sum(coefficient*bhat[None,:,:],axis=1)
            indices=np.argsort(np.max(np.abs(parallel),axis=0))[-min(5,context.op.Nc):]
            fits=[]
            for q in indices:
                if args.mode==0:
                    fit=fit_real_mode(t,parallel[:,q].real,) if count>=9 else dict(available=False,reason='fewer than nine samples')
                else:
                    from tools.warp_longitudinal import fit_complex_mode
                    try:
                        fit=fit_complex_mode(t,parallel[:,q])
                        measured=fit.get('measured',{})
                        if fit.get('available') and (measured.get('complex_relative_rms_residual',0)>.5 or measured.get('phase_coherence',1)<.95):
                            fit['available']=False;fit['reason']='multimode or incoherent complex signal'
                    except (ValueError,TypeError) as error:fit=dict(available=False,reason=str(error))
                fits.append(dict(cell=int(q),R_m=float(context.geometry.cells.R[q]),Z_m=float(context.geometry.cells.Z[q]),fit=fit))
            from scipy.integrate import cumulative_trapezoid
            displacement=cumulative_trapezoid(du,t,axis=0,initial=0)
            np.savez_compressed(output/'matched-response.npz',time_s=t,delta_displacement_m=displacement,delta_B=coefficient,delta_n=dn,delta_velocity=du,
                                parallel_B=parallel,probe_cells=indices,
                                **{k:np.array([r[k] for r in energy]) for k in energy[0]})
            report['mode_fits']=fits
        if report['status']!='budget_stopped':
            report['status']='completed' if all(v['accepted_steps']==steps and not v['failures'] for k,v in report.items() if k in ('control','perturbed')) else 'failed'
    except Exception as error:
        import traceback
        report['status']='failed';report['failure']=dict(code=getattr(error,'code',type(error).__name__),message=str(error),traceback=traceback.format_exc())
        if context is not None:
            payload={k:host(v,context) for k,v in (getattr(error,'partial_arrays',{}) or {}).items()}
            if payload:np.savez_compressed(output/'failure.npz',**payload)
            if getattr(error,'checks',None):
                (output/'failure-checks.json').write_text(json.dumps([io._check_json(c) for c in error.checks],indent=2)+'\n')
    report['elapsed_s']=time.monotonic()-started
    _write_report(output/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report['status']=='completed' else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=('circle','east'),default='circle')
    p.add_argument('--mode',type=int,choices=(0,1),default=0)
    p.add_argument('--K',type=int,choices=(0,1))
    p.add_argument('--nodes',type=int,default=12)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--equilibrium',type=Path)
    p.add_argument('--amplitude',type=float,default=1e-4)
    p.add_argument('--cycles',type=float,default=8.)
    p.add_argument('--horizon',type=float)
    p.add_argument('--steps',type=int)
    p.add_argument('--dt-divisor',type=int,choices=(1,2),default=1)
    p.add_argument('--reference-order',type=int,default=8)
    p.add_argument('--wall-seconds',type=float,default=3600)
    p.add_argument('--control-only',action='store_true')
    p.add_argument('--spectrum',type=Path,nargs='*',default=[])
    p.add_argument('--seed-frequency',type=float)
    args=p.parse_args()
    if args.mode>(args.K if args.K is not None else args.mode):p.error('K must retain the requested mode')
    if not 0<=args.amplitude<=.01 or args.cycles<=0 or args.wall_seconds<=0:p.error('invalid amplitude, cycles, or wall budget')
    raise SystemExit(run(args))


if __name__=='__main__':main()
