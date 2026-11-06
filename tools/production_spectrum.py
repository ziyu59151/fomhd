#!/usr/bin/env python3
"""CPU finite-difference spectrum of the actual closed massless discrete RHS."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from scipy.linalg import eig, null_space

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from fomhd import core, stage
from fomhd.types import StageTag
from tools.warp_tokamak_cae import make_context, background, encode_mode, host, physical_identity


def progress(stages,name,started):
    event=dict(stage=name,elapsed_s=time.monotonic()-started,rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024)
    print(json.dumps(event),flush=True)
    stages.append(event)


def constrained_basis(context,mode):
    op=context.op;g=context.geometry;K=context.p.K
    nr=len(g.r_faces.R);nc=op.Nc
    edges=[w.face if w.component=='r_face' else nr+nc+w.face for w in g.walls]
    rows=[host(op.DB[K+mode],context),host(op.Cdag[K+mode],context)[edges]]
    if mode==0:rows.append(host(op.topology_flux_row,context).reshape(1,-1))
    A=np.vstack(rows);norm=np.linalg.norm(A,axis=1);keep=norm>0
    B=null_space(A[keep]/norm[keep,None],rcond=context.p.rank_rtol)
    V=null_space(host(op.T,context),rcond=context.p.rank_rtol)
    return V,B


def pack_rhs(frame,context,V,B,mode):
    nc=context.op.Nc;nb=context.op.Nb;idx=context.p.K+mode
    factor=2. if mode else 1.
    f=frame.rhs.fluid[0];e=frame.rhs.fluid[1];p=context.p
    values=[host(f.n,context).reshape(-1,nc)[idx]/p.ref.ni,
            V.conj().T@host(f.mom,context).reshape(-1,3*nc)[idx]/(p.species[0].mass*p.ref.ni*p.ref.velocity),
            host(f.perp,context).reshape(-1,nc)[idx]/p.ref.pressure,
            host(e.perp,context).reshape(-1,nc)[idx]/p.ref.pressure,
            host(e.parallel,context).reshape(-1,nc)[idx]/p.ref.pressure,
            B.conj().T@host(frame.rhs.B,context).reshape(-1,nb)[idx]/p.ref.magnetic]
    out=np.concatenate(values)*factor
    return out.real.astype(complex) if mode==0 else out


def direction(vector,context,V,B):
    nc=context.op.Nc;p=context.p;cursor=0
    def take(n):
        nonlocal cursor
        a=vector[cursor:cursor+n];cursor+=n;return a
    density=take(nc)*p.ref.ni
    momentum=V@take(V.shape[1])*(p.species[0].mass*p.ref.ni*p.ref.velocity)
    pi=take(nc)*p.ref.pressure;perp=take(nc)*p.ref.pressure;parallel=take(nc)*p.ref.pressure
    magnetic=B@take(B.shape[1])*p.ref.magnetic
    return dict(n=density,mom=momentum,pi=pi,perp=perp,parallel=parallel,B=magnetic)


def shifted(base,fields,weight,context,mode):
    out=core.allocate_state(context);out.B=base.B.copy();out.generation=1
    for a in range(2):
        for key in ('n','mom','perp','parallel'):
            setattr(out.fluid[a],key,getattr(base.fluid[a],key).copy())
    for key,field in (('n','n'),('mom','mom'),('perp','pi')):
        a=getattr(out.fluid[0],key);a+=encode_mode(weight*fields[field],mode,context.p.K)
    out.fluid[1].perp+=encode_mode(weight*fields['perp'],mode,context.p.K)
    out.fluid[1].parallel+=encode_mode(weight*fields['parallel'],mode,context.p.K)
    out.B+=encode_mode(weight*fields['B'],mode,context.p.K)
    return out


def compute(kind,nodes,K,mode,output,eq_path=None,epsilon=1e-5,memory_limit_bytes=24<<30):
    if type(memory_limit_bytes) is not int or memory_limit_bytes<=0:
        raise ValueError('memory_limit_bytes must be a positive integer')
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    stages=[]
    context,meta,eq=make_context(kind,nodes,K,'cpu',eq_path,memory_limit_bytes=memory_limit_bytes)
    progress(stages,'context_ready',started,)
    initial,arrays,mapping=background(context,meta,eq)
    V,B=constrained_basis(context,mode);nc=context.op.Nc
    size=4*nc+V.shape[1]+B.shape[1]
    J=np.empty((size,size),complex)
    tag=StageTag(0,1,1,0,0.,'linear_spectrum')
    baseline=stage.evaluate_stage(initial,tag,context,False)
    f0=pack_rhs(baseline,context,V,B,mode)
    for col in range(size):
        unit=np.zeros(size);unit[col]=1
        fields=direction(unit,context,V,B)
        plus=stage.evaluate_stage(shifted(initial,fields,epsilon,context,mode),tag,context,False)
        minus=stage.evaluate_stage(shifted(initial,fields,-epsilon,context,mode),tag,context,False)
        J[:,col]=(pack_rhs(plus,context,V,B,mode)-pack_rhs(minus,context,V,B,mode))/(2*epsilon)
        if col%50==0:print(json.dumps(dict(stage='jacobian',column=col,size=size,elapsed_s=time.monotonic()-started,rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024)),flush=True)
    progress(stages,'eig_start',started,)
    values,vectors=eig(J)
    progress(stages,'eig_complete',started,)
    weights=np.asarray(context.op.MB).real
    rho=context.p.species[0].mass*context.p.ref.ni
    b0=np.asarray(baseline.BN).reshape(5,3,nc).real.mean(axis=0)
    bhat=b0/np.linalg.norm(b0,axis=0)
    qb=host(context.op.QB,context)
    candidate_fields_by_index={}
    candidates=[]
    for i,value in enumerate(values):
        if value.imag<=1e-5/context.p.ref.time or abs(value.real)>.1*value.imag:continue
        fields=direction(vectors[:,i],context,V,B)
        bnode=(qb@fields['B']).reshape(3,nc)
        parallel=float(np.sum(abs(np.sum(bnode*bhat,axis=0))**2*context.op.M0))
        total=float(np.sum(abs(bnode)**2*context.op.M0))
        mb=float(np.sum(abs(fields['B'])**2*weights)/context.p.mu0)
        mv=float(np.sum(abs(fields['mom'].reshape(3,nc))**2*context.op.M0)/rho)
        if parallel/max(total,1e-300)>.5 and mb/max(mb+mv,1e-300)>.1:
            nodal_magnetic=float(np.sum(abs(bnode)**2*context.op.M0))
            staggered_magnetic=float(np.sum(abs(fields['B'])**2*weights))
            candidates.append(dict(index=i,frequency_hz=float(value.imag/(2*np.pi)),growth_rate_per_s=float(value.real),
                                   parallel_fraction=parallel/total,magnetic_fraction=mb/(mb+mv),
                                   magnetic_transfer_ratio=nodal_magnetic/max(staggered_magnetic,1e-300),
                                   MB_magnetic_norm=mb,M0_ion_kinetic_norm=mv))
            candidate_fields_by_index[i]=dict(velocity=fields['mom'].reshape(3,nc)/rho,
                                              B_node=(qb@fields['B']).reshape(3,nc),
                                              density=fields['n'])
    candidates.sort(key=lambda c:c['frequency_hz'])
    candidate_fields=[candidate_fields_by_index[item['index']] for item in candidates]
    source_root=Path(core.__file__).resolve().parent
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_root.glob('*.py') if p.name!='energy.py'}
    progress(stages,'classification_complete',started,)
    report=dict(kind=kind,nodes=nodes,K=K,mode=mode,epsilon=epsilon,size=size,
                spectral_radius_rad_s=float(np.max(abs(values))),maximum_growth_per_s=float(np.max(values.real)),
                background_rhs_scaled_linf_per_s=float(np.max(abs(f0))),
                candidates=candidates,elapsed_s=time.monotonic()-started,source_hashes=hashes,
                scope='production baseline linearization; independent cold oracle is separate; near-zero finite candidates require epsilon convergence',background_mapping=mapping,physical_identity=physical_identity(context),
                memory_limit_bytes=memory_limit_bytes,stages=stages)
    payload=dict(jacobian=J,eigenvalues=values,eigenvectors=vectors,velocity_basis=V,magnetic_basis=B,
                 background_B=np.asarray(initial.B),background_n=np.asarray(initial.fluid[0].n),
                 background_mom=np.asarray(initial.fluid[0].mom),background_pi=np.asarray(initial.fluid[0].perp),
                 background_perp=np.asarray(initial.fluid[1].perp),background_parallel=np.asarray(initial.fluid[1].parallel))
    payload.update(dict(
        candidate_indices=np.asarray([item['index'] for item in candidates],dtype=np.int64),
        candidate_frequency_hz=np.asarray([item['frequency_hz'] for item in candidates],dtype=float),
        candidate_magnetic_transfer_ratio=np.asarray([item['magnetic_transfer_ratio'] for item in candidates],dtype=float),
        candidate_MB_magnetic_norm=np.asarray([item['MB_magnetic_norm'] for item in candidates],dtype=float),
        candidate_M0_ion_kinetic_norm=np.asarray([item['M0_ion_kinetic_norm'] for item in candidates],dtype=float),
        candidate_velocity=np.asarray([item['velocity'] for item in candidate_fields],dtype=complex).reshape(-1,3,nc),
        candidate_B_node=np.asarray([item['B_node'] for item in candidate_fields],dtype=complex).reshape(-1,3,nc),
        candidate_density=np.asarray([item['density'] for item in candidate_fields],dtype=complex).reshape(-1,nc),
        R=host(context.geometry.cells.R,context).reshape(nc),Z=host(context.geometry.cells.Z,context).reshape(nc),
        M0=host(context.op.M0,context).real.reshape(nc),
        background_B_node=b0,
        rho=np.asarray(rho),mu0=np.asarray(context.p.mu0),
        wall_R=np.asarray([point.R for point in context.geometry.polygon]),
        wall_Z=np.asarray([point.Z for point in context.geometry.polygon])))
    if candidates:
        selected=candidates[0];vector=vectors[:,selected['index']]
        report['selected']=selected
        report['eigen_relative_residual']=float(np.linalg.norm(J@vector-values[selected['index']]*vector)/max(np.linalg.norm(J)*np.linalg.norm(vector),1e-300))
        payload.update({'seed_'+k:v for k,v in direction(vector,context,V,B).items()})
    np.savez_compressed(output/'spectrum.npz',**payload)
    progress(stages,'export_complete',started,)
    report['elapsed_s']=time.monotonic()-started
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def positive_int(text):
    value=int(text)
    if value<=0:raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=('circle','east'),default='circle');p.add_argument('--nodes',type=int,default=12)
    p.add_argument('--K',type=int,default=0);p.add_argument('--mode',type=int,default=0)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--equilibrium',type=Path)
    p.add_argument('--epsilon',type=float,default=1e-5);p.add_argument('--memory-gib',type=positive_int,default=24);a=p.parse_args()
    print(json.dumps(compute(a.kind,a.nodes,a.K,a.mode,a.output,a.equilibrium,a.epsilon,a.memory_gib<<30),indent=2))


if __name__=='__main__':main()
