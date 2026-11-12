"""Read accepted RK4 records and compare their mode to independent Ritz fields."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fomhd import io, massless
from fomhd.__main__ import _context
from fomhd.types import StageTag
from fomhd.backend import Backend


def load_run(path):
    path = Path(path)
    case = io.load_case(path/'case.toml',backend=Backend('cpu'))
    manifest=json.loads((path/'run/run.json').read_text())
    if manifest['input_hashes'] != case.input_hashes:
        raise ValueError("Input fingerprint mismatch")
    op,geom=case.operators,case.geometry
    n=op.Nc
    phase_nodes=np.linspace(0,2*np.pi,9,endpoint=False)
    angular=np.exp(1j*phase_nodes[:,None]*np.arange(-1,2))
    times=[]
    series=[]
    modal_b=[]
    modal_u=[]
    total=[]
    for file in sorted((path/'run/snapshots').glob('state_*.npz')):
        with np.load(file,allow_pickle=False) as data:
            assert bool(data['accepted']) and data['matrix_fingerprint'].item()==manifest['matrix_fingerprint']
            state=io.decode_state(data,geom,1)
            b=(op.QB @ state.B.reshape(3,-1).T).T.reshape(3,3,n)
            e=(op.H @ data['derived_E'].reshape(3,-1).T).T.reshape(3,3,n) if 'derived_E' in data else None
            mom=state.fluid[0].mom.reshape(3,3,n)
            ion=state.fluid[0]
            electron=state.fluid[1]
            den=ion.n.reshape(3,n)
            velocity=(angular@mom.reshape(3,-1)).real.reshape(-1,3,n)/(case.parameters.species[0].mass*(angular@den).real[:,None,:])
            u_coeff=np.einsum('a,aij->ij',np.exp(-1j*phase_nodes),velocity)/phase_nodes.size
            modal_b.append(2*b[2])
            modal_u.append(2*u_coeff)
            times.append(float(data['time']))
            series.append(file)
            total.append(dict(
                density=den.sum(axis=0).real,
                ion_pressure=ion.perp.reshape(3,n).sum(axis=0).real,
                electron_perp=electron.perp.reshape(3,n).sum(axis=0).real,
                electron_parallel=electron.parallel.reshape(3,n).sum(axis=0).real,
                velocity=velocity[0],
                B=b.sum(axis=0).real,
                E=e.sum(axis=0).real if e is not None else None,
            ))
    with (path/'run/history.csv').open() as stream:
        history=list(csv.DictReader(stream))
    guard=[json.loads(row) for row in (path/'run/time_estimates.jsonl').read_text().splitlines()]
    return dict(path=path,case=case,op=op,geometry=geom,manifest=manifest,
                time=np.asarray(times),B=np.asarray(modal_b),u=np.asarray(modal_u),
                total=total,history=history,guard=guard,snapshot_files=series,
                source=json.loads((path/'source.json').read_text()))


def weighted_overlap(first,second,w):
    norm=np.sum(w*abs(first)**2)*np.sum(w*abs(second)**2)
    return float(abs(np.sum(w*np.conj(first)*second))/max(np.sqrt(norm),1e-300))


def project_mode(signal,seed,w):
    return (signal*w) @ np.conj(seed)/np.sum(w*abs(seed)**2)


def fit_frequency(times, coefficient):
    phase=np.unwrap(np.angle(coefficient))
    scale=max(times[-1]-times[0],1e-300)
    omega,phi0=np.polyfit((times-times[0])/scale,phase,1)
    defect=phase-(omega*(times-times[0])/scale+phi0)
    return dict(omega_rad_s=float(omega/scale),frequency_hz=float(omega/scale/(2*np.pi)),
                phase_rms_rad=float(np.sqrt(np.mean(defect**2))),
                relative_amplitude_drift=float((abs(coefficient[-1])-abs(coefficient[0]))/max(abs(coefficient[0]),1e-300)),
                cycles=float(abs(phase[-1]-phase[0])/(2*np.pi)))


def panel(ax,geom,values,caption,*,cmap='viridis',clim=None):
    shape=(len(geom.Z),len(geom.R))
    low,high=(None,None) if clim is None else clim
    im=ax.pcolormesh(geom.cells.R.reshape(shape),geom.cells.Z.reshape(shape),values.reshape(shape),
                    shading='nearest',cmap=cmap,vmin=low,vmax=high)
    # Black frame denotes actual grid-aligned conducting wall.
    hr,hz=geom.R[1]-geom.R[0],geom.Z[1]-geom.Z[0]
    r=[geom.R[0]-hr/2,geom.R[-1]+hr/2]
    z=[geom.Z[0]-hz/2,geom.Z[-1]+hz/2]
    ax.plot([r[0],r[1],r[1],r[0],r[0]],[z[0],z[0],z[1],z[1],z[0]],color='black',lw=1.4)
    ax.set_aspect('equal')
    ax.set_xlabel('R (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title(caption)
    return im


def load_total_E(data,index):
    total=data['total'][index]
    if total['E'] is not None:
        return total['E']
    case,op=data['case'],data['op']
    with np.load(data['snapshot_files'][index],allow_pickle=False) as record:
        state=io.decode_state(record,case.geometry,1)
        state.time=float(record['time'])
        state.step=int(record['step'])
        state.generation=int(record['generation'])
    context=_context(case)
    tag=StageTag(step=state.step,generation=state.generation,trial=0,
                 spectral_context=context.spectral_id,time=state.time,stage='plot')
    stage=massless.evaluate_stage(state,tag,context)
    edge=stage.cache.E.reshape(3,-1)
    total['E']=(op.H@edge.T).sum(axis=1).reshape(3,-1).real
    return total['E']


def plot_fields(out,geom,total,control,time,control_label):
    fig,axes=plt.subplots(4,3,figsize=(10.3,12),layout='constrained')
    rows=(
        (total['density']/1e20,control['density']/1e20,'Ion density',r'$10^{20}$ m$^{-3}$'),
        (np.linalg.norm(total['B'],axis=0),np.linalg.norm(control['B'],axis=0),'Magnetic field magnitude','T'),
        (np.linalg.norm(total['velocity'],axis=0),np.linalg.norm(control['velocity'],axis=0),'Ion speed','m/s'),
        (np.linalg.norm(total['E'],axis=0),np.linalg.norm(control['E'],axis=0),'Electric field magnitude','V/m'),
    )
    for row,(q,p,label,unit) in zip(axes,rows,strict=True):
        low=0.0 if label=='Ion density' else min(q.min(),p.min())
        shared=(low,max(q.max(),p.max()))
        im=panel(row[0],geom,p,control_label,clim=shared)
        panel(row[1],geom,q,'Perturbed total',clim=shared)
        fig.colorbar(im,ax=row[:2],label=f'{label} ({unit})')
        limit=max(abs(q-p).max(),np.finfo(float).tiny)
        im=panel(row[2],geom,q-p,'Perturbed − control',cmap='RdBu_r',clim=(-limit,limit))
        fig.colorbar(im,ax=row[2],label=unit)
    fig.suptitle(f'Physical fields at φ=0 and t={time*1e6:.3f} μs')
    fig.savefig(out/'total-fields.png',dpi=145)
    plt.close(fig)

    fig,axes=plt.subplots(3,3,figsize=(10.5,9),layout='constrained')
    for row,key,name in zip(axes,('ion_pressure','electron_perp','electron_parallel'),
                            ('Ion pressure','Electron perpendicular pressure',
                             'Electron parallel pressure'),strict=True):
        control_p,total_p=control[key],total[key]
        low=0.0
        high=max(control_p.max(),total_p.max())
        im=panel(row[0],geom,control_p,control_label,clim=(low,high))
        panel(row[1],geom,total_p,'Perturbed total',clim=(low,high))
        fig.colorbar(im,ax=row[:2],label=f'{name} (Pa)')
        diff=total_p-control_p
        limit=max(abs(diff).max(),np.finfo(float).eps)
        im=panel(row[2],geom,diff,'Perturbed − control',cmap='RdBu_r',clim=(-limit,limit))
        fig.colorbar(im,ax=row[2],label='Pa')
    fig.suptitle(f'Physical pressure at φ=0 and t={time*1e6:.3f} μs')
    fig.savefig(out/'total-pressure.png',dpi=145)
    plt.close(fig)



def review(path, output, quiet=None):
    d=load_run(path)
    output.mkdir(parents=True,exist_ok=True)
    n=d['op'].Nc
    with np.load(Path(path)/'seed.npz') as seed:
        b_ref=seed['B_reference'].reshape(3,n)
        v_ref=seed['u_reference'].reshape(3,n)
    b=d['B'];u=d['u'];times=d['time'];w=d['op'].M0
    if times.size<2:
        raise ValueError('At least two accepted snapshots are needed for a time comparison')
    coeff=project_mode(b[:,1],b[0,1],w)
    fit=fit_frequency(times,coeff)
    last=(times[-1]+times[0])/2
    index=np.flatnonzero(times>=last)
    z=np.exp(-1j*fit['omega_rad_s']*times[index])
    b_time=np.einsum('k,kij->ij',z,b[index])/index.size
    u_time=np.einsum('k,kij->ij',z,u[index])/index.size
    ref_frequency=d['source']['reference_frequency_hz']
    ref_phase=np.exp(-1j*np.angle(b_ref[1,np.argmax(abs(b_ref[1]))]))
    act_phase=np.exp(-1j*np.angle(np.sum(w*np.conj(b_ref[1]*ref_phase)*b_time[1])))
    scale=lambda x: np.sqrt(np.average(abs(x)**2,weights=w))
    normalized=[b_ref[1]*ref_phase/scale(b_ref[1]),b_time[1]*act_phase/scale(b_time[1])]
    geom=d['geometry']
    wall=np.minimum.reduce((geom.cells.R-geom.polygon[0].R,geom.polygon[1].R-geom.cells.R,
                            geom.cells.Z-geom.polygon[0].Z,geom.polygon[2].Z-geom.cells.Z))
    interior=wall>=.2
    fit['reference_relative_frequency_error']=fit['frequency_hz']/ref_frequency-1
    fit['parallel_shape_overlap_full']=weighted_overlap(b_ref[1],b_time[1],w)
    fit['parallel_shape_overlap_wall_excluded']=weighted_overlap(b_ref[1,interior],b_time[1,interior],w[interior])
    fit['all_B_overlap_full']=weighted_overlap(b_ref,b_time,w)
    fit['all_u_overlap_full']=weighted_overlap(v_ref,u_time,w)
    fit['parallel_magnetic_fraction']=float(np.sum(w*abs(b_time[1])**2)/np.sum(w*abs(b_time)**2))
    fit['independent_parallel_magnetic_fraction']=float(np.sum(w*abs(b_ref[1])**2)/np.sum(w*abs(b_ref)**2))
    fit['velocity_field_orthogonal_to_seed']=float(np.sqrt(max(0,1-weighted_overlap(u[0],u_time,w)**2)))

    fit['interior_modal_magnetic_energy_fraction']=float(
        np.sum(w[interior]*abs(b_time[:,interior])**2)/np.sum(w*abs(b_time)**2))
    component_fraction=lambda q: (np.sum(w*abs(q)**2,axis=1)/np.sum(w*abs(q)**2)).tolist()
    fit['magnetic_component_fractions_R_phi_Z']=component_fraction(b_time)
    fit['reference_magnetic_component_fractions_R_phi_Z']=component_fraction(b_ref)
    fit['velocity_component_fractions_R_phi_Z']=component_fraction(u_time)
    fit['reference_velocity_component_fractions_R_phi_Z']=component_fraction(v_ref)
    fit['velocity_phase_relative_to_magnetic_vs_reference_rad']=float(np.angle(
        np.sum(w*np.conj(v_ref)*u_time)/np.sum(w*np.conj(b_ref)*b_time)))

    # Sparse individual probes cross-check the global phase fit.
    top=np.argsort(abs(b[0,1]))[-5:]
    fit['local_probe_frequencies_hz']=[fit_frequency(times,b[:,1,i])['frequency_hz'] for i in top]
    fig,axes=plt.subplots(1,3,figsize=(10.5,4),layout='constrained')
    for ax,field,label in zip(axes,(*normalized,normalized[1]-normalized[0]),
                              ('Independent cold Hall Ritz','Accepted RK4, demodulated','RK4 − Ritz'),strict=True):
        im=panel(ax,geom,field.real,label,cmap='RdBu_r',clim=(-3,3))
    fig.colorbar(im,ax=axes,label=r'Re($\delta B_\parallel$), divided by its own weighted RMS')
    fig.savefig(output/'mode-structure.png',dpi=150)
    plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(10.6,6.4),layout='constrained')
    ref_vec=b_ref*ref_phase/scale(b_ref[1])
    act_vec=b_time*act_phase/scale(b_time[1])
    for k,component in enumerate(('R','φ','Z')):
        top=max(np.max(abs(ref_vec[k].real)),np.max(abs(act_vec[k].real)),1e-15)
        panel(axes[0,k],geom,ref_vec[k].real,'Ritz B'+component,cmap='RdBu_r',clim=(-top,top))
        im=panel(axes[1,k],geom,act_vec[k].real,'RK4 B'+component,cmap='RdBu_r',clim=(-top,top))
        fig.colorbar(im,ax=axes[:,k],label='Re perturbation / RMS parallel field')
    fig.suptitle('Magnetic polarization at one common mode phase')
    fig.savefig(output/'polarization.png',dpi=145)
    plt.close(fig)

    fig,axes=plt.subplots(2,1,figsize=(8,5.1),sharex=True,layout='constrained')
    axes[0].plot(times*1e6,np.unwrap(np.angle(coeff)),label='Accepted RK4')
    axes[0].plot(times*1e6,2*np.pi*ref_frequency*times,'--',label='Independent Ritz')
    axes[0].set_ylabel('Unwrapped phase (rad)')
    axes[0].legend()
    axes[1].plot(times*1e6,abs(coeff))
    axes[1].set_ylabel('Modal amplitude / initial')
    axes[1].set_xlabel('Time (μs)')
    fig.savefig(output/'phase-amplitude.png',dpi=150)
    plt.close(fig)
    matched={}
    if quiet is not None:
        control=load_run(quiet)
        qtimes={t:j for j,t in enumerate(control['time'])}
        shared=[(i,qtimes[t]) for i,t in enumerate(times) if t in qtimes]
        if not shared:
            raise ValueError('Quiet case and perturbation have no exact matched time')
        ii,jj=shared[-1]
        load_total_E(control,0);load_total_E(control,jj);load_total_E(d,ii)
        q0=control['total'][0];q1=control['total'][jj]
        matched=dict(time_s=float(times[ii]),accepted_matching_count=len(shared),
                     **{k+'_peak_quiet_drift':float(np.max(abs(q1[k]-q0[k]))) for k in q1 if q0[k] is not None and q1[k] is not None})
        plot_fields(output,geom,d['total'][ii],q1,times[ii],'Quiet total, same time')
        np.savez(output/'matched-physical-fields.npz',R=geom.cells.R,Z=geom.cells.Z,
                 time_s=times[ii],**{'pulse_'+k:v for k,v in d['total'][ii].items()},
                 **{'quiet_'+k:v for k,v in q1.items()})
    else:
        load_total_E(d,-1)
        q={k:v.copy() for k,v in d['total'][-1].items()}
        ref=d['case'].parameters
        q['B']=np.zeros_like(q['B'])
        with np.load(Path(path)/'seed.npz') as seed:
            q['B'][:]=seed['background_B'].real.reshape(3,n)
        q['density'][:]=ref.ref.ni
        q['ion_pressure'][:]=ref.ref.pressure/2
        q['electron_perp'][:]=ref.ref.pressure/2
        q['electron_parallel'][:]=ref.ref.pressure/2
        q['velocity'][:]=0
        q['E'][:]=0
        plot_fields(output,geom,d['total'][-1],q,times[-1],'Static background')
        np.savez(output/'accepted-total-and-background.npz',R=geom.cells.R,Z=geom.cells.Z,
                 time_s=times[-1],**{'pulse_'+k:v for k,v in d['total'][-1].items()},
                 **{'background_'+k:v for k,v in q.items()})
    records=d['history']
    final=records[-1]
    df=np.diff([float(x['elapsed_s']) for x in records])
    guard=d['guard']
    fit.update(reference_frequency_hz=ref_frequency, final_saved_time_s=float(times[-1]),
               accepted_saved_snapshots=int(times.size),case_status=d['manifest']['status'],
               accepted_steps=int(final['step']),elapsed_s=float(final['elapsed_s']),
               median_step_plus_output_s=float(np.median(df)),
               largest_guard_omega_dt=max(x['omega_dt'] for x in guard),
               hard_guard_passed=all(x['accepted'] for x in guard),quiet=matched)
    (output/'summary.json').write_text(json.dumps(fit,indent=2)+'\n')
    np.savez(output/'mode-data.npz',times_s=times,B=b,u=u,B_time=b_time,u_time=u_time,
             B_reference=b_ref,u_reference=v_ref,complex_coefficient=coeff)
    print(json.dumps(fit,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pulse',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--quiet',type=Path)
    args=parser.parse_args()
    review(args.pulse,args.output,args.quiet)
