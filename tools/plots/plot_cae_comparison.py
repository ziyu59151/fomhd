#!/usr/bin/env python3
"""Offline frequency/control comparisons for explicitly named CAE cases."""
from __future__ import annotations
import argparse
import hashlib
import json
import tomllib
from pathlib import Path
import numpy as np


def backend_pairing(base, other, base_resolved, other_resolved):
    """Require identical physics, seed and cadence before backend comparison."""
    for key in ('source_hashes','amplitude','selected_frequency_hz','dt_s'):
        if not base.get(key) or base.get(key)!=other.get(key):
            return dict(available=False,reason=f'backend provenance differs or is missing: {key}')
    # Projection residuals differ by backend roundoff; the prescribed geometry,
    # physical input and seed identity must still match exactly.
    a={k:v for k,v in base.get('case',{}).items() if k!='background_mapping'}
    b={k:v for k,v in other.get('case',{}).items() if k!='background_mapping'}
    if not a or a!=b or not base_resolved or base_resolved!=other_resolved:
        return dict(available=False,reason='backend physical configuration differs or is missing')
    return dict(available=True)


def summarize_probes(report):
    values=[];residuals=[];cycles=[];reasons=[]
    for item in report.get('mode_fits',[]):
        fit=item['fit']
        if not fit.get('available'):
            reasons.append(fit.get('reason','unavailable'));continue
        measured=fit.get('measured',{})
        frequency=fit.get('frequency_hz',measured.get('frequency_hz'))
        if frequency is None or not np.isfinite(frequency):
            reasons.append('nonfinite frequency');continue
        values.append(abs(float(frequency)))
        residuals.append(fit.get('relative_residual',measured.get('complex_relative_rms_residual')))
        cycles.append(fit.get('measured_cycles',fit.get('complete_measured_cycles')))
    if not values:return dict(available=False,reasons=sorted(set(reasons)))
    center=float(np.median(values))
    return dict(available=True,frequency_hz=center,probe_count=len(values),
                probe_spread_ppm=float((max(values)-min(values))/center*1e6),
                minimum_measured_cycles=float(min(v for v in cycles if v is not None)),
                maximum_relative_residual=float(max(v for v in residuals if v is not None)))


def common_difference(base,fine,weights,*,stride=1,amplitude_scale=1.):
    times=np.asarray(base['time_s']);other=np.asarray(fine['time_s'])[::stride]
    if not np.array_equal(times,other):
        return dict(available=False,reason='different cadence; no interpolation performed')
    a=np.asarray(base['delta_B']);b=np.asarray(fine['delta_B'])[::stride]*amplitude_scale
    if a.shape!=b.shape:return dict(available=False,reason='different grid/state shapes')
    error=(a-b)*np.sqrt(weights)[None,None,:]
    reference=a*np.sqrt(weights)[None,None,:]
    return dict(available=True,relative_history_L2=float(np.linalg.norm(error)/max(np.linalg.norm(reference),1e-300)),
                relative_endpoint_L2=float(np.linalg.norm(error[-1])/max(np.linalg.norm(reference[-1]),1e-300)))


def generate(root,case_names,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=False)
    hashes={};rows=[];arrays={};reports={};resolved={}
    def read(path):
        hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix=='.json':return json.loads(path.read_text())
        if path.suffix=='.toml':return tomllib.loads(path.read_text())
        with np.load(path,allow_pickle=False) as archive:return {k:archive[k].copy() for k in archive.files}
    for name in case_names:
        directory=root/name;path=directory/'report.json'
        if not path.exists():
            rows.append(dict(name=name,status='missing',frequency=dict(available=False,reasons=['missing report'])));continue
        r=read(path);case=r.get('case',{});freq=summarize_probes(r)
        reports[name]=r
        resolved[name]=read(directory/'resolved.json') if (directory/'resolved.json').exists() else {}
        row=dict(name=name,status=r.get('status'),backend=r.get('device'),kind=case.get('kind'),nodes=case.get('nodes'),
                 K=case.get('K'),frequency=freq,dt_s=r.get('dt_s'),
                 discrete_frequency_hz=r.get('selected_frequency_hz'),
                 cold_reference_hz=case.get('cold_reference',{}).get('frequency_hz'))
        config=read(directory/'case.toml') if (directory/'case.toml').exists() else {}
        mode=case.get('nphi',config.get('nphi',0 if case.get('K')==0 else 1))
        row['mode']=mode
        if freq.get('available'):
            for key,ref in [('relative_discrete_error',row['discrete_frequency_hz']),('relative_cold_error',row['cold_reference_hz'])]:
                if ref:row[key]=freq['frequency_hz']/ref-1
        signal_path=directory/'matched-response.npz'
        if signal_path.exists():arrays[name]=read(signal_path)
        history_path=directory/'perturbed-history.json'
        if history_path.exists():
            h=read(history_path);e=np.array([x['energy']['balance_J'] for x in h],float)
            total0=h[0]['energy']['total_J'];row['max_abs_energy_balance_J']=float(max(abs(e)))
            row['max_balance_over_background']=float(max(abs(e))/total0)
            if name in arrays:
                a=arrays[name];signal=a['perturbation_magnetic_J']+a['perturbation_ion_kinetic_J']
                row['max_quadratic_mode_signal_J']=float(max(signal))
                row['max_balance_over_mode_signal']=float(max(abs(e))/max(max(signal),1e-300))
            control_path=directory/'control-history.json'
            if control_path.exists():
                control=read(control_path)
                if [x['time_s'] for x in h]==[x['time_s'] for x in control]:
                    matched=e-np.array([x['energy']['balance_J'] for x in control])
                    row['max_abs_matched_energy_balance_J']=float(max(abs(matched)))
                    if name in arrays:
                        row['max_matched_balance_over_mode_signal']=float(max(abs(matched))/max(max(signal),1e-300))
        mapping=directory/'input_mapping.npz'
        if mapping.exists() and name in arrays:
            a=read(mapping);arrays[name]['cell_weights']=a['R']*case['h_R_m']*case['h_Z_m']
        rows.append(row)
    comparisons=[]
    for row in rows:
        name=row['name']
        if '-base-cpu-' not in name or name not in arrays:continue
        for variant,stride,factor in [('halfdt',2,1.),('halfamp',1,2.)]:
            other=name.replace('-base-cpu-',f'-{variant}-cpu-')
            if other in arrays:
                comparisons.append(dict(base=name,other=other,kind=variant,
                    **common_difference(arrays[name],arrays[other],arrays[name]['cell_weights'],stride=stride,amplitude_scale=factor)))
        gpu=[r['name'] for r in rows if r.get('backend')=='cuda' and r.get('kind')==row.get('kind') and r.get('nodes')==row.get('nodes') and r.get('mode')==row.get('mode') and r['name'] in arrays]
        for other in gpu:
            identity=backend_pairing(reports[name],reports[other],resolved[name],resolved[other])
            if not identity['available']:
                comparisons.append(dict(base=name,other=other,kind='explicit_cpu_cuda_common_prefix',**identity))
                continue
            # A budget-limited CUDA run still provides an exact common prefix.
            count=min(len(arrays[name]['time_s']),len(arrays[other]['time_s']))
            a={k:v[:count] if k!='cell_weights' else v for k,v in arrays[name].items()}
            b={k:v[:count] for k,v in arrays[other].items()}
            comparisons.append(dict(base=name,other=other,kind='explicit_cpu_cuda_common_prefix',**common_difference(a,b,arrays[name]['cell_weights'])))
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for k,row in enumerate(rows):
        f=row['frequency']
        if not f.get('available'):continue
        label=row['name'].replace('formal-','').replace('-r1','').replace('-r2','')
        axes[0].plot(k,f['frequency_hz']/1e6,'o',label=label)
        if row.get('cold_reference_hz'):axes[0].plot(k,row['cold_reference_hz']/1e6,'k_',ms=13)
        if row.get('relative_discrete_error') is not None:axes[1].plot(k,row['relative_discrete_error']*1e6,'o')
    for ax in axes:ax.grid(alpha=.25);ax.set_xlabel('Explicit case index (see summary.json)')
    axes[0].set_ylabel('Measured frequency [MHz]; black dash: independent cold reference')
    axes[1].set_ylabel('Observed minus named discrete frequency [ppm]')
    axes[0].legend(fontsize=6,ncol=2,loc='best')
    fig.suptitle('A0 cold-fast branch response: frequency and numerical controls')
    for ext in ('png','svg'):fig.savefig(output/('frequency-comparison.'+ext),dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for k,row in enumerate(rows):
        for ax,key in [(axes[0],'max_balance_over_background'),(axes[1],'max_balance_over_mode_signal')]:
            value=row.get(key)
            if value is not None:ax.plot(k,value,'o')
        matched=row.get('max_matched_balance_over_mode_signal')
        if matched is not None:axes[1].plot(k,matched,'x',color='tab:green')
    for ax in axes:ax.set_yscale('symlog',linthresh=1e-15);ax.grid(alpha=.25);ax.set_xlabel('Explicit case index (see summary.json)')
    axes[0].set_ylabel('Maximum |energy balance| / initial total energy')
    axes[1].set_ylabel('|Energy balance| / mode signal; green x: minus control')
    fig.suptitle('Energy budget on background and perturbation scales (not pass gates)')
    for ext in ('png','svg'):fig.savefig(output/('energy-comparison.'+ext),dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,(family,mode) in zip(axes.flat,[('circle',0),('circle',1),('east',0),('east',1)]):
        candidates=[r for r in rows if r.get('kind')==family and r.get('mode')==mode
                    and r.get('nodes')==12 and '-base-cpu-' in r['name'] and r['name'] in arrays]
        if not candidates:
            ax.text(.5,.5,'No archived base response',ha='center',transform=ax.transAxes)
            continue
        row=candidates[0];a=arrays[row['name']];signal=np.asarray(a['parallel_B'])
        cell=int(np.argmax(np.max(abs(signal),axis=0)));scale=float(max(abs(signal[:,cell])))
        ax.plot(a['time_s']*1e9,signal[:,cell].real/max(scale,1e-300),label='CPU: full window')
        for comparison in comparisons:
            if comparison['base']==row['name'] and comparison['kind']=='explicit_cpu_cuda_common_prefix' and comparison['available']:
                b=arrays[comparison['other']]
                ax.plot(b['time_s']*1e9,np.asarray(b['parallel_B'])[:,cell].real/max(scale,1e-300),
                        '--',label='CUDA: accepted window',alpha=.7)
        ax.set(title=f'{family}, n_phi={mode}, grid 12 x 12, cell {cell}: matched magnetic compression',
               xlabel='Time [ns]',ylabel='Re(delta B parallel) / peak magnitude')
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('FOMHD time-evolved responses after matched-control subtraction; n_phi=1 retains A/2')
    for ext in ('png','svg'):fig.savefig(output/('matched-waves.'+ext),dpi=160)
    plt.close(fig)
    result=dict(cases=rows,comparisons=comparisons,input_sha256=hashes,
                scope='Observed global responses; discrete seeding and independent cold reference are distinct')
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--cases',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    generate(a.run_root,a.cases.split(','),a.output)
