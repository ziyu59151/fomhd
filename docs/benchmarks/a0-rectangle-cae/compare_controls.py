"""Compare accepted spatial, time-step and physical wall controls."""

from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT=Path(__file__).resolve().parent


def load(group):
    where=ROOT/('analysis-'+group)
    report=json.loads((where/'summary.json').read_text())
    with np.load(where/'mode-data.npz') as arrays:
        data=dict(arrays)
    return where,report,data


def exact_common(first,second):
    t1=first['times_s']
    t2=second['times_s']
    near=np.searchsorted(t1,t2)
    right=np.clip(near,0,len(t1)-1)
    left=np.clip(near-1,0,len(t1)-1)
    ij=np.where(abs(t1[left]-t2)<abs(t1[right]-t2),left,right)
    # Decimal SI time parameters are binary floats. Tolerate only one
    # rounding-level shift in time computed as integer step times h.
    match=abs(t1[ij]-t2)<=4*np.finfo(float).eps*np.maximum(abs(t2),1e-30)
    if match.sum()<2:
        raise ValueError('No common accepted physical time')
    return ij[match],np.flatnonzero(match)


def regular(path):
    with np.load(path/'accepted-total-and-background.npz') as fields:
        r,z=fields['R'],fields['Z']
    return np.unique(r),np.unique(z)


def similarity(coarse,fine,dx):
    rr,zz=np.meshgrid(dx+15,dx)
    weights=rr.ravel()
    xy=np.c_[zz.ravel(),rr.ravel()]
    vectors=[]
    for field,r,z in (coarse,fine):
        interpolate=RegularGridInterpolator((z,r),field.reshape(len(z),len(r)),bounds_error=True)
        vectors.append(interpolate(xy))
    x,y=vectors
    return float(abs(np.vdot(x,weights*y))/np.sqrt(np.vdot(x,weights*x).real*np.vdot(y,weights*y).real))


def wall_collar(values,r,z,thickness_m=.2):
    rr,zz=np.meshgrid(r,z)
    hr,hz=r[1]-r[0],z[1]-z[0]
    distance=np.minimum.reduce((rr-r[0]+hr/2,r[-1]+hr/2-rr,
                               zz-z[0]+hz/2,z[-1]+hz/2-zz)).ravel()
    inside=distance>=thickness_m
    weights=rr.ravel()
    ref,actual=values['B_reference'][1],values['B_time'][1]

    def energy(field,region):
        return np.sum(weights[region]*abs(field[region])**2)

    collar=~inside
    norm=np.sqrt(energy(ref,collar)*energy(actual,collar))
    return dict(
        thickness_m=thickness_m,
        reference_parallel_energy_fraction=float(energy(ref,collar)/energy(ref,slice(None))),
        accepted_parallel_energy_fraction=float(energy(actual,collar)/energy(actual,slice(None))),
        parallel_shape_overlap=float(abs(np.vdot(ref[collar],weights[collar]*actual[collar]))/norm),
    )


def control_summary():
    coarse_path,g12,s12=load('r15-g12')
    fine_path,g16,s16=load('r15-g16')
    _,half,sh=load('r15-g16-half')
    wide_path,wide,sw=load('r15-wide')
    i,j=exact_common(s16,sh)
    b_diff=s16['B'][i]-sh['B'][j]
    u_diff=s16['u'][i]-sh['u'][j]
    phase_delta=np.angle(s16['complex_coefficient'][i]*np.conj(sh['complex_coefficient'][j]))
    t=s16['times_s'][i]
    g12_r,g12_z=regular(coarse_path)
    with np.load(fine_path/'matched-physical-fields.npz') as arrays:
        g16_r,g16_z=np.unique(arrays['R']),np.unique(arrays['Z'])
    wide_r,wide_z=regular(wide_path)
    p=(s12['B_time'][1],g12_r,g12_z)
    q=(s16['B_time'][1],g16_r,g16_z)
    shape_common=similarity(p,q,np.linspace(-.54,.54,100))
    shape_inner=similarity(p,q,np.linspace(-.4,.4,100))
    summary=dict(
        refinement=dict(
            relative_frequency_shift_12_to_16=g16['frequency_hz']/g12['frequency_hz']-1,
            common_support_parallel_shape_overlap=shape_common,
            inner_parallel_shape_overlap_away_0p2m=shape_inner,
        ),
        half_step=dict(
            matched_times=len(t),final_matched_time_s=float(t[-1]),
            peak_delta_B_T=float(np.max(abs(b_diff))),
            peak_delta_u_m_per_s=float(np.max(abs(u_diff))),
            peak_relative_delta_B=float(np.max(abs(b_diff))/np.max(abs(s16['B'][i]))),
            peak_relative_delta_u=float(np.max(abs(u_diff))/np.max(abs(s16['u'][i]))),
            peak_phase_gap_rad=float(np.max(abs(phase_delta))),
        ),
        wall_distance=dict(
            narrow_box_radius_m=.6,wide_box_radius_m=.75,common_cell_spacing_m=.075,
            relative_frequency_shift_observed=wide['frequency_hz']/g16['frequency_hz']-1,
            relative_frequency_shift_independent=wide['reference_frequency_hz']/g16['reference_frequency_hz']-1,
        ),
        wall_collar=dict(g12=wall_collar(s12,g12_r,g12_z),g16=wall_collar(s16,g16_r,g16_z),
                         g20_wide=wall_collar(sw,wide_r,wide_z)),
        cases={'g12':g12,'g16':g16,'g16_half':half,'g20_wide':wide},
    )
    (ROOT/'control-comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(12.1,3.8),layout='constrained')
    grids=[12,16]
    axes[0].plot(grids,[g12['frequency_hz']/1e6,g16['frequency_hz']/1e6],'o-',label='Full RK4')
    axes[0].axhline(g16['reference_frequency_hz']/1e6,color='black',ls='--',label='Independent Ritz')
    axes[0].set(xlabel='Cells along each side',ylabel='Frequency (MHz)',title='Fixed physical box')
    axes[0].set_xticks(grids)
    axes[0].legend()
    axes[1].plot(1e6*t,1e6*phase_delta,'o-')
    axes[1].set(xlabel='Time (μs)',ylabel='Phase gap (μrad)',title='130 ps versus 65 ps')
    axes[2].plot([.6,.75],[g16['frequency_hz']/1e6,wide['frequency_hz']/1e6],'o-',label='Full RK4')
    axes[2].plot([.6,.75],[g16['reference_frequency_hz']/1e6,wide['reference_frequency_hz']/1e6],'s--',label='Independent Ritz')
    axes[2].set(xlabel='Wall distance from box center (m)',ylabel='Frequency (MHz)',title='Fixed 0.075 m cells')
    axes[2].legend()
    fig.savefig(ROOT/'refinement-time-wall-controls.png',dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11,6.5),layout='constrained')
    for column,(values,r,z,description) in enumerate((
            (s12,g12_r,g12_z,'12×12 fixed box'),
            (s16,g16_r,g16_z,'16×16 fixed box'),
            (sw,wide_r,wide_z,'20×20 wider wall'))):
        rr,zz=np.meshgrid(r,z)
        w=rr.ravel()
        reference=values['B_reference'][1]
        actual=values['B_time'][1]
        reference=reference*np.exp(-1j*np.angle(reference[np.argmax(abs(reference))]))
        actual=actual*np.exp(-1j*np.angle(np.vdot(reference,w*actual)))
        hr,hz=r[1]-r[0],z[1]-z[0]
        limits=(r[0]-hr/2,r[-1]+hr/2,z[0]-hz/2,z[-1]+hz/2)
        for row,(field,label) in enumerate(((reference,'Independent Ritz'),(actual,'Accepted RK4'))):
            ax=axes[row,column]
            amp=np.sqrt(np.average(abs(field)**2,weights=w))
            im=ax.pcolormesh(rr,zz,field.real.reshape(rr.shape)/amp,shading='nearest',
                             cmap='RdBu_r',vmin=-3,vmax=3)
            rl,rh,zl,zh=limits
            ax.plot([rl,rh,rh,rl,rl],[zl,zl,zh,zh,zl],'k-',lw=1.2)
            ax.plot([rl+.2,rh-.2,rh-.2,rl+.2,rl+.2],
                    [zl+.2,zl+.2,zh-.2,zh-.2,zl+.2],'k:',lw=.9)
            ax.set_aspect('equal')
            ax.set(xlabel='R (m)',ylabel='Z (m)',title=f'{description} · {label}')
    fig.colorbar(im,ax=axes,label='Re parallel magnetic perturbation / weighted RMS')
    fig.savefig(ROOT/'refinement-structure.png',dpi=155)
    plt.close(fig)



if __name__=='__main__':
    control_summary()
