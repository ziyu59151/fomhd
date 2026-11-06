"""Independent Fourier/Ritz cold Hall modes with an explicitly constrained wall.

Host-only analysis: no production geometry, differential matrix, or RHS import.
The weak wall restrictions include their orthogonal reactions. They are not a
claim of equivalence to the unreacted smooth Hall boundary-value problem.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial import polynomial as poly
from scipy.linalg import eigh, null_space, svd

MU0=1.25663706127e-6
E=1.602176634e-19
MP=1.67262192595e-27


@dataclass(frozen=True)
class ReferenceCase:
    mode: int = 0
    order: int = 5
    R0: float = 3.0
    a: float = 1.0
    B0: float = 3.4
    density: float = 3.423e18
    mass: float = 2*MP
    rank_rtol: float = 1e-11

    def __post_init__(self):
        if self.mode not in (0,1) or not 2<=self.order<=12:
            raise ValueError('reference requires mode 0/1 and order in [2,12]')
        if not self.R0>self.a>0 or min(self.B0,self.density,self.mass)<=0:
            raise ValueError('invalid circular reference physical scales')
        if not 0<self.rank_rtol<1e-5:
            raise ValueError('invalid reference mass rank threshold')


def disk_quadrature(case,radial_points=None,angular_points=None):
    """Return flattened R,Z and R*r*dr*dtheta weights (one radian in phi)."""
    nr=radial_points or (2*case.order+8)
    nt=angular_points or (8*case.order+32)
    g,w=np.polynomial.legendre.leggauss(nr)
    r=(g+1)*case.a/2
    theta=2*np.pi*np.arange(nt)/nt
    R=case.R0+r[:,None]*np.cos(theta)
    Z=np.broadcast_to(r[:,None]*np.sin(theta),(nr,nt))
    weights=R*r[:,None]*(w[:,None]*case.a/2)*(2*np.pi/nt)
    return R.ravel(),Z.ravel(),weights.ravel()


def _terms(case):
    d=case.order+4
    def scalar(items):
        c=np.zeros((d,d))
        for i,j,value in items:c[i,j]=value
        return c
    one=scalar([(0,0,1)])
    x=scalar([(1,0,1)]);z=scalar([(0,1,1)])
    bubble=scalar([(0,0,1),(2,0,-1),(0,2,-1)])
    def mul(c,i,j):
        out=np.zeros_like(c)
        out[i:,j:]=c[:d-i,:d-j]
        return out
    velocities=[];potentials=[]
    for degree in range(case.order+1):
        for i in range(degree+1):
            j=degree-i
            f=mul(one,i,j);bf=mul(bubble,i,j)
            for component in range(3):
                a=np.zeros((3,d,d));a[component]=bf
                potentials.append((a,0))
            a=np.zeros((3,d,d));a[0]=mul(x,i,j);a[2]=mul(z,i,j)
            potentials.append((a,0))
            for component in (0,2):
                v=np.zeros((3,d,d));v[component]=bf
                velocities.append((v,0))
            v=np.zeros((3,d,d));v[0]=-mul(z,i,j);v[2]=mul(x,i,j)
            velocities.append((v,0))
            v=np.zeros((3,d,d));v[1]=f
            velocities.append((v,0))
    # Relative potential, zero flux, with the missing nonzero constant R*b_phi trace.
    a=np.zeros((3,d,d));a[0]=-mul(bubble,0,1)*case.a
    a[2]=mul(bubble,1,0)*case.a
    potentials.append((a,1))
    return velocities,potentials


def _jets(case,terms,R,Z):
    R,Z=np.broadcast_arrays(np.asarray(R),np.asarray(Z))
    R=R.ravel();Z=Z.ravel();x=(R-case.R0)/case.a;z=Z/case.a
    result={key:np.zeros((R.size,3,len(terms)),dtype=complex)
            for key in ('v','r','z','rr','rz','zz')}
    for b,(coeff,power) in enumerate(terms):
        for c in range(3):
            p=coeff[c]
            def ev(dr=0,dz=0):
                q=poly.polyder(poly.polyder(p,dr,axis=0),dz,axis=1)
                return poly.polyval2d(x,z,q)/case.a**(dr+dz)
            f,fr,fz,frr,frz,fzz=ev(),ev(1),ev(0,1),ev(2),ev(1,1),ev(0,2)
            den=R**power
            result['v'][:,c,b]=f/den
            result['r'][:,c,b]=(fr-power*f/R)/den
            result['z'][:,c,b]=fz/den
            result['rr'][:,c,b]=(frr-2*power*fr/R+power*(power+1)*f/R**2)/den
            result['rz'][:,c,b]=(frz-power*fz/R)/den
            result['zz'][:,c,b]=fzz/den
    return result


def basis_fields(case,R,Z):
    """Evaluate analytic basis values/derivatives at arbitrary host positions."""
    R,Z=np.broadcast_arrays(np.asarray(R),np.asarray(Z));R=R.ravel();Z=Z.ravel()
    vt,at=_terms(case);v=_jets(case,vt,R,Z);a=_jets(case,at,R,Z)
    rr=R[:,None];ik=1j*case.mode
    av,ar,az=a['v'],a['r'],a['z']
    b=np.zeros_like(av);br=np.zeros_like(av);bz=np.zeros_like(av)
    b[:,0]=ik*av[:,2]/rr-az[:,1]
    b[:,1]=az[:,0]-ar[:,2]
    b[:,2]=ar[:,1]+av[:,1]/rr-ik*av[:,0]/rr
    br[:,0]=ik*(ar[:,2]/rr-av[:,2]/rr**2)-a['rz'][:,1]
    bz[:,0]=ik*az[:,2]/rr-a['zz'][:,1]
    br[:,1]=a['rz'][:,0]-a['rr'][:,2]
    bz[:,1]=a['zz'][:,0]-a['rz'][:,2]
    br[:,2]=a['rr'][:,1]+ar[:,1]/rr-av[:,1]/rr**2-ik*(ar[:,0]/rr-av[:,0]/rr**2)
    bz[:,2]=a['rz'][:,1]+az[:,1]/rr-ik*az[:,0]/rr
    current=np.zeros_like(b)
    current[:,0]=(ik*b[:,2]/rr-bz[:,1])/MU0
    current[:,1]=(bz[:,0]-br[:,2])/MU0
    current[:,2]=(br[:,1]+b[:,1]/rr-ik*b[:,0]/rr)/MU0
    div_b=br[:,0]+b[:,0]/rr+ik*b[:,1]/rr+bz[:,2]
    div_v=v['r'][:,0]+v['v'][:,0]/rr+ik*v['v'][:,1]/rr+v['z'][:,2]
    return dict(velocity=v['v'],div_velocity=div_v,potential=av,b=b,current=current,div_b=div_b)


def _whiten(fields,weights,rtol):
    sampled=(fields*np.sqrt(weights)[:,None,None]).reshape(-1,fields.shape[-1])
    _,singular,vh=svd(sampled,full_matrices=False,check_finite=True)
    keep=singular>rtol*singular[0]
    return vh.conj().T[:,keep]/singular[keep],singular,keep.sum()


def _cross_bphi(fields,bphi):
    result=np.zeros_like(fields)
    result[:,0]=-fields[:,2]*bphi[:,None]
    result[:,2]=fields[:,0]*bphi[:,None]
    return result


def solve_reference(case,*,radial_points=None,angular_points=None):
    """Solve the cold, constrained weak-wall reference on CPU in SI units."""
    R,Z,w=disk_quadrature(case,radial_points,angular_points)
    fields=basis_fields(case,R,Z)
    count=max(64,8*case.order+16)
    theta=2*np.pi*np.arange(count)/count
    boundary=basis_fields(case,case.R0+case.a*np.cos(theta),case.a*np.sin(theta))
    row=(np.cos(theta)[:,None]*boundary['current'][:,0]
         +np.sin(theta)[:,None]*boundary['current'][:,2])
    scale=np.max(np.abs(row))
    restriction=null_space(row/scale if scale else row,rcond=case.rank_rtol)
    bm=fields['b']@restriction
    tv,sv,nv=_whiten(fields['velocity'],w*case.mass*case.density,case.rank_rtol)
    tb,sb,nb=_whiten(bm,w/MU0,case.rank_rtol)
    potential_transform=restriction@tb
    v=fields['velocity']@tv
    b=fields['b']@potential_transform
    current=fields['current']@potential_transform
    force=_cross_bphi(current,case.B0*case.R0/R)
    L=np.einsum('qia,qib,q->ab',v.conj(),force,w)
    hall=-np.einsum('qia,qib,q->ab',current.conj(),force,w)/(E*case.density)
    generator=np.block([[np.zeros((nv,nv),dtype=complex),L],[-L.conj().T,hall]])
    skew=np.linalg.norm(generator+generator.conj().T)/np.linalg.norm(generator)
    if skew>1e-10:raise ValueError('cold reference weak generator is not skew-Hermitian')
    omega,eigenvectors=eigh(-1j*generator)
    va=case.B0/math.sqrt(MU0*case.mass*case.density)
    finite=np.abs(omega)>1e-7*va/case.a
    fractions=[];candidates=[]
    for k,freq in enumerate(omega):
        b_mode=np.einsum('qib,b->qi',b,eigenvectors[nv:,k])
        magnetic=np.sum(w[:,None]*np.abs(b_mode)**2)/MU0
        parallel=np.sum(w*np.abs(b_mode[:,1])**2)/MU0
        fraction=float(parallel/magnetic) if magnetic>1e-20 else 0.
        fractions.append(fraction)
        if finite[k] and freq>0 and fraction>.5:candidates.append(k)
    if not candidates:raise ValueError('no finite compressional cold-wall candidate')
    chosen=candidates[0];vector=eigenvectors[:,chosen]
    cv=tv@vector[:nv];ca=potential_transform@vector[nv:]
    # A physical amplitude convention is applied later when creating production data.
    theta=2*np.pi*(np.arange(2*count+1)+.317)/(2*count+1)
    wall=basis_fields(case,case.R0+case.a*np.cos(theta),case.a*np.sin(theta))
    jw=np.einsum('qib,b->qi',wall['current'],ca)
    normal=jw[:,0]*np.cos(theta)+jw[:,2]*np.sin(theta)
    jscale=max(float(np.max(np.abs(jw))),np.finfo(float).tiny)
    return dict(case=case,model='cold_hall_constrained_weak_wall',time_convention='exp(+i*omega*t)',
                omega_rad_s=float(omega[chosen]),frequency_hz=float(omega[chosen]/(2*np.pi)),
                parallel_magnetic_fraction=fractions[chosen],
                velocity_coefficients=cv,potential_coefficients=ca,
                all_omega_rad_s=omega,all_parallel_fractions=np.array(fractions),
                zero_mode_count=int(np.sum(~finite)),velocity_rank=int(nv),magnetic_rank=int(nb),
                current_constraint_rank=int(row.shape[1]-restriction.shape[1]),
                skew_relative_defect=float(skew),
                relative_eigen_residual=float(np.linalg.norm(generator@vector-1j*omega[chosen]*vector)/max(abs(omega[chosen]),np.finfo(float).tiny)),
                independent_wall_current_relative_defect=float(np.max(np.abs(normal))/jscale),
                volume_m3_per_radian=float(np.sum(w)),
                rank_threshold=case.rank_rtol)


def evaluate_mode(reference,R,Z):
    """Return independent complex physical eigenfields; time factor exp(+iwt)."""
    case=reference['case'];fields=basis_fields(case,R,Z)
    cv=reference['velocity_coefficients'];ca=reference['potential_coefficients']
    result={key:np.einsum('qib,b->qi',fields[key],cv if key=='velocity' else ca)
            for key in ('velocity','b','current','potential')}
    result['div_velocity']=fields['div_velocity']@cv
    return result
