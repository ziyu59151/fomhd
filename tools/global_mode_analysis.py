"""Read-only analysis for axisymmetric real and paired Fourier perturbations."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def physical_mode_coefficients(amplitude, mode):
    """Return retained coefficients with equal physical peak amplitude."""
    a=np.asarray(amplitude,dtype=np.complex128)
    if mode not in (0,1) or not np.isfinite(a).all():
        raise ValueError('finite amplitude and mode 0 or 1 required')
    if mode==0:
        if np.max(np.abs(a.imag),initial=0)>1e-13*max(np.max(np.abs(a),initial=0),1e-300):
            raise ValueError('zero-mode coefficient must be real')
        return a.real[None,:].astype(np.complex128)
    return np.stack((a.conj()/2,np.zeros_like(a),a/2))


def matched_difference(time,signal,control_time,control):
    """Subtract a matched control without hiding missing/cadence differences."""
    t,y,tc,c=map(np.asarray,(time,signal,control_time,control))
    if t.shape!=tc.shape or not np.array_equal(t,tc) or y.shape!=c.shape or y.shape[0]!=t.size:
        raise ValueError('matched control must have identical times and signal shape')
    if not all(np.isfinite(a).all() for a in (t,y,tc,c)):
        raise ValueError('nonfinite matched diagnostic')
    return y-c


def fit_real_mode(time,signal,*,frequency_hint_hz=None):
    """Fit an offset/trend plus damped real oscillation; never unwrap its phase."""
    t=np.asarray(time,dtype=float);y=np.asarray(signal)
    if t.ndim!=1 or y.shape!=t.shape or t.size<9 or np.iscomplexobj(y):
        raise ValueError('real scalar series with at least nine samples required')
    if not np.isfinite(t).all() or not np.isfinite(y).all():
        raise ValueError('finite times and signal required')
    dt=np.diff(t)
    if np.any(dt<=0) or not np.allclose(dt,dt[0],rtol=1e-8,atol=0):
        raise ValueError('uniform increasing cadence required')
    span=t[-1]-t[0];x=(t-t[0])/span
    scale=max(np.ptp(y),np.finfo(float).tiny)
    yn=(y-np.mean(y))/scale
    trend=np.column_stack((np.ones_like(x),x))
    detrended=yn-trend@np.linalg.lstsq(trend,yn,rcond=None)[0]
    spectrum=np.abs(np.fft.rfft(detrended*np.hanning(t.size)))
    freq=np.fft.rfftfreq(t.size,dt[0]);spectrum[0]=0
    if frequency_hint_hz is not None:
        if not np.isfinite(frequency_hint_hz) or frequency_hint_hz<=0:
            raise ValueError('positive frequency hint required')
        spectrum[(freq<.5*frequency_hint_hz)|(freq>1.5*frequency_hint_hz)]=0
    if np.max(spectrum)<=1e-12 or np.ptp(y)<=1e-13*max(np.max(np.abs(y)),1e-300):
        return dict(available=False,reason='no resolved oscillatory signal')
    peak=int(np.argmax(spectrum));cycles=freq[peak]*span
    lo=max(.25,cycles-1.5);hi=min(.499*(t.size-1),cycles+1.5)
    def linear(p):
        angle=2*np.pi*p[0]*x;env=np.exp(p[1]*(x-.5))
        basis=np.column_stack((np.ones_like(x),x,env*np.cos(angle),env*np.sin(angle)))
        coef=np.linalg.lstsq(basis,yn,rcond=None)[0]
        return basis@coef,coef
    result=least_squares(lambda p:linear(p)[0]-yn,[cycles,0.],
                         bounds=([lo,-10.],[hi,10.]),xtol=1e-12,ftol=1e-12,gtol=1e-12)
    predicted,coef=linear(result.x)
    measured=float(result.x[0]);samples=(t.size-1)/measured
    residual=float(np.linalg.norm(predicted-yn)/max(np.linalg.norm(detrended),1e-300))
    available=bool(result.success and measured>=4-1e-6 and samples>=32 and residual<.5)
    reasons=[]
    if measured<4-1e-6:reasons.append('fewer than four measured cycles')
    if samples<32:reasons.append('fewer than 32 samples per cycle')
    if residual>=.5:reasons.append('single-mode fit residual exceeds 0.5')
    return dict(available=available,reason='; '.join(reasons),
                frequency_hz=measured/span,omega_rad_s=2*np.pi*measured/span,
                growth_rate_per_s=float(result.x[1]/span),measured_cycles=measured,
                samples_per_cycle=samples,relative_residual=residual,
                frequency_hint_hz=frequency_hint_hz,linear_coefficients=coef.tolist(),
                fit_success=bool(result.success))


def fluctuation_energy(B,control_B,velocity,control_velocity,density,context):
    """Positive quadratic magnetic/ion-velocity signals, distinct from delta E."""
    mb=np.asarray(context.compute.to_host(context.op.MB)).real
    m0=np.asarray(context.compute.to_host(context.op.M0)).real
    db=np.asarray(B)-np.asarray(control_B)
    du=np.asarray(velocity)-np.asarray(control_velocity)
    rho=context.p.species[0].mass*np.asarray(density).real.reshape(context.N,-1)
    if not np.isfinite(db).all() or not np.isfinite(du).all() or np.any(rho<=0):
        raise ValueError('finite perturbations and positive control density required')
    magnetic=float(np.sum(np.abs(db.reshape(-1,mb.size))**2*mb)*np.pi/context.p.mu0)
    kinetic=float(np.sum(rho*np.sum(np.abs(du.reshape(context.N,3,-1))**2,axis=1)*m0)*np.pi/context.N)
    return dict(perturbation_magnetic_J=magnetic,perturbation_ion_kinetic_J=kinetic)
