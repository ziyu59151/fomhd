"""Explicit oscillator accuracy budgets and SI plasma frequency estimates.

These estimates do not replace a resolved spectrum or nonlinear diagnostics.
Cold parallel whistler asymptotes are labeled explicitly in returned metadata.
"""
from __future__ import annotations

import math

E_CHARGE = 1.602176634e-19
E_MASS = 9.1093837139e-31
P_MASS = 1.67262192595e-27
EPSILON0 = 8.8541878188e-12
MU0 = 1.25663706127e-6


def amplification(method: str, z: complex) -> complex:
    """Return the actual fully explicit method's stability polynomial."""
    if method == 'midpoint':
        return 1 + z + z*z/2
    if method == 'rk4':
        return 1 + z + z*z/2 + z*z*z/6 + z*z*z*z/24
    raise ValueError('integrator must be midpoint or rk4')


def temporal_metrics(method: str, omega: float, dt: float, nsteps: int) -> dict:
    """Measure scalar imaginary-mode amplification over the requested horizon."""
    if method not in ('midpoint', 'rk4'):
        raise ValueError('integrator must be midpoint or rk4')
    if not math.isfinite(omega) or omega < 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError('omega must be nonnegative and dt positive finite')
    if isinstance(nsteps, bool) or not isinstance(nsteps, int) or nsteps < 0:
        raise ValueError('nsteps must be a nonnegative integer')
    x = omega * dt
    if not math.isfinite(x) or x > 1e50:
        return dict(omega_dt=x, horizon_s=dt*nsteps, samples_per_cycle=0.,
                    log_amplitude_error=math.inf, amplitude_relative_error=math.inf,
                    phase_error_rad=math.inf, absolute_imaginary_stable=False)
    R = amplification(method, 1j*x)
    if method == 'midpoint':
        log_abs = .5*math.log1p(x**4/4)
    elif x < .5:
        log_abs = .5*math.log1p(-x**6/72+x**8/576)
    else:
        log_abs = math.log(abs(R))
    accumulated = nsteps*log_abs
    amplitude_error = abs(math.expm1(accumulated)) if accumulated < 709 else math.inf
    return dict(omega_dt=x, horizon_s=dt*nsteps,
                samples_per_cycle=2*math.pi/x if x else math.inf,
                log_amplitude_error=accumulated,
                amplitude_relative_error=amplitude_error,
                phase_error_rad=abs(nsteps*(math.atan2(R.imag, R.real)-x)),
                absolute_imaginary_stable=(x == 0 if method == 'midpoint' else x <= 2*math.sqrt(2)))


def assess_dt(method: str, omega: float, dt: float, nsteps: int, *,
              minimum_samples: float = 32., amplitude_budget: float = .01,
              phase_budget: float = .1) -> dict:
    """Assess finite-horizon accuracy; midpoint is never labeled absolutely stable."""
    budgets = (minimum_samples, amplitude_budget, phase_budget)
    if any(not math.isfinite(v) or v <= 0 for v in budgets):
        raise ValueError('time-step budgets must be positive finite')
    m = temporal_metrics(method, omega, dt, nsteps)
    reasons=[]
    if m['samples_per_cycle'] < minimum_samples:
        reasons.append('insufficient samples per fastest estimated cycle')
    if m['amplitude_relative_error'] > amplitude_budget:
        reasons.append('finite-horizon amplitude error exceeds budget')
    if m['phase_error_rad'] > phase_budget:
        reasons.append('finite-horizon phase error exceeds budget')
    if method == 'rk4' and not m['absolute_imaginary_stable']:
        reasons.append('outside RK4 imaginary-axis stability interval')
    return dict(m, accepted=not reasons, reasons=reasons, integrator=method,
                minimum_samples=minimum_samples, amplitude_budget=amplitude_budget,
                phase_budget_rad=phase_budget,
                scope='imaginary scalar-mode estimate; not a nonlinear stability proof')


def frequency_scales(*, ne: float = 1e19, ni: float = 1e19, B: float = 3.,
                     mi: float = P_MASS, me: float = E_MASS,
                     qi: float = E_CHARGE, qe: float = -E_CHARGE,
                     k: float = 0., pi: float = 0., pperp: float = 0.,
                     pparallel: float = 0., gamma_i: float = 5/3,
                     epsilon0: float = EPSILON0, mu0: float = MU0) -> dict:
    """Compute frequencies with physical me>0 even for massless model metadata."""
    positive=(ne, ni, mi, me, epsilon0, mu0)
    nonnegative=(B, k, pi, pperp, pparallel)
    if any(not math.isfinite(v) or v <= 0 for v in positive):
        raise ValueError('densities, masses, and electromagnetic constants must be positive finite')
    if any(not math.isfinite(v) or v < 0 for v in nonnegative):
        raise ValueError('magnetic magnitude, k, and pressures must be nonnegative finite')
    if not math.isfinite(qi) or not math.isfinite(qe) or not qi > 0 or not qe < 0 or not gamma_i > 1:
        raise ValueError('require positive ion charge, negative electron charge, gamma_i>1')
    wpe=math.sqrt(ne*qe*qe/(epsilon0*me));wpi=math.sqrt(ni*qi*qi/(epsilon0*mi))
    wce=abs(qe)*B/me;wci=abs(qi)*B/mi
    de=math.sqrt(me/(mu0*ne*qe*qe))
    hall=B*k*k/(mu0*abs(qe)*ne)
    electron_sound=math.sqrt(max(3*pparallel,2*pperp)/(me*ne))
    ion_sound=math.sqrt(gamma_i*pi/(mi*ni))
    return dict(omega_pe=wpe, omega_pi=wpi, omega_plasma=math.hypot(wpe,wpi),
                omega_ce=wce, omega_ci=wci, fpe_Hz=wpe/(2*math.pi),
                fpi_Hz=wpi/(2*math.pi), fce_Hz=wce/(2*math.pi),
                fci_Hz=wci/(2*math.pi), electron_skin_depth_m=de,
                omega_whistler_massless=hall,omega_whistler_finite=hall/(1+(k*de)**2),
                electron_pressure_rate=k*electron_sound,ion_pressure_rate=k*ion_sound,
                alfven_rate=k*B/math.sqrt(mu0*mi*ni),
                whistler_scope='cold parallel Hall asymptote; not a complete CGL/Darwin spectrum')


def stage_frequency_estimate(stage, context) -> dict:
    """Reduce physical stage scales without copying state arrays to the host.

    The rate is a resolution-based estimate, not a rigorous Jacobian bound.
    Density extrema intentionally differ for plasma and transport estimates.
    """
    p, xp, backend = context.p, context.compute.xp, context.compute
    nc = context.op.Nc

    def scalar(value):
        return float(backend.scalar(value))

    n_min = [scalar(xp.min(xp.real(f.n))) for f in stage.fluid]
    n_max = [scalar(xp.max(xp.real(f.n))) for f in stage.fluid]
    pi = scalar(xp.max(xp.real(stage.fluid[0].perp)))
    pe = scalar(xp.max(xp.real(stage.fluid[1].perp)))
    pp = scalar(xp.max(xp.real(stage.fluid[1].parallel)))
    B = scalar(xp.max(xp.sqrt(xp.sum(xp.abs(
        stage.BN.reshape(context.N, 3, nc)) ** 2, axis=1))))
    u = max(scalar(xp.max(xp.sqrt(xp.sum(xp.abs(
        f.u.reshape(context.N, 3, nc)) ** 2, axis=1)))) for f in stage.fluid)
    r_min = min(context.geometry.cells.R)
    if (any(not math.isfinite(v) or v <= 0 for v in n_min+n_max)
            or not math.isfinite(r_min) or r_min <= 0
            or any(not math.isfinite(v) or v < 0 for v in (pi, pe, pp, B, u))):
        raise ValueError('invalid physical stage for frequency assessment')
    kr = scalar(xp.max(xp.sum(xp.abs(context.op.DR), axis=1))) + 2/r_min
    kz = scalar(xp.max(xp.sum(xp.abs(context.op.DZ), axis=1)))
    k = math.sqrt(kr*kr+kz*kz+(p.K/r_min)**2)
    ion, electron = p.species
    scales = frequency_scales(
        ne=n_max[1], ni=n_max[0], B=B, mi=ion.mass, me=electron.mass,
        qi=ion.charge, qe=electron.charge, k=k, pi=pi, pperp=pe,
        pparallel=pp, gamma_i=ion.gamma, epsilon0=p.epsilon0, mu0=p.mu0)
    hall = B*k*k/(p.mu0*abs(electron.charge)*n_min[1])
    ci = math.sqrt(ion.gamma*pi/(ion.mass*n_min[0]))
    ce = math.sqrt(max(3*pp, 2*pe)/(electron.mass*n_min[1]))
    va = B/math.sqrt(p.mu0*ion.mass*n_min[0])
    cb = math.sqrt((ion.gamma*pi+max(3*pp, 2*pe))/(ion.mass*n_min[0]))
    inertia = int(getattr(p, "electron_inertia", 0))
    if inertia == 1:
        omega = math.sqrt(scales['omega_plasma']**2+scales['omega_ce']**2
                          + scales['omega_ci']**2) + k*(ci+ce+u)
        active = ['plasma_upper_hybrid', 'ion_cyclotron', 'species_pressure', 'advection']
    elif inertia == 0:
        omega = hall + scales['omega_ci'] + k*(va+cb+u)
        active = ['hall', 'ion_cyclotron', 'alfven', 'bulk_pressure', 'advection']
    else:
        raise ValueError('electron_inertia must be 0 or 1')
    return dict(scales, omega_estimate=omega, electron_inertia=inertia,
                active_limits=active, k_estimate_per_m=k, density_min=n_min,
                density_max=n_max, magnetic_max_T=B, speed_max_m_per_s=u,
                omega_whistler_massless=hall, ion_pressure_rate=k*ci,
                electron_pressure_rate=k*ce, alfven_rate=k*va,
                bulk_pressure_rate=k*cb, advection_rate=k*u,
                scope='operator-resolution estimate; not a rigorous full-spectrum or nonlinear bound')


def assess_runtime(estimate: dict, context, *, step: int) -> dict:
    """Return a record; the time transaction owns enforcement and logging."""
    p = context.p
    if p.dt_guard not in ('error', 'warn'):
        raise ValueError('dt_guard must be error or warn')
    if step < 0 or step > p.nsteps:
        raise ValueError('invalid step for remaining-horizon assessment')
    report = assess_dt(
        p.integrator, estimate['omega_estimate'], p.dt, p.nsteps-step,
        minimum_samples=p.minimum_samples_per_cycle,
        amplitude_budget=p.max_amplitude_error, phase_budget=p.max_phase_error_rad)
    report.update(estimate=estimate, step=step,
                  action='continue' if report['accepted'] else p.dt_guard)
    return report


def assess_predictor_runtime(estimate: dict, context, *, step: int) -> dict:
    """Assess only the explicit RK4 microsteps over the same physical horizon."""
    p = context.p
    micro_steps = getattr(p, "micro_steps", None)
    coupling = getattr(p, "coupling", None)
    if (
        coupling not in ("predictor", "waveform")
        or getattr(p, "electron_inertia", None) != 0
        or getattr(p, "integrator", None) != "rk4"
    ):
        raise ValueError(
            "microstep guard requires coupling=predictor|waveform, "
            "massless electrons, and rk4"
        )
    if (
        isinstance(micro_steps, bool)
        or type(micro_steps) is not int
        or micro_steps <= 0
    ):
        raise ValueError("micro_steps must be a positive non-boolean integer")
    if (
        not math.isfinite(p.dt)
        or p.dt <= 0.0
        or type(step) is not int
        or step < 0
        or step > p.nsteps
    ):
        raise ValueError("invalid predictor step/horizon")
    micro_dt = p.dt / micro_steps
    remaining_micro_steps = (p.nsteps - step) * micro_steps
    report = assess_dt(
        "rk4",
        estimate["omega_estimate"],
        micro_dt,
        remaining_micro_steps,
        minimum_samples=p.minimum_samples_per_cycle,
        amplitude_budget=p.max_amplitude_error,
        phase_budget=p.max_phase_error_rad,
    )
    report.update(
        estimate=estimate,
        step=step,
        action="continue" if report["accepted"] else p.dt_guard,
        coupling=coupling,
        micro_steps=micro_steps,
        micro_dt=micro_dt,
        micro_horizon_steps=remaining_micro_steps,
        macro_remaining_horizon_s=p.dt * (p.nsteps - step),
        scope=(
            "micro-RK4 only; does not prove macro-coupling stability"
        ),
    )
    return report
