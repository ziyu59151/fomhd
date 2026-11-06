# Explicit time-step evidence and policy

The model-specific rate estimate reads an absent `electron_inertia` as the
massless default `0`; explicit `1` selects finite plasma/cyclotron saturation.

## A0-PY-STABILITY-01 - Frequencies and horizon budgets

1. Validate finite positive SI masses/densities/epsilon0/mu0 and nonnegative
   magnetic magnitude, pressure, and resolved wavenumber.
2. Compute omega_ps=sqrt(n_s q_s^2/(epsilon0 m_s)), Omega_s=|q_s| B/m_s,
   d_e=sqrt(m_e/(mu0 n_e e^2)), and Hall branches
   omega_W0=B*k^2/(mu0 e n_e), omega_Wm=omega_W0/(1+k^2*d_e^2).
   These last expressions are parallel, cold Hall asymptotes, not full spectra.
   Report rad/s and Hz separately; target harmonic sampling adds its own budget.
3. For a declared oscillator frequency omega and step h, form x=omega*h.
   R_mid=1+ix-x^2/2; R_RK4=R_mid-ix^3/6+x^4/24.
   Midpoint log|R|=.5*log1p(x^4/4); RK4 log|R| is evaluated robustly.
4. Over exactly nsteps, report nsteps*log|R| and
   abs(nsteps*(arg(R)-x)); overflow is explicit infinity, not a passing result.
5. Accept only if samples/cycle>=32, amplitude error<=.01 and phase error<=.1,
   using configurable positive budgets. Midpoint has NO nonzero imaginary-axis
   absolute-stability interval. RK4 additionally requires x<=2 sqrt(2).
6. A scalar estimate cannot prove nonlinear stability. Runtime diagnostics still
   enforce positivity, finite values, all physical constraints, and solve rank.
   Warn override records the failed guard; it never bypasses those checks.

## A0-PY-STABILITY-02 - Runtime estimate

1. Reduce the current padded physical stage on its owning backend. Transfer only
   density extrema, pressure maxima, magnetic/velocity maxima and operator norm
   scalars. Reject invalid or nonpositive densities before any reciprocal.
2. Let kR=||DR||inf+2/Rmin, kZ=||DZ||inf, kphi=K/Rmin and
   k=sqrt(kR^2+kZ^2+kphi^2). This is an operator-based resolution estimate, not a
   rigorous bound on the complete nonnormal CGL/Darwin Jacobian.
3. Use maximum densities for plasma frequencies and minimum densities for
   pressure, Hall and Alfven estimates. Finite inertia uses
   sqrt(omega_plasma^2+Omega_e^2+Omega_i^2) + k*(c_i+c_e+u_max).
   This includes an upper-hybrid scale; plasma frequency alone need not dominate.
4. Massless mode uses Hall(k)+Omega_i+k*(v_A+c_bulk+u_max), where
   c_bulk^2=(gamma_i*p_i+max(3*p_parallel,2*p_perp))/(m_i*n_i).
   Plasma/electron-cyclotron frequencies remain metadata, not active limits.
5. Assess the configured integrator over the remaining requested horizon. The
   caller records the full report; a failing error policy rejects before state
   mutation and a warn policy records the failed estimate without disabling
   physical, positivity, rank, or transaction gates. Recheck every base stage.
