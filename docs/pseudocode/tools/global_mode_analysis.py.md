# Global mode analysis

Host-only, read-only SI diagnostic. Never alter a solver state or infer a
zero-mode perturbation by deleting the zero Fourier coefficient.

1. `matched_difference`: require identical finite times and array shapes;
   subtract the unperturbed control at every saved time. Reject partial/gapped
   or mismatched controls rather than interpolating silently.
2. `fit_real_mode`: require uniform times, detrend a real scalar signal with
   constant and linear terms, locate its nonzero real FFT peak, then fit
   `c0+c1*t+exp(gamma*t)*(a*cos(omega*t)+b*sin(omega*t))` by variable projection.
   Scale time to the observation interval. Refine frequency/growth by bounded
   least squares. Return frequency, growth, fit residual, measured cycles and
   samples/cycle; fewer than four cycles or 32 samples/cycle is unavailable.
   A frequency hint only restricts the search interval and is reported.
3. `physical_mode_coefficients`: n=0 stores A once as a real coefficient;
   n=1 stores conjugate coefficients A/2. Reject non-real zero-mode data.
4. `fluctuation_energy`: use the retained Fourier quadrature for magnetic
   differences and sampled ion velocity differences against a matched control,
   using positive control density. These positive quadratic signal measures
   are distinct from the difference of total energies.
