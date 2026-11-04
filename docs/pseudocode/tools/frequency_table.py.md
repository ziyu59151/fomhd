# Indicative SI frequency and target-sampling table

## A0-PY-FREQUENCY-TABLE-01 - SI table construction

1. Use `fomhd.stability.frequency_scales` with positive physical deuteron and
   electron masses, the declared density/magnetic-field cases, and zero pressure
   and wavenumber. These are indicative dimensional scales, not a solver run.
2. Convert angular frequencies to Hz exactly once. Compute the upper-hybrid
   scale as hypot(omega_pe, omega_ce)/(2*pi).
3. For target harmonic h and samples per target period Nt, compute
   target_sampling_Hz=Nt*h*f_ci and dt_target=1/target_sampling_Hz.
   Compare with 32 samples per electron plasma and upper-hybrid period.
   These sampling estimates do not enforce a finite-horizon midpoint error.
4. Write CSV and strict JSON into a new output directory. Preserve input values,
   model/scope descriptions, and hashes/archives of this script and stability.py.
5. Reject nonpositive/nonfinite inputs and existing output directories. Never
   classify a physical run or choose its dt using this table alone.
