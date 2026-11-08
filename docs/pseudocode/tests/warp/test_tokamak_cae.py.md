# test_tokamak_cae.py

Verify deuterium SI references, massless/RK4/Nphi=5 defaults and prototype observe policy only for power/energy/momentum. Necessary current/quasineutrality checks stay hard. Exercise Fourier embedding including K=1 axisymmetric state and reject imaginary n0.

Create a small real production spectrum and verify background/reference identity
checks reject changed density, changed physical mass, missing Fourier coverage
and changed numerical source identity. Exercise one short actual paired RK4 run:
initial checks and RHS arrays, raw energy partitions and exact matched cadence
must survive; a deliberately excessive dt must stop before advancement even
when the conservative operator estimator is observe.

An explicit short horizon overrides a longer cycles argument in target_periods.

Larger enclosing grids up to40 retain the same memory failure gate; reject
nonpositive caps and grids above the bounded supported range before execution.

The 0.5.0 capacity probe must reach geometry construction for40; intercept that
construction in the CPU unit test so testing the input policy allocates no
large matrices. Grids above40 still reject before construction.
