# test_time_field.py: behavioral verification

Original nonlinear window residual versus independent sequential RK4; truncated propagation powers; distinct ion stage values; invalid endpoint, stale/failed trials, no silent numerical-library retry, allocation preflight, read-only inputs and owned-Slurm full-waveform parity.

No test grants full coupled or production-scale acceptance. GPU tests require
an explicitly owned Slurm allocation and retain their skipped status on CPU.
