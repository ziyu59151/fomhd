# test_warp_t2_field.py: behavioral verification

Independent Hall polarization and safe step, distinct manufactured sources, real nonlinear periodic production RHS, new-run output contract, smooth fixed physical profile, full magnetic/wall compatibility and explicit high spatial mode.
Focused fault injection uses fake lightweight producers to fail during the
independent oracle and during timed execution. The reports must retain the
injected failure code/message/traceback/nodes, phase-specific attempted wrapper
and producer counts, and nonzero host peak RSS. The linear success test remains.

No test grants full coupled or production-scale acceptance. GPU tests require
an explicitly owned Slurm allocation and retain their skipped status on CPU.
