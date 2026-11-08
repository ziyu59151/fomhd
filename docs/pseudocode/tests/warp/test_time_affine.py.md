# test_time_affine.py: behavioral verification

Actual CPU tree and GPU affine tree versus a separately written sequential recurrence; noncommuting blocks, sources, midpoint identity, shape/dtype, ownership and non-power-of-two tails.

No test grants full coupled or production-scale acceptance. GPU tests require
an explicitly owned Slurm allocation and retain their skipped status on CPU.
