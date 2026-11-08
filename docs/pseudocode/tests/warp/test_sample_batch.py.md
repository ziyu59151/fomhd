# test_sample_batch.py: behavioral verification

Independent scalar complex dot products for multiple non-Hermitian samples, ordinary transpose, noncontiguous ownership, immutable inputs and shape failures. Actual old/new full-stage GPU comparison is external evidence, not this CPU test.

No test grants full coupled or production-scale acceptance. GPU tests require
an explicitly owned Slurm allocation and retain their skipped status on CPU.
