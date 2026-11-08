# Complete experiment runner regression mirror

Validate H=M*h arithmetic and independent A0 fine-step control, resolved runtime
caps and new-output ownership. Inspect real accepted checkpoints and retained
failure inputs. Mock only the allocator API to verify device-wide and pool
samples independently, including unavailable pinned-memory aggregates. No test
manufactures a physical acceptance or an experimental speedup result.
