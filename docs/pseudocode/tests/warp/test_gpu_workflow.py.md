# Test gpu workflow

Source: `tests/warp/test_gpu_workflow.py`.

## A0-PY-TEST-TEST-GPU-WORKFLOW-01 - Procedure and assertions

```text
For ordinary/stress native exports and explicit CPU/CUDA backends:
  require an owned Slurm job before CUDA; otherwise skip only CUDA cases
  rebuild the fixture and retain an explicit host observation function
  explicitly request midpoint because the archived native endpoint is midpoint
  forbid Backend.to_host inside production initialize/advance
  compare initial EL/ET/E and both species' RHS to independent native arrays
  advance one real midpoint interval and require exactly one committed step
  require every hard diagnostic available and passing
  compare all accepted fluid arrays and B using dimensional tolerances
  require returned state arrays belong to the selected backend
This tests the same physical input; it does not measure kernel speedup.
```

Covered top-level functions and fixture classes:

- `test_gpu_stage_and_midpoint_against_native`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.
