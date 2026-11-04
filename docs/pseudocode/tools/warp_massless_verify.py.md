# `warp_massless_verify.py` pseudocode

## A0-PY-MASSLESS-VERIFY-CLI-01 - Procedure
1. Parse backend (`cpu` or `cuda:<index>`), output directory, Hall amplitudes,
   step counts, time cycles, and spatial grid sizes.
2. Refuse `cuda:*` unless `SLURM_JOB_ID` is set; also report missing CUDA
   dependencies as unavailable rather than falling back to CPU.
3. Require a new output directory; refuse an existing directory or any output
   outside the explicitly supplied new path.
4. Run the actual periodic source stage for each Hall `k*di`.
5. Before opening NPZ output, require explicit host conversion and reject any
   object dtype, including structured arrays containing objects. Save raw
   per-case arrays and diagnostics as NPZ files plus a machine-readable
   JSON report. Accept only host arrays/scalars already captured through an
   explicit ownership conversion; reject other runtime-owned arrays. Include
   NaN/Inf failure arrays verbatim when present. Preserve available
   `Failure.partial_arrays`, stage tags, and numeric checks in failure artifacts,
   and keep numeric thresholds/references/norms in diagnostic artifacts.
   Complete check serialization must retain R5 fields such as
   `compatibility_bound` and `arithmetic_bound`, including their watt units.
6. Run midpoint and RK4 at requested step levels for equal final times and
   compare both against the independent integrator polynomial and the continuum
   solution. Report observed continuum order, retain numeric dt reports, and
   fail insufficient improvement.
7. Run bounded closed-rectangle operator controls and the actual variable-density
   `K=1` velocity layout. Report interior/wall/total errors separately and fail
   if a repaired regression reappears.
8. Run the nonzero-field doubled-`E0` hard-gate control.
9. Include source and dependency hashes, scope limits, unavailable GPU results,
   and the harness-only file manifest.
10. Report scientific acceptance separately from execution completion. Scope
    limitations and nonconvergent wall norms must prevent a scientific pass,
    while preserving measured evidence.
11. Exit zero only for completed requested evidence contracts; a failed hard
    gate, invalid fixture, or I/O error exits nonzero. A
   CUDA request without Slurm is an explicit refusal with no CPU fallback.
