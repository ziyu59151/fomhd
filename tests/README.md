# A0 test map

The active test suite runs against `src/fomhd`. Pytest collects the offline
analysis, G-EQDSK, and Warp/CuPy tests; native-only executables are preserved
on `reference/a0-native` rather than duplicated here.

| Sources | Contract responsibilities |
|---|---|
| `warp/foundation_*.py` | GEO, OP, FFT, LA; topology, adjoints, all-row constraints, memory limits |
| `warp/test_physics.py` | FLUID, WALL, FIELD; force/work pairing, Darwin, CT and domain checks |
| `warp/test_diagnostic_evidence_io.py` | DIAG and IO; regions, norms, unavailable data and evidence |
| `warp/test_runtime.py`, `warp/test_model_routing.py` | TIME; explicit RK4/midpoint transactions, accepted/rejected evidence, routing and restart |
| `warp/test_massless_pressure_scale.py` | Strict pressure-adjoint operand scaling and corrupted-work negative control |
| `warp/test_time_predictor.py` | A1-T0 fixed-tangent predictor scheduling, micro-RK4 guard, rollback, restart and bounded massless fixtures |
| `warp/test_mms.py`, `warp/test_massless_verification.py` | Independent temporal/spatial manufactured solutions and constrained-model checks |
| `warp/test_stability.py`, `warp/test_longitudinal.py` | Explicit-integrator polynomials, finite-horizon guards and local longitudinal response |
| `warp/test_benchmark_harness.py`, `warp/test_*plots.py` | Python case generation, CLI products and offline figure provenance |
| `warp/test_gpu_*.py` | Device execution, CPU/CUDA parity and failure behavior; require an owned Slurm allocation |
| `test_analyze.py` | AUX; verified run inputs, masks, refinement and complex frequency/growth fits |
| `test_geqdsk.py` | AUX; independent polynomial equilibrium parser checks |

Run the required CPU suite with warnings as errors:

```bash
OPENBLAS_NUM_THREADS=1 \
FOMHD_NATIVE_ORACLE_DIR=/home/lxzeng/projects/fomhd/build/native-oracles \
uv run python -W error -m pytest -q
```

The environment variable selects already exported independent arrays for
oracle-only parity checks. A missing export makes only that comparison
unavailable; required core tests do not depend on a native build.

Passing a focused test does not establish the entire model's scientific
validity. CPU tests do not count as GPU passes. CUDA execution and
Compute Sanitizer require the user's own Slurm allocation, and unavailable GPU
checks are not reported as passing. Preserve raw commands and source identities
in dated run evidence. Historical native test results remain in
`docs/A0_VERIFICATION.md` and on the reference branch.
