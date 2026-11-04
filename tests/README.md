# A0 test map

The contract catalog is `docs/pseudocode/tests/README.md`. Its stable IDs name
test responsibilities, not a single `contracts.cpp` executable. CMake registers
the actual tests individually. Use `ctest --test-dir build/cpu-release -N` to
list the current executable commands.

| Sources | Contract responsibilities |
|---|---|
| `foundation.cpp`, `foundation_review.cpp` | GEO, OP, FFT, LA; topology, adjoints, all-row constraints, memory limits |
| `fluid_fields.cpp`, `physics.cpp` | FLUID, WALL, FIELD; force and work pairing, Darwin, CT, domain checks |
| `diagnostics.cpp` | DIAG; physical regions, norms, unavailable data, angular quadrature |
| `integration.cpp`, `transaction.cpp` | TIME; accepted/rejected stages, evidence retention, commit safety, source identity |
| `npz.cpp`, `npz_probe.cpp`, `numpy_cross_roundtrip.py` | IO; typed NumPy interoperability and parser rejection |
| `runtime.cpp`, `cli_e2e.py`, `runtime_failures.py` | IO/TIME; publication, restart, signals, post-commit output failure |
| `verification.cpp` | Independent gradient accuracy, unforced time refinement, pressure MMS, local CGL compression |
| `coupled_verification.cpp`, `source_controls.cpp` | Coupled temporal MMS using production midpoint and separate prescribed-source accounting |
| `linear_refinement.cpp`, `linear_refinement_fixture.hpp` | LA; same-factor residual correction, independent original-equation error, zero/singular/nonfinite rejection, stepped disk |
| `cuda.cpp` | GPU; CPU/CUDA numerical and stage comparisons, memory limits |
| `spatial_verification.cpp` | OP; independent staggered G/C/DB/DJ convergence, irregular complex identities, workspace bounds |
| `dispersion.cpp`, `periodic_fixture.hpp` | Local theoretical response; independent continuum matrix, admissible eigenmodes, invariants, CGL stable/unstable controls; test-only periodic adapter |
| `test_analyze.py` | AUX; verified run inputs, masked staggered norms, refinement, complex frequency/growth fits and negative controls |

New verification targets are added here when their staged implementations are
integrated. Passing one executable does not establish the entire model's
scientific validity. Analytic solutions, negative controls, spatial regions,
time windows, tolerances, and limitations belong to each test's specification.

CPU tests run without a GPU. CUDA execution and Compute Sanitizer require the
user's Slurm allocation. No unavailable GPU check is counted as a passing test.
Preserve raw command output and build/source identities in the dated evidence
directory; the current V&V record identifies the exact tested scope.
See [A0 verification](../docs/A0_VERIFICATION.md) for integrated results and
the distinction between local theory, production geometry and experiments.
