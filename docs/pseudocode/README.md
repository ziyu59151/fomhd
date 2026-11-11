# A0 实现级伪代码与维护说明书

> Status: Python/Warp source reading guide for the active implementation.
> `reference/a0-native` owns the frozen native implementation and its mirrored
> pseudocode. Acceptance evidence is recorded separately from implementation
> contracts.
> Authority: the design repository owns the mathematical model; this directory
> owns the step-by-step mapping to the active Python/Warp/CuPy source.
> Last synchronized: 2026-11-05.

## File mapping

Remove `docs/pseudocode/` and the final `.md` suffix to obtain the mirrored
source path. For example, `src/fomhd/fields.py.md` maps to
`src/fomhd/fields.py`, and `tests/warp/test_runtime.py.md` maps to
`tests/warp/test_runtime.py`. README and routing narratives are navigation
documents, not fictitious modules. The active entry point is
`src/fomhd/__main__.py`.

| Mainline responsibility | Mirrored pseudocode |
|---|---|
| CLI, strict configuration, run products and restart | [entry](src/fomhd/__main__.py.md), [I/O](src/fomhd/io.py.md) |
| Physical records, layout and state operations | [types](src/fomhd/types.py.md), [core](src/fomhd/core.py.md) |
| Device ownership, Warp kernels and CuPy FFT/LU/SVD | [backend](src/fomhd/backend.py.md) |
| Closed geometry and compatible RZ/Fourier operators | [geometry](src/fomhd/geometry.py.md), [operators](src/fomhd/operators.py.md) |
| Fluid equations, walls and finite-mass Darwin fields | [fluid](src/fomhd/fluid.py.md), [wall](src/fomhd/wall.py.md), [fields](src/fomhd/fields.py.md) |
| Explicit RK4 default, explicit midpoint, stage evaluation and guards | [time](src/fomhd/time.py.md), [stage](src/fomhd/stage.py.md), [stability](src/fomhd/stability.py.md) |
| Diagnostics and source/interval accounting | [diagnostics](src/fomhd/diagnostics.py.md) |
| Coupled manufactured solutions and independent spatial checks | [MMS fixture](tests/warp/mms_fixture.py.md), [MMS CLI](tools/warp_mms.py.md) |
| Quasineutral massless equations, constraints and diagnostics | [massless stage](src/fomhd/massless.py.md), [massless checks](src/fomhd/massless_checks.py.md) |
| Hall response, closed-wall reconstruction and evidence output | [massless fixture](tests/warp/massless_verification_fixture.py.md), [verification CLI](tools/warp_massless_verify.py.md) |
| Python case generation, equal-horizon scans, scales and plots | [benchmark](tools/warp_benchmark.py.md), [frequency table](tools/frequency_table.py.md), [plots](tools/plots/plot_warp_benchmarks.py.md) |
| Documented massless case template | [case config](tools/case_config.py.md) |
| Offline analysis and accepted-run figures | [analyze.py](tools/analyze.py.md), [plot run](tools/plots/plot_run.py.md) |
| User-owned Slurm launcher | [a0.slurm](jobs/a0.slurm.md) |

The native implementation-level pseudocode, CMake contracts, native tests and
native case generator live on `reference/a0-native`. They are not retained here
as broken links or a second source copy.

[TRACEABILITY.tsv](TRACEABILITY.tsv) connects stable `A0-PY-…` step IDs to the
actual `src/fomhd`, test, and tool paths. [SOURCE_BASELINE.json](SOURCE_BASELINE.json)
records design-source hashes and scope without making the design repository a
runtime dependency. Copying this repository retains the pseudocode; numeric
source checks use the recorded authoritative design files when available.

## 记号与执行约定

- `require(predicate, code, context)` returns a `Failure` when false and stops
  the current call. `x = TRY F(...)` propagates the failure without hidden
  retry, clipping, step reduction, or CPU fallback.
- Half-open loops use `[0,N)` and zero-based indices. `adj(A)` is the conjugate
  transpose of the actual discrete matrix. `solve(A,b)` means the checked LU
  interface, never explicit inversion.
- All active state uses FP64/complex128 and SI units. Mode order is `-K..K`;
  scalar layout is `[mode][cell]`; vector layout is
  `[mode][component R/phi/Z][cell]`. Samples replace mode by physical phi
  sample. CuPy owns complex128 buffers, while Warp receives explicit float64
  views with documented order, stream, and lifetime.
- Repeated numerical stages use one selected backend, one NVIDIA device and one
  stream. Device, stream, dtype, or shape mismatch is an error. Whole-state host
  transfers occur only at declared checkpoints, diagnostics, explicit tests, or
  I/O boundaries.
- Geometry and static metadata are host setup. Numerical modules do not perform
  file I/O. Time owns accepted/rejected transactions, diagnostics are
  read-only, and shared physical fluxes/currents have one producer.
- `time.integrator` omitted defaults to `rk4`. Explicit `midpoint` remains a
  supported comparison path. Both use the same transaction, source ownership,
  and ledger contracts; the equations and budgets do not change with the
  selected explicit method.

## 调用和完成边界

```mermaid
flowchart TD
    cli[src/fomhd __main__] --> load[io configuration and exact inputs]
    cli --> geom[geometry and operators]
    cli --> time[time: RK4 default / explicit midpoint transaction]
    time --> stage[stage: one model and stage ownership]
    stage --> fluid[fluid, wall, fields]
    fluid --> backend[Warp kernels and CuPy libraries]
    stage --> diag[diagnostics: read-only records]
    time --> diag
    cli --> runio[io: snapshots, checkpoints, restart, failures]
```

A0 covers the fixed closed staircase domain, strict input loading, the complete
finite-mass workflow, and the optional quasineutral massless branch. A1 changes
time scheduling after its algorithm is selected; A2 and A3 remain separate.
Periodic Cartesian fixtures are tests, not a substitute for the fixed-boundary
production model.

Pseudocode is an executable implementation guide, not proof. MMS, convergence,
conservation, restart, independent-array, and CPU/GPU evidence remain separate.
Fourth-order claims apply only to measured smooth regions; the staircase wall,
global spatial order, scientific validation, and performance remain unclaimed.

[D075 rectangular outer wall](src/fomhd/box_boundary.py.md) explains the
formal geometry entry, fixed magnetic flux, shared stress and work trace,
and restart identity.
