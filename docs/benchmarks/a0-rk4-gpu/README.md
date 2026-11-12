# A0 synchronous RK4 GPU performance

This benchmark uses the [repaired aligned conducting wall](../rectangular-wall/README.md) and the full massless quasineutral Hall/CGL-B A0 synchronous RK4 path. It measures ordinary GPU implementation changes. All cases retain Fourier modes `-1, 0, +1` with five physical toroidal samples. The input is the same constant-$F$ Solov’ev family sampled on each grid, with a 100 m/s sinusoidal toroidal velocity seed. [Case source and physical parameters](inputs/solovev-g12/source.json)

## Source and replay

The original code is `bc809bdd3aa5253e1ffa60c21e7dcf5e7c84d369`. The optimized source is `803b332ce45750b8c6e759f7405ba76ed87ee213`. The guarded follow-up is `95e5b56f23be59f46bbf6f47206d23b23c34645c`. The measured Python package files can be checked with [source-final.sha256](source-final.sha256) from the root of the optimized checkout. The guarded package uses [source-guarded.sha256](source-guarded.sha256). Both sources use Warp 1.19.0, CuPy 14.2.0, NumPy 2.5.4, Python 3.12, FP64/complex128 and the committed `uv.lock`. The measurements below used Linux, an RTX 5090 with 32,607 MiB reported VRAM, and NVIDIA driver 580.173.02.

Check the bundled inputs from this directory with `sha256sum -c INPUTS.sha256`. Prepare the original and selected optimized or guarded checkouts and install their locked environment with `uv sync --locked`. In a Slurm allocation with one GPU, run the following with absolute source and output paths and the environment’s Python executable. Keep the output directory outside the source tree.

```bash
bash replay.sh /path/to/original /path/to/optimized /path/to/new-output /path/to/.venv/bin/python
python3 summarize.py /path/to/new-output/*.jsonl
/path/to/.venv/bin/python compare.py /path/to/new-output/original-g12-r1 /path/to/new-output/optimized-g12-r1
```

[The replay script](replay.sh) runs two interleaved 12×12 comparisons, then a 24×24 optimized capacity control, using one device and a shared Warp disk cache. Set `FOMHD_REPLAY_REPEATS=1` with the guarded checkout to replay the order used in allocation 11571. Set `WARP_CACHE_PATH` to the shared cache from an earlier run to reproduce the warmed process setup. The 12×12 case uses six 50 ps steps through 300 ps. The 24×24 case uses three 15 ps steps through 45 ps. The archived 6×6 and 20×20 controls use six 50 ps and three 25 ps steps respectively. Every run uses the case’s original checks, tolerances, checkpoint and snapshot policy. Each new process begins with its own CUDA context and memory pool.

[The measurement driver](measure.py) synchronizes around the full advance, including RK4 stages and interval hard checks. Warm step is the median after the first accepted advance. The CLI interval starts just before the run command and includes operator assembly, initial state construction, steps, diagnostics and standard output artifacts. It excludes Python module imports and Slurm queue time. Assembly time includes instrumentation to inventory operator shapes, bytes and nonzeros. Process VRAM is the maximum sampled per PID using `nvidia-smi` at roughly 100 ms intervals. The JSONL retains the separate CuPy pool and process host RSS counters. The process samples may miss transient allocations.

## Cost of the original implementation

A separate 12×12 Nsight Systems and Python profile found 9.02 s cumulative in the boundary ledger, 6.05 s in diagnostics measurement including 4.52 s computing repeated wall distances, 3.37 s in singular values, and 1.54 s in sample-matrix application. Inclusive intervals can overlap. Across the profiled process, the trace includes about 633,000 device-to-host transfers, 630,000 runtime stream synchronizations, and 1.40 million runtime and driver kernel launches. The unprofiled timings below were measured independently. [Python profile](raw/profile/python-profile.txt) · [Nsight Systems aggregates](raw/profile/nsys-summary.log)

The optimized path omits finite-electron-inertia-only operators when the case selects massless electrons, directly builds the local `QB` transfer in complex128 CSR, and batches independent sample products, wall work and modal density constraint solves. It caches static wall-distance and trace data for diagnostics. The existing paired curl and constrained-transport relations are retained. The local 12×12 `QB` allocation changes from 3,324,672 bytes with 2,832 nonzeros to 58,372 bytes. Dense composed curl, divergence and other transfers still limit capacity. For example, the two original 12×12 tracked modal curl matrices each take 10.53 MB. [Recorded operator inventories and preliminary comparisons](raw/profile/summary.json)

## Same GPU timing and memory

| Run | Allocation | Accepted steps | Warm step (s) | CLI (s) | Sampled VRAM (MiB) | Tracked operator arrays (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original 12×12, repetition 1 | 11557 | 6 | 2.882 | 20.044 | 742 | 66.241 |
| Optimized 12×12, repetition 1 | 11557 | 6 | 0.695 | 6.075 | 682 | 43.065 |
| Original 12×12, repetition 2 | 11557 | 6 | 2.614 | 17.672 | 738 | 66.241 |

The first pair measures 4.15× for the warm advance and 3.30× for CLI wall time. The original process in this pair compiled Warp kernels. Using the cached original repetition 2 as the comparator gives 3.76× and 2.91× against the cached optimized repetition 1. Repetition 2 of the optimized code exited before assembly after Warp reported CUDA error 100 and no visible CUDA device. The original repetition 2 VRAM value is based on six samples. [Metrics and failure log](raw/11557/summary.json) · [Slurm log](raw/11557/slurm-11557.log)

| Isolated comparison or capacity run | Allocation | Accepted steps | Warm step (s) | CLI (s) | Sampled VRAM (MiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| Original 6×6 | 11541 | 6 | 1.518 | 17.547 | 602 |
| Original 12×12 | 11546 | 6 | 2.733 | 18.402 | 742 |
| Omit unused massless-run arrays, 12×12 | 11546 | 6 | 2.727 | 18.236 | 686 |
| Same omission, 12×12 | 11549 | 6 | 2.733 | 18.308 | 686 |
| Add sparse transfer and batch sample applications, 12×12 | 11549 | 6 | 2.740 | 18.398 | 682 |
| Omit unused massless-run arrays, 20×20 | 11546 | 3 | 5.449 | 19.043 | 1,388 |
| Add sparse transfer and batch sample applications, 24×24 | 11549 | 3 | 7.415 | 29.713 | 2,100 |

Allocation 11546 rejected the original 20×20 case at a checked preflight estimate of 15,385,894,016 bytes against the case’s 12 GiB device cap. This is a software capacity check. The intermediate 24×24 path accepted 2,052 per-step hard checks. These controlled comparisons resolve a capacity improvement from dropping unused matrices and directly constructing the sparse local transfer. The isolated 12×12 timing variation for these two changes is below 1%. The full speed ratio also includes the batched constraint and wall ledger work and diagnostic reuse. [Earlier timing records and endpoint comparisons](raw/profile/summary.json) · [Preflight rejection](raw/profile/slurm-11546.log) · [Accounting](raw/profile/slurm-accounting.psv)

## Guarded replay on the same GPU

Allocation 11571 replayed the original and guarded 12×12 cases sequentially on the same RTX 5090, with both using the warm Warp disk cache. It then ran the guarded 24×24 capacity case.

| Run | Accepted steps | Warm step (s) | CLI (s) | Sampled VRAM (MiB) | Assembly with inventory (s) | Tracked operator arrays (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original 12×12 | 6 | 2.862 | 19.126 | 742 | 0.532 | 66.241 |
| Guarded optimized 12×12 | 6 | 0.692 | 5.996 | 682 | 0.427 | 43.065 |
| Guarded optimized 24×24 | 3 | 1.085 | 6.629 | 2,100 | 0.910 | 637.633 |

The same-GPU warm advance and CLI speed ratios on 12×12 are 4.13× and 3.19×. The 24×24 run retained all three toroidal Fourier coefficients and accepted all 2,052 per-step hard checks through 45 ps. This bounds capacity and short-time throughput in the checked case. The dense composed spatial operators still dominate tracked allocation at 24×24. [Recovery timing and operator data](raw/11571/summary.json) · [Run log](raw/11571/slurm-11571.log)

At 24×24, each of the three retained mode matrices for Faraday curl has
shape (1825, 1776). Its weighted adjoint has shape (1776, 1825). The
two groups occupy 311.16 MB, and the three magnetic divergence matrices
occupy another 54.75 MB. The next capacity measurement should include
the composed field actions and the diagnostics and wall caches that
consume them. An equivalent factored action can reuse the low-order
incidence stencil and the existing one-direction filters and solves.
Its validation needs the weighted adjoint, constrained transport,
wall extraction, and the error envelopes that use absolute matrix
entries. The timed implementation still keeps these composed matrices.
[Measured matrix shapes and bytes](raw/11571/guarded-g24-r1.jsonl)

The original and guarded 12×12 runs match identities and PASS status for all 4,215 saved hard checks. The guarded and earlier optimized 12×12 GPU endpoints match bitwise in all ten stored physical arrays and derived E. The guarded CPU run also replays its accepted endpoint exactly from restart. [Original versus guarded](raw/11571/endpoint-comparison.json) · [Optimized versus guarded](raw/11571/guard-followup-comparison.json) · [CPU restart comparison](raw/11571/cpu-guarded-restart-parity.json) · [Guarded CPU suite](raw/11571/cpu-guarded.log)

## Numerical scope

Each successful 12×12 run in allocation 11557 accepted all 4,104 per-step hard checks at their original budgets. The full timing comparison uses byte-identical case, geometry and initial arrays, the same integrator, and the same physical endpoint. The [field and hard-check comparator](compare.py) found matching identities and PASS statuses on all 4,215 saved hard rows, including initialization and endpoint rows. The relative difference in computed floating thresholds was at most \(2.45\times10^{-16}\) with the same input budgets. The maximum component difference was \(6.93\times10^{-22}\) T in stored magnetic field and \(3.16\times10^{-12}\) V/m in derived electric field. Across all saved checks, the maximum CT-divergence, wall-current and total-energy check residuals used at most \(3.71\times10^{-13}\), \(4.27\times10^{-6}\) and \(1.09\times10^{-5}\) times their respective positive thresholds. [Endpoint and check evidence](raw/11557/endpoint-comparison.json)

The [CPU test log](raw/11557/cpu-full.log) reports 705 passed and 9 skipped, with three subtests passed. The [GPU test log](raw/11557/gpu-tests-11564.log) reports 12 passed. These checks and the timed steps establish a bounded same-discretization comparison. Frequency, mode shape and observation-window convergence require their own evolution tests. This compact replay publishes input files and measurements. The CLI creates complete state, hard-check and restart artifacts under the output path.
