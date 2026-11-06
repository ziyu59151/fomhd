# A1 T0 evidence and limits

T0 is implemented and ready for the user's next-tier decision. Default
coupling remains synchronous RK4. T1--T4, full sparse conversion and multi-GPU
execution remain separate work.

Numerical benchmark: `93219d2f85c5185cefaa97fd3e8125cef632f4cb`.
Closed-fixture explicit-upload repair: `581f20d050df50718e028d7550e9e65d57cc2ea7`;
solver modules are unchanged. A0 maintenance: `release/a0` at
`9a38bf41e7bc1fc50186c81967312a7ec5f028e8`.

## Verification and temporal observations

The final delivery passed 770 CPU tests, with 9 conditional skips and 3
subtests. The final setup repair also passed 33 focused verifier tests.
Independent review found no blocking issue. The frozen RTX 5090 run passed
all 12 GPU regression checks. Development-stage A100 checks also passed 12
cases; the frozen temporal scan used RTX 5090 because both A100s were occupied.

The periodic adapter uses 9 points, K=0, k*di=1, amplitude 2e-6, one physical
period, independent complex-exponential and synchronous h/h/2 references.
It is not closed-wall or global physical validation. Hard checks are retained;
empty-boundary and convergence observations are separate. The operator-
frequency guard runs in documented warn mode: h=period/64 is not certified
stable for the full spectrum.

| Micro h | Method | Relative modal error | Signed phase, rad |
|---|---|---:|---:|
| period/64 | synchronous RK4 | 4.864e-6 | -4.847e-6 |
| period/64 | T0 M=1 | 2.813e-3 | +2.785e-3 |
| period/64 | T0 M=4 | 4.515e-2 | +4.368e-2 |
| period/128 | T0 M=1 | 7.034e-4 | +6.971e-4 |
| period/128 | T0 M=8 | 4.515e-2 | +4.368e-2 |

These data support second-order behavior of this coupling in this case.
Fixed-path quadrature has a separate fourth-order control. First-window ion
path error is much larger than corrected endpoint error; refining h at fixed H
leaves the dominant coupling error nearly unchanged.

CPU/GPU endpoint modal coefficients agree within 7.674e-16 relative. A0/A1
synchronous coefficients differ by 2.335e-15 on CPU and 2.479e-15 on GPU.
Closed GPU initial data are explicitly constructed on the host and uploaded;
perturbation, initialization and numerical stages use the requested GPU.

## Failures, costs and scope

- M=8,h=period/64 rejects the first macro on both backends: ion kinetic-energy
  check 2.2046e-8 J exceeds 8e-9 J. No macro or cumulative ledger is accepted.
- The closed nonzero perturbation fails before T0: wall electron flux 44007.1
  on CPU and 33656.6 on GPU versus 20000. A0 reproduces the CPU failure.
  The earlier GPU fixture TypeError is retained separately from its repair.
- Model counts are 6N+N/M versus 7N for synchronous RK4; both are O(N).
  T0 keeps O(M) stage history, including density-dependent constraint factors.
  Failure-record merging still has potentially quadratic Python work.
- For this small case, synchronous h=period/64 takes 20.95s on RTX 5090 versus
  2.33s on CPU; T0 M=4 takes 18.94s versus 2.10s. CPU uses a median of 3 measured
  samples, GPU 1, after independent warmup. Timings include checks, modal
  history and synchronization. There is no equal-accuracy speedup claim.
- Fixed-wall reuse separately removes repeated curl-curl materialization and
  factorization, with dense-reference E/Bdot differences near roundoff.
  Other dense operators and density-dependent solves remain.

The full report is in the design repository:
`mainline_design/implementation/A1/A1_T0_REPORT.md`. Raw products, commands,
source hashes and failures are under
`/home/lxzeng/runs/fomhd/a1-t0-acceptance__20261106T110000Z/`.
Final periodic GPU products are in `gpu-bounded/`; repaired closed evidence
is in `gpu-closed-repaired/`. The interrupted initial GPU scan is incomplete
and excluded. Earlier failures and audits remain in the preparation/worker
records. RSS/HWM and CuPy pool statistics are process-wide observations,
not isolated live peaks.

Use `tools/warp_t0_verify.py --help` with a fresh output directory. A scan
returns nonzero for retained failed cases. GPU execution requires an owned
Slurm allocation. Enter T1 waveform correction only after the user's judgment.
