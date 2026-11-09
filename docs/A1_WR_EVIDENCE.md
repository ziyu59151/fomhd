# Few-ion-node multirate waveform relaxation

The opt-in `multirate_wr` coupling directly solves three or five ion nodes per
macrostep while advancing the real electron/field equations with RK4
microsteps. Convergence does not reconstruct the complete fine ion RHS path.
Synchronous RK4 remains the default; T0, T1, T2-E and C1/C2 remain available
with their original evidence and failures. D072 makes the latter research
controls rather than prerequisites for this method.

## Discrete problem and ownership

One macrostep has length H, M field steps h=H/M, and s ion nodes. The independent
ion variables are density, momentum and isotropic pressure. Their polynomial
is defined directly by s=3 or s=5 Gauss-Lobatto nodes. The existing ion RHS is
split exactly into coarse transport/pressure A and fine grid response C:

```text
A = (0, projected convective/pressure force, pressure transport)
C = (-divergence of the common ion flux, projected electric/Lorentz force, 0)
P(theta) = sum_l ell_l(theta) P_l
R_i = P_i - P_0 - H sum_l Q_il A(P_l) - I_i[C(P,Z_h)]
```

The same-density wall projection is shared. The fine density integral uses
the field's actual stage flux; this matters for the quartic s=5 waveform.
`I_i` uses all four real RK4 stage responses, with the declared cubic dense
extension for a coarse node inside a microstep. This is a new mixed-quadrature
discretization, not a sparse sample of the rapidly oscillating force and not
the full fine-stage T1 fixed point. Its general order and stability are not
inferred from the names RK4 or Lobatto.

Each sweep reads an immutable old ion-node version. Acceptance uses that
version and the field history that consumed it, after checking the unrelaxed
all-node residual. The two middle RK4 stages retain separate identities.
Endpoint constraints and caches are rebuilt before a single interval commit.
Coarse ion transport and fine electromagnetic/electron/field work use their
declared weights in one immutable mixed ledger. Trials, rejected sweeps and
restarts never duplicate cumulative physical work. The full equations, units,
layouts, backend placement and failure contracts precede source in
`docs/pseudocode/src/fomhd/time_multirate_wr.py.md`,
`multirate_wr_physics.py.md`, `multirate_wr_checks.py.md` and
`time_multirate.py.md`.

## Actual verification scope

The 2026-11-09 campaign is retained outside the source tree at
`/home/lxzeng/runs/fomhd/a050-multirate-wr__20261109T042844Z`.
Every numerical run has an immutable source snapshot and per-file hashes.
These are diagnostic campaign snapshots of the reviewed implementation;
committing that implementation does not relabel them clean-commit runs.

- The complete CPU suite passed 1223 tests, with 26 conditional skips and
  three subtests, with warnings treated as errors and independent native
  reference arrays available. The log is `LOGS/parent-full-cpu-final.log`.
- Owned GPU job 11255 passed all 24 selected integration regressions. They
  cover complete waveforms and both middle stages, accepted endpoints and
  caches, cumulative ledgers, rollback, nonconvergence, official CLI output,
  strict restart and mismatched-scheme rejection. The log is
  `LOGS/gpu-io-cost-11255.log`. A warning from an intentional singular-system
  negative test is retained.
- Complete two-nominal-period periodic CPU references include synchronous RK4, T1,
  s3/M4, s3/M8, s5/M8 and fixed-H s3 with h/2. Complete GPU synchronous and
  s3/M8 runs use the same physical observation window. The initial condition
  is finite-amplitude and nonuniform, with a small grid-high seed; it is not
  a pure eigenmode. Target mode 1 error is measured independently of mode 16.
- The real nonuniform EAST wall/K=1 calculation uses matched background and
  perturbed ensembles, the original mapped equilibrium, and the original
  observed/hard-check policy. The CPU s3/M4 run reaches 176 field steps
  (360.166225 ns). Its nonzero equilibrium residuals and observed energy,
  power and momentum exceedances remain explicit. Successful transactions
  do not establish exact equilibrium or experimental validation.
  The default-rtol GPU M2 run later stops at macro 81 on ion continuity,
  retaining 80 common accepted endpoints. This is not a completed GPU RZ
  two-period qualification. The state-density residual divided by H explains
  the rate-gate rejection; a separately identified tighter-tolerance case is
  required. Old RZ eigenmode seeds differ from this smooth velocity impulse,
  so matched smooth-seed A0/h2 references are used instead of subtracting
  mismatched historical signals.
- New WR spatial capacity reaches 16 enclosing nodes per R/Z axis, 200 active
  cells, K=1 and Nphi=5 over 80 ps, with an explicitly tightened iteration
  tolerance. Larger preflight refusals are memory-plan results, not measured
  OOMs. Historical A0 40-node and T1 32-node capacity belongs to those methods.

The target-mode relative vector error after two nominal periods is approximately
9.71e-10 for s3/M8 against same-h synchronous RK4; this number is a selected
projection error, not a full-state error or a general accuracy guarantee.
M16/s3 and M32/s3,s5 at the base h converge in their first window but fail
original raw energy checks. Fixed-H h/2 and s5 controls distinguish field
quadrature and coarse-waveform contributions. These first-window failures do
not determine a global stability limit. M8 is the largest base-h macro ratio
with a completed two-period acceptance in this campaign.

## Work and measured cost

For the completed periodic fixed-H comparison, halving h changes M from 8 to
16 while retaining three nodes and three outer sweeps. The final observed
macro has 10 coarse A calls including startup in both runs; fine G/C callbacks
increase from 97 to 193. Complete stage evaluations remain 12. Alias probes,
constraint reconstruction and endpoint diagnostics are counted separately.
The grid response still scales with M; this is not a demonstrated reduction
of future particle pushes, gathers, weight updates or deposition traversals.

On the same RTX 5090, one real warmup and three measured complete advances
over the same 8h physical horizon give these median seconds:

| Method | Complete advance, seconds |
| --- | ---: |
| Synchronous RK4 | 3.11348 |
| T1 complete waveform | 8.39766 |
| Few-node WR, s3/M8 | 4.63324 |
| Few-node WR, s5/M8 | 4.96595 |

The s3 method is 1.81 times faster than T1 in this short measured scope and
1.49 times slower than synchronous RK4. This is not a production speedup or a
field time-parallel result. Raw timing includes actual interval checks and
synchronization; setup, reset, warmup and memory are reported separately in
`analysis/matched-gpu-cost-8h.json`.

`analysis/COUNTER_ROLE_ERRATUM.md` corrects report-only labels without changing
that original artifact. Raw `full_ion_calls` counts the ion transport producer;
it does not count complete RHS evaluations. Do not describe 144 to 24 producer
calls as a complete-RHS reduction. Stage evaluator calls are 40 to 12 in the
same measured A0/WR comparison; these overlapping counters are not added.

## Remaining boundaries

The field scan is still causal. Polynomial batching and GPU execution do not
shorten its time dependency. There is no new temporal-prefix speedup, FK
production implementation, multi-GPU or 1024-squared qualification here.
The conservative memory plan includes retained records for all allowed outer
sweeps; it can refuse a case well below physical device capacity. Live-history
retention and dense spatial/constraint operators need separate, verified work.

All spatial results require direct active-cell RZ maps, wall masks, physical
units, matched observation times and perturbation differences. The English
figures reuse the A0 renderer. Historical maps are rebuilt by
`tools/plots/build_050_sections.py`; current WR maps and cost/accuracy panels
are built by `tools/plots/build_wr_gallery.py` from explicit manifests. Neither
builder may write into the input/source tree or replace existing outputs.

The design authority's `A1_WR_050_REPORT.md` records exact case paths, numerical
comparisons, capacity limits, release/merge scope and the current gallery.
It also distinguishes literature statements, project derivation and measured
results. A3 remains bulk fluid plus CCS-coupled FK fast ions, followed by
scoped stable 1.0 acceptance; A4 later makes all ions kinetic.
