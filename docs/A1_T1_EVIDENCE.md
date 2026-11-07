# T1 complete waveform evidence

Status: bounded T1 implementation and controlled report delivered; T2 remains
subject to the user's decision (D065/O31). This is a serial waveform reference,
with existing single-device Warp/CuPy execution, not time-slab or multi-GPU work.
The default coupling remains synchronous RK4; T1 is selected with
`time.coupling="waveform"`.

## Method and independent analysis

All four ion stages of every microstep are independent path entries, including
the two same-time midpoint stages. Each sweep resets electron/field state to
the accepted left endpoint, scans causal RK4 microsteps along the old ion path,
and reconstructs the new ion stages and integral endpoint from the complete
stage RHS and the RK4 tableau. The initial path is tangent; it is not an extra
impulse. At a fixed point these stage equations equal synchronous fine RK4
under the same canonicalization. Finite tolerance is a separate approximation.

The independently rederived exact-fast T0 Hall map reproduces the original
M4/8/16/32 seeded gains. Its positive eigenbranch has an H^4 per-step logarithmic
gain and an H^3 fixed-horizon leading effect. Eigenvalue modulus, eigenvector
powers, actual seeded powers and worst-seed operator norms are distinct.
This explains the tested transverse T0 coupling defect; it is not a general
pressure-branch, nonuniform-background, wall or asymptotic-stability result.

The selected T1 explicit-stage map is causal and nilpotent in exact arithmetic.
Finite sweep caps, invalid intermediate states, transient growth and roundoff
remain practical limits. Intermediate energy defects are observations with
original thresholds; necessary state/solve/constraint gates remain active.
Only a converged window with final hard checks can commit, exactly once.
Failures preserve accepted state/cache and cumulative ledgers. Restart checks
scheme, M, tolerance and sweep cap. Two ion paths and one diagnostic sweep are
retained; there is no complete S-sweep history allocation.

## Same controlled matrix and retained limits

Twenty CPU cases completed 2T at clean benchmark commit
`4e9794a3f4f3d146e4a2badc124ea3dcbe04e3c6`; numerical core commit `4c75394`.
The unchanged nine-point periodic temporal fixture uses K=0, Nphi=1,
target/high spatial harmonics1/4, amplitudes2e-6/2e-8, h=T/2048 and the same
synchronous/independent references. Micro CFL is electron/field whistler CFL;
this transverse fixture does not determine ion CAW/fast-magnetosonic macro CFL.

Default tolerance1e-10/cap12 main cases retain both seeds and M4..256. The
original target vector/amplitude1%, phase0.1rad, high maximumgain1.01 and
energy8e-9J criteria are unchanged. All main target cases pass their declared
criteria; seeded high-mode behavior is separately listed below.

| M | Sweeps (mixed) | Maximum / final high-mode gain | High-mode vector error vs fine RK4 | Growth criterion |
|---|---|---|---|---|
| 4 | 2–2 | 1 / 0.9999999106 | 8.76292e-07 | pass |
| 8 | 2–2 | 1 / 0.9999981316 | 1.4060678e-05 | pass |
| 16 | 2–2 | 1 / 0.9999419104 | 0.00022430498 | pass |
| 32 | 2–2 | 1 / 0.998229543 | 0.0035408532 | pass |
| 64 | 3–3 | 1.000267277 / 1.000267277 | 0.00030697597 | pass |
| 128 | 3–3 | 1.014420105 / 1.014420105 | 0.016841433 | FAIL |
| 256 | 5–5 | 1 / 0.9988288728 | 0.0012983335 | pass |

M4/8/16/32/64 form the passing sampled sequence, eight times the previous T0
M8 window. M128 fails the high-growth budget despite satisfying its residual
gate. M256 is an isolated passing sample, not proof of a contiguous range to256.
A completed trajectory is execution status, distinct from qualification.
The high-mode criterion limits amplification, not all high-mode phase/damping
errors. No unsampled-window or asymptotic conclusion follows.

Tightening M16/M32 mixed controls to1e-14 increases sweeps to3/4 and reduces
high-vector errors against true synchronous fine RK4 to7.87006e-8/4.09106e-9.
The corresponding default errors are2.24305e-4/3.54085e-3. Physical-reference
normalization is not small-signal relative accuracy. The remaining fine RK4
continuous-reference error is about1.17e-6 at h* over2T. Fixed H=T/16,T/8
half-microstep controls retain those distinctions; the default H=T/16 gain
failure remains about1.01442 after microstep halving.

All cases retain `COMPLETED_WITH_SOFT_FAILURES`: nonapplicable wall/massless
channels and available momentum/angular soft observations are distinct.
The temporal adapter uses artificial R_PERIODIC=1e30m; its radius-weighted
angular diagnostics are not realistic-device conservation validation. No
claim that all physical diagnostics passed or all soft defects are harmless
is made. Original checks, accepted histories, ledgers and prior failures remain.

## Verification, cost and artifacts

- Final full CPU suite:886 passed,9 GPU-conditioned skips,3 subtests passed.
- Owned RTX5090 job11081: M4/16/32, four accepted macros each, mixed seed;
  maximum high-mode CPU/GPU relative difference9.57e-15. All path arrays were
  resident complex128; existing GPU regressions12 passed. Not full2T GPU scope.
- Independent reviews reproduced48 fixed-sweep rows and4 exact T0 maps;
  source/transaction, actual archived-layout exporter and scientific evidence
  were reviewed separately. Package and artifact checks are in the run logs.
- Full model calls per accepted macro are S*(6M+1)+1, including2MS midpoint
  alias probes. Every microstage still evaluates the complete ion RHS.
  Storage is a conservative preflight estimate, not a measured isolated peak.
  Concurrent CPU walltime is recorded, with no equal-accuracy speedup claim.
- The first `matrix-default` attempt is preserved: late export layout failure,
  not numerical failure. Raw history now saves before derived comparisons.
  Original case summaries also retained a false aggregate-unavailable field;
  repaired extraction recognizes normalized keys. Derived summaries reconstruct
  it from original iteration records and preserve the original summary/provenance.

[English A0-style HTML gallery](/home/lxzeng/runs/fomhd/a1-benchmark__20261107T043500Z/a1-benchmark.html)
uses the original renderer/CSS, seven figures and eight pages.
[Chinese report](/home/lxzeng/fomhd-design/mainline_design/implementation/A1/A1_T1_REPORT.md)
explains the results and remaining limits.
[Run evidence](/home/lxzeng/runs/fomhd/a1-t1-waveform__20261107T043500Z/)
includes the20-case matrix, independent scripts, raw archives, audit receipts,
CPU/GPU logs, derived plot data and final version state. Original94 T0 files
and the author's time-design body are hash-unchanged.

The T0 baseline is `release/a1-t0` atfc9a19e; A0 remains`release/a0` at9a38bf4.
T1 develops on`codex/a1-t1-waveform`, merges after review/verification, and has
report-scoped maintenance line`release/a1-t1`. One solver tree retains selectable
synchronous/predictor/waveform strategies. No immutable tag or design-repository
push is made. T2-T4, sparse production scaling, realistic walls, full GPU
qualification, multi-GPU execution and final performance remain outside this scope.
