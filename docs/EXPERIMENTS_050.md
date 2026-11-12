# 0.5.0 two-fluid experiment platform

This experimental branch starts at `be5f1c07913409bfd377f22e9ea9d69ac614b158`.
It retains one Python/Warp solver, synchronous A0 and complete-stage T1.
D072 selected few-ion-node multirate waveform relaxation on this experiment
branch. Its code, results and failures are retained during the A0 baseline
work. The [rectangular conductor evidence](benchmarks/rectangular-wall/README.md)
is the shared physical reference. T2-E, C1/C2 and unimplemented T tiers remain research
controls and are not prerequisites. Prior complete real-RZ experiments and
their exact acceptance scopes remain evidence; a new method needs separate
complete interval, rollback, restart, CPU/GPU and scientific verification.

The physical defaults are massless quasineutral CGL-B electrons, SI units,
FP64/complex128 and RK4. A3 adds CCS-coupled FK fast ions to bulk-fluid ions;
scoped stable A3 acceptance is the 1.0 milestone, followed by all-ion FK in A4.
Version 0.5.0 identifies a numerical experiment platform with explicit evidence
bounds, not a completed physical-validation program.

## Retained WR experiment entries

`tools/warp_wr_periodic.py` and `tools/warp_wr_rz.py` exercise the complete
`multirate_wr` transaction. They retain exact accepted endpoints, all cumulative
fields, source/input/scheme hashes, attempted work, failure arrays and strict
checkpoint continuation. The RZ entry uses the existing EAST mapping, matched
control/perturbed runs, K=1 and Nphi=5. Its explicit `--waveform-rtol` is part
of checkpoint identity; tightening it is a separately labeled experiment.

One H contains M field steps and three or five direct ion nodes. The selected
discrete equations differ from T1. Raw hard checks keep their original
thresholds; observed budgets retain their original observations. A converged
window may still fail a budget. A capacity-plan rejection is not a measured
out-of-memory event or a stability conclusion. The current conservative WR
plan reserves retained records for the configured maximum outer sweeps.

Every spatial experiment must include direct active-cell RZ component maps
and perturbed-minus-control maps at matched times. Use the A0 renderer and
English figure style, show units and wall mask, and avoid interpolation over
inactive cells. Retained historical cross sections are rebuilt by
`tools/plots/build_050_sections.py`; they do not establish new WR accuracy.

## Shared implementation scope

Massless inputs omit unused finite-mass field-system construction. Fixed
geometry operators and the wall cache remain shared. The density-dependent
projection uses an exact diagonal normal only when its row connectivity
permits it; coupled rows retain the original solve. Memory plans count actual
operator storage and keep outer-window history ownership when a temporary
alias check increases the sampling count. These are ordinary spatial/algebra
improvements; they do not shorten the temporal dependency graph.

`time_waveform` still evaluates complete ion RHS stages. `time_affine`,
`time_field` and `field_batch` belong to bounded fixed-ion field research;
their gains do not become complete-T1 gains merely by sharing a repository.

## Complete-run entry

`tools/warp_050_experiments.py` uses the real equilibrium mapping and matched
background/perturbed ensembles. A macro length H contains M field microsteps
h=H/M; synchronous A0 uses dt=h and M=1. Run GPU commands in an owned Slurm
allocation with the chosen checkout explicitly on `FOMHD_SOURCE_ROOT` and
`PYTHONPATH`. Supply the external equilibrium through `--equilibrium`.

The bounded spatial ladder is 12,16,24,32,40 enclosing nodes per axis with K=1
and Nphi=5. Spatial probes preserve the same initial data, h and observation
window. Long-time experiments keep their declared grid and full requested
horizon; a resource stop publishes the actual accepted prefix separately.
Host and device caps are 16 GiB and 8 GiB. Each job is at most 30 minutes;
the campaign accounting includes setup, failures, retries and waiting inside
allocations. These caps and the actual run manifest govern execution.

Complete retained-sector archives may supply a local baseline spectral-radius
step policy after physical/background identity validation. The generic
operator estimate stays visible. Selected-target accuracy over the original
horizon and B/n neighborhood checks are separate requirements; this is not a
nonlinear stability theorem or an all-high-frequency dispersion requirement.

## Accepted state, continuation and failures

Official complete-A0/T1 restart is covered by CLI regression tests. The
experiment runner additionally supports strict accepted-pair continuation via
`--resume-from`, with immutable original baselines, integrals, H/h, source and
seed identities. It reconstructs caches using a throwaway ledger and charges
that work. This NPZ format is an experiment-runner format, distinct from the
official production case checkpoint. Failed trial arrays are stored separately
from both ensembles' last common accepted endpoint.

Source/operator fingerprints must match. Pre-0.5 massless checkpoints carrying
the complete finite-mass operator fingerprint require their original checkout;
there is no automatic checkpoint migration. Raw historical results and gates
are retained unchanged.

Actual inputs, source fingerprints, accepted prefixes, full checks, scalar
histories, bounded spatial snapshots, timing samples and failure records live
outside the source repository under the dated personal run directory. The
design authority owns the corresponding report and A0-style English gallery.

## Bounded results (2026-11-09)

The verified campaign reached complete A0 at40 enclosing nodes per R/Z axis and
T1 at32, with K=1/Nphi=5; these are short capacity probes. At40 T1 is rejected
before acceptance by its9.532GB storage plan against8GiB, not by an observed
OOM or instability. Independent real-RZ12-node long runs retain matched A0,
T1 and h/2 A0 endpoints at351.980629ns (about2.018 target periods). The original
16-period plan remains incomplete. Soft energy/power/momentum observations and
nonzero mapped equilibrium residuals remain explicit.

Shared same-GPU complete-advance medians improve by1.374x for A0 and1.163x for
T1 in the measured16-node window; counts are unchanged. This is ordinary
spatial/algebra/backend optimization, not new time parallelism. The separate
reduced-ion research branch retains both successful and failed accuracy
controls, with no demonstrated FK saving or net same-GPU candidate gain.

The full numerical/physical scope, raw paths, strict continuation audit and
A0-style English offline gallery are linked by the design authority's
`mainline_design/implementation/A1/A1_050_MULTIRATE_REPORT.md`. Literature and
large run products remain outside this source checkout. The gallery can be
rebuilt with `tools/plots/build_050_multirate_gallery.py` using explicit run,
source and output roots; output must lie outside both input roots.
