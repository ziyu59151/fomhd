# 0.5.0 two-fluid experiment platform

This experimental branch starts at `be5f1c07913409bfd377f22e9ea9d69ac614b158`.
It retains one Python/Warp solver, synchronous A0 and complete-stage T1.
D071 authorizes complete real-RZ scaling experiments alongside independent
reduced-ion multirate research. Fixed-ion T2-E remains a research module and
is not a prerequisite for these complete simulations.

The physical defaults are massless quasineutral CGL-B electrons, SI units,
FP64/complex128 and RK4. A3 adds CCS-coupled FK fast ions to bulk-fluid ions;
scoped stable A3 acceptance is the 1.0 milestone, followed by all-ion FK in A4.
Version 0.5.0 identifies a numerical experiment platform with explicit evidence
bounds, not a completed physical-validation program.

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
