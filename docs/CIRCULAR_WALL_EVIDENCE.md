# Circular-wall experiment evidence and route status

Status: bounded experimental implementation; physical spatial acceptance is
withheld. The 10 November 2026 route review precedes further scale expansion.
This record is an evidence index, not a completed production benchmark report.

The design authority is
[the spatial route review](/home/lxzeng/fomhd-design/mainline_design/implementation/A1/A050_SPATIAL_ROUTE_REVIEW.md)
and [the selected experimental definition](/home/lxzeng/fomhd-design/mainline_design/numerics/algorithm_design/A1/A050_REAL_WALL.md).
The earlier ranking of clipped Voronoi as lower risk than a structured
true-intercept boundary closure is withdrawn: that alternative was not actually
derived or compared. No main merge, backend switch or model change follows
from the review.

## Established and unresolved scope

- Independent continuous vacuum and constant-F circular GS controls are
  available. Continuous equilibrium verification does not establish a discrete
  equilibrium after sampling, reconstruction and projection.
- The polygon prototype has actual facet geometry, positive complete measures,
  closed incidence, compatible algebra, paired operators and bounded complete
  CPU/GPU transaction checks. These facts do not prove physical accuracy.
- Analytic physical-gradient curl and constant-vector divergence fail to
  converge on irregular closed interior duals. Sampling positions, generator
  distances and barycentric dual integration are not yet physically consistent.
  Changing one sampling point fixes one patch and damages another.
- Current/Ampere remapping improves one source of electron strain but is not a
  complete discrete GS/force/CGL/Faraday solve. Smooth-wall GS physical
  acceptance remains withheld.
- Ordinary projection-cache and wall-diagnostic optimizations are retained.
  They do not prove spatial-route superiority or temporal-parallel speedup.

## Reproducible evidence

The campaign is
[/home/lxzeng/runs/fomhd/a050-clt-boundary__20261109T151903Z](/home/lxzeng/runs/fomhd/a050-clt-boundary__20261109T151903Z).
Raw runs, source snapshots, inputs, failures and existing cross sections remain
there. The base commit is `73fb2a048813d622b643986cf6ea58dfcb82e139` on
`codex/a050-two-fluid-experiments`; actual uncommitted numerical source is
identified by the retained per-file archives and hashes.

- [Local physical patch reproduction](/home/lxzeng/runs/fomhd/a050-clt-boundary__20261109T151903Z/analysis/spatial-review/physical_patch_probes.py).
- [Fresh probe receipt, arrays and units](/home/lxzeng/runs/fomhd/a050-clt-boundary__20261109T151903Z/analysis/spatial-review/recheck-20261110/physical-patch-probes.json).
- [Direct assembled source-matrix confirmation](/home/lxzeng/runs/fomhd/a050-clt-boundary__20261109T151903Z/analysis/spatial-review/source-api-crosscheck.json).
- [Ampere-only magnetic remapping](/home/lxzeng/runs/fomhd/a050-clt-boundary__20261109T151903Z/analysis/full-discrete-gs-probe.json).
- [Existing first-response cross sections](/home/lxzeng/runs/fomhd/a050-clt-boundary__20261109T151903Z/gallery-p02-first-response/index.html).
- [Resource ledger](/home/lxzeng/runs/fomhd/a050-clt-boundary__20261109T151903Z/RESOURCE_BUDGET.json).

These are different evidence scopes. A local manufactured field is not a
time-evolved plasma response; a discrete-current projection is not an
independent equilibrium reference. The existing linear cylinder wave control
has finite-aspect-ratio and initialization/projection limitations when applied
to the nonlinear toroidal run.

The next recommended comparison is one bounded CPU structured-boundary closure
oracle, with the existing polygon prototype as a comparator. It is proposed,
not implemented by this record. The minimum tests and stopping conditions are
in the design review. Larger production runs await that route decision and
physical consistency evidence.
