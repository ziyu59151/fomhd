# Spatial and mode plotting regressions

## A0-PY-TEST-SPATIAL-MODE-PLOTS-01 Evidence integrity

1. Load saved arrays with pickle disabled and reject a changed SHA256.
2. Keep failed cases and missing reports in the ledger, including an initial
   CUDA failure with no accepted history.
3. Reject cadence inconsistent with dt, requested horizon, sample count or
   accepted steps. Keep a rejected partial history only through its accepted
   endpoint and verify its signed impulse sum.
4. Independently refit a clean complex history; reject corrupted frequency,
   growth or phase residual even when a caller reports a successful case.
5. Render missing spatial arrays as an explicit unavailable case, without
   fabricating geometry or attempting to plot absent coordinates.
6. Refuse an existing output directory before collecting inputs.
7. When the archived campaign is present, collect it read-only and require
   the known successful spatial, dynamic and periodic cases. This optional
   archive check does not replace portable synthetic integrity checks.

Source: `tests/warp/test_spatial_mode_plots.py`. All plotting operates on host
files; it neither runs nor modifies the solver. The actual figure review
additionally compares retained arrays against independent analytic derivatives
and fits; unit tests alone do not establish the scientific conclusions.
