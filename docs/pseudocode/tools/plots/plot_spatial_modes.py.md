# Fixed-domain spatial and eight-cycle mode figures

## A0-PY-PLOT-SPATIAL-MODE-01 Collect and audit evidence

1. Accept `--root RUNROOT` and `--output NEWDIR`. Refuse an existing output
   before reading any report or array. Read all evidence as host files on CPU;
   never import or run the solver.
2. Visit the known spatial, static, dynamic, quiet, and mode run directories.
   SHA256 every present `report.json` and every sibling NPZ byte stream. Parse
   reports from those same bytes. Store backend, device, suite scope, source
   snapshot, software, run status, and input hashes in the output data.
3. Build one ledger containing every reported case and every explicitly
   expected-but-missing case. Preserve measured, completed, failed, skipped,
   running/incomplete, missing-report, missing-array, and invalid-evidence
   states. A missing or failed case does not abort the complete set; it is
   retained and omitted only from measured-success marks.
4. For every measured spatial case, require a matching `arrays_sha256` for the
   exact NPZ bytes loaded. Check all arrays are finite and that coordinate,
   mask, and wall-slot lengths agree. Require saved cell centers, spacing `h`,
   fixed bounds `[1,2] x [0,1]`, target area, and the 0.15 m region partition
   from the report. Compute area relative error from the saved active-cell count
   and `h`; do not synthesize cell geometry.
5. Extract spatial measurements from report objects with their native units:
   corrected-field relative weighted L2 (dimensionless), regional corrected
   field weighted RMS (V/m), E0 and wall-correction comparisons, and projected
   pressure-perturbation relative weighted L2 (dimensionless). Draw only
   hard-pass measured cases. Keep failures such as rectangle n24 memory
   exhaustion in the ledger and figure annotation.
6. For completed static cases, retain analytic outward x traction and total x
   rate in N and the maximum Cartesian balance-defect component in N. Never
   add numerical wall reaction to physical outward traction.
7. For completed or rejected dynamic cases, locate the sibling case NPZ and require
   `len(time_s) = accepted_steps + 1`,
   `len(momentum_x_interval) = accepted_steps`, finite arrays, increasing time,
   and matching momentum row counts. Require zero start and reported dt cadence;
   accepted time equals accepted steps times dt, while requested steps times dt
   equals the requested horizon. Rejected partial histories stop at their last
   accepted sample and retain failure/count labels. Verify that the signed cumulative sum of
   `momentum_x_interval` equals the report's
   `signed_cumulative_momentum_x_defect`; this is the Cartesian
   `p_R cos(phi) - p_phi sin(phi)` defect. Preserve endpoint defects, full and
   half-dt histories, and quiet field drift over declared scales. Keep the
   endpoint semidiscrete force defect in N distinct from accumulated impulse
   defects in kg m/s.
   Read `quiet-full` and `quiet-refined` as separate source versions, labeling
   their points original/refined. Never replace the original rejected history
   with the result of the repaired projection.
8. For each hard-pass mode case, require accepted steps equal requested steps,
   `time_s` length `steps + 1`, `mode` shape `(steps + 1, 7)`, finite values,
   strictly increasing time, endpoint equal to the reported accepted/target
   horizon, `dt * steps` consistent with that horizon, the report's common
   eight-cycle horizon, and mode-fit sample/interval counts equal to the loaded
   history. Independently refit unwrapped phase and log amplitude from the loaded
   electron-density coefficients; verify report frequency/growth, phase-fit
   residual and measured cycles. Plot the refitted values. Require eight
   analytic cycles and enough measured samples. Include
   separately named retries without replacing the original failed attempts.
   Draw only cases satisfying these checks and `hard_pass`; retain
   other cases as failures/incomplete evidence.

## A0-PY-PLOT-SPATIAL-MODE-02 Render four standalone figures

1. `fixed-geometry`: 2x2 rectangle/disk masks at n=8 and the largest measured
   level per shape. Draw every saved active cell center as an `h` by `h`
   square. Overlay the fixed rectangle or 128-vertex disk reference and the
   distinct true circle of radius 0.45 m at `(1.5, 0.5)`. Annotate backend, h,
   area relative error, and unavailable/failed n24 evidence.
2. `spatial-continuum`: 2x3 panels, compact/wall rows. Columns show corrected
   field global relative weighted L2 with E0 comparison, corrected field
   weighted RMS for all/inner/wall regions plus wall-correction RMS, and
   projected pressure-perturbation relative weighted L2. Distinguish rectangle
   and disk, label units, and state that slopes are descriptive rather than an
   assumed asymptotic order.
3. `momentum-controls`: 2x3 panels for static CUDA analytic x traction versus
   discrete x rate, static balance residual, dynamic CPU signed cumulative
   momentum-x defect versus time, endpoint defect versus h, n6 320/640 cadence
   comparison, and quiet declared-scale drift. Show missing dynamic-halfdt or
   quiet evidence explicitly.
4. `eight-cycle-modes`: 2x3 panels for actual complex electron-density real
   histories versus the continuum line, actual history reconstruction over one
   periodic wavelength, measured frequency relative error versus dt, measured
   damping versus RK4 prediction and zero continuum damping, normalized
   full/half-amplitude history differences when both exist, and phase-fit
   residual versus dt. Label the periodic spectral adapter as test-only and do
   not describe the reconstruction as a global Tokamak field.

## A0-PY-PLOT-SPATIAL-MODE-03 Write bounded outputs

1. Create the new output directory only after successful collection. Write
   exactly four PNG/PDF figure pairs, `plot-data.json`, `provenance.json`, and
   `SCOPE.md`. Write no HTML and no cache files in the output directory.
2. Keep SI values in plot-data. Scale axes only with explicit labels.
3. `provenance.json` records every read report/NPZ hash, report metadata,
   command, Python/NumPy/Matplotlib versions, and script SHA256.
4. `SCOPE.md` states that these are offline fixed-domain numerical measurements,
   with no new physical acceptance threshold, no ICE/nonlinear/closed-wall
   validation claim, and no Warp-versus-C++ performance claim.
5. If evidence is malformed where a success mark would be produced, fail with
   a specific error. Missing/incomplete runs remain visible in the ledger and
   figure annotations rather than being silently discarded.
