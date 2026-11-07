# Build A1 T1 report

Purpose: turn a parent-owned compact T1 JSON file plus the preserved T0 gallery
into a fresh, offline A0-style gallery. The builder never runs a solver, changes
an input, or changes a scientific threshold.

## Command line

```text
build_a1_t1_report.py --data COMPACT_JSON --output FRESH_DIRECTORY
```

Optional source-location arguments may identify the read-only FOMHD checkout,
canonical A0 renderer, and canonical A0 HTML. The default source root is the
checkout containing the integrated script, or `FOMHD_SOURCE_ROOT`.

## A1-T1-GALLERY-01 - Executable sequence

1. Resolve the compact input, output destination, source root, canonical A0
   renderer, canonical A0 HTML, and the two A0 plot-style sources.
2. Refuse an existing output path before loading or creating anything.
3. Hash and parse the compact JSON once. Require the frozen top-level groups:
   `protocol`, `t0_gallery`, `linear`, `cases`, `gpu`, and `provenance`.
4. Validate the fixed protocol. Require finite positive `T_s` and `h_s`,
   `cycles=2`, positive `rtol`, `max_iterations=12`, the four supplied metric
   limits, and positive `energy_budget_J`. Preserve the values without widening
   them.
5. Validate every linear row. Require unique positive integer `M`, finite
   nonnegative map quantities, and rows for M=4, 8, 16, and 32. Keep the
   per-step modulus, its Nth power, the exact seeded vector gain, operator power
   norm, and production gain as separate fields.
6. Validate every case:
   - unique `id`;
   - coupling in predictor, waveform, synchronous;
   - seed in target, mixed;
   - positive `M`, 2048 or 4096 microsteps per T, and positive or null `rtol`;
   - nondecreasing finite time and energy-time arrays;
   - exact lengths for target, optional high-ratio, high-absolute, total-energy,
     and kinetic-energy series;
   - optional scalar `high_sync_vector_error`, finite and nonnegative when
     supplied, treated as unavailable when absent or null;
   - explicit empty arrays or null are retained as unavailable;
   - no supplied negative error, gain, norm, residual magnitude, iteration
     residual, work, storage, or elapsed quantity;
   - rejected cases have an explicit rejection time and accepted prefixes use
     samples strictly before that time;
   - unique `(macro_index, iteration)` pairs and nonnegative path/endpoint
     maxima in every iteration list.
7. Validate GPU status, notes, and unique parity rows. Parity values are copied
   exactly; absent rows produce a visible unavailable note rather than a
   fabricated zero or speedup.
8. Validate provenance text, SHA-256 source hashes, and the preserved T0 gallery
   reference. Resolve an absolute T0 HTML or directory reference to its sibling
   `data.json`; load that read-only file for T0 overlays and hash both artifacts.
9. Resolve evidence and baseline roots relative to the compact input when
   needed. Require existing local anchors. Build English known-issue links to
   the preserved T0 gallery, evidence root, and baseline root.
10. Load `tools/plots/build_a0_report.py` and
    `tools/plots/build_a1_t0_report.py` from the source root. Verify the
    renderer hash and canonical CSS, then use the existing
    `configure_matplotlib`, `style_axis`, `panel_label`, `save_figure`, and
    `figure_record` functions unchanged.
11. Recheck the compact input, preserved T0 data, canonical renderer, and
    builder hashes. Only after all validation succeeds, create the output
    directory with exclusive semantics.
12. Render exactly seven figures as PNG and SVG:
    - Figure 1: exact-fast T0 seeded-map quantities and observed nonlinear production gain for
      M=4, 8, 16, 32.
    - Figure 2: T0 predictor versus primary waveform-coupling T1 maximum target
      error and high-mode gain over the original M=4..256 matrix; rejected fields
      remain distinct. Completed primary T1 MIXED cases whose supplied maximum
      gain exceeds `protocol.growth_limit` receive a separate annotation and
      marker; they are not rejected-prefix crosses.
    - Figure 3: M=4, 16, 32 high-mode gain histories, T0 predictor dashed and
      primary T1 waveform solid, with a lower deviation row around 1.01.
    - Figure 4: primary T1 waveform MIXED-seed path and endpoint iteration
      residuals by sweep for M=4, 16, 32; label the first and worst macro and
      draw `rtol`. The two same-time RK4 midpoint stages are distinct from each
      other; the converged discrete target is synchronous fine RK4.
    - Figure 5: primary T1 waveform raw total and ion-kinetic residual histories
      plus observed intermediate-sweep maxima; only completed final windows gate
      acceptance.
    - Figure 6: four panels. The top row preserves fixed-H `h*` and `h*/2`
      target-error and high-gain comparisons at H=T/16 and T/8. The lower row
      uses M16/32 mixed `h*=T/2048` cases at rtol=1e-14 and 1e-10: supplied
      high-mode relative vector error versus true synchronous fine RK4, and
      abs(final high-mode gain-1). Missing vector references stay unavailable;
      gain is never treated as vector accuracy. Fixed-H incomplete controls stay
      visible in text and marks.
    - Figure 7: model calls and conservative storage estimates, plus bounded GPU
      parity rows or an explicit unavailable note.
13. Close every Matplotlib figure and require each PNG/SVG artifact to be
    nonempty.
14. Copy the compact input byte-for-byte to `data.json`.
15. Build the A0 manifest with eight pages, English text, unique page and figure
    IDs, NL/LIN/REF/VER badges matched to source kind, and links to each
    figure's SVG/PNG, data, and provenance. State that the fixture is periodic
    nine-point controlled temporal evidence, with no global wall, CAW,
    large-production, or asymptotic-stability claim.
16. Call the unchanged A0 `render_report`. Embed the seven generated SVGs into
    the HTML and confirm no figure references a PNG as its visible image.
17. Recheck that the compact input, T0 data, renderer, and builder did not
    change during rendering. Require extracted HTML CSS to equal the canonical
    A0 CSS byte-for-byte.
18. Write `manifest.json`, `a1-benchmark.html`, and a final `provenance.json`
    containing the compact-input SHA-256, T0-data SHA-256, source hashes,
    canonical source hashes, builder hash, generated UTC time, output hashes,
    limitations, and external reference paths.
19. Print one JSON summary to stdout. Failures after fresh-directory creation
    remain for inspection and the builder never replaces that unique output.

## Figure source discipline

- NL means supplied nonlinear evolution.
- LIN is reserved for a local production-RHS Jacobian and its derived map.
  The independent two-variable Hall map, eigenvalues and seeded powers are REF;
  gains read from actual accepted production trajectories are NL.
- REF is reserved for the independently supplied exact-fast T0 map.
- VER means a numerical diagnostic or bounded parity comparison.
- T0 overlays come only from the `data.json` associated with `t0_gallery`.
- T1 primary comparisons use `coupling=waveform` and
  `rtol == protocol.rtol`; T0 overlays use `coupling=predictor`. Every
  `select_case` call names its coupling explicitly. Tolerance variants stay in
  the controls figure.
- Rejected cases stop at their accepted prefix and are never relabeled
  completed. Null unavailable values never become zero.
- Wall time is diagnostic context, not a T1 performance verdict. Memory is an
  estimate, not a measured peak. Cross-slab time-parallel work is deferred T3.
- Completed primary T1 MIXED high-gain-budget exceedances are derived from the
  supplied cases and protocol threshold. Their M, gain, and threshold appear in
  Figure 2, the matrix page, the report introduction, and conclusions.
  Completion is execution status, not physical or accuracy qualification. A
  larger passing M is not evidence of a contiguous usable range.

## Figure readability

- Keep right-edge prefix and completed-budget labels inside their axes.
- Wrap full GPU device names inside their table cells; retain hardware text.

## Failure behavior

- Validation or canonical-source failure before output creation leaves the
  requested destination untouched.
- Rendering or write failure after creation preserves the unique partial output
  for diagnosis.
- No source file under `/home/lxzeng/projects/fomhd` is written.
