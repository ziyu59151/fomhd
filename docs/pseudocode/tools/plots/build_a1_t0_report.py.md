# Build A1 T0 report

Purpose: turn the parent-owned compact `plot-data.json` into a fresh,
self-contained A0-style HTML report without running a solver or changing the
scientific data.

## Command line

```text
build_a1_t0_report.py --data COMPACT_JSON --output FRESH_DIR
```

Optional runtime-only arguments select the read-only canonical renderer and A0
HTML. The source root is `FOMHD_SOURCE_ROOT` when set; otherwise it is the
repository containing this script after integration. Plot-style paths are
derived from that source root.

## A1-PY-PLOT-T0-CONTROLLED-01 - Executable sequence

1. Resolve all paths without creating the requested output.
2. Refuse immediately if `--output` already exists.
3. Load the compact JSON. Reject invalid JSON, missing top-level keys, malformed
   required records, nonfinite numeric values, mismatched series lengths, and
   visible Chinese characters.
4. Resolve every `input_provenance` path relative to the compact JSON directory
   when needed. Require a regular local file and verify its SHA-256.
5. Resolve the canonical renderer, A0 HTML, and two A0 plot-style files. Confirm
   that they exist and that the renderer CSS block equals the canonical A0 HTML
   CSS block byte-for-byte.
6. Record hashes of the compact JSON, renderer, A0 HTML, style files, and this
   builder before rendering.
7. Create the output directory with exclusive semantics. If creation fails,
   stop without replacing anything.
8. Configure Matplotlib with Agg, DejaVu Sans, 9-point base text, 9.5-point
   titles, 8.5-point axis labels, 7.5-point ticks, grid alpha 0.18, and text-as-
   text SVG (`svg.fonttype = none`). Render at 160 dpi.
9. Render exactly seven figures:
   - Figure 1: full, frozen, acoustic, and generic frequencies against spatial
     harmonic; a second panel plots `omega*h/(2*sqrt(2))` for candidate
     microsteps, with old and chosen steps explicitly labeled.
   - Figure 2: high-mode micro controls. Plot actual and reference real
     components only in the first 0.3 target periods, but plot vector error on
     the full supplied horizon. State completion/status and actual horizon for each control; only frozen-ion controls omit the coupled accepted transaction.
   - Figure 3: completed target-seed accuracy, phase, and amplitude metrics
     against `H/T`. Separate predictor/synchronous coupling and
     microstep classes. Rejected cases are annotations only.
   - Figure 4: absolute high-mode norm for mixed-seed nonlinear histories against
     accepted `t/T`; use a logarithmic axis, retain matching synchronous curves,
     and mark the last accepted point of each rejected prefix. Give every M a distinguishable line style. A second panel shows normalized gains for M4/M8/M16 and the predeclared 1% line.
   - Figure 5: fixed-`H` `T/16` and `T/8` comparisons at `h*` and `h*/2`, with
     both seeds, completion and accepted-horizon context. Preserve high-mode and energy comparisons in an adjacent table, including partial outcomes.
   - Figure 6: total and kinetic residual magnitudes on accepted prefixes,
     divided by the unchanged supplied energy budget, plus distinct rejected
     candidate markers.
   - Figure 7: explicit LIN macro-map radius and seed gain, a qualification table,
     and a cost table. Cost is diagnostic-inclusive elapsed run time; memory is
     the supplied estimate, not measured peak memory. Each seed has its own accepted horizon beside elapsed time.
10. Write each figure as both SVG and PNG. Require nonempty files.
11. Copy the compact JSON byte-for-byte to `data.json`.
12. Build an eight-page manifest. Use only completed endpoint values in Figure
    3. Never fill a rejected endpoint with zero or a final metric.
13. Call the unchanged canonical `render_report` function. Verify its hash did
    not change during rendering.
14. Base64-embed each generated SVG in the HTML, replacing exactly one matching
    PNG source per figure. Verify seven embedded images and no remaining
    figure-PNG image sources.
15. Write `manifest.json` and `a1-benchmark.html`.
16. Build `provenance.json` with actual resolved input paths and hashes; script,
    renderer, A0 HTML, and style hashes; compact-data copy; source commit;
    generation time; rendering policy; and SHA-256 plus byte size for every
    output that can be hashed before provenance is finalized.
17. Recheck the renderer and compact input hashes after rendering. Fail if
    either changed.
18. Print one JSON summary to standard output and return success only if all
    writes and internal assertions complete.

Failures after output creation are left in place for inspection. The builder
never deletes a unique output or replaces an existing output path.

## Figure and page rules

- All source, comments, logs, manifests, and visible text are English.
- The badge vocabulary is the renderer vocabulary: NL, LIN, REF, VER, and MIX.
- Generic frequency is an estimate, never a measured eigenfrequency.
- The m=4 fast branch is never called CAW. CAW is reserved for the fast
  magnetosonic/compressional Alfven branch.
- Rejected requested windows are marked undetermined and never plotted as
  completed final metrics.
- Prefix histories use only samples strictly before `rejected_time_over_T`; candidate-at-boundary samples are excluded.
- Macro-step high-mode phase aliases are retained in source data and are not
  used to infer a frequency.
- The conclusions page uses the parent-supplied `recommendation` text.

## Checker

`check_a1_t0_report.py --data DATA --output FRESH_DIR` performs assert-based
offline checks:

- output refusal leaves the requested destination unchanged;
- all required artifacts exist and copied data equals the input byte-for-byte;
- seven PNG and seven SVG figures exist and are structurally readable;
- HTML has eight pages, seven figure IDs/numbered headings, working fragments,
  and seven embedded SVG data URIs;
- extracted HTML CSS equals canonical A0 CSS;
- HTML has no CDN, remote URL, script, or runtime-network asset;
- SVG files contain text elements and no embedded raster image;
- provenance input/output hashes match current files;
- completed-only endpoint values and rejected-endpoint exclusions agree with
  source data;
- all visible HTML text passes the English-character check.

The checker must not use pixel comparisons. It reports each check in plain
English and exits nonzero on any assertion.

The reviewed canonical renderer fingerprint includes the A0 mobile CSS rule;
CSS equality remains byte-for-byte. Historical solver inputs/data stay intact.
