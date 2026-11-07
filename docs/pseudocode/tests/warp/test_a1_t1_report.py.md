# Test A1 T1 report builder

Purpose: verify the compact-input contract and a complete small synthetic render
without running a solver or using production numerical data.

## Fixture

1. Create a temporary compact JSON with the frozen protocol, map rows for
   M=4..256, bounded synthetic histories for M=4, 16, 32, and 256, fixed-H
   `h*`/`h*/2` controls, M16/32 mixed `h*=T/2048` controls at rtol=1e-14 and
   1e-10, supplied high-sync vector-error scalars, iteration diagnostics, and one
   GPU parity row. Primary M128 mixed is completed above the growth limit and
   primary M256 mixed is completed below it. Every T1 fixture row uses
   `coupling=waveform`.
2. Mark one case rejected with samples at, before, and after the rejection time.
3. Keep target-only `high_norm_ratio=null`; do not manufacture a gain.
4. Create a preserved T0 `data.json` and HTML under an absolute temporary path,
   with predictor target/mixed rows across M=4..256 for overlay checks.
5. Create English evidence/baseline roots and a bounded `chinese-draft` HTML
   anchor. Use only temporary files for all writes.

## Checks

1. A fresh build creates `a1-benchmark.html`, `manifest.json`, `data.json`,
   `provenance.json`, and seven nonempty PNG/SVG pairs.
2. The copied data is byte-for-byte identical to the compact input.
3. Extracted HTML CSS equals the canonical A0 CSS byte-for-byte.
4. HTML is English, has eight unique pages, seven unique figure IDs, embedded
   SVGs, local fragment/file links, and artifact links to SVG/PNG/data/
   provenance.
5. Captions preserve the independent Hall eigen/singular/seed quantities as REF
   and actual production trajectory gain as NL. LIN remains reserved for a
   production-RHS Jacobian and is not presented. Text states the finite
   two-period transverse-model scope and makes no asymptotic-stability or
   all-soft-checks-passed claim.
6. The manifest preserves the M=4..256 T0 matrix context and the original T0
   gallery/evidence/baseline anchors. Primary T1 selection is nonempty and
   waveform-only; M=4/16/32 T1 gain histories, path residuals, and energy
   residuals are selected from MIXED seed, while T0 predictor mixed overlays
   remain present.
7. Invalid series lengths, nonfinite values, duplicate JSON object keys,
   duplicate case IDs, invalid tolerance/cadence, negative magnitudes, and
   duplicate iteration keys raise `InputError`.
8. Target-only gain is unavailable rather than computed, explicit nulls remain
   null in copied data, and accepted-prefix filtering is strict before the
   rejection time.
9. An existing output path is refused without changing its contents.
10. Input data, T0 data, and canonical sources retain their bytes after a
    successful build.
11. Focused Figure 6 regression verifies M16/32 tight-tolerance cases exist,
    `high_sync_vector_error` validates as optional/nonnegative, Figure 4 uses
    primary mixed residuals, and Figure 6 renders both true tolerance ticks,
    fine-RK4 vector error, and separate final-gain deviation.
12. Focused high-gain-budget regression verifies data-derived M128 exceedance
    notes and annotation, exclusion of the larger passing M256 point, and that
    report introduction/conclusions state completion is execution status rather
    than physical or accuracy qualification. Figure 4 text distinguishes the
    two same-time RK4 midpoint stages from each other.

The tests do not inspect production claims or visuals. Parent owns the final
actual-data render and visual/browser review.
