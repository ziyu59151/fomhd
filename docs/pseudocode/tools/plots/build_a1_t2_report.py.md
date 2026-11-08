# A1-PLOT-T2E-01: A0-style reviewed T2-E gallery

Input is parent-reviewed English JSON containing figure questions, controls,
criteria, evidence scope and numeric series. This renderer does not derive a
scientific result, acceptance status or speedup from unreviewed solver output.

1. Resolve the source root, input and new output directory. Refuse overwrites.
   Load the existing A0 renderer and T0 Matplotlib style helpers by exact path.
2. Validate finite numeric series, matching lengths, unique safe IDs, explicit
   evidence kinds, one to four panels, English metadata and existing local links.
3. For each figure, plot the supplied series and horizontal criteria. Use the
   existing palette/fonts/grid/panel-label/export helpers; save PNG and SVG.
4. Build A0-compatible page/figure records with the supplied question, controls,
   criterion and evidence components. Use the exact A0 renderer and CSS.
5. Embed the exported SVG bytes in the HTML for offline viewing. Retain SVG,
   data and archive links. Copy reviewed JSON byte-for-byte and save a manifest.
6. Verify CSS identity, English output, exported assets, local links and embedded
   bytes, then record input/renderer/style/HTML hashes. Input and historical
   galleries are read-only. Any validation or I/O error propagates explicitly.

The builder has no GPU/backend solver dependency, no network assets and no
runtime model controls. Renderer-fixture and negative-schema checks are external
mechanical evidence; the final scientific plots require review and inspection.
