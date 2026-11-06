# Static report builder tests

Target: `tests/warp/test_a0_report.py`. Tests exercise real files in pytest's
temporary directory and assert user-visible HTML or filesystem behavior; they
mock neither report rendering nor manifest validation.

```text
fixture write a real PNG/placeholder image, SVG, data and provenance files
fixture construct a hand-checked two-page manifest with:
    historical figure numbers 17 and 18
    every source badge family, including explicit mixed components
    grid N_R=64, N_Z=128, harmonic n_phi=3, retained harmonic cutoff k=96,
    N_phi quadrature=5, device, and available/unavailable statuses
    panel letters and optional links

test render:
    require escaped literal prose instead of executable markup
    require each historical figure number and unique figure anchor
    require visible "Page i / 2" labels and Index/Previous/Next anchors
    require IDs resolve uniquely within the document
    parse all local href/src links and verify that they resolve to real inputs
    require teal NL, purple LIN, gray REF and explicit MIX labels
    require distinct N_R, N_Z, n_phi, K (retained harmonic cutoff) and
    N_phi quadrature fields, with no "K quadrature" label
    require unavailable status text to remain visibly labelled
    require @page print counters and an unmodified natural-aspect image rule

test limited/provisional statuses:
    run a one-figure report with literal running, initializing,
    budget_stopped and completed statuses
    require the first three to show "Limited/provisional evidence" and retain
    their exact supplied text
    require completed to remain visible without the caution prefix
    do not rewrite budget_stopped as a completed target

test rejection:
    replace source_kind with an unsupported literal and require failure
    remove the image/provenance file and require failure
    duplicate page ids and duplicate global figure numbers each require failure
    remove output before every invalid build remains unaffected

test overwrite policy:
    first build a valid report
    second non-replace build with changed prose raises and preserves bytes
    replace build creates a timestamp/hash sibling backup holding old bytes
    identical replace build succeeds with os.replace disabled and creates no backup
    changed replace build with os.replace disabled leaves old output intact
    no temporary sibling remains after the atomic failure
```

The tests also prove that an accepted-looking file name cannot supply
`source_kind`; unknown categories fail regardless of image extension.
