# A0 static report builder

Target: `tools/plots/build_a0_report.py`. This host-only standard-library tool
renders a bounded, accessible HTML report from a JSON manifest. It neither
computes scientific quantities nor rerenders figures.

## Input contract

```text
manifest := {
  title: nonempty text,
  introduction: text,
  notation: nonempty [{symbol: text, meaning: text}],
  archive: {label: text, path: manifest-relative local path},
  pages: [{id: unique HTML id, title: text, introduction: text,
           notes: [text], figures: [figure]}],
  conclusions: text,
  known_issues: [{label: text, path: manifest-relative local path}]
}

figure := {
  number: unique positive integer,
  title: text,
  source_kind: one of nonlinear_evolution, production_linearization,
              independent_reference, verification, mixed_comparison,
  image: manifest-relative regular file,
  caption: text,
  grid?: {n_r: value, n_z: value},
  n_phi?: value, k?: value, n_phi_quadrature?: value,
  device?: text, status?: text,
  svg?: manifest-relative regular file,
  data?: manifest-relative regular file,
  provenance?: manifest-relative regular file,
  panels?: [{letter: a-z, description: text}],
  collapsed?: boolean,
  components?: nonempty [text] required only for mixed_comparison
}
```

The source category comes only from `source_kind`. File names and extensions
never select or change it. `n_phi` is the toroidal harmonic and remains
separate from `n_phi_quadrature`. `k` is the retained toroidal harmonic cutoff,
not a quadrature parameter, and is also rendered separately. The builder copies
prose and statuses literally after HTML escaping.

## Render

```text
resolve manifest directory and prospective output directory
parse JSON and validate the complete manifest shape
validate reserved/unique page ids and globally unique figure numbers
require every image and optional figure/provenance file to exist
require every archive and known-issue local path to exist
map each local path to a URL-quoted path relative to the output directory
render title, introduction, notation, archive and page index
for page index i in pages:
    render visible "Page i / total" and Index/Previous/Next anchors
    render page prose and notes
    for each figure:
        render stable "Figure number" anchor and complete title
        render the source-specific text/color badge
        render image at natural aspect ratio with full-width readable limits
        render supplied grid, harmonic, retained harmonic cutoff, separate
        quadrature, device and status
        label pending/unavailable, running/initializing and budget-limited or
        incomplete statuses as limited/provisional evidence without changing
        the supplied status text
        render optional SVG/data/provenance links and lettered panel references
        when collapsed is true, place the figure in a native details block
render conclusions and checked known-issue links
escape all manifest text and attributes; never accept raw HTML
add print page counters with CSS @page while retaining logical page labels
```

The static footer states only that supplied figures do not themselves establish
convergence or acceptance. Badge text and status wording are not used to derive
scientific conclusions.

## Commit output

```text
render the complete HTML in memory before touching the output
if output exists without --replace: fail and preserve it
if output bytes equal rendered bytes: return without writing or backup
resolve the output parent and create it only after all validation succeeds
write and flush a unique temporary sibling
if replacing different output:
    copy the old output once to
    OUTPUT.bak-UTC_TIMESTAMP-FIRST_12_SHA256_HEX (with a numeric suffix on collision)
atomically os.replace temporary sibling onto output
on any later failure remove only the temporary sibling and leave old output intact
```

The command-line interface is
`python tools/plots/build_a0_report.py --manifest PATH --output PATH [--replace]`.
Invalid JSON, unsupported categories, duplicate ids/numbers, absent local files,
and malformed fields are errors. A nonterminal or terminally incomplete status
is displayed, never silently upgraded or described as a completed target.
