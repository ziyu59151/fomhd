# Tokamak CAE response figures

## A0-PY-PLOT-CAE-01 Select cases and audit inputs

1. Accept `--run-root PATH`, `--output NEWDIR`, and optional
   `--cases name[,name...]`. Refuse an existing output before reading any
   input. Explicit names must be direct child directory names.
2. With no `--cases`, inspect direct child names only. Recognize semantic
   cases named `(circle|east)<nodes>-n<0|1>-...-r<revision>`. For each
   `(kind,nodes,harmonic)` key, select the greatest revision present. This
   excludes generic smoke directories and superseded faulty pilot revisions.
   A selected failed or budget-stopped latest revision remains selected.
3. For every requested case, read `report.json`, optional `case.toml`, the two
   history JSON files, and the signal/mapping/matched NPZs that exist. Load JSON
   and NPZ from the same bytes that are hashed, with pickle disabled. Retain a
   case even when its report or any arrays are missing or malformed.
4. Record every byte stream read in `receipts/report.json` with its case-relative
   path, role, size and SHA256. Include backend, nodes, retained `K`, physical
   `nphi`, source hashes/source-bundle digest, equilibrium source hash and
   assumptions. Distinguish plot completion from solver status and never mark
   scientific acceptance.

## A0-PY-PLOT-CAE-02 Preserve field meanings

1. Nodal signal files store `density` as `time x (5*Nc)`, and `BN` or
   `velocity` as `time x (5*3*Nc)`. Accept the equivalent explicit five-sample
   shape only after cross-checking the cell count against the mappings and
   matched response. Retain the zero toroidal Fourier coefficient for `n_phi=0`
   as the five-phase mean; do not skip or drop it. For `n_phi=1`, obtain the
   retained complex coefficient by quadrature at the five supplied toroidal
   phases. `matched-response.npz`
   `delta_B` is already a retained coefficient; ion-density `delta_n` and
   `delta_velocity` retain the five-sample axis and need the same extraction.
2. Keep three field meanings distinct in labels: the perturbed original field,
   its change from the perturbed initial state, and its difference from the
   matched control. Do not label any of these as a new physical mode.
3. The retained `n_phi=1` coefficient is `A/2`. Reconstruct physical spatial fields
   with `physical_at_phi(c,mode,phi=0)`: return `Re(c)` for `n_phi=0`, and
   `2 Re(c exp(i*n_phi*phi))` for `n_phi=1`. Use this helper for signed
   `delta_B_parallel`, signed ion density `delta n_i`, ion velocity magnitude,
   and ion velocity vectors.
   Overview histories remain retained coefficients with the `A/2` convention
   stated explicitly.
4. Use only `R`, `Z`, and `polygon` from `input_mapping.npz` for cell maps.
   Do not place staggered `background_raw_B`, `background_mapped_B`, `raw_B`,
   or `mapped_B` values on those centers. Reuse `flat_cell_grid` from
   `plot_grid_convergence` to give every active cell its rectangular tile and
   mask inactive cells. Draw with `pcolormesh(..., shading="flat")`; do not
   normalize, smooth, or interpolate the physical fields. Show the input wall
   as one solid black outline. Record the flat-cell and no-interpolation
   geometry, physical units, selected frame times, speed definition, and
   direction-only arrow meaning in the existing public receipt metadata.
5. Determine the compressional time only from matched ion-density perturbation at
   the supplied probe cells. Report the maximizing time, probe cell and exact
   initial control ion-density scale in `m^-3`. Report the physical `phi=0`
   compressional fraction. If matched ion density, probes or coordinates are
   unavailable, state why and do not choose a substitute time.

## A0-PY-PLOT-CAE-03 Render overview and structure

1. For each requested case write one `overview` PNG/SVG pair and one
   `structure` PNG/SVG pair. A failed, stopped or incomplete case still receives
   both pairs with its actual status, failure and unavailable-panel reasons.
2. The overview shows control and perturbed energy histories separately:
   absolute `ion_J`, `electron_J`, `field_J`, `total_J`; relative changes from
   each baseline including supplied `relative_total_change`; and
   `balance_J`, `outward_J`, `source_J`, plus matched perturbation magnetic and
   ion kinetic energy on a signal-scale view. Label `ion_J` as ion total energy
   because it combines kinetic and internal energy. Also show probe histories:
   `n_phi=0` uses the real five-phase zero-Fourier coefficient; `n_phi=1` shows
   real and imaginary coefficient histories. Use one legend convention for
   original, change-from-initial and matched-control difference.
   For `n_phi=1`, render both real and imaginary retained-coefficient probe
   panels with one shared generic ion-density label. Select that label before
   the `n_phi=0`-only `real`/`changed`/`matched` label map.
3. Show only a frequency explicitly flagged available in
   `report.mode_fits[*].fit.available`. If none is available, display the
   recorded reasons and no numeric frequency. Never infer a frequency from a
   planned duration, guessed cycle count, target period count or pilot horizon.
4. The structure figure is one compact 1 x 3 row. Show signed physical
   `delta_B_parallel` from stored `parallel_B`, signed physical ion density
   `delta n_i`, and total reconstructed ion speed
   `sqrt(|delta v_R|^2 + |delta v_phi|^2 + |delta v_Z|^2)` at phase `phi=0`.
   Use `RdBu_r` for each signed map and `viridis` for speed. Keep each panel's
   zero-centered or nonnegative physical color limits; a zero field is still a
   finite physical extent and uses a readable default extent. Add short dark
   arrows at the tile centers for the normalized reconstructed R/Z velocity
   direction only. Thin arrows on large grids and bound arrow length to a
   fraction of the cell spacing. The density maps use the selected
   compressional time; speed and arrows use the independently selected strongest
   velocity time. Label every panel's timestamp. Name the numerical grid as
   `N_R x N_Z` (`grid 20 x 20`), the harmonic as `n_phi`, and the phase as
   `phi=0`. Show the solid black wall outline, with no repeated wall legend,
   probe ring, center-dot overlay, in-axis footnote, or data-area prose.
   Reserve canvas margins for all three colorbars and their SI labels; verify
   rendered axis decorations stay inside the exported image.
   Use the same mapping/alignment availability reason in the renderer and
   spatial receipt. A missing or misaligned mapping produces unavailable maps
   and preserves its concrete reason in metadata.
5. Put one teal corner banner `NL | FOMHD time evolution` on every overview and
   structure figure. Include the device and exact solver status in that banner,
   preserving `completed`, `budget_stopped`, `failed`, `unavailable`, and
   `initializing`. Put a concise case/grid/`n_phi`/`phi` heading on the structure
   figure. Do not describe an initial mapping, missing report, or unavailable
   response as completed evolution. Keep the nonspatial overview style
   unchanged.

## A0-PY-PLOT-CAE-04 Write bounded gallery products

1. Write figures under `overview/` and `structure/`, provenance under
   `receipts/report.json`, and an appendable HTML fragment at `fragment.html`.
   The fragment has the fixed unique section id `cae-20261106` and uses only
   relative links to the PNG/SVG products. The same figures carry the teal
   nonlinear identity rather than the linear selected-mode identity.
2. Keep statuses, failure reasons, frequency availability, equilibrium
   assumptions, source identity and missing-evidence reasons visible in the
   fragment and receipt. Keep physical field meanings, reconstruction rules,
   plotted species, normalization/caption limitations, and staircase-boundary
   limitations recorded even when explanatory text is moved out of the figure
   header. Report figure generation success separately from physical or
   scientific acceptance.
3. Keep solver status and scientific limitations in the existing receipt and
   captions rather than a structure-figure footer. Preserve source identity in
   the standalone PNG/SVG banner and heading. Do not edit a combined gallery,
   rerun a solver, access a GPU, install dependencies, or make a final
   physical-frequency claim.

The `n_phi=0` original ion density has a separate zoomed axis from
initial/matched differences so the background cannot hide the wave. The spatial
speed panel uses the strongest velocity snapshot with its own timestamp,
distinct from the density-compression snapshot. Its color is the total physical
3D speed; arrows show only the normalized R/Z direction.

For zero mode, use three separate axes for original, initial-subtracted and
matched signals so neither the background nor startup drift hides the wave. Compression maps
use the strongest ion-density time; velocity maps use their strongest velocity
time and display that distinct timestamp.
