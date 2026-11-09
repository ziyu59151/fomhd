# build_050_sections.py pseudocode

## Purpose and boundary

Build a bounded, offline A0-style gallery from existing 0.5.0 saved evidence.
Read `--run-root` and `--source-root`; write only below a new `--output`
directory. Reject an output inside either input root or an existing output
path. Do not run the solver, interpolate, triangulate, infer pressure, or
modify source/run evidence.

Register the plotting program itself and the canonical renderers as hashed
producer inputs. Saved retained pressure arrays exist, but this bounded
gallery does not reconstruct physical-phi pressure observations from them.

## Inputs

1. Load the canonical source modules from
   `source_root/tools/plots/build_a1_t2_report.py`,
   `build_a0_report.py`, and `build_a1_t0_report.py`.
2. Use the A1-T2 loader to obtain the A0 HTML renderer and canonical A0
   Matplotlib style functions.
3. Hash every consumed file on first use.
4. For each `optimized-east*` and `retry-east*` directory, read `report.json`
   and record completed or failed/preflight-rejected status.
5. For a completed run, read:
   - `input_mapping.npz` with `R`, `Z`, and `polygon`;
   - matched `snapshots/control-step-*.npz` and
     `snapshots/perturbed-step-*.npz`.
6. For the long-RZ chain, follow only recorded continuation parents:
   - T1: `long-east12-waveform-probe` through the unique valid continuation
     child chain ending at `long-east12-waveform-segment5`;
   - A0: `long-east12-synchronous`;
   - A0 half-h: `long-east12-synchronous-halfh` through its final
     continuation.

## Snapshot decoding

For each snapshot:

1. Require scalar `step` and `time_s`.
2. Require `observed_BN`, `observed_density`, and `observed_velocity`.
3. Require `Nphi * 3 * Nc` lengths for `observed_BN` and
   `observed_velocity`, and `Nphi * Nc` for `observed_density`.
4. Require finite full sample arrays.
5. Reshape as `(Nphi, 3, Nc)` and `(Nphi, Nc)`; select physical phi index 0.
6. Require the selected real/imaginary representation to be real within
   `atol + rtol * max(abs(real))`.
7. Return real arrays in SI:
   - magnetic `[B_R, B_phi, B_Z]`;
   - ion velocity `[u_R, u_phi, u_Z]`;
   - density.

Control/perturbed pairs are matched by exact step key and time tolerance.
Reject missing, duplicate, or time-mismatched pairs. Delta is always
`perturbed - control`.

## Spatial cells

1. Validate finite active-center `R`, `Z` and a finite nondegenerate polygon.
2. Derive each cell's radial and axial half-extent from neighboring unique
   center spacing, using midpoint boundaries.
3. Construct one Cartesian quadrilateral per active center. Do not create
   cells from the rectangular bounding grid and do not triangulate.
4. Overlay the recorded wall polygon distinctly and keep equal coordinate
   aspect with SI-axis labels.

## Figures

For each completed short spatial probe:

1. Select the first and final matched saved snapshots.
2. Make two figures for first/final magnetic perturbation maps. Each
   component uses one shared symmetric limit across its two times.
3. Select the final matched snapshot.
4. For `B_R`, `B_phi`, `B_Z`, `u_R`, `u_phi`, `u_Z`, and ion density, make
   four figures with at most four panels each. Every quantity has a control
   section and `perturbed-control` delta section.
5. Use sequential normalization for full controls and centered symmetric
   normalization for deltas. Delta limits include the selected first/final
   evidence, never invented data.

For each final long-RZ method (T1, A0, A0 half-h):

1. Verify that the final control and perturbed snapshots are matched at the
   same actual time.
2. Verify all three methods use the same grid and mapping.
3. Build the same bounded control/delta quantity figures.

For the long chain:

1. Verify every continuation's recorded parent path, parent checkpoint hashes,
   start step/time, grid, and mapping.
2. Match T1, A0, and A0 half-h snapshots only at common saved times.
3. Choose the earliest, middle, and latest common times without interpolation.
4. At each time, plot T1 and A0 `B_phi` and `u_phi`, plus `T1-A0` and
   `T1-A0_halfh` differences on the same grid with shared signed limits.

## Report and provenance

1. Create a compact `data.json` with availability, selected times, figure
   limits, and precision notes.
2. Create an A0 manifest and call the unmodified canonical
   `render_report`.
3. Embed SVG data in the HTML image elements so the HTML opens offline.
   Keep the emitted PNG and SVG figures as separate artifacts.
4. Record exact input bytes/SHA256 and every output SHA256. Emit
   `SHA256SUMS` after all other final files.
5. Add English notes for failed/preflight-rejected probes, nonterminal bounded
   chains, FP64 cancellation limits, absence of pressure reconstruction,
   and explicit rejection of eigenmode/convergence claims.

## Failure behavior

Return a nonzero status with the exact missing file/key, shape, mismatch, or
unsafe path. Continue past a failed run by emitting an explicit no-data note.
Stop if the shared long-chain mapping cannot be verified.
