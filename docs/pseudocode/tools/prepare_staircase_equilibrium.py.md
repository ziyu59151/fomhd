# `tools/prepare_staircase_equilibrium.py` implementation pseudocode

## EQ-STAGE-01 Inputs and ownership

1. Read a frozen `circular_equilibrium.GSModel` and create a formal A0
   staircase case under this audit directory only.
2. Use `K=0`, `Nphi=5`, `electron_inertia=0`, CPU FP64 and no time steps.
3. Build the old structured geometry from the same circle used by the
   continuous model. Keep the actual grid spacing in metadata.
4. Do not import or invoke any polygon/intercept runtime.
5. Keep the promoted helper source-root-independent. Default to the parent
   directory of the promoted `tools/` file; accept an explicit `source_root`
   argument for scratch execution. Do not embed an active checkout path.

## EQ-STAGE-02 Formal files

1. Write `geometry.npz` with R/Z nodes and the circle LCFS.
2. Sample continuous ion density/pressures on active cell centers and B on
   the old R-face, vertex and Z-face tables through the existing physical
   layout.
3. Encode the compact state with `fomhd.io.encode_state` and write
   `initial.npz` with the formal fixed state keys.
4. Apply only the documented input compatibility projection for `div(B)=0`
   and the existing wall-current constraint; record it as projection change.
5. Write a schema-valid case TOML, geometry, initial state and source
   manifest. Compute hashes only after immutable bytes are final.

## EQ-STAGE-03 Formal mapping

1. Call `fomhd.io.load_case`.
2. Call `fomhd.io.load_initial`.
3. Call `fomhd.stage.initialize` and `fomhd.stage.evaluate_stage` without
   advancing time.
4. If a hard check fails earlier, retain its exact class/message/checks and all
   available mapped arrays.
5. Otherwise save full initial RHS, canonical/projected changes, B/n/
   momentum/pressure fields, and wall-band/interior norms.

## EQ-STAGE-04 Interpretation limits

This mapping tests the old formal staircase initialization path. It does not
validate an intercepted boundary, approve equilibrium evolution, or claim that
the discretely projected state remains static.

The bounded helper only supports isotropic electron pressure in its force/Ohm
comparison. This restriction is explicit; no generic anisotropic tensor
auditor is added here.
