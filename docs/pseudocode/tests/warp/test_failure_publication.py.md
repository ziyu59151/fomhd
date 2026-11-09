# Rejected WR failure publication tests

```text
Build a real closed-wall A1-WR case with:
  waveform_rtol=1e-30
  waveform_max_iterations=1

Initialize accepted step 0, retain state/cache/cumulative identities, then run
the complete advance and require actual Failure("MultirateConvergence").

Before publication:
  require object keys fine_identities/fine_records and multirate all-record
    tuples remain on the same in-memory Failure
  require partial numerical arrays remain on the Failure
  require accepted state/cache and cumulative identity are unchanged

Call real io.Run.failure:
  require partial_published=True
  load the NPZ through io.read_npz and raw np.load(allow_pickle=False)
  require all published dtypes are safe and numerical arrays are unchanged
  require structured keys are absent from NPZ
  require JSON lists every omitted key with type/reason
  require JSON coupling_metadata contains nonempty actual residual history,
    final residual above tolerance, operation counts and source/attempted stage
    data
  require JSON mixed_diagnostic_work contains accumulator/checker null or actual
    count fields without object serialization
  require JSON stage identity equals the original Failure stage

Reuse existing malformed/nonfinite NPZ and ordinary failure-record tests.
```
