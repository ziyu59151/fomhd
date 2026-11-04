# A0 Warp MMS verification CLI

## A0-PY-MMS-CLI-01 - Procedure
Stable documentation entry: `A0-TOOL-WARP-MMS`.

The executable is `tools/warp_mms.py`. The trajectory, source, temporal
verification, spatial verification, raw-check serializer, and failure-array
codec are defined in
[`../tests/warp/mms_fixture.py.md`](../tests/warp/mms_fixture.py.md). This document
records only the CLI orchestration so the diagnostics algorithm has one
authoritative narrative.

```text
main(arguments):
  output = CreateNewOrEmptyDirectory(arguments.output)
  identity = HashCompleteRuntimeAndHarness(
    runtime_root=DeriveFromImportedFomhdPackage(),
    harness_root=ExecutedHarnessFiles())
  snapshot = ArchiveBoundedSource(
    output/source_snapshot,
    all current runtime fomhd/*.py,
    pyproject.toml,
    uv.lock,
    MMS fixture, test, CLI and pseudocode)
  require every snapshot destination hash to match SOURCE_MANIFEST.json

  device = RequireCPUOrOwnedSlurmCUDA(arguments.device)
  for requested integrator:
    result = RunVerification(
      integrator,
      raw_root=output/raw,
      spatial=(device==cpu and spatial requested))
    result records production guard reports, raw checks JSONL,
    accumulated/source ledgers and source omissions separately

  negative = RunOmittedIonPressureControl(
    raw_root=output/raw/negative)
  require negative physical gates pass and analytic trajectory error fails
  retain the negative control's eight full actual production reports
  aggregate unique sidecars: 73,820 normal + 7,592 negative = 81,412

  status = PASS only if every requested normal result passed and the
           omitted-source negative control was detected
  write strict results.json with source_snapshot manifest hash

on Failure:
  if Failure.partial_arrays is nonempty:
    write failure/cli_failure.npz with NaN and infinity preserved
  if Failure.checks is nonempty:
    write failure/cli_failure.jsonl with serialized raw checks
  retain Failure code, stage, partial-array names, array descriptors and hash
  write FAIL results.json

No solver source under scratch/proposal is imported or delivered. CUDA without
an existing SLURM_JOB_ID is refused before backend construction.
```

An explicitly supplied runtime root is accepted only when it resolves to the
same checkout as the imported `fomhd` package; otherwise provenance stops with
`RuntimeMismatch`. A frozen checkout therefore archives and executes the same
module bytes.

Raw diagnostics are written to:

```text
output/raw/<integrator>-<steps>/checks.jsonl
output/raw/<integrator>-<steps>/ledgers.json
output/raw/negative/checks.jsonl
output/raw/negative/ledgers.json
```

The source snapshot is evidence, not a second solver implementation. The stable
deliverable whitelist contains only the harness CLI, fixture, tests, and
pseudocode.
