# A0 case fixtures

This directory contains the small reproducible fixtures used by the A0 tests.
The generator is the native `fomhd_make_case` executable in `tools/make_case.cpp`;
it writes a new case directory containing `case.toml`, `geometry.npz`,
`initial.npz`, and `source_manifest.json`.

Generate a CPU reference case in a new directory:

```bash
build/cpu-release/fomhd_make_case /absolute/path/to/a0-case
build/cpu-release/fomhd validate /absolute/path/to/a0-case/case.toml \
  --output /absolute/path/to/a0-validate-run
```

The generated closed staircase wall is an A0 reference fixture. It is suitable
for operator, stage, restart, and diagnostic verification. It is not an
experimental equilibrium and must not be reported as a production tokamak
validation case. External G-EQDSK or VEQDB inputs require an explicit parser,
unit audit, compatible two-fluid state, and a separate validation record.
