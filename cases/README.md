# A0 case generation

The active benchmark generator is the Python implementation in
`tools/warp_benchmark.py`. It uses the production `fomhd.geometry`,
`fomhd.operators`, `fomhd.stage`, and `fomhd.io` functions to create the
controlled closed-reference rectangle, geometry and initial NPZ inputs, strict
case configuration, and source manifest.

Run a bounded CPU scan into a new directory:

```bash
OPENBLAS_NUM_THREADS=1 uv run python tools/warp_benchmark.py scan \
  --output /home/lxzeng/runs/fomhd/a0-benchmark__20261105T120000Z \
  --device cpu --grid 4 --K 1 --cycles 2 --nsteps 2 \
  --integrators midpoint,rk4 --time-budget-s 180
```

There is no native `fomhd_make_case` lookup, `--case-builder` argument, or
required native build. Exported native oracle arrays are optional parity inputs
selected with `FOMHD_NATIVE_ORACLE_DIR`; they are not used to generate benchmark
cases and are never a runtime dependency.

The generated closed staircase-wall domain is an A0 numerical-reference
fixture. It is suitable for operator, stage, restart, and diagnostic
verification, but it is not an experimental equilibrium or a production
tokamak validation case. External G-EQDSK or VEQDB inputs require an explicit
parser, unit audit, compatible two-fluid state, and a separate validation
record.
