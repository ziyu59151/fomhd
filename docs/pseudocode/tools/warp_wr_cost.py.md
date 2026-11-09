# Matched-horizon WR cost harness

Source: `tools/warp_wr_cost.py`.

## Purpose

Measure real complete `advance` calls at one physical horizon. This is not a
kernel microbenchmark, optimization experiment, or scientific acceptance run.

```text
input:
  device cpu or cuda:0
  points: odd >=9; parent GPU run uses 33
  horizon_microsteps: 8 or 32
  wr_M: positive divisor of horizon, default horizon
  wr_ion_nodes: subset of {3,5}, default {3,5}
  methods: synchronous, waveform, multirate_wr
  wall_seconds: (0,900]

fixed physical fixture:
  h=1.9794733885623e-10 s
  amplitude=0.02, nonuniform=0.1
  mode1 plus high-mode16 sentinel=1e-4
  rtol=1e-10, max_iterations=12, tangent WR relaxation=1.0
  valid hard budget registry, including hard quasineutrality and optional
  observe theta-consistency
```

Method horizon:

```text
synchronous:  horizon steps at dt=h
waveform T1: one macro, dt=horizon*h, M=horizon
multirate WR: one macro, dt=horizon*h, M=wr_M, requested s values
```

Each method:

```text
build real periodic context and fixture through tools.warp_wr_periodic
run base stage.initialize once to capture immutable accepted seed
snapshot state and cumulative plus source identity/counter baseline
run one real synchronized warmup complete horizon from cloned seed
run three real synchronized repeats, each cloned from the same seed
reset/setup and timed physical horizon are separate fields
no ledger reset occurs inside a physical horizon
record input state/cumulative digests and fresh pre-run counters per run
```

Timing/resources:

```text
synchronize before and after complete horizon
include checks, metadata, aliases, trial work, commit and endpoint work
record warmup and each repeat run/setup/reset/total seconds
record sampled RSS peak plus CUDA pool used/reserved peaks
enforce host 16 GiB, device 8 GiB, and wall budget
stop on resource/time overspend
```

Counters:

```text
use actual tools.warp_wr_periodic _WorkSpy spies during every horizon
raw spy keys remain unchanged
role ion_transport_producer_calls <- full_ion_calls
  (massless.build_non_electric species0 transport producer)
role stage_evaluator_calls <- full_stage_calls
role full_rhs_finish_calls <- full_rhs_calls
  (raw observations include base/alias stages as counted)
report coarse-A producer, fine G/C callbacks, field, projection, warp dispatch,
and raw transaction/diagnostic counters
do not sum overlapping counters into a synthetic work total
```

Failures and claims:

```text
retain raw checks, code, message, work, metadata and traceback
hard gate/resource/wall failure stops the run
never emit a speedup claim or scientific acceptance claim
CUDA requires FOMHD_TEST_GPU=1 and SLURM_JOB_ID before context creation
refuse an existing output path
```
