# Massless GPU workflow acceptance pseudocode

## A0-PY-GPU-MASSLESS-01: Execution gate

```text
GPU_REQUIRED =
  skip unless FOMHD_TEST_GPU=1

gpu_ready():
  require nonempty SLURM_JOB_ID before any cuda Backend construction
  return "cuda:0"
```

The default CPU suite collects these cases and marks GPU cases skipped. It does
not report a GPU pass.

## A0-PY-GPU-MASSLESS-02: Identical host fixture

```text
make_fixture(integrator):
  one physical Parameters set: electron_inertia=0, dt=2e-13,
    dt_guard=warn, physical mass metadata and massless budgets
  one 6x6 rectangle geometry
  build CPU and cuda:0 Backends/operators from that geometry
  build the nonzero compatible-current uniform massless state on CPU
  extract every B/fluid array explicitly to host
  construct CPU and cuda states from those exact host arrays
  verify cuda input arrays equal the host fixture before stage execution
```

The CPU state is also reconstructed from the saved host arrays so subsequent
comparisons use the same input bytes.

## A0-PY-GPU-MASSLESS-03: Stage and time parity

```text
run_one_step(context, state, integrator):
  prohibit Backend.to_host while initialize and advance execute
  initialize: exactly one accepted massless initial stage
  capture initial E, E0, current, electron flux and padded ue
  capture all independent RHS arrays
  advance one accepted step
  record actual massless.evaluate_stage count:
    midpoint=4   (base, midpoint, alias, endpoint)
    rk4=7        (base, midpoint+alias, midpoint_2+alias, station, endpoint)
  restore to_host only for explicit comparison

compare CPU reference to cuda:0:
  use workflow_fixture.assert_dimensional
  compare E/E0 against electric and current references
  compare current, retained/sampled electron flux and padded ue
  compare ion/electron n,mom,perp,parallel and B RHS
  compare accepted B and independent endpoint state
  require equal step, interval count and cumulative endpoints
  require every retained array owned by its selected backend/device
```

No tolerance is estimated from the observed CPU/GPU difference.

## A0-PY-GPU-MASSLESS-04: Rejected candidate evidence

Construct the physical initial state with the shared CPU fixture and upload
its arrays explicitly, as in the parity test. The host-only nullspace helper
must not implicitly convert CUDA operators to NumPy during setup.

```text
negative_candidate_on_cuda():
  create a wall-compatible real random ion momentum from T_full nullspace
  initialize the genuine massless fixture
  choose a warned dt whose predicted midpoint ion density is negative
  snapshot accepted state and cumulative ledger
  prohibit to_host during advance
  expect Failure("Domain")
  require midpoint step/trial identity
  require candidate_B/n/mom/p/p_parallel evidence on cuda
  require accepted state and ledger unchanged
```

Host transfer is restored only to inspect final evidence.

## A0-PY-GPU-MASSLESS-05: CLI run and restart

```text
gpu_cli_case():
  create the same strict A0 input on CPU
  change only solver.backend from cpu to cuda
  update the source manifest case hash

cli_acceptance():
  count massless.evaluate_stage calls around the whole CLI workflow
  run one accepted step
  restart from that checkpoint for a second accepted step
  require CUDA backend and massless model/applicability metadata
  require cumulative baseline step zero and two accepted intervals
  require canonical electron density and compatibility momentum
  load a finite-model case and require cross-model restart rejection
    before decode_state
```

Expected GPU stage-evaluation budget:

```text
midpoint parity             5 (one initial stage and four advance stages)
rk4 parity                  8 (one initial stage and seven advance stages)
negative candidate          2 (initial + base; canonicalization rejects)
CLI run plus restart       10 (two initial stages and two midpoint advances)
total                      25
```

The count excludes diagnostic reductions and host-side fixture construction.
