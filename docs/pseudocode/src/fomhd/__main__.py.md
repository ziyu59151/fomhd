# Python module command line

## A0-PY-CLI-01 - Strict CLI and transaction output

```text
main(argv):
  accept exactly:
    --help, --version
    check CASE [--output DIR]
    validate CASE [--output DIR]
    run CASE [--output DIR] [--restart CHECKPOINT]
    restart CHECKPOINT CASE [--output DIR]
    inspect RUN_DIR
  reject duplicate/unknown/incomplete options

  install SIGINT/SIGTERM flags before loading input
  runtime_phase=pre_commit_input
  load strict case
  construct explicit Backend(device), Context and Run writer for either
    omitted electron_inertia=0 uses the reviewed massless branch;
    explicit electron_inertia=1 uses the finite branch
  runtime_phase=numerical_trial during advance
  runtime_phase=post_commit_output after accepted advance
  validate/check writes accepted initial diagnostics, snapshot, checkpoint,
    status=validated_initial_state, then exits 0
  run/restart initializes/restores accepted state then loops
  process signal only between accepted steps; checkpoint, status stopped, exit 130
  write one accepted-step history row, requested snapshot/checkpoint
  publish accepted predictor metadata containing coupling, M, stage counts,
    weights, retained-entry count, preflight estimate and endpoint corrections
  mark completed only after final output succeeds
  on exception write failure receipt with actual phase and last accepted identity
  persist accepted time-step guard reports to time_estimates.jsonl

Exit codes: 0 success; 2 input/schema; 3 backend/resource;
4 numerical/model gate; 5 I/O; 6 internal; 130 safe stop.
```

## A1-PY-CLI-02 - Accepted predictor evidence

After an accepted predictor macro, publish its actual stage counts, microstep
count, physical quadrature weights, storage estimate and endpoint tangent
correction in run metadata. Rejected windows publish failure evidence and do
not produce an accepted-step record.
