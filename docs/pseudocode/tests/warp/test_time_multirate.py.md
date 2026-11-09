# Multirate transaction integration tests

```text
controls:
  accept multirate_wr M,s,rtol,cap,initializer,relaxation
  reject invalid/new controls on legacy coupling before initialization

one macro:
  initialize a real periodic massless state
  advance once
  require one accepted interval, standard endpoint tag n+1/time t+H,
  matching cache generation, all12 ledger and6 boundary integrals,
  final s coarse/4M fine adapter records, no fine full-ion calls
  metadata contains the actual nonempty residual_history and final scalar
  residual/endpoint summaries

module identity:
  import the canonical fomhd.time_multirate from sys.modules
  require its file to be the expected transaction source
  require fomhd.time_multirate package dispatch to reference that same module
  so monkeypatched evaluate_stage/_metadata are the calls reached by the real
  fomhd.time advance path
  snapshot/restore both sys.modules and the package attribute as one pair at
  module-fixture teardown, preserving a consistent prior import state

multiple macros:
  advance twice and require monotone accepted clocks, distinct trials, and
  exactly two cumulative intervals

closed wall:
  repeat one macro on the real K=1 wall context

failure:
  inject fine field, coarse full stage, and endpoint full stage failures
  inject first accepted-left base failure and assert no metadata masking
  inject accepted metadata failure before commit and assert no physical commit
  require that failed post-solve metadata retains the actual nonempty history
  force nonconvergence and duplicate trial accumulation
  require state/cache/ledger rollback and retained actual trial evidence
  reject duplicate accumulation and endpoint cache/generation mismatch
  partial nonconvergence evidence includes completed residual_history; first
  base failure reports residual history as explicitly unavailable

config/restart:
  require semantic hash changes for s/initializer/relaxation
  checkpoint one macro and resume the second
  compare uninterrupted and resumed states, all12 ledger and6 boundary fields
  reject a changed scheme at read_restart

CLI:
  run one macro through __main__, complete output, and read its checkpoint

cost:
  at fixed H,s, require s-node coarse work per outer sweep independent of M,
  with base and accepted endpoint charged separately
  report epsilon_K separately from raw gates and expose both diagnostic-work
  objects plus explicit full/fine power-ledger accounting

nonzero start:
  with start_time=1.0 and declared H=2e-12, accept 32 ordinary macros
  require scheduler-exact clocks, H-based weights, and a nonzero-start resume

memory:
  A0/predictor/waveform estimates retain historical literals
  WR estimate grows with M, max_iterations, s, and alias padded N
  wall retention grows with M/max_iterations/s
  bool and out-of-domain controls reject before allocation
```

The declared `H` is authoritative. Clock subtraction cancellation is a ledger
repair-3 contract, not a widened transaction tolerance.
