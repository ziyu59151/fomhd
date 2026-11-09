# Periodic campaign runner tests

```text
2-macro multirate CPU campaign:
  run uninterrupted
  run one-macro prefix and resume in a new directory
  compare checkpoint P/Z and complete io._cumulative_json exactly

negative controls:
  existing output is refused
  resumed foreign coupling/time identity is refused
  injected second-macro rejection preserves first accepted checkpoint/history

fixed-H refinement:
  compare M=1,h and M=2,h/2 actual-producer spies
  full-ion calls do not scale with M
  fine field calls double

actual 33-point M=32 production rejection:
  use default amplitude/nonuniform/sentinel and real new-WR producer
  preserve the original HardGate Failure
  write run.json failed/failure.json nonfinite-safe evidence
  do not leave run.json running or create a fake checkpoint
```
