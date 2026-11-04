# Midpoint transaction failure controls

Stable step ID: `A0-VERIFY-TRANSACTION`.

Target: `tests/transaction.cpp`. This independent executable owns its allocation
fault injection; it does not modify the production allocator.

```text
construct an admissible closed reference state and accepted baseline
arm the backend synchronization hook to fail the next allocation
advance one full step
require the accepted result returns without allocation after commit
require accepted state step and cumulative interval count both advance once

for callback stage in [base, midpoint]:
  inject NumericalError from the verification-only source
  require NumericalLibrary failure with that exact stage
  require completed checks and partial arrays remain available
  require accepted state and cumulative ledger remain unchanged

for combination target in [midpoint, endpoint]:
  inject NumericalError from the corresponding backend AXPY
  require NumericalLibrary failure with the target stage
  require accepted state and ledger remain unchanged
```

The test also statically requires a nonthrowing `StepResult` move constructor.
Allocation injection is disarmed before assertions or output.
