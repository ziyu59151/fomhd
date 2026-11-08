# Matched multirate cost-reference regression mirror

Compare a complete lean synchronous RK4 window against the independent production
RK4 state on bounded periodic and wall fixtures. Count exactly four full hydro,
continuity and Lorentz evaluations per microstep, plus named endpoint work.
Preserve all four distinct stage identities across every macro window, including
both midpoint stages. Test matched cold/warm/repeat controls, unchanged physical
horizon, all saved candidate/reference endpoints and stages, outside-timer
checks, exact input hashing and CUDA rejection outside an owned allocation.
This reference is a research window without production transaction or ledger.

## Host-oracle phase postprocessing regression

A completed CUDA execution returns host arrays for comparison/output. The phase
helper must explicitly transfer these arrays to its context backend before
calling the backend-sensitive mode observer. A fake device backend rejects host
inputs and verifies both complex components, nonzero phase and amplitude errors;
this CPU test requires no CUDA context. The transfer is outside matched timing.
