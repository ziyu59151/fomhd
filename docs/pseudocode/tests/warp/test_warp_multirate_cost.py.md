# Matched multirate cost-reference regression mirror

Compare a complete lean synchronous RK4 window against the independent production
RK4 state on bounded periodic and wall fixtures. Count exactly four full hydro,
continuity and Lorentz evaluations per microstep, plus named endpoint work.
Preserve all four distinct stage identities across every macro window, including
both midpoint stages. Test matched cold/warm/repeat controls, unchanged physical
horizon, all saved candidate/reference endpoints and stages, outside-timer
checks, exact input hashing and CUDA rejection outside an owned allocation.
This reference is a research window without production transaction or ledger.
