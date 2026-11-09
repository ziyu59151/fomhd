# Few-node WR memory-plan verification

```text
Use real bounded contexts and the existing checked entry-count arithmetic.
For multirate_wr, verify microstep, maximum outer-sweep and coarse-node scaling.
Include retained fine/base/coarse/alias frames, both path versions, partial
failure evidence, wall factors and backend-specific owning sample counts.
Reject invalid boolean, zero and unsupported node/iteration controls.
A temporary alias N must not resize the parent's retained history ownership.
Verify that the synchronous, predictor and waveform estimates remain unchanged.
```

These are conservative preflight estimates, not measured host/device peaks.
The actual campaign records allocations separately and enforces its resource cap.
