# Test `warp_t0_controlled.py` pseudocode

Source proposal: `tests/warp/test_warp_t0_controlled.py`.

## A1-PY-TEST-T0-CONTROLLED-01 - Controlled evidence regressions

```text
Controls and outputs:
  deny CUDA, nondivisible H, invalid M, existing output;
  require no output mutation on validation failure.
  require same H for (2048, M=256) and (4096, M=512).

Harmonics and spectra:
  actual DZ eigenvalues recover m*k1 for m=1..4;
  constant m=0 projection uses 1/N, while positive harmonics use 2/N;
  production transverse block matches the independent cold Hall matrix;
  micro Z/Z block matches the full production block;
  full spectrum retains the additional pressure modes.

Independent time algebra:
  scalar exp and one RK4 step match hand-derived values;
  linear T0 map matches known scalar/block systems;
  a production one-macro small-amplitude finite difference agrees with
  the 9x9 linear T0 map.

Seeds and failure transaction:
  target and mixed projections recover exactly 2e-6 and 2e-8;
  analytic references use cold eigenvalues even when production eigenvalues
    deliberately differ; mixed m1/m4 initial REF errors are roundoff;
  unwrapped phase handles rotations beyond one principal interval;
  injected rejection after one accepted macro is UNDETERMINED,
  retains target/error and absolute high-mode prefix histories, streams gzip
  checks, and writes no invented endpoint;
  initialization rejection preserves correctly shaped empty histories;
  success/rejection preserve the accepted cumulative ledger by value.
  initialization rejection still writes case/failure artifacts and empty
    (0,4) modal histories; success/rejection retain the cumulative ledger.

Matched and whole-map evidence:
  numerical source, seed, physical/discrete or horizon mismatch rejects the
    reference before advancement;
  exact shared times align macro output to micro-cadence references;
  [1,2,1] matched gain retains transient maximum separately from final gain;
  source-hash, seed-vector, physical-parameter, and horizon-coverage mismatches
    are rejected before advancement;
  a matched [1,2,1] transient reports maximum and final amplification separately;
  G^n, spectral-radius powers, and actual-seed gains are computed at fixed 2T;
  m=0 missing longitudinal-momentum evidence is masked, not called an
    eigenvalue measurement.

Frozen-ion micro control:
  calls no time.advance; P remains exactly fixed;
  workspace declares micro_steps=1 and dt=h, nsteps=observed microsteps;
  only m4 Z is seeded; production stage hard checks stay active;
  final Z agrees with frozen-block exponential and RK4 references.

Keep tests bounded: points=9 and at most two microsteps per macro.
```
