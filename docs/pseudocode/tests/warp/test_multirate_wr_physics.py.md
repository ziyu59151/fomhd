# Adapter test pseudocode

```text
periodic field-only:
  pack physical fixture state; spy massless.build_non_electric
  invoke one middle field callback
  require no full ion transport, one base + one alias field evaluation

periodic split:
  invoke one coarse callback
  require A.density=0, C.pressure=0, A+C=full RHS, actual full count=1
  retain and check the existing transport_projection value/reaction/multiplier

periodic window:
  solve small WR window and independent full-RHS RK4 window
  require convergence, accepted-old nodes, s surviving nodes, no mutation,
  exactly s final accepted coarse records and 4M final accepted fine records
  keep all callback records separately and expose honest trial endpoint flags

counts:
  hold H and s while M doubles; require coarse count independent of M after
  accounting for actual sweeps and endpoint reconstruction

identity/cache:
  require unique generations, distinct same-time RK middle tags, shared actual
  wall-cache identity, exact accepted-left base clocks
  compute electric exchange from frame.ion_em_terms.electric, never a second
  wall projection or reconstructed force

failure:
  inject a fine-stage exception; require retained counters/partial arrays,
  unchanged caller state and ledger sentinel

closed wall:
  repeat field-only and coarse split smoke on the real K=1 wall context
```
