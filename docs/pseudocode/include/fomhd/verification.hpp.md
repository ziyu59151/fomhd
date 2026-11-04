# Test-only prescribed source interface

## A0-VERIFY-SOURCE-01 — Ownership and units

`verification::Source(state, tag, context) -> owning State` returns SI rates,
with the ordinary retained field layout and physical units divided by seconds.
The reference callback may inspect its arguments; it must not mutate the accepted
state, operators, physical fluxes or ledgers. A manufactured reference depends
only on analytic data and the stage time, never on the numerical error.

`verification::advance` shares the production midpoint transaction. The ordinary
`fomhd::advance` passes no source. No case-file option, physical model term or
manufactured-state restart is introduced. Negative density/pressure rates are
valid; their evolved physical states still must pass positivity checks.

`SourceRates` owns number/energy/flux/momentum rates and the longitudinal electric
field rate. Its values are read-only diagnostics of the prescribed source. The
test harness accumulates them only after the step is accepted, separately from
the production physical cumulative ledger.
