# Test massless support

Source: `tests/warp/test_massless_support.py`.

## A0-PY-TEST-TEST-MASSLESS-SUPPORT-01 - Procedure and assertions

```text
Construct model=massless test parameters with positive physical mass metadata
and dimensional budgets; optionally make selected diagnostics observational.
Build real closed-grid operators on the requested explicit backend.
For each retained mode construct signed full-wall extraction.
Use nullspaces to build nontrivial DBB=0 and wall-current-compatible B;
separately build a current-free equilibrium and a deliberate wall-current
violation. Require nonzero target fields instead of zero-vector shortcuts.
Construct positive uniform/equilibrium states, stage tags and weighted products.
Provide a compatibility-electron-momentum perturbation that must not alter
physical massless production. Expose configured budget names for registry tests.
```

Covered top-level functions and fixture classes:

- `parameters`
- `context`
- `_mode_block`
- `_wall_extraction`
- `compatible_magnetic_field`
- `current_free_magnetic_field`
- `wall_incompatible_current_field`
- `uniform_state`
- `equilibrium_state`
- `perturb_only_compatibility_momentum`
- `stage_tag`
- `weighted_dot`
- `budget_names`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.

SI reference contract: current and charge references are already densities;
use Jref for wall/current definitions, Jref/L for current divergence, rhoref
for quasineutrality, Eref/L**2 for SCdagCE, and P*u*L**2 for advective
power. Interval charge is rhoref*L**3 and interval momentum is mi*ni*u*L**3.
Relative tolerances and hard/soft flags are independent of these units.
