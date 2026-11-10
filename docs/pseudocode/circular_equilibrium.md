# Circular equilibrium promotion pseudocode

## Purpose

Promote the reviewed continuous-equilibrium prototype into:

- `tools/circular_equilibrium.py`: the reusable SI continuous model;
- `tools/prepare_circular_equilibrium.py`: a runnable fixture CLI;
- `tests/warp/test_circular_equilibrium.py`: production import and behavior
  tests.

The overlay contains no fixture products. Generated NPZ/JSON fixtures remain
under the worker scratch tree until a parent explicitly promotes them.

## Library

`tools/circular_equilibrium.py` is the reviewed
`scratch/continuous_equilibrium.py` implementation without physical changes.
Its public interface is:

```text
build_gs_equilibrium(*, R0, a, F0, n0, p_edge, pprime, alpha, degree,
                     collocation_multiplier=60) -> GSModel
sample_gs(model, R, Z) -> dict of SI NumPy arrays
gs_residual_maxima(model) -> dimensionless and raw residual maxima
gs_residual_breakdown(model) -> independent interior/near-wall residual sets
gs_wall_residual_maxima(model, count=257) -> wall flux/B.n/E.t residuals
magnetic_axis(model) -> [R_axis, Z_axis]
verify_nested_flux(model, ...) -> sampled ray-crossing topology
gs_convergence_study(...) -> degree/collocation enrichment rows
sample_vacuum(R, Z, *, F0, n0, pressure, alpha=0.6) -> SI vacuum state
vacuum_residual_maxima(state) -> continuous-RHS maxima
vacuum_wall_residual_maxima(state, wall_RZ) -> wall residuals
cylindrical_compressional_reference(...) -> n=0 J1-root reference
```

The source convention remains:

```text
B_R=-psi_Z/R, B_phi=F/R, B_Z=psi_R/R
Delta*psi=psi_RR-psi_R/R+psi_ZZ=-mu0 R^2 p'
```

The polynomial boundary factor enforces `psi=0` on the fixed circle. Residual
values are continuous-model diagnostics and do not claim agreement with the
production Yee/CT state or RHS.

## CLI

`tools/prepare_circular_equilibrium.py` begins with the standard source-root
setup:

```text
SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SOURCE_ROOT/"src"), str(SOURCE_ROOT)]
from tools.circular_equilibrium import ...
```

No checkout path is hard-coded. The optional design tree is inspected only
when `FOMHD_DESIGN_ROOT` names an existing directory; it is never imported and
is not required for runtime.

The CLI has `build` and `vacuum` modes. It writes caller-selected output paths,
records SI units/layout/provenance, and returns nonzero on a failed check.
Default fixture generation is never redirected into `SOURCE_ROOT`.

## Tests

`tests/warp/test_circular_equilibrium.py` prepends the overlay/production root
to `sys.path`, then imports:

```text
from tools.circular_equilibrium import ...
from tools.prepare_circular_equilibrium import ...
```

Tests cover:

1. the signed constant-F GS system;
2. zero vacuum RHS and arbitrary-wall boundary conditions;
3. circular-wall, magnetic-axis, nested-flux, and pressure/current checks;
4. degree/collocation enrichment;
5. the first-positive-`J1` n=0 cylinder reference;
6. CLI output schema/provenance with output forced under a test scratch path.

Test scratch directories stay below the supplied pytest base directory.

## Vacuum probe boundary

A bounded production probe may use `tools.warp_tokamak_cae` only to construct:

```text
physical_parameters(K, 2e19, 2.0, 0.5, 1000.0)
circle cell-center grids for N=8,12,16
build_geometry(...)
build_operators(..., {"include_finite_mass": False})
core.allocate_state(...)
stage.evaluate_stage(...)
```

The probe uses true `t=0`, writes `B_phi=F/Rbar_v` on the production vertex
block, retains raw continuous vertex samples, and evaluates the actual
initial stage/full RHS. It reports wall/forced rows and fixed physical
`0.1a` band versus interior diagnostics. It must not subtract residual RHS,
mutate production operators, run an evolution loop, or use GPU.
