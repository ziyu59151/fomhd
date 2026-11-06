# `cases/massless-example.toml` documentation mirror

The example is a complete strict A0 v1 configuration for the default
quasineutral massless-electron model:

```text
species.electron_inertia = 0
physical electron mass me > 0
time.integrator = midpoint or rk4
time.dt_guard = error or warn
```

Every assignment line carries an English inline comment with its unit,
allowed choice, and purpose. The example uses the exact massless budget
registry and no additional schema keys. Its geometry and initial-state paths
are local fixture references and must be supplied by the caller.

Finite electron inertia requires an explicit `electron_inertia = 1`, finite
registry, and its own reviewed inputs. The example's local closed-domain scope
does not validate a global Tokamak equilibrium, experimental profile, or
production performance.
