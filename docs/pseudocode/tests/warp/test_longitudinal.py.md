# Longitudinal verification regressions

## A0-PY-TEST-LONG-01: Independent physics and real transactions

Check the warm high-frequency root against the cold two-species limit and
the independent seven-variable linearized equations. Verify an initial
eigenmode's production derivative, so incorrect charge, pressure coefficients
or a dummy Poisson matrix fail. Run midpoint/RK4 through real transactions
at equal physical time; compare coarse amplification and fine refinement
against the declared independent analytic mode and polynomial.

Construct several supported odd point counts and require the same independent
continuous root plus point-specific adapter identities. Reject even, below-9
and above-65 point counts before runtime work.

## A0-PY-TEST-LONG-02: Evidence boundaries

Reject invalid integrators, amplitudes and step counts before evolution.
Refuse an existing output directory without changing its contents. Export
actual initial/interval hard diagnostics, requested/accepted steps, warnings
and numeric histories, without silently substituting a CPU backend.
Reject an empty scan before creating output. Open exported histories with
pickle disabled and verify that initial hard records remain in the JSON.
Replace only the test Poisson block with an inconsistent identity operator
to trigger a real rejected step for midpoint/RK4. Require the failure's
structured timestep warning to remain alongside its diagnostic records,
with zero accepted steps and one reported warning.

Exercise cycle and step upper bounds, nonempty unique integrator selection,
and explicit dt_guard selection. With `error`, require a real coarse
production step rejection and no accepted-history growth; with `warn`, retain
the same warning record but permit the controlled legacy step. Verify CLI
defaults select only RK4 with `error`, and explicit selection is serialized.
Every invalid selection must leave the requested output path absent.

## A0-PY-TEST-LONG-03: History-only fit and provenance

Fit synthetic complex signals with exact known signed frequency and growth
or damping, using at least four cycles and 32 samples per cycle. Require
recovery of frequency, rate, residual and coherence without passing any theory
frequency to the fitter. Use a negative signed frequency to distinguish sign.
Require unavailable reasons for fewer than four measured cycles, fewer than 32
samples per cycle, non-finite/zero samples and non-increasing time. A coarse
real transaction may be unavailable while retaining all endpoint evidence.

Run one short RK4 control slightly beyond four cycles at a step count that
also passes the production samples-per-cycle guard, and require a usable
history fit. Compare measured, analytic and RK4-predicted angular frequencies
as separate report fields with units. Preserve the endpoint polynomial error
and half-amplitude controls.

Simulate today's package-root and future `src/fomhd` layouts. Require both
roots to resolve from the supplied imported package path and matching project
metadata, and require a decoy legacy `fomhd/` tree to be ignored for a
`src/fomhd` import. Missing lock/metadata and loaded modules outside the
imported package are errors, not fallback conditions.
