# Longitudinal verification regressions

## A0-PY-TEST-LONG-01: Independent physics and real transactions

Check the warm high-frequency root against the cold two-species limit and
the independent seven-variable linearized equations. Verify an initial
eigenmode's production derivative, so incorrect charge, pressure coefficients
or a dummy Poisson matrix fail. Run midpoint/RK4 through real transactions
at equal physical time; compare coarse amplification and fine refinement
against the declared independent analytic mode and polynomial.

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
