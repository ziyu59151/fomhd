# Manufactured-source accounting controls

Stable step ID: `A0-VERIFY-ACCOUNTING`.

1. Differentiate the production diagnostic totals independently by centered
   differences of two admissible states U +/- h S. Use a charged background,
   nonzero momentum, pressure rates, and a divergence-free harmonic magnetic
   rate. Combine sine and cosine toroidal-momentum harmonics with a mean
   toroidal-momentum source so Cartesian x/y and angular source rates are all
   nonzero. Require every independently differentiated number, energy, flux and
   momentum component to have a signal above the declared reference floor.
   Compare each derivative with `source_rates` without using that implementation
   to compute the expected derivative.
2. Run ordinary advancement and the shared source-enabled advancement with
   an exactly zero callback from identical contexts. Require identical state,
   field cache and physical cumulative energy ledgers.
3. Advance the charged moving fixture with the same prescribed source; retain
   the actual midpoint. Source-aware conservation must accept. Rechecking the
   same interval without that source must reject a hard balance. Retain the
   unforced check records from the rejection and compare every scalar source
   accounting difference with an independent midpoint centered difference:

   ```text
   expected_rate = (diagnostic(U_mid+h*S)-diagnostic(U_mid-h*S))/(2*h)
   observed_integral = omitted_source_check.raw - source_aware_check.raw
   require observed_integral agrees with dt*expected_rate
   ```

   Cover both species' number and kinetic energy, charge, all internal/field and
   total energies, topology, all Cartesian momentum components, and angular
   momentum. Apply these test assertions even to observational production
   budgets. Require the physical `field_exchange` record to be unchanged.
   Record independent expected rates, observed integrals, relative errors and
   units in JSON. All production budgets stay as declared.

The controls must reject deletion of the total-energy source subtraction and
zeroed Cartesian-x/angular source rates. Only test fixtures use these callbacks;
they do not add production input options.
