# Coupled manufactured temporal verification

Stable step ID: `A0-VERIFY-COUPLED`.

This test uses the production midpoint transaction and stage diagnostics. It does
not add a source option to production case files or change the A0 physical model.

1. At fixed geometry and Fourier resolution prescribe an analytic-in-time state:
   densities with a nonzero toroidal harmonic and a small relative charge
   separation (electron harmonic is 0.9999 times ion), species momenta with
   electron amplitude 1.3 times the mass-scaled ion amplitude, independently
   varying electron perpendicular/parallel pressures, and a
   time-dependent harmonic toroidal magnetic field. The initial sinusoids vanish,
   leaving a neutral isotropic reference equilibrium. Density, momentum, each
   pressure and B all vary. The initial state is the closed reference equilibrium.
2. Differentiate every coefficient analytically. At each stage time compute the
   prescribed source S(t) = U_reference_dot(t) - RHS(U_reference(t)). S depends
   on the reference and time, never on the numerical error.
3. Use the same midpoint/acceptance/commit implementation as the unforced solver,
   adding S only to the two stage combinations. Physical stage diagnostics retain
   their unforced meaning. Source-aware interval diagnostics subtract the known
   midpoint density, momentum, pressure and magnetic sources, including their
   number, energy, longitudinal-field and topology contributions. Record sources
   separately. Production calls use a null source with unchanged behavior.
4. Compare every evolved component and the combined normalized error with the
   analytic reference at a common final time for 8, 16 and 32 steps. Require two
   orders in [1.7, 2.3], finite diagnostics and all hard gates. Print per-component
   errors, raw checks and source rates with build identity in JSON.
   Require nonzero EL, ET and physical magnetic RHS, and finite populated
   diagnostics even when their budgets are observational. Record actual accepted
   source quadrature separately, including units, stage identities and source
   hashes. The near-neutral reference keeps prescribed forcing well conditioned
   relative to its small target momentum; it does not test arbitrary forcing.
5. A deliberately omitted source term must fail the accuracy criterion; a wrong
   interval source must fail a hard conservation gate. A zero-source callback
   must reproduce ordinary advance exactly. The companion `source_controls`
   test compares each scalar source-accounting difference, including observational
   energy/momentum records, against an independently differentiated midpoint
   total with nonzero source signals. Sources must have compatible shape,
   physical Fourier symmetry, wall trace and zero magnetic divergence.

This is temporal verification of the coupled discrete equations under known
manufactured sources. It is not an unforced plasma or experimental validation.
