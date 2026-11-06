# Physical simulation prototype workflow

Status: active project guidance, confirmed by the user on 5–6 November 2026.
Authority: design decision D062 and the user's A0 benchmark instructions.
Scope: development priorities and evidence interpretation; this does not change
the equations, acceptance thresholds, or solver constraints.

## Build a useful prototype, then learn from physical comparisons

The first milestone is a reproducible solver that can calculate an interpretable
physical response. Basic verification makes that comparison meaningful; it is
not an obligation to eliminate every residual before a realistic calculation.
Use comparisons with independent theory, other codes, and suitable experimental
observations to discover which prototype limitations materially affect the
research question. Record limitations that do not yet justify a repair.

Implement small modules and check their interfaces, but also test the assembled
calculation: passing modules do not establish correct coupled physics. Keep
algorithm design, executable-level pseudocode, source, and diagnostics aligned
when a repair changes behavior. Avoid expanding an A0 task into an unrequested
physical model, boundary upgrade, implicit method, or optimization project.

## Minimum verification before interpreting a physical result

Check units and normalization, model routing, initial and boundary conditions,
flux/current ownership, accepted-step transactions, and restart identity. Use
appropriate analytic, manufactured-solution, time-step, amplitude, and backend
controls. State their tested scope rather than treating them as a universal
certificate. A nonfinite or corrupted state, nonpositive required density or
pressure, failed solve, or necessary model-constraint violation still stops a run.

Keep strict verification cases strict. A new exploratory physical case may use
the explicitly documented `observe` policy for energy, pressure-power pairing,
and global momentum. Preserve the measured values, original tolerances, and
exceedances. Never turn an old failure into a pass by changing its label.

## Decide which discrepancies need work now

Compare a defect with the physical quantity it can affect: characteristic
species impulses, perturbation kinetic/magnetic energy, phase accumulated over
the observation window, or the target frequency and polarization. A small
absolute residual alone neither proves irrelevance nor establishes a serious
physical problem. Large background energy must not hide a small wave signal.

For a perturbed run and an otherwise matched unperturbed control, retain both
raw budgets and their difference. Strong cancellation supports a common
background contribution; it does not prove that all remaining response dynamics
are accurate. Record that distinction and defer background-budget work when the
target response and independent comparisons support progress.

Prioritize a reproducible physical disagreement, substantial mode-structure
change, energy discrepancy on the perturbation scale, or instability. Frequency
differences of roughly ten percent are a useful review trigger for the present
A0 cases, not a universal scientific acceptance threshold. Investigate only as
far as needed to make the comparison interpretable.

## Preserve useful clues without making them automatic blockers

An issue record contains the source/input hashes, raw run or failure, units,
normalization, measured size, competing explanations, and the next observation
that would justify reopening it. Separate a hypothesis from a demonstrated
cause. Low-order wall/interior coupling, background mapping, and incomplete
spatial resolution remain candidates until a controlled comparison separates
them. The original evidence stays available after a presentation is shortened.

For A0, the fixed-domain momentum defect, circle 20-by-20 pressure-transport
observation, and common-background energy discrepancy remain such records in
[A0_KNOWN_ISSUES.md](A0_KNOWN_ISSUES.md). They do not automatically block A1.

## Give each result an unmistakable identity

- **NL — FOMHD time evolution:** accepted nonlinear time integration. Label
  backend, grid, toroidal harmonic, time window, matched control, and failures.
- **LIN — FOMHD discrete Jacobian:** a local linearization of the production
  nonlinear right-hand side. It tests the discrete system, not an independent
  physical model. A frozen background with a nonzero right-hand side is not a
  stationary equilibrium, and its eigenvector is not a time-evolved snapshot.
- **REF — independent reference:** separately formulated theory or reference
  code. Explain model, parameter, and boundary differences before a quantitative
  comparison.

Use `N_R` and `N_Z` for enclosing grid counts, `n_phi` for the toroidal harmonic,
`K` for the retained harmonic cutoff, `N_phi,quad` for angular quadrature, and
`n_i`, `n_e` for physical number density. A recognizable shape is a valuable
mode candidate; converged frequency, structure, and polarization require their
own evidence. Coarse-grid branch matching must not exclude a newly resolved
structure merely because the coarse calculation could not represent it.

## Finish a version with a stated scope

Archive reproducible inputs, numerical source, diagnostics, clear figures, and
known limitations. Define the prototype milestone by demonstrated use cases,
not by a claim that every defect is fixed. Carry unresolved issues into the
next version with specific reopening triggers. A1 time decoupling and the
scalability of dense spatial operators are separate engineering questions;
neither should be described as already solving the other.
