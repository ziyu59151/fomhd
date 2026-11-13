# A0 global compressional-response runner

Authority: D062 and A0_TOKAMAK_CAE. Python host orchestration; production
massless/RK4 stage and all stage arrays stay on the explicitly selected backend.
CPU is allowed for tests/input preparation. CUDA requires an owned Slurm job.

1. Validate kind, nphi=0/1, K>=nphi, Nphi=5, grid, amplitude and budget.
   Enclosing grid sizes 4..32 are supported by this bounded dense reference.
   Accept an explicit positive memory limit in bytes (CLI GiB, default 24).
   Apply it to operator construction and the unchanged state peak-memory gate;
   it does not imply that larger grids fit the reserved CPU/GPU allocation.
   Build fixed circular or psiN=.8 EFIT contour and equal enclosing RZ cells.
   Record exact source hashes, input hash, density/profile assumptions and units.
2. Create physical SI references and the production massless diagnostic registry.
   Only power/energy/momentum budgets are observe for prototype cases. Retain
   their thresholds and all raw checks; positivity, state, model, solver and
   electromagnetic identities stay hard. Strict verification cases are unchanged.
3. Prepare B at its actual staggered positions. For the circular current-free
   background normalize the discrete harmonic field to independent continuum
   toroidal flux. For EFIT sample the spline B and record a minimum-MB-norm
   projection to DB B=0 and S Cdag B=0, preserving selected toroidal flux.
   Save unprojected/projected arrays and relative corrections, never change EFIT
   pressure to hide startup forces. Derive electron current/velocity consistently.
4. Circular perturbation comes from the independent cold Hall eigenpotential,
   velocity and linear continuity/pressure relations. Apply one global phase,
   normalize by physical B,vA,n scales, and encode n0 real A or n1 conjugates A/2.
   Project initial ion normal momentum and B compatibility, record corrections.
   EAST uses a documented smooth compressional velocity seed and is a response
   experiment, not an exact eigenmode initialization or a fast-ion calculation.
5. Initialize using the normal production transaction, not equilibrium RHS
   subtraction. Record the initial RHS and background residual. Estimate the
   actual operator-resolution rates, choose dt satisfying existing RK4 guard
   and sample budgets; dt-halving retains exactly the same physical horizon.
6. Run unperturbed and perturbed states on identical times. All snapshots are
   explicit diagnostics. Store accepted energy partitions, cumulative actual
   outward boundary flux (source=0), momentum, full failed checks, initial RHS
   and dense nodal signals. This harness does not write resumable checkpoints;
   restart remains a separately tested core CLI facility. On failure keep exact
   accepted prefix. A wall-clock cap is a budget stop, never a success.
7. Subtract matched controls (also for n0). Measure positive quadratic delta-B
   and delta-v energies as well as total-energy differences, compression,
   polarization, density and several probes. Real-fit n0; complex-fit n1;
   report insufficient cycles explicitly. Save raw fields, initial differences
   and matched-control differences. Never remove the zero Fourier mode.

## Resolved spectrum option

A named production eigenfrequency is an optional, explicitly labeled branch
selection; all lower-frequency candidates remain in the spectrum report. It
is not a claim that this branch is the lowest finite global mode.
Validate geometry, harmonic coverage and exact numerical-source hashes of all
provided production spectra. K=1 requires n=0 and n=1 spectra. Recompute the
selected eigenvector from the archived Jacobian, normalize physical n,p,B,vA
amplitudes, set its B-compression phase, and encode the Hermitian state once.
Keep the independent cold Hall reference separate from this discrete seed.

When a complete spectrum is supplied, choose dt with rho*dt <= .5/1.5 and at
least 64 samples of the selected mode, checking its existing RK4 accuracy
budgets over the full horizon. Keep the conservative isotropic Hall estimate
as observe (`dt_guard=warn`); it is not overwritten. The explicit harness guard
stops if B/density departure exceeds the 50% linear-spectrum headroom or
scaled rho*dt exceeds .5. This is a local linear calibration, not a nonlinear
stability proof; half-dt and half-amplitude controls remain required.
Advance matched control and perturbed runs in alternating intervals so budget
stops preserve a matched physical prefix. The budget never spends an entire
unperturbed run before beginning its perturbation.

Report target_periods = actual requested horizon * selected frequency, including
explicit horizon overrides. CLI cycles is only the default horizon request.
Each generated harness parameter includes an English scope/validation comment.
# Magnetic sampling

Use the common stored-chart magnetic reconstruction to turn a candidate
direction into a cell magnetic field.
