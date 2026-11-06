# A0 global fast-branch prototype acceptance

This is a scoped engineering prototype, not experimental validation or a claim
of globally fourth-order accuracy. The mainline is Python/Warp/CuPy, RK4,
massless quasineutral CGL-B electrons (`electron_inertia=0`). Finite electron
inertia remains an explicit historical/local option. D062 gives massless
priority through A0/A1/A2/A3 to 1.0. All prior failure records are retained.

The user selected cold Hall fast-branch multi-period responses first. Slow,
near-static and zero-frequency candidates remain in the complete production
Jacobian/eigenvalue archives; they are deferred research, not removed modes or
A1 prerequisites. There are no fast particles in these calculations.

## Reproducible evidence

Run root: `/home/lxzeng/runs/fomhd/a0-tokamak-cae__20261105T154538Z`.
The frozen numerical checkout is `campaign-source-r1/`. Each individual report
contains per-file SHA256 identities and an exact source snapshot. These runs
precede the final Git commit; the snapshot hashes, rather than a later commit
label, identify the executed code. Later runner changes affect only explicit
horizon metadata and generated scope comments; the numerical solver is unchanged.

`final-comparison-r2/summary.json` contains every named base/control/backend
comparison and input hash. `final-plots-r2/receipts/report.json` identifies the
six detailed response figures. The combined historical gallery is at
`/home/lxzeng/runs/fomhd/a0-benchmark-figures__20261104T050738Z/index.html`.
The new appendix preserves every earlier figure and failure.

All 18 CPU base/half-step/half-amplitude runs reached their eight-period target.
The quoted cycles are measured fits, not the requested period count. Every
available frequency uses at least four measured cycles and 32 samples/cycle.
Five spatial density probes contribute the median and spread.

| Geometry / grid | Harmonic | CPU median frequency (MHz) | CUDA accepted window | Independent cold reference (MHz, order 8) |
| --- | ---: | ---: | --- | ---: |
| Circle 12 x 12 | 0 | 16.217936 | 8 cycles, 512 steps | 14.995219 |
| Circle 12 x 12 | 1 | 16.476508 | 8 cycles, 512 steps | 16.792938 |
| Circle 16 x 16 | 0 | 15.873733 | 0.125-cycle execution smoke only | 14.995219 |
| Circle 16 x 16 | 1 | 16.059238 | 0.125-cycle execution smoke only | 16.792938 |
| EAST 12 x 12 | 0 | 5.773939 | Budget stop: 511 steps, at least 6.039 measured cycles | Not established |
| EAST 12 x 12 | 1 | 5.748953 | Budget stop: 479 steps, at least 5.598 measured cycles | Not established |

EAST GPU fits over the shorter windows give 5.753640/5.745317 MHz. This window
dependence is compatible with a relaxing background and a mixed response;
it is not a CPU/CUDA discrepancy. Identical physical inputs, numerical source,
amplitude, seed, timestep and exact common times are checked before backend
comparison. The matched magnetic histories agree with relative weighted L2
errors 5.87e-11/2.74e-12 (circle n0/n1) and 1.14e-10/4.95e-12 (EAST n0/n1).
The circle16 CUDA smokes have different cadence and are explicitly excluded
from direct common-prefix parity and frequency acceptance.

## Physical configuration and interpretation

Circle: R0=3 m, a=1 m, Bphi=3.4*3/R T, deuterium, uniform density
3.423e18 m^-3, axis total beta=1e-4, equal ion/electron pressure,
isotropic initial electrons, fixed closed conducting reference wall.
The numerical toroidal flux is normalized to an independent continuum integral.

The cold Hall reference independently assembles Fourier/Ritz fields and a skew
linear generator; it imports no production discrete operators or RHS. It has
a constrained weak-wall formulation, including orthogonal wall reactions.
Equivalence to an ordinary smooth conducting Hall boundary-value problem is
not proven. Consequently this is an independent model/reference comparison,
not direct reproduction of a published smooth-wall frequency.
Increasing reference order 8 to 12 changes n0 by 0.889% and n1 by 0.948%.
The production grid12 to grid16 frequencies change by -2.12%/-2.53%.
At grid16 the order-8 frequency differences are +5.86%/-4.37%.
Energy-weighted reference overlaps increase from 0.499/0.642 to 0.604/0.652;
the structures remain approximate, not pointwise convergence evidence.
Only two production grids were used: no global convergence order is claimed.
The circular eigenfrequency seed is a named production linear eigenvector
selected by overlap with the independent cold-fast reference. Matching the
observed frequency to that same production eigenvalue verifies evolution;
only the separate cold reference supplies an independent physical comparison.

EAST input: discharge 48059 at 3.65 s, original EFIT file from the pinned
[GTAW source](https://github.com/Youjunhu/GTAW/blob/7ae4929f2b4e56637f0462a9a186d92edb17f316/gfile/g048059.03650),
located using the [VEQDB data description](https://github.com/FusionAlpha/veqdb/blob/main/SOURCE_DATA.md).
Raw SHA256: `df8f073bbad202d799b70d0bc2479c9cbcc2d64b729c1019f2575b14de961842`.
The original EFIT suffix and bytes are preserved. The contour is psiN=0.8;
pressure is retained with pi=pe=p_EFIT/2, uniform deuterium density 2e19 m^-3,
no imposed ion flow. These density/pressure splits are assumptions.
An independent spline-curl current integral requires psi_num=-psi_raw to
match the header current sign; F/R and pressure are unchanged. This is an
explicit convention audit, not general COCOS validation.
The minimum magnetic-norm compatibility projection changes the sampled B by
1.11254%. Raw/mapped fields, initial RHS and unperturbed controls are archived.
The resulting initial state is not an exact two-fluid equilibrium. Its
background relaxation and fitted growth are not evidence of fast-ion drive.
EAST has only one production grid and no independent global eigenfrequency
oracle; these are real-geometry response results, not a validated EAST CAE or
ICE prediction. Published CAE/HYM work guides interpretation only.

n0 uses K=0 and one real physical amplitude A; n1 uses K=1 and conjugates A/2.
Both use Nphi=5. n0 extraction retains the zero Fourier coefficient and fits
a real oscillation with offset/trend. n1 fits the complex coefficient.
The separate K0/K1 axisymmetric short test uses identical times and gives
relative matched B/density/velocity L2 differences 2.51e-12/1.51e-8/7.16e-13.
Original fields, changes from initial fields, and matched-control perturbations
remain distinct. Velocity and compression maps label their separate snapshot times.

## Controls and energy scale

Base versus half-step complex magnetic-history differences are 2.1e-5 on
circle12, 2.8e-6 on circle16, and 2.0e-5 to 2.3e-5 on EAST12. Base versus
twice-half-amplitude differences are at most 3.03e-4. The selected frequencies
are stable to approximately ppm changes under these controls. EAST probe
spreads are 0.43%/0.69%, versus at most 1.02 ppm for circle12; a single pure
stationary global eigenmode is not established for EAST.

Energy includes ion kinetic/internal, electron perpendicular plus half parallel
pressure energy, and magnetic energy. Massless electrons contribute neither
electron kinetic nor longitudinal electric energy. No full-Maxwell transverse
electric vacuum energy is inserted. Balance uses the actual accepted RK4
boundary-power ledger and source power (zero here). No numerical wall reaction
is added as physical source power.

| CPU base case | Peak quadratic magnetic + ion kinetic signal (J) | Max raw balance / signal | Max matched-control balance / signal |
| --- | ---: | ---: | ---: |
| Circle12 n0 | 0.295530 | 6.66e-6 | Recorded in comparison JSON |
| Circle12 n1 | 0.397491 | 6.60e-6 | Recorded in comparison JSON |
| EAST12 n0 | 0.00914791 | 0.1663 | 1.466e-5 |
| EAST12 n1 | 0.00618149 | 0.2413 | 2.712e-6 |

EAST raw gaps are important at the perturbation scale despite tiny relative
background energy error. The unperturbed background has almost the same gap;
the displayed matched difference is a diagnostic, not a replacement balance.
Halving dt reduces the raw gap to 0.524%/0.760% of the signal while preserving
the measured frequencies. The half-amplitude raw gap can approach the smaller
signal; its common-background origin and matched balance remain visible.
This targeted comparison does not justify erasing the raw concern or asserting
long-time nonlinear conservation. It does justify retaining these frequencies
as prototype responses without immediately redesigning the boundary.

The archived disk n20 pressure case was replayed for eight actual RK4 steps
with power/energy/momentum gates observe and original thresholds retained.
Its electron pressure transport value is -3.3017e-8 W (CPU) against a strict
1.3020e-17 W threshold; the observe failure remains recorded. Total energy
changes are below FP64 resolution in this very short 1.202 ps window, not a
long-time energy-conservation proof. The original CUDA hard-failure record is
untouched. Necessary physical/numerical constraints remain hard in all cases.

## Execution and remaining scope

CPU uses explicit NumPy/SciPy on the same production stage; it is not silent
CUDA fallback. CUDA runs used owned Slurm RTX5090 allocations. Formal CUDA
harness elapsed times sum to 4.5354 GPU-hours, excluding queue time, launcher
setup and earlier pilots; this is not scheduler billing. Completed/budget-stopped formal allocation
limits total 5 h 40 min, and earlier pilots/short failed launches were bounded.
No job remains running. The 8 GPU-hour cap was not used to demand every control
on GPU. EAST grid16 and longer nonlinear/window studies remain expansion work.
Measured circle16 CPU base windows cost 48/66 minutes (n0/n1); half-step
controls cost 91/123 minutes. Scaling EAST12 by these observed grid factors
suggests roughly 1-2 CPU hours per EAST16 base window and twice that for
half-step controls, plus spectral preparation; this is an estimate, not a
measured EAST16 runtime.

A fresh full CPU suite recorded **594 passed, 9 explicit GPU skips, 3 subtests**
with one BLAS thread and warnings as errors. A separate independent focused
review recorded 64 passes. Actual formal CUDA stage diagnostics are retained;
the CPU skips are not relabeled GPU tests. Final plotting checks are recorded
separately. Known issues remain in `A0_KNOWN_ISSUES.md`.

Reproduction entry points (use new output directories and owned CUDA allocation):

```bash
uv run python tools/production_spectrum.py --help
uv run python tools/cold_hall_reference.py --help
uv run python tools/warp_tokamak_cae.py --help
uv run python tools/warp_pressure_observe.py --help
uv run python tools/plots/plot_tokamak_cae.py --help
uv run python tools/plots/plot_cae_comparison.py --help
```

The actual per-case Slurm scripts, spectrum selection scripts, configurations,
resolved parameters, input file and source hashes are retained under the run
root. Harness `case.toml` describes that response experiment; it is not the
strict core CLI case schema. The harness saves diagnostic histories rather
than resumable full-state checkpoints; core CLI restart is separately tested.
Prototype acceptance covers reproducible fast-branch evolution and these
bounded physical comparisons. Slow modes, boundary order, equilibrium accuracy,
EAST grid convergence, energetic-particle drive and experimental validation
remain explicitly outside the release scope.
