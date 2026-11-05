# A0 runtime guidance and diagnostic audit

Date: 2026-11-05. The active `main` numerical contract is the Python/Warp
implementation under `src/fomhd/`; archived run source and inputs are identified
separately by their recorded SHA-256 hashes. This is an interpretation and
workflow record. The current default is fully explicit RK4 with electron
inertia enabled; numerical budgets and archived benchmark verdicts are unchanged.
The [implementation record](WARP_IMPLEMENTATION.md) owns the executed test
scope; mirrored [pseudocode](pseudocode/README.md) owns the implementation contract.

## CPU and GPU responsibilities

The repeated numerical stage uses the selected backend. Large field/fluid
arrays, fluxes, Fourier transforms, and sufficiently large linear algebra are
natural CUDA workloads. Configuration, geometry/topology metadata, input/output,
plotting, and small analytical eigenproblems are natural CPU work. Small
irregular setup or small solves should be placed according to the complete
workflow, including transfer and synchronization costs; a CPU operation is not
inherently a missing GPU feature.

The current CUDA solver keeps stage/state arrays on the device, with explicit
setup, scalar reductions, and snapshots crossing the host boundary. NumPy/SciPy
is a deliberately selected reference backend. No silent host fallback or
per-stage full-state copy is part of that contract. Shared Python numerical
logic and backend/library calls avoid writing two independent solvers.

Warp is selected for implementation convenience and maintainability. It lowers
kernels to native CPU/CUDA code; it does not make the physical or numerical
algorithm intrinsically faster than an equivalent C++/CUDA implementation.
Correctness parity remains useful. Repeated native/Warp timing contests and
performance tuning are not routine A0 acceptance tasks.

## Observe-only diagnostic audit

Input evidence: `warp-overnight-retry__20261105T0132/scan/` under the personal
run root. Ten finite-mass CUDA cases use the same requested horizon
`5.567242158847295e-7 s`, grid 4, and K=1. Nine reach it; midpoint with eight
steps rejects the eighth step with `WallState`. Only accepted intervals enter
the interval audit. Observe-only failures are `time_step_estimate`,
`momentum_x`, and `momentum_z`; they are separate from that hard wall rejection.

The momentum check is a physical balance:

```text
R_j = P(t[j+1]) - P(t[j])
      + stage-quadrature(outward momentum stress)
      - prescribed source impulse
```

`P` includes the two fluids and the Darwin field momentum `epsilon0 EL x B`.
The outward stress includes the corresponding field terms. A closed fixed wall
can exchange momentum with the plasma. Numerical wall reaction is recorded
separately; it is not silently identified with continuum wall traction. This
balance is not an algebraic identity guaranteed to vanish to roundoff on an
arbitrary mesh. The design explicitly makes no such momentum claim.

The input generator inherits its budgets from `tests/support.hpp`:

```text
reference = m_i * n_ref * L_ref^3 * u_ref
          = 1.3380975407600003e-17 kg m/s
atol = 0; rtol = 1e-9; gate = observe
threshold = atol + rtol * reference
          = 1.3380975407600005e-26 kg m/s
```

The benchmark renderer changes the run parameters but does not calibrate these
physical-balance budgets for the grid, interval length, or scientific target.
An inherited `1e-9` threshold is therefore not an established acceptance
criterion for this coarse-grid run. Small SI values alone do not establish
that the error is physically negligible.

The signed accumulation `sum(R_j)` is the endpoint momentum change plus the
accumulated outward boundary impulse minus source impulse. It is neither the
net momentum drift alone nor `sum(abs(R_j))`; the latter also measures errors
that cancel in the signed sum.

At fixed grid and horizon, the RK4 momentum-x records give:

| Steps | Maximum absolute interval residual (kg m/s) | Signed accumulated residual (kg m/s) |
|---:|---:|---:|
| 80 | 2.42478e-22 | 1.4287101e-21 |
| 160 | 1.21627e-22 | 1.4286501e-21 |
| 320 | 6.08589e-23 | approximately 1.42865e-21 |

The 320-step history omits one passing interval value. Its unknown contribution
is bounded by `+/-1.33810e-26 kg m/s`; the other two table rows have all interval
values. The audit preserves that bound rather than treating the omitted value
as exactly zero.

The local residual approximately halves with the timestep while its accumulated
value converges to a nonzero floor. Time refinement alone is not eliminating
this defect. Spatial/wall/field reconstruction and possible implementation
error still need to be separated; this is not proof that the staircase wall is
the sole cause. No tolerance was increased to turn these records into passes.

Two CPU evaluations of the saved initial/final states, using the captured run
source and matching dependency lock, separate the balance terms without
advancing time. At the 320-step endpoint, ion and electron x momenta are about
`-2.67068e-18` and `+2.67742e-18 kg m/s`, with field momentum about
`2.19460e-24 kg m/s`. Their net x momentum is only `6.74069e-21 kg m/s`.
The signed accumulated residual is about 21.2% of that net value, despite being
only 0.0149% of the peak species component-RMS momentum proxy
`m_i N_i u_i,component_rms + m_e N_e u_e,component_rms`. These RMS values are
unweighted over sampled velocity components. That proxy measures an
uncancelled scale; it is not a volume-weighted speed, the net momentum, or an
acceptance threshold.

At the same endpoint, the semidiscrete physical-balance defect is approximately
`-1.87139e-14 N`, compared with outward stress `2.23300e-13 N` (about 8.38%).
This supports investigation of a non-temporal balance defect; it does not
identify its cause or establish a physical error tolerance. Evaluating these
two states with current source gives the same reported terms. This CPU probe
is separate from the archived CUDA time-integration results.

The reproducible extraction, source/input hashes, probe, and independent
recheck are retained under
`/home/lxzeng/runs/fomhd/a0-diagnostic-audit__20261105T040232Z/`.

Future physical-budget acceptance should use an explicitly declared physical
scale, interval and cumulative measures, and separate time/grid refinement.
Algebraic constraints, positivity, and linear-solve gates retain their own
strict criteria. There is no justification for applying one roundoff-like
relative tolerance to both exact identities and truncation-error balances.

The timestep estimate is a different diagnostic. The scan intentionally uses
`dt_guard=warn`; the default is `error`. Its configured policy is 32 samples
per fastest estimated cycle, at most 1% amplitude error and 0.1 rad phase error
over the remaining horizon. For example, RK4 at 160 steps initially misses
only the sample floor (about 31.44 rather than 32); midpoint at that resolution
also exceeds the amplitude/phase budgets. Both 320-step cases pass the initial
estimate. These are configurable accuracy safeguards on a frequency estimate,
not a measured full-system stability boundary or proof of a physical failure.

Frequency inference is also distinct: the two eight-step records have fewer
than 16 samples; the remaining eight closed-wall records contain fewer than two fitted
cycles (about 0.94--1.00 cycles, R-squared about 0.57--0.60). A longer record is
needed, but by itself cannot guarantee that a mixed observable identifies the
Langmuir branch. The separate unforced longitudinal control already has a
known analytic reference and a usable two-period observation window.

## Electron-model switch and explicit timestep

`species.electron_inertia=0` selects `massless.py` behind the existing stage,
time, state, and I/O interfaces. This is a singular physical limit: removing
electron inertia turns its differential momentum equation into force balance.
It is not implemented by setting a mass divisor to zero.

At every stage, with `e=-q_e>0`, the branch constructs:

```text
n_e = q_i n_i/e
J = Cdag B/mu0
F_e = (q_i F_i - J)/e
```

`F_i` is the same numerical number flux used by ion continuity. Electron
velocity is reconstructed from `F_e` using the shared interpolation and wall
constraint. Thus charge neutrality and current/number-flux continuity are part
of stage assembly, rather than a density correction after the time step.
The continuum interior form of the field is

```text
E = -u_e x B - div(P_e)/(e n_e)
  = -u_i x B + (J x B)/(e n_e) - div(P_e)/(e n_e).
```

`P_e` is the CGL pressure tensor. The actual discrete implementation includes
the compatible transfers, wall projection, and constrained field correction;
the displayed continuum formula does not replace them. Ions, the two electron
pressures, and B evolve. Electron density and compatibility momentum slots are
derived. Physical electron mass stays positive in metadata, but electron
inertia and kinetic energy are absent from this model. Finite-charge Gauss and
Langmuir checks are explicitly inapplicable. Cross-model restart is rejected.

The cold parallel Hall estimate is `omega_H ~ B k^2/(mu0 e n_e)`. Removing
electron inertia removes its high-k saturation. A finite grid has finite
resolved k, so its explicit timestep must shrink approximately as grid spacing
squared when this branch controls the step. `stability.py` estimates resolution
from the actual discrete derivative norms, combines Hall/ion/transport rates,
and checks the selected integrator. This is an estimate, not a rigorous bound
on the full constrained, nonnormal CGL operator. No implicit advance, hidden
filter, or clipping removes this stiffness.

RK4 has the scalar imaginary-axis limit `abs(omega*dt)<=2*sqrt(2)`, with stricter
accuracy limits still needed. Explicit midpoint has no nonzero imaginary-axis
absolute-stability interval; its finite-horizon amplification must be bounded.
RK4 is the default for the active Python/Warp mainline and is the preferred
choice for new explicit oscillation reference cases.
It uses four derivative stages rather than two and has numerical damping, so
growth/damping-rate errors still require measurement. The observed advantage is
case dependent; both implementations also evaluate endpoints for diagnostics.
Existing midpoint implementations and historical controls remain available only
when explicitly requested. A1 time decoupling is deferred to the user's
forthcoming algorithm, including its own global-order verification.

## Remaining A0 acceptance work

The [fixed-domain and multi-period campaign](A0_SPATIAL_MODE_EVIDENCE.md) now
supplies independent continuum Ohm-field errors, finite-mass momentum
refinement, a same-grid half-dt control, and eight-cycle CUDA mode measurements.
It also retains failed cases; these measurements do not close global A0
acceptance. The earlier short-window results above remain historical evidence.

1. Isolate the momentum floor with an independent semidiscrete balance and
   fixed-domain grid refinement, separating wall, transfer, and field terms.
2. Establish global spatial error budgets for the current closed wall, including
   an independent continuum electric/Ohm-field comparison. Current rectangle
   results already show that curved-mask geometry is not the only issue.
3. For selected wave branches, use a clean perturbation and multiple coherent
   periods; compare frequency and growth/damping, not just solver completion.
4. Build a consistent equilibrium/startup case, check residuals and quiet-time
   drift, then validate physical modes against an independent theory or HYM-like
   reference within the actual fluid closure's scope. G-EQDSK import alone is
   not a self-consistent two-fluid or CGL equilibrium.
5. Freeze an explicit benchmark scope before an A0 release tag. Small-domain
   CPU/GPU verification does not yet establish production Tokamak/ICE validity.

Higher-order boundaries, sparse scalability, tuning, implicit electrons,
kinetic extensions, and A1--A3 are separate work. They need not all be completed
to freeze a carefully limited A0 reference, but its accepted and unresolved
scientific scope must remain explicit.
