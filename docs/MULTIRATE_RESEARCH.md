# Reduced-ion-work multirate research

D071 selects two bounded two-fluid prototypes. A macro window H contains M
field RK4 microsteps h=H/M. They evolve both ions and fields, with a different
macro discretization from the complete-stage T1 reference. CPU/GPU evidence,
raw failures and cost comparisons are indexed by the dated design report.

C1 uses the centered ion feedback Pc=(P0+P1)/2. Each outer sweep evaluates one
complete hydro/pressure node, scans the real field RK4 history at fixed Pc,
then applies one integrated electromagnetic impulse. Fixed-density projection
is linear in that impulse. C2 freezes the expensive hydro terms at the actual
mid-window ion state while continuity and Lorentz grid responses remain live
at every field stage. Both have formal second-order macro accuracy with
fourth-order field micro-integration, subject to controlled iteration error.
The production full-stage T1 scheme remains unchanged.

`src/fomhd/time_multirate.py` owns the research callback algorithms and the
adapter to actual production primitives. `tools/warp_multirate_ion.py` compares
with complete synchronous RK4, saves all stage identities, endpoint split
checks, raw energy observations, operation counts and retained failures.
`tools/warp_multirate_cost.py` adds a lean synchronous reference with the same
D/A/L/G primitives and matching research checks, cold/warm/repeat protocol and
resource scope. This separates algorithm costs from production transaction and
diagnostic overhead. Shared fixture setup, repeated work and transfers are
reported separately; post-campaign device pool samples are not allocation traces.

## Interface and ownership

Inputs are immutable CuPy/NumPy complex128 owning arrays in SI units:
P has shape `(2*K+1,5*Nc)`, containing ion density, three momentum components
and scalar pressure; Z has shape `(2*K+1,2*Nc+Nb)`, containing electron
perpendicular/parallel pressures and retained staggered magnetic field.
Operators/geometry are borrowed read-only on the selected backend. Results,
old/new midpoint guesses and all four distinct RK4 stages own separate trial
arrays. The two midpoint stages remain distinct even at the same physical time.
Outer, macro, micro and stage identities and times are saved explicitly by the
runner. Endpoint fields/caches are marked research objects and cannot be
promoted as accepted production caches.

No accepted state or cumulative physical ledger is modified. A failure retains
completed windows, current completed stages, actual failed inputs and attempted
work counts. Every outer sweep restarts field integration from the same
accepted Z0 and re-evaluates the true nonlinear field equations. C1's constant
Pc is a quadrature state, not a claimed continuous ion trajectory. C2 does not
freeze continuity. Neither method replaces explicit field RK4 or removes Hall
physics.

## Evidence and performance boundary

The bounded suite separates fixed-h H scans from fixed-H h refinement. It uses
finite-amplitude/nonuniform periodic controls, distinct high-mode seeds, short
nonlinear wall compatibility fixtures, and a selected C2 2T/half-h comparison.
CPU and GPU compare the same discretization, accepted-trial endpoints and all
final-outer stages on physical scales. Closed-wall phase and long-time claims
are not inferred from the short compatibility fixtures.

There is one expensive hydro node per outer sweep. Optional independent
endpoint split checks add two hydro nodes per macro and are charged. Cheap
continuity, Lorentz, wall and field operations are counted individually; they
are not renamed full-ion work. The cold Hall matrix control distinguishes
removing T0 amplification from retaining phase accuracy.

The current field scans are causal host-scheduled RK4. Their GPU backend and
reduced hydro counts are not new time-parallel deployment. T2-E remains a
separate fixed-ion research subproblem. Known-RHS prefixes only accelerate
accumulation. Orbit Newton prefixes require actual multiple orbit nodes and a
fixed complete E/B history, and have their own matrix/vector and nonlinear
correction costs.

No interval-work transaction, production restart, multi-ion CCS closure or FK
pusher is added by these prototypes. Fluid work counts do not establish
particle pushes/gathers/deposits saved against a reasonable hybrid reference.
A3 retains bulk fluid plus self-consistent CCS-coupled FK fast ions; A4 replaces
all ions. Particle orbit phase, resonance and feedback require their own tests.
