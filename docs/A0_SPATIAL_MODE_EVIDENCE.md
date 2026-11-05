# Fixed-domain spatial and multi-period A0 evidence

The original numerical source for this campaign is `b2dc5797fbe11ef32bb0e42d1b52a411591370e8`.
The separately frozen quiet-state repair uses `c55f1b6` and retains the original failures.
It is the Python/Warp mainline with explicit RK4 and the finite/massless electron
switch. The native solver remains on `reference/a0-native`. This campaign is
verification of specified equations and discrete operators, not Tokamak/ICE or
experimental validation, and does not freeze an A0 release.

The immutable run root on the development server is
`/home/lxzeng/runs/fomhd/a0-spatial-modes__20261105T055727Z`.
Each report retains its source identity, input arrays, device, checks and failed
attempts. The frozen checkout under `source/` was not changed while jobs ran.
The mainline migration CPU suite recorded 474 passes, nine explicit skips and
three subtests. The owned CUDA pilot recorded 12 passes; its one singular-LU
warning belongs to a deliberate negative test.

## Spatial experiment

The physical reference region stays fixed: `[1,2] x [0,1]` metres, with either
an aligned rectangle or a fixed 128-sided polygon approximating a circle of
radius 0.45 m. Refinement changes the spacing, not the domain. The inner/wall
partition uses a fixed 0.15 m distance from the reference boundary. Geometric
mask errors are recorded separately and need not vary monotonically.

Two smooth pressure perturbations are used. One is a compact bump away from
the wall; the other reaches the wall with compatible value and gradient.
The complete massless stage computes the Ohm field. Its independent continuum
reference is `E = -grad(p_e)/(e*n_e)`, evaluated analytically at the actual
staggered field coordinates, without applying production transfer or gradient
operators to the reference. Pressure-force, uncorrected E0, corrected E and
the wall correction are retained separately. These are spatial stage checks;
there is no time-discretization error in this comparison.

For the wall-reaching profile, the corrected-field relative weighted L2 errors
are:

| Shape | Cells per enclosing direction | Global | Inner |
| --- | ---: | ---: | ---: |
| Rectangle | 12 | 0.0439268 | 0.0219500 |
| Rectangle | 16 | 0.0158830 | 0.00738304 |
| Rectangle | 20 | 0.00757303 | 0.00306628 |
| Disk mask | 12 | 0.0433068 | 0.0218365 |
| Disk mask | 16 | 0.0305021 | 0.00722779 |
| Disk mask | 24 | 0.0156493 | 0.00141578 |

The last rectangle pair gives inner/global slopes 3.94/3.32. The last disk
pair gives 4.02/1.65. At the finest successful levels the fixed wall band
contains 83.8% and 99.2%, respectively, of the squared electric-field error.
This supports a boundary contribution and a substantial disk-wall limitation
for this profile. It is not a universal convergence-order certificate.

The compact profile remains underresolved: global errors are 0.2077 on the
20-cell rectangle and 0.1417 on the 24-cell disk. Most of that error is in the
interior. Thus neither a single convergence curve nor the fourth-order
interior stencil establishes global fourth order for every field and profile.

The original 8 GiB storage cap rejected the complete 24-cell stage before
evaluation. Declared storage requirements were about 31.56 GB for the
rectangle and 13.73 GB for the disk. A separately recorded 20 GiB-cap run on an
owned 32 GiB GPU completed the 20-cell rectangle and 24-cell disk. The 20-cell
disk failed a hard pressure-adjoint check and remains a failed case, not an
interpolated convergence point. All original attempts remain on disk.

A retained-array audit separates this failure from pressure-gradient accuracy.
The direct gradient/adjoint pairing closes near 1e-22 W. The failed diagnostic
includes shared electron pressure transport, with a constant-background defect
`1.5*P0/(e*n) * integral(DJ*J) = -6.262972757605026e-8 W`. This reproduces both
failed profiles to within 5.3e-22 W. Small numerical current from the harmonic
magnetic field has nonzero wall/divergence leakage after finite-precision
assembly and multiplication. Initial harmonic refinement alone did not remove
it reliably. No current masking, independent electron-flux projection or gate
relaxation was applied. A factor-chain audit, including nonzero physical
current, remains necessary before selecting a correction.

## Momentum and quiet controls

Cartesian x momentum is an observation reconstructed from the cylindrical
components: `p_x = p_R*cos(phi) - p_phi*sin(phi)`. It does not introduce an x
grid. A closed wall can exchange momentum with the plasma, so the balance is
`dP_x/dt + outward physical traction = 0`, including the defined field
momentum. The numerical constraint reaction is recorded separately and is not
added to physical wall traction to remove a diagnostic residual.

The finite-mass static pressure control has a nonzero analytic balance:
`dP_x/dt = -pi*A*area_h`, with opposite outward pressure traction. CUDA
measurements on both shapes through 16 cells reproduce this cancellation at
roughly 1e-21 N absolute residual. This tests signs and metric bookkeeping;
it does not prove the evolving nonlinear balance.

The dynamic control uses the same aligned rectangle `[0.9,1.5] x [-0.1,0.5]`
at all resolutions, a fixed toroidal flux, the same perturbation and a common
556.724 ns horizon. It explicitly selects the CPU backend of the mainline
solver. A short same-case CPU/CUDA control agrees to 5.3e-14 relative maximum
error across final-state fields; it is not a performance comparison.

| Grid | RK4 steps | Signed cumulative x defect (kg m/s) | Endpoint semidiscrete x defect (N) |
| --- | ---: | ---: | ---: |
| 3 x 3 | 320 | 1.3792080e-21 | -1.8121010e-14 |
| 6 x 6 | 320 | 1.8097449e-21 | 1.4631961e-14 |
| 9 x 9 | 320 | 2.0267636e-21 | 8.7967467e-15 |
| 6 x 6 | 640 | 1.8097461e-21 | 1.4631933e-14 |

Halving dt changes the six-cell cumulative defect by only 6.68e-7 relative.
The grid sequence does not show a decreasing cumulative defect. This confirms
that reducing the time error alone does not remove the observed floor. Since
the domain has straight aligned walls, curved-mask staircasing cannot be its
sole cause. A separate balance-consistency investigation is needed before
attributing it to a particular spatial transfer, wall closure or field term.

The quiet control removes the density perturbation. The three-cell run
completed 64 steps with a maximum state drift of 3.03e-13 relative to declared
physical scales. Six- and nine-cell runs were rejected after 41 and eight
accepted steps by a wall-state check at approximately 4.2e-31 momentum-density
residual. Their retained partial drifts are about 2.6e-13. These are not passed
static-preservation tests and are not evidence of a plasma instability.

The rejected cases were traced to cancellation in the weighted wall projection:
forming the difference of nearly equal force and reaction left wall-normal RHS
residuals of 5.79e-24 and 2.73e-23 N/m3. Their accumulation predicts the rejected
wall-state trace. The bounded repair solves the same normal system for the
forward trace residual, updating the result, reaction and multiplier
incrementally. At most five corrections are allowed; selected and full wall
rows must satisfy their projected-output FP64 bound. Stagnation is rejected.
No model, wall geometry, source ownership or existing diagnostic tolerance
changed. The design and mirrored pseudocode record the same contract.

The separate `quiet-refined` CPU run, Slurm 10843, completes all 64 steps at the
unchanged 111.345 ns horizon on n=3,6,9. Maximum drift over the declared physical
scales is 1.70e-13, 4.91e-13 and 1.18e-12, respectively. These are short-horizon
static-preservation controls, not an assertion of exact equilibrium preservation
for arbitrary grids or duration. Original rejected histories remain displayed.

The final working-tree CPU suite recorded 492 passes, nine explicit GPU skips
and three subtests. The frozen repair's owned CUDA suite recorded 12 passes;
the singular-LU warning is a deliberate negative test. Separate CUDA n=6/9
quiet-trace probes pass the unchanged wall-state gate at a five-times longer
projected horizon, with full-trace/bound ratios 0.00121/0.01027. Those probes
are stage/RHS controls, not full CUDA time integrations. Independent review
also checked dependent full-wall rows and empty selected-row rejection. These
checks do not repair or reclassify the separate disk n20 pressure-transport
failure or the old evolving momentum defect.

## Eight-cycle longitudinal mode

The test uses production finite-mass stage evaluation, RK4 transactions and
hard diagnostics with a periodic spectral test adapter. Its 17 or 33 points
are samples of that local periodic problem, not an LCFS grid refinement.
An independent warm two-species longitudinal root predicts 8.9811120865 MHz.
Frequency and growth are fitted from the complete complex density history;
the theoretical frequency is not an input to that fit.

The 17-point CUDA runs cover eight periods (890.758 ns):

| Steps | Samples per period | Measured frequency (MHz) | Relative frequency error | Measured growth (1/s) |
| --- | ---: | ---: | ---: | ---: |
| 640 | 80 | 8.9811092449 | 3.164e-7 | -1.16983 |
| 1280 | 160 | 8.9811119091 | 1.976e-8 | -0.03440 |

The frequency error drops by about sixteen when dt halves, consistent with
fourth-order time accuracy here. The continuum growth is zero. RK4 predicts
growth rates -1.17020 and -0.03659 1/s; the finest measured damping is close to
the numerical/fit floor and must not be interpreted as physical damping.
The half-amplitude control and fit residuals are retained. All three 17-point
cases completed without dt warnings and passed their hard checks.

The original 33-point attempt failed during CUDA initialization before any
step. Its failure remains separately archived. The fresh-allocation retry,
Slurm 10841, completes both full- and half-amplitude 1280-step CUDA histories
and passes every hard check without dt warnings. The full-amplitude measured
frequency is 8.9811119104 MHz (relative error 1.96e-8), measured growth is
-0.03667 1/s and phase-fit RMS residual is 2.91e-9 rad. The half-amplitude
sensitivity measure is 9.18e-10 and passes the existing criterion. Its growth
estimate is -0.04045 1/s, illustrating why this very small damping must not be
interpreted as a separately resolved physical effect.

## Implications for further A0 work

A higher-accuracy boundary is justified if global fourth-order accuracy is an
A0 acceptance objective. Geometry approximation and the discrete wall closure
must be evaluated separately. Any replacement must preserve the compatible
divergence/curl identities, adjoint power pairings and shared flux ownership;
merely filling ghosts with a higher-degree polynomial is insufficient.
The present measurements do not select or implement a replacement scheme.

A static-preservation test also does not require a new large Tokamak
equilibrium solver. It first checks whether a simple analytically stationary
state remains stationary under the actual discrete equations. An imported GS
equilibrium gives total force balance and magnetic geometry, but does not
automatically specify separate ion/electron pressures, current partition,
flows and electric field satisfying both species equations. Checking those
initial residuals distinguishes startup adjustment from the mode of interest.
