# T2 window algorithms: selected schedule and remaining integration

Status: implementation authorized by the user on 8 November 2026 (D070).
E1→E2→E3 are internal verification nodes and proceed continuously within the
existing budgets. Their numerical/implementation status is reported separately.
This schedule preceded implementation. The E1/E2/E3 field modules now implement
the selected bounded schedule; their measured scope and failures are recorded
in A1_T2_E_VERIFICATION.md. No complete coupled runtime control is added.
Existing mirrored T0/T1 files remain their authoritative guides.

Authority: design `A1/A1_T2_CANDIDATES.md`, `A1_VALIDATION_AND_SCALING.md`,
`A1_ALGORITHM_DESIGN.md`, and shared K7. T0/T1 references and their acceptance facts are retained.
The authorized implementation scope is T2-E1/E2/E3. The concrete window and
operator-power schedule is in `src/fomhd/time_field.py.md`. T2-I, integration,
FK and T3 production runs remain outside this authorization.

## Planned data and ownership

- Retained physical arrays remain FP64/complex128, SI, with the existing entity
  and Fourier layout. A window adds an explicit leading microstep dimension;
  RK stage1..4 is separate, including the two midpoint stages.
- For E1, a periodic spatial harmonic has a two-component transverse magnetic
  block or the selected four-component Hall/CGL block. This is not the complete
  production spatial state. General field state size grows with the mesh.
- Ion path, accepted left endpoint, forcing and previous iteration arrays are
  read-only. New window, residuals, affine tree and recovered stages have
  distinct owners. No updates become visible to another node in the same old
  batch. Scratch lifetime includes all queued stream operations.
- CPU uses actual tree levels. GPU follows the same time order, synchronizes
  level dependencies on the declared stream, and keeps all arrays on device.
  Small residual scalars and declared output may reach the host; no silent
  fallback or per-stage whole-state transfer.
- The pure field API owns each invocation and explicitly indexes microstep/RK
  stage; it does not consume or mutate Context stage counters. Producer/cache
  identity describes immutable physical/operator inputs. Trial data remain local
  and are not cached across old/new generations. Complete StageTag expansion
  covering outer/field/particle iteration and accepted transactions remains an
  integration requirement; no production runtime identity is claimed here.

## E1: affine RK4 window and all prefixes

Use zero-based j=0..M-1. The selected fixed L and four supplied sources f[j,a]
define the same RK4 step as the independent sequential oracle.

```text
For each microstep j independently:
    Calculate the four-stage RK4 response with zero initial state:
        k1 = f[j,1]
        k2 = L * (h*k1/2) + f[j,2]
        k3 = L * (h*k2/2) + f[j,3]
        k4 = L * (h*k3)   + f[j,4]
        b[j] = h*(k1 + 2*k2 + 2*k3 + k4)/6
    A[j] = I + h*L + (h*L)^2/2 + (h*L)^3/6 + (h*L)^4/24
    Retain the distinct source and stage identities.

Define COMPOSE(earlier, later):
    return (later.A * earlier.A,
            later.A * earlier.b + later.b)
Identity = (I, 0)

Pad leaves to next_power_of_two(M) with Identity.
Up-sweep, from leaves to root:
    aggregate[parent] = COMPOSE(aggregate[left], aggregate[right])
    Each parent reads only the previous level.
Down-sweep, from root to leaves:
    exclusive[root] = Identity
    exclusive[left] = exclusive[parent]
    exclusive[right] = COMPOSE(exclusive[parent], aggregate[left])
    Retain aggregate[left] until its consumer completes.
For each real leaf j:
    inclusive[j] = COMPOSE(exclusive[j], leaf[j])
    Z_left[j] = exclusive[j].A * Z_accepted + exclusive[j].b
    Z_end[j]  = inclusive[j].A * Z_accepted + inclusive[j].b
    Recover all four actual RK4 stages from Z_left[j] and f[j,1..4].
Discard padded leaves; verify every original local RK4 relation.
```

This defines temporal order explicitly. The independent oracle uses a plain
sequential recurrence, not COMPOSE or the tree implementation. Tests need
noncommuting matrices, nonzero sources, nonzero starts, M=1/3/5/17/65, constant
and varying maps, midpoint-specific forcing, and intentional reversed-order or
missing-source controls. The constant-L layer-power specialization is checked
against the general tree; setup and refresh are counted.

E1 is a direct affine scan; nonlinear field r_Z is not applicable. Its result
does not establish real nonlinear field-inner convergence.

The first physical fixture uses stable, uniform periodic Hall/CGL blocks and
matched discrete derivative symbols. The production modal finite difference is
LIN evidence; independent Hall/CGL equations or manufactured driving are REF.
No whole-space dense propagation matrix is constructed. Fourier decoupling of
this model is not a claim about nonlinear RZ production geometry.

## E2: selected true-residual correction

```text
Input: fixed complete P[j,a], accepted Z0, valid old window W_old.
Build/version the frozen fast propagator approximation.
Charge all stage-source construction and initial affine-prefix work.
For each bounded field-inner iteration:
    For all microsteps independently:
        Use W_old[j-1] (or Z0) and fixed P[j,1..4].
        Reconstruct real massless states/Ohm fields at all four RK4 stages.
        Compute original Phi[j] and F_Z[j] = W_old[j] - Phi[j].
    Solve delta[j] = A0*delta_left[j] - F_Z[j], where
        delta_left[0] = 0 and delta_left[j] = delta[j-1] for j > 0,
        using the selected structured approximate inverse, not a hidden rollout.
    Write W_trial = W_old + lambda*delta into separate storage.
    Test the ORIGINAL residual/defect using a signal-scaled tolerance,
        plus constraints and validity. Calibrate the tolerance in bounded
        validation; do not run a serial trajectory oracle inside each trial.
    Charge every allowed line-search trial and refresh.
    Replace W_old only after this trial is selected.
Stop with explicit failure on the configured bound or necessary gate failure.
Recover one final complete stage history from the selected node values.
No inner or replaced history enters the accepted interval ledger.
```

GPU E2 uses FieldBatch with a real independent time axis and trial-local arrays.
The wrapper owns attempted-work and check aggregation; it does not concurrently
share mutable Context counters. Independent microsteps are numerical batches;
only four stages/tree levels retain host control dependencies. The sequential
complete-microstep loop is reserved for the explicitly labeled oracle/baseline.

A frozen derivative is approximate Newton, not exact Newton. Real JVP includes
RK stages and density/field/constraint response. Production real/abs/domain
branches rule out an automatic complex-step shortcut. RZ extension needs
operator-based structured inverses; matrix-free JVP alone is not a fast inverse.
Arbitrary B is not repaired by electron canonicalization: trial magnetic data
must stay in the compatible CT/wall perturbation subspace.

## I: proposed known-RHS prefix, after field-unit review

```text
Input: read-only full ion RHS F[j,1..4], accepted P0.
For every j: delta[j] = h*(F[j,1]+2F[j,2]+2F[j,3]+F[j,4])/6.
Use a genuine addition up/down tree to compute all exclusive prefixes.
For every j independently:
    Qleft = P0 + exclusive_delta[j]
    P_new[j,1] = Qleft
    P_new[j,2] = Qleft + h*F[j,1]/2
    P_new[j,3] = Qleft + h*F[j,2]/2
    P_new[j,4] = Qleft + h*F[j,3]
P_endpoint_new = P0 + inclusive_delta[M-1]
```

This does not compute fewer ion RHS values and does not turn spatially coupled
fluid transport into independent small matrices. Strong cancellation and
non-power-of-two lengths use physical and signal-scaled roundoff budgets;
bitwise equality with sequential summation is not required.

## Integration, validation and failure boundary

Retain the implemented T1 outer transaction. An approximate field solve requires
both the original field-window residual and the complete ion-stage/endpoint
map defect. Small ion updates alone no longer certify the original fixed
point. T1's exact-sweep nilpotence is not transferred to a new preconditioned
map without analysis.

Only final selected stage sources and physical-second quadrature weights enter
one pending ledger. Rebuild the compatible endpoint and cache, apply unchanged
hard gates, synchronize, and commit exactly once. Any failure leaves accepted
state/cache/ledger unchanged and retains attempted operations and residuals.
Restart restores accepted endpoints only; new controls/propagator semantics
need fingerprint and compatibility tests before runtime integration.

For each unit: a bounded real CPU A1 schedule plus independent oracle, then
same-algorithm GPU parity over all nodes/stages/impulses/constraints/cache and
rollback, then same-GPU old/new timings at matched error and observation time.
CPU A0 is a numerical reference, not exact truth. CPU/GPU hardware ratios are
reported separately from time-algorithm gains. Work, span, host/device peaks,
S, each r_Z, setup/refresh, derivatives, communication, checks and rejected
replay are all recorded. Particle q and r_p are not applicable in these units.

The design authority supplies the finite configuration/job/memory budgets and
report template. No planned pseudocode step constitutes completed code, a new
traceability implementation row, or benchmark evidence.
