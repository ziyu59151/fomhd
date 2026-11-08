# T2-E: fixed-ion electron/field window

Status: implemented in a bounded experimental API under the 2026-11-08 user authorization. E1→E2→E3
continue through internal checks within the existing budgets. No T2-I, complete
coupled acceptance, FK or multi-GPU integration is included.

Input P[j,a,m,:] holds SI ion density, three momentum components and pressure;
Z[m,:] holds two electron pressures and constrained magnetic cochains. All are
complex128 on one explicit backend/stream. The four RK stages retain identity.
Input, accepted state and ledger are immutable. A result is a field-subproblem
trial, never a physical commit. Cache/source identities belong to this window
and its outer/inner/trial/node/stage version, not mutable global stage counters.

## Real RK4 batch and residual

For j=0..M−1 simultaneously, read left=Z0 if j=0 else W_old[j−1].
Compute stage states left, left+h*k1/2, left+h*k2/2, left+h*k3, with
k_a=G(P[j,a],stage_a). Each G uses actual massless reconstruction, CGL-B,
constrained Ohm and the real spatial operators. Four stage dependencies remain
causal; no dependency crosses microsteps within this residual evaluation.
Phi=left+h*(k1+2*k2+2*k3+k4)/6; F=W_old−Phi.
Record full states/RHS/E/constraint results. No ion RHS or full-system energy
acceptance is inferred from this pure field producer.

Initial guess: parallel tangent nodes Z0+(j+1)*h*G(P[0,0],Z0); charge its RHS.
An explicitly supplied initial trajectory is copied and its provenance recorded.
For each bounded inner round, solve delta[j]=Aref*delta_left[j]−F[j] using the
operator prefix below, with delta_left[0]=0. Test lambda=1,1/2,1/4 against a NEW
evaluation of the ORIGINAL residual. The entire round reads one old version.
Accept a trial only if it reduces the scaled residual or meets tolerance.
Return the last already evaluated stage history at convergence, without final
serial replay. Failure returns residual/cost evidence and leaves inputs and
accepted ledger unchanged. Signal-scale tolerances are calibrated against the
serial same-discrete oracle outside the iterative solve.

## Constant-reference operator prefix

Represent a segment by integer length n and vector offset b. Composition is
(n_a,b_a) then (n_b,b_b) → (n_a+n_b, Aref^n_b b_a+b_b).
Pad offsets with zero only at the tail; padded propagation can be discarded
because it never precedes a real leaf. Up-sweep all independent parent offsets
at a level in one batch. Down-sweep propagate left states: left child inherits
parent left state; right child=Aref^n_left(parent left)+b_left. Recover every
real leaf endpoint Aref(left)+b_leaf. Never represent powers by an unevaluated
chain of n dependent short steps and call its cost constant.

E2 periodic reference: uniform frozen Hall symbol in spatial Fourier space;
two magnetic components have a 2×2 RK4 rotation polynomial, electron pressures
and longitudinal flux have identity reference. Cache level powers. Real G keeps
nonuniform density, finite amplitude, pressure transport and nonlinear fields.
FFT/IFFT and all mode powers operate across the time batch on the same device.
This reference is specific to the periodic verification geometry.

E3 bounded reference: freeze the TRUE spatial derivative at one declared Pbar,
Zbar; centered real, scale-normalized JVP includes density-weighted constraints
and boundary response. Approximate Aref^n by the truncated polynomial
[R4(x)^n]_(degree<=p), x=h*Lbar, p<=30. Merge powers by scalar polynomial
convolution/truncation; store coefficients and integer lengths, not d×d maps.
Apply powers by Horner JVPs to an entire vector batch. This is a defined
approximate inverse, not exact composition or quadratic Newton. Charge every
degree, JVP pair and true residual; test against explicit small Jacobians and
serial RK4. Polynomial truncation/non-normal amplification are measured risks.
If it is costly or fails, report that specific representation as unsuccessful;
do not declare all matrix-free or time-parallel field algorithms impossible.

For degree p_l at level l, power application has p_l sequential derivative
actions, with independent subtree vectors batched. The tree depth is therefore
proportional to sum_l p_l (plus true RK stages/inner rounds), not just log M.
Vector history costs O(M*d); coefficients O(p*log M). True spatial operators,
projection factors and their temporary workspaces are counted separately.
Increasing degree or iteration cap beyond the selected budget is prohibited.
Retained long-window evidence shows degree1 can fail after halving h at fixed H,
degree2 can pass that bounded case, and degree4 can amplify a tiny nontangent
roundoff component inside Horner before a JVP. The original DB check stops that
candidate; no hidden projection or relaxed constraint repairs it.

## Oracles and deployment

Independent sequential original RK4 loops are validation/timing baselines only.
CPU executes the actual up/down tree. GPU executes independent time nodes in
batched spatial products/FFTs/solves; a host loop over tree levels or four RK
stages is permitted, a host loop over full microsteps is not temporal deployment.
Check all nodes/stages/E/RHS, phase, div B/wall flux, current/Ohm response,
input immutability and failed trial rollback. The pure field subproblem has no
standalone full ion energy-acceptance result. Count outer S as supplied/fixed,
field inner r_Z and spatial derivative/solve work separately.

At convergence, reuse selected-stage E/J/u records. Charge one final RHS at
the last supplied right-stage P and returned field endpoint to enforce domain/
constraint validity. This is not an ion endpoint commit or serial replay.
Every failed trial retains its attempted counters and residual evidence.
