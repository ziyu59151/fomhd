# Few-node multirate WR window

Status: bounded implementation pseudocode for the selected A1-WR-050
few-node history-integral method. This is not numerical acceptance or a
production ledger.

Source mirror: `src/fomhd/time_multirate_wr.py`.
Design authority:
`mainline_design/numerics/algorithm_design/A1/A1_WR_050.md`.

## A1-PY-WR-01 - Roles, layouts and execution contract

```text
solve_window(p0,z0,H,M,s,transport,field; xp,p_scales,...)

inputs:
  P = (n_i, m_i*n_i*u_i, p_i)
    n_i       [m^-3]
    m_i*n_i*u_i [kg*m^-2*s^-1]
    p_i       [Pa]
  Z = (p_e_perp,p_e_parallel,B)
    pressures [Pa]
    B         [T in the current physical implementation]
  p_scales    finite-positive physical reference, broadcastable to P
  H           finite-positive macro duration [s]
  M           positive non-bool integer field microstep count
  h = H/M     finite-positive microstep [s]
  s           coarse Gauss-Lobatto node count, exactly3 or5
  theta       dimensionless macro fraction in [0,1]

caller layout:
  P and Z are explicit xp.complex128 arrays in the physical implementation's
  existing leading component/mode/cell layout. This generic layer treats all
  dimensions after the coarse-node axis as independent state components. It
  does not relayout, flatten, project or assign physics to components.
  p0,z0 and callback outputs must have the same shape as their declared role.

adapter roles:
  transport(theta,P,Z,identity) -> (A,record)
    A = coarse transport/convective/pressure contribution with zero density
        supplied by parent physics.
  field(theta,P,Z,identity) -> (G,C,record)
    G = field RK4 rate.
    C = grid_response containing density continuity plus projected Lorentz.
        Generic code does not assume C is zero outside momentum.
  records are caller-owned real physics objects; this module borrows them.

identity:
  role in {tangent_initialization,field_rk4,transport_coarse}
  includes outer iteration/OLD version, microstep, RK stage or coarse node,
  and theta. The two middle field stages have distinct identities and Z states.

backend:
  xp is NumPy or CuPy for the whole call. Retained P,Z and callback state stay
  as xp.complex128 arrays on that backend. No CPU retry, fallback or implicit
  backend switch is allowed.
  The module uses the caller's selected device/current stream and never creates
  or switches one. Caller adapter/context lifetime owns device and stream.
  Only small CPU Lobatto/Lagrange/Q/weight coefficients and scalar residual
  observations cross the backend boundary. No per-stage whole-state host copy.

ownership:
  p0,z0 are read-only and never mutated.
  Result arrays are owned by the result; records remain borrowed.
  Failure keeps device arrays and the original adapter failure evidence for
  diagnostics. No accepted state or cumulative ledger is committed here.
```

## A1-PY-WR-02 - Coarse rule and initialization

```text
require controls finite/positive and M,max_iterations non-bool positive integers
require s in {3,5}, rtol positive, 0<relaxation<=1, valid initial mode
require p0,z0 complex128 xp arrays, finite, matching callback output shapes
require p_scales real floating xp array, finite positive and broadcastable to p0

CPU-only coefficient construction cached once per s:
  interior roots = roots of derivative(Legendre_(s-1)) restricted to [-1,1]
  require each root explicitly real within a floating roundoff bound
  nodes = [0, sort((real(interior)+1)/2), 1], stored read-only
  ell_l(theta) = product_(k != l)(theta-node_k)/(node_l-node_k)
  Q_il = integral_0^node_i ell_l(theta) dtheta using polynomial integration
  precompute every field-stage interpolation basis and coarse dense weights

OLD version v=0:
  if initial == tangent:
    identity = (-1,None,None,node=0,tangent_initialization,theta=0)
    A0 = TRY transport(0,p0,z0,identity); count callback
    G0,C0 = TRY field(0,p0,z0,identity); count callback and grid response
    F0 = A0+C0
    P_v[l] = p0+H*node_l*F0
  if initial == constant:
    P_v[l] = p0
    no tangent callback
```

## A1-PY-WR-03 - One OLD-version multirate sweep

```text
attempted_sweeps += 1
old = immutable P_v snapshot
build p_stage[theta=(j+a_r)/M] = Lagrange(old), a=(0,1/2,1/2,1)
create partial scan accumulator containing only completed stage evidence
z0_field = accepted z0
C_prefix = 0

for j in [0,M):
  z_left = z0_field
  for r in [0,4):
    z_r = z_left                                      if r=0
          z_left+(h/2)*G_(j,0)                        if r=1
          z_left+(h/2)*G_(j,1)                        if r=2
          z_left+h*G_(j,2)                            if r=3
    identity = (v,j,r,coarse_node=None,field_rk4,(j+a_r)/M)
    publish attempted identity, P stage and Z stage
    count field/grid-response callback before invoking it
    G_jr,C_jr,record = TRY field(theta,p_stage,z_r,identity)
    require complex128 xp shape/finiteness
    after success only, append P,Z,G,C,identity,record to partial evidence
  z0_field = z_left+h*(G_j0+2*G_j1+2*G_j2+G_j3)/6
  C_prefix += h*(C_j0+2*C_j1+2*C_j2+C_j3)/6

for each coarse node c_i:
  j = min(floor(c_i*M),M-1); x=c_i*M-j
  dense b(x) = declared b1,b2,b3,b4, with b(1)=(1,2,2,1)/6
  Z_coarse[i] = Z_j_left+h*sum_r b_r(x)*G_jr
  I_C[i] = completed C_prefix before j + h*sum_r b_r(x)*C_jr

for each coarse node:
  identity = (v,containing_micro,None,i,transport_coarse,c_i)
  count transport before invoking it
  A_i = TRY transport(c_i,old[i],Z_coarse[i],identity)

mapped[0] = p0
mapped[i] = p0+H*sum_l Q_il*A_l+I_C[i]
```

## A1-PY-WR-04 - Defect and acceptance transaction

```text
raw_defect = old-mapped
for each independent component k:
  scale_k = max(p_scales_k, max_abs(old_k), max_abs(mapped_k))
  normalized_k = abs(raw_defect_k)/(rtol*scale_k)
  require every normalized node/component ratio <= 1

converged = max(normalized_k) <= 1
append scalar raw/scaled/normalized norms and endpoint defects to history
only after this raw OLD-vs-mapped test:
  if converged: accept old, not mapped
  else if attempts remain:
    next = old+relaxation*(mapped-old)
    next[0] = p0
    increment OLD version
    discard the old trial field arrays and records
  else:
    raise MultirateConvergence with current full evidence

accepted result:
  P = old, P_endpoint = old[-1]
  Z_endpoint = final scan endpoint, Z_coarse
  mapped nodes/endpoint only as residual evidence
  final-sweep fine P,Z,G,C, identities, records and weights
  coarse P,Z,A,C integrals,Q and dense segment descriptors
  iterations, final OLD input version, ion_inner_iterations=0
```

## A1-PY-WR-05 - Failure and cost contract

```text
Failure preserves any original code,message,checks,stage and adapter arrays.
solver adds:
  attempted_sweeps, completed_residual_sweeps
  completed residual_history (scalar only)
  attempted/current microstep and RK stage identity
  completed fine P,Z,G,C, identities and records for the current partial sweep
  attempted current P and Z inputs; never uninitialized xp.empty slots
  actual callback/grid-response work up to the failure point

partial ownership:
  append owned copies of each completed callback G and C array
  a producer may reuse one scratch G/C buffer on later stages
  retain producer record objects by reference without mutation
  retain the frozen current attempted identity and attempted P/Z inputs

coarse transport partial ownership:
  before each transport call record current coarse identity and copies of
    attempted P/Z state
  after success append an owned copy of A plus the borrowed producer record
  on coarse-node failure expose only completed A/identities/records and the
    current attempted identity/state; never expose empty coarse_A slots
  preserve original physical Failure stage/checks/partial arrays by setdefault
  the failed callback is already charged, so attempted/work counts include it

attempted_sweeps counts an invoked sweep even when no residual was completed.
work:
  tangent: one transport and one field/grid-response callback
  each sweep: s transport callbacks and 4*M field/grid-response callbacks
  rk4_stages counts 4*M field invocations, excluding tangent initialization
  complete-ion, alias, replay and parent endpoint counts are adapter-owned
  and remain None/uninferred in this generic result

No timing, production work reduction, complete-ion count, checkpoint/restart,
or parent acceptance is inferred. GPU execution remains a parent-owned task.
```
