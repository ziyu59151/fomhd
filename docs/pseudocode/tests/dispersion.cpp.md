# A0 local periodic dispersion test

Status: implementation-level pseudocode synchronized with
`tests/periodic_fixture.hpp` and `tests/dispersion.cpp`.

## A0-VERIFY-DISPERSION — Scope

This test measures the local linear response of the production A0
`evaluate_stage` path against an independently assembled 14 by 14 continuum
matrix. It is model-level theoretical confirmation in the local Cartesian
periodic limit. It is not experimental validation, a Yee accuracy test, or an
alternative production boundary.

The fixture deliberately uses:

- an odd real phase chain with 9 or 11 points;
- `K=0` and `Nphi=1`;
- one spectral real skew derivative matrix `D` whose first positive mode has
  eigenvalue `i*k0`;
- `DR=sin(theta)*D` and `DZ=cos(theta)*D`;
- equal entity counts and positive unit quadrature weights for cells, radial
  faces, axial faces, and vertices;
- `I=H=QB=identity(3*n)`;
- empty wall rows;
- all sample radii equal to `1e30 m`.

The last item makes retained cylindrical `1/R` and `1/R^2` response terms
negligible at double precision. This is the declared local Cartesian-limit
assumption, not a change to production geometry.

For an odd `N`, the unscaled real spectral derivative is assembled from paired
sine entries:

```text
D[row,column] = (4*pi/N^2) *
  sum_{m=1}^{(N-1)/2} m*sin(2*pi*m*(column-row)/N)
```

The opposite entry is assigned its exact negative and the diagonal is exact
zero. `D` is then multiplied only by the declared physical component factor.

## Adapter algebra gate

Run the following before any production stage evaluation:

```text
BuildPeriodicAdapter(points,k0,theta):
  construct D and DR,DZ
  construct G=[DR;0;DZ]
  construct C=[[0,-DZ,0],[DZ,0,-DR],[0,DR,0]]
  construct DB=[DR,0,DZ]
  construct DJ=-adjoint(G)
  construct Cdag=adjoint(C)
  Lphi=-DJ*G plus a rank-one constant-potential gauge
  require I=H=QB=identity
  require no walls and zero-row T/T_full
  require DJ=independent-row Bc source

CheckPeriodicAdapter:
  require all imaginary parts of D are zero
  require D+D^T=0 and D*1=0
  q_j=exp(i*2*pi*j/points)
  require DR*q=i*k_R*q and DZ*q=i*k_Z*q
  for each Cartesian basis vector e_component*q:
    require C*(e_component*q)=CartesianCurl(DR,DZ,e_component*q)
  require max|C*G|=0
  require max|DB*C|=0
  require Cdag=adjoint(C)
  require DJ=-adjoint(G)
  if any residual exceeds the algebra roundoff budget:
    fail before constructing a production stage
```

## Independent continuum matrix

Use perturbation order

```text
[n_i, mom_i_R, mom_i_phi, mom_i_Z, p_i,
 n_e, mom_e_R, mom_e_phi, mom_e_Z, p_perp_e, p_parallel_e,
 B_R, B_phi, B_Z]
```

For each basis column:

```text
u_s=delta_mom_s/(m_s*n0)
rho=sum(q_s*delta_n_s)
EL=-i*k*rho/(epsilon0*|k|^2)

Pi=delta_p_i*I
delta_b=(delta_B-b*(b dot delta_B))/B0
Pe=delta_p_perp*I
   +(delta_p_parallel-delta_p_perp)*b*b
   +(p_parallel0-p_perp0)*(delta_b*b+b*delta_b)

F_s=-i*Pe_s*k+q_s*n0*(u_s cross B0)
fJ0=sum(q_s/m_s*F_s)
alpha=sum(q_s^2*n0/m_s)
PT=I-k*k^T/|k|^2
ET=-PT*fJ0/(|k|^2/mu0+alpha)
E=EL+ET

delta_mom_dot_s=F_s+q_s*n0*E
delta_n_dot_s=-i*k dot delta_mom_s/m_s
p_i_dot=-gamma_i*p_i0*i*k dot u_i
p_perp_dot=p_perp0*(i*k_parallel*u_e_parallel-2*i*k dot u_e)
p_parallel_dot=-p_parallel0*(i*k dot u_e+2*i*k_parallel*u_e_parallel)
B_dot=-i*k cross ET
```

Normalize the matrix componentwise with the fixture's declared state scales
and rate scales:

```text
A_normalized=diag(rate_scale)^-1*A_physical*diag(state_scale)
```

The normalized matrix is `D^-1*L_physical*D/reference_frequency`; its
eigenvalues are dimensionless. Before MKL/LAPACKE `zgeev`, apply an invertible
orthogonal change of coordinates: rotate ion/electron densities to their sum
and difference, and rotate momentum and magnetic vector components to a
wave-aligned basis. This separates the stiff electrostatic charge response
from neutral density and transverse components without projecting out any
state or imposing field constraints. Accumulate the similarity transform in
extended precision, solve the full 14 by 14 problem, then map right
eigenvectors back to the original normalized coordinates. Evaluate residuals
and field constraints against the original matrix and original coordinates.
Retain this unrestricted 14-mode spectrum as diagnostic evidence. The physical
eigenvalue gate acts on the dynamically admissible subspace: derive three
rows `K` from `k dot B=0` and the two transverse components of
`i*k cross B-mu0*PT*J=0`, normalize each row, and use a full complex SVD to
construct an orthonormal basis `Q` of `ker(K)`. Require exactly three
independent constraints, `K*Q` and `Q^*Q-I` at roundoff. In the conditioned
coordinates diagonalize `Q^*A*Q` (11 by 11) for both independently formed
matrices and lift its vectors back to the original 14 coordinates. No mode is
discarded according to frequency or growth. The three thermodynamic zero
modes remain in the admissible eigenvalue gate.

Separately gate invariant preservation over every column of each original
14 by 14 matrix: compute `K*A` using extended accumulation and normalize each
entry by `max(1,sum_l abs(K[row,l])*abs(A[l,col]))`. Use the existing `1e-8`
field-constraint budget. This prevents projection from hiding broken
production constraints. Report unrestricted spectra, admissible spectra,
orthogonality, nullspace, and invariant residuals separately. The unrestricted
near-zero eigenvalues are retained as evidence, but are not treated as
physical stability predictions when they violate the field constraints.
Multiply matched eigenvalues by `reference_frequency` before reporting SI
growth rates, angular frequencies, and ordinary frequencies. Physical
eigenvector polarizations use the same state scales.

Match analytic and production eigenvalues with a one-to-one minimum-cost
assignment. For each matched branch, require

```text
abs(production-analytic) <=
  eigen_abs_tolerance + eigen_relative_tolerance*abs(analytic)
```

This gate is mode-local and does not normalize all branches by the fastest
eigenvalue. Its relative term has no unit floor. Assignment costs are divided
by each reference mode's allowed error before one-to-one matching.

Before production evaluation, run literal acceptance regressions: preserve a
nonzero slow branch under permutation and degeneracy; reject its replacement
by zero; and distinguish the true maximum absolute matrix discrepancy from
the entry with maximum componentwise normalized discrepancy. `--preflight`
runs these regressions and all adapter algebra checks without evaluating a
production stage.

## Production response

For each column:

```text
x_plus = uniform_background + amplitude*scale[column]*cos(s)
x_minus= uniform_background - amplitude*scale[column]*cos(s)
rhs_plus = evaluate_stage(x_plus, current tag).rhs
rhs_minus= evaluate_stage(x_minus,current tag).rhs

for each row component y:
  c_plus = (1/N)*sum_j exp(-i*s_j)*y_plus(j)
  c_minus= (1/N)*sum_j exp(-i*s_j)*y_minus(j)
  derivative=(c_plus-c_minus)/(2*amplitude*(1/2))
  production[row,column]=derivative/rate_scale[row]
```

Use two relative amplitudes and require the normalized matrix difference to
remain below the declared linearity budget; compare their arithmetic mean
against the independent continuum matrix. All hard production stage checks
must run and pass. Empty-wall diagnostics are applicable only when production
reports them unavailable and non-hard for an empty wall set.

The test compares:

- the full normalized production and analytic matrices;
- the matrix discrepancy maximum, RMS, and worst row/column;
- analytic and production eigenvalues;
- analytic eigenvector residuals and physical polarizations;
- amplitude sensitivity.

For each selected CGL branch, compute the normalized magnetic divergence
`i*k dot B` and transverse Ampere residual `i*k cross B - mu0*PT*J` from the
physical eigenvector. Divide divergence by `|k|*||B||` and Ampere by
`max(|k|*||B||,mu0*||PT*J||)`. Select the fastest-growing nonzero branch for an
unstable case, and the strongest normalized magnetic polarization for the
stable control. Gate both the analytic vector and its matched production
vector, and report their normalized values and physical growth/frequency.
Do not normalize a divergence residual by itself. Eigenvectors have unit norm
in the declared normalized state coordinates; multiplying componentwise by
the state scales gives SI polarization per unit normalized amplitude.

## Negative controls

Do not compare a matrix with itself. Require:

1. a reference matrix with reversed Lorentz force fails comparison with the
   actual production matrix through the same normalized matrix gate;
2. an anisotropic case with reversed electron stress fails comparison with the
   actual production matrix through that same gate;
3. production evaluated with a sign-reversed curl operator differs from the
   accepted analytic reference.

The reversed-Lorentz and reversed-stress references must each be compared
against the actual production matrix through the same normalized matrix gate,
not only against the accepted reference. A second eigenvalue control zeros all
slow production branches and requires the mode-local eigen gate to reject the
result.

Only the third control checks an actual production operator mutation.
The preflight also injects a nonzero divergence derivative into a matrix and
requires the original-matrix invariant gate to reject it. It verifies the
oblique continuum growing eigenvector on the admissible subspace.

## Cases

Run warm parallel, warm perpendicular, and warm oblique states with real SI ion
and electron masses. Run a documented low-k CGL family:

- stable control near isotropic electron pressure;
- parallel-pressure branch with `p_parallel > p_perp`;
- oblique perpendicular-pressure branch with `p_perp > p_parallel`.

The oblique unstable fixture uses `p_perp=2 Pa`, `p_parallel=0.2 Pa`, and
`theta=1.2 rad`. Its isotropic counterpart is a separate stable control.
For rationale, the long-wavelength CGL compressive stiffness in units of Pa is

```text
T=B0^2/mu0+p_perp-p_parallel
K_RR=sin(theta)^2*(gamma_i*p_i+2*p_perp+B0^2/mu0)+cos(theta)^2*T
K_RZ=sin(theta)*cos(theta)*(gamma_i*p_i+p_perp)
K_ZZ=cos(theta)^2*(gamma_i*p_i+3*p_parallel)
```

The new mirror-like fixture has positive shear tension and a negative
compressive stiffness eigenvalue (`-0.02145417 Pa`); the earlier candidate
`p_parallel=0.5 Pa,theta=0.65` had positive compressive stiffness and is not
used to substantiate a CGL mirror-like branch. This is a physical fixture
correction; all acceptance tolerances remain fixed.

Report CGL growth and polarization. Do not claim kinetic firehose or mirror
thresholds: A0 has neither kinetic closure nor finite-Larmor-radius physics.

## Output and failure

Emit JSON containing the adapter checks, case parameters, tolerances, measured
discrepancies, worst matrix entries, eigen residuals, growth/frequency and
polarization data, hard-check counts, and an overall `passed` value. Exit zero
only when every matrix, linearity, eigen, stability-classification, and sign
control criterion passes, along with full-matrix invariant preservation and
lifted eigenvector residuals. Output has 14 unrestricted eigenvalues as
diagnostics and 11 admissible eigenvalues as the physical gate. The local
phase-chain test is synchronous, host-resident FP64, owns its state/operator
storage, calls production stages with unique trial tags, and throws on any
unavailable/failed hard stage check or failed numerical solve. No accepted
time-step transaction or device execution is performed.
