# Structured-interior CLT-like wall implementation

Status: local implementation and coordinated discriminator in progress. The
intercept geometry is not yet an accepted complete time-evolution path.

The design authority is
[A050_CLT_INTERCEPT_WALL.md](/home/lxzeng/fomhd-design/mainline_design/numerics/algorithm_design/A1/A050_CLT_INTERCEPT_WALL.md)
and DECISIONS D074. This record does not change the massless quasineutral
Hall/CGL-B model, RK4/WR algorithms, or physical check thresholds.

## Source and version identity

The local `main` and `codex/a050-two-fluid-experiments` branches share commit
`73fb2a048813d622b643986cf6ea58dfcb82e139`. The former polygon work is retained
on `codex/a050-polygon-wall-research` at
`d323f875c76b7e6934af423dcf6b21653c3b4f47`. Its source, input and failed evidence
remain available in the preceding campaign. It is an unaccepted research
construction, not a release or a second active solver.

Current development is `codex/a050-clt-intercept-wall`, based on `73fb2a0`.
No automatic `main` merge is authorized. Remote updates require a successful
push; connection failures are retained in the execution logs.

## Fixed physical boundary

The target is a stationary impermeable ideal conductor. The independent
conditions are zero tangential electric field and zero normal number flux for
each species. Normal magnetic flux retains its initial data through the
Faraday boundary circulation; it is not reset to zero. Tangential slip is
allowed. Density, pressure, all velocity components and all magnetic components
are not fixed independently.

All near-wall consumers must share true wall points, normals and quadrature.
With compatible quasineutral current reconstruction, zero normal species flux
implies zero normal current. Electric tangential traces and free current
tangential traces remain distinct. A reaction enforcing impermeability must
use the same normal number-flux trace and its weighted adjoint.

## Implemented local building blocks

- [boundary_geometry.py](../src/fomhd/boundary_geometry.py) computes immutable
  directional intercepts, CLT-style classification metadata, existing-axis
  link endpoints, clipped logical dual metrics, and one shared wall quadrature.
  It preserves every original structured entity and physical component index.
  Its `staircase` option is an identity snapshot; its `clt_intercept` option is
  geometric data, not an evolution-method selector.
- [clt_boundary_coefficients.py](../tools/clt_boundary_coefficients.py) reproduces
  the actual source coefficient branches and tests their independent moments.
  Directional dropping is not silently activated in the physical RHS.
- [clt_boundary_patch_reference.py](../tools/clt_boundary_patch_reference.py) is
  an independent small CPU reference for physical line sampling, the local
  cylindrical Gram/lump, wall lifts, pressure-force sign, flux and power.
  It does not assemble a replacement global mesh or solver.
- [boundary_quadrature.py](../src/fomhd/boundary_quadrature.py) constructs a
  positive boundary-only scalar quadrature candidate with frozen core and
  explicitly checked moments. It is not the complete edge metric or traction
  closure. The diagonal edge-metric candidate has a separately retained
  incompatible physical-moment system.

The actually invoked external GS source, self-circle control, independent full
equation audit and formal old-path mapping are documented in
[VEQ_EQUILIBRIUM_INTERFACE.md](VEQ_EQUILIBRIUM_INTERFACE.md). External GS
convergence and continuous-model eligibility do not accept this spatial path.

Canonical pseudocode precedes each source. The
[CLT source map](CLT_BOUNDARY_SOURCE_MAP.md) records live calls, copied formulas,
source hashes, and differences between CLT physics and the selected FOMHD wall.
In particular, CLT fixed perturbations and active decay factors are not imported.

## Evidence interpretation

The geometry helper and coefficient tests passed independently. A fresh full
CPU regression of the old complete paths plus the first geometry/coefficient
additions reported 1461 passed, 31 skipped, and 3 subtests passed in 299.23 s.
The subsequently added local patch, quadrature and equilibrium tools were
checked separately. A later complete CPU run reported1539 passed,31 skipped,
and3 subtests passed in308.34s; the package source distribution and wheel built
successfully and contain both new boundary modules. These results do not
accept the proposed new physical boundary.

Superseded diagnostics are retained. A scalar `DJ G p` is a Laplacian, not the
pressure force. The actual pressure test uses `fluid.gradient_adjoint(p I)`
with boundary traction. Physical line averages must pass through the correct
T1/T2 representation; a manufactured nonzero wall circulation must be included.
Zero normal trace for toroidal velocity does not establish zero rigid-rotation
strain. Those tests have separate identities and acceptance scopes.

Current local probes and exact arrays are retained under
[/home/lxzeng/runs/fomhd/a050-clt-intercept__20261110T090707Z/](/home/lxzeng/runs/fomhd/a050-clt-intercept__20261110T090707Z/).
The inherited GPU authorization has 19434 single-GPU seconds remaining at the
start of this work; no new GPU job is needed before valid local acceptance.
Runtime configuration, restart identity and complete RK4/WR CPU/GPU validation
remain gated by that acceptance. They must not be presented as finished merely
because geometry data have fingerprints.

## Bounded failed local closures

The physical gradient is still the inherited factorization. Holding G/I fixed,
one oblique N12 patch jointly tested scalar quadrature, edge metric and a shared
real-wall normal trace against primal gradients, pressure traction and flux.
The diagonal family has497 unknowns and1713 equations; its augmented rank is
498 versus497 and its scaled maximum equality residual is1.37e-2. A symmetric
local poloidal metric family has1163 unknowns, rank1148 versus augmented1149,
and residual1.17e-3. Its selected minimum-change metric is also not SPD.

Independent actual-source pressure/gradient actions and saved-array least
squares reproduce these results. They reject the declared finite families,
not CLT, all structured closures, or the physical moment requirements. Frozen
core entries and local support were independently checked. The next required
construction concerns boundary gradient and auxiliary-trace factors rather
than more weight-only tuning; it does not require replacing the regular core.

The complete retained code/data, invalid intermediate diagnostics and final
receipts are under the execution root's `evidence/structured-worker/`. A
separate frozen replay passed21 tests. No failed local context was promoted to
complete evolution. The design-side
[report](/home/lxzeng/fomhd-design/mainline_design/implementation/A1/A050_CLT_VEQ_REPORT.md)
separates continuous inputs, initialization, physical local failures and the
unexecuted runtime/GPU/restart scope.
