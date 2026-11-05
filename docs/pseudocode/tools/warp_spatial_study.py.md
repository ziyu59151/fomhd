# `warp_spatial_study.py` pseudocode

## A0-TOOL-FIXED-SPATIAL-STUDY-01 - Purpose and ownership

Stable documentation entry: `A0-TOOL-FIXED-SPATIAL-STUDY`.

This tool measures two independent continuum references on the actual closed
RZ/Fourier production operators:

1. the pressure gradient implied by the discrete pressure-force adjoint;
2. the massless drift-Ohm field produced by the discrete constrained electron
   path.

The analytic reference is evaluated directly at the actual production
coordinates. It is never passed through `I`, `Q`, `H`, `G`, or `C`. The tool
records measurements only. It does not select a scientific tolerance, infer a
global convergence order from one grid, accept wall behavior, or establish a
finite-inertia momentum floor.

All coordinates are metres, pressure is pascals, pressure force is pascals per
metre, electric field is volts per metre, and density is inverse cubic metres.
The phase coordinate is radians. Geometry weights `M0` and `M1` are the
production cylindrical volume weights in cubic metres. Stored complex fields
use the production mode-major order; physical samples use sample-major order.

The CLI owns its output directory and source snapshot. Production operators own
their backend arrays. Host conversions occur only when persisting evidence or
when NumPy geometry metadata is required. CPU uses NumPy; CUDA uses the
production CuPy/Warp backend and must be synchronized at timing and observation
boundaries. There is no silent backend fallback.

## Fixed geometry

```text
FixedGeometry(nodes, shape) -> FixedGeometryRecord
  require shape in {rectangle, disk}
  require 4 <= nodes <= 32
  h = 1 / nodes
  R_centers = 1 + (0.5, 1.5, ..., nodes-0.5) h
  Z_centers = (0.5, 1.5, ..., nodes-0.5) h

  if rectangle:
    polygon = [(1,0), (2,0), (2,1), (1,1)]
    analytic target RZ area = 1
  else:
    polygon = 128 equally spaced vertices
      center + 0.45 (cos(theta), sin(theta))
    analytic target RZ area = pi 0.45^2

  geometry = production build_geometry(R_centers, Z_centers, polygon)
  require the grid and production mask describe that fixed physical domain
  record:
    active cell count
    cell-mask RZ area = active cell count h^2
    numerical toroidal volume = 2 pi sum(geometry.M0)
    signed fixed-polygon area
    analytic target area
    geometry diagnostics Hausdorff lower and upper bounds when present
```

`build_rectangle` is not used because its endpoint-center convention would
change the represented fixed domain.

## Analytic pressure reference

```text
AnalyticPressure(r, z, phi, profile)
  broadcast r, z, phi
  p0 = 0.4
  A = 0.04

  compact:
    q = ((r-1.5)^2 + (z-0.5)^2) / 0.25^2
    f = exp(1 - 1/(1-q)) where q < 1, otherwise 0
    df/dq = -f/(1-q)^2 where q < 1, otherwise 0
    grad_R(f) = (df/dq) 2(r-1.5)/0.25^2
    grad_Z(f) = (df/dq) 2(z-0.5)/0.25^2

  wall:
    s = sin(pi(r-1)) sin(pi z)
    g = 1 - ((r-1.5)^2 + (z-0.5)^2)/0.45^2
    f = (s g)^2
    df/dr = 2 s g (
      pi cos(pi(r-1)) sin(pi z) g
      - 2 s (r-1.5)/0.45^2)
    df/dz = 2 s g (
      pi sin(pi(r-1)) cos(pi z) g
      - 2 s (z-0.5)/0.45^2)

  variation = A f cos(phi)
  grad_R variation = A df/dr cos(phi)
  grad_Z variation = A df/dz cos(phi)
  grad_phi variation / r = -A f sin(phi)/r
  return variation and the three cylindrical gradient components
```

The returned derivatives are closed-form expressions. Finite differences are
test oracles only and are never called by the tool.

## Reference regions and weighted norms

```text
ReferenceBoundaryDistance(r, z, shape)
  rectangle: min(r-1, 2-r, z, 1-z)
  disk:      0.45 - hypot(r-1.5, z-0.5)

FixedRegionMasks(r, z, shape, width=0.15)
  distance = ReferenceBoundaryDistance(r, z, shape)
  all = true
  inner = distance > width
  wall = not inner
  return {all, inner, wall}
```

These regions use the fixed reference boundary. Distances to the staircase
production wall are recorded separately and never define the partition.

```text
WeightedNorms(actual, reference, entity_weights, regions,
              phi_weight=2 pi/N, unit, component_names)
  require matching [N, component, entity] physical-sample arrays
  require positive finite entity weights
  expand one-dimensional or component-specific mask to [component, entity]

  for each region:
    if no selected entity:
      return None metrics and reason "empty region"
    difference = actual - reference
    weighted_difference_square =
      sum(entity_weights * selected_mask * abs(difference)^2)
    weighted_reference_square =
      sum(entity_weights * selected_mask * abs(reference)^2)
    weighted_l2 = sqrt(phi_weight weighted_difference_square)
    weighted_rms = sqrt(
      weighted_difference_square / sum(selected entity weights))
    reference_weighted_l2 = sqrt(
      phi_weight weighted_reference_square)
    relative_weighted_l2 =
      weighted_l2 / reference_weighted_l2
      if reference_weighted_l2 > 0, otherwise None with a reason
    linf = max(abs(difference)) over selected values
    record phi samples, selected values, selected entities, positive-weight
    count, weight sum, component norms, and the same explicit quantities
```

The `2 pi/N` trapezoidal phase weight is applied exactly once. Pressure-force
vectors use `M0` repeated over the R/phi/Z components. Electric-field vectors
use `M1` in production order: R-face samples, cell-center toroidal samples, and
Z-face samples.

## Spatial case

```text
SpatialCase(context, shape, profile)
  require the build context has K=1, Nphi=5, and matched geometry/operators
  h = 1/n; phi = 2 pi (0, ..., Nphi-1)/Nphi
  ne = 1e16 on every cell and phase sample
  velocity u = 0
  B node-mode vector = 0.1 tesla B_phi in the center mode only
  sample B = production QB applied to that node vector and inverse transformed
    to cell samples

  n = full ne samples
  nf = production _component_flux(n, context)
  fluid = SimpleNamespace(n, nf, u)
  pressure_variation, grad_R, grad_phi_over_R, grad_Z =
    AnalyticPressure(cell coordinates, phi, profile)
  pressure_full = 0.4 + pressure_variation

  tensor_full = isotropic 3x3 tensor of pressure_full
  tensor_delta = isotropic 3x3 tensor of pressure_variation
  tensor_constant = isotropic 3x3 tensor of 0.4
  raw_force_full = production gradient_adjoint(fluid, tensor_full)
  raw_force_delta = production gradient_adjoint(fluid, tensor_delta)
  raw_force_constant = production gradient_adjoint(fluid, tensor_constant)

  projected_full_force =
    production _constraint_project(raw_force_full, n)
  constant_projected_force =
    production _constraint_project(raw_force_constant, n)
  projected_delta_force =
    projected_full_force - constant_projected_force

  exact_force = -(grad_R, grad_phi_over_R, grad_Z)
  e = -species.electron.charge

  electron = SimpleNamespace(pressure_force=raw_force_full)
  derived = SimpleNamespace(
    electron_density=n,
    electron_velocity=u,
    BN=B)
  E0_retained = production _massless_e0(electron, derived, context)
  correction_retained, correction_record =
    production _wall_correction(E0_retained, context)
  E_retained = E0_retained + correction_retained

  E0 = inverse Fourier E0_retained in staggered M1 order:
    R-face samples, cell-center phi samples, Z-face samples
  E = inverse Fourier E_retained in the same order
  E_correction = E - E0

  coordinates = concatenate(
    R-face coordinates,
    cell coordinates,
    Z-face coordinates)
  exact_variation, exact_grad_R, exact_grad_phi_over_R, exact_grad_Z =
    AnalyticPressure(coordinates, phi, profile)
  E_exact[0:nr] = -exact_grad_R[0:nr] / (e ne)
  E_exact[nr:nr+nc] = -exact_grad_phi_over_R[nr:nr+nc] / (e ne)
  E_exact[nr+nc:] = -exact_grad_Z[nr+nc:] / (e ne)

  For each correction Fourier mode:
    retain the production full original-row residual and right-hand side
  record original-row residual scale and relative residual
  classify that residual as an algebraic correction check, not continuum error

  verify E - E0 equals the inverse wall correction to floating-point roundoff
  verify the correction is confined to R-face and Z-face wall slots
  record inner correction and off-wall correction separately

  compute regional/component WeightedNorms for:
    raw_force_full (explicitly not continuum-pass eligible)
    raw_force_delta
    projected_full_force
    constant_projected_force
    projected_delta_force
    E0 against E_exact
    E against E_exact
    E_correction against zero
  the aggregate E norms use the flat production M1 edge order; component
  norms independently use the R-face, cell, and Z-face slices

  return host metadata plus plotting arrays:
    physical coordinates, reference distances, numerical-wall distances,
    region masks, pressure, all force variants and errors,
    E0/E/correction and errors, correction rows, masks, and ranks
```

The raw full force contains the discrete weak boundary lift from the actual
adjoint. It is retained and labelled, but it is not described as pointwise
`-grad(p)` and is not eligible for a continuum pass/fail interpretation.

## Context and CLI

```text
BuildContext(nodes, shape, device, memory_cap=8 GiB)
  normalized_device = existing require_device(device)
  p = existing massless fixture _parameters(HallCase(kdi=1))
  p.K = 1; p.Nphi = 5; p.inner_distance = 0.15
  p.memory_limit = p.device_memory_limit = memory_cap
  backend = production Backend(normalized_device)
  fixed = FixedGeometry(nodes, shape)
  operators = production build_operators(
    fixed.geometry, K=1,
    OperatorOptions(rank_rtol=p.rank_rtol,
                    memory_cap_bytes=memory_cap),
    backend)
  return production Context

main:
  parse --output, --device {cpu,cuda}, comma/space-separated
    --nodes default 6,8,12,16
    --shapes default rectangle,disk
    --profiles default compact,wall
    --memory-cap-gib default 8
  validate limits and require a new or empty output directory
  normalize CUDA through require_device before any backend construction
  create output directory
  snapshot before measurements:
    every Python file in the actual imported fomhd package
    actual imported fixture files
    own tool, test, and pseudocode
  hash each copied file and origin; write SOURCE_MANIFEST.json

  for each nodes, shape:
    build one context and synchronization point
    for each profile:
      try SpatialCase and immediately write label.npz
      on Failure or Exception:
        retain type, code, message, traceback, and elapsed time
        continue the remaining profile/case measurements

  report:
    schema/backend/device/Slurm job ID
    exact imported package and source roots
    source-snapshot hashes
    software/runtime versions and timings
    fixed geometry records
    per-case measured regional/component norms and array file hashes
    case failures
    scope inclusions/exclusions
    status "measured" only when every requested case completed;
      otherwise "failed"
    scientific acceptance "not_assessed"

  write strict JSON report without NaN/Infinity tokens
  return zero only for a completed measurement run; any retained case failure
  returns nonzero
```

Successful execution means the requested quantities were measured. It does not
mean the measured continuum errors are acceptable.

The full massless production stage uses a compatible harmonic magnetic field normalized to fixed toroidal flux 0.1 log(2) Wb. Electron velocity is derived by the production constraint, not prescribed independently. Store every hard-stage check outcome. Weighted RMS divides by the complete phase and entity weight; weighted L2 has field units times m^(3/2). CUDA observations use explicit to_host. Reports are atomically refreshed after each case, and failures retain raw arrays/checks.
