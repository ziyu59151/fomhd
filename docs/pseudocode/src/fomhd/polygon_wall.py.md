# Polygon actual-normal wall extraction and checks

Status: bounded implementation pseudocode for `polygon_dual_v1`. The frozen
design authority is `A050_REAL_WALL.md`. This overlay changes wall extraction
and wall diagnostics only; it does not introduce a new physical model.

## Dispatch and units

```
kind = geometry.kind if present else op.geometry_kind if present
          else "staircase_v1"

if kind == "polygon_dual_v1":
    polygon path
else:
    literal legacy r_face/z_face path
```

State and geometry stay in SI. Complex field state remains mode-major
`complex128`; only physical samples used by wall diagnostics are converted to
real FP64. `S`, `I`, `M0`, `M1`, and the wall frame are backend arrays or
verified host FP64 metadata as supplied by the selected backend.

For a polygon wall `w`, require:

```
w.component = "plane_face"
w.sign = 1
0 <= w.face < Nf
n2 = geometry.face_normals[w.face]
t2 = geometry.face_tangents[w.face] = (-n2.Z, n2.R)
||n2|| = 1
radius = geometry.faces.R[w.face]
ell = geometry.face_lengths[w.face]
w.area_per_radian = radius * ell
```

The physical frame is

```
n      = (n_R, 0, n_Z)       outward normal
t      = (t_R, 0, t_Z)       poloidal tangent
ephi   = (0, 1, 0)
radius = R_f
```

There is no axis direction `0` or `2` in this path.

## Native extraction

The polygon electric layout is `E=[E_n(Nf), E_phi(Nc)]`. A wall extracts its
single global face coordinate:

```
edge(w) = w.face
S[w, edge(w)] = w.sign = 1
```

`wall_cache._signed_wall_extraction` uses this row for the constrained wall
system. `massless._wall_extractions`, `massless_field._wall_flux_samples`, and
`massless_checks._wall_flux_samples` use the same global face index. Their
legacy r/z branches remain unchanged.

## Actual-face trace

For each wall and each retained toroidal sample, let `S_f` be the actual scalar
face-interpolation row `op.S[w.face]`. The sampled quantities are

```
n_f        = S_f n
p_perp,f   = S_f p_perp
p_par,f    = S_f p_par
m_c,f      = S_f m_c                    for c in (R,phi,Z)
u_f        = (m_R,f,m_phi,f,m_Z,f) / (mass * n_f)
F_f        = w.sign * fluid.flux[:, edge(w)]
```

The pressure and all three momentum components use the same row. The velocity
is not formed by selecting one component of `I`; no requirement that `u_R` or
`u_Z` vanish is imposed.

The stage performs each required inverse transform once: electric samples,
magnetic samples, and one physical momentum sample array per species. Wall
extraction then uses only the appropriate `S_f` row. No state-wide host copy is
introduced; host conversion is limited to immutable two-component geometry
metadata and scalar diagnostic reductions.

The magnetic layout is `B=[B_phi(Nv), B_t(Nf)]`. At the face endpoints
`v0,v1`,

```
Bwall = B_t,f t + 0.5 (B_phi,v0 + B_phi,v1) ephi
0     = Bwall . n                         relative closed-wall input
```

The electric layout is `E=[E_n(Nf), E_phi(Nc)]`. For an ideal wall,

```
E_wall = E_n,w n
0      = E_wall . t = E_wall . ephi
```

Each trace carries `normal3=n`, `tangent3=t`, and `radius`, plus its wall
frame, face flux, sampled `Bwall`, and sampled `E_wall`.

## Boundary accounting

Let `A_w = 2*pi*w.area_per_radian/N` be the full toroidal area represented by
one sample. For each sample and species, use the one shared face flux:

```
un_flux = w.sign * F_f / n_f
number += A_w * F_f
```

Ion kinetic transport is

```
kinetic_i += A_w * 0.5*m_i*n_i*||u_i||^2 * un_flux
```

Electron kinetic transport remains absent. Pressure/parallel transport uses
the same wall face and normal, with pressure values reconstructed by `S_f`.
Total pressure work uses

```
u_n = u . n
p_work = p_perp u_n
       + (p_par-p_perp) (bhat . u) (bhat . n)
```

for the electron CGL pressure, and `p_i u_n` for the isotropic ion pressure.
The old axis path retains its formulas and tests.

## Traction, torque, and Maxwell stress

For `b=Bwall`, `u=u_i`, `n=n`, the outward ion/electron traction is

```
Tmaxwell.n = (0.5*||b||^2*n - (b.n)*b) / mu0
tion_ram   = m_i*n_i*u*(u.n)
tion_pres  = p_i*n
te_pres    = p_e_perp*n
           + (p_e_par-p_e_perp)*bhat*(bhat.n)
outward    = Tmaxwell.n + tion_ram + tion_pres + te_pres
```

All frame products are physical dot products. The axial torque density of a
wall traction is `radius * outward_phi`; it is integrated with the same face
area `A_w`. No Cartesian axis shortcut is used for polygon walls.

Because `b.n=0` and `E_wall` is normal for the ideal wall,

```
Poynting_normal = n . (E_wall x Bwall) = 0
```

The fixed-density wall projection remains orthogonal under its declared
weight:

```
winv = mass*n_i/M0
sum_cells (value_i . reaction_i) / winv_i = 0
```

This is an algebraic reaction-work identity, not an extra damping or energy
source.

The retained magnetic alias check measures Bphi on vertices and Bt on faces
with the existing `alias_B` budget key. The second block uses the existing
`r_faces` topology map internally, then emits the component name `tangent`
and records explicit layout metadata. The zero-width normal interval is
omitted, so no diagnostic claims a physical `B_Z` component.

## WR caller seam

`massless._signed_wall_extraction`, its `_wall_extractions` delegate, and
`massless_field._wall_flux_samples` use the corrected native helper names.
`multirate_wr_checks._traction_components`, `_momentum_integrals`, and
`_boundary_summands` are polygon-only adapted to consume `normal3`,
`tangent3`, and `radius` from the shared trace. Their `r_face/z_face` formulas
are retained behind the geometry-kind branch. `accumulate_mixed_interval`
continues to use the same `massless_checks._wall_trace_data`; no WR schema or
history ownership changes are introduced.

## Deliberate non-scope

`field_batch` T2E graph support is not implemented and is explicitly reported
as unsupported. Mesh/operator construction, fluid spatial formulas, time
schemes, boundary constraints, RHS offsets, and GPU execution remain outside
this overlay. Full polygon RHS and new-mesh integration belong to the parent.

## CPU tests

`tests/warp/test_polygon_wall.py` will use an independent diagonal-face
fixture with different R/Z normals, variable face-normal fields, physical
`B_phi` and `B_t`, pressure anisotropy, and nontrivial normal/tangential
momentum. It will verify extraction indices, alias-layout metadata, physical
trace reconstruction, signed flux, Poynting zero, Maxwell/total traction and
axial torque sign, normal-momentum isolation, weighted reaction work, and
unchanged legacy extraction. Tests run only on CPU with one thread and no
cache provider.


# Batched polygon-wall observations

Status: selected read-only diagnostic optimization. Physical source, flux,
current, constraints, time equations, and hard gates are unchanged.

## Device construction

For all wall faces at once, build

```text
density_s[wall,sample] =
    momentum_trace_s[wall,sample] =
        S_boundary component action
Bwall = Bt*t + 0.5(Bphi,v0+Bphi,v1)*ephi
Ewall = En*n
```

All wall normals are converted to one backend array `normal_backend` with
shape `(Nwall,3)`. `electric_wall` is constructed once as

```text
electric_wall = electric_normal[:,:,None] * normal_backend[None,:,:]
```

and has shape `(Nphi,Nwall,3)`. Per-wall records store zero-copy backend views
of this array and of the same `normal_backend`; no per-wall array conversion or
outer-product launch occurs in the view-building loop.

`S_boundary` contains the existing `op.S[wall.face]` rows. Momentum remains the
actual inverse-transformed state electron compatibility momentum. No consumer
substitutes `electron.u`. Density positivity is checked for every wall and
sample before any velocity division.

Before dividing a species momentum trace, its mass must satisfy
`isfinite(mass) and mass > 0`. Failure raises `Failure("WallTrace",
"wall species mass must be positive")` before any host export or diagnostic
velocity is formed, matching the original scalar routine.

The returned sequence has the same per-wall device view fields as before:
`B`, density, velocity, pressures, normal flux, electric normal/wall, normal,
tangent, radius, area, and wall/frame identity.

## Declared host boundary

The only export packs diagnostic wall observations as

```text
obs.shape = (Nphi, Nwall, 18)
channels = Bwall[R,phi,Z], En,
           density[i,e],
           velocity[i,e][R,phi,Z],
           pressure_perp[i,e], pressure_parallel[i,e],
           normal_flux[i,e]
```

One explicit `context.compute.to_host(obs)` call is the read-only wall
observation boundary. Cell fields and stage arrays are never exported as
volume arrays. NumPy then owns the wall diagnostic arithmetic. Host wall
values are not sent back through CuPy per wall or per sample.

Fixed geometry (`normal3`, `tangent3`, wall radius, full toroidal area) remains
host geometry metadata. Volume momentum/reaction integrals remain backend
reductions.

## NumPy diagnostic consumers

After export:

- boundary number, kinetic, pressure, and energy sums use NumPy;
- wall transport/electromagnetic/total traction is computed once with NumPy;
- traction force and axial torque integrals use NumPy wall loops;
- mixed-WR wall summands and role-specific traction use the same host arrays.

The resulting values replace only numerically equivalent diagnostic wall
loops. Backend boundary extraction, flux, current, constraints, and their hard
gates continue to use actual backend fields.

## Identity and lifetime

A trace batch records the immutable source identity: state object/field arrays,
stage/cache arrays, geometry/operator objects, sample count, state clock, and
stage identity. `stage.boundary._trace_data` is reused only when this identity
matches. There is no global state cache.

Geometry packing is temporary for each batch. No fixed-index cache is added.

## Verification

Focused CPU tests at Nphi=1 and Nphi=5 compare every trace channel and every
boundary/traction/power integral against an independent scalar loop using the
same physical formulas. The fixture includes multiple oblique walls, nonzero
flows, anisotropic pressures, and nonzero Bphi/Bt.

A transfer spy requires exactly one export and checks
`obs.shape == (Nphi,Nwall,18)`. It rejects exports whose last dimension is a
cell/volume field count.
