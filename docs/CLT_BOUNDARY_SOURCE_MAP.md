# CLT boundary-source map

## Scope and identity

This is a static reading of two shipped Fortran sources. No CLT build, run, mesh input
decode, or numerical run was performed. Line ranges below are the current source ranges
in the supplied files.

| Input | SHA256 |
|---|---|
| `resistive_kink_benchmark/tokRZ_mpi_pll_acc_merged_v20.f90` | `23d542c10734664e660ecb2e5330589ec2c9efc830a7fd60147ee1a8b8fb9069` |
| `tearingmode_benchmark/tokRZ_mpi_pll_acc_merged.f90` | `ba1924535ecd3675ba2d36ee1be9b738ff42d967dbaf4e54d30a64b991e596a3` |
| CLT reference `README.md` | `fe0c1969222ff95670845e534a7ef8f605ae8860f431e9b37bc86a78cfb11e15` |
| `A050_CLT_READING_2026-11-10.md` | `fe3e4fdc5ab44fe1782bb173d529383fb218b32ee37f82fe23dc78004d804045` |

FOMHD baseline context was read at pinned commit
`73fb2a048813d622b643986cf6ea58dfcb82e139` (branch
`codex/a050-two-fluid-experiments`, which was dirty). No uncommitted FOMHD source was
used as evidence. The tracked baseline identifies `staircase_relative_v1`; the source
mapping here does not select or implement a replacement geometry.

## Live call chain

Both sources use the same driver shape. The named `*_cut_cell` pair is compiled but not
called: the main-loop call is commented out and `stepon_openacc` is called instead.

| Stage | Resistive-kink lines | Tearing-mode lines | Actual contract |
|---|---:|---:|---|
| `program tokrz -> input -> initia` | 495-650, 1073-1370, 1511-2432 | 494-649, 1069-1366, 1507-2428 | Read equilibrium, build/read LCFS samples, grid, and state. |
| Equilibrium contour source | `read_nova` 6499-6875; `map_nova` 5125-5752; `last_grid` 6083-6497 | `read_pgfile` 25889-26190, `openacc=.true.` binary branch 25996-26017 | Kink constructs/interpolates flux surfaces from NOVA data. Tearing reads `xxs5,zzs5` from `eq_efit_all.bin`; no contour construction is performed in this source path. |
| Contour/grid intersections | `find_bnd_grd` 20401-20586 | 20328-20511 | Sort supplied contour samples by poloidal angle, remove duplicate angles, intersect contour edges with each `zzt` and `xxt` grid line using four-point Lagrange interpolation; call the directional classifiers. |
| Boundary values and fixed state | `map_int_to_bnd_grd` 21721-22031 | 21648-21958 | Interpolate 23 equilibrium quantities to boundary points, form 8-component full boundary values and 3-component equilibrium current, set boundary perturbation/current/E arrays to zero. |
| Irregular tags | `find_ir_pnt_bndx/bndz` 20782-20877 | 20709-20804 | Associate the two type-2 points with a nearby boundary record and tag `gdtp_ep` direction/rank as `+/-1` or `+/-2`. |
| Classify on each rank | `decide_grd_type_in_each_proc` 22034-22070 | 21961-21996 | Copy global `grd_type` and tags into rank-local `gdtp_ep(1:6)`. Active type codes are 1, 2, 4, 5. |
| Build difference coefficients | `calculate_dnfm_coeff_for_ir_bndx_point` 22074-22664 | 22001-22591 | Populate 4-node and 5-node nonuniform first/second derivative coefficients from the grid coordinates and fractional boundary locations. |
| Boundary-distance decay setup | `decide_hyperbolic_ratio_v2` 25215-25271; `distance_to_bnd` 25893-26012 | 25029-25085; 25707-25826 | Interpolate an angle-wise ellipse-normalized boundary radius from `xb/zb`, take the absolute normalized radial difference, cap it at `0.8`, and build `hypb_ratio`. This is not a Euclidean point-to-curve distance. |
| Build interpolation lists | `jx_jz_list_bndry8_cut_cell_v2_fixed` 24505-24575 | 24319-24389 | Collect interior directionally dropped points whose missing-direction value is supplied by a neighboring direction. |
| Main time loop | 919-1041 | 918-1037 | `setdt`, diagnostics, then `soundwave ? stepon_atfs : (use_stepon_cut_cell ? stepon_openacc : stepon)`. Both cases set `use_stepon_cut_cell=.true.` and `soundwave=.false.`. |
| RK4 driver | `stepon_openacc` 28865-29473 | 28713-29208 | Four derivative evaluations and the usual low-storage RK4 weights `1/6, 1/3, 1/3, 1/6`; final `bndry8_openacc`; kink-only PLL pressure substep when `conductpll=.true.`. |
| Stage RHS | `right_openacc` 29477-29921 | 29212-29656 | At every stage first calls `bndry8_openacc`; then computes perturbed gradients, current, and E through `convt_current_efield_openacc`; finally fluxes and viscous terms. |
| Fused conversion/current/E | `convt_current_efield_openacc` 30557-31126 | 30273-30824 | Active successor to the standalone `convt_openacc`, `current_openacc`, and `efield_openacc` subroutines; it computes all eight perturbation gradients, current, E, boundary E, and applies `hypb_ratio`. |

Thus `bndry8_openacc` is entered five times per accepted step: at the start of each of
the four `right_openacc` calls and once after the final RK4 combination. The standalone
`stepon_cut_cell`, `right_cut_cell`, `efield_cut_cell_v2_fixed`,
`bndry8_cut_cell_v2_fixed`, `convt_openacc`, `current_openacc`, and `efield_openacc`
are not reached from the live driver.

The free-boundary coefficient arrays `axm_bndz/axp_bndz/azm_bndx/azp_bndx` are
populated at lines 22597-22661, but every use in the supplied sources is commented out.
They are not part of the live coefficient rule described below.

## Exact directional classification

`grd_type_ratio` is the literal `0.5d0`. For an interior coordinate strictly between
two boundary crossings `bnd1 < x < bnd2`, with spacing `h`:

```text
if min(x-bnd1, bnd2-x) >= 0.5*h  -> type 1, regular
if min(x-bnd1, bnd2-x) <  0.5*h  -> type 5, inside but directionally dropped
if x == bnd1 or x == bnd2        -> type 5
otherwise                         -> type 4, outside dropped
```

The active code then changes a regular point and one adjacent regular point to type 2
when that pair bounds a dropped/boundary interval. Type 3 is mentioned in comments,
but every assignment to type 3 is commented out. Boundary values are carried in
separate `x_8bndx/x_8bndz` arrays.

Directional dropped points are not extrapolated. If one direction is retained and the
other is dropped, `bndry8_openacc` supplies the missing-direction perturbation with
`interp1d2l` (actual definition at lines 9379-9392). Despite its name, that routine is
three-point Lagrange quadratic interpolation using two interior samples and the fixed
boundary perturbation. If both directions are dropped, the RHS is zero.

## Copied coefficient rules

The source named these two nonuniform rules. The semantic labels `IR1/IR2` are not used
as an inference about enum ownership; the branch and stencil shape are authoritative.

### Branch `+/-1`: four-node third-order rule

For an upper boundary at `x_b = x_0 + b*h`, the retained geometric range is
`0.5 <= b < 1.5`, source coefficients are:

```text
p = -b + b^3/4
q =  b^2 + b^3/2
r =  3/4
s =  1/2
den = p*s - q*r
A = s/den
B = -q/den
C = (1 + A*b - B)/2

D f = A*(f0-fb) + B*(f0-fm1) + C*(f0-fm2)
```

The lower-boundary `d1fbp` formula is the reflected equivalent. First derivative
weights preserve polynomials through degree 3 at the origin; the degree-4 moment is
`2*b*h^4`, giving leading `O(h^3)` first-derivative error.

Its second derivative uses the source `gauss_solve` moment system
`[1,t,t^2/2,t^3/6] * w = [0,0,1,0]` on coordinates
`{x-2, x-1, x0, xb}`. The source maps solved weights into
`a2=w_b/2`, `b2=w_m1/2`, `c2=w_m2/2`, then

```text
D2 f = 2*(a2*(fb-f0) + b2*(fm1-f0) + c2*(fm2-f0))
```

which preserves polynomials through degree 3. Its degree-4 moment is
`2*(3*b-2)*h^4`; at `b=0.5` it is `-1`. Since the second-derivative weights scale as
`h^-2`, the leading error is generally `O(h^2)`, not third order.

### Branch `+/-2`: five-node fourth-order first derivative and third-order second derivative

For an upper boundary `x_b=x0+h*b` with `1.5 <= b < 2.5`, and with `xm2=x0-2h`,
`xm1=x0-h`, `xp1=x0+h`, `xp2=x0+2h`, set `h=1` and `xm2distance=2`.

```text
p1 = xp1      + xp1^4/xm2distance^3
p2 = xp1^2    - xp1^4/xm2distance^2
p3 = xp1^3    + xp1^4/xm2distance
q1 = xm1      - xm1^4/xm2distance^3
q2 = -xm1^2   + xm1^4/xm2distance^2
q3 = xm1^3    - xm1^4/xm2distance
r1 = b        + b^4/xm2distance^3
r2 = b^2      - b^4/xm2distance^2
r3 = b^3      + b^4/xm2distance
den = (p1*r3-p3*r1)*(q2*r3-q3*r2)
      -(q1*r3-q3*r1)*(p2*r3-p3*r2)
A  = r3*(q2*r3-q3*r2)/den
B  = r3*(r2*p3-r3*p2)/den
C  = r3*(p2*q3-p3*q2)/den
D  = (A-B+b^2*C)/xm2distance^2

D f = A*(fp1-f0) + B*(f0-fm1) + C*(fb-f0) + D*(f0-fm2)
```

This rule preserves polynomials through degree 4. Its degree-5 moment is `-2*b*h^5`,
giving leading `O(h^4)` first-derivative error.

The source second-derivative coefficients use:

```text
p1 = 1+2 = 3     p2 = 1-4 = -3   p3 = 1+8 = 9
q1 = 1           q2 = -3         q3 = 7
r1 = b^2+2*b     r2 = b^3-4*b     r3 = b^4+8*b
same determinant pattern as above
A2 = 2*r3*(q2*r3-q3*r2)/den2
B2 = 2*r3*(r2*p3-r3*p2)/den2
C2 = 2*r3*(p2*q3-p3*q2)/den2
D2 = -(1*A2 + 1*B2 + b*C2)/2

D2 f = A2*(fp1-f0) + B2*(f0-fm1) + C2*(fb-f0) + D2*(f0-fm2)
```

This preserves polynomials through degree 4. Its degree-5 moment is `(4-2*b)*h^5`,
giving generally `O(h^3)` second-derivative error. The lower-boundary branches are
obtained by reflection. Gradients of dropped-direction nodes are not part of a retained
stencil; the directional dropped value is first replaced by the quadratic interpolation
above.

The source `find_ir_pnt_bndx/bndz` routines use
`deltaxz=1.0001*(1+grd_type_ratio)*h=1.50015*h` with the strict association test
`abs(node-boundary) < deltaxz`. In the idealized uniform-grid case, after a node at
`0.25h` is dropped, the nearest retained node at `1.25h` receives the `+/-1` tag and
uses the four-node branch; the next node at `2.25h` receives the `+/-2` tag and uses
the five-node branch. The exact `1.5h` threshold belongs to the five-point branch. The
`1.0001` factor is a source association tolerance, not a new physical classification
range.

## Data tracked independently

The full state is `x(:, :, :, 1:8)` and the evolved perturbation is
`x1=x-xint`:

| Index | Quantity | Boundary/refresh behavior in the live OpenACC path |
|---:|---|---|
| 1 | density | Full equilibrium `xint1` on fixed boundary; perturbation evolved independently. Kink multiplies it by pointwise `hypb_ratio`; tearing does not apply the m=1:5 wall scaling. This is not the unreachable smoothing routine. |
| 2 | pressure | Full equilibrium `xint2` on fixed boundary; perturbation evolved independently. Tearing leaves pressure perturbation unscaled by `bndry8_openacc`; kink scales it via `hypb_ratio` and active PLL substeps. This is pointwise scaling, not live density/pressure smoothing. |
| 3 | radial velocity | Full equilibrium `xint3` on fixed boundary; near-wall perturbation multiplied by `hypb_ratio`. |
| 4 | toroidal velocity | Same treatment as index 3. |
| 5 | axial velocity | Same treatment as index 3. |
| 6 | radial magnetic field | Full equilibrium `xint6` on fixed boundary; perturbation gradient drives current. `hypb_ratio` does not scale this component. |
| 7 | toroidal magnetic field | Full equilibrium `xint7` on fixed boundary; no m=6:8 wall decay. |
| 8 | axial magnetic field | Full equilibrium `xint8` on fixed boundary; no m=6:8 wall decay. |
| E, 1:3 | electric field | `ef_3bndx/ef_3bndz` are reset to zero in the live fused routine; with `hall=.false.` the Hall-only boundary-interpolation branch is inactive. The final full `ef` is multiplied by `hypb_ratio`. |

`xint_8bndx=xint_8bndz=[rho,p,0,uy,0,Br,Bt,Bz]` and
`cint_3bnd*=[Jr,Jt,Jz]` are initialized from equilibrium interpolation. At `nstep=0`,
`x1_8bndx/x1_8bndz` are zeroed and the full `x_8bnd` arrays are reconstructed once.
They are not refreshed on later steps, so the live evolution uses fixed boundary full
fields. `rho_from_p=.false.`; density is not reconstructed from pressure.

The live initialization calls `distance_to_bnd` and `decide_hyperbolic_ratio_v2`.
`hypb_ratio` is zero at/outside the wall region and tends to one toward the interior:

```text
ellipse_radius = sqrt(((R-R0)/Rdim)^2 + ((Z-Z0)/Zdim)^2)
distance = min(abs(interpolated_boundary_ellipse_radius - ellipse_radius), 0.8)
cutoff = max(dz*(2.5+0.5)/Zdim, dR*(2.5+0.5)/Rdim)
hypb_ratio = max(0, tanh(30*(distance-cutoff)))
```

For points tagged outside (`type 4`) it remains zero; for the `rr < 0` branch it is set
to one. Live consumers are the selected perturbation components in
`bndry8_openacc`, the kink-only PLL pressure substeps, and the final electric-field
scaling.

## Active case switches

| Switch/effect | Resistive kink | Tearing mode | Live consequence |
|---|---:|---:|---|
| `use_stepon_cut_cell` | true, line 1097 | true, line 1093 | Selects `stepon_openacc`. |
| `initia_from_pgfile` | false, line 1100 | true, line 1096 | NOVA/flux-surface path versus precomputed PG binary path. |
| `symmetryz` | true | false | Kink uses half-plane symmetry; tearing uses full plane. |
| `rho_unif` | false, line 1144 | true, line 1140 | Tearing replaces loaded ion density by one before state construction. |
| `conductpll` | true, line 1273 | false, line 1269 | Kink executes pressure PLL substeps inside `stepon_openacc`; tearing skips them. |
| `cfl`, modes, perturbation scale | 1.5, `(1,1)`, `eta10=1e-8` | 1.2, `(1,2)`, `eta10=1e-7` | Case-specific numerical setup. |
| boundary wall scaling | `m=1:5`, lines 24404-24413 | `m=3:5`, lines 24269-24278 | Kink applies pointwise `hypb_ratio` to density/pressure/velocity perturbations; tearing applies it only to velocity perturbations. |

All `smooth`, `smoothef`, `smoothpll`, and `smoothp1ll` flags are false in both inputs.
The main loop calls `stepon_openacc`; its live `bndry8_openacc` path has the
`smth_irpt_with_difc_*` calls commented out. The active calls in the unreachable
`bndry8_cut_cell_v2_fixed` path are not evidence of shipped-case smoothing.
`data_type_weight` is initialized, but its live consumers are only the unreachable
smoothing routines. The kink's `hypb_ratio` component scaling is not a smoothing call.
`rmp_east=.false.`, so fixed RMP perturbations and their boundary updates are inactive.

`perturbation` is called once, after equilibrium energy accounting. It fills active
`eta1` resistivity perturbation and `perb` magnetic perturbation. In the live OpenACC
RHS, `eta1` contributes to E for `nstep < nper=100`; `perb` is referenced only by the
non-OpenACC `right`/`right_cut_cell` paths and is not added by `right_openacc`.

## Source/reading differences and unresolved paths

1. The reading note says both irregular classes are fourth order. The exact shipped
   formulas show the four-node branch is order 3 and the five-node branch is order 4
   for first derivatives; their second derivatives are generally order 2 and order 3,
   respectively.
2. The reading note says a point needed by the other direction is updated by linear
   interpolation. The shipped source calls a routine named `interp1d2l`, but its body is
   three-point quadratic Lagrange interpolation. This difference is reported, not
   reconciled by changing either source.
3. The reading note describes a boundary point type. The implementation keeps boundary
   data separately and assigns exact/too-close interior points type 5; active type 3
   assignments are commented out.
4. The sources contain duplicated legacy wrappers and active fused OpenACC code. A
   subroutine name alone is not evidence of execution; the call sites above distinguish
   live from dead code.
5. Brute-force contour, tangency, and multi-process duplicate-boundary behavior were not
   executed. The tearing case's binary contents and whether the delivered cases build or
   run in the current toolchain are unresolved.
6. The exact source semantics of the `+/-1` versus `+/-2` enum labels are internally
   inconsistent enough that the oracle names branches by stencil shape, not by a newly
   invented physical class name.
