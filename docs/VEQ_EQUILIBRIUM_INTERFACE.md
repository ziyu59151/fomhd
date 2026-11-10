# VEQPy equilibrium preprocessing

Status: the pinned external code and FOMHD-side adapter have been executed.
The selected finite VEQ snapshot remains a GS cross-check input; complete
Hall/CGL static acceptance and intercepted-wall evolution are not established.

## Source, environment and entry points

Upstream is <https://github.com/FusionAlpha/veqpy>, commit
`2b64c9a1bffddc25c176bd52231bc3b481715c6c`, Git tree
`ac8635c629d100db8998f9904793dbf697373eb7`. The archive SHA-256 is
`6b553ce32bb9261e4782af4542b7be15f14621f3256d303f9b175fdd582bfc1f`.
All 206 upstream file identities were checked. The source, upstream BSD license,
official provenance, L152/W151 paper identity and selected review scope are
registered in the design repository's
[adoption record](/home/lxzeng/fomhd-design/src_refs/VEQPy_SOURCE_ADOPTION_2026-11-10.md).

The independent preprocessing environment is
`/home/lxzeng/.venvs/fomhd-veqpy-20261110`. Its exact runtime freeze is retained
with the run inputs. It uses Python 3.12.16, VEQPy 1.4.0.dev0 and Numba 0.68.1.
FOMHD's Warp/CuPy environment, `pyproject.toml` and `uv.lock` are unchanged.
VEQPy is imported lazily by the external CLI, not by the solver or NumPy-only
snapshot adapter.

| Entry | Responsibility |
|---|---|
| `tools/prepare_veq_equilibrium.py` | Actual public VEQ API calls, bounded circular refinements, native coefficients and SI products |
| `tools/veq_equilibrium.py` | Immutable conversion, distinct profile/Ampere quantities, collocated wall trace and sampled strong-GS diagnostics |
| `tools/circular_equilibrium.py` | Preserved independent polynomial circle and vacuum controls |
| `tools/equilibrium_compatibility.py` | Independent physical RZ equation checks; force/Ohm checks explicitly limited to isotropic electron pressure |
| `tools/prepare_staircase_equilibrium.py` | Formal old-path sampling, recorded input projection, initialization and initial full RHS; no time advance |

Each entry has canonical pseudocode. Run-specific samplers, independent audit
scripts, data and failed diagnostics remain outside the source tree.

## Physical conventions and ownership

VEQ R/Z are metres, `Psi` is Wb, `Psi/(2*pi)` is Wb/rad, F is T m and pressure
is Pa. Raw `P_psin` has units Pa because psin is dimensionless; raw `FF_psin`
has units `(T*m)^2`. Physical `P_psi` has units Pa/(Wb/rad).

For the selected circle, convert flux orientation by

    psi_F = (Psi_edge - Psi)/(2*pi)
    B_R = -d_Z(psi_F)/R, B_phi = F/R, B_Z = d_R(psi_F)/R.

The target is R0=3 m, a=0.5 m, F=6 T m, pprime=1e4 Pa/(Wb/rad),
n=2e19 m^-3, ion pressure fraction0.6 and ions at rest. Ip is
`pprime*pi*R0*a*a=23561.94490192345 A`, without an additional toroidal factor.

Native `jphi` and `R*pprime` are profile formulas. Their agreement checks
normalization only. Output explicitly separates `J_profile`, actual reconstructed
`J_ampere=curl(B)/mu0`, and the corresponding electron velocities. Solver current
and Ohm reconstruction keep their existing single producer; none of these
preprocessing arrays is injected as a replacement current.

In the Ip route, a provisional pressure-normalization multiplier acts before
the final published alpha1 is renormalized. Final alpha1 times raw p0 is **not**
the native edge pressure. The converter uses native P/p0 directly and records
an additive offset to match 1000 Pa at the wall. For the fine snapshot native
p0=7.126327231 Pa, offset=992.873672769 Pa. Native and shifted pressures both
remain in the output. This does not rescale gradients, B or current. The
archived self-circle, VEQ and FOMHD mu0 constants are retained separately.

The exact constant-F GS family can be static in the massless model:
`u_e=-J/(e*n)=R*Omega(psi)*e_phi`, div(ue)=0, ue.grad(pe)=0 and
`b b:grad(ue)=R*b_phi*Omega'(psi)*b_pol.grad(psi)=0`. Both CGL pressure
equations must be checked, rather than assuming initial isotropy is sufficient.
For general Fprime, the additional field-parallel electron flow can give
nonzero strain; an arbitrary isotropic GS input is not automatically static.

## Verified scope and open limits

Three actual circular solve representations (5,8,12 active coefficients)
converged with projected raw norms6.72e-14,4.27e-14,8.00e-13. The same fine
solution was sampled on33x48,37x64,49x96 output grids. The strong-GS residual
and full continuous-model residual are distinct from those projected norms.

The v4 diagnostic paired a vertex field with a forward-chord normal, producing
a false Bn about6.43e-4 T. Collocated smooth normals give about1.73e-18 T on
the same fields. The old copied first-ring axis gradient and an off-axis
singularity guard were also corrected; all old products remain preserved.
No conclusion about upstream GS accuracy uses those invalid diagnostics.

Independent review of the corrected49x96 product finds near-wall electron CGL
RHS maxima75.03/150.83 Pa/s and electron-force residual0.1325 N/m^3. A separate
analytic Ampere-to-rigid-flow control quantifies derivative contamination
(CGL5.34/2.39 Pa/s), rather than calling every residual physical. An observation
window and target-signal error budget have not accepted this VEQ snapshot as a
static background. It remains category2: GS cross-check, full-model unresolved.

The degree12 self-circle has independently evaluated curl-current discrepancy
about1.48e-14 relative near the wall, with measured derivative refinement and a
nonzero CGL-response control. Its mathematical constant-F family supports static
input construction. Mapping onto a staircase wall changes the physical domain:
the same smooth-circle fields need not satisfy Et/Bn on that different wall.
Formal old-path initial RHS is nonzero and is not called an accepted background.

No new boundary runtime selector, checkpoint identity, RK4/WR evolution or GPU
acceptance follows from this preprocessing work. Those depend on the separate
coupled local-boundary gates. No background RHS is subtracted, current replaced,
normal magnetic flux reset or damping added.

## Reproduction and retained data

From the source root, use the independent environment and a fresh output path:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    NUMBA_CACHE_DIR=/home/lxzeng/runs/fomhd/veq-replay/cache \
    /home/lxzeng/.venvs/fomhd-veqpy-20261110/bin/python \
    tools/prepare_veq_equilibrium.py \
    --output /home/lxzeng/runs/fomhd/veq-replay/equilibrium
```

The source adapter itself needs only the existing NumPy stack. Its tests run
under normal project pytest without VEQ installed. The actual-call receipt,
native inputs, previous failed diagnostics, raw arrays and independent
common-RZ/full-equation audit are linked from
[/home/lxzeng/runs/fomhd/a050-clt-intercept__20261110T090707Z/](/home/lxzeng/runs/fomhd/a050-clt-intercept__20261110T090707Z/).
