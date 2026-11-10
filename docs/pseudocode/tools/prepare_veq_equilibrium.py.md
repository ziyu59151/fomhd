# `tools/prepare_veq_equilibrium.py` implementation pseudocode

## VEQ-CLI-00 Public preprocessing entry

Import `veqpy` only inside the solve path. Build the fixed circular boundary
`R0=3 m`, `a=0.5 m`, `Z0=0`, `ka=1`, and zero cosine/sine offsets through the
public `KernelBoundary`.

Use PF/psin/uniform, `constraint="ip"`, NumPy float64, and the Numba backend.
Raw source arrays are:

```text
P_psin  = constant negative shape [Pa]
FF_psin = zero [(T m)^2]
p0      = 1000 Pa raw source input
Ip      = pprime_F*pi*R0*a^2 = 23561.94490192345 A
```

The final physical current normalization supplied by the Ip constraint must be
checked against `R*1e4 A/m^2`. Do not claim raw `p0` or raw `P_psin` equals the
native physical edge pressure or `P_psi`.

## VEQ-CLI-01 Refinements

Run at most three small representation refinements, all below 100 active
coefficients and `Nr<=64,Nt<=128`:

- coarse;
- medium;
- fine.

Build one fresh immutable Kernel per refinement. Time prepare, solve, and
equilibrium evaluation separately; report cold/JIT-inclusive and solver
timings. Reject nonconverged results and retain the error/output evidence.

## VEQ-CLI-02 Outputs

For each solve:

1. retain `SolveResult` and the native `Equilibrium`;
2. evaluate on a separate uniform `(r,theta)` output grid;
3. pass native arrays into `tools.veq_equilibrium` for physical conversion;
4. write raw/converted NPZ and JSON;
5. record projected residual, profile vs Ampere current, actual Ampere current,
   strong residual by band, collocated smooth wall normal flux,
   pressure offset, native/native p0, raw derivative units, and source identity.

For one fixed solve, repeat conversion/evaluation on at least three output
grids. Report this separately from solver representation changes.

Use the fine solve at fixed solver state for `33x48`, `37x64`, and `49x96`
uniform output grids. Keep the `33x48` product as the case primary product and
write each additional grid as an independent NPZ/JSON receipt. Record
`solver_representation_changed=false`; these are output differentiation and
resampling tests, not new solver degrees.

Write an aggregate refinement summary. These products remain scratch evidence;
this command performs no FOMHD stage or time evolution.
