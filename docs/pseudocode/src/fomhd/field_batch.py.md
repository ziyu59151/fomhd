# Bounded batched massless field RHS

## Status

Selected T2-E producer under D070; actual verification scope is recorded in
the dated T2-E report. It is a field subproblem, not full coupled acceptance.
The original worker failure and all parent repairs remain in external evidence.

## Selected equations

For each independent time row, the module reconstructs the current production
massless branch rather than defining another physical model:

1. `BN = Q_B B` in physical samples, with `bhat = BN / |BN|`.
2. `Fi = I m_i / m_i` in retained modes, then physical `Fi`.
3. `ne = (q_i/e) ni`, and `J = Cdag B / mu0`.
4. `Fe = (q_i Fi - J) / e`.
5. `H Fe` is projected with the density-weighted selected wall rows `T`.
6. `ue = constrained(H Fe) / ne`.
7. The CGL-B pressure tensor, cylindrical velocity gradient, its metric-weighted
   adjoint, pressure-flux divergence, and `pdot_perp`, `pdot_parallel` are
   evaluated with the same formulas as `fomhd.fluid`.
8. The constrained Ohm source is pressure force divided by `e` minus
   `ne (ue x BN)`, projected on the wall and interpolated by `I` to edges.
9. The fixed wall correction is applied without incrementing
   `WallSystemCache.solves` or `WallSystemCache.evaluations`.
10. `Bdot = -C E`, where `E` includes the fixed wall correction.

The returned `rhs` contains only:

```text
[pdot_perp, pdot_parallel, Bdot]
```

The ion pressure block in `P` is retained for layout compatibility but is not
used by this field-only producer. The ion RHS, diagnostics, time integration,
and full scientific acceptance are explicitly outside this subproblem.

## Layout and ownership

- `P`: complex128 `(B, 2*K+1, 5*Nc)`, in mode order `-K..K`.
  Each mode contains `ni, mi_R, mi_phi, mi_Z, pi`.
- `Z`: complex128 `(B, 2*K+1, 2*Nc+Nb)`, in mode order `-K..K`.
  Each mode contains `p_perp_e, p_parallel_e, B`.
- Returned `rhs`: complex128 with the same layout as `Z`.
- Returned `electric`: complex128 `(B, modes, Ne)`.
- Returned `current`: complex128 `(B, modes*Ne)` on electric edges.
- Returned `velocity`: complex128 `(B, N*3*Nc)`, because it is already a
  physical-real sample quantity represented in complex128 storage.

Inputs are borrowed read-only. Outputs own their storage. Numerical work stays
on the context-selected NumPy or CuPy backend. There is no CPU fallback and no
whole-state host transfer.

## Batch convention

The leading axis is independent time-stage data. Fourier transforms act on the
second axis. Sample operators act on the physical sample axis. Python loops are
limited to the three vector components, retained Fourier modes, and fixed
iterative refinements; no loop runs over leading batch items or time nodes.

The inverse convention is `ifft(bins, axis=1) * N`. The forward convention is
`fft(samples, axis=1) / N`, followed by retaining bins `-K..K`. The azimuthal
derivative uses signed FFT frequencies.

Physical-real rejection is calculated independently for every batch item:

```text
max(abs(imag(samples_item)))
    <= 512 * eps64 * max(max(abs(retained_item)), reference)
```

This prevents a large item from hiding a non-real smaller item.

## Wall handling

Assemble the exact normal matrix T diag(w) T* from the nonzero incidence pairs
of T, precomputed once. For each column i and each pair of its rows (r,s),
accumulate w[b,n,i]*T[r,i]*conj(T[s,i]) at (b,n,r,s). This avoids the temporary
(B,N,W,3Nc); it keeps the actual density, all selected rows, and the dense
bounded normal solve. Count setup, scatter, SVD, solves and refinement.
The remaining normal matrices/workspaces scale with B*N*W^2, and production
requires a verified block/structured solve rather than assuming this is cheap.

Preflight every batch before sampling: include retained fields, physical
samples, tensors/gradients, normal/SVD/solve workspaces, fixed operators and
cached wall factors. The window algorithm separately checks its node/stage/
tree history. GPU pool limits additionally enforce the selected allocation
budget. Estimates and measured memory are reported separately.

The producer reuses an identity-matched `WallSystemCache` when one is already
attached. Otherwise it builds a producer-owned local immutable cache. It applies
the prepared selected principal systems to all batch entries and verifies:

- every selected density-weighted system is full rank;
- the componentwise scaled solve is converged;
- every original wall row of each electric correction is satisfied;
- density-weighted velocity rows and the full wall trace remain satisfied.

There is no stored `Nc^2` propagator and no explicit matrix inverse.

## Memory model

Beyond the fixed context operators and wall cache, the dominant temporaries are:

```text
O(B*N*(Nc + Ne + Nb + 9*Nc))   physical samples and tensors
O(B*q*Ne)                      retained electric fields, q=2K+1
O(B*N*W^2)                     batched density-weighted wall systems
O(B*N*W)                       wall right-hand sides and multipliers
```

`W` is the number of selected wall rows. The `B*N*W^2` term is the principal
new batching cost and must be included in any later device-memory decision.

## Historical worker failure and implemented repairs

The isolated worker stopped after its fifth failed repair. Its unmodified logs
remain external evidence. Parent integration corrected the projected-velocity,
current and component-block layouts, independent batch einsum indices and
empty-wall reductions, then compared each physical block against the production
stage on CPU and GPU. This is the current code; the old failure is not its status.

CuPy's indexed complex scatter is not supported by the pinned implementation.
Accumulate the real and imaginary FP64 parts separately into the owning complex
normal matrix. CPU/GPU tests cover this exact wall construction. Cache identity
includes model and rank parameters as well as spatial/spectral identities.

## Exact diagonal normal specialization

During producer construction, inspect the precomputed incidence pairs. If every
pair has r=s, the selected density normal is diagonal for all positive densities.
Unless the explicit dense-control flag is selected, store only B*N*W real
entries. Assemble its diagonal by FP64 incidence scatter, use its positive
maximum/minimum as the exact singular-value extremes, and divide to solve.
All original rank, refinement, selected/full-row and physical checks remain.
If any off-diagonal pair exists, retain the general dense normal path. Never
infer diagonal structure from numerical smallness. Count diagonal systems and
dense SVD systems separately; preflight the actual selected representation.
CPU/GPU compare both paths against each other and the production stage. This
is a separate spatial/constraint optimization, not a time-parallel gain.
