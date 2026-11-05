# Wall projection implementation pseudocode

## A0-PY-WALL-01: BuildWallContext

Input `n` is the physical real density in the retained layout
`[mode][cell]`; `species` is the native ion or electron record; `tag` is the
stored stage identity; `context` owns the backend, geometry and unchanged wall
matrix `T`. Output `WallContext` owns one `WallSample` per sample and borrows
the input species/backend through the call only.

```text
BuildWallContext(n,species,tag,context):
  require n.shape == (2*K+1)*Nc
  require N == context.N and backend/operator identities are present
  for each sample l:
    n_l = inverse(n,Nc,physical=true)
    parallel_entity = tile(n_l, 3)  # [component][cell], never cell-major repeat
    require finite(parallel_entity)
    require parallel_entity > species.n_min
    winv = species.mass*parallel_entity / repeat(M0,3)
    require finite(winv) and winv > 0
    S = T @ diag(winv) @ T.H
    row_scale = mass*n_ref*u_ref/t_ref
    col_scale = row_scale/(mass*n_ref/L_ref^3)
    require rank of S with the configured rank_rtol
    store S, winv, row_scale, col_scale and the stored tag
  return owned WallContext(samples,tag)
```

The solve factors are backend-private. Their construction and all matrix
operations stay on `context.compute`; any synchronization occurs inside the
backend call. A nonpositive density, nonfinite value, zero matrix, or deficient
wall rank raises `Failure` with the native code and no projected result.

## A0-PY-WALL-02: ProjectWall

```text
ProjectWall(wall,r,context):
  require wall.samples has N entries and r has N*3*Nc entries
  require wall tag and backend identity match the caller
  projected = copy(r)
  for each sample l:
    rhs = T @ r[l]
    if T has no selected rows:
      continue
    lambda = linear(S,rhs,row_scale,col_scale,rank_rtol,1e-10).x
    reaction = -winv*(T.H @ lambda)
    projected[l] += reaction
    forward = T @ projected[l]
    for refinement in 0..4:
      bound = 256*eps*max_row_sum(abs(T))*max(max_abs(projected[l]),tiny)+tiny
      if max_abs(forward) <= bound:
        converged = true
        break
      correction_lambda = same_scaled_normal_solve(forward)
      correction_reaction = -winv*(T.H @ correction_lambda)
      corrected_value = projected[l] + correction_reaction
      corrected_forward = T @ corrected_value
      if max_abs(corrected_forward) >= max_abs(forward):
        raise Failure("WallProjection","forward wall trace refinement stagnated")
      projected[l] = corrected_value
      reaction += correction_reaction
      lambda += correction_lambda
      forward = corrected_forward
      bound = 256*eps*max_row_sum(abs(T))*max(max_abs(projected[l]),tiny)+tiny
      if max_abs(forward) <= bound:
        converged = true
        break
    require converged
    require finite projected/reaction/total multiplier
    multiplier_residual = S@lambda-rhs
  require finite projected/reaction/total multiplier/multiplier_residual
  for each sample l, including samples with zero selected rows:
    full_bound = 256*eps*max_row_sum(abs(T_full))*max(max_abs(projected[l]),tiny)+tiny
    require max_abs(T_full@projected[l]) <= full_bound
      otherwise raise Failure("WallProjection","full wall trace exceeds bound")
  return owned Projection(projected,reaction,lambda,multiplier_residual)
```

The correction solves `S*correction_lambda = T*projected[l]` with the same
row/column scaling, rank threshold, residual threshold, and backend path as the
primary solve. The CPU path reuses the stored LU factor; the CUDA path uses the
existing backend linear solve on the active stream. The correction reaction is
accumulated into both `projected` and `reaction`, and
`correction_lambda` is accumulated into the reported multiplier. The final
`multiplier_residual` is always `S@lambda-rhs` for the original sample RHS.

The forward trace bound is evaluated per sample after each update. It uses
FP64 epsilon, the maximum absolute row sum of `T`, the largest projected value,
and the minimum positive FP64 scale. No relative-tolerance relaxation, clipping,
or physical-real conversion is introduced. If five corrections do not reach
the bound, or if a correction does not strictly improve the forward trace, the
operation raises `Failure("WallProjection", ...)`; it does not return the
unconverged candidate. After refinement, all `T_full` rows are checked per
sample against the same output-scaled formula with the norm of `T_full`.
The solve still uses the selected rows of `T`; an unsatisfied unselected row
raises `WallProjection`. The unchanged caller-side checks remain hard
constraints as well.

No physical-real projection is applied: complex unit vectors used to
materialize the electric response preserve their complex coefficients through
this operation. The input `r` is never changed.
