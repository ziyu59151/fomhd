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
    reaction = -(T.H @ lambda)/winv
    projected[l] += reaction
    require finite projected/reaction/multiplier
    store reaction,lambda and S@lambda-rhs
  return owned Projection(projected,reaction,lambda,multiplier_residual)
```

No physical-real projection is applied: complex unit vectors used to
materialize the electric response preserve their complex coefficients through
this operation. The input `r` is never changed. The full `T_full` rows are
checked by the caller after projection.
