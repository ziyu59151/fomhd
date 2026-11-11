# Fixed wall linear systems

See [source](../../../../src/fomhd/wall_cache.py) and
[rectangular wall stage flow](box_boundary.py.md).

For each mode, collect the signed normal-current rows from the fixed
geometry. Apply C and C-adjoint to their metric-weighted transpose to form
the thin boundary matrix. Select independent principal rows, factor once
with the existing checked scaling, and retain the omitted rows for checking.
Reuse only with matching geometry, backend, scale and operator identity.
At a stage, compute the RHS using two matrix-vector products, solve the
selected rows with the saved factor, and check the original full system
before returning the electric correction. The cache is derived data
and is rebuilt on restart.
