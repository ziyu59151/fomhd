# FreeGS source export

The included `freegs-equilibrium.npz` is the 65×65 constant $F=2$
T·m, 200 kA lower-single-null export. Its crop lies inside the coil-free
region. The [receipt](freegs-receipt.json) records the native
free-boundary solve, source audit, topology, and cropping bounds.
For the FOMHD replay use the frozen, supplied geometry and initial
arrays. This separates replay from regenerating the equilibrium.

The optional generator used the [official FreeGS v0.8.2 wheel](https://pypi.org/project/FreeGS/0.8.2/)
with the supplied scalar-extraction compatibility patch for NumPy
2.5.4. Its machine and boundary constraints follow the
[upstream example](https://github.com/freegs-plasma/freegs/blob/v0.8.2/01-freeboundary.py).
The generator relocates two outboard filament coils beyond the numerical
edge, then scales the connected-core pressure derivative to the requested
total current with zero $FF'$. It excludes private flux from the native
plasma current source before the elliptic solve. The exported arrays are
in SI with spatial indexing `[Z,R]`.

For an optional clean regeneration, work outside the source tree and use
the supplied dependency pins.

```sh
uv venv --python 3.12 "$work/freegs-env"
uv pip install --python "$work/freegs-env/bin/python" \
  -r "$bench/reference/requirements.lock.txt"
patch "$work/freegs-env/lib/python3.12/site-packages/freegs/critical.py" \
  < "$bench/reference/upstream-scalar-compat.patch"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  "$work/freegs-env/bin/python" "$bench/reference/solve_freegs.py" \
  "$work/freegs-regenerated" --count 65 --current 200000
```

Archive ZIP metadata and library versions can affect bytes in a regenerated
file. The replay uses the bundled file and checks its archived SHA-256.
