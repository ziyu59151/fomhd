# Bounded G-EQDSK input reader

## A0-TOOL-GEQDSK-01: Numeric core

```
read_geqdsk(path):
  validate a regular ASCII file no larger than 32 MiB
  read header and numeric body
  parse nr, nz from the final three header integers
  bound nr, nz and nr*nz
  tokenize signed decimal/scientific numbers while retaining text offsets
  require enough tokens for ten scalars, four nr profiles, nz*nr psi and q
  read boundary and limiter counts at the fixed core offset
  require finitely bounded nonnegative integral counts and enough core tokens
  for both polygons
  identify the character offset immediately after the declared numeric core
  reject any non-whitespace character inside the numeric core
  require all core numbers to be finite
  build R, Z, psi, F, pressure, FFprime, pprime, q, boundary and limiter arrays
```

The reader never finds a later boundary by searching trailing text and never
truncates a short core. A malformed or short declared core is an error.

## A0-TOOL-GEQDSK-02: Opaque EFIT extension

```
  suffix = exact bytes following the declared numeric core
  if suffix is whitespace only:
    extension = empty
  else if the header first token begins with EFIT:
    retain the exact suffix bytes separately
    record extension format, byte count, SHA256 and whole-file SHA256
  else:
    reject the unsupported suffix
```

The opaque suffix is provenance-bearing input, not silently discarded
metadata. It can contain comma-delimited numeric records and labeled EFIT
fields. No suffix value overrides the standard numeric core.

## A0-TOOL-GEQDSK-03: Reconstruction audit

```
reconstruct(eq, *, psi_factor):
  require explicit finite nonzero psi_factor
  differentiate source psi on the source Z/R grid
  normalize source psi by axis and boundary flux
  interpolate F on source psi_N
  report profile extrapolation masks instead of treating them as valid
```

The audit reconstructs point fields only. It does not create a two-fluid
equilibrium or infer species data.
