# A0 external equilibrium input adapter

## A0-PY-TOKAMAK-EQUILIBRIUM-01: Point samples

```
sample_equilibrium(eq, R, Z, *, psi_factor):
  validate explicit finite nonzero psi_factor
  read source R, Z, psi, F, pressure, q, axis and boundary flux
  require finite monotonic source grids and profile-sized arrays
  broadcast query R and Z to one shape
  reject nonfinite queries and queries outside the source grid
  build RectBivariateSpline(Z, R, source psi)
  evaluate source psi and its Z/R derivatives at query points
  psi_N = (source psi - psi_axis) / (psi_boundary - psi_axis)
  reject any psi_N outside [0, 1]
  profile_x = uniform [0, 1] samples of length number of R cells
  interpolate F, total MHD pressure and q at psi_N
  B_R = -psi_factor * dpsi/dZ / R
  B_phi = F / R
  B_Z = psi_factor * dpsi/dR / R
  return new same-shaped float64 arrays and no input mutation
```

The psi factor is an explicit caller-owned conversion to the Wb/rad flux
convention. It changes B_R and B_Z with the stated sign and does not change
psi_N, F, pressure or q. `pressure_Pa` is the file's total MHD pressure in
pascals after interpolation; it is not a species pressure.

## A0-PY-TOKAMAK-EQUILIBRIUM-02: Closed flux contour

```
closed_flux_contour(eq, level=0.8):
  validate finite normalized level in [0, 1]
  validate source R/Z/psi grid and finite magnetic axis
  psi_N = (psi - psi_axis) / (psi_boundary - psi_axis)
  lazily construct a noninteractive Matplotlib Figure/Canvas
  extract only the raw contour segments at the requested level
  skip the empty placeholder path returned when a level is absent
  classify a segment closed only when its final point repeats its first
  reject any open segment
  drop the repeated closing point from a closed segment
  choose the unique closed segment containing the magnetic axis
  reject zero or multiple such segments, nonfinite points and zero area
  reverse negative-area point order so signed area is positive (CCW in R-Z)
  record the supplied normalized flux level on the returned array
```

The returned `N x 2` float64 array is owned by the caller. Its private ndarray
subclass preserves `flux_level` through ordinary array views and copies. A
contour level is provenance for the requested flux surface; this adapter does
not claim that the level is the last closed flux surface. No X-point or vacuum
field is synthesized.

Ignore unrelated open contour fragments; accept only a unique closed contour containing the magnetic axis. If none exists, fail and report whether open fragments were present. Never close an open path artificially.
