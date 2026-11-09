# WR physical input fixtures

Status: physical setup only; no waveform solver or production acceptance.

```text
periodic_inputs(points, amplitude, nonuniform, seed_mode, sentinel, device):
  validate odd points >=9 and resolved spatial seed modes
  reuse the established periodic Hall geometry, operators and parameters
  allocate backend-owned FP64/complex128 independent state
  prescribe the documented nonuniform density and electron pressures
  prescribe finite transverse velocity and magnetic seed at seed_mode
  add a small, independently labelled highest spatial Fourier seed
  mark it retained when the resolved axial mode is within points//2,
  independently of retained toroidal K
  return context, state and exact physical input origin/mode metadata

closed_wall_inputs(nodes_r=4, nodes_z=4):
  reuse the real K=1 rectangle fixture and compatible uniform state
  retain its declared hard/soft diagnostics and 1 GiB fixture memory bound
  return context, state and actual K/Nphi/cell-count metadata
```

The periodic spatial mode index refers to the resolved axial grid, independently
of retained toroidal K=0. Inputs share the established physical builders; they
import no T2-E, C1 or C2 time solver. Returned arrays belong to the requested
backend, use SI units and remain caller-owned. Invalid inputs raise before
advancing a state. GPU execution requires the caller's owned Slurm allocation.
