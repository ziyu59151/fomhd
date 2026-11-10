# A050 polygon runtime contract

The native codec and complete runner share the [runtime contract](../polygon_runtime.md).

## A050-RUN-INIT-HOST-DEVICE

Initial-state construction is the only retained-state host-to-device boundary.
Host NumPy control arrays must be converted with
`context.compute.array(value, complex128)` before any slice assignment into a
context-owned state array.

```text
device_complex(value):
  return context.compute.array(value, complex128)

FillFluidControl(state, density, p_i, p_e_perp, p_e_parallel):
  n = device_complex(density)
  pi = device_complex(p_i)
  pe_perp = device_complex(p_e_perp)
  pe_parallel = device_complex(p_e_parallel)
  each input is uploaded exactly once before the species loop
  assign only backend-owned n/pi/pe arrays to retained k=K slices
  scalar zero assignments keep their existing form
```

The same rule applies to every initial magnetic-field slice:

```text
Vacuum:
  polygon B_phi = device_complex(host F0/R_v)
  staircase B_phi block = device_complex(host F0/R_v)

GS:
  old staircase B = device_complex(host C Aphi + harmonic)
  polygon B_phi = device_complex(host F0/R_v)
  polygon B_t = device_complex(host discrete tangential field)
```

No stage, RHS, iteration, time, physics, threshold, or later instrumentation
is changed. On the CPU backend, `array` preserves the same values and the
complex128 retained representation. On CUDA, the resulting backend-owned
array is the only right-hand side used for a retained slice assignment.

Initialization metadata may keep host copies for provenance and diagnostics;
they are not assigned directly into retained backend state.
