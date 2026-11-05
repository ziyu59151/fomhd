# `src/fomhd/types.py` implementation pseudocode

## A0-PY-TYPE-01 Public physical records

1. Define `Species`, `References`, `Budget`, `Parameters`, `StageTag`,
   `FluidState`, `State`, and `Context` as dataclasses in the native field order.
2. Preserve native numeric defaults (zero, empty, or stated native default).
   Append only the approved migration controls, each with a default:
   `electron_inertia=1`, `integrator="rk4"`, `dt_guard="error"`,
   `minimum_samples_per_cycle=32`, `max_amplitude_error=0.01`, and
   `max_phase_error_rad=0.1`.
3. Keep `Norm` values identical to native: `linf`, `weighted_l2`, `value`.
4. `Failure` derives from `RuntimeError`, taking `(code, message)` and exposing
   `code`, `checks`, `stage`, and `partial_arrays`.
5. Native result structures are constructed through `types.SimpleNamespace`.
   Export factory functions that initialize every consumed field; no factory
   silently omits a record member.

## A0-PY-TYPE-02 Layout and ownership

1. Scalars have shape `(2K+1, E)`.
2. Vectors have shape `(2K+1, 3E)` with component order R, phi, Z.
3. Samples replace the mode axis with `Nphi`; tensors use
   `(sample, row, column, cell)`.
4. Arrays are complex128 and belong to `Context.compute.xp`. Geometry setup
   tables and small metadata are host data.

## A0-PY-TYPE-03 Failure behavior

1. Constructors do not silently replace invalid physical parameters.
2. Shape, finiteness, rank, time, memory, and Hermitian failures use `Failure`
   or a backend numerical failure with a stable diagnostic message.
3. A failure object is retained by the caller; no exception triggers a CPU
   fallback or an automatic retry.
