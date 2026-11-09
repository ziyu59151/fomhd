# `src/fomhd/types.py` implementation pseudocode

## A0-PY-TYPE-01 Public physical records

1. Define `Species`, `References`, `Budget`, `Parameters`, `StageTag`,
   `FluidState`, `State`, and `Context` as dataclasses in the native field order.
2. Preserve native numeric defaults (zero, empty, or stated native default).
   Append only the approved migration controls, each with a default:
   `electron_inertia=0`, `integrator="rk4"`, `dt_guard="error"`,
   `minimum_samples_per_cycle=32`, `max_amplitude_error=0.01`, and
   `max_phase_error_rad=0.1`. The selected waveform controls are
   `waveform_rtol=1e-10` and `waveform_max_iterations=12`.
3. Keep `Norm` values identical to native: `linf`, `weighted_l2`, `value`.
4. `Failure` derives from `RuntimeError`, taking `(code, message)` and exposing
   `code`, `checks`, `stage`, and `partial_arrays`.
5. Native result structures are constructed through `types.SimpleNamespace`.
   Export factory functions that initialize every consumed field; no factory
   silently omits a record member.

The default electron model is the massless quasineutral branch. Its physical
electron mass remains positive metadata; `electron_inertia=1` explicitly
selects finite electron inertia.

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

## A1-PY-TYPE-01 - Predictor scheduling identity

1. Append `Parameters.coupling="synchronous"`,
   `Parameters.micro_steps=1`, `waveform_rtol=1e-10`, and
   `waveform_max_iterations=12`. The macro interval remains `Parameters.dt`;
   `nsteps` continues to count accepted macro intervals.
2. Append `StageTag.micro_step=-1`, `iteration=0`, and `slab=0` after the A0
   fields. Existing six-positional-argument construction therefore retains its
   synchronous semantics.
3. A predictor micro-RK4 stage uses `micro_step` in `[0, M)`, `iteration=0`,
   and `slab=0`. The macro endpoint uses `micro_step=-1`. Synchronous A0
   stages retain `micro_step=-1`.
4. `time.coupling` accepts `synchronous`, `predictor`, and `waveform`. A
   synchronous request requires `micro_steps=1`; predictor and waveform require
   massless electrons and RK4. Values must be actual non-boolean types.
   `micro_steps` and `waveform_max_iterations` must be positive integers;
   `waveform_rtol` must be positive, finite, and non-boolean.
5. Stage-generation identity remains unique per RHS evaluation. Equal
   predictor stage times do not collapse `midpoint` and `midpoint_2`; their
   generation, micro index, and stage string remain distinct.

7. Append nullable `ion_nodes`, `wr_initial`, and `wr_relaxation` fields.
   `multirate_wr` requires `ion_nodes` in `{3,5}`, initial in
   `{tangent,constant}`, and relaxation in `(0,1]`. These fields are absent for
   every legacy coupling so old defaults and semantic identities remain
   unchanged.

6. Waveform tags use the macro step, generation, trial, spectral context,
   integer microstep, nonnegative sweep iteration, and slab index. Both
   midpoint stages retain separate tags, generations, and values.
