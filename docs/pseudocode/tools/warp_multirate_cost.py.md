# `tools/warp_multirate_cost.py`

## Purpose

Measure the existing C1/C2 research candidate against one independent, lean,
synchronous explicit-RK4 reference using the same input, horizon, device,
warmups, repetitions, and synchronization protocol.

The lean reference is a cost oracle. It owns no accepted production
transaction, interval budget, alias ledger, energy ledger, or checkpoint.

## Inputs

```text
--input-case-directory DIR   existing candidate output containing input.json
--device cpu|cuda:0
--output NEW_DIR
--warmups 1
--repetitions 3
```

Load and fingerprint the exact `input.json`. Preserve its case, points,
amplitude, nonuniformity, seed, sentinel, method, `h`, `M`, macros, and
horizon. Override only the requested device and timing repetitions.

A CUDA request requires the process to own a Slurm allocation and sets a
CuPy default-pool limit of 8 GiB.

## Lean window

For each macro window:

```text
adapter = ReducedHydroAdapter(context)
p = p_window
z = z_window
for micro in 0..M-1:
    for rk in 0..3:
        stage_p, stage_z = RK4 input at the distinct stage identity
        D, A_m, A_p = adapter.macro(stage_p, stage_z)
        G, E, B = adapter.field(stage_p, stage_z)
        L = adapter.lorentz(stage_p, E, B)
        rhs_P = D + A_m + A_p + L
        rhs_Z = G
        retain stage identity, P, Z, D, A_m, A_p, L, G, E, B
    advance p and z by explicit RK4
endpoint = adapter.endpoint(p, z)
```

There are exactly `4*M*macros` full-hydro macro evaluations, matched by
`4*M*macros` D and L evaluations. There are `4*M*macros + macros` field G
evaluations because each endpoint reconstruction is a separate call. Endpoint
work and compatibility checks are included in timing.

## Matched timing

Build the case once. For each algorithm, execute in a fresh isolated context:

```text
cold
warmups
repetitions
```

Synchronize before and after every timed execution when using CUDA. The timed
scope includes adapter construction, all coupling and endpoint reconstruction,
and all stage work. Fixture construction, input I/O, host array extraction,
plots, and pre/post diagnostics are outside that scope.

The returned execution supplies the only complete stage history written to
disk. Cold, warmup, and repetition timings retain only scalar evidence.

## Evidence

Report separately:

- matched candidate-versus-lean window cost;
- the existing candidate-versus-full-production timing copied from the input
  case, if present;
- candidate endpoint versus lean endpoint;
- lean endpoint versus the existing full-production raw oracle, if present;
- equal pre/post diagnostics outside timing;
- state and periodic target-mode phase errors;
- actual outer-sweep and full-hydro/D/L/G/endpoint counters;
- source, input, candidate-output, actual Git, RSS, and CuPy pool identities.
  The Git record is observed from the source checkout; no synthetic baseline
  commit is inserted.

No timing ratio is an acceleration claim. The parent owns the scientific gate
and any GPU deployment.

Preserve candidate and reference endpoints explicitly with the complete stage
arrays across all macro windows. Validate stage identities in every window,
report both total and per-window stage counts, synchronize outside-timer
physical checks, and record shared fixture setup separately. Device-pool bytes
are post-campaign allocator samples; no instrumented allocation peak is claimed.

## Host/device phase observation

Endpoint/oracle arrays stored on the host are converted with the owning context's
`xp.asarray` before backend-sensitive mode reconstruction. This explicit copy is
postprocessing outside matched algorithm timing and is included in total elapsed
runtime. Never pass NumPy arrays directly to a CUDA `core.inverse` operation.
