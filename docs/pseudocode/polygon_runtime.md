# Polygon native-I/O and bounded circular runtime pseudocode

Status: implementation contract for `polygon_dual_v1`. This document precedes
source and test implementation. The existing massless Hall/CGL stage, RK4 and
few-node waveform-relaxation transactions remain the only physics and time
solver.

## Layout and state

All state is SI and mode-major complex128, with retained Fourier modes
`-K..K` stored in that order.

```text
scalar:  n_i, n_e, p_i, p_e_perp, p_e_parallel  [mode, Nc]
velocity mom_i, mom_e                            [mode, 3, Nc]
E:       [E_n Nf, E_phi Nc]
B:       [B_phi Nv, B_t Nf]
```

`Nc` cell entities are polygon area centroids. Generator coordinates remain
separate geometry metadata. `Nf` is the plane-normal facet table and `Nv` is
the dual-vertex table. The compatibility members `r_faces` and `z_faces` do
not carry radial and vertical field meanings in this layout.

## Geometry identity

Compute a SHA-256 geometry fingerprint from canonical shape, dtype and bytes
for:

1. layout kind, schema and reconstruction version;
2. the complete input polygon and every cell polygon in cell order;
3. entity keys and ordering;
4. generator centroids;
5. face/vertex incidence and face ownership;
6. oriented normals, tangents and true endpoints;
7. cell/vertex measures, face lengths, normal distances and metric vectors.

Reject nonfinite, inconsistent or wrong-kind geometry. Never infer native
field semantics from legacy array names.

## Native state codec

`encode_state(state, geometry)`:

1. If geometry is not `polygon_dual_v1`, dispatch to the unchanged legacy
   codec outside this module.
2. Infer `M=2K+1` from the requested/accepted mode count and validate every
   compact state extent.
3. Require host complex128 arrays with exact native shapes and finite values.
   Require the central mode to be real and
   `x[K-m] = conjugate(x[K+m])`; this rejects reversed or otherwise wrong mode
   ordering.
4. Emit fluid arrays `[mode,Nc]`, momentum arrays `[mode,3,Nc]`,
   `B_phi_native [mode,Nv]` and `B_t_native [mode,Nf]`. Do not emit,
   scatter or relabel `B_t` as `B_Z`.
5. Emit `layout_schema`, `geometry_fingerprint`, entity counts/order IDs,
   generator coordinates and a JSON native-entity metadata record.

`decode_state(arrays, geometry, K)` applies the same checks, rejects missing
or mismatched schema/fingerprint/shape/dtype/mode ordering, reassembles
`B=[B_phi,B_t]`, and returns a fresh state. No legacy checkpoint is guessed
into the native layout.

`encode_vertex` and `decode_vertex` use the same identity checks but preserve
the native baseline divergence cochain `[mode,Nv]`. Legacy vertex scatters are
unchanged.

The operator fingerprint prepends layout kind/version and the geometry
fingerprint. It also hashes the actual bounded `G/C/DB/DJ/Cdag`, `I/H/S`,
`scalar_I/QB/T/T_full`, and `M0/M1/MB` payloads, selected wall rows and
declared reconstruction metadata. Backend arrays are exported explicitly with
`backend.to_host` during setup; sparse storage hashes its CSR buffers rather
than densifying. An optional legacy matrix string is supplemental and cannot
substitute for these payload checks.

At an explicit snapshot, checkpoint, initial-RHS or failure-output boundary,
the runner first exports every state field through `compute.to_host` into a
host `State`, then invokes the codec. No stage performs this copy implicitly.

## Circular context

`make_context()`:

1. Validate all CLI values before creating the output directory.
2. Call `tools.warp_tokamak_cae.physical_parameters(K,n0=2e19,B0=2,L=.5,
   pressure=1000)` and preserve its budget registry, hard/observe gate choices
   and observation policy.
3. Set `Nphi=4K+1`, massless-electron, RK4 and the selected coupling.
4. For synchronous coupling use `M=1`, `dt=h`; for WR use
   `coupling=multirate_wr`, `dt=M*h`, `micro_steps=M`, `ion_nodes=s`.
5. Check 32 GiB host and 16 GiB device caps against 75 percent of currently
   available memory from inside the intended allocation. CUDA additionally
   requires the existing owned-Slurm check.
6. Apply `--grid-shift-R/Z` as fractions of the local generator spacing. The
   full polygon box remains covered for shifts in `[-0.5,0.5]`.
7. Build the numerical circle polygon and either a polygon geometry or the old
   staircase geometry, then build the matching operators.

Polygon node counts are not capped by the retired 40-node experiment limit.
They are checked by actual geometry construction and actual operator-storage
estimation against the selected CPU/CUDA cap. `--preflight-only` builds the
actual geometry but does not allocate operators or state; it reports entity
counts, measured geometry coverage/quality diagnostics, estimated operator
bytes, cap headroom, and the estimate implementation.

Every run and preflight writes exactly one deterministic `geometry.npz`. It
stores flattened `cell_polygon_points` with offsets/counts, cell centers,
`M0`, true physical wall distance, input circle, numerical wall polygon, face
vertices/coordinates/normals/tangents/lengths/areas/volumes and the geometry
fingerprint. The report and accepted checkpoint carry its relative filename
and SHA-256. Fixed ZIP timestamps make the file hash stable across frozen CPU
or GPU checkout roots.

The default `h` is only a finite positive input supplied by the caller. It is
not labeled or claimed as a CFL value. `--waveform-rtol` defaults explicitly to
`1e-12` for this new runner; the old WR history/default contract is unchanged.
`--amplitude 0` is a valid unperturbed control.
`--seed=smooth` is the default. `--seed=cylindrical-caw` is mode-0 only and
lazily imports `tools.cylindrical_caw_control.seed_velocity`; this runner
passes the actual cell sample positions and model scales but does not
duplicate any CAW formula. The helper supplies the t0 velocity only, so scalar
pressure and magnetic perturbations remain zero. Missing or incompatible
helpers fail with `IntegrationUnavailable`. Seed selection is part of the
scheme/input fingerprint and snapshot metadata.

## Initialization

Create state only through `core.allocate_state`.

For vacuum:

1. Fill both densities with `n0`, ion pressure with `600 Pa`, and electron
   perpendicular/parallel pressures with `400 Pa`.
2. Set ion momentum to zero.
3. Set `B_phi=F0/Rbar_v` in the native graph cochain. The old staircase path
   writes only its old B-phi block.
4. Retain raw continuous `F/R` samples, the discrete harmonic and their
   difference for the report.

The polygon raw sample uses the actual face/vertex radius
`geometry.vertices.R`; the mapped harmonic remains `F0/Rbar_v`. Their actual
difference is retained and must not be constructed as zero.

Initialization metadata also records the exact compact equilibrium values
already used: `R0_m`, `a_m`, `F0_T_m`, `n0_m^-3`, `p_edge_Pa`, `pprime_Pa_per_Wb_rad`,
`alpha_ion_fraction` and `degree` (`0` for vacuum, `10` for the GS control).
The perturbation metadata includes `vA_m_s=reference.velocity` for reporting
the supplied seed scale.

For circular GS:

1. Build `tools.circular_equilibrium.build_gs_equilibrium` with the circular
   major/minor radii, `F0=B0*R0`, `p_edge=pressure`, positive `pprime`, and
   `alpha=0.6`; reject a zero or negative pressure on the parsed wall.
2. Sample density and species pressures at polygon cell centroids.
3. Set harmonic `B_phi=F0/Rbar_v`.
4. Form `B_t` from the cell flux using the existing normal-gradient operator,
   divided by face radius. This is the discrete curl-free poloidal control;
   impose zero Dirichlet wall flux by construction.
5. Record raw continuous face tangential field, continuous normal field,
   projected-normal interpretation, and discrete tangential field. Do not
   subtract the physical RHS from the state.

For the old staircase GS control, do not sample `sample_gs` at every vertex:
some old-grid vertices lie outside the continuous circular wall. Use the
provided `C @ Aphi + F0*conj(topology_flux_row)/MB` mapping, with continuum
flux/derivatives sampled only at active cell centers. Vertex `B_phi`
comparisons retain an inside-circle validity mask; no exterior extrapolation
is introduced.

The default perturbation is a smooth radial velocity profile that vanishes at
the true polygon wall. For mode 0 it is axisymmetric; for mode 1 it is a
physical real `cos(phi)` harmonic represented by equal half-amplitude complex
coefficients at modes `-1` and `+1`. Project the complete momentum cochain so
`T_full*mom=0`, retaining raw and projected seeds and their weighted
difference.

The initialization `initial-rhs.npz` also exports the six actual force
decompositions already produced by the stage, each as physical
`[Nphi,3,Nc]` complex128 samples:

```text
frame_ion_pressure_force_samples
frame_ion_magnetic_force_samples
frame_ion_electric_force_samples
frame_ion_convection_samples
frame_ion_wall_reaction_samples
frame_ion_net_force_samples
```

Pressure, magnetic and convection fields come from `frame.fluid[0]`; electric,
wall reaction and net force come from `frame.terms[0]`. These arrays are moved
with `compute.to_host` and reshaped only at initial output. They are not
recomputed, sign-relabeled or synthesized for absent force terms.

## Transaction and checkpoint lifecycle

For a fresh run:

1. Create cumulative state with `types.cumulative_record`.
2. Call `stage.initialize` exactly once; retain its actual stage result as the
   initial RHS/cache evidence and snapshot it before any advance.

For a resume:

1. Read strict NPZ metadata and require the native checkpoint schema, layout
   fingerprint, source/scheme/mesh/input identities and
   `complete_accepted=true`.
2. Rebuild and re-evaluate the context from current inputs, decode the
   checkpoint state and restore the full cumulative record.
3. Call `stage.evaluate_stage` once with a restart tag to establish the current
   physical cache. Do not call `stage.initialize`, which would reset the
   accepted cumulative baseline.

Each accepted window:

1. Check the wall deadline only between windows. A started `time.advance` is
   allowed to complete.
2. Call `time.advance(committed,cumulative,context)` exactly once.
3. Publish diagnostics and snapshots only after the returned commit.
4. Atomically checkpoint the accepted endpoint, generation, next
   generation/trial/source counters and full native cumulative record.

Timing boundaries call the selected backend synchronize immediately before and
after setup, initialization or restart evaluation, each complete advance, and
each output write. Report setup time separately from initialization, advance
and output time.

Device pool used and reserved bytes are reported separately. Their sampled
peaks are taken only at setup, initialization/resume completion, complete-step
boundaries and final completion; the report explicitly says in-stage temporary
peaks are unsampled. Host resident VmHWM remains the actual host high-water
mark. The final pool allocation is never labeled as a measured peak.

Under CuPy 14.2, `reserved` means `MemoryPool.total_bytes()`; used bytes remain
`MemoryPool.used_bytes()` and free pool bytes are reported as
`total_bytes - used_bytes`. A diagnostic sampling exception is recorded under
`sampling_errors` with corresponding fields set unavailable; it must not mask
the original scientific failure or substitute zero for an unmeasured GPU pool.

For a complete WR macro, copy the existing `result.coupling_metadata` fields
directly into the advance report, including `sweeps`, `residual_history`,
`work`, `interval_attribution`, endpoint residual summaries and endpoint
transaction flags. Failure receipts likewise copy the existing
`failure.coupling_metadata`. No work or iteration count is inferred by the
runner.

Timing reports retain `total_s` and separately identify `attributed_s` and
`unattributed_s`. The unattributed scope is validation, final bookkeeping and
final JSON publication outside the measured setup/initialization/resume/
advance/output boundaries.

On any input/numerical failure, preserve the previous checkpoint and write a
strict JSON failure receipt plus actual NPZ evidence. Do not mutate the
accepted cumulative ledger from a trial failure.

Checkpoint identity includes coupling, `h`, `H`, `M`, `s`,
`waveform_rtol`, grid shifts, amplitude, retained modes, `Nphi`, geometry and
the physical baseline. The target total step count may be extended on restart;
the saved accepted prefix must remain within the new bound.

## Snapshots and report

At `t0` and requested accepted endpoints, move arrays to the host and save:

```text
B_N_samples                         [Nphi,3,Nc]
ion_velocity_samples                [Nphi,3,Nc]
ion_density_samples                 [Nphi,Nc]
ion_perp_pressure_samples           [Nphi,Nc]
electron_density_samples            [Nphi,Nc]
electron_perp_pressure_samples      [Nphi,Nc]
electron_parallel_pressure_samples  [Nphi,Nc]
```

These are the actual physical samples from `frame.BN` and
`frame.fluid[0/1]`; velocity is never formed by dividing Fourier momentum by
Fourier density. Component arrays `B_N_R/phi/Z_samples` and
`ion_velocity_R/phi/Z_samples` are also emitted. There is no density floor.

Native coefficients are separate arrays named `n_i_modes`, `mom_i_modes`,
`ion_perp_pressure_modes`, the corresponding electron fields, and
`B_phi_modes/B_t_modes`. Metadata records `sample_phis = 2*pi*j/Nphi`, the
retained mode axis `-K..K`, and explicit SI units for samples and modes.
`current_modes` is `compute.to_host(frame.current).reshape(modes,Ne)` and
remains complex128; it is never passed through the physical-sample validator.

Do not retain whole-state host copies during a stage.

The strict JSON report records:

1. resolved CLI parameters and source/input/geometry/operator fingerprints;
2. global and segment step counters, accepted intervals and raw checks;
3. `h`, WR `H=M*h`, expected horizon and actual endpoint;
4. initialization, advance and diagnostic timings;
5. snapshot/checkpoint paths and bounded counts;
6. host RSS peak and device peak memory;
7. explicit scientific limitations.

The actual restart regression uses N=8, amplitude zero and `h=1e-12`, not the
rejected N=4/`1e-10` launch. Prefix and uninterrupted target totals are both two
steps; only `segment_steps` differs. It compares `n_i_modes`, `mom_i_modes`,
`n_e_modes`, `mom_e_modes`, `B_phi_modes` and `B_t_modes`, and checks accepted
step, interval count and physical time.

The source fingerprint is a SHA-256 of a manifest keyed only by declared
relative source paths and per-file hashes. Absolute checkout paths never enter
the digest, so frozen CPU and GPU roots with identical sources compare
identically.

Status is `completed` only when the requested segment and global horizon are
fully accepted. A deadline stop is `budget_stopped`; failures are `failed`.
No worker-only run establishes scientific success.

## Test-first sequence

1. Red: fake native geometry satisfies the frozen interface enough for actual
   `State` objects; round trip and negative mutation tests fail because no
   native codec exists.
2. Green: implement the minimum codec, identity checks and dispatch.
3. Red: runner parser/cap/horizon checks fail.
4. Green: implement context/control validation and transaction helpers.
5. Run the old I/O target tests against the full overlay package.
6. When the polygon geometry/operator files are available, run small real-mesh
   vacuum and GS initialization/snapshot/restart tests. Otherwise report those
   exact tests as pending integration seams, never as passes.
