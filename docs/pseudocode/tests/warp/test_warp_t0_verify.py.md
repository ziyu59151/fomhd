# Test `warp_t0_verify`

Source: `tests/warp/test_warp_t0_verify.py`.

## A1-PY-TEST-T0-VERIFY-01 - Output and control refusal

```text
Run the real tool entry points with invalid devices, point counts, cycle
counts, microstep sets, repetition counts and incompatible H/horizon counts.
Require ValueError before output creation.
Create an output directory with a marker, request the same output, and require
FileExistsError with the marker and contents unchanged.
Require CUDA without SLURM_JOB_ID to return status 2 before output creation.
```

## A1-PY-TEST-T0-VERIFY-02 - Phase and error definitions

```text
Construct independent synthetic complex modal references.
Require amplitude ratio from the projection and signed phase from atan2 in
[-pi,pi], including wrap near +pi.
Require the labels to state modulo-2*pi and principal-angle ambiguity.
Require analytic_mode to apply exp(lambda*T).
Require mode_samples to produce real cosine/sine samples and recover the same
complex coefficient; this catches an unphysical imaginary quadrature.

Construct six independent synthetic field records and explicit physical
scales. Require one SI result per field with absolute error, reference
perturbation norm, floor, scale and relative error. Verify zero linear
components are marked as amplitude-floor limited. Require P, Z and full P+Z
dimensionless aggregates.

Multiply one field's actual, reference, baseline and physical scale by a
common factor. Require every per-field relative error and every aggregate to
remain invariant. This catches accidental concatenation of mixed units and
loss of per-field normalization.

Corrupt derived electron density and compatibility momentum, canonicalize
through fomhd.massless.canonicalize_state, and require nonzero SI corrections
for both fields plus a nonzero compatibility aggregate. Do not treat unchanged
independent P/Z as evidence that reconstruction occurred.
```

## A1-PY-TEST-T0-VERIFY-03 - Metadata and compact checks

```text
Provide actual predictor metadata for M=4 over two macros.
Require aggregate complete, full-RHS, endpoint, alias and total evaluations;
retain expected per-macro values and the last tangent correction.
Inject a wrong complete-stage count and require a metadata failure.

Provide passing, soft-failing, hard-failing and unavailable checks.
Require accurate counts, retained failed records, and no retention of passing
records. Repeated failures are grouped with counts.
```

## A1-PY-TEST-T0-VERIFY-04 - Closed fixture and failure retention

```text
Build the existing closed context with hard diagnostics enabled.
Record all budget hard flags, apply the tool's smooth density/pressure
perturbation, and require nonzero independent P and Z perturbations.
Require every budget hard flag to remain unchanged.

Mock only the backend/operator construction boundary and request cuda:0.
Require the constructed context to retain compute.device=cuda:0. Inject a
backend that reports CPU and require BackendIdentity failure with no fallback.

Test `_closed_initial_state` CPU identity by requiring the existing fixture
helper to receive the requested CPU context unchanged.

Test non-CPU setup routing with a mechanical fake target backend. Require the
existing CPU-only uniform/nullspace helper to receive only a CPU setup context,
never the target context. Require exactly one complex128 upload for each
species n,mom,perp,parallel and B (nine arrays), preserving values, shapes,
step, generation and time. This is routing/ownership evidence only, not GPU
numerical parity.

Use a bounded synthetic closed trajectory/execution record to exercise
run_closed_case through _execute_case and _success_artifacts. Require NPZ and
metadata artifacts, explicit `resources`, and before/after derived
compatibility arrays. This synthetic record proves output integration only;
it is not labeled physical evidence.

Inject a solver Failure with one failed check and one partial array into the
periodic case runner.
Require report.json to retain a failed case, failure JSON to retain the check,
and failure NPZ to retain the array.
```

## A1-PY-TEST-T0-VERIFY-05 - Provenance minimization

```text
Supply selected Slurm/CUDA environment values plus an unrelated secret.
Require provenance to retain only the selected two environment keys, exact
argv, UTC, selected device, purpose, Git SHA/dirty flag, Python/library
versions, and SHA-256 hashes for runtime module, fixture support files, tool,
pyproject and uv.lock.
Require the secret to be absent.
```

Passing these tests proves harness behavior only. It does not establish T0
scientific accuracy, a GPU result, a speedup, closed-wall acceptance, or
production restart validity.
