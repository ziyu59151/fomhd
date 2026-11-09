# `warp_multirate_wr.py` runner pseudocode

```text
parse required --input and --output
read JSON without importing the solver from the runner module's import phase
validate exact required keys and physical domains
use h=DEFAULT_H=1.9794733885623e-10 s when h is omitted
refuse if output exists

resolve all tool/source/test/pseudocode paths from SOURCE_ROOT =
dirname(dirname(realpath(__file__))); never depend on the worker scratch root

build the requested context:
  periodic -> copied physical HallCase/Context/operator builders
  east    -> explicit temporary unsupported response with the exact pinned
             parent-owned input/factory seam; no representative claim

record source hashes before execution
record baseline CPU RSS and CuPy pool samples
hard-gate actual host RSS at 16 GiB and allocated device bytes at 8 GiB
record, but do not gate, CuPy pool reserved bytes

for macro in range(macros):
  adapter.solve_window(accepted P/Z, H=h*M, M, s, ...)
  require residual <= rtol
  accept only window.P[-1] and window.Z_endpoint
  retain mapped P separately
  append mode amplitudes, endpoint source/check/work evidence
  append compact macro scalar/counter evidence only
  do not retain a completed full WindowPhysicsResult, P/Z arrays, or records
  keep the last full result only for final Q/fine-array extraction
  append exact accepted s coarse and 4M fine evidence for the final result
  retain all-callback counts separately
  append a bounded state snapshot
  check wall budget after each completed window

run independent synchronous full-RHS RK4 references from the source P/Z:
  reference_h   uses M substeps per H
  reference_h2  uses 2M substeps per H
  both use identical macro endpoint times and check scope
  both use alias at RK4 middle stages; this is correctness-only reference work

compare WR/source, WR/reference_h, WR/reference_h2
mark endpoint trial=true and interval_transaction_required=true
serialize arrays into one in-memory NPZ, base64 it into JSON
write a temporary sibling file, flush, fsync, rename to exact target
on failure retain incomplete report/evidence and nonzero exit status
```
