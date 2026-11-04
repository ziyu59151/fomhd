# Test diagnostic evidence io

Source: `tests/warp/test_diagnostic_evidence_io.py`.

## A0-PY-TEST-TEST-DIAGNOSTIC-EVIDENCE-IO-01 - Procedure and assertions

```text
Create ordinary and extended wall-work diagnostic records.
Publish them through normal and failure I/O paths and load the JSONL.
Require the fixed base keys plus only the allowed optional bound/work keys;
retain SI W units, signed work, separate direct/compatibility/arithmetic bounds,
modal/sample work and transform difference.
Encode each nonfinite scalar using the documented typed JSON representation.
Do not serialize arbitrary private attributes or runtime device arrays.
```

Covered top-level functions and fixture classes:

- `_extended_record`
- `_read_jsonl`
- `test_optional_wall_work_evidence_survives_normal_and_failure_json`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.
