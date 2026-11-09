# Accepted WR metadata publication tests

```text
Build and advance a real closed-wall A1-WR macro to acceptance.

Insert a three-element NumPy vector into the accepted coupling metadata:
  call real io.Run.time_coupling_evidence
  reload run.json through json.loads
  require the exact vector values/dtype-level meaning in JSON
  recursively scan metadata JSON for no ndarray/backend object remains
  require source metadata and accepted state/cache/cumulative unchanged

Call real Run.failure with phase="post_commit_output":
  require accepted_trial=True and output_after_commit classification
  require the state/cumulative remain the accepted values

Separate sibling regression test_failure_publication.py:
  require rejected publication still preserves all 119 numerical arrays
    and explicitly excludes the four known structured records
```
