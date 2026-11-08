# Test alias retained-history ownership

Source: `tests/warp/test_alias_history_ownership.py`.

## Independent constraints

The focused file imports `core.py` and `massless.py` from the selected scratch
override and asserts those local paths before testing behavior. The EAST32
fixture stores dimensions and item sizes only; it does not allocate the large
operator arrays.

The expected plans are literal arithmetic, not values obtained from the
functions under test:

```text
outer waveform N=5 plan = 5,081,734,872 bytes
alias N=11 plan        = 5,262,540,504 bytes
one-stage delta        = 180,805,632 bytes
```

## Tests

```text
test_focused_tests_import_selected_scratch_files:
  fail unless core.py and massless.py resolve to the scratch overlay

test_alias_history_keeps_outer_samples_and_charges_only_alias_workspace:
  shape-only EAST32 K1 waveform M4 fixture
  require outer history entries = 179,995,368
  require outer wall-history entries = 27,412,480
  require outer plan = 5,081,734,872
  set alias N=11 and history samples=5
  require both history counters unchanged
  require alias plan = 5,262,540,504
  require plan delta = 180,805,632 and alias plan below 8 GiB
  require preflight accepts the literal alias plan

test_history_sample_identity_rejects_invalid_values:
  both history counters reject bool, negative, zero, and >N identities

test_nested_alias_probe_propagates_original_history_samples:
  first alias changes N=5 to N=11 and records 5
  nested alias changes N=11 to N=23 and still records the original 5
  microstep identity remains unchanged
```

## Evidence

Before implementation, the focused run produced 12 expected failures and one
pass: the alias history counter was `393,695,976`, invalid identities were
accepted, and the alias copy lacked `_history_samples`. After implementation,
all 13 focused tests passed. The bounded CPU gate then passed 425 tests with
seven dependency/GPU skips and one unavailable Warp-subprocess test deselected.

The numerical fixture rerun passed the genuine tiny massless predictor interval,
actual alias wall-cache stage, two-macro T1 continuation, and T1 rollback. The
correction changes planning identity only; it changes no physical, diagnostic,
or fixed-wall-cache operation.
