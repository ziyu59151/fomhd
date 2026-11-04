# Longitudinal plot evidence checks

## A0-PY-TESTPLOTLONG-01: Offline history and rejection controls

Use numeric NPZ fixtures with a prescribed quarter-cycle electron mode.
Check normalized electron coefficients and cycle coordinates, preserved
summary/control evidence, and hashes for all inputs. Corrupt completion,
horizon, or history identity and require rejection. Existing output must
be refused before reading inputs, preserving the user's file.
Reject missing, empty or duplicate amplitude controls, missing paired
half-amplitude cases, and a control value inconsistent with that case pair.
