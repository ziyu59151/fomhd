# Foundation operators

Source: `tests/warp/foundation_operators.py`.

## A0-PY-TEST-FOUNDATION-OPERATORS-01 - Procedure and assertions

```text
Build rectangle and stepped-domain operators with deterministic trial vectors.
Require exact integer boundary-of-boundary identity and positive FP64 weights.
For retained Fourier modes check C*G=0, DB*C=0, divergence theorem and the
weighted I/H pairing against independent dot products.
Use tracking namespaces to verify backend tile and solve operations keep the
operator-owned backend; these checks do not claim device execution.
Use deliberately noncommuting transforms: require reversed-order defect,
then verify the implemented forward/inverse round trip.
Solve the zero-mode Dirichlet operator and check all original wall/constraint
rows remain represented after independent-row selection.
Refine the smooth interior gradient and require observed order above 3.5;
reject insufficient memory and rank-defective operator constructions.
```

Covered top-level functions and fixture classes:

- `rectangle`
- `ell`
- `deterministic`
- `test_integer_incidence_and_positive_weights`
- `test_compatible_complex_identities_and_divergence`
- `test_i_h_pairing`
- `_BackendXp`
- `test_validation_uses_backend_tile_on_device_arrays`
- `_TrackingBackend`
- `test_t2_inverse_block_defaults_to_operators_backend`
- `test_noncommuting_transform_order_and_roundtrip`
- `test_full_rows_and_k0_dirichlet_operator`
- `test_gradient_inner_fourth_order`
- `test_operator_memory_and_rank_rejections`

This mirror describes the implemented checks. Passing it is evidence only
for its stated fixtures and backend; it does not imply global Tokamak, ICE,
experimental, or production-performance validation.
