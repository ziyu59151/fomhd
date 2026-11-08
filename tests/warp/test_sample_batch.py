from __future__ import annotations

import numpy as np
import pytest

from fomhd.backend import Backend
from fomhd.types import Failure


from fomhd import core


def _sequential_matvec(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    rows, columns = matrix.shape
    assert vector.shape == (columns,)
    result = np.empty(rows, dtype=np.complex128)
    for row in range(rows):
        total = 0.0j
        for column in range(columns):
            total += matrix[row, column] * vector[column]
        result[row] = total
    return result


def _sequential_samples(
    matrix: np.ndarray, values: np.ndarray, sample_count: int
) -> np.ndarray:
    columns = matrix.shape[1]
    sample_rows = values.reshape(sample_count, columns)
    return np.concatenate(
        [
            _sequential_matvec(matrix, sample_rows[sample])
            for sample in range(sample_count)
        ]
    )


class _TrackingBackend(Backend):
    def __init__(self) -> None:
        super().__init__("cpu")
        self.mm_calls = 0
        self.mv_calls = 0
        self.mm_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.mm_right: np.ndarray | None = None

    def mv(self, matrix, vector):
        self.mv_calls += 1
        raise AssertionError("sample_apply must use one multiple-RHS mm call")

    def mm(self, left, right):
        self.mm_calls += 1
        self.mm_shapes.append((left.shape, right.shape))
        self.mm_right = np.array(right, copy=True)
        return super().mm(left, right)


@pytest.mark.parametrize(
    ("sample_count", "rows", "columns"),
    [(1, 2, 3), (5, 3, 2), (17, 3, 4)],
    ids=["samples-1", "samples-5", "samples-17"],
)
def test_sample_apply_matches_independent_complex_oracle(
    sample_count: int, rows: int, columns: int
) -> None:
    matrix_index = np.arange(rows * columns, dtype=np.float64)
    value_index = np.arange(sample_count * columns, dtype=np.float64)
    matrix_grid = matrix_index.reshape(rows, columns)
    matrix = (0.25 * matrix_grid + 0.5) + 1j * (
        -0.125 * matrix_grid + 0.75
    )
    values = (0.5 * value_index + 1.0) + 1j * (
        -0.25 * value_index - 0.5
    )
    assert np.all(values != 0.0)
    expected = _sequential_samples(matrix, values, sample_count)

    backend = Backend("cpu")
    matrix_array = backend.array(matrix)
    value_array = backend.array(values)
    matrix_before = matrix_array.copy()
    values_before = value_array.copy()

    actual = core.sample_apply(matrix_array, value_array, sample_count, backend)

    np.testing.assert_allclose(actual, expected, rtol=4.0e-16, atol=4.0e-16)
    assert actual.shape == (sample_count * rows,)
    assert actual.dtype == np.complex128
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(matrix_array, matrix_before)
    np.testing.assert_array_equal(value_array, values_before)
    assert not np.shares_memory(actual, matrix_array)
    assert not np.shares_memory(actual, value_array)

    actual.fill(123.0 - 456.0j)
    np.testing.assert_array_equal(matrix_array, matrix_before)
    np.testing.assert_array_equal(value_array, values_before)


def test_sample_apply_uses_one_multiple_rhs_backend_product() -> None:
    matrix = np.array(
        [
            [1.0 + 2.0j, -3.0 + 0.5j, 2.5 - 1.0j],
            [-0.25 + 4.0j, 2.0 - 0.75j, 1.0 + 3.0j],
        ],
        dtype=np.complex128,
    )
    values = np.array(
        [
            1.0 + 0.5j,
            -2.0 + 1.5j,
            0.25 - 0.75j,
            3.0 + 2.0j,
            -1.0 - 0.25j,
            0.5 + 4.0j,
            -0.5 + 2.5j,
            1.25 - 3.0j,
            2.0 + 0.25j,
            0.75 - 1.5j,
            -4.0 + 0.5j,
            1.5 + 1.0j,
            -2.5 - 0.5j,
            0.25 + 3.0j,
            4.0 - 2.0j,
        ],
        dtype=np.complex128,
    )
    sample_count = 5
    backend = _TrackingBackend()
    expected = _sequential_samples(matrix, values, sample_count)

    actual = core.sample_apply(
        backend.array(matrix), backend.array(values), sample_count, backend
    )

    np.testing.assert_allclose(actual, expected, rtol=4.0e-16, atol=4.0e-16)
    assert backend.mm_calls == 1
    assert backend.mv_calls == 0
    assert backend.mm_shapes == [((2, 3), (3, 5))]
    np.testing.assert_array_equal(
        backend.mm_right, values.reshape(sample_count, 3).T
    )


@pytest.mark.parametrize(
    ("sample_count", "matrix", "values"),
    [
        (0, np.ones((2, 3)), np.ones(3)),
        (2, np.ones((2, 3)), np.ones(5)),
        (2, np.ones((2, 3)), np.ones((2, 3))),
        (1, np.ones(3), np.ones(3)),
    ],
    ids=[
        "nonpositive-samples",
        "wrong-value-length",
        "values-not-1d",
        "matrix-not-2d",
    ],
)
def test_sample_apply_rejects_invalid_sample_shape(
    sample_count: int, matrix: np.ndarray, values: np.ndarray
) -> None:
    backend = Backend("cpu")
    with pytest.raises(Failure, match="sample operator shape"):
        core.sample_apply(
            backend.array(matrix),
            backend.array(values),
            sample_count,
            backend,
        )
