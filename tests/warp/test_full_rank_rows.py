from __future__ import annotations

import numpy as np
import pytest

from fomhd.backend import Backend
from fomhd.types import Failure


def _old_prefix_rows(
    backend: Backend, host_matrix: np.ndarray, rank_rtol: float
) -> list[int]:
    """Preserve the pre-gate sequential normalization and SVD decision."""
    matrix = backend.array(host_matrix)
    selected: list[int] = []
    current_rank = 0
    for row in range(matrix.shape[0]):
        magnitude = float(backend.xp.linalg.norm(matrix[row]).item())
        if magnitude == 0.0:
            continue
        candidate = backend.xp.empty(
            (len(selected) + 1, matrix.shape[1]), dtype=backend.xp.complex128
        )
        for offset, source in enumerate(selected):
            source_norm = float(
                backend.xp.linalg.norm(matrix[source]).item()
            )
            candidate[offset] = matrix[source] / source_norm
        candidate[len(selected)] = matrix[row] / magnitude
        values = backend.singular_values(candidate)
        new_rank = (
            int(
                backend.xp.count_nonzero(
                    values > rank_rtol * float(values[0].item())
                ).item()
            )
            if values.size
            else 0
        )
        if new_rank not in (current_rank, current_rank + 1):
            raise Failure("Linear", "independent-row rank is ambiguous")
        if new_rank == current_rank + 1:
            selected.append(row)
            current_rank = new_rank
    return selected


def _assert_old_prefix_result(
    backend: Backend,
    host_matrix: np.ndarray,
    rank_rtol: float,
    expected: list[int],
) -> None:
    actual = backend.independent_rows(backend.array(host_matrix), rank_rtol)
    assert actual == expected
    oracle = _old_prefix_rows(Backend("cpu"), host_matrix, rank_rtol)
    assert actual == oracle


class _CountingBackend(Backend):
    def __init__(self) -> None:
        super().__init__("cpu")
        self.svd_calls = 0

    def singular_values(self, matrix):
        self.svd_calls += 1
        return super().singular_values(matrix)


class _FailingSvdBackend(Backend):
    def __init__(self, fail_on_call: int) -> None:
        super().__init__("cpu")
        self.svd_calls = 0
        self.fail_on_call = fail_on_call

    def singular_values(self, matrix):
        self.svd_calls += 1
        if self.svd_calls == self.fail_on_call:
            raise Failure("Linear", "injected singular-value failure")
        return super().singular_values(matrix)


def test_full_rank_real_rows_use_one_svd() -> None:
    """The proof gate must avoid one SVD per prefix."""
    backend = _CountingBackend()
    matrix = np.array(
        [
            [1.0, 2.0, 0.0],
            [-1.0, 1.0, 1.0],
            [0.0, 1.0, -1.0],
        ]
    )
    _assert_old_prefix_result(backend, matrix, 1e-12, [0, 1, 2])
    assert backend.svd_calls == 1


def test_complex_full_rank_rows_match_old_prefix_oracle() -> None:
    """Complex normalization must preserve the selected-row semantics."""
    matrix = np.array(
        [
            [1.0 + 1.0j, 1.0, 0.0],
            [0.0, 2.0 - 1.0j, 2.0j],
            [0.0, 0.0, 3.0 + 0.5j],
        ],
        dtype=np.complex128,
    )
    _assert_old_prefix_result(Backend("cpu"), matrix, 1e-12, [0, 1, 2])


def test_rank_deficient_rows_fall_back_and_match_old_prefix_oracle() -> None:
    """A failed full-rank proof must preserve deterministic prefix selection."""
    backend = _CountingBackend()
    matrix = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
        ]
    )
    _assert_old_prefix_result(backend, matrix, 1e-12, [0, 1])
    assert backend.svd_calls == 4


def test_zero_rows_are_skipped_in_original_order() -> None:
    """Zero rows must not be selected or reorder surviving rows."""
    backend = _CountingBackend()
    matrix = np.array(
        [
            [1.0 + 2.0j, 0.0],
            [0.0, 0.0],
            [-3.0j, 1.0],
        ],
        dtype=np.complex128,
    )
    _assert_old_prefix_result(backend, matrix, 1e-12, [0, 2])
    assert backend.svd_calls == 1


def test_all_zero_rows_do_not_run_svd() -> None:
    """An all-zero matrix has no full-rank matrix to allocate or probe."""
    backend = _CountingBackend()
    actual = backend.independent_rows(backend.array(np.zeros((3, 2))), 1e-12)
    assert actual == []
    assert backend.svd_calls == 0


@pytest.mark.parametrize(
    ("small_angle", "expected"),
    [
        (3.0e-8, [0, 1]),
        (8.0e-9, [0]),
    ],
    ids=["just-above-threshold", "just-below-threshold"],
)
def test_near_threshold_rows_match_old_prefix_oracle(
    small_angle: float, expected: list[int]
) -> None:
    """The existing strict rank threshold must decide both sides identically."""
    matrix = np.array(
        [
            [1.0, 0.0],
            [np.cos(small_angle), np.sin(small_angle)],
        ]
    )
    _assert_old_prefix_result(Backend("cpu"), matrix, 1e-8, expected)


def test_tall_matrix_matches_old_prefix_oracle() -> None:
    """An m>n matrix must skip the impossible full-rank probe."""
    backend = _CountingBackend()
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 2.0],
        ]
    )
    _assert_old_prefix_result(backend, matrix, 1e-12, [0, 1])
    assert backend.svd_calls == 4


@pytest.mark.parametrize(
    ("value", "rank_rtol", "code"),
    [
        (np.array([1.0, 2.0]), 1e-12, "Shape"),
        (np.array([[1.0, np.nan]]), 1e-12, "Finite"),
        (np.array([[1.0, 0.0]]), 0.0, "Linear"),
    ],
    ids=["not-a-matrix", "nonfinite", "invalid-tolerance"],
)
def test_invalid_inputs_preserve_failure_codes(
    value: np.ndarray, rank_rtol: float, code: str
) -> None:
    """Input validation must remain ahead of every optimization decision."""
    backend = Backend("cpu")
    with pytest.raises(Failure) as caught:
        backend.independent_rows(backend.array(value), rank_rtol)
    assert caught.value.code == code


@pytest.mark.parametrize(
    ("matrix", "fail_on_call"),
    [
        (
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
            1,
        ),
        (
            np.array(
                [
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 1.0, 2.0],
                ]
            ),
            2,
        ),
    ],
    ids=["full-rank-probe", "sequential-fallback"],
)
def test_singular_value_failure_propagates(
    matrix: np.ndarray, fail_on_call: int
) -> None:
    """SVD failures must remain failures in the probe and in the fallback."""
    backend = _FailingSvdBackend(fail_on_call)
    with pytest.raises(
        Failure, match="injected singular-value failure"
    ) as caught:
        backend.independent_rows(backend.array(matrix), 1e-12)
    assert caught.value.code == "Linear"
