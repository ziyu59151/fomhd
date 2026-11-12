"""Compare the batched wall normal to an independent dense projection."""

from types import SimpleNamespace

import numpy as np
import pytest

from fomhd.backend import Backend
from fomhd.massless import _constraint_project


@pytest.mark.parametrize("coupled", [False, True])
def test_electron_projection_preserves_wall_rows(coupled, monkeypatch):
    rng = np.random.default_rng(38)
    nc, samples = 4, 5
    mass = np.linspace(1.0, 2.0, nc)
    matrix = np.zeros((2, 3 * nc), dtype=np.complex128)
    matrix[0, [0, 1]] = [1.5, -0.5]
    matrix[1, [1 if coupled else 2, 3]] = [-0.5, 1.5]
    density = 1 + rng.random((samples, nc))
    source = rng.normal(size=(samples, 3 * nc)).astype(np.complex128)
    backend = Backend("cpu")
    context = SimpleNamespace(
        p=SimpleNamespace(ref=SimpleNamespace(length=2.0), rank_rtol=1e-10),
        op=SimpleNamespace(Nc=nc, M0=mass, T=matrix, T_full=matrix),
        compute=backend,
        N=samples,
    )
    expected = np.empty_like(source)
    for i in range(samples):
        adjoint = np.tile(density[i] / mass, 3)[:, None] * matrix.conj().T
        expected[i] = source[i] - adjoint @ np.linalg.solve(
            matrix @ adjoint, matrix @ source[i]
        )
    calls = 0
    singular_values = backend.singular_values

    def counted(values):
        nonlocal calls
        calls += 1
        return singular_values(values)

    monkeypatch.setattr(backend, "singular_values", counted)
    actual = _constraint_project(source.ravel(), density.ravel(), context)
    np.testing.assert_allclose(actual.reshape(source.shape), expected, atol=2e-15)
    assert calls == (samples if coupled else 0)


def test_ion_wall_projection_uses_exact_diagonal_normal(monkeypatch):
    from fomhd.wall import build_wall, project_wall

    rng = np.random.default_rng(27)
    nc, samples = 5, 5
    mass = np.linspace(1.0, 2.0, nc)
    matrix = np.zeros((2, 3 * nc), dtype=np.complex128)
    matrix[0, [0, 1]] = [1.5, -0.5]
    matrix[1, [2, 3]] = [-0.5, 1.5]
    density = 1 + rng.random((samples, nc))
    source = rng.normal(size=(samples, 3 * nc)).astype(np.complex128)
    backend = Backend("cpu")
    ion = SimpleNamespace(id="ion", mass=2.0, n_min=0.0)
    context = SimpleNamespace(
        p=SimpleNamespace(
            ref=SimpleNamespace(ni=1.0, ne=1.0, length=2.0, velocity=1.0, time=1.0),
            rank_rtol=1e-10, species=[ion],
        ),
        geometry=SimpleNamespace(lcfs_polygon=[(1.0, 1.0)]),
        op=SimpleNamespace(Nc=nc, M0=mass, T=matrix, T_full=matrix),
        compute=backend,
        N=samples,
    )
    expected = np.empty_like(source)
    for i in range(samples):
        adjoint = np.tile(ion.mass * density[i] / mass, 3)[:, None] * matrix.conj().T
        expected[i] = source[i] - adjoint @ np.linalg.solve(
            matrix @ adjoint, matrix @ source[i]
        )

    def unused(*args):
        raise AssertionError("a diagonal wall normal does not need an SVD")

    monkeypatch.setattr(backend, "singular_values", unused)
    wall = build_wall(density.ravel(), ion, None, context)
    actual = project_wall(wall, source.ravel(), context)
    np.testing.assert_allclose(actual.value.reshape(source.shape), expected, atol=2e-15)
    np.testing.assert_allclose(
        actual.reaction.reshape(source.shape), expected - source, atol=2e-15
    )


def test_diagonal_electron_projection_rejects_nonfinite_solution():
    from fomhd.types import Failure

    matrix = np.array([[1.0, 0.0, 0.0]], dtype=np.complex128)
    context = SimpleNamespace(
        p=SimpleNamespace(ref=SimpleNamespace(length=1.0), rank_rtol=1e-10),
        op=SimpleNamespace(Nc=1, M0=np.ones(1), T=matrix, T_full=matrix),
        compute=Backend("cpu"),
        N=1,
    )
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(Failure, match="finite"):
        _constraint_project(
            np.array([1e200, 0.0, 0.0], dtype=np.complex128),
            np.array([1e-150], dtype=np.complex128),
            context,
        )
