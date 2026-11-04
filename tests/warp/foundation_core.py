from __future__ import annotations

import numpy as np
import pytest

from fomhd.backend import Backend
from fomhd.core import (
    allocate_state,
    check_retained,
    combine,
    estimated_peak_bytes,
    forward,
    inverse,
    mode_apply,
    sample_apply,
    time_at_midpoint,
    time_at_step,
)
from fomhd.geometry import build_rectangle
from fomhd.operators import build_operators
from fomhd.types import Context, StageTag, State

from foundation_fixture import parameters


def context():
    p = parameters(K=1)
    geometry = build_rectangle(4, 3, 0.5, -0.2, 0.1, 0.1)
    backend = Backend("cpu")
    operators = build_operators(geometry, 1, backend=backend)
    return Context(p=p, geometry=geometry, op=operators, compute=backend, N=p.Nphi)


def test_allocate_layout_and_memory_rejection() -> None:
    c = context()
    state = allocate_state(c)
    assert state.fluid[0].n.shape == (3 * c.op.Nc,)
    assert state.fluid[0].mom.shape == (3 * 3 * c.op.Nc,)
    assert state.fluid[0].parallel.shape == (0,)
    assert state.fluid[1].parallel.shape == (3 * c.op.Nc,)
    assert state.B.shape == (3 * len(c.op.MB),)
    assert estimated_peak_bytes(c) > 0
    c.p.memory_limit = 0
    with pytest.raises(Exception, match="positive host memory cap"):
        allocate_state(c)


def test_combine_and_retained_hermitian_checks() -> None:
    c = context()
    base = allocate_state(c)
    rhs = allocate_state(c)
    rhs.fluid[0].n.fill(0.0)
    rhs.fluid[1].n.fill(0.0)
    tag = StageTag(step=2, generation=3, trial=1, spectral_context=4, time=0.2, stage="test")
    out = combine(base, rhs, 0.5, tag, c)
    assert out.step == 2 and out.generation == 3 and out.time == 0.2
    check_retained(out, c)
    bad = allocate_state(c)
    bad.fluid[0].n[(c.p.K + 1) * c.op.Nc] = 1.0
    with pytest.raises(Exception, match="Hermitian"):
        check_retained(bad, c)


def test_mode_and_sample_apply_shapes() -> None:
    backend = Backend("cpu")
    matrices = [
        backend.array([[1.0, 2.0], [0.0, 1.0]]),
        backend.array([[2.0, 0.0], [1.0, 3.0]]),
    ]
    x = backend.array([1.0, 2.0, 3.0, 4.0])
    y = mode_apply(matrices, x, backend)
    np.testing.assert_allclose(y.reshape(2, 2), np.array([[5.0, 2.0], [6.0, 15.0]]))
    A = backend.array([[1.0, 1.0], [0.0, 2.0]])
    samples = sample_apply(A, backend.array([1.0, 2.0, 3.0, 4.0]), 2, backend)
    np.testing.assert_allclose(samples.reshape(2, 2), np.array([[3.0, 4.0], [7.0, 8.0]]))


def test_physical_inverse_accepts_hermitian_and_rejects_residue() -> None:
    c = context()
    E = 2
    hat = np.zeros((2 * c.p.K + 1, E), dtype=np.complex128)
    hat[c.p.K, :] = [1.0, 2.0]
    hat[c.p.K + 1, :] = [0.5 - 0.2j, -1.0 + 0.3j]
    hat[c.p.K - 1, :] = np.conj(hat[c.p.K + 1, :])
    samples = inverse(c.compute.array(hat).reshape(-1), E, c, physical=True, reference=1.0)
    assert np.max(np.abs(np.asarray(samples).imag)) == 0.0
    bad = hat.copy()
    bad[c.p.K + 1, 0] += 1j
    with pytest.raises(Exception, match="Hermitian|physical sample"):
        inverse(c.compute.array(bad).reshape(-1), E, c, physical=True, reference=1.0)
    recovered = forward(samples, E, c)
    np.testing.assert_allclose(recovered.reshape(hat.shape), hat, atol=3e-13, rtol=0)


def test_inverse_default_is_physical_and_rejects_imaginary_residue() -> None:
    c = context()
    E = 2
    spectrum = np.zeros((2 * c.p.K + 1, E), dtype=np.complex128)
    spectrum[c.p.K, :] = [1.0, 2.0]
    spectrum[c.p.K + 1, 0] = 0.5 - 0.2j
    with pytest.raises(Exception, match="Hermitian|physical sample|Imaginary"):
        inverse(c.compute.array(spectrum).reshape(-1), E, c)
    complex_trial = inverse(
        c.compute.array(spectrum).reshape(-1), E, c, physical=False
    )
    assert np.max(np.abs(np.asarray(complex_trial).imag)) > 0.0


def test_reproducible_times_and_invalid_midpoint() -> None:
    p = parameters()
    assert time_at_step(p, 4) == pytest.approx(4e-9)
    assert time_at_step(p, 4) < time_at_midpoint(p, 4) < time_at_step(p, 5)
    p.dt = 0.0
    with pytest.raises(Exception, match="step/dt"):
        time_at_step(p, 1)
    p.dt = 1.0
    with pytest.raises(Exception, match="step/dt"):
        time_at_step(p, -1)
    p.dt = 1.0
    p.start_time = 0.0
    with pytest.raises(Exception, match="midpoint not representable"):
        time_at_midpoint(p, 1 << 52)
