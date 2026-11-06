from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

from fomhd.backend import Backend
from fomhd.core import allocate_state
from fomhd.geometry import build_rectangle
from fomhd.operators import build_operators
from fomhd.types import (
    Budget,
    Context,
    Failure,
    FluidState,
    Parameters,
    References,
    Species,
    StageTag,
    State,
)

from foundation_fixture import parameters


SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"


def _subprocess_environment():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        item
        for item in (str(SOURCE_ROOT), environment.get("PYTHONPATH", ""))
        if item
    )
    return environment


def test_native_field_order_and_defaults() -> None:
    assert [field.name for field in dataclasses.fields(Species)] == [
        "id",
        "mass",
        "charge",
        "gamma",
        "n_min",
        "face_min",
        "p_min",
        "parallel_min",
    ]
    assert [field.name for field in dataclasses.fields(StageTag)] == [
        "step",
        "generation",
        "trial",
        "spectral_context",
        "time",
        "stage",
    ]
    assert [field.name for field in dataclasses.fields(Parameters)][-6:] == [
        "electron_inertia",
        "integrator",
        "dt_guard",
        "minimum_samples_per_cycle",
        "max_amplitude_error",
        "max_phase_error_rad",
    ]
    assert Parameters().electron_inertia == 0
    assert Parameters(electron_inertia=1).electron_inertia == 1
    assert Parameters().integrator == "rk4"
    assert Parameters().dt_guard == "error"
    assert Parameters().minimum_samples_per_cycle == 32
    assert Parameters().max_amplitude_error == 0.01
    assert Parameters().max_phase_error_rad == 0.1
    assert [field.name for field in dataclasses.fields(References)] == [
        "length",
        "time",
        "ni",
        "ne",
        "velocity",
        "electric",
        "magnetic",
        "current",
        "charge",
        "pressure",
    ]
    assert [field.name for field in dataclasses.fields(Budget)] == [
        "atol",
        "rtol",
        "reference",
        "unit",
        "norm",
        "hard",
    ]
    assert [field.name for field in dataclasses.fields(FluidState)] == [
        "n",
        "mom",
        "perp",
        "parallel",
    ]
    assert [field.name for field in dataclasses.fields(State)] == [
        "fluid",
        "B",
        "step",
        "generation",
        "time",
    ]
    assert [field.name for field in dataclasses.fields(Context)] == [
        "p",
        "geometry",
        "op",
        "compute",
        "N",
        "spectral_id",
        "next_generation",
        "next_trial",
        "next_source",
        "geometry_hash",
        "operator_hash",
        "rule_hash",
    ]
    assert FluidState() == FluidState(n=[], mom=[], perp=[], parallel=[])
    assert State().B == []


def test_parameters_fixture_matches_native_constants() -> None:
    p = parameters(K=2)
    assert p.species[0].mass == 1.67262192595e-27
    assert p.species[1].charge == -1.602176634e-19
    assert p.Nphi == 9
    assert p.ref.current > 0.0
    assert p.ref.charge == 1e-7
    assert p.ref.pressure == 1e-9
    expected_power = p.ref.pressure * p.ref.length**3 / p.ref.time
    power_budget = p.budgets["electric_pairing.ion.value"]
    assert power_budget.reference == expected_power
    assert power_budget.unit == "W"
    assert p.minimum_samples_per_cycle == 32
    assert p.max_amplitude_error == 0.01
    assert p.max_phase_error_rad == 0.1
    assert p.budgets["initial_divB.linf"].hard


def test_failure_carries_code_and_message() -> None:
    failure = Failure("Shape", "bad shape")
    assert failure.code == "Shape"
    assert str(failure) == "bad shape"
    assert failure.checks == []
    assert failure.partial_arrays == {}


def test_cpu_backend_is_explicit_and_preserves_complex_contiguous_storage() -> None:
    backend = Backend("cpu")
    assert backend.name == "cpu"
    assert backend.xp is np
    value = backend.array([[1.0 + 2.0j, 3.0 - 1.0j]])
    assert value.dtype == np.complex128
    assert value.flags.c_contiguous
    assert backend.scalar(np.float64(3.5)) == 3.5


def test_cpu_import_does_not_eagerly_load_warp_or_cupy() -> None:
    script = (
        "import sys; import fomhd.backend; "
        "assert 'warp' not in sys.modules; assert 'cupy' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        env=_subprocess_environment(),
        check=True,
    )


def test_warp_kernels_compile_and_execute_on_warp_cpu_device() -> None:
    script = r"""
import numpy as np
import warp as wp
from fomhd.backend import Backend

wp.init()
holder = object.__new__(Backend)
holder._warp = wp
kernels = Backend._make_warp_kernels(holder)
a = wp.array(np.array([[1.0, 2.0], [-1.0, 3.0]]), dtype=wp.vec2d, device="cpu")
b = wp.array(np.array([[2.0, -1.0], [4.0, 0.0]]), dtype=wp.vec2d, device="cpu")
out = wp.zeros(2, dtype=wp.vec2d, device="cpu")
wp.launch(kernels.axpy, dim=2, inputs=[a, b, 0.5], outputs=[out], device="cpu")
np.testing.assert_allclose(out.numpy(), [[2.0, 1.5], [1.0, 3.0]])
source = wp.array(np.array([[1.0, 0.0], [2.0, 3e-12]]), dtype=wp.vec2d, device="cpu")
physical = wp.zeros(2, dtype=wp.vec2d, device="cpu")
bad = wp.array(np.array([2147483647], dtype=np.int32), device="cpu")
wp.launch(
    kernels.physical,
    dim=2,
    inputs=[source, 1e-11],
    outputs=[physical, bad],
    device="cpu",
)
np.testing.assert_allclose(physical.numpy(), [[1.0, 0.0], [2.0, 0.0]])
assert bad.numpy().tolist() == [2147483647]

def rejected(values, limit, expected):
    source = wp.array(np.asarray(values, dtype=np.float64), dtype=wp.vec2d, device="cpu")
    physical = wp.zeros(source.shape[0], dtype=wp.vec2d, device="cpu")
    bad = wp.array(np.array([2147483647], dtype=np.int32), device="cpu")
    wp.launch(
        kernels.physical,
        dim=source.shape[0],
        inputs=[source, limit],
        outputs=[physical, bad],
        device="cpu",
    )
    assert bad.numpy().tolist() == [expected]

rejected([[1.0, 0.0], [np.nan, 0.0], [3.0, 0.0]], 1e-11, 1)
rejected([[1.0, 0.0], [2.0, 0.0], [np.inf, 0.0]], 1e-11, 2)
rejected([[1.0, 0.0], [2.0, 2e-3], [3.0, 0.0]], 1e-11, 1)
rejected([[1.0, 0.0], [2.0, 2e-3], [3.0, 0.0], [4.0, -3e-3]], 1e-11, 1)
"""
    with tempfile.TemporaryDirectory(prefix="a0-warp-cache-") as cache:
        environment = _subprocess_environment()
        environment["WARP_CACHE_PATH"] = cache
        completed = subprocess.run(
            [sys.executable, "-c", script],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_physical_bad_buffer_starts_at_int32_max() -> None:
    backend = Backend("cpu")
    buffer = backend._physical_bad_buffer()
    assert buffer.shape == (1,)
    assert buffer.dtype == np.int32
    assert int(buffer[0]) == np.iinfo(np.int32).max


class _DeviceArray:
    def __init__(self, values, dtype, device_id: int, contiguous: bool) -> None:
        self._values = np.asarray(values, dtype=dtype)
        if not contiguous:
            self._values = self._values[:, ::2]
        self.dtype = self._values.dtype
        self.flags = self._values.flags
        self.size = self._values.size
        self.device = type("Device", (), {"id": device_id})()

    def view(self, dtype):
        return self._values.view(dtype)

    def __getitem__(self, item):
        return self._values[item]


def _device_array(values, dtype, device_id: int = 0, contiguous: bool = True):
    return _DeviceArray(values, dtype, device_id, contiguous)


def test_cuda_kernel_storage_validation_and_float64_view() -> None:
    holder = object.__new__(Backend)
    holder._cuda = True
    holder.xp = type(
        "Xp",
        (),
        {"ndarray": _DeviceArray, "complex128": np.complex128, "float64": np.float64},
    )
    holder._wp_device = type("Device", (), {"ordinal": 0})()
    values = _device_array([[1 + 2j, 3 + 4j]], np.complex128)
    Backend._validate_kernel_storage(holder, values)
    storage = Backend._kernel_storage_view(holder, values)
    assert storage.dtype == np.float64
    assert storage.shape == (2, 2)
    storage[0, 1] = 7.0
    assert values[0, 0].imag == 7.0

    with pytest.raises(Exception, match="complex128"):
        Backend._validate_kernel_storage(
            holder, _device_array([[1.0, 2.0]], np.float64)
        )
    with pytest.raises(Exception, match="contiguous"):
        Backend._validate_kernel_storage(
            holder,
            _device_array(
                [[1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j]],
                np.complex128,
                contiguous=False,
            ),
        )
    with pytest.raises(Exception, match="device"):
        Backend._validate_kernel_storage(
            holder, _device_array([[1 + 1j]], np.complex128, device_id=1)
        )


def test_axpy_and_analytic_fft() -> None:
    backend = Backend("cpu")
    a = backend.array([1 + 2j, -1j, 2 + 0j])
    b = backend.array([3 - 1j, 4j, -2 + 1j])
    actual = backend.axpy(a, b, 0.25)
    np.testing.assert_allclose(actual, a + 0.25 * b, rtol=0, atol=0)

    K, N, E = 2, 9, 3
    modes = (-2, 1, -1)
    cycles = np.array([1.3 - 0.2j, -0.7 + 0.4j, 0.2 + 0.9j])
    samples = np.zeros((N, E), dtype=np.complex128)
    for sample in range(N):
        phi = 2.0 * np.pi * sample / N
        samples[sample] = cycles * np.exp(1j * np.array(modes) * phi)
    flattened = backend.array(samples.reshape(-1))
    hat = backend.forward_fft(flattened, E, K, N)
    expected = np.zeros((2 * K + 1, E), dtype=np.complex128)
    for entity, mode in enumerate(modes):
        expected[mode + K, entity] = cycles[entity]
    np.testing.assert_allclose(hat.reshape(2 * K + 1, E), expected, atol=3e-14)
    recovered = backend.inverse_fft(hat, E, K, N)
    np.testing.assert_allclose(
        recovered.reshape(N, E), samples, atol=3e-13, rtol=0
    )


def test_derivative_is_antiadjoint() -> None:
    backend = Backend("cpu")
    K, N, E = 1, 9, 2
    rng = np.random.default_rng(4)
    x = backend.array(rng.normal(size=N * E) + 1j * rng.normal(size=N * E))
    y = backend.array(rng.normal(size=N * E) + 1j * rng.normal(size=N * E))
    dx = backend.derivative(x, E, N)
    dy = backend.derivative(y, E, N)
    pairing = np.vdot(y, dx) + np.vdot(dy, x)
    assert abs(pairing) < 2e-11


def test_scaled_solve_reuses_one_factor_and_checks_rank() -> None:
    backend = Backend("cpu")
    A = backend.array(
        [
            [4 + 1j, 2 - 1j, 2j],
            [-1 + 0.5j, 3 + 2j, 1 - 1j],
            [2 - 2j, 1j, 5],
        ]
    )
    expected = np.array([1 - 2j, -0.5 + 0.25j, 2 + 1j])
    rhs = backend.array(A @ expected)
    result = backend.linear(
        A,
        rhs,
        backend.array([1.0, 2.0, 3.0]),
        backend.array([2.0, 0.5, 1.5]),
        1e-12,
        1e-12,
    )
    np.testing.assert_allclose(result.x, expected, atol=2e-13, rtol=0)
    assert result.refinement_steps <= 5
    assert result.backward_error <= 1e-12
    np.testing.assert_allclose(
        result.residual, A @ result.x - rhs, atol=1e-13, rtol=0
    )
    assert result.singular_values.shape == (3,)
    assert result.singular_values.dtype == np.float64


def test_singular_and_invalid_solves_fail() -> None:
    backend = Backend("cpu")
    singular = backend.array([[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(Exception, match="singular|rank"):
        backend.linear(
            singular,
            backend.array([1.0, 2.0]),
            backend.array([1.0, 1.0]),
            backend.array([1.0, 1.0]),
            1e-10,
            1e-10,
        )
    with pytest.raises(Exception, match="row scale"):
        backend.linear(
            backend.array([[2.0]]),
            backend.array([1.0]),
            backend.array([0.0]),
            backend.array([1.0]),
            1e-10,
            1e-10,
        )


def test_independent_rows_are_deterministic() -> None:
    backend = Backend("cpu")
    matrix = backend.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [0.0, 0.0],
            [0.0, 3.0],
        ]
    )
    assert backend.independent_rows(matrix, 1e-12) == [0, 3]


def test_singular_lu_factorization_is_rejected_without_warning_mode() -> None:
    backend = Backend("cpu")
    singular = backend.array([[1.0, 2.0], [2.0, 4.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with pytest.raises(Exception, match="singular"):
            backend.factor(singular)


def test_zero_rhs_backward_error_has_zero_row_semantics_without_warning() -> None:
    backend = Backend("cpu")
    matrix = backend.array([[2.0, 0.0], [0.0, 3.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = backend.linear(
            matrix,
            backend.array([0.0, 0.0]),
            backend.real_array([1.0, 1.0]),
            backend.real_array([1.0, 1.0]),
            1e-12,
            1e-14,
        )
    assert result.backward_error == 0.0
    np.testing.assert_array_equal(result.x, np.zeros(2))


class _NoWhereXp:
    ndarray = np.ndarray
    complex128 = np.complex128
    float64 = np.float64

    def __init__(self) -> None:
        self.divide_calls = 0

    def __getattr__(self, name):
        return getattr(np, name)

    def divide(self, left, right, *args, **kwargs):
        self.divide_calls += 1
        if "where" in kwargs:
            raise TypeError("CuPy divide does not support where")
        return np.divide(left, right, *args, **kwargs)


def test_backward_error_uses_portable_safe_denominator() -> None:
    backend = Backend("cpu")
    wrapper = _NoWhereXp()
    backend.xp = wrapper
    matrix = backend.array([[2.0, 0.0], [0.0, 3.0]])
    result = backend.linear(
        matrix,
        backend.array([0.0, 0.0]),
        backend.real_array([1.0, 1.0]),
        backend.real_array([1.0, 1.0]),
        1e-12,
        1e-14,
    )
    assert result.backward_error == 0.0
    assert wrapper.divide_calls == 0


class _StagnatingBackend(Backend):
    def __init__(self) -> None:
        super().__init__("cpu")
        self.in_linear = False
        self.solve_count = 0

    def linear(self, *args, **kwargs):
        self.in_linear = True
        try:
            return super().linear(*args, **kwargs)
        finally:
            self.in_linear = False

    def solve(self, factorization, rhs):
        result = super().solve(factorization, rhs)
        if self.in_linear:
            self.solve_count += 1
            if self.solve_count == 1:
                return self.array(result + 0.01)
            return self.array(self.xp.zeros_like(rhs))
        return result


def test_numpy_boolean_stagnation_is_reported_as_failure() -> None:
    backend = _StagnatingBackend()
    matrix = backend.array([[2.0, 0.0], [0.0, 3.0]])
    with pytest.raises(Exception, match="stagnated"):
        backend.linear(
            matrix,
            backend.array([1.0, 1.0]),
            backend.real_array([1.0, 1.0]),
            backend.real_array([1.0, 1.0]),
            1e-12,
            1e-14,
        )


def test_spectrum_checks_do_not_transfer_spectrum_to_host() -> None:
    backend = Backend("cpu")
    backend.to_host = lambda value: (_ for _ in ()).throw(
        AssertionError("unexpected spectrum host transfer")
    )
    matrix = backend.array([[2.0, 0.0], [0.0, 3.0]])
    singular = backend.singular_values(matrix)
    assert singular.shape == (2,)
    result = backend.linear(
        matrix,
        backend.array([2.0, 3.0]),
        backend.real_array([1.0, 1.0]),
        backend.real_array([1.0, 1.0]),
        1e-12,
        1e-12,
    )
    assert result.refinement_steps == 0
    assert backend.independent_rows(matrix, 1e-12) == [0, 1]


class _CudaBudgetView:
    name = "cuda"

    def __init__(self, cpu_backend: Backend) -> None:
        self.xp = np
        self._cpu_backend = cpu_backend

    def array(self, value, dtype=None):
        return self._cpu_backend.array(value, dtype)


def test_gpu_like_allocation_consumes_device_memory_limit() -> None:
    backend = Backend("cpu")
    p = parameters(K=1)
    geometry = build_rectangle(4, 3, 0.5, -0.2, 0.1, 0.1)
    operators = build_operators(geometry, 1, backend=backend)
    p.device_memory_limit = 1
    c = Context(
        p=p,
        geometry=geometry,
        op=operators,
        compute=_CudaBudgetView(backend),
        N=p.Nphi,
    )
    with pytest.raises(Exception, match="device.*memory budget exceeded"):
        allocate_state(c)
