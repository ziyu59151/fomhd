"""Strict initial-state host-to-backend assignment contracts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fomhd.backend import Backend


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools" / "warp_clt_boundary.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "warp_clt_boundary_initialization", RUNNER
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StrictDeviceArray(np.ndarray):
    """NumPy storage that rejects unexported host arrays on item assignment."""

    def __array_finalize__(self, source):
        del source

    def __setitem__(self, key, value):
        if not (
            np.isscalar(value)
            or isinstance(value, (StrictDeviceArray, np.generic))
        ):
            raise ValueError("non-scalar host array cannot be assigned")
        super().__setitem__(key, value)


class StrictBackend:
    name = "cpu"
    xp = np
    complex128 = np.complex128
    float64 = np.float64

    def __init__(self):
        self.calls = []

    def array(self, value, dtype=None):
        target = np.complex128 if dtype is None else dtype
        self.calls.append((np.asarray(value).shape, target))
        return np.asarray(value, dtype=target).view(StrictDeviceArray)

    def real_array(self, value):
        return self.array(value, np.float64)

    def to_host(self, value):
        if not isinstance(value, StrictDeviceArray):
            raise AssertionError("to_host received an unowned array")
        return np.asarray(value)

    def mv(self, matrix, vector):
        return self.array(np.asarray(matrix) @ np.asarray(vector))


def _strict_array(*shape):
    return np.zeros(shape, dtype=np.complex128).view(StrictDeviceArray)


def _state(nc, nb, K=0):
    modes = 2 * K + 1
    return SimpleNamespace(
        fluid=(
            SimpleNamespace(
                n=_strict_array(modes * nc),
                mom=_strict_array(modes * 3 * nc),
                perp=_strict_array(modes * nc),
                parallel=_strict_array(0),
            ),
            SimpleNamespace(
                n=_strict_array(modes * nc),
                mom=_strict_array(modes * 3 * nc),
                perp=_strict_array(modes * nc),
                parallel=_strict_array(modes * nc),
            ),
        ),
        B=_strict_array(modes * nb),
        step=0,
        generation=0,
        time=0.0,
    )


def test_fill_fluid_control_uploads_each_host_field_once():
    runner = _load_runner()
    backend = StrictBackend()
    context = SimpleNamespace(
        op=SimpleNamespace(Nc=2),
        p=SimpleNamespace(K=1),
        compute=backend,
    )
    state = _state(2, 2, K=1)
    density = np.asarray([1.0, 2.0])
    p_i = np.asarray([3.0, 4.0])
    p_e_perp = np.asarray([5.0, 6.0])
    p_e_parallel = np.asarray([7.0, 8.0])

    runner._fill_fluid_control(
        state, context, density, p_i, p_e_perp, p_e_parallel
    )

    assert backend.calls == [
        ((2,), np.complex128),
        ((2,), np.complex128),
        ((2,), np.complex128),
        ((2,), np.complex128),
    ]
    np.testing.assert_array_equal(
        state.fluid[0].n.reshape(3, 2)[1], density
    )
    np.testing.assert_array_equal(
        state.fluid[1].perp.reshape(3, 2)[1], p_e_perp
    )
    np.testing.assert_array_equal(
        state.fluid[1].parallel.reshape(3, 2)[1], p_e_parallel
    )
def test_fill_fluid_control_cpu_values_are_unchanged():
    runner = _load_runner()
    backend = Backend("cpu")
    context = SimpleNamespace(
        op=SimpleNamespace(Nc=2),
        p=SimpleNamespace(K=1),
        compute=backend,
    )
    state = SimpleNamespace(
        fluid=(
            SimpleNamespace(
                n=backend.array(np.zeros(6)),
                perp=backend.array(np.zeros(6)),
                parallel=backend.array(np.zeros(0)),
            ),
            SimpleNamespace(
                n=backend.array(np.zeros(6)),
                perp=backend.array(np.zeros(6)),
                parallel=backend.array(np.zeros(6)),
            ),
        )
    )

    runner._fill_fluid_control(
        state,
        context,
        np.asarray([1.0, 2.0]),
        np.asarray([3.0, 4.0]),
        np.asarray([5.0, 6.0]),
        np.asarray([7.0, 8.0]),
    )

    np.testing.assert_array_equal(
        state.fluid[0].n.reshape(3, 2)[1], [1.0, 2.0]
    )
    np.testing.assert_array_equal(
        state.fluid[0].perp.reshape(3, 2)[1], [3.0, 4.0]
    )
    np.testing.assert_array_equal(
        state.fluid[1].perp.reshape(3, 2)[1], [5.0, 6.0]
    )
    np.testing.assert_array_equal(
        state.fluid[1].parallel.reshape(3, 2)[1], [7.0, 8.0]
    )


def _vacuum_context(mesh):
    backend = StrictBackend()
    plan = SimpleNamespace(major_radius=3.0, minor_radius=0.5)
    if mesh == "polygon":
        geometry = SimpleNamespace(
            kind="polygon_dual_v1",
            vertices=SimpleNamespace(R=np.asarray([2.0, 3.0, 4.0])),
            faces=SimpleNamespace(R=np.asarray([3.0])),
            Rbar_v=np.asarray([2.0, 3.0, 4.0]),
        )
        nb = 4
    else:
        geometry = SimpleNamespace(
            kind="staircase_v1",
            vertices=SimpleNamespace(R=np.asarray([2.0, 3.0, 4.0])),
            z_faces=SimpleNamespace(R=np.asarray([3.0])),
            r_faces=SimpleNamespace(R=np.asarray([2.5, 3.5])),
            Rbar_v=np.asarray([2.0, 3.0, 4.0]),
        )
        nb = 4
    context = SimpleNamespace(
        clt_plan=plan,
        op=SimpleNamespace(Nc=2, Nb=nb),
        p=SimpleNamespace(K=0),
        geometry=geometry,
        compute=backend,
    )
    return context, _state(2, nb)


def test_vacuum_initial_magnetic_assignments_are_backend_owned(
    monkeypatch,
):
    runner = _load_runner()
    for mesh in ("polygon", "staircase"):
        context, state = _vacuum_context(mesh)
        monkeypatch.setattr(
            runner.core, "allocate_state", lambda current: state
        )

        result, _metadata = runner._vacuum_state(context)

        assert isinstance(result.B, StrictDeviceArray)
        assert result.B.dtype == np.complex128
        assert len(context.compute.calls) == 5
        assert np.count_nonzero(np.asarray(result.B)) > 0


def _fake_gs_model():
    return SimpleNamespace(
        R0=3.0,
        a=0.5,
        F0=6.0,
        n0=2.0e19,
        p_edge=1000.0,
        pprime=1.0e4,
        alpha=0.6,
        degree=10,
        solve_residual=0.0,
    )


def _sample_gs(model, radius, axial):
    del model, axial
    radius = np.asarray(radius)
    if radius.size == 3:
        return {"B_phi": 6.0 / radius}
    if radius.size == 1:
        return {
            "B_R": np.asarray([0.25]),
            "B_Z": np.asarray([-0.5]),
        }
    return {
        "n": np.asarray([1.0, 2.0]),
        "p_ion": np.asarray([3.0, 4.0]),
        "p_e_perp": np.asarray([5.0, 6.0]),
        "p_e_parallel": np.asarray([7.0, 8.0]),
        "psi": np.asarray([0.1, 0.2]),
        "B_R": np.asarray([0.3, 0.4]),
        "B_Z": np.asarray([0.5, 0.6]),
        "B_phi": np.asarray([1.0, 2.0]),
    }


def _gs_context(mesh):
    backend = StrictBackend()
    plan = SimpleNamespace(major_radius=3.0, minor_radius=0.5)
    if mesh == "polygon":
        geometry = SimpleNamespace(
            kind="polygon_dual_v1",
            cells=SimpleNamespace(
                R=np.asarray([2.8, 3.2]),
                Z=np.asarray([0.0, 0.1]),
            ),
            vertices=SimpleNamespace(
                R=np.asarray([2.8, 3.0, 3.2]),
                Z=np.asarray([0.0, 0.0, 0.0]),
            ),
            faces=SimpleNamespace(
                R=np.asarray([3.0]),
                Z=np.asarray([0.0]),
            ),
            face_normals=np.asarray([[1.0, 0.0]]),
            face_tangents=np.asarray([[0.0, 1.0]]),
            Rbar_v=np.asarray([2.0, 3.0, 4.0]),
        )
        op = SimpleNamespace(
            Nc=2,
            Nb=4,
            G=[
                np.asarray([[0.2, -0.1]]).view(StrictDeviceArray)
            ],
        )
    else:
        geometry = SimpleNamespace(
            kind="staircase_v1",
            cells=SimpleNamespace(
                R=np.asarray([2.8, 3.2]),
                Z=np.asarray([0.0, 0.1]),
            ),
            vertices=SimpleNamespace(
                R=np.asarray([2.8, 3.0, 3.2]),
                Z=np.asarray([0.0, 0.0, 0.0]),
            ),
            z_faces=SimpleNamespace(
                R=np.asarray([3.0]),
                Z=np.asarray([0.0]),
            ),
            r_faces=SimpleNamespace(
                R=np.asarray([2.5, 3.5]),
                Z=np.asarray([0.0, 0.0]),
            ),
            Rbar_v=np.asarray([2.0, 3.0, 4.0]),
        )
        op = SimpleNamespace(
            Nc=2,
            Nb=4,
            Ne=5,
            C=[np.ones((4, 5))],
            topology_flux_row=np.ones((1, 4)).view(StrictDeviceArray),
            MB=np.ones(4).view(StrictDeviceArray),
        )
    context = SimpleNamespace(
        clt_plan=plan,
        op=op,
        p=SimpleNamespace(K=0),
        geometry=geometry,
        compute=backend,
    )
    return context, _state(2, 4)


def test_gs_initial_magnetic_assignments_are_backend_owned(monkeypatch):
    runner = _load_runner()
    monkeypatch.setitem(
        sys.modules,
        "tools.circular_equilibrium",
        SimpleNamespace(
            build_gs_equilibrium=lambda **values: _fake_gs_model(),
            sample_gs=_sample_gs,
        ),
    )
    for mesh in ("polygon", "staircase"):
        context, state = _gs_context(mesh)
        monkeypatch.setattr(
            runner.core, "allocate_state", lambda current: state
        )

        result, metadata = runner._gs_state(context)

        assert metadata["baseline"] == "gs"
        assert isinstance(result.B, StrictDeviceArray)
        assert result.B.dtype == np.complex128
        assert np.count_nonzero(np.asarray(result.B)) > 0
        assert len(context.compute.calls) >= 6
