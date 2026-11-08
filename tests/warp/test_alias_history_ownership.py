"""Independent regressions for retained-history ownership in alias views."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(
    os.environ.get(
        "FOMHD_ALIAS_SOURCE_ROOT",
        str(PROJECT_ROOT),
    )
)
for path in (
    SOURCE_ROOT / "src",
    SOURCE_ROOT,
    SOURCE_ROOT / "tests" / "warp",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fomhd


_PACKAGE_OVERRIDE = PROJECT_ROOT / "src" / "fomhd"
if _PACKAGE_OVERRIDE.is_dir():
    override = str(_PACKAGE_OVERRIDE)
    if override not in fomhd.__path__:
        fomhd.__path__.insert(0, override)

from fomhd import core, diagnostics, massless
from fomhd.types import Failure, StageTag


def test_imports_selected_checkout_files():
    assert Path(core.__file__).resolve() == (
        PROJECT_ROOT / "src" / "fomhd" / "core.py"
    )
    assert Path(massless.__file__).resolve() == (
        PROJECT_ROOT / "src" / "fomhd" / "massless.py"
    )


class _ShapeOnlyArray:
    def __init__(self, shape, itemsize=16):
        self.shape = tuple(shape)
        self.itemsize = itemsize
        self.size = int(np.prod(self.shape, dtype=np.int64))
        self.nbytes = self.size * self.itemsize


def _east32_shape_operators():
    modes = 3
    nc, nr, nz, nv, ne, nb, walls = 796, 828, 828, 861, 2452, 2517, 128
    complex_matrix = lambda rows, columns: _ShapeOnlyArray((rows, columns))
    real_vector = lambda size: _ShapeOnlyArray((size,), itemsize=8)
    integer_vector = lambda size: _ShapeOnlyArray((size,), itemsize=4)
    per_mode = {
        "G": complex_matrix(ne, nc),
        "C": complex_matrix(nb, ne),
        "DB": complex_matrix(nv, nb),
        "DJ": complex_matrix(nc, ne),
        "Cdag": complex_matrix(ne, nb),
    }
    operators = SimpleNamespace(
        K=1,
        Nc=nc,
        Ne=ne,
        Nb=nb,
        **{
            name: [value] * modes
            for name, value in per_mode.items()
        },
        I=complex_matrix(ne, 3 * nc),
        H=complex_matrix(3 * nc, ne),
        QB=complex_matrix(3 * nc, nb),
        T_full=complex_matrix(walls, 3 * nc),
        T=complex_matrix(walls, 3 * nc),
        DR=complex_matrix(nc, nr),
        DZ=complex_matrix(nc, nz),
        topology_flux_row=complex_matrix(1, nb),
        PR_r=complex_matrix(nr, nr),
        PZ_r=complex_matrix(nr, nr),
        PR_z=complex_matrix(nz, nz),
        PZ_z=complex_matrix(nz, nz),
        PR_v=complex_matrix(nv, nv),
        PZ_v=complex_matrix(nv, nv),
        b1r=complex_matrix(nr, nv),
        b1z=complex_matrix(nz, nv),
        b2r=complex_matrix(nc, nr),
        b2z=complex_matrix(nc, nz),
        M0=real_vector(nc),
        M1=real_vector(ne),
        MB=real_vector(nb),
        Rbar_r=real_vector(nr),
        Rbar_v=real_vector(nv),
        Lphi=[],
        phi_factor=[],
        poisson_spectra=[],
        Bc_full=complex_matrix(0, modes * ne),
        Bc=complex_matrix(0, modes * ne),
        finite_mass_available=False,
    )
    for name in ("PR_r", "PZ_r", "PR_z", "PZ_z", "PR_v", "PZ_v"):
        matrix = getattr(operators, name)
        setattr(
            operators,
            f"{name}_factor",
            SimpleNamespace(
                original=matrix,
                factors=complex_matrix(*matrix.shape),
                pivots=integer_vector(matrix.shape[0]),
            ),
        )
    return operators


def _east32_context():
    parameters = SimpleNamespace(
        K=1,
        electron_inertia=0,
        coupling="waveform",
        micro_steps=4,
        waveform_rtol=1.0e-10,
        waveform_max_iterations=12,
        memory_limit=8 << 30,
        device_memory_limit=8 << 30,
    )
    return SimpleNamespace(
        p=parameters,
        op=_east32_shape_operators(),
        geometry=SimpleNamespace(walls=[None] * 128),
        compute=SimpleNamespace(name="cpu"),
        N=5,
        spectral_id=0,
        next_source=1,
    )


def test_alias_history_keeps_outer_samples_and_charges_only_alias_workspace():
    base = _east32_context()
    assert core.predictor_history_entries(base) == 179_995_368
    assert core.predictor_wall_history_entries(base) == 27_412_480
    assert core.estimated_peak_bytes(base) == 5_081_734_872

    alias = copy.copy(base)
    alias.N = 2 * base.N + 1
    alias.spectral_id = base.spectral_id + 1
    alias._history_samples = base.N

    assert core.predictor_history_entries(alias) == 179_995_368
    assert core.predictor_wall_history_entries(alias) == 27_412_480
    assert core.estimated_peak_bytes(alias) == 5_262_540_504
    assert core.estimated_peak_bytes(alias) - core.estimated_peak_bytes(base) == (
        180_805_632
    )
    assert core.estimated_peak_bytes(alias) < 8 << 30
    assert core.preflight_memory(alias) == 5_262_540_504


@pytest.mark.parametrize(
    "counter",
    [
        core.predictor_history_entries,
        core.predictor_wall_history_entries,
    ],
)
@pytest.mark.parametrize("invalid", [True, False, -1, 0, 6])
def test_history_sample_identity_rejects_invalid_values(counter, invalid):
    context = _east32_context()
    context._history_samples = invalid

    with pytest.raises(Failure, match="history sample"):
        counter(context)


def _tiny_stage_rhs():
    return SimpleNamespace(
        fluid=(
            SimpleNamespace(
                n=np.zeros(1),
                mom=np.zeros(3),
                perp=np.zeros(1),
            ),
            SimpleNamespace(
                n=np.zeros(1),
                mom=np.zeros(3),
                perp=np.zeros(1),
                parallel=np.zeros(1),
            ),
        ),
        B=np.zeros(2),
    )


def test_nested_alias_probe_propagates_original_history_samples(monkeypatch):
    context = SimpleNamespace(
        N=5,
        spectral_id=0,
        next_source=1,
        compute=SimpleNamespace(xp=np),
        p=SimpleNamespace(
            K=0,
            species=(
                SimpleNamespace(id="i"),
                SimpleNamespace(id="e"),
            ),
        ),
        op=SimpleNamespace(
            Nb=2,
            MB=np.ones(2),
            M0=np.ones(1),
        ),
        geometry=SimpleNamespace(
            z_faces=SimpleNamespace(keys=[0]),
            vertices=SimpleNamespace(keys=[0]),
        ),
    )
    tag = StageTag(
        step=2,
        generation=3,
        trial=4,
        spectral_context=0,
        time=0.5,
        stage="waveform_m001_r2",
        micro_step=1,
        iteration=0,
        slab=0,
    )
    result = SimpleNamespace(checks=[], rhs=_tiny_stage_rhs())
    captured = []

    def fake_evaluate(_state, alternate_tag, alternate_context, alias=False):
        captured.append((alternate_tag, alternate_context))
        return SimpleNamespace(checks=[], rhs=_tiny_stage_rhs())

    monkeypatch.setattr(massless, "evaluate_stage", fake_evaluate)
    monkeypatch.setattr(diagnostics, "measure", lambda *args, **kwargs: None)
    monkeypatch.setattr(diagnostics, "gate", lambda *args, **kwargs: None)

    massless._alias_probe(None, tag, context, result)
    first_tag, first_context = captured[-1]
    assert first_context.N == 11
    assert first_context._history_samples == 5
    assert first_tag.micro_step == tag.micro_step

    massless._alias_probe(None, tag, first_context, result)
    _, nested_context = captured[-1]
    assert nested_context.N == 23
    assert nested_context._history_samples == 5
