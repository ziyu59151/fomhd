"""Small real-workflow fixtures; no substitute solver or diagnostics."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from foundation_fixture import parameters
from fomhd.backend import Backend
from fomhd.core import allocate_state
from fomhd.geometry import build_rectangle
from fomhd.operators import build_operators
from fomhd.types import Context


ORACLE_ROOT = Path(os.environ.get(
    'FOMHD_NATIVE_ORACLE_DIR',
    Path(__file__).resolve().parents[2] / 'build' / 'native-oracles'))


def cumulative():
    names = ('electric_node electric_edge pressure_kinetic pressure_internal '
             'magnetic wall convection perp_rate parallel_rate spectral_kinetic '
             'spectral_perp spectral_parallel').split()
    return SimpleNamespace(
        baseline=None,
        integrals=[SimpleNamespace(**dict.fromkeys(names, 0.)) for _ in range(2)],
        boundary_integrals=SimpleNamespace(
            number=[0., 0.], kinetic=[0., 0.], internal_perp=[0., 0.],
            internal_parallel=[0., 0.], field_energy=0., total_energy=0.),
        accepted_intervals=0, last_trial=0, last_step=0, last_time=0.)


def from_oracle(device='cpu', stress=True):
    name = 'native-oracle-stress.npz' if stress else 'native-oracle-4-k1.npz'
    with np.load(ORACLE_ROOT/name, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays['metadata_json'].item()))
    p = parameters(metadata['K']); p.dt = metadata['dt']
    backend = Backend(device)
    geometry = build_rectangle(metadata['points'], metadata['points'], 1., 0., .2, .2)
    op = build_operators(geometry, p.K, backend=backend)
    context = Context(p, geometry, op, backend, p.Nphi)
    state = allocate_state(context)
    state.B = backend.array(arrays['state_B'])
    for species in range(2):
        for key in ('n', 'mom', 'perp', 'parallel'):
            setattr(state.fluid[species], key, backend.array(arrays[f'state_{species}_{key}']))
    return context, state, arrays


def assert_dimensional(actual, expected, reference, label, rtol=2e-9):
    actual = np.asarray(actual); expected = np.asarray(expected)
    assert actual.shape == expected.shape, label
    if not expected.size:
        return
    magnitude = np.max(np.abs(expected))
    limit = rtol*magnitude + 2048*np.finfo(float).eps*reference
    error = np.max(np.abs(actual-expected))
    assert error <= limit, f'{label}: error={error:.9g}, limit={limit:.9g}'
