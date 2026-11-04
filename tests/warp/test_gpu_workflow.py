"""Native-oracle full-stage/transaction parity inside an owned Slurm GPU job."""
import os

import numpy as np
import pytest

from fomhd.stage import initialize
from fomhd.time import advance
from workflow_fixture import assert_dimensional, cumulative, from_oracle

@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('stress', [True, False])
def test_gpu_stage_and_midpoint_against_native(device, stress, monkeypatch):
    if device.startswith('cuda'):
        if os.environ.get('FOMHD_TEST_GPU') != '1':
            pytest.skip('owned Slurm GPU run required')
        assert os.environ.get('SLURM_JOB_ID')
    context, state, native = from_oracle(device, stress)
    backend = context.compute
    # Explicit observations below own host copies. Stages must not call this API.
    to_host = backend.to_host
    monkeypatch.setattr(backend, 'to_host', lambda *args, **kwargs: pytest.fail(
        'whole-array host transfer inside a stage'))
    ledger = cumulative(); captured = {}
    committed = initialize(state, context, ledger, False, captured)
    for key in ('E', 'EL', 'ET'):
        assert_dimensional(to_host(getattr(committed.cache, key)), native[key],
                           context.p.ref.electric, key)
    for species in range(2):
        for key in ('n', 'mom', 'perp', 'parallel'):
            expected = native[f'rhs_{species}_{key}']
            ref = (context.p.ref.ni if key == 'n' else
                   context.p.species[species].mass*context.p.ref.ni*context.p.ref.velocity
                   if key == 'mom' else context.p.ref.pressure)/context.p.ref.time
            if key == 'mom':
                # A pressure/wall cancellation is scaled by its force terms.
                ref = max(ref, context.p.ref.pressure/context.p.ref.length)
            actual = getattr(captured['rhs'].fluid[species], key)
            assert_dimensional(to_host(actual), expected, ref, f'rhs_{species}_{key}')
    result = advance(committed, ledger, context)
    assert committed.state.step == 1 and ledger.accepted_intervals == 1
    assert result.accepted and result.interval.accepted
    assert all(not x.hard or (x.available and x.passed) for x in result.checks)
    for species in range(2):
        for key in ('n', 'mom', 'perp', 'parallel'):
            actual = getattr(committed.state.fluid[species], key)
            ref = (context.p.ref.ni if key == 'n' else
                   context.p.species[species].mass*context.p.ref.ni*context.p.ref.velocity
                   if key == 'mom' else context.p.ref.pressure)
            assert_dimensional(to_host(actual), native[f'endpoint_{species}_{key}'],
                               ref, f'endpoint_{species}_{key}')
    assert_dimensional(to_host(committed.state.B), native['endpoint_B'],
                       context.p.ref.magnetic, 'endpoint_B')
    assert all(isinstance(f.n, backend.xp.ndarray) for f in committed.state.fluid)
