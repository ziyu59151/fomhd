"""Fixed-grid distance diagnostics reuse physical wall geometry."""

from types import SimpleNamespace

import numpy as np

from fomhd import diagnostics
from fomhd.geometry import build_aligned_box


def test_wall_distances_reuse_the_same_physical_segments(monkeypatch):
    r = 1.0 + np.arange(5) * 0.2
    z = np.arange(4) * 0.25
    marker = [(1.2, 0.2), (1.6, 0.2), (1.4, 0.6)]
    geometry = build_aligned_box(r, z, marker)
    context = SimpleNamespace(geometry=geometry)
    calls = 0
    point_distance = diagnostics._point_segment_distance

    def counted(*args):
        nonlocal calls
        calls += 1
        return point_distance(*args)

    monkeypatch.setattr(diagnostics, "_point_segment_distance", counted)
    first = diagnostics._entity_wall_distances(geometry.vertices, context)
    expected_calls = len(geometry.vertices.keys) * len(geometry.walls)
    assert calls == expected_calls
    np.testing.assert_array_equal(
        diagnostics._entity_wall_distances(geometry.vertices, context), first
    )
    assert calls == expected_calls
    context.geometry = build_aligned_box(r, z + 0.05, marker)
    diagnostics._entity_wall_distances(context.geometry.vertices, context)
    assert calls == 2 * expected_calls
