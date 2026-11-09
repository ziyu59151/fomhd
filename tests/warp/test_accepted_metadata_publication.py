"""Accepted WR metadata publication tests."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from fomhd import io, stage
from fomhd import time as time_module
from fomhd.__main__ import _context as command_context
from fomhd.types import Failure
from test_model_routing import _context, _state
from test_time_multirate import (
    _configure,
    _empty_cumulative,
    _host_state,
    _write_multirate_case,
)


def _assert_json_primitives(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, dict):
        assert all(isinstance(key, str) for key in value)
        for child in value.values():
            _assert_json_primitives(child)
        return
    if isinstance(value, list):
        for child in value:
            _assert_json_primitives(child)
        return
    raise AssertionError(f"non-JSON metadata value: {type(value).__name__}")


def test_accepted_cuda_style_metadata_vector_publishes_and_fails_post_commit(
    tmp_path,
):
    context = _context(inertia=0, nsteps=1, hard=True)
    state = _state(context)
    _configure(context.p, nsteps=1, micro_steps=2, ion_nodes=3)
    case_path = _write_multirate_case(
        tmp_path,
        context,
        state,
        nsteps=1,
    )
    output = tmp_path / "accepted-metadata"
    loaded = io.load_case(
        case_path,
        overrides={"output": str(output)},
    )
    runtime = command_context(loaded)
    cumulative = _empty_cumulative(runtime.p.start_time)
    committed = stage.initialize(
        io.load_initial(loaded),
        runtime,
        cumulative,
        False,
    )
    run = io.Run(loaded, require_project_snapshot=False)
    try:
        accepted = time_module.advance(committed, cumulative, runtime)
        accepted_state = committed.state
        accepted_cache = committed.cache
        accepted_host = _host_state(runtime, accepted_state)
        accepted_cumulative = copy.deepcopy(io._cumulative_json(cumulative))
        metadata = accepted.coupling_metadata
        vector = np.asarray([1.25, -2.5, 3.75], dtype=np.float64)
        metadata["accepted_cuda_style_vector"] = vector

        run.time_coupling_evidence(metadata)
        manifest = json.loads(
            (run.path() / "run.json").read_text(encoding="utf-8")
        )
        published = manifest["time_coupling_evidence"]
        assert published["accepted_cuda_style_vector"] == [
            1.25,
            -2.5,
            3.75,
        ]
        _assert_json_primitives(published)
        np.testing.assert_array_equal(
            metadata["accepted_cuda_style_vector"],
            vector,
        )
        assert committed.state is accepted_state
        assert committed.cache is accepted_cache
        current_host = _host_state(runtime, committed.state)
        for name, expected in accepted_host.items():
            np.testing.assert_array_equal(current_host[name], expected)
        assert io._cumulative_json(cumulative) == accepted_cumulative
        io._check_accepted(committed)

        post_commit = Failure(
            "PostCommitMetadata",
            "injected post-commit output failure",
        )
        record = run.failure(
            post_commit,
            accepted_state.step,
            "post_commit_output",
        )
        assert record["runtime_phase"] == "post_commit_output"
        assert record["failure_class"] == "output_after_commit"
        assert record["accepted_trial"] is True
        assert record.get("rollback") is not True
        assert committed.state is accepted_state
        assert committed.cache is accepted_cache
        assert io._cumulative_json(cumulative) == accepted_cumulative
    finally:
        run.close()
