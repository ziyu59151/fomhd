"""Durable publication tests for rejected complete WR macros."""

from __future__ import annotations

import copy
import json
import sys

import numpy as np
import pytest

import fomhd
from fomhd import io, stage
from fomhd import time as time_module
from fomhd.__main__ import _context as command_context
from fomhd.types import Failure
from test_time_multirate import (
    _configure,
    _empty_cumulative,
    _write_multirate_case,
)
from test_model_routing import _context, _state


OBJECT_PARTIAL_KEYS = {
    "fine_identities",
    "fine_records",
    "multirate_all_coarse_records",
    "multirate_all_fine_records",
}


@pytest.fixture(autouse=True)
def _restore_multirate_module():
    name = "fomhd.time_multirate"
    missing = object()
    previous = sys.modules.pop(name, missing)
    previous_attribute = getattr(fomhd, "time_multirate", missing)
    if previous_attribute is not missing:
        delattr(fomhd, "time_multirate")
    try:
        yield
    finally:
        sys.modules.pop(name, None)
        if hasattr(fomhd, "time_multirate"):
            delattr(fomhd, "time_multirate")
        if previous is not missing:
            sys.modules[name] = previous
        if previous_attribute is not missing:
            fomhd.time_multirate = previous_attribute


def _host_partial_arrays(failure):
    result = {}
    omitted = []
    for key, value in failure.partial_arrays.items():
        try:
            array = io._host_array(value)
        except Exception as error:
            omitted.append((key, type(error).__name__))
            continue
        if array.dtype.hasobject or array.dtype.kind not in "fcbiuU":
            omitted.append((key, str(array.dtype)))
            continue
        result[key] = np.array(array, copy=True)
    return result, omitted


def test_rejected_wr_failure_publishes_safe_archive_and_metadata(
    tmp_path,
):
    input_context = _context(inertia=0, nsteps=1, hard=True)
    input_state = _state(input_context)
    _configure(
        input_context.p,
        nsteps=1,
        micro_steps=2,
        ion_nodes=3,
    )
    input_context.p.waveform_rtol = 1.0e-30
    input_context.p.waveform_max_iterations = 1
    case_path = _write_multirate_case(
        tmp_path,
        input_context,
        input_state,
        nsteps=1,
        ion_nodes=3,
    )
    output = tmp_path / "rejected-publication"
    loaded = io.load_case(
        case_path,
        overrides={"output": str(output)},
    )
    context = command_context(loaded)
    initial = io.load_initial(loaded)
    cumulative = _empty_cumulative(context.p.start_time)
    committed = stage.initialize(
        initial,
        context,
        cumulative,
        False,
    )
    old_state = committed.state
    old_cache = committed.cache
    old_cumulative = copy.deepcopy(io._cumulative_json(cumulative))
    run = io.Run(loaded, require_project_snapshot=False)
    try:
        with pytest.raises(Failure) as caught:
            time_module.advance(committed, cumulative, context)
        failure = caught.value
        assert failure.code == "MultirateConvergence"
        assert committed.state is old_state
        assert committed.cache is old_cache
        assert io._cumulative_json(cumulative) == old_cumulative
        io._check_accepted(committed)

        partial_reference = failure.partial_arrays
        object_reference = {
            key: failure.partial_arrays[key]
            for key in OBJECT_PARTIAL_KEYS
            if key in failure.partial_arrays
        }
        assert set(object_reference) == OBJECT_PARTIAL_KEYS
        numeric_expected, omitted_expected = _host_partial_arrays(failure)
        assert len(partial_reference) == 123
        assert len(numeric_expected) == 119
        assert {key for key, _ in omitted_expected} == OBJECT_PARTIAL_KEYS

        record = run.failure(failure, 0, "numerical_trial")
        assert failure.partial_arrays is partial_reference
        for key, value in object_reference.items():
            assert failure.partial_arrays[key] is value

        assert record["partial_published"] is True
        assert record["partial_array_count"] == len(partial_reference)
        assert record["partial_numeric_count"] == len(numeric_expected)
        assert set(record["partial_numeric_keys"]) == set(numeric_expected)
        omitted_keys = {
            entry["key"] for entry in record["partial_omitted"]
        }
        assert omitted_keys == OBJECT_PARTIAL_KEYS
        assert set(record["partial_omitted_keys"]) == omitted_keys
        assert all(
            entry["type"] and entry["reason"]
            for entry in record["partial_omitted"]
        )

        partial_path = run.path() / "failure" / record["partial_file"]
        with np.load(partial_path, allow_pickle=False) as archive:
            assert set(archive.files) == set(numeric_expected)
            for key, expected in numeric_expected.items():
                actual = archive[key]
                assert actual.dtype == expected.dtype
                assert actual.shape == expected.shape
                np.testing.assert_array_equal(actual, expected)
        loaded_arrays = io.read_npz(
            partial_path,
            io.Limits(),
            allowed_keys=set(numeric_expected),
            allow_nonfinite=True,
        )
        assert set(loaded_arrays) == set(numeric_expected)

        failure_path = next(
            (run.path() / "failure").glob("failure_*.json")
        )
        published = json.loads(failure_path.read_text(encoding="utf-8"))
        assert published["partial_sha256"] == record["partial_sha256"]
        metadata = published["coupling_metadata"]
        assert metadata["rollback"] is True
        assert metadata["macro_step"] == 0
        assert metadata["right_step"] == 1
        assert metadata["trial"] > 0
        assert metadata["work"]["total_full_stage_attempts"] >= 1
        assert len(metadata["residual_history"]) == 1
        assert metadata["final_normalized_residual_max"] > 1.0
        assert metadata["endpoint_residual_summary"] is not None
        diagnostic_work = published["mixed_diagnostic_work"]
        assert set(diagnostic_work) == {"accumulator", "checker"}
        assert diagnostic_work["accumulator"] is None
        assert diagnostic_work["checker"] is None

        identities = published["partial_identity_metadata"]
        assert identities["fine_identities"]
        first_identity = identities["fine_identities"][0]
        assert first_identity["iteration"] == 0
        assert first_identity["micro_step"] == 0
        assert first_identity["rk_stage"] == 0
        expected_stage = failure.stage
        assert expected_stage is not None
        assert published["stage"] == {
            "step": expected_stage.step,
            "generation": expected_stage.generation,
            "trial": expected_stage.trial,
            "spectral_context": expected_stage.spectral_context,
            "time": expected_stage.time,
            "stage": expected_stage.stage,
            "micro_step": expected_stage.micro_step,
            "iteration": expected_stage.iteration,
            "slab": expected_stage.slab,
        }
    finally:
        run.close()
