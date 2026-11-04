"""Focused tests for optional scientific diagnostic JSON evidence."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np

from fomhd import io
from fomhd.__main__ import _context as runtime_context
from fomhd.stage import closed_reference, initialize
from fomhd.types import Failure, check_record
from test_model_routing import _context, _cumulative, _write_case


BASE_KEYS = {
    "name",
    "population",
    "component",
    "producer",
    "region",
    "selected_count",
    "step",
    "generation",
    "trial",
    "spectral_context",
    "time",
    "stage",
    "raw",
    "reference",
    "threshold",
    "unit",
    "value_unit",
    "norm",
    "gate",
    "status",
    "reason",
    "worst_index",
}
OPTIONAL_KEYS = {
    "modal_work",
    "sample_work",
    "transform_difference",
    "absolute_final_terms",
    "upstream_bound",
    "arithmetic_bound",
    "compatibility_bound",
    "configured_threshold",
}


def _extended_record(tag, name, **values):
    record = check_record(
        name=name,
        population="",
        region="all",
        unit="W",
        value_unit="W",
        producer="massless_checks",
        tag=tag,
        raw=0.0,
        reference=1.0,
        threshold=1.0e-12,
        norm="value",
        hard=False,
        available=True,
        passed=True,
        selected=1,
    )
    for key, value in values.items():
        setattr(record, key, value)
    record.arbitrary_private_note = 17
    record.backend_array = np.asarray([1.0, 2.0])
    return record


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_optional_wall_work_evidence_survives_normal_and_failure_json(tmp_path):
    context = _context(inertia=1, nsteps=0, hard=True)
    state = closed_reference(
        context,
        context.p.ref.ni,
        context.p.ref.pressure,
        context.p.ref.pressure,
        context.p.ref.magnetic,
    )
    case_path = _write_case(
        tmp_path / "input",
        context.p,
        context.geometry,
        state,
        "midpoint",
        0,
    )
    case = io.load_case(
        case_path, {"output": str(tmp_path / "run")}
    )
    cumulative = _cumulative()
    captured = {}
    committed = initialize(
        io.load_initial(case),
        runtime_context(case),
        cumulative,
        False,
        captured,
    )
    frame = SimpleNamespace(**captured)
    default = next(
        check
        for check in frame.checks
        if check.name == "initial_divB"
    )
    finite = _extended_record(
        frame.tag,
        "wall_field_reaction_work",
        modal_work=2.0,
        sample_work=2.5,
        transform_difference=0.5,
        absolute_final_terms=3.0,
        upstream_bound=4.0,
        arithmetic_bound=5.0,
        compatibility_bound=0.25,
        configured_threshold=6.0,
    )
    nonfinite = _extended_record(
        frame.tag,
        "wall_field_reaction_work_nonfinite",
        modal_work=math.nan,
        sample_work=math.inf,
        transform_difference=-math.inf,
        absolute_final_terms=1.0,
        upstream_bound=1.0,
        arithmetic_bound=1.0,
        compatibility_bound=-math.inf,
        configured_threshold=1.0,
    )

    run = io.Run(case)
    try:
        run.diagnostics(
            committed.state,
            [default, finite, nonfinite],
            frame.totals,
            frame.momentum,
            0.0,
        )
        persisted = {
            record["name"]: record
            for record in _read_jsonl(run.path() / "checks.jsonl")
        }
        assert set(persisted["initial_divB"]) == BASE_KEYS
        assert set(persisted["wall_field_reaction_work"]) == (
            BASE_KEYS | OPTIONAL_KEYS
        )
        assert persisted["wall_field_reaction_work"]["unit"] == "W"
        assert (
            persisted["wall_field_reaction_work"]["value_unit"] == "W"
        )
        for key, value in (
            ("modal_work", 2.0),
            ("sample_work", 2.5),
            ("transform_difference", 0.5),
            ("absolute_final_terms", 3.0),
            ("upstream_bound", 4.0),
            ("arithmetic_bound", 5.0),
            ("compatibility_bound", 0.25),
            ("configured_threshold", 6.0),
        ):
            assert persisted["wall_field_reaction_work"][key] == value
        assert "arbitrary_private_note" not in (
            persisted["wall_field_reaction_work"]
        )
        assert "backend_array" not in (
            persisted["wall_field_reaction_work"]
        )
        nonfinite_json = persisted["wall_field_reaction_work_nonfinite"]
        assert nonfinite_json["modal_work"] == {
            "value": "NaN",
            "type": "float64",
        }
        assert nonfinite_json["sample_work"] == {
            "value": "+Inf",
            "type": "float64",
        }
        assert nonfinite_json["transform_difference"] == {
            "value": "-Inf",
            "type": "float64",
        }
        assert nonfinite_json["compatibility_bound"] == {
            "value": "-Inf",
            "type": "float64",
        }

        failure = Failure("DiagnosticEvidence", "injected")
        failure.stage = frame.tag
        failure.checks = [default, finite, nonfinite]
        run.failure(
            failure,
            committed.state.step,
            "numerical_trial",
        )
        failure_path = next((run.path() / "failure").glob("failure_*.json"))
        receipt = json.loads(
            failure_path.read_text(encoding="utf-8")
        )
        failure_checks = {
            record["name"]: record for record in receipt["checks"]
        }
        assert set(failure_checks["initial_divB"]) == BASE_KEYS
        assert set(failure_checks["wall_field_reaction_work"]) == (
            BASE_KEYS | OPTIONAL_KEYS
        )
        assert (
            failure_checks["wall_field_reaction_work"]["modal_work"]
            == 2.0
        )
        assert (
            failure_checks["wall_field_reaction_work"][
                "compatibility_bound"
            ]
            == 0.25
        )
        assert failure_checks[
            "wall_field_reaction_work_nonfinite"
        ]["modal_work"] == {"value": "NaN", "type": "float64"}
        assert "arbitrary_private_note" not in failure_checks[
            "wall_field_reaction_work"
        ]
    finally:
        run.close()
