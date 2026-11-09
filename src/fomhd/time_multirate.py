"""Complete accepted macro transaction for the selected massless A1-WR scheme."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import SimpleNamespace

from . import core, diagnostics, massless_checks, multirate_wr_checks
from .multirate_wr_physics import (
    MasslessWRPhysicsAdapter,
    pack_state,
)
from .stage import evaluate_stage, retain_failure_evidence
from .types import Failure, StageTag


def _scheme(parameters):
    return {
        "coupling": "multirate_wr",
        "micro_steps": int(parameters.micro_steps),
        "ion_nodes": int(parameters.ion_nodes),
        "waveform_rtol": float(parameters.waveform_rtol),
        "waveform_max_iterations": int(
            parameters.waveform_max_iterations
        ),
        "wr_initial": str(parameters.wr_initial),
        "wr_relaxation": float(parameters.wr_relaxation),
    }


def _compact_residual_history(history):
    if history is None:
        return None
    rows = []
    for record in history:
        if not isinstance(record, Mapping):
            raise Failure(
                "WRMetadata",
                "residual_history must contain actual mapping records",
            )
        rows.append(dict(record))
    return rows or None


def _residual_metrics(history):
    if not history:
        return None, None, None
    final = history[-1]
    normalized = final.get(
        "normalized_max",
        final.get("normalized_linf"),
    )
    raw = final.get("raw_linf")
    endpoint = {
        key: final.get(key)
        for key in (
            "endpoint_raw_linf",
            "endpoint_scaled_linf",
            "endpoint_normalized_linf",
        )
    }
    return (
        None if normalized is None else float(normalized),
        None if raw is None else float(raw),
        None if not endpoint or any(value is None for value in endpoint.values())
        else endpoint,
    )


def _metadata(
    context,
    *,
    trial,
    left,
    endpoint,
    physics,
    bundle,
    base,
    result,
    rollback,
):
    scheme = _scheme(context.p)
    interval = None if bundle is None else bundle.interval
    work = (
        {
            "full_stage_calls": 0,
            "adapter": None,
        }
        if physics is None
        else physics.work.snapshot()
    )
    if physics is None:
        work.setdefault("full_stage_attempts", 0)
        work.setdefault("full_stage_calls", 0)
        work.setdefault("full_stage_failures", 0)
        work.setdefault("full_ion_power_ledger_values", 0)
        work.setdefault("full_electron_power_ledger_values", 0)
        work.setdefault("fine_ion_power_ledger_calls", 0)
    adapter_attempts = int(work.get("full_stage_attempts", 0))
    adapter_calls = int(work.get("full_stage_calls", 0))
    adapter_failures = int(work.get("full_stage_failures", 0))
    work["adapter_full_stage_attempts"] = adapter_attempts
    work["adapter_full_stage_calls"] = adapter_calls
    work["adapter_full_stage_failures"] = adapter_failures
    work["base_full_stage_attempts"] = 1
    work["base_full_stage_calls"] = int(base is not None)
    work["base_full_stage_failures"] = int(base is None)
    base_ion_ledger = (
        0
        if base is None
        else int(getattr(base, "ledger", (None, None))[0] is not None)
    )
    base_electron_ledger = (
        0
        if base is None
        else int(getattr(base, "ledger", (None, None))[1] is not None)
    )
    work["total_full_stage_attempts"] = adapter_attempts + 1
    work["total_full_stage_calls"] = adapter_calls + int(base is not None)
    work["total_full_stage_failures"] = adapter_failures + int(base is None)
    work["full_ion_power_ledger_values"] = (
        int(work.get("full_ion_power_ledger_values", 0))
        + base_ion_ledger
    )
    work["full_electron_power_ledger_values"] = (
        int(work.get("full_electron_power_ledger_values", 0))
        + base_electron_ledger
    )
    if physics is not None:
        work["accepted_endpoint_full_stage_calls"] = 1
    bundle_diagnostic_work = (
        None
        if bundle is None
        else bundle.diagnostic_work.as_dict()
    )
    result_diagnostic_work = None
    if result is not None and getattr(result, "diagnostic_work", None) is not None:
        result_diagnostic_work = result.diagnostic_work.as_dict()
    base_ion_kinetic = (
        None if base is None else float(base.totals.kinetic[0])
    )
    endpoint_ion_kinetic = (
        None if endpoint is None else float(endpoint.totals.kinetic[0])
    )
    epsilon_k = (
        None
        if interval is None or base is None or endpoint is None
        else endpoint_ion_kinetic - base_ion_kinetic - interval.kinetic_path
    )
    checks = [] if result is None else list(result.checks)
    residual_history = (
        None
        if physics is None
        else _compact_residual_history(physics.window.residual_history)
    )
    (
        final_normalized,
        final_raw,
        endpoint_summary,
    ) = _residual_metrics(residual_history)
    return {
        **scheme,
        "macro_step": int(left.step),
        "left_step": int(left.step),
        "right_step": int(left.step) + 1,
        "left_time": float(left.time),
        "right_time": float(core.time_at_step(context.p, left.step + 1)),
        "trial": int(trial),
        "node_version": (
            None if physics is None else int(physics.window.accepted_node_version)
        ),
        "sweeps": None if physics is None else int(physics.window.sweeps),
        "waveform_residual_scaled": (
            None
            if physics is None
            else float(physics.window.residual_max_scaled)
        ),
        "waveform_raw_norm": (
            None if physics is None else float(physics.window.raw_norm)
        ),
        "residual_history": residual_history,
        "final_normalized_residual_max": final_normalized,
        "final_raw_norm": final_raw,
        "endpoint_residual_summary": endpoint_summary,
            "work": work,
            "bundle_diagnostic_work": bundle_diagnostic_work,
            "result_diagnostic_work": result_diagnostic_work,
            "base_source_id": (
                None if base is None else int(base.source_id)
            ),
        "endpoint_source_id": (
            None if endpoint is None else int(endpoint.source_id)
        ),
            "interval": interval,
            "base_ion_kinetic_J": base_ion_kinetic,
            "endpoint_ion_kinetic_J": endpoint_ion_kinetic,
            "epsilon_K_J": epsilon_k,
        "coarse_generations": (
            ()
            if physics is None
            else tuple(record.tag.generation for record in physics.coarse_records)
        ),
        "fine_generations": (
            ()
            if physics is None
            else tuple(record.tag.generation for record in physics.fine_records)
        ),
        "interval_attribution": (
            None
            if interval is None
            else {
                "delta_D": (
                    interval.density_work_coarse
                    - interval.density_work_fine
                ),
                "density_work_coarse": interval.density_work_coarse,
                "density_work_fine": interval.density_work_fine,
                "kinetic_path": interval.kinetic_path,
            }
        ),
        "endpoint": (
            None
            if endpoint is None
            else {
                "step": int(endpoint.tag.step),
                "time": float(endpoint.tag.time),
                "generation": int(endpoint.tag.generation),
                "trial": int(endpoint.tag.trial),
                "production_accepted": bool(
                    getattr(physics.endpoint, "production_accepted", False)
                ),
                "trial_endpoint": bool(
                    getattr(physics.endpoint, "trial_endpoint", True)
                ),
                "interval_transaction_required": bool(
                    getattr(
                        physics.endpoint,
                        "interval_transaction_required",
                        True,
                    )
                ),
            }
        ),
        "guard_scope": "micro-RK4 h; does not prove macro coupling stability",
        "checks": checks,
        "rollback": bool(rollback),
        "scope": (
            "one complete massless A1-WR macro transaction; "
            "no scientific acceptance claim"
        ),
    }


def _retain_failure(
    failure,
    context,
    *,
    trial,
    left,
    base,
    adapter,
    physics,
    bundle,
    endpoint,
    result,
):
    failure.stage = getattr(failure, "stage", None)
    if base is not None:
        retain_failure_evidence(
            failure,
            base.tag,
            context,
            left,
            base,
            "multirate_base",
        )
    if endpoint is not None:
        state = getattr(endpoint, "canonical_state", None)
        if state is not None:
            retain_failure_evidence(
                failure,
                endpoint.tag,
                context,
                state,
                endpoint,
                "multirate_endpoint",
            )
    evidence = {
        "trial": int(trial),
        "base": base,
        "adapter": adapter,
        "physics": physics,
        "bundle": bundle,
        "endpoint": endpoint,
        "step_result": result,
    }
    failure.multirate_evidence = evidence
    failure.coupling_metadata = _metadata(
        context,
        trial=trial,
        left=left,
        endpoint=endpoint,
        physics=physics,
        bundle=bundle,
        base=base,
        result=result,
        rollback=True,
    )
    failure_history = _compact_residual_history(
        getattr(failure, "residual_history", None)
    )
    if failure_history is None:
        failure_history = failure.coupling_metadata.get("residual_history")
    (
        failure_final_normalized,
        failure_final_raw,
        failure_endpoint,
    ) = _residual_metrics(failure_history)
    failure.coupling_metadata["residual_history"] = failure_history
    failure.coupling_metadata[
        "final_normalized_residual_max"
    ] = failure_final_normalized
    failure.coupling_metadata["final_raw_norm"] = failure_final_raw
    failure.coupling_metadata[
        "endpoint_residual_summary"
    ] = failure_endpoint
    if adapter is not None:
        work = dict(failure.coupling_metadata["work"])
        adapter_work = adapter.work.snapshot()
        for name in (
            "full_stage_attempts",
            "full_stage_calls",
            "full_stage_failures",
            "full_ion_power_ledger_values",
            "full_electron_power_ledger_values",
            "fine_ion_power_ledger_calls",
            "endpoint_reconstruction_calls",
            "endpoint_full_stage_calls",
            "coarse_transport_calls",
            "field_callback_calls",
            "field_evaluations",
            "alias_field_evaluations",
        ):
            work[name] = adapter_work[name]
        work["adapter_full_stage_attempts"] = adapter_work["full_stage_attempts"]
        work["adapter_full_stage_calls"] = adapter_work["full_stage_calls"]
        work["adapter_full_stage_failures"] = adapter_work["full_stage_failures"]
        work["total_full_stage_attempts"] = (
            adapter_work["full_stage_attempts"]
            + work["base_full_stage_attempts"]
        )
        work["total_full_stage_calls"] = (
            adapter_work["full_stage_calls"]
            + work["base_full_stage_calls"]
        )
        work["total_full_stage_failures"] = (
            adapter_work["full_stage_failures"]
            + work["base_full_stage_failures"]
        )
        base_ion_ledger = int(
            base is not None
            and getattr(base, "ledger", (None, None))[0] is not None
        )
        base_electron_ledger = int(
            base is not None
            and getattr(base, "ledger", (None, None))[1] is not None
        )
        work["full_ion_power_ledger_values"] = (
            adapter_work["full_ion_power_ledger_values"]
            + base_ion_ledger
        )
        work["full_electron_power_ledger_values"] = (
            adapter_work["full_electron_power_ledger_values"]
            + base_electron_ledger
        )
        work["accepted_endpoint_full_stage_calls"] = 0
        failure.coupling_metadata["work"] = work
    if not hasattr(failure, "partial_arrays"):
        failure.partial_arrays = {}
    if adapter is not None:
        failure.adapter_work = adapter.work.snapshot()
        failure.partial_arrays.setdefault(
            "multirate_all_coarse_records",
            tuple(adapter.coarse_records),
        )
        failure.partial_arrays.setdefault(
            "multirate_all_fine_records",
            tuple(adapter.fine_records),
        )
        for index, record in enumerate(adapter.coarse_records):
            failure.partial_arrays.setdefault(
                f"multirate_coarse_{index}_A", record.A
            )
            failure.partial_arrays.setdefault(
                f"multirate_coarse_{index}_C", record.C
            )
            failure.partial_arrays.setdefault(
                f"multirate_coarse_{index}_F", record.F
            )
        for index, record in enumerate(adapter.fine_records):
            failure.partial_arrays.setdefault(
                f"multirate_fine_{index}_P", record.input_P
            )
            failure.partial_arrays.setdefault(
                f"multirate_fine_{index}_Z", record.input_Z
            )
            failure.partial_arrays.setdefault(
                f"multirate_fine_{index}_G", record.G
            )
            failure.partial_arrays.setdefault(
                f"multirate_fine_{index}_C", record.C
            )
    return failure


def advance(committed, cumulative, context):
    """Advance one accepted multirate macro or leave all commits untouched."""

    from . import time as time_module
    from .time_predictor import _check_guard

    coupling, micro_steps = time_module.validate_time_controls(context.p)
    if coupling != "multirate_wr":
        raise Failure("Config", "multirate transaction requires multirate_wr")
    n = time_module._check_clock(context, committed, cumulative)
    left = committed.state
    dt = float(context.p.dt)
    t0 = core.time_at_step(context.p, n)
    t1 = core.time_at_step(context.p, n + 1)
    if not math.isfinite(dt) or dt <= 0.0 or not (t0 < t1):
        raise Failure("Time", "invalid multirate macro interval")

    next_trial, trial = time_module._fresh(context.next_trial)
    context.next_trial = next_trial
    base_tag = StageTag(
        step=n,
        generation=int(left.generation),
        trial=trial,
        spectral_context=int(context.spectral_id),
        time=t0,
        stage="multirate_base",
        slab=0,
    )
    base = endpoint = adapter = physics = bundle = result = None
    try:
        base = evaluate_stage(left, base_tag, context, False)
        _check_guard(base, context)
        p0, z0 = pack_state(context, left)
        adapter = MasslessWRPhysicsAdapter(
            context,
            left,
            slab=0,
            trial_id=trial,
        )
        def endpoint_tag_factory():
            next_generation, endpoint_generation = time_module._fresh(
                context.next_generation
            )
            context.next_generation = next_generation
            return StageTag(
                step=n + 1,
                generation=endpoint_generation,
                trial=trial,
                spectral_context=int(context.spectral_id),
                time=t1,
                stage="endpoint",
                slab=0,
            )

        physics = adapter.solve_window(
            p0,
            z0,
            H=dt,
            M=micro_steps,
            s=int(context.p.ion_nodes),
            rtol=float(context.p.waveform_rtol),
            max_iterations=int(context.p.waveform_max_iterations),
            relaxation=float(context.p.wr_relaxation),
            initial=str(context.p.wr_initial),
            endpoint_tag_factory=endpoint_tag_factory,
        )
        endpoint = physics.endpoint.full
        if (
            endpoint.tag.step != n + 1
            or endpoint.canonical_state.step != n + 1
            or endpoint.canonical_state.time != t1
            or endpoint.canonical_state.generation
            != endpoint.tag.generation
            or endpoint.cache.tag != endpoint.tag
            or endpoint.tag.trial != trial
            or endpoint.tag.stage != "endpoint"
        ):
            raise Failure(
                "WREndpoint",
                "adapter endpoint does not match accepted macro clock",
            )
        bundle = multirate_wr_checks.accumulate_mixed_interval(
            physics.window,
            left,
            endpoint,
            context,
        )
        fine_weights = physics.window.fine_weights.reshape(-1)
        if len(fine_weights) != len(bundle.fine_views):
            raise Failure("WRQuadrature", "fine view count mismatch")
        quadrature = [
            (view.state, view, float(weight))
            for view, weight in zip(bundle.fine_views, fine_weights)
        ]
        if not quadrature:
            raise Failure("WRQuadrature", "fine quadrature is empty")
        result = massless_checks.check_step(
            left,
            base,
            quadrature[0][0],
            quadrature[0][1],
            endpoint.canonical_state,
            endpoint,
            cumulative,
            context,
            quadrature=quadrature,
            mixed_interval=bundle.interval,
        )
        checks = list(base.checks)
        for record in physics.coarse_records:
            checks.extend(record.full.checks)
        for record in physics.fine_records:
            checks.extend(record.frame.checks)
        checks.extend(endpoint.checks)
        checks.extend(result.checks)
        result.checks = checks
        diagnostics.gate(checks)
        if (
            getattr(cumulative, "accepted_intervals", 0)
            and getattr(cumulative, "last_trial", None) == trial
        ):
            raise Failure(
                "DuplicateInterval",
                "multirate interval already accumulated",
            )
        pending = time_module._pending_cumulative(
            cumulative,
            quadrature,
            endpoint.canonical_state,
            trial,
            mixed_interval=bundle.interval,
        )
        coupling_metadata = _metadata(
            context,
            trial=trial,
            left=left,
            endpoint=endpoint,
            physics=physics,
            bundle=bundle,
            base=base,
            result=result,
            rollback=False,
        )
        context.compute.synchronize()
        time_module._commit(
            committed,
            cumulative,
            pending,
            endpoint.canonical_state,
            endpoint.cache,
            result,
        )
        result.frame = endpoint
        result.endpoint = endpoint
        result.time_step_report = getattr(context, "last_dt_report", None)
        result.coupling_metadata = coupling_metadata
        physics.endpoint.production_accepted = True
        physics.endpoint.trial_endpoint = False
        physics.endpoint.interval_transaction_required = False
        metadata_endpoint = result.coupling_metadata["endpoint"]
        metadata_endpoint["production_accepted"] = True
        metadata_endpoint["trial_endpoint"] = False
        metadata_endpoint["interval_transaction_required"] = False
        committed.frame = endpoint
        committed.coupling_metadata = result.coupling_metadata
        return result
    except Failure as failure:
        _retain_failure(
            failure,
            context,
            trial=trial,
            left=left,
            base=base,
            adapter=adapter,
            physics=physics,
            bundle=bundle,
            endpoint=endpoint,
            result=result,
        )
        raise
    except Exception as error:
        failure = Failure("NumericalLibrary", str(error))
        failure.stage = (
            endpoint.tag
            if endpoint is not None
            else getattr(base, "tag", base_tag)
        )
        _retain_failure(
            failure,
            context,
            trial=trial,
            left=left,
            base=base,
            adapter=adapter,
            physics=physics,
            bundle=bundle,
            endpoint=endpoint,
            result=result,
        )
        raise failure from error
