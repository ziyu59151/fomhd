"""A1-T1 complete ion stage-path waveform fixed-point transaction."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

from . import core, diagnostics, time_predictor
from . import time as time_module
from .stage import evaluate_stage
from .types import Failure, StageTag

_RK_WEIGHTS = (1.0, 2.0, 2.0, 1.0)
_FIELDS = ("n_i", "mom_i", "perp_i")
_INTERVAL_ENERGY_BUDGETS = {
    "kinetic_energy",
    "ion_internal",
    "electron_perp",
    "electron_parallel",
    "field_energy_B_interval",
    "field_exchange_interval",
    "total_energy_interval",
}

# Pinned aliases preserve the tested assembly and guard behavior. Tests replace
# these module attributes without changing the shared T0 implementation.
_assemble_state = time_predictor._assemble_state
_check_guard = time_predictor._check_guard
_linear_combination = time_predictor._linear_combination
_maximum = time_predictor._maximum
_retain = time_predictor._retain


def _tag(
    context,
    step,
    generation,
    trial,
    time,
    stage,
    micro_step,
    iteration,
):
    return StageTag(
        step=step,
        generation=generation,
        trial=trial,
        spectral_context=getattr(context, "spectral_id", 0),
        time=time,
        stage=stage,
        micro_step=micro_step,
        iteration=iteration,
        slab=0,
    )


def _stage_name(iteration, micro, rk_stage):
    return (
        f"waveform_i{iteration:03d}_m{micro:03d}_r{rk_stage + 1}"
    )


def _fresh_generation(context):
    next_generation, generation = time_predictor._fresh(
        context.next_generation
    )
    context.next_generation = next_generation
    return generation


def _physical_reference(parameters, field):
    ref = parameters.ref
    if field == "n_i":
        return float(ref.ni)
    if field == "mom_i":
        return (
            float(parameters.species[0].mass)
            * float(ref.ni)
            * float(ref.velocity)
        )
    if field == "perp_i":
        return float(ref.pressure)
    raise Failure("Time", f"unknown waveform field: {field}")


def _field_metric(context, field, old, new):
    difference = context.compute.xp.asarray(new) - context.compute.xp.asarray(
        old
    )
    raw = _maximum(context, difference)
    old_scale = _maximum(context, old)
    new_scale = _maximum(context, new)
    scale = max(
        _physical_reference(context.p, field),
        old_scale,
        new_scale,
        1.0e-300,
    )
    normalized = raw / scale
    if not (math.isfinite(raw) and math.isfinite(scale)):
        raise Failure("Time", f"nonfinite waveform {field} residual")
    return {
        "raw_max_abs": float(raw),
        "scale": float(scale),
        "normalized_max_abs": float(normalized),
    }


def _convergence_metrics(
    context,
    old_path,
    old_endpoint,
    new_path,
    new_endpoint,
):
    path = {}
    endpoint = {}
    for index, field in enumerate(_FIELDS):
        raw_path = 0.0
        old_path_scale = 0.0
        new_path_scale = 0.0
        for old, new in zip(old_path, new_path):
            metric = _field_metric(context, field, old[index], new[index])
            raw_path = max(raw_path, metric["raw_max_abs"])
            old_path_scale = max(old_path_scale, metric["scale"])
            new_path_scale = max(new_path_scale, metric["scale"])
        scale = max(
            _physical_reference(context.p, field),
            old_path_scale,
            new_path_scale,
            1.0e-300,
        )
        path[field] = {
            "raw_max_abs": float(raw_path),
            "scale": float(scale),
            "normalized_max_abs": float(raw_path / scale),
        }
        endpoint[field] = _field_metric(
            context, field, old_endpoint[index], new_endpoint[index]
        )
    path_max = max(
        value["normalized_max_abs"] for value in path.values()
    )
    endpoint_max = max(
        value["normalized_max_abs"] for value in endpoint.values()
    )
    rtol = float(context.p.waveform_rtol)
    return {
        "path": path,
        "endpoint": endpoint,
        "path_max_normalized": float(path_max),
        "endpoint_max_normalized": float(endpoint_max),
        "converged": bool(path_max <= rtol and endpoint_max <= rtol),
    }


def _budget_name(key):
    return key.split(".", 1)[0]


def _trial_context(context):
    """Soften energy interval budgets without mutating the configured context."""
    trial = copy.copy(context)
    trial.p = copy.copy(context.p)
    budgets = {}
    for key, budget in getattr(context.p, "budgets", {}).items():
        value = copy.copy(budget)
        if _budget_name(key) in _INTERVAL_ENERGY_BUDGETS:
            value.hard = False
        budgets[key] = value
    trial.p.budgets = budgets
    return trial


def _scalar(value):
    value = float(value)
    if not math.isfinite(value):
        raise Failure("Time", "nonfinite waveform budget residual")
    return value


def _energy_budgets(checks):
    out = {}
    for record in checks:
        if record.name not in _INTERVAL_ENERGY_BUDGETS:
            continue
        key = record.name + (
            f".{record.population}" if record.population else ""
        )
        out[key] = {
            "raw": _scalar(record.raw),
            "reference": _scalar(record.reference),
            "threshold": _scalar(record.threshold),
            "passed": bool(record.passed),
        }
    return out


def _metadata(
    context,
    *,
    converged,
    sweeps,
    iterations,
    quadrature,
    complete_calls,
    stage_calls,
    endpoint_calls,
    alias_calls,
    assemble_calls,
    peak_bytes,
    attempted_iteration=None,
    attempted_stage=None,
):
    source_start = int(getattr(context, "_waveform_source_start", 0))
    source_now = int(getattr(context, "next_source", source_start + 1))
    actual_model_calls = max(0, source_now - 1 - source_start)
    out = {
        "coupling": "waveform",
        "micro_steps": int(context.p.micro_steps),
        "sweeps": int(sweeps),
        "sweep_count": int(sweeps),
        "converged": bool(converged),
        "waveform_rtol": float(context.p.waveform_rtol),
        "waveform_max_iterations": int(
            context.p.waveform_max_iterations
        ),
        "rk4_tableau_weights": [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
        "quadrature_weights": [
            float(weight) for _, _, weight in quadrature
        ],
        "quadrature_stage_evaluations": len(quadrature),
        "scheduled_quadrature_stage_evaluations": (
            4 * int(context.p.micro_steps)
        ),
        "full_rhs_stage_evaluations": int(stage_calls),
        "endpoint_full_evaluations": int(endpoint_calls),
        "complete_stage_evaluations": int(complete_calls),
        "assemble_canonicalization_calls": int(assemble_calls),
        "alias_evaluations": int(alias_calls),
        "total_model_evaluations": actual_model_calls,
        "actual_model_call_count": actual_model_calls,
        "retained_quadrature_entries": len(quadrature),
        "retained_sweep_entries": 4 * int(context.p.micro_steps),
        "estimated_peak_bytes": int(peak_bytes or 0),
        "iterations": iterations,
        "guard_scope": (
            "micro-RK4 only; waveform convergence is reported separately"
        ),
        "controls": {
            "waveform_rtol": float(context.p.waveform_rtol),
            "waveform_max_iterations": int(
                context.p.waveform_max_iterations
            ),
        },
    }
    if attempted_iteration is not None:
        out["attempted_iteration"] = int(attempted_iteration)
    if attempted_stage is not None:
        out["attempted_stage"] = {
            "step": int(attempted_stage.step),
            "generation": int(attempted_stage.generation),
            "trial": int(attempted_stage.trial),
            "spectral_context": int(attempted_stage.spectral_context),
            "time": float(attempted_stage.time),
            "stage": str(attempted_stage.stage),
            "micro_step": int(attempted_stage.micro_step),
            "iteration": int(attempted_stage.iteration),
            "slab": int(attempted_stage.slab),
        }
    return out


def advance(committed, cumulative, context):
    """Advance one macro interval by complete ion-path waveform sweeps."""
    coupling, micro_steps = time_module.validate_time_controls(context.p)
    if coupling != "waveform":
        raise Failure("Config", "waveform scheduler requires waveform coupling")
    n = time_module._check_clock(context, committed, cumulative)
    peak_bytes = core.preflight_memory(context)
    time_predictor._validate_clocks(context.p, n)
    dt = float(context.p.dt)
    h = dt / micro_steps
    if not math.isfinite(h) or h <= 0.0:
        raise Failure("TimeResolution", "waveform microstep is not representable")

    next_trial, trial = time_predictor._fresh(context.next_trial)
    context.next_trial = next_trial
    context._waveform_source_start = int(getattr(context, "next_source", 1)) - 1
    initial_ion, initial_z = time_predictor._parts(committed.state)

    iterations = []
    evidence = []
    evidence_result = None
    metadata = None
    quadrature = []
    base = None
    final_endpoint = None
    candidate = None
    complete_calls = 0
    stage_calls = 0
    endpoint_calls = 0
    alias_calls = 0
    assemble_calls = 0
    old_path = None
    old_endpoint = initial_ion
    coupling_converged = False
    current_iteration = None
    current_record = None

    try:
        for iteration in range(int(context.p.waveform_max_iterations)):
            current_iteration = iteration
            current_record = None
            evidence_result = None
            scheduled = []
            for micro in range(micro_steps):
                for rk_stage in range(4):
                    name = _stage_name(iteration, micro, rk_stage)
                    generation = (
                        committed.state.generation
                        if iteration == 0 and micro == 0 and rk_stage == 0
                        else _fresh_generation(context)
                    )
                    tag = _tag(
                        context,
                        n,
                        generation,
                        trial,
                        core.microstage_time(
                            context.p, n, micro, rk_stage
                        ),
                        name,
                        micro,
                        iteration,
                    )
                    scheduled.append(
                        SimpleNamespace(
                            prefix=name,
                            tag=tag,
                            state=None,
                            stage=None,
                        )
                    )
            endpoint_name = f"waveform_i{iteration:03d}_endpoint"
            endpoint_tag = _tag(
                context,
                n + 1,
                _fresh_generation(context),
                trial,
                core.time_at_step(context.p, n + 1),
                endpoint_name,
                -1,
                iteration,
            )
            endpoint_evidence = SimpleNamespace(
                prefix=endpoint_name,
                tag=endpoint_tag,
                state=None,
                stage=None,
            )
            evidence = scheduled + [endpoint_evidence]
            by_prefix = {item.prefix: item for item in evidence}

            def populate(prefix, state, stage, slots=by_prefix):
                item = slots[prefix]
                if item.tag != stage.tag:
                    raise Failure(
                        "Time", f"stale waveform evidence slot: {prefix}"
                    )
                item.state = state
                item.stage = stage

            if iteration == 0:
                base_prefix = _stage_name(0, 0, 0)
                complete_calls += 1
                stage_calls += 1
                base = evaluate_stage(
                    committed.state,
                    by_prefix[base_prefix].tag,
                    context,
                    False,
                )
                populate(base_prefix, committed.state, base)
                _check_guard(base, context)
                f0 = (
                    base.rhs.fluid[0].n,
                    base.rhs.fluid[0].mom,
                    base.rhs.fluid[0].perp,
                )
                old_path = []
                for micro in range(micro_steps):
                    for rk_stage in range(4):
                        offset = core.microstage_offset(
                            context.p, micro, rk_stage
                        )
                        old_path.append(
                            tuple(
                                _linear_combination(
                                    context,
                                    initial_ion[index],
                                    [(offset, f0[index])],
                                )
                                for index in range(3)
                            )
                        )
                old_endpoint = tuple(
                    _linear_combination(
                        context,
                        initial_ion[index],
                        [(dt, f0[index])],
                    )
                    for index in range(3)
                )

            z_current = initial_z
            q = initial_ion
            new_path = []
            quadrature = []
            evaluated = []
            for micro in range(micro_steps):
                derivatives_z = []
                derivatives_ion = []
                for rk_stage in range(4):
                    name = _stage_name(iteration, micro, rk_stage)
                    tag = by_prefix[name].tag
                    z_stage = time_predictor._rk_stage_state(
                        context,
                        z_current,
                        derivatives_z,
                        rk_stage,
                        h,
                    )
                    if iteration == 0 and micro == 0 and rk_stage == 0:
                        state = committed.state
                        stage = base
                    else:
                        alias = rk_stage in (1, 2)
                        assemble_calls += 1
                        state = _assemble_state(
                            context,
                            old_path[4 * micro + rk_stage],
                            z_stage,
                            tag,
                        )
                        complete_calls += 1
                        stage_calls += 1
                        stage = evaluate_stage(
                            state, tag, context, alias
                        )
                        if alias:
                            alias_calls += 1
                        populate(name, state, stage)
                    derivatives_z.append(
                        (
                            stage.rhs.fluid[1].perp,
                            stage.rhs.fluid[1].parallel,
                            stage.rhs.B,
                        )
                    )
                    derivatives_ion.append(
                        (
                            stage.rhs.fluid[0].n,
                            stage.rhs.fluid[0].mom,
                            stage.rhs.fluid[0].perp,
                        )
                    )
                    evaluated.append(stage)
                    quadrature.append(
                        (
                            state,
                            stage,
                            h * _RK_WEIGHTS[rk_stage] / 6.0,
                        )
                    )

                f1, f2, f3, _f4 = derivatives_ion
                new_path.extend(
                    [
                        tuple(q),
                        tuple(
                            _linear_combination(
                                context, q[i], [(h / 2.0, f1[i])]
                            )
                            for i in range(3)
                        ),
                        tuple(
                            _linear_combination(
                                context, q[i], [(h / 2.0, f2[i])]
                            )
                            for i in range(3)
                        ),
                        tuple(
                            _linear_combination(
                                context, q[i], [(h, f3[i])]
                            )
                            for i in range(3)
                        ),
                    ]
                )
                q = tuple(
                    _linear_combination(
                        context,
                        q[i],
                        [
                            (
                                h * weight / 6.0,
                                derivative[i],
                            )
                            for weight, derivative in zip(
                                _RK_WEIGHTS, derivatives_ion
                            )
                        ],
                    )
                    for i in range(3)
                )
                z_current = tuple(
                    _linear_combination(
                        context,
                        z_current[i],
                        [
                            (h * weight / 6.0, derivative[i])
                            for weight, derivative in zip(
                                _RK_WEIGHTS, derivatives_z
                            )
                        ],
                    )
                    for i in range(3)
                )

            metrics = _convergence_metrics(
                context,
                old_path,
                old_endpoint,
                new_path,
                q,
            )
            current_record = {
                "iteration": iteration,
                **metrics,
                "energy_budgets": {},
                "energy_available": False,
            }
            iterations.append(current_record)
            coupling_converged = bool(metrics["converged"])

            assemble_calls += 1
            endpoint_state = _assemble_state(
                context, q, z_current, endpoint_tag
            )
            complete_calls += 1
            endpoint_calls += 1
            endpoint = evaluate_stage(
                endpoint_state, endpoint_tag, context, False
            )
            populate(endpoint_name, endpoint_state, endpoint)

            first_state, first_stage, _ = quadrature[0]
            evidence_result = diagnostics.check_step(
                committed.state,
                base,
                first_state,
                first_stage,
                endpoint_state,
                endpoint,
                cumulative,
                _trial_context(context),
                quadrature=quadrature,
            )
            energy = _energy_budgets(evidence_result.checks)
            current_record["energy_budgets"] = energy
            current_record["energy_available"] = bool(energy)
            metadata = _metadata(
                context,
                converged=metrics["converged"],
                sweeps=iteration + 1,
                iterations=iterations,
                quadrature=quadrature,
                complete_calls=complete_calls,
                stage_calls=stage_calls,
                endpoint_calls=endpoint_calls,
                alias_calls=alias_calls,
                assemble_calls=assemble_calls,
                peak_bytes=peak_bytes,
            )
            if not metrics["converged"]:
                if iteration + 1 == context.p.waveform_max_iterations:
                    failure = Failure(
                        "WaveformConvergence",
                        "waveform fixed point did not converge after "
                        f"{iteration + 1} sweeps",
                    )
                    failure.stage = endpoint_tag
                    failure.coupling_metadata = metadata
                    raise failure
                old_path = new_path
                old_endpoint = q
                continue

            final_name = "waveform_final_endpoint"
            final_tag = _tag(
                context,
                n + 1,
                _fresh_generation(context),
                trial,
                core.time_at_step(context.p, n + 1),
                "endpoint",
                -1,
                iteration,
            )
            final_evidence = SimpleNamespace(
                prefix=final_name,
                tag=final_tag,
                state=None,
                stage=None,
            )
            evidence.append(final_evidence)
            assemble_calls += 1
            candidate = _assemble_state(
                context, q, z_current, final_tag
            )
            complete_calls += 1
            endpoint_calls += 1
            final_endpoint = evaluate_stage(
                candidate, final_tag, context, False
            )
            final_evidence.state = candidate
            final_evidence.stage = final_endpoint

            final_result = diagnostics.check_step(
                committed.state,
                base,
                first_state,
                first_stage,
                candidate,
                final_endpoint,
                cumulative,
                context,
                quadrature=quadrature,
            )
            evidence_result = final_result
            checks = list(base.checks)
            for stage in evaluated:
                if stage.tag != base.tag:
                    checks.extend(stage.checks)
            checks.extend(final_endpoint.checks)
            checks.extend(final_result.checks)
            final_result.checks = checks
            diagnostics.gate(checks)
            if (
                getattr(cumulative, "accepted_intervals", 0)
                and getattr(cumulative, "last_trial", None) == trial
            ):
                raise Failure(
                    "DuplicateInterval", "interval already accumulated"
                )
            pending = time_module._pending_cumulative(
                cumulative, quadrature, candidate, trial
            )
            metadata = _metadata(
                context,
                converged=True,
                sweeps=iteration + 1,
                iterations=iterations,
                quadrature=quadrature,
                complete_calls=complete_calls,
                stage_calls=stage_calls,
                endpoint_calls=endpoint_calls,
                alias_calls=alias_calls,
                assemble_calls=assemble_calls,
                peak_bytes=peak_bytes,
            )
            context.compute.synchronize()
            time_module._commit(
                committed,
                cumulative,
                pending,
                candidate,
                final_endpoint.cache,
                final_result,
            )
            final_result.frame = final_endpoint
            final_result.time_step_report = getattr(
                context, "last_dt_report", None
            )
            final_result.coupling_metadata = metadata
            final_result.waveform_metadata = metadata
            committed.frame = final_endpoint
            return final_result
    except Failure as failure:
        failure.time_step_report = getattr(
            context, "last_dt_report", None
        )
        fallback = (
            evidence[-1].tag
            if evidence
            else StageTag(
                step=n,
                generation=committed.state.generation,
                trial=trial,
                time=committed.state.time,
                stage="waveform",
            )
        )
        if current_record is not None and not current_record[
            "energy_available"
        ]:
            energy = _energy_budgets(getattr(failure, "checks", ()))
            if energy:
                current_record["energy_budgets"] = energy
                current_record["energy_available"] = True
        attempted_stage = failure.stage or fallback
        failure.coupling_metadata = _metadata(
            context,
            converged=coupling_converged,
            sweeps=len(iterations),
            iterations=iterations,
            quadrature=quadrature,
            complete_calls=complete_calls,
            stage_calls=stage_calls,
            endpoint_calls=endpoint_calls,
            alias_calls=alias_calls,
            assemble_calls=assemble_calls,
            peak_bytes=peak_bytes,
            attempted_iteration=current_iteration,
            attempted_stage=attempted_stage,
        )
        _retain(
            failure,
            context,
            evidence,
            evidence_result,
            fallback,
        )
        raise
    except Exception as error:
        failure = Failure("NumericalLibrary", str(error))
        fallback = (
            final_endpoint.tag
            if final_endpoint is not None
            else evidence[-1].tag
            if evidence
            else StageTag(
                step=n,
                generation=committed.state.generation,
                trial=trial,
                time=committed.state.time,
                stage="waveform",
            )
        )
        failure.stage = fallback
        failure.coupling_metadata = _metadata(
            context,
            converged=coupling_converged,
            sweeps=len(iterations),
            iterations=iterations,
            quadrature=quadrature,
            complete_calls=complete_calls,
            stage_calls=stage_calls,
            endpoint_calls=endpoint_calls,
            alias_calls=alias_calls,
            assemble_calls=assemble_calls,
            peak_bytes=peak_bytes,
            attempted_iteration=current_iteration,
            attempted_stage=fallback,
        )
        _retain(
            failure,
            context,
            evidence,
            evidence_result,
            failure.stage,
        )
        raise failure from error
