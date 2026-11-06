"""A0 synchronous explicit RK4 and explicit midpoint transactions."""

from __future__ import annotations

import copy
import inspect
import math
import sys
from types import SimpleNamespace

from . import core, diagnostics
from .stage import evaluate_stage, retain_failure_evidence
from .types import Failure, StageTag


MAX_STEPS = 1 << 52
UINT64_MAX = (1 << 64) - 1


def _check_dt(stage, context):
    """Assess the remaining horizon before creating any trial state."""
    from .stability import assess_runtime, stage_frequency_estimate
    from .types import check_record

    report = assess_runtime(stage_frequency_estimate(stage, context), context,
                            step=stage.tag.step)
    context.last_dt_report = report
    samples_ratio = (
        report['minimum_samples'] / report['samples_per_cycle']
        if report['samples_per_cycle'] > 0 else math.inf
    )
    ratio = max(
        samples_ratio,
        report['amplitude_relative_error']/report['amplitude_budget'],
        report['phase_error_rad']/report['phase_budget_rad'])
    stage.checks.append(check_record(
        name='time_step_estimate', unit='1', value_unit='1', tag=stage.tag,
        raw=ratio, reference=1., threshold=1., norm='value', selected=1,
        passed=report['accepted'], hard=context.p.dt_guard == 'error',
        reason=(f"omega_estimate={report['estimate']['omega_estimate']:.17g} rad/s; "
                f"omega_dt={report['omega_dt']:.17g}; "
                f"remaining_horizon={report['horizon_s']:.17g} s; "
                + '; '.join(report['reasons']))))
    if report['action'] == 'error':
        failure = Failure('TimeStepGuard', '; '.join(report['reasons']))
        failure.stage = stage.tag
        failure.time_step_report = report
        raise failure


def _tag(context, step, generation, trial, time, stage):
    return StageTag(
        step,
        generation,
        trial,
        getattr(context, "spectral_id", 0),
        time,
        stage,
    )


def _fresh(counter):
    value = int(counter)
    if value >= UINT64_MAX:
        raise Failure("CounterOverflow", "time identity exhausted")
    return value + 1, value


def _check_clock(context, left, cumulative):
    if not getattr(left, "accepted", False):
        raise Failure("Time", "invalid advancement request")
    if left.state.step >= context.p.nsteps:
        raise Failure("Time", "invalid advancement request")
    n = left.state.step
    expected = core.time_at_step(context.p, n)
    if left.state.time != expected:
        raise Failure("Time", "accepted time does not match scheduler")
    if (
        cumulative.last_step != n
        or cumulative.last_time != left.state.time
    ):
        raise Failure("Ledger", "cumulative endpoint mismatch")
    dt = float(context.p.dt)
    if not math.isfinite(dt) or dt <= 0 or context.p.nsteps > MAX_STEPS:
        raise Failure("Time", "invalid step/dt")
    return n


def _checked_tag(state, tag):
    return (
        state.step == tag.step
        and state.generation == tag.generation
        and state.time == tag.time
    )


def _add_source(rhs, state, tag, source, context, sources):
    if source is None:
        return rhs, None, False
    if int(getattr(context.p, "electron_inertia", 0)) == 0:
        raise Failure(
            "ManufacturedSource",
            "massless independent electron density/momentum sources are "
            "unsupported until their physical semantics are reviewed",
        )
    try:
        parameters = inspect.signature(source).parameters
        payload = (
            source(state, tag, context)
            if len(parameters) >= 3
            else source(state, tag)
        )
        sources.append((tag, payload))
    except Failure as failure:
        if failure.stage is None:
            failure.stage = tag
        raise
    except Exception as error:
        failure = Failure("NumericalLibrary", str(error))
        failure.stage = tag
        raise failure from error
    try:
        diagnostics.check_source(payload, context)
        out = copy.copy(rhs)
        out.fluid = [copy.copy(fluid) for fluid in rhs.fluid]
        for a in range(2):
            out.fluid[a].n = out.fluid[a].n + payload.fluid[a].n
            out.fluid[a].mom = out.fluid[a].mom + payload.fluid[a].mom
            out.fluid[a].perp = out.fluid[a].perp + payload.fluid[a].perp
            if a:
                out.fluid[a].parallel = (
                    out.fluid[a].parallel + payload.fluid[a].parallel
                )
        out.B = out.B + payload.B
        return out, payload, True
    except Failure as failure:
        if failure.stage is None:
            failure.stage = tag
        raise
    except Exception as error:
        failure = Failure("NumericalLibrary", str(error))
        failure.stage = tag
        raise failure from error


def _combine(base, rhs, weight, tag, context):
    try:
        return core.combine(base, rhs, weight, tag, context)
    except Failure as failure:
        if failure.stage is None:
            failure.stage = tag
        raise
    except Exception as error:
        failure = Failure("NumericalLibrary", str(error))
        failure.stage = tag
        raise failure from error


def _append_missing(target, source):
    def identity(record):
        return (
            record.name,
            record.population,
            record.component,
            getattr(record, "producer", ""),
            record.region,
            str(record.norm),
            record.tag.step,
            record.tag.generation,
            record.tag.trial,
            record.tag.spectral_context,
            float(record.tag.time).hex(),
            record.tag.stage,
        )

    seen = {identity(record) for record in target}
    for record in source:
        key = identity(record)
        if key not in seen:
            target.append(record)
            seen.add(key)


def _retain(
    failure,
    context,
    evidence_stages,
    sources,
    step_result,
    fallback_tag,
):
    if failure.stage is None:
        failure.stage = fallback_tag
    for evidence in evidence_stages:
        retain_failure_evidence(
            failure,
            evidence.tag,
            context,
            evidence.state,
            evidence.stage,
            evidence.prefix,
        )
    for index, (tag, source) in enumerate(sources):
        if source is None:
            continue
        retain_failure_evidence(
            failure, tag, context, source, None, f"source_{index}"
        )
    if step_result is not None:
        _append_missing(failure.checks, step_result.checks)
        for a, suffix in enumerate(("i", "e")):
            number_flux = getattr(
                step_result.interval, "number_flux", [None, None]
            )[a]
            current = getattr(
                step_result.interval, "current", [None, None]
            )[a]
            if number_flux is not None:
                failure.partial_arrays.setdefault(
                    f"interval_number_flux_{suffix}",
                    number_flux,
                )
            if current is not None:
                failure.partial_arrays.setdefault(
                    f"interval_current_{suffix}",
                    current,
                )


def _pending_cumulative(cumulative, quadrature, endpoint, trial):
    pending = copy.deepcopy(cumulative)
    ledger_fields = (
        "electric_node",
        "electric_edge",
        "pressure_kinetic",
        "pressure_internal",
        "magnetic",
        "wall",
        "convection",
        "perp_rate",
        "parallel_rate",
        "spectral_kinetic",
        "spectral_perp",
        "spectral_parallel",
    )
    boundary_fields = (
        "number",
        "kinetic",
        "internal_perp",
        "internal_parallel",
    )
    for _, stage, weight in quadrature:
        for a in range(2):
            for name in ledger_fields:
                value = (
                    float(getattr(pending.integrals[a], name))
                    + weight * float(getattr(stage.ledger[a], name, 0.0))
                )
                if not math.isfinite(value):
                    raise Failure("Ledger", "nonfinite cumulative ledger")
                setattr(pending.integrals[a], name, value)
            for name in boundary_fields:
                value = float(getattr(pending.boundary_integrals, name)[a])
                value += weight * float(
                    getattr(stage.boundary, name, [0.0, 0.0])[a]
                )
                if not math.isfinite(value):
                    raise Failure("Ledger", "nonfinite cumulative ledger")
                getattr(pending.boundary_integrals, name)[a] = value
        for name in ("field_energy", "total_energy"):
            value = float(getattr(pending.boundary_integrals, name))
            value += weight * float(getattr(stage.boundary, name, 0.0))
            if not math.isfinite(value):
                raise Failure("Ledger", "nonfinite cumulative ledger")
            setattr(pending.boundary_integrals, name, value)
    pending.accepted_intervals = int(pending.accepted_intervals) + 1
    pending.last_trial = trial
    pending.last_step = endpoint.step
    pending.last_time = endpoint.time
    return pending


def _commit(committed, cumulative, pending, candidate, cache, result):
    committed.state = candidate
    committed.cache = cache
    for name in (
        "baseline",
        "integrals",
        "boundary_integrals",
        "accepted_intervals",
        "last_trial",
        "last_step",
        "last_time",
    ):
        setattr(cumulative, name, getattr(pending, name))
    committed.accepted = True
    result.accepted = True
    result.interval.accepted = True


def _advance(committed, cumulative, context, source, integrator):
    n = _check_clock(context, committed, cumulative)
    # Parent-owned dt guard hook: assess the resolved physical spectrum here
    # before allocating or evaluating any trial stage.
    dt = float(context.p.dt)
    t0 = core.time_at_step(context.p, n)
    tm = core.time_at_midpoint(context.p, n)
    t1 = core.time_at_step(context.p, n + 1)
    if not (t0 < tm < t1):
        raise Failure("TimeResolution", "midpoint not representable")
    next_trial, trial = _fresh(context.next_trial)
    context.next_trial = next_trial
    next_generation, generation = _fresh(context.next_generation)
    context.next_generation = next_generation
    base_tag = _tag(
        context, n, committed.state.generation, trial, t0, "base"
    )
    base = midpoint_state = midpoint = candidate = endpoint = None
    k2 = k3 = k4 = None
    sources = []
    evidence_stages = []
    step_result = None
    if integrator == "midpoint":
        next_generation, mid_generation = _fresh(context.next_generation)
        context.next_generation = next_generation
        midpoint_tag = _tag(
            context, n, mid_generation, trial, tm, "midpoint"
        )
        next_generation, end_generation = _fresh(context.next_generation)
        context.next_generation = next_generation
        endpoint_tag = _tag(
            context, n + 1, end_generation, trial, t1, "endpoint"
        )
        scheduled = (
            ("base", base_tag),
            ("midpoint", midpoint_tag),
            ("endpoint", endpoint_tag),
        )
    else:
        tags = []
        for stage, stage_time in (
            ("midpoint", tm),
            ("midpoint_2", tm),
            ("station", t1),
        ):
            next_generation, stage_generation = _fresh(
                context.next_generation
            )
            context.next_generation = next_generation
            tags.append(
                _tag(context, n, stage_generation, trial, stage_time, stage)
            )
        k2_tag, k3_tag, k4_tag = tags
        next_generation, end_generation = _fresh(context.next_generation)
        context.next_generation = next_generation
        endpoint_tag = _tag(
            context, n + 1, end_generation, trial, t1, "endpoint"
        )
        scheduled = (
            ("base", base_tag),
            ("midpoint", k2_tag),
            ("midpoint_2", k3_tag),
            ("station", k4_tag),
            ("endpoint", endpoint_tag),
        )
    evidence_stages = [
        SimpleNamespace(prefix=prefix, tag=tag, state=None, stage=None)
        for prefix, tag in scheduled
    ]

    def populate_evidence(prefix, state, stage):
        for evidence in evidence_stages:
            if evidence.prefix == prefix and evidence.tag == stage.tag:
                evidence.state = state
                evidence.stage = stage
                return
        raise Failure("Time", f"missing scheduled evidence slot: {prefix}")

    try:
        base = evaluate_stage(committed.state, base_tag, context, False)
        populate_evidence("base", committed.state, base)
        _check_dt(base, context)
        if integrator == "midpoint":
            rhs, source_payload, _ = _add_source(
                base.rhs, committed.state, base.tag, source, context, sources
            )
            midpoint_state = _combine(
                committed.state, rhs, dt / 2, midpoint_tag, context
            )
            midpoint = evaluate_stage(midpoint_state, midpoint_tag, context, True)
            populate_evidence("midpoint", midpoint_state, midpoint)
            rhs, source_payload, _ = _add_source(
                midpoint.rhs,
                midpoint_state,
                midpoint.tag,
                source,
                context,
                sources,
            )
            candidate = _combine(
                committed.state, rhs, dt, endpoint_tag, context
            )
            endpoint = evaluate_stage(candidate, endpoint_tag, context, False)
            populate_evidence("endpoint", candidate, endpoint)
            quadrature = [(midpoint_state, midpoint, dt)]
        elif integrator == "rk4":
            rhs1, source1, _ = _add_source(
                base.rhs, committed.state, base.tag, source, context, sources
            )
            y2 = _combine(committed.state, rhs1, dt / 2, k2_tag, context)
            k2 = evaluate_stage(y2, k2_tag, context, True)
            populate_evidence("midpoint", y2, k2)
            rhs2, source2, _ = _add_source(
                k2.rhs, y2, k2.tag, source, context, sources
            )
            y3 = _combine(committed.state, rhs2, dt / 2, k3_tag, context)
            k3 = evaluate_stage(y3, k3_tag, context, True)
            populate_evidence("midpoint_2", y3, k3)
            rhs3, source3, _ = _add_source(
                k3.rhs, y3, k3.tag, source, context, sources
            )
            y4 = _combine(committed.state, rhs3, dt, k4_tag, context)
            k4 = evaluate_stage(y4, k4_tag, context, False)
            populate_evidence("station", y4, k4)
            rhs4, source4, _ = _add_source(
                k4.rhs, y4, k4.tag, source, context, sources
            )
            # RK4 weights are applied to the four derivative stages. The final
            # endpoint/station evaluation is validation, not an extra stage.
            combined = copy.copy(base.rhs)
            combined.fluid = [copy.copy(f) for f in base.rhs.fluid]
            for a in range(2):
                combined.fluid[a].n = (
                    rhs1.fluid[a].n
                    + 2 * rhs2.fluid[a].n
                    + 2 * rhs3.fluid[a].n
                    + rhs4.fluid[a].n
                ) / 6
                combined.fluid[a].mom = (
                    rhs1.fluid[a].mom
                    + 2 * rhs2.fluid[a].mom
                    + 2 * rhs3.fluid[a].mom
                    + rhs4.fluid[a].mom
                ) / 6
                combined.fluid[a].perp = (
                    rhs1.fluid[a].perp
                    + 2 * rhs2.fluid[a].perp
                    + 2 * rhs3.fluid[a].perp
                    + rhs4.fluid[a].perp
                ) / 6
                if a:
                    combined.fluid[a].parallel = (
                        rhs1.fluid[a].parallel
                        + 2 * rhs2.fluid[a].parallel
                        + 2 * rhs3.fluid[a].parallel
                        + rhs4.fluid[a].parallel
                    ) / 6
            combined.B = (
                rhs1.B + 2 * rhs2.B + 2 * rhs3.B + rhs4.B
            ) / 6
            candidate = _combine(
                committed.state, combined, dt, endpoint_tag, context
            )
            endpoint = evaluate_stage(candidate, endpoint_tag, context, False)
            populate_evidence("endpoint", candidate, endpoint)
            quadrature = [
                (committed.state, base, dt / 6),
                (y2, k2, dt / 3),
                (y3, k3, dt / 3),
                (y4, k4, dt / 6),
            ]
            midpoint_state, midpoint = y2, k2
        else:
            raise Failure("Config", f"unknown time integrator: {integrator}")

        source_iter = None
        if source is not None:
            by_tag = {
                (
                    tag.step,
                    tag.generation,
                    tag.trial,
                    tag.spectral_context,
                    tag.time,
                    tag.stage,
                ): payload
                for tag, payload in sources
            }
            source_iter = iter(
                by_tag[
                    (
                        stage.tag.step,
                        stage.tag.generation,
                        stage.tag.trial,
                        stage.tag.spectral_context,
                        stage.tag.time,
                        stage.tag.stage,
                    )
                ]
                for _, stage, _ in quadrature
            )
        step_result = diagnostics.check_step(
            committed.state,
            base,
            midpoint_state,
            midpoint,
            candidate,
            endpoint,
            cumulative,
            context,
            source_iter,
            quadrature=quadrature,
        )
        checks = list(base.checks) + list(midpoint.checks)
        if integrator == "rk4":
            checks += list(k3.checks) + list(k4.checks)
        checks += list(endpoint.checks) + list(step_result.checks)
        step_result.checks = checks
        diagnostics.gate(checks)
        if (
            getattr(cumulative, "accepted_intervals", 0)
            and getattr(cumulative, "last_trial", None) == trial
        ):
            raise Failure("DuplicateInterval", "interval already accumulated")
        pending = _pending_cumulative(
            cumulative, quadrature, candidate, trial
        )
        context.compute.synchronize()
        result = step_result
        _commit(
            committed,
            cumulative,
            pending,
            candidate,
            endpoint.cache,
            result,
        )
        result.frame = endpoint
        result.time_step_report = getattr(context, 'last_dt_report', None)
        committed.frame = endpoint
        return result
    except Failure as failure:
        failure.time_step_report = getattr(context, 'last_dt_report', None)
        _retain(
            failure,
            context,
            evidence_stages,
            sources,
            step_result,
            locals().get("endpoint_tag", base_tag),
        )
        raise
    except Exception as error:
        failure = Failure("NumericalLibrary", str(error))
        failure.stage = getattr(endpoint, "tag", None) if endpoint else base_tag
        _retain(
            failure,
            context,
            evidence_stages,
            sources,
            step_result,
            locals().get("endpoint_tag", base_tag),
        )
        raise failure from error


def advance(committed, cumulative, context, source=None):
    """Advance one accepted step; the default and production path is RK4."""
    inertia = int(getattr(context.p, "electron_inertia", 0))
    if inertia not in (0, 1):
        raise Failure("Config", f"unknown electron_inertia: {inertia}")
    integrator = getattr(context.p, "integrator", "rk4")
    if integrator not in ("midpoint", "rk4"):
        raise Failure("Config", f"unknown time integrator: {integrator}")
    return _advance(committed, cumulative, context, source, integrator)
