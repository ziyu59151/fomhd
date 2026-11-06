"""A1-T0 fixed-tangent ion path with RK4 electron-field microsteps."""

from __future__ import annotations

import math
from types import SimpleNamespace

from . import core, diagnostics, stability
from . import time as time_module
from .massless import canonicalize_state
from .stage import evaluate_stage, retain_failure_evidence
from .types import Failure, FluidState, StageTag, State, check_record


_RK_WEIGHTS = (1.0, 2.0, 2.0, 1.0)


def _tag(
    context,
    step,
    generation,
    trial,
    time,
    stage,
    micro_step,
):
    return StageTag(
        step=step,
        generation=generation,
        trial=trial,
        spectral_context=getattr(context, "spectral_id", 0),
        time=time,
        stage=stage,
        micro_step=micro_step,
        iteration=0,
        slab=0,
    )


def _fresh(counter):
    value = int(counter)
    if value >= time_module.UINT64_MAX:
        raise Failure("CounterOverflow", "time identity exhausted")
    return value + 1, value


def _parts(state):
    return (
        (
            state.fluid[0].n,
            state.fluid[0].mom,
            state.fluid[0].perp,
        ),
        (
            state.fluid[1].perp,
            state.fluid[1].parallel,
            state.B,
        ),
    )


def _assemble_state(context, ion, electron, tag):
    """Build and canonicalize a full massless state from independent P and Z."""
    xp = context.compute.xp
    n, mom, perp = ion
    p_perp, p_parallel, magnetic = electron
    fluid = (
        FluidState(
            n=xp.asarray(n, dtype=xp.complex128).copy(),
            mom=xp.asarray(mom, dtype=xp.complex128).copy(),
            perp=xp.asarray(perp, dtype=xp.complex128).copy(),
            parallel=xp.empty(0, dtype=xp.complex128),
        ),
        FluidState(
            n=xp.zeros_like(xp.asarray(n)),
            mom=xp.zeros_like(xp.asarray(mom)),
            perp=xp.asarray(p_perp, dtype=xp.complex128).copy(),
            parallel=xp.asarray(p_parallel, dtype=xp.complex128).copy(),
        ),
    )
    raw = State(
        fluid=fluid,
        B=xp.asarray(magnetic, dtype=xp.complex128).copy(),
        step=tag.step,
        generation=tag.generation,
        time=tag.time,
    )
    try:
        canonical = canonicalize_state(raw, context)
    except Failure as failure:
        if failure.stage is None:
            failure.stage = tag
        retain_failure_evidence(
            failure,
            tag,
            context,
            raw,
            None,
            "candidate",
        )
        raise
    except Exception as error:
        failure = Failure("NumericalLibrary", str(error))
        failure.stage = tag
        retain_failure_evidence(
            failure,
            tag,
            context,
            raw,
            None,
            "candidate",
        )
        raise failure from error
    canonical.step = tag.step
    canonical.generation = tag.generation
    canonical.time = tag.time
    return canonical


def _linear_combination(context, base, terms):
    """Return base + sum(weight*increment) using the selected backend."""
    out = context.compute.array(context.compute.xp.asarray(base).copy())
    for weight, increment in terms:
        out = context.compute.axpy(out, increment, weight)
    return out


def _tangent(context, initial, derivative, offset):
    return tuple(
        _linear_combination(context, base, [(offset, increment)])
        for base, increment in zip(initial, derivative)
    )


def _rk_stage_state(context, z0, derivatives, stage_index, h):
    if stage_index == 0:
        return z0
    if stage_index == 1:
        terms = [(h / 2.0, derivatives[0])]
    elif stage_index == 2:
        terms = [(h / 2.0, derivatives[1])]
    elif stage_index == 3:
        terms = [(h, derivatives[2])]
    else:
        raise Failure("Time", "invalid RK4 stage index")
    return tuple(
        _linear_combination(
            context,
            z0[index],
            [(weight, derivative[index]) for weight, derivative in terms],
        )
        for index in range(3)
    )


def _rk_next(context, z0, derivatives, h):
    return tuple(
        _linear_combination(
            context,
            z0[index],
            [
                (h * weight / 6.0, derivative[index])
                for weight, derivative in zip(_RK_WEIGHTS, derivatives)
            ],
        )
        for index in range(3)
    )


def _add_ion_increment(context, ion, derivative, weight):
    return tuple(
        context.compute.axpy(value, increment, weight)
        for value, increment in zip(ion, derivative)
    )


def _maximum(context, values):
    if values.size == 0:
        return 0.0
    maximum = getattr(context.compute, "max_abs", None)
    value = maximum(values) if maximum is not None else context.compute.xp.max(
        context.compute.xp.abs(values)
    )
    return float(value)


def _correction_metadata(context, initials, tangent_end, accepted_end):
    references = (
        float(context.p.ref.ni),
        float(context.p.species[0].mass)
        * float(context.p.ref.ni)
        * float(context.p.ref.velocity),
        float(context.p.ref.pressure),
    )
    out = {}
    for name, initial, tangent, accepted, reference in zip(
        ("n_i", "mom_i", "perp_i"),
        initials,
        tangent_end,
        accepted_end,
        references,
    ):
        correction = context.compute.xp.asarray(accepted) - context.compute.xp.asarray(
            tangent
        )
        raw = _maximum(context, correction)
        scale = max(
            _maximum(context, tangent),
            float(reference),
            1.0e-300,
        )
        relative = raw / scale
        if not math.isfinite(relative):
            raise Failure("Time", "nonfinite endpoint tangent correction")
        out[name] = {
            "raw_max_abs": raw,
            "scale": scale,
            "relative_max_abs": relative,
        }
    return out


def _check_guard(stage, context):
    """Apply the existing budgets to RK4 microsteps only."""
    estimate = stability.stage_frequency_estimate(stage, context)
    report = stability.assess_predictor_runtime(
        estimate, context, step=stage.tag.step
    )
    context.last_dt_report = report
    samples_ratio = (
        report["minimum_samples"] / report["samples_per_cycle"]
        if report["samples_per_cycle"] > 0.0
        else math.inf
    )
    ratio = max(
        samples_ratio,
        report["amplitude_relative_error"] / report["amplitude_budget"],
        report["phase_error_rad"] / report["phase_budget_rad"],
    )
    stage.checks.append(
        check_record(
            name="time_step_estimate",
            unit="1",
            value_unit="1",
            tag=stage.tag,
            raw=ratio,
            reference=1.0,
            threshold=1.0,
            norm="value",
            selected=1,
            passed=report["accepted"],
            hard=context.p.dt_guard == "error",
            reason=(
                f"omega_estimate={estimate['omega_estimate']:.17g} rad/s; "
                f"micro_omega_dt={report['omega_dt']:.17g}; "
                f"macro_remaining_horizon={report['horizon_s']:.17g} s; "
                f"scope={report['scope']}; "
                + "; ".join(report["reasons"])
            ),
        )
    )
    if report["action"] == "error":
        failure = Failure(
            "TimeStepGuard",
            "; ".join(report["reasons"]),
        )
        failure.stage = stage.tag
        failure.time_step_report = report
        raise failure


def _retain(failure, context, evidence_stages, step_result, fallback_tag):
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
    if step_result is not None:
        time_module._append_missing(failure.checks, step_result.checks)
        for species, suffix in enumerate(("i", "e")):
            number_flux = getattr(
                step_result.interval, "number_flux", [None, None]
            )[species]
            current = getattr(
                step_result.interval, "current", [None, None]
            )[species]
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


def _validate_clocks(p, macro_step):
    micro_steps = p.micro_steps
    for micro in range(micro_steps):
        stage_times = [
            core.microstage_time(p, macro_step, micro, stage)
            for stage in range(4)
        ]
        for left, right in zip(stage_times, stage_times[1:]):
            if left > right:
                raise Failure(
                    "TimeResolution",
                    "predictor microstage clocks are not ordered",
                )
        if not (
            stage_times[0]
            < stage_times[1]
            <= stage_times[2]
            < stage_times[3]
        ):
            raise Failure(
                "TimeResolution",
                "predictor microstage clocks are not representable",
            )


def advance(committed, cumulative, context):
    """Advance one accepted macro interval with the selected T0 predictor."""
    coupling, micro_steps = time_module.validate_time_controls(context.p)
    if coupling != "predictor":
        raise Failure("Config", "predictor scheduler requires predictor coupling")
    n = time_module._check_clock(context, committed, cumulative)
    peak_bytes = core.preflight_memory(context)
    _validate_clocks(context.p, n)
    dt = float(context.p.dt)
    h = dt / micro_steps
    if not math.isfinite(h) or h <= 0.0:
        raise Failure("TimeResolution", "predictor microstep is not representable")
    t0 = core.time_at_step(context.p, n)
    t1 = core.time_at_step(context.p, n + 1)

    next_trial, trial = _fresh(context.next_trial)
    context.next_trial = next_trial
    base_tag = _tag(
        context,
        n,
        committed.state.generation,
        trial,
        t0,
        "base",
        0,
    )
    next_generation, generation = _fresh(context.next_generation)
    context.next_generation = next_generation
    end_tag = _tag(
        context,
        n + 1,
        generation,
        trial,
        t1,
        "endpoint",
        -1,
    )

    scheduled = [("base", base_tag)]
    for micro in range(micro_steps):
        for stage_index in range(4):
            if micro == 0 and stage_index == 0:
                continue
            prefix = f"predictor_m{micro:03d}_r{stage_index + 1}"
            tag_time = core.microstage_time(
                context.p, n, micro, stage_index
            )
            next_generation, stage_generation = _fresh(
                context.next_generation
            )
            context.next_generation = next_generation
            scheduled.append(
                (
                    prefix,
                    _tag(
                        context,
                        n,
                        stage_generation,
                        trial,
                        tag_time,
                        prefix,
                        micro,
                    ),
                )
            )
    scheduled.append(("endpoint", end_tag))
    evidence = [
        SimpleNamespace(prefix=prefix, tag=tag, state=None, stage=None)
        for prefix, tag in scheduled
    ]
    evidence_by_prefix = {item.prefix: item for item in evidence}

    def populate(prefix, state, stage):
        evidence_stage = evidence_by_prefix[prefix]
        if evidence_stage.tag != stage.tag:
            raise Failure("Time", f"stale scheduled predictor evidence: {prefix}")
        evidence_stage.state = state
        evidence_stage.stage = stage

    base = endpoint = candidate = None
    step_result = None
    quadrature = []
    evaluated_stages = []
    alias_evaluations = 0
    try:
        base = evaluate_stage(
            committed.state,
            base_tag,
            context,
            False,
        )
        populate("base", committed.state, base)
        _check_guard(base, context)
        initial_ion, initial_z = _parts(committed.state)
        f0 = (
            base.rhs.fluid[0].n,
            base.rhs.fluid[0].mom,
            base.rhs.fluid[0].perp,
        )
        accepted_ion = initial_ion
        z_current = initial_z

        for micro in range(micro_steps):
            derivatives = []
            micro_start = initial_ion
            for stage_index in range(4):
                weight = h * _RK_WEIGHTS[stage_index] / 6.0
                if micro == 0 and stage_index == 0:
                    state = committed.state
                    stage = base
                else:
                    prefix = f"predictor_m{micro:03d}_r{stage_index + 1}"
                    tag = evidence_by_prefix[prefix].tag
                    z_stage = _rk_stage_state(
                        context,
                        z_current,
                        derivatives,
                        stage_index,
                        h,
                    )
                    offset = core.microstage_offset(
                        context.p, micro, stage_index
                    )
                    tangent = _tangent(
                        context,
                        micro_start,
                        f0,
                        offset,
                    )
                    state = _assemble_state(
                        context,
                        tangent,
                        z_stage,
                        tag,
                    )
                    alias = stage_index in (1, 2)
                    alias_evaluations += 1 if alias else 0
                    stage = evaluate_stage(
                        state,
                        tag,
                        context,
                        alias,
                    )
                    populate(prefix, state, stage)
                derivative_ion = (
                    stage.rhs.fluid[0].n,
                    stage.rhs.fluid[0].mom,
                    stage.rhs.fluid[0].perp,
                )
                derivative_z = (
                    stage.rhs.fluid[1].perp,
                    stage.rhs.fluid[1].parallel,
                    stage.rhs.B,
                )
                derivatives.append(derivative_z)
                quadrature.append((state, stage, weight))
                evaluated_stages.append(stage)
                accepted_ion = _add_ion_increment(
                    context,
                    accepted_ion,
                    derivative_ion,
                    weight,
                )
            z_current = _rk_next(context, z_current, derivatives, h)

        candidate = _assemble_state(
            context,
            accepted_ion,
            z_current,
            end_tag,
        )
        endpoint = evaluate_stage(
            candidate,
            end_tag,
            context,
            False,
        )
        populate("endpoint", candidate, endpoint)

        first_state, first_stage, _ = quadrature[0]
        step_result = diagnostics.check_step(
            committed.state,
            base,
            first_state,
            first_stage,
            candidate,
            endpoint,
            cumulative,
            context,
            None,
            quadrature=quadrature,
        )
        checks = list(base.checks)
        for stage in evaluated_stages[1:]:
            checks.extend(stage.checks)
        checks.extend(endpoint.checks)
        checks.extend(step_result.checks)
        step_result.checks = checks
        diagnostics.gate(checks)
        if (
            getattr(cumulative, "accepted_intervals", 0)
            and getattr(cumulative, "last_trial", None) == trial
        ):
            raise Failure("DuplicateInterval", "interval already accumulated")
        pending = time_module._pending_cumulative(
            cumulative,
            quadrature,
            candidate,
            trial,
        )

        tangent_end = _tangent(context, initial_ion, f0, dt)
        correction = _correction_metadata(
            context,
            initial_ion,
            tangent_end,
            accepted_ion,
        )
        metadata = {
            "coupling": "predictor",
            "micro_steps": micro_steps,
            "macro_dt": dt,
            "micro_dt": h,
            "quadrature_weights": [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
            "quadrature_stage_evaluations": 4 * micro_steps,
            "full_rhs_stage_evaluations": 4 * micro_steps,
            "endpoint_full_evaluations": 1,
            "complete_stage_evaluations": 4 * micro_steps + 1,
            "alias_evaluations": alias_evaluations,
            "total_model_evaluations": 6 * micro_steps + 1,
            "assemble_canonicalization_calls": 4 * micro_steps,
            "retained_quadrature_entries": len(quadrature),
            "endpoint_tangent_correction": correction,
            "estimated_peak_bytes": int(peak_bytes or 0),
            "guard_scope": (
                "micro-RK4 only; does not prove macro-coupling stability"
            ),
        }
        context.compute.synchronize()
        time_module._commit(
            committed,
            cumulative,
            pending,
            candidate,
            endpoint.cache,
            step_result,
        )
        step_result.frame = endpoint
        step_result.time_step_report = getattr(context, "last_dt_report", None)
        step_result.coupling_metadata = metadata
        step_result.predictor_metadata = metadata
        committed.frame = endpoint
        return step_result
    except Failure as failure:
        failure.time_step_report = getattr(context, "last_dt_report", None)
        _retain(
            failure,
            context,
            evidence,
            step_result,
            endpoint.tag if endpoint is not None else end_tag,
        )
        raise
    except Exception as error:
        failure = Failure("NumericalLibrary", str(error))
        failure.stage = (
            endpoint.tag
            if endpoint is not None
            else base_tag
        )
        _retain(
            failure,
            context,
            evidence,
            step_result,
            end_tag,
        )
        raise failure from error
