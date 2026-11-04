"""Regression tests for formal review round 2 wall-stress signs and scaling."""

from __future__ import annotations

import numpy as np

import fomhd.massless as massless

from test_massless_support import (
    context,
    equilibrium_state,
    stage_tag,
    uniform_state,
)


def _independent_traction(stage, context):
    samples = max(context.N, 3)
    wall_count = len(context.geometry.walls)
    outward = np.zeros(3)
    absolute_integral = 0.0
    for q in range(samples):
        sample = 0 if context.N == 1 else q
        angle = 2.0 * np.pi * q / samples
        for wall_index, wall in enumerate(context.geometry.walls):
            trace = stage.boundary.traces[sample * wall_count + wall_index]
            direction = 0 if wall.component == "r_face" else 2
            normal_vector = np.zeros(3)
            normal_vector[direction] = wall.sign
            b = np.asarray(trace.B)
            b2 = float(np.dot(b, b))
            bhat = b / np.sqrt(b2)
            velocity = np.asarray(trace.velocity[0])
            stress = (
                trace.perp[0] * np.eye(3)
                + trace.perp[1] * np.eye(3)
                + (trace.parallel[1] - trace.perp[1])
                * np.outer(bhat, bhat)
                + context.p.species[0].mass
                * trace.density[0]
                * np.outer(velocity, velocity)
                + 0.5 * b2 / context.p.mu0 * np.eye(3)
                - np.outer(b, b) / context.p.mu0
            )
            traction = stress @ normal_vector
            weight = 2.0 * np.pi * wall.area_per_radian / samples
            cartesian = np.array(
                [
                    traction[0] * np.cos(angle)
                    - traction[1] * np.sin(angle),
                    traction[0] * np.sin(angle)
                    + traction[1] * np.cos(angle),
                    traction[2],
                ]
            )
            outward += weight * cartesian
            absolute_integral += weight * np.linalg.norm(cartesian)
    return outward, absolute_integral


def test_oblique_anisotropic_pressure_uses_physical_b_and_bhat_stress():
    c = context(hard_diagnostics=False, nodes_r=6, nodes_z=6)
    state = uniform_state(c)
    state.fluid[1].perp[:] *= 0.7
    state.fluid[1].parallel[:] *= 1.8
    stage = massless.evaluate_stage(state, stage_tag(state), c)
    traces = stage.boundary.traces[: len(c.geometry.walls)]
    assert any(
        np.linalg.norm(trace.B) != 1.0
        and np.count_nonzero(np.asarray(trace.B)) >= 2
        for trace in traces
    )
    expected, scale = _independent_traction(stage, c)
    actual = np.asarray(stage.momentum.outward)
    assert np.max(np.abs(actual - expected)) <= (
        2.0e-11 * max(scale, 1.0e-300)
    )


def test_current_free_equilibrium_opposite_wall_tractions_cancel():
    c = context(hard_diagnostics=False, nodes_r=6, nodes_z=6)
    state = equilibrium_state(c)
    stage = massless.evaluate_stage(state, stage_tag(state), c)
    expected, scale = _independent_traction(stage, c)
    actual = np.asarray(stage.momentum.outward)
    roundoff = 8192.0 * np.finfo(np.float64).eps * max(scale, 1.0e-300)
    assert np.max(np.abs(actual - expected)) <= roundoff
    assert np.max(np.abs(actual)) <= roundoff
    for name in (
        "momentum_balance_x",
        "momentum_balance_y",
        "momentum_balance_z",
    ):
        record = next(
            item
            for item in stage.checks
            if item.name == name
            and item.region == "all"
            and str(item.norm) == "value"
        )
        assert record.unit == "kg*m/s^2"
