"""Physical input fixtures for the bounded WR benchmark.

The periodic state and field construction is extracted only from the physical
builder in ``tools/warp_t2_field.py`` (and its backing
``tests/warp/massless_verification_fixture.py``).  It contains no T2-E solver,
C1/C2 scheduler, or propagation approximation.  The closed-wall helper uses
the existing real K=1 rectangle operator/equilibrium fixture solely as a
bounded field and split smoke case.
"""

from __future__ import annotations

import math

from massless_verification_fixture import (
    HallCase,
    build_periodic_context,
    uniform_state,
)


def periodic_inputs(
    points: int,
    amplitude: float,
    nonuniform: float,
    *,
    seed_mode: int,
    sentinel: float,
    device: str = "cpu",
):
    """Build the prior periodic physical seed and independent high-mode seed.

    Source origin:
    ``tools/warp_t2_field.py::periodic_inputs`` for the density, pressure,
    velocity, and magnetic base seed.  The high-mode addition follows the
    established ``points // 2`` control with a coefficient relative to the
    base amplitude.
    """

    if type(points) is not int or points < 9 or points % 2 == 0:
        raise ValueError("points must be an odd integer >= 9")
    if type(seed_mode) is not int or not 1 <= seed_mode <= points // 2:
        raise ValueError("seed_mode must be an integer in [1, points//2]")
    for value, name in (
        (amplitude, "amplitude"),
        (nonuniform, "nonuniform"),
        (sentinel, "sentinel"),
    ):
        if not math.isfinite(float(value)) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")

    case = HallCase(points=points)
    context = build_periodic_context(case, device=device)
    xp = context.compute.xp
    state = uniform_state(context, case)
    phase = xp.arange(points, dtype=xp.float64) * 2.0 * math.pi / points
    density = case.n0 * (1.0 + nonuniform * xp.cos(phase))
    state.fluid[0].n[:] = density
    state.fluid[1].n[:] = density

    carrier = seed_mode * phase
    velocity = xp.stack(
        (
            amplitude * case.v_alfven * xp.cos(carrier),
            0.7 * amplitude * case.v_alfven * xp.sin(carrier),
            0.03 * nonuniform * case.v_alfven * xp.sin(phase),
        )
    )
    state.fluid[0].mom[:] = (
        velocity * density * float(context.p.species[0].mass)
    ).reshape(-1)
    state.fluid[1].perp[:] = case.pressure * (
        1.0 + 0.2 * nonuniform * xp.cos(phase)
    )
    state.fluid[1].parallel[:] = case.pressure * (
        1.0 - 0.15 * nonuniform * xp.cos(phase)
    )
    state.B[:points] = amplitude * case.B0 * xp.cos(carrier)
    state.B[points : 2 * points] = (
        -amplitude * case.B0 * xp.sin(carrier)
    )

    high_mode = points // 2
    if sentinel > 0.0:
        high_phase = high_mode * phase
        high_velocity = (
            amplitude
            * sentinel
            * case.v_alfven
            * xp.stack(
                (
                    xp.cos(high_phase),
                    0.5 * xp.sin(high_phase),
                    xp.zeros_like(high_phase),
                )
            )
        )
        state.fluid[0].mom += (
            density[None, :]
            * high_velocity
            * float(context.p.species[0].mass)
        ).reshape(-1)
        state.B[:points] += (
            amplitude * sentinel * case.B0 * xp.cos(high_phase)
        )
        state.B[points : 2 * points] += (
            -amplitude * sentinel * case.B0 * xp.sin(high_phase)
        )

    metadata = {
        "case": case,
        "target_mode": seed_mode,
        "sentinel_mode": high_mode,
        "sentinel_retained": high_mode <= points // 2,
        "physical_source_origin": (
            "tools/warp_t2_field.py::periodic_inputs and "
            "tests/warp/massless_verification_fixture.py"
        ),
    }
    return context, state, metadata


def closed_wall_inputs(*, nodes_r: int = 4, nodes_z: int = 4):
    """Build one real K=1 closed-wall equilibrium state for smoke tests."""

    from test_massless_support import context, uniform_state as wall_state

    value = context(K=1, hard_diagnostics=False, nodes_r=nodes_r, nodes_z=nodes_z)
    value.p.memory_limit = value.p.device_memory_limit = 1 << 30
    state = wall_state(value, compatible=True)
    return value, state, {
        "case": "closed-wall",
        "K": int(value.p.K),
        "Nphi": int(value.N),
        "cells": int(value.op.Nc),
        "physical_source_origin": (
            "tests/warp/test_massless_support.py real rectangle context"
        ),
    }
