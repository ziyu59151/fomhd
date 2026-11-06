"""Read-only energy partitions and optional balance in SI units."""

from __future__ import annotations

import math
import operator


def _finite(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _model_flag(value):
    if isinstance(value, bool):
        raise ValueError("electron_inertia must be 0 or 1")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError("electron_inertia must be 0 or 1") from error
    if result not in (0, 1):
        raise ValueError("electron_inertia must be 0 or 1")
    return result


def summarize_energy(
    totals,
    electron_inertia,
    *,
    baseline_total_J=None,
    outward_J=0.0,
    source_J=0.0,
):
    """Return a new dictionary of SI energy partitions and signed balance terms."""
    inertia = _model_flag(electron_inertia)
    try:
        if len(totals.kinetic) != 2:
            raise ValueError("kinetic must contain exactly two scalars")
    except TypeError as error:
        raise ValueError("kinetic must contain exactly two scalars") from error

    ion_kinetic = _finite(totals.kinetic[0], "kinetic[0]")
    electron_kinetic = _finite(totals.kinetic[1], "kinetic[1]")
    ion_internal = _finite(totals.ion_internal, "ion_internal")
    electron_perp = _finite(totals.electron_perp, "electron_perp")
    electron_parallel = _finite(totals.electron_parallel, "electron_parallel")
    magnetic = _finite(totals.magnetic, "magnetic")
    longitudinal = _finite(totals.longitudinal, "longitudinal")
    reported_total = _finite(totals.energy, "energy")

    if baseline_total_J is None:
        baseline = None
    else:
        baseline = _finite(baseline_total_J, "baseline_total_J")
    outward = _finite(outward_J, "outward_J")
    source = _finite(source_J, "source_J")

    if inertia == 0:
        if electron_kinetic != 0.0:
            raise ValueError(
                "kinetic[1] must be zero when electron_inertia is 0"
            )
        if longitudinal != 0.0:
            raise ValueError(
                "longitudinal must be zero when electron_inertia is 0"
            )
        electron = electron_perp + electron_parallel
        field = magnetic
        electron_kinetic_label = "not_applicable"
        longitudinal_label = "not_applicable"
    else:
        electron = electron_kinetic + electron_perp + electron_parallel
        field = magnetic + longitudinal
        electron_kinetic_label = "applicable"
        longitudinal_label = "applicable"

    ion = ion_kinetic + ion_internal
    total = ion + electron + field
    partition_difference = total - reported_total
    ion = _finite(ion, "ion_J")
    electron = _finite(electron, "electron_J")
    field = _finite(field, "field_J")
    total = _finite(total, "total_J")
    partition_difference = _finite(
        partition_difference, "partition_difference_J"
    )

    if baseline is None:
        balance = None
        relative_change = None
    else:
        balance = _finite(
            total - baseline + outward - source,
            "balance_J",
        )
        if baseline == 0.0:
            relative_change = None
        else:
            relative_change = _finite(
                (total - baseline) / baseline,
                "relative_total_change",
            )

    return {
        "ion_J": ion,
        "electron_J": electron,
        "field_J": field,
        "total_J": total,
        "reported_total_J": reported_total,
        "partition_difference_J": partition_difference,
        "baseline_total_J": baseline,
        "balance_J": balance,
        "relative_total_change": relative_change,
        "outward_J": outward,
        "source_J": source,
        "electron_kinetic_energy": electron_kinetic_label,
        "longitudinal_electric_energy": longitudinal_label,
    }
