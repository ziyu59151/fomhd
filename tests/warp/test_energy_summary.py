"""Contract tests for the read-only A0 energy summary."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest

from fomhd.energy import summarize_energy


def _totals(**changes):
    values = {
        "kinetic": [1.0, 2.0],
        "ion_internal": 3.0,
        "electron_perp": 4.0,
        "electron_parallel": 5.0,
        "magnetic": 6.0,
        "longitudinal": 7.0,
        "energy": 28.0,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_finite_mass_partitions_retain_reported_total_and_do_not_mutate():
    totals = _totals(
        kinetic=[3.0, 5.0],
        ion_internal=11.0,
        electron_perp=13.0,
        electron_parallel=17.0,
        magnetic=19.0,
        longitudinal=23.0,
        energy=90.0,
    )
    before = copy.deepcopy(totals)

    result = summarize_energy(
        totals,
        1,
        baseline_total_J=81.0,
        outward_J=7.5,
        source_J=2.5,
    )

    assert type(result) is dict
    assert set(result) == {
        "ion_J",
        "electron_J",
        "field_J",
        "total_J",
        "reported_total_J",
        "partition_difference_J",
        "baseline_total_J",
        "balance_J",
        "relative_total_change",
        "outward_J",
        "source_J",
        "electron_kinetic_energy",
        "longitudinal_electric_energy",
    }
    assert result["ion_J"] == 14.0
    assert result["electron_J"] == 35.0
    assert result["field_J"] == 42.0
    assert result["total_J"] == 91.0
    assert result["reported_total_J"] == 90.0
    assert result["partition_difference_J"] == 1.0
    assert result["baseline_total_J"] == 81.0
    assert result["balance_J"] == 15.0
    assert result["relative_total_change"] == pytest.approx(10.0 / 81.0)
    assert result["outward_J"] == 7.5
    assert result["source_J"] == 2.5
    assert result["electron_kinetic_energy"] == "applicable"
    assert result["longitudinal_electric_energy"] == "applicable"
    assert totals == before


def test_massless_partitions_omit_unavailable_channels():
    totals = _totals(
        kinetic=[4.5, 0.0],
        ion_internal=10.5,
        electron_perp=12.0,
        electron_parallel=6.0,
        magnetic=8.0,
        longitudinal=0.0,
        energy=40.0,
    )

    result = summarize_energy(totals, 0)

    assert result["ion_J"] == 15.0
    assert result["electron_J"] == 18.0
    assert result["field_J"] == 8.0
    assert result["total_J"] == 41.0
    assert result["reported_total_J"] == 40.0
    assert result["partition_difference_J"] == 1.0
    assert result["baseline_total_J"] is None
    assert result["balance_J"] is None
    assert result["relative_total_change"] is None
    assert result["electron_kinetic_energy"] == "not_applicable"
    assert result["longitudinal_electric_energy"] == "not_applicable"


@pytest.mark.parametrize("changes", [
    {"kinetic": [1.0, 2.0], "longitudinal": 0.0},
    {"kinetic": [1.0, 0.0], "longitudinal": 2.0},
])
def test_massless_rejects_nonzero_unavailable_channels(changes):
    with pytest.raises(ValueError):
        summarize_energy(_totals(**changes), 0)


@pytest.mark.parametrize("electron_inertia", [0, 1])
def test_zero_baseline_has_no_relative_infinity(electron_inertia):
    totals = _totals(
        kinetic=[0.0, 0.0],
        ion_internal=0.0,
        electron_perp=0.0,
        electron_parallel=0.0,
        magnetic=0.0,
        longitudinal=0.0,
        energy=0.0,
    )

    result = summarize_energy(totals, electron_inertia, baseline_total_J=0.0)

    assert result["total_J"] == 0.0
    assert result["balance_J"] == 0.0
    assert result["relative_total_change"] is None


@pytest.mark.parametrize(
    ("outward_J", "source_J", "expected_balance_J"),
    [
        (7.0, -3.0, 20.0),
        (-4.0, 9.0, -3.0),
    ],
)
def test_signed_exchange_follows_declared_balance_sign(
    outward_J,
    source_J,
    expected_balance_J,
):
    totals = _totals(
        kinetic=[10.0, 20.0],
        ion_internal=30.0,
        electron_perp=15.0,
        electron_parallel=5.0,
        magnetic=20.0,
        longitudinal=0.0,
        energy=99.0,
    )

    result = summarize_energy(
        totals,
        1,
        baseline_total_J=90.0,
        outward_J=outward_J,
        source_J=source_J,
    )

    assert result["total_J"] == 100.0
    assert result["balance_J"] == expected_balance_J


@pytest.mark.parametrize("electron_inertia", [-1, 2, 0.5, None, "0", True])
def test_invalid_model_flag_is_rejected(electron_inertia):
    with pytest.raises(ValueError):
        summarize_energy(_totals(), electron_inertia)


@pytest.mark.parametrize(
    "field",
    [
        "kinetic[0]",
        "kinetic[1]",
        "ion_internal",
        "electron_perp",
        "electron_parallel",
        "magnetic",
        "longitudinal",
        "energy",
        "baseline_total_J",
        "outward_J",
        "source_J",
    ],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_scalar_inputs_are_rejected(field, value):
    totals = _totals()
    kwargs = {}
    if field.startswith("kinetic"):
        index = int(field[-2])
        kinetic = list(totals.kinetic)
        kinetic[index] = value
        totals.kinetic = kinetic
    elif field == "baseline_total_J":
        kwargs[field] = value
    elif field in {"outward_J", "source_J"}:
        kwargs[field] = value
    else:
        setattr(totals, field, value)

    with pytest.raises(ValueError):
        summarize_energy(totals, 1, **kwargs)


@pytest.mark.parametrize("kinetic", [(1.0,), (1.0, 2.0, 3.0), 1.0])
def test_kinetic_must_contain_exactly_two_scalars(kinetic):
    with pytest.raises(ValueError):
        summarize_energy(_totals(kinetic=kinetic), 1)


def test_electron_parallel_is_used_as_stored_and_et_is_excluded():
    totals = _totals(
        kinetic=[1.25, 2.5],
        ion_internal=3.75,
        electron_perp=4.0,
        electron_parallel=32.75,
        magnetic=6.0,
        longitudinal=7.0,
        energy=57.0,
        epsilon0=1.0e300,
        ET=(1.0e150, -1.0e150, 1.0e150),
    )

    result = summarize_energy(totals, 1)

    assert result["electron_J"] == 39.25
    assert result["field_J"] == 13.0
    assert result["total_J"] == 57.25
    assert result["partition_difference_J"] == 0.25
