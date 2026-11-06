# `tests/warp/test_energy_summary.py`

Status: implementation-level test pseudocode for `summarize_energy`.

Use independent hand-summed records rather than reconstructing expected values
through the production helper.

```text
TEST finite_mass_partitions_retain_reported_total:
    record:
        kinetic = (3, 5)
        ion_internal = 11
        electron_perp = 13
        electron_parallel = 17
        magnetic = 19
        longitudinal = 23
        energy = 90
    snapshot = deep copy of record

    result = summarize_energy(
        record, 1,
        baseline_total_J=81,
        outward_J=7.5,
        source_J=2.5,
    )

    REQUIRE result is a built-in dict
    REQUIRE ion_J == 14
    REQUIRE electron_J == 35
    REQUIRE field_J == 42
    REQUIRE total_J == 91
    REQUIRE reported_total_J == 90
    REQUIRE partition_difference_J == 1
    REQUIRE balance_J == 15
    REQUIRE relative_total_change == 10 / 81
    REQUIRE both physical channels are labelled "applicable"
    REQUIRE record still equals snapshot

TEST massless_partitions_omit_unavailable_channels:
    record has zero kinetic[1] and zero longitudinal, with distinct nonzero
    pressure and magnetic energies
    result = summarize_energy(record, 0)

    REQUIRE electron_J equals only the two electron pressure terms
    REQUIRE field_J equals magnetic energy
    REQUIRE availability labels are "not_applicable"
    REQUIRE balance_J and relative_total_change are None

TEST massless_rejects_nonzero_unavailable_channels:
    separately use nonzero kinetic[1] and nonzero longitudinal
    REQUIRE ValueError

TEST zero_baseline_has_no_relative_infinity:
    use an all-zero record and baseline_total_J=0
    REQUIRE balance_J == 0 and relative_total_change is None

TEST signed_exchange_follows_declared_balance_sign:
    use a hand-summed total and both positive and negative outward/source values
    REQUIRE balance_J equals
        total - baseline + outward - source

TEST invalid_flags_and_nonfinite_scalars_are_rejected:
    REQUIRE ValueError for model flags other than integer-like 0 or 1
    REQUIRE ValueError when any required scalar input is NaN or infinity

TEST stored_electron_parallel_is_not_halved:
    use a distinct electron_parallel value and assert the literal electron and
    total sums; include irrelevant ET/epsilon0 attributes to show they do not
    enter the summary
```
