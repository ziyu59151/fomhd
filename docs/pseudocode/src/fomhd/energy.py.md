# `src/fomhd/energy.py`

Status: implementation-level pseudocode for the bounded A0 energy summary.

## Scope

`summarize_energy` is a read-only host-side presentation seam. It consumes only
the already computed SI totals and optional already integrated signed exchanges.
It does not integrate fields, allocate solver state, define tolerances, or claim
an all-energy conservation law.

The input record provides:

```text
kinetic[0:2], ion_internal, electron_perp, electron_parallel,
magnetic, longitudinal, energy
```

All quantities are scalar values in joules. `energy` is the value already
reported by the caller and is not silently replaced.

## Public contract

```text
FUNCTION summarize_energy(
    totals,
    electron_inertia,
    *,
    baseline_total_J = None,
    outward_J = 0.0,
    source_J = 0.0
)
```

Return a built-in `dict` with this schema:

```text
ion_J                       ion kinetic plus ion internal energy
electron_J                  electron pressure energy, plus electron kinetic
                            energy only for electron_inertia == 1
field_J                     magnetic energy, plus longitudinal electric energy
                            only for electron_inertia == 1
total_J                     ion_J + electron_J + field_J
reported_total_J            the input totals.energy value
partition_difference_J      total_J - reported_total_J
baseline_total_J            normalized baseline, or None
balance_J                   total_J - baseline_total_J + outward_J - source_J
                            only when a baseline was supplied, else None
relative_total_change       (total_J - baseline_total_J) / baseline_total_J
                            only when the supplied baseline is nonzero, else None
outward_J                   normalized signed outward energy exchange
source_J                    normalized signed source energy exchange
electron_kinetic_energy     "applicable" or "not_applicable"
longitudinal_electric_energy
                            "applicable" or "not_applicable"
```

`outward_J` and `source_J` may be negative to represent the opposite signed
exchange direction. They are not included in `total_J`; they appear only in
`balance_J`.

## Procedure

```text
FUNCTION finite_float(value, label):
    TRY value_float = float(value)
    IF conversion fails OR value_float is NaN or infinity:
        raise ValueError(label + " must be finite")
    RETURN value_float

FUNCTION summarize_energy(...):
    REQUIRE electron_inertia is an integer-like value equal to 0 or 1
    REQUIRE totals.kinetic has exactly two entries

    ion_kinetic = finite_float(totals.kinetic[0], "kinetic[0]")
    electron_kinetic = finite_float(totals.kinetic[1], "kinetic[1]")
    ion_internal = finite_float(totals.ion_internal, "ion_internal")
    electron_perp = finite_float(totals.electron_perp, "electron_perp")
    electron_parallel = finite_float(totals.electron_parallel, "electron_parallel")
    magnetic = finite_float(totals.magnetic, "magnetic")
    longitudinal = finite_float(totals.longitudinal, "longitudinal")
    reported_total = finite_float(totals.energy, "energy")

    IF baseline_total_J is supplied:
        baseline = finite_float(baseline_total_J, "baseline_total_J")
    ELSE:
        baseline = None
    outward = finite_float(outward_J, "outward_J")
    source = finite_float(source_J, "source_J")

    IF electron_inertia == 0:
        REQUIRE electron_kinetic == 0
        REQUIRE longitudinal == 0
        electron = electron_perp + electron_parallel
        field = magnetic
        electron_kinetic_label = "not_applicable"
        longitudinal_label = "not_applicable"
    ELSE:
        electron = electron_kinetic + electron_perp + electron_parallel
        field = magnetic + longitudinal
        electron_kinetic_label = "applicable"
        longitudinal_label = "applicable"

    ion = ion_kinetic + ion_internal
    total = ion + electron + field
    partition_difference = total - reported_total

    IF baseline is None:
        balance = None
        relative_change = None
    ELSE:
        balance = total - baseline + outward - source
        IF baseline == 0:
            relative_change = None
        ELSE:
            relative_change = (total - baseline) / baseline

    RETURN the schema above as a new built-in dict
```

No input field is assigned. Electron parallel pressure is consumed exactly as
stored, without an additional factor of one half. Transverse electric-field
storage is outside this seam.
