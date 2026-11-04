"""Independent SI-scale checks for massless verification fixture budgets."""

import pytest

import massless_verification_fixture as hall
import test_massless_support as support
import test_model_routing as routing


@pytest.mark.parametrize("fixture_name", ["support", "routing", "hall"])
def test_massless_fixture_budgets_follow_physical_units(fixture_name, monkeypatch):
    if fixture_name == "support":
        parameters = support.parameters()
    elif fixture_name == "routing":
        parameters = routing._parameters(inertia=0)
    else:
        references = hall.References

        def rescaled_references(**values):
            values["length"] = 0.3
            return references(**values)

        monkeypatch.setattr(hall, "References", rescaled_references)
        parameters = hall._parameters(hall.HallCase(kdi=1.0))

    ref = parameters.ref
    volume = ref.length**3
    advective_time = ref.length / ref.velocity
    pressure_energy = ref.pressure * volume
    total_charge = ref.charge * volume
    ion_momentum = parameters.species[0].mass * ref.ni * volume * ref.velocity
    expected = {
        "quasineutrality": ("C/m^3", total_charge / volume),
        "quasineutrality_interval": ("C/(m^3*s)", total_charge / volume / ref.time),
        "current_definition": ("A/m^2", ref.current),
        "wall_initial_current": ("A/m^2", ref.current),
        "wall_current": ("A/m^2", ref.current),
        "wall_current_interval": ("A/m^2", ref.current),
        "djj": ("A/m^3", ref.current / ref.length),
        "djj_interval": ("A/m^3", ref.current / ref.length),
        "wall_field_original_rows": ("V/m^3", ref.electric / ref.length**2),
        "electron_pressure_adjoint": ("W", pressure_energy / advective_time),
        "electron_electric_pairing": ("W", pressure_energy / advective_time),
        "total_charge_balance_interval": ("C", total_charge),
        "momentum_balance_interval_x": ("kg*m/s", ion_momentum),
        "momentum_balance_interval_y": ("kg*m/s", ion_momentum),
        "momentum_balance_interval_z": ("kg*m/s", ion_momentum),
    }
    for name, (unit, reference) in expected.items():
        matches = [
            parameters.budgets[f"{name}.{suffix}"]
            for suffix in ("linf", "weighted_L2", "value")
            if f"{name}.{suffix}" in parameters.budgets
        ]
        assert matches, name
        for budget in matches:
            assert budget.unit == unit, name
            assert budget.reference == pytest.approx(reference, rel=1e-14, abs=0.0), name
