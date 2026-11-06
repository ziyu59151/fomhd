"""Foundation fixtures matching the native test support contracts."""

from __future__ import annotations

from fomhd.types import Budget, Norm, Parameters, References, Species


def parameters(K: int = 1) -> Parameters:
    ion = Species(
        id="ion",
        mass=1.67262192595e-27,
        charge=1.602176634e-19,
        gamma=5.0 / 3.0,
    )
    electron = Species(
        id="electron",
        mass=9.1093837139e-31,
        charge=-1.602176634e-19,
        gamma=0.0,
    )
    ref = References(
        length=0.2,
        time=1e-6,
        ni=1e12,
        ne=1e12,
        velocity=1.0,
        electric=1.0,
        magnetic=1e-5,
        current=0.0,
        charge=1e-7,
        pressure=1e-9,
    )
    response = sum(
        species.charge * species.charge * ref.ni / species.mass
        for species in (ion, electron)
    )
    ref.current = ref.time * ref.electric * (
        response + 1.0 / (1.25663706127e-6 * ref.length * ref.length)
    )
    p = Parameters(
        species=(ion, electron),
        epsilon0=8.8541878188e-12,
        mu0=1.25663706127e-6,
        B_min=1e-12,
        dt=1e-9,
        start_time=0.0,
        rank_rtol=1e-12,
        nsteps=4,
        K=K,
        Nphi=4 * K + 1,
        memory_limit=512 << 20,
        device_memory_limit=512 << 20,
        inner_distance=0.05,
        ref=ref,
        electron_inertia=1,
    )
    L, t, n, P = ref.length, ref.time, ref.ni, ref.pressure
    electric, magnetic = ref.electric, ref.magnetic
    force = ion.mass * n * ref.velocity / t
    power = P * L**3 / t
    energy = P * L**3 + magnetic * magnetic * L**3 / p.mu0

    def array(key: str, unit: str, reference: float, hard: bool = True) -> None:
        p.budgets[key + ".linf"] = Budget(
            atol=0.0,
            rtol=1e-9,
            reference=reference,
            unit=unit,
            norm=Norm.linf,
            hard=hard,
        )
        p.budgets[key + ".weighted_L2"] = Budget(
            atol=0.0,
            rtol=1e-9,
            reference=reference,
            unit=unit,
            norm=Norm.weighted_l2,
            hard=hard,
        )

    def scalar(key: str, unit: str, reference: float, hard: bool = True) -> None:
        p.budgets[key + ".value"] = Budget(
            atol=0.0,
            rtol=1e-9,
            reference=reference,
            unit=unit,
            norm=Norm.value,
            hard=hard,
        )

    for key in ("gauss", "poisson_linear", "transverse_divergence"):
        array(key, "V/m^2", electric / L)
    array("longitudinal_projection", "A/m^3", ref.current / L)
    array("darwin_dynamic", "A/(m^2*s)", ref.current / t)
    for key in ("current_divB", "current_ampere"):
        array(key, "T/m", magnetic / L, False)
    for key in (
        "initial_divB",
        "initial_ampere",
        "divB_update",
        "long_term_divB_change",
    ):
        array(key, "T/m", magnetic / L)
    array("CT_increment", "T/(m*s)", magnetic / (L * t))
    for key in ("CT_update", "equilibrium_B", "alias_B"):
        array(key, "T/s", magnetic / t)
    array("current_response", "A/(m^2*s)", ref.current / t)
    for key in ("RC_flux", "RC_mom", "deltaC"):
        array(key, "C/(m^3*s)", ref.charge / t)
    array("RL", "A/m^2", ref.current)
    for key in (
        "ion_internal",
        "electron_perp",
        "electron_parallel",
        "field_energy_B",
        "field_energy_L",
        "total_energy",
    ):
        scalar(key, "J", energy, False)
    scalar("field_exchange", "W", energy / t, False)
    scalar("topology_increment", "Wb", magnetic * L * L)
    scalar("total_charge_balance", "C", ref.charge * L**3)
    for key in ("momentum_x", "momentum_y", "momentum_z"):
        scalar(key, "kg*m/s", ion.mass * n * L**3 * ref.velocity, False)
    scalar("angular_momentum", "kg*m^2/s", ion.mass * n * L**4 * ref.velocity, False)
    for species in (ion, electron):
        suffix = "." + species.id
        for key in (
            "stage_continuity",
            "species_continuity",
            "alias_n",
            "equilibrium_n",
        ):
            array(key + suffix, "1/(m^3*s)", n / t)
        for key in (
            "wall_rhs",
            "wall_multiplier",
            "alias_mom",
            "equilibrium_mom",
        ):
            array(key + suffix, "kg/(m^2*s^2)", force)
        array("wall_state" + suffix, "kg/(m^2*s)", force * t)
        array("theta_identity" + suffix, "1/s", 1 / t)
        array("wall_number_flux" + suffix, "1/(m^2*s)", n * ref.velocity)
        array("wall_perp_flux" + suffix, "Pa*m/s", P * ref.velocity)
        if species.id == "electron":
            array("wall_parallel_flux" + suffix, "Pa*m/s", P * ref.velocity)
        for key in ("alias_perp", "equilibrium_perp"):
            array(key + suffix, "Pa/s", P / t)
        if species.id == "electron":
            for key in (
                "cgl_source",
                "alias_parallel",
                "equilibrium_parallel",
            ):
                array(key + suffix, "Pa/s", P / t)
        for key in (
            "electric_pairing",
            "pressure_pairing",
            "magnetic_work",
            "wall_work",
            "convection_pairing",
        ):
            scalar(key + suffix, "W", power)
        scalar("spectral_transfer_defect" + suffix, "W", power, False)
        scalar("total_number_balance" + suffix, "1", n * L**3)
        scalar("kinetic_energy" + suffix, "J", energy, False)
    return p
