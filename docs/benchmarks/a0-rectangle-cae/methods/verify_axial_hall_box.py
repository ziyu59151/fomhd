"""Independent centered-difference checks of the cold Hall sampling oracle."""

import json

import numpy as np

from axial_hall_box import make_mode, wave


def peak(values):
    return float(np.max(np.abs(values)))


def relative(left, right):
    return peak(left - right) / max(peak(left), peak(right), np.finfo(float).tiny)


def derivative(function, points, axis, step):
    lower = list(points)
    upper = list(points)
    lower[axis] = lower[axis] - step
    upper[axis] = upper[axis] + step
    return (function(*upper) - function(*lower)) / (2.0 * step)


def curl(dr, dz, vector, radius):
    return np.stack(
        (-dz[..., 1], dz[..., 0] - dr[..., 2], dr[..., 1] + vector[..., 1] / radius),
        axis=-1,
    )


def divergence(dr, dz, vector, radius):
    return dr[..., 0] + vector[..., 0] / radius + dz[..., 2]


def pde_residuals(mode, fraction):
    """Differentiate public physical samples without using profile derivatives."""
    r = np.linspace(mode.r[0] + 0.07, mode.r[1] - 0.07, 39)[:, None]
    z = np.linspace(mode.z[0] + 0.07, mode.z[1] - 0.07, 41)[None, :]
    points = (r, z, 0.31 / mode.omega)
    steps = (fraction * np.ptp(mode.r), fraction * np.ptp(mode.z), fraction / mode.omega)
    sample = wave(*points, mode=mode)

    def d(key, axis):
        return derivative(
            lambda r, z, t: wave(r, z, t, mode=mode)[key],
            points,
            axis,
            steps[axis],
        )

    b = sample["delta_B"]
    e = sample["electric"]
    u_i = sample["ion_velocity"]
    u_e = sample["electron_velocity"]
    j = sample["current"]
    dr_b, dz_b = d("delta_B", 0), d("delta_B", 1)
    div_b = divergence(dr_b, dz_b, b, r)
    div_i = divergence(d("ion_velocity", 0), d("ion_velocity", 1), u_i, r)
    div_e = divergence(d("electron_velocity", 0), d("electron_velocity", 1), u_e, r)
    b0 = np.array([0.0, 0.0, mode.b0])
    return {
        "div_B": peak(div_b) / max(peak(dr_b), peak(dz_b), peak(b[..., 0] / r)),
        "ampere": relative(curl(dr_b, dz_b, b, r), mode.mu0 * j),
        "faraday": relative(d("delta_B", 2), -curl(d("electric", 0), d("electric", 1), e, r)),
        "ion_momentum": relative(
            mode.density * mode.ion_mass * d("ion_velocity", 2), np.cross(j, b0)
        ),
        "ion_continuity": relative(d("delta_density", 2), -mode.density * div_i),
        "electron_continuity": relative(d("delta_density", 2), -mode.density * div_e),
        "electron_velocity": relative(u_e, u_i - j / (mode.charge * mode.density)),
        "ohm": relative(e, -np.cross(u_e, b0)),
    }


def traces(mode):
    r = np.linspace(*mode.r, 97)
    z = np.linspace(*mode.z, 99)
    result = {}
    for side, rr, zz, normal in (
        ("R_lo", mode.r[0], z, 0),
        ("R_hi", mode.r[1], z, 0),
        ("Z_lo", r, mode.z[0], 2),
        ("Z_hi", r, mode.z[1], 2),
    ):
        tangents = [component for component in range(3) if component != normal]
        phase_checks = []
        for phase in (0.0, np.pi / 4.0, np.pi / 2.0):
            wall = wave(rr, zz, phase=phase, mode=mode)
            interior = wave(r[:, None], z[None, :], phase=np.pi / 4.0, mode=mode)
            phase_checks.append(
                {
                    "delta_B_normal": peak(wall["delta_B"][..., normal]) / peak(interior["delta_B"]),
                    "B_normal_minus_initial": peak(wall["B"][..., normal] - (mode.b0 if normal == 2 else 0.0)),
                    "E_tangential": peak(wall["electric"][..., tangents]) / peak(interior["electric"]),
                    "J_normal": peak(wall["current"][..., normal]) / peak(interior["current"]),
                    "u_i_normal": peak(wall["ion_velocity"][..., normal]) / peak(interior["ion_velocity"]),
                    "u_e_normal": peak(wall["electron_velocity"][..., normal]) / peak(interior["electron_velocity"]),
                }
            )
        result[side] = {key: max(check[key] for check in phase_checks) for key in phase_checks[0]}
    return result


def wave_energy(mode):
    nodes, weights = np.polynomial.legendre.leggauss(48)
    r = np.mean(mode.r) + 0.5 * np.ptp(mode.r) * nodes[:, None]
    z = np.mean(mode.z) + 0.5 * np.ptp(mode.z) * nodes[None, :]
    metric = 0.25 * np.ptp(mode.r) * np.ptp(mode.z) * r
    quadrature = metric * weights[:, None] * weights[None, :]
    values = []
    for fraction in (0.0, 0.125, 0.25, 0.375, 0.5):
        sample = wave(r, z, fraction * mode.period, mode=mode)
        kinetic = 0.5 * mode.ion_mass * mode.density * np.sum(
            sample["ion_velocity"] ** 2, axis=-1
        )
        magnetic = np.sum(sample["delta_B"] ** 2, axis=-1) / (2.0 * mode.mu0)
        values.append(float(np.sum(quadrature * (kinetic + magnetic))))
    return values


def main():
    mode = make_mode()
    pde = {fraction: pde_residuals(mode, fraction) for fraction in (0.002, 0.001, 0.0005)}
    wall = traces(mode)
    energy = wave_energy(mode)
    r = np.linspace(*mode.r, 901)
    z = np.linspace(*mode.z, 903)
    amplitude = 0.0
    for band in np.array_split(r, 29):
        sampled = wave(band[:, None], z[None, :], phase=np.pi / 2.0, mode=mode)
        amplitude = max(amplitude, peak(sampled["delta_B"][..., 1]) / mode.b0)
        assert np.all(sampled["density"] > 0.0)
        assert peak(sampled["delta_density"]) < 1e-13 * mode.density
        assert all(
            peak(sampled[key]) == 0.0
            for key in ("delta_p_ion", "delta_p_e_perp", "delta_p_e_parallel")
        )
    assert np.ptp(energy) / max(energy) < 1e-12, energy
    assert abs(amplitude / 1e-4 - 1.0) < 1e-5, amplitude
    for key, value in pde[0.0005].items():
        assert value < 2e-5, (key, value)
        if key not in ("electron_velocity", "ohm"):
            assert pde[0.002][key] > 12.0 * value, (key, pde)
    assert max(max(side.values()) for side in wall.values()) < 1e-12, wall
    print(json.dumps(
        {
            "omega_rad_s": mode.omega,
            "period_s": mode.period,
            "density_m^-3": mode.density,
            "vA_m_s": mode.alfven_speed,
            "di_m": mode.ion_inertial_length,
            "kR_m^-1": mode.kappa,
            "kZ_m^-1": mode.k,
            "peak_delta_Bphi_over_B0": amplitude,
            "relative_pde_residuals": pde,
            "wave_energy_J_per_rad": energy,
            "wall_traces": wall,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
