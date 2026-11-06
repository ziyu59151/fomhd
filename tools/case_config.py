#!/usr/bin/env python3
"""Render a documented, strict A0 massless-electron case configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fomhd.io import _massless_budget_registry  # noqa: E402
from fomhd.types import Parameters, Species  # noqa: E402


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def _line(
    key: str,
    value: Any,
    *,
    unit: str,
    allowed: str,
    purpose: str,
) -> str:
    return (
        f"{key} = {_toml(value)}  "
        f"# Unit: {unit}; allowed: {allowed}; purpose: {purpose}; validation: local template, global scope pending"
    )


def _budget_scale(unit: str) -> float:
    scales = {
        "1": 1.0e12,
        "C/m^3": 1.0e-7,
        "1/(m^3*s)": 1.0e18,
        "kg/(m^2*s^2)": 1.0e-12,
        "A/m^2": 1.0e3,
        "1/(m^2*s)": 1.0e12,
        "A/m^3": 5.0e3,
        "T/(m*s)": 1.0e1,
        "T/s": 1.0e1,
        "T/m": 5.0e-5,
        "V/m^3": 25.0,
        "V/m": 1.0,
        "Pa/s": 1.0e-3,
        "W": 1.0e-6,
        "J": 1.0e-9,
        "C/(m^3*s)": 1.0e-1,
        "1/s": 1.0e6,
        "kg*m/s": 1.0e-27,
        "kg*m^2/s": 1.0e-28,
    }
    return scales.get(unit, 1.0)


def render_massless_example() -> str:
    """Return the complete, commented strict massless case template."""
    ion = Species(id="ion", mass=1.67262192595e-27, charge=1.602176634e-19)
    electron = Species(
        id="electron", mass=9.1093837139e-31, charge=-1.602176634e-19
    )
    registry = _massless_budget_registry(
        Parameters(species=(ion, electron), electron_inertia=0)
    )
    lines = [
        "# Default A0 massless quasineutral CGL-B case.",
        "# Scope: bounded local numerical fixture; no global Tokamak validation.",
        "",
        _line(
            "schema_version",
            1,
            unit="1",
            allowed="integer 1",
            purpose="select the strict A0 configuration schema",
        ),
        _line(
            "model",
            "A0",
            unit="1",
            allowed="A0",
            purpose="select the A0 solver contract",
        ),
        _line(
            "units",
            "SI",
            unit="1",
            allowed="SI",
            purpose="require SI input and output units",
        ),
        _line(
            "layout_version",
            1,
            unit="1",
            allowed="integer 1",
            purpose="select the A0 state and operator layout",
        ),
        _line(
            "case_name",
            "massless-example",
            unit="1",
            allowed="safe name using letters, digits, underscore, or hyphen",
            purpose="identify this local numerical case",
        ),
        "",
        "[species]",
        _line(
            "ion_id",
            "ion",
            unit="1",
            allowed="unique safe species identifier",
            purpose="name the ion population",
        ),
        _line(
            "electron_id",
            "electron",
            unit="1",
            allowed="unique safe species identifier different from ion_id",
            purpose="name the electron population",
        ),
        _line(
            "qi",
            1.602176634e-19,
            unit="C",
            allowed="finite positive ion charge",
            purpose="set the ion charge",
        ),
        _line(
            "mi",
            1.67262192595e-27,
            unit="kg",
            allowed="finite positive ion mass",
            purpose="set the ion inertial mass",
        ),
        _line(
            "qe",
            -1.602176634e-19,
            unit="C",
            allowed="finite negative electron charge",
            purpose="set the electron charge",
        ),
        _line(
            "me",
            9.1093837139e-31,
            unit="kg",
            allowed="finite positive physical electron mass",
            purpose="retain physical electron metadata in the massless model",
        ),
        _line(
            "gamma_i",
            1.6666666666666667,
            unit="1",
            allowed="finite adiabatic index greater than 1",
            purpose="set ion pressure closure",
        ),
        _line(
            "electron_inertia",
            0,
            unit="1",
            allowed="0 for default massless or 1 for explicit local finite mass",
            purpose="select quasineutral massless electron dynamics",
        ),
        "",
        "[grid]",
        _line(
            "R0",
            1.0,
            unit="m",
            allowed="finite radial origin",
            purpose="place the fixture in major-radius coordinates",
        ),
        _line(
            "Z0",
            0.0,
            unit="m",
            allowed="finite axial origin",
            purpose="place the fixture in vertical coordinates",
        ),
        _line(
            "hR",
            0.2,
            unit="m",
            allowed="finite positive radial spacing",
            purpose="set radial grid spacing",
        ),
        _line(
            "hZ",
            0.2,
            unit="m",
            allowed="finite positive axial spacing",
            purpose="set axial grid spacing",
        ),
        _line(
            "NR",
            4,
            unit="1",
            allowed="integer at least 2",
            purpose="set radial node count",
        ),
        _line(
            "NZ",
            4,
            unit="1",
            allowed="integer at least 2",
            purpose="set axial node count",
        ),
        _line(
            "K",
            1,
            unit="1",
            allowed="nonnegative integer",
            purpose="set maximum retained toroidal mode number",
        ),
        _line(
            "Nphi",
            5,
            unit="1",
            allowed="positive odd integer at least 4*K+1",
            purpose="set physical toroidal sample count",
        ),
        "",
        "[time]",
        _line(
            "start_time",
            0.0,
            unit="s",
            allowed="finite nonnegative start time",
            purpose="set the initial scheduler time",
        ),
        _line(
            "dt",
            2.0e-13,
            unit="s",
            allowed="finite positive explicit step",
            purpose="resolve the local massless Hall scale",
        ),
        _line(
            "nsteps",
            1,
            unit="1",
            allowed="nonnegative integer no greater than 2^52",
            purpose="request one local explicit step",
        ),
        _line(
            "initial_kind",
            "non_equilibrium",
            unit="1",
            allowed="equilibrium or non_equilibrium",
            purpose="load the supplied state without an equilibrium solve",
        ),
        _line(
            "integrator",
            "rk4",
            unit="1",
            allowed="rk4 default or midpoint for an explicit comparison",
            purpose="select the explicit time integrator",
        ),
        _line(
            "dt_guard",
            "warn",
            unit="1",
            allowed="error or warn",
            purpose="record guard exceedance for this demonstration",
        ),
        _line(
            "minimum_samples_per_cycle",
            32.0,
            unit="1",
            allowed="finite positive sample count",
            purpose="require time resolution of the fastest estimate",
        ),
        _line(
            "max_amplitude_error",
            0.01,
            unit="1",
            allowed="finite positive relative error",
            purpose="bound finite-horizon amplitude error",
        ),
        _line(
            "max_phase_error_rad",
            0.1,
            unit="rad",
            allowed="finite positive phase error",
            purpose="bound finite-horizon phase error",
        ),
        "",
        "[boundary]",
        _line(
            "kind",
            "fixed_lcfs_static",
            unit="1",
            allowed="fixed_lcfs_static",
            purpose="select the supported static wall boundary",
        ),
        _line(
            "potential",
            0.0,
            unit="V",
            allowed="0.0 for this boundary version",
            purpose="fix the wall potential",
        ),
        _line(
            "version",
            "staircase_relative_v1",
            unit="1",
            allowed="staircase_relative_v1",
            purpose="select the supported staircase boundary contract",
        ),
        "",
        "[solver]",
        _line(
            "backend",
            "cpu",
            unit="1",
            allowed="cpu or cuda",
            purpose="select the execution backend explicitly",
        ),
        _line(
            "precision",
            "fp64",
            unit="1",
            allowed="fp64",
            purpose="retain double-precision real storage",
        ),
        _line(
            "complex_precision",
            "complex128",
            unit="1",
            allowed="complex128",
            purpose="retain double-precision complex storage",
        ),
        _line(
            "memory_limit_bytes",
            536870912,
            unit="B",
            allowed="positive bounded integer",
            purpose="cap host solver allocation",
        ),
        _line(
            "device_memory_limit_bytes",
            536870912,
            unit="B",
            allowed="positive bounded integer",
            purpose="cap selected-device allocation",
        ),
        _line(
            "rank_rtol",
            1.0e-12,
            unit="1",
            allowed="finite positive value below 1",
            purpose="set numerical rank detection",
        ),
        _line(
            "device_ordinal",
            0,
            unit="1",
            allowed="0 for CPU or a valid CUDA ordinal",
            purpose="select CPU or one explicit CUDA device",
        ),
        _line(
            "cpu_threads",
            1,
            unit="1",
            allowed="integer 1",
            purpose="require deterministic one-thread reference reductions",
        ),
        "",
        "[solver.constants]",
        _line(
            "epsilon0_Fpm",
            8.8541878188e-12,
            unit="F/m",
            allowed="finite positive permittivity",
            purpose="set the electrostatic constant",
        ),
        _line(
            "mu0_Hpm",
            1.25663706127e-6,
            unit="H/m",
            allowed="finite positive permeability",
            purpose="set the magnetic constant",
        ),
        _line(
            "version",
            "A0-SI-v1",
            unit="1",
            allowed="nonempty provenance label",
            purpose="identify the SI constant set",
        ),
        "",
        "[solver.references]",
        _line(
            "L_m",
            0.2,
            unit="m",
            allowed="finite positive length",
            purpose="set the nondimensionalization length",
        ),
        _line(
            "t_s",
            1.0e-6,
            unit="s",
            allowed="finite positive time",
            purpose="set the nondimensionalization time",
        ),
        _line(
            "n_i_m3",
            1.0e12,
            unit="1/m^3",
            allowed="finite positive ion density",
            purpose="set the ion density scale",
        ),
        _line(
            "n_e_m3",
            1.0e12,
            unit="1/m^3",
            allowed="finite positive electron density",
            purpose="set the electron density scale",
        ),
        _line(
            "u_mps",
            1.0,
            unit="m/s",
            allowed="finite positive velocity",
            purpose="set the velocity scale",
        ),
        _line(
            "E_Vpm",
            1.0,
            unit="V/m",
            allowed="finite positive electric field",
            purpose="set the electric-field scale",
        ),
        _line(
            "B_T",
            1.0e-5,
            unit="T",
            allowed="finite positive magnetic field",
            purpose="set the magnetic-field scale",
        ),
        _line(
            "J_Apm2",
            1.0e3,
            unit="A/m^2",
            allowed="finite positive current density",
            purpose="set the current-density scale",
        ),
        _line(
            "rho_Cpm3",
            1.0e-7,
            unit="C/m^3",
            allowed="finite positive charge density",
            purpose="set the charge-density scale",
        ),
        _line(
            "p_Pa",
            1.0e-9,
            unit="Pa",
            allowed="finite positive pressure",
            purpose="set the pressure scale",
        ),
        "",
        "[budgets.domain]",
        _line(
            "n_min_i",
            0.0,
            unit="1/m^3",
            allowed="finite nonnegative floor",
            purpose="set the ion density positivity floor",
        ),
        _line(
            "n_min_e",
            0.0,
            unit="1/m^3",
            allowed="finite nonnegative floor",
            purpose="set the electron density positivity floor",
        ),
        _line(
            "n_face_min_i",
            0.0,
            unit="1/m^3",
            allowed="finite nonnegative floor",
            purpose="set the ion face-density positivity floor",
        ),
        _line(
            "n_face_min_e",
            0.0,
            unit="1/m^3",
            allowed="finite nonnegative floor",
            purpose="set the electron face-density positivity floor",
        ),
        _line(
            "p_min_i",
            0.0,
            unit="Pa",
            allowed="finite nonnegative floor",
            purpose="set the ion pressure positivity floor",
        ),
        _line(
            "p_min_perp_e",
            0.0,
            unit="Pa",
            allowed="finite nonnegative floor",
            purpose="set the perpendicular electron pressure floor",
        ),
        _line(
            "p_min_parallel_e",
            0.0,
            unit="Pa",
            allowed="finite nonnegative floor",
            purpose="set the parallel electron pressure floor",
        ),
        _line(
            "B_min",
            1.0e-12,
            unit="T",
            allowed="finite nonnegative floor",
            purpose="retain a nonzero magnetic field for quasineutral closure",
        ),
        "",
        "# Exact massless registry; each key's unit, norm, and gate are strict.",
    ]
    for name, (unit, norm, required_hard) in registry.items():
        allowed_gate = (
            "hard (required for this key)"
            if required_hard
            else "hard or observe"
        )
        lines.extend(
            [
                "",
                f"[budgets.checks.{json.dumps(name, ensure_ascii=True)}]",
                _line(
                    "atol",
                    0.0,
                    unit=unit,
                    allowed="finite nonnegative value",
                    purpose="set the absolute acceptance tolerance",
                ),
                _line(
                    "rtol",
                    2.0e-6,
                    unit="1",
                    allowed="finite nonnegative dimensionless value",
                    purpose="set the relative acceptance tolerance",
                ),
                _line(
                    "reference",
                    _budget_scale(unit),
                    unit=unit,
                    allowed="finite positive scale",
                    purpose="set the dimensional residual scale",
                ),
                _line(
                    "unit",
                    unit,
                    unit=unit,
                    allowed=f"exact SI unit {json.dumps(unit)}",
                    purpose="bind this residual to its physical dimension",
                ),
                _line(
                    "norm",
                    norm,
                    unit="1",
                    allowed=f"exact norm {json.dumps(norm)}",
                    purpose="select the residual reduction",
                ),
                _line(
                    "gate",
                    "hard" if required_hard else "observe",
                    unit="1",
                    allowed=allowed_gate,
                    purpose="control whether failure rejects a candidate",
                ),
            ]
        )
    lines.extend(
        [
            "",
            "[diagnostics]",
            _line(
                "inner_distance_m",
                0.05,
                unit="m",
                allowed="finite positive distance",
                purpose="set the wall-proximity diagnostic radius",
            ),
            _line(
                "alias_policy",
                "initial_midpoint_restart",
                unit="1",
                allowed="initial_midpoint_restart",
                purpose="select the supported alias audit policy",
            ),
            _line(
                "timing_repetitions",
                1,
                unit="1",
                allowed="positive integer",
                purpose="set optional timing repetitions",
            ),
            _line(
                "timing_warmup",
                0,
                unit="1",
                allowed="nonnegative integer",
                purpose="set optional timing warmup count",
            ),
            _line(
                "mode_observations",
                [],
                unit="1",
                allowed="array of valid observation tables",
                purpose="select optional Fourier observations",
            ),
            "",
            "[io]",
            _line(
                "geometry_file",
                "geometry.npz",
                unit="1",
                allowed="relative or absolute regular NPZ path",
                purpose="identify the local geometry input",
            ),
            _line(
                "initial_file",
                "initial.npz",
                unit="1",
                allowed="relative or absolute regular NPZ path",
                purpose="identify the accepted initial-state input",
            ),
            _line(
                "output_dir",
                "run",
                unit="1",
                allowed="new output directory path",
                purpose="select the local run directory",
            ),
            _line(
                "snapshot_stride",
                1,
                unit="1",
                allowed="positive integer",
                purpose="set accepted-state snapshot cadence",
            ),
            _line(
                "checkpoint_stride",
                1,
                unit="1",
                allowed="positive integer",
                purpose="set accepted-state checkpoint cadence",
            ),
            _line(
                "max_input_bytes",
                16777216,
                unit="B",
                allowed="positive bounded integer",
                purpose="cap each input file",
            ),
            _line(
                "max_uncompressed_bytes",
                33554432,
                unit="B",
                allowed="positive bounded integer",
                purpose="cap NPZ expansion",
            ),
            _line(
                "max_entries",
                4096,
                unit="1",
                allowed="positive bounded integer",
                purpose="cap NPZ entry count",
            ),
            _line(
                "max_header_bytes",
                1048576,
                unit="B",
                allowed="positive bounded integer",
                purpose="cap NPZ header size",
            ),
            _line(
                "source_manifest_file",
                "source_manifest.json",
                unit="1",
                allowed="relative or absolute regular JSON path",
                purpose="identify input provenance and hashes",
            ),
            _line(
                "write_residual_arrays",
                False,
                unit="1",
                allowed="true or false",
                purpose="control optional residual-array output",
            ),
            "",
            "# Finite electron inertia requires electron_inertia = 1 and the",
            "# finite budget registry. That historical local option is not the",
            "# default and does not establish global Tokamak validation.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new configuration file to create",
    )
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        handle.write(render_massless_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
