"""Command-line fixture writer for the continuous equilibrium prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SOURCE_ROOT / "src"), str(SOURCE_ROOT)]

from tools.circular_equilibrium import (  # noqa: E402
    ELEMENTARY_CHARGE,
    GAMMA_ION,
    INTERIOR_RESIDUAL_SAMPLES,
    MU0,
    NEAR_WALL_RESIDUAL_SAMPLES,
    build_gs_equilibrium,
    cylindrical_compressional_reference,
    gs_convergence_study,
    gs_residual_breakdown,
    gs_wall_residual_maxima,
    magnetic_axis,
    sample_gs,
    sample_vacuum,
    vacuum_residual_maxima,
    vacuum_wall_residual_maxima,
    verify_nested_flux,
)


DEFAULT_R0 = 3.0
DEFAULT_A = 0.5
DEFAULT_F0 = 6.0
DEFAULT_N0 = 2.0e19
DEFAULT_P_EDGE = 1000.0
DEFAULT_PPRIME = 1.0e4
DEFAULT_ALPHA = 0.6
DEFAULT_ION_MASS = 3.343583719e-27


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _provenance() -> dict[str, object]:
    source_files = (
        SOURCE_ROOT / "src/fomhd/fluid.py",
        SOURCE_ROOT / "tools/tokamak_equilibrium.py",
        SOURCE_ROOT / "tools/circular_equilibrium.py",
        SOURCE_ROOT / "docs/pseudocode/circular_equilibrium.md",
    )
    inspected = [
        {"path": str(path), "sha256": _sha256(path)} for path in source_files
    ]
    result: dict[str, object] = {
        "source_tree": str(SOURCE_ROOT),
        "source_commit": _git_commit(SOURCE_ROOT),
        "coordinate_convention": (
            "right-handed (R,phi,Z); psi Wb/rad; "
            "B=grad(psi) x grad(phi)+F grad(phi); F=R B_phi"
        ),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    optional_design = os.environ.get("FOMHD_DESIGN_ROOT")
    if optional_design:
        design_root = Path(optional_design).expanduser().resolve()
        result["design"] = {
            "tree": str(design_root),
            "commit": _git_commit(design_root),
            "available": design_root.is_dir(),
        }
        if design_root.is_dir():
            inspected.extend(
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                }
                for path in (
                    design_root
                    / "mainline_design/numerics/algorithm_design/01_equilibrium/equilibrium.md",
                    design_root
                    / "mainline_design/implementation/A0/A0_IMPLEMENTATION_SPEC.md",
                )
            )
    result["inspected_files"] = inspected
    return result


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threads": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }


def _units() -> dict[str, str]:
    return {
        "R_profile": "m",
        "Z_profile": "m",
        "psi": "Wb/rad",
        "psi_R": "T m",
        "psi_Z": "T m",
        "psi_RR": "T",
        "psi_RZ": "T",
        "psi_ZZ": "T",
        "B_R": "T",
        "B_phi": "T",
        "B_Z": "T",
        "J_R": "A/m^2",
        "J_phi": "A/m^2",
        "J_Z": "A/m^2",
        "E_R": "V/m",
        "E_phi": "V/m",
        "E_Z": "V/m",
        "u_i_R": "m/s",
        "u_i_phi": "m/s",
        "u_i_Z": "m/s",
        "u_e_R": "m/s",
        "u_e_phi": "m/s",
        "u_e_Z": "m/s",
        "n": "m^-3",
        "pressure": "Pa",
        "omega": "rad/s",
    }


def _masked_field(values: np.ndarray, mask_array: np.ndarray) -> np.ndarray:
    output = np.full(mask_array.shape, np.nan, dtype=np.float64)
    output[mask_array] = values
    return output


def _fixture_arrays(
    model,
    *,
    grid_size: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    if isinstance(grid_size, bool) or int(grid_size) != grid_size or int(grid_size) < 33:
        raise ValueError("grid_size must be an integer of at least 33")
    size = int(grid_size)
    R_profile = np.linspace(model.R0 - model.a, model.R0 + model.a, size)
    Z_profile = np.linspace(-model.a, model.a, size)
    R, Z = np.meshgrid(R_profile, Z_profile, indexing="xy")
    mask = (R - model.R0) ** 2 + Z**2 <= model.a**2 * (
        1.0 + 32.0 * np.finfo(np.float64).eps
    )
    state = sample_gs(model, R[mask], Z[mask])
    field_names = (
        "psi",
        "psi_R",
        "psi_Z",
        "psi_RR",
        "psi_RZ",
        "psi_ZZ",
        "B_R",
        "B_phi",
        "B_Z",
        "J_R",
        "J_phi",
        "J_Z",
        "E_R",
        "E_phi",
        "E_Z",
        "u_i_R",
        "u_i_phi",
        "u_i_Z",
        "u_e_R",
        "u_e_phi",
        "u_e_Z",
        "p_total",
        "p_ion",
        "p_e_perp",
        "p_e_parallel",
        "n",
    )
    fields = {
        name: _masked_field(np.asarray(state[name]), mask) for name in field_names
    }
    angle = np.linspace(0.0, 2.0 * np.pi, 513)
    wall_R = model.R0 + model.a * np.cos(angle)
    wall_Z = model.a * np.sin(angle)
    wall_R[-1] = wall_R[0]
    wall_Z[-1] = wall_Z[0]
    return R_profile, Z_profile, mask, fields, wall_R, wall_Z


def _all_checks_pass(
    *,
    residuals: dict[str, float],
    wall: dict[str, float],
    topology: dict[str, float | bool | int],
    minimum_pressure: float,
    maximum_current: float,
    maximum_bpol: float,
    convergence: list[dict[str, float | int]] | None,
) -> bool:
    core_residual_names = (
        "grad_shafranov_scaled",
        "ampere_scaled",
        "magnetic_divergence_scaled",
        "ion_momentum_scaled",
        "electron_momentum_scaled",
        "ion_continuity_scaled",
        "electron_continuity_scaled",
        "ion_pressure_scaled",
        "electron_cgl_perp_scaled",
        "electron_cgl_parallel_scaled",
        "faraday_scaled",
        "ue_grad_pe_scaled",
        "bb_grad_ue_scaled",
        "div_ue_scaled",
    )
    checks = [
        all(residuals[name] < 1.0e-9 for name in core_residual_names),
        wall["boundary_flux"] < 1.0e-14,
        wall["normal_B_scaled"] < 1.0e-10,
        wall["tangential_E_scaled"] < 1.0e-10,
        bool(topology["unique_crossings"]),
        bool(topology["nested_ordering"]),
        minimum_pressure > 0.0,
        maximum_current < 4.0e4,
        maximum_bpol < 2.0e-2,
    ]
    if convergence is not None:
        checks.append(float(convergence[-1]["psi_relative_error"]) < 1.0e-8)
    return all(checks)


def write_gs_fixture(
    output_dir: Path,
    *,
    grid_size: int = 129,
    degree: int = 12,
    run_convergence: bool = True,
    R0: float = DEFAULT_R0,
    a: float = DEFAULT_A,
    F0: float = DEFAULT_F0,
    n0: float = DEFAULT_N0,
    p_edge: float = DEFAULT_P_EDGE,
    pprime: float = DEFAULT_PPRIME,
    alpha: float = DEFAULT_ALPHA,
    ion_mass: float = DEFAULT_ION_MASS,
) -> dict[str, object]:
    """Solve, verify, and write one GS fixture plus its JSON sidecar."""
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = build_gs_equilibrium(
        R0=R0,
        a=a,
        F0=F0,
        n0=n0,
        p_edge=p_edge,
        pprime=pprime,
        alpha=alpha,
        degree=degree,
    )
    R_profile, Z_profile, mask, fields, wall_R, wall_Z = _fixture_arrays(
        model, grid_size=grid_size
    )
    residual_breakdown = gs_residual_breakdown(model)
    residuals = {
        name: max(
            residual_breakdown["interior"][name],
            residual_breakdown["near_wall"][name],
        )
        for name in residual_breakdown["interior"]
    }
    wall = gs_wall_residual_maxima(model)
    topology = verify_nested_flux(model)
    axis = magnetic_axis(model)
    axis_state = sample_gs(model, axis[0:1], axis[1:2])
    convergence = None
    if run_convergence:
        convergence = gs_convergence_study(
            R0=R0,
            a=a,
            F0=F0,
            n0=n0,
            p_edge=p_edge,
            pprime=pprime,
            alpha=alpha,
            degrees=(8, 10, 12),
            collocation_multipliers=(40, 60, 80),
        )
    inside_pressure = fields["p_total"][mask]
    minimum_pressure = float(np.min(inside_pressure))
    maximum_current = float(np.max(np.abs(fields["J_phi"][mask])))
    maximum_bpol = float(
        np.max(np.hypot(fields["B_R"][mask], fields["B_Z"][mask]))
    )
    checks_pass = _all_checks_pass(
        residuals=residuals,
        wall=wall,
        topology=topology,
        minimum_pressure=minimum_pressure,
        maximum_current=maximum_current,
        maximum_bpol=maximum_bpol,
        convergence=convergence,
    )
    cylinder = cylindrical_compressional_reference(
        a=a,
        B0=abs(F0 / R0),
        n0=n0,
        p_ion=float(axis_state["p_ion"][0]),
        p_electron=float(axis_state["p_e_perp"][0]),
        ion_mass=ion_mass,
    )
    metadata: dict[str, object] = {
        "status": "PASS" if checks_pass else "FAIL",
        "family": "constant-F, constant-positive-pprime Grad-Shafranov",
        "parameters": {
            "R0_m": float(R0),
            "a_m": float(a),
            "F0_T_m": float(F0),
            "n0_m^-3": float(n0),
            "p_edge_Pa": float(p_edge),
            "pprime_Pa_per_Wb_rad": float(pprime),
            "alpha_ion_fraction": float(alpha),
            "degree": int(degree),
            "collocation_count": int(model.collocation_count),
            "e_C": ELEMENTARY_CHARGE,
            "mu0_H_per_m": MU0,
            "gamma_i": GAMMA_ION,
        },
        "magnetic_axis_m": {"R": float(axis[0]), "Z": float(axis[1])},
        "axis_shift_m": float(axis[0] - R0),
        "psi_axis_Wb_rad": float(axis_state["psi"][0]),
        "layout": (
            "psi[iz,ir] with Z_profile[iz] increasing and "
            "R_profile[ir] increasing; invalid rectangular corners are NaN"
        ),
        "units": _units(),
        "provenance": _provenance(),
        "environment": _environment(),
        "checks": {
            "independently_sampled_continuous_residuals": residuals,
            "interior_continuous_residuals": residual_breakdown["interior"],
            "near_wall_continuous_residuals": residual_breakdown["near_wall"],
            "residual_sample_counts": {
                "interior": INTERIOR_RESIDUAL_SAMPLES,
                "near_wall": NEAR_WALL_RESIDUAL_SAMPLES,
            },
            "wall": wall,
            "topology": topology,
            "minimum_interior_pressure_Pa": minimum_pressure,
            "maximum_abs_J_phi_A_m2": maximum_current,
            "maximum_abs_Bpol_T": maximum_bpol,
            "solve_relative_residual": float(model.solve_residual),
            "convergence": convergence,
        },
        "cylinder_reference": {
            **{key: float(value) for key, value in cylinder.items()},
            "scope": (
                "n=0 planar cylindrical/large-aspect-ratio compressional "
                "reference with first positive J1(kr) wall root; not an n=1 "
                "or shifted-torus validation"
            ),
        },
        "pressure_constraint": (
            "p_total=p_edge+pprime*psi with positive pprime and p_edge; "
            "minimum is checked on the generated fixture mask"
        ),
        "runtime_seconds": 0.0,
    }
    metadata["runtime_seconds"] = float(time.perf_counter() - started)
    np.savez_compressed(
        output / "gs_equilibrium.npz",
        R_profile=R_profile,
        Z_profile=Z_profile,
        wall_mask=mask,
        wall_R=wall_R,
        wall_Z=wall_Z,
        psi_axis=float(axis_state["psi"][0]),
        magnetic_axis_R=float(axis[0]),
        magnetic_axis_Z=float(axis[1]),
        grid_shape_Z_R=np.array(mask.shape, dtype=np.int64),
        **fields,
    )
    (output / "gs_equilibrium.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        **metadata,
        "layout": metadata["layout"],
        "units": metadata["units"],
        "provenance": metadata["provenance"],
        "checks": metadata["checks"],
    }


def _load_wall(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            try:
                wall = np.column_stack((data["R"], data["Z"]))
            except KeyError as error:
                raise ValueError("wall NPZ must contain R and Z arrays") from error
    else:
        document = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(document, dict):
            wall = np.column_stack((document["R"], document["Z"]))
        else:
            wall = np.asarray(document, dtype=np.float64)
    points = np.asarray(wall, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("wall data must have N rows and R,Z columns")
    if not np.allclose(points[0], points[-1], rtol=0.0, atol=1.0e-14):
        raise ValueError("wall polygon must be closed")
    points[-1] = points[0]
    return points


def write_vacuum_fixture(
    output_dir: Path,
    *,
    wall: np.ndarray,
    F0: float = DEFAULT_F0,
    n0: float = DEFAULT_N0,
    pressure: float = DEFAULT_P_EDGE,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, object]:
    """Write one arbitrary-wall vacuum fixture."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = sample_vacuum(
        wall[:, 0], wall[:, 1], F0=F0, n0=n0, pressure=pressure, alpha=alpha
    )
    residuals = vacuum_residual_maxima(state)
    wall_residuals = vacuum_wall_residual_maxima(state, wall)
    status = (
        "PASS"
        if max(residuals.values()) == 0.0
        and wall_residuals["boundary_flux_variation"] == 0.0
        and wall_residuals["normal_B"] == 0.0
        and wall_residuals["tangential_E"] == 0.0
        else "FAIL"
    )
    metadata = {
        "status": status,
        "family": "vacuum B_phi=F0/R, B_pol=0",
        "parameters": {
            "F0_T_m": float(F0),
            "n0_m^-3": float(n0),
            "pressure_Pa": float(pressure),
            "alpha_ion_fraction": float(alpha),
        },
        "layout": "wall vertex arrays R[i],Z[i]; scalar fields indexed by i",
        "units": _units(),
        "provenance": _provenance(),
        "environment": _environment(),
        "checks": {
            "continuous_rhs_maxima": residuals,
            "wall": wall_residuals,
        },
    }
    np.savez_compressed(
        output / "vacuum_equilibrium.npz",
        wall_R=wall[:, 0],
        wall_Z=wall[:, 1],
        B_R=state["B_R"],
        B_phi=state["B_phi"],
        B_Z=state["B_Z"],
        J_R=state["J_R"],
        J_phi=state["J_phi"],
        J_Z=state["J_Z"],
        E_R=state["E_R"],
        E_phi=state["E_phi"],
        E_Z=state["E_Z"],
        u_i=state["u_i"],
        u_i_R=state["u_i_R"],
        u_i_phi=state["u_i_phi"],
        u_i_Z=state["u_i_Z"],
        u_e=state["u_e"],
        u_e_R=state["u_e_R"],
        u_e_phi=state["u_e_phi"],
        u_e_Z=state["u_e_Z"],
        n=state["n"],
        p_total=state["p_total"],
        p_ion=state["p_ion"],
        p_e_perp=state["p_e_perp"],
        p_e_parallel=state["p_e_parallel"],
        psi=state["psi"],
    )
    (output / "vacuum_equilibrium.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate verified continuous axisymmetric equilibria."
    )
    parser.add_argument("command", choices=("build", "vacuum"), nargs="?", default="build")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=129)
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--skip-convergence", action="store_true")
    parser.add_argument("--wall", type=Path)
    parser.add_argument("--F0", type=float, default=DEFAULT_F0)
    parser.add_argument("--n0", type=float, default=DEFAULT_N0)
    parser.add_argument("--pressure", type=float, default=DEFAULT_P_EDGE)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        result = write_gs_fixture(
            args.out_dir,
            grid_size=args.grid_size,
            degree=args.degree,
            run_convergence=not args.skip_convergence,
        )
    else:
        if args.wall is None:
            raise SystemExit("vacuum requires --wall RZ.json or RZ.npz")
        result = write_vacuum_fixture(
            args.out_dir,
            wall=_load_wall(args.wall),
            F0=args.F0,
            n0=args.n0,
            pressure=args.pressure,
            alpha=args.alpha,
        )
    print(json.dumps({"status": result["status"], "out_dir": str(args.out_dir)}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
