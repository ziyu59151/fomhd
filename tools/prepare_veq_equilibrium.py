"""Run the pinned VEQPy circular control and export FOMHD preprocessing data."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from veq_equilibrium import (
    SOURCE_ARCHIVE_SHA256,
    SOURCE_COMMIT,
    SOURCE_TREE,
    build_constant_f_snapshot,
    make_source_metadata,
    strong_gs_metrics,
    write_equilibrium_product,
)


R0_M = 3.0
Z0_M = 0.0
A_M = 0.5
F_EDGE_TM = 6.0
B0_T = F_EDGE_TM / R0_M
P_EDGE_TARGET_PA = 1000.0
PPRIME_F = 1.0e4
DENSITY_M3 = 2.0e19
ION_PRESSURE_FRACTION = 0.6
RAW_P0_INPUT_PA = 1000.0
IP_TARGET_A = PPRIME_F * math.pi * R0_M * A_M**2


@dataclass(frozen=True, slots=True)
class Refinement:
    name: str
    Nr: int
    Nt: int
    h_count: int
    kappa_count: int
    s_count: int
    psin_count: int
    sample_count: int


CIRCLE_REFINEMENTS = (
    Refinement("coarse", 12, 16, 1, 1, 1, 2, 9),
    Refinement("medium", 16, 24, 2, 2, 1, 3, 13),
    Refinement("fine", 24, 32, 3, 3, 2, 4, 17),
)
PRIMARY_EVALUATION_GRID = (33, 48)
FINE_EVALUATION_GRIDS = ((33, 48), (37, 64), (49, 96))


def _native_shape_profile_arrays(equilibrium):
    arrays = {}
    records = {}
    for name, profile in sorted(equilibrium.shape_profiles.items()):
        coeff = (
            np.empty(0, dtype=np.float64)
            if profile.coeff is None
            else np.asarray(profile.coeff, dtype=np.float64)
        )
        key = f"profile_coeff_{name}"
        arrays[key] = coeff
        records[name] = {
            "profile": name,
            "class": type(profile).__name__,
            "scale": float(profile.scale),
            "power": int(profile.power),
            "envelope_power": int(profile.envelope_power),
            "amplitude_power": float(profile.amplitude_power),
            "offset": float(profile.offset),
            "coeff": [float(value) for value in coeff],
        }
    return arrays, records


def _convert_equilibrium(equilibrium, evaluation_grid):
    metadata = make_source_metadata(
        source_commit=SOURCE_COMMIT,
        source_tree=SOURCE_TREE,
        archive_sha256=SOURCE_ARCHIVE_SHA256,
        R0_m=R0_M,
        Z0_m=Z0_M,
        a_m=A_M,
        F_edge_Tm=F_EDGE_TM,
        p_edge_target_Pa=P_EDGE_TARGET_PA,
        pprime_F_Pa_per_Wb_rad=PPRIME_F,
        density_m3=DENSITY_M3,
        ion_pressure_fraction=ION_PRESSURE_FRACTION,
        native_p0_Pa=float(equilibrium.p0),
        raw_p0_input_Pa=RAW_P0_INPUT_PA,
        Ip_A=IP_TARGET_A,
    )
    r = np.asarray(equilibrium.r, dtype=np.float64)
    theta = np.asarray(equilibrium.theta, dtype=np.float64)
    return build_constant_f_snapshot(
        metadata=metadata,
        r=r,
        theta=theta,
        R=np.asarray(equilibrium.R, dtype=np.float64),
        Z=np.asarray(equilibrium.Z, dtype=np.float64),
        radial_differentiator=np.asarray(
            evaluation_grid.differentiator, dtype=np.float64
        ),
        psi_Wb=np.broadcast_to(
            np.asarray(equilibrium.Psi, dtype=np.float64)[:, None],
            equilibrium.R.shape,
        ),
        p_Pa=np.broadcast_to(equilibrium.P[:, None], equilibrium.R.shape),
        p0_Pa=float(equilibrium.p0),
        jphi_A_m2=np.asarray(equilibrium.jphi, dtype=np.float64),
        native_p_psi_Pa_per_Wb_rad=np.asarray(
            equilibrium.P_psi, dtype=np.float64
        ),
    )


def _solve_one(plan: Refinement, output_dir: Path) -> dict[str, object]:
    import veqpy as veq

    topology = veq.KernelTopology(
        h_count=plan.h_count,
        v_count=0,
        kappa_count=plan.kappa_count,
        psin_count=plan.psin_count,
        F_count=0,
        c_counts=(),
        s_counts=(plan.s_count,),
        Nr=plan.Nr,
        Nt=plan.Nt,
        route="PF",
        coordinate="psin",
        nodes="uniform",
        constraint="ip",
        sample_count=plan.sample_count,
    )
    if topology.x_size > 100 or topology.Nr > 64 or topology.Nt > 128:
        raise ValueError(
            f"refinement {plan.name} exceeds the bounded preprocessing caps"
        )
    boundary = veq.KernelBoundary(
        a=A_M,
        R0=R0_M,
        Z0=Z0_M,
        B0=B0_T,
        ka=1.0,
        c_offsets=(0.0,),
        s_offsets=(0.0,),
    )
    source_axis = np.linspace(
        0.0, 1.0, topology.sample_count, dtype=np.float64
    )
    source = veq.KernelSource(
        P_psin=np.full(source_axis.size, -PPRIME_F, dtype=np.float64),
        FF_psin=np.zeros(source_axis.size, dtype=np.float64),
        p0=RAW_P0_INPUT_PA,
        Ip=IP_TARGET_A,
        case_name=f"veqpy-circle-{plan.name}",
    )
    kernel = veq.Kernel(
        topology=topology,
        recipe=veq.KernelRecipe(backend="numba", layout="degree"),
        config=veq.KernelConfig(
            method="powell",
            max_residual=1.0e-8,
            max_evaluations=3000,
            initial="cold",
            norm="fast",
        ),
    )
    try:
        prepare_start = time.perf_counter()
        kernel.prepare()
        prepare_wall_s = time.perf_counter() - prepare_start
        solve_start = time.perf_counter()
        result = kernel.solve(boundary=boundary, source=source)
        solve_wall_s = time.perf_counter() - solve_start
        if not result.success:
            raise RuntimeError(
                f"VEQPy solve failed for {plan.name}: raw_norm={result.raw_norm}"
            )
        evaluation_specs = (
            FINE_EVALUATION_GRIDS
            if plan.name == "fine"
            else (PRIMARY_EVALUATION_GRID,)
        )
        evaluations = []
        for Nr, Nt in evaluation_specs:
            evaluation_grid = veq.Grid(
                Nr=Nr, Nt=Nt, quadrature_scheme="uniform"
            )
            evaluation_start = time.perf_counter()
            equilibrium = kernel.build_equilibrium(grid=evaluation_grid)
            evaluation_wall_s = time.perf_counter() - evaluation_start
            evaluations.append(
                (
                    Nr,
                    Nt,
                    evaluation_grid,
                    equilibrium,
                    evaluation_wall_s,
                )
            )
    finally:
        kernel.close()
    converted_evaluations = []
    for Nr, Nt, evaluation_grid, equilibrium, evaluation_wall_s in evaluations:
        converted = _convert_equilibrium(equilibrium, evaluation_grid)
        metrics = strong_gs_metrics(converted)
        converted_evaluations.append(
            (
                Nr,
                Nt,
                equilibrium,
                evaluation_wall_s,
                converted,
                metrics,
            )
        )
    evaluation_refinement = {
        "fixed_solve_id": f"{plan.name}-singular-solve",
        "solver_representation_changed": False,
        "comparison": (
            "same solver state and coefficients on varied output grids"
        ),
        "grids": [
            {
                "name": f"{Nr}x{Nt}",
                "Nr": Nr,
                "Nt": Nt,
                "evaluation_wall_s": evaluation_wall_s,
                "strong_gs": metrics,
                "wall_normal_B_smooth": metrics["wall_normal_B_smooth"],
                "wall_normal_B_polygon_midpoint": (
                    metrics["wall_normal_B_polygon_midpoint"]
                ),
                "I_p_profile_A": converted.I_p_profile_A,
                "I_p_ampere_A": converted.I_p_ampere_A,
                "axis_B_poloidal_max_abs_T": (
                    converted.axis_B_poloidal_max_abs_T
                ),
            }
            for (
                Nr,
                Nt,
                _,
                evaluation_wall_s,
                converted,
                metrics,
            ) in converted_evaluations
        ],
    }
    written = []
    for index, row in enumerate(converted_evaluations):
        Nr, Nt, equilibrium, evaluation_wall_s, converted, metrics = row
        if index == 0:
            prefix = output_dir / plan.name
        else:
            prefix = output_dir / "evaluation-grids" / f"{plan.name}-{Nr}x{Nt}"
        npz_path, json_path = write_equilibrium_product(prefix, converted)
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
        native_profiles, profile_records = _native_shape_profile_arrays(
            equilibrium
        )
        arrays.update(native_profiles)
        arrays.update(
            {
                "solve_x": np.asarray(result.x),
                "solve_raw_residual": np.asarray(result.raw),
                "solve_scaled_residual": np.asarray(result.scaled),
                "native_R_m": np.asarray(equilibrium.R),
                "native_Z_m": np.asarray(equilibrium.Z),
                "native_r": np.asarray(equilibrium.r),
                "native_theta_rad": np.asarray(equilibrium.theta),
                "native_G": np.asarray(equilibrium.G),
                "native_psin": np.asarray(equilibrium.psin),
                "native_psin_r": np.asarray(equilibrium.psin_r),
                "native_psin_rr": np.asarray(equilibrium.psin_rr),
                "native_FFn_psin": np.asarray(equilibrium.FFn_psin),
                "native_Pn_psin": np.asarray(equilibrium.Pn_psin),
                "native_F_Tm": np.asarray(equilibrium.F),
                "native_P_Pa": np.asarray(equilibrium.P),
            }
        )
        np.savez_compressed(npz_path, **arrays)
        record = json.loads(json_path.read_text(encoding="utf-8"))
        record["refinement"] = asdict(plan)
        record["evaluation_grid"] = {"Nr": Nr, "Nt": Nt}
        record["strong_gs"] = metrics
        record["evaluation_refinement"] = evaluation_refinement
        record["native_shape_profiles"] = profile_records
        record["same_physical_rz_export"] = {
            "coordinates": ["native_R_m", "native_Z_m"],
            "profile_coefficients": [
                f"profile_coeff_{name}" for name in profile_records
            ],
            "source_grid": ["native_r", "native_theta_rad"],
            "solver_state": "solve_x",
        }
        record["veq_solve"] = {
            "backend": "numba",
            "topology_x_size": topology.x_size,
            "prepare_wall_s_including_jit": prepare_wall_s,
            "solve_wall_s": solve_wall_s,
            "evaluation_wall_s": evaluation_wall_s,
            "success": bool(result.success),
            "info": int(result.info),
            "nfev": int(result.nfev),
            "njev": int(result.njev),
            "raw_norm": float(result.raw_norm),
            "scaled_norm": float(result.scaled_norm),
            "elapsed_ms": float(result.elapsed_ms),
            "preprocess_ms": float(result.preprocess_ms),
            "solver_ms": float(result.solver_ms),
            "postprocess_ms": float(result.postprocess_ms),
            "alpha": [float(item) for item in result.alpha],
            "native_G_max_abs": float(np.max(np.abs(equilibrium.G))),
            "raw_source": {
                "P_psin_unit": "Pa",
                "FF_psin_unit": "(T*m)^2",
                "P_psin_constant": -PPRIME_F,
                "FF_psin_constant": 0.0,
                "p0_input_Pa": RAW_P0_INPUT_PA,
                "Ip_constraint_A": IP_TARGET_A,
            },
        }
        json_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append((npz_path, json_path, equilibrium, converted, metrics))
    (
        npz_path,
        json_path,
        equilibrium,
        converted,
        metrics,
    ) = written[0]
    return {
        "case": plan.name,
        "npz": str(npz_path),
        "json": str(json_path),
        "success": bool(result.success),
        "projected_raw_norm": float(result.raw_norm),
        "native_G_max_abs": float(np.max(np.abs(equilibrium.G))),
        "strong_gs": metrics,
        "native_p0_Pa": float(equilibrium.p0),
        "converted_p_edge_Pa": float(np.mean(converted.p_total_Pa[-1])),
        "pressure_offset_Pa": converted.metadata.pressure_offset_Pa,
        "native_p_psi_offset_from_minus_pprime": (
            converted.native_p_psi_offset_from_minus_pprime
        ),
        "native_profile_jphi_residual_max_A_m2": (
            converted.native_profile_jphi_residual_max_A_m2
        ),
        "ampere_profile_phi_max_abs_residual_A_m2": (
            converted.ampere_profile_phi_max_abs_residual_A_m2
        ),
        "I_p_profile_A": converted.I_p_profile_A,
        "I_p_ampere_A": converted.I_p_ampere_A,
        "wall_normal_B_smooth_max_abs_T": (
            converted.wall_normal_B_smooth_max_abs_T
        ),
        "wall_normal_B_polygon_midpoint_max_abs_T": (
            converted.wall_normal_B_polygon_midpoint_max_abs_T
        ),
        "prepare_wall_s_including_jit": prepare_wall_s,
        "solve_wall_s": solve_wall_s,
        "evaluation_wall_s": converted_evaluations[0][3],
        "evaluation_refinement": evaluation_refinement,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("cases/refinements-v1"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = [_solve_one(plan, args.output) for plan in CIRCLE_REFINEMENTS]
    summary = {
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "archive_sha256": SOURCE_ARCHIVE_SHA256,
        "target": {
            "R0_m": R0_M,
            "a_m": A_M,
            "F_edge_Tm": F_EDGE_TM,
            "B0_T": B0_T,
            "p_edge_target_Pa": P_EDGE_TARGET_PA,
            "pprime_F_Pa_per_Wb_rad": PPRIME_F,
            "n_m3": DENSITY_M3,
            "ion_pressure_fraction": ION_PRESSURE_FRACTION,
            "Ip_A": IP_TARGET_A,
        },
        "elapsed_wall_s": time.perf_counter() - started,
        "cases": rows,
        "interpretation": {
            "profile_current_injected_into_reconstructed_B": False,
            "J_profile": "R*pprime_F profile target",
            "J_ampere": "curl(B)/mu0 from reconstructed B",
            "strong_gs_nonmonotonicity_attribution": (
                "not established; VEQ nonconvergence is not claimed"
            ),
        },
        "note": (
            "Raw PF p0/P_psin are not final physical edge pressure/P_psi. "
            "Native pressure and p0 are retained; converted pressure applies "
            "the declared uniform offset."
        ),
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
