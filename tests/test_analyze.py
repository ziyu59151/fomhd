"""Behavioral tests for the offline A0 convergence and mode analysis tool."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import analyze  # noqa: E402


HISTORY_COLUMNS = (
    "step,time_s,number_i,number_e,kinetic_i_J,kinetic_e_J,ion_internal_J,"
    "electron_perp_J,electron_parallel_J,magnetic_J,longitudinal_J,"
    "total_energy_J,topology_Wb,physical_flux_Wb,Lz_kg_m2_per_s,elapsed_s"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _history_row(step: int, time: float) -> list[object]:
    return [
        step,
        repr(float(time)),
        "1.0",
        "1.0",
        "0.0",
        "0.0",
        "1.0",
        "1.0",
        "1.0",
        "1.0",
        "0.0",
        "4.0",
        "1.0",
        "1.0",
        "0.0",
        "0.01",
    ]


def _write_run(
    root: Path,
    *,
    field: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    polygon_R: np.ndarray | None = None,
    polygon_Z: np.ndarray | None = None,
    extra_arrays: dict[str, np.ndarray] | None = None,
    initial_field: np.ndarray | None = None,
    case_name: str = "a0-test",
    model: str = "A0",
    units: str = "SI",
    scan_value: float = 0.1,
    scan_key: str = "grid.hR",
    grid_overrides: dict[str, object] | None = None,
    time_overrides: dict[str, object] | None = None,
    resolved_scan_value: float | None = None,
    snapshot_time: float | None = None,
    observations: list[tuple[float, complex]] | None = None,
    definition: dict[str, object] | None = None,
    definitions: list[dict[str, object]] | None = None,
    observation_rows: list[dict[str, object]] | None = None,
) -> Path:
    """Write the accepted runtime v1 layout used by the C++ solver."""

    run = root / f"run-{len(list(root.glob('run-*'))):02d}"
    (run / "inputs").mkdir(parents=True)
    (run / "snapshots").mkdir()
    source_file = run / "provenance/source/src/model.cpp"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("// measured source snapshot\n", encoding="utf-8")
    case = run / "inputs/case.toml"
    case.write_text("schema_version = 1\n", encoding="utf-8")
    geometry = run / "inputs/geometry.npz"
    if polygon_R is None:
        polygon_R = np.asarray([R[0], R[-1], R[-1], R[0]], dtype=np.float64)
    if polygon_Z is None:
        polygon_Z = np.asarray([Z[0], Z[0], Z[-1], Z[-1]], dtype=np.float64)
    np.savez(
        geometry,
        R_nodes=R,
        Z_nodes=Z,
        lcfs_R=polygon_R,
        lcfs_Z=polygon_Z,
    )
    initial = run / "inputs/initial.npz"
    K = 1
    modes = 2 * K + 1
    zeros = np.zeros_like(field, dtype=np.complex128)
    arrays: dict[str, np.ndarray] = {
        "n_i": (
            np.asarray(initial_field, dtype=np.complex128)
            if initial_field is not None
            else zeros
        ),
        "n_e": zeros.copy(),
        "mom_i": np.zeros((3, modes, len(Z), len(R)), dtype=np.complex128),
        "mom_e": np.zeros((3, modes, len(Z), len(R)), dtype=np.complex128),
        "p_i": np.zeros_like(field, dtype=np.complex128),
        "p_perp_e": np.zeros_like(field, dtype=np.complex128),
        "p_parallel_e": np.zeros_like(field, dtype=np.complex128),
        "B_R": np.zeros((modes, len(Z) + 1, len(R)), dtype=np.complex128),
        "B_phi": np.zeros((modes, len(Z) + 1, len(R) + 1), dtype=np.complex128),
        "B_Z": np.zeros((modes, len(Z), len(R) + 1), dtype=np.complex128),
    }
    np.savez(initial, **arrays)
    source = run / "inputs/source_manifest.json"
    input_hashes = {
        "case": _sha(case),
        "geometry": _sha(geometry),
        "initial": _sha(initial),
    }
    _write_json(
        source,
        {
            "provider_name": "a0-analysis-test",
            "provider_version": "v1",
            "units": "SI",
            "conversion_method": "discrete fixture with runtime identity fields",
            "input_hashes": {
                "geometry": input_hashes["geometry"],
                "initial": input_hashes["initial"],
            },
            "continuous_residual_status": "synthetic review fixture",
        },
    )
    input_hashes["source"] = _sha(source)
    resolved = {
        "schema_version": 1,
        "layout_version": 1,
        "case_name": case_name,
        "model": model,
        "units": units,
        "species": {
            "ion_id": "ion",
            "electron_id": "electron",
            "qi": 1.0,
            "qe": -1.0,
            "mi": 1.0,
            "me": 1.0,
            "gamma_i": 5.0 / 3.0,
        },
        "boundary": {
            "kind": "fixed_lcfs_static",
            "potential": 0.0,
            "version": "staircase_relative_v1",
        },
        "grid": {
            "K": K,
            "Nphi": 5,
            "NR": len(R),
            "NZ": len(Z),
            "R0": float(R[0]),
            "Z0": float(Z[0]),
            "hR": float(R[1] - R[0]),
            "hZ": float(Z[1] - Z[0]),
        },
        "time": {
            "dt": 0.01,
            "nsteps": 100,
            "start_time": 0.0,
            "initial_kind": "non_equilibrium",
        },
        "solver": {
            "backend": "cpu",
            "precision": "fp64",
            "complex_precision": "complex128",
            "cpu_threads": 1,
            "device_ordinal": 0,
            "memory_limit_bytes": 1,
            "device_memory_limit_bytes": 1,
            "rank_rtol": 1.0e-12,
            "constants": {"epsilon0_Fpm": 1.0, "mu0_Hpm": 1.0, "version": "test"},
            "references": {
                "L_m": 1.0,
                "t_s": 1.0,
                "n_i_m3": 1.0,
                "n_e_m3": 1.0,
                "u_mps": 1.0,
                "E_Vpm": 1.0,
                "B_T": 1.0,
                "J_Apm2": 1.0,
                "rho_Cpm3": 1.0,
                "p_Pa": 1.0,
            },
        },
        "budgets": {
            "domain": {
                "n_min_i": 0.0,
                "n_min_e": 0.0,
                "n_face_min_i": 0.0,
                "n_face_min_e": 0.0,
                "p_min_i": 0.0,
                "p_min_perp_e": 0.0,
                "p_min_parallel_e": 0.0,
                "B_min": 0.0,
            },
            "checks": {},
        },
        "io": {
            "geometry_file": "geometry.npz",
            "initial_file": "initial.npz",
            "source_manifest_file": "source_manifest.json",
            "output_dir": "run",
            "snapshot_stride": 1,
            "checkpoint_stride": 1,
            "max_input_bytes": 1,
            "max_uncompressed_bytes": 1,
            "max_entries": 1,
            "max_header_bytes": 1,
            "write_residual_arrays": True,
        },
        "diagnostics": {
            "inner_distance_m": 0.0,
            "alias_policy": "initial_midpoint_restart",
            "mode_observations": (
                definitions
                if definitions is not None
                else ([definition] if definition is not None else [])
            ),
            "timing_repetitions": 1,
            "timing_warmup": 0,
        },
    }
    section, key = scan_key.split(".", 1)
    resolved[section][key] = (
        scan_value if resolved_scan_value is None else resolved_scan_value
    )
    if grid_overrides:
        resolved["grid"].update(grid_overrides)
    if time_overrides:
        resolved["time"].update(time_overrides)
    if scan_key == "time.dt":
        aligned_final_time = 1.2
        resolved["time"]["nsteps"] = int(round(aligned_final_time / scan_value))
        if snapshot_time is None:
            snapshot_time = aligned_final_time
    if snapshot_time is None:
        snapshot_time = 1.0
    resolved["resolved_paths"] = {
        "geometry": str(geometry.resolve()),
        "initial": str(initial.resolve()),
        "source": str(source.resolve()),
        "output": str((run / "output").resolve()),
    }
    resolved["input_hashes"] = input_hashes
    resolved_path = run / "inputs/resolved.json"
    resolved_path.write_text(
        json.dumps(resolved, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    source_hashes = {"src/model.cpp": _sha(source_file)}
    config_hash = hashlib.sha256(
        analyze._minify_original_json(resolved_path.read_bytes())
    ).hexdigest()
    semantic_hash = hashlib.sha256(
        json.dumps(
            [case_name, model, units, resolved["species"], resolved["boundary"]],
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    geometry_hash = _sha(geometry)
    run_id = run.name
    arrays["n_i"] = np.asarray(field, dtype=np.complex128)
    if extra_arrays:
        for key_name, value in extra_arrays.items():
            arrays[key_name] = np.asarray(value, dtype=np.complex128)
    observation_rows = observation_rows or []
    if observations:
        for step, (time_value, value) in enumerate(observations, start=1):
            observation_rows.append(
                {
                    "step": step,
                    "time": time_value,
                    "definition": definition,
                    "real": float(value.real),
                    "imag": float(value.imag),
                }
            )
    accepted_pairs = sorted(
        {
            (int(row["step"]), float(row["time"])) for row in observation_rows
        }
    )
    snapshot_step = max((step for step, _ in accepted_pairs), default=0) + 1
    accepted_pairs.append((snapshot_step, float(snapshot_time)))
    snapshot = run / f"snapshots/state_{snapshot_step:010d}.npz"
    np.savez(
        snapshot,
        **arrays,
        step=np.asarray(snapshot_step, dtype=np.uint64),
        time=np.asarray(snapshot_time, dtype=np.float64),
        dt=np.asarray(resolved["time"]["dt"], dtype=np.float64),
        start_time=np.asarray(0.0, dtype=np.float64),
        accepted=np.asarray(True),
        schema_version=np.asarray(1, dtype=np.uint64),
        layout_version=np.asarray(1, dtype=np.uint64),
        model_id=np.asarray(model),
        config_hash=np.asarray(config_hash),
        semantic_hash=np.asarray(semantic_hash),
        geometry_hash=np.asarray(geometry_hash),
        operator_rule_hash=np.asarray("a" * 64),
        matrix_fingerprint=np.asarray("b" * 64),
        original_run_id=np.asarray(run_id),
    )

    with (run / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(HISTORY_COLUMNS.split(","))
        for step, time_value in accepted_pairs:
            writer.writerow(_history_row(step, time_value))
    (run / "observations.jsonl").write_text("", encoding="utf-8")
    (run / "checks.jsonl").write_text("", encoding="utf-8")
    if observation_rows:
        (run / "observations.jsonl").write_text(
            "".join(
                json.dumps(row, allow_nan=False) + "\n"
                for row in observation_rows
            ),
            encoding="utf-8",
        )

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "completed",
        "model": model,
        "units": units,
        "layout_version": 1,
        "config_hash": config_hash,
        "semantic_hash": semantic_hash,
        "geometry_hash": geometry_hash,
        "matrix_fingerprint": "b" * 64,
        "input_hashes": input_hashes,
        "build": {
            "schema_version": 1,
            "binary_sha256": "c" * 64,
            "source_commit": "d" * 40,
            "dirty": True,
            "source_sha256": "e" * 64,
            "source_files": source_hashes,
        },
        "backend": "cpu",
        "precision": "fp64/complex128",
        "source": {
            "provider_name": "a0-analysis-test",
            "provider_version": "v1",
            "units": "SI",
        },
        "last_accepted_step": snapshot_step,
        "last_accepted_time": snapshot_time,
        "history_columns": HISTORY_COLUMNS,
        "durable_checkpoint": None,
    }
    _write_json(run / "run.json", manifest)
    return run


def _descriptor(levels: list[dict[str, object]], **overrides: object) -> dict:
    value: dict[str, object] = {
        "schema_version": 1,
        "case_id": "a0-test",
        "scan_variable": "grid.hR",
        "observable": {
            "name": "n_i",
            "key": "n_i",
            "component": None,
            "mode": 0,
            "unit": "1/m^3",
            "region": {"R": [0.2, 1.8], "Z": [0.0, 1.0]},
            "norm": {
                "kind": "weighted_RMS",
                "weights": {"kind": "uniform"},
            },
        },
        "levels": levels,
        "roundoff_floor": 1.0e-13,
        "time_tolerance": 1.0e-12,
        "case_comparison": {
            "kind": "audited_refinement",
            "physical_case_id": "a0-test",
            "initial_profile_comparison": {
                "kind": "analytic_reference",
                "unit": "1/m^3",
                "evidence_sha256": "f" * 64,
            },
        },
    }
    value.update(overrides)
    if value["scan_variable"].startswith("time."):
        value["case_comparison"] = {"kind": "same_grid_same_initial"}
    return value


def _rewrite_config_hash(run: Path) -> None:
    resolved_path = run / "inputs/resolved.json"
    resolved = json.loads(resolved_path.read_text())
    config_hash = hashlib.sha256(
        analyze._minify_original_json(resolved_path.read_bytes())
    ).hexdigest()
    manifest_path = run / "run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config_hash"] = config_hash
    _write_json(manifest_path, manifest)
    for snapshot in (run / "snapshots").glob("state_*.npz"):
        with np.load(snapshot, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        arrays["config_hash"] = np.asarray(config_hash)
        np.savez(snapshot, **arrays)


class AnalysisTestCase(unittest.TestCase):
    def setUp(self) -> None:
        scratch = ROOT / "tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="test-analyze-", dir=scratch)
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _run_descriptor(self, descriptor: dict) -> tuple[Path, dict]:
        path = self.root / "comparison.json"
        _write_json(path, descriptor)
        result = analyze.analyze_convergence(path)
        return path, result

    def test_analytic_reference_recovers_known_second_order(self) -> None:
        levels = []
        for count in (5, 9, 17):
            R = np.linspace(0.2, 1.8, count)
            Z = np.linspace(0.0, 1.0, count)
            h = float(R[1] - R[0])
            RR, ZZ = np.meshgrid(R, Z)
            exact = np.exp(1j * (0.3 * RR - 0.2 * ZZ))
            error = np.full_like(exact, 0.25 * h**2 * (1.0 + 0.1j))
            field = np.zeros((3, count, count), dtype=np.complex128)
            field[1] = exact + error
            run = _write_run(
                self.root,
                field=field,
                R=R,
                Z=Z,
                scan_value=h,
                scan_key="grid.hR",
            )
            levels.append({"run": str(run), "h": h})

        descriptor = _descriptor(
            levels,
            reference={
                "kind": "plane_wave",
                "amplitude": 1.0,
                "k_R": 0.3,
                "k_Z": -0.2,
                "omega": 0.0,
                "phase": 0.0,
                "unit": "1/m^3",
            },
        )
        _, result = self._run_descriptor(descriptor)

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["reference"]["kind"], "plane_wave")
        self.assertAlmostEqual(result["orders"]["p01"]["value"], 2.0, places=10)
        self.assertAlmostEqual(result["orders"]["p12"]["value"], 2.0, places=10)
        self.assertEqual(result["scan_variable"], "grid.hR")
        self.assertEqual(len(result["levels"]), 3)
        self.assertTrue(result["levels"][0]["provenance"]["verified"])
        self.assertEqual(result["levels"][0]["time_s"], 1.0)

    def test_zero_analytic_error_is_unavailable_not_infinite_order(self) -> None:
        levels = []
        for count in (5, 9, 17):
            R = np.linspace(0.2, 1.8, count)
            Z = np.linspace(0.0, 1.0, count)
            field = np.zeros((3, count, count), dtype=np.complex128)
            field[1] = 1.0 + 0.0j
            run = _write_run(
                self.root,
                field=field,
                R=R,
                Z=Z,
                scan_value=float(R[1] - R[0]),
            )
            levels.append({"run": str(run), "h": float(R[1] - R[0])})
        descriptor = _descriptor(
            levels,
            reference={
                "kind": "plane_wave",
                "amplitude": 1.0,
                "k_R": 0.0,
                "k_Z": 0.0,
                "omega": 0.0,
                "phase": 0.0,
                "unit": "1/m^3",
            },
        )
        _, result = self._run_descriptor(descriptor)

        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertEqual(result["orders"]["p01"]["status"], "UNAVAILABLE")
        self.assertIsNone(result["orders"]["p01"]["value"])
        self.assertNotEqual(result["orders"]["p01"].get("value"), math.inf)

    def test_richardson_recovers_order_for_equal_refinement(self) -> None:
        R = np.linspace(0.2, 1.8, 9)
        Z = np.linspace(0.0, 1.0, 9)
        RR, ZZ = np.meshgrid(R, Z)
        base = np.exp(-((RR - 1.0) ** 2 + (ZZ - 0.5) ** 2))
        levels = []
        for h in (0.4, 0.2, 0.1):
            field = np.zeros((3, len(Z), len(R)), dtype=np.complex128)
            field[1] = base + (0.03 * h**2) * (1.0 - 0.2j) * (1.0 + RR)
            run = _write_run(
                self.root,
                field=field,
                R=R,
                Z=Z,
                scan_value=h,
                scan_key="time.dt",
            )
            levels.append({"run": str(run), "h": h})
        descriptor = _descriptor(
            levels,
            scan_variable="time.dt",
            reference={"kind": "none"},
            mapping={
                "kind": "bilinear",
                "comparison_grid": {"R": R.tolist(), "Z": Z.tolist()},
            },
        )
        _, result = self._run_descriptor(descriptor)

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["method"], "richardson")
        self.assertEqual(result["refinement_relation"], "equal")
        self.assertAlmostEqual(result["orders"]["richardson"]["value"], 2.0, places=8)
        self.assertEqual(result["mapping"]["interpolation_error"]["status"], "PASS")

    def test_richardson_unequal_refinement_recovers_order(self) -> None:
        R = np.linspace(0.2, 1.8, 9)
        Z = np.linspace(0.0, 1.0, 9)
        RR, ZZ = np.meshgrid(R, Z)
        base = np.exp(1j * (0.2 * RR + 0.1 * ZZ))
        levels = []
        for h in (0.4, 0.2, 0.15):
            field = np.zeros((3, len(Z), len(R)), dtype=np.complex128)
            field[1] = base + 0.05 * h**2 * (1.0 + 0.5j)
            run = _write_run(
                self.root,
                field=field,
                R=R,
                Z=Z,
                scan_value=h,
                scan_key="time.dt",
            )
            levels.append({"run": str(run), "h": h})
        descriptor = _descriptor(
            levels,
            scan_variable="time.dt",
            reference={"kind": "none"},
            order_interval=[0.25, 6.0],
            mapping={
                "kind": "bilinear",
                "comparison_grid": {"R": R.tolist(), "Z": Z.tolist()},
            },
        )
        _, result = self._run_descriptor(descriptor)

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["method"], "richardson")
        self.assertEqual(result["refinement_relation"], "unequal")
        self.assertAlmostEqual(result["orders"]["richardson"]["value"], 2.0, places=7)
        self.assertGreater(result["richardson_estimate"]["leading_error"], 0.0)

    def test_unequal_richardson_is_invariant_to_si_refinement_scale(self) -> None:
        R = np.linspace(0.2, 1.8, 9)
        Z = np.linspace(0.0, 1.0, 9)
        orders = []
        for scale in (1.0, 1.0e-9):
            levels = []
            for unscaled_h in (0.4, 0.2, 0.15):
                h = unscaled_h * scale
                field = np.zeros((3, len(Z), len(R)), dtype=np.complex128)
                field[1] = 0.05 * h**2
                run = _write_run(
                    self.root,
                    field=field,
                    R=R,
                    Z=Z,
                    scan_value=h,
                    scan_key="time.dt",
                )
                levels.append({"run": str(run), "h": h})
            descriptor = _descriptor(
                levels,
                scan_variable="time.dt",
                reference={"kind": "none"},
                order_interval=[0.25, 6.0],
                roundoff_floor=0.0,
            )
            result = analyze.analyze_convergence(descriptor)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["refinement_relation"], "unequal")
            orders.append(result["orders"]["richardson"]["value"])

        self.assertAlmostEqual(orders[0], 2.0, places=7)
        self.assertAlmostEqual(orders[1], 2.0, places=7)
        self.assertAlmostEqual(orders[0], orders[1], places=10)

    def test_weighted_norm_names_have_distinct_definitions(self) -> None:
        R = np.linspace(0.2, 1.8, 9)
        Z = np.linspace(0.0, 1.0, 9)
        levels = []
        for h in (0.4, 0.2, 0.1):
            field = np.zeros((3, len(Z), len(R)), dtype=np.complex128)
            field[1] = 2.0
            run = _write_run(
                self.root,
                field=field,
                R=R,
                Z=Z,
                scan_value=h,
                scan_key="time.dt",
            )
            levels.append({"run": str(run), "h": h})
        reference = {
            "kind": "constant",
            "amplitude": 0.0,
            "phase": 0.0,
            "unit": "1/m^3",
        }
        physical = _descriptor(
            levels,
            scan_variable="time.dt",
            reference=reference,
        )
        physical["observable"]["norm"]["kind"] = "weighted_L2"
        _, physical_result = self._run_descriptor(physical)
        self.assertAlmostEqual(physical_result["errors"][0], 16.0, places=12)
        self.assertEqual(
            physical_result["observable"]["norm"]["definition"],
            "sqrt(sum(weights * abs(value)^2))",
        )

        normalized = _descriptor(
            levels,
            scan_variable="time.dt",
            reference=reference,
        )
        normalized["observable"]["norm"]["kind"] = "weighted_RMS"
        _, normalized_result = self._run_descriptor(normalized)
        self.assertAlmostEqual(normalized_result["errors"][0], 2.0, places=12)
        self.assertEqual(
            normalized_result["observable"]["norm"]["definition"],
            "sqrt(sum(weights * abs(value)^2) / sum(weights))",
        )

    def test_relative_observable_scale_guards_near_zero_dt_differences(self) -> None:
        R = np.linspace(0.2, 1.8, 9)
        Z = np.linspace(0.0, 1.0, 9)
        base = 4.0e-32
        levels = []
        for h, offset in ((4.0e-10, 0.0), (2.0e-10, 1.0e-48), (1.5e-10, 2.0e-48)):
            field = np.zeros((3, len(Z), len(R)), dtype=np.complex128)
            field[1] = base + offset
            run = _write_run(
                self.root,
                field=field,
                R=R,
                Z=Z,
                scan_value=h,
                scan_key="time.dt",
            )
            levels.append({"run": str(run), "h": h})
        descriptor = _descriptor(
            levels,
            scan_variable="time.dt",
            reference={"kind": "none"},
            observable_scale=base,
            roundoff_relative=1.0e-12,
            roundoff_floor=0.0,
        )
        _, result = self._run_descriptor(descriptor)

        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertIn("roundoff", result["reason"].lower())
        self.assertEqual(result["roundoff_guard"]["status"], "PASS")
        self.assertAlmostEqual(
            result["roundoff_guard"]["effective_norm"], base * 1.0e-12
        )

    def test_roundoff_sized_differences_are_unavailable(self) -> None:
        R = np.linspace(0.2, 1.8, 9)
        Z = np.linspace(0.0, 1.0, 9)
        levels = []
        for h in (0.4, 0.2, 0.1):
            field = np.zeros((3, len(Z), len(R)), dtype=np.complex128)
            field[1] = (1.0e-16 * h) * np.ones((len(Z), len(R)))
            run = _write_run(
                self.root,
                field=field,
                R=R,
                Z=Z,
                scan_value=h,
                scan_key="time.dt",
            )
            levels.append({"run": str(run), "h": h})
        descriptor = _descriptor(
            levels,
            scan_variable="time.dt",
            reference={"kind": "none"},
            roundoff_floor=1.0e-12,
            mapping={
                "kind": "bilinear",
                "comparison_grid": {"R": R.tolist(), "Z": Z.tolist()},
            },
        )
        _, result = self._run_descriptor(descriptor)

        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertIn("roundoff", result["reason"].lower())

    def test_nonpositive_richardson_order_is_warn_not_pass(self) -> None:
        R = np.linspace(0.2, 1.8, 9)
        Z = np.linspace(0.0, 1.0, 9)
        levels = []
        for h, offset in ((0.4, 0.0), (0.2, 0.1), (0.1, 0.4)):
            field = np.zeros((3, len(Z), len(R)), dtype=np.complex128)
            field[1] = offset
            run = _write_run(
                self.root,
                field=field,
                R=R,
                Z=Z,
                scan_value=h,
                scan_key="time.dt",
            )
            levels.append({"run": str(run), "h": h})
        descriptor = _descriptor(
            levels,
            scan_variable="time.dt",
            reference={"kind": "none"},
        )
        _, result = self._run_descriptor(descriptor)

        self.assertEqual(result["status"], "WARN")
        self.assertEqual(result["orders"]["richardson"]["status"], "WARN")
        self.assertLess(result["orders"]["richardson"]["value"], 0.0)
        self.assertEqual(result["richardson_estimate"]["status"], "UNAVAILABLE")
        self.assertTrue(any("nonpositive" in item for item in result["warnings"]))

    def test_different_grids_without_mapping_fail_loudly(self) -> None:
        levels = []
        for count, h in ((5, 0.4), (9, 0.2), (17, 0.1)):
            R = np.linspace(0.2, 1.8, count)
            Z = np.linspace(0.0, 1.0, count)
            field = np.zeros((3, count, count), dtype=np.complex128)
            field[1] = h * (R[None, :] + Z[:, None])
            run = _write_run(
                self.root,
                field=field,
                R=R,
                Z=Z,
                scan_value=h,
                scan_key="grid.hR",
            )
            levels.append({"run": str(run), "h": h})
        descriptor = _descriptor(levels, reference={"kind": "none"})
        with self.assertRaisesRegex(analyze.AnalysisError, "mapping"):
            analyze.analyze_convergence(descriptor)

    def test_case_unit_and_time_mismatches_are_rejected(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        field = np.zeros((3, 5, 5), dtype=np.complex128)
        good = _write_run(
            self.root,
            field=field,
            R=R,
            Z=Z,
            scan_value=0.4,
            snapshot_time=1.0,
        )
        wrong_unit = _write_run(
            self.root,
            field=field,
            R=R,
            Z=Z,
            units="CGS",
            scan_value=0.2,
            snapshot_time=1.0,
        )
        wrong_case = _write_run(
            self.root,
            field=field,
            R=R,
            Z=Z,
            case_name="other",
            scan_value=0.1,
            snapshot_time=1.0,
        )
        descriptor = _descriptor(
            [
                {"run": str(good), "h": 0.4},
                {"run": str(wrong_unit), "h": 0.2},
                {"run": str(wrong_case), "h": 0.1},
            ],
            reference={"kind": "none"},
        )
        with self.assertRaisesRegex(analyze.AnalysisError, "units"):
            analyze.analyze_convergence(descriptor)

        wrong_unit_manifest = json.loads((wrong_unit / "run.json").read_text())
        wrong_unit_manifest["units"] = "SI"
        _write_json(wrong_unit / "run.json", wrong_unit_manifest)
        wrong_unit_resolved = json.loads(
            (wrong_unit / "inputs/resolved.json").read_text()
        )
        wrong_unit_resolved["units"] = "SI"
        _write_json(wrong_unit / "inputs/resolved.json", wrong_unit_resolved)
        _rewrite_config_hash(wrong_unit)
        with self.assertRaisesRegex(analyze.AnalysisError, "case"):
            analyze.analyze_convergence(descriptor)

    def test_requires_three_levels_and_one_scan_variable(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        runs = []
        for h in (0.4, 0.2):
            field = np.zeros((3, 5, 5), dtype=np.complex128)
            runs.append(
                _write_run(
                    self.root,
                    field=field,
                    R=R,
                    Z=Z,
                    scan_value=h,
                    scan_key="time.dt",
                )
            )
        descriptor = _descriptor(
            [{"run": str(run), "h": h} for run, h in zip(runs, (0.4, 0.2))]
        )
        with self.assertRaisesRegex(analyze.AnalysisError, "three"):
            analyze.analyze_convergence(descriptor)

        descriptor["scan_variable"] = ""
        with self.assertRaisesRegex(analyze.AnalysisError, "scan variable"):
            analyze.analyze_convergence(descriptor)

    def test_exactly_three_levels_fail_cleanly_for_four(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        levels = []
        for h in (0.4, 0.2, 0.1, 0.05):
            field = np.zeros((3, 5, 5), dtype=np.complex128)
            run = _write_run(
                self.root,
                field=field,
                R=R,
                Z=Z,
                scan_value=h,
                scan_key="time.dt",
            )
            levels.append({"run": str(run), "h": h})
        descriptor = _descriptor(
            levels,
            scan_variable="time.dt",
            reference={"kind": "none"},
        )
        with self.assertRaisesRegex(analyze.AnalysisError, "exactly three"):
            analyze.analyze_convergence(descriptor)

    def test_scan_value_must_match_resolved_configuration(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        levels = []
        for h in (0.4, 0.2, 0.1):
            field = np.zeros((3, 5, 5), dtype=np.complex128)
            run = _write_run(
                self.root,
                field=field,
                R=R,
                Z=Z,
                scan_value=h,
                resolved_scan_value=0.01,
                scan_key="time.dt",
            )
            levels.append({"run": str(run), "h": h})
        descriptor = _descriptor(
            levels,
            scan_variable="time.dt",
            reference={"kind": "none"},
        )
        with self.assertRaisesRegex(analyze.AnalysisError, "scan"):
            analyze.analyze_convergence(descriptor)

    def test_tampered_resolved_configuration_fails_replayed_hash(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        run = _write_run(
            self.root,
            field=np.zeros((3, 5, 5), dtype=np.complex128),
            R=R,
            Z=Z,
        )
        resolved_path = run / "inputs/resolved.json"
        resolved = json.loads(resolved_path.read_text())
        resolved["grid"]["K"] = 137
        _write_json(resolved_path, resolved)
        with self.assertRaisesRegex(analyze.AnalysisError, "config_hash"):
            analyze.load_run(run)

    def test_resolved_file_sha_anchor_is_serializer_portable(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        run = _write_run(
            self.root,
            field=np.zeros((3, 5, 5), dtype=np.complex128),
            R=R,
            Z=Z,
        )
        manifest_path = run / "run.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["resolved_config_sha256"] = _sha(
            run / "inputs/resolved.json"
        )
        manifest["config_hash"] = "0" * 64
        _write_json(manifest_path, manifest)
        loaded = analyze.load_run(run)
        self.assertEqual(
            loaded.config_hash_verification["method"],
            "resolved_file_sha256",
        )
        self.assertTrue(
            loaded.config_hash_verification["portable_across_json_serializers"]
        )

    def test_config_hash_preserves_original_json_numeric_and_string_spelling(self) -> None:
        raw = (
            b'{\n'
            b'  "big": 1e+20,\n'
            b'  "small": 1e-07,\n'
            b'  "signed_zero": -0.0,\n'
            b'  "escaped": "outside-inside \\"quoted\\" and \\\\ slash  intact",\n'
            b'  "unicode": "\xce\xbc  preserves  spaces"\n'
            b'}\n'
        )
        expected = (
            '{"big":1e+20,"small":1e-07,"signed_zero":-0.0,'
            '"escaped":"outside-inside \\"quoted\\" and \\\\ slash  intact",'
            '"unicode":"\u03bc  preserves  spaces"}'
        ).encode("utf-8")
        self.assertEqual(analyze._minify_original_json(raw), expected)

        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        run = _write_run(
            self.root,
            field=np.zeros((3, 5, 5), dtype=np.complex128),
            R=R,
            Z=Z,
        )
        resolved_path = run / "inputs/resolved.json"
        original = resolved_path.read_bytes().rstrip()
        self.assertTrue(original.endswith(b"}"))
        serializer_body = raw.strip()[1:-1]
        extended = (
            original[:-1]
            + b',\n  "serializer_test": {'
            + serializer_body
            + b'}\n}\n'
        )
        resolved_path.write_bytes(extended)
        config_hash = hashlib.sha256(
            analyze._minify_original_json(extended)
        ).hexdigest()
        manifest_path = run / "run.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["config_hash"] = config_hash
        _write_json(manifest_path, manifest)
        loaded = analyze.load_run(run)
        self.assertEqual(
            loaded.config_hash_verification["method"],
            "lexical_minification_of_original_utf8",
        )

    def test_time_scan_requires_identical_initial_content(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        field = np.zeros((3, 5, 5), dtype=np.complex128)
        levels = []
        for index, h in enumerate((0.4, 0.2, 0.1)):
            initial = np.zeros((3, 5, 5), dtype=np.complex128)
            initial[1] = float(index)
            run = _write_run(
                self.root,
                field=field,
                initial_field=initial,
                R=R,
                Z=Z,
                scan_value=h,
                scan_key="time.dt",
            )
            levels.append({"run": str(run), "h": h})
        descriptor = _descriptor(
            levels,
            scan_variable="time.dt",
            reference={"kind": "none"},
        )
        with self.assertRaisesRegex(analyze.AnalysisError, "initial"):
            analyze.analyze_convergence(descriptor)

    def test_time_scan_rejects_nonscanned_nphi_drift(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        levels = []
        for h, nphi in ((0.2, 5), (0.1, 9), (0.05, 13)):
            run = _write_run(
                self.root,
                field=np.zeros((3, 5, 5), dtype=np.complex128),
                R=R,
                Z=Z,
                scan_value=h,
                scan_key="time.dt",
                grid_overrides={"Nphi": nphi},
            )
            levels.append({"run": str(run), "h": h})
        descriptor = _descriptor(
            levels,
            scan_variable="time.dt",
            reference={"kind": "none"},
        )
        with self.assertRaisesRegex(analyze.AnalysisError, "Nphi"):
            analyze.analyze_convergence(descriptor)

    def test_time_scan_rejects_start_time_and_initial_kind_drift(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        field = np.zeros((3, 5, 5), dtype=np.complex128)
        for change, pattern in (
            ({"start_time": 0.0}, "start_time"),
            ({"start_time": 0.2}, "start_time"),
            ({"start_time": 0.4}, "start_time"),
        ):
            levels = []
            for h, start_time in zip((0.2, 0.1, 0.05), (0.0, 0.2, 0.4)):
                run = _write_run(
                    self.root,
                    field=field,
                    R=R,
                    Z=Z,
                    scan_value=h,
                    scan_key="time.dt",
                    time_overrides={"start_time": start_time},
                )
                levels.append({"run": str(run), "h": h})
            descriptor = _descriptor(
                levels,
                scan_variable="time.dt",
                reference={"kind": "none"},
            )
            with self.subTest(change=change):
                with self.assertRaisesRegex(analyze.AnalysisError, pattern):
                    analyze.analyze_convergence(descriptor)

        levels = []
        for h, kind in (
            (0.2, "equilibrium"),
            (0.1, "non_equilibrium"),
            (0.05, "equilibrium"),
        ):
            run = _write_run(
                self.root,
                field=field,
                R=R,
                Z=Z,
                scan_value=h,
                scan_key="time.dt",
                time_overrides={"initial_kind": kind},
            )
            levels.append({"run": str(run), "h": h})
        descriptor = _descriptor(
            levels,
            scan_variable="time.dt",
            reference={"kind": "none"},
        )
        with self.assertRaisesRegex(analyze.AnalysisError, "initial_kind"):
            analyze.analyze_convergence(descriptor)

    def test_time_scan_rejects_k_drift(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        field = np.zeros((5, 5, 5), dtype=np.complex128)
        levels = []
        for h, K, nphi in ((0.2, 0, 5), (0.1, 1, 9), (0.05, 2, 13)):
            run = _write_run(
                self.root,
                field=field,
                R=R,
                Z=Z,
                scan_value=h,
                scan_key="time.dt",
                grid_overrides={"K": K, "Nphi": nphi},
            )
            levels.append({"run": str(run), "h": h})
        descriptor = _descriptor(
            levels,
            scan_variable="time.dt",
            reference={"kind": "none"},
        )
        descriptor["observable"]["mode"] = 0
        with self.assertRaisesRegex(analyze.AnalysisError, "K"):
            analyze.analyze_convergence(descriptor)

    def test_grid_scan_rejects_nonscanned_dt_drift(self) -> None:
        levels = []
        for count, h, dt in ((5, 0.4, 0.01), (9, 0.2, 0.02), (17, 0.1, 0.03)):
            R = np.linspace(0.2, 1.8, count)
            Z = np.linspace(0.0, 1.0, count)
            field = np.zeros((3, count, count), dtype=np.complex128)
            field[1] = 1.0
            run = _write_run(
                self.root,
                field=field,
                R=R,
                Z=Z,
                scan_value=h,
                time_overrides={"dt": dt},
            )
            levels.append({"run": str(run), "h": h})
        descriptor = _descriptor(
            levels,
            reference={
                "kind": "constant",
                "amplitude": 1.0,
                "phase": 0.0,
                "unit": "1/m^3",
            },
        )
        with self.assertRaisesRegex(analyze.AnalysisError, "time.dt"):
            analyze.analyze_convergence(descriptor)

    def test_different_grids_require_audited_physical_case_declaration(self) -> None:
        levels = []
        for count, h in ((5, 0.4), (9, 0.2), (17, 0.1)):
            R = np.linspace(0.2, 1.8, count)
            Z = np.linspace(0.0, 1.0, count)
            field = np.zeros((3, count, count), dtype=np.complex128)
            field[1] = 1.0
            run = _write_run(
                self.root,
                field=field,
                R=R,
                Z=Z,
                scan_value=h,
            )
            levels.append({"run": str(run), "h": h})
        descriptor = _descriptor(
            levels,
            reference={
                "kind": "constant",
                "amplitude": 1.0,
                "phase": 0.0,
                "unit": "1/m^3",
            },
        )
        descriptor["case_comparison"] = {"kind": "same_grid_same_initial"}
        with self.assertRaisesRegex(analyze.AnalysisError, "audited"):
            analyze.analyze_convergence(descriptor)

    def test_active_geometry_mask_excludes_inactive_padding(self) -> None:
        R = np.linspace(0.2, 1.0, 5)
        Z = np.linspace(0.0, 0.8, 5)
        right = R[3] + 0.5 * (R[1] - R[0])
        top = Z[-1] + 0.5 * (Z[1] - Z[0])
        polygon_R = np.asarray([R[0], right, right, R[0]])
        polygon_Z = np.asarray([Z[0], Z[0], top, top])
        levels = []
        for h in (0.04, 0.02, 0.01):
            field = np.zeros((3, 5, 5), dtype=np.complex128)
            field[1, :, :4] = 1.0
            field[1, :, 4] = 0.0
            run = _write_run(
                self.root,
                field=field,
                R=R,
                Z=Z,
                polygon_R=polygon_R,
                polygon_Z=polygon_Z,
                scan_value=h,
                scan_key="time.dt",
            )
            levels.append({"run": str(run), "h": h})
        descriptor = _descriptor(
            levels,
            scan_variable="time.dt",
            reference={
                "kind": "constant",
                "amplitude": 0.0,
                "phase": 0.0,
                "unit": "1/m^3",
            },
        )
        descriptor["observable"]["norm"] = {
            "kind": "weighted_RMS",
            "weights": {"kind": "a0_quadrature"},
        }
        _, result = self._run_descriptor(descriptor)
        self.assertAlmostEqual(result["errors"][0], 1.0, places=14)
        self.assertEqual(result["levels"][0]["norm"]["points"], 20)

    def test_real_b_staggered_layouts_use_active_coordinates_and_units(self) -> None:
        R = np.linspace(0.2, 1.0, 5)
        Z = np.linspace(0.0, 0.8, 5)
        right = R[3] + 0.5 * (R[1] - R[0])
        top = Z[-1] + 0.5 * (Z[1] - Z[0])
        polygon_R = np.asarray([R[0], right, right, R[0]])
        polygon_Z = np.asarray([Z[0], Z[0], top, top])
        B_R = np.zeros((3, 6, 5), dtype=np.complex128)
        B_phi = np.zeros((3, 6, 6), dtype=np.complex128)
        B_Z = np.zeros((3, 5, 6), dtype=np.complex128)
        B_R[1, :, :4] = 2.0
        B_phi[1, :, :5] = 2.0
        B_Z[1, :, :5] = 2.0
        levels = []
        for h in (0.04, 0.02, 0.01):
            run = _write_run(
                self.root,
                field=np.zeros((3, 5, 5), dtype=np.complex128),
                R=R,
                Z=Z,
                polygon_R=polygon_R,
                polygon_Z=polygon_Z,
                extra_arrays={"B_R": B_R, "B_phi": B_phi, "B_Z": B_Z},
                scan_value=h,
                scan_key="time.dt",
            )
            levels.append({"run": str(run), "h": h})
        for key in ("B_R", "B_phi", "B_Z"):
            descriptor = _descriptor(
                levels,
                scan_variable="time.dt",
                reference={
                    "kind": "constant",
                    "amplitude": 0.0,
                    "phase": 0.0,
                    "unit": "T",
                },
            )
            descriptor["observable"].update(
                {"key": key, "component": None, "unit": "T"}
            )
            descriptor["observable"]["norm"] = {
                "kind": "weighted_L2",
                "weights": {"kind": "a0_quadrature"},
            }
            _, result = self._run_descriptor(descriptor)
            self.assertGreater(result["errors"][0], 0.0)
            self.assertIn("m^(3/2)", result["observable"]["norm"]["unit"])
            self.assertEqual(
                result["levels"][0]["snapshot"]["layout_role"],
                {
                    "B_R": "radial_x_zface",
                    "B_phi": "rface_x_zface",
                    "B_Z": "rface_x_zcell",
                }[key],
            )

    def test_snapshot_time_mismatch_is_rejected(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        field = np.zeros((3, 5, 5), dtype=np.complex128)
        runs = []
        for h, time_value in ((0.4, 1.0), (0.2, 1.1), (0.1, 1.2)):
            runs.append(
                _write_run(
                    self.root,
                    field=field,
                    R=R,
                    Z=Z,
                    scan_value=h,
                    scan_key="time.dt",
                    snapshot_time=time_value,
                )
            )
        descriptor = _descriptor(
            [{"run": str(run), "h": h} for run, h in zip(runs, (0.4, 0.2, 0.1))],
            scan_variable="time.dt",
            reference={"kind": "none"},
        )
        with self.assertRaisesRegex(analyze.AnalysisError, "same physical output time"):
            analyze.analyze_convergence(descriptor)

    def test_mode_fit_uses_exp_minus_i_omega_t_and_growth(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        definition = {
            "field": "B",
            "component": "phi",
            "mode": 1,
            "entity_key": [1, 1],
        }
        times = np.arange(0.0, 1.01, 0.01)
        omega = 2.0 * math.pi * 2.5
        gamma = 0.4
        phase0 = 0.3
        values = 2.0 * np.exp(gamma * times) * np.exp(
            1j * (phase0 - omega * times)
        )
        run = _write_run(
            self.root,
            field=np.zeros((3, 5, 5), dtype=np.complex128),
            R=R,
            Z=Z,
            observations=list(zip(times, values)),
            definition=definition,
        )

        result = analyze.fit_mode(
            run,
            t_start=0.1,
            t_end=0.9,
            amplitude_floor=1.0e-8,
            target_frequency_hz=2.5,
        )

        self.assertEqual(result["status"], "PASS")
        self.assertAlmostEqual(result["angular_frequency_rad_s"], omega, places=9)
        self.assertAlmostEqual(result["growth_rate_per_s"], gamma, places=9)
        self.assertAlmostEqual(result["frequency_hz"], 2.5, places=9)
        self.assertGreater(result["amplitude"], 0.0)
        self.assertEqual(result["phase_convention"], "exp(-i*omega*t)")
        self.assertEqual(result["fit_window"], {"start_s": 0.1, "end_s": 0.9})
        self.assertEqual(len(result["samples"]["time_s"]), 81)
        self.assertAlmostEqual(
            result["fft_crosscheck"]["physical_frequency_hz"], 2.5, delta=0.2
        )
        self.assertFalse(any("multi" in item for item in result["warnings"]))
        self.assertFalse(any("nonstation" in item for item in result["warnings"]))

    def test_mode_fit_si_time_offset_uses_scaled_slope_uncertainty(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        definition = {
            "field": "B",
            "component": "phi",
            "mode": 1,
            "entity_key": [1, 1],
        }
        relative_times = np.arange(101) * 1.0e-10
        omega = 2.0 * math.pi * 1.0e7
        growth = 1.0e6
        values = 2.0 * np.exp(growth * relative_times) * np.exp(
            -1j * omega * relative_times
        )
        perturbation = 1.0e-5 * (
            np.sin(np.arange(relative_times.size))
            + 1j * np.cos(1.7 * np.arange(relative_times.size))
        )
        results = []
        for offset in (0.0, 1.0e-8):
            times = offset + relative_times
            run = _write_run(
                self.root,
                field=np.zeros((3, 5, 5), dtype=np.complex128),
                R=R,
                Z=Z,
                observations=list(zip(times, values + perturbation)),
                definition=definition,
                snapshot_time=float(times[-1] + 1.0e-10),
            )
            result = analyze.fit_mode(
                run,
                t_start=float(times[0]),
                t_end=float(times[-1]),
                amplitude_floor=1.0e-9,
                target_frequency_hz=1.0e7,
            )
            self.assertAlmostEqual(result["frequency_hz"], 1.0e7, delta=1.0e2)
            self.assertAlmostEqual(result["growth_rate_per_s"], growth, delta=1.0e2)
            self.assertTrue(
                math.isfinite(result["uncertainty"]["frequency_hz_1sigma"])
            )
            results.append(result)
        self.assertTrue(
            math.isclose(
                results[0]["uncertainty"]["frequency_hz_1sigma"],
                results[1]["uncertainty"]["frequency_hz_1sigma"],
                rel_tol=1.0e-10,
                abs_tol=1.0e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                results[0]["uncertainty"]["growth_rate_per_s_1sigma"],
                results[1]["uncertainty"]["growth_rate_per_s_1sigma"],
                rel_tol=1.0e-10,
                abs_tol=1.0e-12,
            )
        )

    def test_observations_must_match_accepted_history_step_and_time(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        definition = {
            "field": "B",
            "component": "phi",
            "mode": 1,
            "entity_key": [1, 1],
        }
        times = np.linspace(0.0, 1.0, 21)
        run = _write_run(
            self.root,
            field=np.zeros((3, 5, 5), dtype=np.complex128),
            R=R,
            Z=Z,
            observations=list(zip(times, np.exp(-1j * times))),
            definition=definition,
            snapshot_time=1.1,
        )
        with (run / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(HISTORY_COLUMNS.split(","))
            writer.writerow(_history_row(22, 1.1))
        with self.assertRaisesRegex(analyze.AnalysisError, "accepted history"):
            analyze.fit_mode(
                run,
                t_start=0.0,
                t_end=1.0,
                amplitude_floor=1.0e-9,
                target_frequency_hz=0.1,
            )

    def test_multiple_projection_definitions_may_share_times(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        first = {
            "field": "B",
            "component": "phi",
            "mode": 1,
            "entity_key": [1, 1],
        }
        second = {
            "field": "B",
            "component": "R",
            "mode": 1,
            "entity_key": [1, 1],
        }
        times = np.linspace(0.0, 1.0, 41)
        rows = []
        for index, time_value in enumerate(times):
            for definition, multiplier in ((first, 1.0), (second, 2.0)):
                value = multiplier * np.exp(-1j * 2.0 * math.pi * time_value)
                rows.append(
                    {
                        "step": index + 1,
                        "time": float(time_value),
                        "definition": definition,
                        "real": float(value.real),
                        "imag": float(value.imag),
                    }
                )
        run = _write_run(
            self.root,
            field=np.zeros((3, 5, 5), dtype=np.complex128),
            R=R,
            Z=Z,
            definitions=[first, second],
            observation_rows=rows,
            snapshot_time=1.1,
        )
        result = analyze.fit_mode(
            run,
            field="B",
            component="R",
            mode=1,
            entity_key=[1, 1],
            t_start=0.0,
            t_end=1.0,
            amplitude_floor=1.0e-9,
            target_frequency_hz=1.0,
        )
        self.assertAlmostEqual(result["frequency_hz"], 1.0, places=10)

    def test_mode_fit_warns_for_multitone(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        definition = {
            "field": "n_i",
            "component": "scalar",
            "mode": 1,
            "entity_key": [1, 1],
        }
        times = np.arange(0.0, 2.01, 0.01)
        omega = 2.0 * math.pi * 2.5
        values = np.exp(-1j * omega * times) + 0.45 * np.exp(
            -1j * 2.0 * math.pi * 5.1 * times
        )
        run = _write_run(
            self.root,
            field=np.zeros((3, 5, 5), dtype=np.complex128),
            R=R,
            Z=Z,
            observations=list(zip(times, values)),
            definition=definition,
        )
        result = analyze.fit_mode(
            run,
            t_start=0.2,
            t_end=1.8,
            amplitude_floor=1.0e-9,
            target_frequency_hz=2.5,
        )
        self.assertEqual(result["status"], "WARN")
        self.assertTrue(any("multi" in item for item in result["warnings"]))

    def test_mode_fit_warns_for_nonstationarity(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        definition = {
            "field": "p_i",
            "component": "scalar",
            "mode": -1,
            "entity_key": [2, 2],
        }
        times = np.arange(0.0, 1.21, 0.01)
        omega = 2.0 * math.pi * 2.0
        growth = np.where(times < 0.6, 0.05, 0.8)
        envelope = np.exp(np.cumsum(growth) * 0.01)
        values = envelope * np.exp(-1j * omega * times)
        run = _write_run(
            self.root,
            field=np.zeros((3, 5, 5), dtype=np.complex128),
            R=R,
            Z=Z,
            observations=list(zip(times, values)),
            definition=definition,
        )
        result = analyze.fit_mode(
            run,
            t_start=0.05,
            t_end=1.15,
            amplitude_floor=1.0e-9,
            target_frequency_hz=2.0,
        )
        self.assertEqual(result["status"], "WARN")
        self.assertTrue(any("nonstation" in item for item in result["warnings"]))

    def test_mode_fit_requires_resolved_cadence_and_explicit_window(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        definition = {
            "field": "B",
            "component": "R",
            "mode": 0,
            "entity_key": [0, 0],
        }
        times = np.arange(0.0, 0.21, 0.01)
        values = np.exp(-2j * math.pi * 60.0 * times)
        run = _write_run(
            self.root,
            field=np.zeros((3, 5, 5), dtype=np.complex128),
            R=R,
            Z=Z,
            observations=list(zip(times, values)),
            definition=definition,
        )
        with self.assertRaisesRegex(analyze.AnalysisError, "Nyquist"):
            analyze.fit_mode(
                run,
                t_start=0.0,
                t_end=0.2,
                amplitude_floor=1.0e-9,
                target_frequency_hz=60.0,
            )
        with self.assertRaisesRegex(analyze.AnalysisError, "window"):
            analyze.fit_mode(
                run,
                t_start=None,
                t_end=0.2,
                amplitude_floor=1.0e-9,
                target_frequency_hz=1.0,
            )

    def test_mode_fit_rejects_malformed_observations_and_wrong_definition(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        definition = {
            "field": "B",
            "component": "phi",
            "mode": 1,
            "entity_key": [1, 1],
        }
        times = np.linspace(0.0, 1.0, 21)
        run = _write_run(
            self.root,
            field=np.zeros((3, 5, 5), dtype=np.complex128),
            R=R,
            Z=Z,
            observations=list(zip(times, np.exp(-1j * times))),
            definition=definition,
        )
        observations_path = run / "observations.jsonl"
        original = observations_path.read_text()
        observations_path.write_text(original + "{broken\n", encoding="utf-8")
        with self.assertRaisesRegex(analyze.AnalysisError, "observations"):
            analyze.fit_mode(
                run,
                t_start=0.0,
                t_end=1.0,
                amplitude_floor=1.0e-9,
                target_frequency_hz=0.1,
            )

        observations_path.write_text(original, encoding="utf-8")
        with self.assertRaisesRegex(analyze.AnalysisError, "definition"):
            analyze.fit_mode(
                run,
                field="B",
                component="R",
                mode=1,
                entity_key=[1, 1],
                t_start=0.0,
                t_end=1.0,
                amplitude_floor=1.0e-9,
                target_frequency_hz=0.1,
            )

    def test_reports_do_not_overwrite_inputs_or_existing_results(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        field = np.zeros((3, 5, 5), dtype=np.complex128)
        runs = []
        for h in (0.4, 0.2, 0.1):
            runs.append(
                _write_run(
                    self.root,
                    field=field,
                    R=R,
                    Z=Z,
                    scan_value=h,
                    scan_key="time.dt",
                )
            )
        descriptor = _descriptor(
            [{"run": str(run), "h": h} for run, h in zip(runs, (0.4, 0.2, 0.1))],
            scan_variable="time.dt",
            reference={"kind": "none"},
        )
        descriptor_path = self.root / "comparison.json"
        _write_json(descriptor_path, descriptor)
        before = {
            path: _sha(path)
            for run in runs
            for path in (
                run / "run.json",
                run / "history.csv",
                run / "snapshots/state_0000000001.npz",
            )
        }
        output = self.root / "result.json"
        analyze.analyze_convergence(descriptor_path, output=output)
        after = {path: _sha(path) for path in before}
        self.assertEqual(before, after)
        first = output.read_bytes()
        with self.assertRaises(FileExistsError):
            analyze.analyze_convergence(descriptor_path, output=output)
        self.assertEqual(output.read_bytes(), first)

    def test_polygon_orientation_matches_certified_solver(self) -> None:
        pr = np.array([0.5380638578276709, 1.4057461350441045, 1.4057461350441045])
        pz = np.array([-0.48763668207452715, 0.7041696714041104, -0.48763668207452715])
        self.assertFalse(analyze._inside_polygon(
            0.9028076671160615, 0.013357749546752173, pr, pz))
        with self.assertRaisesRegex(analyze.AnalysisError, "GeometryAmbiguous"):
            analyze._inside_polygon(0.5, 0.5, np.array([0., 1., 1.]), np.array([0., 1., 0.]))

    def test_unsupported_scan_variable_is_rejected(self) -> None:
        levels = []
        for count in (100, 50, 25):
            run = _write_run(
                self.root, field=np.ones((3, 5, 5), dtype=complex),
                R=np.linspace(.2, 1.8, 5), Z=np.linspace(0., 1., 5),
                scan_value=count, scan_key="time.nsteps")
            levels.append({"run": str(run), "h": count})
        descriptor = _descriptor(levels, reference={"kind": "none"})
        descriptor["scan_variable"] = "time.nsteps"
        with self.assertRaisesRegex(analyze.AnalysisError, "supported scan"):
            analyze.analyze_convergence(descriptor)

    def test_grid_scan_rejects_inconsistent_transverse_refinement(self) -> None:
        levels = []
        for nr, nz in ((5, 5), (9, 7), (17, 17)):
            R, Z = np.linspace(.2, 1.8, nr), np.linspace(0., 1., nz)
            h = float(R[1] - R[0])
            field = np.zeros((3, nz, nr), dtype=complex)
            field[1] = 1.0 + h*h
            run = _write_run(self.root, field=field, R=R, Z=Z, scan_value=h)
            levels.append({"run": str(run), "h": h})
        descriptor = _descriptor(levels, reference={
            "kind": "constant", "amplitude": 1., "unit": "1/m^3"})
        with self.assertRaisesRegex(analyze.AnalysisError, "transverse spacing"):
            analyze.analyze_convergence(descriptor)

    def test_audited_declaration_cannot_bypass_time_initial_identity(self) -> None:
        levels = []
        for h in (.04, .02, .01):
            field = np.full((3, 5, 5), 1. + h*h, dtype=complex)
            run = _write_run(self.root, field=field, initial_field=field,
                             R=np.linspace(.2, 1.8, 5), Z=np.linspace(0., 1., 5),
                             scan_value=h, scan_key="time.dt")
            levels.append({"run": str(run), "h": h})
        descriptor = _descriptor(levels, reference={
            "kind": "constant", "amplitude": 1., "unit": "1/m^3"})
        descriptor["scan_variable"] = "time.dt"
        with self.assertRaisesRegex(analyze.AnalysisError, "same_grid_same_initial"):
            analyze.analyze_convergence(descriptor)

    def test_time_scan_rejects_changed_boundary_geometry(self) -> None:
        levels = []
        for index, h in enumerate((.04, .02, .01)):
            field = np.zeros((3, 5, 5), dtype=complex)
            field[1] = 1. + h*h
            run = _write_run(
                self.root, field=field, R=np.linspace(.2, 1.8, 5),
                Z=np.linspace(0., 1., 5), scan_value=h, scan_key="time.dt",
                polygon_R=np.array([.2, 1.8 - index*.01, 1.8, .2]))
            levels.append({"run": str(run), "h": h})
        descriptor = _descriptor(levels, scan_variable="time.dt", reference={
            "kind": "constant", "amplitude": 1., "unit": "1/m^3"})
        with self.assertRaisesRegex(analyze.AnalysisError, "geometry"):
            analyze.analyze_convergence(descriptor)

    def test_short_si_cadence_is_not_duplicate_time(self) -> None:
        times = np.arange(101) * 1e-13
        frequency = 1e12
        values = np.exp(-2j * np.pi * frequency * times)
        run = _write_run(self.root, field=np.zeros((3, 5, 5), dtype=complex),
                         R=np.linspace(.2, 1.8, 5), Z=np.linspace(0., 1., 5),
                         definition={"field": "n_i", "component": "scalar", "mode": 0, "entity_key": [1, 1]},
                         observations=list(zip(times, values)))
        result = analyze.fit_mode(run, t_start=0., t_end=float(times[-1]),
                                  amplitude_floor=1e-8, target_frequency_hz=frequency)
        self.assertEqual(result["status"], "PASS")
        self.assertAlmostEqual(result["angular_frequency_rad_s"] / (2*np.pi*frequency), 1., places=10)

    def test_declared_raw_config_anchor_must_match_even_with_valid_config_hash(self) -> None:
        run = _write_run(self.root, field=np.zeros((3, 5, 5), dtype=complex),
                         R=np.linspace(.2, 1.8, 5), Z=np.linspace(0., 1., 5))
        path = run / "run.json"
        manifest = json.loads(path.read_text())
        manifest["resolved_config_sha256"] = "0" * 64
        _write_json(path, manifest)
        with self.assertRaisesRegex(analyze.AnalysisError, "anchor"):
            analyze.load_run(run)

    def test_convergence_rejects_incorrect_physical_units(self) -> None:
        levels = []
        for h in (.04, .02, .01):
            run = _write_run(self.root, field=np.full((3, 5, 5), 1+h*h, dtype=complex),
                             R=np.linspace(.2, 1.8, 5), Z=np.linspace(0., 1., 5),
                             scan_key="time.dt", scan_value=h)
            levels.append({"run": str(run), "h": h})
        descriptor = _descriptor(levels, scan_variable="time.dt", reference={
            "kind": "constant", "amplitude": 1., "unit": "Pa"})
        descriptor["observable"]["unit"] = "Pa"
        with self.assertRaisesRegex(analyze.AnalysisError, "unit"):
            analyze.analyze_convergence(descriptor)
        descriptor["observable"]["unit"] = "1/m^3"
        descriptor["reference"]["unit"] = "1/m^3"
        descriptor["observable"]["norm"]["unit"] = "Pa"
        with self.assertRaisesRegex(analyze.AnalysisError, "unit"):
            analyze.analyze_convergence(descriptor)

    def test_decay_fit_uses_finite_reference_amplitude_at_large_time_offset(self) -> None:
        relative_time = np.arange(101) * 1e-8
        times = .001 + relative_time
        frequency, growth = 2e6, -1e6
        values = np.exp((growth-2j*np.pi*frequency)*relative_time)
        run = _write_run(self.root, field=np.zeros((3, 5, 5), dtype=complex),
                         R=np.linspace(.2, 1.8, 5), Z=np.linspace(0., 1., 5),
                         definition={"field": "n_i", "component": "scalar", "mode": 0, "entity_key": [1, 1]},
                         observations=list(zip(times, values)))
        result = analyze.fit_mode(run, t_start=float(times[0]), t_end=float(times[-1]),
                                  amplitude_floor=1e-8, target_frequency_hz=frequency)
        self.assertEqual(result["status"], "PASS")
        self.assertAlmostEqual(result["frequency_hz"]/frequency, 1., places=9)
        self.assertAlmostEqual(result["growth_rate_per_s"]/growth, 1., places=9)
        self.assertIsNone(result["amplitude"])
        self.assertAlmostEqual(result["amplitude_at_reference_time"], math.exp(-.5), places=9)

    def test_duplicate_manifest_keys_are_rejected(self) -> None:
        R = np.linspace(0.2, 1.8, 5)
        Z = np.linspace(0.0, 1.0, 5)
        run = _write_run(
            self.root,
            field=np.zeros((3, 5, 5), dtype=np.complex128),
            R=R,
            Z=Z,
            scan_value=0.4,
        )
        manifest = (run / "run.json").read_text()
        manifest = manifest.replace('"run_id":', '"run_id": "duplicate", "run_id":')
        (run / "run.json").write_text(manifest, encoding="utf-8")
        with self.assertRaisesRegex(analyze.AnalysisError, "duplicate"):
            analyze.load_run(run)


if __name__ == "__main__":
    unittest.main()
