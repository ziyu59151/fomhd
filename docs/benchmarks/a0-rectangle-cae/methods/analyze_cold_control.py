"""Compare accepted axisymmetric A0 snapshots against the analytic Hall wave."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from axial_hall_box import make_mode, wave
from fomhd import io
from fomhd.backend import Backend
from fomhd.operators import apply_t2


def compare(path: Path) -> dict:
    case = io.load_case(path / "case.toml", backend=Backend("cpu"))
    run = path / "run"
    manifest = json.loads((run / "run.json").read_text())
    if manifest["input_hashes"] != case.input_hashes:
        raise ValueError("Input fingerprint mismatch")
    if manifest["status"] != "completed":
        raise ValueError("The cold Hall control must complete")

    geometry, op = case.geometry, case.operators
    p = case.parameters
    source = json.loads((path / "source.json").read_text())
    mode = make_mode(mu0=p.mu0, ion_mass=p.species[0].mass)
    # An axial constant has zero radial and toroidal magnetic link averages.
    # Store its physical axial link averages in the normal solver chart.
    axial = np.r_[np.zeros(len(geometry.z_faces.R) + len(geometry.vertices.R)),
                  np.full(len(geometry.r_faces.R), mode.b0)]
    background = apply_t2(op, axial.astype(np.complex128))
    velocity_scale = np.sqrt(mode.mu0 * mode.ion_mass * mode.density)
    weight = np.tile(np.sqrt(op.M0), 6)

    def energy_vector(b, u):
        return np.r_[b.ravel(), velocity_scale * u.ravel()].real * weight

    basis = []
    for phase in (0.0, np.pi / 2):
        ref = wave(geometry.cells.R, geometry.cells.Z, phase=phase,
                   amplitude=source["amplitude"], mode=mode)
        basis.append(energy_vector(ref["delta_B"].T, ref["ion_velocity"].T))
    basis = np.column_stack(basis)
    norm = np.linalg.norm(basis[:, 0])
    rows = []
    for snapshot in sorted((run / "snapshots").glob("state_*.npz")):
        with np.load(snapshot, allow_pickle=False) as data:
            assert bool(data["accepted"])
            assert data["matrix_fingerprint"].item() == manifest["matrix_fingerprint"]
            state = io.decode_state(data, geometry, 0)
            time_s = float(data["time"])
        ion = state.fluid[0]
        field = op.QB @ (state.B - background)
        velocity = ion.mom.reshape(3, -1) / (mode.ion_mass * ion.n[None, :])
        actual = energy_vector(field, velocity)
        coefficients = np.linalg.lstsq(basis, actual, rcond=None)[0]
        phase = -mode.omega * time_s
        expected = np.array([np.cos(phase), np.sin(phase)])
        rows.append({
            "t_s": time_s,
            "coeff": coefficients.tolist(),
            "expected": expected.tolist(),
            "outside_reference_span": float(np.linalg.norm(actual - basis @ coefficients) / norm),
            "error_to_exact": float(np.linalg.norm(actual - basis @ expected) / norm),
        })
    return {
        "run_status": manifest["status"],
        "saved_samples": len(rows),
        "final_saved_ns": rows[-1]["t_s"] * 1e9,
        "reference_frequency_MHz": mode.omega / (2 * np.pi * 1e6),
        "max_off_profile": max(row["outside_reference_span"] for row in rows),
        "initial_total_error": rows[0]["error_to_exact"],
        "final_total_error": rows[-1]["error_to_exact"],
        "samples": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.case_dir), indent=2, allow_nan=False))
