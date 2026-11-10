"""Validation and bounded integration tests for the circular CLT runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools" / "warp_clt_boundary.py"


def _load_runner():
    assert RUNNER.is_file(), f"runner is missing: {RUNNER}"
    spec = importlib.util.spec_from_file_location(
        "warp_clt_boundary_under_test", RUNNER
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _args(*extra):
    return [
        "--nodes",
        "8",
        "--mesh",
        "polygon",
        "--baseline",
        "vacuum",
        "--major-radius",
        "3",
        "--minor-radius",
        "0.5",
        "--K",
        "1",
        "--mode",
        "1",
        "--amplitude",
        "1e-4",
        "--h",
        "1e-7",
        "--M",
        "2",
        "--s",
        "3",
        "--steps",
        "4",
        "--segment-steps",
        "2",
        "--coupling",
        "wr",
        "--device",
        "cpu",
        "--output",
        "/tmp/not-created-by-validation-test",
        "--wall-seconds",
        "20",
        "--snapshot-every",
        "1",
        *extra,
    ]


def test_cli_controls_resolve_nphi_and_physically_matched_horizon():
    runner = _load_runner()
    args = runner.parse_args(_args())

    plan = runner.validate_controls(args)

    assert plan.micro_count == 2
    assert plan.micro_h == pytest.approx(1e-7)
    assert plan.macro_h == pytest.approx(2e-7)
    assert plan.horizon_s == pytest.approx(8e-7)
    assert plan.Nphi == 5


def test_axisymmetric_mode_uses_one_distinct_toroidal_angle():
    runner = _load_runner()
    values = _args()
    values[values.index("--K") + 1] = "0"
    values[values.index("--mode") + 1] = "0"
    plan = runner.validate_controls(runner.parse_args(values))

    assert plan.Nphi == 1
    assert plan.macro_h == pytest.approx(2e-7)


def test_zero_amplitude_is_a_valid_actual_control():
    runner = _load_runner()
    values = runner.parse_args(_args("--amplitude", "0"))

    plan = runner.validate_controls(values)

    assert plan.amplitude == 0.0
    staircase = runner.parse_args(
        [
            "--nodes",
            "6",
            "--mesh",
            "staircase",
            "--K",
            "0",
            "--mode",
            "0",
            "--amplitude",
            "0",
            "--device",
            "cpu",
            "--output",
            "/tmp/zero-control-not-created",
            "--steps",
            "1",
        ]
    )
    context, metadata = runner.make_context(staircase)
    state, _ = runner.initial_state(context, metadata)
    seeded, perturbation = runner.apply_default_perturbation(state, context)
    assert np.max(np.abs(seeded.fluid[0].mom), initial=0.0) == 0.0
    assert np.max(np.abs(perturbation["projected_mom_i"]), initial=0.0) == 0.0
    assert perturbation["vA_m_s"] == context.p.ref.velocity


def test_seed_defaults_to_smooth_and_is_in_identity():
    runner = _load_runner()
    smooth = runner.validate_controls(runner.parse_args(_args()))
    caw = runner.validate_controls(
        runner.parse_args(
            _args("--mode", "0", "--seed", "cylindrical-caw")
        )
    )

    assert smooth.seed == "smooth"
    assert caw.seed == "cylindrical-caw"
    assert runner.scheme_parameters(caw)["seed"] == "cylindrical-caw"
    assert runner.input_identity(
        caw, geometry_fingerprint="geometry"
    ) != runner.input_identity(smooth, geometry_fingerprint="geometry")


def test_cylindrical_caw_is_mode0_only():
    runner = _load_runner()
    values = runner.parse_args(_args("--seed", "cylindrical-caw"))

    assert values.mode == 1
    with pytest.raises(runner.RunnerError, match="mode 0"):
        runner.validate_controls(values)


def test_cylindrical_caw_helper_is_lazy_and_fails_explicitly_if_unavailable(
    tmp_path,
):
    runner = _load_runner()
    args = runner.parse_args(
        [
            "--nodes",
            "6",
            "--mesh",
            "staircase",
            "--baseline",
            "vacuum",
            "--K",
            "0",
            "--mode",
            "0",
            "--amplitude",
            "1e-4",
            "--seed",
            "cylindrical-caw",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "caw"),
            "--steps",
            "1",
        ]
    )
    context, metadata = runner.make_context(args)
    state, _ = runner.initial_state(context, metadata)
    before_B = np.asarray(context.compute.to_host(state.B)).copy()
    before_n = np.asarray(context.compute.to_host(state.fluid[0].n)).copy()
    before_p = np.asarray(context.compute.to_host(state.fluid[0].perp)).copy()
    spec = importlib.util.find_spec("tools.cylindrical_caw_control")
    if spec is None:
        with pytest.raises(runner.RunnerError) as error:
            runner.apply_default_perturbation(state, context)
        assert error.value.code == "IntegrationUnavailable"
        assert "cylindrical_caw_control" in str(error.value)
    else:
        seeded, evidence = runner.apply_default_perturbation(state, context)
        assert evidence["seed"] == "cylindrical-caw"
        np.testing.assert_array_equal(
            np.asarray(context.compute.to_host(seeded.B)), before_B
        )
        np.testing.assert_array_equal(
            np.asarray(context.compute.to_host(seeded.fluid[0].n)), before_n
        )
        np.testing.assert_array_equal(
            np.asarray(context.compute.to_host(seeded.fluid[0].perp)), before_p
        )


@pytest.mark.parametrize("nodes", [32, 64, 128, 256, 512, 1024])
def test_polygon_nodes_are_not_capped_at_legacy_runner_limit(nodes):
    runner = _load_runner()
    values = runner.parse_args(_args())
    values.nodes = nodes

    plan = runner.validate_controls(values)

    assert plan.nodes == nodes


def test_generator_cells_cover_the_full_circular_bounding_box():
    runner = _load_runner()
    plan = runner.validate_controls(runner.parse_args(_args()))

    R, Z = runner._generator_grid(plan)

    h_R = R[1] - R[0]
    h_Z = Z[1] - Z[0]
    assert R[0] - h_R / 2 == pytest.approx(plan.major_radius - plan.minor_radius)
    assert R[-1] + h_R / 2 == pytest.approx(plan.major_radius + plan.minor_radius)
    assert Z[0] - h_Z / 2 == pytest.approx(-plan.minor_radius)
    assert Z[-1] + h_Z / 2 == pytest.approx(plan.minor_radius)


def test_grid_shift_is_applied_as_a_fraction_of_generator_spacing():
    runner = _load_runner()
    values = runner.parse_args(
        _args("--grid-shift-R", "0.25", "--grid-shift-Z", "-0.5")
    )
    plan = runner.validate_controls(values)

    R, Z = runner._generator_grid(plan)
    h_R = R[1] - R[0]
    h_Z = Z[1] - Z[0]
    unshifted_R = plan.major_radius - plan.minor_radius + 0.5 * h_R
    unshifted_Z = -plan.minor_radius + 0.5 * h_Z

    assert R[0] == pytest.approx(unshifted_R + 0.25 * h_R)
    assert Z[0] == pytest.approx(unshifted_Z - 0.5 * h_Z)


def test_waveform_tolerance_is_explicit_in_parameters_and_identity():
    runner = _load_runner()
    values = runner.parse_args(_args("--waveform-rtol", "2e-12"))
    plan = runner.validate_controls(values)

    parameters = runner.build_parameters(values, plan)
    scheme = runner.scheme_parameters(plan)
    identity = runner.input_identity(plan, geometry_fingerprint="geometry")

    assert plan.waveform_rtol == pytest.approx(2.0e-12)
    assert parameters.waveform_rtol == pytest.approx(2.0e-12)
    assert scheme["waveform_rtol"] == pytest.approx(2.0e-12)
    changed = runner.validate_controls(
        runner.parse_args(_args("--waveform-rtol", "3e-12"))
    )
    assert runner.input_identity(
        changed, geometry_fingerprint="geometry"
    ) != identity


@pytest.mark.parametrize(
    "override",
    [
        ["--h", "2e-7"],
        ["--M", "3"],
        ["--s", "5"],
        ["--amplitude", "0"],
        ["--grid-shift-R", "0.25"],
        ["--grid-shift-Z", "-0.25"],
        ["--coupling", "synchronous", "--M", "1"],
    ],
)
def test_input_identity_includes_scheme_and_placement_parameters(override):
    runner = _load_runner()
    base = runner.validate_controls(runner.parse_args(_args()))
    changed = runner.validate_controls(
        runner.parse_args(_args(*override))
    )

    assert runner.input_identity(
        changed, geometry_fingerprint="geometry"
    ) != runner.input_identity(base, geometry_fingerprint="geometry")


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"coupling": "synchronous", "M": 2}, "micro_steps"),
        ({"mode": 1, "K": 0}, "retain"),
        ({"seed": "unsupported"}, "seed"),
        ({"nodes": 3}, "nodes"),
        ({"grid_shift_R": 0.75}, "grid-shift"),
        ({"waveform_rtol": 0.0}, "waveform-rtol"),
        ({"minor_radius": 3.0}, "minor"),
        ({"amplitude": -1e-4}, "amplitude"),
        ({"h": 0.0}, "h"),
        ({"wall_seconds": 1700.1}, "1700"),
        ({"segment_steps": 5}, "segment"),
        ({"s": 4}, "s"),
    ],
)
def test_invalid_controls_fail_before_context_or_allocation(changes, match):
    runner = _load_runner()
    values = runner.parse_args(_args())
    for key, value in changes.items():
        setattr(values, key, value)

    with pytest.raises((ValueError, runner.RunnerError), match=match):
        runner.validate_controls(values)


def test_memory_caps_are_checked_at_75_percent_of_available():
    runner = _load_runner()
    gib = 1 << 30

    result = runner.check_memory_caps(
        host_available_bytes=48 * gib,
        device_available_bytes=24 * gib,
        host_cap_bytes=32 * gib,
        device_cap_bytes=16 * gib,
    )

    assert result["host_fraction"] == pytest.approx(2.0 / 3.0)
    assert result["device_fraction"] == pytest.approx(2.0 / 3.0)
    with pytest.raises(runner.RunnerError, match="75%"):
        runner.check_memory_caps(
            host_available_bytes=40 * gib,
            device_available_bytes=24 * gib,
            host_cap_bytes=32 * gib,
            device_cap_bytes=16 * gib,
        )
    with pytest.raises(runner.RunnerError, match="75%"):
        runner.check_memory_caps(
            host_available_bytes=48 * gib,
            device_available_bytes=20 * gib,
            host_cap_bytes=32 * gib,
            device_cap_bytes=16 * gib,
        )


def test_physical_budgets_and_gate_policy_are_preserved():
    runner = _load_runner()
    from tools.warp_tokamak_cae import physical_parameters

    values = runner.parse_args(_args())
    plan = runner.validate_controls(values)
    actual = runner.build_parameters(values, plan)
    expected = physical_parameters(
        1, 2.0e19, 2.0, 0.5, 1000.0
    )

    assert set(actual.budgets) == set(expected.budgets)
    for name, budget in expected.budgets.items():
        assert actual.budgets[name].hard is budget.hard
        assert actual.budgets[name].unit == budget.unit
        assert actual.budgets[name].norm == budget.norm
        assert actual.budgets[name].atol == budget.atol
        assert actual.budgets[name].rtol == budget.rtol
        assert actual.budgets[name].reference == budget.reference


def test_output_must_be_a_new_non_symlink_directory(tmp_path):
    runner = _load_runner()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(runner.RunnerError, match="NEWDIR|empty"):
        runner.validate_output(existing)
    symlink = tmp_path / "link"
    symlink.symlink_to(existing, target_is_directory=True)
    with pytest.raises(runner.RunnerError, match="symlink"):
        runner.validate_output(symlink)


def test_old_checkpoint_schema_is_rejected_without_guessing(tmp_path):
    runner = _load_runner()
    path = tmp_path / "old.npz"
    np.savez(path, n_i=np.zeros((1, 1), dtype=np.complex128))

    with pytest.raises(runner.RunnerError, match="checkpoint schema"):
        runner.read_checkpoint_arrays(path)


def test_strict_report_rejects_nonfinite_numbers(tmp_path):
    runner = _load_runner()
    with pytest.raises(ValueError, match="nonfinite"):
        runner.write_strict_json(tmp_path / "bad.json", {"value": np.nan})


def test_initial_evidence_arrays_are_summarized_in_strict_json(tmp_path):
    runner = _load_runner()
    evidence = runner._evidence_json(
        {
            "values": np.asarray([1.0, 2.0, 3.0]),
            "complex_values": np.asarray([1.0 + 2.0j, 3.0 - 4.0j]),
            "mask": np.asarray([True, False]),
        }
    )
    path = tmp_path / "evidence.json"

    runner.write_strict_json(path, evidence)
    persisted = json.loads(path.read_text())

    assert persisted["values"]["shape"] == [3]
    assert persisted["values"]["maximum"] == 3.0
    assert persisted["complex_values"]["maximum"] == 5.0
    assert persisted["mask"]["finite_count"] == 2


def test_backend_state_is_explicitly_converted_before_native_codec():
    runner = _load_runner()
    args = runner.parse_args(
        [
            "--nodes",
            "6",
            "--mesh",
            "staircase",
            "--K",
            "0",
            "--mode",
            "0",
            "--device",
            "cpu",
            "--output",
            "/tmp/host-boundary-not-created",
            "--steps",
            "1",
        ]
    )
    context, _ = runner.make_context(args)
    from fomhd import core, io
    from fomhd.types import FluidState, State

    reference = core.allocate_state(context)

    class ExplicitExportOnly:
        def __init__(self, value):
            self.value = value

        def __array__(self, *args, **kwargs):
            raise AssertionError("implicit host conversion attempted")

    def export(value):
        assert isinstance(value, ExplicitExportOnly)
        return value.value

    backend_context = type("Boundary", (), {})()
    backend_context.compute = type("Compute", (), {"to_host": staticmethod(export)})()
    state = State(
        fluid=(
            FluidState(
                n=ExplicitExportOnly(reference.fluid[0].n),
                mom=ExplicitExportOnly(reference.fluid[0].mom),
                perp=ExplicitExportOnly(reference.fluid[0].perp),
                parallel=ExplicitExportOnly(reference.fluid[0].parallel),
            ),
            FluidState(
                n=ExplicitExportOnly(reference.fluid[1].n),
                mom=ExplicitExportOnly(reference.fluid[1].mom),
                perp=ExplicitExportOnly(reference.fluid[1].perp),
                parallel=ExplicitExportOnly(reference.fluid[1].parallel),
            ),
        ),
        B=ExplicitExportOnly(reference.B),
        step=0,
        generation=1,
        time=0.0,
    )

    host = runner.host_state_for_codec(state, backend_context)
    arrays = io.encode_state(host, context.geometry)

    assert isinstance(host.B, np.ndarray)
    assert arrays["B_phi"].shape[0] == 1


def test_preflight_only_reports_real_geometry_and_storage_without_operators(
    tmp_path,
):
    runner = _load_runner()
    output = tmp_path / "preflight"
    args = runner.parse_args(
        [
            "--nodes",
            "6",
            "--mesh",
            "staircase",
            "--baseline",
            "vacuum",
            "--K",
            "0",
            "--mode",
            "0",
            "--amplitude",
            "0",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--steps",
            "1",
            "--preflight-only",
        ]
    )

    assert runner.run(args) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "preflight_complete"
    assert report["geometry"]["Nc"] > 0
    assert report["preflight"]["estimated_operator_bytes"] > 0
    assert report["preflight"]["operators_allocated"] is False
    assert not (output / "checkpoints" / "restart.npz").exists()
    geometry = np.load(output / "geometry.npz", allow_pickle=False)
    assert "cell_polygon_points" in geometry.files
    assert "cell_polygon_offsets" in geometry.files
    assert "physical_wall_distance" in geometry.files
    assert "face_volumes" in geometry.files
    assert report["geometry"]["file"] == "geometry.npz"
    assert len(report["geometry"]["sha256"]) == 64


def test_staircase_vacuum_context_keeps_legacy_boolean_state_layout(tmp_path):
    runner = _load_runner()
    args = runner.parse_args(
        [
            "--nodes",
            "6",
            "--mesh",
            "staircase",
            "--baseline",
            "vacuum",
            "--K",
            "0",
            "--mode",
            "0",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "staircase"),
            "--steps",
            "1",
        ]
    )
    context, metadata = runner.make_context(args)
    state, _ = runner.initial_state(context, metadata)
    from fomhd import io as fomhd_io

    arrays = fomhd_io.encode_state(state, context.geometry)

    assert metadata["layout_kind"] == "staircase_v1"
    assert {"B_R", "B_phi", "B_Z"} <= set(arrays)
    assert np.all(np.isfinite(state.B))


def test_staircase_gs_uses_valid_cell_flux_and_harmonic_mapping():
    runner = _load_runner()
    args = runner.parse_args(
        [
            "--nodes",
            "6",
            "--mesh",
            "staircase",
            "--baseline",
            "gs",
            "--K",
            "0",
            "--mode",
            "0",
            "--amplitude",
            "0",
            "--device",
            "cpu",
            "--output",
            "/tmp/staircase-gs-not-created",
            "--steps",
            "1",
        ]
    )
    context, metadata = runner.make_context(args)

    state, initialization = runner.initial_state(context, metadata)

    assert initialization["old_staircase_mapping"] == (
        "C@Aphi + F0*conj(topology_flux_row)/MB"
    )
    assert initialization["project_B_used"] is False
    assert initialization["vertex_polygon_mask"].any()
    assert not initialization["vertex_polygon_mask"].all()
    assert np.all(np.isfinite(state.B))
    equilibrium = initialization["equilibrium_parameters"]
    assert equilibrium["R0_m"] == 3.0
    assert equilibrium["a_m"] == 0.5
    assert equilibrium["F0_T_m"] == 6.0
    assert equilibrium["n0_m^-3"] == 2.0e19
    assert equilibrium["p_edge_Pa"] == 1000.0
    assert equilibrium["pprime_Pa_per_Wb_rad"] == 1.0e4
    assert equilibrium["alpha_ion_fraction"] == 0.6
    assert equilibrium["degree"] == 10


def test_k1_snapshot_uses_physical_samples_not_ratio_of_modes():
    runner = _load_runner()
    args = runner.parse_args(
        [
            "--nodes",
            "6",
            "--mesh",
            "staircase",
            "--baseline",
            "vacuum",
            "--K",
            "1",
            "--mode",
            "1",
            "--amplitude",
            "0",
            "--device",
            "cpu",
            "--output",
            "/tmp/k1-snapshot-not-created",
            "--steps",
            "1",
        ]
    )
    context, _ = runner.make_context(args)
    K = context.p.K
    modes = 2 * K + 1
    nc = context.op.Nc
    nphi = 4 * K + 1
    from fomhd import core
    from fomhd.types import StageTag, fluid_context_record, stage_result_record

    state = core.allocate_state(context)
    density_i_modes = np.zeros((modes, nc), dtype=np.complex128)
    density_i_modes[K] = 2.0e19
    density_i_modes[K - 1] = 1.0e18 + 2.0e17j
    density_i_modes[K + 1] = np.conj(density_i_modes[K - 1])
    state.fluid[0].n[...] = context.compute.array(
        density_i_modes.reshape(-1), np.complex128
    )
    state.fluid[0].mom[...] = 0
    state.fluid[0].perp[...] = context.compute.array(
        np.full(modes * nc, 600.0, dtype=np.complex128)
    )
    state.fluid[1].n[...] = context.compute.array(
        density_i_modes.reshape(-1), np.complex128
    )
    state.fluid[1].mom[...] = 0
    state.fluid[1].perp[...] = context.compute.array(
        np.full(modes * nc, 400.0, dtype=np.complex128)
    )
    state.fluid[1].parallel[...] = context.compute.array(
        np.full(modes * nc, 390.0, dtype=np.complex128)
    )
    state.B[...] = 0

    phi = 2.0 * np.pi * np.arange(nphi) / nphi
    density_samples = (
        0.1 + (2.0e19 + np.arange(nc))[None, :] * (1.0 + 0.2 * np.cos(phi)[:, None])
    )
    velocity_samples = np.empty((nphi, 3, nc), dtype=np.complex128)
    velocity_samples[:, 0, :] = (
        10.0 * np.cos(phi)[:, None] + np.arange(nc)[None, :]
    )
    velocity_samples[:, 1, :] = (
        20.0 * np.sin(phi)[:, None] - 0.25 * np.arange(nc)[None, :]
    )
    velocity_samples[:, 2, :] = (
        -30.0 * np.cos(phi)[:, None] + 0.5 * np.arange(nc)[None, :]
    )
    bn_samples = np.empty((nphi, 3, nc), dtype=np.complex128)
    bn_samples[:, 0, :] = np.cos(phi)[:, None]
    bn_samples[:, 1, :] = 2.0 + 0.3 * np.sin(phi)[:, None]
    bn_samples[:, 2, :] = -0.5 * np.cos(phi)[:, None]
    electron_density = 1.01 * density_samples
    cell_axis = np.arange(nc, dtype=np.float64)[None, :]
    ion_pressure = 600.0 + 3.0 * np.cos(phi)[:, None] + 0.0 * cell_axis
    electron_perp = 400.0 + 2.0 * np.sin(phi)[:, None] + 0.0 * cell_axis
    electron_parallel = 390.0 - np.cos(phi)[:, None] + 0.0 * cell_axis

    tag = StageTag(
        step=0,
        generation=1,
        trial=1,
        spectral_context=0,
        time=0.0,
        stage="snapshot-test",
    )
    frame = stage_result_record(tag=tag)
    frame.BN = context.compute.array(bn_samples.reshape(-1), np.complex128)
    current_modes = np.arange(
        modes * context.op.Ne, dtype=np.float64
    ).reshape(modes, context.op.Ne).astype(np.complex128)
    current_modes += 0.25j * np.arange(
        modes * context.op.Ne, dtype=np.float64
    ).reshape(modes, context.op.Ne)
    frame.current = context.compute.array(
        current_modes.reshape(-1), np.complex128
    )
    frame.fluid = (
        fluid_context_record(
            n=context.compute.array(density_samples.reshape(-1)),
            u=context.compute.array(velocity_samples.reshape(-1)),
            perp=context.compute.array(ion_pressure.reshape(-1)),
        ),
        fluid_context_record(
            n=context.compute.array(electron_density.reshape(-1)),
            perp=context.compute.array(electron_perp.reshape(-1)),
            parallel=context.compute.array(electron_parallel.reshape(-1)),
        ),
    )

    arrays = runner._snapshot_payload(state, frame, context)
    metadata = json.loads(str(arrays["metadata_json"]))

    np.testing.assert_array_equal(arrays["B_N_samples"], bn_samples)
    np.testing.assert_array_equal(arrays["ion_velocity_samples"], velocity_samples)
    np.testing.assert_array_equal(arrays["ion_density_samples"], density_samples)
    np.testing.assert_array_equal(
        arrays["ion_perp_pressure_samples"], ion_pressure
    )
    np.testing.assert_array_equal(
        arrays["electron_density_samples"], electron_density
    )
    np.testing.assert_array_equal(
        arrays["electron_perp_pressure_samples"], electron_perp
    )
    np.testing.assert_array_equal(
        arrays["electron_parallel_pressure_samples"], electron_parallel
    )
    assert arrays["n_i_modes"].shape == (modes, nc)
    assert arrays["B_phi_modes"].shape[0] == modes
    np.testing.assert_array_equal(arrays["current_modes"], current_modes)
    assert "ion_velocity_R" not in arrays
    np.testing.assert_allclose(metadata["sample_phis"], phi, rtol=0, atol=0)
    assert metadata["mode_axis"] == list(range(-K, K + 1))


def test_initial_force_exports_use_actual_physical_sample_arrays():
    runner = _load_runner()
    args = runner.parse_args(
        [
            "--nodes",
            "6",
            "--mesh",
            "staircase",
            "--K",
            "1",
            "--mode",
            "0",
            "--device",
            "cpu",
            "--output",
            "/tmp/force-samples-not-created",
            "--steps",
            "1",
        ]
    )
    context, _ = runner.make_context(args)
    modes = 2 * context.p.K + 1
    nphi = 4 * context.p.K + 1
    nc = context.op.Nc
    shape = (nphi, 3, nc)
    base = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    pressure_force = base + 1.0
    magnetic_force = 1j * (base + 2.0)
    electric_force = base + 3.0
    convection = base + 4.0
    reaction = base + 5.0
    net_force = base + 6.0
    from fomhd.types import fluid_context_record, fluid_terms_record

    frame = SimpleNamespace(
        fluid=(
            fluid_context_record(
                pressure_force=context.compute.array(
                    pressure_force.reshape(-1)
                ),
                magnetic_force=context.compute.array(
                    magnetic_force.reshape(-1)
                ),
                convection=context.compute.array(convection.reshape(-1)),
            ),
            None,
        ),
        terms=(
            fluid_terms_record(
                electric=context.compute.array(electric_force.reshape(-1)),
                reaction=context.compute.array(reaction.reshape(-1)),
                force=context.compute.array(net_force.reshape(-1)),
            ),
            None,
        ),
    )

    samples = runner._initial_force_samples(context, frame)

    expected = {
        "frame_ion_pressure_force_samples": pressure_force,
        "frame_ion_magnetic_force_samples": magnetic_force,
        "frame_ion_electric_force_samples": electric_force,
        "frame_ion_convection_samples": convection,
        "frame_ion_wall_reaction_samples": reaction,
        "frame_ion_net_force_samples": net_force,
    }
    for name, value in expected.items():
        assert samples[name].shape == shape
        np.testing.assert_array_equal(samples[name], value)
    assert modes == 3
    assert nphi == 5


def test_source_fingerprint_manifest_uses_relative_names_only():
    runner = _load_runner()

    manifest = runner.source_fingerprint_manifest()

    assert manifest
    assert all(not path.startswith("/") for path in manifest)
    assert all("20261109T" not in path for path in manifest)
    assert runner._source_fingerprint() == runner._source_fingerprint()


def test_wr_coupling_metadata_is_serialized_without_inferred_counts():
    runner = _load_runner()
    metadata = {
        "sweeps": 7,
        "residual_history": [
            {"iteration": 1, "normalized_max": 0.25},
            {"iteration": 2, "normalized_max": 0.05},
        ],
        "work": {
            "total_full_stage_calls": 11,
            "field_evaluations": 23,
        },
        "interval_attribution": {
            "density_work_coarse": 1.5,
            "density_work_fine": 1.25,
            "kinetic_path": -0.2,
            "delta_D": 0.25,
        },
    }

    serialized = runner._coupling_metadata_json(metadata)

    assert serialized["sweeps"] == 7
    assert serialized["residual_history"] == metadata["residual_history"]
    assert serialized["work"] == metadata["work"]
    assert serialized["interval_attribution"] == metadata["interval_attribution"]


def test_wr_failure_metadata_is_preserved_in_failure_receipt(tmp_path):
    runner = _load_runner()
    args = runner.parse_args(
        [
            "--nodes",
            "6",
            "--mesh",
            "staircase",
            "--K",
            "0",
            "--mode",
            "0",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "unused"),
            "--steps",
            "1",
        ]
    )
    context, metadata = runner.make_context(args)
    coupling = {
        "sweeps": 3,
        "residual_history": [{"normalized_max": 0.1}],
        "work": {"total_full_stage_calls": 5},
        "interval_attribution": None,
    }
    error = SimpleNamespace(
        code="InjectedWRFailure",
        checks=[],
        stage=None,
        partial_arrays={},
        coupling_metadata=coupling,
    )

    receipt = runner._save_failure(
        tmp_path, error, None, context, metadata
    )

    assert receipt["coupling_metadata"]["sweeps"] == 3
    persisted = json.loads((tmp_path / "failure.json").read_text())
    assert persisted["coupling_metadata"] == receipt["coupling_metadata"]


def test_timing_attribution_identifies_unattributed_time():
    runner = _load_runner()
    timing = {
        "setup_s": 1.0,
        "initialization_s": 0.5,
        "resume_evaluation_s": 0.25,
        "advance_s": 1.5,
        "output_s": 0.25,
        "total_s": 4.0,
    }

    runner._finalize_timing(timing)

    assert timing["total_s"] == 4.0
    assert timing["attributed_s"] == 3.5
    assert timing["unattributed_s"] == 0.5
    assert "bookkeeping" in timing["unattributed_scope"]


def test_pool_snapshot_uses_cupy_total_bytes_api_only():
    runner = _load_runner()

    class CuPyPoolAPI:
        def used_bytes(self):
            return 7

        def total_bytes(self):
            return 11

    sample = runner._pool_snapshot_from_object(CuPyPoolAPI())

    assert sample == {"used_bytes": 7, "reserved_bytes": 11}
    assert not hasattr(CuPyPoolAPI(), "reserved_bytes")


def test_pool_probe_failure_is_explicit_and_never_reports_zero(
    monkeypatch,
):
    runner = _load_runner()
    tracker = {
        "used_bytes_final": 0,
        "reserved_bytes_final": 0,
        "sampled_peak_used_bytes": 0,
        "sampled_peak_reserved_bytes": 0,
        "sampling_points": [],
        "_context": SimpleNamespace(
            compute=SimpleNamespace(synchronize=lambda: None)
        ),
    }

    monkeypatch.setattr(
        runner,
        "_device_pool_snapshot",
        lambda device: (_ for _ in ()).throw(RuntimeError("pool probe down")),
    )
    runner._record_pool_sample("cuda:0", tracker, "probe_failure")
    pool = runner._pool_report(tracker)

    assert pool["available"] is False
    assert pool["used_bytes_final"] is None
    assert pool["reserved_bytes_final"] is None
    assert pool["sampling_errors"][0]["label"] == "probe_failure"
    assert "pool probe down" in pool["sampling_errors"][0]["message"]


def test_memory_probe_failure_does_not_mask_original_failure(
    tmp_path, monkeypatch
):
    runner = _load_runner()
    output = tmp_path / "failed-run"
    args = runner.parse_args(
        [
            "--nodes",
            "6",
            "--mesh",
            "staircase",
            "--K",
            "0",
            "--mode",
            "0",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--steps",
            "1",
            "--resume",
            str(tmp_path / "missing-checkpoint.npz"),
        ]
    )
    monkeypatch.setattr(
        runner,
        "_device_pool_snapshot",
        lambda device: (_ for _ in ()).throw(RuntimeError("pool probe down")),
    )

    assert runner.run(args) == 2

    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure"]["code"] == "Checkpoint"
    assert "regular file" in report["failure"]["message"]
    assert report["diag"]["device_pool"]["available"] is False
    assert report["diag"]["device_pool"]["used_bytes_final"] is None


def test_real_polygon_vacuum_initialization_when_mesh_is_available(tmp_path):
    if importlib.util.find_spec("fomhd.polygon_geometry") is None:
        pytest.skip("polygon geometry/operators are pending parent integration")
    if importlib.util.find_spec("fomhd.polygon_operators") is None:
        pytest.skip("polygon geometry/operators are pending parent integration")

    runner = _load_runner()
    args = runner.parse_args(
        _args(
            "--output",
            str(tmp_path / "run"),
            "--steps",
            "1",
            "--segment-steps",
            "1",
        )
    )
    context, metadata = runner.make_context(args)
    state, initialization = runner.initial_state(context, metadata)

    assert context.geometry.kind == "polygon_dual_v1"
    np.testing.assert_allclose(
        initialization["raw_B_phi"],
        6.0 / context.geometry.vertices.R,
    )
    np.testing.assert_allclose(
        initialization["mapped_B_phi"], 6.0 / context.geometry.Rbar_v
    )
    np.testing.assert_allclose(
        initialization["raw_minus_mapped_B_phi"],
        6.0 / context.geometry.vertices.R
        - 6.0 / context.geometry.Rbar_v,
    )
    assert np.any(
        np.abs(initialization["raw_minus_mapped_B_phi"]) > 1.0e-12
    )
    equilibrium = initialization["equilibrium_parameters"]
    assert equilibrium["pprime_Pa_per_Wb_rad"] == 0.0
    assert equilibrium["degree"] == 0
    assert equilibrium["alpha_ion_fraction"] == 0.6
    assert np.all(np.isfinite(state.B))


def test_real_polygon_n32_preflight_allocates_no_operators_or_state(tmp_path):
    if importlib.util.find_spec("fomhd.polygon_geometry") is None:
        pytest.skip("polygon geometry/operators are pending parent integration")
    if importlib.util.find_spec("fomhd.polygon_operators") is None:
        pytest.skip("polygon geometry/operators are pending parent integration")

    runner = _load_runner()
    output = tmp_path / "polygon-preflight"
    args = runner.parse_args(
        [
            "--nodes",
            "32",
            "--mesh",
            "polygon",
            "--baseline",
            "vacuum",
            "--K",
            "0",
            "--mode",
            "0",
            "--amplitude",
            "0",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--steps",
            "1",
            "--preflight-only",
        ]
    )

    assert runner.run(args) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "preflight_complete"
    assert report["geometry"]["kind"] == "polygon_dual_v1"
    assert report["geometry"]["Nc"] > 0
    assert report["preflight"]["estimated_operator_bytes"] > 0
    assert report["preflight"]["operators_allocated"] is False
    assert report["preflight"]["state_allocated"] is False


def test_real_polygon_segment_restart_matches_uninterrupted(tmp_path):
    if importlib.util.find_spec("fomhd.polygon_geometry") is None:
        pytest.skip("polygon geometry/operators are pending parent integration")
    if importlib.util.find_spec("fomhd.polygon_operators") is None:
        pytest.skip("polygon geometry/operators are pending parent integration")

    runner = _load_runner()
    prefix = tmp_path / "prefix"
    continued = tmp_path / "continued"
    uninterrupted = tmp_path / "uninterrupted"

    def command(output, steps, segment, resume=None):
        # Match the demonstrated supported control configuration; do not use
        # the rejected N=4/1e-10 launch or relax any acceptance gate.
        values = [
            "--nodes",
            "8",
            "--mesh",
            "polygon",
            "--baseline",
            "vacuum",
            "--K",
            "0",
            "--mode",
            "0",
            "--amplitude",
            "0",
            "--h",
            "1e-12",
            "--M",
            "1",
            "--s",
            "3",
            "--steps",
            str(steps),
            "--segment-steps",
            str(segment),
            "--coupling",
            "synchronous",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--wall-seconds",
            "120",
            "--snapshot-every",
            "1",
        ]
        if resume is not None:
            values.extend(["--resume", str(resume)])
        return runner.parse_args(values)

    assert runner.run(command(prefix, 2, 1)) == 0
    assert runner.run(
        command(
            continued,
            2,
            1,
            prefix / "checkpoints" / "restart.npz",
        )
    ) == 0
    assert runner.run(command(uninterrupted, 2, 2)) == 0

    with np.load(continued / "snapshots" / "step-0000000002.npz") as left:
        with np.load(uninterrupted / "snapshots" / "step-0000000002.npz") as right:
            for name in (
                "n_i_modes",
                "mom_i_modes",
                "n_e_modes",
                "mom_e_modes",
                "B_phi_modes",
                "B_t_modes",
            ):
                np.testing.assert_allclose(left[name], right[name], rtol=0, atol=0)
    prefix_report = json.loads((prefix / "report.json").read_text())
    report = json.loads((continued / "report.json").read_text())
    uninterrupted_report = json.loads(
        (uninterrupted / "report.json").read_text()
    )
    assert report["resume"]["re_evaluated_stage"] is True
    assert report["resume"]["cumulative_reset"] is False
    assert prefix_report["acceptance"]["accepted_step"] == 1
    assert prefix_report["acceptance"]["accepted_intervals"] == 1
    assert prefix_report["acceptance"]["accepted_time_s"] == 1e-12
    assert report["acceptance"]["accepted_step"] == 2
    assert report["acceptance"]["accepted_intervals"] == 2
    assert report["acceptance"]["accepted_time_s"] == 2e-12
    assert uninterrupted_report["acceptance"]["accepted_step"] == 2
    assert uninterrupted_report["acceptance"]["accepted_intervals"] == 2
    assert uninterrupted_report["acceptance"]["accepted_time_s"] == 2e-12
    assert report["timing"]["setup_s"] > 0.0
    assert report["timing"]["initialization_s"] >= 0.0
    assert report["timing"]["advance_s"] > 0.0
    assert report["timing"]["output_s"] > 0.0
    assert report["timing"]["total_s"] >= report["timing"]["attributed_s"]
    assert report["timing"]["unattributed_s"] >= 0.0
    assert "device_peak_bytes" not in report["diag"]
    pool = report["diag"]["device_pool"]
    assert pool["used_bytes_final"] >= 0
    assert pool["reserved_bytes_final"] >= 0
    assert pool["sampled_peak_used_bytes"] >= pool["used_bytes_final"]
    assert pool["sampled_peak_reserved_bytes"] >= pool["reserved_bytes_final"]
    assert "unsampled" in pool["peak_scope"]
    assert "resume_evaluation_complete" in pool["sampling_points"]
    assert "initialization_complete" in (
        prefix_report["diag"]["device_pool"]["sampling_points"]
    )
    assert "initialization_complete" in (
        uninterrupted_report["diag"]["device_pool"]["sampling_points"]
    )
