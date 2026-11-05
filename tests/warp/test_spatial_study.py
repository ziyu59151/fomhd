from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

import fomhd
from fomhd import massless
from fomhd.types import Failure


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import warp_spatial_study as study  # noqa: E402


def _five_point_derivative(function, value: float, step: float) -> float:
    return (
        function(value - 2.0 * step)
        - 8.0 * function(value - step)
        + 8.0 * function(value + step)
        - function(value + 2.0 * step)
    ) / (12.0 * step)


@pytest.mark.parametrize("profile", ("compact", "wall"))
@pytest.mark.parametrize("point", ((1.38, 0.41, 0.73), (1.63, 0.58, 2.1)))
def test_analytic_pressure_gradient_matches_independent_finite_differences(
    profile: str,
    point: tuple[float, float, float],
) -> None:
    r, z, phi = point
    _, radial, azimuthal, axial = study.analytic_pressure(r, z, phi, profile)

    def pressure_at_r(value: float) -> float:
        return study.analytic_pressure(value, z, phi, profile)[0]

    def pressure_at_z(value: float) -> float:
        return study.analytic_pressure(r, value, phi, profile)[0]

    def pressure_at_phi(value: float) -> float:
        return study.analytic_pressure(r, z, value, profile)[0]

    step = 1.0e-4
    expected = (
        _five_point_derivative(pressure_at_r, r, step),
        _five_point_derivative(pressure_at_phi, phi, step) / r,
        _five_point_derivative(pressure_at_z, z, step),
    )
    np.testing.assert_allclose(
        (radial, azimuthal, axial),
        expected,
        rtol=2.0e-9,
        atol=2.0e-10,
    )


@pytest.mark.parametrize("profile", ("compact", "wall"))
def test_analytic_pressure_and_rz_gradient_vanish_on_reference_walls(
    profile: str,
) -> None:
    side = np.linspace(0.0, 1.0, 17)
    rectangle_r = np.concatenate(
        (
            np.ones_like(side),
            2.0 * np.ones_like(side),
            np.linspace(1.0, 2.0, 17),
            np.linspace(1.0, 2.0, 17),
        )
    )
    rectangle_z = np.concatenate(
        (
            side,
            side,
            np.zeros_like(side),
            np.ones_like(side),
        )
    )
    angle = 2.0 * math.pi * np.arange(128, dtype=np.float64) / 128.0
    disk_r = 1.5 + 0.45 * np.cos(angle)
    disk_z = 0.5 + 0.45 * np.sin(angle)
    for r, z in ((rectangle_r, rectangle_z), (disk_r, disk_z)):
        phi = np.linspace(-math.pi, math.pi, max(r.size, 2), endpoint=False)
        variation, radial, azimuthal, axial = study.analytic_pressure(
            r[:, None], z[:, None], phi[None, :], profile
        )
        for values in (variation, radial, azimuthal, axial):
            assert float(np.max(np.abs(values))) <= 1.0e-14


@pytest.mark.parametrize("shape", ("rectangle", "disk"))
def test_fixed_geometry_uses_reference_domain_and_reference_regions(
    shape: str,
) -> None:
    nodes = 6
    fixed = study.fixed_geometry(nodes, shape)
    geometry = fixed.geometry
    h = 1.0 / nodes
    centers = 1.0 + (np.arange(nodes, dtype=np.float64) + 0.5) * h
    z_centers = (np.arange(nodes, dtype=np.float64) + 0.5) * h
    np.testing.assert_array_equal(geometry.R, centers)
    np.testing.assert_array_equal(geometry.Z, z_centers)
    assert geometry.R[0] - 0.5 * h == pytest.approx(1.0)
    assert geometry.R[-1] + 0.5 * h == pytest.approx(2.0)
    assert geometry.Z[0] - 0.5 * h == pytest.approx(0.0)
    assert geometry.Z[-1] + 0.5 * h == pytest.approx(1.0)
    assert fixed.nodes == nodes
    assert fixed.h == h
    assert fixed.active_cells == int(np.count_nonzero(geometry.active))
    assert fixed.cell_mask_area_m2 == pytest.approx(
        fixed.active_cells * h * h, rel=0.0, abs=0.0
    )
    assert fixed.toroidal_volume_m3 == pytest.approx(
        2.0 * math.pi * float(np.sum(geometry.M0)),
        rel=2.0e-15,
        abs=0.0,
    )
    assert fixed.hausdorff_upper_m >= fixed.hausdorff_lower_m >= 0.0
    if shape == "rectangle":
        assert len(geometry.polygon) == 4
        assert fixed.active_cells == nodes * nodes
        assert fixed.cell_mask_area_m2 == pytest.approx(1.0, abs=2.0e-15)
        assert fixed.analytic_target_area_m2 == 1.0
    else:
        assert len(geometry.polygon) == 128
        assert fixed.active_cells < nodes * nodes
        assert fixed.analytic_target_area_m2 == pytest.approx(
            math.pi * 0.45**2, rel=1.0e-15
        )

    masks = study.fixed_region_masks(
        geometry.cells.R, geometry.cells.Z, shape, width=0.15
    )
    independent_distance = (
        np.minimum.reduce(
            (
                geometry.cells.R - 1.0,
                2.0 - geometry.cells.R,
                geometry.cells.Z,
                1.0 - geometry.cells.Z,
            )
        )
        if shape == "rectangle"
        else 0.45
        - np.hypot(geometry.cells.R - 1.5, geometry.cells.Z - 0.5)
    )
    assert np.all(masks["all"])
    np.testing.assert_array_equal(masks["inner"], independent_distance > 0.15)
    np.testing.assert_array_equal(masks["wall"], ~masks["inner"])
    assert not np.any(masks["inner"] & masks["wall"])


def test_weighted_norms_apply_phase_weight_and_report_empty_regions() -> None:
    samples = 4
    components = 3
    entities = 5
    actual = np.zeros((samples, components, entities), dtype=np.complex128)
    reference = np.zeros_like(actual)
    actual[:, 0, 0] = 2.0
    weights = np.arange(1.0, entities + 1.0)
    regions = {
        "all": np.ones(entities, dtype=bool),
        "inner": np.array([True, False, False, False, False]),
        "empty": np.zeros(entities, dtype=bool),
    }
    phi_weight = 2.0 * math.pi / samples
    norms = study.weighted_norms(
        actual,
        reference,
        weights,
        regions,
        phi_weight=phi_weight,
        unit="Pa/m",
        component_names=("R", "phi", "Z"),
    )

    all_region = norms["all"]
    assert all_region["available"]
    assert all_region["counts"]["phi_samples"] == samples
    assert all_region["counts"]["components"] == components
    assert all_region["counts"]["entity_count"] == entities
    assert all_region["counts"]["value_count"] == samples * components * entities
    assert all_region["weighted_rms"] == pytest.approx(
        math.sqrt(
            16.0 / (samples * components * float(np.sum(weights)))
        ),
        rel=2.0e-15,
    )
    assert all_region["weighted_l2"] == pytest.approx(
        math.sqrt(phi_weight * 16.0), rel=2.0e-15
    )
    assert all_region["linf"] == 2.0
    assert all_region["reference_weighted_l2"] == 0.0
    assert all_region["relative_weighted_l2"] is None
    assert "zero reference" in all_region["relative_reason"]
    assert norms["inner"]["components"]["R"]["weighted_rms"] == pytest.approx(
        2.0, rel=2.0e-15
    )

    empty = norms["empty"]
    assert not empty["available"]
    assert empty["counts"]["value_count"] == 0
    assert empty["weighted_l2"] is None
    assert empty["linf"] is None
    assert empty["reason"] == "empty region"


@pytest.fixture(scope="module")
def small_context():
    return study.build_context(
        4,
        "rectangle",
        "cpu",
        memory_cap=512 << 20,
    )


def test_small_n4_production_case_retains_coordinates_fields_and_norms(
    small_context,
) -> None:
    result = study.spatial_case(small_context, "rectangle", "compact")
    assert small_context.p.K == 1
    assert small_context.N == 5
    assert small_context.op.K == 1
    assert result.metadata["nodes"] == 4
    assert result.metadata["shape"] == "rectangle"
    assert result.metadata["profile"] == "compact"
    assert result.metadata["force"]["raw_full_force"][
        "continuum_pass_fail_eligible"
    ] is False
    required = {
        "cell_r_m",
        "cell_z_m",
        "edge_r_m",
        "edge_z_m",
        "pressure_variation_Pa",
        "B_node_mode_vector_T",
        "B_cell_samples_T",
        "raw_full_force_Pa_per_m",
        "raw_delta_force_Pa_per_m",
        "projected_full_force_Pa_per_m",
        "constant_projected_force_Pa_per_m",
        "force_exact_Pa_per_m",
        "E0_V_per_m",
        "E_V_per_m",
        "E_exact_V_per_m",
        "E_correction_V_per_m",
        "wall_correction_residual",
        "wall_correction_rhs",
    }
    assert required <= set(result.arrays)
    for name, values in result.arrays.items():
        assert isinstance(values, np.ndarray), name
        assert values.dtype != object, name
        assert np.all(np.isfinite(values)), name
    correction = result.metadata["ohm"]["correction"]
    assert correction["off_wall_linf"] <= correction["confined_limit"]
    assert correction["confined_to_actual_wall_slots"]
    assert correction["residual_role"] == (
        "algebraic original-row correction residual; not continuum error"
    )
    magnetic = result.arrays["B_cell_samples_T"]
    assert np.max(np.abs(magnetic[:, 0, :])) == 0.0
    assert np.min(np.abs(magnetic[:, 1, :])) > small_context.p.B_min
    retained = result.arrays["B_node_mode_vector_T"].reshape(3, -1)[1]
    flux = (small_context.op.topology_flux_row @ retained).item().real
    assert flux == pytest.approx(.1*math.log(2.0), rel=1e-12)
    np.testing.assert_allclose(small_context.op.Cdag[1] @ retained, 0., atol=1e-12)
    assert result.metadata["full_stage_hard_checks_pass"]
    assert np.max(np.abs(magnetic[:, 2, :])) == 0.0


def test_exact_ohm_reference_does_not_use_production_projection(
    small_context,
    monkeypatch,
) -> None:
    baseline = study.spatial_case(small_context, "rectangle", "compact")

    def wrong_projection(nodal, density, context):
        return context.compute.array(
            np.zeros(context.N * 3 * context.op.Nc, dtype=np.complex128)
        )

    monkeypatch.setattr(massless, "_constraint_project", wrong_projection)
    with pytest.raises(Failure, match="electron_pressure_adjoint"):
        study.spatial_case(small_context, "rectangle", "compact")
    assert baseline.metadata["ohm"]["reference_path"] == (
        "direct analytic evaluation at production sample locations; "
        "no I/Q/H/G/C application"
    )


def test_source_snapshot_partition_supports_src_layout_and_imported_path(
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "layout"
    fake_package = fake_root / "src" / "fomhd"
    fake_package.mkdir(parents=True)
    assert study._resolve_source_parent(fake_root) == fake_root / "src"

    imported_package = Path(fomhd.__file__).resolve().parent
    assert study.PACKAGE_DIR == imported_package
    fixture = Path(study.massless_verification_fixture.__file__).resolve()
    harness = (
        ROOT / "tools" / "warp_spatial_study.py",
        ROOT / "tests" / "warp" / "test_spatial_study.py",
        ROOT / "docs" / "pseudocode" / "tools" / "warp_spatial_study.py.md",
    )
    snapshot = tmp_path / "snapshot"
    manifest = study.snapshot_source(
        snapshot,
        package_dir=imported_package,
        fixture_paths=(fixture,),
        harness_paths=harness,
    )
    entries = manifest["files"]
    assert any(
        entry["source_path"] == str(imported_package / "__init__.py")
        for entry in entries
    )
    assert any(entry["source_path"] == str(fixture) for entry in entries)
    assert (snapshot / "SOURCE_MANIFEST.json").is_file()


def test_cuda_context_refuses_without_owned_slurm_allocation(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(Failure, match="Slurm") as error:
        study.build_context(4, "rectangle", "cuda", memory_cap=512 << 20)
    assert error.value.code == "Backend"
