"""Analytic checks for the CPU tokamak-equilibrium input adapter."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

from geqdsk import read_geqdsk
from tokamak_equilibrium import closed_flux_contour, sample_equilibrium


def _write_polynomial_gfile(path: pathlib.Path) -> None:
    R = np.linspace(0.1, 2.1, 41)
    Z = np.linspace(-1.0, 1.0, 41)
    psi = (R[None, :] - 1.0) ** 2 + Z[:, None] ** 2
    profile_x = np.linspace(0.0, 1.0, R.size)
    F = 4.0 + 2.0 * profile_x
    pressure = 100.0 - 20.0 * profile_x
    q = 1.5 + 3.0 * profile_x
    values = [
        R[-1] - R[0],
        Z[-1] - Z[0],
        1.0,
        R[0],
        0.5 * (Z[0] + Z[-1]),
        1.0,
        0.0,
        0.0,
        1.0,
        2.0,
        *([0.0] * 10),
        *F,
        *pressure,
        *np.zeros(R.size),
        *np.zeros(R.size),
        *psi.ravel(),
        *q,
        4.0,
        1.0,
        0.1,
        -1.0,
        2.1,
        -1.0,
        2.1,
        1.0,
        0.1,
        1.0,
        2.1,
        0.0,
    ]
    header = f"ANALYTIC POLYNOMIAL 0 {R.size:4d} {Z.size:4d}\n"
    body = "\n".join(
        "".join(f"{value:16.9E}" for value in values[start : start + 5])
        for start in range(0, len(values), 5)
    )
    path.write_text(header + body + "\n", encoding="ascii")


@pytest.fixture
def polynomial_eq(tmp_path):
    path = tmp_path / "polynomial.gfile"
    _write_polynomial_gfile(path)
    return read_geqdsk(path)


def test_sample_equilibrium_matches_polynomial_derivatives_and_profiles(
    polynomial_eq,
):
    R = np.array([[0.8], [1.2]])
    Z = np.array([[-0.2, 0.0, 0.2]])
    psi_factor = -2.5

    fields = sample_equilibrium(polynomial_eq, R, Z, psi_factor=psi_factor)

    expected_shape = (2, 3)
    assert set(fields) == {
        "psi_normalized",
        "B_R",
        "B_phi",
        "B_Z",
        "pressure_Pa",
        "F",
        "q",
    }
    for value in fields.values():
        assert value.shape == expected_shape
        assert np.isfinite(value).all()

    squeezed_R, squeezed_Z = np.broadcast_arrays(R, Z)
    psi_normalized = (squeezed_R - 1.0) ** 2 + squeezed_Z**2
    F = 4.0 + 2.0 * psi_normalized
    pressure = 100.0 - 20.0 * psi_normalized
    q = 1.5 + 3.0 * psi_normalized

    np.testing.assert_allclose(
        fields["psi_normalized"], psi_normalized, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        fields["B_R"],
        -psi_factor * 2.0 * squeezed_Z / squeezed_R,
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        fields["B_Z"],
        psi_factor * 2.0 * (squeezed_R - 1.0) / squeezed_R,
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(fields["B_phi"], F / squeezed_R)
    np.testing.assert_allclose(fields["F"], F)
    np.testing.assert_allclose(fields["pressure_Pa"], pressure)
    np.testing.assert_allclose(fields["q"], q)


def test_equilibrium_factor_scales_only_poloidal_field(polynomial_eq):
    R = np.array([0.8, 1.2, 1.6])
    Z = np.array([-0.2, 0.1, 0.2])

    positive = sample_equilibrium(polynomial_eq, R, Z, psi_factor=2.0)
    negative = sample_equilibrium(polynomial_eq, R, Z, psi_factor=-2.0)

    for name in ("psi_normalized", "B_phi", "pressure_Pa", "F", "q"):
        np.testing.assert_array_equal(positive[name], negative[name])
    np.testing.assert_array_equal(positive["B_R"], -negative["B_R"])
    np.testing.assert_array_equal(positive["B_Z"], -negative["B_Z"])


@pytest.mark.parametrize("psi_factor", [0.0, np.nan, np.inf, -np.inf])
def test_sample_equilibrium_rejects_invalid_factor(polynomial_eq, psi_factor):
    with pytest.raises(ValueError, match="psi_factor"):
        sample_equilibrium(polynomial_eq, 1.0, 0.0, psi_factor=psi_factor)


def test_sample_equilibrium_rejects_bad_query_domains(polynomial_eq):
    with pytest.raises(ValueError, match="broadcast"):
        sample_equilibrium(
            polynomial_eq, np.ones(2), np.ones(3), psi_factor=1.0
        )
    with pytest.raises(ValueError, match="source grid"):
        sample_equilibrium(polynomial_eq, 0.05, 0.0, psi_factor=1.0)
    with pytest.raises(ValueError, match="finite"):
        sample_equilibrium(polynomial_eq, np.nan, 0.0, psi_factor=1.0)
    with pytest.raises(ValueError, match="normalized flux"):
        sample_equilibrium(polynomial_eq, 0.1, -1.0, psi_factor=1.0)


def test_closed_flux_contour_is_ccw_circle_at_requested_level(polynomial_eq):
    level = 0.8
    contour = closed_flux_contour(polynomial_eq, level=level)

    assert isinstance(contour, np.ndarray)
    assert contour.ndim == 2
    assert contour.shape[1] == 2
    assert contour.shape[0] > 3
    assert np.isfinite(contour).all()
    assert not np.array_equal(contour[0], contour[-1])
    assert contour.flux_level == pytest.approx(level)

    R, Z = contour[:, 0], contour[:, 1]
    signed_area = 0.5 * np.sum(R * np.roll(Z, -1) - np.roll(R, -1) * Z)
    assert signed_area > 0.0
    radius = np.hypot(R - 1.0, Z)
    np.testing.assert_allclose(radius, np.sqrt(level), rtol=0.0, atol=0.012)


def test_closed_flux_contour_rejects_open_and_absent_contours(polynomial_eq):
    truncated = dict(polynomial_eq)
    R_mask = polynomial_eq["R"] <= 1.4
    truncated["R"] = polynomial_eq["R"][R_mask]
    truncated["psi"] = polynomial_eq["psi"][:, R_mask]
    for name in ("F", "pressure", "FFprime", "pprime", "q"):
        truncated[name] = polynomial_eq[name][R_mask]

    with pytest.raises(ValueError, match="open"):
        closed_flux_contour(truncated, level=0.8)

    absent = dict(polynomial_eq)
    absent["psi"] = 0.05 * polynomial_eq["psi"]
    with pytest.raises(ValueError, match="no closed"):
        closed_flux_contour(absent, level=0.8)


@pytest.mark.parametrize("level", [-0.1, 1.1, np.nan, np.inf])
def test_closed_flux_contour_rejects_invalid_level(polynomial_eq, level):
    with pytest.raises(ValueError, match="level"):
        closed_flux_contour(polynomial_eq, level=level)


def test_closed_flux_contour_rejects_nonfinite_source(polynomial_eq):
    bad = dict(polynomial_eq)
    bad["psi"] = polynomial_eq["psi"].copy()
    bad["psi"][5, 5] = np.nan
    with pytest.raises(ValueError, match="finite"):
        closed_flux_contour(bad, level=0.8)


def test_closed_axis_contour_ignores_unrelated_open_fragments():
    R=np.linspace(.1,5.,129);Z=np.linspace(-2.5,2.5,129)
    x=R[None,:]-2.
    eq=dict(R=R,Z=Z,psi=x*x+Z[:,None]**2-.2*x**4,
            psi_axis=0.,psi_boundary=1.,magnetic_axis=[2.,0.])
    contour=closed_flux_contour(eq,.3)
    assert np.min(contour[:,0])>1. and np.max(contour[:,0])<3.
