"""Independent polynomial and retained EFIT fixtures for the G-EQDSK reader."""
import hashlib
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from geqdsk import read_geqdsk, reconstruct
from tokamak_equilibrium import closed_flux_contour, sample_equilibrium

RETAINED_EFIT = pathlib.Path(
    "/home/lxzeng/runs/fomhd/a0-tokamak-cae__20261105T154538Z/"
    "inputs/g048059.03650"
)


class EquilibriumInput(unittest.TestCase):
    def fixture(
        self, path, *, tag="POLYNOMIAL TEST", extension="",
        boundary_count=4, limiter_count=0,
    ):
        r = np.array([1.0, 2.0, 3.0])
        z = np.array([-1.0, 0.0, 1.0])
        psi = r[None, :] ** 2 + 2 * z[:, None] ** 2
        scalars = [2, 2, 2, 1, 0, 1, 0, 1, 11, 3] + [0] * 10
        values = scalars + [6] * 3 + [0] * 9 + psi.ravel().tolist() + [1] * 3
        values += [
            boundary_count, limiter_count,
            1, -1, 3, -1, 3, 1, 1, 1,
        ]
        text = f"{tag:<54}0   3   3\n"
        text += "\n".join("".join(f"{v:16.9E}" for v in values[i:i + 5]) for i in range(0, len(values), 5))
        path.write_text(text + extension)

    def test_analytic_magnetic_field(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "gfile"
            self.fixture(path)
            equilibrium = read_geqdsk(path)
            field = reconstruct(equilibrium, psi_factor=1.0)
            np.testing.assert_allclose(field["B_R"], -4 * equilibrium["Z"][:, None] / equilibrium["R"][None, :])
            np.testing.assert_allclose(field["B_Z"], 2)
            np.testing.assert_allclose(field["B_phi"], np.broadcast_to(6 / equilibrium["R"], (3, 3)))
            self.assertEqual(equilibrium["boundary"].shape, (4, 2))

    def test_truncation_and_extension_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "gfile"
            self.fixture(path)
            text = path.read_text()
            path.write_text(text[:-25])
            with self.assertRaises(ValueError):
                read_geqdsk(path)
            path.write_text(text + "\n1.0\n")
            with self.assertRaises(ValueError):
                read_geqdsk(path)

    def test_efit_extension_is_retained_separately(self):
        extension = "\n 0 1.700000E+00 51\n SAIPR = 1.0,\n SAIIP = 2.0\n"
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "gfile"
            self.fixture(path, tag="EFITD TEST", extension=extension)
            equilibrium = read_geqdsk(path)
            np.testing.assert_allclose(
                equilibrium["psi"], np.array([[3.0, 6.0, 11.0], [1.0, 4.0, 9.0], [3.0, 6.0, 11.0]])
            )
            self.assertEqual(equilibrium["extension"], extension.encode("ascii"))
            self.assertEqual(equilibrium["extension_format"], "EFITD")
            self.assertEqual(
                equilibrium["extension_sha256"],
                hashlib.sha256(extension.encode("ascii")).hexdigest(),
            )

    def test_nonfinite_and_unbounded_counts_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "gfile"
            for count in (1e20, float("inf")):
                self.fixture(path, boundary_count=count)
                with self.assertRaises(ValueError):
                    read_geqdsk(path)

    @unittest.skipUnless(RETAINED_EFIT.is_file(), "retained EFIT input unavailable")
    def test_retained_efit_input_through_adapter(self):
        equilibrium = read_geqdsk(RETAINED_EFIT)
        self.assertEqual(
            equilibrium["sha256"],
            "df8f073bbad202d799b70d0bc2479c9cbcc2d64b729c1019f2575b14de961842",
        )
        self.assertEqual(equilibrium["psi"].shape, (129, 129))
        self.assertEqual(equilibrium["extension_format"], "EFITD")
        self.assertGreater(len(equilibrium["extension"]), 0)
        R, Z = equilibrium["magnetic_axis"]
        fields = sample_equilibrium(equilibrium, R, Z, psi_factor=1.0)
        self.assertLess(abs(float(fields["psi_normalized"])), 1e-7)
        self.assertTrue(all(np.isfinite(value).all() for value in fields.values()))
        contour = closed_flux_contour(equilibrium, level=0.8)
        self.assertEqual(contour.shape[1], 2)
        self.assertGreater(len(contour), 3)


if __name__ == "__main__":
    unittest.main()
