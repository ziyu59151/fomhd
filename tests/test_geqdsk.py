"""Independent polynomial fixture for the auxiliary G-EQDSK reader."""
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from geqdsk import read_geqdsk, reconstruct


class EquilibriumInput(unittest.TestCase):
    def fixture(self, path):
        r = np.array([1.0, 2.0, 3.0])
        z = np.array([-1.0, 0.0, 1.0])
        psi = r[None, :] ** 2 + 2 * z[:, None] ** 2
        scalars = [2, 2, 2, 1, 0, 1, 0, 1, 11, 3] + [0] * 10
        values = scalars + [6] * 3 + [0] * 9 + psi.ravel().tolist() + [1] * 3
        values += [4, 0, 1, -1, 3, -1, 3, 1, 1, 1]
        text = "POLYNOMIAL TEST                                      0   3   3\n"
        text += "\n".join("".join(f"{v:16.9E}" for v in values[i:i + 5]) for i in range(0, len(values), 5))
        path.write_text(text)

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


if __name__ == "__main__":
    unittest.main()
