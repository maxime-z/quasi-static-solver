"""The three dimensional manufactured solution converges under refinement.

``manufactured_3d.errors`` solves the Darwin system on an n x n x n cube and
measures the relative L2 error of E and B against the polynomial solution.
One refinement is enough for a unit test: the errors must drop.  The full
order study is the script itself, run as

    venv/bin/python tests/manufactured_3d.py
"""

import sys
import unittest

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import manufactured_3d


class TestManufacturedDarwin3D(unittest.TestCase):
    def test_errors_drop_from_two_to_three_cells(self):
        coarse_e, coarse_b = manufactured_3d.errors(2)
        fine_e, fine_b = manufactured_3d.errors(3)
        self.assertGreater(coarse_e / fine_e, 1.15, f"E {coarse_e:.3e} -> {fine_e:.3e}")
        self.assertGreater(coarse_b / fine_b, 1.15, f"B {coarse_b:.3e} -> {fine_b:.3e}")

    def test_exact_fields_are_a_curl_pair(self):
        """B is the curl of A, checked on a grid of points by finite differences."""
        h = 1.0e-5
        points = np.array(
            [
                [0.13, -0.27, 0.41],
                [-0.62, 0.35, 0.08],
                [0.74, 0.51, -0.33],
            ]
        )
        pairs = []
        for axis in range(3):
            step = np.zeros(3)
            step[axis] = h
            pairs.append((manufactured_3d._vector_potential(points + step)
                          - manufactured_3d._vector_potential(points - step)) / (2 * h))
        curl = np.empty_like(points)
        curl[:, 0] = pairs[1][:, 2] - pairs[2][:, 1]
        curl[:, 1] = pairs[2][:, 0] - pairs[0][:, 2]
        curl[:, 2] = pairs[0][:, 1] - pairs[1][:, 0]
        np.testing.assert_allclose(curl, manufactured_3d._magnetic_field(points), rtol=1e-6, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
