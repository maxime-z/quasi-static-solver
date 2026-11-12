"""Block scaling, Schur scaling, and recovery of the physical coefficient."""

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from darwin.assemble import System
from darwin.solve import prepare, solve_direct


def _saddle() -> tuple[System, np.ndarray]:
    """One edge, one node, one multiplier. The field block is invertible."""
    matrix = np.array(
        [
            [4.0, 1.0, 1.0],
            [1.0, 4.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=complex,
    )
    physical = np.array([0.5, -0.25, 0.3], dtype=complex)
    load = matrix @ physical
    system = System(
        csr_matrix(matrix),
        load,
        np.array([0]),
        np.array([1]),
        np.array([2]),
    )
    return system, physical


class TestScaling(unittest.TestCase):
    def test_scaled_solution_recovers_the_physical_coefficient(self):
        system, physical = _saddle()
        prepared = prepare(system)
        self.assertFalse(np.allclose(prepared.unknown_scale, 1.0))

        recovered = solve_direct(system)
        np.testing.assert_allclose(recovered, physical, rtol=1e-10, atol=1e-10)

        # The solved vector of the scaled matrix is not the physical one.
        scaled_unknown = recovered / prepared.unknown_scale
        self.assertGreater(np.linalg.norm(scaled_unknown - physical), 1e-8)

    def test_congruence_uses_the_same_scale_as_the_recovery(self):
        system, _ = _saddle()
        prepared = prepare(system)
        y = np.array([0.2, -0.7, 1.5], dtype=complex)
        x = prepared.unknown_scale * y
        left = prepared.system.k_globals @ y
        right = prepared.unknown_scale * (system.k_globals @ x)
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
