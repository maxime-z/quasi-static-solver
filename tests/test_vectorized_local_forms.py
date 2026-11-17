"""Smoke checks for batched Darwin local forms on triangle and tetrahedron."""

import unittest

import numpy as np

from darwin.elements import (
    GaussianQuadrature,
    TetrahedronWhitney,
    TriangleWhitney,
)
from darwin.formulation import Formulation
from darwin.material import MU_0, Material


class TestBatchedLocalForms(unittest.TestCase):
    def test_triangle_blocks_have_expected_shapes_and_charge_load(self):
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        edges = [(0, 1), (1, 2), (2, 0)]
        material = Material(name="unit", sigma=1.0, mu_r=1.0 / MU_0, eps_r=2.0)
        rho = 2.5
        blocks = TriangleWhitney().darwin_local_forms(
            Formulation.DARWIN,
            nodes,
            edges,
            1.0e6,
            material,
            GaussianQuadrature(2, 2),
            source=lambda x: np.array([0.0, 1.0]),
            charge=lambda _x: rho,
        )
        self.assertEqual(blocks.K_AA.shape, (3, 3))
        self.assertEqual(blocks.K_Aphi.shape, (3, 3))
        self.assertEqual(blocks.K_phiA.shape, (3, 3))
        self.assertEqual(blocks.K_phiphi.shape, (3, 3))
        self.assertEqual(blocks.mass_edge.shape, (3, 3))
        self.assertEqual(blocks.rhs_edge.shape, (3,))
        self.assertEqual(blocks.rhs_node.shape, (3,))
        np.testing.assert_allclose(blocks.rhs_node, rho / 6.0, atol=1e-12)

    def test_tetrahedron_blocks_have_expected_shapes(self):
        nodes = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        material = Material(name="unit", sigma=1.0, mu_r=1.0 / MU_0, eps_r=2.0)
        blocks = TetrahedronWhitney().darwin_local_forms(
            Formulation.DARWIN,
            nodes,
            edges,
            1.0e6,
            material,
            GaussianQuadrature(3, 2),
            source=lambda x: np.array([0.0, 0.0, 1.0]),
            charge=lambda _x: 1.25,
        )
        self.assertEqual(blocks.K_AA.shape, (6, 6))
        self.assertEqual(blocks.K_Aphi.shape, (6, 4))
        self.assertEqual(blocks.K_phiA.shape, (4, 6))
        self.assertEqual(blocks.K_phiphi.shape, (4, 4))
        self.assertEqual(blocks.mass_edge.shape, (6, 6))
        self.assertEqual(blocks.rhs_edge.shape, (6,))
        self.assertEqual(blocks.rhs_node.shape, (4,))
        self.assertTrue(np.all(np.isfinite(blocks.K_AA)))
        # int lambda_i over the reference tet is 1/24 for each node
        np.testing.assert_allclose(blocks.rhs_node, 1.25 / 24.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
