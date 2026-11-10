"""Tree-cotree gauge on the full edge-node system."""

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from darwin.assemble import System
from darwin.gauge import GaugeKind, apply_gauge
from darwin.topology import Boundary, Topology


def _line_topology() -> Topology:
    """Four nodes in a path. Edges 0 and 2 can stand for two disjoint walls."""
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    incidence = np.zeros((3, 4), dtype=int)
    for e, (n1, n2) in enumerate(edges):
        incidence[e, n1] = -1
        incidence[e, n2] = 1
    skin = Boundary(np.arange(4), np.array([0, 2]))
    return Topology(edges, np.zeros((0, 3), dtype=int), incidence, skin)


def _system(matrix: np.ndarray, load: np.ndarray) -> System:
    n_edges = 3
    return System(
        csr_matrix(matrix),
        load,
        np.arange(n_edges),
        np.arange(4) + n_edges,
    )


class TestTreeCotree(unittest.TestCase):
    def test_one_root_component_ties_the_other_with_one_edge(self):
        # Edges 0 and 2 are already a_e = 0, so {0, 1} and {2, 3} are two walls.
        # The single free edge (1, 2) must become the one tree edge. A nodal
        # entry in that column must be cleared, and the diagonal must stay 1.
        matrix = np.zeros((7, 7), dtype=complex)
        matrix[0, 0] = 1.0
        matrix[2, 2] = 1.0
        matrix[1, 1] = 4.0
        matrix[1, 6] = 3.0
        matrix[6, 1] = 3.0
        matrix[6, 6] = 8.0
        load = np.array([0, 5, 0, 0, 0, 0, 7], dtype=complex)

        gauged = apply_gauge(_system(matrix, load), _line_topology(), GaugeKind.TREE)
        k = gauged.k_globals.toarray()

        self.assertTrue(np.isclose(k[1, 1], 1.0))
        self.assertTrue(np.allclose(k[1, :], np.eye(7)[1]))
        self.assertTrue(np.allclose(k[:, 1], np.eye(7)[:, 1]))
        self.assertTrue(np.isclose(gauged.f_globals[1], 0.0))
        self.assertTrue(np.isclose(k[6, 6], 8.0))
        self.assertTrue(np.isclose(gauged.f_globals[6], 7.0))
        self.assertTrue(np.isclose(k[0, 0], 1.0))
        self.assertTrue(np.isclose(k[2, 2], 1.0))

    def test_no_fixed_edge_roots_the_tree_at_node_zero(self):
        matrix = np.eye(7, dtype=complex)
        matrix[0, 0] = 2.0
        matrix[1, 1] = 2.0
        matrix[2, 2] = 2.0
        load = np.ones(7, dtype=complex)

        gauged = apply_gauge(_system(matrix, load), _line_topology(), GaugeKind.TREE)
        k = gauged.k_globals.toarray()
        tree = [e for e in range(3) if np.count_nonzero(k[e, :]) == 1 and np.isclose(k[e, e], 1.0)]
        # Node 0 is the only root, so the path of three edges is the whole tree.
        self.assertEqual(tree, [0, 1, 2])
        self.assertTrue(all(np.isclose(gauged.f_globals[e], 0.0) for e in tree))
        self.assertTrue(np.allclose(gauged.f_globals[3:], np.ones(4)))


if __name__ == "__main__":
    unittest.main()
