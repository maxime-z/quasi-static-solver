"""Rank of K_AA and K_phiphi on the two-triangle square, before any boundary condition.

Mesh
----
Nodes and cells, matching tests/test_assemble.py::test_shared_edge_and_node_add_both_cells::

    3 ---- 2
    | \\    |
    |  \\   |
    0 ---- 1

    cells = [[0, 1, 3], [1, 2, 3]]

build_topology discovers edges with combinations() in cell order, then sorts each
pair so the lower node index comes first. The five edges are therefore

    (0, 1), (0, 3), (1, 3), (1, 2), (2, 3).

The incidence matrix G is (n_edges, n_nodes). For edge i -> j with i < j,
G[e, i] = -1 and G[e, j] = +1. Column n of G is the edge coefficient vector of
the nodal hat phi_n: on edge i -> j the entry is phi_n(j) - phi_n(i).

K_AA
----
sigma = 0, so the edge block is only the curl-curl gram. A coefficient vector a
satisfies K_AA a = 0 when curl A vanishes on every triangle at once, with one
shared value a_e on each edge.

The hat at the shared node 1 is 1 at node 1 and 0 at nodes 0, 2, 3. Its column is

    edge (0, 1):  1 - 0 = +1     <- triangle (0, 1, 3)
    edge (0, 3):  0 - 0 =  0
    edge (1, 3):  0 - 1 = -1     <- shared by both triangles
    edge (1, 2):  0 - 1 = -1     <- triangle (1, 2, 3)
    edge (2, 3):  0 - 0 =  0

The first three entries are the local Whitney combination for grad lambda_1 on
(0, 1, 3). The last three edges, with values -1, -1, 0, are the local combination
on (1, 2, 3). One vector, not two: the shared edge is forced to carry -1 in both.

A vector taken from only one triangle is not in the kernel. Matching the gradient
on (0, 1, 3) and putting 0 on (1, 2) and (2, 3) leaves edge (1, 3) equal to -1
inside the second triangle, whose other two edges are 0. Those three numbers are
not differences of three vertex values, so the curl on (1, 2, 3) is nonzero.

The four columns, one per node, each lie in the kernel. They sum to 0 because
every edge row of G is -1 and +1, which is the statement

    grad phi_0 + grad phi_1 + grad phi_2 + grad phi_3 = 0

on the whole square. Inside one triangle the three barycentric gradients sum to
0; that is the same partition of unity, not an extra dependency. A continuous
piecewise-linear function has one constant mode on a connected mesh, so

    nullity(K_AA) = n_nodes - 1 = 3
    rank(K_AA)    = n_edges - (n_nodes - 1) = 5 - 3 = 2.

K_phiphi
--------
With sigma = 0 and omega * epsilon = 1 the nodal block is the stiffness
int grad lambda_i · grad lambda_j. The all-ones vector is the constant function,
because the hats sum to 1, and its gradient is 0. So K_phiphi @ ones = 0, and

    nullity(K_phiphi) = 1
    rank(K_phiphi)    = n_nodes - 1 = 3.
"""

import unittest

import numpy as np

from darwin.assemble import assemble
from darwin.elements import TriangleWhitney
from darwin.material import EPSILON_0, MU_0, Material
from darwin.mesh_io import Mesh
from darwin.topology import build_topology


class TestBlockRank(unittest.TestCase):
    def test_two_triangle_square_block_ranks(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        cells = np.array([[0, 1, 3], [1, 2, 3]])
        mesh = Mesh(points, cells, np.zeros(len(cells), dtype=int), {0: "unit"})
        topology = build_topology(mesh)

        # nu = 1/mu = 1, and sigma = 0 so K_AA has no edge mass to fill the kernel.
        # eps = 1 and frequency = 1/(2*pi) so omega = 1 and K_phiphi = j * grad-grad.
        material = Material(name="unit", sigma=0.0, mu_r=1.0 / MU_0, eps_r=1.0 / EPSILON_0)
        system = assemble(
            mesh,
            topology,
            {0: material},
            frequency=1.0 / (2.0 * np.pi),
            source=None,
            element=TriangleWhitney(),
        )

        edges = [tuple(int(n) for n in edge) for edge in topology.edges]
        self.assertEqual(edges, [(0, 1), (0, 3), (1, 3), (1, 2), (2, 3)])

        # Columns are grad phi_n. Column 1 is the shared-node vector in the docstring.
        expected_g = np.array(
            [
                [-1, 1, 0, 0],
                [-1, 0, 0, 1],
                [0, -1, 0, 1],
                [0, -1, 1, 0],
                [0, 0, -1, 1],
            ],
            dtype=float,
        )
        np.testing.assert_array_equal(topology.incidence_matrix, expected_g)
        self.assertEqual(int(np.linalg.matrix_rank(expected_g)), 3)

        k_global = system.k_globals.toarray()
        n_edges = len(edges)
        n_nodes = len(points)
        k_aa = k_global[:n_edges, :n_edges]
        k_phiphi = k_global[n_edges:, n_edges:]

        self.assertEqual(k_aa.shape, (5, 5))
        self.assertEqual(k_phiphi.shape, (4, 4))
        np.testing.assert_allclose(k_aa @ expected_g, 0.0, atol=1e-12)
        np.testing.assert_allclose(k_phiphi @ np.ones(n_nodes), 0.0, atol=1e-12)

        rank_k_aa = int(np.linalg.matrix_rank(k_aa))
        print(f"rank_k_aa = {rank_k_aa}")
        rank_k_phiphi = int(np.linalg.matrix_rank(k_phiphi))
        print(f"rank_k_phiphi = {rank_k_phiphi}")
        self.assertEqual(int(np.linalg.matrix_rank(k_aa)), n_edges - (n_nodes - 1))
        self.assertEqual(int(np.linalg.matrix_rank(k_phiphi)), n_nodes - 1)


if __name__ == "__main__":
    unittest.main()
