import unittest

import numpy as np

from darwin.material import Material, materials_by_region
from darwin.mesh_io import Mesh
from darwin.topology import build_topology, tag_boundary


def two_triangle_mesh(cells: np.ndarray, region_names: dict[int, str] | None = None) -> Mesh:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return Mesh(points, cells, np.array([0, 0]), region_names or {})


class TestTwoTriangleTopology(unittest.TestCase):
    def test_edges_cell_edges_and_incidence(self):
        mesh = two_triangle_mesh(np.array([[0, 1, 3], [1, 2, 3]]))
        topology = build_topology(mesh)
        expected_edges = np.array([[0, 1], [0, 3], [1, 3], [1, 2], [2, 3]])
        expected_cell_edges = np.array([[0, 1, 2], [3, 2, 4]])
        expected_g = np.array(
            [
                [-1, 1, 0, 0],
                [-1, 0, 0, 1],
                [0, -1, 0, 1],
                [0, -1, 1, 0],
                [0, 0, -1, 1],
            ]
        )
        np.testing.assert_array_equal(topology.edges, expected_edges)
        np.testing.assert_array_equal(topology.cell_edges, expected_cell_edges)
        np.testing.assert_array_equal(topology.incidence_matrix, expected_g)

    def test_reversed_cell_reuses_the_shared_edge(self):
        mesh = two_triangle_mesh(np.array([[0, 1, 3], [3, 1, 2]]))
        topology = build_topology(mesh)
        expected_edges = np.array([[0, 1], [0, 3], [1, 3], [2, 3], [1, 2]])
        expected_cell_edges = np.array([[0, 1, 2], [2, 3, 4]])
        np.testing.assert_array_equal(topology.edges, expected_edges)
        np.testing.assert_array_equal(topology.cell_edges, expected_cell_edges)
        self.assertTrue(np.all(topology.edges[:, 0] < topology.edges[:, 1]))

    def test_gradient_of_a_nodal_vector(self):
        mesh = two_triangle_mesh(np.array([[0, 1, 3], [1, 2, 3]]))
        topology = build_topology(mesh)
        q = np.array([1.0, 4.0, -2.0, 7.0])
        edge_from_grad = topology.incidence_matrix @ q
        by_hand = q[topology.edges[:, 1]] - q[topology.edges[:, 0]]
        np.testing.assert_allclose(edge_from_grad, by_hand)
        np.testing.assert_allclose(topology.incidence_matrix @ np.ones(4), np.zeros(5))

    def test_materials_by_region(self):
        mesh = two_triangle_mesh(
            np.array([[0, 1, 3], [1, 2, 3]]),
            region_names={0: "air"},
        )
        air = Material(name="air", sigma=0.0, mu_r=1.0, eps_r=1.0)
        materials = materials_by_region(mesh, {"air": air})
        self.assertEqual(materials, {0: air})

    def test_materials_by_region_missing_name(self):
        mesh = two_triangle_mesh(np.array([[0, 1, 3], [1, 2, 3]]))
        with self.assertRaises(KeyError):
            materials_by_region(mesh, {"air": Material("air", 0.0, 1.0, 1.0)})

    def test_four_walls_of_the_square(self):
        mesh = two_triangle_mesh(
            np.array([[0, 1, 3], [1, 2, 3]]),
            region_names={0: "air"},
        )
        predicates = {
            "bottom": lambda x: np.isclose(x[1], 0.0),
            "top": lambda x: np.isclose(x[1], 1.0),
            "right": lambda x: np.isclose(x[0], 1.0),
            "left": lambda x: np.isclose(x[0], 0.0),
        }
        facets, tags = tag_boundary(mesh, predicates)
        topology = build_topology(mesh, boundary_tags=(facets, tags))

        # Skin is the four outer edges; the diagonal (1, 3) is interior.
        np.testing.assert_array_equal(topology.skin.nodes, [0, 1, 2, 3])
        skin_edge_pairs = {tuple(topology.edges[e]) for e in topology.skin.edges}
        self.assertEqual(skin_edge_pairs, {(0, 1), (1, 2), (2, 3), (0, 3)})
        self.assertNotIn((1, 3), skin_edge_pairs)

        bottom = topology.boundary["bottom"]
        np.testing.assert_array_equal(bottom.nodes, [0, 1])
        self.assertEqual([tuple(topology.edges[e]) for e in bottom.edges], [(0, 1)])

        top = topology.boundary["top"]
        np.testing.assert_array_equal(top.nodes, [2, 3])
        self.assertEqual([tuple(topology.edges[e]) for e in top.edges], [(2, 3)])

        right = topology.boundary["right"]
        np.testing.assert_array_equal(right.nodes, [1, 2])
        self.assertEqual([tuple(topology.edges[e]) for e in right.edges], [(1, 2)])

        left = topology.boundary["left"]
        np.testing.assert_array_equal(left.nodes, [0, 3])
        self.assertEqual([tuple(topology.edges[e]) for e in left.edges], [(0, 3)])


if __name__ == "__main__":
    unittest.main()
