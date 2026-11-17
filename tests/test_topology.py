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


def single_tet_mesh() -> Mesh:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    return Mesh(points, np.array([[0, 1, 2, 3]]), np.array([0]), {0: "air"})


def cube_mesh() -> Mesh:
    """Six tetrahedra around the main diagonal (0, 6)."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    cells = np.array(
        [[0, 1, 2, 6], [0, 2, 3, 6], [0, 1, 5, 6], [0, 4, 5, 6], [0, 3, 7, 6], [0, 4, 7, 6]]
    )
    return Mesh(points, cells, np.zeros(6, dtype=int), {0: "air"})


def cube_predicates() -> dict[str, np.ndarray]:
    return {
        "xmin": lambda x: np.isclose(x[0], 0.0),
        "xmax": lambda x: np.isclose(x[0], 1.0),
        "ymin": lambda x: np.isclose(x[1], 0.0),
        "ymax": lambda x: np.isclose(x[1], 1.0),
        "zmin": lambda x: np.isclose(x[2], 0.0),
        "zmax": lambda x: np.isclose(x[2], 1.0),
    }


class TestTetrahedronTopology(unittest.TestCase):
    def test_single_tet_incidence_by_hand(self):
        topology = build_topology(single_tet_mesh())

        np.testing.assert_array_equal(
            topology.edges,
            np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]),
        )
        np.testing.assert_array_equal(
            topology.incidence_matrix,
            np.array(
                [
                    [-1, 1, 0, 0],
                    [-1, 0, 1, 0],
                    [-1, 0, 0, 1],
                    [0, -1, 1, 0],
                    [0, -1, 0, 1],
                    [0, 0, -1, 1],
                ]
            ),
        )
        np.testing.assert_array_equal(topology.cell_edges, np.arange(6).reshape(1, 6))

    def test_cube_counts(self):
        topology = build_topology(cube_mesh())

        self.assertEqual(len(topology.edges), 19)
        self.assertEqual(len(topology.cell_edges), 6)
        self.assertEqual(len(topology.skin.edges), 18)
        np.testing.assert_array_equal(topology.skin.nodes, np.arange(8))
        interior = set(range(len(topology.edges))) - set(topology.skin.edges)
        self.assertEqual(len(interior), 1)
        self.assertEqual(tuple(topology.edges[interior.pop()]), (0, 6))

    def test_cell_edges_follow_the_local_pairs(self):
        topology = build_topology(cube_mesh())
        edge_id = {tuple(edge): i for i, edge in enumerate(topology.edges)}

        for cell, local in zip(cube_mesh().cells, topology.cell_edges):
            pairs = [(int(cell[i]), int(cell[j])) for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]]
            np.testing.assert_array_equal(
                local, [edge_id[tuple(sorted(pair))] for pair in pairs]
            )

    def test_incidence_matrix_is_a_gradient(self):
        topology = build_topology(cube_mesh())
        q = np.array([1.0, 4.0, -2.0, 7.0, 0.5, -3.0, 2.0, 6.0])

        np.testing.assert_allclose(
            topology.incidence_matrix @ q,
            q[topology.edges[:, 1]] - q[topology.edges[:, 0]],
        )
        np.testing.assert_allclose(topology.incidence_matrix @ np.ones(8), np.zeros(19))

    def test_six_tags_on_the_cube(self):
        mesh = cube_mesh()
        facets, tags = tag_boundary(mesh, cube_predicates())
        topology = build_topology(mesh, boundary_tags=(facets, tags))

        self.assertEqual(set(topology.boundary), set(cube_predicates()))
        for name, boundary in topology.boundary.items():
            self.assertEqual(len(boundary.edges), 1, name)  # only the face diagonal
            self.assertEqual(len(boundary.nodes), 4, name)
        # The twelve cube edges are shared between two faces, so no tag owns them.
        tagged = {int(e) for b in topology.boundary.values() for e in b.edges}
        self.assertEqual(len(tagged), 6)
        self.assertEqual(len(set(topology.skin.edges) - tagged), 12)

    def test_facet_tags_are_used_when_no_explicit_tags_are_given(self):
        mesh = cube_mesh()
        facets, tags = tag_boundary(mesh, cube_predicates())
        mesh.facet_tags = {facet: tag for facet, tag in zip(facets, tags)}

        topology = build_topology(mesh)

        self.assertEqual(set(topology.boundary), set(cube_predicates()))


if __name__ == "__main__":
    unittest.main()
