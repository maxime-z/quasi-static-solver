"""Tree-cotree gauge on the full edge-node system."""

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from darwin.assemble import System, assemble
from darwin.bc import DirichletWall, apply_dirichlet
from darwin.elements import TriangleWhitney
from darwin.gauge import GaugeKind, apply_gauge
from darwin.material import Material
from darwin.mesh_io import Mesh
from darwin.topology import Boundary, Topology, build_topology, tag_boundary
from darwin.visu import plot_configuration
from pathlib import Path

result_path = Path(__file__).parent / "results"


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


def _grid_square(origin: tuple[float, float], n: int = 2, size: float = 1.0):
    """(n+1)^2 nodes and 2 n^2 triangles over a square, with '/' diagonals."""
    ox, oy = origin
    h = size / n
    points = np.array(
        [(ox + i * h, oy + j * h) for j in range(n + 1) for i in range(n + 1)],
        dtype=float,
    )
    cells = []
    for j in range(n):
        for i in range(n):
            n0 = j * (n + 1) + i
            n1 = n0 + 1
            n2 = n0 + (n + 1)
            n3 = n2 + 1
            cells.append([n0, n1, n3])
            cells.append([n0, n3, n2])
    return points, np.array(cells, dtype=int)


def _disjoint_squares():
    """Two unit squares at x in [0, 1] and x in [2, 3], with no shared node."""
    points_a, cells_a = _grid_square((0.0, 0.0))
    points_b, cells_b = _grid_square((2.0, 0.0))
    n_a = points_a.shape[0]
    mesh = Mesh(
        np.vstack([points_a, points_b]),
        np.vstack([cells_a, cells_b + n_a]),
        np.zeros(2 * cells_a.shape[0], dtype=int),
        {0: "body"},
    )
    tags = tag_boundary(
        mesh,
        {
            "A": lambda p: p[0] <= 1.01 and p[1] <= 1.01,
            "B": lambda p: p[0] >= 1.99 and p[1] <= 1.01,
        },
    )
    topology = build_topology(mesh, boundary_tags=tags)

    plot_configuration(mesh, topology, result_path / "disjoint_squares.png")
    return mesh, topology


def _assemble(mesh: Mesh, topology: Topology) -> System:
    material = Material("body", sigma=1.0, mu_r=1.0, eps_r=1.0)
    return assemble(
        mesh,
        topology,
        {0: material},
        frequency=1.0e6,
        source=None,
        element=TriangleWhitney(),
    )


def _identity_rows(K) -> set[int]:
    rows = set()
    for i in range(K.shape[0]):
        row = K.getrow(i)
        if row.nnz == 1 and int(row.indices[0]) == i and np.isclose(row.data[0], 1.0):
            rows.add(i)
    return rows


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
        k = gauged.system.k_globals.toarray()

        np.testing.assert_array_equal(gauged.tree_edges, [1])
        self.assertTrue(np.isclose(k[1, 1], 1.0))
        self.assertTrue(np.allclose(k[1, :], np.eye(7)[1]))
        self.assertTrue(np.allclose(k[:, 1], np.eye(7)[:, 1]))
        self.assertTrue(np.isclose(gauged.system.f_globals[1], 0.0))
        self.assertTrue(np.isclose(k[6, 6], 8.0))
        self.assertTrue(np.isclose(gauged.system.f_globals[6], 7.0))
        self.assertTrue(np.isclose(k[0, 0], 1.0))
        self.assertTrue(np.isclose(k[2, 2], 1.0))

    def test_no_fixed_edge_roots_the_tree_at_node_zero(self):
        matrix = np.eye(7, dtype=complex)
        matrix[0, 0] = 2.0
        matrix[1, 1] = 2.0
        matrix[2, 2] = 2.0
        load = np.ones(7, dtype=complex)

        gauged = apply_gauge(_system(matrix, load), _line_topology(), GaugeKind.TREE)
        # Node 0 is the only root, so the path of three edges is the whole tree.
        np.testing.assert_array_equal(gauged.tree_edges, [0, 1, 2])
        self.assertTrue(
            all(np.isclose(gauged.system.f_globals[e], 0.0) for e in gauged.tree_edges)
        )
        self.assertTrue(np.allclose(gauged.system.f_globals[3:], np.ones(4)))

    def test_disjoint_squares_get_one_tree_per_component(self):
        # Each square is a separate boundary component when both walls are
        # grounded. One free edge per square must become its tree edge, even
        # though no free edge connects the two squares.
        mesh, topology = _disjoint_squares()
        system = _assemble(mesh, topology)
        grounded = apply_dirichlet(
            system,
            topology,
            [DirichletWall("A", 0.0), DirichletWall("B", 0.0)],
        )

        gauged = apply_gauge(grounded, topology, GaugeKind.TREE)

        before = _identity_rows(grounded.k_globals.tocsr())
        after = _identity_rows(gauged.system.k_globals.tocsr())
        self.assertEqual(set(gauged.tree_edges.tolist()), after - before)
        self.assertEqual(len(gauged.tree_edges), 2)

        n_edges = topology.edges.shape[0]
        k_aa = gauged.system.k_globals[:n_edges, :n_edges]
        self.assertEqual(np.linalg.matrix_rank(k_aa.toarray()), n_edges)
        self.assertEqual(
            np.linalg.matrix_rank(gauged.system.k_globals.toarray()),
            gauged.system.k_globals.shape[0],
        )

    def test_disjoint_square_without_reference_keeps_one_constant(self):
        # The second square has no phi reference. The gauge still spans its
        # free edges, but the constant phi of that square survives: nullity 1.
        mesh, topology = _disjoint_squares()
        system = _assemble(mesh, topology)
        grounded = apply_dirichlet(system, topology, [DirichletWall("A", 0.0)])

        gauged = apply_gauge(grounded, topology, GaugeKind.TREE)

        before = _identity_rows(grounded.k_globals.tocsr())
        after = _identity_rows(gauged.system.k_globals.tocsr())
        self.assertEqual(set(gauged.tree_edges.tolist()), after - before)
        self.assertEqual(len(gauged.tree_edges), 9)
        nullity = gauged.system.k_globals.shape[0] - np.linalg.matrix_rank(
            gauged.system.k_globals.toarray()
        )
        self.assertEqual(nullity, 1)


def _coulomb_block(system: System, n_edges: int):
    p_dofs = system.p_dofs
    assert p_dofs is not None
    return system.k_globals[:n_edges, p_dofs]


class TestCoulomb(unittest.TestCase):
    def test_grounded_disjoint_squares_drop_no_extra_column(self):
        mesh, topology = _disjoint_squares()
        system = _assemble(mesh, topology)
        grounded = apply_dirichlet(
            system,
            topology,
            [DirichletWall("A", 0.0), DirichletWall("B", 0.0)],
        )
        gauged = apply_gauge(grounded, topology, GaugeKind.COULOMB)
        n_edges = topology.edges.shape[0]
        n_interior = 2
        self.assertEqual(len(gauged.tree_edges), 0)
        self.assertIsNotNone(gauged.system.p_dofs)
        self.assertEqual(len(gauged.system.p_dofs), n_interior)
        c = _coulomb_block(gauged.system, n_edges).toarray()
        self.assertEqual(np.linalg.matrix_rank(c), n_interior)
        self.assertEqual(
            np.linalg.matrix_rank(gauged.system.k_globals.toarray()),
            gauged.system.k_globals.shape[0],
        )

    def test_one_floating_square_drops_one_column_on_that_piece(self):
        mesh, topology = _disjoint_squares()
        system = _assemble(mesh, topology)
        grounded = apply_dirichlet(system, topology, [DirichletWall("A", 0.0)])
        gauged = apply_gauge(grounded, topology, GaugeKind.COULOMB)

        n_a = mesh.points.shape[0] // 2
        n_interior_a = 1
        n_p = n_interior_a + (n_a - 1)
        n_edges = topology.edges.shape[0]
        self.assertEqual(len(gauged.system.p_dofs), n_p)
        c = _coulomb_block(gauged.system, n_edges).toarray()
        self.assertEqual(np.linalg.matrix_rank(c), n_p)

        # Coulomb does not pin phi. The constant on the floating square is
        # still a kernel vector; the saddle itself is too ill-conditioned for
        # numpy's default rank cutoff to count that kernel as size 1.
        kernel = np.zeros(gauged.system.k_globals.shape[0], dtype=complex)
        kernel[n_edges + n_a : n_edges + 2 * n_a] = 1.0
        residual = gauged.system.k_globals @ kernel
        self.assertLess(
            np.linalg.norm(residual) / np.linalg.norm(kernel),
            1e-10,
        )


if __name__ == "__main__":
    unittest.main()
