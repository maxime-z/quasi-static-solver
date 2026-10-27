"""Global assembly of one triangle, two triangles, and one tetrahedron."""

import unittest

import numpy as np

from darwin.assemble import assemble
from darwin.elements import GaussianQuadrature, TetrahedronWhitney, TriangleWhitney
from darwin.material import EPSILON_0, MU_0, Material, materials_by_region
from darwin.mesh_io import Mesh
from darwin.topology import build_topology


def _mesh(points, cells, name="unit") -> Mesh:
    region_ids = np.zeros(len(cells), dtype=int)
    return Mesh(np.asarray(points, dtype=float), np.asarray(cells, dtype=int), region_ids, {0: name})


def _materials(mesh: Mesh, material: Material) -> dict[int, Material]:
    return materials_by_region(mesh, {material.name: material})


def _curl_only() -> Material:
    """nu = 1 and sigma = eps = 0, so K contains only the curl–curl block."""
    return Material(name="unit", sigma=0.0, mu_r=1.0 / MU_0, eps_r=0.0)


def _hand_triangle_aa() -> np.ndarray:
    curl = np.array([2.0, -2.0, 2.0])
    return 0.5 * np.outer(curl, curl)


def _hand_tet_aa() -> np.ndarray:
    grads = np.array(
        [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    curls = np.vstack([2.0 * np.cross(grads[i], grads[j]) for i, j in edges])
    return (1.0 / 6.0) * (curls @ curls.T)


class TestAssemble(unittest.TestCase):
    def test_one_triangle_matches_the_hand_curl_curl(self):
        mesh = _mesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]])
        topology = build_topology(mesh)
        system = assemble(
            mesh, topology, _materials(mesh, _curl_only()), 0.0, None, TriangleWhitney()
        )

        k_global = system.k_globals.toarray()
        self.assertEqual(k_global.shape, (6, 6))
        np.testing.assert_allclose(k_global[:3, :3], _hand_triangle_aa(), atol=1e-12)
        np.testing.assert_allclose(k_global[:3, 3:], 0.0, atol=1e-12)
        np.testing.assert_allclose(k_global[3:, :3], 0.0, atol=1e-12)
        np.testing.assert_allclose(k_global[3:, 3:], 0.0, atol=1e-12)
        np.testing.assert_allclose(system.f_globals, 0.0, atol=1e-12)
        self.assertEqual(system.f_globals.shape, (6,))
        np.testing.assert_array_equal(system.edge_dofs, np.arange(3))
        np.testing.assert_array_equal(system.node_dofs, np.arange(3) + 3)

    def test_shared_edge_and_node_add_both_cells(self):
        # eps_r = 1/eps0 makes sigma + j*omega*eps = j*omega, so the nodal block is O(1).
        material = Material(name="unit", sigma=0.0, mu_r=1.0 / MU_0, eps_r=1.0 / EPSILON_0)
        mesh = _mesh(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            [[0, 1, 3], [1, 2, 3]],
        )
        topology = build_topology(mesh)
        element = TriangleWhitney()
        frequency = 1.0
        system = assemble(
            mesh, topology, _materials(mesh, material), frequency, None, element
        )

        gauss = GaussianQuadrature(2, 2)
        locals_aa = []
        locals_phiphi = []
        for cell in mesh.cells:
            local_of = {int(g): i for i, g in enumerate(cell)}
            cell_edges = topology.cell_edges[len(locals_aa)]
            edge_pair = [
                (local_of[int(n1)], local_of[int(n2)])
                for n1, n2 in topology.edges[cell_edges]
            ]
            blocks = element.darwin_local_forms(
                mesh.points[cell], edge_pair, frequency, material, gauss, None
            )
            locals_aa.append(blocks.K_AA)
            locals_phiphi.append(blocks.K_phiphi)

        k_global = system.k_globals.toarray()
        n_edges = len(topology.edges)
        # Global edge 2 is (1, 3): local slot 2 of cell 0 and local slot 1 of cell 1.
        shared = locals_aa[0][2, 2] + locals_aa[1][1, 1]
        np.testing.assert_allclose(k_global[2, 2], shared, atol=1e-12)
        # Global edge 0 belongs only to the first cell.
        np.testing.assert_allclose(k_global[0, 0], locals_aa[0][0, 0], atol=1e-12)
        # Node 1 is local vertex 1 of cell 0 and local vertex 0 of cell 1.
        shared_node = locals_phiphi[0][1, 1] + locals_phiphi[1][0, 0]
        np.testing.assert_allclose(k_global[n_edges + 1, n_edges + 1], shared_node, atol=1e-12)
        self.assertGreater(abs(shared_node), 1e-8)

    def test_source_is_added_to_the_load_only(self):
        mesh = _mesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]])
        topology = build_topology(mesh)
        materials = _materials(mesh, _curl_only())
        element = TriangleWhitney()

        def source(_x):
            return np.array([1.0, 0.0])

        plain = assemble(mesh, topology, materials, 0.0, None, element)
        loaded = assemble(mesh, topology, materials, 0.0, source, element)

        np.testing.assert_allclose(
            loaded.k_globals.toarray(), plain.k_globals.toarray(), atol=1e-12
        )
        self.assertGreater(np.max(np.abs(loaded.f_globals)), 1e-8)
        np.testing.assert_allclose(plain.f_globals, 0.0, atol=1e-12)

    def test_reversed_second_cell_matches_after_renumbering_edges(self):
        material = Material(name="unit", sigma=0.0, mu_r=1.0 / MU_0, eps_r=1.0)
        points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        mesh_a = _mesh(points, [[0, 1, 3], [1, 2, 3]])
        mesh_b = _mesh(points, [[0, 1, 3], [3, 1, 2]])
        topo_a = build_topology(mesh_a)
        topo_b = build_topology(mesh_b)
        kwargs = dict(frequency=1.0, source=None, element=TriangleWhitney())
        k_a = assemble(mesh_a, topo_a, _materials(mesh_a, material), **kwargs).k_globals.toarray()
        k_b = assemble(mesh_b, topo_b, _materials(mesh_b, material), **kwargs).k_globals.toarray()

        b_of = {tuple(int(n) for n in edge): i for i, edge in enumerate(topo_b.edges)}
        edge_perm = np.array([b_of[tuple(int(n) for n in edge)] for edge in topo_a.edges])
        node_perm = len(topo_a.edges) + np.arange(len(mesh_a.points))
        perm = np.concatenate([edge_perm, node_perm])
        np.testing.assert_allclose(k_a, k_b[np.ix_(perm, perm)], atol=1e-12)

    def test_reference_tet_matches_the_hand_curl_curl(self):
        mesh = _mesh(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0, 1, 2, 3]],
        )
        topology = build_topology(mesh)
        system = assemble(
            mesh, topology, _materials(mesh, _curl_only()), 0.0, None, TetrahedronWhitney()
        )
        k_global = system.k_globals.toarray()
        self.assertEqual(k_global.shape, (10, 10))
        np.testing.assert_allclose(k_global[:6, :6], _hand_tet_aa(), atol=1e-12)
        np.testing.assert_allclose(k_global[:6, 6:], 0.0, atol=1e-12)
        np.testing.assert_allclose(k_global[6:, 6:], 0.0, atol=1e-12)
        np.testing.assert_allclose(system.f_globals, 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
