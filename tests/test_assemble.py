"""Global assembly of one triangle, two triangles, and one tetrahedron."""

import unittest

import numpy as np

from darwin.assemble import assemble
from darwin.elements import GaussianQuadrature, TetrahedronWhitney, TriangleWhitney
from darwin.formulation import (
    Formulation,
    NodalEquation,
    coefficients,
)
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
            Formulation.MQS,
            mesh,
            topology,
            _materials(mesh, _curl_only()),
            0.0,
            None,
            TriangleWhitney(),
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
            Formulation.DARWIN,
            mesh,
            topology,
            _materials(mesh, material),
            frequency,
            None,
            element,
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
                Formulation.DARWIN,
                mesh.points[cell],
                edge_pair,
                frequency,
                material,
                gauss,
                None,
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

        plain = assemble(
            Formulation.MQS, mesh, topology, materials, 0.0, None, element
        )
        loaded = assemble(
            Formulation.MQS, mesh, topology, materials, 0.0, source, element
        )

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
        k_a = assemble(
            Formulation.DARWIN, mesh_a, topo_a, _materials(mesh_a, material), **kwargs
        ).k_globals.toarray()
        k_b = assemble(
            Formulation.DARWIN, mesh_b, topo_b, _materials(mesh_b, material), **kwargs
        ).k_globals.toarray()

        b_of = {tuple(int(n) for n in edge): i for i, edge in enumerate(topo_b.edges)}
        edge_perm = np.array([b_of[tuple(int(n) for n in edge)] for edge in topo_a.edges])
        node_perm = len(topo_a.edges) + np.arange(len(mesh_a.points))
        perm = np.concatenate([edge_perm, node_perm])
        np.testing.assert_allclose(k_a, k_b[np.ix_(perm, perm)], atol=1e-12)

    def test_edge_mass_is_added_once(self):
        """K_AA(sigma=1) - K_AA(sigma=0) must be exactly j*omega*M."""
        element = TriangleWhitney()
        gauss = GaussianQuadrature(2, 2)
        triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        local_edges = [(0, 1), (1, 2), (2, 0)]
        frequency = 1.0
        conducting = Material(name="unit", sigma=1.0, mu_r=1.0 / MU_0, eps_r=0.0)
        insulating = Material(name="unit", sigma=0.0, mu_r=1.0 / MU_0, eps_r=0.0)

        with_mass = element.darwin_local_forms(
            Formulation.MQS, triangle, local_edges, frequency, conducting, gauss, None
        )
        without_mass = element.darwin_local_forms(
            Formulation.MQS, triangle, local_edges, frequency, insulating, gauss, None
        )
        omega = 2.0 * np.pi * frequency
        self.assertTrue(np.isrealobj(with_mass.mass_edge))
        np.testing.assert_allclose(
            with_mass.K_AA - without_mass.K_AA,
            1j * omega * with_mass.mass_edge,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_darwin_local_blocks_match_the_hand_integrals(self):
        """Complete Darwin block check on the reference right triangle.

        sigma, eps and omega are all nonzero, so every block is exercised:
        K_AA = curl-curl + j*omega*sigma*M, K_Aphi = (sigma + j*omega*eps)*D,
        K_phiA = j*omega*eps*D^T and K_phiphi = eps*G. Mixing up the phi_a
        coefficient (j*omega*sigma instead of j*omega*eps) fails here by 1e7.
        """
        triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        local_edges = [(0, 1), (0, 2), (1, 2)]
        sigma, eps_r, frequency = 1.0, 2.0, 1.0e6
        material = Material(name="unit", sigma=sigma, mu_r=1.0 / MU_0, eps_r=eps_r)
        omega = 2.0 * np.pi * frequency
        eps = eps_r * EPSILON_0

        blocks = TriangleWhitney().darwin_local_forms(
            Formulation.DARWIN,
            triangle,
            local_edges,
            frequency,
            material,
            GaussianQuadrature(2, 2),
        )

        # D[e, j] = int w_e . grad(lambda_j), G = int grad(lambda_i).grad(lambda_j),
        # M = int w_e . w_e' and curls = (2, -2, 2) on this triangle.
        d = np.array([[-0.5, 1.0 / 3.0, 1.0 / 6.0], [-0.5, 1.0 / 6.0, 1.0 / 3.0], [0.0, -1.0 / 6.0, 1.0 / 6.0]])
        g = np.array([[1.0, -0.5, -0.5], [-0.5, 0.5, 0.0], [-0.5, 0.0, 0.5]])
        m = np.array([[1.0 / 3.0, 1.0 / 6.0, 0.0], [1.0 / 6.0, 1.0 / 3.0, 0.0], [0.0, 0.0, 1.0 / 6.0]])
        curl = np.array([2.0, -2.0, 2.0])

        np.testing.assert_allclose(blocks.K_Aphi, (sigma + 1j * omega * eps) * d, atol=1e-12)
        np.testing.assert_allclose(blocks.K_phiA, 1j * omega * eps * d.T, rtol=1e-10, atol=1e-18)
        np.testing.assert_allclose(blocks.K_phiphi, eps * g, rtol=1e-10, atol=1e-24)
        np.testing.assert_allclose(
            blocks.K_AA, 0.5 * np.outer(curl, curl) + 1j * omega * sigma * m, atol=1e-6
        )
        np.testing.assert_allclose(blocks.mass_edge, m, atol=1e-12)
        np.testing.assert_allclose(blocks.rhs_edge, 0.0, atol=1e-12)
        np.testing.assert_allclose(blocks.rhs_node, 0.0, atol=1e-12)

    def test_darwin_charge_enters_the_nodal_load(self):
        """A constant rho gives int(lambda_i) rho = rho/6 on the reference triangle."""
        mesh = _mesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]])
        topology = build_topology(mesh)
        rho = 2.5
        system = assemble(
            Formulation.DARWIN,
            mesh,
            topology,
            _materials(mesh, Material(name="unit", sigma=1.0, mu_r=1.0 / MU_0, eps_r=2.0)),
            1.0e6,
            None,
            TriangleWhitney(),
            charge=lambda _point: rho,
        )

        self.assertEqual(system.formulation, Formulation.DARWIN)
        np.testing.assert_allclose(system.f_globals[:3], 0.0, atol=1e-12)
        np.testing.assert_allclose(system.f_globals[3:], rho / 6.0, atol=1e-12)

    def test_charge_is_rejected_by_the_mqs_continuity_row(self):
        mesh = _mesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]])
        topology = build_topology(mesh)
        with self.assertRaises(ValueError):
            assemble(
                Formulation.MQS,
                mesh,
                topology,
                _materials(mesh, _curl_only()),
                1.0e6,
                None,
                TriangleWhitney(),
                charge=lambda _point: 1.0,
            )

    def test_coefficients_follow_the_formulation(self):
        """Darwin uses (sigma + j*omega*eps, j*omega*eps, eps) + the Gauss row."""
        frequency = 1.0e6
        omega = 2.0 * np.pi * frequency
        eps = 2.0 * EPSILON_0
        material = Material(name="unit", sigma=1.0, mu_r=1.0, eps_r=2.0)

        darwin = coefficients(Formulation.DARWIN, material, frequency)
        self.assertEqual(darwin.nodal, NodalEquation.GAUSS)
        np.testing.assert_allclose(darwin.a_phi, 1.0 + 1j * omega * eps, rtol=1e-12)
        np.testing.assert_allclose(darwin.phi_a, 1j * omega * eps, rtol=1e-12)
        np.testing.assert_allclose(darwin.phi_phi, eps, rtol=1e-12)

        mqs = coefficients(Formulation.MQS, material, frequency)
        self.assertEqual(mqs.nodal, NodalEquation.CONTINUITY)
        np.testing.assert_allclose(mqs.a_phi, 1.0, rtol=1e-12)
        np.testing.assert_allclose(mqs.phi_a, 1j * omega * 1.0, rtol=1e-12)
        np.testing.assert_allclose(mqs.phi_phi, 1.0, rtol=1e-12)

        air = Material(name="air", sigma=0.0, mu_r=1.0, eps_r=0.0)
        with self.assertRaises(ValueError):
            coefficients(Formulation.DARWIN, air, frequency)

    def test_skewed_tet_matches_the_hand_darwin_blocks(self):
        """A tet with J != eye exercises the 3D metric in the curl-curl block."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.2, 0.1, 0.0],
                [0.2, 1.1, 0.05],
                [0.1, 0.3, 1.3],
            ]
        )
        mesh = _mesh(points, [[0, 1, 2, 3]])
        topology = build_topology(mesh)
        material = Material(name="unit", sigma=1.0e-2, mu_r=1.0, eps_r=3.0)
        frequency = 2.0e6
        omega = 2.0 * np.pi * frequency
        current = np.array([1.0, 2.0, 3.0])
        rho = 2.5
        element = TetrahedronWhitney()
        system = assemble(
            Formulation.DARWIN,
            mesh,
            topology,
            _materials(mesh, material),
            frequency,
            lambda x: current,
            element,
            charge=lambda x: rho,
            quadrature_degree=6,
        )
        k = system.k_globals.toarray()
        coeffs = coefficients(Formulation.DARWIN, material, frequency)
        nu = 1.0 / (material.mu_r * MU_0)

        # Barycentric coordinates of the physical tet: lambda_i(x) = row i.
        matrix = np.array([[1.0, 1.0, 1.0, 1.0], *points.T])
        inverse = np.linalg.inv(matrix)
        grads = inverse[:, 1:]  # (4, 3), row i = grad lambda_i
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        jacobian = np.column_stack(
            [points[1] - points[0], points[2] - points[0], points[3] - points[0]]
        )
        det_j = abs(np.linalg.det(jacobian))
        volume = det_j / 6.0

        quadrature = GaussianQuadrature(3, 6)
        lam = np.column_stack(
            [1.0 - quadrature.points.sum(axis=1), quadrature.points]
        )
        measure = quadrature.weights * det_j
        whitney = np.empty((len(measure), len(edges), 3))
        for a, (i, j) in enumerate(edges):
            whitney[:, a, :] = lam[:, i, None] * grads[j] - lam[:, j, None] * grads[i]
        curls = np.vstack([2.0 * np.cross(grads[i], grads[j]) for i, j in edges])

        gram = np.einsum("q,qab,qcb->ac", measure, whitney, whitney)
        mass = gram
        k_aa = nu * (curls @ curls.T) * volume + 1j * omega * material.sigma * mass
        k_a_phi = coeffs.a_phi * np.einsum("q,qab,ib->ai", measure, whitney, grads)
        k_phi_a = coeffs.phi_a * np.einsum("q,qab,ib->ia", measure, whitney, grads)
        k_phi_phi = coeffs.phi_phi * volume * (grads @ grads.T)
        rhs_edge = np.einsum("q,qab,b->a", measure, whitney, current)
        rhs_node = rho * (measure @ lam)

        n_edges = len(topology.edges)
        np.testing.assert_allclose(k[:n_edges, :n_edges], k_aa, rtol=1e-10, atol=1e-6)
        np.testing.assert_allclose(k[:n_edges, n_edges:], k_a_phi, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(k[n_edges:, :n_edges], k_phi_a, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(k[n_edges:, n_edges:], k_phi_phi, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(system.f_globals[:n_edges], rhs_edge, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(system.f_globals[n_edges:], rhs_node, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            system.m_edge_globals.toarray(), mass, rtol=1e-10, atol=1e-15
        )
        # J != eye must change the curl-curl block: the metric is really used.
        reference = _hand_tet_aa()
        self.assertGreater(np.max(np.abs(k_aa.real - nu * reference)), 1e-3)

    def test_reference_tet_matches_the_hand_curl_curl(self):
        mesh = _mesh(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0, 1, 2, 3]],
        )
        topology = build_topology(mesh)
        system = assemble(
            Formulation.MQS,
            mesh,
            topology,
            _materials(mesh, _curl_only()),
            0.0,
            None,
            TetrahedronWhitney(),
        )
        k_global = system.k_globals.toarray()
        self.assertEqual(k_global.shape, (10, 10))
        np.testing.assert_allclose(k_global[:6, :6], _hand_tet_aa(), atol=1e-12)
        np.testing.assert_allclose(k_global[:6, 6:], 0.0, atol=1e-12)
        np.testing.assert_allclose(k_global[6:, 6:], 0.0, atol=1e-12)
        np.testing.assert_allclose(system.f_globals, 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
