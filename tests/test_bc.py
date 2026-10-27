"""Dirichlet walls and floating conductors on assembled Darwin systems."""

import unittest

import numpy as np

from darwin.assemble import System, assemble
from darwin.bc import DirichletWall, FloatingConductor, apply_dirichlet, apply_floating
from darwin.elements import TetrahedronWhitney, TriangleWhitney
from darwin.material import EPSILON_0, MU_0, Material, materials_by_region
from darwin.mesh_io import Mesh
from darwin.topology import build_topology, tag_boundary


def square_mesh() -> Mesh:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = np.array([[0, 1, 3], [1, 2, 3]])
    return Mesh(points, cells, np.array([0, 0]), region_names={0: "air"})


def cube_mesh() -> Mesh:
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
        [
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 1, 5, 6],
            [0, 4, 5, 6],
            [0, 3, 7, 6],
            [0, 4, 7, 6],
        ]
    )
    return Mesh(points, cells, np.zeros(6, dtype=int), region_names={0: "air"})


def square_predicates():
    return {
        "bottom": lambda x: np.isclose(x[1], 0.0),
        "top": lambda x: np.isclose(x[1], 1.0),
        "left": lambda x: np.isclose(x[0], 0.0),
        "right": lambda x: np.isclose(x[0], 1.0),
    }


def cube_predicates():
    return {
        "xmin": lambda x: np.isclose(x[0], 0.0),
        "xmax": lambda x: np.isclose(x[0], 1.0),
        "ymin": lambda x: np.isclose(x[1], 0.0),
        "ymax": lambda x: np.isclose(x[1], 1.0),
        "zmin": lambda x: np.isclose(x[2], 0.0),
        "zmax": lambda x: np.isclose(x[2], 1.0),
    }


def _unit_material() -> Material:
    return Material(name="air", sigma=0.0, mu_r=1.0 / MU_0, eps_r=1.0 / EPSILON_0)


def _assemble_square() -> tuple[Mesh, object, System]:
    mesh = square_mesh()
    facets, tags = tag_boundary(mesh, square_predicates())
    topology = build_topology(mesh, boundary_tags=(facets, tags))
    materials = materials_by_region(mesh, {_unit_material().name: _unit_material()})
    system = assemble(mesh, topology, materials, 1.0, None, TriangleWhitney())
    return mesh, topology, system


def _assemble_cube() -> tuple[Mesh, object, System]:
    mesh = cube_mesh()
    facets, tags = tag_boundary(mesh, cube_predicates())
    topology = build_topology(mesh, boundary_tags=(facets, tags))
    materials = materials_by_region(mesh, {_unit_material().name: _unit_material()})
    system = assemble(mesh, topology, materials, 1.0, None, TetrahedronWhitney())
    return mesh, topology, system


def _identity_row(K: np.ndarray, i: int, value: complex = 1.0) -> None:
    np.testing.assert_allclose(K[i, i], value, atol=1e-12)
    row = np.array(K[i, :], copy=True)
    row[i] = 0.0
    np.testing.assert_allclose(row, 0.0, atol=1e-12)
    col = np.array(K[:, i], copy=True)
    col[i] = 0.0
    np.testing.assert_allclose(col, 0.0, atol=1e-12)


class TestDirichlet(unittest.TestCase):
    def test_pec_on_left_side_of_the_square(self):
        _, topology, system = _assemble_square()
        k0 = system.k_globals.toarray()
        rhs0 = system.f_globals.copy()
        n_edges = len(topology.edges)

        constrained = apply_dirichlet(
            system, topology, [DirichletWall("left", 0.0)]
        )
        k1 = constrained.k_globals.toarray()
        rhs1 = constrained.f_globals

        left_nodes = [int(n) for n in topology.boundary["left"].nodes]
        left_edges = [
            eid
            for eid, (a, b) in enumerate(topology.edges)
            if int(a) in left_nodes and int(b) in left_nodes
        ]
        constrained_dofs = left_edges + [n_edges + n for n in left_nodes]
        free = [i for i in range(k0.shape[0]) if i not in constrained_dofs]

        for dof in constrained_dofs:
            _identity_row(k1, dof)
            self.assertEqual(rhs1[dof], 0.0)

        np.testing.assert_allclose(k1[np.ix_(free, free)], k0[np.ix_(free, free)], atol=1e-12)
        np.testing.assert_allclose(rhs1[free], rhs0[free], atol=1e-12)

    def test_electrode_lifts_a_nonzero_voltage(self):
        _, topology, system = _assemble_square()
        k0 = system.k_globals.toarray()
        rhs0 = system.f_globals.copy()
        n_edges = len(topology.edges)
        voltage = 2.5

        constrained = apply_dirichlet(
            system, topology, [DirichletWall("right", voltage)]
        )
        k1 = constrained.k_globals.toarray()
        rhs1 = constrained.f_globals

        right_nodes = [int(n) for n in topology.boundary["right"].nodes]
        right_edges = [
            eid
            for eid, (a, b) in enumerate(topology.edges)
            if int(a) in right_nodes and int(b) in right_nodes
        ]
        node_dofs = [n_edges + n for n in right_nodes]
        constrained_dofs = right_edges + node_dofs
        free = [i for i in range(k0.shape[0]) if i not in constrained_dofs]

        expected_rhs = rhs0.copy()
        for dof in node_dofs:
            expected_rhs -= k0[:, dof] * voltage
        for dof in constrained_dofs:
            expected_rhs[dof] = 0.0
        for dof in node_dofs:
            expected_rhs[dof] = voltage

        for dof in right_edges:
            _identity_row(k1, dof)
            self.assertEqual(rhs1[dof], 0.0)
        for dof in node_dofs:
            _identity_row(k1, dof)
            np.testing.assert_allclose(rhs1[dof], voltage, atol=1e-12)

        np.testing.assert_allclose(k1[np.ix_(free, free)], k0[np.ix_(free, free)], atol=1e-12)
        np.testing.assert_allclose(rhs1[free], expected_rhs[free], atol=1e-12)

    def test_shared_cube_outline_edge_is_eliminated(self):
        _, topology, system = _assemble_cube()
        # Edge (1, 2) lies on xmax and ymax; exclusive Boundary.edges drop it.
        shared = None
        for eid, (a, b) in enumerate(topology.edges):
            if {int(a), int(b)} == {1, 2}:
                shared = eid
                break
        self.assertIsNotNone(shared)
        self.assertNotIn(shared, topology.boundary["xmax"].edges)
        self.assertNotIn(shared, topology.boundary["ymax"].edges)

        constrained = apply_dirichlet(
            system,
            topology,
            [DirichletWall("xmax", 0.0), DirichletWall("ymax", 0.0)],
        )
        k1 = constrained.k_globals.toarray()
        _identity_row(k1, shared)
        self.assertEqual(constrained.f_globals[shared], 0.0)

    def test_voltage_port_keeps_inter_electrode_edges(self):
        _, topology, system = _assemble_square()
        k0 = system.k_globals.toarray()
        n_edges = len(topology.edges)
        # Diagonal (1, 3) joins bottom node 1 to top node 3.
        diagonal = next(
            eid
            for eid, (a, b) in enumerate(topology.edges)
            if {int(a), int(b)} == {1, 3}
        )
        electrode_nodes = [n_edges + n for n in (0, 1, 2, 3)]

        constrained = apply_dirichlet(
            system,
            topology,
            [DirichletWall("bottom", 0.0), DirichletWall("top", 2.0)],
        )
        k1 = constrained.k_globals.toarray()

        # The joining edge is not replaced by an identity constraint.
        self.assertFalse(
            np.isclose(k1[diagonal, diagonal], 1.0)
            and np.count_nonzero(np.abs(k1[diagonal]) > 1e-14) == 1
        )
        np.testing.assert_allclose(
            k1[diagonal, diagonal], k0[diagonal, diagonal], atol=1e-12
        )
        # Couplings into electrode node columns are cleared by the Dirichlet lift;
        # couplings to other free edge dofs stay.
        free_edges = [
            i
            for i in range(n_edges)
            if i != diagonal
            and not (
                np.isclose(k1[i, i], 1.0)
                and np.count_nonzero(np.abs(k1[i]) > 1e-14) == 1
            )
        ]
        np.testing.assert_allclose(
            k1[np.ix_([diagonal], free_edges)],
            k0[np.ix_([diagonal], free_edges)],
            atol=1e-12,
        )
        for dof in electrode_nodes:
            np.testing.assert_allclose(k1[diagonal, dof], 0.0, atol=1e-12)
            np.testing.assert_allclose(k1[dof, diagonal], 0.0, atol=1e-12)

    def test_conflicting_node_voltages_raise(self):
        _, topology, system = _assemble_square()
        with self.assertRaises(ValueError):
            apply_dirichlet(
                system,
                topology,
                [DirichletWall("left", 0.0), DirichletWall("bottom", 1.0)],
            )

    def test_unknown_tag_raises(self):
        _, topology, system = _assemble_square()
        with self.assertRaises(KeyError):
            apply_dirichlet(system, topology, [DirichletWall("missing", 0.0)])

    def test_empty_wall_list_returns_the_same_system(self):
        _, topology, system = _assemble_square()
        out = apply_dirichlet(system, topology, [])
        self.assertIs(out, system)


class TestFloating(unittest.TestCase):
    def test_left_side_is_tied_to_one_master(self):
        _, topology, system = _assemble_square()
        k0 = system.k_globals.toarray()
        rhs0 = system.f_globals.copy()
        n_edges = len(topology.edges)
        current = 3.0 + 1.0j

        left_nodes = sorted(int(n) for n in topology.boundary["left"].nodes)
        master_node, slave_node = left_nodes[0], left_nodes[1]
        master = n_edges + master_node
        slave = n_edges + slave_node
        edge = next(
            eid
            for eid, (a, b) in enumerate(topology.edges)
            if {int(a), int(b)} == set(left_nodes)
        )

        constrained = apply_floating(
            system, topology, [FloatingConductor("left", current)]
        )
        k1 = constrained.k_globals.toarray()
        rhs1 = constrained.f_globals

        _identity_row(k1, edge)
        self.assertEqual(rhs1[edge], 0.0)

        # Slave row is phi_s - phi_m = 0.
        np.testing.assert_allclose(k1[slave, slave], 1.0, atol=1e-12)
        np.testing.assert_allclose(k1[slave, master], -1.0, atol=1e-12)
        row = np.array(k1[slave, :], copy=True)
        row[slave] = 0.0
        row[master] = 0.0
        np.testing.assert_allclose(row, 0.0, atol=1e-12)
        self.assertEqual(rhs1[slave], 0.0)

        # Off-master free columns are the sum of the two original node rows.
        # The master diagonal is the condensed 2x2 block K_mm+K_ms+K_sm+K_ss.
        free = [i for i in range(k0.shape[0]) if i not in (edge, slave, master)]
        expected_master_row = k0[master, free] + k0[slave, free]
        np.testing.assert_allclose(k1[master, free], expected_master_row, atol=1e-12)
        expected_diag = (
            k0[master, master]
            + k0[master, slave]
            + k0[slave, master]
            + k0[slave, slave]
        )
        np.testing.assert_allclose(k1[master, master], expected_diag, atol=1e-12)
        np.testing.assert_allclose(
            rhs1[master], rhs0[master] + rhs0[slave] + current, atol=1e-12
        )

    def test_dirichlet_and_floating_on_the_same_node_raise(self):
        _, topology, system = _assemble_square()
        system = apply_dirichlet(system, topology, [DirichletWall("left", 0.0)])
        with self.assertRaises(ValueError):
            apply_floating(system, topology, [FloatingConductor("left", 0.0)])

    def test_two_floating_tags_sharing_a_node_raise(self):
        _, topology, system = _assemble_square()
        with self.assertRaises(ValueError):
            apply_floating(
                system,
                topology,
                [FloatingConductor("left", 0.0), FloatingConductor("bottom", 0.0)],
            )

    def test_opposite_dirichlet_and_floating_keep_joining_edges(self):
        _, topology, system = _assemble_square()
        k0 = system.k_globals.toarray()
        diagonal = next(
            eid
            for eid, (a, b) in enumerate(topology.edges)
            if {int(a), int(b)} == {1, 3}
        )

        constrained = apply_floating(
            apply_dirichlet(system, topology, [DirichletWall("bottom", 0.0)]),
            topology,
            [FloatingConductor("top", 1.0)],
        )
        k1 = constrained.k_globals.toarray()
        # Joining edge must remain a real stiffness row, not an identity.
        self.assertFalse(
            np.isclose(k1[diagonal, diagonal], 1.0)
            and np.count_nonzero(np.abs(k1[diagonal]) > 1e-14) == 1
        )
        np.testing.assert_allclose(
            k1[diagonal, diagonal], k0[diagonal, diagonal], atol=1e-12
        )

    def test_four_node_face_condenses_into_one_master(self):
        _, topology, system = _assemble_cube()
        k0 = system.k_globals.toarray()
        n_edges = len(topology.edges)
        nodes = sorted(int(n) for n in topology.boundary["zmax"].nodes)
        self.assertEqual(len(nodes), 4)
        master = n_edges + nodes[0]
        slaves = [n_edges + n for n in nodes[1:]]
        face_edges = [
            eid
            for eid, (a, b) in enumerate(topology.edges)
            if int(a) in nodes and int(b) in nodes
        ]

        constrained = apply_floating(
            system, topology, [FloatingConductor("zmax", 2.0 + 0.5j)]
        )
        k1 = constrained.k_globals.toarray()
        rhs1 = constrained.f_globals

        for edge in face_edges:
            _identity_row(k1, edge)
            self.assertEqual(rhs1[edge], 0.0)

        for slave in slaves:
            np.testing.assert_allclose(k1[slave, slave], 1.0, atol=1e-12)
            np.testing.assert_allclose(k1[slave, master], -1.0, atol=1e-12)
            self.assertEqual(rhs1[slave], 0.0)

        expected_diag = sum(k0[n_edges + a, n_edges + b] for a in nodes for b in nodes)
        np.testing.assert_allclose(k1[master, master], expected_diag, atol=1e-12)
        np.testing.assert_allclose(
            rhs1[master],
            sum(system.f_globals[n_edges + n] for n in nodes) + (2.0 + 0.5j),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
