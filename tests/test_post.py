"""Tests for the field postprocessing.

The fields are rebuilt from the edge degrees of freedom and the node
potentials. The check here uses a field that the lowest order element
represents exactly, so the sampled values must match to roundoff.
"""

import unittest

import numpy as np

from darwin.elements import GaussianQuadrature, TetrahedronWhitney, TriangleWhitney
from darwin.mesh_io import Mesh
from darwin.post import cell_fields, relative_l2
from darwin.topology import build_topology

FREQUENCY = 1.0e6
OMEGA = 2 * np.pi * FREQUENCY


def _vector_potential(points: np.ndarray) -> np.ndarray:
    """A = 0.5 * (-y, x, 0), a solid body rotation with B = e_z."""

    return 0.5 * np.stack(
        (-points[:, 1], points[:, 0], np.zeros(len(points))), axis=1
    )


def _edge_dofs(mesh: Mesh, topology) -> np.ndarray:
    """Circulation of A along every edge, in the global (n1, n2) order."""

    a = np.zeros(len(topology.edges), dtype=complex)
    for index, (node_1, node_2) in enumerate(topology.edges):
        p_1 = mesh.points[node_1]
        p_2 = mesh.points[node_2]
        a[index] = _vector_potential(0.5 * (p_1 + p_2)[None, :])[0] @ (p_2 - p_1)
    return a


def _reference_tet() -> Mesh:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    return Mesh(points, np.array([[0, 1, 2, 3]], dtype=np.int64), np.array([0]))


def _skewed_tet() -> Mesh:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, 0.0],
            [0.2, 1.1, 0.05],
            [0.1, 0.3, 1.3],
        ]
    )
    return Mesh(points, np.array([[0, 1, 2, 3]], dtype=np.int64), np.array([0]))


class TestCellFields(unittest.TestCase):
    def test_reference_tet_reproduces_a_linear_field(self):
        mesh = _reference_tet()
        topology = build_topology(mesh)
        x = np.concatenate(
            [_edge_dofs(mesh, topology), np.zeros(mesh.points.shape[0], dtype=complex)]
        )
        rule = GaussianQuadrature(3, 2)

        e_field, b_field = cell_fields(
            mesh,
            topology,
            TetrahedronWhitney(),
            x,
            FREQUENCY,
            rule.points,
            rule.weights,
        )

        points = e_field.point[0]
        expected_b = np.array([0.0, 0.0, 1.0])
        expected_e = -1j * OMEGA * _vector_potential(points)

        for sample in range(points.shape[0]):
            np.testing.assert_allclose(b_field.value[0, sample], expected_b, atol=1e-12)
            np.testing.assert_allclose(
                e_field.value[0, sample], expected_e[sample], atol=1e-6
            )
        np.testing.assert_allclose(e_field.measure[0], rule.weights, atol=1e-14)

    def test_skewed_tet_reproduces_a_linear_field(self):
        mesh = _skewed_tet()
        topology = build_topology(mesh)
        x = np.concatenate(
            [_edge_dofs(mesh, topology), np.zeros(mesh.points.shape[0], dtype=complex)]
        )
        rule = GaussianQuadrature(3, 3)

        e_field, b_field = cell_fields(
            mesh,
            topology,
            TetrahedronWhitney(),
            x,
            FREQUENCY,
            rule.points,
            rule.weights,
        )

        points = e_field.point[0]
        expected_b = np.array([0.0, 0.0, 1.0])
        expected_e = -1j * OMEGA * _vector_potential(points)

        for sample in range(points.shape[0]):
            np.testing.assert_allclose(b_field.value[0, sample], expected_b, atol=1e-12)
            np.testing.assert_allclose(
                e_field.value[0, sample], expected_e[sample], atol=1e-6
            )

    def test_triangle_reproduces_a_linear_field(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        mesh = Mesh(points, np.array([[0, 1, 2]], dtype=np.int64), np.array([0]))
        topology = build_topology(mesh)

        a = np.zeros(len(topology.edges), dtype=complex)
        for index, (node_1, node_2) in enumerate(topology.edges):
            p_1 = points[node_1]
            p_2 = points[node_2]
            a[index] = _vector_potential(0.5 * (p_1 + p_2)[None, :])[0][:2] @ (
                p_2 - p_1
            )

        x = np.concatenate([a, np.zeros(3, dtype=complex)])
        rule = GaussianQuadrature(2, 2)
        e_field, b_field = cell_fields(
            mesh, topology, TriangleWhitney(), x, FREQUENCY, rule.points, rule.weights
        )

        points = e_field.point[0]
        for sample in range(points.shape[0]):
            np.testing.assert_allclose(b_field.value[0, sample], 1.0, atol=1e-12)
            np.testing.assert_allclose(
                e_field.value[0, sample],
                (-1j * OMEGA * _vector_potential(points))[sample][:2],
                atol=1e-6,
            )

    def test_relative_l2_is_zero_for_an_exact_field(self):
        mesh = _reference_tet()
        topology = build_topology(mesh)
        x = np.concatenate(
            [_edge_dofs(mesh, topology), np.zeros(mesh.points.shape[0], dtype=complex)]
        )
        rule = GaussianQuadrature(3, 2)
        _, b_field = cell_fields(
            mesh,
            topology,
            TetrahedronWhitney(),
            x,
            FREQUENCY,
            rule.points,
            rule.weights,
        )

        exact = np.broadcast_to(b_field.value, b_field.value.shape)
        self.assertLess(relative_l2(b_field, exact), 1e-12)


if __name__ == "__main__":
    unittest.main()
