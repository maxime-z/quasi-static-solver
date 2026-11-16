"""Discrete de Rham identity on a cube split into six tetrahedra.

The cube [0, l1] x [0, l2] x [0, l3] has the eight corners of a 2x2x2 grid,
flattened with the x index fastest. Six tets fill it (volume l1*l2*l3) and
share 19 edges.

With sigma = 0 the edge block is pure curl-curl, so every column of the
incidence matrix G is a kernel vector:

    K_AA @ G = 0,          nullity(K_AA) = n_nodes - 1
    K_phiphi @ 1 = 0,      nullity(K_phiphi) = 1

on one connected mesh. A sign error between a tet Whitney curl and G fails
the first product even when a direct residual looks small.
"""

import unittest
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from darwin.assemble import assemble
from darwin.elements import TetrahedronWhitney
from darwin.formulation import Formulation
from darwin.material import EPSILON_0, MU_0, Material
from darwin.mesh_io import Mesh
from darwin.topology import build_topology

TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))

CUBE_CELLS = np.array(
    [
        [0, 1, 3, 4],
        [4, 5, 7, 1],
        [1, 3, 4, 7],
        [0, 2, 3, 4],
        [4, 7, 6, 2],
        [2, 4, 3, 7],
    ]
)


def basic_mesh(l1: float, l2: float, l3: float) -> Mesh:
    """Cube corner grid, x index fastest, split into the six tets above."""
    grid = np.meshgrid([0.0, l1], [0.0, l2], [0.0, l3], indexing="ij")
    points = np.stack(grid, axis=-1).reshape(-1, 3, order="F")
    region_ids = np.zeros(len(CUBE_CELLS), dtype=int)
    return Mesh(points, CUBE_CELLS.copy(), region_ids, {0: "air"})


def _tet_volume(points: np.ndarray, cell: np.ndarray) -> float:
    a, b, c, d = points[cell]
    return float(np.dot(b - a, np.cross(c - a, d - a)) / 6.0)


def plot_tetrahedra_mesh(
    mesh: Mesh, path: Path | None = None, show: bool = True
) -> None:
    """Plot a 3D tetrahedral mesh: wireframe edges, translucent faces, node labels."""
    points = np.asarray(mesh.points, dtype=float)
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    face_polys = []
    for cell_id, cell in enumerate(mesh.cells):
        color = f"C{cell_id % 10}"
        for a, b in combinations(cell, 2):
            ax.plot(*points[[a, b]].T, color=color, linewidth=1.0, alpha=0.85)
        for face in TET_FACES:
            face_polys.append(points[list(cell[list(face)])])
        centroid = points[cell].mean(axis=0)
        ax.text(*centroid, str(cell_id), fontsize=8, ha="center", color=color)

    ax.add_collection3d(
        Poly3DCollection(
            face_polys,
            alpha=0.12,
            facecolor="C0",
            edgecolor="0.35",
            linewidths=0.4,
        )
    )

    for nid, xyz in enumerate(points):
        ax.scatter(*xyz, color="black", s=22, zorder=5)
        ax.text(xyz[0] + 0.03, xyz[1] + 0.03, xyz[2] + 0.03, str(nid), fontsize=8)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Tetrahedra mesh ({len(mesh.cells)} cells, {len(points)} nodes)")
    _set_equal_3d(ax, points)
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=140)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _set_equal_3d(ax, points: np.ndarray) -> None:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.5 * float(np.max(hi - lo))
    if radius == 0.0:
        radius = 0.5
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


class TestTetDeRham(unittest.TestCase):
    def test_six_tet_cube_kernel(self):
        side = 2.0
        mesh = basic_mesh(side, side, side)
        topology = build_topology(mesh)

        n_edges, n_nodes = topology.incidence_matrix.shape
        self.assertEqual(n_nodes, 8)
        self.assertEqual(n_edges, 19)
        self.assertEqual(mesh.cells.shape, (6, 4))
        volumes = [_tet_volume(mesh.points, cell) for cell in mesh.cells]
        self.assertTrue(all(abs(v) > 0.0 for v in volumes))
        self.assertAlmostEqual(sum(abs(v) for v in volumes), side**3)

        # (G q)_e = q[head] - q[tail] on every oriented edge n1 < n2.
        q = np.arange(n_nodes, dtype=float)
        gq = topology.incidence_matrix @ q
        for e, (n1, n2) in enumerate(topology.edges):
            self.assertEqual(gq[e], q[int(n2)] - q[int(n1)])
        self.assertEqual(int(np.linalg.matrix_rank(topology.incidence_matrix)), n_nodes - 1)

        # nu = 1 and eps = 1. sigma = 0 leaves K_AA as curl-curl only.
        material = Material(
            name="unit", sigma=0.0, mu_r=1.0 / MU_0, eps_r=1.0 / EPSILON_0
        )
        system = assemble(
            Formulation.DARWIN,
            mesh,
            topology,
            {0: material},
            frequency=1.0 / (2.0 * np.pi),
            source=None,
            element=TetrahedronWhitney(),
        )

        k = system.k_globals.toarray()
        k_aa = k[:n_edges, :n_edges]
        k_phiphi = k[n_edges:, n_edges:]
        self.assertEqual(k_aa.shape, (n_edges, n_edges))
        self.assertEqual(k_phiphi.shape, (n_nodes, n_nodes))

        np.testing.assert_allclose(
            k_aa @ topology.incidence_matrix, 0.0, atol=1e-9
        )
        np.testing.assert_allclose(k_phiphi @ np.ones(n_nodes), 0.0, atol=1e-12)
        self.assertEqual(int(np.linalg.matrix_rank(k_aa)), n_edges - (n_nodes - 1))
        self.assertEqual(int(np.linalg.matrix_rank(k_phiphi)), n_nodes - 1)


if __name__ == "__main__":
    unittest.main()
