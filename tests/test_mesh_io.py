import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from darwin.material import Material, materials_by_region
from darwin.mesh_io import read_msh
from darwin.topology import build_topology

RESULTS = Path(__file__).parent / "results"

CUBE_MSH = """\
$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
7
2 1 "xmin"
2 2 "xmax"
2 3 "ymin"
2 4 "ymax"
2 5 "zmin"
2 6 "zmax"
3 7 "air"
$EndPhysicalNames
$Nodes
8
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
5 0 0 1
6 1 0 1
7 1 1 1
8 0 1 1
$EndNodes
$Elements
18
1 4 2 7 7 1 2 3 7
2 4 2 7 7 1 3 4 7
3 4 2 7 7 1 2 6 7
4 4 2 7 7 1 5 6 7
5 4 2 7 7 1 4 8 7
6 4 2 7 7 1 5 8 7
7 2 2 5 5 1 2 3
8 2 2 5 5 1 3 4
9 2 2 6 6 5 6 7
10 2 2 6 6 5 7 8
11 2 2 3 3 1 2 6
12 2 2 3 3 1 6 5
13 2 2 4 4 3 4 7
14 2 2 4 4 4 7 8
15 2 2 1 1 1 4 8
16 2 2 1 1 1 8 5
17 2 2 2 2 2 3 7
18 2 2 2 2 2 7 6
$EndElements
"""

SQUARE_41_MSH = """\
$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
2
1 20 "wall"
2 10 "body"
$EndPhysicalNames
$Entities
4 4 1 0
1 0 0 0 0
2 1 0 0 0
3 1 1 0 0
4 0 1 0 0
5 0 0 0 1 0 0 1 20 2 1 2
6 1 0 0 1 1 0 1 20 2 2 3
7 0 1 0 1 1 0 1 20 2 3 4
8 0 0 0 0 1 0 1 20 2 4 1
9 0 0 0 1 1 0 1 10 4 5 6 7 8
$EndEntities
$Nodes
1 4 1 4
2 9 0 4
1
2
3
4
0 0 0
1 0 0
1 1 0
0 1 0
$EndNodes
$Elements
5 6 1 6
2 9 2 2
1 1 2 3
2 1 3 4
1 5 1 1
3 1 2
1 6 1 1
4 2 3
1 7 1 1
5 3 4
1 8 1 1
6 4 1
$EndElements
"""


def _write(directory: str, name: str, text: str) -> Path:
    path = Path(directory) / name
    path.write_text(text)
    return path


class TestReadMsh(unittest.TestCase):
    def test_reads_gmsh_22_cube_with_physical_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            mesh = read_msh(_write(directory, "cube.msh", CUBE_MSH))

        self.assertEqual(mesh.points.shape, (8, 3))
        self.assertEqual(mesh.cells.shape, (6, 4))
        np.testing.assert_array_equal(mesh.region_ids, np.full(6, 7))
        self.assertEqual(mesh.region_names, {7: "air"})
        self.assertEqual(len(mesh.facet_tags), 12)
        self.assertEqual(
            Counter(mesh.facet_tags.values()),
            Counter({name: 2 for name in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}),
        )

        topology = build_topology(mesh)
        self.assertEqual(len(topology.edges), 19)
        self.assertEqual(len(topology.skin.edges), 18)
        self.assertEqual(set(topology.boundary), {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
        # Only the six face diagonals belong to a single tag; the twelve cube
        # edges are shared between two faces and are thus in no tag.
        tagged_edges = [int(e) for b in topology.boundary.values() for e in b.edges]
        self.assertEqual(len(tagged_edges), 6)
        self.assertEqual(len(set(tagged_edges)), 6)

    def test_reads_gmsh_41_square(self):
        with tempfile.TemporaryDirectory() as directory:
            mesh = read_msh(_write(directory, "square.msh", SQUARE_41_MSH))

        self.assertEqual(mesh.points.shape, (4, 2))
        self.assertEqual(mesh.cells.shape, (2, 3))
        np.testing.assert_array_equal(mesh.region_ids, np.full(2, 10))
        self.assertEqual(mesh.region_names, {10: "body"})
        self.assertEqual(set(mesh.facet_tags.values()), {"wall"})

        topology = build_topology(mesh)
        self.assertEqual(len(topology.edges), 5)
        self.assertEqual(len(topology.skin.edges), 4)
        self.assertEqual(set(topology.boundary), {"wall"})
        self.assertEqual(len(topology.boundary["wall"].edges), 4)

    @unittest.skipUnless(
        (RESULTS / "square.msh").exists(), "generated square.msh is not available"
    )
    def test_reads_the_generated_square(self):
        mesh = read_msh(RESULTS / "square.msh")

        self.assertEqual(mesh.points.shape, (514, 2))
        self.assertEqual(mesh.cells.shape, (946, 3))
        np.testing.assert_array_equal(mesh.region_ids, np.zeros(946, dtype=int))
        self.assertEqual(mesh.region_names, {0: "region0"})
        self.assertEqual(mesh.facet_tags, {})

        topology = build_topology(mesh)
        self.assertEqual(len(topology.edges), 1459)
        self.assertEqual(len(topology.skin.edges), 80)

    def test_physical_groups_index_the_materials(self):
        with tempfile.TemporaryDirectory() as directory:
            mesh = read_msh(_write(directory, "cube.msh", CUBE_MSH))
        library = {"air": Material("air", sigma=0.0, mu_r=1.0, eps_r=1.0)}

        materials = materials_by_region(mesh, library)

        self.assertEqual(set(materials), {7})
        self.assertEqual(materials[7].mu_r, 1.0)

    def test_rejects_unknown_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write(directory, "mesh.msh", "$MeshFormat\n4.0 0 8\n$EndMeshFormat\n")
            with self.assertRaises(NotImplementedError):
                read_msh(path)

    def test_rejects_file_without_mesh_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write(directory, "mesh.msh", "$Nodes\n1\n1 0 0 0\n$EndNodes\n")
            with self.assertRaises(ValueError):
                read_msh(path)


if __name__ == "__main__":
    unittest.main()
