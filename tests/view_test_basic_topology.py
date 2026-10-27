import numpy as np
from darwin.mesh_io import Mesh
from darwin.topology import build_topology


points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
cells = np.array([[0, 1, 3], [1, 2, 3]])
region_ids = np.array([0, 0])

mesh = Mesh(points, cells, region_ids)

# 
topology = build_topology(mesh)

for edge in topology.edges:
    print("edge: ", edge)
for cell_edge in topology.cell_edges:
    print("cell_edge: ", cell_edge)
print("incidence matrix:\n ", topology.incidence_matrix)







