from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from darwin.elements import Element, GaussianQuadrature
from darwin.material import Material
from darwin.mesh_io import Mesh
from darwin.topology import Topology



@dataclass
class System:
    k_globals: csr_matrix
    f_globals: np.ndarray
    edge_dofs: np.ndarray
    node_dofs: np.ndarray
    p_dofs: np.ndarray | None = None


def _scatter(rows, cols, vals, row_dofs, col_dofs, block) -> None:
    """Append one local block at the given global degree-of-freedom ids."""
    ii, jj = np.meshgrid(np.asarray(row_dofs), np.asarray(col_dofs), indexing="ij")
    rows.append(ii.ravel())
    cols.append(jj.ravel())
    vals.append(np.asarray(block).ravel())


def assemble(
    mesh: Mesh,
    topology: Topology,
    materials: dict[int, Material],
    frequency: float,
    source: Callable | None,
    element: Element,
    quadrature_degree: int = 2,
) -> System:
    """Scatter Darwin local blocks into one complex CSR matrix and one load vector.

    Unknown order is ``[all edges | all nodes]``. ``materials`` is keyed by
    region id, as returned by ``materials_by_region``. The mass and coupling
    terms are quadratic, so the default quadrature degree is 2.
    """
    n_edges = topology.edges.shape[0]
    n_nodes = mesh.points.shape[0]
    n_dof = n_edges + n_nodes

    edge_dofs = np.arange(n_edges, dtype=int)
    node_dofs = np.arange(n_nodes, dtype=int) + n_edges

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    rhs = np.zeros(n_dof, dtype=complex)

    gauss_points = GaussianQuadrature(mesh.points.shape[1], degree=quadrature_degree)

    for id_cell, cell in enumerate(mesh.cells):
        cell_nodes = mesh.points[cell]
        local_of = {int(g): i for i, g in enumerate(cell)}
        edge_pair = [
            (local_of[int(n1)], local_of[int(n2)])
            for n1, n2 in topology.edges[topology.cell_edges[id_cell]]
        ]
        region = int(mesh.region_ids[id_cell])
        blocks = element.darwin_local_forms(
            cell_nodes,
            edge_pair,
            frequency,
            materials[region],
            gauss_points,
            source,
        )

        edge_dof = topology.cell_edges[id_cell]
        node_dof = mesh.cells[id_cell] + n_edges
        _scatter(rows, cols, vals, edge_dof, edge_dof, blocks.K_AA)
        _scatter(rows, cols, vals, edge_dof, node_dof, blocks.K_Aphi)
        _scatter(rows, cols, vals, node_dof, edge_dof, blocks.K_phiA)
        _scatter(rows, cols, vals, node_dof, node_dof, blocks.K_phiphi)
        rhs[edge_dof] += blocks.rhs_edge
        rhs[node_dof] += blocks.rhs_node

    if rows:
        row_index = np.concatenate(rows)
        col_index = np.concatenate(cols)
        data = np.concatenate(vals)
    else:
        row_index = np.array([], dtype=int)
        col_index = np.array([], dtype=int)
        data = np.array([], dtype=complex)

    k_matrix = coo_matrix(
        (data, (row_index, col_index)),
        shape=(n_dof, n_dof),
        dtype=complex,
    ).tocsr()
    return System(k_matrix, rhs, edge_dofs, node_dofs)
