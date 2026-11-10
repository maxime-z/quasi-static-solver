"""Boundary conditions on an assembled Darwin system.

Grounded PEC and a fixed-voltage electrode are the same Dirichlet operation:
eliminate tangential edge degrees of freedom and set phi on the wall nodes.
A magnetic wall (PMC) is a tag that is never passed in; the volume assembly
already omitted that boundary integrand. Surface current and impedance need
a facet integral and are not applied here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix

from darwin.assemble import System
from darwin.topology import Topology


@dataclass(frozen=True)
class DirichletWall:
    """Named wall with a prescribed nodal voltage.

    ``voltage`` is 0 for a grounded PEC wall and the source value for an
    electrode. Tangential edge circulations on the wall are always set to 0.
    """

    tag: str
    voltage: float


@dataclass(frozen=True)
class FloatingConductor:
    """Perfect conductor with one unknown voltage and an injected current.

    ``current`` may be complex so a later lumped stamp can use the same record.
    """

    tag: str
    current: complex = 0.0


def _require_tag(topology: Topology, tag: str) -> None:
    if tag not in topology.boundary:
        raise KeyError(f"boundary tag {tag!r} is not on the topology")


def _n_edges(system: System) -> int:
    return int(system.edge_dofs.shape[0])


def _node_voltages(
    topology: Topology, walls: list[DirichletWall]
) -> dict[int, float]:
    """Map each Dirichlet node id to its prescribed voltage."""
    voltages: dict[int, float] = {}
    for wall in walls:
        _require_tag(topology, wall.tag)
        for node in topology.boundary[wall.tag].nodes:
            node = int(node)
            if node in voltages and not np.isclose(voltages[node], wall.voltage):
                raise ValueError(
                    f"node {node} is claimed by Dirichlet walls at "
                    f"{voltages[node]} and {wall.voltage}"
                )
            voltages[node] = float(wall.voltage)
    return voltages


def _skin_edges_with_both_ends_in(
    topology: Topology, nodes: set[int]
) -> list[int]:
    """Boundary edges whose two endpoints both lie in ``nodes``.

    Only an edge that lies on the skin is tangential to a wall. Two nodes can
    belong to two different walls at the same potential; the edge between them
    then cuts through the interior and is not tangential to either wall.
    Zeroing its circulation would impose ``A.dl = 0`` on a line that is not on
    a conductor.
    """
    if len(nodes) < 2:
        return []
    on_skin = {int(e) for e in topology.skin.edges}
    edges: list[int] = []
    for eid, (n1, n2) in enumerate(topology.edges):
        if eid not in on_skin:
            continue
        if int(n1) in nodes and int(n2) in nodes:
            edges.append(eid)
    return edges


def _equipotential_tangential_edges(
    topology: Topology, voltages: dict[int, float]
) -> list[int]:
    """Tangential edges of each equipotential Dirichlet set.

    An edge is eliminated when it is on the skin, both endpoints carry the
    same prescribed voltage, and it therefore has ``n x A = 0``. An edge that
    joins two different walls at the same voltage is a chord through the
    interior: it is not on the skin and must stay free. Shared outline edges
    of a grounded box are on the skin and are still eliminated.
    """
    on_skin = {int(e) for e in topology.skin.edges}
    edges: list[int] = []
    for eid, (n1, n2) in enumerate(topology.edges):
        if eid not in on_skin:
            continue
        n1, n2 = int(n1), int(n2)
        if n1 not in voltages or n2 not in voltages:
            continue
        if np.isclose(voltages[n1], voltages[n2]):
            edges.append(eid)
    return edges


def _clear_row_and_column(K: lil_matrix, i: int) -> None:
    K[i, :] = 0
    K[:, i] = 0


def _set_identity_dof(K: lil_matrix, rhs: np.ndarray, i: int, value: complex) -> None:
    _clear_row_and_column(K, i)
    K[i, i] = 1.0
    rhs[i] = value


def _is_identity_row(K: lil_matrix, i: int) -> bool:
    """True when row ``i`` is already a Dirichlet identity constraint."""
    row = K.getrow(i).tocoo()
    if row.nnz == 0:
        return False
    if row.nnz > 1:
        return False
    return int(row.col[0]) == i and np.isclose(row.data[0], 1.0)


def apply_dirichlet(
    system: System, topology: Topology, walls: list[DirichletWall]
) -> System:
    """Eliminate tangential edges and set nodal voltages on Dirichlet walls.

    The matrix keeps its size. Each constrained degree of freedom becomes an
    identity row. Nonzero nodal voltages are lifted into the load before the
    corresponding columns are cleared.
    """
    if not walls:
        return system

    voltages = _node_voltages(topology, walls)
    tangential_edges = _equipotential_tangential_edges(topology, voltages)
    n_edges = _n_edges(system)

    K = system.k_globals.tolil(copy=True)
    rhs = np.array(system.f_globals, dtype=complex, copy=True)

    # Lift nonzero nodal values using the columns of the unmodified matrix.
    for node, voltage in voltages.items():
        if voltage == 0.0:
            continue
        dof = n_edges + node
        column = K.getcol(dof).toarray().ravel()
        rhs -= column * voltage

    # edge dof is always zero when applying dirichlet, so no need to 
    for edge in tangential_edges:
        _set_identity_dof(K, rhs, int(edge), 0.0)

    for node, voltage in voltages.items():
        _set_identity_dof(K, rhs, n_edges + node, complex(voltage))

    return System(
        K.tocsr(),
        rhs,
        system.edge_dofs.copy(),
        system.node_dofs.copy(),
    )


def apply_floating(
    system: System, topology: Topology, conductors: list[FloatingConductor]
) -> System:
    """Tie each floating surface to one master node and inject its current.

    Call after ``apply_dirichlet``. Tangential edges on each floating surface
    alone are set to 0; edges that join different conductors stay free. The
    master is the lowest node id on the tag. Slave rows enforce
    ``phi_s - phi_m = 0``.
    """
    if not conductors:
        return system

    n_edges = _n_edges(system)
    K = system.k_globals.tolil(copy=True)
    rhs = np.array(system.f_globals, dtype=complex, copy=True)

    dirichlet_nodes = {
        node
        for node in range(len(system.node_dofs))
        if _is_identity_row(K, n_edges + node)
    }

    floating_nodes_by_tag: dict[str, set[int]] = {}
    claimed: dict[int, str] = {}
    for conductor in conductors:
        _require_tag(topology, conductor.tag)
        nodes = {int(n) for n in topology.boundary[conductor.tag].nodes}
        if not nodes:
            raise ValueError(f"floating conductor {conductor.tag!r} has no nodes")
        overlap = nodes & dirichlet_nodes
        if overlap:
            raise ValueError(
                f"nodes {sorted(overlap)} are both Dirichlet and floating"
            )
        for node in nodes:
            if node in claimed:
                raise ValueError(
                    f"node {node} belongs to floating tags "
                    f"{claimed[node]!r} and {conductor.tag!r}"
                )
            claimed[node] = conductor.tag
        floating_nodes_by_tag[conductor.tag] = nodes

    for nodes in floating_nodes_by_tag.values():
        for edge in _skin_edges_with_both_ends_in(topology, nodes):
            if not _is_identity_row(K, edge):
                _set_identity_dof(K, rhs, int(edge), 0.0)

    for conductor in conductors:
        nodes = sorted(floating_nodes_by_tag[conductor.tag])
        master_node = nodes[0]
        master = n_edges + master_node
        for slave_node in nodes[1:]:
            slave = n_edges + slave_node
            slave_row = K.getrow(slave).tocsr()
            slave_col = K.getcol(slave).tocsr()
            k_ss = complex(K[slave, slave])

            # Add the original slave contributions into the master equation.
            K[master, :] = K.getrow(master).tocsr() + slave_row
            K[:, master] = K.getcol(master).tocsr() + slave_col
            K[master, master] = complex(K[master, master]) + k_ss
            rhs[master] += rhs[slave]

            _clear_row_and_column(K, slave)
            K[slave, slave] = 1.0
            K[slave, master] = -1.0
            rhs[slave] = 0.0

        rhs[master] += complex(conductor.current)

    return System(
        K.tocsr(),
        rhs,
        system.edge_dofs.copy(),
        system.node_dofs.copy(),
    )
