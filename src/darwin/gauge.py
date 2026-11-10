from collections import defaultdict, deque
from enum import Enum

import numpy as np

from darwin.assemble import System
from darwin.topology import Topology


class GaugeKind(Enum):
    """Gauge applied to the edge unknowns after the boundary conditions."""

    NONE = 0
    TREE = 1
    COULOMB = 2


def apply_gauge(system: System, topology: Topology, kind: GaugeKind = GaugeKind.NONE) -> System:
    """Return the system with the requested gauge written into K and the load.

    Tree–cotree replaces a spanning set of free edge rows by a_e = 0. It does
    not change nodal unknowns. Coulomb is not implemented.
    """
    if kind == GaugeKind.NONE:
        return system
    if kind == GaugeKind.TREE:
        return _apply_tree_cotree_gauge(system, topology)
    if kind == GaugeKind.COULOMB:
        raise NotImplementedError("Coulomb gauge is not implemented")
    raise ValueError(f"unknown gauge kind {kind!r}")


def _is_identity_row(K, i: int) -> bool:
    """True when row i is the statement that this one unknown equals the load."""
    row = K.getrow(i)
    return row.nnz == 1 and int(row.indices[0]) == i and np.isclose(row.data[0], 1.0)


def _fixed_edge_components(n_nodes: int, fixed_ends: list[tuple[int, int]]) -> list[set[int]]:
    """Nodes joined by edges that are already a_e = 0, plus every other node alone."""
    parent = list(range(n_nodes))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for n1, n2 in fixed_ends:
        r1, r2 = find(n1), find(n2)
        if r1 != r2:
            parent[r2] = r1

    groups: dict[int, set[int]] = defaultdict(set)
    for node in range(n_nodes):
        groups[find(node)].add(node)
    return list(groups.values())


def _apply_tree_cotree_gauge(system: System, topology: Topology) -> System:
    """Set one spanning tree of the free edges to zero on the full system.

    Identity edge rows are edges already eliminated by a boundary condition.
    Their connected components share one gauge value. The search starts from
    the component that contains the lowest such node, or from node 0 when no
    edge is fixed. Reaching any node of another component marks that whole
    component, because the fixed edges already tie it together. Nodal rows are
    not roots: a prescribed phi does not fix the edge gradient.
    """
    k_gauge = system.k_globals.tolil(copy=True)
    f_gauge = np.array(system.f_globals, dtype=complex, copy=True)
    n_edges, n_nodes = topology.incidence_matrix.shape

    neighbors: dict[int, list[tuple[int, int]]] = defaultdict(list)
    fixed_ends: list[tuple[int, int]] = []
    for e in range(n_edges):
        n1, n2 = (int(n) for n in topology.edges[e])
        if _is_identity_row(k_gauge, e):
            fixed_ends.append((n1, n2))
        else:
            neighbors[n1].append((n2, e))
            neighbors[n2].append((n1, e))

    components = _fixed_edge_components(n_nodes, fixed_ends)
    component_of = [0] * n_nodes
    for i_comp, nodes in enumerate(components):
        for node in nodes:
            component_of[node] = i_comp

    if fixed_ends:
        root_node = min(node for pair in fixed_ends for node in pair)
    else:
        root_node = 0
    root = component_of[root_node]

    visited = set(components[root])
    seen_component = {root}
    queue = deque(visited)
    tree_edges: list[int] = []
    while queue:
        u = queue.popleft()
        for v, e in neighbors[u]:
            comp = component_of[v]
            if comp in seen_component:
                continue
            seen_component.add(comp)
            for node in components[comp]:
                visited.add(node)
                queue.append(node)
            tree_edges.append(e)

    if len(visited) != n_nodes:
        raise RuntimeError(
            f"{n_nodes - len(visited)} nodes are not reachable from the root nodes"
        )

    for e in tree_edges:
        k_gauge[e, :] = 0.0
        k_gauge[:, e] = 0.0
        k_gauge[e, e] = 1.0
        f_gauge[e] = 0.0

    return System(
        k_gauge.tocsr(),
        f_gauge,
        system.edge_dofs.copy(),
        system.node_dofs.copy(),
    )
